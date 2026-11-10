"""
Frozen MS-vs-HC fMRI SW-GNN inference pipeline.

FINAL MODEL
-----------
HANDCRAFTED13 + dense signed Pearson FC + Signed Weighted GNN

Expected input for one subject:
    preprocessed Schaefer-200 ROI time series, shape (990, 200)
    [time, ROI]

This module does NOT perform model selection or holdout tuning.

Important:
- Input is preprocessed ROI time series, NOT scanner/NIfTI raw data.
- ROI order must match the Schaefer-200 order used during training.
- The pipeline exactly reproduces the historical frozen feature construction:
    1) ROI-wise temporal z-score in float64, population SD (ddof=0)
    2) TSFresh7 computed from float64 harmonized signals
    3) canonical float32 time-series save precision
    4) Pearson FC recomputed from float32 canonical signals
    5) FC-summary6 from the 199 off-diagonal FC values per ROI
    6) HANDCRAFTED13 = TSFresh7 + FC-summary6
    7) development-fitted feature scaling
    8) 3-seed SW-GNN probability ensemble
"""

from pathlib import Path
import json
import hashlib
import importlib.metadata

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch import nn

try:
    from tsfresh import feature_extraction
except ImportError as e:
    raise ImportError(
        "tsfresh is required for exact TSFresh7 reconstruction. "
        "Install it in the environment before using this module."
    ) from e


EXPECTED_TIMEPOINTS = 990
EXPECTED_ROIS = 200
EXPECTED_FEATURES = 13

TSFRESH7_NAMES = [
    "absolute_sum_of_changes",
    "benford_correlation",
    "c3_lag7",
    "cid_ce_normalized",
    "longest_strike_above_mean",
    "skewness",
    "quantile_q25",
]

FC_SUMMARY6_NAMES = [
    "mean_signed_fc",
    "mean_absolute_fc",
    "std_signed_fc",
    "mean_positive_fc",
    "mean_negative_magnitude",
    "fraction_negative_edges",
]

HANDCRAFTED13_NAMES = TSFRESH7_NAMES + FC_SUMMARY6_NAMES

FINAL_SEEDS = [20260808, 20260818, 20260828]
HIDDEN = 64
DROPOUT = 0.30
FINAL_THRESHOLD = 0.50
FREEZE_ID = "FINAL_SWGNN_MS_HC_V1_PREHOLDOUT"


def sha256_file(path, chunk_size=1024 * 1024):
    path = Path(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def validate_and_orient_subject_ts(subject_ts):
    """
    Validate a single subject and return [time, ROI] float64.

    Accepted exact shapes:
      (990, 200) -> unchanged
      (200, 990) -> transposed, with audit flag

    Any other shape fails.
    """
    x = np.asarray(subject_ts)

    if x.ndim != 2:
        raise ValueError(
            f"One subject must be 2D. Received shape {x.shape}."
        )

    auto_transposed = False

    if x.shape == (EXPECTED_TIMEPOINTS, EXPECTED_ROIS):
        pass
    elif x.shape == (EXPECTED_ROIS, EXPECTED_TIMEPOINTS):
        x = x.T
        auto_transposed = True
    else:
        raise ValueError(
            "Expected one subject as (990,200) [time,ROI] "
            f"or (200,990) [ROI,time]. Received {x.shape}."
        )

    x = x.astype(np.float64, copy=False)

    if not np.isfinite(x).all():
        bad = np.argwhere(~np.isfinite(x))
        raise ValueError(
            f"NaN/Inf values detected. First bad positions: {bad[:10].tolist()}"
        )

    temporal_sd = x.std(axis=0)
    if np.any(temporal_sd < 1e-12):
        bad_rois = np.flatnonzero(temporal_sd < 1e-12)
        raise ValueError(
            "Near-zero temporal variance in ROI(s): "
            + ", ".join(map(str, bad_rois[:20]))
        )

    return x, auto_transposed


def roiwise_temporal_zscore(subject_ts_time_roi):
    """
    Exact class-blind harmonization used in Notebook 09D:
      mean/std within subject and ROI, np.std default ddof=0.
    """
    x = np.asarray(subject_ts_time_roi, dtype=np.float64)

    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)

    if np.any(sd < 1e-12):
        raise ValueError("Near-zero variance ROI encountered during z-scoring.")

    z64 = (x - mu) / sd

    if not np.isfinite(z64).all():
        raise ValueError("Non-finite values after ROI-wise z-scoring.")

    return z64


def extract_tsfresh7_1d(x):
    """
    Exact frozen TSFresh7 order.

    Historical TSFresh9:
      absolute_sum_of_changes
      benford_correlation
      c3(lag=7)
      cid_ce(normalize=True)
      longest_strike_above_mean
      variance
      standard_deviation
      skewness
      quantile(q=.25)

    Variance and SD were removed after harmonization because they are
    essentially constant, leaving the seven values below.
    """
    x = np.asarray(x, dtype=np.float64)

    return np.asarray(
        [
            feature_extraction.feature_calculators.absolute_sum_of_changes(x),
            feature_extraction.feature_calculators.benford_correlation(x),
            feature_extraction.feature_calculators.c3(x, 7),
            feature_extraction.feature_calculators.cid_ce(x, True),
            feature_extraction.feature_calculators.longest_strike_above_mean(x),
            feature_extraction.feature_calculators.skewness(x),
            feature_extraction.feature_calculators.quantile(x, 0.25),
        ],
        dtype=np.float32,
    )


def extract_subject_tsfresh7(z64):
    """
    z64 shape: (990, 200)
    returns: (200, 7) float32
    """
    z64 = np.asarray(z64, dtype=np.float64)

    if z64.shape != (EXPECTED_TIMEPOINTS, EXPECTED_ROIS):
        raise ValueError(f"Expected (990,200), got {z64.shape}.")

    out = np.empty((EXPECTED_ROIS, 7), dtype=np.float32)

    for roi in range(EXPECTED_ROIS):
        out[roi] = extract_tsfresh7_1d(z64[:, roi])

    if not np.isfinite(out).all():
        raise ValueError("Non-finite TSFresh7 values.")

    return out


def compute_canonical_fc_from_z64(z64):
    """
    Reproduce the canonical FC precision path used by the frozen project.

    Notebook 09D saved harmonized time series as float32.
    Notebook 09E loaded those float32 values, cast them to float64,
    and computed Pearson correlation.

    returns:
      fc: (200,200) float32 with diagonal = 1
      adjacency: (200,200) float32 with diagonal = 0
    """
    z32 = np.asarray(z64, dtype=np.float64).astype(np.float32)

    fc64 = np.corrcoef(
        z32.astype(np.float64),
        rowvar=False,
    )

    if not np.isfinite(fc64).all():
        raise ValueError("Non-finite Pearson FC values.")

    fc = fc64.astype(np.float32)

    if np.max(np.abs(fc - fc.T)) > 1e-5:
        raise ValueError("FC matrix is not symmetric within tolerance.")

    adjacency = fc.copy()
    np.fill_diagonal(adjacency, 0.0)

    return fc, adjacency, z32


def compute_fc_summary6(fc):
    """
    Exact FC-summary6 construction from Notebook 10B.

    For each ROI, summarize the 199 off-diagonal signed FC values.
    """
    fc = np.asarray(fc, dtype=np.float32)

    if fc.shape != (EXPECTED_ROIS, EXPECTED_ROIS):
        raise ValueError(f"Expected FC shape (200,200), got {fc.shape}.")

    out = np.empty((EXPECTED_ROIS, 6), dtype=np.float32)

    mat = fc.astype(np.float64)

    for i in range(EXPECTED_ROIS):
        vals = np.delete(mat[i], i)

        pos = vals[vals > 0]
        neg = vals[vals < 0]

        out[i, 0] = vals.mean()
        out[i, 1] = np.abs(vals).mean()
        out[i, 2] = vals.std()
        out[i, 3] = pos.mean() if len(pos) else 0.0
        out[i, 4] = (-neg).mean() if len(neg) else 0.0
        out[i, 5] = np.mean(vals < 0)

    if not np.isfinite(out).all():
        raise ValueError("Non-finite FC-summary6 values.")

    return out


def build_handcrafted13(tsfresh7, fc_summary6):
    tsfresh7 = np.asarray(tsfresh7, dtype=np.float32)
    fc_summary6 = np.asarray(fc_summary6, dtype=np.float32)

    if tsfresh7.shape != (200, 7):
        raise ValueError(f"Expected TSFresh7 (200,7), got {tsfresh7.shape}.")

    if fc_summary6.shape != (200, 6):
        raise ValueError(f"Expected FC-summary6 (200,6), got {fc_summary6.shape}.")

    h13 = np.concatenate([tsfresh7, fc_summary6], axis=1).astype(np.float32)

    if h13.shape != (200, 13):
        raise RuntimeError("HANDCRAFTED13 shape construction failed.")

    return h13


class SignedWeightedLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim, bias=True)
        self.pos_linear = nn.Linear(in_dim, out_dim, bias=False)
        self.neg_linear = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, H, A):
        A_pos = torch.clamp(A, min=0.0)
        A_neg = torch.clamp(-A, min=0.0)

        pos_den = A_pos.sum(dim=-1, keepdim=True)
        neg_den = A_neg.sum(dim=-1, keepdim=True)

        A_pos_n = torch.where(
            pos_den > 0,
            A_pos / pos_den.clamp_min(1e-8),
            torch.zeros_like(A_pos),
        )

        A_neg_n = torch.where(
            neg_den > 0,
            A_neg / neg_den.clamp_min(1e-8),
            torch.zeros_like(A_neg),
        )

        return self.norm(
            self.self_linear(H)
            + torch.bmm(A_pos_n, self.pos_linear(H))
            + torch.bmm(A_neg_n, self.neg_linear(H))
        )


class FinalFrozenSWGNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.g1 = SignedWeightedLayer(13, HIDDEN)
        self.g2 = SignedWeightedLayer(HIDDEN, HIDDEN)

        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN, 32),
            nn.ELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 2),
        )

    def forward(self, x, adjacency):
        H = F.elu(self.g1(x, adjacency))
        H = F.dropout(H, p=DROPOUT, training=self.training)

        residual = H
        H = F.elu(self.g2(H, adjacency) + residual)

        H = H.mean(dim=1)

        return self.classifier(H)


class FrozenMSPipeline:
    """
    Loads the final frozen scaler and 3 final model checkpoints.
    """

    def __init__(self, project_root, device=None):
        self.root = Path(project_root)

        self.eval_dir = self.root / "results" / "final_holdout_evaluation_v1"
        self.model_dir = self.eval_dir / "final_models"
        self.scaler_path = self.eval_dir / "final_development_feature_scaler.npz"
        self.freeze_path = self.eval_dir / "PREHOLDOUT_FINAL_MODEL_FREEZE.json"
        self.atlas_manifest_path = (
            self.root / "data" / "atlas" / "schaefer200_7network_roi_manifest.csv"
        )

        required = [
            self.scaler_path,
            self.freeze_path,
            *[
                self.model_dir / f"FINAL_SWGNN_seed{seed}.pt"
                for seed in FINAL_SEEDS
            ],
        ]

        for p in required:
            if not p.exists():
                raise FileNotFoundError(p)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        scaler = np.load(self.scaler_path, allow_pickle=True)
        self.feature_mu = np.asarray(scaler["mean"], dtype=np.float32)
        self.feature_sd = np.asarray(scaler["std"], dtype=np.float32)

        if self.feature_mu.shape != (13,) or self.feature_sd.shape != (13,):
            raise ValueError("Frozen scaler must have 13 features.")

        with open(self.freeze_path, "r") as f:
            self.freeze_spec = json.load(f)

        if self.freeze_spec.get("freeze_id") != FREEZE_ID:
            raise ValueError(
                f"Unexpected freeze ID: {self.freeze_spec.get('freeze_id')}"
            )

        self.models = {}

        for seed in FINAL_SEEDS:
            ckpt_path = self.model_dir / f"FINAL_SWGNN_seed{seed}.pt"
            ckpt = torch.load(
                ckpt_path,
                map_location=self.device,
                weights_only=False,
            )

            if ckpt.get("freeze_id") != FREEZE_ID:
                raise ValueError(f"Checkpoint {seed} freeze ID mismatch.")

            if int(ckpt.get("seed")) != seed:
                raise ValueError(f"Checkpoint {seed} seed mismatch.")

            model = FinalFrozenSWGNN().to(self.device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            self.models[seed] = model

        if self.atlas_manifest_path.exists():
            self.atlas_manifest = pd.read_csv(self.atlas_manifest_path)
        else:
            self.atlas_manifest = None

    def prepare_subject(self, subject_ts):
        x, auto_transposed = validate_and_orient_subject_ts(subject_ts)

        z64 = roiwise_temporal_zscore(x)
        tsfresh7 = extract_subject_tsfresh7(z64)
        fc, adjacency, z32 = compute_canonical_fc_from_z64(z64)
        fc6 = compute_fc_summary6(fc)
        h13 = build_handcrafted13(tsfresh7, fc6)

        h13_scaled = (
            (h13 - self.feature_mu.reshape(1, -1))
            / self.feature_sd.reshape(1, -1)
        ).astype(np.float32)

        return {
            "input_shape_received": tuple(np.asarray(subject_ts).shape),
            "auto_transposed": bool(auto_transposed),
            "canonical_zscore_float64": z64,
            "canonical_zscore_float32": z32,
            "tsfresh7": tsfresh7,
            "fc": fc,
            "adjacency": adjacency,
            "fc_summary6": fc6,
            "handcrafted13": h13,
            "handcrafted13_scaled": h13_scaled,
        }

    def predict_prepared(self, prepared):
        x = torch.as_tensor(
            prepared["handcrafted13_scaled"][None, :, :],
            dtype=torch.float32,
            device=self.device,
        )

        A = torch.as_tensor(
            prepared["adjacency"][None, :, :],
            dtype=torch.float32,
            device=self.device,
        )

        seed_probs = {}

        for seed, model in self.models.items():
            with torch.no_grad():
                logits = model(x, A)
                p_ms = torch.softmax(logits, dim=1)[0, 1].item()

            seed_probs[seed] = float(p_ms)

        ensemble = float(np.mean(list(seed_probs.values())))
        pred = int(ensemble >= FINAL_THRESHOLD)

        return {
            "p_ms_by_seed": seed_probs,
            "p_ms_ensemble": ensemble,
            "predicted_class": pred,
            "predicted_label": "MS" if pred == 1 else "HC",
            "threshold": FINAL_THRESHOLD,
        }

    def predict_array(self, subject_ts):
        prepared = self.prepare_subject(subject_ts)
        result = self.predict_prepared(prepared)

        return {
            **result,
            "auto_transposed": prepared["auto_transposed"],
            "input_shape_received": prepared["input_shape_received"],
        }

    def pipeline_manifest(self):
        rows = []

        for p in [
            self.freeze_path,
            self.scaler_path,
            *[
                self.model_dir / f"FINAL_SWGNN_seed{seed}.pt"
                for seed in FINAL_SEEDS
            ],
        ]:
            rows.append(
                {
                    "path": str(p),
                    "sha256": sha256_file(p),
                    "size_bytes": int(p.stat().st_size),
                }
            )

        versions = {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
        }

        try:
            versions["tsfresh"] = importlib.metadata.version("tsfresh")
        except Exception:
            versions["tsfresh"] = "unknown"

        return {
            "freeze_id": FREEZE_ID,
            "final_seeds": FINAL_SEEDS,
            "threshold": FINAL_THRESHOLD,
            "feature_names": HANDCRAFTED13_NAMES,
            "software_versions": versions,
            "artifacts": rows,
        }


def load_subject_file(
    path,
    roi_manifest=None,
    confirm_unlabeled_roi_order=False,
):
    """
    Load one subject from .npy, .csv, or .parquet.

    .npy:
      Because column names are unavailable, the caller must explicitly
      confirm that ROI ordering matches the frozen Schaefer-200 order.

    .csv/.parquet:
      If all ROI names from roi_manifest are present as columns, they
      are reordered automatically to the frozen atlas order.
      Otherwise explicit order confirmation is required.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(path, allow_pickle=False)

        if not confirm_unlabeled_roi_order:
            raise ValueError(
                "Numpy input has no ROI column labels. Set "
                "confirm_unlabeled_roi_order=True only after confirming "
                "that the 200 ROI columns exactly match the frozen "
                "Schaefer-200 order."
            )

        return arr

    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in [".parquet", ".pq"]:
        df = pd.read_parquet(path)
    else:
        raise ValueError(
            "Supported input formats: .npy, .csv, .parquet"
        )

    if roi_manifest is not None and "roi_name" in roi_manifest.columns:
        roi_names = roi_manifest["roi_name"].astype(str).tolist()

        if set(roi_names).issubset(set(map(str, df.columns))):
            return df.loc[:, roi_names].to_numpy(dtype=np.float64)

    # Try a pure 200-column numeric table only with explicit confirmation.
    numeric = df.select_dtypes(include=[np.number])

    if numeric.shape[1] == EXPECTED_ROIS and confirm_unlabeled_roi_order:
        return numeric.to_numpy(dtype=np.float64)

    raise ValueError(
        "Could not verify ROI ordering from table columns. "
        "Use the exact Schaefer-200 ROI-name columns or explicitly "
        "confirm the frozen ROI order."
    )
