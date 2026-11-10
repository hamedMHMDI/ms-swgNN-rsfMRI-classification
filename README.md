# Frozen SW-GNN pipeline for MS-vs-HC classification from resting-state fMRI

This repository is the reproducibility package accompanying:

**Mohammadi H, Karwowski W, Hashemi Moghanjoughi P. _Disentangling Regional Signal Dynamics and Signed Functional Connectivity in Graph Neural Network Classification of Multiple Sclerosis from Resting-State Functional Magnetic Resonance Imaging._**

## Scope

The repository reconstructs the frozen subject-level inference pathway used in the manuscript:

preprocessed Schaefer-200 ROI time series (990 × 200)
→ ROI-wise temporal z-scoring
→ 7 regional temporal descriptors (T7)
→ dense signed Pearson functional connectivity
→ 6 local connectivity summaries (FC6)
→ H13 = T7 + FC6
→ development-fitted feature scaling
→ 3 frozen Signed Weighted Graph Neural Network models
→ mean P(MS)
→ fixed threshold 0.50.

The code **does not** perform scanner/NIfTI preprocessing and **does not** contain participant-level neuroimaging or clinical data.

## Frozen model

Freeze ID: `FINAL_SWGNN_MS_HC_V1_PREHOLDOUT`

Prespecified seeds:

- `20260808`
- `20260818`
- `20260828`

Decision threshold: `0.50`

The final holdout has already been opened and must not be reused for model selection or tuning.

## Repository contents

- `frozen_ms_swgNN_pipeline.py` — standalone inference implementation.
- `FROZEN_PIPELINE_MANIFEST.json` — frozen feature names, versions, artifact hashes, and input contract.
- `frozen_artifacts/PREHOLDOUT_FINAL_MODEL_FREEZE.json` — immutable pre-holdout freeze record.
- `frozen_artifacts/final_development_feature_scaler.npz` — scaler fitted on all 159 development participants.
- `frozen_artifacts/FINAL_SWGNN_seed*.pt` — three frozen model checkpoints.
- `frozen_artifacts/schaefer200_7network_roi_manifest.csv` — ROI-order manifest.
- `tests/representation_reconstruction_self_test.csv` — representation reconstruction audit.
- `tests/model_output_reproduction_self_test.csv` — final probability reconstruction audit.
- `requirements.txt` and `environment.yml` — frozen computational environment.
- `SHA256SUMS.txt` — checksums for the public release candidate.

## Input contract

One subject must be supplied as preprocessed regional BOLD time series with shape `(990, 200)` arranged as `[time, ROI]`. A `(200, 990)` input may be auto-transposed by the frozen code. ROI ordering must exactly match the included Schaefer-200 manifest.

## Reproducibility result

The historical reconstruction audit reproduced the canonical standardized time series, T7 descriptors, Pearson FC, FC6 summaries, and H13 representation with maximum absolute difference `0.0` for six audited participants. The maximum absolute difference in reconstructed ensemble MS probability was approximately `5.22 × 10^-8`.

## Data availability

Participant-level fMRI and clinical data are **not included**. They remain subject to the original institutional, ethics, consent, and data-governance conditions associated with the Jagiellonian University collaboration.

## Model-weight release note

The release candidate contains the three frozen model checkpoints because they are required to reproduce the final inference pathway. Before making the repository public, the authors should confirm with the relevant data custodian/coauthors that public distribution of trained model weights is permitted under the source data-governance agreement. If weight release is not permitted, retain the source code, scaler specification, hashes, model architecture, and self-test metadata publicly and provide checkpoints only through an approved access mechanism.

## Citation

A DOI should be created from the exact tagged release through Zenodo (or another archival repository) before manuscript submission. Replace the manuscript code-availability placeholder with the permanent repository URL and DOI.

## Contact

Hamed Mohammadi  
University of Central Florida  
Hamed.Mohammadi@UCF.edu  
ORCID: 0000-0003-4787-4322
