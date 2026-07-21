# Reproducibility Status

This repository combines a clearly labelled **historical research archive** with clean reproductions where they have been completed. C-MAPSS FD003 is reproduced; N-CMAPSS DS05 and the battery dataset remain archive-status.

## C-MAPSS FD003 — REPRODUCED

A clean end-to-end rerun of the parallel CNN-LSTM has been completed in a pinned environment (Python 3.8, TensorFlow 2.10.0, recorded in [`artifacts/cmapss_fd003/environment.json`](artifacts/cmapss_fd003/environment.json)):

- engine-level split: 80 training units, 20 validation units, official test set held out for one-time evaluation;
- sequence length 30, RUL cap 125, seed 42;
- single-model test result: **RMSE 14.680; MAE 11.042; R2 0.874** — matching the thesis-reported 14.75 within 0.07 RMSE;
- 5-seed ensemble (seeds 42, 7, 123, 2024, 31415), mean of predictions: **test RMSE 13.391; MAE 9.793; R2 0.895**;
- full metrics for train, exhaustive-window validation, test-like validation, and test are in [`artifacts/cmapss_fd003/parallel_reconstruction/metrics.json`](artifacts/cmapss_fd003/parallel_reconstruction/metrics.json) and [`ensemble_metrics.json`](artifacts/cmapss_fd003/parallel_reconstruction/ensemble_metrics.json).

Code entry point: [`datasets/cmapss_fd003/01_cmapss_fd003_parallel_cnn_lstm_reconstruction.ipynb`](datasets/cmapss_fd003/01_cmapss_fd003_parallel_cnn_lstm_reconstruction.ipynb).

Remaining historical gap: the original executed notebooks behind the thesis's 14.75 parallel and 14.98 GA figures are not preserved in this archive. The clean rerun above supersedes them as verifiable evidence.

## N-CMAPSS DS05

Current status:

- the best available executed notebook reports RMSE 6.7927, MAE 5.1498, and R2 0.9280 on test units 2, 4, and 8;
- the closing batch preserves the completed 30-trial tuner stage and incomplete continuation stages;
- no newly received notebook supersedes the 6.7927 result.

Clean-rerun blockers:

- `N-CMAPSS_DS05.h5` is not included;
- the tuner searches on `X_train, Y_train` while validating on a subset drawn from that same input;
- thesis GA 13.91 and Hyperband 13.47 executed evidence is missing.

Next action: search on `X_train_split, Y_train_split`, validate on `X_val, Y_val`, and preserve units 2, 4, and 8 for one-time final evaluation.

## NASA Battery B0005

Current status:

- PDF evidence supports the parallel-model result RMSE 0.047 and R2 0.877;
- the closest real-data code reconstruction is retained as the best-code entry point;
- the real-data GA run is archived as incomplete.

Clean-rerun blockers:

- the received `B0005.mat` copy is unreadable;
- the code reconstruction's split is nonchronological;
- the original executed notebook for the 0.047 result is missing.

Next action: obtain a valid authorized B0005 file, use chronological train/validation/test cycles, create output directories safely, and rerun the comparison in one pinned environment.

## Requirements for reproduced-result labels

A future result should be labelled reproduced only after the corresponding workflow:

- runs from a fresh documented environment without hidden notebook state;
- obtains data through documented authorized steps;
- fits preprocessing only on training observations;
- isolates validation and test observations correctly;
- records the split, random seeds, package versions, and model configuration;
- reports RMSE, MAE, R2, and any dataset-specific score consistently.
