# Reproducibility Status

This repository is ready to publish as a clearly labelled **historical research archive**. It is not yet a clean reproduction package.

## Option C — coverage completion runs

The `option_c/` folders under `cmapss_fd003` and `nasa_battery_b0005` contain freshly executed GA/Hyperband notebooks that optimize the architecture previously evaluated only at baseline (C-MAPSS sequential; Battery parallel). They were run in a documented environment (Python 3.8, TensorFlow 2.6.0, DEAP 1.3.1, keras-tuner, seed 42) with saved output, and each carries its own README with results and pipeline. In both datasets optimization did not improve the baseline: C-MAPSS sequential 15.31 → 15.84 (GA) / 16.26 (Hyperband); Battery parallel 0.047 → 0.527 (GA) / 0.492 (Hyperband), the latter degraded by small-data overfitting. These are retained as optimization findings; they do not change the best result per architecture.

## C-MAPSS FD003

Current status:

- the current-PC inventory is complete;
- the best available executed notebook reports RMSE 15.032 and R2 0.868;
- the thesis's 14.75 parallel and 14.98 GA execution evidence is still missing.

Clean-rerun blockers:

- raw FD003 data is not included;
- final environment, model weights, and tuner artifacts are missing;
- the historical preprocessing and validation protocol needs explicit reconstruction.

Next action: retrieve the university-PC items in `datasets/cmapss_fd003/CURRENT_PC_CHECKPOINT.md`, then rerun the selected architectures with documented engine-level splits.

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
