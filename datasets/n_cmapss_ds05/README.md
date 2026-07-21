# NASA N-CMAPSS DS05

## Best available executed notebook

Open [`best_result/best_available_executed_notebook.ipynb`](best_result/best_available_executed_notebook.ipynb).

Its saved final evaluation on units 2, 4, and 8 reports:

| Metric | Value |
|---|---:|
| MSE | 46.1410 |
| RMSE | **6.7927** |
| MAE | 5.1498 |
| R2 | 0.9280 |
| NASA total score | 1,357,657.42 |

The notebook also reports per-unit RMSE values of 6.27, 8.15, and 6.63 for units 2, 4, and 8 respectively.

This is the lowest completed test RMSE across all my N-CMAPSS notebooks. The closing batch documents the 30-trial tuning and continuation lineage but does not contain a lower final test result.

## How to trace the evidence

1. Open the best-result notebook.
2. Locate the final evaluation cell headed by the overall regression metrics.
3. Confirm that the saved output reports RMSE 6.7927, MAE 5.1498, and R2 0.9280.
4. Compare it with [`MANIFEST.md`](MANIFEST.md), which inventories the baseline and optimization runs.
5. Use [`archive/evidence`](archive/evidence) for the thesis-era sequential and parallel baseline exports.

## Methodological caveat

The official test units remain outside final model training, but the tuner call uses `X_train, Y_train` while validating on `X_val, Y_val`, and `X_val` was drawn from `X_train`. The tuner therefore sees validation observations in its training input. The historical metric is preserved as evidence, but it must not be described as a clean reproduced benchmark.

A corrected rerun must search on `X_train_split, Y_train_split`, validate on `X_val, Y_val`, and evaluate units 2, 4, and 8 only once after model selection.

## Result progression

| Experiment | Recorded result | Evidence status |
|---|---:|---|
| Sequential CNN-LSTM baseline | RMSE 26.012; R2 -0.056 | PDF export |
| Parallel CNN-LSTM baseline | RMSE 13.791; R2 0.703 | PDF export |
| Thesis GA | RMSE 13.91 | Thesis-reported; executed file missing |
| Thesis Hyperband | RMSE 13.47 | Thesis-reported; executed file missing |
| Later Bayesian sequential/hybrid | derived RMSE 7.570 | Notebook/PDF output |
| Later Bayesian parallel | RMSE 8.365; R2 0.891 | Notebook output |
| Intermediate continuation | RMSE 7.437; R2 0.914 | Notebook output; test used for validation |
| Best available Bayesian parallel | **RMSE 6.7927; R2 0.9280** | Executed notebook; tuner overlap caveat |

The experiments expect `N-CMAPSS_DS05.h5`. Raw data is not included; see [`data/README.md`](data/README.md).
