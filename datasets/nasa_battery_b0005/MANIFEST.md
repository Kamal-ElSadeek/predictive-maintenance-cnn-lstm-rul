# NASA Battery B0005 Experiment Manifest

## Best evidence/code pair

| File | Role | Outcome |
|---|---|---:|
| `best_result/best_available_code_reconstruction.ipynb` | My closest real-B0005 code reconstruction for sequential/parallel CNN-LSTM work | No saved execution output; split requires correction |
| `archive/evidence/02_parallel_cnn_lstm_rmse_0_047.pdf` | Evidence for the strongest result | RMSE 0.047; R2 0.877 |

## Archived evidence and experiments

| File | Experiment | Recorded outcome |
|---|---|---:|
| `archive/evidence/01_sequential_cnn_lstm_rmse_0_367.pdf` | Sequential CNN-LSTM | RMSE 0.367; R2 -6.353 |
| `archive/evidence/03_hyperband_sequential_rmse_0_094.pdf` | Hyperband sequential optimization | RMSE 0.094; R2 0.516 |
| `archive/evidence/04_ga_dense_rmse_0_403.pdf` | GA dense-model optimization | RMSE 0.403; R2 -7.878 |
| `archive/notebooks/02_ga_real_b0005_partial_run.ipynb` | Real-data GA experiment | Generation 1 validation RMSE 0.007467; save-directory failure; no test result |

## Exclusions

- Duplicate PDF exports were not retained twice.
- `Battery_Health_Genetic_Algorithm.ipynb` was excluded because it generates synthetic measurements rather than loading NASA B0005.
- The uploaded `B0005.mat` was excluded because its compressed MATLAB stream is unreadable. Its SHA-256 is `2b14bfdd9ab935622cdd8562f39248ebf2a8d8f3701a17be0eefc2f8e41bd3f5`.
