# C-MAPSS FD003 Experiment Manifest

| File | Experiment | Recorded outcome | Role |
|---|---|---:|---|
| `01_cmapss_fd003_parallel_cnn_lstm_reconstruction.ipynb` | Clean parallel CNN-LSTM reproduction | Test RMSE 14.680 (single, seed 42); 13.391 (5-seed ensemble) | Reproduced result — primary entry point |
| `best_result/best_available_executed_notebook.ipynb` | Parallel/GA-submitted lineage | RMSE 15.032; R2 0.868 | Best available historical executed notebook |
| `archive/notebooks/01_lstm_baseline.ipynb` | LSTM baseline | RMSE 15.746 | Archived experiment |
| `archive/notebooks/02_ga_early_attempt.ipynb` | Early GA attempt | No valid optimized result | Archived failure |
| `archive/notebooks/03_ga_longer_memory.ipynb` | Longer-memory GA | best-individual RMSE 17.745 | Archived experiment |
| `archive/notebooks/04_parallel_initial.ipynb` | Initial parallel model | RMSE 17.493 | Archived experiment |
| `archive/notebooks/05_parallel_improved.ipynb` | Improved parallel model | RMSE 15.402 | Archived experiment |
| `archive/notebooks/06_parallel_tuning_variant.ipynb` | Parallel tuning variant | RMSE 16.176 | Archived experiment |
| `archive/notebooks/08_sequential_cnn_lstm_run_rmse_15_313.ipynb` | Sequential CNN-LSTM | RMSE 15.313; R2 0.863 | Archived experiment |
| `archive/evidence/01_parallel_run_rmse_17_493_export.pdf` | Export of initial parallel run | RMSE 17.493 | PDF evidence |
| `archive/evidence/02_parallel_run_rmse_15_402_export.pdf` | Export of improved parallel run | RMSE 15.402 | PDF evidence |
| `archive/evidence/03_ga_early_failed_export.pdf` | Export of early GA attempt | Failure, no valid optimized metric | PDF evidence |

Exact-duplicate and serialization-only uploads are consolidated rather than listed twice. For the thesis's 14.75 parallel and 14.98 GA figures, the clean reproduction listed above is the canonical, verifiable version and the recommended entry point.
