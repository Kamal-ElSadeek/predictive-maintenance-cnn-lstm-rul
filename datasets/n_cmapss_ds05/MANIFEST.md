# N-CMAPSS DS05 Experiment Manifest

## Best-result entry point

| File | Recorded outcome | Status |
|---|---:|---|
| `best_result/best_available_executed_notebook.ipynb` | RMSE 6.7927; MAE 5.1498; R2 0.9280 | Strongest received test result; test units isolated, tuner train/validation overlap remains |

## Archived notebooks

| File | Experiment | Recorded outcome |
|---|---|---:|
| `archive/notebooks/01_bayesian_sequential_hybrid.ipynb` | Bayesian sequential/hybrid | MSE 57.303; MAE 5.392; derived RMSE 7.570 |
| `archive/notebooks/02_bayesian_parallel.ipynb` | Bayesian parallel | RMSE 8.365; R2 0.891; later plotting cell fails |
| `archive/notebooks/03_bayesian_parallel_resume_safe.ipynb` | Resume-safe revision | Preserves RMSE 8.365 output |
| `archive/notebooks/04_bayesian_parallel_intermediate_rmse_7_437_test_leak.ipynb` | Intermediate extended training | RMSE 7.437; R2 0.914; test used as validation |
| `archive/notebooks/06_auc_optimized_interrupted_rmse_14_234.ipynb` | AUC continuation | RMSE 14.234 after interrupted training |
| `archive/notebooks/07_retrain_continuation_test_validation_rmse_7_437.ipynb` | Resume-safe continuation | Preserves RMSE 7.437; training validates on test data |
| `archive/notebooks/08_one_trial_incomplete_precursor.ipynb` | Validation-isolation precursor | One tuning trial; no final test metric |
| `archive/notebooks/09_completed_30_trial_tuning_no_final_test.ipynb` | Completed Bayesian tuning stage | 30 trials; best validation loss 47.3481; no final test metric |

## Archived PDF evidence

| File | Experiment | Recorded outcome |
|---|---|---:|
| `archive/evidence/01_sequential_baseline_rmse_26_012.pdf` | Sequential CNN-LSTM baseline | RMSE 26.012; R2 -0.056 |
| `archive/evidence/02_ga_tuner_failed_reference.pdf` | GA-tuner attempt | Memory failure; subsequent prior-model evaluation remains RMSE 26.012 |
| `archive/evidence/03_parallel_baseline_rmse_13_791.pdf` | Parallel CNN-LSTM baseline | RMSE 13.791; R2 0.703 |
| `archive/evidence/04_bayesian_sequential_mse_57_303.pdf` | Later Bayesian sequential/hybrid | MSE 57.303; MAE 5.392 |

## Closing-batch deduplication

- `Parallel_CNN_LSTM_Bayesian_Optimization_resume_safe__5__-_Val-n_v2.ipynb` is byte-for-byte identical to the retained 30-trial tuning notebook.
- `Parallel_CNN_LSTM_Bayesian_Optimization_resume_safe (5) - Val-n.ipynb` preserves the same tuning output but is a shorter development state; the more complete representative is retained.
- `Parallel_CNN_LSTM_Bayesian_Optimization_RETRAIN_30_TRIALS.ipynb` and `...Val__1_.ipynb` are retained because they preserve distinct incomplete continuation stages.
