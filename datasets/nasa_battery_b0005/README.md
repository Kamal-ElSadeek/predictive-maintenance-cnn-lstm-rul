# NASA Battery B0005

## Best supported result

The strongest result is the parallel CNN-LSTM run:

- test RMSE: **0.047**
- test R2: **0.877**

The result is visually verified in [`archive/evidence/02_parallel_cnn_lstm_rmse_0_047.pdf`](archive/evidence/02_parallel_cnn_lstm_rmse_0_047.pdf).

The closest reconstruction of my source code is [`best_result/best_available_code_reconstruction.ipynb`](best_result/best_available_code_reconstruction.ipynb). It loads real B0005 data and implements sequential and parallel CNN-LSTM models, but it has no saved outputs and does not establish that it is byte-identical to the original 0.047 run.

## How to trace the evidence

1. Open the parallel-model PDF export and locate the final test evaluation reporting RMSE 0.047 and R2 0.877.
2. Open the best-code reconstruction and locate the parallel CNN-LSTM construction and evaluation sections.
3. Read [`MANIFEST.md`](MANIFEST.md) to compare the sequential, parallel, Hyperband, and GA evidence.

The result and the notebook are linked as the best **evidence/code pair** available in this archive; they are not represented as a clean reproduced run.

## Received result comparison

| Experiment | Test RMSE | R2 | Evidence |
|---|---:|---:|---|
| Sequential CNN-LSTM | 0.367 | -6.353 | PDF export |
| Parallel CNN-LSTM | **0.047** | **0.877** | PDF export; strongest result |
| Hyperband sequential optimization | 0.094 | 0.516 | PDF export |
| GA dense-model optimization | 0.403 | -7.878 | PDF export |

These files support the detailed results chapter. They do not support wording that assigns the approximately 0.09 result to GA; my evidence assigns it to Hyperband.

## Code and data limitations

- The best-code reconstruction uses a nonchronological split that puts some later cycles in training before earlier test cycles. A clean rerun must use a strict chronological split.
- The real-data GA notebook completes one generation with best validation RMSE 0.007467, then fails because `ga_generations/` does not exist. It has no final test metric and is archived as an incomplete experiment.
- The separate synthetic-data GA notebook is excluded because it is not NASA B0005 evidence.
- The uploaded `B0005.mat` copy is structurally unreadable and is not included.

See [`data/README.md`](data/README.md) and [`REPRODUCIBILITY_STATUS.md`](../../REPRODUCIBILITY_STATUS.md) before attempting a rerun.
