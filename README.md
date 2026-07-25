# Predictive Maintenance with Hybrid CNN-LSTM Models

This repository preserves the code and result evidence available for Kamal M. El Sadeek's MSc work on remaining-useful-life and battery-capacity prediction. Files are organized by dataset. Each dataset has one clearly identified best-available notebook and a separate archive of the other experiments.

## Start with the best result for each dataset

| Dataset | Best result supported by the received files | Best-code entry point | Where the result is evidenced |
|---|---:|---|---|
| NASA C-MAPSS FD003 | RMSE 15.032; R2 0.868 | [`datasets/cmapss_fd003/best_result/best_available_executed_notebook.ipynb`](datasets/cmapss_fd003/best_result/best_available_executed_notebook.ipynb) | Executed output inside that notebook. The thesis's lower 14.75 result is reported separately but its executed notebook was not available on this PC. |
| NASA N-CMAPSS DS05 | RMSE 6.7927; MAE 5.1498; R2 0.9280 | [`datasets/n_cmapss_ds05/best_result/best_available_executed_notebook.ipynb`](datasets/n_cmapss_ds05/best_result/best_available_executed_notebook.ipynb) | Final evaluation cells inside that notebook, including per-unit results for test units 2, 4, and 8. |
| NASA Battery B0005 | RMSE 0.047; R2 0.877 | [`datasets/nasa_battery_b0005/best_result/best_available_code_reconstruction.ipynb`](datasets/nasa_battery_b0005/best_result/best_available_code_reconstruction.ipynb) | [`archive/evidence/02_parallel_cnn_lstm_rmse_0_047.pdf`](datasets/nasa_battery_b0005/archive/evidence/02_parallel_cnn_lstm_rmse_0_047.pdf). The notebook is the closest received code reconstruction and has no saved execution output. |

Follow each dataset's README before interpreting a metric. It explains exactly why the notebook was selected, what evidence supports it, and what methodological limitations remain.


## Repository layout

```text
datasets/
  <dataset>/
    README.md          selection and evidence guide
    MANIFEST.md        file-level experiment inventory
    best_result/       one best-available code entry point
    archive/
      notebooks/       alternative, incomplete, or superseded experiments
      evidence/        retained notebook-to-PDF result exports
    data/               acquisition or missing-data notes
```

## Evidence labels

- **Executed result:** the metric is present in saved notebook output.
- **Visually verified result:** the metric is present in a retained PDF export.
- **Reported result:** the thesis states the value, but the corresponding executed file was not received.
- **Reproduced result:** a clean notebook has been run end to end in a documented environment. None of the historical metrics in this archive are currently labelled reproduced.

## Important limitations

- Raw datasets and trained model artifacts are not included.
- Historical notebooks retain their original research state, including machine-specific paths, interrupted cells, and known split or validation issues documented in the READMEs.
- This is an evidence-preserving research archive. It does not silently rewrite a historical notebook and present the altered code as the source of an old metric.

See [`REPRODUCIBILITY_STATUS.md`](REPRODUCIBILITY_STATUS.md) for the work needed to produce clean reruns.
