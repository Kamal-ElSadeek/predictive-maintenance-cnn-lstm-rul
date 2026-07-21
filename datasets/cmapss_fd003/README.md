# NASA C-MAPSS FD003

## Reproduced result

Open [`01_cmapss_fd003_parallel_cnn_lstm_reconstruction.ipynb`](01_cmapss_fd003_parallel_cnn_lstm_reconstruction.ipynb).

This is a clean end-to-end rerun of the parallel CNN-LSTM in a pinned environment
(Python 3.8, TensorFlow 2.10.0 — see [`../../artifacts/cmapss_fd003/environment.json`](../../artifacts/cmapss_fd003/environment.json)):

- single model (seed 42): test **RMSE 14.680; MAE 11.042; R2 0.874**
- 5-seed ensemble (mean of predictions): test **RMSE 13.391; MAE 9.793; R2 0.895**

Protocol: engine-level split with 80 training and 20 validation units, preprocessing fitted
on training observations only, sequence length 30, RUL cap 125, official test set reserved
for one-time final evaluation. Full metrics, unit lists, seeds, and prediction plots are in
[`../../artifacts/cmapss_fd003/parallel_reconstruction`](../../artifacts/cmapss_fd003/parallel_reconstruction).

The thesis reports a parallel-model RMSE of **14.75**; the reproduced single-model run matches
it within 0.07 RMSE, and the ensemble improves on it.

## Historical archive

The best historical executed notebook is
[`best_result/best_available_executed_notebook.ipynb`](best_result/best_available_executed_notebook.ipynb),
whose saved evaluation output reports test RMSE **15.032** and test R2 **0.868**. It was selected
because it has the lowest executed test RMSE among the historical FD003 notebooks preserved in
this archive. The other unique executed notebooks report RMSE values of 15.313, 15.402, 15.746,
16.176, 17.493, and 17.745, or contain a failed/incomplete GA attempt.

The original executed notebooks behind the thesis's 14.75 parallel and 14.98 GA figures are not
preserved in this archive; the clean reproduction above supersedes them as verifiable evidence.

## How to verify the selection

1. Open the reconstruction notebook and compare its final evaluation output with the recorded metrics JSON files.
2. Compare the historical notebooks with the experiment inventory in [`MANIFEST.md`](MANIFEST.md).
3. Use the retained PDF exports in [`archive/evidence`](archive/evidence) to confirm the 17.493 and 15.402 parallel runs and the failed GA attempt.

## Dataset and task

- task: turbofan remaining-useful-life regression
- subset: FD003
- training engines: 100
- test engines: 100
- operating conditions: one
- fault modes: two

Raw C-MAPSS files are not included. See [`data/README.md`](data/README.md).

## Notes on the historical notebooks

The archived notebooks retain their original research state, including machine-specific data
paths, an unavailable `val_accuracy` checkpoint monitor in some regression runs, and split or
validation issues documented above. The reconstruction notebook addresses these; the archive
preserves them unchanged for auditability.
