# NASA C-MAPSS FD003

## Best available code on this PC

Open [`best_result/best_available_executed_notebook.ipynb`](best_result/best_available_executed_notebook.ipynb).

Its saved evaluation output reports:

- test RMSE: **15.032**
- test R2: **0.868**

This notebook was selected because it has the lowest executed test RMSE among the FD003 notebooks received from the current PC. The other unique executed notebooks report RMSE values of 15.313, 15.402, 15.746, 16.176, 17.493, and 17.745, or contain a failed/incomplete GA attempt.

## Important distinction

The thesis reports a lower parallel-model RMSE of **14.75** and a GA result of **14.98**, but the corresponding executed notebooks or exports were not found on this PC. Therefore:

- 15.032 is the **best available executed notebook** in this repository;
- 14.75 remains the **best reported thesis result**;
- neither value is labelled independently reproduced here.

See [`CURRENT_PC_CHECKPOINT.md`](CURRENT_PC_CHECKPOINT.md) for the missing university-PC items.

## How to verify the selection

1. Open the best-result notebook and locate its final evaluation output.
2. Compare it with the experiment inventory in [`MANIFEST.md`](MANIFEST.md).
3. Use the retained PDF exports in [`archive/evidence`](archive/evidence) to confirm the 17.493 and 15.402 parallel runs and the failed GA attempt.

## Dataset and task

- task: turbofan remaining-useful-life regression
- subset: FD003
- training engines: 100
- test engines: 100
- operating conditions: one
- fault modes: two

Raw C-MAPSS files are not included. See [`data/README.md`](data/README.md).

## Before rerunning

- replace machine-specific data paths;
- document the RUL target construction and clipping policy;
- fit preprocessing only on training observations;
- reserve validation and test engines explicitly;
- replace the historical regression checkpoint's unavailable `val_accuracy` monitor;
- pin the software environment and random seeds.
