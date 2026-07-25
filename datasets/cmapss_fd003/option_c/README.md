# Option C — C-MAPSS FD003 sequential optimization

This notebook completes the optimization coverage for C-MAPSS FD003. In the original study the
**sequential** model was evaluated only at its baseline, while the parallel model was optimized
with GA and Hyperband. Here the sequential model is optimized with the **same two optimizers**,
so the two architectures can be compared under equal optimization.

## Result (executed)

| Model | Baseline | GA | Hyperband |
|---|---|---|---|
| C-MAPSS FD003 sequential | 15.31 | 15.84 | 16.26 |

Optimization did not improve on the baseline. Because the optimized parallel model (14.98) still
outperforms the optimized sequential model, the parallel advantage on this dataset is not an
artifact of unequal optimization.

- **Evidence label:** Executed result — the RMSE values are present in the saved notebook output.
- **Entry point:** [`cmapss_fd003_sequential_GA_Hyperband.ipynb`](cmapss_fd003_sequential_GA_Hyperband.ipynb)

## Pipeline

Matches the reported FD003 experiments: condition-specific StandardScaler, exponential smoothing
(alpha 0.4), RUL clipped at 125, 14 selected sensors, sliding window of 20, engine-unit-grouped
80/20 train/validation split, standard FD003 test set. GA: population 10, 5 generations, two-point
crossover 0.5, Gaussian mutation, tournament selection. Hyperband: factor 3, max_epochs 30,
searching the training/head hyperparameters with the convolutional and recurrent structure fixed.

## Environment

Python 3.8, TensorFlow 2.6.0, Keras 2.6.0, DEAP 1.3.1, keras-tuner (pip). Random seed 42. Run on
an NVIDIA RTX 3090.
