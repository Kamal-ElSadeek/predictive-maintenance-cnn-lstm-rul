# Option C — NASA Battery B0005 parallel optimization

This notebook completes the optimization coverage for the NASA Battery dataset. In the original
study the **parallel** model was evaluated only at its baseline, while the sequential model was
optimized with GA and Hyperband. Here the parallel model is optimized with the **same two
optimizers**, so the two architectures can be compared under equal optimization.

## Result (executed)

| Model | Baseline | GA | Hyperband |
|---|---|---|---|
| NASA Battery B0005 parallel | 0.047 | 0.527 | 0.492 |

Optimization did not improve on the baseline; it degraded it substantially. The large
validation-to-test gap indicates overfitting driven by the small number of discharge-cycle windows
available for training (train cycles < 50). The **0.047 baseline remains the best parallel result**,
and comparing the best achievable per architecture (parallel 0.047 vs sequential 0.094) the parallel
model still wins. These runs are retained as an optimization finding, not as a replacement for the
0.047 result.

- **Evidence label:** Executed result — the RMSE values are present in the saved notebook output.
- **Entry point:** [`battery_b0005_parallel_GA_Hyperband.ipynb`](battery_b0005_parallel_GA_Hyperband.ipynb)

## Pipeline

Matches the reported Battery experiments: capacity signal, sliding window of 10 cycles, MinMax
scaling, train on cycles < 50 and test on cycles >= 50 of B0005, RMSE computed on capacity. GA:
population 10, 5 generations, two-point crossover 0.5, Gaussian mutation, tournament selection.
Hyperband: factor 3, max_epochs 10, searching CNN filters, LSTM units, and learning rate.

## Environment

Python 3.8, TensorFlow 2.6.0, Keras 2.6.0, DEAP 1.3.1, keras-tuner (pip). Random seed 42.
