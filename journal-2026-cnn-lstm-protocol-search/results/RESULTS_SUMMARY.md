# Review experiments v16: complete results (lab PC + RTX 3090)

All values are test RMSE, shown as mean ± s.d. over seeds, unless stated otherwise. A paired difference is (first − second) on shared seeds, given with its 95% CI and paired t-test p-value.

**Machines.** Parts A–E and H were run on the lab PC (Quadro RTX 6000). Parts F, G, I, J and K were run on an RTX 3090 with TF32 off. The lab PC also repeated Part G and six seeds of Part I.

## Cross-machine check

| Part | Result on the two GPUs |
|---|---|
| G, within-cell battery runs | Agree to 5 decimals (largest difference 7.6×10⁻⁵) |
| G, leave-one-cell-out | Largest difference 0.0015 |
| I, DS05 | Differ by up to 0.76 RMSE per seed, about the size of the seed-to-seed spread |

The paper therefore compares within one GPU only:
- **3090:** Parts I, J and K. The DS05 parallel reference for Part K is the Part I 28-channel run, which uses the same seeds.
- **Lab PC:** Parts A–E and H.

## Part A: sequential vs parallel under the final protocols (lab PC, 10 seeds)

| Set | Parallel | Sequential | Parallel − sequential | p |
|---|---|---|---|---|
| DS05 capped | 4.869 ± 0.200 | 5.146 ± 0.161 | −0.277 [−0.490, −0.064] | 0.017 |
| DS05 uncapped | 7.561 ± 0.288 | 7.917 ± 0.371 | −0.356 [−0.792, +0.080] | 0.098 |
| FD003 | 11.046 ± 0.480 | 11.207 ± 0.796 | −0.160 [−0.808, +0.487] | 0.59 |
| B0005, trend-anchored | 0.00958 ± 0.00258 | 0.00860 ± 0.00146 | +0.0010 [−0.0014, +0.0034] | 0.38 |

MAE and NASA score: DS05 parallel MAE 3.77, score 19,736; FD003 parallel MAE 7.47, score 284.

## Part B: equal-budget search for the sequential model

- **Budget:** 45 PSO evaluations plus two-stage selection.
- **Selected configuration:** pso_14. It has 4 conv blocks of 120 filters, kernel 2 and pooling on, 1 LSTM layer of 128 units, and dense layers of 201 and 69 units; 142,784 parameters.

## Part C: pooling (6 seeds)

| Model | Pooling stages = 0 | 1 | 2 | Pooling 2 − 0 |
|---|---|---|---|---|
| DS05 parallel | 4.887 | – | 5.885 | +0.999, p = 0.030 |
| DS05 sequential | 5.066 | 5.145 | 5.163 | +0.097, p = 0.48 |
| FD003 parallel | 11.231 | – | 11.166 | not significant |
| FD003 sequential | 10.746 | 10.859 | 10.641 | not significant |

Interaction on DS05: −0.90 (p = 0.074). Removing pooling does not rescue the sequential model.

## Part D: model size and single branches

- **Parameter counts:**

  | Model | Parameters |
  |---|---|
  | Parallel | 162,001 |
  | Sequential | 142,784 |
  | Capacity-matched sequential | 164,665 (+1.6%) |
  | CNN only | 120,113 |
  | LSTM only | 41,905 |

- **Capacity-matched sequential vs parallel, DS05:** 5.299 vs 4.869, +0.40 [+0.14, +0.66], p = 0.01.
- **FD003, 10 seeds:** CNN only 10.97, LSTM only 11.01, parallel 11.02; all p > 0.78.
- **B0005, 10 seeds:** CNN only 0.00877, LSTM only 0.00999, parallel 0.00958; all p > 0.44.

## Part E: five search methods × 3 independent campaigns (DS05, lab PC)

Equal budget for every method: 45 evaluations, or an equal total of 4,500 epochs for Hyperband. Initial designs were far from known configurations.

| Method | Best val. | Test of best | Median val. | GPU-h (3 campaigns) |
|---|---|---|---|---|
| Random | 3.672 ± 0.086 | 5.016 ± 0.488 | 4.547 | 15.7 |
| BO | 3.714 ± 0.012 | 5.146 ± 0.102 | 4.391 | 20.5 |
| PSO | **3.550 ± 0.089** | **4.733 ± 0.238** | 4.199 | 22.4 |
| GA | 3.688 ± 0.063 | 5.114 ± 0.624 | 4.097 | 15.8 |
| Hyperband | 3.698 ± 0.100 | 4.946 ± 0.384 | 4.041 | 41.8 |

- **Friedman test across the 3 campaigns:**
  - best validation: χ² = 4.27, p = 0.37;
  - test of best: χ² = 2.40, p = 0.66;
  - median validation: χ² = 9.33, p = 0.053.
- **Reading:** PSO has the best mean rank, but no method is reliably better than another. Random search is not beaten significantly.
- **Hyperband cost:** with an equal epoch budget it used about twice the GPU time, because each of its 436 evaluations per campaign carries the full overhead of loading data and scoring the test set.

## Part F: Holm correction of the 12-factor screen

Smallest Holm-adjusted p = 0.288, so no screening gain replicates. This is unchanged from v17.

## Part G: battery B0005, 30 seeds

| Model | Test RMSE |
|---|---|
| Parallel, trend-anchored | 0.00892 ± 0.00174 |
| Sequential, trend-anchored | 0.00819 ± 0.00123 |
| CNN only, trend-anchored | 0.00927 ± 0.00203 |
| LSTM only, trend-anchored | 0.01050 ± 0.00129 |
| Parallel, network alone | 0.01525 ± 0.01368 |
| Sequential, network alone | 0.00919 ± 0.00296 |
| Trend alone | 0.01439 |
| Flat line | 0.02927 |

**Comparisons:**
- **Trend-anchored vs network alone (parallel):** −0.0063, p = 0.015.
- **Parallel vs trend alone:** p < 10⁻¹⁶.
- **Parallel − sequential, both trend-anchored:** +0.0007 [+0.0001, +0.0014], p = 0.034. The sequential model is slightly better.
- **CNN only vs parallel:** p = 0.50. **LSTM only vs parallel:** worse, p < 0.001.

**Leave one cell out (10 seeds; difference from training on the target cell itself):**

| Cell | Train on target cell | Zero-shot (other cells only) | Pooled (all cells) | Flat line |
|---|---|---|---|---|
| B0005 | 0.00958 | 0.01557 (+0.0060, p = 0.035) | 0.01276 (+0.0032, p < 0.001) | 0.0293 |
| B0006 | 0.01324 | 0.02124 (+0.0080, p = 0.013) | 0.01440 (n.s.) | 0.0461 |
| B0007 | 0.00775 | 0.00905 (n.s.) | 0.00708 (−0.0007, p = 0.053) | 0.0248 |
| B0018 | 0.05152 | 0.04831 (n.s.) | 0.04619 (−0.0053, p < 0.001) | **0.0202** |

**B0018 is a failure case.** Every model, and the trend alone (0.0497), is worse than the flat line on this cell.

## Part H: FD003 L2 re-selected on validation

- **Selection (6 seeds per dose):** validation picked λ = 0.3 for the parallel model and 0.03 for the sequential.
- **Confirmation (15 fresh seeds, parallel model):**
  - λ = 0.3: 11.041 ± 0.397;
  - λ = 0.01: 10.900 ± 0.510;
  - difference +0.141 [−0.269, +0.550], p = 0.47.
- **Classical baselines on FD003:**

  | Model | Train | Test |
  |---|---|---|
  | Linear regression | 18.38 | 19.12 |
  | Random forest | 4.76 | 16.26 |

## Part I: DS05 inputs (3090, 6 seeds)

| Input | Capped | Uncapped |
|---|---|---|
| 28 channels | 4.908 ± 0.071 | 9.924 |
| 4 operating conditions + 14 physical | 5.433 ± 0.348 | 11.374 |
| 14 physical only | 6.237 ± 0.470 | 12.267 |

- **14 physical only vs 28 channels:** +1.329 [+0.864, +1.793], p = 0.0007.
- **4 operating conditions + 14 physical vs 28 channels:** +0.525 [+0.208, +0.841], p = 0.008.

## Part J: other subsets (3090, 6 seeds)

| Dataset | Parallel | Sequential | Parallel − sequential (p) | Trivial | Skill (parallel) |
|---|---|---|---|---|---|
| FD001 | 12.38 ± 0.21 | 12.14 ± 0.36 | +0.23 (0.35) | 40.09 | +0.691 |
| FD002 | 12.99 ± 0.42 | 13.23 ± 0.82 | −0.24 (0.64) | 43.01 | +0.698 |
| FD004 | 13.27 ± 0.48 | 14.14 ± 1.44 | −0.87 (0.20) | 43.65 | +0.696 |
| DS02 (capped) | 4.21 ± 0.19 | 4.35 ± 0.43 | −0.14 (0.48) | 18.525 | +0.773 |

The DS02 trivial baselines (18.525 capped, 20.400 uncapped) were computed with the notebook's method, which reproduces the DS05 values of 18.393 and 23.142 exactly.

## Part K: external baselines (3090, 6 seeds)

| Model | DS05 | − parallel (3090) | FD003 | NASA score DS05 / FD003 |
|---|---|---|---|---|
| Parallel (Part I, 28 channels) | 4.908 | – | – | – |
| TCN | **4.611** | −0.298 [−0.499, −0.096], p = 0.013 | 12.22 | 18,327 / 274 |
| GRU | 5.407 | +0.499, p = 0.063 | 11.27 | 22,032 / 220 |
| BiLSTM | 5.395 | +0.486, p = 0.005 | 11.71 | 22,253 / 246 |
| Transformer | 6.944 (4.838 without one diverged seed) | n.s. | 13.06 | 49,357 / 352 |
| Parallel, MC-dropout mean | 5.045 | +0.137, p = 0.17 | **10.97** | 20,833 / 267 |

- **On FD003 the parallel model is best.** Compared with the parallel model on the same GPU (MC-dropout mean, the only parallel FD003 run on the 3090):
  - TCN +1.25 (p = 0.005);
  - Transformer +2.09 (p = 0.014);
  - GRU and BiLSTM not significant.
- **MC-dropout uncertainty:** the 95% intervals cover only 31% of DS05 test points (mean width 3.8) and 78% of FD003 points (width 21.3). They are badly under-dispersed.

## Compute

| Machine | Part | GPU-h |
|---|---|---|
| Lab PC | Part E search (Hyperband 41.8 of it) | about 120 |
| Lab PC | A | 7.7 |
| Lab PC | C | 5.4 |
| Lab PC | H | 6.2 |
| Lab PC | D | 2.2 |
| Lab PC | B two-stage | 2.2 |
| 3090 | F–K | 3.8 |
