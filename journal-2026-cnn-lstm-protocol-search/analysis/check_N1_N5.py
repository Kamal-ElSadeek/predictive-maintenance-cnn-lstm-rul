"""Reproduces the five numbers requested in the professor's v21 notes (N1-N5) from the files in this repository.
Run from the repository folder:  python analysis/check_N1_N5.py
"""
import json, os, numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(HERE, *a)

print('N1  Protocol factorial (02_ncmapss_ds05_campaigns/logs/ncmapss_v7_protocol_factorial_cells.csv)')
F = pd.read_csv(P('02_ncmapss_ds05_campaigns', 'logs', 'ncmapss_v7_protocol_factorial_cells.csv'))
x = np.c_[np.where(F.sensors == 28, 1, -1), np.where(F.window == 'T=30', 1, -1), np.where(F.train_cap.astype(str) == '55', 1, -1)]
m, s = F.capped_mean.values, F.capped_sd.values; n = F.seeds.values[0]
mse = np.mean(s ** 2); df = len(F) * (n - 1); tc = stats.t.ppf(.975, df)
def effect(c):
    e = (m * c).sum() / 4; se = np.sqrt(mse / n * ((c / 4) ** 2).sum()); return e, e - tc * se, e + tc * se
for name, c in [('14 virtual sensors', x[:, 0]), ('30-step window', x[:, 1]), ('cap 55 on training target', x[:, 2]),
                ('sensors x window', x[:, 0] * x[:, 1]), ('sensors x cap', x[:, 0] * x[:, 2]), ('window x cap', x[:, 1] * x[:, 2])]:
    print('    %-26s %+.2f  95%% CI [%+.2f, %+.2f]' % ((name,) + effect(c)))
print('    (pooled within-cell variance, %d df; per-seed values were not saved, so the CI uses the cell s.d.)' % df)

print('\nN2  B0005 architecture comparison (04_battery_b0005/logs/recursive_bo_b0005_start134.json)')
J = json.load(open(P('04_battery_b0005', 'logs', 'recursive_bo_b0005_start134.json')))
A = pd.DataFrame(J['ablation']); r = stats.spearmanr(A.val_mean, A.test_mean)
print('    10 variants: Spearman rho(val, test) = %+.2f, p = %.3f' % (r.statistic, r.pvalue))
w = A.sort_values('val_mean').iloc[0]
print('    validation winner: %s, %s target, test %.4f;  60-trial search best config: test %.4f (%d params)'
      % (w.config, w.target, w.test_mean, J['tuned']['mean'], J['tuned']['params']))
print('    per-trial test errors of the 4x30 and 60-trial searches were not stored (trial.json holds validation only)')

print('\nN3  Cross-GPU (results/v16_labpc_rtx6000/runs.jsonl vs results/v16_rtx3090/runs.jsonl)')
def load(f): return {json.loads(l)['key']: json.loads(l) for l in open(f) if l.strip()}
L = load(P('results', 'v16_labpc_rtx6000', 'runs.jsonl')); T = load(P('results', 'v16_rtx3090', 'runs.jsonl'))
common = set(L) & set(T)
d = np.array([L[k]['test'] - T[k]['test'] for k in common if L[k]['dataset'] == 'DS05'])
b = np.array([L[k]['test'] - T[k]['test'] for k in common if L[k]['dataset'] == 'B0005'])
print('    DS05: %d seeds, mean |diff| %.2f, range %.2f-%.2f, mean signed %+.2f (RTX 6000 - RTX 3090)'
      % (len(d), np.abs(d).mean(), np.abs(d).min(), np.abs(d).max(), d.mean()))
print('    B0005: %d runs, largest |diff| %.5f' % (len(b), np.abs(b).max()))

print('\nN4  Seed spread before/after search (02_ncmapss_ds05_campaigns/logs/ncmapss_v9_stage2.csv)')
S = pd.read_csv(P('02_ncmapss_ds05_campaigns', 'logs', 'ncmapss_v9_stage2.csv')).set_index('name')
s1, s2, n = S.loc['baseline', 'capped_sd'], S.loc['new1_PSO', 'capped_sd'], 6
diff = S.loc['baseline', 'capped'] - S.loc['new1_PSO', 'capped']; t_paired = 4.48   # paired t reported in Section V-D
sdd = diff * np.sqrt(n) / t_paired; r = (s1 ** 2 + s2 ** 2 - sdd ** 2) / (2 * s1 * s2); Fv = s1 ** 2 / s2 ** 2
t = (Fv - 1) * np.sqrt(n - 2) / (2 * np.sqrt(Fv * (1 - r ** 2))); p = 2 * (1 - stats.t.cdf(abs(t), n - 2))
print('    s.d. %.3f -> %.3f, implied seed correlation %.2f, Pitman-Morgan t = %.2f, p = %.2f' % (s1, s2, r, t, p))

print('\nN5  Equivalence margin (results/v16_labpc_rtx6000/partA_runs.csv, results/v16_rtx3090/partJ_runs.csv)')
def tost(dd, mrg):
    se = dd.std(ddof=1) / np.sqrt(len(dd)); k = len(dd) - 1
    return max(1 - stats.t.cdf((dd.mean() + mrg) / se, k), stats.t.cdf((dd.mean() - mrg) / se, k))
A_ = pd.read_csv(P('results', 'v16_labpc_rtx6000', 'partA_runs.csv')); J_ = pd.read_csv(P('results', 'v16_rtx3090', 'partJ_runs.csv'))
for name, df_, triv in [('FD003', A_[A_.set == 'FD003'], 42.17), ('FD001', J_[J_.dataset == 'FD001'], 40.088)]:
    par = df_[df_.arch == 'par'].set_index('seed').test; seq = df_[df_.arch == 'seq'].set_index('seed').test
    dd = (par - seq).dropna().values
    print('    %s (%d seeds): 2%% margin %.2f p = %.3f | 1%% margin %.2f p = %.3f | margin / s.d.(parallel) = %.2f'
          % (name, len(dd), .02 * triv, tost(dd, .02 * triv), .01 * triv, tost(dd, .01 * triv), .02 * triv / par.std(ddof=1)))
