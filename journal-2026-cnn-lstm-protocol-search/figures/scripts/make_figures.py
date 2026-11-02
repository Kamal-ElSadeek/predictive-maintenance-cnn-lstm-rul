"""Regenerates the paper figures from the stored data and run logs, as SVG (vector) and 300-dpi PNG.

Inputs (paths relative to DATA, the thesiswork folder):
  Nasa_battery_dataset/B0005.mat, protocol_study_b0005.csv, bo_full_b0005/start134/trial_*/trial.json
  ncmapss_v9_all_trials.csv (DS05 controlled comparison, all arms)
  l2_dose_summary (per-dose means from FD003_L2_DoseResponse_v9.ipynb, transcribed below)
  stage1_curves.json (Keras training logs parsed from the Stage 1 notebooks)
Run:  python make_figures.py DATA_DIR OUT_DIR
"""
import sys, os, json, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.io import loadmat

DATA = sys.argv[1] if len(sys.argv) > 1 else '.'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'figures'
V9 = sys.argv[3] if len(sys.argv) > 3 else os.path.join(DATA, 'ncmapss_v9_all_trials.csv')
CURVES = sys.argv[4] if len(sys.argv) > 4 else os.path.join(DATA, 'stage1_curves.json')
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'],
                     'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'svg.fonttype': 'path', 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.25, 'lines.linewidth': 1.2})
BLUE, ORANGE, GREY, RED, GREEN = '#0072B2', '#D55E00', '#7F7F7F', '#C00000', '#009E73'
COL = 3.4      # single-column width, inches
def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.svg'), bbox_inches='tight', pad_inches=0.02)
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

def soh_b0005():
    mat = loadmat(os.path.join(DATA, 'Nasa_battery_dataset', 'B0005.mat')); cyc = mat['B0005'][0, 0]['cycle'][0]
    cap = [float(c['data'][0][0]['Capacity'][0][0]) for c in cyc if c['type'][0] == 'discharge']
    cap = np.array(cap); return cap / cap[0], cap[0]

# ---------------- Fig. 2: B0005 state of health
S, C0 = soh_b0005(); k = np.arange(1, len(S) + 1)
fig, ax = plt.subplots(figsize=(COL, 1.95))
ax.plot(k, S, color=BLUE, lw=1.1, label='measured SoH')
ax.axhline(0.70, color=RED, ls='--', lw=0.9, label='70% SoH')
ax.axvline(134, color=GREY, ls=':', lw=0.9); ax.text(135, 0.97, 'forecast origin\n(cycle 134)', fontsize=6.5, va='top')
ax.set_xlabel('Discharge cycle'); ax.set_ylabel('SoH = $C_k/C_0$'); ax.set_xlim(0, 170); ax.legend(loc='lower left', frameon=False)
save(fig, 'fig02_b0005_soh')

# ---------------- Fig. 6: validation vs test, all configurations and best 20%
d = pd.read_csv(V9)
A = d[d.tag.isin(['random', 'bo_init', 'bo', 'pso']) & (d.val < 900)].dropna(subset=['test_capped'])
top = A.nsmallest(int(len(A) * 0.2), 'val')
rho_a, _ = stats.spearmanr(A.val, A.test_capped); rho_t, p_t = stats.spearmanr(top.val, top.test_capped)
x, y = top.val.values, top.test_capped.values; n = len(x); b1, b0 = np.polyfit(x, y, 1); s = np.sqrt(((y - b0 - b1 * x) ** 2).sum() / (n - 2))
xs = np.linspace(x.min(), x.max(), 50); pi = stats.t.ppf(.975, n - 2) * s * np.sqrt(1 + 1 / n + (xs - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum())
fig, ax = plt.subplots(1, 2, figsize=(COL * 2.05, 2.25))
for tags, c, l, mk in [(['random'], GREY, 'Random search', 'o'), (['bo_init', 'bo'], RED, 'BO', 's'), (['pso'], BLUE, 'PSO', '^')]:
    q = A[A.tag.isin(tags)]; ax[0].scatter(q.val, q.test_capped, s=9, color=c, marker=mk, alpha=.75, edgecolors='none', label=l)
    q = top[top.tag.isin(tags)]; ax[1].scatter(q.val, q.test_capped, s=12, color=c, marker=mk, alpha=.85, edgecolors='none')
ax[0].axvspan(A.val.min(), top.val.max(), color='#FFE699', alpha=.45, zorder=0)
ax[0].set_xlabel('Validation RMSE'); ax[0].set_ylabel('Test RMSE (capped at 55)'); ax[0].legend(frameon=False, loc='upper left')
ax[0].set_title(r'All %d configurations: $\rho$ = %+.3f' % (len(A), rho_a))
ax[1].plot(xs, b0 + b1 * xs, 'k-', lw=1); ax[1].fill_between(xs, b0 + b1 * xs - pi, b0 + b1 * xs + pi, color='k', alpha=.1, label='95% prediction interval')
ax[1].set_xlabel('Validation RMSE'); ax[1].set_title(r'Best 20%% (n = %d): $\rho$ = %+.3f, p = %.2f' % (n, rho_t, p_t)); ax[1].legend(frameon=False, loc='upper left')
fig.tight_layout(w_pad=1.5); save(fig, 'fig06_val_test')

# ---------------- Fig. 11 (main Fig. 7): best validation error against evaluations, single campaign per arm
fig, ax = plt.subplots(figsize=(COL, 2.2))
for tags, c, ls, l in [(['random'], GREY, '-', 'Random search'), (['bo_init', 'bo'], RED, '--', 'BO'), (['pso'], BLUE, '-', 'PSO')]:
    g = d[d.tag.isin(tags)].reset_index(drop=True); cm = g.val.cummin().values
    ax.plot(np.arange(1, len(cm) + 1), cm, color=c, ls=ls, lw=1.3, label='%s (best %.3f, eval %d)' % (l, cm[-1], int(np.argmin(g.val.values)) + 1))
ax.set_ylim(3.45, 6.0); ax.set_xlabel('Evaluations'); ax.set_ylabel('Best validation RMSE so far'); ax.legend(frameon=False)
save(fig, 'fig11_search_convergence')

# ---------------- Fig. 16 (main Fig. 10): L2 dose-response on FD003 (15 seeds per dose; FD003_L2_DoseResponse_v9.ipynb)
L2 = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0]); M = np.array([10.911893, 10.883319, 10.637782, 10.397958, 10.887006, 11.298689])
SE = np.array([0.098341, 0.159199, 0.113595, 0.074214, 0.154594, 0.128204]); WN = np.array([0.048497, 0.047266, 0.048098, 0.049964, 0.054427, 0.061561])
lx = np.log10(L2); cq = np.polyfit(lx, M, 2, w=1 / SE); xx = np.linspace(lx.min(), lx.max(), 100)
fig, ax = plt.subplots(1, 2, figsize=(COL * 2.05, 2.2))
ax[0].errorbar(L2, M, SE, fmt='o-', color=BLUE, capsize=3, lw=1.2, label='test RMSE, mean ± SE')
ax[0].plot(10 ** xx, np.polyval(cq, xx), '--', color=RED, lw=1, label='quadratic in log$_{10}\\lambda$')
ax[0].axhline(10.900, color=GREY, ls=':', lw=0.9); ax[0].plot([0.3], [11.041], 'D', color=ORANGE, ms=5, label='$\\lambda$ = 0.3 re-selected on validation,\n15 fresh seeds')
ax[0].set_xscale('log'); ax[0].set_xlabel('L2 strength $\\lambda$'); ax[0].set_ylabel('Test RMSE'); ax[0].legend(frameon=False, loc='upper left', fontsize=6.3)
ax[0].set_ylim(10.15, 11.75)
ax[1].plot(L2, WN, 's-', color=BLUE); ax[1].set_xscale('log'); ax[1].set_xlabel('L2 strength $\\lambda$'); ax[1].set_ylabel('Mean |weight|')
fig.tight_layout(w_pad=1.5); save(fig, 'fig16_l2_dose')

# ---------------- Fig. 17 (main Fig. 11): B0005 forecast region and protocol comparison
o = int(0.8 * len(S)) - 1; kk = np.arange(o - 20, o); a1, a0 = np.polyfit(kk, S[kk], 1); kf = np.arange(o, len(S))
P = pd.read_csv(os.path.join(DATA, 'Nasa_battery_dataset', 'protocol_study_b0005.csv'))
fig, ax = plt.subplots(1, 2, figsize=(COL * 2.05, 2.2))
ax[0].plot(k, S, color='k', lw=1.1, label='measured SoH')
ax[0].plot(kf + 1, a0 + a1 * kf, '--', color=GREEN, lw=1.2, label='linear trend fitted to cycles 114–133')
ax[0].plot(kf + 1, np.full(len(kf), S[o - 1]), ':', color=GREY, lw=1.2, label='flat line (last value)')
ax[0].axvline(o + 1, color=GREY, ls=':', lw=0.8); ax[0].axhline(0.70, color=RED, ls='--', lw=0.7)
ax[0].set_xlabel('Discharge cycle'); ax[0].set_ylabel('SoH'); ax[0].legend(frameon=False, loc='lower left', fontsize=6.3)
cols = ['#1F3A68', '#2E5597', '#8FA9CC', '#C5D2E5']
ax[1].bar(P.id, P['mean'], yerr=P['std'], color=cols, capsize=3, width=0.65)
for i, (m_, s_) in enumerate(zip(P['mean'], P['std'])): ax[1].text(i, m_ + s_ + 0.0002, '%.4f' % m_, ha='center', fontsize=6.5)
ax[1].set_ylabel('Test RMSE (SoH)'); ax[1].set_xlabel('Protocol'); ax[1].set_ylim(0, 0.0185)
fig.tight_layout(w_pad=1.5); save(fig, 'fig17_battery_protocols')

# ---------------- Fig. 18 (main Fig. 12): B0005 Bayesian search, 60 trials
tr = []
for f in sorted(glob.glob(os.path.join(DATA, 'Nasa_battery_dataset', 'bo_full_b0005', 'start134', 'trial_*', 'trial.json'))):
    j = json.load(open(f))
    if j.get('score') is not None: tr.append((int(j['trial_id']), j['score']))
tr.sort(); sc = np.array([s_ for _, s_ in tr]); best = np.minimum.accumulate(sc)
fig, ax = plt.subplots(figsize=(COL, 2.0))
ax.scatter(np.arange(1, len(sc) + 1), sc, s=7, color=GREY, alpha=.6, label='trial (validation forecast RMSE)')
ax.step(np.arange(1, len(sc) + 1), best, where='post', color=BLUE, lw=1.3, label='best so far')
ax.set_yscale('log'); ax.set_xlabel('Trial'); ax.set_ylabel('Validation forecast RMSE'); ax.legend(frameon=True, framealpha=0.9, loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=2)
save(fig, 'fig18_battery_bo')
print('battery BO trials:', len(sc), 'best', best[-1])

# ---------------- Supplementary Figs. S1, S2, S5, S6: Stage 1 training curves from the notebook logs
cur = json.load(open(CURVES))
for key, name in [('S1_fd003_seq', 'figS1_fd003_seq_curves'), ('S2_fd003_par', 'figS2_fd003_par_curves'),
                  ('S5_ncm_seq', 'figS5_ncmapss_seq_curves'), ('S6_ncm_par', 'figS6_ncmapss_par_curves')]:
    v = np.array(cur[key]); e = np.arange(1, len(v) + 1)
    fig, ax = plt.subplots(figsize=(COL, 1.8))
    ax.plot(e, v[:, 0], 'o-', ms=2.5, color=BLUE, label='training loss'); ax.plot(e, v[:, 1], 's-', ms=2.5, color=ORANGE, label='validation loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE loss'); ax.set_xticks(e if len(e) <= 11 else e[::2]); ax.legend(frameon=False)
    if key == 'S1_fd003_seq': ax.set_ylim(0, 400)
    save(fig, name)
print('done ->', OUT)
