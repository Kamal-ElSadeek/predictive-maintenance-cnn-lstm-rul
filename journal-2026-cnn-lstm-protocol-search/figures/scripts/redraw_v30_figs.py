"""Redraws Figs. 7-14 and 21-25 of the v30 manuscript as vector graphics at single-column width
(3.5 in) with text of at least 7 pt, from the logged run data in ../data.
Usage:  python redraw_v30_figs.py DATA_DIR OUT_DIR
"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.io import loadmat

DATA = sys.argv[1] if len(sys.argv) > 1 else 'data'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'out'
os.makedirs(OUT, exist_ok=True)
W = 3.5
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
                     'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.grid': True, 'grid.alpha': 0.25,
                     'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.2,
                     'svg.fonttype': 'none', 'legend.frameon': False})
BLUE, ORANGE, RED, GREY, GREEN, PURPLE = '#0072B2', '#D55E00', '#C00000', '#7F7F7F', '#009E73', '#7030A0'
P = lambda f: os.path.join(DATA, f)

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.svg'), bbox_inches='tight', pad_inches=0.02)
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig); print('wrote', name)

C = json.load(open(P('stage1_curves.json')))
def loss_fig(key, name, logy=False):
    tr, va = np.array(C[key]).T; e = np.arange(1, len(tr) + 1)
    fig, ax = plt.subplots(figsize=(W, 1.75))
    ax.plot(e, tr, 'o-', ms=2.5, color=BLUE, label='training')
    ax.plot(e, va, 's-', ms=2.5, color=ORANGE, label='validation')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (MSE)'); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if logy: ax.set_yscale('log')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2); save(fig, name)

# Figs. 7, 8, 11, 12: Stage 1 loss curves (values printed in the Stage 1 notebooks)
loss_fig('S1_fd003_seq', 'fig07_fd003_seq_loss')
loss_fig('S2_fd003_par', 'fig08_fd003_par_loss')
loss_fig('S5_ncm_seq', 'fig11_ds05_seq_loss')
loss_fig('S6_ncm_par', 'fig12_ds05_par_loss')

# Figs. 9, 10: Stage 1 B0005 one-step forecasts (prediction traced from the original run output)
mat = loadmat(P('B0005.mat')) if os.path.exists(P('B0005.mat')) else None
for f, name in [('fig09_b0005_seq_stage1_traced.csv', 'fig09_b0005_seq_forecast'),
                ('fig10_b0005_par_stage1_savedmodel.csv', 'fig10_b0005_par_forecast')]:
    d = pd.read_csv(P(f)).rename(columns={'pred_Ah': 'pred_Ah_traced'})
    cyc = mat['B0005'][0, 0]['cycle'][0]
    cap = np.array([float(c['data'][0][0]['Capacity'][0][0]) for c in cyc if c['type'][0] == 'discharge'])
    rm = np.sqrt(np.mean((d.pred_Ah_traced - d.actual_Ah) ** 2))
    fig, ax = plt.subplots(figsize=(W, 2.0))
    ax.plot(np.arange(1, len(cap) + 1), cap, color=BLUE, lw=1.0, label='measured capacity')
    ax.plot(d.cycle, d.pred_Ah_traced, color=RED, lw=1.0, label='prediction (test RMSE %.3f Ah)' % rm)
    ax.axhline(1.38, color=GREY, ls='--', lw=0.9, label='failure threshold (1.38 Ah)')
    ax.axvline(49.5, color=GREY, ls=':', lw=0.8)
    ax.text(47, 1.86, 'train', ha='right', fontsize=7, color=GREY); ax.text(52, 1.86, 'test', ha='left', fontsize=7, color=GREY)
    ax.set_xlabel('Discharge cycle'); ax.set_ylabel('Capacity (Ah)'); ax.set_xlim(0, 170)
    ax.set_ylim(1.27, 1.9)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=2); save(fig, name)

# Figs. 13, 14: training histories of the Bayesian-optimized DS05 models (printed epoch logs)
for f, name in [('fig13_ds05_seq_bo_history.csv', 'fig13_ds05_seq_bo_history'),
                ('fig14_ds05_par_bo_history.csv', 'fig14_ds05_par_bo_history')]:
    h = pd.read_csv(P(f))
    fig, axes = plt.subplots(2, 1, figsize=(W, 3.0), sharex=True)
    for ax, (a, b, lab) in zip(axes, [('loss', 'val_loss', 'Loss (MSE)'), ('mae', 'val_mae', 'MAE')]):
        ax.plot(h.epoch, h[a], 'o-', ms=2.2, color=BLUE, label='training')
        ax.plot(h.epoch, h[b], 's-', ms=2.2, color=ORANGE, label='validation')
        ax.set_ylabel(lab)
    axes[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2); axes[1].set_xlabel('Epoch')
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout(h_pad=0.4); save(fig, name)

# Fig. 21: first Bayesian search on FD003 (bo_trials_fd003.csv); baseline printed in the notebook
T = pd.read_csv(P('bo_trials_fd003.csv'))
base = dict(val_exh=9.479, val_tl=10.097, test=10.578)
fig, axes = plt.subplots(3, 1, figsize=(W, 6.4), constrained_layout=True)
for ax, col, lab in [(axes[0], 'val_exh', 'Exhaustive validation RMSE'), (axes[1], 'val_tl', 'Test-like validation RMSE')]:
    sc = ax.scatter(T[col], T.test, c=T.params / 1e6, cmap='viridis', s=14, zorder=3)
    ax.scatter([base[col]], [base['test']], marker='*', s=70, color=RED, zorder=4, label='baseline')
    ax.set_xlabel(lab); ax.set_ylabel('Test RMSE'); ax.legend(loc='upper left')
cb = fig.colorbar(sc, ax=axes[:2], fraction=0.05, pad=0.02, aspect=30); cb.set_label('Parameters (millions)')
ax = axes[2]; tr = T.trial + 1
ax.step(tr, np.minimum.accumulate(T.val_exh), where='post', color=BLUE, label='best validation so far')
ax.scatter(tr, T.test, s=10, color=PURPLE, zorder=3, label='test RMSE of each trial')
ax.axhline(base['test'], color=RED, ls='--', lw=0.9, label='baseline test (%.2f)' % base['test'])
ax.set_xlabel('Trial'); ax.set_ylabel('RMSE'); ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
for a, l in zip(axes, 'abc'): a.text(-0.16, 1.02, '(%s)' % l, transform=a.transAxes, fontsize=8, fontweight='bold')
save(fig, 'fig21_fd003_bo_diagnostic')

# Fig. 22: landscape probe and cold-start searches on FD003 (coldstart_fd003.json)
J = json.load(open(P('coldstart_fd003.json')))
pr = pd.DataFrame(J['probe']).sort_values('test')
L = pd.DataFrame(J['log'])
fig, axes = plt.subplots(3, 1, figsize=(W, 6.0), gridspec_kw=dict(height_ratios=[1.35, 1, 1]))
ax = axes[0]
cols = [RED if c == 'baseline' else BLUE for c in pr.config]
ax.barh(range(len(pr)), pr.test, xerr=pr.test_std, color=cols, height=0.65, error_kw=dict(lw=0.7, capsize=1.5))
ax.set_yticks(range(len(pr))); ax.set_yticklabels(pr.config); ax.invert_yaxis()
ax.axvline(pr[pr.config == 'baseline'].test.iloc[0], color=GREY, ls='--', lw=0.8)
ax.set_xlabel('Test RMSE (single-factor change from baseline)'); ax.grid(axis='y', visible=False)
ax = axes[1]
for tag, col, mk, lab in [('bo-init', GREY, 'o', 'BO initial design'), ('bo', BLUE, 'o', 'BO'), ('pso', ORANGE, '^', 'PSO')]:
    d = L[L.tag == tag]; ax.scatter(d.dist, d.obj, s=12, color=col, marker=mk, label=lab, zorder=3)
b = L[L.tag == 'baseline']; ax.scatter(b.dist, b.obj, marker='*', s=70, color=RED, zorder=4, label='baseline')
ax.set_xlabel('Normalised distance from baseline'); ax.set_ylabel('Objective (validation)'); ax.legend(loc='upper left', ncol=2)
ax = axes[2]
# distance of the best SEARCHED point (the baseline, which each search kept in memory as a reference, is excluded)
init = L[L.tag == 'bo-init']; bo = L[L.tag == 'bo'].reset_index(drop=True)
bo_d = []
for it in range(len(bo) + 1):
    pool = pd.concat([init, bo.iloc[:it]], ignore_index=True); bo_d.append(pool.loc[pool.obj.idxmin(), 'dist'])
ph = pd.DataFrame(J['pso_hist'])
init_best_d = init.loc[init.obj.idxmin(), 'dist']
ax.plot(range(len(bo_d)), bo_d, 'o-', ms=2.5, color=BLUE, label='BO, best point')
ax.plot([0] + list(ph.it), [1.25] + list(ph.mean_dist), 's-', ms=2.5, color=ORANGE, label='PSO, swarm mean')
ax.plot([0] + list(ph.it), [init_best_d] + list(ph.dist_of_best), '^-', ms=2.5, color=GREEN, label='PSO, best point')
ax.axhline(1.14, color=GREY, ls=':', lw=0.8, label='initial points, all ≥ 1.14')
ax.set_xlabel('Iteration (0 = initial design)'); ax.set_ylabel('Distance to baseline'); ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.32), ncol=2); ax.set_ylim(0, 1.5)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
for a, l in zip(axes, 'abc'): a.text(-0.3 if l == 'a' else -0.16, 1.02, '(%s)' % l, transform=a.transAxes, fontsize=8, fontweight='bold')
fig.tight_layout(h_pad=0.6); save(fig, 'fig22_fd003_landscape')

# Fig. 23: reproduction of the published sweeps, five seeds (peng_sweeps_fd003.csv)
S = pd.read_csv(P('peng_sweeps_fd003.csv')); peng = dict(epoch=60, batch=512, dropout=0.2)
fig, axes = plt.subplots(3, 1, figsize=(W, 5.0))
for ax, (prm, lab) in zip(axes, [('batch', 'Batch size'), ('epoch', 'Epochs'), ('dropout', 'Dropout')]):
    d = S[S.param == prm].sort_values('value'); x = np.arange(len(d))
    ax.errorbar(x - 0.06, d.test_mean, d.test_sd, fmt='o-', ms=3, capsize=2, color=BLUE, label='test (5 seeds)')
    ax.errorbar(x + 0.06, d.val_mean, d.val_sd, fmt='s-', ms=3, capsize=2, color=PURPLE, label='validation (5 seeds)')
    k = list(d.value).index(peng[prm]); ax.axvline(k, color=RED, ls='--', lw=0.9, label='published choice')
    ax.set_xticks(x); ax.set_xticklabels([('%g' % v) for v in d.value]); ax.set_xlabel(lab); ax.set_ylabel('RMSE')
axes[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.08), ncol=3, fontsize=7)
for a, l in zip(axes, 'abc'): a.text(-0.16, 1.02, '(%s)' % l, transform=a.transAxes, fontsize=8, fontweight='bold')
fig.tight_layout(h_pad=0.6); save(fig, 'fig23_fd003_peng_sweeps')

# Fig. 24: screening gains against fresh seeds (ncmapss_v12_screening.csv, ncmapss_v12_confirm.csv)
SC = pd.read_csv(P('ncmapss_v12_screening.csv')); SC = SC[SC.val < 900]
CF = pd.read_csv(P('ncmapss_v12_confirm.csv')); inc = CF[CF.name == 'incumbent'].iloc[0]
m = CF[CF.name != 'incumbent'].merge(SC[['label', 'd_val']], left_on='name', right_on='label', how='left')
m['d_conf'] = m.capped - inc.capped; m = m.sort_values('d_val').reset_index(drop=True)
fig, ax = plt.subplots(figsize=(W, 2.9))
lo = min(m.d_val.min(), m.d_conf.min()) - 0.35; hi = max(m.d_val.max(), m.d_conf.max()) + 0.5
ax.axhspan(0, hi, color=RED, alpha=0.06, zorder=1)
ax.plot([lo, hi], [lo, hi], '--', color=GREY, lw=0.9, label='screen = confirmation', zorder=2)
ax.scatter(m.d_val, m.d_conf, s=22, color=BLUE, zorder=4)
offs = {0: (5, 6), 1: (5, -10), 2: (5, 5), 3: (-60, 8), 4: (5, -10)}
for i, r in m.iterrows():
    ax.annotate(r['name'], (r.d_val, r.d_conf), fontsize=7, xytext=offs.get(i, (5, 5)), textcoords='offset points', zorder=5)
ax.axhline(0, color='k', lw=0.7); ax.axvline(0, color='k', lw=0.7)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel('Change in the single-seed screen (negative = better)'); ax.set_ylabel('Change on four fresh seeds')
ax.legend(loc='lower right'); save(fig, 'fig24_screen_vs_confirm')

# Fig. 25: particle swarm in the 27-dimensional space (ncmapss_v12_all_trials.csv)
TR = pd.read_csv(P('ncmapss_v12_all_trials.csv'))
Pso = TR[(TR.tag == 'pso') & (TR.val < 900)].reset_index(drop=True)
bs = np.minimum.accumulate(Pso.val.values)
fig, ax = plt.subplots(figsize=(W, 2.0))
ax.step(np.arange(1, len(bs) + 1), bs, where='post', color=BLUE, lw=1.4, label='PSO, 27 dimensions (best %.3f)' % bs[-1])
ax.axhline(3.615, color=GREY, ls='--', lw=1.0, label='best in 13 dimensions (3.615)')
ax.set_xlabel('Evaluations'); ax.set_ylabel('Best validation RMSE so far'); ax.legend(loc='upper right')
save(fig, 'fig25_pso27_convergence')
