# %% [markdown]
# # Review experiments v16 — SECOND PC (RTX 3090): Parts F, G, I, J and K only
#
# Results go to `results_v16_pc2` and are merged into the lab PC's `results_v16` with `merge_v16.py`.
#
# One notebook, resumable. Every training run is written to `results_v16/runs.jsonl` the moment it finishes,
# keyed by (part, dataset, architecture, configuration, seed). Re-running any cell skips runs that are already
# on disk, so a crash or a closed session costs at most the run that was in progress. Searches are replayed
# from the same file, so a search resumes exactly where it stopped.
#
# | Part | Review item | What it runs |
# |---|---|---|
# | 0 | reporting | environment, GPU, versions, determinism (written to `env.json`) |
# | B | B2 equal tuning | PSO, 45 evaluations + two-stage selection for the **sequential** model on DS05 |
# | A | E1 / B1 | sequential vs parallel under the final protocols, same seeds (DS05 capped + uncapped, FD003, B0005) |
# | C | E2 / B3 / B4 | architecture × pooling (2 × 2) and pooling dose p = 0, 1, 2 (DS05, FD003) |
# | D | E3 / E4 / B5 | parameter counts and FLOPs, a capacity-matched pair, single-branch controls on FD003 and B0005 |
# | E | E5 / E6 | random, BO, PSO, GA, Hyperband on one space and budget, three search seeds each |
# | F | R2 statistics | Holm correction of the 12-factor screen (from the v12 CSVs) |
# | G | battery | 30 seeds per model, trend-anchored residual, leave-one-cell-out |
# | H | FD003 L2 | L2 dose re-selected **on validation**, confirmed on 15 fresh seeds; LR/RF baselines re-run |
# | I | E11 | DS05 with physical sensors only (14 ch) and with operating conditions (18 ch) |
# | J | E7 / E8 | FD001, FD002, FD004 and N-CMAPSS DS02 |
# | K | E10 / E12 | GRU, BiLSTM, TCN, Transformer baselines; NASA score, MC-dropout intervals, RMSE by life stage, trajectories |
# | L | analysis | all tables (T4 etc.), figures F3–F9, compute cost, `summary.json` |
#
# Paths below assume the notebook sits in the `thesiswork` folder; change them if not.
#
# **Approximate cost** (from the per-trial times of v9/v12: ~2 min per DS05 run, ~1.5 min per FD run, ~15 s per
# battery run): Part E ≈ 24 GPU-h, Parts H ≈ 3.3, A ≈ 1.9, C ≈ 1.8, B ≈ 2.7 (incl. two-stage), K ≈ 1.7, J ≈ 1.3,
# G ≈ 1.3, D ≈ 1.1, I ≈ 0.6 — about 40 GPU-h in total. Part E can be run last, or with `SEARCH_SEEDS=[0]` first.
#
# **Before the full run:** set `SMOKE = True` (or environment variable `V16_SMOKE=1`) and run everything once. It uses the
# real data, 2 epochs and 2 seeds, writes to `results_v16_smoke/`, and catches any TensorFlow or path error in under an
# hour. Then set it back to False for the real run.
#
# **Order:** run Part 0, then B and H before A (A uses the sequential DS05 configuration that B selects and the FD003 L2 dose that H selects on validation). Everything else is
# independent. Set `PARTS` below to run a subset. `DRY_RUN=True` exercises the whole pipeline in seconds with
# synthetic numbers (used to test the notebook; never report those numbers).

# %%
import os, sys, json, time, math, gc, random, hashlib, platform, subprocess, warnings, glob
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd

CFG = dict(
    RESULTS_DIR = os.environ.get('V16_RESULTS_DIR', 'results_v16_pc2'),   # a second PC uses its own folder (merged later)
    NCMAPSS_DIR = 'N-Cmapps',          # folder with N-CMAPSS_DS05.h5 and N-CMAPSS_DS02-006.h5
    CMAPSS_DIR  = 'env_eweda/CMaps',   # folder with train_FD00x.txt, test_FD00x.txt, RUL_FD00x.txt
    BATTERY_DIR = 'Nasa_battery_dataset',  # folder with B0005.mat, B0006.mat, B0007.mat, B0018.mat
    V12_DIR     = 'N-Cmapps',          # folder with ncmapss_v12_all_trials.csv (Part F)
    DRY_RUN     = os.environ.get('V16_DRY') == '1' or '--dry' in sys.argv,   # set True to smoke-test the pipeline
    SEED        = 42,
    N_CONFIRM   = 10,           # seeds for confirmatory comparisons (power analysis: >= 6)
    N6          = 6,            # seeds for factorials and secondary comparisons
    N_BATT      = 30,           # seeds per battery model
    N_FD_CONF   = 15,           # fresh seeds for the FD003 L2 confirmation
    SEARCH_SEEDS= [0, 1, 2],    # independent search campaigns per method (Part E)
    BUDGET      = 45,           # evaluations per search arm
)
SMOKE = os.environ.get('V16_SMOKE') == '1'   # set True for a 20-40 min real-data test of every part (2 epochs, 2 seeds)
if SMOKE:
    CFG.update(RESULTS_DIR=os.environ.get('V16_RESULTS_DIR', 'results_v16_smoke'), N_CONFIRM=2, N6=2, N_BATT=2, N_FD_CONF=2, SEARCH_SEEDS=[0], BUDGET=9)
if CFG['DRY_RUN']:   # dry-run numbers are fake: never mix them with real results
    CFG['RESULTS_DIR'] = CFG['RESULTS_DIR'] + '_dry'
PARTS = dict(A=False, B=False, C=False, D=False, E=False, F=True, G=True, H=False, I=True, J=True, K=True, L=False)   # second PC: Parts F, G, I, J, K only
if os.environ.get('V16_PARTS'):   # e.g. V16_PARTS=F,G,I,J,K on a second PC
    _on = {p.strip().upper() for p in os.environ['V16_PARTS'].split(',') if p.strip()}
    PARTS = {k: (k in _on) for k in PARTS}
print('parts to run:', ''.join(k for k, v in PARTS.items() if v))
os.makedirs(CFG['RESULTS_DIR'], exist_ok=True)
os.makedirs(os.path.join(CFG['RESULTS_DIR'],'preds'), exist_ok=True)
os.makedirs(os.path.join(CFG['RESULTS_DIR'],'figures'), exist_ok=True)
R = lambda *p: os.path.join(CFG['RESULTS_DIR'], *p)
print('results ->', os.path.abspath(CFG['RESULTS_DIR']), '| DRY_RUN', CFG['DRY_RUN'])

# ---- locate the data folders. The paths above are tried first; if a file is not there, the notebook
# searches the notebook folder, its parent folders and DATA_ROOT (env var V16_DATA_ROOT) for it.
# To skip the search, put the full path in CFG above, e.g. NCMAPSS_DIR = r'D:\\data\\N-Cmapps'.
DATA_ROOT = os.environ.get('V16_DATA_ROOT', '')
_PROBE = dict(NCMAPSS_DIR='N-CMAPSS_DS05.h5', CMAPSS_DIR='train_FD003.txt',
              BATTERY_DIR='B0005.mat', V12_DIR='ncmapss_v12_all_trials.csv')
_SKIP = {'.git', '__pycache__', 'node_modules', 'AppData', 'anaconda3', 'miniconda3', 'envs', 'site-packages',
         '$Recycle.Bin', 'Windows', 'Program Files', 'Program Files (x86)', 'results_v16', 'results_v16_smoke'}
def _search(root, fname, depth=5):
    root = os.path.abspath(root)
    for cur, dirs, files in os.walk(root):
        if cur[len(root):].count(os.sep) >= depth: dirs[:] = []; continue
        dirs[:] = sorted(d for d in dirs if d not in _SKIP and not d.startswith('.'))
        if fname in files: return cur
    return None
def locate_data():
    here = os.path.abspath(os.getcwd()); ups = [here]
    for _ in range(3): ups.append(os.path.dirname(ups[-1]))
    for key, fname in _PROBE.items():
        given = CFG[key]
        cands = [given] + ([os.path.join(DATA_ROOT, given), DATA_ROOT] if DATA_ROOT else []) + \
                [os.path.join(u, given) for u in ups]
        hit = next((c for c in cands if c and os.path.isfile(os.path.join(c, fname))), None)
        if hit is None:
            for r in ([DATA_ROOT] if DATA_ROOT else []) + ups:
                hit = _search(r, fname)
                if hit: break
        if hit: CFG[key] = os.path.abspath(hit); print('%-12s -> %s' % (key, CFG[key]))
        else: print('%-12s -> NOT FOUND (%s). Set CFG[%r] to the folder that holds it.' % (key, fname, key))
locate_data()
def need(key, fname=None):
    f = os.path.join(CFG[key], fname or _PROBE[key])
    if not DRY_FLAG and not os.path.isfile(f):
        raise FileNotFoundError('%s not found in %r. Set CFG[%r] (or env var V16_DATA_ROOT) to the folder that '
                                'contains it.' % (fname or _PROBE[key], CFG[key], key))
DRY_FLAG = CFG['DRY_RUN']

# %% [markdown]
# ## Part 0 — environment, determinism and the resumable store

# %%
DRY = CFG['DRY_RUN']
if not DRY:
    os.environ['PYTHONHASHSEED'] = str(CFG['SEED'])
    os.environ['TF_DETERMINISTIC_OPS'] = '1'; os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    import tensorflow as tf
    for _g in tf.config.list_physical_devices('GPU'):
        try: tf.config.experimental.set_memory_growth(_g, True)
        except RuntimeError as e: print('memory growth:', e)
    try: tf.config.experimental.enable_tensor_float_32_execution(False)   # full FP32 on Ampere GPUs, as on the lab PC
    except Exception as e: print('TF32 setting:', e)
    try: tf.config.experimental.enable_op_determinism(); DETERMINISM = True
    except Exception as e: print('determinism unavailable:', e); DETERMINISM = False
    from tensorflow.keras import backend as K, regularizers
    from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, Dropout, Dense, LSTM, GRU, Bidirectional,
                                         Concatenate, GlobalAveragePooling1D, Flatten, LayerNormalization,
                                         MultiHeadAttention, Add)
    from tensorflow.keras.models import Model
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    DEVICE = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
else:
    DETERMINISM = None; DEVICE = 'dry-run'

def env_report():
    info = dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), python=sys.version.split()[0], platform=platform.platform(),
                numpy=np.__version__, pandas=pd.__version__, device=DEVICE, determinism=DETERMINISM)
    try:
        import sklearn, scipy; info['sklearn'] = sklearn.__version__; info['scipy'] = scipy.__version__
    except Exception: pass
    if not DRY:
        info['tensorflow'] = tf.__version__
        try:
            bi = tf.sysconfig.get_build_info(); info['cuda'] = bi.get('cuda_version'); info['cudnn'] = bi.get('cudnn_version')
        except Exception: pass
        try:
            info['gpu'] = [tf.config.experimental.get_device_details(g).get('device_name', str(g))
                           for g in tf.config.list_physical_devices('GPU')]
        except Exception: pass
    try:
        info['nvidia_smi'] = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                             '--format=csv,noheader'], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception: info['nvidia_smi'] = 'not available'
    json.dump(info, open(R('env.json'), 'w'), indent=2); return info
print(json.dumps(env_report(), indent=1))

class Store:
    """Append-only JSONL store. A record is written and fsync'ed as soon as a run finishes."""
    def __init__(self, path):
        self.path = path; self.rows = {}
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    try: r = json.loads(line); self.rows[r['key']] = r
                    except Exception: pass   # a half-written last line after a crash is ignored
            with open(path, 'rb+') as f:          # a crash can leave a partial last line: start a fresh line
                f.seek(0, 2)
                if f.tell() > 0:
                    f.seek(-1, 2)
                    if f.read(1) != b'\n': f.write(b'\n')
        print('store: %d runs already on disk (%s)' % (len(self.rows), path))
    @staticmethod
    def key(**k): return json.dumps(k, sort_keys=True, default=str)
    def has(self, k): return k in self.rows
    def get(self, k): return self.rows[k]
    def put(self, k, rec):
        rec = dict(rec); rec['key'] = k; self.rows[k] = rec
        with open(self.path, 'a') as f:
            f.write(json.dumps(rec, default=float) + '\n'); f.flush(); os.fsync(f.fileno())
    def frame(self, **filt):
        rows = [r for r in self.rows.values() if all(r.get(a) == b for a, b in filt.items())]
        return pd.DataFrame(rows)
STORE = Store(R('runs.jsonl'))

def set_seeds(s):
    random.seed(s); np.random.seed(s)
    if not DRY: tf.keras.utils.set_random_seed(s)

def run_cached(part, dataset, arch, kw, seed, fn, **extra):
    """Run fn() once per key; return the stored record afterwards."""
    k = Store.key(part=part, dataset=dataset, arch=arch, kw=kw, seed=seed, **extra)
    if STORE.has(k): return STORE.get(k)
    t0 = time.time(); rec = fn()
    rec.update(part=part, dataset=dataset, arch=arch, seed=seed, kw=kw, secs=round(time.time() - t0, 1), **extra)
    rec.setdefault('host', platform.node())
    STORE.put(k, rec); return rec

# %% [markdown]
# ## Data loaders (same protocols as the paper)
# * **N-CMAPSS**: official split, 1-in-50 subsampling, validation units held out **by unit**, min–max scaling fitted on
#   training units, windows of T consecutive subsampled rows within one flight, RUL capped at 55 or uncapped.
# * **C-MAPSS FDxxx**: 3 settings + 21 sensors + cycle index, min–max on training engines (per operating condition for
#   FD002/FD004, conditions found by k-means on the settings), RUL cap 125, W = 50, 20% of training engines held out by
#   engine, test = last window of each test engine.
# * **Battery**: discharge capacities, SoH = C/C0, recursive forecast from 80% of life (cycle 134 for B0005), W = 10.

# %%
NCM_VAL_UNITS = {'DS05': [5, 6], 'DS02': [18, 20]}
NCM_FILES = {'DS05': 'N-CMAPSS_DS05.h5', 'DS02': 'N-CMAPSS_DS02-006.h5'}
_NCM_RAW = {}; _NCM_WIN = {}
def _read_sub(dset, step, chunk=400_000):
    n = dset.shape[0]; out = []
    for a in range(0, n, chunk):
        b = min(a + chunk, n); blk = dset[a:b]; first = (-a) % step
        if first < len(blk): out.append(blk[first::step].astype(np.float32, copy=False))
    return np.concatenate(out)
def ncm_raw(name, sub=50):
    if (name, sub) in _NCM_RAW: return _NCM_RAW[(name, sub)]
    import h5py
    need('NCMAPSS_DIR', NCM_FILES[name])
    with h5py.File(os.path.join(CFG['NCMAPSS_DIR'], NCM_FILES[name]), 'r') as h:
        d = {k: _read_sub(h[k], sub) for k in ['X_s_dev', 'X_v_dev', 'W_dev', 'A_dev', 'Y_dev',
                                              'X_s_test', 'X_v_test', 'W_test', 'A_test', 'Y_test']}
    _NCM_RAW[(name, sub)] = d; return d
def _windows(F, u, c, y, mask, T):
    uu, cc, FF, yy = u[mask], c[mask], F[mask], y[mask]
    key = uu.astype(np.int64) * 100000 + cc; o = np.argsort(key, kind='stable')
    FF, yy, key, uu = FF[o], yy[o], key[o], uu[o]; bnd = np.flatnonzero(np.diff(key)) + 1
    X, Y, U = [], [], []
    for a, b in zip(np.r_[0, bnd], np.r_[bnd, len(key)]):
        A = FF[a:b]; L = yy[a:b]
        for e in range(T, len(A) + 1): X.append(A[e - T:e]); Y.append(L[e - 1]); U.append(uu[a])
    return np.asarray(X, np.float32), np.asarray(Y, np.float32), np.asarray(U)
def ncm_data(name='DS05', T=30, channels='xs_xv', sub=50):
    """channels: 'xs_xv' (28, final), 'xs' (14 physical only), 'w_xs' (4 conditions + 14 physical)."""
    k = (name, T, channels, sub)
    if k in _NCM_WIN: return _NCM_WIN[k]
    if DRY:
        nf = {'xs_xv': 28, 'xs': 14, 'w_xs': 18}[channels]; rs = np.random.RandomState(0)
        d = dict(Xtr=rs.rand(64, T, nf).astype('f4'), ytr=rs.rand(64).astype('f4') * 80, Xva=rs.rand(32, T, nf).astype('f4'),
                 yva=rs.rand(32).astype('f4') * 80, Xte=rs.rand(48, T, nf).astype('f4'), yte=rs.rand(48).astype('f4') * 80,
                 ute=np.repeat([7, 8, 9, 10], 12), shape=(T, nf))
        _NCM_WIN[k] = d; return d
    r = ncm_raw(name, sub)
    pick = {'xs_xv': lambda s: np.hstack([r['X_s_' + s], r['X_v_' + s]]), 'xs': lambda s: r['X_s_' + s],
            'w_xs': lambda s: np.hstack([r['W_' + s], r['X_s_' + s]])}[channels]
    Fd, Ft = pick('dev'), pick('test')
    ud, cd, yd = r['A_dev'][:, 0].astype(int), r['A_dev'][:, 1].astype(int), r['Y_dev'].ravel()
    ut, ct, yt = r['A_test'][:, 0].astype(int), r['A_test'][:, 1].astype(int), r['Y_test'].ravel()
    vu = NCM_VAL_UNITS[name]; tr = ~np.isin(ud, vu); va = np.isin(ud, vu)
    lo, hi = Fd[tr].min(0), Fd[tr].max(0); sc = lambda F: ((F - lo) / np.where(hi > lo, hi - lo, 1)).astype(np.float32)
    Xtr, ytr, _ = _windows(sc(Fd), ud, cd, yd, tr, T); Xva, yva, _ = _windows(sc(Fd), ud, cd, yd, va, T)
    Xte, yte, ute = _windows(sc(Ft), ut, ct, yt, np.ones(len(yt), bool), T)
    assert not set(np.unique(ud[tr])) & set(vu), 'validation units leaked into training'
    d = dict(Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, yte=yte, ute=ute, shape=Xtr.shape[1:])
    _NCM_WIN[k] = d; print('%s T=%d %s: train %d val %d test %d windows' % (name, T, channels, len(ytr), len(yva), len(yte)))
    return d

_CM = {}
def cmapss_data(fd='FD003', W=50, cap=125, seed=42):
    if (fd, W) in _CM: return _CM[(fd, W)]
    if DRY:
        rs = np.random.RandomState(1)
        d = dict(Xtr=rs.rand(64, W, 25).astype('f4'), ytr=rs.rand(64, 1).astype('f4') * 125, Xva=rs.rand(16, W, 25).astype('f4'),
                 yva=rs.rand(16, 1).astype('f4') * 125, Xte=rs.rand(100, W, 25).astype('f4'), yte=rs.rand(100, 1).astype('f4') * 125,
                 shape=(W, 25), triv=42.0)
        _CM[(fd, W)] = d; return d
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.cluster import KMeans
    idx = ['unit_nr', 'time_cycles']; setn = ['setting_1', 'setting_2', 'setting_3']; sn = ['s_%d' % i for i in range(1, 22)]
    cols = idx + setn + sn; P = lambda f: (need('CMAPSS_DIR', f), os.path.join(CFG['CMAPSS_DIR'], f))[1]
    train = pd.read_csv(P('train_%s.txt' % fd), sep=r'\s+', header=None, names=cols)
    test = pd.read_csv(P('test_%s.txt' % fd), sep=r'\s+', header=None, names=cols)
    y_test = pd.read_csv(P('RUL_%s.txt' % fd), sep=r'\s+', header=None, names=['RUL'])
    g = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=seed); ti, vi = next(g.split(train, groups=train.unit_nr))
    TRU, VAU = np.sort(train.iloc[ti].unit_nr.unique()), np.sort(train.iloc[vi].unit_nr.unique())
    feats = setn + sn + ['time_cycles']
    tr = train.copy(); mx = tr.groupby('unit_nr')['time_cycles'].transform('max')
    tr['RUL'] = (mx - tr['time_cycles']).clip(upper=cap); te = test.copy()
    multi = fd in ('FD002', 'FD004')
    if multi:   # condition-wise scaling: 6 operating regimes found on the training engines only
        km = KMeans(6, n_init=10, random_state=seed).fit(tr[tr.unit_nr.isin(TRU)][setn])
        tr['cond'] = km.predict(tr[setn]); te['cond'] = km.predict(te[setn])
        for c in range(6):
            base = tr[(tr.unit_nr.isin(TRU)) & (tr.cond == c)][sn + ['time_cycles']]
            lo, hi = base.min(), base.max(); rng = (hi - lo).replace(0, 1)
            for D in (tr, te):
                m = D.cond == c; D.loc[m, sn + ['time_cycles']] = ((D.loc[m, sn + ['time_cycles']] - lo) / rng).values
        lo, hi = tr[tr.unit_nr.isin(TRU)][setn].min(), tr[tr.unit_nr.isin(TRU)][setn].max()
        for D in (tr, te): D[setn] = ((D[setn] - lo) / (hi - lo).replace(0, 1)).values
    else:
        lo, hi = tr[tr.unit_nr.isin(TRU)][feats].min(), tr[tr.unit_nr.isin(TRU)][feats].max()
        for D in (tr, te): D[feats] = ((D[feats] - lo) / (hi - lo).replace(0, 1)).values
    tr[feats] = np.nan_to_num(tr[feats].values); te[feats] = np.nan_to_num(te[feats].values)
    def win(df, us):
        X, y = [], []
        for u in us:
            d = df[df.unit_nr == u]; A = d[feats].to_numpy(np.float32); L = d['RUL'].to_numpy(np.float32)
            for e in range(W, len(d) + 1): X.append(A[e - W:e]); y.append(L[e - 1])
        return np.asarray(X, np.float32), np.asarray(y, np.float32).reshape(-1, 1)
    Xtr, ytr = win(tr, TRU); Xva, yva = win(tr, VAU); Xte = []
    for u in te.unit_nr.unique():
        A = te[te.unit_nr == u][feats].to_numpy(np.float32)
        Xte.append(A[-W:] if len(A) >= W else np.vstack([np.repeat(A[:1], W - len(A), axis=0), A]))
    Xte = np.asarray(Xte, np.float32); yte = np.clip(y_test['RUL'].to_numpy(np.float32), 0, cap).reshape(-1, 1)
    triv = float(np.sqrt(np.mean((yte - ytr.mean()) ** 2)))
    d = dict(Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, yte=yte, shape=Xtr.shape[1:], triv=triv)
    _CM[(fd, W)] = d; print('%s: train %d val %d test %d | trivial %.3f' % (fd, len(ytr), len(yva), len(yte), triv)); return d

_BAT = {}
def battery_soh(cell):
    if cell in _BAT: return _BAT[cell]
    if DRY:
        S = (1 - np.linspace(0, 0.3, 168) + 0.005 * np.random.RandomState(2).randn(168)).astype('f4'); _BAT[cell] = S; return S
    from scipy.io import loadmat
    need('BATTERY_DIR', cell + '.mat'); mat = loadmat(os.path.join(CFG['BATTERY_DIR'], cell + '.mat')); cyc = mat[cell][0, 0]['cycle'][0]
    caps = [float(r['data'][0][0]['Capacity'][0][0]) for r in cyc if r['type'][0] == 'discharge']
    C = np.asarray(caps, np.float64); S = (C / C[0]).astype(np.float32); _BAT[cell] = S; return S

# %% [markdown]
# ## Models
# `par` = parallel (as in the paper), `seq` = sequential (convolution stack feeding the LSTM), `cnn` / `lstm` =
# single-branch controls, plus four external baselines. `n_pool` sets the number of pooling stages explicitly
# (T_c = W / 2^p). Parameter counts are taken from Keras; FLOPs are counted analytically (multiply–adds × 2).

# %%
def _pool_stages(kw):
    if 'n_pool' in kw: return int(kw['n_pool'])
    return min(2, kw['n_conv']) if kw.get('use_pool', 0) else 0

def build_ncm(arch, kw, shape, seed):
    reg = regularizers.l2(kw['l2reg']) if kw.get('l2reg', 0) > 1e-6 else None
    kk = dict(kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed), kernel_regularizer=reg)
    inp = Input(shape=shape); dr = kw.get('dropout', 0.0); p = _pool_stages(kw)
    def conv_stack(x):
        for i in range(kw['n_conv']):
            f = max(8, int(round(kw['f1'] * (kw['f_growth'] ** i))))
            x = Conv1D(f, kw['kernel'], padding='same', activation='relu', **kk)(x)
            if dr > 0: x = Dropout(dr, seed=seed)(x)
            if i < p and int(x.shape[1]) > 2: x = MaxPooling1D(2, padding='same')(x)
        return x
    def lstm_stack(x):
        for j in range(kw['n_lstm']):
            x = LSTM(kw['lstm_units'], return_sequences=(j < kw['n_lstm'] - 1), **kk)(x)
        return x
    if arch == 'par':   h = Concatenate()([GlobalAveragePooling1D()(conv_stack(inp)), lstm_stack(inp)])
    elif arch == 'seq': h = lstm_stack(conv_stack(inp))
    elif arch == 'cnn': h = GlobalAveragePooling1D()(conv_stack(inp))
    elif arch == 'lstm':h = lstm_stack(inp)
    elif arch == 'gru': h = GRU(64, return_sequences=True, **kk)(inp); h = GRU(64, **kk)(h)
    elif arch == 'bilstm': h = Bidirectional(LSTM(64, return_sequences=True, **kk))(inp); h = Bidirectional(LSTM(64, **kk))(h)
    elif arch == 'tcn':
        h = inp
        for d in (1, 2, 4, 8):
            y = Conv1D(64, 3, padding='causal', dilation_rate=d, activation='relu', **kk)(h)
            y = Conv1D(64, 3, padding='causal', dilation_rate=d, activation='relu', **kk)(y)
            h = Add()([y, Conv1D(64, 1, **kk)(h) if int(h.shape[-1]) != 64 else h])
        h = GlobalAveragePooling1D()(h)
    elif arch == 'transformer':
        h = Dense(64, **kk)(inp)
        pos = tf.range(start=0, limit=shape[0], delta=1)
        h = h + tf.keras.layers.Embedding(shape[0], 64)(pos)
        for _ in range(2):
            a = MultiHeadAttention(4, 16, dropout=0.1)(h, h); h = LayerNormalization()(h + a)
            f = Dense(128, activation='relu', **kk)(h); f = Dense(64, **kk)(f); h = LayerNormalization()(h + f)
        h = GlobalAveragePooling1D()(h)
    else: raise ValueError(arch)
    if kw.get('dense1', 0) >= 8: h = Dense(kw['dense1'], activation='relu', **kk)(h)
    if kw.get('dense2', 0) >= 16: h = Dense(kw['dense2'], activation='relu', **kk)(h)
    m = Model(inp, Dense(1, activation='linear', **kk)(h))
    m.compile(optimizer=tf.keras.optimizers.Adam(kw.get('lr', 1e-3)), loss='mse'); return m

def build_fd(arch, shape, seed, l2=0.3, filters=128, lstm_units=128, dense=128, merged=256, dropout=0.2, lr=1e-3, n_pool=1, n_conv=None):
    kw = dict(kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)); L = regularizers.l2(l2)
    inp = Input(shape=shape); n_conv = n_conv or max(1, n_pool)
    def conv(x):
        for i in range(n_conv):
            x = Conv1D(filters, 3, padding='same', activation='relu', **kw)(x)
            if i < n_pool: x = MaxPooling1D(2, padding='same')(x)
        return x
    def head(x): x = Dense(dense, activation='relu', kernel_regularizer=L, **kw)(x); return Dropout(dropout, seed=seed)(x)
    if arch == 'par':
        c = head(Flatten()(conv(inp))); r = LSTM(lstm_units, return_sequences=True, **kw)(inp); r = head(LSTM(lstm_units, **kw)(r))
        h = Concatenate()([c, r])
    elif arch == 'seq':
        r = LSTM(lstm_units, return_sequences=True, **kw)(conv(inp)); h = head(LSTM(lstm_units, **kw)(r))
    elif arch == 'cnn': h = head(Flatten()(conv(inp)))
    elif arch == 'lstm': r = LSTM(lstm_units, return_sequences=True, **kw)(inp); h = head(LSTM(lstm_units, **kw)(r))
    elif arch in ('gru', 'bilstm', 'tcn', 'transformer'):
        return build_ncm(arch, dict(n_conv=1, f1=64, f_growth=1, kernel=3, n_lstm=1, lstm_units=64, dense1=128, dense2=0, lr=lr), shape, seed)
    else: raise ValueError(arch)
    h = Dense(merged, activation='relu', kernel_regularizer=L, **kw)(h); h = Dropout(dropout, seed=seed)(h)
    m = Model(inp, Dense(1, activation='linear', **kw)(h)); m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse'); return m

def build_bat(arch, seed, window=10, filters=32, ms=(3, 5, 7), lstm_units=32, dense=64, dropout=0.1, lr=1e-3, n_pool=1):
    kw = dict(kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)); inp = Input(shape=(window, 1))
    def conv(x):
        x = Concatenate()([Conv1D(filters, k, padding='same', activation='relu', **kw)(x) for k in ms])
        x = Dropout(dropout, seed=seed)(x)
        if n_pool and int(x.shape[1]) > 2: x = MaxPooling1D(2, padding='same')(x)
        return x
    if arch == 'par':   h = Concatenate()([GlobalAveragePooling1D()(conv(inp)), Bidirectional(LSTM(lstm_units, **kw))(inp)])
    elif arch == 'seq': h = Bidirectional(LSTM(lstm_units, **kw))(conv(inp))
    elif arch == 'cnn': h = GlobalAveragePooling1D()(conv(inp))
    elif arch == 'lstm':h = Bidirectional(LSTM(lstm_units, **kw))(inp)
    else: raise ValueError(arch)
    h = Dense(dense, activation='relu', **kw)(h); h = Dropout(dropout, seed=seed)(h)
    m = Model(inp, Dense(1, activation='linear', **kw)(h)); m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse'); return m

def analytic_cost(arch, kw, shape):
    """parameters and FLOPs (2 x multiply-adds) of the N-CMAPSS models, independent of TensorFlow"""
    T, C = shape; P = 0; F_ = 0; p = _pool_stages(kw); t = T; cin = C
    convs = []
    for i in range(kw['n_conv']):
        f = max(8, int(round(kw['f1'] * (kw['f_growth'] ** i)))); convs.append((t, cin, f))
        P += (kw['kernel'] * cin + 1) * f; F_ += 2 * kw['kernel'] * cin * f * t; cin = f
        if i < p and t > 2: t = int(math.ceil(t / 2))
    conv_out, conv_t = cin, t
    def lstm(Tin, Fin):
        pp = 0; ff = 0; fin = Fin
        for j in range(kw['n_lstm']):
            U = kw['lstm_units']; pp += 4 * U * (fin + U + 1); ff += 2 * 4 * U * (fin + U) * Tin; fin = U
        return pp, ff
    if arch in ('par', 'cnn'):
        if arch == 'cnn': feat = conv_out
        else:
            lp, lf = lstm(T, C); P += lp; F_ += lf; feat = conv_out + kw['lstm_units']
    elif arch == 'seq':
        lp, lf = lstm(conv_t, conv_out); P += lp; F_ += lf; feat = kw['lstm_units']
    elif arch == 'lstm':
        P = 0; F_ = 0; lp, lf = lstm(T, C); P += lp; F_ += lf; feat = kw['lstm_units']
    else: return None, None
    for n in (kw.get('dense1', 0) if kw.get('dense1', 0) >= 8 else 0, kw.get('dense2', 0) if kw.get('dense2', 0) >= 16 else 0):
        if n: P += (feat + 1) * n; F_ += 2 * feat * n; feat = n
    P += feat + 1; F_ += 2 * feat
    return int(P), int(F_)

# %% [markdown]
# ## Training and evaluation functions (one per dataset family)

# %%
EP_NCM, PAT_NCM = 100, 15
EP_FD, PAT_FD = 120, 20
EP_BAT, PAT_BAT, BATCH_BAT = 300, 30, 16
MC_PASSES = 30
if SMOKE: EP_NCM, EP_FD, EP_BAT, MC_PASSES = 2, 2, 3, 3

def nasa_score(y, yhat):
    e = np.asarray(yhat).ravel() - np.asarray(y).ravel()
    return float(np.sum(np.where(e < 0, np.exp(-e / 13) - 1, np.exp(e / 10) - 1)))
def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a).ravel() - np.asarray(b).ravel()) ** 2)))

def _dry(tag, seed, base, sd):
    h = int(hashlib.md5((tag + str(seed)).encode()).hexdigest()[:8], 16) / 0xffffffff
    return float(base + sd * (h - 0.5) * 3.4)

DRY_BASE = {('DS05', 'par'): 4.86, ('DS05', 'seq'): 5.3, ('DS05', 'cnn'): 5.0, ('DS05', 'lstm'): 5.3, ('DS02', 'par'): 5.5, ('DS02', 'seq'): 6.0,
            ('FD003', 'par'): 10.4, ('FD003', 'seq'): 11.3, ('FD001', 'par'): 12.5, ('FD001', 'seq'): 13.0, ('FD002', 'par'): 15.0,
            ('FD002', 'seq'): 16.0, ('FD004', 'par'): 17.0, ('FD004', 'seq'): 18.5}

def eval_ncm(dataset, arch, kw, seed, cap=55, T=30, channels='xs_xv', epochs=None, save_pred=False, mc=False, tag=''):
    epochs = epochs or EP_NCM
    d = ncm_data(dataset, T=T, channels=channels)
    if DRY:
        b = DRY_BASE.get((dataset, arch), 6.0) * (1 if cap else 1.9); v = _dry(dataset + arch + json.dumps(kw, sort_keys=True), seed, b - 1.1, 0.25)
        P_, F_ = analytic_cost(arch, kw, d['shape']) if arch in ('par', 'seq', 'cnn', 'lstm') else (100000, 1000000)
        return dict(val=v, test=_dry('t' + dataset + arch + json.dumps(kw, sort_keys=True), seed, b, 0.25),
                    test_raw=_dry('r' + dataset + arch, seed, b * 2.0, 0.3), nasa=1000.0, mae=b * 0.8, params=P_, flops=F_, epochs=epochs)
    K.clear_session(); gc.collect(); set_seeds(seed)
    ytr = np.minimum(d['ytr'], cap) if cap else d['ytr']; yva = np.minimum(d['yva'], cap) if cap else d['yva']
    yte = np.minimum(d['yte'], cap) if cap else d['yte']
    with tf.device(DEVICE):
        m = build_ncm(arch, kw, d['shape'], seed)
        h = m.fit(d['Xtr'], ytr, validation_data=(d['Xva'], yva), epochs=epochs, batch_size=kw.get('batch', 256), verbose=0,
                  callbacks=[EarlyStopping('val_loss', patience=PAT_NCM, restore_best_weights=True),
                             ReduceLROnPlateau('val_loss', factor=0.5, patience=PAT_NCM // 2, min_lr=1e-6)])
        pv = m.predict(d['Xva'], batch_size=8192, verbose=0).ravel(); pt = m.predict(d['Xte'], batch_size=8192, verbose=0).ravel()
    if cap: pt = np.clip(pt, 0, cap); pv = np.clip(pv, 0, cap)
    out = dict(val=rmse(yva, pv), test=rmse(yte, pt), test_raw=rmse(d['yte'], pt), nasa=nasa_score(yte, pt), mae=float(np.mean(np.abs(yte - pt))),
               params=int(m.count_params()), flops=analytic_cost(arch, kw, d['shape'])[1] if arch in ('par', 'seq', 'cnn', 'lstm') else None,
               epochs=len(h.history['loss']))
    if mc: out.update(mc_dropout(m, d['Xte'], yte, cap))
    if save_pred:
        fn = R('preds', '%s_%s_%s_s%d%s.npz' % (dataset, arch, 'cap' if cap else 'raw', seed, tag))
        np.savez_compressed(fn, y=yte, yhat=pt, unit=d['ute']); out['pred_file'] = fn
    return out

def mc_dropout(m, X, y, cap=None, passes=MC_PASSES):
    """MC-dropout: dropout kept active at inference; 95% interval from the sample quantiles."""
    P = np.stack([np.concatenate([m(X[i:i + 8192], training=True).numpy().ravel() for i in range(0, len(X), 8192)])
                  for _ in range(passes)])
    if cap: P = np.clip(P, 0, cap)
    lo, hi = np.percentile(P, 2.5, axis=0), np.percentile(P, 97.5, axis=0); yy = np.asarray(y).ravel()
    return dict(picp=float(np.mean((yy >= lo) & (yy <= hi))), mpiw=float(np.mean(hi - lo)), mc_rmse=rmse(yy, P.mean(0)))

def eval_fd(fd, arch, seed, l2=0.3, n_pool=1, n_conv=None, save_pred=False, mc=False, lstm_units=128, filters=128):
    d = cmapss_data(fd)
    if DRY:
        b = DRY_BASE.get((fd, arch), 12.0) + (0.3 if abs(l2 - 0.3) > 1e-9 else 0) + 0.2 * n_pool * (arch == 'seq')
        return dict(val=_dry('v' + fd + arch + str(l2) + str(n_pool), seed, b - 1.5, 0.35), test=_dry(fd + arch + str(l2) + str(n_pool), seed, b, 0.35),
                    nasa=900.0, mae=b * 0.8, params=700000 if arch == 'par' else 600000, triv=d['triv'], epochs=80)
    K.clear_session(); gc.collect(); set_seeds(seed)
    with tf.device(DEVICE):
        m = build_fd(arch, d['shape'], seed, l2=l2, n_pool=n_pool, n_conv=n_conv, lstm_units=lstm_units, filters=filters)
        h = m.fit(d['Xtr'], d['ytr'], validation_data=(d['Xva'], d['yva']), epochs=EP_FD, batch_size=256, verbose=0,
                  callbacks=[EarlyStopping('val_loss', patience=PAT_FD, restore_best_weights=True),
                             ReduceLROnPlateau('val_loss', factor=0.5, patience=10, min_lr=1e-6)])
        pv = m.predict(d['Xva'], verbose=0).ravel(); pt = np.clip(m.predict(d['Xte'], verbose=0).ravel(), 0, 125)
    out = dict(val=rmse(d['yva'], pv), test=rmse(d['yte'], pt), nasa=nasa_score(d['yte'], pt), mae=float(np.mean(np.abs(d['yte'].ravel() - pt))),
               params=int(m.count_params()), triv=d['triv'], epochs=len(h.history['loss']))
    if mc: out.update(mc_dropout(m, d['Xte'], d['yte'], 125))
    if save_pred:
        fn = R('preds', '%s_%s_s%d.npz' % (fd, arch, seed)); np.savez_compressed(fn, y=d['yte'], yhat=pt); out['pred_file'] = fn
    return out

def _bat_windows(series, idxs, W=10):
    X = np.asarray([series[e - W:e] for e in idxs], np.float32).reshape(-1, W, 1)
    y = np.asarray([series[e] - series[e - 1] for e in idxs], np.float32).reshape(-1, 1); return X, y
def _lin_trend(S, origin, L=20):
    k = np.arange(origin - L, origin, dtype=np.float64); b, a = np.polyfit(k, S[origin - L:origin].astype(np.float64), 1)
    return np.clip(a + b * np.arange(len(S), dtype=np.float64), 0, 1.2).astype(np.float32)
def eval_bat(cell, arch, seed, mode='trend', train_cells=None, W=10, n_pool=1, save_pred=False):
    """mode: 'net' (network on SoH deltas) or 'trend' (linear trend + network on the residual).
    train_cells: None = within-cell; list = train on those cells (pre-origin of target is used only if target in list)."""
    S = battery_soh(cell); N = len(S); START = int(0.8 * N) - 1; H = N - START
    floor = rmse(S[START:], np.full(H, S[START - 1]))
    if DRY:
        b = 0.0096 if mode == 'trend' else 0.011; b += {'par': 0, 'seq': 0.002, 'cnn': 0.001, 'lstm': 0.0015}[arch] + (0.01 if train_cells else 0)
        return dict(val=_dry('v' + cell + arch + mode + str(train_cells), seed, b * 0.6, 0.0005),
                    test=_dry(cell + arch + mode + str(train_cells), seed, b, 0.002), floor=floor, trend_only=0.0144)
    def series_of(c):
        Sc = battery_soh(c); st = int(0.8 * len(Sc)) - 1
        if mode == 'trend':
            cur = _lin_trend(Sc, st); return (Sc - cur).astype(np.float32), cur, st
        return Sc, None, st
    tgt, curve, st = series_of(cell)
    cells = train_cells if train_cells else [cell]
    Xs, ys = [], []
    for c in cells:
        sc, _, stc = series_of(c); idx = np.arange(W, stc if c == cell else len(sc))
        X_, y_ = _bat_windows(sc, idx, W); Xs.append(X_); ys.append(y_)
    Xa, ya = np.concatenate(Xs), np.concatenate(ys); rs = np.random.RandomState(seed)
    if train_cells and cell not in train_cells: o = rs.permutation(len(Xa)); Xa, ya = Xa[o], ya[o]
    c_ = int(0.85 * len(Xa))
    K.clear_session(); gc.collect(); set_seeds(seed)
    m = build_bat(arch, seed, window=W, n_pool=n_pool)
    m.fit(Xa[:c_], ya[:c_], validation_data=(Xa[c_:], ya[c_:]), epochs=EP_BAT, batch_size=BATCH_BAT, verbose=0, shuffle=True,
          callbacks=[EarlyStopping('val_loss', patience=PAT_BAT, restore_best_weights=True),
                     ReduceLROnPlateau('val_loss', factor=0.5, patience=10, min_lr=1e-7)])
    val = rmse(ya[c_:], m.predict(Xa[c_:], verbose=0).ravel())
    win = list(tgt[START - W:START]); out = []; prev = float(tgt[START - 1])
    for _ in range(H):
        x = np.asarray(win[-W:], np.float32).reshape(1, W, 1); prev = float(prev + float(m(x, training=False).numpy().ravel()[0]))
        if mode == 'net': prev = float(np.clip(prev, 0, 1.2))
        out.append(prev); win.append(prev)
    pred = np.clip((curve[START:] if mode == 'trend' else 0) + np.asarray(out, np.float32), 0, 1.2)
    res = dict(val=val, test=rmse(S[START:], pred), floor=floor, params=int(m.count_params()),
               trend_only=rmse(S[START:], _lin_trend(S, START)[START:]))
    if save_pred:
        fn = R('preds', 'bat_%s_%s_%s_s%d.npz' % (cell, arch, mode, seed)); np.savez_compressed(fn, y=S[START:], yhat=pred); res['pred_file'] = fn
    return res

# %% [markdown]
# ## Final configurations and seeds

# %%
DS05_PAR = dict(n_conv=2, f1=192, f_growth=0.5, kernel=5, use_pool=0, n_lstm=1, lstm_units=88, dropout=0.3442517771420791,
                l2reg=3.831223699431197e-05, dense1=8, dense2=0, lr=0.005, batch=64)
DS05_SEQ_DEFAULT = dict(DS05_PAR, use_pool=1)      # replaced by the Part B selection
FD_L2_FILE = R('fd003_l2_selected.json')
SEEDS_CONF = [CFG['SEED'] + 5000 + i for i in range(CFG['N_CONFIRM'])]
SEEDS_6 = SEEDS_CONF[:CFG['N6']]
SEEDS_BAT = [CFG['SEED'] + 7000 + i for i in range(CFG['N_BATT'])]
SEEDS_FD_SEL = [CFG['SEED'] + 11000 + i for i in range(CFG['N6'])]
SEEDS_FD_CONF = [CFG['SEED'] + 13000 + i for i in range(CFG['N_FD_CONF'])]
print('confirm seeds', SEEDS_CONF)

# %% [markdown]
# ## Search machinery (13-D space of the paper, Table IV) — random, BO, PSO, GA, Hyperband
# Every evaluation is cached under (campaign, evaluation index), so a search resumes exactly.

# %%
from scipy.stats.qmc import Sobol
from scipy.stats import norm, spearmanr, ttest_rel, ttest_1samp, wilcoxon, friedmanchisquare, t as tdist
SPACE = [('n_conv', 'int', 2, 4), ('f1', 'int', 24, 192), ('f_growth', 'float', 0.5, 2.0), ('kernel', 'cat', [1, 2, 3, 5, 7]),
         ('use_pool', 'cat', [0, 1]), ('n_lstm', 'int', 1, 2), ('lstm_units', 'int', 24, 320), ('dropout', 'float', 0.0, 0.40),
         ('l2reg', 'log', 1e-7, 1e-1), ('dense1', 'int', 8, 256), ('dense2', 'int', 0, 256), ('lr', 'log', 1e-4, 5e-3),
         ('batch', 'cat', [64, 128, 256, 512, 1024])]
D = len(SPACE)
def decode(z):
    z = np.clip(np.asarray(z, float), 0, 1); out = {}
    for zi, s in zip(z, SPACE):
        n, k = s[0], s[1]
        if k == 'int': out[n] = int(round(s[2] + zi * (s[3] - s[2])))
        elif k == 'float': out[n] = float(s[2] + zi * (s[3] - s[2]))
        elif k == 'log': out[n] = float(np.exp(np.log(s[2]) + zi * (np.log(s[3]) - np.log(s[2]))))
        else: out[n] = s[2][min(int(zi * len(s[2])), len(s[2]) - 1)]
    return out
def encode(kw):
    z = []
    for s in SPACE:
        n, k = s[0], s[1]; v = kw[n]
        if k in ('int', 'float'): z.append((v - s[2]) / (s[3] - s[2]))
        elif k == 'log': z.append((np.log(v) - np.log(s[2])) / (np.log(s[3]) - np.log(s[2])))
        else: z.append((s[2].index(v) + 0.5) / len(s[2]))
    return np.clip(np.array(z, float), 0, 1)
def nrmsd(a, b): return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.sqrt(D))
Z_REF = [encode(DS05_PAR)]
def far_design(n, seed, min_d=0.35):
    eng = Sobol(d=D, scramble=True, seed=seed); pts = []; guard = 0
    while len(pts) < n and guard < 50:
        guard += 1
        for c in eng.random(max(8, n * 4)):
            if min(nrmsd(c, r) for r in Z_REF) >= min_d: pts.append(c)
            if len(pts) == n: break
    return np.array(pts[:n])

def search_eval(campaign, i, arch, z, epochs=None):
    kw = decode(z); ep = epochs or EP_NCM
    rec = run_cached('search', 'DS05', arch, kw, CFG['SEED'], lambda: eval_ncm('DS05', arch, kw, CFG['SEED'], epochs=ep),
                     campaign=campaign, idx=int(i), epochs=ep)
    return rec['val'] if np.isfinite(rec['val']) else 1e3

def run_random(campaign, arch, sseed, budget):
    Z = far_design(budget, seed=1000 + sseed); return [search_eval(campaign, i, arch, z) for i, z in enumerate(Z)]
def run_bo(campaign, arch, sseed, budget, n_init=15):
    n_init = min(n_init, max(3, budget // 3))
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
    Z = list(far_design(n_init, seed=2000 + sseed)); Y = [search_eval(campaign, i, arch, z) for i, z in enumerate(Z)]
    for it in range(budget - n_init):
        Za, Ya = np.array(Z), np.array(Y); ok = Ya < 999
        gp = GaussianProcessRegressor(ConstantKernel(1.0) * Matern(length_scale=np.ones(D), nu=2.5) + WhiteKernel(1e-2),
                                      normalize_y=True, n_restarts_optimizer=3, random_state=sseed).fit(Za[ok], Ya[ok])
        cand = np.random.RandomState(3000 + 97 * sseed + it).rand(4000, D); mu, sd = gp.predict(cand, return_std=True)
        imp = Ya[ok].min() - mu - 0.01; zz = imp / np.maximum(sd, 1e-9); ei = imp * norm.cdf(zz) + sd * norm.pdf(zz)
        zb = cand[int(np.argmax(ei))]; Z.append(zb); Y.append(search_eval(campaign, n_init + it, arch, zb))
    return Y
def run_pso(campaign, arch, sseed, budget, swarm=9):
    iters = budget // swarm; rng = np.random.RandomState(4000 + sseed); P = far_design(swarm, seed=4000 + sseed)
    V = (rng.rand(swarm, D) - 0.5) * 0.30; pb = P.copy(); pby = np.full(swarm, np.inf); g = None; gy = np.inf; Y = []; i = 0
    for it in range(iters):
        w = 0.85 - 0.50 * it / max(1, iters - 1)
        for k in range(swarm):
            y = search_eval(campaign, i, arch, P[k]); i += 1; Y.append(y)
            if y < pby[k]: pby[k] = y; pb[k] = P[k].copy()
            if y < gy: gy = y; g = P[k].copy()
        r1, r2 = rng.rand(swarm, D), rng.rand(swarm, D)
        V = np.clip(w * V + 1.5 * r1 * (pb - P) + 1.5 * r2 * (g - P), -0.25, 0.25); P = np.clip(P + V, 0, 1)
    return Y
def run_ga(campaign, arch, sseed, budget, pop=9, tour=3, pm=None, sigma=0.15):
    """generational GA on the unit cube: tournament selection, uniform crossover, Gaussian mutation, 1 elite."""
    rng = np.random.RandomState(5000 + sseed); pm = pm or 1.0 / D; P = far_design(pop, seed=5000 + sseed); i = 0; Y = []
    fit = np.array([search_eval(campaign, j, arch, P[j]) for j in range(pop)]); i = pop; Y += list(fit)
    while i < budget:
        elite = P[int(np.argmin(fit))].copy(); kids = [elite]
        while len(kids) < pop:
            def pick():
                c = rng.choice(pop, tour, replace=False); return P[c[np.argmin(fit[c])]]
            a, b = pick(), pick(); mask = rng.rand(D) < 0.5; child = np.where(mask, a, b)
            mut = rng.rand(D) < pm; child = np.clip(child + mut * rng.randn(D) * sigma, 0, 1); kids.append(child)
        P = np.array(kids); fit = np.empty(pop)
        for j in range(pop):
            if i >= budget: fit[j] = 1e3; continue
            fit[j] = search_eval(campaign, i, arch, P[j]); Y.append(fit[j]); i += 1
    return Y[:budget]
def run_hyperband(campaign, arch, sseed, budget, eta=3):
    """Hyperband with R = EP_NCM epochs; total epoch budget = budget x EP_NCM (equal to the other arms)."""
    R_ = EP_NCM; smax = int(math.floor(math.log(R_, eta) + 1e-9)); total = budget * R_; used = 0; i = 0; Y = []
    rng_seed = 6000 + sseed; s = smax
    while used < total:
        n = int(math.ceil((smax + 1) / (s + 1) * eta ** s)); r = R_ * eta ** (-s)
        Z = far_design(n, seed=rng_seed + 31 * i)
        for k in range(s + 1):
            nk = int(math.floor(n * eta ** (-k))); rk = int(max(1, round(r * eta ** k)))
            ys = []
            for z in Z[:nk]:
                if used + rk > total: break
                ys.append(search_eval(campaign, i, arch, z, epochs=rk)); used += rk; i += 1
            if not ys: break
            if rk == R_: Y += ys
            order = np.argsort(ys); Z = Z[order][:max(1, int(nk / eta))]
        s = s - 1 if s > 0 else smax
    # report the full-budget evaluations as the arm's trace (lower-budget validation errors are not comparable)
    return Y if Y else [1e3]
SEARCHERS = dict(random=run_random, bo=run_bo, pso=run_pso, ga=run_ga, hyperband=run_hyperband)

def campaign_frame(campaign):
    df = STORE.frame(part='search', campaign=campaign)
    return df.sort_values('idx') if len(df) else df

def two_stage(name, arch, candidates, seeds, extra_ref=None):
    """re-evaluate a shortlist on fresh seeds; select on mean validation only"""
    rows = []
    for nm, kw in candidates:
        rs = [run_cached('two_stage', 'DS05', arch, kw, s, lambda kw=kw, s=s: eval_ncm('DS05', arch, kw, s), study=name, cand=nm) for s in seeds]
        rows.append(dict(cand=nm, kw=kw, val=np.mean([r['val'] for r in rs]), test=np.mean([r['test'] for r in rs]),
                         test_sd=np.std([r['test'] for r in rs], ddof=1), per_seed=[r['test'] for r in rs]))
    return pd.DataFrame(rows)

# %% [markdown]
# ## Statistics helpers

# %%
def paired(a, b):
    """a - b per seed: mean, 95% CI, t, p, d_z"""
    a, b = np.asarray(a, float), np.asarray(b, float); d = a - b; n = len(d); m = d.mean(); s = d.std(ddof=1) if n > 1 else np.nan
    t_ = m / (s / np.sqrt(n)) if s > 0 else np.nan; p = 2 * tdist.sf(abs(t_), n - 1) if s > 0 else np.nan
    h = tdist.ppf(0.975, n - 1) * s / np.sqrt(n) if n > 1 else np.nan
    return dict(n=n, mean_a=a.mean(), sd_a=a.std(ddof=1), mean_b=b.mean(), sd_b=b.std(ddof=1), diff=m, ci_lo=m - h, ci_hi=m + h, t=t_, p=p,
                dz=m / s if s > 0 else np.nan)
def tost(d, margin):
    d = np.asarray(d, float); n = len(d); s = d.std(ddof=1); se = s / np.sqrt(n)
    p1 = tdist.sf((d.mean() + margin) / se, n - 1); p2 = tdist.cdf((d.mean() - margin) / se, n - 1); return float(max(p1, p2))
def holm(pvals):
    p = np.asarray(pvals, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj
def seeds_needed(sd, delta, alpha=0.05, power=0.8):
    return int(math.ceil(((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * sd / delta) ** 2 + 1))
print('power check: s_d 0.24, delta 0.3 ->', seeds_needed(0.24, 0.3), 'seeds')

# %% [markdown]
# ## Part B — equal-budget search for the sequential model on DS05 (B2)
# PSO with the same space, evaluator, budget (45) and two-stage selection as the parallel model in the paper.

# %%
SEQ_SEL_FILE = R('ds05_seq_selected.json')
if PARTS['B']:
    t0 = time.time(); Y = run_pso('B_seq_pso', 'seq', 0, CFG['BUDGET'])
    df = campaign_frame('B_seq_pso'); df = df[df.val < 999]
    short = df.nsmallest(15, 'val')
    seen = set(); cands = [('default', DS05_SEQ_DEFAULT)]
    for r in short.itertuples():
        k = json.dumps(r.kw, sort_keys=True)
        if k in seen: continue
        seen.add(k); cands.append(('pso_%d' % r.idx, r.kw))
        if len(cands) == 6: break
    S2 = two_stage('B_seq', 'seq', cands, SEEDS_6)
    win = S2.sort_values('val').iloc[0]
    json.dump(dict(selected=win['cand'], kw=win['kw'], val=float(win['val']), test=float(win['test'])), open(SEQ_SEL_FILE, 'w'), indent=2, default=float)
    print(S2[['cand', 'val', 'test', 'test_sd']].to_string(index=False)); print('selected (validation only):', win['cand'], '| %.1f min' % ((time.time() - t0) / 60))
DS05_SEQ = json.load(open(SEQ_SEL_FILE))['kw'] if os.path.exists(SEQ_SEL_FILE) else DS05_SEQ_DEFAULT
if not PARTS['B'] and not os.path.exists(SEQ_SEL_FILE) and any(PARTS[p] for p in 'ACDJK'):
    raise FileNotFoundError('%s is missing. Copy ds05_seq_selected.json (written by Part B on the first PC) into %s.'
                            % (SEQ_SEL_FILE, CFG['RESULTS_DIR']))

# %% [markdown]
# ## Part H — FD003: L2 dose re-selected on validation, confirmed on fresh seeds; LR / RF baselines

# %%
DOSES = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
if PARTS['H']:
    rows = []
    for arch in ('par', 'seq'):
        for l2 in DOSES:
            for s in SEEDS_FD_SEL:
                r = run_cached('H_sel', 'FD003', arch, dict(l2=l2), s, lambda arch=arch, l2=l2, s=s: eval_fd('FD003', arch, s, l2=l2))
                rows.append(dict(stage='select', arch=arch, l2=l2, seed=s, val=r['val'], test=r['test']))
    Hs = pd.DataFrame(rows); sel = Hs.groupby(['arch', 'l2']).val.mean().reset_index()
    chosen = {a: float(sel[sel.arch == a].sort_values('val').iloc[0].l2) for a in ('par', 'seq')}
    json.dump(dict(l2=chosen['par'], chosen=chosen, rule='lowest mean validation RMSE over %d seeds' % len(SEEDS_FD_SEL)), open(FD_L2_FILE, 'w'), indent=2)
    print('selected on validation:', chosen)
    for arch in ('par', 'seq'):
        for l2 in sorted({chosen[arch], 0.01}):
            for s in SEEDS_FD_CONF:
                r = run_cached('H_conf', 'FD003', arch, dict(l2=l2), s, lambda arch=arch, l2=l2, s=s: eval_fd('FD003', arch, s, l2=l2))
                rows.append(dict(stage='confirm', arch=arch, l2=l2, seed=s, val=r['val'], test=r['test']))
    H = pd.DataFrame(rows); H.to_csv(R('partH_runs.csv'), index=False)
    # classical baselines on the same windows (last time step + window mean as features)
    if not DRY:
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        d = cmapss_data('FD003'); fe = lambda X: np.hstack([X[:, -1, :], X.mean(1)])
        base = []
        for nm, mdl in [('LinearRegression', LinearRegression()), ('RandomForest', RandomForestRegressor(300, min_samples_leaf=5, n_jobs=-1, random_state=0))]:
            mdl.fit(fe(d['Xtr']), d['ytr'].ravel()); base.append(dict(model=nm, train=rmse(d['ytr'], mdl.predict(fe(d['Xtr']))),
                                                                     val=rmse(d['yva'], mdl.predict(fe(d['Xva']))), test=rmse(d['yte'], np.clip(mdl.predict(fe(d['Xte'])), 0, 125))))
        pd.DataFrame(base).to_csv(R('partH_classical.csv'), index=False); print(pd.DataFrame(base))

# %% [markdown]
# ## Part A — confirmatory sequential vs parallel under the final protocols (E1 / B1)

# %%
def confirm_rows():
    jobs = []
    for cap, tag in [(55, 'DS05_capped'), (None, 'DS05_uncapped')]:
        for arch, kw in [('par', DS05_PAR), ('seq', DS05_SEQ)]:
            for s in SEEDS_CONF:
                jobs.append((tag, arch, s, lambda arch=arch, kw=kw, s=s, cap=cap: run_cached('A', 'DS05', arch, kw, s,
                    lambda: eval_ncm('DS05', arch, kw, s, cap=cap, save_pred=(s == SEEDS_CONF[0])), cap=cap)))
    for arch in ('par', 'seq'):
        for s in SEEDS_CONF:
            jobs.append(('FD003', arch, s, lambda arch=arch, s=s: run_cached('A', 'FD003', arch, dict(l2=FD_L2), s,
                lambda: eval_fd('FD003', arch, s, l2=FD_L2, save_pred=(s == SEEDS_CONF[0])))))
            jobs.append(('B0005', arch, s, lambda arch=arch, s=s: run_cached('A', 'B0005', arch, dict(mode='trend'), s,
                lambda: eval_bat('B0005', arch, s, mode='trend', save_pred=(s == SEEDS_CONF[0])))))
    return jobs
FD_L2 = json.load(open(FD_L2_FILE))['l2'] if os.path.exists(FD_L2_FILE) else 0.3   # set by Part H (validation-selected)
if not PARTS['H'] and not os.path.exists(FD_L2_FILE) and any(PARTS[p] for p in 'ACDJK'):
    raise FileNotFoundError('%s is missing. Copy fd003_l2_selected.json (written by Part H on the first PC) into %s.'
                            % (FD_L2_FILE, CFG['RESULTS_DIR']))
if PARTS['A']:
    t0 = time.time(); res = []
    for tag, arch, s, job in confirm_rows():
        r = job(); res.append(dict(set=tag, arch=arch, seed=s, test=r['test'], val=r['val'], params=r.get('params')))
    A = pd.DataFrame(res); A.to_csv(R('partA_runs.csv'), index=False); print('Part A: %d runs, %.1f min' % (len(A), (time.time() - t0) / 60))

# %% [markdown]
# ## Part C — architecture × pooling (2 × 2) and pooling dose-response (E2 / B3 / B4)

# %%
if PARTS['C']:
    rows = []
    for arch, kw in [('par', DS05_PAR), ('seq', DS05_SEQ)]:
        for p in (0, 1, 2) if arch == 'seq' else (0, 2):
            kwp = dict(kw, n_pool=p)
            for s in SEEDS_6:
                r = run_cached('C', 'DS05', arch, kwp, s, lambda kwp=kwp, s=s, arch=arch: eval_ncm('DS05', arch, kwp, s))
                rows.append(dict(dataset='DS05', arch=arch, n_pool=p, Tc=30 / 2 ** p, seed=s, test=r['test']))
    for arch in ('par', 'seq'):
        for p in (0, 1, 2) if arch == 'seq' else (0, 2):
            for s in SEEDS_6:
                r = run_cached('C', 'FD003', arch, dict(l2=FD_L2, n_pool=p, n_conv=2), s,
                               lambda arch=arch, p=p, s=s: eval_fd('FD003', arch, s, l2=FD_L2, n_pool=p, n_conv=2))
                rows.append(dict(dataset='FD003', arch=arch, n_pool=p, Tc=50 / 2 ** p, seed=s, test=r['test']))
    C = pd.DataFrame(rows); C.to_csv(R('partC_runs.csv'), index=False); print(C.groupby(['dataset', 'arch', 'n_pool']).test.agg(['mean', 'std', 'count']))

# %% [markdown]
# ## Part D — capacity (parameters, FLOPs), a capacity-matched pair, single-branch controls (E3 / E4 / B5)

# %%
if PARTS['D']:
    cost = []
    for arch, kw in [('par', DS05_PAR), ('seq', DS05_SEQ), ('cnn', DS05_PAR), ('lstm', DS05_PAR)]:
        P_, F_ = analytic_cost(arch, kw, (30, 28)); cost.append(dict(dataset='DS05', arch=arch, params=P_, flops=F_))
    # capacity-matched sequential model: scale widths until parameters agree within 10%
    P_par = analytic_cost('par', DS05_PAR, (30, 28))[0]; best = None
    for mlt in np.linspace(0.3, 4.0, 75):
        kw = dict(DS05_SEQ, f1=max(8, int(DS05_SEQ['f1'] * mlt)), lstm_units=max(8, int(DS05_SEQ['lstm_units'] * mlt)))
        P_ = analytic_cost('seq', kw, (30, 28))[0]
        if best is None or abs(P_ / P_par - 1) < abs(best[1] / P_par - 1): best = (kw, P_)
    SEQ_MATCH = best[0]; cost.append(dict(dataset='DS05', arch='seq_matched', params=best[1], flops=analytic_cost('seq', SEQ_MATCH, (30, 28))[1]))
    print('capacity match: parallel %d, matched sequential %d (%.1f%%)' % (P_par, best[1], 100 * (best[1] / P_par - 1)))
    rows = []
    for s in SEEDS_6:
        r = run_cached('D', 'DS05', 'seq', SEQ_MATCH, s, lambda s=s: eval_ncm('DS05', 'seq', SEQ_MATCH, s)); rows.append(dict(set='DS05_matched', arch='seq_matched', seed=s, test=r['test']))
    for arch in ('cnn', 'lstm', 'par'):
        for s in SEEDS_CONF:
            r = run_cached('D', 'FD003', arch, dict(l2=FD_L2), s, lambda arch=arch, s=s: eval_fd('FD003', arch, s, l2=FD_L2))
            rows.append(dict(set='FD003_branch', arch=arch, seed=s, test=r['test'], params=r.get('params')))
            r = run_cached('D', 'B0005', arch, dict(mode='trend'), s, lambda arch=arch, s=s: eval_bat('B0005', arch, s, mode='trend'))
            rows.append(dict(set='B0005_branch', arch=arch, seed=s, test=r['test'], params=r.get('params')))
    pd.DataFrame(cost).to_csv(R('partD_cost.csv'), index=False); pd.DataFrame(rows).to_csv(R('partD_runs.csv'), index=False)
    print(pd.DataFrame(cost))

# %% [markdown]
# ## Part F — Holm correction for the 12-factor screen (statistics only, from the v12 files)

# %%
if PARTS['F']:
    fa = os.path.join(CFG['V12_DIR'], 'ncmapss_v12_all_trials.csv')
    if os.path.exists(fa):
        TRv = pd.read_csv(fa); c = TRv[TRv.tag.str.startswith('confirm', na=False)].copy(); c['name'] = c.tag.str[8:]
        inc = c[c.name == 'incumbent'].sort_values('seed').test_capped.values; rows = []
        for nm in c.name.unique():
            if nm == 'incumbent': continue
            v = c[c.name == nm].sort_values('seed').test_capped.values; r = paired(v, inc); r['factor'] = nm; rows.append(r)
        F_ = pd.DataFrame(rows); F_['p_holm'] = holm(F_.p.values); F_.to_csv(R('partF_screen_holm.csv'), index=False)
        SUMMARY_F = F_; print(F_[['factor', 'diff', 'ci_lo', 'ci_hi', 'p', 'p_holm']].round(3).to_string(index=False))
    else: print('ncmapss_v12_all_trials.csv not found in V12_DIR; skipped')

# %% [markdown]
# ## Part G — battery: 30 seeds per model and leave-one-cell-out

# %%
if PARTS['G']:
    rows = []
    for arch, mode in [('par', 'trend'), ('par', 'net'), ('seq', 'trend'), ('seq', 'net'), ('cnn', 'trend'), ('lstm', 'trend')]:
        for s in SEEDS_BAT:
            r = run_cached('G', 'B0005', arch, dict(mode=mode), s, lambda arch=arch, mode=mode, s=s: eval_bat('B0005', arch, s, mode=mode))
            rows.append(dict(cell='B0005', arch=arch, mode=mode, seed=s, test=r['test'], floor=r['floor'], trend_only=r['trend_only']))
    CELLS = ['B0005', 'B0006', 'B0007', 'B0018']
    for tgt in CELLS:
        others = [c for c in CELLS if c != tgt]
        for cond, cells in [('within', None), ('loco_zero_shot', others), ('loco_pooled', others + [tgt])]:
            for s in SEEDS_CONF:
                r = run_cached('G_loco', tgt, 'par', dict(mode='trend', cond=cond), s, lambda tgt=tgt, cells=cells, s=s: eval_bat(tgt, 'par', s, mode='trend', train_cells=cells))
                rows.append(dict(cell=tgt, arch='par', mode='trend', cond=cond, seed=s, test=r['test'], floor=r['floor'], trend_only=r['trend_only']))
    G = pd.DataFrame(rows); G['cond'] = G.get('cond', pd.Series(index=G.index, dtype=object)).fillna('main')
    G.to_csv(R('partG_runs.csv'), index=False); print(G.groupby(['cell', 'cond', 'arch', 'mode']).test.agg(['mean', 'std', 'count']))

# %% [markdown]
# ## Part I — DS05 with physical sensors only (E11)

# %%
if PARTS['I']:
    rows = []
    for ch in ('xs', 'w_xs', 'xs_xv'):
        for s in SEEDS_6:
            r = run_cached('I', 'DS05', 'par', DS05_PAR, s, lambda ch=ch, s=s: eval_ncm('DS05', 'par', DS05_PAR, s, channels=ch), channels=ch)
            rows.append(dict(channels=ch, seed=s, test=r['test'], test_raw=r['test_raw']))
    I_ = pd.DataFrame(rows); I_.to_csv(R('partI_runs.csv'), index=False); print(I_.groupby('channels').test.agg(['mean', 'std']))

# %% [markdown]
# ## Part J — FD001, FD002, FD004 and N-CMAPSS DS02 (E7 / E8)

# %%
if PARTS['J']:
    rows = []
    for fd in ('FD001', 'FD002', 'FD004'):
        for arch in ('par', 'seq'):
            for s in SEEDS_6:
                r = run_cached('J', fd, arch, dict(l2=FD_L2), s, lambda fd=fd, arch=arch, s=s: eval_fd(fd, arch, s, l2=FD_L2))
                rows.append(dict(dataset=fd, arch=arch, seed=s, test=r['test'], nasa=r['nasa'], triv=r['triv']))
    _ds02 = os.path.join(CFG['NCMAPSS_DIR'], NCM_FILES['DS02'])
    if DRY or os.path.isfile(_ds02):
        for arch, kw in [('par', DS05_PAR), ('seq', DS05_SEQ)]:
            for s in SEEDS_6:
                r = run_cached('J', 'DS02', arch, kw, s, lambda arch=arch, kw=kw, s=s: eval_ncm('DS02', arch, kw, s))
                rows.append(dict(dataset='DS02', arch=arch, seed=s, test=r['test'], nasa=r['nasa']))
    else:   # the DS02 file is not on this PC: skip it now; re-running later with the file present adds only these 12 runs
        print('WARNING: %s not found - DS02 skipped for now. Put the file next to N-CMAPSS_DS05.h5 and run again.' % NCM_FILES['DS02'])
    J = pd.DataFrame(rows); J.to_csv(R('partJ_runs.csv'), index=False); print(J.groupby(['dataset', 'arch']).test.agg(['mean', 'std']))

# %% [markdown]
# ## Part K — external baselines and prognostic metrics (E10 / E12)

# %%
if PARTS['K']:
    rows = []
    base_kw = dict(n_conv=1, f1=64, f_growth=1, kernel=3, n_lstm=1, lstm_units=64, dense1=64, dense2=0, lr=1e-3, batch=256, dropout=0.1, l2reg=0)
    for arch in ('gru', 'bilstm', 'tcn', 'transformer'):
        for s in SEEDS_6:
            r = run_cached('K', 'DS05', arch, base_kw, s, lambda arch=arch, s=s: eval_ncm('DS05', arch, base_kw, s)); rows.append(dict(dataset='DS05', arch=arch, seed=s, test=r['test'], nasa=r['nasa']))
            r = run_cached('K', 'FD003', arch, {}, s, lambda arch=arch, s=s: eval_fd('FD003', arch, s)); rows.append(dict(dataset='FD003', arch=arch, seed=s, test=r['test'], nasa=r['nasa']))
    # uncertainty and life-stage error for the final parallel models (MC-dropout, 30 passes)
    for s in SEEDS_6:
        r = run_cached('K_mc', 'DS05', 'par', DS05_PAR, s, lambda s=s: eval_ncm('DS05', 'par', DS05_PAR, s, mc=True, save_pred=True, tag='_mc'))
        rows.append(dict(dataset='DS05', arch='par_mc', seed=s, test=r['test'], nasa=r['nasa'], picp=r.get('picp'), mpiw=r.get('mpiw')))
        r = run_cached('K_mc', 'FD003', 'par', dict(l2=FD_L2), s, lambda s=s: eval_fd('FD003', 'par', s, l2=FD_L2, mc=True))
        rows.append(dict(dataset='FD003', arch='par_mc', seed=s, test=r['test'], nasa=r['nasa'], picp=r.get('picp'), mpiw=r.get('mpiw')))
    Kd = pd.DataFrame(rows); Kd.to_csv(R('partK_runs.csv'), index=False); print(Kd.groupby(['dataset', 'arch'])[['test', 'nasa']].mean())

# %% [markdown]
# ## Part E — five search methods, three independent campaigns each (E5 / E6)
# Run last because it is the longest part (about 45 GPU-h): every other part finishes first.
# All arms start from designs at distance ≥ 0.35 from the paper's final configuration (no known configurations are
# injected), use the same 13-D space, the same training budget and 45 full evaluations (Hyperband: the same total epochs).

# %%
if PARTS['E']:
    t0 = time.time()
    for meth, fn in SEARCHERS.items():
        for ss in CFG['SEARCH_SEEDS']:
            fn('E_%s_%d' % (meth, ss), 'par', ss, CFG['BUDGET']); print('  %-9s seed %d done (%.1f min)' % (meth, ss, (time.time() - t0) / 60))
    rows = []
    for meth in SEARCHERS:
        for ss in CFG['SEARCH_SEEDS']:
            df = campaign_frame('E_%s_%d' % (meth, ss))
            if meth == 'hyperband': df = df[df.epochs == EP_NCM]
            df = df[df.val < 999]
            if not len(df): continue
            b = df.loc[df.val.idxmin()]
            rows.append(dict(method=meth, search_seed=ss, best_val=b.val, test_of_best=b.test, median_val=df.val.median(), n=len(df),
                             gpu_h=campaign_frame('E_%s_%d' % (meth, ss)).secs.sum() / 3600))
    E = pd.DataFrame(rows); E.to_csv(R('partE_summary.csv'), index=False); print(E.groupby('method')[['best_val', 'test_of_best', 'median_val']].agg(['mean', 'std']))

# %% [markdown]
# ## Part L — analysis: tables and figures for the revision

# %%
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.grid': True, 'grid.alpha': 0.25})
SUMMARY = {}
def fig_save(fig, name): fig.tight_layout(); fig.savefig(R('figures', name), dpi=300, bbox_inches='tight'); plt.close(fig)
TRIV = {'DS05_capped': 18.393, 'DS05_uncapped': 23.142, 'FD003': 42.17, 'B0005': 0.02927}
if PARTS['L']:
    # ---- T4: confirmatory comparison
    if os.path.exists(R('partA_runs.csv')):
        A = pd.read_csv(R('partA_runs.csv')); rows = []
        for st in A.set.unique():
            a = A[(A.set == st) & (A.arch == 'seq')].sort_values('seed').test.values; b = A[(A.set == st) & (A.arch == 'par')].sort_values('seed').test.values
            if len(a) != len(b) or len(a) < 2: continue
            r = paired(a, b); r['set'] = st; r['tost_p_2pct'] = tost(a - b, 0.02 * TRIV.get(st, 1.0)); rows.append(r)
        T4 = pd.DataFrame(rows)
        if len(T4): T4['p_holm'] = holm(T4.p.values)
        T4.to_csv(R('T4_confirmatory.csv'), index=False); SUMMARY['T4'] = T4.to_dict('records'); print(T4.round(4).to_string(index=False))
        fig, ax = plt.subplots(1, max(1, len(A.set.unique())), figsize=(7, 2.4))
        ax = np.atleast_1d(ax)
        for k, st in enumerate(A.set.unique()):
            d = A[A.set == st].pivot(index='seed', columns='arch', values='test')
            for _, rw in d.iterrows(): ax[k].plot([0, 1], [rw.get('seq', np.nan), rw.get('par', np.nan)], '-', color='#888', lw=.7)
            ax[k].plot(np.zeros(len(d)), d.get('seq'), 'o', color='#C00000', ms=3); ax[k].plot(np.ones(len(d)), d.get('par'), 'o', color='#0070C0', ms=3)
            ax[k].set_xticks([0, 1]); ax[k].set_xticklabels(['sequential', 'parallel']); ax[k].set_title(st.replace('_', ' '), fontsize=7.5)
        ax[0].set_ylabel('Test RMSE'); fig_save(fig, 'F3_paired_seeds.png')
    # ---- F4 / F5: pooling
    if os.path.exists(R('partC_runs.csv')):
        C = pd.read_csv(R('partC_runs.csv')); out = []
        for ds in C.dataset.unique():
            c = C[C.dataset == ds]
            def per(arch, p): return c[(c.arch == arch) & (c.n_pool == p)].sort_values('seed').test.values
            if all(len(per(a, p)) for a in ('seq', 'par') for p in (0, 2)):
                dint = (per('seq', 2) - per('seq', 0)) - (per('par', 2) - per('par', 0))
                r = paired(dint, np.zeros_like(dint)); r['dataset'] = ds; out.append(r)
            fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.4))
            for arch, col in [('seq', '#C00000'), ('par', '#0070C0')]:
                g = c[c.arch == arch].groupby('n_pool').test.agg(['mean', 'std', 'count'])
                ax[0].errorbar(g.index, g['mean'], yerr=1.96 * g['std'] / np.sqrt(g['count']), marker='o', color=col, label=arch, capsize=3)
            ax[0].set_xlabel('pooling stages p'); ax[0].set_ylabel('Test RMSE'); ax[0].legend(); ax[0].set_title(ds + ': architecture × pooling')
            g = c[c.arch == 'seq'].groupby('Tc').test.agg(['mean', 'std', 'count'])
            ax[1].errorbar(g.index, g['mean'], yerr=1.96 * g['std'] / np.sqrt(g['count']), marker='s', color='#C00000', capsize=3)
            ax[1].set_xlabel('T_c (steps reaching the LSTM)'); ax[1].set_title(ds + ': sequential dose-response'); fig_save(fig, 'F4_F5_pooling_%s.png' % ds)
        pd.DataFrame(out).to_csv(R('interaction_tests.csv'), index=False); SUMMARY['interaction'] = out
    # ---- F6 + optimizer statistics
    if os.path.exists(R('partE_summary.csv')):
        E = pd.read_csv(R('partE_summary.csv')); fig, ax = plt.subplots(figsize=(3.6, 2.5))
        for meth, col in zip(SEARCHERS, ['#888', '#C00000', '#0070C0', '#008000', '#7030A0']):
            curves = []
            for ss in CFG['SEARCH_SEEDS']:
                df = campaign_frame('E_%s_%d' % (meth, ss))
                if meth == 'hyperband': df = df[df.epochs == EP_NCM]
                if len(df): v = np.minimum.accumulate(np.where(df.val < 999, df.val, np.nan).astype(float)); curves.append(pd.Series(v).ffill().values)
            if curves:
                L_ = min(len(c) for c in curves); M = np.array([c[:L_] for c in curves])
                ax.plot(np.arange(1, L_ + 1), M.mean(0), color=col, label=meth); ax.fill_between(np.arange(1, L_ + 1), M.min(0), M.max(0), color=col, alpha=.15)
        ax.set_xlabel('Evaluations'); ax.set_ylabel('Best validation RMSE so far'); ax.legend(fontsize=6); fig_save(fig, 'F6_search_curves.png')
        piv = E.pivot(index='search_seed', columns='method', values='best_val').dropna()
        if piv.shape[0] >= 2 and piv.shape[1] >= 3:
            fr = friedmanchisquare(*[piv[c] for c in piv.columns]); SUMMARY['friedman'] = dict(stat=float(fr.statistic), p=float(fr.pvalue), methods=list(piv.columns))
            print('Friedman:', SUMMARY['friedman'])
        SUMMARY['partE'] = E.groupby('method')[['best_val', 'test_of_best', 'median_val', 'gpu_h']].mean().reset_index().to_dict('records')
    # ---- F8: trajectories and RMSE by life stage
    pf = sorted(glob.glob(R('preds', 'DS05_par_cap_s*.npz')))
    if pf:
        z = np.load(pf[0]); y, yh, u = z['y'], z['yhat'], z['unit']; fig, ax = plt.subplots(1, 2, figsize=(7, 2.4))
        for un in np.unique(u)[:3]:
            m = u == un; ax[0].plot(y[m], 'k-', lw=.8); ax[0].plot(yh[m], '.', ms=1.5, label='unit %d' % un)
        ax[0].set_xlabel('test window'); ax[0].set_ylabel('RUL (capped at 55)'); ax[0].legend(fontsize=6)
        bins = [(0, 20, 'late'), (20, 40, 'mid'), (40, 1e9, 'early')]; br = [rmse(y[(y >= a) & (y < b)], yh[(y >= a) & (y < b)]) for a, b, _ in bins]
        ax[1].bar([b[2] for b in bins], br, color='#0070C0'); ax[1].set_ylabel('RMSE'); ax[1].set_title('error by life stage (true RUL)'); fig_save(fig, 'F8_trajectories_stage.png')
        SUMMARY['life_stage_rmse_DS05'] = dict(zip([b[2] for b in bins], br))
    # ---- F9: RMSE vs parameters
    allr = pd.DataFrame(list(STORE.rows.values()))
    if len(allr) and 'params' in allr:
        q = allr[(allr.dataset == 'DS05') & allr.params.notna() & (allr.part.isin(['A', 'C', 'D', 'search']))]
        if len(q):
            fig, ax = plt.subplots(figsize=(3.4, 2.4))
            for arch, col in [('par', '#0070C0'), ('seq', '#C00000')]:
                w = q[q.arch == arch]; ax.scatter(w.params, w.test, s=6, color=col, alpha=.6, label=arch)
            ax.set_xscale('log'); ax.set_xlabel('parameters'); ax.set_ylabel('Test RMSE (capped)'); ax.legend(fontsize=6); fig_save(fig, 'F9_rmse_vs_params.png')
    # ---- compute cost
    if len(allr):
        cost = allr.groupby('part').secs.sum() / 3600; SUMMARY['gpu_hours_by_part'] = cost.round(2).to_dict(); print('GPU-hours by part:\n', cost.round(2))
    SUMMARY['env'] = json.load(open(R('env.json')))
    json.dump(SUMMARY, open(R('summary.json'), 'w'), indent=2, default=float); print('wrote', R('summary.json'))
