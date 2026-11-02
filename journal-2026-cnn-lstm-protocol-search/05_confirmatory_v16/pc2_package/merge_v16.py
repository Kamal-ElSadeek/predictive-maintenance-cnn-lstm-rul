"""Merge the second PC's results into the lab PC's results folder.

Usage (on the lab PC, in the folder that holds results_v16, with the lab run STOPPED):
    python merge_v16.py results_v16_pc2 results_v16
    python merge_v16.py results_v16_pc2 results_v16 --override I,J,K
      (--override: for these parts the second PC's records win even where the lab already has the run,
       so that every part is measured on one GPU; the lab record stays in the file but is superseded,
       because the store keeps the last line written for each key)

- appends every run from results_v16_pc2/runs.jsonl whose key is not already in results_v16/runs.jsonl
- copies prediction files (preds/*.npz) that the lab folder does not have
- never changes or deletes an existing lab record; a backup of runs.jsonl is written first
Restart run_v16.bat full afterwards: the merged runs are skipped and Part L analyses everything together.
"""
import json, os, shutil, sys, time

def load(path):
    rows = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line); rows[r['key']] = r
                except Exception:
                    pass
    return rows

def main(src, dst, override=()):
    s_runs, d_runs = os.path.join(src, 'runs.jsonl'), os.path.join(dst, 'runs.jsonl')
    S, D = load(s_runs), load(d_runs)
    print('source %d runs, destination %d runs' % (len(S), len(D)))
    new = [r for k, r in S.items() if k not in D or r.get('part') in override]
    if override: print('override parts:', ','.join(override), '-', sum(1 for k, r in S.items() if k in D and r.get('part') in override), 'lab records superseded')
    if os.path.exists(d_runs):
        bak = d_runs + '.bak_' + time.strftime('%Y%m%d_%H%M%S'); shutil.copy2(d_runs, bak); print('backup:', bak)
        with open(d_runs, 'rb+') as f:            # make sure the file ends with a newline before appending
            f.seek(0, 2)
            if f.tell() > 0:
                f.seek(-1, 2)
                if f.read(1) != b'\n': f.write(b'\n')
    with open(d_runs, 'a') as f:
        for r in new: f.write(json.dumps(r, default=float) + '\n')
        f.flush(); os.fsync(f.fileno())
    by_part = {}
    for r in new: by_part[r.get('part')] = by_part.get(r.get('part'), 0) + 1
    print('appended %d runs: %s' % (len(new), by_part))
    sp, dp = os.path.join(src, 'preds'), os.path.join(dst, 'preds')
    if os.path.isdir(sp):
        os.makedirs(dp, exist_ok=True); n = 0
        for fn in os.listdir(sp):
            if not os.path.exists(os.path.join(dp, fn)): shutil.copy2(os.path.join(sp, fn), dp); n += 1
        print('copied %d prediction files' % n)
    if os.path.exists(os.path.join(src, 'env.json')):
        shutil.copy2(os.path.join(src, 'env.json'), os.path.join(dst, 'env_pc2.json')); print('second-PC environment saved as env_pc2.json')

if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    ov = ()
    if '--override' in sys.argv: ov = tuple(x.strip() for x in sys.argv[sys.argv.index('--override') + 1].split(','))
    args = [a for a in args if a not in (','.join(ov),)]
    if len(args) != 2: print(__doc__); sys.exit(1)
    main(args[0], args[1], ov)
