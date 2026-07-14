#!/bin/bash
# merge_real_data.sh — merge all real-capture NPZs into real_combined_v2_1024.npz
# Re-run after any new capture session to update the combined dataset.
# Output is gitignored (data/ dir); run this to regenerate locally.
set -euo pipefail
cd /home/brendan/AnalysisApp/tools/ml

.venv/bin/python3 - <<'EOF'
import numpy as np
from pathlib import Path
from collections import Counter

# Priority order: old combined first, then new captures (newest wins on duplicates)
candidates = sorted(Path('data').glob('real_*.npz')) + \
             sorted(Path('data').glob('pi_capture_*.npz'))
files = [Path('data/real_combined_1024.npz')] + \
        [f for f in candidates if f.name != 'real_combined_v2_1024.npz'
                                and f.name != 'real_combined_1024.npz']

all_X, all_y, all_snrs = [], [], []
classes_ref = None

for f in dict.fromkeys(files):   # deduplicate paths, preserve order
    if not f.exists():
        continue
    d = np.load(f)
    if classes_ref is None:
        classes_ref = d['classes'].tolist()
    src_cls = d['classes'].tolist()
    remap = {i: classes_ref.index(c) for i, c in enumerate(src_cls) if c in classes_ref}
    mask = np.isin(d['y'], list(remap.keys()))
    X = d['X'][mask]
    y = np.array([remap[int(v)] for v in d['y'][mask]])
    snrs = d['snrs'][mask]
    all_X.append(X); all_y.append(y); all_snrs.append(snrs)
    print(f'  {f.name}: {len(X):,} samples')

X_out = np.concatenate(all_X)
y_out = np.concatenate(all_y)
snrs_out = np.concatenate(all_snrs)

c = Counter(y_out.tolist())
print(f'\nMerged: {len(X_out):,} samples across {len(c)} classes')
for cid, n in sorted(c.items(), key=lambda x: -x[1]):
    print(f'  {classes_ref[cid]}: {n:,}')

np.savez_compressed('data/real_combined_v2_1024.npz',
    X=X_out, y=y_out, snrs=snrs_out, classes=np.array(classes_ref))
print('\nSaved → data/real_combined_v2_1024.npz')
EOF
