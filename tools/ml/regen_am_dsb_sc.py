#!/usr/bin/env python3
"""Regenerate AM_DSB_SC samples in-place in existing .npz datasets, using the
fixed _am_dsb_sc generator (was Hermitian-symmetric/DC-centered, confusable
with 4ASK/16ASK/OOK). Splices new samples into the same index range, preserving
ordering (class then SNR then n_per_snr), so y/snrs stay valid.
"""
import numpy as np
import generate_v3 as g

JOBS = [
    ("data/v6_holdout_1024.npz", 34, 3333, False),
    ("data/synth_v6_47class_1024.npz", 5000, 1111, False),
    ("data/synth_v9_47class_1024.npz", 5000, 2222, True),
]

for path, n_per_snr, seed, multi_sdr in JOBS:
    print(f"=== {path} (n_per_snr={n_per_snr}, seed={seed}, multi_sdr={multi_sdr}) ===", flush=True)
    d = dict(np.load(path, allow_pickle=True))
    classes = [str(c) for c in d["classes"]]
    am_idx = classes.index("AM_DSB_SC")
    mask = d["y"] == am_idx
    n_old = int(mask.sum())
    print(f"  existing AM_DSB_SC samples: {n_old}")

    X_new, y_new, s_new, _ = g.generate(
        n_per_snr=n_per_snr, length=1024,
        snr_min=-10.0, snr_max=30.0, snr_step=5.0,
        seed=seed, impair=True, multi_sdr=multi_sdr,
        filter_classes=["AM_DSB_SC"],
    )
    print(f"  regenerated: {X_new.shape[0]}")
    assert X_new.shape[0] == n_old, (X_new.shape[0], n_old)
    # Same generation order (class -> snr -> n_per_snr) as the original full
    # run, so per-SNR ordering lines up; splice in place.
    idx = np.where(mask)[0]
    d["X"][idx] = X_new
    d["snrs"][idx] = s_new
    np.savez_compressed(path, **d)
    print(f"  saved {path}", flush=True)

print("Done.")
