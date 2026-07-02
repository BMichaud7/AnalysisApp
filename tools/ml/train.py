#!/usr/bin/env python3
# ========================================================================
# Project: OpenRFStack
# Author:  Brendan Michaud
# Year:    2026
# Part of OpenRFStack (https://github.com/OpenRFStack)
#
# Licensed under the Personal Use License.
# Do not use for commercial, organizational, or military purposes.
# Contact author for permission: https://github.com/OpenRFStack
# ========================================================================

"""
Train an AMR classifier on any supported dataset.

── Quick start (no downloads needed) ─────────────────────────────────────────
    # 1. Generate synthetic training data (~2 min, 13 mods × 26 SNRs × 200 = 67 600 examples)
    python -m datasets generate --out data/synthetic.npz --n 200

    # 2. Train
    python train.py --npz data/synthetic.npz --model resnet --epochs 30 \
                    --out models/synthetic_resnet.onnx

── RadioML datasets (requires Kaggle account) ────────────────────────────────
    python -m datasets download radioml2016 --dest data/
    python -m datasets download radioml2018 --dest data/

    python train.py --data data/RML2016.10a.hdf5  --version 2016 \
                    --model resnet --epochs 50 \
                    --out models/radioml2016_resnet.onnx

    python train.py --data data/RML2018.01A_dict.hdf5 --version 2018 \
                    --model fusion --epochs 100 --cuda \
                    --out models/radioml2018_fusion.onnx

── HuggingFace datasets ──────────────────────────────────────────────────────
    python train.py --hf-dataset username/radio-amr --model resnet --epochs 50 \
                    --out models/hf_resnet.onnx

── Mix multiple sources ──────────────────────────────────────────────────────
    python train.py --npz data/synthetic.npz \
                    --data data/RML2016.10a.hdf5 --version 2016 \
                    --model resnet --epochs 50 \
                    --out models/combined_resnet.onnx

After training the ONNX model can be dropped into AnalysisApp and loaded
by the C++ OnnxClassifier (include/OnnxClassifier.hpp).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler, random_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "test_harness"))
from datasets import (
    load_radioml_2016, load_radioml_2018,
    RADIOML_2016_MODS, RADIOML_2018_MODS,
    generate_synthetic, save_synthetic_npz,
    load_numpy_npz, load_hf_dataset,
    SYNTHETIC_MODS, IqSample, normalise,
)
from model import RadioCNN, RadioResNet, RadioFusion, export_onnx


# ── Dataset builders ──────────────────────────────────────────────────────────

def samples_to_tensors(samples: list[IqSample],
                       class_names: list[str] | None = None
                       ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """
    Convert a list of IqSample into (X, y, class_names).
    X: (N, 2, L) float32   y: (N,) int64
    """
    if class_names is None:
        class_names = sorted(set(s.ground_truth for s in samples))

    label_map = {m: i for i, m in enumerate(class_names)}
    X_list, y_list = [], []
    for s in samples:
        label = label_map.get(s.ground_truth)
        if label is None:
            continue
        iq = normalise(s.iq)
        X_list.append(np.stack([iq.real, iq.imag], axis=0).astype(np.float32))
        y_list.append(label)

    X = torch.from_numpy(np.stack(X_list, axis=0))
    y = torch.tensor(y_list, dtype=torch.long)
    return X, y, class_names


def build_tensors(args, dann_real_set: set | None = None
                  ) -> tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor | None]:
    """Load all requested sources and merge into a single tensor dataset.

    If dann_real_set is provided, also returns a domain-label tensor (0=synthetic,
    1=real) whose length exactly matches X — derived from IqSample.source_file
    during the initial load, avoiding a second full re-load of all NPZ files.
    """
    all_samples: list[IqSample] = []
    class_names: list[str] | None = None

    # RadioML HDF5
    if args.data:
        print(f"Loading RadioML {args.version} from {args.data} …")
        if args.version == "2016":
            s = load_radioml_2016(args.data, snr_min_db=args.snr_min,
                                  max_per_class=args.max_per_class)
            class_names = RADIOML_2016_MODS
        else:
            s = load_radioml_2018(args.data, snr_min_db=args.snr_min,
                                  max_per_class=args.max_per_class)
            class_names = RADIOML_2018_MODS
        print(f"  {len(s)} RadioML samples")
        all_samples.extend(s)

    # Numpy .npz
    if args.npz:
        for npz_path in args.npz:
            print(f"Loading numpy npz: {npz_path} …")
            s = load_numpy_npz(npz_path, snr_min_db=args.snr_min,
                               max_per_class=args.max_per_class)
            print(f"  {len(s)} npz samples")
            all_samples.extend(s)

    # HuggingFace dataset
    if args.hf_dataset:
        print(f"Loading HuggingFace: {args.hf_dataset} …")
        s = load_hf_dataset(args.hf_dataset, snr_min_db=args.snr_min,
                            max_per_class=args.max_per_class)
        all_samples.extend(s)

    # Synthetic generation
    if args.synthetic:
        print(f"Generating {args.synthetic} synthetic samples per (mod, SNR) …")
        s = generate_synthetic(
            n_per_class=args.synthetic,
            snr_range=(args.snr_min, 30),
            sample_len=args.synthetic_len,
        )
        print(f"  {len(s)} synthetic samples, {len(SYNTHETIC_MODS)} mods")
        all_samples.extend(s)

    if not all_samples:
        raise RuntimeError(
            "No data loaded. Provide at least one of:\n"
            "  --data HDF5  --npz FILE  --hf-dataset ID  --synthetic N"
        )

    # ── Family specialist: keep only this family's classes ──────────────────
    if getattr(args, "family", None):
        from family_map import CLASS_FAMILY
        fam = args.family.upper()
        before = len(all_samples)
        all_samples = [s for s in all_samples if CLASS_FAMILY.get(s.ground_truth) == fam]
        if not all_samples:
            raise RuntimeError(
                f"No samples found for family '{fam}'. "
                f"Valid families: {sorted(set(CLASS_FAMILY.values()))}"
            )
        print(f"  Family filter '{fam}': {before:,} → {len(all_samples):,} samples")
        class_names = None  # let samples_to_tensors derive from the filtered set

    # ── Router: relabel each sample to its coarse family ────────────────────
    if getattr(args, "router", False):
        from family_map import CLASS_FAMILY, FAMILY_NAMES
        before = len(all_samples)
        relabeled = []
        for s in all_samples:
            fam = CLASS_FAMILY.get(s.ground_truth)
            if fam is None:
                continue
            s.ground_truth = fam
            relabeled.append(s)
        all_samples = relabeled
        print(f"  Router relabel: {before:,} → {len(all_samples):,} samples "
              f"across {len(FAMILY_NAMES)} families")
        class_names = FAMILY_NAMES  # fixed order, independent of what's present

    # ── Global per-class cap: applies across all combined sources ────────────
    # Per-file caps above limit contribution from any single source, but the
    # combined dataset still exceeds RAM when multiple files each contribute
    # their full per-class quota (4 files × 20K × 47 classes ≈ 3.76M samples
    # ≈ 30+ GB peak, which OOM-kills on 48 GB machines). Apply a global cap
    # here so total size stays within budget regardless of how many files are
    # loaded. Shuffle each class's pool first so we draw evenly from all files.
    if args.max_per_class and args.max_per_class > 0:
        import random as _rng
        from collections import defaultdict as _dd
        per_class: dict = _dd(list)
        for s in all_samples:
            per_class[s.ground_truth].append(s)
        all_samples = []
        for cls_samples in per_class.values():
            _rng.shuffle(cls_samples)
            all_samples.extend(cls_samples[:args.max_per_class])
        _rng.shuffle(all_samples)
        print(f"  After global max-per-class ({args.max_per_class}): {len(all_samples):,}")

    print(f"Total samples: {len(all_samples)}")
    X, y, cn = samples_to_tensors(all_samples, class_names)
    print(f"  Input shape: {X.shape}  Classes: {len(cn)}")

    domain_tensor: torch.Tensor | None = None
    if dann_real_set is not None:
        label_map = {m: i for i, m in enumerate(cn)}
        domain_parts = [
            1 if s.source_file in dann_real_set else 0
            for s in all_samples
            if label_map.get(s.ground_truth) is not None
        ]
        domain_tensor = torch.tensor(domain_parts, dtype=torch.long)

    return X, y, cn, domain_tensor


# ── Focal loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal loss (Lin et al. 2017): down-weights easy examples so training
    focuses on hard ones.  γ=2 is a good default; γ=0 reduces to cross-entropy.
    """
    def __init__(self, gamma: float = 2.0, weight=None, label_smoothing: float = 0.05):
        super().__init__()
        self.gamma           = gamma
        self.weight          = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ── Mixup augmentation ────────────────────────────────────────────────────────

def mixup_batch(X: torch.Tensor, y: torch.Tensor, alpha: float = 0.3):
    """Beta-distributed linear interpolation between pairs of samples."""
    lam  = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(len(X), device=X.device)
    return lam * X + (1 - lam) * X[perm], y, y[perm], lam


def mixup_loss(crit, logits, y_a, y_b, lam):
    return lam * crit(logits, y_a) + (1 - lam) * crit(logits, y_b)


# ── Online channel augmentation ───────────────────────────────────────────────

class IqAugment(nn.Module):
    """
    GPU-vectorized channel impairments applied per-batch during training.
    No CFO — CFO ±3% SR (±6000 Hz) destroyed narrowband classes (RTTY ±85 Hz,
    NAVTEX ±150 Hz, DSC ±400 Hz) that the generator already impairs at ±600 Hz.
    Augments: phase rotation, IQ imbalance, real hardware noise (or AWGN fallback).

    If noise_bank is provided (shape (M, 2, L) float32 from capture_noise.py),
    real PlutoSDR noise segments replace synthetic AWGN.  This forces the model
    to learn features that survive actual hardware noise characteristics instead
    of idealised Gaussian noise.
    """
    def __init__(self,
                 iq_imbal:   float = 0.04,
                 snr_lo_db:  float = 5.0,
                 snr_hi_db:  float = 40.0,
                 p:          float = 0.85,
                 noise_bank: torch.Tensor | None = None):
        super().__init__()
        self.iq_imbal = iq_imbal
        self.snr_lo   = snr_lo_db
        self.snr_hi   = snr_hi_db
        self.p        = p

        if noise_bank is not None:
            # Store as complex (M, L) so GPU indexing is fast
            nb_cplx = torch.complex(noise_bank[:, 0], noise_bank[:, 1])
            self.register_buffer("noise_bank", nb_cplx)
            print(f"  IqAugment: real noise bank loaded  "
                  f"{len(noise_bank):,} segments × {noise_bank.shape[-1]} samples")
        else:
            self.noise_bank = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return X
        B, _, N = X.shape
        device  = X.device

        iq = torch.complex(X[:, 0], X[:, 1])  # (B, N)

        # Random phase rotation — safe for all classes
        phase = (torch.rand(B, 1, device=device) * 2 - 1) * math.pi
        iq = iq * torch.polar(torch.ones(B, 1, device=device), phase)

        # IQ imbalance: amplitude α and phase φ mismatch (small-angle approx)
        alpha = 1.0 + (torch.rand(B, 1, device=device) * 2 - 1) * self.iq_imbal
        phi   = (torch.rand(B, 1, device=device) * 2 - 1) * self.iq_imbal
        I_new = iq.real * alpha
        Q_new = -iq.real * phi + iq.imag
        iq    = torch.complex(I_new, Q_new)

        # Noise at random per-sample SNR
        snr_db  = (torch.rand(B, 1, device=device) * (self.snr_hi - self.snr_lo)
                   + self.snr_lo)
        snr_lin = 10.0 ** (snr_db / 10.0)
        sig_pwr = iq.abs().pow(2).mean(dim=1, keepdim=True)
        n_pwr   = sig_pwr / snr_lin.clamp(min=1e-4)

        if self.noise_bank is not None:
            # Sample random real-noise segments from the bank, scale to target power
            bank = self.noise_bank.to(device)
            M    = bank.shape[0]
            L    = bank.shape[1]
            idx  = torch.randint(0, M, (B,), device=device)
            segs = bank[idx]                                    # (B, L)
            if L > N:
                off  = torch.randint(0, L - N, (B,), device=device)
                segs = torch.stack([segs[i, off[i]: off[i] + N] for i in range(B)])
            elif L < N:
                # tile — rare if window sizes match
                repeats = (N + L - 1) // L
                segs    = segs.repeat(1, repeats)[:, :N]
            raw_pwr = segs.abs().pow(2).mean(dim=1, keepdim=True).clamp(min=1e-12)
            noise   = segs * (n_pwr / raw_pwr).sqrt()
        else:
            noise = ((torch.randn(B, N, device=device)
                      + 1j * torch.randn(B, N, device=device))
                     * (n_pwr / 2).sqrt())

        iq += noise

        # Re-normalise to unit power — matches evaluate.py and OnnxClassifier.cpp
        pwr = iq.abs().pow(2).mean(dim=1, keepdim=True).clamp(min=1e-8)
        iq  = iq / pwr.sqrt()

        X_aug = torch.stack([iq.real, iq.imag], dim=1)
        mask  = (torch.rand(B, device=device) < self.p).view(B, 1, 1)
        return torch.where(mask, X_aug, X)


# ── Training loop ─────────────────────────────────────────────────────────────

class DomainDataset(torch.utils.data.Dataset):
    """Wraps a Subset/Dataset and appends a domain label (0=synthetic, 1=real)."""
    def __init__(self, subset, domain_tensor: torch.Tensor):
        self.subset = subset
        self.domain = domain_tensor

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx):
        batch = self.subset[idx]
        global_idx = self.subset.indices[idx]
        return batch[0], batch[1], self.domain[global_idx]


def train(model:          nn.Module,
          loader:         DataLoader,
          val_loader:     DataLoader,
          device:         torch.device,
          epochs:         int,
          lr:             float = 1e-3,
          use_mixup:      bool  = True,
          warmup_epochs:  int   = 5,
          ckpt_path:      str | None = None,
          class_weights:  torch.Tensor | None = None,
          focal_gamma:    float | None = None,
          label_smoothing: float = 0.05,
          sgdr_t0:        int   = 0,
          augment:        IqAugment | None = None,
          family_aux_weight: float = 0.0,
          family_proj:    torch.Tensor | None = None,
          dann_weight:    float = 0.0,
          dann_alpha_max: float = 1.0) -> nn.Module:
    """
    Train model with:
      - AdamW optimiser, gradient clipping
      - Linear warmup + cosine annealing (or SGDR with sgdr_t0 restart period)
      - Optional focal loss + label smoothing
      - Online IQ channel augmentation (CFO, IQ imbalance, AWGN) if augment given
      - Mixup augmentation after warmup
      - Per-class val_acc tracking; checkpoint on composite score (val + min_class)
    """
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    if sgdr_t0 > 0:
        # Cosine annealing with warm restarts (SGDR) — resets LR every T_0 epochs
        sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=sgdr_t0, T_mult=1, eta_min=1e-6)
        print(f"  LR schedule: SGDR T_0={sgdr_t0} (warm restart every {sgdr_t0} epochs)")
    else:
        # Linear warmup then single cosine decay
        def lr_lambda(ep):
            if ep < warmup_epochs:
                return float(ep + 1) / warmup_epochs
            prog = (ep - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * prog))
        sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    w = class_weights.to(device) if class_weights is not None else None
    if focal_gamma is not None:
        crit = FocalLoss(gamma=focal_gamma, weight=w, label_smoothing=label_smoothing)
        print(f"  Loss: FocalLoss(γ={focal_gamma}, label_smoothing={label_smoothing})")
    else:
        crit = nn.CrossEntropyLoss(weight=w, label_smoothing=label_smoothing)
        print(f"  Loss: CrossEntropy(label_smoothing={label_smoothing})")

    # Auxiliary family head: fixed projection (no learned params), just
    # sum logits within each family. Acts as a regularisation signal that
    # pushes the model to separate modulation families before fine classes.
    fam_proj = None
    if family_aux_weight > 0.0 and family_proj is not None:
        fam_proj = family_proj.to(device)
        print(f"  Family aux loss weight: {family_aux_weight} "
              f"({fam_proj.shape[1]} families)")
    best_state  = None
    # Use the model's actual output dimension, not the first-batch max label.
    # The first-batch approach can under-count when rare classes don't appear
    # in batch 0 (especially with WeightedRandomSampler boosting other classes).
    num_classes = next(p for p in reversed(list(model.parameters()))
                       if p.ndim == 2).shape[0]

    if augment is not None:
        augment = augment.to(device).train()

    use_dann = dann_weight > 0.0 and hasattr(model, "forward_dann")

    def eval_per_acc(m: nn.Module) -> torch.Tensor:
        m.eval()
        per_correct = torch.zeros(num_classes)
        per_total   = torch.zeros(num_classes)
        with torch.no_grad():
            for X_b, y_b, *_ in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                preds = m(X_b).argmax(dim=1).cpu()
                y_cpu = y_b.cpu()
                for c in range(num_classes):
                    mask = y_cpu == c
                    per_correct[c] += (preds[mask] == c).sum()
                    per_total[c]   += mask.sum()
        return per_correct / per_total.clamp(min=1)

    # Floor checkpoint selection at the model's score BEFORE any fine-tuning.
    # Without this, best_score started at 0.0 and a fine-tune step that never
    # beat its own starting point would still overwrite ckpt_path with a
    # WORSE model the moment epoch 1 scored > 0.0 -- silently regressing
    # --resume chains step over step (confirmed live: a 9-step family
    # fine-tune chain dropped 53.3% -> 51.7% -> 49.7% overall holdout
    # accuracy across its first two steps, with the boosted family itself
    # getting worse each time). Seeding best_state here also guarantees
    # ckpt_path always gets written, even if every epoch underperforms —
    # a failed fine-tune step is now a no-op, not a regression.
    _per_acc0  = eval_per_acc(model)
    best_score = 0.7 * _per_acc0.mean().item() + 0.3 * _per_acc0.min().item()
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    print(f"  Starting score (pre-finetune): {best_score:.3f}")
    if ckpt_path:
        torch.save(best_state, ckpt_path)

    for epoch in range(1, epochs + 1):
        model.train()
        if augment is not None:
            augment.train()
        train_loss = 0.0
        # DANN reversal strength: 0 → dann_alpha_max following Ganin et al. schedule
        p = (epoch - 1) / max(1, epochs - 1)
        dann_alpha = dann_alpha_max * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)

        for batch in tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            X_b, y_b = batch[0].to(device), batch[1].to(device)
            d_b = batch[2].to(device) if len(batch) > 2 else None
            if augment is not None:
                X_b = augment(X_b)
            opt.zero_grad()
            if use_mixup and epoch > warmup_epochs:
                X_b, y_a, y_b2, lam = mixup_batch(X_b, y_b)
                logits = model(X_b)
                loss   = mixup_loss(crit, logits, y_a, y_b2, lam)
            elif use_dann and d_b is not None:
                logits, dom_logits = model.forward_dann(X_b, alpha=dann_alpha)
                loss = crit(logits, y_b)
                loss = loss + dann_weight * F.cross_entropy(dom_logits, d_b)
            else:
                logits = model(X_b)
                loss   = crit(logits, y_b)
            if fam_proj is not None:
                # family logits = sum of class logits within each family
                fam_logits = logits @ fam_proj          # (B, F)
                fam_labels = (y_b.unsqueeze(1) == torch.arange(
                    logits.shape[1], device=device).unsqueeze(0)
                ).float() @ fam_proj                    # (B, F) one-hot family
                fam_labels = fam_labels.argmax(dim=1)   # (B,) family index
                loss = loss + family_aux_weight * nn.functional.cross_entropy(
                    fam_logits, fam_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * len(y_b)

        train_loss /= len(loader.dataset)
        sched.step()

        if augment is not None:
            augment.eval()
        per_acc  = eval_per_acc(model)
        val_acc  = per_acc.mean().item()
        min_acc  = per_acc.min().item()
        cur_lr   = opt.param_groups[0]["lr"]
        score    = 0.7 * val_acc + 0.3 * min_acc
        print(f"  Epoch {epoch:3d}  loss={train_loss:.4f}  val_acc={val_acc:.3f}  "
              f"min_class={min_acc:.3f}  score={score:.3f}  lr={cur_lr:.2e}")

        # Checkpoint on composite score so we never sacrifice the worst class
        if score > best_score:
            best_score = score
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if ckpt_path:
                torch.save(best_state, ckpt_path)

    print(f"\nBest composite score: {best_score:.3f}")
    if best_state:
        model.load_state_dict(best_state)
    return model


# ── Confusion report ──────────────────────────────────────────────────────────

def confusion_report(model:       nn.Module,
                     loader:      DataLoader,
                     class_names: list[str],
                     device:      torch.device) -> None:
    from sklearn.metrics import classification_report, confusion_matrix  # type: ignore
    import pandas as pd

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            X_b, y_b = batch[0], batch[1]
            preds = model(X_b.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_b.numpy())

    print("\n── Per-class Report ──")
    print(classification_report(all_labels, all_preds,
                                  target_names=class_names, digits=3))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train AMR CNN — supports RadioML, synthetic, npz, HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Data sources (at least one required) ──────────────────────────────────
    src = ap.add_argument_group("data sources (use one or more)")
    src.add_argument("--data",       metavar="HDF5",
                     help="RadioML .hdf5 file path")
    src.add_argument("--version",    default="2016", choices=["2016","2018"],
                     help="RadioML version when using --data (default 2016)")
    src.add_argument("--npz",        nargs="+", metavar="FILE",
                     help="One or more .npz files from 'python -m datasets generate'")
    src.add_argument("--hf-dataset", metavar="ID",
                     help="HuggingFace dataset ID (e.g. user/radio-amr)")
    src.add_argument("--synthetic",  type=int, metavar="N",
                     help="Generate N synthetic examples per (mod, SNR) bucket")
    src.add_argument("--synthetic-len", type=int, default=1024,
                     help="Samples per synthetic IQ window (default 1024)")

    # ── Filters ───────────────────────────────────────────────────────────────
    ap.add_argument("--snr-min",       type=float, default=-20)
    ap.add_argument("--max-per-class", type=int,   default=10000)
    ap.add_argument("--family",        default=None, metavar="NAME",
                    help="Train a specialist on only this family's classes "
                         "(see family_map.py, e.g. QAM, PSK, FSK, AM). "
                         "Mutually exclusive with --router.")
    ap.add_argument("--router",        action="store_true",
                    help="Train a coarse family router: labels become the "
                         "9 family names instead of fine-grained classes. "
                         "Mutually exclusive with --family.")

    # ── Model ─────────────────────────────────────────────────────────────────
    ap.add_argument("--model",   default="resnet",
                    choices=["cnn","resnet","fusion"],
                    help="Model architecture (default resnet)")
    ap.add_argument("--channels",  type=int,   default=128,
                    help="ResNet channel width (default 128)")
    ap.add_argument("--n-blocks",  type=int,   default=12,
                    help="Number of residual blocks (default 12)")
    ap.add_argument("--n-fft",     type=int,   default=512,
                    help="STFT n_fft for RadioFusion spectrogram path (default 512 → 390 Hz/bin at 200 kHz)")
    ap.add_argument("--family-aux", type=float, default=0.0, metavar="WEIGHT",
                    help="Auxiliary family-classification loss weight (e.g. 0.2). "
                         "Adds a linear family head on pooled features; encourages "
                         "the model to organise representations by modulation family "
                         "before fine-grained class discrimination.")
    ap.add_argument("--dann",      type=float, default=0.0, metavar="WEIGHT",
                    help="DANN domain adversarial loss weight (0=disabled). "
                         "Drives the shared representation to be domain-invariant "
                         "between synthetic and real data via gradient reversal.")
    ap.add_argument("--dann-alpha", type=float, default=1.0, metavar="ALPHA",
                    help="Max gradient-reversal strength for DANN (default 1.0). "
                         "Linearly scheduled from 0→ALPHA over training epochs.")
    ap.add_argument("--dann-real", nargs="+", default=[], metavar="FILE",
                    help="NPZ files (subset of --npz) that are real captures "
                         "(domain=1). All other --npz files are synthetic (domain=0). "
                         "Required when --dann > 0.")
    ap.add_argument("--epochs",  type=int,   default=50)
    ap.add_argument("--batch",   type=int,   default=256)
    ap.add_argument("--lr",      type=float, default=1e-3)
    ap.add_argument("--cuda",      action="store_true")
    ap.add_argument("--out",       default="models/classifier.onnx", metavar="ONNX")
    ap.add_argument("--resume",    default=None, metavar="PT",
                    help="Fine-tune from an existing checkpoint (.pt)")
    ap.add_argument("--focal",     type=float, default=None, metavar="GAMMA",
                    help="Use focal loss with this γ (e.g. 2.0). Default: cross-entropy.")
    ap.add_argument("--label-smoothing", type=float, default=0.05, metavar="EPS",
                    help="Label smoothing ε for loss function (default 0.05).")
    ap.add_argument("--sgdr-t0",   type=int,   default=0, metavar="EPOCHS",
                    help="Cosine annealing warm restart period T_0 (0 = disabled, use single decay).")
    ap.add_argument("--boost-hard", type=float, default=None, metavar="FACTOR",
                    help="Oversample PSK/QAM hard classes by FACTOR via WeightedRandomSampler.")
    ap.add_argument("--boost-qam", type=float, default=None, metavar="FACTOR",
                    help="(deprecated) Use --boost-hard instead.")
    ap.add_argument("--boost-only", default=None, metavar="NAME[,NAME...]",
                    help="With --boost-hard, restrict the boost to exactly these "
                         "class/family names (comma-separated, exact match) instead "
                         "of the hardcoded HARD_CLASSES set. For targeting one "
                         "specific weak family without touching others' sampling.")
    ap.add_argument("--augment",   action="store_true",
                    help="Apply online IQ channel augmentation during training "
                         "(phase, IQ imbalance, AWGN). Strongly recommended.")
    ap.add_argument("--no-mixup",  action="store_true",
                    help="Disable mixup. For raw IQ, linearly superimposing two "
                         "different modulations' samples (with a blended label) "
                         "can confuse the classifier rather than regularize it.")
    ap.add_argument("--val-npz",   default=None, metavar="FILE",
                    help="External NPZ to use as the validation set (e.g. the holdout). "
                         "Checkpoints will be saved on best performance against this set, "
                         "so the model actually optimises for the right distribution.")
    ap.add_argument("--noise-npz", default=None, metavar="FILE",
                    help="NPZ of real captured noise windows from capture_noise.py. "
                         "When --augment is also set, replaces synthetic AWGN with real "
                         "PlutoSDR hardware noise during IQ augmentation.")
    args = ap.parse_args()

    if args.family and args.router:
        ap.error("--family and --router are mutually exclusive")
    if args.family:
        from family_map import CLASS_FAMILY
        valid = sorted(set(CLASS_FAMILY.values()))
        if args.family.upper() not in valid:
            ap.error(f"--family '{args.family}' not recognised. Valid: {valid}")

    device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    if args.cuda and not torch.cuda.is_available():
        print("  [warn] CUDA requested but not available, using CPU")

    # ── Load / generate data ──────────────────────────────────────────────────
    # Pass dann_real_set into build_tensors so domain labels are derived from
    # IqSample.source_file during the initial load — no second NPZ reload needed.
    dann_real_set: set | None = None
    if args.dann > 0.0 and args.dann_real and args.npz:
        dann_real_set = set(args.dann_real)
        print(f"  DANN: real files = {args.dann_real}")
    X, y, class_names, domain_tensor = build_tensors(args, dann_real_set=dann_real_set)
    input_len   = X.shape[-1]
    num_classes = len(class_names)
    print(f"  Samples: {len(X)}  Input length: {input_len}  Classes: {num_classes}")

    # ── DANN domain label summary ─────────────────────────────────────────────
    if dann_real_set is not None:
        if domain_tensor is not None:
            n_real = int((domain_tensor == 1).sum())
            print(f"  Domain labels: {len(X) - n_real:,} synthetic, {n_real:,} real")
        else:
            print("  [warn] domain labels unavailable — DANN disabled")

    # ── Split ─────────────────────────────────────────────────────────────────
    # Include domain labels as batch[2] so the DANN training loop sees them.
    # Without this, d_b is always None and DANN loss is silently skipped.
    dataset = TensorDataset(X, y) if domain_tensor is None \
              else TensorDataset(X, y, domain_tensor)
    import torch as _torch
    _use_gpu = args.cuda and _torch.cuda.is_available()
    kw = dict(batch_size=args.batch,
              num_workers=4 if _use_gpu else 0,
              pin_memory=_use_gpu)

    if args.val_npz:
        # Use the external holdout as the validation set so checkpoints are
        # optimised for the actual target distribution, not the training split.
        print(f"  Using external val set: {args.val_npz}")
        vd  = np.load(args.val_npz)
        vX  = torch.from_numpy(vd["X"]).float()
        vy_raw = vd["y"]
        # Remap holdout labels to match this run's class_names order. Router
        # mode maps each holdout mod name to its family name first.
        h_classes = list(vd["classes"])
        if args.router:
            from family_map import CLASS_FAMILY
            h_classes = [CLASS_FAMILY.get(c, c) for c in h_classes]
        label_map = {h_classes[i]: class_names.index(h_classes[i])
                     for i in range(len(h_classes)) if h_classes[i] in class_names}
        vy = torch.tensor([label_map.get(h_classes[int(l)], -1) for l in vy_raw],
                          dtype=torch.long)
        keep = vy >= 0
        vX, vy = vX[keep], vy[keep]
        val_ds   = TensorDataset(vX, vy)
        # Use all training data for training (no split wasted on val)
        n_test   = int(len(dataset) * 0.10)
        n_train  = len(dataset) - n_test
        train_ds, test_ds = random_split(
            dataset, [n_train, n_test],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        n_val   = int(len(dataset) * 0.15)
        n_test  = int(len(dataset) * 0.15)
        n_train = len(dataset) - n_val - n_test
        train_ds, val_ds, test_ds = random_split(
            dataset, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )

    # Hard-class oversampling — boosts PSK variants, mid-order QAM, and other
    # known weak classes (0% on holdout) via WeightedRandomSampler.
    HARD_CLASSES = {
        '8PSK', '16PSK', '32PSK', 'PSK31', 'P25_PHASE2',
        'QAM16', 'QAM32', 'QAM64',
        'AM_DSB', 'DSTAR', 'TETRA', 'NAVTEX', 'RTTY',
        'P25_C4FM', 'NXDN', 'DTMF', 'GFSK', 'FSK', 'TONE', '4FSK', '8FSK',
    }
    boost_factor = args.boost_hard or args.boost_qam  # accept either flag
    # Optionally wrap train_ds with domain labels for DANN
    dann_train_ds = (DomainDataset(train_ds, domain_tensor)
                     if domain_tensor is not None else train_ds)

    if boost_factor and boost_factor > 1.0:
        if args.boost_only:
            only_names = {s.strip() for s in args.boost_only.split(",")}
            hard_idx   = {i for i, n in enumerate(class_names) if n in only_names}
        else:
            hard_idx   = {i for i, n in enumerate(class_names) if n in HARD_CLASSES
                          or "QAM" in n or "PSK" in n}
        train_labels = y[train_ds.indices]
        sample_w = torch.where(
            torch.tensor([int(l) in hard_idx for l in train_labels]),
            torch.tensor(float(boost_factor)),
            torch.ones(len(train_ds)),
        )
        sampler = WeightedRandomSampler(sample_w, len(train_ds), replacement=True)
        train_loader = DataLoader(dann_train_ds, sampler=sampler, **kw)
        boosted = sorted(class_names[i] for i in hard_idx)
        print(f"  Hard-class boost ×{boost_factor:.1f}: {boosted}")
    else:
        train_loader = DataLoader(dann_train_ds, shuffle=True, **kw)

    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kw)

    # ── Build model ───────────────────────────────────────────────────────────
    if args.model == "cnn":
        model = RadioCNN(num_classes=num_classes, input_len=input_len)
    elif args.model == "resnet":
        model = RadioResNet(num_classes=num_classes, input_len=input_len,
                            channels=args.channels, n_blocks=args.n_blocks)
        print(f"  RadioResNet: channels={args.channels}, n_blocks={args.n_blocks}")
    else:
        model = RadioFusion(num_classes=num_classes, input_len=input_len,
                            n_fft=args.n_fft)
        print(f"  RadioFusion: IQ(channels=128,n_blocks=8) + STFT(n_fft={args.n_fft})")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    if args.resume:
        print(f"  Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        current = model.state_dict()
        # When resuming a RadioResNet checkpoint (keys: stem.*, blocks.*, head.*)
        # into RadioFusion (which wraps RadioResNet as iq_path.*), the key prefix
        # mismatch means zero weights transfer without remapping. Detect this by
        # checking whether any raw checkpoint key exists in the current model.
        if args.model == 'fusion' and not any(k in current for k in ckpt):
            print("  Remapping RadioResNet checkpoint keys → iq_path.* for RadioFusion")
            ckpt = {f'iq_path.{k}': v for k, v in ckpt.items()}
        # Keep only layers whose shape matches — mismatch happens when num_classes
        # differs between the checkpoint and the current model (e.g. 24→28 classes).
        compatible = {k: v for k, v in ckpt.items()
                      if k in current and current[k].shape == v.shape}
        reinit = sorted(set(current) - set(compatible))
        model.load_state_dict(compatible, strict=False)
        print(f"  Loaded {len(compatible)}/{len(current)} layers.")
        if reinit:
            print(f"  Re-initialised (shape mismatch): {reinit}")

    # ── Class weights (inverse frequency, capped at 10×) ─────────────────────
    train_y = y[train_ds.indices]
    counts  = torch.bincount(train_y, minlength=num_classes).float()
    weights = 1.0 / (counts + 1.0)
    weights = (weights / weights.mean()).clamp(max=10.0)
    print("  Class weights (top 5 heaviest):")
    top = weights.argsort(descending=True)[:5]
    for i in top:
        print(f"    {class_names[i]:15s} {weights[i]:.2f}x  ({int(counts[i])} samples)")

    # ── Train ─────────────────────────────────────────────────────────────────
    noise_bank = None
    if args.augment and args.noise_npz:
        nd = np.load(args.noise_npz)
        noise_bank = torch.from_numpy(nd["X"]).float()  # (M, 2, L)
        print(f"  Real noise bank: {len(noise_bank):,} segments from {args.noise_npz}")
    augment = IqAugment(noise_bank=noise_bank) if args.augment else None
    ckpt_path = args.out.replace(".onnx", ".best.pt")

    family_proj = None
    if args.family_aux > 0.0:
        from family_map import family_projection_matrix
        family_proj = family_projection_matrix(class_names)
        print(f"  Family aux loss enabled (weight={args.family_aux})")

    if args.dann > 0.0:
        print(f"  DANN: weight={args.dann}, alpha_max={args.dann_alpha}")
    model = train(model, train_loader, val_loader, device,
                  epochs=args.epochs, lr=args.lr,
                  use_mixup=not args.no_mixup, warmup_epochs=min(8, args.epochs // 10),
                  ckpt_path=ckpt_path, class_weights=weights,
                  focal_gamma=args.focal,
                  label_smoothing=args.label_smoothing,
                  sgdr_t0=args.sgdr_t0,
                  augment=augment,
                  family_aux_weight=args.family_aux,
                  family_proj=family_proj,
                  dann_weight=args.dann,
                  dann_alpha_max=args.dann_alpha)

    # ── Test + report ─────────────────────────────────────────────────────────
    print("\n── Test Set Evaluation ──")
    confusion_report(model, test_loader, class_names, device)

    # ── Export ONNX ───────────────────────────────────────────────────────────
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    input_shape = (1, 2, input_len)
    export_onnx(model.cpu(), str(out), input_shape=input_shape)

    # Save class names alongside the model (needed by C++ inference layer)
    classes_json = out.with_suffix(".classes.json")
    import json
    with open(classes_json, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"Class names saved → {classes_json}")


if __name__ == "__main__":
    main()

# ========================================================================
# End of file — OpenRFStack
# Subject to Personal Use License
# https://github.com/OpenRFStack
# ========================================================================
