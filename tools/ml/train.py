#!/usr/bin/env python3
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
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
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


def build_tensors(args) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Load all requested sources and merge into a single tensor dataset."""
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

    print(f"Total samples: {len(all_samples)}")
    X, y, cn = samples_to_tensors(all_samples, class_names)
    print(f"  Input shape: {X.shape}  Classes: {len(cn)}")
    return X, y, cn


# ── Mixup augmentation ────────────────────────────────────────────────────────

def mixup_batch(X: torch.Tensor, y: torch.Tensor, alpha: float = 0.3):
    """Beta-distributed linear interpolation between pairs of samples."""
    lam  = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(len(X), device=X.device)
    return lam * X + (1 - lam) * X[perm], y, y[perm], lam


def mixup_loss(crit, logits, y_a, y_b, lam):
    return lam * crit(logits, y_a) + (1 - lam) * crit(logits, y_b)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(model:       nn.Module,
          loader:      DataLoader,
          val_loader:  DataLoader,
          device:      torch.device,
          epochs:      int,
          lr:          float = 1e-3,
          use_mixup:   bool  = True,
          warmup_epochs: int = 5,
          ckpt_path:   str | None = None,
          class_weights: torch.Tensor | None = None) -> nn.Module:

    opt  = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Linear warmup then cosine annealing
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return float(ep + 1) / warmup_epochs
        prog = (ep - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    w = class_weights.to(device) if class_weights is not None else None
    crit  = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
    best_acc   = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X_b, y_b in tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            X_b, y_b = X_b.to(device), y_b.to(device)
            opt.zero_grad()
            if use_mixup and epoch > warmup_epochs:
                X_b, y_a, y_b2, lam = mixup_batch(X_b, y_b)
                logits = model(X_b)
                loss   = mixup_loss(crit, logits, y_a, y_b2, lam)
            else:
                logits = model(X_b)
                loss   = crit(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * len(y_b)

        train_loss /= len(loader.dataset)
        sched.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                preds = model(X_b).argmax(dim=1)
                correct += (preds == y_b).sum().item()
                total   += len(y_b)

        val_acc = correct / total
        cur_lr  = opt.param_groups[0]["lr"]
        print(f"  Epoch {epoch:3d}  loss={train_loss:.4f}  val_acc={val_acc:.3f}  "
              f"lr={cur_lr:.2e}")

        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if ckpt_path:
                torch.save(best_state, ckpt_path)

    print(f"\nBest val accuracy: {best_acc:.3f}")
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
        for X_b, y_b in loader:
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

    # ── Model ─────────────────────────────────────────────────────────────────
    ap.add_argument("--model",   default="resnet",
                    choices=["cnn","resnet","fusion"],
                    help="Model architecture (default resnet)")
    ap.add_argument("--epochs",  type=int,   default=50)
    ap.add_argument("--batch",   type=int,   default=256)
    ap.add_argument("--lr",      type=float, default=1e-3)
    ap.add_argument("--cuda",    action="store_true")
    ap.add_argument("--out",     default="models/classifier.onnx", metavar="ONNX")
    args = ap.parse_args()

    device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    if args.cuda and not torch.cuda.is_available():
        print("  [warn] CUDA requested but not available, using CPU")

    # ── Load / generate data ──────────────────────────────────────────────────
    X, y, class_names = build_tensors(args)
    input_len   = X.shape[-1]
    num_classes = len(class_names)
    print(f"  Samples: {len(X)}  Input length: {input_len}  Classes: {num_classes}")

    # ── Split ─────────────────────────────────────────────────────────────────
    dataset  = TensorDataset(X, y)
    n_val    = int(len(dataset) * 0.15)
    n_test   = int(len(dataset) * 0.15)
    n_train  = len(dataset) - n_val - n_test
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )
    import torch as _torch
    _use_gpu = args.cuda and _torch.cuda.is_available()
    kw = dict(batch_size=args.batch,
              num_workers=4 if _use_gpu else 0,
              pin_memory=_use_gpu)
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kw)

    # ── Build model ───────────────────────────────────────────────────────────
    if args.model == "cnn":
        model = RadioCNN(num_classes=num_classes, input_len=input_len)
    elif args.model == "resnet":
        model = RadioResNet(num_classes=num_classes, input_len=input_len,
                            channels=160, n_blocks=12)
    else:
        model = RadioFusion(num_classes=num_classes, input_len=input_len)

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

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
    ckpt_path = args.out.replace(".onnx", ".best.pt")
    model = train(model, train_loader, val_loader, device,
                  epochs=args.epochs, lr=args.lr,
                  use_mixup=True, warmup_epochs=min(5, args.epochs // 10),
                  ckpt_path=ckpt_path, class_weights=weights)

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
