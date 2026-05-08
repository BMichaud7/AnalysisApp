#!/usr/bin/env python3
"""
Train an AMR classifier on RadioML 2016.10a or 2018.01a.

Usage:
    python train.py --data  /path/to/RML2016.10a.hdf5 \
                   --model  resnet \
                   --epochs 50 \
                   --out    models/radioml2016_resnet.onnx

    python train.py --data  /path/to/RML2018.01a.hdf5 \
                   --model  fusion \
                   --epochs 100 \
                   --cuda \
                   --out    models/radioml2018_fusion.onnx

After training the ONNX model can be dropped into AnalysisApp and loaded
by the C++ OnnxClassifier (include/OnnxClassifier.hpp).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "test_harness"))
from datasets import load_radioml_2016, load_radioml_2018, RADIOML_2016_MODS, RADIOML_2018_MODS
from model import RadioCNN, RadioResNet, RadioFusion, export_onnx


# ── Dataset builders ──────────────────────────────────────────────────────────

def build_radioml_tensors(path: str,
                           version: str = "2016",
                           snr_min: float = -20) \
        -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """
    Returns (X, y, class_names) where X is (N, 2, L) float32 and y is (N,) int64.
    """
    if version == "2016":
        samples    = load_radioml_2016(path, snr_min_db=snr_min, max_per_class=10000)
        class_names = RADIOML_2016_MODS
    else:
        samples    = load_radioml_2018(path, snr_min_db=snr_min, max_per_class=5000)
        class_names = RADIOML_2018_MODS

    # Build label map
    label_map = {m: i for i, m in enumerate(class_names)}

    X_list, y_list = [], []
    for s in samples:
        iq = s.iq
        I  = iq.real.astype(np.float32)
        Q  = iq.imag.astype(np.float32)
        x  = np.stack([I, Q], axis=0)   # (2, L)
        label = label_map.get(s.ground_truth)
        if label is None:
            continue
        X_list.append(x)
        y_list.append(label)

    X = torch.from_numpy(np.stack(X_list, axis=0))
    y = torch.tensor(y_list, dtype=torch.long)
    return X, y, class_names


# ── Training loop ─────────────────────────────────────────────────────────────

def train(model:      nn.Module,
          loader:     DataLoader,
          val_loader: DataLoader,
          device:     torch.device,
          epochs:     int,
          lr:         float = 1e-3) -> nn.Module:

    opt   = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.CrossEntropyLoss()
    best_acc  = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for X_b, y_b in tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            X_b, y_b = X_b.to(device), y_b.to(device)
            opt.zero_grad()
            logits = model(X_b)
            loss   = crit(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * len(y_b)

        train_loss /= len(loader.dataset)
        sched.step()

        # Validate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                preds = model(X_b).argmax(dim=1)
                correct += (preds == y_b).sum().item()
                total   += len(y_b)

        val_acc = correct / total
        print(f"  Epoch {epoch:3d}  loss={train_loss:.4f}  val_acc={val_acc:.3f}")

        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

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
    ap = argparse.ArgumentParser(description="Train AMR CNN on RadioML dataset")
    ap.add_argument("--data",     required=True, metavar="HDF5",
                    help="Path to RadioML .hdf5 file")
    ap.add_argument("--version",  default="2016", choices=["2016","2018"],
                    help="RadioML version (default 2016)")
    ap.add_argument("--model",    default="resnet",
                    choices=["cnn","resnet","fusion"],
                    help="Model architecture (default resnet)")
    ap.add_argument("--epochs",   type=int, default=50)
    ap.add_argument("--batch",    type=int, default=256)
    ap.add_argument("--lr",       type=float, default=1e-3)
    ap.add_argument("--snr-min",  type=float, default=-20)
    ap.add_argument("--cuda",     action="store_true",
                    help="Use CUDA if available")
    ap.add_argument("--out",      default="models/classifier.onnx",
                    metavar="ONNX")
    args = ap.parse_args()

    device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    if args.cuda and not torch.cuda.is_available():
        print("  [warn] CUDA requested but not available, using CPU")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading RadioML {args.version} from {args.data} …")
    X, y, class_names = build_radioml_tensors(
        args.data, version=args.version, snr_min=args.snr_min
    )
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
    kw = dict(batch_size=args.batch, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kw)

    # ── Build model ───────────────────────────────────────────────────────────
    if args.model == "cnn":
        model = RadioCNN(num_classes=num_classes, input_len=input_len)
    elif args.model == "resnet":
        model = RadioResNet(num_classes=num_classes, input_len=input_len)
    else:
        model = RadioFusion(num_classes=num_classes, input_len=input_len)

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # ── Train ─────────────────────────────────────────────────────────────────
    model = train(model, train_loader, val_loader, device,
                  epochs=args.epochs, lr=args.lr)

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
