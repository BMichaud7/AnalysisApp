#!/usr/bin/env python3
"""Standalone ONNX export for v12 checkpoint (RadioFusion, 47 classes)."""
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model import RadioFusion, export_onnx

CKPT   = "models/amr_cnn_v12_47class.best.pt"
OUT    = "models/amr_cnn_v12_47class.onnx"
N_FFT  = 256
N_CLS  = 47
INPUT_LEN = 1024

model = RadioFusion(num_classes=N_CLS, input_len=INPUT_LEN, n_fft=N_FFT)
ckpt  = torch.load(CKPT, map_location="cpu", weights_only=True)
missing, unexpected = model.load_state_dict(ckpt, strict=False)
if missing:
    print(f"  Missing keys (new buffers expected): {missing}")
if unexpected:
    print(f"  Unexpected keys: {unexpected}")

export_onnx(model.cpu(), OUT, input_shape=(1, 2, INPUT_LEN))
