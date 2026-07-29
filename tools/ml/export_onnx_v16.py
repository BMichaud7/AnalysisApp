#!/usr/bin/env python3
"""Standalone ONNX export for v16 checkpoint (RadioFusion, 47 classes, n_fft=512).

Uses torch.onnx.export with dynamo=True (the default in PyTorch 2.11) to handle
AdaptiveAvgPool2d whose time dimension (2 after two MaxPool2d(2) passes on T=9)
can't be exported by the legacy TorchScript exporter (model.py's export_onnx
hardcodes dynamo=False, which fails for this case).
"""
import sys
import json
import torch
import onnx          # type: ignore
import onnxruntime as ort  # type: ignore
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model import RadioFusion

CKPT      = "models/amr_cnn_v16_47class.best.pt"
OUT       = "models/amr_cnn_v16_47class.onnx"
CLASSES   = "models/amr_cnn_v16_47class.classes.json"
N_FFT     = 512
N_CLS     = 47
INPUT_LEN = 1024

SRC_CLASSES = "models/amr_cnn_v14_47class.classes.json"
with open(SRC_CLASSES) as f:
    class_names = json.load(f)
assert len(class_names) == N_CLS, f"Expected {N_CLS} classes, got {len(class_names)}"

model = RadioFusion(num_classes=N_CLS, input_len=INPUT_LEN, n_fft=N_FFT)
ckpt  = torch.load(CKPT, map_location="cpu", weights_only=True)
missing, unexpected = model.load_state_dict(ckpt, strict=False)
if missing:
    print(f"  Missing keys: {missing}")
if unexpected:
    print(f"  Unexpected keys: {unexpected}")

model.eval()
dummy = torch.randn(1, 2, INPUT_LEN)

# dynamo=True (default in 2.11) lowers AdaptiveAvgPool2d to Resize, bypassing
# the TorchScript exporter's restriction on non-factor output sizes.
onnx_prog = torch.onnx.export(
    model,
    (dummy,),
    OUT,
    input_names=["iq_input"],
    output_names=["logits"],
    dynamo=True,
)
print(f"ONNX exported → {OUT}")

# Validate with onnx checker
onnx_model = onnx.load(OUT)
onnx.checker.check_model(onnx_model)

# Validate with onnxruntime
sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
out = sess.run(None, {input_name: dummy.numpy()})
assert out[0].shape == (1, N_CLS), f"Unexpected output shape: {out[0].shape}"
print(f"  Input:  (1, 2, {INPUT_LEN})")
print(f"  Output: {out[0].shape}")
print("ONNX model validated OK.")

with open(CLASSES, "w") as f:
    json.dump(class_names, f, indent=2)
print(f"Classes saved → {CLASSES}")
