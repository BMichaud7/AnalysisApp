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
CNN model for automatic modulation recognition (AMR).

Architecture inspired by:
  - Panoradio HF deep-learning paper (28-layer CNN)
  - "Convolutional Radio Modulation Recognition Networks" (O'Shea et al. 2016)
  - "Over-the-Air Deep Learning Based Radio Signal Classification" (O'Shea et al. 2018)

Input:  (batch, 2, N) — channel 0 = I, channel 1 = Q, N samples per window
Output: class logits (batch, num_classes)

Two variants:
  RadioCNN    — lightweight, suitable for deployment (ONNX → C++)
  RadioResNet — deeper residual network, higher accuracy
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── RadioCNN (O'Shea-style, fast inference) ───────────────────────────────────

class RadioCNN(nn.Module):
    """
    7-block 1D CNN on raw IQ.
    Input:  (B, 2, N)
    Output: (B, num_classes)
    """

    def __init__(self, num_classes: int = 11, input_len: int = 128):
        super().__init__()
        self.input_len = input_len

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),

            # Block 2
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),

            # Block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),

            # Block 4
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(4),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, N) — I on channel 0, Q on channel 1
        return self.classifier(self.features(x))


# ── RadioResNet (residual, higher accuracy) ───────────────────────────────────

class _SEBlock1d(nn.Module):
    """Squeeze-and-Excitation channel attention — cheap but effective."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x).unsqueeze(-1)


class _ResBlock1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm1d(channels),
        )
        self.se      = _SEBlock1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.relu    = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.se(self.dropout(self.block(x))) + x)


class _MultiScaleStem(nn.Module):
    """Parallel convolutions at three kernel sizes, concatenated along channels."""
    def __init__(self, channels: int):
        super().__init__()
        c = channels // 3
        r = channels - 2 * c   # absorb rounding remainder in the wide branch
        self.b3  = nn.Conv1d(2, c, kernel_size=3,  padding=1)
        self.b7  = nn.Conv1d(2, c, kernel_size=7,  padding=3)
        self.b15 = nn.Conv1d(2, r, kernel_size=15, padding=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.b3(x), self.b7(x), self.b15(x)], dim=1)


class RadioResNet(nn.Module):
    """
    Residual CNN for AMR — closer to the 28-layer panoradio architecture.
    Input:  (B, 2, N)
    Output: (B, num_classes)
    """

    def __init__(self, num_classes: int = 11, input_len: int = 128,
                 channels: int = 128, n_blocks: int = 12):
        super().__init__()
        self.stem = nn.Sequential(
            _MultiScaleStem(channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[_ResBlock1d(channels) for _ in range(n_blocks)]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        return self.head(x)


# ── IQ + spectrogram fusion model ────────────────────────────────────────────
# Combines raw IQ path + STFT magnitude spectrogram path.
# Useful for signals where spectral shape is the most discriminative feature.

class RadioFusion(nn.Module):
    """
    Dual-path model:
      Path A: RadioResNet on raw IQ (time-domain)
      Path B: ResNet-style on STFT magnitude spectrogram (frequency-domain)
    Outputs are concatenated and fed to a final classifier.
    """

    def __init__(self, num_classes: int = 24, input_len: int = 1024,
                 n_fft: int = 64):
        super().__init__()
        self.n_fft     = n_fft
        self.hop_len   = n_fft // 2

        # IQ path
        self.iq_path   = RadioResNet(num_classes=256, input_len=input_len,
                                     channels=128, n_blocks=6)
        self.iq_path.head = nn.Identity()   # remove final classifier

        # Spectrogram path
        spec_time = input_len // self.hop_len
        self.spec_path = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128*16, 256),
            nn.ReLU(inplace=True),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(128 + 256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def _spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, N) → complex → STFT magnitude (B, 1, F, T)
        iq = torch.complex(x[:, 0, :], x[:, 1, :])
        window = torch.hann_window(self.n_fft, device=x.device)
        spec = torch.stft(
            iq.reshape(-1, iq.shape[-1]),
            n_fft=self.n_fft, hop_length=self.hop_len,
            win_length=self.n_fft, window=window,
            return_complex=True,
        )
        mag = spec.abs().unsqueeze(1)   # (B, 1, F, T)
        return (mag + 1e-6).log()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        iq_feat   = self.iq_path(x)          # (B, 128)
        spec      = self._spectrogram(x)     # (B, 1, F, T)
        spec_feat = self.spec_path(spec)     # (B, 256)
        fused     = torch.cat([iq_feat, spec_feat], dim=1)
        return self.fusion(fused)


# ── ONNX export helper ────────────────────────────────────────────────────────

def export_onnx(model: nn.Module,
                out_path: str,
                input_shape: tuple[int, int, int] = (1, 2, 128),
                opset: int = 17) -> None:
    """
    Export a trained model to ONNX for deployment in C++ via ONNX Runtime.

    The exported model accepts (batch, 2, N) float32 input and returns
    (batch, num_classes) logit tensor. The C++ AnalysisService loads this
    model with onnxruntime::InferenceSession.

    Example C++ usage:
        Ort::Env env;
        Ort::Session session(env, "model.onnx", Ort::SessionOptions{});
        // feed (1, 2, 128) float32 tensor → get logits → argmax → class
    """
    import onnx  # type: ignore
    import onnxruntime as ort  # type: ignore

    model.eval()
    dummy = torch.randn(*input_shape)

    torch.onnx.export(
        model, dummy, out_path,
        export_params=True,
        opset_version=opset,
        dynamo=False,
        input_names=["iq_input"],
        output_names=["logits"],
        dynamic_axes={"iq_input": {0: "batch"}, "logits": {0: "batch"}},
    )

    # Validate
    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)

    sess = ort.InferenceSession(out_path, providers=["CUDAExecutionProvider",
                                                      "CPUExecutionProvider"])
    out  = sess.run(None, {"iq_input": dummy.numpy()})
    assert out[0].shape[0] == input_shape[0], "ONNX output shape mismatch"

    print(f"ONNX model exported and validated → {out_path}")
    print(f"  Input:  {input_shape}")
    print(f"  Output: {out[0].shape}")

# ========================================================================
# End of file — OpenRFStack
# Subject to Personal Use License
# https://github.com/OpenRFStack
# ========================================================================
