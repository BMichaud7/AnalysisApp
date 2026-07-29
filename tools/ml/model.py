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


# ── Gradient Reversal Layer (DANN domain adaptation) ─────────────────────────

class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return _GRL.apply(x, alpha)


# ── IQ + spectrogram + instantaneous-features fusion model ───────────────────

class RadioFusion(nn.Module):
    """
    Triple-path model:
      Path A: RadioResNet on raw IQ (time-domain), channels=128 n_blocks=8
              — matches v10/v11 checkpoint for weight loading
      Path B: 2D CNN on STFT log-magnitude spectrogram (frequency-domain)
      Path C: 1D CNN on instantaneous amplitude / phase / frequency (IA/IP/IF)
              — directly encodes what a human demodulator would compute;
                FM vs AM diverge in IF, PSK orders diverge in IP trajectory

    Also computes 4 higher-order cumulant features (C20, C42, C40, C4P)
    fed through a small MLP — analytically optimal for distinguishing QAM
    orders and PSK constellation sizes.

    The shared 256-dim hidden representation supports DANN domain adaptation
    via forward_dann(): a gradient-reversal layer drives the feature extractor
    to be domain-invariant between synthetic (label=0) and real (label=1) data,
    directly closing the synthetic-to-real accuracy gap.

    Weight loading from v11 checkpoints (strict=False):
      iq_path.*, spec_path.*  → loaded (shape-compatible)
      inst_path, cumulant_head, fusion_hidden, fusion_out,
      domain_classifier       → randomly initialised (new or reshaped)
    """

    IQ_CHANNELS = 128   # must match v10/v11 checkpoint
    INST_FEAT   = 64    # instantaneous features path output dim
    CUM_FEAT    = 32    # cumulant MLP output dim

    def __init__(self, num_classes: int = 24, input_len: int = 1024,
                 n_fft: int = 256):
        super().__init__()
        self.n_fft   = n_fft
        self.hop_len = n_fft // 4

        # Precompute windowed-DFT conv1d weights — avoids aten::complex during ONNX export.
        # F.conv1d with a fixed kernel is a standard ONNX MatMul and is fully supported.
        # Weight layout: (2*n_fft, 2, n_fft): first n_fft out-channels are Re, next Im.
        #   Re[k] = Σ_n (I[n]*cos_kn + Q[n]*sin_kn) * hann[n]
        #   Im[k] = Σ_n (Q[n]*cos_kn - I[n]*sin_kn) * hann[n]
        _n = torch.arange(n_fft, dtype=torch.float32)
        _k = torch.arange(n_fft, dtype=torch.float32)
        _theta = 2 * torch.pi * _k.unsqueeze(1) * _n.unsqueeze(0) / n_fft  # (n_fft, n_fft)
        _cos = torch.cos(_theta)
        _sin = torch.sin(_theta)
        _hann = torch.hann_window(n_fft)
        _cos_h = _cos * _hann   # (n_fft, n_fft)
        _sin_h = _sin * _hann
        _w_re = torch.stack([ _cos_h,  _sin_h], dim=1)  # (n_fft, 2, n_fft): Re from (I,Q)
        _w_im = torch.stack([-_sin_h,  _cos_h], dim=1)  # (n_fft, 2, n_fft): Im from (I,Q)
        self.register_buffer('_dft_weight',
                             torch.cat([_w_re, _w_im], dim=0))  # (2*n_fft, 2, n_fft)

        # Path A: IQ residual network (weights transfer from v10/v11)
        self.iq_path = RadioResNet(num_classes=num_classes, input_len=input_len,
                                   channels=self.IQ_CHANNELS, n_blocks=8)
        self.iq_path.head = nn.Flatten()   # strip classifier head → (B, IQ_CHANNELS)

        # Path B: STFT spectrogram
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
            nn.Linear(128 * 16, 256),
            nn.ReLU(inplace=True),
        )

        # Path C: instantaneous amplitude / phase / frequency
        self.inst_path = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, self.INST_FEAT, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        # Higher-order cumulant features (4 scalars → CUM_FEAT)
        self.cumulant_head = nn.Sequential(
            nn.Linear(4, self.CUM_FEAT),
            nn.ReLU(inplace=True),
        )

        # Fusion: concat all paths → 256-dim hidden → class logits
        total = self.IQ_CHANNELS + 256 + self.INST_FEAT + self.CUM_FEAT  # 480
        self.fusion_hidden = nn.Sequential(
            nn.Linear(total, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )
        self.fusion_out = nn.Linear(256, num_classes)

        # DANN domain classifier — only used via forward_dann() during training
        self.domain_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),   # 0=synthetic, 1=real
        )

    def _spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        # Windowed DFT via conv1d — fully ONNX-safe (no aten::complex).
        # padding= is a static attribute in ONNX Conv, giving statically-inferrable T.
        # n_fft//2 zero-pad each side matches torch.stft(center=True) frame count (T=17).
        out = F.conv1d(x, self._dft_weight,
                       stride=self.hop_len, padding=self.n_fft // 2)  # (B, 2*n_fft, T)
        Re  = out[:, :self.n_fft, :]                                # (B, n_fft, T)
        Im  = out[:, self.n_fft:, :]
        mag = (Re * Re + Im * Im + 1e-12).sqrt()                    # (B, n_fft, T)
        return (mag.unsqueeze(1) + 1e-6).log()                      # (B, 1, n_fft, T)

    def _inst_features(self, x: torch.Tensor) -> torch.Tensor:
        I, Q = x[:, 0, :], x[:, 1, :]
        ia   = (I * I + Q * Q).clamp(min=1e-12).sqrt()       # (B, N)
        ip   = torch.atan2(Q, I)                               # (B, N)
        dip  = ip[:, 1:] - ip[:, :-1]                         # (B, N-1)
        dip  = dip - 2.0 * torch.pi * torch.round(dip / (2.0 * torch.pi))
        dip  = torch.cat([dip[:, :1], dip], dim=1)            # (B, N)
        feats = torch.stack([ia, ip, dip], dim=1)             # (B, 3, N)
        mu   = feats.mean(dim=-1, keepdim=True)
        std  = feats.std(dim=-1, keepdim=True).clamp(min=1e-6)
        return (feats - mu) / std

    def _cumulants(self, x: torch.Tensor) -> torch.Tensor:
        I, Q  = x[:, 0, :], x[:, 1, :]
        mag2  = I * I + Q * Q
        p     = mag2.mean(-1).clamp(min=1e-8)                 # (B,)
        # C20: non-circularity (~0 circular, ~1 real/AM)
        re2   = I * I - Q * Q
        im2   = 2.0 * I * Q
        c20   = (re2.mean(-1).pow(2) + im2.mean(-1).pow(2)).sqrt() / p
        # C42: amplitude kurtosis (distinguishes QAM orders)
        c42   = (mag2 * mag2).mean(-1) / p.pow(2) - 2.0
        # C40: 4th-order cumulant magnitude
        sq    = I * I - Q * Q
        cr    = I * Q
        re4   = sq * sq - 4.0 * cr * cr                       # Re(sig^4)
        im4   = 4.0 * cr * sq                                  # Im(sig^4)
        c40   = (re4.mean(-1).pow(2) + im4.mean(-1).pow(2)).sqrt() / p.pow(2)
        # C4P: 4th-order phase moment (sensitive to PSK constellation order)
        cos2  = (I * I - Q * Q) / mag2.clamp(min=1e-12)
        cos4  = 2.0 * cos2 * cos2 - 1.0
        c4p   = cos4.mean(-1).abs()
        return torch.stack([c20, c42, c40, c4p], dim=1)       # (B, 4)

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        """Shared feature extraction → 256-dim hidden representation."""
        iq_feat   = self.iq_path(x)                              # (B, 128)
        spec_feat = self.spec_path(self._spectrogram(x))         # (B, 256)
        inst_feat = self.inst_path(self._inst_features(x))       # (B, 64)
        cum_feat  = self.cumulant_head(self._cumulants(x))       # (B, 32)
        fused     = torch.cat([iq_feat, spec_feat, inst_feat, cum_feat], dim=1)
        return self.fusion_hidden(fused)                          # (B, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fusion_out(self._extract(x))

    def forward_dann(self, x: torch.Tensor,
                     alpha: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (class_logits, domain_logits) for DANN training."""
        h = self._extract(x)
        return self.fusion_out(h), self.domain_classifier(grad_reverse(h, alpha))


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
        model, (dummy,), out_path,
        input_names=["iq_input"],
        output_names=["logits"],
        dynamo=True,
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
