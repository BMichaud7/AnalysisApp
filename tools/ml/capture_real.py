#!/usr/bin/env python3
"""
capture_real.py — Capture labeled IQ data from PlutoSDR via AMQP controller.

Tunes to known signal frequencies, captures raw IQ, slices into 512-sample
windows, filters by SNR, and saves an .npz dataset compatible with train.py.

Signal types captured (frequency-based labeling):
  FM_WB   — FM broadcast band 88-108 MHz (stereo, ~200 kHz BW)
  AM_DSB  — Aviation AM 118-137 MHz
  FM_NB   — NOAA weather radio 162.4-162.55 MHz + maritime VHF 156 MHz

Usage:
    PYTHONPATH=/tmp/proton_pkg python3 capture_real.py \
        --out data/real.npz [--gain 50] [--snr-min 3] [--max-per-class 8000]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import socket
import struct
import sys
import threading
import time
import uuid

import numpy as np
from scipy import signal as sig

sys.path.insert(0, "/tmp/proton_pkg")
import proton
import proton.handlers
import proton.reactor

# ── AMQP constants ────────────────────────────────────────────────────────────
BROKER   = "amqp://localhost:5672"
REQ_Q    = "sdr.task.request"
RESP_Q   = "sdr.task.response"
CREDS    = ("sdr_ctrl", "sdr_hw_test")
DEST_IP  = "127.0.0.1"
IQ_HDR   = struct.Struct("<I I Q Q I H B B")
IQ_MAGIC = 0x49515030

WINDOW = 512    # samples per window — must match model input_len
STRIDE = 128    # stride between windows (75 % overlap for diversity)

# Tuning overhead: PlutoSDR PLL calibration takes ~3-4 s per hop.
# We add this to every collection window so the first packets aren't missed.
TUNE_OVERHEAD_S = 6.0


# ── Capture targets ───────────────────────────────────────────────────────────
# (label, center_hz, bw_hz, sr_sps, dwell_s, max_windows)
#
# FM_WB stations detected in our spectrum scan.
# AM_DSB: aviation guard / common ATC — activity is intermittent.
# FM_NB:  NOAA Weather Radio (always on) + maritime VHF.
# _NOISE: quiet channel for hardware impairment measurement.

TARGETS: list[tuple] = [
    # ── FM broadcast (FM_WB) ─────────────────────────────────────────────────
    ("FM_WB",  89.1e6,  2e6, 2e6, 20, 6000),
    ("FM_WB",  90.4e6,  2e6, 2e6, 20, 6000),
    ("FM_WB",  91.7e6,  2e6, 2e6, 20, 6000),
    ("FM_WB",  93.1e6,  2e6, 2e6, 20, 6000),
    ("FM_WB", 100.8e6,  2e6, 2e6, 20, 6000),
    ("FM_WB", 102.3e6,  2e6, 2e6, 20, 6000),
    ("FM_WB", 106.4e6,  2e6, 2e6, 20, 6000),

    # ── Aviation AM (AM_DSB) ─────────────────────────────────────────────────
    ("AM_DSB", 121.5e6,  1e6, 1e6, 30, 4000),   # international guard
    ("AM_DSB", 123.45e6, 1e6, 1e6, 30, 4000),   # air-to-air
    ("AM_DSB", 126.45e6, 1e6, 1e6, 30, 4000),
    ("AM_DSB", 127.0e6,  1e6, 1e6, 30, 4000),
    ("AM_DSB", 128.0e6,  1e6, 1e6, 30, 4000),
    ("AM_DSB", 132.7e6,  1e6, 1e6, 30, 4000),

    # ── NOAA Weather Radio (FM_NB, always broadcasting) ──────────────────────
    ("FM_NB", 162.400e6, 1e6, 1e6, 20, 5000),
    ("FM_NB", 162.425e6, 1e6, 1e6, 20, 5000),
    ("FM_NB", 162.450e6, 1e6, 1e6, 20, 5000),
    ("FM_NB", 162.475e6, 1e6, 1e6, 20, 5000),
    ("FM_NB", 162.500e6, 1e6, 1e6, 20, 5000),
    ("FM_NB", 162.525e6, 1e6, 1e6, 20, 5000),
    ("FM_NB", 162.550e6, 1e6, 1e6, 20, 5000),

    # ── Maritime VHF (FM_NB) ─────────────────────────────────────────────────
    ("FM_NB", 156.800e6, 1e6, 1e6, 20, 3000),   # ch.16 guard/distress
    ("FM_NB", 156.000e6, 1e6, 1e6, 20, 3000),   # ch.01

    # ── Noise-only capture for PlutoSDR impairment measurement ───────────────
    ("_NOISE", 500.0e6,  2e6, 2e6, 10, 0),
]


# ── AMQP session (exact pattern from raw_spectrum.py, known good) ─────────────

class _H(proton.handlers.MessagingHandler):
    def __init__(self, sess):
        super().__init__()
        self._s = sess

    def on_start(self, ev):
        c = ev.container.connect(BROKER, user=CREDS[0], password=CREDS[1],
                                 sasl_enabled=True, allowed_mechs="PLAIN")
        ev.container.create_receiver(c, RESP_Q)
        self._sender = ev.container.create_sender(c, REQ_Q)
        self._s._handler = self

    def on_sendable(self, ev):
        self._s._ready.set()

    def on_message(self, ev):
        try:
            msg = json.loads(ev.message.body)
        except Exception:
            return
        rid = msg.get("request_id", "")
        with self._s._lock:
            e = self._s._pending.get(rid)
        if e:
            e[1].append(msg)
            e[0].set()

    def send(self, d):
        self._sender.send(proton.Message(body=json.dumps(d),
                                         content_type="application/json"))


class Session:
    def __init__(self):
        self._pending: dict = {}
        self._lock    = threading.Lock()
        self._ready   = threading.Event()
        self._handler = None
        self._ctr     = proton.reactor.Container(_H(self))
        threading.Thread(target=self._ctr.run, daemon=True).start()
        self._ready.wait(15)

    def rpc(self, req: dict, timeout: float = 40) -> dict | None:
        rid = req["request_id"]
        ev, box = threading.Event(), []
        with self._lock:
            self._pending[rid] = (ev, box)
        self._handler.send(req)
        ev.wait(timeout)
        with self._lock:
            self._pending.pop(rid, None)
        return box[0] if box else None

    def fire(self, req: dict) -> None:
        self._handler.send(req)

    def close(self) -> None:
        try:
            self._ctr.stop()
        except Exception:
            pass


# ── IQ collection ─────────────────────────────────────────────────────────────

def collect_iq(port: int, dwell_s: float) -> np.ndarray:
    """
    Receive UDP IQ packets for dwell_s + TUNE_OVERHEAD_S seconds.

    Uses short recv() timeout (0.3 s) with `continue` so we keep polling
    through the PlutoSDR PLL calibration delay (~3-4 s) without missing data.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 32 * 1024 * 1024)
    s.settimeout(0.3)
    s.bind(("", port))
    chunks: list[np.ndarray] = []
    deadline = time.time() + dwell_s + TUNE_OVERHEAD_S
    try:
        while time.time() < deadline:
            try:
                data = s.recv(65536)
            except socket.timeout:
                continue          # keep polling — don't exit early
            if len(data) < IQ_HDR.size:
                continue
            if IQ_HDR.unpack_from(data)[0] != IQ_MAGIC:
                continue
            n   = IQ_HDR.unpack_from(data)[5]
            raw = np.frombuffer(data[IQ_HDR.size: IQ_HDR.size + n * 8],
                                dtype=np.float32)
            if len(raw) == n * 2:
                chunks.append(raw[0::2] + 1j * raw[1::2])
    finally:
        s.close()
    return np.concatenate(chunks) if chunks else np.array([], dtype=np.complex64)


def request_capture(sess: Session,
                    freq_hz: float, bw_hz: float, sr_sps: float,
                    gain_db: float, dwell_s: float,
                    verbose: bool = False) -> np.ndarray | None:
    """Submit WIDEBAND task, collect IQ.  Returns None on failure."""
    rid = str(uuid.uuid4())
    resp = sess.rpc({
        "msg_type": "TASK_REQUEST", "schema_version": "2.0",
        "request_id": rid, "timestamp_ms": int(time.time() * 1000),
        "task_type": "WIDEBAND", "rank": 1,
        "schedule": {
            "mode": "IMMEDIATE",
            "duration_ms": int((dwell_s + TUNE_OVERHEAD_S + 1) * 1000),
        },
        "rf": {
            "center_freq_hz": freq_hz,
            "bandwidth_hz":   bw_hz,
            "sample_rate_sps": sr_sps,
            "rx_count": 1,
            "rx_gain_db": [gain_db],
            "rx_agc": [False],
        },
        "streaming": {"dest_ip": DEST_IP},
        "wideband": {"record_raw_iq": True, "fft_size": 4096},
    }, timeout=40)

    if not resp or resp.get("status") != "ACCEPTED":
        reason = resp.get("reject_reason", "timeout") if resp else "no response"
        if verbose:
            print(f"    REJECTED: {reason}")
        return None

    port    = resp["streams"][0].get("udp_port")
    task_id = resp["task_id"]
    if verbose:
        print(f"    Accepted udp_port={port}", flush=True)

    iq = collect_iq(port, dwell_s)

    sess.fire({
        "msg_type": "TASK_STOP", "request_id": str(uuid.uuid4()),
        "task_id": task_id, "timestamp_ms": int(time.time() * 1000),
        "reason": "capture done",
    })
    return iq if len(iq) >= WINDOW else None


# ── Signal processing ─────────────────────────────────────────────────────────

def normalise(w: np.ndarray) -> np.ndarray:
    p = float(np.sqrt(np.mean(np.abs(w) ** 2)))
    return w / (p + 1e-12)


def estimate_snr_db(w: np.ndarray, sr: float) -> float:
    """
    Estimate per-window SNR using Welch PSD.
    Returns (90th-percentile PSD bin) - (median PSD bin) in dB.
    """
    nperseg = min(128, len(w))
    _, psd  = sig.welch(w, fs=sr, nperseg=nperseg, scaling="density")
    psd_db  = 10.0 * np.log10(psd + 1e-30)
    return float(np.percentile(psd_db, 90) - np.median(psd_db))


def extract_windows(iq: np.ndarray, sr: float,
                    snr_min: float, max_wins: int) -> list[np.ndarray]:
    """Slice IQ into WINDOW-length segments, filter by SNR, normalise."""
    n       = len(iq)
    indices = list(range(0, n - WINDOW, STRIDE))
    np.random.shuffle(indices)      # shuffle so we sample the whole capture
    windows: list[np.ndarray] = []
    for i in indices:
        w = iq[i: i + WINDOW]
        if estimate_snr_db(w, sr) < snr_min:
            continue
        windows.append(normalise(w))
        if len(windows) >= max_wins:
            break
    return windows


def iq_to_2ch(windows: list[np.ndarray]) -> np.ndarray:
    """(N, 2, WINDOW) float32 — same layout as generate_extended.py."""
    arr = np.stack(windows)
    return np.stack([arr.real, arr.imag], axis=1).astype(np.float32)


# ── PlutoSDR impairment measurement ──────────────────────────────────────────

def measure_impairments(iq: np.ndarray) -> dict:
    if len(iq) < 4096:
        return {}
    I, Q  = iq.real, iq.imag
    rms   = float(np.sqrt(np.mean(np.abs(iq) ** 2)))
    dc_i  = float(np.mean(I)) / (rms + 1e-12)
    dc_q  = float(np.mean(Q)) / (rms + 1e-12)
    std_i = float(np.std(I))
    std_q = float(np.std(Q))
    amp_db = float(20.0 * np.log10(std_i / (std_q + 1e-12) + 1e-12))
    corr   = float(np.mean(I * Q)) / (std_i * std_q + 1e-12)
    phase  = float(math.degrees(math.asin(max(-1.0, min(1.0, corr)))))
    return {"dc_i": dc_i, "dc_q": dc_q,
            "iq_amp_imbalance_db": amp_db, "iq_phase_deg": phase, "rms": rms}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",           default="data/real.npz")
    ap.add_argument("--gain",    type=float, default=50.0)
    ap.add_argument("--snr-min", type=float, default=3.0)
    ap.add_argument("--max-per-class", type=int, default=8000)
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    print("=" * 60)
    print(" PlutoSDR Real IQ Capture")
    print(f"  Output:        {args.out}")
    print(f"  Gain:          {args.gain} dB")
    print(f"  SNR threshold: {args.snr_min} dB")
    print(f"  Max/class:     {args.max_per_class}")
    print("=" * 60, flush=True)

    sess = Session()
    print("AMQP session ready.\n", flush=True)

    all_X:   list[np.ndarray] = []
    all_y:   list[int]        = []
    all_snr: list[float]      = []
    label_names: list[str]    = []
    label_map:   dict[str, int] = {}
    class_counts: dict[str, int] = {}
    impairments: dict = {}

    for (label, freq_hz, bw_hz, sr_sps, dwell_s, max_wins) in TARGETS:
        cap_max = min(max_wins, args.max_per_class) if label != "_NOISE" else 0
        print(f"[{label}]  {freq_hz/1e6:.3f} MHz  "
              f"BW={bw_hz/1e6:.1f} MHz  {dwell_s}s  max={cap_max}", flush=True)

        iq = request_capture(sess, freq_hz, bw_hz, sr_sps,
                             args.gain, dwell_s, verbose=args.verbose)

        if iq is None:
            print("  → No IQ (controller rejected or silent)\n")
            continue

        print(f"  Received {len(iq):,} samples", end="", flush=True)

        if label == "_NOISE":
            imp = measure_impairments(iq)
            impairments = imp
            print()
            print(f"  DC offset    I={imp.get('dc_i', 0):.5f}  "
                  f"Q={imp.get('dc_q', 0):.5f}")
            print(f"  IQ imbalance {imp.get('iq_amp_imbalance_db', 0):.3f} dB")
            print(f"  IQ phase err {imp.get('iq_phase_deg', 0):.2f}°\n")
            continue

        windows = extract_windows(iq, sr_sps, args.snr_min, cap_max)
        print(f"  → {len(windows)} windows kept (SNR≥{args.snr_min} dB)")

        if not windows:
            print("  (signal absent at this frequency — skipping)\n")
            continue

        if label not in label_map:
            label_map[label] = len(label_names)
            label_names.append(label)
        idx = label_map[label]

        X_b  = iq_to_2ch(windows)
        snrs = np.array([estimate_snr_db(w, sr_sps) for w in windows],
                        dtype=np.float32)

        all_X.append(X_b)
        all_y.extend([idx] * len(windows))
        all_snr.extend(snrs.tolist())
        class_counts[label] = class_counts.get(label, 0) + len(windows)
        print(f"  Class '{label}' total: {class_counts[label]}\n", flush=True)

    sess.close()

    if not all_X:
        print("ERROR: No data captured — check controller/SDR connectivity.")
        sys.exit(1)

    X   = np.concatenate(all_X, axis=0)
    y   = np.array(all_y, dtype=np.int64)
    snr = np.array(all_snr, dtype=np.float32)
    cls = np.array(label_names)

    print("=" * 60)
    print(f"Total windows : {len(X)}")
    for lbl, cnt in sorted(class_counts.items()):
        print(f"  {lbl:15s}: {cnt:,}")
    print(f"Classes       : {list(cls)}")

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y, snrs=snr, classes=cls)
    print(f"\nSaved → {args.out}  ({X.nbytes/1e6:.0f} MB uncompressed)")

    if impairments:
        imp_path = str(pathlib.Path(args.out).parent / "pluto_impairments.json")
        with open(imp_path, "w") as f:
            json.dump(impairments, f, indent=2)
        print(f"Saved → {imp_path}")


if __name__ == "__main__":
    main()
