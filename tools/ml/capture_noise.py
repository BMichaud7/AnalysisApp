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
capture_noise.py — Capture real hardware noise from PlutoSDR for use as
training noise in IqAugment (replaces synthetic AWGN with real ADC noise).

Tunes to a set of quiet frequencies at SR=200 kHz (matching generate_v3.py),
captures raw IQ, and slices it into 1024-sample windows — the same window
length used during training.  Windows are normalised to unit power so they
can be added at any target SNR.

Usage (PlutoSDR + AMQP controller running):
    PYTHONPATH=/tmp/proton_pkg python3 capture_noise.py \
        --out data/noise_bank.npz [--gain 50] [--n-windows 20000]

Pass the output to train.py via --noise-npz:
    python3 train.py --npz data/synth.npz --noise-npz data/noise_bank.npz ...
"""
from __future__ import annotations

import argparse
import json
import pathlib
import socket
import struct
import sys
import threading
import time
import uuid

import numpy as np

sys.path.insert(0, "/tmp/proton_pkg")
import proton
import proton.handlers
import proton.reactor

# ── Constants ─────────────────────────────────────────────────────────────────
BROKER   = "amqp://localhost:5672"
REQ_Q    = "sdr.task.request"
RESP_Q   = "sdr.task.response"
CREDS    = ("sdr_ctrl", "sdr_hw_test")
DEST_IP  = "127.0.0.1"
IQ_HDR   = struct.Struct("<I I Q Q I H B B")
IQ_MAGIC = 0x49515030

SR      = 200_000       # Hz — must match generate_v3.py SR
WINDOW  = 1024          # samples — must match training input_len
STRIDE  = 256           # 75% overlap for diversity

# Quiet frequencies to sweep at 200 kHz SR.  Chosen for minimal ambient RF in
# an urban US environment; all are between major band allocations.
NOISE_TARGETS = [
    # (center_hz, dwell_s)
    (400.100e6, 30),   # 400 MHz — intersat guard, generally quiet
    (401.050e6, 30),
    (406.200e6, 20),   # 406.0-406.1 is EPIRB, 406.2 is quiet
    (433.250e6, 20),   # ISM edge, generally quiet at narrow BW
    (470.100e6, 30),   # below UHF TV (UHF starts 470), quiet near large cities
    (510.000e6, 30),   # white space between UHF channels
    (602.000e6, 20),   # depends on local TV; quiet if no local broadcaster
    (703.000e6, 20),   # 700 MHz gap between LTE bands
    (758.000e6, 20),   # FDD pair gap
    (820.000e6, 20),   # 800 MHz gap
    (880.000e6, 20),   # 800/850 gap
]


# ── AMQP session ──────────────────────────────────────────────────────────────
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

    def rpc(self, req: dict, timeout: float = 60) -> dict | None:
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


# ── IQ capture ────────────────────────────────────────────────────────────────
def collect_iq(port: int, dwell_s: float) -> np.ndarray:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 32 * 1024 * 1024)
    s.settimeout(0.3)
    s.bind(("", port))
    chunks: list[np.ndarray] = []
    deadline = time.time() + dwell_s + 0.1
    try:
        while time.time() < deadline:
            try:
                data = s.recv(65536)
            except socket.timeout:
                continue
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


def request_capture(sess: Session, freq_hz: float, dwell_s: float,
                    gain_db: float) -> np.ndarray | None:
    rid = str(uuid.uuid4())
    resp = sess.rpc({
        "msg_type": "TASK_REQUEST", "schema_version": "2.0",
        "request_id": rid, "timestamp_ms": int(time.time() * 1000),
        "task_type": "WIDEBAND", "rank": 5,
        "schedule": {
            "mode": "IMMEDIATE",
            "duration_ms": int((dwell_s + 1.0) * 1000),
        },
        "rf": {
            "center_freq_hz": freq_hz,
            "bandwidth_hz":   SR,
            "sample_rate_sps": SR,
            "rx_count": 1,
            "rx_gain_db": [gain_db],
            "rx_agc": [False],
        },
        "streaming": {"dest_ip": DEST_IP},
        "wideband": {"record_raw_iq": True, "fft_size": 512},
    }, timeout=dwell_s + 30)

    if not resp or resp.get("status") != "ACCEPTED":
        reason = resp.get("reject_reason", "timeout") if resp else "no response"
        print(f"  REJECTED: {reason}")
        return None

    port    = resp["streams"][0].get("udp_port")
    task_id = resp["task_id"]
    iq = collect_iq(port, dwell_s)
    sess.fire({
        "msg_type": "TASK_STOP", "request_id": str(uuid.uuid4()),
        "task_id": task_id, "timestamp_ms": int(time.time() * 1000),
        "reason": "noise capture done",
    })
    return iq if len(iq) >= WINDOW else None


# ── Signal processing ─────────────────────────────────────────────────────────
def extract_noise_windows(iq: np.ndarray, max_wins: int) -> list[np.ndarray]:
    """Slice into WINDOW-length segments; filter out any with strong signal."""
    n       = len(iq)
    indices = list(range(0, n - WINDOW, STRIDE))
    np.random.shuffle(indices)
    windows: list[np.ndarray] = []
    for i in indices:
        w   = iq[i: i + WINDOW]
        pwr = np.mean(np.abs(w) ** 2)
        if pwr < 1e-12:
            continue
        # Reject windows where a strong signal component is present:
        # compare peak PSD bin to median — if gap > 15 dB there's a signal
        psd = np.abs(np.fft.fft(w)) ** 2
        gap = 10.0 * np.log10(psd.max() / (np.median(psd) + 1e-30))
        if gap > 15.0:
            continue
        # Normalise to unit power
        windows.append((w / np.sqrt(pwr)).astype(np.complex64))
        if len(windows) >= max_wins:
            break
    return windows


def to_2ch(windows: list[np.ndarray]) -> np.ndarray:
    """(N, 2, WINDOW) float32 matching training data layout."""
    arr = np.stack(windows)
    return np.stack([arr.real, arr.imag], axis=1).astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Capture real PlutoSDR noise for IqAugment")
    ap.add_argument("--out",       default="data/noise_bank.npz",
                    help="Output NPZ path (default: data/noise_bank.npz)")
    ap.add_argument("--gain",      type=float, default=50.0,
                    help="PlutoSDR RX gain dB (default 50)")
    ap.add_argument("--n-windows", type=int,   default=20000,
                    help="Total noise windows to collect (default 20000)")
    args = ap.parse_args()

    per_target = max(200, args.n_windows // len(NOISE_TARGETS))

    print("=" * 60)
    print(f" PlutoSDR Noise Capture  SR={SR/1e3:.0f} kHz  window={WINDOW}")
    print(f"  Output:  {args.out}")
    print(f"  Gain:    {args.gain} dB")
    print(f"  Target:  {args.n_windows} windows ({per_target}/freq)")
    print("=" * 60, flush=True)

    sess = Session()
    print("AMQP session ready.\n", flush=True)

    all_windows: list[np.ndarray] = []

    for freq_hz, dwell_s in NOISE_TARGETS:
        if len(all_windows) >= args.n_windows:
            break
        need = min(per_target, args.n_windows - len(all_windows))
        print(f"  {freq_hz/1e6:.3f} MHz  {dwell_s}s  want={need}", end=" ", flush=True)
        iq = request_capture(sess, freq_hz, dwell_s, args.gain)
        if iq is None:
            print("→ skipped")
            continue
        wins = extract_noise_windows(iq, need)
        all_windows.extend(wins)
        print(f"→ {len(wins)} windows  (total {len(all_windows)})", flush=True)

    sess.close()

    if not all_windows:
        print("ERROR: No noise windows captured — check controller/SDR connectivity.")
        sys.exit(1)

    X = to_2ch(all_windows)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=X)
    print(f"\nSaved {len(X)} noise windows → {args.out}  "
          f"({X.nbytes/1e6:.1f} MB uncompressed, "
          f"{pathlib.Path(args.out).stat().st_size/1e6:.1f} MB compressed)")


if __name__ == "__main__":
    main()

# ========================================================================
# End of file — OpenRFStack
# Subject to Personal Use License
# https://github.com/OpenRFStack
# ========================================================================
