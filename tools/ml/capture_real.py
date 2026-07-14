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

pass  # proton now installed in venv directly
import proton
import proton.handlers
import proton.reactor

# ── AMQP constants ────────────────────────────────────────────────────────────
BROKER   = "amqp://localhost:5672"
REQ_Q    = "sdr.task.request"
RESP_Q   = "sdr.task.response"
CREDS    = ("sdr_ctrl", "sdr_ctrl")   # dev-local broker; override with --broker-password
DEST_IP  = "127.0.0.1"
IQ_HDR   = struct.Struct("<I I Q Q I H B B")
IQ_MAGIC = 0x49515030

WINDOW = 1024   # samples per window — must match model input_len (all NPZs use 1024)
STRIDE = 256    # stride between windows (75 % overlap for diversity)

# AD9361 LO settles in ~25 µs after setFrequency.  The sample rate is now
# locked (fixed_sample_rate_hz in devices.xml) so setSampleRate is never
# called mid-session.  50 ms is a conservative margin for USB/network latency.
TUNE_OVERHEAD_S = 0.05


# ── Capture targets ───────────────────────────────────────────────────────────
# (label, center_hz, bw_hz, sr_sps, dwell_s, max_windows)
#
# FM_WB stations detected in our spectrum scan.
# AM_DSB: aviation guard / common ATC — activity is intermittent.
# FM_NB:  NOAA Weather Radio (always on) + maritime VHF.
# _NOISE: quiet channel for hardware impairment measurement.

TARGETS: list[tuple] = [
    # ── ILS Localizer (AM_DSB) — 108.1–111.975 MHz, odd-tenth channels ───────
    # Carrier + 90 Hz + 150 Hz tones; capture at narrow BW to stay on signal
    ("AM_DSB", 108.100e6, 0.5e6, 0.5e6, 30, 3000),
    ("AM_DSB", 108.300e6, 0.5e6, 0.5e6, 30, 3000),
    ("AM_DSB", 108.500e6, 0.5e6, 0.5e6, 30, 3000),
    ("AM_DSB", 108.700e6, 0.5e6, 0.5e6, 30, 3000),
    ("AM_DSB", 108.900e6, 0.5e6, 0.5e6, 30, 3000),
    ("AM_DSB", 109.100e6, 0.5e6, 0.5e6, 30, 3000),
    ("AM_DSB", 109.300e6, 0.5e6, 0.5e6, 30, 3000),
    ("AM_DSB", 109.500e6, 0.5e6, 0.5e6, 30, 3000),
    ("AM_DSB", 109.700e6, 0.5e6, 0.5e6, 30, 3000),
    ("AM_DSB", 110.100e6, 0.5e6, 0.5e6, 30, 3000),
    ("AM_DSB", 110.300e6, 0.5e6, 0.5e6, 30, 3000),
    ("AM_DSB", 110.500e6, 0.5e6, 0.5e6, 30, 3000),
    ("AM_DSB", 110.700e6, 0.5e6, 0.5e6, 30, 3000),
    ("AM_DSB", 110.900e6, 0.5e6, 0.5e6, 30, 3000),
    ("AM_DSB", 111.100e6, 0.5e6, 0.5e6, 30, 3000),
    ("AM_DSB", 111.300e6, 0.5e6, 0.5e6, 30, 3000),
    ("AM_DSB", 111.500e6, 0.5e6, 0.5e6, 30, 3000),
    ("AM_DSB", 111.700e6, 0.5e6, 0.5e6, 30, 3000),
    ("AM_DSB", 111.900e6, 0.5e6, 0.5e6, 30, 3000),

    # ── FM broadcast (FM_WB) — always present ────────────────────────────────
    ("FM_WB",  89.1e6,  2e6,  2e6,  20, 6000),
    ("FM_WB",  90.4e6,  2e6,  2e6,  20, 6000),
    ("FM_WB",  91.7e6,  2e6,  2e6,  20, 6000),
    ("FM_WB",  93.1e6,  2e6,  2e6,  20, 6000),
    ("FM_WB", 100.8e6,  2e6,  2e6,  20, 6000),
    ("FM_WB", 102.3e6,  2e6,  2e6,  20, 6000),
    ("FM_WB", 106.4e6,  2e6,  2e6,  20, 6000),

    # ── Aviation AM (AM_DSB) — intermittent, guard freq usually active ────────
    ("AM_DSB", 121.5e6,  1e6,  1e6,  30, 4000),   # international guard
    ("AM_DSB", 123.45e6, 1e6,  1e6,  30, 4000),   # air-to-air
    ("AM_DSB", 126.45e6, 1e6,  1e6,  30, 4000),
    ("AM_DSB", 127.0e6,  1e6,  1e6,  30, 4000),
    ("AM_DSB", 128.0e6,  1e6,  1e6,  30, 4000),
    ("AM_DSB", 132.7e6,  1e6,  1e6,  30, 4000),

    # ── FSK — APRS digipeaters (always active in populated areas) ────────────
    ("FSK", 144.390e6,  0.5e6, 0.5e6, 60, 3000),  # APRS national 1200-baud

    # ── NOAA Weather Radio (FM_NB) — always broadcasting ─────────────────────
    ("FM_NB", 162.400e6, 1e6,  1e6,  20, 5000),
    ("FM_NB", 162.425e6, 1e6,  1e6,  20, 5000),
    ("FM_NB", 162.450e6, 1e6,  1e6,  20, 5000),
    ("FM_NB", 162.475e6, 1e6,  1e6,  20, 5000),
    ("FM_NB", 162.500e6, 1e6,  1e6,  20, 5000),
    ("FM_NB", 162.525e6, 1e6,  1e6,  20, 5000),
    ("FM_NB", 162.550e6, 1e6,  1e6,  20, 5000),

    # ── Maritime VHF (FM_NB) ─────────────────────────────────────────────────
    ("FM_NB", 156.800e6, 1e6,  1e6,  20, 3000),   # ch.16 guard/distress
    ("FM_NB", 156.000e6, 1e6,  1e6,  20, 3000),   # ch.01

    # ── GMSK — paging band (POCSAG/Flex, continuous broadcast) ───────────────
    ("GMSK", 152.000e6, 1e6,  1e6,  30, 3000),
    ("GMSK", 152.480e6, 1e6,  1e6,  30, 3000),
    ("GMSK", 157.450e6, 1e6,  1e6,  20, 2000),

    # ── FSK — ISM 915 MHz (LoRa gateways, Z-Wave, 802.15.4, RFID) ───────────
    ("FSK", 903.0e6,  4e6,  4e6,  30, 4000),
    ("FSK", 915.0e6,  4e6,  4e6,  30, 4000),
    ("FSK", 925.0e6,  4e6,  4e6,  20, 3000),

    # ── OFDM — LTE cellular downlinks (present everywhere in US) ─────────────
    ("OFDM",  739.0e6, 12e6, 15e6, 20, 5000),   # Band 17 DL (AT&T 700)
    ("OFDM",  751.0e6, 10e6, 12e6, 20, 5000),   # Band 13 DL (Verizon 700)
    ("OFDM",  881.0e6, 24e6, 30e6, 20, 5000),   # Band 5 DL (850 MHz)
    ("OFDM", 1960.0e6, 25e6, 30e6, 20, 5000),   # Band 2 DL (PCS 1900)
    ("OFDM", 2140.0e6, 40e6, 50e6, 20, 5000),   # Band 4/66 DL (AWS)
    ("OFDM", 2665.0e6, 40e6, 50e6, 20, 5000),   # Band 41 TDD (mid-band)

    # ── GFSK — Bluetooth 2.4 GHz (present near any BT device) ───────────────
    ("GFSK", 2420.0e6, 40e6, 40e6, 20, 4000),
    ("GFSK", 2441.0e6, 40e6, 40e6, 20, 4000),
    ("GFSK", 2462.0e6, 40e6, 40e6, 20, 3000),

    # ── ADS-B — 1090 MHz Mode-S transponder (every aircraft, ~1 msg/sec) ────────
    # Highest-reliability real-data source: always-on, enormous volume, high SNR.
    # PPM at 1 Mbit/s, 1 MHz occupied BW. Capture wide enough to avoid IQ edge
    # roll-off but not so wide noise drowns weaker aircraft.
    ("ADS_B", 1090.0e6, 4e6,  4e6,  60, 8000),

    # ── ACARS — aviation VHF data link (AM-MSK, ~2 kbps) ────────────────────
    # Active near any commercial airport. 129.125 and 136.900 MHz are highest
    # activity in the US; 130.025 is secondary. Good SNR from overflying aircraft.
    ("ACARS", 129.125e6, 0.5e6, 0.5e6, 60, 4000),
    ("ACARS", 130.025e6, 0.5e6, 0.5e6, 60, 3000),
    ("ACARS", 136.900e6, 0.5e6, 0.5e6, 60, 4000),

    # ── VDL2 — VHF Digital Link Mode 2 (D8PSK, 31.5 kbps) ───────────────────
    # Aviation data link, same band as ACARS, replacing it on newer aircraft.
    # 136.900 MHz is also used by VDL2 — share the dwell, label separately.
    ("VDL2", 136.725e6, 0.5e6, 0.5e6, 60, 3000),
    ("VDL2", 136.775e6, 0.5e6, 0.5e6, 60, 3000),
    ("VDL2", 136.875e6, 0.5e6, 0.5e6, 60, 2000),

    # ── AIS — maritime VHF (GMSK, 9.6 kbps, 25 kHz channels) ───────────────
    # Always-on near any body of water with ship/boat traffic. Channels 87B/88B.
    # Shares GMSK modulation but distinct protocol frame structure.
    ("AIS", 161.975e6, 0.25e6, 0.25e6, 30, 3000),
    ("AIS", 162.025e6, 0.25e6, 0.25e6, 30, 3000),

    # ── DSC — Maritime VHF Channel 70 (FSK, 1200 baud) ──────────────────────
    # Digital Selective Calling, mandatory on all GMDSS vessels. Near-continuous
    # poll traffic on ch70 plus routine position reports.
    ("DSC", 156.525e6, 0.25e6, 0.25e6, 30, 2000),

    # ── P25 Phase 1 C4FM — public safety VHF/UHF (4-level FSK) ─────────────
    # Most common digital voice protocol for US police/fire/EMS.
    # Frequencies are area-specific — these are widely-used US defaults.
    # Adjust to your local public safety trunked system if signal is absent.
    ("P25_C4FM", 155.340e6, 0.025e6, 0.025e6, 120, 5000),  # common fire dispatch
    ("P25_C4FM", 155.370e6, 0.025e6, 0.025e6, 120, 5000),
    ("P25_C4FM", 154.920e6, 0.025e6, 0.025e6, 120, 4000),
    ("P25_C4FM", 460.050e6, 0.025e6, 0.025e6, 120, 5000),  # common UHF public safety
    ("P25_C4FM", 460.125e6, 0.025e6, 0.025e6, 120, 5000),
    ("P25_C4FM", 460.225e6, 0.025e6, 0.025e6, 120, 4000),
    ("P25_C4FM", 460.500e6, 0.025e6, 0.025e6, 120, 4000),
    ("P25_C4FM", 851.025e6, 0.025e6, 0.025e6, 120, 4000),  # 800 MHz trunked
    ("P25_C4FM", 851.525e6, 0.025e6, 0.025e6, 120, 4000),
    ("P25_C4FM", 852.025e6, 0.025e6, 0.025e6, 120, 3000),

    # ── DMR — commercial/public safety VHF/UHF (4FSK, TDMA) ─────────────────
    # Same 12.5 kHz channels as P25, growing rapidly in both commercial and
    # public safety. Check radioreference.com for local DMR talkgroups.
    ("DMR", 462.550e6, 0.025e6, 0.025e6, 120, 4000),  # GMRS/FRS overlap, many DMR
    ("DMR", 462.575e6, 0.025e6, 0.025e6, 120, 4000),
    ("DMR", 462.600e6, 0.025e6, 0.025e6, 120, 3000),
    ("DMR", 463.000e6, 0.025e6, 0.025e6, 120, 4000),
    ("DMR", 464.550e6, 0.025e6, 0.025e6, 120, 3000),
    ("DMR", 851.000e6, 0.025e6, 0.025e6, 120, 3000),  # 800 MHz trunked DMR

    # ── POCSAG — numeric/text paging VHF (OOK/FSK, 512/1200/2400 bps) ───────
    # Still active for hospital staff, fire alerts, industrial systems.
    # 152–158 MHz in North America. Continuous broadcast on active channels.
    ("POCSAG", 152.240e6, 0.025e6, 0.025e6, 60, 3000),
    ("POCSAG", 152.480e6, 0.025e6, 0.025e6, 60, 3000),
    ("POCSAG", 153.350e6, 0.025e6, 0.025e6, 60, 3000),
    ("POCSAG", 157.450e6, 0.025e6, 0.025e6, 60, 2000),
    ("POCSAG", 158.100e6, 0.025e6, 0.025e6, 60, 2000),
    ("POCSAG", 931.9375e6, 0.025e6, 0.025e6, 60, 3000),  # US national paging

    # ── FLEX — Motorola paging protocol VHF/UHF (4-level FSK, 1.6–6.4 kbps) ─
    # Still widely used for hospital paging (more reliable than POCSAG in
    # high-interference environments). Same 152–158 MHz band.
    ("FLEX", 152.840e6, 0.025e6, 0.025e6, 60, 2000),
    ("FLEX", 154.040e6, 0.025e6, 0.025e6, 60, 2000),

    # ── CSS / LoRa — ISM 915 MHz chirp spread spectrum ───────────────────────
    # LoRaWAN gateways are ubiquitous in urban/suburban US. Always active on
    # 902–928 MHz. 8 uplink channels + 1 downlink. Long dwell to catch sparse
    # uplink traffic from battery-powered IoT sensors.
    ("CSS", 902.300e6, 0.5e6, 0.5e6, 120, 4000),
    ("CSS", 902.500e6, 0.5e6, 0.5e6, 120, 4000),
    ("CSS", 903.900e6, 0.5e6, 0.5e6, 120, 3000),
    ("CSS", 905.300e6, 0.5e6, 0.5e6, 120, 3000),
    ("CSS", 433.175e6, 0.5e6, 0.5e6, 120, 2000),  # EU/Asia LoRa, also common in US

    # ── NXDN — Kenwood/Icom narrowband digital (4FSK or FDMA) ───────────────
    # Business radio protocol, 6.25 kHz or 12.5 kHz channels, VHF/UHF.
    # Less common than DMR/P25 but present in industrial and utility systems.
    ("NXDN", 451.000e6, 0.015e6, 0.015e6, 60, 2000),
    ("NXDN", 451.100e6, 0.015e6, 0.015e6, 60, 2000),
    ("NXDN", 456.050e6, 0.015e6, 0.015e6, 60, 2000),

    # ── TETRA — public safety digital trunked (π/4-DQPSK, 25 kHz) ───────────
    # Dominant public safety digital voice in Europe/APAC; limited US presence.
    # US trunked TETRA is rare — add if known local system exists.
    # European frequencies shown; replace with local if applicable.
    # ("TETRA", 380.000e6, 0.025e6, 0.025e6, 120, 2000),  # EU Airwave/BOS
    # ("TETRA", 390.000e6, 0.025e6, 0.025e6, 120, 2000),

    # ── D-STAR — Icom amateur digital voice (GMSK, 4.8 kbps) ────────────────
    # Ham radio digital voice — active on 2m/70cm repeaters.
    ("DSTAR", 144.1125e6, 0.025e6, 0.025e6, 60, 2000),  # 2m simplex
    ("DSTAR", 145.375e6,  0.025e6, 0.025e6, 60, 2000),  # common 2m repeater output
    ("DSTAR", 441.000e6,  0.025e6, 0.025e6, 60, 2000),  # 70cm repeater output
    ("DSTAR", 443.000e6,  0.025e6, 0.025e6, 60, 2000),

    # ── EAS_SAME — NOAA Weather Radio emergency tones (FSK, 520.83 bps) ─────
    # EAS tones broadcast before every weather alert/test on NOAA WX channels
    # (already captured above as FM_NB). Here we capture longer dwells on the
    # most active NOAA channels specifically to catch weekly required tests
    # (Wednesdays 11am ET) and real alerts. Long dwell, low max_windows since
    # actual EAS tone bursts are infrequent.
    ("EAS_SAME", 162.400e6, 0.5e6, 0.5e6, 300, 500),
    ("EAS_SAME", 162.550e6, 0.5e6, 0.5e6, 300, 500),

    # ── Noise-only capture for PlutoSDR impairment measurement ───────────────
    ("_NOISE", 500.0e6,  2e6,  2e6,  10, 0),
]


# ── AMQP session (exact pattern from raw_spectrum.py, known good) ─────────────

class _H(proton.handlers.MessagingHandler):
    def __init__(self, sess):
        super().__init__()
        self._s = sess

    def on_start(self, ev):
        c = ev.container.connect(self._s._broker, user=self._s._user,
                                 password=self._s._password,
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
    def __init__(self, broker: str = BROKER,
                 user: str = CREDS[0], password: str = CREDS[1]):
        self._broker   = broker
        self._user     = user
        self._password = password
        self._pending: dict = {}
        self._lock    = threading.Lock()
        self._ready   = threading.Event()
        self._handler = None
        self._ctr     = proton.reactor.Container(_H(self))
        threading.Thread(target=self._ctr.run, daemon=True).start()
        if not self._ready.wait(15):
            raise RuntimeError(
                f"AMQP connect to {broker} as {user!r} timed out after 15 s "
                "(wrong password or broker not running?)")

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
        "task_type": "WIDEBAND", "rank": 5,
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

    streams = resp.get("streams", [])
    if not streams:
        if verbose:
            print("    ACCEPTED but no streams in response")
        return None
    port    = streams[0].get("udp_port")
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
    ap.add_argument("--filter-classes", nargs="+", metavar="CLS",
                    help="Only capture these classes (subset of TARGETS)")
    ap.add_argument("--broker-url",      default=BROKER)
    ap.add_argument("--broker-user",     default=CREDS[0])
    ap.add_argument("--broker-password", default=CREDS[1])
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    print("=" * 60)
    print(" PlutoSDR Real IQ Capture")
    print(f"  Output:        {args.out}")
    print(f"  Broker:        {args.broker_url}  user={args.broker_user}")
    print(f"  Gain:          {args.gain} dB")
    print(f"  SNR threshold: {args.snr_min} dB")
    print(f"  Max/class:     {args.max_per_class}")
    print("=" * 60, flush=True)

    sess = Session(args.broker_url, args.broker_user, args.broker_password)
    print("AMQP session ready.\n", flush=True)

    all_X:   list[np.ndarray] = []
    all_y:   list[int]        = []
    all_snr: list[float]      = []
    label_names: list[str]    = []
    label_map:   dict[str, int] = {}
    class_counts: dict[str, int] = {}
    impairments: dict = {}

    filter_cls = set(args.filter_classes) if args.filter_classes else None

    for (label, freq_hz, bw_hz, sr_sps, dwell_s, max_wins) in TARGETS:
        if filter_cls and label not in filter_cls and label != "_NOISE":
            continue
        cap_max = min(max_wins, args.max_per_class) if label != "_NOISE" else 0
        print(f"[{label}]  {freq_hz/1e6:.3f} MHz  "
              f"BW={bw_hz/1e3:.0f} kHz  {dwell_s}s  max={cap_max}", flush=True)

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

# ========================================================================
# End of file — OpenRFStack
# Subject to Personal Use License
# https://github.com/OpenRFStack
# ========================================================================
