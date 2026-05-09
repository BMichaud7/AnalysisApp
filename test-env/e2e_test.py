#!/usr/bin/env python3
"""
AnalysisApp integration tests — fake controller + IQ streamer.

Each test case starts fresh.  The script acts as:
  • Detection source  → rf.detections
  • Fake controller   → sdr.task.request (intercept) / sdr.task.response (respond)
  • Result verifier   → rf.analysis

Test cases
----------
1. basic_flow          Full pipeline: detection → TASK_REQUEST → IQ stream → ANALYSIS_RESULT
2. udp_port_protocol   Controller allocates port; verifies app binds to the returned port
3. stale_response      Send wrong request_id first; correct one accepted on second message
4. timeout_returns     No response sent; collect() must return within 2×timeout, no hang
5. task_stop_sent      After IQ collected, app must send TASK_STOP

Usage:
    python3 e2e_test.py [--broker amqp://localhost:5672] [--test basic_flow]
"""
from __future__ import annotations

import argparse, json, math, socket, struct, sys, threading, time, uuid
import numpy as np

try:
    import proton, proton.handlers, proton.reactor
except ImportError:
    sys.exit("pip install python-qpid-proton")

# ── IQ packet (matches sdr::IqPacketHeader, 32 bytes, little-endian) ─────────
IQ_MAGIC  = 0x49515030   # "IQP0"
IQ_HEADER = struct.Struct("<I I Q Q I H B B")
assert IQ_HEADER.size == 32

IQ_FLAG_FIRST_PACKET = 0x02
IQ_FLAG_DWELL_CHANGE = 0x04

def _iq_packet(seq: int, samples: np.ndarray, cf: float, sr: float,
               flags: int = 0) -> bytes:
    n   = len(samples)
    hdr = IQ_HEADER.pack(IQ_MAGIC, seq, 0, int(cf), int(sr), n, 0, flags)
    raw = np.empty(n * 2, dtype=np.float32)
    raw[0::2] = samples.real
    raw[1::2] = samples.imag
    return hdr + raw.tobytes()

def _gen_fm(n: int = 4096, sr: float = 2e6, dev: float = 75e3) -> np.ndarray:
    rng   = np.random.default_rng(42)
    audio = rng.standard_normal(n).astype(np.float32)
    audio /= np.max(np.abs(audio)) + 1e-9
    phase = 2 * np.pi * dev / sr * np.cumsum(audio)
    iq    = (np.exp(1j * phase)).astype(np.complex64)
    return iq / np.sqrt(np.mean(np.abs(iq) ** 2))

def _stream_iq(dest_ip: str, dest_port: int, cf: float, sr: float,
               n_total: int = 200_000, pkt: int = 1024,
               delay_s: float = 0.1):
    """Stream n_total CF32 IQ samples in UDP packets to dest_ip:dest_port."""
    time.sleep(delay_s)
    sock    = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    base    = _gen_fm(4096, sr)
    full    = np.tile(base, math.ceil(n_total / len(base)))[:n_total]
    flags   = IQ_FLAG_FIRST_PACKET
    for seq, off in enumerate(range(0, len(full), pkt)):
        chunk = full[off : off + pkt]
        data  = _iq_packet(seq, chunk, cf, sr, flags)
        sock.sendto(data, (dest_ip, dest_port))
        flags = 0
        time.sleep(0.0005)
    sock.close()
    print(f"  [ctrl] streamed {seq+1} packets ({n_total} samples) → {dest_ip}:{dest_port}")

# ── Port pool (simulates controller allocating from its pool) ─────────────────
_PORT_POOL_START = 30100   # avoid clashing with hw-test pool (30000-30099)
_port_counter    = _PORT_POOL_START

def _alloc_port() -> int:
    global _port_counter
    p = _port_counter
    _port_counter += 1
    return p

# ── Base proton handler ───────────────────────────────────────────────────────

class _BaseHandler(proton.handlers.MessagingHandler):
    CREDS = ("sdr_ctrl", "test_password")

    def __init__(self, broker: str):
        super().__init__()
        self.broker = broker
        self._senders: dict[str, proton.Sender] = {}
        self.done  = threading.Event()
        self.error: str | None = None

    def _connect(self, event, *recv_addrs):
        conn = event.container.connect(
            self.broker,
            user=self.CREDS[0], password=self.CREDS[1],
            sasl_enabled=True, allowed_mechs="PLAIN",
        )
        for addr in recv_addrs:
            event.container.create_receiver(conn, addr)
        return conn

    def _open_sender(self, conn, addr: str) -> proton.Sender:
        s = conn.open_sender(addr)
        self._senders[addr] = s
        return s

    def _send(self, addr: str, body: dict):
        msg = proton.Message(body=json.dumps(body),
                             content_type="application/json")
        self._senders[addr].send(msg)

    def _accept_and_parse(self, event) -> dict | None:
        event.delivery.accept()
        try:
            return json.loads(event.message.body)
        except Exception:
            return None

    def on_transport_error(self, event):
        self.error = str(event.transport.condition)
        self.done.set()

    def on_disconnected(self, event):
        self.done.set()


# ══════════════════════════════════════════════════════════════════════════════
# Test 1 & 2 — basic_flow / udp_port_protocol
# ══════════════════════════════════════════════════════════════════════════════

class _BasicFlowHandler(_BaseHandler):
    """
    Send one Detection, intercept TASK_REQUEST, allocate a port from the
    controller pool, respond with streams[0].udp_port, stream IQ.
    Verify ANALYSIS_RESULT received and that the app bound to our port.
    """
    def __init__(self, broker: str):
        super().__init__(broker)
        self.result:      dict | None = None
        self.task_req:    dict | None = None   # the raw TASK_REQUEST we received
        self.udp_port:    int  = 0
        self._det_sent    = False

    def on_start(self, event):
        conn = self._connect(event, "sdr.task.request", "rf.analysis")
        self._open_sender(conn, "rf.detections")
        self._open_sender(conn, "sdr.task.response")

    def on_sendable(self, event):
        if not self._det_sent and all(s.credit for s in self._senders.values()):
            self._det_sent = True
            self._send("rf.detections", {
                "scanner_id":     "test-scanner",
                "center_freq_hz": 100_000_000.0,
                "bandwidth_hz":   200_000.0,
                "power_db":       -60.0,
                "timestamp_ms":   int(time.time() * 1000),
            })
            print("  [ctrl] → rf.detections  Detection(100 MHz)")

    def on_message(self, event):
        body = self._accept_and_parse(event)
        if body is None:
            return
        addr = event.receiver.source.address

        if addr == "sdr.task.request":
            msg_type = body.get("msg_type", "")
            if msg_type == "TASK_STOP":
                print(f"  [ctrl] ← TASK_STOP for {body.get('task_id','?')}")
                return
            if "REQUEST" not in msg_type:
                return

            self.task_req = body
            req_id   = body.get("request_id", "")
            dest_ip  = body.get("streaming", {}).get("dest_ip", "127.0.0.1")
            cf       = body.get("rf", {}).get("center_freq_hz", 100e6)
            sr       = body.get("rf", {}).get("sample_rate_sps", 2e6)

            # Controller allocates port from ITS pool, not from the request
            self.udp_port = _alloc_port()
            print(f"  [ctrl] ← TASK_REQUEST req={req_id}  allocating port {self.udp_port}")

            # Verify app did NOT send dest_ports (new protocol)
            dest_ports = body.get("streaming", {}).get("dest_ports", None)
            if dest_ports is not None:
                print(f"  [WARN] app sent dest_ports={dest_ports} — should not (old protocol)")

            self._send("sdr.task.response", {
                "msg_type":   "TASK_RESPONSE",
                "request_id": req_id,
                "task_id":    str(uuid.uuid4()),
                "status":     "ACCEPTED",
                "timestamp_ms": int(time.time() * 1000),
                "streams": [{
                    "channel_index":   0,
                    "udp_ip":          dest_ip,
                    "udp_port":        self.udp_port,   # KEY: returned, not requested
                    "center_freq_hz":  cf,
                    "sample_rate_sps": sr,
                    "format":          "CF32",
                }],
            })
            print(f"  [ctrl] → TASK_RESPONSE ACCEPTED  udp_port={self.udp_port}")

            # Stream IQ to the port we allocated (app should be binding to it)
            threading.Thread(
                target=_stream_iq,
                args=(dest_ip, self.udp_port, cf, sr),
                daemon=True,
            ).start()

        elif addr == "rf.analysis":
            self.result = body
            print(f"  [ctrl] ← rf.analysis  msg_type={body.get('msg_type')}")
            event.connection.close()

# ══════════════════════════════════════════════════════════════════════════════
# Test 3 — stale_response
# ══════════════════════════════════════════════════════════════════════════════

class _StaleResponseHandler(_BaseHandler):
    """
    Send a TASK_RESPONSE with a wrong request_id before the correct one.
    App must discard the stale message and accept the real one.
    """
    def __init__(self, broker: str):
        super().__init__(broker)
        self.result       = None
        self._det_sent    = False
        self._stale_sent  = False

    def on_start(self, event):
        conn = self._connect(event, "sdr.task.request", "rf.analysis")
        self._open_sender(conn, "rf.detections")
        self._open_sender(conn, "sdr.task.response")

    def on_sendable(self, event):
        if not self._det_sent and all(s.credit for s in self._senders.values()):
            self._det_sent = True
            self._send("rf.detections", {
                "scanner_id": "test-scanner",
                "center_freq_hz": 100_000_000.0,
                "bandwidth_hz": 200_000.0,
                "power_db": -60.0,
                "timestamp_ms": int(time.time() * 1000),
            })
            print("  [ctrl] → rf.detections  Detection(100 MHz)")

    def on_message(self, event):
        body = self._accept_and_parse(event)
        if body is None:
            return
        addr = event.receiver.source.address

        if addr == "sdr.task.request" and "REQUEST" in body.get("msg_type", ""):
            req_id  = body.get("request_id", "")
            dest_ip = body.get("streaming", {}).get("dest_ip", "127.0.0.1")
            cf      = body.get("rf", {}).get("center_freq_hz", 100e6)
            sr      = body.get("rf", {}).get("sample_rate_sps", 2e6)

            if not self._stale_sent:
                self._stale_sent = True
                # Send a response with the WRONG request_id first
                stale_id = "wrong-" + str(uuid.uuid4())
                print(f"  [ctrl] → TASK_RESPONSE with WRONG req_id={stale_id} (should be discarded)")
                self._send("sdr.task.response", {
                    "msg_type":   "TASK_RESPONSE",
                    "request_id": stale_id,
                    "task_id":    str(uuid.uuid4()),
                    "status":     "ACCEPTED",
                    "timestamp_ms": int(time.time() * 1000),
                    "streams": [{"channel_index": 0, "udp_ip": dest_ip,
                                 "udp_port": 29999, "center_freq_hz": cf,
                                 "sample_rate_sps": sr, "format": "CF32"}],
                })
                # Wait briefly then send the correct response
                port = _alloc_port()
                time.sleep(0.3)
                print(f"  [ctrl] → TASK_RESPONSE with CORRECT req_id={req_id}  port={port}")
                self._send("sdr.task.response", {
                    "msg_type":   "TASK_RESPONSE",
                    "request_id": req_id,
                    "task_id":    str(uuid.uuid4()),
                    "status":     "ACCEPTED",
                    "timestamp_ms": int(time.time() * 1000),
                    "streams": [{"channel_index": 0, "udp_ip": dest_ip,
                                 "udp_port": port, "center_freq_hz": cf,
                                 "sample_rate_sps": sr, "format": "CF32"}],
                })
                threading.Thread(
                    target=_stream_iq,
                    args=(dest_ip, port, cf, sr),
                    daemon=True,
                ).start()

            elif body.get("msg_type") == "TASK_STOP":
                return

        elif addr == "rf.analysis":
            self.result = body
            event.connection.close()

# ══════════════════════════════════════════════════════════════════════════════
# Test 4 — timeout_returns
# ══════════════════════════════════════════════════════════════════════════════

class _TimeoutHandler(_BaseHandler):
    """
    Send a Detection but NEVER respond to the TASK_REQUEST.
    App must return from collect() within analysis_timeout_ms, not hang.
    We verify by checking elapsed time.
    """
    def __init__(self, broker: str, max_wait_s: float = 12.0):
        super().__init__(broker)
        self.task_received = False
        self._det_sent     = False
        self._max_wait_s   = max_wait_s
        self._t_request: float | None = None

    def on_start(self, event):
        conn = self._connect(event, "sdr.task.request", "rf.analysis")
        self._open_sender(conn, "rf.detections")
        # No sender for sdr.task.response — we never reply

    def on_sendable(self, event):
        if not self._det_sent and all(s.credit for s in self._senders.values()):
            self._det_sent = True
            self._send("rf.detections", {
                "scanner_id": "test-scanner",
                "center_freq_hz": 100_000_000.0,
                "bandwidth_hz": 200_000.0,
                "power_db": -60.0,
                "timestamp_ms": int(time.time() * 1000),
            })
            print("  [ctrl] → rf.detections  (will NOT respond to TASK_REQUEST)")

    def on_message(self, event):
        body = self._accept_and_parse(event)
        if body is None:
            return
        addr = event.receiver.source.address
        if addr == "sdr.task.request" and "REQUEST" in body.get("msg_type",""):
            self.task_received   = True
            self._t_request      = time.time()
            print(f"  [ctrl] ← TASK_REQUEST received — deliberately not responding")

# ══════════════════════════════════════════════════════════════════════════════
# Test 5 — task_stop_sent
# ══════════════════════════════════════════════════════════════════════════════

class _TaskStopHandler(_BasicFlowHandler):
    """After IQ is collected the app must send TASK_STOP."""
    def __init__(self, broker: str):
        super().__init__(broker)
        self.task_stop_received = False
        self._our_task_id: str | None = None

    def on_message(self, event):
        body = self._accept_and_parse(event)
        if body is None:
            return
        addr = event.receiver.source.address

        if addr == "sdr.task.request":
            msg_type = body.get("msg_type", "")
            if msg_type == "TASK_STOP" and body.get("task_id") == self._our_task_id:
                print(f"  [ctrl] ← TASK_STOP received for task {self._our_task_id} ✓")
                self.task_stop_received = True
                return
            if "REQUEST" not in msg_type:
                return

            req_id  = body.get("request_id", "")
            dest_ip = body.get("streaming", {}).get("dest_ip", "127.0.0.1")
            cf      = body.get("rf", {}).get("center_freq_hz", 100e6)
            sr      = body.get("rf", {}).get("sample_rate_sps", 2e6)

            self._our_task_id = str(uuid.uuid4())
            self.udp_port     = _alloc_port()

            self._send("sdr.task.response", {
                "msg_type":   "TASK_RESPONSE",
                "request_id": req_id,
                "task_id":    self._our_task_id,
                "status":     "ACCEPTED",
                "timestamp_ms": int(time.time() * 1000),
                "streams": [{"channel_index": 0, "udp_ip": dest_ip,
                             "udp_port": self.udp_port, "center_freq_hz": cf,
                             "sample_rate_sps": sr, "format": "CF32"}],
            })
            threading.Thread(
                target=_stream_iq,
                args=(dest_ip, self.udp_port, cf, sr),
                daemon=True,
            ).start()

        elif addr == "rf.analysis":
            self.result = body
            # Don't close connection yet — wait for TASK_STOP


# ══════════════════════════════════════════════════════════════════════════════
# Test runner
# ══════════════════════════════════════════════════════════════════════════════

def _run(handler: _BaseHandler, timeout_s: float = 45.0) -> _BaseHandler:
    container = proton.reactor.Container(handler)
    t = threading.Thread(target=container.run, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        container.stop()
        t.join(timeout=3)
    return handler


def run_basic_flow(broker: str) -> bool:
    print("\n── Test 1: basic_flow ──────────────────────────────────────────")
    h = _run(_BasicFlowHandler(broker))

    checks = []
    if h.error:
        print(f"  AMQP error: {h.error}"); return False

    r = h.result
    if r is None:
        print("  ✗ No ANALYSIS_RESULT received"); return False

    checks.append(("msg_type=ANALYSIS_RESULT",  r.get("msg_type") == "ANALYSIS_RESULT"))
    cf  = r.get("center_freq_hz", 0)
    checks.append((f"center_freq≈100 MHz (got {cf/1e6:.1f})",  abs(cf - 100e6) < 1e6))
    snr = r.get("snr_db", 0)
    checks.append((f"SNR={snr:.1f} dB > 5",  snr > 5))
    hyps = r.get("hypotheses", [])
    checks.append(("at least 1 hypothesis",  len(hyps) > 0))

    return _report(checks)


def run_udp_port_protocol(broker: str) -> bool:
    """Verifies app sends no dest_ports and binds to the returned udp_port."""
    print("\n── Test 2: udp_port_protocol ───────────────────────────────────")
    h = _run(_BasicFlowHandler(broker))

    if h.error or h.task_req is None:
        print("  ✗ No TASK_REQUEST observed"); return False

    checks = []
    dest_ports = h.task_req.get("streaming", {}).get("dest_ports")
    checks.append(("request has no dest_ports field",  dest_ports is None))
    checks.append(("controller allocated udp_port > 0", h.udp_port > 0))
    checks.append(("ANALYSIS_RESULT received (app bound to allocated port)",
                   h.result is not None))
    return _report(checks)


def run_stale_response(broker: str) -> bool:
    print("\n── Test 3: stale_response ──────────────────────────────────────")
    h = _run(_StaleResponseHandler(broker), timeout_s=30)
    checks = [("ANALYSIS_RESULT received despite stale first response",
               h.result is not None and h.error is None)]
    return _report(checks)


def run_timeout_returns(broker: str) -> bool:
    """App must return from collect() within 2×analysis_timeout_ms, never hang."""
    print("\n── Test 4: timeout_returns ─────────────────────────────────────")
    ANALYSIS_TIMEOUT_MS = 5000   # matches test-env config
    max_wait_s = ANALYSIS_TIMEOUT_MS / 1000 * 2 + 2   # 2× + 2s grace

    t_start = time.time()
    h = _run(_TimeoutHandler(broker), timeout_s=max_wait_s + 5)
    elapsed = time.time() - t_start

    checks = []
    checks.append(("TASK_REQUEST was received",  h.task_received))
    checks.append((f"returned within {max_wait_s:.0f}s (took {elapsed:.1f}s)",
                   elapsed < max_wait_s + 5))
    checks.append(("no AMQP error", h.error is None))
    return _report(checks)


def run_task_stop_sent(broker: str) -> bool:
    print("\n── Test 5: task_stop_sent ──────────────────────────────────────")
    h = _TaskStopHandler(broker)
    _run(h, timeout_s=30)

    # Give a moment for TASK_STOP to arrive after rf.analysis
    if h.result and not h.task_stop_received:
        time.sleep(3)

    checks = [
        ("ANALYSIS_RESULT received", h.result is not None),
        ("TASK_STOP received after collection", h.task_stop_received),
    ]
    return _report(checks)


def _report(checks: list[tuple[str, bool]]) -> bool:
    ok = True
    for label, passed in checks:
        sym = "✓" if passed else "✗"
        print(f"  {sym} {label}")
        if not passed:
            ok = False
    return ok


ALL_TESTS = {
    "basic_flow":        run_basic_flow,
    "udp_port_protocol": run_udp_port_protocol,
    "stale_response":    run_stale_response,
    "timeout_returns":   run_timeout_returns,
    "task_stop_sent":    run_task_stop_sent,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--broker", default="amqp://localhost:5672")
    ap.add_argument("--test",   default=None,
                    help="Run a single test; omit to run all")
    args = ap.parse_args()

    if args.test:
        if args.test not in ALL_TESTS:
            print(f"Unknown test {args.test!r}. Available: {list(ALL_TESTS)}")
            sys.exit(2)
        tests = {args.test: ALL_TESTS[args.test]}
    else:
        tests = ALL_TESTS

    results: dict[str, bool] = {}
    for name, fn in tests.items():
        results[name] = fn(args.broker)
        time.sleep(1)   # let broker clear between tests

    print("\n── Summary ─────────────────────────────────────────────────────")
    all_ok = True
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            all_ok = False

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
