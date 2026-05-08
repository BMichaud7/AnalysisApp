#!/usr/bin/env python3
"""
End-to-end test for AnalysisApp — simulates the full SdrResourceManager role.

Flow:
  1. Send Detection event → rf.detections  (triggers AnalysisApp)
  2. AnalysisApp sends TASK_REQUEST → sdr.task.request  (we intercept this)
  3. We reply TASK_RESPONSE → sdr.task.response  (simulate ResourceManager accept)
  4. We stream synthetic IQ packets over UDP to the port AnalysisApp is listening on
  5. AnalysisApp analyzes and publishes → rf.analysis  (we verify this)

Usage:
    python3 e2e_test.py --broker amqp://10.89.0.2:5672
"""

from __future__ import annotations
import argparse, json, math, socket, struct, sys, threading, time, uuid
import numpy as np

try:
    import proton, proton.handlers, proton.reactor
except ImportError:
    sys.exit("pip install python-qpid-proton")

# ── IQ packet format (matches SdrTaskApi IqPacketHeader, 32 bytes) ───────────
#   uint32 magic         = 0x49515030 ("IQP0")
#   uint32 sequence
#   uint64 timestamp_ns
#   uint64 center_freq_hz
#   uint32 sample_rate
#   uint16 num_samples
#   uint8  channel_index
#   uint8  flags
IQ_MAGIC  = 0x49515030
IQ_HEADER = struct.Struct("<I I Q Q I H B B")   # little-endian, 32 bytes
assert IQ_HEADER.size == 32

def build_iq_packet(seq: int, iq: np.ndarray, cf: float, sr: float) -> bytes:
    n   = len(iq)
    hdr = IQ_HEADER.pack(IQ_MAGIC, seq, 0, int(cf), int(sr), n, 0, 0)
    samples = np.empty(n * 2, dtype=np.float32)
    samples[0::2] = iq.real
    samples[1::2] = iq.imag
    return hdr + samples.tobytes()


def gen_fm(n: int = 4096, sr: float = 2e6, dev: float = 75e3) -> np.ndarray:
    """FM signal at 100 MHz — AnalysisApp should classify as FM / FM Broadcast."""
    rng   = np.random.default_rng(42)
    audio = rng.standard_normal(n).astype(np.float32)
    audio /= np.max(np.abs(audio)) + 1e-9
    phase = 2 * np.pi * dev / sr * np.cumsum(audio)
    iq    = np.exp(1j * phase).astype(np.complex64)
    return (iq / np.sqrt(np.mean(np.abs(iq)**2))).astype(np.complex64)


# ── AMQP orchestrator ─────────────────────────────────────────────────────────

class E2EHandler(proton.handlers.MessagingHandler):
    def __init__(self, broker: str, creds: tuple[str,str]):
        super().__init__()
        self.broker  = broker
        self.user, self.pwd = creds
        self.result: dict | None = None
        self.error:  str  | None = None
        self._senders: dict[str, proton.Sender] = {}
        self._det_sent  = False
        self._task_id   = str(uuid.uuid4())
        self._req_id    = str(uuid.uuid4())

    # ── Connect & open senders/receivers ─────────────────────────────────────
    def on_start(self, event):
        conn = event.container.connect(
            self.broker,
            user=self.user,
            password=self.pwd,
            sasl_enabled=True,
            allowed_mechs="PLAIN",
        )

        # Sender for: rf.detections, sdr.task.response
        self._senders["rf.detections"]      = event.container.create_sender(conn, "rf.detections")
        self._senders["sdr.task.response"]  = event.container.create_sender(conn, "sdr.task.response")

        # Receivers for: sdr.task.request (we intercept), rf.analysis (result)
        event.container.create_receiver(conn, "sdr.task.request")
        event.container.create_receiver(conn, "rf.analysis")
        print("[e2e] Connected to", self.broker)

    # ── When all senders are open, fire detection ─────────────────────────────
    def on_sendable(self, event):
        if not self._det_sent and all(s.credit for s in self._senders.values()):
            self._det_sent = True
            self._send_detection(event)

    def _send_detection(self, event):
        det = {
            "scanner_id":     "test-scanner",
            "center_freq_hz": 100_000_000.0,
            "bandwidth_hz":   200_000.0,
            "power_db":       -60.0,
            "timestamp_ms":   int(time.time() * 1000),
        }
        msg = proton.Message(body=json.dumps(det), content_type="application/json")
        self._senders["rf.detections"].send(msg)
        print("[e2e] → rf.detections  Detection(100 MHz, 200 kHz)")

    # ── Handle incoming messages ──────────────────────────────────────────────
    def on_message(self, event):
        try:
            body = json.loads(event.message.body)
        except Exception:
            body = {"raw": str(event.message.body)}

        topic = event.receiver.source.address

        if topic == "sdr.task.request":
            if body.get("msg_type") == "TASK_STOP":
                print(f"[e2e] ← sdr.task.request  TASK_STOP — sending ack")
                ack = {
                    "msg_type":      "TASK_RESPONSE",
                    "request_id":    body.get("request_id", ""),
                    "task_id":       body.get("task_id", self._task_id),
                    "status":        "ACCEPTED",
                    "reject_reason": "",
                    "timestamp_ms":  int(time.time() * 1000),
                    "streams":       [],
                }
                msg = proton.Message(body=json.dumps(ack),
                                     content_type="application/json")
                self._senders["sdr.task.response"].send(msg)
                return
            self._handle_task_request(body)

        elif topic == "rf.analysis":
            print("\n[e2e] ← rf.analysis  Analysis result received:")
            print(json.dumps(body, indent=2))
            self.result = body
            event.connection.close()

    def _handle_task_request(self, req: dict):
        print(f"[e2e] ← sdr.task.request  TASK_REQUEST req_id={req.get('request_id','?')}")

        # Extract UDP destination from request
        streaming = req.get("streaming", {})
        dest_ip   = streaming.get("dest_ip", "127.0.0.1")
        ports     = streaming.get("dest_ports", [20000])
        dest_port = ports[0] if ports else 20000
        sr        = req.get("rf", {}).get("sample_rate_sps", 2_000_000)
        cf        = req.get("rf", {}).get("center_freq_hz", 100e6)

        print(f"[e2e]   IQ destination: {dest_ip}:{dest_port}  sr={sr:.0f}")

        # Send TASK_RESPONSE (ACCEPTED)
        resp = {
            "msg_type":      "TASK_RESPONSE",
            "request_id":    req.get("request_id", self._req_id),
            "task_id":       self._task_id,
            "status":        "ACCEPTED",
            "reject_reason": "",
            "timestamp_ms":  int(time.time() * 1000),
            "streams": [{
                "channel_index":  0,
                "center_freq_hz": cf,
                "sample_rate_sps": sr,
                "dest_ip":        dest_ip,
                "dest_port":      dest_port,
            }],
        }
        msg = proton.Message(body=json.dumps(resp), content_type="application/json")
        self._senders["sdr.task.response"].send(msg)
        print(f"[e2e] → sdr.task.response  ACCEPTED task_id={self._task_id}")

        # Stream IQ in background
        threading.Thread(
            target=self._stream_iq,
            args=(dest_ip, dest_port, cf, sr),
            daemon=True,
        ).start()

    def _stream_iq(self, dest_ip: str, dest_port: int, cf: float, sr: float):
        time.sleep(0.2)   # give AnalysisApp time to start UDP receive loop
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        iq_base   = gen_fm(4096, sr)
        pkt_size  = 1024
        n_total   = 200_000
        iq_full   = np.tile(iq_base, math.ceil(n_total / len(iq_base)))[:n_total]

        for seq, off in enumerate(range(0, len(iq_full), pkt_size)):
            chunk = iq_full[off:off + pkt_size]
            pkt   = build_iq_packet(seq, chunk, cf, sr)
            sock.sendto(pkt, (dest_ip, dest_port))
            time.sleep(0.0005)

        sock.close()
        print(f"[e2e] Streamed {seq+1} IQ packets ({n_total} samples) → {dest_ip}:{dest_port}")

    def on_transport_error(self, event):
        self.error = str(event.transport.condition)
        print(f"[e2e] Transport error: {self.error}")

    def on_disconnected(self, event):
        pass


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--broker",  default="amqp://10.89.0.2:5672")
    ap.add_argument("--timeout", type=int, default=45)
    args = ap.parse_args()

    handler   = E2EHandler(args.broker, ("sdr_ctrl", "test_password"))
    container = proton.reactor.Container(handler)
    t = threading.Thread(target=container.run, daemon=True)
    t.start()

    deadline = time.time() + args.timeout
    while t.is_alive() and time.time() < deadline:
        t.join(timeout=1.0)

    if t.is_alive():
        print(f"\n[e2e] TIMEOUT after {args.timeout}s")
        sys.exit(2)
    if handler.error:
        print(f"\n[e2e] FAILED: {handler.error}")
        sys.exit(1)
    if not handler.result:
        print("\n[e2e] FAILED: no result received")
        sys.exit(1)

    # ── Validate result ───────────────────────────────────────────────────────
    r = handler.result
    checks = []

    msg_type = r.get("msg_type","")
    checks.append(("msg_type=ANALYSIS_RESULT",  msg_type == "ANALYSIS_RESULT"))

    cf = r.get("center_freq_hz", 0)
    checks.append((f"center_freq={cf/1e6:.1f} MHz",  abs(cf - 100e6) < 1e6))

    snr = r.get("snr_db", 0)
    checks.append((f"SNR={snr:.1f} dB > 5",  snr > 5))

    mod = r.get("modulation", {})
    analog = mod.get("analog","") if isinstance(mod, dict) else ""
    digital = mod.get("digital","") if isinstance(mod, dict) else ""
    detected = analog or digital or str(mod)
    is_fm = "FM" in detected.upper()
    checks.append((f"modulation contains FM (got {detected!r})",  is_fm))

    hyps = r.get("hypotheses", [])
    top  = hyps[0].get("system","") if hyps else ""
    conf = hyps[0].get("confidence", 0) if hyps else 0
    checks.append((f"hypothesis={top!r} conf={conf:.2f}",  bool(top)))

    print("\n[e2e] ── Result checks ──────────────────────────────")
    all_ok = True
    for label, ok in checks:
        print(f"  {'✓' if ok else '✗'} {label}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\n[e2e] ALL CHECKS PASSED ✓")
        sys.exit(0)
    else:
        print("\n[e2e] SOME CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
