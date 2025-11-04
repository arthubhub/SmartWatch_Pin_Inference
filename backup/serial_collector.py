#!/usr/bin/env python3
"""
End-to-end IMU + PIN dataset collector
- Reads IMU frames from Arduino (GY-61 or similar) via SerialCollector
- Serves a Flask keypad UI (train/test modes)
- Captures keypress timestamps (server-side monotonic clock)
- Assembles per-digit IMU windows with 400ms pre-roll for the first digit
- Saves dataset samples to JSONL and Parquet with the structure:
    {
      "id": <int>,
      "pin_label": "1234",
      "sensor_values": [
        [[ax, ay, az, gx, gz], [ax, ay, az, gx, gz], ...],  # digit 1 window
        [...],                                               # digit 2 window
        [...],                                               # digit 3 window
        [...],                                               # digit 4 window
      ],
      "sampling_rate": 200
    }
Notes
- Gyro axes mapping: we export gx := pitch_rate, gz := yaw_rate from the collector.
- Authoritative time base: time.perf_counter_ns() everywhere.
- Requires: pyserial, flask, pyarrow
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, List, Tuple

import serial  # pyserial
import struct

from flask import Flask, request, jsonify, Response

import pyarrow as pa
import pyarrow.parquet as pq

# ----------------------------- Timing helpers -----------------------------
_now_ns = time.perf_counter_ns  # monotonic, process-wide

# -------------------------- Shared IMU ring buffer -------------------------
@dataclass
class Sample:
    t_ns: int
    ax: float
    ay: float
    az: float
    gx: float  # mapped from pitch_rate
    gz: float  # mapped from yaw_rate


class IMURing:
    """Thread-safe time-indexed ring of IMU samples."""

    def __init__(self, max_seconds: float = 120.0, target_hz: int = 200):
        self.lock = threading.Lock()
        self.ring: Deque[Sample] = deque(maxlen=int(max_seconds * target_hz * 1.5))
        self.target_hz = target_hz

    def push(self, s: Sample):
        with self.lock:
            self.ring.append(s)

    def get_window(self, t0_ns: int, t1_ns: int) -> List[Sample]:
        """Return samples with t0_ns <= t <= t1_ns."""
        with self.lock:
            if not self.ring:
                return []
            # Fast skip when window is entirely newer than our last sample
            if t0_ns > self.ring[-1].t_ns:
                return []
            # Linear scan is OK for moderate sizes; could binary-search by time if needed
            return [s for s in self.ring if t0_ns <= s.t_ns <= t1_ns]

    def earliest_time(self) -> int | None:
        with self.lock:
            return self.ring[0].t_ns if self.ring else None

    def latest_time(self) -> int | None:
        with self.lock:
            return self.ring[-1].t_ns if self.ring else None


# ------------------------------ Serial reader -----------------------------
class SerialCollector:
    """Collects calibrated IMU data from Arduino (binary protocol, no sync)."""

    MAGIC_DATA = 0xA1B2C3D4  # 54-byte IMU frame
    FRAME_SIZE = 54

    def __init__(self, port: str, baudrate: int = 460800, print_every: int = 50, imu_ring: IMURing | None = None):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.running = False
        self.print_every = max(1, int(print_every))
        self._valid_count = 0
        self.imu_ring = imu_ring or IMURing()

        # Optional: write raw frames parquet (disabled by default here)
        self.write_raw = False
        self.raw_schema = pa.schema([
            ("t_ns", pa.int64()),
            ("seq", pa.int32()),
            ("ax_g", pa.float32()),
            ("ay_g", pa.float32()),
            ("az_g", pa.float32()),
            ("pitch_rate", pa.float32()),
            ("yaw_rate", pa.float32()),
        ])
        self.raw_writer = None
        self.raw_batch: List[dict] = []
        self.raw_dir: Path | None = None

    def connect(self) -> bool:
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=0.05)
            time.sleep(2.0)
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
            print(f"[Serial] Connected {self.port} @ {self.baudrate}")
            return True
        except Exception as e:
            print(f"[Serial] Failed to connect: {e}")
            return False

    def start(self, write_raw_dir: Path | None = None):
        if not self.connect():
            raise RuntimeError("Cannot open serial port")
        self.running = True
        if write_raw_dir is not None:
            self.write_raw = True
            self.raw_dir = Path(write_raw_dir)
            self.raw_dir.mkdir(parents=True, exist_ok=True)
        t = threading.Thread(target=self._read_loop, daemon=True)
        t.start()

    def stop(self):
        self.running = False
        try:
            if self.serial:
                self.serial.close()
        finally:
            self.serial = None
        if self.raw_writer:
            self._flush_raw(force=True)
            self.raw_writer.close()
            self.raw_writer = None
        print("[Serial] Stopped")

    # ----------------------- internal -----------------------
    def _read_loop(self):
        buffer = bytearray()
        magic = struct.pack('<I', self.MAGIC_DATA)

        while self.running:
            try:
                n = self.serial.in_waiting if self.serial else 0
                if n:
                    buffer += self.serial.read(n)

                while len(buffer) >= 4:
                    if buffer.startswith(magic):
                        if len(buffer) < self.FRAME_SIZE:
                            break
                        frame = buffer[:self.FRAME_SIZE]
                        del buffer[:self.FRAME_SIZE]
                        parsed = self._parse_frame(frame)
                        if parsed:
                            self._valid_count += 1
                            s = Sample(
                                t_ns=parsed['t_ns'],
                                ax=parsed['ax_g'],
                                ay=parsed['ay_g'],
                                az=parsed['az_g'],
                                gx=parsed['pitch_rate'],  # exported as gx
                                gz=parsed['yaw_rate'],    # exported as gz
                            )
                            self.imu_ring.push(s)

                            if self.write_raw:
                                self.raw_batch.append({
                                    't_ns': s.t_ns,
                                    'seq': parsed['seq'],
                                    'ax_g': s.ax,
                                    'ay_g': s.ay,
                                    'az_g': s.az,
                                    'pitch_rate': parsed['pitch_rate'],
                                    'yaw_rate': parsed['yaw_rate'],
                                })
                                if len(self.raw_batch) >= 1000:
                                    self._flush_raw()

                            if (self._valid_count % self.print_every) == 0:
                                print(f"[DATA] seq={parsed['seq']} ax={s.ax:.3f} ay={s.ay:.3f} az={s.az:.3f} gx={s.gx:.3f} gz={s.gz:.3f}")
                    else:
                        idx = buffer.find(magic, 1)
                        if idx != -1:
                            del buffer[:idx]
                        else:
                            buffer[:] = buffer[-3:]
                            break

                if not n:
                    time.sleep(0.002)
            except Exception as e:
                print(f"[Serial] Read error: {e}")
                time.sleep(0.05)

    def _parse_frame(self, data: bytes):
        try:
            magic, seq, tick_us, ax_raw, ay_raw, az_raw, gp_raw, gy_raw, \
            ax_g, ay_g, az_g, pitch_rate, yaw_rate, pitch_filt, roll_filt = \
                struct.unpack('<IIQhhhhhfffffff', data)
            if magic != self.MAGIC_DATA:
                return None
            return {
                'seq': seq,
                'tick_us': tick_us,
                'ax_raw': ax_raw,
                'ay_raw': ay_raw,
                'az_raw': az_raw,
                'gp_raw': gp_raw,
                'gy_raw': gy_raw,
                'ax_g': float(ax_g),
                'ay_g': float(ay_g),
                'az_g': float(az_g),
                'pitch_rate': float(pitch_rate),
                'yaw_rate': float(yaw_rate),
                'pitch_filtered': float(pitch_filt),
                'roll_filtered': float(roll_filt),
                't_ns': _now_ns(),  # authoritative host timestamp
            }
        except Exception as e:
            print(f"[Serial] Parse error: {e}")
            return None

    def _flush_raw(self, force: bool = False):
        if not self.raw_batch and not force:
            return
        try:
            if self.raw_writer is None:
                ts = time.strftime('%Y%m%d_%H%M%S')
                out = self.raw_dir / f"imu_raw_{ts}.parquet"
                self.raw_writer = pq.ParquetWriter(out, self.raw_schema)
                print(f"[RAW] Writing to {out}")
            arrays = [
                pa.array([r['t_ns'] for r in self.raw_batch], type=pa.int64()),
                pa.array([r['seq'] for r in self.raw_batch], type=pa.int32()),
                pa.array([r['ax_g'] for r in self.raw_batch], type=pa.float32()),
                pa.array([r['ay_g'] for r in self.raw_batch], type=pa.float32()),
                pa.array([r['az_g'] for r in self.raw_batch], type=pa.float32()),
                pa.array([r['pitch_rate'] for r in self.raw_batch], type=pa.float32()),
                pa.array([r['yaw_rate'] for r in self.raw_batch], type=pa.float32()),
            ]
            batch = pa.RecordBatch.from_arrays(arrays, schema=self.raw_schema)
            self.raw_writer.write_batch(batch)
            print(f"[RAW] Flushed {len(self.raw_batch)} samples")
        finally:
            self.raw_batch = []


# ------------------------------ Dataset writer -----------------------------

class SequenceDatasetWriter:
    def __init__(self, out_dir, sampling_rate=200):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.out_dir / 'sequences.jsonl'
        self.round_val = 3

        sample_struct = pa.struct([
            ("ax", pa.float32()),
            ("ay", pa.float32()),
            ("az", pa.float32()),
            ("gx", pa.float32()),
            ("gz", pa.float32()),
        ])
        self.schema = pa.schema([
            ("id", pa.int64()),
            ("pin_label", pa.string()),
            ("sensor_values", pa.list_(pa.list_(sample_struct))),
            ("sampling_rate", pa.int16()),
        ])
        self.parquet_path = self.out_dir / 'sequences.parquet'
        self.writer = pq.ParquetWriter(self.parquet_path, self.schema)
        self._next_id = 1
        self.sampling_rate = int(sampling_rate)
        self._lock = threading.Lock()

    def append(self, pin_label: str, digit_windows: List[List[Sample]]) -> int:
        with self._lock:
            seq_id = self._next_id
            self._next_id += 1

            # Save JSONL (human-readable)
            py_rec = {
                "id": seq_id,
                "pin_label": pin_label,
                "sensor_values": [
                    [[round(float(s.ax),self.round_val), round(float(s.ay),self.round_val), round(float(s.az),self.round_val), round(float(s.gx),self.round_val), round(float(s.gz),self.round_val)] for s in win]
                    for win in digit_windows
                ],
                "sampling_rate": self.sampling_rate,
            }
            with open(self.jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(py_rec) + "\n")

            # ✅ Correct Parquet build
            sensor_values_type = self.schema.field("sensor_values").type
            
            digit_structs_py = [
                [
                    {"ax": round(float(s.ax),self.round_val), "ay": round(float(s.ay),self.round_val), "az": round(float(s.az),self.round_val), "gx": round(float(s.gx),self.round_val), "gz": round(float(s.gz),self.round_val)}
                    for s in win
                ]
                for win in digit_windows
            ]

            # Wrap in list so batch length = 1
            sensor_values_array = pa.array([digit_structs_py], type=sensor_values_type)

            batch = pa.RecordBatch.from_arrays(
                [
                    pa.array([seq_id], type=pa.int64()),
                    pa.array([pin_label], type=pa.string()),
                    sensor_values_array,
                    pa.array([self.sampling_rate], type=pa.int16()),
                ],
                schema=self.schema,
            )

            self.writer.write_batch(batch)
            return seq_id
    def close(self):
        with self._lock:
            if self.writer:
                self.writer.close()
                self.writer = None




# ------------------------------ Sequence state -----------------------------
@dataclass
class SequenceState:
    mode: str = "train"  # or "test"
    digits: List[str] = field(default_factory=list)
    t_presses: List[int] = field(default_factory=list)  # perf_counter_ns at server
    t_start_ns: int | None = None  # first press - 400ms
    assemble_timer: threading.Timer | None = None

    def reset(self):
        self.digits.clear()
        self.t_presses.clear()
        self.t_start_ns = None
        if self.assemble_timer:
            self.assemble_timer.cancel()
            self.assemble_timer = None

# ------------------------------ html index page -----------------------------
HTML_INDEX = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no" />
  <title>Unlock</title>
  <style>
    html, body {
      margin: 0;
      padding: 0;
      height: 100%;
      width: 100%;
      background-color: #000;
      color: #fff;
      font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', system-ui, sans-serif;
      display: flex;
      justify-content: center;
      align-items: center;
      overflow: hidden;
    }
    .container {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 100%;
      width: 100%;
    }
    .status {
      text-align: center;
      margin-bottom: 30px;
      min-height: 88px;
    }
    #typed {
      display: inline-block;
      letter-spacing: 10px;
      font-size: 36px;
      font-weight: 400;
      min-height: 44px;       /* keeps same height whether empty or not */
      line-height: 44px;
      transition: opacity 0.2s ease-in-out;
    }
    #msg {
      font-size: 16px;
      margin-top: 10px;
      color: #bbb;
      min-height: 20px;       /* prevent page jump when message appears/disappears */
    }
    .keypad {
      display: grid;
      grid-template-columns: repeat(3, 100px);
      grid-gap: 20px;
      justify-content: center;
      align-content: center;
    }
    button.key {
      width: 100px;
      height: 100px;
      border-radius: 50%;
      border: none;
      font-size: 32px;
      font-weight: 500;
      color: #fff;
      background: rgba(255, 255, 255, 0.15);
      backdrop-filter: blur(10px);
      cursor: pointer;
      transition: background 0.15s, transform 0.1s;
    }
    button.key:active {
      transform: scale(0.94);
      background: rgba(255, 255, 255, 0.25);
    }
    button.action {
      font-size: 18px;
      background: transparent;
      color: #bbb;
      text-transform: uppercase;
      letter-spacing: 1px;
      border: none;
      margin-top: 20px;
      cursor: pointer;
    }
    .mode {
      position: absolute;
      top: 15px;
      left: 15px;
      font-size: 14px;
      color: #bbb;
    }
    .mode label {
      margin-right: 10px;
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="mode">
      Mode:
      <label><input type="radio" name="mode" value="train" checked> train</label>
      <label><input type="radio" name="mode" value="test"> test</label>
    </div>
    <div class="status">
      <div id="typed"></div>
      <div id="msg"></div>
    </div>
    <div class="keypad">
      <button class="key">1</button>
      <button class="key">2</button>
      <button class="key">3</button>
      <button class="key">4</button>
      <button class="key">5</button>
      <button class="key">6</button>
      <button class="key">7</button>
      <button class="key">8</button>
      <button class="key">9</button>
      <div></div>
      <button class="key">0</button>
      <div></div>
    </div>
    <button id="undo" class="action">Undo</button>
    <button id="abort" class="action">Abort</button>
  </div>

  <script>
    const typed = document.getElementById('typed');
    const msg = document.getElementById('msg');

    function currentMode(){
      const el = document.querySelector('input[name="mode"]:checked');
      return el ? el.value : 'train';
    }

    function setMsg(t){ msg.textContent = t; }

    async function sendKey(d){
      const res = await fetch('/api/key', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({digit: String(d), mode: currentMode()})
      });
      const j = await res.json();
      typed.textContent = j.typed || '';
      setMsg(j.message || '');
      if ((j.count || 0) >= 4) {
        typed.textContent = '';
      }
    }

    async function undo(){
      const res = await fetch('/api/undo', {method:'POST'});
      const j = await res.json();
      typed.textContent = j.typed || '';
      setMsg(j.message || '');
    }

    async function abort(){
      const res = await fetch('/api/abort', {method:'POST'});
      const j = await res.json();
      typed.textContent = '';
      setMsg(j.message || '');
    }

    document.querySelectorAll('button.key').forEach(b => {
      b.addEventListener('click', () => sendKey(b.textContent.trim()));
    });
    document.getElementById('undo').addEventListener('click', undo);
    document.getElementById('abort').addEventListener('click', abort);
  </script>
</body>
</html>
"""




def create_app(seq_writer: SequenceDatasetWriter, imu_ring: IMURing,
                sampling_rate: int,
                pre_first_ms: int, pre_ms: int, post_ms: int) -> Flask:
    app = Flask(__name__)
    state = SequenceState()

    def assemble_and_persist():
        if len(state.digits) != 4 or len(state.t_presses) != 4:
            return

        digit_windows: List[List[Sample]] = []
        press_times = state.t_presses

        # Start 200 ms before first press
        t0 = press_times[0] - pre_first_ms * 1_000_000

        for i in range(4):
            # End time: next press or last press + post_ms
            if i < 3:
                t1 = press_times[i + 1]
            else:
                t1 = press_times[i] + post_ms * 1_000_000

            wins = imu_ring.get_window(t0, t1)
            digit_windows.append(wins)
            t0 = t1  # next digit starts where previous ended

        pin = ''.join(state.digits)
        seq_id = seq_writer.append(pin, digit_windows)
        print(f"[SEQ] Saved id={seq_id} pin={pin} lens={[len(w) for w in digit_windows]}")
        state.reset()

    @app.get('/')
    def index() -> Response:
        return Response(HTML_INDEX, mimetype='text/html')

    @app.post('/api/key')
    def api_key():
        data = request.get_json(force=True)
        digit = str(data.get('digit', ''))
        mode = str(data.get('mode', 'train'))
        if not digit.isdigit() or len(digit) != 1:
            return jsonify({"error": "digit must be 0-9"}), 400

        t_now = _now_ns()
        if not state.digits:
            # first key → start window 400ms earlier
            state.t_start_ns = t_now - pre_first_ms * 1_000_000
            state.mode = 'test' if mode == 'test' else 'train'
        state.digits.append(digit)
        state.t_presses.append(t_now)

        # When 4th digit arrives, schedule assembly after post_ms to ensure we captured tail
        message = ''
        if len(state.digits) == 4:
            # schedule assemble
            if state.assemble_timer:
                state.assemble_timer.cancel()
            delay_s = (post_ms + 50) / 1000.0  # small margin
            state.assemble_timer = threading.Timer(delay_s, assemble_and_persist)
            state.assemble_timer.start()
            if state.mode == 'test':
                message = 'prediction: [feature needs to be added]'
            else:
                message = 'saved sequence'

        return jsonify({
            'typed': ''.join(state.digits),
            'count': len(state.digits),
            'mode': state.mode,
            'message': message
        })

    @app.post('/api/undo')
    def api_undo():
        if state.digits:
            state.digits.pop()
            state.t_presses.pop()
        return jsonify({'typed': ''.join(state.digits), 'message': 'undone' if state.digits else 'cleared'})

    @app.post('/api/abort')
    def api_abort():
        state.reset()
        return jsonify({'message': 'aborted'})

    @app.get('/api/status')
    def api_status():
        return jsonify({
            'typed': ''.join(state.digits),
            'digits': state.digits,
            't_presses_ns': state.t_presses,
            'ring_earliest': imu_ring.earliest_time(),
            'ring_latest': imu_ring.latest_time(),
        })

    return app


# ---------------------------------- Main -----------------------------------

def main():
    parser = argparse.ArgumentParser(description='IMU + PIN Dataset Collector (Flask + Serial)')
    parser.add_argument('--serial-port', required=True, help='Serial port (e.g., /dev/ttyUSB0, COM3)')
    parser.add_argument('--baud', type=int, default=460800)
    parser.add_argument('--sampling-rate', type=int, default=200)
    parser.add_argument('--raw-out', type=Path, default=None, help='Optional: directory to write raw IMU parquet')
    parser.add_argument('--dataset-out', type=Path, default=Path('data/sequences'))
    parser.add_argument('--pre-first-ms', type=int, default=400, help='pre-roll before first keypress (ms)')
    parser.add_argument('--pre-ms', type=int, default=200, help='per-digit pre window (ms)')
    parser.add_argument('--post-ms', type=int, default=300, help='per-digit post window (ms)')
    parser.add_argument('--web-host', default='127.0.0.1')
    parser.add_argument('--web-port', type=int, default=5000)
    parser.add_argument('--print-every', type=int, default=50)

    args = parser.parse_args()

    imu_ring = IMURing(max_seconds=120, target_hz=args.sampling_rate)
    collector = SerialCollector(args.serial_port, baudrate=args.baud, print_every=args.print_every, imu_ring=imu_ring)
    collector.start(write_raw_dir=args.raw_out)

    seq_writer = SequenceDatasetWriter(args.dataset_out, sampling_rate=args.sampling_rate)

    app = create_app(seq_writer, imu_ring, args.sampling_rate, args.pre_first_ms, args.pre_ms, args.post_ms)

    try:
        print(f"[Web] Serving on http://{args.web_host}:{args.web_port}")
        app.run(host=args.web_host, port=args.web_port, threaded=True)
    finally:
        print("[Shutdown] Closing writers and serial…")
        collector.stop()
        seq_writer.close()


if __name__ == '__main__':
    main()
