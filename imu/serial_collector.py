"""Serial collector for Arduino IMU data."""
import struct
import threading
import time
from pathlib import Path
from typing import List

import pyarrow as pa
import pyarrow.parquet as pq
import serial

from utils.timing import now_ns
from .models import Sample
from .ring_buffer import IMURing


class SerialCollector:
    """Collects calibrated IMU data from Arduino (binary protocol)."""

    MAGIC_DATA = 0xA1B2C3D4  # 54-byte IMU frame
    FRAME_SIZE = 54

    def __init__(
        self,
        port: str,
        baudrate: int = 460800,
        print_every: int = 50,
        imu_ring: IMURing | None = None
    ):
        """
        Initialize serial collector.
        
        Args:
            port: Serial port path (e.g., /dev/ttyUSB0, COM3)
            baudrate: Serial baud rate
            print_every: Print debug info every N samples
            imu_ring: Shared IMU ring buffer (created if None)
        """
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.running = False
        self.print_every = max(1, int(print_every))
        self._valid_count = 0
        self.imu_ring = imu_ring or IMURing()

        # Optional: write raw frames parquet
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
        """Open serial connection."""
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

    def start(self, write_raw_dir: Path | None = None) -> None:
        """
        Start collection thread.
        
        Args:
            write_raw_dir: Optional directory to write raw IMU parquet files
        """
        if not self.connect():
            raise RuntimeError("Cannot open serial port")
        self.running = True
        if write_raw_dir is not None:
            self.write_raw = True
            self.raw_dir = Path(write_raw_dir)
            self.raw_dir.mkdir(parents=True, exist_ok=True)
        t = threading.Thread(target=self._read_loop, daemon=True)
        t.start()

    def stop(self) -> None:
        """Stop collection and close serial port."""
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

    # ----------------------- Internal methods -----------------------

    def _read_loop(self) -> None:
        """Main read loop (runs in background thread)."""
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

    def _parse_frame(self, data: bytes) -> dict | None:
        """Parse binary IMU frame."""
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
                't_ns': now_ns(),  # authoritative host timestamp
            }
        except Exception as e:
            print(f"[Serial] Parse error: {e}")
            return None

    def _flush_raw(self, force: bool = False) -> None:
        """Flush raw sample batch to parquet file."""
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