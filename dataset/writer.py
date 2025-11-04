"""Dataset writer for IMU sequences."""
import json
import threading
from pathlib import Path
from typing import List

import pyarrow as pa
import pyarrow.parquet as pq

from imu.models import Sample


class SequenceDatasetWriter:
    """Writes PIN sequences with IMU data to JSONL and Parquet."""

    def __init__(self, out_dir: Path, sampling_rate: int = 200):
        """
        Initialize dataset writer.
        
        Args:
            out_dir: Output directory for dataset files
            sampling_rate: Expected sampling rate (Hz)
        """
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.out_dir / 'sequences.jsonl'
        self.round_val = 3

        # Define Parquet schema
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
        """
        Append a PIN sequence with IMU windows to the dataset.
        
        Args:
            pin_label: The PIN string (e.g., "1234")
            digit_windows: List of 4 windows, one per digit
            
        Returns:
            Sequence ID
        """
        with self._lock:
            seq_id = self._next_id
            self._next_id += 1

            # Save JSONL (human-readable)
            py_rec = {
                "id": seq_id,
                "pin_label": pin_label,
                "sensor_values": [
                    [
                        [
                            round(float(s.ax), self.round_val),
                            round(float(s.ay), self.round_val),
                            round(float(s.az), self.round_val),
                            round(float(s.gx), self.round_val),
                            round(float(s.gz), self.round_val)
                        ]
                        for s in win
                    ]
                    for win in digit_windows
                ],
                "sampling_rate": self.sampling_rate,
            }
            with open(self.jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(py_rec) + "\n")

            # Build Parquet record
            sensor_values_type = self.schema.field("sensor_values").type
            
            digit_structs_py = [
                [
                    {
                        "ax": round(float(s.ax), self.round_val),
                        "ay": round(float(s.ay), self.round_val),
                        "az": round(float(s.az), self.round_val),
                        "gx": round(float(s.gx), self.round_val),
                        "gz": round(float(s.gz), self.round_val)
                    }
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

    def close(self) -> None:
        """Close the Parquet writer."""
        with self._lock:
            if self.writer:
                self.writer.close()
                self.writer = None