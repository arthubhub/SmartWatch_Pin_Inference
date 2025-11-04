"""Configuration dataclasses for the IMU PIN collector."""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CollectorConfig:
    serial_port: str
    baudrate: int = 460800
    sampling_rate: int = 200
    print_every: int = 1000
    raw_out: Path | None = None


@dataclass
class DatasetConfig:
    dataset_out: Path
    sampling_rate: int = 200
    pre_first_ms: int = 150  # pre-roll before first keypress 
    pre_ms: int = 0        # per-digit pre window<- not used anymore
    post_ms: int = 0       # per-digit post window
    post_last_ms: int = 50    # post-roll after last keypress (digit 4)

@dataclass
class WebConfig:
    host: str = '0.0.0.0'
    port: int = 5000
