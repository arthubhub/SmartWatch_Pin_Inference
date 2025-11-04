"""IMU data models."""
from dataclasses import dataclass


@dataclass
class Sample:
    """Single IMU sample with timestamp and sensor values."""
    t_ns: int      # nanosecond timestamp (perf_counter_ns)
    ax: float      # acceleration x (g)
    ay: float      # acceleration y (g)
    az: float      # acceleration z (g)
    gx: float      # gyro x / pitch rate (deg/s)
    gz: float      # gyro z / yaw rate (deg/s)