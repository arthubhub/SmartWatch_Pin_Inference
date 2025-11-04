"""Thread-safe time-indexed ring buffer for IMU samples."""
import threading
from collections import deque
from typing import Deque, List

from .models import Sample


class IMURing:
    """Thread-safe time-indexed ring buffer of IMU samples."""

    def __init__(self, max_seconds: float = 120.0, target_hz: int = 200):
        """
        Initialize ring buffer.
        
        Args:
            max_seconds: Maximum time window to store (seconds)
            target_hz: Expected sampling rate (Hz)
        """
        self.lock = threading.Lock()
        self.ring: Deque[Sample] = deque(maxlen=int(max_seconds * target_hz * 1.5))
        self.target_hz = target_hz

    def push(self, s: Sample) -> None:
        """Add a sample to the ring buffer."""
        with self.lock:
            self.ring.append(s)

    def get_window(self, t0_ns: int, t1_ns: int) -> List[Sample]:
        """
        Return samples with t0_ns <= t <= t1_ns.
        
        Args:
            t0_ns: Start time (nanoseconds)
            t1_ns: End time (nanoseconds)
            
        Returns:
            List of samples within the time window
        """
        with self.lock:
            if not self.ring:
                return []
            # Fast skip when window is entirely newer than our last sample
            if t0_ns > self.ring[-1].t_ns:
                return []
            # Linear scan is OK for moderate sizes
            return [s for s in self.ring if t0_ns <= s.t_ns <= t1_ns]

    def earliest_time(self) -> int | None:
        """Get timestamp of earliest sample in buffer."""
        with self.lock:
            return self.ring[0].t_ns if self.ring else None

    def latest_time(self) -> int | None:
        """Get timestamp of latest sample in buffer."""
        with self.lock:
            return self.ring[-1].t_ns if self.ring else None