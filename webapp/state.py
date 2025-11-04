"""Web application state management."""
import threading
from dataclasses import dataclass, field
from typing import List


@dataclass
class SequenceState:
    """Tracks current PIN entry sequence state."""
    mode: str = "train"  # or "test"
    digits: List[str] = field(default_factory=list)
    t_presses: List[int] = field(default_factory=list)  # perf_counter_ns at server
    t_start_ns: int | None = None  # first press - pre_first_ms
    assemble_timer: threading.Timer | None = None

    def reset(self) -> None:
        """Clear current sequence state."""
        self.digits.clear()
        self.t_presses.clear()
        self.t_start_ns = None
        if self.assemble_timer:
            self.assemble_timer.cancel()
            self.assemble_timer = None