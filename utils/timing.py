"""Timing utilities for monotonic timestamps."""
import time

# Authoritative time base: monotonic, process-wide
now_ns = time.perf_counter_ns