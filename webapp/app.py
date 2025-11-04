"""Flask web application for PIN entry and data collection."""
import threading
from typing import List

from flask import Flask, Response, jsonify, request

from dataset.writer import SequenceDatasetWriter
from imu.models import Sample
from imu.ring_buffer import IMURing
from utils.timing import now_ns

from .state import SequenceState
from .templates import HTML_INDEX


def create_app(
    seq_writer: SequenceDatasetWriter,
    imu_ring: IMURing,
    sampling_rate: int,
    pre_first_ms: int,
    pre_ms: int,
    post_ms: int,
    post_last_ms: int
) -> Flask:
    """
    Create Flask application for PIN entry interface.
    
    Args:
        seq_writer: Dataset writer instance
        imu_ring: Shared IMU ring buffer
        sampling_rate: Expected sampling rate (Hz)
        pre_first_ms: Pre-roll before first keypress (ms)
        pre_ms: Per-digit pre window (ms, currently unused)
        post_ms: Per-digit post window (ms, currently unused)
        post_last_ms: Post-roll after last keypress (ms)
        
    Returns:
        Flask application instance
    """
    app = Flask(__name__)
    state = SequenceState()

    def assemble_and_persist() -> None:
        """Assemble IMU windows for each digit and persist to dataset."""
        if len(state.digits) != 4 or len(state.t_presses) != 4:
            return

        digit_windows: List[List[Sample]] = []
        press_times = state.t_presses

        # Start pre_first_ms before first press
        t0 = press_times[0] - pre_first_ms * 1_000_000

        for i in range(4):
            # End time depends on digit position
            if i < 3:
                # Digits 1-3: end at next press (no gap, no overlap)
                t1 = press_times[i] + post_ms * 1_000_000
            else:
                # Digit 4: extend up to post_last_ms OR ring buffer end
                t1_desired = press_times[i] + post_last_ms * 1_000_000
                ring_latest = imu_ring.latest_time()
                t1 = min(t1_desired, ring_latest) if ring_latest else t1_desired

            wins = imu_ring.get_window(t0, t1)
            digit_windows.append(wins)
            
            # Next window starts where this one ended
            t0 = press_times[i]

        pin = ''.join(state.digits)
        seq_id = seq_writer.append(pin, digit_windows)
        print(f"[SEQ] Saved id={seq_id} pin={pin} lens={[len(w) for w in digit_windows]}")
        state.reset()

    @app.get('/')
    def index() -> Response:
        """Serve main HTML interface."""
        return Response(HTML_INDEX, mimetype='text/html')

    @app.post('/api/key')
    def api_key():
        """Handle keypress event."""
        data = request.get_json(force=True)
        digit = str(data.get('digit', ''))
        mode = str(data.get('mode', 'train'))
        
        if not digit.isdigit() or len(digit) != 1:
            return jsonify({"error": "digit must be 0-9"}), 400

        t_now = now_ns()
        
        if not state.digits:
            # First key → start window pre_first_ms earlier
            state.t_start_ns = t_now - pre_first_ms * 1_000_000
            state.mode = 'test' if mode == 'test' else 'train'
        
        state.digits.append(digit)
        state.t_presses.append(t_now)

        # When 4th digit arrives, schedule assembly after post_last_ms
        message = ''
        if len(state.digits) == 4:
            if state.assemble_timer:
                state.assemble_timer.cancel()
            
            # Schedule assemble with small margin to ensure we captured tail
            delay_s = (post_last_ms + 50) / 1000.0
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
        """Undo last digit entry."""
        if state.digits:
            state.digits.pop()
            state.t_presses.pop()
        return jsonify({
            'typed': ''.join(state.digits),
            'message': 'undone' if state.digits else 'cleared'
        })

    @app.post('/api/abort')
    def api_abort():
        """Abort current sequence."""
        state.reset()
        return jsonify({'message': 'aborted'})

    @app.get('/api/status')
    def api_status():
        """Get current system status."""
        return jsonify({
            'typed': ''.join(state.digits),
            'digits': state.digits,
            't_presses_ns': state.t_presses,
            'ring_earliest': imu_ring.earliest_time(),
            'ring_latest': imu_ring.latest_time(),
        })

    return app