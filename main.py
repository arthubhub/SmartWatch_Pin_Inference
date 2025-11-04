#!/Users/mcarthur/pyvenv/venv/bin/python3
"""
End-to-end IMU + PIN dataset collector.

Main entry point that orchestrates:
- IMU data collection from Arduino via serial
- Flask web interface for PIN entry
- Dataset storage in JSONL and Parquet formats
"""
import argparse
from pathlib import Path

from config import CollectorConfig, DatasetConfig, WebConfig
from dataset.writer import SequenceDatasetWriter
from imu.ring_buffer import IMURing
from imu.serial_collector import SerialCollector
from webapp.app import create_app


def main():
    """Main entry point."""
    # Create default config instances to extract default values
    default_collector = CollectorConfig(serial_port='')
    default_dataset = DatasetConfig(dataset_out=Path('data/sequences'))
    default_web = WebConfig()
    
    parser = argparse.ArgumentParser(
        description='IMU + PIN Dataset Collector (Flask + Serial)'
    )
    
    # Serial / IMU configuration
    parser.add_argument(
        '--serial-port',
        required=True,
        help='Serial port (e.g., /dev/ttyUSB0, COM3)'
    )
    parser.add_argument(
        '--baud',
        type=int,
        default=default_collector.baudrate,
        help=f'Baud rate (default: {default_collector.baudrate})'
    )
    parser.add_argument(
        '--sampling-rate',
        type=int,
        default=default_collector.sampling_rate,
        help=f'Sampling rate in Hz (default: {default_collector.sampling_rate})'
    )
    parser.add_argument(
        '--print-every',
        type=int,
        default=default_collector.print_every,
        help=f'Print debug info every N samples (default: {default_collector.print_every})'
    )
    parser.add_argument(
        '--raw-out',
        type=Path,
        default=None,
        help='Optional: directory to write raw IMU parquet'
    )
    
    # Dataset configuration
    parser.add_argument(
        '--dataset-out',
        type=Path,
        default=default_dataset.dataset_out,
        help=f'Output directory for dataset (default: {default_dataset.dataset_out})'
    )
    parser.add_argument(
        '--pre-first-ms',
        type=int,
        default=default_dataset.pre_first_ms,
        help=f'Pre-roll before first keypress in ms (default: {default_dataset.pre_first_ms})'
    )
    parser.add_argument(
        '--pre-ms',
        type=int,
        default=default_dataset.pre_ms,
        help=f'Per-digit pre window in ms (default: {default_dataset.pre_ms})'
    )
    parser.add_argument(
        '--post-ms',
        type=int,
        default=default_dataset.post_ms,
        help=f'Per-digit post window in ms (default: {default_dataset.post_ms})'
    )
    parser.add_argument(
        '--post-last-ms',
        type=int,
        default=default_dataset.post_last_ms,
        help=f'Post-roll after last keypress in ms (default: {default_dataset.post_last_ms})'
    )
    
    # Web server configuration
    parser.add_argument(
        '--web-host',
        default=default_web.host,
        help=f'Web server host (default: {default_web.host})'
    )
    parser.add_argument(
        '--web-port',
        type=int,
        default=default_web.port,
        help=f'Web server port (default: {default_web.port})'
    )

    args = parser.parse_args()

    # Initialize configurations from parsed arguments
    collector_config = CollectorConfig(
        serial_port=args.serial_port,
        baudrate=args.baud,
        sampling_rate=args.sampling_rate,
        print_every=args.print_every,
        raw_out=args.raw_out
    )
    
    dataset_config = DatasetConfig(
        dataset_out=args.dataset_out,
        sampling_rate=args.sampling_rate,
        pre_first_ms=args.pre_first_ms,
        pre_ms=args.pre_ms,
        post_ms=args.post_ms,
        post_last_ms=args.post_last_ms
    )
    
    web_config = WebConfig(
        host=args.web_host,
        port=args.web_port
    )

    # Initialize shared IMU ring buffer
    imu_ring = IMURing(max_seconds=120, target_hz=collector_config.sampling_rate)
    
    # Initialize serial collector
    collector = SerialCollector(
        port=collector_config.serial_port,
        baudrate=collector_config.baudrate,
        print_every=collector_config.print_every,
        imu_ring=imu_ring
    )
    collector.start(write_raw_dir=collector_config.raw_out)
    
    # Initialize dataset writer
    seq_writer = SequenceDatasetWriter(
        dataset_config.dataset_out,
        sampling_rate=dataset_config.sampling_rate
    )
    
    # Create Flask app
    app = create_app(
        seq_writer=seq_writer,
        imu_ring=imu_ring,
        sampling_rate=dataset_config.sampling_rate,
        pre_first_ms=dataset_config.pre_first_ms,
        pre_ms=dataset_config.pre_ms,
        post_ms=dataset_config.post_ms,
        post_last_ms=dataset_config.post_last_ms
    )

    try:
        print(f"[Web] Serving on http://{web_config.host}:{web_config.port}")
        app.run(host=web_config.host, port=web_config.port, threaded=True)
    finally:
        print("[Shutdown] Closing writers and serial…")
        collector.stop()
        seq_writer.close()


if __name__ == '__main__':
    main()