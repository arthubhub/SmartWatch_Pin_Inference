#!/usr/bin/env python3
"""
Extended IMU + PIN visualization tool.

Features:
- Displays dataset info (sample count, sequences per PIN)
- Compare multiple sequences (IDs)
- Compare occurrences of a same PIN
- Compare occurrences per digit
- Compare all occurrences of one PIN vs another PIN (by digit)
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import pyarrow.parquet as pq

# ------------------- Configuration -------------------
DATA_PATH = Path("data/sequences/sequences.parquet")  # or .jsonl

# ------------------- Load the dataset -------------------
def load_jsonl(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples

def load_parquet(path):
    table = pq.read_table(path)
    return table.to_pylist()

def load_dataset(path):
    if path.suffix == ".jsonl":
        return load_jsonl(path)
    elif path.suffix == ".parquet":
        return load_parquet(path)
    else:
        raise ValueError("Unsupported format: use .jsonl or .parquet")

# ------------------- Info summary -------------------
def summarize_dataset(samples):
    from statistics import mean

    print("\n📊 Dataset Summary:")
    print(f"  → Total samples: {len(samples)}")

    # Count occurrences of each PIN
    pins = [s["pin_label"] for s in samples]
    from collections import Counter
    pin_counts = Counter(pins)
    print("  → Number of sequences per PIN:")
    for pin, count in pin_counts.items():
        print(f"     {pin}: {count}")

    # Compute stats per digit position
    digit_lengths = {i: [] for i in range(4)}  # assuming 4 digits
    for s in samples:
        for i, win in enumerate(s["sensor_values"]):
            digit_lengths[i].append(len(win))

    print("\n  → Average number of IMU samples per digit:")
    for i in range(4):
        if not digit_lengths[i]:
            continue
        lens = digit_lengths[i]
        print(f"     Digit {i+1}: mean={mean(lens):.1f}, min={min(lens)}, max={max(lens)}")

    # Also overall stats
    all_lens = [l for sub in digit_lengths.values() for l in sub]
    if all_lens:
        print(f"\n  → Overall mean window length: {mean(all_lens):.1f} samples")
    print("")


# ------------------- Utility -------------------
def extract_axes_values(win):
    if not win:
        return [], [], [], [], []
    if isinstance(win[0], dict):
        ax = [w["ax"] for w in win]
        ay = [w["ay"] for w in win]
        az = [w["az"] for w in win]
        gx = [w["gx"] for w in win]
        gz = [w["gz"] for w in win]
    else:
        ax = [w[0] for w in win]
        ay = [w[1] for w in win]
        az = [w[2] for w in win]
        gx = [w[3] for w in win]
        gz = [w[4] for w in win]
    return ax, ay, az, gx, gz

# ------------------- Visualization -------------------
def plot_sample(sample, ax_acc=None, ax_gyro=None, color=None):
    pin = sample["pin_label"]
    sid = sample["id"]
    sensor_values = sample["sensor_values"]

    if ax_acc is None or ax_gyro is None:
        fig, (ax_acc, ax_gyro) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        fig.suptitle(f"IMU Sequence for PIN {pin} (ID={sid})")

    t_offset = 0
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, win in enumerate(sensor_values):
        n = len(win)
        if n == 0:
            continue
        t = range(t_offset, t_offset + n)
        axv, ayv, azv, gxv, gzv = extract_axes_values(win)
        c = color or colors[i % len(colors)]
        label = f"Digit {i+1} '{pin[i]}' (ID={sid})"

        ax_acc.plot(t, axv, color=c, alpha=0.8, label=label + " ax")
        ax_acc.plot(t, ayv, color=c, alpha=0.4)
        ax_acc.plot(t, azv, color=c, alpha=0.2)
        ax_gyro.plot(t, gxv, color=c, alpha=0.8, label=label + " gx")
        ax_gyro.plot(t, gzv, color=c, alpha=0.4, linestyle="dotted")
        t_offset += n

    ax_acc.set_title("Accelerometer (ax, ay, az)")
    ax_gyro.set_title("Gyroscope (gx, gz)")
    ax_gyro.set_xlabel("Sample index")
    ax_acc.legend(fontsize=8)
    ax_acc.grid(True, linestyle="--", alpha=0.5)
    ax_gyro.grid(True, linestyle="--", alpha=0.5)
    return ax_acc, ax_gyro


# ------------------- Comparison Modes -------------------
def compare_sequences(samples, ids):
    fig, (ax_acc, ax_gyro) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"Comparison of sequences: {ids}")
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for idx, sid in enumerate(ids):
        try:
            sample = next(s for s in samples if s["id"] == sid)
        except StopIteration:
            print(f"ID {sid} not found, skipping.")
            continue
        plot_sample(sample, ax_acc, ax_gyro, color=palette[idx % len(palette)])
    plt.tight_layout()
    plt.show()


def compare_same_pin(samples, pin):
    same_pin_samples = [s for s in samples if s["pin_label"] == pin]
    if len(same_pin_samples) < 2:
        print(f"Not enough sequences for PIN {pin} to compare.")
        return
    ids = [s["id"] for s in same_pin_samples]
    print(f"Comparing {len(ids)} sequences with PIN {pin}: IDs = {ids}")
    compare_sequences(samples, ids)


def compare_same_pin_by_digit(samples, pin):
    same_pin_samples = [s for s in samples if s["pin_label"] == pin]
    if len(same_pin_samples) < 2:
        print(f"Not enough sequences for PIN {pin} to compare by digit.")
        return

    fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=False)
    fig.suptitle(f"PIN {pin} — Comparison by Digit Transitions", fontsize=14)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i in range(4):  # 4 digits
        ax_acc, ax_gyro = axes[i]
        for idx, sample in enumerate(same_pin_samples):
            if i >= len(sample["sensor_values"]):
                continue
            win = list(sample["sensor_values"][i])
            axv, ayv, azv, gxv, gzv = extract_axes_values(win)
            t = range(len(axv))
            c = palette[idx % len(palette)]
            ax_acc.plot(t, axv, color=c, alpha=0.8, label=f"Seq {sample['id']} ax")
            ax_acc.plot(t, ayv, color=c, alpha=0.5)
            ax_acc.plot(t, azv, color=c, alpha=0.3)
            ax_gyro.plot(t, gxv, color=c, alpha=0.9, linewidth=1.8, label=f"Seq {sample['id']} gx")
            ax_gyro.plot(t, gzv, color=c, alpha=0.4, linestyle="dotted", linewidth=1.0)
        ax_acc.set_title(f"Digit {i+1} '{pin[i]}' — Accelerometer")
        ax_gyro.set_title(f"Digit {i+1} '{pin[i]}' — Gyroscope")
        ax_acc.grid(True, linestyle="--", alpha=0.4)
        ax_gyro.grid(True, linestyle="--", alpha=0.4)
        ax_acc.legend(fontsize=7)
        ax_gyro.legend(fontsize=7)

    plt.tight_layout()
    plt.show()


def compare_pins(samples, pin_a, pin_b):
    """Compare all occurrences of one PIN vs another PIN (by digit).
    PIN A = orange, PIN B = blue.
    gx = solid, gz = dotted.
    """
    data_a = [s for s in samples if s["pin_label"] == pin_a]
    data_b = [s for s in samples if s["pin_label"] == pin_b]
    if not data_a or not data_b:
        print("❌ One or both PINs have no samples.")
        return

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=False)
    fig.suptitle(f"Comparison: PIN {pin_a} (orange) vs PIN {pin_b} (blue)", fontsize=14)

    colors = {pin_a: "#ff7f0e", pin_b: "#1f77b4"}  # orange / blue

    for i in range(4):
        ax = axes[i]
        ax.set_title(f"Digit {i+1} transition")
        ax.grid(True, linestyle="--", alpha=0.4)

        # --- Plot both PINs ---
        for pin, data in [(pin_a, data_a), (pin_b, data_b)]:
            for sample in data:
                if i >= len(sample["sensor_values"]):
                    continue
                win = list(sample["sensor_values"][i])
                _, _, _, gxv, gzv = extract_axes_values(win)
                t = range(len(gxv))
                c = colors[pin]

                # gx = solid, gz = dotted
                ax.plot(
                    t, gxv,
                    color=c, alpha=0.7, linewidth=1.8,
                    linestyle="-", label=f"{pin} gx" if sample == data[0] else ""
                )
                ax.plot(
                    t, gzv,
                    color=c, alpha=0.4, linewidth=1.0,
                    linestyle="dotted", label=f"{pin} gz" if sample == data[0] else ""
                )

        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


# ------------------- Main -------------------
if __name__ == "__main__":
    samples = load_dataset(DATA_PATH)
    summarize_dataset(samples)

    print("Available options:")
    print("  [1] Visualize a single sequence")
    print("  [2] Compare multiple sequences (by IDs)")
    print("  [3] Compare multiple occurrences of the same PIN")
    print("  [4] Compare same PIN by digit transitions (4×2 plots)")
    print("  [5] Compare all occurrences of one PIN vs another PIN (by digit)")
    choice = input("Select an option (1–5): ").strip()

    if choice == "1":
        ids = [s["id"] for s in samples]
        print("Available IDs:", ids)
        while True:
            try:
                choice = int(input("Enter sample ID to visualize: "))
                sample = next(s for s in samples if s["id"] == choice)
                break
            except (ValueError, StopIteration):
                print("Invalid ID, please try again.")
        plot_sample(sample)
        plt.show()

    elif choice == "2":
        ids_input = input("Enter sequence IDs to compare (comma-separated): ")
        ids = [int(i.strip()) for i in ids_input.split(",") if i.strip()]
        compare_sequences(samples, ids)

    elif choice == "3":
        pin = input("Enter the PIN to compare its occurrences: ").strip()
        compare_same_pin(samples, pin)

    elif choice == "4":
        pin = input("Enter the PIN to compare its occurrences (by digit): ").strip()
        compare_same_pin_by_digit(samples, pin)

    elif choice == "5":
        pin_a = input("Enter first PIN: ").strip()
        pin_b = input("Enter second PIN: ").strip()
        compare_pins(samples, pin_a, pin_b)

    else:
        print("Invalid choice.")
