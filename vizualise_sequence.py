#!/usr/bin/env python3
"""
Extended IMU + PIN visualization tool.

Features:
- Displays dataset info (sample count, sequences per PIN)
- Compare multiple sequences (IDs)
- Compare occurrences of a same PIN
- Compare occurrences per digit
- Compare all occurrences of one PIN vs another PIN (by digit)
- Normalize sequences to same length
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import pyarrow.parquet as pq
import numpy as np
from scipy.ndimage import gaussian_filter1d

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

    pin_arr=[ [pin, dict([["count",pins.count(pin)],["ids",[]]])] for pin in pins]
    pin_dict=dict(pin_arr)
    for s in samples:
        pin_dict[s["pin_label"]]["ids"].append(s["id"])
    

    
    print(pin_dict)


    for key, value in pin_dict.items():
        count,pinid = value["count"], value["ids"]
        print(f" {key} ids[{count}] = {pinid}")


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

def interpolate_signal(signal, target_length):
    """Interpolate a signal to target_length using linear interpolation."""
    if len(signal) == 0:
        return [0] * target_length
    if len(signal) == target_length:
        return signal
    x_old = np.linspace(0, 1, len(signal))
    x_new = np.linspace(0, 1, target_length)
    return np.interp(x_new, x_old, signal).tolist()

# ------------------- Visualization -------------------
def plot_sample(sample, ax_acc=None, ax_gyro_x=None,ax_gyro_z=None, color=None):
    pin = sample["pin_label"]
    sid = sample["id"]
    sensor_values = sample["sensor_values"]

    if ax_acc is None or ax_gyro_x is None or ax_gyro_z is None:
        fig, (ax_acc, ax_gyro_x, ax_gyro_z) = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
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
        ax_gyro_x.plot(t, gxv, color=c, alpha=1, label=label + " gx")
        ax_gyro_z.plot(t, gzv, color=c, alpha=1, label=label + " gz")
        t_offset += n

    ax_acc.set_title("Accelerometer (ax, ay, az)")
    ax_gyro_x.set_title("Gyroscope X")
    ax_gyro_x.set_xlabel("Sample index")
    ax_gyro_z.set_title("Gyroscope Z")
    ax_gyro_z.set_xlabel("Sample index")
    ax_acc.legend(fontsize=8)
    ax_acc.grid(True, linestyle="--", alpha=0.5)
    ax_gyro_x.grid(True, linestyle="--", alpha=0.5)
    ax_gyro_z.grid(True, linestyle="--", alpha=0.5)

    return ax_acc, ax_gyro_x, ax_gyro_z


# ------------------- Comparison Modes -------------------
def compare_sequences(samples, ids):
    from matplotlib.widgets import Slider
    
    # Get all samples to compare
    samples_to_plot = []
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for sid in ids:
        try:
            sample = next(s for s in samples if s["id"] == sid)
            samples_to_plot.append(sample)
        except StopIteration:
            print(f"ID {sid} not found, skipping.")
    
    if not samples_to_plot:
        print("No valid samples to plot.")
        return
    
    # Calculate max length per digit across all sequences
    max_digit_lengths = [0, 0, 0, 0]
    for sample in samples_to_plot:
        for i in range(min(4, len(sample["sensor_values"]))):
            max_digit_lengths[i] = max(max_digit_lengths[i], len(sample["sensor_values"][i]))
    
    # Initial offsets: [sequence_idx][digit_idx] = offset
    offsets = [[0, 0, 0, 0] for _ in range(len(samples_to_plot))]
    
    # Create figure with space for sliders
    num_sliders = len(samples_to_plot) * 4
    fig = plt.figure(figsize=(12, 10))
    
    # Create subplots for data (leave space at bottom for sliders)
    ax_acc = plt.subplot(3, 1, 1)
    ax_gyro_x = plt.subplot(3, 1, 2)
    ax_gyro_z = plt.subplot(3, 1, 3)
    
    #fig.suptitle(f"Comparison of sequences: {ids} (use sliders to align each digit)")
    plt.subplots_adjust(bottom= num_sliders * 0.012)
    
    
    def plot_all_samples():
        """Clear and replot all samples with current offsets."""
        ax_acc.clear()
        ax_gyro_x.clear()
        ax_gyro_z.clear()
        
        ax_acc.set_title("Accelerometer (ax, ay, az)")
        ax_gyro_x.set_title("Gyroscope X")
        ax_gyro_z.set_title("Gyroscope Z")
        ax_gyro_z.set_xlabel("Sample index")
        
        # Plot each sequence with its per-digit offsets
        for idx, sample in enumerate(samples_to_plot):
            pin = sample["pin_label"]
            sid = sample["id"]
            sensor_values = sample["sensor_values"]
            c = palette[idx % len(palette)]
            
            t_offset = 0
            
            for i, win in enumerate(sensor_values[:4]):  # Only first 4 digits
                n = len(win)
                if n == 0:
                    continue
                
                # Apply offset for this specific digit of this sequence
                t_offset += offsets[idx][i]
                
                t = range(t_offset, t_offset + n)
                axv, ayv, azv, gxv, gzv = extract_axes_values(win)
                
                label = f"Digit {i+1} '{pin[i]}' (ID={sid})"
                
                ax_acc.plot(t, axv, color=c, alpha=0.8, label=label + " ax")
                ax_acc.plot(t, ayv, color=c, alpha=0.4)
                ax_acc.plot(t, azv, color=c, alpha=0.2)
                ax_gyro_x.plot(t, gxv, color=c, alpha=1, label=label + " gx")
                ax_gyro_z.plot(t, gzv, color=c, alpha=1, label=label + " gz")
                
                # Draw vertical line at digit boundary
                ax_acc.axvline(t_offset, color=c, alpha=0.3, linestyle="dotted", linewidth=1.0)
                ax_gyro_x.axvline(t_offset, color=c, alpha=0.3, linestyle="dotted", linewidth=1.0)
                ax_gyro_z.axvline(t_offset, color=c, alpha=0.3, linestyle="dotted", linewidth=1.0)
                
                t_offset += n
            
            # Final boundary
            ax_acc.axvline(t_offset, color=c, alpha=0.3, linestyle="dotted", linewidth=1.0)
            ax_gyro_x.axvline(t_offset, color=c, alpha=0.3, linestyle="dotted", linewidth=1.0)
            ax_gyro_z.axvline(t_offset, color=c, alpha=0.3, linestyle="dotted", linewidth=1.0)
        
        #ax_acc.legend(fontsize=8)
        ax_acc.grid(True, linestyle="--", alpha=0.5)
        ax_gyro_x.grid(True, linestyle="--", alpha=0.5)
        ax_gyro_z.grid(True, linestyle="--", alpha=0.5)
        
        fig.canvas.draw_idle()
    
    # Create sliders - 4 per sequence
    sliders = []
    slider_height = 0.01
    slider_spacing = 0.01
    slider_idx = 0
    
    for seq_idx, sample in enumerate(samples_to_plot):
        for digit_idx in range(4):
            ax_slider = plt.axes([0.15, 0.02 + slider_idx * slider_spacing, 0.7, slider_height])
            
            # Slider range: ±max length for this digit
            max_shift = max_digit_lengths[digit_idx]
            slider = Slider(
                ax_slider, 
                f'ID{sample["id"]}-D{digit_idx+1}', 
                -max_shift, 
                max_shift, 
                valinit=0, 
                valstep=1
            )
            sliders.append(slider)
            
            def make_update(s_idx, d_idx):
                def update(val):
                    offsets[s_idx][d_idx] = int(val)
                    plot_all_samples()
                return update
            
            slider.on_changed(make_update(seq_idx, digit_idx))
            slider_idx += 1
    
    # Initial plot
    plot_all_samples()
    
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
    #fig.suptitle(f"PIN {pin} — Comparison by Digit Transitions", fontsize=14)
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


def normalize_sequences(samples):
    """Normalize all sequences to have the same length (max length found in dataset)."""
    print("\n🔄 Normalizing sequences...")
    
    # Find maximum length for each digit position
    max_lengths = [0, 0, 0, 0]
    for sample in samples:
        for i, win in enumerate(sample["sensor_values"]):
            if i < 4:
                max_lengths[i] = max(max_lengths[i], len(win))
    
    print(f"  → Maximum lengths per digit: {max_lengths}")
    print(f"  → Total max length: {sum(max_lengths)}")
    
    # Create normalized copies
    normalized_samples = []
    for sample in samples:
        new_sensor_values = []
        for i, win in enumerate(sample["sensor_values"]):
            if i >= 4:
                new_sensor_values.append(win)
                continue
                
            target_len = max_lengths[i]
            if len(win) == 0:
                new_sensor_values.append(win)
                continue
            
            # Extract all axes
            axv, ayv, azv, gxv, gzv = extract_axes_values(win)
            
            # Interpolate each axis
            axv_norm = interpolate_signal(axv, target_len)
            ayv_norm = interpolate_signal(ayv, target_len)
            azv_norm = interpolate_signal(azv, target_len)
            gxv_norm = interpolate_signal(gxv, target_len)
            gzv_norm = interpolate_signal(gzv, target_len)
            
            # Reconstruct window
            if isinstance(win[0], dict):
                new_win = [
                    {"ax": axv_norm[j], "ay": ayv_norm[j], "az": azv_norm[j], 
                     "gx": gxv_norm[j], "gz": gzv_norm[j]}
                    for j in range(target_len)
                ]
            else:
                new_win = [
                    [axv_norm[j], ayv_norm[j], azv_norm[j], gxv_norm[j], gzv_norm[j]]
                    for j in range(target_len)
                ]
            
            new_sensor_values.append(new_win)
        
        normalized_sample = sample.copy()
        normalized_sample["sensor_values"] = new_sensor_values
        normalized_samples.append(normalized_sample)
    
    print(f"✅ Normalized {len(normalized_samples)} sequences")
    
    # Save option
    save = input("\nSave normalized data? (y/n): ").strip().lower()
    if save == 'y':
        output_path = DATA_PATH.parent / f"{DATA_PATH.stem}_normalized{DATA_PATH.suffix}"
        
        if output_path.suffix == ".jsonl":
            with open(output_path, "w", encoding="utf-8") as f:
                for sample in normalized_samples:
                    f.write(json.dumps(sample) + "\n")
        elif output_path.suffix == ".parquet":
            import pyarrow as pa
            table = pa.Table.from_pylist(normalized_samples)
            pq.write_table(table, output_path)
        
        print(f"💾 Saved to: {output_path}")
    
    return normalized_samples


def detect_pin_patterns(samples, sample_id):
    """Detect PIN entry patterns using sliding window analysis on gx values.
    
    Modified Algorithm with configurable parameters:
    - Phase 1: Find first positive value (dynamic length, between 100-200 samples)
    - Phase 2: 20 sample gap
    - Phase 3: Find first value ≤5 (dynamic length, between 100-200 samples)
    - Phase 4: Search for stable window (30 samples in bounds, max 50 iterations)
    - Phase 5: Dynamic - starts at 50, grows until next 100 samples have variance < current/2
    - Phase 6: Validation on original data (the stable 100 samples after Phase 5)
    """
    
    # ============= CONFIGURABLE PARAMETERS =============
    # Phase 1 parameters
    PHASE1_MIN_LENGTH = 120  # Must find positive value after this
    PHASE1_MAX_LENGTH = 200  # Must find positive value before this
    
    # Phase 2 parameters
    PHASE2_GAP = 20  # Fixed gap between Phase 1 and Phase 3
    
    # Phase 3 parameters
    PHASE3_THRESHOLD = 5  # Signal must go <= this value
    PHASE3_MIN_LENGTH = 80  # Must find threshold after this
    PHASE3_MAX_LENGTH = 200  # Must find threshold before this
    
    # Phase 4 parameters (search for stable window)
    PHASE4_WINDOW_SIZE = 30  # Check this many samples
    PHASE4_MAX_ITERATIONS = 100  # Maximum search iterations
    PHASE4_MIN_BOUND = -15  # Same as Phase 5
    PHASE4_MAX_BOUND = 15   # Same as Phase 5
    
    # Phase 5 parameters (original data, dynamic length)
    PHASE5_INITIAL_LENGTH = 50  # Start with this many samples
    PHASE5_MAX_LENGTH = 500  # Maximum length
    PHASE5_LOOKAHEAD = 100  # Check next 100 samples
    PHASE5_VARIANCE_RATIO = 0.1  # Next variance must be < current * this
    PHASE5_MIN_BOUND = -30
    PHASE5_MAX_BOUND = 30
    
    # Phase 6 parameters (original data - the stable continuation)
    PHASE6_LENGTH = 100  # This is the lookahead window that became stable
    
    # Gaussian filter
    GAUSSIAN_SIGMA = 20
    
    # Sliding window
    WINDOW_JUMP_NORMAL = 20
    WINDOW_JUMP_AFTER_PATTERN = 100
    # ===================================================
    
    print("\n" + "="*80)
    print("🔧 CONFIGURATION PARAMETERS:")
    print("="*80)
    print(f"Phase 1: Find positive value in range [{PHASE1_MIN_LENGTH}, {PHASE1_MAX_LENGTH}] samples")
    print(f"Phase 2: Fixed gap of {PHASE2_GAP} samples")
    print(f"Phase 3: Find value ≤{PHASE3_THRESHOLD} in range [{PHASE3_MIN_LENGTH}, {PHASE3_MAX_LENGTH}] samples")
    print(f"Phase 4: Search for {PHASE4_WINDOW_SIZE} stable samples in [{PHASE4_MIN_BOUND}, {PHASE4_MAX_BOUND}] (max {PHASE4_MAX_ITERATIONS} iterations)")
    print(f"Phase 5: Dynamic length starting at {PHASE5_INITIAL_LENGTH}, max {PHASE5_MAX_LENGTH} in [{PHASE5_MIN_BOUND}, {PHASE5_MAX_BOUND}]")
    print(f"         Grows until next {PHASE5_LOOKAHEAD} samples have variance < {PHASE5_VARIANCE_RATIO} * current")
    print(f"Phase 6: {PHASE6_LENGTH} samples (the stable lookahead window)")
    print(f"Gaussian sigma: {GAUSSIAN_SIGMA}")
    print("="*80 + "\n")
    
    # Find the sample by ID
    try:
        sample = next(s for s in samples if s["id"] == sample_id)
    except StopIteration:
        print(f"❌ Sample ID {sample_id} not found.")
        return
    
    pin = sample["pin_label"]
    sensor_values = sample["sensor_values"]
    
    # Extract all gx values (concatenate, remove boundaries)
    all_gx_original = []
    digit_info = []
    
    for i, win in enumerate(sensor_values[:4]):
        start_idx = len(all_gx_original)
        _, _, _, gxv, _ = extract_axes_values(win)
        all_gx_original.extend(gxv)
        end_idx = len(all_gx_original)
        digit_info.append({
            'digit': i,
            'char': pin[i] if i < len(pin) else '?',
            'start': start_idx,
            'end': end_idx
        })
    
    all_gx_original = np.array(all_gx_original)
    
    if len(all_gx_original) == 0:
        print("❌ No gx data found.")
        return
    
    # Apply Gaussian filter
    all_gx_filtered = gaussian_filter1d(all_gx_original, sigma=GAUSSIAN_SIGMA)
    
    # Plot the original sequence with Gaussian filtered signal
    print(f"📊 Analyzing PIN {pin} (ID={sample_id})")
    print(f"📏 Total gx samples: {len(all_gx_original)}")
    
    fig_original, (ax_acc, ax_gyro_x, ax_gyro_z) = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig_original.suptitle(f"Original IMU Sequence for PIN {pin} (ID={sample_id})")
    
    plot_sample(sample, ax_acc, ax_gyro_x, ax_gyro_z)
    
    # Add Gaussian filtered gx
    t_filtered = np.arange(len(all_gx_filtered))
    ax_gyro_x.plot(t_filtered, all_gx_filtered, color='black', linewidth=2.5, 
                   alpha=0.7, label=f'Gaussian filtered (σ={GAUSSIAN_SIGMA})', linestyle='--')
    ax_gyro_x.legend(fontsize=8, loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    # Sliding window detection
    detected_patterns = []
    
    print("\n🔍 Starting pattern detection...")
    print("="*80)
    
    i = 0
    window_count = 0
    
    while i < len(all_gx_filtered):
        window_count += 1
        window_start = i
        
        # Check if we have enough data for minimum pattern
        min_required = PHASE1_MAX_LENGTH + PHASE2_GAP + PHASE3_MAX_LENGTH + PHASE4_MAX_ITERATIONS + PHASE4_WINDOW_SIZE + PHASE5_INITIAL_LENGTH + PHASE5_LOOKAHEAD + PHASE6_LENGTH
        if i + min_required > len(all_gx_filtered):
            if window_count % 50 == 0:
                print(f"\nWindow {window_count} at {i}: ⚠️ Not enough data remaining ({len(all_gx_filtered) - i} samples)")
            i += WINDOW_JUMP_NORMAL
            continue
        
        # ============= PHASE 1: Find first positive value =============
        phase1_start = window_start
        
        # Check if starting value is negative
        if all_gx_filtered[phase1_start] >= 0:
            if window_count % 50 == 0:
                print(f"\nWindow {window_count} at {i}: ❌ Phase 1 - Starting value not negative ({all_gx_filtered[phase1_start]:.2f})")
            i += WINDOW_JUMP_NORMAL
            continue
        
        # Search for first positive value in range [MIN, MAX]
        phase1_search_end = phase1_start + PHASE1_MAX_LENGTH
        if phase1_search_end > len(all_gx_filtered):
            i += WINDOW_JUMP_NORMAL
            continue
        
        phase1_search_data = all_gx_filtered[phase1_start:phase1_search_end]
        positive_indices = np.where(phase1_search_data > 0)[0]
        
        # Must find positive value AND it must be after PHASE1_MIN_LENGTH
        if len(positive_indices) == 0:
            if window_count % 50 == 0:
                print(f"\nWindow {window_count} at {i}: ❌ Phase 1 - No positive value found in {PHASE1_MAX_LENGTH} samples")
            i += WINDOW_JUMP_NORMAL
            continue
        
        first_positive_idx = positive_indices[0]
        
        if first_positive_idx < PHASE1_MIN_LENGTH:
            if window_count % 50 == 0:
                print(f"\nWindow {window_count} at {i}: ❌ Phase 1 - Positive value found too early (at sample {first_positive_idx}, need >{PHASE1_MIN_LENGTH})")
            i += WINDOW_JUMP_NORMAL
            continue
        
        phase1_end = phase1_start + first_positive_idx
        phase1_length = phase1_end - phase1_start
        phase1_data = all_gx_filtered[phase1_start:phase1_end]
        
        print(f"\n{'='*80}")
        print(f"✅ Window {window_count} at index {window_start}: Phase 1 PASSED")
        print(f"   Range: [{phase1_start}:{phase1_end}], Length: {phase1_length}")
        print(f"   Min: {phase1_data.min():.2f}, Max: {phase1_data.max():.2f}, Mean: {phase1_data.mean():.2f}")
        print(f"   Variance: {np.var(phase1_data):.2f}, Std: {np.std(phase1_data):.2f}")
        
        # ============= PHASE 2: Gap =============
        phase2_start = phase1_end
        phase2_end = phase2_start + PHASE2_GAP
        phase2_data = all_gx_filtered[phase2_start:phase2_end]
        
        print(f"✅ Phase 2 PASSED (gap phase)")
        print(f"   Range: [{phase2_start}:{phase2_end}], Length: {PHASE2_GAP}")
        print(f"   Min: {phase2_data.min():.2f}, Max: {phase2_data.max():.2f}, Mean: {phase2_data.mean():.2f}")
        
        # ============= PHASE 3: Find first value ≤ threshold =============
        phase3_start = phase2_end
        phase3_search_end = phase3_start + PHASE3_MAX_LENGTH
        
        if phase3_search_end > len(all_gx_filtered):
            print(f"❌ Phase 3 - Not enough data")
            i += WINDOW_JUMP_NORMAL
            continue
        
        phase3_search_data = all_gx_filtered[phase3_start:phase3_search_end]
        under_threshold_indices = np.where(phase3_search_data <= PHASE3_THRESHOLD)[0]
        
        # Must find threshold AND it must be after PHASE3_MIN_LENGTH
        if len(under_threshold_indices) == 0:
            print(f"❌ Phase 3 FAILED - No value ≤{PHASE3_THRESHOLD} found in {PHASE3_MAX_LENGTH} samples")
            print(f"   Range: [{phase3_start}:{phase3_search_end}], Min: {phase3_search_data.min():.2f}")
            i += WINDOW_JUMP_NORMAL
            continue
        
        first_threshold_idx = under_threshold_indices[0]
        
        if first_threshold_idx < PHASE3_MIN_LENGTH:
            print(f"❌ Phase 3 FAILED - Value ≤{PHASE3_THRESHOLD} found too early (at sample {first_threshold_idx}, need >{PHASE3_MIN_LENGTH})")
            i += WINDOW_JUMP_NORMAL
            continue
        
        phase3_end = phase3_start + first_threshold_idx
        phase3_length = phase3_end - phase3_start
        phase3_data = all_gx_filtered[phase3_start:phase3_end]
        
        print(f"✅ Phase 3 PASSED")
        print(f"   Range: [{phase3_start}:{phase3_end}], Length: {phase3_length}")
        print(f"   Min: {phase3_data.min():.2f}, Max: {phase3_data.max():.2f}, Mean: {phase3_data.mean():.2f}")
        print(f"   Variance: {np.var(phase3_data):.2f}, Std: {np.std(phase3_data):.2f}")
        print(f"   Value at end: {all_gx_filtered[phase3_end]:.2f}")
        
        # ============= PHASE 4: Search for stable window =============
        phase4_search_start = phase3_end
        phase4_found = False
        phase4_start = None
        phase4_iterations = 0
        
        print(f"🔍 Phase 4: Searching for {PHASE4_WINDOW_SIZE} stable samples in [{PHASE4_MIN_BOUND}, {PHASE4_MAX_BOUND}]...")
        
        for iteration in range(PHASE4_MAX_ITERATIONS):
            cursor = phase4_search_start + iteration
            window_end = cursor + PHASE4_WINDOW_SIZE
            
            if window_end > len(all_gx_original):
                print(f"❌ Phase 4 FAILED - Not enough data (reached end at iteration {iteration})")
                break
            
            window_data = all_gx_original[cursor:window_end]
            
            # Check if all values in window are within bounds
            in_bounds = np.all((window_data >= PHASE4_MIN_BOUND) & (window_data <= PHASE4_MAX_BOUND))
            
            if in_bounds:
                phase4_start = cursor
                phase4_iterations = iteration
                phase4_found = True
                phase4_data = window_data
                print(f"✅ Phase 4 PASSED - Found stable window at iteration {iteration}")
                print(f"   Range: [{phase4_start}:{window_end}], Length: {PHASE4_WINDOW_SIZE}")
                print(f"   Min: {window_data.min():.2f}, Max: {window_data.max():.2f}, Mean: {window_data.mean():.2f}")
                print(f"   Variance: {np.var(window_data):.2f}, Std: {np.std(window_data):.2f}")
                break
            else:
                out_of_bounds_count = np.sum((window_data < PHASE4_MIN_BOUND) | (window_data > PHASE4_MAX_BOUND))
                if iteration % 10 == 0:  # Print every 10th iteration
                    print(f"   Iteration {iteration}: {out_of_bounds_count}/{PHASE4_WINDOW_SIZE} samples out of bounds at position {cursor}")
        
        if not phase4_found:
            print(f"❌ Phase 4 FAILED - No stable window found in {PHASE4_MAX_ITERATIONS} iterations")
            i += WINDOW_JUMP_NORMAL
            continue
        
        # ============= PHASE 5: Dynamic length (ORIGINAL data) =============
        phase5_start = phase4_start
        phase5_found = False
        phase5_end = None
        phase5_length = PHASE5_INITIAL_LENGTH
        
        print(f"🔍 Phase 5: Starting with {PHASE5_INITIAL_LENGTH} samples, growing until variance transition...")
        
        while phase5_length <= PHASE5_MAX_LENGTH:
            current_end = phase5_start + phase5_length
            lookahead_end = current_end + PHASE5_LOOKAHEAD
            
            if lookahead_end > len(all_gx_original):
                print(f"❌ Phase 5 FAILED - Not enough data for lookahead at length {phase5_length}")
                break
            
            # Get current Phase 5 window
            current_window = all_gx_original[phase5_start:current_end]
            
            # Check if all values in current window are within bounds
            out_of_bounds = np.sum((current_window < PHASE5_MIN_BOUND) | (current_window > PHASE5_MAX_BOUND))
            if out_of_bounds > 0:
                if phase5_length % 50 == 0:
                    print(f"   Length {phase5_length}: {out_of_bounds} samples out of bounds")
                phase5_length += 1
                continue
            
            # Calculate variance of current Phase 5 window
            current_variance = np.var(current_window)
            
            # Get next 100 samples (Phase 6 candidate)
            lookahead_window = all_gx_original[current_end:lookahead_end]
            lookahead_variance = np.var(lookahead_window)
            
            # Check if lookahead has significantly lower variance
            if lookahead_variance < current_variance * PHASE5_VARIANCE_RATIO:
                # Found the transition point!
                phase5_end = current_end + 30
            
                phase5_found = True
                phase5_data_orig = current_window
                phase5_variance = current_variance
                phase5_mean = np.mean(current_window)
                phase5_std = np.std(current_window)
                
                print(f"✅ Phase 5 PASSED - Found variance transition at length {phase5_length}")
                print(f"   Range: [{phase5_start}:{phase5_end}], Length: {phase5_length}")
                print(f"   Min: {phase5_data_orig.min():.2f}, Max: {phase5_data_orig.max():.2f}, Mean: {phase5_mean:.2f}")
                print(f"   Variance: {phase5_variance:.2f}, Std: {phase5_std:.2f}")
                print(f"   Lookahead variance: {lookahead_variance:.2f}")
                print(f"   Ratio (lookahead/current): {lookahead_variance/current_variance:.3f} < {PHASE5_VARIANCE_RATIO} ✓")
                break
            else:
                # Variance not low enough yet, increase Phase 5 size
                if phase5_length % 50 == 0:
                    print(f"   Length {phase5_length}: current_var={current_variance:.2f}, lookahead_var={lookahead_variance:.2f}, ratio={lookahead_variance/current_variance:.3f} (need < {PHASE5_VARIANCE_RATIO})")
                phase5_length += 1
        
        if not phase5_found:
            print(f"❌ Phase 5 FAILED - No variance transition found up to length {PHASE5_MAX_LENGTH}")
            i += WINDOW_JUMP_NORMAL
            continue
        
        # ============= PHASE 6: The stable continuation (ORIGINAL data) =============
        phase6_start = phase5_end 
        phase6_end = phase6_start + PHASE6_LENGTH
        
        if phase6_end > len(all_gx_original):
            print(f"❌ Phase 6 FAILED - Not enough data")
            i += WINDOW_JUMP_NORMAL
            continue
        
        phase6_data_orig = all_gx_original[phase6_start:phase6_end]
        phase6_variance = np.var(phase6_data_orig)
        phase6_mean = np.mean(phase6_data_orig)
        phase6_std = np.std(phase6_data_orig)
        
        variance_ratio = phase6_variance / phase5_variance if phase5_variance > 0 else 0
        
        print(f"✅ Phase 6 PASSED (stable continuation)")
        print(f"   Range: [{phase6_start}:{phase6_end}], Length: {PHASE6_LENGTH}")
        print(f"   Min: {phase6_data_orig.min():.2f}, Max: {phase6_data_orig.max():.2f}, Mean: {phase6_mean:.2f}")
        print(f"   Variance: {phase6_variance:.2f}, Std: {phase6_std:.2f}")
        print(f"   Ratio (Phase6/Phase5): {variance_ratio:.3f}")
        
        # ============= PATTERN FOUND! =============
        pattern = {
            'window_start': window_start,
            'phase1': (phase1_start, phase1_end),
            'phase1_stats': {
                'length': phase1_length,
                'min': phase1_data.min(),
                'max': phase1_data.max(),
                'mean': phase1_data.mean(),
                'variance': np.var(phase1_data),
                'std': np.std(phase1_data)
            },
            'phase2': (phase2_start, phase2_end),
            'phase3': (phase3_start, phase3_end),
            'phase3_stats': {
                'length': phase3_length,
                'min': phase3_data.min(),
                'max': phase3_data.max(),
                'mean': phase3_data.mean(),
                'variance': np.var(phase3_data),
                'std': np.std(phase3_data)
            },
            'phase4': (phase4_start, phase4_start + PHASE4_WINDOW_SIZE),
            'phase4_iterations': phase4_iterations,
            'phase5': (phase5_start, phase5_end),
            'phase5_var': phase5_variance,
            'phase5_mean': phase5_mean,
            'phase5_length': phase5_length,
            'phase6': (phase6_start, phase6_end),
            'phase6_var': phase6_variance,
            'phase6_mean': phase6_mean,
            'variance_ratio': variance_ratio
        }
        
        detected_patterns.append(pattern)
        
        print(f"\n{'🎯'*40}")
        print(f"🎯 COMPLETE PATTERN DETECTED at index {window_start}!")
        print(f"{'🎯'*40}\n")
        
        # Jump after finding pattern
        i += WINDOW_JUMP_AFTER_PATTERN
    
    # Summary
    print("\n" + "="*80)
    print(f"📊 DETECTION SUMMARY")
    print("="*80)
    print(f"🎯 Total patterns detected: {len(detected_patterns)}")
    print(f"📊 Total windows analyzed: {window_count}")
    
    if len(detected_patterns) == 0:
        print("\n❌ No valid patterns found.")
        print("\n💡 Suggestions for parameter tuning:")
        print(f"   - Adjust PHASE1_MIN_LENGTH (currently {PHASE1_MIN_LENGTH})")
        print(f"   - Adjust PHASE1_MAX_LENGTH (currently {PHASE1_MAX_LENGTH})")
        print(f"   - Adjust PHASE3_THRESHOLD (currently {PHASE3_THRESHOLD})")
        print(f"   - Adjust PHASE3_MIN_LENGTH (currently {PHASE3_MIN_LENGTH})")
        print(f"   - Adjust PHASE3_MAX_LENGTH (currently {PHASE3_MAX_LENGTH})")
        print(f"   - Adjust PHASE4_WINDOW_SIZE (currently {PHASE4_WINDOW_SIZE})")
        print(f"   - Adjust PHASE4_MAX_ITERATIONS (currently {PHASE4_MAX_ITERATIONS})")
        print(f"   - Adjust PHASE5_INITIAL_LENGTH (currently {PHASE5_INITIAL_LENGTH})")
        print(f"   - Adjust PHASE5_MAX_LENGTH (currently {PHASE5_MAX_LENGTH})")
        print(f"   - Adjust PHASE5_VARIANCE_RATIO (currently {PHASE5_VARIANCE_RATIO})")
        print(f"   - Relax PHASE5 bounds (currently [{PHASE5_MIN_BOUND}, {PHASE5_MAX_BOUND}])")
        print(f"   - Try different GAUSSIAN_SIGMA (currently {GAUSSIAN_SIGMA})")
        return
    
    # Plot each detected pattern
    for idx, pattern in enumerate(detected_patterns):
        plot_detected_pattern_v2(all_gx_original, all_gx_filtered, pattern, 
                                 sample_id, pin, idx + 1, digit_info)


def plot_detected_pattern_v2(gx_original, gx_filtered, pattern, sample_id, pin, pattern_num, digit_info):
    """Plot a detected pattern with colored phases (updated with dynamic Phase 5)."""
    
    # Extract pattern boundaries
    p1_start, p1_end = pattern['phase1']
    p2_start, p2_end = pattern['phase2']
    p3_start, p3_end = pattern['phase3']
    p4_start, p4_end = pattern['phase4']
    p5_start, p5_end = pattern['phase5']
    p6_start, p6_end = pattern['phase6']
    
    # Determine plot range (add context)
    plot_start = max(0, p1_start - 100)
    plot_end = min(len(gx_original), p6_end + 100)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    fig.suptitle(f"Pattern #{pattern_num} - PIN {pin} (ID={sample_id}) | Var Ratio (P6/P5): {pattern['variance_ratio']:.3f} | Phase4 iter: {pattern['phase4_iterations']} | Phase5 len: {pattern['phase5_length']}", 
                 fontsize=13, fontweight='bold')
    
    # Plot range
    t = np.arange(plot_start, plot_end)
    
    # Color scheme
    phase_colors = {
        1: '#e74c3c',  # Red - negative phase
        2: '#f39c12',  # Orange - gap
        3: '#2ecc71',  # Green - descent to threshold
        4: '#3498db',  # Blue - stable search window
        5: '#9b59b6',  # Purple - dynamic variance
        6: '#34495e'   # Dark gray - stable
    }
    
    # Top plot: Filtered data
    ax1.plot(t, gx_filtered[plot_start:plot_end], 'k-', linewidth=1, alpha=0.3, label='Filtered (background)')
    
    # Phase 1
    t1 = np.arange(p1_start, p1_end)
    p1_len = pattern['phase1_stats']['length']
    ax1.plot(t1, gx_filtered[p1_start:p1_end], color=phase_colors[1], linewidth=3, 
             label=f'Phase 1: Negative ({p1_len} samples)')
    ax1.axvspan(p1_start, p1_end, alpha=0.15, color=phase_colors[1])
    
    # Phase 2
    t2 = np.arange(p2_start, p2_end)
    ax1.plot(t2, gx_filtered[p2_start:p2_end], color=phase_colors[2], linewidth=3, 
             label=f'Phase 2: Gap ({p2_end-p2_start} samples)')
    ax1.axvspan(p2_start, p2_end, alpha=0.15, color=phase_colors[2])
    
    # Phase 3
    t3 = np.arange(p3_start, p3_end)
    p3_len = pattern['phase3_stats']['length']
    ax1.plot(t3, gx_filtered[p3_start:p3_end], color=phase_colors[3], linewidth=3, 
             label=f'Phase 3: Descent ({p3_len} samples)')
    ax1.axvspan(p3_start, p3_end, alpha=0.15, color=phase_colors[3])
    
    ax1.axhline(0, color='gray', linestyle=':', alpha=0.3, linewidth=1)
    ax1.axhline(5, color=phase_colors[3], linestyle='--', alpha=0.5, linewidth=1.5, label='Threshold = 5')
    
    ax1.set_ylabel('gx (filtered)', fontsize=11, fontweight='bold')
    ax1.set_title('Detection Phases (Filtered Data)', fontsize=12)
    ax1.legend(loc='upper right', fontsize=9, ncol=3)
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Original data
    ax2.plot(t, gx_original[plot_start:plot_end], 'k-', linewidth=1, alpha=0.3, label='Original (background)')
    
    # Phase 4
    t4 = np.arange(p4_start, p4_end)
    ax2.plot(t4, gx_original[p4_start:p4_end], color=phase_colors[4], linewidth=3, 
             label=f'Phase 4: Stable Window ({p4_end-p4_start} samples, iter={pattern["phase4_iterations"]})')
    ax2.axvspan(p4_start, p4_end, alpha=0.15, color=phase_colors[4])
    
    # Phase 5
    t5 = np.arange(p5_start, p5_end)
    p5_len = pattern['phase5_length']
    ax2.plot(t5, gx_original[p5_start:p5_end], color=phase_colors[5], linewidth=3, 
             label=f'Phase 5: Dynamic ({p5_len} samples, σ²={pattern["phase5_var"]:.2f})')
    ax2.axvspan(p5_start, p5_end, alpha=0.15, color=phase_colors[5])
    
    # Phase 6
    t6 = np.arange(p6_start, p6_end)
    ax2.plot(t6, gx_original[p6_start:p6_end], color=phase_colors[6], linewidth=3, 
             label=f'Phase 6: Stable ({p6_end-p6_start} samples, σ²={pattern["phase6_var"]:.2f})')
    ax2.axvspan(p6_start, p6_end, alpha=0.15, color=phase_colors[6])
    
    ax2.axhline(30, color='red', linestyle='--', alpha=0.4, linewidth=1.5, label='±30 bounds')
    ax2.axhline(-30, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
    ax2.axhline(0, color='gray', linestyle=':', alpha=0.3, linewidth=1)
    
    # Digit boundaries
    for dinfo in digit_info:
        if plot_start <= dinfo['start'] <= plot_end:
            ax1.axvline(dinfo['start'], color='gray', alpha=0.25, linestyle=':', linewidth=2)
            ax2.axvline(dinfo['start'], color='gray', alpha=0.25, linestyle=':', linewidth=2)
            y_pos = ax1.get_ylim()[1] * 0.95
            ax1.text(dinfo['start'] + 10, y_pos, f"D{dinfo['digit']+1}:'{dinfo['char']}'", 
                    fontsize=9, alpha=0.6, ha='left', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))
    
    ax2.set_xlabel('Sample index', fontsize=11, fontweight='bold')
    ax2.set_ylabel('gx (original)', fontsize=11, fontweight='bold')
    ax2.set_title('Validation Phases (Original Data)', fontsize=12)
    ax2.legend(loc='upper right', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()




def extract_all_patterns_and_save(samples, output_format='parquet'):
    """Process all samples through pattern detection (option 8) and save Phase 5 data.
    
    Extracts Phase 5 data for each detected pattern and saves in the same format
    as the original data, but with sensor_values containing only the Phase 5 sequence.
    """
    
    # ============= CONFIGURABLE PARAMETERS (same as option 8) =============
    PHASE1_MIN_LENGTH = 120
    PHASE1_MAX_LENGTH = 200
    PHASE2_GAP = 20
    PHASE3_THRESHOLD = 5
    PHASE3_MIN_LENGTH = 80
    PHASE3_MAX_LENGTH = 200
    PHASE4_WINDOW_SIZE = 30
    PHASE4_MAX_ITERATIONS = 150
    PHASE4_MIN_BOUND = -15
    PHASE4_MAX_BOUND = 15
    PHASE5_INITIAL_LENGTH = 50
    PHASE5_MAX_LENGTH = 500
    PHASE5_LOOKAHEAD = 100
    PHASE5_VARIANCE_RATIO = 0.1
    PHASE5_MIN_BOUND = -30
    PHASE5_MAX_BOUND = 30
    PHASE6_LENGTH = 100
    GAUSSIAN_SIGMA = 20
    WINDOW_JUMP_NORMAL = 20
    WINDOW_JUMP_AFTER_PATTERN = 100
    # ====================================================================
    
    print("\n" + "="*80)
    print("🚀 BATCH PROCESSING: Extracting Phase 5 patterns from all samples")
    print("="*80)
    print(f"Total samples to process: {len(samples)}")
    print(f"Output format: {output_format}")
    print("="*80 + "\n")
    
    all_extracted_patterns = []
    pattern_id = 0
    
    for sample_idx, sample in enumerate(samples):
        sample_id = sample["id"]
        pin = sample["pin_label"]
        sensor_values = sample["sensor_values"]
        
        print(f"\n{'='*80}")
        print(f"Processing sample {sample_idx + 1}/{len(samples)}: ID={sample_id}, PIN={pin}")
        print(f"{'='*80}")
        
        # Extract all gx values (concatenate, remove boundaries)
        all_gx_original = []
        for i, win in enumerate(sensor_values[:4]):
            _, _, _, gxv, _ = extract_axes_values(win)
            all_gx_original.extend(gxv)
        
        all_gx_original = np.array(all_gx_original)
        
        if len(all_gx_original) == 0:
            print(f"⚠️ No gx data found for sample {sample_id}, skipping...")
            continue
        
        # Apply Gaussian filter
        all_gx_filtered = gaussian_filter1d(all_gx_original, sigma=GAUSSIAN_SIGMA)
        
        # Pattern detection (same logic as option 8)
        i = 0
        patterns_found = 0
        
        while i < len(all_gx_filtered):
            window_start = i
            
            # Check minimum required data
            min_required = PHASE1_MAX_LENGTH + PHASE2_GAP + PHASE3_MAX_LENGTH + PHASE4_MAX_ITERATIONS + PHASE4_WINDOW_SIZE + PHASE5_INITIAL_LENGTH + PHASE5_LOOKAHEAD + PHASE6_LENGTH
            if i + min_required > len(all_gx_filtered):
                break
            
            # Phase 1
            phase1_start = window_start
            if all_gx_filtered[phase1_start] >= 0:
                i += WINDOW_JUMP_NORMAL
                continue
            
            phase1_search_end = phase1_start + PHASE1_MAX_LENGTH
            if phase1_search_end > len(all_gx_filtered):
                i += WINDOW_JUMP_NORMAL
                continue
            
            phase1_search_data = all_gx_filtered[phase1_start:phase1_search_end]
            positive_indices = np.where(phase1_search_data > 0)[0]
            
            if len(positive_indices) == 0 or positive_indices[0] < PHASE1_MIN_LENGTH:
                i += WINDOW_JUMP_NORMAL
                continue
            
            phase1_end = phase1_start + positive_indices[0]
            
            # Phase 2
            phase2_start = phase1_end
            phase2_end = phase2_start + PHASE2_GAP
            
            # Phase 3
            phase3_start = phase2_end
            phase3_search_end = phase3_start + PHASE3_MAX_LENGTH
            
            if phase3_search_end > len(all_gx_filtered):
                i += WINDOW_JUMP_NORMAL
                continue
            
            phase3_search_data = all_gx_filtered[phase3_start:phase3_search_end]
            under_threshold_indices = np.where(phase3_search_data <= PHASE3_THRESHOLD)[0]
            
            if len(under_threshold_indices) == 0 or under_threshold_indices[0] < PHASE3_MIN_LENGTH:
                i += WINDOW_JUMP_NORMAL
                continue
            
            phase3_end = phase3_start + under_threshold_indices[0]
            
            # Phase 4
            phase4_search_start = phase3_end
            phase4_found = False
            phase4_start = None
            
            for iteration in range(PHASE4_MAX_ITERATIONS):
                cursor = phase4_search_start + iteration
                window_end = cursor + PHASE4_WINDOW_SIZE
                
                if window_end > len(all_gx_original):
                    break
                
                window_data = all_gx_original[cursor:window_end]
                in_bounds = np.all((window_data >= PHASE4_MIN_BOUND) & (window_data <= PHASE4_MAX_BOUND))
                
                if in_bounds:
                    phase4_start = cursor
                    phase4_found = True
                    break
            
            if not phase4_found:
                i += WINDOW_JUMP_NORMAL
                continue
            
            # Phase 5 (dynamic)
            phase5_start = phase4_start
            phase5_found = False
            phase5_end = None
            phase5_length = PHASE5_INITIAL_LENGTH
            
            while phase5_length <= PHASE5_MAX_LENGTH:
                current_end = phase5_start + phase5_length
                lookahead_end = current_end + PHASE5_LOOKAHEAD
                
                if lookahead_end > len(all_gx_original):
                    break
                
                current_window = all_gx_original[phase5_start:current_end]
                out_of_bounds = np.sum((current_window < PHASE5_MIN_BOUND) | (current_window > PHASE5_MAX_BOUND))
                
                if out_of_bounds > 0:
                    phase5_length += 1
                    continue
                
                current_variance = np.var(current_window)
                lookahead_window = all_gx_original[current_end:lookahead_end]
                lookahead_variance = np.var(lookahead_window)
                
                if lookahead_variance < current_variance * PHASE5_VARIANCE_RATIO:
                    phase5_end = current_end + 25
                    phase5_found = True
                    break
                
                phase5_length += 1
            
            if not phase5_found:
                i += WINDOW_JUMP_NORMAL
                continue
            
            # Phase 6 check
            phase6_start = phase5_end
            phase6_end = phase6_start + PHASE6_LENGTH
            
            if phase6_end > len(all_gx_original):
                i += WINDOW_JUMP_NORMAL
                continue
            
            # Pattern found! Extract Phase 5 data
            phase5_gx_data = all_gx_original[phase5_start:phase5_end].tolist()
            
            # Reconstruct full sensor data for Phase 5 from original sample
            # We need to find which original samples correspond to Phase 5 indices
            phase5_sensor_values = []
            
            # Map back to original sensor structure
            current_idx = 0
            for digit_idx, win in enumerate(sensor_values[:4]):
                for sample_point in win:
                    if phase5_start <= current_idx < phase5_end:
                        # Extract the full sensor reading
                        if isinstance(sample_point, dict):
                            phase5_sensor_values.append([
                                sample_point['ax'],
                                sample_point['ay'],
                                sample_point['az'],
                                sample_point['gx'],
                                sample_point['gz']
                            ])
                        else:
                            phase5_sensor_values.append(list(sample_point))
                    current_idx += 1
            
            # Create extracted pattern entry
            extracted_pattern = {
                "id": pattern_id,
                "original_sample_id": sample_id,
                "pin_label": pin,
                "sensor_values": phase5_sensor_values,
                "phase5_start_idx": int(phase5_start),
                "phase5_end_idx": int(phase5_end),
                "phase5_length": int(phase5_length)
            }
            
            all_extracted_patterns.append(extracted_pattern)
            pattern_id += 1
            patterns_found += 1
            
            print(f"  ✅ Pattern {patterns_found} found: Phase 5 range [{phase5_start}:{phase5_end}], length={phase5_length}")
            
            i += WINDOW_JUMP_AFTER_PATTERN
        
        if patterns_found == 0:
            print(f"  ⚠️ No patterns found in sample {sample_id}")
        else:
            print(f"  📊 Total patterns found in sample {sample_id}: {patterns_found}")
    
    # Summary
    print("\n" + "="*80)
    print("📊 EXTRACTION SUMMARY")
    print("="*80)
    print(f"Total samples processed: {len(samples)}")
    print(f"Total patterns extracted: {len(all_extracted_patterns)}")
    print("="*80 + "\n")
    
    if len(all_extracted_patterns) == 0:
        print("❌ No patterns extracted. Nothing to save.")
        return
    
    # Save to file
    output_path = DATA_PATH.parent / f"sequences_phase5.{output_format}"
    
    print(f"💾 Saving extracted patterns to: {output_path}")
    
    if output_format == 'jsonl':
        with open(output_path, 'w', encoding='utf-8') as f:
            for pattern in all_extracted_patterns:
                f.write(json.dumps(pattern) + '\n')
        print(f"✅ Saved {len(all_extracted_patterns)} patterns to JSONL format")
    
    elif output_format == 'parquet':
        import pyarrow as pa
        table = pa.Table.from_pylist(all_extracted_patterns)
        pq.write_table(table, output_path)
        print(f"✅ Saved {len(all_extracted_patterns)} patterns to Parquet format")
    
    else:
        print(f"❌ Unsupported format: {output_format}")
        return
    
    # Show some statistics
    print("\n" + "="*80)
    print("📈 EXTRACTED DATA STATISTICS")
    print("="*80)
    
    lengths = [p['phase5_length'] for p in all_extracted_patterns]
    print(f"Phase 5 lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")
    
    # Count patterns per PIN
    pin_counts = {}
    for p in all_extracted_patterns:
        pin = p['pin_label']
        pin_counts[pin] = pin_counts.get(pin, 0) + 1
    
    print(f"\nPatterns per PIN:")
    for pin, count in sorted(pin_counts.items()):
        print(f"  PIN {pin}: {count} patterns")
    
    print("\n✨ Extraction complete!")
    
    return all_extracted_patterns


# ------------------- Main (update) -------------------
if __name__ == "__main__":
    samples = load_dataset(DATA_PATH)
    summarize_dataset(samples)

    print("Available options:")
    print("  [1] Visualize a single sequence")
    print("  [2] Compare multiple sequences (by IDs)")
    print("  [3] Compare multiple occurrences of the same PIN")
    print("  [4] Compare same PIN by digit transitions (4×2 plots)")
    print("  [5] Compare all occurrences of one PIN vs another PIN (by digit)")
    print("  [6] Normalize all sequences to same length")
    print("  [7] Visualize gx with Gaussian filters for a specific sample ID")
    print("  [8] Detect PIN entry patterns using sliding window analysis")
    print("  [9] Batch extract Phase 5 patterns from all samples and save")
    choice = input("Select an option (1–9): ").strip()

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

    elif choice == "6":
        normalized_samples = normalize_sequences(samples)
        print("\n✨ Normalization complete!")
        print("You can now use the normalized data for further analysis.")

    elif choice == "7":
        ids = [s["id"] for s in samples]
        print("Available IDs:", ids)
        while True:
            try:
                sample_id = int(input("Enter sample ID to visualize with Gaussian filters: "))
                visualize_gx_with_gaussian_filters(samples, sample_id)
                break
            except (ValueError, KeyError):
                print("Invalid ID, please try again.")

    elif choice == "8":
        ids = [s["id"] for s in samples]
        print("Available IDs:", ids)
        while True:
            try:
                sample_id = int(input("Enter sample ID to detect PIN patterns: "))
                detect_pin_patterns(samples, sample_id)
                break
            except (ValueError, KeyError):
                print("Invalid ID, please try again.")

    elif choice == "9":
        format_choice = input("Output format (jsonl/parquet) [default: parquet]: ").strip().lower()
        if format_choice not in ['jsonl', 'parquet', '']:
            print("Invalid format, using parquet")
            format_choice = 'parquet'
        if format_choice == '':
            format_choice = 'parquet'
        
        extracted_patterns = extract_all_patterns_and_save(samples, output_format=format_choice)

    else:
        print("Invalid choice.")
