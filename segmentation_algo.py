import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import sys

# --- CONFIGURATION ---
FILENAME = '/home/ghali/Documents/Kaist_Semester/Intro_to_IoT/Iot-Projet/SmartWatch_Pin_Inference/data/sequences_pins/sequences_phase5.jsonl'
WINDOW_BEFORE = 25 
WINDOW_AFTER = 35
WINDOW_TOTAL = WINDOW_BEFORE + WINDOW_AFTER
SMOOTHING_SIZE = 10 

def load_data(filename):
    data = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        sys.exit()
    return data

def smooth_signal(signal, window_size):
    """Applies a moving average to reduce noise."""
    window = np.ones(window_size) / window_size
    return np.convolve(signal, window, mode='same')

def robust_segmentation(raw_data):
    """
    Adaptive algorithm to find exactly 4 taps.
    """
    # 1. Dimension check
    if raw_data.shape[0] < WINDOW_TOTAL:
        return None, np.zeros(raw_data.shape[0]), [], "TOO SHORT"

    # 2. Energy calculation (Jerk)
    # Acc (columns 0,1,2) and Gyro (columns 3,4)
    # Acc is multiplied by 100 to balance with Gyro magnitude
    acc_diff = np.diff(raw_data[:, :3], axis=0) * 100
    gyro_diff = np.diff(raw_data[:, 3:], axis=0)
    
    # Raw energy
    energy = np.linalg.norm(acc_diff, axis=1) + np.linalg.norm(gyro_diff, axis=1)
    
    # Padding to match original size
    energy = np.pad(energy, (0, 1), 'constant')
    
    # 3. SMOOTHING (Crucial to avoid false positives)
    energy_smooth = smooth_signal(energy, SMOOTHING_SIZE)

    # 4. Adaptive Search
    # Start with high prominence requirements, lower if not enough peaks found
    found_peaks = []
    current_prominence = 10.0 # Starting threshold (aggressive)
    min_prominence = 0.5      # Minimum threshold (sensitive)
    decay_rate = 0.5          # Step size

    while len(found_peaks) < 4 and current_prominence >= min_prominence:
        found_peaks, properties = find_peaks(
            energy_smooth, 
            distance=30,            # Min distance between taps
            prominence=current_prominence
        )
        if len(found_peaks) < 4:
            current_prominence -= decay_rate 

    # 5. Selection of best peaks
    if len(found_peaks) < 4:
        return None, energy_smooth, found_peaks, "FAILURE: <4 taps"
    
    # If too many peaks, keep the 4 with the highest prominence
    if len(found_peaks) > 4:
        prominences = properties['prominences']
        top_indices = np.argsort(prominences)[-4:]
        found_peaks = np.sort(found_peaks[top_indices])

    # 6. Slicing
    segments = []
    for peak in found_peaks:
        template = np.zeros((WINDOW_TOTAL, 5))
        
        start = peak - WINDOW_BEFORE
        end = peak + WINDOW_AFTER
        
        idx_start_src = max(0, start)
        idx_end_src = min(len(raw_data), end)
        
        idx_start_dst = max(0, -start)
        idx_end_dst = idx_start_dst + (idx_end_src - idx_start_src)
        
        if idx_end_dst <= WINDOW_TOTAL:
            template[idx_start_dst:idx_end_dst] = raw_data[idx_start_src:idx_end_src]
            segments.append(template)
        else:
            segments.append(template)

    return np.array(segments), energy_smooth, found_peaks, "SUCCESS"

def main():
    records = load_data(FILENAME)
    available_pins = sorted(list(set([r['pin_label'] for r in records])))
    print(f"Available PINs: {available_pins}")
    print(f"Total sequences loaded: {len(records)}")
    
    target_pin = input(">> PIN to analyze: ").strip()
    target_records = [r for r in records if r['pin_label'] == target_pin]

    if not target_records:
        print("No records found for this PIN.")
        return

    # Plot configuration
    num_plots = min(len(target_records), 5)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 3 * num_plots), sharex=False)
    if num_plots == 1: axes = [axes]

    valid_cnt = 0
    X_dataset = []

    for i, record in enumerate(target_records):
        # Clean loading
        raw_list = record['sensor_values']
        raw_data = np.array(raw_list)

        # Safety check for dimensions (N, 5)
        if raw_data.ndim != 2 or raw_data.shape[1] != 5:
            print(f"Ignored ID {record['id']} : Invalid format {raw_data.shape}")
            continue

        segments, energy, peaks, status = robust_segmentation(raw_data)

        if segments is not None:
            valid_cnt += 1
            X_dataset.extend(segments)
            color = 'green'
        else:
            color = 'red'

        # Plotting
        if i < num_plots:
            ax = axes[i]
            ax.plot(energy, color='black', alpha=0.6, label='Smoothed Energy')
            
            # Mark found peaks
            ax.plot(peaks, energy[peaks], "o", color=color, markersize=8)
            
            ax.set_title(f"ID {record['id']} (Len: {len(raw_data)}) [{status}]", color=color, fontweight='bold')
            
            # Draw captured windows
            if segments is not None:
                for p in peaks:
                    ax.axvspan(p-WINDOW_BEFORE, p+WINDOW_AFTER, color='limegreen', alpha=0.2)

    plt.tight_layout()
    plt.show()

    print(f"\nSummary for PIN {target_pin}:")
    print(f"   Valid sequences : {valid_cnt} / {len(target_records)}")
    
    if len(X_dataset) > 0:
        final_data = np.array(X_dataset)
        print(f"   Extracted dataset shape : {final_data.shape}")

if __name__ == "__main__":
    main()