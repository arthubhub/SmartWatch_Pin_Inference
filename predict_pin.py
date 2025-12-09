import json
import numpy as np
import joblib
from scipy.signal import find_peaks
import sys

# --- CONFIGURATION ---
MODEL_FILE = 'pin_model.pkl'
FILENAME = '/home/ghali/Documents/Kaist_Semester/Intro_to_IoT/Iot-Projet/SmartWatch_Pin_Inference/data/sequences_pins/clean_data.json'

# Segmentation parameters (MUST BE IDENTICAL TO TRAIN SCRIPT)
WINDOW_BEFORE = 25
WINDOW_AFTER = 35
WINDOW_TOTAL = 60
SMOOTHING_SIZE = 10

def smooth_signal(signal, window_size):
    window = np.ones(window_size) / window_size
    return np.convolve(signal, window, mode='same')

def robust_segmentation(raw_data):
    # Safety check for size
    if raw_data.shape[0] < WINDOW_TOTAL:
        return None

    # Exact copy of the function used for training
    acc_diff = np.diff(raw_data[:, :3], axis=0) * 100
    gyro_diff = np.diff(raw_data[:, 3:], axis=0)
    
    energy = np.pad(np.linalg.norm(acc_diff, axis=1) + np.linalg.norm(gyro_diff, axis=1), (0, 1), 'constant')
    energy_smooth = smooth_signal(energy, SMOOTHING_SIZE)

    found_peaks = []
    current_prominence = 10.0
    
    while len(found_peaks) < 4 and current_prominence >= 0.5:
        found_peaks, properties = find_peaks(energy_smooth, distance=30, prominence=current_prominence)
        if len(found_peaks) < 4: current_prominence -= 0.5

    if len(found_peaks) < 4: return None
    
    if len(found_peaks) > 4:
        top_indices = np.argsort(properties['prominences'])[-4:]
        found_peaks = np.sort(found_peaks[top_indices])

    segments = []
    for peak in found_peaks:
        template = np.zeros((WINDOW_TOTAL, 5))
        
        idx_start_src = max(0, peak - WINDOW_BEFORE)
        idx_end_src = min(len(raw_data), peak + WINDOW_AFTER)
        
        idx_start_dst = max(0, -(peak - WINDOW_BEFORE))
        idx_end_dst = idx_start_dst + (idx_end_src - idx_start_src)
        
        if idx_end_dst <= WINDOW_TOTAL:
            template[idx_start_dst:idx_end_dst] = raw_data[idx_start_src:idx_end_src]
            segments.append(template)
        else:
            segments.append(template)
            
    return np.array(segments)

def extract_features(segment):
    # Exact copy of the training function
    features = []
    for axis in range(5):
        data = segment[:, axis]
        features.append(np.mean(data))
        features.append(np.std(data))
        features.append(np.max(data))
        features.append(np.min(data))
        features.append(np.max(data) - np.min(data))
    return np.array(features)

def main():
    # Load Model
    try:
        clf = joblib.load(MODEL_FILE)
        print(f"Model '{MODEL_FILE}' loaded.")
    except FileNotFoundError:
        print("Model not found. Run train_model.py first.")
        sys.exit()

    # Load Data
    records = []
    try:
        with open(FILENAME, 'r') as f:
            for line in f:
                if line.strip(): records.append(json.loads(line))
    except FileNotFoundError:
        print(f"File '{FILENAME}' not found.")
        sys.exit()
    
    print(f"Loaded {len(records)} sequences.")
    target_pin = input(">> Enter a PIN to test prediction (e.g., 5151): ").strip()
    
    matches = [r for r in records if r['pin_label'] == target_pin]
    if not matches:
        print("PIN not found in the file.")
        return

    print(f"\n--- Testing recorded sequences for {target_pin} ---")
    
    correct_pins = 0
    total_attempts = 0

    for record in matches:
        # Data Cleaning (Simplified for new JSON format)
        raw_list = record['sensor_values']
        raw_data = np.array(raw_list)

        # Check format
        if raw_data.ndim != 2 or raw_data.shape[1] != 5:
            continue

        # Segmentation
        segments = robust_segmentation(raw_data)
        
        if segments is not None:
            # Predict the 4 digits
            predicted_digits = []
            for i in range(4):
                feat = extract_features(segments[i])
                pred = clf.predict([feat])[0] # Predict single digit
                predicted_digits.append(pred)
            
            predicted_pin = "".join(predicted_digits)
            total_attempts += 1
            
            # Display result
            if predicted_pin == target_pin:
                print(f"ID {record['id']}: [CORRECT] {predicted_pin}")
                correct_pins += 1
            else:
                print(f"ID {record['id']}: [WRONG]   {predicted_pin} (Expected: {target_pin})")
        else:
            print(f"ID {record['id']}: [SKIP]    Segmentation failed")

    if total_attempts > 0:
        print(f"\nFull PIN Accuracy: {correct_pins}/{total_attempts} ({correct_pins/total_attempts*100:.1f}%)")

if __name__ == "__main__":
    main()