import json
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.signal import find_peaks
import sys

# --- CONFIGURATION ---
FILENAME = '/home/ghali/Documents/Kaist_Semester/Intro_to_IoT/Iot-Projet/SmartWatch_Pin_Inference/data/sequences_pins/clean_data.json'
MODEL_FILE = 'pin_model.pkl'

# Window parameters
WINDOW_BEFORE = 25
WINDOW_AFTER = 35
WINDOW_TOTAL = WINDOW_BEFORE + WINDOW_AFTER
SMOOTHING_SIZE = 10

def smooth_signal(signal, window_size):
    """Simple moving average."""
    window = np.ones(window_size) / window_size
    return np.convolve(signal, window, mode='same')

def robust_segmentation(raw_data):
    """
    Extracts the 4 segments corresponding to the 4 taps.
    Returns None if the algorithm fails.
    """
    # 0. Size safety check
    if raw_data.shape[0] < WINDOW_TOTAL:
        return None

    # 1. Energy Calculation (Jerk)
    # Acc (0-2) * 100 + Gyro (3-4)
    acc_diff = np.diff(raw_data[:, :3], axis=0) * 100
    gyro_diff = np.diff(raw_data[:, 3:], axis=0)
    
    # Norm + Padding to maintain size
    energy = np.linalg.norm(acc_diff, axis=1) + np.linalg.norm(gyro_diff, axis=1)
    energy = np.pad(energy, (0, 1), 'constant')
    
    # 2. Smoothing
    energy_smooth = smooth_signal(energy, SMOOTHING_SIZE)

    # 3. Adaptive Peak Search
    found_peaks = []
    current_prominence = 10.0
    decay_rate = 0.5
    min_prominence = 0.5

    while len(found_peaks) < 4 and current_prominence >= min_prominence:
        found_peaks, properties = find_peaks(
            energy_smooth, 
            distance=30, 
            prominence=current_prominence
        )
        if len(found_peaks) < 4:
            current_prominence -= decay_rate

    # 4. Filtering and Selection
    if len(found_peaks) < 4:
        return None # Failure
    
    # If more than 4 peaks, keep the top 4 based on prominence
    if len(found_peaks) > 4:
        top_indices = np.argsort(properties['prominences'])[-4:]
        found_peaks = np.sort(found_peaks[top_indices])

    # 5. Window Extraction
    segments = []
    for peak in found_peaks:
        template = np.zeros((WINDOW_TOTAL, 5))
        
        # Relative indices
        start = peak - WINDOW_BEFORE
        end = peak + WINDOW_AFTER
        
        # Safe bounds for source
        idx_start_src = max(0, start)
        idx_end_src = min(len(raw_data), end)
        
        # Safe bounds for destination (template)
        idx_start_dst = max(0, -start)
        idx_end_dst = idx_start_dst + (idx_end_src - idx_start_src)
        
        # Copy data
        if idx_end_dst <= WINDOW_TOTAL:
            template[idx_start_dst:idx_end_dst] = raw_data[idx_start_src:idx_end_src]
            segments.append(template)
        else:
            segments.append(template)

    return np.array(segments)

# --- FEATURE EXTRACTION (Machine Learning) ---
def extract_features(segment):
    """
    Transforms a window (60, 5) into a vector of 25 features.
    Stats: Mean, Std, Max, Min, Amplitude for each axis.
    """
    features = []
    # For each axis (Ax, Ay, Az, Gx, Gy)
    for axis in range(5):
        data = segment[:, axis]
        features.append(np.mean(data))
        features.append(np.std(data))
        features.append(np.max(data))
        features.append(np.min(data))
        features.append(np.max(data) - np.min(data)) # Amplitude
    return np.array(features)

# --- MAIN ---
def main():
    print("Building Dataset...")
    X = []
    y = []
    
    total_seq = 0
    valid_seq = 0

    try:
        with open(FILENAME, 'r') as f:
            for line in f:
                if not line.strip(): continue
                record = json.loads(line)
                total_seq += 1
                
                # --- NEW SIMPLE LOADING ---
                raw_list = record['sensor_values']
                raw_data = np.array(raw_list)

                # Dimension check (N rows, 5 columns)
                if raw_data.ndim != 2 or raw_data.shape[1] != 5:
                    continue # Skip corrupted data
                # --------------------------

                # Segmentation
                segments = robust_segmentation(raw_data)
                
                if segments is not None:
                    pin_str = record['pin_label'] # e.g., "1234"
                    
                    # Associate each segment with its digit
                    # Segment 0 -> 1st digit of PIN, etc.
                    for i in range(4):
                        digit_char = pin_str[i]
                        
                        # Extract features for this specific tap
                        feats = extract_features(segments[i])
                        
                        X.append(feats)
                        y.append(digit_char)
                    
                    valid_seq += 1
                    
    except FileNotFoundError:
        print(f"Error: File '{FILENAME}' not found.")
        sys.exit()

    print(f"Successfully processed sequences: {valid_seq}/{total_seq}")
    print(f"Total samples (individual taps) for ML: {len(X)}")

    if len(X) == 0:
        print("No valid samples found. Check your data.")
        sys.exit()

    # Convert to numpy
    X = np.array(X)
    y = np.array(y)

    # Train / Test Split (80% / 20%)
    print("\nTrain/Test Split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"   Train set: {X_train.shape}")
    
    print(f"   Test set:  {X_test.shape}")

    print("\nTraining model (Random Forest)...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluation
    print("\nResults on Test Set:")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    for i, j in zip(y_test, y_pred):
        print(f"True: {i}, Predicted: {j}")
    print(f"Global Accuracy: {acc * 100:.2f}%")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred))

    # Save
    joblib.dump(clf, MODEL_FILE)
    print(f"\nModel saved to '{MODEL_FILE}'")

if __name__ == "__main__":
    main()