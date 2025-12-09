import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import sys

# --- CONFIGURATION ---
FILENAME = '/home/ghali/Documents/Kaist_Semester/Intro_to_IoT/Iot-Projet/SmartWatch_Pin_Inference/data/sequences_pins/sequences.jsonl'
WINDOW_BEFORE = 25  # Un peu plus large pour être sûr
WINDOW_AFTER = 35
WINDOW_TOTAL = WINDOW_BEFORE + WINDOW_AFTER
SMOOTHING_SIZE = 10  # Taille de la fenêtre de lissage (moyenne mobile)

def load_data(filename):
    data = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except FileNotFoundError:
        print(f"❌ Fichier '{filename}' introuvable.")
        sys.exit()
    return data

def smooth_signal(signal, window_size):
    """Applique une moyenne mobile pour réduire le bruit"""
    window = np.ones(window_size) / window_size
    return np.convolve(signal, window, mode='same')

def robust_segmentation(raw_data):
    """
    Algorithme adaptatif pour trouver exactement 4 tapes.
    """
    # 1. Nettoyage et calcul de l'énergie (Jerk)
    # Acc (0-2) * 100 + Gyro (3-4)
    acc_diff = np.diff(raw_data[:, :3], axis=0) * 100
    gyro_diff = np.diff(raw_data[:, 3:], axis=0)
    
    # Énergie brute
    energy = np.linalg.norm(acc_diff, axis=1) + np.linalg.norm(gyro_diff, axis=1)
    
    # Padding pour garder la taille
    energy = np.pad(energy, (0, 1), 'constant')
    
    # 2. LISSAGE (Crucial pour éviter les faux positifs)
    energy_smooth = smooth_signal(energy, SMOOTHING_SIZE)

    # 3. Recherche Adaptative
    # On commence avec une exigence haute (prominence), et on baisse si on ne trouve pas assez de pics
    found_peaks = []
    current_prominence = 10.0 # Seuil de départ (agressif)
    min_prominence = 0.5      # Seuil plancher (très sensible)
    decay_rate = 0.5          # Pas de descente

    while len(found_peaks) < 4 and current_prominence >= min_prominence:
        found_peaks, properties = find_peaks(
            energy_smooth, 
            distance=30,            # 150ms min entre 2 tapes
            prominence=current_prominence
        )
        if len(found_peaks) < 4:
            current_prominence -= decay_rate # On relâche la contrainte

    # 4. Sélection des meilleurs pics
    if len(found_peaks) < 4:
        return None, energy_smooth, found_peaks, "ECHEC: Pas assez de tapes (<4)"
    
    # Si trop de pics, on garde les 4 qui ont la plus grande "prominence" (saillie)
    if len(found_peaks) > 4:
        # On recupère les prominences calculées par scipy
        prominences = properties['prominences']
        # On trie pour garder les 4 plus gros
        top_indices = np.argsort(prominences)[-4:]
        found_peaks = np.sort(found_peaks[top_indices])

    # 5. Découpage (Slicing Robuste)
    segments = []
    for peak in found_peaks:
        # Création d'un moule vide de taille fixe
        template = np.zeros((WINDOW_TOTAL, 5))
        
        start = peak - WINDOW_BEFORE
        end = peak + WINDOW_AFTER
        
        # Calcul des indices de copie sûrs
        idx_start_src = max(0, start)
        idx_end_src = min(len(raw_data), end)
        
        idx_start_dst = max(0, -start)
        idx_end_dst = idx_start_dst + (idx_end_src - idx_start_src)
        
        # Copie
        if idx_end_dst <= WINDOW_TOTAL:
            template[idx_start_dst:idx_end_dst] = raw_data[idx_start_src:idx_end_src]
            segments.append(template)
        else:
            segments.append(template) # Cas rare (bordure extrême)

    return np.array(segments), energy_smooth, found_peaks, "SUCCES"

def main():
    records = load_data(FILENAME)
    available_pins = sorted(list(set([r['pin_label'] for r in records])))
    print(f"PINs dispos: {available_pins}")
    
    target_pin = input(">> PIN à analyser : ").strip()
    target_records = [r for r in records if r['pin_label'] == target_pin]

    if not target_records: return

    # Configuration de l'affichage
    num_plots = min(len(target_records), 5)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 3 * num_plots), sharex=False)
    if num_plots == 1: axes = [axes]

    valid_cnt = 0
    X_dataset = []

    for i, record in enumerate(target_records):
        # --- CHARGEMENT ROBUSTE (Fix jagged array) ---
        raw_list = record['sensor_values']
        try:
            # Tente de convertir direct
            raw_data = np.array(raw_list)
            if raw_data.ndim == 3: raw_data = np.vstack(raw_data)
        except ValueError:
            # Si échec, on empile manuellement
            chunks = [np.array(c) for c in raw_list if len(c) > 0]
            if len(chunks) > 0:
                raw_data = np.vstack(chunks)
            else:
                continue # Donnée vide
        # ---------------------------------------------

        segments, energy, peaks, status = robust_segmentation(raw_data)

        if segments is not None:
            valid_cnt += 1
            X_dataset.extend(segments)
            color = 'green'
        else:
            color = 'red'

        # Affichage
        if i < num_plots:
            ax = axes[i]
            ax.plot(energy, color='black', alpha=0.6, label='Energie Lissée')
            # Marquer les pics trouvés
            ax.plot(peaks, energy[peaks], "o", color=color, markersize=8)
            ax.set_title(f"ID {record['id']} [{status}]", color=color, fontweight='bold')
            
            # Dessiner les zones capturées
            if segments is not None:
                for p in peaks:
                    ax.axvspan(p-WINDOW_BEFORE, p+WINDOW_AFTER, color='limegreen', alpha=0.2)

    plt.tight_layout()
    plt.show()

    print(f"\n📊 Bilan pour le PIN {target_pin}:")
    print(f"   Séquences valides : {valid_cnt} / {len(target_records)}")
    
    if len(X_dataset) > 0:
        final_data = np.array(X_dataset)
        print(f"   Dataset extrait : {final_data.shape}")
        # np.save(f"X_{target_pin}.npy", final_data)
        # print(f"   Sauvegardé dans X_{target_pin}.npy")

if __name__ == "__main__":
    main()