import cv2
import pandas as pd
import os
import shutil
from pathlib import Path

# --- CONFIGURATION ---
VIDEO_PATH = 'dataset1.avi'
CSV_PATH = 'dataset1.csv'
OUTPUT_DIR = 'shrimp_dataset'
TRAIN_SPLIT = 0.8  # 80% pour l'entraînement
scale_factor = 0.5  # Facteur de mise à l'échelle pour les boîtes englobantes

def create_folders():
    for split in ['train', 'val']:
        os.makedirs(f"{OUTPUT_DIR}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/labels/{split}", exist_ok=True)

def generate_yolo_data():
    create_folders()
    
    # Lecture du CSV
    df = pd.read_csv(CSV_PATH)
    
    # Ouverture de la vidéo
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Vidéo : {width}x{height}, {total_frames} frames.")

    # On groupe par Timestep (chaque timestep correspond à une frame)
    grouped = df.groupby('Timestep')
    
    frame_count = 0
    for timestep, group in grouped:
        # Lire la frame correspondante
        cap.set(cv2.CAP_PROP_POS_FRAMES, timestep - 1) # Timestep commence souvent à 1
        ret, frame = cap.read()
        if not ret:
            continue

        # Déterminer si c'est pour Train ou Val
        split = 'train' if (timestep / total_frames) <= TRAIN_SPLIT else 'val'
        
        # Nom du fichier
        file_name = f"shrimp_frame_{int(timestep):05d}"
        
        # Sauvegarder l'image
        img_path = f"{OUTPUT_DIR}/images/{split}/{file_name}.jpg"
        cv2.imwrite(img_path, frame)
        
        # Créer le fichier label .txt
        label_path = f"{OUTPUT_DIR}/labels/{split}/{file_name}.txt"
        with open(label_path, 'w') as f:
            for _, row in group.iterrows():
                # On utilise CX, CY et on estime W, H via LAMBDA1/2 ou AREA
                # YOLO format: class x_center y_center width height (normalized 0-1)
                
                # Note: On approxime la taille par LAMBDA ou AREA
                # Ici j'utilise une estimation basée sur LAMBDA pour la longueur/largeur
                w_box = (row['LAMBDA1'] * scale_factor) / width
                h_box = (row['LAMBDA2'] * scale_factor) / height
                x_center = row['CX'] / width
                y_center = row['CY'] / height
                
                # Sécurité : clamp entre 0 et 1
                x_center, y_center = min(max(x_center, 0), 1), min(max(y_center, 0), 1)
                w_box, h_box = min(w_box, 1), min(h_box, 1)

                f.write(f"0 {x_center} {y_center} {w_box} {h_box}\n")
        
        frame_count += 1

    cap.release()
    print(f"Terminé ! {frame_count} frames extraites.")

    # Création du fichier YAML pour YOLO
    yaml_content = f"""
path: ../{OUTPUT_DIR}
train: images/train
val: images/val

names:
  0: crevettes
    """
    with open(f"{OUTPUT_DIR}/shrimp_data.yaml", 'w') as f:
        f.write(yaml_content)

if __name__ == "__main__":
    generate_yolo_data()