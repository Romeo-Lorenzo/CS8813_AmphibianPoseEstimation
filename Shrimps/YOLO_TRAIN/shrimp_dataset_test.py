import cv2
import random
import matplotlib.pyplot as plt
from pathlib import Path

###############################################################################
# CONFIGURATION
###############################################################################
# Remplacez par le chemin de votre dataset généré
DATASET_DIR = Path("SHRIMP_Dataset")
###############################################################################

def get_random_sample(split_name: str, dataset_dir: Path):
    """Récupère un chemin d'image au hasard et son label pour un split donné."""
    img_dir = dataset_dir / "images" / split_name
    lbl_dir = dataset_dir / "labels" / split_name
    
    if not img_dir.exists():
        return None, None

    # Chercher toutes les images (png, jpg, jpeg)
    images = [p for p in img_dir.glob("*.*") if p.suffix.lower() in ['.png', '.jpg', '.jpeg']]
    
    if not images:
        return None, None

    img_path = random.choice(images)
    lbl_path = lbl_dir / f"{img_path.stem}.txt"
    
    return img_path, lbl_path

def draw_labels_on_image(img_path: Path, lbl_path: Path):
    """Charge l'image, dessine les BBox et les points clés, et la renvoie."""
    # Lire l'image avec OpenCV et la convertir en RGB (pour Matplotlib)
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h, w = img.shape[:2]

    if lbl_path.exists():
        with open(lbl_path, "r") as f:
            lines = f.readlines()
            
            for line in lines:
                data = line.strip().split()
                if len(data) < 5:
                    continue
                
                # 1. Dessiner la Bounding Box
                # Format YOLO : class_id xc yc width height
                cx, cy, bw, bh = map(float, data[1:5])
                xmin = int((cx - bw / 2) * w)
                ymin = int((cy - bh / 2) * h)
                xmax = int((cx + bw / 2) * w)
                ymax = int((cy + bh / 2) * h)
                
                # Rectangle vert
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                
                # 2. Dessiner les points (Keypoints)
                if len(data) > 5:
                    kpts_data = data[5:]
                    for i in range(0, len(kpts_data), 3):
                        px = float(kpts_data[i]) * w
                        py = float(kpts_data[i+1]) * h
                        vis = float(kpts_data[i+2])
                        
                        # Si le point est visible (vis > 0), on le dessine
                        if vis > 0:
                            # Cercle rouge
                            cv2.circle(img, (int(px), int(py)), 4, (255, 0, 0), -1)
                            # Petit contour blanc pour mieux le voir
                            cv2.circle(img, (int(px), int(py)), 4, (255, 255, 255), 1)

    return img

def main():
    splits = ["train", "val", "test"]
    
    # Préparation de la figure Matplotlib
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Vérification rapide du Dataset YOLO Pose (Échantillons aléatoires)", fontsize=16)

    for i, split in enumerate(splits):
        ax = axes[i]
        
        img_path, lbl_path = get_random_sample(split, DATASET_DIR)
        
        if img_path is None:
            ax.set_title(f"[{split.upper()}] - Dossier vide ou introuvable")
            ax.axis('off')
            continue
            
        annotated_img = draw_labels_on_image(img_path, lbl_path)
        
        if annotated_img is not None:
            ax.imshow(annotated_img)
            ax.set_title(f"[{split.upper()}]\n{img_path.name}")
            ax.axis('off') # Cache les axes X et Y pour faire plus propre
        else:
            ax.set_title(f"[{split.upper()}] - Erreur lecture image")
            ax.axis('off')

    plt.tight_layout()
    # Affiche la fenêtre (bloque le script tant que la fenêtre n'est pas fermée)
    plt.show()

if __name__ == "__main__":
    main()