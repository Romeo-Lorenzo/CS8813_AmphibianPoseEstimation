import argparse
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
from PIL import Image
import sleap_io as sio

###############################################################################
# CONFIGURATION PAR DÉFAUT
###############################################################################
DEFAULT_TRAIN_SLP = "shrimp_train.pkg.slp"
DEFAULT_VAL_SLP = "shrimp_val.pkg.slp"
DEFAULT_TEST_SLP = "shrimp_last150.pkg.slp"
DEFAULT_OUT_DIR = "SHRIMP_Dataset"
BBOX_PADDING = 0.1  # Ajoute 10% de marge autour des points pour la boîte YOLO
###############################################################################

def process_split(slp_path: Path, split_name: str, out_dir: Path) -> int:
    """Traite un fichier .slp et génère les images et labels pour un sous-ensemble (train/val/test)."""
    if not slp_path.exists():
        print(f"⚠️  Fichier introuvable, skip : {slp_path}")
        return 0

    slp_loader = cast(Callable[[str], Any], getattr(sio, "load_slp", None))
    labels = slp_loader(str(slp_path))
    
    if not labels.skeletons:
        raise RuntimeError(f"Aucun squelette trouvé dans {slp_path}.")

    # Création des dossiers images/ et labels/ pour ce split
    images_dir = out_dir / "images" / split_name
    labels_dir = out_dir / "labels" / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    exported_count = 0

    for row_index, labeled_frame in enumerate(labels.labeled_frames):
        image_array = labeled_frame.image
        
        # Récupération de TOUTES les instances (toutes les crevettes)
        instances = labeled_frame.user_instances if labeled_frame.user_instances else labeled_frame.instances

        if image_array is None or not instances:
            continue

        # Récupération des dimensions de l'image
        image_array = image_array.squeeze()
        img_h, img_w = image_array.shape[:2]

        yolo_lines_for_this_frame = []

        # Boucle sur TOUTES les crevettes de l'image
        for instance in instances:
            points = instance.numpy() # Forme: (num_nodes, 2)
            
            # Filtrer les points valides (ignorer les NaN) pour calculer la Bounding Box
            valid_points = points[~np.isnan(points).any(axis=1)]
            if len(valid_points) == 0:
                continue # Aucun point valide pour cette crevette, on passe à la suivante

            # Calcul de la bounding box YOLO
            xmin, ymin = np.min(valid_points, axis=0)
            xmax, ymax = np.max(valid_points, axis=0)

            # Ajout d'une marge (padding)
            pad_x = (xmax - xmin) * BBOX_PADDING
            pad_y = (ymax - ymin) * BBOX_PADDING
            xmin = max(0, xmin - pad_x)
            ymin = max(0, ymin - pad_y)
            xmax = min(img_w, xmax + pad_x)
            ymax = min(img_h, ymax + pad_y)

            # Normalisation YOLO pour la bounding box (centre_x, centre_y, largeur, hauteur)
            xc = ((xmin + xmax) / 2) / img_w
            yc = ((ymin + ymax) / 2) / img_h
            w = (xmax - xmin) / img_w
            h = (ymax - ymin) / img_h

            # Construction de la ligne YOLO pour CETTE crevette
            yolo_line = f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
            
            for x, y in points:
                if np.isnan(x) or np.isnan(y):
                    # Point manquant : visibilité = 0
                    yolo_line += " 0.000000 0.000000 0"
                else:
                    # Point présent : normalisation et visibilité = 2 (visible)
                    yolo_line += f" {x/img_w:.6f} {y/img_h:.6f} 2"
            
            yolo_lines_for_this_frame.append(yolo_line)

        # S'il n'y a finalement aucune ligne valide générée, on ignore cette frame
        if not yolo_lines_for_this_frame:
            continue

        # Sauvegarde
        base_filename = f"{slp_path.stem.replace('.pkg', '')}_{row_index:05d}"
        
        # 1. Sauvegarde de l'image (une seule fois par frame)
        image_path = images_dir / f"{base_filename}.png"
        Image.fromarray(image_array).save(image_path)

        # 2. Sauvegarde du label txt (fusion de toutes les crevettes avec des retours à la ligne)
        label_path = labels_dir / f"{base_filename}.txt"
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines_for_this_frame) + "\n")

        exported_count += 1

    print(f"✅ [{split_name.upper()}] Exportation terminée : {exported_count} frames.")
    
    # On renvoie le nombre de keypoints pour le fichier YAML
    num_keypoints = len(labels.skeletons[0].nodes)
    return num_keypoints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SLEAP to YOLO Pose format.")
    parser.add_argument("--train", type=Path, default=Path(DEFAULT_TRAIN_SLP))
    parser.add_argument("--val", type=Path, default=Path(DEFAULT_VAL_SLP))
    parser.add_argument("--test", type=Path, default=Path(DEFAULT_TEST_SLP))
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    
    print(f"🚀 Début de l'exportation YOLO vers : {out_dir}\n")

    num_kpts_train = process_split(args.train, "train", out_dir)
    num_kpts_val = process_split(args.val, "val", out_dir)
    num_kpts_test = process_split(args.test, "test", out_dir)

    num_keypoints = max(num_kpts_train, num_kpts_val, num_kpts_test)

    if num_keypoints > 0:
        yaml_path = out_dir / "dataset.yaml"
        yaml_content = f"""path: {out_dir}
train: images/train
val: images/val
test: images/test

# Configuration Pose
kpt_shape: [{num_keypoints}, 3] # [nombre de points, dimension(x,y,visibilité)]
nc: 1
names:
  0: animal
"""
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        
        print(f"\n📄 Fichier de configuration généré : {yaml_path}")
        print("\n🎉 Exportation réussie ! Vous êtes prêt pour YOLO.")
    else:
        print("\n❌ Aucun point clé exporté. Vérifiez vos fichiers .slp.")

if __name__ == "__main__":
    main()