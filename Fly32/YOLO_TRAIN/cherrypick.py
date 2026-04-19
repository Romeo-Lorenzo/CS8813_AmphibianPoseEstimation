import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

###############################################################################
# CONFIGURATION
###############################################################################
# Remplacez par le chemin exact de votre modèle
MODEL_PATH = Path("SHRIMP_TRAINED/yolo26_best.pt") 

IMAGES_DIR = Path("SHRIMP_Dataset/images/test")
LABELS_DIR = Path("SHRIMP_Dataset/labels/test")

# Dossier où seront sauvegardées les images avec le squelette et l'erreur
OUTPUT_DIR = Path("SHRIMP_TEST_RESULTS_VISUALIZATION")
###############################################################################

def read_ground_truth(label_path, img_w, img_h):
    """Lit le fichier texte YOLO pour extraire les keypoints de la vérité terrain."""
    if not label_path.exists(): 
        return []
    with open(label_path, "r") as f:
        line = f.readline().strip()
        if not line: return []
        kpts_data = line.split()[5:]
        kpts = []
        for i in range(0, len(kpts_data), 3):
            vis = float(kpts_data[i+2])
            if vis == 0:
                kpts.append((np.nan, np.nan))
            else:
                x = float(kpts_data[i]) * img_w
                y = float(kpts_data[i+1]) * img_h
                kpts.append((x, y))
        return np.array(kpts)

def main():
    if not MODEL_PATH.exists():
        print(f"[ERREUR] Modèle introuvable : {MODEL_PATH}")
        return
    
    if not IMAGES_DIR.exists():
        print(f"[ERREUR] Dossier d'images introuvable : {IMAGES_DIR}")
        return

    # Création du dossier de sortie
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Chargement du modèle : {MODEL_PATH.name}")
    model = YOLO(str(MODEL_PATH))

    image_paths = list(IMAGES_DIR.glob("*.*"))
    # Filtrer pour ne garder que les images (png, jpg, jpeg)
    image_paths = [p for p in image_paths if p.suffix.lower() in ['.png', '.jpg', '.jpeg']]

    print(f"[INFO] {len(image_paths)} images trouvées. Début de l'évaluation visuelle...")

    # --- NOUVEAUTÉ : Variables pour traquer la plus grande et la plus petite erreur ---
    global_min_error = float('inf')
    global_max_error = float('-inf')
    min_error_img = ""
    max_error_img = ""

    for img_path in image_paths:
        label_path = LABELS_DIR / f"{img_path.stem}.txt"
        img = cv2.imread(str(img_path))
        
        if img is None:
            print(f"[ATTENTION] Impossible de lire l'image {img_path.name}")
            continue
            
        img_h, img_w = img.shape[:2]
        
        center_x = img_w // 2
        center_y = img_h // 2

        # 1. Lecture de la vérité terrain (Ground Truth)
        gt_kpts = read_ground_truth(label_path, img_w, img_h)
        
        # 2. Inférence YOLO
        results = model(img_path, verbose=False)
        
        # 3. Récupération de l'image annotée par YOLO
        annotated_img = results[0].plot()

        # 4. Traitement des points prédits et calcul de l'erreur
        img_errors_l2 = []
        
        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            pred_kpts = results[0].keypoints.xy[0].cpu().numpy() 

            # Marquer le centre de l'image
            cv2.circle(annotated_img, (center_x, center_y), 4, (0, 0, 255), -1) 
            
            for kp in pred_kpts:
                px, py = int(kp[0]), int(kp[1])
                if px > 0 and py > 0: 
                    cv2.line(annotated_img, (center_x, center_y), (px, py), (255, 255, 255), 1)

            # Calcul de l'erreur L2
            if len(gt_kpts) > 0:
                valid_gt_mask = ~np.isnan(gt_kpts[:, 0])
                for i in range(min(len(gt_kpts), len(pred_kpts))):
                    if valid_gt_mask[i] and (pred_kpts[i][0] > 0 or pred_kpts[i][1] > 0):
                        dist = np.linalg.norm(gt_kpts[i] - pred_kpts[i])
                        img_errors_l2.append(float(dist))

        # 5. Préparation du texte et mise à jour des statistiques globales
        if img_errors_l2:
            mean_l2 = np.mean(img_errors_l2)
            
            # --- NOUVEAUTÉ : Mise à jour des records d'erreurs ---
            if mean_l2 < global_min_error:
                global_min_error = mean_l2
                min_error_img = img_path.name
                
            if mean_l2 > global_max_error:
                global_max_error = mean_l2
                max_error_img = img_path.name
            
            text = f"Erreur L2 moy: {mean_l2:.2f} px"
            color = (0, 255, 0) if mean_l2 < 15 else (0, 0, 255) 
        elif len(gt_kpts) == 0:
            text = "Aucun label GT (Verite Terrain)"
            color = (255, 255, 0) 
        else:
            text = "Non detecte par le modele"
            color = (0, 0, 255) 

        # 6. Ajout du texte sur l'image
        cv2.putText(annotated_img, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(annotated_img, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)    

        # 7. Sauvegarde de l'image
        output_path = OUTPUT_DIR / img_path.name
        cv2.imwrite(str(output_path), annotated_img)

    print(f"\n[SUCCES] Les images sont sauvegardées dans : {OUTPUT_DIR.absolute()}")

    # --- NOUVEAUTÉ : Affichage du bilan final dans le terminal ---
    if global_max_error != float('-inf'):
        print("\n" + "="*50)
        print(" 📊 BILAN DES ERREURS L2 SUR LE TEST SET")
        print("="*50)
        print(f"✅ Plus PETITE erreur  : {global_min_error:.2f} px (Image: {min_error_img})")
        print(f"❌ Plus GRANDE erreur  : {global_max_error:.2f} px (Image: {max_error_img})")
        print("="*50 + "\n")
    else:
        print("\n[ATTENTION] Aucune erreur n'a pu être calculée (vérifiez vos labels GT).")

if __name__ == "__main__":
    main()