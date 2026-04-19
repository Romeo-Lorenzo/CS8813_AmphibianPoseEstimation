import cv2
import os
import random

# --- CONFIGURATION ---
DATASET_PATH = 'shrimp_dataset'
SPLIT = 'train'  # On vérifie le dossier train

def verify_annotations():
    img_dir = os.path.join(DATASET_PATH, 'images', SPLIT)
    lbl_dir = os.path.join(DATASET_PATH, 'labels', SPLIT)
    
    # Choisir une image au hasard
    images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    if not images:
        print("Aucune image trouvée. Lance le premier script d'abord !")
        return
    
    img_name = random.choice(images)
    lbl_name = img_name.replace('.jpg', '.txt')
    
    img_path = os.path.join(img_dir, img_name)
    lbl_path = os.path.join(lbl_dir, lbl_name)
    
    # Charger l'image
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    
    if os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # Format YOLO : class x_center y_center width height (normalisé)
                data = line.split()
                x_c, y_c, wb, hb = map(float, data[1:])
                
                # Conversion en pixels pour OpenCV
                # (x_center * w) - (width * w / 2) = x_min
                xmin = int((x_c - wb/2) * w)
                ymin = int((y_c - hb/2) * h)
                xmax = int((x_c + wb/2) * w)
                ymax = int((y_c + hb/2) * h)
                
                # Dessiner le rectangle (Vert)
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(img, "Crevette", (xmin, ymin-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Affichage
    cv2.imshow(f"Verification: {img_name}", img)
    print(f"Affichage de {img_name}. Appuie sur une touche pour fermer.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    verify_annotations()