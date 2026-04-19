from pathlib import Path
from ultralytics import YOLO

###############################################################################
# CONFIGURATION EN DUR (HARDCODED)
###############################################################################
YAML_PATH = "SHRIMP_Dataset/dataset.yaml"
MODEL_NAME = "yolo26n-pose.pt"  # Corrigé : v8n ou yolo11n-pose.pt
EPOCHS = 30                     # Parfait pour un premier test rapide (vise 100-300 plus tard)
BATCH_SIZE = 32                 # Très bien pour les 8 Go VRAM de ta RTX 5060
IMAGE_SIZE = 640                # Résolution standard de YOLO
###############################################################################

def main():
    yaml_file = Path(YAML_PATH)
    if not yaml_file.exists():
        raise FileNotFoundError(f"Fichier introuvable: {yaml_file}. Vérifiez que le dossier existe.")

    print(f"Initialisation du modèle : {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    print(f"Début de l'entraînement pour {EPOCHS} époques avec des batchs de {BATCH_SIZE}...")
    model.train(
        data=str(yaml_file),
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        project="shrimp_pose_training", 
        name="shrimp_yolo",                
        device="0"  # Va bien taper dans ta RTX 5060 configurée avec CUDA 12.8
    )

    print("\n[SUCCES] Entraînement terminé !")
    # Le chemin d'affichage corrigé pour correspondre à tes variables project et name
    print("Les meilleurs poids ont été sauvegardés sous : yolo_pose_training/shrimp_yolo/weights/best.pt")

if __name__ == "__main__":
    main()