from ultralytics import YOLO
import torch

def train_shrimp_gpu_light():
    # Vérification CUDA
    if not torch.cuda.is_available():
        print("Erreur : GPU non détecté.")
        return

    # 1. Utiliser le modèle 'Nano' (le plus léger, idéal pour 4GB)
    # yolo26x.pt est très rapide et consomme peu de mémoire vive
    model = YOLO('yolo26n.pt')

    # 2. Lancer l'entraînement avec paramètres bridés pour la VRAM
    model.train(
        data='shrimp_dataset/shrimp_data.yaml',
        epochs=100,
        imgsz=640,          # On garde 640 pour la précision, mais on peut descendre à 480 si ça plante
        batch=8,            # 8 est le "sweet spot" pour 4GB. Si ça crash, passe à 4.
        workers=2,          # Moins de threads CPU pour libérer de la RAM système
        device=0,           # Force l'usage du GPU
        amp=True,           # Active l'Automatic Mixed Precision (gain de mémoire énorme !)
        save=True,
        project='shrimp_project',
        name='vram_optimized'
    )

if __name__ == "__main__":
    train_shrimp_gpu_light()