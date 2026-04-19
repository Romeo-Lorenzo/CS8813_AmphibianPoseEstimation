import cv2
import time
import torch
from ultralytics import YOLO

# 1. Vérification et configuration du GPU
device = 0 if torch.cuda.is_available() else 'cpu'
print(f"Utilisation du périphérique : {device}")

# 2. Charger le modèle et l'envoyer sur le GPU
model = YOLO('yolo26nbest.pt').to(device)

# 3. Accéder à la webcam
cap = cv2.VideoCapture(0)

# Variables pour le calcul des FPS
prev_time = 0
fps = 0

if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la webcam.")
    exit()

print("Appuyez sur 'q' pour quitter.")

while True:
    success, frame = cap.read()
    if not success:
        break

    # Calcul du temps actuel pour le FPS
    current_time = time.time()
    
    # 4. Inférence optimisée
    # device=device : force l'utilisation du GPU
    # half=True : utilise la demi-précision (FP16) pour doubler la vitesse sur GPU
    # stream=True : gestion efficace de la mémoire pour la vidéo
    results = model.predict(frame, stream=True, device=device, half=True, verbose=False)

    # 5. Afficher les résultats
    for r in results:
        annotated_frame = r.plot()

    # Calcul des FPS (moyenne glissante simple)
    # On calcule la différence de temps entre deux frames
    diff = current_time - prev_time
    if diff > 0:
        fps = 1 / diff
    prev_time = current_time

    # 6. Afficher le compteur de FPS sur l'image
    # Texte en haut à gauche, couleur verte (0, 255, 0)
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Affichage de la fenêtre
    cv2.imshow("YOLO Real-Time GPU + FPS", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()