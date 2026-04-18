import cv2
import time
import torch
from ultralytics import YOLO

def main():
    model = YOLO('yolov8n-pose.pt')
    
    try:
        model.to('cuda:0')
        print("Succès : Utilisation du device 0 (GPU) confirmée.")
    except Exception as e:
        print(f"Erreur : Impossible de forcer le device 0. PyTorch ne détecte pas CUDA. Détails : {e}")
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la webcam.")
        return

    print("Appuyez sur 'q' pour quitter.")

    while True:
        start_time = time.time()
        
        success, frame = cap.read()
        
        if not success:
            print("Erreur : Impossible de lire la frame.")
            break

        results = model(frame, stream=True, verbose=False)

        for result in results:
            annotated_frame = result.plot()

            end_time = time.time()
            fps = 1 / (end_time - start_time)
            
            cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("YOLO-Pose : Skeletonize Temps Reel", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Application fermée.")

if __name__ == "__main__":
    main()