import cv2
import numpy as np
import pandas as pd
import time
import json
from pathlib import Path
from ultralytics import YOLO

###############################################################################
# CONFIGURATION
###############################################################################
MODELS_DIR = Path("SHRIMP_TRAINED")
IMAGES_DIR = Path("SHRIMP_Dataset/images/test")
LABELS_DIR = Path("SHRIMP_Dataset/labels/test")

OUTPUT_CSV = Path("shrimp_metrics_comparison.csv")
OUTPUT_JSON = Path("shrimp_raw_errors.json")
###############################################################################

def read_ground_truth(label_path, img_w, img_h):
    if not label_path.exists(): return []
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
    if not MODELS_DIR.exists():
        print(f"[ERREUR] Dossier introuvable : {MODELS_DIR}")
        return

    model_paths = list(MODELS_DIR.glob("*.pt"))
    all_results = []
    raw_errors_dict = {}

    for model_path in model_paths:
        model_name = model_path.stem
        print(f"\n[INFO] Évaluation du modèle : {model_name}")
        model = YOLO(str(model_path))
        
        errors_l2 = []
        total_gt_points = 0
        detected_points = 0
        total_inference_time = 0.0
        image_count = 0

        for img_path in IMAGES_DIR.glob("*.png"):
            label_path = LABELS_DIR / f"{img_path.stem}.txt"
            img = cv2.imread(str(img_path))
            if img is None: continue
            img_h, img_w = img.shape[:2]

            gt_kpts = read_ground_truth(label_path, img_w, img_h)
            if len(gt_kpts) == 0: continue

            valid_gt_mask = ~np.isnan(gt_kpts[:, 0])
            total_gt_points += np.sum(valid_gt_mask)
            image_count += 1

            start_time = time.time()
            results = model(img_path, verbose=False)
            total_inference_time += (time.time() - start_time)
            
            if len(results[0].keypoints.xy) == 0: continue
            pred_kpts = results[0].keypoints.xy[0].cpu().numpy()

            for i in range(min(len(gt_kpts), len(pred_kpts))):
                if valid_gt_mask[i] and (pred_kpts[i][0] > 0 or pred_kpts[i][1] > 0):
                    detected_points += 1
                    dist = np.linalg.norm(gt_kpts[i] - pred_kpts[i])
                    errors_l2.append(float(dist))

        if errors_l2:
            fps = image_count / total_inference_time if total_inference_time > 0 else 0
            rmse = np.sqrt(np.mean(np.square(errors_l2)))
            mae = np.mean(errors_l2)
            
            all_results.append({
                "Model": model_name,
                "RMSE (px)": round(rmse, 2),
                "MAE (px)": round(mae, 2),
                "Detection Rate (%)": round((detected_points / total_gt_points) * 100, 2),
                "Total Inference Time (s)": round(total_inference_time, 2),
                "FPS": round(fps, 1)
            })
            
            raw_errors_dict[model_name] = errors_l2
            print(f"   => RMSE: {rmse:.2f} px | Vitesse: {fps:.1f} FPS")

    pd.DataFrame(all_results).to_csv(OUTPUT_CSV, index=False)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(raw_errors_dict, f)
        
    print(f"\n[SUCCES] Tableau récapitulatif sauvegardé : {OUTPUT_CSV}")
    print(f"[SUCCES] Données brutes pour les graphiques : {OUTPUT_JSON}")

if __name__ == "__main__":
    main()