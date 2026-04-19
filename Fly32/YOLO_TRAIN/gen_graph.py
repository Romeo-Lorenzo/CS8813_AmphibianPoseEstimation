import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

###############################################################################
# CONFIGURATION
###############################################################################
INPUT_JSON = "shrimp_raw_errors.json"

# Noms exacts des fichiers générés pour le LaTeX
OUTPUT_PCK = "shrimp_Figure_PCK_Curve.png"
OUTPUT_VIOLIN = "shrimp_Figure_Violin_Plot.png"
OUTPUT_PARETO = "shrimp_Figure_Speed_Accuracy.png"

PCK_MAX_THRESHOLD = 15.0 

# DONNÉES POUR LE GRAPHIQUE SPEED VS ACCURACY (Pareto Front)
# Mettez à jour ces valeurs avec vos résultats finaux
SUMMARY_METRICS = {
    "yolov8n_best": {"fps": 17.2, "rmse": 3.91, "color": "#1f77b4"},
    "yolo26x_best": {"fps": 1.3,  "rmse": 3.57, "color": "#ff7f0e"},
    "DeepLabCut":   {"fps": 8.5,  "rmse": 3.25, "color": "#2ca02c"},
    "SLEAP":        {"fps": 45.0, "rmse": 3.80, "color": "#d62728"}
}
###############################################################################

def load_data():
    if not Path(INPUT_JSON).exists():
        print(f"[ERREUR] Fichier {INPUT_JSON} introuvable.")
        return None
    with open(INPUT_JSON, "r") as f:
        return json.load(f)

def plot_violin(data_dict):
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    records = []
    for model_name, errors in data_dict.items():
        for err in errors:
            records.append({"Model": model_name, "Error (pixels)": err})
            
    df = pd.DataFrame(records)
    df_filtered = df[df["Error (pixels)"] <= 30]

    sns.violinplot(
        x="Model", y="Error (pixels)", data=df_filtered, 
        inner="quartile", linewidth=1.5, cut=0, palette="muted"
    )

    plt.title("Keypoint Error Distribution Comparison", fontsize=14, fontweight="bold")
    plt.xlabel("Pose Estimation Framework", fontsize=12)
    plt.ylabel("L2 Error Distance (pixels)", fontsize=12)
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_VIOLIN, dpi=300)
    print(f"[SUCCES] Graphique sauvegardé : {OUTPUT_VIOLIN}")
    plt.close()

def plot_pck(data_dict):
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    
    thresholds = np.linspace(0, PCK_MAX_THRESHOLD, 100)
    
    for model_name, errors in data_dict.items():
        errors = np.array(errors)
        pck_values = []
        for t in thresholds:
            correct = np.sum(errors <= t)
            pck = (correct / len(errors)) * 100 if len(errors) > 0 else 0
            pck_values.append(pck)
            
        linestyle = "--" if model_name in ["DeepLabCut", "SLEAP"] else "-"
        plt.plot(thresholds, pck_values, label=model_name, linewidth=2.5, linestyle=linestyle)

    plt.title("Percentage of Correct Keypoints (PCK)", fontsize=14, fontweight="bold")
    plt.xlabel("Distance Threshold (pixels)", fontsize=12)
    plt.ylabel("Keypoints Detected (%)", fontsize=12)
    plt.xlim(0, PCK_MAX_THRESHOLD)
    plt.ylim(0, 100)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_PCK, dpi=300)
    print(f"[SUCCES] Graphique sauvegardé : {OUTPUT_PCK}")
    plt.close()

def plot_pareto():
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")

    for model, metrics in SUMMARY_METRICS.items():
        plt.scatter(
            metrics["fps"], metrics["rmse"], 
            s=200, c=metrics["color"], edgecolor='black', zorder=5, label=model
        )
        # Annotation du texte légèrement décalée
        plt.annotate(
            model, (metrics["fps"], metrics["rmse"]), 
            xytext=(10, 0), textcoords='offset points', 
            fontsize=11, fontweight='bold', va='center'
        )

    plt.title("Speed vs. Accuracy Trade-off", fontsize=14, fontweight="bold")
    plt.xlabel("Inference Speed (FPS) $\\rightarrow$ Higher is better", fontsize=12)
    plt.ylabel("RMSE (pixels) $\\rightarrow$ Lower is better", fontsize=12)
    
    # Inverser l'axe Y pour que le meilleur score (erreur basse) soit en haut
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PARETO, dpi=300)
    print(f"[SUCCES] Graphique sauvegardé : {OUTPUT_PARETO}")
    plt.close()

def main():
    print("[INFO] Génération des graphiques pour la publication...")
    data = load_data()
    if data:
        plot_violin(data)
        plot_pck(data)
    plot_pareto()
    print("[INFO] Terminé. Images prêtes pour Overleaf.")

if __name__ == "__main__":
    main()