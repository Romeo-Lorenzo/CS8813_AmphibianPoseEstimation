import pandas as pd
import json
from pathlib import Path

###############################################################################
# CONFIGURATION
###############################################################################
DLC_CSV_PATH = "l2_per_keypoint_distances.csv"
SLEAP_CSV_PATH = "sleap_l2_distances_per_point.csv"
JSON_PATH = "raw_errors.json"

DLC_MODEL_NAME = "DeepLabCut"
SLEAP_MODEL_NAME = "SLEAP"
###############################################################################

def main():
    json_file = Path(JSON_PATH)
    
    # 1. Chargement du dictionnaire JSON existant
    if json_file.exists():
        print(f"[INFO] Chargement du fichier existant : {JSON_PATH}")
        with open(json_file, "r") as f:
            data = json.load(f)
    else:
        print(f"[INFO] Fichier {JSON_PATH} non trouvé. Création d'un nouveau dictionnaire.")
        data = {}

    # 2. Traitement de DeepLabCut
    dlc_file = Path(DLC_CSV_PATH)
    if dlc_file.exists():
        print(f"\n[INFO] Lecture des erreurs {DLC_MODEL_NAME}...")
        # On lit les 2 premières colonnes, peu importe s'il y a un en-tête ou non
        df_dlc = pd.read_csv(dlc_file, header=None, usecols=[0, 1], names=["bodypart", "error"])
        
        # Sécurité : force la conversion en nombre, ignore le texte (devient NaN) puis supprime les NaN
        dlc_errors = pd.to_numeric(df_dlc["error"], errors="coerce").dropna().tolist()
        
        data[DLC_MODEL_NAME] = dlc_errors
        print(f"[SUCCES] {len(dlc_errors)} valeurs L2 ajoutées pour {DLC_MODEL_NAME}.")
    else:
        print(f"[ATTENTION] Fichier {DLC_CSV_PATH} introuvable. Ignoré.")

    # 3. Traitement de SLEAP
    sleap_file = Path(SLEAP_CSV_PATH)
    if sleap_file.exists():
        print(f"\n[INFO] Lecture des erreurs {SLEAP_MODEL_NAME}...")
        df_sleap = pd.read_csv(sleap_file)
        
        # SLEAP a un en-tête officiel, on cherche la colonne l2 (ou L2)
        col_name = "l2" if "l2" in df_sleap.columns else "L2" if "L2" in df_sleap.columns else None
        
        if col_name:
            # Sécurité : force la conversion en nombre
            sleap_errors = pd.to_numeric(df_sleap[col_name], errors="coerce").dropna().tolist()
            data[SLEAP_MODEL_NAME] = sleap_errors
            print(f"[SUCCES] {len(sleap_errors)} valeurs L2 ajoutées pour {SLEAP_MODEL_NAME}.")
        else:
            print(f"[ERREUR] La colonne 'l2' est introuvable dans {SLEAP_CSV_PATH}.")
    else:
        print(f"[ATTENTION] Fichier {SLEAP_CSV_PATH} introuvable. Ignoré.")

    # 4. Sauvegarde du fichier JSON mis à jour
    if data:
        with open(json_file, "w") as f:
            json.dump(data, f)
        print(f"\n[SUCCES] Mise à jour de {JSON_PATH} terminée avec succès !")
        print("[INFO] Vous pouvez maintenant relancer le script des graphiques (plot_paper_metrics.py).")
    else:
        print("\n[ERREUR] Aucune donnée n'a été ajoutée. Le JSON n'a pas été sauvegardé.")

if __name__ == "__main__":
    main()