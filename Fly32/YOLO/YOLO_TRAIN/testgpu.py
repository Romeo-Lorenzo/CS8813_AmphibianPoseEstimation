import torch

# Vérifie si le GPU est disponible
available = torch.cuda.is_available()
print(f"GPU disponible : {available}")

if available:
    # Récupère le nom de la carte
    print(f"Nom du GPU : {torch.cuda.get_device_name(0)}")
    # Récupère la mémoire totale (en Go)
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Mémoire totale : {total_mem:.2f} GB")
else:
    print("PyTorch ne voit pas de GPU. Vérifie l'installation de tes drivers NVIDIA.")
