import numpy as np

# 1. Carica il test set reale
data = np.load(
    r"C:\Users\ciara\Desktop\STMCUBEMODEL\val_features.npz"
)  # Modifica il percorso se l'hai spostato sul Desktop
X_test = data["features"]  # Estrae solo le features numeriche[cite: 2]

# 2. Formatta nel tensore atteso: (N, 51, 10, 1) in Float32[cite: 1, 2]
X_test_formatted = X_test[..., np.newaxis].astype(np.float32)

# 3. Salva il file pulito per STM32Cube.AI Studio
out_path = r"C:\Users\ciara\Desktop\STMCUBEMODEL\val_features.npy"
np.save(out_path, X_test_formatted)

print(
    f"File di TEST salvato con successo: {out_path} (Shape:"
    f" {X_test_formatted.shape})"
)