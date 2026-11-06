import numpy as np

# Carica il dataset di test originale
data = np.load(r"C:\Users\ciara\Desktop\STMCUBEMODEL\test_features.npz")
y_test_str = data["labels"]

# Mappa le stringhe in interi esattamente come nel training
LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "unknown", "silence"]
label_to_idx = {lbl: i for i, lbl in enumerate(LABELS)}
y_test_int = np.array([label_to_idx[l] for l in y_test_str])

# Converte in One-Hot Encoding per combaciare con l'output (N, 12) della rete
NUM_CLASSES = 12
y_test_onehot = np.eye(NUM_CLASSES, dtype=np.float32)[y_test_int]

# Salva il file degli output attesi
out_path = r"C:\Users\ciara\Desktop\STMCUBEMODEL\test_outputs_expected.npy"
np.save(out_path, y_test_onehot)
print(f"File etichette salvato: {out_path} (Shape: {y_test_onehot.shape})")