import numpy as np
import tensorflow as tf

# Carica il modello Keras
model = tf.keras.models.load_model("ds_cnn_s_best.keras")

# Carica il dataset di test originale con le labels
data = np.load(r"C:\Users\ciara\Desktop\STMCUBEMODEL\test_features.npz")
X_test, y_test_str = data["features"], data["labels"]

X_test = X_test[..., np.newaxis].astype(np.float32)

# Mappa le stringhe in interi (assicurati di usare lo stesso dict del training)
LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "unknown", "silence"]
label_to_idx = {lbl: i for i, lbl in enumerate(LABELS)}
y_test = np.array([label_to_idx[l] for l in y_test_str])

# Valuta l'accuratezza predittiva reale
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Vera Accuratezza sul Test Set: {accuracy * 100:.2f}%")