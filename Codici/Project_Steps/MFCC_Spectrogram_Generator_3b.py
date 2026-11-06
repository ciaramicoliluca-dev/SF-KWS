"""
STEP 2b — Verifica visiva di alcuni spettrogrammi MFCC gia' estratti
-----------------------------------------------------------------------
Legge il file .npz prodotto da step2_extract_dataset.py e disegna una
griglia di esempi presi a caso (uno per classe, se possibile), cosi' da
controllare a colpo d'occhio che l'estrazione abbia senso prima di
passare al training.

Uso:
    Modifica NPZ_FILE e N_EXAMPLES_PER_CLASS qui sotto, poi:
        python step2b_verify_spectrograms.py
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
NPZ_FILE = "val_features.npz"       # quale file controllare
N_EXAMPLES_PER_CLASS = 3            # quanti esempi a testa per classe
SEED = 42
OUT_PNG = "spectrogram_check.png"
# ============================================================

data = np.load(NPZ_FILE)
X, y = data["features"], data["labels"]
print(f"File: {NPZ_FILE}")
print(f"Shape features: {X.shape}  |  esempi totali: {len(y)}")

classes = sorted(set(y))
print(f"Classi trovate ({len(classes)}): {classes}")

rng = np.random.default_rng(SEED)
picks = []  # lista di (indice, label)
for c in classes:
    idx_c = np.where(y == c)[0]
    chosen = rng.choice(idx_c, size=min(N_EXAMPLES_PER_CLASS, len(idx_c)), replace=False)
    for i in chosen:
        picks.append((i, c))

n = len(picks)
n_cols = 4
n_rows = int(np.ceil(n / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.6 * n_rows))
axes = np.array(axes).reshape(-1)

for ax, (idx, label) in zip(axes, picks):
    ax.imshow(X[idx].T, origin="lower", aspect="auto", cmap="magma")
    ax.set_title(label, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

# spegni gli assi vuoti in eccesso
for ax in axes[n:]:
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
print(f"Griglia salvata in {OUT_PNG} ({n} esempi, {N_EXAMPLES_PER_CLASS} per classe)")
