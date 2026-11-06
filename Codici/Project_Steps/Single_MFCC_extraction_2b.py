"""
STEP 1 — Estrazione feature MFCC con SpeechBrain
---------------------------------------------------
Non fa altro: prende UN file audio, calcola le MFCC, stampa la shape
e disegna lo spettrogramma. Nessun DS-CNN, nessun CSV, nessuna pipeline.

Uso:
    python step1_mfcc_extraction.py "C:/percorso/al/tuo/file.wav"
"""

import sys

import matplotlib.pyplot as plt
import soundfile as sf
import torch
from speechbrain.lobes.features import MFCC

# --- 1. Leggo il file audio dal disco -------------------------------------
path = sys.argv[1] if len(sys.argv) > 1 else "test.wav"
audio, sample_rate = sf.read(path, dtype="float32")
print(f"File letto: {path}")
print(f"Sample rate: {sample_rate} Hz, durata: {len(audio) / sample_rate:.2f} s")

if sample_rate != 16000:
    raise ValueError(
        f"Il file ha sample rate {sample_rate}, ma Google Speech Commands "
        "usa 16000 Hz. Verifica di aver preso un file giusto."
    )

# --- 2. Creo l'estrattore MFCC di SpeechBrain ------------------------------
# deltas=False, context=False: vogliamo solo i 10 coefficienti MFCC "puri"
# per ogni frame, niente derivate o contesto extra (ci penserà la rete dopo).
mfcc_extractor = MFCC(
    deltas=False,
    context=False,
    sample_rate=16000,
    n_fft=1024,         # potenza di 2
    n_mels=40,          # numero di filtri Mel
    n_mfcc=10,          # numero di coefficienti MFCC in uscita
    win_length=40,      # finestra di analisi, in ms
    hop_length=20,      # passo tra una finestra e l'altra, in ms
)
mfcc_extractor.eval()

# --- 3. Calcolo le feature --------------------------------------------------
# SpeechBrain vuole un tensore [batch, tempo]: aggiungo la dimensione batch.
waveform = torch.from_numpy(audio).unsqueeze(0)   # shape: [1, num_campioni]

with torch.no_grad():
    features = mfcc_extractor(waveform)            # shape: [1, T, n_mfcc]

features = features.squeeze(0).numpy()              # shape: [T, n_mfcc]
print(f"Shape delle feature MFCC (frame temporali, coefficienti): {features.shape}")

# --- 4. Verifica visiva ------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.imshow(features.T, origin="lower", aspect="auto", cmap="magma")
plt.xlabel("Frame temporale")
plt.ylabel("Coefficiente MFCC")
plt.colorbar(label="Ampiezza")
plt.title("MFCC estratte con SpeechBrain")
plt.tight_layout()
plt.savefig("mfcc_output.png", dpi=150)
print("Grafico salvato in mfcc_output.png")
