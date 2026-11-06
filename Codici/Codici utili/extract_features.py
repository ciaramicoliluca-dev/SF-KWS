"""
extract_features.py
------------------------------------------------------------------
Estrazione feature MFCC dal GSCD per il training del DS-CNN,
a partire dai manifest CSV generati da split_gscd.py.

Configurazione MFCC (Zhang et al., "Hello Edge: Keyword Spotting on
Microcontrollers"): finestra 40 ms, stride 20 ms, 40 filtri mel,
10 coefficienti MFCC -> 49 frame x 10 coeff per clip di 1s a 16 kHz.
Questi stessi parametri andranno ripresi pari pari nella fase 2 della
roadmap (implementazione embedded del preprocessing, es. con
CMSIS-DSP), per garantire coerenza tra le feature usate in training
e quelle calcolate a runtime sulla BlueCoin.

Augmentation (solo su train, mai su val/test):
  - time shift casuale (+-100ms)
  - mixing di rumore di fondo a basso volume (prob. 0.8)
seguendo la ricetta standard usata nello script di training
originale di Google per Speech Commands.

Output: per ciascuno split, un file .npz con:
  X: array (N, 49, 10) di feature MFCC
  y: array (N,) di etichette intere
e un classes.json con la lista ordinata delle 12 classi.

Dipendenze:
    pip install numpy librosa soundfile
"""

import csv
import json
from pathlib import Path

import numpy as np
import librosa

# ---------------------------------------------------------------------------
# CONFIGURAZIONE
# ---------------------------------------------------------------------------

SPLITS_DIR = Path("./splits")
NOISE_DIR = Path("./speech_commands_v0.02/_background_noise_")
OUTPUT_DIR = Path("./features")
SEED = 42

SAMPLE_RATE = 16000
CLIP_DURATION_S = 1.0
CLIP_LEN = int(SAMPLE_RATE * CLIP_DURATION_S)  # 16000 campioni

FRAME_LENGTH_MS = 40
FRAME_STRIDE_MS = 20
N_FFT = int(SAMPLE_RATE * FRAME_LENGTH_MS / 1000)        # 640
HOP_LENGTH = int(SAMPLE_RATE * FRAME_STRIDE_MS / 1000)    # 320
N_MELS = 40
N_MFCC = 10
N_FRAMES = 1 + (CLIP_LEN - N_FFT) // HOP_LENGTH            # 49, con center=False

TIME_SHIFT_MS = 100
BACKGROUND_FREQUENCY = 0.8    # frazione di campioni "train" con rumore mixato
BACKGROUND_VOLUME_MAX = 0.1   # volume massimo del rumore mixato

CLASSES = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
           "unknown", "silence"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

rng = np.random.RandomState(SEED)

# ---------------------------------------------------------------------------
# CARICAMENTO AUDIO
# ---------------------------------------------------------------------------


def load_clip(filepath: str) -> np.ndarray:
    """
    Carica un file audio e lo porta a esattamente CLIP_LEN campioni
    (pad con zeri se più corto, taglio se più lungo). Gestisce anche le
    voci "silence" nella forma 'path#silence', ritagliando una finestra
    di 1s a offset casuale dal file di rumore sorgente.
    """
    if filepath.endswith("#silence"):
        src_path = filepath[: -len("#silence")]
        audio, _ = librosa.load(src_path, sr=SAMPLE_RATE, mono=True)
        if len(audio) <= CLIP_LEN:
            return np.pad(audio, (0, CLIP_LEN - len(audio)))
        start = rng.randint(0, len(audio) - CLIP_LEN)
        return audio[start:start + CLIP_LEN]

    audio, _ = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    if len(audio) < CLIP_LEN:
        return np.pad(audio, (0, CLIP_LEN - len(audio)))
    return audio[:CLIP_LEN]


# ---------------------------------------------------------------------------
# AUGMENTATION (solo train)
# ---------------------------------------------------------------------------


def time_shift(audio: np.ndarray) -> np.ndarray:
    """Trasla il clip di uno shift casuale in [-TIME_SHIFT_MS, +TIME_SHIFT_MS] ms,
    riempiendo con zeri i campioni scoperti."""
    max_shift = int(SAMPLE_RATE * TIME_SHIFT_MS / 1000)
    shift = rng.randint(-max_shift, max_shift + 1)
    shifted = np.zeros_like(audio)
    if shift > 0:
        shifted[shift:] = audio[:CLIP_LEN - shift]
    elif shift < 0:
        shifted[:CLIP_LEN + shift] = audio[-shift:]
    else:
        shifted[:] = audio
    return shifted


def mix_background_noise(audio: np.ndarray, noise_files) -> np.ndarray:
    """Mixa nel clip un frammento di rumore di fondo a basso volume, con
    probabilità BACKGROUND_FREQUENCY."""
    if not noise_files or rng.rand() > BACKGROUND_FREQUENCY:
        return audio
    noise_path = noise_files[rng.randint(len(noise_files))]
    noise, _ = librosa.load(noise_path, sr=SAMPLE_RATE, mono=True)
    if len(noise) <= CLIP_LEN:
        noise = np.pad(noise, (0, CLIP_LEN - len(noise) + 1))
    start = rng.randint(0, len(noise) - CLIP_LEN)
    noise_clip = noise[start:start + CLIP_LEN]
    volume = rng.uniform(0, BACKGROUND_VOLUME_MAX)
    return audio + volume * noise_clip


# ---------------------------------------------------------------------------
# MFCC
# ---------------------------------------------------------------------------


def compute_mfcc(audio: np.ndarray) -> np.ndarray:
    """Estrae 10 MFCC su finestre di 40ms/stride 20ms -> matrice (49, 10).
    center=False per ottenere esattamente N_FRAMES senza padding ai bordi."""
    mfcc = librosa.feature.mfcc(
        y=audio.astype(np.float32),
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        window="hann",
        center=False,
    )
    return mfcc.T  # (N_FRAMES, N_MFCC)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def build_split(split_name: str, noise_files):
    csv_path = SPLITS_DIR / f"{split_name}.csv"
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    X = np.zeros((len(rows), N_FRAMES, N_MFCC), dtype=np.float32)
    y = np.zeros(len(rows), dtype=np.int64)

    for i, row in enumerate(rows):
        audio = load_clip(row["filepath"])
        if split_name == "train":
            audio = time_shift(audio)
            audio = mix_background_noise(audio, noise_files)
        X[i] = compute_mfcc(audio)
        y[i] = CLASS_TO_IDX[row["label"]]
        if (i + 1) % 500 == 0:
            print(f"  {split_name}: {i + 1}/{len(rows)}")

    return X, y


def main():
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    noise_files = sorted(NOISE_DIR.glob("*.wav")) if NOISE_DIR.exists() else []

    for split_name in ("train", "val", "test"):
        print(f"Estrazione feature: {split_name}")
        X, y = build_split(split_name, noise_files)
        out_path = OUTPUT_DIR / f"{split_name}.npz"
        np.savez_compressed(out_path, X=X, y=y)
        print(f"  -> salvato {out_path}  (X: {X.shape}, y: {y.shape})")

    with open(OUTPUT_DIR / "classes.json", "w") as f:
        json.dump(CLASSES, f, indent=2, ensure_ascii=False)

    print("\nFatto. Classi (indice -> nome):")
    for i, c in enumerate(CLASSES):
        print(f"  {i:2d}: {c}")


if __name__ == "__main__":
    main()
