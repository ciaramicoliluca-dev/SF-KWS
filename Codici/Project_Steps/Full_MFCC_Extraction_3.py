
# Formato CSV atteso:
#     filepath,label
#     Comandi_Speech\\right\\5b1db3ee_nohash_0.wav,right
#     Comandi_Speech\\_background_noise_\\pink_noise.wav#silence,silence
# Uso:
# python step2_extract_dataset.py --csv train.csv --data-root "C:/.../cartella_che_contiene_Comandi_Speech" --out train_features.npz

# Lo script punta ad estrarre le feature MFCC dei file audio indicati in un CSV per poi salvarli in un unico file .npz.
# Per i file del rumore di fondo (silence), abbiamo deciso di ritagliare un secondo casuale piuttosto che l'intero file, per renderlo coerente.
# Per ottere i file mfcc utilizzimo la libreria speechbrain, che permette di calcolare le feature facilmente ed in modo efficiente.

#**************************************************************************************************************************#
#**************************************************************************************************************************#
# IMPORTANTE!!!! Lo script lavora per singolo file csv, quindi l'operazione va ripetuta per ogni file csv (train,val,test) #
#**************************************************************************************************************************#
#**************************************************************************************************************************#

# import argparse
import os

import numpy as np
import soundfile as sf
import torch
from speechbrain.lobes.features import MFCC

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

#*****************************#
# PARAMETRI DI CONFIGURAZIONE #
#*****************************#
CSV_DIR   = r"C:\Users\ciara\OneDrive - Università Politecnica delle Marche\Magistrale\2° Anno\Da fare\Digital Adaptive Circuits and Learning Systems\splits"           # Cartella con i csv
DATA_ROOT = r"C:\Users\ciara\OneDrive - Università Politecnica delle Marche\Magistrale\2° Anno\Da fare\Digital Adaptive Circuits and Learning Systems"                  # Cartella che contiene Comandi_Speech
CSV_NAME  = "test.csv"                                                                                                                                                 # Quale csv vogliamo usare
OUT_FILE  = "test_features.npz"                                                                                                                                        # Quale file output vogliamo generare
SEED = 42                                                                                                                                                               # Seed casuale per la riproducibilita' dei ritagli di rumore        
SAMPLE_RATE = 16000
N_SAMPLES = 16000  # 1 secondo


#**********************#
# FUNZIONI DI UTILITA' #
#**********************#

# Per cambiare il doppio \\ con /, in caso lo script venga eseguito su sistema unix
def to_local_path(rel_path: str) -> str:
    return rel_path.replace("\\", os.sep).replace("/", os.sep)

# Carica una clip partendo dalla riga del csv, gestendo sia parole normali sia silence, e nel caso sia silence ritaglia un secondo casuale.
def load_clip(data_root: str, filepath: str, rng: np.random.Generator) -> np.ndarray:
    """Carica un clip a partire dalla riga del CSV, gestendo sia le parole
    normali sia il caso speciale '#silence'."""
    if "#silence" in filepath:
        real_path, _ = filepath.split("#silence")
        full_path = os.path.join(data_root, to_local_path(real_path))
        noise, sr = sf.read(full_path, dtype="float32")
        if sr != SAMPLE_RATE:
            raise ValueError(f"{full_path}: sample rate {sr} != {SAMPLE_RATE}")
        if len(noise) <= N_SAMPLES:
            noise = np.tile(noise, N_SAMPLES // len(noise) + 1)
        start = rng.integers(0, len(noise) - N_SAMPLES)
        clip = noise[start : start + N_SAMPLES]
    else:
        full_path = os.path.join(data_root, to_local_path(filepath))
        clip, sr = sf.read(full_path, dtype="float32")
        if sr != SAMPLE_RATE:
            raise ValueError(f"{full_path}: sample rate {sr} != {SAMPLE_RATE}")
        if len(clip) < N_SAMPLES:
            clip = np.pad(clip, (0, N_SAMPLES - len(clip)))
        elif len(clip) > N_SAMPLES:
            clip = clip[:N_SAMPLES]

    return clip.astype(np.float32)

#********#
#  MAIN  #
#********#

def main():
    # ap = argparse.ArgumentParser()
    # ap.add_argument("--csv", required=True, help="es. train.csv")
    # ap.add_argument("--data-root", required=True, help="cartella che CONTIENE Comandi_Speech")
    # ap.add_argument("--out", required=True, help="es. train_features.npz")
    # ap.add_argument("--seed", type=int, default=42, help="per i ritagli casuali del rumore, riproducibilita'")
    # args = ap.parse_args()

    csv_path = os.path.join(CSV_DIR, CSV_NAME)

    # Inizializziamo l'estrattore MFCC di SpeechBrain
    mfcc_extractor = MFCC(
        deltas=False,
        context=False,
        sample_rate=SAMPLE_RATE,
        n_fft=1024,         # potenza di 2
        n_mels=40,          # numero di filtri Mel
        n_mfcc=10,          # numero di coefficienti MFCC in uscita
        win_length=40,      # finestra di analisi, in ms
        hop_length=20,      # passo tra una finestra e l'altra, in ms
    )
    mfcc_extractor.eval()

    import csv as csv_module

    # apriamo il csv e leggiamo le righe
    with open(csv_path, newline="") as f:
        rows = list(csv_module.DictReader(f))
    print(f"Righe da processare: {len(rows)}")

    # rng = np.random.default_rng(args.seed)
    rng = np.random.default_rng(SEED)
    features_list = []
    labels_list = []

    # per ogni riga nel csv, carichiamo il clip audio corrispondente ed estraiamo e salviamo le feature MFCC.
    for row in tqdm(rows):
        # clip = load_clip(args.data_root, row["filepath"], rng)
        clip = load_clip(DATA_ROOT, row["filepath"], rng)
        waveform = torch.from_numpy(clip).unsqueeze(0)
        with torch.no_grad():
            feats = mfcc_extractor(waveform).squeeze(0).numpy()  # [T, n_mfcc]
        features_list.append(feats)
        labels_list.append(row["label"])

    X = np.stack(features_list)          # [N, T, n_mfcc]
    y = np.array(labels_list)            # [N], stringhe

    print(f"Shape finale delle feature: {X.shape}")
    print(f"Numero di etichette uniche: {len(set(labels_list))} -> {sorted(set(labels_list))}")
    # Infine salviamo in un unico file .npz compresso.
    np.savez_compressed(OUT_FILE, features=X, labels=y)
    print(f"Salvato in {OUT_FILE}")


if __name__ == "__main__":
    main()
