

# Lo script consiste nella suddivisione del dataset GSCD in Train, validation e test,
# per poter addestrare la rete DS-CNN e bilanciare le classi unknown e silence.
# Le classi sono state suddivise in 10 parole target, più unknown e silence, per un totale di 12 classi.
# Lo split ufficiale è fornito da google mediante validation_list.txt e testing_list.txt.
# Lo script fornisce in uscita tre file csv, con colonne suddivise in path e label per la successiva
# estrazione delel feature MFCC.

# Inoltre, il dataset effettua un bilanciamento delle classi unknown e silence,
# basato sulla dimensione media delle 10 classi target, per evitare che la classe unknown contenga troppi elementi.
# La classe silence è stata invece generata ritagliando clip casuali di 1 secondo dai file 
# più lunghi presenti nella cartella _background_noise_.

import random
import csv
from pathlib import Path

#*********************************#
# Settaggio delle costanti e path #
#*********************************#

DATASET_ROOT = Path("./Comandi_Speech")
OUTPUT_DIR = Path("./splits")
SEED = 42 # La risposta al senso della vita, dell'universo e di tutto quanto (cit. Guida galattica per autostoppisti)

TARGET_WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
BACKGROUND_NOISE_DIR = "_background_noise_"

# "unknown" e "silence" vengono dimensionate come multiplo della dimensione
# media delle 10 classi target, per split,
# per evitare che "unknown" (che raggruppa ~25 parole extra) domini il training
UNKNOWN_FRACTION = 1.0
SILENCE_FRACTION = 1.0

random.seed(SEED)

#*******************#
# Split del dataset #
#*******************#

# Apriamo i file validation_list.txt e testing_list.txt per leggere i path dei file audio
# assegnati ai due blocchi, e li salviamo per accedervi successivamente.
# Il restante rientra nel training. La suddivisione viene usata per evitare lo stesso speaker
# in training e test, evidando data leakage.
def load_split_lists(root: Path):
    def read_list(name):
        with open(root / name) as f:
            return {line.strip() for line in f if line.strip()}

    return read_list("validation_list.txt"), read_list("testing_list.txt")

# La funzione restituisce la stringa relativa alla suddivisione di appartenenza del path del file audio.
def which_set(rel_path: str, val_set: set, test_set: set) -> str:
    if rel_path in val_set:
        return "val"
    if rel_path in test_set:
        return "test"
    return "train"


#***********************#
# Scansione del dataset #
#***********************#

# La funzione assegna ad ogni file wav del dataset una delle 12 classi escluso silence.
# Silence viene gestita a parte ritagliando le clip di 1 secondo.
# Il ritorno è una tupla, contenente filepath, label e split di appartenenza. 
def scan_dataset(root: Path, val_set: set, test_set: set):
    entries = []
    for word_dir in sorted(root.iterdir()):
        if not word_dir.is_dir() or word_dir.name == BACKGROUND_NOISE_DIR:
            continue
        label = word_dir.name if word_dir.name in TARGET_WORDS else "unknown"
        for wav_path in sorted(word_dir.glob("*.wav")):
            rel_path = f"{word_dir.name}/{wav_path.name}"
            split = which_set(rel_path, val_set, test_set)
            entries.append((str(wav_path), label, split))
    return entries

# La funzione calcola il numero medio di campioni per classe target, per poter poi bilanciare le classi unknown e silence.
def average_target_count(entries):
    avg = {}
    for split in ("train", "val", "test"):
        counts = [
            sum(1 for p, l, s in entries if l == w and s == split)
            for w in TARGET_WORDS
        ]
        avg[split] = sum(counts) / len(counts)
    return avg

# La funzione bilancia la classe unknown, sottocampionando a UNKNOWN_FRACTION x la dimensione media delle classi target, per split.
def balance_unknown(unknown_entries, avg_count):
    by_split = {"train": [], "val": [], "test": []}
    for e in unknown_entries:
        by_split[e[2]].append(e)

    kept = []
    for split, items in by_split.items():
        n_keep = int(avg_count[split] * UNKNOWN_FRACTION)
        random.shuffle(items)
        kept.extend(items[:n_keep])
    return kept

# Per la classe silence si ritagliano clip di 1 secondo, in modo randomico, 
# dai file audio più lunghi presenti nella cartella _background_noise_.
def generate_silence_entries(root: Path, avg_count):
    noise_files = sorted((root / BACKGROUND_NOISE_DIR).glob("*.wav"))
    if not noise_files:
        raise FileNotFoundError(
            f"Nessun file audio trovato in {root / BACKGROUND_NOISE_DIR}"
        )

    entries = []
    for split in ("train", "val", "test"):
        n_needed = int(avg_count[split] * SILENCE_FRACTION)
        for _ in range(n_needed):
            src = random.choice(noise_files)
            entries.append((f"{src}#silence", "silence", split))
    return entries


#**********#
#   MAIN   #
#**********#


def main():
    # Per prima cosa, andiamo a caricare il dataset, e poi a scansionarlo per calcolare
    # la dimensione media delle classi, e poi bilanciare unknown e silence.
    print(f"Scansione dataset in: {DATASET_ROOT}")
    val_set, test_set = load_split_lists(DATASET_ROOT)
    entries = scan_dataset(DATASET_ROOT, val_set, test_set)

    target_entries = [e for e in entries if e[1] != "unknown"]
    unknown_raw = [e for e in entries if e[1] == "unknown"]

    avg_count = average_target_count(target_entries)
    unknown_entries = balance_unknown(unknown_raw, avg_count)
    silence_entries = generate_silence_entries(DATASET_ROOT, avg_count)

    all_entries = target_entries + unknown_entries + silence_entries

    # Qua andiamo a scrivere i tre file csv, uno per ogni split, con le colonne filepath e label. 
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    for split in ("train", "val", "test"):
        rows = [(p, l) for p, l, s in all_entries if s == split]
        random.shuffle(rows)
        out_path = OUTPUT_DIR / f"{split}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filepath", "label"])
            writer.writerows(rows)
        print(f"{split:5s}: {len(rows):6d} campioni -> {out_path}")

    # Infine, stampiamo a video la distribuzione dei campioni per classe e split, per verificare che il bilanciamento sia stato effettuato correttamente.
    print("\nDistribuzione per classe e split:")
    all_labels = TARGET_WORDS + ["unknown", "silence"]
    print(f"{'classe':10s}" + "".join(f"{s:>10s}" for s in ("train", "val", "test")))
    for label in all_labels:
        row = f"{label:10s}"
        for split in ("train", "val", "test"):
            n = sum(1 for e in all_entries if e[1] == label and e[2] == split)
            row += f"{n:10d}"
        print(row)


if __name__ == "__main__":
    main()