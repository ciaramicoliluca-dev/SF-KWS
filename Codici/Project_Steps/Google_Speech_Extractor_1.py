
# Lo script in questione viene usato per estrarre il dataset Google Speech Commands in una cartella.
# Il dataset scelto è la v0.02, che contiene 105.829 file audio in formato WAV, ciascuno della durata di 1 secondo e con sample rate di 16 kHz.
# Unica differenza è per i file "silence" che sono più lunghi di 1 secondo, ma vengono comunque considerati validi.
# Per utilizzarlo al meglio, viene modificata il path DEST_DIR in modo da puntare alla cartella desiderata sul proprio PC.

import os
import ssl
import tarfile
import urllib.request
import certifi
 
# Usa il bundle di certificati di certifi invece dello store di Windows,
# che su alcuni PC restituisce dati certificato corrotti a causa di antivirus/VPN
_ssl_context = ssl.create_default_context(cafile=certifi.where())
urllib.request.install_opener(
    urllib.request.build_opener(urllib.request.HTTPSHandler(context=_ssl_context))
)

#*******************************************#
# Selezione della cartella di destinazione  #
#*******************************************#
DEST_DIR = r"C:\Users\ciara\OneDrive - Università Politecnica delle Marche\Magistrale\2° Anno\Da fare\Digital Adaptive Circuits and Learning Systems\Comandi_Speech"

# URL a cui puntare per scaricare il dataset in locale
DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"


#*************************************#
# Barra di progressione del download  #
#*************************************#

# Andiamo ad inserire una semplice barra di progresso del download per rencerci conto del funzionamento ed avanzamento
def download_progress(block_num, block_size, total_size):
    """Mostra una barra di progresso testuale durante il download."""
    downloaded = block_num * block_size
    percent = min(downloaded / total_size * 100, 100) if total_size > 0 else 0
    print(f"\rDownload in corso: {percent:5.1f}%", end="", flush=True)


#********************#
#  Script effettivo  #
#********************#

def main():
    # Verifichiamo l'esistenza della cartella di destinazione, e se non esiste la creiamo
    os.makedirs(DEST_DIR, exist_ok=True)

    # Verifichiamo l'esistenza dell'archivio, e se non esiste lo scarichiamo
    # Se l'archivio esiste già, non lo scarichiamo nuovamente
    archive_path = os.path.join(DEST_DIR, "speech_commands_v0.02.tar.gz")
    if os.path.exists(archive_path):
        print(f"L'archivio esiste già in: {archive_path}")
    else:
        print(f"Scarico il dataset in: {archive_path}")
        urllib.request.urlretrieve(DATA_URL, archive_path, reporthook=download_progress)
        print("\nDownload completato.")

    # Creiamo la cartella di destinazione se non esiste ed iniziamo ad estrarre l'archivio 
    print(f"Estraggo l'archivio in: {DEST_DIR}")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=DEST_DIR)
    print("Estrazione completata.")

    # Stampiamo a video le cartelle keyword trovate e il numero di file in ciascuna
    print("\nCartelle keyword trovate:")
    for entry in sorted(os.listdir(DEST_DIR)):
        full_path = os.path.join(DEST_DIR, entry)
        if os.path.isdir(full_path):
            n_files = len(os.listdir(full_path))
            print(f"  {entry:20s} -> {n_files} file")

    # al termine dell'estrazione, stampiamo a video il path della cartella di destinazione
    print(f"\nDataset pronto in: {DEST_DIR}")

if __name__ == "__main__":
    main()
