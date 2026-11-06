I file corrispondono ai seguenti step:



1. Google\_Speech\_Extractor: Download ed estrazione del dataset GSCD V0.02 in una cartella dedicata;
2. Speech\_CSV\_Divider: Divisione del dataset mediante il file di testo presente nel GSCD in Train/Validation/Test in 3 file CSV dedicati, contenente la classe di riferimento e la label; Inoltre, nel file Single\_MFCC\_extraction\_2b viene testata l'estrazione delle features MFCC di un singolo file audio e visualizzazione dello spettrogramma risultante dei coefficienti;
3. Full\_MFCC\_Extraction: Estrazione dei coefficienti MFCC di una singola suddivisione di tr/val/tst e salvataggio dei parametri in formato npz. Inoltre, in MFCC\_Spectrogram\_Generator\_3b otteniamo un'immagine dai file NPZ per la visualizzazione dello spettrogramma per ogni classe;
4. DS\_CNN\_Model: Creazione dell'architettura di rete, basata su una semplificazione della MobileNet proposta nel paper "Hello Edge: Keyword Spotting on Microcontrollers" espresso in tabella 7, risultante in 38.6KB e 5.4M di operazioni.

