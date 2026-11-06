# SF-KWS: Keyword Spotting su STM32F446RE

Progetto per il corso di Digital Adaptive Circuits and Learning Systems (DACLS), a cura di Ciaramicoli e Marchetti.

Il progetto riconosce parole chiave pronunciate (keyword spotting) con una rete neurale che gira su un microcontrollore STM32F446RET6. La rete è una DS-CNN (Depthwise Separable CNN) presa dal paper "Hello Edge: Keyword Spotting on Microcontrollers" (tabella 7), addestrata sul Google Speech Commands Dataset V0.02 e poi quantizzata a int8 per stare nella memoria della scheda.

Le classi riconosciute sono 12: `yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`, più `unknown` (altre parole) e `silence`. In ingresso la rete riceve i coefficienti MFCC di un secondo di audio, in una matrice di dimensione (51, 10, 1).

La spiegazione completa del lavoro si trova nella tesina e nella presentazione, entrambe nella cartella principale.

## Struttura del repository

```
SF-KWS/
├── Codici/
│   ├── Project_Steps/      # script della pipeline, numerati per step (vedi il suo README)
│   └── Codici utili/       # materiale di partenza: notebook di esempio e funzioni per le features
├── splits/                 # train.csv, val.csv, test.csv con la suddivisione del dataset
├── STMCUBEMODEL/           # modelli convertiti, file per STM32Cube.AI Studio e risultati sulla scheda
│   ├── Heavier_Model/      # seconda versione del modello, più grande, con i suoi log
│   └── Risultati_fisso/    # workspace di STM32Cube.AI Studio con i run di analisi e validazione
├── Tesina DACLS Ciaramicoli-Marchetti.pdf
└── Presentazione DACLS Ciaramicoli - Marchetti.pptx
```

## Pipeline

Gli script in [Codici/Project_Steps](Codici/Project_Steps) vanno eseguiti in ordine. Il [README di quella cartella](Codici/Project_Steps/README.md) descrive ogni step nel dettaglio; in breve:

1. **Download del dataset**: scarica ed estrae il GSCD V0.02.
2. **Suddivisione**: divide i file audio in train, validation e test usando le liste fornite con il dataset, e salva i tre CSV in `splits/`.
3. **Estrazione delle features**: calcola i coefficienti MFCC per ogni split e li salva in file `.npz`.
4. **Modello**: definisce la DS-CNN (un primo strato convolutivo seguito da 4 blocchi depthwise + pointwise con 64 filtri, poi average pooling e strato denso).
5. **Addestramento**: Adam, batch size 100, 20000 iterazioni (54 epoche), learning rate da 5e-4 a 1e-4 a metà addestramento. La migliore accuratezza di validation è del 93,9%, raggiunta all'epoca 30.
6. **Conversione**: esporta il modello in TensorFlow Lite in float32 e in int8, calibrando la quantizzazione su 100 campioni di validation.

Gli script in [STMCUBEMODEL](STMCUBEMODEL) preparano poi i file per la validazione: `Test_Conversion_Labels.py` crea le etichette attese del test set in formato one-hot, mentre `gemini-code-1788196662614.py` misura l'accuratezza del modello Keras sul test set, da confrontare con quella ottenuta sulla scheda.

## Risultati sulla scheda

I modelli int8 sono stati analizzati e validati con STM32Cube.AI Studio (ST Edge AI Core v4.0.1) per l'STM32F446RET6. La validazione usa i 4888 campioni del test set, che non è mai stato usato durante l'addestramento.

| | Modello base | Heavier_Model |
|---|---|---|
| Parametri | 13.004 | 22.604 |
| Pesi (flash) | 13,67 KiB | 23,80 KiB |
| RAM per le attivazioni | 18,34 KiB | 18,34 KiB |
| MACC | 1.557.120 | 2.772.096 |
| Tempo di inferenza sulla scheda | 26,3 ms | 47,6 ms |
| Accuratezza sul test set | 92,37% (validazione su PC) | 93,94% (validazione sulla scheda, via seriale) |

Per il modello base l'accuratezza viene dalla validazione del modello int8 eseguita su PC, mentre il tempo di inferenza è misurato sulla scheda. Per l'Heavier_Model entrambi i valori sono misurati sulla scheda. In entrambi i casi la quantizzazione int8 riduce i pesi del 73% rispetto al modello float. Il modello più grande guadagna circa un punto e mezzo di accuratezza, ma impiega quasi il doppio del tempo per ogni inferenza.

I log completi sono in `STMCUBEMODEL/Risultati_fisso/SF-KWS` (modello base) e in `STMCUBEMODEL/Heavier_Model/SF-KWS_4_BL_run-5_logs.txt`.

## Come riprodurre il lavoro

Servono Python con TensorFlow, NumPy, scikit-learn e Matplotlib, e STM32Cube.AI Studio per la parte sul microcontrollore.

Gli script usano percorsi assoluti delle nostre macchine (per esempio la cartella delle features su OneDrive). Prima di eseguirli bisogna modificare le variabili di configurazione all'inizio di ogni file, come `NPZ_DIR` in `Training_5.py`, in modo che puntino alle cartelle locali.

Il dataset audio non è incluso nel repository e si scarica con lo step 1. Le features già estratte sono invece in `STMCUBEMODEL` (`train_features.npz`, `test_features.npz`, `val_features.npy` e i file `.npy` per la validazione), quindi si può partire direttamente dall'addestramento.
