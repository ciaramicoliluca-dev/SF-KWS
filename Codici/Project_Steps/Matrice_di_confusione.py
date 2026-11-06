import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Percorsi dei file (ho usato i percorsi dal tuo output)
VAL_DATA_PATH = r'C:\Users\ciara\OneDrive - Università Politecnica delle Marche\Magistrale\2° Anno\Da fare\Digital Adaptive Circuits and Learning Systems\Features\val_features.npz'
MODEL_PATH = 'ds_cnn_s_best.keras'

def plot_confusion_matrix():
    print("Caricamento dati di validazione...")
    # 2. Caricamento dei dati .npz
    with np.load(VAL_DATA_PATH) as data:
        # print("I nomi degli array salvati sono:", data.files)
        # NOTA: Sostituisci 'x' e 'y' con le chiavi reali usate nel tuo .npz 
        # (se non le sai, aggiungi print(data.files) per vederle, spesso sono 'arr_0' e 'arr_1')
        X_val = data['features']  
        y_val = data['labels']  

    # Se le tue label sono in formato "one-hot" (shape 4443, 12), le convertiamo in indici numerici singoli
    if len(y_val.shape) > 1 and y_val.shape[1] > 1:
        y_true = np.argmax(y_val, axis=1)
    else:
        y_true = y_val

    # 3. Caricamento del modello
    print("Caricamento del modello...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # 4. Definizione delle classi (spostata qui in alto)
    class_names = [
        'yes', 'no', 'up', 'down', 'left', 'right', 
        'on', 'off', 'stop', 'go', 'unknown', 'silence'
    ]

    # Se y_true contiene stringhe di testo, lo convertiamo in indici numerici
    if isinstance(y_true[0], (str, np.str_)):
        class_to_idx = {name: i for i, name in enumerate(class_names)}
        y_true = np.array([class_to_idx[label] for label in y_true])

    # 5. Generazione delle predizioni
    print("Generazione predizioni...")
    y_pred_probs = model.predict(X_val)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 6. Calcolo e plot della matrice
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    
    plt.title('Matrice di Confusione - Validation Set')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_confusion_matrix()