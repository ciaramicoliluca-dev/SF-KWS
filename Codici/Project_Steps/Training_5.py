
# In questo script andiamo ad eseguire l'addestramento della rete DS-CNN, sfruttando i parametri
# indicati in Hello Edge, in cui viene utilizzato Adam per ottimizzare, un batch size di 100 e 20000 iterazioni totali.
# Il learning rate viene ridotto a metà addestramento per ridurre l'overfitting.
# Essendo il batch size abbastanza elevato, non viene effettuata data agumentation.
# Il file di test non viene toccato qua, ma lo useremo poi per la validation on target.


import json
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Ci carichiamo la rete DSCNN
from DS_CNN_Model_4 import build_ds_cnn, INPUT_SHAPE, NUM_CLASSES, LABELS

# ================
# CONFIGURAZIONE 
# ================
NPZ_DIR = r"C:\Users\ciara\OneDrive - Università Politecnica delle Marche\Magistrale\2° Anno\Da fare\Digital Adaptive Circuits and Learning Systems\Features"   # cartella che CONTIENE i file .npz (separata dal codice)
NPZ_TRAIN = "train_features.npz"
NPZ_VAL = "val_features.npz"
MODEL_OUT = "ds_cnn_s_best.keras"
HISTORY_OUT = "training_history.json"

BATCH_SIZE = 100           
TOTAL_ITERATIONS = 20000   
LR_INITIAL = 5e-4          # (prima meta' delle iterazioni)
LR_AFTER_HALF = 1e-4       # (seconda meta' delle iterazioni)

SEED = 42
# ============================================================


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_split(npz_path, label_to_idx):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"Non trovo '{npz_path}'. Va generato con step2_extract_dataset.py "
            f"(imposta CSV_NAME e OUT_FILE di conseguenza) prima di lanciare il training."
        )
    data = np.load(npz_path)
    X, y_str = data["features"], data["labels"]

    unknown = set(y_str) - set(label_to_idx)
    if unknown:
        raise ValueError(f"Etichette in '{npz_path}' non previste da LABELS: {unknown}")

    y = np.array([label_to_idx[l] for l in y_str], dtype=np.int64)
    X = X[..., np.newaxis].astype(np.float32)  # (N, 51, 10) -> (N, 51, 10, 1)
    return X, y


def print_class_distribution(y, labels, title):
    print(f"\nDistribuzione classi ({title}):")
    counts = np.bincount(y, minlength=len(labels))
    for lbl, c in zip(labels, counts):
        print(f"  {lbl:10s}: {c}")


def main():
    set_seeds(SEED)

    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPU rilevate: {len(gpus)} -> {[g.name for g in gpus]}")

    label_to_idx = {lbl: i for i, lbl in enumerate(LABELS)}

    train_path = os.path.join(NPZ_DIR, NPZ_TRAIN)
    val_path = os.path.join(NPZ_DIR, NPZ_VAL)

    print(f"\nCarico training set da '{train_path}'...")
    X_train, y_train = load_split(train_path, label_to_idx)
    print(f"  Shape: {X_train.shape}, esempi: {len(y_train)}")
    print_class_distribution(y_train, LABELS, "train")

    print(f"\nCarico validation set da '{val_path}'...")
    X_val, y_val = load_split(val_path, label_to_idx)
    print(f"  Shape: {X_val.shape}, esempi: {len(y_val)}")
    print_class_distribution(y_val, LABELS, "val")

    # Traduzione iterazioni -> epoche, in base alla dimensione REALE del train set
    n_train = len(y_train)
    total_epochs = round(TOTAL_ITERATIONS * BATCH_SIZE / n_train)
    half_epoch = total_epochs // 2
    print(
        f"\n{TOTAL_ITERATIONS} iterazioni x batch {BATCH_SIZE} / {n_train} esempi "
        f"-> {total_epochs} epoche (drop LR a epoca {half_epoch})"
    )

    def lr_schedule(epoch, lr):
        return LR_INITIAL if epoch < half_epoch else LR_AFTER_HALF

    model = build_ds_cnn(INPUT_SHAPE, NUM_CLASSES)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR_INITIAL),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1),
        keras.callbacks.ModelCheckpoint(
            MODEL_OUT, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=total_epochs,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )

    with open(HISTORY_OUT, "w") as f:
        # json.dump(history.history, f, indent=2)
        # Converte tutti i numpy.float32 in classici float di Python
        history_dict = {key: [float(val) for val in value] for key, value in history.history.items()}
        json.dump(history_dict, f, indent=2)

    best_val_acc = max(history.history["val_accuracy"])
    print(f"\nMiglior val_accuracy raggiunta: {best_val_acc:.4f}")
    print(f"Modello migliore salvato in: {MODEL_OUT}")
    print(f"Storia del training salvata in: {HISTORY_OUT}")
    print(
        "\nNota: test_features.npz non e' stato toccato in questo script -- "
        "va usato solo a valle, per la valutazione finale del modello scelto."
    )


if __name__ == "__main__":
    main()
