
# In questo script andiamo a realizzare l'architettura della nostra rete, con numero di parametri 
# e dimensione definiti in Zhang et al., "Hello Edge: Keyword Spotting on Microcontrollers" in tabella 7.

import numpy as np
from tensorflow.keras import layers, models

# L'input atteso ha lo shape pari alle feature estratte negli step precedenti, con 51 frame temporali, 10 coefficienti MFCC
# ed 1 canale. L'otput sarà una softmax su 12 classi (10 parole + unknown + silence).
INPUT_SHAPE = (51, 10, 1)
NUM_CLASSES = 12
LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "unknown", "silence"]

# Definiamo di seguito la struttura della nostra DS-CNN, che sarà composta da 4 blocchi depthwise, ed una parte iniziale e finale.
# La rete sarà poi utilizzabile da un altro script esterno.
def build_ds_cnn(
  input_shape,
  num_classes: int,
  num_filters: int = 64,
  n_blocks: int = 4,
  first_kernel=(10, 4),
  first_strides=(2, 2),
  dropout: float = 0.2,
  name: str = "ds_cnn_s",
):
  inputs = layers.Input(shape=input_shape, name="mfcc_input")
  # Primo layer: convoluzione "normale" (non depthwise-separable)
  x = layers.Conv2D(num_filters, first_kernel, strides=first_strides, padding="same")(inputs)
  x = layers.BatchNormalization()(x)
  x = layers.ReLU()(x)
  # Blocchi depthwise-separable: il cuore della DS-CNN, quello che la
  # rende leggera rispetto a una CNN "piena" equivalente
  
  for i in range(n_blocks):
    x = layers.DepthwiseConv2D((3, 3), padding="same", name=f"dw_conv_{i}")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(num_filters, (1, 1), padding="same", name=f"pw_conv_{i}")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
  x = layers.GlobalAveragePooling2D()(x)
  if dropout > 0:
    x = layers.Dropout(dropout)(x)
  outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
  return models.Model(inputs, outputs, name=name)


if __name__ == "__main__":
  model = build_ds_cnn(INPUT_SHAPE, NUM_CLASSES)
  model.summary()

  n_params = model.count_params()
  print(f"\nParametri totali: {n_params:,}")
  print(f"Dimensione stimata (float32): {n_params * 4 / 1024:.1f} KB")
  print("(confronto: DS-CNN-S in Tab. 7 della Fonte 1 ~ 98.7 KB dopo TFLite Int8)")

  # Forward pass su un batch finto, solo per controllare la shape di output
  dummy_batch = np.random.randn(8, *INPUT_SHAPE).astype("float32")
  preds = model.predict(dummy_batch, verbose=0)
  print(f"\nShape output su batch finto di 8 esempi: {preds.shape}")
  print(f"Somma delle probabilita' per il primo esempio (deve fare ~1.0): {preds[0].sum():.4f}")
