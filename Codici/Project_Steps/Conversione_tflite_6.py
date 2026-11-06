import numpy as np
import tensorflow as tf

# Carichiamo il nostro modello della rete neurale
model = tf.keras.models.load_model("ds_cnn_s_best.keras")

# Convertiamo in tflite, mantenendo i calcoli a virgola mobile f32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model_float = converter.convert()
with open("ds_cnn_s.tflite", "wb") as f:
  f.write(tflite_model_float)
print("Modello ds_cnn_s.tflite (Float32) salvato!")


# Conversione int8
def representative_dataset_gen():
  # Carica un piccolo sottoinsieme di 100 valori di validation
  data = np.load(
      r"C:\Users\ciara\OneDrive - Università Politecnica delle Marche\Magistrale\2° Anno\Da fare\Digital Adaptive Circuits and Learning Systems\Features\val_features.npz"
  )
  X_val = data["features"][..., np.newaxis].astype(np.float32)
  for i in range(min(100, len(X_val))):
    yield [X_val[i : i + 1]]
# Il comando yeld ci permette di inviare i campioni uno alla volta, effettuando la fase pratica della calibrazione

# Con l'optimize default andiamo a prendere i 100 camppioni e registriamo i picchi massimi e minimi
converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = representative_dataset_gen

# Forziamo a int8 la conversione
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int8.inference_input_type = tf.int8
converter_int8.inference_output_type = tf.int8

# Convertiamo e salviamo
tflite_model_int8 = converter_int8.convert()

with open("ds_cnn_s_int8.tflite", "wb") as f:
  f.write(tflite_model_int8)
print("Modello ds_cnn_s_int8.tflite salvato!")