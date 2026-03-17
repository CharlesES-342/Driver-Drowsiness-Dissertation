import tensorflow as tf
from pathlib import Path
import os

# Get the directory where new.py is located
BASE_DIR = Path(__file__).resolve().parent

# Combine the directory with the filename
model_path = os.path.join(BASE_DIR, 'xception_drowsiness_model_v6.h5')

print(f"Loading model from: {model_path}")

# Load the model using the full path
model = tf.keras.models.load_model(model_path)

# --- PI 2.14.0 COMPATIBILITY SETTINGS ---
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter._experimental_lower_tensor_list_ops = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]

print("Converting... this will take a moment.")
tflite_model = converter.convert()

# Save the TFLite file in the same folder
output_path = os.path.join(BASE_DIR, 'xception_v6_pi.tflite')
with open(output_path, 'wb') as f:
    f.write(tflite_model)

print(f"✅ Success! Saved to: {output_path}")