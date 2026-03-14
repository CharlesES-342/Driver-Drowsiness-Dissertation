import tensorflow as tf
import numpy as np
from pathlib import Path
import sys

def convert_model_to_tflite(keras_model_path):
    # Convert path to string for Keras compatibility
    keras_model_path = str(keras_model_path)
    
    print(f"Loading model from: {keras_model_path}")
    model = tf.keras.models.load_model(keras_model_path)

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # --- OPTIMIZATION FOR PI ---
    # This reduces size and increases speed on ARM processors
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()

    # Create the new filename
    new_model_path = Path(keras_model_path).with_suffix('.tflite')

    # Save the TFLite model
    with open(new_model_path, "wb") as f:
        f.write(tflite_model)

    print(f"✅ Conversion complete! Saved to: {new_model_path}")

if __name__ == "__main__":
    # Get the directory where this script is located
    SCRIPT_DIR = Path(__file__).resolve().parent
    
    # Go up one level to 'Driver-Drowsiness-Dissertation'
    PROJECT_ROOT = SCRIPT_DIR.parent 
    
    model_name = "xception_drowsiness_model_v6.h5"
    model_location = PROJECT_ROOT / "AI_Models" / model_name

    if not model_location.exists():
        print(f"❌ ERROR: Cannot find {model_location}")
        # List files to help debug
        if model_location.parent.exists():
            print(f"Files in AI_Models: {list(model_location.parent.glob('*.h5'))}")
    else:
        convert_model_to_tflite(model_location)