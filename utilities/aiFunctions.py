'''
Outline functions that will be used thorught the development process for a variety of models.
'''
import tensorflow as tf
from pathlib import Path

def convert_model_to_tflite(keras_model_path):
    '''
    Convert the model from a '.h5' file to a '.tflite' model for use on the Raspberry Pi
    This was later adapted further to the 'model_conversions.py' file as some runtime envirmnments
    were changed to allow for other Libraries which were only compatable with older versions.
    Input: model path
    Output: none -> model saved as different file type (same name)
    '''
    # Convert path to string for Keras compatibility
    keras_model_path_str = str(keras_model_path)
    
    print(f"Loading H5 model from: {keras_model_path_str}")
    
    # 1. Load the Keras model (H5 format)
    model = tf.keras.models.load_model(keras_model_path_str)

    # 2. Use the Keras-specific converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # --- PI 2.14.0 COMPATIBILITY & OPTIMIZATION ---
    # This prevents 'Op Version 12' errors on the Pi
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    
    # Enable optimization to make the Xception model faster on the Pi's CPU
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Crucial for older runtimes like 2.14.0
    converter._experimental_lower_tensor_list_ops = True

    print("Converting... Xception models can take a moment to optimize.")
    tflite_model = converter.convert()

    # 3. Save the new file in the same AI_Models folder
    new_model_path = keras_model_path.with_suffix('.tflite')
    
    with open(new_model_path, "wb") as f:
        f.write(tflite_model)

    print(f"✅ Conversion complete! Saved to: {new_model_path}")

if __name__ == "__main__":
    # Get the directory where this script is located (On your Lenovo laptop)
    SCRIPT_DIR = Path(__file__).resolve().parent
    
    # Navigate to 'Driver-Drowsiness-Dissertation'
    PROJECT_ROOT = SCRIPT_DIR.parent 
    
    model_name = "xception_drowsiness_model_v6.h5"
    model_location = PROJECT_ROOT / "AI_Models" / model_name

    if not model_location.exists():
        print(f"❌ ERROR: Cannot find {model_location}")
        if model_location.parent.exists():
            print(f"Files in AI_Models: {list(model_location.parent.glob('*.h5'))}")
    else:
        convert_model_to_tflite(model_location)