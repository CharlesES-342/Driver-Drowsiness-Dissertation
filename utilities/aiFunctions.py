#going form tensor to tensor lite
import tensorflow as tf
import numpy as np

#converting my model into a Pi compatible tflite model
def convert_model_to_tflite(keras_model_path):
    # Load the Keras model (.h5)
    model = tf.keras.models.load_model(keras_model_path)

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    newModelName = keras_model_path.split(".h5")[0] + "_pi.tflite"

    # Save the TFLite model
    with open(newModelName, "wb") as f:
        f.write(tflite_model)

    print("Conversion complete!")

if __name__ == "__main__":
    convert_model_to_tflite("drowsiness_model_V2.h5")