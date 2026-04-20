'''
used to generate and test a simple AI model for driver drowsiness detection
this was my first example and created "drowsiness_model_V1.h5"
an example useage can be found in 'AI/initialAI.py'
'''

import tensorflow as tf
import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp
import utilities.imageProcessing as imageProcessing

mp_face_mesh = mp.solutions.face_mesh

# ---------- LOAD TRAINING DATA ---------- #

def load_training_data(data_dir, label_map):
    '''
    generate features from a video (extracting every frame)
    Input: data locaiton, label map (for the desired features)
    Output: DIR{features extracted}, list[classificaitons] -> both index links to eachother
    '''
    X, y = [], []

    for label, label_idx in label_map.items():
        video_dir = Path(data_dir) / label
        print(f"\nSearching in: {video_dir}") #for debugging
        
        print(f"Files inside folder: {list(video_dir.iterdir())}")

        for video_file in video_dir.glob("*.mp4"):
            print(f"Processing video: {video_file}") #for debugging
            cap = cv2.VideoCapture(str(video_file))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                features = imageProcessing.extract_features(frame)
                if features is not None:
                    X.append(features)
                    y.append(label_idx)

            cap.release()
    
    print(f"\nTotal samples collected: {len(X)}")
    return np.array(X), np.array(y)

# ---------- MODEL ---------- #

def build_model(input_shape):
    '''
    Construct a model based on the number of feature atributes given
    Input: Input level shape (input structure size)
    Output: AI model
    '''
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(data_dir, label_map, epochs=20):
    '''
    Training the model provided (the model could be un-trained or trained)
    Input: data directory, label map for classification meaning (usually 0 =Alert, 1=Tired),
    number of epochs (default 20)
    Output: trained model
    '''
    X, y = load_training_data(data_dir, label_map)
    print("X shape:", X.shape) #for debugging
    model = build_model(X.shape[1])
    model.fit(X, y, epochs=epochs, batch_size=32, validation_split=0.2)
    model.save("SUST_Simple_model.h5")
    return model

# ---------- TEST ON VIDEO ---------- #

def test_on_video(video_path, model):
    '''
    Testing the model on Video input
    Input: path to video, model
    output: none -> Terminal output for model confidence and overall
    classificaont across all fromaes in the video (avg)
    '''
    cap = cv2.VideoCapture(video_path)
    predictions = []
    confidences = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        features = imageProcessing.extract_features(frame)
        if features is not None:
            pred = model.predict(features.reshape(1, -1), verbose=0)[0]
            class_id = np.argmax(pred)
            confidence = pred[class_id]

            predictions.append(class_id)
            confidences.append(confidence)

    cap.release()

    if not predictions:
        print("No face detected in any frame.")
        return

    # Majority vote classification
    final_class = max(set(predictions), key=predictions.count)

    # Average confidence
    avg_confidence = np.mean(confidences) * 100
    
    # Percentage of frames classified tired
    tired_percentage = (predictions.count(1) / len(predictions)) * 100

    print("\n===== TEST RESULTS =====")
    print(f"Final Classification: {'Tired' if final_class == 1 else 'Not Tired'}")
    print(f"Model Confidence: {avg_confidence:.2f}%")
    print(f"Tired frames: {tired_percentage:.2f}%")
    print("========================\n")



# Example usage:
label_map = {"Not-Tired": 0, "Tired": 1}
train_model(r"C:/Users/smelt/OneDrive/Documents/Uni/Year_3/Dissertation/Git_Repo_Dest/Driver-Drowsiness-Dissertation/TrainingData", label_map) #TODO :insert route to training data
model = tf.keras.models.load_model("drowsiness_model.h5")
test_on_video(r"C:\Users\smelt\OneDrive\Documents\Uni\Year_3\Dissertation\Git_Repo_Dest\TrainingData\Not-Tired\SUST_Driver_Drowsiness_Dataset\n_13.mp4", model)

