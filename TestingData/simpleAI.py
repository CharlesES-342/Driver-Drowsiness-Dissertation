# used to generate and test a simple AI model for driver drowsiness detection
# this was my first example and created "drowsiness_model_V1.h5"
# an example useage can be found in 'AI/initialAI.py'

import tensorflow as tf
import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# ---------- ASPECT RATIO FUNCTIONS ---------- #

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[2] - mouth[9])
    B = np.linalg.norm(mouth[4] - mouth[7])
    C = np.linalg.norm(mouth[0] - mouth[6])
    return (A + B) / (2.0 * C)

# ---------- HEAD POSE (APPROX) ---------- #

def extract_head_pose(landmarks):
    nose = landmarks[1]     # Nose tip
    chin = landmarks[152]   # Chin
    left_eye = landmarks[33]  
    right_eye = landmarks[263]

    pitch = np.arctan2(chin[1] - nose[1], chin[0] - nose[0]) * 180 / np.pi
    yaw = np.arctan2(right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]) * 180 / np.pi

    return pitch, yaw

# ---------- EXTRACT FEATURES FROM FRAME ---------- #

def extract_features(frame):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3, #addjusted from 0.5
        min_tracking_confidence=0.3 #addjusted from 0.5
    ) as face_mesh:

        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        lm = results.multi_face_landmarks[0]
        pts = np.array([(p.x * w, p.y * h) for p in lm.landmark])

        # Eye + mouth landmarks (MediaPipe indexes)
        left_eye_idx = [33, 160, 158, 133, 153, 144]
        right_eye_idx = [263, 387, 385, 362, 380, 373]
        mouth_idx = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375]

        left_eye = pts[left_eye_idx]
        right_eye = pts[right_eye_idx]
        mouth = pts[mouth_idx]

        ear_left = eye_aspect_ratio(left_eye)
        ear_right = eye_aspect_ratio(right_eye)
        mar = mouth_aspect_ratio(mouth)
        pitch, yaw = extract_head_pose(pts)

        return np.array([ear_left, ear_right, mar, pitch, yaw])

# ---------- LOAD TRAINING DATA ---------- #

def load_training_data(data_dir, label_map):
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

                features = extract_features(frame)
                if features is not None:
                    X.append(features)
                    y.append(label_idx)

            cap.release()
    
    print(f"\nTotal samples collected: {len(X)}")
    return np.array(X), np.array(y)

# ---------- MODEL ---------- #

def build_model(input_shape):
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
    X, y = load_training_data(data_dir, label_map)
    print("X shape:", X.shape) #for debugging
    model = build_model(X.shape[1])
    model.fit(X, y, epochs=epochs, batch_size=32, validation_split=0.2)
    model.save("drowsiness_model.h5")
    return model

# ---------- TEST ON VIDEO ---------- #

def test_on_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    predictions = []
    confidences = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        features = extract_features(frame)
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

