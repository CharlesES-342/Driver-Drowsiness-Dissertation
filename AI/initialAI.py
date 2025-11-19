import tensorflow as tf
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

# ---------- GLOBAL VARIABLES ---------- #
mp_face_mesh = mp.solutions.face_mesh

# ---------- ASPECT RATIO FUNCTIONS / VALUE FUNCTIONS ---------- #
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

def extract_head_pose(landmarks):
    nose = landmarks[1]     # Nose tip
    chin = landmarks[152]   # Chin
    left_eye = landmarks[33]  
    right_eye = landmarks[263]

    pitch = np.arctan2(chin[1] - nose[1], chin[0] - nose[0]) * 180 / np.pi
    yaw = np.arctan2(right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]) * 180 / np.pi

    return pitch, yaw


# ---------- EXTRACTING FEATURES FROM FRAME ---------- #
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


# -------------------------------------------
# Run inference on a new video - GPT generated
# -------------------------------------------

def run_drowsiness_check(video_path, AI_model):

    # Load trained model
    model = AI_model
    cap = cv2.VideoCapture(video_path)
    predictions = []
    confidences = []

    print("\nRunning inference...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        features = extract_features(frame)
        if features is None:
            continue

        pred = model.predict(features.reshape(1, -1), verbose=0)[0]
        class_id = np.argmax(pred)
        confidence = pred[class_id]

        predictions.append(class_id)
        confidences.append(confidence)

    cap.release()

    if not predictions:
        print("No face detected in the video.")
        return

    final_class = max(set(predictions), key=predictions.count)
    avg_conf = np.mean(confidences) * 100
    tired_pct = (predictions.count(1) / len(predictions)) * 100

    print("\n========= RESULTS =========")
    print(f"Final Classification : {'Tired' if final_class == 1 else 'Not Tired'}")
    print(f"Average Confidence   : {avg_conf:.2f}%")
    print(f"Tired Frame Percent  : {tired_pct:.2f}%")
    print("===========================\n")


# ---------- MAIN ---------- #
if __name__ == "__main__":
    #TODO : change path for testing video
    video = r"C:/Users/smelt/OneDrive/Documents/Uni/Year_3/Dissertation/Git_Repo_Dest/TrainingData/Tired/SUST_Driver_Drowsiness_Dataset/d_378.mp4"

    #TODO: change name of model used
    model = tf.keras.models.load_model("drowsiness_model_V1.h5")

    run_drowsiness_check(video,model)
