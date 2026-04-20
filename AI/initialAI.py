'''
Given a model and a video file, run the model over every frame and generate an
overall classificaiton.
This helped lead to the issues that while someone maybe tired, the fact of whole datasets being
considered tired, the few frames that actually hold tired actions (the blinking and ywning) are drowned
out by the amount of Alert frames.
'''
import tensorflow as tf
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

#suporting functions
import utilities.cameraUtility as cu
import utilities.imageProcessing as ip


# ---------- GLOBAL VARIABLES ---------- #
mp_face_mesh = mp.solutions.face_mesh

def run_drowsiness_check(video_path, AI_model):
    '''
    Input: Video path, Model path
    Output: none -> show in terminal the overall confidence and amount of a classfification within the video
    '''

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

        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        ) as face_mesh:

            features = ip.extract_features(frame, face_mesh)
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
