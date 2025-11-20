import tensorflow as tf
import cv2
import numpy as np
import json
import mediapipe as mp
from pathlib import Path

# ---------- GLOBAL VARIABLES ---------- #
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

def extract_head_pose(landmarks):
    nose = landmarks[1]
    chin = landmarks[152]
    left_eye = landmarks[33]
    right_eye = landmarks[263]

    pitch = np.arctan2(chin[1] - nose[1], chin[0] - nose[0]) * 180 / np.pi
    yaw = np.arctan2(right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]) * 180 / np.pi

    return pitch, yaw

# ---------- EXTRACT FEATURES ---------- #
def extract_features(frame):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as face_mesh:

        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None

        lm = results.multi_face_landmarks[0]
        pts = np.array([(p.x * w, p.y * h) for p in lm.landmark])

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


# ---------- MAPPING INDONESIAN LABELS → TIRED / NOT TIRED ---------- #
def is_tired(labels_list):
    tired_labels = {"mata_terpejam", "menguap"}
    for box in labels_list:
        if box["label"] in tired_labels:
            return True
    return False


# ---------- MAIN IMAGE PROCESSING FUNCTION ---------- #
def run_drowsiness_check_images(image_folder, json_labels, model):

    with open(json_labels, "r") as f:
        data = json.load(f)

    box_data = data["boundingBoxes"]

    predictions = []
    model_outputs = []

    print("\nRunning inference on image dataset...\n")

    for filename, label_list in box_data.items():

        img_path = Path(image_folder) / filename

        if not img_path.exists():
            print(f"❗ Missing image: {filename}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"❗ Failed to load: {filename}")
            continue

        features = extract_features(img)
        if features is None:
            print(f"❗ No face detected: {filename}")
            continue

        pred = model.predict(features.reshape(1, -1), verbose=0)[0]
        pred_class = int(np.argmax(pred))

        predictions.append(pred_class)
        model_outputs.append(pred)

    # ---------- SUMMARY ---------- #
    tired_percentage = (predictions.count(1) / len(predictions)) * 100
    avg_confidence = np.mean([p.max() for p in model_outputs]) * 100

    print("========= RESULTS =========")
    print(f"Images processed       : {len(predictions)}")
    print(f"Tired Frame Percent    : {tired_percentage:.2f}%")
    print(f"Average Model Confidence: {avg_confidence:.2f}%")
    print(f"Final Classification   : {'TIRED' if tired_percentage > 50 else 'NOT TIRED'}")
    print("==========================\n")


def predict_images(image_folder, model, json_path):
    """
    Runs prediction on ALL images in a folder.
    Returns total correct and incorrect predictions.
    """

    folder = Path(image_folder)
    correct = 0
    incorrect = 0
    total = 0

    # Load labels file once
    with open(json_path, "r") as f:
        data = json.load(f)
    box_data = data["boundingBoxes"]

    print("\n========== RUNNING BATCH EVALUATION ==========\n")

    for image_path in folder.iterdir():
        if not image_path.suffix.lower() in [".jpg", ".png", ".jpeg"]:
            continue  # skip non-images

        filename = image_path.name

        if filename not in box_data:
            print(f"Skipping (no label found): {filename}")
            continue

        # ---- Load image ----
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Could not load image: {filename}")
            continue

        # ---- Extract features ----
        features = extract_features(img)
        if features is None:
            print(f"No face detected: {filename}")
            continue

        # ---- Model prediction ----
        pred = model.predict(features.reshape(1, -1), verbose=0)[0]
        pred_class = int(np.argmax(pred))

        # ---- Ground truth ----
        true_class = 1 if is_tired(box_data[filename]) else 0

        # ---- Update counters ----
        if pred_class == true_class:
            correct += 1
            result = "✅ Correct"
        else:
            incorrect += 1
            result = "❌ Incorrect"

        total += 1

        print(f"{filename}: Pred={pred_class}, True={true_class} → {result}")

    # ---------- SUMMARY ----------
    print("\n=========== SUMMARY ===========")
    print(f"Total images tested : {total}")
    print(f"Correct predictions : {correct}")
    print(f"Incorrect predictions: {incorrect}")
    if total > 0:
        print(f"Accuracy            : {(correct / total) * 100:.2f}%")
    print("================================\n")

    return correct, incorrect


def get_ground_truth_label(json_path, filename):
    """
    Returns 1 if image is labeled TIRED in the JSON file,
    returns 0 if NOT TIRED.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    box_data = data["boundingBoxes"]

    if filename not in box_data:
        print(f"No label found for: {filename}")
        return None

    return 1 if is_tired(box_data[filename]) else 0


# ---------- MAIN ---------- #
if __name__ == "__main__":

    image_folder = r"C:\Users\smelt\OneDrive\Documents\Uni\Year_3\Dissertation\Git_Repo_Dest\Bina Nusantara University Data\modelling\training\training" #TODO: change to your image folder
    json_path = r"C:\Users\smelt\OneDrive\Documents\Uni\Year_3\Dissertation\Git_Repo_Dest\Bina Nusantara University Data\modelling\training\training\bounding_boxes.labels"#TODO: change to your json labels file

    model = tf.keras.models.load_model("drowsiness_model_V2.h5")

    # run_drowsiness_check_images(image_folder, json_path, model)

    test_images_path = r"C:\Users\smelt\OneDrive\Documents\Uni\Year_3\Dissertation\Git_Repo_Dest\Bina Nusantara University Data\modelling\testing\testing" #TODO: change to your test image path
    test_json_path = r"C:\Users\smelt\OneDrive\Documents\Uni\Year_3\Dissertation\Git_Repo_Dest\Bina Nusantara University Data\modelling\testing\testing\bounding_boxes.labels"#TODO: change to your test json labels file
    correct, incorrect = predict_images(test_images_path, model, test_json_path)
