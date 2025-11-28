import tensorflow as tf
import numpy as np
import cv2
import json
from pathlib import Path
import mediapipe as mp

#changing project root to this so it can find the utility files
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
#suporting functions
import utilities.imageProcessing as imageProcessing


# ========================================= #
# LOAD FUNCTIONS FROM YOUR ORIGINAL SCRIPT
# ========================================= #

mp_face_mesh = mp.solutions.face_mesh

# ------------------- TIRED LABEL MAPPING ------------------- #
def is_tired(labels_list):
    tired_labels = {"mata_terpejam", "menguap"} # "eyes_closed", "yawning"
    return any(b["label"] in tired_labels for b in labels_list)


# ========================================= #
# BUILD DATASET
# ========================================= #

def build_dataset(image_folder, json_labels):
    print("\nBuilding dataset...")

    with open(json_labels, "r") as f:
        data = json.load(f)

    boxes = data["boundingBoxes"]

    X = []   # features
    y = []   # labels (0 = not tired, 1 = tired)

    #add in the face mesh extractor
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as face_mesh:

        for filename, labels_list in boxes.items():
            img_path = Path(image_folder) / filename

            if not img_path.exists():
                print(f"Missing: {filename}")
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            feats = imageProcessing.extract_features(img, face_mesh)
            if feats is None:
                continue

            X.append(feats)
            y.append(1 if is_tired(labels_list) else 0)

    X = np.array(X)
    y = np.array(y)

    print(f"Dataset size: {len(X)} samples")
    return X, y


# ========================================= #
# TRAIN MODEL
# ========================================= #

def train_model(X, y):

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(5,)),     # five features
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(2, activation="softmax")   # tired / not tired
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(X, y, epochs=25, batch_size=16, validation_split=0.2)

    model_name = "drowsiness_model_V3.h5" #change this as needed

    saveLocation = Path(__file__).resolve().parent.parent.parent / "AI_Models" / model_name
    model.save(str(saveLocation))

    print(model_name + " saved to AI_Models folder.")
    return model


# ============= MAIN ============= #

if __name__ == "__main__":
    # ========================================= #
    # TRAINING THE MODEL:
    # ========================================= #

    image_folder = r"C:\Users\smelt\OneDrive\Documents\Uni\Year_3\Dissertation\Git_Repo_Dest\Bina Nusantara University Data\modelling\training\training"
    json_path   = r"C:\Users\smelt\OneDrive\Documents\Uni\Year_3\Dissertation\Git_Repo_Dest\Bina Nusantara University Data\modelling\training\training\bounding_boxes.labels"

    X, y = build_dataset(image_folder, json_path)
    train_model(X, y)