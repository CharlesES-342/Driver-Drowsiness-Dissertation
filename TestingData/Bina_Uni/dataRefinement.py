import csv
from pathlib import Path
from collections import defaultdict


#check if there exists more that one lable for a given image
def checkLabelConsistency(labelsDest):
    imageLabels = {}
    #given a file of labels, check if there exists more that one lable for a given image
    with open(labelsDest, newline='', encoding="utf-8") as f:
        reader = csv.reader(f)
        for row_num, (imageName, label) in enumerate(reader, start=1):
            if imageName not in imageLabels:
                imageLabels[imageName] = set()
            imageLabels[imageName].add(label)

    #check for inconsistencies
    for imageName in imageLabels:
        if len(imageLabels[imageName]) > 1:
            print(f"Inconsistent labels found for image {imageName}: {imageLabels[imageName]}")

def additionalLabels(path):
    #add in the additional labels for eyes open and eyes closed based on eye position
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh

    tanslations = {
        'tidak_menguap': 'not_yawning',
        'mata_terbuka': 'neutral',
        'menguap': 'yawning'
    }

    newLabels = defaultdict(set) #give a name of image and the new labels to be added

    with open(path / "Img_labels_refined.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
            
        #initialise the face mesh model
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
            #process images eye openess and closed
            
            #look at evey line (every file label)
            with open(path / "Img_labels.csv", newline='', encoding="utf-8") as f:
                reader = csv.reader(f)
                for row_num, (imageName, label) in enumerate(reader, start=1):
                    #load the image
                    image_path = path / imageName
                    if not image_path.is_file():
                        print(f"Image file {image_path} not found.")
                        continue

                    import cv2
                    import numpy as np

                    image = cv2.imread(str(image_path))
                    if image is None:
                        print(f"Failed to load image {image_path}.")
                        continue

                    # Convert the BGR image to RGB before processing.
                    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                    if not results.multi_face_landmarks:
                        print(f"No face landmarks detected in image {imageName}.")
                        continue

                    face_landmarks = results.multi_face_landmarks[0]

                    # Get eye landmarks
                    left_eye_indices = [33, 133, 159, 145]
                    right_eye_indices = [362, 263, 386, 374]

                    def eye_aspect_ratio(landmarks, indices):
                        p1 = np.array([landmarks.landmark[indices[0]].x, landmarks.landmark[indices[0]].y])
                        p2 = np.array([landmarks.landmark[indices[1]].x, landmarks.landmark[indices[1]].y])
                        p3 = np.array([landmarks.landmark[indices[2]].x, landmarks.landmark[indices[2]].y])
                        p4 = np.array([landmarks.landmark[indices[3]].x, landmarks.landmark[indices[3]].y])

                        # Calculate distances
                        vertical_distance = np.linalg.norm(p3 - p4)
                        horizontal_distance = np.linalg.norm(p1 - p2)

                        if horizontal_distance == 0:
                            return 0.0

                        ear = vertical_distance / horizontal_distance
                        return ear

                    left_ear = eye_aspect_ratio(face_landmarks, left_eye_indices)
                    right_ear = eye_aspect_ratio(face_landmarks, right_eye_indices)
                    avg_ear = (left_ear + right_ear) / 2.0

                    # Threshold for eye open/closed
                    EAR_THRESHOLD = 0.25

                    if avg_ear < EAR_THRESHOLD:
                        eye_label = 'eyes_closed'  # eyes closed
                    else:
                        eye_label = 'eyes_open'    # eyes open
                    
                    #convert the current label to english
                    if label in tanslations:
                        translated_label = tanslations[label]
                    else:
                        translated_label = label  # keep original if not found
                        print(f"Label {label} not found in translations.")
                    newLabels[imageName].add(eye_label)
                    newLabels[imageName].add(translated_label)
                    
                    writer.writerow([imageName, ",".join(sorted(newLabels[imageName]))])



if __name__ == "__main__":
    CURRENT_FILE = Path(__file__).resolve()
    ROOT_DIR = CURRENT_FILE.parents[3]
    #checkLabelConsistency(str(ROOT_DIR / "Bina Nusantara University Data" / "modelling" / "training" / "training" / "Img_labels.csv"))
    additionalLabels(ROOT_DIR / "Bina Nusantara University Data" / "modelling" / "testing" / "testing" )