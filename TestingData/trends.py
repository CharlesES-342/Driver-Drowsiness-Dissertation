'''
Used to detect and visualise general trends within the data. This extends to the presence of 
features:
.eye openness proportional to width
. mouth openness
. pitch
. yaw
. roll
, this is used to give proof (or suggest there is efficient evidence) that the datasets are viable
for constructing an AI model.
Show all findings as graphs. The idea being you can play around with thresholds to help visualise
the spead of data within the datasets and hopefully identify a difference in the results beween tired
and alert cases.
- from testing, there is sufficinet evidence to suggest that the datasets are sufficuient enough for
training. This is outlined in the report.
'''
import cv2
import mediapipe as mp
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Landmark groups for reference
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
OUTER_MOUTH = [61, 81, 13, 311, 308, 402, 14, 178]
NOSE_TIP = 1
CHIN = 152
LEFT_SIDE = 234
RIGHT_SIDE = 454


# Get frames at approximately 2 fps
def get2fps(videoPath):
    '''
    Extract 2 frames per second from a video
    Input: video location
    Output: frames extracted
    '''
    #open video feed
    cap = cv2.VideoCapture(videoPath)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return []
    fps = cap.get(cv2.CAP_PROP_FPS)
    # determine the frames interval to form a resulting 2fps (e.g. from a 30fps video, capture a 
    # frame every 15 frames)
    frameInterval = int(fps / 2) if fps > 0 else 1
    frames = []
    for i in range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), frameInterval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            print(f"Warning: Could not read frame at index {i}.")
    cap.release()
    return frames


def euclidean_dist(a, b):
    '''
    Determine the Eucliden Distance between 2 points
    Input: point 1, point 2
    Output: distance
    '''
    return np.linalg.norm(a - b)


def aspect_ratio(landmarks, eye_indices):
    '''
    Determine the aspect ration of the eyes
    Input: landmarks (from the image), eye landmark 
    Output: aspect ratio
    '''
    p = np.array([landmarks[i] for i in eye_indices])
    return (euclidean_dist(p[1], p[5]) + euclidean_dist(p[2], p[4])) / (2 * euclidean_dist(p[0], p[3]))


def mouth_aspect_ratio(landmarks, mouth_indices):
    '''
    Determine the aspect ration of the mouth
    Input: landmarks (from the image), mouth landmark 
    Output: aspect ratio
    '''
    p = np.array([landmarks[i] for i in mouth_indices])
    return (euclidean_dist(p[1], p[7]) + euclidean_dist(p[2], p[6]) + euclidean_dist(p[3], p[5])) / (3 * euclidean_dist(p[0], p[4]))


def get_head_pose(landmarks):
    '''
    Determine the Pitc and Yaw of the head within the image
    Input: landmarks (taken form Mediapipe)
    Output: pitch, yaw
    '''
    nose = landmarks[NOSE_TIP]
    chin = landmarks[CHIN]
    left_side = landmarks[LEFT_SIDE]
    right_side = landmarks[RIGHT_SIDE]

    pitch = np.degrees(np.arctan2(chin[1] - nose[1], chin[0] - nose[0]))
    yaw = np.degrees(np.arctan2(right_side[0] - left_side[0], right_side[1] - left_side[1]))
    return pitch, yaw

#TODO: can remove this function later
def accRange(data):
    array = np.array(data)
    str = np.std(array)
    accMax = np.median(array) + 1.5*str
    accMin = np.median(array) - 1.5*str
    extremes = 0
    for val in array:
        if val > accMax or val < accMin:
            extremes += 1
    return extremes


def determineEvent(frame, face_mesh):
    '''
    Used to classify that tere is substantial evidence to suggest that an event (eyes
    closed, mouth open/yawning, ...) is present
    Input: frame, face mesh (constructed)
    Output: List[frame events], yaw
    '''
    frameEvents = {
        "face_detected": False,
        "eyes_closed": False,
        "mouth_open": False,
        "head_down": False,
        "head_turned": False,
        "EAR": None,
        "MAR": None,
        "pitch": None,
        "yaw": None
    }

    if frame is None:
        print(f"ERROR: Could not load image: {frame_path}")
        return None, None
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    yaw = 0

    #if there is a face present
    if results.multi_face_landmarks:
        frameEvents["face_detected"] = True

        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = np.array([[lm.x * w, lm.y * h] for lm in face_landmarks.landmark])

            left_EAR = aspect_ratio(landmarks, LEFT_EYE)
            right_EAR = aspect_ratio(landmarks, RIGHT_EYE)
            EAR = (left_EAR + right_EAR) / 2.0
            MAR = mouth_aspect_ratio(landmarks, OUTER_MOUTH)
            pitch, yaw = get_head_pose(landmarks)

            # values for a potential event
            frameEvents["EAR"] = EAR
            frameEvents["MAR"] = MAR
            frameEvents["pitch"] = pitch
            frameEvents["yaw"] = yaw

            # this needs to change based on calibration (threshold for identifying a feature)
            # if there is substantial evidence to sugges that an event has occured...
            frameEvents["eyes_closed"] = EAR < 0.3
            frameEvents["mouth_open"] = MAR > 0.05 
            frameEvents["head_down"] = pitch > 10
            frameEvents["head_turned"] = abs(yaw) > 15

    return frameEvents, yaw


# ------------------ MAIN ------------------
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mp_face_mesh = mp.solutions.face_mesh

    # ---------- Function to process one category ----------
    def process_videos(video_dir):
        event_counts = Counter({
            "eyes_closed": 0,
            "mouth_open": 0,
            "head_down": 0,
            "head_turned": 0,
            "face_detected": 0
        })

        total_frames = 0
        total_extreme_headturns = 0   # <- NEW: sum of per-video extremes

        video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]

        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            for video in video_files:

                videoPath = os.path.join(video_dir, video)
                frames = get2fps(videoPath)
                total_frames += len(frames)

                yaw_values = []   # <- Per-video yaw list

                for frame in frames:
                    features, headTilt = determineEvent(frame, face_mesh)

                    if features["face_detected"]:
                        event_counts["face_detected"] += 1
                        if features["eyes_closed"]:
                            event_counts["eyes_closed"] += 1
                        if features["mouth_open"]:
                            event_counts["mouth_open"] += 1
                        if features["head_down"]:
                            event_counts["head_down"] += 1
                        if features["head_turned"]:
                            event_counts["head_turned"] += 1

                        yaw_values.append(headTilt)

                # ---- PER-VIDEO head-tilt calibration ----
                if len(yaw_values) > 5:
                    arr = np.array(yaw_values)
                    std = np.std(arr)
                    mid = np.median(arr)

                    max_acc = mid + 2.5 * std #TODO: change apropriately to test extreme head tilts
                    min_acc = mid - 2.5 * std

                    extreme_count = int(np.sum((arr > max_acc) | (arr < min_acc)))
                else:
                    extreme_count = 0

                total_extreme_headturns += extreme_count   # accumulate

        return event_counts, total_frames, total_extreme_headturns



    # ---------- Process both datasets ----------
    not_tired_dir = os.path.abspath(os.path.join(base_dir, "..", "TrainingData", "Not-Tired")) # TODO:Adjust path as needed
    tired_dir = os.path.abspath(os.path.join(base_dir, "..", "TrainingData", "Tired")) # TODO:Adjust path as needed

    print("Processing Not-Tired videos...")
    not_tired_counts, not_tired_total,not_tired_headTilt = process_videos(not_tired_dir)

    print("Processing Tired videos...")
    tired_counts, tired_total, tired_headTilt = process_videos(tired_dir)

        # ---------- Compare results side-by-side ----------
    not_tired_headTiltExtreme = not_tired_headTilt
    tired_headTiltExtreme = tired_headTilt

    not_tired_counts["extreme_head_tilt"] = not_tired_headTiltExtreme
    tired_counts["extreme_head_tilt"] = tired_headTiltExtreme


    #labels
    labels = list(not_tired_counts.keys())


    #values
    not_tired_values = [not_tired_counts[k] for k in labels]
    tired_values = [tired_counts[k] for k in labels]

    x = np.arange(len(labels))
    width = 0.35  # bar width

    # Compute percentages for each event
    not_tired_percent = [(v / not_tired_total * 100) if not_tired_total > 0 else 0 for v in not_tired_values]
    tired_percent = [(v / tired_total * 100) if tired_total > 0 else 0 for v in tired_values]


    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width/2, not_tired_values, width,
                    label=f"Not-Tired (frames: {not_tired_total})", color="skyblue", edgecolor="black")
    bars2 = plt.bar(x + width/2, tired_values, width,
                    label=f"Tired (frames: {tired_total})", color="salmon", edgecolor="black")

    # Add percentage labels above bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + max(not_tired_values)*0.02, f"{not_tired_percent[i]:.1f}%", ha='center', va='bottom', fontsize=9, color='black')
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + max(tired_values)*0.02, f"{tired_percent[i]:.1f}%", ha='center', va='bottom', fontsize=9, color='black')

    plt.xticks(x, labels)
    plt.xlabel("Event Type")
    plt.ylabel("Number of Frames Detected")
    plt.title("Event Occurrence Comparison: Not-Tired vs Tired Videos")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Optional: Print summary table to console
    print("\n--- Summary ---")
    print(f"{'Event':<15} | {'Not-Tired (%)':<15} | {'Tired (%)':<15}")
    print("-" * 50)
    for i, label in enumerate(labels):
        print(f"{label:<15} | {not_tired_percent[i]:<15.2f} | {tired_percent[i]:<15.2f}")

    print(f"\nTotal frames (Not-Tired): {not_tired_total}")
    print(f"Total frames (Tired): {tired_total}")



    # reading the data from a csv file and plot it accordingly
    #for the Bina Nusantara University data
    BinaUni_overviewFile = os.path.abspath(os.path.join(base_dir, "..", "..", "Bina Nusantara University Data", "modelling", "training", "training")) # TODO:Adjust path as needed
    
    #open the file as read
    import json
    
    with open(os.path.join(BinaUni_overviewFile, "info.labels"), "r") as f:
        data = json.load(f)

    eyes_open_count = 0
    eyes_closed_count = 0
    yawning_count = 0
    not_yawning_count = 0
    mouthOpen_count = 0
    mouthClosed_count = 0

    #go through entries
    for item in data["files"]:
        sections = item["boundingBoxes"]
        for part in sections:
            if part["label"] == "mata_terbuka":
                eyes_open_count += 1
            elif part["label"] == "mata_terpejam":
                eyes_closed_count += 1
            elif part["label"] == "menguap":
                yawning_count += 1
            elif part["label"] == "tidak_menguap":
                not_yawning_count += 1
            elif part["label"] == "mulut_terbuka":
                mouthOpen_count += 1   
            elif part["label"] == "mulut_tertutup":
                mouthClosed_count += 1
    
    #display as a graph
    labels = [
    "Eyes Open",
    "Eyes Closed",
    "Yawning",
    "Not Yawning",
    "Mouth Open",
    "Mouth Closed"
]

values = [
    eyes_open_count,
    eyes_closed_count,
    yawning_count,
    not_yawning_count,
    mouthOpen_count,
    mouthClosed_count
]

plt.figure(figsize=(10, 6))
plt.bar(labels, values)
plt.title("Bounding Box Label Frequency")
plt.xlabel("Label")
plt.ylabel("Count")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

#go through the file of frames and determine the difference in event classification (between them and me)
with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            #get all the frames from the directory
            
            frame_files =[f for f in os.listdir(BinaUni_overviewFile) if f.endswith(".jpg")]
            event_counts = Counter({
                "eyes_closed": 0,
                "mouth_open": 0,
                "head_down": 0,
                "head_turned": 0,
                "face_detected": 0
            })

            for frame in frame_files:
                frame_path = os.path.join(BinaUni_overviewFile, frame)
                features, headTilt = determineEvent(frame_path, face_mesh)
                if features["face_detected"]:
                    event_counts["face_detected"] += 1
                    if features["eyes_closed"]:
                        event_counts["eyes_closed"] += 1
                    if features["mouth_open"]:
                        event_counts["mouth_open"] += 1
                    if features["head_down"]:
                        event_counts["head_down"] += 1
                    if features["head_turned"]:
                        event_counts["head_turned"] += 1
                        
#Display results
labels = list(event_counts.keys())
values = list(event_counts.values())

plt.figure(figsize=(10, 6))
plt.bar(labels, values)
plt.title("Event Detection Counts")
plt.ylabel("Count")
plt.xlabel("Event Type")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()