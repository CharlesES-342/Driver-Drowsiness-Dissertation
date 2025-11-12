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
    cap = cv2.VideoCapture(videoPath)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return []
    fps = cap.get(cv2.CAP_PROP_FPS)
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
    return np.linalg.norm(a - b)


def aspect_ratio(landmarks, eye_indices):
    p = np.array([landmarks[i] for i in eye_indices])
    return (euclidean_dist(p[1], p[5]) + euclidean_dist(p[2], p[4])) / (2 * euclidean_dist(p[0], p[3]))


def mouth_aspect_ratio(landmarks, mouth_indices):
    p = np.array([landmarks[i] for i in mouth_indices])
    return (euclidean_dist(p[1], p[7]) + euclidean_dist(p[2], p[6]) + euclidean_dist(p[3], p[5])) / (3 * euclidean_dist(p[0], p[4]))


def get_head_pose(landmarks):
    nose = landmarks[NOSE_TIP]
    chin = landmarks[CHIN]
    left_side = landmarks[LEFT_SIDE]
    right_side = landmarks[RIGHT_SIDE]

    pitch = np.degrees(np.arctan2(chin[1] - nose[1], chin[0] - nose[0]))
    yaw = np.degrees(np.arctan2(right_side[0] - left_side[0], right_side[1] - left_side[1]))
    return pitch, yaw


def determineEvent(frame, face_mesh):
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

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

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

            frameEvents["EAR"] = EAR
            frameEvents["MAR"] = MAR
            frameEvents["pitch"] = pitch
            frameEvents["yaw"] = yaw

            frameEvents["eyes_closed"] = EAR < 0.3
            frameEvents["mouth_open"] = MAR > 0.05 #this needs to change based on calibration
            frameEvents["head_down"] = pitch > 10
            frameEvents["head_turned"] = abs(yaw) > 15

    return frameEvents


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

                for frame in frames:
                    features = determineEvent(frame, face_mesh)
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
        return event_counts, total_frames


    # ---------- Process both datasets ----------
    not_tired_dir = os.path.abspath(os.path.join(base_dir, "..", "..", "TrainingData", "Not-Tired", "SUST_Driver_Drowsiness_Dataset", "subset")) # TODO:Adjust path as needed
    tired_dir = os.path.abspath(os.path.join(base_dir, "..", "..", "TrainingData", "Tired", "SUST_Driver_Drowsiness_Dataset", "subset")) # TODO:Adjust path as needed

    print("Processing Not-Tired videos...")
    not_tired_counts, not_tired_total = process_videos(not_tired_dir)

    print("Processing Tired videos...")
    tired_counts, tired_total = process_videos(tired_dir)

        # ---------- Compare results side-by-side ----------
    labels = list(not_tired_counts.keys())
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
