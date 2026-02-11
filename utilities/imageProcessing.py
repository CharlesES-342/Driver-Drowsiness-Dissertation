import numpy as np
import mediapipe as mp
import cv2
import tensorflow as tf


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

def extract_features(frame, face_mesh):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

def imageEffects(img):
    # Apply augmentations BEFORE preprocessing
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
        
    return img