import cv2

def open_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None
    return cap

def close_webcam(cap):
    cap.release()
    cv2.destroyAllWindows()

def capture_frame(cap):
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        return None
    return frame
