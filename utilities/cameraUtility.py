'''
Camera Utilities and functions
'''
import cv2

def open_webcam(camera_index=1):
    '''
    Open the webcam
    Input: camera index (default 1)
    Output: camera instance
    '''
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None
    return cap

def close_webcam(cap):
    '''
    Close camera Instance
    Input: camera Instance
    Outout: none
    '''
    cap.release()
    cv2.destroyAllWindows()

def capture_frame(cap):
    '''
    Capture a single frame
    Input: camera instance
    Output: single frame
    '''
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        return None
    return frame
