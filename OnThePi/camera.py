import cv2

class Camera:
    def __init__(self, device_index=0, resolution=(299, 299)):
        """
        Initializes the camera capture.
        :param device_index: 0 is usually the default built-in camera/USB cam.
        :param resolution: The (width, height) expected by your Xception model.
        """
        self.cap = cv2.VideoCapture(device_index)
        self.resolution = resolution
        
        if not self.cap.isOpened():
            print("Error: Could not open video stream.")
        else:
            # Set resolution at the hardware level if supported
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    def getFrame(self):
        """
        Captures a single frame, processes it for the model, and returns it.
        """
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            return None

        # The Xception model expects RGB, but OpenCV captures in BGR.
        # We convert it here so the AI thread doesn't have to.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Ensure the frame is exactly 299x299
        frame_resized = cv2.resize(frame_rgb, self.resolution)
        
        return frame_resized

    def release(self):
        """Release the camera hardware."""
        self.cap.release()

# --- HELPER FUNCTIONS (Matching your code's import style) ---
# This allows you to call cu.getFrame() directly if you instantiate globally.

_cam_instance = None

def getFrame():
    global _cam_instance
    if _cam_instance is None:
        _cam_instance = Camera()
    return _cam_instance.getFrame()