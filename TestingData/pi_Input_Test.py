#testing AI models based on pi camera input
import mediapipe as mp
import cv2
import tensorflow as tf
#changing project root to this so it can find the utility files
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

#suporting functions
import utilities.cameraUtility as cu
import utilities.imageProcessing as ip

if __name__ == "__main__":
    # Initialize Mediapipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh

    # Open webcam
    cap = cu.open_webcam()
    if cap is None:
        exit()

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as face_mesh:

        while True:
            #wait for key input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # space bar pressed -> save image
                frame = cu.capture_frame(cap)
                if frame is not None:
                    print("failed to capture")
                else:
                    #display frame taken
                    cv2.imshow('Driver Drowsiness Detection - Pi Camera', frame)

                    #process the captured frame
                    features = ip.extract_features(frame)
                    if features is None:
                        print("No face detected.")
                    else:
                        #pass throgh AI model
                        print("Extracted Features:", features)
                        model = tf.keras.models.load_model("drowsiness_model_V2.h5")
                        pred = model.predict(features.reshape(1, -1), verbose=0)[0]
                        print("Model Prediction:", pred)




    # Release resources
    cu.close_webcam(cap)
