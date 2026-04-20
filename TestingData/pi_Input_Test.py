#testing AI models based on pi camera input
'''
testing the models based on the camera on the Raspberry Pi.
This an overall usage file for models which use the Fature Extraction method using
the Mediapipe face mesh method.
'''
import mediapipe as mp
import cv2
# import tensorflow as tf # Not using full TensorFlow on Pi, using tflite instead
import tflite_runtime.interpreter as tflite #IGNORE ISSUE AS IT IS ON THE PI AND NOT ON LAPTOP
import numpy as np


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

    # Load TFLite model once
    interpreter = tflite.Interpreter(model_path="AI_Models/drowsiness_model_V2_pi.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()


    # Open webcam
    cap = cu.open_webcam()
    if cap is None:
        exit()
    print("webcam opened")

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as face_mesh:

        print("please press ' ' to take photo or 'q' to guit")
        while True:
            # Grab frame
            frame = cu.capture_frame(cap)
            if frame is None:
                continue

            # Show live feed
            cv2.imshow('Driver Drowsiness Detection - Pi Camera', frame)
            
            #wait for key input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # space bar pressed -> save image
                print("space pressed")
                if frame is None:
                    print("failed to capture")
                    continue
                else:
                    print("frame taken")
                    #process the captured frame
                    features = ip.extract_features(frame, face_mesh)
                    if features is not None:
                        #pass throgh AI model
                        print("Extracted Features:", features)
                        input_data = np.array(features, dtype=np.float32).reshape(1, -1)

                        interpreter.set_tensor(input_details[0]['index'], input_data)
                        interpreter.invoke()

                        pred = interpreter.get_tensor(output_details[0]['index'])[0]
                        print("Prediction:", pred)
                        #convert this to a class
                        pred_class = np.argmax(pred)
                        if( pred_class == 1):
                            print("Drowsiness Detected!")
                        else:
                            print("Driver is Alert.")

                        break
                    else:
                        print("No face detected, cannot extract features.")

        # Release resources
        cu.close_webcam(cap)