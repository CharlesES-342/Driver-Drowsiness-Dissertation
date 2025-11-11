# this file will be used to process all of teh videos in a file and
# convert them into on CSV file for easier AI input for training.
import cv2
import mediapipe as mp
#mediapipe set up
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
keyLandmarks = {
    "left_eye_top": 159,
    "left_eye_bottom": 145,
    "right_eye_top": 386,
    "right_eye_bottom": 374,
    "upper_lip": 13,
    "lower_lip": 14,
    "nose_tip": 1,
    "forehead": 10
}


import os
import time

# copied code for frame processing - in pi_dection
def process_frame(frame, face_mesh):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    annotated_frame = frame.copy()

    left_eye_height = right_eye_height = mouth_height = dx = dy = 0

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
            )

            mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
            )

            landmarks = face_landmarks.landmark
            h, w, _ = frame.shape

            def normalized_to_pixel(index):
                pt = landmarks[index] 
                return int(pt.x * w), int(pt.y * h)

            # Eyes
            left_eye_height = abs(normalized_to_pixel(keyLandmarks["left_eye_bottom"])[1] - normalized_to_pixel(keyLandmarks["left_eye_top"])[1])
            right_eye_height = abs(normalized_to_pixel(keyLandmarks["right_eye_bottom"])[1] - normalized_to_pixel(keyLandmarks["right_eye_top"])[1])

            # Mouth
            mouth_height = abs(normalized_to_pixel(keyLandmarks["lower_lip"])[1] - normalized_to_pixel(keyLandmarks["upper_lip"])[1])

            # Head angles
            dx = normalized_to_pixel(keyLandmarks["forehead"])[0] - normalized_to_pixel(keyLandmarks["nose_tip"])[0]
            dy = normalized_to_pixel(keyLandmarks["forehead"])[1] - normalized_to_pixel(keyLandmarks["nose_tip"])[1]

    return {
        'left_eye_height': left_eye_height,
        'right_eye_height': right_eye_height,
        'mouth_height': mouth_height,
        'dx': dx,
        'dy': dy
    }




# get fromes from a specific video file
def getSampleFrames(videoPath, numFrames=10):
    cap = cv2.VideoCapture(videoPath)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return []

    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # get the indicies of the frames to sample
    frameIndices = [int(i * totalFrames / numFrames) for i in range(numFrames)]
    frames = [] #empty list to hold the frames

    for idx in frameIndices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            print(f"Warning: Could not read frame at index {idx}.")
    
    cap.release()
    return frames

#conver the features into valid CSV format
def featuresToCSV(name, features, classification):
    if features is None:
        return ""
    return f"{name}, {features['left_eye_height']},{features['right_eye_height']},{features['mouth_height']},{features['dx']},{features['dy']}, {classification}\n"

#create CSV file if not exists and add headers
def createCSVFile(filePath):
    with open(filePath, 'w') as f:
        f.write("name,left_eye_height,right_eye_height,mouth_height,dx,dy,classification\n")

#main function to process videos and save to CSV
if __name__ == "__main__":
    # Base directory of this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Build correct paths regardless of where you run from
    csvFilePath = os.path.join(base_dir, "..", "TrainingData", "SUST_output_features.csv")

    os.makedirs(os.path.dirname(csvFilePath), exist_ok=True)


    #check if there is a CSV file, if not create one
    if(not os.path.exists(csvFilePath)):
        createCSVFile(csvFilePath)

    #eventually do for all videos in a folder
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # directory that contains all the videos to process (Not-Tired videos) TODO: for all videos later
    video_dir = os.path.abspath(os.path.join(base_dir, "..", "..", "TrainingData", "Not-Tired", "SUST_Driver_Drowsiness_Dataset"))

    # Get all .mp4 files in the folder
    video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        with open(csvFilePath, 'a') as f:
            print(len(video_files))
            #pause of 10 seconds before processing for testing only
            time.sleep(10)

            #process each video file
            for video in video_files:
                videoPath = os.path.join(video_dir, video)
                frames = getSampleFrames(videoPath, numFrames=10)

                for i, frame in enumerate(frames):
                    features = process_frame(frame, face_mesh)
                    # use filename (without extension) as ID
                    video_id = os.path.splitext(video)[0]
                    csvLine = featuresToCSV(video_id, features, "not_tired")
                    f.write(csvLine)
    
