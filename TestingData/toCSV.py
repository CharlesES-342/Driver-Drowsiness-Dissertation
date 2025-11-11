# this file will be used to process all of teh videos in a file and
# convert them into on CSV file for easier AI input for training.
import cv2
import mediapipe as mp
# for use of my existing frame analysis code
from Face_Detection.pi_detection import process_frame


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

def analyzeFrame(frame):
    # call my existing face detection code to get the features
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        features = process_frame(frame, face_mesh)
    return features

#conver the features into valid CSV format
def featuresToCSV(name, features, classification):
    if features is None:
        return ""
    return f"{name, features['left_eye_height']},{features['right_eye_height']},{features['mouth_height']},{features['dx']},{features['dy'], classification}\n"

#create CSV file if not exists and add headers
def createCSVFile(filePath):
    with open(filePath, 'w') as f:
        f.write("name,left_eye_height,right_eye_height,mouth_height,dx,dy,classification\n")

#main function to process videos and save to CSV
if __name__ == "__main__":
    #check if there is a CSV file, if not create one
    csvFilePath = "../TrainingData/SUST_output_features.csv"
    if(not os.path.exists(csvFilePath)):
        createCSVFile(csvFilePath)

    #eventually do for all videos in a folder
    videoPath = "../TrainingData/Not-Tired/n_1.mp4"
    frames = getSampleFrames(videoPath, numFrames=10)
    with open(csvFilePath, 'a') as f:
        for i, frame in enumerate(frames):
            features = analyzeFrame(frame)
            csvLine = featuresToCSV(f"n_1_frame_{i}", features, "not_tired")
            f.write(csvLine) 
    
