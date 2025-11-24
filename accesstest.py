import TestingData.SUST_Uni.simpleAI as simpleAI
import tensorflow as tf
import cv2

if __name__ == "__main__":
    model = tf.keras.models.load_model("SUST_Simple_model.h5")
    video_path = r"C:\Users\smelt\OneDrive\Documents\Uni\Year_3\Dissertation\Git_Repo_Dest\TrainingData\Not-Tired\SUST_Driver_Drowsiness_Dataset\n_13.mp4"  # Replace with your test video path
    simpleAI.test_on_video(video_path, model)