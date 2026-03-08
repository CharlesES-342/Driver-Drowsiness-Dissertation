'''
To be run on the Pi to be used to sample data for teh Evaluation.
No assignment or use of the data is done here, just sampling and saving to a file.
This will then undergo processing and be used for the evaluation of the project in its entirety.

while functionality will be used form other areas of the project, inorder to keep
it as light weight as possible, sections of code will be used from other areas of the project,
and will be referenced as such, but not imported as a module, as this will add to the processing
time and memory usage of the Pi, and will also require the other files to be on the device
(they will not be on the device as this is a stand-alone file)
'''

import time
import cv2
import numpy as np


class cam:
    """Class to handle camera recording on Raspberry Pi"""
    
    def __init__(self, camera_index=0, fps=30):
        """
        Initialize camera object
        
        Args:
            camera_index: Index of camera device (default: 0)
            fps: Frames per second (default: 30)
        """
        self.camera_index = camera_index
        self.fps = fps
        self.frame_width = None
        self.frame_height = None
        self.cap = None
        self.writer = None
        
    def start_capture(self):
        """Initialize and start camera capture"""
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_index}")
    
    def start_recording(self, output_file):
        """
        Start recording video to file
        
        Args:
            output_file: Path to output video file
        """
        if self.cap is None:
            self.start_capture()
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_file, fourcc, self.fps, (self.frame_width, self.frame_height))
        
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_file}")
    
    def get_frame(self):
        """
        Capture a single frame
        
        Returns:
            Tuple of (success, frame) where success is bool and frame is numpy array
        """
        if self.cap is None:
            raise RuntimeError("Camera not initialized. Call start_capture() first.")
        
        ret, frame = self.cap.read()
        return ret, frame
    
    def write_frame(self, frame):
        """
        Write frame to video file
        
        Args:
            frame: Frame to write (numpy array)
        """
        if self.writer is None:
            raise RuntimeError("Recording not started. Call start_recording() first.")
        
        self.writer.write(frame)
    
    def stop_recording(self):
        """Stop recording and release video writer"""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
    
    def stop_capture(self):
        """Stop camera capture and release resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def release(self):
        """Release all resources"""
        self.stop_recording()
        self.stop_capture()


if __name__ == "__main__":
    # Example usage of cam class to record video for 5 mins
    recording_duration_mins = 5
    vid_number = 1  # number of videos to record

    #given the model runs between 4 and 5 fps, we will set the recording fps to 6 to
    # ensure we have enough frames for processing and evaluation while not being too much for me
    # to manually cassify the data (as well as for limited memory size)
    cam_obj = cam(0, 6)

    for i in range(vid_number):
        #repeat the recording process for the number of videos specified in the config file
        output_file = f"/home/pi/test_output_{i}.mp4"        
        cam_obj.start_recording(output_file)
        frames_to_capture = cam_obj.fps * (recording_duration_mins * 60)  # number of frames to capture based on fps and duration
        for j in range(frames_to_capture):
            success, frame = cam_obj.get_frame()
            if success:
                cam_obj.write_frame(frame)
            else:
                print("Failed to capture frame")
        cam_obj.stop_recording()

    cam_obj.release()