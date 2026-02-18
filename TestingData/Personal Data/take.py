'''
This contains the code to take the images and run the tests on my self given a
model and a video feed from the webcam
'''
#changing project root to this so it can find the utility files
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
import utilities.cameraUtility as camUtil
import cv2
import os
import csv

#using the camera, take images
def take_images():
    print("image taking has begun")
    # Open webcam
    cap = camUtil.open_webcam()
    count = 0
    
    # Create a window to display the stream
    cv2.namedWindow('Camera Stream', cv2.WINDOW_NORMAL)
    
    while True:
        # Continuously read and display frames
        frame = camUtil.capture_frame(cap)
        
        if frame is not None:
            # Display the frame with instructions
            display_frame = frame.copy()
            
            # Add text overlay with instructions
            cv2.putText(display_frame, f'Images captured: {count}/100', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
            cv2.putText(display_frame, 'Press SPACE to capture | Q to quit', 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            
            # Show the frame
            cv2.imshow('Camera Stream', display_frame)
        
        # Wait for key press (1ms delay for smooth video)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord(' '):  # Spacebar to capture image
            if frame is not None:
                # Save the image in current file directory
                output_path = os.path.join(os.path.dirname(__file__), f'captured_image{count}.jpg')
                cv2.imwrite(output_path, frame)
                print(f"Image captured and saved to {output_path}")
                count += 1
                
                # Visual feedback - flash the screen
                flash = frame.copy()
                cv2.rectangle(flash, (0, 0), (flash.shape[1], flash.shape[0]), 
                            (255, 255, 255), 20)
                cv2.imshow('Camera Stream', flash)
                cv2.waitKey(100)  # Show flash for 100ms
                
                if count == 100:
                    print("Captured 100 images, stopping.")
                    break
    
    # Clean up
    cv2.destroyAllWindows()
    camUtil.close_webcam(cap)
    print(f"Total images captured: {count}")


def assign_labels(image_directory, output_csv='labels.csv'):
    """
    Assign labels to images in a directory and save to a single CSV file.
    
    Args:
        image_directory: Path to directory containing images
        output_csv: Name of output CSV file (default: 'labels.csv')
    """
    # Get all image files from the directory
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(image_extensions)]
    image_files.sort()  # Sort for consistent ordering
    
    if not image_files:
        print(f"No images found in {image_directory}")
        return
    
    print(f"Found {len(image_files)} images to label")
    print("Press '1' for Alert, '2' for Drowsy, 's' to skip, or 'q' to quit and save\n")
    
    labels_data = []
    
    for idx, image_name in enumerate(image_files):
        image_path = os.path.join(image_directory, image_name)
        
        # Read and display image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read {image_name}, skipping...")
            continue
        
        cv2.imshow('Classify Image', img)
        print(f"[{idx+1}/{len(image_files)}] Labeling: {image_name}")
        
        while True:
            key = cv2.waitKey(1)
            if key == ord('1'):
                label = 'alert'
                break
            elif key == ord('2'):
                label = 'drowsy'
                break
            elif key == ord('s'):
                print(f"Skipped {image_name}\n")
                label = None
                break
            elif key == ord('q'):
                cv2.destroyAllWindows()
                print("\nQuitting and saving labels...")
                save_labels_to_csv(labels_data, output_csv)
                return
        
        cv2.destroyAllWindows()
        
        if label:
            labels_data.append({'filename': image_name, 'label': label})
            print(f"Labeled as: {label}\n")
    
    # Save all labels to CSV
    save_labels_to_csv(labels_data, output_csv)
    print(f"\nAll done! Labeled {len(labels_data)} images.")

def save_labels_to_csv(labels_data, output_csv):
    """Save labels to CSV file."""
    if not labels_data:
        print("No labels to save.")
        return
    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['filename', 'label'])
        writer.writeheader()
        writer.writerows(labels_data)
    
    print(f"Saved {len(labels_data)} labels to {output_csv}")



if __name__ == "__main__":
    # assign_labels('/path/to/your/images', 'my_labels.csv')
    # or with default CSV name:
    # assign_labels('/path/to/your/images')
    from pathlib import Path

    # Get current file's directory
    current_dir = Path(__file__).parent

    # Go up directories
    Git_Repo_dest = current_dir.parent.parent.parent
    BinaTraining = Git_Repo_dest / 'Bina Nusantara University Data' / 'modelling' / 'testing' / 'testing'
    # Convert to string if needed
    #image_dir = str(image_dir)
    print(f"Looking for images in: {BinaTraining}")
    #assign_labels(BinaTraining, 'manual_labels.csv')

    # take images of me, then assign the labels to use for testing the model on myself
    take_images()
    assign_labels(os.path.join(current_dir), 'personal_labels.csv')
