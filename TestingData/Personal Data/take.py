'''
This contains the code to take the images and run the tests on my self given a model and a video feed from the webcam
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
    #open webcam
    cap = camUtil.open_webcam()
    count = 0
    
    while True:
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            frame = camUtil.capture_frame(cap)
            if frame is not None:
                # Save the image in current file directory
                output_path = os.path.join(os.path.dirname(__file__), 'captured_image{count}.jpg')
                cv2.imwrite(output_path, frame)
                print(f"Image captured and saved to {output_path}")
                count+= 1
                if count == 100:
                    print("Captured 100 images, stopping.")
                    break
                
    camUtil.close_webcam(cap)


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
            labels_data.append({'image_name': image_name, 'label': label})
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
        writer = csv.DictWriter(csvfile, fieldnames=['image_name', 'label'])
        writer.writeheader()
        writer.writerows(labels_data)
    
    print(f"Saved {len(labels_data)} labels to {output_csv}")



if __name__ == "__main__":
    # take_images()

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
    assign_labels(BinaTraining, 'manual_labels.csv')
