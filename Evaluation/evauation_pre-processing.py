import pandas as pd
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import csv
import cv2
import threading
import time
from pathlib import Path
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input

def getDarkness(img):
    grayscale = img.convert('L')
    data = np.array(grayscale)
    return round(1.0 - (np.mean(data) / 255.0), 4)

class ImageAnnotator:
    def __init__(self, root):
        '''
        initialiser
        Defines the model used, frame sizes and the variables used for the annotation of the images.
        Also defines the UI layout and the buttons and their functions.
        Input: location
        Output: none
        '''
        self.root = root
        self.root.title("Driver Drowsiness Annotation Tool")
        self.root.geometry("1200x800")
        
        self.image_list = []
        self.current_idx_img = -1
        self.image_path = ""
        self.list_lock = threading.Lock()
        
        # Load Model
        self.model = None
        try:
            current_script_path = Path(__file__).resolve()
            model_path = current_script_path.parents[1] / "AI_Models" / "xception_drowsiness_model_v6.h5"
            if model_path.exists():
                self.model = tf.keras.models.load_model(str(model_path))
        except Exception as e:
            print(f"Model Load Error: {e}")
        
        # Observables
        self.image_id = tk.StringVar(value="None")
        self.progress_text = tk.StringVar(value="Ready")
        self.camera_pos = tk.IntVar(value=4) 
        self.accessories = tk.BooleanVar()
        self.hair_obscured = tk.BooleanVar()
        self.gender = tk.StringVar(value="Other")
        self.alertness = tk.StringVar(value="Alert")
        self.darkness_val = tk.DoubleVar(value=0.0)
        self.prediction_val = tk.StringVar(value="N/A")

        self.setup_ui()
        self.root.bind('<Return>', lambda event: self.save_and_next())

    def setup_ui(self):
        '''
        Window layout and design
        - left = image
        - right = controls
        '''
        # Main Paned Window
        self.paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashwidth=6, bg="#444")
        self.paned.pack(fill="both", expand=True)

        # LEFT: Image Display (Flexible)
        self.img_container = tk.Frame(self.paned, bg="black")
        self.paned.add(self.img_container, stretch="always")
        
        self.img_panel = tk.Label(self.img_container, text="No Image Loaded", bg="black", fg="white")
        self.img_panel.pack(expand=True, fill="both")

        # RIGHT: Controls (Fixed Width)
        self.controls = tk.Frame(self.paned, width=350, padx=15, pady=10)
        self.controls.pack_propagate(False) # prevent children from re-sizing the frame
        self.paned.add(self.controls, stretch="never")

        # Layout inside controls
        tk.Label(self.controls, textvariable=self.progress_text, font=('Arial', 9, 'italic')).pack(anchor="w")
        self.load_progress = ttk.Progressbar(self.controls, orient="horizontal", mode="determinate")
        self.load_progress.pack(fill="x", pady=5)

        # Classification Stats
        stats_frame = tk.LabelFrame(self.controls, text=" AI Classification ", padx=10, pady=5)
        stats_frame.pack(fill="x", pady=10)
        tk.Label(stats_frame, text="Darkness:").grid(row=0, column=0, sticky="w")
        tk.Label(stats_frame, textvariable=self.darkness_val, font=('Arial', 9, 'bold')).grid(row=0, column=1, sticky="w", padx=5)
        tk.Label(stats_frame, text="AI Predict:").grid(row=1, column=0, sticky="w")
        tk.Label(stats_frame, textvariable=self.prediction_val, fg="red", font=('Arial', 9, 'bold')).grid(row=1, column=1, sticky="w", padx=5)

        # Camera Position
        tk.Label(self.controls, text="Camera Pos (Grid):", font=('Arial', 10, 'bold')).pack(anchor="w")
        grid_frame = tk.Frame(self.controls)
        grid_frame.pack(pady=5)
        for i in range(9):
            tk.Radiobutton(grid_frame, variable=self.camera_pos, value=i).grid(row=i//3, column=i%3, padx=8, pady=2)

        # Inputs
        tk.Checkbutton(self.controls, text="Accessories", variable=self.accessories).pack(anchor="w")
        tk.Checkbutton(self.controls, text="Hair in Face", variable=self.hair_obscured).pack(anchor="w")

        tk.Label(self.controls, text="Gender:", font=('Arial', 10, 'bold')).pack(anchor="w", pady=(10, 0))
        for g in ["Male", "Female", "Other"]:
            tk.Radiobutton(self.controls, text=g, variable=self.gender, value=g).pack(anchor="w")

        tk.Label(self.controls, text="Label State:", font=('Arial', 10, 'bold')).pack(anchor="w", pady=(10, 0))
        for state in ["Alert", "Tired"]:
            tk.Radiobutton(self.controls, text=state, variable=self.alertness, value=state).pack(anchor="w")

        # Bottom Buttons
        btn_frame = tk.Frame(self.controls)
        btn_frame.pack(fill="x", side="bottom", pady=5)
        
        tk.Button(btn_frame, text="Open Folder", command=self.load_folder_threaded).pack(fill="x", pady=2)
        tk.Button(btn_frame, text="Previous", command=self.go_back).pack(fill="x", pady=2)
        self.btn_save = tk.Button(btn_frame, text="SAVE & NEXT", command=self.save_and_next, bg="#2ecc71", fg="white", font=('Arial', 11, 'bold'), height=2)
        self.btn_save.pack(fill="x", pady=5)

    def load_folder_threaded(self):
        '''
        Thread for loading the folder containing the images and videos
        Allows you to continue to load images (especially from videos), while also being able to
        annotate those already loaded
        '''
        folder = filedialog.askdirectory()
        if not folder: return
        with self.list_lock:
            self.image_list = []
            self.current_idx_img = -1
        threading.Thread(target=self.background_loader, args=(folder,), daemon=True).start()

    def background_loader(self, folder_path):
        '''
        allows you to the images and videos in the folder
        Also contains resuming logic (load last image - but only after it has been loaded)
        Doesn't work if annotations is already populated (means you have to wait for the entire folder
        to load before images will load) 
        Input: folder path for images
        Output: none
        '''
        temp_dir = Path("extracted_frames")
        temp_dir.mkdir(exist_ok=True)
        all_items = os.listdir(folder_path)
        self.root.after(0, lambda: self.load_progress.configure(maximum=len(all_items)))

        for f in all_items:
            full_path = os.path.join(folder_path, f)
            if not os.path.isfile(full_path): continue
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                with self.list_lock: self.image_list.append(full_path)
            elif f.lower().endswith(('.mp4', '.avi', '.mov')):
                self.extract_frames_fast(full_path, temp_dir)
            self.root.after(0, self.load_progress.step)

        # Sort list so it matches the expected order
        with self.list_lock: 
            self.image_list.sort()

        # Auto-resuming for joining back in
        csv_path = Path(__file__).resolve().parent / 'annotations.csv'
        resume_idx = 0
        
        if csv_path.exists():
            try:
                with open(csv_path, 'r', newline='') as f:
                    data = list(csv.DictReader(f))
                    if data:
                        last_saved_id = data[-1]['image_id']
                        # Find the index of the last saved image in our current list
                        for i, path in enumerate(self.image_list):
                            if os.path.basename(path) == last_saved_id:
                                resume_idx = i + 1
                                break
            except Exception as e:
                print(f"Error reading CSV for resume: {e}")

        # ensure the index is valid for the current images
        if resume_idx >= len(self.image_list):
            resume_idx = len(self.image_list) - 1 if len(self.image_list) > 0 else 0

        self.current_idx_img = resume_idx

        if len(self.image_list) > 0:
            self.root.after(200, lambda: self.display_image(self.image_list[self.current_idx_img]))

        self.root.after(0, lambda: self.progress_text.set(f"Loaded: {len(self.image_list)} (Resumed at {self.current_idx_img + 1})"))

    def extract_frames_fast(self, v_path, temp_dir):
        '''
        Get the images from videos (every 5 seconds - time period can be changed)
        Input: video path, temporary image path (where to put them)
        Output: none
        '''
        cap = cv2.VideoCapture(v_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0: return
        duration_ms = int((total_f / fps) * 1000)
        v_name = Path(v_path).stem
        skip_duration = 30000 #ms
        for msec in range(0, duration_ms, skip_duration):
            cap.set(cv2.CAP_PROP_POS_MSEC, msec)
            ret, frame = cap.read()
            if ret:
                s_path = str(temp_dir / f"{v_name}_t{msec//1000}s.jpg")#temp file ID
                cv2.imwrite(s_path, frame)
                with self.list_lock: self.image_list.append(s_path)
            else: break
        cap.release()

    def display_image(self, path):
        '''
        Show the image on screen (takes file location - could be temp file location for video images)
        Input: file location
        '''
        # Wait for file write
        for _ in range(5):
            if os.path.exists(path) and os.path.getsize(path) > 0: break
            time.sleep(0.1)

        try:
            self.image_path = path
            self.image_id.set(os.path.basename(path))
            raw_pil = Image.open(path).convert('RGB')
            self.darkness_val.set(getDarkness(raw_pil))
            
            # Model Predict
            if self.model:
                img_prep = raw_pil.resize((299, 299))
                img_arr = image.img_to_array(img_prep)
                img_arr = np.expand_dims(img_arr, axis=0)
                img_arr = preprocess_input(img_arr)
                preds = self.model.predict(img_arr, verbose=0)
                self.prediction_val.set("Tired" if np.argmax(preds[0]) == 0 else "Alert")
            
            # RESIZING LOGIC: Keep it contained
            self.root.update_idletasks()
            # We subtract a small margin to prevent the image from "pushing" the window bigger
            max_w = self.img_container.winfo_width() - 20
            max_h = self.img_container.winfo_height() - 20
            
            if max_w < 100: max_w, max_h = 800, 600 # Fallback
            
            display_pil = raw_pil.copy()
            display_pil.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
            
            img_tk = ImageTk.PhotoImage(display_pil)
            self.img_panel.config(image=img_tk, text="")
            self.img_panel.image = img_tk
            self.progress_text.set(f"Item {self.current_idx_img+1} / {len(self.image_list)}")
            
        except Exception as e:
            print(f"Error: {e}")

    def save_and_next(self):
        '''
        For image value submissions
        Store the changed/added values in the CSV
        '''
        if self.current_idx_img == -1 or not self.image_path: return
        
        current_id = self.image_id.get()
        new_row = {
            "image_id": current_id,
            "darkness": self.darkness_val.get(),
            "model_prediction": self.prediction_val.get(),
            "camera_pos": self.camera_pos.get(),
            "accessories": int(self.accessories.get()),
            "hair_in_face": int(self.hair_obscured.get()),
            "gender": self.gender.get(),
            "manual_state": self.alertness.get()
        }

        csv_path = Path(__file__).resolve().parent / 'annotations.csv'
        data = []
        if csv_path.exists():
            with open(csv_path, 'r', newline='') as f:
                data = list(csv.DictReader(f))
        
        found = False
        for row in data: #updating existing values
            if row['image_id'] == current_id:
                row.update(new_row)
                found = True
                break
        if not found: data.append(new_row)

        with open(csv_path, 'w', newline='') as f: #adding new values to bottom (appending)
            writer = csv.DictWriter(f, fieldnames=new_row.keys())
            writer.writeheader()
            writer.writerows(data)

        self.current_idx_img += 1
        with self.list_lock:
            if self.current_idx_img < len(self.image_list):
                self.display_image(self.image_list[self.current_idx_img])
            else:
                messagebox.showinfo("Done", "End of list reached.")

    def go_back(self):
        '''
        jump back a file (in case of mistakes or to check consistency)
        '''
        if self.current_idx_img > 0:
            self.current_idx_img -= 1
            self.display_image(self.image_list[self.current_idx_img]) #reload the section

if __name__ == "__main__":
    root = tk.Tk() #initialise window
    app = ImageAnnotator(root)
    root.mainloop()