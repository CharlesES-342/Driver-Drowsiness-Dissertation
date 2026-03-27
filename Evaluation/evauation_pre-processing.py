import pandas as pd
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import csv
from pathlib import Path
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input

# --- Helper Function ---
def getDarkness(img):
    """Calculates image darkness: 0 (White) to 1 (Black)"""
    grayscale = img.convert('L')
    data = np.array(grayscale)
    avg_brightness = np.mean(data)
    darkness = 1.0 - (avg_brightness / 255.0)
    return round(darkness, 4)

class ImageAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Annotation Tool")
        
        # 1. Window Configuration
        self.root.geometry("1400x850") 
        
        # 2. Load Model from AI_Models Folder
        self.model = None
        try:
            # Get the path of the current script (evauation_pre-processing.py)
            current_script_path = Path(__file__).resolve()
            
            # Step back 1 level into 'Driver-Drowsiness-Dissertation'
            # Then enter 'AI_Models' and find the file
            model_path = current_script_path.parents[1] / "AI_Models" / "xception_drowsiness_model_v6.h5"
            
            if model_path.exists():
                self.model = tf.keras.models.load_model(str(model_path))
                print(f"SUCCESS: Loaded model from {model_path}")
            else:
                print(f"ERROR: Model not found at {model_path}")
                messagebox.showerror("Model Error", f"File not found:\n{model_path}")
        except Exception as e:
            print(f"Load Error: {e}")
        
        # 3. Data State
        self.image_path = ""
        self.image_list = []
        self.current_idx = -1
        
        # 4. Observables
        self.image_id = tk.StringVar(value="None")
        self.progress_text = tk.StringVar(value="No folder loaded")
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
        self.root.columnconfigure(0, weight=4) 
        self.root.columnconfigure(1, weight=1) 
        self.root.rowconfigure(0, weight=1)

        img_frame = tk.Frame(self.root, bg="black")
        img_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.img_panel = tk.Label(img_frame, text="No Image Loaded", bg="black", fg="white")
        self.img_panel.pack(expand=True, fill="both")

        controls = tk.Frame(self.root, width=320)
        controls.grid(row=0, column=1, sticky="nsew", padx=10, pady=5)
        controls.pack_propagate(False) 

        tk.Label(controls, textvariable=self.progress_text, font=('Arial', 8, 'italic')).pack(anchor="w")
        tk.Label(controls, text="Image ID:", font=('Arial', 9, 'bold')).pack(anchor="w")
        tk.Label(controls, textvariable=self.image_id, fg="blue", font=('Arial', 8)).pack(anchor="w", pady=(0, 2))

        stats_frame = tk.LabelFrame(controls, text="Auto Stats", padx=5, pady=2, font=('Arial', 8))
        stats_frame.pack(fill="x", pady=2)
        tk.Label(stats_frame, text="Dark:", font=('Arial', 8)).grid(row=0, column=0, sticky="w")
        tk.Label(stats_frame, textvariable=self.darkness_val, font=('Arial', 8, 'bold')).grid(row=0, column=1, sticky="w", padx=5)
        tk.Label(stats_frame, text="Model:", font=('Arial', 8)).grid(row=0, column=2, sticky="w", padx=(10,0))
        tk.Label(stats_frame, textvariable=self.prediction_val, fg="red", font=('Arial', 8, 'bold')).grid(row=0, column=3, sticky="w")

        tk.Label(controls, text="Camera Pos:", font=('Arial', 9, 'bold')).pack(anchor="w", pady=(2, 0))
        grid_frame = tk.Frame(controls)
        grid_frame.pack(pady=2)
        for i in range(9):
            tk.Radiobutton(grid_frame, variable=self.camera_pos, value=i).grid(row=i//3, column=i%3, padx=2, pady=1)

        toggle_frame = tk.Frame(controls)
        toggle_frame.pack(fill="x", pady=2)
        tk.Checkbutton(toggle_frame, text="Accessory", variable=self.accessories, font=('Arial', 8)).pack(side="left")
        tk.Checkbutton(toggle_frame, text="Hair in Face", variable=self.hair_obscured, font=('Arial', 8)).pack(side="left", padx=5)

        tk.Label(controls, text="Gender:", font=('Arial', 9, 'bold')).pack(anchor="w", pady=(2, 0))
        g_frame = tk.Frame(controls)
        g_frame.pack(anchor="w")
        for g in ["Male", "Female", "Other"]:
            tk.Radiobutton(g_frame, text=g, variable=self.gender, value=g, font=('Arial', 8)).pack(side="left")

        tk.Label(controls, text="State:", font=('Arial', 9, 'bold')).pack(anchor="w", pady=(2, 0))
        s_frame = tk.Frame(controls)
        s_frame.pack(anchor="w")
        for state in ["Alert", "Tired"]:
            tk.Radiobutton(s_frame, text=state, variable=self.alertness, value=state, font=('Arial', 8)).pack(side="left")

        button_frame = tk.Frame(controls)
        button_frame.pack(fill="x", side="bottom", pady=5)
        tk.Button(button_frame, text="Open Folder", command=self.load_folder, font=('Arial', 9)).pack(fill="x", pady=2)
        self.btn_save = tk.Button(button_frame, text="SAVE & NEXT", command=self.save_and_next, bg="#2ecc71", fg="white", font=('Arial', 10, 'bold'), height=1)
        self.btn_save.pack(fill="x", pady=2)
        tk.Button(button_frame, text="Previous", command=self.go_back, font=('Arial', 9)).pack(fill="x", pady=2)

    def predict_model(self, pil_img):
        if self.model is None: return "N/A"
        try:
            img = pil_img.resize((299, 299))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            preds = self.model.predict(img_array, verbose=0)
            
            # Logic: preds[0] = [drowsy_conf, alert_conf]
            # np.argmax returns index 0 (Tired) if drowsy is higher, 1 (Alert) if alert is higher
            idx = np.argmax(preds[0])
            return "Tired" if idx == 0 else "Alert"
        except:
            return "Error"

    def load_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.image_list = [os.path.join(folder_selected, f) for f in os.listdir(folder_selected) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.image_list.sort()
            if self.image_list:
                self.current_idx = 0
                self.display_image(self.image_list[self.current_idx])

    def display_image(self, path):
        self.image_path = path
        self.image_id.set(os.path.basename(path))
        self.progress_text.set(f"Img {self.current_idx + 1}/{len(self.image_list)}")
        raw_img = Image.open(path).convert('RGB')
        self.darkness_val.set(getDarkness(raw_img))
        self.prediction_val.set(self.predict_model(raw_img))
        
        self.root.update_idletasks()
        p_w, p_h = self.img_panel.winfo_width(), self.img_panel.winfo_height()
        if p_w < 10: p_w, p_h = 1000, 700
        
        scale = min(p_w / raw_img.size[0], p_h / raw_img.size[1])
        if scale < 1.0:
            display_img = raw_img.resize((int(raw_img.size[0]*scale), int(raw_img.size[1]*scale)), Image.Resampling.LANCZOS)
        else:
            display_img = raw_img
            
        img_tk = ImageTk.PhotoImage(display_img)
        self.img_panel.configure(image=img_tk, text="")
        self.img_panel.image = img_tk 
        #self.reset_specific_inputs()

    def reset_specific_inputs(self):
        self.camera_pos.set(4)
        self.alertness.set("Alert")

    def go_back(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.display_image(self.image_list[self.current_idx])

    def save_and_next(self):
        if not self.image_path: return
        current_id = self.image_id.get()
        new_data = {
            "image_id": current_id,
            "darkness": self.darkness_val.get(),
            "model_prediction": self.prediction_val.get(),
            "camera_pos": self.camera_pos.get(),
            "accessories": int(self.accessories.get()),
            "hair_in_face": int(self.hair_obscured.get()),
            "gender": self.gender.get(),
            "manual_state": self.alertness.get()
        }

        script_dir = Path(__file__).resolve().parent
        file_path = script_dir / 'annotations.csv'
        all_rows = []
        found = False

        if file_path.exists():
            with open(file_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                all_rows = list(reader)
            
            for row in all_rows:
                if row['image_id'] == current_id:
                    row.update(new_data)
                    found = True
                    break

        if not found: all_rows.append(new_data)

        with open(file_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=new_data.keys())
            writer.writeheader()
            writer.writerows(all_rows)

        # Move to next image
        self.current_idx += 1
        if self.current_idx < len(self.image_list):
            self.display_image(self.image_list[self.current_idx])
        else:
            messagebox.showinfo("Done", "Folder finished!")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAnnotator(root)
    root.mainloop()