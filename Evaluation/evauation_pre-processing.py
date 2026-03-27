'''
This will be used to iterate though images and apply labels/classificaitons to them to then
combine into one CSV file which will be used in Ornage to help identify features such as
areas of strength or weakness within the model.
Most of the code in this file was formed by Gemini.ai
Prompt:
please improve this so that I can have a window open for an image to allow me to select boxes (on the right) which will then be converted to a CSV file input.
the options I want to add are:
    previous imputs were my helper fuctions
camera position (selected from a 3x3 grid of tick boxes), tick box if they are wearing accesories,
tick box for if their hair is infrount of their face, ratio for gender, image ID at teh top (will
also need to be stored) a ratio selector for alert or tired)
'''
import pandas as pd
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import csv

# Helper function for determining the darkness of an image
def getDarkness(img):
    """
    input = PIL Image
    output = value between 0-1 (0=White, 1=Black)
    """
    # Convert PIL image to grayscale and then numpy array
    grayscale = img.convert('L')
    data = np.array(grayscale)
    avg_brightness = np.mean(data)
    # 1.0 is pure black, 0.0 is pure white
    darkness = 1.0 - (avg_brightness / 255.0)
    return round(darkness, 4)

class ImageAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Annotation Tool")
        
        # Load your model here if available
        # self.model = tf.keras.models.load_model('path_to_model.h5')
        self.model = None 
        
        # Data storage
        self.image_path = ""
        self.image_list = []
        self.current_idx = -1
        
        # Observable Variables for UI
        self.image_id = tk.StringVar(value="None")
        self.camera_pos = tk.IntVar(value=-1)
        self.accessories = tk.BooleanVar()
        self.hair_obscured = tk.BooleanVar()
        self.gender = tk.StringVar(value="Other")
        self.alertness = tk.StringVar(value="Alert")
        self.darkness_val = tk.DoubleVar(value=0.0)
        self.prediction_val = tk.StringVar(value="N/A")

        self.setup_ui()

    def setup_ui(self):
        # --- Left Side: Image Display ---
        self.img_panel = tk.Label(self.root, text="No Image Loaded", width=60, height=30, bg="gray")
        self.img_panel.pack(side="left", padx=10, pady=10)

        # --- Right Side: Controls ---
        controls = tk.Frame(self.root)
        controls.pack(side="right", fill="y", padx=20, pady=10)

        # 1. Image Info & Automated Stats
        tk.Label(controls, text="Image ID:", font=('Arial', 10, 'bold')).pack(anchor="w")
        tk.Label(controls, textvariable=self.image_id, fg="blue").pack(anchor="w")
        
        stats_frame = tk.LabelFrame(controls, text="Automated Analysis", padx=5, pady=5)
        stats_frame.pack(fill="x", pady=10)
        
        tk.Label(stats_frame, text="Darkness:").grid(row=0, column=0, sticky="w")
        tk.Label(stats_frame, textvariable=self.darkness_val).grid(row=0, column=1, sticky="w")
        
        tk.Label(stats_frame, text="Model Pred:").grid(row=1, column=0, sticky="w")
        tk.Label(stats_frame, textvariable=self.prediction_val, fg="red").grid(row=1, column=1, sticky="w")

        # 2. Camera Position
        tk.Label(controls, text="Camera Position (3x3):", font=('Arial', 10, 'bold')).pack(anchor="w")
        grid_frame = tk.Frame(controls)
        grid_frame.pack(pady=5)
        for i in range(9):
            tk.Radiobutton(grid_frame, variable=self.camera_pos, value=i).grid(row=i//3, column=i%3)

        # 3. Toggles
        tk.Checkbutton(controls, text="Wearing Accessories", variable=self.accessories).pack(anchor="w")
        tk.Checkbutton(controls, text="Hair in Front of Face", variable=self.hair_obscured).pack(anchor="w")

        # 4. Feature Selection
        tk.Label(controls, text="Gender:", font=('Arial', 10, 'bold')).pack(anchor="w", pady=(10,0))
        for g in ["Male", "Female", "Other"]:
            tk.Radiobutton(controls, text=g, variable=self.gender, value=g).pack(anchor="w")

        tk.Label(controls, text="State:", font=('Arial', 10, 'bold')).pack(anchor="w", pady=(10,0))
        for state in ["Alert", "Tired"]:
            tk.Radiobutton(controls, text=state, variable=self.alertness, value=state).pack(anchor="w")

        # 5. Action Buttons
        tk.Button(controls, text="Open Folder", command=self.load_folder, bg="#ddd").pack(fill="x", pady=(20, 5))
        tk.Button(controls, text="Save & Next", command=self.save_and_next, bg="lightgreen", font=('Arial', 10, 'bold')).pack(fill="x", pady=5)

    def predict_model(self, pil_img):
        """
        Placeholder for model prediction logic.
        """
        if self.model is None:
            return "No Model"
        
        # Example preprocessing (standard for many TFLite/Keras models)
        img_resized = pil_img.resize((224, 224)) 
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = self.model.predict(img_array)
        # Assuming binary classification for Alert/Tired
        return "Alert" if prediction[0][0] > 0.5 else "Tired"

    def load_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.image_list = [os.path.join(folder_selected, f) for f in os.listdir(folder_selected) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.image_list.sort()
            if self.image_list:
                self.current_idx = 0
                self.display_image(self.image_list[self.current_idx])
            else:
                messagebox.showwarning("Empty", "No images found in folder.")

    def display_image(self, path):
        self.image_path = path
        self.image_id.set(os.path.basename(path))
        
        # Open image
        raw_img = Image.open(path)
        
        # Run Automated Functions
        self.darkness_val.set(getDarkness(raw_img))
        self.prediction_val.set(self.predict_model(raw_img))
        
        # Resize for GUI
        display_img = raw_img.copy()
        display_img.thumbnail((500, 500))
        img_tk = ImageTk.PhotoImage(display_img)
        
        self.img_panel.configure(image=img_tk, text="")
        self.img_panel.image = img_tk
        self.reset_inputs()

    def reset_inputs(self):
        self.camera_pos.set(-1)
        self.accessories.set(False)
        self.hair_obscured.set(False)
        self.gender.set("Other")
        self.alertness.set("Alert")

    def save_and_next(self):
        if not self.image_path:
            return

        # Prepare Data Row
        data = {
            "image_id": self.image_id.get(),
            "darkness": self.darkness_val.get(),
            "model_prediction": self.prediction_val.get(),
            "camera_pos": self.camera_pos.get(),
            "accessories": int(self.accessories.get()),
            "hair_in_face": int(self.hair_obscured.get()),
            "gender": self.gender.get(),
            "manual_state": self.alertness.get()
        }

        # Write to CSV
        file_name = 'annotations.csv'
        file_exists = os.path.isfile(file_name)
        with open(file_name, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)

        # Move to Next
        self.current_idx += 1
        if self.current_idx < len(self.image_list):
            self.display_image(self.image_list[self.current_idx])
        else:
            messagebox.showinfo("Done", "All images in folder annotated!")
            self.image_path = ""
            self.img_panel.configure(image='', text="Finished!")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAnnotator(root)
    root.mainloop()