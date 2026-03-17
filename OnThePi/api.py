import threading
import time
from flask import Flask, jsonify, render_template, make_response
from flask_cors import CORS
from collections import deque
import camera as cam
import numpy as np
import tflite_runtime.interpreter as tflite

app = Flask(__name__)
CORS(app)

def preprocess_input(frame):
    import cv2
    img = cv2.resize(frame, (299, 299)) # Xception standard size
    img = img.astype(np.float32) / 255.0 # Normalize 0 to 1
    return img

# --- GLOBAL DATA ---
# holds latest state of the model predictions (and bonus data)
shared_data = {
    "prediction": "alert",
    "confidence": 0.0,
    "spread": 0.0,
    "last_updated": 0
}

# Lock ensures the thread doesn't update the data at the exact time the API tries to read it
data_lock = threading.Lock()

# --- AI WORKER THREAD ---
def ai_processing_loop():
    global shared_data
    print("AI Model Thread Started...")
    
    # 1. FIXED: Use the correct alias from your import
    try:
        interpreter = tflite.Interpreter(model_path="model/xception_drowsiness_model_v6.tflite")
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    frame_buffer = deque(maxlen=30) 
    conf_buffer = deque(maxlen=30) 

    while True:
        try:
            frame = cam.getFrame()
            if frame is None:
                print("frame failed")
                continue

            # 2. PRO-TIP: Ensure preprocess_input exists! 
            # It should resize to the model's expected shape (e.g., 299, 299, 3)
            img_ready = preprocess_input(frame) 
            img_batch = np.expand_dims(img_ready, axis=0).astype(np.float32)

            interpreter.set_tensor(input_details[0]['index'], img_batch)
            interpreter.invoke()


            # 3. FIXED: Consistently use 'preds'
            preds = interpreter.get_tensor(output_details[0]['index'])[0]


            # Index 0 = Drowsy, Index 1 = Alert (based on your comment)
            is_drowsy = int(preds[0] > preds[1])
            frame_buffer.append(is_drowsy)
            
            current_conf = float(max(preds[0], preds[1]))
            conf_buffer.append(current_conf)

            if len(frame_buffer) > 0:
                spread_val = sum(frame_buffer) / len(frame_buffer)
                avg_conf = sum(conf_buffer) / len(conf_buffer)
                
                # Threshold logic
                new_pred = "drowsy" if spread_val > 0.5 else "alert"
                
                with data_lock:
                    shared_data["prediction"] = new_pred
                    shared_data["confidence"] = round(avg_conf, 2)
                    shared_data["spread"] = round(spread_val, 2)
                    shared_data["last_updated"] = time.time()
        
        except Exception as e:
            print(f"Error in AI Loop: {e}")

        time.sleep(0.1)

        time.sleep(0.1)

# --- FLASK ROUTES ---
@app.route('/')
def index():
    """Serves the dashboard UI"""
    return render_template('index.html')
'''
@app.route('/api/data')
def get_data():
    """Returns the latest data from the AI thread"""
    with data_lock:
        return jsonify(shared_data)
'''
@app.route('/api/data')
def get_data():
    with data_lock:
        response = make_response(jsonify(shared_data))
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response

#===== for setup of the pi camera in the vehicle =====
# @app.route('/setup')
# def setup():
#     """Serves the setup page for initial configuration"""
#     return render_template('setup.html')

# @app.route('/api/setup')
# def get_img():
#     """Endpoint to receive setup images or data from the client"""
#     # Here you would handle the incoming data, save it, and possibly update the AI model
#     return jsonify({"status": "success", "message": "Setup data received"})



if __name__ == '__main__':
    # Start the AI thread before launching Flask
    ai_thread = threading.Thread(target=ai_processing_loop, daemon=True)
    ai_thread.start()

    # Launch Flask (Debug=False is safer when using threading)
    app.run(host='0.0.0.0', port=5000, debug=False)
