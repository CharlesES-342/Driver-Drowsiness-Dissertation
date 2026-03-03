import threading
import time
from flask import Flask, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- GLOBAL DATA ---
# This dictionary holds the latest state from your AI model
# Using a dictionary makes it easy to add more metrics later
shared_data = {
    "prediction": "alert",
    "confidence": 0.0,
    "spread": 0.0,
    "last_updated": 0
}

# Lock ensures the thread doesn't update the data at the exact 
# millisecond the API tries to read it
data_lock = threading.Lock()

# --- AI WORKER THREAD ---
def ai_processing_loop():
    """
    This function runs continuously in the background.
    Replace the 'time.sleep' and 'random' logic with your actual AI calls.
    """
    global shared_data
    print("AI Model Thread Started...")
    
    # 1. Initialize your model here (load weights, etc.)
    # model = YourModel().load()

    while True:
        # 2. Capture Frame / Get Data
        # frame = camera.read()
        
        # 3. Run Inference
        # result = model.predict(frame)
        
        # --- MOCK DATA LOGIC (Replace this) --- TODO - remove
        import random
        new_val = random.random()
        new_pred = "drowsy" if new_val > 0.7 else "alert"
        # --------------------------------------

        # 4. Update the global state securely
        with data_lock:
            shared_data["confidence"] = round(new_val, 2)
            shared_data["prediction"] = new_pred
            shared_data["spread"] = round(random.random(), 2)
            shared_data["last_updated"] = time.time()

        # Small sleep to prevent CPU Max-out if the AI is very fast
        # this will also help to keep the device cooler and save power
        time.sleep(0.1)

# --- FLASK ROUTES ---
@app.route('/')
def index():
    """Serves the dashboard UI"""
    return render_template('index.html')

@app.route('/api/data')
def get_data():
    """Returns the latest data from the AI thread"""
    with data_lock:
        return jsonify(shared_data)


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