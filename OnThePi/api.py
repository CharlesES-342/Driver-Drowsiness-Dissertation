import threading
import time
from flask import Flask, jsonify, render_template
from flask_cors import CORS
from collections import deque
import camera as cam
import numpy as np
import tflite_runtime.interpreter as tflite

app = Flask(__name__)
CORS(app)

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
    
    #Load model from memory
    interpreter = tf.lite.Interpreter(model_path="model/xception_drowsiness_model_v6.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    #buffer definitions
    frame_buffer = deque(maxlen=30) #value of each prediction
    conf_buffer = deque(maxlen=30) #confidence of each prediction

    while True:
        try:
            frame = cam.getFrame()
            #incase there is an issue
            if frame is None:
                continue

            #pre-processing
            img_ready = preprocess_input(frame.astype('float32'))
            img_batch = np.expand_dims(img_ready, axis=0)

            #input the image
            interpreter.set_tensor(input_details[0]['index'], img_batch)

            #start the model
            interpreter.invoke()

            #extract the results
            preds = interpreter.get_tensor(output_details[0]['index'])[0]


                # => models output: [drowsy_confidence, alert_confidence]
            # add the overall classificaiton to the list of classificaitons
            frame_buffer.append(int(pred[0] > pred[1]))
            #add the confidence to the array to be used for average confidence later
            current_conf = max(pred[0], pred[1])
            conf_buffer.append(current_conf)

            buffer_occupancy = len(frame_buffer)

            #variable assignment (for displaying)
            if buffer_occupancy > 0:
                spread_val = sum(frame_buffer) / buffer_occupancy
                #overall state based on spread
                threshold = 0.5 #TODO - change this as needed
                new_pred = "drowsy" if spread_val > threshold else "alert"
                avg_conf = sum(conf_buffer) / len(conf_buffer)
                
                with data_lock:
                    shared_data["prediction"] = new_pred
                    shared_data["confidence"] = round(avg_conf, 2)
                    shared_data["spread"] = round(spread_val, 2)
                    shared_data["last_updated"] = time.time()
        
        except Exception as e:
            print(f"Error in AI Loop: {e}")

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