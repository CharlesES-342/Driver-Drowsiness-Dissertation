# Driver-Drowsiness-Dissertation

This project is my final year dissertation. It includes both final production and exploratory trial code.

The aim of this project is to improve road safety. This is achieved by using an AI model to detect the tiredness of a vehicle driver and act accordingly. This involves notifying them when they are drowsy to suggest they stop, or alternatively, creating an audible alert to help keep them awake.

All findings and the project structure are documented in the [__Report__](Report/Dissertation_report.pdf)

---

## Project Stages

### Before Starting

- **Order Components** - Camera  
  - Raspberry Pi  
  - Power cables  
  - Speaker
  - Micro HDMI cable (optional - can SSH in)

- **Download training data** - **Sort/group training data**

---

### Code Elements

- **Face Detection (Image)**
  - Use a single image to establish basic functionality.
  - Extract data (e.g. eye positions, mouth positions, etc.).

- **Face Detection (Video)**
  - Verify frame-by-frame reading.
  - Extract temporal features (e.g. yawning frequency, eye closure duration, blink rate, etc.).

- **Formulating the AI Structure**
  - Use the collected data elements from videos and images to form your neural network.

- **Implement Learning Algorithms**
  - Extract features from the dataset.
  - Train an appropriate model using your findings (on a data subset).
  - Test performance (on a separate validation subset).

- **Network Wi-Fi Device** *(requires hardware)* - Set up and test a connection to run your Pi as a server hosting a web page.  
  - **Additional:** Attempt to auto-open the browser to the web page upon initial connection to the network.

- **Camera Integration** *(requires hardware)* - Ensure the camera device functions and relays information correctly.  
  - Run facial detection (without utilising the AI model initially).  
  - Pair this with your AI model—can you detect your own tiredness?

- **Deployment on Raspberry Pi** *(requires hardware)*
  - Check Python version (typically 3.8 to 3.11, depending on library requirements).
  - Verify camera access and face detection on the Pi.
  - Perform real-time tiredness detection.
  - Convert models to appropriate formats (e.g. from TensorFlow `.h5` to TensorFlow Lite `.tflite`).

---

## Important File Isolation
Some files within this project are informally labelled or were part of early iterations. To make review easier, here are the key files and their functions:

*Note: This is not a definitive list; depending on your specific area of interest, further exploration of the repository may be required.*

### 1. Initial Face Identification
* `Face_Detection.py`: Utilises **MediaPipe** for real-time facial detection and landmark extraction.
* `pi_detection.py`: Implementation of the detection logic **optimised** for the **Raspberry Pi** environment.

### 2. Data Viability & Trends
* `trends.py`: Analysis script used to determine trends within the data. It assesses whether the captured data is viable for training purposes (detailed further in the full report).

### 3. Simple AI Model (Feature-Based)
This section tracks the evolution from video-based training to refined image-based classification.

#### *Phase A: Initial Video Training (Exploratory)*
* `simpleAI.py`: My first attempt at forming a model using all facial landmarks as inputs. While a significant milestone, this version was eventually superseded.
* `initialAI.py`: Testing and validation script. It provided the **key insight** that "Tired" videos often contain a high percentage of "Alert" frames, leading to classification bias.

#### *Phase B: Refined BINA Dataset Approach*
* `trainingImageAI.py`: Trains a model based on static images and factual classifications from the **BINA University dataset**. Utilises a reduced set of landmarks deduced from research.
* `testingImageAI.py`: Validation script testing the model against ground-truth labels from the BINA dataset.
* `bufferAI.py`: Implements a "Temporal Buffer" over video regions. By using a threshold to determine tiredness over a period of frames, it reduces "Alert frame bias" to achieve more accurate results.

### 4. Data Refinement
* `dataRefinement.py`: A utility developed after noticing inconsistencies in training data; used to audit and re-classify datasets for better accuracy.

### 5. Xception Model (Transfer Learning)
* `Xception.py`: The final version of the Xception-based architecture. It handles the formation and testing of the deep learning model.
* `model_conversion.py`: Handles the transition from training to deployment (e.g. converting `.h5` to `.tflite` and **optimising** weights for edge computing).

### 6. On-Device Deployment (Raspberry Pi)
The following files reside on the Raspberry Pi to handle the live interface and inference:

* `api.py`: The **main communication layer** and entry point. Run this file to start the system on the Pi.
* `camera.py`: A helper utility containing specialised functions for camera stream management.
* `cap.py`: Data capture utility used during the Evaluation phase.
* `structure.txt`: An overview of the files in this location and their purposes.
* `/model`: Directory containing the compiled TFLite models.
* `/templates/index.html`: The web-based dashboard that displays real-time data and alerts.

---

## 🛠 Libraries & Environment Requirements

To run the software in this submission, the following dependencies must be installed in your Python environment:

| Library | Purpose |
| :--- | :--- |
| `tensorflow` / `tflite-runtime` | Model inference and training |
| `mediapipe` | Facial landmark extraction (for initial feature-based models) |
| `opencv-python` | Image processing and camera handling |
| `numpy` | Numerical operations and array handling |
| `flask` | Web server for the Pi dashboard |
| `pandas` | Data analysis (for `trends.py`) |

---
*Created as part of the Year 3 Dissertation Project.*