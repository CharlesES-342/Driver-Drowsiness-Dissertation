# Driver-Drowsiness-Dissertation

This project is my final year dissertation. This will include both used and trial code.

The aim of this project is to improve safety on the road. This will be done by using an AI model to detect the tiredness of  
the driver of a vehicle and act accordingly. This could be notifying them when they are drowsy and suggest they stop, or  
alternatively create an annoying beep to keep them awake.

---

## Project Stages

### Before Starting

- **Order Components**
  - Camera  
  - Raspberry Pi  
  - Power cables  
  - Speaker
  - Micro HDMI cable (optional - can ssh in)

- **Download training data**  
- **Sort/group training data**

---

### Code Elements

- **Face Detection (Image)**
  - Can use a single image for this to get the basics down  
  - Extract data (e.g. eye positions, mouth positions, etc.)

- **Face Detection (Video)**
  - Check that you can read each frame  
  - Extract resources (e.g. yawning, time eyes are closed, blink rate, etc.)

- **Formulating the AI Structure**
  - Use the collected data elements from the videos and images to form your node network  

- **Implement Learning Algorithms**
  - Back propagation  
  - Momentum  
  - Bold driver / annealing (choose one)  
  - Weight decay  

- **Network Wi-Fi Device** *(requires hardware)*  
  - Set up and test a connection to run your Pi as a server hosting a web page  
  - **Additional:** Try and have it auto open your browser to the webpage on initial connection to the network  

- **Make Use of Camera** *(requires hardware)*  
  - Ensure that your camera device works and relays information  
  - Try running your facial detection (not utilising the AI model yet)  
  - Pair this with your AI model
