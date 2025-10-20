# Driver-Drowsiness-Dissertation
This project is my final year dissertation. This will include both used and trial code.

The aim of this project is to improve safety on teh road. This will be done by using an AI model to detect the tiredness of
the driver of a vehicle and act accordingly. This could be notifying them when they are drowsy and suggest they stop, or
alternitively create an annoying beep to keep them awake.

Project Stages:
    Before starting
        .Order Components
            - camera
            - pi
            - power cables
            - speaker

        .Download training data
        
        .Sort/group training data



    Code Elements
        .Face detection (image)
            - can use a single image for this, to get the basics down
            - extract data (e.g. eye positions, mouth positions...)

        .Face detection (video)
            - chewck that you can read each frame
            - extract resources (e.g. yawning, time eyes are closed, blink rate...)

        .Formulating the AI structure
            use the collected date elements from teh videos and images to form your node network

        .Implement learning algorithms
            - back propagation
            - momentum
            - bold driver / annealing (choose one)
            - weight decay

        .network WIFI device (requires hardware)
            set up and test a connection to run your pi as a server to host a web page
            ADDITIONAL: try and have it auto open your browser to the webpage on initial conneciton to the network
        
        .make use of camera (requires hardware)
            - ensure that your camera device works and relays information
            - try running your facial detection (not utilising the AI model yet)
            - pair this with your AI model