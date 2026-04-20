'''
Contains the funcitons to make an Xception model, test on ground truths
as well as test on camera feed
'''
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing import image
import time
import numpy as np
import pandas as pd
import cv2
import os

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import utilities.imageProcessing as ip


def create(model_loc, BASE_DIR, CSV_PATH):
    '''
    Create an xception model for SINGLE-LABEL classification (model give 2 confidences as a
    result, choose the higher one as the class that it is more likely to be -> drowsy_confidence,
    alert_confidence)
    Input: storage location, base directory, path to training data (csv of classifications)
    Output: none -> model saveed to location
    '''
    
    IMG_SIZE = (299, 299)
    BATCH_SIZE = 16
    EPOCHS = 20

    # Single-label setup
    LABELS = ["drowsy", "alert"]
    label_to_index = {label: i for i, label in enumerate(LABELS)}
    num_classes = len(LABELS)

    # LOAD & ENCODE CSV (SINGLE-LABEL)
    df = pd.read_csv(CSV_PATH)

    def encode_single_label(label_string):
        """Convert single label to integer index"""
        label = label_string.strip()
        return label_to_index.get(label, 0)  # Default to 'drowsy' if unknown

    # Convert to categorical (one-hot encoding)
    df["label_index"] = df["label"].apply(encode_single_label)
    
    file_paths = [os.path.join(BASE_DIR, fname) for fname in df["filename"]]
    labels = tf.keras.utils.to_categorical(df["label_index"].values, num_classes)

    # TF.DATA IMAGE LOADER
    def load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = ip.imageEffects(img)
        img = preprocess_input(img)
        return img, label

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Train / Validation split with stratification if possible
    train_size = int(0.8 * len(df))
    train_ds = (
        dataset
        .take(train_size)
        .shuffle(500)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        dataset
        .skip(train_size)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # load pretrained Xception
    base_model = Xception(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base_model.trainable = False

    # model architectre
    inputs = layers.Input(shape=(299, 299, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    # softmax for single-label classification
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.summary()

    # compile model
    model.compile(
        optimizer=optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # training - frozen feature extraction
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # model fine-tuning
    base_model.trainable = True
    for layer in base_model.layers[:200]: 
        layer.trainable = False

    # re-compile with added changes to model structure
    model.compile(
        optimizer=optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10
    )

    # save the model to the location
    model.save(model_loc)
    print(f"Model saved to {model_loc}")






def test_with_ground_truth(model_location, image_dir, csv_path):
    '''
    Test single-label classifier against the pre-determined classified images
    Input: model location, image directory (where are the images to test on), ground-truth path (image
    labels)
    Ouput: Dic{
            "filename": filename,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "correct": is_correct
            }
            -> and terminal output for an overview on how well the model did across all the frames
    '''
    
    LABELS = ["drowsy", "alert"]
    num_classes = len(LABELS)
    label_to_index = {label: i for i, label in enumerate(LABELS)}

    # Load test CSV
    df_test = pd.read_csv(csv_path, header=None, names=["filename", "labels"])
    print(f"Loaded {len(df_test)} test samples")

    # Load model
    print("\nLoading model...")
    model = tf.keras.models.load_model(model_location)
    print("Model loaded successfully.")

    results = []
    matched = 0
    not_found = 0

    # go through every row
    for _, row in df_test.iterrows():
        filename = row["filename"]
        true_label = row["labels"].strip()

        image_path = os.path.join(image_dir, filename)

        if not os.path.isfile(image_path):
            not_found += 1
            continue

        # Load image
        img = image.load_img(image_path, target_size=(299, 299))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict
        predictions = model.predict(img_array, verbose=0)[0]
        
        # Get predicted class (highest probability)
        predicted_index = np.argmax(predictions)
        predicted_label = LABELS[predicted_index]
        confidence = predictions[predicted_index] * 100

        matched += 1

        # Check if correct
        is_correct = (predicted_label == true_label)

        results.append({
            "filename": filename,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "correct": is_correct
        })

        status = "✓" if is_correct else "✗"
        print(f"{status} {filename}")
        print(f"   True: {true_label}")
        print(f"   Pred: {predicted_label} ({confidence:.1f}%)")

    # RESULTS SUMMARY
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Images in CSV: {len(df_test)}")
    print(f"Images evaluated: {matched}")
    print(f"Images not found: {not_found}")

    if len(results_df) == 0:
        print("❌ No valid results to evaluate.")
        return results_df

    accuracy = results_df["correct"].mean() * 100
    print(f"\nAccuracy: {accuracy:.2f}%")
    print(f"Correct predictions: {results_df['correct'].sum()} / {matched}")
    
    # Per-class accuracy
    for label in LABELS:
        label_df = results_df[results_df["true_label"] == label]
        if len(label_df) > 0:
            label_acc = label_df["correct"].mean() * 100
            print(f"  {label}: {label_acc:.2f}% ({label_df['correct'].sum()}/{len(label_df)})")

    # Save results
    results_df.to_csv("test_results_single_label.csv", index=False)
    print("\n✓ Results saved to test_results_single_label.csv")

    return results_df


def from_camera(model_loc):
    '''
    test the model on a live webcam feed, printing the predicted labels and confidence
    for each frame
    Input: model locaiton
    Output: none -> print predicted labels and confidences in terminal (for each frame)
    '''
    # This is the label order used during training, must match exactly for correct interpretation
    # label mappings - this order is taken from the file used during training
    LABELS = [
        "drowsy",
        "alert"
    ]

    #changing project root to this so it can find the utility files
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.append(str(ROOT))
    
    #suporting functions
    import utilities.cameraUtility as cu
    import utilities.imageProcessing as ip

    cap = cu.open_webcam()

    # Load model
    model = tf.keras.models.load_model(model_loc)
    while True:
        frame = cu.capture_frame(cap)
        startTime = time.time()
        if frame is None:
            continue
        cv2.imshow('Webcam Feed', frame)
        #refresh webcam feed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        #send the image to AI model
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (299, 299))
        img = img.astype("float32")
        img_array = preprocess_input(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)[0]
        threshold = 0.5  # adjust depending on model confidence

        active = [
            (LABELS[i], predictions[i] * 100)
            for i in range(len(predictions))
            if predictions[i] >= threshold
        ]
        #confidence for each label
        for i, score in enumerate(predictions):
            print(f"{LABELS[i]}: {score:.2f}")

        predicted_label = ", ".join([f"{lbl} ({conf:.2f}%)" for lbl, conf in active])

        endTime = time.time()
        fps = 1 / (endTime - startTime)

        print(f"Predicted: {predicted_label:<30s} FPS: {fps:6.2f}")
        
        #slow down for visibility
        time.sleep(10)


def from_camera_individual(model_loc):
    '''
    test the model on a live webcam feed, printing the predicted labels and confidence on an individual frame
    using SPACE to capture an image
    Input: model location
    Output: none -> print the preicted labels and confidences for each image
    '''
    # LABELS = [
    #     "eyes open",
    #     "eyes closed",
    #     "yawning",
    #     "not yawning"
    # ]
    LABELS = [
        "drowsy",
        "alert"
    ]

    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.append(str(ROOT))
    
    import utilities.cameraUtility as cu
    import utilities.imageProcessing as ip

    cap = cu.open_webcam()
    model = tf.keras.models.load_model(model_loc)
    print("waiting on input") 
    while True:
        # Show the live feed continuously
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow('Webcam Feed', frame)
        
        # Get key press ONCE
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            # Process the current frame
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (299, 299))
            img = img.astype("float32")
            img_array = preprocess_input(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            predictions = model.predict(img_array, verbose=0)[0]
            threshold = 0.5
            
            active = [
                (LABELS[i], predictions[i] * 100)
                for i in range(len(predictions))
                if predictions[i] >= threshold
            ]
            
            # Print confidence for each label
            for i, score in enumerate(predictions):
                print(f"{LABELS[i]}: {score:.2f}")
            
            predicted_label = ", ".join([f"{lbl} ({conf:.2f}%)" for lbl, conf in active])
            print(f"Predicted: {predicted_label}")
    
    cap.release()
    cv2.destroyAllWindows()
            

def original_classes():
    '''
    Go through the file defined and get all of the labels that are given
    This was used to check the actual outputs from the dataset as I began to feel that there
    was an issue with the dataset
    '''
    labels = []
    #go through dataset and get all the classes
    csv_path = "Bina Nusantara University Data/modelling/training/training/Img_labels.csv"
    df = pd.read_csv(csv_path) 
    for label in df['label'].unique():
        labels.append(label)
    return labels






if __name__ == "__main__":

    import os
    print("Current working directory:", os.getcwd())
    print("Files in directory:", os.listdir())


    #=====MAKING MODEL EXAMPLE=====
    # model_location = "Driver-Drowsiness-Dissertation/AI_Models/xception_drowsiness_model_v6.h5"
    # training_data_dir = "Bina Nusantara University Data/modelling/training/training"
    # training_csv = "Bina Nusantara University Data/modelling/training/training/manual_labels.csv"

    # create(model_location, training_data_dir, training_csv)
    # model = tf.keras.models.load_model(model_location)
    # model.summary()




    #=====TESTING MODEL WITH GROUND TRUTH EXAMPLE=====
    model_location = "Driver-Drowsiness-Dissertation/AI_Models/xception_drowsiness_model_v6.h5"
    model = tf.keras.models.load_model(model_location)
    image_dir = "Driver-Drowsiness-Dissertation/TestingData/Personal Data" #"Bina Nusantara University Data/modelling/testing/testing"
    csv_path = "Driver-Drowsiness-Dissertation/TestingData/Personal Data/personal_labels.csv"

    testinging_data_dir = "Bina Nusantara University Data/modelling/testing/testing"
    testinging_csv = "Bina Nusantara University Data/modelling/testing/testing/manual_labels.csv"

    test_with_ground_truth(model_location, testinging_data_dir, testinging_csv)




    #=====FROM CAMERA EXAMPLE=====
    # model_location = "Driver-Drowsiness-Dissertation/AI_Models/xception_drowsiness_model_v3_imageChanges.h5"
    # from_camera_individual(model_location)