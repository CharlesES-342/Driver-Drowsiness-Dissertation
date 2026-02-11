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


def create(model_loc):

    # SETTINGS -------------------------
    IMG_SIZE = (299, 299)
    BATCH_SIZE = 16
    EPOCHS = 20

    # locations
    BASE_DIR = "Bina Nusantara University Data/modelling/training/training"
    CSV_PATH = "Bina Nusantara University Data/modelling/training/training/Img_labels_refined.csv"

    # Predefined label space
    LABELS = [
        "eyes_open",
        "eyes_closed",
        "yawning",
        "not_yawning"
    ]

    # labels to index mapping
    label_to_index = {label: i for i, label in enumerate(LABELS)}
    num_classes = len(LABELS)

    # LOAD & ENCODE CSV (MULTI-LABEL)
    df = pd.read_csv(CSV_PATH) #read the csv file

    def encode_labels(label_string): #convert a label (string) into binary represenation (if it has calssification x, it retruns 1 at position x)
        multi_hot = np.zeros(num_classes, dtype=np.float32)
        for lbl in label_string.split(","):
            lbl = lbl.strip() #remove leading/trailing spaces
            if lbl in label_to_index:
                multi_hot[label_to_index[lbl]] = 1.0
        return multi_hot

    df["encoded_labels"] = df["label"].apply(encode_labels)

    file_paths = [os.path.join(BASE_DIR, fname) for fname in df["filename"]]
    labels = np.stack(df["encoded_labels"].values)

    # ============================================================
    # TF.DATA IMAGE LOADER
    # ============================================================
    def load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        #TODO - add in changes to the images here
        img = ip.imageEffects(img) #apply the image effects (augmentations) to the image
        img = preprocess_input(img)
        return img, label

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Train / Validation split
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

    # ============================================================
    # LOAD PRETRAINED XCEPTION
    # ============================================================
    base_model = Xception(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )

    base_model.trainable = False

    # model architecture/settings
    inputs = layers.Input(shape=(299, 299, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    model.summary()

    # COMPILE MODEL
    model.compile(
        optimizer=optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # TRAINING (FROZEN FEATURE EXTRACTION)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # additional training features
    base_model.trainable = True # allow additional training of base model - specialisation

    #keeps the first 200 layers as non-trainable (out of 299)
    for layer in base_model.layers[:200]: 
        layer.trainable = False

    # RE-COMPILE MODEL WITH LOWER LEARNING RATE
    model.compile(
        optimizer=optimizers.Adam(1e-5), # fine tuning
        loss="binary_crossentropy", # each label of classification is independent
        metrics=["accuracy"] # % of label correctly predicted
    )

    # small leaerning rate on later layers (fine tuning) - better for sensitivity
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10
    )

    # ============================================================
    # SAVE MODEL
    # ============================================================
    model.save(model_loc)
    print("Model saved as model_loc, this is the encoding for multiple (more refined) labeling")






def test_with_ground_truth(model_location, image_dir, csv_path):
    # ============================================================
    # LABEL DEFINITIONS (MUST MATCH TRAINING EXACTLY)
    # ============================================================
    LABELS = [
        "eyes_open",
        "eyes_closed",
        "yawning",
        "not_yawning"
    ]

    num_classes = len(LABELS)
    label_to_index = {label: i for i, label in enumerate(LABELS)}
    index_to_label = {i: label for i, label in enumerate(LABELS)}

    # ============================================================
    # LOAD TEST CSV (image, "label1,label2,...")
    # ============================================================
    df_test = pd.read_csv(
        csv_path,
        header=None,
        names=["filename", "labels"]
    )

    print(f"Loaded {len(df_test)} test samples")

    # ============================================================
    # MULTI-LABEL ENCODER
    # ============================================================
    def encode_labels(label_string):
        multi_hot = np.zeros(num_classes, dtype=np.int32)
        for lbl in label_string.split(","):
            lbl = lbl.strip()
            if lbl in label_to_index:
                multi_hot[label_to_index[lbl]] = 1
        return multi_hot

    df_test["encoded_labels"] = df_test["labels"].apply(encode_labels)

    # ============================================================
    # LOAD MODEL
    # ============================================================
    print("\nLoading model...")
    model = tf.keras.models.load_model(model_location)
    print("Model loaded successfully.")
    print(f"Model output size: {model.output_shape[-1]}")

    # ============================================================
    # EVALUATION LOOP
    # ============================================================
    results = []
    matched = 0
    not_found = 0

    for _, row in df_test.iterrows():

        filename = row["filename"]
        true_multi = row["encoded_labels"]

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

        # Apply threshold for multi-label prediction
        threshold = 0.5
        predicted_multi = (predictions >= threshold).astype(int)

        matched += 1

        # Decode labels
        true_labels = [LABELS[i] for i in np.where(true_multi == 1)[0]]
        predicted_labels = [LABELS[i] for i in np.where(predicted_multi == 1)[0]]

        # Exact match check
        is_correct = np.array_equal(true_multi, predicted_multi)

        results.append({
            "filename": filename,
            "true_labels": ",".join(true_labels),
            "predicted_labels": ",".join(predicted_labels),
            "correct": is_correct
        })

        status = "✓" if is_correct else "✗"
        print(f"{status} {filename}")
        print(f"   True: {true_labels}")
        print(f"   Pred: {predicted_labels}")

    # ============================================================
    # RESULTS SUMMARY
    # ============================================================
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

    exact_match_accuracy = results_df["correct"].mean() * 100

    print(f"\nExact-match accuracy: {exact_match_accuracy:.2f}%")
    print(f"Correct predictions: {results_df['correct'].sum()} / {matched}")

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    results_df.to_csv("test_results_multilabel.csv", index=False)
    print("\n✓ Results saved to test_results_multilabel.csv")

    return results_df


def from_camera(model_loc):
    # label mappings - this order is taken from the file used during training
    LABELS = [
        "eyes open",
        "eyes closed",
        "yawning",
        "not yawning"
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
    LABELS = [
        "eyes open",
        "eyes closed",
        "yawning",
        "not yawning"
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

    image_dir = "Bina Nusantara University Data/modelling/testing/testing"
    csv_path = "Bina Nusantara University Data/modelling/testing/testing/Img_labels_refined.csv"


    #=====MAKING MODEL EXAMPLE=====
    # model_location = "Driver-Drowsiness-Dissertation/AI_Models/xception_drowsiness_model_v2.h5"
    # model = tf.keras.models.load_model(model_location)
    # model.summary()

    #=====TESTING MODEL WITH GROUND TRUTH EXAMPLE=====
    # model_location = "Driver-Drowsiness-Dissertation/AI_Models/xception_drowsiness_model_v3_imageChanges.h5"
    # model = tf.keras.models.load_model(model_location)
    # test_with_ground_truth(model_location, image_dir, csv_path)

    #=====FROM CAMERA EXAMPLE=====
    model_location = "Driver-Drowsiness-Dissertation/AI_Models/xception_drowsiness_model_v3_imageChanges.h5"
    from_camera(model_location)