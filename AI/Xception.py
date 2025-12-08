import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import os


def create():
    # ============================================================
    # SETTINGS
    # ============================================================
    IMG_SIZE = (299, 299)
    BATCH_SIZE = 16
    EPOCHS = 20

    BASE_DIR = "Bina Nusantara University Data/modelling/training/training"
    CSV_PATH = "Bina Nusantara University Data/modelling/training/training/Img_labels.csv"

    # Load labels CSV
    df = pd.read_csv(CSV_PATH)

    # Convert labels → category integers
    df["label_id"] = df["label"].astype("category").cat.codes
    num_classes = df["label_id"].nunique()

    # Full file paths
    file_paths = [os.path.join(BASE_DIR, fname) for fname in df["filename"]]
    labels = df["label_id"].values

    # ============================================================
    # TF.DATA IMAGE LOADER (REPLACES flow_from_directory)
    # ============================================================
    def load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = preprocess_input(img)
        return img, tf.one_hot(label, num_classes)

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Split into train/val
    train_size = int(0.8 * len(df))
    train_ds = dataset.take(train_size).shuffle(500).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = dataset.skip(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # ============================================================
    # LOAD PRETRAINED XCEPTION
    # ============================================================
    base_model = Xception(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )

    base_model.trainable = False  # freeze for feature extraction

    # ============================================================
    # CLASSIFIER HEAD
    # ============================================================
    inputs = layers.Input(shape=(299, 299, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.summary()

    # ============================================================
    # COMPILE MODEL
    # ============================================================
    model.compile(
        optimizer=optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # ============================================================
    # TRAINING (FROZEN FEATURE EXTRACTION)
    # ============================================================
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # ============================================================
    # OPTIONAL FINE TUNING
    # ============================================================
    base_model.trainable = True

    for layer in base_model.layers[:200]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history_ft = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10
    )

    # ============================================================
    # SAVE MODEL
    # ============================================================
    model.save("xception_drowsiness_model.h5")
    print("Model saved as xception_drowsiness_model.h5")





def test_with_ground_truth(model_location, image_dir, csv_path):
    # Load the TEST CSV
    df_test = pd.read_csv(csv_path, header=None, names=['filename', 'label'])
    print(f"Total labels in TEST CSV: {len(df_test)}")
    print(f"Unique labels in TEST: {sorted(df_test['label'].unique())}")
    print(f"\nLabel distribution:\n{df_test['label'].value_counts()}")
    
    # Load TRAINING CSV to get the exact same label mapping
    train_csv = "Bina Nusantara University Data/modelling/training/training/Img_labels.csv"
    df_train = pd.read_csv(train_csv)
    
    print(f"\nUnique labels in TRAINING: {sorted(df_train['label'].unique())}")
    
    # Recreate EXACT same mapping as training
    df_train["label_id"] = df_train["label"].astype("category").cat.codes
    
    # Get the category mapping - THIS IS THE KEY FIX
    categories = df_train["label"].astype("category").cat.categories
    id_to_label = {i: label for i, label in enumerate(categories)}
    label_to_id = {label: i for i, label in enumerate(categories)}
    
    print(f"\nLabel mapping from training data:")
    for idx, label in sorted(id_to_label.items()):
        print(f"  {idx}: {label}")
    
    # Check if test labels exist in training
    test_labels = set(df_test['label'].unique())
    train_labels = set(df_train['label'].unique())
    missing = test_labels - train_labels
    if missing:
        print(f"\n⚠️  WARNING: Test set has labels not in training: {missing}")
        print("These images will be skipped!")
    
    # Load model
    print("\nLoading model...")
    model = tf.keras.models.load_model(model_location)
    print("Model loaded successfully.")
    print(f"Model expects {model.output_shape[-1]} classes")
    
    results = []
    matched = 0
    not_found = 0
    label_mismatch = 0
    
    for filename in os.listdir(image_dir):
        if not filename.endswith('.jpg'):
            continue
            
        # Find the ground truth label from CSV
        row = df_test[df_test['filename'] == filename]
        
        if row.empty:
            not_found += 1
            continue
        
        true_label = row['label'].values[0]
        
        # Check if this label exists in training
        if true_label not in label_to_id:
            label_mismatch += 1
            print(f"Skipping {filename}: label '{true_label}' not in training set")
            continue
        
        matched += 1
        
        # Process image
        image_path = os.path.join(image_dir, filename)
        
        try:
            img = image.load_img(image_path, target_size=(299, 299))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # Predict
            predictions = model.predict(img_array, verbose=0)[0]
            predicted_class_id = np.argmax(predictions)
            
            # Check if predicted ID is valid
            if predicted_class_id not in id_to_label:
                print(f"Error: Model predicted invalid class {predicted_class_id} for {filename}")
                continue
                
            predicted_label = id_to_label[predicted_class_id]
            predicted_confidence = predictions[predicted_class_id] * 100
            
            # Check if correct
            is_correct = (predicted_label == true_label)
            
            results.append({
                'filename': filename,
                'true_label': true_label,
                'predicted_label': predicted_label,
                'confidence': predicted_confidence,
                'correct': is_correct
            })
            
            status = "✓" if is_correct else "✗"
            if not is_correct or matched <= 10:  # Show first 10 and all errors
                print(f"{status} {filename}: True={true_label}, Pred={predicted_label}, Conf={predicted_confidence:.2f}%")
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total images in directory: {len([f for f in os.listdir(image_dir) if f.endswith('.jpg')])}")
    print(f"Images not found in CSV: {not_found}")
    print(f"Images with unknown labels: {label_mismatch}")
    print(f"Images successfully tested: {matched}")
    
    if len(results_df) == 0:
        print("\n❌ No results to evaluate! Check the warnings above.")
        return results_df
    
    if matched > 0:
        overall_accuracy = (results_df['correct'].sum() / matched) * 100
        
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")
        print(f"Correct predictions: {results_df['correct'].sum()}/{matched}")
        
        # Per-class accuracy
        print(f"\n{'='*60}")
        print(f"PER-CLASS ACCURACY")
        print(f"{'='*60}")
        for label in sorted(results_df['true_label'].unique()):
            class_data = results_df[results_df['true_label'] == label]
            class_acc = (class_data['correct'].sum() / len(class_data)) * 100
            print(f"{label:20s}: {class_acc:6.2f}% ({class_data['correct'].sum()}/{len(class_data)})")
        
        # Confidence stats
        print(f"\n{'='*60}")
        print(f"CONFIDENCE STATISTICS")
        print(f"{'='*60}")
        print(f"Average confidence (all): {results_df['confidence'].mean():.2f}%")
        correct_conf = results_df[results_df['correct']==True]['confidence'].mean()
        print(f"Average confidence (correct): {correct_conf:.2f}%")
        
        wrong = results_df[results_df['correct'] == False]
        if len(wrong) > 0:
            wrong_conf = wrong['confidence'].mean()
            print(f"Average confidence (wrong): {wrong_conf:.2f}%")
            
            print(f"\n{'='*60}")
            print(f"MISCLASSIFICATIONS (showing first 20)")
            print(f"{'='*60}")
            for idx, (_, row) in enumerate(wrong.head(20).iterrows()):
                print(f"{row['filename']}: {row['true_label']} → {row['predicted_label']} ({row['confidence']:.1f}%)")
            if len(wrong) > 20:
                print(f"... and {len(wrong) - 20} more misclassifications")
    
    results_df.to_csv("test_results.csv", index=False)
    print(f"\n✓ Results saved to test_results.csv")
    
    return results_df

if __name__ == "__main__":
    model_loc = "Driver-Drowsiness-Dissertation/AI_Models/xception_drowsiness_model.h5"    
    print("Testing model at:", model_loc)
    import os
    print("Current working directory:", os.getcwd())
    print("Files in directory:", os.listdir())
    #create()

    image_dir = "Bina Nusantara University Data/modelling/testing/testing"
    csv_path = "Bina Nusantara University Data/modelling/testing/testing/Img_labels.csv"
    results = test_with_ground_truth(model_loc, image_dir, csv_path)
    print("\nFinal Results DataFrame:")
    print(results)
