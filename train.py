"""
Training script for the DeepFake detection model.
"""

import os
import warnings
import logging

# Suppress all TensorFlow warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=exclude INFO, 2=exclude INFO+WARNING, 3=exclude all
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

# Suppress TensorFlow logging before importing
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('tensorflow').disabled = True

from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import data_loader
import model


# Training hyperparameters
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
INPUT_SHAPE = (224, 224, 3)

# Data paths (update these to match your dataset locations)
IMAGE_DATA_DIR = "./data/images"
VIDEO_DATA_DIR = "./data/videos"  # Optional
VIDEO_METADATA_PATH = None  # Optional: path to CSV with video labels

# Model save path
MODEL_SAVE_DIR = "./models"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "deepfake_detector_best.h5")


def main():
    """Main training function."""
    
    # Create models directory if it doesn't exist
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    print("=" * 60)
    print("DeepFake Detection Model Training")
    print("=" * 60)
    
    # Check if data directory exists
    if not os.path.exists(IMAGE_DATA_DIR):
        print(f"\n[ERROR] Image data directory not found: {IMAGE_DATA_DIR}")
        print("\nPlease create the following directory structure:")
        print(f"  {IMAGE_DATA_DIR}/")
        print(f"    +-- real/")
        print(f"    |   +-- image1.jpg")
        print(f"    |   +-- ...")
        print(f"    +-- fake/")
        print(f"        +-- image1.jpg")
        print(f"        +-- ...")
        print("\nDownload datasets from:")
        print("  - Images: https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images")
        print("  - Videos: https://www.kaggle.com/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset")
        return
    
    # Check if subdirectories exist
    real_dir = os.path.join(IMAGE_DATA_DIR, 'real')
    fake_dir = os.path.join(IMAGE_DATA_DIR, 'fake')
    
    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        print(f"\n[ERROR] Required subdirectories not found in {IMAGE_DATA_DIR}")
        print("Please create 'real' and 'fake' subdirectories with your images.")
        return
    
    # Load data
    print("\n[1/4] Loading data...")
    try:
        X_train, X_val, y_train, y_val = data_loader.load_combined_data(
            image_dir=IMAGE_DATA_DIR,
            video_dir=None,  # Skip video data for now
            video_metadata_path=VIDEO_METADATA_PATH,
            num_frames_per_video=1,
            max_video_faces=2000,
            test_size=0.2,
            random_state=42
        )
    except ValueError as e:
        if "n_samples=0" in str(e):
            print(f"\n[ERROR] No data found in {IMAGE_DATA_DIR}")
            print("\nPossible issues:")
            print("  1. The 'real' and 'fake' directories are empty")
            print("  2. Images don't contain detectable faces")
            print("  3. Image formats are not supported (use .png, .jpg, .jpeg)")
            print("\nPlease ensure you have:")
            print(f"  - Images in {real_dir}/")
            print(f"  - Images in {fake_dir}/")
            print("  - Images contain clear, visible faces")
            print("\nDownload datasets from:")
            print("  - Images: https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images")
            print("  - Videos: https://www.kaggle.com/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset")
            return
        else:
            raise
    
    print(f"\nData shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_val: {y_val.shape}")
    
    # Build model
    print("\n[2/4] Building model...")
    deepfake_model = model.build_model(
        input_shape=INPUT_SHAPE,
        learning_rate=LEARNING_RATE,
        fine_tune=False  # Start with frozen base model
    )
    
    print("\nModel architecture:")
    deepfake_model.summary()
    
    # Define callbacks
    print("\n[3/4] Setting up callbacks...")
    
    # Save best model based on validation accuracy
    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        verbose=1
    )
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Reduce learning rate on plateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    callbacks = [checkpoint, early_stopping, reduce_lr]
    
    # Train model
    print("\n[4/4] Starting training...")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print("-" * 60)
    
    history = deepfake_model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best model saved to: {MODEL_SAVE_PATH}")
    print("=" * 60)
    
    # Print final metrics
    print("\nFinal Training Metrics:")
    print(f"  Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"  Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"  Training AUC: {history.history['AUC'][-1]:.4f}")
    print(f"  Validation AUC: {history.history['val_AUC'][-1]:.4f}")


if __name__ == "__main__":
    # Set GPU memory growth (if using GPU)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    main()

