"""
Evaluation script for the DeepFake detection model.
Loads a trained model and evaluates it on test data.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import data_loader
import model


# Paths
MODEL_PATH = "./models/deepfake_detector_best.h5"
IMAGE_DATA_DIR = "./data/images"
VIDEO_DATA_DIR = "./data/videos"  # Optional
VIDEO_METADATA_PATH = None  # Optional


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot and optionally save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Real', 'Fake']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()


def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """Plot and optionally save ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"ROC curve saved to {save_path}")
    else:
        plt.show()


def main():
    """Main evaluation function."""
    
    print("=" * 60)
    print("DeepFake Detection Model Evaluation")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please train the model first using train.py")
        return
    
    # Load model
    print("\n[1/3] Loading model...")
    try:
        loaded_model = keras.models.load_model(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load test data
    print("\n[2/3] Loading test data...")
    X_train, X_test, y_train, y_test = data_loader.load_combined_data(
        image_dir=IMAGE_DATA_DIR,
        video_dir=VIDEO_DATA_DIR if os.path.exists(VIDEO_DATA_DIR) else None,
        video_metadata_path=VIDEO_METADATA_PATH,
        num_frames_per_video=20,
        test_size=0.2,
        random_state=42
    )
    
    print(f"Test set size: {len(X_test)} samples")
    print(f"  Real samples: {np.sum(y_test == 0)}")
    print(f"  Fake samples: {np.sum(y_test == 1)}")
    
    # Make predictions
    print("\n[3/3] Making predictions...")
    y_pred_proba = loaded_model.predict(X_test, batch_size=32, verbose=1)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Calculate metrics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"\n  True Negatives (Real predicted as Real): {cm[0, 0]}")
    print(f"  False Positives (Real predicted as Fake): {cm[0, 1]}")
    print(f"  False Negatives (Fake predicted as Real): {cm[1, 0]}")
    print(f"  True Positives (Fake predicted as Fake): {cm[1, 1]}")
    
    # AUC score
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC AUC Score: {auc_score:.4f}")
    
    # Accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Plot visualizations
    print("\nGenerating visualizations...")
    os.makedirs("./models", exist_ok=True)
    plot_confusion_matrix(y_test, y_pred, save_path="./models/confusion_matrix.png")
    plot_roc_curve(y_test, y_pred_proba, save_path="./models/roc_curve.png")
    
    print("\n" + "=" * 60)
    print("Evaluation completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

