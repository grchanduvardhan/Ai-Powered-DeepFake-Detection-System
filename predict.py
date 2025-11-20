"""
Prediction module for DeepFake detection.
Contains functions for predicting on single images and videos.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from mtcnn import MTCNN
from data_loader import extract_face
from typing import Dict, Optional


def predict_image(image_path: str, model: keras.Model, 
                 mtcnn_detector: MTCNN, threshold: float = 0.5) -> Dict:
    """
    Predict if an image is real or fake (DeepFake).
    
    Args:
        image_path: Path to the image file
        model: Loaded Keras model
        mtcnn_detector: MTCNN face detector instance
        threshold: Confidence threshold for classification (default: 0.5)
    
    Returns:
        Dictionary with prediction results:
        {
            "status": "success" or "error",
            "prediction": "REAL" or "FAKE" (if success),
            "confidence": float (0-1, if success),
            "message": str (error message if error),
            "face_detected": bool,
            "face_image": np.ndarray or None (cropped face if detected)
        }
    """
    try:
        # Load image
        if not os.path.exists(image_path):
            return {
                "status": "error",
                "message": f"Image file not found: {image_path}",
                "face_detected": False
            }
        
        image = cv2.imread(image_path)
        if image is None:
            return {
                "status": "error",
                "message": f"Could not read image file: {image_path}",
                "face_detected": False
            }
        
        # Extract face
        face = extract_face(image)
        
        if face is None:
            return {
                "status": "error",
                "message": "No face detected in the image. Please ensure the image contains a clear face.",
                "face_detected": False
            }
        
        # Prepare face for prediction (add batch dimension)
        face_batch = np.expand_dims(face, axis=0)
        
        # Make prediction
        prediction_proba = model.predict(face_batch, verbose=0)[0][0]
        
        # Determine class
        is_fake = prediction_proba >= threshold
        prediction_label = "FAKE" if is_fake else "REAL"
        confidence = prediction_proba if is_fake else (1 - prediction_proba)
        
        # Convert face back to uint8 for display (0-255 range)
        face_display = (face * 255).astype(np.uint8)
        
        return {
            "status": "success",
            "prediction": prediction_label,
            "confidence": float(confidence),
            "raw_score": float(prediction_proba),
            "face_detected": True,
            "face_image": face_display
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error during prediction: {str(e)}",
            "face_detected": False
        }


def predict_video(video_path: str, model: keras.Model, 
                 mtcnn_detector: MTCNN, num_frames: int = 30,
                 threshold: float = 0.5) -> Dict:
    """
    Predict if a video is real or fake (DeepFake) by analyzing multiple frames.
    
    Args:
        video_path: Path to the video file
        model: Loaded Keras model
        mtcnn_detector: MTCNN face detector instance
        num_frames: Number of frames to sample from the video
        threshold: Confidence threshold for classification (default: 0.5)
    
    Returns:
        Dictionary with prediction results:
        {
            "status": "success" or "error",
            "prediction": "REAL" or "FAKE" (if success),
            "confidence": float (0-1, if success),
            "message": str (error message if error),
            "frames_analyzed": int (number of frames with detected faces),
            "total_frames": int (total frames in video)
        }
    """
    try:
        # Open video
        if not os.path.exists(video_path):
            return {
                "status": "error",
                "message": f"Video file not found: {video_path}",
                "frames_analyzed": 0
            }
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                "status": "error",
                "message": f"Could not open video file: {video_path}",
                "frames_analyzed": 0
            }
        
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames == 0:
            cap.release()
            return {
                "status": "error",
                "message": "Video file appears to be empty or corrupted",
                "frames_analyzed": 0
            }
        
        # Sample frames evenly
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        predictions = []
        faces_detected = 0
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Extract face from frame
            face = extract_face(frame)
            
            if face is not None:
                # Prepare face for prediction
                face_batch = np.expand_dims(face, axis=0)
                
                # Make prediction
                prediction_proba = model.predict(face_batch, verbose=0)[0][0]
                predictions.append(prediction_proba)
                faces_detected += 1
        
        cap.release()
        
        if faces_detected == 0:
            return {
                "status": "error",
                "message": "No faces detected in any sampled frames. Please ensure the video contains clear faces.",
                "frames_analyzed": 0,
                "total_frames": total_frames
            }
        
        # Aggregate predictions (average probability)
        avg_prediction = np.mean(predictions)
        
        # Determine class
        is_fake = avg_prediction >= threshold
        prediction_label = "FAKE" if is_fake else "REAL"
        confidence = avg_prediction if is_fake else (1 - avg_prediction)
        
        return {
            "status": "success",
            "prediction": prediction_label,
            "confidence": float(confidence),
            "raw_score": float(avg_prediction),
            "frames_analyzed": faces_detected,
            "total_frames": total_frames,
            "fps": float(fps) if fps > 0 else None
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error during video prediction: {str(e)}",
            "frames_analyzed": 0
        }


if __name__ == "__main__":
    # Example usage
    print("Prediction module loaded successfully!")
    print("This module is designed to be imported by app.py")

