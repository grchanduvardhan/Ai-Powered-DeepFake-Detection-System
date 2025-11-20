"""
Data loading and preprocessing module for DeepFake detection.
Handles face extraction from images and videos, and data preparation for training.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from mtcnn import MTCNN
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, List
import warnings
import logging
from contextlib import contextmanager

# Suppress TensorFlow warnings about shape mismatches (these are handled gracefully)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

# Suppress TensorFlow logging before MTCNN uses it
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('tensorflow').disabled = True
    # Suppress specific TensorFlow warnings
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except (ImportError, AttributeError):
    pass

# Initialize MTCNN detector (will be reused)
# MTCNN parameters: steps_threshold controls detection sensitivity
detector = MTCNN()


def extract_face(image: np.ndarray, required_size=(224, 224)) -> Optional[np.ndarray]:
    """
    Extract and preprocess a face from an image using MTCNN.
    
    Args:
        image: Input image as NumPy array (BGR format from OpenCV)
        required_size: Target size for the extracted face (width, height)
    
    Returns:
        Preprocessed face image as NumPy array, or None if no face detected
    """
    try:
        # Validate input image
        if image is None or image.size == 0:
            return None
        
        # Check image dimensions
        if len(image.shape) != 3 or image.shape[2] != 3:
            return None
        
        # Convert BGR to RGB for MTCNN
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Validate RGB image
        if rgb_image is None or rgb_image.size == 0:
            return None
        
        # Detect faces with error handling and suppressed warnings
        try:
            # Temporarily redirect stderr to suppress TensorFlow warnings
            @contextmanager
            def suppress_stderr():
                with open(os.devnull, 'w') as devnull:
                    old_stderr = sys.stderr
                    sys.stderr = devnull
                    try:
                        yield
                    finally:
                        sys.stderr = old_stderr
            
            with suppress_stderr():
                faces = detector.detect_faces(rgb_image)
        except Exception:
            # If MTCNN fails on this image, skip it
            return None
        
        if len(faces) == 0:
            return None
        
        # Get the largest face (by bounding box area)
        largest_face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
        x, y, width, height = largest_face['box']
        
        # Validate bounding box
        if width <= 0 or height <= 0:
            return None
        
        # Extract face region (add some padding)
        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(rgb_image.shape[1], x + width + padding)
        y2 = min(rgb_image.shape[0], y + height + padding)
        
        # Ensure valid dimensions
        if x2 <= x1 or y2 <= y1:
            return None
        
        face = rgb_image[y1:y2, x1:x2]
        
        # Validate extracted face region
        if face.size == 0 or face.shape[0] == 0 or face.shape[1] == 0:
            return None
        
        # Ensure minimum size before resizing
        min_size = 10
        if face.shape[0] < min_size or face.shape[1] < min_size:
            return None
        
        # Resize to required size
        face_resized = cv2.resize(face, required_size)
        
        # Validate resized face
        if face_resized.shape != (*required_size, 3):
            return None
        
        # Normalize pixel values to [0, 1]
        face_normalized = face_resized.astype('float32') / 255.0
        
        return face_normalized
        
    except Exception as e:
        # Silently skip images that cause errors
        return None


def load_image_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess image data from directory structure.
    Expected structure: data_dir/real/ and data_dir/fake/
    
    Args:
        data_dir: Root directory containing 'real' and 'fake' subdirectories
    
    Returns:
        Tuple of (faces, labels) as NumPy arrays
    """
    faces = []
    labels = []
    
    # Process real images (label = 0)
    real_dir = os.path.join(data_dir, 'real')
    if os.path.exists(real_dir):
        print(f"Loading real images from {real_dir}...")
        real_count = 0
        skipped_count = 0
        total_files = len([f for f in os.listdir(real_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        processed_count = 0
        
        for filename in os.listdir(real_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                processed_count += 1
                image_path = os.path.join(real_dir, filename)
                try:
                    image = cv2.imread(image_path)
                    if image is not None and image.size > 0:
                        # Validate image can be read properly
                        if image.shape[0] > 0 and image.shape[1] > 0:
                            face = extract_face(image)
                            if face is not None:
                                faces.append(face)
                                labels.append(0)  # Real = 0
                                real_count += 1
                            else:
                                skipped_count += 1
                        else:
                            skipped_count += 1
                    else:
                        skipped_count += 1
                except Exception:
                    skipped_count += 1
                
                # Print progress every 100 files or at completion
                if processed_count % 100 == 0 or processed_count == total_files:
                    print(f"  Progress: {processed_count}/{total_files} files processed | "
                          f"Loaded: {real_count} faces | Skipped: {skipped_count} (no face/invalid)")
        
        print(f"  ✓ Completed: {real_count} real faces loaded, {skipped_count} images skipped")
    
    # Process fake images (label = 1)
    fake_dir = os.path.join(data_dir, 'fake')
    if os.path.exists(fake_dir):
        print(f"Loading fake images from {fake_dir}...")
        fake_count = 0
        skipped_count = 0
        total_files = len([f for f in os.listdir(fake_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        processed_count = 0
        
        for filename in os.listdir(fake_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                processed_count += 1
                image_path = os.path.join(fake_dir, filename)
                try:
                    image = cv2.imread(image_path)
                    if image is not None and image.size > 0:
                        # Validate image can be read properly
                        if image.shape[0] > 0 and image.shape[1] > 0:
                            face = extract_face(image)
                            if face is not None:
                                faces.append(face)
                                labels.append(1)  # Fake = 1
                                fake_count += 1
                            else:
                                skipped_count += 1
                        else:
                            skipped_count += 1
                    else:
                        skipped_count += 1
                except Exception:
                    skipped_count += 1
                
                # Print progress every 100 files or at completion
                if processed_count % 100 == 0 or processed_count == total_files:
                    print(f"  Progress: {processed_count}/{total_files} files processed | "
                          f"Loaded: {fake_count} faces | Skipped: {skipped_count} (no face/invalid)")
        
        print(f"  ✓ Completed: {fake_count} fake faces loaded, {skipped_count} images skipped")
    
    print(f"Loaded {len(faces)} faces from images")
    return np.array(faces), np.array(labels)


def load_video_data(data_dir: str, metadata_path: Optional[str] = None,
                   num_frames_per_video: int = 3, max_video_faces: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess video data.
    
    Args:
        data_dir: Directory containing video files
        metadata_path: Optional path to CSV file with video labels
                      (should have columns: 'filename' and 'label' or 'is_fake')
        num_frames_per_video: Number of frames to sample from each video
    
    Returns:
        Tuple of (faces, labels) as NumPy arrays
    """
    faces = []
    labels = []
    
    # Load metadata if provided
    metadata = {}
    if metadata_path and os.path.exists(metadata_path):
        df = pd.read_csv(metadata_path)
        # Try different possible column names
        filename_col = None
        label_col = None
        
        for col in df.columns:
            if 'filename' in col.lower() or 'file' in col.lower():
                filename_col = col
            if 'label' in col.lower() or 'is_fake' in col.lower() or 'fake' in col.lower():
                label_col = col
        
        if filename_col and label_col:
            for _, row in df.iterrows():
                filename = row[filename_col]
                label_val = row[label_col]
                # Convert to binary: 0 for real, 1 for fake
                if isinstance(label_val, str):
                    label_val = 1 if 'fake' in label_val.lower() else 0
                else:
                    label_val = int(label_val)
                metadata[filename] = label_val
    
    # Process video files
    print(f"Loading videos from {data_dir}...")
    video_files = [f for f in os.listdir(data_dir) 
                   if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]
    
    total_videos = len(video_files)
    processed_videos = 0
    video_faces_count = 0

    print(f"  Found {total_videos} video files to process...")

    for video_file in video_files:
        processed_videos += 1
        video_path = os.path.join(data_dir, video_file)

        # Determine label
        label = 0  # Default to real
        if metadata:
            # Try to find label in metadata
            for key in metadata.keys():
                if key in video_file or video_file in key:
                    label = metadata[key]
                    break
        else:
            # If no metadata, try to infer from filename
            if 'fake' in video_file.lower():
                label = 1

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue

            # Get total frame count
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                cap.release()
                continue

            # Sample frames evenly
            frame_indices = np.linspace(0, total_frames - 1, num_frames_per_video, dtype=int)

            frames_processed = 0
            for frame_idx in frame_indices:
                # Check if we've reached the maximum number of faces
                if video_faces_count >= max_video_faces:
                    break

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    continue

                face = extract_face(frame)
                if face is not None:
                    faces.append(face)
                    labels.append(label)
                    frames_processed += 1
                    video_faces_count += 1

            cap.release()

            if frames_processed > 0:
                pass  # video_faces_count already updated

        except Exception as e:
            pass  # Silently skip problematic videos

        # Stop processing if we've reached the maximum
        if video_faces_count >= max_video_faces:
            print(f"  Reached maximum video faces limit ({max_video_faces}), stopping video processing...")
            break

        # Print progress every 50 videos or at completion
        if processed_videos % 50 == 0 or processed_videos == total_videos:
            print(f"  Progress: {processed_videos}/{total_videos} videos processed | "
                  f"Faces extracted: {video_faces_count}")

    print(f"  ✓ Completed: {len(faces)} faces extracted from {processed_videos} videos")
    return np.array(faces), np.array(labels)


def load_combined_data(image_dir: str, video_dir: Optional[str] = None,
                      video_metadata_path: Optional[str] = None,
                      num_frames_per_video: int = 3,
                      max_video_faces: int = 5000,
                      test_size: float = 0.2,
                      random_state: int = 42) -> Tuple:
    """
    Load and combine image and video data, then split into train/validation sets.
    
    Args:
        image_dir: Directory containing image data (with 'real' and 'fake' subdirs)
        video_dir: Optional directory containing video files
        video_metadata_path: Optional path to video metadata CSV
        num_frames_per_video: Number of frames to sample per video
        test_size: Proportion of data to use for validation
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (X_train, X_val, y_train, y_val)
    """
    # Load image data
    image_faces, image_labels = load_image_data(image_dir)
    
    # Load video data if provided
    if video_dir and os.path.exists(video_dir):
        video_faces, video_labels = load_video_data(video_dir, video_metadata_path,
                                                   num_frames_per_video, max_video_faces)

        # Combine image and video data
        if len(video_faces) > 0:
            all_faces = np.concatenate([image_faces, video_faces], axis=0)
            all_labels = np.concatenate([image_labels, video_labels], axis=0)
        else:
            all_faces = image_faces
            all_labels = image_labels
    else:
        all_faces = image_faces
        all_labels = image_labels
    
    print(f"Total faces loaded: {len(all_faces)}")
    print(f"Real faces: {np.sum(all_labels == 0)}, Fake faces: {np.sum(all_labels == 1)}")
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        all_faces, all_labels, 
        test_size=test_size, 
        random_state=random_state,
        stratify=all_labels
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    return X_train, X_val, y_train, y_val


if __name__ == "__main__":
    # Example usage
    print("Data loader module loaded successfully!")
    print("Use load_combined_data() to load your dataset.")

