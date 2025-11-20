"""
Flask web application for DeepFake detection.
Provides a user-friendly interface for uploading images/videos and getting predictions.
"""

import os
import json
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras
from mtcnn import MTCNN
import predict

# Initialize Flask app
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'mov', 'avi'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Global variables for model and detector (loaded once at startup)
model = None
mtcnn_detector = None
MODEL_PATH = './models/deepfake_detector_best.h5'


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_image_file(filename):
    """Check if file is an image."""
    return filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}


def is_video_file(filename):
    """Check if file is a video."""
    return filename.rsplit('.', 1)[1].lower() in {'mp4', 'mov', 'avi'}


def load_model_and_detector():
    """Load the trained model and MTCNN detector once at startup."""
    global model, mtcnn_detector
    
    print("Loading model and detector...")
    
    # Load MTCNN detector
    mtcnn_detector = MTCNN()
    print("MTCNN detector loaded.")
    
    # Load TensorFlow model
    if os.path.exists(MODEL_PATH):
        try:
            model = keras.models.load_model(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please ensure the model file exists and is valid.")
            model = None
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}")
        print("Please train the model first using train.py")
        model = None


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    Handle file upload and prediction.
    Returns JSON response with prediction results.
    """
    # Check if model is loaded
    if model is None or mtcnn_detector is None:
        return jsonify({
            "status": "error",
            "message": "Model not loaded. Please ensure the model file exists."
        }), 500
    
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({
            "status": "error",
            "message": "No file provided in request."
        }), 400
    
    file = request.files['file']
    
    # Check if file was selected
    if file.filename == '':
        return jsonify({
            "status": "error",
            "message": "No file selected."
        }), 400
    
    # Check if file type is allowed
    if not allowed_file(file.filename):
        return jsonify({
            "status": "error",
            "message": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        # Create uploads directory if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save file
        file.save(filepath)
        
        # Determine file type and make prediction
        if is_image_file(filename):
            result = predict.predict_image(filepath, model, mtcnn_detector)
        elif is_video_file(filename):
            result = predict.predict_video(filepath, model, mtcnn_detector, num_frames=30)
        else:
            # Should not reach here due to allowed_file check, but just in case
            os.remove(filepath)
            return jsonify({
                "status": "error",
                "message": "Unsupported file type."
            }), 400
        
        # Clean up: delete temporary file
        try:
            os.remove(filepath)
        except Exception as e:
            print(f"Warning: Could not delete temporary file {filepath}: {e}")
        
        # Convert numpy array to list for JSON serialization (if face_image exists)
        if 'face_image' in result and result['face_image'] is not None:
            # For web display, we'll send the face image as base64 or just indicate it exists
            # For now, we'll remove it from JSON response (can be handled client-side if needed)
            face_img = result.pop('face_image')
            result['face_detected'] = True
        
        return jsonify(result)
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
        
        return jsonify({
            "status": "error",
            "message": f"Error processing file: {str(e)}"
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "detector_loaded": mtcnn_detector is not None
    })


if __name__ == '__main__':
    # Load model and detector before starting the server
    load_model_and_detector()
    
    # Create necessary directories
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs('./models', exist_ok=True)
    
    print("\n" + "=" * 60)
    print("DeepFake Detection Web Application")
    print("=" * 60)
    print(f"Model loaded: {model is not None}")
    print(f"Detector loaded: {mtcnn_detector is not None}")
    print("\nStarting Flask server...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    
    # Run Flask app
    app.run(debug=True, host='127.0.0.1', port=5000)

