# AI-Based DeepFake Detection System

A comprehensive deep learning-based system for detecting DeepFake images and videos using transfer learning with TensorFlow/Keras and a user-friendly Flask web interface.

## 🎯 Project Overview

This project implements an end-to-end DeepFake detection system that can:
- Analyze images and videos to detect if they contain DeepFake content
- Extract faces using MTCNN for accurate face detection
- Classify media as "REAL" or "FAKE" with confidence scores
- Provide a modern web interface for easy interaction

## 🎥 Demo Video

[![Watch the demo](https://img.shields.io/badge/Watch-Demo%20Video-blue?style=for-the-badge&logo=github)](./demo_video.mp4)

## 📁 Project Structure

```
/deepfake-detector
|
|-- /data/                   # Placeholder for datasets
|   |-- /images/             # Image dataset (with 'real' and 'fake' subdirectories)
|   |-- /videos/             # Video dataset
|
|-- /models/                 # Saved model files
|   |-- deepfake_detector_best.h5
|
|-- /static/                 # Static files for web app
|   |-- /css/
|       |-- style.css        # Custom CSS (optional)
|
|-- /templates/              # Flask HTML templates
|   |-- index.html           # Main web interface
|
|-- /uploads/                # Temporary storage for uploaded files
|
|-- data_loader.py           # Data loading and preprocessing
|-- model.py                 # CNN model architecture
|-- train.py                 # Training script
|-- evaluate.py              # Model evaluation script
|-- predict.py               # Inference functions
|-- app.py                   # Flask web server
|-- requirements.txt         # Python dependencies
|-- README.md                # This file
```

## 🚀 Setup Instructions

### 1. Prerequisites

- **Python 3.8, 3.9, 3.10, 3.11, or 3.12** (TensorFlow does not support Python 3.13+ yet)
- pip (Python package manager)

**⚠️ Important:** If you have Python 3.14 or newer, you'll need to install Python 3.11 or 3.12. See `INSTALLATION_GUIDE.md` for detailed instructions.

### 2. Install Dependencies

**First, ensure you're using Python 3.8-3.12:**
```bash
python --version
```

**Create a virtual environment (recommended):**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

**Install dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** 
- TensorFlow installation may vary based on your system. 
- For GPU support, install `tensorflow[and-cuda]` or use `tensorflow-gpu` (depending on your TensorFlow version).
- If you encounter Python version errors, see `INSTALLATION_GUIDE.md` for solutions.

### 3. Prepare Your Dataset

#### For Images:
1. Download the dataset from [manjilkarki/deepfake-and-real-images](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)
2. Extract and organize the data as follows:
   ```
   ./data/images/
   ├── real/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   └── fake/
       ├── image1.jpg
       ├── image2.jpg
       └── ...
   ```

#### For Videos:
1. Download the dataset from [sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset](https://www.kaggle.com/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset)
2. Place video files in `./data/videos/`
3. (Optional) Create a CSV metadata file with columns: `filename` and `label` (0 for real, 1 for fake)

### 4. Train the Model

Before using the web application, you need to train the model:

```bash
python train.py
```

**Training Configuration:**
- Update the paths in `train.py` if your data is in different locations:
  - `IMAGE_DATA_DIR`: Path to image dataset
  - `VIDEO_DATA_DIR`: Path to video dataset (optional)
  - `VIDEO_METADATA_PATH`: Path to video metadata CSV (optional)

**Training Parameters:**
- Epochs: 50 (with early stopping)
- Batch Size: 32
- Learning Rate: 0.0001
- Model: XceptionNet (transfer learning)

The best model will be saved to `./models/deepfake_detector_best.h5`

### 5. Evaluate the Model (Optional)

To evaluate the trained model on test data:

```bash
python evaluate.py
```

This will generate:
- Classification report
- Confusion matrix
- ROC curve
- Accuracy and AUC metrics

## 🌐 How to Run the Web Application

1. **Ensure the model is trained:**
   - The model file should exist at `./models/deepfake_detector_best.h5`
   - If not, train the model first using `python train.py`

2. **Start the Flask server:**
   ```bash
   python app.py
   ```

3. **Open your browser:**
   - Navigate to: `http://127.0.0.1:5000`
   - You should see the DeepFake Detection System interface

4. **Use the application:**
   - Drag and drop an image or video file, or click "Browse Files"
   - Supported formats: PNG, JPG, JPEG, MP4, MOV, AVI
   - Click "Detect DeepFake" to analyze the file
   - View the results with confidence scores

## 📖 Usage Examples

### Command Line Prediction

You can also use the prediction functions directly in Python:

```python
import tensorflow as tf
from tensorflow import keras
from mtcnn import MTCNN
import predict

# Load model and detector
model = keras.models.load_model('./models/deepfake_detector_best.h5')
detector = MTCNN()

# Predict on an image
result = predict.predict_image('path/to/image.jpg', model, detector)
print(result)

# Predict on a video
result = predict.predict_video('path/to/video.mp4', model, detector, num_frames=30)
print(result)
```

## 🔧 Technical Details

### Model Architecture
- **Base Model:** XceptionNet (pre-trained on ImageNet)
- **Transfer Learning:** Frozen base model with custom classification head
- **Classification Head:**
  - GlobalAveragePooling2D
  - Dense(128, ReLU)
  - Dropout(0.5)
  - Dense(1, Sigmoid) - Binary classification

### Face Detection
- **Method:** MTCNN (Multi-task Cascaded Convolutional Networks)
- **Face Size:** 224x224 pixels
- **Preprocessing:** Normalized to [0, 1] range

### Training Features
- Early stopping to prevent overfitting
- Learning rate reduction on plateau
- Model checkpointing (saves best model based on validation accuracy)
- Stratified train/validation split

## 📊 Performance Metrics

After training, the model will display:
- Training and validation accuracy
- Training and validation AUC (Area Under Curve)
- Classification report with precision, recall, and F1-score
- Confusion matrix
- ROC curve

## 🛠️ Customization

### Adjust Training Parameters

Edit `train.py` to modify:
- `EPOCHS`: Number of training epochs
- `BATCH_SIZE`: Batch size for training
- `LEARNING_RATE`: Learning rate for optimizer
- `INPUT_SHAPE`: Input image dimensions

### Use Different Base Model

In `model.py`, you can switch to EfficientNetB0:
```python
model = model.build_model_efficientnet(input_shape=(224, 224, 3))
```

### Fine-tuning

To fine-tune the base model (unfreeze some layers), modify `train.py`:
```python
deepfake_model = model.build_model(
    input_shape=INPUT_SHAPE,
    learning_rate=LEARNING_RATE,
    fine_tune=True,  # Enable fine-tuning
    fine_tune_from_layer='block14_sepconv1'  # Optional: specify layer
)
```

## ⚠️ Important Notes

1. **Face Detection:** The system requires clear faces in the input media. If no face is detected, an error will be returned.

2. **Model Performance:** Model accuracy depends on:
   - Quality and size of training data
   - Similarity between training and test data
   - Image/video quality

3. **Computational Requirements:**
   - Training: GPU recommended for faster training
   - Inference: Can run on CPU, but GPU will be faster

4. **File Size Limits:** The web app has a 100MB file size limit (configurable in `app.py`)

## 🐛 Troubleshooting

### Model not found error
- Ensure you've trained the model: `python train.py`
- Check that `./models/deepfake_detector_best.h5` exists

### No face detected error
- Ensure the image/video contains a clear, visible face
- Try a different image/video with better face visibility

### Memory errors during training
- Reduce `BATCH_SIZE` in `train.py`
- Reduce `num_frames_per_video` in data loading
- Use a smaller dataset subset for testing

### Import errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+ required)

## 📝 License

This project is provided as-is for educational and research purposes.

## 🙏 Acknowledgments

- Datasets:
  - [manjilkarki/deepfake-and-real-images](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)
  - [sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset](https://www.kaggle.com/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset)
- Technologies:
  - TensorFlow/Keras
  - MTCNN for face detection
  - Flask for web interface
  - Tailwind CSS for styling

<<<<<<< HEAD
---
=======
>>>>>>> 8c06eef68fa4627d5742e5b3b1ecc0d152240706

## Future Enhancements

I wanted to implement video detection as well, but due to limited RAM in my system, I wasn't able to implement it. However, the project already contains code for video training, which can be implemented. I'm going to implement it someday.

