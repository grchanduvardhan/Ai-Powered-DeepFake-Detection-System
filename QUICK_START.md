# Quick Start Guide

## ✅ Installation Complete!

All dependencies have been successfully installed in a virtual environment using Python 3.12.

## 🚀 Next Steps

### 1. Activate the Virtual Environment

**Every time you open a new terminal, activate the virtual environment first:**

```powershell
.\venv\Scripts\Activate.ps1
```

You should see `(venv)` at the beginning of your command prompt.

### 2. Prepare Your Dataset

Before training, organize your data:

- **Images:** Place in `./data/images/` with subdirectories:
  - `./data/images/real/` - Real images
  - `./data/images/fake/` - Fake/DeepFake images

- **Videos (optional):** Place in `./data/videos/`

### 3. Train the Model

```powershell
python train.py
```

**Note:** Update the data paths in `train.py` if your data is in different locations.

### 4. Run the Web Application

After training (or if you have a pre-trained model):

```powershell
python app.py
```

Then open your browser to: `http://127.0.0.1:5000`

## 📦 Installed Packages

- ✅ TensorFlow 2.20.0
- ✅ OpenCV 4.11.0
- ✅ MTCNN 1.0.0
- ✅ Flask 3.1.2
- ✅ NumPy, Pandas, Scikit-learn
- ✅ All other dependencies

## 💡 Tips

- Always activate the virtual environment before running scripts
- The virtual environment is located in the `venv/` folder
- To deactivate: `deactivate` (when you're done working)

## 🐛 Troubleshooting

If you get "command not found" errors:
- Make sure the virtual environment is activated
- Check that you're in the project directory

If TensorFlow import errors occur:
- Verify Python version: `python --version` (should be 3.12.x)
- Reinstall if needed: `pip install -r requirements.txt`

