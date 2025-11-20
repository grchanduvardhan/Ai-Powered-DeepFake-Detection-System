# Installation Guide - DeepFake Detection System

## ⚠️ Python Version Compatibility

**Important:** TensorFlow currently supports Python 3.8 through 3.12. If you're using Python 3.14 or newer, you'll need to use a compatible Python version.

### Option 1: Use Python 3.11 or 3.12 (Recommended)

1. **Install Python 3.11 or 3.12:**
   - Download from: https://www.python.org/downloads/
   - Or use a version manager like `pyenv` (Windows: `pyenv-win`)

2. **Create a virtual environment with the correct Python version:**
   ```powershell
   # Using Python 3.11 (adjust path as needed)
   py -3.11 -m venv venv
   
   # Activate the virtual environment
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

### Option 2: Use Python 3.12 via pyenv (Windows)

1. **Install pyenv-win:**
   ```powershell
   git clone https://github.com/pyenv-win/pyenv-win.git $HOME\.pyenv
   ```

2. **Install Python 3.12:**
   ```powershell
   pyenv install 3.12.0
   pyenv local 3.12.0
   ```

3. **Create virtual environment and install:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

### Option 3: Use Conda (Alternative)

1. **Install Miniconda or Anaconda:**
   - Download from: https://docs.conda.io/en/latest/miniconda.html

2. **Create a conda environment with Python 3.11:**
   ```powershell
   conda create -n deepfake python=3.11
   conda activate deepfake
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

## 📦 Installation Steps (After Python Version is Correct)

1. **Navigate to project directory:**
   ```powershell
   cd "C:\Users\m092m\OneDrive\Desktop\Ai-Based DeepFake Detection System"
   ```

2. **Create virtual environment (if not using conda):**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Upgrade pip:**
   ```powershell
   python -m pip install --upgrade pip
   ```

4. **Install requirements:**
   ```powershell
   pip install -r requirements.txt
   ```

## 🔍 Verify Installation

Check if TensorFlow is installed correctly:
```powershell
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
```

## 🐛 Troubleshooting

### Issue: "No matching distribution found for tensorflow"

**Solution:** You're likely using an unsupported Python version. Use Python 3.11 or 3.12.

### Issue: "Defaulting to user installation because normal site-packages is not writeable"

**Solution:** Use a virtual environment:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Issue: MTCNN installation fails

**Solution:** Install dependencies separately:
```powershell
pip install tensorflow
pip install opencv-python
pip install mtcnn
pip install numpy pandas scikit-learn flask pillow
```

## 📝 Current Python Version Check

To check your current Python version:
```powershell
python --version
```

If it shows Python 3.14 or higher, you need to install Python 3.11 or 3.12.

