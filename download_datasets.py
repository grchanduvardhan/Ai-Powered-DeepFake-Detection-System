"""
Script to download and organize DeepFake detection datasets from Kaggle.
"""

import os
import shutil
import kagglehub
from pathlib import Path


# Dataset paths
DATA_DIR = "./data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")
REAL_DIR = os.path.join(IMAGES_DIR, "real")
FAKE_DIR = os.path.join(IMAGES_DIR, "fake")


def download_image_dataset():
    """Download the image dataset and organize it."""
    print("=" * 60)
    print("Downloading Image Dataset")
    print("=" * 60)
    
    try:
        # Download dataset
        print("\n[1/3] Downloading dataset from Kaggle...")
        path = kagglehub.dataset_download("manjilkarki/deepfake-and-real-images")
        print(f"Dataset downloaded to: {path}")
        
        # Find the actual data directory
        dataset_path = Path(path)
        
        # Look for real and fake directories in the downloaded dataset
        print("\n[2/3] Searching for image directories...")
        
        # Common structures in Kaggle datasets
        possible_paths = [
            dataset_path,
            dataset_path / "deepfake-and-real-images",
            dataset_path / "data",
            dataset_path / "images",
        ]
        
        real_path = None
        fake_path = None
        
        # Search for real and fake directories
        for base_path in possible_paths:
            if base_path.exists():
                # Look for directories containing "real" or "fake"
                for item in base_path.rglob("*"):
                    if item.is_dir():
                        dir_name_lower = item.name.lower()
                        if "real" in dir_name_lower and real_path is None:
                            real_path = item
                        elif "fake" in dir_name_lower and fake_path is None:
                            fake_path = item
                
                # Also check direct subdirectories
                for item in base_path.iterdir():
                    if item.is_dir():
                        dir_name_lower = item.name.lower()
                        if "real" in dir_name_lower and real_path is None:
                            real_path = item
                        elif "fake" in dir_name_lower and fake_path is None:
                            fake_path = item
        
        if real_path is None or fake_path is None:
            print("\n[WARNING] Could not automatically find 'real' and 'fake' directories.")
            print(f"Please manually organize images from: {path}")
            print(f"Expected structure:")
            print(f"  {REAL_DIR}/")
            print(f"  {FAKE_DIR}/")
            return False
        
        print(f"Found real images in: {real_path}")
        print(f"Found fake images in: {fake_path}")
        
        # Create target directories
        os.makedirs(REAL_DIR, exist_ok=True)
        os.makedirs(FAKE_DIR, exist_ok=True)
        
        # Copy images
        print("\n[3/3] Copying images to organized structure...")
        
        real_count = 0
        fake_count = 0
        
        # Copy real images
        for img_file in real_path.rglob("*"):
            if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                dest = os.path.join(REAL_DIR, img_file.name)
                if not os.path.exists(dest):  # Avoid overwriting
                    shutil.copy2(img_file, dest)
                    real_count += 1
        
        # Copy fake images
        for img_file in fake_path.rglob("*"):
            if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                dest = os.path.join(FAKE_DIR, img_file.name)
                if not os.path.exists(dest):  # Avoid overwriting
                    shutil.copy2(img_file, dest)
                    fake_count += 1
        
        print(f"\n[SUCCESS] Image dataset organized!")
        print(f"  Real images: {real_count}")
        print(f"  Fake images: {fake_count}")
        print(f"  Location: {IMAGES_DIR}")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Failed to download image dataset: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure you have kagglehub installed: pip install kagglehub")
        print("  2. Make sure you're logged into Kaggle: kagglehub login")
        print("  3. Check your internet connection")
        return False


def download_video_dataset():
    """Download the video dataset."""
    print("\n" + "=" * 60)
    print("Downloading Video Dataset")
    print("=" * 60)
    
    try:
        # Create videos directory
        os.makedirs(VIDEOS_DIR, exist_ok=True)
        
        # Download dataset
        print("\n[1/2] Downloading dataset from Kaggle...")
        path = kagglehub.dataset_download("sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset")
        print(f"Dataset downloaded to: {path}")
        
        # Find video files
        print("\n[2/2] Searching for video files...")
        
        dataset_path = Path(path)
        video_count = 0
        
        # Search for video files
        video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
        for video_file in dataset_path.rglob("*"):
            if video_file.is_file() and video_file.suffix.lower() in video_extensions:
                dest = os.path.join(VIDEOS_DIR, video_file.name)
                if not os.path.exists(dest):  # Avoid overwriting
                    shutil.copy2(video_file, dest)
                    video_count += 1
        
        print(f"\n[SUCCESS] Video dataset organized!")
        print(f"  Videos found: {video_count}")
        print(f"  Location: {VIDEOS_DIR}")
        
        if video_count == 0:
            print("\n[NOTE] No video files found. The dataset might be organized differently.")
            print(f"Please check: {path}")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Failed to download video dataset: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure you have kagglehub installed: pip install kagglehub")
        print("  2. Make sure you're logged into Kaggle: kagglehub login")
        print("  3. Check your internet connection")
        return False


def main():
    """Main function to download all datasets."""
    print("\n" + "=" * 60)
    print("DeepFake Detection Dataset Downloader")
    print("=" * 60)
    
    # Check if kagglehub is installed
    try:
        import kagglehub
    except ImportError:
        print("\n[ERROR] kagglehub is not installed.")
        print("Please install it first: pip install kagglehub")
        print("Then authenticate: kagglehub login")
        return
    
    # Create data directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    
    # Download datasets
    image_success = download_image_dataset()
    video_success = download_video_dataset()
    
    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"Image dataset: {'SUCCESS' if image_success else 'FAILED'}")
    print(f"Video dataset: {'SUCCESS' if video_success else 'FAILED'}")
    
    if image_success:
        print("\nNext steps:")
        print("  1. Verify images are in ./data/images/real/ and ./data/images/fake/")
        print("  2. Run: python train.py")
    else:
        print("\nPlease check the error messages above and try again.")


if __name__ == "__main__":
    main()

