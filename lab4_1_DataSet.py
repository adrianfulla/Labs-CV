import os
import numpy as np
import requests
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm
import random
from pathlib import Path
import zipfile
import shutil

# Import your anisotropic diffusion function
from Anisotropic import anisodiff


BASE_DIR="bsds500_data"
WINDOW_SIZE=32
NUM_SAMPLES=500000


def download_bsds500_images():
        """
        Download BSDS500 images from GitHub repository
        """
        print("Downloading BSDS500 dataset...")
        
        # GitHub repository URL for the entire BSDS500 dataset
        repo_url = "https://github.com/BIDS/BSDS500/archive/refs/heads/master.zip"
        zip_path = base_dir / "bsds500.zip"
        
        # Download the zip file
        response = requests.get(repo_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as file, tqdm(
            desc="Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                size = file.write(chunk)
                bar.update(size)
        
        # Extract the zip file
        print("Extracting images...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(base_dir)
        
        # Move images to the images directory
        extracted_path = base_dir / "BSDS500-master" / "BSDS500" / "data" / "images"
        
        # Copy all images from train, test, and val directories
        for split in ["train", "test", "val"]:
            split_path = extracted_path / split
            if split_path.exists():
                for img_file in split_path.glob("*.jpg"):
                    shutil.copy2(img_file, images_dir)
        
        # Clean up
        os.remove(zip_path)
        shutil.rmtree(base_dir / "BSDS500-master")
        
        print(f"Downloaded {len(list(images_dir.glob('*.jpg')))} images")

def convert_to_grayscale_and_filter():
        """
        Convert images to grayscale and apply anisotropic filtering
        """
        print("Converting to grayscale and applying anisotropic filtering...")
        
        image_files = list(images_dir.glob("*.jpg"))
        
        # Anisotropic filter parameters
        filter_params = {
            'niter': 50,
            'kappa': 20,
            'gamma': 0.2,
            'step': (1., 1.),
            'option': 1,
            'ploton': False
        }
        
        for img_file in tqdm(image_files, desc="Processing images"):
            # Load and convert to grayscale
            img = Image.open(img_file).convert('L')
            img_array = np.array(img, dtype=np.float32)
            
            # Apply anisotropic filtering
            filtered_img = anisodiff(img_array, **filter_params)
            
            # Save filtered image
            filtered_path = filtered_dir / img_file.name
            filtered_img_pil = Image.fromarray(np.clip(filtered_img, 0, 255).astype(np.uint8))
            filtered_img_pil.save(filtered_path)
            
            # Also save grayscale original for consistency
            grayscale_path = images_dir / f"gray_{img_file.name}"
            img.save(grayscale_path)

def generate_window_samples():
        """
        Generate window samples from original and filtered image pairs
        """
        print(f"Generating {NUM_SAMPLES} window samples...")
        
        # Get list of processed images
        gray_images = list(images_dir.glob("gray_*.jpg"))
        
        if not gray_images:
            raise ValueError("No grayscale images found. Run convert_to_grayscale_and_filter() first.")
        
        X_samples = []  # Original image windows
        y_samples = []  # Filtered image windows
        
        samples_per_image = max(1, NUM_SAMPLES // len(gray_images))
        
        for gray_img_path in tqdm(gray_images, desc="Extracting windows"):
            # Load original grayscale image
            original_img = np.array(Image.open(gray_img_path), dtype=np.float32)
            
            # Load corresponding filtered image
            filtered_img_path = filtered_dir / gray_img_path.name.replace("gray_", "")
            filtered_img = np.array(Image.open(filtered_img_path), dtype=np.float32)
            
            # Get image dimensions
            h, w = original_img.shape
            
            # Skip if image is too small
            if h < WINDOW_SIZE or w < WINDOW_SIZE:
                continue
            
            # Generate random windows for this image
            for _ in range(samples_per_image):
                if len(X_samples) >= NUM_SAMPLES:
                    break
                
                # Random position ensuring window fits within image
                y_pos = random.randint(0, h - WINDOW_SIZE)
                x_pos = random.randint(0, w - WINDOW_SIZE)
                
                # Extract windows
                original_window = original_img[y_pos:y_pos+WINDOW_SIZE, 
                                             x_pos:x_pos+WINDOW_SIZE]
                filtered_window = filtered_img[y_pos:y_pos+WINDOW_SIZE,
                                             x_pos:x_pos+WINDOW_SIZE]
                
                X_samples.append(original_window)
                y_samples.append(filtered_window)
            
            if len(X_samples) >= NUM_SAMPLES:
                break
        
        # Convert to numpy arrays
        X_samples = np.array(X_samples)
        y_samples = np.array(y_samples)
        
        print(f"Generated {len(X_samples)} samples of size {WINDOW_SIZE}x{WINDOW_SIZE}")
        
        return X_samples, y_samples

def split_dataset(X, y, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split dataset into train, validation, and test sets
        
        Args:
            X: Input samples
            y: Target samples
            test_size: Proportion of test set
            val_size: Proportion of validation set (from remaining data after test split)
            random_state: Random seed for reproducibility
        """
        print("Splitting dataset...")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Second split: separate train and validation from remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=random_state
        )
        
        print(f"Dataset split:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")
        
        return (X_train, X_val, X_test), (y_train, y_val, y_test)

def save_dataset(X_splits, y_splits):
    """
    Save the dataset splits to disk
    """
    print("Saving dataset...")
    
    X_train, X_val, X_test = X_splits
    y_train, y_val, y_test = y_splits
    
    # Save as numpy arrays
    np.save(dataset_dir / "X_train.npy", X_train)
    np.save(dataset_dir / "X_val.npy", X_val)
    np.save(dataset_dir / "X_test.npy", X_test)
    np.save(dataset_dir / "y_train.npy", y_train)
    np.save(dataset_dir / "y_val.npy", y_val)
    np.save(dataset_dir / "y_test.npy", y_test)
    
    # Save metadata
    metadata = {
        'window_size': WINDOW_SIZE,
        'num_samples': len(X_train) + len(X_val) + len(X_test),
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'filter_params': {
            'niter': 50,
            'kappa': 20,
            'gamma': 0.2,
            'step': (1., 1.),
            'option': 1
        }
    }
    
    with open(dataset_dir / "metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Dataset saved to {dataset_dir}")

def visualize_samples(X_splits, y_splits, num_samples=5):
        """
        Visualize some sample pairs
        """
        X_train, _, _ = X_splits
        y_train, _, _ = y_splits
        
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
        
        for i in range(num_samples):
            idx = random.randint(0, len(X_train) - 1)
            
            # Original window
            axes[0, i].imshow(X_train[idx], cmap='gray')
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
            
            # Filtered window
            axes[1, i].imshow(y_train[idx], cmap='gray')
            axes[1, i].set_title(f'Filtered {i+1}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(dataset_dir / "sample_visualization.png", dpi=150, bbox_inches='tight')
        plt.show()

def load_dataset(dataset_dir):
    """
    Utility function to load the saved dataset
    
    Args:
        dataset_dir: Path to the dataset directory
        
    Returns:
        Dictionary containing the loaded data and metadata
    """
    dataset_dir = Path(dataset_dir)
    
    data = {
        'X_train': np.load(dataset_dir / "X_train.npy"),
        'X_val': np.load(dataset_dir / "X_val.npy"),
        'X_test': np.load(dataset_dir / "X_test.npy"),
        'y_train': np.load(dataset_dir / "y_train.npy"),
        'y_val': np.load(dataset_dir / "y_val.npy"),
        'y_test': np.load(dataset_dir / "y_test.npy")
    }
    
    with open(dataset_dir / "metadata.pkl", 'rb') as f:
        data['metadata'] = pickle.load(f)
    
    return data

base_dir = Path(BASE_DIR)

images_dir = base_dir / "images"
filtered_dir = base_dir / "filtered"
dataset_dir = base_dir / "dataset"

for dir_path in [images_dir, filtered_dir, dataset_dir]:
    dir_path.mkdir(parents=True, exist_ok=True)

"""
Running the BSDS500 Dataset Generation pipeline
"""
print("=== BSDS500 Anisotropic Filtering Dataset Generation ===")
print(f"Window size: {WINDOW_SIZE}x{WINDOW_SIZE}")
print(f"Target samples: {NUM_SAMPLES}")
print()

# Step 1: Download images
if not any(images_dir.glob("*.jpg")):
    download_bsds500_images()
else:
    print("Images already downloaded, skipping download step")
    

# Step 2: Convert to grayscale and apply filtering
if not any(filtered_dir.glob("*.jpg")):
    convert_to_grayscale_and_filter()
else:
    print("Filtered images already exist, skipping filtering step")
    
 # Step 3: Generate window samples
X_samples, y_samples = generate_window_samples()

# Step 4: Split dataset
X_splits, y_splits = split_dataset(X_samples, y_samples)

# Step 5: Save dataset
save_dataset(X_splits, y_splits)

# Step 6: Visualize samples
visualize_samples(X_splits, y_splits)

print("\n=== Pipeline Complete ===")
print(f"Dataset saved in: {dataset_dir}")
print("Files created:")
for file in dataset_dir.glob("*"):
    print(f"  - {file.name}")