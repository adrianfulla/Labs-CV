import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import argparse
import pickle
from tqdm import tqdm
import json
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
import time


def setup_gpu():
    """Configure GPU settings for optimal performance"""
    print("=== GPU Configuration ===")
    
    # Check if GPU is available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            print(f"Found {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu}")
                
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                
            # Set GPU as the preferred device
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            
            # Enable mixed precision for better performance on RTX GPUs
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print(f"Mixed precision enabled: {policy.name}")
            
            print("GPU configuration successful!")
            return True
            
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
            return False
    else:
        print("No GPU found. Using CPU.")
        return False

def get_gpu_memory_info():
    """Get current GPU memory usage"""
    try:
        # This requires nvidia-ml-py package: pip install nvidia-ml-py
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        total = info.total / 1024**3  # Convert to GB
        used = info.used / 1024**3
        free = info.free / 1024**3
        
        print(f"GPU Memory: {used:.1f}GB used, {free:.1f}GB free, {total:.1f}GB total")
        return used, free, total
    except ImportError:
        print("Install nvidia-ml-py for GPU memory monitoring: pip install nvidia-ml-py")
        return None, None, None
    except Exception as e:
        print(f"Could not get GPU memory info: {e}")
        return None, None, None

class UNetAnisotropicFilter:
    def __init__(self, window_size=32, depth=4, base_filters=64, lite_mode=False):
        """
        U-Net model for anisotropic filtering
        
        Args:
            window_size: Size of input windows (k x k)
            depth: Number of levels in U-Net (default: 4)
            base_filters: Number of filters in first layer
            lite_mode: Use lighter architecture for faster training
        """
        self.window_size = window_size
        self.depth = depth
        self.base_filters = base_filters
        self.lite_mode = lite_mode
        self.model = None
        self.history = None
        
    def _conv_block(self, inputs, filters, kernel_size=3, activation='relu', padding='same'):
        """Convolutional block with optional optimizations"""
        if self.lite_mode:
            # Lighter version: single conv + batch norm
            x = layers.Conv2D(filters, kernel_size, activation=activation, padding=padding)(inputs)
            x = layers.BatchNormalization()(x)
            return x
        else:
            # Standard version: double conv + batch norm
            x = layers.Conv2D(filters, kernel_size, activation=activation, padding=padding)(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(filters, kernel_size, activation=activation, padding=padding)(x)
            x = layers.BatchNormalization()(x)
            return x
    
    def _separable_conv_block(self, inputs, filters, kernel_size=3, activation='relu', padding='same'):
        """Separable convolution block for speed (MobileNet style)"""
        x = layers.SeparableConv2D(filters, kernel_size, activation=activation, padding=padding)(inputs)
        x = layers.BatchNormalization()(x)
        if not self.lite_mode:
            x = layers.SeparableConv2D(filters, kernel_size, activation=activation, padding=padding)(x)
            x = layers.BatchNormalization()(x)
        return x
    
    def _encoder_block(self, inputs, filters, use_separable=False):
        """Encoder block with optional separable convolutions"""
        if use_separable:
            conv = self._separable_conv_block(inputs, filters)
        else:
            conv = self._conv_block(inputs, filters)
        pool = layers.MaxPooling2D(pool_size=(2, 2))(conv)
        return conv, pool
    
    def _decoder_block(self, inputs, skip_features, filters, use_separable=False):
        """Decoder block with optional separable convolutions"""
        upsample = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding='same')(inputs)
        concat = layers.Concatenate()([upsample, skip_features])
        if use_separable:
            conv = self._separable_conv_block(concat, filters)
        else:
            conv = self._conv_block(concat, filters)
        return conv
    
    def build_model(self, use_separable=False):
        """Build U-Net architecture with speed optimizations"""
        # Input layer
        inputs = layers.Input(shape=(self.window_size, self.window_size, 1))
        
        # Encoder path
        skip_connections = []
        x = inputs
        
        for i in range(self.depth):
            if self.lite_mode:
                # Reduce filters in lite mode
                filters = self.base_filters * (2 ** min(i, 2))  # Cap at 4x base filters
            else:
                filters = self.base_filters * (2 ** i)
            skip_conv, x = self._encoder_block(x, filters, use_separable)
            skip_connections.append(skip_conv)
        
        # Bottleneck
        bottleneck_filters = self.base_filters * (2 ** min(self.depth, 3)) if self.lite_mode else self.base_filters * (2 ** self.depth)
        if use_separable:
            x = self._separable_conv_block(x, bottleneck_filters)
        else:
            x = self._conv_block(x, bottleneck_filters)
        
        # Decoder path
        skip_connections = skip_connections[::-1]  # Reverse for decoder
        
        for i in range(self.depth):
            if self.lite_mode:
                filters = self.base_filters * (2 ** min(self.depth - 1 - i, 2))
            else:
                filters = self.base_filters * (2 ** (self.depth - 1 - i))
            x = self._decoder_block(x, skip_connections[i], filters, use_separable)
        
        # Output layer
        outputs = layers.Conv2D(1, 1, activation='linear', padding='same')(x)
        
        # Create model
        model_name = f'UNet_AnisotropicFilter_{"Lite" if self.lite_mode else "Standard"}'
        self.model = keras.Model(inputs, outputs, name=model_name)
        
        return self.model
    
    def compile_model(self, learning_rate=1e-4, loss='mse', metrics=None, use_mixed_precision=True):
        """Compile the model with GPU optimizations"""
        if metrics is None:
            metrics = ['mae', 'mse']
            
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # For mixed precision, wrap optimizer
        if use_mixed_precision and tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
            print("Using mixed precision optimizer for better GPU performance")
        
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        return self.model
    
    def summary(self):
        """Print model summary"""
        if self.model is None:
            self.build_model()
        return self.model.summary()

class AnisotropicFilterTrainer:
    def __init__(self, dataset_dir, model_dir="models", window_size=32, use_gpu=True, speed_mode=False):
        """
        Trainer for U-Net anisotropic filter
        
        Args:
            dataset_dir: Directory containing the dataset
            model_dir: Directory to save models
            window_size: Size of input windows
            use_gpu: Whether to use GPU acceleration
            speed_mode: Enable speed optimizations
        """
        self.dataset_dir = Path(dataset_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.window_size = window_size
        self.use_gpu = use_gpu
        self.speed_mode = speed_mode
        
        # Setup GPU if requested
        if use_gpu:
            self.gpu_available = setup_gpu()
        else:
            self.gpu_available = False
            print("GPU acceleration disabled by user")
        
        # Load dataset
        self.load_dataset()
        
        # Initialize model with lite mode if speed mode is enabled
        self.unet = UNetAnisotropicFilter(window_size=window_size, lite_mode=speed_mode)
    
    def get_optimal_batch_size(self, aggressive=False):
        """Get optimal batch size based on GPU memory"""
        if not self.gpu_available:
            return 16  # Conservative CPU batch size
        
        # Get GPU memory info
        used, free, total = get_gpu_memory_info()
        
        if free is None:
            return 32  # Default if we can't check memory
        
        # Rough estimation: each sample uses ~window_size^2 * 4 bytes * 2 (input + output)
        sample_memory = (self.window_size ** 2) * 4 * 2 / 1024**3  # GB per sample
        
        # Use 70% of free memory for batch
        usable_memory = free * 0.7
        optimal_batch_size = int(usable_memory / sample_memory)
        
        # More aggressive in aggressive mode
        if aggressive:
            optimal_batch_size = min(optimal_batch_size * 2, 256)
        
        # Clamp to reasonable range
        optimal_batch_size = max(8, min(optimal_batch_size, 128))
        
        print(f"Suggested batch size based on GPU memory: {optimal_batch_size}")
        return optimal_batch_size
        
    def load_dataset(self):
        """Load the preprocessed dataset"""
        print("Loading dataset...")
        
        # Load data
        self.X_train = np.load(self.dataset_dir / "X_train.npy")
        self.X_val = np.load(self.dataset_dir / "X_val.npy")
        self.X_test = np.load(self.dataset_dir / "X_test.npy")
        self.y_train = np.load(self.dataset_dir / "y_train.npy")
        self.y_val = np.load(self.dataset_dir / "y_val.npy")
        self.y_test = np.load(self.dataset_dir / "y_test.npy")
        
        # Load metadata
        with open(self.dataset_dir / "metadata.pkl", 'rb') as f:
            self.metadata = pickle.load(f)
        
        # Normalize data to [0, 1]
        self.X_train = self.X_train.astype(np.float32) / 255.0
        self.X_val = self.X_val.astype(np.float32) / 255.0
        self.X_test = self.X_test.astype(np.float32) / 255.0
        self.y_train = self.y_train.astype(np.float32) / 255.0
        self.y_val = self.y_val.astype(np.float32) / 255.0
        self.y_test = self.y_test.astype(np.float32) / 255.0
        
        # Add channel dimension
        self.X_train = np.expand_dims(self.X_train, axis=-1)
        self.X_val = np.expand_dims(self.X_val, axis=-1)
        self.X_test = np.expand_dims(self.X_test, axis=-1)
        self.y_train = np.expand_dims(self.y_train, axis=-1)
        self.y_val = np.expand_dims(self.y_val, axis=-1)
        self.y_test = np.expand_dims(self.y_test, axis=-1)
        
        print(f"Dataset loaded:")
        print(f"  Train: {self.X_train.shape} -> {self.y_train.shape}")
        print(f"  Validation: {self.X_val.shape} -> {self.y_val.shape}")
        print(f"  Test: {self.X_test.shape} -> {self.y_test.shape}")
    
    def train(self, epochs=100, batch_size=None, learning_rate=1e-4, 
              early_stopping_patience=10, reduce_lr_patience=5, 
              use_separable_conv=False, aggressive_mode=False, validation_freq=1):
        """Train the U-Net model with speed optimizations"""
        print("Building and compiling model...")
        
        # Auto-detect optimal batch size if not specified
        if batch_size is None:
            batch_size = self.get_optimal_batch_size(aggressive=aggressive_mode)
        
        # Build and compile model
        self.unet.build_model(use_separable=use_separable_conv)
        self.unet.compile_model(learning_rate=learning_rate, use_mixed_precision=self.gpu_available)
        
        print(f"Model summary:")
        if not self.speed_mode:  # Only show full summary in normal mode
            self.unet.summary()
        else:
            print(f"Model parameters: {self.unet.model.count_params():,}")
        
        # Speed-optimized callbacks
        callbacks = []
        
        # More aggressive early stopping in speed mode
        patience = early_stopping_patience // 2 if self.speed_mode else early_stopping_patience
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ))
        
        # More aggressive learning rate reduction
        lr_patience = reduce_lr_patience // 2 if self.speed_mode else reduce_lr_patience
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=lr_patience,
            min_lr=1e-7,
            verbose=1
        ))
        
        # Model checkpoint
        callbacks.append(keras.callbacks.ModelCheckpoint(
            self.model_dir / "best_model.h5",
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ))
        
        # Add GPU memory monitoring callback if available
        if self.gpu_available and not self.speed_mode:  # Skip in speed mode to reduce overhead
            class GPUMemoryCallback(keras.callbacks.Callback):
                def on_epoch_begin(self, epoch, logs=None):
                    if epoch % 10 == 0:  # Check every 10 epochs
                        get_gpu_memory_info()
            
            callbacks.append(GPUMemoryCallback())
        
        # TensorBoard callback for monitoring (optional in speed mode)
        if not self.speed_mode:
            callbacks.append(keras.callbacks.TensorBoard(
                log_dir=self.model_dir / "logs",
                histogram_freq=0,  # Disable histogram to save time
                write_graph=False,  # Disable graph writing to save time
                update_freq='epoch'
            ))
        
        print(f"Starting training...")
        print(f"  Mode: {'SPEED' if self.speed_mode else 'STANDARD'}")
        print(f"  Device: {'GPU' if self.gpu_available else 'CPU'}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Mixed precision: {tf.keras.mixed_precision.global_policy().name}")
        print(f"  Validation frequency: every {validation_freq} epoch(s)")
        print(f"  Early stopping patience: {patience}")
        print(f"  Architecture: {'Lite' if self.unet.lite_mode else 'Standard'}")
        print(f"  Convolutions: {'Separable' if use_separable_conv else 'Standard'}")
        
        start_time = time.time()
        
        # Create data generators for better memory efficiency
        if aggressive_mode:
            # Use tf.data for better performance
            train_dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
            train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            
            val_dataset = tf.data.Dataset.from_tensor_slices((self.X_val, self.y_val))
            val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            
            # Train with tf.data
            with tf.device('/GPU:0' if self.gpu_available else '/CPU:0'):
                self.history = self.unet.model.fit(
                    train_dataset,
                    validation_data=val_dataset,
                    epochs=epochs,
                    callbacks=callbacks,
                    verbose=1,
                    validation_freq=validation_freq
                )
        else:
            # Standard training
            with tf.device('/GPU:0' if self.gpu_available else '/CPU:0'):
                self.history = self.unet.model.fit(
                    self.X_train, self.y_train,
                    validation_data=(self.X_val, self.y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    verbose=1,
                    validation_freq=validation_freq,
                    use_multiprocessing=True if not self.gpu_available else False,
                    workers=4 if not self.gpu_available else 1
                )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Average time per epoch: {training_time/len(self.history.history['loss']):.2f} seconds")
        
        # Save final model
        self.unet.model.save(self.model_dir / "final_model.h5")
        
        # Save training history
        with open(self.model_dir / "training_history.pkl", 'wb') as f:
            pickle.dump(self.history.history, f)
        
        return self.history
    
    def evaluate(self):
        """Evaluate model on test set"""
        print("Evaluating model on test set...")
        
        # Load best model
        self.unet.model = keras.models.load_model(self.model_dir / "best_model.h5")
        
        # Evaluate
        test_loss, test_mae, test_mse = self.unet.model.evaluate(
            self.X_test, self.y_test, verbose=0
        )
        
        # Additional metrics
        y_pred = self.unet.model.predict(self.X_test, verbose=0)
        
        # Convert back to [0, 255] for PSNR calculation
        y_test_255 = (self.y_test * 255).astype(np.uint8)
        y_pred_255 = np.clip(y_pred * 255, 0, 255).astype(np.uint8)
        
        # Calculate PSNR for each sample and average
        psnr_scores = []
        for i in range(len(y_test_255)):
            psnr = peak_signal_noise_ratio(y_test_255[i], y_pred_255[i])
            psnr_scores.append(psnr)
        
        avg_psnr = np.mean(psnr_scores)
        
        print(f"Test Results:")
        print(f"  Loss: {test_loss:.6f}")
        print(f"  MAE: {test_mae:.6f}")
        print(f"  MSE: {test_mse:.6f}")
        print(f"  PSNR: {avg_psnr:.2f} dB")
        
        # Save evaluation results
        results = {
            'test_loss': float(test_loss),
            'test_mae': float(test_mae),
            'test_mse': float(test_mse),
            'test_psnr': float(avg_psnr)
        }
        
        with open(self.model_dir / "evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MAE
        axes[0, 1].plot(self.history.history['mae'], label='Training MAE')
        axes[0, 1].plot(self.history.history['val_mae'], label='Validation MAE')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # MSE
        axes[1, 0].plot(self.history.history['mse'], label='Training MSE')
        axes[1, 0].plot(self.history.history['val_mse'], label='Validation MSE')
        axes[1, 0].set_title('Mean Squared Error')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MSE')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate (if available)
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.model_dir / "training_history.png", dpi=150, bbox_inches='tight')
        plt.show()

class ImageInference:
    def __init__(self, model_path, window_size=32):
        """
        Image inference using trained U-Net
        
        Args:
            model_path: Path to trained model
            window_size: Size of windows for inference
        """
        self.model = keras.models.load_model(model_path)
        self.window_size = window_size
    
    def extract_windows(self, image, stride=None):
        """
        Extract overlapping windows from an image
        
        Args:
            image: Input image (H x W)
            stride: Step size for window extraction (default: window_size//2)
        
        Returns:
            windows: Array of windows (N, window_size, window_size, 1)
            positions: List of (y, x) positions for each window
        """
        if stride is None:
            stride = self.window_size // 2
        
        h, w = image.shape
        windows = []
        positions = []
        
        # Extract windows with stride
        for y in range(0, h - self.window_size + 1, stride):
            for x in range(0, w - self.window_size + 1, stride):
                window = image[y:y+self.window_size, x:x+self.window_size]
                windows.append(window)
                positions.append((y, x))
        
        # Handle edge cases - extract windows at borders
        # Right edge
        if w % stride != 0:
            for y in range(0, h - self.window_size + 1, stride):
                x = w - self.window_size
                window = image[y:y+self.window_size, x:x+self.window_size]
                windows.append(window)
                positions.append((y, x))
        
        # Bottom edge
        if h % stride != 0:
            for x in range(0, w - self.window_size + 1, stride):
                y = h - self.window_size
                window = image[y:y+self.window_size, x:x+self.window_size]
                windows.append(window)
                positions.append((y, x))
        
        # Bottom-right corner
        if h % stride != 0 and w % stride != 0:
            y = h - self.window_size
            x = w - self.window_size
            window = image[y:y+self.window_size, x:x+self.window_size]
            windows.append(window)
            positions.append((y, x))
        
        windows = np.array(windows)
        windows = np.expand_dims(windows, axis=-1)  # Add channel dimension
        
        return windows, positions
    
    def reconstruct_image(self, predicted_windows, positions, original_shape, stride=None):
        """
        Reconstruct image from predicted windows with overlap averaging
        
        Args:
            predicted_windows: Predicted windows (N, window_size, window_size, 1)
            positions: List of (y, x) positions for each window
            original_shape: Shape of original image (H, W)
            stride: Step size used for window extraction
        
        Returns:
            reconstructed_image: Reconstructed image (H, W)
        """
        if stride is None:
            stride = self.window_size // 2
        
        h, w = original_shape
        reconstructed = np.zeros((h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)
        
        # Accumulate predictions and weights
        for window, (y, x) in zip(predicted_windows, positions):
            window = window.squeeze()  # Remove channel dimension
            reconstructed[y:y+self.window_size, x:x+self.window_size] += window
            weight_map[y:y+self.window_size, x:x+self.window_size] += 1
        
        # Average overlapping regions
        reconstructed = np.divide(reconstructed, weight_map, 
                                out=np.zeros_like(reconstructed), 
                                where=weight_map!=0)
        
        return reconstructed
    
    def predict_image(self, image, stride=None, batch_size=None):
        """
        Predict filtered version of an image with GPU optimization
        
        Args:
            image: Input image (H x W) in range [0, 255]
            stride: Step size for window extraction
            batch_size: Batch size for prediction (auto-detect if None)
        
        Returns:
            filtered_image: Predicted filtered image (H x W) in range [0, 255]
        """
        # Auto-detect batch size if not specified
        if batch_size is None:
            batch_size = 64 if tf.config.list_physical_devices('GPU') else 16
        
        # Normalize image
        image_norm = image.astype(np.float32) / 255.0
        
        # Extract windows
        windows, positions = self.extract_windows(image_norm, stride)
        
        # Predict in batches with GPU acceleration
        predicted_windows = []
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            for i in tqdm(range(0, len(windows), batch_size), desc="Predicting windows"):
                batch = windows[i:i+batch_size]
                pred_batch = self.model.predict(batch, verbose=0)
                predicted_windows.extend(pred_batch)
        
        predicted_windows = np.array(predicted_windows)
        
        # Reconstruct image
        reconstructed = self.reconstruct_image(predicted_windows, positions, 
                                             image.shape, stride)
        
        # Convert back to [0, 255]
        filtered_image = np.clip(reconstructed * 255, 0, 255).astype(np.uint8)
        
        return filtered_image

def main():
    parser = argparse.ArgumentParser(
        description='Train U-Net for Anisotropic Filtering',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--dataset_dir', type=str, required=True,
                       help='Directory containing the dataset')
    parser.add_argument('--model_dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--action', type=str, choices=['train', 'evaluate', 'infer'], 
                       default='train', help='Action to perform')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Model parameters
    parser.add_argument('--window_size', type=int, default=32,
                       help='Window size (should match dataset)')
    parser.add_argument('--depth', type=int, default=4,
                       help='U-Net depth (number of levels)')
    parser.add_argument('--base_filters', type=int, default=64,
                       help='Base number of filters')
    
    # Speed optimization parameters
    parser.add_argument('--speed_mode', action='store_true',
                       help='Enable aggressive speed optimizations')
    parser.add_argument('--lite_model', action='store_true',
                       help='Use lighter U-Net architecture')
    parser.add_argument('--separable_conv', action='store_true',
                       help='Use separable convolutions (faster but different quality)')
    parser.add_argument('--aggressive_mode', action='store_true',
                       help='Most aggressive optimizations (may impact quality)')
    parser.add_argument('--reduced_data', type=float, default=1.0,
                       help='Fraction of training data to use (0.1-1.0)')
    parser.add_argument('--validation_freq', type=int, default=1,
                       help='Validate every N epochs (higher = faster)')
    
    # Inference parameters
    parser.add_argument('--input_image', type=str,
                       help='Input image for inference')
    parser.add_argument('--output_image', type=str,
                       help='Output image path for inference')
    parser.add_argument('--model_path', type=str,
                       help='Path to trained model for inference')
    parser.add_argument('--stride', type=int,
                       help='Stride for window extraction (default: window_size//2)')
    
    # GPU optimization parameters
    parser.add_argument('--use_gpu', action='store_true', default=True,
                       help='Use GPU acceleration (default: True)')
    parser.add_argument('--disable_gpu', action='store_true',
                       help='Disable GPU acceleration')
    parser.add_argument('--auto_batch_size', action='store_true', default=True,
                       help='Automatically determine optimal batch size')
    parser.add_argument('--gpu_memory_limit', type=int,
                       help='Limit GPU memory usage (MB)')
    
    args = parser.parse_args()
    
    # Handle GPU settings
    if args.disable_gpu:
        args.use_gpu = False
    
    # Handle speed mode implications
    if args.speed_mode:
        args.lite_model = True
        args.aggressive_mode = True
        if args.reduced_data == 1.0:
            args.reduced_data = 0.2  # Use 20% of data in speed mode
        if args.validation_freq == 1:
            args.validation_freq = 5  # Validate every 5 epochs
        if args.early_stopping_patience == 10:
            args.early_stopping_patience = 5  # Reduce patience
    
    # Set GPU memory limit if specified
    if args.gpu_memory_limit and args.use_gpu:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_memory_growth(gpus[0], False)
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=args.gpu_memory_limit)]
                )
                print(f"GPU memory limited to {args.gpu_memory_limit}MB")
            except RuntimeError as e:
                print(f"Error setting GPU memory limit: {e}")
    
    # Auto batch size handling
    if args.auto_batch_size:
        args.batch_size = None  # Will be auto-detected
    
    # Validate reduced_data parameter
    if args.reduced_data <= 0 or args.reduced_data > 1:
        parser.error("reduced_data must be between 0.1 and 1.0")
    
    if args.action == 'train':
        # Training with speed optimization
        trainer = AnisotropicFilterTrainer(
            dataset_dir=args.dataset_dir,
            model_dir=args.model_dir,
            window_size=args.window_size,
            use_gpu=args.use_gpu,
            speed_mode=args.speed_mode or args.lite_model
        )
        
        # Apply data reduction if specified
        if args.reduced_data < 1.0:
            print(f"Using {args.reduced_data:.1%} of training data for faster training")
            train_size = int(len(trainer.X_train) * args.reduced_data)
            val_size = int(len(trainer.X_val) * args.reduced_data)
            
            trainer.X_train = trainer.X_train[:train_size]
            trainer.y_train = trainer.y_train[:train_size]
            trainer.X_val = trainer.X_val[:val_size]
            trainer.y_val = trainer.y_val[:val_size]
            
            print(f"Reduced dataset: {len(trainer.X_train)} train, {len(trainer.X_val)} val")
        
        # Train model with speed optimizations
        trainer.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            early_stopping_patience=args.early_stopping_patience,
            use_separable_conv=args.separable_conv,
            aggressive_mode=args.aggressive_mode,
            validation_freq=args.validation_freq
        )
        
        # Plot training history (skip in aggressive mode for speed)
        if not args.aggressive_mode:
            trainer.plot_training_history()
        
        # Evaluate
        trainer.evaluate()
        
    elif args.action == 'evaluate':
        # Evaluation only
        trainer = AnisotropicFilterTrainer(
            dataset_dir=args.dataset_dir,
            model_dir=args.model_dir,
            window_size=args.window_size,
            use_gpu=args.use_gpu
        )
        trainer.evaluate()
        
    elif args.action == 'infer':
        # Inference
        if not args.input_image or not args.output_image or not args.model_path:
            parser.error("For inference, you must specify --input_image, --output_image, and --model_path")
        
        print(f"Loading model from {args.model_path}")
        inference = ImageInference(args.model_path, args.window_size)
        
        print(f"Loading image from {args.input_image}")
        image = cv2.imread(args.input_image, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image from {args.input_image}")
        
        print("Running inference...")
        filtered_image = inference.predict_image(image, stride=args.stride)
        
        print(f"Saving result to {args.output_image}")
        cv2.imwrite(args.output_image, filtered_image)
        
        print("Inference complete!")

if __name__ == "__main__":
    main()