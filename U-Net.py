import argparse
import json
import pickle
import time
import warnings
from pathlib import Path

import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from shapely.geometry import Polygon
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

warnings.filterwarnings('ignore')


def setup_gpu():
    """Configure GPU settings for optimal performance"""
    print("=== GPU Configuration ===")

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            print(f"Found {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu}")

            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

            # Enable mixed precision for better performance
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


class LandfillSegmentationUNet:
    """
    U-Net model for landfill detection via semantic segmentation.
    Takes composite images (3-channel: NDVI, Temperature, Methane) and outputs binary masks.
    """

    def __init__(self, input_size=(256, 256), input_channels=3, depth=4, base_filters=64,
                 lite_mode=False, use_attention=False):
        """
        Initialize U-Net for landfill segmentation

        Args:
            input_size: (height, width) of input images
            input_channels: Number of input channels (3 for NDVI, Temp, Methane)
            depth: Number of levels in U-Net
            base_filters: Number of filters in first layer
            lite_mode: Use lighter architecture for faster training
            use_attention: Add attention gates for better feature focusing
        """
        self.input_size = input_size
        self.input_channels = input_channels
        self.depth = depth
        self.base_filters = base_filters
        self.lite_mode = lite_mode
        self.use_attention = use_attention
        self.model = None
        self.history = None

    def _conv_block(self, inputs, filters, kernel_size=3, activation='relu', padding='same'):
        """Convolutional block with batch normalization and dropout"""
        if self.lite_mode:
            # Lighter version: single conv + batch norm
            x = layers.Conv2D(filters, kernel_size, activation=activation, padding=padding)(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.1)(x)
            return x
        else:
            # Standard version: double conv + batch norm + dropout
            x = layers.Conv2D(filters, kernel_size, activation=activation, padding=padding)(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.1)(x)
            x = layers.Conv2D(filters, kernel_size, activation=activation, padding=padding)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.1)(x)
            return x

    def _attention_gate(self, gate_signal, skip_connection, filters):
        """Attention gate for focusing on relevant features"""
        # Gate signal processing
        gate = layers.Conv2D(filters, 1, padding='same')(gate_signal)
        gate = layers.BatchNormalization()(gate)

        # Skip connection processing
        skip = layers.Conv2D(filters, 1, padding='same')(skip_connection)
        skip = layers.BatchNormalization()(skip)

        # Attention computation
        attention = layers.Add()([gate, skip])
        attention = layers.Activation('relu')(attention)
        attention = layers.Conv2D(1, 1, padding='same')(attention)
        attention = layers.Activation('sigmoid')(attention)

        # Apply attention
        attended = layers.Multiply()([skip_connection, attention])

        return attended

    def _encoder_block(self, inputs, filters):
        """Encoder block with convolution and pooling"""
        conv = self._conv_block(inputs, filters)
        pool = layers.MaxPooling2D(pool_size=(2, 2))(conv)
        return conv, pool

    def _decoder_block(self, inputs, skip_features, filters):
        """Decoder block with upsampling and skip connections"""
        # Upsampling
        upsample = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding='same')(inputs)

        # Apply attention gate if enabled
        if self.use_attention:
            skip_features = self._attention_gate(upsample, skip_features, filters // 4)

        # Concatenate with skip connection
        concat = layers.Concatenate()([upsample, skip_features])
        conv = self._conv_block(concat, filters)
        return conv

    def build_model(self):
        """Build U-Net architecture for semantic segmentation"""
        # Input layer - 3 channels for composite environmental indices
        inputs = layers.Input(shape=(*self.input_size, self.input_channels))

        # Encoder path
        skip_connections = []
        x = inputs

        for i in range(self.depth):
            if self.lite_mode:
                filters = self.base_filters * (2 ** min(i, 2))  # Cap at 4x base filters
            else:
                filters = self.base_filters * (2 ** i)

            skip_conv, x = self._encoder_block(x, filters)
            skip_connections.append(skip_conv)

        # Bottleneck
        bottleneck_filters = self.base_filters * (2 ** min(self.depth, 3)) if self.lite_mode else self.base_filters * (
                    2 ** self.depth)
        x = self._conv_block(x, bottleneck_filters)

        # Decoder path
        skip_connections = skip_connections[::-1]  # Reverse for decoder

        for i in range(self.depth):
            if self.lite_mode:
                filters = self.base_filters * (2 ** min(self.depth - 1 - i, 2))
            else:
                filters = self.base_filters * (2 ** (self.depth - 1 - i))

            x = self._decoder_block(x, skip_connections[i], filters)

        # Output layer - sigmoid activation for binary segmentation
        outputs = layers.Conv2D(1, 1, activation='sigmoid', padding='same', dtype='float32')(x)

        # Create model
        model_name = f'LandfillSegmentation_UNet_{"Lite" if self.lite_mode else "Standard"}'
        if self.use_attention:
            model_name += "_Attention"

        self.model = keras.Model(inputs, outputs, name=model_name)

        return self.model

    def compile_model(self, learning_rate=1e-4, use_mixed_precision=True):
        """Compile model with appropriate loss functions for segmentation"""

        # Segmentation-specific loss functions
        def dice_loss(y_true, y_pred, smooth=1e-6):
            """Dice loss for binary segmentation"""
            y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
            y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

            intersection = tf.reduce_sum(y_true_f * y_pred_f)
            dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
            return 1 - dice

        def combined_loss(y_true, y_pred):
            """Combined Binary Crossentropy + Dice Loss"""
            bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            dice = dice_loss(y_true, y_pred)
            return 0.5 * bce + 0.5 * dice

        def iou_metric(y_true, y_pred, threshold=0.5):
            """Intersection over Union metric"""
            y_pred_thresh = tf.cast(y_pred > threshold, tf.float32)
            intersection = tf.reduce_sum(y_true * y_pred_thresh)
            union = tf.reduce_sum(tf.cast(tf.logical_or(tf.cast(y_true, tf.bool),
                                                        tf.cast(y_pred_thresh, tf.bool)), tf.float32))
            return intersection / (union + 1e-7)

        # Optimizer
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        # For mixed precision, wrap optimizer
        if use_mixed_precision and tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
            print("Using mixed precision optimizer for segmentation")

        # Compile with segmentation-appropriate metrics
        self.model.compile(
            optimizer=optimizer,
            loss=combined_loss,
            metrics=[
                'binary_accuracy',
                iou_metric,
                dice_loss,
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )

        return self.model

    def summary(self):
        """Print model summary"""
        if self.model is None:
            self.build_model()
        return self.model.summary()


class LandfillDataLoader:
    """
    Data loader for the generated landfill dataset
    """

    def __init__(self, dataset_dir, batch_size=32, validation_split=0.2, test_split=0.1):
        """
        Initialize data loader

        Args:
            dataset_dir: Path to DataSet directory containing images and polygons folders
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
            test_split: Fraction of data for testing
        """
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / "images"
        self.polygons_dir = self.dataset_dir / "polygons"
        self.metadata_dir = self.dataset_dir / "metadata"
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split

        # Load dataset info
        self._load_dataset_info()

    def _load_dataset_info(self):
        """Load dataset information and create train/val/test splits"""
        print("Loading dataset information...")

        # Get all image files
        image_files = sorted(list(self.images_dir.glob("*.png")))
        print(f"Found {len(image_files)} images")

        # Verify corresponding mask files exist
        valid_files = []
        for img_file in image_files:
            mask_file = self.polygons_dir / img_file.name
            if mask_file.exists():
                valid_files.append(img_file.stem)  # Store filename without extension

        print(f"Found {len(valid_files)} valid image-mask pairs")

        # Shuffle and split
        np.random.seed(42)  # For reproducible splits
        indices = np.random.permutation(len(valid_files))

        # Calculate split sizes
        n_test = int(len(valid_files) * self.test_split)
        n_val = int(len(valid_files) * self.validation_split)
        n_train = len(valid_files) - n_test - n_val

        # Create splits
        self.train_files = [valid_files[i] for i in indices[:n_train]]
        self.val_files = [valid_files[i] for i in indices[n_train:n_train + n_val]]
        self.test_files = [valid_files[i] for i in indices[n_train + n_val:]]

        print(f"Dataset splits:")
        print(f"  Training: {len(self.train_files)}")
        print(f"  Validation: {len(self.val_files)}")
        print(f"  Testing: {len(self.test_files)}")

        # Load metadata if available
        metadata_file = self.metadata_dir / "dataset_summary.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
                print(f"Loaded dataset metadata")
        else:
            self.metadata = None

    def _load_image_and_mask(self, filename):
        """Load a single image and corresponding mask"""
        # Load composite image (RGB-like with environmental indices)
        img_path = self.images_dir / f"{filename}.png"
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Load binary mask
        mask_path = self.polygons_dir / f"{filename}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        return image, mask

    def _preprocess_data(self, image, mask):
        """Preprocess image and mask for training"""
        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Normalize mask to [0, 1] (binary)
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=-1)  # Add channel dimension

        return image, mask

    def create_data_generators(self, augment=True):
        """Create data generators for training, validation, and testing"""

        def data_generator(file_list, augment=False):
            """Generator function for batches"""
            while True:
                # Shuffle files each epoch
                np.random.shuffle(file_list)

                for i in range(0, len(file_list), self.batch_size):
                    batch_files = file_list[i:i + self.batch_size]

                    batch_images = []
                    batch_masks = []

                    for filename in batch_files:
                        try:
                            image, mask = self._load_image_and_mask(filename)
                            image, mask = self._preprocess_data(image, mask)

                            # Simple data augmentation
                            if augment and np.random.random() > 0.5:
                                # Random horizontal flip
                                if np.random.random() > 0.5:
                                    image = np.fliplr(image)
                                    mask = np.fliplr(mask)

                                # Random vertical flip
                                if np.random.random() > 0.5:
                                    image = np.flipud(image)
                                    mask = np.flipud(mask)

                                # Random 90-degree rotation
                                if np.random.random() > 0.5:
                                    k = np.random.randint(1, 4)  # 90, 180, or 270 degrees
                                    image = np.rot90(image, k)
                                    mask = np.rot90(mask, k)

                            batch_images.append(image)
                            batch_masks.append(mask)

                        except Exception as e:
                            print(f"Error loading {filename}: {e}")
                            continue

                    if batch_images:  # Only yield if we have valid data
                        yield np.array(batch_images), np.array(batch_masks)

        # Create generators
        train_gen = data_generator(self.train_files, augment=augment)
        val_gen = data_generator(self.val_files, augment=False)
        test_gen = data_generator(self.test_files, augment=False)

        # Calculate steps per epoch
        steps_per_epoch = len(self.train_files) // self.batch_size
        validation_steps = len(self.val_files) // self.batch_size
        test_steps = len(self.test_files) // self.batch_size

        return (train_gen, val_gen, test_gen), (steps_per_epoch, validation_steps, test_steps)


class LandfillSegmentationTrainer:
    """
    Trainer class for landfill segmentation U-Net
    """

    def __init__(self, dataset_dir, model_dir="segmentation_models",
                 input_size=(256, 256), use_gpu=True, lite_mode=False):
        """
        Initialize trainer

        Args:
            dataset_dir: Directory containing the generated dataset
            model_dir: Directory to save models
            input_size: Input image size
            use_gpu: Whether to use GPU acceleration
            lite_mode: Use lighter model for faster training
        """
        self.dataset_dir = Path(dataset_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.input_size = input_size
        self.lite_mode = lite_mode

        # Setup GPU if requested
        if use_gpu:
            self.gpu_available = setup_gpu()
        else:
            self.gpu_available = False
            print("GPU acceleration disabled")

        # Initialize model
        self.unet = LandfillSegmentationUNet(
            input_size=input_size,
            input_channels=3,  # NDVI, Temperature, Methane
            lite_mode=lite_mode,
            use_attention=not lite_mode  # Use attention gates in standard mode
        )

        # Initialize data loader
        self.data_loader = None

    def prepare_data(self, batch_size=32, validation_split=0.2, test_split=0.1):
        """Prepare data loaders"""
        print("Preparing data loaders...")
        self.data_loader = LandfillDataLoader(
            dataset_dir=self.dataset_dir,
            batch_size=batch_size,
            validation_split=validation_split,
            test_split=test_split
        )

        return self.data_loader

    def train(self, epochs=100, batch_size=32, learning_rate=1e-4,
              early_stopping_patience=15, augment_data=True):
        """Train the segmentation model"""

        # Prepare data if not already done
        if self.data_loader is None:
            self.prepare_data(batch_size, validation_split=0.2, test_split=0.1)

        # Build and compile model
        print("Building and compiling model...")
        self.unet.build_model()
        self.unet.compile_model(learning_rate=learning_rate, use_mixed_precision=self.gpu_available)

        print("Model summary:")
        if not self.lite_mode:
            self.unet.summary()
        else:
            print(f"Model parameters: {self.unet.model.count_params():,}")

        # Get data generators
        generators, steps = self.data_loader.create_data_generators(augment=augment_data)
        train_gen, val_gen, test_gen = generators
        steps_per_epoch, validation_steps, test_steps = steps

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=early_stopping_patience // 3,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                self.model_dir / "best_segmentation_model.h5",
                monitor='val_iou_metric',
                mode='max',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            keras.callbacks.CSVLogger(
                self.model_dir / "training_log.csv"
            )
        ]

        # Add TensorBoard callback
        if not self.lite_mode:
            callbacks.append(keras.callbacks.TensorBoard(
                log_dir=self.model_dir / "logs",
                histogram_freq=0,
                write_graph=False,
                update_freq='epoch'
            ))

        print(f"Starting training...")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Validation steps: {validation_steps}")
        print(f"  Data augmentation: {augment_data}")
        print(f"  Mode: {'LITE' if self.lite_mode else 'STANDARD'}")

        start_time = time.time()

        # Train model
        with tf.device('/GPU:0' if self.gpu_available else '/CPU:0'):
            self.history = self.unet.model.fit(
                train_gen,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_data=val_gen,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=1
            )

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        # Save final model
        self.unet.model.save(self.model_dir / "final_segmentation_model.h5")

        # Save training history
        with open(self.model_dir / "training_history.pkl", 'wb') as f:
            pickle.dump(self.history.history, f)

        return self.history

    def evaluate(self, model_path=None):
        """Evaluate model on test set"""
        print("Evaluating model on test set...")

        # Load model if path provided
        if model_path:
            try:
                self.unet.model = keras.models.load_model(model_path, compile=False)
                self.unet.compile_model(use_mixed_precision=self.gpu_available)
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                return None
        elif (self.model_dir / "best_segmentation_model.h5").exists():
            try:
                self.unet.model = keras.models.load_model(
                    self.model_dir / "best_segmentation_model.h5",
                    compile=False
                )
                self.unet.compile_model(use_mixed_precision=self.gpu_available)
                print("Loaded best model")
            except Exception as e:
                print(f"Error loading best model: {e}")
                return None

        # Prepare data if needed
        if self.data_loader is None:
            self.prepare_data()

        # Get test generator
        generators, steps = self.data_loader.create_data_generators(augment=False)
        _, _, test_gen = generators
        _, _, test_steps = steps

        # Evaluate
        results = self.unet.model.evaluate(test_gen, steps=test_steps, verbose=1)

        # Create results dictionary
        metric_names = self.unet.model.metrics_names
        evaluation_results = dict(zip(metric_names, results))

        print(f"Test Results:")
        for metric, value in evaluation_results.items():
            print(f"  {metric}: {value:.4f}")

        # Save results
        with open(self.model_dir / "evaluation_results.json", 'w') as f:
            json.dump(evaluation_results, f, indent=2)

        return evaluation_results

    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # IoU
        axes[0, 1].plot(self.history.history['iou_metric'], label='Training IoU')
        axes[0, 1].plot(self.history.history['val_iou_metric'], label='Validation IoU')
        axes[0, 1].set_title('Intersection over Union')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('IoU')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Binary Accuracy
        axes[0, 2].plot(self.history.history['binary_accuracy'], label='Training Accuracy')
        axes[0, 2].plot(self.history.history['val_binary_accuracy'], label='Validation Accuracy')
        axes[0, 2].set_title('Binary Accuracy')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].legend()
        axes[0, 2].grid(True)

        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        # Learning rate (if available)
        if 'lr' in self.history.history:
            axes[1, 2].plot(self.history.history['lr'])
            axes[1, 2].set_title('Learning Rate')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Learning Rate')
            axes[1, 2].set_yscale('log')
            axes[1, 2].grid(True)
        else:
            axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(self.model_dir / "training_history.png", dpi=150, bbox_inches='tight')
        plt.show()


class LandfillInference:
    """
    Comprehensive inference class for trained landfill segmentation models.
    Converts predictions to polygons and handles various output formats.
    """

    def __init__(self, model_path, input_size=(256, 256), original_crs="EPSG:4326"):
        """
        Initialize inference engine

        Args:
            model_path: Path to trained model
            input_size: Expected input size
            original_crs: Coordinate reference system for georeferencing
        """
        self.input_size = input_size
        self.original_crs = original_crs

        # Load model with error handling
        try:
            self.model = keras.models.load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            try:
                self.model = keras.models.load_model(model_path, compile=False)
                print("Model loaded without compilation info")
            except Exception as e2:
                raise ValueError(f"Could not load model: {e2}")

    def _preprocess_image(self, image):
        """Preprocess image for model input"""
        # Ensure RGB format
        if len(image.shape) == 3 and image.shape[2] == 3:
            pass  # Already RGB
        elif len(image.shape) == 3 and image.shape[2] == 4:
            image = image[:, :, :3]  # Remove alpha channel
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")

        # Resize if necessary
        if image.shape[:2] != self.input_size:
            image = cv2.resize(image, self.input_size)

        # Normalize to [0, 1]
        image_norm = image.astype(np.float32) / 255.0

        return image_norm

    def _mask_to_polygons(self, mask, min_area=100, simplify_tolerance=1.0):
        """
        Convert binary mask to polygon geometries

        Args:
            mask: Binary mask (2D numpy array)
            min_area: Minimum area threshold for polygons
            simplify_tolerance: Simplification tolerance for polygon smoothing

        Returns:
            List of Polygon objects
        """
        # Ensure binary mask
        binary_mask = (mask > 0.5).astype(np.uint8)

        # Apply morphological operations to clean up mask
        kernel = np.ones((3, 3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polygons = []
        for contour in contours:
            # Filter small contours
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            # Convert contour to polygon
            if len(contour) >= 3:
                # Reshape contour points
                points = contour.reshape(-1, 2)

                try:
                    # Create polygon
                    polygon = Polygon(points)

                    # Simplify polygon if needed
                    if simplify_tolerance > 0:
                        polygon = polygon.simplify(simplify_tolerance, preserve_topology=True)

                    # Check if polygon is valid
                    if polygon.is_valid and polygon.area > min_area:
                        polygons.append(polygon)

                except Exception as e:
                    print(f"Warning: Could not create polygon from contour: {e}")
                    continue

        return polygons

    def _polygons_to_geojson(self, polygons, image_path=None, transform=None):
        """
        Convert polygons to GeoJSON format

        Args:
            polygons: List of Polygon objects
            image_path: Optional path to original image for metadata
            transform: Optional rasterio transform for georeferencing

        Returns:
            GeoJSON-like dictionary
        """
        features = []

        for i, polygon in enumerate(polygons):
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [list(polygon.exterior.coords)]
                },
                "properties": {
                    "id": i,
                    "area": polygon.area,
                    "perimeter": polygon.length,
                    "confidence": 1.0  # Could be enhanced with actual confidence scores
                }
            }

            if image_path:
                feature["properties"]["source_image"] = str(image_path)

            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "total_polygons": len(polygons),
                "total_area": sum(p.area for p in polygons),
                "crs": self.original_crs
            }
        }

        return geojson

    def predict_single_image(self, image_path, threshold=0.5, min_area=100,
                             simplify_tolerance=1.0, return_polygons=True,
                             save_results=None):
        """
        Predict landfill areas in a single image and return polygons

        Args:
            image_path: Path to input image
            threshold: Threshold for binary prediction
            min_area: Minimum area for polygons
            simplify_tolerance: Polygon simplification tolerance
            return_polygons: Whether to return polygon objects
            save_results: Directory to save results (optional)

        Returns:
            Dictionary containing prediction results
        """
        # Load image
        image_path = Path(image_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image = image.copy()
        original_size = image.shape[:2]

        # Preprocess for model
        image_processed = self._preprocess_image(image)
        image_batch = np.expand_dims(image_processed, axis=0)

        # Predict
        print(f"Running inference on {image_path.name}...")
        prediction = self.model.predict(image_batch, verbose=0)[0]

        # Resize prediction back to original size
        if original_size != self.input_size:
            prediction_resized = cv2.resize(
                prediction.squeeze(),
                (original_size[1], original_size[0])
            )
        else:
            prediction_resized = prediction.squeeze()

        # Create binary mask
        binary_mask = (prediction_resized > threshold).astype(np.uint8)

        # Convert to polygons
        polygons = []
        geojson = None

        if return_polygons:
            print(f"Converting mask to polygons...")
            polygons = self._mask_to_polygons(
                binary_mask,
                min_area=min_area,
                simplify_tolerance=simplify_tolerance
            )

            # Create GeoJSON
            geojson = self._polygons_to_geojson(polygons, image_path)

            print(f"Found {len(polygons)} landfill polygons")

        # Prepare results
        results = {
            "image_path": str(image_path),
            "original_image": original_image,
            "prediction_mask": prediction_resized,
            "binary_mask": binary_mask,
            "polygons": polygons,
            "geojson": geojson,
            "num_polygons": len(polygons),
            "total_area": sum(p.area for p in polygons) if polygons else 0,
            "prediction_stats": {
                "threshold": threshold,
                "min_area": min_area,
                "mean_confidence": float(np.mean(prediction_resized)),
                "max_confidence": float(np.max(prediction_resized)),
                "positive_pixels": int(np.sum(binary_mask))
            }
        }

        # Save results if requested
        if save_results:
            self._save_prediction_results(results, save_results, image_path.stem)

        return results

    def _save_prediction_results(self, results, output_dir, image_name):
        """Save prediction results to files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save binary mask
        mask_path = output_dir / f"{image_name}_mask.png"
        cv2.imwrite(str(mask_path), results["binary_mask"] * 255)

        # Save prediction heatmap
        heatmap_path = output_dir / f"{image_name}_heatmap.png"
        heatmap = (results["prediction_mask"] * 255).astype(np.uint8)
        cv2.imwrite(str(heatmap_path), heatmap)

        # Save GeoJSON
        if results["geojson"]:
            geojson_path = output_dir / f"{image_name}_polygons.geojson"
            with open(geojson_path, 'w') as f:
                json.dump(results["geojson"], f, indent=2)

        # Save visualization
        vis_path = output_dir / f"{image_name}_visualization.png"
        self._create_visualization(results, vis_path)

        # Save summary stats
        stats_path = output_dir / f"{image_name}_stats.json"
        stats = {
            "num_polygons": results["num_polygons"],
            "total_area": results["total_area"],
            "prediction_stats": results["prediction_stats"]
        }
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"Results saved to {output_dir}")

    def _create_visualization(self, results, save_path):
        """Create comprehensive visualization of results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Original image
        axes[0, 0].imshow(results["original_image"])
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        # Prediction heatmap
        im1 = axes[0, 1].imshow(results["prediction_mask"], cmap='hot', vmin=0, vmax=1)
        axes[0, 1].set_title('Prediction Heatmap')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

        # Binary mask
        axes[0, 2].imshow(results["binary_mask"], cmap='gray')
        axes[0, 2].set_title(f'Binary Mask ({results["num_polygons"]} polygons)')
        axes[0, 2].axis('off')

        # Overlay
        overlay = results["original_image"].copy()
        mask_colored = np.zeros_like(overlay)
        mask_colored[:, :, 0] = results["binary_mask"] * 255  # Red overlay
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title('Overlay')
        axes[1, 0].axis('off')

        # Polygon visualization
        axes[1, 1].imshow(results["original_image"])
        if results["polygons"]:
            for i, polygon in enumerate(results["polygons"]):
                x, y = polygon.exterior.xy
                axes[1, 1].plot(x, y, 'r-', linewidth=2, alpha=0.8)
                # Add polygon ID
                centroid = polygon.centroid
                axes[1, 1].text(centroid.x, centroid.y, str(i),
                                color='yellow', fontweight='bold',
                                ha='center', va='center')
        axes[1, 1].set_title('Detected Polygons')
        axes[1, 1].axis('off')

        # Statistics
        axes[1, 2].axis('off')
        stats_text = f"""
        Prediction Statistics:

        Number of Polygons: {results["num_polygons"]}
        Total Area: {results["total_area"]:.1f} pixels²

        Confidence Stats:
        Mean: {results["prediction_stats"]["mean_confidence"]:.3f}
        Max: {results["prediction_stats"]["max_confidence"]:.3f}

        Threshold: {results["prediction_stats"]["threshold"]}
        Min Area: {results["prediction_stats"]["min_area"]}
        Positive Pixels: {results["prediction_stats"]["positive_pixels"]}
        """

        if results["polygons"]:
            areas = [p.area for p in results["polygons"]]
            stats_text += f"""

        Polygon Areas:
        Min: {min(areas):.1f} pixels²
        Max: {max(areas):.1f} pixels²
        Mean: {np.mean(areas):.1f} pixels²
        """

        axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def batch_predict(self, image_directory, output_directory, threshold=0.5,
                      min_area=100, simplify_tolerance=1.0, save_individual=True):
        """
        Predict on a batch of images and return polygons for all

        Args:
            image_directory: Directory containing input images
            output_directory: Directory to save results
            threshold: Threshold for binary prediction
            min_area: Minimum area for polygons
            simplify_tolerance: Polygon simplification tolerance
            save_individual: Save individual results for each image

        Returns:
            Dictionary with results for all images
        """
        image_dir = Path(image_directory)
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get all image files
        image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(image_dir.glob(ext)))
            image_files.extend(list(image_dir.glob(ext.upper())))

        print(f"Processing {len(image_files)} images...")

        batch_results = {
            "summary": {
                "total_images": len(image_files),
                "successful_predictions": 0,
                "total_polygons": 0,
                "total_area": 0,
                "failed_images": []
            },
            "individual_results": {}
        }

        all_geojson_features = []

        for img_file in tqdm(image_files, desc="Processing images"):
            try:
                # Predict single image
                individual_output = output_dir / img_file.stem if save_individual else None

                result = self.predict_single_image(
                    img_file,
                    threshold=threshold,
                    min_area=min_area,
                    simplify_tolerance=simplify_tolerance,
                    return_polygons=True,
                    save_results=individual_output
                )

                # Store results
                batch_results["individual_results"][img_file.name] = {
                    "num_polygons": result["num_polygons"],
                    "total_area": result["total_area"],
                    "prediction_stats": result["prediction_stats"]
                }

                # Update summary
                batch_results["summary"]["successful_predictions"] += 1
                batch_results["summary"]["total_polygons"] += result["num_polygons"]
                batch_results["summary"]["total_area"] += result["total_area"]

                # Collect all features for combined GeoJSON
                if result["geojson"]:
                    for feature in result["geojson"]["features"]:
                        feature["properties"]["source_image"] = img_file.name
                        all_geojson_features.append(feature)

            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                batch_results["summary"]["failed_images"].append({
                    "filename": img_file.name,
                    "error": str(e)
                })
                continue

        # Save combined results
        self._save_batch_results(batch_results, all_geojson_features, output_dir)

        print(f"Batch processing completed!")
        print(f"Successful: {batch_results['summary']['successful_predictions']}/{len(image_files)}")
        print(f"Total polygons found: {batch_results['summary']['total_polygons']}")

        return batch_results

    def _save_batch_results(self, batch_results, all_features, output_dir):
        """Save combined batch results"""
        # Save batch summary
        summary_path = output_dir / "batch_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(batch_results, f, indent=2)

        # Save combined GeoJSON
        if all_features:
            combined_geojson = {
                "type": "FeatureCollection",
                "features": all_features,
                "metadata": {
                    "total_polygons": len(all_features),
                    "total_images": batch_results["summary"]["successful_predictions"],
                    "crs": self.original_crs
                }
            }

            geojson_path = output_dir / "all_polygons.geojson"
            with open(geojson_path, 'w') as f:
                json.dump(combined_geojson, f, indent=2)

        # Create batch visualization
        self._create_batch_visualization(batch_results, output_dir)

    def _create_batch_visualization(self, batch_results, output_dir):
        """Create visualization of batch processing results"""
        individual_results = batch_results["individual_results"]

        if not individual_results:
            return

        # Extract data for plotting
        image_names = list(individual_results.keys())
        polygon_counts = [individual_results[name]["num_polygons"] for name in image_names]
        total_areas = [individual_results[name]["total_area"] for name in image_names]
        mean_confidences = [individual_results[name]["prediction_stats"]["mean_confidence"]
                            for name in image_names]

        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Polygon count distribution
        axes[0, 0].bar(range(len(polygon_counts)), polygon_counts)
        axes[0, 0].set_title('Polygon Count per Image')
        axes[0, 0].set_xlabel('Image Index')
        axes[0, 0].set_ylabel('Number of Polygons')
        axes[0, 0].grid(True, alpha=0.3)

        # Total area distribution
        axes[0, 1].bar(range(len(total_areas)), total_areas)
        axes[0, 1].set_title('Total Area per Image')
        axes[0, 1].set_xlabel('Image Index')
        axes[0, 1].set_ylabel('Total Area (pixels²)')
        axes[0, 1].grid(True, alpha=0.3)

        # Confidence distribution
        axes[1, 0].hist(mean_confidences, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Mean Confidence Distribution')
        axes[1, 0].set_xlabel('Mean Confidence')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)

        # Summary statistics
        axes[1, 1].axis('off')
        summary_text = f"""
        Batch Processing Summary:

        Total Images: {batch_results["summary"]["total_images"]}
        Successfully Processed: {batch_results["summary"]["successful_predictions"]}
        Failed: {len(batch_results["summary"]["failed_images"])}

        Total Polygons Found: {batch_results["summary"]["total_polygons"]}
        Total Area: {batch_results["summary"]["total_area"]:.1f} pixels²

        Statistics:
        Mean Polygons/Image: {np.mean(polygon_counts):.1f}
        Mean Area/Image: {np.mean(total_areas):.1f} pixels²
        Mean Confidence: {np.mean(mean_confidences):.3f}
        """

        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()
        plt.savefig(output_dir / "batch_summary_plots.png", dpi=150, bbox_inches='tight')
        plt.close()

    def export_to_shapefile(self, geojson_path, shapefile_path):
        """
        Convert GeoJSON to Shapefile

        Args:
            geojson_path: Path to GeoJSON file
            shapefile_path: Output path for shapefile
        """
        try:
            # Read GeoJSON
            gdf = gpd.read_file(geojson_path)

            # Save as shapefile
            gdf.to_file(shapefile_path)
            print(f"Shapefile saved to {shapefile_path}")

        except Exception as e:
            print(f"Error converting to shapefile: {e}")

    def get_polygon_statistics(self, polygons):
        """
        Calculate detailed statistics for polygon results

        Args:
            polygons: List of Polygon objects

        Returns:
            Dictionary with statistics
        """
        if not polygons:
            return {"count": 0}

        areas = [p.area for p in polygons]
        perimeters = [p.length for p in polygons]

        stats = {
            "count": len(polygons),
            "total_area": sum(areas),
            "area_stats": {
                "min": min(areas),
                "max": max(areas),
                "mean": np.mean(areas),
                "median": np.median(areas),
                "std": np.std(areas)
            },
            "perimeter_stats": {
                "min": min(perimeters),
                "max": max(perimeters),
                "mean": np.mean(perimeters),
                "median": np.median(perimeters),
                "std": np.std(perimeters)
            }
        }

        return stats


class SimpleSegmentationNet:
    """
    Simple 2-layer neural network for quick testing and comparison
    """

    def __init__(self, input_size=(256, 256), input_channels=3):
        """
        Initialize simple segmentation network

        Args:
            input_size: (height, width) of input images
            input_channels: Number of input channels
        """
        self.input_size = input_size
        self.input_channels = input_channels
        self.model = None

    def build_model(self):
        """Build simple 2-layer network"""
        inputs = layers.Input(shape=(*self.input_size, self.input_channels))

        # First layer: Conv2D + ReLU + BatchNorm
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        # Second layer: Conv2D + Sigmoid for binary output
        outputs = layers.Conv2D(1, 3, activation='sigmoid', padding='same', dtype='float32')(x)

        self.model = keras.Model(inputs, outputs, name='SimpleSegmentationNet')
        return self.model

    def compile_model(self, learning_rate=1e-3):
        """Compile simple model"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['binary_accuracy', 'precision', 'recall']
        )

        return self.model


def quick_test_simple_model(dataset_dir, epochs=20):
    """
    Quick test function using the simple 2-layer network

    Args:
        dataset_dir: Path to dataset
        epochs: Number of epochs for quick test
    """
    print("🧪 Quick test with simple 2-layer network...")

    # Initialize simple model
    simple_net = SimpleSegmentationNet(input_size=(256, 256), input_channels=3)
    simple_net.build_model()
    simple_net.compile_model(learning_rate=1e-3)

    print("Simple model summary:")
    simple_net.model.summary()

    # Prepare data
    data_loader = LandfillDataLoader(dataset_dir, batch_size=16)
    generators, steps = data_loader.create_data_generators(augment=False)
    train_gen, val_gen, _ = generators
    steps_per_epoch, validation_steps, _ = steps

    # Quick training
    print(f"Training simple model for {epochs} epochs...")

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    ]

    history = simple_net.model.fit(
        train_gen,
        steps_per_epoch=min(steps_per_epoch, 50),  # Limit steps for quick test
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=min(validation_steps, 10),
        callbacks=callbacks,
        verbose=1
    )

    # Save simple model
    simple_net.model.save("simple_segmentation_model.h5")
    print("Simple model saved as 'simple_segmentation_model.h5'")

    # Plot quick results
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(history.history['binary_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_binary_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(history.history['precision'], label='Training Precision')
    plt.plot(history.history['val_precision'], label='Validation Precision')
    plt.title('Precision')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("simple_model_training.png", dpi=150)
    plt.show()

    print("✅ Quick test completed!")
    return simple_net.model, history


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description='Train U-Net for Landfill Detection via Semantic Segmentation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Directory containing the generated dataset (DataSet folder)')
    parser.add_argument('--model_dir', type=str, default='segmentation_models',
                        help='Directory to save models')
    parser.add_argument('--action', type=str, choices=['train', 'evaluate', 'both'],
                        default='both', help='Action to perform')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                        help='Early stopping patience')

    # Model parameters
    parser.add_argument('--input_size', type=int, nargs=2, default=[256, 256],
                        help='Input image size (height width)')
    parser.add_argument('--lite_mode', action='store_true',
                        help='Use lighter U-Net architecture')

    # Data parameters
    parser.add_argument('--validation_split', type=float, default=0.2,
                        help='Fraction of data for validation')
    parser.add_argument('--test_split', type=float, default=0.1,
                        help='Fraction of data for testing')
    parser.add_argument('--no_augmentation', action='store_true',
                        help='Disable data augmentation')

    # GPU parameters
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='Use GPU acceleration')
    parser.add_argument('--disable_gpu', action='store_true',
                        help='Disable GPU acceleration')

    # Evaluation parameters
    parser.add_argument('--model_path', type=str,
                        help='Path to trained model for evaluation')

    # Quick test parameter
    parser.add_argument('--quick_test', action='store_true',
                        help='Run quick test with simple model')

    # Inference parameters
    parser.add_argument('--inference', action='store_true',
                        help='Run inference on images')
    parser.add_argument('--inference_images', type=str,
                        help='Path to image or directory for inference')
    parser.add_argument('--inference_output', type=str, default='inference_results',
                        help='Output directory for inference results')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary prediction')
    parser.add_argument('--min_area', type=int, default=100,
                        help='Minimum area for polygons (pixels)')
    parser.add_argument('--simplify_tolerance', type=float, default=1.0,
                        help='Polygon simplification tolerance')
    parser.add_argument('--export_shapefile', action='store_true',
                        help='Export results as shapefile')

    args = parser.parse_args()

    # Handle GPU settings
    if args.disable_gpu:
        args.use_gpu = False

    # Inference mode
    if args.inference:
        if not args.inference_images:
            print("Error: --inference_images required for inference mode")
            return

        if not args.model_path:
            # Try to find best model
            model_dir = Path(args.model_dir)
            best_model_path = model_dir / "best_segmentation_model.h5"
            final_model_path = model_dir / "final_segmentation_model.h5"

            if best_model_path.exists():
                args.model_path = str(best_model_path)
                print(f"Using best model: {args.model_path}")
            elif final_model_path.exists():
                args.model_path = str(final_model_path)
                print(f"Using final model: {args.model_path}")
            else:
                print("Error: No trained model found. Please specify --model_path")
                return

        print("🔍 Running inference mode...")

        # Initialize inference engine
        inference_engine = LandfillInference(
            model_path=args.model_path,
            input_size=tuple(args.input_size)
        )

        inference_path = Path(args.inference_images)

        if inference_path.is_file():
            # Single image inference
            print(f"Processing single image: {inference_path}")

            result = inference_engine.predict_single_image(
                image_path=inference_path,
                threshold=args.threshold,
                min_area=args.min_area,
                simplify_tolerance=args.simplify_tolerance,
                return_polygons=True,
                save_results=args.inference_output
            )

            print(f"✅ Single image inference completed!")
            print(f"Found {result['num_polygons']} landfill polygons")
            print(f"Total area: {result['total_area']:.1f} pixels²")

            # Export shapefile if requested
            if args.export_shapefile and result['geojson']:
                geojson_path = Path(args.inference_output) / f"{inference_path.stem}_polygons.geojson"
                shapefile_path = Path(args.inference_output) / f"{inference_path.stem}_polygons.shp"
                inference_engine.export_to_shapefile(geojson_path, shapefile_path)

        elif inference_path.is_dir():
            # Batch inference
            print(f"Processing directory: {inference_path}")

            batch_results = inference_engine.batch_predict(
                image_directory=inference_path,
                output_directory=args.inference_output,
                threshold=args.threshold,
                min_area=args.min_area,
                simplify_tolerance=args.simplify_tolerance,
                save_individual=True
            )

            print(f"✅ Batch inference completed!")
            print(f"Processed {batch_results['summary']['successful_predictions']} images")
            print(f"Total polygons found: {batch_results['summary']['total_polygons']}")

            # Export combined shapefile if requested
            if args.export_shapefile:
                geojson_path = Path(args.inference_output) / "all_polygons.geojson"
                if geojson_path.exists():
                    shapefile_path = Path(args.inference_output) / "all_polygons.shp"
                    inference_engine.export_to_shapefile(geojson_path, shapefile_path)

        else:
            print(f"Error: {inference_path} is neither a file nor a directory")

        return

    args = parser.parse_args()

    # Handle GPU settings
    if args.disable_gpu:
        args.use_gpu = False

    # Quick test mode
    if args.quick_test:
        print("🚀 Running quick test mode...")
        quick_test_simple_model(args.dataset_dir, epochs=20)
        return

    # Initialize trainer
    trainer = LandfillSegmentationTrainer(
        dataset_dir=args.dataset_dir,
        model_dir=args.model_dir,
        input_size=tuple(args.input_size),
        use_gpu=args.use_gpu,
        lite_mode=args.lite_mode
    )

    if args.action in ['train', 'both']:
        print("🚀 Starting training phase...")

        # Prepare data
        trainer.prepare_data(
            batch_size=args.batch_size,
            validation_split=args.validation_split,
            test_split=args.test_split
        )

        # Train model
        history = trainer.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            early_stopping_patience=args.early_stopping_patience,
            augment_data=not args.no_augmentation
        )

        # Plot training history
        trainer.plot_training_history()

    if args.action in ['evaluate', 'both']:
        print("📊 Starting evaluation phase...")

        # Evaluate model
        results = trainer.evaluate(model_path=args.model_path)

        if results:
            print("✅ Evaluation completed successfully!")
        else:
            print("❌ Evaluation failed!")


if __name__ == "__main__":
    main()