import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.interpolate import griddata
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
import cv2
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import os
from PIL import Image
import json
from tqdm import tqdm
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from typing import Tuple, List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')


class LandfillDatasetGenerator:
    """
    High-performance dataset generator for creating large-scale landfill detection datasets.
    Generates composite images and corresponding polygon masks for neural network training.
    """

    def __init__(self, grid_size: Tuple[int, int] = (256, 256),
                 output_dir: str = "DataSet",
                 num_workers: int = None):
        """
        Initialize the dataset generator.

        Args:
            grid_size: (height, width) of generated images
            output_dir: Base directory for dataset storage
            num_workers: Number of parallel workers (default: CPU count)
        """
        self.grid_size = grid_size
        self.height, self.width = grid_size
        self.output_dir = output_dir
        self.num_workers = num_workers or min(mp.cpu_count(), 8)  # Limit to prevent memory issues

        # Create output directories
        self.images_dir = os.path.join(output_dir, "images")
        self.polygons_dir = os.path.join(output_dir, "polygons")
        self.metadata_dir = os.path.join(output_dir, "metadata")

        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.polygons_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

        # Dataset configuration
        self.dataset_config = {
            'grid_size': grid_size,
            'total_samples': 0,
            'generated_samples': 0,
            'failed_samples': 0,
            'landfill_parameters': {
                'min_sites': 1,
                'max_sites': 8,
                'min_size': 0.3,
                'max_size': 2.5,
                'min_intensity': 0.4,
                'max_intensity': 1.0
            },
            'detection_parameters': {
                'ndvi_threshold': 0.3,
                'temp_threshold_percentile': 75,
                'methane_threshold': 0.4,
                'min_area_pixels': 25
            }
        }

    def generate_single_sample(self, sample_id: int,
                               bounds: Tuple[float, float, float, float] = None) -> Dict:
        """
        Generate a single composite image and its corresponding polygon mask.

        Args:
            sample_id: Unique identifier for this sample
            bounds: Geographic bounds for the sample area

        Returns:
            Dictionary with generation results and metadata
        """

        try:
            # Random geographic bounds for diversity
            if bounds is None:
                center_x = np.random.uniform(-50, 50)  # Larger geographic diversity
                center_y = np.random.uniform(-50, 50)
                size = np.random.uniform(5, 15)  # 5-15 km area
                bounds = (center_x - size / 2, center_y - size / 2,
                          center_x + size / 2, center_y + size / 2)

            # Create coordinate grids
            x = np.linspace(bounds[0], bounds[2], self.width)
            y = np.linspace(bounds[1], bounds[3], self.height)
            X, Y = np.meshgrid(x, y)

            # Generate base terrain
            terrain = self._generate_base_terrain(X, Y)

            # Generate random landfill sites
            num_sites = np.random.randint(
                self.dataset_config['landfill_parameters']['min_sites'],
                self.dataset_config['landfill_parameters']['max_sites'] + 1
            )

            landfill_locations = []
            landfill_characteristics = []

            for _ in range(num_sites):
                # Random location within bounds with some margin
                margin = (bounds[2] - bounds[0]) * 0.1
                center_x = np.random.uniform(bounds[0] + margin, bounds[2] - margin)
                center_y = np.random.uniform(bounds[1] + margin, bounds[3] - margin)

                # Random characteristics
                size = np.random.uniform(
                    self.dataset_config['landfill_parameters']['min_size'],
                    self.dataset_config['landfill_parameters']['max_size']
                )
                intensity = np.random.uniform(
                    self.dataset_config['landfill_parameters']['min_intensity'],
                    self.dataset_config['landfill_parameters']['max_intensity']
                )
                age = np.random.uniform(0.2, 1.0)

                landfill_locations.append((center_x, center_y))
                landfill_characteristics.append({
                    'size': size,
                    'intensity': intensity,
                    'age': age,
                    'temperature_anomaly': 3 + np.random.uniform(0, 6),
                    'methane_level': 0.6 + np.random.uniform(0, 0.4),
                    'vegetation_suppression': 0.7 + np.random.uniform(0, 0.3)
                })

            # Generate layers
            ndvi_layer = self._simulate_ndvi(X, Y, terrain, landfill_locations, landfill_characteristics)
            temp_layer = self._simulate_temperature(X, Y, terrain, landfill_locations, landfill_characteristics)
            methane_layer = self._simulate_methane(X, Y, landfill_locations, landfill_characteristics)

            # Create composite image
            composite = self._create_composite(ndvi_layer, temp_layer, methane_layer)

            # Detect landfill polygons
            polygons = self._detect_polygons(ndvi_layer, temp_layer, methane_layer, bounds)

            # Create polygon mask
            polygon_mask = self._create_polygon_mask(polygons, bounds)

            # Save composite image
            composite_uint8 = (composite * 255).astype(np.uint8)
            composite_image = Image.fromarray(composite_uint8)
            image_path = os.path.join(self.images_dir, f"{sample_id:07d}.png")
            composite_image.save(image_path)

            # Save polygon mask
            mask_image = Image.fromarray((polygon_mask * 255).astype(np.uint8), mode='L')
            mask_path = os.path.join(self.polygons_dir, f"{sample_id:07d}.png")
            mask_image.save(mask_path)

            # Create metadata
            metadata = {
                'sample_id': sample_id,
                'bounds': bounds,
                'num_landfills': num_sites,
                'num_detected_polygons': len(polygons),
                'landfill_locations': landfill_locations,
                'landfill_characteristics': landfill_characteristics,
                'has_landfills': len(polygons) > 0,
                'image_path': image_path,
                'mask_path': mask_path,
                'ndvi_stats': {
                    'min': float(ndvi_layer.min()),
                    'max': float(ndvi_layer.max()),
                    'mean': float(ndvi_layer.mean())
                },
                'temperature_stats': {
                    'min': float(temp_layer.min()),
                    'max': float(temp_layer.max()),
                    'mean': float(temp_layer.mean())
                },
                'methane_stats': {
                    'min': float(methane_layer.min()),
                    'max': float(methane_layer.max()),
                    'mean': float(methane_layer.mean())
                }
            }

            return {'success': True, 'metadata': metadata, 'sample_id': sample_id}

        except Exception as e:
            return {'success': False, 'error': str(e), 'sample_id': sample_id}

    def _generate_base_terrain(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Generate base terrain using multiple octaves of noise."""
        terrain = np.zeros(self.grid_size)

        for octave in range(4):
            freq = 2 ** octave * 0.1
            amplitude = 1.0 / (2 ** octave)

            # Use coordinate-based noise generation
            noise = amplitude * (np.sin(X * freq * np.pi) * np.cos(Y * freq * np.pi) +
                                 np.sin(X * freq * 1.3 * np.pi) * np.cos(Y * freq * 0.7 * np.pi))
            terrain += noise

        return ndimage.gaussian_filter(terrain, sigma=1.5)

    def _simulate_ndvi(self, X: np.ndarray, Y: np.ndarray, terrain: np.ndarray,
                       landfill_locations: List, landfill_characteristics: List) -> np.ndarray:
        """Simulate NDVI layer with landfill impacts."""

        # Base vegetation influenced by terrain
        base_vegetation = 0.5 + np.random.uniform(-0.2, 0.3)
        ndvi = base_vegetation + 0.15 * terrain

        # Add vegetation clusters
        num_clusters = np.random.randint(2, 6)
        for _ in range(num_clusters):
            cluster_x = np.random.uniform(X.min(), X.max())
            cluster_y = np.random.uniform(Y.min(), Y.max())
            cluster_size = np.random.uniform(1.0, 3.0)
            cluster_strength = np.random.uniform(0.1, 0.4)

            distance = np.sqrt((X - cluster_x) ** 2 + (Y - cluster_y) ** 2)
            vegetation_boost = cluster_strength * np.exp(-distance ** 2 / (2 * cluster_size ** 2))
            ndvi += vegetation_boost

        # Apply landfill impacts
        for i, (lf_x, lf_y) in enumerate(landfill_locations):
            char = landfill_characteristics[i]
            distance = np.sqrt((X - lf_x) ** 2 + (Y - lf_y) ** 2)
            suppression_radius = char['size'] * 1.5
            suppression = char['vegetation_suppression'] * np.exp(-distance ** 2 / (2 * suppression_radius ** 2))
            ndvi -= suppression * 0.8

        # Add noise and smooth
        noise = np.random.normal(0, 0.04, self.grid_size)
        ndvi += noise
        ndvi = ndimage.gaussian_filter(ndvi, sigma=0.8)

        return np.clip(ndvi, -1, 1)

    def _simulate_temperature(self, X: np.ndarray, Y: np.ndarray, terrain: np.ndarray,
                              landfill_locations: List, landfill_characteristics: List) -> np.ndarray:
        """Simulate surface temperature layer."""

        # Base temperature with seasonal and daily variation
        base_temp = np.random.uniform(10, 30)  # Seasonal variation
        daily_variation = np.random.uniform(-3, 3)  # Time of day effect

        temperature = base_temp + daily_variation + 1.5 * terrain

        # Add elevation and solar exposure effects
        elevation_effect = -0.3 * terrain
        solar_gradient = np.random.uniform(-1, 1) * (X - np.mean(X)) / (np.max(X) - np.min(X))
        temperature += elevation_effect + solar_gradient

        # Apply landfill heat signatures
        for i, (lf_x, lf_y) in enumerate(landfill_locations):
            char = landfill_characteristics[i]
            distance = np.sqrt((X - lf_x) ** 2 + (Y - lf_y) ** 2)
            heat_radius = char['size'] * 1.3
            heat_anomaly = char['temperature_anomaly'] * np.exp(-distance ** 2 / (2 * heat_radius ** 2))
            temperature += heat_anomaly * char['intensity']

        # Add noise and smooth
        noise = np.random.normal(0, 0.8, self.grid_size)
        temperature += noise
        temperature = ndimage.gaussian_filter(temperature, sigma=0.6)

        return temperature

    def _simulate_methane(self, X: np.ndarray, Y: np.ndarray,
                          landfill_locations: List, landfill_characteristics: List) -> np.ndarray:
        """Simulate methane concentration layer with lower resolution."""

        # Start with low resolution simulation
        lr_factor = 4
        methane_size = (self.height // lr_factor, self.width // lr_factor)

        # Create low-resolution grids
        x_lr = np.linspace(X.min(), X.max(), methane_size[1])
        y_lr = np.linspace(Y.min(), Y.max(), methane_size[0])
        X_lr, Y_lr = np.meshgrid(x_lr, y_lr)

        # Base methane level
        base_methane = np.random.uniform(0.05, 0.15)
        methane_lr = np.full(methane_size, base_methane)

        # Add landfill emissions
        for i, (lf_x, lf_y) in enumerate(landfill_locations):
            char = landfill_characteristics[i]
            distance_lr = np.sqrt((X_lr - lf_x) ** 2 + (Y_lr - lf_y) ** 2)
            emission_radius = char['size'] * 2.5
            methane_emission = char['methane_level'] * np.exp(-distance_lr ** 2 / (2 * emission_radius ** 2))
            methane_lr += methane_emission * char['intensity'] * char['age']

        # Add wind dispersion
        if landfill_locations:
            wind_direction = np.random.uniform(0, 2 * np.pi)
            wind_strength = np.random.uniform(0.1, 0.4)

            for i, (lf_x, lf_y) in enumerate(landfill_locations):
                char = landfill_characteristics[i]
                dx = X_lr - lf_x
                dy = Y_lr - lf_y

                # Wind-rotated coordinates
                rotated_x = dx * np.cos(wind_direction) + dy * np.sin(wind_direction)
                rotated_y = -dx * np.sin(wind_direction) + dy * np.cos(wind_direction)

                # Asymmetric plume
                plume = char['methane_level'] * 0.4 * np.exp(
                    -(rotated_x ** 2 / (2 * (char['size'] * 2.0) ** 2) +
                      rotated_y ** 2 / (2 * (char['size'] * 1.0) ** 2))
                )

                methane_lr += plume * char['intensity'] * wind_strength

        # Add noise and smooth
        noise_lr = np.random.normal(0, 0.02, methane_size)
        methane_lr += noise_lr
        methane_lr = ndimage.gaussian_filter(methane_lr, sigma=0.4)

        # Upsample to full resolution
        methane_upsampled = cv2.resize(methane_lr, (self.width, self.height),
                                       interpolation=cv2.INTER_CUBIC)

        return np.clip(methane_upsampled, 0, 1)

    def _create_composite(self, ndvi: np.ndarray, temperature: np.ndarray,
                          methane: np.ndarray) -> np.ndarray:
        """Create composite RGB-like image from the three indices."""

        # Normalize each layer
        scaler = MinMaxScaler()

        # Invert NDVI (low vegetation = bright for landfill detection)
        ndvi_norm = 1 - ((ndvi + 1) / 2)

        # Normalize temperature
        temp_norm = scaler.fit_transform(temperature.reshape(-1, 1)).reshape(self.grid_size)

        # Methane is already normalized
        methane_norm = methane

        # Stack as RGB
        composite = np.stack([ndvi_norm, temp_norm, methane_norm], axis=-1)

        return composite

    def _detect_polygons(self, ndvi: np.ndarray, temperature: np.ndarray,
                         methane: np.ndarray, bounds: Tuple) -> List[Polygon]:
        """Detect landfill polygons based on the three criteria."""

        # Create detection masks
        ndvi_threshold = self.dataset_config['detection_parameters']['ndvi_threshold']
        temp_percentile = self.dataset_config['detection_parameters']['temp_threshold_percentile']
        methane_threshold = self.dataset_config['detection_parameters']['methane_threshold']
        min_area = self.dataset_config['detection_parameters']['min_area_pixels']

        low_ndvi_mask = ndvi < ndvi_threshold
        high_temp_mask = temperature > np.percentile(temperature, temp_percentile)
        high_methane_mask = methane > methane_threshold

        # Combine criteria
        landfill_mask = low_ndvi_mask & high_temp_mask & high_methane_mask

        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        landfill_mask = cv2.morphologyEx(landfill_mask.astype(np.uint8),
                                         cv2.MORPH_CLOSE, kernel)
        landfill_mask = cv2.morphologyEx(landfill_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(landfill_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        polygons = []
        for contour in contours:
            if cv2.contourArea(contour) >= min_area:
                # Convert to geographic coordinates
                geo_coords = []
                for point in contour.reshape(-1, 2):
                    geo_x = bounds[0] + (point[0] / self.width) * (bounds[2] - bounds[0])
                    geo_y = bounds[1] + (point[1] / self.height) * (bounds[3] - bounds[1])
                    geo_coords.append((geo_x, geo_y))

                if len(geo_coords) >= 3:
                    polygons.append(Polygon(geo_coords))

        return polygons

    def _create_polygon_mask(self, polygons: List[Polygon],
                             bounds: Tuple) -> np.ndarray:
        """Create binary mask image from polygons."""

        mask = np.zeros(self.grid_size, dtype=np.uint8)

        for polygon in polygons:
            # Convert polygon coordinates to pixel coordinates
            pixel_coords = []
            for x, y in polygon.exterior.coords:
                pixel_x = int((x - bounds[0]) / (bounds[2] - bounds[0]) * self.width)
                pixel_y = int((y - bounds[1]) / (bounds[3] - bounds[1]) * self.height)
                pixel_coords.append([pixel_x, pixel_y])

            if len(pixel_coords) >= 3:
                # Fill polygon in mask
                cv2.fillPoly(mask, [np.array(pixel_coords, dtype=np.int32)], 1)

        return mask

    def generate_dataset_batch(self, start_id: int, batch_size: int) -> List[Dict]:
        """Generate a batch of samples with parallel processing."""

        results = []
        sample_ids = list(range(start_id, start_id + batch_size))

        # Use ThreadPoolExecutor for I/O-bound operations
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(self.generate_single_sample, sample_id)
                       for sample_id in sample_ids]

            # Collect results with progress bar
            for future in tqdm(futures, desc=f"Batch {start_id // batch_size + 1}",
                               leave=False):
                result = future.result()
                results.append(result)

        return results

    def generate_large_dataset(self, total_samples: int = 500000,
                               batch_size: int = 1000,
                               save_metadata_every: int = 10000) -> None:
        """
        Generate large-scale dataset with progress tracking and error handling.

        Args:
            total_samples: Total number of samples to generate
            batch_size: Number of samples per batch
            save_metadata_every: Save metadata every N samples
        """

        print(f"🚀 Starting generation of {total_samples:,} samples")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🖼️  Image size: {self.grid_size}")
        print(f"⚡ Using {self.num_workers} parallel workers")
        print(f"📦 Batch size: {batch_size}")
        print("=" * 60)

        # Update dataset config
        self.dataset_config['total_samples'] = total_samples

        # Track overall progress
        all_metadata = []
        successful_samples = 0
        failed_samples = 0

        # Main generation loop
        with tqdm(total=total_samples, desc="Overall Progress", unit="samples") as pbar:
            for batch_start in range(0, total_samples, batch_size):
                current_batch_size = min(batch_size, total_samples - batch_start)

                # Generate batch
                batch_results = self.generate_dataset_batch(batch_start, current_batch_size)

                # Process batch results
                batch_metadata = []
                batch_successful = 0
                batch_failed = 0

                for result in batch_results:
                    if result['success']:
                        batch_metadata.append(result['metadata'])
                        batch_successful += 1
                    else:
                        batch_failed += 1
                        print(f"⚠️  Sample {result['sample_id']} failed: {result['error']}")

                # Update counters
                successful_samples += batch_successful
                failed_samples += batch_failed
                all_metadata.extend(batch_metadata)

                # Update progress
                pbar.update(current_batch_size)
                pbar.set_postfix({
                    'Success': successful_samples,
                    'Failed': failed_samples,
                    'Success Rate': f"{successful_samples / (successful_samples + failed_samples) * 100:.1f}%"
                })

                # Periodic metadata save
                if (batch_start + current_batch_size) % save_metadata_every == 0:
                    self._save_metadata_checkpoint(all_metadata, batch_start + current_batch_size)

        # Final metadata save
        print("\n💾 Saving final metadata...")
        self._save_final_metadata(all_metadata, successful_samples, failed_samples)

        print(f"\n✅ Dataset generation completed!")
        print(f"   📊 Total samples generated: {successful_samples:,}")
        print(f"   ❌ Failed samples: {failed_samples:,}")
        print(f"   📈 Success rate: {successful_samples / (successful_samples + failed_samples) * 100:.1f}%")
        print(f"   📁 Dataset saved in: {self.output_dir}")

    def _save_metadata_checkpoint(self, metadata: List[Dict], samples_processed: int) -> None:
        """Save metadata checkpoint."""
        checkpoint_path = os.path.join(self.metadata_dir, f"checkpoint_{samples_processed:07d}.json")

        checkpoint_data = {
            'samples_processed': samples_processed,
            'timestamp': str(np.datetime64('now')),
            'metadata': metadata[-min(1000, len(metadata)):]  # Save last 1000 entries
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

    def _save_final_metadata(self, all_metadata: List[Dict],
                             successful: int, failed: int) -> None:
        """Save final comprehensive metadata."""

        # Update dataset config
        self.dataset_config['generated_samples'] = successful
        self.dataset_config['failed_samples'] = failed
        self.dataset_config['success_rate'] = successful / (successful + failed) if (successful + failed) > 0 else 0
        self.dataset_config['generation_timestamp'] = str(np.datetime64('now'))

        # Save dataset configuration
        config_path = os.path.join(self.metadata_dir, "dataset_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.dataset_config, f, indent=2)

        # Save complete metadata
        metadata_path = os.path.join(self.metadata_dir, "complete_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(all_metadata, f, indent=2)

        # Save summary statistics
        self._generate_dataset_summary(all_metadata)

        print(f"   📄 Configuration saved: {config_path}")
        print(f"   📋 Metadata saved: {metadata_path}")

    def _generate_dataset_summary(self, metadata: List[Dict]) -> None:
        """Generate and save dataset summary statistics."""

        if not metadata:
            return

        # Calculate statistics
        total_samples = len(metadata)
        samples_with_landfills = sum(1 for m in metadata if m['has_landfills'])

        # NDVI statistics
        ndvi_values = [m['ndvi_stats']['mean'] for m in metadata]
        temp_values = [m['temperature_stats']['mean'] for m in metadata]
        methane_values = [m['methane_stats']['mean'] for m in metadata]

        summary = {
            'dataset_overview': {
                'total_samples': total_samples,
                'samples_with_landfills': samples_with_landfills,
                'samples_without_landfills': total_samples - samples_with_landfills,
                'landfill_ratio': samples_with_landfills / total_samples if total_samples > 0 else 0
            },
            'ndvi_statistics': {
                'mean': float(np.mean(ndvi_values)),
                'std': float(np.std(ndvi_values)),
                'min': float(np.min(ndvi_values)),
                'max': float(np.max(ndvi_values))
            },
            'temperature_statistics': {
                'mean': float(np.mean(temp_values)),
                'std': float(np.std(temp_values)),
                'min': float(np.min(temp_values)),
                'max': float(np.max(temp_values))
            },
            'methane_statistics': {
                'mean': float(np.mean(methane_values)),
                'std': float(np.std(methane_values)),
                'min': float(np.min(methane_values)),
                'max': float(np.max(methane_values))
            }
        }

        summary_path = os.path.join(self.metadata_dir, "dataset_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"   📊 Summary statistics saved: {summary_path}")


# Convenience function for easy dataset generation
def generate_landfill_dataset(num_samples: int = 500000,
                              grid_size: Tuple[int, int] = (256, 256),
                              output_dir: str = "DataSet",
                              batch_size: int = 1000,
                              num_workers: int = None) -> None:
    """
    Convenience function to generate a large landfill detection dataset.

    Args:
        num_samples: Number of samples to generate (default: 500,000)
        grid_size: Image dimensions (default: 256x256)
        output_dir: Output directory (default: "DataSet")
        batch_size: Batch size for parallel processing
        num_workers: Number of parallel workers
    """

    generator = LandfillDatasetGenerator(
        grid_size=grid_size,
        output_dir=output_dir,
        num_workers=num_workers
    )

    generator.generate_large_dataset(
        total_samples=num_samples,
        batch_size=batch_size
    )


if __name__ == "__main__":
    # Example usage: Generate the full 500k dataset
    print("🌍 Landfill Detection Dataset Generator")
    print("=" * 50)

    # Generate smaller test dataset first (remove this for full generation)
    print("🧪 Generating test dataset (1000 samples)...")
    generate_landfill_dataset(
        num_samples=1000,  # Change to 500000 for full dataset
        grid_size=(256, 256),
        output_dir="DataSet",
        batch_size=100,
        num_workers=4
    )

    # Uncomment below for full 500k dataset generation
    # print("🚀 Generating full dataset (500k samples)...")
    # generate_landfill_dataset(
    #     num_samples=500000,
    #     grid_size=(256, 256),
    #     output_dir="DataSet",
    #     batch_size=1000,
    #     num_workers=8
    # )