import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.interpolate import griddata
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
import cv2
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import geopandas as gpd
from rasterio.transform import from_bounds
import rasterio
from typing import Tuple, List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')


class EnvironmentalIndexModeler:
    """
    A comprehensive system for modeling and simulating environmental indices
    for landfill detection using NDVI, Surface Temperature, and Methane levels.
    """

    def __init__(self, grid_size: Tuple[int, int] = (512, 512),
                 bounds: Tuple[float, float, float, float] = (0, 0, 10, 10)):
        """
        Initialize the environmental index modeler.

        Args:
            grid_size: (height, width) of the simulation grid
            bounds: (min_x, min_y, max_x, max_y) geographical bounds
        """
        self.grid_size = grid_size
        self.bounds = bounds
        self.height, self.width = grid_size

        # Create coordinate grids
        self.x = np.linspace(bounds[0], bounds[2], self.width)
        self.y = np.linspace(bounds[1], bounds[3], self.height)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Initialize data layers
        self.ndvi_layer = None
        self.temperature_layer = None
        self.methane_layer = None

        # Landfill parameters
        self.landfill_locations = []
        self.landfill_characteristics = []

    def generate_base_terrain(self, noise_scale: float = 0.1,
                              smoothing_factor: float = 2.0) -> None:
        """Generate base terrain characteristics using Perlin-like noise."""

        # Create multiple octaves of noise for realistic terrain
        terrain_base = np.zeros(self.grid_size)

        for octave in range(4):
            freq = 2 ** octave
            amplitude = 1.0 / (2 ** octave)

            # Generate noise at different frequencies
            noise_x = np.random.random(self.grid_size) * freq
            noise_y = np.random.random(self.grid_size) * freq

            terrain_base += amplitude * np.sin(noise_x * np.pi) * np.cos(noise_y * np.pi)

        # Smooth the terrain
        self.base_terrain = ndimage.gaussian_filter(terrain_base, smoothing_factor)

    def add_landfill_sites(self, num_sites: int = 5,
                           min_size: float = 0.5, max_size: float = 2.0) -> None:
        """Add simulated landfill sites with varying characteristics."""

        self.landfill_locations = []
        self.landfill_characteristics = []

        for i in range(num_sites):
            # Random location within bounds
            center_x = np.random.uniform(self.bounds[0] + 1, self.bounds[2] - 1)
            center_y = np.random.uniform(self.bounds[1] + 1, self.bounds[3] - 1)

            # Random size and characteristics
            size = np.random.uniform(min_size, max_size)
            intensity = np.random.uniform(0.6, 1.0)  # Environmental impact intensity
            age = np.random.uniform(0.2, 1.0)  # Age factor (affects characteristics)

            self.landfill_locations.append((center_x, center_y))
            self.landfill_characteristics.append({
                'size': size,
                'intensity': intensity,
                'age': age,
                'temperature_anomaly': 3 + np.random.uniform(0, 5),  # 3-8 degrees above ambient
                'methane_level': 0.7 + np.random.uniform(0, 0.3),  # High methane
                'vegetation_suppression': 0.8 + np.random.uniform(0, 0.2)  # Low NDVI
            })

    def simulate_ndvi_layer(self, base_vegetation: float = 0.6) -> np.ndarray:
        """
        Simulate NDVI layer with realistic vegetation patterns and landfill impacts.
        NDVI values range from -1 to 1, with healthy vegetation typically 0.3-0.8
        """

        # Start with base vegetation influenced by terrain
        ndvi = base_vegetation + 0.2 * self.base_terrain
        ndvi = np.clip(ndvi, -1, 1)

        # Add natural vegetation patterns
        for i in range(3):
            # Create vegetation clusters
            cluster_x = np.random.uniform(self.bounds[0], self.bounds[2])
            cluster_y = np.random.uniform(self.bounds[1], self.bounds[3])
            cluster_size = np.random.uniform(1.0, 3.0)

            distance = np.sqrt((self.X - cluster_x) ** 2 + (self.Y - cluster_y) ** 2)
            vegetation_boost = 0.3 * np.exp(-distance ** 2 / (2 * cluster_size ** 2))
            ndvi += vegetation_boost

        # Apply landfill impacts (reduced vegetation)
        for i, (lf_x, lf_y) in enumerate(self.landfill_locations):
            char = self.landfill_characteristics[i]

            # Distance from landfill center
            distance = np.sqrt((self.X - lf_x) ** 2 + (self.Y - lf_y) ** 2)

            # Vegetation suppression around landfill
            suppression_radius = char['size'] * 1.5
            suppression = char['vegetation_suppression'] * np.exp(-distance ** 2 / (2 * suppression_radius ** 2))

            # Reduce NDVI in landfill areas
            ndvi -= suppression * 0.7

        # Add realistic noise
        noise = np.random.normal(0, 0.05, self.grid_size)
        ndvi += noise

        # Apply smoothing to make it more realistic
        ndvi = ndimage.gaussian_filter(ndvi, sigma=1.0)

        self.ndvi_layer = np.clip(ndvi, -1, 1)
        return self.ndvi_layer

    def simulate_temperature_layer(self, base_temp: float = 15.0) -> np.ndarray:
        """
        Simulate surface temperature layer with landfill heat anomalies.
        Temperature in Celsius.
        """

        # Start with base temperature influenced by terrain and time of day variation
        temperature = base_temp + 2 * self.base_terrain

        # Add natural temperature variations (e.g., elevation effects, solar exposure)
        elevation_effect = -0.5 * self.base_terrain  # Higher elevations are cooler
        temperature += elevation_effect

        # Add solar exposure patterns
        solar_x_gradient = 2 * (self.X - np.mean(self.X)) / (np.max(self.X) - np.min(self.X))
        solar_y_gradient = 1 * (self.Y - np.mean(self.Y)) / (np.max(self.Y) - np.min(self.Y))
        temperature += solar_x_gradient + solar_y_gradient

        # Apply landfill heat signatures
        for i, (lf_x, lf_y) in enumerate(self.landfill_locations):
            char = self.landfill_characteristics[i]

            # Distance from landfill center
            distance = np.sqrt((self.X - lf_x) ** 2 + (self.Y - lf_y) ** 2)

            # Temperature anomaly (landfills are warmer due to decomposition)
            heat_radius = char['size'] * 1.2
            heat_anomaly = char['temperature_anomaly'] * np.exp(-distance ** 2 / (2 * heat_radius ** 2))

            temperature += heat_anomaly * char['intensity']

        # Add realistic temperature noise
        noise = np.random.normal(0, 0.5, self.grid_size)
        temperature += noise

        # Smooth for realism
        temperature = ndimage.gaussian_filter(temperature, sigma=0.8)

        self.temperature_layer = temperature
        return self.temperature_layer

    def simulate_methane_layer(self, base_methane: float = 0.1) -> np.ndarray:
        """
        Simulate methane concentration layer (lower resolution: 250m vs 20m).
        Values normalized between 0-1 where 1 represents high methane concentration.
        """

        # Create lower resolution grid for methane (250m resolution simulation)
        methane_size = (self.height // 4, self.width // 4)  # Simulate lower resolution

        # Start with low base methane levels
        methane_lr = np.full(methane_size, base_methane)

        # Create coordinate grids for low resolution
        x_lr = np.linspace(self.bounds[0], self.bounds[2], methane_size[1])
        y_lr = np.linspace(self.bounds[1], self.bounds[3], methane_size[0])
        X_lr, Y_lr = np.meshgrid(x_lr, y_lr)

        # Add landfill methane emissions
        for i, (lf_x, lf_y) in enumerate(self.landfill_locations):
            char = self.landfill_characteristics[i]

            # Distance from landfill center in low resolution
            distance_lr = np.sqrt((X_lr - lf_x) ** 2 + (Y_lr - lf_y) ** 2)

            # Methane emission pattern (broader than temperature due to gas dispersion)
            emission_radius = char['size'] * 2.0
            methane_emission = char['methane_level'] * np.exp(-distance_lr ** 2 / (2 * emission_radius ** 2))

            methane_lr += methane_emission * char['intensity'] * char['age']

        # Add wind dispersion effects (asymmetric plume)
        wind_direction = np.random.uniform(0, 2 * np.pi)
        wind_strength = np.random.uniform(0.1, 0.3)

        for i, (lf_x, lf_y) in enumerate(self.landfill_locations):
            char = self.landfill_characteristics[i]

            # Wind-dispersed methane plume
            dx = X_lr - lf_x
            dy = Y_lr - lf_y

            # Rotate coordinates based on wind direction
            rotated_x = dx * np.cos(wind_direction) + dy * np.sin(wind_direction)
            rotated_y = -dx * np.sin(wind_direction) + dy * np.cos(wind_direction)

            # Asymmetric Gaussian plume
            plume = char['methane_level'] * 0.3 * np.exp(
                -(rotated_x ** 2 / (2 * (char['size'] * 1.5) ** 2) +
                  rotated_y ** 2 / (2 * (char['size'] * 0.8) ** 2))
            )

            # Apply wind displacement
            plume_shifted = np.roll(plume, int(wind_strength * 10), axis=1)
            methane_lr += plume_shifted * char['intensity']

        # Add noise and smooth
        noise_lr = np.random.normal(0, 0.02, methane_size)
        methane_lr += noise_lr
        methane_lr = ndimage.gaussian_filter(methane_lr, sigma=0.5)

        # Upsample to match other layers (simulate interpolation from lower resolution)
        methane_upsampled = cv2.resize(methane_lr, (self.width, self.height),
                                       interpolation=cv2.INTER_CUBIC)

        self.methane_layer = np.clip(methane_upsampled, 0, 1)
        return self.methane_layer

    def create_composite_image(self) -> np.ndarray:
        """
        Create a composite image using the three indices as RGB-like channels.
        """

        if any(layer is None for layer in [self.ndvi_layer, self.temperature_layer, self.methane_layer]):
            raise ValueError("All layers must be simulated before creating composite image")

        # Normalize layers to 0-1 range for visualization
        scaler = MinMaxScaler()

        # Invert NDVI so low vegetation appears bright (landfill indicator)
        ndvi_inverted = 1 - ((self.ndvi_layer + 1) / 2)  # Convert from [-1,1] to [0,1] then invert

        # Normalize temperature (higher temps brighter)
        temp_norm = scaler.fit_transform(self.temperature_layer.reshape(-1, 1)).reshape(self.grid_size)

        # Methane is already 0-1
        methane_norm = self.methane_layer

        # Stack as RGB-like image
        composite = np.stack([ndvi_inverted, temp_norm, methane_norm], axis=-1)

        return composite

    def detect_landfill_candidates(self, ndvi_threshold: float = 0.3,
                                   temp_threshold_percentile: float = 75,
                                   methane_threshold: float = 0.4,
                                   min_area_pixels: int = 50) -> List[Polygon]:
        """
        Detect potential landfill areas based on the three criteria.

        Args:
            ndvi_threshold: Maximum NDVI value for landfill areas
            temp_threshold_percentile: Temperature percentile for anomaly detection
            methane_threshold: Minimum methane concentration
            min_area_pixels: Minimum area in pixels for valid detection

        Returns:
            List of Shapely Polygon objects representing detected areas
        """

        # Create binary masks for each criterion
        low_ndvi_mask = self.ndvi_layer < ndvi_threshold

        # Temperature anomaly: areas significantly warmer than surroundings
        temp_threshold = np.percentile(self.temperature_layer, temp_threshold_percentile)
        high_temp_mask = self.temperature_layer > temp_threshold

        high_methane_mask = self.methane_layer > methane_threshold

        # Combine all criteria
        landfill_mask = low_ndvi_mask & high_temp_mask & high_methane_mask

        # Clean up the mask (remove noise, fill holes)
        kernel = np.ones((3, 3), np.uint8)
        landfill_mask = cv2.morphologyEx(landfill_mask.astype(np.uint8),
                                         cv2.MORPH_CLOSE, kernel)
        landfill_mask = cv2.morphologyEx(landfill_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(landfill_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Convert contours to polygons with geographic coordinates
        polygons = []

        for contour in contours:
            if cv2.contourArea(contour) >= min_area_pixels:
                # Convert pixel coordinates to geographic coordinates
                geo_coords = []
                for point in contour.reshape(-1, 2):
                    geo_x = self.bounds[0] + (point[0] / self.width) * (self.bounds[2] - self.bounds[0])
                    geo_y = self.bounds[1] + (point[1] / self.height) * (self.bounds[3] - self.bounds[1])
                    geo_coords.append((geo_x, geo_y))

                if len(geo_coords) >= 3:  # Valid polygon needs at least 3 points
                    polygons.append(Polygon(geo_coords))

        return polygons

    def run_full_simulation(self) -> Dict:
        """
        Run the complete simulation pipeline.

        Returns:
            Dictionary containing all simulation results
        """

        print("Generating base terrain...")
        self.generate_base_terrain()

        print("Adding landfill sites...")
        self.add_landfill_sites()

        print("Simulating NDVI layer...")
        self.simulate_ndvi_layer()

        print("Simulating temperature layer...")
        self.simulate_temperature_layer()

        print("Simulating methane layer...")
        self.simulate_methane_layer()

        print("Creating composite image...")
        composite = self.create_composite_image()

        print("Detecting landfill candidates...")
        detected_polygons = self.detect_landfill_candidates()

        results = {
            'ndvi_layer': self.ndvi_layer,
            'temperature_layer': self.temperature_layer,
            'methane_layer': self.methane_layer,
            'composite_image': composite,
            'detected_polygons': detected_polygons,
            'true_landfill_locations': self.landfill_locations,
            'landfill_characteristics': self.landfill_characteristics,
            'coordinate_bounds': self.bounds
        }

        return results

    def visualize_results(self, results: Dict) -> None:
        """
        Create comprehensive visualization of simulation results.
        """

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # NDVI layer
        im1 = axes[0, 0].imshow(results['ndvi_layer'], cmap='RdYlGn', vmin=-1, vmax=1)
        axes[0, 0].set_title('NDVI Layer\n(Green=Vegetation, Red=No Vegetation)')
        plt.colorbar(im1, ax=axes[0, 0])

        # Temperature layer
        im2 = axes[0, 1].imshow(results['temperature_layer'], cmap='hot')
        axes[0, 1].set_title('Surface Temperature Layer\n(°C)')
        plt.colorbar(im2, ax=axes[0, 1])

        # Methane layer
        im3 = axes[0, 2].imshow(results['methane_layer'], cmap='plasma', vmin=0, vmax=1)
        axes[0, 2].set_title('Methane Concentration Layer\n(Normalized 0-1)')
        plt.colorbar(im3, ax=axes[0, 2])

        # Composite image
        axes[1, 0].imshow(results['composite_image'])
        axes[1, 0].set_title('Composite Image\n(R: Low NDVI, G: High Temp, B: High Methane)')

        # Detection results
        axes[1, 1].imshow(results['composite_image'])

        # Plot detected polygons
        for poly in results['detected_polygons']:
            x_coords = [coord[0] for coord in poly.exterior.coords]
            y_coords = [coord[1] for coord in poly.exterior.coords]

            # Convert geographic coordinates back to pixel coordinates for plotting
            x_pixels = [(x - self.bounds[0]) / (self.bounds[2] - self.bounds[0]) * self.width for x in x_coords]
            y_pixels = [(y - self.bounds[1]) / (self.bounds[3] - self.bounds[1]) * self.height for y in y_coords]

            axes[1, 1].plot(x_pixels, y_pixels, 'r-', linewidth=2)
            axes[1, 1].fill(x_pixels, y_pixels, 'red', alpha=0.3)

        # Plot true landfill locations
        for lf_x, lf_y in results['true_landfill_locations']:
            x_pixel = (lf_x - self.bounds[0]) / (self.bounds[2] - self.bounds[0]) * self.width
            y_pixel = (lf_y - self.bounds[1]) / (self.bounds[3] - self.bounds[1]) * self.height
            axes[1, 1].plot(x_pixel, y_pixel, 'wo', markersize=8, markeredgecolor='black')

        axes[1, 1].set_title('Detection Results\n(Red: Detected, White: True Locations)')

        # Statistics
        stats_text = f"""
        Simulation Statistics:

        True Landfills: {len(results['true_landfill_locations'])}
        Detected Areas: {len(results['detected_polygons'])}

        NDVI Range: [{results['ndvi_layer'].min():.2f}, {results['ndvi_layer'].max():.2f}]
        Temp Range: [{results['temperature_layer'].min():.1f}°C, {results['temperature_layer'].max():.1f}°C]
        Methane Range: [{results['methane_layer'].min():.3f}, {results['methane_layer'].max():.3f}]

        Grid Size: {self.grid_size}
        Geographic Bounds: {self.bounds}
        """

        axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes,
                        fontsize=10, verticalalignment='center')
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.show()


# Example usage and demonstration
def demonstrate_landfill_detection():
    """
    Demonstrate the complete landfill detection modeling system.
    """

    print("🌍 Initializing Environmental Index Modeler for Landfill Detection")
    print("=" * 60)

    # Initialize the modeler
    modeler = EnvironmentalIndexModeler(
        grid_size=(400, 400),  # 400x400 pixel grid
        bounds=(-5, -5, 5, 5)  # 10km x 10km area
    )

    # Run full simulation
    results = modeler.run_full_simulation()

    # Display results
    print(f"\n📊 Simulation Results:")
    print(f"   • Generated {len(results['true_landfill_locations'])} synthetic landfill sites")
    print(f"   • Detected {len(results['detected_polygons'])} potential landfill areas")
    print(f"   • NDVI range: {results['ndvi_layer'].min():.2f} to {results['ndvi_layer'].max():.2f}")
    print(
        f"   • Temperature range: {results['temperature_layer'].min():.1f}°C to {results['temperature_layer'].max():.1f}°C")
    print(f"   • Methane range: {results['methane_layer'].min():.3f} to {results['methane_layer'].max():.3f}")

    # Visualize results
    modeler.visualize_results(results)

    return modeler, results


if __name__ == "__main__":
    # Run the demonstration
    modeler, results = demonstrate_landfill_detection()