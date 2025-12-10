"""
EST_Statistics.py - Statistical Analysis Tool for EST Simulations
=================================================================
Comprehensive statistical analysis of EST simulations including:
- Fractal dimension analysis
- Correlation functions
- Void and filament statistics
- Morphological analysis
- Comparison with observational data
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import json
import csv
import os
import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import scipy modules
try:
    from scipy import stats, ndimage, optimize, spatial
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ Scipy not available. Some statistics will be limited.")
    # Create dummy functions
    class DummyModule:
        def __getattr__(self, name):
            return None
    stats = ndimage = optimize = spatial = DummyModule()

# Set dark theme
plt.style.use('dark_background')

class EST_Statistics:
    """
    Comprehensive statistical analysis of EST universe structures
    """
    
    def __init__(self):
        self.universe_data = None
        self.config = None
        self.output_dir = None
        self.statistics = {}
        
    def load_data(self, data_path):
        """Load universe data from various formats (including older versions)"""
        data_path = Path(data_path)
        
        print(f"\n🔍 Loading data from: {data_path}")
        
        if data_path.is_file():
            if data_path.suffix == '.npz':
                return self.load_npz_data(data_path)
            elif data_path.suffix == '.npy':
                return self.load_npy_data(data_path)
            else:
                print(f"❌ Unsupported file type: {data_path.suffix}")
                return False
        else:
            # It's a folder - look for data files
            return self.load_folder_data(data_path)
    
    def load_npz_data(self, npz_path):
        """Load data from .npz file (compatible with v2.7, v3.0, and older)"""
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                # Try different possible keys (compatibility with older versions)
                possible_keys = ['final_state', 'u', 'universe', 'grid', 'state', 'data']
                
                for key in possible_keys:
                    if key in data:
                        self.universe_data = data[key]
                        print(f"  Found data with key: '{key}'")
                        break
                
                # If no standard key found, look for 3D arrays
                if self.universe_data is None:
                    for arr_name in data.files:
                        arr = data[arr_name]
                        if hasattr(arr, 'ndim') and arr.ndim == 3:
                            self.universe_data = arr
                            print(f"  Found 3D array: '{arr_name}'")
                            break
                
                if self.universe_data is None:
                    print("❌ Could not find 3D universe data in .npz file")
                    # Try to load the first array anyway
                    if len(data.files) > 0:
                        first_key = data.files[0]
                        self.universe_data = data[first_key]
                        print(f"  Using first array: '{first_key}'")
            
            # Try to load config from various sources
            self.config = self.extract_config(npz_path)
            
            # Create output directory
            self.output_dir = npz_path.parent / "Statistical_Analysis"
            self.output_dir.mkdir(exist_ok=True)
            
            print(f"✅ Loaded 3D universe data: {self.universe_data.shape}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading .npz file: {e}")
            return False
    
    def extract_config(self, npz_path):
        """Extract configuration from various sources"""
        config = {}
        
        try:
            # Try to load from .npz file
            with np.load(npz_path, allow_pickle=True) as data:
                # Try config_str (v3.0 format)
                if 'config_str' in data:
                    config_str = str(data['config_str'][0])
                    config = json.loads(config_str)
                    print(f"  Loaded config from config_str")
                
                # Try parameters (v2.7 format)
                elif 'parameters' in data:
                    params = data['parameters'].item() if hasattr(data['parameters'], 'item') else data['parameters']
                    config = dict(params)
                    print(f"  Loaded config from parameters")
                
                # Try metadata
                elif 'metadata' in data:
                    meta = data['metadata'].item() if hasattr(data['metadata'], 'item') else data['metadata']
                    if isinstance(meta, dict):
                        config.update(meta)
                        print(f"  Loaded config from metadata")
        
        except:
            pass
        
        # Try to load from JSON file in same directory
        json_files = list(npz_path.parent.glob("*.json"))
        for json_file in json_files:
            if 'param' in json_file.name.lower() or 'config' in json_file.name.lower():
                try:
                    with open(json_file, 'r') as f:
                        json_config = json.load(f)
                        config.update(json_config)
                    print(f"  Loaded config from {json_file.name}")
                    break
                except:
                    continue
        
        return config if config else None
    
    def load_npy_data(self, npy_path):
        """Load data from .npy file"""
        try:
            self.universe_data = np.load(npy_path)
            
            if self.universe_data.ndim != 3:
                print(f"❌ Expected 3D array, got {self.universe_data.ndim}D")
                return False
            
            # Try to find config in same directory
            self.config = self.find_config_in_directory(npy_path.parent)
            
            # Create output directory
            self.output_dir = npy_path.parent / "Statistical_Analysis"
            self.output_dir.mkdir(exist_ok=True)
            
            print(f"✅ Loaded 3D universe data: {self.universe_data.shape}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading .npy file: {e}")
            return False
    
    def find_config_in_directory(self, directory):
        """Find configuration files in directory"""
        config = {}
        
        # Look for JSON files
        json_files = list(directory.glob("*.json"))
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        config.update(data)
                        print(f"  Found config in {json_file.name}")
            except:
                continue
        
        # Look for parameter files
        param_files = list(directory.glob("*param*.txt"))
        param_files.extend(list(directory.glob("*config*.txt")))
        
        for param_file in param_files:
            try:
                with open(param_file, 'r') as f:
                    for line in f:
                        if '=' in line:
                            key, value = line.strip().split('=', 1)
                            config[key.strip()] = value.strip()
                print(f"  Found parameters in {param_file.name}")
            except:
                continue
        
        return config if config else None
    
    def load_folder_data(self, folder_path):
        """Load data from folder (look for various file types)"""
        folder = Path(folder_path)
        
        print(f"  Scanning folder: {folder.name}")
        
        # Look for data files in order of preference
        file_patterns = [
            "*.npz",
            "final_state*.npy",
            "universe*.npy",
            "data*.npy",
            "*.npy"  # Any .npy file
        ]
        
        for pattern in file_patterns:
            files = list(folder.glob(pattern))
            if files:
                # Sort by file size (largest first, likely contains universe data)
                files.sort(key=lambda f: f.stat().st_size, reverse=True)
                
                for file in files:
                    print(f"  Trying: {file.name}")
                    if pattern == "*.npz":
                        if self.load_npz_data(file):
                            return True
                    else:
                        if self.load_npy_data(file):
                            return True
        
        print(f"❌ No suitable data files found in {folder.name}")
        return False
    
    def calculate_basic_statistics(self):
        """Calculate basic statistical properties"""
        print("\n📊 Calculating basic statistics...")
        
        data = self.universe_data
        active_cells = np.sum(data > 0)
        total_cells = data.size
        
        stats_dict = {
            'grid_size': data.shape[0],
            'active_cells': int(active_cells),
            'total_cells': int(total_cells),
            'density': float(active_cells / total_cells),
            'density_std': float(np.std(data)),
            'active_fraction': float(active_cells / total_cells),
            'void_fraction': float(np.sum(data == 0) / total_cells),
            'data_type': str(data.dtype),
            'data_range': [float(data.min()), float(data.max())]
        }
        
        # Cluster statistics using connected components (if scipy available)
        if SCIPY_AVAILABLE:
            try:
                labeled_array, num_features = ndimage.label(data > 0)
                stats_dict['num_clusters'] = int(num_features)
                
                if num_features > 0:
                    cluster_sizes = ndimage.sum(data > 0, labeled_array, range(1, num_features + 1))
                    stats_dict['mean_cluster_size'] = float(np.mean(cluster_sizes))
                    stats_dict['max_cluster_size'] = float(np.max(cluster_sizes))
                    stats_dict['cluster_size_std'] = float(np.std(cluster_sizes))
                    
                    # Largest cluster properties
                    largest_cluster_label = np.argmax(cluster_sizes) + 1
                    largest_cluster_mask = (labeled_array == largest_cluster_label)
                    stats_dict['largest_cluster_fraction'] = float(np.sum(largest_cluster_mask) / active_cells)
                    
                    print(f"  Number of clusters: {num_features}")
                    print(f"  Mean cluster size: {stats_dict['mean_cluster_size']:.1f}")
            except Exception as e:
                print(f"  ⚠️ Could not calculate clusters: {e}")
                stats_dict['num_clusters'] = 0
        else:
            stats_dict['num_clusters'] = 0
            print("  ⚠️ Scipy not available for cluster analysis")
        
        self.statistics['basic'] = stats_dict
        
        print(f"  Density: {stats_dict['density']:.4f}")
        print(f"  Active cells: {stats_dict['active_cells']:,}")
        print(f"  Active fraction: {stats_dict['active_fraction']:.1%}")
        print(f"  Data type: {stats_dict['data_type']}")
        
        return stats_dict
    
    def calculate_fractal_dimension(self):
        """Calculate fractal dimension using box-counting method"""
        print("\n🔍 Calculating fractal dimension...")
        
        data = (self.universe_data > 0).astype(int)
        size = data.shape[0]
        
        # Box sizes (powers of 2)
        max_power = int(np.log2(size)) - 1
        if max_power < 2:
            print("  ⚠️ Grid too small for fractal analysis")
            return None
        
        box_sizes = 2**np.arange(1, max_power)
        counts = []
        
        for box_size in box_sizes:
            # Downsample by averaging
            scale = size // box_size
            if scale > 1:
                # Simple downsampling without scipy
                downsampled = data[::scale, ::scale, ::scale]
                # Count boxes with any active cell
                count = np.sum(downsampled > 0)
                counts.append(count)
            else:
                break
        
        if len(counts) < 3:
            print("  ⚠️ Not enough data points for fractal dimension")
            return None
        
        # Linear fit in log-log space
        valid_box_sizes = box_sizes[:len(counts)]
        log_sizes = np.log(1.0 / valid_box_sizes)
        log_counts = np.log(np.array(counts))
        
        # Simple linear regression without scipy
        A = np.vstack([log_sizes, np.ones(len(log_sizes))]).T
        m, c = np.linalg.lstsq(A, log_counts, rcond=None)[0]
        fractal_dim = m
        
        # Calculate R²
        y_pred = m * log_sizes + c
        ss_res = np.sum((log_counts - y_pred) ** 2)
        ss_tot = np.sum((log_counts - np.mean(log_counts)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        result = {
            'fractal_dimension': float(fractal_dim),
            'r_squared': float(r_squared),
            'box_sizes': valid_box_sizes.tolist(),
            'counts': counts
        }
        
        self.statistics['fractal'] = result
        
        print(f"  Fractal dimension: {fractal_dim:.3f}")
        print(f"  R²: {r_squared:.3f}")
        
        return result
    
    def calculate_correlation_function(self, max_r=None):
        """Calculate two-point correlation function ξ(r)"""
        print("\n📏 Calculating correlation function...")
        
        data = self.universe_data > 0
        size = data.shape[0]
        
        if max_r is None:
            max_r = min(50, size // 4)  # Limit for performance
        
        # Get coordinates of active cells
        coords = np.argwhere(data)
        
        if len(coords) < 100:
            print(f"  ⚠️ Too few active cells ({len(coords)}) for correlation function")
            return None
        
        # Sample pairs for efficiency
        n_samples = min(5000, len(coords))
        sample_indices = np.random.choice(len(coords), n_samples, replace=False)
        sample_coords = coords[sample_indices]
        
        # Calculate pair distances (manual calculation for performance)
        n_points = len(sample_coords)
        distances = []
        
        # Calculate distances between sample points
        for i in range(n_points):
            for j in range(i+1, n_points):
                dist = np.sqrt(np.sum((sample_coords[i] - sample_coords[j])**2))
                if dist <= max_r:
                    distances.append(dist)
        
        distances = np.array(distances)
        
        if len(distances) < 100:
            print(f"  ⚠️ Too few pairs ({len(distances)}) for correlation analysis")
            return None
        
        # Bin distances
        r_bins = np.linspace(1, max_r, 20)
        r_centers = (r_bins[:-1] + r_bins[1:]) / 2
        counts, _ = np.histogram(distances, bins=r_bins)
        
        # Expected counts for uniform distribution
        volume_fraction = np.sum(data) / data.size
        expected_counts = volume_fraction * (4/3 * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3))
        
        # Correlation function ξ(r) = (DD/RR) - 1
        with np.errstate(divide='ignore', invalid='ignore'):
            xi = counts / expected_counts - 1
            xi = np.nan_to_num(xi, nan=0, posinf=0, neginf=0)
        
        # Fit power law ξ(r) ∝ r^(-γ) for valid points
        valid = (xi > 0) & (r_centers > 2) & (r_centers < max_r/2)
        gamma = None
        
        if valid.sum() > 3:
            log_r = np.log(r_centers[valid])
            log_xi = np.log(xi[valid])
            # Simple linear fit
            A = np.vstack([log_r, np.ones(len(log_r))]).T
            m, c = np.linalg.lstsq(A, log_xi, rcond=None)[0]
            gamma = -m
        
        result = {
            'r_bins': r_centers.tolist(),
            'xi_r': xi.tolist(),
            'gamma': float(gamma) if gamma is not None else None,
            'num_pairs': len(distances),
            'num_samples': n_samples
        }
        
        self.statistics['correlation'] = result
        
        if gamma is not None:
            print(f"  Correlation slope γ: {gamma:.3f}")
        print(f"  Sampled {n_samples} cells, {len(distances):,} pairs")
        
        return result
    
    def calculate_void_statistics(self):
        """Analyze void properties and distributions"""
        print("\n🕳️ Calculating void statistics...")
        
        data = self.universe_data > 0
        
        # Simple void analysis without scipy
        # Count zeros and their connectivity
        zeros = data == 0
        
        if not np.any(zeros):
            print("  ⚠️ No voids found")
            return None
        
        # Simple void size estimation using flood fill algorithm
        def flood_fill(grid, start):
            """Simple 3D flood fill"""
            stack = [tuple(start)]
            filled = set()
            size = 0
            
            while stack:
                x, y, z = stack.pop()
                if (0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] and 
                    0 <= z < grid.shape[2] and not grid[x, y, z]):
                    grid[x, y, z] = True  # Mark as visited
                    filled.add((x, y, z))
                    size += 1
                    
                    # Add neighbors
                    stack.append((x+1, y, z))
                    stack.append((x-1, y, z))
                    stack.append((x, y+1, z))
                    stack.append((x, y-1, z))
                    stack.append((x, y, z+1))
                    stack.append((x, y, z-1))
            
            return size
        
        # Sample voids for performance
        void_sizes = []
        sampled_positions = set()
        max_voids_to_sample = 100
        
        # Find zero positions
        zero_positions = np.argwhere(zeros)
        
        if len(zero_positions) == 0:
            print("  ⚠️ No zero positions found")
            return None
        
        # Sample some starting positions
        sample_indices = np.random.choice(len(zero_positions), 
                                         min(max_voids_to_sample, len(zero_positions)), 
                                         replace=False)
        
        visited = zeros.copy()
        
        for idx in sample_indices:
            start = zero_positions[idx]
            if not visited[tuple(start)]:
                size = flood_fill(visited, start)
                void_sizes.append(size)
        
        if not void_sizes:
            print("  ⚠️ Could not measure void sizes")
            return None
        
        void_sizes = np.array(void_sizes)
        
        # Statistics
        void_stats = {
            'num_voids_estimated': len(void_sizes),
            'mean_void_size': float(np.mean(void_sizes)),
            'median_void_size': float(np.median(void_sizes)),
            'max_void_size': float(np.max(void_sizes)),
            'void_size_std': float(np.std(void_sizes)),
            'total_void_volume': int(np.sum(zeros)),
            'void_fraction': float(np.sum(zeros) / zeros.size)
        }
        
        self.statistics['voids'] = void_stats
        
        print(f"  Estimated voids: {len(void_sizes)}")
        print(f"  Mean void size: {void_stats['mean_void_size']:.1f} cells")
        print(f"  Max void size: {void_stats['max_void_size']:.0f} cells")
        print(f"  Void fraction: {void_stats['void_fraction']:.1%}")
        
        return void_stats
    
    def calculate_minkowski_functionals(self):
        """Calculate Minkowski functionals for morphological analysis"""
        print("\n🔷 Calculating Minkowski functionals...")
        
        data = (self.universe_data > 0).astype(float)
        
        # Volume (V0)
        V0 = np.sum(data)
        
        # Surface area (V1) - approximation using gradients
        # Calculate gradients manually
        grad_x = np.zeros_like(data)
        grad_y = np.zeros_like(data)
        grad_z = np.zeros_like(data)
        
        # Central differences for interior points
        grad_x[1:-1, :, :] = (data[2:, :, :] - data[:-2, :, :]) / 2
        grad_y[:, 1:-1, :] = (data[:, 2:, :] - data[:, :-2, :]) / 2
        grad_z[:, :, 1:-1] = (data[:, :, 2:] - data[:, :, :-2]) / 2
        
        # Forward/backward differences for boundaries
        grad_x[0, :, :] = data[1, :, :] - data[0, :, :]
        grad_x[-1, :, :] = data[-1, :, :] - data[-2, :, :]
        grad_y[:, 0, :] = data[:, 1, :] - data[:, 0, :]
        grad_y[:, -1, :] = data[:, -1, :] - data[:, -2, :]
        grad_z[:, :, 0] = data[:, :, 1] - data[:, :, 0]
        grad_z[:, :, -1] = data[:, :, -1] - data[:, :, -2]
        
        # Gradient magnitude
        grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        surface_area = np.sum(grad_mag)
        V1 = surface_area / 6  # Normalization for cubic grid
        
        # Mean curvature (V2) - approximation using Laplacian
        # Calculate Laplacian manually
        laplacian = np.zeros_like(data)
        
        # Interior points
        for i in range(1, data.shape[0]-1):
            for j in range(1, data.shape[1]-1):
                for k in range(1, data.shape[2]-1):
                    laplacian[i, j, k] = (data[i+1, j, k] + data[i-1, j, k] +
                                         data[i, j+1, k] + data[i, j-1, k] +
                                         data[i, j, k+1] + data[i, j, k-1] -
                                         6 * data[i, j, k])
        
        V2 = np.sum(laplacian) / (3 * np.pi) if np.any(laplacian) else 0
        
        # Euler characteristic (V3) - simplified version
        # Count connected components using a simple algorithm
        def count_components(binary_data):
            """Count connected components in 3D binary data"""
            visited = np.zeros_like(binary_data, dtype=bool)
            components = 0
            
            def dfs(x, y, z):
                stack = [(x, y, z)]
                while stack:
                    cx, cy, cz = stack.pop()
                    if (0 <= cx < binary_data.shape[0] and 
                        0 <= cy < binary_data.shape[1] and 
                        0 <= cz < binary_data.shape[2] and 
                        binary_data[cx, cy, cz] and not visited[cx, cy, cz]):
                        visited[cx, cy, cz] = True
                        # Add neighbors
                        stack.append((cx+1, cy, cz))
                        stack.append((cx-1, cy, cz))
                        stack.append((cx, cy+1, cz))
                        stack.append((cx, cy-1, cz))
                        stack.append((cx, cy, cz+1))
                        stack.append((cx, cy, cz-1))
            
            for i in range(binary_data.shape[0]):
                for j in range(binary_data.shape[1]):
                    for k in range(binary_data.shape[2]):
                        if binary_data[i, j, k] and not visited[i, j, k]:
                            components += 1
                            dfs(i, j, k)
            
            return components
        
        V3 = count_components(data > 0.5)
        
        minkowski = {
            'V0_volume': float(V0),
            'V1_surface_area': float(V1),
            'V2_mean_curvature': float(V2),
            'V3_euler_characteristic': float(V3),
            'morphological_ratios': {
                'V1_V0': float(V1 / V0) if V0 > 0 else 0,
                'V2_V1': float(V2 / V1) if V1 > 0 else 0
            }
        }
        
        self.statistics['minkowski'] = minkowski
        
        print(f"  Volume (V0): {V0:,.0f}")
        print(f"  Surface area (V1): {V1:,.1f}")
        print(f"  Mean curvature (V2): {V2:,.1f}")
        print(f"  Euler characteristic (V3): {V3}")
        print(f"  Morphology ratio V1/V0: {minkowski['morphological_ratios']['V1_V0']:.3f}")
        
        return minkowski
    
    def calculate_power_spectrum_statistics(self):
        """Calculate statistics from power spectrum (simplified)"""
        print("\n📡 Calculating power spectrum statistics...")
        
        data = self.universe_data.astype(float)
        size = data.shape[0]
        
        # Simple power spectrum calculation (2D projection for speed)
        projection = np.mean(data, axis=0)  # Average along z-axis
        
        # 2D FFT
        fft_2d = np.fft.fft2(projection)
        power_2d = np.abs(fft_2d)**2
        power_shifted = np.fft.fftshift(power_2d)
        
        # Get k-space coordinates
        k_indices = np.fft.fftshift(np.fft.fftfreq(size)) * size
        kx, ky = np.meshgrid(k_indices, k_indices, indexing='ij')
        k_mag = np.sqrt(kx**2 + ky**2).flatten()
        power_flat = power_shifted.flatten()
        
        # Remove k=0 mode
        non_zero = k_mag > 0
        k_mag = k_mag[non_zero]
        power_flat = power_flat[non_zero]
        
        # Simple binning
        k_bins = np.logspace(np.log10(k_mag.min()), np.log10(k_mag.max()), 20)
        k_centers = (k_bins[:-1] + k_bins[1:]) / 2
        Pk_bins = np.zeros_like(k_centers)
        
        for i in range(len(k_centers)):
            mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
            if np.any(mask):
                Pk_bins[i] = np.mean(power_flat[mask])
        
        # Remove empty bins
        valid = Pk_bins > 0
        k_centers = k_centers[valid]
        Pk_bins = Pk_bins[valid]
        
        # Calculate spectral index if enough points
        n_s = None
        if len(k_centers) >= 3:
            log_k = np.log(k_centers)
            log_Pk = np.log(Pk_bins)
            A = np.vstack([log_k, np.ones(len(log_k))]).T
            m, c = np.linalg.lstsq(A, log_Pk, rcond=None)[0]
            n_s = m
        
        ps_stats = {
            'spectral_index_ns': float(n_s) if n_s is not None else None,
            'power_spectrum_variance': float(np.var(Pk_bins)) if len(Pk_bins) > 0 else None,
            'k_range': [float(k_centers.min()), float(k_centers.max())] if len(k_centers) > 0 else None,
            'Pk_range': [float(Pk_bins.min()), float(Pk_bins.max())] if len(Pk_bins) > 0 else None,
            'note': '2D projection approximation'
        }
        
        self.statistics['power_spectrum'] = ps_stats
        
        if n_s is not None:
            print(f"  Spectral index n_s (approx): {n_s:.3f}")
        else:
            print(f"  Could not calculate spectral index")
        
        return ps_stats
    
    def calculate_cosmic_web_statistics(self):
        """Simple cosmic web classification"""
        print("\n🌌 Classifying cosmic web elements...")
        
        data = self.universe_data.astype(float)
        
        # Simple threshold-based classification
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        # Thresholds based on mean and std
        high_thresh = mean_val + std_val
        low_thresh = mean_val - std_val
        
        clusters = data > high_thresh
        filaments = (data > mean_val) & (data <= high_thresh)
        walls = (data > low_thresh) & (data <= mean_val)
        voids = data <= low_thresh
        
        web_stats = {
            'cluster_fraction': float(np.sum(clusters) / clusters.size),
            'filament_fraction': float(np.sum(filaments) / filaments.size),
            'wall_fraction': float(np.sum(walls) / walls.size),
            'void_fraction': float(np.sum(voids) / voids.size),
            'thresholds': {
                'high': float(high_thresh),
                'mean': float(mean_val),
                'low': float(low_thresh)
            }
        }
        
        # Calculate ratios if possible
        if web_stats['void_fraction'] > 0:
            web_stats['web_ratio_cluster_void'] = web_stats['cluster_fraction'] / web_stats['void_fraction']
        if web_stats['wall_fraction'] > 0:
            web_stats['web_ratio_filament_wall'] = web_stats['filament_fraction'] / web_stats['wall_fraction']
        
        self.statistics['cosmic_web'] = web_stats
        
        print(f"  Clusters: {web_stats['cluster_fraction']:.1%}")
        print(f"  Filaments: {web_stats['filament_fraction']:.1%}")
        print(f"  Walls: {web_stats['wall_fraction']:.1%}")
        print(f"  Voids: {web_stats['void_fraction']:.1%}")
        
        return web_stats
    
    def create_statistical_report(self):
        """Create comprehensive statistical report"""
        print("\n📋 Creating statistical report...")
        
        # Calculate all statistics (skip those that fail)
        statistics_to_calculate = [
            ('Basic Statistics', self.calculate_basic_statistics),
            ('Fractal Dimension', self.calculate_fractal_dimension),
            ('Correlation Function', self.calculate_correlation_function),
            ('Void Statistics', self.calculate_void_statistics),
            ('Minkowski Functionals', self.calculate_minkowski_functionals),
            ('Power Spectrum', self.calculate_power_spectrum_statistics),
            ('Cosmic Web', self.calculate_cosmic_web_statistics)
        ]
        
        for name, calculation_func in statistics_to_calculate:
            print(f"\n  🔍 {name}...")
            try:
                calculation_func()
                print(f"    ✅ Completed")
            except Exception as e:
                print(f"    ⚠️ Failed: {e}")
        
        # Save statistics to JSON
        stats_file = self.output_dir / "statistics_summary.json"
        with open(stats_file, 'w') as f:
            json.dump(self.statistics, f, indent=2)
        
        # Save to CSV for easy analysis
        self.save_statistics_csv()
        
        # Create visualizations
        self.create_statistics_plots()
        
        # Create HTML report
        self.create_html_report()
        
        print(f"\n✅ Statistical analysis complete!")
        print(f"📁 Output saved to: {self.output_dir}")
        
        return self.statistics
    
    def save_statistics_csv(self):
        """Save key statistics to CSV"""
        csv_file = self.output_dir / "key_statistics.csv"
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Category', 'Metric', 'Value', 'Description'])
            
            # Basic statistics
            if 'basic' in self.statistics:
                basic = self.statistics['basic']
                writer.writerow(['Basic', 'Grid Size', basic['grid_size'], 'Grid dimension'])
                writer.writerow(['Basic', 'Active Cells', basic['active_cells'], 'Number of active cells'])
                writer.writerow(['Basic', 'Density', f"{basic['density']:.6f}", 'Mean density'])
                writer.writerow(['Basic', 'Active Fraction', f"{basic['active_fraction']:.3f}", 'Fraction of active volume'])
                if 'num_clusters' in basic:
                    writer.writerow(['Basic', 'Number of Clusters', basic['num_clusters'], 'Connected components'])
            
            # Fractal dimension
            if 'fractal' in self.statistics:
                fractal = self.statistics['fractal']
                writer.writerow(['Fractal', 'Dimension', f"{fractal['fractal_dimension']:.3f}", 
                               'Box-counting fractal dimension'])
                writer.writerow(['Fractal', 'R²', f"{fractal['r_squared']:.3f}", 'Goodness of fit'])
            
            # Correlation function
            if 'correlation' in self.statistics and self.statistics['correlation']['gamma'] is not None:
                corr = self.statistics['correlation']
                writer.writerow(['Correlation', 'Slope γ', f"{corr['gamma']:.3f}", 
                               'Power law slope of ξ(r)'])
            
            # Power spectrum
            if 'power_spectrum' in self.statistics:
                ps = self.statistics['power_spectrum']
                if ps['spectral_index_ns'] is not None:
                    writer.writerow(['Power Spectrum', 'Spectral Index n_s', 
                                   f"{ps['spectral_index_ns']:.3f}", 'Power law slope'])
            
            # Cosmic web
            if 'cosmic_web' in self.statistics:
                web = self.statistics['cosmic_web']
                writer.writerow(['Cosmic Web', 'Cluster Fraction', 
                               f"{web['cluster_fraction']:.3f}", 'Fraction of volume in clusters'])
                writer.writerow(['Cosmic Web', 'Filament Fraction', 
                               f"{web['filament_fraction']:.3f}", 'Fraction of volume in filaments'])
                writer.writerow(['Cosmic Web', 'Void Fraction', 
                               f"{web['void_fraction']:.3f}", 'Fraction of volume in voids'])
        
        print(f"  💾 Saved key statistics to CSV")
    
    def create_statistics_plots(self):
        """Create visualization plots for statistics"""
        print("  🎨 Creating statistical plots...")
        
        try:
            # 1. Summary dashboard
            self.create_summary_dashboard()
            
            # 2. Fractal dimension plot
            if 'fractal' in self.statistics:
                self.plot_fractal_dimension()
            
            # 3. Correlation function plot
            if 'correlation' in self.statistics:
                self.plot_correlation_function()
            
            # 4. Cosmic web classification
            if 'cosmic_web' in self.statistics:
                self.plot_cosmic_web_pie()
            
            print(f"  ✅ Created statistical plots")
        except Exception as e:
            print(f"  ⚠️ Could not create all plots: {e}")
    
    def create_summary_dashboard(self):
        """Create summary dashboard with key statistics"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot 0: Density map (projection)
        if self.universe_data is not None:
            projection = np.mean(self.universe_data, axis=0)
            im0 = axes[0].imshow(projection, cmap='inferno', origin='lower')
            axes[0].set_title('Density Projection (avg along z)', fontsize=10)
            axes[0].set_xlabel('X')
            axes[0].set_ylabel('Y')
            plt.colorbar(im0, ax=axes[0], label='Density')
        
        # Plot 1: Basic statistics bar chart
        if 'basic' in self.statistics:
            basic = self.statistics['basic']
            stats_labels = ['Density', 'Active\nFraction', 'Clusters']
            stats_values = [basic['density'], 
                          basic['active_fraction'],
                          basic.get('num_clusters', 0) / 1000]
            stats_units = ['', '', '×10³']
            
            colors = ['cyan', 'magenta', 'yellow']
            bars = axes[1].bar(range(3), stats_values, color=colors)
            axes[1].set_xticks(range(3))
            axes[1].set_xticklabels(stats_labels, fontsize=9)
            axes[1].set_title('Basic Statistics', fontsize=11)
            axes[1].set_ylabel('Value')
            axes[1].grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, val, unit in zip(bars, stats_values, stats_units):
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2., height, 
                           f'{val:.3f}{unit}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Fractal dimension
        if 'fractal' in self.statistics:
            fractal = self.statistics['fractal']
            box_sizes = fractal['box_sizes']
            counts = fractal['counts']
            
            if len(box_sizes) >= 3:
                axes[2].loglog(box_sizes, counts, 'o-', color='lime', linewidth=2)
                axes[2].set_xlabel('Box Size (inverse)', fontsize=9)
                axes[2].set_ylabel('N(ε)', fontsize=9)
                axes[2].set_title(f'Fractal D: {fractal["fractal_dimension"]:.3f}', fontsize=11)
                axes[2].grid(True, alpha=0.3, which='both')
            else:
                axes[2].text(0.5, 0.5, 'Insufficient data\nfor fractal plot',
                           ha='center', va='center', transform=axes[2].transAxes)
                axes[2].set_title('Fractal Analysis', fontsize=11)
                axes[2].axis('off')
        
        # Plot 3: Correlation function
        if 'correlation' in self.statistics:
            corr = self.statistics['correlation']
            r = corr['r_bins']
            xi = corr['xi_r']
            
            if len(r) >= 3:
                axes[3].loglog(r, xi, 's-', color='orange', linewidth=2, markersize=3)
                axes[3].set_xlabel('r [grid units]', fontsize=9)
                axes[3].set_ylabel('ξ(r)', fontsize=9)
                title = 'Correlation Function'
                if corr['gamma'] is not None:
                    title += f', γ={corr["gamma"]:.2f}'
                axes[3].set_title(title, fontsize=11)
                axes[3].grid(True, alpha=0.3, which='both')
            else:
                axes[3].text(0.5, 0.5, 'Insufficient data\nfor correlation plot',
                           ha='center', va='center', transform=axes[3].transAxes)
                axes[3].set_title('Correlation Function', fontsize=11)
                axes[3].axis('off')
        
        # Plot 4: Power spectrum info
        if 'power_spectrum' in self.statistics:
            ps = self.statistics['power_spectrum']
            info_text = "Power Spectrum Info\n"
            if ps['spectral_index_ns'] is not None:
                info_text += f"n_s = {ps['spectral_index_ns']:.3f}\n"
            if ps['note']:
                info_text += f"Note: {ps['note']}"
            
            axes[4].text(0.5, 0.5, info_text,
                       ha='center', va='center', transform=axes[4].transAxes, fontsize=10)
            axes[4].set_title('Power Spectrum', fontsize=11)
            axes[4].axis('off')
        
        # Plot 5: Cosmic web pie chart
        if 'cosmic_web' in self.statistics:
            web = self.statistics['cosmic_web']
            labels = ['Clusters', 'Filaments', 'Walls', 'Voids']
            values = [web['cluster_fraction'], web['filament_fraction'], 
                     web['wall_fraction'], web['void_fraction']]
            colors = ['red', 'orange', 'yellow', 'blue']
            
            # Only show if we have non-zero values
            if any(v > 0 for v in values):
                axes[5].pie([max(v, 0.01) for v in values], labels=labels, colors=colors, 
                           autopct=lambda pct: f'{pct:.1f}%' if pct > 5 else '')
                axes[5].set_title('Cosmic Web Composition', fontsize=11)
            else:
                axes[5].text(0.5, 0.5, 'No cosmic web\ndata available',
                           ha='center', va='center', transform=axes[5].transAxes)
                axes[5].set_title('Cosmic Web', fontsize=11)
                axes[5].axis('off')
        
        # Hide empty subplots
        for i in range(6):
            if not axes[i].has_data() and not axes[i].texts:
                axes[i].axis('off')
        
        plt.suptitle('EST Statistical Analysis Dashboard', fontsize=16, y=0.98)
        plt.tight_layout()
        
        output_path = self.output_dir / "statistics_dashboard.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_fractal_dimension(self):
        """Plot fractal dimension analysis"""
        fractal = self.statistics['fractal']
        box_sizes = fractal['box_sizes']
        counts = fractal['counts']
        
        if len(box_sizes) < 3:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Log-log plot
        ax1.loglog(box_sizes, counts, 'o-', color='lime', linewidth=2, markersize=6)
        ax1.set_xlabel('1/ε (Inverse Box Size)', fontsize=12)
        ax1.set_ylabel('N(ε)', fontsize=12)
        ax1.set_title(f'Box Counting: D = {fractal["fractal_dimension"]:.3f}', fontsize=14)
        ax1.grid(True, alpha=0.3, which='both')
        
        # Linear fit visualization
        log_sizes = np.log(1.0 / np.array(box_sizes))
        log_counts = np.log(np.array(counts))
        
        ax2.scatter(log_sizes, log_counts, color='cyan', s=50)
        
        # Fit line
        slope = fractal['fractal_dimension']
        intercept = np.mean(log_counts - slope * log_sizes)
        fit_line = slope * log_sizes + intercept
        ax2.plot(log_sizes, fit_line, 'r--', linewidth=2, 
                label=f'Fit: D = {slope:.3f}\nR² = {fractal["r_squared"]:.3f}')
        
        ax2.set_xlabel('log(1/ε)', fontsize=12)
        ax2.set_ylabel('log N(ε)', fontsize=12)
        ax2.set_title('Linear Fit for Fractal Dimension', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "fractal_dimension_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_correlation_function(self):
        """Plot correlation function"""
        corr = self.statistics['correlation']
        r = np.array(corr['r_bins'])
        xi = np.array(corr['xi_r'])
        
        if len(r) < 3:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Log-log plot
        ax1.loglog(r, xi, 's-', color='orange', linewidth=2, markersize=4)
        ax1.set_xlabel('r [grid units]', fontsize=12)
        ax1.set_ylabel('ξ(r)', fontsize=12)
        ax1.set_title('Two-Point Correlation Function', fontsize=14)
        ax1.grid(True, alpha=0.3, which='both')
        
        # Add power law fit if available
        if corr['gamma'] is not None:
            # Generate power law
            r_fit = np.logspace(np.log10(r.min()), np.log10(r.max()), 100)
            # Normalize at median point
            med_idx = len(r) // 2
            if med_idx < len(r) and r[med_idx] > 0 and xi[med_idx] > 0:
                norm = xi[med_idx] / (r[med_idx] ** (-corr['gamma']))
                xi_fit = norm * r_fit ** (-corr['gamma'])
                ax1.loglog(r_fit, xi_fit, 'r--', linewidth=2, 
                          label=f'Power law: ξ(r) ∝ r^{{-{corr["gamma"]:.2f}}}')
                ax1.legend()
        
        # Linear plot for small scales
        ax2.plot(r, xi, 'o-', color='magenta', linewidth=2, markersize=4)
        ax2.set_xlabel('r [grid units]', fontsize=12)
        ax2.set_ylabel('ξ(r)', fontsize=12)
        ax2.set_title('Correlation Function (Linear)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, min(50, r.max())])
        
        plt.tight_layout()
        output_path = self.output_dir / "correlation_function.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_cosmic_web_pie(self):
        """Plot cosmic web composition pie chart"""
        web = self.statistics['cosmic_web']
        
        labels = ['Clusters', 'Filaments', 'Walls', 'Voids']
        values = [web['cluster_fraction'], web['filament_fraction'], 
                 web['wall_fraction'], web['void_fraction']]
        
        # Only plot if we have meaningful values
        if sum(values) == 0 or max(values) < 0.01:
            return
        
        colors = ['red', 'orange', 'yellow', 'blue']
        explode = (0.1, 0.05, 0, 0)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        wedges, texts, autotexts = ax.pie(values, explode=explode, labels=labels, 
                                         colors=colors, autopct='%1.1f%%',
                                         shadow=True, startangle=90)
        
        # Enhance autopct text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(11)
            autotext.set_fontweight('bold')
        
        ax.set_title('Cosmic Web Morphology Classification', fontsize=16, pad=20)
        
        plt.tight_layout()
        output_path = self.output_dir / "cosmic_web_composition.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_html_report(self):
        """Create HTML statistical report"""
        html_path = self.output_dir / "statistical_report.html"
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>EST Statistical Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; background-color: #0a0a0a; color: #ffffff; margin: 20px; }}
        .header {{ text-align: center; padding: 20px; background: linear-gradient(45deg, #1a237e, #4a148c); border-radius: 10px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(600px, 1fr)); gap: 20px; margin: 20px 0; }}
        .card {{ background-color: #1a1a1a; padding: 15px; border-radius: 10px; }}
        .card img {{ width: 100%; border-radius: 5px; }}
        .stats {{ background-color: #2d2d2d; padding: 15px; border-radius: 10px; margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #444; }}
        .highlight {{ color: #4fc3f7; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 EST Statistical Analysis Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
""")
            
            # File information
            f.write(f"""
    <div class="stats">
        <h2>📁 File Information</h2>
        <p><strong>Data Source:</strong> {Path(self.output_dir).parent.name}</p>
        <p><strong>Grid Size:</strong> {self.universe_data.shape[0]}³</p>
        <p><strong>Data Type:</strong> {self.universe_data.dtype}</p>
""")
            
            if self.config:
                f.write("""
        <h3>Configuration</h3>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
""")
                for key, value in list(self.config.items())[:10]:  # Show first 10
                    f.write(f"<tr><td>{key}</td><td class='highlight'>{value}</td></tr>\n")
                f.write("</table>\n")
            
            f.write("""
    </div>
""")
            
            # Basic Statistics
            if 'basic' in self.statistics:
                basic = self.statistics['basic']
                f.write(f"""
    <div class="stats">
        <h2>📈 Basic Statistics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th><th>Description</th></tr>
            <tr><td>Grid Size</td><td class="highlight">{basic['grid_size']}³</td><td>Simulation volume</td></tr>
            <tr><td>Active Cells</td><td class="highlight">{basic['active_cells']:,}</td><td>Matter-containing voxels</td></tr>
            <tr><td>Density</td><td class="highlight">{basic['density']:.6f}</td><td>Mean density (fraction)</td></tr>
            <tr><td>Active Fraction</td><td class="highlight">{basic['active_fraction']:.3%}</td><td>Fraction of active volume</td></tr>
""")
                
                if 'num_clusters' in basic and basic['num_clusters'] > 0:
                    f.write(f"""            <tr><td>Number of Clusters</td><td class="highlight">{basic['num_clusters']:,}</td><td>Connected components</td></tr>
            <tr><td>Mean Cluster Size</td><td class="highlight">{basic.get('mean_cluster_size', 0):.1f}</td><td>Average cluster size</td></tr>
""")
                
                f.write("""        </table>
    </div>
""")
            
            # Other statistics sections would follow similarly...
            
            # Visualizations
            f.write("""
    <h2>📊 Visualizations</h2>
    <div class="grid">
""")
            
            # Add all plot images
            plot_files = list(self.output_dir.glob("*.png"))
            for plot_file in sorted(plot_files):
                rel_path = plot_file.relative_to(self.output_dir.parent)
                title = plot_file.stem.replace('_', ' ').title()
                
                f.write(f"""        <div class="card">
            <h3>{title}</h3>
            <img src="{rel_path}" alt="{title}">
        </div>
""")
            
            f.write("""
    </div>
    
    <div class="stats">
        <h2>📋 Files Generated</h2>
        <ul>
            <li><strong>statistics_dashboard.png</strong> - Summary dashboard</li>
            <li><strong>key_statistics.csv</strong> - All statistics in CSV format</li>
            <li><strong>statistics_summary.json</strong> - Complete statistics in JSON</li>
""")
            
            # List other plot files
            for plot_file in plot_files:
                if 'dashboard' not in plot_file.name:
                    f.write(f"<li><strong>{plot_file.name}</strong> - {plot_file.stem.replace('_', ' ').title()}</li>\n")
            
            f.write("""        </ul>
    </div>
</body>
</html>
""")
        
        print(f"  📋 Created HTML statistical report")

# ==============================================================================
#   COMPATIBILITY SCANNER
# ==============================================================================

def scan_older_simulations():
    """Scan and analyze older simulation files"""
    print(r"""
     _____ ___  _____     _       _       
    | ____/ __||_   _|   | | __ _| |__    
    |  _| \__ \  | |_____| |/ _` | '_ \   
    | |___|___/  | |_____| | (_| | |_) |  
    |_____|___/  |_|     |_|\__,_|_.__/   
    
    EST OLDER SIMULATION SCANNER
    
    🔍 Scans older EST simulation files
    📊 Generates statistical analysis
    📈 Creates comparison reports
    """)
    
    import tkinter as tk
    from tkinter import filedialog
    
    root = tk.Tk()
    root.withdraw()
    
    print("\n📂 Select folder containing older simulations...")
    
    initial_dir = Path.cwd() / "Simulations"
    if not initial_dir.exists():
        initial_dir = Path.cwd()
    
    folder_path = filedialog.askdirectory(
        title="Select Folder with Older Simulations",
        initialdir=str(initial_dir)
    )
    
    if not folder_path:
        print("❌ No folder selected.")
        input("Press Enter to exit...")
        return
    
    folder = Path(folder_path)
    
    # Find all simulation data files
    print(f"\n🔍 Scanning {folder.name} for simulation data...")
    
    data_files = []
    
    # Look for various file types
    file_patterns = [
        "*.npz",
        "final_state*.npy",
        "universe*.npy",
        "data*.npy"
    ]
    
    for pattern in file_patterns:
        files = list(folder.rglob(pattern))
        for file in files:
            # Skip files in analysis subdirectories
            if 'Analysis' not in str(file) and 'Visualization' not in str(file):
                data_files.append(file)
    
    print(f"  Found {len(data_files)} potential data files")
    
    if not data_files:
        print("❌ No simulation data files found")
        input("Press Enter to exit...")
        return
    
    # Create batch analysis directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = folder / f"Batch_Analysis_{timestamp}"
    batch_dir.mkdir(exist_ok=True)
    
    all_statistics = {}
    
    # Analyze each file
    for i, data_file in enumerate(data_files):
        print(f"\n{'='*60}")
        print(f"Analyzing file {i+1}/{len(data_files)}: {data_file.name}")
        print(f"{'='*60}")
        
        try:
            analyzer = EST_Statistics()
            
            if analyzer.load_data(data_file):
                stats = analyzer.create_statistical_report()
                all_statistics[data_file.name] = stats
                
                # Copy results to batch directory
                source_dir = analyzer.output_dir
                dest_dir = batch_dir / data_file.stem
                
                if source_dir.exists():
                    import shutil
                    shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)
                    print(f"  📁 Copied results to batch directory")
            else:
                print(f"  ❌ Failed to load {data_file.name}")
                
        except Exception as e:
            print(f"  ❌ Error analyzing {data_file.name}: {e}")
    
    # Create batch summary
    if all_statistics:
        create_batch_summary(batch_dir, all_statistics)
        
        print(f"\n{'='*60}")
        print("BATCH ANALYSIS COMPLETE!")
        print(f"{'='*60}")
        print(f"📁 Batch results saved to: {batch_dir}")
        print(f"📊 Analyzed {len(all_statistics)} simulations")
        print(f"📋 Summary: batch_summary.csv")
        print(f"📈 Comparison: batch_comparison.png")
        print(f"{'='*60}")
        
        # Try to open the folder
        try:
            os.startfile(batch_dir)
        except:
            pass
    else:
        print("\n❌ No simulations were successfully analyzed")
    
    input("\nPress Enter to exit...")

def create_batch_summary(batch_dir, all_statistics):
    """Create summary of batch analysis"""
    print("\n📋 Creating batch summary...")
    
    # Create CSV summary
    csv_path = batch_dir / "batch_summary.csv"
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        header = ['File', 'Grid_Size', 'Density', 'Active_Cells', 'Active_Fraction', 
                 'Fractal_Dim', 'Correlation_Gamma', 'Spectral_Index_ns',
                 'Cluster_Fraction', 'Filament_Fraction', 'Void_Fraction']
        writer.writerow(header)
        
        # Write data for each simulation
        for filename, stats in all_statistics.items():
            row = [filename]
            
            # Basic stats
            if 'basic' in stats:
                basic = stats['basic']
                row.extend([
                    basic['grid_size'],
                    f"{basic['density']:.6f}",
                    basic['active_cells'],
                    f"{basic['active_fraction']:.3f}"
                ])
            else:
                row.extend(['', '', '', ''])
            
            # Fractal dimension
            if 'fractal' in stats:
                row.append(f"{stats['fractal']['fractal_dimension']:.3f}")
            else:
                row.append('')
            
            # Correlation gamma
            if 'correlation' in stats and stats['correlation']['gamma'] is not None:
                row.append(f"{stats['correlation']['gamma']:.3f}")
            else:
                row.append('')
            
            # Spectral index
            if 'power_spectrum' in stats and stats['power_spectrum']['spectral_index_ns'] is not None:
                row.append(f"{stats['power_spectrum']['spectral_index_ns']:.3f}")
            else:
                row.append('')
            
            # Cosmic web fractions
            if 'cosmic_web' in stats:
                web = stats['cosmic_web']
                row.extend([
                    f"{web['cluster_fraction']:.3f}",
                    f"{web['filament_fraction']:.3f}",
                    f"{web['void_fraction']:.3f}"
                ])
            else:
                row.extend(['', '', ''])
            
            writer.writerow(row)
    
    print(f"  💾 Saved batch summary: {csv_path.name}")
    
    # Create comparison plot if we have enough data
    create_batch_comparison_plot(batch_dir, all_statistics)

def create_batch_comparison_plot(batch_dir, all_statistics):
    """Create comparison plot for batch analysis"""
    # Extract data for plotting
    filenames = []
    densities = []
    fractal_dims = []
    spectral_indices = []
    
    for filename, stats in all_statistics.items():
        if 'basic' in stats and 'fractal' in stats:
            filenames.append(filename[:20] + '...' if len(filename) > 20 else filename)
            densities.append(stats['basic']['density'])
            fractal_dims.append(stats['fractal']['fractal_dimension'])
            
            if 'power_spectrum' in stats and stats['power_spectrum']['spectral_index_ns'] is not None:
                spectral_indices.append(stats['power_spectrum']['spectral_index_ns'])
            else:
                spectral_indices.append(np.nan)
    
    if len(densities) < 2:
        print("  ⚠️ Not enough data for comparison plot")
        return
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Density comparison
    axes[0].bar(range(len(densities)), densities, color='cyan', alpha=0.7)
    axes[0].set_xlabel('Simulation', fontsize=10)
    axes[0].set_ylabel('Density', fontsize=10)
    axes[0].set_title('Density Comparison', fontsize=12)
    axes[0].set_xticks(range(len(filenames)))
    axes[0].set_xticklabels(filenames, rotation=45, ha='right', fontsize=8)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Fractal dimension comparison
    axes[1].bar(range(len(fractal_dims)), fractal_dims, color='lime', alpha=0.7)
    axes[1].set_xlabel('Simulation', fontsize=10)
    axes[1].set_ylabel('Fractal Dimension', fontsize=10)
    axes[1].set_title('Fractal Dimension Comparison', fontsize=12)
    axes[1].set_xticks(range(len(filenames)))
    axes[1].set_xticklabels(filenames, rotation=45, ha='right', fontsize=8)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Spectral index comparison
    valid_indices = ~np.isnan(spectral_indices)
    if valid_indices.sum() > 0:
        valid_names = [filenames[i] for i in range(len(filenames)) if valid_indices[i]]
        valid_values = [spectral_indices[i] for i in range(len(spectral_indices)) if valid_indices[i]]
        
        axes[2].bar(range(len(valid_values)), valid_values, color='magenta', alpha=0.7)
        axes[2].axhline(y=0.965, color='white', linestyle='--', label='Planck n_s=0.965', alpha=0.7)
        axes[2].set_xlabel('Simulation', fontsize=10)
        axes[2].set_ylabel('Spectral Index n_s', fontsize=10)
        axes[2].set_title('Spectral Index Comparison', fontsize=12)
        axes[2].set_xticks(range(len(valid_names)))
        axes[2].set_xticklabels(valid_names, rotation=45, ha='right', fontsize=8)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3, axis='y')
    else:
        axes[2].text(0.5, 0.5, 'No spectral index\ndata available',
                    ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('Spectral Index Comparison', fontsize=12)
        axes[2].axis('off')
    
    plt.suptitle('EST Batch Analysis Comparison', fontsize=14, y=0.98)
    plt.tight_layout()
    
    output_path = batch_dir / "batch_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  📈 Created batch comparison plot")

# ==============================================================================
#   MAIN FUNCTION
# ==============================================================================

def main():
    """Main menu for statistics tools"""
    print(r"""
     _____ ___  _____     _       _       
    | ____/ __||_   _|   | | __ _| |__    
    |  _| \__ \  | |_____| |/ _` | '_ \   
    | |___|___/  | |_____| | (_| | |_) |  
    |_____|___/  |_|     |_|\__,_|_.__/   
    
    EST STATISTICAL ANALYSIS TOOLS
    
    1. Analyze Single Simulation
    2. Scan & Analyze Older Simulations (Batch)
    3. Exit
    """)
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == '1':
        analyze_single_simulation()
    elif choice == '2':
        scan_older_simulations()
    elif choice == '3':
        print("\n👋 Goodbye!")
        sys.exit(0)
    else:
        print("\n⚠️ Invalid choice.")
        input("Press Enter to continue...")
        main()

def analyze_single_simulation():
    """Analyze a single simulation file"""
    import tkinter as tk
    from tkinter import filedialog
    
    root = tk.Tk()
    root.withdraw()
    
    print("\n📂 Select EST data file for analysis...")
    
    initial_dir = Path.cwd() / "Simulations"
    if not initial_dir.exists():
        initial_dir = Path.cwd()
    
    file_path = filedialog.askopenfilename(
        title="Select EST Data File",
        initialdir=str(initial_dir),
        filetypes=[("All supported", "*.npz;*.npy"), ("NPZ files", "*.npz"), ("NPY files", "*.npy"), ("All files", "*.*")]
    )
    
    if not file_path:
        print("❌ No file selected.")
        input("Press Enter to exit...")
        return
    
    # Create statistical analyzer
    analyzer = EST_Statistics()
    
    # Load data
    if not analyzer.load_data(file_path):
        print("❌ Failed to load data.")
        input("Press Enter to exit...")
        return
    
    # Create comprehensive statistical report
    analyzer.create_statistical_report()
    
    print(f"\n{'='*60}")
    print("STATISTICAL ANALYSIS COMPLETE!")
    print(f"{'='*60}")
    print(f"📁 Output saved to: {analyzer.output_dir}")
    print(f"📋 HTML report: statistical_report.html")
    print(f"📊 CSV data: key_statistics.csv")
    print(f"📈 JSON data: statistics_summary.json")
    print(f"{'='*60}")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")