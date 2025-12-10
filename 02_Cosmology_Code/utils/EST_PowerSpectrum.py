"""
EST_PowerSpectrum.py - Power Spectrum Analysis for EST Simulations
===================================================================
Calculates matter power spectrum P(k) from 3D universe data
Compares with ΛCDM expectations and between different simulations
Outputs both simulation units and physical cosmological units
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

# Try to import scipy for better power spectrum calculations
try:
    from scipy import stats, ndimage, optimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ Scipy not available. Using basic power spectrum calculations.")

# Set dark theme for all plots
plt.style.use('dark_background')

class CosmologicalScaler:
    """Convert simulation units to physical cosmological units"""
    
    def __init__(self, box_size_mpc=500.0, H0=67.4, Ω_m=0.315, Ω_b=0.049):
        """
        Parameters:
        - box_size_mpc: Size of simulation box in Mpc/h (comoving)
        - H0: Hubble constant in km/s/Mpc
        - Ω_m: Matter density parameter
        - Ω_b: Baryon density parameter (Planck 2018)
        """
        self.box_size = box_size_mpc  # Mpc/h
        self.H0 = H0  # km/s/Mpc
        self.Ω_m = Ω_m
        self.Ω_b = Ω_b
        self.ρ_crit = 2.775e11 * (H0/100)**2  # h² M☉/Mpc³
        
        # Physical constants
        self.Mpc_to_m = 3.086e22  # meters per Mpc
        self.km_to_m = 1000
        self.Gyr_to_sec = 3.156e16
        
    def k_grid_to_physical(self, k_grid, grid_size):
        """Convert simulation k (grid units) to physical k [h/Mpc]"""
        # k_physical = k_grid * (2π / box_size)
        return k_grid * (2 * np.pi / self.box_size)
    
    def pk_to_physical(self, Pk_grid, grid_size):
        """Convert P(k) to (Mpc/h)^3"""
        voxel_size = self.box_size / grid_size  # Mpc per voxel
        return Pk_grid * (voxel_size**3)
    
    def get_physical_scales(self, grid_size):
        """Return useful scale information"""
        return {
            'box_size_mpc': self.box_size,
            'voxel_size_mpc': self.box_size / grid_size,
            'volume_mpc3': self.box_size**3,
            'H0': self.H0,
            'Ω_m': self.Ω_m,
            'Ω_b': self.Ω_b,
            'critical_density': self.ρ_crit,
            'k_Nyquist': (np.pi * grid_size / self.box_size)  # Nyquist frequency in h/Mpc
        }

class EST_PowerSpectrum:
    """
    Power spectrum analysis tool for EST simulations
    Calculates P(k) from 3D density fields and compares with theoretical models
    """
    
    def __init__(self):
        self.universe_data = None
        self.config = None
        self.scaler = None
        self.physical_scaling = False
        self.output_dir = None
        
    def file_picker(self):
        """Open file picker to select EST data file"""
        try:
            import tkinter as tk
            from tkinter import filedialog
            
            root = tk.Tk()
            root.withdraw()
            
            print("\n📂 Please select your EST data file...")
            print("   Can be: simulation_data.npz, final_state_*.npy, or folder")
            
            # Start in Simulations folder
            initial_dir = Path.cwd() / "Simulations"
            if not initial_dir.exists():
                initial_dir = Path.cwd()
            
            file_path = filedialog.askopenfilename(
                title="Select EST Data File",
                initialdir=str(initial_dir),
                filetypes=[
                    ("NPZ files", "*.npz"),
                    ("NPY files", "*.npy"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                return Path(file_path)
            else:
                print("❌ No file selected.")
                return None
                
        except ImportError:
            # Fallback
            print("\n📂 Manual file selection")
            path_str = input("Enter path to EST data file: ").strip()
            return Path(path_str) if path_str else None
    
    def load_data(self, data_path):
        """Load universe data from various formats"""
        print(f"\n🔍 Loading data from: {data_path.name}")
        
        data_path = Path(data_path)
        
        # Determine if it's a file or folder
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
        """Load data from .npz file"""
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                # Try different possible keys
                if 'final_state' in data:
                    self.universe_data = data['final_state']
                elif 'u' in data:
                    self.universe_data = data['u']
                elif len(data.files) > 0:
                    # Try first array
                    first_key = data.files[0]
                    arr = data[first_key]
                    if arr.ndim == 3:
                        self.universe_data = arr
            
            if self.universe_data is None:
                print("❌ Could not find 3D universe data in .npz file")
                return False
            
            # Try to load config
            try:
                if 'config_str' in data:
                    config_str = str(data['config_str'][0])
                    self.config = json.loads(config_str)
            except:
                print("⚠️ Could not load config from .npz")
            
            # Check for physical scaling
            if self.config and self.config.get('physical_scaling', False):
                self.physical_scaling = True
                box_size = self.config.get('box_size_mpc', 500.0)
                H0 = self.config.get('H0', 67.4)
                Ω_m = self.config.get('Ω_m', 0.315)
                self.scaler = CosmologicalScaler(box_size_mpc=box_size, H0=H0, Ω_m=Ω_m)
            
            # Create output directory
            self.output_dir = npz_path.parent / "Power_Spectrum"
            self.output_dir.mkdir(exist_ok=True)
            
            print(f"✅ Loaded 3D universe data: {self.universe_data.shape}")
            if self.physical_scaling:
                print(f"🌍 Physical scaling enabled: Box={box_size} Mpc/h, H0={H0}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading .npz file: {e}")
            return False
    
    def load_npy_data(self, npy_path):
        """Load data from .npy file"""
        try:
            self.universe_data = np.load(npy_path)
            
            if self.universe_data.ndim != 3:
                print(f"❌ Expected 3D array, got {self.universe_data.ndim}D")
                return False
            
            # Create output directory
            self.output_dir = npy_path.parent / "Power_Spectrum"
            self.output_dir.mkdir(exist_ok=True)
            
            print(f"✅ Loaded 3D universe data: {self.universe_data.shape}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading .npy file: {e}")
            return False
    
    def load_folder_data(self, folder_path):
        """Load data from folder (look for .npy files)"""
        folder = Path(folder_path)
        
        # Look for .npy files
        npy_files = list(folder.glob("*.npy"))
        if not npy_files:
            print(f"❌ No .npy files found in folder")
            return False
        
        # Try to find final_state files first
        final_state_files = [f for f in npy_files if 'final_state' in f.name.lower()]
        if final_state_files:
            # Load the first final_state file
            return self.load_npy_data(final_state_files[0])
        else:
            # Load the largest .npy file (likely the universe data)
            file_sizes = [(f, f.stat().st_size) for f in npy_files]
            file_sizes.sort(key=lambda x: x[1], reverse=True)
            return self.load_npy_data(file_sizes[0][0])
    
    def calculate_power_spectrum(self, universe_grid=None):
        """
        Calculate matter power spectrum P(k) from 3D density field
        
        Parameters:
        -----------
        universe_grid : np.ndarray, optional
            3D array of density values. If None, uses self.universe_data
            
        Returns:
        --------
        dict with keys:
            k_bins : array of k values (simulation units)
            Pk_bins : array of P(k) values (simulation units)
            k_physical : array of k values (h/Mpc, if physical scaling)
            Pk_physical : array of P(k) values ((Mpc/h)^3, if physical scaling)
            n_s : spectral index (power law slope)
        """
        if universe_grid is None:
            universe_grid = self.universe_data
        
        if universe_grid is None:
            raise ValueError("No universe data provided")
        
        size = universe_grid.shape[0]
        print(f"  📊 Calculating power spectrum for {size}³ grid...")
        
        # Convert to overdensity field δ = ρ/ρ̄ - 1
        mean_density = np.mean(universe_grid)
        if mean_density == 0:
            print("  ⚠️ Zero density field - using raw values")
            density_field = universe_grid.astype(float)
        else:
            density_field = (universe_grid.astype(float) / mean_density) - 1.0
        
        # 3D FFT
        print("  🌀 Performing 3D FFT...")
        fft_result = np.fft.fftn(density_field)
        power = np.abs(fft_result)**2
        power_shifted = np.fft.fftshift(power)
        
        # Get k-space coordinates
        k_indices = np.fft.fftshift(np.fft.fftfreq(size)) * size
        kx, ky, kz = np.meshgrid(k_indices, k_indices, k_indices, indexing='ij')
        k_mag = np.sqrt(kx**2 + ky**2 + kz**2).flatten()
        power_flat = power_shifted.flatten()
        
        # Remove k=0 mode (mean)
        non_zero_mask = k_mag > 0
        k_mag = k_mag[non_zero_mask]
        power_flat = power_flat[non_zero_mask]
        
        # Bin by |k|
        k_max = np.sqrt(3) * (size // 2)  # Maximum k magnitude
        n_bins = min(50, size // 4)  # Reasonable number of bins
        
        # Logarithmic binning
        k_min_nonzero = k_mag[k_mag > 0].min()
        log_k_bins = np.logspace(np.log10(k_min_nonzero), 
                                np.log10(k_max), 
                                n_bins + 1)
        
        # Manual binning
        k_bins = np.zeros(n_bins)
        Pk_bins = np.zeros(n_bins)
        counts = np.zeros(n_bins)
        
        for k_val, p_val in zip(k_mag, power_flat):
            if k_val >= log_k_bins[0] and k_val <= log_k_bins[-1]:
                # Find bin index
                bin_idx = np.searchsorted(log_k_bins, k_val) - 1
                if 0 <= bin_idx < n_bins:
                    k_bins[bin_idx] += k_val
                    Pk_bins[bin_idx] += p_val
                    counts[bin_idx] += 1
        
        # Average
        valid = counts > 0
        k_bins[valid] /= counts[valid]
        Pk_bins[valid] /= counts[valid]
        
        # Normalize by volume
        volume_factor = size**3
        Pk_bins[valid] /= volume_factor
        
        # Filter out bins with too few samples
        min_samples = 5
        good_bins = counts >= min_samples
        k_bins = k_bins[good_bins]
        Pk_bins = Pk_bins[good_bins]
        counts = counts[good_bins]
        
        if len(k_bins) < 3:
            print("  ⚠️ Not enough good bins for power spectrum")
            return None
        
        # Calculate power law fit
        n_s = self.fit_power_law(k_bins, Pk_bins)
        
        # Convert to physical units if scaling enabled
        k_physical = None
        Pk_physical = None
        
        if self.physical_scaling and self.scaler:
            k_physical = self.scaler.k_grid_to_physical(k_bins, size)
            Pk_physical = self.scaler.pk_to_physical(Pk_bins, size)
        
        print(f"  ✅ Power spectrum calculated: {len(k_bins)} bins")
        if n_s is not None:
            print(f"     Spectral index n_s = {n_s:.3f}")
        
        return {
            'k_bins': k_bins,
            'Pk_bins': Pk_bins,
            'counts': counts,
            'k_physical': k_physical,
            'Pk_physical': Pk_physical,
            'n_s': n_s,
            'mean_density': mean_density,
            'variance': np.var(density_field)
        }
    
    def fit_power_law(self, k, Pk):
        """Fit power law P(k) ∝ k^n to determine spectral index"""
        if len(k) < 3:
            return None
        
        # Use only well-sampled region (avoid very small and very large k)
        valid = (k > k.min() * 1.1) & (k < k.max() * 0.9) & (Pk > 0)
        if valid.sum() < 3:
            valid = Pk > 0
        
        k_fit = k[valid]
        Pk_fit = Pk[valid]
        
        if len(k_fit) < 2:
            return None
        
        # Linear regression in log-log space
        log_k = np.log10(k_fit)
        log_Pk = np.log10(Pk_fit)
        
        try:
            if SCIPY_AVAILABLE:
                slope, intercept, r_value, p_value, std_err = stats.linregress(log_k, log_Pk)
                n_s = slope
            else:
                # Simple linear regression
                A = np.vstack([log_k, np.ones(len(log_k))]).T
                m, c = np.linalg.lstsq(A, log_Pk, rcond=None)[0]
                n_s = m
            
            return n_s
            
        except:
            return None
    
    def calculate_multiple_ps(self, data_dict):
        """Calculate power spectra for multiple datasets for comparison"""
        results = {}
        
        for name, data in data_dict.items():
            print(f"\n📊 Calculating power spectrum for: {name}")
            ps_result = self.calculate_power_spectrum(data)
            if ps_result:
                results[name] = ps_result
                print(f"  ✅ {name}: n_s = {ps_result['n_s']:.3f}")
            else:
                print(f"  ❌ Failed to calculate power spectrum for {name}")
        
        return results
    
    def compare_with_lcdm(self, ps_result):
        """Compare calculated P(k) with ΛCDM theoretical expectations"""
        if not self.physical_scaling or ps_result['k_physical'] is None:
            print("  ⚠️ Physical scaling not available for ΛCDM comparison")
            return None
        
        k_physical = ps_result['k_physical']
        
        # Simple ΛCDM power spectrum approximation
        # P(k) ∝ k^n_s * T(k)^2, where T(k) is transfer function
        
        # Eisenstein & Hu (1998) approximation parameters
        Ω_m = self.scaler.Ω_m if self.scaler else 0.315
        Ω_b = self.scaler.Ω_b if self.scaler else 0.049
        h = (self.scaler.H0 / 100) if self.scaler else 0.674
        
        # Scale factor
        k_eq = 0.0746 * Ω_m * h**2  # equality scale [h/Mpc]
        
        # Transfer function approximation (BBKS style)
        q = k_physical / (Ω_m * h**2)  # scaled wavenumber
        T_k = np.log(1 + 2.34*q) / (2.34*q)
        T_k *= (1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4)**(-0.25)
        
        # Normalization (σ8 = 0.81 from Planck)
        σ8 = 0.81
        # This is a simplified normalization - real calculation requires integration
        
        # Power spectrum with n_s = 0.965 (Planck 2018)
        n_s_cmb = 0.965
        Pk_lcdm = T_k**2 * (k_physical / 0.05)**(n_s_cmb - 1)
        
        # Normalize to match amplitude at a reference scale
        if len(k_physical) > 5:
            ref_idx = len(k_physical) // 2
            if ps_result['Pk_physical'][ref_idx] > 0:
                scale_factor = ps_result['Pk_physical'][ref_idx] / Pk_lcdm[ref_idx]
                Pk_lcdm *= scale_factor
        
        return {
            'k_lcdm': k_physical,
            'Pk_lcdm': Pk_lcdm,
            'transfer_function': T_k,
            'parameters': {
                'Ω_m': Ω_m,
                'Ω_b': Ω_b,
                'h': h,
                'n_s': n_s_cmb,
                'σ8': σ8
            }
        }
    
    def create_power_spectrum_plots(self, ps_result, compare_result=None):
        """Create comprehensive power spectrum plots"""
        print("\n🎨 Creating power spectrum plots...")
        
        k_bins = ps_result['k_bins']
        Pk_bins = ps_result['Pk_bins']
        n_s = ps_result['n_s']
        
        # 1. Main power spectrum plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot 1: Linear scale
        ax1 = axes[0, 0]
        ax1.plot(k_bins, Pk_bins, 'o-', color='cyan', linewidth=2, markersize=4)
        ax1.set_xlabel('k [grid units]', fontsize=12)
        ax1.set_ylabel('P(k) [simulation units]', fontsize=12)
        ax1.set_title('Matter Power Spectrum (Linear Scale)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Add power law fit if available
        if n_s is not None:
            # Generate fitted curve
            k_fit = np.logspace(np.log10(k_bins.min()), np.log10(k_bins.max()), 100)
            # Find normalization at median k
            med_idx = len(k_bins) // 2
            if med_idx < len(k_bins):
                norm = Pk_bins[med_idx] / (k_bins[med_idx]**n_s)
                Pk_fit = norm * k_fit**n_s
                ax1.plot(k_fit, Pk_fit, 'r--', linewidth=2, 
                        label=f'Power law: n_s = {n_s:.3f}')
                ax1.legend()
        
        # Plot 2: Log-log scale
        ax2 = axes[0, 1]
        ax2.loglog(k_bins, Pk_bins, 's-', color='magenta', linewidth=2, markersize=4)
        ax2.set_xlabel('k [grid units]', fontsize=12)
        ax2.set_ylabel('P(k) [simulation units]', fontsize=12)
        ax2.set_title('Matter Power Spectrum (Log-Log Scale)', fontsize=14)
        ax2.grid(True, alpha=0.3, which='both')
        
        # Plot 3: Physical units (if available)
        ax3 = axes[1, 0]
        if ps_result['k_physical'] is not None and ps_result['Pk_physical'] is not None:
            k_phys = ps_result['k_physical']
            Pk_phys = ps_result['Pk_physical']
            
            ax3.loglog(k_phys, Pk_phys, 'o-', color='yellow', linewidth=2, markersize=4)
            ax3.set_xlabel('k [h/Mpc]', fontsize=12)
            ax3.set_ylabel('P(k) [(Mpc/h)$^3$]', fontsize=12)
            ax3.set_title('Physical Power Spectrum', fontsize=14)
            ax3.grid(True, alpha=0.3, which='both')
            
            # Add ΛCDM comparison if available
            if compare_result is not None:
                ax3.loglog(compare_result['k_lcdm'], compare_result['Pk_lcdm'], 
                         'r--', linewidth=2, label='ΛCDM (approx)')
                ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'Physical scaling not enabled', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Physical Power Spectrum (Not Available)', fontsize=14)
        
        # Plot 4: Residuals or additional info
        ax4 = axes[1, 1]
        
        if n_s is not None and len(k_bins) > 3:
            # Calculate residuals from power law
            med_idx = len(k_bins) // 2
            norm = Pk_bins[med_idx] / (k_bins[med_idx]**n_s)
            Pk_expected = norm * k_bins**n_s
            residuals = (Pk_bins - Pk_expected) / Pk_expected
            
            ax4.plot(k_bins, residuals, 'o-', color='lime', linewidth=2, markersize=4)
            ax4.axhline(y=0, color='white', linestyle='--', alpha=0.5)
            ax4.set_xlabel('k [grid units]', fontsize=12)
            ax4.set_ylabel('(P - P_fit)/P_fit', fontsize=12)
            ax4.set_title('Residuals from Power Law Fit', fontsize=14)
            ax4.grid(True, alpha=0.3)
        else:
            # Show statistics instead
            stats_text = f"Grid size: {self.universe_data.shape[0]}³\n"
            stats_text += f"Mean density: {ps_result['mean_density']:.4f}\n"
            stats_text += f"Density variance: {ps_result['variance']:.4f}\n"
            if n_s is not None:
                stats_text += f"Spectral index n_s: {n_s:.3f}\n"
                stats_text += f"Planck n_s: 0.965 ± 0.004"
            
            ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            ax4.set_title('Power Spectrum Statistics', fontsize=14)
            ax4.axis('off')
        
        plt.suptitle('EST Universe Power Spectrum Analysis', fontsize=16, y=0.98)
        plt.tight_layout()
        
        output_path = self.output_dir / "power_spectrum_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Created comprehensive power spectrum plots")
        
        # 2. Create comparison plot if multiple datasets
        if hasattr(self, 'comparison_results') and self.comparison_results:
            self.create_comparison_plot()
        
        # 3. Create ΛCDM detailed comparison if available
        if compare_result is not None:
            self.create_lcdm_comparison_plot(ps_result, compare_result)
    
    def create_comparison_plot(self):
        """Create plot comparing multiple power spectra"""
        if not hasattr(self, 'comparison_results') or not self.comparison_results:
            return
        
        print("  📈 Creating comparison plot...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.comparison_results)))
        
        # Plot in simulation units
        for idx, (name, result) in enumerate(self.comparison_results.items()):
            color = colors[idx % len(colors)]
            ax1.loglog(result['k_bins'], result['Pk_bins'], 'o-',
                      color=color, linewidth=2, markersize=3,
                      label=f"{name} (n_s={result['n_s']:.3f})")
        
        ax1.set_xlabel('k [grid units]', fontsize=12)
        ax1.set_ylabel('P(k) [simulation units]', fontsize=12)
        ax1.set_title('Power Spectrum Comparison', fontsize=14)
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3, which='both')
        
        # Plot in physical units if available
        ax2_has_data = False
        for idx, (name, result) in enumerate(self.comparison_results.items()):
            if result['k_physical'] is not None and result['Pk_physical'] is not None:
                color = colors[idx % len(colors)]
                ax2.loglog(result['k_physical'], result['Pk_physical'], 's-',
                          color=color, linewidth=2, markersize=3,
                          label=name)
                ax2_has_data = True
        
        if ax2_has_data:
            ax2.set_xlabel('k [h/Mpc]', fontsize=12)
            ax2.set_ylabel('P(k) [(Mpc/h)$^3$]', fontsize=12)
            ax2.set_title('Physical Power Spectrum Comparison', fontsize=14)
            ax2.legend(loc='best', fontsize=9)
            ax2.grid(True, alpha=0.3, which='both')
        else:
            ax2.text(0.5, 0.5, 'Physical scaling not available\nfor comparison',
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Physical Comparison (Not Available)', fontsize=14)
            ax2.axis('off')
        
        plt.suptitle('EST Power Spectrum Comparison Analysis', fontsize=16, y=0.98)
        plt.tight_layout()
        
        output_path = self.output_dir / "power_spectrum_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Created comparison plot")
    
    def create_lcdm_comparison_plot(self, ps_result, lcdm_result):
        """Create detailed ΛCDM comparison plot"""
        print("  🌌 Creating ΛCDM comparison plot...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Direct comparison
        k_phys = ps_result['k_physical']
        Pk_phys = ps_result['Pk_physical']
        Pk_lcdm = lcdm_result['Pk_lcdm']
        
        ax1.loglog(k_phys, Pk_phys, 'o-', color='cyan', linewidth=2, 
                  markersize=4, label='EST Simulation')
        ax1.loglog(k_phys, Pk_lcdm, 'r--', linewidth=2, label='ΛCDM (approx)')
        
        ax1.set_xlabel('k [h/Mpc]', fontsize=12)
        ax1.set_ylabel('P(k) [(Mpc/h)$^3$]', fontsize=12)
        ax1.set_title('EST vs ΛCDM Power Spectrum', fontsize=14)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3, which='both')
        
        # Add ΛCDM parameters
        params = lcdm_result['parameters']
        param_text = f"ΛCDM Parameters:\n"
        param_text += f"Ω_m = {params['Ω_m']:.3f}\n"
        param_text += f"Ω_b = {params['Ω_b']:.3f}\n"
        param_text += f"h = {params['h']:.3f}\n"
        param_text += f"n_s = {params['n_s']:.3f}\n"
        param_text += f"σ₈ = {params['σ8']:.2f}"
        
        ax1.text(0.02, 0.98, param_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # Plot 2: Ratio
        ratio = Pk_phys / Pk_lcdm
        ax2.semilogx(k_phys, ratio, 's-', color='yellow', linewidth=2, markersize=4)
        ax2.axhline(y=1, color='white', linestyle='--', alpha=0.5)
        ax2.set_xlabel('k [h/Mpc]', fontsize=12)
        ax2.set_ylabel('P(k)_EST / P(k)_ΛCDM', fontsize=12)
        ax2.set_title('Ratio: EST / ΛCDM', fontsize=14)
        ax2.grid(True, alpha=0.3, which='both')
        
        # Add statistics
        mean_ratio = np.mean(ratio)
        std_ratio = np.std(ratio)
        stats_text = f"Mean ratio: {mean_ratio:.3f}\n"
        stats_text += f"Std dev: {std_ratio:.3f}\n"
        stats_text += f"Min ratio: {ratio.min():.3f}\n"
        stats_text += f"Max ratio: {ratio.max():.3f}"
        
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        plt.suptitle('Cosmological Power Spectrum Comparison: EST vs ΛCDM', fontsize=16, y=0.98)
        plt.tight_layout()
        
        output_path = self.output_dir / "lcdm_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Created ΛCDM comparison plot")
    
    def save_power_spectrum_data(self, ps_result, filename="power_spectrum_data.csv"):
        """Save power spectrum data to CSV"""
        csv_path = self.output_dir / filename
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            if ps_result['k_physical'] is not None:
                writer.writerow(['k_grid', 'Pk_grid', 'k_hMpc', 'Pk_Mpc3', 'counts'])
                for kg, Pkg, kp, Pkp, cnt in zip(ps_result['k_bins'],
                                                ps_result['Pk_bins'],
                                                ps_result['k_physical'],
                                                ps_result['Pk_physical'],
                                                ps_result['counts']):
                    writer.writerow([f"{kg:.6f}", f"{Pkg:.6e}", 
                                    f"{kp:.6f}", f"{Pkp:.6e}", f"{int(cnt)}"])
            else:
                writer.writerow(['k_grid', 'Pk_grid', 'counts'])
                for kg, Pkg, cnt in zip(ps_result['k_bins'],
                                       ps_result['Pk_bins'],
                                       ps_result['counts']):
                    writer.writerow([f"{kg:.6f}", f"{Pkg:.6e}", f"{int(cnt)}"])
        
        print(f"  💾 Saved power spectrum data: {csv_path.name}")
    
    def create_html_report(self, ps_result, lcdm_comparison=None):
        """Create HTML report with all power spectrum results"""
        html_path = self.output_dir / "power_spectrum_report.html"
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>EST Power Spectrum Analysis Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #0a0a0a;
            color: #ffffff;
            margin: 0;
            padding: 20px;
        }
        .header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(45deg, #1a237e, #4a148c);
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 20px;
        }
        .card {
            background-color: #1a1a1a;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }
        .card img {
            width: 100%;
            height: auto;
            border-radius: 5px;
        }
        .card-title {
            font-size: 18px;
            font-weight: bold;
            margin: 10px 0;
            color: #4fc3f7;
        }
        .stats-box {
            background-color: #2d2d2d;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>📡 EST Power Spectrum Analysis Report</h1>
""")
            
            if self.config and 'experiment_name' in self.config:
                f.write(f"<h2>{self.config['experiment_name']}</h2>")
            
            f.write(f"""        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="stats-box">
        <h2>📊 Power Spectrum Statistics</h2>
        <ul>
""")
            
            # Add statistics
            f.write(f"<li><strong>Grid Size:</strong> {self.universe_data.shape[0]}³</li>\n")
            f.write(f"<li><strong>Mean Density:</strong> {ps_result['mean_density']:.6f}</li>\n")
            f.write(f"<li><strong>Density Variance:</strong> {ps_result['variance']:.6f}</li>\n")
            if ps_result['n_s'] is not None:
                f.write(f"<li><strong>Spectral Index n_s:</strong> {ps_result['n_s']:.3f}</li>\n")
                f.write(f"<li><strong>Planck 2018 n_s:</strong> 0.965 ± 0.004</li>\n")
            
            if self.physical_scaling and self.scaler:
                scales = self.scaler.get_physical_scales(self.universe_data.shape[0])
                f.write(f"<li><strong>Box Size:</strong> {scales['box_size_mpc']} Mpc/h</li>\n")
                f.write(f"<li><strong>Voxel Size:</strong> {scales['voxel_size_mpc']:.3f} Mpc</li>\n")
                f.write(f"<li><strong>Hubble Constant H₀:</strong> {self.scaler.H0} km/s/Mpc</li>\n")
            
            f.write("""        </ul>
    </div>
    
    <h2>📈 Visualizations</h2>
    <div class="grid">
""")
            
            # Add all plot images
            plot_files = list(self.output_dir.glob("*.png"))
            for plot_file in sorted(plot_files):
                rel_path = plot_file.relative_to(self.output_dir.parent)
                title = plot_file.stem.replace('_', ' ').title()
                
                f.write(f"""
        <div class="card">
            <div class="card-title">{title}</div>
            <img src="{rel_path}" alt="{title}">
        </div>
""")
            
            f.write("""
    </div>
    
    <div class="stats-box">
        <h2>🔍 Analysis Notes</h2>
        <p>The power spectrum P(k) measures how matter is distributed at different scales (wavenumber k).</p>
""")
            
            if ps_result['n_s'] is not None:
                diff = abs(ps_result['n_s'] - 0.965)
                f.write(f"<p>Spectral index n_s = {ps_result['n_s']:.3f} ")
                if diff < 0.1:
                    f.write(f"(close to Planck value of 0.965, difference: {diff:.3f})</p>")
                else:
                    f.write(f"(differs from Planck value of 0.965 by {diff:.3f})</p>")
            
            f.write("""
        <p>Run EST_Compare.py to compare multiple simulations.</p>
    </div>
</body>
</html>
""")
        
        print(f"  📋 Created HTML report: {html_path.name}")

# ==============================================================================
#   COMPARISON TOOL
# ==============================================================================

class EST_Compare:
    """Tool to compare multiple EST simulations"""
    
    def __init__(self):
        self.simulations = {}  # name -> data dictionary
        self.power_spectra = {}  # name -> power spectrum results
        self.output_dir = None
    
    def load_multiple_simulations(self, folder_paths):
        """Load multiple simulation folders for comparison"""
        print("\n📂 Loading multiple simulations for comparison...")
        
        for folder_path in folder_paths:
            folder = Path(folder_path)
            sim_name = folder.name
            
            print(f"  Loading: {sim_name}")
            
            # Look for data files
            data_files = []
            
            # Check for .npz files
            npz_files = list(folder.glob("*.npz"))
            if npz_files:
                data_files.extend(npz_files)
            
            # Check for .npy files
            npy_files = list(folder.glob("*final_state*.npy"))
            if npy_files:
                data_files.extend(npy_files)
            
            if not data_files:
                print(f"    ⚠️ No data files found in {sim_name}")
                continue
            
            # Load the first suitable file
            ps_tool = EST_PowerSpectrum()
            if ps_tool.load_data(data_files[0]):
                self.simulations[sim_name] = {
                    'data': ps_tool.universe_data,
                    'config': ps_tool.config,
                    'scaler': ps_tool.scaler
                }
                print(f"    ✅ Loaded {ps_tool.universe_data.shape} grid")
            else:
                print(f"    ❌ Failed to load data from {sim_name}")
        
        if not self.simulations:
            print("❌ No simulations loaded successfully")
            return False
        
        print(f"\n✅ Successfully loaded {len(self.simulations)} simulations")
        return True
    
    def calculate_comparison_power_spectra(self):
        """Calculate power spectra for all loaded simulations"""
        print("\n📊 Calculating power spectra for comparison...")
        
        ps_tool = EST_PowerSpectrum()
        
        for name, sim_data in self.simulations.items():
            print(f"  Calculating for: {name}")
            
            # Set up the tool with this simulation's data
            ps_tool.universe_data = sim_data['data']
            ps_tool.config = sim_data['config']
            ps_tool.scaler = sim_data['scaler']
            ps_tool.physical_scaling = (sim_data['scaler'] is not None)
            
            # Calculate power spectrum
            ps_result = ps_tool.calculate_power_spectrum()
            if ps_result:
                self.power_spectra[name] = ps_result
                print(f"    ✅ n_s = {ps_result['n_s']:.3f}")
            else:
                print(f"    ❌ Failed to calculate power spectrum")
        
        if not self.power_spectra:
            print("❌ No power spectra calculated successfully")
            return False
        
        print(f"\n✅ Calculated power spectra for {len(self.power_spectra)} simulations")
        return True
    
    def create_comparison_report(self, output_dir):
        """Create comprehensive comparison report"""
        self.output_dir = Path(output_dir) / "Comparison_Analysis"
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"\n📈 Creating comparison analysis...")
        
        # 1. Create comparison plots
        self.create_comparison_plots()
        
        # 2. Create statistical summary
        self.create_statistical_summary()
        
        # 3. Create HTML report
        self.create_comparison_html_report()
        
        print(f"\n✅ Comparison analysis complete!")
        print(f"📁 Output saved to: {self.output_dir}")
        
        try:
            os.startfile(self.output_dir)
        except:
            pass
    
    def create_comparison_plots(self):
        """Create comparison plots"""
        if len(self.power_spectra) < 2:
            print("⚠️ Need at least 2 simulations for comparison plots")
            return
        
        print("  🎨 Creating comparison plots...")
        
        # 1. Power spectrum comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.power_spectra)))
        
        # Plot all power spectra
        for idx, (name, ps_result) in enumerate(self.power_spectra.items()):
            color = colors[idx]
            
            # Simulation units
            ax1.loglog(ps_result['k_bins'], ps_result['Pk_bins'], 'o-',
                      color=color, linewidth=2, markersize=3,
                      label=f"{name} (n_s={ps_result['n_s']:.3f})")
            
            # Physical units if available
            if ps_result['k_physical'] is not None:
                ax2.loglog(ps_result['k_physical'], ps_result['Pk_physical'], 's-',
                          color=color, linewidth=2, markersize=3,
                          label=name)
        
        ax1.set_xlabel('k [grid units]', fontsize=12)
        ax1.set_ylabel('P(k) [simulation units]', fontsize=12)
        ax1.set_title('Power Spectrum Comparison', fontsize=14)
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3, which='both')
        
        if any(ps['k_physical'] is not None for ps in self.power_spectra.values()):
            ax2.set_xlabel('k [h/Mpc]', fontsize=12)
            ax2.set_ylabel('P(k) [(Mpc/h)$^3$]', fontsize=12)
            ax2.set_title('Physical Power Spectrum Comparison', fontsize=14)
            ax2.legend(loc='best', fontsize=9)
            ax2.grid(True, alpha=0.3, which='both')
        else:
            ax2.text(0.5, 0.5, 'Physical scaling not available\nfor all simulations',
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Physical Comparison (Limited)', fontsize=14)
            ax2.axis('off')
        
        plt.suptitle('EST Simulation Power Spectrum Comparison', fontsize=16, y=0.98)
        plt.tight_layout()
        
        output_path = self.output_dir / "power_spectrum_comparison_all.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Spectral index comparison
        n_s_values = []
        sim_names = []
        for name, ps_result in self.power_spectra.items():
            if ps_result['n_s'] is not None:
                n_s_values.append(ps_result['n_s'])
                sim_names.append(name)
        
        if len(n_s_values) >= 2:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x_pos = np.arange(len(n_s_values))
            bars = ax.bar(x_pos, n_s_values, color=colors[:len(n_s_values)], alpha=0.7)
            
            ax.axhline(y=0.965, color='white', linestyle='--', 
                      label='Planck 2018: n_s = 0.965', alpha=0.7)
            
            ax.set_xlabel('Simulation', fontsize=12)
            ax.set_ylabel('Spectral Index n_s', fontsize=12)
            ax.set_title('Spectral Index Comparison', fontsize=14)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(sim_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, val in zip(bars, n_s_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            
            output_path = self.output_dir / "spectral_index_comparison.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"    ✅ Created comparison plots")
    
    def create_statistical_summary(self):
        """Create statistical summary CSV"""
        csv_path = self.output_dir / "comparison_statistics.csv"
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Simulation', 'Grid_Size', 'Mean_Density', 'Variance', 
                           'Spectral_Index_n_s', 'Physical_Scaling', 'Notes'])
            
            for name, ps_result in self.power_spectra.items():
                sim_data = self.simulations[name]
                grid_size = sim_data['data'].shape[0]
                physical = 'Yes' if sim_data['scaler'] else 'No'
                
                writer.writerow([
                    name,
                    f"{grid_size}",
                    f"{ps_result['mean_density']:.6f}",
                    f"{ps_result['variance']:.6f}",
                    f"{ps_result['n_s']:.3f}" if ps_result['n_s'] is not None else "N/A",
                    physical,
                    ""
                ])
        
        print(f"    💾 Saved comparison statistics: {csv_path.name}")
    
    def create_comparison_html_report(self):
        """Create HTML comparison report"""
        html_path = self.output_dir / "comparison_report.html"
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>EST Simulation Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; background-color: #0a0a0a; color: #ffffff; margin: 20px; }}
        .header {{ text-align: center; padding: 20px; background: linear-gradient(45deg, #1a237e, #4a148c); border-radius: 10px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(600px, 1fr)); gap: 20px; margin: 20px 0; }}
        .card {{ background-color: #1a1a1a; padding: 15px; border-radius: 10px; }}
        .card img {{ width: 100%; border-radius: 5px; }}
        .stats {{ background-color: #2d2d2d; padding: 15px; border-radius: 10px; margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #444; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 EST Simulation Comparison Report</h1>
        <p>Comparing {len(self.power_spectra)} simulations | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="stats">
        <h2>📊 Statistical Summary</h2>
        <table>
            <tr>
                <th>Simulation</th>
                <th>Grid Size</th>
                <th>Mean Density</th>
                <th>Variance</th>
                <th>n_s</th>
                <th>Physical</th>
            </tr>
""")
            
            for name, ps_result in self.power_spectra.items():
                sim_data = self.simulations[name]
                grid_size = sim_data['data'].shape[0]
                physical = 'Yes' if sim_data['scaler'] else 'No'
                n_s = f"{ps_result['n_s']:.3f}" if ps_result['n_s'] is not None else "N/A"
                
                f.write(f"""            <tr>
                <td>{name}</td>
                <td>{grid_size}³</td>
                <td>{ps_result['mean_density']:.6f}</td>
                <td>{ps_result['variance']:.6f}</td>
                <td>{n_s}</td>
                <td>{physical}</td>
            </tr>
""")
            
            f.write("""        </table>
    </div>
    
    <h2>📈 Comparison Visualizations</h2>
    <div class="grid">
""")
            
            # Add plot images
            plot_files = list(self.output_dir.glob("*.png"))
            for plot_file in sorted(plot_files):
                rel_path = plot_file.relative_to(self.output_dir.parent)
                title = plot_file.stem.replace('_', ' ').title()
                
                f.write(f"""        <div class="card">
            <h3>{title}</h3>
            <img src="{rel_path}" alt="{title}">
        </div>
""")
            
            f.write("""    </div>
    
    <div class="stats">
        <h2>📋 Analysis Notes</h2>
        <p>This report compares {len(self.power_spectra)} EST simulations.</p>
        <p>Spectral index n_s values close to 0.965 (Planck 2018) indicate consistency with ΛCDM predictions.</p>
        <p>Use the CSV file for further statistical analysis in external tools.</p>
    </div>
</body>
</html>
""")
        
        print(f"    📋 Created comparison HTML report")

# ==============================================================================
#   MAIN FUNCTIONS
# ==============================================================================

def run_power_spectrum_analysis():
    """Main function for power spectrum analysis"""
    print(r"""
     _____ ___  _____     _       _       
    | ____/ __||_   _|   | | __ _| |__    
    |  _| \__ \  | |_____| |/ _` | '_ \   
    | |___|___/  | |_____| | (_| | |_) |  
    |_____|___/  |_|     |_|\__,_|_.__/   
    
    EST POWER SPECTRUM ANALYZER
    
    📡 Calculates matter power spectrum P(k) from 3D universe
    🌌 Compares with ΛCDM theoretical expectations
    📊 Creates professional cosmological analysis plots
    """)
    
    ps_tool = EST_PowerSpectrum()
    
    # Select data file
    data_file = ps_tool.file_picker()
    if not data_file:
        print("\n❌ No data file selected.")
        input("Press Enter to exit...")
        return
    
    # Load data
    if not ps_tool.load_data(data_file):
        print("\n❌ Failed to load data.")
        input("Press Enter to exit...")
        return
    
    # Calculate power spectrum
    ps_result = ps_tool.calculate_power_spectrum()
    if not ps_result:
        print("\n❌ Failed to calculate power spectrum.")
        input("Press Enter to exit...")
        return
    
    # Compare with ΛCDM if physical scaling is available
    lcdm_comparison = None
    if ps_tool.physical_scaling:
        lcdm_comparison = ps_tool.compare_with_lcdm(ps_result)
    
    # Create plots
    ps_tool.create_power_spectrum_plots(ps_result, lcdm_comparison)
    
    # Save data
    ps_tool.save_power_spectrum_data(ps_result)
    
    # Create HTML report
    ps_tool.create_html_report(ps_result, lcdm_comparison)
    
    print(f"\n{'='*60}")
    print("POWER SPECTRUM ANALYSIS COMPLETE!")
    print(f"{'='*60}")
    print(f"📁 Output saved to: {ps_tool.output_dir}")
    print(f"📊 Spectral index n_s: {ps_result['n_s']:.3f}")
    print(f"📈 ΛCDM comparison: {'Available' if lcdm_comparison else 'Not available'}")
    print(f"📋 HTML report: power_spectrum_report.html")
    print(f"{'='*60}")
    
    input("\nPress Enter to exit...")

def run_comparison_tool():
    """Main function for comparing multiple simulations"""
    print(r"""
     _____ ___  _____     _       _       
    | ____/ __||_   _|   | | __ _| |__    
    |  _| \__ \  | |_____| |/ _` | '_ \   
    | |___|___/  | |_____| | (_| | |_) |  
    |_____|___/  |_|     |_|\__,_|_.__/   
    
    EST COMPARISON TOOL
    
    🔬 Compare multiple EST simulations
    📊 Analyze differences in power spectra
    📈 Create comprehensive comparison reports
    """)
    
    import tkinter as tk
    from tkinter import filedialog
    
    root = tk.Tk()
    root.withdraw()
    
    print("\n📂 Select multiple simulation folders to compare...")
    print("   (Hold Ctrl/Cmd to select multiple)")
    
    initial_dir = Path.cwd() / "Simulations"
    if not initial_dir.exists():
        initial_dir = Path.cwd()
    
    folder_paths = filedialog.askdirectory(
        title="Select Simulation Folders to Compare",
        initialdir=str(initial_dir),
        mustexist=True
    )
    
    if not folder_paths:
        print("❌ No folders selected.")
        input("Press Enter to exit...")
        return
    
    # Convert to list if single folder
    if isinstance(folder_paths, str):
        folder_paths = [folder_paths]
    
    print(f"\nSelected {len(folder_paths)} folders:")
    for path in folder_paths:
        print(f"  • {Path(path).name}")
    
    # Create comparison tool
    compare_tool = EST_Compare()
    
    # Load simulations
    if not compare_tool.load_multiple_simulations(folder_paths):
        input("Press Enter to exit...")
        return
    
    # Calculate power spectra
    if not compare_tool.calculate_comparison_power_spectra():
        input("Press Enter to exit...")
        return
    
    # Create comparison report
    compare_tool.create_comparison_report(Path.cwd() / "Simulations")
    
    input("\nPress Enter to exit...")

def main():
    """Main menu for power spectrum tools"""
    print(r"""
     _____ ___  _____     _       _       
    | ____/ __||_   _|   | | __ _| |__    
    |  _| \__ \  | |_____| |/ _` | '_ \   
    | |___|___/  | |_____| | (_| | |_) |  
    |_____|___/  |_|     |_|\__,_|_.__/   
    
    EST COSMOLOGICAL ANALYSIS TOOLS
    
    1. Power Spectrum Analysis (Single Simulation)
    2. Compare Multiple Simulations
    3. Exit
    """)
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == '1':
        run_power_spectrum_analysis()
    elif choice == '2':
        run_comparison_tool()
    elif choice == '3':
        print("\n👋 Goodbye!")
        sys.exit(0)
    else:
        print("\n⚠️ Invalid choice.")
        input("Press Enter to continue...")
        main()

# ==============================================================================
#   ENTRY POINT
# ==============================================================================

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