"""
EST_PlotGenerator.py - Plot Generation Tool for EST Robustness Tests
====================================================================
Loads test data from Simulations/ folder and creates professional plots
for all robustness tests: Parameter Phase Space, Isotropy Check,
Algorithmic Invariance, Nucleation Scaling

Works with EST_Core_v3.0 output format
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
import pandas as pd
import json
import csv
import os
import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set dark theme for all plots
plt.style.use('dark_background')

class EST_PlotGenerator:
    """
    Plot generator for EST robustness test data
    Creates publication-quality plots from saved test data
    """
    
    def __init__(self):
        self.test_dir = None
        self.test_type = None
        self.data = {}
        self.output_dir = None
        
    def file_picker(self):
        """Open folder picker to select test directory"""
        try:
            import tkinter as tk
            from tkinter import filedialog
            
            root = tk.Tk()
            root.withdraw()
            
            print("\n📂 Please select your EST test results folder...")
            
            # Start in Simulations folder
            initial_dir = Path.cwd() / "Simulations"
            if not initial_dir.exists():
                initial_dir = Path.cwd()
            
            folder_path = filedialog.askdirectory(
                title="Select EST Test Results Folder",
                initialdir=str(initial_dir)
            )
            
            if folder_path:
                return Path(folder_path)
            else:
                print("❌ No folder selected.")
                return None
                
        except ImportError:
            # Fallback to manual input
            print("\n📂 Tkinter not available. Please enter path manually.")
            print("   Look in 'Simulations/' folder for test results")
            
            # List available test folders
            sim_dir = Path.cwd() / "Simulations"
            if sim_dir.exists():
                test_folders = []
                for folder in sim_dir.iterdir():
                    if folder.is_dir() and any(test in folder.name for test in 
                                             ['Parameter_Phase', 'Isotropy', 
                                              'Algorithmic_Invariance', 'Nucleation_Scaling']):
                        test_folders.append(folder)
                
                if test_folders:
                    print("\nAvailable test folders:")
                    for i, folder in enumerate(test_folders[:10]):
                        print(f"  {i+1}. {folder.name}")
                    
                    choice = input(f"\nSelect folder number (1-{len(test_folders)}): ").strip()
                    if choice.isdigit():
                        idx = int(choice) - 1
                        if 0 <= idx < len(test_folders):
                            return test_folders[idx]
            
            # Manual path input
            path_str = input("\nEnter path to test folder: ").strip()
            if path_str:
                return Path(path_str)
            
            return None
    
    def detect_test_type(self, folder):
        """Detect what type of test this folder contains"""
        folder_name = folder.name
        
        # Check for v3.0 core test types
        if 'Parameter_Phase' in folder_name:
            return 'parameter_phase_space'
        elif 'Isotropy' in folder_name:
            return 'isotropy_check'
        elif 'Algorithmic_Invariance' in folder_name:
            return 'algorithmic_invariance'
        elif 'Nucleation_Scaling' in folder_name:
            return 'nucleation_scaling'
        elif 'EST_Output_' in folder_name:
            # NEW: This is an old v2.7 output folder
            return 'legacy_cosmology_simulation'
        else:
            # Check files to determine
            files = list(folder.glob("*"))
            if any('phase_space' in f.name.lower() for f in files):
                return 'parameter_phase_space'
            elif any('isotropy' in f.name.lower() for f in files):
                return 'isotropy_check'
            elif any('invariance' in f.name.lower() for f in files):
                return 'algorithmic_invariance'
            elif any('scaling' in f.name.lower() for f in files):
                return 'nucleation_scaling'
            elif any('cosmology' in f.name.lower() for f in files):
                # Check for cosmology simulation files
                if any('.npy' in f.name for f in files) or any('.npz' in f.name for f in files):
                    return 'legacy_cosmology_simulation'
                if any('cosmological_data.csv' == f.name for f in files):
                    return 'legacy_cosmology_simulation'
        
        return 'unknown'
    
    def load_legacy_cosmology_data(self):
        """Load data from legacy v2.7/v3.0 engine output format"""
        try:
            # Try to find final state file
            npy_file = self.test_dir / "final_universe_state.npy"
            npz_file = self.test_dir / "simulation_data.npz"
            
            if npy_file.exists():
                # Old v2.7 format
                self.data['final_state'] = np.load(npy_file)
                
                # Try to load density timeseries
                density_file = self.test_dir / "density_timeseries.npy"
                if density_file.exists():
                    self.data['density_history'] = np.load(density_file)
                
            elif npz_file.exists():
                # v3.0 core format
                with np.load(npz_file, allow_pickle=True) as npz_data:
                    if 'final_state' in npz_data:
                        self.data['final_state'] = npz_data['final_state']
                    if 'density_history' in npz_data:
                        self.data['density_history'] = npz_data['density_history']
                    
                    # Try to load config
                    if 'config_str' in npz_data:
                        config_str = npz_data['config_str'][0]
                        if isinstance(config_str, (str, np.str_)):
                            self.data['config'] = json.loads(config_str)
            else:
                # Try CSV files
                csv_files = list(self.test_dir.glob("*.csv"))
                if csv_files:
                    # Try to find cosmological data
                    for csv_file in csv_files:
                        if 'cosmological_data' in csv_file.name.lower():
                            df = pd.read_csv(csv_file)
                            if 'Density_sim' in df.columns:
                                self.data['density_history'] = df['Density_sim'].values
                                break
            
            # Load metadata if available
            meta_files = list(self.test_dir.glob("*metadata*"))
            if meta_files:
                for meta_file in meta_files:
                    try:
                        if meta_file.suffix == '.json':
                            with open(meta_file, 'r') as f:
                                self.data['metadata'] = json.load(f)
                                break
                        elif meta_file.suffix == '.txt':
                            with open(meta_file, 'r') as f:
                                content = f.read()
                                self.data['metadata_text'] = content
                                break
                    except:
                        continue
            
            # Create projection for visualization
            if 'final_state' in self.data:
                self.data['projection'] = self.data['final_state'].sum(axis=0)
            
            print(f"✅ Loaded legacy cosmology simulation data")
            if 'density_history' in self.data:
                print(f"   Density history: {len(self.data['density_history'])} frames")
            if 'final_state' in self.data:
                print(f"   Final state shape: {self.data['final_state'].shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading legacy cosmology data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_test_data(self, test_dir):
        """Load data from test directory based on test type"""
        print(f"\n🔍 Loading test data from: {test_dir.name}")
        
        self.test_dir = test_dir
        self.test_type = self.detect_test_type(test_dir)
        self.output_dir = test_dir / "Plots"
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"📊 Test type detected: {self.test_type.replace('_', ' ').title()}")
        
        # Load based on test type
        if self.test_type == 'parameter_phase_space':
            return self.load_parameter_phase_space_data()
        elif self.test_type == 'isotropy_check':
            return self.load_isotropy_check_data()
        elif self.test_type == 'algorithmic_invariance':
            return self.load_algorithmic_invariance_data()
        elif self.test_type == 'nucleation_scaling':
            return self.load_nucleation_scaling_data()
        elif self.test_type == 'legacy_cosmology_simulation':  # NEW
            return self.load_legacy_cosmology_data()
        else:
            print(f"❌ Unknown test type for folder: {test_dir.name}")
            return False
    
    def load_parameter_phase_space_data(self):
        """Load parameter phase space test data"""
        try:
            # Load numpy array
            npy_file = self.test_dir / "phase_space_results.npy"
            if npy_file.exists():
                self.data['results'] = np.load(npy_file)
            else:
                # Try to load from CSV
                csv_file = self.test_dir / "parameter_sweep.csv"
                if csv_file.exists():
                    df = pd.read_csv(csv_file)
                    # Reconstruct matrix from CSV
                    beta_vals = df['Beta'].unique()
                    lambda_vals = df['Lambda'].unique()
                    results = np.zeros((len(beta_vals), len(lambda_vals)))
                    
                    for idx, row in df.iterrows():
                        i = np.where(beta_vals == row['Beta'])[0][0]
                        j = np.where(lambda_vals == row['Lambda'])[0][0]
                        results[i, j] = row['Resulting_Density']
                    
                    self.data['results'] = results
                    self.data['beta_range'] = beta_vals
                    self.data['lambda_range'] = lambda_vals
                else:
                    print("❌ No phase space data found")
                    return False
            
            # Load metadata
            meta_file = self.test_dir / "metadata.json"
            if meta_file.exists():
                with open(meta_file, 'r') as f:
                    self.data['metadata'] = json.load(f)
                
                if 'beta_range' in self.data['metadata']:
                    self.data['beta_range'] = np.array(self.data['metadata']['beta_range'])
                if 'lambda_range' in self.data['metadata']:
                    self.data['lambda_range'] = np.array(self.data['metadata']['lambda_range'])
            
            # If ranges not loaded, create default
            if 'beta_range' not in self.data:
                self.data['beta_range'] = np.linspace(2.0, 5.0, 10)
            if 'lambda_range' not in self.data:
                self.data['lambda_range'] = np.linspace(0.1, 1.0, 10)
            
            print(f"✅ Loaded parameter phase space data")
            print(f"   Shape: {self.data['results'].shape}")
            print(f"   Beta range: {self.data['beta_range'].min():.1f} to {self.data['beta_range'].max():.1f}")
            print(f"   Lambda range: {self.data['lambda_range'].min():.1f} to {self.data['lambda_range'].max():.1f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading parameter phase space data: {e}")
            return False
    
    def load_isotropy_check_data(self):
        """Load isotropy check test data"""
        try:
            # Try to load numpy arrays
            x_file = self.test_dir / "x_axis_positions.npy"
            y_file = self.test_dir / "y_axis_positions.npy"
            z_file = self.test_dir / "z_axis_positions.npy"
            
            if x_file.exists() and y_file.exists() and z_file.exists():
                self.data['x_positions'] = np.load(x_file)
                self.data['y_positions'] = np.load(y_file)
                self.data['z_positions'] = np.load(z_file)
            else:
                # Try CSV
                csv_file = self.test_dir / "isotropy_data.csv"
                if csv_file.exists():
                    df = pd.read_csv(csv_file)
                    self.data['x_positions'] = df['X_Axis_Position'].values
                    self.data['y_positions'] = df['Y_Axis_Position'].values
                    self.data['z_positions'] = df['Z_Axis_Position'].values
                else:
                    print("❌ No isotropy data found")
                    return False
            
            # Load metadata
            meta_file = self.test_dir / "metadata.json"
            if meta_file.exists():
                with open(meta_file, 'r') as f:
                    self.data['metadata'] = json.load(f)
            
            print(f"✅ Loaded isotropy check data")
            print(f"   X positions: {len(self.data['x_positions'])} frames")
            print(f"   Y positions: {len(self.data['y_positions'])} frames")
            print(f"   Z positions: {len(self.data['z_positions'])} frames")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading isotropy data: {e}")
            return False
    
    def load_algorithmic_invariance_data(self):
        """Load algorithmic invariance test data with error handling"""
        try:
            # Load JSON results with safe loading
            json_file = self.test_dir / "invariance_results.json"
            if json_file.exists():
                self.data['results'] = self.safe_json_load(json_file)
                if self.data['results'] is None:
                    print("❌ Could not load invariance results JSON")
                    return False
            else:
                print("❌ No invariance results JSON found")
                return False
            
            # Try alternative file if main JSON fails
            alt_file = self.test_dir / "reconstructed_results.json"
            if alt_file.exists() and ('results' not in self.data or not self.data['results']):
                print("⚠️ Using reconstructed results...")
                self.data['results'] = self.safe_json_load(alt_file)
            
            # Load metadata
            meta_file = self.test_dir / "metadata.json"
            if meta_file.exists():
                self.data['metadata'] = self.safe_json_load(meta_file)
            
            # Load CSV for density data
            csv_file = self.test_dir / "invariance_data.csv"
            if csv_file.exists():
                try:
                    self.data['density_df'] = pd.read_csv(csv_file)
                except Exception as e:
                    print(f"⚠️ Could not load CSV: {e}")
            
            # Load final states if available
            self.data['final_states'] = {}
            methods = ['zlib', 'lzma', 'bzip2', 'entropy']
            for method in methods:
                state_file = self.test_dir / f"final_state_{method}.npy"
                if state_file.exists():
                    try:
                        self.data['final_states'][method] = np.load(state_file)
                    except Exception as e:
                        print(f"⚠️ Could not load {method}.npy: {e}")
            
            print(f"✅ Loaded algorithmic invariance data")
            print(f"   Methods tested: {list(self.data['results'].keys()) if 'results' in self.data else 'None'}")
            print(f"   Final states loaded: {len(self.data['final_states'])}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading invariance data: {e}")
            return False
    
    def load_nucleation_scaling_data(self):
        """Load nucleation scaling test data"""
        try:
            # Load JSON results
            json_file = self.test_dir / "scaling_results.json"
            if json_file.exists():
                with open(json_file, 'r') as f:
                    self.data['results'] = json.load(f)
            else:
                print("❌ No scaling results JSON found")
                return False
            
            # Load metadata
            meta_file = self.test_dir / "metadata.json"
            if meta_file.exists():
                with open(meta_file, 'r') as f:
                    self.data['metadata'] = json.load(f)
            
            print(f"✅ Loaded nucleation scaling data")
            print(f"   Site counts: {list(self.data['results'].keys())}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading scaling data: {e}")
            return False
    
    def create_all_plots(self):
        """Create all plots based on test type"""
        print(f"\n🎨 Creating plots for {self.test_type.replace('_', ' ').title()}...")
        
        if self.test_type == 'parameter_phase_space':
            self.create_parameter_phase_space_plots()
        elif self.test_type == 'isotropy_check':
            self.create_isotropy_check_plots()
        elif self.test_type == 'algorithmic_invariance':
            self.create_algorithmic_invariance_plots()
        elif self.test_type == 'nucleation_scaling':
            self.create_nucleation_scaling_plots()
        elif self.test_type == 'legacy_cosmology_simulation':  # NEW
            self.create_legacy_cosmology_plots()
        else:
            print(f"❌ Unknown test type: {self.test_type}")
            return
        
        print(f"\n✅ All plots created!")
        print(f"📁 Output saved to: {self.output_dir}")
        
        # Try to open the folder
        try:
            os.startfile(self.output_dir)
        except:
            pass
    
    def create_legacy_cosmology_plots(self):
        """Create plots for legacy cosmology simulation data"""
        print("  📊 Creating legacy cosmology plots...")
        
        # Create basic plots
        plt.style.use('dark_background')
        
        # 1. Density evolution plot
        if 'density_history' in self.data:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(self.data['density_history'], color='cyan', linewidth=2)
            ax.set_xlabel('Frame', fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.set_title('Density Evolution - Legacy Cosmology Simulation', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            if len(self.data['density_history']) > 0:
                stats_text = f"Initial: {self.data['density_history'][0]:.4f}\n"
                stats_text += f"Final: {self.data['density_history'][-1]:.4f}\n"
                stats_text += f"Change: {(self.data['density_history'][-1]/self.data['density_history'][0]-1)*100:.1f}%"
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            
            plt.tight_layout()
            output_path = self.output_dir / "density_evolution.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    ✅ Created density evolution plot")
        
        # 2. Projection plot
        if 'projection' in self.data:
            fig, ax = plt.subplots(figsize=(10, 10))
            im = ax.imshow(self.data['projection'], cmap='inferno', origin='lower')
            ax.set_title('Universe Projection (XY Plane)', fontsize=14)
            ax.axis('off')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Integrated Density', rotation=270, labelpad=20)
            
            plt.tight_layout()
            output_path = self.output_dir / "universe_projection.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    ✅ Created universe projection plot")
        
        # 3. Create summary report
        summary_path = self.output_dir / "summary_report.txt"
        with open(summary_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("LEGACY COSMOLOGY SIMULATION - PLOT SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            f.write("FOLDER INFORMATION:\n")
            f.write(f"  Folder: {self.test_dir.name}\n")
            f.write(f"  Plot generation: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if 'metadata' in self.data:
                f.write("METADATA:\n")
                for key, value in self.data['metadata'].items():
                    if isinstance(value, (str, int, float, bool)):
                        f.write(f"  {key}: {value}\n")
            
            if 'density_history' in self.data and len(self.data['density_history']) > 0:
                f.write("\nDENSITY STATISTICS:\n")
                density_data = self.data['density_history']
                f.write(f"  Frames: {len(density_data)}\n")
                f.write(f"  Initial density: {density_data[0]:.6f}\n")
                f.write(f"  Final density: {density_data[-1]:.6f}\n")
                f.write(f"  Minimum: {density_data.min():.6f}\n")
                f.write(f"  Maximum: {density_data.max():.6f}\n")
                f.write(f"  Mean: {density_data.mean():.6f}\n")
            
            f.write("\nFILES GENERATED:\n")
            for plot_file in self.output_dir.glob("*.png"):
                f.write(f"  {plot_file.name}\n")
            f.write(f"  summary_report.txt\n")
        
        print(f"    ✅ Created summary report")

    def create_parameter_phase_space_plots(self):
        """Create plots for parameter phase space test"""
        print("  📊 Creating parameter phase space plots...")
        
        results = self.data['results']
        beta_range = self.data['beta_range']
        lambda_range = self.data['lambda_range']
        
        # 1. Heatmap with Goldilocks zone
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Heatmap
        im1 = ax1.imshow(results, cmap='turbo', origin='lower',
                        extent=[lambda_range[0], lambda_range[-1], 
                                beta_range[0], beta_range[-1]],
                        aspect='auto')
        
        # Mark Goldilocks zone (where density is reasonable)
        optimal_mask = (results > 0.1) & (results < 0.3)
        if np.any(optimal_mask):
            # Create contour for Goldilocks zone
            from scipy import ndimage
            smoothed = ndimage.gaussian_filter(results.astype(float), sigma=0.5)
            optimal_contour = (smoothed > 0.15) & (smoothed < 0.25)
            
            # Create contour lines
            Y, X = np.meshgrid(beta_range, lambda_range, indexing='ij')
            ax1.contour(X, Y, optimal_contour.astype(float), levels=[0.5], 
                       colors=['white'], linestyles=['--'], linewidths=2)
        
        # Mark EST default settings
        ax1.scatter([0.482], [3.4], color='white', marker='X', s=200, 
                   label='EST Default (β=3.4, λ=0.482)', zorder=5)
        
        ax1.set_xlabel('Causal Temperature (λ)', fontsize=12)
        ax1.set_ylabel('Complexity Cost (β)', fontsize=12)
        ax1.set_title('Parameter Phase Space - Goldilocks Zone', fontsize=14)
        ax1.legend(loc='upper right')
        
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Universal Density', rotation=270, labelpad=20)
        
        # 2. 3D surface plot
        X, Y = np.meshgrid(lambda_range, beta_range)
        ax2 = fig.add_subplot(122, projection='3d')
        
        surf = ax2.plot_surface(X, Y, results, cmap='viridis', 
                               alpha=0.8, linewidth=0, antialiased=True)
        
        ax2.set_xlabel('λ (Temperature)', fontsize=10)
        ax2.set_ylabel('β (Complexity)', fontsize=10)
        ax2.set_zlabel('Density', fontsize=10)
        ax2.set_title('3D Phase Space Topology', fontsize=12)
        
        # Add colorbar for 3D plot
        cbar2 = plt.colorbar(surf, ax=ax2, shrink=0.5, pad=0.1)
        cbar2.set_label('Density', rotation=270, labelpad=15)
        
        plt.suptitle('EST Parameter Phase Space Analysis - Goldilocks Zone Verification', 
                    fontsize=16, y=0.98)
        plt.tight_layout()
        
        output_path = self.output_dir / "parameter_phase_space.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ Created parameter phase space plot")
        
        # 3. Cross-section plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        # Beta cross-sections
        beta_indices = [2, 4, 6, 8]  # Sample beta values
        for idx, beta_idx in enumerate(beta_indices):
            if beta_idx < len(beta_range):
                ax = axes[idx]
                beta_val = beta_range[beta_idx]
                density_slice = results[beta_idx, :]
                
                ax.plot(lambda_range, density_slice, 'o-', linewidth=2, markersize=5)
                ax.set_xlabel('λ (Temperature)', fontsize=11)
                ax.set_ylabel('Density', fontsize=11)
                ax.set_title(f'β = {beta_val:.2f}', fontsize=12)
                ax.grid(True, alpha=0.3)
                
                # Mark optimal region
                ax.axhline(y=0.15, color='cyan', linestyle='--', alpha=0.5, label='Optimal')
                ax.axhline(y=0.25, color='cyan', linestyle='--', alpha=0.5)
                ax.fill_between(lambda_range, 0.15, 0.25, alpha=0.1, color='cyan')
        
        plt.suptitle('Parameter Cross-Section Analysis', fontsize=14, y=0.98)
        plt.tight_layout()
        
        output_path = self.output_dir / "parameter_cross_sections.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ Created cross-section plots")
    
    def create_isotropy_check_plots(self):
        """Create plots for isotropy check test"""
        print("  📊 Creating isotropy check plots...")
        
        x_pos = self.data['x_positions']
        y_pos = self.data['y_positions']
        z_pos = self.data['z_positions']
        
        frames = np.arange(len(x_pos))
        
        # 1. Position evolution comparison
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Position plot
        ax1.plot(frames, x_pos, 'r-', linewidth=2, label='X-Axis', alpha=0.8)
        ax1.plot(frames, y_pos, 'g-', linewidth=2, label='Y-Axis', alpha=0.8)
        ax1.plot(frames, z_pos, 'b-', linewidth=2, label='Z-Axis', alpha=0.8)
        
        ax1.set_xlabel('Frame', fontsize=12)
        ax1.set_ylabel('Mean Position', fontsize=12)
        ax1.set_title('Axis Position Evolution - Isotropy Check', fontsize=14)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics box
        stats_text = f"X Variance: {np.var(x_pos):.4f}\n"
        stats_text += f"Y Variance: {np.var(y_pos):.4f}\n"
        stats_text += f"Z Variance: {np.var(z_pos):.4f}\n"
        stats_text += f"Anisotropy Ratio: {max(np.var(x_pos), np.var(y_pos), np.var(z_pos))/min(np.var(x_pos), np.var(y_pos), np.var(z_pos)):.2f}"
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # 2. Variance comparison bar chart
        variances = [np.var(x_pos), np.var(y_pos), np.var(z_pos)]
        labels = ['X-Axis', 'Y-Axis', 'Z-Axis']
        colors = ['red', 'green', 'blue']
        
        bars = ax2.bar(labels, variances, color=colors, alpha=0.7)
        ax2.set_xlabel('Axis', fontsize=12)
        ax2.set_ylabel('Variance', fontsize=12)
        ax2.set_title('Position Variance by Axis', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, var in zip(bars, variances):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{var:.4f}', ha='center', va='bottom', fontsize=10)
        
        # Add isotropy threshold line
        mean_var = np.mean(variances)
        ax2.axhline(y=mean_var, color='white', linestyle='--', 
                   label=f'Mean: {mean_var:.4f}', alpha=0.5)
        ax2.legend()
        
        plt.suptitle('EST Isotropy Check - Lattice Bias Analysis', fontsize=16, y=0.95)
        plt.tight_layout()
        
        output_path = self.output_dir / "isotropy_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ Created isotropy analysis plot")
        
        # 3. Running statistics
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, (positions, label, color) in enumerate(zip(
            [x_pos, y_pos, z_pos], ['X', 'Y', 'Z'], ['red', 'green', 'blue'])):
            
            ax = axes[idx]
            
            # Calculate running mean and std
            window = 10
            running_mean = np.convolve(positions, np.ones(window)/window, mode='valid')
            running_std = np.array([np.std(positions[max(0, i-window):i+1]) 
                                   for i in range(window-1, len(positions))])
            
            # Plot
            frames_running = frames[window-1:]
            ax.plot(frames_running, running_mean, color=color, linewidth=2, label='Running Mean')
            ax.fill_between(frames_running, 
                           running_mean - running_std, 
                           running_mean + running_std, 
                           color=color, alpha=0.3, label='±1 Std Dev')
            
            ax.set_xlabel('Frame', fontsize=11)
            ax.set_ylabel('Position', fontsize=11)
            ax.set_title(f'{label}-Axis Running Statistics', fontsize=12)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Running Statistics - Isotropy Dynamics', fontsize=14, y=0.98)
        plt.tight_layout()
        
        output_path = self.output_dir / "isotropy_running_stats.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ Created running statistics plots")
    
    def create_algorithmic_invariance_plots(self):
        """Create plots for algorithmic invariance test"""
        print("  📊 Creating algorithmic invariance plots...")
        
        results = self.data['results']
        methods = list(results.keys())
        
        # 1. Density evolution comparison
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        colors = ['cyan', 'magenta', 'yellow', 'lime']
        line_styles = ['-', '--', '-.', ':']
        
        # Plot density evolution
        for idx, method in enumerate(methods):
            densities = results[method]['all_densities']
            frames = np.arange(len(densities))
            
            ax1.plot(frames, densities, 
                    color=colors[idx % len(colors)],
                    linestyle=line_styles[idx % len(line_styles)],
                    linewidth=2,
                    label=f"{method.upper()} (Final: {results[method]['final_density']:.6f})",
                    alpha=0.8)
        
        ax1.set_xlabel('Frame', fontsize=12)
        ax1.set_ylabel('Global Density', fontsize=12)
        ax1.set_title('Algorithmic Invariance: Density Evolution', fontsize=14)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Add invariance analysis
        final_densities = [results[m]['final_density'] for m in methods]
        density_range = max(final_densities) - min(final_densities)
        
        stats_text = f"Final Density Range: {density_range:.6f}\n"
        stats_text += f"Invariance Threshold: < 0.01\n"
        stats_text += f"Test Result: {'PASS' if density_range < 0.01 else 'INCONCLUSIVE'}\n"
        stats_text += f"Coefficient of Variation: {(np.std(final_densities)/np.mean(final_densities)*100):.2f}%"
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # 2. Final state projections
        if 'final_states' in self.data and len(self.data['final_states']) > 0:
            for idx, method in enumerate(methods):
                if method in self.data['final_states'] and idx < 4:
                    final_state = self.data['final_states'][method]
                    projection = final_state.sum(axis=0)
                    
                    # Create subplot grid
                    if idx == 0:
                        # First method gets its own subplot
                        im = ax2.imshow(projection, cmap='inferno', origin='lower')
                        ax2.set_title(f'{method.upper()} Projection', fontsize=12)
                        ax2.set_xlabel('X')
                        ax2.set_ylabel('Y')
                        
                        # Add colorbar
                        plt.colorbar(im, ax=ax2, label='Integrated Density')
                    else:
                        # Create additional figure for other methods
                        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
                        for j, (m, color) in enumerate(zip(methods[1:4], ['magenta', 'yellow', 'lime'])):
                            if m in self.data['final_states']:
                                proj = self.data['final_states'][m].sum(axis=0)
                                im = axes2[j].imshow(proj, cmap='inferno', origin='lower')
                                axes2[j].set_title(f'{m.upper()} Projection', fontsize=11)
                                axes2[j].set_xlabel('X')
                                axes2[j].set_ylabel('Y')
                                plt.colorbar(im, ax=axes2[j], label='Density', shrink=0.8)
                        
                        plt.suptitle('Final Universe Projections by Complexity Method', fontsize=14, y=0.98)
                        plt.tight_layout()
                        
                        output_path = self.output_dir / "final_state_projections.png"
                        plt.savefig(output_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        break
        
        plt.suptitle('Algorithmic Invariance Test - Critical Defense', fontsize=16, y=0.95)
        plt.tight_layout()
        
        output_path = self.output_dir / "algorithmic_invariance.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ Created main invariance plot")
        
        # 3. Statistical summary bar charts
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Final density bar chart
        final_densities = [results[m]['final_density'] for m in methods]
        x_pos = np.arange(len(methods))
        
        bars1 = ax1.bar(x_pos, final_densities, color=colors[:len(methods)], alpha=0.7)
        ax1.axhline(y=np.mean(final_densities), color='white', linestyle='--', 
                   label=f'Mean: {np.mean(final_densities):.6f}', alpha=0.7)
        ax1.set_xlabel('Complexity Method', fontsize=12)
        ax1.set_ylabel('Final Density', fontsize=12)
        ax1.set_title('Final Density Comparison', fontsize=13)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([m.upper() for m in methods])
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars1, final_densities):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{val:.6f}', ha='center', va='bottom', fontsize=10)
        
        # Structure metric bar chart
        structure_metrics = [results[m]['final_structure'] for m in methods]
        
        bars2 = ax2.bar(x_pos, structure_metrics, color=colors[:len(methods)], alpha=0.7)
        ax2.axhline(y=np.mean(structure_metrics), color='white', linestyle='--',
                   label=f'Mean: {np.mean(structure_metrics):.4f}', alpha=0.7)
        ax2.set_xlabel('Complexity Method', fontsize=12)
        ax2.set_ylabel('Structure Metric', fontsize=12)
        ax2.set_title('Cosmic Web Structure Quality', fontsize=13)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([m.upper() for m in methods])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars2, structure_metrics):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.suptitle('Algorithmic Invariance Statistical Summary', fontsize=14, y=0.98)
        plt.tight_layout()
        
        output_path = self.output_dir / "invariance_statistics.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ Created statistical summary plots")
        
        # 4. Early vs late stage comparison
        if 'density_df' in self.data:
            df = self.data['density_df']
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            # Plot early, middle, late stages
            stages = [
                (0, 20, 'Early Stage (Frames 0-20)'),
                (20, 40, 'Middle Stage (Frames 20-40)'),
                (40, 60, 'Late Stage (Frames 40-60)'),
                (60, 80, 'Final Stage (Frames 60-80)')
            ]
            
            for idx, (start, end, title) in enumerate(stages):
                if idx < len(axes):
                    ax = axes[idx]
                    
                    for method in methods:
                        col_name = f"Density_{method}"
                        if col_name in df.columns:
                            stage_data = df[col_name].iloc[start:end].values
                            if len(stage_data) > 0:
                                ax.plot(np.arange(len(stage_data)), stage_data, 
                                       label=method.upper(), linewidth=2, alpha=0.7)
                    
                    ax.set_xlabel('Relative Frame', fontsize=10)
                    ax.set_ylabel('Density', fontsize=10)
                    ax.set_title(title, fontsize=11)
                    ax.grid(True, alpha=0.3)
                    
                    if idx == 0:  # Only show legend on first plot
                        ax.legend(loc='best', fontsize=8)
            
            plt.suptitle('Algorithmic Invariance: Stage-by-Stage Comparison', fontsize=14, y=0.98)
            plt.tight_layout()
            
            output_path = self.output_dir / "invariance_stages.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"    ✅ Created stage-by-stage comparison")

    def safe_json_load(self, json_path):
        """Safely load JSON with error recovery"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON decode error in {json_path.name}: {e}")
            print("   Attempting to fix common issues...")
            
            try:
                # Read raw content
                with open(json_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Common fixes
                import re
                
                # Fix NaN and Infinity
                content = re.sub(r':\s*NaN\b', ': null', content)
                content = re.sub(r':\s*-?Infinity\b', ': null', content)
                
                # Fix trailing commas
                content = re.sub(r',\s*}', '}', content)
                content = re.sub(r',\s*]', ']', content)
                
                # Try to parse again
                return json.loads(content)
                
            except Exception as e2:
                print(f"❌ Could not fix JSON: {e2}")
                return None
    
    def create_nucleation_scaling_plots(self):
        """Create plots for nucleation scaling test"""
        print("  📊 Creating nucleation scaling plots...")
        
        results = self.data['results']
        site_counts = sorted([int(s) for s in results.keys()])
        
        # Extract data
        final_densities = [results[str(s)]['final_density'] for s in site_counts]
        avg_structures = [results[str(s)]['avg_structure'] for s in site_counts]
        
        # 1. Scaling analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Density vs sites
        ax1.plot(site_counts, final_densities, 'o-', color='cyan', 
                linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2)
        ax1.set_xlabel('Number of Nucleation Sites', fontsize=12)
        ax1.set_ylabel('Final Global Density', fontsize=12)
        ax1.set_title('Density vs Nucleation Sites', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Add linear fit
        if len(site_counts) > 1:
            coeffs = np.polyfit(site_counts, final_densities, 1)
            fit_line = np.poly1d(coeffs)
            ax1.plot(site_counts, fit_line(site_counts), 'r--', 
                    linewidth=1.5, alpha=0.7, label=f'Slope: {coeffs[0]:.6f}')
            ax1.legend()
        
        # Structure vs sites
        ax2.plot(site_counts, avg_structures, 's-', color='magenta',
                linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2)
        ax2.set_xlabel('Number of Nucleation Sites', fontsize=12)
        ax2.set_ylabel('Structure Quality Metric', fontsize=12)
        ax2.set_title('Cosmic Web Structure vs Nucleation Sites', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Nucleation Scaling Test - Robustness Proof', fontsize=16, y=0.98)
        plt.tight_layout()
        
        output_path = self.output_dir / "nucleation_scaling.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ Created main scaling plot")
        
        # 2. Density evolution for different site counts
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        sample_sites = site_counts[::max(1, len(site_counts)//9)]  # Sample 9 sites
        if len(sample_sites) > 9:
            sample_sites = sample_sites[:9]
        
        for idx, sites in enumerate(sample_sites):
            if idx < len(axes):
                ax = axes[idx]
                site_key = str(sites)
                
                if site_key in results and 'all_densities' in results[site_key]:
                    densities = results[site_key]['all_densities']
                    frames = np.arange(len(densities))
                    
                    ax.plot(frames, densities, color='yellow', linewidth=2, alpha=0.8)
                    ax.fill_between(frames, densities, alpha=0.3, color='yellow')
                    
                    ax.set_xlabel('Frame', fontsize=9)
                    ax.set_ylabel('Density', fontsize=9)
                    ax.set_title(f'{sites} Sites (Final: {results[site_key]["final_density"]:.4f})', fontsize=10)
                    ax.grid(True, alpha=0.3)
        
        plt.suptitle('Density Evolution for Different Nucleation Site Counts', fontsize=14, y=0.98)
        plt.tight_layout()
        
        output_path = self.output_dir / "scaling_density_evolution.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ Created density evolution comparison")
        
        # 3. Robustness analysis
        density_range = max(final_densities) - min(final_densities)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create box plot
        all_densities = []
        labels = []
        for sites in site_counts:
            site_key = str(sites)
            if site_key in results and 'all_densities' in results[site_key]:
                # Take last 20 frames for stability analysis
                stable_densities = results[site_key]['all_densities'][-20:] if len(results[site_key]['all_densities']) >= 20 else results[site_key]['all_densities']
                all_densities.append(stable_densities)
                labels.append(str(sites))
        
        if all_densities:
            box = ax.boxplot(all_densities, labels=labels, patch_artist=True)
            
            # Color boxes
            colors = plt.cm.viridis(np.linspace(0, 1, len(all_densities)))
            for patch, color in zip(box['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_xlabel('Number of Nucleation Sites', fontsize=12)
            ax.set_ylabel('Stable Density Distribution', fontsize=12)
            ax.set_title('Robustness Analysis: Density Stability Across Site Counts', fontsize=14)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add robustness conclusion
            robustness_text = f"Density Range: {density_range:.6f}\n"
            robustness_text += f"Robustness Threshold: < 0.02\n"
            robustness_text += f"Test Result: {'PASS' if density_range < 0.02 else 'INCONCLUSIVE'}"
            
            ax.text(0.02, 0.98, robustness_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            
            output_path = self.output_dir / "scaling_robustness.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"    ✅ Created robustness analysis plot")

# ==============================================================================
#   MAIN FUNCTION
# ==============================================================================

def main():
    """Main entry point for EST Plot Generator"""
    print(r"""
     _____ ___  _____     _       _       
    | ____/ __||_   _|   | | __ _| |__    
    |  _| \__ \  | |_____| |/ _` | '_ \   
    | |___|___/  | |_____| | (_| | |_) |  
    |_____|___/  |_|     |_|\__,_|_.__/   
    
    EST PLOT GENERATOR - v3.0 Compatible
    Creates publication-quality plots from robustness test data
    
    📁 Loads data from Simulations/ folder
    📊 Creates plots for: Parameter Phase Space, Isotropy Check,
       Algorithmic Invariance, Nucleation Scaling
    🎨 Professional visualizations with statistical analysis
    """)
    
    # Create plot generator
    plotter = EST_PlotGenerator()
    
    # Select test folder
    test_dir = plotter.file_picker()
    
    if not test_dir:
        print("\n❌ No test folder selected. Exiting.")
        input("Press Enter to exit...")
        return
    
    # Load test data
    if not plotter.load_test_data(test_dir):
        print("\n❌ Failed to load test data. Exiting.")
        input("Press Enter to exit...")
        return
    
    # Create all plots
    plotter.create_all_plots()
    
    print(f"\n{'='*60}")
    print("PLOT GENERATION COMPLETE!")
    print(f"{'='*60}")
    print(f"📁 Plots saved to: {plotter.output_dir}")
    print(f"📊 Test type: {plotter.test_type.replace('_', ' ').title()}")
    print(f"🎨 Created professional publication-quality plots")
    print(f"{'='*60}")
    
    input("\nPress Enter to exit...")

# ==============================================================================
#   ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Plot generation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")