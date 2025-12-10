"""
EST_Visualizer3D.py - Visualization Tool for EST v3.0 Data
==========================================================
Loads .npz files from Simulations/ folder and creates 3D visualizations
No modifications to core engine - purely data visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import imageio.v2 as imageio
import os
import sys
import json
import csv
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs): 
        return iterable

# Set dark theme for all plots
plt.style.use('dark_background')

class EST_Visualizer3D:
    """
    3D Visualization tool for EST simulation data
    Loads .npz files and creates visualizations without touching engine
    """
    
    def __init__(self, data_file=None):
        """
        Initialize visualizer with optional data file
        If no file provided, will prompt user to select one
        """
        self.data = None
        self.config = None
        self.metadata = None
        self.final_state = None
        self.density_history = None
        self.output_dir = None
        self.visualizations_dir = None
        
        if data_file:
            self.load_data(data_file)
    
    def file_picker(self):
        """Open file picker to select EST .npz file"""
        try:
            import tkinter as tk
            from tkinter import filedialog
            
            root = tk.Tk()
            root.withdraw()
            
            print("\n📂 Please select your EST simulation_data.npz file...")
            
            # Start in Simulations folder if it exists
            initial_dir = Path.cwd() / "Simulations"
            if not initial_dir.exists():
                initial_dir = Path.cwd()
            
            file_path = filedialog.askopenfilename(
                title="Select EST Simulation Data (.npz)",
                initialdir=str(initial_dir),
                filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")]
            )
            
            if file_path:
                return Path(file_path)
            else:
                print("❌ No file selected.")
                return None
                
        except ImportError:
            # Fallback to manual input
            print("\n📂 Tkinter not available. Please enter path manually.")
            print("   Look in 'Simulations/' folder for .npz files")
            
            # List available .npz files
            sim_dir = Path.cwd() / "Simulations"
            if sim_dir.exists():
                npz_files = list(sim_dir.rglob("*.npz"))
                if npz_files:
                    print("\nAvailable .npz files:")
                    for i, f in enumerate(npz_files[:10]):  # Show first 10
                        rel_path = f.relative_to(Path.cwd())
                        print(f"  {i+1}. {rel_path}")
                    
                    choice = input(f"\nSelect file number (1-{len(npz_files)}): ").strip()
                    if choice.isdigit():
                        idx = int(choice) - 1
                        if 0 <= idx < len(npz_files):
                            return npz_files[idx]
            
            # Manual path input
            path_str = input("\nEnter path to simulation_data.npz: ").strip()
            if path_str:
                return Path(path_str)
            
            return None
    
   
    def load_data(self, data_file):
        """Load data from .npz or .npy file"""
        print(f"\n🔍 Loading data from: {data_file.name}")
        
        try:
            # Handle .npy files differently
            if data_file.suffix == '.npy':
                # .npy files don't use context manager
                self.final_state = np.load(data_file, allow_pickle=True)
                self.density_history = np.array([self.final_state.mean()])
                self.frames = np.array([0])
                
                # Set default config
                self.config = {
                    'experiment_name': data_file.stem,
                    'size': self.final_state.shape[0],
                    'frames': 1
                }
                
            elif data_file.suffix == '.npz':
                # .npz files use context manager
                with np.load(data_file, allow_pickle=True) as loaded:
                    self.final_state = loaded['final_state']
                    self.density_history = loaded['density_history']
                    
                    # Load config and metadata from strings
                    if 'config_str' in loaded:
                        config_str = str(loaded['config_str'][0])
                        self.config = json.loads(config_str)
                    
                    if 'metadata_str' in loaded:
                        metadata_str = str(loaded['metadata_str'][0])
                        self.metadata = json.loads(metadata_str)
                    
                    # Try to load frames if available
                    if 'frames' in loaded:
                        self.frames = loaded['frames']
                    else:
                        self.frames = np.arange(len(self.density_history))
            else:
                print(f"❌ Unsupported file type: {data_file.suffix}")
                return False
            
            # Set output directory
            self.output_dir = data_file.parent / "Visualizations"
            self.output_dir.mkdir(exist_ok=True)
            
            # Also create subdirectories
            (self.output_dir / "slices").mkdir(exist_ok=True)
            (self.output_dir / "projections").mkdir(exist_ok=True)
            (self.output_dir / "gif_frames").mkdir(exist_ok=True)
            
            print(f"✅ Data loaded successfully")
            print(f"   Grid size: {self.final_state.shape}")
            print(f"   Frames: {len(self.density_history)}")
            print(f"   Output directory: {self.output_dir.name}")
            
            if self.config:
                print(f"   Experiment: {self.config.get('experiment_name', 'Unknown')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_all_visualizations(self):
        """Create all 3D visualizations"""
        if self.data is None and self.final_state is None:
            print("❌ No data loaded. Please load a .npz file first.")
            return
        
        print(f"\n🎨 Creating 3D visualizations...")
        
        # 1. Create 3D scatter plot
        self.create_3d_scatter()
        
        # 2. Create slice visualizations
        self.create_slice_visualizations()
        
        # 3. Create projections
        self.create_projections()
        
        # 4. Create GIF animation (if enough data)
        self.create_3d_gif()
        
        # 5. Create density evolution plot
        self.create_density_plot()
        
        # 6. Create summary dashboard
        self.create_summary_dashboard()
        
        print(f"\n✅ All visualizations created!")
        print(f"📁 Output saved to: {self.output_dir}")
        
        # Try to open the folder
        try:
            os.startfile(self.output_dir)
        except:
            pass
    
    def create_3d_scatter(self):
        """Create 3D scatter plot of the universe"""
        print("  📊 Creating 3D scatter plot...")
        
        # Get active points (where value == 1)
        points = np.argwhere(self.final_state == 1)
        
        if len(points) == 0:
            print("    ⚠️ No active points found in final state")
            return
        
        # Limit points for performance
        max_points = 10000
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
            print(f"    Downsampled to {max_points} points for performance")
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create scatter plot
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                           c=points[:, 2],  # Color by Z coordinate
                           cmap='viridis',
                           s=1,  # Small points
                           alpha=0.6,
                           depthshade=True)
        
        # Configure plot
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set title
        if self.config:
            title = f"3D Universe: {self.config.get('experiment_name', 'EST Simulation')}"
        else:
            title = "3D Universe Visualization"
        
        ax.set_title(title, fontsize=14, pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Z Coordinate', rotation=270, labelpad=15)
        
        # Set equal aspect ratio
        max_range = np.array([points[:, 0].max()-points[:, 0].min(),
                             points[:, 1].max()-points[:, 1].min(),
                             points[:, 2].max()-points[:, 2].min()]).max() / 2.0
        
        mid_x = (points[:, 0].max()+points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max()+points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max()+points[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Save figure
        output_path = self.output_dir / "3d_universe_scatter.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ Saved: {output_path.name}")
        
        # Also create multiple view angles
        self.create_multiple_views(points)
    
    def create_multiple_views(self, points):
        """Create multiple 3D views from different angles"""
        angles = [
            (30, 45, "iso_view"),
            (90, 0, "top_view"),
            (0, 0, "front_view"),
            (0, 90, "side_view")
        ]
        
        for elev, azim, name in angles:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                               c=points[:, 2],
                               cmap='plasma',
                               s=0.5,
                               alpha=0.5)
            
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'{name.replace("_", " ").title()} - 3D Universe')
            
            # Hide axes for cleaner look
            ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
            ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
            ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
            
            output_path = self.output_dir / f"3d_{name}.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
    
    def create_slice_visualizations(self):
        """Create 2D slice visualizations through the 3D volume"""
        print("  🗺️ Creating slice visualizations...")
        
        size = self.final_state.shape[0]
        
        # Create slices at different depths
        slice_positions = [size//4, size//2, 3*size//4]
        slice_names = ['quarter', 'half', 'three_quarters']
        
        for axis in range(3):  # X, Y, Z axes
            axis_name = ['X', 'Y', 'Z'][axis]
            
            for pos, name in zip(slice_positions, slice_names):
                if axis == 0:
                    slice_data = self.final_state[pos, :, :]
                elif axis == 1:
                    slice_data = self.final_state[:, pos, :]
                else:  # axis == 2
                    slice_data = self.final_state[:, :, pos]
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Binary view
                im1 = ax1.imshow(slice_data, cmap='binary', origin='lower')
                ax1.set_title(f'{axis_name}-axis Slice at {name} (Binary)')
                ax1.set_xlabel(['Y', 'X', 'X'][axis])
                ax1.set_ylabel(['Z', 'Z', 'Y'][axis])
                plt.colorbar(im1, ax=ax1, label='State (0/1)')
                
                # Density gradient view
                from scipy.ndimage import gaussian_filter
                smoothed = gaussian_filter(slice_data.astype(float), sigma=1)
                im2 = ax2.imshow(smoothed, cmap='viridis', origin='lower')
                ax2.set_title(f'{axis_name}-axis Slice at {name} (Smoothed)')
                ax2.set_xlabel(['Y', 'X', 'X'][axis])
                ax2.set_ylabel(['Z', 'Z', 'Y'][axis])
                plt.colorbar(im2, ax=ax2, label='Density Gradient')
                
                plt.suptitle(f'Slice Analysis: {axis_name}-axis at {pos}', y=0.98)
                plt.tight_layout()
                
                output_path = self.output_dir / "slices" / f"slice_{axis_name}_{name}.png"
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
        
        print(f"    ✅ Created {3*3} slice visualizations")
    
    def create_projections(self):
        """Create 2D projections along each axis"""
        print("  📐 Creating axis projections...")
        
        # Projections (sum along each axis)
        projections = {
            'XY': self.final_state.sum(axis=2),  # Sum along Z
            'XZ': self.final_state.sum(axis=1),  # Sum along Y
            'YZ': self.final_state.sum(axis=0)   # Sum along X
        }
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, (name, proj) in enumerate(projections.items()):
            ax = axes[idx]
            
            # Use different colormaps for variety
            cmaps = ['hot', 'plasma', 'inferno']
            im = ax.imshow(proj, cmap=cmaps[idx], origin='lower')
            
            ax.set_title(f'{name} Projection (Deep Field)')
            ax.set_xlabel(name[1])  # Second letter
            ax.set_ylabel(name[0])  # First letter
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Integrated Density')
        
        plt.suptitle('Universe Projections Along Principal Axes', fontsize=14, y=0.98)
        plt.tight_layout()
        
        output_path = self.output_dir / "projections" / "all_projections.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also save individual projections
        for name, proj in projections.items():
            plt.figure(figsize=(8, 6))
            plt.imshow(proj, cmap='magma', origin='lower')
            plt.title(f'{name} Projection')
            plt.colorbar(label='Integrated Density')
            plt.tight_layout()
            
            output_path = self.output_dir / "projections" / f"projection_{name}.png"
            plt.savefig(output_path, dpi=150)
            plt.close()
        
        print(f"    ✅ Created 3 projection visualizations")
    
    def create_3d_gif(self):
        """Create rotating 3D GIF animation"""
        print("  🎬 Creating 3D rotation GIF...")
        
        # Check if we have enough points
        points = np.argwhere(self.final_state == 1)
        
        if len(points) < 100:
            print("    ⚠️ Not enough points for GIF (minimum 100)")
            return
        
        # Limit points for performance
        max_points = 5000
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
        
        # Create temporary directory for frames
        temp_dir = self.output_dir / "gif_frames"
        temp_dir.mkdir(exist_ok=True)
        
        frames = []
        angles = range(0, 360, 15)  # 24 frames
        
        if TQDM_AVAILABLE:
            angle_iter = tqdm(angles, desc="Rendering frames")
        else:
            angle_iter = angles
            print(f"    Rendering {len(angles)} frames...")
        
        for angle in angle_iter:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Create scatter plot
            scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                               c='cyan',
                               s=0.3,
                               alpha=0.4,
                               depthshade=True)
            
            # Set view angle
            ax.view_init(elev=20, azim=angle)
            
            # Configure plot
            ax.set_xlabel('X', fontsize=10)
            ax.set_ylabel('Y', fontsize=10)
            ax.set_zlabel('Z', fontsize=10)
            ax.set_title(f'Universe Rotation: {angle}°', fontsize=12, pad=20)
            
            # Set dark background
            ax.set_facecolor('black')
            ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
            ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
            ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
            
            # Remove grid for cleaner look
            ax.grid(False)
            
            # Set equal aspect
            max_range = np.array([points[:, 0].max()-points[:, 0].min(),
                                 points[:, 1].max()-points[:, 1].min(),
                                 points[:, 2].max()-points[:, 2].min()]).max() / 2.0
            
            mid_x = (points[:, 0].max()+points[:, 0].min()) * 0.5
            mid_y = (points[:, 1].max()+points[:, 1].min()) * 0.5
            mid_z = (points[:, 2].max()+points[:, 2].min()) * 0.5
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            # Save frame
            frame_path = temp_dir / f"frame_{angle:03d}.png"
            plt.savefig(frame_path, dpi=100, bbox_inches='tight', facecolor='black')
            plt.close()
            
            # Read frame for GIF
            frames.append(imageio.imread(frame_path))
        
        # Create GIF
        gif_path = self.output_dir / "universe_rotation.gif"
        imageio.mimsave(gif_path, frames, fps=15, loop=0)
        
        # Clean up temp files
        for frame_file in temp_dir.glob("*.png"):
            frame_file.unlink()
        temp_dir.rmdir()
        
        print(f"    ✅ GIF created: {gif_path.name} ({len(frames)} frames, 15 fps)")
    
    def create_density_plot(self):
        """Create density evolution plot"""
        if self.density_history is None or len(self.density_history) < 2:
            print("    ⚠️ Not enough density history data")
            return
        
        print("  📈 Creating density evolution plot...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Raw density
        ax1.plot(self.density_history, color='cyan', linewidth=2)
        ax1.fill_between(range(len(self.density_history)), 
                         self.density_history, 
                         alpha=0.3, color='cyan')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Density')
        ax1.set_title('Density Evolution Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = f"Initial: {self.density_history[0]:.4f}\n"
        stats_text += f"Final: {self.density_history[-1]:.4f}\n"
        stats_text += f"Change: {(self.density_history[-1]/self.density_history[0]-1)*100:+.1f}%\n"
        stats_text += f"Mean: {np.mean(self.density_history):.4f}"
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # Plot 2: Moving average
        window = min(20, len(self.density_history) // 10)
        if window > 1:
            moving_avg = np.convolve(self.density_history, 
                                     np.ones(window)/window, 
                                     mode='valid')
            
            ax2.plot(self.density_history, color='gray', alpha=0.5, label='Raw')
            ax2.plot(range(window-1, len(self.density_history)), 
                    moving_avg, color='magenta', linewidth=2, 
                    label=f'{window}-frame moving average')
            ax2.set_xlabel('Frame')
            ax2.set_ylabel('Density')
            ax2.set_title('Smoothed Density Evolution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Universe Density Analysis', fontsize=14, y=0.95)
        plt.tight_layout()
        
        output_path = self.output_dir / "density_evolution.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ Density plot saved")
    
    def create_summary_dashboard(self):
        """Create a summary dashboard with multiple visualizations"""
        print("  📊 Creating summary dashboard...")
        
        # Try to load existing images
        image_files = []
        for pattern in ["3d_universe_scatter.png", "density_evolution.png"]:
            for ext in ["png", "jpg"]:
                path = self.output_dir / pattern.replace("png", ext)
                if path.exists():
                    image_files.append(path)
                    break
        
        # Also add projection images
        proj_dir = self.output_dir / "projections"
        if proj_dir.exists():
            for proj_file in proj_dir.glob("*.png"):
                if proj_file.stat().st_size < 5e6:  # Skip files >5MB
                    image_files.append(proj_file)
        
        if len(image_files) >= 4:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            for i, ax in enumerate(axes):
                if i < len(image_files):
                    try:
                        img = plt.imread(image_files[i])
                        ax.imshow(img)
                        ax.set_title(image_files[i].stem.replace('_', ' ').title(), 
                                   fontsize=10)
                        ax.axis('off')
                    except Exception as e:
                        ax.text(0.5, 0.5, f"Could not load image", 
                               ha='center', va='center', transform=ax.transAxes)
                        ax.axis('off')
                else:
                    ax.axis('off')
            
            # Add title
            if self.config:
                title = f"EST Visualization Dashboard: {self.config.get('experiment_name', '')}"
            else:
                title = "EST Visualization Dashboard"
            
            plt.suptitle(title, fontsize=16, y=0.98)
            plt.tight_layout()
            
            output_path = self.output_dir / "visualization_dashboard.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"    ✅ Dashboard created with {min(4, len(image_files))} images")
    
    def create_html_report(self):
        """Create HTML report with all visualizations"""
        print("  📋 Creating HTML report...")
        
        html_path = self.output_dir / "visualization_report.html"
        
        # Collect all image files
        image_files = []
        for ext in ["png", "gif", "jpg"]:
            image_files.extend(list(self.output_dir.glob(f"*.{ext}")))
        
        # Also include subdirectories
        for subdir in ["slices", "projections"]:
            subdir_path = self.output_dir / subdir
            if subdir_path.exists():
                image_files.extend(list(subdir_path.glob("*.png")))
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>EST 3D Visualization Report</title>
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
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
        }
        .card {
            background-color: #1a1a1a;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            transition: transform 0.3s;
        }
        .card:hover {
            transform: translateY(-5px);
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
        .info-box {
            background-color: #2d2d2d;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🌌 EST 3D Visualization Report</h1>
""")
            
            if self.config:
                f.write(f"<h2>{self.config.get('experiment_name', 'EST Simulation')}</h2>")
            
            f.write(f"""        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="info-box">
        <h2>📊 Simulation Information</h2>
""")
            
            if self.config:
                f.write("<ul>")
                f.write(f"<li><strong>Grid Size:</strong> {self.final_state.shape[0]}³</li>")
                f.write(f"<li><strong>Frames:</strong> {len(self.density_history)}</li>")
                f.write(f"<li><strong>Beta (β):</strong> {self.config.get('beta', 'N/A')}</li>")
                f.write(f"<li><strong>Lambda (λ):</strong> {self.config.get('lambda_t', 'N/A')}</li>")
                f.write(f"<li><strong>Final Density:</strong> {self.density_history[-1]:.6f}</li>")
                f.write("</ul>")
            
            f.write("""
    </div>
    
    <h2>📈 Visualizations</h2>
    <div class="grid">
""")
            
            # Add all images to HTML
            for img_file in sorted(image_files):
                rel_path = img_file.relative_to(self.output_dir.parent)
                title = img_file.stem.replace('_', ' ').title()
                
                f.write(f"""
        <div class="card">
            <div class="card-title">{title}</div>
            <img src="{rel_path}" alt="{title}">
        </div>
""")
            
            f.write("""
    </div>
    
    <div style="margin-top: 40px; padding: 20px; background-color: #1a237e; border-radius: 10px;">
        <h3>🔍 About This Report</h3>
        <p>This report was automatically generated by EST_Visualizer3D.py</p>
        <p>All visualizations are created from the raw .npz data file without modifying the simulation engine.</p>
        <p>To regenerate visualizations, run: <code>python EST_Visualizer3D.py</code></p>
    </div>
</body>
</html>
""")
        
        print(f"    ✅ HTML report created: {html_path.name}")

# ==============================================================================
#   MAIN FUNCTION
# ==============================================================================

def main():
    """Main entry point for EST Visualizer"""
    print(r"""
     _____ ___  _____     _       _       
    | ____/ __||_   _|   | | __ _| |__    
    |  _| \__ \  | |_____| |/ _` | '_ \   
    | |___|___/  | |_____| | (_| | |_) |  
    |_____|___/  |_|     |_|\__,_|_.__/   
    
    EST VISUALIZER 3D - v3.0 Compatible
    Creates 3D visualizations from EST simulation data
    
    📁 Looks for .npz files in Simulations/ folder
    🎨 Creates: 3D plots, slices, projections, GIFs, HTML reports
    ⚠️  No engine modifications - pure data visualization
    """)
    
    # Create visualizer instance
    visualizer = EST_Visualizer3D()
    
    # File picker
    data_file = visualizer.file_picker()
    
    if not data_file:
        print("\n❌ No data file selected. Exiting.")
        input("Press Enter to exit...")
        return
    
    # Load data
    if not visualizer.load_data(data_file):
        print("\n❌ Failed to load data. Exiting.")
        input("Press Enter to exit...")
        return
    
    # Create all visualizations
    visualizer.create_all_visualizations()
    
    # Create HTML report
    visualizer.create_html_report()
    
    print(f"\n{'='*60}")
    print("VISUALIZATION COMPLETE!")
    print(f"{'='*60}")
    print(f"📁 All outputs saved to: {visualizer.output_dir}")
    print(f"🌐 HTML report: visualization_report.html")
    print(f"🎬 GIF animation: universe_rotation.gif")
    print(f"📊 3D plots, slices, and projections created")
    print(f"{'='*60}")
    
    input("\nPress Enter to exit...")

# ==============================================================================
#   ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Visualization interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")