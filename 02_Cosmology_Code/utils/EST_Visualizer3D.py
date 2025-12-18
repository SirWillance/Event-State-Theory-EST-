"""
EST_Visualizer3D.py - Visualization Tool for EST v3.1 Data
==========================================================
Loads .npz or legacy .npy files and creates comprehensive 3D visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import imageio.v2 as imageio
import os
import sys
import json
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

    def tqdm(iterable, **kwargs):
        return iterable

# Global style
plt.style.use('dark_background')


class EST_Visualizer3D:
    def __init__(self, data_file=None):
        self.data = None
        self.config = {}
        self.metadata = {}
        self.final_state = None
        self.density_history = None
        self.frames = None
        self.output_dir = None

        if data_file:
            self.load_data(data_file)

    def file_picker(self):
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            print("\n📂 Please select your EST simulation data file...")
            initial_dir = Path.cwd() / "Simulations"
            if not initial_dir.exists():
                initial_dir = Path.cwd()
            file_path = filedialog.askopenfilename(
                title="Select EST Simulation Data",
                initialdir=str(initial_dir),
                filetypes=[
                    ("NPZ files", "*.npz"),
                    ("NPY files", "*.npy"),
                    ("All files", "*.*"),
                ],
            )
            return Path(file_path) if file_path else None
        except ImportError:
            print("\n📂 Tkinter not available - manual input")
            path_str = input(
                "Enter full path to simulation_data.npz or .npy file: "
            ).strip()
            return Path(path_str) if path_str else None

    def load_data(self, data_file: Path):
        print(f"\n🔍 Loading data from: {data_file.name}")
        try:
            if data_file.suffix == ".npy":
                arr = np.load(data_file, allow_pickle=True)

                if arr.ndim == 3:
                    # 3D snapshot: treat as final universe state
                    self.final_state = arr
                    self.density_history = np.array([self.final_state.mean()])
                    self.frames = np.array([0])
                    self.config = {
                        "experiment_name": data_file.stem,
                        "size": self.final_state.shape[0],
                        "frames": 1,
                    }

                elif arr.ndim == 1:
                    # 1D timeseries: no 3D grid available
                    self.final_state = None
                    self.density_history = arr
                    self.frames = np.arange(len(arr))
                    self.config = {
                        "experiment_name": data_file.stem,
                        "size": "timeseries",
                        "frames": len(arr),
                    }
                    print("   ℹ️ Detected 1D timeseries (.npy) – 3D visuals will be skipped.")

                else:
                    print(f"❌ Unsupported .npy shape: {arr.shape}")
                    return False


            elif data_file.suffix == ".npz":
                with np.load(data_file, allow_pickle=True) as loaded:
                    self.final_state = loaded["final_state"]
                    self.density_history = loaded["density_history"]
                    self.frames = loaded.get(
                        "frames",
                        np.arange(len(self.density_history)),
                    )

                    # Robust config loading
                    if "config_str" in loaded:
                        try:
                            self.config = json.loads(
                                str(loaded["config_str"][0])
                            )
                        except Exception as e:
                            print(f"    ⚠️ Config parse error: {e}")
                            self.config = {}

                    if "metadata_str" in loaded:
                        try:
                            meta = json.loads(
                                str(loaded["metadata_str"][0])
                            )
                            self.metadata.update(meta)
                        except Exception as e:
                            print(f"    ⚠️ Metadata parse error: {e}")

            else:
                print(f"❌ Unsupported file type: {data_file.suffix}")
                return False

            # Output folders
            self.output_dir = data_file.parent / "Visualizations"
            self.output_dir.mkdir(exist_ok=True)
            for sub in ["slices", "projections"]:
                (self.output_dir / sub).mkdir(exist_ok=True)

            print("✅ Data loaded successfully")
            if self.final_state is not None and self.final_state.ndim == 3:
                print(f"   Grid size: {self.final_state.shape}")
            else:
                print("   Grid size: N/A (no 3D grid)")
            print(f"   Frames: {len(self.density_history) if self.density_history is not None else 0}")

            print(
                f"   Experiment: {self.config.get('experiment_name', 'Unknown')}"
            )
            if self.config.get("physical_scaling"):
                print(
                    f"   Physical box: "
                    f"{self.config.get('box_size_mpc', 500)} Mpc/h"
                )
            return True

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            import traceback

            traceback.print_exc()
            return False

    # ------------------------------------------------------
    # Visual creators
    # ------------------------------------------------------

    def create_3d_scatter(self):
        if self.final_state is None or self.final_state.ndim != 3:
            print("    ⚠️ No 3D grid available – skipping this visualization.")
            return

        print("  📊 Creating 3D scatter plot...")
        points = np.argwhere(self.final_state == 1)
        if len(points) == 0:
            print("    ⚠️ No active cells")
            return

        max_points = 15000
        if len(points) > max_points:
            idx = np.random.choice(len(points), max_points, replace=False)
            points = points[idx]

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=points[:, 2],
            cmap="viridis",
            s=1.5,
            alpha=0.8,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Universe Structure")
        plt.colorbar(scatter, shrink=0.6, label="Z coordinate")
        path = self.output_dir / "3d_universe_scatter.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    ✅ Saved {path.name}")

    def create_slice_visualizations(self):
        if self.final_state is None or self.final_state.ndim != 3:
            print("    ⚠️ No 3D grid available – skipping this visualization.")
            return

        print("  📏 Creating 2D slices...")
        for axis, name in zip([0, 1, 2], ["X", "Y", "Z"]):
            mid = self.final_state.shape[axis] // 2
            if axis == 0:
                slice_2d = self.final_state[mid, :, :]
            elif axis == 1:
                slice_2d = self.final_state[:, mid, :]
            else:
                slice_2d = self.final_state[:, :, mid]

            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(slice_2d.T, cmap="hot", origin="lower")
            ax.set_title(f"Middle Slice along {name}-axis")
            plt.colorbar(im, ax=ax, label="Density")
            path = self.output_dir / "slices" / f"slice_{name}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"    ✅ {path.name}")

    def create_projections(self):
        if self.final_state is None or self.final_state.ndim != 3:
            print("    ⚠️ No 3D grid available – skipping this visualization.")
            return

        print("  📉 Creating projections...")
        for axis, name in zip([0, 1, 2], ["X", "Y", "Z"]):
            proj = self.final_state.sum(axis=axis)
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(proj.T, cmap="plasma", origin="lower")
            ax.set_title(f"Projection along {name}-axis")
            plt.colorbar(im, ax=ax, label="Integrated Density")
            path = self.output_dir / "projections" / f"projection_{name}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"    ✅ {path.name}")

    def create_3d_gif(self):
        if self.final_state is None or self.final_state.ndim != 3:
            print("    ⚠️ No 3D grid available – skipping this visualization.")
            return

        print("  🎬 Creating rotating GIF (this may take a while)...")
        points = np.argwhere(self.final_state == 1)
        if len(points) == 0:
            print("    ⚠️ No active cells - skipping GIF")
            return

        max_points = 12000
        if len(points) > max_points:
            idx = np.random.choice(len(points), max_points, replace=False)
            points = points[idx]
            print(f"    Downsampled to {max_points} points for GIF")

        gif_path = self.output_dir / "universe_rotation.gif"
        frames = []

        # Force non-interactive backend for GIF generation
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

        azims = range(0, 360, 4)
        iterator = tqdm(azims, desc="Rendering frames") if TQDM_AVAILABLE else azims

        for azim in iterator:
            fig = plt.figure(figsize=(10, 8), dpi=100)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111, projection="3d")

            ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                c=points[:, 2],
                cmap="viridis",
                s=1.2,
                alpha=0.8,
                depthshade=False,
            )

            ax.view_init(elev=20, azim=azim)
            ax.axis("off")
            fig.tight_layout(pad=0)

            canvas.draw()
            buf = np.asarray(canvas.buffer_rgba())
            image = buf[:, :, :3]  # RGB
            frames.append(image)
            plt.close(fig)

        imageio.mimsave(gif_path, frames, duration=0.1, loop=0)
        print(f"    ✅ Rotating GIF saved: {gif_path.name}")

    def create_summary_dashboard(self):
        print("  📊 Creating summary dashboard...")
        image_files = []

        # Prioritize main overview graphics
        for name in ["3d_universe_scatter.png", "density_evolution.png"]:
            path = self.output_dir / name
            if path.exists():
                image_files.append(path)
        for proj in sorted((self.output_dir / "projections").glob("*.png")):
            image_files.append(proj)

        if len(image_files) >= 4:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            for i, ax in enumerate(axes):
                if i < len(image_files):
                    img = plt.imread(image_files[i])
                    ax.imshow(img)
                    ax.set_title(
                        image_files[i]
                        .stem.replace("_", " ")
                        .title()
                    )
                    ax.axis("off")
                else:
                    ax.axis("off")
            plt.suptitle(
                f"EST Dashboard: {self.config.get('experiment_name', 'Simulation')}",
                fontsize=16,
            )
            plt.tight_layout()
            path = self.output_dir / "visualization_dashboard.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"    ✅ Dashboard created")

    # ------------------------------------------------------
    # HTML report
    # ------------------------------------------------------

    def create_html_report(self):
        print("  📋 Creating HTML report...")
        html_path = self.output_dir / "visualization_report.html"

        # Collect images that live inside self.output_dir (or subdirs)
        image_files = []
        for ext in ["png", "gif"]:
            image_files.extend(self.output_dir.glob(f"*.{ext}"))
        for sub in ["slices", "projections"]:
            image_files.extend((self.output_dir / sub).glob("*.png"))

        experiment_name = self.config.get("experiment_name", "EST Simulation")
        final_density = float(self.final_state.mean()) if self.final_state is not None else 0.0
        frames_count = self.config.get("frames", 1)

        # NEW:
        if self.final_state is not None and self.final_state.ndim == 3:
            grid_size = self.final_state.shape[0]
        else:
            grid_size = "N/A"


        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>EST 3D Visualization Report</title>
    <style>
        body {{font-family: Arial, sans-serif; background:#0a0a0a; color:#fff; margin:0; padding:20px;}}
        .header {{text-align:center; padding:30px; background:linear-gradient(45deg,#1a237e,#4a148c); border-radius:10px; margin-bottom:30px;}}
        .info {{background:#2d2d2d; padding:20px; border-radius:10px; margin-bottom:30px;}}
        .grid {{display:grid; grid-template-columns:repeat(auto-fit,minmax(480px,1fr)); gap:20px;}}
        .card {{background:#1a1a1a; border-radius:10px; padding:15px; box-shadow:0 4px 8px rgba(0,0,0,0.5);}}
        .card img {{width:100%; border-radius:8px;}}
        .card h3 {{color:#4fc3f7; margin:10px 0; font-size:18px;}}
    </style>
</head>
<body>
    <div class="header">
        <h1>🌌 EST 3D Visualization Report</h1>
        <h2>{experiment_name}</h2>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="info">
        <h2>📊 Simulation Information</h2>
        <ul>
            <li><strong>Grid Size:</strong> {grid_size}³</li>
            <li><strong>Frames:</strong> {frames_count}</li>
            <li><strong>Final Density:</strong> {final_density:.6f}</li>"""

        if self.config.get("physical_scaling"):
            html_content += f"""
            <li><strong>Box Size:</strong> {self.config.get('box_size_mpc', 500)} Mpc/h</li>
            <li><strong>H0:</strong> {self.config.get('H0', 67.4)} km/s/Mpc</li>
            <li><strong>Ω_m:</strong> {self.config.get('Ω_m', 0.315)}</li>"""

        html_content += """
        </ul>
    </div>

    <h2>🎨 Visualizations</h2>
    <div class="grid">"""

        # IMPORTANT: paths relative to the HTML file (output_dir)
        for img in sorted(image_files):
            rel_path = img.relative_to(self.output_dir).as_posix()
            title = img.stem.replace("_", " ").title()
            html_content += f"""
        <div class="card">
            <h3>{title}</h3>
            <img src="{rel_path}" alt="{title}">
        </div>"""

        html_content += """
    </div>

    <div style="margin-top:50px; padding:20px; background:#1a237e; border-radius:10px; text-align:center;">
        <p>Generated by EST_Visualizer3D.py • v3.1 compatible</p>
    </div>
</body>
</html>"""

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"    ✅ HTML report created: {html_path.name}")

    # ------------------------------------------------------

    def create_all_visualizations(self):
        if self.final_state is None:
            print("ℹ️ No 3D grid available – skipping 3D visualizations.")
            return

        # Full 3D mode
        self.create_3d_scatter()
        self.create_slice_visualizations()
        self.create_projections()
        self.create_3d_gif()
        self.create_summary_dashboard()




def main():
    print(
        r"""
 _____ ___  _____     _       _       
| ____/ __||_   _|   | | __ _| |__    
|  _| \__ \  | |_____| |/ _` | '_ \   
| |___|___/  | |_____| | (_| | |_) |  
|_____|___/  |_|     |_|\__,_|_.__/   

EST VISUALIZER 3D - v3.1 Compatible
"""
    )

    visualizer = EST_Visualizer3D()
    data_file = visualizer.file_picker()
    if not data_file:
        print("❌ No file selected")
        input("Press Enter to exit...")
        return
    if not visualizer.load_data(data_file):
        input("Press Enter to exit...")
        return

    visualizer.create_all_visualizations()
    visualizer.create_html_report()

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE!")
    print("=" * 60)
    print(f"📁 All files saved to: {visualizer.output_dir}")
    print("🌐 Open visualization_report.html in your browser")
    print("=" * 60)
    input("\nPress Enter to exit...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback

        traceback.print_exc()
        input("\nPress Enter to exit...")
