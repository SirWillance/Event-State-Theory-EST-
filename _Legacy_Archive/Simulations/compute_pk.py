# compute_pk.py  –  Power spectrum analysis with file browser
import numpy as np
import matplotlib.pyplot as plt
import os
import traceback
import sys
import tkinter as tk
from tkinter import filedialog
import json
from datetime import datetime

# Global variables that will be initialized properly
data_dir = None
error_file = None

# ────────────────────── ERROR HANDLING SETUP ──────────────────────
def setup_error_logging():
    """Create error log directory with multiple fallbacks - GUARANTEED to work"""
    try:
        # Try multiple possible locations in order of preference
        import tempfile
        
        # 1. First try: Script directory (most organized)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "power_spectrum_data")
        
        print(f"Attempting to create data directory...")
        
        try:
            os.makedirs(data_dir, exist_ok=True)
            # Test if we can write
            test_file = os.path.join(data_dir, ".write_test")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print(f"✓ Using: {data_dir}")
            
        except (PermissionError, OSError) as e:
            print(f"  Could not write to script directory: {e}")
            
            # 2. Second try: User's desktop
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            data_dir = os.path.join(desktop, "power_spectrum_data")
            
            try:
                os.makedirs(data_dir, exist_ok=True)
                test_file = os.path.join(data_dir, ".write_test")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                print(f"✓ Using: {data_dir} (on Desktop)")
                
            except (PermissionError, OSError) as e2:
                print(f"  Could not write to Desktop: {e2}")
                
                # 3. Third try: System temp directory (ALWAYS writable)
                temp_dir = tempfile.gettempdir()
                data_dir = os.path.join(temp_dir, "power_spectrum_data")
                os.makedirs(data_dir, exist_ok=True)
                print(f"✓ Using: {data_dir} (in temp folder)")
        
        # Now create error log file
        error_log_path = os.path.join(data_dir, "error_log.txt")
        
        # Redirect stderr to file AND console
        class Tee:
            def __init__(self, *files):
                self.files = files
            def write(self, obj):
                for f in self.files:
                    f.write(obj)
            def flush(self):
                for f in self.files:
                    f.flush()
        
        # Open error log file
        error_file = open(error_log_path, 'w')
        
        # Tee stderr to both console and file
        sys.stderr = Tee(sys.stderr, error_file)
        
        print(f"Error logging initialized. Log file: {error_log_path}")
        return data_dir, error_file
        
    except Exception as e:
        print(f"CRITICAL: All directory attempts failed: {e}")
        print("Using current directory with console-only output")
        return ".", None  # Current directory, no error file

def log_error(error_msg, exception=None):
    """Log error message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"\n[{timestamp}] ERROR: {error_msg}\n"
    
    if exception:
        log_entry += f"Exception type: {type(exception).__name__}\n"
        log_entry += f"Exception message: {str(exception)}\n"
        log_entry += "Traceback:\n"
        log_entry += traceback.format_exc()
    
    print(log_entry)
    
    # Also write to error log file if it exists
    if error_file:
        error_file.write(log_entry)
        error_file.flush()

# ────────────────────── FILE SELECTION GUI ──────────────────────
def select_npy_file():
    """Open file dialog to select .npy file"""
    print("\nOpening file browser...")
    
    # Create hidden tkinter root window
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Set file dialog options
    filetypes = [
        ("NumPy files", "*.npy"),
        ("All files", "*.*")
    ]
    
    filepath = filedialog.askopenfilename(
        title="Select NPY field file for power spectrum analysis",
        initialdir=os.getcwd(),  # Start in current directory
        filetypes=filetypes
    )
    
    if filepath:
        print(f"✓ Selected: {filepath}")
        return filepath
    else:
        print("No file selected.")
        return None

def load_npy_file(filepath):
    """Load and validate NPY file"""
    try:
        print(f"Loading {os.path.basename(filepath)}...")
        field = np.load(filepath)
        
        print(f"✓ File loaded successfully")
        print(f"  Shape: {field.shape}")
        print(f"  Type: {field.dtype}")
        print(f"  Memory: {field.nbytes / (1024**2):.2f} MB")
        
        # Check if it's 3D
        if len(field.shape) != 3:
            print(f"⚠ Warning: Expected 3D field, got {len(field.shape)}D")
            print("  Trying to reshape...")
            
            # Try to reshape if it's 1D (flattened)
            if len(field.shape) == 1:
                n = int(round(field.size ** (1/3)))
                if n**3 == field.size:
                    field = field.reshape(n, n, n)
                    print(f"  Reshaped to {n}³")
                else:
                    raise ValueError(f"Cannot reshape 1D array of size {field.size} to cube")
            
            # If it's 2D, try to add third dimension
            elif len(field.shape) == 2:
                field = field.reshape(field.shape[0], field.shape[1], 1)
                print(f"  Reshaped to {field.shape[0]}x{field.shape[1]}x1")
        
        return field, filepath
        
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return None, None

# ────────────────────── POWER SPECTRUM ANALYSIS ──────────────────────
def compute_power_spectrum(field, source_file=None):
    """Compute power spectrum from 3D field"""
    print("\n" + "="*60)
    print("COMPUTING POWER SPECTRUM")
    print("="*60)
    
    print("Computing overdensity...")
    density = field.astype(float)
    mean_density = density.mean()
    delta = density / mean_density - 1.0
    
    print("Performing 3D FFT...")
    f = np.fft.fftn(delta)
    pk3d = np.abs(f)**2
    
    # Wave numbers
    nx, ny, nz = field.shape
    print(f"Grid dimensions: {nx} x {ny} x {nz}")
    
    knyq = nx // 2
    kx = np.fft.fftfreq(nx, d=1) * nx
    ky = np.fft.fftfreq(ny, d=1) * ny
    kz = np.fft.fftfreq(nz, d=1) * nz
    kgrid = np.meshgrid(kx, ky, kz, indexing='ij')
    k = np.sqrt(kgrid[0]**2 + kgrid[1]**2 + kgrid[2]**2)
    
    # Radial binning
    print("Binning power spectrum...")
    k = k.ravel()
    pk = pk3d.ravel()
    bins = np.logspace(0.01, np.log10(knyq), 35)
    k_centers = (bins[:-1] + bins[1:]) / 2       
    
    pk_binned = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (k >= lo) & (k < hi)
        if mask.any():  # Check if there are any points in this bin
            pk_binned.append(pk[mask].mean())
        else:
            pk_binned.append(np.nan)  # Use NaN for empty bins

    pk_binned = np.array(pk_binned)

    # Filter out NaN values
    valid_mask = ~np.isnan(pk_binned)
    k_centers = k_centers[valid_mask]
    pk_binned = pk_binned[valid_mask]
    
    # Fit spectral index
    print("Fitting spectral index...")
    mask = (k_centers > 8) & (k_centers < 50)
    if mask.sum() < 2:
        # Adjust range based on grid size
        low_k = max(2, nx // 64)  # Dynamic lower bound
        high_k = min(50, nx // 4)  # Dynamic upper bound
        mask = (k_centers > low_k) & (k_centers < high_k)
        print(f"  Adjusted fit range: k = [{low_k}, {high_k}]")
    
    if mask.sum() < 2:
        raise ValueError(f"Not enough points in fit range. Found {mask.sum()} points.")
    
    logk = np.log(k_centers[mask])
    logpk = np.log(pk_binned[mask])
    coeffs = np.polyfit(logk, logpk, 1)
    n_s = coeffs[0]
    
    print(f"\n" + "="*60)
    print(f"ANALYSIS RESULTS")
    print("="*60)
    print(f"Grid size: {nx}³")
    print(f"Mean density: {mean_density:.6f}")
    print(f"Spectral index n_s = {n_s:.4f}")
    print(f"   (Planck 2018: n_s = 0.9649 ± 0.0042)")
    print("="*60)
    
    return {
        'k_centers': k_centers,
        'pk_binned': pk_binned,
        'n_s': n_s,
        'bins': bins,
        'field_shape': field.shape,
        'mean_density': mean_density,
        'knyq': knyq,
        'nx': nx,
        'ny': ny,
        'nz': nz,
        'source_file': source_file
    }

# ────────────────────── SAVE RESULTS ──────────────────────
def save_results(results, data_dir):
    """Save analysis results to files"""
    print(f"\nSaving results to directory: '{data_dir}/'")
    
    # Extract results
    k_centers = results['k_centers']
    pk_binned = results['pk_binned']
    n_s = results['n_s']
    field_shape = results['field_shape']
    mean_density = results['mean_density']
    source_file = results['source_file']
    
    # Create timestamped subdirectory for better organization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_dir = os.path.join(data_dir, f"analysis_{timestamp}")
    os.makedirs(analysis_dir, exist_ok=True)
    
    print(f"Created analysis folder: {analysis_dir}")
    
    # Save as separate .npy files
    np.save(os.path.join(analysis_dir, "k_centers.npy"), k_centers)
    np.save(os.path.join(analysis_dir, "pk_binned.npy"), pk_binned)
    
    # Save as .npz (single compressed file)
    np.savez(os.path.join(analysis_dir, "power_spectrum.npz"), 
             k_centers=k_centers, 
             pk_binned=pk_binned,
             n_s=n_s,
             bins=results['bins'],
             field_shape=field_shape,
             mean_density=mean_density)
    
    # Save as plain text
    np.savetxt(os.path.join(analysis_dir, "power_spectrum.txt"),
               np.column_stack([k_centers, pk_binned]),
               header=f"Power Spectrum Data\nGrid: {field_shape}\nn_s = {n_s:.4f}\nk [grid units]\tP(k)",
               fmt='%.6e')
    
    # Save as CSV
    np.savetxt(os.path.join(analysis_dir, "power_spectrum.csv"),
               np.column_stack([k_centers, pk_binned]),
               header=f"k,P(k)\n# Grid:{field_shape}\n# n_s={n_s:.4f}",
               delimiter=',',
               fmt='%.6e',
               comments='')
    
    # Save metadata as JSON
    metadata = {
        'analysis': 'EST Power Spectrum',
        'grid_size': field_shape,
        'mean_density': float(mean_density),
        'spectral_index_ns': float(n_s),
        'source_file': source_file,
        'timestamp': datetime.now().isoformat(),
        'fit_range': [8, 50],
        'k_bins': len(k_centers),
        'nyquist_frequency': results['knyq']
    }
    
    with open(os.path.join(analysis_dir, "analysis_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save simple metadata text
    with open(os.path.join(analysis_dir, "metadata.txt"), 'w') as f:
        f.write(f"Power Spectrum Analysis\n")
        f.write(f"======================\n")
        f.write(f"Field dimensions: {field_shape}\n")
        f.write(f"Number of k-bins: {len(k_centers)}\n")
        f.write(f"Spectral index n_s: {n_s:.6f}\n")
        f.write(f"Fit range: k = [8, 50] grid units\n")
        f.write(f"Mean density: {mean_density:.6f}\n")
        f.write(f"Nyquist frequency: {results['knyq']}\n")
        if source_file:
            f.write(f"Source file: {source_file}\n")
        f.write(f"\nFile formats available:\n")
        f.write(f"  • power_spectrum.npz - all data (compressed)\n")
        f.write(f"  • k_centers.npy, pk_binned.npy - separate arrays\n")
        f.write(f"  • power_spectrum.txt - plain text\n")
        f.write(f"  • power_spectrum.csv - CSV format\n")
        f.write(f"  • analysis_metadata.json - JSON metadata\n")
    
    print(f"\n✓ Data saved in multiple formats in folder:")
    print(f"  {analysis_dir}")
    for file in ["k_centers.npy", "pk_binned.npy", "power_spectrum.npz", 
                 "power_spectrum.txt", "power_spectrum.csv", "metadata.txt", 
                 "analysis_metadata.json"]:
        print(f"    • {file}")
    
    return metadata, analysis_dir

# ────────────────────── CREATE PLOT ──────────────────────
def create_plot(results, analysis_dir):
    """Create and save power spectrum plot"""
    print("\nGenerating plot...")
    
    k_centers = results['k_centers']
    pk_binned = results['pk_binned']
    n_s = results['n_s']
    nx = results['nx']
    
    plt.figure(figsize=(10, 6), facecolor='black')
    plt.rcParams['text.color'] = 'white'
    plt.loglog(k_centers, pk_binned, 'o-', color='#00ffff', lw=2.5, 
               label=f'EST {nx}³ (n_s = {n_s:.3f})')
    plt.axvline(8, color='gray', alpha=0.4, ls='--', label='Fit range start')
    plt.axvline(50, color='gray', alpha=0.4, ls='--', label='Fit range end')
    plt.xlabel('k  [grid units]', fontsize=14, color='white')
    plt.ylabel('P(k)', fontsize=14, color='white')
    plt.title(f'EST Matter Power Spectrum ({nx}³ Grid)', color='white', fontsize=16)
    plt.grid(alpha=0.3, color='gray')
    plt.legend()
    plt.tight_layout()
    
    # Save in analysis directory
    plot_name = f"EST_Power_Spectrum_{nx}.png"
    plot_path = os.path.join(analysis_dir, plot_name)
    plt.savefig(plot_path, dpi=300, facecolor='black')
    
    # Also save in current directory for easy access
    current_plot_path = f"EST_Power_Spectrum_{nx}.png"
    plt.savefig(current_plot_path, dpi=300, facecolor='black')
    
    print(f"\n✓ PNG saved as: {plot_path}")
    print(f"✓ Also saved as: {current_plot_path} (in current directory)")
    
    plt.show()
    
    return plot_path

# ────────────────────── MAIN EXECUTION ──────────────────────
def main():
    global data_dir, error_file
    
    try:
        print("\n" + "="*60)
        print("POWER SPECTRUM ANALYSIS FOR EST SIMULATIONS")
        print("="*60)
        
        # Initialize error logging FIRST
        data_dir, error_file = setup_error_logging()
        
        # Show where data will be saved
        if data_dir:
            print(f"📁 Results will be saved to: {data_dir}")
        else:
            print("⚠ WARNING: Using current directory for results")
        
        print("This script analyzes 3D density fields from EST simulations")
        print("and computes the matter power spectrum with spectral index n_s")
        print("="*60)
        
        # Step 1: Select file using GUI
        filepath = select_npy_file()
        if not filepath:
            print("No file selected. Exiting.")
            return
        
        # Step 2: Load and validate file
        field, loaded_path = load_npy_file(filepath)
        if field is None:
            raise ValueError("Failed to load field file")
        
        # Step 3: Compute power spectrum
        results = compute_power_spectrum(field, source_file=loaded_path)
        
        # Step 4: Save results
        metadata, analysis_dir = save_results(results, data_dir)
        
        # Step 5: Create plot
        plot_path = create_plot(results, analysis_dir)
        
        # Step 6: Save success log
        with open(os.path.join(analysis_dir, "success.log"), 'w') as f:
            f.write(f"Analysis completed successfully\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Grid size: {results['field_shape']}\n")
            f.write(f"Spectral index n_s: {results['n_s']:.6f}\n")
            f.write(f"Mean density: {results['mean_density']:.6f}\n")
            f.write(f"Source file: {loaded_path}\n")
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print(f"✓ Power spectrum computed for {results['field_shape']} grid")
        print(f"✓ Spectral index: n_s = {results['n_s']:.4f}")
        print(f"✓ All results saved to: '{analysis_dir}/'")
        print(f"✓ Plot saved as: EST_Power_Spectrum_{results['nx']}.png")
        print("="*60)
        
        # Option to analyze another file
        print("\nOptions:")
        print("1. Analyze another file")
        print("2. Exit")
        
        choice = input("\nEnter choice (1-2): ").strip()
        if choice == '1':
            # Clear previous plot to avoid overlap
            plt.close('all')
            main()  # Restart
        else:
            print("Goodbye!")
        
    except Exception as e:
        log_error("Critical error during analysis", e)
        print("\n" + "="*60)
        print("ANALYSIS FAILED!")
        print(f"Error: {type(e).__name__}: {e}")
        print("\nCheck the error log in power_spectrum_data/error_log.txt")
        print("="*60)
        
        # Keep console open on Windows
        if sys.platform == 'win32':
            input("\nPress Enter to exit...")
        
        # Re-raise to exit with error code
        raise
    
    finally:
        # Close error log file
        if error_file:
            error_file.close()

# ────────────────────── EXECUTION ──────────────────────
if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
    except KeyboardInterrupt:
        print("\n\nAnalysis cancelled by user.")
        if sys.platform == 'win32':
            input("Press Enter to exit...")
    except Exception as e:
        # Final catch-all
        print(f"\nFatal error occurred: {e}")
        traceback.print_exc()
        if sys.platform == 'win32':
            input("Press Enter to exit...")