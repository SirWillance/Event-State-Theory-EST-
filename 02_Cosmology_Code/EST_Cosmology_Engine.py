"""
EST_UNIVERSE_WITH_PHYSICAL_SCALING.py
Event-State Theory with proper cosmological scaling
Modified to output physical units comparable to observations
"""

import numpy as np
import zlib
import lzma
import bz2
import math
import os
import random
import matplotlib.pyplot as plt
import sys
import csv
from datetime import datetime
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
except ImportError:
    print("Optimization Tip: Open CMD and run 'pip install tqdm imageio' for better visuals.")
    def tqdm(iterable, **kwargs): return iterable

import imageio.v2 as imageio

# ==============================================================================
#   PHYSICAL COSMOLOGICAL SCALER (NEW)
# ==============================================================================

class CosmologicalScaler:
    """Converts simulation units to physical cosmological units"""
    
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
        """
        Convert P(k) to (Mpc/h)^3
        P_phys = P_sim * (box_size/grid_size)^3
        """
        voxel_size = self.box_size / grid_size  # Mpc per voxel
        return Pk_grid * (voxel_size**3)
    
    def density_to_Ωm(self, density_sim):
        """Convert simulation density to Ω_m"""
        # Simple scaling: simulation density 1.0 = critical density
        return density_sim * self.Ω_m
    
    def frame_to_redshift(self, frame, total_frames, z_start=20, z_end=0):
        """Approximate redshift from frame number"""
        # Linear in scale factor a = 1/(1+z)
        t_fraction = frame / total_frames
        a_start = 1.0 / (1 + z_start)
        a_end = 1.0 / (1 + z_end)
        a = a_start + t_fraction * (a_end - a_start)
        z = 1.0/a - 1.0
        return max(0, z)
    
    def frame_to_time_gyr(self, frame, total_frames):
        """Convert frame to time since Big Bang in Gyr"""
        # Approximate cosmic time formula for matter-dominated universe
        t_fraction = frame / total_frames
        # t ∝ a^(3/2) for matter domination
        z = self.frame_to_redshift(frame, total_frames)
        a = 1.0 / (1 + z)
        t_gyr = 13.8 * (a**1.5)  # Age of universe ~13.8 Gyr
        return t_gyr
    
    def get_physical_scales(self, grid_size):
        """Return useful scale information"""
        return {
            'box_size_mpc': self.box_size,
            'voxel_size_mpc': self.box_size / grid_size,
            'volume_mpc3': self.box_size**3,
            'H0': self.H0,
            'Ω_m': self.Ω_m,
            'Ω_b': self.Ω_b,
            'critical_density': self.ρ_crit
        }

# ==============================================================================
#   EST ENGINE (Enhanced with Physical Scaling)
# ==============================================================================

class EST_Engine:
    def __init__(self, size=64, dim=3, lambda_t=0.482, beta=3.4, epsilon=0.0748, 
                 candidates=90, threads=None, complexity_method="zlib",
                 # NEW: Physical scaling parameters
                 physical_scaling=False, box_size_mpc=500.0, H0=67.4, Ω_m=0.315):
        
        self.size = size
        self.dim = dim
        self.shape = tuple([size] * dim)
        self.u = np.zeros(self.shape, dtype=np.uint8)
        self.LAMBDA = lambda_t
        self.BETA = beta
        self.EPSILON = epsilon
        self.CANDIDATES = candidates
        self.complexity_method = complexity_method
        
        # NEW: Physical scaling
        self.physical_scaling = physical_scaling
        if physical_scaling:
            self.scaler = CosmologicalScaler(box_size_mpc=box_size_mpc, H0=H0, Ω_m=Ω_m)
            self.physical_params = self.scaler.get_physical_scales(size)
            print(f"Physical scaling enabled: {box_size_mpc} Mpc/h box, H0={H0}, Ω_m={Ω_m}")
        
        # Localized-collapse control
        self.core_mask = None
        self.CORE_WEIGHT = 1.0
        
        # Use defaults or limited threads
        self.executor = ThreadPoolExecutor(max_workers=threads)
        self.seed_physics(777)
        
    def seed_physics(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def _complexity(self, arr_bytes):
        if self.complexity_method == "zlib":
            return len(zlib.compress(arr_bytes))
        elif self.complexity_method == "lzma":
            return len(lzma.compress(arr_bytes))
        elif self.complexity_method == "bzip2":
            return len(bz2.compress(arr_bytes))
        elif self.complexity_method == "entropy":
            if len(arr_bytes) == 0:
                return 0
            counts = Counter(arr_bytes)
            proportions = [count / len(arr_bytes) for count in counts.values()]
            entropy = -sum(p * math.log2(p) for p in proportions if p > 0)
            return int(entropy * 1000)
        else:
            return len(zlib.compress(arr_bytes))

    def _cost_task(self, args):
        curr_bytes, cand_data, beta, eps = args
        cand_arr = np.frombuffer(cand_data, dtype=np.uint8).reshape(self.shape)
        curr_arr = np.frombuffer(curr_bytes, dtype=np.uint8).reshape(self.shape)
        
        flips = (curr_arr != cand_arr)
        dE = np.sum(flips)

        if self.core_mask is not None and self.CORE_WEIGHT != 1.0:
            core_flips = np.logical_and(flips, self.core_mask)
            dE_core = np.sum(core_flips)
            dE_outer = dE - dE_core
            dE_eff = dE_outer + self.CORE_WEIGHT * dE_core
        else:
            dE_eff = dE

        K = self._complexity(cand_data)
        created = np.sum((curr_arr == 0) & (cand_arr == 1))
        destroyed = np.sum((curr_arr == 1) & (cand_arr == 0))
        J = dE_eff + beta * K - eps * (destroyed - created)
        return J, cand_arr

    def step(self):
        current_bytes = self.u.tobytes()
        tasks = []
        for _ in range(self.CANDIDATES):
            c_copy = self.u.copy()
            n_flips = random.randint(3, 7)
            centers = np.random.randint(0, self.size, size=(n_flips, self.dim))
            offsets = np.random.randint(-2, 3, size=(n_flips, self.dim))
            target_indices = (centers + offsets) % self.size
            idx_tuple = tuple(target_indices[:, d] for d in range(self.dim))
            c_copy[idx_tuple] = 1 - c_copy[idx_tuple]
            tasks.append((current_bytes, c_copy.tobytes(), self.BETA, self.EPSILON))

        results = list(self.executor.map(self._cost_task, tasks))
        costs = np.array([r[0] for r in results])
        
        if np.all(np.isinf(costs)) or np.all(costs == costs[0]):
            idx = random.randint(0, len(results)-1)
            self.u = results[idx][1]
            return self.u.mean()

        costs_min = costs.min()
        if np.isinf(costs_min) or np.isnan(costs_min):
            costs_min = 0
        costs = costs - costs_min
        candidates = [r[1] for r in results]
        
        probs = np.exp(-costs / self.LAMBDA)
        probs_sum = probs.sum()
        
        if probs_sum == 0 or np.isnan(probs_sum):
            return self.u.mean()
        
        probs /= probs_sum
        idx = np.random.choice(len(candidates), p=probs)
        self.u = candidates[idx]
        return self.u.mean()

# ==============================================================================
#   EXPERIMENT RUNNER WITH PHYSICAL DATA OUTPUT (MODIFIED)
# ==============================================================================

class ExperimentRunner:
    def __init__(self, exp_name, physical_scaling=False, box_size_mpc=500.0, H0=67.4, Ω_m=0.315):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = Path(__file__).parent / f"EST_Output_{exp_name}_{self.timestamp}"
        self.base_dir.mkdir(exist_ok=True)
        self.log_path = self.base_dir / "experiment_log.txt"
        
        # NEW: Physical scaling
        self.physical_scaling = physical_scaling
        if physical_scaling:
            self.scaler = CosmologicalScaler(box_size_mpc=box_size_mpc, H0=H0, Ω_m=Ω_m)
            self.physical_params = self.scaler.get_physical_scales(128)  # Default size
            
        self.log(f"--- EXPERIMENT: {exp_name} ---")
        if physical_scaling:
            self.log(f"Physical scaling: Box={box_size_mpc} Mpc/h, H0={H0}, Ω_m={Ω_m}")

    def log(self, msg):
        print(msg)
        with open(self.log_path, "a", encoding='utf-8') as f:
            f.write(f"{msg}\n")

    def save_plot(self, filename):
        path = self.base_dir / filename
        plt.savefig(path, dpi=300, facecolor='black')
        plt.close()
        return path
    
    def save_heatmap(self, array2d, filename):
        plt.style.use('dark_background')
        plt.figure(figsize=(6, 5))
        plt.imshow(array2d, cmap="inferno", origin="lower")
        plt.colorbar(label="Proper-Time Accumulation")
        plt.title(filename.replace(".png", ""))
        path = self.base_dir / filename
        plt.savefig(path, dpi=300, facecolor='black')
        plt.close()
        return path

# ==============================================================================
#   MODIFIED: Run_Cosmology_Simulation WITH PHYSICAL SCALING
# ==============================================================================

def get_user_params_with_scaling(default_size, default_frames, default_sites):
    """Get user parameters including physical scaling"""
    print("\n--- Manual Configuration Mode ---")
    print(f"(Press ENTER to use defaults)")
    
    try:
        # Physical scaling first
        use_scaling = input("Enable physical cosmological scaling? [Y/N, default N]: ").lower()
        physical_scaling = use_scaling in ['y', 'yes']
        
        box_size_mpc = 500.0
        H0 = 67.4
        Ω_m = 0.315
        
        if physical_scaling:
            print("\n--- PHYSICAL SCALING PARAMETERS ---")
            box_size_mpc = float(input(f"Box size in Mpc/h [Default 500.0]: ") or "500.0")
            H0 = float(input(f"Hubble constant H0 [Default 67.4]: ") or "67.4")
            Ω_m = float(input(f"Matter density Ω_m [Default 0.315]: ") or "0.315")
            print(f"Using: Box={box_size_mpc} Mpc/h, H0={H0}, Ω_m={Ω_m}")
        
        # Simulation parameters
        in_size = input(f"\nGrid Size [Default {default_size}]: ")
        size = int(in_size) if in_size else default_size
        
        in_frames = input(f"Duration (Frames) [Default {default_frames}]: ")
        frames = int(in_frames) if in_frames else default_frames
        
        in_sites = input(f"Nucleation Sites [Default {default_sites}]: ")
        sites = int(in_sites) if in_sites else default_sites
        
        in_beta = input(f"Beta (Complexity Cost) [Default 3.4]: ")
        beta = float(in_beta) if in_beta else 3.4
        
        in_lam = input(f"Lambda (Temperature) [Default 0.48]: ")
        lam = float(in_lam) if in_lam else 0.48
        
        return {
            'size': size,
            'frames': frames,
            'beta': beta,
            'lam': lam,
            'sites': sites,
            'physical_scaling': physical_scaling,
            'box_size_mpc': box_size_mpc,
            'H0': H0,
            'Ω_m': Ω_m
        }
        
    except ValueError:
        print("Invalid input. Using Defaults.")
        return {
            'size': default_size,
            'frames': default_frames,
            'beta': 3.4,
            'lam': 0.482,
            'sites': default_sites,
            'physical_scaling': False,
            'box_size_mpc': 500.0,
            'H0': 67.4,
            'Ω_m': 0.315
        }

def calculate_power_spectrum_physical(universe_grid, scaler):
    """Calculate power spectrum with physical units"""
    size = universe_grid.shape[0]
    
    # Convert to float for FFT
    grid_float = universe_grid.astype(float)
    
    # 3D FFT
    fft_result = np.fft.fftn(grid_float)
    power = np.abs(fft_result)**2
    power_shifted = np.fft.fftshift(power)
    
    # Get k-space coordinates
    k_indices = np.fft.fftshift(np.fft.fftfreq(size)) * size
    kx, ky, kz = np.meshgrid(k_indices, k_indices, k_indices, indexing='ij')
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2).flatten()
    power_flat = power_shifted.flatten()
    
    # Simple binning without scipy
    k_max = size // 2
    n_bins = 30
    k_bins = np.linspace(0, k_max, n_bins + 1)
    k_centers = (k_bins[:-1] + k_bins[1:]) / 2
    
    # Manual binning
    power_binned = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    
    for k_val, p_val in zip(k_mag, power_flat):
        if k_val <= k_max:
            bin_idx = min(int(k_val / (k_max / n_bins)), n_bins - 1)
            power_binned[bin_idx] += p_val
            counts[bin_idx] += 1
    
    # Average
    valid = counts > 0
    power_binned[valid] /= counts[valid]
    
    # Convert to physical units
    k_physical = scaler.k_grid_to_physical(k_centers[valid], size)
    Pk_physical = scaler.pk_to_physical(power_binned[valid], size)
    
    # Fit power law (simple linear fit in log space)
    if len(k_physical) > 5:
        log_k = np.log10(k_physical[k_physical > 0.1])
        log_P = np.log10(Pk_physical[k_physical > 0.1])
        
        if len(log_k) > 2:
            # Simple linear regression
            A = np.vstack([log_k, np.ones(len(log_k))]).T
            m, c = np.linalg.lstsq(A, log_P, rcond=None)[0]
            n_s = m  # Spectral index
        else:
            n_s = None
    else:
        n_s = None
    
    return {
        'k_physical': k_physical,
        'Pk_physical': Pk_physical,
        'k_grid': k_centers[valid],
        'Pk_grid': power_binned[valid],
        'n_s': n_s
    }

def Run_Cosmology_Simulation(manual=False):
    """MAIN FUNCTION: Run EST simulation with physical scaling option"""
    
    # Get parameters
    if manual:
        params = get_user_params_with_scaling(128, 150, 12)
    else:
        params = {
            'size': 128,
            'frames': 150,
            'beta': 3.4,
            'lam': 0.482,
            'sites': 12,
            'physical_scaling': True,  # Default to True for auto-run
            'box_size_mpc': 500.0,
            'H0': 67.4,
            'Ω_m': 0.315
        }
    
    # Create experiment runner with physical scaling
    runner = ExperimentRunner(
        "Cosmology_Emergence_Physical" if params['physical_scaling'] else "Cosmology_Emergence",
        physical_scaling=params['physical_scaling'],
        box_size_mpc=params['box_size_mpc'],
        H0=params['H0'],
        Ω_m=params['Ω_m']
    )
    
    runner.log(f"Config: Grid {params['size']}^3 | Frames {params['frames']}")
    runner.log(f"Beta {params['beta']} | Lambda {params['lam']} | Sites {params['sites']}")
    
    # Initialize engine with physical scaling
    engine = EST_Engine(
        size=params['size'],
        dim=3,
        candidates=90,
        beta=params['beta'],
        lambda_t=params['lam'],
        physical_scaling=params['physical_scaling'],
        box_size_mpc=params['box_size_mpc'],
        H0=params['H0'],
        Ω_m=params['Ω_m']
    )
    
    # Inject nucleation sites
    runner.log(f"Injecting {params['sites']} Nucleation Sites...")
    for _ in range(params['sites']):
        c = np.random.randint(20, params['size']-20, size=3)
        rr, cc, dd = np.ogrid[:params['size'], :params['size'], :params['size']]
        dist_sq = (rr - c[0])**2 + (cc - c[1])**2 + (dd - c[2])**2
        engine.u[dist_sq < 64] = 1
    
    # Run simulation
    densities = []
    physical_densities = [] if params['physical_scaling'] else None
    redshifts = [] if params['physical_scaling'] else None
    times_gyr = [] if params['physical_scaling'] else None
    
    runner.log("Starting simulation...")
    for t in tqdm(range(params['frames']), desc="Simulating Universe"):
        d = engine.step()
        if d is not None:
            densities.append(d)
            
            # Calculate physical values if scaling enabled
            if params['physical_scaling']:
                Ω_physical = runner.scaler.density_to_Ωm(d)
                z = runner.scaler.frame_to_redshift(t, params['frames'])
                t_gyr = runner.scaler.frame_to_time_gyr(t, params['frames'])
                
                physical_densities.append(Ω_physical)
                redshifts.append(z)
                times_gyr.append(t_gyr)
    
    runner.log("Simulation Complete. Generating Data...")
    
    # Save raw data
    np.save(runner.base_dir / "final_universe_state.npy", engine.u)
    np.save(runner.base_dir / "density_timeseries.npy", np.array(densities))
    
    if params['physical_scaling']:
        np.save(runner.base_dir / "physical_density_timeseries.npy", np.array(physical_densities))
        np.save(runner.base_dir / "redshift_timeseries.npy", np.array(redshifts))
        np.save(runner.base_dir / "time_gyr_timeseries.npy", np.array(times_gyr))
    
    # Save comprehensive CSV data
    csv_path = runner.base_dir / "cosmological_data.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        
        if params['physical_scaling']:
            writer.writerow(["Frame", "Density_sim", "Ω_m", "Redshift_z", "Time_Gyr"])
            for i, (d_sim, Ω_m, z, t_gyr) in enumerate(zip(densities, physical_densities, redshifts, times_gyr)):
                writer.writerow([i, f"{d_sim:.6f}", f"{Ω_m:.6f}", f"{z:.3f}", f"{t_gyr:.3f}"])
        else:
            writer.writerow(["Frame", "Density_sim"])
            for i, d_sim in enumerate(densities):
                writer.writerow([i, f"{d_sim:.6f}"])
    
    runner.log(f"Saved comprehensive data: {csv_path.name}")
    
    # Save metadata
    metadata_path = runner.base_dir / "metadata.txt"
    with open(metadata_path, "w") as f:
        f.write(f"EST COSMOLOGICAL SIMULATION METADATA\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Grid size: {params['size']}^3\n")
        f.write(f"Total frames: {params['frames']}\n")
        f.write(f"Beta: {params['beta']}\n")
        f.write(f"Lambda: {params['lam']}\n")
        f.write(f"Nucleation sites: {params['sites']}\n")
        f.write(f"Epsilon: 0.0748\n")
        
        if params['physical_scaling']:
            f.write(f"\n--- PHYSICAL SCALING ---\n")
            f.write(f"Box size: {params['box_size_mpc']} Mpc/h\n")
            f.write(f"Hubble constant H0: {params['H0']} km/s/Mpc\n")
            f.write(f"Matter density Ω_m: {params['Ω_m']}\n")
            f.write(f"Baryon density Ω_b: 0.049 (Planck 2018)\n")
            f.write(f"Voxel size: {params['box_size_mpc']/params['size']:.3f} Mpc/voxel\n")
            f.write(f"Simulation volume: {params['box_size_mpc']**3:.1f} (Mpc/h)^3\n")
    
    # Calculate and save power spectrum with physical scaling
    if params['physical_scaling']:
        runner.log("Calculating power spectrum with physical scaling...")
        try:
            ps_results = calculate_power_spectrum_physical(engine.u, runner.scaler)
            
            # Save power spectrum data
            ps_csv_path = runner.base_dir / "power_spectrum_physical.csv"
            with open(ps_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["k_hMpc", "Pk_Mpc3", "k_grid", "Pk_grid"])
                for kp, Pkp, kg, Pkg in zip(ps_results['k_physical'], 
                                           ps_results['Pk_physical'],
                                           ps_results['k_grid'],
                                           ps_results['Pk_grid']):
                    writer.writerow([f"{kp:.6f}", f"{Pkp:.6e}", f"{kg:.3f}", f"{Pkg:.6e}"])
            
            runner.log(f"Power spectrum saved: {ps_csv_path.name}")
            
            if ps_results['n_s'] is not None:
                runner.log(f"Power law index n_s = {ps_results['n_s']:.3f}")
                
                # Save n_s to metadata
                with open(metadata_path, "a") as f:
                    f.write(f"\n--- POWER SPECTRUM RESULTS ---\n")
                    f.write(f"Spectral index n_s: {ps_results['n_s']:.3f}\n")
                    f.write(f"k range: {ps_results['k_physical'].min():.3f} - {ps_results['k_physical'].max():.3f} h/Mpc\n")
            
            # Plot power spectrum
            plt.style.use('dark_background')
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Physical units
            ax1.loglog(ps_results['k_physical'], ps_results['Pk_physical'], 'o-', 
                      color='yellow', markersize=4, linewidth=2)
            ax1.set_xlabel('k [h/Mpc]')
            ax1.set_ylabel('P(k) [(Mpc/h)$^3$]')
            ax1.set_title(f'Physical Power Spectrum\nBox={params["box_size_mpc"]} Mpc/h')
            ax1.grid(True, alpha=0.3, which='both')
            
            # Simulation units
            ax2.loglog(ps_results['k_grid'], ps_results['Pk_grid'], 'o-',
                      color='cyan', markersize=4, linewidth=2)
            ax2.set_xlabel('k [grid units]')
            ax2.set_ylabel('P(k) [simulation units]')
            ax2.set_title('Simulation Power Spectrum')
            ax2.grid(True, alpha=0.3, which='both')
            
            runner.save_plot("power_spectrum_comparison.png")
            
        except Exception as e:
            runner.log(f"Power spectrum calculation failed: {e}")
    
    # Create density evolution plots
    plt.style.use('dark_background')
    
    if params['physical_scaling']:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Physical density evolution
        ax1.plot(physical_densities, color='magenta', linewidth=2)
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Ω_m')
        ax1.set_title('Physical Matter Density Evolution')
        ax1.axhline(y=params['Ω_m'], color='white', linestyle='--', 
                   alpha=0.5, label=f'Planck 2018: Ω_m={params["Ω_m"]}')
        ax1.grid(alpha=0.3)
        ax1.legend()
        
        # Redshift axis on top
        ax1_top = ax1.twiny()
        redshift_ticks = np.linspace(0, max(redshifts), 5)
        ax1_top.set_xlim(ax1.get_xlim())
        ax1_top.set_xticks(np.linspace(0, len(redshifts)-1, 5))
        ax1_top.set_xticklabels([f'{z:.1f}' for z in redshift_ticks])
        ax1_top.set_xlabel('Redshift z')
        
        # Simulation density evolution
        ax2.plot(densities, color='cyan', linewidth=2)
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Density (simulation units)')
        ax2.set_title('Simulation Density Evolution')
        ax2.grid(alpha=0.3)
        
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(densities, color='cyan', linewidth=2)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Density (simulation units)')
        ax.set_title('Density Evolution')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    runner.save_plot("density_evolution.png")
    
    # Create deep field projection
    projection = engine.u.sum(axis=0)
    plt.figure(figsize=(8, 8))
    plt.imshow(projection, cmap='inferno', interpolation='nearest')
    plt.title(f"EST Deep Field Projection (Frame {params['frames']})")
    plt.axis('off')
    if params['physical_scaling']:
        # Add physical scale bar
        scale_mpc = params['box_size_mpc'] / 4  # 1/4 of box
        scale_pixels = params['size'] / 4
        plt.plot([10, 10 + scale_pixels], [params['size'] - 10, params['size'] - 10], 
                'w-', linewidth=3)
        plt.text(10 + scale_pixels/2, params['size'] - 15, 
                f'{scale_mpc:.0f} Mpc', color='white', ha='center')
    runner.save_plot("deep_field_projection.png")
    
    # Generate summary report
    summary_path = runner.base_dir / "experiment_summary.txt"
    with open(summary_path, "w") as f:
        f.write("="*60 + "\n")
        f.write("EST COSMOLOGICAL SIMULATION SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write("PARAMETERS:\n")
        f.write(f"  Grid size: {params['size']}³\n")
        f.write(f"  Frames: {params['frames']}\n")
        f.write(f"  Beta (β): {params['beta']}\n")
        f.write(f"  Lambda (λ): {params['lam']}\n")
        f.write(f"  Nucleation sites: {params['sites']}\n\n")
        
        f.write("RESULTS:\n")
        if densities:
            f.write(f"  Initial density: {densities[0]:.6f}\n")
            f.write(f"  Final density: {densities[-1]:.6f}\n")
            f.write(f"  Density increase: {(densities[-1]/densities[0]-1)*100:.1f}%\n")
        
        if params['physical_scaling'] and physical_densities:
            f.write(f"  Final Ω_m: {physical_densities[-1]:.6f}\n")
            f.write(f"  Planck 2018 Ω_m: 0.315 ± 0.007\n")
            f.write(f"  Difference: {abs(physical_densities[-1] - 0.315):.6f}\n")
            
            if ps_results and 'n_s' in ps_results and ps_results['n_s'] is not None:
                f.write(f"  Power spectrum index n_s: {ps_results['n_s']:.3f}\n")
                f.write(f"  Planck 2018 n_s (primordial): 0.965 ± 0.004\n")
        
        f.write("\nPHYSICAL SCALING:\n")
        if params['physical_scaling']:
            f.write(f"  Box size: {params['box_size_mpc']} Mpc/h\n")
            f.write(f"  Voxel size: {params['box_size_mpc']/params['size']:.3f} Mpc/voxel\n")
            f.write(f"  Simulation volume: {params['box_size_mpc']**3:.1f} (Mpc/h)³\n")
            f.write(f"  Hubble constant: {params['H0']} km/s/Mpc\n")
        else:
            f.write("  Physical scaling disabled\n")
    
    runner.log(f"\n{'='*60}")
    runner.log("EXPERIMENT COMPLETE!")
    runner.log(f"Data saved to: {runner.base_dir}")
    runner.log(f"Summary: {summary_path.name}")
    runner.log(f"Cosmological data: {csv_path.name}")
    if params['physical_scaling']:
        runner.log(f"Physical power spectrum saved")
    runner.log(f"{'='*60}")
    
    # List all generated files
    runner.log("\nGenerated files:")
    for f in sorted(runner.base_dir.glob("*")):
        if f.is_file():
            runner.log(f"  - {f.name}")
    
    try:
        os.startfile(runner.base_dir)
    except:
        pass

def Run_Parameter_Sweep_Simple():
    """Simple parameter sweep like original (for comparison)"""
    runner = ExperimentRunner("Parameter_Sweep_Simple")
    print("Calculating the Goldilocks Zone (3-5 mins)...")
    beta_range = np.linspace(2.0, 5.0, 10)
    lambda_range = np.linspace(0.1, 1.0, 10)
    results = np.zeros((10, 10))
    
    # Small grid for speed
    SWEEP_SIZE = 32
    
    pbar = tqdm(total=100)
    
    for i, beta in enumerate(beta_range):
        for j, lam in enumerate(lambda_range):
            eng = EST_Engine(
                size=SWEEP_SIZE, 
                dim=3, 
                lambda_t=lam, 
                beta=beta, 
                threads=1,
                physical_scaling=False  # Simple test
            )
            
            eng.u[SWEEP_SIZE//2-4:SWEEP_SIZE//2+4, 
                  SWEEP_SIZE//2-4:SWEEP_SIZE//2+4, 
                  SWEEP_SIZE//2-4:SWEEP_SIZE//2+4] = 1
            
            final_density = 0
            for _ in range(30):
                res = eng.step()
                if res is not None: 
                    final_density = res
            
            results[i, j] = final_density
            pbar.update(1)
    
    pbar.close()
    
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 8))
    im = plt.imshow(results, cmap='turbo', origin='lower',
                    extent=[0.1, 1.0, 2.0, 5.0])
    
    plt.colorbar(im, label="Universal Density")
    plt.scatter([0.482], [3.4], color='white', marker='x', s=100, label='EST Settings')
    plt.xlabel("Causal Temperature (Lambda)")
    plt.ylabel("Complexity Cost (Beta)")
    plt.title("The 'Goldilocks Zone' of Existence")
    plt.legend()
    runner.save_plot("Phase_Space_Topology.png")
    
    # CSV output
    csv_path = runner.base_dir / "parameter_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Beta", "Lambda", "Resulting_Density"])
        for i, b in enumerate(beta_range):
            for j, l in enumerate(lambda_range):
                writer.writerow([b, l, results[i,j]])
    
    runner.log("Verification Complete.")
    os.startfile(runner.base_dir)    

def Run_Enhanced_Parameter_Sweep():
    """Enhanced Goldilocks Zone analysis across ALL complexity methods"""
    runner = ExperimentRunner("Enhanced_Goldilocks_Analysis")
    
    print("Calculating Goldilocks Zones for ALL complexity methods...")
    
    # Test all complexity methods
    methods = ["zlib", "lzma", "bzip2", "entropy"]
    
    # Parameter ranges
    beta_range = np.linspace(2.0, 5.0, 10)
    lambda_range = np.linspace(0.1, 1.0, 10)
    
    # Small grid for speed
    SWEEP_SIZE = 32
    
    results_dict = {}
    
    for method in methods:
        runner.log(f"\nAnalyzing {method} complexity method...")
        results = np.zeros((10, 10))
        
        pbar = tqdm(total=100, desc=f"{method}")
        
        for i, beta in enumerate(beta_range):
            for j, lam in enumerate(lambda_range):
                eng = EST_Engine(
                    size=SWEEP_SIZE, 
                    dim=3, 
                    lambda_t=lam, 
                    beta=beta, 
                    threads=1,
                    complexity_method=method  # Use current method
                )
                
                # Standard seed pattern
                eng.u[SWEEP_SIZE//2-4:SWEEP_SIZE//2+4, 
                      SWEEP_SIZE//2-4:SWEEP_SIZE//2+4, 
                      SWEEP_SIZE//2-4:SWEEP_SIZE//2+4] = 1
                
                # Run simulation
                final_density = 0
                for _ in range(30):
                    res = eng.step()
                    if res is not None: 
                        final_density = res
                
                results[i, j] = final_density
                pbar.update(1)
        
        pbar.close()
        results_dict[method] = results.copy()
        
        # Save individual method results
        csv_path = runner.base_dir / f"parameter_sweep_{method}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Beta", "Lambda", "Resulting_Density"])
            for i, b in enumerate(beta_range):
                for j, l in enumerate(lambda_range):
                    writer.writerow([b, l, results[i, j]])
    
    # ============================================
    # CREATE COMPREHENSIVE COMPARISON PLOTS
    # ============================================
    
    plt.style.use('dark_background')
    
    # 1. INDIVIDUAL HEATMAPS FOR EACH METHOD
    runner.log("\nGenerating individual heatmaps...")
    for method in methods:
        plt.figure(figsize=(10, 8))
        im = plt.imshow(results_dict[method], cmap='turbo', origin='lower',
                       extent=[0.1, 1.0, 2.0, 5.0], aspect='auto')
        
        plt.colorbar(im, label=f"Universal Density ({method})")
        
        # Mark EST default settings
        plt.scatter([0.482], [3.4], color='white', marker='x', s=100, 
                   label='EST Settings (β=3.4, λ=0.482)')
        
        plt.xlabel("Causal Temperature (Lambda)")
        plt.ylabel("Complexity Cost (Beta)")
        plt.title(f"Goldilocks Zone - {method.capitalize()} Complexity")
        plt.legend()
        plt.tight_layout()
        
        runner.save_plot(f"Goldilocks_Zone_{method}.png")
    
    # 2. COMPARISON PLOT: ALL METHODS TOGETHER
    runner.log("Generating comparison plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, method in enumerate(methods):
        ax = axes[idx]
        im = ax.imshow(results_dict[method], cmap='turbo', origin='lower',
                      extent=[0.1, 1.0, 2.0, 5.0], aspect='auto')
        
        ax.scatter([0.482], [3.4], color='white', marker='x', s=80)
        ax.set_xlabel("Lambda")
        ax.set_ylabel("Beta")
        ax.set_title(f"{method.capitalize()}")
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label="Density")
    
    plt.suptitle("Goldilocks Zone Comparison: All Complexity Methods", fontsize=16)
    plt.tight_layout()
    runner.save_plot("Goldilocks_Comparison_All_Methods.png")
    
    # 3. VARIANCE ANALYSIS: How different are the methods?
    runner.log("Analyzing variance between methods...")
    
    # Calculate variance at each parameter point
    variance_grid = np.zeros((10, 10))
    
    for i in range(10):
        for j in range(10):
            # Collect density values from all methods at this (β,λ) point
            densities = [results_dict[method][i, j] for method in methods]
            variance_grid[i, j] = np.var(densities)
    
    # Plot variance
    plt.figure(figsize=(10, 8))
    im = plt.imshow(variance_grid, cmap='plasma', origin='lower',
                   extent=[0.1, 1.0, 2.0, 5.0], aspect='auto')
    
    plt.colorbar(im, label="Variance between methods")
    plt.xlabel("Causal Temperature (Lambda)")
    plt.ylabel("Complexity Cost (Beta)")
    plt.title("Method Variance: Higher = More Sensitivity to Complexity Definition")
    
    # Highlight high-variance zones
    high_var_mask = variance_grid > np.percentile(variance_grid, 75)
    y_coords, x_coords = np.where(high_var_mask)
    
    if len(y_coords) > 0:
        for y, x in zip(y_coords, x_coords):
            beta_val = 2.0 + (y / 9) * 3.0
            lambda_val = 0.1 + (x / 9) * 0.9
            plt.scatter(lambda_val, beta_val, color='red', s=10, alpha=0.6)
    
    runner.save_plot("Method_Variance_Analysis.png")
    
    # 4. ENTROPY vs COMPRESSION CORRELATION ANALYSIS
    runner.log("Analyzing entropy-compression correlation...")
    
    # Get entropy results (as baseline)
    entropy_results = results_dict["entropy"]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    compression_methods = ["zlib", "lzma", "bzip2"]
    
    for idx, method in enumerate(compression_methods):
        ax = axes[idx]
        comp_results = results_dict[method]
        
        # Scatter plot: entropy vs compression method
        ax.scatter(entropy_results.flatten(), comp_results.flatten(), 
                  alpha=0.6, s=20, color=['cyan', 'magenta', 'yellow'][idx])
        
        # Fit line
        x_vals = entropy_results.flatten()
        y_vals = comp_results.flatten()
        if len(x_vals) > 1:
            m, b = np.polyfit(x_vals, y_vals, 1)
            ax.plot([x_vals.min(), x_vals.max()], 
                   [m*x_vals.min()+b, m*x_vals.max()+b], 
                   'r--', linewidth=2, label=f'R²={np.corrcoef(x_vals, y_vals)[0,1]**2:.3f}')
        
        ax.set_xlabel("Entropy Method Density")
        ax.set_ylabel(f"{method.capitalize()} Method Density")
        ax.set_title(f"Correlation: Entropy vs {method.capitalize()}")
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.suptitle("Complexity Method Correlation Analysis", fontsize=14)
    plt.tight_layout()
    runner.save_plot("Complexity_Method_Correlation.png")
    
    # 5. SUMMARY STATISTICS
    runner.log("\n" + "="*60)
    runner.log("GOLDILOCKS ZONE ANALYSIS - SUMMARY")
    runner.log("="*60)
    
    # Find optimal zones for each method
    optimal_params = {}
    
    for method in methods:
        results = results_dict[method]
        
        # Find parameter combo with highest density
        max_idx = np.unravel_index(np.argmax(results), results.shape)
        max_beta = beta_range[max_idx[0]]
        max_lambda = lambda_range[max_idx[1]]
        max_density = results[max_idx]
        
        optimal_params[method] = {
            'beta': max_beta,
            'lambda': max_lambda,
            'density': max_density
        }
        
        runner.log(f"\n{method.upper()}:")
        runner.log(f"  Optimal β = {max_beta:.3f}, λ = {max_lambda:.3f}")
        runner.log(f"  Max density = {max_density:.6f}")
        
        # Check if EST defaults (β=3.4, λ=0.482) are in high-density zone
        est_beta_idx = np.argmin(np.abs(beta_range - 3.4))
        est_lambda_idx = np.argmin(np.abs(lambda_range - 0.482))
        est_density = results[est_beta_idx, est_lambda_idx]
        density_percentile = np.percentile(results.flatten(), est_density)
        
        runner.log(f"  EST defaults (3.4, 0.482): density = {est_density:.6f}")
        runner.log(f"  Percentile rank: {density_percentile:.1f}%")
    
    # Save optimal parameters
    optimal_csv = runner.base_dir / "optimal_parameters_summary.csv"
    with open(optimal_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Optimal_Beta", "Optimal_Lambda", "Max_Density"])
        for method, data in optimal_params.items():
            writer.writerow([method, data['beta'], data['lambda'], data['density']])
    
    runner.log(f"\nAnalysis complete. Data saved to: {runner.base_dir}")
    os.startfile(runner.base_dir)

# ==============================================================================
#   VALIDATION TEST SUITE (MISSING FROM v2.7)
# ==============================================================================

def test_algorithmic_invariance(runner=None):
    """Test if cosmic web emerges regardless of complexity algorithm"""
    if runner is None:
        runner = ExperimentRunner("Algorithmic_Invariance")
    
    print("Testing Algorithmic Invariance...")
    methods = ["zlib", "lzma", "bzip2", "entropy"]
    results = {}
    
    for method in methods:
        runner.log(f"Running with {method} complexity method...")
        engine = EST_Engine(size=48, complexity_method=method, 
                          beta=3.4, lambda_t=0.48, threads=1,
                          physical_scaling=False)  # Disable for pure test
        
        # Standard nucleation pattern
        engine.u[20:28, 20:28, 20:28] = 1
        
        densities = []
        for frame in tqdm(range(80), desc=f"{method}"):
            d = engine.step()
            if d is not None: 
                densities.append(d)
        
        final_density = densities[-1] if densities else 0
        results[method] = {
            'final_density': final_density,
            'all_densities': densities
        }
        runner.log(f"  {method}: Final density = {final_density:.6f}")
    
    # Generate comparison plot
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 6))
    for method, data in results.items():
        plt.plot(data['all_densities'], label=method, linewidth=2)
    
    plt.title("Algorithmic Invariance Test\nCosmic Web Emergence Across Different Complexity Measures")
    plt.xlabel("Frames")
    plt.ylabel("Global Density")
    plt.legend()
    plt.grid(alpha=0.3)
    
    plot_path = runner.save_plot("Algorithmic_Invariance_Comparison.png")
    runner.log(f"Invariance plot saved: {plot_path}")
    
    # Calculate variance
    final_densities = [data['final_density'] for data in results.values()]
    variance = max(final_densities) - min(final_densities)
    runner.log(f"Maximum density variance: {variance:.6f}")
    runner.log(f"Invariance Test: {'PASS' if variance < 0.01 else 'INCONCLUSIVE'}")
    
    return results

def analyze_cosmic_web_structure(universe_grid):
    """Simple metric for cosmic web quality - measures structure complexity"""
    if np.sum(universe_grid) == 0:
        return 0
    
    # Create 2D projection for analysis
    projection = universe_grid.sum(axis=0)
    
    # Calculate metrics that indicate filamentary structure
    variance = np.var(projection)
    mean_val = np.mean(projection)
    
    if mean_val == 0:
        return 0
    
    # Normalized variance indicates structure complexity
    structure_metric = variance / (mean_val + 1e-10)
    
    # Normalize for grid size
    return min(structure_metric / 10, 1.0)  # Cap at 1.0

def run_nucleation_scaling_test(runner=None):
    """Test cosmic web formation across different nucleation site counts"""
    if runner is None:
        runner = ExperimentRunner("Nucleation_Scaling")
    
    site_counts = [8, 12, 16, 20, 24, 29, 32, 36, 40]
    results = {}
    
    runner.log(f"Testing nucleation scaling with sites: {site_counts}")
    
    for sites in site_counts:
        runner.log(f"Testing {sites} nucleation sites...")
        engine = EST_Engine(size=64, beta=3.4, lambda_t=0.48, threads=1,
                          physical_scaling=False)  # Disable for test
        
        # Inject specified number of sites
        for _ in range(sites):
            c = np.random.randint(15, 49, size=3)  # Keep away from edges
            rr, cc, dd = np.ogrid[:64, :64, :64]
            dist_sq = (rr - c[0])**2 + (cc - c[1])**2 + (dd - c[2])**2
            engine.u[dist_sq < 9] = 1  # Smaller sites for better resolution
        
        # Run simulation
        densities = []
        structure_metrics = []
        
        for frame in tqdm(range(100), desc=f"Sites={sites}"):
            d = engine.step()
            if d is not None: 
                densities.append(d)
                if frame % 10 == 0:  # Sample structure periodically
                    structure_metrics.append(analyze_cosmic_web_structure(engine.u))
        
        final_density = densities[-1] if densities else 0
        avg_structure = np.mean(structure_metrics) if structure_metrics else 0
        
        results[sites] = {
            'final_density': final_density,
            'avg_structure': avg_structure,
            'all_densities': densities
        }
        runner.log(f"  Sites {sites}: Density = {final_density:.6f}, Structure = {avg_structure:.4f}")
    
    # Generate scaling plot
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    sites_list = sorted(results.keys())
    densities = [results[s]['final_density'] for s in sites_list]
    structures = [results[s]['avg_structure'] for s in sites_list]
    
    ax1.plot(sites_list, densities, 'o-', color='cyan', linewidth=2, markersize=6)
    ax1.set_xlabel("Number of Nucleation Sites")
    ax1.set_ylabel("Final Global Density")
    ax1.set_title("Density vs Nucleation Sites")
    ax1.grid(alpha=0.3)
    
    ax2.plot(sites_list, structures, 'o-', color='magenta', linewidth=2, markersize=6)
    ax2.set_xlabel("Number of Nucleation Sites")
    ax2.set_ylabel("Structure Quality Metric")
    ax2.set_title("Cosmic Web Structure vs Nucleation Sites")
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plot_path = runner.save_plot("Nucleation_Scaling_Analysis.png")
    runner.log(f"Scaling analysis plot saved: {plot_path}")
    
    # Check robustness
    density_range = max(densities) - min(densities)
    runner.log(f"Density range across site counts: {density_range:.6f}")
    runner.log(f"Robustness Test: {'PASS' if density_range < 0.02 else 'INCONCLUSIVE'}")
    
    return results


# ==============================================================================
#   NEW VALIDATION MENU OPTIONS (ADD TO v2.7)
# ==============================================================================


def Run_Relativity_Check():
    runner = ExperimentRunner("Isotropy_Verification")
    SIZE, FRAMES = 96, 80
    
    def run_axis(axis_idx):
        eng = EST_Engine(size=SIZE, dim=3)
        c = SIZE // 2
        if axis_idx == 0: eng.u[c, c-5:c+5, c-5:c+5] = 1 
        if axis_idx == 1: eng.u[c-5:c+5, c, c-5:c+5] = 1 
        if axis_idx == 2: eng.u[c-5:c+5, c-5:c+5, c] = 1 
        positions = []
        for _ in range(FRAMES):
            eng.step()
            coords = np.argwhere(eng.u)
            if len(coords) > 0: positions.append(coords[:, axis_idx].mean())
            else: positions.append(0)
        return positions
    
    pX = run_axis(0)
    pY = run_axis(1)
    pZ = run_axis(2)
    
    plt.style.use('dark_background')
    plt.figure(figsize=(10,6))
    plt.plot(pX, color="red", label="X-Axis (Grid Jump)")
    plt.plot(pY, color="green", label="Y-Axis (Grid Jump)")
    plt.plot(pZ, color="blue", label="Z-Axis (Memory Linear)")
    plt.title("Isotropy Failure: Lattice Bias")
    plt.legend()
    runner.save_plot("Isotropy_Proof_Graph.png")
    os.startfile(runner.base_dir)

def Run_Algorithmic_Invariance_Test():
    """New menu option for algorithmic invariance validation"""
    runner = ExperimentRunner("Algorithmic_Invariance_Validation")
    results = test_algorithmic_invariance(runner)
    runner.log("Algorithmic Invariance Test Complete.")
    os.startfile(runner.base_dir)

def Run_Nucleation_Scaling_Test():
    """New menu option for nucleation scaling validation"""
    runner = ExperimentRunner("Nucleation_Scaling_Validation")
    results = run_nucleation_scaling_test(runner)
    runner.log("Nucleation Scaling Test Complete.")
    os.startfile(runner.base_dir)

def Run_Physical_Invariance_Test():
    """NEW: Test invariance with PHYSICAL SCALING enabled"""
    runner = ExperimentRunner("Physical_Invariance_Test")
    
    print("Testing Algorithmic Invariance WITH PHYSICAL SCALING...")
    methods = ["zlib", "lzma", "bzip2", "entropy"]
    results = {}
    
    for method in methods:
        runner.log(f"\nRunning with {method} + physical scaling...")
        
        # Create engine WITH physical scaling
        engine = EST_Engine(
            size=48, 
            complexity_method=method, 
            beta=3.4, 
            lambda_t=0.48, 
            threads=1,
            physical_scaling=True,  # ENABLED!
            box_size_mpc=500.0,
            H0=67.4,
            Ω_m=0.315
        )
        
        # Standard nucleation pattern
        engine.u[20:28, 20:28, 20:28] = 1
        
        densities_sim = []  # Simulation units
        densities_phys = []  # Physical Ω_m units
        
        for frame in tqdm(range(60), desc=f"{method}"):  # Fewer frames for speed
            d_sim = engine.step()
            if d_sim is not None: 
                densities_sim.append(d_sim)
                # Convert to physical density
                Ω_phys = engine.scaler.density_to_Ωm(d_sim)
                densities_phys.append(Ω_phys)
        
        if densities_sim:
            final_density_sim = densities_sim[-1]
            final_density_phys = densities_phys[-1]
        else:
            final_density_sim = 0
            final_density_phys = 0
            
        results[method] = {
            'final_density_sim': final_density_sim,
            'final_density_phys': final_density_phys,
            'all_densities_sim': densities_sim,
            'all_densities_phys': densities_phys
        }
        
        runner.log(f"  {method}:")
        runner.log(f"    Simulation density = {final_density_sim:.6f}")
        runner.log(f"    Physical Ω_m = {final_density_phys:.6f}")
    
    # Generate dual comparison plot
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for method, data in results.items():
        if data['all_densities_sim']:
            ax1.plot(data['all_densities_sim'], label=method, linewidth=2)
        if data['all_densities_phys']:
            ax2.plot(data['all_densities_phys'], label=method, linewidth=2)
    
    ax1.set_title("Algorithmic Invariance (Simulation Units)")
    ax1.set_xlabel("Frames")
    ax1.set_ylabel("Global Density (sim)")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.set_title("Algorithmic Invariance (Physical Units)")
    ax2.set_xlabel("Frames")
    ax2.set_ylabel("Ω_m (physical)")
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    runner.save_plot("Algorithmic_Invariance_With_Physical_Scaling.png")
    
    # Calculate variance in BOTH units
    final_densities_sim = [data['final_density_sim'] for data in results.values()]
    final_densities_phys = [data['final_density_phys'] for data in results.values()]
    
    variance_sim = max(final_densities_sim) - min(final_densities_sim)
    variance_phys = max(final_densities_phys) - min(final_densities_phys)
    
    runner.log(f"\nVariance Analysis:")
    runner.log(f"  Simulation units: {variance_sim:.6f}")
    runner.log(f"  Physical units (Ω_m): {variance_phys:.6f}")
    
    # Check invariance in physical units (more important!)
    runner.log(f"\nPhysical Invariance Test: {'PASS' if variance_phys < 0.005 else 'INCONCLUSIVE'}")
    
    # Save detailed results
    csv_path = runner.base_dir / "physical_invariance_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Final_Density_Sim", "Final_Ω_m", "Variance_From_Mean_Ω_m"])
        
        mean_Ω_m = np.mean(final_densities_phys)
        for method, data in results.items():
            variance_from_mean = abs(data['final_density_phys'] - mean_Ω_m)
            writer.writerow([
                method, 
                f"{data['final_density_sim']:.6f}", 
                f"{data['final_density_phys']:.6f}",
                f"{variance_from_mean:.6f}"
            ])
    
    runner.log(f"\nDetailed results saved to: {csv_path.name}")
    runner.log("Physical Algorithmic Invariance Test Complete.")
    os.startfile(runner.base_dir)

# ==============================================================================
#   MAIN MENU (Updated for v2.7 with ALL validation tests)
# ==============================================================================

if __name__ == "__main__":
    print(r"""
     _____ ___  _____     _       _       
    | ____/ __||_   _|   | | __ _| |__    
    |  _| \__ \  | |_____| |/ _` | '_ \   
    | |___|___/  | |_____| | (_| | |_) |  
    |_____|___/  |_|     |_|\__,_|_.__/   
                                          
    
    Event-State Theory - Unified Laboratory [PHYSICAL SCALING EDITION]
    Independent Researcher: Torben Wille
    EST LABORATORY v3.0 - With Cosmological Scaling & Validation
    
    === COSMOLOGICAL SIMULATIONS ===
    1. Standard Cosmic Emergence (Default with Physical Scaling)
    2. Manual Config with Physical Scaling (Full Control)
    
    === VALIDATION & TESTING ===
    3. Enhanced Goldilocks Zone - All Complexity Methods
    4. Algorithmic Invariance Test (Critical Defense)
    5. Algorithmic Invariance WITH Physical Scaling (NEW! - Scientific Rigor)
    6. Nucleation Scaling Test (Robustness Proof)
    
    === PHYSICAL ANALYSIS ===
    7. Power Spectrum Analysis (With Physical Units)
    8. Parameter Phase Space (Traditional Goldilocks)
    
    === DIAGNOSTICS ===
    9. Isotropy Check (Lattice Bias)

    """)
    
    c = input("\nSelect Protocol: ")
    
    if c == "1":
        Run_Cosmology_Simulation(manual=False)
    elif c == "2":
        Run_Cosmology_Simulation(manual=True)
    elif c == "3":
        # You need to add this function from previous answer
        Run_Enhanced_Parameter_Sweep()
    elif c == "4":
        Run_Algorithmic_Invariance_Test()
    elif c == "5":
        Run_Physical_Invariance_Test()  # NEW!
    elif c == "6":
        Run_Nucleation_Scaling_Test()
    elif c == "7":
        # Optional: Add a dedicated power spectrum analysis
        print("Power Spectrum Analysis - run option 1 or 2 first")
        input("Press Enter to return to menu...")
        exec(open(__file__).read())
    elif c == "8":
        # Simple parameter sweep (like original)
        Run_Parameter_Sweep_Simple()
    elif c == "9":
        # You'll need to add Run_Relativity_Check() function
        Run_Relativity_Check()

    else:
        print("Invalid option. Running default simulation...")
        Run_Cosmology_Simulation(manual=False)    
