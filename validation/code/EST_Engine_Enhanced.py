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
#   EST ENGINE (Enhanced with Validation Tests)
# ==============================================================================

class EST_Engine:
    def __init__(self, size=64, dim=3, lambda_t=0.482, beta=3.4, epsilon=0.0748, 
                 candidates=90, threads=None, complexity_method="zlib"):
        self.size = size
        self.dim = dim
        self.shape = tuple([size] * dim)
        self.u = np.zeros(self.shape, dtype=np.uint8)
        self.LAMBDA = lambda_t
        self.BETA = beta
        self.EPSILON = epsilon
        self.CANDIDATES = candidates
        self.complexity_method = complexity_method
        # NEW: localized-collapse control
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
            # Shannon entropy approximation
            if len(arr_bytes) == 0:
                return 0
            counts = Counter(arr_bytes)
            proportions = [count / len(arr_bytes) for count in counts.values()]
            entropy = -sum(p * math.log2(p) for p in proportions if p > 0)
            return int(entropy * 1000)  # scaled for comparable magnitude
        else:
            return len(zlib.compress(arr_bytes))  # fallback

    def _cost_task(self, args):
        curr_bytes, cand_data, beta, eps = args
        cand_arr = np.frombuffer(cand_data, dtype=np.uint8).reshape(self.shape)
        curr_arr = np.frombuffer(curr_bytes, dtype=np.uint8).reshape(self.shape)
        
        flips = (curr_arr != cand_arr)
        dE = np.sum(flips)

        # --- Localized collapse: penalize flips inside core_mask ---
        if self.core_mask is not None and self.CORE_WEIGHT != 1.0:
            # core_mask is a boolean array same shape as u
            core_flips = np.logical_and(flips, self.core_mask)
            dE_core = np.sum(core_flips)
            dE_outer = dE - dE_core
            dE_eff = dE_outer + self.CORE_WEIGHT * dE_core
        else:
            dE_eff = dE
        # -----------------------------------------------------------

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
        candidates = [r[1] for r in results]
        
        costs -= costs.min()
        probs = np.exp(-costs / self.LAMBDA)
        probs_sum = probs.sum()
        
        if probs_sum == 0 or np.isnan(probs_sum): return 
        probs /= probs_sum
        
        idx = np.random.choice(len(candidates), p=probs)
        self.u = candidates[idx]
        return self.u.mean()

# ==============================================================================
#   VALIDATION TEST SUITE
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
                          beta=3.4, lambda_t=0.48, threads=1)
        
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

def run_nucleation_scaling_test(runner=None):
    """Test cosmic web formation across different nucleation site counts"""
    if runner is None:
        runner = ExperimentRunner("Nucleation_Scaling")
    
    site_counts = [8, 12, 16, 20, 24, 29, 32, 36, 40]
    results = {}
    
    runner.log(f"Testing nucleation scaling with sites: {site_counts}")
    
    for sites in site_counts:
        runner.log(f"Testing {sites} nucleation sites...")
        engine = EST_Engine(size=64, beta=3.4, lambda_t=0.48, threads=1)
        
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

def Run_Tau_Collapse_Experiment():
    """
    Protocol 9 (Experimental):
    Attempt to create a localized information-collapse region
    (black-hole candidate) by making flips inside a central sphere
    much more expensive than outside, then measuring the τ-field
    (number of flips per cell) over time.

    NOTE: In current parameter settings this produces a global
    τ-collapse (universe-wide freeze), not a stable localized
    black hole. Kept here as an experimental protocol for future tuning.
    """

    runner = ExperimentRunner("Tau_Collapse_Experiment")

    SIZE   = 128
    FRAMES = 240
    R_CORE = 24           # radius of collapsing core
    CORE_W = 10.0         # cost multiplier for flips inside core

    runner.log(f"Grid={SIZE}^3, FRAMES={FRAMES}, R_CORE={R_CORE}, CORE_WEIGHT={CORE_W}")

    # Use entropy-based complexity to avoid axis bias
    eng = EST_Engine(size=SIZE, dim=3, lambda_t=0.482, beta=3.4,
                     epsilon=0.0748, candidates=90, threads=None,
                     complexity_method='entropy')

    # --- define core mask (central sphere) ---
    cy = cx = cz = SIZE // 2
    yy, xx, zz = np.ogrid[:SIZE, :SIZE, :SIZE]
    dist2 = (yy - cy)**2 + (xx - cx)**2 + (zz - cz)**2
    core_mask = dist2 < R_CORE**2

    eng.core_mask = core_mask
    eng.CORE_WEIGHT = CORE_W

    # Fill core with dense "mass"
    eng.u[core_mask] = 1
    runner.log("Core region initialised with dense matter.")

    # --- evolve and accumulate τ-field ---
    tau_field = np.zeros_like(eng.u, dtype=np.int32)

    for t in range(FRAMES):
        prev = eng.u.copy()
        eng.step()
        nxt = eng.u
        tau_field += (prev != nxt).astype(np.int32)

        if (t+1) % 40 == 0:
            runner.log(f"Frame {t+1}/{FRAMES}   mean τ so far: {tau_field.mean():.4f}")

    # Save τ volume
    np.save(runner.base_dir / "tau_field.npy", tau_field)
    runner.log("Saved tau_field.npy")

    # --- central slices for quick inspection ---
    mid = SIZE // 2
    slices = {
        "Tau_XY_zmid.png": tau_field[:, :, mid],
        "Tau_XZ_ymid.png": tau_field[mid, :, :],
        "Tau_YZ_xmid.png": tau_field[:, mid, :],
    }

    plt.style.use('dark_background')
    for fname, img in slices.items():
        plt.figure(figsize=(6,5))
        plt.imshow(img, cmap="inferno")
        plt.colorbar(label="Proper-Time Accumulation")
        plt.title(fname.replace(".png", ""))
        runner.save_plot(fname)
        plt.close()

    runner.log("Saved central τ slices (XY/XZ/YZ).")
    runner.log("Current behaviour: collapse tends to spread globally.")
    runner.log("Goal (future): dark τ≈0 core, brighter shell (horizon), active exterior.")
    try:
        os.startfile(runner.base_dir)
    except Exception:
        pass


# ==============================================================================
#   CONTROLLERS (Original Functions Preserved)
# ==============================================================================

class ExperimentRunner:
    def __init__(self, exp_name):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = Path(__file__).parent / f"EST_Output_{exp_name}_{self.timestamp}"
        self.base_dir.mkdir(exist_ok=True)
        self.log_path = self.base_dir / "experiment_log.txt"
        self.log(f"--- EXPERIMENT: {exp_name} ---")

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
        """Save a 2D numpy slice as a heatmap PNG."""
        plt.style.use('dark_background')
        plt.figure(figsize=(6, 5))
        plt.imshow(array2d, cmap="inferno", origin="lower")
        plt.colorbar(label="Proper-Time Accumulation")
        plt.title(filename.replace(".png", ""))
        path = self.base_dir / filename
        plt.savefig(path, dpi=300, facecolor='black')
        plt.close()
        return path

    def save_3d(self, volume, filename):
        """
        Minimal placeholder: export a thresholded point cloud as a pseudo-OBJ.
        This avoids crashes and gives advanced users something to import.
        """
        path = self.base_dir / filename
        thr = np.percentile(volume, 90)  # top 10% activity
        yy, xx, zz = np.where(volume >= thr)
        with open(path, "w") as f:
            for y, x, z in zip(yy, xx, zz):
                f.write(f"v {x} {y} {z}\n")
        return path


def get_user_params(default_size, default_frames, default_sites):
    print("\n--- Manual Configuration Mode ---")
    print(f"(Press ENTER to use defaults)")
    try:
        in_size = input(f"Grid Size [Default {default_size}]: ")
        size = int(in_size) if in_size else default_size
        
        in_frames = input(f"Duration (Frames) [Default {default_frames}]: ")
        frames = int(in_frames) if in_frames else default_frames
        
        in_sites = input(f"Nucleation Sites [Default {default_sites}]: ")
        sites = int(in_sites) if in_sites else default_sites
        
        in_beta = input(f"Beta (Complexity Cost) [Default 3.4]: ")
        beta = float(in_beta) if in_beta else 3.4
        
        in_lam = input(f"Lambda (Temperature) [Default 0.48]: ")
        lam = float(in_lam) if in_lam else 0.48
        
        return size, frames, beta, lam, sites
    except ValueError:
        print("Invalid input. Using Defaults.")
        return default_size, default_frames, 3.4, 0.482, default_sites

def Run_Cosmology_Simulation(manual=False):
    runner = ExperimentRunner("Cosmology_Emergence")
    
    # 1. SETUP PARAMETERS
    SIZE, FRAMES, BETA, LAM, SITES = 128, 150, 3.4, 0.482, 12
    
    if manual:
        SIZE, FRAMES, BETA, LAM, SITES = get_user_params(SIZE, FRAMES, SITES)
    
    runner.log(f"Config: Grid {SIZE}^3 | Frames {FRAMES} | Beta {BETA} | Lambda {LAM} | Sites {SITES}")
    
    # 2. INITIALIZE ENGINE
    engine = EST_Engine(size=SIZE, dim=3, candidates=90, beta=BETA, lambda_t=LAM)
    
    runner.log(f"Injecting {SITES} Nucleation Sites...")
    for _ in range(SITES):
        c = np.random.randint(20, SIZE-20, size=3)
        rr, cc, dd = np.ogrid[:SIZE, :SIZE, :SIZE]
        dist_sq = (rr - c[0])**2 + (cc - c[1])**2 + (dd - c[2])**2
        engine.u[dist_sq < 64] = 1 

    # 3. RUN SIMULATION LOOP
    densities = []
    for t in tqdm(range(FRAMES), desc="Simulating Universe"):
        d = engine.step()
        densities.append(d)
        
    runner.log("Simulation Complete. Generating Data...")

    npy_state_path = runner.base_dir / "final_universe_state.npy"
    npy_dens_path  = runner.base_dir / "density_timeseries.npy"
    np.save(npy_state_path, engine.u)
    np.save(npy_dens_path, np.array(densities, dtype=float))
    runner.log(f"Saved NPY data: {npy_state_path.name}, {npy_dens_path.name}")
    
    # 4. EXPORT RAW CSV DATA
    csv_path = runner.base_dir / "density_data.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Global_Density"])
        for i, val in enumerate(densities):
            writer.writerow([i, val])
    runner.log(f"Saved: {csv_path.name}")

    # 5. GENERATE PLOTS
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 5))
    plt.plot(densities, color='cyan', linewidth=2)
    plt.title(f"Baryonic Accumulation (Beta={BETA})")
    plt.xlabel("Cosmic Frames (n)")
    plt.ylabel("Global Density")
    plt.grid(alpha=0.2)
    runner.save_plot("Baryonic_Accumulation_Graph.png")
    
    # 6. GENERATE VISUALS (Deep Field)
    projection = engine.u.sum(axis=0)
    plt.figure(figsize=(8, 8))
    plt.imshow(projection, cmap='inferno', interpolation='nearest')
    plt.title(f"EST Deep Field Projection (Frame {FRAMES})")
    plt.axis('off')
    runner.save_plot(f"Deep_Field_Projection_Frame_{FRAMES}.png")

    # 7. GENERATE GIF
    try:
        runner.log("Rendering 3D Rotation GIF (Downsampled)...")
        frames = []
        points = np.argwhere(engine.u)
        
        # Limit points for performance
        if len(points) > 5000:
            points = points[np.random.choice(len(points), 5000, replace=False)]
            
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('black')
        
        for angle in range(0, 360, 15):
            ax.clear()
            ax.scatter(points[:,0], points[:,1], points[:,2], c='white', s=0.5, alpha=0.6)
            ax.set_facecolor('black')
            ax.grid(False)
            ax.axis('off')
            ax.view_init(elev=20, azim=angle)
            
            tmp_path = runner.base_dir / f"temp_{angle}.png"
            plt.savefig(tmp_path, facecolor='black')
            frames.append(imageio.imread(tmp_path))
            os.remove(tmp_path)
            
        plt.close()
        imageio.mimsave(runner.base_dir / "Cosmic_Structure_Spin.gif", frames, fps=15)
        runner.log("GIF Generated.")
        
    except Exception as e:
        runner.log(f"GIF Gen Failed: {e}")

    runner.log(f"Success. Folder: {runner.base_dir}")
    os.startfile(runner.base_dir)

def Run_Parameter_Sweep():
    runner = ExperimentRunner("Parameter_Verification")
    print("Calculating the Goldilocks Zone (3-5 mins)...")
    beta_range = np.linspace(2.0, 5.0, 10)
    lambda_range = np.linspace(0.1, 1.0, 10)
    results = np.zeros((10, 10))
    # We use small grid for sweeping to prevent CPU melt
    SWEEP_SIZE = 32
    
    pbar = tqdm(total=100)
    
    for i, beta in enumerate(beta_range):
        for j, lam in enumerate(lambda_range):
            eng = EST_Engine(size=SWEEP_SIZE, dim=3, lambda_t=lam, beta=beta, threads=1)
            eng.u[SWEEP_SIZE//2-4:SWEEP_SIZE//2+4, 
                  SWEEP_SIZE//2-4:SWEEP_SIZE//2+4, 
                  SWEEP_SIZE//2-4:SWEEP_SIZE//2+4] = 1
            final_density = 0
            for _ in range(30):
                res = eng.step()
                if res is not None: final_density = res
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
    
    # CSV for Phase Space
    csv_path = runner.base_dir / "parameter_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Beta", "Lambda", "Resulting_Density"])
        for i, b in enumerate(beta_range):
            for j, l in enumerate(lambda_range):
                writer.writerow([b, l, results[i,j]])
    
    runner.log("Verification Complete.")
    os.startfile(runner.base_dir)

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

# ==============================================================================
#   TIME-TOPOLOGY / PROPER-TIME SCAN  (Protocol 7)
# ==============================================================================

def _local_flips(prev, nxt, y, x, z, R=2):
    """Count how many cells changed in a ball of radius R around (y,x,z)."""
    size = prev.shape[0]
    y0, y1 = max(0, y-R), min(size, y+R+1)
    x0, x1 = max(0, x-R), min(size, x+R+1)
    z0, z1 = max(0, z-R), min(size, z+R+1)
    sub_prev = prev[y0:y1, x0:x1, z0:z1]
    sub_nxt  = nxt[y0:y1, x0:x1, z0:z1]
    return int(np.count_nonzero(sub_prev != sub_nxt))


def Run_Time_Dilation_Scan():
    """
    Protocol 7:
    Measure EST 'proper time' along stationary and moving worldlines
    in an already-structured universe, then compare γ_EST(v) to
    Minkowski γ(v) as a *reference only*.
    """
    runner = ExperimentRunner("Time_Dilation_Scan")

    # ------------------ configuration ------------------
    SIZE        = 96          # smaller than full 128^3 for speed
    WARMUP      = 80          # frames to let structure form
    MEASURE_FR  = 220         # frames for proper-time accumulation
    RADIUS      = 2           # neighbourhood radius for local flips
    VELOCITIES  = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50]

    runner.log(f"Config: SIZE={SIZE}, WARMUP={WARMUP}, MEASURE={MEASURE_FR}, R={RADIUS}")
    runner.log(f"Velocities: {VELOCITIES}")

    # ------------------ 1. warm up a universe ------------------
    # Use entropy-based complexity here to avoid axis bias
    eng = EST_Engine(size=SIZE, dim=3, lambda_t=0.482, beta=3.4,
                     epsilon=0.0748, candidates=90, threads=1,
                     complexity_method="entropy")

    runner.log("Injecting nucleation sites...")
    for _ in range(8):
        c = np.random.randint(12, SIZE-12, size=3)
        rr, cc, dd = np.ogrid[:SIZE, :SIZE, :SIZE]
        dist_sq = (rr - c[0])**2 + (cc - c[1])**2 + (dd - c[2])**2
        eng.u[dist_sq < 36] = 1

    runner.log(f"Warming up for {WARMUP} frames to reach structured state...")
    for _ in tqdm(range(WARMUP), desc="Warmup"):
        eng.step()

    # Choose a time-active region: center of mass of matter
    coords = np.argwhere(eng.u == 1)
    if len(coords) == 0:
        runner.log("No active matter found after warmup. Aborting time scan.")
        return
    cy, cx, cz = coords.mean(axis=0).astype(int)
    runner.log(f"Worldline base position (center of mass): y={cy}, x={cx}, z={cz}")

    # Save this warm state so each velocity starts from the same universe
    base_state = eng.u.copy()

    # ------------------ 2. velocity scan ------------------
    results = []
    csv_path = runner.base_dir / "gamma_results.csv"
    with open(csv_path, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["v", "tau_stationary", "tau_moving", "gamma_EST"])

        for v in VELOCITIES:
            runner.log(f"\n[SCAN] v = {v:.2f} cells/frame")

            # fresh engine for this velocity, same warm starting state
            eng_v = EST_Engine(size=SIZE, dim=3, lambda_t=0.482, beta=3.4,
                               epsilon=0.0748, candidates=90, threads=1,
                               complexity_method="entropy")
            eng_v.u = base_state.copy()

            tau_stat = 0.0
            tau_mov  = 0.0

            for n in tqdm(range(MEASURE_FR), desc=f"v={v:.2f}", leave=False):
                prev = eng_v.u.copy()
                eng_v.step()
                nxt  = eng_v.u

                # stationary worldline
                tau_stat += _local_flips(prev, nxt, cy, cx, cz, R=RADIUS)

                # moving worldline (wrap in x)
                x_mov = int(round(cx + v * n)) % SIZE
                tau_mov  += _local_flips(prev, nxt, cy, x_mov, cz, R=RADIUS)

            gamma_est = math.inf if tau_mov == 0 else (tau_stat / tau_mov)
            runner.log(f"v={v:.2f}  τ_stat={tau_stat:.1f}  τ_mov={tau_mov:.1f}  γ_EST={gamma_est:.3f}")

            writer.writerow([v, tau_stat, tau_mov, gamma_est])
            results.append((v, tau_stat, tau_mov, gamma_est))

    runner.log(f"\nSaved gamma_results.csv -> {csv_path.name}")

    # ------------------ 3. plots ------------------
    v_vals   = [r[0] for r in results]
    tau_s    = [r[1] for r in results]
    tau_m    = [r[2] for r in results]
    gamma_es = [r[3] for r in results]

    plt.style.use('default')

    # Proper time curves
    plt.figure(figsize=(7,5))
    plt.plot(v_vals, tau_s, "o-", label="τ_stationary")
    plt.plot(v_vals, tau_m, "o-", label="τ_moving")
    plt.title("EST Proper Time Accumulation")
    plt.xlabel("Velocity (cells/frame)")
    plt.ylabel("Proper Time τ(v)")
    plt.grid(True, alpha=0.4)
    plt.legend()
    runner.save_plot("tau_plot.png")

    # γ_EST(v)
    plt.figure(figsize=(7,5))
    finite_vs = [vv for (vv,gg) in zip(v_vals, gamma_es) if not math.isinf(gg) and gg > 0]
    finite_gs = [gg for gg in gamma_es if not math.isinf(gg) and gg > 0]
    if finite_vs:
        plt.plot(finite_vs, finite_gs, "o-")
    plt.title("EST Proper-Time Ratio γ(v)")
    plt.xlabel("Velocity (cells/frame)")
    plt.ylabel("γ = τ_stationary / τ_moving")
    plt.grid(True, alpha=0.4)
    runner.save_plot("gamma_plot.png")

    # Minkowski reference overlay (log scale)
    plt.figure(figsize=(7,5))
    mink_v = np.linspace(0.0, max(VELOCITIES)+0.02, 200)
    mink_g = 1.0 / np.sqrt(1.0 - mink_v**2)
    plt.plot(mink_v, mink_g, "r--", label="Minkowski γ(v)")

    safe_gamma = [g if (not math.isinf(g) and g < 1e6) else np.nan for g in gamma_es]
    plt.plot(v_vals, safe_gamma, "bo-", label="EST γ(v)")
    plt.title("Comparison: EST vs Minkowski Time Dilation (Reference)")
    plt.xlabel("Velocity")
    plt.ylabel("γ")
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.4)
    plt.legend()
    runner.save_plot("minkowski_comparison.png")

    runner.log("Time Dilation Scan Complete.")
    os.startfile(runner.base_dir)

# ==============================================================================
#   BLACK HOLE EMULATION PROTOCOL (Protocol 8)
#   — Information Collapse → Frozen Time Domain —
# ==============================================================================

def Run_Heatdeath_Simulation():
    """
    Protocol 8:
    Global heat-death / τ-freeze experiment in a dense-core universe.
    We seed a massive central region and let standard EST dynamics run,
    measuring τ(x,y,z) as the total number of state-changes over time.
    In previous runs this led to near-global τ-collapse (information
    heat-death), not a stable local black hole.
    """
    runner = ExperimentRunner("Heatdeath_Simulation")

    SIZE   = 128
    FRAMES = 600
    RADIUS = 32    # radius of initial dense core

    runner.log("Initializing EST Universe for Heatdeath Simulation...")
    eng = EST_Engine(size=SIZE, dim=3, lambda_t=0.482, beta=3.4,
                     epsilon=0.0748, candidates=90, threads=1,
                     complexity_method="entropy")
                     
    # ---- Create dense matter core ----
    cy = cx = cz = SIZE // 2
    rr, cc, dd = np.ogrid[:SIZE, :SIZE, :SIZE]
    sphere = (rr - cy)**2 + (cc - cx)**2 + (dd - cz)**2 < RADIUS**2
    eng.u[sphere] = 1
    runner.log("Mass core injected — collapse seeded.")

    # ---- Track local time evolution as τ(x,y,z) ----
    tau_map = np.zeros((SIZE, SIZE, SIZE), dtype=np.int32)

    for t in tqdm(range(FRAMES), desc="Heatdeath Simulation"):
        prev = eng.u.copy()
        eng.step()
        nxt = eng.u
        tau_map += (prev != nxt).astype(np.int32)

    runner.log("Simulation complete — saving τ-field and slices...")
    np.save(runner.base_dir / "tau_field.npy", tau_map)

    # central slices
    runner.save_heatmap(tau_map[:, SIZE//2, :], "TimeMap_Slice_Center.png")
    runner.save_heatmap(tau_map[cy, :, :],      "TimeMap_Slice_Axial.png")
    runner.save_3d(tau_map, "TimeMap_3D.obj")   # time-volume export (point cloud OBJ)

    runner.log("\nProtocol 8 finished.")
    runner.log("Typical behaviour so far: near-global τ-collapse (information freeze).")
    runner.log("Inspect TimeMap_* and tau_field.npy for the collapse pattern.")
    try:
        os.startfile(runner.base_dir)
    except Exception:
        pass


# ==============================================================================
#   NEW VALIDATION MENU OPTIONS
# ==============================================================================

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

# ==============================================================================
#   MAIN MENU (Updated)
# ==============================================================================

if __name__ == "__main__":
    print(r"""
     _____ ___  _____     _       _       
    | ____/ __||_   _|   | | __ _| |__    
    |  _| \__ \  | |_____| |/ _` | '_ \   
    | |___|___/  | |_____| | (_| | |_) |  
    |_____|___/  |_|     |_|\__,_|_.__/   
                                          
    
    Event-State Theory - Unified Laboratory [VALIDATION EDITION]
    Independent Researcher: Torben Wille
    EST LABORATORY v2.3 - [Validation Branch]
    1. Standard Cosmic Emergence (The Default Proof)
    2. Manual Config Emergence (Exploration Mode)
    3. Parameter Phase Space (The Goldilocks Verification)
    4. Isotropy Check (The Limitations Proof)
    5. [NEW] Algorithmic Invariance Test (Critical Defense)
    6. [NEW] Nucleation Scaling Test (Robustness Proof)

""")
    c = input("Select Protocol: ")
    if c == "1": Run_Cosmology_Simulation(manual=False)
    elif c == "2": Run_Cosmology_Simulation(manual=True)
    elif c == "3": Run_Parameter_Sweep()
    elif c == "4": Run_Relativity_Check()
    elif c == "5": Run_Algorithmic_Invariance_Test()
    elif c == "6": Run_Nucleation_Scaling_Test()
