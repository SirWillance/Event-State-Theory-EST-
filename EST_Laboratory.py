import numpy as np
import zlib
import os
import random
import matplotlib.pyplot as plt
import sys
import csv
from datetime import datetime
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
#   EST ENGINE (Windows Optimized)
# ==============================================================================

class EST_Engine:
    def __init__(self, size=64, dim=3, lambda_t=0.482, beta=3.4, epsilon=0.0748, candidates=90, threads=None):
        self.size = size
        self.dim = dim
        self.shape = tuple([size] * dim)
        self.u = np.zeros(self.shape, dtype=np.uint8)
        self.LAMBDA = lambda_t
        self.BETA = beta
        self.EPSILON = epsilon
        self.CANDIDATES = candidates
        self.executor = ThreadPoolExecutor(max_workers=threads)
        self.seed_physics(777)
        
    def seed_physics(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def _complexity(self, arr_bytes):
        return len(zlib.compress(arr_bytes))

    def _cost_task(self, args):
        curr_bytes, cand_data, beta, eps = args
        cand_arr = np.frombuffer(cand_data, dtype=np.uint8).reshape(self.shape)
        curr_arr = np.frombuffer(curr_bytes, dtype=np.uint8).reshape(self.shape)
        
        dE = np.sum(curr_arr != cand_arr)
        K = self._complexity(cand_data)
        
        created = np.sum((curr_arr == 0) & (cand_arr == 1))
        destroyed = np.sum((curr_arr == 1) & (cand_arr == 0))
        return dE + beta * K - eps * (destroyed - created), cand_arr

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
        
        self.u = candidates[np.random.choice(len(candidates), p=probs)]
        return self.u.mean()

# ==============================================================================
#   CONTROLLERS
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

def get_user_params(default_size, default_frames):
    print("\n--- Manual Configuration Mode ---")
    print(f"(Press ENTER to use defaults)")
    try:
        in_size = input(f"Grid Size [Default {default_size}]: ")
        size = int(in_size) if in_size else default_size
        
        in_frames = input(f"Duration (Frames) [Default {default_frames}]: ")
        frames = int(in_frames) if in_frames else default_frames
        
        in_beta = input(f"Beta (Complexity Cost) [Default 3.4]: ")
        beta = float(in_beta) if in_beta else 3.4
        
        in_lam = input(f"Lambda (Temperature) [Default 0.48]: ")
        lam = float(in_lam) if in_lam else 0.48
        
        return size, frames, beta, lam
    except ValueError:
        print("Invalid input. Using Defaults.")
        return default_size, default_frames, 3.4, 0.482

def Run_Cosmology_Simulation(manual=False):
    runner = ExperimentRunner("Cosmology_Emergence")
    
    # DEFAULT SETTINGS
    SIZE, FRAMES, BETA, LAM = 128, 150, 3.4, 0.482
    
    if manual:
        SIZE, FRAMES, BETA, LAM = get_user_params(SIZE, FRAMES)
    
    runner.log(f"Config: Grid {SIZE}^3 | Frames {FRAMES} | Beta {BETA} | Lambda {LAM}")
    
    runner.log("Generating Data...")
    
    # --- PATCH: EXPORT RAW CSV DATA ---
    csv_path = runner.base_dir / "density_data.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Global_Density"])
        for i, val in enumerate(densities):
            writer.writerow([i, val])
    runner.log(f"Saved: {csv_path.name}")
    # ----------------------------------

    plt.style.use('dark_background')
    
    engine = EST_Engine(size=SIZE, dim=3, candidates=90, beta=BETA, lambda_t=LAM)
    
    runner.log("Injecting Asynchronous Nucleation Sites...")
    for _ in range(12):
        c = np.random.randint(20, SIZE-20, size=3)
        rr, cc, dd = np.ogrid[:SIZE, :SIZE, :SIZE]
        dist_sq = (rr - c[0])**2 + (cc - c[1])**2 + (dd - c[2])**2
        engine.u[dist_sq < 64] = 1 

    densities = []
    for t in tqdm(range(FRAMES), desc="Simulating Universe"):
        d = engine.step()
        densities.append(d)
        
    runner.log("Generating Data...")
    
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 5))
    plt.plot(densities, color='cyan', linewidth=2)
    plt.title(f"Baryonic Accumulation (Beta={BETA})")
    plt.xlabel("Cosmic Frames (n)")
    plt.ylabel("Global Density")
    plt.grid(alpha=0.2)
    runner.save_plot("Baryonic_Accumulation_Graph.png")
    
    projection = engine.u.sum(axis=0)
    plt.figure(figsize=(8, 8))
    plt.imshow(projection, cmap='inferno', interpolation='nearest')
    plt.title(f"EST Deep Field Projection (Frame {FRAMES})")
    plt.axis('off')
    runner.save_plot(f"Deep_Field_Projection_Frame_{FRAMES}.png")

    # --- RESTORING THE 3D GIF MODULE ---
    try:
        runner.log("Rendering 3D Rotation GIF (Downsampled)...")
        frames = []
        
        # We only plot points that are MATTER (u=1)
        points = np.argwhere(engine.u)
        
        # SAFETY LIMIT: Only plot max 5,000 points to save RAM
        if len(points) > 5000:
            points = points[np.random.choice(len(points), 5000, replace=False)]
            
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('black')
        
        # Spin loop
        for angle in range(0, 360, 15):
            ax.clear()
            # Plot the dots
            ax.scatter(points[:,0], points[:,1], points[:,2], c='white', s=0.5, alpha=0.6)
            
            # Styling
            ax.set_facecolor('black')
            ax.grid(False)
            ax.axis('off')
            ax.view_init(elev=20, azim=angle)
            
            # Save frame temporarily
            tmp_path = runner.base_dir / f"temp_{angle}.png"
            plt.savefig(tmp_path, facecolor='black')
            frames.append(imageio.imread(tmp_path))
            os.remove(tmp_path)
            
        plt.close()
        imageio.mimsave(runner.base_dir / "Cosmic_Structure_Spin.gif", frames, fps=15)
        runner.log("GIF Generated.")
        
    except Exception as e:
        runner.log(f"GIF Gen Failed (System limits): {e}")

    runner.log(f"Success. Folder: {runner.base_dir}")
    os.startfile(runner.base_dir)

def Run_Parameter_Sweep():
    runner = ExperimentRunner("Parameter_Verification")
    print("Calculating the Goldilocks Zone (3-5 mins)...")
    beta_range = np.linspace(2.0, 5.0, 10)
    lambda_range = np.linspace(0.1, 1.0, 10)
    results = np.zeros((10, 10))
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
                final_density = eng.step() or 0
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
#   MAIN
# ==============================================================================

if __name__ == "__main__":
    print(r"""
     _____ ___  _____     _       _       
    | ____/ __||_   _|   | | __ _| |__    
    |  _| \__ \  | |_____| |/ _` | '_ \   
    | |___|___/  | |_____| | (_| | |_) |  
    |_____|___/  |_|     |_|\__,_|_.__/   
                                          
    Event-State Theory - Unified Laboratory [Windows Optimized]
    Independent Researcher: Torben Wille
    EST LABORATORY v2.0 - [Experimental Branch]
    1. Standard Cosmic Emergence (The Default Proof)
    2. Manual Config Emergence (Exploration Mode) <--- NEW!
    3. Parameter Phase Space (The Goldilocks Verification)
    4. Isotropy Check (The Limitations Proof)
    """)
    c = input("Select Protocol: ")
    if c == "1": Run_Cosmology_Simulation(manual=False)
    elif c == "2": Run_Cosmology_Simulation(manual=True)
    elif c == "3": Run_Parameter_Sweep()
    elif c == "4": Run_Relativity_Check()
