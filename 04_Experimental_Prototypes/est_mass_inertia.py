import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import convolve2d
import os
import csv
import json
import sys
from datetime import datetime

# --- CONFIGURATION ---
# You can change these, and they will be saved automatically
SIZE = 128          
FRAMES = 300       
EPSILON = 0.000001 # Tiny Bias
THRESHOLD = 2       
SEED = None           # Fixed seed for reproducibility (Change to None for random)

# --- SETUP WORKSPACE ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"EST_Ternary_Data_{timestamp}"
os.makedirs(save_dir, exist_ok=True)

print(f"--- EST TERNARY LABORATORY v2.0 ---")
print(f"Config: Size={SIZE}, Epsilon={EPSILON}, Seed={SEED}")
print(f"Saving to: {save_dir}/")

# --- PHYSICS ENGINE ---
class TernaryEngine:
    def __init__(self, size, epsilon, seed):
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize Soup: -1 (Anti), 0 (Void), 1 (Matter)
        self.grid = np.random.choice([-1, 0, 1], size=(size, size), p=[0.33, 0.34, 0.33])
        self.epsilon = epsilon
        self.kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        self.history = []

    def step(self):
        # 1. Pressure Calculation
        neighbor_sum = convolve2d(self.grid, self.kernel, mode='same', boundary='wrap')
        total_pressure = neighbor_sum + self.epsilon
        
        new_grid = self.grid.copy()
        
        # 2. The "Go" Rules (Surface Tension)
        growth_zone = total_pressure > THRESHOLD
        new_grid[(self.grid == -1) & growth_zone] = 0
        new_grid[(self.grid == 0) & growth_zone] = 1
        
        decay_zone = total_pressure < -THRESHOLD
        new_grid[(self.grid == 1) & decay_zone] = 0
        new_grid[(self.grid == 0) & decay_zone] = -1
        
        self.grid = new_grid
        
        # Track Stats
        m = np.sum(self.grid == 1)
        a = np.sum(self.grid == -1)
        v = np.sum(self.grid == 0)
        self.history.append((int(m), int(a), int(v)))
        
        return self.grid

# --- EXECUTION ---
engine = TernaryEngine(SIZE, EPSILON, SEED)

# --- SAVE METADATA (The Fix) ---
metadata = {
    "simulation_type": "Ternary_Matter_Antimatter_War",
    "timestamp": timestamp,
    "grid_size": SIZE,
    "frames": FRAMES,
    "epsilon_bias": EPSILON,
    "pressure_threshold": THRESHOLD,
    "seed": SEED,
    "initial_distribution": "33% Matter, 33% Anti, 34% Void"
}

with open(os.path.join(save_dir, "simulation_parameters.json"), "w") as f:
    json.dump(metadata, f, indent=4)
print("Metadata saved.")

# --- RUN SIMULATION ---
print("Simulating...")

# Setup Animation
fig, ax = plt.subplots(figsize=(8, 8))
cmap = plt.cm.colors.ListedColormap(['#ff0044', '#000000', '#00ccff']) 
bounds = [-1.5, -0.5, 0.5, 1.5]
norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
im = ax.imshow(engine.grid, cmap=cmap, norm=norm)
ax.axis('off')
title = ax.set_title("Initializing...")

def animate(i):
    grid = engine.step()
    im.set_data(grid)
    m, a, v = engine.history[-1]
    title.set_text(f"Frame {i} | Matter: {m} | Anti: {a}")
    if i % 50 == 0:
        sys.stdout.write(f"\rFrame {i}/{FRAMES}")
        sys.stdout.flush()
    return [im, title]

# Save GIF
anim = animation.FuncAnimation(fig, animate, frames=FRAMES, interval=50, blit=False)
anim.save(os.path.join(save_dir, "Matter_War.gif"), writer='pillow', fps=15)
print("\nGIF Saved.")

# Save Data CSV
with open(os.path.join(save_dir, "battle_log.csv"), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Frame", "Matter", "Antimatter", "Void"])
    for i, row in enumerate(engine.history):
        writer.writerow([i, *row])

# Save Final Outcome to Metadata
final_m, final_a, _ = engine.history[-1]
metadata["final_result"] = {
    "matter_count": final_m,
    "antimatter_count": final_a,
    "winner": "Matter" if final_m > final_a else "Antimatter"
}
# Update JSON
with open(os.path.join(save_dir, "simulation_parameters.json"), "w") as f:
    json.dump(metadata, f, indent=4)

# Plot Graph
history = np.array(engine.history)
plt.figure(figsize=(10, 6))
plt.style.use('dark_background')
plt.plot(history[:, 0], color='cyan', label='Matter')
plt.plot(history[:, 1], color='red', label='Antimatter')
plt.plot(history[:, 2], color='gray', linestyle='--', label='Void')
plt.title(f"Victory Graph (Epsilon={EPSILON})")
plt.legend()
plt.savefig(os.path.join(save_dir, "Victory_Graph.png"))

print("Done.")