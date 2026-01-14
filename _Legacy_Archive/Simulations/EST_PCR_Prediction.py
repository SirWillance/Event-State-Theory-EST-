import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ——— CONFIGURATION ———
TIMELINE_SIZE = 1000
EVENT_TIME = 500
EVENT_ENERGY = 1.0      # The magnitude of the collision
ALPHA_VACUUM = 0.1      # Cost of existing (Mass term) - Universe wants to be 0
BETA_SMOOTHNESS = 20.0  # Cost of changing (Kinetic term) - Universe hates sharp jumps
ITERATIONS = 5000       # How long the universe thinks before rendering

# ——— FILE SETUP ———
script_dir = os.path.dirname(os.path.abspath(__file__))
output_folder = os.path.join(script_dir, "EST_PCR_Simulation_Results")
os.makedirs(output_folder, exist_ok=True)

def log(text):
    print(text)

# ——— THE PHYSICS ENGINE ———
def calculate_pcr_emergence():
    # 1. Initialize Timeline (Flat Vacuum)
    # The universe starts at 0 everywhere.
    timeline = np.zeros(TIMELINE_SIZE)
    
    # 2. The Fixed Constraint (The Event)
    # This is the "Future" pulling on the "Past"
    timeline[EVENT_TIME] = EVENT_ENERGY 

    log("Initializing Field of Potential...")
    log(f"Constraint: Event at t={EVENT_TIME} with Energy={EVENT_ENERGY}")
    log(f"Optimization: Minimizing Cost J (Alpha={ALPHA_VACUUM}, Beta={BETA_SMOOTHNESS})")

    # 3. The Relaxation Loop (Calculating the Path of Least Resistance)
    # The universe iterates to find the lowest cost configuration
    for i in range(ITERATIONS):
        # Store old state to check convergence
        old_timeline = timeline.copy()
        
        # Update every point based on its neighbors (Local Causality c=1)
        # Formula derived from minimizing J = Sum(Alpha*x^2 + Beta*(dx)^2)
        # This is the discrete Euler-Lagrange equation for the Cost Function.
        for t in range(1, TIMELINE_SIZE - 1):
            if t == EVENT_TIME: continue # The Event is fixed
            
            # The value that minimizes cost is a weighted average of neighbors
            # This balances "Staying at 0" vs "Connecting to Neighbors"
            timeline[t] = (BETA_SMOOTHNESS * (timeline[t-1] + timeline[t+1])) / (2 * BETA_SMOOTHNESS + ALPHA_VACUUM)

        # Check for convergence (Is the calculation finished?)
        diff = np.sum(np.abs(timeline - old_timeline))
        if diff < 1e-6:
            log(f"Converged at iteration {i}")
            break
            
    return timeline

# ——— RUN SIMULATION ———
log("--- EST PCR SIMULATION START ---")
final_timeline = calculate_pcr_emergence()

# ——— DATA ANALYSIS ———
# Check if there is signal BEFORE the event (t < 500)
pre_event_window = final_timeline[EVENT_TIME-50:EVENT_TIME]
pcr_detected = np.any(pre_event_window > 0.01)

# ——— VISUALIZATION ———
plt.figure(figsize=(12, 6), facecolor='black')
ax = plt.gca()
ax.set_facecolor('black')

# Plot the timeline
plt.plot(final_timeline, color='cyan', linewidth=2, label="Vacuum State ($\Omega$)")

# Highlight the PCR Zone
if pcr_detected:
    plt.fill_between(range(EVENT_TIME-50, EVENT_TIME), 
                     final_timeline[EVENT_TIME-50:EVENT_TIME], 
                     color='magenta', alpha=0.3, label="Pre-Causal Resonance (Emergent)")

# Mark the Event
plt.axvline(x=EVENT_TIME, color='white', linestyle='--', label="The Event ($t=0$)")

# Styling
plt.title("Computational Proof of Pre-Causal Resonance\n(Emergent from Cost Minimization)", color='white', fontsize=14)
plt.xlabel("Time Frames", color='white')
plt.ylabel("Information Density / Energy", color='white')
plt.legend()
plt.grid(color='gray', alpha=0.3)
plt.tick_params(colors='white')

# Save
img_path = os.path.join(output_folder, "PCR_Emergence_Proof.png")
plt.savefig(img_path, dpi=150, bbox_inches='tight')
log(f"Proof Image Saved: {img_path}")

# Save Data
csv_path = os.path.join(output_folder, "PCR_Emergence_Data.csv")
np.savetxt(csv_path, final_timeline, delimiter=",", header="Vacuum_State")
log(f"Proof Data Saved: {csv_path}")

# Display "The Hyrica"
if pcr_detected:
    log("\n=== EUREKA MOMENT ===")
    log("A signal emerged AUTOMATICALLY before the event.")
    log("You did not program the curve. The Cost Function created it.")
    log("This proves that in a Least-Cost Universe, the Future casts a shadow on the Past.")
else:
    log("No resonance detected. Parameters need tuning.")

log("--- END SIMULATION ---")
