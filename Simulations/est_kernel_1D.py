import random
import zlib
import math
import copy
import os
import sys

# --- AXIOMS ---
UNIVERSE_SIZE = 64
FRAMES = 200
CANDIDATES = 50
LAMBDA = 0.5
ALPHA = 1.0
BETA = 2.5

# --- TARGETING PROTOCOL ---
# Force the file to save next to the script
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

filename = os.path.join(script_dir, "EST_Simulation_Log_1D.txt")

print("==========================================")
print("      EST KERNEL v0.3 - INITIALIZING      ")
print("==========================================")
print(f"Target Directory: {script_dir}")
print(f"Output File:      {filename}")
print("Running...")

# --- THE UNIVERSE ---
current_state = [random.randint(0, 1) for _ in range(UNIVERSE_SIZE)]

def calculate_complexity(state_list):
    data = bytes(state_list)
    return len(zlib.compress(data))

def calculate_energy(state_a, state_b):
    return sum(a != b for a, b in zip(state_a, state_b))

def generate_potential(seed_state):
    omega = []
    for _ in range(CANDIDATES):
        new_state = copy.copy(seed_state)
        flips = random.randint(1, 3) 
        for _ in range(flips):
            idx = random.randint(0, UNIVERSE_SIZE - 1)
            new_state[idx] = 1 - new_state[idx]
        omega.append(new_state)
    return omega

def cost_function(prev_state, candidate_state):
    dE = calculate_energy(prev_state, candidate_state)
    K = calculate_complexity(candidate_state)
    return (ALPHA * dE) + (BETA * K)

# --- EXECUTION ---
try:
    with open(filename, "w") as log_file:
        header = f"EST KERNEL v0.3 | Size: {UNIVERSE_SIZE} | Alpha: {ALPHA} | Beta: {BETA}"
        print(header)
        log_file.write(header + "\n")
        
        separator = "-" * UNIVERSE_SIZE
        print(separator)
        log_file.write(separator + "\n")

        for frame in range(FRAMES):
            omega = generate_potential(current_state)
            
            candidates_with_scores = []
            for candidate in omega:
                J = cost_function(current_state, candidate)
                prob = math.exp(-J / LAMBDA)
                candidates_with_scores.append((candidate, prob))
            
            total_prob = sum(p for c, p in candidates_with_scores)
            if total_prob == 0: total_prob = 1e-9
            
            weights = [p / total_prob for c, p in candidates_with_scores]
            states = [c for c, p in candidates_with_scores]
            
            current_state = random.choices(states, weights=weights, k=1)[0]
            
            visual = "".join(['#' if x == 1 else '.' for x in current_state])
            
            print(f"{frame:03}: {visual}")
            log_file.write(f"{frame:03}: {visual}\n")

        print(separator)
        print("SIMULATION COMPLETE.")
        print(f"Data saved to: {filename}")

except Exception as e:
    print("\nCRITICAL ERROR:")
    print(e)

# --- PAUSE PROTOCOL ---
# This keeps the window open!
print("\n==========================================")
input("Press ENTER to close this window...")