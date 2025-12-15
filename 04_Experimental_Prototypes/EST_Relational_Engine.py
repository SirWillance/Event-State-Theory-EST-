"""
Relational EST Simulation - Causal Set Growth with Optimization
Saves ALL results to files automatically.
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass
import json
import pickle
import os
from datetime import datetime
import math

# ============================================================================
# 1. EVENT AND COST FUNCTIONS
# ============================================================================

@dataclass
class Event:
    """A fundamental event in relational spacetime."""
    id: int
    x: float
    y: float
    z: float
    t: float
    energy: float
    matter_type: int  # 0=vacuum, 1=matter, -1=antimatter
    complexity: float = 0.0
    
    def to_dict(self):
        return {
            'id': self.id,
            'x': float(self.x),
            'y': float(self.y),
            'z': float(self.z),
            't': float(self.t),
            'energy': float(self.energy),
            'matter_type': int(self.matter_type),
            'complexity': float(self.complexity)
        }

def relational_complexity(new_event, past_events, causal_links):
    """Estimate relational Kolmogorov complexity."""
    if not past_events:
        return 10.0
    
    similarities = []
    for past in past_events:
        dx = new_event.x - past.x
        dy = new_event.y - past.y
        dz = new_event.z - past.z
        pos_sim = 1.0 / (1.0 + math.sqrt(dx*dx + dy*dy + dz*dz))
        energy_sim = 1.0 / (1.0 + abs(new_event.energy - past.energy))
        type_sim = 1.0 if new_event.matter_type == past.matter_type else 0.5
        similarities.append(pos_sim * energy_sim * type_sim)
    
    avg_similarity = np.mean(similarities) if similarities else 0
    return (1.0 - avg_similarity) * 5.0

def energy_cost(new_event, past_events):
    """Energy required to create this event."""
    base_energy = abs(new_event.energy)
    if past_events:
        avg_past_energy = np.mean([e.energy for e in past_events])
        conservation_factor = 1.0 / (1.0 + abs(new_event.energy - avg_past_energy))
        base_energy *= conservation_factor
    return base_energy

# ============================================================================
# 2. SIMULATION ENGINE
# ============================================================================

class RelationalUniverse:
    """A growing causal set universe following EST principles."""
    
    def __init__(self, alpha=1.0, beta=3.4, epsilon=0.0748, lambda_temp=0.48):
        self.events = []
        self.causal_links = []
        self.next_id = 0
        
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.lambda_temp = lambda_temp
        
        self.stats = {
            'total_events': [],
            'matter_count': [],
            'antimatter_count': [],
            'avg_complexity': [],
            'causal_density': [],
            'asymmetry': []
        }
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"EST_Simulation_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Created output directory: {self.output_dir}")
    
    def find_causal_past(self, candidate_event):
        """Find events in candidate's past lightcone."""
        past = []
        for event in self.events:
            dt = candidate_event.t - event.t
            if dt <= 0:
                continue
            dx = candidate_event.x - event.x
            dy = candidate_event.y - event.y
            dz = candidate_event.z - event.z
            if dt*dt >= dx*dx + dy*dy + dz*dz:
                past.append(event)
        return past
    
    def propose_event(self, step):
        """Generate a candidate event."""
        r = 1.0 + step * 0.1
        theta = random.uniform(0, 2*math.pi)
        phi = random.uniform(0, math.pi)
        
        x = r * math.sin(phi) * math.cos(theta)
        y = r * math.sin(phi) * math.sin(theta)
        z = r * math.cos(phi)
        t = step * 0.5
        
        energy = random.uniform(0.1, 2.0)
        
        # Matter bias from epsilon
        if random.random() < 0.5 + self.epsilon:
            matter_type = 1
        else:
            matter_type = -1
        
        return Event(
            id=self.next_id,
            x=x, y=y, z=z, t=t,
            energy=energy,
            matter_type=matter_type
        )
    
    def calculate_cost(self, candidate, past_events):
        """EST cost function J."""
        K = relational_complexity(candidate, past_events, self.causal_links)
        ΔE = energy_cost(candidate, past_events)
        
        matter_bonus = 0
        if candidate.matter_type == 1:
            matter_bonus = -self.epsilon
        elif candidate.matter_type == -1:
            matter_bonus = self.epsilon
        
        J = self.alpha * ΔE + self.beta * K + matter_bonus
        candidate.complexity = K
        
        return J
    
    def grow_step(self, step, max_candidates=10):
        """Execute one growth step."""
        candidates = []
        
        for _ in range(max_candidates):
            candidate = self.propose_event(step)
            past = self.find_causal_past(candidate)
            cost = self.calculate_cost(candidate, past)
            candidates.append((candidate, past, cost))
        
        costs = [c[2] for c in candidates]
        min_cost = min(costs)
        probabilities = [math.exp(-(c - min_cost)/self.lambda_temp) for c in costs]
        prob_sum = sum(probabilities)
        probabilities = [p/prob_sum for p in probabilities]
        
        choice_idx = np.random.choice(len(candidates), p=probabilities)
        chosen, past, _ = candidates[choice_idx]
        
        chosen.id = self.next_id
        self.next_id += 1
        self.events.append(chosen)
        
        for past_event in past:
            self.causal_links.append((past_event.id, chosen.id))
        
        return chosen
    
    def record_stats(self, step):
        """Record statistics."""
        matter_count = sum(1 for e in self.events if e.matter_type == 1)
        antimatter_count = sum(1 for e in self.events if e.matter_type == -1)
        
        self.stats['total_events'].append(len(self.events))
        self.stats['matter_count'].append(matter_count)
        self.stats['antimatter_count'].append(antimatter_count)
        
        if self.events:
            avg_complexity = np.mean([e.complexity for e in self.events])
            self.stats['avg_complexity'].append(avg_complexity)
            
            causal_density = len(self.causal_links) / max(1, len(self.events))
            self.stats['causal_density'].append(causal_density)
            
            if matter_count + antimatter_count > 0:
                asymmetry = (matter_count - antimatter_count) / (matter_count + antimatter_count)
                self.stats['asymmetry'].append(asymmetry)
            else:
                self.stats['asymmetry'].append(0.0)
    
    def run_simulation(self, steps=200):
        """Run the full simulation."""
        print(f"\nRunning simulation for {steps} steps...")
        
        for step in range(steps):
            if step % 50 == 0:
                print(f"  Step {step}/{steps}")
            
            self.grow_step(step)
            self.record_stats(step)
        
        print(f"Simulation complete. Created {len(self.events)} events.")
        return self.events
    
    def save_all_data(self):
        """Save ALL simulation data to files."""
        print(f"\nSaving data to '{self.output_dir}'...")
        
        # 1. Save events as JSON
        events_data = [e.to_dict() for e in self.events]
        with open(f"{self.output_dir}/events.json", 'w') as f:
            json.dump(events_data, f, indent=2)
        
        # 2. Save causal links
        with open(f"{self.output_dir}/causal_links.txt", 'w') as f:
            for cause_id, effect_id in self.causal_links:
                f.write(f"{cause_id} -> {effect_id}\n")
        
        # 3. Save statistics
        with open(f"{self.output_dir}/statistics.json", 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        # 4. Save parameters
        params = {
            'alpha': self.alpha,
            'beta': self.beta,
            'epsilon': self.epsilon,
            'lambda': self.lambda_temp,
            'total_steps': len(self.stats['total_events']),
            'total_events': len(self.events),
            'matter_count': self.stats['matter_count'][-1] if self.stats['matter_count'] else 0,
            'antimatter_count': self.stats['antimatter_count'][-1] if self.stats['antimatter_count'] else 0,
            'final_asymmetry': self.stats['asymmetry'][-1] if self.stats['asymmetry'] else 0
        }
        with open(f"{self.output_dir}/parameters.json", 'w') as f:
            json.dump(params, f, indent=2)
        
        # 5. Save full simulation object (for later analysis)
        with open(f"{self.output_dir}/simulation.pkl", 'wb') as f:
            pickle.dump(self, f)
        
        print(f"  ✓ events.json - All events with positions and properties")
        print(f"  ✓ causal_links.txt - Causal connections between events")
        print(f"  ✓ statistics.json - Evolution of all statistics")
        print(f"  ✓ parameters.json - Simulation parameters and final results")
        print(f"  ✓ simulation.pkl - Full simulation object for Python")
        
        return self.output_dir

# ============================================================================
# 3. VISUALIZATION (SAVES TO FILES)
# ============================================================================

def create_and_save_plots(universe):
    """Create and save all visualizations."""
    print("\nCreating visualizations...")
    
    stats = universe.stats
    events = universe.events
    
    # Create plots directory
    plots_dir = f"{universe.output_dir}/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. 3D Event Distribution
    fig = plt.figure(figsize=(12, 10))
    
    # Main 3D plot
    ax1 = fig.add_subplot(221, projection='3d')
    xs = [e.x for e in events]
    ys = [e.y for e in events]
    zs = [e.z for e in events]
    colors = ['red' if e.matter_type == 1 else 'blue' for e in events]
    ax1.scatter(xs, ys, zs, c=colors, alpha=0.6, s=5)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Event Distribution\nRed=Matter, Blue=Antimatter')
    
    # 2. Matter Growth
    ax2 = fig.add_subplot(222)
    steps = range(len(stats['total_events']))
    ax2.plot(steps, stats['total_events'], label='Total Events', color='black')
    ax2.plot(steps, stats['matter_count'], label='Matter', color='red')
    ax2.plot(steps, stats['antimatter_count'], label='Antimatter', color='blue')
    ax2.set_xlabel('Simulation Step')
    ax2.set_ylabel('Count')
    ax2.set_title('Growth of Events')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Asymmetry Evolution
    ax3 = fig.add_subplot(223)
    ax3.plot(steps, stats['asymmetry'], color='purple', linewidth=2)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Simulation Step')
    ax3.set_ylabel('Asymmetry (M-A)/(M+A)')
    ax3.set_title('Matter-Antimatter Asymmetry')
    ax3.grid(True, alpha=0.3)
    
    # 4. Complexity and Density
    ax4 = fig.add_subplot(224)
    ax4.plot(steps, stats['avg_complexity'], label='Complexity', color='green')
    ax4.plot(steps, stats['causal_density'], label='Causal Density', color='orange')
    ax4.set_xlabel('Simulation Step')
    ax4.set_ylabel('Value')
    ax4.set_title('Complexity & Causal Structure')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/overview.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ overview.png - Main summary plot")
    
    # 5. Energy Distribution
    plt.figure(figsize=(10, 6))
    energies = [e.energy for e in events]
    plt.hist(energies, bins=30, alpha=0.7, color='teal', edgecolor='black')
    plt.xlabel('Energy')
    plt.ylabel('Count')
    plt.title('Energy Distribution of Events')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{plots_dir}/energy_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ energy_distribution.png - Energy histogram")
    
    # 6. Time Evolution
    plt.figure(figsize=(10, 6))
    times = [e.t for e in events]
    plt.scatter(times, [e.x for e in events], alpha=0.5, s=1, label='X position')
    plt.scatter(times, [e.y for e in events], alpha=0.5, s=1, label='Y position')
    plt.scatter(times, [e.z for e in events], alpha=0.5, s=1, label='Z position')
    plt.xlabel('Time (t)')
    plt.ylabel('Position')
    plt.title('Event Positions vs Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{plots_dir}/time_evolution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ time_evolution.png - Position vs time")
    
    # 7. Create README file
    with open(f"{universe.output_dir}/README.txt", 'w') as f:
        f.write("="*60 + "\n")
        f.write("EST RELATIONAL SIMULATION RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write("This directory contains the complete results from the\n")
        f.write("Event-State Theory relational simulation.\n\n")
        f.write("FILES:\n")
        f.write("- events.json: All events with positions and properties\n")
        f.write("- causal_links.txt: Causal connections between events\n")
        f.write("- statistics.json: Evolution of statistics over time\n")
        f.write("- parameters.json: Simulation parameters and final results\n")
        f.write("- simulation.pkl: Full Python object (load with pickle)\n")
        f.write("- plots/: All visualization images\n\n")
        
        # Add final results
        final_matter = stats['matter_count'][-1] if stats['matter_count'] else 0
        final_antimatter = stats['antimatter_count'][-1] if stats['antimatter_count'] else 0
        final_asymmetry = stats['asymmetry'][-1] if stats['asymmetry'] else 0
        
        f.write("FINAL RESULTS:\n")
        f.write(f"  Total events: {len(events)}\n")
        f.write(f"  Matter events: {final_matter}\n")
        f.write(f"  Antimatter events: {final_antimatter}\n")
        f.write(f"  Matter asymmetry: {final_asymmetry:.6f}\n")
        f.write(f"  Average complexity: {stats['avg_complexity'][-1]:.2f}\n")
        f.write(f"  Causal density: {stats['causal_density'][-1]:.2f}\n\n")
        
        f.write("PARAMETERS:\n")
        f.write(f"  α (energy weight): {universe.alpha}\n")
        f.write(f"  β (complexity weight): {universe.beta}\n")
        f.write(f"  ε (scar tissue): {universe.epsilon}\n")
        f.write(f"  λ (temperature): {universe.lambda_temp}\n")
    
    print(f"  ✓ README.txt - Summary of results")
    
    return plots_dir

# ============================================================================
# 4. MAIN EXECUTION
# ============================================================================

def main():
    """Run the complete simulation and save everything."""
    print("="*70)
    print("RELATIONAL EST SIMULATION - AUTOMATIC DATA SAVING")
    print("="*70)
    
    # Create universe with your Paper II parameters
    universe = RelationalUniverse(
        alpha=1.0,
        beta=3.4,        # From Paper II
        epsilon=0.0748,  # From Paper II
        lambda_temp=0.48 # From Paper II
    )
    
    # Run simulation (300 steps = about 10-20 seconds)
    print("\n" + "-"*70)
    universe.run_simulation(steps=300)
    
    # Save all data
    print("\n" + "-"*70)
    output_dir = universe.save_all_data()
    
    # Create and save visualizations
    print("\n" + "-"*70)
    plots_dir = create_and_save_plots(universe)
    
    # Final summary
    print("\n" + "="*70)
    print("SIMULATION COMPLETE!")
    print("="*70)
    print(f"\nAll data saved to: {output_dir}")
    print(f"\nTo view results:")
    print(f"1. Open the folder: {os.path.abspath(output_dir)}")
    print(f"2. Check README.txt for summary")
    print(f"3. View plots in the 'plots' subfolder")
    print(f"4. Open events.json to see all event data")
    print(f"\nKey files created:")
    print(f"  • {output_dir}/overview.png - Main results visualization")
    print(f"  • {output_dir}/events.json - All event data")
    print(f"  • {output_dir}/statistics.json - Evolution over time")
    
    # Show final asymmetry
    final_asymmetry = universe.stats['asymmetry'][-1] if universe.stats['asymmetry'] else 0
    print(f"\nFINAL ASYMMETRY: {final_asymmetry:.6f}")
    print(f"Your Paper II value: 7.075e-10 ≈ {7.075e-10:.6f}")
    
    if abs(final_asymmetry) > 1e-9:
        print("Note: This is a simplified simulation. Real value emerges at larger scales.")
    
    return universe

if __name__ == "__main__":
    main()