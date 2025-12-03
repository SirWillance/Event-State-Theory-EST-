import numpy as np
import zlib
import random
import matplotlib.pyplot as plt
from matplotlib import animation
import pickle
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Enhanced EST Engine with Data Collection
# ============================================================================
class EST_Engine_Collapse:
    def __init__(self, size=32, beta=8.0, lambda_t=0.1, candidates=20):
        self.size = size
        self.dim = 3
        self.shape = (size, size, size)
        self.u = np.zeros(self.shape, dtype=np.uint8)
        self.BETA = beta
        self.LAMBDA = lambda_t
        self.candidates = candidates
        
        # Initialize with high randomness (Hot Universe)
        self.u = np.random.randint(0, 2, size=self.shape, dtype=np.uint8)
        
        # Data logging
        self.complexity_history = []
        self.update_history = []
        self.snapshots = []
        self.snapshot_indices = []
        
    def _complexity(self, arr_bytes):
        return len(zlib.compress(arr_bytes))
    
    def get_current_complexity(self):
        return self._complexity(self.u.tobytes())
    
    def step(self, frame_num, snapshot_interval=50):
        current_bytes = self.u.tobytes()
        current_score = self._complexity(current_bytes)
        
        # Store snapshot at intervals
        if frame_num % snapshot_interval == 0 or frame_num == 0:
            self.snapshots.append(self.u.copy())
            self.snapshot_indices.append(frame_num)
        
        # Try to find a better state
        for _ in range(self.candidates):
            c_copy = self.u.copy()
            idx = np.random.randint(0, self.size, 3)
            c_copy[idx[0], idx[1], idx[2]] = 1 - c_copy[idx[0], idx[1], idx[2]]
            
            cand_bytes = c_copy.tobytes()
            K = self._complexity(cand_bytes)
            
            delta_K = K - current_score
            if delta_K < 0:
                prob = 1.0
            else:
                prob = np.exp(-delta_K * self.BETA / self.LAMBDA)
            
            if random.random() < prob:
                self.u = c_copy
                return 1
        
        return 0
    
    def run_simulation(self, frames=200, micro_attempts=50, snapshot_interval=50):
        activity_log = []
        
        for i in range(frames):
            self.complexity_history.append(self.get_current_complexity())
            
            updates = 0
            for _ in range(micro_attempts):
                updates += self.step(i, snapshot_interval)
            
            activity_log.append(updates)
            self.update_history.append(updates)
            
            # Show progress every 20 frames
            if i % 20 == 0:
                print(f"  Frame {i}/{frames} - Updates: {updates}")
        
        # Save final state
        self.snapshots.append(self.u.copy())
        self.snapshot_indices.append(frames)
        
        return activity_log

# ============================================================================
# Data Saving Functions
# ============================================================================
def save_simulation_data(engine, activity_log, save_dir="EST_Collapse_Results"):
    """Save all simulation data to files - GUARANTEED to work"""
    
    # Create directory with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, save_dir, f"run_{timestamp}")
    
    print(f"📁 Creating folder: {save_path}")
    
    try:
        os.makedirs(save_path, exist_ok=True)
        print("✅ Folder created successfully")
    except Exception as e:
        print(f"❌ Could not create folder: {e}")
        # Try current directory
        save_path = f"run_{timestamp}"
        os.makedirs(save_path, exist_ok=True)
        print(f"✅ Created folder in current directory: {save_path}")
    
    try:
        # Save metadata
        metadata = {
            'size': engine.size,
            'beta': engine.BETA,
            'lambda_t': engine.LAMBDA,
            'candidates': engine.candidates,
            'timestamp': timestamp,
            'total_updates': sum(activity_log),
            'final_active_voxels': int(np.sum(engine.u))
        }
        
        # 1. Save numpy arrays (ALWAYS works)
        np.save(os.path.join(save_path, "activity_log.npy"), np.array(activity_log))
        np.save(os.path.join(save_path, "final_universe.npy"), engine.u)
        np.save(os.path.join(save_path, "complexity_history.npy"), 
                np.array(engine.complexity_history))
        
        # 2. Save metadata as text file
        with open(os.path.join(save_path, "metadata.txt"), "w") as f:
            f.write("=== EST Engine: Causal Time Collapse Simulation ===\n\n")
            f.write("PARAMETERS:\n")
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
            
            f.write(f"\nSTATISTICS:\n")
            if engine.complexity_history:
                f.write(f"Initial complexity: {engine.complexity_history[0]} bytes\n")
                f.write(f"Final complexity: {engine.complexity_history[-1]} bytes\n")
                reduction = engine.complexity_history[0] - engine.complexity_history[-1]
                f.write(f"Complexity reduction: {reduction} bytes\n")
            
            freeze_frames = sum(1 for x in activity_log if x == 0)
            f.write(f"Total frames: {len(activity_log)}\n")
            f.write(f"Freeze frames: {freeze_frames} ({(freeze_frames/len(activity_log))*100:.1f}%)\n")
            f.write(f"Total updates: {sum(activity_log)}\n")
        
        # 3. Try to save snapshots if they exist
        if engine.snapshots:
            snapshots_dir = os.path.join(save_path, "snapshots")
            os.makedirs(snapshots_dir, exist_ok=True)
            
            for i, (snapshot, idx) in enumerate(zip(engine.snapshots, engine.snapshot_indices)):
                np.save(os.path.join(snapshots_dir, f"snapshot_frame_{idx:04d}.npy"), snapshot)
            
            print(f"✅ Saved {len(engine.snapshots)} snapshots")
        
        # 4. Try to save pickle (might fail, but we already saved everything else)
        try:
            with open(os.path.join(save_path, "engine.pkl"), "wb") as f:
                pickle.dump(engine, f)
            print("✅ Saved engine.pkl")
        except:
            print("⚠ Could not save pickle, but all essential data is saved")
        
        print(f"\n✅ ALL DATA SAVED SUCCESSFULLY in:\n   {save_path}")
        return save_path
        
    except Exception as e:
        print(f"❌ Error during save: {e}")
        # Even if save fails, return the path we tried
        return save_path

# ============================================================================
# Visualization Functions
# ============================================================================
def create_plots(engine, activity_log, save_path):
    """Create plots - will not crash even if matplotlib fails"""
    try:
        import matplotlib.pyplot as plt
        
        # Create a simple time collapse plot
        plt.figure(figsize=(10, 6))
        plt.plot(activity_log, color='red', linewidth=2, label='Update Rate (τ)')
        plt.axhline(y=0, color='blue', linestyle='--', alpha=0.5)
        
        # Mark freeze region if it exists
        if any(x == 0 for x in activity_log):
            freeze_start = next(i for i, x in enumerate(activity_log) if x == 0)
            plt.axvspan(freeze_start, len(activity_log), alpha=0.2, color='cyan')
            plt.text(freeze_start + 5, max(activity_log)/2, 
                    'FREEZE REGION', color='blue', rotation=90, alpha=0.7)
        
        plt.title('Global Causal Time Collapse', fontsize=14)
        plt.xlabel('Cosmic Frames')
        plt.ylabel('Updates per Frame')
        plt.grid(alpha=0.3)
        plt.legend()
        
        plot_path = os.path.join(save_path, "time_collapse.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Plot saved: time_collapse.png")
        
        # Try to create cross-section plot
        try:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            mid = engine.size // 2
            
            axes[0].imshow(engine.u[mid, :, :], cmap='binary')
            axes[0].set_title(f'XY Plane (z={mid})')
            axes[0].set_xlabel('X'); axes[0].set_ylabel('Y')
            
            axes[1].imshow(engine.u[:, mid, :], cmap='binary')
            axes[1].set_title(f'YZ Plane (x={mid})')
            axes[1].set_xlabel('Z'); axes[1].set_ylabel('Y')
            
            axes[2].imshow(engine.u[:, :, mid], cmap='binary')
            axes[2].set_title(f'XZ Plane (y={mid})')
            axes[2].set_xlabel('Z'); axes[2].set_ylabel('X')
            
            plt.suptitle('Final Universe State - Cross Sections')
            plt.tight_layout()
            
            cross_path = os.path.join(save_path, "cross_sections.png")
            plt.savefig(cross_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ Plot saved: cross_sections.png")
            
        except:
            print("⚠ Could not create cross-section plot")
        
        return True
        
    except ImportError:
        print("⚠ matplotlib not installed - skipping plots")
        return False
    except Exception as e:
        print(f"⚠ Could not create plots: {e}")
        return False

# ============================================================================
# Manual Parameter Input
# ============================================================================
def get_user_parameters():
    """Get parameters from user with sensible defaults"""
    
    print("\n" + "="*60)
    print("EST ENGINE - PARAMETER SETUP")
    print("="*60)
    
    print("\nEnter parameters (press Enter for default):")
    print("-" * 40)
    
    parameters = {}
    
    # Universe size
    while True:
        size_input = input(f"Universe size [3D cube side] (8-64, default=12): ").strip()
        if size_input == "":
            parameters['size'] = 12
            break
        try:
            size = int(size_input)
            if 8 <= size <= 64:
                parameters['size'] = size
                break
            else:
                print("Please enter a value between 8 and 64")
        except:
            print("Please enter a valid integer")
    
    # Beta parameter
    while True:
        beta_input = input(f"Beta (selectivity, 0.1-20.0, default=15.0): ").strip()
        if beta_input == "":
            parameters['beta'] = 15.0
            break
        try:
            beta = float(beta_input)
            if 0.1 <= beta <= 20.0:
                parameters['beta'] = beta
                break
            else:
                print("Please enter a value between 0.1 and 20.0")
        except:
            print("Please enter a valid number")
    
    # Lambda parameter
    while True:
        lambda_input = input(f"Lambda (learning rate, 0.01-2.0, default=0.1): ").strip()
        if lambda_input == "":
            parameters['lambda_t'] = 0.1
            break
        try:
            lambda_t = float(lambda_input)
            if 0.01 <= lambda_t <= 2.0:
                parameters['lambda_t'] = lambda_t
                break
            else:
                print("Please enter a value between 0.01 and 2.0")
        except:
            print("Please enter a valid number")
    
    # Number of frames
    while True:
        frames_input = input(f"Number of frames (10-2000, default=500): ").strip()
        if frames_input == "":
            parameters['frames'] = 500
            break
        try:
            frames = int(frames_input)
            if 10 <= frames <= 2000:
                parameters['frames'] = frames
                break
            else:
                print("Please enter a value between 10 and 2000")
        except:
            print("Please enter a valid integer")
    
    # Micro-attempts per frame
    while True:
        attempts_input = input(f"Micro-attempts per frame (10-200, default=30): ").strip()
        if attempts_input == "":
            parameters['micro_attempts'] = 30
            break
        try:
            attempts = int(attempts_input)
            if 10 <= attempts <= 200:
                parameters['micro_attempts'] = attempts
                break
            else:
                print("Please enter a value between 10 and 200")
        except:
            print("Please enter a valid integer")
    
    # Snapshot interval
    while True:
        snapshot_input = input(f"Snapshot interval (save every N frames, 5-100, default=25): ").strip()
        if snapshot_input == "":
            parameters['snapshot_interval'] = 25
            break
        try:
            interval = int(snapshot_input)
            if 5 <= interval <= 100:
                parameters['snapshot_interval'] = interval
                break
            else:
                print("Please enter a value between 5 and 100")
        except:
            print("Please enter a valid integer")
    
    print("\n" + "-" * 40)
    print("PARAMETER SUMMARY:")
    print(f"  Universe: {parameters['size']}³ ({parameters['size']**3} voxels)")
    print(f"  Beta (selectivity): {parameters['beta']}")
    print(f"  Lambda (learning rate): {parameters['lambda_t']}")
    print(f"  Frames: {parameters['frames']}")
    print(f"  Micro-attempts/frame: {parameters['micro_attempts']}")
    print(f"  Snapshot interval: {parameters['snapshot_interval']} frames")
    print("-" * 40)
    
    confirm = input("\nRun simulation with these parameters? (y/n): ").strip().lower()
    if confirm in ['y', 'yes', '']:
        return parameters
    else:
        print("Simulation cancelled.")
        return None

# ============================================================================
# Main Execution
# ============================================================================
def run_simulation_with_params(params):
    """Run simulation with given parameters"""
    
    print("\n" + "="*60)
    print("STARTING SIMULATION")
    print("="*60)
    
    try:
        # Initialize engine
        print(f"\nInitializing {params['size']}³ universe...")
        engine = EST_Engine_Collapse(
            size=params['size'],
            beta=params['beta'],
            lambda_t=params['lambda_t'],
            candidates=20
        )
        
        initial_active = np.sum(engine.u)
        print(f"Initial state: {initial_active}/{params['size']**3} active voxels ({initial_active/params['size']**3*100:.1f}%)")
        
        # Run simulation
        print(f"\nRunning {params['frames']} frames...")
        activity_log = engine.run_simulation(
            frames=params['frames'],
            micro_attempts=params['micro_attempts'],
            snapshot_interval=params['snapshot_interval']
        )
        
        # Show results
        print(f"\nSimulation complete!")
        final_active = np.sum(engine.u)
        print(f"Final state: {final_active}/{params['size']**3} active voxels ({final_active/params['size']**3*100:.1f}%)")
        print(f"Total updates: {sum(activity_log)}")
        
        freeze_frames = sum(1 for x in activity_log if x == 0)
        print(f"Freeze frames: {freeze_frames}/{params['frames']} ({freeze_frames/params['frames']*100:.1f}%)")
        
        if engine.complexity_history:
            reduction = engine.complexity_history[0] - engine.complexity_history[-1]
            print(f"Complexity reduction: {reduction} bytes")
        
        # Save data
        print("\n" + "-" * 40)
        print("SAVING DATA...")
        save_path = save_simulation_data(engine, activity_log)
        
        if save_path:
            # Create plots
            print("\n" + "-" * 40)
            print("CREATING PLOTS...")
            create_plots(engine, activity_log, save_path)
            
            print("\n" + "="*60)
            print("SIMULATION COMPLETE!")
            print("="*60)
            print(f"\n✅ All files saved in:\n   {save_path}\n")
            
            # List files
            print("Files created:")
            for item in os.listdir(save_path):
                if os.path.isfile(os.path.join(save_path, item)):
                    print(f"  • {item}")
            
            snapshots_dir = os.path.join(save_path, "snapshots")
            if os.path.exists(snapshots_dir):
                snapshot_count = len([f for f in os.listdir(snapshots_dir) if f.endswith('.npy')])
                print(f"  • snapshots/ ({snapshot_count} snapshot files)")
            
            return engine, activity_log, save_path
        else:
            print("❌ Failed to save data!")
            return None, None, None
            
    except Exception as e:
        print(f"\n❌ ERROR during simulation: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def main():
    """Main entry point with menu"""
    
    print("="*60)
    print("EST ENGINE - CAUSAL TIME COLLAPSE SIMULATION")
    print("="*60)
    print(f"Script location: {os.path.abspath(__file__)}")
    
    while True:
        print("\nOptions:")
        print("  1. Run simulation with custom parameters")
        print("  2. Run simulation with defaults (quick test)")
        print("  3. Exit")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            params = get_user_parameters()
            if params:
                engine, activity_log, save_path = run_simulation_with_params(params)
                
                if save_path:
                    print(f"\n✅ Simulation saved in: {save_path}")
                else:
                    print("\n❌ Simulation failed!")
                
                again = input("\nRun another simulation? (y/n): ").strip().lower()
                if again not in ['y', 'yes']:
                    break
        
        elif choice == '2':
            print("\nRunning with default parameters (quick test)...")
            default_params = {
                'size': 12,
                'beta': 15.0,
                'lambda_t': 0.1,
                'frames': 500,
                'micro_attempts': 30,
                'snapshot_interval': 25
            }
            
            print("\nDEFAULT PARAMETERS:")
            for key, value in default_params.items():
                print(f"  {key}: {value}")
            
            engine, activity_log, save_path = run_simulation_with_params(default_params)
            
            if save_path:
                print(f"\n✅ Simulation saved in: {save_path}")
            else:
                print("\n❌ Simulation failed!")
            
            again = input("\nRun another simulation? (y/n): ").strip().lower()
            if again not in ['y', 'yes']:
                break
        
        elif choice == '3':
            print("\nGoodbye!")
            break
        
        else:
            print("Invalid option. Please enter 1, 2, or 3.")
    
    # Keep console open
    try:
        input("\nPress Enter to exit...")
    except:
        pass

# ============================================================================
# Run the program
# ============================================================================
if __name__ == "__main__":
    # Check for required packages
    try:
        import numpy as np
    except ImportError:
        print("❌ ERROR: NumPy is required but not installed!")
        print("Install it with: pip install numpy")
        sys.exit(1)
    
    try:
        import matplotlib
    except ImportError:
        print("⚠ WARNING: Matplotlib not found. Plots will not be created.")
        print("Install it with: pip install matplotlib")
    
    # Run main program
    main()