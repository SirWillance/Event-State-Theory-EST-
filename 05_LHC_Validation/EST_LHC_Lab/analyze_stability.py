import uproot
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import sys
import io
from scipy.stats import ttest_ind

class TeeLogger:
    """Class to capture both print statements and logger output to file"""
    def __init__(self, filename):
        self.filename = filename
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        
    def __enter__(self):
        self.file = open(self.filename, 'w')
        self.buffer = io.StringIO()
        
        # Create a custom stream that writes to both file and stdout
        class TeeStream(io.TextIOBase):
            def __init__(self, file, stdout):
                self.file = file
                self.stdout = stdout
                
            def write(self, data):
                self.file.write(data)
                self.stdout.write(data)
                return len(data)
                
            def flush(self):
                self.file.flush()
                self.stdout.flush()
        
        # Redirect stdout and stderr
        sys.stdout = TeeStream(self.file, self.stdout)
        sys.stderr = TeeStream(self.file, self.stderr)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close()

# Create log filename with timestamp
log_filename = f'EST_Analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

# Start capturing ALL output
with TeeLogger(log_filename):
    file_path = "nano_data2016_42.root"
    print(f"--- STABILITY ANALYSIS: The Cost Function ---")
    print(f"Log file: {log_filename}")
    print(f"Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    with uproot.open(file_path) as file:
        tree = file["Events"]
        
        # Load Data
        print("Loading data...")
        data = tree.arrays(
            ["fixedGridRhoFastjetCentralCalo", "CaloMET_sumEt"],
            library="np",
            entry_stop=10000
        )
        
        n_events = len(data["fixedGridRhoFastjetCentralCalo"])
        rho = data["fixedGridRhoFastjetCentralCalo"]
        met = data["CaloMET_sumEt"]
        
        print(f"Loaded {n_events:,} events")
        
        # Filter out empty events to avoid divide-by-zero
        mask = rho > 0.1
        rho = rho[mask]
        met = met[mask]
        filtered_events = len(rho)
        print(f"After filtering (rho > 0.1): {filtered_events:,} events")
        
        # Calculate Ratio (The "Cost" Coefficient)
        ratios = met / rho
        
        # Define Thresholds
        avg_rho = np.mean(rho)
        std_rho = np.std(rho)
        high_resistance_threshold = avg_rho + (2.5 * std_rho) # Top ~1%
        
        print(f"\n--- THRESHOLD CALCULATION ---")
        print(f"Average Rho: {avg_rho:.2f}")
        print(f"Std Dev Rho: {std_rho:.2f}")
        print(f"High-Resistance Threshold: {high_resistance_threshold:.2f}")
        
        # Split into Groups
        normal_indices = rho < high_resistance_threshold
        high_res_indices = rho >= high_resistance_threshold
        
        ratio_normal = ratios[normal_indices]
        ratio_high_res = ratios[high_res_indices]
        
        print(f"\n--- DATASET ---")
        print(f"Normal Frames: {len(ratio_normal)} ({len(ratio_normal)/filtered_events*100:.1f}%)")
        print(f"High-Resistance Frames: {len(ratio_high_res)} ({len(ratio_high_res)/filtered_events*100:.1f}%)")
        
        print(f"\n--- STABILITY CHECK (The 'J' Constant) ---")
        
        # Mean
        mean_norm = np.mean(ratio_normal)
        mean_high = np.mean(ratio_high_res)
        
        # Standard Deviation (Volatility)
        std_norm = np.std(ratio_normal)
        std_high = np.std(ratio_high_res)
        
        # Coefficient of Variation (Relative Volatility)
        cv_norm = (std_norm / mean_norm) * 100
        cv_high = (std_high / mean_high) * 100
        
        print(f"Normal Ratio: {mean_norm:.2f} (Volatility: {cv_norm:.2f}%)")
        print(f"High-Res Ratio: {mean_high:.2f} (Volatility: {cv_high:.2f}%)")
        
        print("-" * 40)
        if cv_high < cv_norm:
            print(">>> RESULT: SUPPORT FOR EST. The system becomes MORE STABLE under load.")
            print(f"Stability Improvement: {cv_norm - cv_high:.2f}%")
        else:
            print(">>> RESULT: NULL. The system becomes LESS STABLE (Chaotic) under load.")
        
        # Calculate p-value for significance
        print("\n--- STATISTICAL SIGNIFICANCE ---")
        t_stat, p_value = ttest_ind(ratio_normal, ratio_high_res, equal_var=False)
        print(f"T-statistic: {t_stat:.4f}")
        print(f"P-value: {p_value:.6f}")
        print(f"Significance: {'SIGNIFICANT' if p_value < 0.01 else 'NOT SIGNIFICANT'}")

        # Setup logger for structured logging (still goes to same file)
        logger = logging.getLogger('EST_Analysis')
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        if logger.hasHandlers():
            logger.handlers.clear()
        
        # File handler - writes to same file
        fh = logging.FileHandler(log_filename, mode='a')  # Append mode
        fh.setLevel(logging.INFO)
        
        # Format for logger
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        logger.addHandler(fh)
        
        # Log structured results
        logger.info("=== STRUCTURED ANALYSIS RESULTS ===")
        logger.info(f"Total events loaded: {n_events:,}")
        logger.info(f"Events after filtering: {filtered_events:,}")
        logger.info(f"Normal frames: {len(ratio_normal)}")
        logger.info(f"High-resistance frames: {len(ratio_high_res)}")
        logger.info(f"Normal Ratio Mean: {mean_norm:.2f}")
        logger.info(f"High-Res Ratio Mean: {mean_high:.2f}")
        logger.info(f"Normal Volatility (CV): {cv_norm:.2f}%")
        logger.info(f"High-Res Volatility (CV): {cv_high:.2f}%")
        logger.info(f"P-value: {p_value:.6f}")
        logger.info(f"Statistical Significance: {'SIGNIFICANT' if p_value < 0.01 else 'NOT SIGNIFICANT'}")
        if cv_high < cv_norm:
            logger.info(f"RESULT: SUPPORT FOR EST. Stability Improvement: {cv_norm - cv_high:.2f}%")
        else:
            logger.info("RESULT: NULL. System becomes LESS STABLE under load.")

        # Visualization
        print("\n--- VISUALIZATION ---")
        print("Generating plot...")
        plt.figure(figsize=(10, 6))
        plt.hist(ratio_normal, bins=50, density=True, alpha=0.5, color='gray', 
                label=f'Normal (Low Load, n={len(ratio_normal)})', range=(0, 200))
        plt.hist(ratio_high_res, bins=20, density=True, alpha=0.8, color='red', 
                label=f'Anomalies (High Load, n={len(ratio_high_res)})', range=(0, 200))
        plt.axvline(mean_high, color='black', linestyle='--', 
                   label=f'High Load Mean ({mean_high:.1f})')
        
        plt.title("Stability of the Vacuum-Energy Ratio")
        plt.xlabel("Cost Ratio (MET / Rho)")
        plt.ylabel("Probability Density")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_filename = "stability_test.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved as '{plot_filename}'")
        logger.info(f"Plot saved as '{plot_filename}'")
        
        print("\n--- ANALYSIS COMPLETE ---")
        print(f"Log file saved as: {log_filename}")
        logger.info("=== ANALYSIS COMPLETED ===")
        logger.info(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# After the TeeLogger context, print final message
print(f"\n[All output has been saved to: {log_filename}]")