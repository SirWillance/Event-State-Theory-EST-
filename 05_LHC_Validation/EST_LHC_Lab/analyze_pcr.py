import uproot
import numpy as np
import matplotlib.pyplot as plt
import sys
import io
from datetime import datetime

class TeeLogger:
    """Class to capture both print statements and logger output to file"""
    def __init__(self, filename):
        self.filename = filename
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        
    def __enter__(self):
        self.file = open(self.filename, 'w')
        
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
log_filename = f'PCR_Analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

# Start capturing ALL output
with TeeLogger(log_filename):
    file_path = "nano_data2016_42.root"
    
    print(f"=== Deep Analysis: {file_path} ===")
    print(f"Log file: {log_filename}")
    print(f"Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    try:
        with uproot.open(file_path) as file:
            tree = file["Events"]
            
            # 1. LOAD THE TARGET DATA
            # We are grabbing the Vacuum Density and the Proton Timing
            print("1. Loading data branches...")
            
            # Using .arrays() to get numpy arrays. 
            # library="np" ensures we get standard numpy arrays.
            data = tree.arrays(
                ["fixedGridRhoFastjetCentralCalo", "CaloMET_sumEt"],
                library="np",
                entry_stop=10000 # Let's analyze the first 10k events first to be fast
            )
            
            rho = data["fixedGridRhoFastjetCentralCalo"]
            met = data["CaloMET_sumEt"]
            
            print(f"   Loaded {len(rho):,} events.")
            
            # 2. STATISTICAL ANALYSIS OF THE VACUUM
            # We are looking for "Anomalous Spikes" in the background density (Rho)
            
            avg_rho = np.mean(rho)
            std_rho = np.std(rho)
            
            print(f"\n2. Vacuum Energy Statistics")
            print(f"   Average Background Energy Density (Rho): {avg_rho:.4f}")
            print(f"   Standard Deviation: {std_rho:.4f}")
            
            # Define a "Pre-Causal Spike" as anything 3 Sigma above the mean
            threshold = avg_rho + (3 * std_rho)
            print(f"   Anomaly Threshold (3 Sigma): {threshold:.4f}")
            
            anomalies = rho[rho > threshold]
            anomaly_percentage = (len(anomalies) / len(rho)) * 100
            print(f"   Number of Anomalous Vacuum Events: {len(anomalies):,}")
            print(f"   Percentage of Total Events: {anomaly_percentage:.2f}%")
            
            # Additional anomaly statistics
            if len(anomalies) > 0:
                print(f"   Max Anomaly Value: {np.max(anomalies):.4f}")
                print(f"   Min Anomaly Value: {np.min(anomalies):.4f}")
                print(f"   Mean Anomaly Value: {np.mean(anomalies):.4f}")
            
            # 3. CORRELATION CHECK
            # Does high vacuum energy correlate with high Missing Energy (MET)?
            # If EST is right, the "ripple" might create a ghost signal in MET.
            
            correlation = np.corrcoef(rho, met)[0, 1]
            print(f"\n3. Correlation Analysis")
            print(f"   Correlation between Vacuum Density and Missing Energy: {correlation:.4f}")
            
            # Interpret correlation
            if abs(correlation) > 0.5:
                print(f"   >>> Strong correlation detected")
            elif abs(correlation) > 0.3:
                print(f"   >>> Moderate correlation detected")
            elif abs(correlation) > 0.1:
                print(f"   >>> Weak correlation detected")
            else:
                print(f"   >>> Very weak or no correlation")
            
            # 4. RATIO ANALYSIS (MET/Rho)
            print(f"\n4. Ratio Analysis (MET/Rho)")
            ratios = met / rho
            avg_ratio = np.mean(ratios)
            std_ratio = np.std(ratios)
            cv_ratio = (std_ratio / avg_ratio) * 100
            
            print(f"   Average MET/Rho Ratio: {avg_ratio:.2f}")
            print(f"   Ratio Standard Deviation: {std_ratio:.2f}")
            print(f"   Coefficient of Variation: {cv_ratio:.2f}%")
            
            # 5. SUMMARY STATISTICS
            print(f"\n5. Summary Statistics")
            print(f"   Rho Statistics:")
            print(f"     Min: {np.min(rho):.4f}, Max: {np.max(rho):.4f}")
            print(f"     25th Percentile: {np.percentile(rho, 25):.4f}")
            print(f"     Median: {np.median(rho):.4f}")
            print(f"     75th Percentile: {np.percentile(rho, 75):.4f}")
            
            print(f"\n   MET Statistics:")
            print(f"     Min: {np.min(met):.2f}, Max: {np.max(met):.2f}")
            print(f"     Mean: {np.mean(met):.2f}, Std: {np.std(met):.2f}")

            # 6. VISUALIZATION
            # Let's plot the Vacuum Density to see if it looks like random noise or structured ripples.
            
            print(f"\n6. Generating Visualization...")
            plt.figure(figsize=(12, 6))
            
            # Plot first 500 events
            plt.plot(rho[:500], label="Vacuum Density (Rho)", color="cyan", linewidth=0.8)
            plt.axhline(y=avg_rho, color='r', linestyle='--', label=f"Mean ({avg_rho:.2f})")
            plt.axhline(y=threshold, color='orange', linestyle='--', 
                       label=f"Anomaly Threshold ({threshold:.2f})")
            
            # Mark anomalies in the first 500 events
            anomaly_indices = np.where(rho[:500] > threshold)[0]
            if len(anomaly_indices) > 0:
                plt.scatter(anomaly_indices, rho[:500][anomaly_indices], 
                          color='red', s=50, zorder=5, label='Anomalies')
            
            plt.title("Vacuum Energy Density (First 500 Events)")
            plt.xlabel("Event Number")
            plt.ylabel("Energy Density (GeV)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save the plot
            plot_filename = "vacuum_analysis.png"
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   Plot saved as '{plot_filename}'")
            
            # 7. ADDITIONAL PLOT: Histogram of Rho values
            print(f"   Generating histogram...")
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.hist(rho, bins=100, alpha=0.7, color='cyan', edgecolor='black')
            plt.axvline(threshold, color='orange', linestyle='--', linewidth=2, 
                       label=f'3σ Threshold ({threshold:.2f})')
            plt.axvline(avg_rho, color='red', linestyle='-', linewidth=1, 
                       label=f'Mean ({avg_rho:.2f})')
            plt.title("Distribution of Vacuum Energy Density (Rho)")
            plt.xlabel("Energy Density (GeV)")
            plt.ylabel("Frequency")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.hist(ratios, bins=100, alpha=0.7, color='green', edgecolor='black', range=(0, 200))
            plt.title("Distribution of MET/Rho Ratio")
            plt.xlabel("MET / Rho Ratio")
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.3)
            
            histogram_filename = "vacuum_distributions.png"
            plt.tight_layout()
            plt.savefig(histogram_filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   Histogram saved as '{histogram_filename}'")
            
            print(f"\n7. Analysis Complete")
            print(f"   Total processing time: N/A")
            print(f"   Files created:")
            print(f"     - {plot_filename}")
            print(f"     - {histogram_filename}")
            print(f"     - {log_filename}")

    except FileNotFoundError:
        print(f"ERROR: File '{file_path}' not found!")
        print("Please check the file path and try again.")
    except KeyError as e:
        print(f"ERROR: Branch {e} not found in the ROOT file!")
        print("Available branches may be different than expected.")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        print("Traceback:")
        traceback.print_exc()

print(f"\n[All PCR analysis output saved to: {log_filename}]")