import uproot
import numpy as np
import matplotlib.pyplot as plt
import time
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
        self.file = open(self.filename, 'w', encoding='utf-8')  # Add UTF-8 encoding
        
        # Create a custom stream that writes to both file and stdout
        class TeeStream(io.TextIOBase):
            def __init__(self, file, stdout):
                self.file = file
                self.stdout = stdout
                
            def write(self, data):
                self.file.write(data)
                # Encode for console output to avoid encoding issues
                try:
                    self.stdout.write(data)
                except UnicodeEncodeError:
                    # Replace non-ASCII characters for console
                    self.stdout.write(data.encode('ascii', 'replace').decode('ascii'))
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
log_filename = f'FullScale_Verification_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

# Start capturing ALL output
with TeeLogger(log_filename):
    file_path = "nano_data2016_42.root"
    
    print(f"=== FULL SCALE VERIFICATION: {file_path} ===")
    print(f"Log file: {log_filename}")
    print(f"Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    start_time = time.time()

    try:
        with uproot.open(file_path) as file:
            tree = file["Events"]
            
            # Load ALL data for the two relevant branches
            print("1. Loading ALL events...")
            print("   (This may take a moment depending on file size)")
            
            data_start = time.time()
            data = tree.arrays(
                ["fixedGridRhoFastjetCentralCalo", "CaloMET_sumEt"],
                library="np"
            )
            
            data_load_time = time.time() - data_start
            
            rho = data["fixedGridRhoFastjetCentralCalo"]
            met = data["CaloMET_sumEt"]
            
            print(f"   Raw events loaded: {len(rho):,}")
            print(f"   Data load time: {data_load_time:.2f}s")

            # Filter invalid data (Rho=0 causes divide-by-zero)
            filter_start = time.time()
            mask = rho > 0.1
            rho = rho[mask]
            met = met[mask]
            
            ratios = met / rho
            n_events = len(ratios)
            filter_time = time.time() - filter_start
            
            print(f"   Valid events after filtering: {n_events:,}")
            print(f"   Filter time: {filter_time:.2f}s")
            print(f"   Total load & filter time: {time.time() - start_time:.2f}s")
            
            # Basic statistics
            print(f"\n   Dataset Statistics:")
            print(f"     Rho: Mean={np.mean(rho):.2f}, Std={np.std(rho):.2f}")
            print(f"     MET: Mean={np.mean(met):.2f}, Std={np.std(met):.2f}")
            print(f"     Ratio: Mean={np.mean(ratios):.2f}, Std={np.std(ratios):.2f}")

            # 2. DEFINE GROUPS
            print(f"\n2. Defining Population Groups")
            avg_rho = np.mean(rho)
            std_rho = np.std(rho)
            
            # We stick to the 2.5 Sigma threshold to capture the top ~1% of "High Resistance" frames
            threshold = avg_rho + (2.5 * std_rho)
            
            # FIXED: Replace sigma character with 'sigma' text
            print(f"   Threshold Calculation:")
            print(f"     Average Rho: {avg_rho:.4f}")
            print(f"     Standard Deviation: {std_rho:.4f}")
            print(f"     Threshold (2.5 sigma): > {threshold:.4f} GeV")
            
            high_res_mask = rho >= threshold
            normal_mask = rho < threshold
            
            ratio_high = ratios[high_res_mask]
            ratio_norm = ratios[normal_mask]
            
            n_high = len(ratio_high)
            n_norm = len(ratio_norm)
            
            print(f"\n   Population Analysis:")
            print(f"     Normal Frames: {n_norm:,} ({n_norm/n_events*100:.1f}%)")
            print(f"     High-Res Frames: {n_high:,} ({n_high/n_events*100:.1f}%)")
            print(f"     Avg Rho (Normal): {np.mean(rho[normal_mask]):.2f}")
            print(f"     Avg Rho (High-Res): {np.mean(rho[high_res_mask]):.2f}")

            # 3. CALCULATE OBSERVED VOLATILITY (CV)
            # CV = (StdDev / Mean) * 100
            
            def get_cv(data):
                if len(data) == 0:
                    return 0
                mean_val = np.mean(data)
                if mean_val == 0:
                    return float('inf')
                return (np.std(data) / mean_val) * 100

            cv_norm = get_cv(ratio_norm)
            cv_high = get_cv(ratio_high)
            observed_diff = cv_norm - cv_high
            
            print(f"\n3. Observed Stability Analysis")
            print(f"   Normal Group:")
            print(f"     Mean Ratio: {np.mean(ratio_norm):.2f}")
            print(f"     Std Ratio: {np.std(ratio_norm):.2f}")
            print(f"     Volatility (CV): {cv_norm:.4f}%")
            
            print(f"\n   High-Resistance Group:")
            print(f"     Mean Ratio: {np.mean(ratio_high):.2f}")
            print(f"     Std Ratio: {np.std(ratio_high):.2f}")
            print(f"     Volatility (CV): {cv_high:.4f}%")
            
            print(f"\n   Stability Comparison:")
            print(f"     Volatility Difference: {observed_diff:.4f}%")
            print(f"     Relative Change: {(cv_high/cv_norm*100 - 100):.2f}%")
            
            if observed_diff > 0:
                print(f"\n   >>> SIGNAL CONFIRMED: High load creates stability.")
                print(f"   >>> Stability Gain: {observed_diff:.4f}%")
            else:
                print(f"\n   >>> SIGNAL LOST: Scale wiped out the effect.")
                print(f"   >>> Instability Increase: {-observed_diff:.4f}%")

            # 4. BOOTSTRAP VALIDATION (The "Luck" Test)
            print(f"\n4. Bootstrapping Analysis")
            print(f"   Starting bootstrap with 1000 iterations...")
            
            bootstrap_start = time.time()
            n_bootstrap = 1000
            simulated_diffs = []
            
            # We mix all ratios together to create a "Null Hypothesis" universe where Rho doesn't matter
            all_ratios = ratios
            
            for i in range(n_bootstrap):
                if i % 200 == 0 and i > 0:
                    elapsed = time.time() - bootstrap_start
                    eta = (elapsed / i) * (n_bootstrap - i)
                    print(f"     Progress: {i}/{n_bootstrap} ({i/n_bootstrap*100:.0f}%) - ETA: {eta:.1f}s", end="\r")
                
                # Pick random events to act as "fake" High-Res events
                fake_high = np.random.choice(all_ratios, size=n_high, replace=True)
                
                # Calculate CV for this random group
                cv_fake = get_cv(fake_high)
                
                # We calculate (Global CV - Fake CV) to match our observed metric
                sim_diff = cv_norm - cv_fake
                simulated_diffs.append(sim_diff)
            
            bootstrap_time = time.time() - bootstrap_start
            print(f"   Bootstrap completed in {bootstrap_time:.2f}s            ")
            
            simulated_diffs = np.array(simulated_diffs)
            
            # Calculate bootstrap statistics
            print(f"\n   Bootstrap Statistics:")
            print(f"     Mean simulated diff: {np.mean(simulated_diffs):.4f}%")
            print(f"     Std simulated diff: {np.std(simulated_diffs):.4f}%")
            print(f"     Min simulated diff: {np.min(simulated_diffs):.4f}%")
            print(f"     Max simulated diff: {np.max(simulated_diffs):.4f}%")

            # 5. STATISTICAL SIGNIFICANCE
            # How many times did random chance produce a stability gain as big as ours?
            p_value = np.sum(simulated_diffs >= observed_diff) / n_bootstrap
            
            # Calculate confidence interval
            sorted_diffs = np.sort(simulated_diffs)
            ci_lower = sorted_diffs[int(0.025 * n_bootstrap)]
            ci_upper = sorted_diffs[int(0.975 * n_bootstrap)]
            
            print(f"\n5. Statistical Significance")
            print(f"   Observed Stability Gain: {observed_diff:.4f}%")
            print(f"   95% Confidence Interval: [{ci_lower:.4f}%, {ci_upper:.4f}%]")
            print(f"   P-Value: {p_value:.6f}")
            
            # Determine significance level
            if observed_diff > ci_upper:
                significance = "EXTREMELY SIGNIFICANT"
            elif observed_diff > np.mean(simulated_diffs) + 2*np.std(simulated_diffs):
                significance = "HIGHLY SIGNIFICANT"
            elif observed_diff > np.mean(simulated_diffs) + np.std(simulated_diffs):
                significance = "SIGNIFICANT"
            else:
                significance = "NOT SIGNIFICANT"
            
            print(f"   Statistical Assessment: {significance}")
            
            if p_value < 0.001:
                print("   >>> EXTREMELY SIGNIFICANT (p < 0.001)")
            elif p_value < 0.01:
                print("   >>> HIGHLY SIGNIFICANT (p < 0.01) - arXiv Ready")
            elif p_value < 0.05:
                print("   >>> STATISTICALLY SIGNIFICANT (p < 0.05)")
            else:
                print("   >>> NOT STATISTICALLY SIGNIFICANT (p >= 0.05)")
            
            # Calculate Z-score
            z_score = (observed_diff - np.mean(simulated_diffs)) / np.std(simulated_diffs)
            print(f"   Z-score: {z_score:.2f}")

            # 6. VISUALIZATION
            print(f"\n6. Generating Visualizations...")
            
            # Create main bootstrap plot
            fig = plt.figure(figsize=(14, 10))
            
            # Plot 1: Bootstrap distribution
            ax1 = plt.subplot(2, 2, 1)
            ax1.hist(simulated_diffs, bins=30, color='gray', alpha=0.7, 
                    edgecolor='black', label='Random Chance (Null Hypothesis)')
            ax1.axvline(observed_diff, color='red', linestyle='-', linewidth=3, 
                       label=f'Observed Signal ({observed_diff:.2f}%)')
            ax1.axvline(np.mean(simulated_diffs), color='blue', linestyle='--', 
                       linewidth=2, label=f'Mean Random ({np.mean(simulated_diffs):.2f}%)')
            ax1.axvspan(ci_lower, ci_upper, alpha=0.2, color='blue', 
                       label='95% Confidence Interval')
            ax1.set_title(f"Bootstrap Analysis: Is the Stability Real?\n(N={n_events:,}, Iterations={n_bootstrap})")
            ax1.set_xlabel("Stability Gain (Normal CV - High CV) %")
            ax1.set_ylabel("Frequency")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Ratio distributions
            ax2 = plt.subplot(2, 2, 2)
            bins = np.linspace(0, 200, 100)
            ax2.hist(ratio_norm, bins=bins, alpha=0.5, color='blue', 
                    density=True, label=f'Normal (n={n_norm:,})')
            ax2.hist(ratio_high, bins=bins, alpha=0.7, color='red', 
                    density=True, label=f'High-Res (n={n_high:,})')
            ax2.axvline(np.mean(ratio_norm), color='blue', linestyle='--', linewidth=2)
            ax2.axvline(np.mean(ratio_high), color='red', linestyle='--', linewidth=2)
            ax2.set_title("Ratio Distributions by Group")
            ax2.set_xlabel("MET / Rho Ratio")
            ax2.set_ylabel("Density")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Rho vs Ratio scatter
            ax3 = plt.subplot(2, 2, 3)
            sample_size = min(5000, len(rho))
            indices = np.random.choice(len(rho), sample_size, replace=False)
            scatter = ax3.scatter(rho[indices], ratios[indices], alpha=0.3, s=1, color='green')
            ax3.axvline(threshold, color='orange', linestyle='--', linewidth=2, 
                       label=f'Threshold ({threshold:.1f})')
            ax3.set_title(f"Rho vs Ratio (Random Sample: {sample_size:,} points)")
            ax3.set_xlabel("Rho (Energy Density)")
            ax3.set_ylabel("MET / Rho Ratio")
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Cumulative distribution
            ax4 = plt.subplot(2, 2, 4)
            sorted_sim = np.sort(simulated_diffs)
            cumul_prob = np.arange(1, len(sorted_sim)+1) / len(sorted_sim)
            ax4.plot(sorted_sim, cumul_prob, 'b-', linewidth=2, label='Cumulative Distribution')
            ax4.axvline(observed_diff, color='red', linestyle='-', linewidth=2, 
                       label=f'Observed ({observed_diff:.2f}%)')
            ax4.axhline(p_value, color='green', linestyle='--', linewidth=1, 
                       label=f'P-value ({p_value:.4f})')
            ax4.set_title("Cumulative Probability Distribution")
            ax4.set_xlabel("Stability Gain (%)")
            ax4.set_ylabel("Cumulative Probability")
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_filename = "full_scale_bootstrap.png"
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   Main plot saved as '{plot_filename}'")
            
            # Create summary plot
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            # Bar chart comparison
            categories = ['Normal', 'High-Resistance']
            cv_values = [cv_norm, cv_high]
            mean_values = [np.mean(ratio_norm), np.mean(ratio_high)]
            
            x_pos = np.arange(len(categories))
            width = 0.35
            
            # Plot CV values
            bars1 = ax1.bar(x_pos - width/2, cv_values, width, label='Volatility (CV)', 
                          color=['blue', 'red'], alpha=0.7)
            ax1.set_ylabel('Coefficient of Variation (%)', color='black')
            ax1.tick_params(axis='y', labelcolor='black')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(categories)
            
            # Create second y-axis for mean ratios
            ax2 = ax1.twinx()
            bars2 = ax2.bar(x_pos + width/2, mean_values, width, label='Mean Ratio', 
                          color=['lightblue', 'pink'], alpha=0.7)
            ax2.set_ylabel('Mean MET/Rho Ratio', color='black')
            ax2.tick_params(axis='y', labelcolor='black')
            
            # Add value labels
            for i, (cv, mean) in enumerate(zip(cv_values, mean_values)):
                ax1.text(i - width/2, cv + 0.1, f'{cv:.1f}%', ha='center', va='bottom')
                ax2.text(i + width/2, mean + 0.1, f'{mean:.1f}', ha='center', va='bottom')
            
            # Add title
            plt.title(f"Stability Analysis Summary\nTotal Events: {n_events:,}, High-Res: {n_high:,} ({n_high/n_events*100:.1f}%)")
            
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            summary_filename = "stability_summary.png"
            plt.savefig(summary_filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   Summary plot saved as '{summary_filename}'")
            
            print(f"\n7. Analysis Complete")
            total_time = time.time() - start_time
            print(f"   Total processing time: {total_time:.2f}s")
            print(f"   Files created:")
            print(f"     - {plot_filename}")
            print(f"     - {summary_filename}")
            print(f"     - {log_filename}")
            
            print(f"\n=== FINAL VERDICT ===")
            print(f"Dataset Size: {n_events:,} events")
            print(f"High-Resistance Events: {n_high:,} events ({n_high/n_events*100:.2f}%)")
            print(f"Stability Gain: {observed_diff:.4f}%")
            print(f"P-value: {p_value:.6f}")
            print(f"Significance: {significance}")
            print(f"Z-score: {z_score:.2f}")
            
            if p_value < 0.05 and observed_diff > 0:
                print(f">>> CONCLUSION: EST HYPOTHESIS SUPPORTED")
                print(f">>> The system shows increased stability under high load")
            elif observed_diff > 0:
                print(f">>> CONCLUSION: TREND OBSERVED BUT NOT STATISTICALLY SIGNIFICANT")
                print(f">>> Further investigation recommended")
            else:
                print(f">>> CONCLUSION: NO SUPPORT FOR EST HYPOTHESIS")
                print(f">>> The system does not show increased stability under high load")

    except FileNotFoundError:
        print(f"ERROR: File '{file_path}' not found!")
        print("Please check the file path and try again.")
    except KeyError as e:
        print(f"ERROR: Branch {e} not found in the ROOT file!")
        print("Available branches may be different than expected.")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        print("\nTraceback:")
        traceback.print_exc()

print(f"\n[All full-scale verification output saved to: {log_filename}]")