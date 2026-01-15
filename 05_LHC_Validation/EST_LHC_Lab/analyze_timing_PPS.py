import uproot
import numpy as np
import matplotlib.pyplot as plt
import sys
import io
from datetime import datetime
import warnings
import time

# Suppress specific warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

class TeeLogger:
    """Class to capture both print statements and logger output to file"""
    def __init__(self, filename):
        self.filename = filename
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        
    def __enter__(self):
        self.file = open(self.filename, 'w', encoding='utf-8')
        
        class TeeStream(io.TextIOBase):
            def __init__(self, file, stdout):
                self.file = file
                self.stdout = stdout
                
            def write(self, data):
                self.file.write(data)
                try:
                    self.stdout.write(data)
                except UnicodeEncodeError:
                    self.stdout.write(data.encode('ascii', 'replace').decode('ascii'))
                return len(data)
                
            def flush(self):
                self.file.flush()
                self.stdout.flush()
        
        sys.stdout = TeeStream(self.file, self.stdout)
        sys.stderr = TeeStream(self.file, self.stderr)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close()

# Create log filename with timestamp
log_filename = f'PPS_Timing_Analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

# Start capturing ALL output
with TeeLogger(log_filename):
    file_path = "nano_data2016_1-9.root"
    
    print(f"=== PPS TIMING ANALYSIS: Alternative Timing Data ===")
    print(f"File: {file_path}")
    print(f"Log file: {log_filename}")
    print(f"Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    start_time = time.time()

    try:
        with uproot.open(file_path) as file:
            tree = file["Events"]
            
            # 1. INVESTIGATE PPS TIMING BRANCHES
            print("1. Investigating PPS Timing Branches...")
            
            pps_branches = ['PPSLocalTrack_time', 'PPSLocalTrack_timeUnc']
            
            # Test load a small sample
            print("   Testing PPS branches with small sample...")
            test_data = tree.arrays(
                pps_branches + ["fixedGridRhoFastjetCentralCalo"],
                library="np",
                entry_stop=100
            )
            
            rho_sample = test_data["fixedGridRhoFastjetCentralCalo"]
            pps_time_sample = test_data["PPSLocalTrack_time"]
            pps_unc_sample = test_data["PPSLocalTrack_timeUnc"]
            
            print(f"   Sample size: {len(rho_sample)} events")
            
            # Analyze PPS timing data structure
            print("\n   Analyzing PPS timing data structure...")
            
            valid_pps_times = []
            valid_pps_uncs = []
            events_with_pps = 0
            
            for i in range(len(pps_time_sample)):
                pps_times = pps_time_sample[i]
                pps_uncs = pps_unc_sample[i]
                
                # Check for valid PPS timing data (not -999 or similar)
                valid_indices = (pps_times > -900) & (pps_times < 900)  # Reasonable time range
                
                if np.any(valid_indices):
                    events_with_pps += 1
                    valid_pps_times.extend(pps_times[valid_indices])
                    valid_pps_uncs.extend(pps_uncs[valid_indices])
                
                if i < 5 and np.any(valid_indices):
                    print(f"     Event {i}: {np.sum(valid_indices)} valid PPS times")
                    print(f"          Times: {pps_times[valid_indices][:3]}...")  # First 3
            
            print(f"\n   PPS Timing Data Summary:")
            print(f"     Events with valid PPS data: {events_with_pps}/{len(rho_sample)}")
            print(f"     Total valid PPS measurements: {len(valid_pps_times)}")
            
            if len(valid_pps_times) > 0:
                print(f"     PPS Time Range: [{np.min(valid_pps_times):.2f}, {np.max(valid_pps_times):.2f}] ns")
                print(f"     Mean PPS Time: {np.mean(valid_pps_times):.4f} ± {np.std(valid_pps_times):.4f} ns")
                print(f"     Mean PPS Uncertainty: {np.mean(valid_pps_uncs):.4f} ns")
                
                # Check if PPS times look reasonable
                if np.abs(np.mean(valid_pps_times)) > 100:
                    print(f"\n   WARNING: PPS times seem unusually large (>100 ns)")
                    print(f"   This might indicate a different time scale or calibration.")
                else:
                    print(f"\n   SUCCESS: Found valid PPS timing data!")
            else:
                print(f"\n   WARNING: No valid PPS timing data found.")
                print(f"   Trying alternative approach...")
                
                # Try loading larger sample in case PPS data is sparse
                print("   Loading larger sample (1000 events)...")
                larger_test = tree.arrays(
                    ["PPSLocalTrack_time", "fixedGridRhoFastjetCentralCalo"],
                    library="np",
                    entry_stop=1000
                )
                
                pps_times_large = larger_test["PPSLocalTrack_time"]
                all_pps_times = []
                
                for times in pps_times_large:
                    # Flatten and filter
                    flat_times = times.flatten()
                    valid = flat_times[(flat_times > -900) & (flat_times < 900)]
                    all_pps_times.extend(valid)
                
                print(f"   Found {len(all_pps_times)} valid PPS times in 1000 events")
                
                if len(all_pps_times) == 0:
                    print(f"\n   CRITICAL: No valid PPS timing data available.")
                    print(f"   All PPS times appear to be placeholder values.")
                    print(f"\n   Alternative timing sources exhausted.")
                    print(f"   This dataset does not contain usable timing information.")
                    print(f"\n   SUGGESTIONS:")
                    print(f"     1. Use a different dataset with timing information")
                    print(f"     2. Contact data providers about timing availability")
                    print(f"     3. Focus on non-timing analyses (stability, correlations)")
                    
                    # Exit gracefully
                    print(f"\n=== ANALYSIS TERMINATED ===")
                    print(f"Reason: No valid timing data in dataset.")
                    
                    # Create diagnostic report
                    diagnostic_data = {
                        'Proton_multiRP_time': 'All values = -999 (placeholder)',
                        'PPSLocalTrack_time': 'No valid values found',
                        'Recommendation': 'Use different dataset for timing analysis'
                    }
                    
                    print(f"\nDiagnostic Summary:")
                    for key, value in diagnostic_data.items():
                        print(f"  {key}: {value}")
                    
                    total_time = time.time() - start_time
                    print(f"\nTotal processing time: {total_time:.2f}s")
                    
                    # Still create a simple diagnostic plot
                    plt.figure(figsize=(10, 6))
                    plt.text(0.5, 0.5, 'NO VALID TIMING DATA\nIN DATASET\n\n- Proton_multiRP_time: All -999\n- PPSLocalTrack_time: No valid values', 
                            ha='center', va='center', fontsize=14)
                    plt.title("Timing Data Diagnostic")
                    plt.axis('off')
                    diagnostic_filename = "timing_data_unavailable.png"
                    plt.savefig(diagnostic_filename, dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"\nDiagnostic plot saved as '{diagnostic_filename}'")
                    
                    raise SystemExit(0)
            
            # 2. LOAD FULL DATASET FOR PPS ANALYSIS
            print(f"\n2. Loading full dataset for PPS timing analysis...")
            
            # Load a reasonable number of events (not all 447k to avoid memory issues)
            sample_size = min(50000, tree.num_entries)
            print(f"   Loading {sample_size:,} events...")
            
            data = tree.arrays(
                ["fixedGridRhoFastjetCentralCalo", "PPSLocalTrack_time", "PPSLocalTrack_timeUnc"],
                library="np",
                entry_stop=sample_size
            )
            
            rho = data["fixedGridRhoFastjetCentralCalo"]
            pps_times = data["PPSLocalTrack_time"]
            pps_uncs = data["PPSLocalTrack_timeUnc"]
            
            print(f"   Loaded {len(rho):,} events")
            
            # 3. DEFINE GROUPS BASED ON RHO
            avg_rho = np.mean(rho)
            std_rho = np.std(rho)
            threshold = avg_rho + (3 * std_rho)  # 3-sigma threshold for anomalies
            
            print(f"\n3. Defining Groups")
            print(f"   Average Rho: {avg_rho:.4f} GeV")
            print(f"   Standard Deviation: {std_rho:.4f} GeV")
            print(f"   Anomaly Threshold (3σ): {threshold:.4f} GeV")
            
            # 4. PROCESS PPS TIMING DATA
            print(f"\n4. Processing PPS timing data...")
            
            normal_pps_times = []
            anomaly_pps_times = []
            normal_pps_uncs = []
            anomaly_pps_uncs = []
            
            normal_events_with_pps = 0
            anomaly_events_with_pps = 0
            
            for i in range(len(rho)):
                # Get PPS times for this event
                event_times = pps_times[i]
                event_uncs = pps_uncs[i]
                
                # Filter valid times
                valid_mask = (event_times > -900) & (event_times < 900)
                valid_times = event_times[valid_mask]
                valid_uncs = event_uncs[valid_mask]
                
                if len(valid_times) > 0:
                    # Take mean of valid PPS times for this event
                    avg_pps_time = np.mean(valid_times)
                    avg_pps_unc = np.mean(valid_uncs) if len(valid_uncs) > 0 else 0
                    
                    if rho[i] > threshold:
                        anomaly_pps_times.append(avg_pps_time)
                        anomaly_pps_uncs.append(avg_pps_unc)
                        anomaly_events_with_pps += 1
                    else:
                        normal_pps_times.append(avg_pps_time)
                        normal_pps_uncs.append(avg_pps_unc)
                        normal_events_with_pps += 1
            
            # Convert to numpy arrays
            normal_pps_times = np.array(normal_pps_times)
            anomaly_pps_times = np.array(anomaly_pps_times)
            normal_pps_uncs = np.array(normal_pps_uncs)
            anomaly_pps_uncs = np.array(anomaly_pps_uncs)
            
            print(f"   Normal events with PPS timing: {normal_events_with_pps:,}")
            print(f"   Anomalous events with PPS timing: {anomaly_events_with_pps:,}")
            
            # 5. STATISTICAL ANALYSIS
            print(f"\n5. Statistical Analysis of PPS Timing")
            
            if len(normal_pps_times) > 0:
                print(f"\n   NORMAL EVENTS (PPS Timing):")
                print(f"     Mean PPS Time: {np.mean(normal_pps_times):.4f} ns")
                print(f"     Std Dev:       {np.std(normal_pps_times):.4f} ns")
                print(f"     Mean Uncertainty: {np.mean(normal_pps_uncs):.4f} ns")
                print(f"     N = {len(normal_pps_times):,}")
            
            if len(anomaly_pps_times) > 0:
                print(f"\n   ANOMALOUS EVENTS (PPS Timing):")
                print(f"     Mean PPS Time: {np.mean(anomaly_pps_times):.4f} ns")
                print(f"     Std Dev:       {np.std(anomaly_pps_times):.4f} ns")
                print(f"     Mean Uncertainty: {np.mean(anomaly_pps_uncs):.4f} ns")
                print(f"     N = {len(anomaly_pps_times):,}")
                
                # Calculate time shift
                time_shift = np.mean(anomaly_pps_times) - np.mean(normal_pps_times)
                shift_error = np.sqrt(
                    (np.std(normal_pps_times)**2 / len(normal_pps_times)) +
                    (np.std(anomaly_pps_times)**2 / len(anomaly_pps_times))
                )
                
                print(f"\n   >>> PPS TIME SHIFT ANALYSIS <<<")
                print(f"     Shift (Anomaly - Normal): {time_shift:.4f} ± {shift_error:.4f} ns")
                print(f"     Relative Shift: {time_shift/np.abs(np.mean(normal_pps_times))*100:.2f}%")
                
                # Statistical significance test
                from scipy.stats import ttest_ind
                t_stat, p_value = ttest_ind(normal_pps_times, anomaly_pps_times, equal_var=False)
                
                print(f"     T-statistic: {t_stat:.4f}")
                print(f"     P-value: {p_value:.6f}")
                
                if p_value < 0.001:
                    significance = "EXTREMELY SIGNIFICANT (***)"
                elif p_value < 0.01:
                    significance = "HIGHLY SIGNIFICANT (**)"
                elif p_value < 0.05:
                    significance = "SIGNIFICANT (*)"
                else:
                    significance = "NOT SIGNIFICANT"
                
                print(f"     Significance: {significance}")
                
            else:
                print(f"\n   WARNING: No PPS timing data for anomalous events")
                print(f"   Cannot perform timing shift analysis.")
            
            # 6. CORRELATION ANALYSIS
            print(f"\n6. Correlation Analysis")
            
            # Collect all data for correlation
            all_rhos = []
            all_pps_times = []
            
            for i in range(len(rho)):
                event_times = pps_times[i]
                valid_mask = (event_times > -900) & (event_times < 900)
                valid_times = event_times[valid_mask]
                
                if len(valid_times) > 0:
                    avg_time = np.mean(valid_times)
                    all_rhos.append(rho[i])
                    all_pps_times.append(avg_time)
            
            if len(all_rhos) > 10:  # Need enough points for correlation
                correlation = np.corrcoef(all_rhos, all_pps_times)[0, 1]
                print(f"   Correlation (Rho vs PPS Time): {correlation:.4f}")
                
                if abs(correlation) > 0.3:
                    print(f"   >>> Moderate correlation detected")
                elif abs(correlation) > 0.1:
                    print(f"   >>> Weak correlation detected")
                else:
                    print(f"   >>> Very weak or no correlation")
            
            # 7. VISUALIZATION
            print(f"\n7. Generating Visualizations...")
            
            if len(normal_pps_times) > 0 or len(anomaly_pps_times) > 0:
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                
                # Plot 1: PPS Time distributions
                if len(normal_pps_times) > 0:
                    axes[0, 0].hist(normal_pps_times, bins=30, density=True, alpha=0.5,
                                   color='blue', label=f'Normal (n={len(normal_pps_times):,})')
                
                if len(anomaly_pps_times) > 0:
                    axes[0, 0].hist(anomaly_pps_times, bins=15, density=True, alpha=0.7,
                                   color='red', label=f'Anomalous (n={len(anomaly_pps_times):,})')
                
                axes[0, 0].set_title("PPS Local Track Time Distribution")
                axes[0, 0].set_xlabel("PPS Time (ns)")
                axes[0, 0].set_ylabel("Density")
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                
                # Plot 2: Time vs Rho scatter
                if len(all_rhos) > 0:
                    scatter = axes[0, 1].scatter(all_rhos, all_pps_times, alpha=0.3, s=10, c='green')
                    axes[0, 1].axvline(threshold, color='orange', linestyle='--',
                                      label=f'Threshold ({threshold:.1f} GeV)')
                    axes[0, 1].set_title("PPS Time vs Vacuum Energy Density")
                    axes[0, 1].set_xlabel("Rho (GeV)")
                    axes[0, 1].set_ylabel("PPS Time (ns)")
                    axes[0, 1].legend()
                    axes[0, 1].grid(True, alpha=0.3)
                
                # Plot 3: Box plot comparison
                if len(normal_pps_times) > 0 and len(anomaly_pps_times) > 0:
                    data_to_plot = [normal_pps_times, anomaly_pps_times]
                    bp = axes[1, 0].boxplot(data_to_plot, labels=['Normal', 'Anomalous'],
                                           patch_artist=True)
                    bp['boxes'][0].set_facecolor('blue')
                    bp['boxes'][1].set_facecolor('red')
                    axes[1, 0].set_title("PPS Time Distribution Comparison")
                    axes[1, 0].set_ylabel("PPS Time (ns)")
                    axes[1, 0].grid(True, alpha=0.3)
                
                # Plot 4: Time uncertainty
                if len(normal_pps_uncs) > 0 and len(anomaly_pps_uncs) > 0:
                    axes[1, 1].hist(normal_pps_uncs, bins=20, alpha=0.5, color='blue',
                                   label=f'Normal (mean={np.mean(normal_pps_uncs):.3f})')
                    axes[1, 1].hist(anomaly_pps_uncs, bins=10, alpha=0.7, color='red',
                                   label=f'Anomalous (mean={np.mean(anomaly_pps_uncs):.3f})')
                    axes[1, 1].set_title("PPS Time Uncertainty Distribution")
                    axes[1, 1].set_xlabel("Time Uncertainty (ns)")
                    axes[1, 1].set_ylabel("Count")
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                pps_plot_filename = "pps_timing_analysis.png"
                plt.savefig(pps_plot_filename, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"   PPS timing analysis plot saved as '{pps_plot_filename}'")
            
            print(f"\n8. Analysis Complete")
            total_time = time.time() - start_time
            print(f"   Total processing time: {total_time:.2f}s")
            print(f"   Log file: {log_filename}")
            
            if len(anomaly_pps_times) > 0:
                print(f"\n=== KEY PPS TIMING FINDINGS ===")
                print(f"PPS Time Shift: {time_shift:.4f} ± {shift_error:.4f} ns")
                print(f"Statistical Significance: {significance}")
                print(f"P-value: {p_value:.6f}")
                
                if p_value < 0.05 and time_shift != 0:
                    print(f"\n>>> RESULT: TIMING EFFECT DETECTED IN PPS DATA")
                    print(f">>> Anomalous events show different timing characteristics")
                else:
                    print(f"\n>>> RESULT: NO SIGNIFICANT TIMING EFFECT IN PPS DATA")
            
    except SystemExit:
        pass  # Graceful exit from earlier
    except FileNotFoundError:
        print(f"ERROR: File '{file_path}' not found!")
    except KeyError as e:
        print(f"ERROR: Branch {e} not found!")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        print("\nTraceback:")
        traceback.print_exc()

print(f"\n[All PPS timing analysis output saved to: {log_filename}]")