import uproot
import numpy as np
import matplotlib.pyplot as plt
import sys
import io
from datetime import datetime
import warnings

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
log_filename = f'Timing_Analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

# Start capturing ALL output
with TeeLogger(log_filename):
    file_path = "nano_data2016_1-9.root"
    
    print(f"=== TIMING ANALYSIS: Moment of Truth ===")
    print(f"File: {file_path}")
    print(f"Log file: {log_filename}")
    print(f"Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    try:
        with uproot.open(file_path) as file:
            tree = file["Events"]
            
            # 1. INVESTIGATE AVAILABLE BRANCHES
            print("1. Investigating available branches...")
            print("   Available branches (partial list):")
            branches_list = list(tree.keys())[:20]  # Show first 20 branches
            for i, branch in enumerate(branches_list):
                print(f"     {i+1:2d}. {branch}")
            
            # Check specifically for timing-related branches
            print("\n   Looking for timing-related branches...")
            timing_keywords = ['time', 'Time', 'TIME', 'proton', 'Proton', 'PROTON', 'rp', 'RP']
            timing_branches = []
            for branch in tree.keys():
                if any(keyword in branch for keyword in timing_keywords):
                    timing_branches.append(branch)
            
            if timing_branches:
                print("   Found timing-related branches:")
                for branch in timing_branches:
                    print(f"     - {branch}")
            else:
                print("   WARNING: No obvious timing branches found!")
            
            # 2. LOAD DATA
            print("\n2. Loading data...")
            
            # First, let's see what's actually in the proton timing branch
            print("   Examining 'Proton_multiRP_time' branch structure...")
            
            # Load a small sample to understand the data structure
            test_data = tree.arrays(
                ["fixedGridRhoFastjetCentralCalo", "Proton_multiRP_time"],
                library="np",
                entry_stop=100
            )
            
            rho_sample = test_data["fixedGridRhoFastjetCentralCalo"]
            time_sample = test_data["Proton_multiRP_time"]
            
            print(f"   Sample size: {len(rho_sample)} events")
            
            # Analyze the structure of timing data
            valid_times_count = 0
            total_protons = 0
            time_values = []
            
            for i in range(len(time_sample)):
                proton_times = time_sample[i]
                total_protons += len(proton_times)
                
                # Filter out -999 values
                valid_times = [t for t in proton_times if t > -900]
                valid_times_count += len(valid_times)
                time_values.extend(valid_times)
                
                if i < 5:  # Show first 5 events as examples
                    print(f"     Event {i}: {len(proton_times)} protons, valid times: {len(valid_times)}")
                    if len(valid_times) > 0:
                        print(f"          Valid times: {valid_times}")
            
            print(f"\n   Timing Data Summary (first 100 events):")
            print(f"     Total protons recorded: {total_protons}")
            print(f"     Valid proton times (t > -900): {valid_times_count}")
            print(f"     Invalid timing entries (-999): {total_protons - valid_times_count}")
            print(f"     Percentage valid: {valid_times_count/max(total_protons, 1)*100:.1f}%")
            
            if valid_times_count == 0:
                print(f"\n   CRITICAL: No valid timing data found in sample!")
                print(f"   All proton times appear to be placeholder values (-999).")
                print(f"   Timing analysis cannot proceed with this dataset.")
                print(f"\n   Possible reasons:")
                print(f"     1. Timing information was not recorded in this dataset")
                print(f"     2. The timing branch contains only placeholder values")
                print(f"     3. A different branch name contains timing data")
                
                # Look for alternative timing branches
                print(f"\n   Alternative timing search...")
                all_branches = list(tree.keys())
                time_patterns = ['time', 'Time', 't', 'T']
                
                for branch in all_branches[:50]:  # Check first 50 branches
                    if any(pattern in branch.lower() for pattern in time_patterns):
                        # Load a small sample
                        try:
                            test_branch = tree.arrays([branch], library="np", entry_stop=10)
                            values = test_branch[branch]
                            if hasattr(values[0], '__len__'):
                                # Jagged array
                                flat_vals = []
                                for v in values:
                                    flat_vals.extend(v)
                                flat_vals = np.array(flat_vals)
                            else:
                                flat_vals = values
                            
                            # Check if contains real values (not just -999)
                            real_vals = flat_vals[flat_vals > -900]
                            if len(real_vals) > 0:
                                print(f"     Found alternative: '{branch}' with {len(real_vals)} real values")
                        except:
                            pass
                
                print(f"\n   SUGGESTION: Check the dataset documentation or contact")
                print(f"   the data provider for timing information availability.")
                
                # Create a diagnostic plot showing the issue
                plt.figure(figsize=(10, 6))
                plt.hist([-999] * 100, bins=1, alpha=0.5, color='red', label='All Times = -999 (Invalid)')
                plt.title("Timing Data Diagnostic: All Values are -999 (Placeholder)")
                plt.xlabel("Proton Time (ns)")
                plt.ylabel("Frequency")
                plt.legend()
                plt.grid(True, alpha=0.3)
                diagnostic_filename = "timing_diagnostic.png"
                plt.savefig(diagnostic_filename, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"\n   Diagnostic plot saved as '{diagnostic_filename}'")
                
                # Exit gracefully
                print(f"\n=== ANALYSIS TERMINATED ===")
                print(f"Reason: No valid timing data available in dataset.")
                
            else:
                # Proceed with full analysis if we found valid data
                print(f"\n   Proceeding with full analysis...")
                
                # Load full dataset
                print("   Loading complete dataset...")
                branches = tree.arrays(
                    ["fixedGridRhoFastjetCentralCalo", "Proton_multiRP_time"],
                    library="np",
                    entry_stop=10000
                )
                
                rho = branches["fixedGridRhoFastjetCentralCalo"]
                proton_times = branches["Proton_multiRP_time"]
                
                print(f"   Loaded {len(rho):,} events")
                
                # 3. DEFINE GROUPS WITH PROPER FILTERING
                avg_rho = np.mean(rho)
                std_rho = np.std(rho)
                threshold = avg_rho + (3 * std_rho)
                
                print(f"\n3. Defining Groups")
                print(f"   Average Rho: {avg_rho:.4f} GeV")
                print(f"   Standard Deviation: {std_rho:.4f} GeV")
                print(f"   Anomaly Threshold (3σ): {threshold:.4f} GeV")
                
                normal_times = []
                anomaly_times = []
                normal_events_with_timing = 0
                anomaly_events_with_timing = 0
                
                print("\n4. Processing timing data...")
                
                for i in range(len(rho)):
                    proton_times_event = proton_times[i]
                    
                    # Filter out invalid times (-999)
                    valid_times = [t for t in proton_times_event if t > -900]
                    
                    if len(valid_times) > 0:
                        avg_time = np.mean(valid_times)
                        
                        if rho[i] > threshold:
                            anomaly_times.append(avg_time)
                            anomaly_events_with_timing += 1
                        else:
                            normal_times.append(avg_time)
                            normal_events_with_timing += 1
                
                # Convert to numpy arrays
                normal_times = np.array(normal_times)
                anomaly_times = np.array(anomaly_times)
                
                # 5. RESULTS
                print(f"\n5. Timing Results")
                print(f"   Normal Events with Valid Timing: {normal_events_with_timing:,}")
                print(f"   Anomalous Events with Valid Timing: {anomaly_events_with_timing:,}")
                
                if len(normal_times) > 0:
                    print(f"\n   NORMAL Timing Statistics:")
                    print(f"     Mean Time: {np.mean(normal_times):.4f} ns")
                    print(f"     Std Dev:   {np.std(normal_times):.4f} ns")
                    print(f"     Min:       {np.min(normal_times):.4f} ns")
                    print(f"     Max:       {np.max(normal_times):.4f} ns")
                    print(f"     25th %ile: {np.percentile(normal_times, 25):.4f} ns")
                    print(f"     Median:    {np.median(normal_times):.4f} ns")
                    print(f"     75th %ile: {np.percentile(normal_times, 75):.4f} ns")
                
                if len(anomaly_times) > 0:
                    print(f"\n   ANOMALY Timing Statistics:")
                    print(f"     Mean Time: {np.mean(anomaly_times):.4f} ns")
                    print(f"     Std Dev:   {np.std(anomaly_times):.4f} ns")
                    print(f"     Min:       {np.min(anomaly_times):.4f} ns")
                    print(f"     Max:       {np.max(anomaly_times):.4f} ns")
                    print(f"     25th %ile: {np.percentile(anomaly_times, 25):.4f} ns")
                    print(f"     Median:    {np.median(anomaly_times):.4f} ns")
                    print(f"     75th %ile: {np.percentile(anomaly_times, 75):.4f} ns")
                    
                    # Calculate the shift with confidence
                    shift = np.mean(anomaly_times) - np.mean(normal_times)
                    shift_error = np.sqrt((np.std(normal_times)**2/len(normal_times)) + 
                                         (np.std(anomaly_times)**2/len(anomaly_times)))
                    
                    print(f"\n   >>> TIME SHIFT ANALYSIS <<<")
                    print(f"     Shift (Anomaly - Normal): {shift:.4f} ± {shift_error:.4f} ns")
                    print(f"     Relative Shift: {shift/np.abs(np.mean(normal_times))*100:.2f}%")
                    
                    # Statistical test
                    from scipy.stats import ttest_ind
                    t_stat, p_value = ttest_ind(normal_times, anomaly_times, equal_var=False)
                    print(f"     T-statistic: {t_stat:.4f}")
                    print(f"     P-value: {p_value:.6f}")
                    
                    if p_value < 0.05:
                        print(f"     >>> STATISTICALLY SIGNIFICANT DIFFERENCE")
                    else:
                        print(f"     >>> NOT STATISTICALLY SIGNIFICANT")
                
                else:
                    print(f"\n   WARNING: No valid timing data for anomalous events!")
                    print(f"   Cannot perform timing shift analysis.")
                
                # 6. VISUALIZATION
                print(f"\n6. Generating visualizations...")
                
                if len(normal_times) > 0 or len(anomaly_times) > 0:
                    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                    
                    # Plot 1: Histogram comparison
                    if len(normal_times) > 0:
                        axes[0, 0].hist(normal_times, bins=50, density=True, alpha=0.5, 
                                       color='gray', label=f'Normal (n={len(normal_times):,})', 
                                       range=(-2, 2) if len(normal_times) > 0 else None)
                    
                    if len(anomaly_times) > 0:
                        axes[0, 0].hist(anomaly_times, bins=20, density=True, alpha=0.7, 
                                       color='red', label=f'Anomalous (n={len(anomaly_times):,})',
                                       range=(-2, 2) if len(anomaly_times) > 0 else None)
                    
                    axes[0, 0].set_title("Proton Arrival Time Distribution")
                    axes[0, 0].set_xlabel("Time (nanoseconds)")
                    axes[0, 0].set_ylabel("Normalized Frequency")
                    axes[0, 0].legend()
                    axes[0, 0].grid(True, alpha=0.3)
                    
                    # Plot 2: Box plot comparison
                    if len(normal_times) > 0 and len(anomaly_times) > 0:
                        data_to_plot = [normal_times, anomaly_times]
                        bp = axes[0, 1].boxplot(data_to_plot, labels=['Normal', 'Anomalous'], 
                                               patch_artist=True)
                        bp['boxes'][0].set_facecolor('gray')
                        bp['boxes'][1].set_facecolor('red')
                        axes[0, 1].set_title("Time Distribution Comparison")
                        axes[0, 1].set_ylabel("Time (ns)")
                        axes[0, 1].grid(True, alpha=0.3)
                    
                    # Plot 3: Time vs Rho scatter
                    if len(normal_times) > 0:
                        # Get corresponding rho values for normal events with timing
                        normal_rhos = []
                        normal_time_list = []
                        for i in range(len(rho)):
                            proton_times_event = proton_times[i]
                            valid_times = [t for t in proton_times_event if t > -900]
                            if len(valid_times) > 0 and rho[i] <= threshold:
                                normal_rhos.append(rho[i])
                                normal_time_list.append(np.mean(valid_times))
                        
                        axes[1, 0].scatter(normal_rhos, normal_time_list, alpha=0.3, s=10, 
                                          color='gray', label='Normal')
                    
                    if len(anomaly_times) > 0:
                        # Get corresponding rho values for anomaly events with timing
                        anomaly_rhos = []
                        anomaly_time_list = []
                        for i in range(len(rho)):
                            proton_times_event = proton_times[i]
                            valid_times = [t for t in proton_times_event if t > -900]
                            if len(valid_times) > 0 and rho[i] > threshold:
                                anomaly_rhos.append(rho[i])
                                anomaly_time_list.append(np.mean(valid_times))
                        
                        axes[1, 0].scatter(anomaly_rhos, anomaly_time_list, alpha=0.7, s=30, 
                                          color='red', label='Anomalous')
                        
                        axes[1, 0].axvline(threshold, color='orange', linestyle='--', 
                                          label=f'Threshold ({threshold:.1f} GeV)')
                    
                    axes[1, 0].set_title("Time vs Vacuum Energy Density")
                    axes[1, 0].set_xlabel("Rho (GeV)")
                    axes[1, 0].set_ylabel("Average Proton Time (ns)")
                    axes[1, 0].legend()
                    axes[1, 0].grid(True, alpha=0.3)
                    
                    # Plot 4: Cumulative distribution
                    if len(normal_times) > 0 and len(anomaly_times) > 0:
                        sorted_normal = np.sort(normal_times)
                        sorted_anomaly = np.sort(anomaly_times)
                        cumul_normal = np.arange(1, len(sorted_normal)+1) / len(sorted_normal)
                        cumul_anomaly = np.arange(1, len(sorted_anomaly)+1) / len(sorted_anomaly)
                        
                        axes[1, 1].plot(sorted_normal, cumul_normal, 'b-', linewidth=2, 
                                       label='Normal')
                        axes[1, 1].plot(sorted_anomaly, cumul_anomaly, 'r-', linewidth=2, 
                                       label='Anomalous')
                        axes[1, 1].set_title("Cumulative Time Distributions")
                        axes[1, 1].set_xlabel("Time (ns)")
                        axes[1, 1].set_ylabel("Cumulative Probability")
                        axes[1, 1].legend()
                        axes[1, 1].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plot_filename = "timing_analysis_comprehensive.png"
                    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    print(f"   Comprehensive plot saved as '{plot_filename}'")
                    
                    # Also create simple histogram
                    plt.figure(figsize=(10, 6))
                    if len(normal_times) > 0:
                        plt.hist(normal_times, bins=50, density=True, alpha=0.5, 
                                color='gray', label=f'Normal (n={len(normal_times):,})')
                    if len(anomaly_times) > 0:
                        plt.hist(anomaly_times, bins=20, density=True, alpha=0.7, 
                                color='red', label=f'High-Resistance (n={len(anomaly_times):,})')
                    
                    plt.title("Proton Arrival Time Distribution")
                    plt.xlabel("Time (nanoseconds)")
                    plt.ylabel("Normalized Frequency")
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    simple_plot_filename = "timing_histogram.png"
                    plt.savefig(simple_plot_filename, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    print(f"   Simple histogram saved as '{timing_histogram.png}'")
                
                print(f"\n7. Analysis Complete")
                total_time = time.time() - start_time
                print(f"   Total processing time: {total_time:.2f}s")
                print(f"   Log file: {log_filename}")
                
                if len(anomaly_times) > 0:
                    print(f"\n=== KEY FINDING ===")
                    print(f"Time shift detected: {shift:.4f} ± {shift_error:.4f} ns")
                    print(f"Statistical significance: {'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'}")
                
    except FileNotFoundError:
        print(f"ERROR: File '{file_path}' not found!")
    except KeyError as e:
        print(f"ERROR: Branch {e} not found!")
        print(f"Available branches: {list(tree.keys())[:20]}...")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        print("\nTraceback:")
        traceback.print_exc()

print(f"\n[All timing analysis output saved to: {log_filename}]")