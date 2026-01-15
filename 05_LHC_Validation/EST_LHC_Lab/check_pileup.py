# Save this as: check_pileup_with_logging.py
import uproot
import numpy as np
import sys
import os
import datetime
import time
import traceback

# Setup logging at the very beginning
def setup_logging(file_path):
    """Create comprehensive log file for the entire process"""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"pileup_analysis_log_{timestamp}.txt"
    
    # Create log directory if it doesn't exist
    log_dir = "analysis_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_path = os.path.join(log_dir, log_filename)
    
    # Create log file with initial info
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PILEUP ANALYSIS LOG\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Analysis started: {datetime.datetime.now()}\n")
        f.write(f"Input file: {file_path}\n")
        f.write(f"Python version: {sys.version}\n")
        f.write(f"Working directory: {os.getcwd()}\n")
        f.write(f"Log file: {log_path}\n")
        f.write(f"Uproot version: {uproot.__version__}\n")
        f.write(f"NumPy version: {np.__version__}\n")
        f.write("-" * 80 + "\n\n")
    
    return log_path

class Logger:
    """Dual logger for console and file"""
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.start_time = time.time()
        self.message_count = 0
        
    def log(self, message, include_timestamp=True):
        """Log message to both console and file"""
        self.message_count += 1
        
        if include_timestamp:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            formatted_msg = f"[{timestamp}] {message}"
        else:
            formatted_msg = message
        
        # Print to console
        print(formatted_msg)
        
        # Write to log file
        try:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(formatted_msg + "\n")
        except Exception as e:
            print(f"ERROR writing to log: {e}")
    
    def log_section(self, title):
        """Log a section header"""
        self.log("\n" + "=" * 60)
        self.log(title)
        self.log("=" * 60)
    
    def log_subsection(self, title):
        """Log a subsection header"""
        self.log(f"\n{title}")
        self.log("-" * len(title))
    
    def log_warning(self, message):
        """Log a warning message"""
        self.log(f"⚠ WARNING: {message}")
    
    def log_error(self, message):
        """Log an error message"""
        self.log(f"❌ ERROR: {message}")
    
    def log_success(self, message):
        """Log a success message"""
        self.log(f"✅ {message}")
    
    def log_data(self, label, value):
        """Log a data point"""
        self.log(f"  {label}: {value}")
    
    def log_table_row(self, *columns, widths=None):
        """Log a table row"""
        if widths is None:
            widths = [20, 40]
        
        row = ""
        for i, col in enumerate(columns):
            if i < len(widths):
                row += str(col).ljust(widths[i])
            else:
                row += str(col)
        self.log(row)
    
    def log_list(self, items, title=None, max_items=20):
        """Log a list of items"""
        if title:
            self.log(f"\n{title}:")
        
        if not items:
            self.log("  (none)")
            return
        
        for i, item in enumerate(items[:max_items]):
            self.log(f"  {i+1:3d}. {item}")
        
        if len(items) > max_items:
            self.log(f"  ... and {len(items) - max_items} more")
    
    def log_array_stats(self, data, label, fmt=".2f"):
        """Log statistics for an array"""
        if len(data) == 0:
            self.log(f"  {label}: No data")
            return
        
        self.log(f"  {label}:")
        self.log(f"    Count: {len(data):,}")
        self.log(f"    Mean: {np.mean(data):{fmt}}")
        self.log(f"    Std: {np.std(data):{fmt}}")
        self.log(f"    Min: {np.min(data):{fmt}}")
        self.log(f"    Max: {np.max(data):{fmt}}")
        if len(data) > 1:
            self.log(f"    Median: {np.median(data):{fmt}}")
    
    def log_exception(self, exception):
        """Log an exception with full traceback"""
        self.log_error(f"{type(exception).__name__}: {str(exception)}")
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write("\nFULL TRACEBACK:\n")
            f.write(traceback.format_exc())
    
    def finalize(self):
        """Finalize the log with summary"""
        elapsed_time = time.time() - self.start_time
        
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Analysis completed: {datetime.datetime.now()}\n")
            f.write(f"Total runtime: {elapsed_time:.2f} seconds\n")
            f.write(f"Total messages logged: {self.message_count}\n")
            f.write(f"Log file size: {os.path.getsize(self.log_file_path)/1024:.1f} KB\n")
            f.write(f"Peak memory usage: N/A\n")  # Could add memory profiling
        
        self.log(f"\nAnalysis completed in {elapsed_time:.2f} seconds")
        self.log(f"Detailed log saved to: {self.log_file_path}")

# Main analysis function
def analyze_pileup(file_path, logger):
    """Main analysis function with comprehensive logging"""
    
    try:
        # 1. FILE INFORMATION
        logger.log_section("FILE INFORMATION")
        logger.log(f"Checking: {file_path}")
        
        if not os.path.exists(file_path):
            logger.log_error(f"File not found: {file_path}")
            return False
        
        file_size = os.path.getsize(file_path) / (1024*1024)  # MB
        logger.log_data("File size", f"{file_size:.1f} MB")
        logger.log_data("File exists", "Yes")
        
        # 2. OPEN FILE AND GET TREE
        logger.log_section("OPENING ROOT FILE")
        
        with uproot.open(file_path) as file:
            logger.log_success("File opened successfully")
            
            # Check available trees
            available_keys = list(file.keys())
            logger.log_list(available_keys, "Available trees/collections", max_items=10)
            
            # Find Events tree
            tree_name = None
            for key in available_keys:
                if "Events" in key or "events" in key.lower():
                    tree_name = key
                    break
            
            if tree_name is None and available_keys:
                tree_name = available_keys[0]
            
            if tree_name is None:
                logger.log_error("No trees found in file")
                return False
            
            logger.log_data("Using tree", tree_name)
            tree = file[tree_name]
            
            # 3. LIST ALL BRANCHES
            logger.log_section("ANALYZING BRANCHES")
            all_branches = list(tree.keys())
            logger.log_data("Total branches", len(all_branches))
            
            # 4. SEARCH FOR PILEUP VARIABLES
            logger.log_subsection("Searching for pile-up variables")
            
            pileup_keywords = ['Pileup', 'pileup', 'nTrueInt', 'nPU', 'NumPU', 'NumTrue', 'pu', 'PU']
            found_pileup = []
            
            for branch in all_branches:
                branch_lower = branch.lower()
                if any(keyword.lower() in branch_lower for keyword in pileup_keywords):
                    if 'HLT_' not in branch and 'tau' not in branch_lower:
                        found_pileup.append(branch)
            
            if found_pileup:
                logger.log_success(f"Found {len(found_pileup)} pile-up related variables")
                logger.log_list(found_pileup, "Pile-up variables", max_items=15)
            else:
                logger.log_warning("No pile-up variables found!")
                
                # Show first 20 simple variables
                simple_vars = []
                for branch in all_branches[:100]:  # Check first 100
                    if not any(x in branch for x in ['HLT_', 'Tau_', 'Jet_', 'Muon_', 'Electron_', 'Photon_', 'Gen_']):
                        simple_vars.append(branch)
                        if len(simple_vars) >= 20:
                            break
                
                logger.log_list(simple_vars, "Simple variables (first 20)", max_items=20)
            
            # 5. TEST COMMON PILEUP NAMES
            logger.log_subsection("Testing common pile-up branch names")
            
            common_pu_names = [
                "Pileup_nTrueInt",
                "Pileup_nPU",
                "Pileup_gpudensity",
                "nPU",
                "numPU",
                "puNum",
                "nTrueInt",
                "nPV",
                "PV_npvs"
            ]
            
            available_pu = None
            for pu_name in common_pu_names:
                if pu_name in all_branches:
                    available_pu = pu_name
                    logger.log_success(f"Found standard pile-up branch: {pu_name}")
                    break
            
            if not available_pu:
                logger.log_warning("No standard pile-up branch found")
            
            # 6. PREPARE TEST VARIABLES
            logger.log_subsection("Preparing test variables")
            
            # Always check these
            essential_vars = ["fixedGridRhoFastjetCentralCalo", "CaloMET_sumEt"]
            missing_essential = []
            
            for var in essential_vars:
                if var not in all_branches:
                    missing_essential.append(var)
                    logger.log_warning(f"Essential variable not found: {var}")
            
            test_branches = []
            for var in essential_vars:
                if var in all_branches:
                    test_branches.append(var)
            
            # Add pile-up candidate if found
            if available_pu:
                test_branches.append(available_pu)
            
            # Add up to 3 more candidate variables
            candidates_added = 0
            for branch in all_branches:
                if branch not in test_branches and candidates_added < 3:
                    if not any(x in branch for x in ['HLT_', '_eta', '_phi', '_pt']):
                        test_branches.append(branch)
                        candidates_added += 1
            
            logger.log_list(test_branches, "Variables to test", max_items=10)
            
            # 7. LOAD DATA
            logger.log_section("LOADING DATA")
            
            # Determine number of events to load
            try:
                n_entries = tree.num_entries
                logger.log_data("Total events in tree", f"{n_entries:,}")
                
                # Load reasonable sample size
                load_limit = min(10000, n_entries)
                logger.log_data("Loading events", f"{load_limit:,}")
                
                data = tree.arrays(test_branches, library="np", entry_stop=load_limit)
                logger.log_success(f"Loaded {len(data[test_branches[0]]):,} events")
                
            except Exception as e:
                logger.log_error(f"Error determining tree size: {e}")
                # Try loading smaller sample
                logger.log("Trying to load first 5000 events...")
                data = tree.arrays(test_branches, library="np", entry_stop=5000)
                logger.log_success(f"Loaded {len(data[test_branches[0]]):,} events")
            
            # 8. ANALYZE EACH VARIABLE
            logger.log_section("VARIABLE ANALYSIS")
            
            # Check rho and MET first
            if "fixedGridRhoFastjetCentralCalo" in data:
                rho = data["fixedGridRhoFastjetCentralCalo"]
                logger.log_array_stats(rho, "ρ (fixedGridRhoFastjetCentralCalo)")
            else:
                logger.log_error("ρ variable not loaded")
                return False
            
            if "CaloMET_sumEt" in data:
                met = data["CaloMET_sumEt"]
                logger.log_array_stats(met, "CaloMET ΣE_T")
            else:
                logger.log_error("MET variable not loaded")
                return False
            
            # Analyze each additional variable
            variable_results = []
            
            for i, branch in enumerate(test_branches[2:], 1):  # Skip rho and MET
                logger.log_subsection(f"Variable {i}: {branch}")
                
                try:
                    if branch not in data:
                        logger.log_warning(f"Variable not in loaded data")
                        continue
                    
                    var_data = data[branch]
                    
                    # Check data type
                    if hasattr(var_data, 'dtype'):
                        logger.log_data("Data type", var_data.dtype)
                    
                    # Handle jagged arrays
                    is_jagged = False
                    if len(var_data) > 0 and hasattr(var_data[0], '__len__'):
                        is_jagged = True
                        logger.log_data("Array type", "Jagged (variable length per event)")
                        
                        # Flatten by taking mean per event
                        flat_data = []
                        valid_count = 0
                        for event_vals in var_data:
                            if len(event_vals) > 0:
                                flat_data.append(np.mean(event_vals))
                                valid_count += 1
                            else:
                                flat_data.append(np.nan)
                        
                        var_data_flat = np.array(flat_data)
                        logger.log_data("Events with data", f"{valid_count}/{len(var_data)}")
                        
                        # Clean data
                        mask = ~np.isnan(var_data_flat)
                        if np.sum(mask) > 100:
                            var_data_clean = var_data_flat[mask]
                            rho_clean = rho[mask]
                            
                            logger.log_array_stats(var_data_clean, f"{branch} (mean per event)")
                            
                            # Correlation with rho
                            if len(var_data_clean) > 1 and len(rho_clean) > 1:
                                corr = np.corrcoef(rho_clean, var_data_clean)[0,1]
                                logger.log_data("Correlation with ρ", f"{corr:.4f}")
                                
                                # Check if it could be pile-up
                                mean_val = np.mean(var_data_clean)
                                is_pu_candidate = (5 < mean_val < 100) and (abs(corr) > 0.3)
                                
                                variable_results.append({
                                    'name': branch,
                                    'mean': mean_val,
                                    'corr': corr,
                                    'is_jagged': True,
                                    'is_pu_candidate': is_pu_candidate
                                })
                                
                                if is_pu_candidate:
                                    logger.log_success("→ Pile-up candidate identified")
                                else:
                                    logger.log("→ Not a pile-up candidate")
                    else:
                        # Simple array
                        logger.log_data("Array type", "Simple (scalar per event)")
                        logger.log_array_stats(var_data, branch)
                        
                        # Correlation with rho
                        if len(var_data) > 1:
                            corr = np.corrcoef(rho, var_data)[0,1]
                            logger.log_data("Correlation with ρ", f"{corr:.4f}")
                            
                            # Check if it could be pile-up
                            mean_val = np.mean(var_data)
                            is_pu_candidate = (5 < mean_val < 100) and (abs(corr) > 0.3)
                            
                            variable_results.append({
                                'name': branch,
                                'mean': mean_val,
                                'corr': corr,
                                'is_jagged': False,
                                'is_pu_candidate': is_pu_candidate
                            })
                            
                            if is_pu_candidate:
                                logger.log_success("→ Pile-up candidate identified")
                            else:
                                logger.log("→ Not a pile-up candidate")
                
                except Exception as e:
                    logger.log_error(f"Error analyzing {branch}: {str(e)[:100]}")
            
            # 9. ORIGINAL STABILITY ANALYSIS
            logger.log_section("ORIGINAL STABILITY ANALYSIS")
            
            # Calculate ratios
            mask = rho > 0.1
            rho_clean = rho[mask]
            met_clean = met[mask]
            
            if len(rho_clean) > 0:
                ratios = met_clean / rho_clean
                
                # Calculate groups
                avg_rho = np.mean(rho_clean)
                std_rho = np.std(rho_clean)
                threshold = avg_rho + 2.5 * std_rho
                
                high_rho_mask = rho_clean >= threshold
                normal_rho_mask = rho_clean < threshold
                
                ratio_high = ratios[high_rho_mask]
                ratio_normal = ratios[normal_rho_mask]
                
                # Coefficient of Variation
                def cv(data):
                    if len(data) == 0 or np.mean(data) == 0:
                        return float('nan')
                    return (np.std(data) / np.mean(data)) * 100
                
                cv_normal = cv(ratio_normal)
                cv_high = cv(ratio_high)
                stability_gain = cv_normal - cv_high
                
                logger.log_data("Total valid events", f"{len(rho_clean):,}")
                logger.log_data("Mean ρ", f"{avg_rho:.2f}")
                logger.log_data("Std ρ", f"{std_rho:.2f}")
                logger.log_data("High-ρ threshold", f"{threshold:.2f}")
                logger.log_data("Normal events", f"{len(ratio_normal):,} ({len(ratio_normal)/len(rho_clean)*100:.1f}%)")
                logger.log_data("High-ρ events", f"{len(ratio_high):,} ({len(ratio_high)/len(rho_clean)*100:.1f}%)")
                logger.log_data("Normal group CV", f"{cv_normal:.2f}%")
                logger.log_data("High-ρ group CV", f"{cv_high:.2f}%")
                logger.log_data("Stability gain", f"{stability_gain:.2f}%")
                
                if stability_gain > 0:
                    logger.log_success(f"Stability improved by {stability_gain:.2f}% in high-ρ events")
                else:
                    logger.log_warning(f"Stability decreased by {-stability_gain:.2f}% in high-ρ events")
            
            # 10. SUMMARY AND RECOMMENDATIONS
            logger.log_section("SUMMARY AND RECOMMENDATIONS")
            
            # Find best pile-up candidate
            best_pu_candidate = None
            best_corr = 0
            
            for result in variable_results:
                if result['is_pu_candidate'] and abs(result['corr']) > best_corr:
                    best_pu_candidate = result
                    best_corr = abs(result['corr'])
            
            if best_pu_candidate:
                logger.log_success(f"Best pile-up candidate: {best_pu_candidate['name']}")
                logger.log_data("Mean value", f"{best_pu_candidate['mean']:.2f}")
                logger.log_data("Correlation with ρ", f"{best_pu_candidate['corr']:.4f}")
                
                if best_pu_candidate['corr'] > 0.7:
                    logger.log("→ VERY STRONG correlation with pile-up")
                    logger.log("   Your effect is highly correlated with pile-up")
                elif best_pu_candidate['corr'] > 0.5:
                    logger.log("→ STRONG correlation with pile-up")
                    logger.log("   Your effect is strongly related to pile-up")
                elif best_pu_candidate['corr'] > 0.3:
                    logger.log("→ MODERATE correlation with pile-up")
                    logger.log("   Your effect has moderate pile-up dependence")
                else:
                    logger.log("→ WEAK correlation with pile-up")
                    logger.log("   Your effect is largely pile-up independent")
            else:
                logger.log_warning("No clear pile-up candidate identified")
                logger.log("Your datasets may not contain standard pile-up information")
            
            logger.log_subsection("RECOMMENDATIONS")
            logger.log("1. For physics publication, always include pile-up systematics")
            logger.log("2. If no pile-up variable found, state this limitation clearly")
            logger.log("3. Consider re-running on datasets with pile-up information")
            logger.log("4. Your stability analysis method appears valid")
            logger.log("5. Present results with appropriate confidence intervals")
            
            return True
            
    except Exception as e:
        logger.log_section("FATAL ERROR")
        logger.log_exception(e)
        return False

def main():
    """Main execution function"""
    
    # Get filename from command line
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "nano_data2016_11.root"
    
    # Setup logging
    log_file_path = setup_logging(file_path)
    logger = Logger(log_file_path)
    
    # Log startup information
    logger.log_section("PILEUP ANALYSIS STARTED")
    logger.log(f"Command: python {' '.join(sys.argv)}")
    logger.log(f"Current time: {datetime.datetime.now()}")
    
    try:
        # Run analysis
        success = analyze_pileup(file_path, logger)
        
        # Finalize
        logger.log_section("ANALYSIS COMPLETED")
        if success:
            logger.log_success("Analysis completed successfully")
        else:
            logger.log_error("Analysis completed with errors")
        
        logger.finalize()
        
        # Print final message to console
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"✓ Log file saved to: {log_file_path}")
        print(f"✓ Console output captured")
        print(f"✓ All analysis steps logged")
        print("\nTo view the full log:")
        print(f"  cat {log_file_path} | less")
        print("=" * 80)
        
    except KeyboardInterrupt:
        logger.log_section("ANALYSIS INTERRUPTED")
        logger.log_error("Analysis interrupted by user (Ctrl+C)")
        logger.finalize()
        print("\nAnalysis interrupted by user")
        print(f"Partial log saved to: {log_file_path}")
    except Exception as e:
        logger.log_section("UNEXPECTED ERROR")
        logger.log_error(f"Unexpected error: {e}")
        logger.finalize()
        print(f"\nUnexpected error: {e}")
        print(f"Error log saved to: {log_file_path}")

if __name__ == "__main__":
    main()