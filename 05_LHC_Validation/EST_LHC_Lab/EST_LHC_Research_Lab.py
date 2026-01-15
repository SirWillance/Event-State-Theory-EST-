#!/usr/bin/env python3
import subprocess
import sys
import re
import os
import tempfile
import shutil

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    HAS_TK = True
except ImportError:
    HAS_TK = False

# Mapping of options to the exact script filenames
scripts = {
    '1': 'probe.py',                      # Basic file probing for branches    
    '2': 'analyze_stability.py',          # Stability analysis    
    '3': 'verify_full_scale.py',          # Full-scale stability verification with bootstrap
    '4': 'analyze_pcr.py',                # Vacuum energy statistics and anomalies
    '5': 'check_pileup.py',               # Pile-up check (this one already supports sys.argv, but we'll handle uniformly)
    '6': 'analyze_timing_PPS.py',         # PPS timing analysis
    '7': 'analyze_timing.py'              # Proton timing analysis
}

def display_menu():
    print("""###############################################
#### === EST LHC Research Lab Selector === ####
###############################################""")
    print("")
    print("Select an option to run the corresponding analysis:")
    print("")
    print("1: File Probing (probe.py)")    
    print("2: Stability Analysis (analyze_stability.py)")    
    print("3: Full-Scale Verification (verify_full_scale.py)")
    print("4: Vacuum Energy Statistics (analyze_pcr.py)")
    print("5: Pile-up Check (check_pileup.py)")
    print("6: PPS Timing Analysis (analyze_timing_PPS.py)")
    print("7: Proton Timing Analysis (analyze_timing.py)")
    print("q: Quit")
    print("")
    print("=================================================")
    print("")

def select_file():
    """Select a ROOT file using GUI or fallback to input."""
    selected_file = None
    if HAS_TK:
        try:
            root = tk.Tk()
            root.withdraw()
            selected_file = filedialog.askopenfilename(
                title="Select ROOT File",
                filetypes=[("ROOT files", "*.root"), ("All files", "*.*")]
            )
            if not selected_file:
                print("File selection canceled.")
        except Exception as e:
            print(f"GUI file picker failed: {e}. Falling back to manual input.")
    if not selected_file:
        selected_file = input("Enter the full path to the ROOT file (or leave blank to use hardcoded): ").strip()
        if not selected_file:
            return None
    if not os.path.exists(selected_file):
        print(f"Warning: Selected file '{selected_file}' does not exist.")
    return selected_file

def run_script(script, custom_file=None):
    """Run the script, using a temp copy with overwritten file_path if custom_file provided."""
    if not os.path.exists(script):
        print(f"Error: {script} not found in the current directory.")
        return

    if custom_file:
        # Create temp file
        temp_dir = tempfile.mkdtemp()
        temp_script = os.path.join(temp_dir, os.path.basename(script))
        shutil.copy(script, temp_script)

        # Read original lines
        with open(temp_script, 'r') as f:
            lines = f.readlines()

        # Overwrite file_path lines
        with open(temp_script, 'w') as f:
            for line in lines:
                if 'file_path =' in line:
                    # Replace the string value after =
                    line = re.sub(r'file_path\s*=\s*["\'].*?["\']', f'file_path = "{custom_file}"', line)
                f.write(line)

        print(f"Running modified temp version of {script} with file_path='{custom_file}'...")
        try:
            result = subprocess.run(['python', temp_script], capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("Errors:", result.stderr)
        except Exception as e:
            print(f"Error running modified {script}: {str(e)}")
        finally:
            # Clean up
            shutil.rmtree(temp_dir)
    else:
        print(f"Running {script} with hardcoded file...")
        try:
            result = subprocess.run(['python', script], capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("Errors:", result.stderr)
        except Exception as e:
            print(f"Error running {script}: {str(e)}")

def main():
    while True:
        display_menu()
        choice = input("Enter your choice (1-7 or q): ").strip()
        
        if choice.lower() == 'q':
            print("Exiting lab.")
            sys.exit(0)
        
        if choice in scripts:
            script = scripts[choice]
            
            # Ask if custom file
            use_custom = input("Use a custom ROOT file instead of hardcoded? (y/n): ").strip().lower()
            custom_file = None
            if use_custom == 'y':
                custom_file = select_file()
                if not custom_file:
                    print("Using hardcoded file.")
            
            run_script(script, custom_file)
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()