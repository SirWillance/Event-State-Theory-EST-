import uproot

# The path to your file
file_path = "nano_data2016_42.root"

print(f"--- Probing File: {file_path} ---")

try:
    # Open the file
    with uproot.open(file_path) as file:
        # ROOT files contain "Trees" (like folders). The main one is usually "Events".
        # Let's list all keys to find the Tree name.
        print(f"Keys in file: {file.keys()}")
        
        # Access the Events tree (standard for CMS NanoAOD)
        # Note: The key might be "Events;1" or just "Events"
        if "Events" in file:
            tree = file["Events"]
            
            # Get all branch names (variable names)
            branches = tree.keys()
            
            print(f"\nTotal Branches (Variables) found: {len(branches)}")
            
            # SEARCH FOR TIME
            # We are hunting for anything related to 'time', 'timestamp', 'trigger', or 'calo'
            print("\n--- SUSPECT BRANCHES (Potential Time/Noise Data) ---")
            relevant_keywords = ["time", "Time", "trigger", "Trigger", "Calo", "noise", "Energy"]
            
            count = 0
            for branch in branches:
                if any(keyword in branch for keyword in relevant_keywords):
                    print(f"Found: {branch}")
                    count += 1
            
            if count == 0:
                print("WARNING: No obvious timing or raw energy branches found in the index.")
                
        else:
            print("CRITICAL: Could not find 'Events' tree. Structure may be non-standard.")

except Exception as e:
    print(f"Error opening file: {e}")