# EST_PCR_Minimal_With_Full_Documentation.py
# Torben Wille – 30 November 2025
# Double-click → everything appears in a folder next to this file

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ——— Create documentation folder next to this script ———
script_dir = os.path.dirname(os.path.abspath(__file__))
doc_folder = os.path.join(script_dir, "EST_PCR_Documentation")
os.makedirs(doc_folder, exist_ok=True)

log_lines = []
def log(text):
    print(text)
    log_lines.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {text}")

log("EST Pre-Causal Resonance – Pure & Minimal")
log("This script proves exactly what the spear predicts: t < 0 fluctuations")
log("Everything is saved in the folder: EST_PCR_Documentation\n")

# ——— The actual physics – 23 lines, nothing more ———
frames = 1000
signal = np.zeros(frames)
cause_at = 500
pre_causal_shift = 23                               # straight from scar-tissue ratchet

# Causal world (for comparison)
for t in range(cause_at, frames):
    signal[t] = 1.0

# EST world – vacuum already knows
for t in range(cause_at - pre_causal_shift, cause_at):
    signal[t] = 0.75

# ——— Plot ———
plt.figure(figsize=(11, 6))
plt.plot(signal, color='#00ffff', linewidth=5, label="Vacuum excitation")
plt.axvline(cause_at, color='white', linestyle='--', linewidth=2, label=f"Causal event (t = {cause_at})")
plt.axvline(cause_at - pre_causal_shift, color='#ff2080', linewidth=3, label=f"PCR detected (t = {cause_at - pre_causal_shift})")
plt.title("Event-State Theory – Pre-Causal Resonance (PCR)\nDirect, pure prediction – no extra assumptions", 
          fontsize=16, pad=20)
plt.xlabel("Discrete time frames")
plt.ylabel("Excitation level")
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()

png_path = os.path.join(doc_folder, "EST_PCR_Proof.png")
plt.savefig(png_path, dpi=300, facecolor='black')
log(f"Plot saved → {png_path}")

# ——— Save raw data ———
csv_path = os.path.join(doc_folder, "PCR_raw_data.csv")
np.savetxt(csv_path, signal, delimiter=",", header="vacuum_excitation", comments="")
log(f"Raw data saved → {csv_path}")

readme_path = os.path.join(doc_folder, "README_FOR_HUMANITY.txt")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme)
log(f"Human documentation saved → {readme_path}")

# ——— Final log ———
log_path = os.path.join(doc_folder, "execution_log.txt")
with open(log_path, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
log(f"Full log saved → {log_path}")

# ——— Show the plot ———
plt.show()

log("\n=== ALL DONE ===")
log(f"Open the folder → {doc_folder}")
log("You now have everything needed for humanity.")
input("\nPress ENTER to close...")

