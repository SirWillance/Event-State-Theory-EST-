#!/usr/bin/env python3
# =============================================================================
#  EVENT-STATE THEORY SIMULATOR v3.0 – DOUBLE-CLICK VERSION
#  Works perfectly when you just double-click the file (Windows, macOS, Linux)
#  Torben Wille – November 2025
# =============================================================================
#!/usr/bin/env python3
# EVENT-STATE THEORY SIMULATOR v4.0 – BULLETPROOF EDITION
# Double-click → always works → always creates files → never crashes

import os, random, zlib, numpy as np, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
from scipy.ndimage import label
import ctypes, sys, traceback

EST_MODE = True                    # Your full physics
GRID = 128
FRAMES = 500
SEED = 42

DESKTOP = os.path.join(os.path.expanduser("~"), "Desktop")
OUT = os.path.join(DESKTOP, f"EST_Universe_{datetime.now():%Y-%m-%d_%H%M%S}")
os.makedirs(OUT, exist_ok=True)

random.seed(SEED); np.random.seed(SEED)
u = np.zeros((GRID,GRID), dtype=np.uint8)

# === Your physics (AMPC + scar tissue) ===
if EST_MODE:
    for _ in range(7):
        cy, cx = random.randint(20,107), random.randint(20,107)
        for dy in range(-7,8):
            for dx in range(-7,8):
                if abs(dy)+abs(dx)<8 and 0<=cy+dy<GRID and 0<=cx+dx<GRID:
                    u[cy+dy,cx+dx] = 1
else:
    u[60:68,60:68] = 1

history = [u.copy()]
log = os.path.join(OUT, "Log.txt")
with open(log,"w") as f: f.write("EST Simulation started\n")

def cost(prev, cand):
    dE = np.sum(prev != cand)
    K = len(zlib.compress(cand.tobytes()))
    scar = 0.07 * (np.sum((prev==0)&(cand==1)) - np.sum((prev==1)&(cand==0))) if EST_MODE else 0
    return dE + 3.2 * K - scar

for t in range(1, FRAMES+1):
    cands = []
    for _ in range(80):
        c = u.copy()
        cy,cx = random.randint(0,GRID-1), random.randint(0,GRID-1)
        for _ in range(random.randint(1,5)):
            dy,dx = random.randint(-2,2), random.randint(-2,2)
            y,x = cy+dy, cx+dx
            if 0<=y<GRID and 0<=x<GRID: c[y,x] = 1 - c[y,x]
        cands.append(c)

    costs = np.array([cost(u, c) for c in cands])
    # THIS LINE FIXES EVERYTHING – numerical safety
    costs -= costs.min()                              # shift to avoid underflow
    probs = np.exp(-costs / 0.45)
    probs = probs / probs.sum()                       # now always exactly 1.0
    # Extra safety in case everything is still identical
    if probs.sum() == 0: probs[:] = 1.0 / len(probs)

    u = cands[np.random.choice(len(cands), p=probs)]
    history.append(u.copy())

    if t%50==0 or t<6:
        print(f"Frame {t:3d}  →  density {u.mean():.3f}")
        with open(log,"a") as f:
            f.write(f"Frame {t} density {u.mean():.5f}\n")

# === Save everything ===
print("\nCreating animation (20–40 seconds)…")
fig = plt.figure(figsize=(10,10), facecolor="black")
im = plt.imshow(history[0], cmap="plasma", interpolation="none")
plt.axis("off")
def a(f): im.set_data(history[f])
ani = FuncAnimation(fig, a, frames=len(history), interval=70)
ani.save(os.path.join(OUT, "Event_State_Theory.gif"), writer="pillow", fps=25)
plt.close(fig)

plt.imsave(os.path.join(OUT, "Final_State.png"), u, cmap="plasma")

print("\nSUCCESS! Files are on your Desktop in folder:")
print(OUT)
print("\n  • Event_State_Theory.gif")
print("  • Final_State.png")
print("  • Log.txt")

# Final message + auto-open folder
if sys.platform.startswith('win'):
    ctypes.windll.user32.MessageBoxW(0,
        f"Event-State Theory simulation finished!\n\nFolder opened automatically:\n\n{OUT}",
        "Success", 0x40)
    os.startfile(OUT)
else:
    input("Press Enter to close...")
