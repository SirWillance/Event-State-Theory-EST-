# Event-State-Theory-EST-
A Discrete Causal-Information Framework for Cosmology and Fundamental Physics

## 🔬 Scientific Context

This repository contains the computational engine and results for the paper **"Computational Emergence of Cosmic Structure from Algorithmic Probability"** (Wille, 2025). It provides:

- **Empirical validation** of Event-State Theory's prediction of cosmic structure formation
- **Phase space analysis** revealing critical constants (λ=0.48, β=3.4) at the edge of chaos  
- **Falsifiable evidence** that can be independently verified by running the code

**Peer reviewers should focus on:**
- Reproducing the phase transition boundary shown in `fig_heatmap.png`
- Verifying that deviation from (λ=0.48, β=3.4) destroys cosmic web formation
- Examining the linear baryon accumulation in `density_evolution.csv`

# Event-State Theory (EST) Computational Engine

**Research Author:** Torben Wille  
**System Version:** v2.0 (Windows Optimized)  
**Related Papers:** [Zenodo DOI Pending]

---

### 🌌 Abstract
This repository contains the `EST_Laboratory.py` source code, a discrete 3D cellular automaton designed to test the **Event-State Theory** cosmological framework. By utilizing **Conditional Kolmogorov Complexity** ($K$) as a thermodynamic cost function, this engine simulates:
*   **Asynchronous Multi-Point Crystallization (AMPC)**: The emergence of the cosmic web from void nucleation.
*   **Linear Baryonic Accumulation**: Validation of the "Scar Tissue" stability mechanism.
*   **Algorithmic Time Dilation**: The emergent variable frame rate of causal updates.

### 🧪 Features
The `EST_Laboratory` provides four experimental protocols:
1.  **Cosmology Simulation**: Generates 3D filaments and calculates density evolution ($N=192^3$).
2.  **Manual Exploration**: Custom parameter testing.
3.  **Phase Space Sweep**: Verification of the "Goldilocks" $\beta / \lambda$ constants at the edge of chaos.
4.  **Isotropy Check**: Analysis of lattice-based artifacts (System Limitations).

### 🔬 Experimental Modes
The Unified Engine (`EST_Laboratory.py`) operates in two distinct modalities:

*   **Mode A: Verified Replication (Default)**  
    Runs the simulation using the precise `λ=0.48`, `β=3.4` constants identified in the phase-space analysis. This reproduces the 12-site Cosmic Web and linear density accumulation documented in Paper 2.

*   **Mode B: The Sandbox (Manual Configuration)**  
    Allows peer reviewers to manually input Grid Size, Frame Count, and Thermodynamic Constants.  
    *Purpose:* To enable independent verification of the "Edge of Chaos" boundary (i.e., verifying that modifying `λ` or `β` destroys the structure).

### 🛠️ Installation & Usage
Requires **Python 3.8+** and standard scientific libraries.

```bash
# 1. Clone the repository
git clone https://github.com/SirWillance/Event-State-Theory-EST-.git

# 2. Install dependencies
pip install numpy matplotlib scipy tqdm imageio

# 3. Run the Laboratory
python EST_Laboratory.py


### 📂 Primary Data (Artifacts)
This repository hosts the results of a **High-Resolution Run ($256^3$ Grid)** conducted on Nov 29, 2025:
*   **Duration:** 440 Frames (2.6 Hours computation)
*   **Resolution:** 16,777,216 nodes
*   **Results:** 
    *   `density_evolution.csv`: Raw tracking of baryonic accumulation.
    *   `Deep_Field_Projection.png`: Filamentary structure visualization.

### 📊 Experimental Results
Raw data and visualizations from the high-resolution run are available in the `/artifacts/` directory, including:
- Density evolution measurements (`density_evolution.csv`)
- Cosmic web formation evidence (`Deep_Field_Projection.png`) 
- Phase space analysis (`Phase_Space_Topology.png`)
