# Event-State Theory (EST): A Computational Framework
*A Discrete Causal-Information Framework for Cosmology and Fundamental Physics*

## 🔬 Scientific Context

This repository contains the computational engine and empirical results for the paper **"Computational Emergence of Cosmic Structure from Algorithmic Probability"** (Wille, 2025). It provides:

- **Empirical validation** of Event-State Theory's prediction of cosmic structure formation.
- **Phase space analysis** revealing critical constants (**λ=0.48, β=3.4**) at a sharp phase boundary.
- **Falsifiable evidence** that can be independently verified by running the code or inspecting the raw data.

**For Peer Review:** The core claims can be tested by:
1.  Reproducing the phase transition shown in `/artifacts/01_phase_diagram.png`.
2.  Verifying that deviation from `(λ=0.48, β=3.4)` destroys cosmic web formation (using Mode B in the engine).
3.  Examining the linear baryon accumulation in `/artifacts/04_density_evolution_data.csv`.

---

## 🧪 The EST Computational Engine

**Research Author:** Torben Wille  
**Related Papers:** See `/papers/` directory for the complete research triad.

### 🌌 Abstract
The `EST_Laboratory.py` is a discrete 3D cellular automaton that tests the **Event-State Theory** framework. By using a thermodynamic Cost Function (`J`) based on **Conditional Kolmogorov Complexity**, it simulates:
*   **Asynchronous Multi-Point Crystallization (AMPC)**: Emergence of the cosmic web from void nucleation.
*   **Linear Baryonic Accumulation**: Validation of the "Scar Tissue" stability mechanism.
*   **Algorithmic Time Dilation**: The emergent variable frame rate of causal updates.

### 🔬 Experimental Modes
The engine operates in two distinct modalities:

*   **Mode A: Verified Replication (Default)**
    Runs the simulation using the precise `λ=0.48`, `β=3.4` constants identified in the phase-space analysis. This reproduces the 12-site Cosmic Web and linear density accumulation documented in the paper.

*   **Mode B: The Sandbox (Manual Configuration)**
    Allows for manual input of Grid Size, Frame Count, and Thermodynamic Constants.  
    *Purpose:* To enable independent verification of the "Edge of Chaos" boundary (i.e., demonstrating that modifying `λ` or `β` destroys the emergent structure).

### 🛠️ Installation & Usage
Requires **Python 3.8+** and standard scientific libraries.

```bash
# 1. Clone the repository
git clone https://github.com/SirWillance/Event-State-Theory-EST-.git
cd Event-State-Theory-EST-

# 2. Install dependencies
pip install numpy matplotlib scipy tqdm imageio

# 3. Run the Laboratory
python EST_Laboratory.py
```

---

## 📁 Repository Structure

Event-State-Theory-EST-/
├── README.md                          # This file
├── EST_Laboratory.py                  # Main simulation engine
├── /papers/                           # Formal research documents
│   ├── README.md                      # Guide to the paper triad
│   ├── 01_Event-State_Theory.pdf      # The foundational axioms (Spear)
│   ├── 02_Computational_Emergence.pdf # The empirical proof (Log)
│   └── 03_The_Event-State_Universe.pdf# The grand synthesis (Verdict)
├── /artifacts/                        # Raw results & evidence
│   ├── 01_phase_diagram.png
│   ├── 02_cosmic_web_snapshot.png
│   ├── 03_density_evolution_plot.png
│   ├── 04_density_evolution_data.csv
│   └── 05_parameter_sweep_data.csv
└── /validation/                       # For peer reviewers
    └── VALIDATION_PROTOCOL.md         # Step-by-step verification guide


---

## 📊 Empirical Results & Data

The `/artifacts/` directory contains the definitive results from a **High-Resolution Run (256³ Grid)**. This is the empirical foundation of the claims.

*   **Duration:** 440 Frames
*   **Resolution:** 16,777,216 nodes (16.7 million)
*   **Contents:**
    *   `01_phase_diagram.png` - The "Goldilocks Zone": Phase space analysis showing the critical boundary at `(λ=0.48, β=3.4)`.
    *   `02_cosmic_web_snapshot.png` - EST Deep Field: Visual proof of filamentary structure emergence (Frame 180).
    *   `03_density_evolution_plot.png` - Baryon Growth: Visualization of the linear "Scar Tissue" accumulation.
    *   `04_density_evolution_data.csv` - Raw Data: Machine-readable data for the density evolution plot.
    *   `05_parameter_sweep_data.csv` - Systematic Search: Data from the coarse-grained parameter sweep that identified the critical constants.

*These artifacts provide a complete chain of evidence from systematic parameter discovery to visual and quantitative validation of the theory's predictions.*

---

## 🔍 For Reviewers & Collaborators

Please see the `/validation/` directory for a detailed protocol on independently verifying the results. For questions about the theoretical framework or collaboration on the LHC pre-causal resonance prediction, please contact the author.


