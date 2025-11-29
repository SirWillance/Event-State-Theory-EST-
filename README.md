# Event-State Theory (EST): A Computational Framework
*A Discrete Causal-Information Framework for Cosmology and Fundamental Physics*

## 🔬 Scientific Context

This repository contains the computational engine and empirical results for the paper **"Computational Emergence of Cosmic Structure from Algorithmic Probability"** (Wille, 2025). It provides:

- **Empirical validation** of Event-State Theory's prediction of cosmic structure formation.
- **Phase space analysis** revealing critical constants (**λ=0.48, β=3.4**) at a sharp phase boundary.
- **Falsifiable evidence** that can be independently verified by running the code or inspecting the raw data.

**For Peer Review:** The core claims can be tested by:
1.  Reproducing the phase transition shown in `/artifacts/01_phase_diagram.png`.
2.  Verifying that deviation from `(λ=0.48, β=3.4)` destroys cosmic web formation (using Mode B).
3.  Examining the linear baryon accumulation in `/artifacts/04_density_evolution_data.csv`.

---

## 🧪 The EST Computational Engine

**Research Author:** Torben Wille
**Related Papers:** Zenodo DOIs provided upon publication.

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
