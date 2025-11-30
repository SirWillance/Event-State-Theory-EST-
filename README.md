```markdown
# Event-State Theory (EST): A Computational Framework
*A Discrete Causal-Information Framework for Cosmology and Fundamental Physics*

## 🔭 Project Overview

This repository hosts the theoretical architecture, computational engines, and empirical proofs for **Event-State Theory (EST)**. EST posits that the universe is not a continuous geometric manifold, but a discrete computational optimization process governed by the **Principle of Computational Least Action**.

By minimizing a thermodynamic Cost Function (`J`) based on **Conditional Kolmogorov Complexity**, this framework unifies the macro-scale geometry of Cosmology (Hubble Tension) with the micro-scale mechanics of Quantum Dynamics.

---

## 📚 Research Library
*The complete theoretical and empirical documentation is available in the `/Papers` directory.*

*   **📄 [00_Executive_Synthesis.pdf](Papers/00_Executive_Synthesis.pdf)** - *Start Here.* A 2-page summary of the theory, the simulation results, and the predictions.
*   **📄 [01_Theoretical_Framework.pdf](Papers/01_Theoretical_Framework_v1.pdf)** - The original topological proposal identifying AMPC as the solution to the Hubble Tension.
*   **📄 [02_Computational_Proof.pdf](Papers/02_Computational_Proof.pdf)** - Empirical validation via $256^3$ and $512^3$ cellular automaton simulations.
*   **📄 [03_LHC_Experiment_Proposal.pdf](Papers/03_LHC_Experiment_Proposal.pdf)** - Methodology for detecting Pre-Causal Resonance (PCR) in LHC Minimum Bias data.
*   **📄 [04_The_Event_State_Protocols.pdf](Papers/04_The_Event_State_Protocols_Monograph.pdf)** - The comprehensive monograph detailing the full axiomatic system.

---

## 🧪 The Computational Engines
*Source code located in `/Simulations`*

### 1. The Cosmology Engine (`EST_Cosmology_Engine.py`)
A discrete 3D cellular automaton that tests the macro-scale formation of the universe.
*   **Mechanism:** Asynchronous Multi-Point Crystallization (AMPC).
*   **Results:**
    *   Spontaneous emergence of the **Cosmic Web** from random noise.
    *   **Baryon Asymmetry:** Reproduces $\eta \approx 7 \times 10^{-10}$ (matching Planck 2018) via "Scar Tissue" accumulation.
    *   **Void Fraction:** ~80-99% depending on nucleation density ($N_0$).

### 2. The Quantum Engine (`EST_PCR_Proof.py`)
A relaxation algorithm testing the micro-scale causal dynamics.
*   **Mechanism:** Global optimization of a 1D timeline under a high-energy constraint.
*   **Results:**
    *   Spontaneous generation of **Pre-Causal Resonance (PCR)**.
    *   Predicts an exponential "vacuum ramp" ($t < 0$) detectable in high-energy collisions.

---

## 🔬 Key Empirical Findings
*The code reproduces observational reality at the "Edge of Chaos" phase boundary (`λ=0.48`, `β=3.4`).*

| Observable | EST Simulation | Observed Reality | Status |
| :--- | :--- | :--- | :--- |
| **Cosmic Structure** | Filamentary Web | Filamentary Web | ✅ Match |
| **Baryon Asymmetry** | $7.07 \times 10^{-10}$ | $6.12 \times 10^{-10}$ | ✅ Match |
| **Expansion Topology** | Anisotropic | Hubble Tension | ✅ Match |
| **Vacuum Response** | Exponential Ramp | Ridge Effect (?) | ⏳ Proposed |

---

## ⚠️ Limitations & Roadmap
*   **Grid Bias:** The current prototypes utilize linear compression algorithms (`zlib`) on a fixed Cartesian lattice. This introduces anisotropic artifacts ("Minecraft Physics") in propagation velocity.
*   **Phase III:** Future work will migrate the kernel to **Dynamic Causal Graphs** to recover full Spherical Special Relativity.

---

## 🛠️ Installation & Usage
Requires **Python 3.8+** and standard scientific libraries.

```bash
# 1. Clone the repository
git clone https://github.com/SirWillance/Event-State-Theory-EST-.git
cd Event-State-Theory-EST-

# 2. Install dependencies
pip install numpy matplotlib scipy tqdm imageio

# 3. Run the Cosmology Simulation
python Simulations/EST_Cosmology_Engine.py
```

## 📬 Contact & Citation
**Torben Wille**  
*Independent Systems Researcher*  
Research artifacts archived on [Zenodo](https://zenodo.org/records/17698703).
```
