# Event-State Theory: Computational Core

This directory contains the Python kernels used to validate the Event-State Theory (EST) framework.

## 1. The Cosmology Engine (`EST_Cosmology_Engine.py`)
**Purpose:** Simulates the formation of Large-Scale Structure (LSS) from thermodynamic first principles.
*   **Input:** Random Noise + 9 Asynchronous Nucleation Sites.
*   **Mechanism:** A 3D Cellular Automaton governed by the EST Cost Function ($J \approx \Delta E + K$).
*   **Output:** 
    *   Spontaneous emergence of the **Cosmic Web** (Filaments/Voids).
    *   Linear accumulation of **Baryonic Density** ("Scar Tissue").
    *   **Metric:** Uses `zlib` compression as a proxy for Kolmogorov Complexity ($K$).
*   **Usage:** Run the script. It generates a 3D rotating GIF and a density CSV log.

## 2. The Quantum Engine (`EST_PCR_Proof.py`)
**Purpose:** Demonstrates the emergence of Pre-Causal Resonance (PCR).
*   **Input:** A 1D vacuum timeline with a high-energy constraint at $t=500$.
*   **Mechanism:** A global relaxation algorithm minimizing the Cost Function across the timeline.
*   **Result:** The system spontaneously generates an **Exponential Ramp** ($t < 500$) to minimize the thermodynamic shock of the event.
*   **Significance:** Provides the specific waveform signature for the proposed LHC Minimum Bias analysis.

## Requirements
*   Python 3.8+
*   `numpy`
*   `matplotlib`
*   `imageio`

## Citation
If you use this code for verification or fork it for graph-based simulations, please cite:
*   Wille, T. (2025). *Computational Emergence of Cosmic Structure from Algorithmic Probability*. Zenodo.