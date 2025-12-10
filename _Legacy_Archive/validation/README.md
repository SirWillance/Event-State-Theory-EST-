# EST Validation Suite

This directory contains the complete robustness and invariance proofs for Event-State Theory (EST), addressing peer critiques from Phase 3 review. All results are reproducible—clone the repo, run `python EST_Engine_Enhanced.py`, select menu 5 (invariance) or 6 (scaling).

## Key Proofs
- **Algorithmic Invariance:** Cosmic web emerges identically across zlib, LZMA, BZIP2, and Shannon entropy. Max density variance: 0.000344.  
  - Log: [invariance/experiment_log.txt](invariance/experiment_log.txt)  
  - Graph: [invariance/Algorithmic_Invariance_Comparison.png](invariance/Algorithmic_Invariance_Comparison.png)  
  - Raw Data: [invariance/raw_densities.csv](invariance/raw_densities.csv) (if you have CSVs; add them below)

- **Nucleation Scaling:** Web robust over 8–40 sites (5x range). Density linear; structure quality stable 0.47–0.55.  
  - Log: [nucleation/experiment_log.txt](nucleation/experiment_log.txt)  
  - Graph: [nucleation/Nucleation_Scaling_Analysis.png](nucleation/Nucleation_Scaling_Analysis.png)  
  - Raw Data: [nucleation/raw_scaling.csv](nucleation/raw_scaling.csv)

---

## How to Reproduce
1. Install deps: `pip install numpy matplotlib scipy imageio tqdm`  
2. Run: `python Simulations/EST_Engine_Enhanced.py` → menu 5 or 6.  
3. Outputs in /EST_Output_* folders.

These validations confirm EST's axiomatic strength (see Paper 03 on Zenodo). Questions? Open an issue.

---

### Scope of Validation Results
Current robustness tests apply to cosmological-scale emergence and
thermodynamic causal dynamics within discrete 3D lattices.
Quantum-scale PCR dynamics and localized τ-collapse (event horizon formation)
are scheduled for Phase IV validation and are *not yet claimed as solved*.

---

## Disclaimer:
This engine is designed for scientific falsification of EST.
Results are not assumed to be true until independently reproduced.

---



— Torben Wille, Dec 1, 2025


