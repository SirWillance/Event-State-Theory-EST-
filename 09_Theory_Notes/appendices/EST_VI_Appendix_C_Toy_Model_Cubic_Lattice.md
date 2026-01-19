### Supplementary Note: Toy Model of Lattice Anisotropy Cost and Emergent Lorentz Invariance on a Cubic Grid

This note presents a simple illustrative example of how local anisotropy in propagation speed on a discrete lattice can motivate an effective continuum description with Lorentz invariance. The treatment remains exploratory and serves to illustrate one possible concrete realization of the lattice-tax mechanism discussed in Event-State Theory VI.

#### 1\. Local Propagation Anisotropy on a Cubic Lattice

Consider a regular 3D cubic lattice with unit spacing. Causal influence (update signals) propagates with fixed hop rules:

*   Along lattice axes (x, y, z directions): speed normalized to c\_axis = 1 (one lattice unit per time step).
    
*   Along face or space diagonals: effective speed c\_diag ≈ √2 (face) or ≈ √3 (space), depending on allowed hops.
    

This leads to directional dependence in signal arrival times, breaking local Lorentz invariance at the discrete level.

Define a scalar measure of the anisotropy ("lattice tax") as the squared relative deviation from isotropy:

T\_L = (c\_diag / c\_axis − 1)²

For the space-diagonal case (c\_diag ≈ √3, c\_axis = 1), T\_L is non-zero and positive, representing an additional computational or causal cost associated with preferred directions.

#### 2\. Global Cost and Averaging

The total geometric cost functional in EST includes a contribution from local anisotropies:

J\_geom = ∑\_v ℓ(v)^{d−2} C\[Δ(v)\]

where C\[Δ\] penalizes the anisotropy tensor Δ^{ab} (as defined in the main paper and earlier appendices). In the toy cubic case, the scalar T\_L can be viewed as a simplified proxy for the leading quadratic term ½ κ Δ\_{ab} Δ^{ab}.

Under coarse-graining (block averaging over many lattice sites, ℓ → 0 with macroscopic scales fixed), persistent local anisotropies contribute to an average cost density. The system is assumed to select successor states that minimize (or probabilistically favor) low total J, including this geometric term.

#### 3\. Emergent Continuum Dynamics – Candidate Ansatz

One possible way to encode the macroscopic response is to introduce an effective metric field g\_{μν} that adjusts local causal structure so as to reduce the cumulative effect of residual anisotropy along causal paths.

A schematic dynamical relation can be postulated in which the curvature of the emergent geometry responds to the rate of change of the local tax and to residual deviation from minimal cost:

∇² g\_{μν} ∝ (∂ T\_L / ∂ n) × (J − J\_min)

where:

*   ∇² g\_{μν} represents leading curvature terms (schematic placeholder for components of the Einstein tensor or Ricci scalar variation),
    
*   ∂ T\_L / ∂ n is the rate of change of the anisotropy measure per discrete time step (or frame index n),
    
*   J − J\_min is the residual cost above the global minimum (informational or computational stress),
    
*   ∝ indicates that the precise coefficient and tensor structure remain to be determined.
    

This form suggests that non-vanishing local anisotropy (T\_L > 0) and its variation act as an effective source that drives deformation of the emergent metric until average residual cost and directional bias are minimized at large scales.

#### 4\. Relation to Full Emergent General Relativity

In the broader EST framework (see Event-State Theory VI and Appendices A–D), the leading infrared effective action is constrained by locality, second-order truncation, and emergent diffeomorphism invariance to the Einstein–Hilbert form:

S = ∫ d⁴x √−g (R − 2Λ) + matter terms

The toy-model equation above is not derived rigorously from the discrete cost functional but serves as a candidate bridge between local lattice tax and macroscopic field equations. It illustrates how anisotropy-induced cost could, in principle, feed into metric dynamics.

#### 5\. Limitations and Open Questions

*   The specific form T\_L = (c\_diag / c\_axis − 1)² is lattice-dependent and illustrative only.
    
*   The proposed relation ∇² g\_{μν} ∝ … lacks a controlled derivation from the full J minimization; tensor structure, coefficients, and stability are unspecified.
    
*   Explicit homogenization (map from discrete Δ^{ab} or T\_L to continuum h\_{μν}) and matching of effective constants (G\_eff, Λ) remain open technical tasks.
    
*   No claim is made that this toy model captures all features of the theory or provides uniqueness.