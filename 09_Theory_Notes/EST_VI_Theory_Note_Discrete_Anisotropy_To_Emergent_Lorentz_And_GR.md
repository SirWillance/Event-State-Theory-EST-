# EST VI Theory Note: From Discrete Anisotropy and Causal Optimization to Emergent Lorentz Invariance and General Relativity

## Purpose
This note combines technical details on local anisotropies in discrete causal update structures (leading to emergent Lorentz invariance) with the broader derivation of Einstein gravity as the infrared effective theory. All assumptions are labeled; no uniqueness or fundamentality claims are made. See Appendix A for a refined version addressing time tensions.

---

## Discrete Update Geometry and Anisotropy
Consider a discrete causal graph with vertices \(v\) and finite update stencils \(\mathcal{S}(v)\). Each update propagates along edges with bounded hops.

A spatial embedding \(x(v) \in \mathbb{R}^3\) defines local direction statistics (bookkeeping only, not fundamental).

### Directional Statistics
For \(v' \in \mathcal{S}(v)\), the unit direction vector:
\[
\hat{n}^a(v \to v') = \frac{x^a(v') - x^a(v)}{\|x(v') - x(v)\|}.
\]

Anisotropy tensor:
\[
\Delta^{ab}(v) = \frac{1}{|\mathcal{S}(v)|} \sum_{v'} \left( n^a n^b - \frac{1}{3} \delta^{ab} \right).
\]

Vanishes iff isotropic.

---

## Geometric Cost (Lattice Tax)
**Assumption A1 (Computational Cost):** Anisotropy increases cost.

Geometric contribution:
\[
J_{\mathrm{geom}} = \sum_v \ell(v)^{d-2} C[\Delta(v)].
\]

Leading term (rotational invariance, locality):
\[
C[\Delta] = \frac{\kappa}{2} \Delta_{ab} \Delta^{ab} + O(\Delta^3).
\]

---

## Coarse-Graining Procedure
Partition into blocks; average anisotropy:
\[
\overline{\Delta}^{ab}(B) = \frac{1}{N_B} \sum_{v \in B} \Delta^{ab}(v).
\]

Limit \(N_B \to \infty\), \(\ell \to 0\), fixed macro scale. Yields smooth \(\Delta^{ab}(x)\); averages to zero under generic conditions.

Vertex relabeling induces diffeomorphism invariance.

---

## Emergent Continuum Field and Lorentz Invariance
Smooth field \(\Delta^{ab}(x)\) from averages.

**Assumption A2 (Metric Absorption):** Bias absorbed into emergent metric \(g_{\mu\nu} = \eta_{\mu\nu} + h_{\mu\nu}\).

Propagation depends on connections: \(\Gamma \sim \partial h\), so \(\Delta \sim \mathcal{D}(\partial h)\).

Effective action:
\[
S_{\mathrm{geom}} \sim \int d^4x \, (\partial h)^2.
\]

Lorentz invariant in continuum, despite microscopic breaking.

---

## Discrete Event-State Sequence and Broader Emergence
Sequence \(\{ \Psi_n \}\) (ordering only, no fundamental time). Each with DAG.

### Emergent Temporal Ordering
Ordering induces foliation \(\Sigma_n \to \Sigma_\tau\).

### Causal Cones
Bounded updates imply null cones.

### Metric Degrees of Freedom
Effective Lorentzian \(g_{\mu\nu}\) encodes causal/distance relations.

---

## Effective Action Assumptions
**Assumption G1:** Locality.  
**Assumption G2:** Second-order equations.

Leads to Lovelock form.

### Einstein–Hilbert Action (4D)
\[
S = \int d^4x \sqrt{-g} (R - 2\Lambda).
\]

### Field Equations
\[
G_{\mu\nu} + \Lambda g_{\mu\nu} = 8\pi G_{\mathrm{eff}} T_{\mu\nu}.
\]

### Parameter Matching
\(G_{\mathrm{eff}} \sim \frac{\ell^2}{\kappa}\).

---

## Open Technical Steps
- Explicit \(\mathcal{D}\) for update rules.
- Coefficient matching.
- Matter inclusion.

---

## Explicit Non-Claims
- No Genesis-level dynamics derivation.
- No uniqueness.
- No coupling constant predictions.
- No fundamental Lorentz invariance or spacetime.