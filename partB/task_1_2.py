#!/usr/bin/env python
# coding: utf-8

# # Task 1.2 — Key Assumptions
# **Paper:** SGD-QN: Careful Quasi-Newton Stochastic Gradient Descent
# 
# ---
# 
# ## Assumption 1
# 
# **Assumption:** The loss function `ℓ(s)` is convex and twice continuously differentiable (`ℓ ∈ C²[ℝ]`), so the primal cost `Pₙ(w)` has a positive definite Hessian `H` at the optimum.
# 
# **Why the method needs it:** The secant equation (Eq. 6) underpinning the `B` update is derived from a second-order Taylor expansion, which is only valid when the Hessian exists and is continuous. Without this, the ratio `[wₜ − wₜ₋₁]ᵢ / [pₜ]ᵢ` used in the leaky-average update (Figure 2, line 7) has no curvature interpretation, and Theorem 1's convergence rate bound does not hold.
# 
# **Violation scenario:** The standard hinge loss `max(0, 1−s)` has zero second derivative almost everywhere and is non-differentiable at `s=1`. A practitioner insisting on the standard hinge loss without smoothing would find that the curvature estimates degenerate, because the denominator `[pₜ]ᵢ` is frequently zero, causing the `B` update to be numerically undefined or to fall back to the `λ⁻¹` default everywhere.
# 
# **Paper reference:** Section 2.1 (explicit assumption: *"we assume in this paper that the loss ℓ(s) is convex and twice differentiable"*); Section 6 (experiments switch to squared hinge loss for this reason).
# 
# ---
# 
# ## Assumption 2
# 
# **Assumption:** The inverse Hessian `H⁻¹` is well-approximated by a diagonal matrix, meaning the dominant curvature structure of the problem lies along the feature axes rather than in cross-feature directions.
# 
# **Why the method needs it:** SGD-QN deliberately restricts its rescaling matrix `B` to be diagonal (Section 5.1, 5.3) to keep the per-iteration cost at `O(d)` rather than `O(d²)`. If the true `H⁻¹` has significant off-diagonal mass — i.e. features are strongly correlated — the diagonal of `H⁻¹` is a poor preconditioner, and Theorem 1 shows the convergence constant still scales unfavourably with the residual condition number.
# 
# **Violation scenario:** A bag-of-words text classification dataset where synonymous terms co-occur at near-identical frequencies (e.g. "automobile" and "car") creates a near-rank-deficient Hessian with large off-diagonal blocks. The diagonal entries of `H⁻¹` underestimate the inverse curvature in the shared synonym subspace, so the `B` rescaling provides little improvement over plain SGD.
# 
# **Paper reference:** Section 5.1 (explicit design choice to restrict to diagonal `B`); Section 5.3 (derivation of diagonal secant update); Table 1 (convergence factor `κ` when `B ≠ H⁻¹`).
# 
# ---
# 
# ## Assumption 3
# 
# **Assumption:** The curvature of the primal cost changes slowly relative to the `skip` interval, so that a stale diagonal `B` — updated only every `skip` iterations — still provides useful preconditioning between updates.
# 
# **Why the method needs it:** The efficiency of SGD-QN comes from amortising the `B` reestimation cost by updating it only every `skip` iterations (Figure 2, lines 11–14). This is valid only if the Hessian is approximately stationary over `skip` steps. If the true curvature changes faster than the update frequency, `B` lags behind the current geometry and can misdirect the gradient.
# 
# **Violation scenario:** A non-stationary streaming dataset — for instance, a financial time-series classification task where volatility regimes shift abruptly — would cause the Hessian to change substantially within a few dozen examples. With a large `skip` value (e.g. `skip=16` on a 1600-example dataset), `B` would remain calibrated to the old regime for an entire sub-epoch, actively worsening convergence compared to a method that adapts its learning rate at every step.
# 
# **Paper reference:** Section 5.3 (*"since a fixed diagonal rescaling matrix already works quite well, there is little need to update its coefficients very often"*); Figure 2 (lines 11–14); Section 6.1 (Table 5, choice of `skip` parameter).
