#!/usr/bin/env python
# coding: utf-8

# # Task 1.1 — Core Contribution / Architecture
# **Paper:** SGD-QN: Careful Quasi-Newton Stochastic Gradient Descent  
# **Authors:** Antoine Bordes, Léon Bottou, Patrick Gallinari (JMLR 2009)
# 
# ---
# 
# ## Step-by-Step Method Description
# 
# ### Step 1: Initialise parameters
# - **Description:** The weight vector `w₀` is set to zero. The diagonal rescaling matrix `B` is initialised to `λ⁻¹ · I` (a scalar multiple of identity). The regularisation counter `count = skip`, the flag `updateB = False`, and the running denominator `r = 2`.
# - **Reference:** Figure 2, SGD-QN pseudo-code lines 1–2; Section 5.3.
# - **Purpose:** Initialising `B = λ⁻¹I` means the very first pattern update is identical to first-order SGD. This guarantees safe convergence before any curvature estimates are computed.
# 
# ### Step 2: Pattern gradient update with diagonal rescaling
# - **Description:** For each randomly drawn example `(xₜ, yₜ)`, compute the loss derivative `ℓ'(yₜwₜᵀxₜ)` and update: `wₜ₊₁ = wₜ − (t+t₀)⁻¹ · ℓ'(yₜwₜᵀxₜ) · yₜ · B·xₜ`. Note that the gradient direction is rescaled component-wise by `B` before subtracting.
# - **Reference:** Equation (2) (generic SGD update with matrix B); Figure 2, line 4.
# - **Purpose:** Multiplying by `B ≈ H⁻¹` effectively normalises the gradient by per-feature curvature, reducing the condition number κ and allowing convergence in fewer passes.
# 
# ### Step 3: Diagonal B reestimation (when `updateB = True`)
# - **Description:** Using the *same* previous example `(xₜ₋₁, yₜ₋₁)`, compute `pₜ = gₜ₋₁(wₜ₊₁) − gₜ₋₁(wₜ)` (gradient difference at two consecutive parameter values). For each dimension `i`: `Bᵢᵢ ← Bᵢᵢ + (2/r)([wₜ₊₁ − wₜ]ᵢ/[pₜ]ᵢ − Bᵢᵢ)`, then clip to `[10⁻²λ⁻¹, λ⁻¹]`. Increment `r`.
# - **Reference:** Section 5.3, secant equation (Eq. 6); Figure 2, lines 5–9.
# - **Purpose:** This is the diagonal quasi-Newton update. Using the same example at two parameter values eliminates stochastic noise (cf. oLBFGS), while restricting `B` to diagonal keeps per-step cost at `O(d)`.
# 
# ### Step 4: Scheduled regularisation update (every `skip` iterations)
# - **Description:** When `count` reaches zero: `wₜ₊₁ ← wₜ₊₁ − skip·(t+t₀)⁻¹·λ·B·wₜ₊₁`. Reset `count = skip` and set `updateB = True`.
# - **Reference:** Figure 2, lines 11–14; Section 3 (scheduling trick); Table 2.
# - **Purpose:** Amortises the O(d) regularisation cost over `skip` iterations, reducing per-iteration complexity to `O(sd)` on sparse data. Piggybacking the `B` reestimation on the same schedule ensures the quasi-Newton overhead is also amortised.
# 
# ### Step 5: Return final weights
# - **Description:** After `T` iterations, return the parameter vector `wₜ` as the trained linear SVM.
# - **Reference:** Figure 2, line 18.
# - **Purpose:** Because fewer passes are needed (the condition-number factor κ² is reduced to κ or lower), the total training time is competitive with first-order SGD despite slightly higher cost per iteration.
# 
# ---
# 
# ## Final Summary Sentence
# 
# This paper solves the poor convergence of first-order SGD on ill-conditioned linear SVM problems; the authors claim their approach is better than existing alternatives because it obtains near-second-order convergence speed by estimating a diagonal inverse-Hessian approximation at nearly the same per-iteration cost as first-order SGD — by cleverly scheduling the expensive curvature reestimation at the same frequency as the regularisation update, so that the overhead is negligible while the pass-count reduction is substantial.
