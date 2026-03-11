#!/usr/bin/env python3
"""
Part B — Run all computational tasks end-to-end.
Combines code from: task 2 1, task 2 2, task 2 3, task 3 1, task 3 2 notebooks.
Outputs results to results/ directory.
"""

# ── Reproducibility ──────────────────────────────────────────────────────────
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os, time, json

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.makedirs('results', exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# TASK 2.1 — Dataset Selection and Setup
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TASK 2.1 — Dataset Selection and Setup")
print("=" * 70)

X_raw, y_raw = make_classification(
    n_samples=2000, n_features=30, n_informative=5,
    n_redundant=20, n_classes=2, flip_y=0.05,
    class_sep=0.8, random_state=SEED)
y = 2 * y_raw - 1   # {0,1} → {-1,+1}
X = StandardScaler().fit_transform(X_raw)
X_tr, X_te = X[:1600], X[1600:]
y_tr, y_te = y[:1600], y[1600:]
print(f"Train: {X_tr.shape}  Test: {X_te.shape}")
print(f"Class balance (train): +1={np.sum(y_tr==1)}, -1={np.sum(y_tr==-1)}")

# ── Loss functions (Section 2.1, Eq.1) ───────────────────────────────────────
def sq_hinge_d(s):
    """Derivative of squared hinge loss: ℓ'(s) = -max(0, 1-s)"""
    return -np.maximum(0.0, 1.0 - s)

def primal_cost(w, X, y, lam):
    """Pₙ(w) = (λ/2)||w||² + (1/n)∑ 0.5·max(0,1-yᵢwᵀxᵢ)²   [Eq. 1]"""
    h = np.maximum(0, 1 - y*(X@w))
    return 0.5*lam*float(w@w) + float(np.mean(0.5*h*h))

# ══════════════════════════════════════════════════════════════════════════════
# TASK 2.2 — SVMSGD2 and SGD-QN Implementation
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TASK 2.2 — SVMSGD2 and SGD-QN Implementation")
print("=" * 70)

def svmsgd2(X, y, lam, t0, n_epochs, skip, seed=SEED):
    """Implements Figure 1 (right): SVMSGD2 first-order SGD with skip scheduling"""
    np.random.seed(seed); n, d = X.shape
    w = np.zeros(d); count = skip; t = 0; costs = []
    for _ in range(n_epochs):
        idx = np.random.permutation(n)
        for i in idx:
            xt, yt = X[i], y[i]
            dl = sq_hinge_d(yt * float(w @ xt))
            w = w - (1.0/(lam*(t+t0))) * dl * yt * xt
            count -= 1
            if count <= 0:
                w *= (1.0 - skip / (t + t0))
                count = skip
            t += 1
        costs.append(primal_cost(w, X, y, lam))
    return w, costs

def sgdqn(X, y, lam, t0, n_epochs, skip, seed=SEED, no_B_update=False, no_skip_schedule=False):
    """Implements Figure 2 (right): SGD-QN with diagonal quasi-Newton B update"""
    np.random.seed(seed); n, d = X.shape
    w = np.zeros(d)
    B = np.full(d, 1.0 / lam)
    r = 2; update_B = False; count = skip; t = 0; costs = []
    x_prev, y_prev = None, None

    for _ in range(n_epochs):
        idx = np.random.permutation(n)
        for i in idx:
            xt, yt = X[i], y[i]
            dl = sq_hinge_d(yt * float(w @ xt))
            w_new = w - (1.0/(t + t0)) * dl * yt * (B * xt)

            if (not no_B_update) and update_B and x_prev is not None:
                m1 = y_prev * float(w_new @ x_prev)
                m0 = y_prev * float(w     @ x_prev)
                g1 = lam*w_new + sq_hinge_d(m1)*y_prev*x_prev
                g0 = lam*w     + sq_hinge_d(m0)*y_prev*x_prev
                pt = g1 - g0; dw = w_new - w
                safe = np.abs(pt) > 1e-10
                ratio = np.where(safe, dw / np.where(safe, pt, 1.0), 1.0/lam)
                B += (2.0/r) * (ratio - B)
                B[:] = np.clip(B, 1e-2/lam, 1.0/lam)
                r += 1; update_B = False

            count -= 1
            if count <= 0:
                if not no_skip_schedule:
                    w_new = w_new - (skip/(t+t0)) * lam * (B * w_new)
                else:
                    w_new = w_new - (1.0/(t+t0)) * lam * w_new
                count = skip; update_B = True

            x_prev, y_prev = xt.copy(), yt
            w = w_new; t += 1
        costs.append(primal_cost(w, X, y, lam))
    return w, costs

# ── Hyperparameters ───────────────────────────────────────────────────────────
LAM  = 1e-2
T0   = 1000
SKIP = 16
EPOCHS = 20

# Run SVMSGD2
ts = time.time(); w_s, c_s = svmsgd2(X_tr, y_tr, LAM, T0, EPOCHS, SKIP); time_s = time.time() - ts
# Run SGD-QN
ts = time.time(); w_q, c_q = sgdqn(X_tr, y_tr, LAM, T0, EPOCHS, SKIP); time_q = time.time() - ts
a_s = accuracy_score(y_te, np.sign(X_te@w_s))
a_q = accuracy_score(y_te, np.sign(X_te@w_q))
print(f"SVMSGD2: final_cost={c_s[-1]:.5f}  test_acc={a_s:.4f}  time={time_s:.3f}s")
print(f"SGD-QN:  final_cost={c_q[-1]:.5f}  test_acc={a_q:.4f}  time={time_q:.3f}s")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 2.3 — Primal Cost Plot (replicating Figure 3)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TASK 2.3 — Primal Cost Plot (Figure 3 reproduction)")
print("=" * 70)

ep = np.arange(1, EPOCHS+1)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(ep[1:], c_s[1:], 'b-o', ms=4, label='SVMSGD2 (1st-order)')
ax.plot(ep[1:], c_q[1:], 'r-s', ms=4, label='SGD-QN (diagonal QN)')
ax.set_xlabel('Epoch'); ax.set_ylabel('Primal cost Pₙ(w)')
ax.set_title(f'Primal cost vs epochs — λ={LAM}, skip={SKIP}\nmake_classification (30 features, 20 redundant)')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/task2_primal_cost.png', dpi=150)
print("Saved results/task2_primal_cost.png")

# Save Task 2 results as JSON
task2_results = {
    "svmsgd2": {"final_cost": c_s[-1], "test_accuracy": a_s, "time_seconds": time_s, "cost_per_epoch": c_s},
    "sgdqn":   {"final_cost": c_q[-1], "test_accuracy": a_q, "time_seconds": time_q, "cost_per_epoch": c_q}
}
with open('results/task2_results.json', 'w') as f:
    json.dump(task2_results, f, indent=2)
print("Saved results/task2_results.json")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 3.1 — Two-Component Ablation Study
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TASK 3.1 — Ablation Study")
print("=" * 70)

# Full SGD-QN (already computed)
w_full, c_full = w_q, c_q
a_full = a_q

# Ablation 1: B fixed at λ⁻¹I throughout (no quasi-Newton update)
print("\nAblation 1: Remove B update (fix B = λ⁻¹I)")
w_ab1, c_ab1 = sgdqn(X_tr, y_tr, LAM, T0, EPOCHS, SKIP, no_B_update=True)
a_ab1 = accuracy_score(y_te, np.sign(X_te@w_ab1))
print(f"  Full SGD-QN:    cost={c_full[-1]:.5f}  acc={a_full:.4f}")
print(f"  Ablation 1:     cost={c_ab1[-1]:.5f}  acc={a_ab1:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(ep[1:], c_full[1:], 'r-s', ms=4, label='Full SGD-QN')
axes[0].plot(ep[1:], c_ab1[1:], 'b-o', ms=4, label='Ablation 1: B=λ⁻¹I (no update)')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Primal cost')
axes[0].set_title('Ablation 1: Remove B update'); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].bar(['Full SGD-QN', 'No B update'], [a_full, a_ab1], color=['red', 'blue'], alpha=0.7)
axes[1].set_ylabel('Test accuracy'); axes[1].set_ylim(0.5, 0.9)
axes[1].set_title('Test accuracy comparison')
for i, (v, c) in enumerate(zip([a_full, a_ab1], ['red', 'blue'])):
    axes[1].text(i, v+0.005, f'{v:.3f}', ha='center')
axes[1].grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('results/task3_ablation1.png', dpi=150)
print("Saved results/task3_ablation1.png")

# Ablation 2: Remove skip scheduling (regularise every step)
print("\nAblation 2: Remove skip schedule (regularise every step)")
w_ab2, c_ab2 = sgdqn(X_tr, y_tr, LAM, T0, EPOCHS, SKIP, no_skip_schedule=True)
a_ab2 = accuracy_score(y_te, np.sign(X_te@w_ab2))
print(f"  Full SGD-QN:    cost={c_full[-1]:.5f}  acc={a_full:.4f}")
print(f"  Ablation 2:     cost={c_ab2[-1]:.5f}  acc={a_ab2:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(ep[1:], c_full[1:], 'r-s', ms=4, label='Full SGD-QN (skip=16)')
axes[0].plot(ep[1:], c_ab2[1:], 'g-^', ms=4, label='Ablation 2: regularise every step')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Primal cost')
axes[0].set_title('Ablation 2: Remove skip scheduling'); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].bar(['Full SGD-QN', 'No skip schedule'], [a_full, a_ab2], color=['red', 'green'], alpha=0.7)
axes[1].set_ylabel('Test accuracy'); axes[1].set_ylim(0.5, 0.9)
axes[1].set_title('Test accuracy comparison')
for i, v in enumerate([a_full, a_ab2]):
    axes[1].text(i, v+0.005, f'{v:.3f}', ha='center')
axes[1].grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('results/task3_ablation2.png', dpi=150)
print("Saved results/task3_ablation2.png")

# Accuracy comparison bar chart
fig, ax = plt.subplots(figsize=(7, 4))
labels = ['Full SGD-QN', 'No B update\n(Ablation 1)', 'No skip schedule\n(Ablation 2)']
vals = [a_full, a_ab1, a_ab2]
colors = ['red', 'blue', 'green']
bars = ax.bar(labels, vals, color=colors, alpha=0.7)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.005, f'{v:.3f}', ha='center')
ax.set_ylabel('Test accuracy'); ax.set_ylim(0.5, 0.9)
ax.set_title('Accuracy comparison: Full SGD-QN vs ablations')
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('results/task3_accuracy_comparison.png', dpi=150)
print("Saved results/task3_accuracy_comparison.png")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 3.2 — Failure Mode Analysis (Well-conditioned Hessian)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TASK 3.2 — Failure Mode Analysis (Well-conditioned κ≈1)")
print("=" * 70)

X_fail, y_fail = make_classification(
    n_samples=2000, n_features=30, n_informative=28,
    n_redundant=0, n_repeated=0, n_classes=2, flip_y=0.0,
    class_sep=3.0, random_state=SEED)
y_fail = 2 * y_fail - 1
X_fail = StandardScaler().fit_transform(X_fail)
Xf_tr, Xf_te = X_fail[:1600], X_fail[1600:]
yf_tr, yf_te = y_fail[:1600], y_fail[1600:]

LARGE_LAM = 0.5
w_fail_s, c_fail_s = svmsgd2(Xf_tr, yf_tr, LARGE_LAM, T0, EPOCHS, SKIP)
w_fail_q, c_fail_q = sgdqn(Xf_tr, yf_tr, LARGE_LAM, T0, EPOCHS, SKIP)
a_fs = accuracy_score(yf_te, np.sign(Xf_te@w_fail_s))
a_fq = accuracy_score(yf_te, np.sign(Xf_te@w_fail_q))
print(f"Well-conditioned (λ={LARGE_LAM}):")
print(f"  SVMSGD2: cost={c_fail_s[-1]:.5f}  acc={a_fs:.4f}")
print(f"  SGD-QN:  cost={c_fail_q[-1]:.5f}  acc={a_fq:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(ep[1:], c_fail_s[1:], 'b-o', ms=4, label='SVMSGD2')
axes[0].plot(ep[1:], c_fail_q[1:], 'r-s', ms=4, label='SGD-QN')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Primal cost')
axes[0].set_title(f'Failure mode: λ={LARGE_LAM} (well-conditioned)\n28/30 informative features, no redundancy')
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].bar(['SVMSGD2', 'SGD-QN'], [a_fs, a_fq], color=['blue', 'red'], alpha=0.7)
for i, v in enumerate([a_fs, a_fq]):
    axes[1].text(i, v+0.001, f'{v:.4f}', ha='center')
axes[1].set_ylabel('Test accuracy'); axes[1].set_ylim(0.9, 1.01)
axes[1].set_title('Test accuracy (near-perfect data)')
axes[1].grid(alpha=0.3, axis='y')
plt.suptitle('SGD-QN failure: well-conditioned Hessian (κ≈1)', y=1.02)
plt.tight_layout()
plt.savefig('results/task3_failure_mode.png', dpi=150, bbox_inches='tight')
print("Saved results/task3_failure_mode.png")

# Save Task 3 results as JSON
task3_results = {
    "ablation1_no_B_update": {"final_cost": c_ab1[-1], "test_accuracy": a_ab1},
    "ablation2_no_skip_schedule": {"final_cost": c_ab2[-1], "test_accuracy": a_ab2},
    "full_sgdqn": {"final_cost": c_full[-1], "test_accuracy": a_full},
    "failure_mode": {
        "dataset": "well-conditioned (28/30 informative, λ=0.5)",
        "svmsgd2": {"final_cost": c_fail_s[-1], "test_accuracy": a_fs},
        "sgdqn": {"final_cost": c_fail_q[-1], "test_accuracy": a_fq}
    }
}
with open('results/task3_results.json', 'w') as f:
    json.dump(task3_results, f, indent=2)
print("Saved results/task3_results.json")

print("\n" + "=" * 70)
print("ALL TASKS COMPLETE — Results saved to results/")
print("=" * 70)
