#!/usr/bin/env python
# coding: utf-8

# # Task 2.1 — Dataset Selection and Setup
# 
# ## Dataset: Synthetic binary classification (sklearn `make_classification`)
# 
# The dataset is a 2000-sample, 30-feature binary classification problem generated with `make_classification`, configured with only 5 informative features and 20 redundant features (linear combinations of the informative ones). Labels are mapped to {−1, +1}. Features are standardised to zero mean and unit variance.
# 
# **Why this is a reasonable testbed:** The 20 redundant features create a Hessian of the primal cost (Eq. 1) with a large condition number κ, because many feature columns are near-collinear. This is exactly the regime where SGD-QN's diagonal quasi-Newton rescaling is theoretically beneficial — the paper's Table 1 shows that first-order SGD requires κ² more iterations than second-order SGD, and a large κ magnifies this gap. Dense features (sparsity s=1) also match the Alpha and Delta datasets in Table 4, which are the paper's main showcases for SGD-QN's advantage.
# 
# **Limitations compared to the paper's datasets:** The paper uses 100,000-example datasets (Alpha, Delta) with 500 features, evaluated across multiple training sizes. Our dataset has only 2,000 examples and 30 features, which means (a) the asymptotic regime analysed in Theorem 1 and Table 1 may not be fully reached, (b) the large-scale speedup of sparsity scheduling is not demonstrated, and (c) per-pass times are too short to reflect wall-clock advantages. Additionally, the paper reports results in terms of training *time* (seconds), whereas we can only compare *epochs* and final accuracy.
# 
# **Preprocessing:** Features are standardised with `StandardScaler` (mean=0, std=1). An 80/20 train-test split is used. Random seed 42 is fixed throughout.

# In[1]:


# ── Reproducibility ──────────────────────────────────────────────────────────
import numpy as np
import random
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os, time, json

os.makedirs('results', exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
# ill-conditioned: 20/30 features are redundant → high condition number κ
X_raw, y_raw = make_classification(
    n_samples=2000, n_features=30, n_informative=5,
    n_redundant=20, n_classes=2, flip_y=0.05,
    class_sep=0.8, random_state=SEED)
y = 2 * y_raw - 1   # {0,1} → {-1,+1}
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
print(f"Train: {X_train.shape}  Test: {X_test.shape}")
print(f"Class balance (train): +1={np.sum(y_train==1)}, -1={np.sum(y_train==-1)}")

