# Data README

## Dataset used in Part B

All notebooks in Part B use a **synthetic binary classification dataset** generated entirely by `sklearn.datasets.make_classification`. No external files are needed.

### Generation parameters

```python
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

X_raw, y_raw = make_classification(
    n_samples=2000,
    n_features=30,
    n_informative=5,
    n_redundant=20,
    n_classes=2,
    flip_y=0.05,
    class_sep=0.8,
    random_state=42
)
y = 2 * y_raw - 1  # {0,1} -> {-1,+1}
X = StandardScaler().fit_transform(X_raw)
```

### Rationale

The 20 redundant features (linear combinations of the 5 informative features) create an ill-conditioned Hessian with a large condition number κ, which is exactly the regime where SGD-QN's diagonal quasi-Newton update provides a convergence advantage over first-order SGD (per Table 1 of the paper).

### No manual steps required

Running any notebook from top to bottom will generate the dataset automatically. No downloads or file copies are needed.
