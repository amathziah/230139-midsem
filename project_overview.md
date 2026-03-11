# Project Overview: Part B

## Objective
This part of the project focuses on implementing, analyzing, and comparing two different Stochastic Gradient Descent optimization algorithms applied to Support Vector Machines (with squared hinge loss and L2 regularization). 

The goal is to reproduce and study the claims made in a research paper regarding Quasi-Newton optimization methods by directly implementing the algorithms and tracking their convergence properties.

## The Two Algorithms
1. **SVMSGD2 (Standard SGD)**: A first-order Stochastic Gradient Descent algorithm that features a dynamic "skip scheduling" mechanism to efficiently amortize the cost of L2 regularization updates.
2. **SGD-QN (Quasi-Newton SGD)**: A more advanced algorithm extending standard SGD by utilizing a diagonal Quasi-Newton approximation of the Hessian matrix. By adjusting the step size per feature (using matrix `B`), it attempts to converge much faster on complex "ill-conditioned" datasets where features are highly correlated or scale differently.

## Tasks Breakdown

All the computational logic is divided among several Jupyter Notebooks, and can be run end-to-end via the `run_all.py` script. The specific milestones involve:

### 1. Dataset Generation (Task 2.1)
Using `scikit-learn` to purposefully generate artificial classification datasets (`make_classification`). The datasets are built with many redundant/correlated features to make the optimization landscape harder (ill-conditioned), providing a scenario where SGD-QN should outperform standard SVMSGD2.

### 2. Implementation & Evaluation (Tasks 2.2 & 2.3)
Implementing `svmsgd2` and `sgdqn` from scratch using NumPy. The training loops calculate and track the Primal Cost $P_n(w)$ over successive epochs to reproduce "Figure 3" from the source paper, showing that SGD-QN arrives at lower loss much quicker.

### 3. Ablation Studies (Task 3.1)
To understand *why* SGD-QN works, the algorithm is stripped of its key innovations one at a time to observe changes in test accuracy and primal cost:
- **Ablation 1 (No B-update)**: Fixing the $B$ matrix and removing the quasi-Newton curvature updates.
- **Ablation 2 (No Skip Schedule)**: Removing the skip scheduling and doing expensive regularization at every iteration.

### 4. Failure Mode Analysis (Task 3.2)
Engineering a specific dataset where SGD-QN offers no benefits. By creating a perfectly "well-conditioned" dataset (all informative features, zero redundancies), the Hessian matrix condition number approaches $\kappa \approx 1$. In this environment, the extra computational overhead of the Quasi-Newton $B$ matrix provides no advantage over standard SVMSGD2.

## Outputs
Everything outputs directly into the local `results/` folder, including JSON files tracking computation time and accuracy, as well as bar charts and line graphs visualizing the primal cost across epochs.
