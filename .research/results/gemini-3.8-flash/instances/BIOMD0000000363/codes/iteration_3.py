import numpy as np
import pandas as pd
from scipy.optimize import minimize

df = experiment_history['iteration_0']
species = ['id_7zj4', 'id_3vln', 'id_51dd', 'id_jktq']
X = df[species].values
t = df['Time'].values

# Calculate numerical derivatives with central differences or spline
from scipy.interpolate import CubicSpline
cs = CubicSpline(t, X)
dX = cs(t, 1) # first derivative

# If dX = K @ X (where K has column sums = 0)
# Let's fit dX_i = sum_j K_ij * X_j
# Using ridge/lasso or unconstrained least squares:
K, residuals, rank, s = np.linalg.lstsq(X, dX, rcond=None)
# K is (n_species, n_species), where dX ≈ X @ K, so dX/dt = K^T X
print("Matrix K^T:")
print(pd.DataFrame(K.T, index=species, columns=species))

# Check max absolute error of linear model dX/dt = X @ K
pred_dX = X @ K
print("Max abs error in dX/dt:", np.max(np.abs(dX - pred_dX)))
print("Relative error:", np.max(np.abs(dX - pred_dX)) / np.max(np.abs(dX)))