import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize, curve_fit
import libsbml

df = experiment_history['iteration_0']
t = df['Time'].values
x = df['id_jxje'].values
y = df['id_sywq'].values
z = df['id_7cn0'].values
p = df['id_puar'].values

cs_x = CubicSpline(t, x)
cs_y = CubicSpline(t, y)
cs_z = CubicSpline(t, z)
cs_p = CubicSpline(t, p)

dx = cs_x(t, 1)
dy = cs_y(t, 1)
dz = cs_z(t, 1)
dp = cs_p(t, 1)

# Check d(log p)/dt vs x, y, z
dlogp = dp / p
# Fit dlogp = a + b*x + c*y + d*z
A = np.column_stack([np.ones_like(x), x, y, z])
coeff, residuals, rank, s = np.linalg.lstsq(A, dlogp, rcond=None)
print("dlogp fit with [1, x, y, z]:", coeff)
fit_dlogp = A @ coeff
print("dlogp max error:", np.max(np.abs(dlogp - fit_dlogp)))

# Let's check single variables for dlogp:
for name, val in [('x', x), ('y', y), ('z', z)]:
    c, r, _, _ = np.linalg.lstsq(np.column_stack([np.ones_like(val), val]), dlogp, rcond=None)
    print(f"dlogp vs {name}: slope={c[1]}, const={c[0]}, residual={np.sum((dlogp - (c[0]+c[1]*val))**2)}")

# Let's check if dlogp is proportional to x:
# Note x is id_jxje
c_x, _, _, _ = np.linalg.lstsq(x[:, None], dlogp, rcond=None)
print(f"dlogp = c * x: c = {c_x[0]}, max err = {np.max(np.abs(dlogp - c_x[0]*x))}")

# Now let's analyze the 3-variable system (x, y, z)
# What are typical 3-variable biological oscillators?
# 1. Repressilator:
# dx/dt = alpha / (1 + z^n) - beta * x  (or similar)
# dy/dt = alpha / (1 + x^n) - beta * y
# dz/dt = alpha / (1 + y^n) - beta * z
# Let's test negative feedback loop:
# Look at peaks:
print("Peaks of x:", t[np.where((dx[:-1] > 0) & (dx[1:] < 0))[0]])
print("Peaks of y:", t[np.where((dy[:-1] > 0) & (dy[1:] < 0))[0]])
print("Peaks of z:", t[np.where((dz[:-1] > 0) & (dz[1:] < 0))[0]])