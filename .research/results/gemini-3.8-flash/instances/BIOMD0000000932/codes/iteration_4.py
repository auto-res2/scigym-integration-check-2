import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize, curve_fit

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

# Let's test standard oscillator models in SBML:
# Repressilator:
# dx/dt = alpha / (1 + z^n) + alpha0 - beta * x
# dy/dt = alpha / (1 + x^n) + alpha0 - beta * y
# dz/dt = alpha / (1 + y^n) + alpha0 - beta * z
# Let's test if dx/dt is related to z, dy/dt to x, dz/dt to y, or similar!
# Let's plot/print correlations between dx, dy, dz and various terms:
print("Correlations for dx:")
for name, v in [('x', x), ('y', y), ('z', z), ('1/(1+z^2)', 1/(1+z**2)), ('1/(1+y^2)', 1/(1+y**2)), ('1/(1+x^2)', 1/(1+x**2))]:
    print(f"  {name}: {np.corrcoef(dx, v)[0,1]:.4f}")

# What about Goodwin oscillator?
# Goodwin model:
# dx/dt = a / (k + z^n) - b * x
# dy/dt = c * x - d * y
# dz/dt = e * y - f * z
# In Goodwin:
# dy/dt = alpha * x - beta * y?
# Let's test dy = c * x - d * y:
A_y = np.column_stack([x, -y])
coeff_y, res_y, _, _ = np.linalg.lstsq(A_y, dy, rcond=None)
err_y = np.linalg.norm(dy - A_y @ coeff_y) / np.linalg.norm(dy)
print(f"dy = {coeff_y[0]:.4f} * x - {coeff_y[1]:.4f} * y, rel err: {err_y:.4f}")

# What about dz = e * y - f * z?
A_z = np.column_stack([y, -z])
coeff_z, res_z, _, _ = np.linalg.lstsq(A_z, dz, rcond=None)
err_z = np.linalg.norm(dz - A_z @ coeff_z) / np.linalg.norm(dz)
print(f"dz = {coeff_z[0]:.4f} * y - {coeff_z[1]:.4f} * z, rel err: {err_z:.4f}")

# What about other permutations?
# Let's test all pairs:
for v1_name, v1, dv in [('dx', x, dx), ('dy', y, dy), ('dz', z, dz)]:
    for u_name, u in [('x', x), ('y', y), ('z', z)]:
        if u_name != v1_name:
            A = np.column_stack([u, -v1])
            c, _, _, _ = np.linalg.lstsq(A, dv, rcond=None)
            err = np.linalg.norm(dv - A @ c) / np.linalg.norm(dv)
            print(f"d{v1_name}/dt = {c[0]:.4f} * {u_name} - {c[1]:.4f} * {v1_name}, rel err: {err:.4f}")