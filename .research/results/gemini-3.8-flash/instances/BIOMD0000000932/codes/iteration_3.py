import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

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

# Let's inspect dp / dt:
# Is dp/dt = k * p * z? or k * p * something?
# Let's check dp / (p * z):
print("dp / (p * z) head:", (dp / (p * z))[:10])
print("dp / (p * x) head:", (dp / (p * x))[:10])
print("dp / (p * y) head:", (dp / (p * y))[:10])

# What about dp/dt = k * z * ...? Or Monod kinetics?
# dp/dt = mu_max * z / (K + z) * p ?
# Let's check:
for i in range(10):
    print(f"t={t[i]:.2f}: x={x[i]:.3f}, y={y[i]:.3f}, z={z[i]:.3f}, p={p[i]:.3e}, dlogp={dp[i]/p[i]:.4f}")

# Let's check the relation between dlogp and (x, y, z)
# Could dlogp = mu(z) or mu(y) or mu(x)?
# Look at t=0: dlogp?
# Let's print dlogp values:
dlogp = dp / p
print("dlogp values:", dlogp[:15])
print("z values:", z[:15])
print("ratio dlogp / z:", (dlogp / z)[:15])
print("ratio dlogp / y:", (dlogp / y)[:15])
print("ratio dlogp / x:", (dlogp / x)[:15])