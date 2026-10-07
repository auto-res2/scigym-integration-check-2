import numpy as np
import pandas as pd

df = experiment_history['iteration_0']
t = df['Time'].values
s_7 = df['id_7zj4'].values
s_3 = df['id_3vln'].values
s_5 = df['id_51dd'].values
s_j = df['id_jktq'].values

# Fit exponential to s_7:
log_s7 = np.log(s_7)
poly = np.polyfit(t, log_s7, 1)
print("k_decay of 7zj4:", -poly[0], "intercept:", poly[1])

# Calculate numerical derivatives
dt = np.diff(t)
ds7_dt = np.diff(s_7) / dt
ds3_dt = np.diff(s_3) / dt
ds5_dt = np.diff(s_5) / dt
dsj_dt = np.diff(s_j) / dt

# Midpoint values
m7 = 0.5 * (s_7[:-1] + s_7[1:])
m3 = 0.5 * (s_3[:-1] + s_3[1:])
m5 = 0.5 * (s_5[:-1] + s_5[1:])
mj = 0.5 * (s_j[:-1] + s_j[1:])

# Let's check ds7_dt / m7
print("ds7/dt / m7 min, max, mean:", np.min(ds7_dt/m7), np.max(ds7_dt/m7), np.mean(ds7_dt/m7))

# Now let's see ds3_dt:
# If 7zj4 -> 3vln -> ...
# ds3/dt = k1*m7 - rate_out
# What is rate_out?
rate_out_3 = - poly[0] * m7 - ds3_dt
# Let's see if rate_out_3 is proportional to m3 or something else:
print("rate_out_3 / m3:", rate_out_3[:10] / m3[:10], rate_out_3[100:110] / m3[100:110])
print("mean rate_out_3 / m3:", np.mean((rate_out_3 / m3)[10:1000]))

# Check if rate_out_3 goes to 51dd or jktq:
print("ds5_dt head:", ds5_dt[:10])
print("dsj_dt head:", dsj_dt[:10])
print("ds5/dt vs m3, m5:")
print("s_5 head:", s_5[:10])