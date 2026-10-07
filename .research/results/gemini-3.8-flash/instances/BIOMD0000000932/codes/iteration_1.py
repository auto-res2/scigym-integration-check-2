import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

df = experiment_history['iteration_0']
print("Columns:", df.columns.tolist())
print("Time points count:", len(df))
print("Time range:", df['Time'].min(), "to", df['Time'].max())

# Let's inspect the growth of id_puar:
log_puar = np.log(df['id_puar'])
d_log_puar = np.gradient(log_puar, df['Time'])

print("Mean d(log(puar))/dt:", np.mean(d_log_puar))
print("Correlations with d(log(puar))/dt:")
for col in ['id_jxje', 'id_sywq', 'id_7cn0']:
    corr = np.corrcoef(df[col], d_log_puar)[0, 1]
    print(f"  {col}: {corr:.4f}")

# Let's check d(puar)/dt / puar vs species
# Print first 20 rows of df
print(df.head(15))