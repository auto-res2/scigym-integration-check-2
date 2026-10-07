import numpy as np
import pandas as pd

df = experiment_history['iteration_0']
print("Columns:", df.columns.tolist())
print("Data head:\n", df.head(10))
print("Sum of species at various times:")
df['sum'] = df['id_51dd'] + df['id_jktq'] + df['id_7zj4'] + df['id_3vln']
print(df[['Time', 'sum', 'id_7zj4', 'id_3vln', 'id_51dd', 'id_jktq']].iloc[::1000])

# Check initial derivatives:
dt = df['Time'].iloc[1] - df['Time'].iloc[0]
print("dt:", dt)
for col in ['id_7zj4', 'id_3vln', 'id_51dd', 'id_jktq']:
    d = (df[col].iloc[1] - df[col].iloc[0]) / dt
    print(f"Initial d({col})/dt: {d}")
    
# Max of id_51dd
print("Max id_51dd:", df['id_51dd'].max(), "at time", df.loc[df['id_51dd'].idxmax(), 'Time'])
print("Max id_3vln:", df['id_3vln'].max(), "at time", df.loc[df['id_3vln'].idxmax(), 'Time'])