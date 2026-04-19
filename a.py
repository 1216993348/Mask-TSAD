# check_dim23.py
import sys
sys.path.insert(0, 'src')
import numpy as np
from data_loader import DataLoader

loader = DataLoader()
data = loader.load_dataset('MSL')

X_normal = data['X_test'][data['y_test'] == 0]
X_anomaly = data['X_test'][data['y_test'] == 1]

std_n = X_normal.std(axis=0)
std_a = X_anomaly.std(axis=0)

var_ratio = std_a / std_n

print(f"Dim 23: normal_std={std_n[23]:.6f}, anomaly_std={std_a[23]:.6f}, ratio={var_ratio[23]:.2f}")

# 找 top 5 var_ratio
top5 = np.argsort(var_ratio)[-5:][::-1]
print(f"\nTop 5 var_ratio:")
for d in top5:
    print(f"  Dim {d}: {var_ratio[d]:.2f}")