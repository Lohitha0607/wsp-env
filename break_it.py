import numpy as np
from wsp_env.checker import DATA, accuracy

# 1. load the healthy patient
healthy = dict(np.load(DATA / "healthy.npz"))
print(f"Healthy: {accuracy(healthy):.0%}")

# 2. plant the bug: flip the signs of helper 5
sick = {name: table.copy() for name, table in healthy.items()}   # copy first!
sick["0.weight"][5] *= -1
print(f"Sick (helper 5 flipped): {accuracy(sick):.0%}")

# 3. save the sick patient = our first exam question
np.savez("sick.npz", **sick)
print("Saved sick.npz")
