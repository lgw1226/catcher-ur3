import numpy as np


a = np.random.randn(5) * 5
ub = np.ones(5)
lb = -np.ones(5)

clip_a = np.clip(a, lb, ub)

print(a)
print(clip_a)