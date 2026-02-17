import numpy as np


P = np.array([[0.5, 0, 0, 0.5], [0.5, 0, 0, 0.5], [0.5, 0, 0, 0.5], [0.4, 0.4, 0.2, 0]])


X0 = np.array([0.25, 0.25, 0.25, 0.25])

p10 = X0 @ np.linalg.matrix_power(P, 10)

print(p10)
