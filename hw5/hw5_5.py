import numpy as np
import matplotlib.pyplot as plt

P = np.array([[1 / 3, 2 / 3], [1, 0]])

# Show: irreducible, aperiodic, P^m PD for m>=2, lim m-> inf P^m


# aperiodic
P_inf = np.linalg.matrix_power(P, 100)
ss = np.array([0.75, 0.25]) @ P_inf

print(f"P2: {np.linalg.matrix_power(P, 2)}")


print(f"lim p to inf: {P_inf}")
