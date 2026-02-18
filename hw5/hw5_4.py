import numpy as np
import matplotlib.pyplot as plt

P = np.array([[0, 1, 0, 0], [1 / 3, 0, 2 / 3, 0], [0, 0, 0, 1], [1 / 2, 0, 1 / 2, 0]])

p01 = np.array([1 / 4] * 4)  # Convergence behavior/value

p02 = np.array([1, 0, 0, 0])  # Show periodic and determine period

n = 1000  # Number of steps
p = np.zeros((n + 1, 4))
p[0, :] = p01
x = p01
for i in range(n):
    pn = x @ P
    p[i + 1, :] = pn

# Plotting
plt.plot(p)
plt.ylim([0, 1])
plt.xlabel("Trials")
plt.ylabel("Probability")
plt.legend(["P1", "P2", "P3", "P4"])
plt.title("Limiting behavior of MC")

plt.show()
