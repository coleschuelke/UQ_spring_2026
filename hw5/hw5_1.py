import numpy as np
import matplotlib.pyplot as plt


P = np.array([[0.5, 0, 0, 0.5], [0.5, 0, 0, 0.5], [0.5, 0, 0, 0.5], [0.4, 0.4, 0.2, 0]])


X0 = np.array([0.25, 0.25, 0.25, 0.25])

p10 = X0 @ np.linalg.matrix_power(P, 10)

print(f"The initial distribution is {X0}")
print(f"The distribution after ten steps is {p10}")

n = 1000  # Number of steps
p = np.zeros((n + 1, 4))
p[0, :] = X0
x = X0
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
