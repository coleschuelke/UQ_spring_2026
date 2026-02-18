import matplotlib.pyplot as plt
import numpy as np

P = np.array([[0, 1, 0, 0], [1 / 3, 0, 2 / 3, 0], [0, 0, 0, 1], [1 / 2, 0, 1 / 2, 0]])

p01 = np.array([1 / 4] * 4)  # Convergence behavior/value

p02 = np.array([1, 0, 0, 0])  # Show periodic and determine period

n = 20  # Number of steps
# stationary
phist1 = np.zeros((n + 1, 4))
phist1[0, :] = p01
x = p01
for i in range(n):
    pn = x @ P
    phist1[i + 1, :] = pn
    x = pn

# periodic
phist2 = np.zeros((n + 1, 4))
phist2[0, :] = p02
x = p02
for i in range(n):
    pn = x @ P
    phist2[i + 1, :] = pn
    x = pn

print(f"The limiting probability is {phist1[-1, :]}")

# Plotting
fig, ax = plt.subplots()
ax.plot(phist1)
ax.set_ylim([0, 1])
ax.set_xlabel("Trials")
ax.set_ylabel("Probability")
ax.legend(["P1", "P2", "P3", "P4"])
ax.set_title("Limiting behavior of MC")

fig, ax = plt.subplots()
ax.plot(phist2)
ax.set_ylim([0, 1])
ax.set_xlabel("Trials")
ax.set_ylabel("Probability")
ax.legend(["P1", "P2", "P3", "P4"])
ax.set_title("Limiting behavior of MC")

plt.show()
