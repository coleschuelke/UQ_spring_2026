import numpy as np
import matplotlib.pyplot as plt

P = np.array(
    [
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0.5, 0, 0, 0.5, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0],
    ]
)

pstat = np.array([1 / 6, 1 / 6, 1 / 3, 1 / 6, 1 / 6])  # show stationary

# pper = np.array([1 / 2, 0, 0, 1 / 2, 0])  # show period k=3
pper = np.array([1, 0, 0, 0, 0])


# Perodic distribution
n = 10  # Number of steps
p = np.zeros((n + 1, 5))
p[0, :] = pper
x = pper
for i in range(n):
    pn = x @ P
    p[i + 1, :] = pn

# Plotting
plt.plot(p)
plt.ylim([0, 1])
plt.xlabel("Trials")
plt.ylabel("Probability")
plt.legend(["P1", "P2", "P3", "P4", "P5"])
plt.title("Limiting behavior of MC")

plt.show()
