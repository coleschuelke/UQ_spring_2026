import matplotlib.pyplot as plt
import numpy as np

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

pper = np.array([1 / 2, 0, 0, 1 / 2, 0])  # show period k=3
# pper = np.array([1, 0, 0, 0, 0])

n = 20  # Number of steps
# Stationary distribution
pstat_hist = np.zeros((n + 1, 5))
pstat_hist[0, :] = pstat
x = pstat
for i in range(n):
    pn = x @ P
    pstat_hist[i + 1, :] = pn
    x = pn

# Perodic distribution
pper_hist = np.zeros((n + 1, 5))
pper_hist[0, :] = pper
x = pper
for i in range(n):
    pn = x @ P
    pper_hist[i + 1, :] = pn
    x = pn

# Plotting
fig, ax = plt.subplots()
ax.plot(pstat_hist)
ax.set_ylim([0, 1])
ax.set_xlabel("Trials")
ax.set_ylabel("Probability")
ax.legend(["P1", "P2", "P3", "P4", "P5"])
ax.set_title("Stationary Case")

fig, ax = plt.subplots()
ax.plot(pper_hist)
ax.set_ylim([0, 1])
ax.set_xlabel("Trials")
ax.set_ylabel("Probability")
ax.legend(["P1", "P2", "P3", "P4", "P5"])
ax.set_title("Periodic Case")

plt.show()
