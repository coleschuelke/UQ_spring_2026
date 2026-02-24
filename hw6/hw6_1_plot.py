import numpy as np
import scipy
import matplotlib.pyplot as plt

r_1k = np.load("hw6_1_results_1k.npz")
r_100k = np.load("hw6_1_results_100k.npz")

q_hist_1k = r_1k["q_hist"]
qMAP_1k = q_hist_1k[:, np.argmax(r_1k["post_hist"])]

q_hist_100k = r_100k["q_hist"]
qMAP_100k = q_hist_100k[:, np.argmax(r_100k["post_hist"])]


print(f"The MAP estimate with M=1_000 run is {qMAP_1k}")
print(f"The acceptance ratio for the M=1_000 MCMC run was {r_1k["accr"]}")


print(f"The MAP estimate with M=100_000 run is {qMAP_100k}")
print(f"The acceptance ratio for the M=100_000 MCMC run was {r_100k["accr"]}")

# KDE of the data


# Plotting
fig, ax = plt.subplots()
ax.plot(q_hist_1k[0, :], q_hist_1k[1, :], color="b", marker="x")
ax.set_xlabel("Phi")
ax.set_ylabel("h")
ax.set_title("MCMC 1000 Steps")

# fig, ax = plt.subplots()
# ax.plot(q_hist_100k[0, :], q_hist_100k[1, :], color="b", marker="x")
# ax.set_xlabel("Phi")
# ax.set_ylabel("h")
# ax.set_title("MCMC 100_000 Steps")

# fig, axes = plt.subplots(2, 1)
# for k in range(2):
#     axes[k].hist(q_hist_100k[k, :], density=True, bins=25)

plt.show()
