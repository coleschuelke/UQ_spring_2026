import matplotlib.pyplot as plt
import numpy as np
import scipy

result = np.load("hw6_2_results.npz")

q_hist = result["q_hist"]

print(f"The acceptance ratio for the run was {result["accr"]}")
print(f"The final variance was {result["V"]}")
print(f"The MAP estimate for the run was {q_hist[:, np.argmax(result["post_hist"])]}")

# KDE of the data
kde_phi = scipy.stats.gaussian_kde(q_hist[0, :])

# Plotting
fig, ax = plt.subplots()
ax.plot(q_hist[0, :], q_hist[1, :], color="b", marker="x")
ax.set_xlabel("Phi")
ax.set_ylabel("h")
ax.set_title("MCMC 1000 Steps")

kdephix = np.linspace(-2e5, -1.3e5, 1000)

fig, axes = plt.subplots(2, 1)
axes[0].plot(kdephix, kde_phi(kdephix))
for i in range(2):
    axes[i].hist(q_hist[i, :], density=True, bins=25)

plt.show()
