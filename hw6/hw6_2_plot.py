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
kde_h = scipy.stats.gaussian_kde(q_hist[1, :])

# Plotting
fig, ax = plt.subplots()
ax.plot(q_hist[0, :], q_hist[1, :], color="b", marker="x")
ax.set_xlabel(r"Phi $\frac{W}{m^2}$")
ax.set_ylabel(r"h   $\frac{W}{m^2C}$")
ax.set_title("MCMC 1000 Steps")
ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

kdephix = np.linspace(-2e5, -1.3e5, 1000)
kdehx = np.linspace(16, 20, 1000)

fig, axes = plt.subplots(2, 1)
axes[0].plot(kdephix, kde_phi(kdephix), label="KDE")
axes[0].hist(q_hist[0, :], density=True, bins=25, label="Histogram")
axes[0].set_xlabel(r"Phi $\frac{W}{m^2}$")
axes[0].ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
axes[0].legend()

axes[1].plot(kdehx, kde_h(kdehx), label="KDE")
axes[1].hist(q_hist[1, :], density=True, bins=25, label="Histogram")
axes[1].set_xlabel(r"h   $\frac{W}{m^2C}$")
axes[1].legend()
fig.suptitle("Estimated marginal densities")

plt.show()
