import numpy as np
import scipy
import matplotlib.pyplot as plt

r_1k = np.load("hw6_1_results_1k.npz")
r_100k = np.load("hw6_1_results_100k.npz")

q_hist_1k = r_1k["q_hist"]
qMAP_1k = q_hist_1k[:, np.argmax(r_1k["post_hist"])]

q_hist_100k = r_100k["q_hist"]
q_hist_100k = q_hist_100k[:, 100:]  # Remove the burn-in for plotting later
qMAP_100k = q_hist_100k[:, np.argmax(r_100k["post_hist"])]


print(f"The MAP estimate with M=1_000 run is {qMAP_1k}")
print(f"The acceptance ratio for the M=1_000 MCMC run was {r_1k["accr"]}")


print(
    f"The MAP estimate with M=100_000 run is {qMAP_100k}"
)  # Phi doesn't make any sense
print(f"The acceptance ratio for the M=100_000 MCMC run was {r_100k["accr"]}")

# KDE of the data
xphi = np.linspace(-220000, -140000, 1000)
xh = np.linspace(15, 23, 1000)
kde_phi_100k = scipy.stats.gaussian_kde(q_hist_100k[0, :])
kde_h_100k = scipy.stats.gaussian_kde(q_hist_100k[1, :])
kde_100k_list = [kde_phi_100k, kde_h_100k]

# Plotting
fig, ax = plt.subplots()  # 1000 step 2d
ax.plot(q_hist_1k[0, :], q_hist_1k[1, :], color="b", marker="x")
ax.set_xlabel("Phi")
ax.set_ylabel("h")
ax.set_title("MCMC 1_000 Steps")

fig, ax = plt.subplots()  # 100_000 step 2d
ax.plot(q_hist_100k[0, :], q_hist_100k[1, :], color="b", marker="x")
ax.set_xlabel("Phi")
ax.set_ylabel("h")
ax.set_title("MCMC 100_000 Steps")
ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

fig, axes = plt.subplots(2, 1)  # 100_000 step marginals with KDE

axes[0].hist(q_hist_100k[0, :], density=True, bins=25, label="Histogram")
axes[0].plot(xphi, kde_phi_100k(xphi), label="KDE")
axes[0].set_xlabel(r"Phi $\frac{W}{m^2}$")
axes[0].ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
axes[0].legend()

axes[1].hist(q_hist_100k[1, :], density=True, bins=25, label="Histogram")
axes[1].plot(xh, kde_h_100k(xh), label="KDE")
axes[1].set_xlabel(r"h   $\frac{W}{m^2C}$")
axes[1].legend()
fig.suptitle("Estimated marginal densities")

plt.show()
