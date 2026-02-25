import matplotlib.pyplot as plt
import numpy as np
import scipy

result_ng = np.load("hw6_3_results_ng.npz")
result_g = np.load("hw6_3_results_g")

q_hist_ng = result_ng["q_hist"]
q_accr_ng = result_ng["q_accr"]
post_hist_ng = result_ng["post_hist"]
s_hist_ng = result_ng["s_hist"]

q_hist_g = result_g["q_hist"]
q_accr_g = result_g["q_accr"]
post_hist_g = result_g["post_hist"]
s_hist_g = result_g["s_hist"]

print(f"The acceptance ratio for the run without gibbs was {result_ng["accr"]}")
print(
    f"The MAP estimate for the run without gibbs was {q_hist_ng[:, np.argmax(result_ng["post_hist"][:, 100:])]}"
)

print(f"The acceptance ratio for the run with gibbs was {result_g["accr"]}")
print(
    f"The MAP estimate for the run with gibbs was {q_hist_g[:, np.argmax(result_g["post_hist"][:, 100:])]}"
)


# Plotting
# Chain plots
fig, ax = plt.subplots()
ax.plot(q_hist_ng[0, :], q_hist_ng[1, :], color="b", marker="x")
ax.set_xlabel(r"Phi $\frac{W}{m^2}$")
ax.set_ylabel(r"h   $\frac{W}{m^2C}$")
ax.set_title("MCMC 1000 Steps Without Gibbs Step")
ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

fig, ax = plt.subplots()
ax.plot(q_hist_g[0, :], q_hist_g[1, :], color="b", marker="x")
ax.set_xlabel(r"Phi $\frac{W}{m^2}$")
ax.set_ylabel(r"h   $\frac{W}{m^2C}$")
ax.set_title("MCMC 1000 Steps With Gibbs Step")
ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

# Histograms
kde_phix_ng = np.linspace(-2e5, -1.3e5, 1000)
kde_hx_ng = np.linspace(16, 20, 1000)
kde_phi_ng = scipy.stats.gaussian_kde(q_hist_ng[0, :])
kde_h_ng = scipy.stats.gaussian_kde(q_hist_ng[1, :])

kde_phix_g = np.linspace(-2e5, -1.3e5, 1000)
kde_hx_g = np.linspace(16, 20, 1000)
kde_phi_g = scipy.stats.gaussian_kde(q_hist_g[0, :])
kde_h_g = scipy.stats.gaussian_kde(q_hist_g[1, :])


fig, axes = plt.subplots(2, 1)
axes[0].plot(kde_phix_ng, kde_phi_ng(kde_phix_ng), label="KDE")
axes[0].hist(q_hist_ng[0, :], density=True, bins=25, label="Histogram")
axes[0].set_xlabel(r"Phi $\frac{W}{m^2}$")
axes[0].ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
axes[0].legend()

axes[1].plot(kde_hx_ng, kde_h_ng(kde_hx_ng), label="KDE")
axes[1].hist(q_hist_ng[1, :], density=True, bins=25, label="Histogram")
axes[1].set_xlabel(r"h   $\frac{W}{m^2C}$")
axes[1].legend()
fig.suptitle("Estimated marginal densities without Gibbs Step")
