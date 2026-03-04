import matplotlib.pyplot as plt
import numpy as np
from scipy import io
from statsmodels.graphics.tsaplots import plot_acf

from utils.heat_equation import HeatEquation
from utils.markov_chain_monte_carlo import MCMC, plot_mcmc_2d

# Load the data
data = io.loadmat("HW07_Problem3.mat")

phi = []
h = []
s = []

phi_samples = data["samples"][0]
h_samples = data["samples"][1]
s_samples = data["samples"][2]

#### Subsampling ####
l_vals = [1, 10, 20, 100]

for l in l_vals:
    phi_temp = phi_samples[3000::l]
    h_temp = h_samples[3000::l]
    s_temp = s_samples[3000::l]

    phi.append(phi_temp)
    h.append(h_temp)
    s.append(s_temp)

#### Plotting ####
# Trace plots
fig, axes = plt.subplots(3, 1, constrained_layout=True)
axes[0].plot(phi[0])
axes[0].set_xlabel(r"$\Phi$")
axes[0].set_xscale("log")
axes[1].plot(h[0])
axes[1].set_xlabel(r"$h$")
axes[1].set_xscale("log")
axes[2].plot(s[0])
axes[2].set_xlabel(r"$\sigma_0^2$")
axes[2].set_xscale("log")

fig.suptitle("Trace Plots")

# Correleograms
for i, l in enumerate(l_vals):
    fig, axes = plt.subplots(3, 1, constrained_layout=True)
    plot_acf(
        phi[i],
        ax=axes[0],
        alpha=None,
        lags=range(150),
        use_vlines=False,
        marker=None,
        linestyle="-",
        title=None,
    )
    axes[0].hlines(0, 0, 150, color="k", linestyle=":")
    axes[0].set_xlabel(r"$\Phi$")
    plot_acf(
        h[i],
        ax=axes[1],
        alpha=None,
        lags=range(150),
        use_vlines=False,
        marker=None,
        linestyle="-",
        title=None,
    )
    axes[1].hlines(0, 0, 150, color="k", linestyle=":")
    axes[1].set_xlabel(r"$h$")
    plot_acf(
        s[i],
        ax=axes[2],
        alpha=None,
        lags=range(150),
        use_vlines=False,
        marker=None,
        linestyle="-",
        title=None,
    )
    axes[2].hlines(0, 0, 150, color="k", linestyle=":")
    axes[2].set_xlabel(r"$\sigma_0^2$")
    fig.suptitle(f"Autocorrelation, l={l}")

plt.show()
