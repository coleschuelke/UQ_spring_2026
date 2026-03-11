import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from utils.heat_equation import HeatEquation

eq = HeatEquation(k=2.3e2, Tamb=21.29)


# Helper functions
def Ts(x, q):
    q_dict = {
        "phi": q[0],
        "h": q[1],
    }

    return eq.Ts_vals(x, q_dict)


def draw_lhs(N):

    # Get the bins
    dp = 1 / N
    probs = [n / N for n in range(N + 1)]  # Includes 0, 1

    # Random on the y axis
    ru = np.random.uniform(0, 1, 2 * N)

    phi = np.zeros((N,))
    h = np.zeros((N,))

    # Draw from the bins
    for i in range(N):
        p_phi = probs[i] + ru[i] * dp
        p_h = probs[i] + ru[N + i] * dp
        phi[i] = stats.norm.ppf(p_phi, loc=-15e4, scale=np.sqrt(10e8))
        h[i] = 5 + p_h * 10

    # Recombine to realizations
    rng = np.random.default_rng()
    rng.shuffle(phi)
    rng.shuffle(h)

    return np.column_stack((phi, h))


# Brute force monte-carlo
big = 10  # _000_000
phi_big = np.random.normal(-15e4, np.sqrt(10e8), big)
h_big = np.random.uniform(5, 15, big)
q_big = np.column_stack((phi_big, h_big))
temp_big = Ts(0.38, q_big.T)
print(q_big.shape)
mean = np.mean(temp_big)
var = np.var(temp_big)
# Quantify the uncertainty on the estimates
# print(f"The mean of the brute force method is: {mean}")
# print(f"The variance of the brute force method is: {var}")
print(f"The mean and variance using brute force: E[Y] = {mean}, Var(Y) = {var}")
# Converges at a rate of 1/sqrt(N)
print(f"The variance in the estimate of the mean is {np.var(temp_big)/np.sqrt(big)}")
print(
    f"The variance in the estimate of the variance is {(stats.moment(temp_big, order=4) - np.var(temp_big)**2)/big}"
)

# MC vs LHS head-to-head 100 samples
n_samp = 5
phi_samp = np.random.normal(-15e4, np.sqrt(10e8), n_samp)
h_samp = np.random.uniform(5, 15, n_samp)
mc_samples = np.column_stack((phi_samp, h_samp))
lhs_samples = draw_lhs(n_samp)

# Push the samples through the function
lhs_temps = Ts(0.38, lhs_samples.T)
mc_temps = Ts(0.38, mc_samples.T)

lhs_mean = np.mean(lhs_temps)
lhs_var = np.var(lhs_temps)
mc_mean = np.mean(mc_temps)
mc_var = np.var(mc_temps)
print(f"The mean and variance using LHS: E[Y] = {lhs_mean}, Var(Y) = {lhs_var}")
print(f"The mean and variance using MC: E[Y] = {mc_mean}, Var(Y) = {mc_var}")


#### Plotting ####
# 2D plots
fig, ax = plt.subplots()
ax.scatter(lhs_samples[:, 0], lhs_samples[:, 1])
ax.set_ylim([5, 15])
ax.set_xlim([-210000, -65000])
ax.set_xlabel(r"$\Phi$")
ax.set_ylabel(r"h")
ax.set_title("Latin Hypercube Sampling")
fig, ax = plt.subplots()
ax.scatter(mc_samples[:, 0], mc_samples[:, 1])
ax.set_ylim([5, 15])
ax.set_xlim([-210000, -65000])
ax.set_xlabel(r"$\Phi$")
ax.set_ylabel(r"h")
ax.set_title("Monte-Carlo")

# Histograms
fig, axes = plt.subplots(2, 1)
axes[0].hist(lhs_samples[:, 0], density=True, bins=20)
axes[0].set_xlabel(r"$\Phi$")
axes[1].hist(lhs_samples[:, 1], density=True, bins=20)
axes[1].set_xlabel("h")
fig.suptitle("LHS Histograms")
plt.tight_layout()

fig, axes = plt.subplots(2, 1)
axes[0].hist(mc_samples[:, 0], density=True, bins=20)
axes[0].set_xlabel(r"$\Phi$")
axes[1].hist(mc_samples[:, 1], density=True, bins=20)
axes[1].set_xlabel("h")
fig.suptitle("Monte-Carlo Histograms")

plt.tight_layout()
# plt.show()
