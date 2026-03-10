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
    l = np.array(range(N)).T
    rng = np.random.default_rng()
    r = rng.permutation(N)
    perms = np.column_stack((l, r.T))

    samples = np.zeros((N, 2))
    for i in range(N):
        samples[i, :] = (phi[perms[i, 0]], h[perms[i, 1]])

    return samples


# Brute force monte-carlo
big = 100_000
phi_big = np.random.normal(-15e4, np.sqrt(10e8), big)
h_big = np.random.uniform(5, 15, big)
q_big = np.column_stack((phi_big, h_big))
temp_big = Ts(0.38, q_big.T)
print(q_big.shape)
mean = np.mean(temp_big)
var = np.var(temp_big)
# Quantify the uncertainty on the estimates
print(f"The mean of the brute force method is: {mean}")
print(f"The variance of the brute force method is: {var}")
# Converges at a rate of 1/sqrt(N)

# MC vs LHS head-to-head 100 samples
n_samp = 100
phi_samp = np.random.normal(-15e4, np.sqrt(10e8), n_samp)
h_samp = np.random.uniform(5, 15, n_samp)
mc_samples = np.column_stack((phi_samp, h_samp))
lhs_samples = draw_lhs(n_samp)


#### Plotting ####
# 2D plots
plt.scatter(lhs_samples[:, 0], lhs_samples[:, 1], label="LHS")
plt.scatter(mc_samples[:, 0], mc_samples[:, 1], label="MC")
plt.legend()

# Histogram H2H samples


plt.show()
