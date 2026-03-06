import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import io

from utils.heat_equation import HeatEquation
from utils.markov_chain_monte_carlo import MCMC, plot_mcmc_2d

# Load the data
data = io.loadmat("HW06_Problem3.mat")
ups = data["observations"][0]
x0 = 0.1  # m
dx = 0.04  # m
x = np.array([x0 + j * dx for j in range(15)])  # X values of the data

# Setup
sigma02 = 2**2  # C^2
D = np.diag([1e6, 2e-2])  # Converted to units in meters
q0 = np.array([-150000, 20]).T
eq = HeatEquation(k=230, Tamb=21.29)

# Control
run = False


##### ALL HELPER FUNCTIONS #####
def Ts_q(q):
    q_dict = {
        "phi": q[0],
        "h": q[1],
    }

    return eq.Ts_vals(x, q_dict)


def pi(q, s):
    c = (1 / (np.sqrt(2 * np.pi * s))) ** (len(x))
    arg = -1 / (2 * s) * (ups - Ts_q(q)).T @ (ups - Ts_q(q))
    p = c * np.exp(arg)
    return p


def pi0(q):
    phi = q[0]
    h = q[1]
    mu_phi = -150000
    mu_h = 10
    sigma_phi = 1e9
    sigma_h = 5e5

    cphi = 1 / (np.sqrt(2 * np.pi * sigma_phi))
    argphi = -1 / (2 * sigma_phi**2) * (phi - mu_phi) ** 2

    ch = 1 / (np.sqrt(2 * np.pi * sigma_h))
    argh = -1 / (2 * sigma_h**2) * (h - mu_h) ** 2

    p = cphi * ch * np.exp(argphi + argh)
    return p


def prop_dist(q, qmu, V):
    p = (
        1
        / (np.sqrt((2 * np.pi) ** (len(np.diag(V))) * np.linalg.det(V)))
        * np.exp(-1 / 2 * (q - qmu).T @ np.linalg.solve(V, (q - qmu)))
    )
    return p


def prop_rand(q, V):
    if len(q >= 2):
        return np.random.multivariate_normal(q, V)
    else:
        return np.random.normal(q, np.sqrt(V))


def ratio(q_star, qk, V, s):
    num = pi(q_star, s) * pi0(q_star) * prop_dist(qk, q_star, V)
    denom = pi(qk, s) * pi0(qk) * prop_dist(q_star, qk, V)
    return num / denom


def posterior(q, s):
    return pi(q, s) * pi0(q)


def cf_ls(q):
    residuals = ups - Ts_q(q)
    return residuals


def cf_gibbs(q):
    residuals = ups - Ts_q(q)
    return residuals.T @ residuals


def jac(q):
    q_dict = {
        "phi": q[0],
        "h": q[1],
    }

    return -1 * eq.Ts_jac(x, ["phi", "h"], q_dict)


##### END OF HELPER FUNCTIONS #####

#### Least Squares Fit ####
q_hat = scipy.optimize.least_squares(cf_ls, q0, jac)
q_ls = q_hat.x
print(f"The point estimate of q is: {q_ls}")

residuals = cf_ls(q_ls)
s2ls = residuals.T @ residuals / len(x)
Vls = s2ls * np.linalg.inv((jac(q_ls).T @ jac(q_ls)))
print(f"The estimated covariance is {Vls}")

# Actual Problem
mcmc = MCMC(
    q0=q_ls, J_func=prop_rand, r_calc=ratio, post=posterior, s=s2ls, D=D, M=1_000
)

if run:
    dram_results = mcmc.metropolis_hastings(
        adaptive=True,
        k0=100,
        sp=2.38**2 / 2,
        eps=0,
        V0=Vls,
        gibbs_step=True,
        ns=0.01,
        n_meas=len(x),
        cf=cf_gibbs,
        delayed_rejection=True,
        gamma2=0.2,
        save_output=True,
        filename="hw7_4_results.npz",
    )

# Intervals
results = np.load("hw7_4_results.npz")
alpha = 0.05

phi = results["q_hist"][0, 100:]
h = results["q_hist"][1, 100:]
s = results["s_hist"][100:]

phi_int = np.array([np.quantile(phi, alpha / 2), np.quantile(phi, 1 - alpha / 2)])
h_int = np.array([np.quantile(h, alpha / 2), np.quantile(h, 1 - alpha / 2)])
s_int = np.array([np.quantile(s, alpha / 2), np.quantile(s, 1 - alpha / 2)])

print(f"The approximated credible intervals for phi are {phi_int}")
print(f"The approximated credible intervals for h are {h_int}")
print(f"The approximated credible intervals for s are {s_int}")

# Sanity checks
maxs_idx = np.argmax(s)
s_here = s[maxs_idx]
q_here = results["q_hist"][:, maxs_idx]
post_here = posterior(q_here, s_here)
print(f"The posterior at the max value of s is {post_here}")
print(f"The max value of the posterior is {np.max(results["post_hist"])}")

# Plotting
plot_mcmc_2d("hw7_4_results.npz")
