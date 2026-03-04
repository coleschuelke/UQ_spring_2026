import numpy as np
from scipy import io

from utils.heat_equation import HeatEquation
from utils.markov_chain_monte_carlo import MCMC, plot_mcmc_2d


# Load the data
data = io.loadmat("HW06_Problem1.mat")
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


##### END OF HELPER FUNCTIONS #####

# Results with DR
mcmc = MCMC(
    q0=q0, J_func=prop_rand, r_calc=ratio, post=posterior, s=sigma02, D=D, M=1_000
)

if run:
    dr_results = mcmc.metropolis_hastings(
        adaptive=False,
        gibbs_step=False,
        delayed_rejection=True,
        gamma2=0.2,
        save_output=True,
        filename="hw7_1_results.npz",
    )

plot_mcmc_2d("hw7_1_results.npz")
