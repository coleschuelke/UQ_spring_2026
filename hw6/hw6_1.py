import numpy as np
from utils.markov_chain_monte_carlo import MCMC
from utils.heat_equation import HeatEquation
from scipy import io
import matplotlib.pyplot as plt

# Load the data
data = io.loadmat("HW06_Problem1.mat")
ups = data["observations"][0]
x0 = 0.1  # m
dx = 0.04  # m
x = np.array([x0 + j * dx for j in range(15)])  # X values of the data

sigma02 = 2**2  # C^2

D = np.diag([1e3, 2e-5])  # Converted to units in meters

q0 = np.array([-150000, 20]).T

eq = HeatEquation(k=230, Tamb=21.29)


# Define some helper functions
def Ts_q(q):
    q_dict = {
        "phi": q[0],
        "h": q[1],
    }

    return eq.Ts_vals(x, q_dict)


def pi(q):
    c = (1 / (np.sqrt(2 * np.pi * sigma02))) ** (len(x))
    arg = -1 / (2 * sigma02) * (ups - Ts_q(q)).T @ (ups - Ts_q(q))
    p = c * np.exp(arg)
    if p < 1e-15:
        print("WARNING, LIKELIHOOD NEAR ZERO")
    return p


def pi0(q):
    phi = q[0]
    h = q[1]
    mu_phi = -150000
    mu_h = 10
    sigma_phi = 1e9
    sigma_h = 5e6
    nphi = (
        1
        / (np.sqrt(2 * np.pi * sigma_phi))
        * np.exp(-1 / (2 * sigma_phi**2) * (phi - mu_phi) ** 2)
    )
    nh = (
        1
        / (np.sqrt(2 * np.pi * sigma_h))
        * np.exp(-1 / (2 * sigma_h**2) * (h - mu_h) ** 2)
    )

    p = nphi * nh
    if p < 1e-15:
        print("WARNING, PRIOR NEAR ZERO")

    return p


def prop_dist(q, qmu):
    p = (
        1
        / (np.sqrt((2 * np.pi) ** (len(np.diag(D)))) * np.linalg.det(D))
        * np.exp(-1 / 2 * (q - qmu).T @ np.linalg.solve(D, (q - qmu)))
    )
    if p < 1e-15:
        print("WARNING, PROPOSAL DISTRIBIUTION NEAR ZERO")
    return p


def prop_rand(q, D):
    return np.random.multivariate_normal(q, np.sqrt(D))


def ratio(q_star, qk):
    num = pi(q_star) * pi0(q_star) * prop_dist(qk, q_star)
    denom = (
        pi(qk) * pi0(qk) * prop_dist(q_star, qk)
    )  # Sometimes getting a zero, I'm assuming its to do with units
    return num / denom


# Set up the actual problem
M = 1_000
mcmc = MCMC(q0, prop_rand, ratio, M, 2012)

r_1k = mcmc.metropolis_hastings()
q_hist_1k = r_1k[0]
qMAP_1k = 0  # TODO

r_100k = mcmc.metropolis_hastings(q0, prop_rand, ratio, 100000, 2012)
q_hist_100k = r_100k[0]
qMAP_100k = 0  # TODO

print(f"The MAP estimate with M=1_000 run is {qMAP_1k}")
print(f"The acceptance ratio for the M=1_000 MCMC run was {r_1k[1]}")
print(f"The MAP estimate with M=100_000 run is {qMAP_100k}")
print(f"The acceptance ratio for the M=100_000 MCMC run was {r_100k[1]}")

# KDE of the

# Plotting
fig, ax = plt.subplots()
ax.plot(q_hist_1k[0, :], q_hist_1k[1, :], color="b", marker="x")
ax.set_xlabel("Phi")
ax.set_ylabel("h")
ax.set_title("MCMC 1000 Steps")

fig, ax = plt.subplots()
ax.plot(q_hist_100k[0, :], q_hist_100k[1, :], color="b", marker="x")
ax.set_xlabel("Phi")
ax.set_ylabel("h")
ax.set_title("MCMC 100_000 Steps")

fig, axes = plt.subplots(2, 1)
for k in range(2):
    axes[k].hist(q_hist_100k[k, :], density=True)
    axes[k].set_title("Phi Density")
