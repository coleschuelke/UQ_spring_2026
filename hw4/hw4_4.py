import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from utils import HeatEquation

# Test points
x0 = 0.1  # m
dx = 0.04  # m
xj = np.array([x0 + (j - 1) * dx for j in range(1, 16)])

eq = HeatEquation()


# Wrap heat equation methods appropriately
def Ts_q(q):
    q_dict = {
        "phi": q[0],
        "h": q[1],
    }

    return eq.Ts_vals(xj, q_dict)


def jac(q):
    q_dict = {
        "phi": q[0],
        "h": q[1],
    }

    return -1 * eq.Ts_jac(xj, ["phi", "h"], q_dict)


# Approximating with finite difference
def num_jac(q, delta):
    q_dict = {
        "phi": q[0],
        "h": q[1],
    }

    return -1 * eq.Ts_numder(xj, ["phi", "h"], q_dict, delta)


# Compute errors for all deltas
deltas = [10**-x for x in range(11, -6, -1)]
phi_out = np.zeros((15, len(deltas)))
h_out = np.zeros((15, len(deltas)))

q_star = [-100000, 10]

for i, d in enumerate(deltas):
    numerical = num_jac(q_star, d)
    analytical = jac(q_star)

    a_phi = analytical[:, 0]
    n_phi = numerical[:, 0]
    a_h = analytical[:, 1]
    n_h = numerical[:, 1]

    err_phi = abs((n_phi - a_phi) / a_phi)
    err_h = abs((n_h - a_h) / a_h)

    phi_out[:, i] = err_phi  # must be transposed prior to plotting
    h_out[:, i] = err_h

# Rework of number 3 with numeric derivatives
# Import the data
data = scipy.io.loadmat("exercise_7p7_data")
data = data["copper_data"]
df = pd.DataFrame(data)
ups = np.array(df.iloc[:, 1])


def cf(q):
    return ups - Ts_q(q)


def num_jac_opt(q):
    return num_jac(q, 1e2)


# Estimate, CI for phi, h

# RMS of residuals

# Obersvation errors and sampling distribution

# Residual and model fit plots

# Initial guess
x0 = [-100000, 10]

q_hat = scipy.optimize.least_squares(cf, x0, num_jac_opt)
print(f"The point estimate of q is: {q_hat.x}")

# Need 95% CI
n, p = (15, 2)
residuals = cf(q_hat.x)
sum_sq = residuals.T @ residuals
chi = jac(np.array(q_hat.x))
delta = np.diag(scipy.linalg.inv(chi.T @ chi))  # CHECK THAT THIS IS CORRECT DEFINITION
sigma2 = (1 / (n - p)) * sum_sq  # Estimated variance of the observations
print(f"The estimated std of the observations is: {np.sqrt(sigma2)}")

ci_lo, ci_hi = scipy.stats.t.interval(
    0.95, df=n - p, loc=q_hat.x, scale=np.sqrt(sigma2 * delta)
)

print(f"Confidence Intervals (95%): {ci_lo}, {ci_hi}")

# Residual RMS calculation
RMS = np.sqrt(sum_sq / len(residuals))
print(f"The RMS of the residuals is: {RMS}")

# Standard Deviation of Observations and sampling distribution
var_est = sigma2 * delta  # Estimated variance of the estimator
print(f"The estimated std of the estimator is: {np.sqrt(var_est)}")


# Plot of residuals and model fit
fig, ax = plt.subplots()
ax.scatter(range(1, len(residuals) + 1), residuals)
ax.hlines(0, -1, 15, "r", "--")
ax.set_title("Model Residuals")
ax.set_xlabel("Observation")
ax.set_ylabel("Residual (Deg C)")

fig, ax = plt.subplots()
ax.plot(xj, Ts_q(q_hat.x))
ax.scatter(xj, ups, marker="o", c="r")
ax.set_title("Model Fit")
ax.set_xlabel("x (m)")
ax.set_ylabel("T (Deg C)")

# Plotting
legend_strings = [f"x={x:.2}" for x in xj]

fig, ax = plt.subplots()
ax.plot(deltas, np.transpose(phi_out))
ax.set_yscale("log")
ax.set_xscale("log")
ax.legend(legend_strings, loc="right")

fig, ax = plt.subplots()
ax.plot(deltas, np.transpose(h_out))
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlabel(r"\delta")
ax.set_ylabel("Numerical Error")
ax.legend(legend_strings, loc="right")
plt.show()
