import scipy
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from utils import HeatEquation

# Import the data
data = scipy.io.loadmat("exercise_7p7_data")
data = data["copper_data"]
df = pd.DataFrame(data)
ups = np.array(df.iloc[:, 1])

# Test points
x0 = 0.1  # m
dx = 0.04  # m
xj = np.array([x0 + (j - 1) * dx for j in range(1, 16)])

eq = HeatEquation()  # Make sure to configure this to use the proper parameters

# Estimating q = [phi, h]


# Wrap heat equation methods appropriately
def Ts_q(q):
    q_dict = {
        "phi": q[0],
        "h": q[1],
    }

    return eq.Ts_vals(xj, q_dict)


def cf(q):

    return ups - Ts_q(q)


def jac(q):
    q_dict = {
        "phi": q[0],
        "h": q[1],
    }

    return -1 * eq.Ts_jac(xj, ["phi", "h"], q_dict)


# Initial guess
x0 = [-100000, 10]

q_hat = scipy.optimize.least_squares(cf, x0, jac)
print(f"The point estimate of q is: {q_hat.x}")

# Need 95% CI
n, p = (15, 2)
residuals = cf(q_hat.x)
sum_sq = residuals.T @ residuals
chi = jac(np.array(q_hat.x))
delta = np.diag(scipy.linalg.inv(chi.T @ chi))  # CHECK THAT THIS IS CORRECT DEFINITION
sigma2 = (1 / (n - p)) * sum_sq  # Estimated variance of the observations
print(f"The estimated variance of the observations is: {np.sqrt(sigma2)}")

ci_lo, ci_hi = scipy.stats.t.interval(
    0.95, df=n - p, loc=q_hat.x, scale=np.sqrt(sigma2 * delta)
)

print(f"Confidence Intervals (95%): {ci_lo}, {ci_hi}")

# Residual RMS calculation
RMS = np.sqrt(sum_sq / len(residuals))
print(f"The RMS of the residuals is: {RMS}")

# Standard Deviation of Observations and sampling distribution
var_est = sigma2 * delta  # Estimated variance of the estimator
print(f"The estimated variance of the estimator is: {np.sqrt(var_est)}")


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

plt.show()
