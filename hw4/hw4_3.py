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

# Residual RMS calculation
residuals = cf(q_hat.x)
RMS = np.sqrt(residuals.T @ residuals)
print(f"The RMS of the residuals is: {RMS}")

# Standard Deviation of Observations and sampling distribution


# Plot of residuals and model fit
fig, ax = plt.subplots()
ax.scatter(range(len(residuals)), residuals)
ax.hlines(0, -1, 15, "r", "--")

fig, ax = plt.subplots()
ax.plot(xj, Ts_q(q_hat.x))
ax.scatter(xj, ups, marker="o", c="r")


plt.show()
