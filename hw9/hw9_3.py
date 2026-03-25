import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

M = 512
data_dict = loadmat("HW09_Problem3")
q_vals = data_dict["q_values"]  # 3xM
outputs = data_dict["outputs"]  # 1xM


# Kernel
def R(qi, qj):
    theta = [0.5, 0.3, 0.3]
    gamma = 1.5
    s = 0
    for k in range(3):
        s += theta[k] * abs(qi[k] - qj[k]) ** gamma  # Over all the input dimensions
    return np.exp(-s)


# Compute all relevant quantities for the data
one = np.ones((M, 1))
y = outputs.T

scriptR = np.zeros((M, M))
for m in range(M):
    for n in range(M):
        scriptR[m, n] = R(q_vals[:, m], q_vals[:, n])
scriptR_inv = np.linalg.inv(scriptR)

beta0 = np.linalg.inv(one.T @ scriptR_inv @ one) @ one.T @ scriptR_inv @ y
print(f"beta0 for this dataset is {beta0}")


# The estimator
def fs(q):
    for m in range(M):
        r = np.zeros((M, 1))
        r[m] = R(q_vals[:, m], q)

        est = beta0 + r.T @ scriptR_inv @ (y - beta0 * one)
        s2 = 1 / M * (y - beta0 * one).T @ scriptR_inv @ (y - beta0 * one)
        var = s2 * (
            1
            - r.T @ scriptR_inv @ r
            + (r.T @ scriptR_inv @ one - 1) ** 2 / (one.T @ scriptR_inv @ one)
        )
        return (est, s2, var)


print(fs(q_vals[:, 1]))
