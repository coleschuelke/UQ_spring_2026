import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

data_dict = loadmat("HW09_Problem3")
q_vals = data_dict["q_values"]  # 3xM
outputs = data_dict["outputs"]  # 1xM
M = 512

# Load datasets for problems 2 and 3
data_1 = loadmat("HW09_Problem3_Q1_Only")
data_3 = loadmat("HW09_Problem3_Q3_Only")
q_vals1 = data_1["q_values"]
y1 = data_1["outputs"]
q_vals3 = data_3["q_values"]
y3 = data_3["outputs"]


# Kernel
def R(qi, qj):
    theta = [5, 5, 5]
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
s2 = (1 / M) * (y - beta0 * one).T @ scriptR_inv @ (y - beta0 * one)
print(f"beta0 for this dataset is {beta0}")
print(f"s2 for this dataset is {s2}")


# The estimator
def fs(q):
    r = np.zeros((M, 1))
    for m in range(M):
        r[m] = R(q_vals[:, m], q)
    est = beta0 + r.T @ scriptR_inv @ (y - beta0 * one)
    var = s2 * (
        1
        - r.T @ scriptR_inv @ r
        + (r.T @ scriptR_inv @ one - 1) ** 2 / (one.T @ scriptR_inv @ one)
    )
    return np.array((est, var))


# Plotting
q2 = np.array([-np.pi / 4] * 1000)
q3 = np.array([np.pi / 4] * 1000)
q_lin = np.linspace(-np.pi, np.pi, 1000)
q_tot = np.array((q_lin, q2, q3))
estvec = []
varvec = []
for i in range(1000):
    est, var = fs(q_tot[:, i])
    estvec.append(est)
    varvec.append(var[0, 0])
fig, ax = plt.subplots()
ax.plot(q_lin, np.array(estvec).flatten())
ax.scatter(q_vals1[0, :], y1)
ax.plot(
    q_lin,
    np.array(estvec).flatten() + 3 * np.sqrt(np.array(varvec).flatten()),
    color="r",
)
ax.plot(
    q_lin,
    np.array(estvec).flatten() - 3 * np.sqrt(np.array(varvec).flatten()),
    color="r",
)
ax.set_xlabel("q1")
ax.set_ylabel("u(1, q)")

q1 = np.array([np.pi / 4] * 1000)
q2 = np.array([-np.pi / 4] * 1000)
q_lin = np.linspace(-np.pi, np.pi, 1000)
q_tot = np.array((q1, q2, q_lin))
estvec = []
varvec = []
for i in range(1000):
    est, var = fs(q_tot[:, i])
    estvec.append(est)
    varvec.append(var)
fig, ax = plt.subplots()
ax.plot(q_lin, np.array(estvec).flatten())
ax.scatter(q_vals3[2, :], y3)
ax.plot(
    q_lin,
    np.array(estvec).flatten() + 3 * np.sqrt(np.array(varvec).flatten()),
    color="r",
)
ax.plot(
    q_lin,
    np.array(estvec).flatten() - 3 * np.sqrt(np.array(varvec).flatten()),
    color="r",
)
ax.set_xlabel("q3")
ax.set_ylabel("u(1, q)")

plt.show()
