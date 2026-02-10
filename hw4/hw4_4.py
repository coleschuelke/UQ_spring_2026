import numpy as np
import matplotlib.pyplot as plt
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

    return eq.Ts_numder(xj, ["phi", "h"], q_dict, delta)


# Compute errors for all deltas
deltas = [10**-x for x in range(15, 0, -1)]
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
ax.legend(legend_strings, loc="right")
plt.show()
