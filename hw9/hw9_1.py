import matplotlib.pyplot as plt
import numpy as np

# Weights for different values of R
fig, ax = plt.subplots()
for R in range(1, 6):
    points_phys, weights_phys = np.polynomial.hermite.hermgauss(R)
    points = np.sqrt(2) * points_phys
    weights = 1 / np.sqrt(np.pi) * weights_phys
    ax.scatter(points, R * np.ones(len(points)))
    if R == 4:
        print(f"Points for R=4: {points}")
        print(f"Weights for R=4: {weights}")
ax.set_xlabel("Point")
ax.set_ylabel("R")


# Solving for the moments of u
def mean_fun(q):
    return np.exp(
        -q
    )  # * np.exp(-0.5 * q**2) / np.sqrt(2 * np.pi)  # Maybe don't need the dist here based on the equation for PCE for 2.6


# Why do we not include the distribution? Isn't that the whole point of what we are trying integrate?????


def var_fun(q):
    return mean_fun(2 * q) - mean_fun(q) ** 2


points_phys, weights_phys = np.polynomial.hermite.hermgauss(10)

points = np.sqrt(2) * points_phys
weights = 1 / np.sqrt(np.pi) * weights_phys
mean = weights @ mean_fun(points).T
var = weights @ var_fun(points).T
print(mean)  # Something is going quite wrong with the calculations here
print(var)
mean_true = np.exp(0.5)
var_true = np.exp(2) - np.exp(1)
errs = np.zeros((10, 2))
for R in range(1, 11):
    points_phys, weights_phys = np.polynomial.hermite.hermgauss(R)

    points = np.sqrt(2) * points_phys
    weights = 1 / np.sqrt(np.pi) * weights_phys
    mean = weights @ mean_fun(points).T
    var = weights @ var_fun(points).T
    mean_err = abs(mean - mean_true)
    var_err = abs(var - var_true)
    errs[R - 1, :] = (mean_err, var_err)

print(errs)  # Why doesn't the estimate of the variance change? At all.

fig, axes = plt.subplots(2, 1)
axes[0].plot(errs[:, 0])
axes[0].set_yscale("log")
axes[1].plot(errs[:, 1])
axes[1].set_yscale("log")
plt.show()
