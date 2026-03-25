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
ax.set_xlabel("Points")
ax.set_ylabel("R")


# Solving for the moments of u
def mean_fun(q):
    return np.exp(-q)


def var_fun(q):
    return mean_fun(2 * q) - mean_fun(q) ** 2


points_phys, weights_phys = np.polynomial.hermite.hermgauss(10)

points = np.sqrt(2) * points_phys
weights = 1 / np.sqrt(np.pi) * weights_phys
mean = weights @ mean_fun(points).T
Eu2 = weights @ mean_fun(2 * points).T
var = Eu2 - mean**2
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
    Eu2 = weights @ mean_fun(2 * points).T
    var = Eu2 - mean**2
    mean_err = abs(mean - mean_true)
    var_err = abs(var - var_true)
    errs[R - 1, :] = (mean_err, var_err)

print(errs)  # Why doesn't the estimate of the variance change? At all.

fig, ax = plt.subplots()
ax.plot(range(1, 11), errs[:, 0])
ax.set_yscale("log")
ax.set_ylabel("Error")
ax.set_xlabel("R")
fig, ax = plt.subplots()
ax.plot(range(1, 11), errs[:, 1])
ax.set_yscale("log")
ax.set_ylabel("Error")
ax.set_xlabel("R")
plt.show()
