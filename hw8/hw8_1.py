import numpy as np
import matplotlib.pyplot as plt
import scipy


def f(X):
    return X**3


def grad_f(X):
    return 2 * X**2


q_bar = 1

s2s = [0.01, 0.1]

reals = np.zeros((2, 10**7))

for i, s2 in enumerate(s2s):
    print("Entering the loop")
    EY = f(q_bar)
    varY = grad_f(q_bar) * s2 * grad_f(q_bar)
    print(f"Variance of X: {s2}")
    print(f"E[Y] = {EY}")
    print(f"Var(Y) = {varY}\n")

    print("Drawing random numbers")
    reals[i, :] = np.random.normal(q_bar, np.sqrt(s2), 10**7)

    # Plot
    print("Plotting")
    kdex = np.linspace(min(reals[i, :]), max(reals[i, :]), 1000)
    kde = scipy.stats.gaussian_kde(reals[i, :])
    fig, ax = plt.subplots()
    ax.hist(reals[i, :], bins=500, density=True)
    ax.plot(kdex, kde(kdex))

plt.show()
