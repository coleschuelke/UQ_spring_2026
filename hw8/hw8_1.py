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
    EY = f(q_bar)
    varY = grad_f(q_bar) * s2 * grad_f(q_bar)
    print(f"Variance of X: {s2}")
    print(f"Perturbation E[Y] = {EY}")
    print(f"perturbation Var(Y) = {varY}")

    X = np.random.normal(q_bar, np.sqrt(s2), 10**7)
    reals = f(X)
    mean = np.mean(reals)
    var = np.var(reals)

    print(f"True E[Y] = {mean}")
    print(f"True Var(Y) = {var}\n")

    ## Plot
    # kdex = np.linspace(min(reals), max(reals), 1000)
    # kde = scipy.stats.gaussian_kde(reals)
    # fig, ax = plt.subplots()
    # ax.hist(reals, bins=500, density=True)
    # ax.plot(kdex, kde(kdex))

plt.show()
