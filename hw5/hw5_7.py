import matplotlib.pyplot as plt
import numpy as np
import scipy

np.random.seed(57)

# Laplace parameters
mu = 0
b = 1
sigma02 = 2 * b**2


def laplace(q):
    return 1 / (2 * b) * np.exp(-abs(q - mu) / b)


# Set up MCMC
q0 = 0
sigmas = [sigma02, sigma02 / 100, sigma02 * 100]
M = 10_100
q_hist = []
qk = q0
for s in sigmas:
    inner_hist = []
    inner_hist.append(qk)
    accepted = 0
    for i in range(M):
        q_star = np.random.normal(qk, np.sqrt(s))

        r = laplace(q_star) / laplace(qk)

        if np.random.uniform(0, 1) < min(1, r):
            qk = q_star
            inner_hist.append(qk)
            accepted += 1
        else:
            inner_hist.append(qk)
    q_hist.append(inner_hist)
    print(f"Acceptance Ratio for sigma = {s} is {accepted/M}")

q_hist = np.array(q_hist)

# Plotting
x = np.linspace(-10, 10, 500)
for j in range(q_hist.shape[0]):
    fig, ax = plt.subplots()
    ax.hist(q_hist[j, 101:], density=True, bins=40)
    ax.set_title(f"Sigma = {sigmas[j]}")
    ax.plot(x, laplace(x))

plt.show()
