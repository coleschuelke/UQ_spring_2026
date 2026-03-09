import numpy as np
from scipy import stats


def draw_lhs(N):

    # Get the bins
    dp = 1 / N
    probs = [n / N for n in range(N + 1)]  # Includes 0, 1
    # phi_divs = stats.Normal.icdf(probs)  # TODO: Both of these are wrong
    # h_divs = 0

    # Random on the y axis
    ru = np.random.uniform(0, 1, 2 * N)

    phi_bins = []
    h_bins = []

    # Draw from the bins
    for i in range(N):
        p_phi = probs[i] + ru[i] * dp
        p_h = probs[i] + ru[N + i] * dp
        # Phi
        phi[i] = 0  # TODO: Inverse draw
        h[i] = 0  # TODO: Inverse draw

        # h
    phi = []
    h = []

    # Recombine to realizations
    l = np.array(range(N)).T
    rng = np.random.default_rng()
    r = rng.permutation(N)
    perms = np.column_stack((l, r.T))

    samples = np.zeros(N, 2)
    for i in range(N):
        samples[i, :] = (phi[perms[i, 0]], h(perms[i, 1]))

    return samples


draw_lhs(5)
# Brute force monte-carlo

# MC vs LHS head-to-head 100 samples


#### Plotting ####
# 2D plots

# Histogram H2H samples
