from itertools import product
from math import factorial

import matplotlib.pyplot as plt
import numpy as np
from aux import HestonModel, generate_multi_indices
from scipy.stats.qmc import LatinHypercube, scale

from utils import ASE379L_UQ_PCE as pce
from utils import ASE379L_UQ_Polynomials as poly

# Control
Nsamps = 10  # Number of function evaluations to fit the PCE
d = 5  # Order of the multivariate PCE (Should be able to justify this) (Start small for now)
p = 4  # Dimension of the input

S0 = 100  # Initial price of the asset
T = 1  # Time at which the asset price is being predicted
K = 100
V0 = 0.04
r = 0.03

# Create the model
model = HestonModel(S0, T, K, V0, r)

something = model.price(2, 0.04, 0.3, -0.7)
print(something)

# Set up the input variables
lower_bounds = [0.5, 0.01, 0.1, -0.9]
upper_bounds = [4, 0.1, 0.6, -0.2]

# Sample the input variables using LHS and scale
sampler = LatinHypercube(d=4)
raw_samps = sampler.random(n=Nsamps)  # Draw the uniforms samples [0, 1)
scaled_samps = scale(
    raw_samps, lower_bounds, upper_bounds
)  # Map the uniform to proper ranges

# Evaluate the model using the closed-form solution to the Heston model
prices = np.zeros((Nsamps, 4))
for ii in range(Nsamps):
    prices[ii, :] = model.price(
        *(scaled_samps[ii, :])
    )  # WARN: this is not fully working

# Fit the PCE using least squares
# Create multi-indices
K_idx = generate_multi_indices(4, d)
magK = len(K_idx)
P = np.zeros((d + 1, p, Nsamps))
# WARN: Double check that these are supposed to be legendre
kappa_polys = poly.legendrePoly(scaled_samps[:, 0], d)
theta_polys = poly.legendrePoly(scaled_samps[:, 1], d)
sig_polys = poly.legendrePoly(scaled_samps[:, 2], d)
rho_polys = poly.legendrePoly(scaled_samps[:, 3], d)
P[:, 0, :] = kappa_polys
P[:, 1, :] = theta_polys
P[:, 2, :] = sig_polys
P[:, 3, :] = rho_polys

# This is close but I'm not sure it's perfect
Psi = np.ones((Nsamps, magK))
for ii in range(magK):
    for jj in range(p):
        Psi[:, ii] = Psi[:, ii] * P[K_idx(ii, jj) + 1, jj, :]


# Fit the PCE using cubature
# Generate the 1D rules
# Standardized weights
points_phys_std, weights_phys_std = np.polynomial.legendre.leggauss(d + 1)
points_std = np.sqrt(2) * points_phys_std
weights = 1 / np.sqrt(np.pi) * weights_phys_std
polys = poly.legendrePoly(points_std, d)
grid_points_std = np.array(list(product(points_std, repeat=p)))
grid_weights_std = np.array([np.prod(w) for w in product(weights_std, repeat=p)])
# Map the grid into the scaled space
grid_points_scaled = np.zeros_like(grid_points_std)
for i in range(p):
    grid_points_scaled[:, i] = (grid_points_std[:, i] + 1) * (
        upper_bounds[i] - lower_bounds[i]
    )

price_grid = np.array([model.price(*pt) for pt in grid_points_scaled])

# Calculate the cubature

# Calculate the Sobol indices

# Compare with market data?
