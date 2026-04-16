from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from aux import HestonModel, generate_multi_indices, build_Psi_uni
from scipy.stats.qmc import LatinHypercube, scale

from utils import ASE379L_UQ_PCE as pce
from utils import ASE379L_UQ_Polynomials as poly

# Control
Nsamps = 500  # Number of function evaluations to fit the PCE
d = 5  # Order of the multivariate PCE (Should be able to justify this) (Start small for now)
p = 4  # Dimension of the input

run_full = True

S0 = 100  # Initial price of the asset
T = 1  # Time at which the asset price is being predicted
V0 = 0.04
r = 0.03
K = S0 * np.exp(r * T)

# Create the model
model = HestonModel(S0, T, K, V0, r)

example = model.price(2, 0.04, 0.3, -0.7)  # Verify that the model is working properly
print(example)

if run_full:
    # Set up the input variables
    lower_bounds = [0.5, 0.03, 0.1, -0.9]
    upper_bounds = [4, 0.05, 0.6, -0.2]

    # -------- Least squares PCE generation --------

    # Sample the input variables using LHS and scale
    sampler = LatinHypercube(d=p)
    raw_samps = sampler.random(n=Nsamps)  # Draw the uniforms samples [0, 1)
    poly_samps = (
        raw_samps * 2 - 1
    )  # Map the uniform to proper range for standard polynomials
    scaled_samps = scale(raw_samps, lower_bounds, upper_bounds)

    # Evaluate the Heston model using the closed-form solution
    prices = np.zeros(Nsamps)
    for ii in range(Nsamps):
        prices[ii] = model.price(*(scaled_samps[ii]))

    # Create multi-indices
    K_idx = generate_multi_indices(4, d)
    magK = len(K_idx)

    # Build the Psi matrix for the LS problem
    Psi_ls = build_Psi_uni(poly_samps, K_idx, d)

    # Solve the LS problem
    u_k_ls, _, _, _ = np.linalg.lstsq(Psi_ls, prices, rcond=None)

    print(f"The LS mean is {u_k_ls[0]}")

    # -------- Cubature PCE generation --------
    # Extract the Legendre polynomial points
    points_std, weights_std = np.polynomial.legendre.leggauss(d + 1)
    weights_std /= 2
    polys = poly.legendrePoly(points_std, d)
    grid_points_std = np.array(list(product(points_std, repeat=p)))
    grid_weights_std = np.array([np.prod(w) for w in product(weights_std, repeat=p)])

    # Build the Psi matrix (normalized)
    Psi_cub = build_Psi_uni(grid_points_std, K_idx, d)

    # Map the grid into the scaled space
    grid_points_scaled = np.zeros_like(grid_points_std)
    for i in range(p):
        grid_points_scaled[:, i] = grid_points_scaled[:, i] = (
            (grid_points_std[:, i] + 1) / 2
        ) * (upper_bounds[i] - lower_bounds[i]) + lower_bounds[i]

    price_grid = np.array([model.price(*pt) for pt in grid_points_scaled])
    weighted_prices = price_grid * grid_weights_std

    gamma_uni = 1.0 / (2 * np.arange(d + 1) + 1)  # WARN: need to check on this
    gamma_multi = np.ones(magK)
    for k in range(magK):
        for dim in range(p):
            gamma_multi[k] *= gamma_uni[K_idx[k, dim]]

    u_k_cub = (1.0 / gamma_multi) * (Psi_cub.T @ weighted_prices)

    print(f"The cubature mean is {u_k_cub[0]}")
    print(f"The MC mean is {np.mean(prices)}")

    # Sobol indices
    print("-------- Least Squares --------")
    # Total variance (sum of weighted squares for all non-mean terms)
    ls_term_variances = gamma_multi[1:] * (u_k_ls[1:] ** 2)
    ls_total_var = np.sum(ls_term_variances)

    ls_sobol_total = np.zeros(p)
    for i in range(p):
        # Mask to find every multi-index where dimension i is 'active' (degree > 0)
        mask = K_idx[1:, i] > 0
        ls_sobol_total[i] = np.sum(ls_term_variances[mask]) / ls_total_var

    # Print sensitivity results
    labels = ["kappa", "theta", "sig_v", "rho"]
    for label, s_val in zip(labels, ls_sobol_total):
        print(f"Total Sensitivity (S_T) for {label} (LS): {s_val:.4f}")

    print("-------- Cubature --------")
    cub_term_variances = gamma_multi[1:] * (u_k_cub[1:] ** 2)
    cub_total_var = np.sum(cub_term_variances)

    cub_sobol_total = np.zeros(p)
    for i in range(p):
        # Mask to find every multi-index where dimension i is 'active' (degree > 0)
        mask = K_idx[1:, i] > 0
        cub_sobol_total[i] = np.sum(cub_term_variances[mask]) / cub_total_var

    # Print sensitivity results
    labels = ["kappa", "theta", "sig_v", "rho"]
    for label, s_val in zip(labels, cub_sobol_total):
        print(f"Total Sensitivity (S_T) for {label} (cubature): {s_val:.4f}")

    # Compare with market data?
