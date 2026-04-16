from itertools import combinations

import numpy as np
from scipy.integrate import quad
from utils import ASE379L_UQ_Polynomials as poly
from utils import ASE379L_UQ_PCE as pce


class HestonModel:
    def __init__(self, S0, T, K, V0, r):
        self.S0 = S0
        self.T = T
        self.K = K
        self.V0 = V0
        self.r = r

    def price(self, kappa, theta, sigma_v, rho):

        FP = self.S0 * np.exp(self.r * self.T)

        def characteristic_function(w):
            alpha = (
                -(w**2) / 2 - w * 1j / 2
            )  # TODO: Grok says this is not general between PI1 and PI2
            Beta = (
                kappa - rho * sigma_v * 1j * w
            )  # TODO: Possible problem with alpha vs kappa
            gamma = (sigma_v**2) / 2
            h = np.sqrt(Beta**2 - 4 * alpha * gamma)
            rp = (Beta + h) / (sigma_v**2)
            rm = (Beta - h) / (sigma_v**2)
            g = rm / rp

            D = rm * (1 - np.exp(-h * self.T)) / (1 - g * np.exp(-h * self.T))
            C = kappa * (
                rm * self.T
                - (2 / (sigma_v**2)) * np.log((1 - g * np.exp(-h * self.T)) / (1 - g))
            )

            return np.exp(
                C * theta
                + D * self.V0
                + 1j * w * np.log(self.S0 * np.exp(self.r * self.T))
            )

        def integrand1(w):
            result = (
                np.exp(-1j * w * np.log(self.K)) * characteristic_function(w - 1j)
            ) / (1j * w * FP)
            return result.real

        def integrand2(w):
            result = (np.exp(-1j * w * np.log(self.K)) * characteristic_function(w)) / (
                1j * w
            )
            return result.real

        I1 = quad(integrand1, 1e-10, 500)  # Integral of PI1 (approx 0 to inf)
        I2 = quad(integrand2, 1e-10, 500)  # Integral of PI2 (approx 0 to inf)

        PI1 = 1 / 2 + 1 / np.pi * I1[0]
        PI2 = 1 / 2 + 1 / np.pi * I2[0]
        C0 = self.S0 * PI1 - np.exp(-self.r * self.T) * self.K * PI2

        return C0


def generate_multi_indices(p, d):
    indices = []
    # Loop through each total degree from 0 to d
    for s in range(d + 1):
        for combo in combinations(range(s + p - 1), p - 1):
            idx = np.zeros(p, dtype=int)
            prev = -1
            for i, c in enumerate(combo):
                idx[i] = c - prev - 1
                prev = c
            idx[p - 1] = s + p - 2 - prev
            # Flip to match convention
            indices.append(idx[::-1])

    return np.array(indices)


def build_Psi_uni(samps_std, K_idx, d):
    """
    Standardized basis builder
    samps_std: (N, p) array in [-1, 1]
    """
    N, p = samps_std.shape
    magK = len(K_idx)

    # Compute Univariate Tensor
    # Shape: (degree, dimension, realization)
    P = np.zeros((d + 1, p, N))
    for dim in range(p):
        P[:, dim, :] = poly.legendrePoly(samps_std[:, dim], d)

    # Build Multivariate Psi
    Psi = np.ones((N, magK))
    for k in range(magK):
        for dim in range(p):
            deg = K_idx[k, dim]
            Psi[:, k] *= P[deg, dim, :]
    return Psi
