import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.qmc import LatinHypercube, scale
from utils import ASE379L_UQ_PCE as pce
from utils import ASE379L_UQ_Polynomials as poly

from aux import HestonModel

# Control
Nsamps = 10  # Number of function evaluations to fit the PCE
K_pce = 0  # Order of the PCE (Should be able to justify this)


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

# Evaluate the option price using the closed-form solution to the Heston model
evals = np.zeros((Nsamps, 4))
for ii in range(Nsamps):
    evals[ii, :] = model.price(*(scaled_samps[ii, :]))

# Fit the PCE using least squares

# Fit the PCE using quadrature

# Calculate the Sobol indices

# Compare with market data?
