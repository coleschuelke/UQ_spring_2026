import matplotlib.pyplot as plt
import numpy as np
from scipy import io

from utils.heat_equation import HeatEquation
from utils.markov_chain_monte_carlo import MCMC

debug = 0

# Load the data
data = io.loadmat("HW06_Problem3.mat")
ups = data["observations"][0]
x0 = 0.1  # m
dx = 0.04  # m
x = np.array([x0 + j * dx for j in range(15)])  # X values of the data
