import numpy as np
from scipy.io import loadmat

# Load the PCE components
data = loadmat("HW12_Problem1.mat")
uk = data["u_k"][0]
uk_var = uk[1:]
K = data["multi_index"]

# Calculate the total variance
var = uk_var @ uk_var.T
print(f"The total variance is; {var}")


# Values containing q1 at all
mask = K[1:, 0] > 0
uk_q1_tot = uk_var[mask]
print(f"St1: {uk_q1_tot @ uk_q1_tot.T}")

# Values containing q1 alone
mask2 = K[1:, 1] == 0
mask_tot = mask & mask2
uk_q1_fo = uk_var[mask_tot]
print(f"S1: {uk_q1_fo @ uk_q1_fo.T}")

# Values containing q2 at all
mask = K[1:, 1] > 0
uk_q2_tot = uk_var[mask]
print(f"St2: {uk_q2_tot @ uk_q2_tot.T}")

# Values containing q2 alone
mask2 = K[1:, 0] == 0
mask_tot = mask & mask2
uk_q2_fo = uk_var[mask_tot]
print(f"S2: {uk_q2_fo @ uk_q2_fo.T}")
