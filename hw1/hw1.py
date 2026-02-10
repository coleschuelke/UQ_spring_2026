import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy import symbols, diff, sin, exp, sqrt

np.random.seed(50)

# Problem 2
means = []
vars = []
sizes = [1, 2, 3, 4, 5, 6]
for n in sizes:
    # Draw Samples
    samples = np.random.normal(loc=0, scale=2, size=10**n)
    trans = np.sin(samples)
    
    # Compute mean and variance
    mean = np.mean(trans)
    var = np.var(trans)

    # Save the results
    means.append(mean)
    vars.append(var)

# True values
mean_true = 0
var_true = .49983

# Errors
mean_err = [abs(mean - mean_true) for mean in means]
var_err = [abs(var - var_true) for var in vars]

print("The mean error is", mean_err)
print("The variance error is", var_err)

plt.loglog([10**x for x in sizes], mean_err,)
plt.loglog([10**x for x in sizes], var_err,)
plt.legend(["mean", 'variance'])

print("The means are: ", means)
print("The variances are: ", vars)

# Problem 3 (Will need to pull this into another file for later homeworks)
# Set up the heat equation
a, b, h, k, x, Tamb, phi, L = symbols('a b h k x Tamb phi L')
gamma = sqrt((2*(a+b)*h) / (a*b*k))
top = exp(gamma*L)*(h+k*gamma)
bottom = exp(-gamma*L)*(h-k*gamma) + exp(gamma*L)*(h+k*gamma)
c1 = -(phi / (k*gamma)) * (top / bottom)
c2 = (phi / (k*gamma)) + c1
Ts = c1*exp(-gamma*x) + c2*exp(gamma*x) + Tamb

dTs_dh = diff(Ts, h)
dTs_dk = diff(Ts, k)
dTs_dphi = diff(Ts, phi)

# Test points
x0 = 0.1 # m
dx = 0.04 # m
xj = [x0 + (j-1)*dx for j in range(1, 16)]

# Parameters
h_pass = 20 # W / (m2 * C)
k_pass = 237 # W / (m * C)
phi_pass = -180000 # W / m2
Tamb_pass = 21 # deg C
a_pass = 0.0095 # m
b_pass = 0.0095 # m
L_pass = 0.7 # m

Ts_vals = []
dh_vals = []
dk_vals = []
dphi_vals = []

for x_pass in xj:
    # Construct dictionary
    values = {
        a: a_pass,
        b: b_pass, 
        h: h_pass,
        k: k_pass,
        phi: phi_pass,
        Tamb: Tamb_pass,
        x: x_pass,
        L: L_pass,
    }
    Ts_i = Ts.subs(values).evalf()
    dh_i = dTs_dh.subs(values).evalf()
    dk_i = dTs_dk.subs(values).evalf()
    dphi_i = dTs_dphi.subs(values).evalf()
    Ts_vals.append(Ts_i)
    dh_vals.append(dh_i)
    dk_vals.append(dk_i)
    dphi_vals.append(dphi_i)

cols = ['x', 'ts', 'dh', 'dk', 'dphi']
col_labels = ['x (m)', 'Ts (C)', 'dTs/dh (m2*C2 / W)', 'dTs/dk (m*C2 / W)', 'dTs/dphi (m2*C / W)']

df = pd.DataFrame(columns=cols)

df['x'] = xj
df['ts'] = Ts_vals
df['dh'] = dh_vals
df['dk'] = dk_vals
df['dphi'] = dphi_vals

fig, ax = plt.subplots(figsize=(8, 3)) 
ax.axis('tight')
ax.axis('off')

# Create table
formatted_values = df.map(lambda x: f"{x:.5g}").values
table = ax.table(cellText=formatted_values, colLabels=col_labels, loc='center')

# styling manually
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

plt.tight_layout()
plt.show()

print("Script Finished!")