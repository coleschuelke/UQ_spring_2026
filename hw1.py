import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

print("The means are: ", means)
print("The variances are: ", vars)

# Problem 3 (Will need to pull this into another file for later homeworks)
def T_s(x, h, k, phi, T_amb, a=0.095, b=0.095, L=0.7):
    gamma = np.sqrt(2*(a+b)*h/(a*b*k))
    c1 = (-phi/(k*gamma)) * (np.exp(gamma*L)*(h + k*gamma)) / (np.exp(-gamma*L)*(h + k*gamma) + np.exp(gamma*L)*(h + k*gamma))
    c2 = phi/(k*gamma) + c1
    T_s = c1*np.exp(-gamma*x) + c2*np.exp(gamma*x) + T_amb

    return T_s

def dTs_dh(x, h, k, phi, T_amb, a=0.095, b=0.095, L=0.7):
    dh = (
        (2**(1/2)*phi*np.exp(2*2**(1/2)*L*((h*(a + b))/(a*b*k))**(1/2)) 
         - 2**(1/2)*phi*np.exp(2*2**(1/2)*(L + x)*((h*(a + b))/(a*b*k))**(1/2)) 
         + 2**(1/2)*phi*np.exp(4*2**(1/2)*L*((h*(a + b))/(a*b*k))**(1/2)) 
         - 2**(1/2)*phi*np.exp(2*2**(1/2)*x*((h*(a + b))/(a*b*k))**(1/2)) 
         + 2*phi*x*np.exp(2*2**(1/2)*x*((h*(a + b))/(a*b*k))**(1/2))*((h*(a + b))/(a*b*k))**(1/2) 
         - 4*L*phi*np.exp(2*2**(1/2)*(L + x)*((h*(a + b))/(a*b*k))**(1/2))*((h*(a + b))/(a*b*k))**(1/2) 
         - 4*L*phi*np.exp(2*2**(1/2)*L*((h*(a + b))/(a*b*k))**(1/2))*((h*(a + b))/(a*b*k))**(1/2) 
         + 2*phi*x*np.exp(2*2**(1/2)*(L + x)*((h*(a + b))/(a*b*k))**(1/2))*((h*(a + b))/(a*b*k))**(1/2) 
         + 2*phi*x*np.exp(2*2**(1/2)*L*((h*(a + b))/(a*b*k))**(1/2))*((h*(a + b))/(a*b*k))**(1/2) 
         + 2*phi*x*np.exp(4*2**(1/2)*L*((h*(a + b))/(a*b*k))**(1/2))*((h*(a + b))/(a*b*k))**(1/2))/(4*h*k*(np.exp(2**(1/2)*x*((h*(a + b))/(a*b*k))**(1/2)) 
         + 2*np.exp(2**(1/2)*(2*L + x)*((h*(a + b))/(a*b*k))**(1/2)) 
         + np.exp(2**(1/2)*(4*L + x)*((h*(a + b))/(a*b*k))**(1/2)))*((h*(a + b))/(a*b*k))**(1/2))
    )
    return dh

def dTs_dk(x, h, k, phi, T_amb, a=0.095, b=0.095, L=0.7):
    dk = (
        -(2**(1/2)*phi*np.exp(2*2**(1/2)*(L + x)*((h*(a + b))/(a*b*k))**(1/2)) 
          - 2**(1/2)*phi*np.exp(2*2**(1/2)*L*((h*(a + b))/(a*b*k))**(1/2)) 
          - 2**(1/2)*phi*np.exp(4*2**(1/2)*L*((h*(a + b))/(a*b*k))**(1/2)) 
          + 2**(1/2)*phi*np.exp(2*2**(1/2)*x*((h*(a + b))/(a*b*k))**(1/2)) 
          + 2*phi*x*np.exp(2*2**(1/2)*x*((h*(a + b))/(a*b*k))**(1/2))*((h*(a + b))/(a*b*k))**(1/2) 
          - 4*L*phi*np.exp(2*2**(1/2)*(L + x)*((h*(a + b))/(a*b*k))**(1/2))*((h*(a + b))/(a*b*k))**(1/2) 
          - 4*L*phi*np.exp(2*2**(1/2)*L*((h*(a + b))/(a*b*k))**(1/2))*((h*(a + b))/(a*b*k))**(1/2) 
          + 2*phi*x*np.exp(2*2**(1/2)*(L + x)*((h*(a + b))/(a*b*k))**(1/2))*((h*(a + b))/(a*b*k))**(1/2) 
          + 2*phi*x*np.exp(2*2**(1/2)*L*((h*(a + b))/(a*b*k))**(1/2))*((h*(a + b))/(a*b*k))**(1/2) 
          + 2*phi*x*np.exp(4*2**(1/2)*L*((h*(a + b))/(a*b*k))**(1/2))*((h*(a + b))/(a*b*k))**(1/2))/(4*k**2*(np.exp(2**(1/2)*x*((h*(a + b))/(a*b*k))**(1/2)) 
          + 2*np.exp(2**(1/2)*(2*L + x)*((h*(a + b))/(a*b*k))**(1/2)) 
          + np.exp(2**(1/2)*(4*L + x)*((h*(a + b))/(a*b*k))**(1/2)))*((h*(a + b))/(a*b*k))**(1/2))
    )
    return dk

def dTs_dphi(x, h, k, phi, T_amb, a=0.095, b=0.095, L=0.7):
    dphi = (
        -(2**(1/2)*(np.exp(2*2**(1/2)*L*((h*(a + b))/(a*b*k))**(1/2)) 
        - np.exp(2*2**(1/2)*x*((h*(a + b))/(a*b*k))**(1/2))))/(2*k*(np.exp(2**(1/2)*x*((h*(a + b))/(a*b*k))**(1/2)) 
        + np.exp(2**(1/2)*(2*L + x)*((h*(a + b))/(a*b*k))**(1/2)))*((h*(a + b))/(a*b*k))**(1/2))
    )
    return dphi

x0 = 0.1 # m
dx = 0.04 # m
xj = [x0 + (j-1)*dx for j in range(1, 16)]

# Parameters
h = 20 # W / (m2 * C)
k = 237 # W / (m * C)
phi = -180000 # W / m2
T_amb = 21 # deg C

Ts_vals = []
dh_vals = []
dk_vals = []
dphi_vals = []

for x in xj:
    Ts = T_s(x, h, k, phi, T_amb)
    dh = dTs_dh(x, h, k, phi, T_amb)
    dk = dTs_dk(x, h, k, phi, T_amb)
    dphi = dTs_dphi(x, h, k, phi, T_amb)
    Ts_vals.append(Ts)
    dh_vals.append(dh)
    dk_vals.append(dk)
    dphi_vals.append(dphi)

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
table = ax.table(cellText=df.round(4).values, colLabels=col_labels, loc='center')

# styling manually
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

plt.show()