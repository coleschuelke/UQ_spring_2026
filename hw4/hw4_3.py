import scipy
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from sympy import symbols, diff, sin, exp, sqrt, lambdify


data = scipy.io.loadmat('exercise_7p7_data')

data = data['copper_data']

df = pd.DataFrame(data)

ups = np.array(df.iloc[:, 1])

syms = symbols('a b h k x Tamb phi L')
a, b, h, k, x, Tamb, phi, L = syms
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
k_pass = 401 # W / (m * C)
phi_pass = -180000 # W / m2
Tamb_pass = 21.29 # deg C
a_pass = 0.0095 # m
b_pass = 0.0095 # m
L_pass = 0.7 # m

# Estimating q = [phi, h]

def cf(q):
    phi_guess = q[0]
    h_guess = q[1]

    x_vals = np.array([0.1 + k*0.04 for k in range(15)])

    values = {
        a: a_pass,
        b: b_pass, 
        h: h_guess,
        k: k_pass,
        Tamb: Tamb_pass,
        phi: phi_guess,
        L: L_pass,
    }

    Ts_fun = lambdify(syms, Ts, 'numpy')
    args = [x_vals if s == x else values[s] for s in syms]

    Ts_vals = Ts_fun(*args)

    return ups - Ts_vals

def jac(q):
    phi_guess = q[0]
    h_guess = q[1]

    x_vals = np.array([0.1 + k*0.04 for k in range(15)])

    values = {
        a: a_pass,
        b: b_pass, 
        h: h_guess,
        k: k_pass,
        Tamb: Tamb_pass,
        phi: phi_guess,
        L: L_pass,
    }
    
    args = [x_vals if s == x else values[s] for s in syms]

    dTs_dphi_fun = lambdify(syms, dTs_dphi, 'numpy')
    dTs_dh_fun = lambdify(syms, dTs_dh, 'numpy')

    dTs_dphi_vals = dTs_dphi_fun(*args)
    dTs_dh_vals = dTs_dh_fun(*args)

    return np.transpose(np.array([-dTs_dphi_vals, -dTs_dh_vals]))

x0 = [-100000, 10]

q_hat  = scipy.optimize.least_squares(cf, x0, jac)

print(q_hat.x)

    