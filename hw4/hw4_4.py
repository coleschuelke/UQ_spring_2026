import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, sin, exp, sqrt, lambdify


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

def Ts_fun(q):
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

    return Ts_fun(*args)

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

# Approximating with finite difference

def num_der_phi(q, delta):

    # Calculate the value of the function at the point here
    Ts_vec = Ts_fun(q)

    # Calculate the value of the function at phi+dphi, h
    qdelta = q+np.array([0, delta])
    Tsdelta_vec = Ts_fun(qdelta)

    # Calculate finite difference
    chi = (Tsdelta_vec - Ts_vec) / delta

    return -chi

def num_der_h(q, delta):
    # Calculate the value of the function at the point here
    Ts_vec = Ts_fun(q)

    # Calculate the value of the function at phi+dphi, h
    qdelta = q+np.array([0, delta])
    Tsdelta_vec = Ts_fun(qdelta)

    # Calculate finite difference
    chi = (Tsdelta_vec - Ts_vec) / delta

    return -chi


# Compute errors for all deltas 
deltas = [10**-x for x in range(15, 0, -1)]
phi_out = np.zeros((15, len(deltas)))
h_out = np.zeros((15, len(deltas)))

q_star = [-100000, 10]

for i, d in enumerate(deltas):
    num_phi = num_der_phi(q_star, d)
    num_h = num_der_phi(q_star, d)

    analytical = jac(q_star)

    a_phi = analytical[:, 0]
    a_h = analytical[:, 1]

    err_phi = abs((num_phi - a_phi)/a_phi)
    err_h = abs((num_h - a_h)/a_h)


    phi_out[:, i] = err_phi # must be transposed prior to plotting
    h_out[:, i] = err_h

legend_strings = [f'delta={delta}' for delta in deltas]

fig, ax = plt.subplots()
ax.plot(deltas, np.transpose(phi_out))
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend(legend_strings, loc='right')

fig, ax = plt.subplots()
ax.plot(deltas, np.transpose(phi_out))
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend(legend_strings, loc='right')
plt.show()