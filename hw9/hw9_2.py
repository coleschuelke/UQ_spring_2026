import matplotlib.pyplot as plt
import numpy as np

from utils import ASE379L_UQ_Polynomials as poly

np.random.seed(7)
q = np.linspace(-4, 4, 1000)
polys = poly.hermitePoly(q, 5)
# What does max min value for ordinate of pm 4 mean??
fig, ax = plt.subplots()
for k in range(6):
    ax.plot(q, polys[k])
ax.set_ylim([-4, 4])
ax.set_xlabel("Abiscissa")
ax.set_ylabel("Ordinate")

# Calculating expectations of the polynomials
points_phys, weights_phys = np.polynomial.hermite.hermgauss(20)

points = np.sqrt(2) * points_phys
weights = 1 / np.sqrt(np.pi) * weights_phys
polys = poly.hermitePoly(points, 5)
print("The EVs of the inner products are: ")
EV22 = polys[1] ** 2 @ weights.T
print(EV22)  # This is one as we would expect

EV23 = (polys[1] * polys[2]) @ weights.T
print(EV23)  # This is 0 as we would expect

# Creating training data
q_vals = np.random.normal(0, 1, 100)
u_vals = np.exp(-q_vals)
emp_mean = np.mean(u_vals)
emp_var = np.var(u_vals)
print("The empirical mean and variance are: ")
print(emp_mean, emp_var)

# Least squares approach to finding coefficients
Y = u_vals.T
polys = poly.hermitePoly(q_vals, 10)
Psi = polys.T

U = np.linalg.inv(Psi.T @ Psi) @ Psi.T @ Y

# PCE mean and variance
print(f"The analytic mean is {np.exp(0.5)}")
print(f"The analytic variance is {np.exp(2) - np.exp(1)}")
print("------------------------- Least squares")
print(f"The PCE mean is {U[0]}. Error: {U[0] - np.exp(0.5)}")
print(f"The analytic mean is {np.exp(0.5)}")
print(
    f"The PCE variance is {sum(U[1:]**2)}. Error: {sum(U[1:]**2)-np.exp(2) + np.exp(1)}"
)

# Quadrature PCE
u_analytic = np.zeros(11)
points_phys, weights_phys = np.polynomial.hermite.hermgauss(20)

points = np.sqrt(2) * points_phys
weights = 1 / np.sqrt(np.pi) * weights_phys
polys = poly.hermitePoly(points, 10)

for k in range(11):
    u_analytic[k] = np.exp(-points) * polys[k] @ weights.T


print("------------------------- Analytic Coefficients")
print(f"The PCE mean is {u_analytic[0]}. Error: {u_analytic[0] - np.exp(0.5)}")
print(
    f"The PCE variance is {sum(u_analytic[1:]**2)}. Error: {-sum(u_analytic[1:]**2) + np.exp(2) - np.exp(1)}"
)

# Plotting
fig, ax = plt.subplots()
ax.plot(abs(U))
ax.set_yscale("log")
ax.set_xlabel("k")
ax.set_ylabel(r"$u^k$")

fig, ax = plt.subplots()
ax.plot(abs(U))
ax.set_yscale("log")
ax.set_xlabel("k")
ax.set_ylabel(r"$u^k$")
plt.show()
