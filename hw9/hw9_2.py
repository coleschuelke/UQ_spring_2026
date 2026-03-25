import matplotlib.pyplot as plt
import numpy as np

from utils import ASE379L_UQ_Polynomials as poly

q = np.linspace(-4, 4, 1000)
polys = poly.hermitePoly(q, 5)
# What does max min value for ordinate of pm 4 mean??
fig, ax = plt.subplots()
for k in range(5):
    ax.plot(q, polys[k])
ax.set_ylim([-10, 10])


# Calculating expectations of the polynomials
points_phys, weights_phys = np.polynomial.hermite.hermgauss(20)

points = np.sqrt(2) * points_phys
weights = 1 / np.sqrt(np.pi) * weights_phys
polys = poly.hermitePoly(points, 5)

EV22 = (
    polys[1] ** 2 @ weights.T
)  # * np.exp(-0.5 * points**2) / np.sqrt(2 * np.pi) @ weights.T
print(EV22)  # This is .141 not 1 as we would expect

EV23 = polys[1] * polys[2] * np.exp(-0.5 * points**2) / np.sqrt(2 * np.pi) @ weights.T
print(EV23)  # This is 0 as we would expect

# Creating training data
q_vals = np.random.normal(0, 1, 100)
u_vals = np.exp(-q_vals)
emp_mean = np.mean(u_vals)
emp_var = np.var(u_vals)
print(emp_mean, emp_var)

# Least squares approach to finding coefficients
Y = u_vals.T
polys = poly.hermitePoly(q_vals, 10)
Psi = polys.T

U = np.linalg.inv(Psi.T @ Psi) @ Psi.T @ Y
print(U)

# PCE mean and variance
print(f"The analytic mean is {np.exp(0.5)}")
print(f"The PCE mean is {U[0]}")
print(f"The analytic variance is {np.exp(2) - np.exp(1)}")
print(f"The PCE variance is {sum(U**2)}")

# Quadrature PCE
u_analytic = np.zeros(11)
points_phys, weights_phys = np.polynomial.hermite.hermgauss(20)

points = np.sqrt(2) * points_phys
weights = 1 / np.sqrt(np.pi) * weights_phys
polys = poly.hermitePoly(points, 10)

for k in range(11):
    u_analytic[k] = np.exp(-points) * polys[k] @ weights.T

print(u_analytic)

print(f"The analytic mean is {np.exp(0.5)}")
print(f"The PCE mean is {u_analytic[0]}")
print(f"The analytic variance is {np.exp(2) - np.exp(1)}")
print(f"The PCE variance is {sum(u_analytic**2)}")

# Plotting
fig, ax = plt.subplots()
ax.plot(abs(U))
ax.set_yscale("log")

fig, ax = plt.subplots()
ax.plot(abs(U))
ax.set_yscale("log")
plt.show()
