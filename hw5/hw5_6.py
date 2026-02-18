import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

# Import the data
data = scipy.io.loadmat("HW05_Problem6")
keep_keys = ["observations", "times"]
data = {key: data[key][0] for key in keep_keys}
df = pd.DataFrame(data)

# Problem Setup
t = np.array(df["times"])
K = 20.5

ups = np.array(df["observations"])
sigma02 = 0.1**2

# Setting up the posterior
bounds = (1.2, 1.8)


def pi_denom(zeta, q):
    fq = 2 * np.exp(-q * t / 2) * np.cos(t * np.sqrt(K - q**2 / 4))
    fz = 2 * np.exp(-zeta * t / 2) * np.cos(t * np.sqrt(K - zeta**2 / 4))

    SSq = (ups - fq).T @ (ups - fq)
    SSz = (ups - fz).T @ (ups - fz)

    return np.exp((SSq - SSz) / (2 * sigma02))


integral = scipy.integrate.quad(pi_denom, *bounds, args=(1.45))

print(integral)
piqv = 1 / integral[0]

print(piqv)

q_vals = np.arange(*bounds, 0.005)
density = np.zeros((len(q_vals)))
for i, q in enumerate(q_vals):
    d = scipy.integrate.quad(pi_denom, *bounds, args=(q))
    density[i] = 1 / d[0]

print(f"qMAP = {q_vals[np.argmax(density)]:.3f}")

# Model of the system with estimated damping
CMAP = 1.45
z_est = 2 * np.exp(-CMAP * t / 2) * np.cos(t * np.sqrt(K - CMAP**2 / 4))

# Plotting
fig, ax = plt.subplots()
ax.plot(q_vals, density)
ax.set_xlabel("Damping Parameter C")
ax.set_title("True Posterior")

fig, ax = plt.subplots()
ax.plot(t, z_est, color="b", label="Model")
ax.scatter(t, ups, color="m", label="Measurements")
ax.set_ylabel("Amplitude")
ax.set_xlabel("Time")
ax.legend()
ax.set_title("Estimated Model")

plt.show()
