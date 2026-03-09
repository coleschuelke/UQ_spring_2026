import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd

from utils.heat_equation import HeatEquation

# Load the data
data = scipy.io.loadmat("HW08_Problem2")
print(data.keys())

q_samples = data["q_samples"]
s02_samples = data["variance0_samps"][0]

df = pd.DataFrame(q_samples.T)
df["s02"] = s02_samples

df.columns = ["Phi", "h", "s02"]
# Scale for m
df["Phi"] = df["Phi"] * 1e4
df["h"] = df["h"] * 1e4

q_bar = df.mean(0).to_numpy()
print(f"The mean of the sample is: \n{q_bar}")


eq = HeatEquation(k=2.37e2, Tamb=21)


# Helper functions
def Ts(x, q):
    q_dict = {
        "phi": q[0],
        "h": q[1],
    }

    return eq.Ts_vals(x, q_dict)


# Pass all values through the function
q_vals = [df["Phi"].values, df["h"].values]
temps10 = Ts(0.1, q_vals)

# Moments
print(f"The empirical mean is {np.mean(temps10)}")
print(f"The empirical variance is {np.var(temps10)}")

# Credibility intervals
lo10, hi10 = (np.quantile(temps10, 0.05), np.quantile(temps10, 0.95))
print(f"The approximated credible interval is [{lo10}, {hi10}]")

# Prediction intervals
eps = np.random.normal(0, np.sqrt((df["s02"].values)))
v = temps10 + eps
lo, hi = (np.quantile(v, 0.05), np.quantile(v, 0.95))
print(f"The prediction interval is [{lo}, {hi}]")

# Loop over all values of x
x0 = 0.1  # m
dx = 0.04  # m
x_vals = np.array([x0 + j * dx for j in range(15)])
n = len(x_vals)

ups = np.array(
    [
        96.14,
        80.12,
        67.66,
        57.96,
        50.90,
        44.84,
        39.75,
        36.16,
        33.31,
        31.15,
        29.28,
        27.88,
        27.18,
        26.40,
        25.86,
    ]
)

means = np.zeros((n,))
vars = np.zeros((n,))
cred_int = np.zeros((n, 2))
pred_int = np.zeros((n, 2))

for i, x in enumerate(x_vals):
    # Pass all values through the function
    q_vals = [df["Phi"].values, df["h"].values]
    temps = Ts(x, q_vals)

    # Moments
    mean = np.mean(temps)
    var = np.var(temps)
    print(f"The empirical mean is {mean}")
    print(f"The empirical variance is {var}")
    means[i] = mean
    vars[i] = var

    # Credibility intervals
    lo, hi = (np.quantile(temps, 0.05), np.quantile(temps, 0.95))
    print(f"The approximated credible interval is [{lo}, {hi}]")
    cred_int[i, :] = (lo, hi)

    # Prediction intervals
    eps = np.random.normal(0, np.sqrt((df["s02"].values)))
    v = temps + eps
    lo, hi = (np.quantile(v, 0.05), np.quantile(v, 0.95))
    print(f"The prediction interval is [{lo}, {hi}]")
    pred_int[i, :] = (lo, hi)

Ts_mean = Ts(x_vals, q_bar)
residuals = ups - Ts_mean

# Check for observations within prediction intervals
within = 0
for k, u in enumerate(ups):
    if (u < pred_int[k, 1]) and (u > pred_int[k, 0]):
        within += 1

print(f"Total measurements within prediction interval: {within}")
print(f"Percentage of measurements within prediction interval: {within/len(ups)}")
# Plot the results
# x=10cm
kdex = np.linspace(min(temps10), max(temps10), 1000)
kde = scipy.stats.gaussian_kde(temps10)
plt.hist(temps10, density=True, bins=100)
plt.plot(kdex, kde(kdex))
plt.vlines([lo10, hi10], 0, 0.5, colors="orange")

# All values of x
import matplotlib.pyplot as plt
import numpy as np

# Assuming these are your existing variables:
# x_vals, cred_int, pred_int, residuals, Ts_mean

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)

# --- Top Plot: Model, CI, and PI ---
# Plot Prediction Intervals (PI)
ax1.fill_between(
    x_vals,
    pred_int[:, 0],
    pred_int[:, 1],
    color="gray",
    alpha=0.2,
    label="95% Prediction Interval",
)

# Plot Credibility Intervals (CI)
ax1.fill_between(
    x_vals,
    cred_int[:, 0],
    cred_int[:, 1],
    color="blue",
    alpha=0.3,
    label="95% Credibility Interval",
)

# Plot the mean prediction Ts(x, q_bar)
ax1.plot(x_vals, Ts_mean, color="blue", lw=2, label=r"$T_s(x, \bar{q})$")
# Plot the measurements
ax1.scatter(x_vals, ups, color="red", marker="o", label=r"$\Upsilon (x)$")

ax1.set_ylabel(r"$T_s(x, q)$ (C)")
ax1.set_title("Model Predictions with Uncertainty Intervals")
ax1.legend(loc="upper right")
ax1.grid(True, linestyle="--", alpha=0.6)

# --- Bottom Plot: Residuals ---
# Plot the difference y(x) - Ts(x, q)
ax2.scatter(x_vals, residuals, color="red", s=15, label="Residuals")
ax2.axhline(0, color="black", lw=1, linestyle="-")

ax2.set_ylabel("Residuals (C)")
ax2.set_xlabel(r"$x$ (m)")
ax2.grid(True, linestyle="--", alpha=0.6)

# Refine layout
plt.tight_layout()

plt.show()
