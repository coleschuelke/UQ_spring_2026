import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt


data = scipy.io.loadmat("exercise_7p8_data")

df = pd.DataFrame(data["cement_data"])
df.columns = ["x1", "x2", "x3", "x4", "v"]
df.insert(0, "const", np.ones(len(df)))

X = np.array(df.iloc[:, 0:5])
ups = np.array(df["v"])

n, p = X.shape

beta_hat = scipy.linalg.solve(X.T @ X, X.T @ ups)
beta_var = (1 / (n - p)) * (ups - X @ beta_hat).T @ (ups - X @ beta_hat)

xtxinv = scipy.linalg.inv(X.T @ X)
xtxinv_diag = np.diag(xtxinv)

beta_ci95_lo, beta_ci95_hi = scipy.stats.t.interval(
    0.95, df=n - p, loc=beta_hat, scale=np.sqrt(beta_var * xtxinv_diag)
)  # 95% confidence intervals
beta_ci2sig_lo = beta_hat - 2 * np.sqrt(beta_var * xtxinv_diag)
beta_ci2sig_hi = beta_hat + 2 * np.sqrt(beta_var * xtxinv_diag)

print(beta_hat)
print(beta_ci95_hi, beta_ci95_lo)
print(beta_ci2sig_hi)
print(beta_ci2sig_lo)

##### x1 only ####

X_1 = np.array(df.iloc[:, 0:2])
ups_1 = np.array(df["v"])

n_1, p_1 = X_1.shape

beta_hat_1 = scipy.linalg.solve(X_1.T @ X_1, X_1.T @ ups_1)
beta_var_1 = (
    (1 / (n_1 - p_1)) * (ups_1 - X_1 @ beta_hat_1).T @ (ups_1 - X_1 @ beta_hat_1)
)

xtxinv_1 = scipy.linalg.inv(X_1.T @ X_1)
xtxinv_diag_1 = np.diag(xtxinv_1)

beta_ci95_lo_1, beta_ci95_hi_1 = scipy.stats.t.interval(
    0.95, df=n_1 - p_1, loc=beta_hat_1, scale=np.sqrt(beta_var_1 * xtxinv_diag_1)
)  # 95% confidence intervals
beta_ci2sig_lo_1 = beta_hat_1 - 2 * np.sqrt(beta_var_1 * xtxinv_diag_1)
beta_ci2sig_hi_1 = beta_hat_1 + 2 * np.sqrt(beta_var_1 * xtxinv_diag_1)

print("Begin x1 only")
print(beta_hat_1)
print(beta_ci95_hi_1, beta_ci95_lo_1)
print(beta_ci2sig_hi_1)
print(beta_ci2sig_lo_1)


##### x1 & x2 #####
X_2 = np.array(df.iloc[:, 0:3])
ups_2 = np.array(df["v"])

n_2, p_2 = X_2.shape

beta_hat_2 = scipy.linalg.solve(X_2.T @ X_2, X_2.T @ ups_2)
beta_var_2 = (
    (1 / (n_2 - p_2)) * (ups_2 - X_2 @ beta_hat_2).T @ (ups_2 - X_2 @ beta_hat_2)
)

xtxinv_2 = scipy.linalg.inv(X_2.T @ X_2)
xtxinv_diag_2 = np.diag(xtxinv_2)

beta_ci95_lo_2, beta_ci95_hi_2 = scipy.stats.t.interval(
    0.95, df=n_2 - p_2, loc=beta_hat_2, scale=np.sqrt(beta_var_2 * xtxinv_diag_2)
)  # 95% confidence intervals
beta_ci2sig_lo_2 = beta_hat_2 - 2 * np.sqrt(beta_var_2 * xtxinv_diag_2)
beta_ci2sig_hi_2 = beta_hat_2 + 2 * np.sqrt(beta_var_2 * xtxinv_diag_2)

print("Begin x1 & x2")
print(beta_hat_2)
print(beta_ci95_hi_2, beta_ci95_lo_2)
print(beta_ci2sig_hi_2)
print(beta_ci2sig_lo_2)


# Plotting
fig, ax = plt.subplots()
ax.scatter(range(len(ups)), (ups - X @ beta_hat).T)
ax.set_ylim([-12, 12])
ax.hlines(0, -2, 14, "r", "--")
ax.set_title("Residuals")
ax.set_xlabel("Observation #")
ax.set_ylabel("Residual (cal/g)")

fig, ax = plt.subplots()
ax.scatter(range(len(ups_1)), (ups_1 - X_1 @ beta_hat_1).T)
ax.set_ylim([-12, 12])
ax.hlines(0, -2, 14, "r", "--")
ax.set_title("Residuals")
ax.set_xlabel("Observation #")
ax.set_ylabel("Residual (cal/g)")

fig, ax = plt.subplots()
ax.scatter(range(len(ups_2)), (ups_2 - X_2 @ beta_hat_2).T)
ax.set_ylim([-12, 12])
ax.hlines(0, -2, 14, "r", "--")
ax.set_title("Residuals")
ax.set_xlabel("Observation #")
ax.set_ylabel("Residual (cal/g)")
plt.show()
