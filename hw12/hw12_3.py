import numpy as np
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt

data = loadmat("HW12_Problem03")
df = pd.DataFrame(data["M"])

# Normalize the data to zero mean, unit covariance
means = np.array(df.mean())
vars = np.array(df.var())
df = df - means
df = df / np.sqrt(vars)

# Confirm normalizaton
# print(df.mean())
# print(df.var())

# SVD
U, S, V = np.linalg.svd(df)


# Fraction of variance as a function of component
tot_var = sum(S)
part_var = np.zeros(S.shape)
for k in range(len(S)):
    part_var[k] = sum(S[:k])
per_var = part_var / tot_var

# Scatter plots
normalized_partial_matrices = []
for k in range(1, 7):
    smat = np.zeros(df.shape)
    smat[:k, :k] = np.diag(S[:k])
    mat = np.dot(U, np.dot(smat, V))
    normalized_partial_matrices.append(mat)

normalized_partial_matrices = np.array(normalized_partial_matrices)


# ======== Plotting ========
plt.plot(per_var, "b*-")

fig, axes = plt.subplots(2, 3)
for i, ax in enumerate(axes.flat):
    mat = normalized_partial_matrices[i, :, :]
    mat = np.sqrt(vars).reshape(1, 10) * mat
    mat = mat + means

    ax.scatter(mat[:, 0], mat[:, 1], marker="o", s=0.75)
    ax.set_xlabel(f"{i+1} components")

fix, ax = plt.subplots()
lines = ["m-", "g:", "b-."]
for i in range(3):
    ax.plot(V[:, i], lines[i], label=f"V({i+1})")
ax.legend()

plt.show()
