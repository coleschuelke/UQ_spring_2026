import matplotlib.pyplot as plt
import numpy as np
from scipy import io
import pandas as pd
np.random.seed(400)


########## Problem 1 ##########
def C(x, y, L):
    return (1 / (2 * L)) * np.exp(-abs(x - y) / L)


y = np.linspace(-100, 100, 10000)
L_set = [0.1, 10, 100, 10000]

results = []

for L in L_set:
    result = C(0, y, L)
    results.append(result)

# Plotting (Used Gemini)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8), sharex=True)

# 3. Iterate over data and flat axes simultaneously
for ax, y_series, i in zip(axes.flat, results, range(4)):
    ax.plot(y, y_series, color="tab:blue")
    ax.set_title(f"L = {L_set[i]}")
    ax.set_xlabel("y")
    ax.set_ylabel("C(0, y)")
    ax.grid(True)

# 4. Auto-adjust spacing to prevent label overlap
plt.tight_layout()


########## Problem 2 ##########
# 2.1
CMat = np.zeros((101, 101))  # Autocorrelation
x = [-1 + (i - 1) * 0.02 for i in range(1, 102)]
x = np.array(x)
x_gauss = [-1 + (i - 1) * 0.02 for i in range(1, 204)]

# Construct Covariance Matrix
for i in range(1, 102):
    xi = -1 + (i - 1) * 0.02
    for j in range(1, 102):
        xj = -1 + (j - 1) * 0.02
        CMat[i - 1, j - 1] = C(xi, xj, 1)

# Construct means
alpha_bar = [np.cos(np.pi * (-1 + (i - 1) * 0.02)) for i in range(1, 102)]

ans_21 = np.random.multivariate_normal(alpha_bar, np.sqrt(CMat), 5)

# 2.2
data_22 = np.random.multivariate_normal(alpha_bar, np.sqrt(CMat), 500)
means_22 = np.mean(data_22, 0)
vars_22 = np.var(data_22, 0, ddof=1)

# Plot 2.1
fig, ax = plt.subplots()

ax.plot(x, np.transpose(ans_21))

ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.legend([f"Realization {i+1}" for i in range(5)])
ax.grid(True)

# Plot 2.2
fig, ax = plt.subplots()
ax.plot(x, alpha_bar, label="True mean")
ax.plot(x, means_22, label="Emperical mean")
ax.plot(
    x, alpha_bar + 3 * np.sqrt(np.diag(CMat)), label=f"3 Sigma"
)  # TODO: Clean up the legend and label axes
ax.plot(x, alpha_bar - 3 * np.sqrt(np.diag(CMat)), label=f"3 Sigma")
ax.plot(x, means_22 + 3 * np.sqrt(vars_22), label="3 STD")
ax.plot(x, means_22 - 3 * np.sqrt(vars_22), label="3 STD")
ax.legend()
ax.grid(True)

# Plot 2.3
fig, ax = plt.subplots()
ax.hist(data_22[:, 50], density=True)
ax.plot(
    x_gauss,
    [
        1
        / (np.sqrt(2 * np.pi * CMat[50, 50]))
        * np.exp(-((xi - alpha_bar[50]) ** 2) / (2 * CMat[50, 50]))
        for xi in x_gauss
    ],
)

########## Problem 3 ##########
# Load the Eta values
Eta = io.loadmat("HW03_EtaVals")
L_vals = Eta['L_values'][0]
keep_keys = ['etaVals_000p1', 'etaVals_001p0', 'etaVals_010p0', 'etaVals_100p0']
Eta = {k: Eta[k] for k in keep_keys if k in Eta}
Eta = {k: v[0] for k, v, in Eta.items()}
Eta = pd.DataFrame(Eta)
Eta.columns = ['0.1', '1', '10', '100']

def lambda_n(n, L):
    eta = Eta.iloc[n, (L_vals == float(L)).argmax()]
    l = 1 / (1 + L**2*eta**2)
    return l

def phi_n(n, L, x):
    if type(x) != np.ndarray:
        x = np.array(x)
    eta = Eta.iloc[n, (L_vals == float(L)).argmax()]
    if n % 2 ==0:
        phi = np.cos(eta*x) / (np.sqrt(1 + (np.sin(2*eta) / (2*eta))))
    else:
        phi = np.sin(eta*x) / (np.sqrt(1 - (np.sin(2*eta) / (2*eta))))

    return phi

def alpha_n(n, L, x):
    if type(x) != np.ndarray:
        x = np.array(x)
    a = np.sqrt(lambda_n(n, L))*phi_n(n, L, x)
    return a

# print(alpha_n(3, 1, x)) # works for now

# 3.1 
num = 11
eigenvalues = np.zeros((num, 4))
for n in range(num):
    for i, lval in enumerate(L_vals):
        eigenvalues[n, i-1] = lambda_n(n, lval)

# 3.2
Q = np.random.normal(0, 1, 100)
Ns = [2, 10, 50, 100]
aN = np.zeros((len(x), 4))

for i, N in enumerate(Ns):
    sum = np.zeros((len(x), ))
    for n in range(N):
        sum += Q[n]*alpha_n(n, 1.0, x)
    a = np.cos(np.pi*x) + sum
    aN[:, i] = a

# 3.3
Q = np.reshape(np.random.normal(0, 1, 500), (5, 100))

ans_33 = np.zeros((len(x), 4, 5))
for i in range(np.size(Q, 0)):
    Qi = Q[i, :]
    for j, N in enumerate(Ns):
        sum = np.zeros((len(x),))
        for n in range(N):
            sum += Qi[n]*alpha_n(n, 1.0, x)
        a = np.cos(np.pi*x) + sum
        ans_33[:, j, i] = a

# 3.5 (rework of 2.2)
Q_35 = np.reshape(np.random.normal(0, 1, 25000), (500, 50))  # using N=50

ans_35 = np.zeros((len(x), 500))
for i in range(np.size(Q_35, 0)):
    Qi = Q_35[i, :]
    sum = np.zeros((len(x),))
    for n in range(np.size(Q_35, 1)): # 50
        sum += Qi[n]*alpha_n(n, 1.0, x)
    a = np.cos(np.pi*x) + sum
    ans_35[:, i] = a

means_35 = np.mean(ans_35, 1)
vars_35 = np.mean(ans_35, 1)

# 3.6 (rework of 2.3)
# Just plotting


# Plot 3.1
fig, ax = plt.subplots()
ax.plot(eigenvalues, '-x')
ax.set_yscale('log')

# Plot 3.2
fig, ax = plt.subplots()
ax.plot(aN) # Check mean different from andrea

# Plot 3.3
fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

for i in range(4):
    # Select the current subplot axis
    ax = axes[i]
    
    # Iterate through the 5 series in the third dimension
    for j in range(5):
        # data[:, i, j] extracts all n elements for subplot i, series j
        ax.plot(ans_33[:, i, j], label=f'Q({j+1})')
    
    ax.set_title(f'N = {Ns[i]}')
    ax.set_ylabel('Value')
    ax.legend(loc='upper right', fontsize='small', ncol=5)

# Plot 3.5
fig, ax = plt.subplots()
ax.plot(x, means_35)
# TODO: Plot bounds

# Plot 3.6
fig, ax = plt.subplots()
ax.hist(ans_35[50, :], density=True)
ax.plot(
    x_gauss,
    [
        1
        / (np.sqrt(2 * np.pi * CMat[50, 50]))
        * np.exp(-((xi - alpha_bar[50]) ** 2) / (2 * CMat[50, 50]))
        for xi in x_gauss
    ],
)


plt.show()
