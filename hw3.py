import numpy as np
import matplotlib.pyplot as plt

np.random.seed(530)

########## Problem 1 ##########
def C(x, y, L): 
    return (1/(2*L))*np.exp(-abs(x - y)/L)

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
    ax.plot(y, y_series, color='tab:blue')
    ax.set_title(f"L = {L_set[i]}")
    ax.set_xlabel("y")
    ax.set_ylabel("C(0, y)")
    ax.grid(True)

# 4. Auto-adjust spacing to prevent label overlap
plt.tight_layout()


########## Problem 2 ##########
# 2.1
CMat = np.zeros((101, 101)) # Autocorrelation
x = [-1 + (i-1)*0.02 for i in range(1, 102)]

# Construct Covariance Matrix
for i in range(1, 102):
    xi = -1 + (i-1)*0.02
    for j in range(1, 102):
        xj = -1 + (j-1)*0.02
        CMat[i-1, j-1] = C(xi, xj, 1)

# Construct means
alpha_bar = [np.cos(np.pi*(-1 + (i-1)*0.02)) for i in range(1, 102)]

ans_21 = np.random.multivariate_normal(alpha_bar, np.sqrt(CMat), 5)

# 2.2
data_22 = np.random.multivariate_normal(alpha_bar, np.sqrt(CMat), 500)
means_22 = np.mean(data_22, 0)
vars_22 = np.var(data_22, 0, ddof=1)

# Plot 2.1
fig, ax = plt.subplots()

ax.plot(x, np.transpose(ans_21))

ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.legend([f'Realization {i+1}' for i in range(5)]) 
ax.grid(True)

# Plot 2.2
fig, ax = plt.subplots()
ax.plot(x, alpha_bar, label='True mean')
ax.plot(x, means_22, label="Emperical mean")
ax.plot(x, alpha_bar+3*np.sqrt(np.diag(CMat)), label=f"3 Sigma") # TODO: Clean up the legend and label axes
ax.plot(x, alpha_bar-3*np.sqrt(np.diag(CMat)), label=f"3 Sigma")
ax.plot(x, means_22+3*np.sqrt(vars_22), label="3 STD")
ax.plot(x, means_22-3*np.sqrt(vars_22), label="3 STD")
ax.legend()
ax.grid(True)

# Plot 2.3



########## Problem 3 ##########


plt.show()