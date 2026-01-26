import pandas as pd
from scipy import stats, io
import numpy as np

np.random.seed(42)

##### Problem 1 #####

# Import and format data
data = io.loadmat("HW02_Problem1.mat")
data = data["X"]
data = [x for sl in data for x in sl]

# Get statistics
mean_data = np.mean(data)
var_data = np.var(data)  # maybe should set ddof=1?
print(mean_data, var_data)

# Confidence intervals
ci_low, ci_hi = stats.t.interval(0.8, len(data), loc=mean_data, scale=stats.sem(data))

print(ci_low, ci_hi)

# Generating my own samples
gen_means = []
gen_vars = []
gen_bounds = []
for i in range(50):
    my_samples = np.random.normal(1, np.sqrt(5), 10)
    my_mean = np.mean(my_samples)
    my_var = np.var(my_samples)

    ci_low, ci_hi = stats.t.interval(
        0.8, len(my_samples), loc=my_mean, scale=stats.sem(my_samples)
    )

    # Save statistics
    gen_means.append(my_mean)
    gen_vars.append(my_var)
    gen_bounds.append((ci_low, ci_hi))


# Plot the bounds and the true statistics


##### Problem 3 #####


def draw(n):
    reals = np.random.uniform(-1, 1, n)
    x_bar = np.mean(reals)

    return x_bar


runs = [1, 10, 100, 1000, 10000]
cdfs = []
for j in range(len(runs)):
    means = []
    for jj in range(1000):
        means.append(draw(runs(j)))

    cdfs[j] = stats.ecdf(means)


# Plot the cdfs of the means
