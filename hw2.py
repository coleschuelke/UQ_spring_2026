import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from scipy import io, stats

np.random.seed(42)

##### Problem 1 #####

# Import and format data
data = io.loadmat("HW02_Problem1.mat")
data = data["X"]
data = [x for sl in data for x in sl]

# Get statistics
mean_data = np.mean(data)
var_data = np.var(data, ddof=1)  # maybe should set ddof=1?
print(f"The sample mean is {mean_data}")
print(f"The sample variance is {var_data}")

# Confidence intervals
ci_low, ci_hi = stats.t.interval(0.9, len(data), loc=mean_data, scale=stats.sem(data))
print(f"The confidence interval given the data is ({ci_low}, {ci_hi})")

# Generating my own samples
gen_means = []
gen_vars = []
gen_bounds = []
true_mean = 1
true_var = 5
for i in range(50):
    my_samples = np.random.normal(true_mean, np.sqrt(true_var), 10)
    my_mean = np.mean(my_samples)
    my_var = np.var(my_samples)

    ci_low, ci_hi = stats.t.interval(
        0.8, len(my_samples), loc=my_mean, scale=stats.sem(my_samples)
    )

    # Save statistics
    gen_means.append(my_mean)
    gen_vars.append(my_var)
    gen_bounds.append((ci_low, ci_hi))


# Plot the bounds and the true statistics (Used Gemini)

x = np.arange(1, len(gen_means) + 1)
yerr_low = []
yerr_high = []
colors = []

for m, (low, high) in zip(gen_means, gen_bounds):
    # Calculate relative offsets for yerr
    yerr_low.append(m - low)
    yerr_high.append(high - m)

    # Check if true mean is captured
    if low <= true_mean <= high:
        colors.append("blue")
    else:
        colors.append("red")

# 3. Create the scatter plot with error bars
plt.figure(figsize=(10, 6))

# Plot each point individually to maintain specific colors for error bars
for i in range(len(gen_means)):
    plt.errorbar(
        x[i],
        gen_means[i],
        yerr=[[yerr_low[i]], [yerr_high[i]]],
        fmt="o",
        color=colors[i],
        capsize=5,
        markersize=8,
        elinewidth=2,
    )

# Add the true mean horizontal line
plt.axhline(
    true_mean,
    color="green",
    linestyle="--",
    linewidth=2,
    label=f"True Mean ({true_mean})",
)

# 4. Formatting and Custom Legend
legend_elements = [
    Line2D([0], [0], color="blue", lw=2, label="CI contains True Mean", marker="o"),
    Line2D([0], [0], color="red", lw=2, label="CI misses True Mean", marker="o"),
    Line2D([0], [0], color="green", lw=2, linestyle="--", label="True Mean"),
]

ax = plt.gca()  # Get current axes

# Set the major locator to 10
ax.xaxis.set_major_locator(MultipleLocator(10))

# Optional: Add minor ticks for every 1 to keep the scale clear
ax.xaxis.set_minor_locator(MultipleLocator(1))

plt.title("Scatter Plot of Means with Confidence Intervals")
plt.xlabel("Sample Index")
plt.ylabel("Value")
plt.legend(handles=legend_elements)
plt.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()


##### Problem 4 #####


def draw(n):
    reals = np.random.uniform(-1, 1, n)
    x_bar = np.mean(reals)

    return x_bar


runs = [1, 10, 100, 1000, 10000]
cdfs = []
for j in range(len(runs)):
    means = []
    for jj in range(1000):
        means.append(draw(runs[j]))

    cdfs.append(stats.ecdf(means))


# Plot the cdfs of the means (Used Gemini)
fig, ax = plt.subplots(figsize=(10, 6))

for i, res in enumerate(cdfs):
    # .cdf is the EmpiricalDistributionFunction object
    # we pass the existing 'ax' to keep them on one plot
    res.cdf.plot(ax, label=f"{runs[i]} Samples")

ax.plot([-1, 0, 0, 1], [0, 0, 1, 1], label="Theoretical CDF")

# Formatting the plot
ax.set_title("Multiple Empirical CDFs")
ax.set_xlabel("Value")
ax.set_ylabel("Cumulative Probability")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")  # Legend outside if list is long
ax.grid(True, linestyle=":", alpha=0.7)

plt.tight_layout()
plt.show()
