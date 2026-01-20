import numpy as np


# Problem 2
means = []
vars = []
sizes = [1, 2, 3, 4, 5, 6]
for n in sizes:
    # Draw Samples
    samples = np.random.normal(loc=0, scale=2, size=10**n)
    trans = np.sin(samples)
    
    # Compute mean and variance
    mean = np.mean(trans)
    var = np.var(trans)

    # Save the results
    means.append(mean)
    vars.append(var)

print("The means are: ", means)
print("The variances are: ", vars)