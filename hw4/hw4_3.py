import scipy
import matplotlib.pylab as plt
import numpy as np
import pandas as pd


data = scipy.io.loadmat('exercise_7p7_data')

data = data['copper_data']

df = pd.DataFrame(data)
print(df.head())