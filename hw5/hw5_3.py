import numpy as np

P = np.array(
    [
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0.5, 0, 0, 0.5, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0],
    ]
)

p01 = np.array([1 / 6, 1 / 6, 1 / 3, 1 / 6, 1 / 6])  # show stationary

p02 = np.array([0.5, 0, 0, 0.5, 0])  # show period k=3
