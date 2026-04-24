import numpy as np

A = np.array(
    [
        [1, 2, 3],
        [2, 3, 5],
        [4, 5, 9],
        [5, 6, 11],
    ]
)

U, S, V = np.linalg.svd(A)
print(S)
