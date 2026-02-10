from utils import HeatEquation
import numpy as np

eq = HeatEquation()

print(eq.params)

x_vals = np.array([0.1 + k * 0.04 for k in range(15)])

q = {
    "phi": -100000,
    "h": 10,
}

print("analytic: \n", eq.Ts_jac(x_vals, ["phi", "h"], q))

print("numerical: \n", eq.Ts_numder(x_vals, ["phi", "h"], q, 1e-6))
