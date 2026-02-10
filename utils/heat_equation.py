import numpy as np
from sympy import symbols, diff, sin, exp, sqrt, lambdify


class HeatEquation:
    def __init__(
        self,
        a=0.0095,
        b=0.0095,
        h=20,
        k=401,
        x=0.1,
        Tamb=21.29,
        phi=180000,
        L=0.7,
    ):  # Default values for copper

        self.params = {  # Important that these are ordered the same as syms
            "a": a,  # m
            "b": b,  # m
            "h": h,  # W / (m2 * C)
            "k": k,  # W / (m * C)
            "x": x,  # m
            "Tamb": Tamb,  # deg C
            "phi": phi,  # W / m2
            "L": L,  # m
        }

        # Set up the symbols we are going to need
        self.syms = symbols("a b h k x Tamb phi L")
        a_s, b_s, h_s, k_s, x_s, Tamb_s, phi_s, L_s = self.syms

        # Build the heat equation a little at a time
        gamma = sqrt((2 * (a_s + b_s) * h_s) / (a_s * b_s * k_s))
        top = exp(gamma * L_s) * (h_s + k_s * gamma)
        bottom = exp(-gamma * L_s) * (h_s - k_s * gamma) + exp(gamma * L_s) * (
            h_s + k_s * gamma
        )
        c1 = -(phi_s / (k_s * gamma)) * (top / bottom)
        c2 = (phi_s / (k_s * gamma)) + c1

        # The final symbolic expressions
        self.Ts = c1 * exp(-gamma * x_s) + c2 * exp(gamma * x_s) + Tamb_s
        self.derivatives = {
            "h": diff(self.Ts, h_s),
            "k": diff(self.Ts, k_s),
            "phi": diff(self.Ts, phi_s),
            "x": diff(self.Ts, x_s),
        }

    # Numeric evaluations
    def Ts_vals(self, x_vals: np.ndarray, q: dict = {}):
        """If q is empty, evaluate with default parameters at x_vals
        Otherwise q overrides any default value"""
        Ts_eval = lambdify(self.syms, self.Ts, "numpy")

        args = []
        for key in self.params.keys():
            if key in q.keys():
                args.append(q[key])
            elif key == "x":
                args.append(x_vals)
            else:
                args.append(self.params[key])

        return Ts_eval(*args)

    def Ts_jac(self, x_vals, vars: list[str], q: dict = {}):
        """
        Analytical derivatives of Ts with respect to the variables in q, evaluated at x_vals

        """
        args = []
        for key in self.params.keys():
            if key in q.keys():
                args.append(q[key])
            elif key == "x":
                args.append(x_vals)
            else:
                args.append(self.params[key])

        jac = np.zeros((len(x_vals), len(vars)))
        for i, var in enumerate(vars):
            der = self.derivatives[var]
            der_eval = lambdify(self.syms, der, "numpy")
            jac[:, i] = der_eval(*args)

        return jac

    def Ts_numder(self, x_vals, vars: list[str], q: dict = {}, delta=1e-6):
        """Numerical derivative with step size delta"""

        der = np.zeros((len(x_vals), len(vars)))
        for i, var in enumerate(vars):
            fv = self.Ts_vals(x_vals, q)
            if var in q.keys():
                q[var] += delta
            else:
                q[var] = self.params[var] + delta
            dfv = self.Ts_vals(x_vals, q)
            der[:, i] = (dfv - fv) / delta

        return der
