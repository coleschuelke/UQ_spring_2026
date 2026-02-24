import numpy as np


class MCMC:
    def __init__(self, q0, J_func, r_calc, D, M=1_000, rng_seed=42):
        self.q0 = q0
        self.J_func = J_func
        self.r_calc = r_calc
        self.D = D
        self.M = M
        self.rng_seed = rng_seed

        np.random.seed(self.rng_seed)

    def metropolis_hastings(self):
        qk = self.q0
        q_hist = np.zeros((len(qk), self.M))
        q_hist[:, 0] = qk

        acc = 1

        for i in range(1, self.M):

            # Generate a new point
            q_star = self.J_func(qk, self.D)

            # Calculate the acceptance ratio
            r = self.r_calc(q_star, qk, self.D)

            # Accept according to acceptance ratio
            u = np.random.uniform(0, 1)
            alpha = min(1, r)
            if u <= alpha:
                qk = q_star
                q_hist[:, i] = qk
                acc += 1
            else:
                q_hist[:, i] = qk

        return (q_hist, acc / self.M)

    def adaptive_metropolis(self, k0, sp, eps, V0):
        # If I was good, I would set things up so this could call metropolis hastings
        qk = self.q0
        Vk = V0
        q_hist = np.zeros((len(qk), self.M))
        q_hist[:, 0] = qk

        acc = 1

        for i in range(1, self.M):
            if (i >= k0) and (i % k0 == 0):
                Vk = sp * np.cov(q_hist[:, :i]) + eps * np.eye(len(qk))
            # Generate a new point
            q_star = self.J_func(qk, Vk)

            # Calculate the acceptance ratio
            r = self.r_calc(q_star, qk, Vk)

            # Accept according to acceptance ratio
            u = np.random.uniform(0, 1)
            alpha = min(1, r)
            if u <= alpha:
                qk = q_star
                q_hist[:, i] = qk
                acc += 1
            else:
                q_hist[:, i] = qk

        return (q_hist, acc / self.M, Vk)

    def gibbs(self):
        raise NotImplementedError
        qk = self.q0
        p = len(qk)
        q_hist = np.zeros((p, self.M))

        for i in range(self.M):
            q_star = qk
            for j in range(p):
                x = 0  # TODO: get marginal based on all other elements
                q_star[j] = x
            qk = q_star
            q_hist[:, i] = qk

        return q_hist
