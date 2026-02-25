import numpy as np
from scipy.stats import invgamma


class MCMC:
    def __init__(self, q0, J_func, r_calc, s, D, M=1_000, rng_seed=42):
        self.q0 = q0
        self.J_func = J_func
        self.r_calc = r_calc
        self.s = s
        self.D = D
        self.M = M
        self.rng_seed = rng_seed

        self.n = len(q0)

        np.random.seed(self.rng_seed)

    def metropolis_hastings(self, adaptive=False, gibbs_step=False, ns=0, cf=None):
        qk = self.q0
        sk = self.s
        q_hist = np.zeros((len(qk), self.M))
        post_hist = np.zeros(self.M)
        s_hist = np.zeros(self.M)
        q_hist[:, 0] = qk
        post_hist[0] = 0
        s_hist[0] = sk

        acc = 1

        for i in range(1, self.M):

            # Generate a new point
            q_star = self.J_func(qk, self.D)

            # Calculate the acceptance ratio
            ratio = self.r_calc(q_star, qk, self.D, sk)
            if len(ratio) > 1:
                r = ratio[0]
                post = ratio[1]
            else:
                r = ratio

            # Accept according to acceptance ratio
            u = np.random.uniform(0, 1)
            alpha = min(1, r)
            if u <= alpha:
                qk = q_star
                q_hist[:, i] = qk
                acc += 1
                if len(ratio) > 1:
                    post_hist[i] = post
            else:
                q_hist[:, i] = qk
                post_hist[i] = post_hist[i - 1]

            if gibbs_step:
                a_val = (self.n + ns) / 2
                b_val = (ns * sk + cf(qk)) / 2

                sk = invgamma.rvs(a=a_val, scale=b_val)
                s_hist[i] = sk

        return (q_hist, post_hist, s_hist, acc / self.M)

    def adaptive_metropolis(self, k0, sp, eps, V0):
        # If I was good, I would set things up so this could call metropolis hastings
        qk = self.q0
        Vk = V0
        q_hist = np.zeros((len(qk), self.M))
        post_hist = np.zeros(self.M)
        q_hist[:, 0] = qk
        post_hist[0] = 0

        acc = 1
        fh = False
        for i in range(1, self.M):
            if (i >= k0) and (i % k0 == 0):
                Vk = sp * np.cov(q_hist[:, :i]) + eps * np.eye(len(qk))
                if fh == False:
                    V1 = Vk
                    fh = True
            # Generate a new point
            q_star = self.J_func(qk, Vk)

            # Calculate the acceptance ratio
            ratio = self.r_calc(q_star, qk, Vk, self.s)
            if len(ratio) > 1:
                r = ratio[0]
                post = ratio[1]
            else:
                r = ratio

            # Accept according to acceptance ratio
            u = np.random.uniform(0, 1)
            alpha = min(1, r)
            if u <= alpha:
                qk = q_star
                q_hist[:, i] = qk
                acc += 1
                if len(ratio) > 1:
                    post_hist[i] = post
            else:
                q_hist[:, i] = qk
                post_hist[i] = post_hist[i - 1]

            if self.gibbs == True:
                pass

        return (q_hist, post_hist, acc / self.M, (V1, Vk))

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
