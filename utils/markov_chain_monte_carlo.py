import numpy as np
import scipy
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf


class MCMC:
    def __init__(
        self,
        q0=None,
        J_func=None,
        r_calc=None,
        post=None,
        s=None,
        D=None,
        M=1_000,
        rng_seed=42,
    ):
        self.q0 = q0
        self.J_func = J_func
        self.r_calc = r_calc
        self.post = post
        self.s = s
        self.D = D
        self.M = M
        self.rng_seed = rng_seed

        np.random.seed(self.rng_seed)

    def metropolis_hastings(
        self,
        # Flags
        adaptive=False,
        gibbs_step=False,
        delayed_rejection=False,
        save_output=False,
        # Adaptive params
        k0=None,
        sp=None,
        eps=None,
        V0=None,
        # Gibbs params
        ns=None,
        n_meas=None,
        cf=None,
        # Delayed rejection params
        gamma2=None,
        # Save
        filename=None,
    ):
        # Perform checks on arguments
        if adaptive:
            if any(p is None for p in [k0, sp, eps, V0]):
                raise ValueError("All adaptive parameters must be provided.")
        if gibbs_step:
            if any(p is None for p in [ns, n_meas, cf]):
                raise ValueError("All Gibbs parameters must be provided.")
        if save_output:
            if filename is None:
                raise ValueError("Please provide a file name.")

        # Set up
        qk = self.q0
        sk = self.s
        if adaptive:
            Vk = V0
        else:
            Vk = self.D

        n = len(qk)

        # Initialize outputs
        q_hist = np.zeros((len(qk), self.M))
        post_hist = np.zeros(self.M)
        s_hist = np.zeros(self.M)
        # Store initial values
        q_hist[:, 0] = qk
        post_hist[0] = 0
        s_hist[0] = sk

        # Init acceptance counter
        acc = 1

        for i in range(1, self.M):
            # Adapt (or don't)
            if adaptive:
                if (i >= k0) and (i % k0 == 0):
                    Vk = sp * np.cov(q_hist[:, :i]) + eps * np.eye(n)

            # Generate a new point
            rejected = False
            Vin = Vk
            while True:
                if not rejected:
                    q_star = self.J_func(
                        qk, Vin
                    )  # Should make these moments kwargs since it may not always be gaussian
                else:
                    q_star = np.random.multivariate_normal(qk, Vin)

                # Calculate the acceptance ratio
                r = self.r_calc(q_star, qk, Vin, sk)

                # Accept according to acceptance ratio
                u = np.random.uniform(0, 1)
                alpha = min(1, r)
                if u <= alpha:  # Accept
                    qk = q_star
                    q_hist[:, i] = qk
                    acc += 1
                    # print("Accepted")
                    break
                else:  # Failed r_calc
                    # print("Failed r_calc!!")
                    if delayed_rejection and not rejected:
                        Vin = gamma2**2 * Vk
                        rejected = True
                        # print("Decreased the covariance!")
                        continue
                    else:
                        q_hist[:, i] = qk
                        break

            if gibbs_step:
                a_val = (n_meas + ns) / 2
                b_val = (ns * self.s + cf(qk)) / 2

                sk = 1 / np.random.gamma(shape=a_val, scale=1 / b_val)
                s_hist[i] = sk

                # Evaluate the posterior using the sigma prior
                p = (
                    self.post(qk, sk)
                    * sk ** (-ns / 2 + 1)
                    * np.exp((self.s * ns / 2) / sk)
                )
            else:
                # Can evaluate posterior without sigma prior (would be nice to make this a log situation)
                p = self.post(qk, sk)
            post_hist[i] = p

        if save_output:
            print(f"Saving output to {filename}")
            np.savez(
                filename,
                q_hist=q_hist,
                post_hist=post_hist,
                s_hist=s_hist,
                accr=acc / self.M,
            )

        return (q_hist, post_hist, s_hist, acc / self.M)


def plot_mcmc_2d(filepath):

    result = np.load(filepath)

    q_hist = result["q_hist"]
    post_hist = result["post_hist"]
    s_hist = result["s_hist"]
    accr = result["accr"]

    MAPidx = np.argmax(post_hist[100:])
    MAP = np.array([q_hist[0, MAPidx], q_hist[1, MAPidx], s_hist[MAPidx]])
    gibbs_est = np.mean(s_hist[-10:])

    print("RESULTS: ")
    print(f"The acceptance ratio was {accr}")
    print(f"The MAP estimate was {MAP}")
    print(f"The Gibbs estimate of the variance was {gibbs_est}")

    # Chain plot
    fig, ax = plt.subplots()
    ax.plot(q_hist[0, :], q_hist[1, :], color="b", marker="x")
    ax.set_xlabel(r"$q_1$")
    ax.set_ylabel(r"$q_2$")
    ax.set_title(f"MCMC Path")
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    # Marginal Histograms
    kde_q1x = np.linspace(min(q_hist[0, :]), max(q_hist[0, :]), 1000)
    kde_q2x = np.linspace(min(q_hist[1, :]), max(q_hist[1, :]), 1000)
    kde_q1 = scipy.stats.gaussian_kde(q_hist[0, :])
    kde_q2 = scipy.stats.gaussian_kde(q_hist[1, :])

    fig, axes = plt.subplots(2, 1, constrained_layout=True)
    axes[0].plot(kde_q1x, kde_q1(kde_q1x), label="KDE")
    axes[0].hist(q_hist[0, :], density=True, bins=100, label="Histogram")
    axes[0].set_xlabel(r"$q_1$")
    axes[0].ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    axes[0].legend()

    axes[1].plot(kde_q2x, kde_q2(kde_q2x), label="KDE")
    axes[1].hist(q_hist[1, :], density=True, bins=100, label="Histogram")
    axes[1].set_xlabel(r"$q_2$")
    axes[1].ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    axes[1].legend()
    fig.suptitle("Estimated marginal densities")

    # Distribution of s
    kde_sx = np.linspace(0, max(s_hist), 500)
    kde_s = scipy.stats.gaussian_kde(s_hist)

    fig, ax = plt.subplots()
    ax.hist(s_hist[100:], density=True, bins=100, label="Histogram")
    ax.set_xlabel(r"$\sigma_0^2$")
    ax.plot(kde_sx, kde_s(kde_sx))
    ax.set_title(r"Distribution of $\sigma^2_0$")
    ax.legend()

    # Trace plots
    fig, axes = plt.subplots(3, 1, constrained_layout=True)
    axes[0].plot(q_hist[0, :])
    axes[0].set_xlabel(r"$q_1$")
    # axes[0].set_xscale("log")
    axes[1].plot(q_hist[1, :])
    axes[1].set_xlabel(r"$q_2$")
    # axes[1].set_xscale("log")
    axes[2].plot(s_hist)
    axes[2].set_xlabel(r"$\sigma_0^2$")
    # axes[2].set_xscale("log")

    fig.suptitle("Trace Plots")

    # Correleograms
    fig, axes = plt.subplots(3, 1, constrained_layout=True)
    plot_acf(
        q_hist[0, :],
        ax=axes[0],
        alpha=None,
        lags=range(1000),
        use_vlines=False,
        marker=None,
        linestyle="-",
        title=None,
    )
    axes[0].hlines(0, 0, 1000, color="k", linestyle=":")
    axes[0].set_xlabel(r"$q_1$")
    plot_acf(
        q_hist[1, :],
        ax=axes[1],
        alpha=None,
        lags=range(1000),
        use_vlines=False,
        marker=None,
        linestyle="-",
        title=None,
    )
    axes[1].hlines(0, 0, 1000, color="k", linestyle=":")
    axes[1].set_xlabel(r"$q_2$")
    plot_acf(
        s_hist,
        ax=axes[2],
        alpha=None,
        lags=range(1000),
        use_vlines=False,
        marker=None,
        linestyle="-",
        title=None,
    )
    axes[2].hlines(0, 0, 1000, color="k", linestyle=":")
    axes[2].set_xlabel(r"$\sigma_0^2$")
    fig.suptitle("Autocorrelation")

    plt.show()
