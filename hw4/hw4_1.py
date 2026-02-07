import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(12)

# Set up the problem
t = np.arange(start=0, stop=10.01, step=0.05)

X = np.zeros((len(t), 2))
X[:, 0] = np.ones(len(t))
X[:, 1] = np.exp(t)

q = np.transpose(np.array([10, 1]))

eps = np.random.uniform(-1, 1, len(t))

ups = np.matmul(X, q) + eps

# Estimate qois 
q_hat_ls = la.solve(X.T@X, X.T@ups)

q_var = (1/ (len(t) - 1)) * (ups - X@q_hat_ls).T@(ups - X@q_hat_ls)

xtxinv = la.inv(X.T@X)
xtxinv_diag = np.diag(xtxinv)

q1_ci_lo, q1_ci_hi = stats.t.interval(0.95, len(t), loc=q_hat_ls[0], scale=np.sqrt(q_var*xtxinv_diag[0]))
q2_ci_lo, q2_ci_hi = stats.t.interval(0.95, len(t), loc=q_hat_ls[1], scale=np.sqrt(q_var*xtxinv_diag[1]))

print(q_hat_ls)
print(q_var)
print(q1_ci_lo, q1_ci_hi)
print(q2_ci_lo, q2_ci_hi)

num_mc = 1000

eps_big = np.random.uniform(-1, 1, len(t)*num_mc)
eps_big = eps_big.reshape((len(t), num_mc))

ups_big = np.zeros((len(t), num_mc))
ci_big = np.zeros((num_mc, 4))
q_hat_big = np.zeros((num_mc, 2))

xtxinvxt = la.inv(X.T@X)@X.T

num_contains_q1 = 0
num_contains_q2 = 0

for i in range(num_mc):

    eps_i = eps_big[:, i]
    ups_i = X@q + eps_i

    ups_big[:, i] = ups_i

    q_hat_i = xtxinvxt@ups_i

    q_hat_big[i, :] = q_hat_i
    q_var_i = (1/ (len(t) - 1)) * (ups_i - X@q_hat_i).T@(ups_i - X@q_hat_i)

    q1_ci_lo_i, q1_ci_hi_i = stats.t.interval(0.95, len(t), loc=q_hat_i[0], scale=np.sqrt(q_var_i*xtxinv_diag[0]))
    q2_ci_lo_i, q2_ci_hi_i = stats.t.interval(0.95, len(t), loc=q_hat_i[1], scale=np.sqrt(q_var_i*xtxinv_diag[1]))

    ci_big[i, :] = [q1_ci_lo_i, q1_ci_hi_i, q2_ci_lo_i, q2_ci_hi_i]

    if (q1_ci_hi_i > q[0]) and (q1_ci_lo_i < q[0]):
        num_contains_q1 += 1

    if (q2_ci_hi_i > q[1]) and (q2_ci_lo_i < q[1]):
        num_contains_q2 += 1

print(f'% of q1 intervals which bounded the true solution: {num_contains_q1/num_mc:.2%}')
print(f'% of q2 intervals which bounded the true solution: {num_contains_q2/num_mc:.2%}')


plt.hist(eps)
plt.title("Histogram of uniform noise")
plt.xlabel("Random Noise")
plt.ylabel("Frequency")
plt.show()
