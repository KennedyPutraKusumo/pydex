from scipy.optimize import minimize
import numpy as np
# np.random.seed(123)


def simulate(x, theta=1):
    x1, x2 = x.T
    return np.array(
        [x1 * np.exp(-theta * x2), 1 - x1 * np.exp(-theta * x2)]
    ).T


def lsq(theta, candidates, data):
    pred = simulate(candidates, theta)
    error = data - pred
    error = error.flatten()
    return error @ error


candidates1 = np.array([
    [-1, -1],
    [-1,  0],
    [-1,  1],
    [ 0, -1],
    [ 0,  0],
    [ 0,  1],
    [ 1, -1],
    [ 1,  0],
    [ 1,  1],
])
noiseless1 = simulate(candidates1)
data1 = noiseless1 + np.random.normal(0, scale=0.2, size=noiseless1.shape)
candidates2 = np.array([
    [ 0,  0],
    [ 0,  0],
    [-1, -1],
    [-1,  1],
    [ 1, -1],
    [ 1,  1],
])
noiseless2 = simulate(candidates2)
data2 = noiseless2 + np.random.normal(0, scale=0.2, size=noiseless2.shape)

if __name__ == '__main__':
    opt_result = minimize(
        fun=lsq,
        x0=-2,
        args=(candidates1, data1),
        method="l-bfgs-b",
        tol=1e-10,
        options={"disp": True},
    )
    print(opt_result.x)
