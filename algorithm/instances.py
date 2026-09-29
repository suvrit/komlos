import numpy as np

def colnormalize(A):
    nrm = np.linalg.norm(A, axis=0)
    nrm[nrm == 0] = 1
    return A / nrm

def gauss(n, m, rng):
    return colnormalize(rng.standard_normal((m, n)))

def rademacher(n, m, rng):
    return rng.choice([-1.0, 1.0], size=(m, n)) / np.sqrt(m)

def sparse_bf(n, m, rng, k=8):
    A = np.zeros((m, n))
    for j in range(n):
        rows = rng.choice(m, size=k, replace=False)
        A[rows, j] = rng.choice([-1.0, 1.0], size=k) / np.sqrt(k)
    return A

def hadamard(n, m, rng):
    H = np.array([[1.0]])
    while H.shape[0] < max(n, m):
        H = np.block([[H, H], [H, -H]])
    H = H[:m, :n]
    return H / np.sqrt(m)

def prefix(n, m, rng):
    # a_ij = 1/sqrt(m) for i >= j (lower triangular ones), random column signs
    A = np.tril(np.ones((m, n))) / np.sqrt(m)
    return A * rng.choice([-1.0, 1.0], size=n)[None, :]

def heavy_rows(n, m, rng):
    # a few very heavy rows plus a Gaussian bulk: exercises the large-row class
    A = rng.standard_normal((m, n))
    A[: m // 8] *= 6.0
    return colnormalize(A)

FAMILIES = dict(gauss=gauss, rademacher=rademacher, sparse_bf=sparse_bf,
                hadamard=hadamard, prefix=prefix, heavy_rows=heavy_rows)
