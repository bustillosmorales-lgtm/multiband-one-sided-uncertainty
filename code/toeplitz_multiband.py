"""Toeplitz analogue of the numerical section: smallest eigenvalue of principal submatrices on multiband index sets.

Local model: spectrum in K subset [0,1]. Discrete analogue: principal submatrix of
T_n(xi) = I/2 + (i/2pi) H_n on the indices k with k/n in K.  With S = pi^2 n lambda_min,
the interval K = [0,1] gives S ~ l = log n.  The local model suggests S_K ~ l / kappa(K), with

    kappa(K) = 4 max_{x in K} a_K(x),  1/a_K = z'(x) = sum_i 1/(x - a_i) - 1/(x - b_i)

(Casini--Huerta variable z = log(-prod (x-a_i)/(x-b_i)); kappa([0,1]) = 1).
Test: slope dS_K/dl against dS_1/dl, and S_K/S_1, for n = 2^8..2^16.
Run from code/:  python toeplitz_multiband.py 16 > ../results/toeplitz_multiband.txt   (about 10 minutes)
"""
import sys
import time
import numpy as np
from scipy.signal import fftconvolve
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.optimize import minimize_scalar

PI = np.pi


def kappa_K(K):
    best = 0.0
    for (a, b) in K:
        def neg_a(x):
            zp = sum(1 / (x - ai) - 1 / (x - bi) for (ai, bi) in K)
            return -1 / zp
        r = minimize_scalar(neg_a, bounds=(a + 1e-9, b - 1e-9), method="bounded", options={"xatol": 1e-12})
        best = max(best, -r.fun)
    return 4 * best


def norm_sub(n, K, tol=1e-14):
    k = np.arange(n)
    mask = np.zeros(n, bool)
    for (a, b) in K:
        mask |= (k / n >= a) & (k / n < b)
    idx = np.nonzero(mask)[0]
    m = np.arange(-(n - 1), n)
    with np.errstate(divide="ignore"):
        ker = np.where(m == 0, 0.0, 1.0 / np.where(m == 0, 1, m))

    def mv(x):
        full = np.zeros(n, complex)
        full[idx] = np.ravel(x)
        return 1j * fftconvolve(full, ker, mode="full")[n - 1:2 * n - 1][idx]

    op = LinearOperator((len(idx), len(idx)), matvec=mv, dtype=complex)
    v0 = np.exp(-1j * PI / 2 * idx) * np.sin(PI * (idx + 0.5) / n)
    return eigsh(op, k=1, which="LA", tol=tol, v0=v0, ncv=60, maxiter=200000, return_eigenvectors=False)[0]


SETS = {
    "[0,1]": [(0.0, 1.0)],
    "[0,.4]u[.6,1]": [(0.0, 0.4), (0.6, 1.0)],
    "[0,.45]u[.55,1]": [(0.0, 0.45), (0.55, 1.0)],
    "[0,.7]u[.8,1]": [(0.0, 0.7), (0.8, 1.0)],
    "[0,.2]u[.3,1]": [(0.0, 0.2), (0.3, 1.0)],
    "[0,.6]": [(0.0, 0.6)],
}

if __name__ == "__main__":
    kmax = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    ns = [2 ** k for k in range(8, kmax + 1)]
    S = {}
    for name, K in SETS.items():
        kap = kappa_K(K)
        lmax = max(b - a for a, b in K)
        print(f"\n{name}: kappa_K = {kap:.5f}   L_max = {lmax:.3f}   1/kappa = {1/kap:.5f}", flush=True)
        S[name] = []
        for n in ns:
            t0 = time.time()
            gap = PI - norm_sub(n, K)
            s = (PI / 2) * n * gap
            S[name].append(s)
            print(f"  n={n:6d}  S={s:10.5f}  S/l={s/np.log(n):8.5f}  ({time.time()-t0:.1f}s)", flush=True)
    ell = np.log(ns)
    print("\nratios against [0,1]  (prediction for large n: 1/kappa_K)")
    s1 = np.array(S["[0,1]"])
    for name, K in SETS.items():
        sk = np.array(S[name])
        slope_ratio = np.diff(sk) / np.diff(s1)
        print(f"{name:>16}: 1/kappa={1/kappa_K(K):.4f}  S_K/S_1 last={sk[-1]/s1[-1]:.4f}  "
              f"slope ratios last 3 = {np.round(slope_ratio[-3:], 4)}")
