"""Orthogonality of the fibre matrix W(zeta) of Section 2 (Lemma 2.2).
W_{km}(z) = Theta_m q_k(x_m) / (N_k sqrt(|prod(x_m-a)(x_m-b)|) sqrt(z'(x_m))),  q_k = prod_{i!=k}(x-a_i),
x_m(z) the preimage of z in component m.  Claim: W(z) orthogonal for all z (=> the CH transform is unitary).
Compares the normalizations N_k of Casini--Huerta (37) and Hollands (73); the article uses the latter.
Run from code/:  python fibre_orthogonality.py > ../results/fibre_orthogonality.txt"""
import numpy as np
def setup(K):
    A = np.array([a for a, b in K]); B = np.array([b for a, b in K]); n = len(K)
    z = lambda x: np.log(np.abs(np.prod(x - A) / np.prod(x - B)))
    zp = lambda x: np.sum(1 / (x - A) - 1 / (x - B))
    def pre(zz, m):
        lo, hi = A[m] + 1e-15, B[m] - 1e-15
        for _ in range(200):
            mid = (lo + hi) / 2; lo, hi = (mid, hi) if z(mid) < zz else (lo, mid)
        return (lo + hi) / 2
    NH = [np.sqrt(-np.prod([A[k] - A[i] for i in range(n) if i != k]) / np.prod(A[k] - B)) for k in range(n)]
    def nch(k):
        s1 = sum(np.prod([B[j] - A[l] for l in range(n) if l != k]) / ((B[j] - A[k]) * np.prod([B[j] - B[l] for l in range(n) if l != j])) for j in range(n))
        s2 = np.prod([A[k] - A[j] for j in range(n) if j != k]) / np.prod([A[k] - B[j] for j in range(n)])
        return np.sqrt(np.pi * (s1 - s2))
    NC = [nch(k) for k in range(n)]
    def W(zz, N):
        M = np.zeros((n, n))
        for m in range(n):
            x = pre(zz, m)
            for k in range(n):
                q = np.prod([x - A[i] for i in range(n) if i != k])
                M[k, m] = (-1) ** m * q / (N[k] * np.sqrt(abs(np.prod(x - A) * np.prod(x - B))) * np.sqrt(zp(x)))
        return M
    return W, NH, NC
for K in ([(0, .4), (.6, 1)], [(0, .2), (.35, .5), (.8, 1.3)], [(-2, -1), (0, .1), (.5, 3), (4, 4.2)]):
    W, NH, NC = setup(K)
    for label, N, scale in (("Hollands (73)", NH, 1.0), ("CH (37)/sqrt(pi)... raw", NC, 1.0)):
        errs = []
        for zz in np.linspace(-8, 8, 33):
            M = W(zz, N); G = M @ M.T
            errs.append(np.abs(G / G[0, 0] - np.eye(len(K))).max())
        M0 = W(0.3, N); G0 = M0 @ M0.T
        print(f"n={len(K)} {label:>24}: max |WW^T/c - I| = {max(errs):.1e}   c = WW^T[0,0] = {G0[0,0]:.6f}")
