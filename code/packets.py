"""Packets of the sharpness section (Lemma 5.1) computed in the variable zeta (the numerical section).

K = (0,0.4) u (0.6,1).  phi = V^{-1} W^T psi, psi = g(zeta) e^{-i s0 zeta} v, v = W(zeta0) e_l.
Quadrature in zeta (uniform step h), which resolves the packet near the endpoints, unlike the
logistic grid in x used by leak_setup.py.  Prints, for each packet:
  * the leak from F (t in [-T,0]) against int m dnu for the exact Gaussian spectral measure;
  * the first moment int t|F|^2 computed from F (t in [-T,T]) against (s0/2pi) sum_j int z' A_j^2;
  * the delta scan of 4 / sum_j int z' A_j^2 and of the mass outside the component of x*;
  * the companion test (companion removed) at s0 = 4, delta = 1.
Run from code/:  python packets.py > ../results/packets.txt   (about 20 minutes)
"""
import numpy as np
from scipy.optimize import minimize_scalar

K = [(0.0, 0.4), (0.6, 1.0)]
A = np.array([a for a, b in K]); B = np.array([b for a, b in K]); n = len(K)
zp = lambda x: sum(1 / (x - a) - 1 / (x - b) for a, b in K)
z = lambda x: np.log(-np.prod([x - a for a in A], 0) / np.prod([x - b for b in B], 0))
xstar = minimize_scalar(zp, bounds=(1e-9, 0.4 - 1e-9), method="bounded", options={"xatol": 1e-14}).x
z0 = z(np.array([xstar]))[0]
Nk = np.sqrt([-np.prod([A[k] - A[i] for i in range(n) if i != k]) / np.prod(A[k] - B) for k in range(n)])


def Wcol(x):
    Pa = np.prod([x - a for a in A], 0); Pb = np.prod([x - b for b in B], 0)
    return np.array([np.sign(Pb) * np.prod([x - A[i] for i in range(n) if i != k], 0)
                     / (Nk[k] * np.sqrt(np.abs(Pa * Pb) * zp(x))) for k in range(n)])


v = Wcol(np.array([xstar]))[:, 0]


def xinv(zeta, a, b):
    lo = np.full_like(zeta, a); hi = np.full_like(zeta, b)
    for _ in range(80):
        mid = (lo + hi) / 2; f = z(mid) < zeta
        lo = np.where(f, mid, lo); hi = np.where(f, hi, mid)
    return (lo + hi) / 2


def packet(s0, d, h=0.002, keep=(0, 1)):
    zeta = np.arange(z0 - 15 * d, z0 + 15 * d, h)
    g = (2 * np.pi * d * d) ** -0.25 * np.exp(-(zeta - z0) ** 2 / (4 * d * d))
    comps = []
    for j, (a, b) in enumerate(K):
        x = xinv(zeta, a, b); Aj = (v @ Wcol(x)) * g
        if j not in keep:
            Aj = 0 * Aj
        comps.append((x, Aj))
    return zeta, h, comps


def F_of(t, s0, zeta, h, comps):
    F = np.zeros_like(t, complex)
    for x, Aj in comps:
        w = Aj * np.exp(-1j * s0 * zeta) * np.sqrt(1 / zp(x)) * h
        for i in range(0, len(t), 500):
            F[i:i + 500] += np.exp(2j * np.pi * np.outer(t[i:i + 500], x)) @ w
    return F


def exact_leak(s0, d):
    s = np.linspace(s0 - 12 / d, s0 + 12 / d, 400001); ds = s[1] - s[0]
    nu = np.sqrt(2 * d * d / np.pi) * np.exp(-2 * d * d * (s - s0) ** 2)
    return np.sum(nu / (1 + np.exp(np.clip(2 * np.pi * s, -700, 700)))) * ds


T, dt = 150.0, 0.01
tneg = np.arange(-T, 0, dt) + dt / 2
tall = np.arange(-T, T, dt) + dt / 2
print(f"x* = {xstar:.6f}, z0 = {z0:.6f}, kappa(K) = {4 / zp(xstar):.6f}")
print("\nleak and first moment of the packets (delta = 1)")
for s0 in (1, 2, 3, 4, 5):
    zeta, h, comps = packet(s0, 1.0)
    nrm = sum(np.sum(Aj ** 2) * h for x, Aj in comps)
    mu = np.sum(np.abs(F_of(tneg, s0, zeta, h, comps)) ** 2) * dt / nrm
    ex = exact_leak(s0, 1.0)
    P = np.abs(F_of(tall, s0, zeta, h, comps)) ** 2 / nrm
    m1_F = np.sum(tall * P) * dt
    m1_pred = s0 / (2 * np.pi) * sum(np.sum(zp(x) * Aj ** 2) * h for x, Aj in comps) / nrm
    print(f"  s0={s0}: leak {mu:.5e}  exact {ex:.5e}  rel.diff {mu / ex - 1:+.2e} | "
          f"first moment from F {m1_F:.6f}  predicted {m1_pred:.6f}")
print("\ndelta scan: 4 / sum_j int z' A_j^2 and mass outside the component of x*")
for d in (0.5, 0.25, 0.1, 0.05):
    zeta, h, comps = packet(1.0, d, h=d / 400)
    nrm = sum(np.sum(Aj ** 2) * h for x, Aj in comps)
    zbar = sum(np.sum(zp(x) * Aj ** 2) * h for x, Aj in comps) / nrm
    out = np.sum(comps[1][1] ** 2) * h / nrm
    print(f"  delta={d}: 4/<z'> = {4 / zbar:.5f}  mass outside = {out:.3e}")
print("\ncompanion test, s0 = 4, delta = 1")
res = []
for keep in ((0, 1), (0,)):
    zeta, h, comps = packet(4, 1.0, keep=keep)
    nrm = sum(np.sum(Aj ** 2) * h for x, Aj in comps)
    res.append(np.sum(np.abs(F_of(tneg, 4, zeta, h, comps)) ** 2) * dt / nrm)
print(f"  with companion {res[0]:.4e}, without {res[1]:.4e}, ratio {res[1] / res[0]:.1f}")
