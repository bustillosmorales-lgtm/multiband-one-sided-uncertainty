"""Shared setup for leak_moment_tests.py: the set K = (0,0.4) u (0.6,1), the logistic quadrature in x,
the transform U (Casini--Huerta eigenfunctions, CH normalization (37) with the 2 pi of the Fourier
convention), and the function analyse(phi), which returns the norm check, the mass of F on the t-grid,
the leak from F, int m dnu, the one-sided moment D and <Lambda^+>."""
import numpy as np
import warnings; warnings.filterwarnings("ignore")
K = [(0.0, 0.4), (0.6, 1.0)]; A = [a for a, b in K]; B = [b for a, b in K]; n = len(K)
z = lambda x: np.log(np.abs(np.prod([x - a for a in A], 0) / np.prod([x - b for b in B], 0)))
zp = lambda x: sum(1 / (x - a) - 1 / (x - b) for a, b in K)
def Nk(k):
    s1 = sum(np.prod([B[j] - A[l] for l in range(n) if l != k]) / ((B[j] - A[k]) * np.prod([B[j] - B[l] for l in range(n) if l != j])) for j in range(n))
    s2 = np.prod([A[k] - A[j] for j in range(n) if j != k]) / np.prod([A[k] - B[j] for j in range(n)])
    return np.sqrt(np.pi * (s1 - s2))
NK = [Nk(k) for k in range(n)]
xs_ = np.linspace(1e-6, .4 - 1e-6, 400001); amax = 1 / zp(xs_).min(); kappa = 1 / (2 * np.pi * amax)
# quadrature: logistic variable per component
u = np.linspace(-22, 22, 8001); du = u[1] - u[0]
X, W, TH = [], [], []
for j, (a, b) in enumerate(K):
    sg = 1 / (1 + np.exp(-u)); X.append(a + (b - a) * sg); W.append((b - a) * sg * (1 - sg) * du); TH.append((-1) ** j)
X = np.concatenate(X); W = np.concatenate(W); TH = np.concatenate([np.full(len(u), t) for t in TH])
ZX = z(X); ROOT = np.sqrt(np.abs(np.prod([X - a for a in A], 0) * np.prod([X - b for b in B], 0)))
s = np.linspace(-60, 60, 6001); ds = s[1] - s[0]
t = np.linspace(-80, 80, 8001); dt = t[1] - t[0]
PSIamp = [TH * np.prod([X - A[i] for i in range(n) if i != k], 0) / (NK[k] * ROOT) for k in range(n)]
def analyse(phi):
    nrm = np.sum(np.abs(phi) ** 2 * W)
    nu = np.zeros_like(s)
    for k in range(n):
        uk = np.zeros_like(s, complex)
        for i in range(0, len(s), 500):
            uk[i:i+500] = np.exp(1j * np.outer(s[i:i+500], ZX)) @ (PSIamp[k] * phi * W)
        nu += np.abs(uk) ** 2
    F = np.zeros_like(t, complex)
    for i in range(0, len(t), 500):
        F[i:i+500] = np.exp(2j * np.pi * np.outer(t[i:i+500], X)) @ (phi * W)
    P = np.abs(F) ** 2 / nrm; nu = nu / nrm
    mu = np.sum(P[t < 0]) * dt; D = np.sum((t * P)[t > 0]) * dt
    m = 1 / (1 + np.exp(np.clip(2 * np.pi * s, -700, 700)))
    return dict(pars=np.sum(nu) * ds, mu=mu, mu_spec=np.sum(m * nu) * ds, D=D,
                Lp=np.sum(np.maximum(s, 0) * nu) * ds, tot=np.sum(P) * dt)
bump = lambda c, w: np.exp(-((X - c) / w) ** 2)
rng = np.random.default_rng(1)
fams = {
 "one comp (x=.2)": lambda om: bump(.2, .05) * np.exp(2j*np.pi*om*X),
 "both comps":      lambda om: (bump(.2, .05) + 0.7*bump(.8, .05)) * np.exp(2j*np.pi*om*X),
 "random 4 bumps":  lambda om: sum(rng.normal()*bump(c, .04)*np.exp(2j*np.pi*om*(1+.3*rng.normal())*X) for c in (.1,.3,.7,.9)),
}
# partner packet (near-extremal), parameterised by s0 with delta = 1
z0 = z(np.array([xs_[np.argmin(zp(xs_))]]))[0]
lo, hi = .6 + 1e-15, 1 - 1e-15
for _ in range(200):
    mid = (lo + hi) / 2; lo, hi = (mid, hi) if z(np.array([mid]))[0] < z0 else (lo, mid)
xo = (lo + hi) / 2
fams["partner packet"] = lambda s0: TH * (X - xo) / ROOT * np.exp(-(ZX - z0) ** 2 / 4) * np.exp(-1j * s0 * ZX)
