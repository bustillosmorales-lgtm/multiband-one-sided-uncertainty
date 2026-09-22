"""Second-order heuristic for the Toeplitz analogue (the numerical section).  Packet balance with <z'> = z'_min (1 + c_K delta^2) gives
   S_K = (l + pi sqrt(2 c_K l) + O(1)) / kappa_cand,   interval: c = 1/4  ->  pi sqrt(l/2).
So D(n) := S_K(n) - S_1(kappa n)/kappa should grow like  pi (sqrt(4 c_K) - 1) sqrt(l/2) / kappa + const.
c_K is measured from the packet (<z'> excess / delta^2).  Reads S_K and the S_1 grid from ../results/toeplitz_multiband.txt and ../results/toeplitz_keff.txt.
Run from code/:  python toeplitz_second_order.py > ../results/toeplitz_second_order.txt"""
import re
import numpy as np
from scipy.interpolate import CubicSpline
import warnings; warnings.filterwarnings("ignore")

def geom(K):
    z = lambda x: np.log(np.abs(np.prod([x - a for a, b in K], 0) / np.prod([x - b for a, b in K], 0)))
    zp = lambda x: sum(1 / (x - a) - 1 / (x - b) for a, b in K)
    best = None
    for l, (a, b) in enumerate(K):
        xs = np.linspace(a + 1e-7, b - 1e-7, 200001); i = np.argmin(zp(xs))
        if best is None or zp(xs[i]) < best[1]:
            best = (xs[i], zp(xs[i]), l)
    xstar, zmin, l = best; z0 = z(xstar)
    partners = []
    for m, (a, b) in enumerate(K):
        if m == l: continue
        lo, hi = a + 1e-15, b - 1e-15
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if z(mid) < z0 else (lo, mid)
        partners.append((lo + hi) / 2)
    u = np.linspace(-25, 25, 500001); du = u[1] - u[0]
    out = []
    for d in (0.05, 0.025):
        n = m1 = 0
        for a, b in K:
            sg = 1 / (1 + np.exp(-u)); x = a + (b - a) * sg; jac = (b - a) * sg * (1 - sg)
            P = np.prod([x - p for p in partners], 0) if partners else 1.0
            A2 = P ** 2 / np.abs(np.prod([x - aa for aa, bb in K], 0) * np.prod([x - bb for aa, bb in K], 0)) \
                * np.exp(-(z(x) - z0) ** 2 / (2 * d * d))
            n += np.sum(A2 * jac) * du; m1 += np.sum(zp(x) * A2 * jac) * du
        out.append((m1 / n / zmin - 1) / d ** 2)
    return 4 / zmin, out

txt = open("../results/toeplitz_keff.txt").read()
g = re.findall(r"S_1\((\d+)\) = ([\d.]+)", txt)
spl = CubicSpline(np.log([int(a) for a, _ in g]), [float(b) for _, b in g])
txt2 = open("../results/toeplitz_multiband.txt").read()
blocks = re.split(r"\n(?=\S+: kappa_K)", txt2.split("ratios against")[0].strip())
SETS = {"[0,.6]": [(0, .6)], "[0,.4]u[.6,1]": [(0, .4), (.6, 1)], "[0,.45]u[.55,1]": [(0, .45), (.55, 1)],
        "[0,.7]u[.8,1]": [(0, .7), (.8, 1)]}
for b in blocks:
    name = b.splitlines()[0].split(":")[0]
    if name not in SETS: continue
    kap, cs = geom(SETS[name])
    c = cs[-1]
    rows = re.findall(r"n=\s*(\d+)\s+S=\s*([\d.]+)", b)
    n = np.array([int(r[0]) for r in rows], float); S = np.array([float(r[1]) for r in rows]); l = np.log(n)
    diff = S - spl(np.log(kap * n)) / kap
    slope_meas, const = np.polyfit(np.sqrt(l), diff, 1)
    slope_pred = PI_ = np.pi * (np.sqrt(4 * c) - 1) / np.sqrt(2) / kap
    print(f"{name:>16}: kappa={kap:.4f} c_K={cs[0]:.4f},{cs[1]:.4f}  "
          f"slope of diff vs sqrt(l): measured {slope_meas:+.3f}  predicted {slope_pred:+.3f}   "
          f"diff n=2^8..2^16: " + " ".join(f"{x:+.2f}" for x in diff))
