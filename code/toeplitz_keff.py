"""Effective kappa: solve S_K(n) = S_1(kappa n)/kappa, with S_1 on a quarter-octave grid.
Exact for a single interval of length L (kappa_eff = L). Reads S_K from ../results/toeplitz_multiband.txt.
Run from code/:  python toeplitz_keff.py > ../results/toeplitz_keff.txt   (about 5 minutes)"""
import re, sys
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
sys.path.insert(0, ".")
from toeplitz_multiband import norm_sub, kappa_K, SETS
PI = np.pi
ms = sorted(set(int(round(2 ** (k / 4))) for k in range(24, 65)))
S1 = []
for m in ms:
    S1.append((PI / 2) * m * (PI - norm_sub(m, [(0.0, 1.0)])))
    print(f"S_1({m}) = {S1[-1]:.6f}", flush=True)
spl = CubicSpline(np.log(ms), S1)
txt = open("../results/toeplitz_multiband.txt").read()
blocks = re.split(r"\n(?=\S+: kappa_K)", txt.split("ratios against")[0].strip())
print("\nkappa_eff(n) with S_K(n) = S_1(kappa n)/kappa")
for b in blocks:
    name = b.splitlines()[0].split(":")[0]
    if name == "[0,1]":
        continue
    K = SETS[name]
    rows = re.findall(r"n=\s*(\d+)\s+S=\s*([\d.]+)", b)
    out = []
    for n, s in rows:
        n, s = int(n), float(s)
        f = lambda k: spl(np.log(k * n)) / k - s
        out.append(brentq(f, 0.2, 1.0))
    lmax = max(bb - a for a, bb in K)
    print(f"{name:>16}: cand={kappa_K(K):.4f} L_max={lmax:.3f}  keff n=2^8..2^16: "
          + " ".join(f"{x:.4f}" for x in out))
