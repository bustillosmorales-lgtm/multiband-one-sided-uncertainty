"""Recomputes every numerical value quoted in the article and reports discrepancies.

Cheap quantities (kappa(K), Proposition C, r_0, the maximizer and its partner) are recomputed here.
The rest is read from ../results/, produced by the other scripts in this folder.
Run from code/:  python verify_all.py
Each line shows the value quoted in the article, the recomputed value and OK or FAIL.
"""
import re
import numpy as np
from scipy.optimize import minimize_scalar

RES = "../results/"
rows = []


def check(name, quoted, computed, ok):
    rows.append((name, quoted, computed, bool(ok)))


def zprime(K):
    return lambda x: sum(1 / (x - a) - 1 / (x - b) for a, b in K)


def kappa(K):
    zp = zprime(K); best = np.inf; arg = None
    for a, b in K:
        r = minimize_scalar(zp, bounds=(a + 1e-12, b - 1e-12), method="bounded", options={"xatol": 1e-14})
        if r.fun < best:
            best, arg = r.fun, r.x
    return 4 / best, arg


def rounds(x, q, d):
    return abs(round(x, d) - q) < 10 ** (-d) / 2 + 1e-12


# ---- introduction and Proposition C
closed = lambda r: 2 * (1 - np.sqrt(r)) * (1 + r) / (1 + np.sqrt(r))
k02, _ = kappa([(-1, -0.2), (0.2, 1)])
check("kappa(K_0.2) (Intro, Prop. C example)", 0.917, round(k02, 3), rounds(k02, 0.917, 3))
check("Prop. C closed form vs numerical minimum, r=0.2", "equal", f"{closed(0.2) - k02:.1e}", abs(closed(0.2) - k02) < 1e-9)
for r in (0.05, 0.5, 0.9):
    kk, _ = kappa([(-1, -r), (r, 1)])
    check(f"Prop. C closed form vs numerical minimum, r={r}", "equal", f"{closed(r) - kk:.1e}", abs(closed(r) - kk) < 1e-8)
check("kappa(K_r)/(1-r) -> 1 as r -> 1 (Intro)", 1, round(closed(0.999) / 0.001, 3), abs(closed(0.999) / 0.001 - 1) < 1e-3)
check("kappa(K_r) -> 2 as r -> 0 (Intro)", 2, round(closed(1e-10), 4), abs(closed(1e-10) - 2) < 1e-4)
xs = np.sqrt((0.2 ** 2 + np.sqrt(0.2)) / (1 + np.sqrt(0.2)))
_, arg = kappa([(-1, -0.2), (0.2, 1)])
check("Prop. C maximizer x = ((r^2+sqrt r)/(1+sqrt r))^(1/2), r=0.2", round(xs, 6), round(abs(arg), 6), abs(abs(arg) - xs) < 1e-6)

# ---- Corollary B(iii) on random sets (not a quoted number; a sanity check of the statement)
rng = np.random.default_rng(0); bad = 0
for _ in range(200):
    n = rng.integers(2, 5); p = np.sort(rng.random(2 * n)); K = [(p[2 * i], p[2 * i + 1]) for i in range(n)]
    kk, _ = kappa(K)
    bad += not (max(b - a for a, b in K) < kk <= (p[-1] - p[0]) * (1 + 1e-9))
check("Cor. B(iii) max|I_j| < kappa(K) <= |hull K| on 200 random sets", "holds", f"{bad} violations", bad == 0)

# ---- Section 3 constant
lam = np.linspace(0, 5, 2000001); r0 = np.max(2 * lam / (np.exp(2 * np.pi * lam) + 1))
check("r_0 = 0.0886...", 0.0886, round(r0, 4), rounds(r0, 0.0886, 4))

# ---- numerical section: the set (0,0.4) u (0.6,1)
K = [(0, 0.4), (0.6, 1)]
kK, _ = kappa(K)
xstar = minimize_scalar(zprime(K), bounds=(1e-12, 0.4 - 1e-12), method="bounded", options={"xatol": 1e-14}).x
check("kappa(K) for (0,0.4)u(0.6,1)", 0.4584, round(kK, 4), rounds(kK, 0.4584, 4))
check("maximizer x* in (0,0.4)", 0.2099, round(xstar, 4), rounds(xstar, 0.2099, 4))
check("second maximizer 1 - x* (symmetric set)", round(1 - 0.2099, 4), round(kappa(K)[1], 4), abs(zprime(K)(1 - xstar) - zprime(K)(xstar)) < 1e-9)
z = lambda x: np.log(-np.prod([x - a for a, b in K]) / np.prod([x - b for a, b in K]))
lo, hi = 0.6 + 1e-15, 1 - 1e-15
for _ in range(200):
    mid = (lo + hi) / 2; lo, hi = (mid, hi) if z(mid) < z(xstar) else (lo, mid)
check("partner of x*", 0.6723, round(lo, 4), rounds(lo, 0.6723, 4))
k1 = 1 / (2 * np.pi * kK / 4)
check("kappa_1 = 1/(2 pi a*)", 1.389, round(k1, 3), rounds(k1, 1.389, 3))

# fibre orthogonality
t = open(RES + "fibre_orthogonality.txt").read()
err = max(float(x) for x in re.findall(r"max \|WW\^T/c - I\| = ([\d.e+-]+)", t))
check("W^T W = I to 1e-15 (n = 2, 3, 4)", "1e-15", f"{err:.1e}", err < 5e-15)

# leak identity and moment bound on 16 bumps
t = open(RES + "leak_moment_tests.txt").read()
rel = [abs(float(x)) for x in re.findall(r"rel\.diff=([+-][\d.]+)%", t)]
par = [float(x) for x in re.findall(r"Parseval=([\d.]+)", t)]
mass = [float(x) for x in re.findall(r"mass=([\d.]+)", t)]
dk = [float(x) for x in re.findall(r"D-kappa_1<Lam\+>=([+-][\d.]+)", t)]
edge = [float(x) for x in re.findall(r"edge/max=([\d.e+-]+)", t)]
check("number of bump test functions", 16, len(rel), len(rel) == 16)
check("U preserves the norm to six digits", "6 digits", f"{max(abs(p - 1) for p in par):.1e}", max(abs(p - 1) for p in par) < 5e-7)
check("t-grid captures the mass to seven digits", "7 digits", f"{max(abs(m - 1) for m in mass):.1e}", max(abs(m - 1) for m in mass) < 5e-8)
check("leak vs int m dnu, all cases within 3.3 per cent", 3.3, round(max(rel), 2), max(rel) <= 3.3)
check("leak vs int m dnu within 1.2 per cent", "13 of 16", f"{sum(r <= 1.2 for r in rel)} of 16", sum(r <= 1.2 for r in rel) == 13)
check("bumps below 2e-7 of their maximum at the endpoints", "2e-7", f"{max(edge):.1e}", max(edge) < 2e-7)
check("D - kappa_1 <Lambda^+> positive in every case", "> 0", f"min {min(dk):+.4f}", min(dk) > 0)
m = re.search(r"one comp \(x=\.2\).*?omega= 20:.*?D=([\d.]+) <Lam\+>=([\d.]+)", t, re.S)
ratio = float(m.group(1)) / float(m.group(2))
check("D/<Lambda^+> for the bump at x=0.2, frequency 20", 1.416, round(ratio, 3), rounds(ratio, 1.416, 3))
fam = re.split(r"\n(?=\S[^\n]*\n  omega)", t)
grp = {}
for blk in fam:
    name = blk.strip().splitlines()[0]
    grp[name] = [abs(float(x)) for x in re.findall(r"rel\.diff=([+-][\d.]+)%", blk)]
mono = grp["one comp (x=.2)"] + grp["both comps"]
incr = all(np.diff(grp["one comp (x=.2)"]) > 0) and all(np.diff(grp["both comps"]) > 0)
check("one and two bumps: difference grows with frequency, 0.4 to 3.2 per cent", "0.4 to 3.2",
      f"{min(mono):.1f} to {max(mono):.1f}", incr and round(min(mono), 1) == 0.4 and round(max(mono), 1) == 3.2)
rnd = grp["random 4 bumps"] + grp["random phases 6 bumps"]
check("random-frequency families below 0.5 per cent", 0.5, round(max(rnd), 2), max(rnd) < 0.5)

# packets
t = open(RES + "packets.txt").read()
lk = [abs(float(x)) for x in re.findall(r"rel\.diff ([+-][\d.e+-]+)", t)]
check("packet leak vs exact Gaussian value, within 0.11 per cent", 0.11, round(100 * max(lk), 3), 100 * max(lk) <= 0.11)
fm = [abs(float(a) / float(b) - 1) for a, b in re.findall(r"first moment from F ([\d.]+)\s+predicted ([\d.]+)", t)]
check("first moment from F vs Lemma 5.1(ii), within 0.6 per cent", 0.6, round(100 * max(fm), 2), 100 * max(fm) <= 0.6)
sc = [float(x) for x in re.findall(r"4/<z'> = ([\d.]+)", t)]
for q, c in zip((0.4158, 0.4474, 0.4566, 0.4579), sc):
    check("4/sum_j int z' A_j^2 (delta = 0.5, 0.25, 0.1, 0.05)", q, round(c, 4), rounds(c, q, 4))
mo = [float(x) for x in re.findall(r"mass outside = ([\d.e+-]+)", t)]
dl = (0.5, 0.25, 0.1, 0.05)
slope = np.polyfit(np.log(dl), np.log(mo), 1)[0]
check("mass outside I_l scales as delta^2", 2, round(slope, 2), abs(slope - 2) < 0.1)
w, wo, rt = re.search(r"with companion ([\d.e+-]+), without ([\d.e+-]+), ratio ([\d.]+)", t).groups()
check("leak with companion, s0=4, delta=1", 1.69e-9, float(f"{float(w):.2e}"), abs(float(w) - 1.69e-9) < 0.005e-9)
check("leak without companion", 1.76e-7, float(f"{float(wo):.2e}"), abs(float(wo) - 1.76e-7) < 0.005e-7)
check("ratio (more than two orders of magnitude)", 104, round(float(rt)), round(float(rt)) == 104)

# Toeplitz table and text
t = open(RES + "toeplitz_keff.txt").read()
table = {"[0,.6]": (0.600, 0.600, 0.6016, 0.6000), "[0,.4]u[.6,1]": (0.400, 0.458, 0.4667, 0.4559),
         "[0,.45]u[.55,1]": (0.450, 0.571, 0.5693, 0.5533), "[0,.7]u[.8,1]": (0.700, 0.802, 0.7471, 0.7520)}
for name, (lm, kq, e8, e16) in table.items():
    m = re.search(re.escape(name) + r": cand=([\d.]+) L_max=([\d.]+)\s+keff n=2\^8\.\.2\^16: ([\d. ]+)", t)
    cand, lmax = float(m.group(1)), float(m.group(2)); ke = [float(x) for x in m.group(3).split()]
    check(f"Table: max|I_j| for {name}", lm, round(lmax, 3), rounds(lmax, lm, 3))
    check(f"Table: kappa(K) for {name}", kq, round(cand, 3), rounds(cand, kq, 3))
    check(f"Table: kappa_eff m=2^8 for {name}", e8, round(ke[0], 4), rounds(ke[0], e8, 4))
    check(f"Table: kappa_eff m=2^16 for {name}", e16, round(ke[-1], 4), rounds(ke[-1], e16, 4))
    if name != "[0,.6]":
        table[name] = (lm, cand, ke[0], ke[-1])
below = [100 * (1 - v[3] / v[1]) for k, v in table.items() if k != "[0,.6]"]
check("at m=2^16, 0.5 to 6.2 per cent below kappa(K)", "0.5 to 6.2", f"{min(below):.2f} to {max(below):.2f}",
      rounds(min(below), 0.5, 1) and rounds(max(below), 6.2, 1))
v = table["[0,.4]u[.6,1]"]; above = 100 * (v[2] / v[1] - 1)
check("at m=2^8, 1.8 per cent above kappa(K) for the first set", 1.8, round(above, 1), rounds(above, 1.8, 1))
t = open(RES + "toeplitz_second_order.txt").read()
ck = [float(a) for a in re.findall(r"c_K=([\d.]+),", t)]
for q, c in zip((0.385, 0.552, 0.505, 0.25), ck):
    check("c_K (three sets) and c = 1/4 (interval)", q, round(c, 3), rounds(c, q, 3))
sl = [(float(a), float(b)) for a, b in re.findall(r"measured ([+-][\d.]+)\s+predicted ([+-][\d.]+)", t)]
pc = [100 * a / b for a, b in sl if b > 0.01]
check("slope 40 to 61 per cent of the packet slope", "40 to 61", f"{min(pc):.0f} to {max(pc):.0f}", round(min(pc)) == 40 and round(max(pc)) == 61)

w = max(len(r[0]) for r in rows)
for name, q, c, ok in rows:
    print(f"{'OK  ' if ok else 'FAIL'} | {name:<{w}} | quoted {str(q):<12} | computed {c}")
print(f"\n{len(rows)} checks, {sum(not r[3] for r in rows)} FAIL")
