"""Leak identity (Lemma 3.3) and moment bound (Proposition 4.1) on 16 bump test functions (the numerical section).
Every bump is negligible at the endpoints of K
(value below 1e-6), so that the test functions behave as elements of C_c^infty(K) and F decays fast.
Positive frequencies only (small leak).  t in [-150,150].  Run from code/:
    python leak_moment_tests.py > ../results/leak_moment_tests.txt   (about 15 minutes)
"""
import numpy as np, warnings
warnings.filterwarnings("ignore")
src = open("leak_setup.py").read()
src = src.replace("t = np.linspace(-80, 80, 8001)", "t = np.linspace(-150, 150, 15001)")
exec(src)
rng = np.random.default_rng(7)
fams = {
 "one comp (x=.2)": lambda om: bump(.2, .05) * np.exp(-2j*np.pi*om*X),
 "both comps":      lambda om: (bump(.2, .05) + 0.7*bump(.8, .05)) * np.exp(-2j*np.pi*om*X),
 "random 4 bumps":  lambda om: sum(rng.normal()*bump(c, .025)*np.exp(-2j*np.pi*om*(1+.3*rng.normal())*X) for c in (.1,.3,.7,.9)),
 "random phases 6 bumps": lambda om: sum(rng.normal()*np.exp(2j*np.pi*rng.random())*bump(c, .02)*np.exp(-2j*np.pi*om*(1+.5*rng.random())*X) for c in (.08,.2,.32,.68,.8,.92)),
}
print(f"kappa_1 = 1/(2 pi a*) = {kappa:.5f}")
worst = 0
for name, f in fams.items():
    print(f"\n{name}")
    for p in [2, 5, 10, 20]:
        phi = f(p)
        edge = max(abs(phi[np.argmin(np.abs(X - e))]) for e in (1e-3, .4 - 1e-3, .6 + 1e-3, 1 - 1e-3)) / np.abs(phi).max()
        r = analyse(phi)
        rd = r['mu'] / r['mu_spec'] - 1; worst = max(worst, abs(rd))
        print(f"  omega={p:>3}: Parseval={r['pars']:.6f} mass={r['tot']:.7f} leak={r['mu']:.4e} m(Lambda)={r['mu_spec']:.4e} "
              f"rel.diff={rd:+.3%}  D={r['D']:.3f} <Lam+>={r['Lp']:.3f}  D-kappa_1<Lam+>={r['D']-kappa*r['Lp']:+.4f}  edge/max={edge:.1e}")
print(f"\nworst relative difference leak vs int m dnu: {worst:.3%}")
