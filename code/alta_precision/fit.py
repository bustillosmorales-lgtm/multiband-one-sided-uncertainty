"""Slope of the optimal frontier from the high-precision runs (par_*.jsonl in the working folder).
INFO[name] = (pi^2 kappa(K), pi^2 max|I_j|, pi^2 |hull K|) for the sets scaled to total width 1:
"0" is the control interval, "A" is K_A = (0,3/7) u (4/7,8/7) and "B" is K_B of the article.
For each D it prints the local slopes of -log(mu_min) at the three highest degrees and their geometric extrapolation in the degree,
then fits s(D) = s_inf - c D^(-p) for p = 1/2 and p = 1."""
import json,glob,numpy as np
from scipy.optimize import least_squares
pi2=np.pi**2
INFO={"0":(pi2,pi2,pi2),"A":(6.8333,5.6400,11.2800),"B":(4.3860,3.2900,13.159)}
def load(fs):
    r=[]
    for f in fs: r+= [json.loads(l) for l in open(f)]
    r=[x for x in r if x['relres']<1e-25 and x['lowest'] is not False]
    D=np.array([x['D'] for x in r]);y=-np.array([x['logmu'] for x in r])
    o=np.argsort(D);D,y=D[o],y[o]
    keep=np.r_[True,np.diff(D)>1e-9];return D[keep],y[keep]
files={}
for f in glob.glob("par_*.jsonl"):
    p=f[4:-6].split("_");files.setdefault(p[0],{}).setdefault(int(p[1][1:]),[]).append(f)
res={}
for nm in "0AB":
    pk,pm,ph=INFO[nm]
    dat={d:load(fs) for d,fs in files[nm].items()}
    degs=sorted(dat)
    print(f"\n=== {nm}  pi2k={pk:.3f} pi2max={pm:.3f}  degrees:",[(d,round(dat[d][0].max(),1)) for d in degs])
    # largest converged D: |y_hi - y_lo| < tol
    hi=degs[-1]; lo=degs[-2]
    Dh,yh=dat[hi]; Dl,yl=dat[lo]
    grid=np.arange(2,min(Dh.max(),Dl.max()),0.5)
    diff=np.array([abs(np.interp(g,Dh,yh)-np.interp(g,Dl,yl)) for g in grid])
    # geometric Richardson extrapolation in the degree, with three degrees when available
    print("  D:     ", " ".join(f"{g:6.1f}" for g in grid[::2]))
    print("  |dy|hi-lo:", " ".join(f"{v:6.3f}" for v in diff[::2]))
    print("  local slopes (window 1.5), three highest degrees that cover D, and geometric extrapolation in the degree:")
    for g in np.arange(3,13,1.0):
        cov=[]
        for d in degs:
            D,y=dat[d]; m=(D>=g-.75)&(D<=g+.75)
            if m.sum()>=4: cov.append((d,np.polyfit(D[m],y[m],1)[0]))
        if len(cov)<3: continue
        (d1,a),(d2,b),(d3,c)=cov[-3:]
        r=(c-b)/(b-a) if (b-a)!=0 else np.nan
        ext=c+(c-b)*r/(1-r) if (0<r<0.9) else np.nan
        gap=(ext-pm)/(ph-pm)*100
        print(f"   D={g:4.1f} deg{d1}:{a:6.3f} deg{d2}:{b:6.3f} deg{d3}:{c:6.3f} r={r:5.2f} ext={ext:6.3f} ext/pi2k={ext/pk:5.3f} ext/pi2max={ext/pm:5.3f} gap%={gap:5.1f}")

print("\n=== FIT slope(D)=s - c*D^(-p) on the extrapolated slope ===")
def ext_series(nm):
    pk,pm,ph=INFO[nm]
    dat={d:load(fs) for d,fs in files[nm].items()}; degs=sorted(dat); out=[]
    for g in np.arange(3,13,0.5):
        cov=[]
        for d in degs:
            D,y=dat[d]; m=(D>=g-.75)&(D<=g+.75)
            if m.sum()>=4: cov.append((d,np.polyfit(D[m],y[m],1)[0]))
        if len(cov)<3: continue
        (d1,a),(d2,b),(d3,c)=cov[-3:]
        r=(c-b)/(b-a) if (b-a)!=0 else np.nan
        if 0<r<0.9: out.append((g,c+(c-b)*r/(1-r)))
    return np.array(out)
for nm in "0AB":
    pk,pm,ph=INFO[nm]; S=ext_series(nm)
    m=(S[:,0]>=4.5)&(S[:,0]<=(9.0 if nm=="A" else 20))
    for p in (0.5,1.0):
        X=np.c_[np.ones(m.sum()),-S[m,0]**(-p)]; co=np.linalg.lstsq(X,S[m,1],rcond=None)[0]
        print(f"{nm}: D in [{S[m,0].min():.1f},{S[m,0].max():.1f}] p={p}: s={co[0]:.3f} c={co[1]:.3f}  s/pi2k={co[0]/pk:.3f} s/pi2max={co[0]/pm:.3f}")
