import sys, time, math, json
import numpy as np
from flint import arb, acb, acb_mat, ctx, arb_mat
from hp import setK, build, tofloat

name=sys.argv[1]; deg=int(sys.argv[2]); ht=int(sys.argv[3]); lam0=float(sys.argv[4]); nsteps=int(sys.argv[5]); per_dec=int(sys.argv[6])
bprec=int(sys.argv[7]) if len(sys.argv)>7 else 1200
tag=f"{name}_d{deg}_h{ht}"+(sys.argv[8] if len(sys.argv)>8 else "")
out=open(f"par_{tag}.jsonl","a")
K=setK(name); t0=time.time()
Q,Qp,E=build(K,deg,bprec,ht)
n=Q.nrows(); print(tag,"built",time.time()-t0,flush=True)
pi=arb.pi()
def setprec(bits): ctx.prec=bits
def mats(prec):
    setprec(prec)
    A=acb_mat(n,n); B=acb_mat(n,n)
    for i in range(n):
        for j in range(n):
            A[i,j]=acb((0.5 if i==j else 0), -Q[i,j].mid()/(2*pi))
            B[i,j]=acb(Qp[i,j].mid()/(4*pi*pi), -E[i,j].mid()/(4*pi))
    return A,B
def inner(x,Mx):  # <x, M x>, x column acb_mat
    s=acb(0)
    for i in range(n): s+=x[i,0].conjugate()*Mx[i,0]
    return s
def nrm2(x):
    s=arb(0)
    for i in range(n): s+=(x[i,0].real**2+x[i,0].imag**2)
    return s
def midm(M):
    R=acb_mat(M.nrows(),M.ncols())
    for i in range(M.nrows()):
        for j in range(M.ncols()): R[i,j]=M[i,j].mid()
    return R
def chol_ok(M):
    # Cholesky; return True if positive definite (pivots >0)
    L=[[acb(0)]*n for _ in range(n)]
    Mm=[[M[i,j] for j in range(n)] for i in range(n)]
    for k in range(n):
        d=Mm[k][k].real
        if not (d>0): return False
        r=d.sqrt()
        col=[Mm[i][k]/r for i in range(k+1,n)]
        for ii,i in enumerate(range(k+1,n)):
            ci=col[ii]
            for jj in range(ii+1):
                j=k+1+jj
                Mm[i][j]=Mm[i][j]-ci*col[jj].conjugate()
    return True
# initial double precision vector
A0,B0=mats(300)
def tofl(M):
    return np.array([[complex(float(M[i,j].real.mid()),float(M[i,j].imag.mid())) for j in range(n)] for i in range(n)])
Af=tofl(A0);Bf=tofl(B0)
Mf=Af+lam0*Bf;Mf=(Mf+Mf.conj().T)/2
e,v=np.linalg.eigh(Mf); c0=v[:,0]
print("lam0",lam0,"e1",e[0],"e2",e[1],flush=True)
mu_last=1e-3; x=None; prec_cur=0; A=B=None
logl=math.log10(lam0); e1_est=e[0]
for step in range(nsteps):
    lam=10**(logl-step/per_dec)
    prec=max(500,int(3.33*(max(0,-math.log10(max(e1_est,1e-300)))+90)))
    prec=((prec+63)//64)*64
    if prec!=prec_cur or A is None:
        A,B=mats(prec); prec_cur=prec
    setprec(prec)
    if x is None:
        x=acb_mat(n,1)
        for i in range(n): x[i,0]=acb(float(c0[i].real),float(c0[i].imag))
    else:
        x=midm(x)
    laM=arb(lam)  # use exactly the double
    M=acb_mat(n,n)
    for i in range(n):
        for j in range(n): M[i,j]=A[i,j]+laM*B[i,j]
    sig=acb(arb(0.3*e1_est) if step>0 else arb(0.5*e1_est))
    Ms=acb_mat(n,n)
    for i in range(n):
        for j in range(n): Ms[i,j]=M[i,j]-(sig if i==j else 0)
    Minv=midm(Ms.inv())
    nit=0; mu_est=mu_last if step>0 else 1e-3
    target=arb(10)**(-(int(-math.log10(max(mu_est,1e-300)))+30))
    for it in range(12000):
        y=Minv*x
        ny=nrm2(y).sqrt(); x=midm(y*acb(1/ny)); nit+=1
        if it%10==9:
            Mx=M*x; nx=nrm2(x)
            rho=(inner(x,Mx)/acb(nx)).real.mid()
            res=Mx-x*acb(rho); rel=((nrm2(res)/nx).sqrt()/rho)
            if rel<target: break
    it=nit
    mu=(inner(x,A*x)/acb(nrm2(x))).real.mid(); D=(inner(x,B*x)/acb(nrm2(x))).real.mid()
    rho=mu+laM*D
    e1_est=float(rho) if rho>0 else 1e-300
    mu_last=float(mu)
    chk=None
    if step%12==0:
        Ms=acb_mat(n,n)
        for i in range(n):
            for j in range(n): Ms[i,j]=M[i,j]-(acb(rho*arb("0.999")) if i==j else 0)
        chk=bool(chol_ok(Ms))
    rec=dict(lam=lam,mu=mu.str(30,radius=False),D=float(D),logmu=float(mu.log()),slope=float(laM/mu),
             rho=rho.str(30,radius=False),relres=float(rel),its=it,prec=prec,lowest=chk,t=time.time()-t0)
    out.write(json.dumps(rec)+"\n");out.flush()
    print(f"{step} lam={lam:.3e} mu={float(mu.log())/math.log(10):.2f}(log10) D={float(D):.4f} slope={rec['slope']:.4f} res={float(rel):.1e} it={it} low={chk} t={rec['t']:.0f}",flush=True)
