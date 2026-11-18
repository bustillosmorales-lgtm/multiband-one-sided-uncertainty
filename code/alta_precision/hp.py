# High precision: matrices in the frequency domain (exact Hilbert transform as polynomial*log) with python-flint
import sys, time, pickle
from flint import fmpq, fmpq_poly, arb, arb_mat, ctx
import numpy as np

def setK(name):
    if name=="A": K=[(0,3/8),(4/8,1)]; Kq=[(fmpq(0),fmpq(3,8)),(fmpq(4,8),fmpq(1))]
    elif name=="B": Kq=[(fmpq(0),fmpq(1,4)),(fmpq(3,8),fmpq(5,8)),(fmpq(3,4),fmpq(1))]
    else: Kq=[(fmpq(0),fmpq(1))]
    w=sum(b-a for a,b in Kq); return [(a/w,b/w) for a,b in Kq]

def jacobi_polys(deg,al=8):
    # J_k^{(al,al)} exact rational, as lists of fmpq coeffs
    x=fmpq_poly([0,1]); P=[fmpq_poly([1]), None]
    s=2*al
    P[1]=fmpq_poly([0,1])*(al+1)   # P1 = (al+1) x  for alpha=beta
    out=[P[0],P[1]]
    for n in range(1,deg-1):
        a1=2*(n+1)*(n+s+1)*(2*n+s)
        c1=(2*n+s+1)*(2*n+s+2)*(2*n+s)
        c3=2*(n+al)*(n+al)*(2*n+s+2)
        nxt=(fmpq_poly([0,1])*out[n]*c1 - out[n-1]*c3)/a1
        out.append(nxt)
    return out[:deg]

def build_coeffs(deg,al=8):
    J=jacobi_polys(deg,al); w=fmpq_poly([1,0,-1])**4
    q=[w*Jk for Jk in J]
    nrm=[]
    for k in range(deg):
        p=(w*J[k])*(w*J[k]) # (1-s^2)^8 J^2
        c=p.coeffs(); nrm.append(sum(c[m]*fmpq(2,m+1) for m in range(0,len(c),2)))
    qd=[p.derivative() for p in q]
    R=[]
    for p in q:
        c=p.coeffs(); n_=len(c)
        r=[fmpq(0)]*n_
        for pp in range(n_-1):
            t=fmpq(0)
            for n in range(pp+1,n_,2):
                t+=c[n]*fmpq(2,n-pp)
            r[pp]=t
        R.append(r)
    N=deg+8
    def mat(polys_coef):
        rows=[[arb(0)]*len(polys_coef) for _ in range(N)]
        return rows
    return q,qd,R,nrm

def to_arb_mat(coefs,N,prec):
    # coefs: list over functions of coefficient lists (fmpq) -> N x deg arb_mat
    deg=len(coefs); M=arb_mat(N,deg)
    for l,c in enumerate(coefs):
        for n,v in enumerate(c):
            if n<N and v!=0: M[n,l]=arb(v)
    return M

def tanh_sinh(hden,U):
    ht=arb(1)/hden
    nodes=[];wts=[];vs=[]
    pi=arb.pi(); jmax=int(U*hden)
    for j in range(-jmax,jmax+1):
        u=arb(j)*ht
        v=pi/2*u.sinh()
        s=v.tanh(); ch=v.cosh()
        w=ht*pi/2*u.cosh()/(ch*ch)
        nodes.append(s); wts.append(w); vs.append(v)
    return nodes,wts,vs

def build(K,deg,prec_bits=1200,ht=None,U=4.2):
    ctx.prec=prec_bits
    ht=ht or 64
    q,qd,R,nrm=build_coeffs(deg)
    N=deg+9
    Cq=to_arb_mat(q,N,prec_bits); Cd=to_arb_mat(qd,N,prec_bits); Cr=to_arb_mat(R,N,prec_bits)
    nodes,wts,vs=tanh_sinh(ht,U); nn=len(nodes)
    m=len(K); n=m*deg
    cc=[(arb(a.p)/arb(a.q)+arb(b.p)/arb(b.q))/2 for a,b in K]
    hh=[(arb(b.p)/arb(b.q)-arb(a.p)/arb(a.q))/2 for a,b in K]
    nu=[[ (1/(hh[I]*arb(nrm[k].p)/arb(nrm[k].q))).sqrt() for k in range(deg)] for I in range(m)]
    def powers(svals):
        V=arb_mat(nn,N)
        for i,s in enumerate(svals):
            p=arb(1)
            for e in range(N):
                V[i,e]=p; p=p*s
        return V
    Vself=powers(nodes)
    Qs=[[None]*m for _ in range(m)]
    PhiI=[]; PhidI=[]
    for I in range(m):
        PhiI.append(Vself*Cq); PhidI.append(Vself*Cd)
    Q=arb_mat(n,n); Qp=arb_mat(n,n); E=arb_mat(n,n)
    for I in range(m):
        # weighted target matrices (nn x deg)
        Wq=arb_mat(nn,deg); Wd=arb_mat(nn,deg)
        for i in range(nn):
            for k in range(deg):
                Wq[i,k]=wts[i]*PhiI[I][i,k]; Wd[i,k]=wts[i]*PhidI[I][i,k]
        Wqt=Wq.transpose(); Wdt=Wd.transpose()
        for J in range(m):
            if I==J:
                Vx=Vself; Lx=[-2*v for v in vs]
            else:
                sv=[(cc[I]+hh[I]*s-cc[J])/hh[J] for s in nodes]
                Vx=powers(sv); Lx=[abs((1-s)/(1+s)).log() for s in sv]
            Pq=Vx*Cq; Pr=Vx*Cr
            H=arb_mat(nn,deg)
            for i in range(nn):
                for l in range(deg): H[i,l]=Pq[i,l]*Lx[i]+Pr[i,l]
            B1=Wqt*H; B2=Wdt*H
            for k in range(deg):
                for l in range(deg):
                    sc=nu[I][k]*nu[J][l]
                    Q[I*deg+k,J*deg+l]=hh[I]*sc*B1[k,l]
                    Qp[I*deg+k,J*deg+l]=sc*B2[k,l]
            if I==J:
                Ee=Wdt*PhiI[I]
                for k in range(deg):
                    for l in range(deg):
                        E[I*deg+k,I*deg+l]=nu[I][k]*nu[I][l]*Ee[k,l]
    return Q,Qp,E

def tofloat(M):
    n=M.nrows(); return np.array([[float(M[i,j].mid()) for j in range(n)] for i in range(n)])
