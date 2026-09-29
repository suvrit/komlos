"""Basic noncommuting and sharp commuting examples for the joint inertia lemma.
The current lemma is proved in komlos_constructive_final.tex. Floating-point tests
are regression/basic-example checks, not theorem certificates.
"""
import numpy as np
from fractions import Fraction as F
import json
from pathlib import Path
rng=np.random.default_rng(280926)
cases=0
smallest_margin=1e99
for n in (2,3,5,9,16):
  for trial in range(60):
    V=np.linalg.qr(rng.normal(size=(n,n)))[0]
    eig=rng.uniform(0,1,size=n);eig[0]=1
    N=(V*eig)@V.T
    eps=(0,.0001,.01)[trial%3]
    C=(V*np.sqrt(eig+eps))@np.linalg.qr(rng.normal(size=(n,n)))[0].T
    C*=rng.uniform(.6,1)
    G=rng.normal(size=(n,n))
    if trial%4==0: G[:,0]=0
    rho=rng.uniform(.05,1.5)
    D=np.sum(G*G,axis=0)/rho
    active=D>1e-14
    Gn=G[:,active]/np.sqrt(D[active])
    high=eig>=.72
    R=(V[:,~high]/(1-eig[~high]))@V[:,~high].T
    S=Gn.T@(np.eye(n)+2*C.T@R@C)@Gn-np.eye(sum(active))
    k=int(sum(np.linalg.eigvalsh(S)>1e-8));h=int(sum(high))
    bound=25/23*rho*n+32/23*eig.sum()+50/23*eps*n
    margin=bound-h-k
    assert margin>=-1e-7,(n,trial,k,h,bound)
    smallest_margin=min(smallest_margin,float(margin));cases+=1
# Exact supporting-line identity and sharp diagonal interior examples.
for num in range(101):
    x=F(num,100)
    lhs=(1-x)/(1+x)-(F(23,25)-F(32,25)*x)
    rhs=F(32,25)*(x-F(1,4))**2/(1+x)
    assert lhs==rhs>=0
# At lambda=1/4, rho=3/5 the tangent count equals the dimension.
assert F(25,23)*F(3,5)+F(32,23)*F(1,4)==1
# Explicit finite-Perron complementary-gap and remainder constants.
assert (1-F(34,1024))/(1+F(1,64))**2>F(1,2)
assert F(4096,2**48)+F(65536,2**96)<F(1,2**16)
assert F(32,1024**2)<F(1,6)
assert 80+F(25,64)+F(400,1024)+F(7,3)<256
# PSD/nonnegative finite curves, forcing canceled in high eigenspace.
perron_cases=0
for n in (2,4,8):
  for trial in range(24):
    r=3*n+5
    P=rng.uniform(0,1,size=(n,r));a=rng.uniform(.2,1,size=r)
    M=(P*a)@P.T+.02*np.ones((n,n))+.1*np.eye(n)
    ev,U=np.linalg.eigh(M);M*=1.05/ev[-1];a*=1.05/ev[-1]
    ev,U=np.linalg.eigh(M);L=ev[-1];v=U[:,-1]
    if v.sum()<0:v=-v
    high=ev>.75*L
    B=(P*(a*(P.T@v)))
    equations=U[:,high].T@B
    slopes=rng.normal(size=r)
    slopes-=equations.T@np.linalg.lstsq(equations@equations.T,equations@slopes,rcond=None)[0]
    slopes*=.7/max(abs(slopes));K=1.
    q=rng.uniform(0,.5,size=r)
    E=(P*(a*slopes))@P.T
    N2=(P*(a*(slopes**2-q)))@P.T
    alpha=float(v@E@v)
    gap=L-ev[-2];zeta=min(gap,1.)
    nu=min(v);kv=1+np.log(1/nu)
    forcing=np.linalg.norm(U[:,high].T@E@v)
    assert forcing<1e-12
    R0=(U[:,:-1]/(L-ev[:-1]))@U[:,:-1].T
    L2=float(v@N2@v+2*(E@v)@R0@(E@v))
    phi0=max(L-1,0)**3
    phi1=3*max(L-1,0)**2*alpha
    phi2=3*max(L-1,0)**2*L2+6*max(L-1,0)*alpha**2
    base=M-(P*a)@P.T
    for sign in (-1,1):
      t=sign*.9*np.sqrt(zeta)/(1024*kv)
      Mt=base+(P*(a*np.exp(slopes*t-q*t*t/2)))@P.T
      Lt=np.linalg.eigvalsh(Mt)[-1]
      exact=max(Lt-1,0)**3
      rhs=phi0+phi1*t+.5*phi2*t*t+4096*abs(t)**3+65536*t**4/zeta
      assert exact<=rhs+1e-12,(exact,rhs)
      perron_cases+=1
report={'noncommuting_inertia_examples':cases,'minimum_joint_bound_margin':smallest_margin,
        'exact_supporting_line_checks':101,'exact_sharp_diagonal_case':True,
        'finite_perron_curve_checks':perron_cases,'all_checks_passed':True}
print(json.dumps(report,indent=2))
