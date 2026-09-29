"""Basic noncommuting examples for the generalized response and finite step.
The mathematical proofs are in komlos_constructive_final.tex.
These floating-point tests are not replacements for the proof.
"""
import numpy as np
from fractions import Fraction as F
import json
rng=np.random.default_rng(28092642)
identity_count=finite_count=0
max_response_error=max_norm_ratio=max_relative_ratio=0.
for n in (2,3,5,9,16):
 for trial in range(32):
  r=3*n+7
  P=rng.uniform(.01,1,size=(n,r)); weights=rng.uniform(.1,1,size=r)
  b=rng.uniform(0,.875,size=n)
  B=np.diag(b)
  gram=(P*weights)@P.T+.01*np.ones((n,n))
  lo,hi=0.,10.
  for _ in range(70):
    scale=(lo+hi)/2
    if np.linalg.eigvalsh(B+scale*gram)[-1]<1.05:lo=scale
    else:hi=scale
  M=B+hi*gram; weights*=hi
  ev,V=np.linalg.eigh(M);L=ev[-1];v=V[:,-1]
  if v.sum()<0:v=-v
  gap=L-ev[-2];zeta=min(1.,gap)
  DL=L-b;S=np.diag(1/np.sqrt(DL));t0=np.sqrt(np.dot(DL*v,v))
  KM=S@(M-B)@S
  ke,U=np.linalg.eigh(KM);u=np.sqrt(DL)*v/t0
  assert np.linalg.norm(KM@u-u)<1e-12
  assert abs(ke[-1]-1)<1e-12
  assert 1-ke[-2]>=gap/L-1e-12
  R=(V[:,:-1]/(L-ev[:-1]))@V[:,:-1].T
  RK=(U[:,:-1]/(1-ke[:-1]))@U[:,:-1].T
  for _ in range(3):
    y=rng.normal(size=n);y-=v*np.dot(v,y)
    lhs=float(y@R@y);rhs=float((S@y)@RK@(S@y))
    err=abs(lhs-rhs)/max(1.,abs(lhs),abs(rhs))
    max_response_error=max(max_response_error,err)
    assert err<2e-12
    identity_count+=1
  # Exact generalized high-forcing cancellation at this anchor.
  force=P*(weights*(P.T@v))
  high=ke>.75
  equations=U[:,high].T@S@force
  slopes=rng.normal(size=r)
  slopes-=equations.T@np.linalg.lstsq(equations@equations.T,equations@slopes,rcond=None)[0]
  slopes*=.8/max(abs(slopes))
  E=(P*(weights*slopes))@P.T
  Y=S@E@v
  assert np.linalg.norm(U[:,high].T@Y)<1e-12
  assert np.all(np.abs(Y)<=t0*u+1e-12)
  z=R@E@v;nu=min(v);kv=1+np.log(1/nu)
  max_norm_ratio=max(max_norm_ratio,float(np.linalg.norm(z)/16))
  max_relative_ratio=max(max_relative_ratio,float(max(abs(z)/v)/(64*kv)))
  assert np.linalg.norm(z)<=16+1e-12
  assert np.all(abs(z)<=64*kv*v+1e-12)
  q=rng.uniform(0,.5,size=r)
  N=(P*(weights*(slopes*slopes-q)))@P.T
  a=float(v@E@v);L2=float(v@N@v+2*(E@v)@R@(E@v))
  F0=max(L-1,0)**3
  F1=3*max(L-1,0)**2*a
  F2=3*max(L-1,0)**2*L2+6*max(L-1,0)*a*a
  base=M-(P*weights)@P.T
  for sign in (-1,1):
    tt=sign*.9*np.sqrt(zeta)/(4096*kv)
    Mt=base+(P*(weights*np.exp(slopes*tt-q*tt*tt/2)))@P.T
    Lt=np.linalg.eigvalsh(Mt)[-1]
    lhs=max(Lt-1,0)**3
    rhs=F0+F1*tt+.5*F2*tt*tt+2**16*abs(tt)**3+2**20*tt**4/zeta
    assert lhs<=rhs+1e-12
    finite_count+=1
# Exact algebraic cancellation of B0 in the large-row maximum.
exact_count=0
for B in (F(1,8),F(1,2),F(3,4)):
 for L in (F(1),F(257,256),F(9,8)):
  for delta in (F(1,25),F(57,185)**2):
   rr=F(9,4);slope=F(25,17)
   if slope*delta*B*rr/(L-B)>1:continue
   value=slope*delta*L/(L-B)+(1-slope*delta*B*rr/(L-B))/rr
   assert value==1/rr+slope*delta
   exact_count+=1
assert (1-F(129,4096)-F(1,1024))/(1+F(1,64))**2>F(1,2)
assert 608+4+F(768+5376,4096)+F(7,3)<1024
assert F(2**16,2**56)+F(2**20,2**112)<F(1,2**16)
report={'generalized_response_identity_checks':identity_count,
 'largest_relative_identity_error':max_response_error,
 'finite_generalized_curve_checks':finite_count,
 'maximum_response_norm_to_proved_bound':max_norm_ratio,
 'maximum_relative_response_to_proved_bound':max_relative_ratio,
 'exact_large_row_cancellation_checks':exact_count,
 'all_checks_passed':True}
print(json.dumps(report,indent=2))
