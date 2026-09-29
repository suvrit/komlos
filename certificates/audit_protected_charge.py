"""Exact algebra and basic state-budget checks for the joint protected charge."""
from fractions import Fraction as F
from random import Random
import json
r=Random(280926)
checks=0
for t in (F(1,10),F(1,6),F(1,5),F(2,9)):
 at=(1+t)**2/(1+2*t-t*t);bt=2/(1+2*t-t*t)
 for j in range(101):
  x=F(j,100)
  AA=1/at;BB=bt/at
  assert (1-x)/(1+x)-(AA-BB*x)==2*(x-t)**2/((1+x)*(1+t)**2)
 for B0 in (F(1,4),F(1,2),F(3,4)):
  for L in (F(1),F(257,256),F(9,8)):
   for Delta in (F(1,25),F(57,185)**2):
    R2=F(9,4);cp=F(1,25)
    coeff=1-B0*R2*(cp+bt*Delta/(L-B0))
    if coeff<0:continue
    # Exact endpoint simplification, with rho carried separately.
    old=cp*L+bt*Delta*L/(L-B0)+coeff/R2
    new=1/R2+bt*Delta+cp*(L-B0)
    assert old==new
    for _ in range(10):
      ellfrac=F(r.randrange(0,1001),1000)/R2
      raw=cp*L+bt*Delta*L/(L-B0)+coeff*ellfrac
      assert raw<=new
      checks+=1
print(json.dumps({'exact_affine_state_budget_checks':checks,'exact_tangent_checks':404,
                  'all_checks_passed':True},indent=2))
