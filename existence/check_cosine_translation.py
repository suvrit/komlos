"""Basic checks for the exact cosine translation transform. Not the proof."""
import json,math
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

def A(delta,t):
 b=2*abs(t)
 if b<1e-10:return (1-delta/math.pi)*math.cos(delta)+math.sin(delta)/math.pi
 # scaled hyperbolic functions
 ratio_s=math.exp(-b*delta)*(-math.expm1(-2*b*(math.pi-delta)))/(-math.expm1(-2*b*math.pi))
 ratio_c=math.exp(-b*delta)*(1+math.exp(-2*b*(math.pi-delta)))/(-math.expm1(-2*b*math.pi))
 return math.cos(delta)*ratio_s+b*math.sin(delta)*ratio_c

def direct(delta,t):
 a=delta/2;end=math.pi/2-a
 def f(x):
  pp=math.cos(x+a);mm=math.cos(x-a)
  return 4/math.pi*pp*mm*math.cos(2*t*math.log(pp/mm))
 return quad(f,0,end,epsabs=2e-12,epsrel=2e-12,limit=500)[0]

def direct_overlap_two(d1,d2):
 a=d1/2;b=d2/2;ex=math.pi/2-a;ey=math.pi/2-b
 def primitive(y,shift):return y/2+math.sin(2*(y+shift))/4
 def integrand(x):
  y0=max(-ey,min(ey,math.atan(-math.tan(x)*math.tan(a)/math.tan(b))))
  low=math.cos(x-a)**2*(primitive(y0,-b)-primitive(-ey,-b))
  high=math.cos(x+a)**2*(primitive(ey,b)-primitive(y0,b))
  return 4/math.pi**2*(low+high)
 return quad(integrand,-ex,ex,epsabs=2e-11,epsrel=2e-11,limit=400)[0]

if __name__=='__main__':
 records=[]
 for delta in [.05,.4,1.,math.pi/2]:
  for t in [0.,.1,.5,1.,3.]:
   exact=A(delta,t);num=direct(delta,t)
   assert abs(exact-num)<3e-10,(delta,t,exact,num)
   records.append(dict(delta=delta,t=t,formula=exact,quadrature=num,error=abs(exact-num)))
 mingap=1.
 for delta in np.linspace(.001,math.pi/2,500):
  for t in np.r_[0,np.geomspace(1e-5,1e4,250)]:
   a=A(delta,t);g=math.exp(-delta*delta*(1+4*t*t)/2)
   assert a>=g-2e-14,(delta,t,a,g)
   mingap=min(mingap,a-g)
 overlaps=[]
 for shifts in [[.2],[1.],[math.pi/2],[.5,.5],[1.,.3,.2],[.2]*20]:
  ov=quad(lambda t:math.prod(A(d,t) for d in shifts)/(t*t+.25),0,math.inf,epsabs=2e-11,epsrel=2e-11,limit=300)[0]/math.pi
  lower=2*ndtr(-np.linalg.norm(shifts));assert ov>=lower-1e-10
  overlaps.append(dict(shifts=shifts,overlap=ov,gaussian_lower=lower))
  if len(shifts)==1:assert abs(ov-(1-(shifts[0]+math.sin(shifts[0]))/math.pi))<1e-10
  if len(shifts)==2:assert abs(ov-direct_overlap_two(*shifts))<1e-9
 for shifts in [[.8,.3],[1.5,.2],[1.2,1.1]]:
  ov=quad(lambda t:math.prod(A(d,t) for d in shifts)/(t*t+.25),0,math.inf,epsabs=2e-11,epsrel=2e-11,limit=300)[0]/math.pi
  direct_value=direct_overlap_two(*shifts);assert abs(ov-direct_value)<1e-9
  overlaps.append(dict(shifts=shifts,overlap=ov,direct_2d_overlap=direct_value,absolute_error=abs(ov-direct_value)))
 out=dict(status='numerical identity and basic-example checks only',identity_cases=records,minimum_transform_gap=mingap,overlap_cases=overlaps)
 Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'max_identity_error':max(r['error'] for r in records),'transform_grid':'PASS','overlaps':overlaps},indent=2))
