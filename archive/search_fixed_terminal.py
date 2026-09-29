from scipy.optimize import differential_evolution,brentq
from math import log,sqrt
import json
res={}
for N in [336,337,338]:
 wt=19/6
 def calc(R,z):
  d,c,beta,A=z;aa=25/23;bb=32/23;cp=1.001**4/(A*(1-beta/100));ell=.999999
  co=1+R*R*(cp*ell+bb*d*d)
  B=2*ell/(co+sqrt(co*co-4*cp*R*R*ell))*(1-1e-7)
  a0=(-log(1-c)/c-1)/d;b=a0+4/beta
  H=R*(a0+b+log(2*A/(B*R*R))/beta);D=2*R*(1+sqrt(1-d*d))/d;C=H+D;reserve=b+D/R
  loss=aa*beta*d/c+bb*d*d+cp*(257/256-B)
  dim=1-1/R**2-loss-2/(N+1)-1/4094-.0001
  return C,dim,B,a0,b,H,D,reserve
 def solve(z):
  if calc(5,z)[1]<.0025:return None
  R=brentq(lambda r:calc(r,z)[1]-.0025,1,5,xtol=1e-13)
  return R,calc(R,z)
 def obj(z):
  sol=solve(z)
  if not sol:return 1000+1000*(.0025-calc(5,z)[1])
  C=sol[1][0];reserve=sol[1][-1]
  return C+1e5*max(0,(1+wt)*N-C*C)**2+1e5*max(0,(1+wt)*N/wt-reserve*reserve)**2
 opt=differential_evolution(obj,[(.28,.32),(.64,.69),(.65,.72),(5.5,7.0)],seed=827,popsize=25,tol=1e-11,maxiter=1200)
 R,vals=solve(opt.x);entry={'terminal_cutoff':N,'rounding_weight':wt,'parameters':[R]+opt.x.tolist(),'values':vals}
 res[str(N)]=entry;print(json.dumps(entry,indent=2),flush=True)
with open(__file__.replace('search_fixed_terminal.py','fixed_terminal_results.json'),'w') as f:json.dump(res,f,indent=2)
