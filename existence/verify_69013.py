"""Exact rational certificate for G(pi/C)<1 at C=69013/10000.
Only integer/Fraction arithmetic enters assertions.
"""
from fractions import Fraction as F
from math import factorial
from pathlib import Path
import json

def atan_bounds(x,N=80):
 s=sum(((-1)**n*x**(2*n+1)/F(2*n+1) for n in range(N)),F(0))
 t=s+(-1)**N*x**(2*N+1)/F(2*N+1)
 return min(s,t),max(s,t)
a,b=atan_bounds(F(1,5));c,d=atan_bounds(F(1,239))
pi_lo=16*a-4*d;pi_hi=16*b-4*c
assert F(314159265358979323846264338327950288419716939937510,10**50)<pi_lo
assert pi_hi<F(314159265358979323846264338327950288419716939937511,10**50)
def G_squared_bounds(C):
 zlo=9*pi_lo*pi_lo/(2*C*C);zhi=9*pi_hi*pi_hi/(2*C*C)
 # Even Taylor polynomials majorize exp(-x), and odd ones minorize it.
 upper=sum(((-zlo)**n/F(factorial(n)*(2*n+1)*(2*n+2)*(2*n+3)) for n in range(21)),F(0))
 lower=sum(((-zhi)**n/F(factorial(n)*(2*n+1)*(2*n+2)*(2*n+3)) for n in range(22)),F(0))
 assert 0<lower<=upper
 return 648*pi_lo*lower*lower/(C*C),648*pi_hi*upper*upper/(C*C)
C=F(69013,10000)
assert C>6
G_squared_lower,G_squared_upper=G_squared_bounds(C)
assert G_squared_upper<1
assert G_squared_upper<F(99999,100000)
assert G_squared_bounds(F(6))[0]>1
critical_lo=F('6.9012574');critical_hi=F('6.9012575')
assert G_squared_bounds(critical_lo)[0]>1
assert G_squared_bounds(critical_hi)[1]<1
out={'constant':str(C),'constant_decimal':float(C),
     'G_squared_upper_decimal':float(G_squared_upper),
     'certified_rational_majorant':'99999/100000',
     'critical_C_exact_bracket':[str(critical_lo),str(critical_hi)],
     'pi_machin_terms':80,'exponential_upper_degree':20,
     'status':'PASS: exact rational arithmetic proves G(pi/C)^2 < 99999/100000 < 1'}
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
