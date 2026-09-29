"""Independent symbolic checks of the transform proof's algebra."""
import sympy as s
import json
from pathlib import Path
x,b,a,w=s.symbols('x b a w',positive=True,real=True)
N=s.cos(x)*s.sinh(b*(s.pi-x))+b*s.sin(x)*s.cosh(b*(s.pi-x))
f=x*(s.pi-x)*s.cos(x)-(s.pi-2*x)*s.sin(x)
checks={
 'logarithmic_derivative_numerator':s.simplify(s.diff(N,x)+(1+b*b)*s.sin(x)*s.sinh(b*(s.pi-x))),
 'trigonometric_comparison_derivative':s.simplify(s.diff(f,x)-(2-x*(s.pi-x))*s.sin(x)),
 'zero_frequency_normalization':s.simplify(s.limit(N/s.sinh(b*s.pi),b,0)-((s.pi-x)*s.cos(x)+s.sin(x))/s.pi),
}
I=2*s.pi/s.sin(2*a)*s.sinh((s.pi/2-a)*w)/s.sinh(s.pi*w/2)
A=(2/s.pi)*s.sin(a)**3*s.cos(a)**3*(-s.diff(I,a)/s.sin(2*a))
target=(s.cos(2*a)*s.sinh((s.pi-2*a)*w/2)+(w/2)*s.sin(2*a)*s.cosh((s.pi-2*a)*w/2))/s.sinh(s.pi*w/2)
checks['hyperbolic_integral_differentiation']=s.trigsimp(s.expand_trig(s.simplify(A-target)))
assert all(v==0 for v in checks.values()),checks
out={k:'PASS' for k in checks};Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
