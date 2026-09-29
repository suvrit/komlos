"""Independent exact audit of the C=37.54 certificate and numerical contracts."""
from fractions import Fraction as F
from math import factorial
import json
R=F(7139,5000);d=F(52,173);c=F(2,3);beta=F(17,25)
amp=F(25,4);dep=F(7463,10000);a0=F(5389,2500);b=F(341613,42500);H0=F(94893,5000)
At=F(23,25);Bt=F(32,25);ac=1/At;bc=Bt/At
cut=F(18,25);truecut=F(3,4)
theta=F(1001,1000);lmax=F(257,256);lmin=F(999999,1000000)
eta=F(1,2**40);delta=eta**5/F(2**100)
# Independent log3 enclosure via the atanh series at1/2.
z=F(1,2);terms=80
loglo=2*sum((z**(2*j+1)/F(2*j+1) for j in range(terms)),F())
loghi=loglo+2*z**(2*terms+1)/(F(2*terms+1)*(1-z*z))
flo=(F(3,2)*loglo-1)/d;fhi=(F(3,2)*loghi-1)/d
assert 0<flo<fhi<a0
assert beta*(b-a0)==4
X=beta*(H0/R-a0-b)
exp_lower=sum((X**j/F(factorial(j)) for j in range(80)),F())
assert X>0
assert exp_lower>2*amp/(dep*R*R)
assert beta*H0/R>2
assert -H0/R+a0+b<-3
# Pythagorean tracking and both terminal objectives.
assert F(165,173)**2==1-d*d
AD=(1+F(165,173))/d
assert AD==F(13,2)
C=H0+2*R*AD
assert C==F(1877,50)
stop=336;mu=F(19,6)
terminal=(1+mu)*stop
assert terminal==1400
assert terminal<C*C
assert terminal/mu<(b+2*AD)**2
# Fully joint protected/large/spectral charge.
rho=beta*d/c;cp=theta**4/(amp*(1-beta/F(100)))
condition=dep*R*R*(cp+bc*d*d/(lmin-dep))
assert condition<1
margin=1-1/(R*R)-ac*rho-bc*d*d-cp*(lmax-dep)-F(1,10000)-F(2,stop+1)-F(1,4094)
assert margin>F(1,400)
assert At/Bt==F(23,32)<cut<truecut
# Reconstructed normalized spectrum and unchanged low-resolvent bound.
assert F(256)*delta<F(1,1000000)
assert truecut-cut*(1+256*delta)>F(1,50)
assert 256*50<2**14
assert (1-cut)*(1-256*delta)>F(1,4)
assert ac<2 and bc<2
assert 18*eta+2**12*delta<F(1,10000)
# Unit gradient tests are (1-c)*g, not the baseline8g/25.
G=1/(1-c)
fpp=c/(d*(1-c)**2);fppp=2*c*c/(d*(1-c)**3)
assert G==3
assert beta*G==F(51,25)<F(25,8)
assert beta*fpp/F(16**2)<F(1,2)
assert beta*fppp/F(16**3)<F(1,4)
assert beta*fpp*d*d<F(9,4)
# Robust proxy norm using Rlow<4 and computed normalized Gram norm<1+256delta.
proxy_bound=(1+8*(1+256*delta))*(beta*G)**2*(lmax+delta)+beta*fpp*d*d*(lmax+delta)
assert proxy_bound<128
# Frozen cold derivative and finite-step remainder.
step=F(1,2**56)
assert beta*G/F(16)+step/F(4)<F(1,4)
assert 2**16*step+2**20*step*step<F(1,2**16)
assert 2**40*delta/eta**3<eta/F(1000)
assert 3*eta<(lmax-1)**3
report={'all_exact_checks_passed':True,'constant_exact':str(C),
 'profile_upper':float(fhi),'initialization_relative_margin':float(exp_lower/(2*amp/(dep*R*R))-1),
 'joint_deposit_condition':float(condition),'dimension_fraction':float(margin),
 'terminal_discrepancy_squared':float(terminal),'terminal_displacement_squared':float(terminal/mu),
 'terminal_reserve':float(b+2*AD),'proxy_norm_upper_bound':float(proxy_bound),
 'normalized_witness_cut':str(cut),'minimum_true_low_block_separation':'1/50',
 'high_low_overlap_bound':'2^14 delta','gradient_unit_normalization':'(1-c) g = g/3'}
print(json.dumps(report,indent=2))
