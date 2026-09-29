"""Exact endpoint envelope and first-phase schedule for the final profile."""
from fractions import Fraction as F
from math import factorial
import json

R, beta, H0 = F(7139,5000), F(17,25), F(94893,5000)
a0, b, amplitude, deposit = F(5389,2500), F(341613,42500), F(25,4), F(7463,10000)
tracked = F(3,5)
exponent = beta*((H0-tracked)/R-a0-b)
assert exponent > 0 and beta*(H0-tracked) > 2*R
taylor = sum((exponent**j/factorial(j) for j in range(32)), F())
envelope = F(1,2**40)+max(deposit,2*amplitude/(R*R*taylor))
assert envelope < F(993,1000) < 1

def ceil_log2(q):
    k=0
    while F(2)**k < q: k+=1
    return k

records=[]
for n in (640,10000):
    J=1+ceil_log2(F(n)); eta=F(1,2**40*J*n)
    H=ceil_log2(4/eta); cap=F(1,2**56)
    t=F(1); exponent_two=0
    while t>cap or t>cap*eta or H*H*t*t>cap*cap*eta:
        t/=2; exponent_two-=1
    assert t<=cap and t<=cap*eta and H*H*t*t<=cap*cap*eta
    assert 2*t>cap or 2*t>cap*eta or H*H*(2*t)**2>cap*cap*eta
    records.append(dict(n=n,K=1,tau_power_of_two=exponent_two,
                        displacement_upper_power_of_two=exponent_two-4))
report=dict(all_exact_checks_passed=True, tracked_sum_bound=str(tracked),
            spectral_upper=float(envelope), spectral_upper_exact=str(envelope),
            barrier_upper=float((tracked-H0)/R+a0+b), schedules=records)
if __name__=='__main__': print(json.dumps(report,indent=2))
