#!/usr/bin/env python3
"""Exact rational parameter certificates for the constructive <67 bound.

Uses only the Python standard library. These are finite inequality
certificates; the accompanying paper proves their algorithmic relevance.
"""
from fractions import Fraction as F
from math import factorial, sqrt
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
import json

R, delta, d, c = F(149, 100), F(4, 121), F(2, 11), F(17, 25)
beta, amp, deposit = F(4, 5), F(36), F(4, 625)
a0, b, height = F(93, 25), F(218, 25), F(689, 20)
theta, lmax = F(1001, 1000), F(257, 256)
cut, true_cut, lmin = F(47, 200), F(59, 250), F(999999, 1000000)
eta_max = F(1, 2**40)
precision_max = eta_max**5 / 2**100

# log(25/8) = 2 atanh(17/33). A positive series and geometric
# upper bound for its tail provide a two-sided rational enclosure.
t = F(17, 33)
terms = 32
log_lower = 2 * sum((t**(2*k+1) / (2*k+1) for k in range(terms)), F(0))
log_upper = log_lower + 2*t**(2*terms+1) / ((2*terms+1)*(1-t*t))
profile_lower = F(275, 34)*log_lower - F(11, 2)
profile_upper = F(275, 34)*log_upper - F(11, 2)
assert 0 < profile_lower < profile_upper < a0
assert beta*(b-a0) == 4

# Initialization: a finite positive Taylor sum is already enough.
exponent = beta*(height/R-a0-b)
exp_lower = sum((exponent**k / factorial(k) for k in range(40)), F(0))
required = 2*amp/(deposit*R*R)
assert exp_lower > required
assert beta*height/R > 2
assert -height/R+a0+b < -10

# Pointwise optimal ratio and resulting positive-direction count.
rho = beta*d/c
gamma = (1+true_cut)/(1-true_cut)
assert rho == F(40, 187)
assert gamma == F(309, 191)
assert rho*gamma == F(12360, 35717)

large_cost = 1/R**2
protected_cost = lmax*theta**4 / (amp*(1-beta/F(100)))
spectral_base = (delta+deposit/lmin)/cut
spectral_safe = (delta+(deposit+eta_max+precision_max)/lmin) / (cut*(1-precision_max/lmin))
assert spectral_safe-spectral_base < F(1, 10000)
assert 1-2*precision_max > lmin
assert (true_cut-cut)*lmin-cut*precision_max > F(1, 2000)
dimension = 1-large_cost-protected_cost-spectral_base-F(1, 10000)-gamma*rho-F(1, 513)
assert dimension > F(1, 200)

# C = 1271/25 + (149/100)*sqrt(117) < 67, without floats.
base = height+11*R
assert base == F(1271, 25)
assert (F(67)-base)**2 > 117*R**2
assert F(67) > base

# Terminal reserve: b+11+sqrt(117)>sqrt(640).
# The weaker rational sqrt(117)>10 suffices.
assert 10**2 < 117
assert (b+21)**2 > 640
# Large-row CE discrepancy sqrt(2560) is below even base(C).
assert base**2 > 2560

# Scalar derivative and normalization bounds used in the finite walk.
max_fp, max_fpp, max_fppp = F(187, 16), F(4675, 128), F(79475, 512)
gradient_bound = 1+d*max_fp
assert gradient_bound == F(25, 8)
assert beta*gradient_bound == F(5, 2)
assert beta*max_fpp/8**2 < F(1, 2)
assert beta*max_fppp/8**3 < F(1, 4)
assert beta*max_fpp*delta < 1
assert gamma*(beta*gradient_bound)**2*(lmax+eta_max) + (lmax+eta_max) < 16

# Finite-step coefficients, using ||h||>=1/32.
step_cap = F(1, 2**40)
assert F(19, 10)/32**2 > F(1, 1024)
assert 128*step_cap+4096*step_cap**2 < F(1, 2**16)
assert F(2, 2**16)-F(1, 2048) < -F(1, 4096)
assert 3*eta_max < F(1, 256)**3

def decimal_bound(x, rounding):
    with localcontext() as context:
        context.prec = 100
        context.rounding = rounding
        value = (Decimal(x.numerator)/Decimal(x.denominator)).quantize(Decimal('1e-40'))
        return format(value, '.40f')

report = {
    "all_exact_assertions_passed": True,
    "profile_max_interval": [decimal_bound(profile_lower, ROUND_FLOOR),
                             decimal_bound(profile_upper, ROUND_CEILING)],
    "initialization_exponent": float(exponent),
    "initialization_taylor_margin": float(exp_lower/required-1),
    "dimension_fraction_after_all_budgets": float(dimension),
    "required_dimension_fraction": 1/200,
    "curvature_positive_fraction": float(gamma*rho),
    "barrier_height": float(height),
    "deletion_allowance": float(R)*(11+sqrt(117)),
    "discrepancy_constant": float(base)+float(R)*sqrt(117),
    "terminal_movement_bound": sqrt(640),
    "terminal_movement_reserve": float(b)+11+sqrt(117),
}
if __name__ == '__main__':
    print(json.dumps(report, indent=2))
