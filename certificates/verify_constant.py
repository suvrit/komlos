#!/usr/bin/env python3
"""Exact rational certificate for the September 28 final certificate C=37.54.

This verifies the finite scalar hypotheses for the generalized joint spectral-inertia
walk. The new count and finite-walk lemmas are proved in the accompanying
mathematical note; scalar certification does not replace those proofs.
Only the Python standard library is needed.
"""
from fractions import Fraction as F
from math import factorial
from decimal import Decimal, localcontext
import json

R, d, c = F(7139,5000), F(52,173), F(2,3)
Delta, beta = d*d, F(17,25)
amp, deposit = F(25,4), F(7463,10000)
a0, b, height = F(5389,2500), F(341613,42500), F(94893,5000)
theta, lmax, lmin = F(1001,1000), F(257,256), F(999999,1000000)
cut, true_cut = F(18,25), F(3,4)
stop, rounding_weight = 336, F(19,6)
eta_max = F(1,2**40)
precision_max = eta_max**5/2**100

# Exact rational logarithm enclosure: log(3) = 2 atanh(1/2).
t = F(1,2)
N = 48
log_lower = 2*sum((t**(2*k+1)/(2*k+1) for k in range(N)),F(0))
log_upper = log_lower + 2*t**(2*N+1)/((2*N+1)*(1-t*t))
profile_lower = F(519,104)*log_lower-F(173,52)
profile_upper = F(519,104)*log_upper-F(173,52)
assert 0 < profile_lower < profile_upper < a0
assert beta*(b-a0)==4

# New row initialization, including monotonicity in the row scale.
exponent = beta*(height/R-a0-b)
exp_lower = sum((exponent**k/factorial(k) for k in range(48)),F(0))
required = 2*amp/(deposit*R*R)
assert exp_lower > required
assert beta*height/R > 2
assert -height/R+a0+b < -3

# Tangent to q(lambda)=(1-lambda)/(1+lambda) at lambda=1/4.
# q(lambda) - (23/25-32 lambda/25)
# = 32*(lambda-1/4)^2/(25*(1+lambda)).
rho = beta*d/c
assert rho == F(1326,4325)
tangent_intercept,tangent_slope=F(23,25),F(32,25)
assert cut > tangent_intercept/tangent_slope == F(23,32)
assert cut-tangent_intercept/tangent_slope == F(1,800)
assert true_cut > cut
spectral_trace = Delta
assert 0 < deposit < lmin
# Charge protected pairs, large rows, and normalized response together.
a_joint,b_joint=F(25,23),F(32,23)
protected_coefficient=theta**4/(amp*(1-beta/100))
deposit_joint_coefficient = deposit*R*R*(protected_coefficient+b_joint*Delta/(lmin-deposit))
assert deposit_joint_coefficient < 1
# Ridge and diagonalization errors remain in the explicit 1/10000 allowance.
assert b_joint*eta_max/(lmin-deposit) < F(1,2**30)
assert 18*eta_max+2**12*precision_max < F(1,10000)
joint_loss = a_joint*rho+b_joint*spectral_trace
protected = protected_coefficient*(lmax-deposit)

# The new screened walk retains both exact tangencies, removes Q's high
# spectrum as before, and leaves > s/400 feasible negative directions.
large = 1/R**2
numerical_slack = F(1,10000)
single_equation = F(1,stop+1)
row_energy_high_rank = F(1,4094)
dimension = 1-large-protected-joint_loss-numerical_slack-2*single_equation-row_energy_high_rank
assert dimension > F(1,400)
assert 1-large-protected_coefficient*lmax-2*single_equation-row_energy_high_rank > F(1,400)
assert 1-2*precision_max > lmin
assert (true_cut-cut)*lmin-cut*precision_max > F(29,1000)

# Exact deletion cost: the 52-165-173 Pythagorean triple.
sqrt_one_minus_delta = F(165,173)
assert sqrt_one_minus_delta**2 == 1-Delta
deletion_ratio = (1+sqrt_one_minus_delta)/d
assert deletion_ratio == F(13,2)
constant = height+2*R*deletion_ratio
assert constant == F(1877,50)
assert 37 < constant < 38

# Terminal deterministic conditional-expectation rounding.
terminal_energy = (1+rounding_weight)*stop
movement_squared = terminal_energy/rounding_weight
assert terminal_energy == 1400
assert constant*constant > terminal_energy
assert (b+2*deletion_ratio)**2 > movement_squared

# Profile derivatives and conservative flattening normalization.
max_fp = c/(d*(1-c))
max_fpp = c/(d*(1-c)**2)
max_fppp = 2*c*c/(d*(1-c)**3)
gradient_bound = 1+d*max_fp
assert gradient_bound == 3
assert beta*gradient_bound == F(51,25)
flat_coordinate_factor = F(16)
assert beta*max_fpp/flat_coordinate_factor**2 < F(1,2)
assert beta*max_fppp/flat_coordinate_factor**3 < F(1,4)
assert beta*max_fpp*Delta == F(5304,4325)
assert beta*max_fpp*Delta < F(9,4)
assert beta*gradient_bound < F(25,8)
# With cutoff3/4, the full coupled Hessian proxy has norm<128.
assert 9*(beta*gradient_bound)**2*(lmax+eta_max) + beta*max_fpp*Delta*(lmax+eta_max) < 128

# Larger absolute constants from the general-cutoff finite Perron lemma.
step_cap=F(1,2**56)
assert 2**16*step_cap+2**20*step_cap**2 < F(1,2**16)
assert F(19,10)/32**2 > F(1,1024)
assert 3*eta_max < F(1,256)**3

report={
 'all_exact_assertions_passed': True,
 'parameter_fractions': {k:str(v) for k,v in {
  'R':R,'d':d,'Delta':Delta,'c':c,'beta':beta,'A0':amp,'B0':deposit,
  'a0':a0,'b':b,'H0':height,'spectral_witness_cut':cut,'spectral_true_cut':true_cut,
  'rounding_weight':rounding_weight,'tangent':F(1,4),
  'joint_rho_coefficient':a_joint,'joint_trace_coefficient':b_joint}.items()},
 'terminal_cutoff': stop,
 'profile_max_interval': [float(profile_lower),float(profile_upper)],
 'initialization_exponent':float(exponent),
 'initialization_taylor_relative_margin':float(exp_lower/required-1),
 'joint_large_spectral_trace_fraction':float(spectral_trace),
 'deposit_joint_coefficient':float(deposit_joint_coefficient),
 'joint_spectral_inertia_fraction':float(joint_loss),
 'joint_protected_fraction':float(protected),
 'protected_coefficient':float(protected_coefficient),
 'dimension_fraction_after_all_screening_budgets':float(dimension),
 'required_dimension_fraction':float(F(1,400)),
 'terminal_discrepancy_squared':int(terminal_energy),
 'terminal_movement_squared':str(movement_squared),
 'terminal_movement_reserve':float(b+2*deletion_ratio),
 'deletion_allowance':str(2*R*deletion_ratio),
 'discrepancy_constant_exact':str(constant),
 'discrepancy_constant':float(constant),
 'harmless_row_l1_threshold':37,
 'scope':'Scalar hypotheses certified; requires the proved generalized joint-inertia, fully joint protected/large-row charge, and finite-walk lemmas.'
}
if __name__=='__main__':print(json.dumps(report,indent=2))
