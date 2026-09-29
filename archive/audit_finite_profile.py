#!/usr/bin/env python3
"""High-precision stress checks of the new C^{2,1} Perron lemma.

These numerical checks supplement, and do not replace, the proof. The
curves cross the profile's third-derivative kink at zero, have two nearly
colliding leading eigenvalues, and have noncommuting derivatives.
Requires mpmath.
"""
import json
import mpmath as mp

mp.mp.dps = 140

def col(xs):
    return mp.matrix([mp.mpf(x) for x in xs])

def norm(x):
    return mp.sqrt(sum(t*t for t in x))

def opnorm(x):
    return max(abs(t) for t in mp.eigsy(x, eigvals_only=True))

def profile(x):
    y = abs(x)
    return mp.mpf(275)/34*mp.log((25-17*y)/8) + mp.mpf(11)/2*(y-1)

def positive_cube(x):
    return max(x-1, 0)**3

rows = []
for exponent in (3, 9, 24):
    small = mp.mpf(10)**(-exponent)
    ps = [col(v) for v in [(1,0,0),(0,1,0),(0,0,1),(1,0,1),
                           (0,1,1),(1,1,0),(1,'0.5','0.2')]]
    weights = [mp.mpf(1),1-small/2,mp.mpf('0.04'),small,small/3,small/2,small/5]
    base = small*mp.ones(3,3)
    m0 = base.copy()
    for w,p in zip(weights, ps):
        m0 += w*p*p.T
    scale = mp.mpf('1.002') / mp.eigsy(m0, eigvals_only=True)[2]
    weights = [w*scale for w in weights]
    base *= scale
    m0 *= scale
    vals, vecs = mp.eigsy(m0)
    L = vals[2]
    v = vecs[:,2]
    if sum(v) < 0:
        v = -v
    assert min(v) > 0
    gap = L-vals[1]
    high = [j for j in range(3) if vals[j] > L/4]
    constraints = mp.matrix(len(high), len(ps))
    for r,j in enumerate(high):
        for i,(w,p) in enumerate(zip(weights,ps)):
            constraints[r,i] = w*(p.T*v)[0]*(vecs[:,j].T*p)[0]
    raw = col((1,-2,3,-1,2,4,-3))
    scores = raw - constraints.T * mp.lu_solve(constraints*constraints.T, constraints*raw)
    scores /= max(abs(x) for x in scores)
    assert norm(constraints*scores) < mp.mpf('1e-115')
    hvals = [mp.mpf(i+2)/80 for i in range(7)]
    zvals = [mp.mpf(i+1)/8 for i in range(7)]
    beta = mp.mpf(4)/5
    second_profile_zero = -mp.mpf(187)/50
    E, N = mp.zeros(3,3), mp.zeros(3,3)
    for i,(w,p) in enumerate(zip(weights,ps)):
        E += w*scores[i]*p*p.T
        N += w*(scores[i]**2 + beta*zvals[i]*second_profile_zero*hvals[i]**2)*p*p.T
    assert norm(m0*E-E*m0) > 0
    alpha = (v.T*E*v)[0]
    response = mp.zeros(3,1)
    for j in (0,1):
        u = vecs[:,j]
        response += u*((u.T*E*v)[0]/(L-vals[j]))
    second = (v.T*N*v)[0] + 2*((E*v).T*response)[0]
    nu, kv = min(v), 1+mp.log(1/min(v))
    leakage = mp.sqrt(sum((vecs[:,j].T*E*v)[0]**2 for j in high))
    assert leakage <= gap*nu/64
    assert norm(response) <= 2
    assert max(abs(response[j])/v[j] for j in range(3)) <= 4*kv

    C = mp.matrix(3,len(ps))
    for i,(w,p) in enumerate(zip(weights,ps)):
        C[:,i] = mp.sqrt(w)*p
    rlow = mp.zeros(3,3)
    cutoff = mp.mpf(59)/250
    for j in range(3):
        if vals[j] < cutoff*L:
            u=vecs[:,j]
            rlow += u*u.T/(L-vals[j])
    assert opnorm(C.T*rlow*C) <= cutoff/(1-cutoff)

    step = mp.sqrt(gap)/(64*kv)
    worst_temple_ratio = mp.mpf(0)
    worst_remainder_ratio = mp.mpf(0)
    for sign in (-1,1):
        for fraction in (mp.mpf('0.25'), mp.mpf('0.5'), mp.mpf(1)):
            t = sign*fraction*step
            mt = base.copy()
            for i,(w,p) in enumerate(zip(weights,ps)):
                ell = scores[i]*t + beta*zvals[i]*(profile(hvals[i]*t)-profile(0))
                mt += w*mp.exp(ell)*p*p.T
            newvals = mp.eigsy(mt,eigvals_only=True)
            psi = v+t*response
            rayleigh = (psi.T*mt*psi)[0]/(psi.T*psi)[0]
            assert newvals[1] <= L+alpha*t-gap/2
            assert abs(rayleigh-L-alpha*t) <= 13*t*t
            temple = 512*t**4/gap
            assert -mp.mpf('1e-115') <= newvals[2]-rayleigh <= temple
            worst_temple_ratio = max(worst_temple_ratio, (newvals[2]-rayleigh)/temple)
            phi1 = 3*(L-1)**2*alpha
            phi2 = 3*(L-1)**2*second + 6*(L-1)*alpha**2
            actual_remainder = positive_cube(newvals[2])-positive_cube(L)-phi1*t-phi2*t*t/2
            allowance = 128*abs(t)**3+4096*t**4/gap
            assert actual_remainder <= allowance
            worst_remainder_ratio=max(worst_remainder_ratio,actual_remainder/allowance)
    rows.append({
        'small_parameter': f'1e-{exponent}',
        'perron_gap': mp.nstr(gap,12),
        'minimum_perron_coordinate': mp.nstr(nu,12),
        'tested_signed_steps': 6,
        'worst_temple_ratio': mp.nstr(worst_temple_ratio,12),
        'worst_positive_remainder_ratio': mp.nstr(worst_remainder_ratio,12),
    })
print(json.dumps({'all_checks_passed':True,'precision_digits':mp.mp.dps,'cases':rows},indent=2))
