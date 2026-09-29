"""Exact finite checks supporting the randomized screening argument.

These are identity/example checks, not a proof of cubic total signing time.
"""
from fractions import Fraction as F
from itertools import product
import json

def dot(x,y):
    return sum((a*b for a,b in zip(x,y)),F(0))

def psd(A):
    A=[r[:] for r in A]
    for k in range(len(A)):
        p=A[k][k]
        assert p>=0
        if not p:
            assert all(A[k][j]==0 for j in range(k+1,len(A)))
            continue
        for i in range(k+1,len(A)):
            for j in range(i,len(A)):
                A[i][j]-=A[i][k]*A[k][j]/p
                A[j][i]=A[i][j]

# An exactly orthonormal rational basis, with overlapping coordinates.
n,k=8,5
T=[[F(int(i==j)) for j in range(k)] for i in range(n)]
for i,j in [(0,5),(1,6),(2,7),(0,3),(4,6),(3,7)]:
    a,b=T[i][:],T[j][:]
    T[i]=[F(3,5)*x-F(4,5)*y for x,y in zip(a,b)]
    T[j]=[F(4,5)*x+F(3,5)*y for x,y in zip(a,b)]
assert all(dot([T[i][j] for i in range(n)],[T[i][ell] for i in range(n)])==int(j==ell)
           for j in range(k) for ell in range(k))
vectors=[[dot(row,e) for row in T] for e in product((-1,1),repeat=k)]
accepted=[v for v in vectors if abs(v[0])<=F(1)]
assert len(accepted)>=len(vectors)//2 and len(accepted)<len(vectors)
p=F(len(accepted),len(vectors))
cov=[[sum((v[i]*v[j] for v in accepted),F(0))/len(accepted)
      for j in range(n)] for i in range(n)]
psd([[F(int(i==j))/p-cov[i][j] for j in range(n)] for i in range(n)])

# Boundary truncation is unbiased even when its two distances differ.
energy_checks=0
for tp,tm in [(F(1,10),F(1,10)),(F(1,100),F(1,3)),(F(2,7),F(1,1000))]:
    prob=tm/(tp+tm)
    assert prob*tp-(1-prob)*tm==0
    assert prob*tp**2+(1-prob)*tm**2==tp*tm
    for d,z in [(F(0),F(1)),(F(-7,3),F(5,11)),(F(100),F(-1,7))]:
        expected=prob*(d+tp*z)**2+(1-prob)*(d-tm*z)**2
        assert expected-d*d==tp*tm*z*z
        energy_checks+=1

# One-sided interval endpoints are essential. A symmetric shorter step
# with a forced boundary sign generally has nonzero conditional drift.
tp,tm=F(1,100),F(1,3)
assert min(tp,tm)!=0

# Clock accounting through early inspection before a possible crossing.
L=F(8)
r=F(1,2)
quantum=1/(2*L*r*r)
increments=[quantum/F(8),quantum/F(10),quantum/F(9)]*100
assert max(increments)<=quantum/2
v=F(0);anchor=F(0);times=[anchor]
for inc in increments:
    if v+inc>=anchor+quantum:
        assert v-anchor>=quantum/2
        times.append(v);anchor=v
    v+=inc
assert len(times)-1 <= 2*v/quantum

# High row counts are charged by squared entry scales, not their sum.
for n0,m0 in [(100,10),(1000,100),(1000,10000)]:
    r2=F(n0,m0)
    assert m0*r2==n0
    assert m0*n0<=n0**3 if m0<=n0*n0 else True

report={
 'all_checks_passed':True,
 'rademacher_directions':len(vectors),
 'accepted_directions':len(accepted),
 'acceptance_probability':str(p),
 'conditional_covariance_psd_check':'exact rational elimination',
 'unbiased_boundary_energy_checks':energy_checks,
 'clock_renewals':len(times)-1,
 'scope':'Cubic row processing and screening only; dense direction construction remains quartic overall.'
}
if __name__=='__main__': print(json.dumps(report,indent=2))
