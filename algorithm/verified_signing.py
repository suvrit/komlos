"""Bounded candidate search and an outward-rounded 37.54 output check.

Certificates concern the binary64 matrix actually supplied, without a unit-
column assumption. Each addition is enclosed using numpy.nextafter; no BLAS
dot-product error model is used. IEEE-754 binary64 arithmetic with gradual
underflow is required. The walk fallback is numerical and may report failure.
"""
from fractions import Fraction
import time
import numpy as np
from clw_gsw import GSWWalk
from final_walk import FinalWalk


LIMIT = Fraction(1877, 50)
LIMIT_LOWER = float(LIMIT)
if Fraction.from_float(LIMIT_LOWER) > LIMIT:
    LIMIT_LOWER = np.nextafter(LIMIT_LOWER, -np.inf)


def signing_enclosure(A, eps):
    A, eps = np.asarray(A, dtype=np.float64), np.asarray(eps, dtype=np.float64)
    if A.ndim != 2 or eps.shape != (A.shape[1],) or not np.isfinite(A).all():
        raise ValueError('a finite binary64 matrix and one sign per column are required')
    if not np.all(np.abs(eps) == 1):
        raise ValueError('every output coordinate must be a sign')
    lower = np.zeros(A.shape[0]); upper = lower.copy()
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        for j in range(A.shape[1]):
            term = A[:, j] if eps[j] == 1 else -A[:, j]
            lower = np.nextafter(lower + term, -np.inf)
            upper = np.nextafter(upper + term, np.inf)
    bound = float(np.maximum(np.abs(lower), np.abs(upper)).max(initial=0))
    return lower, upper, dict(certified_bound=bound,
        bound_verified=bool(np.isfinite(bound) and bound < LIMIT_LOWER),
        verification='outward binary64 additions', represented_input='binary64',
        target_exact='1877/50')


def verified_sign(A, seed=0, candidates=8, mode='fast', fallback=True):
    """Try a bounded fast or GSW proposal, then a checked final-profile walk.

    Return only a signing whose row-sum intervals prove the strict 37.54 bound.
    Failure is explicit; this is not a complete implementation of the exact
    fallback theorem. Fast-mode proposal selection minimizes the observed
    row maximum among a fixed number of independent random signings.
    """
    start = time.perf_counter()
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or not np.isfinite(A).all():
        raise ValueError('A must be a finite matrix')
    if mode not in ('fast', 'gsw') or candidates < 0:
        raise ValueError('invalid proposal mode or budget')
    proposals = []
    attempted_walk = None
    if mode == 'gsw':
        eps, attempted_walk = GSWWalk(A, seed=seed).run()
        proposals.append(('quadratic-gsw', eps))
    elif candidates:
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1., 1.], size=(A.shape[1], candidates))
        scores = np.max(np.abs(A @ signs), axis=0, initial=0)
        proposals.append(('random-candidates', signs[:, int(np.argmin(scores))]))
    for method, eps in proposals:
        _, _, cert = signing_enclosure(A, eps)
        if cert['bound_verified']:
            return eps, dict(cert, method=method, candidates=candidates if mode=='fast' else 0,
                             fallback_used=False, exact_fallback_implemented=False,
                             wall_seconds=time.perf_counter()-start,
                             disc=float(np.abs(A@eps).max(initial=0)))
    if not fallback:
        raise RuntimeError('no proposal passed the interval certificate')
    eps, stats = FinalWalk(A, seed=seed).run()
    _, _, cert = signing_enclosure(A, eps)
    if not cert['bound_verified']:
        raise RuntimeError('endpoint fallback did not establish the strict 37.54 bound')
    return eps, dict(cert, method='final37-endpoint', candidates=candidates,
                     fallback_used=True, exact_fallback_implemented=False,
                     walk=stats, wall_seconds=time.perf_counter()-start,
                     disc=float(np.abs(A@eps).max(initial=0)))
