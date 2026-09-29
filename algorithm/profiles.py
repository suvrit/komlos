"""Retained-row parameters; the original quadratic profile remains the default.

These functions use binary64 arithmetic. They implement formulas from the
papers, not the papers' rational enclosures or exact rank decisions.
"""
from dataclasses import dataclass, replace
import math
import numpy as np


@dataclass(frozen=True)
class Profile:
    name: str
    radius: float
    deletion_fraction: float
    theta: float
    beta: float
    amplitude: float
    deposit: float
    height: float
    offset: float
    energy_coefficient: float
    ridge_denominator: int
    spectral_cut: float
    curvature_factor: float
    protected_threshold: float
    spectral_ceiling: float
    terminal_size: int
    discrepancy_bound: float
    nonquadratic: bool = False

    def energy(self, x):
        x = np.asarray(x)
        if not self.nonquadratic:
            return 1 - x * x
        # log1p is accurate near the boundary, where the energy vanishes.
        distance = 1 - np.abs(x)
        return (275 / 34) * np.log1p((17 / 8) * distance) - 5.5 * distance

    def first(self, x):
        x = np.asarray(x)
        if not self.nonquadratic:
            return -2 * x
        return -187 * x / (2 * (25 - 17 * np.abs(x)))

    def second(self, x):
        x = np.asarray(x)
        if not self.nonquadratic:
            return np.full_like(x, -2, dtype=float)
        return -4675 / (2 * (25 - 17 * np.abs(x)) ** 2)


QUADRATIC = Profile(
    "quadratic", 1.5, .01, 1.01, 2 / 7, 32, 1 / 128,
    75, 17.5, 3.5, 4096, .2, 279 / 79, .01, 9 / 8, 99,
    105 + 3 * math.sqrt(99),
)
BELOW67 = Profile(
    "below67", 1.49, 4 / 121, 1.001, .8, 36, 4 / 625,
    34.45, 8.72, 1, 2**40, 47 / 200, 309 / 191, 1 / 200,
    257 / 256, 512, 1271 / 25 + (149 / 100) * math.sqrt(117), True,
)
# The coupled response estimate also improves the old profile's high-branch
# curvature proxy; it does not require changing its discrepancy analysis.
QUADRATIC_COUPLED = replace(QUADRATIC, name="quadratic-coupled", curvature_factor=121 / 79)
PROFILES = {p.name: p for p in (QUADRATIC, QUADRATIC_COUPLED, BELOW67)}


def conditional_round(A, x, active):
    """Greedily round with ||A dx||^2 + 4||dx||^2 <= its initial expectation.

Frozen signs are left unchanged. Choosing +1 on ties makes this routine
deterministic. This is a floating-point implementation of the exact rational
conditional-expectation rule in the below-67 supplement.
"""
    A, x = np.asarray(A, dtype=float), np.asarray(x, dtype=float)
    active = np.asarray(active, dtype=bool)
    if active.shape != x.shape or A.shape[1] != x.size:
        raise ValueError("incompatible rounding dimensions")
    if not np.isfinite(A).all() or not np.isfinite(x).all() or np.any(np.abs(x) > 1):
        raise ValueError("rounding requires finite data in the cube")
    if np.any(np.abs(x[~active]) != 1):
        raise ValueError("inactive coordinates must already be signs")
    out = x.copy()
    partial = np.zeros(A.shape[0])
    for j in np.flatnonzero(active):
        col = A[:, j]
        diagonal = float(col @ col) + 4
        cross = float(col @ partial)
        plus, minus = 1 - x[j], -1 - x[j]
        cost_plus = diagonal * plus**2 + 2 * plus * cross
        cost_minus = diagonal * minus**2 + 2 * minus * cross
        out[j] = 1 if cost_plus <= cost_minus else -1
        partial += (out[j] - x[j]) * col
    dx = out - x
    energy = float(partial @ partial + 4 * (dx @ dx))
    budget = float(np.sum((1 - x[active] ** 2) *
                         (np.sum(A[:, active] ** 2, axis=0) + 4)))
    if energy > budget + 1e-10 * max(1, budget):
        raise ArithmeticError("terminal conditional-expectation check failed")
    return out, {"rounding_energy": energy, "rounding_expectation": budget,
                 "rounding_budget": 5 * int(active.sum())}
