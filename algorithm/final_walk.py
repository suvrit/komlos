"""Numerical endpoint implementation of the final 37.54 profile.

The high branch uses the normalized coupled response, not a scalar curvature
surrogate. Row-energy constraints are retained. The exact tiny-step fallback
and lazy warm/cold screening are not implemented. An unresolved step aborts.
The optional input filter only removes rows safe for every signing.
"""
from dataclasses import replace
import numpy as np
from clw2 import Walk, U_TOL, REBUILD
from clw_gsw import GSWWalk
from profiles import BELOW67, PROFILES, Profile


class FinalProfile(Profile):
    def energy(self, x):
        distance = 1 - np.abs(np.asarray(x))
        return (519 / 104) * np.log1p(2 * distance) - (173 / 52) * distance

    def first(self, x):
        x = np.asarray(x)
        return -(173 / 26) * x / (3 - 2 * np.abs(x))

    def second(self, x):
        x = np.asarray(x)
        return -(519 / 26) / (3 - 2 * np.abs(x))**2


FINAL37 = FinalProfile(**replace(
    BELOW67, name='final37', radius=7139/5000,
    deletion_fraction=2704/29929, beta=17/25, amplitude=25/4,
    deposit=7463/10000, height=94893/5000, offset=341613/42500,
    spectral_cut=18/25, curvature_factor=float('nan'), terminal_size=336,
    discrepancy_bound=1877/50).__dict__)
# Keep the original profile module unchanged, preserving its benchmark hashes.
PROFILES['final37'] = FINAL37
ROUNDING_WEIGHT = 19 / 6


def weighted_round(A, x, active, weight=ROUNDING_WEIGHT):
    """Conditional expectation for ||A dx||^2 + weight*||dx||^2.

    This routine uses floating point. A final interval check is separate.
    """
    A, x = np.asarray(A, dtype=float), np.asarray(x, dtype=float)
    active = np.asarray(active, dtype=bool)
    if A.ndim != 2 or A.shape[1] != x.size or active.shape != x.shape:
        raise ValueError('incompatible rounding dimensions')
    if (not np.isfinite(A).all() or not np.isfinite(x).all()
            or np.any(np.abs(x) > 1) or np.any(np.abs(x[~active]) != 1)
            or not np.isfinite(weight) or weight <= 0):
        raise ValueError('invalid terminal state or weight')
    eps, partial = x.copy(), np.zeros(A.shape[0])
    for j in np.flatnonzero(active):
        col = A[:, j]
        diagonal, cross = float(col @ col) + weight, float(col @ partial)
        plus, minus = 1 - x[j], -1 - x[j]
        eps[j] = 1 if diagonal*plus**2 + 2*plus*cross <= diagonal*minus**2 + 2*minus*cross else -1
        partial += (eps[j] - x[j]) * col
    dx = eps - x
    energy = float(partial @ partial + weight * (dx @ dx))
    expectation = float(np.sum((1-x[active]**2) *
                               (np.sum(A[:, active]**2, axis=0) + weight)))
    if energy > expectation + 1e-10 * max(1, expectation):
        raise ArithmeticError('terminal expectation check failed')
    return eps, dict(rounding_energy=energy, rounding_expectation=expectation,
                     rounding_weight=weight, rounding_budget=(1+weight)*int(active.sum()))


class MatchedTerminalWalk(Walk):
    """Old profile with the same 336-coordinate, 19/6 terminal objective."""
    def __init__(self, A, **kwargs):
        super().__init__(A, stop_at=336, terminal='conditional', **kwargs)

    def finish(self):
        eps, stats = weighted_round(self.A, self.x, self.alive)
        self.stats.update(stats)
        return eps


def normalized_response(M, diagonal, forcing, positive, negative, cutoff=18/25):
    """Return Q, high-forcing equations, and diagnostics for a dense anchor.

    Q = U-D+2 B1^T S R_K S B1, K = S(M-B)S, S=(L I-B)^(-1/2).
    The computed high equations use K's eigenvectors at cutoff*k_max.
    """
    L = float(np.linalg.eigvalsh(M)[-1])
    gap_diagonal = L - diagonal
    if np.any(gap_diagonal <= 0):
        raise ArithmeticError('normalized response has nonpositive diagonal')
    scaling = 1 / np.sqrt(gap_diagonal)
    K = scaling[:, None] * (M - np.diag(diagonal)) * scaling[None, :]
    values, vectors = np.linalg.eigh(K)
    high = values >= cutoff * values[-1]
    low = ~high
    resolvent = (vectors[:, low] / (values[-1] - values[low])) @ vectors[:, low].T
    scaled_forcing = scaling[:, None] * forcing
    equations = vectors[:, high].T @ scaled_forcing
    Q = positive - np.diag(negative) + 2 * scaled_forcing.T @ resolvent @ scaled_forcing
    Q = (Q + Q.T) / 2
    return Q, equations, dict(normalized_top=float(values[-1]),
                              normalized_high_rank=int(high.sum()))


class FinalWalk(Walk):
    def __init__(self, A, *, filter_rows=True, stop_at=336, **kwargs):
        self.original_A = np.ascontiguousarray(A, dtype=float)
        if self.original_A.ndim != 2 or not np.isfinite(self.original_A).all():
            raise ValueError('A must be a finite matrix')
        if np.any(np.sum(self.original_A**2, axis=0) > 1+1e-10):
            raise ValueError('column norms must be at most one')
        if not 1 <= stop_at <= 336:
            raise ValueError('the final profile must stop at at most 336')
        if filter_rows and ('x0' in kwargs or 'c0' in kwargs):
            raise ValueError('supplied states require filter_rows=False')
        if filter_rows:
            # Retain uncertain rows too: a roundoff cushion makes this test
            # conservative. Every returned signing is checked on the original A.
            n = self.original_A.shape[1]
            magnitude = np.sum(np.abs(self.original_A), axis=1)
            cushion = min(.5, 4 * max(1, n) * np.finfo(float).eps)
            self.kept_rows = np.flatnonzero(magnitude > 37 * (1-cushion))
        else:
            self.kept_rows = np.arange(self.original_A.shape[0])
        self.filter_rows = filter_rows
        super().__init__(self.original_A[self.kept_rows], profile='final37',
                         stop_at=stop_at, terminal='conditional', **kwargs)
        self.stats.update(original_m=self.original_A.shape[0],
                          retained_input_rows=self.kept_rows.size,
                          input_filter=filter_rows, normalized_proxy=True,
                          lazy_screening_implemented=False)

    def row_energy_constraints(self):
        rows = [self.Bm.T @ self.d]
        V = np.flatnonzero(self.alive)
        # trace(B^T B) <= s: below 4095 its high space is empty.
        if V.size >= 4095:
            retained = self.Bm[:, V]
            values, vectors = np.linalg.eigh(retained.T @ retained)
            for v in vectors[:, values >= 4095].T:
                row = np.zeros(self.n); row[V] = v; rows.append(row)
        return rows

    def direction(self, extra_rows):
        return super().direction(extra_rows + self.row_energy_constraints())

    def high_step(self, med, u):
        V = np.flatnonzero(self.alive)
        r = self.r[med]
        M, Z, alpha = self.dense_M(V, med, u)
        values, vectors = np.linalg.eigh(M)
        v = vectors[:, -1] * np.sign(vectors[:, -1].sum())
        Bn = self.Bm[med][:, V] / r[:, None]
        derivative = Z * self.profile.first(self.x[V])[None, :]
        gradients = np.vstack([Bn + derivative, -Bn + derivative])
        weights = np.concatenate([alpha[:, 0], alpha[:, 1]])
        zv = np.tile(Z @ v, 2)
        beta = self.profile.beta
        positive = (gradients * (weights * beta**2 * zv**2)[:, None]).T @ gradients
        negative = -self.profile.second(self.x[V]) * (
            Z.T @ (alpha.sum(1) * beta * (Z @ v)**2))
        forcing = np.vstack([Z, Z]).T @ (
            gradients * (weights * beta * zv)[:, None])
        diagonal = self.profile.deposit * self.P[self.large][:, V].sum(0)
        proxy, high_equations, diagnostic = normalized_response(
            M, diagonal, forcing, positive, negative)
        tight = np.concatenate([u[:, 0], u[:, 1]]) >= -self.profile.protected_threshold
        rows = [self.Bm[self.large][:, V], gradients[tight], self.x[V][None, :], high_equations]
        rows += [g[V][None, :] for g in self.row_energy_constraints()]
        constraints = np.vstack(rows)
        U, singular, _ = np.linalg.svd(constraints.T, full_matrices=True)
        rank = int((singular > singular.max(initial=0)*1e-11).sum())
        kernel = U[:, rank:]
        curvature, directions = np.linalg.eigh(kernel.T @ proxy @ kernel)
        keep = curvature < self.eta
        if not keep.any():
            raise RuntimeError('no resolved negative subspace; exact fallback unavailable')
        hv = kernel @ (directions[:, keep] @ self.rng.choice([-1., 1.], int(keep.sum())))
        h = np.zeros(self.n); h[V] = hv / np.linalg.norm(hv)
        residual = float(np.abs(constraints @ h[V]).max(initial=0))
        if residual > 1e-8:
            raise ArithmeticError('high-branch feasibility residual too large')
        self.stats.update(diagnostic)
        self.stats['negative_dimension'] = int(keep.sum())
        self.stats['high_constraint_residual'] = residual
        self.stats['tight_events'] += int(tight.sum())
        self.stats['max_tight'] = max(self.stats['max_tight'], int(tight.sum()))
        best = None
        for sign in (1., -1.):
            sh = sign*h[V]; xv = self.x[V]
            with np.errstate(divide='ignore', invalid='ignore'):
                room = np.where(sh > 0, (1-xv)/sh,
                                np.where(sh < 0, (1+xv)/-sh, np.inf))
            cube = float(room.min())
            barrier = self.profile_barrier_limit(h, sign, cube, med)
            t = min(cube, barrier); kind = 'freeze' if t == cube else 'barrier'
            for _ in range(60):
                self.stats['high_trials'] += 1
                xn = np.clip(self.x + sign*t*h, -1, 1)
                if kind == 'freeze':
                    j = V[np.argmin(room)]; xn[j] = np.sign(xn[j])
                dn = self.d + self.Bm @ (xn-self.x)
                energy = self.G + self.P @ (self.profile.energy(xn)-self.profile.energy(self.x))
                base = energy[med]/r**2 + self.profile.offset
                un = np.stack([(dn[med]-self.profile.height)/r+base,
                               (-dn[med]-self.profile.height)/r+base], 1)
                if un.max(initial=-np.inf) <= U_TOL:
                    phi = max(self.certified_L(V, med, un)-1, 0)**3-self.eta*(xn@xn)
                    if phi <= self.Bphi:
                        if best is None or t > best[0]:
                            best = (t, xn, dn, energy, phi, kind)
                        break
                t /= 2; kind = 'short'
        if best is None:
            raise RuntimeError('no resolved final-profile endpoint; exact fallback unavailable')
        _, self.x, self.d, self.G, self.Bphi, kind = best
        self.stats['steps'] += 1; self.stats['high_steps'] += 1
        self.stats[('freeze' if kind == 'freeze' else 'barrier' if kind == 'barrier' else 'spectral')+'_steps'] += 1
        self.cleanup()
        _, after = self.barriers()
        if after.max(initial=-np.inf) > U_TOL:
            raise ArithmeticError('final-profile cleanup violated a barrier')
        if self.stats['steps'] % REBUILD == 0:
            self.refresh()

    def finish(self):
        eps, stats = weighted_round(self.A, self.x, self.alive)
        self.stats.update(stats)
        return eps

    def run(self, max_steps=None):
        # With no surviving rows any signs are safe. Preserve all n columns.
        if self.m == 0:
            eps = np.ones(self.n)
            stats = dict(self.stats, steps=0, tracking_identity_err=0.,
                         high_branch=False, profile='final37', terminal_coordinates=0)
        else:
            eps, stats = super().run(max_steps=max_steps)
        discrepancy = float(np.abs(self.original_A @ eps).max(initial=0))
        stats.update(disc=discrepancy, n=self.n, m=self.original_A.shape[0],
                     row_energy=float(self.d@self.d),
                     discrepancy_bound=self.profile.discrepancy_bound,
                     within_bound_numerically=discrepancy < self.profile.discrepancy_bound)
        return eps, stats


class FinalGSWWalk(FinalWalk, GSWWalk):
    """The final profile with constrained GSW directions and its own CE finish."""

