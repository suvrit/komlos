"""Certified walk with Gram-Schmidt-walk directions restricted to the constraint kernel.

Only the direction changes: u = argmin ||A u||^2 + ridge ||u||^2 subject to
u_p = 1, u_frozen = 0, C u = 0, where C holds the large rows, the nearly tight
barrier gradients and x_V.  All step limits, certificates and cleanup are those
of clw2.Walk, so every guarantee of the certified walk is unchanged.
"""
import numpy as np
from clw2 import Walk, D_STAR


class GSWWalk(Walk):
    def __init__(self, A, ridge=1e-10, **kw):
        n = A.shape[1]
        self.ridge = ridge
        self.Vg = np.arange(n)
        self.Ginv = np.linalg.inv(A.T @ A + ridge * np.eye(n))
        self.pivot = None
        super().__init__(A, **kw)

    def cleanup(self, initial=False):
        before = self.alive.copy()
        super().cleanup(initial)
        if not hasattr(self, "Ginv"):
            return
        gone = np.flatnonzero(before & ~self.alive)
        for j in sorted(np.flatnonzero(np.isin(self.Vg, gone)), reverse=True):
            keep = np.r_[0:j, j + 1:self.Vg.size]
            G = self.Ginv
            self.Ginv = G[np.ix_(keep, keep)] - np.outer(G[keep, j], G[j, keep]) / G[j, j]
            self.Vg = self.Vg[keep]

    def direction(self, extra_rows):
        V = self.Vg
        if self.pivot is None or not self.alive[self.pivot]:
            self.pivot = int(self.rng.choice(V))
        k = int(np.flatnonzero(V == self.pivot)[0])
        rows = [self.Bm[self.large][:, V]] if self.large.size else []
        rows += [g[V][None, :] for g in extra_rows]
        C = np.vstack(rows) if rows else np.zeros((0, V.size))
        C = C[np.linalg.norm(C, axis=1) > 0]
        E = np.hstack([np.eye(V.size)[:, [k]], C.T])
        Y = self.Ginv @ E
        f = np.zeros(E.shape[1])
        f[0] = 1.0
        coef = np.linalg.lstsq(E.T @ Y, f, rcond=None)[0]
        u = Y @ coef
        h = np.zeros(self.n)
        h[V] = u / np.linalg.norm(u)
        resid = float(np.abs(C @ h[V]).max()) if C.size else 0.0
        if resid > 1e-8 or not np.isfinite(resid):
            return super().direction(extra_rows)       # fall back to a random kernel vector
        return h

    def finish(self):
        """Plain Gram-Schmidt walk on the remaining coordinates.

        A GSW finish may move a coordinate by up to 2, so the large-row terminal bound
        is not automatic; the result is kept only if it passes the exact final check
        ||A eps||_inf < D_*, otherwise nearest-sign rounding (always certified) is used.
        """
        # The old D_* acceptance test cannot certify the stronger profile.
        # Retain its conditional-expectation finish and remaining-mass bound.
        if self.terminal_rule == 'conditional':
            return super().finish()
        x = self.x.copy()
        V, G = self.Vg.copy(), self.Ginv.copy()
        p = None
        while V.size:
            if p is None or p not in set(V.tolist()):
                p = int(self.rng.choice(V))
            k = int(np.flatnonzero(V == p)[0])
            u = G[:, k] / G[k, k]
            xv = x[V]
            with np.errstate(divide="ignore", invalid="ignore"):
                up = np.where(u > 0, (1 - xv) / u, np.where(u < 0, (1 + xv) / -u, np.inf))
                um = np.where(u < 0, (1 - xv) / -u, np.where(u > 0, (1 + xv) / u, np.inf))
            dp, dm = up.min(), um.min()
            x[V] = np.clip(xv + (dp if self.rng.random() < dm / (dp + dm) else -dm) * u, -1, 1)
            fz = np.flatnonzero(np.abs(x[V]) >= 1 - 1e-12)
            x[V[fz]] = np.sign(x[V[fz]])
            for j in sorted(fz, reverse=True):
                keep = np.r_[0:j, j + 1:V.size]
                G = G[np.ix_(keep, keep)] - np.outer(G[keep, j], G[j, keep]) / G[j, j]
                V = V[keep]
        eps = np.where(x >= 0, 1.0, -1.0)
        if np.abs(self.A @ eps).max() < D_STAR:
            return eps
        self.stats["finish_fallback"] = True
        return super().finish()
