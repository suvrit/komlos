"""Floating-point long-step prototype with selectable retained-row profiles.

The default quadratic parameters reproduce the original clw.py experiment.
profile="below67" selects the optimized profile and conditional-expectation
finish. Neither option implements the exact finite fallback; failures abort.
The legacy stats["certified"] field is only a numerical final-discrepancy check.
Constructor/factorization time must be included by any external benchmark.

For the default quadratic profile, the implementation refinements are:
  * d_i, G_i, q_i are maintained incrementally (full recomputation every
    REBUILD steps, with the drift recorded);
  * the large-row constraints are kept as K = (B_L B_L^T)^{-1}, updated by
    Sherman-Morrison when a coordinate freezes and by a Schur downdate when a
    large row turns medium (rebuilt on any residual failure);
  * nearly tight gradients and x_V are temporary rows, orthogonalized after
    the large-row projection.
"""
import math
import time
import numpy as np
from profiles import PROFILES, conditional_round

THETA = 1.01
R_MAX = 1.5
BETA = 2.0 / 7.0
A_TIGHT = 1.0 / 100.0
D_STAR = 105 + 3 * math.sqrt(99)
U_TOL = 1e-12
REBUILD = 50
HIGH = 0.9                       # use the high-branch direction when Lambda >= HIGH


def r_candidate(q, amax, theta=THETA):
    q = np.asarray(q, float)
    amax = np.asarray(amax, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        l = np.ceil(np.log(np.sqrt(q) / amax) / math.log(theta))
    l = np.where(np.isfinite(l), np.maximum(l, 0), 0)
    r = amax * theta ** l
    lo = r * r < q
    r = np.where(lo, r * theta, r)
    l = np.where(lo, l + 1, l)
    hi = (l > 0) & ((r / theta) ** 2 >= q)
    return np.where(hi, r / theta, r)


class Walk:
    def __init__(self, A, seed=0, coin=True, x0=None, c0=None,
                 profile="quadratic", stop_at=None, terminal=None):
        self.A = np.ascontiguousarray(A, dtype=float)
        if self.A.ndim != 2 or not np.isfinite(self.A).all():
            raise ValueError("A must be a finite matrix")
        if np.any(np.sum(self.A ** 2, axis=0) > 1 + 1e-10):
            raise ValueError("column norms must be at most one")
        self.profile = PROFILES[profile]
        self.terminal_rule = terminal or ('conditional' if self.profile.nonquadratic else 'nearest')
        if self.terminal_rule not in ('nearest', 'conditional'):
            raise ValueError("terminal must be nearest or conditional")
        if self.profile.nonquadratic and self.terminal_rule != 'conditional':
            raise ValueError("below67 requires conditional-expectation rounding")
        terminal_limit = 512 if self.terminal_rule == 'conditional' else 99
        self.stop_at = terminal_limit if stop_at is None else int(stop_at)
        if not 1 <= self.stop_at <= terminal_limit:
            raise ValueError("stop_at must be between 1 and the terminal rule's limit")
        m, n = self.A.shape
        self.m, self.n = m, n
        self.rng = np.random.default_rng(seed)
        self.coin = coin
        self.x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
        if self.x.shape != (n,) or not np.isfinite(self.x).all() or np.any(np.abs(self.x) > 1):
            raise ValueError("x0 must be a finite cube point")
        self.alive = np.ones(n, dtype=bool)
        self.Bm = self.A.copy()                  # retained entries, zero elsewhere
        self.P = self.Bm ** 2
        self.cnt = (self.Bm != 0).sum(1)
        self.q = self.P.sum(1)
        self.c = np.zeros(m) if c0 is None else np.array(c0, dtype=float)
        self.d = self.c + self.Bm @ self.x
        self.G = self.P @ self.profile.energy(self.x)
        self.cls = np.zeros(m, dtype=np.int8)    # 2 large, 1 medium, 0 completed
        self.cls[self.q > self.profile.radius ** 2] = 2
        self.cls[(self.cnt > 0) & (self.q <= self.profile.radius ** 2)] = 1
        self.r = np.zeros(m)
        med = self.cls == 1
        if med.any():
            self.r[med] = np.minimum(self.profile.radius, r_candidate(
                self.q[med], np.sqrt(self.P[med].max(1)), self.profile.theta))
        self.deleted = []
        self.J = 1 + math.ceil(math.log2(max(1, n)))
        self.S = max(1, n)
        self.eta = 1.0 / (self.profile.ridge_denominator * self.J * self.S)
        self.large = np.flatnonzero(self.cls == 2)
        self.K = None
        self.y = None
        self.drift = 0.0
        self.stats = dict(steps=0, freeze_steps=0, barrier_steps=0, spectral_steps=0,
                          max_Lambda=0.0, max_u=-np.inf, tight_events=0, max_tight=0,
                          max_large=int(self.large.size), max_large_resid=0.0,
                          high_branch=False, K_rebuilds=0, lam_iters=0,
                          high_steps=0, high_trials=0, max_Lambda_after_start=0.0,
                          numerical_only=True, fallback_implemented=False)
        self.rebuild_K()
        self.cleanup(initial=True)
        med, u = self.barriers()
        lam0 = self.lambda_upper(med, u)
        if lam0 >= HIGH:                      # tighter certificate for supplied high-branch starts
            lam0 = min(lam0, self.certified_L(np.flatnonzero(self.alive), med, u))
        self.Bphi = max(lam0 - 1, 0) ** 3 - self.eta * (self.x @ self.x)

    # ---- large-row Gram inverse ---------------------------------------------
    def rebuild_K(self):
        self.large = np.flatnonzero(self.cls == 2)
        if self.large.size == 0:
            self.K = np.zeros((0, 0))
            self.K_full_rank = True
            return
        BL = self.Bm[self.large]
        Gm = BL @ BL.T
        w, U = np.linalg.eigh(Gm)
        keep = w > w.max() * 1e-12
        self.K = (U[:, keep] / w[keep]) @ U[:, keep].T     # pseudo-inverse
        self.K_full_rank = bool(keep.all())
        self.stats["K_rebuilds"] += 1

    def K_freeze(self, cols):
        """B_L loses the columns in cols (Sherman-Morrison per column)."""
        if self.large.size == 0:
            return
        if not self.K_full_rank:
            self.K = None
            return
        for j in cols:
            b = self.Bm[self.large, j]
            if not b.any():
                continue
            Kb = self.K @ b
            den = 1.0 - b @ Kb
            if den < 1e-8:
                # Cleanup must first transfer the original column to c and
                # subtract its mass. Zeroing Bm here loses that contribution.
                self.K = None
                return
            self.K += np.outer(Kb, Kb) / den

    def K_remove_rows(self, rows):
        if rows.size == 0:
            return
        if not self.K_full_rank:
            self.rebuild_K()
            return
        pos = np.flatnonzero(np.isin(self.large, rows))
        keep = np.setdiff1d(np.arange(self.large.size), pos)
        K = self.K
        Kkk, Kkp, Kpp = K[np.ix_(keep, keep)], K[np.ix_(keep, pos)], K[np.ix_(pos, pos)]
        try:
            self.K = Kkk - Kkp @ np.linalg.solve(Kpp, Kkp.T)
            self.large = self.large[keep]
        except np.linalg.LinAlgError:
            self.large = self.large[keep]
            self.rebuild_K()

    # ---- cleanup (paper order) ----------------------------------------------
    def cleanup(self, initial=False):
        newly = np.flatnonzero((np.abs(self.x) >= 1 - 1e-12) & self.alive)
        if newly.size:
            self.x[newly] = np.sign(self.x[newly])
            self.K_freeze(newly)
            for j in newly:
                col = self.Bm[:, j]
                self.c += col * self.x[j]
                self.q -= self.P[:, j]
                self.cnt -= col != 0
                self.Bm[:, j] = 0.0
                self.P[:, j] = 0.0
            self.alive[newly] = False
            if self.K is None:
                self.rebuild_K()
        # large rows that became medium
        became = np.flatnonzero((self.cls == 2) & (self.q <= self.profile.radius ** 2))
        if became.size:
            empty = became[self.cnt[became] == 0]
            self.cls[empty] = 0
            nm = became[self.cnt[became] > 0]
            self.cls[nm] = 1
            self.r[nm] = np.minimum(self.profile.radius, r_candidate(self.q[nm], np.sqrt(self.P[nm].max(1)), self.profile.theta))
            # Rebuild paths in K_remove_rows must see the updated classes.
            self.K_remove_rows(became)
        # deliberate deletions in medium rows, lexicographic
        med = np.flatnonzero(self.cls == 1)
        if med.size:
            viol = med[self.P[med].max(1) > self.q[med] * self.profile.deletion_fraction]
            for i in viol:
                while self.cnt[i] > 0:
                    row = self.P[i]
                    bad = np.flatnonzero(row > self.q[i] * self.profile.deletion_fraction)
                    if bad.size == 0:
                        break
                    j = bad[0]
                    self.c[i] += self.Bm[i, j] * self.x[j]
                    self.deleted.append((i, j, self.x[j]))
                    self.G[i] -= row[j] * self.profile.energy(self.x[j])
                    self.q[i] -= row[j]
                    self.cnt[i] -= 1
                    self.Bm[i, j] = 0.0
                    self.P[i, j] = 0.0
            done = med[self.cnt[med] == 0]
            self.cls[done] = 0
            self.q[done] = 0.0
            self.G[done] = 0.0
            med = np.flatnonzero(self.cls == 1)
            if med.size:
                self.r[med] = np.minimum(self.r[med],
                                         r_candidate(self.q[med], np.sqrt(self.P[med].max(1)), self.profile.theta))
        s = int(self.alive.sum())
        if not initial and 0 < s <= self.S / 2:
            eta_old = self.eta
            self.S = s
            self.eta = 1.0 / (self.profile.ridge_denominator * self.J * self.S)
            med, u = self.barriers()
            fresh = max(self.lambda_upper(med, u) - 1, 0) ** 3 - self.eta * (self.x @ self.x)
            charged = self.Bphi + self.reset_charge() - (self.eta - eta_old) * (self.x @ self.x)
            self.Bphi = min(fresh, charged)          # both bound Phi; the charged one follows the budget

    # ---- barriers and certified spectral bound ------------------------------
    def barriers(self):
        med = np.flatnonzero(self.cls == 1)
        r = self.r[med]
        base = self.profile.energy_coefficient * self.G[med] / r ** 2 + self.profile.offset
        dm = self.d[med]
        return med, np.stack([(dm - self.profile.height) / r + base, (-dm - self.profile.height) / r + base], 1)

    def reset_charge(self):
        if self.profile.nonquadratic:
            return 1 / (2**40 * self.J)
        return 3 / (16 * 4096 * self.J)

    def lambda_upper(self, med, u, max_iters=25):
        V = self.alive
        if not V.any():
            return 0.0
        Z = self.P[med] / self.r[med, None] ** 2
        wsum = (self.profile.amplitude * np.exp(self.profile.beta * u) * (1 + 1e-13)).sum(1)
        diag = self.P[self.large].sum(0) * self.profile.deposit if self.large.size else np.zeros(self.n)
        eta = self.eta

        def Mv(y):
            out = Z.T @ (wsum * (Z @ y)) if med.size else np.zeros(self.n)
            return out + diag * y + eta * y[V].sum()

        y = self.y if self.y is not None else np.ones(self.n)
        y = np.where(V, np.maximum(y, 1e-300), 0.0)
        best, it = np.inf, 0
        for it in range(1, max_iters + 1):
            My = Mv(y)
            val = float(np.max(My[V] / y[V]))
            improved = val < best * (1 - 1e-3)
            best = min(best, val)
            y = np.where(V, My, 0.0)
            y /= np.linalg.norm(y)
            if best < 0.5 and not improved:
                break
        self.y = y
        self.stats["lam_iters"] += it
        return best * (1 + 1e-12)

    # ---- direction ----------------------------------------------------------
    def direction(self, extra_rows):
        V = self.alive
        xi = np.where(V, self.rng.standard_normal(self.n), 0.0)

        def projL(v):
            if self.large.size == 0:
                return v
            BL = self.Bm[self.large]
            return v - BL.T @ (self.K @ (BL @ v))

        for attempt in range(2):
            h = projL(xi)
            W = []
            for g in extra_rows:
                w = projL(np.where(V, g, 0.0))
                for ww in W:
                    w -= (w @ ww) * ww
                nw = np.linalg.norm(w)
                if nw > 1e-10 * max(1.0, np.linalg.norm(g)):
                    W.append(w / nw)
            for ww in W:
                h -= (h @ ww) * ww
            nh = np.linalg.norm(h)
            if nh < 1e-10:
                raise RuntimeError("empty kernel")
            h /= nh
            resid = 0.0
            if self.large.size:
                resid = float(np.abs(self.Bm[self.large] @ h).max())
            for g in extra_rows:
                resid = max(resid, abs(float(np.where(V, g, 0.0) @ h)) / max(1.0, np.linalg.norm(g)))
            if resid < 1e-9:
                return h
            self.rebuild_K()
        raise RuntimeError(f"projection residual {resid}")

    # ---- one move -----------------------------------------------------------
    def step(self):
        st = self.stats
        med, u = self.barriers()
        Lam = self.lambda_upper(med, u)
        st["max_Lambda"] = max(st["max_Lambda"], Lam)
        if u.size:
            st["max_u"] = max(st["max_u"], float(u.max()))
        if self.large.size:
            st["max_large_resid"] = max(st["max_large_resid"], float(np.abs(self.d[self.large]).max()))
        if Lam >= HIGH:
            st["high_branch"] = True
            return self.high_step(med, u)
        V = self.alive
        r = self.r[med]
        tight = u >= -self.profile.protected_threshold
        extra = []
        if tight.any():
            Zx = (self.profile.energy_coefficient * self.P[med]
                  * self.profile.first(self.x)[None, :] / r[:, None] ** 2)
            for k, sg in ((0, 1.0), (1, -1.0)):
                for ii in np.flatnonzero(tight[:, k]):
                    extra.append(sg * self.Bm[med[ii]] / r[ii] + Zx[ii])
            st["tight_events"] += int(tight.sum())
            st["max_tight"] = max(st["max_tight"], int(tight.sum()))
        extra.append(self.x.copy())
        h = self.direction(extra)
        bh = (self.Bm[med] @ h) / r
        zxh = (self.profile.energy_coefficient *
               (self.P[med] @ (self.profile.first(self.x) * h)) / r ** 2)
        gh = np.stack([bh + zxh, -bh + zxh], 1)
        zh2 = (self.P[med] @ (h * h)) / r ** 2
        limits = {}
        hv, xv = h[V], self.x[V]
        for sgn in (1.0, -1.0):
            sh = sgn * hv
            with np.errstate(divide="ignore", invalid="ignore"):
                room = np.where(sh > 0, (1 - xv) / sh, np.where(sh < 0, (1 + xv) / (-sh), np.inf))
            t_cube = float(room.min())
            a1 = sgn * gh
            a2 = 3.5 * zh2[:, None]
            uu = np.minimum(u, -U_TOL)
            disc = a1 * a1 + 4 * a2 * uu
            ok = (a1 > 0) & (disc >= 0) & ~tight
            tb = np.full(u.shape, np.inf)
            tb[ok] = 2 * (-uu[ok]) / (a1[ok] + np.sqrt(disc[ok]))
            t_bar = float(tb.min()) * (1 - 1e-9) if tb.size else np.inf
            if self.profile.nonquadratic:
                t_bar = self.profile_barrier_limit(h, sgn, t_cube, med)
            K = float(max(0.0, self.profile.beta * a1.max())) if a1.size else 0.0
            t_spec = math.log(1.0 / Lam) / K * (1 - 1e-9) if K > 0 else np.inf
            t = min(t_cube, t_bar, t_spec)
            kind = "freeze" if t == t_cube else ("barrier" if t == t_bar else "spectral")
            limits[sgn] = (t, kind, int(np.flatnonzero(V)[np.argmin(room)]))
        tp, tm = limits[1.0][0], limits[-1.0][0]
        if not self.coin or self.rng.random() < tm / (tp + tm):
            sgn = 1.0
        else:
            sgn = -1.0
        t, kind, jstar = limits[sgn]
        xnew = np.clip(self.x + sgn * t * h, -1, 1)
        if kind == "freeze":
            xnew[jstar] = np.sign(xnew[jstar])
        dx = xnew - self.x
        self.d += self.Bm @ dx
        if self.profile.nonquadratic:
            self.G += self.P @ (self.profile.energy(xnew) - self.profile.energy(self.x))
        else:
            self.G -= self.P @ (2 * self.x * dx + dx * dx)
        self.x = xnew
        self.Bphi = -self.eta * (self.x @ self.x)          # L <= 1 certified at the new state
        st["steps"] += 1
        st[kind + "_steps"] += 1
        med2, u2 = self.barriers()
        if u2.size and u2.max() > U_TOL:
            raise AssertionError(f"barrier violated after move: {u2.max()}")
        self.cleanup()
        med3, u3 = self.barriers()
        if u3.size and u3.max() > U_TOL:
            raise AssertionError(f"barrier violated after cleanup: {u3.max()}")
        if st["steps"] % REBUILD == 0:
            self.refresh()

    def profile_barrier_limit(self, h, sign, t_cube, med):
        """Check the nonquadratic endpoint; bracket a crossing only if needed.

        Concavity and feasibility at zero justify bisection when the cube
        endpoint violates a barrier. Feasible endpoints may cross an infeasible
        interior; the endpoint correctness lemma permits such moves.
        """
        if not med.size:
            return np.inf
        r = self.r[med]
        def feasible(t):
            xn = np.clip(self.x + sign * t * h, -1, 1)
            dn = self.d[med] + self.Bm[med] @ (xn - self.x)
            energy = self.P[med] @ self.profile.energy(xn)
            return np.all((np.abs(dn) - self.profile.height) / r +
                          self.profile.energy_coefficient * energy / r**2 +
                          self.profile.offset <= -U_TOL)
        if feasible(t_cube):
            return np.inf
        lo, hi = 0.0, t_cube
        for _ in range(50):
            mid = (lo + hi) / 2
            if feasible(mid):
                lo = mid
            else:
                hi = mid
        if lo <= 0:
            raise RuntimeError("no resolved profile step; exact fallback is not implemented")
        return lo * (1 - 1e-9)

    def dense_M(self, V, med, u):
        r = self.r[med]
        Z = self.P[med][:, V] / r[:, None] ** 2
        alpha = self.profile.amplitude * np.exp(self.profile.beta * u)
        diag = self.P[self.large][:, V].sum(0) * self.profile.deposit if self.large.size else np.zeros(V.size)
        M = (Z.T * alpha.sum(1)) @ Z + np.diag(diag) + self.eta
        return M, Z, alpha

    def certified_L(self, V, med, u):
        """Collatz-Wielandt bound from the dense Perron vector (upper bound on L)."""
        M, _, _ = self.dense_M(V, med, u)
        w, U = np.linalg.eigh(M)
        y = np.abs(U[:, -1]) + 1e-300
        return float((M @ y / y).max()) * (1 + 1e-12)

    def high_step(self, med, u):
        """Paper section 2.3 direction data + certified line search on Phi."""
        st = self.stats
        V = np.flatnonzero(self.alive)
        s = V.size
        r = self.r[med]
        M, Z, alpha = self.dense_M(V, med, u)
        lam, U = np.linalg.eigh(M)
        L = lam[-1]
        v = U[:, -1] * np.sign(U[:, -1].sum())
        Bn = self.Bm[med][:, V] / r[:, None]
        Zx = self.profile.energy_coefficient * Z * self.profile.first(self.x[V])[None, :]
        Gs = np.vstack([Bn + Zx, -Bn + Zx])
        al = np.concatenate([alpha[:, 0], alpha[:, 1]])
        Zv = Z @ v
        Zv2 = np.concatenate([Zv, Zv])
        tight = np.concatenate([u[:, 0], u[:, 1]]) >= -self.profile.protected_threshold
        F = U[:, lam >= self.profile.spectral_cut * L]
        ZF = Z @ F
        leak = (Gs * (al * self.profile.beta * Zv2)[:, None]).T @ np.vstack([ZF, ZF])
        rows = [R for R in (self.Bm[self.large][:, V], Gs[tight], self.x[V][None, :], leak.T) if R.size]
        C = np.vstack(rows)
        Uc, sv, _ = np.linalg.svd(C.T, full_matrices=True)
        rank = int((sv > sv.max() * 1e-11).sum())
        N = Uc[:, rank:]
        Ubar = (Gs * (al * self.profile.beta ** 2 * Zv2 ** 2)[:, None]).T @ Gs
        Dbar = (-self.profile.energy_coefficient * self.profile.second(self.x[V]) *
                (Z.T @ ((alpha.sum(1) * self.profile.beta) * Zv ** 2)))
        Cbar = self.profile.curvature_factor * Ubar - np.diag(Dbar)
        cn, Vn = np.linalg.eigh(N.T @ Cbar @ N)
        neg = cn < self.eta
        if not neg.any():
            raise RuntimeError("no negative-curvature feasible direction")
        h = np.zeros(self.n)
        hv = N @ (Vn[:, neg] @ self.rng.choice([-1.0, 1.0], size=int(neg.sum())))
        h[V] = hv / np.linalg.norm(hv)
        st["tight_events"] += int(tight.sum())
        st["max_tight"] = max(st["max_tight"], int(tight.sum()))
        gh = Gs @ h[V]
        zh2 = (Z * h[V][None, :] ** 2).sum(1)
        ush = np.concatenate([u[:, 0], u[:, 1]])
        best = None
        for sgn in (1.0, -1.0):
            sh = sgn * h[V]
            xv = self.x[V]
            with np.errstate(divide="ignore", invalid="ignore"):
                room = np.where(sh > 0, (1 - xv) / sh, np.where(sh < 0, (1 + xv) / -sh, np.inf))
            t_cube = float(room.min())
            a1, a2 = sgn * gh, 3.5 * np.concatenate([zh2, zh2])
            uu = np.minimum(ush, -U_TOL)
            disc = a1 * a1 + 4 * a2 * uu
            ok = (a1 > 0) & (disc >= 0) & ~tight
            tb = np.full(uu.shape, np.inf)
            tb[ok] = 2 * (-uu[ok]) / (a1[ok] + np.sqrt(disc[ok]))
            t_bar = float(tb.min()) * (1 - 1e-9) if tb.size else np.inf
            if self.profile.nonquadratic:
                t_bar = self.profile_barrier_limit(h, sgn, t_cube, med)
            t = min(t_cube, t_bar)
            kind = "freeze" if t == t_cube else "barrier"
            for _ in range(60):
                st["high_trials"] += 1
                dx = np.zeros(self.n)
                dx[V] = sgn * t * h[V]
                xn = np.clip(self.x + dx, -1, 1)
                if kind == "freeze":
                    j = V[np.argmin(room)]
                    xn[j] = np.sign(xn[j])
                dx = xn - self.x
                dn = self.d + self.Bm @ dx
                if self.profile.nonquadratic:
                    Gn = self.G + self.P @ (self.profile.energy(xn) - self.profile.energy(self.x))
                else:
                    Gn = self.G - self.P @ (2 * self.x * dx + dx * dx)
                base = self.profile.energy_coefficient * Gn[med] / r ** 2 + self.profile.offset
                un = np.stack([(dn[med] - self.profile.height) / r + base, (-dn[med] - self.profile.height) / r + base], 1)
                if not un.size or un.max() <= U_TOL:
                    phin = max(self.certified_L(V, med, un) - 1, 0) ** 3 - self.eta * (xn @ xn)
                    if phin <= self.Bphi:
                        if best is None or t > best[0]:
                            best = (t, xn, dn, Gn, phin, kind)
                        break
                t /= 2
                kind = "short"
        if best is None:
            raise RuntimeError("no certified high-branch step; the paper's tau-step is the fallback")
        t, xn, dn, Gn, phin, kind = best
        self.x, self.d, self.G, self.Bphi = xn, dn, Gn, phin
        st["steps"] += 1
        st["high_steps"] += 1
        st[("freeze" if kind == "freeze" else "barrier" if kind == "barrier" else "spectral") + "_steps"] += 1
        self.cleanup()
        med3, u3 = self.barriers()
        if u3.size and u3.max() > U_TOL:
            raise AssertionError(f"barrier violated after cleanup: {u3.max()}")
        if st["steps"] % REBUILD == 0:
            self.refresh()

    def refresh(self):
        d = self.c + self.Bm @ self.x
        G = self.P @ self.profile.energy(self.x)
        q = self.P.sum(1)
        self.drift = max(self.drift, float(np.abs(d - self.d).max(initial=0)),
                         float(np.abs(G - self.G).max(initial=0)), float(np.abs(q - self.q).max(initial=0)))
        self.d, self.G, self.q = d, G, q
        self.rebuild_K()

    def finish(self):
        """Terminal signing with the rule belonging to the selected profile.

        Nearest-sign rounding moves each coordinate by at most 1, which is what the
        large-row bound (<= 99) in Prop. s:detinv needs.
        The nonquadratic profile instead uses its quadratic expectation bound
        on at most 512 coordinates; stop_at=99 gives a matched-threshold check.
        """
        if self.terminal_rule == 'conditional':
            eps, rounding = conditional_round(self.A, self.x, self.alive)
            self.stats.update(rounding)
            return eps
        return np.where(self.x >= 0, 1.0, -1.0)

    def run(self, max_steps=None):
        t0 = time.time()
        if max_steps is None:
            max_steps = max(100, 10 * self.n)
        while self.alive.sum() > self.stop_at:
            if self.stats["steps"] >= max_steps:
                raise RuntimeError("prototype move cap exhausted; exact fallback is not implemented")
            self.step()
        self.refresh()
        track = np.zeros(self.m)
        for i, j, xd in self.deleted:
            track[i] += self.A[i, j] * (self.x[j] - xd)
        ident = float(np.abs(self.A @ self.x - self.d - track).max(initial=0))
        eps = self.finish()
        disc = np.abs(self.A @ eps)
        completed = self.cls == 0
        st = self.stats
        st.update(n=self.n, m=self.m, time=time.time() - t0, disc=float(disc.max(initial=0)),
                  max_abs_d_completed=float(np.abs(self.d[completed]).max()) if completed.any() else 0.0,
                  max_track=float(np.abs(track).max(initial=0)), n_deleted=len(self.deleted),
                  tracking_identity_err=ident, drift=self.drift,
                  certified=bool(disc.max(initial=0) < self.profile.discrepancy_bound),
                  within_bound_numerically=bool(disc.max(initial=0) < self.profile.discrepancy_bound),
                  profile=self.profile.name, stop_at=self.stop_at,
                  terminal_rule=self.terminal_rule,
                  discrepancy_bound=self.profile.discrepancy_bound,
                  terminal_coordinates=int(self.alive.sum()), final_fractional=int((np.abs(self.x) < 1).sum()))
        return eps, st
