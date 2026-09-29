"""Regression, terminal-rounding, and retained-state checks (unittest)."""
from fractions import Fraction as F
import unittest
import numpy as np
from clw2 import Walk
from instances import FAMILIES, gauss
from profiles import BELOW67, QUADRATIC, conditional_round


class ProfileTests(unittest.TestCase):
    def test_profile_identity_and_derivatives(self):
        x = np.linspace(-1, 1, 1001)
        f, fp, fpp = BELOW67.energy(x), BELOW67.first(x), BELOW67.second(x)
        self.assertEqual(float(f[0]), 0)
        self.assertEqual(float(f[-1]), 0)
        self.assertTrue(np.all(f >= 0))
        np.testing.assert_allclose(f, f[::-1], atol=1e-14)
        np.testing.assert_allclose((1 + (2 / 11) * np.abs(fp)) ** 2 / -fpp,
                                   50 / 187, rtol=1e-14)
        eps = 1e-6
        x = np.linspace(-.99, .99, 37)
        np.testing.assert_allclose((BELOW67.energy(x + eps) - BELOW67.energy(x - eps)) / (2 * eps),
                                   BELOW67.first(x), atol=2e-9)
        np.testing.assert_allclose((BELOW67.first(x + eps) - BELOW67.first(x - eps)) / (2 * eps),
                                   BELOW67.second(x), atol=1e-5)

    def test_terminal_rounding_against_exact_fractions(self):
        # Evaluate both exact conditional quadratic expectations independently.
        A = [[F(1, 2), F(-1, 3), F(1, 4), F(1, 2)],
             [F(1, 3), F(1, 2), F(-1, 2), F(-1, 3)]]
        x = [F(-1), F(1, 4), F(-1, 2), F(0)]
        exact = list(x)
        for j in range(1, len(x)):
            def cost(sign):
                trial = list(exact); trial[j] = F(sign)
                mean = [sum(row[k] * (trial[k] - x[k]) for k in range(j + 1)) for row in A]
                fixed = sum(z*z for z in mean) + 4 * sum((trial[k] - x[k])**2 for k in range(j + 1))
                variance = sum((1 - x[k]**2) * (4 + sum(row[k]**2 for row in A)) for k in range(j + 1, len(x)))
                return fixed + variance
            exact[j] = F(1 if cost(1) <= cost(-1) else -1)
        arr = np.array(A, dtype=float); xx = np.array(x, dtype=float)
        eps, st = conditional_round(arr, xx, np.array([False, True, True, True]))
        np.testing.assert_array_equal(eps, np.array(exact, dtype=float))
        self.assertLessEqual(st['rounding_energy'], st['rounding_expectation'] + 1e-12)

    def test_terminal_random_inputs_and_empty_case(self):
        rng = np.random.default_rng(8)
        for n in (0, 1, 17, 99, 512):
            A = gauss(n, 20, rng)
            x = rng.uniform(-1, 1, n)
            active = np.ones(n, dtype=bool)
            if n > 1:
                active[0] = False; x[0] = -1
            eps, st = conditional_round(A, x, active)
            self.assertTrue(np.all(np.abs(eps) == 1))
            self.assertLessEqual(st['rounding_energy'], 5 * active.sum() + 1e-9)
            self.assertTrue(np.array_equal(eps[~active], x[~active]))
        eps, st = Walk(np.zeros((3, 0)), profile='below67').run()
        self.assertEqual(eps.size, 0)
        self.assertEqual(st['disc'], 0)

    def test_nonquadratic_energy_cleanup_tracking_and_rounding(self):
        for family in ('gauss', 'sparse_bf', 'heavy_rows', 'prefix'):
            A = FAMILIES[family](540, 96, np.random.default_rng(19))
            walk = Walk(A, profile='below67', seed=2)
            eps, st = walk.run()
            self.assertTrue(st['within_bound_numerically'])
            self.assertLess(st['tracking_identity_err'], 1e-9)
            self.assertLess(st['max_large_resid'], 1e-9)
            self.assertLess(st['drift'], 1e-9)
            self.assertLessEqual(st['rounding_energy'], st['rounding_expectation'] + 1e-8)
            med, u = walk.barriers()
            self.assertTrue(not u.size or u.max() <= 1e-12)
            np.testing.assert_allclose(walk.G, walk.P @ BELOW67.energy(walk.x), atol=1e-11)

    def test_terminal_threshold_and_invalid_input(self):
        A = gauss(513, 30, np.random.default_rng(3))
        _, st = Walk(A, profile='below67').run()
        self.assertGreater(st['steps'], 0)
        self.assertLessEqual(st['terminal_coordinates'], 512)
        with self.assertRaises(ValueError):
            Walk(A, profile='below67', stop_at=513)
        with self.assertRaises(ValueError):
            Walk(2 * A, profile='below67')

    def test_rank_loss_does_not_delete_offsets(self):
        A = FAMILIES['hadamard'](640, 80, np.random.default_rng(1))
        for profile in ('quadratic', 'below67'):
            walk = Walk(A, seed=0, profile=profile, stop_at=99)
            _, st = walk.run()
            self.assertLess(st['tracking_identity_err'], 1e-9)
            self.assertLess(st['max_large_resid'], 1e-9)
            self.assertLess(st['drift'], 1e-9)
            self.assertTrue(np.all(walk.r[walk.cls == 1] > 0))

    def test_early_conditional_finish_for_both_direction_options(self):
        from clw_gsw import GSWWalk
        A = gauss(540, 96, np.random.default_rng(29))
        for cls in (Walk, GSWWalk):
            walk = cls(A, seed=4, terminal='conditional')
            eps, st = walk.run()
            self.assertEqual(st['stop_at'], 512)
            self.assertLessEqual(st['terminal_coordinates'], 512)
            self.assertLessEqual(st['rounding_energy'], 5 * st['terminal_coordinates'])
            self.assertLess(st['disc'], QUADRATIC.discrepancy_bound)
        with self.assertRaises(ValueError):
            Walk(A, profile='below67', terminal='nearest')

    def test_high_branch_nonquadratic_endpoint(self):
        # Supplied offsets make this a local-state test, not a reachable signing
        # run or evidence for an end-to-end discrepancy guarantee.
        rng = np.random.default_rng(10)
        A = gauss(540, 540, rng)
        walk = Walk(A, x0=rng.uniform(-.2, .2, 540), profile='below67')
        med, u = walk.barriers()
        hot = med[:35]
        base = walk.G[hot] / walk.r[hot]**2 + BELOW67.offset
        original_d = walk.d.copy()
        lo, hi = -6., -.02
        V = np.flatnonzero(walk.alive)
        for _ in range(45):
            target_u = (lo + hi) / 2
            walk.d[hot] = BELOW67.height + walk.r[hot] * (target_u - base)
            med, u = walk.barriers()
            L = np.linalg.eigvalsh(walk.dense_M(V, med, u)[0])[-1]
            if L < 1.001:
                lo = target_u
            else:
                hi = target_u
        self.assertAlmostEqual(L, 1.001, places=9)
        walk.c += walk.d - original_d
        before = walk.certified_L(V, med, u)
        walk.Bphi = max(before - 1, 0)**3 - walk.eta * (walk.x @ walk.x)
        walk.step()
        self.assertEqual(walk.stats['high_steps'], 1)
        med, u = walk.barriers()
        self.assertLessEqual(u.max(initial=-np.inf), 1e-12)
        self.assertLess(walk.certified_L(np.flatnonzero(walk.alive), med, u), BELOW67.spectral_ceiling)
        np.testing.assert_allclose(walk.d, walk.c + walk.Bm @ walk.x, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
