"""Final-profile, normalized-response, and output-certificate regressions."""
from fractions import Fraction as F
import unittest
import numpy as np
from final_walk import FINAL37, FinalWalk, FinalGSWWalk, normalized_response, weighted_round
from verified_signing import signing_enclosure, verified_sign, LIMIT
from instances import FAMILIES, gauss


class FinalTests(unittest.TestCase):
    def test_profile_derivatives_and_ratio(self):
        x = np.linspace(-1, 1, 1001)
        f, fp, fpp = FINAL37.energy(x), FINAL37.first(x), FINAL37.second(x)
        self.assertEqual(f[0], 0); self.assertEqual(f[-1], 0)
        self.assertTrue(np.all(f >= 0))
        np.testing.assert_allclose((1+(52/173)*abs(fp))**2/-fpp, 78/173, rtol=1e-14)
        x = np.linspace(-.98, .98, 51); e = 1e-6
        np.testing.assert_allclose((FINAL37.energy(x+e)-FINAL37.energy(x-e))/(2*e),
                                   FINAL37.first(x), atol=2e-9)
        # At zero f''' jumps, so this central difference has O(e) error.
        np.testing.assert_allclose((FINAL37.first(x+e)-FINAL37.first(x-e))/(2*e),
                                   FINAL37.second(x), atol=2e-6)

    def test_weighted_round_against_fractions(self):
        A = [[F(1, 2), F(1, 3), -F(1, 4)], [F(1, 2), -F(1, 3), F(1, 4)]]
        x = [F(1), F(1, 5), -F(2, 3)]; exact = list(x); weight = F(19, 6)
        for j in (1, 2):
            def conditional_cost(sign):
                trial = exact.copy(); trial[j] = sign
                partial = [sum(a[k]*(trial[k]-x[k]) for k in range(j+1)) for a in A]
                return sum(z*z for z in partial)+weight*sum((trial[k]-x[k])**2 for k in range(j+1))
            exact[j] = F(1 if conditional_cost(1) <= conditional_cost(-1) else -1)
        eps, st = weighted_round(A, x, [False, True, True])
        np.testing.assert_array_equal(eps, np.array(exact, dtype=float))
        self.assertLessEqual(st['rounding_energy'], st['rounding_expectation']+1e-12)

    def test_actual_normalized_response(self):
        rng = np.random.default_rng(27)
        for n in (4, 11, 24):
            C = rng.uniform(.1, 1, (n, n+3)); C /= np.linalg.norm(C, 2)
            diagonal = rng.uniform(.1, .74, n)
            M = C@C.T + np.diag(diagonal)
            values, V = np.linalg.eigh(M); L = values[-1]; v = V[:, -1]
            scaling = 1/np.sqrt(L-diagonal)
            K = scaling[:, None]*(M-np.diag(diagonal))*scaling[None, :]
            ke, U = np.linalg.eigh(K); high = ke >= .72*ke[-1]
            # Choose forcing entirely in the normalized low space.
            force = (U[:, ~high]@rng.normal(size=((~high).sum(), n)))/scaling[:, None]
            positive = np.eye(n); negative = np.ones(n)*2
            Q, equations, st = normalized_response(M, diagonal, force, positive, negative)
            R = (V[:, :-1]/(L-values[:-1]))@V[:, :-1].T
            expected = -np.eye(n)+2*force.T@R@force
            np.testing.assert_allclose(Q, expected, atol=2e-12)
            np.testing.assert_allclose(equations, 0, atol=1e-12)
            self.assertAlmostEqual(st['normalized_top'], 1)

    def test_threshold_empty_filter_and_gsw(self):
        for shape in ((0, 0), (3, 0), (0, 20), (20, 10)):
            A = np.zeros(shape); eps, st = FinalWalk(A).run()
            self.assertEqual(eps.size, shape[1]); self.assertEqual(st['disc'], 0)
        A = gauss(337, 72, np.random.default_rng(2))
        for cls in (FinalWalk, FinalGSWWalk):
            w = cls(A, filter_rows=False, seed=1); eps, st = w.run()
            self.assertGreater(st['steps'], 0)
            self.assertLessEqual(st['terminal_coordinates'], 336)
            self.assertLessEqual(st['rounding_energy'], 1400+1e-8)
            self.assertLess(st['tracking_identity_err'], 1e-9)
            self.assertTrue(signing_enclosure(A, eps)[2]['bound_verified'])
        with self.assertRaises(ValueError): FinalWalk(A, stop_at=337)
        with self.assertRaises(ValueError): FinalWalk(A, x0=np.zeros(337))

    def test_endpoints_and_rank_loss(self):
        for family in ('gauss', 'hadamard', 'heavy_rows', 'prefix'):
            A = FAMILIES[family](400, 64, np.random.default_rng(3))
            w = FinalWalk(A, filter_rows=False, seed=0)
            eps, st = w.run()
            self.assertLess(st['tracking_identity_err'], 1e-9)
            self.assertLess(st['max_large_resid'], 1e-9)
            self.assertLessEqual(st['row_energy'], 4096*(w.x@w.x)+1e-7)
            self.assertTrue(signing_enclosure(A, eps)[2]['bound_verified'])

    def test_supplied_high_branch(self):
        rng = np.random.default_rng(10); A = gauss(384, 384, rng)
        w = FinalWalk(A, filter_rows=False, x0=rng.uniform(-.2, .2, 384))
        med, u = w.barriers(); hot = med[:90]
        base = w.G[hot]/w.r[hot]**2+FINAL37.offset
        original = w.d.copy(); lo, hi = -10., -.02; V = np.flatnonzero(w.alive)
        for _ in range(45):
            target = (lo+hi)/2
            w.d[hot] = FINAL37.height+w.r[hot]*(target-base)
            med, u = w.barriers(); L = np.linalg.eigvalsh(w.dense_M(V, med, u)[0])[-1]
            if L < 1.001: lo = target
            else: hi = target
        self.assertAlmostEqual(L, 1.001, places=9)
        w.c += w.d-original
        w.Bphi = max(w.certified_L(V, med, u)-1, 0)**3-w.eta*(w.x@w.x)
        before = w.x.copy(); w.step()
        self.assertEqual(w.stats['high_steps'], 1)
        self.assertLess(w.stats['high_constraint_residual'], 1e-8)
        self.assertGreater(w.stats['negative_dimension'], 0)
        self.assertGreater(np.linalg.norm(w.x-before), 0)
        np.testing.assert_allclose(w.d, w.c+w.Bm@w.x, atol=1e-10)

    def test_enclosures_against_exact_dyadic_sums(self):
        rng = np.random.default_rng(8)
        A = rng.normal(size=(7, 53)); eps = rng.choice([-1., 1.], size=53)
        A[0, :6] = [1e16, 1, -1e16, 1e-300, -1e-300, np.nextafter(0., 1.)]
        lo, hi, st = signing_enclosure(A, eps)
        for i, row in enumerate(A):
            exact = sum((F.from_float(float(a))*int(e) for a, e in zip(row, eps)), F())
            self.assertLessEqual(F.from_float(float(lo[i])), exact)
            self.assertGreaterEqual(F.from_float(float(hi[i])), exact)
        for a in (37., 37.5, 37.625, 38.):
            _, _, st = signing_enclosure([[a]], [1])
            self.assertEqual(st['bound_verified'], F.from_float(a) < LIMIT)
        with self.assertRaises(ValueError): signing_enclosure([[1]], [0])
        with self.assertRaises(ValueError): signing_enclosure([[np.nan]], [1])

    def test_fast_path_and_forced_fallback(self):
        A = gauss(400, 80, np.random.default_rng(2))
        eps, st = verified_sign(A, seed=3)
        self.assertTrue(st['bound_verified']); self.assertFalse(st['fallback_used'])
        # The sole proposal is deliberately maximally bad for this input.
        seed = 7; A = np.random.default_rng(seed).choice([-1., 1.], size=(337, 1)).T
        with self.assertRaises(RuntimeError): verified_sign(A, seed=seed, candidates=1, fallback=False)
        eps, st = verified_sign(A, seed=seed, candidates=1)
        self.assertTrue(st['fallback_used']); self.assertTrue(st['bound_verified'])
        self.assertEqual(eps.size, 337)


if __name__ == '__main__':
    unittest.main()
