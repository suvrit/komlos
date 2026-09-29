# Open problems and issues

Section numbers refer to the paper. Items are ordered by how much they would
change the results, not by difficulty.

## Mathematics

1. **Make an existence constant near 7 algorithmic.** The existence proofs
   (Section 2) propagate convex sets and densities backward; the
   walk (Sections 3.3–4) moves forward under barriers and a spectral potential.
   The constants differ by a factor above five. Nothing is known about whether
   the lifting argument admits a computational version.
2. **Cubic arithmetic cost.** The Õ(n⁴) term comes from the dense feasible
   negative basis and the coupled response at each of Õ(n) anchors
   (Section 4.4). Two partial routes are recorded in `archive/`:
   - the randomized variance clock gives O(mn)+Õ(n³) expected work for row
     processing alone; the direction construction is still quartic;
   - the below-67 construction has the bound O(mn)+Õ(m₆₆n²+n^(ω+1)) under
     imported fast spectral and rank primitives with square roots, but the
     fast spectral contract has not been shown to preserve the exactly
     orthogonal normalized reconstruction that the dimension count of
     Section 3.4.6 requires.
3. **Bit complexity.** States are rational for rational input, but no
   polynomial bound on their bit length is asserted (Theorem 3.1).
4. **A smaller constant for the spectral walk.** The parameters (3.6) were
   found by a local numerical search (`archive/search_fixed_terminal.py`) and
   are not claimed optimal. The rational dimension certificate (Section 3.4.6)
   leaves a dimension fraction of 0.002576 against the required 1/400.
5. **The terminal rule.** Weighted conditional-expectation rounding at 336
   coordinates has a proof but does not minimize the maximum row sum; on an
   orthonormal remaining matrix every signing has the same objective, and
   positive tie breaking on a normalized Hadamard matrix gives a large row
   maximum (Appendix F.3). A terminal rule with both a proof and small
   observed discrepancy is wanted.
6. **Better initial densities for the existence argument.** C_cos is the exact
   threshold of the product-cosine density under the integrated-translation
   criterion of Lemma 2.5, uniformly over dimensions (Section 2.3.3). Other
   initial densities, or a criterion that is not uniform over directions,
   might lower the constant.

## Implementation

7. **Implement the exact finite construction** (rational Jacobi rotations,
   exact projector, deterministic flattening, dyadic step schedule of
   Section 3.5 and Appendix E) as the fallback of `verified_sign`. This turns
   the checked routine into a complete solver with the worst-case guarantee of
   Theorem 3.1. The proved step cap 2⁻⁵⁶ makes this slow; the point is
   completeness, not speed.
8. **Lazy warm/cold screening** (Section 4) is proved but not implemented;
   the prototypes evaluate every retained row at every move.
9. **Acceptance probability.** No uniform bound is proved for the eight-proposal
   stage of Algorithm 5.1; every tested case accepted, including n = 4096.
10. **The high branch has not been exercised by a trajectory.** Every matched
    run stayed in the low branch; the normalized response is tested only on a
    fixed test state. Inputs that reach the high branch would test the code path
    that the improved constant depends on.
11. **Numerical robustness of the prototypes.** A rank-loss repair in the
    Gram–Schmidt projector was needed during the comparison; the prototypes
    report, rather than resolve, an unresolved step.
