# Final profile: discrepancy, arithmetic complexity, and measured execution

The deterministic theorem of the paper gives **discrepancy < 37.54**, with **O(mn) + soft-O(n^4)** ordinary arithmetic work. The screened below-67 baseline retains its separate imported fast-arithmetic guarantee. Neither the randomized cubic row-processing result nor these measurements establish cubic total signing time.

## Matched comparison: 48 cases, 336 executions

Six matrix families, two seeds, and configurations (n,m): [(640, 80), (640, 640), (640, 2560), (1024, 1024)].
Each ratio is the median of paired elapsed times to the original quadratic GSW configuration. Times include construction, terminal work, and interval verification. Input generation is outside the timer; method order rotates. These are single-run measurements, not confidence intervals or an asymptotic scaling result.

| Implementation | Passed / cases | Median seconds | Time / original GSW | Discrepancy range | Walk moves |
|---|---:|---:|---:|---:|---:|
| Quadratic, original GSW finish | 48/48 | 2.7407 | 1.0000 | 0.000–3.438 | 512–925 |
| Quadratic, CE336 | 48/48 | 1.2699 | 0.4012 | 0.415–4.384 | 304–688 |
| Below 67, CE336 | 48/48 | 1.1141 | 0.4709 | 0.415–4.384 | 304–688 |
| Final 37.54, no input filter | 48/48 | 1.1664 | 0.6029 | 0.316–4.250 | 304–690 |
| Final 37.54, input filter | 48/48 | 0.0163 | 0.0022 | 0.593–32.000 | 0–688 |
| Final 37.54 + GSW, CE336 | 48/48 | 2.2244 | 0.8750 | 0.059–10.375 | 289–688 |
| Eight proposals + interval check | 48/48 | 0.0077 | 0.0016 | 0.257–11.006 | 0–0 |

CE336 uses weight 19/6 on the remaining-coordinate displacement. The old GSW configuration keeps its original finish; the new GSW configuration uses CE336. “Input filter” means the one-time l1 filter, not lazy warm/cold screening. Matching the terminal rule still leaves the final profile’s new energy constraints.

- Final / quadratic, matched finish: median time ratio 1.521; lower/equal/higher discrepancy in 19/15/14 cases.
- Final / below67, matched finish: median time ratio 1.112; lower/equal/higher discrepancy in 20/14/14 cases.

The eight-proposal path used the walk fallback in 0 matched cases. Its discrepancy was lower/equal/higher than original GSW in 5/4/39 cases. Its speed does not imply that it minimizes discrepancy. Keep the original GSW option when output quality justifies the extra time.

The input filter avoided all walk moves in 36 cases. All-positive completion of filtered instances can have larger discrepancy while satisfying 37.54. The fast proposal path evaluates its candidates on every original row.

## Larger fast-path checks

18/18 additional executions passed. Elapsed-time range: 0.008–0.230 seconds; discrepancy range: 0.688–15.431. Walk fallbacks: 0. These have no matched old-walk timing and are not used in paired speed comparisons.

## Verification and limits

Failures: 0. Largest tracking residual: 1.73e-14; largest large-row residual: 9.75e-12. High-branch matched runs: 0.

Every successful run passed the outward-rounded row-sum check against the exact rational target 1877/50. This certifies the sums of the represented binary64 matrix, not the trajectory or any unrepresented real input. The numerical endpoint fallback and lazy screening do not implement the exact finite algorithm. A separate fixed-state regression checks the normalized high branch. All raw records and source hashes are preserved.

