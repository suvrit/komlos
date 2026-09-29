# Komlós discrepancy: existence bounds, a deterministic algorithm, and certificates

[![code: Apache 2.0](https://img.shields.io/badge/code-Apache%202.0-4c566a)](LICENSE)
[![docs: CC BY 4.0](https://img.shields.io/badge/docs-CC%20BY%204.0-4c566a)](LICENSE-DOCS)

Companion repository for *Tighter bounds on Komlós discrepancy: existence and
algorithmic results* (E. Akbas and S. Sra). It holds the code, the exact
certificates, and the experiment records that the paper refers to, and a list
of open problems.

## What is proved, what is implemented, and what is measured

Keep these apart when reading the code:

| Claim | Where | Status |
|---|---|---|
| Existence: discrepancy < 6.9013 (Theorem 2.4) | `existence/` | proved in the paper; the enclosure of the constant is certified by exact rational arithmetic (`verify_cosine_constant.py`) |
| Deterministic algorithm: discrepancy < 37.54 in O(mn)+Õ(n⁴) operations (Theorem 3.1) | `certificates/` | proved in the paper; every rational inequality it uses is certified by `verify_constant.py` and an independent recomputation. **The exact finite algorithm is not implemented.** |
| Checked proposal routine (Algorithm 5.1) | `algorithm/verified_signing.py` | implemented; returns a signing only when an outward-rounded interval check certifies the bound; can report failure |
| Numerical endpoint walks (final profile, quadratic and below-67 profiles, GSW directions) | `algorithm/final_walk.py`, `clw2.py`, `clw_gsw.py` | floating-point prototypes that take large endpoint steps accepted by numerical tests; not the proved tiny-step schedule |
| Measurements (Appendix F) | `results/` | raw JSONL records with source hashes and metadata; single-run timings |

## Layout

```
existence/     certificates for Section 2 of the paper (standard library; optional SymPy and NumPy/SciPy checks)
algorithm/     the checked routine, the walks, unit tests, and the benchmark driver (NumPy)
certificates/  exact certificates and identity audits for the 37.54 proof, with their recorded outputs
results/       the matched-suite records behind Appendix F and the summary they produce
archive/       frozen technical sources: the final and below-67 constructions, the development record, and their scripts
paper/         the paper source and build files (added when the paper is final)
```

## Reproduce

Existence constant (the first two use the standard library; the optional checks need SymPy, and NumPy with SciPy):

```sh
python3 existence/verify_cosine_constant.py
python3 existence/verify_69013.py            # independent certificate from the original note
python3 existence/audit_algebra.py           # optional symbolic identities (SymPy)
python3 existence/check_cosine_translation.py  # optional quadrature cross-check (NumPy, SciPy)
```

The last three also write their reports as JSON next to the script; the
committed copies are the recorded outputs.

Certificates for the 37.54 construction (the first five use the standard library; the two audits need NumPy; recorded outputs are in `certificates/results/`):

```sh
python3 certificates/verify_constant.py
python3 certificates/audit_constant_independent.py
python3 certificates/audit_protected_charge.py
python3 certificates/check_randomized_screening.py
python3 certificates/check_endpoint_schedule.py
OPENBLAS_NUM_THREADS=1 python3 certificates/audit_joint_response.py
OPENBLAS_NUM_THREADS=1 python3 certificates/audit_generalized_response.py
```

Implementation tests and the matched benchmark (Python 3 with NumPy; last run with Python 3.12 and NumPy 2.3; run timing suites sequentially, one thread):

```sh
cd algorithm
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 -m unittest test_final test_profiles -v
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 compare_final.py --output ../results/n640.jsonl --sizes 640
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 compare_final.py --output ../results/n1024.jsonl --sizes 1024 --ratios 1
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 compare_final.py --output ../results/fast2048.jsonl --sizes 2048 --ratios .125 1 --seeds 0 --fast-only
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 compare_final.py --output ../results/fast4096.jsonl --sizes 4096 --ratios .125 1 --seeds 0 --families gauss hadamard heavy_rows --fast-only
python3 summarize_final.py ../results/n640.jsonl ../results/n1024.jsonl ../results/fast2048.jsonl ../results/fast4096.jsonl --output ../results/summary.json
```

`summarize_final.py` checks each record's `source_sha256` against the files
in `algorithm/`, so it refuses records produced by different code.

Archived scripts (the certificate uses the standard library; the audit needs mpmath and the parameter search SciPy):

```sh
python3 archive/verify_constant67.py         # below-67 constants; recorded in exact_certificate_results.json
python3 archive/audit_finite_profile.py      # high-precision Perron stress checks; recorded in finite_profile_audit_results.json
python3 archive/search_fixed_terminal.py     # parameter search behind (3.6); rewrites fixed_terminal_results.json
```

Use the checked entry point from Python:

```python
from verified_signing import verified_sign, signing_enclosure
eps, result = verified_sign(A, seed=0)      # eight random proposals, then the numerical walk
assert result['bound_verified']
lower, upper, cert = signing_enclosure(A, eps)   # enclose any candidate's row sums
```

`verified_sign` certifies the binary64 matrix it is given, needs no unit-column
assumption, and raises an exception instead of returning an uncertified
signing. Installing the exact finite construction as its fallback is the first
item in `OPEN_PROBLEMS.md`.

## Citation

If you find this repository useful, please cite the paper:

```bibtex
@misc{akbas2026komlos,
  title         = {Tighter bounds on {K}oml{\'o}s discrepancy: existence and algorithmic results},
  author        = {Akbas, Emrullah and Sra, Suvrit},
  year          = {2026},
  eprint        = {2609.27172},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2609.27172}
}
```

## License

Copyright 2026 Emrullah Akbas and Suvrit Sra.

Code, that is `algorithm/`, `certificates/`, `existence/`, and every script
under `archive/`, is licensed under the Apache License 2.0
([LICENSE](LICENSE)). Prose, mathematical exposition, and data, that is
`paper/`, the archived manuscripts under `archive/`, `results/`,
`OPEN_PROBLEMS.md`, and this file, are licensed under CC BY 4.0
([LICENSE-DOCS](LICENSE-DOCS)). Contributions are accepted under the same
terms.
