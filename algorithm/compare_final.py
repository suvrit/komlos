"""Matched runtime/quality comparison for the final profile and the retained earlier walks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
from clw_gsw import GSWWalk
from final_walk import FinalWalk, FinalGSWWalk, MatchedTerminalWalk
from verified_signing import verified_sign, signing_enclosure
from instances import FAMILIES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sizes', nargs='+', type=int, default=[640])
    p.add_argument('--ratios', nargs='+', type=float, default=[.125, 1, 4])
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1])
    p.add_argument('--families', nargs='+', choices=list(FAMILIES), default=list(FAMILIES))
    p.add_argument('--fast-only', action='store_true')
    args = p.parse_args(); args.output.parent.mkdir(parents=True, exist_ok=True)
    variants = [
        ('quadratic-gsw', GSWWalk, {}),
        ('quadratic-ce336', MatchedTerminalWalk, {}),
        ('below67-ce336', MatchedTerminalWalk, {'profile':'below67'}),
        ('final37', FinalWalk, {'filter_rows':False}),
        ('final37-screened', FinalWalk, {'filter_rows':True}),
        ('final37-gsw', FinalGSWWalk, {'filter_rows':False}),
        ('verified-fast', None, {}),
    ]
    if args.fast_only: variants = variants[-1:]
    sources = ['clw2.py', 'clw_gsw.py', 'profiles.py', 'final_walk.py',
               'verified_signing.py', 'instances.py', 'compare_final.py']
    metadata = dict(python=platform.python_version(), numpy=np.__version__,
                    platform=platform.platform(), variants=[v[0] for v in variants],
                    environment={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']},
                    timing='constructor, proposals/walk/rounding, and outward interval verification',
                    source_sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in sources})
    args.output.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    with args.output.open('w') as output:
        case = 0
        for n in args.sizes:
            for ratio in args.ratios:
                m = max(8, round(ratio*n))
                for family in args.families:
                    for seed in args.seeds:
                        matrix_seed = 20260928+seed
                        A = FAMILIES[family](n, m, np.random.default_rng(matrix_seed))
                        shift = case % len(variants); order = variants[shift:]+variants[:shift]; case += 1
                        for name, constructor, options in order:
                            row = dict(variant=name, family=family, n=n, m=m, seed=seed, matrix_seed=matrix_seed)
                            start = time.perf_counter()
                            try:
                                if constructor is None:
                                    eps, stats = verified_sign(A, seed=seed)
                                    certificate = {k:stats[k] for k in ['bound_verified','certified_bound','verification','target_exact']}
                                else:
                                    eps, stats = constructor(A, seed=seed, **options).run()
                                    certificate = signing_enclosure(A, eps)[2]
                                elapsed = time.perf_counter()-start
                                if not certificate['bound_verified']:
                                    raise AssertionError('strict 37.54 interval certificate failed')
                                if (stats.get('tracking_identity_err', 0) > 1e-8 or
                                        stats.get('max_large_resid', 0) > 1e-8):
                                    raise AssertionError('retained-state residual exceeded tolerance')
                                row.update(stats); row.update(certificate)
                                row.update(status='ok', wall_seconds=elapsed,
                                           disc=float(np.abs(A@eps).max(initial=0)),
                                           signing_sha256=hashlib.sha256(eps.astype(np.int8).tobytes()).hexdigest())
                            except Exception as exc:
                                row.update(status='error', wall_seconds=time.perf_counter()-start,
                                           error=f'{type(exc).__name__}: {exc}')
                            for k, value in list(row.items()):
                                if isinstance(value, float) and not np.isfinite(value): row[k] = None
                            output.write(json.dumps(row, allow_nan=False)+'\n'); output.flush()
                            print(f'{family} {m}x{n} seed={seed} {name}: {row["status"]} '
                                  f'{row["wall_seconds"]:.4f}s disc={row.get("disc")}', flush=True)


if __name__ == '__main__':
    main()
