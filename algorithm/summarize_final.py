"""Reproduce the final comparison from raw records and recorded source hashes."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

LABELS = {
    'quadratic-gsw':'Quadratic, original GSW finish',
    'quadratic-ce336':'Quadratic, CE336',
    'below67-ce336':'Below 67, CE336',
    'final37':'Final 37.54, no input filter',
    'final37-screened':'Final 37.54, input filter',
    'final37-gsw':'Final 37.54 + GSW, CE336',
    'verified-fast':'Eight proposals + interval check',
}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('records', nargs='+', type=Path)
    p.add_argument('--output', required=True, type=Path)
    args=p.parse_args(); rows=[]
    for path in args.records:
        metadata=json.loads(path.with_suffix('.metadata.json').read_text())
        for name, expected in metadata['source_sha256'].items():
            assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==expected, name
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    cases=defaultdict(dict)
    for r in rows:
        key=(r['family'],r['m'],r['n'],r['seed'],r['matrix_seed'])
        if r['variant'] in cases[key]: raise ValueError(f'duplicate observation {key}')
        cases[key][r['variant']]=r
    matched=[c for c in cases.values() if set(c)==set(LABELS)]
    if any(set(c) not in (set(LABELS),{'verified-fast'}) for c in cases.values()):
        raise ValueError('incomplete matched case')
    def paired(name, baseline):
        pairs=[(c[name],c[baseline]) for c in matched if c[name]['status']==c[baseline]['status']=='ok']
        ratios=[a['wall_seconds']/b['wall_seconds'] for a,b in pairs]
        lower=sum(a['disc']<b['disc']-1e-9 for a,b in pairs)
        higher=sum(a['disc']>b['disc']+1e-9 for a,b in pairs)
        return dict(cases=len(pairs), median_time_ratio=statistics.median(ratios) if ratios else None,
                    median_speedup=statistics.median(1/r for r in ratios) if ratios else None,
                    lower_discrepancy=lower,tied_discrepancy=len(pairs)-lower-higher,higher_discrepancy=higher)
    variants={}
    for name in LABELS:
        selected=[c[name] for c in matched]; good=[r for r in selected if r['status']=='ok']
        variants[name]=dict(successes=len(good),failures=len(selected)-len(good),
            median_seconds=statistics.median(r['wall_seconds'] for r in good) if good else None,
            discrepancy_range=[min(r['disc'] for r in good),max(r['disc'] for r in good)] if good else None,
            moves_range=[min(r.get('steps',0) for r in good),max(r.get('steps',0) for r in good)] if good else None,
            high_branch_runs=sum(bool(r.get('high_branch')) for r in good),
            vs_original_gsw=paired(name,'quadratic-gsw'))
    extra=[r for c in cases.values() if set(c)=={'verified-fast'} for r in c.values()]
    extra_good=[r for r in extra if r['status']=='ok']
    matched_configurations=sorted({(r['n'],r['m']) for c in matched for r in c.values()})
    summary=dict(matched_cases=len(matched),matched_runs=len(matched)*len(LABELS),
        matched_configurations=matched_configurations, variants=variants,
        final_vs_quadratic_matched_finish=paired('final37','quadratic-ce336'),
        final_vs_below67_matched_finish=paired('final37','below67-ce336'),
        input_filter_saves_walk=sum(c['final37-screened'].get('steps')==0 for c in matched),
        fast_fallbacks=sum(c['verified-fast'].get('fallback_used',False) for c in matched),
        fast_extra_runs=len(extra),fast_extra_successes=len(extra_good),
        fast_extra_seconds=[min(r['wall_seconds'] for r in extra_good),max(r['wall_seconds'] for r in extra_good)] if extra_good else None,
        fast_extra_discrepancy=[min(r['disc'] for r in extra_good),max(r['disc'] for r in extra_good)] if extra_good else None,
        fast_extra_fallbacks=sum(r.get('fallback_used',False) for r in extra_good),
        max_tracking_residual=max((r.get('tracking_identity_err',0) for r in rows if r['status']=='ok'),default=0),
        max_large_row_residual=max((r.get('max_large_resid',0) for r in rows if r['status']=='ok'),default=0),
        max_interval_bound=max((r.get('certified_bound',0) for r in rows if r['status']=='ok'),default=0),
        failures=[r for r in rows if r['status']!='ok'])
    args.output.write_text(json.dumps(summary,indent=2)+'\n')
    md=['# Final profile: discrepancy, arithmetic complexity, and measured execution','',
        'The deterministic theorem of the paper gives **discrepancy < 37.54**, with '
        '**O(mn) + soft-O(n^4)** ordinary arithmetic work. The screened below-67 baseline '
        'retains its separate imported fast-arithmetic guarantee. Neither the randomized cubic '
        'row-processing result nor these measurements establish cubic total signing time.','',
        f"## Matched comparison: {len(matched)} cases, {summary['matched_runs']} executions",'',
        'Six matrix families, two seeds, and configurations (n,m): '+str(matched_configurations)+'.',
        'Each ratio is the median of paired elapsed times to the original quadratic GSW '
        'configuration. Times include construction, terminal work, and interval verification. '
        'Input generation is outside the timer; method order rotates. These are single-run '
        'measurements, not confidence intervals or an asymptotic scaling result.','',
        '| Implementation | Passed / cases | Median seconds | Time / original GSW | Discrepancy range | Walk moves |',
        '|---|---:|---:|---:|---:|---:|']
    tex=[r'\begin{table}[htbp]',r'\centering\small',
         r'\caption{Final-profile comparison. Times include construction and output verification; ratios are paired medians relative to the original quadratic GSW configuration.}',
         r'\label{tab:final-comparison}',r'\begin{tabular}{@{}lrrr@{}}',r'\toprule',
         r'Implementation & Time/GSW & Discrepancy range & Moves\\',r'\midrule']
    for name,label in LABELS.items():
        v=variants[name]; dr=v['discrepancy_range']; mr=v['moves_range']; ratio=v['vs_original_gsw']['median_time_ratio']
        if dr is None:
            md.append(f'| {label} | 0/{len(matched)} | failed | failed | — | — |'); continue
        md.append(f'| {label} | {v["successes"]}/{len(matched)} | {v["median_seconds"]:.4f} | {ratio:.4f} | {dr[0]:.3f}–{dr[1]:.3f} | {mr[0]}–{mr[1]} |')
        tex.append(f'{label} & {ratio:.4f} & $[{dr[0]:.3f},{dr[1]:.3f}]$ & ${mr[0]}$--${mr[1]}$'+r'\\')
    tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    md += ['', 'CE336 uses weight 19/6 on the remaining-coordinate displacement. The old GSW '
           'configuration keeps its original finish; the new GSW configuration uses CE336. '
           '“Input filter” means the one-time l1 filter, not lazy warm/cold screening. '
           'Matching the terminal rule still leaves the final profile’s new energy constraints.','']
    for name,title in [('final_vs_quadratic_matched_finish','Final / quadratic, matched finish'),
                       ('final_vs_below67_matched_finish','Final / below67, matched finish')]:
        z=summary[name]
        if z['median_time_ratio'] is not None:
            md += [f'- {title}: median time ratio {z["median_time_ratio"]:.3f}; lower/equal/higher discrepancy in '
                   f'{z["lower_discrepancy"]}/{z["tied_discrepancy"]}/{z["higher_discrepancy"]} cases.']
    z=variants['verified-fast']['vs_original_gsw']
    md += ['',f'The eight-proposal path used the walk fallback in {summary["fast_fallbacks"]} matched cases. '
           f'Its discrepancy was lower/equal/higher than original GSW in '
           f'{z["lower_discrepancy"]}/{z["tied_discrepancy"]}/{z["higher_discrepancy"]} cases. '
           'Its speed does not imply that it minimizes discrepancy. Keep the original GSW option '
           'when output quality justifies the extra time.','',
           f'The input filter avoided all walk moves in {summary["input_filter_saves_walk"]} cases. '
           'All-positive completion of filtered instances can have larger discrepancy while '
           'satisfying 37.54. The fast proposal path evaluates its candidates on every original row.','',
           '## Larger fast-path checks','',
           f'{len(extra_good)}/{len(extra)} additional executions passed. '
           + (f'Elapsed-time range: {summary["fast_extra_seconds"][0]:.3f}–{summary["fast_extra_seconds"][1]:.3f} seconds; '
              f'discrepancy range: {summary["fast_extra_discrepancy"][0]:.3f}–{summary["fast_extra_discrepancy"][1]:.3f}. '
              if extra_good else '') + f'Walk fallbacks: {summary["fast_extra_fallbacks"]}. '
           'These have no matched old-walk timing and are not used in paired speed comparisons.','',
           '## Verification and limits','',
           f'Failures: {len(summary["failures"])}. Largest tracking residual: '
           f'{summary["max_tracking_residual"]:.3g}; largest large-row residual: '
           f'{summary["max_large_row_residual"]:.3g}. High-branch matched runs: '
           f'{sum(v["high_branch_runs"] for v in variants.values())}.','',
           'Every successful run passed the outward-rounded row-sum check against the exact '
           'rational target 1877/50. This certifies the sums of the represented binary64 matrix, '
           'not the trajectory or any unrepresented real input. The numerical endpoint fallback '
           'and lazy screening do not implement the exact finite algorithm. A separate fixed-state '
           'regression checks the normalized high branch. All raw records and source hashes are preserved.','']
    if summary['failures']: md += ['```json',json.dumps(summary['failures'],indent=2),'```']
    args.output.with_name('comparison.md').write_text('\n'.join(md)+'\n')
    args.output.with_name('comparison.tex').write_text('\n'.join(tex)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
