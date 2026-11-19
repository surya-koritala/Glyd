"""Pair every scored position and separate native-engine correctness from approximation quality."""
import argparse
import json
import math
import random
from pathlib import Path


def paired(reference, candidate):
    for key in ('model', 'revision', 'evaluation_sha256'):
        if not reference.get(key) or reference[key] != candidate.get(key):
            raise ValueError(f'{key} mismatch')
    a, b = reference['samples'], candidate['samples']
    if not a or len(a) != len(b):
        raise ValueError('sample count mismatch')
    errors, nll_deltas, counts = [], [], []
    matches = 0
    for r, c in zip(a, b):
        n = len(r['score_lp'])
        if r['id'] != c['id'] or not n or any(len(x) != n for x in (c['score_lp'], r['score_top1'], c['score_top1'])):
            raise ValueError('scored positions differ')
        values = r['score_lp'] + c['score_lp']
        if not all(math.isfinite(v) for v in values):
            raise ValueError('non-finite likelihood')
        errors.extend(abs(x-y) for x,y in zip(r['score_lp'], c['score_lp']))
        nll_deltas.append(sum(x-y for x,y in zip(r['score_lp'], c['score_lp']))/n)
        counts.append(n)
        matches += sum(x==y for x,y in zip(r['score_top1'], c['score_top1']))
    n = sum(counts)
    delta = sum(d*k for d,k in zip(nll_deltas, counts))/n
    rng = random.Random(20261006)
    boots = []
    for _ in range(2000):
        ix = [rng.randrange(len(a)) for _ in a]
        boots.append(sum(nll_deltas[i]*counts[i] for i in ix)/sum(counts[i] for i in ix))
    boots.sort()
    return {'scored_tokens': n, 'top1_agreement': matches/n, 'mean_abs_logprob_difference': sum(errors)/n,
            'max_abs_logprob_difference': max(errors), 'mean_nll_delta': delta,
            'perplexity_ratio': math.exp(delta), 'nll_delta_window_bootstrap_95': [boots[49], boots[1949]],
            'window_nll_deltas': nll_deltas, 'window_count': len(a), 'scope': 'paired held-out likelihood screen, not task accuracy',
            'quality_evidence': ('degradation_detected' if boots[49]>0 else 'no_worse_on_this_screen' if boots[1949]<=0 else 'inconclusive')}


def control_passes(result):
    # Screening bounds for runtime agreement; no lossless byte/output check is replaced or relaxed.
    return (result['top1_agreement'] >= 0.95 and result['mean_abs_logprob_difference'] <= 0.05
            and max(abs(x) for x in result['window_nll_deltas']) <= 0.10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('reference', type=Path)
    ap.add_argument('candidate', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--control', action='store_true')
    a = ap.parse_args()
    result = paired(json.loads(a.reference.read_text()), json.loads(a.candidate.read_text()))
    if a.control:
        result['control_pass'] = control_passes(result)
        result['control_bounds'] = {'top1_agreement_min': 0.95, 'mean_abs_logprob_max': 0.05, 'absolute_window_nll_delta_max': 0.10}
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, allow_nan=False))
    if a.control and not result['control_pass']:
        raise SystemExit('native BF16 control failed; do not attribute candidate differences to compression')


if __name__ == '__main__':
    main()
