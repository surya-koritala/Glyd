"""Versioned records and paired decisions for fixed-choice weight evaluation."""
import hashlib
import json
import math
from pathlib import Path

SCHEMA = 1
ARMS = ('bf16', 'fp8', 'candidate-a', 'candidate-a7')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def read_rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_question(row, max_length):
    expected = row['sha256']
    if digest({k: v for k, v in row.items() if k != 'sha256'}) != expected:
        raise ValueError('question identity mismatch')
    count = len(row['requests'])
    if count < 2 or len(row['choice_lengths']) != count or not row['gold']:
        raise ValueError('invalid choice/answer coverage')
    if any(type(i) is not int or not 0 <= i < count for i in row['gold']):
        raise ValueError('invalid answer index')
    if row['metric'] not in ('acc', 'acc_norm') or any(n <= 0 for n in row['choice_lengths']):
        raise ValueError('invalid metric or choice length')
    for req in row['requests']:
        context, continuation = req['context_ids'], req['continuation_ids']
        if not context or not continuation or len(context)+len(continuation)-1 > max_length:
            raise ValueError('empty or overlong input; truncation is forbidden')
        if any(type(t) is not int or t < 0 for t in context+continuation):
            raise ValueError('invalid token input')


def choice_metrics(row, scores):
    if len(scores) != len(row['requests']) or not all(math.isfinite(x) for x in scores):
        raise ValueError('incomplete or non-finite choice scores')
    raw = max(range(len(scores)), key=lambda i: scores[i])
    norm = max(range(len(scores)), key=lambda i: scores[i]/row['choice_lengths'][i])
    return {'acc': int(raw in row['gold']), 'acc_norm': int(norm in row['gold']),
            'prediction': raw, 'prediction_norm': norm}


def validate_arm(manifest, header, rows):
    for key in ('schema', 'profile', 'suite_sha256', 'model_key', 'model_sha256', 'request_manifest_sha256', 'source_commit'):
        if header.get(key) != manifest[key]:
            raise ValueError(f'arm identity differs: {key}')
    if header.get('arm') not in ARMS or header.get('runtime') != 'common-bf16-eager':
        raise ValueError('unexpected arm or arithmetic')
    if not header.get('complete') or header.get('errors') != 0:
        raise ValueError('partial or failed arm cannot establish a baseline')
    if len(rows) != manifest['question_count'] or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('missing or duplicate evaluated questions')
    expected = {r['id']: r for r in manifest['question_index']}
    if set(expected) != {r['id'] for r in rows}:
        raise ValueError('evaluated question set differs')
    for row in rows:
        original = expected[row['id']]
        if any(row.get(k) != original[k] for k in ('sha256', 'group', 'task', 'metric')):
            raise ValueError('evaluated question identity differs')
        if type(row.get('correct')) is not int or row['correct'] not in (0, 1):
            raise ValueError('accuracy must be a scored binary outcome')
        if len(row['scores']) != original['choices'] or not all(math.isfinite(x) for x in row['scores']):
            raise ValueError('invalid retained scores')
        scoring = {'requests': [None]*original['choices'], 'gold': original['gold'],
                   'choice_lengths': original['choice_lengths']}
        metrics = choice_metrics(scoring, row['scores'])
        if row['correct'] != metrics[original['metric']] or any(row.get(k) != v for k,v in metrics.items()):
            raise ValueError('reported accuracy differs from retained scores')


def paired_interval(reference, candidate, alpha):
    """Conservative exact interval for paired binary accuracy loss.

    Bound the multinomial harm and gain probabilities separately using exact
    Clopper-Pearson binomial limits with a union bound over their four tails.
    Unlike an ordinary bootstrap, identical observed answers retain uncertainty.
    """
    from scipy.stats import beta
    if len(reference) != len(candidate) or not reference or not 0 < alpha < 1:
        raise ValueError('invalid paired observations or confidence level')
    if any(type(x) is not int or x not in (0, 1) for x in reference+candidate):
        raise ValueError('non-binary observations')
    n = len(reference)
    harm = sum(a == 1 and b == 0 for a, b in zip(reference, candidate))
    gain = sum(a == 0 and b == 1 for a, b in zip(reference, candidate))
    def bounds(k):
        return (0.0 if k == 0 else float(beta.ppf(alpha/4, k, n-k+1)),
                1.0 if k == n else float(beta.ppf(1-alpha/4, k+1, n-k)))
    hl, hu = bounds(harm)
    gl, gu = bounds(gain)
    return {'n': n, 'reference_accuracy': sum(reference)/n,
            'candidate_accuracy': sum(candidate)/n, 'harm_count': harm, 'gain_count': gain,
            'accuracy_loss': (harm-gain)/n, 'loss_interval': [hl-gu, hu-gl]}


def decision(comparisons, policy):
    task_limit = policy['max_task_loss_pp']/100
    mean_limit = policy['max_macro_loss_pp']/100
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in (task_limit, mean_limit)):
        raise ValueError('invalid accuracy loss margin')
    if not comparisons or any(not c['groups'] for c in comparisons.values()):
        raise ValueError('missing decision comparisons')
    failed, uncertain = [], []
    for label, comparison in comparisons.items():
        tests = [(name, value['loss_interval'], task_limit)
                 for name, value in comparison['groups'].items()]
        tests.append(('macro', comparison['macro_loss_interval'], mean_limit))
        for name, (lo, hi), limit in tests:
            if lo > limit:
                failed.append(f'{label}/{name}')
            elif hi > limit:
                uncertain.append(f'{label}/{name}')
    return {'quality_decision': 'refocus' if failed else 'inconclusive' if uncertain else 'continue',
            'failed': failed, 'uncertain': uncertain,
            'overall_decision': 'incomplete',
            'remaining_gates': ['generated_math', 'instruction_following', 'code_execution',
                                'native_runtime_outliers', 'multimodal_quality', 'compact_gpu_memory', 'serving_speed']}
