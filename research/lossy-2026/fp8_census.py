"""Exact e4m3fn fit of every Candidate A matrix value under power-of-two block scales.
Run on CI or a prepared pod, never a workstation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import urllib.parse

from inventory import download, tensor_header
from sample import MODEL, ROOT, fetch

# Matrices multiplied during text generation; the embedding is a lookup, vision and MTP are separate paths.
TEXT_PRODUCTS = {'full_attention', 'linear_attention_matrix', 'mlp', 'head'}
COUNTS = ('blocks', 'blocks_inexact', 'inexact', 'underflow')


def aggregate(rows):
    if not rows:
        return None
    total = {k: sum(r[k] for r in rows) for k in ('values', 'zeros', 'specials')}
    total['tensors'] = len(rows)
    total['geometries'] = []
    for i, first in enumerate(rows[0]['geometries']):
        g = [r['geometries'][i] for r in rows]
        if any(x['block'] != first['block'] or x['format'] != first['format'] for x in g):
            raise ValueError('inconsistent block geometries')
        item = {'format': first['format'], 'block': first['block'], **{k: sum(x[k] for x in g) for k in COUNTS}}
        item['max_block_inexact'] = max(x['max_block_inexact'] for x in g)
        item['min_scale_exponent'] = min(x['min_scale_exponent'] for x in g)
        item['max_scale_exponent'] = max(x['max_scale_exponent'] for x in g)
        item['exact_fraction'] = 1-item['inexact']/total['values']
        item['tensors_all_exact'] = sum(x['inexact'] == 0 for x in g)
        total['geometries'].append(item)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--binary', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    results = a.out/'tensors'
    scratch = a.out/'scratch'
    results.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(exist_ok=True)
    model = json.loads((ROOT/'models.json').read_text())[MODEL]
    revision = model['sha']
    if not model['createdAt'].startswith('2026-') or not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('checkpoint must be a pinned 2026 model')
    base = f'https://huggingface.co/{MODEL}/resolve/{revision}/'
    metadata = json.loads(fetch(f'https://huggingface.co/api/models/{MODEL}/revision/{revision}?blobs=true'))
    if metadata['sha'] != revision:
        raise ValueError('Hub returned another revision')
    index = json.loads(fetch(base+'model.safetensors.index.json'))
    expected = {x['name']: x for x in json.loads((ROOT/'qwen38-inventory.json').read_text())}
    if set(expected) != set(index['weight_map']):
        raise ValueError('checkpoint tensor inventory changed')
    files = {x['rfilename']: x for x in metadata['siblings']}
    records, other, shards = [], [], []
    for filename in sorted(set(index['weight_map'].values())):
        if Path(filename).name != filename or not filename.endswith('.safetensors'):
            raise ValueError('unsafe shard filename')
        entry = files[filename]
        size, digest = entry['size'], entry['lfs']['sha256']
        if not re.fullmatch('[0-9a-f]{64}', digest) or not 8 <= size <= 8 << 30:
            raise ValueError('invalid pinned shard size or hash')
        path = scratch/filename
        download(base+urllib.parse.quote(filename), path, size, digest)
        rows = tensor_header(path, index['weight_map'], expected)
        other.extend({k: r[k] for k in ('name', 'group', 'shape')} for r in rows if len(r['shape']) != 2)
        def analyze(row):
            output = results/(hashlib.sha256(row['name'].encode()).hexdigest()[:24]+'.json')
            rows_, cols = row['shape']
            subprocess.run([str(a.binary.resolve()), str(path), str(row['offset']), str(row['bytes']),
                            str(rows_), str(cols), str(output)], check=True)
            result = {**{k: row[k] for k in ('name', 'group', 'shape', 'file')}, **json.loads(output.read_text())}
            output.write_text(json.dumps(result, indent=2)+'\n')
            return result
        with ThreadPoolExecutor(max_workers=2) as pool:
            records.extend(pool.map(analyze, [r for r in rows if len(r['shape']) == 2]))
        shards.append({'file': filename, 'bytes': size, 'sha256': digest, 'tensor_count': len(rows)})
        print(json.dumps({'completed_shard': filename, 'matrices': len(records)}), flush=True)
        path.unlink()
    if {r['name'] for r in records} | {r['name'] for r in other} != set(expected):
        raise ValueError('incomplete whole-model coverage')
    summary = {
        'completed': True, 'model': MODEL, 'revision': revision, 'fraction_bits': 3,
        'scope': 'every two-dimensional BF16 tensor after Candidate A rounding; counts of values that are exactly '
                 'values of each format (e4m3fn; 7-bit e3m3 and 6-bit e2m3 with three fraction bits, subnormals and no NaN) '
                 'times the smallest power-of-two scale covering their block. Not a quality, memory or speed result.',
        'all_matrices': aggregate(records),
        'text_products': aggregate([r for r in records if r['group'] in TEXT_PRODUCTS]),
        'groups': {g: aggregate([r for r in records if r['group'] == g]) for g in sorted({r['group'] for r in records})},
        'not_matrices': {'tensors': len(other), 'values': sum(math.prod(r['shape']) for r in other)},
        'shards': shards,
    }
    (a.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k: summary[k] for k in ('all_matrices', 'text_products')}), flush=True)


if __name__ == '__main__':
    main()
