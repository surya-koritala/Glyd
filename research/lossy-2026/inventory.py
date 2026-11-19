"""Complete Candidate A byte inventory. Run on CI or a prepared pod, never a workstation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import time
import urllib.parse
import urllib.request

from sample import MODEL, ROOT, fetch, group


def tensor_header(path, weight_map, expected):
    with path.open('rb') as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError('short safetensors header')
        length = struct.unpack('<Q', length_bytes)[0]
        if length > 16 << 20:
            raise ValueError('oversized safetensors header')
        raw = stream.read(length)
        if len(raw) != length:
            raise ValueError('truncated safetensors header')
    header = json.loads(raw)
    tensors = []
    for name, info in header.items():
        if name == '__metadata__':
            continue
        if weight_map.get(name) != path.name or name not in expected:
            raise ValueError('header, index and pinned inventory disagree')
        if info['dtype'] != expected[name]['dtype'] or info['shape'] != expected[name]['shape']:
            raise ValueError('tensor differs from pinned inventory')
        # The complete pinned checkpoint is BF16; fail rather than silently omit a new dtype.
        if info['dtype'] != 'BF16' or any(not isinstance(x, int) or x < 0 for x in info['shape']):
            raise ValueError('unexpected tensor dtype or shape')
        begin, end = info['data_offsets']
        if not isinstance(begin, int) or not isinstance(end, int) or begin < 0 or end-begin != 2*math.prod(info['shape']):
            raise ValueError('tensor byte count differs from its shape')
        tensors.append({'name': name, 'file': path.name, 'shape': info['shape'], 'dtype': info['dtype'],
                        'group': group(name, info), 'offset': length+8+begin, 'bytes': end-begin})
    cursor = length+8
    for row in sorted(tensors, key=lambda x: (x['offset'], x['bytes'])):
        if row['offset'] != cursor:
            raise ValueError('tensor data overlaps or contains a gap')
        cursor += row['bytes']
    if cursor != path.stat().st_size:
        raise ValueError('safetensors data size mismatch')
    if {r['name'] for r in tensors} != {n for n, f in weight_map.items() if f == path.name}:
        raise ValueError('missing tensor from shard')
    return tensors


def file_digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        while data := stream.read(8 << 20):
            sha.update(data)
    return sha.hexdigest()


def download(url, path, size, digest):
    part = path.with_suffix('.part')
    try:
        time.sleep(2.5)
        req = urllib.request.Request(url, headers={'User-Agent': 'glyd-weight-inventory/1'})
        with urllib.request.urlopen(req, timeout=120) as source, part.open('wb') as target:
            if source.status != 200:
                raise ValueError('expected a complete checkpoint shard')
            count = 0
            sha = hashlib.sha256()
            while data := source.read(8 << 20):
                count += len(data)
                if count > size:
                    raise ValueError('checkpoint shard exceeds pinned size')
                sha.update(data)
                target.write(data)
        if count != size or sha.hexdigest() != digest:
            raise ValueError('checkpoint shard digest or size mismatch')
        part.rename(path)
    finally:
        part.unlink(missing_ok=True)


def aggregate(rows):
    if not rows or len({r['name'] for r in rows}) != len(rows):
        raise ValueError('empty or duplicate tensor inventory')
    counts = {key: sum(r[key] for r in rows) for key in (
        'source_bytes', 'packed_bytes', 'tensor_header_bytes', 'block_header_bytes',
        'dictionary_bytes', 'field_bytes', 'exception_bytes', 'raw_bytes', 'values', 'changed_values')}
    if any(not r['exact_after_rounding'] or r['source_bytes'] != r['bytes'] for r in rows):
        raise ValueError('unverified tensor or input length mismatch')
    if counts['packed_bytes'] != sum(counts[k] for k in (
            'tensor_header_bytes', 'block_header_bytes', 'dictionary_bytes', 'field_bytes', 'exception_bytes', 'raw_bytes')):
        raise ValueError('full byte accounting mismatch')
    counts['bits_per_weight'] = 8*counts['packed_bytes']/counts['values']
    counts['reduction_vs_bf16'] = 1-counts['packed_bytes']/counts['source_bytes']
    counts['tensor_count'] = len(rows)
    counts['empirical_iid_entropy_bits_conditioned_on_tensor'] = sum(
        r['empirical_iid_entropy_bits']*r['values'] for r in rows)/counts['values']
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--binary', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--source-dir', type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    results = a.out/'tensors'
    results.mkdir(exist_ok=True)
    scratch = a.out/'scratch'
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
    (a.out/'source.json').write_text(json.dumps({'model':MODEL,'revision':revision,'metadata':metadata,'index':index},indent=2)+'\n')
    records, shards = [], []
    for filename in sorted(set(index['weight_map'].values())):
        if Path(filename).name != filename or not filename.endswith('.safetensors'):
            raise ValueError('unsafe shard filename')
        entry = files[filename]
        size, digest = entry['size'], entry['lfs']['sha256']
        if not re.fullmatch('[0-9a-f]{64}', digest) or not 8 <= size <= 8 << 30:
            raise ValueError('invalid pinned shard size or hash')
        path = (a.source_dir or scratch)/filename
        if a.source_dir:
            if path.stat().st_size != size or file_digest(path) != digest:
                raise ValueError('local checkpoint shard differs from pinned source')
        else:
            download(base+urllib.parse.quote(filename),path,size,digest)
        rows = tensor_header(path,index['weight_map'],expected)
        def analyze(row):
            output = results/(hashlib.sha256(row['name'].encode()).hexdigest()[:24]+'.json')
            subprocess.run([str(a.binary.resolve()),str(path),str(row['offset']),str(row['bytes']),str(output)],check=True)
            result = {**row, **json.loads(output.read_text())}
            output.write_text(json.dumps(result,indent=2)+'\n')
            return result
        with ThreadPoolExecutor(max_workers=2) as pool:
            records.extend(pool.map(analyze,rows))
        shards.append({'file':filename,'bytes':size,'sha256':digest,'tensor_count':len(rows)})
        (a.out/'progress.json').write_text(json.dumps({'completed':False,'shards':shards,'tensor_count':len(records)},indent=2)+'\n')
        print(json.dumps({'completed_shard':filename,'tensor_count':len(records),'source_bytes':sum(r['source_bytes'] for r in records)}),flush=True)
        if not a.source_dir:
            path.unlink()
    if {r['name'] for r in records} != set(expected):
        raise ValueError('incomplete whole-model coverage')
    summary = {'completed':True,'model':MODEL,'revision':revision,'fraction_bits':3,'block_values':1024,
               'scope':'every BF16 tensor encoded and decoded; byte inventory, not runtime VRAM or performance',
               'representation':'GLYDL27 version 1 reference format; no model artifact retained',
               'entropy_scope':'empirical independent-symbol estimate by tensor, not an absolute compression bound',
               'total':aggregate(records),'groups':{},'shards':shards}
    for category in sorted({r['group'] for r in records}):
        summary['groups'][category] = aggregate([r for r in records if r['group']==category])
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary['total']),flush=True)


if __name__ == '__main__':
    main()
