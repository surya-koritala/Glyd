"""Pinned 2026 BF16 weight samples and complete tensor inventory; CI or pods only.
No model inference, quality score or GPU timing is inferred from these samples.
"""
import argparse
import hashlib
import json
import re
import struct
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = 'Qwen/Qwen3.8-27B'
LAST = 0.0
MAX_READ = 16 << 20


def fetch(url, start=None, end=None):
    global LAST
    for attempt in range(4):
        time.sleep(max(0.0, 2.5 - (time.monotonic() - LAST)))
        h = {'User-Agent': 'glyd-lossy-study/1'}
        if start is not None:
            h['Range'] = f'bytes={start}-{end}'
        LAST = time.monotonic()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=90) as r:
                if start is not None:
                    expected = f'bytes {start}-{end}/'
                    if r.status != 206 or not r.headers.get('Content-Range', '').startswith(expected):
                        raise RuntimeError('server did not honor the bounded byte range')
                    limit = end - start + 1
                else:
                    limit = MAX_READ
                data = r.read(limit + 1)
                if len(data) > limit or (start is not None and len(data) != limit):
                    raise RuntimeError('unexpected response size')
                return data
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError('request retries exhausted')


def group(name, info):
    if 'mtp.' in name or '.mtp' in name: return 'mtp'
    if name.startswith('visual.') or '.visual.' in name: return 'vision'
    if 'embed_tokens' in name: return 'embedding'
    if 'lm_head' in name: return 'head'
    if '.self_attn.' in name: return 'full_attention'
    if '.linear_attn.' in name:
        return 'linear_attention_matrix' if len(info['shape']) == 2 else 'linear_attention_other'
    if '.mlp.' in name: return 'mlp'
    return 'other'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--binary', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    models = json.loads((ROOT / 'models.json').read_text())
    model = models[MODEL]
    if not model['createdAt'].startswith('2026-') or not re.fullmatch(r'[0-9a-f]{40}', model['sha']):
        raise RuntimeError('expected a pinned 2026 checkpoint')
    revision = model['sha']
    base = f'https://huggingface.co/{MODEL}/resolve/{revision}/'
    config = json.loads(fetch(base + 'config.json'))
    index = json.loads(fetch(base + 'model.safetensors.index.json'))
    (a.out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    (a.out / 'model.json').write_text(json.dumps(model, indent=2) + '\n')
    headers, inventory = {}, []
    for file in sorted(set(index['weight_map'].values())):
        url = base + urllib.parse.quote(file)
        head = fetch(url, 0, 262143)
        n = struct.unpack('<Q', head[:8])[0]
        if n > MAX_READ:
            raise RuntimeError('safetensors header exceeds limit')
        if n + 8 > len(head):
            head += fetch(url, len(head), n + 7)
        header = json.loads(head[8:8+n])
        headers[file] = {'data_start': n+8, 'tensors': header}
        for name, info in header.items():
            if name == '__metadata__': continue
            if index['weight_map'].get(name) != file:
                raise RuntimeError('header/index mismatch')
            begin, end = info['data_offsets']
            inventory.append({'name': name, 'file': file, 'dtype': info['dtype'], 'shape': info['shape'], 'bytes': end-begin, 'group': group(name, info)})
    if len(inventory) != len(index['weight_map']):
        raise RuntimeError('inventory missing tensors')
    (a.out / 'inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    selected = []
    for category in sorted({x['group'] for x in inventory}):
        candidates = sorted((x for x in inventory if x['group'] == category and x['dtype'] == 'BF16' and x['bytes'] >= 256), key=lambda x: x['name'])
        if not candidates: continue
        # First, middle and last tensors in every category. Small/non-BF16 tensors remain inventoried.
        selected.extend(candidates[i] for i in sorted({0, len(candidates)//2, len(candidates)-1}))
    samples = []
    for entry in selected:
        info = headers[entry['file']]['tensors'][entry['name']]
        size = min(65536, entry['bytes']) // 2 * 2
        last = entry['bytes'] - size
        for offset in sorted({0, (last//2)//2*2, last//2*2}):
            begin = headers[entry['file']]['data_start'] + info['data_offsets'][0] + offset
            data = fetch(base + urllib.parse.quote(entry['file']), begin, begin+size-1)
            ident = hashlib.sha256((entry['name']+':'+str(offset)).encode()).hexdigest()[:20]
            path = a.out / (ident+'.bf16')
            path.write_bytes(data)
            result = a.out / (ident+'.json')
            subprocess.run([str(a.binary.resolve()), str(path), str(result)], check=True)
            samples.append({**entry, 'offset': offset, 'sample_bytes': size, 'sha256': hashlib.sha256(data).hexdigest(), 'result': result.name})
            path.unlink()  # Results carry offsets and hashes; do not upload model weight bytes as an artifact.
    record = {'model': MODEL, 'revision': revision, 'createdAt': model['createdAt'], 'coverage': 'stratified tensor windows, not a whole-model compression measurement', 'samples': samples}
    (a.out / 'samples.json').write_text(json.dumps(record, indent=2) + '\n')
    aggregate = []
    for m in range(8):
        for block in (256, 1024):
            rows = [next(v for v in json.loads((a.out/s['result']).read_text())['variants'] if v['fraction_bits']==m and v['block_values']==block) for s in samples]
            source = sum(s['sample_bytes'] for s in samples)
            packed = sum(v['packed_bytes'] for v in rows)
            aggregate.append({'fraction_bits': m, 'block_values': block, 'source_sample_bytes': source, 'packed_sample_bytes': packed, 'sample_bits_per_weight': 16*packed/source, 'sample_reduction': 1-packed/source, 'max_sample_nrmse': max(v['nrmse'] for v in rows), 'exact_after_rounding': all(v['exact_after_rounding'] for v in rows)})
    (a.out / 'summary.json').write_text(json.dumps({'scope': record['coverage'], 'tensor_count': len(inventory), 'samples': len(samples), 'variants': aggregate}, indent=2)+'\n')
    print(json.dumps({'model': MODEL, 'revision': revision, 'tensor_count': len(inventory), 'samples': len(samples), 'variants': aggregate}, indent=2))


if __name__ == '__main__':
    main()
