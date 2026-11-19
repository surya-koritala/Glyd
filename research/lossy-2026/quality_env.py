"""Check the prepared reference environment on CI or a pod, without loading weights."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('output', type=Path)
    ap.add_argument('--require-cuda', action='store_true')
    args = ap.parse_args()
    if sys.prefix == sys.base_prefix or sys.version_info[:2] != (3, 12):
        raise RuntimeError('reference Python must be an isolated Python 3.12 environment')
    requirements = Path(__file__).with_name('quality-requirements.txt')
    versions = {}
    for line in requirements.read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        name, expected = line.split('==')
        actual = metadata.version(name)
        if actual.split('+')[0] != expected:
            raise RuntimeError(f'{name} {actual} differs from {expected}')
        versions[name] = actual
    import torch
    from transformers import Qwen3_5ForConditionalGeneration
    if torch.version.cuda != '12.8':
        raise RuntimeError(f'reference requires CUDA 12.8 wheels, found {torch.version.cuda}')
    available = torch.cuda.is_available()
    if args.require_cuda and not available:
        raise RuntimeError('reference CUDA device is unavailable')
    result = {'python': sys.version, 'executable': sys.executable,
              'prefix': sys.prefix, 'base_prefix': sys.base_prefix,
              'versions': versions, 'cuda': torch.version.cuda,
              'cuda_available': available,
              'reference_class': Qwen3_5ForConditionalGeneration.__name__}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
