"""Record or verify the complete resolved package environment."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


def versions():
    env=dict(os.environ)
    env.pop('PIP_REQUIRE_VIRTUALENV',None)  # read-only package listing also inspects the pinned base image
    packages=json.loads(subprocess.check_output([sys.executable,'-m','pip','list','--format=json','--disable-pip-version-check'],text=True,env=env))
    return {p['name']:importlib.metadata.version(p['name']) for p in packages}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--write',type=Path)
    ap.add_argument('--check',type=Path)
    ap.add_argument('--require-cuda',action='store_true')
    args=ap.parse_args()
    if sys.version_info[:2]!=(3,12): raise RuntimeError('Python 3.12 is required')
    actual=versions()
    if args.check and actual!=json.loads(args.check.read_text()):
        expected=json.loads(args.check.read_text())
        changed=sorted(k for k in set(actual)|set(expected) if actual.get(k)!=expected.get(k))
        raise RuntimeError('package environment differs: '+', '.join(changed))
    if args.require_cuda:
        import torch
        if not torch.cuda.is_available() or torch.version.cuda!='12.8':
            raise RuntimeError('prepared CUDA 12.8 reference stack is unavailable')
    if args.write: args.write.write_text(json.dumps(actual,indent=2,sort_keys=True)+'\n')
    print('Package environment verified' if args.check else 'Package environment recorded')


if __name__=='__main__': main()
