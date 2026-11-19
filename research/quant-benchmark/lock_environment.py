"""Build the resolved non-system wheels on CI, not on a paid GPU pod."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from protocol import read_json, write_json
from environment import versions


def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);args=ap.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    local=json.loads(subprocess.check_output([sys.executable,'-m','pip','list','--local','--format=json','--disable-pip-version-check'],text=True))
    rows=sorted(f'{p["name"]}=={p["version"]}' for p in local)
    (out/'requirements-local.lock').write_text('\n'.join(rows)+'\n')
    # The unreleased evaluator is built from an immutable official source archive.
    suite=read_json(Path(__file__).with_name('suite.json'))
    source=f'https://github.com/EleutherAI/lm-evaluation-harness/archive/{suite["harness_commit"]}.zip'
    wheels=out/'wheels';wheels.mkdir(exist_ok=True)
    subprocess.run([sys.executable,'-m','pip','wheel','--no-deps','--wheel-dir',str(wheels),source],check=True)
    other=out/'requirements-wheels.lock'
    other.write_text('\n'.join(x for x in rows if x.split('==')[0].lower().replace('_','-')!='lm-eval')+'\n')
    subprocess.run([sys.executable,'-m','pip','wheel','--no-deps','--wheel-dir',str(wheels),'-r',str(other)],check=True)
    write_json(out/'resolved-versions.json',versions())


if __name__=='__main__': main()
