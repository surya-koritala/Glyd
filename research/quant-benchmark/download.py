"""Download and verify only the checkpoints declared by the prepared manifest."""
import argparse
from pathlib import Path
import time
from huggingface_hub import snapshot_download
from protocol import file_digest, read_json, write_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--prepared',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--arms',default='bf16,fp8',help='checkpoints to fetch (bf16, fp8)')
    args=ap.parse_args()
    manifest=read_json(args.prepared/'manifest.json')
    audit=read_json(args.prepared/'model-audit.json')
    if file_digest(args.prepared/'model-audit.json')!=manifest['model_audit_sha256']:
        raise ValueError('model download audit changed')
    paths={}
    wanted=args.arms.split(',')
    if not wanted or set(wanted)-{'bf16','fp8'}:
        raise ValueError('unknown checkpoint')
    for arm in ('bf16','fp8'):
        if arm not in wanted:
            continue
        source=audit[arm]
        expected=manifest['model'][arm]
        if source['repo']!=expected['repo'] or source['revision']!=expected['revision']:
            raise ValueError('checkpoint identity differs from registry')
        time.sleep(2.5)
        path=Path(snapshot_download(source['repo'],revision=source['revision'],token=False,max_workers=1,
            allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','*.model']))
        actual={f.name for f in path.glob('*.safetensors')}
        if actual!={s['name'] for s in source['shards']}:
            raise ValueError('downloaded tensor shard coverage differs')
        for shard in source['shards']:
            f=path/shard['name']
            if f.stat().st_size!=shard['bytes'] or file_digest(f)!=shard['sha256']:
                raise ValueError('downloaded checkpoint shard content differs')
        write_json(path/'glyd-study-source.json',{'model':source['repo'],'revision':source['revision']})
        paths[arm]=str(path)
        print(f'{arm}: verified all {len(source["shards"])} checkpoint shards',flush=True)
    write_json(args.out,paths)


if __name__=='__main__': main()
