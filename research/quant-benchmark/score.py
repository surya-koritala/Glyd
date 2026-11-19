"""Score one immutable weight arm on a prepared GPU; no generation or speed claim."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path

from protocol import canonical, choice_metrics, digest, file_digest, read_json, read_rows, validate_question, write_json
from model_adapter import ROOT, load, registry_entry, wrap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepared', type=Path, required=True)
    ap.add_argument('--model-dir', type=Path, required=True)
    ap.add_argument('--arm', choices=('bf16','candidate-a','candidate-a7','fp8'), required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError('arm output already exists; silent resumption is forbidden')
    args.out.mkdir(parents=True)
    manifest = read_json(args.prepared/'manifest.json')
    if digest({k:v for k,v in manifest.items() if k != 'request_manifest_sha256'}) != manifest['request_manifest_sha256']:
        raise ValueError('prepared manifest changed')
    if digest(read_json(ROOT/'suite.json')) != manifest['suite_sha256'] or digest(registry_entry(manifest['model_key'])) != manifest['model_sha256']:
        raise ValueError('suite or model registry changed after preparation')
    # Validate every input before paying the model load cost.
    for entry in manifest['files']:
        if file_digest(args.prepared/entry['file']) != entry['sha256']:
            raise ValueError('prepared questions changed')
        for row in read_rows(args.prepared/entry['file']):
            validate_question(row, manifest['suite']['max_length'])
    if file_digest(args.prepared/'tokenizer'/'tokenizer.json') != manifest['tokenizer_sha256']:
        raise ValueError('frozen tokenizer changed')
    import torch
    from lm_eval.api.instance import Instance
    model, tokenizer, rounded = load(manifest['model_key'], args.arm, args.model_dir, args.prepared/'tokenizer')
    lm = wrap(model, tokenizer, manifest['suite']['max_length'], manifest['suite']['batch_size'])
    header = {k:manifest[k] for k in ('schema','profile','suite_sha256','model_key','model_sha256','request_manifest_sha256')}
    transform = rounded if isinstance(rounded, dict) else {}
    header.update(arm=args.arm, runtime='common-bf16-eager', complete=False, errors=0,
                  rounded_bf16_values=transform.get('candidate_a_values', rounded), source_commit=manifest['source_commit'],
                  **({'e3m3_changed_values': transform['e3m3_changed_values'], 'code_commit': os.environ.get('GLYD_CODE_COMMIT')} if transform else {}),
                  versions={k:importlib.metadata.version(k) for k in ('torch','transformers','lm_eval')},
                  scope='fixed-choice weight quality; not native FP8 quality, task generation, compact GPU memory or speed')
    write_json(args.out/'header.json', header)
    count = 0
    with (args.out/'samples.jsonl').open('wb') as output:
        for entry in manifest['files']:
            rows = read_rows(args.prepared/entry['file'])
            # Bound retained logits/requests and persist progress; all rows remain required.
            for offset in range(0,len(rows),64):
                batch = rows[offset:offset+64]
                requests = []
                for row in batch:
                    for i, req in enumerate(row['requests']):
                        if lm._encode_pair(req['context'],req['continuation']) != (req['context_ids'],req['continuation_ids']):
                            raise ValueError('runtime tokenization differs from the frozen input')
                        requests.append(Instance('loglikelihood',{},(req['context'],req['continuation']),i))
                values = lm.loglikelihood(requests, disable_tqdm=True)
                if len(values) != len(requests):
                    raise ValueError('model omitted requested choice scores')
                cursor = 0
                for row in batch:
                    scores = [float(x[0]) for x in values[cursor:cursor+len(row['requests'])]]
                    cursor += len(scores)
                    metrics = choice_metrics(row,scores)
                    result = {k:row[k] for k in ('id','group','task','metric','sha256')}
                    result.update(scores=scores,correct=metrics[row['metric']],**metrics)
                    output.write(canonical(result)+b'\n')
                    count += 1
                output.flush()
                print(json.dumps({'arm':args.arm,'task':entry['task'],'completed_questions':count}),flush=True)
    if count != manifest['question_count']:
        raise ValueError('incomplete arm')
    header.update(complete=True, question_count=count, samples_sha256=file_digest(args.out/'samples.jsonl'))
    write_json(args.out/'header.json',header)


if __name__ == '__main__':
    main()
