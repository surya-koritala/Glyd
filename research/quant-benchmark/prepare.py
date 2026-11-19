"""Freeze official task requests and token inputs on CI before GPU allocation."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time

from protocol import canonical, choice_metrics, digest, file_digest, read_json, validate_question, write_json
from model_adapter import ROOT, registry_entry, wrap


def audit_model(entry, output):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    api = HfApi(token=False)
    sources = {}
    for arm in ('bf16', 'fp8'):
        pin = entry[arm]
        time.sleep(2.5)
        info = api.model_info(pin['repo'], revision=pin['revision'], files_metadata=True)
        if info.sha != pin['revision'] or info.created_at.year != 2026:
            raise ValueError('model revision or release year changed')
        config = read_json(hf_hub_download(pin['repo'], 'config.json', revision=pin['revision'], token=False))
        token = Path(hf_hub_download(pin['repo'], 'tokenizer.json', revision=pin['revision'], token=False))
        shards = []
        for item in info.siblings:
            if item.rfilename.endswith('.safetensors'):
                if not item.lfs or not item.lfs.sha256:
                    raise ValueError('checkpoint shard lacks an immutable content hash')
                shards.append({'name': item.rfilename, 'bytes': item.size, 'sha256': item.lfs.sha256})
        if not shards:
            raise ValueError('no pinned checkpoint shards')
        sources[arm] = {'repo': pin['repo'], 'revision': info.sha,
                        'created_at': info.created_at.isoformat(), 'config': config,
                        'tokenizer_sha256': file_digest(token), 'shards': shards}
    a, b = sources['bf16'], sources['fp8']
    if a['tokenizer_sha256'] != b['tokenizer_sha256']:
        raise ValueError('BF16 and FP8 tokenizers differ')
    for key in entry['config_match_keys']:
        if a['config'].get(key) != b['config'].get(key):
            raise ValueError(f'FP8 reference changes configuration: {key}')
    q = b['config'].get('quantization_config', {})
    if q.get('quant_method') != 'fp8' or q.get('weight_block_size') != [128, 128]:
        raise ValueError('unexpected FP8 reference format')
    write_json(output/'model-audit.json', sources)
    time.sleep(2.5)
    snapshot_download(entry['bf16']['repo'], revision=entry['bf16']['revision'], token=False,
                      allow_patterns=['*.json', '*.txt', '*.jinja', '*.model'],
                      local_dir=output/'tokenizer', max_workers=1)
    return sources


def normalize_gold(task, doc, choices):
    gold = task.doc_to_text(doc) if task.multiple_input else task.doc_to_target(doc)
    if isinstance(gold, str):
        gold = choices.index(gold)
    if not isinstance(gold, list):
        gold = [int(gold)]
    return [int(x) for x in gold]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--profile', choices=('screen','full'), required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError('preparation output already exists')
    args.out.mkdir(parents=True)
    suite = read_json(ROOT/'suite.json')
    entry = registry_entry(args.model)
    audit = audit_model(entry, args.out)
    import torch
    import transformers
    from lm_eval.tasks import TaskManager
    from transformers import AutoConfig, AutoTokenizer
    config = AutoConfig.from_pretrained(args.out/'tokenizer', local_files_only=True, trust_remote_code=False)
    with torch.device('meta'):
        model = getattr(transformers, entry['auto_class']).from_config(config)
    tokenizer = AutoTokenizer.from_pretrained(args.out/'tokenizer', local_files_only=True, trust_remote_code=False)
    lm = wrap(model, tokenizer, suite['max_length'], suite['batch_size'])
    manager = TaskManager()
    index, records, counts, work = [], [], defaultdict(int), defaultdict(int)
    leaf_names = set()
    for task_spec in suite['tasks']:
        name = task_spec['task']
        if name in leaf_names:
            raise ValueError('duplicate task')
        leaf_names.add(name)
        # Every dataset fetch uses an immutable, reviewed public revision.
        dataset = suite['datasets'][task_spec['dataset']]
        # A top-level dict is a complete task definition in this pinned harness.
        # Apply the revision override to its registered YAML entry instead.
        task = manager._factory.build(manager.task_index[name],
            overrides={'num_fewshot': 0, 'dataset_kwargs': {'revision': dataset['revision']}},
            registry=manager.task_index)
        if task.config.task != name:
            raise ValueError('unexpected task expansion')
        if task.OUTPUT_TYPE != 'multiple_choice' or task.config.dataset_path != task_spec['dataset']:
            raise ValueError('task family or dataset changed')
        task.set_fewshot_seed(0)
        documents = task.test_docs() if task.has_test_docs() else task.validation_docs()
        cap = suite['profiles'][args.profile]['per_leaf_caps'].get(task_spec['group'])
        selected = None
        if cap is not None and cap < len(documents):
            selected = sorted(sorted(range(len(documents)),
                key=lambda i: digest([suite['selection_seed'],name,i]))[:cap])
        task.build_all_requests(rank=0, world_size=1, cache_requests=False, samples=selected)
        docs = defaultdict(list)
        for request in task.instances:
            if request.request_type != 'loglikelihood' or request.repeats != 1:
                raise ValueError('unexpected request type or repeats')
            # The pinned harness numbers an explicitly sampled subset from zero.
            # Retain the original dataset row index in the frozen evidence.
            source_id = request.doc_id if selected is None else selected[request.doc_id]
            if request.doc != documents[source_id]:
                raise ValueError('sampled request does not match its original dataset row')
            docs[source_id].append(request)
        required_ids = set(range(len(documents))) if selected is None else set(selected)
        if set(docs) != required_ids:
            raise ValueError('official request builder omitted or changed selected questions')
        path = args.out/(name+'.jsonl')
        with path.open('wb') as f:
            for doc_id, instances in sorted(docs.items()):
                instances.sort(key=lambda x: x.idx)
                doc = instances[0].doc
                choices = task.doc_to_choice(doc)
                if len(instances) != len(choices) or [x.idx for x in instances] != list(range(len(choices))):
                    raise ValueError('choice request coverage differs')
                requests = []
                one_token_contexts = set()
                for instance in instances:
                    context, continuation = instance.args
                    context_ids, continuation_ids = lm._encode_pair(context, continuation)
                    requests.append({'context': context, 'continuation': continuation,
                                     'context_ids': context_ids, 'continuation_ids': continuation_ids})
                    work['forward_input_tokens_uncached'] += len(context_ids)+len(continuation_ids)-1
                    work['scored_choice_tokens'] += len(continuation_ids)
                    work['requests'] += 1
                    work['maximum_input_tokens'] = max(work['maximum_input_tokens'],len(context_ids)+len(continuation_ids)-1)
                    if len(continuation_ids) == 1:
                        one_token_contexts.add(tuple(context_ids))
                    else:
                        work['forward_input_tokens_cached_upper'] += len(context_ids)+len(continuation_ids)-1
                work['forward_input_tokens_cached_upper'] += sum(len(x) for x in one_token_contexts)
                row = {'id': f'{name}/{doc_id}', 'group': task_spec['group'], 'task': name,
                       'metric': task_spec['metric'], 'gold': normalize_gold(task, doc, choices),
                       'choice_lengths': [len(x) for x in choices], 'requests': requests}
                row['sha256'] = digest(row)
                validate_question(row, suite['max_length'])
                # Check our retained-score reduction against the official task implementation.
                for winner in range(len(choices)):
                    scores = [-100.0*len(x) for x in choices]
                    scores[winner] = -0.01
                    official = task.process_results(doc, [(v, False) for v in scores])
                    actual = choice_metrics(row, scores)
                    if actual[row['metric']] != official[row['metric']]:
                        raise ValueError('frozen metric differs from the official task')
                f.write(canonical(row)+b'\n')
                index.append({k: row[k] for k in ('id','group','task','metric','sha256','gold','choice_lengths')} | {'choices':len(choices)})
                counts[row['group']] += 1
        split = task.config.test_split or task.config.validation_split
        records.append({'task': name, 'group': task_spec['group'], 'metric': task_spec['metric'],
                        'file': path.name, 'sha256': file_digest(path), 'questions': len(docs),
                        'dataset': task_spec['dataset'], 'revision': dataset['revision'],
                        'dataset_config': task.config.dataset_name, 'split': split,
                        'available_questions': len(documents),
                        'selected_indices': selected,
                        'dataset_fingerprint': task.dataset[split]._fingerprint})
        print(json.dumps({'task': name, 'questions': len(docs), 'complete': True}), flush=True)
        del task, docs
        time.sleep(2.5)
    if set(counts) != set(suite['groups']):
        raise ValueError('missing benchmark group')
    manifest = {'schema': 1, 'suite_sha256': digest(suite), 'profile': args.profile, 'model_key': args.model,
                'model_sha256': digest(entry), 'source_commit': os.environ['GITHUB_SHA'],
                'harness_commit': suite['harness_commit'], 'suite': suite,
                'model': entry, 'model_audit_sha256': file_digest(args.out/'model-audit.json'),
                'question_count': len(index), 'group_counts': dict(counts),
                'work': dict(work), 'files': records, 'question_index': index,
                'tokenizer_sha256': audit['bf16']['tokenizer_sha256']}
    manifest['request_manifest_sha256'] = digest(manifest)
    write_json(args.out/'manifest.json', manifest)
    print(json.dumps({'questions': len(index), 'groups': counts, 'work': work}), flush=True)


if __name__ == '__main__':
    main()
