"""Paired suite-level accuracy decisions from complete frozen evaluation arms."""
import argparse
from collections import defaultdict
from pathlib import Path

from protocol import decision, digest, file_digest, paired_interval, read_json, read_rows, validate_arm, write_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--prepared',type=Path,required=True)
    ap.add_argument('--results',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--candidate',choices=('candidate-a','candidate-a7'),default='candidate-a')
    ap.add_argument('--references',type=Path,help='directory holding the bf16 and fp8 arms, when they are a retained earlier run')
    args=ap.parse_args()
    manifest=read_json(args.prepared/'manifest.json')
    if digest({k:v for k,v in manifest.items() if k!='request_manifest_sha256'})!=manifest['request_manifest_sha256']:
        raise ValueError('manifest identity changed')
    arms={}
    versions=None
    for arm in ('bf16','fp8',args.candidate):
        path=(args.references or args.results)/arm if arm in ('bf16','fp8') else args.results/arm
        header=read_json(path/'header.json');rows=read_rows(path/'samples.jsonl')
        if header['arm']!=arm or file_digest(path/'samples.jsonl')!=header['samples_sha256']:
            raise ValueError('arm or retained evidence mismatch')
        validate_arm(manifest,header,rows)
        if versions is None:
            versions=header['versions']
        if header['versions']!=versions:
            raise ValueError('arithmetic environment differs between arms')
        arms[arm]={r['id']:r for r in rows}
    groups=manifest['suite']['groups']
    policy=manifest['suite']['policy']
    if not 0 < policy['confidence'] < 1 or policy['comparators'] != ['bf16','fp8']:
        raise ValueError('unsupported confidence or reference policy')
    # Simultaneous family-wise coverage across all suites and both references.
    alpha=(1-policy['confidence'])/(len(groups)*2)
    comparisons={}
    for reference in ('bf16','fp8'):
        per_group={}
        for group in groups:
            ids=sorted(k for k,v in arms[reference].items() if v['group']==group)
            per_group[group]=paired_interval([arms[reference][k]['correct'] for k in ids],
                                             [arms[args.candidate][k]['correct'] for k in ids],alpha)
        comparisons[reference]={'groups':per_group,
            'macro_accuracy_loss':sum(c['accuracy_loss'] for c in per_group.values())/len(groups),
            'macro_loss_interval':[sum(c['loss_interval'][i] for c in per_group.values())/len(groups) for i in (0,1)]}
    task_scores={}
    for arm,rows in arms.items():
        buckets=defaultdict(list)
        for row in rows.values(): buckets[row['task']].append(row['correct'])
        task_scores[arm]={name:{'n':len(values),'accuracy':sum(values)/len(values)} for name,values in buckets.items()}
    result={'schema':1,'candidate':args.candidate,'profile':manifest['profile'],'model_key':manifest['model_key'],'request_manifest_sha256':manifest['request_manifest_sha256'],
            'question_count_per_arm':manifest['question_count'],'policy':policy,'comparisons':comparisons,
            'task_scores':task_scores,**decision(comparisons,policy),
            'interval_method':'paired harm-minus-gain with exact binomial limits and Bonferroni family-wise coverage',
            'scope':'six-suite zero-shot fixed-choice screen; overall product decision remains incomplete'}
    write_json(args.out,result)
    print(result['quality_decision'])


if __name__=='__main__':
    main()
