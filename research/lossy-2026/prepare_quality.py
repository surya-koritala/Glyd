"""Prepare a pinned held-out likelihood screen on CI; no calibration or inference."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

MODEL = 'Qwen/Qwen3.8-27B'
REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
DATASET = 'Salesforce/wikitext'
DATA_REV = 'b08601e04326c79dfdd32d625aee71d232d685c3'
DATA_FILE = 'wikitext-2-raw-v1/test-00000-of-00001.parquet'
DATA_SHA = '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'


def choose_windows(tokens, count=8, length=512, prefix=256):
    if count < 1 or len(tokens) < count * length or not 0 < prefix < length:
        raise ValueError('insufficient held-out tokens or invalid score interval')
    starts = [i * (len(tokens)-length) // max(count-1, 1) for i in range(count)]
    return [{'id': f'wikitext-test-offset-{start}', 'dataset_token_offset': start,
             'input_ids': tokens[start:start+length], 'score_start': prefix,
             'token_ids_sha256': hashlib.sha256(json.dumps(tokens[start:start+length], separators=(',', ':')).encode()).hexdigest()}
            for start in starts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--windows',type=int,default=8)
    ap.add_argument('--length',type=int,default=512)
    ap.add_argument('--prefix',type=int,default=256)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    data = urllib.request.urlopen(f'https://huggingface.co/datasets/{DATASET}/resolve/{DATA_REV}/{DATA_FILE}', timeout=90).read(2 << 20)
    if hashlib.sha256(data).hexdigest() != DATA_SHA:
        raise RuntimeError('held-out dataset hash mismatch')
    import pyarrow.parquet as pq
    import pyarrow as pa
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
    rows = pq.read_table(pa.BufferReader(data)).column('text').to_pylist()
    text = '\n\n'.join(rows)
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if not 1 <= a.windows <= 64 or not 2 <= a.length <= 2047:
        raise ValueError('evaluation exceeds the native scorer bounds')
    samples = choose_windows(encoded,a.windows,a.length,a.prefix)
    result = {'model': MODEL, 'revision': REVISION,
              'dataset': DATASET, 'dataset_revision': DATA_REV, 'dataset_file': DATA_FILE, 'dataset_sha256': DATA_SHA,
              'dataset_license': 'CC-BY-SA-3.0 and GFDL (WikiText source)',
              'dataset_text_sha256': hashlib.sha256(text.encode()).hexdigest(),
              'dataset_token_count': len(encoded),
              'protocol': f'{a.windows} disjoint {a.length}-token windows, evenly spaced through test rows joined by two newlines; {a.prefix} context tokens and {a.length-a.prefix} teacher-forced targets; raw text, no chat template or calibration',
              'scope': 'small language-likelihood screen; not task accuracy or multimodal acceptance',
              'samples': samples}
    p = a.out / 'evaluation.json'
    p.write_text(json.dumps(result, indent=2)+'\n')
    (a.out / 'evaluation.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest()+'  evaluation.json\n')
    print(json.dumps({'dataset_token_count': len(encoded), 'samples': len(samples), 'scored_tokens': sum(len(r['input_ids'])-r['score_start'] for r in samples),
                      'evaluation_sha256': hashlib.sha256(p.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
