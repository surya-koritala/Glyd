"""Independent BF16 likelihood reference for the native engine's pinned screen, on a pod only."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model_dir', type=Path)
    ap.add_argument('evaluation', type=Path)
    ap.add_argument('output', type=Path)
    a = ap.parse_args()
    manifest = json.loads((a.model_dir/'glyd-study-source.json').read_text())
    evaluation = json.loads(a.evaluation.read_text())
    if any(manifest[k] != evaluation[k] for k in ('model', 'revision')):
        raise RuntimeError('checkpoint identity mismatch')
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText
    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_pretrained(
        str(a.model_dir), dtype=torch.bfloat16, device_map={'': 0},
        attn_implementation='eager', trust_remote_code=False, local_files_only=True).eval()
    rows = []
    with torch.inference_mode():
        for sample in evaluation['samples']:
            ids = torch.tensor([sample['input_ids']], device='cuda', dtype=torch.long)
            start = sample['score_start']
            logits = model(input_ids=ids[:, :-1], use_cache=False).logits[0, start-1:].float()
            if not torch.isfinite(logits).all():
                raise RuntimeError('non-finite reference logits')
            targets = ids[0, start:]
            lp = torch.log_softmax(logits, dim=-1).gather(1, targets[:, None]).squeeze(1)
            row = {'id': sample['id'], 'score_lp': lp.double().cpu().tolist(),
                   'score_top1': logits.argmax(dim=-1).cpu().tolist(), 'mean_nll': -lp.double().mean().item()}
            rows.append(row)
            print(json.dumps({'id': sample['id'], 'tokens': len(row['score_lp']), 'mean_nll': row['mean_nll']}), flush=True)
            del logits, targets, lp, ids
    n = sum(len(r['score_lp']) for r in rows)
    nll = -sum(sum(r['score_lp']) for r in rows)/n
    result = {'model': evaluation['model'], 'revision': evaluation['revision'],
              'evaluation_sha256': hashlib.sha256(a.evaluation.read_bytes()).hexdigest(),
              'runtime': 'transformers-bf16-eager', 'transformers': transformers.__version__, 'torch': torch.__version__,
              'scope': 'language-only held-out likelihood screen', 'scored_tokens': n,
              'mean_nll': nll, 'perplexity': math.exp(nll), 'samples': rows}
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
