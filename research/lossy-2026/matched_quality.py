"""Compare weight representations with common BF16 arithmetic on a prepared GPU pod."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent
MODEL='Qwen/Qwen3.8-27B'
FP8_MODEL=MODEL+'-FP8'


def round_candidate_a(value):
    import torch
    if value.dtype != torch.bfloat16:
        raise ValueError('Candidate A rounding requires BF16 input')
    words=value.contiguous().view(torch.int16).to(torch.int32)&65535
    rounded=(words+7+((words>>4)&1))&65520
    finite=(words&0x7f80)!=0x7f80
    overflow=(rounded&0x7fff)>=0x7f80
    rounded=torch.where(overflow,(words&0x8000)|0x7f70,rounded)
    return torch.where(finite,rounded,words).to(torch.int16).view(torch.bfloat16)


def validate_parameter_inventory(model, inventory):
    parameters=dict(model.named_parameters())
    for name, value in parameters.items():
        if name not in inventory or list(value.shape)!=inventory[name]['shape']:
            raise ValueError(f'loaded parameter has no matching pinned tensor: {name}')
    text={n for n in inventory if '.visual.' not in n and '.mtp.' not in n and not n.startswith('mtp.')}
    if text-set(parameters):
        raise ValueError('loaded model omits a pinned text parameter')
    return parameters


def apply_candidate_a(model, inventory):
    import torch
    parameters=validate_parameter_inventory(model,inventory)
    count=0
    with torch.no_grad():
        for name, param in parameters.items():
            if inventory[name]['dtype']!='BF16':
                continue
            if param.dtype not in (torch.bfloat16,torch.float32) or not param.is_contiguous():
                raise ValueError(f'unsupported loaded representation: {name}')
            flat=param.view(-1)
            for start in range(0,flat.numel(),1<<20):
                part=flat[start:start+(1<<20)]
                original=part.to(torch.bfloat16)
                # Some BF16 checkpoint parameters are widened by the model loader.
                if param.dtype==torch.float32 and not torch.equal(original.float(),part):
                    raise ValueError(f'parameter changed before Candidate A rounding: {name}')
                part.copy_(round_candidate_a(original).to(param.dtype))
            count+=param.numel()
    return count


def score(model,evaluation,device):
    import torch
    rows=[]
    with torch.inference_mode():
        for sample in evaluation['samples']:
            ids=torch.tensor([sample['input_ids']],device=device,dtype=torch.long)
            start=sample['score_start']
            logits=model(input_ids=ids[:,:-1],use_cache=False).logits[0,start-1:].float()
            if not torch.isfinite(logits).all():
                raise ValueError('non-finite likelihood logits')
            targets=ids[0,start:]
            lp=torch.log_softmax(logits,dim=-1).gather(1,targets[:,None]).squeeze(1)
            if len(lp)!=len(sample['input_ids'])-start:
                raise ValueError('scored token count changed')
            rows.append({'id':sample['id'],'score_lp':lp.double().cpu().tolist(),
                         'score_top1':logits.argmax(dim=-1).cpu().tolist()})
            print(json.dumps({'id':sample['id'],'tokens':len(lp)}),flush=True)
            del ids,logits,targets,lp
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('model_dir',type=Path)
    ap.add_argument('evaluation',type=Path)
    ap.add_argument('output',type=Path)
    ap.add_argument('--kind',choices=['bf16','candidate-a','fp8'],required=True)
    a=ap.parse_args()
    models=json.loads((ROOT/'models.json').read_text())
    source_model=FP8_MODEL if a.kind=='fp8' else MODEL
    source=json.loads((a.model_dir/'glyd-study-source.json').read_text())
    if source!={'model':source_model,'revision':models[source_model]['sha']}:
        raise ValueError('source checkpoint identity mismatch')
    evaluation=json.loads(a.evaluation.read_text())
    if evaluation['model']!=MODEL or evaluation['revision']!=models[MODEL]['sha']:
        raise ValueError('evaluation checkpoint mismatch')
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText,FineGrainedFP8Config
    if not torch.cuda.is_available():
        raise RuntimeError('matched model scoring requires the prepared GPU')
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False
    extra={}
    if a.kind=='fp8':
        config=json.loads((a.model_dir/'config.json').read_text())['quantization_config']
        if config.get('quant_method')!='fp8' or config.get('weight_block_size')!=[128,128]:
            raise ValueError('unexpected pinned FP8 representation')
        extra['quantization_config']=FineGrainedFP8Config(dequantize=True)
    model=AutoModelForImageTextToText.from_pretrained(str(a.model_dir),dtype=torch.bfloat16,
        device_map={'':0},attn_implementation='eager',trust_remote_code=False,local_files_only=True,**extra).eval()
    inventory={r['name']:r for r in json.loads((ROOT/'qwen38-inventory.json').read_text())}
    validate_parameter_inventory(model,inventory)
    if any(p.dtype not in (torch.bfloat16,torch.float32) for p in model.parameters()):
        raise ValueError('comparison did not load common BF16/FP32 arithmetic')
    rounded=apply_candidate_a(model,inventory) if a.kind=='candidate-a' else 0
    rows=score(model,evaluation,'cuda')
    count=sum(len(r['score_lp']) for r in rows)
    nll=-sum(sum(r['score_lp']) for r in rows)/count
    result={'model':MODEL,'revision':models[MODEL]['sha'],'source_checkpoint':source,
            'representation':a.kind,'rounded_bf16_values':rounded,
            'evaluation_sha256':hashlib.sha256(a.evaluation.read_bytes()).hexdigest(),
            'runtime':'transformers-common-bf16-eager','transformers':transformers.__version__,'torch':torch.__version__,
            'scope':'weight-quality comparison with common BF16 arithmetic; no native FP8 speed or VRAM result',
            'scored_tokens':count,'mean_nll':nll,'perplexity':math.exp(nll),'samples':rows}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
