"""Verify configuration and vocabulary identity for the pinned FP8 quality reference on CI."""
import argparse
import json
from pathlib import Path
from sample import fetch,ROOT,MODEL


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    models=json.loads((ROOT/'models.json').read_text())
    sources={}
    for name in [MODEL,MODEL+'-FP8']:
        pin=models[name]
        if not pin['createdAt'].startswith('2026-'):
            raise ValueError('comparison model must be released in 2026')
        metadata=json.loads(fetch(f'https://huggingface.co/api/models/{name}/revision/{pin["sha"]}?blobs=true'))
        if metadata['sha']!=pin['sha']:
            raise ValueError('unexpected checkpoint revision')
        config=json.loads(fetch(f'https://huggingface.co/{name}/resolve/{pin["sha"]}/config.json'))
        tokenizer=next(r for r in metadata['siblings'] if r['rfilename']=='tokenizer.json')
        sources[name]={'revision':pin['sha'],'createdAt':pin['createdAt'],'config':config,
                       'tokenizer_file':tokenizer,'model_card_base':metadata.get('cardData',{}).get('base_model')}
    original,fp8=(sources[n] for n in [MODEL,MODEL+'-FP8'])
    for key in ('text_config','vision_config','model_type','tie_word_embeddings','image_token_id','video_token_id','vision_start_token_id','vision_end_token_id'):
        if original['config'].get(key)!=fp8['config'].get(key):
            raise ValueError(f'FP8 reference changes model configuration: {key}')
    def identity(row):
        return row.get('lfs',{}).get('sha256') or row.get('blobId')
    if not identity(original['tokenizer_file']) or identity(original['tokenizer_file'])!=identity(fp8['tokenizer_file']):
        raise ValueError('FP8 checkpoint tokenizer differs from the BF16 vocabulary')
    q=fp8['config'].get('quantization_config',{})
    if q.get('quant_method')!='fp8' or q.get('weight_block_size')!=[128,128]:
        raise ValueError('unexpected FP8 checkpoint quantization')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps({'sources':sources,'config_and_tokenizer_match':True,
        'scope':'pinned first-party checkpoints; configuration and vocabulary match, not a proof of conversion provenance'},indent=2)+'\n')
    print('Pinned BF16 and FP8 configuration and tokenizer identities match')


if __name__=='__main__':
    main()
