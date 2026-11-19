"""CI-only functional check using a small configuration of the 2026 architecture."""
import json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3_5Config, Qwen3_5ForConditionalGeneration
from lm_eval.api.instance import Instance
from model_adapter import LEGACY, wrap


def main():
    torch.manual_seed(17)
    raw=json.loads((LEGACY/'qwen38-config.json').read_text())
    raw['text_config'].update(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=2,num_key_value_heads=1,head_dim=16,linear_key_head_dim=16,
        linear_value_head_dim=16,linear_num_key_heads=2,linear_num_value_heads=2,
        layer_types=['linear_attention','full_attention'],full_attention_interval=2,
        max_position_embeddings=128,pad_token_id=0,bos_token_id=1,eos_token_id=2,
        partial_rotary_factor=.5,rope_parameters={'rope_type':'default','rope_theta':10000.,
        'partial_rotary_factor':.5,'mrope_section':[1,1,2],'mrope_interleaved':True})
    raw['vision_config'].update(depth=1,hidden_size=32,intermediate_size=64,num_heads=2,
        out_hidden_size=32,num_position_embeddings=16)
    model=Qwen3_5ForConditionalGeneration(Qwen3_5Config.from_dict(raw)).float().eval()
    vocab={'[PAD]':0,'[BOS]':1,'[EOS]':2,'[UNK]':3,'Question':4,':':5,'one':6,'two':7,
           'Answer':8,'yes':9,'no':10,'longer':11}
    backend=Tokenizer(WordLevel(vocab,unk_token='[UNK]'));backend.pre_tokenizer=Whitespace()
    tok=PreTrainedTokenizerFast(tokenizer_object=backend,pad_token='[PAD]',bos_token='[BOS]',
                               eos_token='[EOS]',unk_token='[UNK]')
    lm=wrap(model,tok,128,2)
    pairs=[('Question one Answer:',' yes'),('Question one Answer:',' no'),
           ('Question two longer Answer:',' one two')]
    scored=lm.loglikelihood([Instance('loglikelihood',{},p,i) for i,p in enumerate(pairs)],disable_tqdm=True)
    with torch.no_grad():
        for pair,(actual,_) in zip(pairs,scored):
            ctx,cont=lm._encode_pair(*pair)
            logits=model(input_ids=torch.tensor([ctx+cont[:-1]]),use_cache=False).logits
            expected=torch.log_softmax(logits[0,len(ctx)-1:].float(),-1).gather(1,torch.tensor(cont)[:,None]).sum().item()
            if abs(actual-expected)>1e-4:
                raise ValueError(f'evaluator target alignment differs: {actual} versus {expected}')
    print('2026 architecture adapter, batching, shared-prefix choices and multi-token target alignment pass')


if __name__=='__main__': main()
