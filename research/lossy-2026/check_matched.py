"""CPU preparation checks for the pinned 2026 quality comparison, run in CI."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import torch
from transformers import FineGrainedFP8Config,Qwen3_5Config,Qwen3_5ForConditionalGeneration
from transformers.integrations.finegrained_fp8 import Fp8Dequantize
from matched_quality import round_candidate_a,apply_candidate_a,validate_parameter_inventory

ROOT=Path(__file__).resolve().parent


def main():
    with tempfile.TemporaryDirectory() as temp:
        expected_path=Path(temp)/'rounded.bin'
        subprocess.run([sys.argv[1],'--rounding-reference',str(expected_path)],check=True)
        expected=torch.frombuffer(bytearray(expected_path.read_bytes()),dtype=torch.int16)
        original=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        actual=round_candidate_a(original).view(torch.int16)
        assert torch.equal(actual,expected),'tensor rounding differs from Rust for a BF16 pattern'
    # Check the loader's exact widening before the transform, including signed values and a tie.
    model=torch.nn.Module()
    model.register_parameter('weight',torch.nn.Parameter(torch.tensor([1.0,-1.0,1.0625],dtype=torch.bfloat16).float()))
    inventory={'weight':{'shape':[3],'dtype':'BF16'}}
    assert apply_candidate_a(model,inventory)==3
    assert torch.equal(model.weight,torch.tensor([1.0,-1.0,1.0]))
    try:
        validate_parameter_inventory(model,{})
        raise AssertionError('missing tensor accepted')
    except ValueError:
        pass
    # Materialize only shapes for the actual 2026 architecture; no checkpoint download or model execution.
    config=Qwen3_5Config.from_dict(json.loads((ROOT/'qwen38-config.json').read_text()))
    with torch.device('meta'):
        real_model=Qwen3_5ForConditionalGeneration(config)
    actual_inventory={r['name']:r for r in json.loads((ROOT/'qwen38-inventory.json').read_text())}
    names=validate_parameter_inventory(real_model,actual_inventory)
    # Verify the exact installed dequantizer on a tiny block-scaled tensor, with common BF16 output.
    quantizer=SimpleNamespace(quantization_config=FineGrainedFP8Config(dequantize=True),dtype=torch.bfloat16)
    dequantizer=Fp8Dequantize(quantizer)
    weights=torch.ones((128,128),dtype=torch.float32).to(torch.float8_e4m3fn)
    scales=torch.tensor([[0.125]],dtype=torch.float32)
    target=torch.nn.Linear(128,128,bias=False,dtype=torch.bfloat16)
    result=dequantizer.convert({'weight$':[weights],'weight_scale_inv':[scales]},model=target,full_layer_name='weight')
    value=result['weight'][0] if isinstance(result['weight'],list) else result['weight']
    assert value.dtype==torch.bfloat16 and torch.equal(value,torch.full((128,128),0.125,dtype=torch.bfloat16))
    print(json.dumps({'rounding_patterns':65536,'mapped_model_parameters':len(names),'fp8_dequantizer_checked':True}))


if __name__=='__main__':
    main()
