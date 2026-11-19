"""A calibrated checkpoint end to end on a tiny Qwen3.5 on the CPU (GLY-123): calib.py quantize writes the codes and scales, huggingface/plan.py --lossy plans the save, glyd-gpu pack --plan --codes packs them,
glyd-gpu verify decodes every pack, and calib.py check finds each calibrated matrix decoded to its codes times its scales, bit for bit, which are the weights the calibrated model holds. Needs torch,
transformers and cargo (GLYD_GPU names a built glyd-gpu instead).

    python3 research/lossy-2026/test_calib_pack.py
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

import calib

ROOT = Path(__file__).resolve().parents[2]


def glyd_gpu():
    if os.environ.get('GLYD_GPU'):
        return os.environ['GLYD_GPU']
    subprocess.run(['cargo', 'build', '-q', '--manifest-path', str(ROOT / 'glyd-gpu/Cargo.toml'), '--bin', 'glyd-gpu'], check=True)
    return str(Path(os.environ.get('CARGO_TARGET_DIR', ROOT / 'target')) / 'debug' / 'glyd-gpu')


def tiny(path):
    """A random Qwen3.5 (a linear-attention layer, a full-attention one; every matrix a packable shape but in_proj_a and in_proj_b, as in the real models; an output layer of its own) saved in `path`."""
    torch.manual_seed(0)
    cfg = Qwen3_5Config(text_config=dict(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=32, linear_num_key_heads=2,
                                         linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32, linear_conv_kernel_dim=4, max_position_embeddings=256, tie_word_embeddings=False,
                                         layer_types=['linear_attention', 'full_attention']),
                        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=128), tie_word_embeddings=False)
    model = Qwen3_5ForConditionalGeneration(cfg).to(torch.bfloat16).eval()
    with torch.no_grad():
        for n, p in model.named_parameters():  # (a trained model's norms and weights are not at their initial values)
            if 'layernorm' in n:
                p.copy_(torch.randn_like(p) * 0.3)
            elif p.dim() == 2 and '.visual.' not in n:
                p.copy_(torch.randn_like(p) * 0.05)
    model.save_pretrained(path)


class CalibratedPack(unittest.TestCase):
    def test_codes_to_pack_to_verify(self):
        binary = glyd_gpu()

        def run(*a, **k):
            r = subprocess.run([str(x) for x in a], capture_output=True, text=True, **k)
            self.assertEqual(r.returncode, 0, f'{" ".join(str(x) for x in a[:4])}: {r.stderr[-2000:]}')
            return r
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            src, plan = t / 'src', t / 'plan.json'
            tiny(src)
            run(sys.executable, ROOT / 'huggingface/plan.py', src, plan, '--lossy', env={**os.environ, 'PYTHONPATH': str(ROOT / 'glyd-gpu-py')})
            seqs = torch.randint(0, 256, (16, 64), dtype=torch.int32)
            for bits, layout, level in ((5, 'i5', 'swift'), (6, 'i6', 'kestrel')):
                model = Qwen3_5ForConditionalGeneration.from_pretrained(src, dtype=torch.bfloat16).eval()
                codes, out = t / f'codes{bits}', t / f'out{bits}'
                cal = calib.quantize(model, seqs, bits, codes, calib.SOURCES)
                self.assertEqual((cal['method'], cal['seqs'], cal['tokens']), ('gptq-act', 16, 64))
                self.assertEqual([d['revision'] for d in cal['data']], [s['revision'] for s in calib.SOURCES])
                matrices = {k.rsplit('.', 1)[0] for k in json.loads((codes / 'model.safetensors.index.json').read_text())['weight_map']}
                self.assertEqual(len(matrices), 13)  # (layer 0: in_proj_qkv, in_proj_z, out_proj, gate, up, down; layer 1: q, k, v, o, gate, up, down; not in_proj_a, in_proj_b, the embedding, the output layer)
                self.assertFalse([m for m in matrices if 'in_proj_a' in m or 'in_proj_b' in m or 'embed' in m or 'lm_head' in m])
                packed = run(binary, 'pack', src, out, '--plan', plan, '--layout', layout, '--codes', codes)
                self.assertIn(f'{len(matrices)} from calibrated codes', packed.stdout)
                self.assertIn('decode to glyd.json\'s sha256', run(binary, 'verify', out).stdout)
                m = json.loads((out / 'glyd.json').read_text())
                self.assertEqual((m['format'], m['layout'], m['level'], m['calibration']), ('glyd-v7', layout, level, json.loads((codes / 'calibration.json').read_text())))
                n, plain = calib.check_pack(out, plan, codes)
                self.assertEqual((n, plain), (len(matrices), ['model.embed_tokens', 'lm_head']))
                # what the codes and scales make is the weights the calibrated model holds: the model the study scores
                held = model.state_dict()
                for name in matrices:
                    self.assertTrue(torch.equal(held[f'{name}.weight'], calib.values(codes, name)), name)
                # (and the check is not vacuous: the same plan packed from the weights, by the fit, is not those values)
                run(binary, 'pack', src, t / f'plain{bits}', '--plan', plan, '--layout', layout)
                with self.assertRaises(AssertionError):
                    calib.check_pack(t / f'plain{bits}', plan, codes)


if __name__ == '__main__':
    unittest.main()
