"""Pinned common-arithmetic weight adapter; never shipped with the product."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT.parent/'lossy-2026'
sys.path.insert(0, str(LEGACY))
from matched_quality import apply_candidate_a, validate_parameter_inventory
from protocol import file_digest, read_json


# Largest t with M 2^t <= 15 (e3m3's largest significand), M = 1..15.
E3M3_SHIFT = [0, 3, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
NONE = -(1 << 20)


def round_e3m3(words, block=128):
    """Candidate A words [rows, cols] (int16 view of BF16) as the nearest e3m3 value (sign, three exponent bits with
    bias 3, three fraction bits, subnormals, no NaN: largest 30, step 2^-5) times the smallest power-of-two scale
    covering each 1 x block span of a row, ties to even. Non-finite words are kept. Returns new words."""
    import torch
    w = words.to(torch.int32) & 0xFFFF
    rows, cols = w.shape
    e, m = (w >> 7) & 0xFF, (w >> 4) & 7
    finite = e != 0xFF
    sig = torch.where(e > 0, 8 + m, m)
    power = torch.where(e > 0, e - 130, torch.full_like(e, -129))
    k = power - 1 - torch.tensor(E3M3_SHIFT, device=w.device)[sig.long()]
    k = torch.where((sig > 0) & finite, k, torch.full_like(k, NONE))
    k = torch.nn.functional.pad(k, (0, (-cols) % block), value=NONE).view(rows, -1, block).amax(dim=2)
    k = torch.where(k == NONE, torch.zeros_like(k), k).repeat_interleave(block, dim=1)[:, :cols]
    q = torch.ldexp(words.view(torch.bfloat16).double(), -k)  # exact: a power-of-two scale
    q = torch.where(finite & (q.abs() < 0.25), torch.round(q*32)/32, q)  # the subnormal range, ties to even
    out = torch.ldexp(q, k).to(torch.bfloat16).view(torch.int16)  # at most four significant bits: exact in BF16
    return torch.where(finite, out, words)


def apply_e3m3(model, inventory, rows=1024):
    """Every two-dimensional checkpoint BF16 matrix through round_e3m3 after Candidate A; returns changed values."""
    import torch
    from matched_quality import validate_parameter_inventory
    changed = 0
    with torch.no_grad():
        for name, param in validate_parameter_inventory(model, inventory).items():
            if inventory[name]['dtype'] != 'BF16' or param.dim() != 2:
                continue
            for start in range(0, param.shape[0], rows):
                part = param[start:start+rows]
                words = part.to(torch.bfloat16)
                if param.dtype == torch.float32 and not torch.equal(words.float(), part):
                    raise ValueError(f'parameter is not a BF16 value before e3m3 rounding: {name}')
                new = round_e3m3(words.contiguous().view(torch.int16)).view(torch.bfloat16)
                changed += int((new.view(torch.int16) != words.contiguous().view(torch.int16)).sum())
                part.copy_(new.to(param.dtype))
    return changed


def registry_entry(key):
    entry = read_json(ROOT/'models.json')['models'][key]
    for name in ('bf16', 'fp8'):
        pin = entry[name]
        if not pin['created_at'].startswith('2026-') or len(pin['revision']) != 40:
            raise ValueError('only immutable 2026 model revisions are eligible')
    inventory = ROOT/entry['inventory']
    if file_digest(inventory) != entry['inventory_sha256']:
        raise ValueError('model tensor inventory changed')
    return entry


def configure_math():
    import torch
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def wrap(model, tokenizer, max_length, batch_size):
    from lm_eval.models.huggingface import HFLM
    # Cache state is not needed for fixed-choice teacher forcing.
    model.config.use_cache = False
    if hasattr(model.config, 'text_config'):
        model.config.text_config.use_cache = False
    return HFLM(pretrained=model, tokenizer=tokenizer, backend='causal',
                max_length=max_length, batch_size=batch_size, add_bos_token=False,
                truncation=False, logits_cache=True, softmax_dtype='float32')


def load(key, arm, model_dir, tokenizer_dir):
    import torch
    import transformers
    from transformers import AutoTokenizer, FineGrainedFP8Config
    entry = registry_entry(key)
    source = entry['fp8' if arm == 'fp8' else 'bf16']
    marker = read_json(Path(model_dir)/'glyd-study-source.json')
    if marker != {'model': source['repo'], 'revision': source['revision']}:
        raise ValueError('checkpoint source does not match the model registry')
    if not torch.cuda.is_available():
        raise RuntimeError('weight evaluation requires a prepared GPU pod')
    configure_math()
    extra = {}
    if arm == 'fp8':
        config = read_json(Path(model_dir)/'config.json')['quantization_config']
        if config.get('quant_method') != 'fp8' or config.get('weight_block_size') != [128, 128]:
            raise ValueError('this model entry requires block-scaled official FP8')
        extra['quantization_config'] = FineGrainedFP8Config(dequantize=True)
    cls = getattr(transformers, entry['auto_class'])
    model = cls.from_pretrained(str(model_dir), dtype=torch.bfloat16,
        device_map={'': 0}, attn_implementation='eager', trust_remote_code=False,
        local_files_only=True, **extra).eval()
    inventory = {x['name']: x for x in read_json(ROOT/entry['inventory'])}
    validate_parameter_inventory(model, inventory)
    if any(p.dtype not in (torch.bfloat16, torch.float32) for p in model.parameters()):
        raise ValueError('weight comparison did not load common BF16/FP32 arithmetic')
    rounded = apply_candidate_a(model, inventory) if arm in ('candidate-a', 'candidate-a7') else 0
    if arm == 'candidate-a7':
        rounded = {'candidate_a_values': rounded, 'e3m3_changed_values': apply_e3m3(model, inventory)}
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True,
                                               trust_remote_code=False)
    return model, tokenizer, rounded
