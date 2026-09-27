"""Will a model fit a GPU, in bf16 and with Glyd: the rule of the fits on
getglyd.com (its scripts/site_data.py), standard library only.

    >>> glyd.fit("Qwen/Qwen3-32B", gpu="48GB")
    Qwen3-32B on a 48 GB GPU: bf16 needs 69.3 GB, no; Glyd 47.9 GB, fits

A model needs its weights, a KV cache for `context` tokens and 1.5 GiB
for the runtime, against the GPU's memory as nvidia-smi reports it. The
weights: the checkpoint's safetensors files (those at its top; Mistral's
consolidated copy left out where the repo has both); with Glyd, its bf16
and f16 tensors at 0.673 of their bytes (the tiered layout's mean over
the models measured) and the rest (FP8, 4-bit, f32) as they are. The KV
cache, bf16: every full-attention layer's keys and values for all the
tokens, a sliding-window layer's for its window, an MLA layer's latent
and rope key, none for a linear-attention layer. From the Hugging Face
Hub for a repo id (HF_TOKEN, or the token `hf auth login` saved, for a
gated one), from the files for a directory.
"""
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from .format import MANIFEST, header

RATIO = 0.673  # Glyd's bytes over bf16's, the tiered layout's mean
RUNTIME = 1.5 * 2**30  # bytes the runtime takes: CUDA context, activations, allocator slack
GPUS = {"16GB": 16376, "24GB": 24564, "32GB": 32607, "48GB": 49140, "80GB": 81559, "96GB": 97887, "141GB": 143771}  # MiB, as nvidia-smi reports them


@dataclass
class Fit:
    model: str
    gpu: str  # "48 GB"
    gpu_bytes: int
    context: int  # tokens in the KV cache
    format: str  # the checkpoint's weights: bf16, fp8 or 4-bit
    bf16_weights: float  # bytes
    glyd_weights: float
    kv_cache: float
    runtime: float
    bf16_needs: float  # weights, KV cache and runtime
    glyd_needs: float
    bf16_fits: bool
    glyd_fits: bool

    def __repr__(self):
        say = lambda fits: "fits" if fits else "no"
        return f"{self.model} on a {self.gpu} GPU: bf16 needs {self.bf16_needs / 1e9:.1f} GB, {say(self.bf16_fits)}; Glyd {self.glyd_needs / 1e9:.1f} GB, {say(self.glyd_fits)}"


def kv_bytes(config, context):
    """The KV cache for `context` tokens, bf16, from a model's config."""
    t = config.get("text_config", config)
    L = t.get("num_hidden_layers") or 0
    kinds = t.get("layer_types") or []
    if t.get("kv_lora_rank"):  # MLA: the latent and the rope key, a token a layer
        per = (t["kv_lora_rank"] + (t.get("qk_rope_head_dim") or 0)) * 2
        return (L - kinds.count("linear_attention") if kinds else L) * per * context
    heads = t.get("num_key_value_heads") or t.get("num_attention_heads") or 0
    per = 2 * heads * (t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"]) * 2
    window = t.get("sliding_window") or 0
    if kinds:
        full, slide = kinds.count("full_attention"), kinds.count("sliding_attention")
    elif t.get("sliding_window_pattern"):
        full = L // t["sliding_window_pattern"]
        slide = L - full
    elif t.get("full_attention_interval"):  # the others linear attention
        full, slide = L // t["full_attention_interval"], 0
    else:
        full, slide = L, 0
    if window >= 32768:  # Qwen2's sliding_window, with use_sliding_window off
        full, slide = full + slide, 0
    return full * per * context + slide * per * min(context, window or context)


def checkpoint(files):
    """Of [(path, bytes)], the checkpoint's safetensors: those at the top (original/ and the like hold other copies or
    other models), Mistral's consolidated copy left out where the repo has both layouts."""
    st = [(p, n) for p, n in files if p.endswith(".safetensors") and "/" not in p]
    if any(not p.startswith("consolidated") for p, _ in st):
        st = [(p, n) for p, n in st if not p.startswith("consolidated")]
    return st


def _get(url):
    """JSON from the Hugging Face Hub (HF_ENDPOINT), with the user's token where there is one."""
    token = os.environ.get("HF_TOKEN")
    saved = os.path.join(os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")), "token")
    if not token and os.path.exists(saved):
        with open(saved) as f:
            token = f.read().strip()
    headers = {"User-Agent": "glyd"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urllib.request.urlopen(urllib.request.Request(os.environ.get("HF_ENDPOINT", "https://huggingface.co") + url, headers=headers), timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403, 404):
            raise ValueError(f"{url}: no such repo on the Hub, or a gated or private one (then set HF_TOKEN, or run hf auth login)") from e
        raise


def _hub(repo):
    """A repo's config, its safetensors parameters by dtype and its files [(path, bytes)], from the Hub's API."""
    info = _get(f"/api/models/{repo}")
    files = [(f["path"], f.get("size", 0)) for f in _get(f"/api/models/{repo}/tree/main") if f.get("type") == "file"]
    return _get(f"/{repo}/resolve/main/config.json"), (info.get("safetensors") or {}).get("parameters") or {}, files


def _local(path):
    """A directory's config, its safetensors parameters by dtype (from their headers) and its files."""
    with open(os.path.join(path, "config.json")) as f:
        config = json.load(f)
    files = [(n, os.path.getsize(os.path.join(path, n))) for n in os.listdir(path) if os.path.isfile(os.path.join(path, n))]
    params = {}
    for name, _ in checkpoint(files):
        for k, t in header(os.path.join(path, name)).items():
            if k != "__metadata__":
                n = 1
                for d in t["shape"]:
                    n *= d
                params[t["dtype"]] = params.get(t["dtype"], 0) + n
    return config, params, files


def fit(name_or_path, gpu="48GB", context=8192):
    """Whether a model (a Hugging Face repo id, or a directory) fits a GPU in
    bf16 and with Glyd, and what each needs (weights, a KV cache for
    `context` tokens, 1.5 GiB for the runtime). gpu: 16GB, 24GB, 32GB,
    48GB, 80GB, 96GB or 141GB (the memory nvidia-smi reports for each), or
    the memory in bytes."""
    if isinstance(gpu, str):
        tier = gpu.replace(" ", "").upper()
        if tier not in GPUS:
            raise ValueError(f"gpu: one of {', '.join(GPUS)}, or its memory in bytes")
        label, memory = f"{tier[:-2]} GB", GPUS[tier] * 2**20
    else:
        label, memory = f"{gpu / 1e9:.1f} GB", gpu
    config, params, files = _local(name_or_path) if os.path.isdir(name_or_path) else _hub(name_or_path)
    if any(p == MANIFEST for p, _ in files):
        raise ValueError(f"{name_or_path} is a glyd-v1 checkpoint (its source is named in {MANIFEST}): fit the source")
    weights = sum(n for _, n in checkpoint(files))
    bf16 = 2 * (params.get("BF16", 0) + params.get("F16", 0))
    other = max(weights - bf16, 0)  # FP8, 4-bit, f32: kept as they are
    fmt = "bf16" if other < 0.05 * weights else "fp8" if params.get("F8_E4M3", 0) > max(params.get("U8", 0), params.get("I8", 0)) else "4-bit"
    glyd = bf16 * RATIO + other
    kv = kv_bytes(config, context)
    return Fit(name_or_path.rstrip("/").split("/")[-1], label, memory, context, fmt, weights, glyd, kv, RUNTIME,
               weights + kv + RUNTIME, glyd + kv + RUNTIME, weights + kv + RUNTIME <= memory, glyd + kv + RUNTIME <= memory)
