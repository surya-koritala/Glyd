"""B2 on the GPU: Qwen3-0.6B's checkpoint without one layer's k_proj weight, served with --quantization glyd, must be
refused naming the layer and the piece (vLLM's bf16 refuses the same checkpoint)."""
import glob, json, os, shutil, sys, tempfile
from safetensors.torch import load_file, save_file

src = glob.glob(os.path.expanduser("~/hf/hub/models--Qwen--Qwen3-0.6B/snapshots/*"))[0]
d = tempfile.mkdtemp(prefix="qwen3-0.6b-nok-")
for f in os.listdir(src):
    if not f.endswith(".safetensors"):
        shutil.copy(os.path.join(src, f), d)
t = load_file(os.path.join(src, "model.safetensors"))
del t["model.layers.3.self_attn.k_proj.weight"]
save_file(t, os.path.join(d, "model.safetensors"), metadata={"format": "pt"})
from vllm import LLM

for q in ("glyd", None):
    try:
        LLM(model=d, quantization=q, dtype="bfloat16", gpu_memory_utilization=0.5, max_model_len=1024, enforce_eager=True)
        print(f"{q or 'bf16'}: loaded (not refused)")
    except Exception as e:
        print(f"{q or 'bf16'}: refused: {type(e).__name__}: {str(e)[:300]}")
shutil.rmtree(d)
