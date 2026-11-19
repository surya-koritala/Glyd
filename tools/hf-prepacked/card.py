"""A model card (README.md for glyd/NAME) from check.py's measurement of the save.

    python card.py CHECK.json "an A100 80 GB, 2026-10-06" [--serve] > README.md

--serve adds the glyd run and vllm serve lines: only once the vLLM plugin loads this family's saves (SAVES in
glyd_gpu/vllm_plugin.py); until then a card names from_pretrained alone.
"""
import json
import sys

MAKERS = {"Qwen": ("the Qwen team (Alibaba Cloud)", "Apache-2.0"), "google": ("Google", "Apache-2.0")}
# what the save holds of the original, as transformers loads it (check.py's source_groups and save_groups; cmpnames.py)
PARTS_LEFT = {"Qwen": "It holds the language model, for text: the vision tower and the multi-token prediction head are not in this repository (for images, use the original).",
              "gemma-4-E4B-it": "The key and value projections of the 18 layers that reuse earlier layers' keys and values, which the model never reads (54 tensors, 0.1 GB), are not in this repository."}


def gb(n):
    return f"{n / 1e9:.1f} GB"


def card(r, on, serve=False):
    repo = r["source"]["repo"]
    org, name = repo.split("/")
    maker, lic = MAKERS[org]
    saved = 1 - r["save_bytes"] / r["same_tensors_bf16_bytes"]
    run = [f'model = glyd.from_pretrained("glyd/{name}")']
    shell = [f"glyd run glyd/{name}", f"vllm serve glyd/{name} --quantization glyd"] if serve else []
    lines = [
        "---",
        f"license: {lic.lower()}",
        f"base_model: {repo}",
        "tags: [glyd, lossless]",
        "---",
        "",
        f"# {name}, Glyd",
        "",
        f"[{repo}](https://huggingface.co/{repo}) by {maker}, unchanged: every weight in this repository is bit for bit the original's. "
        f"Stored with [Glyd](https://getglyd.com), it takes {saved:.0%} less space on disk and loads packed into GPU memory, with no packing step on the first run."
        + "".join(f" {PARTS_LEFT[k]}" for k in (org, name) if k in PARTS_LEFT),
        "",
        "Lossless AI compression: 33% less GPU memory, bit for bit.",
        "",
        "| Measured | bf16 (the original's tensors) | Glyd |",
        "| :-- | --: | --: |",
        f"| Weights on disk | {gb(r['same_tensors_bf16_bytes'])} | **{gb(r['save_bytes'])}** ({saved:.1%} less) |",
        f"| Weights in GPU memory, loaded | {gb(r['same_tensors_bf16_bytes'])} | **{gb(r['gpu_bytes'])}** |",
        f"| Tensors checked bit for bit against the original's | | {r['verified'] + r['hashed']:,} |",
        "",
        f"Measured on {on}, Glyd {r['glyd']}.",
        "",
        "## Run it",
        "",
        "```bash",
        'pip install "glyd[gpu]"',
        *shell,
        "```",
        "```python",
        "import glyd",
        *run,
        "```",
        "",
        "## License and attribution",
        "",
        f"The model is {repo} by {maker}, under the {lic} license (LICENSE in this repository); its weights are the original's, "
        "stored in Glyd's format (the safetensors files and glyd.json differ from the original's files; NOTICE says so). "
        "Loading it takes Glyd, open source under the Apache License 2.0 (https://github.com/surya-koritala/Glyd).",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(card(json.load(open(sys.argv[1])), sys.argv[2], "--serve" in sys.argv[3:]), end="")
