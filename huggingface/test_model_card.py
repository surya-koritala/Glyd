"""One check of model_card.render: python3 huggingface/test_model_card.py"""
import json
import tempfile
from pathlib import Path

from model_card import eval_results, render

NUMBERS = {"size_gb": 5.9, "gpus": [{"name": "RTX 4090", "tok_s": 91.5, "first_token_ms": 48, "mem_gib_4k": 7.2, "fits_32k": True},
                                    {"name": "RTX 3060", "tok_s": 40.1, "first_token_ms": 120, "mem_gib_4k": 7.2, "fits_32k": False}],
           "kl_bf16": 0.0123, "kl_fp8_ref": 0.0456, "date": "2026-10-09", "versions": {"glyd": "0.30.0"}}
PLAN = {"class": "Qwen3_5ForCausalLM", "packs": 10, "copies": 5, "not_loaded": {}}


def card(d, **glyd):
    src = {"repo": "Qwen/Qwen3.5-9B", "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a"}
    (d / "glyd.json").write_text(json.dumps({"glyd": "0.30.0", "source": src, **glyd}))
    return d / "glyd.json"


LEVELS = {"penguin": {"size": 12 * 10**9, "kl": 0, "tok_s": None}, "kestrel": {"size": 7_341_796_951, "kl": 0.0147, "tok_s": 121.95}}   # (swift has no repo; kestrel has no numbers file: "–")
LEVELS["kestrel"].update(kl=None, tok_s=None)
FP8 = ("Qwen", 10_300_000_000)
BF16 = 9 * 10**9 * 2

with tempfile.TemporaryDirectory() as t:
    t = Path(t)
    for glyd, numbers in (({"format": "glyd-v6", "layout": "mma"}, None), ({"format": "glyd-v7", "layout": "i5", "level": "swift"}, NUMBERS)):
        m = json.loads(card(t, **glyd).read_text())
        text = render(m, 5_800_000_000 if numbers else 12 * 10**9, BF16, 9 * 10**9, PLAN, "apache-2.0", numbers, LEVELS, FP8)
        level = glyd.get("level", "penguin")
        assert f"glyd run Qwen/Qwen3.5-9B:{level}" in text and "install.sh" in text
        for bad in ("{", "TODO", "None", "Sigbound", "GiB"):
            assert bad not in text, bad
        assert "pipeline_tag: text-generation\n" in text and ("bit for bit" in text) == (level == "penguin")
        assert ("Not lossless" in text) == (level != "penguin") and ("from_pretrained" in text) == (level == "penguin")
        assert ("**5.8 GB, 68% smaller than bf16 (18.0 GB).**" in text) == (numbers is not None) and ("**12.0 GB, 33% smaller than bf16 (18.0 GB).**" in text) == (numbers is None)
        assert ("KL divergence from bf16 is 0.0123 (lower is closer)" in text) == (numbers is not None)
        rows = text.split("## Compared\n\n")[1].split("\n\n")[0].split("\n")
        assert rows[0] == "| | Size | Smaller than bf16 | KL vs bf16 | RTX 4090 |" and rows[1] == "| :-- | --: | --: | --: | --: |"
        assert rows[2] == "| bf16 (original) | 18.0 GB | – | 0 | – |"
        assert rows[3] == "| Qwen FP8 | 10.3 GB | 43% | " + ("0.0456" if numbers else "–") + " | – |"
        assert sum("**" in r for r in rows) == 1 and sum("this repo" in r for r in rows) == 1 and "[%s]" % level not in text.split("## Compared")[1]
        assert "| [kestrel](https://huggingface.co/glyd/Qwen3.5-9B-kestrel) | 7.3 GB | 59% | – | – |" in rows or level == "kestrel"
        assert not any("[swift]" in r for r in rows) or level == "swift"   # (no repo, no row)
        if numbers:
            assert "| **swift (this repo)** | **5.8 GB** | **68%** | **0.0123** | **92 tok/s** |" in rows   # (91.5 rounds to 92)
            assert "| [penguin](https://huggingface.co/glyd/Qwen3.5-9B-penguin) (lossless) | 12.0 GB | 33% | 0 | – |" in rows
        else:
            assert "| **penguin (lossless, this repo)** | **12.0 GB** | **33%** | **0** | **–** |" in rows
        assert "Size: the files you download. KL:" in text and "## Good to know" in text and "\"8-bit precision\" and \"Model size\" labels count packed bytes" in text
        assert "from a model with 9,000,000,000 parameters." in text and "(the Hub's \"Model size\" counts" not in text and "## Glyd levels" not in text
        assert text.endswith("## License\n\nWeights: Apache-2.0, from [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B). Glyd's engine, which runs them: open source, Apache-2.0 ([GitHub](https://github.com/surya-koritala/Glyd)).\n")
        assert ("## Speed and memory" in text) == (numbers is not None)
        if numbers:   # (short names, sorted by tokens/s, GiB to GB: 7.2 GiB = 7.7 GB)
            assert "| GPU | Tokens/s | First token | Memory at 4k | 32k context |" in text
            assert "| RTX 4090 | 92 | 48 ms | 7.7 GB | yes |\n| RTX 3060 | 40 | 120 ms | 7.7 GB | no |" in text
            assert "Measured 2026-10-09 with glyd 0.30.0: one GPU each, a 4k-token context; memory is the GPU memory in use." in text and "Fits a 32k context on: RTX 4090." in text
        print(text if level == "swift" else "")

H = {"inspect_ai": "0.3.278", "inspect_evals": "0.24.0"}
EV = [{"dataset": "openai/gsm8k", "task_id": "gsm8k", "value": 0.912, "n": 1319, "harness": H, "notes": "10-shot, non-thinking", "date": "2026-10-10"},
      {"dataset": "TIGER-Lab/MMLU-Pro", "task_id": "mmlu_pro", "value": 0.7, "n": 12032, "harness": H, "notes": "0-shot, non-thinking", "date": "2026-10-10"}]
swift = {"glyd": "0.30.0", "source": {"repo": "Qwen/Qwen3.5-9B", "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a"}, "format": "glyd-v7", "level": "swift"}
args = (swift, 5_800_000_000, BF16, 9 * 10**9, PLAN, "apache-2.0")
CAL = {"method": "gptq-act", "data": [{"dataset": "Salesforce/wikitext", "revision": "b08601e04326c79dfdd32d625aee71d232d685c3"}], "seqs": 512, "tokens": 2048}   # (glyd.json's "calibration", glyd-gpu pack --codes)
plain, calibrated = render(*args, NUMBERS), render(dict(swift, calibration=CAL), *args[1:], NUMBERS)
assert "quantized once from the original bf16 weights. Not lossless: KL divergence from bf16 is 0.0123" in plain and "calibration" not in plain
assert "quantized once from the original bf16 weights with calibration (GPTQ). Not lossless: KL divergence from bf16 is 0.0123" in calibrated and calibrated.index("calibration") < calibrated.index("## Compared")
assert "weights with calibration (some-new-method). Not lossless" in render(dict(swift, calibration=dict(CAL, method="some-new-method")), *args[1:], NUMBERS)   # (a method the card has no name for: its own)
only = render(*args, {**NUMBERS, "evals": EV})
for c in ("## Benchmarks", "| GSM8K | 91.2% |\n| MMLU-Pro | 70.0% |", "Inspect AI 0.3.278 (inspect_evals 0.24.0)", "12032 questions"):
    assert c in only, c
assert "| bf16 |" not in only
both = render(*args, {**NUMBERS, "evals": EV, "evals_bf16": [dict(EV[0], value=0.92)]})
assert "| Benchmark | this level | bf16 |" in both and "| GSM8K | 91.2% | 92.0% |" in both and "| MMLU-Pro | 70.0% | – |" in both
assert "Benchmarks" not in render(*args, NUMBERS) and "Benchmarks" not in render(*args, None)
y = eval_results({**NUMBERS, "evals": EV}, "swift")
assert y.startswith("- dataset:\n    id: openai/gsm8k\n    task_id: gsm8k\n  value: 0.912\n  date: \"2026-10-10\"\n") and "    id: TIGER-Lab/MMLU-Pro\n    task_id: mmlu_pro\n" in y
assert 'notes: "glyd serve, swift, inspect-ai 0.3.278, 0-shot, non-thinking"' in y and eval_results(NUMBERS, "swift") is None and eval_results({}, "swift") is None
t = render(*args, {**NUMBERS, "kl_fp8_ref": None, "kl_bf16": None}, LEVELS, FP8)   # (null: not measured, as numbers.sh writes it for a model without a reference)
assert "| Qwen FP8 | 10.3 GB | 43% | – | – |" in t and "KL divergence" not in t and "| **swift (this repo)** | **5.8 GB** | **68%** | **–** | **92 tok/s** |" in t
assert "FP8" not in render(*args, NUMBERS) and "| bf16 (original) |" in render(*args, None)   # (no FP8 repo, no FP8 row)
print("ok")
