#!/usr/bin/env python3
"""Fills fuzz/corpus/<target>/ with seeds: files the repository already tracks (its fixtures, small real files) and a few made here. Nothing is committed but this script, so the tree stays source (scripts/check_tree.sh).

    python3 fuzz/seed.py            # every target
    python3 fuzz/seed.py gguf json  # some

A seed already in the corpus is left as it is (a run's finds stay)."""
import base64, hashlib, json, os, struct, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SEP = b"\n\0\n"
FIX = os.path.join(ROOT, "glyd-engine/tests/fixtures")
SMALL = 70_000  # no seed past this: a fuzzer does better with small ones


def put(target, name, data):
    if isinstance(data, str):
        data = data.encode()
    if len(data) > SMALL:
        return
    d = os.path.join(HERE, "corpus", target)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name if name else hashlib.sha1(data).hexdigest()[:16])
    if not os.path.exists(p):
        with open(p, "wb") as f:
            f.write(data)


def fixture(name):
    p = os.path.join(FIX, name)
    return open(p, "rb").read() if os.path.exists(p) else b""


def fixtures(suffix):
    return sorted(f for f in os.listdir(FIX) if f.endswith(suffix)) if os.path.isdir(FIX) else []


def tracked(rel):
    p = os.path.join(ROOT, rel)
    return open(p, "rb").read() if os.path.exists(p) else b""


# ---- JSON ---------------------------------------------------------------------------------------------------------------------------------------

def json_seeds():
    for f in fixtures(".json"):
        put("json", f, fixture(f))
    for i, t in enumerate([
        '{"a": [1, 2.5e-3, -0, true, false, null, "x\\u00e9\\ud83d\\ude00\\n"], "b": {"c": {}}}',
        "[" * 100 + "]" * 100,
        '{"__metadata__": {"format": "pt"}, "w": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]}}',
        "18446744073709551615", "-9223372036854775808", "1e999", '"\\ud800"', "{}", "[]",
    ]):
        put("json", f"inline{i}", t)


# ---- GGUF ---------------------------------------------------------------------------------------------------------------------------------------

def gstr(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def gguf_file(meta, tensors, data=b""):
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta))
    for k, (ty, v) in meta:
        out += gstr(k) + struct.pack("<I", ty)
        out += v
    for name, dims, ty, off in tensors:
        out += gstr(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", ty, off)
    out += b"\0" * (-len(out) % 32)
    return out + data


def gguf_seeds():
    small = tracked("tests/data/weights/small.gguf")
    put("gguf", "small.gguf", small)
    put("gguf", "small-header", small[:2048])
    put("gguf", "small-cut", small[:300])
    s = lambda x: struct.pack("<Q", len(x)) + x.encode()
    meta = [
        ("general.architecture", (8, s("llama"))),
        ("llama.block_count", (4, struct.pack("<I", 2))),
        ("llama.context_length", (4, struct.pack("<I", 2048))),
        ("llama.embedding_length", (4, struct.pack("<I", 64))),
        ("llama.attention.head_count", (4, struct.pack("<I", 4))),
        ("llama.attention.head_count_kv", (4, struct.pack("<I", 2))),
        ("llama.feed_forward_length", (4, struct.pack("<I", 128))),
        ("llama.rope.freq_base", (6, struct.pack("<f", 10000.0))),
        ("tokenizer.ggml.model", (8, s("gpt2"))),
        ("tokenizer.ggml.tokens", (9, struct.pack("<IQ", 8, 3) + s("a") + s("b") + s("<|end|>"))),
        ("tokenizer.ggml.token_type", (9, struct.pack("<IQ", 5, 3) + struct.pack("<iii", 1, 1, 3))),
        ("tokenizer.ggml.merges", (9, struct.pack("<IQ", 8, 1) + s("a b"))),
        ("tokenizer.ggml.eos_token_id", (4, struct.pack("<I", 2))),
        ("tokenizer.chat_template", (8, s("{{ messages[0].content }}"))),
    ]
    put("gguf", "synthetic", gguf_file(meta, [("token_embd.weight", [64, 3], 0, 0), ("blk.0.attn_q.weight", [64, 64], 1, 768)], b"\0" * 9000))


# ---- safetensors --------------------------------------------------------------------------------------------------------------------------------

def st_file(header, data):
    h = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack("<Q", len(h)) + h + data


def st_seeds():
    put("safetensors_header", "small.safetensors", tracked("tests/data/weights/small.safetensors"))
    put("safetensors_header", "one", st_file({"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}, b"\0" * 8))
    put("safetensors_header", "two", st_file({"__metadata__": {"format": "pt"}, "w": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]}, "b": {"dtype": "F8_E4M3", "shape": [4], "data_offsets": [8, 12]}}, b"\0" * 12))
    put("safetensors_header", "empty", st_file({}, b""))
    one = st_file({"model.embed.weight": {"dtype": "BF16", "shape": [4, 2], "data_offsets": [0, 16]}}, b"\0" * 16)
    idx = json.dumps({"metadata": {"total_size": 16}, "weight_map": {"model.embed.weight": "model-00001-of-00002.safetensors", "lm_head.weight": "model-00002-of-00002.safetensors"}})
    two = st_file({"lm_head.weight": {"dtype": "BF16", "shape": [4, 2], "data_offsets": [0, 16]}}, b"\0" * 16)
    cfg = json.dumps({"quantization_config": {"weight_block_size": [128, 128]}})
    put("checkpoint_index", "pair", idx.encode() + SEP + cfg.encode() + SEP + one + SEP + two)


# ---- configs ------------------------------------------------------------------------------------------------------------------------------------

QWEN3 = {"architectures": ["Qwen3ForCausalLM"], "hidden_size": 1024, "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 3072,
         "vocab_size": 151936, "rms_norm_eps": 1e-6, "rope_theta": 1000000, "max_position_embeddings": 40960, "tie_word_embeddings": True, "eos_token_id": 151645}


def config_seeds():
    put("config", "qwen3", json.dumps(QWEN3))
    put("config", "qwen3-gen", json.dumps(QWEN3).encode() + SEP + json.dumps({"eos_token_id": [151645, 151643], "temperature": 0.6}).encode())
    gens = {f[: -len("_generation_config.json")]: f for f in fixtures("_generation_config.json")}
    for f in fixtures("config.json"):
        base = f[: -len("config.json")].rstrip("_")
        g = gens.get(base)
        put("config", f, fixture(f) + (SEP + fixture(g) if g else b""))
    qc = fixture("qwen35_configs.json")
    if qc:
        try:
            for k, v in list(json.loads(qc).items())[:12]:
                put("config", "qwen35-" + str(k)[:20].replace("/", "_"), json.dumps(v))
        except Exception:
            pass
    put("machine", "spark", json.dumps({"name": "box", "ram_free_gib": 100, "ram_bandwidth": 200, "cores": 16, "gpus": [{"name": "NVIDIA GeForce RTX 4090", "vendor": "nvidia", "cc": [8, 9], "total_gib": 24, "free_gib": 22, "bandwidth": 1008, "driver": 580, "unified": False}]}))
    put("machine", "empty", '{"gpus": []}')


# ---- tokenizer ----------------------------------------------------------------------------------------------------------------------------------

def bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def tokenizer_seeds():
    u = bytes_to_unicode()
    vocab = {u[b]: b for b in range(256)}
    merges = []
    for a, b in [("h", "e"), ("l", "l"), ("he", "ll"), ("hell", "o"), (u[32], "w"), ("o", "r")]:
        vocab[a + b] = len(vocab)
        merges.append(f"{a} {b}")
    added = [{"id": len(vocab) + i, "content": c, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True} for i, c in enumerate(["<|im_start|>", "<|im_end|>", "<|endoftext|>"])]
    tj = {"version": "1.0", "truncation": None, "padding": None, "added_tokens": added, "normalizer": None,
          "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
          "post_processor": None, "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
          "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": "", "end_of_word_suffix": "", "fuse_unk": False, "byte_fallback": False, "ignore_merges": False, "vocab": vocab, "merges": merges}}
    tc = {"eos_token": "<|im_end|>", "bos_token": None, "chat_template": "{{ messages[0].content }}", "added_tokens_decoder": {str(a["id"]): {"content": a["content"], "special": True} for a in added}}
    put("tokenizer", "bpe", json.dumps(tj).encode() + SEP + json.dumps(tc).encode())
    put("tokenizer", "bpe-gen", json.dumps(tj).encode() + SEP + b"{}" + SEP + json.dumps({"eos_token_id": [256, 257]}).encode())
    wl = {"version": "1.0", "added_tokens": [], "normalizer": {"type": "Lowercase"}, "pre_tokenizer": {"type": "Whitespace"}, "post_processor": None, "decoder": None,
          "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "hello": 1, "world": 2}, "unk_token": "[UNK]"}}
    put("tokenizer", "wordlevel", json.dumps(wl))
    sp = {"version": "1.0", "added_tokens": [], "normalizer": None, "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": True},
          "post_processor": None, "decoder": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": True},
          "model": {"type": "Unigram", "unk_id": 0, "vocab": [["<unk>", 0.0], ["▁hello", -1.0], ["▁world", -2.0], ["<0x41>", -3.0]], "byte_fallback": True}}
    put("tokenizer", "unigram", json.dumps(sp))
    tk = "".join(f"{base64.b64encode(bytes([b])).decode()} {b}\n" for b in range(256)) + f"{base64.b64encode(b'he').decode()} 256\n"
    put("tokenizer", "tiktoken", SEP * 3 + tk.encode())
    put("tokenizer", "tiktoken-cfg", SEP + json.dumps({"eos_token": "<|im_end|>", "added_tokens_decoder": {"257": {"content": "<|im_end|>", "special": True}}}).encode() + SEP + SEP + tk.encode())


# ---- requests -----------------------------------------------------------------------------------------------------------------------------------

TOOL = {"type": "function", "function": {"name": "get_weather", "description": "Weather of a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer", "minimum": 1}, "unit": {"enum": ["c", "f"]}}, "required": ["city"]}}}
SCHEMA = {"type": "object", "properties": {"name": {"type": "string", "maxLength": 20}, "age": {"type": "integer"}, "tags": {"type": "array", "items": {"type": "string"}}, "kind": {"anyOf": [{"const": "a"}, {"const": "b"}]}}, "required": ["name"], "additionalProperties": False}
REQUESTS = [
    {"model": "m", "messages": [{"role": "user", "content": "Hello"}]},
    {"model": "m", "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}], "temperature": 0.7, "top_p": 0.9, "max_tokens": 64, "stream": True, "stream_options": {"include_usage": True}, "stop": ["\n\n", "END"], "seed": 7, "n": 2},
    {"model": "m", "messages": [{"role": "user", "content": [{"type": "text", "text": "What is this?"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}]}], "logprobs": True, "top_logprobs": 3},
    {"model": "m", "messages": [{"role": "user", "content": "Weather in Paris?"}], "tools": [TOOL], "tool_choice": "auto"},
    {"model": "m", "messages": [{"role": "user", "content": "Weather in Paris?"}], "tools": [TOOL], "tool_choice": {"type": "function", "function": {"name": "get_weather"}}},
    {"model": "m", "messages": [{"role": "user", "content": "Go"}, {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]}, {"role": "tool", "tool_call_id": "call_1", "content": "12C"}], "tools": [TOOL]},
    {"model": "m", "messages": [{"role": "user", "content": "Describe a person"}], "response_format": {"type": "json_schema", "json_schema": {"name": "person", "strict": True, "schema": SCHEMA}}},
    {"model": "m", "messages": [{"role": "user", "content": "JSON please"}], "response_format": {"type": "json_object"}},
    {"model": "m", "messages": [{"role": "user", "content": "A date"}], "guided_regex": "[0-9]{4}-[0-9]{2}-[0-9]{2}"},
    {"model": "m", "messages": [{"role": "user", "content": "Pick"}], "guided_choice": ["yes", "no", "maybe"]},
    {"model": "m", "messages": [{"role": "user", "content": "Grammar"}], "guided_grammar": "start: \"a\"+ (\"b\" | \"c\")"},
    {"model": "m", "messages": [{"role": "user", "content": "Go"}], "structured_outputs": {"json": SCHEMA}},
    {"model": "m", "messages": [{"role": "user", "content": "Go"}], "structured_outputs": {"regex": "(ab)+c?"}},
    {"model": "m", "messages": [{"role": "user", "content": "Go"}], "guided_json": json.dumps(SCHEMA)},
    {"model": "m", "prompt": "Once upon a time", "max_tokens": 16, "logit_bias": {"50256": -100, "11": 5}, "allowed_token_ids": [1, 2, 3], "suffix": " the end", "echo": True, "best_of": 1},
    {"model": "m", "messages": [{"role": "user", "content": "Think"}], "chat_template_kwargs": {"enable_thinking": False, "x": [1, {"y": None}]}, "reasoning_effort": "low", "documents": [{"title": "t", "text": "x"}], "min_tokens": 2, "top_k": 40, "min_p": 0.05, "repetition_penalty": 1.1},
    # Anthropic Messages
    # OpenAI Responses
    {"model": "m", "instructions": "Be brief.", "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}], "max_output_tokens": 50, "temperature": 0.3, "text": {"format": {"type": "json_schema", "name": "p", "schema": SCHEMA, "strict": True}}},
    {"model": "m", "input": "Weather in Paris?", "tools": [{"type": "function", "name": "get_weather", "description": "w", "parameters": TOOL["function"]["parameters"]}], "tool_choice": "auto", "reasoning": {"effort": "low"}},
    {"model": "m", "input": [{"type": "message", "role": "user", "content": "Go"}, {"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}, {"type": "function_call_output", "call_id": "c1", "output": "12C"}]},
    # Ollama
    {"model": "m", "messages": [{"role": "user", "content": "Hi", "images": []}], "stream": False, "options": {"temperature": 0.5, "num_predict": 50, "num_ctx": 4096, "stop": ["x"], "seed": 3}, "format": "json", "think": False},
    {"model": "m", "messages": [{"role": "user", "content": "Go"}], "tools": [TOOL], "format": SCHEMA, "keep_alive": "5m"},
    {"model": "m", "prompt": "Why is the sky blue?", "system": "Be brief.", "stream": True, "raw": False, "options": {"top_k": 20, "repeat_penalty": 1.1}},
    {"model": "m", "max_tokens": 100, "system": [{"type": "text", "text": "Be brief."}], "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}], "stop_sequences": ["x"], "temperature": 0.5},
    {"model": "m", "max_tokens": 100, "messages": [{"role": "user", "content": "Weather?"}, {"role": "assistant", "content": [{"type": "text", "text": "Checking."}, {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "Paris"}}]}, {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": [{"type": "text", "text": "12C"}]}]}],
     "tools": [{"name": "get_weather", "description": "w", "input_schema": TOOL["function"]["parameters"]}], "tool_choice": {"type": "tool", "name": "get_weather"}, "thinking": {"type": "enabled", "budget_tokens": 1024}},
]


def request_seeds():
    for i, r in enumerate(REQUESTS):
        put("chat_request", f"req{i:02}", json.dumps(r))
    for i, (k, t) in enumerate([(0, json.dumps(SCHEMA)), (0, '{"type": "array", "items": {"type": "integer"}, "minItems": 1}'), (0, '{"$ref": "#/$defs/n", "$defs": {"n": {"type": "string", "pattern": "^a+$"}}}'),
                                (1, "[0-9]{4}-[0-9]{2}"), (1, "(a|b)*c\\d+\\s?[^x]"), (1, "\\p{L}+(\\.\\w+){0,3}"), (2, "start: \"a\"+ (\"b\" | \"c\")"), (2, "start: item+\nitem: /[a-z]+/ \",\"?\n%ignore \" \""),
                                (3, "yes\nno\nmaybe"), (4, "")]):
        put("constraint", f"c{i:02}", bytes([k]) + t.encode())


def http_seeds():
    put("http_head", "post", "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost:8000\r\nContent-Type: application/json\r\nAuthorization: Bearer sk-test\r\nContent-Length: 42\r\nExpect: 100-continue\r\n\r\n")
    put("http_head", "get", "GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1:8000\r\nAccept: */*\r\n\r\n")
    put("http_head", "options", "OPTIONS /v1/chat/completions HTTP/1.1\r\nHost: localhost:8000\r\nOrigin: http://localhost:5173\r\nAccess-Control-Request-Method: POST\r\nAccess-Control-Request-Headers: content-type,authorization\r\n\r\n")
    put("http_head", "bare", "GET / HTTP/1.0\n\n")


# ---- chat templates and tool calls --------------------------------------------------------------------------------------------------------------

MESSAGES = json.dumps([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Weather in Paris?"}, {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]}, {"role": "tool", "tool_call_id": "c1", "content": "12C"}, {"role": "user", "content": "Thanks"}])


def template_seeds():
    tools = json.dumps([TOOL])
    for f in fixtures(".jinja"):
        put("chat_template", f, fixture(f) + SEP + MESSAGES.encode() + SEP + tools.encode())
    q = fixture("chat_qwen3.json")
    if q:
        try:
            put("chat_template", "qwen3", json.loads(q)["template"].encode() + SEP + MESSAGES.encode() + SEP + tools.encode())
        except Exception:
            pass
    for i, t in enumerate(["{% for m in messages %}{{ m.role }}: {{ m.content | trim }}\n{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}",
                           "{{ strftime_now('%Y-%m-%d') }}{{ tools | tojson(indent=2) }}{{ raise_exception('no') if messages | length > 9 }}",
                           "{% set ns = namespace(n=0) %}{% for m in messages %}{% set ns.n = ns.n + loop.index %}{% endfor %}{{ ns.n }}{{ range(5) | list }}{{ 'a b'.split(' ') | join(',') }}"]):
        put("chat_template", f"t{i}", t.encode() + SEP + MESSAGES.encode() + SEP + tools.encode())
    for i, f in enumerate(["%Y-%m-%d %H:%M:%S", "%A %B %d %j %U %p %y %b %a %I %%", "%e %-d %Z %z %c %x %X"]):
        put("strftime", f"s{i}", struct.pack("<qi", 1_700_000_000 + i * 86400, 3600 * i) + f.encode())


def tool_seeds():
    tools = json.dumps([TOOL]).encode()
    replies = [
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris", "days": 2}}\n</tool_call>',
        'Let me check.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>',
        '<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value><arg_key>days</arg_key><arg_value>2</arg_value></tool_call>',
        '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>',
        '<tool_call>{"name": "get_weather", "arguments": {"city": "Par',
    ]
    for i, r in enumerate(replies):
        put("tool_calls", f"r{i}", r.encode() + SEP + tools)


def st_tiny(names, metadata=None):
    """A safetensors file of U8 tensors of four bytes each, back to back."""
    header = {} if metadata is None else {"__metadata__": metadata}
    for i, n in enumerate(names):
        header[n] = {"dtype": "U8", "shape": [4], "data_offsets": [4 * i, 4 * i + 4]}
    return st_file(header, b"\1\2\3\4" * len(names))


def pack_seeds():
    """A packed Qwen3 of two layers as the loader plans it: config.json, glyd.json, the shard and the embedding's sidecar."""
    cfg = dict(QWEN3, hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16, intermediate_size=128, vocab_size=32)
    tier0 = 2 << 24 | 0x030201
    shapes = {"self_attn.q_proj": [128, 64], "self_attn.o_proj": [64, 64], "mlp.gate_proj": [256, 64], "mlp.down_proj": [64, 128]}
    for tie, tag in ((True, "tied"), (False, "lm_head")):
        cfg = dict(cfg, tie_word_embeddings=tie)
        packs, names = {}, []
        for l in range(2):
            for k, shape in shapes.items():
                key = f"model.layers.{l}.{k}"
                packs[key] = {"layout": "mma", "shape": shape, "tiers": [tier0, 0, 0]}
                names += [f"{key}.glyd_{x}" for x in ("data", "blocks", "block_base")]
            names += [f"model.layers.{l}.{x}.weight" for x in ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm")]
        if not tie:
            packs["lm_head"] = {"layout": "mma", "shape": [32, 64], "tiers": [tier0, 0, 0]}
            names += [f"lm_head.glyd_{x}" for x in ("data", "blocks", "block_base")]
        names.append("model.norm.weight")
        manifest = {"format": "glyd-v7", "layout": "mma", "packs": packs}
        embed = {"layout": "mma", "shape": [32, 64], "tiers": [tier0, 0, 0]}
        side = st_tiny([f"embed.glyd_{x}" for x in ("data", "blocks", "block_base")], {"embed": json.dumps(embed)})
        put("pack_dir", "qwen3-" + tag, json.dumps(cfg).encode() + SEP + json.dumps(manifest).encode() + SEP + st_tiny(names) + SEP + side)
    put("pack_dir", "twelve", json.dumps(cfg).encode() + SEP + json.dumps({"format": "glyd-v7", "layout": "mma12", "packs": {}}).encode() + SEP + st_tiny(["a"]) + SEP + st_tiny(["b"], {"embed": json.dumps({"layout": "mma12", "shape": [32, 64], "hb": 120})}))
    put("pack_dir", "cb", b"{}" + SEP + json.dumps({"kind": "glyd-g/1", "layers": 2}).encode())


def main():
    want = set(sys.argv[1:])
    for name, fn in [("json", json_seeds), ("gguf", gguf_seeds), ("safetensors_header", st_seeds), ("config", config_seeds), ("tokenizer", tokenizer_seeds),
                     ("chat_request", request_seeds), ("http_head", http_seeds), ("chat_template", template_seeds), ("tool_calls", tool_seeds), ("pack_dir", pack_seeds)]:
        if not want or name in want or (name == "chat_request" and "constraint" in want) or (name == "safetensors_header" and "checkpoint_index" in want) or (name == "config" and "machine" in want) or (name == "chat_template" and "strftime" in want):
            fn()
    d = os.path.join(HERE, "corpus")
    for t in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        print(f"{t}: {len(os.listdir(os.path.join(d, t)))} seeds")


if __name__ == "__main__":
    main()
