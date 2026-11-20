# Fuzzing the parsers

libFuzzer through cargo-fuzz, for every parser that reads what a stranger sends: a model repository's files (config.json, tokenizer.json, the chat template, safetensors headers, GGUF files, glyd.json), and what a client sends the server (the request line and headers, the body of a chat request with its schemas, regular expressions and grammars, the tools and the messages).
This crate is outside the workspace (`exclude` in the root Cargo.toml), so a normal build or test does not build it.

| target | what it reads |
|---|---|
| `json` | `glyd_gpu::json`, the reader every other starts from (and the writers; what it writes it reads back) |
| `safetensors_header` | the engine's reader (`glyd_gpu::safetensors::File::open`) and the codec's (`glyd::safetensors`) |
| `checkpoint_index` | a sharded checkpoint: `model.safetensors.index.json`, the shards, config.json's `quantization_config` |
| `gguf` | `gguf::Gguf::parse` (header, metadata, tensor table) and what is made of it: the description, config.json, the tokenizer's files |
| `config` | config.json by every family's parser and by the planner (`plan::Shape`) |
| `machine` | a machine described in JSON, planned for |
| `tokenizer` | `Tok::load` of tokenizer.json (or tiktoken.model), tokenizer_config.json, then encode, decode and the token bytes grammars compile against |
| `pack_dir` | a packed directory: `glyd.json`, config.json, the shards, `engine.safetensors`; the manifests of the loader's plan |
| `http_head` | the request line and headers (`server::read_head`) |
| `chat_request` | a request body: its validation, sampling, `response_format` and `guided_*`, the Anthropic form turned to OpenAI's, tools and messages |
| `constraint` | a JSON schema, regular expression, Lark grammar or choice compiled by llguidance, and a few tokens run through it |
| `chat_template` | a chat template, and messages and tools rendered through it |
| `strftime` | the format of `strftime_now` |
| `tool_calls` | a reply's tool calls read back in each format, whole and streamed |

A target that reads several files takes them in one input, apart by a line holding a NUL (`fuzz/src/lib.rs`: `parts`), in the order its doc comment gives.

## Run one

```sh
python3 fuzz/seed.py gguf                       # the seeds: the repository's fixtures and a few made by the script (nothing of them is committed)
cd fuzz && cargo +nightly fuzz run gguf -- -max_total_time=600 -timeout=10 -rss_limit_mb=3072 -max_len=65536
```

CI pins the nightly in `.github/workflows/fuzz.yml`. On Linux that is all. On a Mac the engine does not link (`sched_getcpu` in cpu_cb.rs), so build with
`RUSTFLAGS="-C link-arg=-Wl,-undefined,dynamic_lookup"`. Pass `-fork=1 -ignore_crashes=1 -ignore_timeouts=1 -ignore_ooms=1` to keep going past a crash and collect the inputs under `fuzz/artifacts/<target>/`.
A crash is replayed with `cargo +nightly fuzz run <target> fuzz/artifacts/<target>/<file>`; fix it in the parser, with a test of the input in the parser's own module, never by narrowing a target.

## CI

`.github/workflows/fuzz.yml`: a minute for each target on a pull request that touches a parser or this crate, ten minutes each every night (the corpus cached between runs and cut down after each), and by hand (`gh workflow run fuzz.yml -f seconds=3600 -f targets="gguf json"`).
A nightly crash becomes an issue labelled `fuzz-crash`.

A new parser gets a target: a file in `fuzz_targets/`, a `[[bin]]` in Cargo.toml, seeds in `seed.py` and a row above.
