# Changelog

All notable user-facing changes. Measurement history, floors and refuted
ideas live in [research/codec/CHANGELOG-BENCH.md](research/codec/CHANGELOG-BENCH.md).
Versioning follows [SemVer](https://semver.org); the on-disk format has its
own version in every block header (v6, v7) and every release decodes
every earlier format.

## v0.35.0 — 2026-11-27

- The `glyd-gpu` wheel no longer holds the engine's program. Since the engine carries the GPU library's kernels as well, a wheel with both the libraries and the engine was 124 MB, and PyPI takes at most 100 MB. The installer (`curl -LsSf https://getglyd.com/install.sh | sh`) puts `glyd-engine` on the PATH, where the Python package's `glyd run` and `glyd serve` find it; with none there they take the vLLM path, as before. `GLYD_ENGINE=PATH` still names another engine.

- A Nemotron-H config that lists the layers of its multi-token-prediction head (`mtp_layers_block_type`, as Nemotron 3.5 Lightning's does) is read: `fit` plans it and `serve` loads it, the head's `mtp.*` tensors left unread as before. The head's draft layers do not change the model's own logits. Before, `fit` exited 3 and `serve` stopped at once with "config key mtp_layers_block_type is not understood by the description builder", a model v0.34.2 served.
- On Windows a model in a local directory is named by the directory's name in the engine's words (a refusal names `glm`, not the whole path), as on Linux and macOS; `bench --report` names its files with `/` on every system.
- Muse Spark's residual gates compute as transformers' do: the gate's two factors in fp32 (they were rounded to bf16) and the gated add rounded to bf16 once (it was rounded three times), on the GPU and in the CPU reference; a gate stored in fp32 is read (it was refused).
- Muse Spark's config reads as transformers reads it, in the engine and in its CPU reference alike: an absent `final_logit_softcapping` is 20 (it meant no soft-cap, which also dropped the output multiplier), an absent `rms_norm_eps` 1e-5 (it was 1e-6), and the other numbers transformers' defaults; a null value, or a soft-cap or gate temperature of 0, is refused by name. A Muse Spark checkpoint with a `quantization_config` (such as ModelOpt's NVFP4) is refused in words that name its format, not half read. Its release's audio and vision towers are named as the parts the text model does not read.
- Kimi-K3's depth mix scores its sources with the config's `rms_norm_eps`, as vLLM does (it was 1e-5 whatever the config said; Kimi-K3 and K3.1 give 1e-5, so their output is unchanged).
- A GLM latent attention's indexer computes its head weights in fp32 on the GPU and scores with them unrounded, as transformers keeps GLM-5.3's `weights_proj` in float32 and as the CPU reference of the layer kind does; they were rounded to bf16, which took near ties of the pools' scores the other way. Every family whose latent attention has an indexer of its own (GLM-5.3, GLM-5.3-Flash, Kimi-K3's sparse layers) takes it; the unweighted indexer and DeepSeek-V4's are unchanged. `kindcheck` holds each such indexer's head weights and picks to the reference's fp32 weights, and `logitsmodel glm-dsa DIR --ties` makes a random GLM-5.3 whose picks the head weights decide. Part of #121.
- A GLM config that asks for a form the engine does not compute is refused, the key and the value the engine runs named: `rope_interleave` or `indexer_rope_interleave` false where the model has a rotary, `index_kpool_compress` or `index_kpool_always_select_tail` false where it has pools. They were read and the interleaved, gated form computed whatever they said. Part of #121.
- Keys a GLM or DeepSeek-V4 config leaves out take transformers' defaults: GLM-5.3-Flash's pools of 16, the forget gate's lower bound of -5 by transformers' rule (no bound, the softplus decay, only where the config says null), the first three layers dense where neither `mlp_layer_types` nor `first_k_dense_replace` is given; DeepSeek-V4's routed weights times 1.5, its SwiGLU clamp at 10 and its first three layers routed by the token. Every release sets these keys: no released model's description changes. Part of #121.
- A DeepSeek-V4 layer that compresses by 1 is refused in words (it was described as DeepSeek-V4.1's compressor of a token), and a DeepSeek-V4.1 sharing layout the engine does not describe is refused when the model is described, naming the layer and its source, where an index source that keeps no entries took its keys from a kv source that keeps none and failed only when run. Part of #121.
- The Qwen3.5 layout (dense or with experts) and Qwen3.8-Flash-Next read a config the way transformers does, or refuse it in words. A config that gives no `partial_rotary_factor` rotates a quarter of the head (the layout with experts rotated all of it), and a top-level `partial_rotary_factor` or `rope_theta` that `rope_parameters` lacks is read (10,000 where none is given). A config that gives no `head_dim` has heads of 256 (the dense reader took the hidden size over the heads), one that gives no `layer_types` has a full attention at every `full_attention_interval`-th layer, and one with experts that gives no `shared_expert_intermediate_size`, `moe_intermediate_size`, `num_experts_per_tok` or `num_experts` has 512, 512, 8 and 256 (a missing shared-expert width meant no shared expert). A rope that scales its frequencies (yarn, linear and the rest) and a linear-attention output gate that is not the SiLU are refused with experts too, in the dense reader's words ("rope of type yarn, which the engine does not run"); the layout with experts ran the plain rotary and the SiLU gate whatever the config said. No released config's description or plan changes. Part of #121.
- Starlings' download no longer holds a GPU library: a CUDA build of the engine loads none, and the engine bundles of the release hold none, so `scripts/package-beta.sh` takes `glyd-engine` and `LICENSE` from an unpacked engine bundle alone (it names what is missing, and what starlings must be) and `install.sh` installs starlings and glyd-engine. `starlings up` still passes a library named by `--gpu-lib`, or found beside it, to the engine as `GLYD_GPU_LIB`. The site's workflow now builds on a change to any file its generators read (`plan.rs`, `runtime/config.rs`, `runtime/describe.rs` were missing, and `glyd-engine/Cargo.toml` from the pull-request list). Part of #121.
- FP8 matrices on an A100, H100 or B200 (layout H): a pack whose patches are plain (the default there) now writes every byte of its core. A plain patch keeps its fourth row block in var, and the pack left the last 1,024 bytes of each patch's 7,168 unwritten; the products for 2 to 64 rows copy a row block's bytes whole, so they read those bytes (never decoded: no product's sum used them, and every logit is the same bits as before), and the expert cache's host copy of a layer's experts held whatever device memory had been there. The pack writes zeros there now, once when the model loads; the products are unchanged. Part of #121.
- On a machine with several GPUs, the engine makes its own GPU current on the thread it works on at every reply and every scheduler round, and a look at any GPU (its free memory, the report after a load, `/v1/glyd/info`, the costs object) leaves the thread's GPU as it was: a model on a GPU other than the first no longer relies on nothing having looked at another GPU on its thread since the load. Part of #121.
- `/v1/glyd/info`'s `costs.device` now names the GPU the model is served on, not the machine's first.
- `usercheck --ctx plan` loads the model on the GPU its plan names and holds that GPU's memory to the plan. It loaded it on the first GPU and read the first GPU's memory, so on two GPUs a plan of the second read 0 MiB taken and then failed with "invalid resource handle".
- `fit`, `run` and `serve` plan with the same free memory of the GPU. The first command on a GPU measures its memory's and link's rates, and that left the engine's code and a stream in the GPU's context: `fit` read the free memory before measuring, `run` and `serve` after, so their plans said 39.1 and 39.0 GiB free of the same A100 (28 MiB apart on an RTX A6000 even with the code unloaded: what a context keeps once a kernel ran in it). The measurement now runs in a context of its own, destroyed after it.
- `qwenserve` with no `--gpus` plans on every GPU of the machine and takes the one with the most room for the model whole, the GPU `fit` and `serve` plan the pack on. It took GPU 0 alone, so on two equal cards `fit` planned the codes pack on GPU 1 and `qwenserve` on GPU 0. `--gpus` and `--split` are as before (`--split` with no GPUs named is GPU 0's, refused for a cut).
- The guide's models page builds again (`scripts/build_site.sh`, the Pages workflow): `scripts/gen_models.py` had stopped on the levels table's quality bar and the reworked family table, and read neither field order nor line breaks. It now reads `levels.rs`, `support.rs` and `machine.rs` as Rust literals, by field name, and says what it could not read. The page's words follow the tables: the plain dense kind is told by model type (`llama`, `mistral`, `qwen2`, `qwen3`), the context limit is the families' own (`max_ctx`), the driver line names the CUDA 12 kernels' drivers (535, 570 for Blackwell), the example refusal is read from the GLM-5.3-Flash row, and the claim that Windows and macOS are not built or tested is gone. `scripts/gen_reference.py` takes an output folder. `scripts/check_tree.sh` (the Tree size job, every pull request) now runs both generators into a temporary folder, so a refactor they cannot read fails there and not only on the Pages build. Part of #121.
- `glyd-engine pack` and `glyd ship` save a model only at a level its family runs, and refuse any other level with the same words `glyd run` gives. GLM-5.3-Flash runs at penguin and hummingbird, so its kestrel, linnet, wren and three-bit saves are now refused: nothing loaded them, and their kv_b was in the earlier engine's form. Kimi-K3, GLM-5.3, DeepSeek-V4, Muse Spark and Qwen3.8-Flash-Next were already refused at the lossy levels, but by the packer's list of architectures; they now get the same words as `glyd run`. Part of #239.
- Internal: the lossy levels of Gemma 4, the Qwen3.5 layout and the Nemotron-H layout are read by one reader of the model's description, not three readers of their typed configs. It reads the weights by kind at the level, the sequence's rings and states, and the experts and their cache. Each family's row in the support table gives what the description does not: its prompt chunk, its page, its draft model, the layouts its loader packs in, and its config's own checks. The planner and the command line name no family outside a short list that a test holds, each line with its reason.
- Gemma 4's planned weights at its lossy levels count the q and k norms of its sliding layers at their own head width, 40 KB less for gemma-4-12B than before.
- The Nemotron-H layout's description now records its tensors' shapes, as the other families' do.
- `glyd run`, `serve` and `preflight` take the default level of a model from its config's row, never from its name. A Gemma 4 name whose config cannot be read now gets the refusal that says why (the Hub's, for a gated repo), before anything is downloaded.
- `glyd preflight` of a model not yet downloaded, larger than its GPUs, whose row keeps its experts in the computer's memory: the experts' share comes from its config's shape for every such family. Before, a fixed 93% applied to Kimi-K3 alone.
- Part of #239.
- The engine's own recorded bits (the pins of `logitscheck`, `glyd-engine/tests/logits/CASE.KEY[.LEVEL].json.gz`, and the Hummingbird digests of `packcheck --engine`, `tests/packs/nemotron-h-hb.KEY.json.gz`) are keyed by what makes those bits on a GPU: its compute capability, the words of its name the kernel choices read, and its SM count, which sizes the products' blocks and K splits (`sm89-geforce-80sm` for an RTX 4080 SUPER, `sm89-l4-58sm` for an L4, `sm86-a10-72sm` for an A10). Before, they were keyed by the compute capability alone, so an L4 was held to the bits of the RTX 4080 SUPER, whose GeForce-only stream-K kernels it does not run, and failed. The existing pins keep their recorded rows under the new names; the L4's are recorded. A GPU with no pin of its key says so and how to record one; a case it leaves with nothing compared still fails. `logitsrecord --out DIR` names a lossy level's pin `CASE.KEY.LEVEL.json.gz`. Part of #121.
- Internal: `glyd-engine pack` makes every family's save plan from the model's description: each block's matrices by its layer kind, the tensors it keeps by their roles, and the few ways a family's published saves differ (Gemma 4's layer by layer, GLM-5.3-Flash's F32 tensors kept, Nemotron-H's six-bit matrices beside four-bit experts at wren) read from the family's row of the table of families. Every save it wrote before is the same bytes, at every level. A Gemma 4 with layers that share keys and values, refused before, now packs. Part of #136.
- Condor's server (`qwenserve`, `tools/serve.sh`) serves its three modes as v0.34.2 did. `--hi` puts the attention layers' own matrices and the LM head at its layout again (five bits beside the four of fast mode when it is not given, six in precise and best mode); since the old engine went it was accepted and ignored, so fast mode ran them at four bits. `--no-side` (best mode) serves the pack's own experts again, and `--side A,B` names sidecars; before, best mode took a sidecar where its pack had one. The plan counts the second layout's bytes and opens the sidecars the server opens. The server prints its mode, sidecars and layouts after its plan.
- The chat page's Monitor shows the experts kept on the GPU again (`/v1/glyd/status`'s `places`: the plan's places a layer and their GB), as v0.34.2's did; since the old engine went it showed none.
- `qwenserve` refuses any flag it does not read, in words. v0.34.2's settings of its own engine (`--draft-min-p`, `--skip-rank`, `--cache-gb` and the rest) say what holds their place now. Before, they were ignored. Part of #140.
- The AMD development build (GLYD_HIP=1) passes clippy again: the launch configuration of NVIDIA's cluster launches is compiled only where they are (three unused items failed `-D warnings` under `cfg(glyd_hip)`).
- `glyd-engine usercheck --ref` (development builds) now continues each of its prompts with the CPU `Ref` backend's own tokens, given to the scheduler in the place of the engine's (teacher forcing, as `logitscheck`'s fixed tokens), so every row the GPU serves at many users is held to `Ref`'s row at the same inputs. `--floor` measures the test model's own noise floor beside each sequence: every prompt served again alone (the GPU against itself, its sums in another order), judged by the same rule at a tenth of the bound. `Ref` holds a checkpoint's bf16 weights as stored (half the bytes, the same bits; a tied head once), and its exp, log, tanh, pow, sin and cos are the `libm` crate's, so it makes the same bits on every computer. `usercheck --ref --record FILE` writes the reference's rows of a model on any computer with no GPU, and `usercheck --ref-from FILE` holds a GPU run to them; a file of other weights, level, prompts or reference version fails in words. The M2 gate's `--ref-dir DIR` takes such files for models the gate machine's memory does not hold. `kindcheck --sequence --tokens` and `floortrace --sequence --chunk` take one served sequence's tokens. The random test models of the dense Qwen3.5 layout and Gemma 4 are drawn with smaller heads, which brings their noise floors within a tenth of the bound; their fixtures are recorded again. Part of #121.
- Internal: `run`, `serve`, `fit` and `pack` parse a model's config.json once. The planner's parse is now the loaders': the dynamic route's `open_plan` and every stage loader (DeepSeek-V4, Nemotron-H, GLM-5.3, Kimi-K3, Muse Spark, Qwen3.8-Flash-Next, the Qwen3.5 layout), the lossy levels' loaders, the family readers, the description, the images and the tokenizer's end tokens take it where each read and parsed the file again (a `run` parsed it 4 to 8 times). The packer reads a GLM-5.3-Flash release's config once too, where its keys do not parse. No plan, refusal or load changes. A test counts the parses of every family's random model on each path. Part of #136.
- The lossy levels (kestrel, linnet, wren) of the Qwen3.5 layout, dense or with experts, now cut a model's layers into stages over the GPUs of one machine where one GPU does not hold it, by the same rule penguin's dynamic route cuts with: each layer's bytes at the level with its share of one sequence, each GPU's room its free memory less the reserve and the engine's work, a stage only where one may begin. Every expert stays on the GPUs where they hold them all; otherwise the experts go to the computer's memory and each stage gets its own places of them, by the one expert-cache rule. `fit`, `run` and `serve` print and load the same plan. Stages hold one sequence at a time, as penguin's do. `--split CUTS --devices GPUS` cuts where it says. A cut that is not inside the layers, a GPU that is not there, or a GPU whose stages take more than its room is refused in words. Before, these levels planned one GPU and refused the rest.
- Qwen3.8-2.4T-A95B on eight H100s now plans at kestrel, linnet and wren: its layers in stages over the eight GPUs, its experts in the computer's memory, places of them on each GPU. Before, each of these levels was refused for needing 88.6 to 121.5 GiB of one GPU. Penguin is still refused: the experts, held lossless, take 3,444.5 GiB and the machine has 1,800 GiB. Qwen3.8-27B at kestrel on two RTX 3090s now plans in two stages, where before one GPU refused it.
- Each stage loads its layers' packs on its own GPU through the Qwen3.5 layout's stage loader, the one penguin's split uses, and the residual is handed over between them. A plan's layout is the one its stages load: a lossy plan's block layout, or penguin's 12-bit or tiered layout.
- On a machine with several GPUs, a model planned on one of them no longer fails at its first reply with "invalid resource handle". After a load, the report of what it took looks at every GPU, and that switched the loading thread's GPU context to another GPU. The thread's context is now kept as it was.
- `tiercheck --split CUTS --devices GPUS` checks the expert cache in stages, each stage with its own cache, against every expert on one GPU. `splitcheck --dir DIR` checks a model made before. `logitsmodel qwen35moe DIR --real --widths FILE` takes another release's widths (its config.json or its recorded header), such as Qwen3.8-2.4T-A95B's, in place of Qwen3.5-35B-A3B's. Part of #333.
- Condor's server (`qwenserve`, `tools/serve.sh`) loads Qwen3.8-Flash-Next by its plan, the same rule as `glyd run` and `glyd serve` of a codes pack: the users' keys, values and states at the context first, then the engine's work, then the expert cache's places. It prints the plan before it loads (`Plan: ...`) and what the load took against it. A context or a number of users the GPU does not hold is refused in the planner's words, with the most users that fit. `--gpus 0,1` cuts the layers by the plan, `--split` and `--places` are held or refused, and a GPU named twice no longer needs `--places`. Before, the server sized its expert cache and its pools from the GPU's free memory at the load. Only development checks size from free memory now, and they say so.
- Every expert cache is planned by one rule, the codes pack's included: `glyd run` and `glyd serve` of Qwen3.8-Flash-Next at the hummingbird level plan its places by the rule every other expert cache takes. Its places are those of the pack's sidecar, as the loader holds them. A computer whose memory holds less than a token's experts in every layer is refused in words. Before, the codes pack had a planner of its own.
- `fit` counts the tables read by rows (DeepSeek-V4.1's Engram tables, Qwen3.8-Flash-Next's n-gram tables) apart from the active parameters: a token reads a few rows of each. DeepSeek-V4.1-Flash now shows "762.8B parameters (196.6B of them tables a token reads a few rows of), 31.1B active", where it said 227.7B active, and Qwen3.8-Flash-Next "179.6B parameters (51.2B of them tables ...), 9.9B active", where it counted its tables as active too. The figures of models without such tables are unchanged.
- No 8-row bound: Condor's server takes 1 to 64 users (`GLYD_USERS`), as many as its plan holds on the card, and a scheduler on pages runs the users its caller gives, up to 64 a step, as one on slots does. A request's `n` is bounded by the request's own limit, 64 sequences in all (`max_choices` in `/v1/glyd/info`); choices past the users the server runs at once wait their turn. Before, all three stopped at 8.
- `fit --users N` says when its plan holds fewer users at once than asked: a model whose layers are in stages over GPUs runs one sequence at a time. Before, the plan's text said nothing of the users it did not hold. Part of #239 and #333.
- A refusal for the computer's memory names the hummingbird level only for a model that has one (GLM-5.3, DeepSeek-V4.1 and Qwen3.8-2.4T-A95B were pointed at a level they do not run), and the plain dense refusal says its figure is all the users' ("for 4 users at 4096 tokens of context"), where it read as each user's. DeepSeek-V4.1's recorded plans are the planner's with its layers cut over GPUs: planned on eight H100s, refused with the experts' bytes on one GPU.
- DeepSeek-V4.1 runs with its release's quantizations of its activations, as DeepSeek-V4 already did with its own: every quantized product's input rounded to FP8 in blocks of 32 at power-of-two scales (the attention's and the experts', Engram's rows, the experts' activations, the attention's output before its grouped low rank), the window's key FP8 in 32s and the compressed entries FP4 in 16s at FP8 scales after their rotary, and the indexer's queries and keys FP4 in 32s without a rotation; DSpark's drafter the same. Which form a model takes follows from the FP8 blocks its config names (128: DeepSeek-V4's; 32: V4.1's); `GLYD_V4_SIM=0` still turns them off. `logitscheck deepseek-v41` now holds each of these operations to the CPU's bit for bit and prints the whole model against the kinds' reference with the same quantizations; `draftcheck deepseek-v41 --sim` runs. Part of #333.
- The random models at the releases' widths that the expert cache's checks use (`kimirandom --real`, `dsv4random --real`, `logitsmodel qwen35moe --real`) are drawn for width, not depth: a cut is the fewest layers that hold two blocks of experts (Kimi-K3 with its latent attention as the last of a short cut, the Qwen3.5 layout's short cut with both kinds of attention), the head draws logits of an RMS about 0.5 as GLM-5.3's and Kimi-K3's random models do, and Kimi-K3's release width takes the small model's conditioned draw (its decisive router, the depth mix, values and decay as drawn there). On them `storecheck --floor` holds within a tenth of the kind's bound and every host mode of the expert cache (cpu, split, auto, and with the decode calls planned on the GPU) within the kind's bound at 1, 8 and 32 rows. `storecheck` holds DeepSeek-V4 with the release's simulated quantizations in three parts, as its logits case does: every judged line without them, the quantizing operations bit for bit, the whole model with them printed as INFO. The first router choice apart in a row's history now prints the routers' inputs there, layer by layer. The second Qwen3.8-Flash-Next random case (`qwen4x-s2`) draws its Gated DeltaNet's decay as the family initializes it, so that its kernels' check holds at a prompt of 8 rows too. Part of #327.
- `rangecheck --workers` runs on a random model made with `logitsmodel CASE DIR --servable` (its tokenizer spells the text the check sends) and holds the range workers' seeded sampled runs to one GPU's scheduler's reply to the same request: three kinds (a temperature with top-p and top-k, a flat top-p whose cut lies past the GPU's candidates, a top-k of 200 with top-p and min-p), each with the rows' body asked for and without, every token, every row and every reply's `logit` the same to the bit, with one session and two. A worker's `status` now says its `draws` (`from_gpu`, `read_row`): the last worker's sampled tokens are all drawn from what its GPU made of the row, none by reading it; a repetition penalty is refused with the session as it was. Part of #138.
- `glyd-engine usercheck --ref --dir MODEL` (development builds) holds a real model's logits at many users to the CPU `Ref` backend: the model in a Hub directory (or a Hub id in the cache) on the GPU at the level asked, N prompts asked at once through the one scheduler so that their rows share the decode steps, and every row of logits a token was chosen from (the prompt's last position and each decode step) held to `Ref` running the model's description on the same weights, row by row. At the lossless level it is the layer kinds' bound (KL mean 1e-4, max 2.5e-4, the best token at 85% of the rows with a clear lead), on the run's own choices of experts: the reference takes each row's choices, and a choice apart from the reference's own must be a near tie on the engine's own input. At a lossy level it is the level's threshold. The last line says how many rows were held and the worst row's KL. The M2 gate's `logits-vs-ref` step runs it with `--engine`, so it is no longer a standing `SKIP`. The reference's products now split their rows over the computer's cores, four rows at a time a core, each row's sum the same terms in the same order (the same bits): about 7 to 9 times faster on a 10-core Mac. Part of #121.
- Internal: one planning rule. The lossy levels of Gemma 4, the Qwen3.5 layout and the Nemotron-H layout are planned by one function that reads what the model's description says: its weights at the level, a sequence's states and sliding rings, a token's keys and values, its draft model, and its experts. Gemma 4's own planner is gone. Every expert cache, on every route, is planned by one rule: every user's context first, the engine's work next, and the places get what is left. This covers a release's experts, a hummingbird directory, Kimi-K3's packed experts with every layer across the GPUs, and the lossy levels' experts in the computer's memory. The Nemotron-H and Kimi-K3 planners of their own are deleted.
- The Qwen3.5 layout with experts at kestrel, linnet and wren now keeps its experts in the computer's memory, with a cache of them on the GPU, where the GPU does not hold the model whole, as the Nemotron-H layout's lossy levels do. `--expert-cache` and `--host-memory` work there too. Before, these levels refused such a model. Qwen3.5-35B-A3B at wren on an RTX 4080 SUPER now plans 168 places a layer of 256, with 16.9 GiB of experts in memory. Qwen3.8-2.4T-A95B at wren still needs about 89 GiB of one GPU with its experts in memory, so one H100 still refuses it, now with that figure.
- A release's bf16 experts in the expert cache (Qwen3.5 layout, Muse Spark) are counted as the cache holds them, in the lossless tiered layout, at a bf16 matrix's share: in the places on the GPU and in the computer's memory. Before, they were counted at the release's bytes, more than is held. Qwen3.8-2.4T-A95B's experts take 3,180 GiB of memory (3,445 GiB on an HBM GPU), not 4,416.
- The planner's entry points take the model's config.json parsed once (`fit`, `run`, `serve`, `preflight`, `prepare` and `fetch`), not its text. A command parses the file once and passes the result to the family, the level, the plan, the refusals and the cost profile in `/v1/glyd/info`.
- A Gemma 4 refusal for `--users N` no longer says "for each of N users": the figure it gives is one user's, since the server holds fewer users where the GPU holds fewer.
- `--expert-cache` and `--host-memory` are refused for a Gemma 4 model, which has no experts to keep in the computer's memory, as they are for the other models without experts. Before, they were ignored.
- `run`, `serve` and `fit` size an engine's pools and an expert cache's places only by the plan. An engine that no plan sized takes the GPU's free memory only where a development check says so. Every product path, Condor's server (`qwenserve`) too, loads by its plan; anywhere else an engine no plan sized is refused in words, never sized silently.
- The recorded plans now include `fit` of GLM-5.3, Qwen3.8-2.4T-A95B, Kimi-K3.1-Pro and DeepSeek-V4.1-Flash, from their published headers, on an RTX 4080 SUPER, an L4, an H100 and eight H100s. Part of #136, #232 and #239.
- `benchmarks/gates/m2-gate.sh` runs the M2 gate on an RTX 4080 SUPER or an L4 from the pinned list `m2-models.txt`: for each row, level and context the planner's plan for 64 users, then `serve --users 64` with 64 streaming requests at once (each must end with its tokens; `/v1/glyd/info` and `/stats` must say 64 at once, with nothing sent back for room) and the server's memory against the printed plan; a model no one GPU holds must be refused (exit 3, a few lines). `--plan-only --machine PRESET` asks the planner alone, with no GPU. The checks other runs measured (`parity-kimi-k3`, `parity-deepseek-v4`, `parity-glm-5.3`, `bench-no-regression`, `packs-since-v0.29`) are read from `--ext`; the logits against the CPU `Ref` backend are `glyd-engine usercheck --ref` of the `--engine` build, a `SKIP` without it, which keeps the gate red. `scripts/check_gate_docs.py` now checks both gate scripts against their guide pages (`docs/guide/src/gate-m2.md` is the M2 page). Part of #121.
- A sampled token's row of logits is no longer read on the host in any case where the GPU makes the row. A draw weighs each token by exp((logit - largest) / temperature) as a whole number of 2^-40 (the same float operations on the host and on the GPU), so every sum a draw takes is the same in any order: where a top-p or a min-p cut, a top-k above 64, or the draw's point lies past the 64 largest logits, the GPU gives the weights and the tokens of ranges of the row in the draw's order (a few queries of 7 KB and at most 4,096 tokens), never the row. The draw is the one the host makes from the whole row, seed for seed, also for a flat row whose nucleus is thousands of tokens. Min-p is now a floor on the quotient (logit - largest) / temperature (ln min-p), as before up to its last bit; a token the bias names twice takes its last bias, as the server's JSON does.
- A model cut into stages across GPUs draws on its last stage's GPU, and a model with every layer across GPUs on its first, as one GPU does (their rows were always read on the host); drafted tokens checked with a sampler (a draft model or a draft head, on one GPU or across stages) are checked from the drafted rows' candidates on the GPU. The rows a step's draws see are made with the engine's head rows, a named part of the dynamic route's planned work.
- Served log-probabilities of several requests at once are each request's alone: a step whose rows ran one after another gave every row past the first another row's largest logit. `glyd-engine samplecheck CASE` now holds every request's log-probabilities at once to its own alone, fails on any row read on the host, and with `--split CUT` or `--tp A,B` holds each request on the stages or across GPUs to one GPU's reply to the bit; `topkcheck` holds the GPU's weights of every float to the host's and each draw's query to the host's bit for bit; `draftcheck` drafts with a sampler against the host's draws. Part of #138.
- DeepSeek-V4.1 plans and runs beyond one GPU. Its experts go to the expert cache as released (FP4, as DeepSeek-V4's) where the GPUs do not hold them, every one in the computer's memory. Its layers are cut over several GPUs: a stage begins only where no later layer reads a compressed part (entries, picks, candidate blocks) of a layer before it, and lagged mHC's weights are handed to the next stage with the streams. Engram's tables are read by rows on the host as the release stores them (FP8 rows with a power-of-two scale a run of 32, decoded to the bits the GPU's load made), so none of their 189 GiB is GPU memory. `fit` of DeepSeek-V4.1-Flash-1017 on eight H100s now plans it (three stages cut before layers 14 and 20, a 185.6 GiB cache of experts on the GPUs, all 268.9 GiB of them in the computer's memory) where it asked for 652.2 GiB on one GPU; a computer with too little memory for the experts is refused with the numbers. Part of #333.
- Engram's compressed vocabulary is made from the model's tokenizer, as the release's inference code makes it (NFKC, NFD, accents stripped, lowercased, whitespace runs one space, stripped; a byte piece of a character keyed by its name), where the model's directory has no map of its own: the release's tokenizer makes the 99,092 ids its config names. A tokenizer that makes another count is refused in words, as every hash multiplier is made from it. Part of #333.
- `glyd-engine storecheck --floor` (development builds) holds a random draw to its own noise floor before it judges the expert cache's host modes: the reference against itself with its experts' sums split over K on the GPU, the rows with no router choice apart within a tenth of the kind's bound and at least half of the rows judged so; a draw past it fails the check, as it judges no path that is not the GPU's bit for bit. `--pairs` compares each pair the CPU computes with the GPU's from the same inputs, and the CPU's FP4 pairs replayed in the order of an A10's tensor cores (mma.sync bf16: blocks of eight products fused with the accumulator, 24 bits below the largest exponent, rounded toward zero, measured on 2,560,000 outputs) against the GPU's bit for bit. `glyd-engine floortrace` shows, layer by layer, where a draw amplifies such a reordering. Row histories now hold DeepSeek-V4's indexer picks and the pooled indexer's picks of rows that decode together. Part of #327.
- A layer's experts stacked in one tensor a matrix (`mlp.experts.gate_up_proj`, `mlp.experts.down_proj`: transformers' form, the one Qwen3.8-2.4T-A95B is released in) are read an expert at a time, each expert's rows by their range of the file alone, by the Moe kind's block (resident and in the expert cache, at penguin and at kestrel, linnet and wren), the lossy route's packer and the hummingbird packer: no reader holds a stack's bf16 whole. The lossless stack's pack, the cache's places and every save are the bytes the same weights stored one tensor an expert give; a planned save may read a linear's member from rows of a tensor (`{"tensor", "rows"}` in its "from").
- `serve --users N` of a model whose experts do not fit the GPU (its experts in the computer's memory, a cache of them on the GPU) plans every user's keys, values and states first, at the context, as the server's pools take them. The engine's work comes next, and the expert cache gets the places that are left. Before, such a plan counted one sequence, and the cache's places filled the rest of the GPU. On an A10, `serve` of a 30 GB DeepSeek-V4 and a 26 GB Qwen3.5-layout model with experts then refused to start ("their pools hold 0 blocks") although `--users 8` was asked; they, a 37 GB Kimi-K3 and a 31 GB GLM-5.3-Flash in FP8 now serve 8 users at once on it. If the users do not fit beside a token's experts in every layer, the plan is refused with the figures and the number of users that fit. `glyd fit MODEL --users N` shows the same plan.
- The expert caches of a release's experts, of a hummingbird directory and of Kimi-K3's packed experts are planned by one rule. Kimi-K3's plan now reports its cache's places and memory too, as the others did.
- The hummingbird level of a model whose experts come from a codes pack counts its users the same way. Served to 4 users on an RTX 4080 SUPER, Qwen3.8-Flash-Next's plan gives the sequences 2.32 GB instead of 1.50 GB (each user's states are copied at its mark and at block boundaries) and keeps 50 places a layer instead of 59. A run's one sequence is counted with its rings at the rows the loader makes (348 MB instead of 376 MB at 8,192 tokens).
- `glyd-engine usercheck CASE --dir DIR --ctx plan --users N` loads the model in DIR as `serve --users N` plans it and holds its GPU memory and its users to that plan. `kimirandom --real` and `glmrandom` take `--servable`.
- `serve`'s ready line no longer says "one user at a time" before the line that says how many users the server holds. Part of #232 and #239.
- The lossless 12-bit KV pages are planned and served by attention kind on every route of the executor, not for Gemma 4 and the plain dense kind alone. A server's plan (`serve`, or `fit --users`) with `GLYD_KV=lossless` takes them wherever the model's description has rings in pools that the pages take: heads of 128, 256 or 512 channels, and latent attention's one stream. Heads of 64 and the compressed attention's entries stay bf16. The rule is the one-GPU paths' rule: the pages are taken only where they reserve no more than bf16 for the plan's users, each holding the whole context, or where bf16 does not fit and they do. On the dynamic route (the Qwen3.5 layout at penguin, Kimi-K3, GLM-5.3-Flash, GLM-5.3 and every other family there), the sequences are counted by the executor's own rule, with what the pages spend once. The Qwen3.5 layout's and Nemotron-H's lossy levels take them the same way. At the releases' shapes, 32 users at 8,192 tokens reserve less in 12-bit pages than in bf16: Qwen3.8-27B 23.66 GB against 27.53 GB, Kimi-K3 35.21 GB against 36.02 GB, and GLM-5.3-Flash 13.76 GB against 14.33 GB. One user at the context reserves more and stays bf16. The loader makes the model's 12-bit tables on its first run and keeps them, as for Gemma 4. `run`, `serve` and `fit` print the format, and `/v1/glyd/info` reports it. `run` answers one prompt at a time and keeps bf16. Gemma 4's server now takes the pages wherever it has no draft model, also without `--no-draft`, and `fit --users` plans them as `serve` does. Part of #332.
- Glyd's own release of Qwen3.8-Flash-Next loads again on the executor, as it did in 0.34.2. Its `model/` directory is slim: the routed experts are only in the pack beside it, and the n-gram table is stored at 4 bits (NF4 codes and a bf16 scale a row, 26 GB in place of 102 GB). A table read by rows on the host may now be stored that way under any description (`X.q4` beside `X.weight`); its rows are decoded as they are gathered, to the same bits 0.34.2 staged, and a table forced onto the GPU holds the bf16 it decodes to. A loader's check takes a checkpoint that holds none of its routed experts when a codes pack serves them, and says so in words where a level would read them from the checkpoint. The planner counts the table at the bytes it is stored in: `fit` of the release on an RTX 4080 SUPER now plans 72 places a layer for the precise-v2 experts (54 best-v2, 108 fast) and on an L4 154 (116, 231), with 25.6 GiB of tables read from the files, where it counted the 4-bit table as GPU memory and refused. `tools/serve.sh`, `run`, `serve` and `fit` take the release's directory: `run DIR/model:hummingbird --pack DIR/precise-v2` names a pack kept beside `model/`, as the download keeps it (before, `run` read only `DIR/pack`). Part of #140.
- The CUDA backend implements the layer-kind seam for every kind the executor runs: attention (sliding, shared keys and values, pooled indexer), latent and compressed attention, Gated DeltaNet, Kimi delta attention, Mamba-2, the MLP, the mixture of experts and the n-gram embedding, plus the norms and residual rules around them, the embedding and the head (`runtime::cuda`). Each kind's entry point runs the executor's own operations through the one function the executor runs them through, so a check through the seam and a served step launch the same kernels. `kindcheck` now runs every kind through these entry points and also holds the embedding and the head to the reference. The f32 MLP spike (`seamcheck`) is gone. Part of #135.
- A model cut into stages across GPUs serves the log-probabilities one GPU serves, to the bit: the last stage makes the step's largest logits and the sum of the row's exponentials on its GPU with the kernels one GPU's step uses, where the scheduler had read the row and summed it on the host (about 1e-8 apart). `splitcheck` compares each served token's log-probabilities between the two.
- DeepSeek's DSpark drafters draft on the executor, as block draft heads of the one draft-head operation (`lower::draft_head`): every DeepSeek release of V4 and V4.1 ships one. The head's layers are the model's own lowering of its window-only compressed attention and experts; the model taps its target layers' inputs (after an n-gram block before them), and the head keeps each layer's keys of main_x in its window; a block of the last token and its noise tokens runs in one pass whose rows see each other, then the Markov head's bias, one draft after another. The compressed attention verifies drafted tokens and goes back to the ones kept: its window and entries are written again before a row reads them, and its compressors' open windows are pooled again over the kept rows, across their blocks' boundaries; any model of that attention can draft. `draftcheck deepseek-v4` and `deepseek-v41` hold the drafted output to the model alone bit for bit and the head to the kinds' CPU reference on its own choices, and say the random heads' acceptance; `usercheck CASE --draft K` holds the replies through the scheduler with drafting to the model's own. Part of #333.
- Qwen3.8-Flash-Next's mixture-of-experts layers are the Moe kind's block, made by the one expert loader from the layer kind and its roles' tensors, as Kimi-K3's, GLM-5.3-Flash's and Nemotron-H's are; the family's own block of experts is gone. A codes pack's experts (the release's precise, best and fast packs: codebook rows of two, three and four bits, ternary, and format q's rows of two, three and four bits, each expert its own format) are the kind's hummingbird level, on the GPU or in the expert cache with every expert in the computer's memory. `glyd run DIR:hummingbird` and `fit` take the family from the codes pack beside its weights (`DIR/pack`), and `fit` of the release at the hummingbird level plans an RTX 4080 SUPER and an L4 with its experts in the computer's memory and a share of them on the GPU. Part of #140.
- `glyd-engine storecheck` (development builds) runs batched decode through the expert cache at 1, 8 and 32 rows against the same calls with every expert on the GPU, or against the cache in copy mode planned on the CPU for a model larger than the GPU: the logits bit for bit where the CPU computed no pair, else on the run's own router choices at the kind's bound; each decode step's bytes read from host memory for its misses within 2% of the distinct missed experts' bytes; each decode step one CUDA graph where the plan is on the GPU, and the reason where it is not. Part of #327.
- `logitsmodel qwen35moe DIR --real --layers L --experts E` makes the Qwen3.5 layout with experts at its release's width, and `dsv4random --servable` writes a tokenizer and chat template so that `serve` answers on it.
- A prompt chunk of more than 64 tokens through the expert cache's streaming prefill (Qwen3.8-Flash-Next's codebook experts) gives the same bits every run:
  each pair's row is written once at its place in the router's order and each token's rows are summed in that order, where before the experts' products
  were added into the tokens' rows with atomic adds in whatever order the GPU ran them. Which experts the host computes beside the GPU is the call's plan
  alone, no longer moved by the calls' measured times. A long prompt, and every image prompt, read the same logits on one GPU and across GPUs.
- Images on a model cut across GPUs are read again (`qwenserve --gpus 0,1`, `glyd run --split`, `serve`): the first stage runs the vision tower, every
  stage turns its rows by the images' positions, and the replies are one GPU's to the bit. `visioncheck repeat` holds an image prompt and a long text
  prompt to one GPU's first pass on one GPU, on a split and with the experts streamed instead of held. Part of #140.
- `fit` counts the parameters of a model whose routed experts are 4-bit as the release holds them: Kimi-K3.1-Pro is 2,780 billion (it printed 780), DeepSeek-V4-Flash 304 billion (165) and DeepSeek-V4.1-Flash 763 billion (508). Before, the experts' packed bytes were counted at the 16 or 8 bits of the rest of the model, and V4.1's `expert_dtype` in its `quantization_config` was not read.
- `fit` and `run` refuse a GLM-5.3 split asked for at a layer that takes the indexer picks of another stage's layer (`--split 7`, where layer 6 has the indexer of its own), in the loader's words: "a latent attention model: layer 7 takes the picks of a layer of another stage: a split must keep them together". Before, `run` printed its plan and the loader refused it only then; the layers a stage may begin at are the ones whose indexer is their own, read from the model's description.
- Kimi-K3.1-Pro (2026-11-02) is described from its config's keys: the layout is read by its text model (`kimi_linear`), not by the wrapper's release name, and its sparse attention indexer (32 heads of 128, LayerNorm-ed keys) is Kimi-K3's latent attention with one more block. Its safetensors headers (497,340 tensors) and Qwen3.8-2.4T-A95B's (1,609) are recorded by ranges, with no weight read, and held to the description: every tensor read is there at its shape, and the rest is listed.
- A Qwen3.5-layout directory is described in the form its index stores the experts in: Qwen3.8-2.4T-A95B holds each layer's 512 experts as two stacked tensors, where the layout's earlier releases are saved one tensor an expert.
- A request's penalties (repetition, presence, frequency), its logit bias, its allowed tokens, its grammar's mask (llguidance's, which the host still makes beside the step and uploads as a bitmask) and the end tokens held back until `min_tokens` are now put on its row of logits on the GPU, before the GPU makes the row's 64 largest logits, its sum of exponentials and its buckets. The host draws from those, as it did for a request that changed nothing, and reads 520 bytes a row in place of the row. Before, such a request read its whole row on the host at every step wherever the largest logits could not show the draw: always with a top-p and no top-k, a grammar with a top-p or no top-k, a penalty that pushed the 64th largest down, or a min-p with no top-k. The draws are the host's, token for token and seed for seed: the row the GPU makes is the host's bit for bit (each operation rounded alone, in the host's order), and log-probabilities stay those of the model's own row. A min-p with no top-k is now drawn from the largest logits too where they hold its cut.
- `glyd-engine topkcheck` checks the rows the GPU makes (masks of every density, the penalties, biases, end tokens put back) against the host's bit for bit, with their 64 largest, sums and buckets; `glyd-engine samplecheck CASE` serves a random model to requests of every kind of sampling at once (a JSON schema's grammar, penalties, a bias, an allowed list, plain) and holds each to its run alone and to the host's draws from the whole rows, and prints the host's time a step both ways. Part of #138.
- Several users decoding at once replay a CUDA graph of their padded rows on every family whose experts are on the GPU or in the expert cache planned on the GPU: Kimi-K3, GLM-5.3 (its latent attention with its indexers), DeepSeek-V4 and V4.1 (the compressed attention, its compressors, indexer and Engram) and Qwen3.8-Flash-Next (its picked attention, mean-pooled indexer and n-gram embedding, whose tables on the host give a step's rows as the host gathered them, their copy a node of the graph), as the dense, Gemma 4, Muse Spark and Qwen3.5 layouts already did. Each layer kind that reads or writes a sequence's cache or state takes a step's decode rows in one launch of each kernel, each row through its own block tables and states read from the step's metadata on the GPU; each row's sums are those of its own sequence's pass. With the expert cache planned on the GPU the graphs go up to 64 rows, one for each row count, unpadded (a padding row's choices would join the cache's plan): the plan on the GPU takes a decode step's call of up to 64 rows (another call of up to 8), and a call of more rows is planned on the host; a latent attention whose cache is in 12-bit pages keeps its steps of several rows eager. Part of #329.
- `glyd fit`, `run` and `serve` count the keys, values and states of a model on the dynamic route (the Qwen3.5 layout, Kimi-K3, GLM-5.3-Flash, GLM-5.3, DeepSeek-V4, Muse Spark, Qwen3.8-Flash-Next and the Nemotron-H layout at penguin). Before, these plans counted none ("0.0 GiB of cache") and the server's pools took whatever the GPU had left. They are now counted from the model's description, lowered as its loader lowers it, by the engine's own rule: each layer's rings and states, every user's states with a copy at its mark, the pool of state copies at block boundaries and each pool's first block. A model that needs more than one GPU holds one sequence across the GPUs, and its layers are cut with that sequence's share counted. Where no context is asked for and the model's row takes the most that fits, the context is halved until it fits.
- `glyd fit MODEL --users N` plans a server of N users on every path: the dynamic route, the lossy levels of the Qwen3.5 and Nemotron-H layouts (with the experts in the computer's memory too), and Gemma 4. A server of more than one user is planned without Gemma 4's draft model, because a drafted engine runs one sequence at a time. `serve --users N` loads that plan, so a Gemma 4 server of several users no longer loads the draft model.
- A served engine's pools take the bytes its plan gives them and no more. If the GPU has less free than that, the load is refused with the figures. The plan counts what the pools hold beyond their tokens: the states' copies and the first block of each pool. `run`, which answers one prompt at a time, plans one sequence as before.
- Condor's server (`qwenserve`) sizes its expert cache's places and its parts' rooms with the planner's functions. Part of #239.
- A model's embedding tables can stay in the computer's memory: a description may hold its token embedding (where it is not the head) and its n-gram tables on the host, read in place from the checkpoint's files. Each pass's rows are gathered by the host and copied to the GPU with the pass. Qwen3.8-Flash-Next's description does this for its 95 GiB n-gram tables and its embedding, as Glyd 0.34.2 did, so none of them takes GPU memory. The rows are read without the system's read-ahead, so a long prompt does not fill the page cache with neighbouring rows. The planner counts these tables in the computer's memory, not the GPU's, and `fit` names them. Part of #140.
- The engine and its GPU library build with CUDA 13 again, as the GPU job (`nvidia/cuda:13.0.3-devel-ubuntu24.04`) and the bundles build them, with no warning from nvcc 13: the Hopper kernels' tensor maps are aligned as the toolkit that built the kernels aligns a `CUtensorMap` (CUDA 13's `cuda.h`: 128 bytes; CUDA 12's: 64), which `build.rs` takes from the version of the nvcc it runs, and every number the engine passes the driver is held to that toolkit's `cuda.h` when the kernels compile. A tensor map is made at an address the driver accepts (64-byte aligned; it was 8). `build.rs` stops where nvcc does not say its version, and compiles the kernels once (no second try with other flags).
- On Windows the engine reads a GPU's TCC mode from the driver's TCC attribute (35; it read 17, the kernels' time limit), so a GPU under WDDM is seen as one: its plans keep 1 GiB more free and `glyd doctor` says so. Part of #134 and #238.
- DeepSeek-V4.1 runs on the executor: the compressed attention's one lowering with V4.1's forms (CSA2's entries, indexer keys, picks and candidate blocks shared across layers, lagged mHC, Engram blocks), and the n-gram kind's kernels take both its forms (Engram's compressed vocabulary, its pad id through it, orders 2 to 4, no convolution; Qwen3.8-Flash-Next's unchanged). DeepSeek-V4.1 is a family of `glyd run`, `serve` and `fit` (penguin, one GPU). Its release quantizes otherwise than DeepSeek-V4 (FP8 in blocks of 32, its window FP8 whole, its entries FP4), so V4.1's activations are simulated at those, as V4's are at its own. The kinds' CPU reference rounds where the release's model.py does (each RMSNorm's output and the streams' collapse and update, once), takes an engine's picks of a compressed attention as of a latent one, and its case is held on its engine's own choices at the kind's bound; kindcheck takes lagged mHC. DeepSeek-V4.1-Flash's weight map is checked against its released safetensors index (the newest release's headers, read by ranges). Part of #333.
- `glyd run`, `serve`, `fit` and `glyd-engine pack` take the Qwen3.5 layout with experts (`Qwen3_5MoeForCausalLM`, `Qwen3_5MoeForConditionalGeneration`) at the kestrel, linnet and wren levels too. Before, these levels were refused: the lossy route packed and sized the tensor list of the dense layout only. The packer and the planner now read the model's description, so each expert's gate and up are saved as one pack and its down as another, beside the routers and the shared expert. A dense Qwen3.5 model's saved files and plans are the same as before, byte for byte. Part of #333.
- The expert cache, which keeps every expert in the computer's memory and only the busiest on the GPU, now takes the Qwen3.5 layout with experts and GLM-5.3 as released in FP8 (each FP8 expert packed on its own, as released). Their plans at penguin use it where the GPUs do not hold the experts; a plan cut over several GPUs begins a stage only at a GLM-5.3 layer whose indexer is its own. Part of #232.
- The vision tower is a kind of the description (`VisionDesc`, the Qwen-VL vision transformer: Qwen3.8-Flash-Next's) and an input stage of the executor that
  any family's description can use: the description builder reads a multimodal config's `vision_config` and image token and names the tower's tensors
  (a tower with deepstack layers, a text rotary whose M-RoPE sections do not interleave, or a key it does not know is refused in words); the executor
  opens the tower from the description, runs a prompt's images through it when a pass reaches their rows, writes its rows over the image tokens'
  embedding rows, and turns every row by its M-RoPE position (an image's tokens (p, p + row, p + column), the text after it from p + the larger side) in
  the attention and the mean-pooled indexer, as the release's engine did. Images come through the common server and scheduler (OpenAI image parts,
  Anthropic image blocks, `--image-pixels N`), and `glyd run --image PATH --prompt TEXT` reads them too. A prompt with images finds in the cache only
  what prompts with the same images left. The tower has a CPU reference at the release's rounding points. Before, the family's engine on the executor
  served text alone since the release's own engine was deleted. Part of #140.
- Internal: a model's config.json is parsed once into the typed `ModelConfig` and passed down. Each family's directory reader, the pack, hummingbird pack and usage tools, the ceiling tool, the tier checks, the plain dense kind's lowering and the planner's dense sizing take it where each parsed the file again; no plan, refusal or description changes.
- With the lossless 12-bit KV pages (`GLYD_KV=lossless`), a sequence copied back from the computer's memory is counted once in the pools' room. Before, its pages were counted twice from the moment its copy reached the GPU until the scheduler next looked at it. Where a pool of raw slabs ran short, the scheduler then paused sequences that had room and refused copies back, and a decode step of one sequence, which asked for its room a second time, stopped the engine with "the paged cache has no block left for token N". On an A10, Gemma 4's case whose layers share keys and values, with 8 users and every page raw, stopped so on 8 of 10 runs; the other 2 were cut off at 10 minutes, still pausing and swapping sequences. With this change it passed 15 of 15 runs. The pools are now read once every copy to and from the computer's memory has finished, so each page is counted once, and the scheduler's decode step uses the room its plan took. Part of #332.
- `glyd run`, `serve` and `fit` take the Qwen3.5 layout with experts (`Qwen3_5MoeForCausalLM`, `Qwen3_5MoeForConditionalGeneration`) as the Qwen3.5 family at the penguin level: every expert on the GPU, the layers split over several GPUs where one does not hold them. Before, these models were refused as an architecture the engine had no description of. Their lossy levels take them too (the lossy route packs and sizes by the model's description). Part of #333.
- Qwen3.8-Flash-Next is cut over several GPUs again, now through the executor's generic split (the one Qwen3.5, GLM-5.3-Flash, Kimi-K3 and Muse Spark take): each GPU a stage of the layers on a thread of its own, the rows handed over through host memory. `glyd run`, `serve` and `fit` cut it at penguin where one GPU does not hold it (`--split CUTS --devices GPUS` chooses), where before they refused a cut. Condor's `qwenserve CKPT PACK --gpus 0,1` (`GLYD_GPUS=0,1 tools/serve.sh`) cuts the layers over those cards with each part's experts in a cache of its own, sized by its card (the cut by the planner's rule for an expert cache, or `--split K`), one request at a time as the release's engine did; a GPU named twice (`--gpus 0,0`, the split's check on one GPU) shares its memory between its parts. Before, a list of GPUs was refused: the release's own engine that split the family was gone.
- A model's draft head drafts across GPUs on the last stage (it holds the model's last streams, its LM head, and the embedding the head reads): a round's drafts there, its verify pass through every stage, every stage gone back to the drafts kept, the head again on the kept rows. Served through the scheduler, a split model that drafts decodes a round at a time. Before, a draft on a split model was refused.
- A stage after the first now has the rows' tokens, which an operation in it may read as the first stage's embedding hands them over: Qwen3.8-Flash-Next's n-gram embedding when its layer is on a later GPU (hash routing too). Before, such a stage read the tokens of whatever its buffer held.
- `splitcheck` runs a prompt past two chunks as well as the short one and takes the family's cases (`qwen4x`, `qwen4x-s2`, and `qwen4x-cb` with its pack in an expert cache a stage, `--mode copy|cpu`); `draftcheck qwen4x --split K` holds the drafted generation across stages to the model alone bit for bit and its rounds to one GPU's; `servecheck qwen4x-cb --gpus 0,0 --split 1` holds the served split's replies to one GPU's scheduler. Part of #140.
- `/v1/glyd/info`'s `precision.kv` now says the format the keys and values were loaded in: `12-bit` where the engine holds them in the lossless 12-bit pages (`GLYD_KV=lossless`, where the plan took them: the plan's `kv_format`), else `bf16`. Before, it said `bf16` for every model of the executor, 12-bit pages too. `memory.kv_bytes` of a model on 12-bit pages now counts the pages (with their escape and raw pools and the buffers a pass unpacks them into), which it left out, so `scratch_bytes` no longer counts them as working memory. `usercheck --kv12` holds the format the engine reports to the one the run asked for. Part of #332.
- GLM-5.3 (`glm_moe_dsa`) is a family of `glyd run`, `serve` and `fit`: penguin on the dynamic route, opened by the loader of the latent attention with plain residuals (FP8 as released), on one GPU or its layers cut over several, a stage's experts on its GPU or, where the GPUs do not hold them, in the expert cache as released. The planner begins a stage only at a layer whose indexer is its own: a layer that takes the picks of the one before it stays in that one's stage. Part of #333.
- The planner counts a latent attention's kv_b_proj and its indexer's products released in FP8 at the bf16 values the loader holds them as (two bytes a value), where it counted them at the FP8 share: by GLM-5.3's headers, 23.9 MB more on a layer with an indexer of its own and 14.7 MB on one lent its picks. Part of #333.
- GLM-5.3's MTP layer is a draft head on the description executor, on the draft-head operation Qwen3.8-Flash-Next's runs on: one lowering for every family's head (`lower::draft_head`: the head's blocks lowered by the model's own lowering, its input by the head's form), here DeepSeek-V3's joined form (enorm and hnorm, eh_proj of the two rows joined, a latent attention with an indexer of its own and the experts on a plain residual, shared_head.norm into the model's LM head). The description names layer 78's tensors, the CPU reference runs the head (a row run again at a position its caches hold takes its place, the latent attention's too), and the executor loads it as a draft part: FP8 as released, its experts held as the model's, its latents and indexer keys in rings after the model's, the model's LM head and embedding shared. A verify pass now takes the latent attention. `glyd-engine draftcheck glm-dsa` and `glm-dsa-f8` hold a drafted greedy generation to the model alone bit for bit and the head to the CPU reference on its own choices (its experts and its indexer's picks) at the kind's bound. Part of #333.
- Nemotron-H's mixture-of-experts layers are the Moe kind's block, made by the one expert loader from the layer kind and its roles' tensors, as Kimi-K3's and GLM-5.3-Flash's are; Nemotron-H's own blocks of experts are gone. What it needed are options any family can use: one block a layer; a latent with projections of the block's own; the block's other matrices at a layout of their own (six-bit at wren and hummingbird); relu squared experts' second matrices in the level's row layout, in the expert cache's places too, the host computing misses from their rows; a saved directory's packs read as saved; and a saved hummingbird directory's codebook rows, on the GPU, in the expert cache or on the disk. A call of the expert cache in several waves sums once, after its last wave. Part of #136.
- `qwenserve --gpus N` (Condor's `GLYD_GPUS=N tools/serve.sh`) picks the card the executor loads Qwen3.8-Flash-Next on. Before, the executor that serves the family ignored the setting and ran on GPU 0. Part of #140.
- GLM-5.3 (`glm_moe_dsa`) as released in FP8 runs on the description executor: at the lossless level every matrix the release holds in FP8 with fp32 block scales (`weight_scale_inv`) is held as released, its E4M3 bytes bit for bit: the attention's and the dense MLP's matrices by the FP8 product, the experts and the shared expert by the experts' FP8 product (which now takes experts narrower than the model's width and an activation the packed products would fuse, as GLM-5.3's are: 4,096 rows over 6,144, SwiGLU with no limit), and kv_b_proj and the indexer's products as their E4M3 values with their scales, each value times its block's scale once in the latent attention's kernels. The CPU reference reads such a matrix as the release defines it (each value its E4M3 code times its block's scale). A small random model of the layout in FP8 is a logits case (`glm-dsa-f8`) held to the CPU reference of the layer kinds on the engine's own choices. Part of #333.
- The FP8 product of one matrix gives a row the same bits whatever shares its call: its grid is a block a row group of 128 rows that adds all of the row group's K in order (the experts' product's rule), from the matrix's shape alone, where it was as many blocks as the GPU held at once for the call's rows, which split a row group's K by the rows the call held. `foldcheck` holds a row's bits at every count of rows to its own at one row. Part of #333.
- Kimi-K3's two random cases and DeepSeek-V4's are held on their routers' and indexers' choices at the layer kind's bound (KL mean 1e-4, max 2.5e-4, best token 85%), no longer at bounds set from an engine's run. Kimi's cases run on their description's CPU reference, which the hand-written one computes term by term, on draws that are well conditioned (the depth mix's projections, the delta attention's values and decay). DeepSeek-V4's is held in three parts: the engine and the hand-written reference both without the release's simulated quantizations, on the engine's choices at that bound (the random draw's experts at the scale of the other matrices, so that the reference's own noise floor is within a tenth of it); each quantizing operation bit for bit on the reference's own input, a part of the case's check; and the whole model with the quantizations, its figures and each choice apart printed and never a pass or fail. The engine traces its compressed attention's indexer picks so that each pick apart is judged on the engine's own inputs. Part of #135.
- A plan never promises more tokens than the pool it sizes holds, with bf16 pages or the lossless 12-bit pages. The planner's 12-bit bytes per token are now the pool's: a sixteenth of the page plus its escape room and its share of the raw slabs, 1.3% to 2.2% more than before on the random case of every family the pages take (`kv12check` on an A10; it now requires the planner's figure to be at least the pages' own and at most 1% above it). The pool now gives each page exactly that escape room, set by the page's shape, instead of twice what the calibration counted. A page whose values need more room is made raw, and the newest sequence is paused where raw slabs run short. Plans now also count, once for all users, what the 12-bit pages' pools spend as the context grows: a raw slab for every page of one sequence, which lets a sequence alone always fit, and the bf16 rows the pages are unpacked into. This costs up to about half of one sequence's 12-bit pages at long contexts. Both formats are lossless, so a plan now takes the 12-bit pages that `GLYD_KV=lossless` asks for only where they reserve no more than bf16 pages for it, and never plans a lower level for them; the loader takes the plan's word, and `fit` and the plan's JSON (`kv_format`) say which format it is. One user mostly keeps bf16: gemma-4-12B at its whole context of 262,144 tokens takes 4,581 MiB in bf16 and would take 5,226 MiB in 12-bit pages, so on an L4 it is planned at penguin in bf16 (before, at penguin in 12-bit pages counted at 3,465 MiB, fewer bytes than their pools take). Several users share the cost: eight users of gemma-4-12B on an L4 are planned 19,680 tokens each in 12-bit pages against 10,144 in bf16 (23,824 before). Gemma 4's plans size a sliding layer's ring in whole blocks, as the pools give them out. When the pools' budget falls short of every user's context, they now hold one sequence's blocks in every group first, so one sequence holds the context its plan gives it. Part of #332.
- No two runs of one 12-bit KV write fill one page at once. A run whose page another run of the same write names now writes nothing, and the status counts it and names the page (words 14 and 15 of `glyd_gpu_kv_write`'s status). The engine then fails the pass with an error that names the block. A pass's rows past the blocks their sequence holds are not written, so block 0, which belongs to no sequence, is never written. Before, the long-context check grew a sequence once and wrote its later prompt chunks' sliding-window rows into block 0, several runs to a call. Those runs cleared and filled one page at once and gave its chunks and slabs back more than once. The pools' counts then went below zero (on an A10, a group's count of raw slabs taken 1 to 5 below and of escape chunks 6,137 to 6,240 below, read as 2^32 less those), so a later step found no slab spare and failed ("no block left"), on 10 of 31 runs of Gemma 4's long-context line in 12-bit pages (0 of 27 with this change). The check now grows a sequence before each pass, as the scheduler does, and gives up a sequence it has no room for, where the scheduler pauses the newest. `kvshapes` makes such a colliding write on purpose and checks that both runs are refused and that nothing of the pools is lost. `usercheck --ctx plan --kv12` plans with the 12-bit pages (`GLYD_KV=lossless`). Part of #332.
- The expert cache's host path adds a token's experts as copy mode does. Where the host computes some of a call's experts (the `cpu` and `auto` modes; DeepSeek-V4's and Kimi-K3's FP4 experts), each pair's weighted row now goes into the layer's own sum beside the GPU's, and a token's pairs are added in the order of its choices and rounded once, whoever computed them. Before, the GPU's share was summed and rounded to bf16, then the host's share was added and rounded a second time. The error was half a last place of the GPU's partial sum, which, where the pairs cancel, is many last places of the result. Kimi-K3's host pair now rounds where the GPU's does: the second product's fp32 sum times the weight. The activations the host computes beside the GPU (situ, DeepSeek-V4's SiLU) now use one exp and tanh on both sides, Cephes' (0.99 and 1.24 last places at most, against the CUDA library's documented 2), so a host pair is the GPU's to the bit. Checked on an A10 with random models: a reply beside others is now the reply alone to the bit in both modes, where before it differed by a KL of up to 0.29 (DeepSeek-V4) and 0.087 (Kimi-K3). Kimi-K3's cached block now traces its experts' choices for the batch checks. Part of #232.
- Gemma 4 is held to its CPU reference. The head kind has a rounding option, and Gemma 4's description says its release's: the logits a bf16 tensor from the head's product on, through each of the soft-cap's three operations, as transformers runs the released model and the engine already did. Both Gemma 4 random cases are drawn well conditioned (the q and k norms' weights small, the embedding drawn as every other matrix), and the plain case has a fixture from the CPU reference at the kind's bound; its earlier fixtures were the engine's own bits. Part of #135.
- `glyd-engine packcheck --engine` (development builds) checks each format the engine reads a saved directory in (glyd-v7 at penguin, kestrel, linnet and wren) against the engine's own run of the same random model packed as it loads, in the same layout and in the same run, bit for bit. It also checks the case's reference where one bounds the level. This works on every GPU generation with nothing recorded. Before, the formats were checked against bits recorded on one GPU generation, so on a GPU of another generation nothing was compared, and a level with no recording failed. The Hummingbird level, which no level packs as it loads, is checked against a digest recorded for each kernel set (`tests/packs/nemotron-h-hb.KEY.json.gz`, KEY as `sm89-l4-58sm`: the GPU's compute capability, the words of its name the kernel choices read and its SM count). A GPU with no digest now fails that line; before, it printed a skip and passed. `packcheck --engine --record` writes that digest and nothing else. Part of #231.
- `logitscheck` says on each line what held it: another implementation's numbers (`PASS`, a reference) or the engine's own recorded bits (`PIN`, which shows that nothing moved, not that the logits are right). It counts the two apart and names the cases held only by pins. A case with nothing compared on this GPU fails; before, it counted as within bounds. `kindcheck` may leave out at most one row in sixteen near an expert or indexer tie; before, it could leave out any number. A reference value that is not a number now fails the comparison. `kvshapes` fails a comparison where more than one value in a hundred needs the bf16 probability-rounding term of its bound. Part of #135.
- `usercheck --kv12` with small pools fails where no sequence was sent back for room. The GPU job's Kimi-K3 line now runs eight replies, which overflow the pools; the four it ran before never did. `roundcheck` adds a set of prompts totalling 56 rows that fits a pass of 64, and prints SKIP for a set that does not fit.
- `scripts/consumer_bench.sh` exits 1 where a Glyd row did not run, or where other engines were asked for and none of them ran a row.
- `glyd fit`, `run` and `serve` size a model by the fields of its family's row in the table of families (the layout its matrices take, whether its layers
  run across the GPUs, how its expert cache is sized, which matrices may wait in the computer's memory, whether it drafts with its own head), not by the
  family's name; a name the table has no row for is refused instead of being sized as another family. No plan of a family the engine runs changes.
  Part of #239.
- `glyd run`, `serve`, `preflight` and `supports` take the plain dense route from a model's config.json alone: every pre-norm dense decoder the description builder reads runs on it, under any name (the route was one architecture's, and a Hub name beginning `qwen3-` was taken for it where the config could not be read). What the dense kind does not run (biases in the attention, a scaled rope, an activation other than silu) is refused before anything is downloaded, with the setting named.
- `glyd prepare` makes any model the engine runs ready, at the level `glyd run` would load it at: it takes a run's own steps (the level and the support table, the plan, the pre-packed download or the pack into the cache) and stops before the load. It served one family.
- `glyd verify` takes a directory of the Qwen3.5 layout at a lossy level (kestrel, linnet, wren) as `glyd prepare` and a run pack it: the linear attention's products packed as one (the queries, keys and values with the gate z; b with a) are the save's own, where it refused the directory.
- A model of a kind the engine has no description of is refused with that (`X is a YForCausalLM model, which this engine has no description of yet`); no refusal says the engine "runs Qwen3 models so far" any more. The support table's row and a plan's `family` name the kind, `plain dense` (they said `Qwen3`). Part of #240.
- Muse Spark loads through the description executor's own loader. Its matrices and small tensors are read by the names its description gives them, and each layer's experts are made from the layer kind and their roles by the one expert loader (`dynamic::experts`), which now takes experts stacked in one tensor, experts that work in a latent, a router's logit scale and outputs normed before they are weighed, and holds them as a stage does: a hummingbird directory's codebook rows, the expert cache's places, or one GPU's slice of each expert's rows. The executor's loader in turn holds one GPU's slice of a matrix's rows where every layer is across a group of GPUs, and a stage's last layers' matrices in the computer's memory. Muse Spark's own loader and the tensor names it spelled are gone; the random model builds the same description as before. Part of #136.
- The description executor's loader reads what the Kimi delta attention's tensors need, as options any description can use: a matrix whose last member names no tensor has zero rows after its tensors (a fused product padded to rows the library packs and a group of GPUs slices); a small tensor can be several tensors one after another, as saved; the Kimi delta rule's `A_log` and `dt_bias` are read in fp32, its convolutions and gated norm in the type the operation says, and each is held to the length its operation reads; the matrix of a product that reads where another product left its input counts among a stage's matrices. Part of #136.
- Kimi-K3 loads through the description executor's own loader. Its description names every matrix and small tensor of its text model at its shape, the fused products padded as before; each layer's experts are made from their roles' tensors, MXFP4 as released with their scales, and the corpus pass behind the hummingbird level now counts the layers whose experts are all on the GPU. Kimi-K3's own loader of the attention, the dense MLP and the norms is gone. Part of #136.
- Kimi-K3's plan with its last layers' dense matrices in the computer's memory counts there what the load puts there: every matrix of such a layer's own products, its latent attention's and dense MLP's too, so the GPU memory the plan frees and the host memory it holds are the load's. Part of #136.
- GLM-5.3-Flash loads through the description executor's own loader. Its description names every matrix and small tensor of its text model (the delta attention's, the latent attention's and its indexer's, the dense MLP's, the four streams' mixes and the norms), and each layer's experts are the layer kind's block: at the lossless level its experts as released in FP8, with its shared expert where that is FP8 too, an option any family whose description names its matrices' block scales takes, in the expert cache's places as well; at a block level packed at it, in the cache's places of that level. The mHC streams' base and scale are read in fp32 by the loader. GLM's own loader of the weights and of the experts' layer is gone, and its hummingbird directory, which that loader refused, now loads. Part of #136.
- The FP8 experts' product adds each row group's K in one block, so a pair's sum no longer depends on how many tokens or experts share its call: GLM-5.3-Flash's FP8 experts in the expert cache's copy mode give the bits of the experts on the GPU at every prompt length (they differed from 128 tokens). Part of #136.
- Nemotron-H loads through the description executor's own loader. Its description names every matrix and small tensor of its layers (the Mamba-2 layers' products, convolution, A_log, D, dt_bias and gated norm, the attention's products, the norms, the embedding and the head), packed in the layout the level gives every matrix but the routed experts' (six-bit codes at wren and hummingbird); each mixture-of-experts layer's block is made as before. Nemotron-H's own loader of the weights is gone. Part of #136.
- A model of the dynamic route, on one GPU, its layers in stages or across GPUs, with or without the expert cache, is opened by the loader of the layer kinds its description holds, not by its family's name: one stage loader for every description, the experts' level from the options, a cache where one is given (refused in words by a loader whose experts the cache does not take). A description of other kinds is refused in words instead of being loaded as DeepSeek-V4. Part of #136.
- A model family's own config.json keys are read typed from the one parse of the file (`ModelConfig::own`, `own_top`), for Muse Spark, Kimi-K3, GLM-5.3-Flash and Nemotron-H so far: a key of another type than the reader's is refused with its name, and a needed key that is null is reported as missing. Part of #136.
- The quality bar reaches the hummingbird level (R14, part of #235): `glyd-engine ship MODEL hummingbird DIR --usage usage.json` codes the experts
  from a corpus pass's counts and then takes the pack through the same quality card and the same refusal to upload as kestrel, linnet and wren
  (before, `ship` refused the level, so a hummingbird pack had no card and no gate). `levelcheck` holds hummingbird's KL from penguin on random
  DeepSeek-V4, Muse Spark, Kimi-K3 and GLM-5.3-Flash models, each against twice its first measured run (mean KL 0.093, 0.021, 0.015 and 0.029 on an A10), beside the block levels of
  Qwen3.5, Gemma 4 and the state-space case; a level that does not load now fails the check, where it was skipped. Kimi-K3 and GLM-5.3-Flash were open at first
  (the corpus pass counted none of Kimi-K3's layers; GLM-5.3-Flash's hummingbird directory did not load) and are held now. docs/quality.md.
- DeepSeek-V4 is loaded from its config's layer kinds alone: one lowering of the compressed attention kind, its streams and its experts (the same operations as before, to the byte, on DeepSeek-V4-Flash-0731's config and the random V4) names every tensor the loader reads, so the family's own description of its layers is gone. The same lowering takes DeepSeek-V4.1's forms (ratios 1 and 2 without a window bias, compressors, indexer keys and picks shared across layers, candidate blocks, lagged streams, Engram blocks, the router's gate temperature). Part of #333.
- The Qwen3.5 layout's family is checked at the lossless level in the shape its larger checkpoints have, beside the small one's: a second random case, `qwen35-wide` (three value heads to a key head in the linear attention, six query heads to a key-value head, a head of its own, the text model under a model with a vision tower, an MTP layer in the checkpoint), in `kindcheck`, `logitscheck` (its fixture is the CPU reference's of the layer kinds), `usercheck`, `keepcheck`, `roundcheck`, `splitcheck` and the 12-bit pages' checks; the GPU job runs it. The family's first case, `qwen35`, is held to the CPU reference of the layer kinds as well (`logitscheck qwen35 --cpu-ref`). On an A10 the engine's logits on the wide case are within a KL of 1.6e-5 on average and 3.5e-5 at most of the CPU reference's, and 4 and 8 replies at once are the replies alone, token for token, in bf16 and in 12-bit pages. Part of #331.
- The plain dense kind's random case is named for the kind: `dense` (it was `qwen3`) in `logitscheck`, `usercheck`, `keepcheck`, `roundcheck`, `kv12check` and the pack checks, its fixtures `tests/logits/dense*.json.gz`. `kindcheck` runs it, its logits are held to the CPU reference of the layer kinds (`logitscheck dense --cpu-ref`), and the GPU job runs it through the scheduler and on the 12-bit pages again. Part of #139.
- The Qwen3.5 layout with experts runs on the description executor: a layer whose feed-forward is a mixture of experts (a softmax router, the chosen experts renormalised, a shared expert gated by a sigmoid) runs the experts' operation in the MLP's place. Its block is made from the layer kind and its tensors' roles by one loader (`dynamic::experts`), which any family whose description names its experts a tensor each can use: lossless where the level is, or at kestrel, linnet and wren. A small random model of the layout is a logits case (`qwen35moe`) held to the CPU reference of the layer kinds. Part of #333.
- The Kimi family's newer config (its sparse attention, `dsa_config`) runs on the description executor: on the latent attention layers it names, an indexer of a key a token picks the tokens the attention reads, its keys RMS-normed or LayerNorm-ed as the config says. The latent attention's indexer takes the keys' norm and pooled or per-token keys as options of the one operation. A second small random model of the family is a logits case (`kimi-dsa`) held to the CPU reference of the layer kinds. Part of #333.
- GLM-5.3-Flash's random case (`glm`) is held to the CPU reference of the layer kinds on the engine's own choices: its routers' experts and its indexers' picks. The case's choices are near ties that rounding alone takes the other way (rounding its residual to bf16 moves its logits by a KL of 5e-3 on the reference itself), so wherever the engine chose otherwise the reference's router or indexer is run on the engine's own input and must agree with it or be within a near tie, and the logits are held at the kind's bound with the engine's choices taken. Part of #333.
- Prompts that come in together give each the logits of its own pass on GLM-5.3-Flash: the streams' mixing weights are summed by one kernel chosen by the layer's width and rounding, never by the rows the call holds (a prompt of 7 alone and beside two others had summed them in two orders). `kindcheck` holds a latent attention's indexer to the rule the logits checks hold it to: the reference takes the engine's picks, and each pick apart is judged by the reference's indexer on the engine's own input. GLM's head rounds the logits to bf16 as its releases' do. Part of #333.
- GLM-5.3 (the full-size family, `glm_moe_dsa`): its released config describes its released checkpoint (the indexers' schedule as transformers reads it, the indexer's FP8 products), and its layout runs on the description executor by its layer kinds: the latent attention's rotary (interleaved pairs of the shared key part, the query heads and the indexer's first channels), a layer whose indexer is shared attending to the picks of the layer before it, the plain residual with the release's rounded norms. A small random model of the layout is a logits case (`glm-dsa`) held to the CPU reference of the layer kinds on the engine's own choices. Part of #333.
- Gemma 4: a config with `num_kv_shared_layers` (the E2B and E4B releases) and `use_double_wide_mlp` describes. The last shared layers' attentions read the keys and values of the last earlier layer of their type (the attention kind's `kv_from`) and hold no cache of their own; their MLPs are twice as wide. The executor runs such a layer as the one attention operation reading the store layer's cache.
- Gemma 4's per-layer inputs (`hidden_size_per_layer_input`, E2B and E4B) describe and run on the CPU reference: a second table and a projection of the embedding make each token's inputs once, and each layer adds a gated branch of its own input after its MLP (the layer scalar after it). The executor does not run them yet, so neither release loads.
- A Gemma 4 directory whose generation_config.json is present but not valid JSON is refused with the file named; before, its end tokens (the tool response's among them) were dropped silently and generation did not stop there.
- Qwen3.8-Flash-Next (Condor's model) loads on the description executor with its routed experts from a pack of codebook rows (`glyd-hbr-codes/2`: cb2, cb3,
  cb4 or ternary per expert) in the shared expert store: a layer's places on the GPU sized by the room left after the always-on matrices and each user's
  states, the pack's file mapped as their host tier, misses computed by the host. Before, the executor refused the released model (its experts do not fit a
  16 GB GPU). The fast, precise and best packs take the same sidecar and skipping on both loaders (one rule). The executor takes the experts at any
  level, lossless at penguin. Part of #140.
- `paritycheck` and `logitscheck` hold the family's executor to the CPU reference of the layer kinds, not to Condor's own engine: the two run different
  kernels of the delta rule, so their bits part there. The family's two random cases are held at the bound the Qwen3.5 layout's fixtures hold (KL mean
  1e-4, max 2.5e-4); the 5e-4 the first case's fixture carried was fitted to a run of the executor and is withdrawn. A router's choice taken the other
  way is no longer covered by a wider bound: wherever the executor's choice is not the reference's own, the reference's router run on the executor's
  own input must make the same choice or one a near tie from it, and the logits are held to the reference run again on the executor's choices; the
  figures against the reference's own choices are printed beside them. At penguin the checks also read the experts back from the GPU and hold them
  to the checkpoint's bits, and the second random case's experts are no longer on the six-bit grid, so a lossy load under the lossless level's name
  fails. Part of #140.
- Where the engine computes experts on the host (an expert cache whose mode gives the host its misses: cpu, split or auto, and the codebook cache),
  `usercheck` with prompts one a pass, `keepcheck` and `roundcheck` hold each row run beside others to the same row alone at the kind's lossless bound (KL
  mean 1e-4, max 2.5e-4; the best token the same wherever the lead clears 0.1), not bit for bit, and say why: the host and the GPU sum a pair in different
  orders, and which share of a call the host takes turns on what shares the call and on the places' state. Each row is judged by the first discrete
  choice the two runs made apart in its sequence's history (an expert route or an indexer's pick of pools, every layer and step, in causal order): none
  apart, the row is held to the bound; a near tie (below 1e-2 on each run's own input), the row and every later one are reported, not failed, with that
  choice; anything wider fails. The lines count rows held, reported and failed apart. Everywhere else (every expert on the GPU, copy mode) they stay
  bit for bit. Before, those checks failed on the codebook cache's bits alone. Part of #140.
- `fit --machine NAME` no longer turns on the computer that plans it: a preset says whether it has NVIDIA's cuBLAS (none of them: the engine never ships
  it), and a machine described in JSON says so with `cublas`. DGX Spark's plans had a long prompt's chunk of 4,096 tokens when planned on a computer with
  cuBLAS and 2,048 elsewhere. Part of #140.
- Qwen3.8-Flash-Next's MTP layer is a draft head on the description executor: its config's `mtp.*` are described (`MtpInput::Streams`: the model's last
  streams normed as one vector and each stream projected by `fc_hidden`, the next token's embedding normed and projected by `fc_embedding` added to each,
  an indexed attention and its experts in gated residuals as the model's layers, its own final mixer into the model's LM head), the CPU reference runs
  it, and the executor loads it as a draft part (its experts held as the model's, its keys and values in rings after the model's, the model's LM head
  and embedding shared). A drafting round checks the drafts in one verify pass whose rows are the steps', then goes back to the rows kept: the delta
  rule's and its convolution's states and now the n-gram embedding's last tokens and convolution. `glyd-engine draftcheck qwen4x` holds a drafted
  greedy generation to the model's alone bit for bit and the head to the CPU reference at the family's bound. Part of #140.
- `qwenserve CKPT PACK --port P` serves Qwen3.8-Flash-Next on the description executor through the common scheduler (the server every model of the executor
  has), its routed experts from the pack in the expert cache (`--places N` a layer, else the plan's); Condor's own request loop (`Qx`) is gone.
  Not on the executor: drafting over the expert cache (a verify pass takes experts held whole on the GPU). `glyd-engine
  servecheck qwen4x-cb` holds a served greedy completion to the same tokens through the scheduler in one process. Part of #140.
- `glyd run`, `glyd serve` and `fit` take Qwen3.8-Flash-Next at penguin through the dynamic route's openers, the same as every family of the executor: the
  release's weights exactly, every expert on the GPUs (one GPU, or its layers cut over several where one does not hold them), sized from the files' headers. The released model needs about
  327 GiB on one GPU, so every machine preset of one GPU refuses it in those words; a model of the family that fits one GPU runs. Its packs of codebook experts (precise,
  best, fast) are served by `qwenserve CKPT PACK --port` and, at the hummingbird level, by `glyd run` too. Before, `glyd run` and `fit` refused the family whatever its size. Part of #140.
- Qwen3.8-Flash-Next's own engine is gone (`model_qwen4x.rs`, its development checks `qxlayers`, `qxcb`, `qxbench`, `qxprod`, `qxhost`, `qxtoy`, and the
  standard-input server): the description executor runs the family everywhere. What it kept of that engine moved: the family's kernels the executor launches
  (`kern::qwen4x`), its experts held whole on the GPU (`moe_qwen4x::QxMoe`, since replaced by the Moe kind's block). `paritycheck` holds the executor to the CPU reference alone, layer by layer
  (the old engine's record stays in `tests/fixtures/qwen4x_condor_rows.json.gz`). `qwenserve CKPT PACK` without `--port` now says to serve over HTTP:
  the JSON-lines protocol on the standard input was that engine's, and only development scripts used it. `qwenget` downloads as before. Part of #140.
- The description executor decides whether an engine holds several sequences, runs a verify pass, and which state types it reports from the operations of
  the model's description, not from the family's name. Qwen3.8-Flash-Next on the executor serves several users at once, its sequences batched in one pass.
- The planner prices a step of many users through the expert cache by the distinct experts the step misses, each read once, where it had priced one user's misses at any number of users.
- At many users a step's experts are placed busiest last, so that they stay on the GPU for the next step.
- `glyd-engine ceiling --stats` follows the expert cache's queues layer by layer (the copies of a layer wait behind the previous layer's promotions, which wait for its products; a step copied whole has the CPU wait for it). On the RTX 4080 SUPER's measured runs of random DeepSeek-V4 and Kimi-K3 at the release's widths, calibrated on each model's copy run, it predicts the auto runs at 1 and 8 users within 7% (DeepSeek-V4 at 8 users: -6.8%, where it was -13.4%). `storebench --out` writes each layer's totals, with the calls copied whole.
- Development builds can plan the expert cache's decode steps on the GPU (`--device-plan`; `tiercheck` and `storebench --device-plan`): for DeepSeek-V4's and Kimi-K3's FP4 experts, decode calls of up to 64 users split their misses between copies and a CPU thread as the CPU's plan does, with no CPU wait inside a step, so a decode step is one CUDA graph (checked on an A10 with random models). Part of #327.
- DeepSeek-V4 is paged like every other attention model on the executor: its sliding window and its compressed entries (the attention's and the indexer's) live in the executor's block pools, a compressed group's block holding 16 entries (16 x ratio tokens). A sequence takes the blocks its tokens fill instead of a whole context's entries, so the page budget, preemption, swapping to host memory and prefix sharing apply to it. A start is found by name in whole blocks of every group: for DeepSeek-V4's ratio-128 layers, in steps of 2,048 tokens.
- A sequence's swap to host memory does not hold up the next step: its copy runs beside the steps after it, and a step waits only for the part of the copy that reads blocks it takes (the first few blocks of each group are copied first).
- A sequence on 12-bit lossless KV pages is swapped in its 12 bits: each page moves as it is, with its escape chunks and raw slab, and is lent new ones where it is loaded (GPU library C API 21: `glyd_gpu_kv_image_bytes`, `glyd_gpu_kv_save`, `glyd_gpu_kv_load`).
- Deterministic capacity on 12-bit lossless KV pages: a request is never refused because its keys and values compress worse than the calibration. Where the raw pool is short of a slab for every page the next pass writes, the newest request is paused (swapped or recomputed) and its pages' chunks and slabs go back to the pools; a request alone always fits. Part of #141 and #328.
- Part of #138. With `GLYD_ENGINE_OVERLAP=1` (off by default until it is joined with the batched decode, the row-bucket graphs and the pieces), a served model's step is launched before the step before it has been read, where every sequence in that step takes its largest logit (greedy, no log-probabilities): the next step's rows read their tokens on the GPU, and the host sends the tokens, ends the sequences that ended and schedules while the GPU runs. A decode step no longer waits on the whole GPU stream: its inputs and samples are copied through pinned memory and the wait is for the step alone, a block table that changes is copied on the stream, and a step whose requests draw (not greedy) has its 64 candidates a row made and copied with it. On every model on one GPU, an expert cache too where its layers are planned on the GPU (`--device`; its end of a step runs on a step that lands with none queued after it, one in 16 or fewer): a recurrent model's states (Qwen3.5, Kimi-K3, GLM-5.3-Flash, DeepSeek-V4) are copied at a prompt's end and at block boundaries as the next step is launched, the 12-bit pages' status is copied with the samples, and Qwen3.8-Flash-Next's n-gram embedding reads the token ids made on the GPU.
- `GLYD_PROF=1` and `GLYD_ENGINE_STEP_PROF=1` give each step's GPU time and the GPU's idle time before it (median and mean), from events at the step's first and last work.
- Models split over GPUs (`--split`, `--tp`) and models with a draft model are served by the scheduler too, one request at a time: a drafted model's decode step is a drafting round, and a request that asks for log-probabilities or `min_tokens` is no longer refused on these models (it runs plain steps).
- A draw no longer reads the whole row of logits on the host where the GPU can make what it looks at: a top-p with no top-k takes the row's sum of exponentials, log-probabilities the row's log-sum-exp and its own 64 largest logits (under a grammar too, from before the mask), a grammar's mask is put on the row before its top-k, an allowed-token list takes those tokens' logits, and a draw over the whole vocabulary (temperature alone, or a top-k past 64) takes the row's weights in 1,024 buckets and the tokens of the bucket it lands in. The same tokens as the whole row gives.
- DeepSeek-V4 with its experts in the expert cache and the decode calls planned on the GPU (`--device-plan`) plans them there in copy mode too, so its decode step is one graph and its steps can be launched ahead. The expert cache's cold state (`ExpertStore::cold`) now resets the GPU plan's own state (which experts the host computed lately, its step and clock) and gives the plan the times it resets, so two runs from it plan alike; with an expert cache no capture of pieces is tried (a prompt's chunk plans its calls on the host).
- Consumer benchmarks for every release (P20): `scripts/consumer_bench.sh` measures the models of a GPU's memory tier (8, 12, 16, 24 or 32 GB,
  `benchmarks/consumer/matrix.tsv`) on Glyd, llama.cpp, Ollama and LM Studio on the same GPU with `glyd bench`: the same weights at about the same
  bits (llama.cpp's GGUF made from the same checkpoint, the same file for Ollama and LM Studio), no draft model on either side, each run's command
  lines, versions and server log kept beside its result. `glyd bench --report DIR` writes the comparison page: per GPU each engine's largest model at
  20 tokens/s or more, and each model's runs side by side. Results record `--engine`, `--source`, `--bits` and `--params`. The `Consumer bench`
  workflow runs it on a self-hosted GPU runner when a release is published. `glyd bench --suite occupancy` (1, 8, 32 and 64 at once) is the 1x H100
  occupancy check; the H100, H200 and the tiers' consumer cards are in the bench's spec-sheet bandwidths. Part of #242.
- GLM-5.3-Flash runs on the description executor alone: its development engine is removed (`model_glm.rs`, `mla_glm.rs`, `mtp_glm.rs`, `kcheck_glm.rs`, `glm_serve.rs`, `glm_cb.rs`, `draft_loop.rs` and the codes block of `moe_glm_cb.rs`: 5,032 lines), with its development commands (`glmserve`, `glmcbtest`, `glmcbblock`, `glmkernels`, `glmlayers`, `glmdyn`, `glmtoy`, `glmmtp`, `glmmtpreal`, `glmspec`, `glmreal`, `glmprof`, `glmdequant`, `glmdecode`). What the executor loads GLM by is `kern/glm.rs` and `kern_glm.rs` (the kernel tables); `glmrandom` stays. GLM's experts are tiered by hummingbird and the block levels (R12). Its MTP drafting went with the engine: R6 (#145) puts the MTP head kind on the executor.
- Condor keeps the codes pack reader (`cb_pack.rs`). Part of #234.
- The engine runs the GPU library's operations with its own kernels (the packed layouts' unpacking, the experts' plan, the step products of 1 to 64 tokens and the prompts' products, the grouped experts' products, the 12-bit KV pages' write, unpack, gather and attention, the lossless FP8 format and its products), compiled from the GPU library's own source into the engine (`gpu/glyd_gpu_kernels.h`, `glyd-engine/kernels/fold.cu`), not by calling the GPU library. It decides itself which kernel, or the matrix decoded for cuBLAS, takes a packed product of a prompt's length on a GPU, reads the GPU's class from its name, and sizes the 12-bit KV cache's pages, pools and decode workspace. `glyd-engine foldcheck` (a development build; in the GPU job) compares every output and every such answer with the library's, for every GPU class the library knows; a development build's `GLYD_ENGINE_FOLD=lib` runs the library's instead. An NVIDIA build of the engine loads no GPU library: its own kernels take every call, and one they do not take is refused as the library refuses it. `GLYD_GPU_LIB=PATH` still loads the library (`foldcheck` compares with it); an AMD build loads its library as before.
- The GPU driver (NVIDIA's, and HIP's runtime in an AMD build), Glyd's GPU library and cuBLAS are each reached through one interface (`src/device.rs`, `src/gpu_ops.rs`), with nothing changed on the GPU: every check gives the same verdicts and the logits fixtures the same bits, and on an A100 (2026-11-24, `benchmarks/gpu/a100-device-seam-2026-11-24`) `bench --compare` puts every pair of runs before and after it within the noise band.
- `glyd-engine/docs/design/kernel-libraries.md`: which operations use NVIDIA's libraries and which Glyd's own kernels, per GPU, with the numbers measured so far and the test that decides each open row; on the same A100 cuBLAS multiplies a decoded matrix 2-10% faster than Glyd's own dense product, and keeps it.
- The decode attention's workspace puts its partial outputs on a 16-byte line: a batch whose sequences, heads and splits multiplied to an odd number faulted with a misaligned address (C API 23). Part of #134, #238 and #255.
- `serve` plans a Qwen3.5-family or Nemotron-H model for its `--users`: each user's states and context, as the executor gives them. Before, the plan counted one user, and on an A100 SXM4 40GB Qwen3.5-0.8B served to 8 users used 5,054 MiB against a plan of 3,037.
- `glyd-engine plancheck` now reads the GPU memory the server used. On the A100 it printed "memory -" in every case, so its memory gate failed them all.
- Decode for several users at once is predicted as one step for all of them. The step reads the weights once, plus each user's keys and values, plus a per-layer time for each row past the first, fitted from each GPU class's records. Before, the A100's 8-user predictions for Qwen3.5-0.8B and gemma-4-12B were 78-193% too fast. Each class also fits its own figure for the lossless layout: the A100's 12-bit runs had been predicted 28-39% too slow.
- `plancheck` takes `--ctx N`. It also predicts what bench measures when several users send their prompts at once: each reply waits for the prompts read after its own, and the first token comes after half of the prompts.
- A plan where users share pools of blocks (Qwen3, Qwen3.5, Gemma 4, Nemotron-H when served) now holds at most what the GPU has left for their cache. On a 23 GB A10, a dense model served to 8 users at 16,384 tokens had been planned at 31,867 MiB; the server took 21,889. 3 of the 2,250 recorded plans change, all three over their GPU before.
- Penguin's decode is now predicted from the layout the loader takes, the 12-bit one everywhere but Ada, read at a share of its own. Before, it was the tiered layout at the RTX A6000's share, and the 12-bit runs measured on the A10 were predicted 19-25% too slow.
- Prompt speed is now predicted against the GPU's measured memory rate instead of its listed tensor rate. A GPU class with no records takes its prompt figures from every other class.
- The shipped records hold recent releases only: Qwen3.5-0.8B and gemma-4-12B on the RTX 4080 SUPER, the A100 and the A10 (27 records). With them, each class's runs predicted from the other classes' figures were off by a mean of 23.5% (A100), 17.7% (A10) and 24.7% (RTX 4080 SUPER) for prompts; before, the A10 was 119.7%. Decode was off by a mean of 51.9%, 17.0% and 14.4%. From a class's own records less the one predicted, decode was off by 5.9%, 3.5% and 0.9%.
- `fit --json` shows each stage's share of what the computer's memory holds for a model in stages (`spread.host_per_stage`; the shares add up to `host_bytes`). The dynamic route's loaders now load from the plan alone: its stages or slices, each stage's expert places, and the memory the plan gives them. Part of #150 and #239.
- The Qwen3.5 layout (Qwen3.5, 3.6 and 3.8 dense) runs at penguin, lossless: `glyd run Qwen/Qwen3.8-27B` loads it on the dynamic route, on one GPU where it fits (served to several users at once through the scheduler) or with its layers cut over the machine's GPUs (two A100 40 GB or two RTX 3090 cards; one user at a time there). Its lossy levels run on one GPU or, where one does not hold the model, in stages over several. `fit` plans it from the files' headers: Qwen3.8-27B at about 36.1 GiB by the planner's estimate, 39.1 GiB on an A100 or H100, where it is held in the 12-bit layout.
- The dynamic route reads `--ctx` as `--context`, as the one-GPU paths do. Part of #331.
- A quality bar for the lossy levels (R14): `glyd-engine quality` measures any OpenAI-compatible server on one fixed suite: next-token KL after 192
  prefixes of four fixed texts, and small reasoning, code and long-context task sets (40, 40 and 4 a length) made from a seed, with known answers and
  no dataset. `quality --card` takes a pack's card against the lossless level of the same model and holds it to the level's bar (kestrel: mean KL
  0.02, top-1 94%, a task 5 points; linnet 0.05, 91%, 7.5; wren 0.08, 86%, 10; hummingbird 0.12, 90%, 10): publish or refuse. `ship` writes the card
  into the pack (`quality.json`, and a Quality section of its README) and does not upload a lossy pack without a passing one. `glyd fit` shows each
  level's figures, or "not measured". `quality --table` sets cards side by side by bits, for comparisons with GGUF, AWQ and FP8 at equal bits
  (llama.cpp's log-probabilities are read as well). docs/quality.md. Part of #235.
- Kimi-K3's and GLM-5.3-Flash's latent caches can be held in the lossless 12-bit KV pages (`GLYD_KV=lossless`, as for the dense families): a latent is one stream of the page format, Kimi-K3's 576 values a token in five heads of 128 and GLM-5.3's 512 in four, and the latent attention reads a sequence's pages unpacked to bf16, so its outputs are the bf16 cache's bit for bit. GLM-5.3's indexer keys stay bf16. A latent's decode reads more bytes than with the bf16 cache until the attention reads the 12-bit pages in place.
- The GPU library's `glyd_gpu_kv_unpack` skips a negative page number (C API 21).
- The tokens a GiB of the 12-bit pages are held to the planner's for every format: by the pools' layout (a unit test, at 2 GiB: heads of 128 +4.0% against the planner's 1.59 bytes a value, heads of 256 -1.6%, a K=V head of 512 -1.4%, Kimi-K3's latent -2.3%, GLM-5.3's -2.7% against a sixteenth of a page), and on the GPU by `kvshapes`, which fills a pool of each format and holds it to the same 5%. Part of #332.
- A prompt's products of the lossless (tiered) and lossy layouts are decoded once to bf16 and multiplied dense (cuBLAS, or the engine's own product) from 512 rows on a GPU whose route table measured no crossover for that layout: an A100's, where the library's fused prompt kernel took them at any length (the 12-bit layout there is decoded from 769 rows, as measured). The GPUs and layouts the table has a crossover for keep it. The families whose descriptions take the decoded way have it (the Llama-style dense ones, Qwen3.5, Nemotron-H); the decoded matrix's room and cuBLAS's handle are now made only where a matrix takes that way at a chunk's rows.
- A prompt chunk's attention for heads of 64 and 128 (the Llama-style dense models, Nemotron-H, Muse Spark) runs a FlashAttention-2 loop of the attention kind's own: tiles of 64 keys copied by cp.async (the values while the scores are made, the next keys while the values are summed), only the tiles at a row's diagonal or window edge masked, and a key-value head's query heads in one block, so each tile is read once for all of them. Over the 12-bit KV pages the chunk's pages are unpacked to bf16 first and the same kernel runs, so the outputs are the bf16 cache's bits. A row's output is now the same bits whatever chunk or start it is in (`attncheck` runs the rows in two calls). Heads of 256 and 512 keep the GPU library's kernel.
- `GLYD_PROF=1` shows a matrix's decode to bf16 and the 12-bit pages' unpack for a prompt as stages of their own. A development build's `GLYD_ENGINE_DECODE_FROM=N` takes the decoded way from N rows whatever the route table says (a large N: never). Part of #255.
- Several users decoding at once on the description executor: a step's rows that decode take each attention layer in one launch of the preparation and one call of the decode kernel, each row through its own sequence's block table, where there was one of each a sequence a layer, and the gated delta rule's and Mamba-2's convolution and recurrence take a step's rows in one launch, each row through its own state. A step of nothing but such rows replays a CUDA graph of its rows padded to 2, 4, 8, ... up to the users served (every size captured before the first request), as one user's step already did, for models made of attention layers, such recurrent layers and operations on each row alone (the Llama-style dense models, Gemma 4, Muse Spark, the Qwen3.5 layout, Nemotron-H). Each row's tokens are those of its own pass, bit for bit, over bf16 and 12-bit KV pages alike. On an A100 (40 GB) with bf16 pages, the attention's part took 8 users' decode tokens/s per user from 48.5 to 83.8 on gemma-4-12B at wren (with its graph) and from 190.4 to 236.1 on Qwen3.5-0.8B at kestrel, one user unchanged; on an A10, the recurrent layers' part and their graph took Qwen3.5-0.8B's from 230.7 to 292.4 with short prompts and from 165.5 to 194.5 with 1k prompts.
- Full prompt and mixed steps (a chunk's rows: long prompts, alone or beside decode rows) run in pieces: the products, norms and experts between two attention or recurrent layers are one graph each, captured when the server starts, and the attention runs eager between them, so the host queues one launch a piece where it queued every kernel. A replay is the eager pass bit for bit (pieces are captured for a pass's exact rows, not padded); decode steps of models whose steps take no whole graph (a latent attention whose cache is in 12-bit pages) run in pieces too, at 2 to 16, 32 and 64 rows, captured at start; passes of other sizes, and passes whose products take cuBLAS (the A10's and A100's prompt passes, where pieces cost 0.7% of a full pass's GPU time on gemma-4-12B), run eager. A step's metadata (positions, the decode rows' slots and their states' addresses) is written to the GPU in one copy, which every graph reads.
- `GLYD_PROF=1` prints the attention's decode calls and the piece graphs replayed a pass; a development build's `GLYD_ENGINE_ROWS_GRAPH=0` and `GLYD_ENGINE_PIECES=0` run such steps and passes eager. Part of #329.
- A development build set `GLYD_PROF=1` times the description executor's passes by operation on the GPU itself, with no external profiler: CUDA events after each operation (inside the decode step's graph where a step replays one), the GPU time a pass of each kind (decode rows, prompt chunks, both) by operation, and each scheduler step's wall time split into its passes' GPU time and the rest, with the time between steps. `glyd-engine bench` prints them under each load and keeps them in the result (`gpu_profile`); `serve` answers `GET /v1/glyd/prof`; `glyd-engine run` prints them at its end (docs/bench.md).
- A served executor model's sampled replies whose draw looks at no more than the 64 largest logits (a temperature with a top-k of 64 or less: Gemma 4's defaults) draw from those, made on the GPU after the step, in place of copying each row's logits to the host (262,144 of Gemma 4's) and drawing there; any other draw reads the row as before. `usercheck` holds them to the rows' logits. Part of #138.
- `serve` with `--users 1` of a model with sliding-window layers (Gemma 4) answers prompts longer than the window and a chunk (from 1,552 tokens for Gemma 4, whose window is 1,024 and chunk 512). Before, such a request never got a reply: the room taken for the whole prompt held the sliding layers' blocks of the prompt's last window, the one user's pool had none left for the first pass, and the request was sent back and let in again without end (seen on an A100 with 4,000 and 8,000-token prompts: no reply after 82 minutes). The room now holds what the first pass reads. A request that the cache cannot hold even alone now ends with an error rather than waiting.
- The same rule gave back, at any number of users, the window before a kept conversation's end (or a start found in the cache) when the prompt added more than a chunk to it: its first pass read the sliding layers' window from blocks taken again, not the conversation's keys and values. It keeps them now.
- `bench` ends the server it started however the bench ends: Ctrl-C, SIGTERM (`timeout`) and SIGHUP kill it first, and on Linux the server is killed when the bench is, a SIGKILL included. Benches ended by `timeout` used to leave their servers running.
- One latent attention operation for every latent-attention model on the description executor (Kimi-K3, GLM-5.3-Flash): its heads, latent widths, shared key part, where its input products are, its output gate and its indexer come from the model's layer kinds, and one set of kernels runs it (kernels/mla.cu). Kimi-K3's and GLM-5.3-Flash's own latent attention blocks on that route are gone.
- Kimi-K3's and GLM-5.3-Flash's latent caches (and GLM-5.3's indexer keys) are now pages of the executor's pools, as every dense attention's keys and values are: the scheduler serves these models with the same page budget, swaps a sequence sent back for room to the computer's memory and copies it back, keeps their delta-attention states at a prompt's end, and shares a prompt's start between requests. Before, each sequence held a cache of the whole context, never shared or swapped.
- DeepSeek-V4's window cache takes half the GPU memory it did (its keys are its values: one buffer where two were made). Its indexer's pick is the same operation and kernel as GLM-5.3's.
- The kernels are the same ones in the same order (on the host emulator every moved kernel gives the old one's output bit for bit), and the GPU job's logits, user and tier checks run these models. Part of #135 and #141.
- Qwen3 dense runs on the description executor, the one that runs Gemma 4, Qwen3.5 and the other families (R3c): its layers are lowered from config.json to the shared operations (the attention kind, the gated SiLU MLP, the RMS norms), and the same lowering takes every Llama-style config the description builder reads (Llama, Mistral; Llama 3's scaled rope and Qwen2's attention biases are refused, named). The hand-written Qwen3 engine and what only it used (its KV cache tables, `kvtest`, `stress`, `recompute`, `rowcheck`, `batchcheck`, `kvcal`, `calcheck`, `score`, `info`) are removed.
- `glyd run` and `glyd serve` take Qwen3 at the kestrel, linnet and wren levels as well as penguin. A server takes up to 64 users at once (it took 8), each user's keys and values in pages; the 12-bit lossless pages are there with GLYD_KV=lossless, as for Gemma 4. Without it the keys and values are bf16 (lossless, 2 bytes a value where the old engine's cache took about 1.6), so `fit` plans less context for Qwen3 on the same GPU.
- A Qwen3 model packed by an earlier glyd for its own engine (glyd-v4 or glyd-v3 with `engine.safetensors`) is packed again in the cache on the first run (a glyd-v7 directory); a directory you packed yourself is refused with the words to pack it again. `glyd run Qwen/...` in a terminal chats without serving the page at a URL, as the other families do. Part of #139.
- Qwen3.8-Flash-Next (Condor's model, `qwen4_exp`) is described from its `config.json` like the other recent families: its gated residuals over four streams and their final mixer, the Qwen Sparse Attention indexer on its GQA layers, and its hashed n-gram embedding are layer kinds with parameters, with a CPU reference each. The description names every tensor the released checkpoint's language model has (the vision tower and the MTP layer left), and Condor's own reading of the config is the same model, number for number. On Condor's random test model the reference of the description gives the logits Condor's engine recorded on an RTX 4080 SUPER to a KL of 8.5e-6 on average (5.0e-5 at most) over 13 positions, the same best token at every position with a clear lead. The description lowers to the shared executor's operations (the streams operations take the gated residuals as a parameter, the gated norm a sigmoid gate, the indexer's pick unweighted mean-pooled keys, and a new n-gram embedding operation) for models whose experts and n-gram table fit the GPU, held to a CPU-reference logits fixture (`tests/logits/qwen4x.json.gz`). `glyd run` takes the released model on the executor too (at penguin, and its codes packs at the hummingbird level). Part of #140.
- DeepSeek-V4, GLM-5.3-Flash, Kimi-K3 and Muse Spark run the description executor's operations of their layer kinds for the residual streams (mHC), the depth mix, the gated residual, the RMS norms, the gated MLP activation and the mixture of experts: one operation a kind, the families differing only in the kinds' parameters, each keeping its own rounding points. Part of #135.
- `fit` predicts decode from the rate this computer's GPU reads its memory at, measured on the GPU's first use (256 MiB read 5 times, the link and the computer's memory likewise; kept in the cache) in place of a table of bandwidths, and shows the prompts' speed and a p10-p90 band. On an RTX 4080 SUPER (2026-11-23) the predictions were within 3% of what `glyd-engine bench` measured, in-sample (the constants were fitted on these same runs): gemma-4-12B 84.8 tokens/s predicted at wren against 86.2 measured and 62.3 at kestrel against 63.3, Qwen3.5-0.8B 568.7 at kestrel against 561.9 and 675.0 at wren against 657.9, prompts of about 1,000 tokens 3,571 tokens/s against 3,570 (gemma-4-12B) and 26,718 against 27,073 (Qwen3.5-0.8B). Every reply that runs alone is recorded with the plan that loaded it, and the runs on a GPU then set its predictions and their band.
- The prediction for a Qwen3.5 model no longer counts its draft head (the planner's own factor of 1.7), which no path drafts with yet; only Condor's path counts it.
- `fit --json` has `tokens_per_second_p10`, `tokens_per_second_p90` and `prefill_tokens_per_second`.
- Every path is sized by the planner alone (src/plan.rs): every loader takes its plan, and each of the 765 plans recorded before the change is unchanged. A context asked for that does not fit is refused with the longest that does (`Ask for less: --context N`) on every path; before, only one path said so.
- `glyd run qwen3.5` takes the size `glyd fit` predicts for the machine: 9b on a 16 GB RTX 4080 SUPER, 27b on 24 GB cards and larger.
- `glyd-engine plancheck` (development builds) checks the planner on the machine it runs on. It saves every prediction before it measures anything: decode tokens/s per user at 1 and 8 users, prompt tokens/s, time to first token, the GPU memory the plan takes, its context, level and placement. It then measures each case with `bench`, takes peak memory from the driver, and gates each case: speed within 15%, memory never above the plan, no failed request at 85% of the planned context. It also writes a calibration record per case (GPU, class, driver, measured rates, case, predicted, measured). A decode step is predicted per GPU class: a floor per layer (by its width) plus the weights' and the keys' and values' bytes, each at its own share of the measured memory rate. A prompt token is its products at a share of the tensor rate, plus a time per layer and its attention reads. A class with records (calibration/records.jsonl: the RTX 4080 SUPER's, the A100 SXM4 40GB's and the A10's so far) is fitted on its own records. A class without records takes the fit of the other classes with the same kind of memory (HBM or GDDR), with the error that leave-one-class-out scoring gives it. `plancheck --score` prints both errors per class. On a model's first load on a GPU, `run` and `serve` time one reply (a 512-token prompt and 33 tokens made) and keep it. From then on, `fit` says whether a prediction comes from runs measured on this GPU or from a class prior, and gives that prior's error. Part of #150.
- README: supported models are listed as model families with their compression levels, and the hardware table lists platforms and multi-GPU setups separately.
- Pack-format compatibility (R10): every saved format Glyd has written is held in `tests/data/packs` (16 small saves, 1.71 MB: glyd-v1 to glyd-v7, every layout and level, now with glyd-v2's experts, glyd-v5's packed table, glyd-v7 at penguin and the exact four-bit i4x), and each that holds a model is run on the CPU reference on every merge, its logits held to a digest recorded for it (`glyd-engine packcheck`); on the GPU job the same saves are decoded by the GPU library and held to the same digests (`packcheck --gpu`). A change that reads an old save to other weights or other logits fails there. `packcheck --engine` makes the formats the engine loads a downloaded or cached directory in (glyd-v7 at penguin, kestrel, linnet, wren and hummingbird) from random models of the families published in them, loads each from the saved directory and holds it to the bits of the same model packed as it loads. The GPU job now runs `packcheck --engine` on every merge as well; a format whose digest is not recorded fails it by name with the command that records one. Part of #231.
- Serving on the description executor's paged cache (Gemma 4, the Qwen3.5 layout, Muse Spark, Nemotron-H): a sequence sent back for room is now swapped to the computer's memory and copied back when there is room, rather than run again from its tokens, where its bytes over the link cost less than its recompute at the rate the prompt passes ran at (R4d). It goes on with the same cache bit for bit, its recurrent states too, so no token is run twice and a reply under preemption is the reply alone when prompts run one a pass. The copy back runs on a stream of its own beside the other sequences' steps. The swap area is pinned as needed, up to twice the GPU cache it backs and no more than the computer has free after the experts' memory and a 6 GiB margin; where it is full a sequence is run again from its tokens as before. `/stats` gives the sequences swapped (`swapped`) beside `preemptions`. Part of #328.
- The GPU library has a prompt attention that reads the lossless 12-bit KV pages where they are (`glyd_gpu_kv_prefill`, C API 19): a chunk's rows over its sequence's cached prefix in the pages and over the chunk's own bf16 rows, causal, for heads of 128, 256 and 512, any number of query heads a kv head, a sliding window, bf16 or fp32 scores, soft-capping and attention sinks. Nothing is unpacked to bf16 first. On an RTX 4080 SUPER, a chunk of 256 rows at gemma-4-12B's shapes after 1,024 / 4,096 / 8,192 tokens: its sliding layers' attention 0.18 ms where unpacking the pages and Gemma 4's kernel take 0.23 (bf16 pages: 0.19); its global layers' 0.47 / 1.65 / 3.26 ms against 0.51 / 1.72 / 3.35 (bf16 pages: 0.49 / 1.70 / 3.30). `kvshapes` holds it to attention on the host from the same values. Part of #332.
- `glyd-engine kindcheck` (development builds): every layer of the random models of Gemma 4, the Qwen3.5 layout, Nemotron-H, Muse Spark, Kimi-K3, GLM-5.3-Flash and DeepSeek-V4 run by the engine's CUDA kernels against the CPU f32 reference of its layer kinds, kind by kind (each body on the GPU's own inputs; the norms and residual rules between them), at the lossless level, in about 17 s on an RTX 4080 SUPER; the GPU job runs it. The worst differences there, over the reference's largest value: attention 5.4e-3, latent attention 1.1e-2, DeepSeek-V4's compressed attention 7.3e-3, the linear attentions and Mamba-2 up to 2.1e-2, MLPs 6.1e-3, mixtures of experts 9.9e-3, the norms and residuals 1.6e-2; a row whose expert or indexer choice was within 1e-2 of a tie and went the other way is named and not compared (3 of 3,640 rows).
- The CPU reference of the layer kinds picks the tokens a latent attention's indexer keeps (GLM-5.3-Flash's pooled keys: the best pools and the incomplete last pool), so a GLM-5.3-Flash layer past its indexer's budget is computed instead of refused; and the attention kind names its bf16 rounding points (transformers' eager scores and rope for Gemma 4, the SDPA form for Muse Spark, FlashAttention's probabilities for the Qwen3.5 layout and Nemotron-H), so the reference is held to what the engine computes: with its rope in bf16 arithmetic as transformers' eager form has it, on the same RTX 4080 SUPER Gemma 4's attention is within 3.1-4.1e-3 of the reference (6.9e-2 with the scores rounded alone), and Muse Spark's within 2.1-3.5e-3 with its own rounding points (1.8e-2 with FlashAttention's). Part of #135.
- One attention operation for every dense-attention model on the description executor (Gemma 4, Qwen3.5 and Qwen3.8 dense, Nemotron-H, Muse Spark): its head width, heads, norms, rotary, window, scale, soft-cap, output gate and rounding points come from the model's layer kinds, and one set of kernels (heads of 64, 128, 256 and 512) runs it.
- A prompt chunk of heads of 256 and 512 over 12-bit KV pages reads them where they are (`glyd_gpu_kv_prefill`) instead of unpacking them to bf16 first, and a decode row over bf16 caches takes the 12-bit pages' decode kernel (`glyd_gpu_kv_decode_bf16`, as prompts take `glyd_gpu_kv_prefill_bf16`; C API 20), in an instance of its own that reads a page's keys and values at once: the same kernels, tiles and sums on either cache, so a reply over 12-bit pages is the bf16 reply token for token (`glyd-engine kvshapes` holds a batch of sequences in one call to each alone and to the pages, bit for bit; `glyd-engine attndecode` times a call). Logits move in their last bits from before (sums in another order), within the reference's bounds.
- A Mamba-2 hybrid's attention layers (Nemotron-H's) keep half the GPU memory a token they did: their keys and values are held apart, where a second buffer of the same size was made and never read. Part of #135 and #141.
- The expert cache reports what each mixture-of-experts layer did in every step: experts asked for and found on the GPU, misses copied or computed on the CPU, the CPU's pairs, the bytes read from the computer's memory and the time spent waiting. `/stats` adds an `"experts"` list, one entry a layer, once a model with experts in the computer's memory has run.
- A step of many users reads each missed expert once: Kimi-K3's and DeepSeek-V4's FP4 experts and every family's hummingbird experts computed on the CPU are read once for all the rows routed to them, and DeepSeek-V4's release experts can now be computed on the CPU at all (before, every miss was copied to the GPU). On an RTX 4080 SUPER with random models at the release's widths larger than its memory (18 GB), the bytes read for a step's misses equal the missed experts' bytes at 1, 8 and 32 users, and the CPU's share takes 28-36% off a step at one user and 34-54% at eight.
- The choice between copying a missed expert and computing it on the CPU counts its rows (the first row of an expert costs its bytes, each further row only its arithmetic, both measured as the model runs), and a step whose misses outnumber the GPU's free places is copied in waves where that is quicker: DeepSeek-V4 at 32 users on the same RTX 4080 SUPER and random model, 286 ms a step against a median of 334 before. Part of #327.
- A model's `config.json` is read once into one typed config, which `fit`'s planner, `run`'s refusals, the description loaders, every family's own reader of the file, `costs` and the end-token lookups share: the families' names for one setting (`num_experts`, `num_local_experts`, `n_routed_experts`; a release's `quantization_config` at the top or in its text model) are read alike everywhere, and a shared key of the wrong type is refused with the key named (`config.json: num_hidden_layers = "eight": ...`).
- Tensor names come from one table per naming convention (Hugging Face's Llama-style names with Qwen's, Gemma's, Kimi-linear's and Muse's parts; DeepSeek's V3-style and release names; Nemotron-H's). A Hub-format directory whose checkpoint lacks tensors its `config.json` describes is refused at load with how many are missing, the first of them and the convention, where the load stopped at the first one before. Part of #136.
- Qwen3.5 and Qwen3.8 dense (the Qwen3.5 layout) can keep their full-attention layers' keys and values in 12-bit lossless pages too (`GLYD_KV=lossless`, opt-in), at 76.7% of bf16's bytes a token. On an RTX 4080 SUPER, Qwen3.5-0.8B's prompt pass read back 6.3 million values bit for bit, and its greedy reply matched for 8 of 32 tokens: the decode attention's sums are the library's, and a small model's near ties turn on them.
- Gemma 4 on 12-bit KV pages (`GLYD_KV=lossless`) at bf16's speed. On an RTX 4080 SUPER (gemma-4-12B at wren), one user's decode step at 1,024 tokens takes 12.55 ms against bf16 pages' 12.53, and at 8,192 tokens 13.06 ms against 13.72. Four users decode 294 tokens/s against 299, where they decoded 249. The pages' room for values outside their windows holds long prompts: at 8,192 tokens pages had been made void.
- When Gemma 4's 12-bit KV pages run out of room in their pools (`GLYD_KV=lossless`), the scheduler no longer ends the running replies with an error: it takes the step back for the requests that hit it and runs them again from their tokens, alone if hit again. Only a request too large for the pools even alone ends, with "the lossless KV cache cannot hold this request, even alone". On an RTX 4080 SUPER, four users with the pools made too small for all at once had 2 steps taken back, and every reply was token for token the one it gives alone.
- Gemma 4 can be served on 12-bit lossless KV pages: `GLYD_KV=lossless`, where the server has no draft model. The first run makes the model's KV tables (about 5 s on an RTX 4080 SUPER) and keeps them in Glyd's cache; the plan counts the pages' bytes, so a cache of 3.9 GiB in bf16 is 2.9 GiB. It is opt-in.
- `glyd run MODEL:LEVEL --yes --prompt TEXT` reads its prompt again: `--yes` took the next flag as its value, so the run found no prompt.
- A level's tag is the level asked for, `:penguin` too: a tagged model that does not fit goes to the degrade ladder, which asks, or with `GLYD_DEGRADE=no` refuses and names the steps. Before, `:penguin` was read as no tag, and the planner loaded a lower level without a word.
- The ladder takes a shorter context at a level before a lower level. On a 16 GB card, gemma-4-12B at kestrel with `--context 262144` now loads at kestrel with 131072 tokens (one step, no cost in speed), where it went to linnet.
- `fit` lists each level once: a local DeepSeek-V4 release showed its penguin row twice.
- The GPU library's lossless decode attention reads a sliding window's tokens alone (`glyd_gpu_kv_decode_from`, C API 18): pages before the window's first token are not read, and that page's earlier tokens count for nothing. `kvshapes` checks windows from inside a page, from a page's first token, the last token alone and none, at head dimensions 256 and 512, against attention on the host (on an RTX 4080 SUPER: every value within its bound).
- `fit` and `run` price the ladder's steps on the same shape of the model (its tensors' headers where they can be read): `run` asked or did not ask where `fit` said the opposite (a DeepSeek-V4's experts in the computer's memory: 34% slower in `fit`, 21% in `run`).
- A model the engine runs as a description, and holds paged, replied wrongly to a prompt given outside the scheduler (`reply`): its sequence's block tables never grew, so every token was written to one block. They grow as the scheduler's do.
- Gemma 4's keys and values can be held in 12-bit lossless pages on the description executor, from per-layer tables made on the model's own keys and values (`glyd-engine kv12cal`, development builds). On an RTX 4080 SUPER, gemma-4-12B at kestrel: the prompt's pass reads back the plain bf16 cache's values bit for bit (180 million values, none differ), at 75% of bf16's bytes a token. Decode attention over the pages rounds each score to bf16 as Gemma's kernel does. Serving takes them where `GLYD_KV=lossless` asks for them and the plan finds that they reserve no more than bf16.
- A request with a repetition, presence or frequency penalty, a logit bias, or tokens held back until `min_tokens` (and a top-k of 1 to 64, or greedy) no longer reads its whole logits row each token: the engine reads the changed tokens' own logits on the GPU and draws from those and the row's 64 largest, which gives the token the whole row gives (else the row is read as before). Qwen's generation defaults carry a repetition penalty. On an RTX 4080 SUPER, gemma-4-12B at wren with 8 such requests at once made 220.7 and 229.0 tokens/s in all where it made 215.5 and 222.7.
- The reference backend builds DeepSeek-V4's compressed attention through the layer-kind seam, as its model already ran it.
- A release on a driver older than 580 (which reports CUDA 12) says, in `run`'s and `serve`'s first lines and in `fit`, that it runs its CUDA 12 kernels and GPU library and why, and that they measured as fast as the CUDA 13 ones (gemma-4-12B at wren on an RTX 4080 SUPER, every load of the bench's gate suite within 0.3%); before, it changed to them without a word.
- `glyd serve MODEL --with MODEL2` serves two models from one process on one GPU, MODEL2 on `--with-port N` (else the next free port after MODEL's). Each has its own server, scheduler, requests, figures, failure and `/v1/glyd/info`. The first loads as it would alone and takes up to half of the memory left for its sequences; the second then loads in what is left. The status file says ready when both serve; a model that fails to load or to start its server ends the start of both, and a GPU error after which the GPU cannot be used ends the process. On an RTX 4080 SUPER, gemma-4-12B at wren with Qwen3.5-0.8B, in either order, serve from one process (9.5 GiB of the GPU); two gemma-4-12B do not fit and the start ends with the reason.
- A model on the description executor's single-GPU route writes the weights it holds into the plan's load report (`GLYD_PLAN_REPORT`), where it wrote none.
- GLM-5.3-Flash configs whose layers are all experts load (their two readers had disagreed over the dense MLP's slots).
- DeepSeek-V4 on the description executor is served through the scheduler: several users in one pass, where it served one request at a time (its window, compressed entries, indexer picks and attention output are each sequence's own in a pass over several). On an RTX 4080 SUPER with a random DeepSeek-V4, the scheduler's checks pass with 4 and 8 users, batched and one sequence a pass, and beside Gemma 4 in one process.
- `/v1/glyd/info` lists the engine's GPU memory by GPU under `memory.devices` (each GPU's bytes now, its peak and when); `engine_bytes` and `peak_bytes` stay every GPU's together.
- Nemotron-H on the description executor is served through the scheduler: several users in one pass, where it served one request at a time; its attention's keys and values in the paged pools; a conversation's start found by name, with its Mamba-2 states kept at boundaries (the shared starts that state snapshots gave Qwen3.5's layout). Requests to it may set `logprobs` and `min_tokens`, which were refused. On an RTX 4080 with a random Nemotron-H, the scheduler's checks pass with 4 and 8 users, paged and not; a 384-token start found by name and a second turn give the logits of a sequence run alone (0 of 4,096 differ).
- The cache's events, for a router that sends a request to the server holding its start: `GET /v1/glyd/kv/events?after=N&wait=S` gives each step's blocks stored by name (their names, the name of the block before them, their tokens) and removed, as numbered batches; the first says the cache is empty, the last 1,024 are kept, and a reader behind them is told so (`docs/serve-api.md`; the schema is part of `glyd-serve/1`). Off unless the operator sets `GLYD_ENGINE_KV_EVENTS=1`, as the events carry every caller's prompt tokens.
- Anthropic's usage reports `cache_creation_input_tokens` for a request with a `cache_control` marker: the prompt tokens it left that a request with the same start would find, less what it read (it was always 0). On an RTX 4080 with Qwen3.5-9B, a first request with a 3.4k-token cached system prompt is told 2,048 created, and the next with the same system prompt 2,048 read.
- Each served model's requests (`/v1/glyd/requests/ID`, cancel by id, the counts in `/v1/glyd/info`) and its traces' model name are its own: two models in one process no longer find or cancel each other's requests.
- Models of the Qwen3.5 / 3.8 layout (Gated DeltaNet layers beside paged attention) now find a conversation's start that any earlier request ran (Nemotron-H's Mamba-2 states are kept the same way): their states are kept every chunk of a prompt, in a pool of fixed size made at load (512 MB by default, `GLYD_ENGINE_STATE_SNAP_MB`; boundaries every `GLYD_ENGINE_STATE_EVERY` tokens, the chunk by default), under the name of the block ending there, and a request goes on from the newest such boundary whose attention blocks are held. On an RTX 4080 with Qwen3.5-9B, eight conversations with the same 4k start have their later turns' first token in 0.134 s where they took 1.141 s, and make 233.2 tokens/s in all where they made 107.7; one conversation's later turns 0.10-0.13 s where they took 0.93 s, since Qwen3.5's chat template writes an earlier reply back without the `<think>` block its generation prompt ended with. Loads with nothing in common are unchanged (`benchmarks/gpu/rtx4080-state-snapshots-2026-11-22`).
- The planner counts the second copy of a recurrent model's states kept at each prompt's end, and the pool, in the memory it plans.
- Nemotron 3 Nano's and Nemotron 3.5 Lightning's released configs load again: their `moe_shared_expert_overlap: true` (a schedule of the shared experts beside the exchange between GPUs, not arithmetic) was refused by the description builder, which the Nemotron-H executor loads from since the previous release's change; either value is now read.
- Kimi-K3's loader refuses a delta layer whose `A_log`, `dt_bias`, convolution or gated-norm tensor holds fewer values than the kernels read, or is of a type other than F32 or BF16: such a tensor would have been read past its end on the GPU. A longer one is read in its first values, as the release's `A_log` of 128 values for 96 heads is.
- An FP8 config without `weight_block_size` (scales a tensor or a channel) is refused by the description builder as GLM's loader refuses it, where it took blocks of 128.
- A side stream (the expert cache's copies) is no longer destroyed when its owner is dropped: it waits for its work and goes to the next side of its context, so a copy of its handle can never name a stream the driver reused. A hundred sides one after another make one stream (`glyd-engine bufcheck`).
- Every recent family the description executor runs (Gemma 4, Qwen3.5, Nemotron-H, Muse Spark, GLM-5.3-Flash, Kimi-K3, DeepSeek-V4) now loads from the description lowered from its layer kinds, which the description builder reads from `config.json`. The family's own reader of the config must describe the same model, operation for operation: where the two read a config differently, the model is not loaded, with the first difference named, so a kernel is never sized by one reading and fed by the other. Each lowering is checked on the released configs (whole and as a split's stages) and on the random models the GPU checks run; on an RTX 4080 every family's checks pass as before and the logits are the same.
- DeepSeek-V4 at the lossless level no longer keeps an unused copy of a layer's output-groups matrix on the GPU where that matrix was too small to be held in FP8.
- Development builds: `GLYD_ENGINE_STEP_PROF=1` prints, every 256 decode steps, the step call's time, the GPU's and the host's between steps. On an RTX 4080 with gemma-4-12B the host's time between steps is 0.3–0.4% of a decode step (`benchmarks/gpu/rtx4080-step-profile-2026-11-22`).
- The description builder reads Kimi-K3's and DeepSeek-V4-Flash's released `config.json` files, which it refused before (it had only described random models made from them): transformers' serialized generation defaults and a few training and processor keys are known to say nothing of the model, Kimi-K3's next-token-prediction layers and grouped top-k over one group are read, its routed experts as released (MXFP4 in groups of 32) are named with their scales, and DeepSeek-V4's DSpark drafter settings are read as the drafter's. Any other quantization is refused with what differs.
- Tests: the released configs of gemma-4-12B-it, Kimi-K3, DeepSeek-V4-Flash and Qwen3.8-27B describe their checkpoints' tensors exactly (from the Hub's safetensors headers), less the vision and audio towers and the next-token-prediction layers; every fixture config describes (Condor's too, since R3d); and a GGUF's description is its Hugging Face twin's.
- The GPU driver is called from the device layer alone (`src/dev.rs`); a test fails where another file calls it. A buffer belongs to the context it was made in: it is read, written, zeroed and freed there whichever context is current on the thread, so a model split over GPUs or a second model in the process frees its own memory from any thread.
- Device functions that could cause undefined behaviour from safe code are `unsafe` with their contracts (a raw device address, a graph replay, an event handle another thread uses), copies to and from device memory take plain number types only, and a stream or a buffer can no longer be forged from an arbitrary address.
- Kernel launches are type-checked: `build.rs` reads every kernel's signature in `kernels/*.cu` (280 kernels, those that C macros make included) and gives each a Rust type with its arguments' types, so a launch with an argument of the wrong type does not compile. Every launch in the engine compiled as it was: none had a wrong type.
- On an RTX 4080 with gemma-4-12B, decode and prefill speed are unchanged (within 0.2% on every figure of the gate suite; `benchmarks/gpu/rtx4080-device-seam-2026-11-22`).
- `glyd-engine cb2prefill` took its formats by the pack's letters alone and refused its own default; it takes the names its help gives.
- A prompt finds the start it shares with any earlier request, whether or not a sequence still holds it, on the models the description executor serves with paged keys and values (Gemma 4, Muse Spark): the executor's blocks are found by what they hold (a name for each full block of 16 tokens, of its tokens and all before them), and a block no sequence holds stays until its memory is needed, the oldest first. So a conversation's system prompt is shared with a new conversation after the first has grown far past it or has ended, a conversation whose slot was taken finds its own blocks at its next turn, and a sequence sent back to wait for memory does not run its prompt again. On an RTX 4080 with gemma-4-12B, a new conversation that begins with 4k tokens the server has run before has its first token in 0.05 s alone and 0.21 s with seven others, where it took 1.09 s and 3.44 s, and eight such conversations of 4 turns make 291 tokens/s in all where they made 245 (`benchmarks/gpu/rtx4080-blocks-by-name-2026-11-22`).
- A block's name is the first 16 bytes of SHA-256 over the name of the block before it and its 16 token ids (`kv::block_hash`).
- A request that begins as another one does shares that start instead of running it again, on the models the description executor serves with paged keys and values: it names the other sequence's blocks (whole blocks of 16 tokens; the other may be finished and kept, or still running) and runs only the rest, in a slot of its own, and neither loses anything. A request whose start another is still running waits for it, while at least a chunk of the start is left to run, and then shares it. On an RTX 4080 with gemma-4-12B, eight requests with the same 4k start arriving together at a server that has not seen it have their first token in 1.37 s where it took 5.0 s, and make 298 tokens/s in all where they made 95; and eight requests whose 4k start the server holds make 340 tokens/s where they made 213, since they hold the start once and all fit (`benchmarks/gpu/rtx4080-shared-starts-2026-11-22`). Gemma 4 and Muse Spark share any start, from a sequence that has grown no more than a chunk past it (their sliding layers give back the blocks before); the Qwen3.5 layout shares a prompt another request has run to its end. Requests share only within their cache scope (`cache_salt`). `GLYD_ENGINE_KEEP=0` turns it off.
- A kept sequence that a request begins like but cannot go on in (a cut its sliding layers no longer hold, a recurrent model with no state kept there) is left as it is while a slot is free, for its own conversation's next turn.
- `glyd-engine bench` has requests that begin alike (`--loads same4k+250ms:8`, `--suite shared`).
- Anthropic's `cache_control` has an effect: what a request with a marker leaves in the cache is held before other requests' for the marker's time (5 minutes, an hour with `ttl: "1h"`) when the server needs a place, and never keeps a request waiting. Any request can ask the same with `cache_pin_seconds`.
- A Qwen3.8-Flash-Next config whose n-gram embedding sizes make no embedding (an n-gram size of 0, sizes whose product does not fit) is refused in words; an n-gram size of 0 was a panic in a development build (found by the fuzzer).
- A request's usage can say how many of its prompt tokens the server found in its cache and did not run again: `prompt_tokens_details.cached_tokens` on the OpenAI chat and completions endpoints (where there are any), `input_tokens_details.cached_tokens` on the Responses API, and on the Anthropic messages endpoint `cache_read_input_tokens` with `input_tokens` the rest, as Anthropic's API counts them (in a stream, in `message_delta`). It is told to a request that has a `cache_salt`, and to every request where the server's operator sets `GLYD_ENGINE_CACHED_USAGE=1` (all callers one user); otherwise the count stays 0, as it was. The server's own count of the tokens found follows the same rule: `/stats` has `prefix_hit_tokens` and `prefix_hit_rate_percent` null and `/metrics` has no `glyd_prefix_cache_hit_tokens_total` unless the operator sets it (they were always there before; `glyd-engine bench` sets it for the server it starts).
- `cache_salt` (vLLM's request field, refused before) gives a request a cache scope: it finds in the cache only what requests with the same salt left, and what it leaves is found by them alone, on every endpoint. Callers who share a server and must not learn how each other's prompts begin each use their own.
- Mamba-2 has a CPU reference among the layer kinds, so Nemotron 3's model description is checked by its logits and not only by its tensors' names: on a random model the reference is within a KL of 1.1e-5 of the logits the engine recorded on an RTX 4080 SUPER, and the GPU job now holds the engine to a fixture the reference recorded (`tests/logits/nemotron-h.json.gz`).
- A conversation's next turn no longer runs from its first token on the models the description executor serves (Gemma 4, Qwen3.8 dense, Muse Spark, Kimi-K3, GLM-5.3-Flash). A finished request's keys, values and states stay in its slot, and a request that begins with its tokens goes on from them: from everything the slot holds where the turn sends the whole reply again; from wherever it leaves the earlier tokens on Gemma 4 and Muse Spark; from the prompt's end on the Qwen3.5 layout (a template that drops the thinking, a reply asked for again with more after the prompt). Kept slots never keep a request waiting: with no slot or no room free the oldest is emptied. On an RTX 4080 with gemma-4-12B, a conversation's later turns reach their first token in 0.08 s where they took 0.34 s (1k-token prompts) and in 0.09 s where they took 1.17 s (4k), and eight conversations at once make 47% more tokens/s in all (benchmarks/gpu/rtx4080-kept-sequences-2026-11-22). `/stats` and `glyd-engine bench`'s `prefix_hit_rate` count the prompt tokens found. `GLYD_ENGINE_KEEP=0` empties a finished request's slot at once, as before.
- `GLYD_ENGINE_BATCH_PREFILL=0` (every reply the reply alone, bit for bit) now holds past 16 sequences that decode together too: their rows run in passes of at most 16, the most at which the kernels give each row its own pass's bits.
- Nemotron 3 (`nemotron_h`) has a model description built from its `config.json` like the other recent families: Mamba-2, attention without a position embedding and the latent mixture of experts as layer kinds, and a weight map that names the released checkpoint's tensors exactly (tested against Nemotron 3 Super's tensor list, shapes included).
- A reply no longer stops while the prompt of a request that came in is run: on the models the description executor serves several sequences of (Gemma 4, Qwen3.8 dense, Muse Spark, Kimi-K3, GLM-5.3-Flash), a round of the scheduler is one step of the replies' next tokens and the prompts' chunks together. On an RTX 4080 with gemma-4-12B and eight requests coming in one after another, the longest gaps between a reply's tokens (the 99th percentile) go from 282 to 144 ms with 1k-token prompts and from 465 to 168 ms with 4k-token ones, with the prompts' first token and the tokens/s in all the same or slightly better (benchmarks/gpu/rtx4080-one-step-round-2026-11-22). What it costs: a token made in a step of more than 16 rows is computed by another kernel than the same token alone, so a reply under load can differ from the same request alone in its last bits and, where two tokens are nearly tied, in a token (measured on gemma-4-12B: a KL of 0.002 to 0.005 from the reply alone, the rounding of bf16 values; a decode step of more than 16 sequences already did this). `GLYD_ENGINE_ONE_STEP=0` gives the earlier round; with `GLYD_ENGINE_BATCH_PREFILL=0` as well every reply is the reply alone, bit for bit, while 16 sequences or fewer decode.
- `glyd-engine roundcheck` measures that: a sequence's logits by what its step holds beside it, against its row alone. In the GPU job it requires that what the other sequences hold never changes a row and that a step of 16 rows or fewer gives every row its own pass's bits.
- A model that had replied before it was made a server's step kept that reply's tokens in its first sequence, and the scheduler's first request then failed ("a chunk at 0 of sequence 0, which holds N"); the step's sequences now begin empty. Found by the two-models check on Muse Spark and GLM-5.3-Flash.
- `glyd-engine bench` has loads whose requests begin one after another (`--loads 4k+250ms:8`, `--suite arrive`), which is where a server's handling of prompts that come in under load shows.
- `glyd-engine bench` has conversations too (`--loads 1k*4:8`, `--suite chat`): every later turn resends the conversation, and the result has the later turns' first token (`ttft_later_s`) and the share of prompt tokens the server found in its cache (`prefix_hit_rate`).
- A request that fails because the engine failed is answered 500 (`server_error`) on the OpenAI endpoints, where it was answered 400 as if the request were at fault; a request's own mistake (a prompt the cache does not hold) is answered 400 on the Anthropic, Ollama and Responses endpoints, where it was 500.
- A server whose GPU gives an error after which its context cannot be used again (an illegal address in a kernel, a launch that failed, an uncorrectable memory error, the GPU gone) tells its requests, then ends with status 4; the service starts the model again at the next request for it. Before, it went on and failed every request that came.
- Errors of the driver and of the GPU library carry their status and their kind inside the engine (out of memory, no driver, a lost context), and `run` no longer tells them apart by their words.
- Inside the engine, what a served model counts and keeps (its scheduler's figures, its metrics, its engine's first failure, what `/v1/glyd/info` says of it) belongs to a runtime context of its own and no longer to the process, so a process can hold two models: each one's `/stats`, `/metrics` and `/v1/glyd/info` are its own. `glyd serve` serves one model a process, two with `--with`.
- When a model does not load at the level asked for, the degrade steps are fixed, printed and priced: a shorter context (to the floor), experts in the computer's memory, the last layers' dense matrices there too (one GPU, Kimi-K3 and Muse Spark), then a lower level, then a refusal (the KV cache has one type in this engine, and `fit` says so). `glyd fit` prints the ladder with each step's cost in speed and in `--json` (`degrade`). `glyd run` says the steps in its first lines and asks before a step that costs over 30% of the speed or lowers the level; `--yes` or `GLYD_DEGRADE=yes` takes them, `GLYD_DEGRADE=no` refuses, and `serve`, the service and a run with no terminal never stop for an answer: without `--yes` they refuse (status 3, as before) naming the steps.
- One probe of the memory a process can still take (`offload::host_memory`): it reads the nearest cgroup limit from the process's own cgroup up, so a systemd unit's MemoryMax (a CI runner's) counts; `fit`, `run` and `glyd doctor` use it, and Condor's server lost its copy. Under a 3 GB limit `fit` plans with 3.0 GiB, where it took the machine's 55.9.
- Nemotron-H now runs only on the executor that reads its description: the hand-written engine and the `GLYD_NH_DYNAMIC` switch are gone, and the `nh*` developer commands run on the executor.
- Anthropic API: a reply to a request for thinking begins with its thinking block, whole and streamed, empty where the model wrote no thinking; before, such a reply could have no content block at all.
- Qwen3.5/3.6/3.8 dense run only on the description executor: the hand-written engine (`model35.rs`) and its `GLYD_Q35_DYNAMIC=0` switch are removed (R3b, #137), after the parity check found the two paths' logits the same bits at penguin, kestrel, linnet and wren. Its kernel table and packed weights stay, as `kern35.rs`; the development commands `generate`, `check`, `bench` and `sampletime` on a Qwen3.5 model run the description executor, and `dump35` (the old engine's layer-by-layer dump) is gone.
- `fit` and `preflight` size a model from its checkpoint's file headers before anything is downloaded: the shards' safetensors headers by two range requests each (kept in the cache by repo and commit), a GGUF's tensor table, or a directory's (`src/sizes.rs`). Tensors are classified by name (embedding, head, attention, MLP, routed and shared experts, vision, multi-token prediction); the planner's shape counts each part from its own tensors (a vision tower left out, the prediction head only where the model drafts with it) in place of config.json's estimate, and the dynamic families (Kimi-K3, GLM-5.3-Flash, DeepSeek-V4, Muse Spark) are planned over their GPUs, Kimi's expert cache included, before their files are here, to the same plan their files give.
- A served model whose keys and values are paged (Gemma 4, Qwen3.8 dense, Muse Spark) never fails an admitted request for lack of memory: the context a request may use is the smaller of `--ctx` and what the GPU's pages hold for one sequence alone (`/v1/glyd/info`'s `limits.context_tokens` says which), a reply that reaches it ends with `length`, and a sequence sent back for room always runs again.
- Short names pick the size for the machine: `glyd run gemma4` (also `serve`, `fit` and `pull`) takes the largest size of the family whose plan, the one `glyd fit` shows, reaches 10 tokens a second and 8,192 tokens of context on this computer, and says which and why in one line; `gemma4:31b` or `gemma4:31b:wren` is taken as it is, `gemma4:wren` asks the level of every size, and where no size reaches the floors the fastest that fits is taken (or the refusal says why for each). Tested for six preset machines against `fit`'s own prediction (`tests/names_by_plan.rs`).
- DeepSeek-V4, GLM-5.3-Flash and Muse Spark run at penguin with their experts in the computer's memory where their GPUs do not hold them, a cache of the busiest on the GPU (R12): DeepSeek-V4's FP4 experts each in a place of their own, GLM-5.3-Flash's FP8 experts as released each packed on its own (its places at the largest expert's bytes, each place's heads pointed at its own bytes on the GPU after a copy), Muse Spark's lossless experts cut from their stack's one pack (each place its own block entries, multiplied by the GPU library's new `glyd_gpu_mma_moe_placed`; C API version 17), the misses copied in. The expert cache is now a property of the MoE layer of the dynamic route (`MoeLayer::with_cache`). Checked on an RTX 4080 with random test models (the GPU job repeats it): with the experts in the cache every logit is the same bits as with them all on the GPU for DeepSeek-V4, GLM-5.3-Flash at a block level and Kimi-K3; for GLM-5.3-Flash's FP8 experts and Muse Spark's, the same bits on a short prompt and within 0.05 of a logit on a longer one (the cache's calls are smaller than the resident pass's, and the GPU library adds a product's sums in an order that turns on the call's size).
- Where even a token's experts do not fit beside a model's other weights on one GPU, `run` keeps the last layers' dense matrices in the computer's memory as well (Muse Spark's attention and latent projections, Kimi-K3's delta attention), copying each into one GPU slot just before it runs. Each such layer costs its bytes over the link every token, and the plan and the run's first lines say so. Checked on an RTX 4080 with random Muse Spark and Kimi-K3 models: the same bits with two layers held this way.
- The hummingbird level (routed experts at 3 and 2 bits by use) for Kimi-K3, DeepSeek-V4, GLM-5.3-Flash and Muse Spark, with no Python step: `glyd-engine calibrate-experts MODEL usage.json` counts each expert's use over the calibration corpus on the engine, `glyd-engine pack MODEL DIR --layout hummingbird --usage usage.json` codes the experts in Rust (a blocked Hadamard rotation for any width, a codebook a row) and copies the rest of the release, and `run`, `serve`, `preflight` and `fit` take the directory (its experts on the GPUs or, where they do not hold them, in the expert cache). `fit` plans GLM-5.3-Flash at 115.7 GiB, DeepSeek-V4-Flash at 106.2 GiB and DeepSeek-V4-Pro at 570.1 GiB this way.
- Hummingbird directories of Kimi-K3, DeepSeek-V4, GLM-5.3-Flash and Muse Spark run with their experts in the computer's memory, a cache of the busiest on the GPU and the CPU computing the rest, and with the experts the memory does not hold read from the disk as the router asks for them. `fit` plans GLM-5.3-Flash at hummingbird on a 16 GB or 24 GB card this way and DeepSeek-V4-Flash on 24 GB; Kimi-K3's weights besides its experts (about 28 GB lossless) need a 48 GB card. Checked on an RTX 4080 with a random DeepSeek-V4: the cache gives the same bits as every expert on the GPU where each expert a call asks for has a place there; the pairs the CPU computes differ from the GPU's in their rounding (not yet measured on a released model).
- Gemma 4 runs only on the description executor: the hand-written engine (`model_gemma4.rs`'s `EngineG4`, its drafter) and its `GLYD_G4_DYNAMIC=0` switch are removed (R3b, #137), after the parity check found the two paths' logits the same bits. Gemma 4's config, packed weights and kernel table stay, as `gemma4.rs`; the development commands `g4check`, `g4score`, `g4prefill` and `g4spec` drive the description executor (drafting through its draft model), and the fp32-residual kernels of `GLYD_G4_PRECISE` are gone.

## v0.34.2 — 2026-11-22

- `glyd-gpu` is on PyPI again at this version: v0.34.1's wheels were not published because their install check failed. v0.34.1's refusal of a model that fits nowhere also answered for the machine (no GPU, a driver too old) in the planner's words before `run` could say it in its own; a machine's problem is `run`'s to say again ("no NVIDIA driver was found on this computer"), and the early refusal is for a model that does not fit.

## v0.34.1 — 2026-11-22

- `glyd run`, `serve` and `preflight` without a level load the level `glyd fit` plans for Qwen3.5 to 3.8 too (Qwen3.8-27B on a 24 GB card is linnet; it was refused as penguin, a level that family does not run at), and refuse a model `fit` fits nowhere with fit's reason before anything is downloaded.
- Windows: the free space of a disk is read (`GetDiskFreeSpaceExW`), so a download larger than the disk is refused before it starts, as on Linux; before, the M1 gate's 250 GB machine downloaded GLM-5.3-Flash until its disk was full.
- The M1 gate on Windows (AWS A10G, Windows Server 2022): install in 7 s, Gemma 4 12B at penguin and Nemotron 3.5 Lightning at wren load the plan `fit` printed, valid JSON, no Python (benchmarks/gates/m1-2026-11-21/windows). `m1-gate.ps1` works under Windows PowerShell 5.1 (argument quoting, native stderr, a case-insensitive name).
- Release builds: the GPU library and the engine's compiled kernels are taken from the Actions cache where their sources did not change (keyed by them; a tag builds everything fresh), the engine's nvcc compiles two architectures at once (`GLYD_ENGINE_NVCC_THREADS`), and the Windows build is three jobs side by side (the two DLLs and `glyd.exe`, then the check and the zip) with only the compiler and runtime of each CUDA toolkit installed. The Windows check no longer fails on `glyd doctor`'s exit code on a runner with no GPU, and a release whose Windows build failed still gets its Linux engine bundles attached (v0.34.0's were attached by hand). The install check waits for the release's files.
- Windows x86_64 (preview): the release has `glyd-engine-vX.Y.Z-windows-x86_64.zip` (`glyd.exe` with the engine linked, the CUDA 13 and CUDA 12 GPU library DLLs) and its sha256, which `irm https://getglyd.com/install.ps1 | iex` installs. v0.34.0 has none: its build passed but its check failed on `glyd doctor`'s exit code on a runner with no GPU.

## v0.34.0 — 2026-11-21

- The engine runs Gemma 4 on Blackwell consumer GPUs (RTX 50 series): `g4_add_norm` needed more registers than a 1,024-thread block has on sm_120 ("too many resources requested for launch"); kernels.yml now fails on any kernel past 64 registers without launch bounds.
- With the Rust install alone (no vLLM), `glyd run` uses the engine on every GPU it supports, not only the measured ones; the refusals' "full glyd" line installs with `GLYD_WITH_VLLM=1`.
- `glyd run`, `glyd serve` and `glyd preflight` without a level tag now load the level `glyd fit` plans for the machine. Before, a Gemma 4 model or a model on the dynamic route took the lossless level, so Gemma 4 12B on a 16 GB card, or Nemotron 3.5 Lightning on an 8 GB card, was refused although `fit` showed a plan that fits.
- The aarch64 release build's CUDA 12 kernels leave out Jetson Thor (sm_110), which only CUDA 13 targets and which no driver reporting CUDA 12 runs. Before this, the aarch64 release build stopped because CUDA 12 had no sm_110.
- Forced tool calls in the XML, GLM, DSML and Atem formats limit a parameter given only an `enum` (or a `const`) to one of its values as text. The reader takes such a parameter as text, and before this a model could write the value in JSON quotes, which the reader kept as part of the value.
- Server: the connections are tasks of an async front end (tokio) and no longer a thread each, with no cap on how many (it was 64): a thousand streaming connections at once are answered, on a scheduler over a CPU fake model, with no error (a test). Every bound and time of the request's reading and every check of the server (Host and Origin, the key, drain) stand; `Connection: keep-alive` is kept for an answer with a length; a client that disconnects ends its sequence; a slow reader holds back its answer's thread and not memory. Every API front end now asks the engine through one request-level interface (`engine_api.rs`: submit, stream, cancel, stats). `/v1/glyd/info` says `max_connections: null`. A request with a repeated or non-numeric Content-Length, or a Transfer-Encoding, is refused 400 (never read as one thing here and another by a proxy); what a client sends while it is answered is held to 64 KiB; request bodies share a 1 GiB budget; 50,000 connections and 8,192 requests in flight are the bounds past which the next is answered 503.
- The engine no longer refuses aarch64 Linux (GH200, GB200, DGX Spark, Jetson Thor, Arm workstations): an Arm computer is planned as an x86_64 one and gated only by its GPU (compute capability, driver, and the measured-GPU list, which names no Arm-hosted GPU until one is measured). Kimi-K3's experts computed on the host (MXFP4) have a NEON kernel; every other expert format already had one, and all are checked against the scalar kernel on the aarch64 CI runner.
- The user guide has an API reference and a page on the background service, both built from `docs/serve-api.md` and `docs/service.md`; a page on the M1 gate that names every step of `benchmarks/gates/m1-gate.sh` (a check fails the build if one is missing); and a copy of the guide for each release at `/guide/vX.Y.Z/`.
- The model manifest (`glyd-engine/models.json`) says for every model its license, whether the Hub gates it, and the record in `benchmarks/` or `research/` where the engine ran it (`null` where none); a test fails if an entry lacks one. The guide's catalog page has the process.
- Windows: `glyd doctor` says when a GPU runs under WDDM (the display driver model: an allocation past the card's memory is given system memory without an error, and the run is many times slower than the plan says) and the setting that stops it (NVIDIA Control Panel, CUDA - Sysmem Fallback Policy: Prefer No Sysmem Fallback); plans on such a GPU keep 1 GiB more free (`plan::WDDM_EXTRA`, a first figure to be set from a measurement on a Windows GPU). Unit-tested; not run on Windows.
- `glyd-engine supports MODEL[:LEVEL]` answers from the same table as `fit` and `run`: every family the engine runs (Qwen3, Qwen3.5 to 3.8, Gemma 4, GLM, Kimi, DeepSeek, Muse, Nemotron), at a level where one is given (`--lossy` is `:kestrel`). A level the family does not run is refused with the levels it does (`Qwen3.5-9B (Qwen3.5) runs at the kestrel, linnet and wren levels in this engine, not penguin`); it used to refuse every model that is not Qwen3, and took `MODEL:LEVEL` as a directory name.
- The GPU job runs the client matrix against a random model made in the job (`logitsmodel qwen35 DIR --servable`: Qwen3.8's layout with a character-level tokenizer and its chat template, `tests/clients/tiny_check.py`): the OpenAI SDK (chat, streams, JSON schema, forced tool calls, Responses), the Anthropic SDK, the `ollama` library and LangChain, checking what holds whatever the weights write.
- `preflight` and `fetch` read the capability table (`src/support.rs`) first, as `run` and `serve` do, and every path takes its context limit and default from it: a kind or level the engine does not run, and a context past what the family or the model has, are refused in the same words by `fit`, `run`, `serve`, `preflight` and `fetch`. A test runs every machine preset against every family's fixture at every level and requires `fit` and `run` to give the same refusal or the same plan.
- Releases are signed. Each tarball and wheel carries GitHub build provenance and an SBOM attestation, and each tarball and the `SHA256SUMS` lists a keyless Sigstore signature, made by the release workflows; `docs/verify-releases.md` says how to check them. From release 0.34.0, `install.sh` checks the signature of the tarballs it installs and refuses one that is unsigned or altered (`GLYD_SKIP_VERIFY=1` installs without the check).

- Traces: OpenTelemetry spans per request (`glyd.request` with `glyd.queue`, `glyd.prefill` and `glyd.decode`), sent to an OTLP collector over HTTP/JSON where `OTEL_EXPORTER_OTLP_ENDPOINT` is set, off otherwise; a dead collector never slows a request. Tested against a local collector in `otel.rs` and `sched.rs` and, on a GPU, in the `gpu` job (`tests/ops_check.py`).
- Ollama's API names a GGUF by its file: `/api/tags`, `/api/ps` and `/api/show` list a GGUF served as a checkpoint as its model's manifest name with its quantization (`qwen3:8b-bf16`) or its file's name (`my-model:f16`), `details.format` is `gguf`, and pull, show and delete take those names. `/api/embed` stays 501 until embeddings are served (F2b).
- `fit` sizes Kimi-K3 and DeepSeek-V4 by their experts' own bits as released (MXFP4, FP4: 4.25) apart from the rest, and reads Kimi-K3's names for its experts' width and choices a token: Kimi-K3 is 2,744.5B parameters and 1,347 GiB of experts at penguin (it said 5,462.6B and 7,632.6 GiB). The hummingbird level is not offered for Nemotron 3.5 Lightning, whose experts read its width of 2,688 (the kernels take Nemotron 3 Super's latent of 1,024): it said 11.9 GiB from a pre-packed repo that does not exist.
- The wren level (about 4.5 bits a weight) runs in the release build, for every family that lists it (Qwen3.5/3.6/3.8 dense, Nemotron-H including Nemotron 3.5 Lightning and Nemotron 3 Super's experts, Gemma 4), where before a release ran it on quantization-aware checkpoints alone (`:wren is not built yet`). Its four-bit kernels were in the release's GPU code already, so the binary and the wheels are the same size; the planner offers wren where a model fits no higher level (an 8 GB card, a 24 GB card for a 32B model), and `glyd run`'s Python handoff passes `:wren` to the engine. On a quantization-aware checkpoint wren is still exact.
- The plan that loaded is remembered by model, level and GPU and the next plan is calibrated by it: what a load took of the GPU beyond its plan (at most 1 GiB) is planned as already used, by `fit`, `run`, `serve` and `preflight` alike on the one-GPU paths; `fit` says what the model took here before. Condor's (`qwenserve`) memory terms are the plan's constants (`plan::CONDOR_*`) and its load is one `LoadPlan`, said in its log, remembered and reported as the other paths' are (`GLYD_PLAN_REPORT`).
- Windows x86_64: the release workflow builds `glyd.exe` (the engine linked in) and the GPU library DLLs of CUDA 13 and CUDA 12 (`glyd-engine-vX.Y.Z-windows-x86_64.zip` and its `.sha256`, attached to the release when the build passes). It is built on a runner with no GPU: nothing of it has run on a Windows GPU yet. GPU discovery gives up after 30 s on Windows where the driver does not answer (`GLYD_PROBE_TIMEOUT=SECONDS`, 0 waits).
- The engine runs on drivers from CUDA 12.2 (R535; Blackwell GPUs from CUDA 12.8, R570), where it needed R580. The release builds the engine's kernels twice, with CUDA 13 and with CUDA 12.9 (not compressed, which a driver below CUDA 12.4 needs), and loads the one the driver's version asks for; the engine bundle holds the CUDA 12 GPU library (not compressed) beside the CUDA 13 one, and `install.sh` and `condor/tools/install.sh` take it. A GPU the build has no machine code for is told so ("this build has no code for <GPU>: it carries sm_80 ..."), not asked to update the driver; the release build also carries the compute_89 PTX, which a driver compiles for a GPU newer than the machine code. `GLYD_ENGINE_CUDA=12` runs the CUDA 12 kernels on a newer driver. Checked on clean rented machines with only the driver: R535 (RTX 3060), R565 (RTX 4060 Ti), R580 (RTX A4000) and R595 (RTX 3060 Ti; RTX 5060 Ti, Blackwell), benchmarks/gates/m1-2026-11-21.

- Operations: `GET /health` (up), `GET /ready` (model loaded and taking requests; 503 while loading or draining) and `GET /metrics` (Prometheus: requests by API, refusals by status, prompt and generated tokens, histograms of the first token and of the time between tokens, queue depth, sequences running, KV cache pages, GPU memory, in flight) on `glyd serve`, the lossy levels' servers and `glyd service`; SIGTERM drains (no new requests, the ones in flight finish, exit 0, or 1 at `--drain-timeout SECS`, default 30). Tested in `ops.rs`, `server.rs`, `sched.rs` and `tests/drain.rs` (the service, SIGTERM); the `gpu` job scrapes `/metrics` after real requests and drains a served model with a reply streaming (`tests/ops_check.py`).
- Secure defaults for every listener (`glyd serve`, `glyd run`, `glyd service`, `--ollama-port`): the server listens on 127.0.0.1; an address that is not this computer's own does not start without an API key unless `--insecure-no-auth` says so (the refusal comes before the model loads; the Python server's `glyd serve --host` too); the Host and Origin checks stay on for a loopback `--host`; CORS is off, with `--allowed-origins` answering exactly the pages listed (never `*`) and `--allowed-hosts` adding names; the Host check answers more container names (Docker Desktop's gateway and Kubernetes names, Lima, Minikube); a request that is slow as a whole is answered 408 (30 s for the head, 120 s for the body) and a write to a client that reads nothing gives up after 60 s; image URLs are fetched from the address that was checked, not from a second lookup (DNS rebinding). Tested in `server.rs`, `cli.rs`, `vision.rs`.
- `glyd doctor` checks the driver, CUDA, the GPUs, each backend (detected, version, accepted or rejected with the reason), the computer's memory and the memory terms a plan is made of, the disks, the model caches, the Hub and the service's and a server's ports, and says the plan `run` would load for Gemma 4 12B here; every failure has a one-line fix, and a table of failure classes (`doctor::MATRIX`) is tested against the checks. `glyd doctor --json` prints, and `--bundle [FILE]` writes, one JSON bundle (`glyd-doctor/1`) with the versions, the checks, the environment, the service's configuration and its recent logs, tokens, keys and the home folder taken out; the engine bug form asks for it.
- The plan `glyd-engine fit` prints is what `run`, `serve` and `preflight` load, for Qwen3 dense, Qwen3.5/3.6/3.8, Nemotron-H, Gemma 4 and the dynamic route (Kimi-K3, GLM-5.3-Flash, DeepSeek-V4, Muse Spark: the layers cut over the GPUs, every layer across them, Kimi's expert cache) once the model's files are here: one plan type (`plan::LoadPlan`) and one reserve (768 MiB) with each engine's measured work as its one constant, in `src/plan.rs`. `fit` sizes a model with the functions `run` does and refuses where `run` refuses, in `run`'s words; `run` says its plan before it loads. A server's users are in the plan (the Qwen3 path sized the cache for one). `fit --json` is `glyd-plan/1`. The GPU check loads a model and compares the memory it took with the plan (`GLYD_PLAN_REPORT`, `scripts/ci_plan_check.py`).
- Tool calls and reasoning per family are tested token by token (`src/families.rs`): Qwen3.8 (27B and Flash-Next, which ship one template), GLM-5.3-Flash, Kimi-K3, DeepSeek-V4 and Muse as well as Gemma 4, each from its own template's writing of a call and a reasoning, through the server's reader, reasoning split and tool parser, with the deltas a client sees (reasoning as it comes, a call's name and its arguments in pieces). The GPU job serves `google/gemma-4-12B-it` at kestrel and runs the OpenAI, Anthropic, Ollama and Open Responses client checks against it (`tests/clients`).
- Muse's ATEM calls (`<atem:function_calls>`, in a message to the tool) and its thinking (a message to `self`) are read: tool calls through OpenAI's and Anthropic's APIs, whole and streamed, and the reasoning apart from the answer (`reader::Muse`, `tools::Format::Atem`). Every family's tool call and reasoning is tested from its real chat template (`src/families.rs`: Qwen3.8, GLM-5.3-Flash, Gemma 4, Muse), and each family's call goes through both APIs in `server.rs`'s tests; the families and their formats are in docs/serve-api.md.
- Gemma 4: tool calls and thinking work. Its replies are read by a reader of their own (`reader::Gemma`, behind the `Reader` interface Kimi-K3's tags use too): the thought channel is the reasoning, a call written `<|tool_call>call:NAME{key:<|"|>text<|"|>,...}<tool_call|>` is an OpenAI tool call (arguments streamed as JSON as they come) and an Anthropic `tool_use`, and a function or a call asked for through `tool_choice` is opened in the prompt. Before, a request with tools was refused and the thought leaked into the answer.
- One command table (src/cmdtable.rs) makes `glyd --help`'s command list, `glyd help COMMAND`, `COMMAND --help`, the man page (`glyd man`; docs/glyd.1, which a test holds equal) and the refusal of `--json` or `--machine` on a command that does not take it (the refusal names those that do). `--json` is now on `ls`, `ps`, `service status`, `fit`, `doctor`, `supports`, `preflight` and `bench`; `--machine` on `fit`, `doctor` and `bench`. `glyd bench` is a command of `glyd`. `cargo install --git ... glyd --features engine` builds the one program.
- Model names from the release: `glyd names` lists them (`--json` too), `glyd names update` fetches the latest release's `models.json` (checked against the sha256 the release lists, kept only where its date is later than the program's), and a name that is not known is looked up there once a day (`GLYD_NO_REFRESH=1` stops that), so a model released after the program was built can be named. release.yml attaches `models.json` and its sha256 to each release. `deepseek` is DeepSeek-V4-Flash-0731 (the V4.1 repos are another architecture the engine does not run).
- The service holds models by GPU memory where the driver can be read (nvidia-smi): each model's footprint is measured when it loads, and a load first unloads the least recently used idle models until it fits (`--max-loaded` is a ceiling, 16, and the count, 2, where memory cannot be read); a request waits up to 30 s for a place before a 503; `glyd ps` shows each model's GPU memory. On Windows `glyd service install` makes a scheduled task of the user's, at logon, with the user's own rights; on a Mac the launchd agent was run for real (installed, answered, restarted by launchd after `kill -9`, removed).

## v0.33.1 — 2026-11-20

- The `glyd-gpu` wheels are under PyPI's 100 MB limit again (v0.33.0's were 181 MB each, and PyPI refused them, so `glyd-gpu` 0.33.0 is not on PyPI and `glyd[gpu]==0.33.0` does not install). The release GPU library is built with `GLYD_RELEASE` again, as the releases through v0.32.1 were: the default plans' kernels alone and no tuning variables, about a fifth of the size. The engine bundle's library shrinks the same way. v0.33.0's GitHub release, its `glyd` wheels and its engine bundles are as they were; use v0.33.1 for `glyd-gpu`.

## v0.33.0 — 2026-11-20

- `glyd connect [TOOL]` finds the server that runs here and prints what OpenAI and Anthropic SDKs, Claude Code, Ollama's libraries, Open WebUI (container) and Continue need to talk to it; the Host check answers `host.docker.internal` and `host.containers.internal` (Docker Desktop's and Podman's names for this computer); the client checks are in `tests/clients/` (docs/serve-api.md, Connecting tools).
- Server: Ollama's API under `/api` for the programs that talk to Ollama (Open WebUI, Continue, LangChain, the `ollama` libraries): `chat` and `generate` as JSON lines or one object (`options`, `format` with a JSON schema, `think`, tools, images, `suffix`), `tags`, `show` (with `capabilities` and `context_length`), `ps`, `version`, `pull` and `delete` for the one model a server holds; `--ollama-port N` listens on Ollama's port too where it is free. The key guards `/api`. Checked with the `ollama` Python library on Qwen3-8B.
- The engine's allow-list (`GPUS`) is per (GPU, family), and a measurement names the model it was made on and when that model was released (six months at most): the RTX 4080 SUPER is listed for the Qwen3 dense models; Gemma 4 12B, measured on it at kestrel and linnet against vLLM's 4-bit checkpoint, serves one request at a time and loses at 4 and 8 users (and by 7% to 21% on first token at one), so it is vLLM's by default there (`benchmarks/gpu/rtx4080s-gemma4-12b-2026-11-20`; docs/adding-a-gpu.md).
- Server: the Messages API (`/v1/messages`) is its own module and complete for the Anthropic SDKs and Claude Code: thinking blocks (`thinking_delta`, `signature_delta`), `stop_sequence`, `x-api-key`, Anthropic's error form, `max_tokens` required, tools Anthropic runs left out, `ping` after `message_start`; the request goes through the chat path, so its grammars and refusals apply. Checked with the `anthropic` SDK and with Claude Code 2.1 (Read and Write tools) on Qwen3-8B.
- Server: `POST /v1/responses` (OpenAI's Responses API, stateless: input items, instructions, function tools, `text.format`, every stream event), `suffix` on `/v1/completions` for fill in the middle (a model with Qwen's, StarCoder's or DeepSeek-Coder's FIM tokens), `GET /v1/models/ID`, and every error in OpenAI's form (`error.type`, `param`, `code`; 401 says `invalid_api_key`, a prompt past the context `context_length_exceeded`); a stream that asks for its usage has `usage: null` in each chunk.
- Engine: after a short prompt pass for requests that came in together, the round's decode step waits for the rest of them (first token with 4 short prompts at once 80 to 51 ms on an RTX 4080 SUPER, tokens unchanged), and the prompts of up to 8 requests that arrive together run in one pass (8 short prompts at once 162 to 65 ms on an RTX 4080 SUPER; a reply's last bits can then differ from the same prompt's alone, `GLYD_ENGINE_BATCH_PREFILL=0` keeps every reply identical to solo; `benchmarks/gpu/rtx4080s-batched-prefill-2026-11-20`).
- **Glyd is open source, all of it, under the Apache License 2.0.** The GPU library and its kernels (`gpu/`), the packer (`glyd-gpu/`),
  the Python GPU package (`glyd-gpu-py/`) and Glyd's engine (`glyd-engine/`), released compiled only from v0.27, are in this repository with
  their source, beside the codec and the store. Condor (`condor/`) and Starlings (`starlings/`, one model across GPUs and machines) moved in
  too. Releases v0.29.0 to v0.32.1 shipped under the Business Source License 1.1; their code is Apache-2.0 here.
- `glyd-engine fit MODEL[:LEVEL]` prints what a run would do before anything is downloaded: the level (lossless first, then the
  highest lossy level that fits, then a mixture of experts' experts in RAM), where the weights and experts go (one GPU, split, or
  experts in RAM with a GPU cache), the context, and predicted tokens/s, with every other level's fit. `--machine NAME|FILE` plans
  for another computer (presets: rtx4060-laptop, rtx4080s, rtx4090, 2x3090, l4, a6000, a100-80g, h100, 8xh100, rx7900xtx,
  m4max-64g). Predictions are checked against measured runs (src/plan.rs tests).
- The engine's five level tables and tag parsers are one module (src/levels.rs).
- Constrained decoding (llguidance): `response_format` `json_object` and `json_schema`, vLLM's `structured_outputs` and `guided_json`, `guided_regex`, `guided_choice` and `guided_grammar`, and
  forced tool calls (`tool_choice` a function or `required`) are enforced at sampling time by token masks, on every model the server runs (`glyd-engine/src/constrain.rs`); they were refused before. A constrained draw takes the GPU's 64 largest logits with the grammar's mask put on the row before the top-k (a kernel of its own, `topk64_bf16_masked`; the mask is made by worker threads beside the step and uploaded behind it) and reads the whole row only where those hold too few allowed tokens (the same draw, made on the host from the unmasked candidates, where the kernels lack it).
- `glyd run` and `glyd serve` take a GGUF: a `.gguf` file, a directory of them, or a Hub repo that holds them and no safetensors. The engine reads the file's header, metadata and
  tensors in place (src/gguf.rs; GGUF versions 2 and 3, every ggml type's size checked) and, for a Qwen3 model stored as BF16, makes the Hugging Face checkpoint it describes in the cache
  (config, tokenizer and chat template from the file's own metadata, the weights as they are) and packs that as it packs any checkpoint. F32 and F16 tensors are read where every value is
  a bf16 value exactly (llama.cpp keeps a BF16 model's norms in F32). A quantized GGUF (Q8_0, Q4_K_M, the IQ types ...) is refused, naming the types in it: its kernels are not in the engine yet.
- A GGUF in several files (`NAME-00001-of-00005.gguf`, whose first file may hold the metadata alone) is read as one model, and its files are all downloaded from a Hub repo. `glyd fit`, `supports`,
  `preflight` and `costs` take a GGUF too, a Hub repo of them by its files' headers (nothing of the weights is downloaded). A GGUF of another architecture is refused by name; of the recent families
  (Qwen3.5 to 3.8, Gemma, Muse Spark, Kimi-K3, DeepSeek-V4, GLM) it says what llama.cpp's converter did to the tensors (norm weights stored plus one, exponentials folded, experts stacked), which
  is why it is not the release's checkpoint, and the engine runs the checkpoint.
- A GGUF's tokenizer is built as the model's own tokenizer.json: byte-level BPE with any of llama.cpp's pre-tokenizers (the Qwen, Llama 3, GLM, DeepSeek, Phi and GPT-4o families' among them; src/gguf_pre.rs)
  and SentencePiece (Mistral, Gemma), whose merges come from the pieces' scores as transformers' converter makes them. Token ids and decoded text are the models' own on ten models' tokenizers
  (checked against their Hugging Face files). Another kind of tokenizer, or a pre-tokenizer the table has not, is refused by name.
- Repository: concise developer docs (`docs/architecture.md`, `ROADMAP.md`, `CONTRIBUTING.md`); codec research records moved
  under `research/codec/`, engine design notes under `glyd-engine/docs/design/`. CI split by path (`ci.yml`, `python.yml`,
  `kernels.yml`); GPU tests on a self-hosted GPU machine (`gpu.yml`) or an ephemeral cloud instance (`gpu-cloud.yml`).
- Release builds no longer rename the kernels or strip the engine's development commands: the binaries are built from this source as it
  reads.
- On Linux with an NVIDIA GPU, install.sh installs the Rust `glyd`, Glyd's engine and its GPU library in one folder: about 100 MB, no
  Python, no PyTorch. `GLYD_WITH_VLLM=1` installs the Python tool with vLLM as before (about 8 GB), for the bf16 models the engine does not
  run; a machine that already has the Python tool keeps it.
- `install.sh` takes `GLYD_RELEASE_URL` (a mirror, or `file:///DIR`) in place of the GitHub release's download URL, for the compression program's and the engine's tarballs.
- Every artifact of a release is built from this repository: `release.yml` (the CLI tarballs and wheels) and `release-gpu.yml` (the GPU libraries, the engine, the `glyd-gpu` wheels and the `glyd-engine-vX.Y.Z-linux-{x86_64,aarch64}.tar.gz` bundles with their `.sha256`).
- `glyd run` and `glyd serve` run DeepSeek-V4-Flash, GLM-5.3-Flash, Muse Spark and Nemotron 3.5 Lightning on Glyd's engine at penguin:
  the release's weights exactly (an FP8 or FP4 release as it is, a bf16 one lossless). The layers are split over the machine's GPUs by
  their free memory (`--split CUTS --devices GPUS` chooses); where the GPUs do not hold the model, `run` says how much it needs and stops.
  `glyd preflight` and `glyd fetch` know these models too.
- `glyd run` and `glyd serve` run Kimi-K3 (2.8T parameters, MXFP4 experts) at penguin, the release's weights exactly. Where the GPUs
  don't hold its 1.45 TB of experts, they stay in the computer's memory and each GPU keeps as many as fit, the ones the router asks for
  (8x A100 80 GB: 318 to 430 of a layer's 896). `run` says how many, and refuses with the numbers where the memory is short. On 8x A100
  80 GB: ready in 35 minutes, about 4 tokens a second. The experts the GPUs don't hold are computed on the host's cores beside the copies
  of the others. On A100, H100 and B200 its bf16 matrices are held in Glyd's 12-bit lossless layout, the faster one there (one H100: 22% a
  step). `--context` takes any length the memory holds. Its reasoning comes back apart, and its tool calls as OpenAI tool calls,
  `tool_choice` `required` or a named function too.
- Downloads add connections while that makes them faster (8 to start, up to 64): a datacenter link gives a connection a few MB/s.
- Models with FP4 experts load faster: the packs are checked code by code, with no matrix of values made.
- An FP8 release whose weights do not fit the GPUs as they are is held in Glyd's lossless FP8 coding (the same values, about 7 bits in 8):
  GLM-5.3-Flash fits four 80 GB GPUs.
- `glyd serve` of DeepSeek-V4-Flash and GLM-5.3-Flash takes `tools`: the model's tool calls come back as OpenAI tool calls, whole or
  streamed, with `tool_choice` `required` or a named function.
- Nemotron 3 Super at hummingbird runs on the same engine as the other models, its expert cache in the computer's memory included, with
  the same outputs.
- `glyd-engine costs MODEL[:level]` prints the model's cost profile at the level from its packed files: each layer's resident bytes by component, the bytes a
  decoded token reads, the KV and state bytes per token, with `--predict` the decode speed over this machine's memory tiers (`glyd-engine rates` measures them).
  `glyd serve` gives the same as `costs` in `/v1/glyd/info`, with the allocation measured; `glyd-engine preflight MODEL:level --costs` prints it where `glyd run` asks. docs/cost-profile.md.
- `glyd run google/gemma-4-12B-it` (or `google/gemma-4-31B-it`, or their penguin repos) on a GPU without room for penguin names the wren repo and the memory
  it takes, where it fits (`glyd run glyd/gemma-4-12B-wren fits: ...`), and stops there; it handed the model to vLLM, whose bf16 needs more memory still, and
  which printed a second estimate and suggested a model Glyd does not run.
- One program: `glyd` with the engine linked (the `engine` feature of the root crate, which the engine bundle of a release is built with). `glyd run`, `serve`, `pull`, `fit`, `doctor`,
  `login`, `fetch`, `supports`, `preflight`, `prepare`, `costs`, `rates` and `rangeworker` are its own, with no `glyd-engine` on PATH; `glyd-engine` stays, as a link to `glyd` (or a copy named so), which
  takes the engine's old command line, and as the cargo package. `fit` and `pull` reach the engine where they used to print "Unknown option"; with no engine here the answer says so. `glyd` with no words
  explains the models first and the exit status (0 done, 1 an error, 3 not supported). install.sh puts the one program in place of the compression program and links `glyd-engine` to it.
- Model names: `glyd run gemma4`, `glyd pull qwen3.8:27b:linnet`, `glyd fit gemma4:31b`. A name is `NAME[:SIZE][:LEVEL]` from a versioned manifest (`glyd-engine/models.json`, in the program; `$GLYD_MODELS` or `<cache>/models.json` names another),
  a tag of letters only is a level and `:27b` a size, a name with no size is the family's default (Gemma 4 12B, Qwen3.8-27B, GLM-5.3-Flash, Kimi-K3, Muse-Spark-Open: released within six months), the older families (Qwen3, Qwen3.5, Qwen3.6, Nemotron 3) still resolve but are no default or example, and a name that is none says the names and the nearest one. A Hub id, a directory and a GGUF file are taken as before.
  `glyd pull` downloads a model (the level's pre-packed repo where there is one, else the release's own weights, which the first run packs); `glyd ls [--all] [--json]` lists what the Hub's cache holds with its level and size
  on disk and the packed copies made for this GPU; `glyd rm NAME|ORG/NAME[:LEVEL]` takes a model out of the cache and its packed copies, and no other model's.
- The background service (docs/service.md): `glyd service` owns the API port (127.0.0.1:11434, fixed: a port in use is an error that names the program), loads a model when a request names it, unloads it after
  `--keep-alive` idle (5 minutes) and holds `--max-loaded` models (2; the least recently used idle one makes room, and a load that fails is tried again with the idle ones unloaded). Each model is a `glyd serve` of its own,
  so the server's behaviour is unchanged. `glyd ps` shows what is loaded, `glyd stop MODEL|--all` unloads, `glyd run MODEL` chats through the service where it runs, and `glyd service install` makes it a user unit
  (systemd, launchd; install.sh does it on Linux with an NVIDIA GPU, `GLYD_NO_SERVICE=1` leaves it out). It loads only downloaded models, refuses a foreign Host or Origin, and its stop and unload need a secret only the account can read.

## v0.32.1 — 2026-10-17

- `glyd fetch MODEL` downloads a model (or `MODEL:LEVEL`'s pre-packed repo) without running it; the pip-installed `glyd` sent it to
  the compression program. `glyd fetch glyd/gemma-4-12B-wren` asked the Hub for `glyd/gemma-4-12B-wren-wren`; a repo's own name is now
  fetched as it is.
- `glyd verify glyd/gemma-4-12B-penguin` downloads the repo first when it is not on this computer (it said to download it with `hf download`).
- `glyd run` of a Gemma 4 or lossy-level `glyd/` repo on a GPU without room for it says so and stops; it handed the repo to vLLM, which
  cannot read those files, and printed a second, different estimate.
- Gemma 4 writes faster on Glyd's engine (wren and drafting; the step's attention, and the products of a step and of a drafts' check read
  closer to the card's memory speed). RTX 4080 SUPER, Gemma 4 12B at wren, one user, without the draft model: 81.2 to 88.9 tokens a second
  in chat, 74.7 to 82.1 after a 4K-token prompt, 59.5 to 64.0 after a 30K-token prompt. Outputs are the same, with and without drafting.

## v0.32.0 — 2026-10-17

- Gemma 4 12B and 31B at penguin, lossless: `glyd run glyd/gemma-4-12B-penguin` downloads `glyd/gemma-4-12B-penguin` (16.1 GB, plus
  Google's 0.88 GB draft model); the 31B is `glyd/gemma-4-31B-penguin` (41.3 GB + 0.97 GB). Every weight is Google's bf16 weight, bit for
  bit (`glyd verify` checks it), at about 10.8 bits per weight: the 12B takes 15.0 GiB on the GPU in place of 22.2 GiB and runs on a 24 GB
  card, the 31B 38.4 GiB in place of 57.2 GiB and runs on a 48 GB card. Measured, one user, greedy, with the draft model (without it):
  - RTX 4090, 12B: 89.8 tokens a second in chat (50.8); a 28K-token prompt read in 7.6 s.
  - RTX A6000 48 GB, 31B: 34.3 in chat (13.3); a 24K-token prompt read in 35.5 s. H100 PCIe, 31B: 61.8 (30.0); 23.0 s.
  - GSM8K 72/75 and MMLU-Pro 67/75, both sizes. KLD against transformers' float32 run (wikitext-2, 32 x 512), 12B: 0.170, where
    transformers' own bf16 runs are at 0.132 and 0.143.
- `glyd run google/gemma-4-12B-it` (and 31B), with no level, runs lossless (penguin) on Glyd's engine where the GPU has room: the engine's
  check before the download now takes Gemma 4 the way `run` and `serve` do (it said "runs Qwen3 models so far" and handed the model to vLLM).
  12B takes about 17.5 GiB at 4096 tokens of context with its draft model, 31B about 42.1 GiB.
- `glyd serve`: a request that fails has its reason in the server's log, not only in the reply to the client.

## v0.31.1 — 2026-10-16

- `glyd run google/gemma-4-31B-it:wren` (and `google/gemma-4-12B-it:wren`) runs the wren repo on Glyd's engine; 0.31.0 refused it
  (":wren is not built yet"). `glyd run glyd/gemma-4-31B-wren` works as before.
- `glyd serve` runs Gemma 4 on Glyd's engine. 0.31.0 handed it to vLLM, which has no route for these files.
- Downloads from Hugging Face go on over IPv4 when a request over IPv6 fails or is reset, and stay on IPv4 for the run;
  `GLYD_IPV4=1` uses IPv4 from the start.
- `/health` answers 503 (`{"status":"loading"}`) while the Qwen3.8-Flash-Next server is still loading its model, and 200 once it can serve.
- `glyd run`, `serve`, `pack` and `verify` on a computer other than Linux say they run on Linux with an NVIDIA GPU, instead of
  suggesting `pip install "glyd[gpu]"`, which finds no glyd-gpu build there.

## v0.31.0 — 2026-10-15

- Gemma 4 31B and 12B at `:wren`: `glyd run glyd/gemma-4-31B-wren` downloads
  `glyd/gemma-4-31B-wren` (17.4 GB + its 1.0 GB draft model); 12B is `glyd/gemma-4-12B-wren` (6.8 GB + 0.85 GB). Packed from Google's QAT
  checkpoints, which are 4-bit codes times a scale per 32 weights: wren keeps them bit for bit (4.5 bits a weight; the pack refuses a matrix
  it cannot give back exactly). Release builds now run wren on a QAT checkpoint; wren on other checkpoints stays a development build's.
- Drafting with Google's Gemma 4 draft models (the `-assistant` repos, shipped in each repo's draft/): the draft model proposes up to 8
  tokens a round and the model checks them in one pass; greedy output is the same token for token, sampled output keeps the model's
  distribution. `--no-draft` runs without it. Measured, one user, greedy: RTX 4090, 31B: 46 tokens a second plain, 78 with the draft model
  on chat prompts (up to 176 on predictable text), a 30K-token prompt read in 20 s; RTX 4080 SUPER, 12B: 82.5 plain, 109 served on chat
  prompts (232 on the engine test), 30K-token prompt 12.9 s. GSM8K 72/75 (both), MMLU-Pro 70/75 (31B), 66/75 (12B); KLD 0.018 against
  Google's QAT weights in transformers (31B, wikitext-2 32 x 512).
- Long prompts on Gemma 4 run on tensor cores (attention for both the 256- and the 512-wide heads): 31B's 30K-token prompt 34 s → 20 s.
- `glyd verify MODEL`: decodes every packed tensor on the CPU and checks every tensor against the sha256 recorded when it was packed; says
  whether the level holds the source's weights bit for bit (penguin; wren of a QAT checkpoint). glyd/gemma-4-12B-wren: 7 s on 16 cores.

## v0.30.0 — 2026-10-10

- Download and run: `glyd run MODEL:linnet` (or `:kestrel`) downloads the pre-packed checkpoint `glyd/NAME-LEVEL` from Hugging Face where
  one exists and starts from it; other models are packed once into the cache (glyd-engine/src/cli.rs `pre_packed`, store.rs). Published:
  Qwen3.8-27B, Qwen3.6-27B and Qwen3.5-27B/9B/4B/2B/0.8B at kestrel and linnet, each card with its measured size, speed (RTX 4090, RTX A6000,
  L40S; a 16 GB card for the 9B) and KL against bf16; the 9B and Qwen3.8-27B cards also GSM8K beside bf16 (Inspect AI through `glyd serve`;
  `.eval_results/glyd.yaml`).
- Nemotron 3 Super at `:hummingbird` (release builds now run Nemotron-H models): the routed experts as codebook neuron rows at 3 and 2 bits
  (2.99 bits a weight, from a calibration), every other matrix six-bit, 48.7 GB as `glyd/nemotron-3-hummingbird`. Where the GPU cannot hold
  it, an expert cache keeps the most-asked-for experts on the GPU and the rest in the computer's memory, and the CPU computes the experts that
  are not on the GPU from where they are held (src/offload.rs, src/cpu_hb.rs); where the computer's memory cannot hold them either, the rest
  stay in the packed directory and are read as needed (`--host-memory GIB`, default what the computer has free; src/disk.rs). Measured
  (decode, one user, 255 greedy tokens): RTX 4090 43.6 tokens a second; a 16 GB card's budget 35.4; RTX 3090 37.0; with 20 GiB of RAM for the
  experts (a 32 GB PC) 17.7 and 13.1; llama.cpp with Unsloth's UD-Q2_K_XL (54.7 GB) on the same RTX 4090: 17.4. GSM8K 97.0% (300 questions),
  MMLU-Pro 76.6% (222), non-thinking (benchmarks/gpu/nh-hb-*-2026-10-10, benchmarks/cards/2026-10-10).
- Tool calls in the XML format (Qwen3.5, 3.6, 3.8 and Nemotron 3's templates; vLLM's qwen3_coder parser) come back as `tool_calls`, whole and
  streamed, with `tool_choice` (src/tools.rs). Before, `glyd serve MODEL:linnet` returned them as plain text.
- Sampled decode on the engine's levels: the 64 largest logits are found on the GPU and the draw is made from them (penalties, top-k, top-p),
  in place of copying every logit to the host at each step (its speed on a GPU is still to be measured).
- Levels: Glyd's levels are penguin, kestrel, linnet, wren and hummingbird; a release build runs penguin, kestrel, linnet and hummingbird (wren
  is a development build's), and `:sparrow` is no level.

## v0.29.4 — 2026-10-09

- Levels: `glyd run MODEL:LEVEL` (glyd-engine/src/cli.rs `level()`, glyd_gpu/engine.py `level_of`). `:penguin` lossless (default);
  `:kestrel` = Int6 (v0.29.3's `--lossy`, still accepted); `:linnet` = Int5, new: five-bit block codes, runs of 32 of a row, one bf16
  scale, 5.53 bits per weight (gpu/glyd_gpu.cu `Int5`, glyd-gpu `pack_int5`/`unpack_int5`, `--layout i5`). Kestrel and linnet run on the
  engine only; sparrow, wren and hummingbird are refused as not built. Qwen/Qwen3.8-27B, 1,024 + 256 tokens, one user, A100-SXM4-80GB:
  linnet 17.351 GiB, 47.53 tok/s, 19.50 GiB in use; kestrel 20.479 GiB, 46.34 tok/s (benchmarks/gpu/0294-check-2026-10-09).
- Best-fit scales for Int6 and Int5: the first of 16 scales (0.70 to 1.0 of the run's largest magnitude over the code range) with the
  least squared error, pruned exactly. KL vs bf16 on 253,952 WikiText-2 positions: kestrel 0.00477 (v0.29.3: 0.00513; FP8 0.00465),
  linnet 0.01393 (benchmarks/gpu/int5-a100-2026-10-09). Packing at load is slower: A100 127 s (kestrel) and 157 s (linnet), against 49 s
  and 51 s with the single scale; pre-packed checkpoints (0.30.0) remove it.
- READMEs and help: the levels.

## v0.29.3 — 2026-10-09

- `glyd run MODEL --lossy` stores each run of 32 weights of a row as six-bit codes with one bf16 scale: about 6.5 bits per weight (6.8 in
  v0.29.2), still not lossless, and closer to bf16. With Qwen/Qwen3.8-27B, the next token's distribution differs from bf16's by a KL
  divergence of 0.00513 on 253,952 positions of WikiText-2, against 0.00831 for v0.29.2's weights and 0.00465 for Qwen/Qwen3.8-27B-FP8
  ([benchmarks/gpu/int6-l40-2026-10-09](benchmarks/gpu/int6-l40-2026-10-09/README.md)). It is faster too, and a decode step's small
  products in these models now spread over the whole GPU. Qwen/Qwen3.8-27B, a prompt of 1,024 tokens and 256 more, one user:
  - one L40: 31.3 tokens a second (v0.29.2: 25.8) with 20.5 GiB of weights, first token 481 ms; vLLM 0.30.0 with the FP8 checkpoint
    25.4 tokens a second with 28.0 GiB, first token 504 ms (measured on another L40).
  - one A100 80GB: 45.0 tokens a second (v0.29.2: 18.5), first token 438 ms; vLLM 0.30.0 with the FP8 checkpoint 47.5 tokens a second,
    first token 380 ms, on the same GPU.
  - Loading packs the weights faster: 52 s on the L40 (v0.29.2: 78 s).

## v0.29.2 — 2026-10-08

- `glyd run MODEL --lossy`: Qwen3.5, Qwen3.6 and Qwen3.8 dense models on Glyd's own engine with every weight rounded (about 6.8 bits per
  weight; not lossless), one user at a time, in the terminal (one answer with `--prompt`, or a chat). It downloads the model's bf16
  checkpoint and packs it as it loads. vLLM has no such path: where the engine does not run the model (the GPU, its memory, the driver),
  `--lossy` says why and stops, and nothing runs on vLLM. Measured on one L40 with Qwen/Qwen3.8-27B, a prompt of 1,024 tokens and 256 more:
  21.2 GiB of weights and 25.8 tokens a second, against 28.0 GiB and 25.4 tokens a second for vLLM 0.30.0 with Qwen/Qwen3.8-27B-FP8
  (first token 526 ms against 504 ms); started in 67 s with the checkpoint on disk; 32,768 tokens of context. The same values scored a
  78.68% mean on 13,243 questions of six suites against 78.54% for bf16 and 78.55% for the FP8 checkpoint, a statistically inconclusive
  difference ([benchmarks/weights/candidate-a7-quality-2026-10-08](benchmarks/weights/candidate-a7-quality-2026-10-08/README.md)).
- glyd_gpu loads and saves these weights (`glyd-v6` checkpoints; `glyd-gpu pack --layout a`).

## v0.29.1 — 2026-10-06

- Mixtures of experts whose experts clamp their activation (GLM-5.3-Flash-BF16, Hy4-preview, K-EXAONE-2.0, Ling-3.0-flash): v0.29.0 ran those
  experts without the clamp under `--quantization glyd` (and so under `glyd run` and `glyd serve`), so their outputs could differ from bf16's.
  They now give bf16's result: those layers' experts stay bf16, and `exact` mode is bf16's bit for bit as before.
- `glyd run` and `glyd serve` on 2026 models that v0.29.0 did not start or answered wrongly:
  - Gemma 4 starts (v0.29.0 stopped with `AmbiguousGlobalPerLayerAttributeError` under transformers 5.18).
  - NVIDIA-Nemotron-3-Nano-4B-BF16 is no longer refused (v0.29.0 put its weights at 33.6 GiB).
  - vLLM's tool-call and reasoning parsers for every 2026 family: Qwen3.5, 3.6 and 3.8, Qwen3-Coder-Next, Gemma 4, Nemotron 3 and 3.5,
    GLM-5.x and GLM-4.7-Flash, MiniMax M2.x and M3, Kimi K2.x and K3, DeepSeek V4 and V4.1, MiMo V2, Ling 3, EXAONE 4.5, LFM2.5, Granite 4.1
    and 4.2, MiniCPM5, Apertus, Muse-Glimmer, Inkling, Jamba2, Mellum2, Command A+, Hy3 and Hy4, Step 3.5 and 3.7. v0.29.0 gave Qwen3.5 and
    Gemma 4 none on the Hub's configs, and MiniCPM5's tool calls failed with HTTP 400.
  - The context is sized from each layer of the model (full and sliding-window attention, linear attention, Mamba, experts), so it fits
    vLLM on the first start: v0.29.0 asked vLLM for more than it held on Qwen3.5-9B and restarted it.
  - Where a model does not fit, the smaller one `glyd run` suggests is from the model's own family for Gemma 4, Granite 4.2, LFM2.5,
    Nemotron and MiniCPM5.
  - Measured on an RTX 4090 with vLLM 0.30.0, `glyd run --prompt` and `glyd serve` with a tool call, the turn after it, the thinking
    apart from the answer and a prompt of about 90% of the context, every check passed: Qwen3.5-9B (context 183,296 tokens),
    gemma-4-12B-it (124,928), MiniCPM5-2B (131,072), NVIDIA-Nemotron-3-Nano-4B-BF16 (262,144).
- Less memory on 2026 models: `--quantization glyd` also packs the embeddings and other layers that v0.29.0 left bf16 on these
  models (Qwen3.5, 3.6 and 3.8, Nemotron 3, LFM2.5, Jamba2, Olmo-Hybrid, Ling and others). Measured on an RTX
  4090 with vLLM 0.30.0: Qwen3.5-9B's weights take 12.01 GiB against 12.69 with glyd-gpu 0.29.0, at the same speed (one user 74.1 tokens a
  second against 73.9, 32 users 954.9 against 954.9; 1,024 tokens in, 256 out). Against bf16 in the same run (Qwen3.5-9B's language model alone,
  as bf16 does not fit the card with vLLM's defaults): 11.38 against 16.80 GiB; Qwen3.5-0.8B 1.23 against 1.72, LFM2.5-350M 0.48 against
  0.69, NVIDIA-Nemotron-3-Nano-4B-BF16 5.40 against 7.47, AI21-Jamba2-3B 4.71 against 5.68. Every packed weight reads back bit for bit and
  `exact` mode gives bf16's tokens and log-probabilities bit for bit on all five.
- `python -m glyd_gpu pack MODEL OUT --tables` (and `glyd.save_pretrained(model, path, tables=True)`) also packs a model's embeddings and its other large tables
  in the save, for a smaller save; it loads to the same weights, bit for bit, and the same GPU memory. Such a save is format glyd-v5:
  glyd 0.29.0 and before do not load it, and vLLM does not serve it yet. Saves made without `--tables` are unchanged (glyd-v4, glyd-v3 with
  `--layout mma12`). `glyd verify` reads glyd-v5.

## v0.29.0 — 2026-10-06

- Glyd's own engine, `glyd-engine`, ships in the `glyd-gpu` wheels beside the library (Linux, x86_64 and aarch64): a program of its own (Rust; the CUDA driver and Glyd's GPU library are all it
  opens) that runs Qwen3 models in the terminal, at a page and as an OpenAI-compatible API. `glyd run` hands a Qwen3 model to it only on a GPU where it is measured at least as fast as vLLM and with less memory,
  and only where it will certainly run the model: before anything is downloaded it checks the GPU a run would use (the one with the most free memory), its driver, the model, the memory and the disks. No GPU
  meets that yet, so by default every model runs on vLLM, as in v0.28; the first line a command prints says which one runs the model, and why where it is vLLM. `glyd serve` runs vLLM, which serves more
  users at once than the engine's 8. `GLYD_ENGINE=1` runs the engine on any GPU it runs on (its checks of the GPU, the driver, the model and the memory still apply), `GLYD_ENGINE=0` keeps vLLM for any model, and `glyd-engine run` and `glyd-engine serve` run the engine
  directly. Where the engine does not get up (it refuses the model, runs out of memory, crashes, is killed, or shows no sign of life for 15 minutes) the same command goes on with vLLM; once it is up, a crash is
  reported and the model is not started again. The engine downloads and packs the model on its first run, chooses the context from the GPU's
  memory (8,192 tokens where it has room), and decodes up to 4 users' replies together by default (`--users N`, 1 to 8, as far as the memory allows).
  - Its API takes tool calls (`tools` with `tool_choice` auto, none, a named function or required, and `parallel_tool_calls`; the model's `<tool_call>` blocks as `tool_calls`, whole and streamed, so Open WebUI's
    chats, which carry `tools`, are answered; the arguments of a forced call are the model's, not checked against the tool's schema), `logprobs` and `top_logprobs` up to 20 (the model's own log-probabilities, as
    vLLM's default gives them) and a plain completion's `logprobs`, `n` up to 8 (each choice has its own seed, index and finish reason), `logit_bias`, `stop_token_ids`, `min_tokens`, `allowed_token_ids`,
    `include_stop_str_in_output`, `skip_special_tokens`, `echo`, prompts as token ids or as a list, and the ranges and messages of vLLM's own checks. A request for a JSON object or a JSON schema
    (`response_format`), `guided_*`, `structured_outputs` or another field only vLLM does is refused with a 400 that says so (`GLYD_ENGINE=0` runs vLLM, which takes it). On this computer's own address the server
    makes the Host (421) and Origin (403) checks of the vLLM-based server, a request's header block is bounded (431 past 64 KiB or 256 headers), and `--hf-token` is read.
- `glyd run` and `glyd serve` start vLLM compiled (torch.compile and CUDA graphs, vLLM's own default) wherever the chat keeps
  8,192 tokens of context, and eager only where that gives a longer chat; v0.28 always started it eager. One user's reply of 256
  tokens to a prompt of about 1,000, the median of three, vLLM 0.30.0: Qwen3.5-9B 115.9 tokens a second against 28.7 eager (v0.28's
  mode) on an RTX 5090, 77.3 against 26.6 on an RTX 4090. A first start compiles: Qwen3.5-9B was up in 135 s on the RTX 5090 (72 s eager).
- Qwen3.5 (2026), measured with Qwen3.5-9B and vLLM 0.30.0: the weights take 12.68 GiB on an RTX 4090 (the tiered layout) and
  13.89 GiB on an RTX 5090 (the 12-bit layout) against bf16's 17.66 GiB. On the RTX 4090 (24 GB) bf16 did not start with vLLM's
  default settings (out of memory at a 4,096-token context); `glyd run` runs it there with a 189,440-token context.
- The license: from v0.29.0 all of Glyd is under the Business Source License 1.1 (`LICENSE`): the `glyd` crate and
  command line, the C ABI, the Python and Go bindings, the store and the GPU package `glyd-gpu`. Free forever for
  personal and non-commercial use (including education, research and nonprofits) on computers you own or rent for
  yourself, and for anyone to try, test and develop with; commercial use needs a license (suryakoritala@getglyd.com);
  each version becomes Apache-2.0 four years after its release.
  - Releases up to v0.28 keep the licenses they shipped with: the codec (the `glyd` crate, the command line, the C ABI
    and the bindings) BSD-3-Clause OR GPL-2.0-only, the store and the GPU package BUSL-1.1 on the terms of their own
    LICENSE files. Their tags, wheels and crates are unchanged.
  - `LICENSE` is the one license file: `COPYING` (the GPL's text) and `glyd-store/LICENSE` are gone, and the release
    tarballs, the wheels and the crates carry `LICENSE` alone. Code in `third_party/` keeps its own license.

## v0.28.0 — 2026-10-04

- The whole model smaller under vLLM: `vllm serve --quantization glyd`, and so `glyd run` and `glyd serve`, packs the
  embedding and the output layer (the LM head) as it packs the Linears, so the whole model's weights are about a third
  under bf16's, not only its large matrices, and the memory this frees becomes KV cache. Every row of the embedding
  reads back bit for bit; the output layer's product is within the same tolerance as every other product; a model
  whose config ties the two has one packed copy. `exact` is unchanged: vLLM's bf16 logits bit for bit (it keeps the
  output layer as vLLM runs it).
  - On an L4, v0.27.0 from PyPI against v0.28.0, the same vLLM 0.30.0 in its default mode (compiled, CUDA graphs),
    alternated in one session. The weights and the KV cache as vLLM's log gives them with its own KV cache
    (`--max-model-len 4096`, 90% of the GPU's memory), the cache of a later start with a first start's, on an empty
    compile cache, in brackets:
    - Qwen3-8B: weights 15.27 GiB in bf16, 11.38 with v0.27.0 (25.5% under bf16's) and **10.38 (32.0% under)**; KV cache
      27,024 tokens in bf16, 54,464 (38,912) with v0.27.0 and **60,736 (60,736)**.
    - Llama-3.1-8B: weights 15.0 GiB in bf16, 11.05 with v0.27.0 (26.3% under bf16's) and **10.16 (32.3% under)**; KV
      cache 32,592 tokens in bf16, 64,368 (50,032) with v0.27.0 and **70,112 (70,112)**.
    - Qwen3-4B (the config ties the embedding and the output layer: one packed copy): weights 7.56 GiB in bf16, 5.48
      with v0.27.0 (27.5% under bf16's) and **5.16 (31.7% under)**; KV cache 83,408 tokens in bf16, 97,488 (88,048) with
      v0.27.0 and **99,312 (99,120)**.
  - A first start (on an empty compile cache) no longer logs a much smaller KV cache than a later one. For Qwen3-8B,
    Llama-3.1-8B, Qwen3-4B, v0.27.0's first start held 38,912 against 54,464; 50,032 against 64,368; 88,048 against
    97,488 tokens on a later start; v0.28.0's held 60,736 on both; 70,112 on both; 99,120 against 99,312 (bf16's own,
    Qwen3-8B: 19,760 against 27,024).
  - One user's tokens a second (one request, 1,024 tokens in and 256 out, vLLM's own KV cache, the median of three
    rounds, alternated): Qwen3-8B 22.54 against 21.16 (**1.065x**; rounds 1.059, 1.064, 1.065); Llama-3.1-8B 23.01
    against 21.92 (**1.050x**; rounds 1.050, 1.050, 1.054); Qwen3-4B 39.62 against 37.54 (**1.055x**; rounds 1.053,
    1.055, 1.054).
  - `vllm bench serve`, 1,024 tokens in and 256 out, the same flags but the KV cache at its default (`auto`: lossless on
    an L4), saturated (256 requests at once), two rounds with prompts of their own a pass (the prefix cache's hit rate
    in the servers' logs at most 0.4%): Qwen3-8B 1.33 and 1.31 requests a second against 1.21 and 1.20 (**1.095x** and
    **1.095x**), the first token after 81,904 and 83,147 ms against 87,470 and 88,085, each token 176.9 and 178.4 ms
    against 171.6 and 173.0, the KV cache 79,600 tokens against 71,248 (a first start of v0.27.0's: 50,608);
    Llama-3.1-8B 1.37 and 1.36 requests a second against 1.32 and 1.31 (**1.040x** and **1.041x**), the first token
    after 75,873 and 76,185 ms against 81,325 and 81,793, each token 187.6 and 188.3 ms against 181.0 and 181.8, the KV
    cache 90,384 tokens against 81,792 (a first start of v0.27.0's: 62,784); Qwen3-4B 2.33 and 2.33 requests a second
    against 2.26 and 2.25 (**1.031x** and **1.037x**), the first token after 40,851 and 40,708 ms against 42,008 and
    42,143, each token 151.2 and 150.7 ms against 153.9 and 154.3, the KV cache 129,344 tokens against 127,152 (a first
    start of v0.27.0's: 114,656).
    - At 1 request a second (64 requests, the same flags): Qwen3-8B 0.83 and 0.82 requests a second against 0.82 and
      0.81, the first token 633 and 593 ms against 659 and 608, each token 83.8 and 85.3 ms against 88.4 and 91.0;
      Llama-3.1-8B 0.83 and 0.83 requests a second against 0.83 and 0.82, the first token 638 and 588 ms against 639 and
      601, each token 79.8 and 82.1 ms against 83.6 and 86.0; Qwen3-4B 0.90 and 0.89 requests a second against 0.89 and
      0.89, the first token 298 and 291 ms against 300 and 295, each token 38.1 and 38.5 ms against 38.8 and 39.6.
  - They stay as vLLM runs them, with a line in the log that says why, under a `fraction` below 1, speculative
    decoding, `exact` (the output layer alone: an untied model's embedding is packed, a tied model's one table stays),
    an output layer whose dtype is not bf16, an embedding class that is not a plain lookup, and a hidden size that is
    not a multiple of 64. The lossless KV cache works with them. `glyd run` and `glyd doctor` count the packed tables
    when they work out what fits the GPU.
  - Tensor parallel 2 was checked on two L4s (of an AWS g6.12xlarge): Qwen3-4B (7 of 7 checks) and two tiny models, one
    tied and one untied (8 of 8): each rank packs its share of the tables, every lookup is the checkpoint's row bit for
    bit, and `exact` is bf16 eager's bit for bit. That run was on the code as of 2026-10-04, before the release's last
    merges.
  Logs: [benchmarks/gpu/vllm-v028-l4-2026-10-04](benchmarks/gpu/vllm-v028-l4-2026-10-04), [vllm-full33-l4-2026-10-03](benchmarks/gpu/vllm-full33-l4-2026-10-03),
  [l4x2-vllm-tp2-2026-10-04](benchmarks/gpu/l4x2-vllm-tp2-2026-10-04).
- Mixtures of experts serve faster than bf16. Against vLLM's own bf16 path with its default MoE settings (its log says
  `Using default MoE config. Performance might be sub-optimal!`, and a tuned bf16 would be faster than that baseline:
  not measured), 1,024 tokens in and 256 out, saturated, vLLM 0.30.0:
  - On an H100 SXM, Qwen3-30B-A3B: 13.13 requests a second against 12.32 (**1.07x**), and 17.10 with the lossless KV
    cache (**1.39x**); the first token after 3,853 ms and 3,437 ms against 6,920 (0.56x and 0.50x), each token 45.9 and
    43.7 ms against 27.2 (1.69x and 1.61x as long). The weights take 44.65 GiB against 56.88, and the KV cache holds
    251,696 and 331,008 tokens against 118,720. At 1 request a second: 0.98 and 0.97 requests a second against 0.97,
    each token 6.3 ms against 6.6.
  - On an A100 40 GB, Qwen3-16B-A3B (Qwen3-30B-A3B with half its experts; the 30B does not fit in bf16): 6.18 requests
    a second with the lossless KV cache against 4.12 (**1.50x**), the first token after 14,233 ms against 27,100
    (0.53x), each token 62.9 ms against 30.9 (2.04x as long). The weights take 23.97 GiB against 29.87, and the KV cache
    holds 144,896 tokens against 40,288 (3.60x). At 1 request a second: 0.95 against 0.94 requests a second, each
    token 12.3 ms against 13.9 (0.88x), and the first token 110 ms against 102 (1.08x later).
  - Prompt steps are where Glyd is behind: in the H100 run 5.7 s of the saturated pass's 14.3 s with the KV cache were
    32 prompt steps, 22 us a prompt token against bf16's 15, while its 255 decode steps took 8.6 s against bf16's 15.7.
    The attention products and the router stay bf16 for a mixture of experts' model on an H100 and an A100.
  - Both runs are of the mixture-of-experts work as it stood on 2026-10-03, with no setting changed by hand; they were
    not repeated on the release's final tree.
  Logs: [benchmarks/gpu/h100-moe-v028-2026-10-03](benchmarks/gpu/h100-moe-v028-2026-10-03),
  [a100-moe-v028-2026-10-03](benchmarks/gpu/a100-moe-v028-2026-10-03).
- A newer model under vLLM: Qwen3.8-27B (text only), on an H100 80GB HBM3, vLLM 0.30.0, 1,024 tokens in and 256 out. The
  weights take 38.77 GiB against bf16's 50.22 (22.8% under, the `mma12` layout this GPU takes by default; v0.27.0:
  39.69), and the KV cache holds 204,117 tokens against 122,538 (1.67x; bf16 started again at the end: 132,778).
  Saturated (256 requests at once): 5.05 requests a second against 4.72 and 4.70 (**1.07x**), the first token 0.84x
  as late, each token 1.43x as long. At 1 request a second: 0.92 against 0.92, each token 21.7 ms against 23.4 (0.92x),
  the first token 227 ms against 199. bf16's continuations scored on Glyd: top-1 99.22% (bf16 against itself started
  again: 99.22%). The lossless KV cache stays off for this model, with a line in the log that says why (linear
  attention and head size 256 are not held yet). Glyd's saturated passes drew more power and ran at a lower clock
  (1,852 MHz against 1,980), so the run's own clock check marks it not comparable. Logs: [benchmarks/gpu/h100-qwen38-v028-2026-10-03](benchmarks/gpu/h100-qwen38-v028-2026-10-03).
- The smallest layout (`mma`) is faster at 1 to 16 tokens a step. On an L4, which takes it by default, one user's tokens
  a second on Qwen3-8B with the embedding and the output layer left as vLLM runs them are **1.042x** v0.27.0's (22.07
  against 21.19; rounds 1.041, 1.043, 1.036), and 1.065x with them packed too (above). The GPUs that take the `mma12`
  layout by default (the A10, A100, H100 and GH200) are unchanged. Logs:
  [benchmarks/gpu/vllm-v028-l4-2026-10-04](benchmarks/gpu/vllm-v028-l4-2026-10-04) (`one_user/Qwen3-8B/layout_alone`).
- Saves are format glyd-v4: `glyd pack` and `save_pretrained` write the smallest layout as glyd-v4 (`--layout mma12`
  stays glyd-v3). Every earlier save loads (a glyd-v1 or glyd-v2 save is packed again as it loads); glyd-v4 saves are
  refused, by their format, by v0.27.0 and earlier. The C API is version 15 (v0.27.0's was 14): the header still
  carries 8 functions, and `glyd_gpu_mma_linear` takes the packs of a glyd-v4 save and refuses (`cudaErrorInvalidValue`)
  those of a glyd-v1 or glyd-v2 save, which `glyd_gpu_mma_unpack` still unpacks.
- The `mma12` layout stays the default on the A10, A100, H100 and GH200, and the KV cache defaults are as in v0.27.0
  (`kv` auto: on for an A100, an L4, an H100 and a GH200).

## v0.27.0 — 2026-10-03

- The lossless KV cache in vLLM: vLLM's KV cache held in fewer bits, every key and value read back as written, bit for
  bit. On the five models measured it holds 1.25x to 1.30x vLLM's tokens in the same memory, and on an A100, an L4, an
  H100 and a GH200 decoding is faster than with vLLM's own cache at every batch measured.
  - The option is `kv`: `auto` (the default), `lossless` or `off`, set with `GLYD_KV` or `--additional-config '{"glyd":
    {"kv": "lossless"}}'`. `auto` holds the cache on an A100, an L4, an H100 and a GH200 (by the name the GPU gives;
    measured on an A100 SXM4 40 GB, an L4, an H100 SXM and a GH200) and leaves vLLM's own cache on any other GPU, with
    a line in the log that says so (an L40S, an RTX 40, an H200 and an A10 are not measured yet); it leaves it where
    `exact` or `verify` is on too. `lossless` holds it on any GPU it supports and refuses with why where it cannot;
    `off` is vLLM's own. It works with any `fraction`, 0 included (bf16 weights with the cache held).
  - `glyd run` keeps vLLM's own cache unless `GLYD_KV` is set (`GLYD_KV=auto glyd run MODEL`); `glyd serve` and
    `vllm serve --quantization glyd` use `auto`.
  - A first start with the cache on sets it up once for the model, for the longest context the server will serve
    (4,096, 8,192, 16,384, 32,768 or 40,960 tokens), and keeps the result in `~/.cache/glyd/kv` (`$GLYD_CACHE/kv`). On
    an L4 that took 130 s for 8,192 tokens (through `glyd run`) and 217 s for Qwen3-8B's whole window of 40,960 tokens
    (the file 3.9 MB). A window past 40,960 tokens keeps vLLM's own cache (`auto` and `lossless` alike, with one line
    that says why).
  - `kv: lossless` is refused, with why, for tensor, pipeline and context parallel, speculative decoding, sliding-window,
    linear-attention and state-space layers, MLA, LoRA, KV connectors and offloading, `--kv-cache-dtype` other than
    auto, head sizes other than 128, more than 8 query heads to a KV head, a model that is not bf16, another attention
    backend, GPUs other than Ampere, Ada and Hopper, and a model whose keys and values it cannot hold well.
  - What is exact: the stored values. Every value read back was the value written: 0 of 25,683,296,256 differ in a stress
    run, and `verify`, which compares every write with a bf16 copy, found none that differs in 6,329,327,616 values
    (Qwen3-8B). Prompt logprobs are vLLM's bit for bit (a 36,000-token prompt: 35,999 of 35,999; 14 of 14 prompts with
    chunked prefill on Llama-3.1-8B). A decode step's attention is not bit-equal to vLLM's, so greedy tokens can part
    from vLLM's after some tokens (that 36,000-token prompt: the same for the first 19 of 64), as they do with vLLM's
    own attention when one of its settings changes. In the prefix-hit scenario on an A100 (14 prompts, one request at a
    time) 13 of 14 first tokens were vLLM's bit for bit and 8 of 14 sequences of 64 tokens identical (vLLM's own with the
    setting changed: 13 and 9); with `exact`, which runs vLLM's own attention for decode steps too, 14 of 14 and 14 of
    14.
  - Measured against vLLM's own cache, Qwen3-8B, vLLM 0.30.0 in its default mode (compiled, CUDA graphs), 128 new tokens
    a request, the tokens a second at the same batch:
    - On an L4: 1.023x (1 request of 8,192 tokens), 1.052x (3 of 8,192), 1.047x (24 of 1,024), 1.053x (6 of 4,096) and
      1.007x (1 of 1,024); a decode step 1.005x to 1.053x faster; the KV cache 41,408 tokens against 32,208 (1.286x).
      With `vllm bench serve` (1,024 tokens in and 256 out, every pass with prompts of its own, 0.0% prefix-cache hits),
      saturated: 0.93 requests a second against 0.76 (1.23x) with the cache alone, on 34,912 KV tokens against 27,024
      (1.29x); 1.11x at 1 request a second; one user's token 60.2 ms against 60.6 (0.99x). With Glyd's weights and the
      cache together, against bf16: 1.20 requests a second against 0.76 (**1.59x**) with **2.64x the KV tokens**
      (71,248 against 27,024), the first token 0.58x and each token 1.55x as long.
    - On an A100 SXM4 40 GB: 1.032x (1 of 8,192) and 1.018x (1 of 1,024); a decode step 1.013x to 1.097x faster; the KV
      cache 177,392 tokens against 136,592 (1.299x). With `vllm bench serve`, 1,024 tokens in and 256 out, saturated:
      7.62 requests a second against 5.92 (1.29x), each token 59.0 ms against 63.5 (0.93x), the first token 12,408 ms
      against 15,319 (0.81x), on 183,424 KV tokens against 141,040 (1.30x); at 1 request a second and for one user
      within 4% of vLLM's.
    - On an H100 SXM: 1.005x (1 of 1,024) to 1.197x (32 of 8,192), and 1.342x with each cache at the largest batch it
      holds (46 against 60 requests of 8,192 tokens); the KV cache 499,312 tokens against 382,448 (1.306x). With
      `vllm bench serve`, 1,024 tokens in and 256 out, saturated: 21.68 requests a second against 20.58 (1.05x), each
      token 31.2 ms against 33.9 (0.92x), the first token 1.02x, on 510,288 KV tokens against 388,928 (1.31x). With 8,192
      tokens in and 256 out, saturated, it served more requests a second and its first token came sooner, but each
      generated token took longer: 2.80 requests a second against 2.67 (1.05x), the first token after 8,196 ms against
      9,199 (0.89x), each token 49.2 ms against 43.9 (1.12x as long). In that run the KV cache held 494,256 tokens
      against 377,024 (1.31x) and the model's memory was 1.07x (16.34 against 15.27 GiB).
    - On a GH200: 1.003x (1 of 1,024) to 1.187x (32 of 8,192), and 1.305x with each cache at the largest batch it
      holds (58 against 76 requests of 8,192 tokens); the KV cache 630,112 tokens against 482,864 (1.305x). With
      `vllm bench serve`, 1,024 tokens in and 256 out, saturated: 22.47 requests a second against 21.18 (1.06x), each
      token 29.3 ms against 32.1 (0.91x), the first token 1.01x, on 641,072 KV tokens against 489,344 (1.31x). With 8,192
      tokens in and 256 out, saturated: 3.12 requests a second against 2.78 (1.12x), the first token after 7,944 ms
      against 8,321 (0.95x), each token 47.8 ms against 49.5 (0.97x). An H200 is not measured yet.
    - The KV cache on five models, an L4, eager, `--max-model-len 8704`: Qwen3-4B 1.2982x, Qwen3-8B 1.2834x,
      Qwen2.5-7B 1.2773x, Mistral-7B-v0.3 1.2743x, Llama-3.1-8B 1.2526x.
  - Prompts of a model's rare tokens (tokens whose embeddings were never trained: Llama-3.1-8B has 289, Mistral-7B-v0.3
    902, the Qwen models none) are the cache's known limit. 40 of them of 1,400 tokens at once with 8 ordinary prompts:
    all 40 served on every one of the five models, none ended with an error; the flood ran at 0.88x (Llama-3.1-8B) and
    0.77x (Mistral-7B-v0.3) of vLLM's prompt tokens a second, and at 1.00x to 1.01x on the Qwen models. The ordinary
    prompts sent with it were not held behind it: 8.1 s on Mistral-7B-v0.3 and 10.5 s on Llama-3.1-8B, where vLLM's own
    took 19.0 s and 19.8 s.
  - The model checks against vLLM's own pass on all five models (13 of 13 on Qwen3-4B and Qwen2.5-7B, 21 of 21 on
    Qwen3-8B, Mistral-7B-v0.3 and Llama-3.1-8B); Llama-3.1-8B's full run passes 34 of 35, the one miss being the
    prefix-hit check, where 13 of 14 first tokens were vLLM's bit for bit. On the H100 and the GH200 the kernels'
    81 of 81 checks and the prefix-caching, chunked-prefill and `exact` checks, 12 of 12, pass.

  Logs: [benchmarks/gpu/l4-vllm-kv-step-2026-10-02](benchmarks/gpu/l4-vllm-kv-step-2026-10-02),
  [l4-vllm-kv-graphs-2026-10-01](benchmarks/gpu/l4-vllm-kv-graphs-2026-10-01),
  [a100-vllm-kv-step-2026-10-02](benchmarks/gpu/a100-vllm-kv-step-2026-10-02),
  [a100-vllm-kv-prefix-2026-10-02](benchmarks/gpu/a100-vllm-kv-prefix-2026-10-02),
  [h100-vllm-kv-2026-10-02](benchmarks/gpu/h100-vllm-kv-2026-10-02),
  [gh200-vllm-kv-2026-10-03](benchmarks/gpu/gh200-vllm-kv-2026-10-03),
  [l4-vllm-kv-window-2026-10-02](benchmarks/gpu/l4-vllm-kv-window-2026-10-02),
  [l4-vllm-kv-wait-2026-10-02](benchmarks/gpu/l4-vllm-kv-wait-2026-10-02).
- The GPU parts ship compiled, as the new `glyd-gpu` package: Glyd's GPU half (`glyd.from_pretrained`, the vLLM plugin,
  `glyd run`, `glyd serve`, `glyd pack`, `glyd verify` and `python -m glyd_gpu`) is compiled wheels for Linux x86_64 and
  aarch64. `glyd` stays the public package, with a thin `glyd.gpu` that re-exports the package's API and says what to
  install where it is missing.
  - The install commands are the same: `curl -LsSf https://getglyd.com/install.sh | sh`, `glyd run MODEL`, `pip install
    "glyd[gpu]"` and `pip install "glyd[vllm]"` (each brings `glyd-gpu` of the same version).
  - The license is unchanged: the Business Source License 1.1, on the same terms as before. Versions up to v0.26.0
    stay source-available in their tags.
  - The C API's public header, `glyd_gpu.h`, comes in the `glyd-gpu` wheel with the libraries (CUDA 12 and 13) and
    carries 8 functions: the version, the text of an error, and for each of the two layouts a product with its
    workspace query and an unpack to bf16, bit for bit. The C API is version 14; v0.26.0's header declared 52
    functions (version 7), and it and its libraries stay in that release. The release no longer carries separate CUDA
    library downloads.
  - The compiled GPU package exposes only its public C functions.
  - `glyd pack` and `glyd verify` are the Python tool's commands, as `glyd run` is (Linux, an NVIDIA GPU); the
    `glyd-gpu` program and crate are gone. The release's tarballs and Homebrew carry `glyd` and `glyd-store`.
- Fixes: with transformers 5.18, Inkling's embedding, whose rows are normed after the lookup, is packed with its norm
  kept and gives what it gave bit for bit, and several embedding classes that are not plain lookups are no longer packed.
  `glyd run` with the cache on no longer cuts the window to fit it: a window up to 40,960 tokens keeps the lossless
  cache and a longer one keeps vLLM's.
- Corrected serving numbers. The `vllm bench serve` rows published in v0.26.0 came from a benchmark whose passes
  repeated prompts, so a large KV cache could serve some of them from vLLM's prefix cache; each pass now uses new
  prompts and every summary prints each server's prefix-cache hit rate. The rows measured again that way (Glyd's
  requests a second against bf16's, saturated):

  | GPU | Model | Saturated requests/s, v0.26.0 | Saturated requests/s, now | Prefix-cache hit rate, bf16 / Glyd, v0.26.0 | Prefix-cache hit rate, now |
  | :--- | :--- | ---: | ---: | :--- | :--- |
  | A100 40 GB | Qwen3-8B | 1.19x | **0.98x** (5.73 against 5.85) | 43.1% / 53.5% | 1.3% / 1.3% |
  | A100 40 GB | Qwen3-14B | 1.65x | **1.28x** (3.37 against 2.63) | 40.8% / 52.7% | 0.0% / 0.0% |
  | GH200 | Qwen3-8B | 0.92x | **0.93x** (19.61 against 21.16) | 46.1% / 46.8% | 0.9% / 1.0% |
  | GH200 | Qwen3-32B | 0.88x | **0.89x** (3.88 against 4.36) | 53.5% / 46.3% | 0.0% / 0.0% |
  | H100 SXM | Qwen3-30B-A3B | 0.95x | **0.84x** (10.07 against 12.03) | 46.1% / 51.4% | 1.1% / 1.0% |
  | L4 | Qwen3-8B | 1.33x | 1.33x, unchanged (1.39x measured again: 1.05 against 0.76) | 0.0% / 48.3% | 0.1% / 0.2% |
  | A10 | Qwen3-8B | 1.31x | 1.31x, unchanged (its logs show no hits; not measured again) | 0.0% / 0.0% | |

  - The L4's saturated pass served nothing from the cache (its Glyd server's hit rate rose only in the pass at 1
    request a second), so its 1.33x stands. At 1 request a second its first token is 68% sooner than bf16's (2,323 ms
    against 739), not 85% sooner, and its requests a second 1.18x, not 1.15x.
  - The Qwen3-8B run on the A100 and the run on the H100 have hit rates just above the summary's 1% line (1.3% and 1.1%),
    the same on both sides; on the L4 a run with one repeated prompt of 328 (hit rates 1.1% to 1.2%) and one with none
    (0.0%) gave saturated rows within 1.4%.
  - The GH200's rows moved by 0.01x. 2x RTX A6000 (Qwen3-30B-A3B): being re-measured after a benchmark fix; v0.26.0's
    row for it is withdrawn until then.

  Logs: [benchmarks/gpu/vllm-m3-a100-40gb-new-prompts-2026-10-03](benchmarks/gpu/vllm-m3-a100-40gb-new-prompts-2026-10-03),
  [vllm-m3-gh200-new-prompts-2026-10-03](benchmarks/gpu/vllm-m3-gh200-new-prompts-2026-10-03),
  [vllm-m6-h100-new-prompts-2026-10-03](benchmarks/gpu/vllm-m6-h100-new-prompts-2026-10-03),
  [l4-vllm-m5-new-prompts-2026-10-03](benchmarks/gpu/l4-vllm-m5-new-prompts-2026-10-03); the v0.26.0 runs' hit rates
  are in their own logs ([vllm-m3-a100-40gb-2026-09-30](benchmarks/gpu/vllm-m3-a100-40gb-2026-09-30),
  [vllm-m3-gh200-2026-09-30](benchmarks/gpu/vllm-m3-gh200-2026-09-30), [vllm-m6-h100-2026-10-01](benchmarks/gpu/vllm-m6-h100-2026-10-01), [l4-vllm-m5-2026-09-30](benchmarks/gpu/l4-vllm-m5-2026-09-30)).

## v0.26.0 — 2026-10-01

- A model on an NVIDIA GPU with no flags to find: `curl -LsSf https://getglyd.com/install.sh | sh`, then `glyd run
  Qwen/Qwen3-8B`. The script (`scripts/install.sh`) installs uv where there is none, then Glyd with vLLM and PyTorch
  as an isolated tool on a Python 3.12 that uv manages (no sudo, no virtual environment, no pip; the release is
  pinned, so no pre-release of a dependency is taken), and ends with `glyd doctor`. Where the machine has no gcc or
  clang, it adds ziglang (a C compiler from PyPI, no sudo) and `glyd run` hands vLLM that as `$CC`.
- `glyd run MODEL` checks the GPU, the driver against the CUDA PyTorch was built for (580 or newer for the CUDA 13
  build vLLM 0.30 installs), a C compiler and Python.h, vLLM's version, the model (not found, gated, not bf16),
  whether it fits the memory free now (naming the largest model of its family that does, where it does not), the disk
  the download needs and the port. It downloads the model with one progress bar, picks the settings and prints them on
  one line, starts vLLM with its log in a file, and chats in the terminal (streamed, thinking shown apart; `/bye`,
  `/clear`, `/think`) and at a chat page the same server shows at its address (`--middleware glyd.gpu.page.ChatPage`:
  one HTML file with no external resource, thinking in a collapsible block, a context meter, a clear state where the
  conversation has outgrown the model's window).
- The settings `glyd run` picks: `--gpu-memory-utilization` from the card's own total and free memory, the longest
  context its KV cache holds (at most the model's own), the layout, eager mode (on an L4 with Qwen3-8B one user's
  tokens a second were within 2% of compiled, 8 users' within 3%, and the server was up in 47 seconds against 2
  minutes 45; `-- --no-enforce-eager` compiles it), `VLLM_USE_FLASHINFER_SAMPLER=0` (the sampler that needs no nvcc),
  and the tool-call and reasoning parsers by family (Qwen3 hermes and qwen3, Qwen2.5 hermes, Llama 3.x llama3_json,
  Mistral mistral, Qwen3-Coder, DeepSeek-R1 distills). Any vLLM flag after `--` wins over them.
- `glyd run MODEL --prompt "..."` prints one answer for scripts. `glyd serve MODEL` leaves the server up as an OpenAI
  API. `glyd doctor` reports the GPU, driver, CUDA, free memory, compilers, versions and which 8B, 14B and 32B models
  fit. `glyd login` saves a Hugging Face token. Any other `glyd` command is the compression program's (the Rust
  `glyd`, found on PATH). The two commands were run from nothing, in a container with no CUDA toolkit and no compiler,
  as a user that is not root (the chat page, the API, the thinking apart, a conversation longer than the window, Open
  WebUI by uvx and by Docker with and without host networking), at a 24 GB, a 16 GB and an 8 GB card's memory, and CI
  runs `install.sh` from each release's tag ([gpu/vllm/README.md](gpu/vllm/README.md#local-chat-like-ollama),
  [benchmarks/gpu/l4-onboarding-2026-10-01](benchmarks/gpu/l4-onboarding-2026-10-01)).
- `glyd run`, `glyd serve` and `scripts/install.sh`, from the review of v0.26.0rc3:
  - A server with an API key now starts (`glyd serve -- --api-key KEY`).
  - The server on 127.0.0.1 answers this computer's programs and its own page only: a Host that is not localhost,
    127.0.0.1 or [::1] is refused with 421 (DNS rebinding), an Origin that is not its own with 403, and vLLM's CORS is
    limited to its addresses.
  - The key goes to the server in `VLLM_API_KEY` (a `-- --api-key` is moved there and said so), so it is on no command
    line of the server and in no log. `-- --hf-token` is moved to `HF_TOKEN` the same way. The note for a server on
    another address names plain HTTP and what a key does not guard (the chat page, `/health`, `/metrics`, `/version`,
    `/tokenize`, `/detokenize`, `/docs`).
  - The install script is one function called on its last line, so a cut-short download runs nothing. It installs uv
    at a pinned version, checked against its sha256, and Glyd's packages at the versions the acceptance run installed
    (`GLYD_CONSTRAINTS=none` to resolve fresh). It no longer passes `--force`, and stops where `~/.local/bin/glyd` is
    a program uv did not put there. It says before it edits a shell startup file, says where another `glyd` comes
    first on PATH with the line to add, and checks HOME, 10 GB of disk and uv's age first.
  - On a Mac, or where there is no NVIDIA GPU, it installs the compression program from the release's tarball (checked
    against its sha256) and says that `glyd run` needs Linux with an NVIDIA GPU. `glyd doctor` reports such a machine
    without failing.
  - The Rust `glyd` passes `run`, `serve`, `doctor` and `login` to the Python tool (`GLYD_FORWARDED` guards against
    loops) and says plainly where there is none; a file named run is `./run`. The Python `glyd` finds a compression
    program that is a wrapper script, skips an empty PATH entry, and where there is none hints `cargo install --git
    ...`.
  - In the terminal chat, control and escape sequences in a model's text are removed. An error, a connection closed
    mid-answer, a cut message and an empty answer are said in plain words and the question is kept. Ctrl-C while the
    model loads stops everything it started. A full disk, an interrupted download, a float16 checkpoint (refused
    before its download), a bad `glyd login` token, two GPUs of which one is too old, MIG, a context longer than the
    model's window and a port that is not a number each end in a message with the next step.
  - A machine with no GPU says what `glyd run` needs even with input at its end (a CI step's, cron's). A SIGTERM or
    SIGHUP during a download ends it at once. Several keys after `-- --api-key` are refused. A MIG-enabled GPU is not
    picked ahead of a usable one.
  - Open WebUI's commands no longer set `WEBUI_AUTH=False`: the first account made on its first visit is the
    administrator, and `CORS_ALLOW_ORIGIN` is set to the page's own addresses (its default, `*`, handed the
    administrator's token to any page's script where the login is off: measured on 0.11.4). A short note in the README
    says what the no-login setting leaves open, DNS rebinding included, which Open WebUI does not check for.
- vLLM serves Glyd: `pip install "glyd[vllm]"`, then `vllm serve MODEL --quantization glyd`.
  - The `glyd` package's entry point for vLLM (`vllm.general_plugins`) loads the plugin where vLLM is 0.30, the
    release it is tested with (`glyd[vllm]` pins `vllm>=0.30,<0.31`). With another it logs one line and loads nothing,
    and `--quantization glyd` stops with why.
  - A bf16 checkpoint's Linears are packed on the GPU as each layer loads (the peak: the packs and a layer). A
    checkpoint missing a piece of a layer (a merged qkv's k, an expert's up) is refused, naming the layer and the
    piece, where vLLM's bf16 runs it with that memory never written. A model whose packs cannot fit the GPU's free
    memory is refused before it loads, with the numbers.
  - A glyd save loads as saved. Saves of Qwen3 and Llama models are checked; another family's is refused, naming its
    bf16 source.
  - torch.compile takes each product as one op (`glyd::vllm_linear`) and vLLM's CUDA graphs capture it. Embeddings,
    norms, attention and the KV cache stay vLLM's. vLLM sizes its KV cache after the weights load, so the memory the
    packs save becomes KV cache.
  - The options go in `--additional-config '{"glyd": {...}}'`, a checkpoint's `quantization_config`, or `GLYD_LAYOUT`,
    `GLYD_EXACT` and `GLYD_VERIFY`; another key, or a flag not true or false, is refused. `layout` is `auto` by
    default, which picks the layout for the GPU. `verify` also checks a save's other tensors by their sha256, and a
    save packed again in the other layout against the save. A change of layout, mode or checkpoint never loads
    another's compiled graph ([gpu/vllm/README.md](gpu/vllm/README.md)).
  - The plugin ran on an L4, an A10, an A100 SXM4 40 GB, a GH200, an H100 SXM and two RTX A6000s (the checks and
    benches below). Any GPU from Ampere loads it. An H100 PCIe, an H200, an A100 80 GB, an A800 and Blackwell were not
    run: they take the library's settings for their class, not measured through vLLM.
- Measured with `vllm bench serve`, bf16 against Glyd at the same `--gpu-memory-utilization 0.9`, servers warm, 1,024
  tokens in and 256 out, v0.25.1's library:
  - On an L4, an A10 and an A100 40 GB (Qwen3-8B; Qwen3-14B too on the A100), Glyd held 1.14-1.89x bf16's KV cache and
    served 1.19-1.65x its requests a second saturated, the first token 25-57% sooner. At low load (1 request a second,
    0.25 on the L4) each token came 10-21% sooner there, and the first 16-18% later. Saturated, each token took 30-42%
    longer on the L4 and A10, and was within 4% on the A100.
  - On a GH200, Qwen3-8B and Qwen3-32B: 1.04x and 1.66x the KV cache, but 0.92x and 0.88x the requests a second
    saturated (Hopper's gap is not profiled yet), each token at saturation 9% and 82% slower, and at low load the
    first token 4% and 28% later.
  - On an H100 SXM, Qwen3-30B-A3B (a mixture of experts, one GPU; the v0.26.0 candidate's library, whose behavior for
    the plugin is v0.25.1's): 2.11x the KV cache and 0.95x the requests a second saturated, where bf16 filled its KV
    cache (104 requests running, 152 waiting) and Glyd ran 220 at the most. At saturation the first token was 74%
    sooner and each token 118% slower; at low load the first token 39% later and each token 8% later.

  Logs: [benchmarks/gpu/l4-vllm-m5-2026-09-30](benchmarks/gpu/l4-vllm-m5-2026-09-30),
  [vllm-m3-a10-2026-09-30](benchmarks/gpu/vllm-m3-a10-2026-09-30),
  [vllm-m3-a100-40gb-2026-09-30](benchmarks/gpu/vllm-m3-a100-40gb-2026-09-30),
  [vllm-m3-gh200-2026-09-30](benchmarks/gpu/vllm-m3-gh200-2026-09-30),
  [vllm-m6-h100-2026-10-01](benchmarks/gpu/vllm-m6-h100-2026-10-01).
- Checked against vLLM's own bf16 (`gpu/vllm/check_vllm.py`): Qwen3-1.7B, Qwen3-4B-Instruct-2507, Qwen3-8B,
  Yi-1.5-6B-Chat (the Llama architecture), granite-3.1-3b-a800m-instruct (a mixture of experts) and
  Qwen2.5-1.5B-Instruct (Linears with biases) on an L4, and Qwen3-8B on an A10, an A100, a GH200 and an H100 SXM.
  Every pack decoded to its weights bit for bit, and every product was within 6.2e-3 of the same product on the exact
  weights. The fused tokens were within vLLM's own bf16 noise, bf16 eager's against bf16 with CUDA graphs: fed the
  1,536-token continuation bf16 generated, Glyd ranked 0.988-0.997 of its tokens first and bf16 eager 0.987-0.995.
  Glyd's share was at or above bf16 eager's for 18 of the 20 model, GPU and layout pairs, and 0.13 and 0.20 points
  under it for the other two.
- `exact` in vLLM runs vLLM's own bf16 product on the exactly rebuilt weights.
  - With `--enforce-eager` the logits are vLLM's bf16 eager's bit for bit on every model and GPU checked (above).
    Compiled, they are compiled bf16's in inductor's deterministic mode (`--compilation-config
    '{"inductor_compile_config": {"deterministic": true, "combo_kernels": true, "benchmark_combo_kernel": false}}'`).
    Without that mode compiled logits, bf16's too, differ from one process to the next, so `exact` compiled is refused
    without it.
  - It is refused compiled too where a packed Linear has a bias (Qwen2.5's q, k and v); eager, those give bf16's bits
    (Qwen2.5-1.5B-Instruct on an L4).
  - The mode cost nothing measurable on an L4 (Qwen3-8B's tokens/s at 1, 8 and 32 sequences within 1%, bf16's and
    Glyd's), and in it fused Glyd compiled is the same from one run to the next too.
  - Fused products are refused under `VLLM_BATCH_INVARIANT`, which asks for every product's bits not to depend on the
    batch.
- Speculative decoding in vLLM runs on Glyd's packs. On an L4, Qwen3-8B, one user, greedy: Glyd with an EAGLE-3 draft
  (`RedHatAI/Qwen3-8B-speculator.eagle3`) made 55.0 tokens a second on prompts that edit a given text or code and 38.9
  on chat, against bf16's 16.6 and 16.7 without speculation, and 43.3 and 30.3 with the same draft (which did not fit
  the L4 at vLLM's default memory settings, bf16's weights and the draft leaving no KV cache); n-gram prompt lookup,
  35.5 and 22.1. With speculation, exact eager gave bf16 eager's tokens, 10 of 10 requests with each draft.
  Speculation's greedy tokens can differ from plain decoding's, in vLLM's bf16 too (eager, 5 and 7 of 10 requests
  parted with n-gram and EAGLE-3). Under `VLLM_BATCH_INVARIANT=1` they are plain decoding's, bf16's and exact mode's
  alike (10 of 10). vLLM keeps an EAGLE-3 draft bf16 under `--quantization glyd`; with `"quantization": "glyd"` in
  `--speculative-config` it is packed too, sized by its own config
  ([benchmarks/gpu/l4-vllm-spec-2026-09-30](benchmarks/gpu/l4-vllm-spec-2026-09-30)).
- Mixtures of experts in vLLM (its fused MoE layer): each layer's experts are packed together and captured in vLLM's
  CUDA graphs. With `exact`, the experts run through vLLM's own Triton MoE kernel; it is refused where vLLM picks
  another kernel for bf16's experts. Experts with biases, activations other than SiLU, expert parallelism, or sizes
  the packing does not suit stay bf16, with a warning.
  - Long prompts take a faster path: granite-3.1-3b-a800m-instruct's layer on an L4 took 0.97x the time of the path
    before at 1,152 tokens and 0.65x at 8,192, and a prompt of 4,096 tokens 19% less GPU time a step. It uses extra
    memory for one layer's experts. `GLYD_MOE_DECODE_MIN` moves the prompt length from which it is taken (-1: never)
    ([benchmarks/gpu/l4-vllm-moe-routes-2026-09-30](benchmarks/gpu/l4-vllm-moe-routes-2026-09-30)).
  - granite-3.1-3b-a800m-instruct on an L4 passed every check, exact eager and compiled in the deterministic mode
    bf16's bits ([benchmarks/gpu/l4-vllm-m4-2026-09-30](benchmarks/gpu/l4-vllm-m4-2026-09-30)).
  - Over two RTX A6000s, tensor parallel (each rank packs its shard): Qwen3-8B's and granite's checks passed but one,
    where exact compiled was refused in the workers and the check did not see Glyd's message; Qwen3-30B-A3B's, all 5,
    exact eager bit for bit. Serving Qwen3-30B-A3B there, Glyd held 1.67x bf16's KV cache and served as many requests
    a second at 1 a second, each token 7% sooner, but from 4 a second 0.87-0.88x, each token 28-59% later
    ([benchmarks/gpu/vllm-m4-2xa6000-2026-09-30](benchmarks/gpu/vllm-m4-2xa6000-2026-09-30)).
- Refused at start in vLLM, with why (over several GPUs too, Glyd's message is the error the user sees): dual-batch
  overlap (`--enable-dbo`), LoRA, weight offloading and sleep mode; a glyd save over several GPUs, one with a mixture
  of experts' packs, or one of a family other than Qwen3's and Llama's (their bf16 checkpoints load); exact under
  torch.compile where a packed Linear has a bias; fused products under `VLLM_BATCH_INVARIANT`.
- Long prompts are faster on an A100 SXM4 40 GB, a GH200 and an H100 SXM (measured; opt-in in the C API, below). One
  forward pass as a ratio of v0.25.1's time, the same model and prompt in one process, the median of 3 rounds each way
  in turn (`e2e.py --prefill --merge --without-split --rounds 3`; each ratio is the median of the rounds' own ratios,
  so not quite the ratio of the times' medians: 101.2 ms at 1024 against 115.1 is 0.879; bf16's 93.0):

  | GPU            | Model     |   769 |  1024 |  2048 |  4096 |  8192 |
  | :------------- | :-------- | ----: | ----: | ----: | ----: | ----: |
  | A100 SXM4 40GB | Qwen3-8B  | 0.899 | 0.876 | 0.959 | 0.971 | 0.994 |
  | A100 SXM4 40GB | Qwen3-14B | 0.845 | 0.864 | 0.916 | 0.947 | 0.968 |
  | GH200          | Qwen3-32B |       |       | 0.909 | 0.940 | 0.952 |
  | GH200          | Qwen3-8B  |       |       | 0.978 | 0.994 | 0.994 |
  | H100 SXM       | Qwen3-14B |       |       | 0.893 | 0.937 | 0.938 |

  (Columns are prompt lengths in tokens; empty cells were not measured in this comparison. The H100 SXM was run with
  `GLYD_SPLIT_MIN=2048`.)
  - It is used for 12-bit prompts where it was measured faster. It is not taken for Qwen3-8B at 8192 tokens on the
    A100 (0.994) or for Qwen3-8B on the GH200, which gained 2.2% at 2048 alone. At 1024 tokens on the GH200 it was
    slower (1.106 and 1.012 of v0.25.0's time).
  - An H200, an H100 NVL and the PCIe cards keep v0.25.1's behavior until a session measures them, and nothing past
    8192 tokens takes it, not measured. It is taken on an A100 SXM4 80 GB or an A800 SXM4 too, not measured there.
    Never on a MIG slice.
  - It needs extra GPU memory: 600 MiB for Qwen3-8B, 1.0 GiB for 14B, 1.5 GiB for 32B, and a 32 MiB cuBLAS workspace.
    Measured on an A100-SXM4-40GB, a GH200 480GB and an H100 80GB HBM3, the SXM5.
  - Its results are the same bits run to run and prompt to prompt within a process, but not bit for bit bf16's, so
    `exact=True` never takes it.
  - Where it cannot run (the JIT build, a driver older than CUDA 12.5 (with a warning), too little memory, a CUDA
    graph capture, a torch.compile graph) a prompt takes the path before it. `GLYD_SPLIT_MIN=-1` turns it off;
    `GLYD_SPLIT_MIN`, `GLYD_SPLIT_MAX` and `GLYD_SPLIT_SMS` change the prompt lengths it covers and the part of the
    GPU it uses.
  - A stress check (`gpu/split_stress.py`, in test_gpu.py quick) of every Qwen3 layer's matrices, 0.6B-32B, at
    769-4096 tokens, 36 passes: 16,512 products the same bits across layers and passes, within 1e-2 of fp32, on an L4,
    an A100, a GH200 and an H100 SXM ([benchmarks/gpu/option2-2026-09-29](benchmarks/gpu/option2-2026-09-29), the H100
    SXM's in its `h100-sxm-measure`).
- C API version 7 adds `glyd_gpu_ring_create`, `_destroy`, `_split`, `_reset`, `glyd_gpu_mma12_ring_queue`,
  `glyd_gpu_mma12_ring_linear` (with the caller's cuBLAS as `glyd_gpu_blas`: the library does not link it),
  `glyd_gpu_mma12_unpack_split`, the route `GLYD_GPU_ROUTE_SPLIT`, `glyd_gpu_mma12_split_sms`, the GPU classes
  `GLYD_GPU_PCIE`, `GLYD_GPU_GH200` and `GLYD_GPU_H100` (an H100 NVL and an H200 stay 90), and `GLYD_GPU_WITH_SPLIT`
  (a GPU code's routes with SPLIT). The route is opt-in: only a code with that flag gets it, which the glyd package's
  Linears ask for where it can run. `glyd_gpu_*_linear`'s own route (-1) never takes it, so the C API's other callers
  (the glyd-gpu crate, the vLLM plugin) keep v0.25.1's behavior. v0.25's libraries (version 5) and builds from before
  the route was opt-in (version 6) are refused by this package and the glyd-gpu crate (`Route::Split`, `PCIE`,
  `GH200`, `H100`, `WITH_SPLIT`, `Library::split_sms`; the ring declared, not wrapped yet).
- `gpu/e2e.py --without-split` times a prompt again with the route off in the same process (`--rounds N`: N times each
  way in turn). `--breakdown` gives a pass's host time and its GPU time by kind of kernel.
- `gpu/respond.py`: how fast a model responds through `generate()`: its time to first token for prompts of 128-8192
  tokens, its tokens a second at 1, 8 and 32 sequences, and a chat and a long-document mix, in four modes a process
  each: bf16 eager, bf16 compiled (`fast_generate`, the same calls compiled as Glyd's default), Glyd's default and
  `exact=True`. Glyd's default is compared with bf16 compiled alone, the same path.
  - Its tokens a second over bf16 compiled's at 1 / 8 / 32 sequences, on a GH200 (v0.25.1): Qwen3-8B 1.06 / 1.21 /
    1.20x, Qwen3-32B 1.19 / 1.33 / 1.30x. On an L4 (Qwen3-8B, v0.25.1's behavior) 1.27 / 1.27 / 1.22x. On an A10
    (Qwen3-8B, v0.25.1) 1.31 / 0.94 / 1.11x (at 8 sequences, both eager, the GPU busy 68-73% of each second under Glyd
    against 96-98% under bf16), where Qwen3-14B's Glyd default ran at 20.54 GB and its bf16 did not fit. On an A100
    SXM4 40 GB at v0.25.1's behavior (before this release's long-prompt path) Qwen3-8B 1.23 / 1.05 / 1.04x and
    Qwen3-14B 1.27 / 1.05 / 1.05x, the chat mix's total 0.82x and 0.79x.
  - On the GH200 greedy tokens did not repeat from one identical call to the next at 8 and 32 sequences, bf16 eager's
    own included, nor compiled at one sequence, so no mode's tokens there are compared with bf16's; where bf16 eager's
    repeated, exact's were its tokens. The job that traced it (`benchmarks/gpu/repro-2026-09-30`, an H100 SXM,
    Qwen3-8B eager, v0.25.1's tree) found PyTorch's cuDNN attention kernel the first call to differ between two
    identical `generate()` calls, at 8 and 32 sequences, bf16 and exact alike: at 32 sequences 4 of 154 attention
    calls and none of 1,105 matrix products gave another output. At 1 sequence nothing differed.
  - `resp_job.sh` runs the modes unattended in 35 minutes on a GH200 (Qwen3-8B and 32B), an A100 or an A10
    ([benchmarks/gpu/respond-2026-09-29](benchmarks/gpu/respond-2026-09-29)).
- `fraction` in vLLM (`--additional-config '{"glyd": {"fraction": 0.5}}'`, or `GLYD_FRACTION`) packs only that share
  of the decoder layers and leaves the rest as vLLM runs them without a quantization, so that a server can keep some
  of the memory the packs save and skip some of the rebuild each step costs. The layers packed are floor(L f) of L,
  spread evenly over the depth, a layer's Linears together; 0 is vLLM's bf16 (nothing packed, Glyd's library not
  loaded) and 1, the default, every layer as before. A value outside 0 to 1 is refused, and a glyd save takes only 1.
  - On an L4 with Qwen3-8B (`vllm bench serve`, servers warm, the smallest layout, `mma`; "at once" is every request
    sent at once), against bf16's 0.75 requests a second at once, each token 111.2 ms at once and 100.2 ms at 1
    request a second, and the first token 3,351 ms at 1 request a second:

    | Fraction                            |      0 |    0.5 |      1 |
    | :---------------------------------- | -----: | -----: | -----: |
    | Weights, GiB                        |  15.27 |  13.65 |  11.83 |
    | KV cache, tokens                    | 27,024 | 37,904 | 51,040 |
    | Requests a second, at once          |   0.76 |   0.96 |   1.04 |
    | Each token, at once, ms             |  111.0 |  125.4 |  152.4 |
    | Each token, 1 request a second, ms  |  100.8 |   99.9 |   95.6 |
    | First token, 1 request a second, ms |  3,539 |  1,095 |    772 |

    ([benchmarks/gpu/l4-vllm-fraction-2026-09-30](benchmarks/gpu/l4-vllm-fraction-2026-09-30)).
  - On an 80 GB H100 SXM, Qwen3-32B's bf16 weights leave room for only 32,320 tokens of KV cache, and every packed
    fraction served more requests a second than bf16 (`vllm bench serve`, 192 prompts, 1,024 tokens in and 256 out,
    every request sent at once, each server started cold):

    | Fraction            |      0 |   0.25 |    0.5 |   0.75 |      1 |
    | :------------------ | -----: | -----: | -----: | -----: | -----: |
    | Weights, GiB        |  61.03 |  58.20 |  54.89 |  51.59 |  48.27 |
    | KV cache, tokens    | 32,320 | 43,872 | 57,392 | 70,912 | 84,528 |
    | Requests a second   |   2.51 |   2.85 |   3.54 |   3.50 |   3.91 |
    | Against bf16's 2.51 |        |  1.14x |  1.41x |  1.40x |  1.56x |

    The first token came after 33.0 s on average for bf16 and 20.0 s at fraction 1, each token 36.6 ms and 56.8
    ([benchmarks/gpu/l4-vllm-fraction-2026-09-30/h100-sxm](benchmarks/gpu/l4-vllm-fraction-2026-09-30/h100-sxm)).
  - `check_vllm.py --quick --fraction 0.5` passed all 18 checks on the L4: every pack decoded bit for bit, exact eager
    and compiled (deterministic mode) bf16's bits; fraction 0 in eager gave bf16 eager's tokens, logprobs and
    prompt_logprobs bit for bit; on granite-3.1-3b-a800m-instruct (a mixture of experts) 16 of 32 layers, their
    experts too, were packed, all 7 checks passed. Not measured yet on a GH200, where fraction 1 served 0.88x bf16's
    requests a second with Qwen3-32B.
- The local-chat quickstart (gpu/vllm/README.md) runs on a machine with no CUDA toolkit, and its load no longer fills
  the log. The command now sets `VLLM_USE_FLASHINFER_SAMPLER=0`, so vLLM samples with PyTorch and Triton instead of
  FlashInfer's sampler, which needs nvcc (the same tokens a second for 1 and for 8 users, and the same draws as
  FlashInfer's, which `flashinfer-jit-cache` provides without nvcc), and the plugin logs one warning at start where it
  finds no nvcc and the sampler on, naming both, and sets nothing. Loading no longer logs allocator warnings: on an L4
  held to the 14.48 GiB an RTX 4080 SUPER with a desktop leaves free, loading Qwen3-8B logged 0 allocator warnings,
  left 669 MiB free at the least, and held 11.39 GiB of weights and a KV cache of 1.83 GiB, 13,280 tokens; one user's
  tokens a second were 21.25 and 21.00 (greedy and top-p). `check_vllm.py --brief` passed on Qwen3-1.7B and
  granite-3.1-3b-a800m-instruct. The quickstart's Open WebUI runs by `uvx` or Docker, pinned to 0.11.4 with
  `ENABLE_PERSISTENT_CONFIG=False` (its connection is otherwise the first start's, and a later start with the
  variables lists no models), and the server takes `--enable-auto-tool-choice --tool-call-parser hermes
  --reasoning-parser qwen3` (Open WebUI's chats carry tools). `gpu/vllm/acceptance.sh` runs the quickstart's blocks
  from nothing, in a container with no nvcc, the GPU memory of a 16 GB card, a chat through the OpenAI API and through
  Open WebUI both ways, and passed
  ([benchmarks/gpu/l4-quickstart-2026-09-30](benchmarks/gpu/l4-quickstart-2026-09-30)).

## v0.25.1 — 2026-09-29

- Hopper: rebuilding the weights for long prompts and `exact=True` (steps and prompts, and a mixture of experts' exact
  mode) on the 12-bit layout is faster than in v0.25.0, which took 3.0-6.0% longer than before v0.25.0 (H100 SXM,
  GH200). On a GH200, the time to rebuild layer 10's weights of Qwen3-8B, 14B and 32B, over the time before v0.25.0:
  0.964-0.975 (v0.25.0 1.035-1.060). Measured on a GH200 alone; the H100 SXM, where v0.25.0's 3.0-5.8% was measured,
  and the H100 PCIe were not run again. GeForce Ada's, an A10's and an L40S's prompts and every other GPU keep
  v0.25.0's behavior (on an L4 this release's rebuild took 1.6-1.9% longer than v0.25.0's; the others were not
  measured). The same bits: the self-test and xcheck.py pass on the GH200, and on the L4 with this release's library
  ([benchmarks/gpu/decode-fix-2026-09-29](benchmarks/gpu/decode-fix-2026-09-29)).
- L4: long prompts are faster. Qwen3-8B, one forward pass, over bf16's time in the same run at 1024 / 2048 / 4096 /
  8192 tokens: `mma` +25.5 / +11.3 / +8.4 / +4.7% (were +27.8 / +31.0 / +37.9 / +98.4%), `mma12` at 4096 / 8192 +9.3 /
  +4.4% (were +19.8 / +105.0%); shorter prompts as before. The time to the first token through `generate()` moves with
  the pass.
  - `mma`, the smallest layout, stays the L4's default (33% less memory). For the fastest short prompts, at 25% less,
    load with `layout="mma12"`: its prompts took 5-20% less time than `mma`'s to 1536 tokens, and about the same from
    1792 (Qwen3-8B and Qwen3-4B-Instruct-2507).
  - The L4 is a GPU class of its own in the library's GPU codes (`GLYD_GPU_L4`; the glyd package and the glyd-gpu
    crate have it too, `L4`), so the L40 and RTX 6000 Ada, which share its compute capability and were not measured,
    keep their previous behavior. `GLYD_DEC_MIN` still moves the 12-bit layout's long-prompt point on any GPU
    ([benchmarks/gpu/l4-routes-2026-09-29](benchmarks/gpu/l4-routes-2026-09-29)).
- L40S: long prompts are faster. Qwen3-8B on an AWS g6e.xlarge, one forward pass, over bf16's time in the same run at
  1024 / 2048 / 3072 / 4096 / 8192 tokens: `mma` +30.3 / +11.9 / +8.0 / +10.9 / +3.8% (were +38.1 / +41.2 / +37.3 /
  +39.4 / +35.0%), `mma12` at 2048 / 3072 / 4096 / 8192 +12.9 / +7.7 / +11.3 / +3.9% (were +15.5 / +14.9 / +20.0 /
  +18.2%); shorter prompts as before. Qwen3-8B's working memory there is 0.40 GB (was 0.27).
  - The L40S's `mma12` prompts took 17.3 / 18.1 / 12.8 / 4.0% less time than `mma`'s at 512 / 768 / 1024 / 1536 tokens
    (at 512 tokens -0.3% over bf16's time, `mma`'s +20.6%) and were within 0.9% of them from 2048 tokens. `mma` stays
    its default (33% less memory; `layout="mma12"` for 25%).
  - The L40S is a GPU class of its own (`GLYD_GPU_L40S`; the package and the crate have it too), so the L40 and RTX
    6000 Ada keep their previous behavior until measured.
- `glyd_gpu_*_route` at M = INT64_MAX tokens gives the route before it, as `last` says (with `GLYD_WG_MAX` below
  INT64_MAX - 1; v0.25.0 gave another route there, on every GPU).
- This release's code (dc490e4) built by build_lib.sh and checked on an L4: the self-test, xcheck.py, check_capi.py
  (also with `GLYD_DEC_MIN` 1000 and 3000), test_gpu.py and the crate's tests pass
  ([benchmarks/gpu/decode-fix-2026-09-29/l4-head](https://github.com/surya-koritala/Glyd/tree/7d9e9be6f33395f39421304b33c9f8481afa5962/benchmarks/gpu/decode-fix-2026-09-29/l4-head)).

## v0.25.0 — 2026-09-29

- The 12-bit layout (`mma12`) is faster on the H100 and the A100 and the same on the A10 and L4, with the same size
  (12.04-12.07 bits a weight) and the same bits, so every product is bit for bit what it was. On an L4, an A10, an
  A100 SXM4 40 GB, an H100 PCIe and an H100 SXM (2026-09-29) every output of the layout was main's bits; models'
  logits and greedy tokens, fused and exact, the same as main's: Qwen3-1.7B and granite-3.1-3b-a800m-instruct on the
  L4, A10, A100 and H100 PCIe (round 1), and those two and Qwen3-4B-Instruct-2507 on an L4 on the release candidate.
  - A layer's time against the 12-bit layout's before, main's library and this release's in one process (layer 10 of
    Qwen3-8B with 4B-Instruct-2507, 14B or 32B, two runs each):
    - H100 SXM: faster, 0.933-0.969 at 32-1024 tokens (3.1-6.7% less; the H100 PCIe 0.942-0.991) and 0.986-0.992 at
      1-16.
    - A100: faster, 0.923-0.987 at 64-128 (5.5-7.7% less at 128), 0.969-0.982 at 256-768 (prompts) and 0.976-0.998 at
      1-16; the same to 1.5% faster at 32 (0.985-1.003).
    - A10 and L4: the same, 0.989-1.010 at 1-1024 tokens.
    - Slower: the weights' rebuild that exact mode and long prompts use takes 3.0-5.8% longer on the H100 SXM and
      1.0-1.6% on the A10 (0.998-1.012 on the A100, the same on the L4). Long prompts on Hopper and the A100, the
      A10's 12-bit prompts (their time not measured) and every step of exact mode use it
      ([benchmarks/gpu/splitbyte-2026-09-29](benchmarks/gpu/splitbyte-2026-09-29)).
  - The 12-bit layout's bytes change from v0.24's (v0.19-v0.24): the library refuses v0.24's packs with
    `cudaErrorInvalidValue`, so pack those models again. The C API is version 5. A glyd-v3 save of the layout before
    is refused as it loads: save it again. `glyd pack --layout mma12` packs the new layout on the CPU, byte for byte
    as `pack_mma12` does, and `glyd verify` decodes it and refuses a 12-bit pack of the layout before. `mma` saves
    (glyd-v1, glyd-v2) are unchanged.
- Saved models in the 12-bit layout too: `glyd.save_pretrained(model, path, layout="mma12")`, `python -m glyd.gpu pack
  MODEL OUT --layout mma12` and `glyd pack MODEL OUT --layout mma12` write glyd-v3, which an A10, A100 or H100 runs as
  saved. `from_pretrained` loads it as saved where the 12-bit layout is the one (else it decodes and packs again, as
  it does an `mma` save there; glyd 0.24 and before refuse glyd-v3 by its format). The same bytes from Rust and Python
  for the five models of `glyd pack` below, and `glyd verify` reads it. Loaded for the 12-bit layout on an RTX 4080
  SUPER (`layout="mma12"`, warm cache, three fresh processes each; measured on the 12-bit layout before this release's
  change, whose load is the same): granite-3.1-3b-a800m-instruct in 0.49 s against 1.32-1.33 s from the `mma` save and
  1.44-1.45 s from the bf16 checkpoint, Qwen3-4B-Instruct-2507 in 0.84-0.85 s against 1.76-1.77 and 1.95, Qwen3-8B in
  1.17-1.18 s against 3.14-4.55 and 3.54-3.61 (2.7-3.9x faster than packing again; 2.1-3.9x across the three), its
  peak 0.56 GB below the `mma` save's (14.34 GB against 14.90)
  ([benchmarks/gpu/rtx4080s-rust-2026-09-28](benchmarks/gpu/rtx4080s-rust-2026-09-28)).
- glyd.json holds the sha256 of every tensor saved as it is too (its `tensors`: the norms, biases, an embedding or
  output layer not packed), and verify checks them: `python -m glyd.gpu verify`, `from_pretrained(verify=True)` and
  `glyd verify` check every tensor against a pack's or one of those sha256, and each pack's tensors against their own,
  so a byte changed outside the packs, bytes appended to a shard or a renamed member fails them. Both savers write it
  alike; glyd 0.24 and before load such a save and ignore it. `verify` requires the map where the save's format or
  glyd says it is there (glyd-v3, and a save of glyd 0.25 on); a save of glyd 0.24 or before verifies as it did, its
  other tensors counted unchecked.
- `glyd pack MODEL OUT` and `glyd verify PATH` in the Rust CLI: a bf16 checkpoint (a directory, or a repo in the local
  Hugging Face cache) packed on the CPU and saved as glyd-v1 (glyd-v2 with a mixture of experts' packs, glyd-v3 in the
  12-bit layout) with no Python, PyTorch or GPU, byte for byte as `python -m glyd.gpu pack` saves it.
  - Every file of Qwen3-0.6B, 1.7B, 4B-Instruct-2507 and 8B and of granite-3.1-3b-a800m-instruct is Python's in both
    layouts (their sha256; glyd.json, the shards, the index; three rounds each), each pack decoded back and checked as
    it is made, and each Rust save verified by `glyd verify` on the CPU and the GPU and by `python -m glyd.gpu
    verify`.
  - On the AWS dev machine (a g6.4xlarge: 16 vCPUs of an AMD EPYC 7R13, an NVIDIA L4), 16 threads, three rounds each,
    `glyd pack` takes 18.5-23.0 s in `mma` and 22.8-24.6 s in `mma12` for Qwen3-8B's 16.4 GB (at the pace of the disk
    its shards were written to: 3.2-6.4 of the CPUs busy). About 15 GB of memory at most for a model of several
    shards; Qwen3-8B measured 11.4-13.6 GB peak RSS on 16 threads.
  - `glyd verify` checks a save as `python -m glyd.gpu verify` does, its packs decoded on the CPU or (`--device
    cuda:0`) on the GPU (Qwen3-8B's 253 packed tensors and 146 saved as they are in 11.6-11.8 s in `mma`, 8.8-9.0 s in
    `mma12`, on 16 threads) ([benchmarks/gpu/l4-rust-2026-09-29](benchmarks/gpu/l4-rust-2026-09-29)).
  - The families are written out as transformers 5.17 holds them: Qwen3, Qwen2, Llama, Mistral, Granite and GraniteMoe
    for now (tiny random checkpoints of each family save the same bytes too, both layouts), anything else refused with
    Python's command.
  - The commands are the `glyd-gpu` program's, under the Business Source License as the rest of the GPU code, which
    the glyd CLI runs (it ships beside glyd; a file named `pack` or `verify` is compressed as `./pack`).
- The `glyd-gpu` crate: Rust over the GPU library's C API, the library loaded at run time and refused where its API
  version is not the crate's. Each function is typed, device pointers and streams are the caller's, errors are
  `Result`, and it has no dependency but sha2 and links nothing at build time. Its examples decode a saved model's
  packs on the GPU against the bf16 checkpoint, bit for bit (`unpack.rs`, the C example in Rust), and multiply by
  `linear` (`linear.rs`); its `pack` module packs a matrix in either layout on the CPU, byte for byte as glyd.gpu's
  `pack_mma` and `pack_mma12` do on the GPU.
- The library now chooses which code a product for M tokens takes on a GPU, as glyd.gpu 0.24 chose it (a check against
  0.24's rule on eleven GPU codes, 0-5000 tokens, both layouts): `glyd_gpu_mma_route` and `glyd_gpu_mma12_route` give
  it (and the last token count that takes it).
  - `GLYD_WG_MIN`, `GLYD_WG_MAX`, `GLYD_MID_MIN` and `GLYD_DEC_MIN` (any GPU's 12-bit prompts) move the token counts
    at which a GPU changes code. They are read once a process, at the library's first use (when the first model is
    loaded or compressed; the prebuilt library and the JIT build each at their own): set them in the environment
    before that. A later change has no effect, nor has assigning glyd.gpu.model's `WG_MIN`, `WG_MAX`, `MID_MIN` or
    `DEC_MIN`, or a GLinear's `dec` or `mid`, which are gone (`GLYD_AHEAD_MIN` is read at import as before, and a
    GLinear's `ahead` can still be set, then `lin.step = lin._step()`). A value that is not a whole number fails the
    import of glyd.gpu.model, as it did at `int()` (the library alone takes it as unset).
  - A GPU's code is its compute capability plus a class where the name tells GPUs apart (`GLYD_GPU_GEFORCE`;
    `GLYD_GPU_A10`, an A10 and not an A10G, A40 or A6000).
  - `glyd_gpu_mma_linear` and `glyd_gpu_mma12_linear` run a route's kernel (where glyd decodes the matrix for cuBLAS,
    the prompt kernel, on every GPU; where K is not a multiple of 64, past 64 tokens (12-bit: also from `GLYD_DEC_MIN`
    tokens where that is lower), they refuse those routes with `cudaErrorNotSupported`: decode it there for a GEMM of
    your own; the Hopper route's done counters are at least 1024, as `glyd_gpu_mma12_gemm_wg`'s). The glyd package's
    Linears take their routes from the library and multiply by `linear` in their one C call, so every caller routes
    alike.
  - C API version 5 (4 with the 12-bit layout before this release: builds of main alone, refused). The same bits
    (check_capi on an RTX 4080 SUPER: 6975 calls through both hosts, bit for bit, and 220044 routes as 0.24's rule)
    and the same speed (`generate()` eager, before the merge with 0.24: main's package and library and these in turn,
    four rounds, RTX 4080 SUPER: Qwen3-1.7B and Qwen3-4B-Instruct-2507 at 1, 8 and 32 sequences in both layouts, each
    round -1.1% to +1.3% of main's, their means -0.3% to +0.5%).

## v0.24.0 — 2026-09-28

- `generate()` on a model from `glyd.from_pretrained` or `glyd.gpu.compress` runs compiled by default on PyTorch
  2.13.0 or later (measured on 2.14; below it, a 2.13 pre-release included, it stays eager, as in 0.23), as
  `generate(..., cache_implementation="static")` asks transformers to run it (a static cache, the forward under
  `torch.compile` with CUDA graphs): a step's host time goes. Tokens/s generating 128 tokens at 1 / 8 sequences on an
  RTX 4080 SUPER, plain `generate()`, each in a process of its own: Qwen3-1.7B 184.6 / 1260 (was 96.6 / 772; asked for
  with `cache_implementation="static"` 184.9 / 1260), Qwen3-4B-Instruct-2507 94.3 / 610 (was 75.1 / 563), Qwen3-8B
  55.3 / 386 (was 48.6 / 365), granite-3.1-3b-a800m-instruct 232.3 / 1547 (was 90.3 / 699).
  - The first call compiles: Qwen3-8B's took 17.5 s with PyTorch's compile caches empty, 6.7 s in a later process.
    Warm up with one short `generate()` before serving (a streamer's consumer waits through it).
  - Greedy tokens compiled can differ from 0.23's eager loop's, as a compiled bf16 model's can from its eager ones
    (the first 8 of 32 the same on Qwen3-0.6B, 17 on Qwen3-1.7B, 32 on granite-3.1-3b-a800m-instruct: check_api). In
    the default mode they can also vary within a process, between calls whose cache sizes compile differently
    (Qwen3-1.7B's first compiled call and its later ones, after a longer cache, shared 13 of 32 in one run of
    check_api: benchmarks/gpu/rtx4080s-fastloop-2026-09-28/checks-merge-6e17b3f). `exact=True` is never compiled and
    stays bit-identical to bf16.
  - `compile=False` (`from_pretrained`, `compress`) or `GLYD_COMPILE=0` runs it eager, as before; so do `exact=True`
    (its tokens are bf16's eager ones), a family transformers does not compile whole (its `_can_compile_fullgraph`), a
    model over several GPUs, and a transformers whose generation helpers are not as 5.17 has them (one warning at the
    load).
  - Only greedy and sampled calls compile: a call runs as transformers runs it if it uses several beams, an assistant
    or another assisted mode (prompt lookup, early exit, `use_mtp`), its own cache or a `cache_implementation`,
    `use_cache=False`, `return_dict_in_generate`, attentions or hidden states, `custom_generate`, or
    `disable_compile=True` (one call eager).
  - A call whose static cache would hold more positions in all (its sequences times the prompt and `max_new_tokens`,
    or `max_cache_len` where longer) than 1280 on a GeForce card and 2048 on another runs eager too;
    `GLYD_COMPILE_MAX` sets that limit.
  - Where transformers 5.17's static cache fails (bf16's too) it runs eager from the start: Llama 4, and a model with
    multi-head latent attention whose config has fewer key/value heads than heads (tiny DeepSeek V2, V3, Kimi Linear
    and AXK1 test models; the released checkpoints compile). A call whose forward fails to compile anyway runs again
    eager from its start, and so do the model's later calls, with one warning; any other error, out of memory
    included, is the call's own, and the next call compiles.
  - Several models compile in one process: ten Qwen3-0.6B models one after another all compiled, the second to tenth
    at 296.5-301.3 tokens/s against 99.6 eager. A model that has generated compiled is freed at `del`, as an eager
    one.
- Prompts on an A10 (`exact=True`'s excepted) are faster. Qwen3-8B, one forward pass, over bf16's time at 1024 / 2048
  / 4096 tokens: +10.0 / +5.2 / +2.6% (were +30.3 / +38.8 / +50.9%); shorter prompts as before (+2.2% at 128, +15.8%
  at 512). Qwen3-8B's working memory there is 0.40 GB (was 0.27). The A10G (its prompts at most +5.3% over bf16's to
  4096 tokens) and the L4, L40S and RTX 6000 Ada keep their previous behavior until measured.
- Prompts on Hopper are faster. On an H100 SXM (benchmarks/gpu/h100-hopper2-val-2026-09-28) Qwen3-8B's decoder layer
  (q, k, v and gate, up merged) takes 1.16 / 1.24 / 1.43 / 1.38x cuBLAS's time at 129 / 256 / 512 / 1024 tokens, and
  one forward pass over 1024 tokens (`gpu/e2e.py --prefill --merge`) 45.0 ms against bf16's 36.9; longer prompts as
  before. Before, in an earlier run (benchmarks/gpu/h100-hopper2-2026-09-28), Qwen3-8B's layer took 1.31 / 1.33 / 1.55
  / 1.64x there and its pass 48.4 ms (bf16 36.4).
  - A layer still takes more than cuBLAS's time (`gpu/README.md`).
  - Some prompt outputs differ from before in their last bits.
  - Generation steps took as long as before in the second run's CUDA 13 libraries (a median 0.0% apart, 1.1% less to
    2.2% more).
  - `GLYD_WG_MAX` sets the longest prompt that takes this path.
- On Hopper, the CUDA 12 library's (`libglyd_gpu_cuda12.so`, built with CUDA 12.8; the wheels load it for PyTorch
  built for CUDA 12) 12-bit products, the step products of v0.22.0 and v0.23.0 too, were slower than the CUDA 13
  library's; fixed. In the second run's libraries (on an H100 SXM, benchmarks/gpu/h100-hopper2-cu12-2026-09-28), the
  CUDA 12 library's products (Qwen3-8B's four and Qwen3-32B's o and gate_up at 17-1024 tokens) took a median 4% less
  time than before, up to 9%, Qwen3-8B's layer 2-8% less (1.58x cuBLAS's time at 1024 tokens before, 1.45x with the
  fix alone), as fast as the CUDA 13 library's (a median 0.3% apart); the CUDA 13 library's took as long as before (a
  median 0.2% apart), and every output was the same, bit for bit. As committed, the CUDA 12 library's Qwen3-8B layer
  takes 0.94-1.42x cuBLAS's time at 17-1024 tokens, within 0.8% of the CUDA 13 library's
  (benchmarks/gpu/h100-hopper2-val-2026-09-28).
- Validated as committed on an H100 SXM, with the library built for CUDA 12.8 and for CUDA 13.0
  (benchmarks/gpu/h100-hopper2-val-2026-09-28): the full self-test passed through both; the two libraries' 342 outputs
  of the Hopper path on the self-test's matrices (1-2100 tokens) are the same, bit for bit; and `gpu/e2e.py --exact`
  gives bf16's logits bit for bit through both, 8 of 8 tokens.
- `GLYD_DEC_MIN` (the prompt length from which the 12-bit layout's prompts take the long-prompt path) now applies on
  any GPU where it is set; unset, an A100 keeps its default and elsewhere there is none.
- The C API is version 3: `glyd_gpu_mma12_gemm_wg` takes at least 1024 done counters (as many as O / 64 where that is
  more); the package's calls always gave it that many.

## v0.23.0 — 2026-09-28

- Prompts on GeForce Ada (RTX 40) are faster, with the same bits. On an RTX 4080 SUPER a Qwen3-1.7B, 4B or 8B layer's
  products take 1.7-4.3% less time in `mma12` at 256-4096 tokens (Qwen3-4B-Instruct-2507's 1.03-1.04x cuBLAS's at
  512-4096, were 1.06-1.08x) and about 1-2% less in `mma`.
  - One forward pass (`gpu/e2e.py --prefill --merge`), `mma12`, Qwen3-4B-Instruct-2507 at 256 / 384 / 512 / 640 / 768
    tokens: 28.3 / 40.8 / 50.9 / 64.2 / 75.2 ms against bf16's 28.1 / 39.5 / 49.1 / 62.7 / 73.0 (were 28.9 / 41.9 /
    51.8 / 65.8 / 76.6); Qwen3-1.7B at 384 / 640 / 1024 / 1536: 18.6 / 26.8 / 42.5 / 63.6 against 18.7 / 25.9 / 42.3 /
    63.0 (were 19.2 / 27.8 / 43.5 / 67.8).
  - Generation to 64 sequences is unchanged; larger batches are faster (the same bits): at 128 sequences
    Qwen3-4B-Instruct-2507 makes 5069 tokens/s in `mma12` (were 4977; bf16 4702) and 4971 in `mma` (were 4950),
    Qwen3-1.7B 8845 and 8384 (were 8699 and 8368; bf16 8452).
- The GPU kernels as a library of their own, for engines in C, C++, Rust or any language with a C FFI, with no Python
  or PyTorch: every release carries `glyd-gpu-TAG-linux-ARCH-cudaN.tar.gz` for x86_64 and aarch64, CUDA 12 and 13 (the
  library, its header, the example below, `gpu/LICENSE` and a README with the example's build line), each with its
  `.sha256`. `gpu/glyd_gpu.h` declares the C API: its 37 functions, `GLYD_GPU_API_VERSION` (2), the arrays of each
  packed layout, the workspace queries, the stream and the return codes. `gpu/examples/unpack.c` reads a matrix of a
  model saved by `glyd.save_pretrained`, decodes it on the GPU with the library and checks it against the bf16
  checkpoint: on an RTX 4080 SUPER every one of Qwen3-0.6B's 112 packs (its 196 Linears, q, k, v and gate, up merged)
  decodes to the checkpoint's bits.

## v0.22.0 — 2026-09-28

- Prompts on Hopper are faster. On an H100 PCIe (`gpu/e2e.py --prefill --merge`), Qwen3-32B's prompts of 129 / 256 /
  384 / 512 tokens take 63.1 / 79.7 / 118.3 / 141.0 ms against bf16's 58.6 / 70.3 / 89.3 / 113.1 (were 131.1 / 142.1 /
  165.3 / 189.8), Qwen3-8B's of 384 / 512 tokens 33.2 / 39.2 against 25.9 / 32.6 (were 41.6 / 47.7); Qwen3-8B's pass
  at 129-256 tokens is mostly the host's launches, 22.3-28.7 ms in four runs against bf16's 22.7-51.5 (was 33.4-36.7).
  A layer's products take 1.12-1.97x cuBLAS's time at 129-512 tokens (weights read from memory). `GLYD_WG_MAX` sets
  the longest prompt that takes this path.
- Batched steps on Hopper are faster. On an H100 PCIe (`gpu/e2e.py --merge --profile`), a step's GPU time at 1 / 8 /
  32 / 64 sequences is 11.13 / 12.32 / 13.48 / 14.64 ms for Qwen3-8B against bf16's 12.30 / 13.34 / 14.57 / 15.66
  (were 11.24 / 12.45 / 15.55 / 17.57), 34.65 / 37.34 / 41.24 / 44.41 for Qwen3-32B against 42.44 / 44.40 / 47.10 /
  49.56 (were 34.75 / 37.48 / 44.97 / 49.84). A layer's products (weights read from memory) take 0.85x / 0.86x / 0.96x
  / 1.05x cuBLAS's time at 32 / 64 / 96 / 128 tokens for Qwen3-8B (were 0.92x / 1.00x / 1.20x / 1.20x), 0.78x / 0.81x
  / 0.95x / 1.08x for Qwen3-32B. Small steps are faster for some layers (Qwen3-8B's gate, up and down in layers 1-3):
  210 us at 8 tokens against 242-269 on an H100 PCIe, 2-3% faster on an RTX 4080 SUPER, the others as before; outputs
  bit for bit as before.
- The prebuilt library carries native Blackwell code (sm_100, sm_120) where nvcc has it (CUDA 12.8 on:
  `gpu/build_lib.sh` builds without it with an older nvcc). The Hopper code needs a build with sm_90a: one without it
  now traps on an H100, where it returned its output untouched; `GLYD_GPU_ARCH=sm_90` builds sm_90a.
  `glyd_gpu_mma12_gemm_wg` and its workspace query return cudaErrorInvalidValue for O under 64 (O = 0 returned
  success, launching nothing).
- Short prompts on GeForce Ada are faster. On an RTX 4080 SUPER (`gpu/e2e.py --prefill --merge`), Qwen3-1.7B's prompts
  of 128 / 256 / 512 tokens take 10.3 / 14.6 / 23.8 ms in `mma` and 10.2 / 14.5 / 23.7 in `mma12` against bf16's 10.8
  / 14.0 / 24.3 (0.21.0: 11.4 / 15.8 / 25.4 and 11.4 / 15.4 / 24.7), Qwen3-4B-Instruct-2507's 128 / 256 / 512 tokens
  19.4 / 29.3 / 52.6 and 19.2 / 28.7 / 51.5 against 20.7 / 28.0 / 49.1 (were 19.7 / 29.9 / 53.3 and 18.6 / 29.3 /
  52.0); at 384 tokens Qwen3-1.7B's 20.1 / 19.2 against 18.7 (were 23.9 / 23.2), Qwen3-4B's 44.4 / 41.7 against 39.5
  (were 50.2 / 49.3); the time to the first token with them (Qwen3-1.7B at 128 tokens 11.6 / 11.8 ms against 13.0 /
  12.8); generation as before. The C API's `glyd_gpu_mma_gemm_big` and `glyd_gpu_mma12_gemm_big` take a product's done
  counters, and `glyd_gpu_api_version` tells the C API's version (2; the package refuses a library of another).
- Products on two streams of one GPU at once no longer share done counters (a set a stream, as the workspace): a small
  product's outputs could come out wrong that way.
- Long prompts on GeForce Ada within 0.2-0.7% of bf16's time from 2048 tokens, 1.5-2.9% at 1024 (were 5-10% behind).
  On an RTX 4080 SUPER (`gpu/e2e.py --prefill --merge`), Qwen3-4B-Instruct-2507's prompts of 1024 / 2048 / 4096 tokens
  take 97.5 / 199.5 / 447.5 ms in `mma` against bf16's 96.1 / 198.3 / 445.4 (were 102.8 / 211.5 / 476.5), Qwen3-1.7B's
  43.3 / 86.8 / 186.0 against 42.1 / 86.6 / 185.3 (were 45.9 / 91.5 / 190.1), Qwen3-8B's 175.2 / 345.1 / 767.6 (were
  185.6 / 370.8 / 811.3); the time to the first token with them; generation as before. With `exact=True` the same
  path, the logits bf16's bit for bit. This long-prompt path is off on an H100, L4, L40S and RTX 6000 Ada until
  measured (`GLYD_AHEAD_MIN=513` turns it on there); on an A100 it is off, as measured: it was slower there.
- On an A100, batched steps and prompts are faster in `mma12`. On an A100-SXM4-40GB (`gpu/e2e.py --merge --fused
  --profile 16`), Qwen3-8B's GPU time a step at 32 / 64 / 128 sequences is 18.54 / 21.05 / 28.17 ms against bf16's
  21.12 / 21.13 / 25.71 (was 19.14 / 21.69 / 29.45), Qwen3-14B's 28.70 / 31.81 / 42.41 against 32.89 / 36.01 / 41.15
  (was 29.24 / 32.95 / 49.23); Qwen3-8B generates 2993.6 tokens/s at 128 sequences against bf16's 2997.4 (was 2747.0).
  Qwen3-8B's prompts of 128 / 512 / 1024 / 2048 / 4096 tokens (`--prefill`) take 40.9 / 64.5 / 112.0 / 195.7 / 368.2
  ms against bf16's 41.0 / 48.4 / 90.1 / 174.3 / 347.9 (were 44.9 / 68.3 / 120.7 / 237.1 / 489.2), Qwen3-14B's 512 /
  1024 / 2048 / 4096 tokens 111.7 / 198.6 / 347.5 / 653.7 against 82.7 / 151.2 / 291.1 / 584.6 (were 113.1 / 207.3 /
  416.6 / 855.4). `GLYD_DEC_MIN` sets the prompt length from which the 12-bit layout's prompts take the long-prompt
  path.
- `glyd.save_pretrained` saves a mixture of experts: glyd-v1 holds each layer's experts as one pack, and `glyd.json`
  the sha256 of each weight as the model holds it; such a checkpoint's format is glyd-v2, which glyd 0.21 refuses (a
  dense model's stays glyd-v1). `from_pretrained(path)` loads the packs as saved (the 12-bit layout packed again from
  them), `verify=True` checks every tensor, and `python -m glyd.gpu pack` and `verify` take one. On an RTX 4080 SUPER,
  granite-3.1-3b-a800m saves in 7.4 s to 4.66 GB of safetensors (6.60 GB in bf16) and loads from them in 0.4 s (3.4 s
  verified; 3.9 s from its bf16 checkpoint), its logits bit for bit the model packed as it loaded.
- Every mixture-of-experts family of transformers 5.17 packs its experts: the 54 whose Experts modules transformers
  runs through an experts implementation (run by `glyd`), and Llama 4, DBRX, Aria, JetMoE (its attention experts too),
  Step 3.7 and LongCat-Flash, each one op under `torch.compile` (but JetMoE, whose router calls `.tolist()`, as bf16's
  does not compile); Switch Transformers' and NLLB-MoE's experts are Linears, packed as such. Llama 4's experts stayed
  bf16 before: on an RTX 4080 SUPER a Scout MoE block at its real sizes (hidden 5120, 16 experts of 8192) takes 0.66
  ms at one token against bf16's 6.17, 1.91 against 6.26 at 8 and 6.40 against 23.4 at 512, its experts 2.70 GB
  against 4.03, exact bit for bit. `bindings/python/test_gpu.py` builds each family as a tiny model on the GPU: its
  experts packed, its logits as near fp32's as bf16's, exact bit for bit, saved and loaded in both layouts. Fixed
  along the way: models that failed before (Llama 4's embedding device, Gemma 4's pad row, HunYuan V4's output layer
  in fp32) and a model with packed experts that held reference cycles when let go of.
- `best_layout()` takes a mixture of experts into account: `mma` on an A10 as on Ada (on an A10 1-6% less GPU time a
  step than `mma12`, on an RTX 4080 SUPER 6-7%, and 10-11% smaller); `mma12` on an A100 and an H100.
- `gpu/sizes.py` counts granite's and Mixtral's checkpoint names for their experts (granite-3.1-3b-a800m: 3.22 B
  weights, was 0.20 B); `gpu/e2e.py` frees the Glyd model before bf16's profile (Qwen3-30B-A3B ran out of memory there
  on an H100).

## v0.21.0 — 2026-09-27

- `pip install "glyd[gpu]"`: the Linux wheels (x86_64 and aarch64, manylinux_2_28) carry the GPU library built for
  CUDA 12.8 and 13.0, `libglyd_gpu_cuda12.so` and `libglyd_gpu_cuda13.so`, the one for PyTorch's CUDA taken; no
  compiler and no checkout. The extra installs PyTorch 2.5+, transformers 5.17+, accelerate, safetensors and
  huggingface_hub
  ([bindings/python](bindings/python/README.md#on-the-gpu-a-models-weights-held-compressed-bit-for-bit)).
- `glyd.from_pretrained("Qwen/Qwen3-8B")`: transformers loads the checkpoint and every Linear's weight is packed on
  the GPU as it arrives (a quantizer registered as `glyd`), in `best_layout()`'s layout for the GPU, q, k, v and gate,
  up as one product each. On an RTX 4080 SUPER, Qwen3-0.6B's logits are nearer the fp32 model's than bf16's are (mean
  |difference| 0.030 against 0.041), Qwen3-1.7B's as near (0.034 against 0.032).
- `glyd.from_pretrained` packs a mixture of experts (OLMoE, granite MoE, Qwen3-MoE, Qwen3-Next, Gemma 4, GLM-4.5,
  Mixtral, gpt-oss ...), which stayed bf16 before; the experts run through `glyd`, an experts implementation
  registered with transformers.
  - On an RTX 4080 SUPER (`gpu/e2e.py --from-pretrained --baseline --prompts --merge --profile --prefill`),
    granite-3.1-3b-a800m holds 4.61 GB against 6.60 GB in bf16 and generates 90.0 tokens/s at one sequence against
    bf16's 71.1 and 600.9 at eight different prompts against 195.6 (a step's GPU time 5.0 ms against 6.8, and 9.5
    against 39.4); OLMoE-1B-7B, measured before the eager speedup below, holds 9.28 GB against 13.84 and generates
    135.6 against 99.1 and 607.9 against 142.1 (4.3 ms against 8.8, and 11.8 against 55.7). Prompts of 16 to 2048
    tokens take less time than bf16's (2048 tokens: granite 91.2 ms against 100.8, OLMoE 98.7 against 110.5).
  - Compiled (`generate(..., cache_implementation="static")` or `torch.compile(model.forward, mode="reduce-overhead",
    fullgraph=True)`), a packed Experts module is one op of the graph (`glyd::experts`), no graph break: granite
    generates 225.1 tokens/s at one sequence against bf16's 170.0 compiled the same way, and 909.9 at eight prompts
    against 175.2 (`e2e.py --compile`).
  - With `exact=True` the logits are bf16's bit for bit at every step (both models, one sequence and eight different
    prompts, 16 steps).
  - glyd-v1 (`save_pretrained`) holds no packed experts yet, and a model with them can't be copied or pickled
    (`copy.deepcopy`, `torch.save`): load it again.
- `gpu/e2e.py` takes every timing before its first profile (`--profile`): a profiler session slows each CUDA launch
  after it for the rest of the process, and Glyd, timed after bf16's profile, lost some 20% of its tokens/s
  (granite-3.1-3b-a800m at one sequence on an RTX 4080 SUPER: 68.6 against 85.4 without it); bf16's profile now runs
  last, on the model loaded again. `--prompts` generates for different prompts, not copies of one; `--from-pretrained`
  times the model `glyd.from_pretrained` loads; `--prefill` draws its tokens within the model's vocabulary.
- `exact=True`: every product multiplies as `nn.Linear` does, so the logits are bf16's bit for bit (Qwen3-0.6B and
  1.7B, 32 of 32 tokens as bf16's; `e2e.py --exact` the same for the scripts).
- `glyd.save_pretrained(model, path)` writes the glyd-v1 format (the packs as safetensors, `glyd.json` with every
  tensor's sha256), which `from_pretrained(path)` loads packed (Qwen3-1.7B in 2.52 GB of safetensors); `verify=True`
  decodes every tensor and checks it. `glyd.fit("Qwen/Qwen3-32B", gpu="48GB")` answers from the config and the
  checkpoint's metadata, the measured sizes where there are some; `python -m glyd.gpu fit|pack|verify`.
- The GPU code behind a C API (`gpu/build_lib.sh`), called through ctypes: Qwen3-8B generates 48.4 tokens/s at one
  sequence through the library as through a local build of the extension (363.6 and 363.4 at eight); on models under
  2B, 3-4% fewer.
- Compiled: `model.generate(..., cache_implementation="static")` (transformers' compiled forward) and
  `torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)` take every GLinear and GEmbedding as one op
  of the graph (`glyd::linear`, `glyd::embedding`), with no graph break: on an RTX 4080 SUPER Qwen3-1.7B generates
  187.8 tokens/s at one sequence against bf16's 151.9 compiled the same way (1282 against 1020 at eight),
  Qwen3-4B-Instruct-2507 95.2 against 73.9, Qwen3-8B 55.6.
  - Eager `generate()` is faster than bf16's where the host is the bottleneck too: a step's product takes 4.7 us of
    host time against `F.linear`'s 5.4 (8.7 before), a Qwen3-1.7B layer 295 us against bf16's 322 (344 before),
    Qwen3-1.7B 98.3 tokens/s against 88.9 (86.8 before), Qwen3-4B-Instruct-2507 75.7 against 61.5.
  - On a server's CPU too (AWS g5.2xlarge, an A10G, users' path: the wheel installed, `generate()` a fresh process
    each): Qwen3-8B 26.1 tokens/s at one sequence against bf16's 23.0 and 201.1 against 182.3 at eight, compiled 37.1
    against 27.1 and 256.5 against 197.5; Qwen3-4B-Instruct-2507 compiled 58.3 against 45.8; granite-3.1-3b-a800m (a
    mixture of experts) 31.4 against 25.3.
- On an A100, batched generation steps are faster (compute capability 8.0 only): Qwen3-8B's layer at 17-64 tokens
  takes 0.87-0.98x cuBLAS's GPU time (was 0.96-1.32x), Qwen3-32B's 0.79-0.87x; a Qwen3-8B step at 32 sequences 19.79
  ms against bf16's 20.82 (was 21.27), at 64 21.80 against 21.67 (was 25.96). Every other GPU is unchanged, bit for
  bit ([benchmarks/gpu/a100-mid-2026-09-27](benchmarks/gpu/a100-mid-2026-09-27)).
- `glyd.save_pretrained` copies the source's tokenizer files once and their content only: the Hub's cache keeps them
  read-only, and `tokenizer.model`, which two patterns match, failed on its second copy (Mistral 7B).
  `gpu/check_models.py`: each model against bf16 and the fp32 model over eight prompts, exact mode and save/verify;
  nine dense models from Qwen3-0.6B to Llama 3.1 8B pass on an RTX 4080 SUPER.
- The PyPI page leads with the models on the GPU (`pip install "glyd[gpu]"`, `from_pretrained`), with the project's
  links, keywords and classifiers.
- Blackwell, measured on an RTX PRO 6000 Blackwell Server Edition (compute capability 12.0, AWS g7e) with the wheel's
  library as it ships (compiled by the driver): `gpu/check_api.py` passes (dense, compiled, mixture of experts);
  Qwen3-8B's logits against the fp32 model 0.0509 (bf16's 0.0517), `exact=True` bit for bit; `generate()` 53.0
  tokens/s at one sequence against bf16's 50.0, compiled 93.0 against 75.9; a step's GPU time 11.22 ms against 13.42
  at one sequence, 16.34 against 16.35 at 64. Prompts are slower there (512 tokens 41.2 ms against 33.3, 2048 tokens
  130.5 against 103.2).
- Homebrew installs the release's binaries on Apple silicon and Linux (x86_64, arm64) in seconds; an Intel Mac and
  `--HEAD` build from source. `scripts/bump_formula.py` points the formula at a release.

## v0.20.0 — 2026-09-27

- The Python package on PyPI: `pip install glyd` (wheels for Linux x86_64 and aarch64 and macOS arm64), published from
  the release workflow by trusted publishing.
- Nine more open models measured, nineteen in all: GLM-4.5-Air, Llama 4 Scout, Qwen3-Next 80B-A3B, Muse Glimmer 30B,
  Qwen3.8 27B, Gemma 4 26B-A4B, Gemma 3 12B, Qwen3 4B 2507 and Llama 3.2 3B, every projection's matrix 32.2-33.0%
  smaller in `mma`, bit for bit. `sizes.py` counts a layer's experts kept as one tensor (Gemma 4, Llama 4) a matrix an
  expert, and every projection (the linear-attention inputs of Qwen3-Next and Qwen3.8); `e2e.py` runs Gemma 3 and 4,
  Qwen3.5-style linear-attention layers and checkpoints that load only with their vision tower. Qwen3.8 27B, the
  highest-scoring open model that fits one GPU, uses 41,071 MiB with Glyd against bf16's 51,771 (under a 48 GB card's
  49,140), perplexity 15.1941 against 15.1946, MMLU answers as bf16's on 99.67% of 300
  ([gpu/](gpu/README.md#popular-models)).
- On Hopper, batched steps from the 12-bit layout (`mma12`) are faster ([gpu/](gpu/README.md)): the same result every
  run. On an H100 SXM, Qwen3-32B's MLP matrices at 32 and 64 tokens take 83 and 92 us against cuBLAS's 90-96 (the
  kernel before 105-140); at 128 tokens 121-124 against 98-99. Small matrices still cost more than cuBLAS's, so end to
  end 32 and 64 sequences take 37.71 and 43.04 ms of GPU time a step against bf16's 32.44 and 35.50 (Qwen3-32B, 25%
  less memory); one sequence 26.14 against 28.23. `e2e.py --profile` measures every `--batch` size.
- Steps on Hopper faster again: with q, k, v and gate, up as one product each (`e2e.py --merge`, bf16 alike),
  Qwen3-32B on an H100 takes 24.50 / 26.75 / 33.46 / 37.35 ms of GPU time a token at 1 / 8 / 32 / 64 sequences against
  bf16's 27.90 / 29.78 / 31.44 / 33.52.
- The layouts measured on an RTX 4080 SUPER, an A10, an A100 and an H100
  ([gpu/](gpu/README.md#which-layout-on-which-gpu)); `best_layout()` and `e2e.py --format auto` take the faster for
  the GPU (`mma` on Ada and wherever only it fits). Batched steps on GDDR Ampere and Ada take a new path.
- Side by side with DFloat11 and ZipServ ([README](README.md#related-work)).

## v0.19.0 — 2026-09-26

- A second GPU layout, `mma12` (the 12-bit layout): 12.04 bits a weight, 25% under bf16, bit for bit, and the faster
  one on an H100. On an H100 SXM, **Qwen3-32B in 49.23 GB at 26.39 ms of GPU time a token against bf16's 65.52 GB and
  28.22 ms** (`mma`: 44.45 GB, 40.58 ms); Qwen2.5-7B 7.35 ms against 7.47; the products at 1-16 tokens 1.1-1.2x faster
  than cuBLAS's (Qwen3-32B's MLP at one token: 75-79 us against 86-90); MMLU 78.1% (bf16 78.3%). On an RTX 4080 SUPER,
  where memory is the limit, `mma` stays the faster at 1-32 sequences and `mma12` leads at 64 (2,455.9 tokens/s
  against 2,244.7; bf16 2,160.0). `pack_mma12`; `mma_gemm`, `mma_gemm_big` and `mma_unpack` take either layout;
  `e2e.py --format mma12` ([gpu/](gpu/README.md)).

## v0.18.0 — 2026-09-26

- The KV cache compressed in GPU memory ([gpu/kv.py](gpu/kv.py)), bit for bit: `GlydKVCache(config)` for a Hugging
  Face model; with `fused=True` a step of one new token a sequence runs `attn_decode`, attention straight from the
  compressed cache. Qwen2.5-7B-Instruct on an RTX 4080 SUPER: **the cache 31% smaller** (16K tokens: 651 MB for 947),
  peak memory below the plain cache's, a step as fast (21.7 ms against 21.6 at 16K; 19.2 against 18.3 at 1K); decoded,
  the cache is the plain one's bit for bit (the same tokens); through `attn_decode`, 256 tokens fed one at a time give
  perplexity 2.4778 against 2.4809 (16K).
- Larger models on rented GPUs, bf16 and Glyd in the same runs, MMLU on 1,000 questions
  ([gpu/README.md](gpu/README.md#larger-models)): **Qwen3-32B on one 48 GB RTX A6000** (44.45 GB; bf16 65.52 GB across
  two) at 1.24-1.28x bf16's tokens/s for 1-8 sequences, MMLU 78.0% (bf16 78.5%); **Qwen2.5-72B on three** (97.80 GB;
  bf16 145.41 GB across four) at 1.40-1.42x, MMLU 81.8% (81.9%); on an H100 SXM Qwen3-32B in 44.45 GB with MMLU 78.2%
  as bf16's, a token taking more GPU time than bf16's (40.6 ms against 28.2).
- The README leads with the AI work: `nvidia-smi` from the runs (`e2e.py --smi`, drawn by `scripts/term_svg.py`), the
  measured limits (about 34% off bf16 weights or KV cache for any lossless code, 18% off FP8, 7% off NVFP4).
- `e2e.py --mmlu N`, `--kv LENGTHS`, `--smi PREFIX`; `mma_cat` appends packs; `gpu_lambda.sh` takes models as
  `[org/]name[:B[:G]]` (side by side on B GPUs, then Glyd alone on G) and follows the run as it goes.

## v0.17.0 — 2026-09-26

- Model weights on the GPU ([gpu/](gpu/README.md)): the `mma` layout is smaller: **10.80 bits a weight over
  Qwen2.5-7B's matrices, 32.5% under bf16** (was 11.25), every tensor bit for bit. Qwen2.5-7B-Instruct on an RTX 4080
  SUPER: 10.61 GB where bf16 takes 15.25 (was 11.05); **1.25-1.32x bf16's tokens/s at 1 to 32 sequences** (one: 55.7,
  bf16 43.4; 32: 1,518.9, bf16 1,153.7), 1.13x at 48, 1.04x at 64; prompts of 16 to 128 tokens 19-29 ms (bf16 24-29),
  256 to 4096 within 5-10%. Perplexity as before (Wikipedia, 64-token windows 17.0052 vs bf16's 17.0015, 512-token
  7.5660 vs 7.5677).
- The GPU extension builds for the GPU it runs on (`sm_90a` on Hopper); `e2e.py --profile N` splits GPU time by
  kernel. `scripts/gpu_lambda.sh` runs `gemm.py` and `e2e.py` on one Lambda Cloud GPU instance launched for the run (a
  time cap, terminated and checked on exit).
- gpu/ is under the Business Source License 1.1 from this release (the store's terms); the codec stays BSD-3-Clause OR
  GPL-2.0.

## v0.16.0 — 2026-09-25

- Model weights on the GPU, several tokens at once ([gpu/](gpu/README.md)): a new layout, `mma` (`pack_mma`), for
  batches of tokens and prompts. Qwen2.5-7B-Instruct on an RTX 4080 SUPER, in 11.05 GB where bf16 takes 15.25:
  **1.23-1.33x bf16's tokens/s at 1 to 48 sequences at once** (one: 55.2 tokens/s, bf16 43.3; 32: 1,528.6, bf16
  1,149.0; the batched product it replaces ran 0.72-0.90x); prompts of up to 128 tokens faster than bf16 (128: 27 ms,
  bf16 29; was 60), 256 to 4096 within 5-9% (4096: 696 ms, bf16 645). The same result every run; perplexity as bf16's
  (Wikipedia, 64-token windows 17.0052 vs 17.0015, 512-token 7.5660 vs 7.5677).
- `gpu/e2e.py`: `--format mma`, `--batch`, `--gpus N` (layers spread over N GPUs; bf16 by accelerate's device map),
  `--ppl`; every kernel runs on its tensors' device and PyTorch's current stream. `gpu/setup_env.sh` builds the
  environment without root (nvcc pinned to PyTorch's CUDA); `scripts/gpu_aws.sh` runs Qwen2.5-32B and 72B on 4x L40S.

## v0.15.0 — 2026-09-25

- Model weights on the GPU ([gpu/](gpu/README.md), Python and CUDA beside the library): a bf16 model's weights held
  compressed in GPU memory and rebuilt there bit for bit, in two formats: dense (10.9 bits a weight) and fast (11.25).
  Qwen2.5-7B-Instruct on an RTX 4080 SUPER (16 GB): **55.1 tokens/s in 11.05 GB** (fast) and **52.0 in 10.60 GB**
  (dense), against bf16's 43.2 in 15.25 GB; the fast format's 128 tokens as bf16's. Prompts of up to 64 tokens take
  35-40 ms (bf16 24-27); 2048 tokens 337 ms (bf16 301).
- PyTorch checkpoints (`torch.save`): the zip's tensor storages go in as byte planes, their element widths read from
  the checkpoint's pickle (a reader for the opcodes `torch.save` writes, no dependency); against a base checkpoint
  (`--base`, the store) each storage the base holds under the same name and size goes in as XOR that storage, where
  that is the cheaper (weights move little between checkpoints, Adam's first moment as much as it holds). Qwen2.5-0.5B
  fine-tuned with AdamW (fp32 weights and both moments, 5.93 GB a checkpoint): 83.2% of its size alone, 77.3% against
  the checkpoint 50 steps before; zstd -19 92.2% (its `--patch-from` stops at 2 GB). The store finds a checkpoint's
  predecessor by its storages' names and sizes.

## v0.14.9 — 2026-09-25

- The store: a family's first object sits at the depth cap only until
  its family first needs it; then it is lifted, stored again against a
  shallower base, in place of every new family starting shallow
  (v0.14.8's rule). The terabyte gate, same corpus and instance
  ([report](docs/codec/benchmarks/store-gate-2026-09-24.md)): **43.61 GB,
  3.52× fewer bytes than zstd -3** (44.35 GB, 3.46× in v0.14.8), 27.2×
  against raw; the English Wikipedia tables 8.67 → 7.94 GB, the kernels
  as in v0.14.8 or smaller. Put 374 MB/s, read-back 486 MB/s (zstd -3's
  346), restore 559 MB/s; all 1,192 objects byte-exact, both reads.
- The store finds a model checkpoint's predecessor: a safetensors
  object's fingerprints are its tensors' names, widths and sizes (it
  shares no bytes with the checkpoint before it). Nine Pythia-410M
  checkpoints (14.6 GB): 5.88 GB in the store, where zstd -19 stores
  7.28 GB and zstd -3 8.63 GB; the last two as deltas of 501 MB, every
  object verified ([report](docs/codec/benchmarks/weights-2026-09-25.md)).
- Reads: the decoder no longer spins while a unit waits for the ones
  ahead of it; the unit that completes the run writes it out. Decompress
  CPU at the plain levels fell by up to 65% on 32 threads (logs `--max`
  3.08 → 1.09 s), wall time unchanged.
- The same bytes on every machine: an input is cut into sixteen units
  whatever the core count (it followed the thread count before, so a
  32-thread machine cut twice as finely as a 16-core one). GitHub
  events at `--ultra` 5.4% smaller on 32 threads; unchanged on 16.
- Noise-level bytes of a small alphabet (a model weight's exponent or
  mantissa plane) are coded as literals alone where the parse's short
  matches would cost more: `--max` on Pythia's exponent plane 171.7 →
  140.1 MB (zstd -19: 143.8), on Qwen2.5's 208.3 → 169.8 MB (171.6).
  Text, logs, SQL and kernel tars come out byte-identical in size. The
  store's dense level (stripes on all cores) too: a stored checkpoint
  752 → 697 MB.
- Model weights: a safetensors file is opened, each tensor of 2-, 4- or
  8-byte elements as byte planes (exponents together, mantissas
  together), the header kept; closed byte for byte. Pythia-410M (fp32,
  1,621 MB) at `-9`: 701 MB in 3.5 s, against zstd -19's 809 MB in 92 s
  and zstd -3's 959 MB; Qwen2.5-0.5B (bf16, 988 MB): 663 MB against
  750 and 769. Reads at 1.1 GB/s.
- Checkpoints against checkpoints: in base mode, a safetensors file
  against one holds each tensor the base has under the same name and
  size as that tensor XOR the base's, in planes. Pythia-410M step 71000
  against step 70000: 612 MB in 4.7 s, where zstd -19 `--patch-from`
  stores 805 MB (the raw delta finds nothing to match); step 143000
  against 142000: 501 MB. Qwen2.5-0.5B-Instruct against its base model:
  558 MB.
- The command line asks for `--base` for a container opened against a
  base (a gzip against a gzip, weights against weights): such a file
  started with the container envelope, so decoding it without that
  check failed.
- The savings calculator re-measured at this code (LZ4, gzip, zstd -3
  and -19, four Glyd levels, decompress CPU per row; the bucket row
  from the v0.14.8 gate): [report](docs/codec/benchmarks/savings-2026-09-24.md).

## v0.14.8 — 2026-09-24

- **At a terabyte: 3.46× fewer bytes than zstd -3** (3.32× in
  v0.14.7). The gate's 1,192 objects, 1.18 TB, put through the store
  into S3 from one 16-vCPU instance next to the bucket
  ([report](docs/codec/benchmarks/store-gate-2026-09-24.md)): 44.35 GB
  stored against zstd -3's 153.5 GB, 26.7× against raw; put at 386
  MB/s (372 in v0.14.7, zstd -3's own put 535); every object read back
  by its own process at 464 MB/s (zstd -3's read-back 348), all 1,192
  byte-exact; the whole bucket restored by one process at
  571 MB/s (603 in v0.14.7), all 1,192 byte-exact. Kernel releases 5.15 at 286×,
  6.1 at 406× (155× in v0.14.7), 6.6 at 377× against raw; hourly
  GitHub events 14.1×. One cost in the release's own rule (below):
  the English Wikipedia tables 19.1× against raw, where v0.14.7 kept
  them at 20.9×.
- **Record mode writes 1.6–1.9× faster, the same bytes.** On the
  Ryzen box, output byte-identical to v0.14.7: the NASA access log on
  one core 151 -> 265 MB/s, on all cores 749 -> 1,186 MB/s; a
  Wikipedia table dump 126 -> 198 and 407 -> 790 MB/s. A dictionary
  column's values are hashed once, and its recency list is kept in
  place, searched eight entries at a time and not at all for a value
  it cannot hold; a time value on the last exact value's date is not
  printed back to be checked; field ranges are 32-bit (a log unit's
  ranges had outweighed its text); a delimited line is split in one
  pass; a SQL dump's rows no longer allocate, and its text is crossed
  a word at a time to the next special byte.
- **Parquet files with snappy or zstd pages are opened**
  (`src/parquet.rs`, `src/resnappy.rs`, `src/rezstd/`): the footer's
  column chunks and page headers are read (thrift's compact protocol,
  no dependency), and every page is written back byte for byte by a
  port of the compressor that wrote it, so the page's raw bytes are
  compressed instead of its LZ tokens: google/snappy 1.2 level 1
  (builds differ in their hash, a multiply or the CRC32C instruction,
  and their table, 2^14 entries up to 1.1.10, 2^15 since 1.2.0), and
  zstd 1.5.2 through 1.5.7 at levels 1 and 3 (the fast and
  double-fast finders and their variants past the window's wrap,
  Huffman literals with the previous block's table, FSE sequence
  tables, the capacity rules, the checksum; 1.5.7's pre-block-splitter
  and its two double-fast rules, 1.5.2's three fast-finder rules;
  1.5.4 and 1.5.6 write what 1.5.5 does) as its library's one-shot
  call writes them and as its command line does: the
  `--single-thread` stream of 128 KB chunks, and the default of 2 MB
  jobs from fresh contexts seeded with the 64 KB before each, so a
  file `zstd` wrote opens too. Checked against zstd's own output on
  the fixtures, on a sweep of 3,500 inputs under every version, level
  and writer, and on 200 MB logs and dumps. The opener finds the
  build that wrote a page and keeps a page no build made (zstd 1.4
  and older, other levels, gzip pages). A container under a frame
  opens in turn: the snappy taxi file inside a `zstd -3` frame,
  52.3 MB, comes to 34.8 MB. A whole zstd frame (a `.zst` object) opens the
  same way, a container under it opened in turn: the NASA access log
  as zstd 1.5.5 wrote it at level 1, 22.3 MB, comes to 8.0 MB at
  `--max` (its records modeled), 1.4 s to write and 0.6 s to read
  back, byte-exact.
  A page's plain values are then modeled so the LZ and entropy stages
  see their structure: fixed-width values as byte planes, integers in
  their unit (microseconds that are whole seconds divided down) and
  as deltas, doubles that are decimals as scaled integers, byte
  arrays as lengths then bytes; dictionary-index pages have their
  runs decoded, written again by a port of Arrow's run-length encoder
  and compared, and the indices laid out as planes of the bytes that
  hold them. Each page takes the cheapest of its candidates or stays
  as it is, judged by the max level's own output, and the level
  blocks stay ahead. A NYC taxi month written by pyarrow 21 with
  snappy, 61.7 MB: 174 of 174 pages reproduced; `--max` 34.8 MB in
  0.5 s on ten cores (1.8 s on one), read back in 94 ms (zstd -3 on
  the file 52.3 MB, zstd -19 49.8; the same table's zstd-page file
  50.3; record mode on the table as CSV 37.0), `--ultra` 32.3 MB,
  every decode byte-exact; the same table's zstd-page file, 50.3 MB
  (pyarrow 14, zstd level 1): `--max` 34.8 MB in 0.5 s, read back in
  0.14 s. A month of for-hire trips, 519 MB with snappy, 1,291 pages,
  more of them ids: `--max` 376.5 MB in 3.4 s (zstd -3 on the file
  473.2), read back in 1.0 s; with zstd pages, 472.8 MB: 376.5 MB in
  2.7 s. Pages compressed with gzip, lz4 and brotli are left as they
  are.
  Files polars and DuckDB write open fully too: polars' snappy pages
  are the Rust `snap` crate's (the multiply's older hash, shifted by
  the table's size, which differs on blocks under 8 KB), and each
  writer's run-length encoder for dictionary indices is ported beside
  Arrow's (polars: repeats of more than eight, literal runs of up to
  8192 values packed in blocks of 32; DuckDB: repeats of four or more,
  bit-packed blocks of 256 written whole), the one that writes a
  page's runs again named in its recipe. A bit-packed run's padding
  (a block's earlier values) is taken as the writers leave it. The
  same taxi month at `--max`: polars' snappy file, 86.9 MB, 51.4 ->
  48.1 MB, its zstd file, 57.8 MB, 50.8 -> 48.1 MB; DuckDB's snappy
  file, 61.1 MB, 36.6 -> 35.0 MB, its zstd file, 45.7 MB, 36.6 ->
  35.0 MB; every decode byte-exact.
- **A new family starts shallow.** An object that starts a family (a
  new major release) takes an ancestor at depth 1 or the chain's root
  as its base, so its versions come back to it at the depth cap; one
  that had landed at the cap itself sent them to the chain's root.
  At the gate, 6.1.1 sat at depth 4 on a 5.15 release and every fifth
  6.1 release was a 26 MB delta of 5.15.1. On the Ryzen box, 5.15.1-100
  then the 150 releases of 6.1 in the gate's order: 6.1 1,316 -> 502
  MB, all 250 releases 1,714 -> 899 MB. At the gate the rule also
  moved the monthly Wikipedia page tables, each month a family of its
  own, onto bases two months back and the September one to alone:
  0.73 GB more there, most of the kernels' 0.81 GB gain. The next
  release lifts only a family's first object that sits at the cap,
  when its family first needs it (measured on the box: the kernels
  as here, the Wikipedia tables as in v0.14.7).
- **The store keeps a version's base among its own kind.** An object
  whose base holds under 98% of its fingerprints starts a family (a
  new kernel major holds 0.85–0.94 of the old one's releases; point
  releases hold 0.99–1.00 of the last, a 16-day Ubuntu image 0.97,
  monthly Wikipedia tables 0.69–0.99), and past the depth cap a version's base is its
  family's first object, not the chain's root. At the terabyte gate
  every kernel release sat in one chain rooted at 5.15.1, and every
  fifth 6.1 and 6.6 release was a delta of 5.15.1 at 42 MB against
  2–4 MB within its series: 1.55 of the 6.6 series' 1.84 GB. Measured
  on the Ryzen 9 box, 5.15.1 then the 150 releases of 6.6 in the
  gate's order: 2,024 MB stored before, **643 MB now**, one 42 MB
  delta (6.6.1 itself against 5.15.1). The index line carries the
  family (an eighth field; older lines read as before, their family
  the chain's root). A star-shaped chain tree was tried and dropped:
  6% fewer bytes on kernels, 30% more on monthly tables, where a base
  two months back costs half again the neighbour's.
- **Whether a delta pays is judged on four 8 MB windows spread over
  the object**, each against its own base region, at the same 80% bar
  the whole must meet (the head's 32 MB at a 50% bar before). Across
  the English Wikipedia `page` dump the ratio of delta to alone runs
  56–110% by window, 68% whole; the head's verdict had stored the
  2026-09 dump alone, 1,776 MB where its delta against 2026-08 is
  1,205 MB (v0.12.0 had that delta; v0.13.0's sample lost it). An
  object holding a fingerprint several times now counts once among
  its holders. The four windows run on threads of their own: an hour
  of GitHub events put in 0.50 s on the box against 0.79 s with them
  one after another.
- The gate's read-back compares each object with the corpus file of
  its name, the index line's last field (it read the seventh, which
  the family field made the family's id, and counted every object
  failed); each gate run syncs into a directory of its own (two runs at
  once shared one, and the first to finish stopped the other's wait).

## v0.14.7 — 2026-09-24

- **The store's put is 1.6–4.0× faster, its get 1.1–1.5×.** On a
  Ryzen 9 7950X3D (16 cores), best of three, every object read back
  byte-exact: Linux 6.10.1 as a version of 6.10 (1.5 GB) put at
  1,224 MB/s against 308 before, alone at 2,545 against 668, read
  back at 1,590 against 1,051; an Ubuntu 24.04 cloud root filesystem
  (1.1 GB, gzip inside) as a version 215 against 127, read 287
  against 258; a Wikipedia table a month on (108 MB) as a version 383
  against 233. What changed: `put_file` maps the file and keeps the
  mapping as the cached copy (`put_vec` takes the bytes over; no
  second copy of the object in memory); the container path's deflate
  emulation returns at once when nothing in the object opened; an
  opened base is decoded once and cached; the base region a unit
  searches is cut to what its fingerprints reach (97% of the hits
  kept, never under the unit and 16 MB); each thread keeps one
  region-and-unit buffer; a delta under a thirty-second of the object
  is taken without also compressing the object alone; `get_to` writes
  into the caller's buffer. The Ubuntu root's version is bounded by
  the deflate emulator, which runs at zlib's own search speed per
  thread. At a terabyte, one im4gn.4xlarge next to S3
  ([report](docs/codec/benchmarks/store-gate-2026-09-24.md)): put at 372
  MB/s (243 in the 2026-09-22 run), every object read back by its own
  process at 461 MB/s (166 before; zstd -3's own read-back on that
  instance 341), the bucket restored by one process at 603 MB/s, all
  1,192 objects byte-exact both times; 46.3 GB stored against zstd
  -3's 153.5 GB, 3.32× fewer bytes (3.10× then: the hourly events
  store 9% smaller; one English Wikipedia table of fifteen went alone
  that was a delta before, 0.6 GB, to be looked at).
- **CLI reads: the decoded batch lives on huge pages.** The output
  batch is an anonymous 2 MB-aligned mapping with `MADV_HUGEPAGE`,
  the input is populated on a thread while the first units decode,
  and `-d -s` streams on one thread. One core against `zstd -d -T1`:
  Ryzen 9 7950X3D 1.00–1.30× its speed (0.65–0.81× before), Graviton3
  1.15–1.32× (1.03×), Sapphire Rapids 0.79–0.97× (0.80×). All cores:
  3.9–13.3 GB/s on the Ryzen, 3.6–10.0 GB/s on eight Graviton3 cores,
  1.9–5.1 GB/s on eight Sapphire Rapids cores.
- **CRC-32C in three lanes**: the block checksum runs three CRC
  streams over 1 KB lanes and joins them by table, so the CRC
  instruction's latency overlaps; it was a tenth of a one-core read on
  Sapphire Rapids. Bytes unchanged: the checksum's value is the same.
- **A corrupted unit fails a stream's parallel decode instead of
  hanging it**: the units after a failed one waited for their turn
  for ever; the first error now stops the rest
  (`tests/fuzz_safety.rs`).
- Library: `Store::put_vec`, `Store::put_file`, `Store::get_to`. CI
  keeps the corpus between runs and fetches enwik8 from a second host
  when the first answers with a page.

## v0.14.6 — 2026-09-24

- **Blocks are cut where the bytes' statistics change** (`src/split.rs`,
  the idea of zstd 1.5.7's pre-splitter): before each block's parse,
  sixteen bytes of every 256 in the 256 KB ahead are counted per 16 KB
  segment, and the block ends at the segment boundary where the two
  parts coded on their own statistics beat the whole by more than a few
  blocks' overhead, each part charged for describing its table; never
  under 32 KB, and the search rests after 32 windows without a cut. On
  binaries with sections of different content it pays: Silesia mozilla
  0.9% smaller (now 1.0% under zstd 1.5.5, 0.2% over zstd 1.5.7) for
  5% more write time on that file; JSON events, the NASA log, a table
  dump and enwik8 are unchanged in bytes and time. Every stream decodes
  as before (blocks were always any length up to 256 KB).

## v0.14.5 — 2026-09-24

- **`--max`'s parse is zstd -3's double-fast, no lazy step, with one
  of zstd's rules made stricter.** The lazy compare one byte on cost
  4–8% of the write time for 0.1% (a table dump) to 4% (a log) fewer
  bytes; it is gone. In its place: when only the short table matched,
  the long table's entry one byte on (loaded already) is tried and its
  match taken when it is at least two bytes longer; zstd's
  unconditional form loses 0.7% on the table dump, this one gains on
  every file. The literal Huffman lengths meet their 11-bit limit by
  package-merge (optimal) instead of halving the counts. Bytes against
  zstd -3: GitHub events −7.8%, a Wikipedia table dump −1.2%, the NASA
  log −0.6%, enwik8 −0.5%, Silesia mozilla −0.1% (zstd 1.5.5; against
  1.5.7, whose new block splitter gains 1.2% on mozilla, that file is
  +1.1%). One core on Graviton3 against `zstd -3` as installed: events
  1.05× its speed, mozilla 1.05×, enwik8 1.01×, the log and the dump
  0.89×; against `zstd -3 --single-thread` 1.03–1.26× on all five.
  Eight cores against `zstd -3 -T8`: 1.09×, 1.21×, 0.97×, 1.08×,
  0.83×. On a Ryzen 9 7950X3D (one core, zstd 1.5.7) 1.07–1.26×
  faster on all five. The CLI's one-core path writes on a second
  thread, as zstd's does; the parse loop keeps fewer values live.
  Every stream decodes as before.

## v0.14.4 — 2026-09-23

- **`--max` is now zstd -3's structure: the long-distance matcher is
  opt-in (`--long`, `-L`).** The pass that finds repeats up to 128 MB
  back was a third of the write time on logs and events; zstd -3 has
  no such pass, so `--max` now runs without it and `--max --long` is
  what `--max` was, as `zstd --long`. Library:
  `compress_into_max_long`, `compress_parallel_into_max_long`,
  `compress_records_into_max_long`; C `GLYD_LEVEL_MAX_LONG`; Python
  level `"max-long"`, Go `LevelMaxLong`. `--dense`, `--ultra`, base
  mode and the store keep the long search: their job is the ratio.
  One core on Graviton3 / Sapphire Rapids, `glyd -9` against zstd -3:
  GitHub events 0.93/0.95× the speed at 9.7% fewer bytes, the NASA
  log 0.81/0.80× at 4.1% fewer, a Wikipedia table dump 0.82/0.74× at
  1.0% fewer, Silesia mozilla 0.94/0.87× at 0.5% fewer, enwik8
  0.89/0.80× at 0.3% fewer (v0.14.3 wrote at 0.50–0.83×). Eight cores
  against `zstd -3 -T8`: events 1.08/1.14×, the log 1.07/1.07×,
  mozilla 1.05/0.99×, the dump 0.84/0.85×, enwik8 0.86× (Sapphire
  Rapids). `--max --long` against `zstd -3 --long=27`: 0.83–1.18× the
  speed at 0.3–14% fewer bytes. Every stream decodes as before.

## v0.14.3 — 2026-09-23

- **A repeat offset after zero literals is implied by the literal
  length** (flag `LL0_REP`): a match with no literal before it cannot be
  the last offset going on, so the repeat codes shift and the common
  case — records alternating between two sources — is code 0 every
  time, which is zstd's rule too. Measured with zstd's own sequences in
  both coders: ours was 3.4% behind zstd's on the table dump, now 1.1%
  (table headers). On the servers, one core, the Wikipedia table dump
  32.48 → 31.44 MB per 200 MB (zstd -3 31.18), the NASA log −0.3%,
  GitHub events −0.2%, Silesia mozilla −0.25%, at the same speed.
  Earlier files decode as before.

## v0.14.2 — 2026-09-23

- **The max level's parse takes the last offset one byte on, before
  anything the hash tables say** (zstd's double-fast order): a record
  that differs from the one before it in a byte keeps its offset, which
  codes in a couple of bits. On Graviton3 and Sapphire Rapids, one
  core: a Wikipedia table dump 6.3% smaller (34.67 → 32.48 MB for 200
  MB; zstd -3 31.18), GitHub events 0.8%, the NASA log 1.2%, Silesia
  mozilla 0.2%, at the same speed. The probe steps grow after 256
  misses instead of 64 (as zstd's double-fast).
- **Blocks are checked with CRC-32C** (flag `CRC32C`, the hardware
  instruction on aarch64 and x86-64). The Adler-like sum every earlier
  release wrote kept 16 bits of its weighted half and missed, for one,
  two bytes swapped 8 KB apart — found by the mutation fuzz once the
  parse above changed the bytes it mutates. Earlier files verify as
  before; files written from now on need v0.14.2 or later to verify.

## v0.14.1 — 2026-09-23

- **JPEG on every core.** The range coder's decision is branch-free
  (the mispredicted branch on the bit was the cost); the scan is
  written in bands on every core and, when it has restart intervals,
  parsed in bands at its markers; files of 10 MB and up take eight
  stripes. All cores against v0.13.4 (Lepton, single-threaded), every
  decode byte-exact: the 13.4 MB photo written in 0.62 s and read in
  0.29 (1.77 and 0.91); the 6.4 MB one 0.48 and 0.23 (0.97 and 0.50);
  the three of ~2 MB 0.16–0.21 and 0.07–0.10 (0.37–0.43 and
  0.19–0.22); every one smaller. One core: 12–16% slower than v0.13.4.
  v0.14.0 files read back unchanged (fixtures in `tests/data/legacy`).

## v0.14.0 — 2026-09-23

- **JPEG recoded by Glyd's own model.** `src/jpg/` (design:
  `docs/design/jpeg-recoding.md`) parses a baseline JPEG into its
  markers and coefficients and writes it back bit for bit; the
  coefficients are coded with a range coder under contexts from the
  blocks above and to the left, the first row and column predicted
  from pixel continuity across the block edge, the DC from both
  edges, in four stripes of block rows after a prefix so four cores
  share the work. Stream `GJPG` inside the `GLYDJPEG` envelope and
  inside containers (spec 2c); the kept bytes (EXIF, previews)
  compressed. `lepton_jpeg` stays only to read what v0.13.0–v0.13.4
  wrote (the `jpeg` feature). Against v0.13.4 on this Mac, every
  decode byte-exact: smaller on all five photos (13.4 MB: 9,934,880
  against 9,971,627 bytes; 6.4 MB: 4,997,778 against 5,001,278; the
  three of ~2 MB by 0.4–0.8%), 1.6–1.9× faster to write and 1.6–1.7×
  faster to read on all cores (13.4 MB: 1.02 and 0.54 s against 1.77
  and 0.91); on one core 1.3× slower each way. A progressive JPEG
  stays as it is, as before.
- The binary arithmetic coder is generic over its probability type
  (`reflate::coder::Prob`); the corrections coder is unchanged.

## v0.13.4 — 2026-09-23

- **Containers opened by Glyd's own deflate reconstruction.**
  `src/reflate/` (design: `docs/design/deflate-reconstruction.md`)
  parses a deflate stream into its blocks and tokens, runs zlib's own
  matcher over the plain text — `deflate_fast` and `deflate_slow`,
  the level's chain, lazy and nice limits, the rolling hash, the
  window, memLevel and windowBits detected from the stream — and
  builds each block's Huffman trees the way zlib does, so that for a
  stream zlib made almost nothing needs saying: what differs is coded
  with a binary arithmetic coder. Streams are cut into 1 MB chunks at
  block boundaries, each emulated from the window before it, so they
  open and close on every core. Nothing outside this repository is in
  the codec's path any more; the copy of preflate-rs stays to read
  what v0.12.0 to v0.13.3 wrote and to open the bases their deltas
  were made against. Envelope `GLYDDEF3`; segments 9–12 (spec 2b).
  Against v0.13.3 on this Mac, all cores, every decode byte-exact:
  the NASA gzip 8.19 → 8.17 MB, written in 1.2 s instead of 2.4;
  a PDF (pdfTeX, 4 KB windows) 741 → 728 KB, 0.15 s instead of 0.55,
  read in 0.06 instead of 0.15; a PNG 1.65 → 1.60 MB, 0.16 s instead
  of 0.49; a Guava jar 1.77 → 1.68 MB; a .docx 2.32 → 2.27 MB, 0.25 s
  instead of 0.70; the mixed tar.gz 10.18 → 10.11 MB, 1.8 s instead of
  3.7. On a 1 MB text through zlib at every level and strategy the
  recipe is 63–250 bytes (0.01–0.1% of the stream) where preflate's
  corrections were 28 bytes to 17 KB. Behind on one file: a PDF of
  large images whose streams are level 9 (a 4,096-deep chain per
  token) writes in 3.3 s instead of 2.1 and reads in 1.3 instead of
  0.6 — the matcher's speed on such data is the next piece of work.

## v0.13.3 — 2026-09-22

- **Containers open and close on every core.** A deflate stream of
  16 MB of content or more is cut into 8 MB chunks at block boundaries;
  each is predicted, checked and later re-created by a predictor of its
  own that first learns the 32 KB before it, so every chunk runs on its
  own core and the pieces join bit for bit (segments `DEFLATE_CHUNKED`
  and `DEFLATE_NESTED_CHUNKED`; `preflate-rs` is carried in
  `third_party/` with the chunking added, see its README). Ten cores,
  byte-exact: the 20.7 MB NASA gzip written in 2.3 s instead of 4.6,
  read in 0.26 s instead of 1.56; a 6.8 MB PDF with figures 1.9 s
  instead of 13.7, read in 0.59 s instead of 4.3; a 23 MB tar.gz 3.5 s
  instead of 6.8, read in 0.50 s instead of 1.9. One thread: the same
  as before. Corrections grow by a few hundred bytes per chunk.
- **Content reads.** `glyd -d --content`, `glyd-store --get ID
  --content`, `decompress_content` and `decompress_content_with_base`:
  the content of a gzip or zlib object stored opened — what `gunzip`
  prints, members one after the other, a tar.gz's tar — without
  re-creating the deflate stream, which is 96% of a read. The 20.7 MB
  NASA gzip on one thread: 0.36 s for the content against 1.58 s for
  the gzip back (0.11 s on ten cores; `gunzip` 0.10 s: the stored form
  is record mode, whose decode is the remaining cost). A zip, a PDF, a
  tar of gzips and an object stored closed have no content view.

## v0.13.2 — 2026-09-22

- **The default, fast and turbo levels leave containers closed.** They
  opened gzip, zip, tar, PDF, PNG and JPEG objects like every other
  level, at 0.6–5.5 MB/s on one thread, and on most then wrote the
  closed form anyway, an LZ4-class level on the content losing to the
  file's own deflate; on a jar, a JPEG and a .pptx they kept the opened
  form, whose reads run at 9–20 MB/s. Those levels exist for speed, so
  they no longer open anything: a 20.7 MB gzip goes through the default
  level at 1,529 MB/s instead of 5 (zstd -3: 1,204), a 6.8 MB PDF with
  figures in 0.00 s instead of 11.7. Containers open from `--max` up,
  as before. Files those levels wrote with an envelope still decode.

## v0.13.1 — 2026-09-22

Fixes. Every file v0.12.0 and v0.13.0 wrote reads back with this one.

- **Files that did not decode.** Since v0.12.0 a unit of a multi-unit
  stream could itself be opened as a container: a gzip member (from
  v0.13.0 also a zip, tar or PDF) that began exactly on an internal
  unit boundary got an envelope where a block was due, and the file
  failed to decode ("Implausible block header"): an error, never wrong
  bytes. Seen with v0.12.0's `--max` and v0.13.0's default level and
  `--max` on a file with a gzip member at 8 MB. A part of a stream (a
  unit, a record unit, a trial sample, an opened container's plain
  text) is never opened now, and the decoders read such files: the
  stream is read around the envelope, its inner blocks taken until they
  hold the plain text its recipe needs. Files written by the released
  binaries are in `tests/data/legacy/`, and a test reads them.
- **Containers against a base, and their speed.** v0.13.0 kept a
  container's own bytes (a tar's files and headers, a zip's directory)
  and the corrections in the recipe, out of reach of base mode: an
  Ubuntu image with 6,128 .gz files inside came out of
  `compress_with_base` at 250.7 MB in 17 s, where without opening it is
  25.6 MB. The envelope is now `GLYDDEF2`: the plain text holds the
  streams' content, then every other byte, the corrections and the
  transcoded pictures; the recipe is structure only. The same pair:
  24.8 MB in 3.6 s, decoded in 1.9 s. Tar, zip and PDF entries open on
  every core, and segments close on every core. `GLYDGZIP` (v0.12.0)
  and `GLYDDEFL` (v0.13.0) envelopes still decode, alone and against a
  base.
- When the opened object loses to the closed one, the closed
  compression already made is written instead of being made again.
- A deflate stream expands to at most 200 times its size (at least
  256 MB) before it is left closed.
- `preflate-rs` and `lepton_jpeg` are pinned to exact versions: a base
  must open to the same plain text for as long as its deltas are kept.

## v0.13.0 — 2026-09-22

### Deflate containers opened

- Zip (and so .docx, .xlsx, .pptx, .jar, .apk, .odt), zlib streams and
  PNG join gzip: `src/deflate.rs`, envelope `GLYDDEFL` (replacing
  v0.12.0's `GLYDGZIP`). Every deflate stream inside is decoded with
  what it takes to re-encode it bit for bit; headers, directories,
  stored entries and non-image chunks are kept; a PNG's image stream is
  cut back into its IDAT chunks. Entries preflate cannot reproduce stay
  as they are, and so does an object that would not shrink. The CLI's
  `--max` and `--ultra` take record mode on an opened container where
  it pays. PDF too: every stream whose data is zlib and that ends
  before an `endstream`, found by scanning. The recipe goes into the
  envelope compressed (a PDF's duplicate fonts and a jar's thousand
  entries repeat their corrections), and the object is also compressed
  closed at the same level, the smaller kept. Measured, decodes
  compared: a Guava jar 3.05 → 1.76 MB at `--max` (zstd -19 on the
  jar: 2.70), 1.14 MB cold; a GitHub source zip 2.73 → 2.28, 1.64
  cold; a 60-slide .pptx 88 → 24 KB; a 30,000-row .xlsx 1.34 → 0.47
  MB cold; a pdfTeX paper 2.22 → 0.73 MB (zstd -19: 1.04), a paper
  with figures 6.77 → 4.23 (5.54); a PNG photo 1.83 → 1.64, 1.19 cold.

### JPEG transcoded

- A JPEG is recoded losslessly by Lepton (`lepton_jpeg`, the Rust port
  of Dropbox's, behind the default feature `jpeg`): its DCT
  coefficients under an arithmetic coder with a predictor across
  blocks, envelope `GLYDJPEG`, the identical JPEG back. Six photos,
  35.9 MB: 27.3 MB, 24% fewer bytes (a JPEG XL transcode: 20%), every
  one restored byte for byte; 5–7 MB/s in, 12–14 MB/s out, one core.
  A JPEG Lepton cannot take, or that does not shrink, stays as it is.
  Inside a container too: a JPEG stored in a zip, or deflated (as an
  Office document holds its pictures), is transcoded under its entry;
  any container stored or deflated inside another is opened with a
  recipe of its own nested in the segment, four deep, and tar joins
  the containers. A 12-slide deck of photos, 6.51 MB: 5.29 MB (zstd
  -19: 6.50); a document of six PNG screenshots, 2.51 MB: 2.32 MB at
  `--max`, 1.65 MB cold (zstd -19: 2.50); a tar of six photos, 36.0
  MB: 27.3 MB, and 27.3 MB through a gzip of it; a tar.gz of a gzipped
  log, a PDF, a PNG and a .docx, 22.9 MB: 10.2 MB (zstd -19: 22.9).

### Speed, same bytes

- The store's put, where its time went (`GLYD_STORE_TIMING=1` prints
  it): the fingerprint scan now runs on every core; a 4 GB cache of
  decoded objects serves the next base and every delta on a chain (its
  root decoded once, where each object over 1 GB fetched and decoded
  its base again); a 32 MB sample decides a delta before the whole is
  tried, and its alone size is the estimate (an hour of events shares
  half its fingerprints with the hour before and gained nothing from a
  full delta). Kernels put at 800–1,600 MB/s on this Mac (300–500
  before), events at 1,300–1,900; every stored byte identical. The
  terabyte gate rerun ([report](docs/codec/benchmarks/store-gate-2026-09-22.md)):
  put 243 MB/s against 150, 1 h 21 min against 2 h 12 min, every
  object back byte-exact, the rebuild 51 min against 86.
- Dense max level (`--dense`, `compress_into_max_dense`,
  `compress_max_stream`): units stay at the far matcher's 128 MB
  instead of shrinking to give every core one; a unit's far matches
  are found by one core and its blocks parsed in 16 MB stripes by all
  of them, tables seeded with the 2 MB before each stripe, so the
  bytes are one core's at any core count: 5–9% fewer than the default
  on files of a few hundred MB, the same on files of gigabytes (the
  8.7 GB suite corpus: 3.946 against 3.939). Reads then scale only
  with the units (the suite's 8-thread decode 4.7 GB/s against 10.4),
  so it is opt-in: the CLI's `--dense`, and the store, whose objects
  are written once and read rarely. Raw results of the suite run with
  it: `benchmarks/suite/*-dense/`.

## v0.12.0 — 2026-09-22

### Gzip objects opened

- A gzip object's deflate streams are decoded to their plain text
  with what it takes to re-encode each bit for bit (`preflate-rs`,
  the crate's one dependency, behind the default feature `deflate`);
  the plain text then takes whatever was asked — a level, record
  mode, the cold level, a base — so a gzipped log costs what the log
  costs. Every encode entry point opens gzip input, every decode
  entry point closes it, and an opened object that would cost more
  than the gzip stays as it is. A gzip -6 NASA log: 20.7 MB → 8.2 MB
  at `--max -r`, 6.5 MB cold; a gzipped 512 MB kernel tree: 72.7 MB →
  56.1 MB at `--max`, 42.0 MB ultra; all back byte-exact. Through the
  store too.

### The store at a terabyte

- The gate run ([report](docs/codec/benchmarks/store-gate-2026-09-22.md)):
  1,192 objects, 1.18 TB, put into S3 from one instance, 49.0 GB
  stored against zstd -3's 153.5 GB (3.13× fewer bytes, 24× against
  raw), every object read back byte-exact, the metadata directory
  rebuilt from the bucket and verified.
- The first attempt died of memory on 20 GB objects: put now maps its
  files instead of reading them, and the last object is kept as the
  likeliest next base only up to 1 GB.
- `glyd-store --version`.

## v0.11.2 — 2026-09-21

- A lost metadata directory is rebuilt from the objects: every
  object's index lines now ride beside it in the backend as
  `<id>.index` (a pack's carry its members'), and `--rebuild` (or
  `Store::rebuild_with`) remakes the index from those sidecars and the
  fingerprint table by reading every object back. Checked live: two
  kernel tarballs put to S3, the metadata directory deleted, rebuilt
  in 15 s, verified, read back byte-exact, and a third version then
  stored as a 1.8 MB delta against the second.
- `Backend::list` on both backends.

## v0.11.1 — 2026-09-21

- Multipart upload: objects over 64 MB go to S3 as 64 MB parts on up
  to 8 connections and one completion, so objects up to 640 GB store
  and fat links fill; a failed part or completion aborts the upload,
  leaving no parts behind to be billed. Checked live: a 17 MB object
  in four parts read back whole, undersized parts refused and aborted
  cleanly, the 201 MB kernel object in three parts byte-exact. On a
  home uplink the two-kernel put took 12.2 s against 11.6 s single-put:
  the link, not the client, is the limit there.

## v0.11.0 — 2026-09-21

### The store speaks S3 itself

- `S3Backend` replaces `S3Cli`: PUT, GET, HEAD, DELETE and
  ListObjectsV2 over HTTPS with Signature V4, no AWS CLI on the
  machine. Credentials from the environment, `~/.aws/credentials`
  (`AWS_PROFILE`) or the instance/container role, refreshed before
  they expire; the region from the environment, the profile, or the
  bucket's answer; `AWS_ENDPOINT_URL` for MinIO, R2, B2, Ceph and other
  S3-compatible services. Retries with backoff on 5xx and throttling.
  Two 1.5 GB kernel tarballs put through S3, one as a 3.4 MB delta,
  read back byte-exact and verified; the two-object put took the same
  11.6 s as through the CLI on this link. Single puts up to 5 GB;
  multipart upload is next.
- `--audit s3://...` lists and reads through the same client.
- The `glyd-store` crate takes its first dependencies for this:
  `ureq` (HTTPS through rustls), `sha2`, `hmac`. The `glyd` crate
  stays at zero.

## v0.10.2 — 2026-09-21

- The codec's licenses are now exactly zstd's: BSD 3-Clause (`LICENSE`)
  or GPL version 2 (`COPYING`), at the user's option, replacing
  Apache-2.0 OR GPL-2.0. Whatever may ship zstd may ship Glyd. The store
  stays under BUSL 1.1.
- `glyd.h` and the CLI banner no longer claim GPU or AVX-512 kernels;
  there are none (AVX2 and NEON only).

## v0.10.1 — 2026-09-21

- The codec is dual-licensed, Apache-2.0 or GPL-2.0 at the user's
  option (`LICENSE`, `LICENSE-GPL2`), the choice zstd offers, so GPLv2
  projects can carry it too. The store stays under BUSL 1.1.

## v0.10.0 — 2026-09-21

### Installing
- `brew install surya-koritala/glyd/glyd` (the tap
  github.com/surya-koritala/homebrew-glyd; installs both CLIs and
  `glyd.h`); every release carries prebuilt CLIs, libraries and Python
  wheels for Linux x86_64 / aarch64 and macOS arm64
  (`.github/workflows/release.yml`), and publishes to crates.io and
  PyPI once those tokens are repository secrets.

### The codec under Apache-2.0; the store its own crate under BUSL
- The `glyd` crate (the codec, every level and mode, base mode, packs,
  shape dictionaries, the C ABI, the `glyd` CLI) is licensed under the
  Apache License 2.0. The store moved to the `glyd-store` crate
  (`glyd-store/`, a workspace member) under the Business Source License
  1.1, with its own CLI (`glyd-store DIR --put ...`, `--audit`) and its
  C ABI in `libglyd_store`, which carries the codec's ABI too. The
  bindings load `libglyd_store` when present and `libglyd` otherwise
  (everything but `Store`). `Mapping` moved to `glyd::mmap`.

## v0.9.3 — 2026-09-21

### Adoption: bindings, the audit, the spec, lzbench, packaging
- C ABI, allocating form (`include/glyd.h`): `glyd_compress2` (every
  level, record mode, thread count), `glyd_decompress2` (any stream),
  `glyd_decompressed_len`, base mode, packs (`glyd_pack`,
  `glyd_unpack_object`, `glyd_pack_len`), the store (`glyd_store_*`:
  open with a directory or an S3 url, put, get, id_of, delete,
  compact, flush, rebase, verify, stats, set_level, count), `glyd_free`;
  `glyd_compress_fast` / `_turbo` in the buffer form; `glyd_version`
  reports the crate's version. Every decoder now reads a pack (its
  objects back to back).
- Python (`bindings/python`, ctypes, `pip install`) and Go
  (`bindings/go`, cgo) bindings over that ABI, each with a test that
  round-trips every level, record mode, base mode, packs and the store.
- `glyd --audit DIR|s3://bucket/prefix [--sample N]`: a sample of the
  objects (runs of consecutive names at eight places in the listing)
  through a store, against zstd -3 per object when the CLI is there,
  scaled to the listing with the yearly cost at S3 Standard's list
  price.
- `docs/spec.md`: the formats as a map for decoder writers (block
  framings and flags, every envelope's layout, the store's files).
- `contrib/lzbench`: Glyd in lzbench (`setup.sh <checkout>` builds it
  in; levels 1 default, 2 fast, 3 turbo, 4 max, 5 ultra); run and
  checked on an lzbench checkout.
- Packaging: crates.io metadata (the crate excludes the corpus, the
  benchmarks and the bindings), a Homebrew formula (`Formula/glyd.rb`).

## v0.9.2 — 2026-09-21

### Write speed, and the corpus rerun on AWS
- The suite (`scripts/bench_aws_suite.sh`, both machines, every decode
  checked; [docs/benchmarks/suite-2026-09-21.md](docs/codec/benchmarks/suite-2026-09-21.md)):
  `--max` 3.939 over the 8.7 GB corpus (zstd -3 3.851) at 1,512 MB/s
  on 8 Graviton3 cores against zstd -3's 1,969 — 0.77× (0.61× in the
  previous report), decoding at 10,413 against 1,422; one core 233
  against 313 (0.74×), decode 1,571 against 1,425. Sapphire Rapids:
  1,122 against 1,513 at 8 threads, 266 against 377 on one. `--max -r`
  4.712 at 643 MB/s; `--ultra` 4.663 (zstd -19 4.664); `--ultra -r`
  5.224. In the S3 workflow `--max`'s reads now cost less CPU than
  zstd -3's on Graviton3 (9.7 against 10.7 s over the corpus), so it
  is the cheapest row at every read rate: a terabyte-year at 1 / 10 /
  100 reads a month $71.6 / $81.3 / $178 against zstd -3's $73.3 /
  $84.1 / $191; `--max -r` $61.7 / $82.4 / $289.
- The max level's finder tables are zstd -3's size (17/16 bits, 768
  KB) instead of 2 MB: on Graviton3 and Sapphire Rapids the 2 MB
  tables ran 10–25% slower for 0.5–1.5% fewer bytes (on an M1 the
  difference is 2–7%). One core, `benchmarks/max`: JSON events
  330/463 MB/s against zstd -3's 440/633 (Graviton3 / Sapphire Rapids),
  a table dump 301/420 against 331/462, a root filesystem 178/248
  against 115/294. Where the time goes: the parse 45–80%, the
  long-distance pass 7–45% (JSON), the entropy coder 10–18%.
- The CLI maps its input file instead of reading it first, and at
  `--max` and `--ultra` writes each unit as it finishes from a writer
  thread (`compress_stream`), so the output never sits whole in memory
  and the write overlaps the compressing: a 512 MB file at `--max` on
  ten cores 1,060 → 1,500 MB/s (a root filesystem) and 2,090 → 2,940
  (JSON events); in-process the level runs at 1,760 and 4,200.
- Measured and not taken: probing only the first repeat offset (no
  faster, 2–3% more bytes); 16/15 tables (5–12% faster again, 1–2%
  more bytes).

## v0.9.1 — 2026-09-21

### The store: S3, rebase, a second candidate
- `Backend` trait for the objects' bytes: `LocalBackend` (a directory)
  and `S3Cli` (an S3 bucket through the AWS CLI, `aws s3 cp` per
  object; the library carries no HTTP client); `Store::open_with`.
  CLI `--s3 s3://bucket/prefix`. Metadata (index, table) stays local.
  Round trip through a real bucket verified.
- `rebase(id)` / `--rebase ID`: an object stored alone again, one
  decode to read; a later index line for an id replaces the earlier.
- Two base candidates when the second scores at least half the first:
  both tried on the first 32 MB, the smaller delta wins the object. On
  the bucket: 1,313.6 -> 1,312.1 MB (the single candidate was already
  within a few percent of ideal), put 63 -> 70 s.
- The per-object fingerprint files are gone (the table on disk is the
  record).

## v0.9.0 — 2026-09-21

### The store, complete
- `delete(id)` (a tombstone in the index; the bytes stay while a live
  object's chain runs through them), `compact()` (removes what no live
  object needs: deleted objects off every live chain, packs whose
  members are all deleted; returns the bytes freed), `verify()` (every
  live object read back and checked), `id_of(name)`, `set_level`
  (`Max`, `Ultra`, `Cold` for objects stored alone and packs; deltas at
  the ultra level under `Ultra`). Deleted objects are never chosen as
  bases. CLI: `--find NAME`, `--delete ID`, `--compact`, `--verify`;
  `--ultra` / `--cold` with `--store` set the level.
- Put keeps the last large object in memory as the likeliest next base,
  and judges a delta against the object alone estimated from its first
  64 MB when that settles it either way (a version's delta is a few
  percent of the estimate, an unrelated object's about all of it),
  compressing the whole object alone only in between. The bucket put
  79 -> 63 s (620 MB/s end to end), same bytes; a version of the last
  object put runs at 900 MB/s. An estimate used alone, without the
  exact check in between, chose bases that were not worth it (the
  bucket 1,314 -> 1,890 MB) — measured, and not shipped.

## v0.8.1 — 2026-09-21

### The store at scale
- The fingerprint table lives on disk: an open-addressing hash table
  (12-byte slots, the fingerprint and the object id, linear probing, a
  fingerprint's last eight holders kept) mapped into memory through
  libc's `mmap`, at most half full, doubled in a fresh file when it
  fills. The store's memory no longer grows with what it holds: the
  39 GB bucket's table is 100 MB on disk and the put's memory is the
  object's own working set. Same bytes (1,314 MB, 29.9x), 480 MB/s end
  to end.
- Small objects (under 256 KB) go into packs of about 2 MB, one stored
  object each (`flush` writes the open pack; `Drop` flushes); `get`
  decodes the pack and slices, keeping the last pack decoded. 2,000
  GitHub events put one by one: 9.0x against zstd -3's 3.6x per event.
- Objects stored alone go through record mode where it pays
  (`compress_records_into_max`), so a log or a dump put into a store
  gets its columns.

## v0.8.0 — 2026-09-21

### The store: compression across objects
- `Store::open(dir)`, `put(name, data)`, `get(id)`, `entries`, `stats`;
  CLI `--store DIR --put FILE...`, `--get ID -o`, `--stats`. Each
  object's fingerprints (one sparse anchor in 4 KB) are looked up in
  the store's table (the last eight holders of each); the stored object
  sharing the most is its base, taken when the delta (`--base`) saves a
  fifth or more of the object alone; chains at most four long, the
  chain's root past that. A 39 GB bucket (six Ubuntu image builds, the
  fifteen Linux 6.10 releases, two months of three Wikipedia tables,
  twelve hours of GitHub events; `scripts/download_bucket.sh`): 1,334
  MB against zstd -3's 6,132 MB per object, 4.6x, put at 500 MB/s end
  to end and read back at 270 MB/s with the file written, every object
  byte-exact; by family 13.9x, 5.2x, 2.0x, 1.3x. The
  research pass that led here: experiments/research/README.md, H.

### Packs: small objects as one stream
- `compress_pack(objects, out, level)`, `decompress_pack`,
  `decompress_pack_object(pack, i)`, `pack_len`; CLI `--pack files...`,
  `--unpack dir`. Envelope `GLYDPACK`: the count, the lengths as
  zigzag deltas compressed at the max level, then the concatenation in
  record mode where it pays. 1 MB packs of 1 KB objects at `--max`:
  JSON events 42.4x (zstd -3 + dict per object 10.6x), NASA log 15.2x
  (5.3x), HDFS 16.2x (6.0x), CSV telemetry 8.9x (3.8x), taxi CSV 7.6x
  (3.9x); 40-130 MB/s to pack, an object read back in 0.5-1.7 ms.
  Record mode's pay decision now samples an eighth of a small input
  (at least 256 KB) instead of the whole of it.

### Shape dictionaries: record mode for small objects
- `ShapeDict::train(sample)`, `compress`, `decompress`, `to_bytes`,
  `from_bytes`; CLI `--shape-train`, `--shape`. The dictionary carries
  the shape (delimited, JSON lines, or a log's skeletons: punctuation
  between runs of letters and digits), the frames lines take with a
  column per hole, the columns' types (integers with a recency ring
  and deltas, times, decimals, dictionaries seeded with the sample's
  values by frequency, constants, text) and an LZ `Dict` trained on
  such objects' images. An object is a compact image: a byte per row,
  then every column's values, each column self-delimiting; unknown
  lines stay raw. 1–4 KB objects cut from real files: JSON lines 14.6×
  and 24.3× (zstd -3 + dict 10.6× and 13.0×), CSV telemetry 5.8× and
  7.9× (3.8×, 4.4×), HDFS log 6.7× and 9.7× (6.0×, 7.4×), NASA log
  5.1× and 7.3× (5.3×, 6.4×). 60–150 MB/s to code, 55–430 MB/s to
  decode, one core. The same objects packed into one record-mode
  stream cost 2–4× less than zstd + dict per object.

## v0.7.0 — 2026-09-20

### The cold level: context mixing
- `glyd --cold` (`compress_into_cold`, `compress_parallel_into_cold`,
  `compress_records_into_cold` with `-r`): every bit predicted from
  eleven contexts (byte orders 1-4, 6, 8; the word and the one before;
  the column and the byte above; the JSON key; the longest earlier
  match) through paq-style bit histories, mixed by two networks, two
  SSE stages, a binary arithmetic coder; 32 MB units coded from an
  empty model, in parallel, each with a checksum in the envelope
  (`GLYDCOLD`); every decoder reads it. 64 MB slices, two threads: JSON
  events 22.5x (zstd -19 14.6x, `--ultra` 15.9x, zpaq -m5 22.8x), NASA
  log `-r` 31.1x (15.7x, 26.4x, 31.7x), page_props dump `-r` 11.6x
  (6.2x, 8.6x, 11.1x), webster 7.1x (4.8x, 4.8x, 7.3x); the HDFS and
  Spark logs (128 MB, `-r`) 33.7x and 65.2x against zstd -19's 16.0x
  and 25.2x; 1.2-1.5 MB/s per core each way, 400 MB per thread. Record mode decides whether
  its transform pays at the max level whatever the level.

## v0.6.0 — 2026-09-20

### Base mode: content found wherever it moved, ultra at full speed
- Each unit's region of the base is chosen from a coarse map of the
  base (its sparse anchors, one per KB, found with the matcher's vector
  scan at 3-4 GB/s and sorted by the hash of the 32 bytes at each): the
  96 MB window holding the most of the unit's own anchors, or the base
  around the unit's position when its content is new. A version with
  48 MB inserted before the kernel tree costs 18.4 MB against 33.5 MB
  with the fixed window (zstd -3 --patch-from: 18.7 MB). The consecutive
  pairs are unchanged within 0.5%; `--max --base` runs at 720-1,700
  MB/s on ten M1 cores (was 860-2,020: the map's cost).
- `--ultra --base` inserted each unit's whole 96 MB region into the
  tree finder, which reaches 8 MB back: it now starts at the window's
  edge and runs at the plain `--ultra` speed, 3-11 MB/s on ten M1 cores
  against 1-4 before (the kernel pair 483 s, the new version alone at
  `--ultra` 555 s), the bytes the same.
- Chains measured (`scripts/download_chain.sh`, `scripts/bench_chain.sh`):
  the 15 Linux 6.10 point releases cost 228 MB each against the one
  before (zstd -3 --patch-from 260 MB; stored one by one 3.0-3.2 GB) or
  246 MB each against 6.10 (zstd 265 MB), a step 1.8 MB and the delta
  against a base 14 releases old 3.6 MB.

### Record mode: templates
- Logs whose lines vary in shape (application and system logs) take a
  fourth shape: each line's template (its text with a hole where every
  token holding a digit was) goes into a dictionary, and the tokens
  become typed columns keyed by template and slot; lines past the
  column budget stay raw. loghub 2.0, 128 MB of each, 10 cores: HDFS
  `--max -r` 22.0 against zstd -3's 10.5 and zstd -19's 16.0 (`--ultra
  -r` 27.5), Spark 47.0 against 14.5 and 25.2 (53.6), BGL 15.9 against
  11.0 and 22.4 (28.9), Android 17.9 against 12.9 and 23.0 (25.4);
  writes at 260-460 MB/s, reads at 1,200-1,400 MB/s. The levers were
  sized first (experiments/research/README.md): version chains, these
  log shapes, context mixing for cold data, float columns.

### Reads
- The record-mode rebuild decodes a column at a time into tables and
  assembles the rows by copy (reserved space, no length branch per
  value; integers through a digit-pair table; the minute's prefix of a
  time column kept and copied as a block; a ring for the recency list;
  16-byte padded dict8 entries; varints of up to three bytes from one
  load). One core, M1 Max: int columns 489 -> 810 MB/s, dictionaries
  344 -> 1,076, times 477 -> 1,815; the NASA log 509 -> 734, JSON
  lines 734 -> 1,117, the taxi CSV 260 -> 409. Record images are
  unchanged.
- The S3 workflow rerun on both AWS machines: the CLI's decompress
  CPU per 8.7 GB fell from 14.5 to 11.7 s on Graviton3 for `--max`
  (zstd -3: 10.9) and from 19.2 to 14.2 on Sapphire Rapids (zstd -3:
  10.4); `--ultra` 11.1 against zstd -19's 12.5. A terabyte-year at
  ten reads a month: `--max` $83.1, `--max -r` $83.6, zstd -3 $84.2.
- The reference codecs (zstd, LZ4, Snappy, LZAV) are dev-dependencies:
  the benchmarks link them, the library and CLI carry none.

## v0.5.0 — 2026-09-20

### Base mode
- `glyd --base old new` / `compress_with_base`: a new version of an
  object compressed against the old one, decodable with it (`glyd -d
  --base old`, `decompress_with_base`). Units of 32 MB are parsed with
  the base around their own position as history (32 MB of slack each
  way), the long-distance matcher reaching all of it; the decoder reads
  the base in place. Against zstd 1.5.7 `--patch-from` on the same
  machine, byte-exact: Wikipedia page-table dumps a month apart
  `--max` 1.79 MB at 863 MB/s (zstd -3 patch 3.84 MB at 409, zstd -19
  patch 1.30 MB at 2), `--ultra` 1.23 MB; Ubuntu cloud root filesystems
  16 days apart 5.31 MB at 2,018 MB/s (zstd -3 8.82 MB at 654, zstd -19
  5.61 MB at 39), `--ultra` 4.59 MB; Linux 6.10 -> 6.10.1 3.04 MB at
  1,980 MB/s (zstd -3 3.26 MB at 560, zstd -19 2.58 MB at 30), `--ultra`
  2.04 MB. Plain `--max` on those
  files: 33, 287 and 200 MB. Design notes in docs/design/format-v7.md;
  the measurement that led here in experiments/structure/README.md.
- A far match's cap of 130 bytes per sequence applied to the
  repeat-offset continuation as well; a repeat carries no offset bits,
  so the rest of the match is now one sequence. Plain `--max` gains 1%
  on JSON events.
- The long-distance matcher's table grows with the input (a slot per
  16 bytes, up to 2^25 entries).
- The CLI decodes a batch of units at a time into one reused buffer
  and writes as it goes (`decompress_stream`): the memory is a batch,
  not the file, and no output page is touched for the first time after
  the first batch; the library's parallel paths run on scoped worker
  threads (`set_threads`) instead of rayon.

## v0.4.0 — 2026-09-20

### Long-distance matching
- The max and ultra levels find repeats of 32 bytes or more up to 128 MB
  back (`src/ldm.rs`): one pass over the input before the parse, with
  content-defined anchors (one position in 16, found 16 at a time with
  NEON or AVX2) hashing the 32 bytes after them into a 16 MB table whose
  entries carry a hash check; matches are verified, extended both ways
  and handed to the parse, which takes one wherever it beats the local
  finder. Format v9 offsets grow to 27 bits (30 offset codes; v8 blocks
  keep 26, and a table with fewer symbols than its version allows still
  decodes). A far match is capped at 130 bytes per sequence so the
  decoder's one-load walk holds its extra bits, the rest following as a
  repeat-offset sequence. Parallel units grow to one per core, up to
  128 MB (the matcher's reach is the unit).
- The pass runs at 1.2-2 GB/s on one core. The max level gates it:
  after 4 MB and 16 MB (or half the input) it stops on data whose
  repeats are too few or too near to pay (media, Parquet, most SQL
  dumps), which keep 92-96% of their speed; the ultra level runs it
  whole. Where it stays on, the max level compresses at 63-85% of its
  former speed for 3-16% fewer bytes. One core, M1 Max, 128 MB of
  GitHub Archive JSON: `--max` 11.63 -> 9.73 MB at 577 MB/s (zstd -3
  12.90 MB at 895; `zstd -3 --long=27` 10.20 MB at 433), `--ultra`
  7.78 MB (zstd -19 8.96, `zstd -19 --long=27` 7.80); NASA access
  log 13.22 -> 11.92 MB at 435 MB/s (`zstd -3 --long=27` 13.63 MB at
  383); Silesia `--max` 3.259 -> 3.302 at 247 MB/s (290 before; zstd
  -3 3.205 at 335). The 8.7 GB corpus on 10 cores: `--max` 3.89 ->
  3.96 at 2,000 MB/s (2,400 before; zstd -3 3.85 at 4,000), JSON
  events 11.49 -> 13.26 (zstd -3 10.46); `--ultra` 4.65 (zstd -19
  4.66), JSON events 16.40 (zstd -19 15.07).

### Record mode
- `glyd -r` / `compress_records_with`: delimited lines, SQL dumps and
  JSON lines become typed column streams (integer, decimal and
  date-time deltas, dictionaries with recency ranks, text) before the
  level, in parallel 32 MB units, rebuilt byte for byte; other data is
  left as it is; input the transform does not pay on (API events with
  hashes and free text, binaries) takes the plain parallel path. JSON
  lines: a column per key path, typed values leave holes in a frame of
  the structure, keys and text. Telemetry as rows, 128 MB slices, 10
  cores: a cluster trace as CSV `--max -r` 12.6 against zstd -3's 4.5
  and zstd -19's 6.9, as JSON lines 54.6 against 15.7 and 28.7; daily
  weather 19.6 / 47.0 against 7.0 / 18.7 and 12.0 / 31.6; taxi trips
  exported to CSV 8.9 against 5.6 and 8.4 (`scripts/download_ext_corpus.sh`). Whole 8.7 GB corpus, 10 cores: `--ultra -r`
  5.20 against zstd -19's 4.66 (SQL dumps 1.43x smaller, access logs
  1.54x, JSON 1.09x through the plain level's matcher); `--max -r`
  4.75 at 1,100 MB/s. Design notes in
  docs/design/format-v7.md; the prototypes and measurements that led
  here in experiments/structure/.

### Small objects and dictionaries
- `Dict`: a prepared dictionary (trained content plus entropy tables)
  for small objects; `Dict::train` (cover selection as zstd's fastcover,
  scoring each distinct string once), `to_bytes`/`from_bytes`,
  `compress_with_dict`, `compress_with_dict_ultra`,
  `decompress_with_dict`. The object is parsed in place against the
  dictionary's own seeded tables; the decoder copies from the content
  and borrows the dictionary's built tables.
- Format v9: compact framing for blocks of at most 32 KB (one marker
  byte, varint lengths, a 5-10 byte sub-header, single-stream sections
  under 1,024 symbols, no padding on disk): 207 -> 21 bytes of framing
  on a 4 KB object. Every earlier format decodes unchanged
  (tests/format_compat.rs holds v0.2.0, v0.3.0 and v0.4.0 output).
- The ultra level with a dictionary prices its parse from the
  dictionary's tables.
- Per-object work cut: entropy tables and codes built once per
  dictionary, table costs from a lookup, buffers kept across calls,
  single-stream decode paths (three code chains at once, one-load
  batches).

GitHub Archive JSON objects, Apple M1 Max, one core, 110 KB
dictionaries trained on other objects (zstd's numbers without a
checksum; Glyd writes 4 bytes per object): 1 KB objects `--max` + Dict
ratio 4.75 (zstd -3 + dict 4.96), compress 245 MB/s (454), decode 920
MB/s (1,117); 4 KB 6.28 (6.42), 321 (588), 1,209 (1,440); 16 KB 7.65
(7.66), 394 (635), 1,674 (1,890). `--ultra` + Dict: 5.25 / 7.13 / 8.84
(zstd -19 + dict 5.47 / 7.41 / 8.95). Before this work the same objects
compressed to 2.15 / 3.87 / - with a window-only dictionary at 6 MB/s
and decoded at 170 MB/s.

### Verification and benchmarks
- `scripts/verify_roundtrip.sh` (every level and core mode through the
  CLI, byte-compared; corrupted copies rejected or decoded exactly),
  `scripts/download_bench_corpus.sh` (~9 GB of logs, JSON, SQL dumps
  and Parquet with a separate training set), `examples/bench_suite.rs`
  (Glyd against zstd -3, zstd -19 and LZ4 at a stated thread count,
  every decode checked, peak memory, small-object latencies),
  `scripts/s3_workflow.sh` (compress, upload, download, decompress,
  verify, monthly cost), `scripts/bench_aws_suite.sh` (all of it on
  Graviton3 and Sapphire Rapids), `scripts/report_suite.py`.
- The format-compatibility fixtures are now committed (they were
  ignored by the `*.glyd` rule; CI failed on every push since they were
  added).
- Results of the program on Graviton3 and Sapphire Rapids:
  docs/benchmarks/suite-2026-09.md (raw rows in benchmarks/suite/).

### Parallel paths
- The parallel compressors cut the input into units of at least 2 MB
  (v6 levels), 8 MB (`--max`) and 16 MB (`--ultra`) instead of 256 KB,
  growing with the input (two units per core, up to 64 MB): JSON events
  lost 4.7% to 8 MB units against the sequential ratio, 1% at 64 MB. Each unit is
  still a chain of its own (parallel decode, random access), and the
  ratio now stays within 0.5-0.7% of the sequential path; at 256 KB the
  CLI's multi-core default was giving up 3% (default level), 6.6%
  (`--max`) and 13% (`--ultra`). Multi-core decode of files with few
  units is correspondingly less parallel (Silesia on 10 cores: 31,900
  MB/s against 42,900).

### Ultra level
- Blocks are split where the parse's statistics change
  (`v7_ultra::split_points`); prices carry the parse's own prior at
  weight 2 and half a bit per literal. Silesia 3.925 → 3.946 (zstd -19:
  4.006), decode 2,150 → 2,090 MB/s.
- x86-64: the walk's and the tANS batch's per-stream state through
  memory: +6% max-level decode on Sapphire Rapids.

## v0.3.0 — 2026-09-19

### Format v8
Every level of the entropy-coded family now writes format v8; v7 (and
v6) files from earlier releases decode unchanged (tests/format_compat.rs
holds v0.2.0 output as fixtures).
- 8 MB window (26 offset codes).
- Section layout: 24-bit sub-stream sizes and one padding per section
  instead of per stream; tANS counts as width + mantissa; literal tables
  as nibbles with unused-symbol runs. Per-block overhead 950 → 421
  bytes.
- Length codes with direct codes to 15 (runs) and 34 (matches) and
  short buckets before the log2 ones.

Silesia, M1 Max, same run: `--ultra` 3.80 → **3.93** (zstd -16 3.83,
zstd -19 4.01), decode 2,150 MB/s (1.3× zstd -19); `--max` 3.22 →
**3.25** (zstd -3 3.20), decode 1,890 MB/s (1.27× zstd -3). Design
notes: docs/design/format-v7.md, "Format v8".

### Library
- The ultra finder sizes its tables to the input per call: 2 MB for a
  256 KB chunk, 64 MB for an input that fills the window.
- `bits::Stream` (a sub-stream with its own length and the bytes to the
  section's end) replaces bare slices in the decoders' signatures.

## v0.2.0 — 2026-09-19

### Levels
- **`--ultra` / `-19`** (format v7, same decoder): optimal parse on a
  binary-tree match finder, every position priced in the coder's own
  bits ([design](docs/codec/design/ultra-parse.md)). Silesia ratio 3.80 vs
  `--max`'s 3.22; zstd -16 3.83, zstd -19 4.01 (3.91 inside Glyd's 2 MB
  window). Compresses at 4.8 MB/s; its output decodes at 2,190 MB/s,
  1.3× zstd -19's. `compress_into_ultra`, `compress_parallel_into_ultra`,
  `compress_with_dict_ultra`; C `glyd_compress_ultra`,
  `glyd_compress_ultra_parallel`.

### Platforms
- x86-64 `--max` decoder: an AVX2+BMI2 entry point and loop shapes for
  16 registers (stream-major entropy batches, the NEON copy structure,
  split walk tables). Sapphire Rapids decode 858 → 1,300 MB/s in the
  published run (zstd -3: 1,260 MB/s); default builds gain the same.
- Cross-platform benchmarks re-run; `ultra_bench` added to the script.

### CLI
- `-19` / `--ultra`; `--single-core` is now `-s` (`-1` was `--fast`).

### Fixed
- Nothing user-visible; see CHANGELOG-BENCH.md for the measurement trail.

## v0.1.0 — 2026-09-19

First public release.

### Levels
- **default** (format v6): LZAV-class parse, minimum match 7. Silesia
  ratio 2.19, decode 6,900 MB/s (1.6× liblz4), compress 340 MB/s.
- **`--fast` / `-1`**: LZ4-class finder, minimum match 5. Ratio 2.18,
  decode 4,900 MB/s, compress 550 MB/s.
- **`--turbo` / `-t`**: minimum match 10. Ratio 1.88, decode 9,200 MB/s
  (2.1× liblz4).
- **`--max` / `-9`** (format v7): 8-way interleaved Huffman literals,
  tANS-coded sequences with repeat offsets, 2 MB window, double-fast
  lazy parse; three-pass decoder. Ratio 3.22 vs zstd -3's 3.20, decode
  1,860 MB/s (1.3× zstd -3), compress 300 MB/s.

### Platforms
- aarch64 NEON decoders for v6 and v7; x86-64 AVX2 decoder for v6 and a
  portable scalar path everywhere else.
- Multi-core compression and decompression over independent 256 KB blocks.

### APIs
- Rust: `compress_into{,_fast,_turbo,_max}`, `compress_parallel_into*`,
  `decompress`, `decompress_into`, `decompress_parallel*`,
  `compress_with_dict` / `decompress_with_dict` (v7), `GlydReader` /
  `GlydWriter` (v6 streaming).
- C ABI (`include/glyd.h`, `libglyd`): `glyd_compress*`,
  `glyd_decompress*`, `glyd_max_compressed_len`, `glyd_version`.
- CLI `glyd`: compress/decompress, level flags, pipes, `-b` benchmark.

### Safety
- Every decoder fuzzed with 1,000,000 random mutations per run into
  exact-size buffers with sentinel guards; no per-call allocation in the
  decoder (1.5 MB thread-local scratch for v7).

### Known gaps
- `--max` compresses at ~90% of zstd -3's speed and decodes 1.3× (not 2×).
- `--max` beats zstd -3 on 3 of 5 extended-corpus files (loses 0.6% on
  repetitive JSON).
- No AVX2 v7 decoder yet (scalar on x86); streaming adapters are v6-only.
