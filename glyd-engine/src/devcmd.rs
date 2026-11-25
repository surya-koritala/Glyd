//! The commands of a development build (`--features dev`, the default): the checks, the proofs and the measurements. A release build has none of them.
//!
//!   glyd-engine pack MODEL OUT [--threads N] [--layout auto|mma|mma12|i6|i5|i4]   pack a checkpoint (every matrix, the embedding too); i6, i5 and i4: a lossy level's glyd-v7 directory (Qwen3.5, Nemotron 3 Super: at i4 its experts in four-bit codes, every other matrix in six-bit ones; GLM-5.3-Flash)
//!   glyd-engine calibrate-experts MODEL OUT [--tokens 32768] [--devices 0,1 --split CUTS] [--context 4096] [--synthetic]   hummingbird's corpus pass (src/hb_usage.rs): the model on the engine over the
//!                                                             calibration corpus, each MoE layer's router choices counted per expert into the usage file OUT
//!   glyd-engine tiercheck MODEL [--places 2] [--tokens 24] [--mode copy] [--layout i6] [--dense N] [--device-plan] [--against copy]   the dynamic route's model with every expert on GPU 0, then in the expert cache (src/tiering.rs; --dense: its last N layers'
//!                                                             dense matrices in the computer's memory too, src/dense_ram.rs): every position's logits compared, the same bits in copy mode
//!                                                             (--device-plan: the decode calls planned on the GPU, src/devplan.rs, and the decode step one graph, checked; --pairs: each host pair against
//!                                                             the GPU's; --join: each call through a host half against copy mode from the same inputs)
//!   glyd-engine actcheck   the activations the expert cache's host computes beside the GPU (situ, the release's SiLU), the host's against the GPU kernels' on every bf16 input up to 64 (src/cpu_fp4.rs)
//!   glyd-engine storebench MODEL [--places 16] [--rows 1,8,32] [--steps 8] [--mode auto] [--device-plan] [--out FILE]   batched decode through the expert cache: each count of rows' step time and layer statistics (bytes read for
//!                                                             the misses against the distinct missed experts'), FILE's JSON lines for ceiling --stats (src/tiering.rs)
//!   glyd-engine pack MODEL OUT --layout hummingbird --usage FILE [--threads N]   a mixture of experts the description builder maps (Kimi-K3, DeepSeek-V4, GLM-5.3, Muse Spark, Qwen3.5/3.8 MoE) at the
//!                                                             hummingbird level (src/hb_pack.rs): each expert's bits from FILE's counts (calibrate-experts), its rows coded in Rust, the rest copied
//!   glyd-engine pack MODEL OUT --layout hummingbird --codes DIR [--threads N]   Nemotron 3 Super at the Hummingbird level (src/hbpack.rs): its routed experts' neuron rows from the export in DIR (nemotron_hbr.py --export) in codebook rows of 3 and 2 bits,
//!                                                             the signs of each layer's rotation beside them, every other matrix six-bit; glyd-gpu verify decodes the directory
//!   glyd-engine hbtoy MODEL OUT [--seed N] [--share PERCENT]   a tiny Nemotron-H model (nemotron_ref.py moetoy's toy/) packed at the Hummingbird level from a random export (OUT/export, PERCENT of the experts at 3 bits, the rest at 2) into OUT/packed
//!   glyd-engine cb2prefill [--n 1024] [--k 8] [--experts 216] [--check 16] [--reps 3] [--skew 0] [--pack DIR --layer L]   a prompt chunk through GLM-5.3-Flash's codebook experts on the tensor cores (src/cb2_prefill.rs)
//!   glyd-engine glmfmtcheck FIXTURE_DIR   the four formats of the pack (tern, cb2, cb3, cb4) against fixtures the pack writer made (tools/fmtfix.py)
//!   glyd-engine cb2check PACK_DIR [--layer L] [--expert E]   one expert of a glyd-hbr-codes/2 pack (src/cpu_cb.rs) against the pack's testvec.safetensors
//!   glyd-engine hbcheck [CASE...] [--speed] [--no-check]      the Hummingbird layer's kernels (src/moe_hb.rs, kernels/moe_hb.cu) against the CPU reference on random experts at 3 and 2 bits, all there and through places, and their speed (src/kcheck_hb.rs)
//!   glyd-engine run MODEL [--prompt TEXT] [--no-think] [--context N] [--port N] [--host H] [--temp T]
//!                                                             what `glyd run` starts: the model made ready on the first run (downloaded, packed for this GPU
//!                                                             at its level), then a chat in the terminal, or one answer with --prompt
//!   glyd-engine serve MODEL [--port N] [--host H] [--context N]   the OpenAI-compatible API and the chat page
//!   glyd-engine fetch MODEL                                      the model's files downloaded into the Hugging Face cache (no GPU)
//!   glyd-engine prepare MODEL[:LEVEL]                          a model downloaded and packed for this GPU at its level (any the engine runs: what `run` does first)
//!   glyd-engine doctor | login | supports MODEL                  what this computer has for it; a Hugging Face token saved; whether the engine runs MODEL (exit 3: no)
//!   glyd-engine generate DIR PROMPT [--new N] [--ctx N] [--no-graph] [--expert-cache GIB|auto|auto:GIB] [--host-memory GIB|auto] [--offload-mode MODE[,MODE...]] [--cpu-threads N] [--cpu-pin]   a raw completion (a Nemotron 3 Super directory `pack` made, or a bf16 one with
//!                                                             --layout i6|i5|i4; --expert-cache: its experts in the computer's memory and a cache of them in GIB GiB of the GPU's, src/offload.rs); the prefill and the decode steps timed apart, with the cache's
//!                                                             counts for each; --offload-mode copy|cpu|split|auto[:THREADS][:pin] (a list runs the completion once for each, the cache cold at the start of each): what the cache does with an expert that is not
//!                                                             on the GPU (copy it, or compute its pairs on the CPU: --cpu-threads, --cpu-pin, --offload-window N, --offload-promote N); --host-memory GIB: the computer's memory the
//!                                                             experts may take (a Hummingbird directory keeps the rest on the disk and reads them as the router asks, src/offload.rs `Tier`; the counts of reads from the disk, and the
//!                                                             process's peak resident memory, are in the output)
//!   glyd-engine check DIR PROMPTS.json OUT.json [--new N] [--ctx N] [--no-graph]
//!   glyd-engine microbench DIR [--prompt N[,N...]] [--new N] [--ctx N] [--chunk N] [--reps N] [--no-graph] [--layout L]
//!   glyd-engine kvshapes                                     the lossless KV cache of head dims 128, 256 and 512 (no model): values written, read back bit for bit, and decode and prompt attention against the host's
//!   glyd-engine kvprefill [--chunk 256] [--prefixes 0,1024,4096,8192] [--reps 20]   the prompt attention over 12-bit pages at gemma-4-12B's shapes against the pages unpacked to a bf16 ring and the bf16 kernel on it
//!   glyd-engine kvdecode [--lens 40,1024,4096] [--ctx 32768] [--reps 100]   a decode row's attention over 12-bit pages against a bf16 cache of the same values, and its token's 12-bit write, us a call (8 kv heads of 128 with 32 query heads, gemma-4-12B's and the Qwen3.5 layout's shapes)
//!   glyd-engine topkcheck                                     the sampling kernels' 64 largest logits of a row and sum of its exponentials against the host's
//!   glyd-engine sampletime DIR PROMPT [--new N] [--ctx N]     a Qwen3.5 or Nemotron-H model's decode speed by greedy steps, by sampled ones from the GPU's candidates and from the row on the host (the same tokens)
//!   glyd-engine logitscheck [CASE...] [--fixtures DIR] [--cpu-ref] [--level L] [--list-cases]   each model's engine logits against its fixture in tests/logits (src/logits.rs); logitsrecord CASE --from cpu|engine --out FILE: a fixture made; logitsbound FIXTURE LEVEL=BOUND...: its bounds set; logitsmodel CASE DIR: the case's random model made; paritycheck [CASE...] [--level L] [--choice ROW:LAYER]: each case of a family that still has a hand-written engine (Qwen3.8-Flash-Next's: qwen4x, qwen4x-s2; R3d) on the description executor, held to the CPU reference of the layer kinds on its own routers' choices, and beside the hand-written engine where that runs the case (--choice: no engine; the reference alone with the nearest other choice taken in that row's pass and layer, and what it leaves in the rows after); levelcheck [CASE...] [--level L]: each lossy level's KL from penguin on its families' random models, against the thresholds of the family and level (qwen35, gemma4 and nemotron-h at kestrel, swift and wren; deepseek-v4 and muse at hummingbird; kimi and glm at hummingbird printed as open)
//!   glyd-engine packcheck [--gpu] [--record] [--engine [--scratch DIR]]   every pack format of tests/data/packs read and run on the CPU reference against its digest in tests/packs (--gpu: decoded by the GPU library; --record: the digests written); --engine: the formats the engine reads (glyd-v7 penguin, kestrel, swift, wren) made from the logits cases' random models, loaded from the saved directory, against the same model packed as it loads in the same layout (bits) and the case's reference where one bounds the level, and the Hummingbird level against this GPU generation's digest in tests/packs (--engine --record: that digest written, on a GPU of the generation); a row or a digest with nothing recorded for this GPU fails (src/packcheck.rs)
//!   glyd-engine kcheck IN_DIR OUT_DIR                        the kernels on inputs tools/kcheck.py wrote
//!   glyd-engine usercheck CASE [--users N] [--tokens T] [--dir DIR --prompt-tokens P [--places N [--mode M] [--device-plan]]] [--stagger D]       a random model of the case (gemma4, qwen35, qwen35-wide, qwen35moe, dense, muse, kimi, glm, nemotron, deepseek-v4, qwen4x), or the model in DIR, on the description executor: each reply through the scheduler alone and all at once, against the reply alone; with GLYD_ENGINE_PAGES too few for all at once, at once again swapping sequences sent back for room always, by cost, and never (R4d), each run's first-token and between-token times
//!   glyd-engine roundcheck CASE [--dir MODEL --ctx N --chunk N] [--kl BOUND]   a sequence's logits by what its step holds beside it (other sequences' rows, a prompt's chunk, other prompts in one pass; 2 rows to a chunk's), against its row alone: the bits
//!                                                             that differ, the largest difference, the KL; and that what the other sequence holds changes nothing (src/logits.rs)
//!   glyd-engine keepcheck CASE                                a finished sequence kept in its slot and gone on with (a conversation's next turn): against a sequence that ran the same tokens without ending, bit for bit
//!   glyd-engine twomodels [A B]                               two cases' random models in this process, each with a scheduler and a runtime context of its own: each reply beside the other model against alone
//!   glyd-engine foldcheck [--only WORD]                   R17: what the engine folded from the GPU library (ways, the GPU's class, KV sizes; the products, the KV cache and the FP8 format by its own kernels) against the library's own (--only: the kernels' checks whose line has WORD)
//!   glyd-engine splitcheck CASE [--split CUTS --devices GPUS] [--level L] [--tokens T]   a random model of a dynamic-route family (qwen35, qwen35-wide, glm, kimi, muse, nemotron, deepseek-v4) as `run` opens it, on one GPU and with its layers in stages: every pass's logits the same bits
//!   glyd-engine bufcheck                                  a buffer belongs to its context: one of a second context on GPU 0 written, read, zeroed and freed while the first is current
//!   glyd-engine bandwidth [--gib N]                           the GPU's streaming read bandwidth
//!   glyd-engine rangecheck MODEL[:level] [--prompt 1024] [--new 256] [--cut L | --cuts A,B] [--ctx N] [--chunk N] [--workers [--sessions 2]]   a Qwen3.5 model whole, then as stages of layers chained in this process and (--workers) as worker processes
//!                                                             through the glyd-range/1 protocol (src/dynamic/range.rs, docs/range-worker.md): every step's logits and token compared bit for bit; "all checks passed"
//!   glyd-engine chat MODEL [--ctx N] [--temp T] [--no-think]  a chat in the terminal, also from a pipe (run's, with --chat)
//!   glyd-engine kcheck35                                      Qwen3.5's kernels against the CPU reference (src/kcheck35.rs)
//!   glyd-engine attncheck                                     the attention kind's kernels against an fp64 CPU attention, each family's head and arithmetic (src/dynamic/attn.rs)
//!   glyd-engine attndecode [--lens 64,170,1024,4096,16384] [--ctx 32768] [--reps 200]   a decode row's attention over bf16 pages at gemma-4-12B's and the Qwen3.5 layout's shapes, us a call in a graph: the parts kernel against the library's (src/dynamic/attn.rs)
//!   glyd-engine moecheck [CASE...] [--speed] [--no-check] [--dump DIR | --cpu-dump DIR]   the mixture-of-experts layer (src/moe.rs) against the CPU reference on random routers and Int5, Int6 and Int4 experts, and its speed (src/kcheck_moe.rs)
//!   glyd-engine cpuexperts [--bits 4|5|6] [--threads N] [--kernel auto|scalar|neon|avx2|avx512] [--experts N] [--pairs N] [--reps N] [--tokens N] [--misses M,M,...]   the CPU half of the sparse experts (src/cpu_experts.rs) timed on this computer: Nemotron 3 Super's experts made at random, pairs a second and GB/s read, a decode token's time for M missed experts
//!   glyd-engine nhkernels IN_DIR OUT_DIR                      Nemotron's kernels on the inputs tools/nemotron_ref.py wrote (src/kcheck_nh.rs); nhlayers DIR IO_DIR: layers of the real model or of a random tiny one with mixture-of-experts layers; nhtoy DIR IO_DIR [--expert-cache GIB|auto|auto:GIB] [--offload-mode MODE] [--cpu-threads N]: a few of its layers as a model (or the whole model through the expert cache, one eager run); nhdecode DIR OUT_DIR --layers L,L [--tokens N]: some layers of a glyd-v7 directory decoded to bf16; nhrandom OUT [--latent N]: a random model at Nemotron 3.5 Lightning's widths
//!   glyd-engine diskcheck DIR [--reads N] [--depth N]         the disk a Hummingbird directory is on, as the SSD tier reads it: random whole experts, page cache bypassed, one at a time and 16 at a time, MB/s (src/disk.rs); no GPU
//!   glyd-engine nhoffload DIR IO_DIR [--slots N,N,...] [--offload-mode cpu|split|auto] [--cpu-threads N] [--host-memory GIB,GIB,...]   the random tiny model run with all its experts on the GPU and through the expert cache (src/offload.rs) with N places a layer: the
//!                                                             logits equal bit for bit (the experts' second matrices in the row layout, the sparse product; GLYD_ENGINE_DENSE_DOWN=1: the block layout, the dense one); with a mode that computes the pairs of missed
//!                                                             experts on the CPU, within two places of bf16 of the largest logit (`logits_within`); --host-memory (a Hummingbird directory): the same passes again with every expert in the computer's
//!                                                             memory and with no more than that many GiB of them (the rest read from the directory), every logit the same to the bit in the mode asked for
//!   glyd-engine glmrandom OUT [--fp8] [--real --layers N --experts N] [--seed N]  a random GLM-5.3-Flash in Hub format (src/glm.rs `random`): the executor's logits, user, 12-bit and tier checks' model
//!   glyd-engine kindcheck [CASE...] [--prompt N] [--steps N] [--chain | --sequence] [--scratch DIR]   every layer kind of the families' random models, the executor's kernels against the CPU f32 reference kind by kind (src/dynamic/kindcheck.rs);
//!                                                             --chain: each layer from the residual the executor's layer before it left; --sequence: the same on one sequence's rows (the logits cases'
//!                                                             tokens), and each layer's residual against the reference's own run of them
//!   glyd-engine kv12check [--dir DIR] [--case gemma4] [--level L] [--positions N] [--prompts K] [--prompt-tokens P] [--new T]   a model the executor runs (the case's random one, or DIR's) in bf16 pages and in the lossless 12-bit ones: the same bits (src/dynamic/kvpages.rs)
//!   glyd-engine tpcheck DIR [--tp 0,0] [--split CUT] [--tokens 24] [--chunk 64] [--ctx 1024]   the model of DIR on one GPU, then with every layer across the GPUs of --tp (src/dynamic/tp.rs)
//!   glyd-engine seamcheck [--gpu N]                           the backend seam on a GPU: the dense MLP block through it against the reference, and what `supports` refuses (src/runtime/cuda.rs)
//!   glyd-engine plancheck [--models M,M,...] [--loads 1k:1,4k:1,8k:1,1k:8] [--users 8] [--ctx N] [--runs 3] [--out DIR] [--records FILE] [--predict-only]   the planner against what this machine measures (src/plancheck.rs)
//!   glyd-engine ship MODEL LEVEL DIR [--steps fetch,pack,draft,verify,measure,quality,card,upload] [--usage FILE] [--repo glyd/NAME]   a Hugging Face model to its Glyd repo; a lossy pack is uploaded only with a quality card that says publish (src/ship.rs)
//!
//! `generate`, `check`, `microbench` and `sampletime` run a Qwen3.5-layout model or a plain dense decoder (its config.json says which: a directory in the Hub's layout or a glyd-v7 one, or a Hub
//! id in the cache) on the description executor (src/dynamic/qwen35.rs, src/dynamic/dense.rs), one sequence, packed in memory as it loads (--layout auto|mma|mma12|i6|i5|i4).

use glyd_engine::args::Args;
use glyd_engine::conv::Gen;
use glyd_engine::ffi::{err, Error, Result};
use glyd_engine::sampler::Params;
use glyd_engine::sched::Reply;
use glyd_engine::tok::Tok;
use glyd_engine::{pack, report};
use glyd_gpu::json::{self, Value};
use std::path::Path;
use std::time::Instant;

/// The command a development build runs by name (`run`, `serve`, `doctor`, `login`, `supports`, `fetch` and `prepare` are main.rs's).
pub fn run(cmd: &str, a: &Args) -> Result<()> {
    match cmd {
        "pack" => cmd_pack(a),
        "ship" => glyd_engine::ship::run(a),
        "hbtoy" => cmd_hbtoy(a),
        "calibrate-experts" => cmd_calibrate_experts(a),
        "tiercheck" => glyd_engine::tiering::check(a),
        "storebench" => glyd_engine::tiering::bench(&a),
        "chat" => std::process::exit(glyd_engine::cli::run(&Args::from(std::env::args().skip(1).chain(["--chat".to_string()])))),
        "generate" => cmd_run(a),
        "check" => cmd_check(a),
        "microbench" => cmd_bench(a),
        "kvshapes" => {
            if !glyd_engine::kvshapes::run()? {
                std::process::exit(1);
            }
            Ok(())
        }
        "kvprefill" => glyd_engine::kvshapes::bench(a),
        "kvdecode" => glyd_engine::kvshapes::decode_bench(a),
        // the planner's predictions against this machine, gated, and a calibration record a case (src/plancheck.rs)
        "plancheck" => std::process::exit(glyd_engine::plancheck::cmd(a)),
        "bandwidth" => {
            let dev = glyd_engine::dev::Dev::new(0)?;
            let gbs = report::bandwidth(&dev, a.num("gib", 4usize)?, a.num("reps", 20usize)?)?;
            println!("{}: {gbs:.1} GB/s streaming read", dev.name);
            Ok(())
        }
        "l2check" => {
            let dev = glyd_engine::dev::Dev::new(0)?;
            let mibs: Vec<usize> = a.flag("mib").unwrap_or("4,8,16,32").split(',').filter_map(|v| v.trim().parse().ok()).collect();
            report::l2check(&dev, &mibs, a.num("reps", 20usize)?, a.num("gap-us", 10u64)?)
        }
        "topkcheck" => {
            if !glyd_engine::kcheck::topk_check(&glyd_engine::dev::Dev::new(0)?)? {
                std::process::exit(1);
            }
            Ok(())
        }
        "sampletime" => cmd_sampletime(a),
        "kcheck" => glyd_engine::kcheck::run(Path::new(a.need(1, "IN_DIR")?), Path::new(a.need(2, "OUT_DIR")?)),
        "kcheck35" => {
            if !glyd_engine::kcheck35::run()? {
                std::process::exit(1);
            }
            Ok(())
        }
        "attncheck" => {
            if !glyd_engine::dynamic::attn::check()? {
                std::process::exit(1);
            }
            Ok(())
        }
        "attndecode" => glyd_engine::dynamic::attn::time_decode(a),
        "moecheck" => {
            if !glyd_engine::kcheck_moe::run(a)? {
                std::process::exit(1);
            }
            Ok(())
        }
        "hbcheck" => {
            if !glyd_engine::kcheck_hb::run(a)? {
                std::process::exit(1);
            }
            Ok(())
        }
        "cpuexperts" => glyd_engine::cpu_bench::run(a),
        "glmfmtcheck" => glyd_engine::cpu_cb::check_formats(Path::new(a.need(1, "FIXTURE_DIR")?)),
        "cb2prefill" => glyd_engine::cb2_prefill::run(a),
        "skinnybench" => glyd_engine::skinny::run(a),
        "prodbench" => glyd_engine::dynamic::prodbench(a),
        "prodcheck" => glyd_engine::dynamic::prodcheck(a),
        "fp4bench" => glyd_engine::cpu_fp4::bench(&a),
        "actcheck" => glyd_engine::cpu_fp4::act_check(&a),
        "rangecheck" => glyd_engine::dynamic::range::rangecheck(a),
        "cb2check" => glyd_engine::cpu_cb::check_pack(Path::new(a.need(1, "PACK_DIR")?), a.num("layer", 3usize)?, a.num("expert", 0usize)?),
        "nhkernels" => glyd_engine::kcheck_nh::kernels(Path::new(a.need(1, "IN_DIR")?), Path::new(a.need(2, "OUT_DIR")?)),
        "nhlayers" => glyd_engine::kcheck_nh::layers(Path::new(a.need(1, "DIR")?), Path::new(a.need(2, "IO_DIR")?)),
        "logitscheck" => glyd_engine::logits::check(a),
        "paritycheck" => glyd_engine::logits::parity(a),
        "levelcheck" => glyd_engine::logits::level_check(a),
        "logitsrecord" => glyd_engine::logits::record(a),
        "logitsbound" => glyd_engine::logits::set_bounds(a),
        "logitsmodel" => glyd_engine::logits::make_model(a),
        "packcheck" => glyd_engine::packcheck::check(a),
        "usercheck" => glyd_engine::logits::users_check(a),
        "twomodels" => glyd_engine::logits::two_models_check(a),
        "splitcheck" => glyd_engine::logits::split_check(a),
        "bufcheck" => glyd_engine::dev::buf_check(),
        "foldcheck" => glyd_engine::fold::check(a.flag("only")),
        "seamcheck" => glyd_engine::runtime::cuda::check(&a),
        "kindcheck" => glyd_engine::dynamic::kindcheck::run(&a),
        "draftcheck" => glyd_engine::dynamic::head::check(&a),
        "roundcheck" => glyd_engine::logits::round_check(a),
        "keepcheck" => glyd_engine::logits::keep_check(a),
        "kv12cal" => glyd_engine::dynamic::kv12::cmd(a),
        "kv12check" => glyd_engine::dynamic::kvpages_check(a),
        "muserandom" => glyd_engine::muse_ref::random(Path::new(a.need(1, "OUT")?)),
        "museref" => glyd_engine::muse_ref::cmd(&a),
        "musecheck" => glyd_engine::muse_ref::check(&a),
        "hubslice" => glyd_engine::hub::slice_cmd(&a),
        "dlrate" => glyd_engine::hub::rate_cmd(&a),
        "dsv4random" => {
            glyd_engine::dsv4_ref::random(Path::new(a.need(1, "OUT")?), if a.has("real") { Some((a.num("layers", 3)?, a.num("experts", 32)?)) } else { None })
        }
        "dsv4prof" => glyd_engine::dsv4_ref::prof(&a),
        "dsv4ref" => glyd_engine::dsv4_ref::cmd(&a),
        "dsv4check" => glyd_engine::dsv4_ref::check(&a),
        "v4qcheck" => glyd_engine::dsv4_ref::qcheck(&a),
        "dsv4prompt" => glyd_engine::dsv4_ref::prompt(&a),
        "dsv4layers" => glyd_engine::dsv4_ref::layers(&a),
        "dsv4pack" => glyd_engine::dsv4_ref::pack_check(&a),
        "f8moecheck" => glyd_engine::dsv4_ref::f8moe_check(&a),
        "fp4stat" => glyd_engine::dsv4_ref::fp4_stat(&a),
        "f8stat" => glyd_engine::dsv4_ref::f8_stat(&a),
        "dsv4gen" => glyd_engine::dsv4_ref::gen(&a),
        "kimirandom" if a.has("real") => {
            glyd_engine::kimi_ref::random_real(Path::new(a.need(1, "OUT")?), a.num("layers", 4usize)?, a.num("experts", 320usize)?)
        }
        "kimirandom" => glyd_engine::kimi_ref::random(Path::new(a.need(1, "OUT")?), a.num("dense", 1usize)?, !a.has("no-mla"), a.num("layers", 8usize)?),
        "kimiref" => glyd_engine::kimi_ref::cmd(&a),
        "kimicheck" => glyd_engine::kimi_ref::check(&a),
        "ceiling" => glyd_engine::ceiling::cmd(&a),
        "dcfill" => glyd_engine::dynamic::tp_cache::fill_bench(&a),
        "dcreplay" => glyd_engine::dynamic::tp_cache::replay_bench(&a),
        "tpcheck" => glyd_engine::dynamic::tp::check(&a),
        "tpbench" => glyd_engine::dynamic::tp::bench(&a),
        "tpgather" => glyd_engine::dynamic::tp::gather_bench(&a),
        "kimilayers" => glyd_engine::kimi_ref::layers(&a),
        "nhrandom" => glyd_engine::kcheck_nh::random(Path::new(a.need(1, "OUT")?), a.num("latent", 0usize)?, a.num("experts", 16usize)?),
        "nhtoy" => {
            // --expert-cache (auto, auto:GIB or GIB) runs the model through the expert cache, how it serves a miss as --offload-mode (one mode), --cpu-threads and --cpu-pin say
            let cache = a.flag("expert-cache").map(|spec| glyd_engine::offload::offload_args(a).map(|o| (spec, o))).transpose()?;
            glyd_engine::kcheck_nh::toy(Path::new(a.need(1, "DIR")?), Path::new(a.need(2, "IO_DIR")?), cache)
        }
        "diskcheck" => glyd_engine::kcheck_nh::disk(Path::new(a.need(1, "DIR")?), a.num("reads", 400usize)?, a.num("depth", 16usize)?),
        "nhoffload" => {
            let slots: Vec<usize> = a
                .flag("slots")
                .unwrap_or("3,5,8")
                .split(',')
                .map(|x| x.trim().parse().map_err(|_| Error(format!("--slots {x}: not a number"))))
                .collect::<Result<_>>()?;
            // --host-memory GIB,GIB,...: budgets of the computer's memory for the experts, each run against the tiny model with every expert in memory
            let memory: Vec<u64> = match a.flag("host-memory") {
                Some(v) => v
                    .split(',')
                    .map(|x| {
                        x.trim()
                            .parse::<f64>()
                            .ok()
                            .filter(|g| g.is_finite() && *g > 0.0)
                            .map(|g| (g * (1u64 << 30) as f64) as u64)
                            .ok_or_else(|| Error(format!("--host-memory {x}: not a number of GiB")))
                    })
                    .collect::<Result<_>>()?,
                None => Vec::new(),
            };
            // --offload-mode copy|cpu|split|auto (one), --cpu-threads and --cpu-pin: how the cache serves a miss (the default, copy, is held to the bit; the others to `logits_within`)
            let mut host = glyd_engine::offload::offload_args(a)?;
            if !a.has("offload-mode") {
                host.mode = glyd_engine::offload::Mode::Copy;
            }
            if !glyd_engine::kcheck_nh::offload(Path::new(a.need(1, "DIR")?), Path::new(a.need(2, "IO_DIR")?), &slots, Some(host), &memory)? {
                std::process::exit(1);
            }
            Ok(())
        }
        "nhdecode" => {
            let layers = a.flag("layers").ok_or_else(|| Error("nhdecode needs --layers L,L,...".into()))?;
            let layers: Vec<usize> =
                layers.split(',').map(|x| x.trim().parse().map_err(|_| Error(format!("--layers {x}: not a number")))).collect::<Result<_>>()?;
            glyd_engine::kcheck_nh::decode(Path::new(a.need(1, "DIR")?), Path::new(a.need(2, "OUT_DIR")?), &layers, a.num("tokens", 333usize)?)
        }
        "glmrandom" => glyd_engine::glm::random_seeded(
            Path::new(a.need(1, "OUT")?),
            a.has("fp8"),
            if a.has("real") { Some((a.num("layers", 4)?, a.num("experts", 32)?)) } else { None },
            a.num("seed", glyd_engine::glm::RANDOM_SEED)?,
        ),
        "qxlayers" => {
            if !glyd_engine::kcheck_qwen4x::layers(Path::new(a.need(1, "MODEL")?), Path::new(a.need(2, "REF")?), a)? {
                std::process::exit(1);
            }
            Ok(())
        }
        "qxcb" => {
            if !glyd_engine::kcheck_qwen4x::cb(Path::new(a.need(1, "MODEL")?), Path::new(a.need(2, "PACK")?), Path::new(a.need(3, "REF")?), a)? {
                std::process::exit(1);
            }
            Ok(())
        }
        "g4check" => glyd_engine::gemma4::check(a),
        "i4xcheck" => glyd_engine::gemma4::i4x_check(a),
        "g4spec" => glyd_engine::gemma4::spec_check(a),
        "g4tok" => glyd_engine::gemma4::tok_check(a),
        "g4score" => glyd_engine::gemma4::score(a),
        "g4prefill" => glyd_engine::gemma4::prefill_prof(a),
        "qxbench" => glyd_engine::kcheck_qwen4x::bench(a),
        "qxprod" => glyd_engine::kcheck_qwen4x::bench_products(a),
        "qxtq" => glyd_engine::qx_trellis::cmd(a),
        "qxvision" => glyd_engine::vision::check(a),
        "qxhost" => glyd_engine::kcheck_qwen4x::host_bench(Path::new(a.need(1, "PACK")?), a),
        "qxtoy" => {
            if !glyd_engine::kcheck_qwen4x::toy(Path::new(a.need(1, "MODEL")?), Path::new(a.need(2, "REF")?), a)? {
                std::process::exit(1);
            }
            Ok(())
        }
        _ => err("usage: glyd-engine run|serve|doctor|login|supports|fetch|prepare, or a development command (see the top of src/devcmd.rs)"),
    }
}

/// DIR: a directory (a packed one, or a Hub-format one the engine packs in memory), or a Hub repo id already in the local cache.
fn model_dir(a: &Args) -> Result<std::path::PathBuf> {
    let m = a.need(1, "DIR")?;
    if Path::new(m).is_dir() {
        return Ok(Path::new(m).to_path_buf());
    }
    glyd_gpu::save::Source::find(m).map(|s| s.dir).map_err(|e| Error(e.to_string()))
}

fn cmd_run(a: &Args) -> Result<()> {
    if let Some(dir) = nemotron_dir(a) {
        return cmd_run_nh(a, &dir);
    }
    cmd_run35(a, &exec_dir(a)?)
}

/// PROMPTS.json: {"prompts": [[token ids], ...]}; OUT.json: {"tokens": [[...]], "logprobs": [[...]]}.
fn cmd_check(a: &Args) -> Result<()> {
    if let Some(dir) = nemotron_dir(a) {
        return cmd_check_nh(a, &dir);
    }
    cmd_check35(a, &exec_dir(a)?)
}

fn cmd_bench(a: &Args) -> Result<()> {
    nemotron_h_refused(a)?;
    cmd_bench35(a, &exec_dir(a)?)
}

/// `--layout auto|mma|mma12`: the GPU's by `best_layout` (the default), the tiered one (10.8 bits a weight) or the 12-bit one.
fn layout_flag(a: &Args, cfg: Option<&glyd_engine::weights::Config>) -> Result<Option<glyd_engine::weights::Layout>> {
    let asked = a.layout()?;
    match (asked, cfg) {
        (None, Some(c)) => {
            let dev = glyd_engine::dev::Dev::new(0)?;
            let (l, why) = glyd_engine::weights::choose_layout(None, dev.cc_pair(), dev.total_mem as u64, c.matrix_bytes(), 0);
            eprintln!("layout {}: {why}", l.name());
            Ok(Some(l))
        }
        (asked, _) => Ok(asked),
    }
}

fn cmd_pack(a: &Args) -> Result<()> {
    use glyd_engine::weights::Layout;
    let model = a.need(1, "MODEL")?;
    if a.flag("layout") == Some("hummingbird") {
        // (a model the description builder maps, from the corpus pass's counts: src/hb_pack.rs; Nemotron 3 Super from an export: src/hbpack.rs)
        if let Some(usage) = a.flag("usage") {
            return glyd_engine::hb_pack::run(model, Path::new(a.need(2, "OUT")?), a.num("threads", 16usize)?, Path::new(usage));
        }
        let codes = a.flag("codes").ok_or_else(|| {
            Error(
                "--layout hummingbird needs --usage FILE (the counts glyd-engine calibrate-experts writes), or --codes DIR for Nemotron 3 Super's export"
                    .into(),
            )
        })?;
        return glyd_engine::hbpack::run(model, Path::new(a.need(2, "OUT")?), a.num("threads", 16usize)?, Path::new(codes));
    }
    let src = glyd_gpu::save::Source::find(model).map_err(|e| Error(e.to_string()))?;
    // (a lossy level's layout is the one asked: the model, a Qwen3.5, a Nemotron 3 Super or a GLM-5.3-Flash, has no Qwen3 config to choose a layout by; GLM-5.3-Flash is packed at the lossy levels alone)
    if glyd_engine::glm::is_glm(&std::fs::read_to_string(src.dir.join("config.json")).unwrap_or_default())
        && !matches!(a.layout()?, Some(Layout::Int6 | Layout::Int5 | Layout::Int4 | Layout::Int3))
    {
        return err("GLM-5.3-Flash is packed at a lossy level: --layout i6 or i5");
    }
    // (a Gemma 4 model has no Qwen3 config either: its layout is the one asked, penguin's where none is)
    let gemma = glyd_engine::gemma4::ConfigG4::load(&src.dir).is_ok();
    let layout = match a.layout()?.filter(|l| gemma || matches!(l, Layout::Int6 | Layout::Int5 | Layout::Int4 | Layout::Int4H | Layout::Int3)) {
        Some(l) => l,
        None if gemma => Layout::Tiered,
        None => layout_flag(a, Some(&glyd_engine::weights::Config::load(&src.dir)?))?.unwrap_or(Layout::Tiered),
    };
    pack::run(model, Path::new(a.need(2, "OUT")?), a.num("threads", 16usize)?, layout)
}

/// `hbtoy MODEL OUT`: MODEL a tiny Nemotron-H model (`nemotron_ref.py moetoy`'s toy/ directory, bf16), a random export for its experts in OUT/export and the model packed at the Hummingbird level from it into OUT/packed.
/// `calibrate-experts MODEL OUT [--tokens N] [--devices 0,1] [--split CUTS] [--context N]`: hummingbird's corpus pass (src/hb_usage.rs).
fn cmd_calibrate_experts(a: &Args) -> Result<()> {
    let list = |k: &str| -> Result<Vec<usize>> {
        a.flag(k).map_or(Ok(Vec::new()), |v| {
            v.split(',').map(|x| x.trim().parse::<usize>().map_err(|_| Error(format!("--{k} {v}: numbers separated by commas")))).collect()
        })
    };
    let devices: Vec<i32> = list("devices")?.into_iter().map(|d| d as i32).collect();
    let cuts = list("split")?;
    let devices = if devices.is_empty() { vec![0] } else { devices };
    if devices.len() != cuts.len() + 1 {
        return err(format!("{} GPUs for {} cuts: one more GPU than cuts", devices.len(), cuts.len()));
    }
    glyd_engine::hb_usage::run_with(
        a.need(1, "MODEL")?,
        Path::new(a.need(2, "OUT")?),
        a.num("tokens", glyd_engine::hb_usage::TOKENS)?,
        &devices,
        &cuts,
        a.num("context", 4096usize)?,
        a.has("synthetic"),
    )
}

fn cmd_hbtoy(a: &Args) -> Result<()> {
    let model = a.need(1, "MODEL")?;
    let out = Path::new(a.need(2, "OUT")?);
    let c = glyd_engine::nemotron::ConfigNemotronH::load(&model_dir(a)?)?;
    glyd_engine::hbpack::toy_export(&c, &out.join("export"), a.num("seed", 1u64)?, a.num("share", 40usize)?)?;
    glyd_engine::hbpack::run(model, &out.join("packed"), a.num("threads", 4usize)?, &out.join("export"))
}

/// The model of the command line where it is a Nemotron-H one (its config.json says so: a directory `pack` made or a Hub-format one, or a Hub id in the cache).
fn nemotron_dir(a: &Args) -> Option<std::path::PathBuf> {
    let d = model_dir(a).ok()?;
    glyd_engine::nemotron::is_nemotron_h(&std::fs::read_to_string(d.join("config.json")).ok()?).then_some(d)
}

/// A Nemotron-H model is run by `generate` and not yet by `check` and `bench`: its layers are checked by `nhkernels`, `nhlayers` and `nhtoy` (on a few of the real model's layers, on a random tiny model, or on the
/// whole of a glyd-v7 directory).
fn nemotron_h_refused(a: &Args) -> Result<()> {
    match nemotron_dir(a) {
        Some(_) => err("a Nemotron-H model: generate runs it, this command does not yet (nhtoy takes the logits of a glyd-v7 directory)"),
        None => Ok(()),
    }
}

/// How the cache serves a miss, as the command line says: `--offload-mode` is a mode or a sweep of variants, comma-separated (`copy,cpu,cpu:16,split:8:pin`; each a mode, then its threads and `pin`, on the settings of
/// `--cpu-threads`, `--cpu-pin`, `--offload-window` and `--offload-promote`); the variants, in the order given (one: the mode, or `auto`).
fn offload_variants(a: &Args) -> Result<Vec<glyd_engine::offload::Offload>> {
    use glyd_engine::offload::{offload_args, parse_variant, Mode};
    let mut plain = a.clone();
    plain.flags.retain(|(k, _)| k != "offload-mode");
    let base = offload_args(&plain)?;
    let list = a.flag("offload-mode").unwrap_or(Mode::Auto.name());
    list.split(',')
        .map(|v| {
            parse_variant(v.trim(), &base).ok_or_else(|| Error(format!("--offload-mode {v}: copy, cpu, split or auto, then :THREADS and :pin (cpu:16:pin)")))
        })
        .collect()
}

/// The engine of a Nemotron-H model: a glyd-v7 directory at its own level, else a Hub-format one packed as it loads at the level of `--layout i6`, `i5` or `i4`. `--expert-cache GIB`: every expert in the computer's memory and
/// the GPU's `GIB` GiB for a cache of them (`auto`: what `run` would give this GPU, `auto:GIB`: what a GPU with GIB GiB free would get; src/offload.rs; the store is returned for its counts) and the variants of
/// `--offload-mode` (the first is set; a computer that cannot hold the host's half besides the experts has its `auto` copy, and is refused `cpu` and `split`).
fn open_nh(
    a: &Args,
    dir: &Path,
) -> Result<(glyd_engine::dynamic::Dyn, f64, Option<std::rc::Rc<glyd_engine::offload::ExpertStore>>, Vec<glyd_engine::offload::Offload>)> {
    use glyd_engine::offload::Mode;
    use glyd_engine::weights::Layout;
    let t0 = Instant::now();
    let level = match glyd_engine::weights::v7_manifest(dir)? {
        Some((l, _)) => l,
        None => a
            .layout()?
            .filter(|l| matches!(l, Layout::Int6 | Layout::Int5 | Layout::Int4))
            .ok_or_else(|| Error("a Nemotron-H model runs in six-, five- or four-bit codes: --layout i6, i5 or i4, or a directory `pack` made".into()))?,
    };
    let o = glyd_engine::dynamic::Opts {
        ctx: a.num("ctx", 4096usize)?,
        chunk: a.num("chunk", 2048usize)?,
        graph: !a.has("no-graph"),
        device: 0,
        layout: Some(level),
    };
    let cfg = glyd_engine::nemotron::ConfigNemotronH::load(dir)?;
    let mut variants = offload_variants(a)?;
    let mut places = glyd_engine::kcheck_nh::cache_for(a.flag("expert-cache"), dir, level, &cfg)?;
    if let Some(p) = places.as_mut() {
        glyd_engine::kcheck_nh::host_budget(a, p)?;
    }
    glyd_engine::kcheck_nh::host_setup(places.as_ref(), &mut variants)?;
    let store = glyd_engine::offload::ExpertStore::new();
    // the experts the computer's memory does not hold stay in the packed directory (--host-memory)
    if let Some(p) = places.filter(|p| p.ram < p.host_bytes()) {
        store.set_host_memory(Some(glyd_engine::offload::HostMemory { bytes: p.ram, total: p.host_bytes() }));
    }
    // the layers are loaded with a host half if any variant uses it; the first variant is the one set when the engine is ready
    let loading = variants.iter().copied().find(|v| v.mode != Mode::Copy).unwrap_or(variants[0]);
    store.set_offload(loading);
    let (plain, cached) = (glyd_engine::moe_nh::factory(level), places.map(|p| glyd_engine::moe_nh::factory_cached(level, p.slots, &store)));
    let make: glyd_engine::kern_nh::MoeFactory<'_> = match &cached {
        Some(f) => f,
        None => &plain,
    };
    let e = glyd_engine::dynamic::nemotron::load_with(dir, &o, make, glyd_engine::dynamic::Stage::whole(cfg.layers))?;
    store.set_offload(variants[0]);
    Ok((e, t0.elapsed().as_secs_f64(), places.map(|_| store), variants))
}

/// `sampletime DIR PROMPT [--new N] [--ctx N]` on a Qwen3.5 or Nemotron-H model: `new` tokens after the prompt (the end token ignored) by greedy steps, by sampled steps with the model's own sampling (generation_config.json) drawn from the 64 largest
/// logits the GPU makes, and by the same sampled steps with the row read to the host: the decode speed of each, and that the two sampled replies are the same tokens (the same seed). Exit status 1 where they are not.
fn cmd_sampletime(a: &Args) -> Result<()> {
    let (prompt, new) = (a.need(2, "PROMPT")?, a.num("new", 256usize)?);
    if let Some(dir) = nemotron_dir(a) {
        let (mut e, load, _store, _) = open_nh(a, &dir)?;
        eprintln!("loaded in {load:.1} s: {}", e.describe());
        return time_replies(&mut e, |e, host| e.top.off = host, &Tok::load(&dir)?.encode(prompt)?, &glyd_engine::sampler::generation_defaults(&dir), new);
    }
    let dir = exec_dir(a)?;
    let (mut e, load) = open35(a, &dir)?;
    eprintln!("loaded in {load:.1} s: {}", e.describe());
    time_replies(&mut e, |e, host| e.top.off = host, &Tok::load(&dir)?.encode(prompt)?, &glyd_engine::sampler::generation_defaults(&dir), new)
}

/// `new` tokens after `ids` by `Gen::reply` with `params` (the end token ignored): the tokens and the decode speed from the first token on, tokens a second.
fn timed_reply(e: &mut dyn Gen, ids: &[u32], params: Params, new: usize) -> Result<(Vec<u32>, f64)> {
    let mut first = None;
    let (out, _) = e.reply(ids, &Reply { params, max_new: new, ignore_eos: true, ..Reply::default() }, &mut |_| {
        first.get_or_insert_with(Instant::now);
        true
    })?;
    let secs = first.map_or(f64::NAN, |t: Instant| t.elapsed().as_secs_f64());
    let speed = (out.len().max(1) - 1) as f64 / secs;
    Ok((out, speed))
}

/// The three replies of `sampletime` on `e` (`host_row` makes it read the row for its draws, or not).
fn time_replies<E: Gen>(e: &mut E, host_row: fn(&mut E, bool), ids: &[u32], defaults: &Params, new: usize) -> Result<()> {
    use std::sync::atomic::Ordering::Relaxed;
    let (greedy, sampled) = (Params { temperature: 0.0, ..defaults.clone() }, Params { seed: 1, ..defaults.clone() });
    host_row(e, false);
    timed_reply(e, ids, greedy.clone(), 8)?; // (what is set up at the first step out of the timing)
    let (_, greedy_speed) = timed_reply(e, ids, greedy, new)?;
    let draws = || glyd_engine::sampler::DRAWS.iter().map(|d| d.load(Relaxed)).collect::<Vec<_>>();
    let before = draws();
    let (top_tokens, top_speed) = timed_reply(e, ids, sampled.clone(), new)?;
    let after = draws();
    host_row(e, true);
    let (host_tokens, host_speed) = timed_reply(e, ids, sampled, new)?;
    host_row(e, false);
    let ms = |tps: f64| 1e3 / tps;
    println!(
        "decode of {new} tokens after a prompt of {} (sampled: temperature {}, top-k {}, top-p {}):",
        ids.len(),
        defaults.temperature,
        defaults.top_k,
        defaults.top_p
    );
    println!("  greedy steps                              {greedy_speed:7.1} tokens/s  ({:.2} ms a token)", ms(greedy_speed));
    println!(
        "  sampled, from the GPU's 64 largest logits {top_speed:7.1} tokens/s  ({:.2} ms a token; {} of {} draws from them, {} read the row)",
        ms(top_speed),
        after[1] - before[1],
        top_tokens.len(),
        after[0] - before[0]
    );
    println!("  sampled, from the row on the host         {host_speed:7.1} tokens/s  ({:.2} ms a token)", ms(host_speed));
    let apart = top_tokens
        .iter()
        .zip(&host_tokens)
        .position(|(x, y)| x != y)
        .or((top_tokens.len() != host_tokens.len()).then_some(top_tokens.len().min(host_tokens.len())));
    match apart {
        None => println!("  the two sampled replies: the same {} tokens", top_tokens.len()),
        Some(i) => {
            println!("  the two sampled replies: DIFFERENT, from token {i} on");
            std::process::exit(1);
        }
    }
    Ok(())
}

/// A variant of the offload settings in a word, as `--offload-mode` names it (`copy`, `cpu:20`, `split:20:pin`).
fn variant_label(v: &glyd_engine::offload::Offload) -> String {
    use glyd_engine::offload::Mode;
    if v.mode == Mode::Copy {
        return v.mode.name().to_string();
    }
    format!("{}:{}{}", v.mode.name(), v.threads, if v.pin { ":pin" } else { "" })
}

/// A completion by greedy steps, timed as the prefill and the decode steps apart; with an expert cache, its counts for each (the decode's are the ones that compare with the record's replay). `--offload-mode` with
/// several variants runs the completion once for each on the engine as it is loaded, the cache cold at the start of each: the lines of a variant begin with its name in brackets, and every variant after the first says
/// where its tokens first differ from the first's (the host's sums are not the GPU's to the bit).
fn cmd_run_nh(a: &Args, dir: &Path) -> Result<()> {
    let prompt = a.need(2, "PROMPT")?;
    let (mut e, load, store, variants) = open_nh(a, dir)?;
    let tok = Tok::load(dir)?;
    eprintln!("loaded in {load:.1} s: {}", e.describe());
    // (the process's most resident memory so far: the pinned experts and the engine's own, not the page cache's)
    let peak = || glyd_engine::disk::peak_rss().map_or("not said".to_string(), |b| format!("{:.1} GiB", b as f64 / (1u64 << 30) as f64));
    eprintln!("peak resident memory after the load: {}", peak());
    let ids = tok.encode(prompt)?;
    let (new, ignore_eos) = (a.num("new", 64usize)?, a.has("ignore-eos"));
    let mut first: Option<Vec<u32>> = None;
    for (vi, v) in variants.iter().enumerate().filter(|(vi, _)| store.is_some() || *vi == 0) {
        let label = if store.is_some() && variants.len() > 1 { format!("[{}] ", variant_label(v)) } else { String::new() };
        if let Some(st) = &store {
            if vi > 0 {
                st.cold(&e.dev)?;
            }
            st.set_offload(*v);
        }
        e.reset()?;
        let t0 = Instant::now();
        let mut s = e.prefill(&ids)?;
        let prefill = t0.elapsed().as_secs_f64();
        if let Some(st) = &store {
            eprintln!("{label}prefill: {}", st.report()?);
            st.reset_stats()?;
        }
        let mut out = vec![s];
        let t1 = Instant::now();
        // (the second half of the steps timed apart too: a cold cache fills in the first)
        let (mid, mut half) = (1 + (new.max(2) - 1) / 2, None);
        while out.len() < new && (ignore_eos || !e.desc().eos.contains(&(s.token as i64))) {
            s = e.step(s.token)?;
            out.push(s);
            if out.len() == mid {
                half = Some(Instant::now());
            }
        }
        let decode = t1.elapsed().as_secs_f64();
        let later = half.map(|t| (out.len() - mid) as f64 / t.elapsed().as_secs_f64().max(1e-9));
        let toks: Vec<u32> = out.iter().map(|s| s.token).collect();
        if vi == 0 {
            println!("{prompt}{}", tok.decode(&toks)?);
        }
        eprintln!(
            "{label}{} prompt tokens in {prefill:.2} s ({:.1} a second); {} generated, {} by decode steps in {decode:.2} s ({:.2} tokens a second)",
            ids.len(),
            ids.len() as f64 / prefill,
            out.len(),
            out.len() - 1,
            (out.len() - 1) as f64 / decode.max(1e-9)
        );
        if let Some(st) = &store {
            eprintln!("{label}decode: {}", st.report()?);
            let (c, n) = (st.stats()?, (out.len() - 1).max(1) as f64);
            if let Some(l) = later {
                eprintln!("{label}decode, the second half of the steps: {l:.2} tokens a second");
            }
            eprintln!(
                "{label}decode a token: {:.1} experts read, {:.1} found on the GPU ({:.1}%), {:.1} missed: {:.1} computed by the host in {:.2} ms, {:.1} copied in behind the GPU's work in {:.2} ms of copying and {:.1} more for later, the host waiting {:.2} ms for the GPU",
                c.reads as f64 / n,
                c.hits as f64 / n,
                100.0 * c.hit_rate(),
                c.misses as f64 / n,
                c.cpu as f64 / n,
                c.cpu_ms / n,
                c.copied as f64 / n,
                c.copy_ms / n,
                c.promoted as f64 / n,
                c.wait_ms / n
            );
            if c.disk_reads > 0 || c.ram_hits > 0 {
                eprintln!(
                    "{label}decode a token, the computer's memory: {:.1} of the experts wanted found in it, {:.1} read from the disk ({:.1} MB in {:.2} ms of the host's waiting, {:.2} GB/s)",
                    c.ram_hits as f64 / n,
                    c.disk_reads as f64 / n,
                    c.disk_bytes as f64 / 1e6 / n,
                    c.disk_ms / n,
                    c.disk_gbs()
                );
            }
            eprintln!("{label}peak resident memory so far: {}", peak());
        }
        if let Some(st) = &store {
            // (one line of numbers a variant, for the pod job's table)
            let (c, n) = (st.stats()?, (out.len() - 1).max(1) as f64);
            eprintln!(
                "{label}result: mode={} threads={} pin={} decode_tps={:.2} second_half_tps={:.2} prefill_tps={:.1} found_pct={:.1} reads={:.1} missed={:.1} host={:.1} copied={:.1} promoted={:.1} host_ms={:.2} copy_ms={:.2} wait_ms={:.2} disk_reads={:.1} disk_mb={:.1} disk_ms={:.2}",
                v.mode.name(),
                v.threads,
                v.pin as u8,
                (out.len() - 1) as f64 / decode.max(1e-9),
                later.unwrap_or(f64::NAN),
                ids.len() as f64 / prefill,
                100.0 * c.hit_rate(),
                c.reads as f64 / n,
                c.misses as f64 / n,
                c.cpu as f64 / n,
                c.copied as f64 / n,
                c.promoted as f64 / n,
                c.cpu_ms / n,
                c.copy_ms / n,
                c.wait_ms / n,
                c.disk_reads as f64 / n,
                c.disk_bytes as f64 / 1e6 / n,
                c.disk_ms / n
            );
        }
        match &first {
            None => first = Some(toks),
            Some(f) => {
                let at = f.iter().zip(&toks).position(|(x, y)| x != y);
                eprintln!(
                    "{label}tokens against the first variant's: {}",
                    at.map_or_else(
                        || format!("the same {}", f.len().min(toks.len())),
                        |i| format!("the first differs at token {i} of {}", f.len().min(toks.len()))
                    )
                );
            }
        }
    }
    eprintln!("peak resident memory: {}", peak());
    Ok(())
}

/// The model of the command line where it is a Qwen3.5 one (a directory, or a Hub id in the cache, whose config.json parses as Qwen3.5).
/// The model of the command line where the executor runs it one sequence at a time here: a Qwen3.5-family model, or a plain dense decoder.
fn exec_dir(a: &Args) -> Result<std::path::PathBuf> {
    let d = model_dir(a)?;
    let text = std::fs::read_to_string(d.join("config.json")).map_err(|e| Error(format!("{}: {e}", d.display())))?;
    if glyd_engine::qwen35::Config35::load(&d).is_ok() || glyd_engine::dynamic::dense::is_dense(&text) {
        return Ok(d);
    }
    err(format!("{}: a Qwen3.5-family model, a plain dense decoder or a Nemotron-H model is run here", d.display()))
}

/// A Qwen3.5-family model, or a plain dense decoder, on the description executor (their hand-written engines are gone: R3b, #137; R3c, #139).
fn open35(a: &Args, dir: &Path) -> Result<(glyd_engine::dynamic::Dyn, f64)> {
    let t0 = Instant::now();
    let o = glyd_engine::dynamic::Opts {
        ctx: a.num("ctx", 4096usize)?,
        chunk: a.num("chunk", 2048usize)?,
        graph: !a.has("no-graph"),
        device: 0,
        layout: a.layout()?,
    };
    let dense = std::fs::read_to_string(dir.join("config.json")).is_ok_and(|t| glyd_engine::dynamic::dense::is_dense(&t));
    let e = if dense { glyd_engine::dynamic::dense::load(dir, &o)? } else { glyd_engine::dynamic::qwen35::load(dir, &o)? };
    Ok((e, t0.elapsed().as_secs_f64()))
}

/// Up to `new` greedy tokens after `ids` (stopping at an end token unless `ignore_eos`).
fn greedy(e: &mut glyd_engine::dynamic::Dyn, ids: &[u32], new: usize, ignore_eos: bool) -> Result<Vec<u32>> {
    let eos = e.desc().eos.clone();
    e.reset()?;
    let mut s = e.prefill(ids)?;
    let mut out = Vec::new();
    loop {
        out.push(s.token);
        if out.len() >= new || (!ignore_eos && eos.contains(&(s.token as i64))) {
            return Ok(out);
        }
        s = e.step(s.token)?;
    }
}

fn cmd_run35(a: &Args, dir: &Path) -> Result<()> {
    let prompt = a.need(2, "PROMPT")?;
    let (mut e, load) = open35(a, dir)?;
    let tok = Tok::load(dir)?;
    eprintln!("loaded in {load:.1} s: {}", e.describe());
    let ids = tok.encode(prompt)?;
    let t0 = Instant::now();
    let toks = greedy(&mut e, &ids, a.num("new", 64usize)?, a.has("ignore-eos"))?;
    println!("{prompt}{}", tok.decode(&toks)?);
    eprintln!("{} prompt tokens, {} generated in {:.2} s", ids.len(), toks.len(), t0.elapsed().as_secs_f64());
    Ok(())
}

/// A Nemotron-H model on the executor, on one GPU or split across several.
enum NhRun {
    New(Box<glyd_engine::dynamic::Dyn>),
    Split(Box<glyd_engine::dynamic::split::Split>),
}

impl NhRun {
    fn reset(&mut self) -> Result<()> {
        match self {
            NhRun::New(e) => e.reset(),
            NhRun::Split(e) => e.reset(),
        }
    }

    fn prefill(&mut self, t: &[u32]) -> Result<glyd_engine::sched::Sample> {
        match self {
            NhRun::New(e) => e.prefill(t),
            NhRun::Split(e) => e.prefill(t),
        }
    }

    fn step(&mut self, t: u32) -> Result<glyd_engine::sched::Sample> {
        match self {
            NhRun::New(e) => e.step(t),
            NhRun::Split(e) => e.step(t),
        }
    }
}

/// `check MODEL PROMPTS.json OUT.json --layout LEVEL [--split CUTS --devices GPUS]` on a Nemotron-H model: the greedy tokens and each one's logit (OUT.json {"tokens", "logits", "long_..."}) on the
/// executor, its experts at LEVEL (penguin or a lossy level) all on the GPU (the hand-written engine this once compared with is retired: its parity record is R7's fixtures and `paritycheck`'s).
fn cmd_check_nh(a: &Args, dir: &Path) -> Result<()> {
    let pf = a.need(2, "PROMPTS.json")?;
    let of = a.need(3, "OUT.json")?;
    let v = json::parse(&std::fs::read_to_string(pf).map_err(|e| Error(format!("{pf}: {e}")))?).map_err(|e| Error(format!("{pf}: {e}")))?;
    let new = a.num("new", 64usize)?;
    let level = a.layout()?.ok_or_else(|| Error("--layout LEVEL: the experts' level (penguin, kestrel, swift, wren)".into()))?;
    let (ctx, chunk) = (a.num("ctx", 4096usize)?, a.num("chunk", 2048usize)?);
    let t0 = Instant::now();
    let mut e = if let Some(cuts) = a.flag("split") {
        // --split 26[,39] [--devices 0,1]: the dynamic engine's stages on those GPUs (all on GPU 0 where none is given)
        let o = glyd_engine::dynamic::Opts { ctx, chunk, graph: !a.has("no-graph"), device: 0, layout: Some(level) };
        let num = |x: &str| x.trim().parse::<usize>().map_err(|_| Error(format!("--split/--devices {x}: not a number")));
        let cuts: Vec<usize> = cuts.split(',').map(num).collect::<Result<_>>()?;
        let devices: Vec<i32> = match a.flag("devices") {
            Some(d) => d.split(',').map(|x| num(x).map(|v| v as i32)).collect::<Result<_>>()?,
            None => vec![0; cuts.len() + 1],
        };
        let c = glyd_engine::nemotron::ConfigNemotronH::load(dir)?;
        let d = dir.to_path_buf();
        let make = std::sync::Arc::new(move |st: glyd_engine::dynamic::Stage, o: &glyd_engine::dynamic::Opts| {
            glyd_engine::dynamic::nemotron::load_stage(&d, o, level, st)
        });
        NhRun::Split(Box::new(glyd_engine::dynamic::split::Split::with(c.layers, c.eos.clone(), &cuts, &devices, &o, make)?))
    } else {
        let o = glyd_engine::dynamic::Opts { ctx, chunk, graph: !a.has("no-graph"), device: 0, layout: Some(level) };
        NhRun::New(Box::new(glyd_engine::dynamic::nemotron::load(dir, &o, level)?))
    };
    let load = t0.elapsed().as_secs_f64();
    eprintln!(
        "loaded in {load:.1} s: {}",
        match &e {
            NhRun::New(x) => x.describe(),
            NhRun::Split(x) => x.describe(),
        }
    );
    let eos = glyd_engine::nemotron::ConfigNemotronH::load(dir)?.eos;
    let (mut steps, mut step_s) = (0usize, 0.0f64);
    let mut body = String::from("{");
    for (key, out_key) in [("prompts", ""), ("long_prompts", "long_")] {
        let Some(list) = v.get(key).and_then(Value::as_array) else { continue };
        let prompts: Vec<Vec<u32>> = list
            .iter()
            .map(|p| p.as_u64s().map(|v| v.into_iter().map(|x| x as u32).collect()).ok_or_else(|| Error(format!("{pf}: a prompt is not a list of token ids"))))
            .collect::<Result<_>>()?;
        let (mut tokens, mut logits) = (Vec::new(), Vec::new());
        for p in &prompts {
            e.reset()?;
            let mut s = e.prefill(p)?;
            let (mut ts, mut ls) = (Vec::new(), Vec::new());
            loop {
                ts.push(s.token.to_string());
                ls.push(format!("{:.6}", s.logit));
                if ts.len() >= new || (!a.has("ignore-eos") && eos.contains(&(s.token as i64))) {
                    break;
                }
                let t = Instant::now();
                s = e.step(s.token)?;
                step_s += t.elapsed().as_secs_f64();
                steps += 1;
            }
            tokens.push(ts.join(","));
            logits.push(ls.join(","));
        }
        body.push_str(&format!(
            "\"{out_key}tokens\":[{}],\"{out_key}logits\":[{}],",
            tokens.iter().map(|t| format!("[{t}]")).collect::<Vec<_>>().join(","),
            logits.iter().map(|t| format!("[{t}]")).collect::<Vec<_>>().join(",")
        ));
        eprintln!("{} {key} of up to {} tokens, {} new tokens each", prompts.len(), prompts.iter().map(Vec::len).max().unwrap_or(0), new);
    }
    body.push_str(&format!("\"load_s\":{load:.2}}}\n"));
    std::fs::write(of, body).map_err(|e| Error(format!("{of}: {e}")))?;
    eprintln!("{of}; a decode step {:.3} ms over {steps}", 1e3 * step_s / steps.max(1) as f64);
    Ok(())
}

/// The dynamic engine on one GPU or split across several, as check35 --dynamic runs it.
trait Q35Run {
    fn reset(&mut self) -> Result<()>;
    fn prefill(&mut self, t: &[u32]) -> Result<glyd_engine::sched::Sample>;
    fn step(&mut self, t: u32) -> Result<glyd_engine::sched::Sample>;
    fn logprob(&self, logit: f32) -> Result<f32>;
    fn describe(&self) -> String;
    fn eos(&self) -> Vec<i64>;
    fn bytes(&self) -> usize;
}

impl Q35Run for glyd_engine::dynamic::Dyn {
    fn reset(&mut self) -> Result<()> {
        glyd_engine::dynamic::Dyn::reset(self)
    }
    fn prefill(&mut self, t: &[u32]) -> Result<glyd_engine::sched::Sample> {
        glyd_engine::dynamic::Dyn::prefill(self, t)
    }
    fn step(&mut self, t: u32) -> Result<glyd_engine::sched::Sample> {
        glyd_engine::dynamic::Dyn::step(self, t)
    }
    fn logprob(&self, logit: f32) -> Result<f32> {
        glyd_engine::dynamic::Dyn::logprob(self, logit)
    }
    fn describe(&self) -> String {
        glyd_engine::dynamic::Dyn::describe(self)
    }
    fn eos(&self) -> Vec<i64> {
        self.desc().eos.clone()
    }
    fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Q35Run for glyd_engine::dynamic::split::Split {
    fn reset(&mut self) -> Result<()> {
        glyd_engine::dynamic::split::Split::reset(self)
    }
    fn prefill(&mut self, t: &[u32]) -> Result<glyd_engine::sched::Sample> {
        glyd_engine::dynamic::split::Split::prefill(self, t)
    }
    fn step(&mut self, t: u32) -> Result<glyd_engine::sched::Sample> {
        glyd_engine::dynamic::split::Split::step(self, t)
    }
    fn logprob(&self, logit: f32) -> Result<f32> {
        glyd_engine::dynamic::split::Split::logprob(self, logit)
    }
    fn describe(&self) -> String {
        glyd_engine::dynamic::split::Split::describe(self)
    }
    fn eos(&self) -> Vec<i64> {
        glyd_engine::dynamic::split::Split::eos(self).to_vec()
    }
    fn bytes(&self) -> usize {
        0
    }
}

/// `check35 --dynamic`: the same OUT.json from the dynamic engine (to compare with the old engine's to the digit), and its decode step's time.
fn cmd_check35_dyn(a: &Args, dir: &Path) -> Result<()> {
    let pf = a.need(2, "PROMPTS.json")?;
    let of = a.need(3, "OUT.json")?;
    let v = json::parse(&std::fs::read_to_string(pf).map_err(|e| Error(format!("{pf}: {e}")))?).map_err(|e| Error(format!("{pf}: {e}")))?;
    let new = a.num("new", 64usize)?;
    let t0 = Instant::now();
    let o = glyd_engine::dynamic::Opts {
        ctx: a.num("ctx", 4096usize)?,
        chunk: a.num("chunk", 2048usize)?,
        graph: !a.has("no-graph"),
        device: 0,
        layout: a.layout()?,
    };
    // --split 32[,48] [--devices 0,1]: the stages on those GPUs (all on GPU 0 where none is given)
    let mut e: Box<dyn Q35Run> = match a.flag("split") {
        Some(cuts) => {
            let num = |x: &str| x.trim().parse::<usize>().map_err(|_| Error(format!("--split/--devices {x}: not a number")));
            let cuts: Vec<usize> = cuts.split(',').map(num).collect::<Result<_>>()?;
            let devices: Vec<i32> = match a.flag("devices") {
                Some(d) => d.split(',').map(|x| num(x).map(|v| v as i32)).collect::<Result<_>>()?,
                None => vec![0; cuts.len() + 1],
            };
            let c = glyd_engine::qwen35::Config35::load(dir)?;
            let d = dir.to_path_buf();
            let make =
                std::sync::Arc::new(move |st: glyd_engine::dynamic::Stage, o: &glyd_engine::dynamic::Opts| glyd_engine::dynamic::qwen35::load_stage(&d, o, st));
            Box::new(glyd_engine::dynamic::split::Split::with(c.layers, c.eos.clone(), &cuts, &devices, &o, make)?)
        }
        None => Box::new(open35(a, dir)?.0),
    };
    let load = t0.elapsed().as_secs_f64();
    eprintln!("loaded in {load:.1} s: {}", e.describe());
    let eos = e.eos();
    let (mut steps, mut step_s) = (0usize, 0.0f64);
    let mut body = String::from("{");
    for (key, out_key) in [("prompts", ""), ("long_prompts", "long_")] {
        let Some(list) = v.get(key).and_then(Value::as_array) else { continue };
        let prompts: Vec<Vec<u32>> = list
            .iter()
            .map(|p| p.as_u64s().map(|v| v.into_iter().map(|x| x as u32).collect()).ok_or_else(|| Error(format!("{pf}: a prompt is not a list of token ids"))))
            .collect::<Result<_>>()?;
        let (mut tokens, mut lps) = (Vec::new(), Vec::new());
        for p in &prompts {
            e.reset()?;
            let mut s = e.prefill(p)?;
            let (mut ts, mut ls) = (Vec::new(), Vec::new());
            loop {
                ts.push(s.token.to_string());
                ls.push(format!("{:.6}", e.logprob(s.logit)?));
                if ts.len() >= new || (!a.has("ignore-eos") && eos.contains(&(s.token as i64))) {
                    break;
                }
                let t = Instant::now();
                s = e.step(s.token)?;
                step_s += t.elapsed().as_secs_f64();
                steps += 1;
            }
            tokens.push(ts.join(","));
            lps.push(ls.join(","));
        }
        body.push_str(&format!(
            "\"{out_key}tokens\":[{}],\"{out_key}logprobs\":[{}],",
            tokens.iter().map(|t| format!("[{t}]")).collect::<Vec<_>>().join(","),
            lps.iter().map(|t| format!("[{t}]")).collect::<Vec<_>>().join(",")
        ));
        eprintln!("{} {key} of up to {} tokens, {} new tokens each", prompts.len(), prompts.iter().map(Vec::len).max().unwrap_or(0), new);
    }
    body.push_str(&format!("\"load_s\":{load:.2},\"weights_bytes\":{}}}\n", e.bytes()));
    std::fs::write(of, body).map_err(|e| Error(format!("{of}: {e}")))?;
    eprintln!("{of}; a decode step {:.3} ms over {steps}", 1e3 * step_s / steps.max(1) as f64);
    Ok(())
}

/// As `check` (PROMPTS.json: {"prompts": [[ids]...], "long_prompts": ...}), on a Qwen3.5 model.
fn cmd_check35(a: &Args, dir: &Path) -> Result<()> {
    cmd_check35_dyn(a, dir)
}

/// One user on a Qwen3.5 model: a prompt of N random tokens (report::random_prompt, as tools/ref_speed.py's), timed `reps` times (the median), then `new` tokens a step at a time.
fn cmd_bench35(a: &Args, dir: &Path) -> Result<()> {
    let ns: Vec<usize> = a
        .flag("prompt")
        .unwrap_or("1024")
        .split(',')
        .map(|x| x.trim().parse().map_err(|_| Error(format!("--prompt {x}: not a number"))))
        .collect::<Result<_>>()?;
    let new = a.num("new", 256usize)?;
    let reps = a.num("reps", 5usize)?.max(1);
    let (mut e, load) = open35(a, dir)?;
    eprintln!("loaded in {load:.1} s: {}", e.describe());
    for n in ns {
        let prompt = report::random_prompt(n, e.desc().vocab);
        greedy(&mut e, &prompt[..prompt.len().min(64)], 4, true)?; // (lazy initializations out of the timing)
        let mut pre = Vec::new();
        let mut s = None;
        for _ in 0..=reps {
            e.reset()?;
            let t0 = Instant::now();
            s = Some(e.prefill(&prompt)?);
            pre.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        pre.remove(0); // (the first: the large products' first use)
        let mut sorted = pre.clone();
        sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let prefill_ms = sorted[sorted.len() / 2];
        let mut tok = s.map(|s| s.token).unwrap_or(0);
        let mut ms = Vec::with_capacity(new);
        for _ in 1..new {
            let t = Instant::now();
            tok = e.step(tok)?.token;
            ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let mean = ms.iter().sum::<f64>() / ms.len().max(1) as f64;
        ms.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let p = |q: f64| ms[((ms.len() - 1) as f64 * q) as usize];
        let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
        println!(
            "{{\"prompt_tokens\":{n},\"new_tokens\":{new},\"prefill_ms\":{prefill_ms:.1},\"prefill_ms_all\":[{}],\"decode_ms_mean\":{mean:.3},\"decode_ms_p50\":{:.3},\"decode_ms_p99\":{:.3},\"decode_tokens_per_s\":{:.2},\"load_s\":{load:.2},\"weights_gib\":{:.3},\"graph\":{}}}",
            pre.iter().map(|x| format!("{x:.1}")).collect::<Vec<_>>().join(","),
            p(0.5),
            p(0.99),
            1000.0 / mean,
            gib(e.weights_bytes()),
            e.has_graph()
        );
    }
    Ok(())
}
