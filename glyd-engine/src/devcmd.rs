//! The commands of a development build (`--features dev`, the default): the checks, the proofs and the measurements. A release build has none of them.
//!
//!   glyd-engine pack MODEL OUT [--threads N] [--layout auto|mma|mma12|i6|i5|i4]   pack a checkpoint (every matrix, the embedding too); i6, i5 and i4: a lossy level's glyd-v7 directory (Qwen3.5, Nemotron 3 Super: at i4 its experts in four-bit codes, every other matrix in six-bit ones; GLM-5.3-Flash)
//!   glyd-engine pack MODEL OUT --layout hummingbird --codes DIR [--threads N]   Nemotron 3 Super at the Hummingbird level (src/hbpack.rs): its routed experts' neuron rows from the export in DIR (nemotron_hbr.py --export) in codebook rows of 3 and 2 bits,
//!                                                             the signs of each layer's rotation beside them, every other matrix six-bit; glyd-gpu verify decodes the directory
//!   glyd-engine hbtoy MODEL OUT [--seed N] [--share PERCENT]   a tiny Nemotron-H model (nemotron_ref.py moetoy's toy/) packed at the Hummingbird level from a random export (OUT/export, PERCENT of the experts at 3 bits, the rest at 2) into OUT/packed
//!   glyd-engine cb2prefill [--n 1024] [--k 8] [--experts 216] [--check 16] [--reps 3] [--skew 0] [--pack DIR --layer L]   a prompt chunk through GLM-5.3-Flash's codebook experts on the tensor cores (src/cb2_prefill.rs)
//!   glyd-engine glmfmtcheck FIXTURE_DIR   the four formats of the pack (tern, cb2, cb3, cb4) against fixtures the pack writer made (tools/fmtfix.py)
//!   glyd-engine cb2check PACK_DIR [--layer L] [--expert E]   one expert of a glyd-hbr-codes/2 pack (src/cpu_cb.rs) against the pack's testvec.safetensors
//!   glyd-engine hbcheck [CASE...] [--speed] [--no-check]      the Hummingbird layer's kernels (src/moe_hb.rs, kernels/moe_hb.cu) against the CPU reference on random experts at 3 and 2 bits, all there and through places, and their speed (src/kcheck_hb.rs)
//!   glyd-engine run MODEL [--prompt TEXT] [--no-think] [--context N] [--port N] [--host H] [--temp T] [--layout auto|mma|mma12]
//!                                                             what `glyd run` starts: the model made ready on the first run (downloaded, packed for this GPU,
//!                                                             its KV tables made), then a chat in the terminal and at one URL, or one answer with --prompt
//!   glyd-engine serve MODEL [--port N] [--host H] [--context N]   the OpenAI-compatible API and the chat page
//!   glyd-engine fetch MODEL                                      the model's files downloaded into the Hugging Face cache (no GPU)
//!   glyd-engine prepare MODEL [--layout ...]                     downloaded and packed for this GPU, the KV tables left for the first run
//!   glyd-engine doctor | login | supports MODEL                  what this computer has for it; a Hugging Face token saved; whether the engine runs MODEL (exit 3: no)
//!   glyd-engine generate DIR PROMPT [--new N] [--ctx N] [--no-graph] [--expert-cache GIB|auto|auto:GIB] [--host-memory GIB|auto] [--offload-mode MODE[,MODE...]] [--cpu-threads N] [--cpu-pin]   a raw completion (a Nemotron 3 Super directory `pack` made, or a bf16 one with
//!                                                             --layout i6|i5|i4; --expert-cache: its experts in the computer's memory and a cache of them in GIB GiB of the GPU's, src/offload.rs); the prefill and the decode steps timed apart, with the cache's
//!                                                             counts for each; --offload-mode copy|cpu|split|auto[:THREADS][:pin] (a list runs the completion once for each, the cache cold at the start of each): what the cache does with an expert that is not
//!                                                             on the GPU (copy it, or compute its pairs on the CPU: --cpu-threads, --cpu-pin, --offload-window N, --offload-promote N); --host-memory GIB: the computer's memory the
//!                                                             experts may take (a Hummingbird directory keeps the rest on the disk and reads them as the router asks, src/offload.rs `Tier`; the counts of reads from the disk, and the
//!                                                             process's peak resident memory, are in the output)
//!   glyd-engine check DIR PROMPTS.json OUT.json [--new N] [--ctx N] [--no-graph]
//!   glyd-engine score DIR REF.json OUT.json [--ctx N]        teacher-forced top-1 and logprobs over REF's "score_ids"
//!   glyd-engine microbench DIR [--prompt N[,N...]] [--new N] [--ctx N] [--chunk N] [--reps N] [--no-graph] [--profile] [--profile-prefill] [--attn-decode]
//!   glyd-engine rowcheck DIR [--ctx N] [--tables T]          one sequence's logits in a step with 0 to 7 others, at every batch size
//!   glyd-engine rowprofile DIR [--rows B] [--len N] [--steps N] [--ctx N] [--tables T]   a decode step of B sequences, launch by launch
//!   glyd-engine kvtest DIR [--ctx N] [--tables T]            a known answer for the KV cache: bf16 values of every kind written through the engine's write path, read back, compared bit for bit
//!   glyd-engine stress DIR [--users N] [--requests N] [--kind natural|random|mixed] [--seed S] [--text FILE] [--ctx N] [--pool TOKENS] [--ref FILE | --write-ref FILE]   requests at several users on a small cache, against the same requests alone
//!   glyd-engine recompute DIR [--ctx N] [--tables T]         the keys and values of a sequence made by its steps against those made by running it again as a prompt
//!   glyd-engine topkcheck                                     the sampling kernels' 64 largest logits of a row and sum of its exponentials against the host's
//!   glyd-engine sampletime DIR PROMPT [--new N] [--ctx N]     a Qwen3.5 or Nemotron-H model's decode speed by greedy steps, by sampled ones from the GPU's candidates and from the row on the host (the same tokens)
//!   glyd-engine logitscheck [CASE...] [--fixtures DIR] [--cpu-ref] [--level L] [--list-cases]   each model's engine logits against its fixture in tests/logits (src/logits.rs); logitsrecord CASE --from cpu|engine --out FILE: a fixture made; logitsbound FIXTURE LEVEL=BOUND...: its bounds set; logitsmodel CASE DIR: the case's random model made; usercheck CASE [--users 4] [--level L]: the case's random model (gemma4, qwen35) served by the scheduler, each reply alone and at once against the one-at-a-time reply; paritycheck [CASE...] [--level L]: the hand-written Qwen3.5, Gemma 4 and Nemotron-H engines against the description executor on their R7 fixtures' tokens (R3b)
//!   glyd-engine kcheck IN_DIR OUT_DIR                        the kernels on inputs tools/kcheck.py wrote
//!   glyd-engine bandwidth [--gib N]                           the GPU's streaming read bandwidth
//!   glyd-engine rangecheck MODEL[:level] [--prompt 1024] [--new 256] [--cut L | --cuts A,B] [--ctx N] [--chunk N] [--workers [--sessions 2]]   a Qwen3.5 model whole, then as stages of layers chained in this process and (--workers) as worker processes
//!                                                             through the glyd-range/1 protocol (src/dynamic/range.rs, docs/range-worker.md): every step's logits and token compared bit for bit; "all checks passed"
//!   glyd-engine info DIR [--util F] [--ctx N]
//!   glyd-engine kvcal MODEL OUT.safetensors [--positions N]   the KV cache's tables, calibrated on the engine itself
//!   glyd-engine chat MODEL [--ctx N] [--temp T] [--no-think]  a chat in the terminal, also from a pipe (run's, with --chat)
//!   glyd-engine calcheck                                      the calibration's kernels and the prompt attention against the host's reading
//!   glyd-engine kcheck35                                      Qwen3.5's kernels against the CPU reference (src/kcheck35.rs)
//!   glyd-engine moecheck [CASE...] [--speed] [--no-check] [--dump DIR | --cpu-dump DIR]   the mixture-of-experts layer (src/moe.rs) against the CPU reference on random routers and Int5, Int6 and Int4 experts, and its speed (src/kcheck_moe.rs)
//!   glyd-engine cpuexperts [--bits 4|5|6] [--threads N] [--kernel auto|scalar|neon|avx2|avx512] [--experts N] [--pairs N] [--reps N] [--tokens N] [--misses M,M,...]   the CPU half of the sparse experts (src/cpu_experts.rs) timed on this computer: Nemotron 3 Super's experts made at random, pairs a second and GB/s read, a decode token's time for M missed experts
//!   glyd-engine nhkernels IN_DIR OUT_DIR                      Nemotron's kernels on the inputs tools/nemotron_ref.py wrote (src/kcheck_nh.rs); nhlayers DIR IO_DIR: layers of the real model or of a random tiny one with mixture-of-experts layers; nhtoy DIR IO_DIR [--expert-cache GIB|auto|auto:GIB] [--offload-mode MODE] [--cpu-threads N]: a few of its layers as a model (or the whole model through the expert cache, one eager run); nhdecode DIR OUT_DIR --layers L,L [--tokens N]: some layers of a glyd-v7 directory decoded to bf16; nhrandom OUT [--latent N]: a random model at Nemotron 3.5 Lightning's widths
//!   glyd-engine diskcheck DIR [--reads N] [--depth N]         the disk a Hummingbird directory is on, as the SSD tier reads it: random whole experts, page cache bypassed, one at a time and 16 at a time, MB/s (src/disk.rs); no GPU
//!   glyd-engine nhoffload DIR IO_DIR [--slots N,N,...] [--offload-mode cpu|split|auto] [--cpu-threads N] [--host-memory GIB,GIB,...]   the random tiny model run with all its experts on the GPU and through the expert cache (src/offload.rs) with N places a layer: the
//!                                                             logits equal bit for bit (the experts' second matrices in the row layout, the sparse product; GLYD_ENGINE_DENSE_DOWN=1: the block layout, the dense one); with a mode that computes the pairs of missed
//!                                                             experts on the CPU, within two places of bf16 of the largest logit (`logits_within`); --host-memory (a Hummingbird directory): the same passes again with every expert in the computer's
//!                                                             memory and with no more than that many GiB of them (the rest read from the directory), every logit the same to the bit in the mode asked for
//!   glyd-engine glmkernels IN_DIR OUT_DIR                     GLM-5.3-Flash's kernels on the inputs tools/glm_ref.py wrote (src/kcheck_glm.rs); glmlayers DIR IO_DIR: whole layers of the real model or of a random tiny one; glmtoy DIR IO_DIR: the tiny model whole; glmmtpreal SNAPSHOT REF_DIR OUT_DIR --layout i6|i5|i4: the real MTP head on the reference's hidden states; glmreal SNAPSHOT REF_DIR OUT_DIR --layers FIRST:END --layout i6|i5|i4 [--teacher]: the real model's layers one at a time against a reference's streams; glmmtp DIR IO_DIR: the MTP head on the tiny model's hidden states; glmspec DIR IO_DIR: a draft verified in one pass and the delta rules rewound after it, the tokens the same as greedy decode's whatever the drafts; glmprof DIR --layers L,L [--ctx N] [--reps N]: where a decode step of a layer spends its time, stage by stage in a graph's replay; glmdequant DIR OUT_DIR [--match TEXT]: its FP8 matrices as the loader reads them (bf16); glmdecode DIR OUT_DIR --layers L,L --layout i6|i5|i4: the packs of a mixture-of-experts layer's block decoded to bf16
//!
//! `generate`, `check` and `bench` run a Qwen3.5 model (its config.json says so: a directory in the Hub's layout, or a Hub id in the cache) on the description executor
//! (src/dynamic/qwen35.rs), one sequence, packed in memory as it loads (--layout auto|mma|mma12).
//!
//! DIR is a directory `pack` made, with kv-tables.safetensors beside it (tools/npz_to_tables.py).

use glyd_engine::args::Args;
use glyd_engine::conv::Gen;
use glyd_engine::ffi::{err, Error, Result};
use glyd_engine::model::{Engine, Opts, Reply};
use glyd_engine::sampler::Params;
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
        "chat" => std::process::exit(glyd_engine::cli::run(&Args::from(std::env::args().skip(1).chain(["--chat".to_string()])))),
        "generate" => cmd_run(a),
        "check" => cmd_check(a),
        "microbench" => cmd_bench(a),
        "score" => cmd_score(a),
        "rowcheck" => cmd_rowcheck(a),
        "batchcheck" => cmd_batchcheck(a),
        "kvtest" => cmd_kvtest(a),
        "stress" => cmd_stress(a),
        "recompute" => cmd_recompute(a),
        "rowprofile" => cmd_rowprofile(a),
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
        "rangecheck" => glyd_engine::dynamic::range::rangecheck(a),
        "cb2check" => glyd_engine::cpu_cb::check_pack(Path::new(a.need(1, "PACK_DIR")?), a.num("layer", 3usize)?, a.num("expert", 0usize)?),
        "nhkernels" => glyd_engine::kcheck_nh::kernels(Path::new(a.need(1, "IN_DIR")?), Path::new(a.need(2, "OUT_DIR")?)),
        "nhlayers" => glyd_engine::kcheck_nh::layers(Path::new(a.need(1, "DIR")?), Path::new(a.need(2, "IO_DIR")?)),
        "logitscheck" => glyd_engine::logits::check(a),
        "paritycheck" => glyd_engine::logits::parity(a),
        "logitsrecord" => glyd_engine::logits::record(a),
        "logitsbound" => glyd_engine::logits::set_bounds(a),
        "logitsmodel" => glyd_engine::logits::make_model(a),
        "usercheck" => glyd_engine::logits::users_check(a),
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
        "dsv4prompt" => glyd_engine::dsv4_ref::prompt(&a),
        "dsv4layers" => glyd_engine::dsv4_ref::layers(&a),
        "dsv4pack" => glyd_engine::dsv4_ref::pack_check(&a),
        "f8moecheck" => glyd_engine::dsv4_ref::f8moe_check(&a),
        "fp4stat" => glyd_engine::dsv4_ref::fp4_stat(&a),
        "f8stat" => glyd_engine::dsv4_ref::f8_stat(&a),
        "dsv4gen" => glyd_engine::dsv4_ref::gen(&a),
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
        "glmkernels" => glyd_engine::kcheck_glm::kernels(Path::new(a.need(1, "IN_DIR")?), Path::new(a.need(2, "OUT_DIR")?)),
        "glmlayers" => glyd_engine::kcheck_glm::layers(Path::new(a.need(1, "DIR")?), Path::new(a.need(2, "IO_DIR")?)),
        "glmdyn" => {
            let (cuts, devices) = glyd_engine::dynamic::split::split_args(&a)?;
            glyd_engine::kcheck_glm::dyn_check(
                Path::new(a.need(1, "DIR")?),
                a.num("tokens", 48)?,
                a.has("old"),
                a.layout()?.unwrap_or(glyd_engine::weights::Layout::Int6),
                &cuts,
                &devices,
            )
        }
        "glmrandom" => glyd_engine::kcheck_glm::random(
            Path::new(a.need(1, "OUT")?),
            a.has("fp8"),
            if a.has("real") { Some((a.num("layers", 4)?, a.num("experts", 32)?)) } else { None },
        ),
        "glmtoy" => glyd_engine::kcheck_glm::toy(Path::new(a.need(1, "DIR")?), Path::new(a.need(2, "IO_DIR")?)),
        "glmmtpreal" => {
            let level = a.layout()?.ok_or_else(|| Error("glmmtpreal needs --layout i6, i5 or i4".into()))?;
            glyd_engine::kcheck_glm::mtp_real(Path::new(a.need(1, "SNAPSHOT")?), Path::new(a.need(2, "REF_DIR")?), Path::new(a.need(3, "OUT_DIR")?), level)
        }
        "glmreal" => {
            let range = a.flag("layers").ok_or_else(|| Error("glmreal needs --layers FIRST:END".into()))?;
            let (first, end) = range
                .split_once(':')
                .and_then(|(x, y)| Some((x.parse::<usize>().ok()?, y.parse::<usize>().ok()?)))
                .ok_or_else(|| Error(format!("--layers {range}: not FIRST:END")))?;
            let level = a.layout()?.ok_or_else(|| Error("glmreal needs --layout i6, i5 or i4".into()))?;
            glyd_engine::kcheck_glm::real(
                Path::new(a.need(1, "SNAPSHOT")?),
                Path::new(a.need(2, "REF_DIR")?),
                Path::new(a.need(3, "OUT_DIR")?),
                first,
                end,
                level,
                a.flag("teacher").is_some(),
            )
        }
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
        "g4check" => glyd_engine::model_gemma4::check(a),
        "i4xcheck" => glyd_engine::model_gemma4::i4x_check(a),
        "g4spec" => glyd_engine::model_gemma4::spec_check(a),
        "g4tok" => glyd_engine::model_gemma4::tok_check(a),
        "g4score" => glyd_engine::model_gemma4::score(a),
        "g4prefill" => glyd_engine::model_gemma4::prefill_prof(a),
        "g4fatest" => glyd_engine::model_gemma4::fa_test(a),
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
        "glmserve" => glyd_engine::glm_serve::serve(Path::new(a.need(1, "CKPT")?), Path::new(a.need(2, "PACK")?), a),
        "glmcbtest" => glyd_engine::glm_serve::cb_test(Path::new(a.need(1, "PACK")?), Path::new(a.need(2, "TESTVEC")?)),
        "glmcbblock" => glyd_engine::glm_serve::cb_block(Path::new(a.need(1, "CKPT")?), Path::new(a.need(2, "PACK")?), a),
        "glmmtp" => glyd_engine::kcheck_glm::mtp(Path::new(a.need(1, "DIR")?), Path::new(a.need(2, "IO_DIR")?)),
        "glmspec" => glyd_engine::kcheck_glm::spec(Path::new(a.need(1, "DIR")?), Path::new(a.need(2, "IO_DIR")?)),
        "glmprof" => {
            let layers = a.flag("layers").ok_or_else(|| Error("glmprof needs --layers L,L,...".into()))?;
            let layers: Vec<usize> =
                layers.split(',').map(|x| x.trim().parse().map_err(|_| Error(format!("--layers {x}: not a number")))).collect::<Result<_>>()?;
            let num = |k: &str, d: usize| -> Result<usize> { a.flag(k).map_or(Ok(d), |v| v.parse().map_err(|_| Error(format!("--{k} {v}: not a number")))) };
            glyd_engine::kcheck_glm::prof(Path::new(a.need(1, "DIR")?), &layers, num("ctx", 2600)?, num("reps", 20)?)
        }
        "glmdequant" => glyd_engine::kcheck_glm::fp8(Path::new(a.need(1, "DIR")?), Path::new(a.need(2, "OUT_DIR")?), a.flag("match")),
        "glmdecode" => {
            let layers = a.flag("layers").ok_or_else(|| Error("glmdecode needs --layers L,L,...".into()))?;
            let layers: Vec<usize> =
                layers.split(',').map(|x| x.trim().parse().map_err(|_| Error(format!("--layers {x}: not a number")))).collect::<Result<_>>()?;
            let level = a.layout()?.ok_or_else(|| Error("glmdecode needs --layout i6, i5 or i4".into()))?;
            glyd_engine::kcheck_glm::decode(Path::new(a.need(1, "DIR")?), Path::new(a.need(2, "OUT_DIR")?), &layers, level)
        }
        "info" => cmd_info(a),
        "calcheck" => glyd_engine::calib::kernel_check(&glyd_engine::dev::Dev::new(0)?),
        "kvcal" => cmd_kvcal(a),
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

/// The engine for the model of the command line: a packed directory with its tables beside it (or `--tables`) as it is, else the model made ready in the cache (store::prepare): packed for
/// the GPU and its KV tables calibrated, each once. Also the directory the tokenizer is in.
fn open_dir(a: &Args) -> Result<(Engine, f64, std::path::PathBuf)> {
    open_dir_with(a, a.opts()?)
}

/// open_dir with these options in place of the command line's.
fn open_dir_with(a: &Args, o: Opts) -> Result<(Engine, f64, std::path::PathBuf)> {
    let t0 = Instant::now();
    let m = a.need(1, "DIR")?;
    let direct = Path::new(m);
    if o.tables.is_some() || (direct.join("glyd.json").exists() && direct.join("kv-tables.safetensors").exists()) {
        let dir = model_dir(a)?;
        let e = Engine::new(&dir, &o)?;
        return Ok((e, t0.elapsed().as_secs_f64(), dir));
    }
    let p = glyd_engine::store::prepare(m, o.ctx, o.layout, &|s| eprintln!("{s}"))?;
    let o = Opts { tables: Some(p.tables.clone()), ..o };
    let e = Engine::new(&p.dir, &o)?;
    Ok((e, t0.elapsed().as_secs_f64(), p.dir))
}

fn open(a: &Args) -> Result<(Engine, f64)> {
    open_dir(a).map(|(e, t, _)| (e, t))
}

fn cmd_run(a: &Args) -> Result<()> {
    if let Some(dir) = nemotron_dir(a) {
        return cmd_run_nh(a, &dir);
    }
    if let Some(dir) = qwen35_dir(a) {
        return cmd_run35(a, &dir);
    }
    use std::io::Write;
    let prompt = a.need(2, "PROMPT")?;
    let (mut e, load, dir) = open_dir(a)?;
    let tok = Tok::load(&dir)?;
    eprintln!("loaded in {load:.1} s: {}", e.describe());
    let ids = tok.encode(prompt)?;
    let t0 = Instant::now();
    print!("{prompt}");
    // the text so far is decoded again at every token and what is new is printed, but for a character not yet whole
    let mut toks: Vec<u32> = Vec::new();
    let mut shown = 0usize;
    let mut each = |s: &glyd_engine::model::Sample| {
        toks.push(s.token);
        if let Ok(t) = tok.decode(&toks) {
            if !t.ends_with('\u{FFFD}') && t.len() > shown {
                print!("{}", &t[shown..]);
                shown = t.len();
                std::io::stdout().flush().ok();
            }
        }
    };
    let out = e.generate_with(&ids, a.num("new", 64usize)?, a.has("ignore-eos"), false, &mut each)?;
    println!();
    eprintln!("{} prompt tokens, {} generated in {:.2} s", ids.len(), out.len(), t0.elapsed().as_secs_f64());
    Ok(())
}

/// PROMPTS.json: {"prompts": [[token ids], ...]}; OUT.json: {"tokens": [[...]], "logprobs": [[...]]}.
fn cmd_check(a: &Args) -> Result<()> {
    if let Some(dir) = nemotron_dir(a) {
        return cmd_check_nh(a, &dir);
    }
    if let Some(dir) = qwen35_dir(a) {
        return cmd_check35(a, &dir);
    }
    let pf = a.need(2, "PROMPTS.json")?;
    let of = a.need(3, "OUT.json")?;
    let v = json::parse(&std::fs::read_to_string(pf).map_err(|e| Error(format!("{pf}: {e}")))?).map_err(|e| Error(format!("{pf}: {e}")))?;
    let new = a.num("new", 64usize)?;
    let (mut e, load) = open(a)?;
    eprintln!("loaded in {load:.1} s: {}", e.describe());
    // "prompts" (the short ones) and, where the reference has them, "long_prompts"
    let mut body = String::from("{");
    for (key, out_key) in [("prompts", ""), ("long_prompts", "long_")] {
        let Some(list) = v.get(key).and_then(Value::as_array) else { continue };
        let prompts: Vec<Vec<u32>> = list
            .iter()
            .map(|p| p.as_u64s().map(|v| v.into_iter().map(|x| x as u32).collect()).ok_or_else(|| Error(format!("{pf}: a prompt is not a list of token ids"))))
            .collect::<Result<_>>()?;
        let (mut tokens, mut lps) = (Vec::new(), Vec::new());
        for p in &prompts {
            let out = e.generate(p, new, a.has("ignore-eos"), true)?;
            tokens.push(out.iter().map(|s| s.token.to_string()).collect::<Vec<_>>().join(","));
            lps.push(out.iter().map(|s| format!("{:.6}", s.logprob.unwrap_or(f32::NAN))).collect::<Vec<_>>().join(","));
        }
        body.push_str(&format!(
            "\"{out_key}tokens\":[{}],\"{out_key}logprobs\":[{}],",
            tokens.iter().map(|t| format!("[{t}]")).collect::<Vec<_>>().join(","),
            lps.iter().map(|t| format!("[{t}]")).collect::<Vec<_>>().join(",")
        ));
        eprintln!("{} {key} of up to {} tokens, {} new tokens each", prompts.len(), prompts.iter().map(Vec::len).max().unwrap_or(0), new);
    }
    body.push_str(&format!("\"load_s\":{:.2},\"weights_bytes\":{},\"cache_tokens\":{}}}\n", load, e.w.bytes, e.kv.tokens()));
    std::fs::write(of, body).map_err(|e| Error(format!("{of}: {e}")))?;
    eprintln!("{}", report::kv_status(&e));
    eprintln!("{of}");
    Ok(())
}

/// REF.json: {"score_ids": [token ids]}; OUT.json: {"score_top1": [...], "score_lp": [...]} (positions 0 .. n - 2).
fn cmd_score(a: &Args) -> Result<()> {
    let pf = a.need(2, "REF.json")?;
    let of = a.need(3, "OUT.json")?;
    let v = json::parse(&std::fs::read_to_string(pf).map_err(|e| Error(format!("{pf}: {e}")))?).map_err(|e| Error(format!("{pf}: {e}")))?;
    let ids: Vec<u32> =
        v.get("score_ids").and_then(Value::as_u64s).ok_or_else(|| Error(format!("{pf}: no score_ids")))?.into_iter().map(|x| x as u32).collect();
    let (mut e, load) = open(a)?;
    eprintln!("loaded in {load:.1} s: {}", e.describe());
    let t0 = Instant::now();
    let (top, lp) = e.score(&ids)?;
    eprintln!("{} positions scored in {:.1} s", top.len(), t0.elapsed().as_secs_f64());
    let body = format!(
        "{{\"score_top1\":[{}],\"score_lp\":[{}]}}\n",
        top.iter().map(u32::to_string).collect::<Vec<_>>().join(","),
        lp.iter().map(|x| format!("{x:.6}")).collect::<Vec<_>>().join(",")
    );
    std::fs::write(of, body).map_err(|e| Error(format!("{of}: {e}")))
}

fn cmd_bench(a: &Args) -> Result<()> {
    nemotron_h_refused(a)?;
    if let Some(dir) = qwen35_dir(a) {
        return cmd_bench35(a, &dir);
    }
    let ns: Vec<usize> = a
        .flag("prompt")
        .unwrap_or("1024")
        .split(',')
        .map(|x| x.trim().parse().map_err(|_| Error(format!("--prompt {x}: not a number"))))
        .collect::<Result<_>>()?;
    let new = a.num("new", 256usize)?;
    let (mut e, load) = open(a)?;
    eprintln!("loaded in {load:.1} s: {}", e.describe());
    for n in ns {
        report::bench(&mut e, n, new, load, a.has("profile"), a.has("profile-prefill"), a.num("reps", 5usize)?)?;
    }
    Ok(())
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

/// One sequence's logits in a decode step together with up to 7 others, at every number of rows and at the first and the last row: which of them are bit for bit the same. Exit status 1 where two numbers of rows
/// that round up to the same power of two (3 and 4, 5 to 8) differ.
fn cmd_rowcheck(a: &Args) -> Result<()> {
    use glyd_engine::model::Row;
    let mut o = a.opts()?;
    o.users = 8;
    o.pool = o.pool.max(o.ctx);
    let dir = model_dir(a)?;
    let mut e = Engine::new(&dir, &o)?;
    let per = 24usize; // pages a sequence has
    let lens = [211usize, 97, 150, 33, 264, 180, 120, 75];
    let seq_tokens = |k: usize| -> Vec<u32> { (0..lens[k]).map(|i| 1 + ((k * 7919 + i * 104_729) % 4000) as u32).collect() };
    // (rows, the focus sequence's row, its logits' hash, its token)
    let mut seen: Vec<(usize, usize, u64, u32)> = Vec::new();
    // --churn N: the cache first used by N rounds of 8 other sequences of random lengths, on the pages the check then rewrites (and every full page of them compared with the bf16 copy where GLYD_ENGINE_VERIFY is set)
    let churn = a.num("churn", 0usize)?;
    let mut rng = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move |m: usize| -> usize {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng % m as u64) as usize
    };
    for round in 0..churn {
        let pages: Vec<Vec<i32>> = (0..8).map(|k| (0..per).map(|j| (k * per + j) as i32).collect()).collect();
        let lens: Vec<usize> = (0..8).map(|_| 20 + next(340)).collect();
        for k in 0..8 {
            let t: Vec<u32> = (0..lens[k]).map(|_| 1 + next(4000) as u32).collect();
            let mut at = 0;
            while at < t.len() {
                let n = e.chunk().min(t.len() - at);
                e.prefill_chunk(&pages[k], at, &t[at..at + n], at + n == t.len())?;
                at += n;
            }
        }
        let all: Vec<i32> = (0..8usize).flat_map(|k| (0..lens[k] / 16).map(move |j| (k * per + j) as i32)).collect();
        let count = || {
            (
                glyd_engine::model::VERIFY_CHECKED.load(std::sync::atomic::Ordering::Relaxed),
                glyd_engine::model::VERIFY_DIFFER.load(std::sync::atomic::Ordering::Relaxed),
            )
        };
        let before = count();
        // (the first rounds: the full pages compared right after the prompts, before any decode step)
        e.verify_pages(&all)?;
        let after_prompts = count();
        if round < 3 {
            println!("round {round}: right after the prompts {} values checked, {} differing", after_prompts.0 - before.0, after_prompts.1 - before.1);
        }
        let rows: Vec<Row> = (0..8).map(|k| Row { token: 1 + next(4000) as u32, pos: lens[k], pages: &pages[k] }).collect();
        e.decode_batch(&rows)?;
        e.verify_pages(&all)?;
        if round < 3 {
            let c = count();
            println!("round {round}: after a decode step {} values checked, {} differing", c.0 - after_prompts.0, c.1 - after_prompts.1);
        }
    }
    let (checked, differ) = (
        glyd_engine::model::VERIFY_CHECKED.load(std::sync::atomic::Ordering::Relaxed),
        glyd_engine::model::VERIFY_DIFFER.load(std::sync::atomic::Ordering::Relaxed),
    );
    if churn > 0 {
        println!(
            "differing by head (keys, values): {:?}",
            glyd_engine::model::VERIFY_BY.iter().map(|a| a.load(std::sync::atomic::Ordering::Relaxed)).collect::<Vec<_>>()
        );
    }
    let lost = glyd_engine::model::VERIFY_LOST.load(std::sync::atomic::Ordering::Relaxed);
    println!("{churn} rounds of churn first; values checked against the bf16 copy {checked}, differing {differ}, values in blocks the cache had named lost (not compared) {lost}");
    if churn > 0 {
        println!("{}; blocks named lost {}", report::kv_status(&e), e.kv.void_named);
    }
    for b in 1..=8usize {
        for row in [0, b - 1] {
            // the focus sequence is sequence 0; the others fill the other rows
            let mut order: Vec<usize> = (1..b).collect();
            order.insert(row, 0);
            let pages: Vec<Vec<i32>> = (0..8).map(|k| (0..per).map(|j| (k * per + j) as i32).collect()).collect();
            for &k in &order {
                let t = seq_tokens(k);
                let mut at = 0;
                while at < t.len() {
                    let n = e.chunk().min(t.len() - at);
                    e.prefill_chunk(&pages[k], at, &t[at..at + n], at + n == t.len())?;
                    at += n;
                }
            }
            let rows: Vec<Row> = order.iter().map(|&k| Row { token: 1234 + k as u32, pos: lens[k], pages: &pages[k] }).collect();
            let s = e.decode_batch(&rows)?;
            let l = e.read_logits_row(row)?;
            let h = l.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| (h ^ x.to_bits() as u64).wrapping_mul(0x100_0000_01b3));
            println!("rows {b}, the sequence in row {row}: token {} logits hash {h:016x}", s[row].token);
            seen.push((b, row, h, s[row].token));
        }
    }
    let mut bad = false;
    for (i, x) in seen.iter().enumerate() {
        for y in &seen[i + 1..] {
            if x.0 != y.0 && x.0.next_power_of_two() == y.0.next_power_of_two() && (x.1 == 0) == (y.1 == 0) && x.2 != y.2 {
                println!("DIFFERENT: {} and {} rows (both round up to {})", x.0, y.0, x.0.next_power_of_two());
                bad = true;
            }
        }
    }
    let all = seen.iter().all(|x| x.2 == seen[0].2);
    println!(
        "{}",
        if all {
            "the sequence's logits are the same bit for bit at every number of rows and in either row"
        } else {
            "the sequence's logits differ between some numbers of rows (see above)"
        }
    );
    if bad {
        std::process::exit(1);
    }
    Ok(())
}

/// Prompt chunks run several to a pass (`prefill_batch`) against each alone (`prefill_chunk`): the logits of every sequence's last token and its token, bit for bit, for batches of 2 to 8 sequences of up to 64 tokens,
/// some starting after earlier pages. Exit status 1 where any differs.
fn cmd_batchcheck(a: &Args) -> Result<()> {
    use glyd_engine::model::PrefillJob;
    let mut o = a.opts()?;
    o.users = 8;
    o.pool = o.pool.max(o.ctx);
    let (mut e, _, _) = open_dir_with(a, o)?;
    let per = 24usize;
    let lens = [35usize, 20, 48, 33, 64, 18, 50, 27];
    let starts = [0usize, 0, 0, 0, 0, 0, 0, 0];
    let with_starts = [0usize, 16, 0, 32, 16, 0, 48, 16];
    // (passes of at most 64 rows in all take the library's route as each does alone: bit for bit the same; larger ones the route their rows give: the same token, the logits within noise)
    let (mut exact, mut near, mut bad, mut worst) = (0, 0, 0, 0f32);
    for (name, st) in [("from the start", starts), ("after earlier pages", with_starts)] {
        let toks = |k: usize| -> Vec<u32> { (0..st[k] + lens[k]).map(|i| 1 + ((k * 7919 + i * 104_729) % 4000) as u32).collect() };
        let pages: Vec<Vec<i32>> = (0..8).map(|k| (0..per).map(|j| (k * per + j) as i32).collect()).collect();
        let mut alone: Vec<(Vec<f32>, u32)> = Vec::new();
        for k in 0..8 {
            let t = toks(k);
            if st[k] > 0 {
                e.prefill_chunk(&pages[k], 0, &t[..st[k]], false)?;
            }
            let s = e.prefill_chunk(&pages[k], st[k], &t[st[k]..], true)?.expect("a sample");
            alone.push((e.read_logits_row(0)?, s.token));
        }
        for b in 2..=8usize {
            for shift in [0usize, 3] {
                let ks: Vec<usize> = (0..b).map(|j| (j + shift) % 8).collect();
                for &k in &ks {
                    if st[k] > 0 {
                        e.prefill_chunk(&pages[k], 0, &toks(k)[..st[k]], false)?;
                    }
                }
                let ts: Vec<Vec<u32>> = ks.iter().map(|&k| toks(k)).collect();
                let jobs: Vec<PrefillJob> =
                    ks.iter().enumerate().map(|(j, &k)| PrefillJob { pages: &pages[k], start: st[k], tokens: &ts[j][st[k]..], last: true }).collect();
                let rows: usize = jobs.iter().map(|j| j.tokens.len()).sum();
                let out = e.prefill_batch(&jobs)?;
                for (j, &k) in ks.iter().enumerate() {
                    let l = e.read_logits_row(j)?;
                    let tok = out[j].expect("a sample").token;
                    let d = l.iter().zip(&alone[k].0).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
                    if tok != alone[k].1 {
                        bad += 1;
                        println!("{name}: {b} in a pass ({rows} rows): sequence {k} in row {j}: token {tok}, alone {}", alone[k].1);
                    } else if d == 0.0 {
                        exact += 1;
                    } else if rows <= 64 {
                        bad += 1;
                        println!("{name}: {b} in a pass ({rows} rows, one route): sequence {k}: logits differ by {d}");
                    } else {
                        near += 1;
                        worst = worst.max(d);
                    }
                }
            }
        }
    }
    println!("sequences in passes of 2 to 8: {exact} bit for bit as alone, {near} the same token with logits within {worst} of alone's (passes of more than 64 rows), {bad} wrong");
    if bad > 0 {
        std::process::exit(1);
    }
    Ok(())
}

/// A decode step of `--rows` sequences of `--len` tokens, GPU time by kind of work.
fn cmd_rowprofile(a: &Args) -> Result<()> {
    let b = a.num("rows", 8usize)?;
    let len = a.num("len", 1100usize)?;
    let mut o = a.opts()?;
    o.users = b;
    o.pool = o.pool.max(o.ctx);
    let mut e = Engine::new(&model_dir(a)?, &o)?;
    let rows = e.profile_rows(b, len, a.num("steps", 20usize)?)?;
    println!("a step of {b} sequences of {len} tokens, ms by kind of work (launches a step):");
    for (label, ms, n) in rows {
        println!("  {ms:8.3}  {label}{}", if n > 0 { format!(" x{n}") } else { String::new() });
    }
    Ok(())
}

/// A known answer for the KV cache (src/kvtest.rs): host-made bf16 values of every kind written into blocks of every layer through the engine's write path, read back, compared bit for bit; written again and compared again;
/// the extra memory empty after every block is released. Exit status 1 where anything differs or was lost.
fn cmd_kvtest(a: &Args) -> Result<()> {
    let mut o = a.opts()?;
    o.users = 0;
    o.graph = false;
    // (as `run`: a repo id is made ready in the cache, its tables with it, not opened from its Hub snapshot)
    let (mut e, _, _) = open_dir_with(a, o)?;
    if !glyd_engine::kvtest::run(&mut e)? {
        std::process::exit(1);
    }
    Ok(())
}

/// The requests of src/stress.rs at several users on a small cache (`--pool` tokens, default a window for each user), against the same requests alone (`--ref FILE` of `--write-ref FILE`); exit status 1 where anything is not as it should be.
fn cmd_stress(a: &Args) -> Result<()> {
    let mut o = a.opts()?;
    o.ctx = a.either("context", "ctx", 2048usize)?;
    let writing = a.has("write-ref");
    let users = if writing { 1 } else { a.num("users", 4usize)?.clamp(1, glyd_engine::model::MAX_ROWS) };
    o.users = users;
    o.pool = a.num("pool", o.ctx * users)?;
    let dir = model_dir(a)?;
    let mut e = Engine::new(&dir, &o)?;
    let tok = Tok::load(&dir)?;
    if !glyd_engine::stress::run(&mut e, &tok, a)? {
        std::process::exit(1);
    }
    Ok(())
}

/// A sequence run by its steps (a prompt, then tokens one at a time) and again as one prompt in other pieces: the keys and values the cache holds for it, compared value for value (src/model.rs `compare_blocks`).
fn cmd_recompute(a: &Args) -> Result<()> {
    let mut o = a.opts()?;
    o.ctx = a.either("context", "ctx", 4096usize)?;
    o.users = 2;
    o.pool = o.ctx * 2;
    let dir = model_dir(a)?;
    let mut e = Engine::new(&dir, &o)?;
    let tok = Tok::load(&dir)?;
    let text = glyd_engine::calib::corpus();
    let words: Vec<&str> = text.split_whitespace().collect();
    let prompt_words = a.num("words", 150usize)?;
    let prompt = tok.encode_with(&words[200..200 + prompt_words].join(" "), false)?;
    let new = a.num("new", 40usize)?;
    let (pa, pb) = (glyd_engine::model::first_pages(0, 64), glyd_engine::model::first_pages(64, 64));
    // A: the whole prompt in one chunk, then `new` tokens a step at a time (greedy)
    let mut toks = prompt.clone();
    let mut s = e.prefill_chunk(&pa, 0, &prompt, true)?.expect("a sample");
    for i in 0..new {
        toks.push(s.token);
        let row = glyd_engine::model::Row { token: s.token, pos: prompt.len() + i, pages: &pa };
        s = e.decode_batch(&[row])?[0];
    }
    // the cache holds the prompt and the first `new` tokens' keys and values (the last token chosen has not been run)
    let held = prompt.len() + new;
    println!("a sequence of {} prompt tokens (run in one piece) and {new} tokens made by steps, run again as one prompt in pieces of the sizes below: values of the keys and values that differ from those the first run made", prompt.len());
    // B: the same tokens as one prompt, in pieces of each size
    for piece in [held, 128, 96, 80, 65, 64, 48, 16] {
        let mut at = 0;
        while at < held {
            let n = piece.min(held - at);
            e.prefill_chunk(&pb, at, &toks[at..at + n], at + n == held)?;
            at += n;
        }
        let (checked, differ_prompt) = e.compare_blocks(&pa, &pb, 0, prompt.len())?;
        let (checked2, differ_steps) = e.compare_blocks(&pa, &pb, prompt.len(), new)?;
        println!("  pieces of {piece:3} tokens: the prompt's {checked} values: {differ_prompt} differ; the {new} tokens made by steps, {checked2} values: {differ_steps} differ");
    }
    Ok(())
}

fn cmd_pack(a: &Args) -> Result<()> {
    use glyd_engine::weights::Layout;
    let model = a.need(1, "MODEL")?;
    if a.flag("layout") == Some("hummingbird") {
        let codes = a.flag("codes").ok_or_else(|| Error("--layout hummingbird needs --codes DIR, an export of the experts' neuron rows".into()))?;
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
    let gemma = glyd_engine::model_gemma4::ConfigG4::load(&src.dir).is_ok();
    let layout = match a.layout()?.filter(|l| gemma || matches!(l, Layout::Int6 | Layout::Int5 | Layout::Int4 | Layout::Int4H | Layout::Int3)) {
        Some(l) => l,
        None if gemma => Layout::Tiered,
        None => layout_flag(a, Some(&glyd_engine::weights::Config::load(&src.dir)?))?.unwrap_or(Layout::Tiered),
    };
    pack::run(model, Path::new(a.need(2, "OUT")?), a.num("threads", 16usize)?, layout)
}

/// `hbtoy MODEL OUT`: MODEL a tiny Nemotron-H model (`nemotron_ref.py moetoy`'s toy/ directory, bf16), a random export for its experts in OUT/export and the model packed at the Hummingbird level from it into OUT/packed.
fn cmd_hbtoy(a: &Args) -> Result<()> {
    let model = a.need(1, "MODEL")?;
    let out = Path::new(a.need(2, "OUT")?);
    let c = glyd_engine::nemotron::ConfigNemotronH::load(&model_dir(a)?)?;
    glyd_engine::hbpack::toy_export(&c, &out.join("export"), a.num("seed", 1u64)?, a.num("share", 40usize)?)?;
    glyd_engine::hbpack::run(model, &out.join("packed"), a.num("threads", 4usize)?, &out.join("export"))
}

/// MODEL: a directory or a Hub repo id in the local cache; OUT: the tables' file. The model is loaded as `run` loads it, with no cache, and run over the corpus.
fn cmd_kvcal(a: &Args) -> Result<()> {
    let dir = model_dir(a)?;
    let out = Path::new(a.need(2, "OUT.safetensors")?);
    let positions = a.num("positions", 4096usize)?;
    let mut o = a.opts()?;
    o.ctx = positions;
    o.chunk = a.num("chunk", 4096usize)?;
    let t0 = Instant::now();
    let e = Engine::for_calibration(&dir, &o)?;
    eprintln!("loaded in {:.1} s", t0.elapsed().as_secs_f64());
    let tok = Tok::load(&dir)?;
    let chat = glyd_engine::chat::Chat::load(&dir)?;
    let name = a.need(1, "MODEL")?.to_string();
    let t = glyd_engine::calib::calibrate(&e, &tok, chat.as_ref(), &glyd_engine::calib::Options { positions, model: name }, &|s| eprintln!("{s}"))?;
    glyd_engine::calib::save(out, &t)?;
    eprintln!("{}", out.display());
    match t.refused {
        Some(why) => err(why),
        None => Ok(()),
    }
}

fn cmd_info(a: &Args) -> Result<()> {
    let (e, load) = open(a)?;
    println!("{}", e.describe());
    println!("start {load:.1} s");
    println!("{}", report::memory(&e));
    println!("bits a weight: {}", report::bits(&e));
    Ok(())
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
) -> Result<(glyd_engine::model_nh::EngineNh, f64, Option<std::rc::Rc<glyd_engine::offload::ExpertStore>>, Vec<glyd_engine::offload::Offload>)> {
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
    let o = glyd_engine::model_nh::OptsNh {
        ctx: a.num("ctx", 4096usize)?,
        chunk: a.num("chunk", 2048usize)?,
        graph: !a.has("no-graph"),
        layout: Some(level),
        ..Default::default()
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
    let make: glyd_engine::model_nh::MoeFactory<'_> = match &cached {
        Some(f) => f,
        None => &plain,
    };
    let e = glyd_engine::model_nh::EngineNh::new(dir, &o, Some(make))?;
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
    if let Some(dir) = qwen35_dir(a) {
        let (mut e, load) = open35(a, &dir)?;
        eprintln!("loaded in {load:.1} s: {}", e.describe());
        return time_replies(&mut e, |e, host| e.top.off = host, &Tok::load(&dir)?.encode(prompt)?, &glyd_engine::sampler::generation_defaults(&dir), new);
    }
    err("sampletime runs a Qwen3.5 or a Nemotron-H model")
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
        while out.len() < new && (ignore_eos || !e.w.cfg.eos.contains(&(s.token as i64))) {
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
fn qwen35_dir(a: &Args) -> Option<std::path::PathBuf> {
    let d = model_dir(a).ok()?;
    glyd_engine::qwen35::Config35::load(&d).ok().map(|_| d)
}

/// A Qwen3.5-family model on the description executor (the hand-written engine is gone: R3b, #137).
fn open35(a: &Args, dir: &Path) -> Result<(glyd_engine::dynamic::Dyn, f64)> {
    let t0 = Instant::now();
    let o = glyd_engine::dynamic::Opts {
        ctx: a.num("ctx", 4096usize)?,
        chunk: a.num("chunk", 2048usize)?,
        graph: !a.has("no-graph"),
        device: 0,
        layout: a.layout()?,
    };
    let e = glyd_engine::dynamic::qwen35::load(dir, &o)?;
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

/// A Nemotron-H model on the old engine or the dynamic one (its experts all on the GPU).
enum NhRun {
    Old(Box<glyd_engine::model_nh::EngineNh>),
    New(Box<glyd_engine::dynamic::Dyn>),
    Split(Box<glyd_engine::dynamic::split::Split>),
}

impl NhRun {
    fn reset(&mut self) -> Result<()> {
        match self {
            NhRun::Old(e) => e.reset(),
            NhRun::New(e) => e.reset(),
            NhRun::Split(e) => e.reset(),
        }
    }

    fn prefill(&mut self, t: &[u32]) -> Result<glyd_engine::model::Sample> {
        match self {
            NhRun::Old(e) => e.prefill(t),
            NhRun::New(e) => e.prefill(t),
            NhRun::Split(e) => e.prefill(t),
        }
    }

    fn step(&mut self, t: u32) -> Result<glyd_engine::model::Sample> {
        match self {
            NhRun::Old(e) => e.step(t),
            NhRun::New(e) => e.step(t),
            NhRun::Split(e) => e.step(t),
        }
    }
}

/// `check MODEL PROMPTS.json OUT.json --layout LEVEL [--dynamic]` on a Nemotron-H model: the greedy tokens and each one's logit (OUT.json {"tokens", "logits", "long_..."}), by the old engine or the
/// dynamic one, its experts at LEVEL (penguin or a lossy level) all on the GPU: the two engines' files compare to the digit.
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
    } else if a.has("dynamic") {
        let o = glyd_engine::dynamic::Opts { ctx, chunk, graph: !a.has("no-graph"), device: 0, layout: Some(level) };
        NhRun::New(Box::new(glyd_engine::dynamic::nemotron::load(dir, &o, level)?))
    } else {
        let o = glyd_engine::model_nh::OptsNh { ctx, chunk, graph: !a.has("no-graph"), layout: Some(level), ..Default::default() };
        let make = glyd_engine::moe_nh::factory(level);
        NhRun::Old(Box::new(glyd_engine::model_nh::EngineNh::new(dir, &o, Some(&make))?))
    };
    let load = t0.elapsed().as_secs_f64();
    eprintln!(
        "loaded in {load:.1} s: {}",
        match &e {
            NhRun::Old(x) => x.describe(),
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
    fn prefill(&mut self, t: &[u32]) -> Result<glyd_engine::model::Sample>;
    fn step(&mut self, t: u32) -> Result<glyd_engine::model::Sample>;
    fn logprob(&self, logit: f32) -> Result<f32>;
    fn describe(&self) -> String;
    fn eos(&self) -> Vec<i64>;
    fn bytes(&self) -> usize;
}

impl Q35Run for glyd_engine::dynamic::Dyn {
    fn reset(&mut self) -> Result<()> {
        glyd_engine::dynamic::Dyn::reset(self)
    }
    fn prefill(&mut self, t: &[u32]) -> Result<glyd_engine::model::Sample> {
        glyd_engine::dynamic::Dyn::prefill(self, t)
    }
    fn step(&mut self, t: u32) -> Result<glyd_engine::model::Sample> {
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
    fn prefill(&mut self, t: &[u32]) -> Result<glyd_engine::model::Sample> {
        glyd_engine::dynamic::split::Split::prefill(self, t)
    }
    fn step(&mut self, t: u32) -> Result<glyd_engine::model::Sample> {
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
        None => Box::new(glyd_engine::dynamic::qwen35::load(dir, &o)?),
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
