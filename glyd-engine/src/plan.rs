//! The planner: what a run of MODEL does on a machine, worked out before anything is downloaded. It chooses the level, where
//! the weights, the experts and the KV cache live, the context, and predicts decode tokens/s, from the model's config.json
//! and the machine. Pure functions of explicit inputs (a probed machine, or a preset or JSON description of one), so every
//! branch is a host test. `glyd-engine fit MODEL[:LEVEL] [--machine NAME|FILE]` prints the plan.
//!
//! The order of preference (docs/architecture.md, "Adaptation"): the release's weights exactly (penguin) when they fit the
//! GPUs, else the highest lossy level that does, else (a mixture of experts) its experts in the computer's memory with the
//! CPU computing the ones the GPU does not hold, else a refusal with the numbers. One GPU is preferred to a split.

use crate::levels;
use crate::runtime::config::{ModelConfig, Width};
use glyd_gpu::json::{self, Value};

const GIB: f64 = (1u64 << 30) as f64;
const GB: f64 = 1e9;

// ---- the machine ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Vendor {
    Nvidia,
    Amd,
    Apple,
}

#[derive(Clone, Debug)]
pub struct Device {
    pub name: String,
    pub vendor: Vendor,
    pub cc: (i32, i32),
    pub total: u64,
    pub free: u64,
    /// Memory bandwidth, GB/s: the maker's figure (`BANDWIDTH`), or what a machine's JSON says.
    pub bandwidth: f64,
    /// What a kernel streaming the GPU's memory reads, GB/s, as the rate probe measured it on this computer (`rates_in`); None for a preset or a GPU not probed,
    /// whose rate is the maker's figure times `STREAM_OF_SPEC`.
    pub stream: Option<f64>,
    /// The dense bf16 tensor rate (fp32 sums), TFLOPS, from the makers' sheets (`TFLOPS`); 0 where the planner does not know it (no prefill prediction).
    pub tflops: f64,
    /// What decode reached here against the prediction (observed over predicted: p10, p50, p90), from the runs this computer recorded on this GPU (`observed`);
    /// None before any.
    pub observed: Option<[f64; 3]>,
    /// What prompts reached here against the prefill prediction (the median), from the same runs.
    pub observed_prefill: Option<f64>,
    /// The figures its predictions start from: its class's (`prior_for`), and where they come from with the band they are given in.
    pub consts: Consts,
    pub prior: Prior,
    /// The CUDA version the driver supports (13000 for 13.0); 0 where none.
    pub driver: i32,
    /// The GPU shares the computer's memory (Apple Silicon, DGX Spark, Jetson, Strix Halo): its memory is not a tier apart
    /// from RAM, so experts are never "moved to RAM" and the computer's free memory is not counted twice.
    pub unified: bool,
    /// Whether the computer has NVIDIA's cuBLAS for a long prompt's decoded product (`ffi::Blas`, opened at run time, never shipped): a preset's or a
    /// described machine's own word, so that its plan never turns on the computer that plans it; None for a GPU of this computer, whose loader is asked
    /// (`Blas::present`) when a plan needs it.
    pub cublas: Option<bool>,
}

#[derive(Clone, Debug)]
pub struct Machine {
    pub name: String,
    pub devices: Vec<Device>,
    /// The computer's free memory, and its read bandwidth (GB/s).
    pub ram_free: u64,
    pub ram_bandwidth: f64,
    pub cores: usize,
    /// The CPU architecture (`std::env::consts::ARCH`'s words: x86_64, aarch64).
    pub arch: String,
}

/// Memory bandwidth (GB/s) of the GPUs the planner knows, by the words their names hold (most specific first).
const BANDWIDTH: &[(&str, f64)] = &[
    ("GH200", 4000.0),
    ("H200", 4800.0),
    ("H100 PCIe", 2000.0),
    ("H100", 3350.0),
    ("B200", 8000.0),
    ("A100-SXM4-80GB", 2039.0),
    ("A100 80GB", 1935.0),
    ("A100", 1555.0),
    ("L40S", 864.0),
    ("L40", 864.0),
    ("L4", 300.0),
    ("A10G", 600.0),
    ("A10", 600.0),
    ("T4", 320.0),
    ("V100", 900.0),
    ("RTX 6000 Ada", 960.0),
    ("RTX PRO 6000", 1792.0),
    ("A6000", 768.0),
    ("5090", 1792.0),
    ("5080", 960.0),
    ("5070 Ti", 896.0),
    ("5070", 672.0),
    ("5060 Ti", 448.0),
    ("4090", 1008.0),
    ("4080 SUPER", 736.0),
    ("4080", 717.0),
    ("4070 Ti SUPER", 672.0),
    ("4070 Ti", 504.0),
    ("4070 SUPER", 504.0),
    ("4070", 504.0),
    ("4060 Ti", 288.0),
    ("4060", 272.0),
    ("3090 Ti", 1008.0),
    ("3090", 936.0),
    ("3080 Ti", 912.0),
    ("3080", 760.0),
    ("3070", 448.0),
    ("3060", 360.0),
    ("MI355X", 8000.0),
    ("MI325X", 6000.0),
    ("MI300X", 5300.0),
    ("7900 XTX", 960.0),
    ("7900 XT", 800.0),
    ("9070 XT", 640.0),
    ("M4 Max", 546.0),
    ("M4 Pro", 273.0),
    ("M3 Ultra", 819.0),
    ("M3 Max", 400.0),
    ("M2 Ultra", 800.0),
    ("M2 Max", 400.0),
    ("M1 Max", 400.0),
    ("GB10", 273.0),
    ("Thor", 273.0),
    ("Orin", 204.8),
    ("8060S", 256.0),
];

/// GPUs that share the computer's memory, by the words their names hold.
const UNIFIED: &[&str] = &["GB10", "Thor", "Orin", "Jetson", "Radeon 8060S", "Radeon 8050S", "Radeon 890M", "Radeon 880M"];

/// Whether a GPU shares the computer's memory: every Apple GPU, and the integrated ones named in UNIFIED.
pub fn is_unified(name: &str, vendor: Vendor) -> bool {
    vendor == Vendor::Apple || UNIFIED.iter().any(|w| name.contains(w))
}

/// A GPU's memory bandwidth from its name, None where the planner does not know it (`glyd-engine rates` measures it).
pub fn bandwidth_of(name: &str) -> Option<f64> {
    BANDWIDTH.iter().find(|(w, _)| name.contains(w)).map(|(_, b)| *b)
}

/// The dense bf16 tensor rate (fp32 sums) of the GPUs the planner knows, TFLOPS, from the makers' sheets, by the words their names hold (most specific first):
/// the prefill prediction's (`prefill_of`); a GPU not here has none.
const TFLOPS: &[(&str, f64)] = &[
    ("GH200", 989.0),
    ("H200", 989.0),
    ("H100 PCIe", 756.0),
    ("H100", 989.0),
    ("B200", 2250.0),
    ("A100", 312.0),
    ("L40S", 362.0),
    ("L40", 181.0),
    ("L4", 121.0),
    ("A10G", 0.0), // (not known: the A10's figure is not the A10G's)
    ("A10", 125.0),
    ("RTX 6000 Ada", 364.0),
    ("A6000", 154.8),
    ("5090", 209.5),
    ("5080", 112.6),
    ("5070 Ti", 87.9),
    ("5070", 61.8),
    ("5060 Ti", 47.4),
    ("4090", 165.2),
    ("4080 SUPER", 104.5),
    ("4080", 97.5),
    ("4070 Ti SUPER", 88.2),
    ("4070 Ti", 80.2),
    ("4070 SUPER", 71.0),
    ("4070", 58.3),
    ("4060 Ti", 44.1),
    ("4060 Laptop", 29.1),
    ("4060", 30.2),
    ("3090 Ti", 80.0),
    ("3090", 71.0),
    ("3080 Ti", 68.2),
    ("3080", 59.5),
    ("3070", 40.6),
    ("3060", 25.6),
    ("MI325X", 1307.0),
    ("MI300X", 1307.0),
    ("7900 XTX", 123.0),
];

/// A GPU's dense bf16 tensor rate from its name (`TFLOPS`), None where the planner does not know it.
pub fn tflops_of(name: &str) -> Option<f64> {
    TFLOPS.iter().find(|(w, _)| name.contains(w)).map(|(_, t)| *t).filter(|&t| t > 0.0)
}

impl Device {
    /// A GPU the driver lists, with the rates its name gives (`BANDWIDTH`, `TFLOPS`); nothing measured or observed yet.
    pub fn listed(name: String, vendor: Vendor, cc: (i32, i32), total: u64, free: u64, driver: i32) -> Device {
        Device {
            bandwidth: bandwidth_of(&name).unwrap_or(300.0),
            stream: None,
            tflops: tflops_of(&name).unwrap_or(0.0),
            observed: None,
            observed_prefill: None,
            consts: prior_for(cc).consts,
            prior: prior_for(cc),
            unified: is_unified(&name, vendor),
            cublas: None,
            name,
            vendor,
            cc,
            total,
            free,
            driver,
        }
    }

    /// Bytes a second decode's reads come at: the rate the probe measured here, else the maker's figure times `STREAM_OF_SPEC`.
    pub fn rate(&self) -> f64 {
        self.stream.unwrap_or(self.bandwidth * STREAM_OF_SPEC) * GB
    }

    /// What its predictions stand on, in words: the runs observed on it, else its class's records (a fitted point) with the error of each against the figures
    /// fitted without it, else the other classes' with the error a class with no records had (leave-one-class-out), else the defaults.
    pub fn basis(&self) -> String {
        let pct = |b: [f64; 3]| format!("{:+.0}% to {:+.0}%", (1.0 / b[2] - 1.0) * 100.0, (1.0 / b[0] - 1.0) * 100.0);
        let class = class_of(self.cc);
        match (self.observed, self.prior.source) {
            (Some(b), _) => format!("measured: the runs on this GPU (p10-p90 of their error {})", pct(b)),
            (None, Source::Own) => {
                format!("class prior: {class}'s {} records (error of one predicted from the others: {})", self.prior.records, pct(self.prior.band))
            }
            (None, Source::Others) => format!(
                "class prior: no record of {class}, the other classes' {} records (scored leave-one-class-out: {})",
                self.prior.records,
                pct(self.prior.band)
            ),
            (None, Source::Default) => "the default figures (no records)".to_string(),
        }
    }

    /// The decode prediction's factor and band on this GPU: p10, p50 and p90 of what runs here reached against their prediction (`observed`), else `SPREAD`.
    pub fn band(&self) -> [f64; 3] {
        self.observed.unwrap_or(self.prior.band)
    }
}

fn gpu(name: &str, vendor: Vendor, cc: (i32, i32), total_gib: f64, free_gib: f64, driver: i32) -> Device {
    let gib = |g: f64| (g * GIB) as u64;
    Device {
        name: name.into(),
        vendor,
        cc,
        total: gib(total_gib),
        free: gib(free_gib),
        bandwidth: bandwidth_of(name).unwrap_or(300.0),
        stream: None,
        tflops: tflops_of(name).unwrap_or(0.0),
        observed: None,
        observed_prefill: None,
        consts: prior_for(cc).consts,
        prior: prior_for(cc),
        driver,
        unified: is_unified(name, vendor),
        cublas: Some(false),
    }
}

impl Machine {
    /// The machines `--machine NAME` describes, to see a plan for a computer one does not have.
    pub fn preset(name: &str) -> Option<Machine> {
        let nv = Vendor::Nvidia;
        let (devices, ram_gib, ram_bw, cores) = match name {
            "rtx4060-laptop" => (vec![gpu("NVIDIA GeForce RTX 4060 Laptop GPU", nv, (8, 9), 8.0, 7.6, 12070)], 24.0, 60.0, 8),
            "rtx4080s" => (vec![gpu("NVIDIA GeForce RTX 4080 SUPER", nv, (8, 9), 16.0, 15.2, 13000)], 56.0, 70.0, 16),
            "rtx4090" => (vec![gpu("NVIDIA GeForce RTX 4090", nv, (8, 9), 24.0, 23.0, 13000)], 56.0, 70.0, 16),
            "2x3090" => (
                vec![gpu("NVIDIA GeForce RTX 3090", nv, (8, 6), 24.0, 23.2, 13000), gpu("NVIDIA GeForce RTX 3090", nv, (8, 6), 24.0, 23.2, 13000)],
                120.0,
                60.0,
                16,
            ),
            "l4" => (vec![gpu("NVIDIA L4", nv, (8, 9), 24.0, 22.0, 13000)], 14.0, 40.0, 4),
            "a6000" => (vec![gpu("NVIDIA RTX A6000", nv, (8, 6), 48.0, 47.0, 13000)], 120.0, 60.0, 16),
            "a100-80g" => (vec![gpu("NVIDIA A100-SXM4-80GB", nv, (8, 0), 80.0, 79.0, 13000)], 200.0, 100.0, 32),
            "h100" => (vec![gpu("NVIDIA H100 80GB HBM3", nv, (9, 0), 80.0, 79.0, 13000)], 200.0, 150.0, 32),
            "8xh100" => ((0..8).map(|_| gpu("NVIDIA H100 80GB HBM3", nv, (9, 0), 80.0, 79.0, 13000)).collect(), 1800.0, 300.0, 192),
            "rx7900xtx" => (vec![gpu("AMD Radeon RX 7900 XTX", Vendor::Amd, (11, 0), 24.0, 23.0, 0)], 56.0, 70.0, 16),
            "m4max-64g" => (vec![gpu("Apple M4 Max", Vendor::Apple, (0, 0), 48.0, 44.0, 0)], 12.0, 546.0, 16),
            "dgx-spark" => (vec![gpu("NVIDIA GB10", nv, (12, 1), 119.0, 112.0, 13000)], 112.0, 273.0, 20),
            "strix-halo-128g" => (vec![gpu("AMD Radeon 8060S", Vendor::Amd, (11, 5), 96.0, 92.0, 0)], 28.0, 256.0, 16),
            _ => return None,
        };
        let arch = if name == "dgx-spark" { "aarch64" } else { "x86_64" }.into();
        Some(Machine { name: name.into(), devices, ram_free: (ram_gib * GIB) as u64, ram_bandwidth: ram_bw, cores, arch })
    }

    pub const PRESETS: [&'static str; 13] = [
        "rtx4060-laptop",
        "rtx4080s",
        "rtx4090",
        "2x3090",
        "l4",
        "a6000",
        "a100-80g",
        "h100",
        "8xh100",
        "rx7900xtx",
        "m4max-64g",
        "dgx-spark",
        "strix-halo-128g",
    ];

    /// A machine described in JSON: {"name", "ram_free_gib", "ram_bandwidth", "cores", "gpus": [{"name", "vendor", "cc": [8, 9], "total_gib", "free_gib", "bandwidth", "driver", "unified",
    /// "cublas"}]} ("cublas": whether it has NVIDIA's cuBLAS, false where it does not say).
    pub fn from_json(text: &str) -> Result<Machine, String> {
        let v = json::parse(text)?;
        let gpus = v.get("gpus").and_then(Value::as_array).ok_or("no \"gpus\" array")?;
        let mut devices = Vec::new();
        for g in gpus {
            let name = g.get("name").and_then(Value::as_str).ok_or("a GPU without a name")?;
            let vendor = match g.get("vendor").and_then(Value::as_str).unwrap_or("nvidia") {
                "amd" => Vendor::Amd,
                "apple" => Vendor::Apple,
                _ => Vendor::Nvidia,
            };
            let cc = g.get("cc").and_then(Value::as_u64s).filter(|c| c.len() == 2).map_or((8, 0), |c| (c[0] as i32, c[1] as i32));
            let total = num(g, "total_gib").ok_or("a GPU without total_gib")?;
            let mut d = gpu(name, vendor, cc, total, num(g, "free_gib").unwrap_or(total * 0.95), num(g, "driver").unwrap_or(13000.0) as i32);
            if let Some(b) = num(g, "bandwidth") {
                d.bandwidth = b;
            }
            d.stream = num(g, "stream_gbs");
            if let Some(t) = num(g, "tflops") {
                d.tflops = t;
            }
            if let Some(u) = g.get("unified").and_then(Value::as_bool) {
                d.unified = u;
            }
            if let Some(b) = g.get("cublas").and_then(Value::as_bool) {
                d.cublas = Some(b);
            }
            devices.push(d);
        }
        Ok(Machine {
            name: v.get("name").and_then(Value::as_str).unwrap_or("machine").into(),
            devices,
            ram_free: (num(&v, "ram_free_gib").unwrap_or(16.0) * GIB) as u64,
            ram_bandwidth: num(&v, "ram_bandwidth").unwrap_or(50.0),
            cores: num(&v, "cores").unwrap_or(8.0) as usize,
            arch: v.get("arch").and_then(Value::as_str).unwrap_or("x86_64").into(),
        })
    }

    /// This computer: its GPUs as the driver reports them, its free memory, the rates a GPU's first use here measured (`rates_in`, kept in the cache; not where
    /// `GLYD_NO_PROBE` is set), and what the runs here reached against their predictions (`observed_on`). The thread's current context is as it was after
    /// (`dev::keeping_context`): a model loaded on one GPU of several runs on after a look at all of them.
    pub fn probe() -> Machine {
        crate::dev::keeping_context(Machine::probe_now)
    }

    fn probe_now() -> Machine {
        let vendor = if cfg!(glyd_hip) { Vendor::Amd } else { Vendor::Nvidia };
        let root = crate::store::cache_root();
        let mut ram_bandwidth = None;
        let devices = crate::dev::Dev::probe_all()
            .map(|gs| {
                gs.into_iter()
                    .map(|g| {
                        let mut d = Device::listed(g.name, vendor, g.cc, g.total, g.free, g.driver);
                        if std::env::var_os("GLYD_NO_PROBE").is_none() {
                            let measure = || {
                                crate::costs::probe_rates(g.ordinal).ok().map(|r| Probed {
                                    stream_gbs: r.vram / GB,
                                    link_gbs: r.link / GB,
                                    ram_gbs: r.ram / GB,
                                })
                            };
                            if let Some(r) = rates_in(&root, &d.name, g.ordinal, &measure) {
                                d.stream = Some(r.stream_gbs);
                                ram_bandwidth.get_or_insert(r.ram_gbs);
                            }
                        }
                        (d.observed, d.observed_prefill) = observed_on(&root, &d.name);
                        d
                    })
                    .collect()
            })
            .unwrap_or_default();
        let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
        Machine {
            name: "this computer".into(),
            devices,
            ram_free: crate::offload::host_memory().unwrap_or(8 << 30),
            ram_bandwidth: ram_bandwidth.unwrap_or(50.0),
            cores,
            arch: std::env::consts::ARCH.into(),
        }
    }
}

fn num(v: &Value, k: &str) -> Option<f64> {
    match v.get(k)? {
        Value::Number(s) => s.parse().ok(),
        _ => None,
    }
}

// ---- the model --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// A model's parameters by what a decode step reads of them, its KV cache and state per sequence, from its config.json.
#[derive(Clone, Debug, Default)]
pub struct Shape {
    pub arch: String,
    pub layers: usize,
    pub hidden: usize,
    pub max_ctx: usize,
    /// Parameters: the embedding (a step reads one row), the head (read whole; the embedding's where tied), every dense
    /// matrix (attention, dense MLPs, routers, shared experts) and the routed experts.
    pub embed: f64,
    pub head: f64,
    pub dense: f64,
    pub routed: f64,
    pub experts: usize,
    pub topk: usize,
    pub moe_layers: usize,
    /// KV bytes a token adds (bf16) in layers that keep every position, and in sliding-window layers with their window.
    pub kv_full: f64,
    pub kv_sliding: f64,
    /// The same in the executor's 12-bit lossless pages: a ring that takes them (src/dynamic/kv12.rs `shapes`) at the planner's bytes a token there (`sizing::kv12_token_bytes`),
    /// the others as bf16 (`in_twelve`); and the rings that take them, for what their pools spend once at a context (`kv12::once_bytes`).
    pub kv12_full: f64,
    pub kv12_sliding: f64,
    pub kv12_rings: Vec<crate::dynamic::kv12::KvShape>,
    /// Whether the keys and values are in the 12-bit pages (`in_twelve`): `kv_full` and `kv_sliding` are then `kv12_full`'s and `kv12_sliding`'s, and the
    /// plan counts `kv12_rings`' once-only bytes.
    pub twelve: bool,
    pub window: usize,
    /// Recurrent state per sequence (linear attention, Mamba), bytes.
    pub state: f64,
    /// Bits a weight of the release has (16 for bf16; 8 for FP8; about 4.25 for FP4 releases): every part's but the routed experts'.
    pub release_bits: f64,
    /// Bits a routed expert's weight has as released, where it differs from the rest (Kimi-K3's MXFP4, DeepSeek-V4's FP4 experts:
    /// 4.25); 0: the same as `release_bits`.
    pub expert_release_bits: f64,
    /// The width the routed experts read (the hidden size, or the latent of a latent mixture of experts).
    pub expert_width: usize,
    /// A draft head (multi-token prediction) the engine drafts with.
    pub mtp: bool,
    /// The parameters of its tables read by rows besides the embedding (an n-gram block's: Qwen3.8-Flash-Next's n-gram tables, DeepSeek-V4.1's Engram ones,
    /// `tables::ngram_names`), counted in `dense`: a token reads a few rows of each, so they are not among the parameters a token reads (`active`).
    pub tables: f64,
}

/// The largest prompt chunk a loader takes (the plain dense kind's with cuBLAS): the general plan's chunk for what the 12-bit pages spend once (`Shape::kv_once`).
const ONCE_CHUNK: usize = 4096;

/// The most layers a config.json may say (the deepest models have a few hundred): every family's parser refuses more, for a layer is an entry of a list it makes.
pub const MAX_LAYERS: usize = 10_000;

impl Shape {
    pub fn params(&self) -> f64 {
        self.embed + self.head + self.dense + self.routed
    }

    pub fn is_moe(&self) -> bool {
        self.experts > 0
    }

    /// Bits a routed expert's weight has as released.
    pub fn expert_bits_released(&self) -> f64 {
        if self.expert_release_bits > 0.0 {
            self.expert_release_bits
        } else {
            self.release_bits
        }
    }

    /// The release's bytes: the routed experts at their bits, the rest at theirs.
    pub fn release_bytes(&self) -> f64 {
        (self.embed + self.head + self.dense) * self.release_bits / 8.0 + self.routed * self.expert_bits_released() / 8.0
    }

    /// The parameters a token reads: every one but the routed experts a token does not choose and the tables read by rows (the embedding counted whole, as
    /// every model's figure has it).
    pub fn active(&self) -> f64 {
        (self.params() - self.routed + self.routed_active() - self.tables).max(0.0)
    }

    /// The routed experts' parameters a token reads (top-k of them in each MoE layer).
    pub fn routed_active(&self) -> f64 {
        if self.experts == 0 {
            0.0
        } else {
            self.routed * self.topk as f64 / self.experts as f64
        }
    }

    /// From config.json's text (the text model's keys where it has a `text_config`).
    pub fn from_config(text: &str) -> Result<Shape, String> {
        Shape::from_model(&ModelConfig::parse(text).map_err(|e| e.msg)?)
    }

    /// From a config.json parsed once: its shared keys, typed (`runtime::config::Shared`), the families' names for one thing read as one.
    /// The shape with its keys and values in the executor's 12-bit lossless pages (`kv12_full`, `kv12_sliding`), as a run that takes them holds them (`takes_twelve`).
    pub fn in_twelve(mut self) -> Shape {
        (self.kv_full, self.kv_sliding, self.twelve) = (self.kv12_full, self.kv12_sliding, true);
        self
    }

    /// What the keys and values take once whatever the users at `ctx` tokens of context (`kv12::once_bytes` of the 12-bit pages' rings, with the largest
    /// prompt chunk a loader takes): nothing in bf16.
    pub fn kv_once(&self, ctx: usize) -> f64 {
        if self.twelve {
            crate::dynamic::kv12::once_bytes(&self.kv12_rings, ctx, ONCE_CHUNK) as f64
        } else {
            0.0
        }
    }

    pub fn from_model(c: &ModelConfig) -> Result<Shape, String> {
        let s = &c.shared;
        let f = |x: Option<usize>| x.map(|v| v as f64);
        let model_type = s.model_type.clone().unwrap_or_default();
        let (h, layers, heads) = (s.hidden_size.unwrap_or(0), s.num_hidden_layers.unwrap_or(0), s.num_attention_heads.unwrap_or(0));
        if h == 0 || layers == 0 || heads == 0 {
            return Err("config.json has no hidden_size, num_hidden_layers or num_attention_heads".into());
        }
        // (a layer takes a list entry below: a number of them no model has is a file that is not one)
        if layers > MAX_LAYERS {
            return Err(format!("config.json: num_hidden_layers {layers}, where no model has more than {MAX_LAYERS}"));
        }
        let (hf, kv_heads) = (h as f64, f(s.num_key_value_heads).unwrap_or(heads as f64));
        let hd = f(s.head_dim).filter(|&d| d > 0.0).unwrap_or(hf / heads as f64);
        let vocab = f(s.vocab_size).unwrap_or(32000.0);
        let tied = s.tie_word_embeddings.unwrap_or(false);
        // the layers' kinds: layer_types, Nemotron-H's pattern, else every layer full attention
        let pattern = s.hybrid_override_pattern.clone();
        let kinds: Vec<String> = if let Some(p) = &pattern {
            p.chars()
                .map(|c| {
                    match c {
                        'M' => "mamba",
                        'E' => "moe",
                        '*' => "full_attention",
                        _ => "mlp",
                    }
                    .to_string()
                })
                .collect()
        } else if let Some(a) = &s.layer_types {
            a.clone()
        } else {
            vec!["full_attention".to_string(); layers]
        };
        let (experts, topk) = (s.experts(), s.top_k());
        let moe_inter = f(s.moe_intermediate_size).unwrap_or(0.0);
        let shared_inter =
            f(s.shared_expert_intermediate_size).or(f(s.moe_shared_expert_intermediate_size)).unwrap_or(moe_inter * f(s.n_shared_experts).unwrap_or(0.0));
        // (an MLP of relu² is two matrices, up and down, as Nemotron-H's: the activation its config names; every other one gated, three)
        let act = ["mlp_hidden_act", "hidden_act"].iter().find_map(|k| c.text.get(*k).and_then(serde_json::Value::as_str));
        let gated = if act == Some("relu2") { 2.0 } else { 3.0 };
        // (a latent mixture of experts: the experts work on a narrower vector, projected down and up around them)
        let latent = f(s.latent()).unwrap_or(0.0);
        let expert_width = if latent > 0.0 { latent } else { hf };
        let first_dense = s.first_k_dense_replace.unwrap_or(0);
        // attention
        let (q_lora, kv_lora) = (f(s.q_lora_rank).unwrap_or(0.0), f(s.kv_lora_rank).unwrap_or(0.0));
        let (nope, rope, v_hd) = (f(s.qk_nope_head_dim).unwrap_or(128.0), f(s.qk_rope_head_dim).unwrap_or(64.0), f(s.v_head_dim).unwrap_or(128.0));
        // (a latent attention's indexer keeps its normed key and its gate score a token beside the latent: src/dynamic/mla.rs)
        let indexer = if kv_lora > 0.0 { f(s.index_head_dim).unwrap_or(0.0) } else { 0.0 };
        let gate = if s.attn_output_gate.unwrap_or(false) { 2.0 } else { 1.0 };
        let hq = heads as f64;
        let (g_kv, g_hd) = (f(s.num_global_key_value_heads).unwrap_or(kv_heads), f(s.global_head_dim).unwrap_or(hd));
        // (the 12-bit pages take a latent's one stream, and keys and values of heads of a multiple of 128 channels with no softcap of the scores; a latent attention's indexer stays bf16)
        let softcap = c.text.get("attn_logit_softcapping").and_then(serde_json::Value::as_f64).is_some_and(|v| v != 0.0);
        // (a layer's ring as the 12-bit pages take it, `window` its sliding window or 0, and its bytes a token there (the indexer's beside a latent, bf16); a ring the
        // pages do not take at its bf16 bytes)
        let twelve = |global: bool, window: usize| -> (f64, Option<crate::dynamic::kv12::KvShape>) {
            let ring = if kv_lora > 0.0 {
                Some(crate::dynamic::kv12::latent_shape((kv_lora + rope) as usize))
            } else {
                let (kv, d) = if global { (g_kv, g_hd) } else { (kv_heads, hd) };
                crate::dynamic::kv12::ring_shape(kv as usize, d as usize, heads, window, softcap)
            };
            let (kv, d) = if global { (g_kv, g_hd) } else { (kv_heads, hd) };
            match ring {
                Some(r) => (sizing::kv12_token_bytes(r.heads, r.channels, r.one) as f64 + indexer * 4.0, Some(r)),
                None => (2.0 * kv * d * 2.0, None),
            }
        };
        let attention = |global: bool| -> (f64, f64) {
            if kv_lora > 0.0 {
                let q = if q_lora > 0.0 { hf * q_lora + q_lora * hq * (nope + rope) } else { hf * hq * (nope + rope) };
                (q + hf * (kv_lora + rope) + kv_lora * hq * (nope + v_hd) + hq * v_hd * hf, (kv_lora + rope + 2.0 * indexer) * 2.0)
            } else {
                let (kv, d) = if global { (g_kv, g_hd) } else { (kv_heads, hd) };
                (hf * hq * d * gate + 2.0 * hf * kv * d + hq * d * hf, 2.0 * kv * d * 2.0)
            }
        };
        let (lk_heads, lk_dim, lv_heads) = (f(s.linear_num_key_heads).unwrap_or(16.0), f(s.linear_key_head_dim), f(s.linear_num_value_heads).unwrap_or(32.0));
        let (lk, lv) = (lk_heads * lk_dim.unwrap_or(128.0), lv_heads * f(s.linear_value_head_dim).or(lk_dim).unwrap_or(128.0));
        let (m_heads, m_hd, m_state, m_groups) =
            (f(s.mamba_num_heads).unwrap_or(0.0), f(s.mamba_head_dim).unwrap_or(64.0), f(s.ssm_state_size).unwrap_or(128.0), f(s.n_groups).unwrap_or(8.0));
        let arch = if c.arch.is_empty() { model_type.clone() } else { c.arch.clone() };
        let mut sh = Shape { arch, layers: kinds.len(), hidden: h, experts, topk, ..Default::default() };
        sh.window = s.sliding_window.unwrap_or(0);
        // (compressed attention, DeepSeek-V4's (src/dynamic/deepseek_v4.rs): a layer keeps a window of keys that are its values, and where its ratio r is not 0 one compressed
        // entry of every r tokens, its indexer's entry of every 4 beside it where r is 4)
        let ratios: Option<Vec<usize>> =
            c.text.get("compress_ratios").and_then(serde_json::Value::as_array).map(|a| a.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect());
        for (i, kind) in kinds.iter().enumerate() {
            match kind.as_str() {
                "linear_attention" => {
                    sh.dense += hf * 2.0 * lk + hf * 2.0 * lv + lv * hf;
                    sh.state += lv_heads * lk_dim.unwrap_or(128.0) * f(s.linear_value_head_dim).unwrap_or(128.0) * 4.0;
                }
                "mamba" => {
                    let inner = m_heads * m_hd;
                    sh.dense += hf * (2.0 * inner + 2.0 * m_groups * m_state + m_heads) + inner * hf;
                    sh.state += m_heads * m_hd * m_state * 4.0;
                }
                k if k.contains("attention") => {
                    let sliding = k.contains("sliding") && sh.window > 0;
                    let global = !sliding && pattern.is_none() && s.global_head_dim.is_some();
                    let (p, kv) = attention(global);
                    sh.dense += p;
                    if let Some(r) = &ratios {
                        // (no 12-bit pages for these rings: the same bytes)
                        let (ratio, d, idim) = (r.get(i).copied().unwrap_or(0), hd as usize, s.index_head_dim.unwrap_or(0));
                        let entries = |w: usize, every: usize| (w * 2).div_ceil(every) as f64;
                        let (mut full, slid) = (0.0, (d * 2) as f64);
                        if ratio > 0 {
                            full += entries(d, ratio);
                        }
                        if ratio == 4 {
                            full += entries(idim, 4);
                        }
                        (sh.kv_full, sh.kv_sliding, sh.kv12_full, sh.kv12_sliding) =
                            (sh.kv_full + full, sh.kv_sliding + slid, sh.kv12_full + full, sh.kv12_sliding + slid);
                    } else {
                        let (b, ring) = twelve(global, if sliding { sh.window } else { 0 });
                        sh.kv12_rings.extend(ring);
                        if sliding {
                            (sh.kv_sliding, sh.kv12_sliding) = (sh.kv_sliding + kv, sh.kv12_sliding + b);
                        } else {
                            (sh.kv_full, sh.kv12_full) = (sh.kv_full + kv, sh.kv12_full + b);
                        }
                    }
                }
                _ => {}
            }
            // the MLP of the layer: Nemotron-H's 'E' a mixture of experts and '-' a dense MLP (its 'M' and '*' none);
            // elsewhere a mixture of experts past first_k_dense_replace where the model has experts, else a dense MLP
            let moe = experts > 0 && if pattern.is_some() { kind == "moe" } else { i >= first_dense };
            let mlp = if pattern.is_some() { kind == "mlp" } else { !moe };
            if moe {
                sh.moe_layers += 1;
                sh.routed += experts as f64 * gated * expert_width * moe_inter;
                sh.dense += gated * hf * shared_inter + hf * experts as f64 + if latent > 0.0 { 2.0 * hf * latent } else { 0.0 };
            } else if mlp {
                let inter = match s.intermediate_size {
                    Some(Width::One(w)) => w as f64,
                    _ => 4.0 * hf,
                };
                sh.dense += gated * hf * inter;
            }
        }
        sh.embed = vocab * hf;
        sh.head = if tied { 0.0 } else { vocab * hf };
        sh.max_ctx = s.max_position_embeddings.unwrap_or(0).max(4096);
        // (the release's quantization: the config's, or its text model's, where Kimi-K3 keeps it)
        let qc = s.quantization_config.as_ref();
        let quant = qc.and_then(|q| q.quant_method.as_deref()).unwrap_or("");
        sh.release_bits = match quant {
            "fp8" => 8.0,
            "mxfp4" | "nvfp4" => 4.25,
            _ => 16.0,
        };
        // (FP4 experts beside the rest: DeepSeek-V4's `expert_dtype` (at the top of the config, or V4.1's in its quantization_config), and compressed-tensors'
        // MXFP4 or NVFP4 groups, which Kimi-K3 puts on its routed experts alone and the rest left bf16)
        let format = qc.and_then(|q| q.format.as_deref()).unwrap_or("");
        let expert_dtype = s.expert_dtype.as_deref().or(qc.and_then(|q| q.expert_dtype.as_deref()));
        if expert_dtype == Some("fp4") || (quant == "compressed-tensors" && format.contains("fp4")) {
            sh.expert_release_bits = 4.25;
        }
        sh.expert_width = expert_width as usize;
        sh.mtp = s.mtp_layers() > 0;
        Ok(sh)
    }

    /// The same model with its parameters counted from the checkpoint's own tensors (src/sizes.rs: the headers of its files, a directory's or the Hub's by ranges) in
    /// place of an estimate from config.json: each part by its tensors' weights (a packed type's at the bits the release has that part's weights in: the routed
    /// experts' own where they differ), a vision tower left out, and the multi-token-prediction head counted only where the model drafts with it. A table with no
    /// weights of the parts changes nothing.
    pub fn with_table(mut self, t: &crate::sizes::Table) -> Shape {
        use crate::sizes::Part;
        let p = t.params_by_part(self.layers, self.release_bits.max(1.0), self.expert_bits_released().max(1.0));
        let get = |k: Part| p.get(&k).copied().unwrap_or(0.0);
        let dense = get(Part::Attention) + get(Part::Mlp) + get(Part::SharedExperts) + get(Part::Other) + if self.mtp { get(Part::Mtp) } else { 0.0 };
        if get(Part::Embedding) + get(Part::Head) + dense + get(Part::RoutedExperts) <= 0.0 {
            return self;
        }
        (self.embed, self.head, self.dense, self.routed) = (get(Part::Embedding), get(Part::Head), dense, get(Part::RoutedExperts));
        self
    }

    /// The parameters of the tables read by rows among `t`'s tensors (`names`: `tables::ngram_names` of its config; a table at 4 bits stands for its bf16
    /// name), as `with_table` counts them (`tables`).
    pub fn with_tables(mut self, t: &crate::sizes::Table, names: &[String]) -> Shape {
        let names: std::collections::HashSet<&str> = names.iter().map(String::as_str).collect();
        let rows: Vec<crate::sizes::Row> = t.rows.iter().filter(|r| names.contains(crate::formats::stands_for(&r.name).as_ref())).cloned().collect();
        let p = crate::sizes::Table { rows }.params_by_part(self.layers, self.release_bits.max(1.0), self.expert_bits_released().max(1.0));
        self.tables = p.values().sum();
        self
    }

    /// The same model with its parameters scaled to the checkpoint's bytes (model.safetensors.index.json's total_size), where known.
    pub fn scaled_to(mut self, total_bytes: f64) -> Shape {
        let ours = self.release_bytes();
        if total_bytes > 0.0 && ours > 0.0 {
            let k = total_bytes / ours;
            for p in [&mut self.embed, &mut self.head, &mut self.dense, &mut self.routed] {
                *p *= k;
            }
        }
        self
    }
}

// ---- the plan ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// What the planner may choose from: a level (None: any), a context (None: the largest that fits, to `TARGET_CTX`), users.
#[derive(Clone, Copy, Debug, Default)]
pub struct Ask {
    pub level: Option<&'static str>,
    pub context: Option<usize>,
    pub users: usize,
    /// The model's family as `run` treats it (src/support.rs): its levels, where its weights may go, its context. None:
    /// any level and placement (the planner's own view, as for a model no path runs yet).
    pub family: Option<&'static crate::support::Support>,
    /// The checkpoint is quantization-aware (`levels::is_qat`): its wren is exact (a pre-packed repo of the release's own four-bit weights).
    pub qat: bool,
    /// The positions a decode step reads in the prediction (0: `POSITIONS`); a development build's `GLYD_FIT_POSITIONS` sets it for `fit`, to predict a longer
    /// conversation (the held-out check of the predictions, scripts/planner_heldout_check.sh).
    pub positions: usize,
}

impl Ask {
    fn positions(&self) -> f64 {
        if self.positions > 0 {
            self.positions as f64
        } else {
            POSITIONS
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Placement {
    /// Every weight on one GPU (its index in the machine's list).
    Gpu(usize),
    /// Layers split over these GPUs, in proportion to their room.
    Split(Vec<usize>),
    /// A mixture of experts' experts in the computer's memory, a cache of them on the GPU; `hit`: the share of expert reads the cache serves.
    /// `disk`: the bytes of experts the computer's memory does not hold, read from the packed directory as the router asks for them (0: none).
    ExpertsInRam { gpu: usize, cache: u64, hit: f64, disk: u64 },
}

/// One level and placement considered.
#[derive(Clone, Debug)]
pub struct Candidate {
    pub level: &'static str,
    pub bits: f64,
    pub weights: u64,
    pub placement: Option<Placement>,
    pub context: usize,
    pub kv: u64,
    /// Decode tokens/s predicted (one user, about 1,000 positions): the model's (`prior`) times what runs here reached against it (`Device::band`), and the
    /// band's p10 and p90; prompt tokens/s where every weight is on the GPUs and the GPU's tensor rate is known.
    pub tok_s: f64,
    pub prior: f64,
    pub range: (f64, f64),
    pub prefill: Option<f64>,
    /// What the speeds stand on: this model's runs here, this GPU's, its class's records or the other classes' (`Device::basis`).
    pub basis: String,
    /// Why it does not fit, where it does not.
    pub why: Option<String>,
    /// The plan `run` loads, where this path is sized by the loader's own numbers (`sizing::size`): the figures above are its. None: an
    /// estimate from the model's parameters (a path not yet sized by its loader).
    pub load: Option<LoadPlan>,
}

impl Candidate {
    /// The speeds of an option predicted `prior` tokens/s on `d` (refined by what was observed there), with `prefill`.
    pub fn speeds(prior: f64, d: &Device, prefill: Option<f64>) -> Candidate {
        let b = d.band();
        Candidate {
            level: "",
            bits: 0.0,
            weights: 0,
            placement: None,
            context: 0,
            kv: 0,
            tok_s: prior * b[1],
            prior,
            range: (prior * b[0], prior * b[2]),
            prefill: prefill.map(|p| p * d.observed_prefill.unwrap_or(1.0)),
            basis: d.basis(),
            why: None,
            load: None,
        }
    }

    /// The speeds as `fit --json` has them.
    fn speed_json(&self, v: &mut serde_json::Value) {
        let r = |x: f64| (x * 10.0).round() / 10.0;
        v["tokens_per_second"] = serde_json::json!(r(self.tok_s));
        v["tokens_per_second_p10"] = serde_json::json!(r(self.range.0));
        v["tokens_per_second_p90"] = serde_json::json!(r(self.range.1));
        v["prefill_tokens_per_second"] = self.prefill.map_or(serde_json::Value::Null, |p| serde_json::json!(p.round()));
        v["prediction_basis"] = serde_json::json!(self.basis);
    }

    /// The plan as `fit --json` prints it (`glyd-plan/1`): the loader's own numbers where `load` has them (`exact`), else the planner's estimate.
    pub fn json(&self, model: &str, m: &Machine) -> String {
        if let Some(l) = &self.load {
            let mut v: serde_json::Value = serde_json::from_str(&l.json(model)).unwrap_or_default();
            v["exact"] = true.into();
            self.speed_json(&mut v);
            return v.to_string();
        }
        let placement = match &self.placement {
            Some(Placement::Gpu(i)) => serde_json::json!({"kind": "gpu", "gpus": [i]}),
            Some(Placement::Split(g)) => serde_json::json!({"kind": "split", "gpus": g}),
            Some(Placement::ExpertsInRam { gpu, cache, hit, disk }) => {
                serde_json::json!({"kind": "experts_in_ram", "gpus": [gpu], "cache_bytes": cache, "hit_rate": hit, "disk_bytes": disk})
            }
            None => serde_json::Value::Null,
        };
        let gpu = m.devices.iter().map(|d| d.name.as_str()).collect::<Vec<_>>();
        let mut v = serde_json::json!({
            "schema": SCHEMA, "model": model, "exact": false, "level": self.level, "context": self.context,
            "gpus": gpu, "weights_bytes": self.weights, "kv_bytes": self.kv, "placement": placement,
        });
        self.speed_json(&mut v);
        v.to_string()
    }
}

#[derive(Clone, Debug)]
pub struct Plan {
    pub backend: String,
    pub chosen: Option<Candidate>,
    pub options: Vec<Candidate>,
    pub notes: Vec<String>,
}

/// The context a plan aims for when none is asked (the model's own limit where lower), and the least it takes.
pub const TARGET_CTX: usize = 32768;
pub const MIN_CTX: usize = 4096;
/// The context a plan keeps before it takes a lower level (research: a floor of 8-16K tokens avoids runs that cannot hold a conversation).
pub const PREFERRED_CTX: usize = 8192;
const CONTEXTS: [usize; 8] = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 1048576];
/// Positions a decode step reads on average in the predictions (a chat with a short prompt).
pub const POSITIONS: f64 = 1024.0;
/// What a kernel streaming the GPU's memory reads of the maker's figure: 697.35 of 736 GB/s, measured on an RTX 4080 SUPER (`glyd-engine rates`, 512 MiB, 10
/// passes); the rate of a preset or of a GPU the probe has not measured.
pub const STREAM_OF_SPEC: f64 = 0.9475;
/// Of the streamed rate, what decode's reads reach: the lossy levels' packs all of it, the lossless layout (decoded in the product) 0.72; and a decode step's time
/// besides its reads, 35 µs a layer (its kernels' launches and the gaps between them). Measured on an RTX 4080 SUPER at about 1,000 positions: gemma-4-12B at wren
/// 86.2 tokens/s and kestrel 63.3, Qwen3.5-0.8B at kestrel 561.9 and wren 657.9 (a 35 µs layer and the stream's rate predict them within 4%: fitted on them, in-sample); the lossless
/// figure from a dense model of 64 layers of 5,120 on an RTX A6000 (11.8 tokens/s, likewise fitted; not measured on a recent release yet).
const EFF_LOSSY: f64 = 1.0;
const EFF_LOSSLESS: f64 = 0.72;
const LAYER_S: f64 = 35e-6;
/// Of the computer's memory bandwidth, what the CPU's expert products reach; the time a mixture-of-experts layer with its experts in RAM adds for the GPU and host
/// to meet besides `LAYER_S` (calibrated on Condor's Qwen3.8-Flash-Next on an RTX 4080 SUPER, 0.20 ms a layer in all).
const EFF_CPU: f64 = 0.70;
const SYNC_PER_MOE_LAYER: f64 = 0.165e-3;
/// A prompt's tokens a second: 2 x the parameters a token multiplies over the GPU's tensor rate (`Device::tflops`) at `PROMPT_EFF` of it, and 1.11 µs a token a
/// layer besides (attention, norms, the recurrent layers' scans). Both from two prompts of 936 and 1,002 tokens on an RTX 4080 SUPER: gemma-4-12B at wren 3,570
/// tokens/s, Qwen3.5-0.8B at kestrel 27,073 (two constants from two runs: fitted, in-sample).
const PROMPT_EFF: f64 = 0.92;
const PROMPT_LAYER_S: f64 = 1.11e-6;
/// The same products' time said against the GPU's streamed memory rate, as the figures are fitted (`Consts::f`): the RTX 4080 SUPER's 104.5 TFLOPS at
/// `PROMPT_EFF` over its 697.35 GB/s. Across classes the engine's prompt products follow the memory rate more closely than the tensor rate a maker gives:
/// each class's prompt runs predicted from the other classes' records came to a mean 18-25% off said against the stream (RTX 4080 SUPER, A100 SXM4 40GB,
/// A10, with a 36-layer dense model's runs among them), 27-42% against the tensor rate.
const PROMPT_F: f64 = 697.35e9 / (PROMPT_EFF * 104.5e12);
/// The band a prediction is given in where no run on the GPU was observed (p10 and p90 against the prediction): the runs these figures are calibrated on (above, and
/// `predictions_match_what_was_measured`'s) came to 0.91-1.02 of their predictions on the GPUs they were measured on; another GPU's kernels may differ more.
pub const SPREAD: [f64; 3] = [0.90, 1.0, 1.10];

// ---- the calibration records: each GPU class's figures -------------------------------------------------------------------------------------------------------------------

/// The figures a decode step and a prompt token are predicted with, for a GPU class. A decode step of `rows` sequences takes: its weights' bytes, read once for
/// all of them, over the streamed rate times `w` (`wl` for the lossless layout, decoded in the product), every sequence's keys' and values' bytes over that rate
/// times `kv` (attention reads them at its own share of the rate), a floor of `layer_s` + `width_s` x the hidden width a layer (its kernels' launches and their
/// least times), and `row_s` a layer for each row past the first (a row's attention and the work of its own the weights do not share). A prompt token takes: its
/// products over the tensor rate times `f`, `prompt_layer_s` a layer, and its attention's reads (the keys and values of the positions before it) over the
/// streamed rate times `attn`. (`w`, `wl`, `kv`, `f`, `attn`: the inverse of the share of the rate each reaches.)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Consts {
    pub w: f64,
    pub wl: f64,
    pub kv: f64,
    pub layer_s: f64,
    pub width_s: f64,
    pub row_s: f64,
    /// the 12-bit lossless layout's weights (a layout and kernels of their own: a 36-layer dense model on the A100 and the A10 read a byte faster than its packs; no shipped
    /// record has it, so it is the packs' share until a recent release's penguin run is recorded)
    pub w12: f64,
    pub f: f64,
    pub prompt_layer_s: f64,
    pub attn: f64,
}

impl Consts {
    fn decode(&self) -> [f64; 7] {
        [self.w, self.wl, self.kv, self.layer_s, self.width_s, self.row_s, self.w12]
    }
    fn prompt(&self) -> [f64; 3] {
        [self.f, self.prompt_layer_s, self.attn]
    }

    /// A decode step's seconds from its row of features (`rows_of`'s).
    fn decode_s(&self, x: &[f64; 7]) -> f64 {
        self.decode().iter().zip(x).map(|(c, x)| c * x).sum()
    }

    /// The weights' share of the rate a step reads them at: the packs', the tiered lossless layout's or the 12-bit one's.
    fn w_of(&self, read: Read) -> f64 {
        match read {
            Read::Packs => self.w,
            Read::Tiered => self.wl,
            Read::Twelve => self.w12,
        }
    }

    /// A decode step's seconds: `weights` and `kv` bytes read at `rate` (bytes a second), `layers` of `hidden` width, `rows` sequences (their keys and values in
    /// `kv`).
    #[allow(clippy::too_many_arguments)]
    pub fn step_s(&self, rate: f64, weights: f64, kv: f64, layers: f64, hidden: f64, read: Read, rows: f64) -> f64 {
        (weights * self.w_of(read) + kv * self.kv) / rate + self.floor_s(layers, hidden, rows)
    }

    /// A decode step's time besides its reads: `layers` of `hidden` width, `rows` sequences.
    fn floor_s(&self, layers: f64, hidden: f64, rows: f64) -> f64 {
        layers * (self.layer_s + self.width_s * hidden + self.row_s * (rows - 1.0).max(0.0))
    }

    /// A prompt token's seconds: `flops` of products and `attn` bytes of attention's reads, each at its share of the streamed `rate`, and `layers`.
    pub fn token_s(&self, flops: f64, layers: f64, attn: f64, rate: f64) -> f64 {
        (flops * self.f + attn * self.attn) / rate + layers * self.prompt_layer_s
    }
}

/// The figures where no record says otherwise: the RTX 4080 SUPER's (sm89), fitted on its runs (above).
/// (No run of several rows on it: no time a row besides its reads.)
pub const DEFAULT_CONSTS: Consts = Consts {
    w: 1.0 / EFF_LOSSY,
    wl: 1.0 / EFF_LOSSLESS,
    kv: 1.0 / EFF_LOSSY,
    layer_s: LAYER_S,
    width_s: 0.0,
    row_s: 0.0,
    w12: 1.0 / EFF_LOSSY,
    f: PROMPT_F,
    prompt_layer_s: PROMPT_LAYER_S,
    attn: 0.0,
};
/// How far a figure may move from the one it starts from for each unit of evidence against it: the ridge of the fit (`fit_rows`), in the figure's own scale.
const RIDGE: f64 = 0.001;
/// The scale of each decode and prompt figure, for the ridge (a figure that starts at 0 is held near 0 on this scale).
const DECODE_SCALE: [f64; 7] = [1.0, 1.0, 4.0, 35e-6, 1e-8, 35e-6, 1.0];
const PROMPT_SCALE: [f64; 3] = [0.01, 1.0e-6, 10.0];

/// `glyd-engine plancheck`'s records (`glyd-calib/1`, a JSON object a line), the ones shipped: the runs measured on each GPU class so far. More are collected
/// from other machines' `plancheck` files and added here.
pub const CALIBRATION: &str = include_str!("../calibration/records.jsonl");

pub const CALIB_SCHEMA: &str = "glyd-calib/1";

/// A GPU's class: its compute capability (`sm89`: Ada; `sm80`: A100; `sm86`: Ampere's others; `sm90`: Hopper ...).
pub fn class_of(cc: (i32, i32)) -> String {
    format!("sm{}{}", cc.0, cc.1)
}

/// Whether a class's GPUs have HBM (the datacenter parts, compute capability 8.0, 9.0 and 10.0: A100, H100/H200, B200) or GDDR (the rest): a class with no
/// records of its own takes the classes of its kind first, whose memory and kernels behave alike.
fn hbm_class(class: &str) -> bool {
    let d = class.trim_start_matches("sm");
    let (major, minor) = d.split_at(d.len().saturating_sub(1));
    minor == "0" && major.parse::<u32>().is_ok_and(|m| (8..=10).contains(&m))
}

/// The records of a file's text (`CALIBRATION`'s form), the lines that parse and say their schema.
pub fn records_of(text: &str) -> Vec<serde_json::Value> {
    text.lines().filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok()).filter(|v| v["schema"] == CALIB_SCHEMA).collect()
}

/// The least squares of the rows' relative errors (each row's features against its seconds), held toward `prior` by `RIDGE` in `scale`, with no figure below 0
/// (a figure the fit would make negative is held at 0 and the rest fitted again).
fn fit_rows<const N: usize>(rows: &[([f64; N], f64)], prior: [f64; N], scale: [f64; N]) -> [f64; N] {
    let mut free = [true; N];
    loop {
        // (the normal equations of sum ((x.t - y) / y)^2 + RIDGE * rows * sum ((t - prior) / scale)^2, over the free figures)
        let n_rows = rows.len().max(1) as f64;
        let mut a = [[0.0f64; N]; N];
        let mut b = [0.0f64; N];
        for (x, y) in rows {
            let wy = 1.0 / (y * y);
            for i in 0..N {
                b[i] += x[i] * y * wy;
                for j in 0..N {
                    a[i][j] += x[i] * x[j] * wy;
                }
            }
        }
        for i in 0..N {
            let r = RIDGE * n_rows / (scale[i] * scale[i]);
            a[i][i] += r;
            b[i] += r * prior[i];
        }
        for i in 0..N {
            if !free[i] {
                a[i] = [0.0; N];
                for row in a.iter_mut() {
                    row[i] = 0.0;
                }
                a[i][i] = 1.0;
                b[i] = 0.0;
            }
        }
        // (Gaussian elimination with the largest pivot)
        let mut t = [0.0f64; N];
        let (mut m, mut v) = (a, b);
        for c in 0..N {
            let p = (c..N).max_by(|&i, &j| m[i][c].abs().total_cmp(&m[j][c].abs())).unwrap_or(c);
            m.swap(c, p);
            v.swap(c, p);
            if m[c][c].abs() < 1e-300 {
                return prior;
            }
            for r in c + 1..N {
                let k = m[r][c] / m[c][c];
                for j in c..N {
                    m[r][j] -= k * m[c][j];
                }
                v[r] -= k * v[c];
            }
        }
        for c in (0..N).rev() {
            t[c] = (v[c] - (c + 1..N).map(|j| m[c][j] * t[j]).sum::<f64>()) / m[c][c];
        }
        match (0..N).find(|&i| free[i] && t[i] < 0.0) {
            Some(i) => free[i] = false,
            None => return t,
        }
    }
}

/// The seconds of other users' prompts in the median user's decode, where `n` users send a prompt of `prompt` tokens at once (bench's loads): the scheduler reads
/// the prompts in turn, and each user's reply runs beside the prompts read after its own, (n - 1) / 2 of them for the median one. Measured on an A100:
/// Qwen3.5-0.8B's 8 users took 154.5 tokens/s each on short prompts and 127.3 on 1k ones, where 127 steps of the first and 3.5 prompts of 994 tokens at
/// 20,206 tokens/s come to 127.8.
pub fn prompt_wait(n: usize, prompt: f64, prefill: f64) -> f64 {
    if n <= 1 || prefill <= 0.0 {
        return 0.0;
    }
    (n - 1) as f64 / 2.0 * prompt / prefill
}

/// The decode and prompt rows of records (features against seconds): a decode step for the runs whose weights were all on the GPU (a drafting run's tokens a step
/// divided out; the weights' bytes the lossless layout's or the packs', and the rows past the first a layer), a prompt token for the runs of one user.
#[allow(clippy::type_complexity)]
fn rows_of(records: &[&serde_json::Value]) -> (Vec<([f64; 7], f64)>, Vec<([f64; 3], f64)>) {
    let (mut dec, mut pre) = (Vec::new(), Vec::new());
    for r in records {
        let f = |k: &str| r["features"][k].as_f64();
        let (Some(rate), Some(layers)) = (r["stream_gbs"].as_f64().map(|g| g * GB), f("layers")) else { continue };
        if r["features"]["in_vram"] != true {
            continue;
        }
        let hidden = f("hidden").unwrap_or(0.0);
        if let (Some(w), Some(kv), Some(tok_s)) = (f("weight_bytes"), f("kv_bytes"), r["measured"]["decode"].as_f64()) {
            let (w, wl, w12) = match (r["features"]["lossless"] == true, r["features"]["twelve"] == true) {
                (true, true) => (0.0, 0.0, w),
                (true, false) => (0.0, w, 0.0),
                _ => (w, 0.0, 0.0),
            };
            let users = r["users"].as_u64().unwrap_or(1) as usize;
            // (a step's time: where the users sent their prompts at once, a reply's seconds less the other users' prompts it waited for, over its steps)
            let made = r["made"].as_f64().map(|m| m - 1.0).filter(|m| *m > 0.0);
            let wait = match (made, r["measured"]["prompt_tokens"].as_f64(), r["measured"]["prefill"].as_f64()) {
                (Some(m), Some(p), Some(pf)) => prompt_wait(users, p, pf) / m,
                _ => 0.0,
            };
            let step = 1.0 / tok_s - wait;
            if tok_s > 0.0 && step > 0.0 {
                let more = (users.max(1) - 1) as f64 * layers;
                dec.push(([w / rate, wl / rate, kv / rate, layers, layers * hidden, more, w12 / rate], f("gain").unwrap_or(1.0) * step));
            }
        }
        if let (Some(flops), Some(tok_s)) = (f("prompt_flops"), r["measured"]["prefill"].as_f64()) {
            if tok_s > 0.0 && r["users"].as_u64() == Some(1) {
                pre.push(([flops / rate, layers, f("attn_bytes").unwrap_or(0.0) / rate], 1.0 / tok_s));
            }
        }
    }
    (dec, pre)
}

/// The figures `records` fit, starting from `prior`. Records with no run of a lossless layout keep its share in the prior's ratio to the packs': the layout
/// moves with the class's reads, not apart from them.
pub fn fit_consts(records: &[&serde_json::Value], prior: Consts) -> Consts {
    let (dec, pre) = rows_of(records);
    let mut d = if dec.is_empty() { prior.decode() } else { fit_rows(&dec, prior.decode(), DECODE_SCALE) };
    let p0 = prior.decode();
    for i in [1, 6] {
        if !dec.iter().any(|(x, _)| x[i] > 0.0) {
            d[i] = d[0] * p0[i] / p0[0];
        }
    }
    let p = if pre.is_empty() { prior.prompt() } else { fit_rows(&pre, prior.prompt(), PROMPT_SCALE) };
    Consts { w: d[0], wl: d[1], kv: d[2], layer_s: d[3], width_s: d[4], row_s: d[5], w12: d[6], f: p[0], prompt_layer_s: p[1], attn: p[2] }
}

/// Where a class's figures come from: its own records (a fitted point), the other classes' (an extrapolation), or none (the defaults).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Source {
    Own,
    Others,
    Default,
}

/// A class's figures, where they come from and on how many records, and the band their predictions are given in: what records the figures were not fitted on
/// came to against them (measured over predicted: p10, p50, p90).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prior {
    pub consts: Consts,
    pub source: Source,
    pub records: usize,
    pub band: [f64; 3],
}

/// Measured over predicted, decode, for records predicted from `c`.
fn ratios(records: &[&serde_json::Value], c: Consts) -> Vec<f64> {
    let (dec, _) = rows_of(records);
    dec.iter().map(|(x, y)| c.decode_s(x) / y).collect()
}

/// p10, p50 and p90 of ratios (`SPREAD` where there are fewer than three).
fn quantiles(mut x: Vec<f64>) -> [f64; 3] {
    x.retain(|v| v.is_finite() && *v > 0.0);
    if x.len() < 3 {
        return SPREAD;
    }
    // (fewer than eight: at least `SPREAD`'s width around their middle)
    if x.len() < 8 {
        let q = quantiles([x.clone(), x.clone(), x.clone()].concat());
        return [q[0].min(q[1] * SPREAD[0]), q[1], q[2].max(q[1] * SPREAD[2])];
    }
    x.sort_by(f64::total_cmp);
    let q = |f: f64| {
        let at = f * (x.len() - 1) as f64;
        let (lo, hi) = (at.floor() as usize, at.ceil() as usize);
        x[lo] + (x[hi] - x[lo]) * (at - lo as f64)
    };
    [q(0.1), q(0.5), q(0.9)]
}

/// The figures of class `class` from `records`. With records of its own, they are fitted on them, starting from the other classes' fit (a fitted point), and
/// their band is what each of its records came to against the figures fitted without it. Without, they are the other classes' fit (from the defaults), and their
/// band is how every class with records fared predicted from the classes other than it (leave-one-class-out): the error an unseen class may expect.
pub fn prior_from(records: &[serde_json::Value], class: &str) -> Prior {
    let of = |k: &str| records.iter().filter(|r| r["class"].as_str() == Some(k)).collect::<Vec<_>>();
    let others = others_of(records, class);
    let base = base_of(records, class);
    let own = of(class);
    if own.is_empty() {
        // (leave-one-class-out over the classes that have records)
        let mut classes: Vec<&str> = records.iter().filter_map(|r| r["class"].as_str()).collect();
        classes.sort();
        classes.dedup();
        let mut x = Vec::new();
        for k in classes {
            if records.iter().any(|r| r["class"].as_str() != Some(k)) {
                x.extend(ratios(&of(k), base_of(records, k)));
            }
        }
        let source = if others.is_empty() { Source::Default } else { Source::Others };
        // (their spread, not their middle: the classes err in both directions, and which an unseen class takes is not known)
        let q = quantiles(x);
        return Prior { consts: base, source, records: others.len(), band: [q[0].min(1.0), 1.0, q[2].max(1.0)] };
    }
    let mut x = Vec::new();
    for i in 0..own.len() {
        let rest: Vec<&serde_json::Value> = own.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, r)| *r).collect();
        x.extend(ratios(&own[i..=i], fit_consts(&rest, base)));
    }
    Prior { consts: fit_consts(&own, base), source: Source::Own, records: own.len(), band: quantiles(x) }
}

/// The figures a class starts from before its own records: the decode figures the classes `others_of` gives it fit (from the defaults), and the prompt figures
/// every other class's records fit (a kin's few prompts leave its three figures loose: the A10's prompts from the RTX 4080 SUPER's four came out 120% off on
/// the mean, from every other class's 19%; its decode from the kin 17.6%, from every other class 10.8%, but the 4080 SUPER's from every other 23.9% against
/// 14.5% from its kin: decode keeps the kin. Those with a 36-layer dense model's runs among the records; without them, prompts 17.7%). Where none of the kin ran a step of several rows, a row's time is the one every other class's records fit.
fn base_of(records: &[serde_json::Value], class: &str) -> Consts {
    let kin = others_of(records, class);
    if kin.is_empty() {
        return DEFAULT_CONSTS;
    }
    let rest: Vec<&serde_json::Value> = records.iter().filter(|r| r["class"].as_str() != Some(class)).collect();
    let all = fit_consts(&rest, DEFAULT_CONSTS);
    let mut base = fit_consts(&kin, DEFAULT_CONSTS);
    (base.f, base.prompt_layer_s, base.attn) = (all.f, all.prompt_layer_s, all.attn);
    if !kin.iter().any(|r| r["users"].as_u64().is_some_and(|u| u > 1)) {
        base.row_s = all.row_s;
    }
    base
}

/// The records a class with none of its own is fitted on: the other classes of its kind of memory (`hbm_class`) where there are any, else every other class.
fn others_of<'a>(records: &'a [serde_json::Value], class: &str) -> Vec<&'a serde_json::Value> {
    let rest: Vec<&serde_json::Value> = records.iter().filter(|r| r["class"].as_str() != Some(class)).collect();
    let kin: Vec<&serde_json::Value> = rest.iter().copied().filter(|r| r["class"].as_str().is_some_and(|k| hbm_class(k) == hbm_class(class))).collect();
    if kin.is_empty() {
        rest
    } else {
        kin
    }
}

/// The figures a GPU of capability `cc` starts from (`prior_from` the shipped records), made once a class.
pub fn prior_for(cc: (i32, i32)) -> Prior {
    static PRIORS: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<String, Prior>>> = std::sync::OnceLock::new();
    static RECORDS: std::sync::OnceLock<Vec<serde_json::Value>> = std::sync::OnceLock::new();
    let class = class_of(cc);
    let records = RECORDS.get_or_init(|| records_of(CALIBRATION));
    let mut g = PRIORS.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner());
    *g.entry(class.clone()).or_insert_with(|| prior_from(records, &class))
}

/// What the records say of the predictions, class by class: each class's decode and prompt errors (predicted over measured, less one) from the figures the other
/// classes' records fit (leave-one-class-out: the error of a class with no record of its own), and from the figures fitted on its own records less the one
/// predicted (the error of a class that has records).
#[allow(clippy::type_complexity)]
pub fn score_classes(records: &[serde_json::Value]) -> Vec<(String, Vec<f64>, Vec<f64>, Vec<f64>)> {
    let mut classes: Vec<String> = records.iter().filter_map(|r| r["class"].as_str().map(str::to_string)).collect();
    classes.sort();
    classes.dedup();
    let err = |rs: &[&serde_json::Value], c: Consts| -> (Vec<f64>, Vec<f64>) {
        let (dec, pre) = rows_of(rs);
        let d = dec.iter().map(|(x, y)| y / c.decode_s(x) - 1.0).collect();
        let p = pre.iter().map(|(x, y)| y / (c.f * x[0] + c.prompt_layer_s * x[1] + c.attn * x[2]) - 1.0).collect();
        (d, p)
    };
    classes
        .into_iter()
        .map(|k| {
            let own: Vec<&serde_json::Value> = records.iter().filter(|r| r["class"].as_str() == Some(k.as_str())).collect();
            let base = base_of(records, &k);
            let (dec, pre) = err(&own, base);
            let mut loo = Vec::new();
            for i in 0..own.len() {
                let others: Vec<&serde_json::Value> = own.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, r)| *r).collect();
                loo.extend(err(&own[i..=i], fit_consts(&others, base)).0);
            }
            (k, dec, pre, loo)
        })
        .collect()
}

/// The share of expert reads a cache holding a share `f` of the experts serves at one row, f^HIT_EXPONENT: experts are used unevenly and a sequence uses its own again (Condor: 70-76% at about a quarter;
/// the routing record benchmarks/moe-skew-2026-10-09's decode traces' least-recently-used places: 61, 73 and 80% at 17, 32 and 46%, where this says 67, 77 and 84: offload.rs's test). A router that chooses uniformly (a random model's) is 1.
pub const HIT_EXPONENT: f64 = 0.23;
/// How fast the reuse of the places between steps goes as a step's rows' distinct experts fill them (`step_hit`), fitted to the same record's decode traces at 4 to 20 rows (offload.rs's test).
const ROWS_DECAY: f64 = 0.55;

/// The distinct experts `users` rows choose a step in a layer of `experts` choosing `topk` each, in one row's experts (k): E (1 - (1 - k/E)^U) / k, as a router choosing uniformly makes them, and at least one row's.
pub fn step_distinct(experts: usize, topk: usize, users: usize) -> f64 {
    if experts == 0 || topk == 0 || users <= 1 {
        return 1.0;
    }
    let (e, k) = (experts as f64, topk as f64);
    (e * (1.0 - (1.0 - k / e).powf(users.max(1) as f64)) / k).max(1.0)
}

/// The share of a step's distinct experts a layer's cache of `places` of its `experts` finds there, for `users` rows (`topk` each): f^s at one row (f the places' share, s `skew`: `HIT_EXPONENT` for a release's
/// router, 1 for a uniform one), the exponent going to 1 as the step's distinct experts D fill the places, s + (1 - s)(1 - exp(-ROWS_DECAY (D - k) / places)): what the places keep between steps is what the step's
/// other rows have not taken. A uniform router's share is f at any rows.
pub fn step_hit(experts: usize, topk: usize, places: usize, users: usize, skew: f64) -> f64 {
    let f = (places as f64 / experts.max(1) as f64).min(1.0);
    let fill = (step_distinct(experts, topk, users) - 1.0) * topk as f64 / places.max(1) as f64;
    f.powf(skew + (1.0 - skew) * (1.0 - (-ROWS_DECAY * fill).exp()))
}

/// A step's distinct missed experts of a layer in one row's experts (k), each read once for all its rows: `step_distinct` (1 - `step_hit`); 1 - f^s at one row.
pub fn step_misses(experts: usize, topk: usize, places: usize, users: usize, skew: f64) -> f64 {
    step_distinct(experts, topk, users) * (1.0 - step_hit(experts, topk, places, users, skew))
}
/// The rate an expert's bytes come from an NVMe disk in direct reads of whole experts, 16 in flight (src/disk.rs), GB/s: a planning figure.
pub const DISK_GBS: f64 = 3.0;

/// The rate a GPU reads pinned host memory over its link (PCIe 4.0 x16, pinned copies), GB/s: a planning figure for the dense layers in the computer's memory, which every step reads whole.
pub const LINK_GBS: f64 = 25.0;
/// Tokens a step makes with a draft head that proposes one and is accepted 80-90% of the time, after its cost.
const MTP_GAIN: f64 = 1.7;

// ---- the reserve: one model for every path --------------------------------------------------------------------------------------------------------------------------------------------------

/// What a GPU keeps free for itself: the driver's buffers and a margin for the memory other programs take between the plan and
/// the load (768 MiB). Every path sizes with `reserve_of` (`run`, `serve`, `preflight`, `fit`); what an engine takes besides, its
/// activations, workspaces and graphs, is its `work` (below, one constant a path, each measured). A path's reserve and work
/// together are what its loader needed before the planner drove it: nothing that ran then is refused now.
pub const CONTEXT_FLOOR: u64 = 768 << 20;

/// What a GPU under Windows' WDDM keeps free beyond `CONTEXT_FLOOR`: the display and the programs open on it take memory the driver gives them first, and an allocation of ours that no longer fits is
/// not refused but paged into system memory, which makes the run many times slower. A first figure, to be set from a measurement on a Windows GPU (the testing phase), not from one.
pub const WDDM_EXTRA: u64 = 1 << 30;

static WDDM: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Says that a GPU of this computer runs under WDDM (dev.rs `Dev::probe_all` sets it): every plan then keeps `WDDM_EXTRA` more free.
pub fn set_wddm(on: bool) {
    WDDM.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// `reserve_of` for a GPU that is, or is not, under WDDM.
pub fn reserve_with(_free: u64, wddm: bool) -> u64 {
    CONTEXT_FLOOR + if wddm { WDDM_EXTRA } else { 0 }
}

/// The reserve of a GPU with `free` bytes free (the same on every GPU: other programs do not take more of a larger one; a WDDM GPU keeps `WDDM_EXTRA` more).
pub fn reserve_of(free: u64) -> u64 {
    reserve_with(free, WDDM.load(std::sync::atomic::Ordering::Relaxed))
}

/// What a GPU has for the model's weights, its cache and the engine's `work`.
pub fn room_of(free: u64) -> u64 {
    free.saturating_sub(reserve_of(free))
}

/// The activations and workspaces of the lossy paths' engines (Qwen3.5, Qwen3.6, Qwen3.8, Nemotron-H, Gemma 4): 1.26 GiB measured for
/// Qwen3.8-27B with prompts in chunks of 2,048, about 1 GiB for Nemotron 3 Super by the sizes of its buffers (not yet measured).
pub const LOSSY_WORK: u64 = 1280 << 20;
/// The prompt chunk the lossy path of the Qwen3.5 and Nemotron-H layouts takes (`LoadPlan::chunk`).
pub const LOSSY_CHUNK: usize = 2048;
/// The Qwen3 path's fixed workspaces: the CUDA context and the library's own buffers, cuBLAS's and the pool's, 1,400 MiB (600 + 400 + 400) measured
/// as the path's total before the reserve was one model, less that reserve; a prompt chunk's activations are on top (`sizing::work`).
pub const QWEN3_WORK: u64 = (1400 << 20) - CONTEXT_FLOOR;
/// The dynamic route's work on each GPU: the activations, the scratch, the graphs and the KV cache of its context together, which this route has not
/// split apart yet (3 GiB and 5% of the GPU's free memory in all with the reserve, as it kept before the planner drove it). The rows a step's draws see
/// are a named part of it (`draw_rows_bytes`), which the engine makes with its head's rows.
pub fn dynamic_work_of(free: u64) -> u64 {
    (3 << 30) - CONTEXT_FLOOR + free / 20
}

/// The rows a step's draws see, made on the GPU where a request changes its row (src/kern/mod.rs `ProcGpu`: penalties, a bias, a grammar's mask put on the model's row), in
/// floats: 4 bytes a token of the vocabulary for each of the step's `rows` head rows (64 MiB at 64 rows of 262,144). Taken from the dynamic route's work (`dynamic_work_of`) when
/// the engine's head is made for its users (src/dynamic/mod.rs `set_head_rows`), not when a request first asks.
pub fn draw_rows_bytes(vocab: usize, rows: usize) -> u64 {
    4 * vocab as u64 * rows as u64
}

/// What a GPU has for the dynamic route's weights: its free memory less the reserve and the route's work.
pub fn dynamic_room_of(free: u64) -> u64 {
    room_of(free).saturating_sub(dynamic_work_of(free))
}

/// The context the lossy paths take where none is asked, if it fits, and the least a plan takes.
pub const LOSSY_CTX: usize = 32768;
pub const LOSSY_MIN_CTX: usize = 4096;
/// The computer's memory kept for the system and the engine's own besides the experts pinned in it.
pub const HOST_MARGIN: u64 = 6 << 30;
/// The swap area for sequences sent back for room (R4d), in times the GPU's keys, values and states it backs: SGLang's host cache is 2 times its device pool by default
/// (`hicache_ratio`), which lets every sequence that runs be swapped out once while as many run again.
pub const SWAP_SHARE: f64 = 2.0;

/// The bytes of the computer's memory the swap area may pin (R4d, docs/design/runtime.md section 5): SWAP_SHARE times the GPU's `cache` bytes (its pools of keys and values and
/// its sequences' states), and no more than the computer has free less HOST_MARGIN. `host` is the computer's free memory read after the expert store's host tier was
/// pinned (src/offload.rs `host_memory`), so what the experts hold is spent already: the area takes what they left. None: the computer does not say, and only the share
/// bounds it.
pub fn swap_area(cache: u64, host: Option<u64>) -> u64 {
    let share = (cache as f64 * SWAP_SHARE) as u64;
    host.map_or(share, |h| share.min(h.saturating_sub(HOST_MARGIN)))
}
/// GPU memory kept for a Gemma 4 draft model (an "-assistant" model at six bits, about 0.36 GiB for 31B, and its buffers).
pub const GEMMA_DRAFT_BYTES: u64 = 640 << 20;
/// A Gemma 4 prompt's chunk (rows a pass reads at once; a drafted round's k + 1 rows fit in it).
pub const GEMMA_CHUNK: usize = 512;

/// What a GPU has for the weights and the cache in the planner's estimates: its free memory less the reserve and the engine's work (the
/// lossy engines'; the Qwen3 path's is the same to a few hundred MiB).
fn room(d: &Device) -> u64 {
    room_of(d.free).saturating_sub(LOSSY_WORK)
}

/// Bits a routed expert's weight takes at the hummingbird level for experts of `width` (docs/design/hummingbird.md): the recipe hb40's
/// share at 3 bits and the rest at 2, a row of `width` codes with a bf16 scale a run of 32 and a codebook of its own (2^bits bf16 entries).
pub fn hummingbird_expert_bits(width: usize) -> f64 {
    let w = width.max(32) as f64;
    let at = |b: i32| b as f64 + 0.5 + (1u32 << b) as f64 * 16.0 / w;
    let share = crate::moe_hb::HB40_THREE_BIT_PERCENT as f64 / 100.0;
    share * at(3) + (1.0 - share) * at(2)
}

/// Bits a weight of each part takes at a level: (everything else, the routed experts). None: the level does not apply.
///
/// Hummingbird applies to a mixture of experts whose experts' bits as released are above its own (Kimi-K3's MXFP4 and DeepSeek-V4's FP4
/// at 4.25, GLM's FP8, bf16): Nemotron 3 Super's directories (src/hbpack.rs) hold every other matrix six-bit, and its kernels take experts
/// on a latent that is a power of two from 128 to 1,024 (src/moe_hb.rs: not Nemotron 3.5 Lightning, whose experts read its width of
/// 2,688); the families the description builder maps (src/hb_pack.rs) keep every other matrix as at penguin. `one_gpu_hb`: the model's row's hummingbird level
/// is the one-GPU route's (`Support::hummingbird_one_gpu`), the former.
fn level_bits(level: &str, sh: &Shape, d: &Device, one_gpu_hb: bool) -> Option<(f64, f64)> {
    let lossless = |release: f64| {
        if release <= 8.0 {
            release
        } else if matches!(d.cc, (8, 0) | (9, 0) | (10, 0)) {
            12.04 // (A100, H100, B200 take the 12-bit layout: it is the faster there)
        } else {
            10.75
        }
    };
    let (rest, experts) = (lossless(sh.release_bits), lossless(sh.expert_bits_released()));
    if level == "hummingbird" {
        let eb = hummingbird_expert_bits(sh.expert_width);
        if !sh.is_moe() || eb >= sh.expert_bits_released() {
            return None;
        }
        // (the one-GPU route's hummingbird, its row's: `Support::hummingbird_one_gpu`)
        if one_gpu_hb {
            let w = sh.expert_width;
            return (w.is_power_of_two() && (128..=1024).contains(&w) && 6.0 < sh.release_bits).then_some((6.0, 2.9));
        }
        return Some((rest, eb));
    }
    let b = match level {
        "penguin" => return Some((rest, experts)),
        "kestrel" => (6.5, 6.5),
        "swift" => (5.5, 5.5),
        "wren" => (4.5, 4.5),
        _ => return None,
    };
    // (a uniform level is below every part's bits as released, or it does not apply)
    (b.0 < sh.release_bits && b.1 < sh.expert_bits_released()).then_some(b)
}

/// The backend the machine's GPU `main` runs on, and what its generation or driver says against it.
fn backend_of(m: &Machine, main: usize) -> (String, Vec<String>) {
    let mut notes = Vec::new();
    let d0 = &m.devices[main];
    let backend = match d0.vendor {
        Vendor::Nvidia if d0.driver >= crate::machine::driver_min(d0.cc) => format!("CUDA (driver CUDA {}.{})", d0.driver / 1000, d0.driver % 1000 / 10),
        Vendor::Nvidia => {
            notes.push(format!(
                "the driver supports CUDA {}.{}: the engine's kernels need {}",
                d0.driver / 1000,
                d0.driver % 1000 / 10,
                crate::machine::driver_need(d0.cc)
            ));
            "CUDA (driver too old)".into()
        }
        Vendor::Amd => "ROCm/HIP (development build)".into(),
        Vendor::Apple => "Metal via MLX (planned, ROADMAP V3)".into(),
    };
    if d0.vendor == Vendor::Nvidia && d0.cc.0 < 8 {
        notes.push(format!("{} is older than Ampere: Turing and Volta support is ROADMAP V5", d0.name));
    }
    (backend, notes)
}

/// The plan for `sh` on `m`.
pub fn plan(sh: &Shape, m: &Machine, ask: Ask) -> Plan {
    // (the GPU with the most free memory; the first of equals)
    let Some(main) = (0..m.devices.len()).min_by_key(|&i| (std::cmp::Reverse(m.devices[i].free), i)) else {
        return Plan {
            backend: "none".into(),
            chosen: None,
            options: vec![],
            notes: vec!["no GPU found: the CPU backend is not built yet (ROADMAP V4)".into()],
        };
    };
    let d0 = &m.devices[main];
    let (backend, mut notes) = backend_of(m, main);
    let users = ask.users.max(1) as f64;
    let fam = ask.family;
    let default_ctx = fam.and_then(|f| f.default_ctx).unwrap_or(TARGET_CTX).min(sh.max_ctx);
    let max_ctx = ask.context.unwrap_or(default_ctx).min(fam.and_then(|f| f.max_ctx).unwrap_or(usize::MAX));
    // (a model with recurrent states keeps a second copy of each sequence's states at its prompt's end, and the pool of them kept at block boundaries,
    // src/dynamic/snaps.rs: as the loader makes them; a latent cache is paged as any keys are, so a latent attention's model keeps them too)
    let marked = if sh.state > 0.0 { sh.state } else { 0.0 };
    let pool = if marked > 0.0 { crate::dynamic::snaps::count(sh.state as usize) as f64 * sh.state } else { 0.0 };
    let kv_at = |ctx: usize| (sh.kv_full * ctx as f64 + sh.kv_sliding * ctx.min(sh.window.max(1)) as f64 + sh.state + marked) * users + pool + sh.kv_once(ctx);
    let same: Vec<usize> = (0..m.devices.len()).filter(|&i| m.devices[i].vendor == d0.vendor).collect();
    // (the draft head counts where the family's path drafts with it; a model no path runs yet is taken to)
    let gain = gain_of(sh, fam.is_none_or(|f| f.drafts));
    let step = sh.layers as f64 * (d0.consts.layer_s + d0.consts.width_s * sh.hidden as f64);
    // (every level the family runs, or the one asked)
    let levels: Vec<&'static str> = levels::LEVELS
        .iter()
        .map(|l| l.name)
        .filter(|n| ask.level.map_or(levels::names_now().contains(n), |a| a == *n))
        .filter(|n| fam.is_none_or(|f| f.allows(n)))
        .collect();
    let mut options = Vec::new();
    for &level in &levels {
        let Some((bits, ebits)) = level_bits(level, sh, d0, fam.is_some_and(crate::support::Support::hummingbird_one_gpu)) else { continue };
        let rest = (sh.embed + sh.head + sh.dense) * bits / 8.0;
        let experts = sh.routed * ebits / 8.0;
        let weights = rest + experts;
        let (wb, kvb) = reads_of(sh, bits, ebits, ask.positions(), 1.0);
        let w_share = d0.consts.w_of(read_of(level, sh, d0, None));
        let (read, eff) = (wb + kvb * d0.consts.kv / w_share, 1.0 / w_share);
        // the largest context that fits in `budget` bytes once `fixed` are placed
        let ctx_in = |budget: f64, fixed: f64| -> Option<usize> {
            let fits: Vec<usize> = CONTEXTS.iter().copied().filter(|&c| c <= max_ctx && fixed + kv_at(c) <= budget).collect();
            fits.last()
                .copied()
                .or_else(|| (ask.context.is_none() && fixed + kv_at(MIN_CTX) <= budget).then_some(MIN_CTX))
                .filter(|&c| c >= MIN_CTX.min(max_ctx))
        };
        let option = |placement: Option<Placement>, context: usize, prior: f64, why: Option<String>| {
            let in_vram = matches!(placement, Some(Placement::Gpu(_) | Placement::Split(_)));
            Candidate {
                level,
                bits: if experts > 0.0 && ebits != bits { (rest * 8.0 + experts * 8.0) / (sh.params()) } else { bits },
                weights: weights as u64,
                placement,
                context,
                kv: kv_at(context) as u64,
                why,
                load: None,
                ..Candidate::speeds(prior, d0, if in_vram { prefill_of(sh, d0) } else { None })
            }
        };
        // one GPU
        if let Some(ctx) = ctx_in(room(d0) as f64, weights) {
            options.push(option(Some(Placement::Gpu(main)), ctx, gain / (read / (d0.rate() * eff) + step), None));
            continue;
        }
        // a split over the GPUs of the vendor (where the family's path splits at this level)
        let total_room: f64 = same.iter().map(|&i| room(&m.devices[i]) as f64).sum();
        if same.len() > 1 && fam.is_none_or(|f| f.split_levels.contains(&level)) {
            if let Some(ctx) = ctx_in(total_room, weights) {
                // (one GPU at a time: each reads its share, in proportion to its room)
                let t: f64 = same.iter().map(|&i| read * room(&m.devices[i]) as f64 / total_room / (m.devices[i].rate() * eff)).sum::<f64>()
                    + step
                    + 0.1e-3 * (same.len() - 1) as f64;
                options.push(option(Some(Placement::Split(same.clone())), ctx, gain / t, None));
                continue;
            }
        }
        // a mixture of experts: the experts in the computer's memory, the rest and a cache of experts on the GPU (not where the GPU
        // already uses the computer's memory: there RAM is the same memory, counted once)
        if sh.is_moe() && !d0.unified && fam.is_none_or(|f| f.experts_in_ram_levels.contains(&level)) {
            let ram = (m.ram_free as f64 - 4.0 * GIB) * 0.9;
            // (a family the description builder maps at hummingbird, src/hbg_store.rs: its cache and its memory hold a token's experts in every layer at least; elsewhere the cache a twentieth of the experts)
            let generic_hb = level == "hummingbird" && !fam.is_some_and(crate::support::Support::hummingbird_one_gpu);
            let token = sh.routed_active() * ebits / 8.0;
            let min_cache = if generic_hb { token } else { 0.05 * experts };
            // (the hummingbird level's experts are read from their files where the computer's memory does not hold them all: the disk tier)
            let disk = if level == "hummingbird" && experts > ram && ram >= token.max(experts / 1000.0) { experts - ram } else { 0.0 };
            if experts > ram && disk == 0.0 {
                options.push(option(
                    None,
                    0,
                    0.0,
                    Some(format!("its experts take {:.1} GiB and the computer has {:.1} GiB free", experts / GIB, m.ram_free as f64 / GIB)),
                ));
                continue;
            }
            if let Some(ctx) = ctx_in(room(d0) as f64, rest + min_cache) {
                let cache = (room(d0) as f64 - rest - kv_at(ctx)).max(min_cache);
                let hit = (cache / experts).min(1.0).powf(HIT_EXPONENT);
                let active = sh.routed_active() * ebits / 8.0;
                let gpu_t = (read - active * (1.0 - hit)) / (d0.rate() * eff) + step;
                let cpu_t = active * (1.0 - hit) / (m.ram_bandwidth * GB * EFF_CPU);
                // (the misses the memory does not hold, at the disk's rate: the experts uniformly over the tiers)
                let disk_t = active * (1.0 - hit) * (disk / experts) / (DISK_GBS * GB);
                let t = gpu_t.max(cpu_t + disk_t) + SYNC_PER_MOE_LAYER * sh.moe_layers as f64;
                options.push(option(Some(Placement::ExpertsInRam { gpu: main, cache: cache as u64, hit, disk: disk as u64 }), ctx, gain / t, None));
                continue;
            }
        }
        let need = weights + kv_at(MIN_CTX.min(max_ctx));
        options.push(option(
            None,
            0,
            0.0,
            Some(format!(
                "needs {:.1} GiB with {} tokens of context; the GPU{} {:.1} GiB free for it",
                need / GIB,
                MIN_CTX.min(max_ctx),
                if same.len() > 1 { "s have" } else { " has" },
                total_room.max(room(d0) as f64) / GIB
            )),
        ));
    }
    let chosen = choose(&options, max_ctx);
    if chosen.is_none() {
        notes.push("nothing fits this machine: a smaller model, or more memory (or other machines: glyd cluster, ROADMAP M4)".into());
    }
    Plan { backend, chosen, options, notes }
}

// ---- the plan that loads ----------------------------------------------------------------------------------------------------------------------------------------------------------------

/// The GPU a plan is made for, and the computer's free memory: what `run` reads of the GPU it picked and `fit` of a machine's.
#[derive(Clone, Debug)]
pub struct Target {
    /// The GPU's number in the driver's list.
    pub ordinal: i32,
    pub name: String,
    pub cc: (i32, i32),
    pub total: u64,
    pub free: u64,
    /// The computer's memory the experts may take (None: it does not say, and is taken to have enough).
    pub host: Option<u64>,
    /// Whether the computer has cuBLAS (`Device::cublas`): None for this computer, whose loader is asked.
    pub cublas: Option<bool>,
}

impl Target {
    pub fn of(d: &Device, ordinal: i32, host: Option<u64>) -> Target {
        Target { ordinal, name: d.name.clone(), cc: d.cc, total: d.total, free: d.free, host, cublas: d.cublas }
    }
}

/// `fit --json`'s schema: its `schema` member.
pub const SCHEMA: &str = "glyd-plan/1";

/// The engine path that loads a plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// The plain dense kind (every pre-norm dense decoder the description builder's `llama` reading takes) on the executor: src/dynamic/dense.rs, at penguin, kestrel, swift and wren.
    Qwen3,
    /// Qwen3.5, 3.6 and 3.8 dense at a lossy level.
    Qwen35,
    NemotronH,
    Gemma4,
    /// The dynamic route (src/dynamic): Kimi-K3, GLM-5.3-Flash, DeepSeek-V4, Muse Spark and Nemotron-H lossless, over one GPU or several.
    Dynamic,
}

/// How a plan spreads a model over several GPUs, and the experts the GPUs do not hold (the dynamic route's).
#[derive(Clone, Debug, PartialEq)]
pub struct Spread {
    /// The layers cut before each of `cuts` (ascending), stage k on GPU `devices[k]`; every layer across all `devices` where `across`.
    pub cuts: Vec<usize>,
    pub devices: Vec<i32>,
    pub across: bool,
    /// An FP8 release's matrices held coded (about 7 bits of 8) to fit, and bf16 matrices in the 12-bit layout.
    pub coded: bool,
    pub twelve: bool,
    /// Kimi-K3's expert cache: places of a layer's experts on each stage's GPU (one entry where `across`), the experts a layer has, and the bytes of
    /// every expert, all of which are in the computer's memory.
    pub places: Vec<usize>,
    pub experts_per_layer: usize,
    pub host_bytes: u64,
    /// Each stage's share of `host_bytes`: the experts of its own layers (one entry where `across`; zeros where nothing is in the computer's memory). They add
    /// up to `host_bytes`.
    pub host_per_stage: Vec<u64>,
    /// The bytes the GPUs hold of the model (its weights, and the places' experts).
    pub gpu_bytes: u64,
    /// The last layers whose dense matrices are in the computer's memory (src/dense_ram.rs: one stage, P2d's rung after the experts), and their bytes there.
    pub dense: usize,
    pub dense_bytes: u64,
    /// The tables read by rows its description holds on the host (src/dynamic/tables.rs: the embedding's, the n-gram tables'): the checkpoint's files mapped,
    /// the rows a pass reads in the page cache; no GPU holds them.
    pub tables: u64,
}

/// The one plan: what `fit` prints before anything is downloaded and what `run`, `serve` and `preflight` load. Every field is bytes
/// or a count the loader uses; the engine reports what it allocated and the tests compare the two (`Loaded`).
#[derive(Clone, Debug, PartialEq)]
pub struct LoadPlan {
    pub kind: Kind,
    /// The family as the table names it (src/support.rs).
    pub family: &'static str,
    pub level: &'static str,
    pub layout: crate::weights::Layout,
    pub context: usize,
    /// Sequences the engine holds at once; the cache is `context` tokens of each.
    pub users: usize,
    /// Prompt tokens a pass takes at a time.
    pub chunk: usize,
    /// All the weights as loaded, wherever they go, and what they are in bf16.
    pub weights: u64,
    pub bf16: u64,
    /// Keys, values, rings and states of `users` sequences of `context` tokens.
    pub kv: u64,
    /// The keys and values in the executor's 12-bit lossless pages (GLYD_KV=lossless asked for them, and they reserve no more than bf16 for this plan:
    /// `sizing::one_gpu_pages`), else bf16. The loader takes the plan's word (src/cli.rs `twelve_bit_pages`).
    pub kv12: bool,
    /// Activations and workspaces of the engine, and a draft model's memory.
    pub work: u64,
    pub draft: u64,
    /// What the GPU keeps free (`reserve_of`), and the free memory and the GPU the plan was made for.
    pub reserve: u64,
    pub free: u64,
    pub gpu: String,
    /// A mixture of experts' experts in the computer's memory (and the disk), a cache of them on the GPU.
    pub experts: Option<crate::offload::CachePlan>,
    /// The GPUs a model on the dynamic route is spread over (`free`, `work` and `reserve` are then the sums over them).
    pub spread: Option<Spread>,
}

impl LoadPlan {
    /// The weights the GPU holds: all of them, or all but the experts the computer's memory holds, and the expert cache.
    pub fn weights_on_gpu(&self) -> u64 {
        if let Some(s) = &self.spread {
            return s.gpu_bytes;
        }
        self.experts.map_or(self.weights, |c| self.weights.saturating_sub(c.host_bytes()) + c.gpu_bytes())
    }

    /// Everything the plan puts on the GPU, and the reserve.
    pub fn gpu_total(&self) -> u64 {
        self.weights_on_gpu() + self.kv + self.work + self.draft + self.reserve
    }

    /// The experts in the computer's memory, and those read from the disk as the router asks.
    pub fn ram_bytes(&self) -> u64 {
        self.spread.as_ref().map_or_else(|| self.experts.map_or(0, |c| c.ram.min(c.host_bytes())), |s| s.host_bytes)
    }

    pub fn disk_bytes(&self) -> u64 {
        self.experts.map_or(0, |c| c.disk_bytes())
    }

    /// The plan as `fit --json` prints it (`glyd-plan/1`): the same numbers the terminal's text has.
    pub fn json(&self, model: &str) -> String {
        let experts = self.experts.map(|c| {
            serde_json::json!({
                "places_per_layer": c.slots, "experts_per_layer": c.experts, "layers": c.layers,
                "gpu_bytes": c.gpu_bytes(), "ram_bytes": self.ram_bytes(), "disk_bytes": self.disk_bytes(),
            })
        });
        serde_json::json!({
            "schema": SCHEMA, "model": model, "family": self.family, "level": self.level,
            "context": self.context, "users": self.users, "chunk": self.chunk,
            "gpu": self.gpu, "free_bytes": self.free,
            "weights_bytes": self.weights, "weights_on_gpu_bytes": self.weights_on_gpu(),
            "kv_bytes": self.kv, "kv_format": self.kv_format(), "work_bytes": self.work, "draft_bytes": self.draft,
            "reserve_bytes": self.reserve, "gpu_total_bytes": self.gpu_total(),
            "experts": experts,
            "spread": self.spread.as_ref().map(|s| {
                let mut v = serde_json::json!({
                    "cuts": s.cuts, "devices": s.devices, "across": s.across, "coded": s.coded, "twelve": s.twelve,
                    "places": s.places, "experts_per_layer": s.experts_per_layer, "host_bytes": s.host_bytes, "host_per_stage": s.host_per_stage,
                });
                // (the tables read by rows on the host, where a description holds any there: the other plans read as they did before the field)
                if s.tables > 0 {
                    v["tables_bytes"] = s.tables.into();
                }
                v
            }),
        })
        .to_string()
    }

    /// The plan and what the load took (`glyd-plan/1`'s `loaded`): the weights the engine counts and the GPU's memory the load used; for calibrating the
    /// planner (`GLYD_PLAN_REPORT=FILE` in a development build writes it) and for the GPU check that the plan is what loaded.
    pub fn report(&self, model: &str, weights_loaded: u64, gpu_used: u64) -> String {
        let plan: serde_json::Value = serde_json::from_str(&self.json(model)).unwrap_or_default();
        serde_json::json!({
            "schema": SCHEMA, "plan": plan,
            "loaded": {"weights_bytes": weights_loaded, "gpu_used_bytes": gpu_used, "planned_gpu_used_bytes": self.gpu_total() - self.reserve},
        })
        .to_string()
    }

    /// The keys and values' format: "12-bit" (the executor's lossless 12-bit pages, `kv12`) or "bf16".
    pub fn kv_format(&self) -> &'static str {
        kv_format_of(self.kv12)
    }

    /// The plan in words, for the line `run` says before it loads and `fit` prints.
    pub fn text(&self) -> String {
        let mut s = format!(
            "{} level, {} tokens of context{}: {} of weights on the GPU, {} of cache ({}), {} for the engine's work{}, {} kept free; {} of the {}'s {} free",
            self.level,
            self.context,
            if self.users > 1 { format!(" for each of {} users", self.users) } else { String::new() },
            gb(self.weights_on_gpu()),
            gb(self.kv),
            if self.kv12 { "keys and values in 12-bit pages" } else { "keys and values in bf16" },
            gb(self.work),
            if self.draft > 0 { format!(", {} for the draft model", gb(self.draft)) } else { String::new() },
            gb(self.reserve),
            gb(self.gpu_total()),
            self.gpu,
            gb(self.free),
        );
        if let Some(sp) = &self.spread {
            s += &if sp.across {
                format!("; every layer across GPUs {:?}", sp.devices)
            } else {
                let mut b = vec![0];
                b.extend(&sp.cuts);
                format!("; stages from layer {:?} on GPUs {:?}", b, sp.devices)
            };
            if sp.host_bytes > 0 {
                s += &format!(
                    ", {} experts a layer of {} on the GPUs, {} in the computer's memory",
                    sp.places.iter().min().copied().unwrap_or(0),
                    sp.experts_per_layer,
                    gb(sp.host_bytes)
                );
            }
            if sp.dense > 0 {
                s += &format!(", the last {} layers' dense matrices ({}) in the computer's memory", sp.dense, gb(sp.dense_bytes));
            }
            if sp.tables > 0 {
                s += &format!(", its tables read by rows ({}) from the checkpoint's files", gb(sp.tables));
            }
            if sp.coded {
                s += ", FP8 matrices coded";
            }
        }
        if let Some(c) = &self.experts {
            s += &format!(
                "; experts: {} places a layer of {} on the GPU, {} in the computer's memory, {} on the disk",
                c.slots,
                c.experts,
                gb(self.ram_bytes()),
                gb(self.disk_bytes())
            );
        }
        s
    }
}

/// The bits a weight of each part takes in plan `lp` on `d`: the level's (`level_bits`), and at penguin the layout the loader took (`LoadPlan::layout`: the 12-bit
/// one where it decodes faster and fits, `best_layout`; a dense model of 36 layers of 4,096 on an A10 loaded 11.10 GB in it, where the level's 10.75 bits counted 10.17).
fn plan_bits(lp: &LoadPlan, sh: &Shape, d: &Device) -> Option<(f64, f64)> {
    let (bits, ebits) = level_bits(lp.level, sh, d, crate::support::row_named(lp.family).is_some_and(crate::support::Support::hummingbird_one_gpu))?;
    Some(match lp.layout {
        crate::weights::Layout::Twelve | crate::weights::Layout::Tiered if lossless(lp.level, sh) => (sizing::bits(lp.layout), ebits),
        _ => (bits, ebits),
    })
}

/// Bytes a decode step of `users` sequences reads at `bits` (`ebits` the routed experts') at `positions` positions each: the weights (the matrices it reads whole:
/// the head, the embedding where it is tied, the dense ones; and the routed experts the sequences' tokens choose, as many as uniform choices make between them),
/// and every sequence's keys and values.
fn reads_of(sh: &Shape, bits: f64, ebits: f64, positions: f64, users: f64) -> (f64, f64) {
    let chosen = if sh.experts > 0 && sh.topk > 0 {
        let (e, k) = (sh.experts as f64, sh.topk as f64);
        (e * (1.0 - (1.0 - k / e).powf(users.max(1.0))) / k).max(1.0)
    } else {
        1.0
    };
    let weights = ((sh.head.max(if sh.head == 0.0 { sh.embed } else { 0.0 }) + sh.dense) * bits + sh.routed_active() * chosen * ebits) / 8.0;
    (weights, users.max(1.0) * kv_at(sh, positions))
}

/// One sequence's keys and values at `positions` positions (the sliding layers' window at most).
fn kv_at(sh: &Shape, positions: f64) -> f64 {
    sh.kv_full * positions + sh.kv_sliding * positions.min(sh.window.max(1) as f64)
}

/// Whether `level` is the lossless layout of a bf16 release (its weights decoded in the product).
fn lossless(level: &str, sh: &Shape) -> bool {
    level == "penguin" && sh.release_bits > 8.0
}

/// Which of a class's weight shares a step reads at.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Read {
    Packs,
    Tiered,
    Twelve,
}

/// The share level `level` reads its weights at on `d` (`layout`: the plan's, where there is one; else the level's rule, `level_bits`'s).
fn read_of(level: &str, sh: &Shape, d: &Device, layout: Option<crate::weights::Layout>) -> Read {
    if !lossless(level, sh) {
        return Read::Packs;
    }
    match layout {
        Some(crate::weights::Layout::Twelve) => Read::Twelve,
        Some(crate::weights::Layout::Tiered) => Read::Tiered,
        _ if matches!(d.cc, (8, 0) | (9, 0) | (10, 0)) => Read::Twelve,
        _ => Read::Tiered,
    }
}

/// A decode step's seconds for `users` at `positions` on `d` with every weight on it.
fn step_on(sh: &Shape, d: &Device, level: &str, bits: f64, ebits: f64, positions: f64, users: usize) -> f64 {
    let (w, kv) = reads_of(sh, bits, ebits, positions, users as f64);
    d.consts.step_s(d.rate(), w, kv, sh.layers as f64, sh.hidden as f64, read_of(level, sh, d, None), users as f64)
}

/// A prompt token's attention reads at a prompt of `n` tokens: the keys and values of the positions before it, on the mean (half the prompt).
fn attn_bytes(sh: &Shape, n: f64) -> f64 {
    kv_at(sh, n / 2.0)
}

/// A prompt's tokens a second on `d` with every weight on the GPU, for a prompt of about 1,000 tokens (`prefill_at`): None where the GPU's tensor rate is not known.
pub fn prefill_of(sh: &Shape, d: &Device) -> Option<f64> {
    prefill_at(sh, d, POSITIONS)
}

/// A prompt of `n` tokens' tokens a second on `d` (its class's figures): None where the GPU's tensor rate is not known.
pub fn prefill_at(sh: &Shape, d: &Device, n: f64) -> Option<f64> {
    (d.tflops > 0.0).then(|| 1.0 / d.consts.token_s(2.0 * (sh.dense + sh.routed_active()), sh.layers as f64, attn_bytes(sh, n), d.rate()))
}

/// Decode tokens/s of a plan `run` loads at `positions` positions, from the model's parameters and the machine's rates (the planner's model, for a placement the
/// loader chose), before what runs here observed refines it.
fn speed_at(sh: &Shape, m: &Machine, main: usize, lp: &LoadPlan, positions: f64) -> f64 {
    speed_users(sh, m, main, lp, positions, 1)
}

/// `speed_at` for each of `users` sequences decoded together: one step reads the weights once for all of them, and each one's keys and values, and takes its
/// class's time a layer for each row past the first.
pub fn speed_users(sh: &Shape, m: &Machine, main: usize, lp: &LoadPlan, positions: f64, users: usize) -> f64 {
    speed_routed(sh, m, main, lp, positions, users, HIT_EXPONENT)
}

/// `speed_users` for a router whose choices have the skew `skew` (`step_hit`: `HIT_EXPONENT` for a release's, 1 for a uniform one, a random model's): a step of `users` rows reads each of its distinct missed
/// experts once (`step_misses`).
pub fn speed_routed(sh: &Shape, m: &Machine, main: usize, lp: &LoadPlan, positions: f64, users: usize, skew: f64) -> f64 {
    let d0 = &m.devices[main];
    let Some((bits, ebits)) = plan_bits(lp, sh, d0) else { return 0.0 };
    let gain = gain_of(sh, drafts(lp));
    // (the step's floor, and its reads at the class's shares of the rate: the weights' share and the keys' and values')
    let (wb, kvb) = reads_of(sh, bits, ebits, positions, users as f64);
    let c = &d0.consts;
    let step = c.floor_s(sh.layers as f64, sh.hidden as f64, users as f64);
    let w_share = c.w_of(read_of(lp.level, sh, d0, Some(lp.layout)));
    // (a byte of weights at the class's share; the keys' and values' bytes counted as weights' at the same time)
    let read = wb + kvb * c.kv / w_share;
    let eff = 1.0 / w_share;
    if let Some(sp) = &lp.spread {
        // (the GPUs it is over: stages read in turn, a layer across them reads on all at once; the experts the GPUs do not hold are the CPU's)
        let rates: Vec<f64> = sp.devices.iter().filter_map(|&d| m.devices.get(d as usize)).map(Device::rate).collect();
        let n = rates.len().max(1) as f64;
        let active = sh.routed_active() * ebits / 8.0;
        let missed = if sp.host_bytes > 0 {
            active * step_misses(sp.experts_per_layer, sh.topk, sp.places.iter().min().copied().unwrap_or(0), users, skew)
        } else {
            0.0
        };
        let on_gpu = read - missed;
        let gpu_t = if sp.across {
            on_gpu / (n * rates.iter().copied().fold(f64::MAX, f64::min) * eff)
        } else {
            rates.iter().map(|r| on_gpu / n / (r * eff)).sum::<f64>()
        } + step
            + 0.1e-3 * (n - 1.0);
        let cpu_t = missed / (m.ram_bandwidth * GB * EFF_CPU);
        // (the last layers' dense matrices in the computer's memory, src/dense_ram.rs: each copied to the GPU before it runs, every token)
        let link_t = sp.dense_bytes as f64 / (LINK_GBS * GB);
        return gain / (gpu_t.max(cpu_t) + link_t + if sp.host_bytes > 0 { SYNC_PER_MOE_LAYER * sh.moe_layers as f64 } else { 0.0 });
    }
    match &lp.experts {
        None => gain / (read / (d0.rate() * eff) + step),
        Some(c) => {
            let missed = sh.routed_active() * ebits / 8.0 * step_misses(c.experts, sh.topk, c.slots, users, skew);
            let gpu_t = (read - missed) / (d0.rate() * eff) + step;
            let cpu_t = missed / (m.ram_bandwidth * GB * EFF_CPU);
            gain / (gpu_t.max(cpu_t) + SYNC_PER_MOE_LAYER * sh.moe_layers as f64)
        }
    }
}

/// What a calibration record keeps of a plan at `positions` positions for `users` sequences (`glyd-calib/1`'s `features`): the bytes a decode step reads, the
/// layers, a prompt token's products (2 x the parameters it multiplies), the tokens a step makes, whether the level is lossless and whether every weight is on
/// the GPUs: what `fit_consts` fits the figures on.
pub fn features(sh: &Shape, m: &Machine, main: usize, lp: &LoadPlan, positions: f64, users: usize) -> serde_json::Value {
    let d0 = &m.devices[main];
    let (bits, ebits) = plan_bits(lp, sh, d0).unwrap_or((16.0, 16.0));
    let in_vram = lp.experts.is_none() && lp.spread.as_ref().is_none_or(|s| s.host_bytes == 0 && s.dense == 0);
    let (w, kv) = reads_of(sh, bits, ebits, positions, users as f64);
    serde_json::json!({
        "weight_bytes": w, "kv_bytes": kv, "layers": sh.layers, "hidden": sh.hidden, "prompt_flops": 2.0 * (sh.dense + sh.routed_active()),
        "attn_bytes": attn_bytes(sh, (positions - 64.0).max(0.0)), "gain": gain_of(sh, drafts(lp)), "lossless": lossless(lp.level, sh),
        "twelve": read_of(lp.level, sh, d0, Some(lp.layout)) == Read::Twelve, "in_vram": in_vram,
    })
}

fn speed_of(sh: &Shape, m: &Machine, main: usize, lp: &LoadPlan) -> f64 {
    speed_at(sh, m, main, lp, POSITIONS)
}

/// Whether the path that loads `lp` drafts with the model's own head: its family's row says (`Support::drafts`, the flag `plan` and the ladder read).
fn drafts(lp: &LoadPlan) -> bool {
    crate::support::row_named(lp.family).is_some_and(|r| r.drafts)
}

/// The tokens a decode step makes: `MTP_GAIN` where the model has a draft head and the path drafts with it (`drafts`), else one.
fn gain_of(sh: &Shape, drafts: bool) -> f64 {
    if sh.mtp && drafts {
        MTP_GAIN
    } else {
        1.0
    }
}

/// Decode tokens/s of `level` with every weight on the GPU `main` (the speed a run at that level would have if it fitted), on a path that drafts or not.
fn ideal_speed(sh: &Shape, m: &Machine, main: usize, level: &str, fam: &crate::support::Support) -> f64 {
    let d0 = &m.devices[main];
    let Some((bits, ebits)) = level_bits(level, sh, d0, fam.hummingbird_one_gpu()) else { return 0.0 };
    gain_of(sh, fam.drafts) / step_on(sh, d0, level, bits, ebits, POSITIONS, 1)
}

// ---- the degrade ladder ----------------------------------------------------------------------------------------------------------------------------------------------------------------

/// A step that costs more than this share of the speed (percent) is asked about before it is taken.
pub const CONFIRM_COST_PCT: f64 = 30.0;

/// What the ladder takes, in the order it takes them (ROADMAP P2d): the KV cache's type (this engine's cache has one, lossless), a shorter context down to the floor,
/// experts in the computer's memory, dense layers in it (src/dense_ram.rs: one GPU's last layers, where a token's experts do not fit beside them), then a lower level,
/// then a refusal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum StepKind {
    ShorterContext,
    ExpertsInRam,
    DenseInRam,
    LowerLevel,
}

/// One step of the ladder: what it does, what it costs of the speed (percent, negative where it is faster), and whether it is asked about first.
#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    pub kind: StepKind,
    pub what: String,
    pub cost_pct: f64,
    /// Asked about: a cost over `CONFIRM_COST_PCT` of the speed, or any lower level (it is no longer the level asked for).
    pub confirm: bool,
}

/// The ladder from the level asked for: the steps it takes before the model loads, the plan that loads (None: no step loads it, or the paths' sizing is not exact
/// for this model), the first refusal where none does, and the steps this build does not have.
#[derive(Debug)]
pub struct Ladder {
    pub asked: &'static str,
    pub context: usize,
    pub steps: Vec<Step>,
    pub plan: Option<LoadPlan>,
    pub refusal: Option<crate::ui::Refusal>,
}

/// What this build has not got of the ladder's order.
pub const UNAVAILABLE: [&str; 1] = ["the KV cache's type: this engine's cache has one type (lossless), so there is no step to a smaller one"];

impl Ladder {
    /// Whether the model loads as asked (no step), and whether any step is asked about.
    pub fn is_as_asked(&self) -> bool {
        self.plan.is_some() && self.steps.is_empty()
    }

    pub fn needs_confirmation(&self) -> bool {
        self.steps.iter().any(|s| s.confirm)
    }

    /// The ladder as `fit --json` has it: every step with its cost, and the plan it loads (`degrade` of `glyd-plan/1`).
    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "asked": self.asked,
            "as_asked": self.is_as_asked(),
            "loads": self.plan.is_some(),
            "level": self.plan.as_ref().map(|p| p.level),
            "context": self.plan.as_ref().map(|p| p.context),
            "needs_confirmation": self.needs_confirmation(),
            "steps": self.steps.iter().map(|s| serde_json::json!({
                "kind": match s.kind {
                    StepKind::ShorterContext => "shorter_context",
                    StepKind::ExpertsInRam => "experts_in_ram",
                    StepKind::DenseInRam => "dense_in_ram",
                    StepKind::LowerLevel => "lower_level",
                },
                "what": s.what, "cost_percent": (s.cost_pct * 10.0).round() / 10.0, "confirm": s.confirm,
            })).collect::<Vec<_>>(),
            "refusal": self.refusal.as_ref().map(|r| r.what.clone()),
            "unavailable": UNAVAILABLE,
        })
    }

    /// What `fit` and `run` say of it: nothing where the model loads as asked; else the steps in their order, which of them ask first, and the steps this build has not got.
    pub fn report(&self) -> String {
        if self.is_as_asked() {
            return String::new();
        }
        let mut s = String::new();
        match (&self.plan, &self.refusal) {
            (Some(p), _) => {
                s += &format!("The ladder, from {} at {} tokens (KV type, shorter context, experts in RAM, dense layers in RAM, lower level, refuse): it loads at the {} level, {} tokens, with\n", self.asked, self.context, p.level, p.context);
                s += &self.text();
                if self.needs_confirmation() {
                    s += "  `run` asks before the steps marked; `serve`, the service and a run with no terminal take them only with --yes (or GLYD_DEGRADE=yes), else they refuse.\n";
                }
            }
            (None, Some(r)) => s += &format!("The ladder, from {} at {} tokens: no step loads it ({}).\n", self.asked, self.context, r.what),
            (None, None) => return String::new(),
        }
        for u in UNAVAILABLE {
            s += &format!("  not available: {u}\n");
        }
        s
    }

    /// The steps as the terminal shows them.
    pub fn text(&self) -> String {
        let mut s = String::new();
        for (i, st) in self.steps.iter().enumerate() {
            let cost = if st.cost_pct > 0.5 {
                format!("about {:.0}% slower", st.cost_pct)
            } else if st.cost_pct < -0.5 {
                format!("about {:.0}% faster", -st.cost_pct)
            } else {
                "no change in speed".to_string()
            };
            s += &format!("  {}. {} ({cost}){}\n", i + 1, st.what, if st.confirm { ": asks first (--yes takes it)" } else { "" });
        }
        s
    }
}

/// The ladder of MODEL from `asked` at `want_ctx` tokens of context: each level from the one asked for down, asked of the loader's own sizing (`size`: the plan it
/// loads, or the refusal it gives), the first that loads, and the steps that took it there: the context it had to shorten, the experts it keeps in the computer's
/// memory and the dense layers it keeps there (each priced by itself, from the planner's model), the lower level. `confirm` marks the steps taken only when asked.
pub fn ladder(
    sh: &Shape,
    m: &Machine,
    fam: &crate::support::Support,
    model: &str,
    asked: &'static str,
    want_ctx: usize,
    size: &dyn Fn(&'static str, usize) -> Result<Option<LoadPlan>, crate::ui::Refusal>,
) -> Ladder {
    let levels: Vec<&'static str> = fam.levels_now(model);
    let from = levels.iter().position(|l| *l == asked).unwrap_or(0);
    let main = (0..m.devices.len()).min_by_key(|&i| (std::cmp::Reverse(m.devices[i].free), i)).unwrap_or(0);
    let mut refusal = None;
    // (each level at the context wanted, then shorter ones down to the floor, before the next level: a loader that does not shorten the context itself, Gemma 4's, is
    // taken at the longest that fits)
    let tries: Vec<usize> = std::iter::once(want_ctx).chain(CONTEXTS.iter().rev().copied().filter(|&c| c < want_ctx && c >= PREFERRED_CTX)).collect();
    for &level in &levels[from..] {
        let mut sized = None;
        for &ctx in &tries {
            match size(level, ctx) {
                Ok(None) => return Ladder { asked, context: want_ctx, steps: vec![], plan: None, refusal },
                Err(r) => {
                    refusal.get_or_insert(r);
                }
                Ok(Some(lp)) => {
                    sized = Some(lp);
                    break;
                }
            }
        }
        match sized {
            None => {}
            Some(lp) => {
                let mut steps = Vec::new();
                if lp.context < want_ctx {
                    steps.push(Step {
                        kind: StepKind::ShorterContext,
                        what: format!("shorter context: {} tokens where {} were wanted", lp.context, want_ctx),
                        cost_pct: 0.0,
                        confirm: false,
                    });
                }
                // (the plan with every dense layer on the GPU, were there room: each of the two steps below is priced by itself)
                let dense = lp.spread.as_ref().filter(|s| s.dense > 0).map(|s| (s.dense, s.dense_bytes));
                let mut all_on_gpu = lp.clone();
                if let Some(s) = all_on_gpu.spread.as_mut() {
                    (s.dense, s.dense_bytes) = (0, 0);
                }
                // (the hummingbird level is made for its experts to be in the computer's memory: that is the level asked for, not a step down from it)
                let in_memory = lp.experts.is_some() || lp.spread.as_ref().is_some_and(|s| s.host_bytes > 0);
                if in_memory && level != "hummingbird" && !m.devices.is_empty() {
                    let (ideal, got) = (ideal_speed(sh, m, main, level, fam), speed_of(sh, m, main, &all_on_gpu));
                    let cost = if ideal > 0.0 { (1.0 - got / ideal) * 100.0 } else { 0.0 };
                    steps.push(Step {
                        kind: StepKind::ExpertsInRam,
                        what: format!(
                            "experts in the computer's memory: {} there, a cache of them on the GPU",
                            crate::ui::gib(lp.ram_bytes() + lp.disk_bytes())
                        ),
                        cost_pct: cost,
                        confirm: cost > CONFIRM_COST_PCT,
                    });
                }
                if let Some((layers, bytes)) = dense.filter(|_| !m.devices.is_empty()) {
                    let (before, after) = (speed_of(sh, m, main, &all_on_gpu), speed_of(sh, m, main, &lp));
                    let cost = if before > 0.0 { (1.0 - after / before) * 100.0 } else { 0.0 };
                    steps.push(Step {
                        kind: StepKind::DenseInRam,
                        what: format!(
                            "dense layers in the computer's memory: the last {layers} layers' matrices ({}) there, each copied to the GPU before it runs",
                            crate::ui::gib(bytes)
                        ),
                        cost_pct: cost,
                        confirm: cost > CONFIRM_COST_PCT,
                    });
                }
                if level != asked && !m.devices.is_empty() {
                    let (before, after) = (ideal_speed(sh, m, main, asked, fam), ideal_speed(sh, m, main, level, fam));
                    steps.push(Step {
                        kind: StepKind::LowerLevel,
                        what: format!("lower level: {level} where {asked} was asked for (not the level asked for: smaller weights, not lossless)"),
                        cost_pct: if before > 0.0 { (1.0 - after / before) * 100.0 } else { 0.0 },
                        confirm: true,
                    });
                }
                steps.sort_by_key(|s| s.kind);
                return Ladder { asked, context: want_ctx, steps, plan: Some(lp), refusal: None };
            }
        }
    }
    Ladder { asked, context: want_ctx, steps: vec![], plan: None, refusal }
}

/// An option of the ladder from the plan `run` loads at that level.
fn candidate_of(lp: LoadPlan, sh: &Shape, m: &Machine, main: usize, positions: f64) -> Candidate {
    let tok_s = speed_at(sh, m, main, &lp, positions);
    let placement = match (&lp.spread, &lp.experts) {
        (Some(sp), _) if sp.host_bytes > 0 => Placement::ExpertsInRam {
            gpu: sp.devices.first().map_or(main, |&d| d as usize),
            cache: sp.gpu_bytes,
            hit: (sp.places.iter().min().copied().unwrap_or(0) as f64 / sp.experts_per_layer.max(1) as f64).min(1.0).powf(HIT_EXPONENT),
            disk: lp.disk_bytes(),
        },
        (Some(sp), _) if sp.devices.len() > 1 => Placement::Split(sp.devices.iter().map(|&d| d as usize).collect()),
        (Some(sp), _) => Placement::Gpu(sp.devices.first().map_or(main, |&d| d as usize)),
        (None, None) => Placement::Gpu(main),
        (None, Some(c)) => Placement::ExpertsInRam {
            gpu: main,
            cache: c.gpu_bytes(),
            hit: (c.slots as f64 / c.experts.max(1) as f64).min(1.0).powf(HIT_EXPONENT),
            disk: lp.disk_bytes(),
        },
    };
    let in_vram = matches!(placement, Placement::Gpu(_) | Placement::Split(_));
    let d = &m.devices[main];
    Candidate {
        level: lp.level,
        bits: lp.weights as f64 * 8.0 / sh.params().max(1.0),
        weights: lp.weights,
        placement: Some(placement),
        context: lp.context,
        kv: lp.kv,
        why: None,
        load: Some(lp),
        ..Candidate::speeds(tok_s, d, if in_vram { prefill_of(sh, d) } else { None })
    }
}

/// The option the ladder takes: in the levels' order, the first in GPU memory alone with at least `PREFERRED_CTX` of context, else the first with
/// experts in RAM at that context; only then a shorter context (a model held in 4,096 tokens is a poorer run than one level down at 32,768).
pub(crate) fn choose(options: &[Candidate], max_ctx: usize) -> Option<Candidate> {
    let in_vram = |o: &&Candidate| matches!(o.placement, Some(Placement::Gpu(_) | Placement::Split(_)));
    let roomy = |o: &&Candidate| o.placement.is_some() && o.context >= PREFERRED_CTX.min(max_ctx);
    (options.iter().find(|o| in_vram(o) && roomy(o)))
        .or_else(|| options.iter().find(roomy))
        .or_else(|| options.iter().find(in_vram))
        .or_else(|| options.iter().find(|o| o.placement.is_some()))
        .cloned()
}

// ---- the plan that loaded, remembered -----------------------------------------------------------------------------------------------------------------------------------------------------

fn memo_file(root: &std::path::Path, model: &str, level: &str, gpu: &str) -> std::path::PathBuf {
    let slug: String = format!("{}-{level}-{gpu}", crate::cli::short_name(crate::levels::split_tag(model).0))
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '.' { c } else { '_' })
        .collect();
    root.join("plans").join(format!("{slug}.json"))
}

/// What loaded, kept in `root` (the cache's `plans/`) by model, level and GPU: the plan and what the load took (`LoadPlan::report`), for `fit` to say
/// what a model took here before. A failure to write it is not the run's.
pub fn remember_in(root: &std::path::Path, model: &str, plan: &LoadPlan, weights_loaded: u64, gpu_used: u64) {
    let file = memo_file(root, model, plan.level, &plan.gpu);
    if let Some(dir) = file.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let mut v: serde_json::Value = serde_json::from_str(&plan.report(crate::levels::split_tag(model).0, weights_loaded, gpu_used)).unwrap_or_default();
    // (the runs observed with the plan before are kept: `observe`'s)
    if let Some(seen) = recall_in(root, model, plan.level, &plan.gpu).and_then(|b| b.get("observed").cloned()) {
        v["observed"] = seen;
    }
    let _ = std::fs::write(file, v.to_string());
}

/// The plan that loaded `model` at `level` on `gpu` before (`remember_in`'s), if one did.
pub fn recall_in(root: &std::path::Path, model: &str, level: &str, gpu: &str) -> Option<serde_json::Value> {
    serde_json::from_str(&std::fs::read_to_string(memo_file(root, model, level, gpu)).ok()?).ok()
}

/// The bytes of a GPU's free memory to plan as already taken, from what the last load of `model` at `level` on `gpu` took beyond its plan (`LoadPlan::report`'s
/// `gpu_used_bytes` over `planned_gpu_used_bytes`): the planner's calibration on this machine, at most 1 GiB (a load that took far more met another program
/// at the same time) and none where the load said nothing of the GPU's memory.
pub fn calibration_in(root: &std::path::Path, model: &str, level: &str, gpu: &str) -> u64 {
    let Some(before) = recall_in(root, model, level, gpu) else { return 0 };
    let loaded = &before["loaded"];
    match (loaded["gpu_used_bytes"].as_u64(), loaded["planned_gpu_used_bytes"].as_u64()) {
        (Some(used), Some(planned)) if used > 0 => used.saturating_sub(planned).min(1 << 30),
        _ => 0,
    }
}

/// A sentence for `fit` on how the plan it made compares with the one that loaded here before.
fn remembered(root: &std::path::Path, model: &str, c: &Candidate) -> Option<String> {
    let lp = c.load.as_ref()?;
    let before = recall_in(root, model, lp.level, lp.gpu.split(", ").next().unwrap_or(""))?;
    let (plan, loaded) = (before.get("plan")?, before.get("loaded")?);
    let (ctx, used) = (plan.get("context")?.as_u64()?, loaded.get("gpu_used_bytes")?.as_u64()?);
    let same = ctx == lp.context as u64 && plan.get("weights_bytes")?.as_u64()? == lp.weights && plan.get("users")?.as_u64()? == lp.users as u64;
    Some(format!(
        "loaded here before at {ctx} tokens of context, the GPU lost {} ({}: {} planned now)",
        gb(used),
        if same { "the same plan" } else { "a different plan now" },
        gb(lp.gpu_total() - lp.reserve)
    ))
}

// ---- the rates measured here, and the runs observed ------------------------------------------------------------------------------------------------------------

/// The rates the probe measured for a GPU (`glyd-rates/1`, in the cache's `rates/`): what a kernel streaming its memory reads, the link from pinned host memory,
/// and the computer's memory read by all its threads, GB/s.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Probed {
    pub stream_gbs: f64,
    pub link_gbs: f64,
    pub ram_gbs: f64,
}

pub const RATES_SCHEMA: &str = "glyd-rates/1";

fn slug(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '.' { c } else { '_' }).collect()
}

/// The rates of GPU `ordinal` (`name`) on this computer: kept in `root` (`rates/NAME-ORDINAL.json`) from the first time they were measured, else measured now by
/// `measure` (`costs::probe_rates`) and kept. None where they cannot be measured.
pub fn rates_in(root: &std::path::Path, name: &str, ordinal: i32, measure: &dyn Fn() -> Option<Probed>) -> Option<Probed> {
    let file = root.join("rates").join(format!("{}-{ordinal}.json", slug(name)));
    let kept = std::fs::read_to_string(&file)
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .and_then(|v| Some(Probed { stream_gbs: v["stream_gbs"].as_f64()?, link_gbs: v["link_gbs"].as_f64()?, ram_gbs: v["ram_gbs"].as_f64()? }));
    if kept.is_some() {
        return kept;
    }
    let r = measure().filter(|r| r.stream_gbs > 0.0)?;
    let _ = std::fs::create_dir_all(root.join("rates"));
    let v = serde_json::json!({"schema": RATES_SCHEMA, "gpu": name, "ordinal": ordinal, "stream_gbs": r.stream_gbs, "link_gbs": r.link_gbs, "ram_gbs": r.ram_gbs,
        "measured_unix": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())});
    let _ = std::fs::write(file, v.to_string());
    Some(r)
}

/// The p10, p50 and p90 of `x` (sorted here), where there are three or more; with one or two, their middle in `SPREAD`'s band around it. None where there are none.
fn band_of(mut x: Vec<f64>) -> Option<[f64; 3]> {
    x.retain(|v| v.is_finite() && *v > 0.0);
    x.sort_by(|a, b| a.total_cmp(b));
    let q = |f: f64| {
        let at = f * (x.len() - 1) as f64;
        let (lo, hi) = (at.floor() as usize, at.ceil() as usize);
        x[lo] + (x[hi] - x[lo]) * (at - lo as f64)
    };
    match x.len() {
        0 => None,
        1 | 2 => {
            let m = q(0.5);
            Some([m * SPREAD[0], m, m * SPREAD[2]])
        }
        _ => Some([q(0.1), q(0.5), q(0.9)]),
    }
}

/// The runs observed with the plans kept in `root` that pass `keep` (a plan and its samples): their decode samples' ratios to their prediction, and their prompts'.
fn ratios_in(root: &std::path::Path, keep: &dyn Fn(&serde_json::Value, &serde_json::Value) -> bool) -> (Vec<f64>, Vec<f64>) {
    let (mut decode, mut prefill) = (Vec::new(), Vec::new());
    for e in std::fs::read_dir(root.join("plans")).into_iter().flatten().flatten() {
        let Some(v) = std::fs::read_to_string(e.path()).ok().and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok()) else { continue };
        for s in v["observed"].as_array().into_iter().flatten().filter(|s| keep(&v, s)) {
            decode.extend(s["decode_ratio"].as_f64());
            prefill.extend(s["prefill_ratio"].as_f64());
        }
    }
    (decode, prefill)
}

/// What the runs on the GPU `gpu` reached against their predictions, from every plan that loaded on it here: the decode band (`Device::observed`) and the prompts'
/// median (`Device::observed_prefill`). Runs with a draft model are left out: their tokens a step are not the planner's.
pub fn observed_on(root: &std::path::Path, gpu: &str) -> (Option<[f64; 3]>, Option<f64>) {
    let (d, p) = ratios_in(root, &|v, s| v["plan"]["gpu"].as_str() == Some(gpu) && !s["drafted"].as_bool().unwrap_or(false));
    (band_of(d), band_of(p).map(|b| b[1]))
}

/// The decode band of `model` at `level` on `gpu` from its own runs here (its draft model's included), where it has any.
pub fn observed_for(root: &std::path::Path, model: &str, level: &str, gpu: &str) -> Option<([f64; 3], usize)> {
    let v = recall_in(root, model, level, gpu)?;
    let x: Vec<f64> = v["observed"].as_array()?.iter().filter_map(|s| s["decode_ratio"].as_f64()).collect();
    let n = x.len();
    band_of(x).map(|b| (b, n))
}

/// What a run is measured against: the plan's file, its prediction by positions (`speed_at`), its prompts' prediction, and whether a draft model drafts.
struct Watch {
    file: std::path::PathBuf,
    decode: Box<dyn Fn(f64) -> f64 + Send>,
    prefill: Option<f64>,
    drafted: bool,
}

static WATCH: std::sync::Mutex<Option<Watch>> = std::sync::Mutex::new(None);

/// The runs that follow are measured against the plan `plan` of `model` (`sh` on `m`, as `fit` predicts it), and kept with it in `root` (`observe`).
pub fn watch_in(root: &std::path::Path, model: &str, plan: &LoadPlan, sh: Shape, m: Machine) {
    if let (Some(w), Ok(mut g)) = (watch_of(root, model, plan, sh, m), WATCH.lock()) {
        *g = Some(w);
    }
}

fn watch_of(root: &std::path::Path, model: &str, plan: &LoadPlan, sh: Shape, m: Machine) -> Option<Watch> {
    let main = m.devices.iter().position(|d| plan.gpu.split(", ").next() == Some(d.name.as_str()))?;
    let in_vram = plan.experts.is_none() && plan.spread.as_ref().is_none_or(|s| s.host_bytes == 0 && s.dense == 0);
    let prefill = if in_vram { prefill_of(&sh, &m.devices[main]) } else { None };
    let lp = plan.clone();
    Some(Watch {
        file: memo_file(root, model, plan.level, &plan.gpu),
        decode: Box::new(move |positions| speed_at(&sh, &m, main, &lp, positions)),
        prefill,
        drafted: plan.draft > 0,
    })
}

/// On a model's first load on a GPU (no reply kept with its plan yet: `watch_in`'s), one reply timed right after the load and kept as its first run: a prompt of
/// up to 512 tokens and 33 made. From then on `fit`, `run` and `serve` predict this model on this GPU from what it measured, not from its class's figures.
pub fn first_use(e: &mut dyn crate::conv::Gen, vocab: usize) {
    let file = match WATCH.lock().ok().as_ref().and_then(|g| g.as_ref().map(|w| w.file.clone())) {
        Some(f) => f,
        None => return,
    };
    let seen = std::fs::read_to_string(&file)
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .is_some_and(|v| v["observed"].as_array().is_some_and(|o| !o.is_empty()));
    if !seen {
        if let Some((n, first_s, made, decode_s)) = first_reply(e, vocab) {
            observe(n, first_s, made, decode_s);
        }
    }
}

/// The reply `first_use` times: (prompt tokens, seconds to the first token, tokens made, seconds of the ones after it); None where the engine's context or
/// vocabulary is too small for it, or the reply fails.
fn first_reply(e: &mut dyn crate::conv::Gen, vocab: usize) -> Option<(usize, f64, usize, f64)> {
    let n = 512.min(e.capacity().saturating_sub(64));
    if n < 256 || vocab < 2048 {
        return None;
    }
    // (token ids past the first thousand, which hold the special ones; the reply's tokens are not looked at)
    let prompt: Vec<u32> = (0..n).map(|i| (1000 + (i * 7919) % (vocab - 1100)) as u32).collect();
    let o = crate::sched::Reply { params: Default::default(), max_new: 33, ignore_eos: true, logprobs: None, stop_ids: Vec::new() };
    let t0 = std::time::Instant::now();
    let (mut first, mut last, mut made) = (None, t0, 0usize);
    e.reply(&prompt, &o, &mut |_| {
        let now = std::time::Instant::now();
        first.get_or_insert(now);
        (last, made) = (now, made + 1);
        true
    })
    .ok()?;
    let f = first?;
    Some((n, (f - t0).as_secs_f64(), made, (last - f).as_secs_f64()))
}

/// The runs kept with a plan: the last of them.
const OBSERVED_KEPT: usize = 32;

/// A reply measured: `prompt` tokens read in `first_s` seconds to the first token, then `made` tokens in all, the ones after the first in `decode_s`. Kept with the
/// plan being watched (`watch_in`), against its prediction at the reply's positions: a decode where 16 tokens or more followed the first, a prompt of 256 tokens or more.
pub fn observe(prompt: usize, first_s: f64, made: usize, decode_s: f64) {
    if let Some(w) = WATCH.lock().ok().as_ref().and_then(|g| g.as_ref()) {
        keep(w, prompt, first_s, made, decode_s);
    }
}

/// `observe` against the plan of `w`.
fn keep(w: &Watch, prompt: usize, first_s: f64, made: usize, decode_s: f64) {
    let mut s = serde_json::json!({"prompt_tokens": prompt, "made": made, "drafted": w.drafted});
    if made > 16 && decode_s > 0.0 {
        let (rate, positions) = ((made - 1) as f64 / decode_s, prompt as f64 + made as f64 / 2.0);
        let predicted = (w.decode)(positions);
        s["decode_tok_s"] = serde_json::json!(rate);
        s["positions"] = serde_json::json!(positions);
        if predicted > 0.0 {
            s["decode_ratio"] = serde_json::json!(rate / predicted);
        }
    }
    if prompt >= 256 && first_s > 0.0 {
        let rate = prompt as f64 / first_s;
        s["prefill_tok_s"] = serde_json::json!(rate);
        if let Some(p) = w.prefill {
            s["prefill_ratio"] = serde_json::json!(rate / p);
        }
    }
    if s.get("decode_tok_s").is_none() && s.get("prefill_tok_s").is_none() {
        return;
    }
    let Some(mut v) = std::fs::read_to_string(&w.file).ok().and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok()) else { return };
    let mut seen = v["observed"].as_array().cloned().unwrap_or_default();
    seen.push(s);
    let from = seen.len().saturating_sub(OBSERVED_KEPT);
    v["observed"] = serde_json::Value::Array(seen.split_off(from));
    let _ = std::fs::write(&w.file, v.to_string());
}

// ---- the command ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

fn gb(b: u64) -> String {
    format!("{:.1} GiB", b as f64 / GIB)
}

fn where_(p: &Placement, m: &Machine) -> String {
    match p {
        Placement::Gpu(i) => format!("GPU {i} ({})", m.devices[*i].name),
        Placement::Split(g) => format!("split over GPUs {}", g.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(", ")),
        Placement::ExpertsInRam { cache, hit, disk: 0, .. } => {
            format!("experts in RAM, a {} cache on the GPU (~{:.0}% of expert reads), the CPU computes the rest", gb(*cache), hit * 100.0)
        }
        Placement::ExpertsInRam { cache, hit, disk, .. } => format!(
            "experts in RAM and {} on the disk, a {} cache on the GPU (~{:.0}% of expert reads), the CPU computes the rest",
            gb(*disk),
            gb(*cache),
            hit * 100.0
        ),
    }
}

/// The plan as the terminal shows it.
pub fn text(model: &str, sh: &Shape, m: &Machine, p: &Plan) -> String {
    let mut s = String::new();
    // (its tables read by rows apart: a token reads a few rows of each)
    let tables = if sh.tables > 0.0 { format!(" ({:.1}B of them tables a token reads a few rows of)", sh.tables / 1e9) } else { String::new() };
    let kind = if sh.is_moe() {
        format!("MoE, {:.1}B parameters{tables}, {:.1}B active", sh.params() / 1e9, sh.active() / 1e9)
    } else {
        format!("dense, {:.1}B parameters{tables}", sh.params() / 1e9)
    };
    s += &format!("{model}: {} ({kind}; up to {} tokens of context)\n", sh.arch, sh.max_ctx);
    for (i, d) in m.devices.iter().enumerate() {
        let measured = d.stream.map_or(String::new(), |m| format!(" ({m:.0} measured here)"));
        s += &format!("GPU {i}: {} · {} free of {} · {:.0} GB/s{measured}\n", d.name, gb(d.free), gb(d.total), d.bandwidth);
    }
    s += &format!("RAM: {} free · {} cores · backend: {}\n", gb(m.ram_free), m.cores, p.backend);
    match &p.chosen {
        Some(c) => {
            s += &format!(
                "Plan: {} ({:.1} bits) · {} · {} tokens of context · ~{:.0} tokens/s ({:.0}-{:.0}){}\n",
                c.level,
                c.bits,
                where_(c.placement.as_ref().unwrap(), m),
                c.context,
                c.tok_s,
                c.range.0,
                c.range.1,
                c.prefill.map_or(String::new(), |p| format!(" · prompts ~{p:.0} tokens/s"))
            );
            if let Some(l) = &c.load {
                s += &format!("  {}\n", l.text());
            }
            s += &format!("  speeds: {}\n", c.basis);
        }
        None => s += "Plan: none fits\n",
    }
    for o in &p.options {
        let mark = if p.chosen.as_ref().is_some_and(|c| c.level == o.level) { "  <- chosen" } else { "" };
        match &o.why {
            Some(w) => s += &format!("  {:<12} {:>9}  does not fit: {w}\n", o.level, gb(o.weights)),
            None => {
                s += &format!(
                    "  {:<12} {:>9}  {:>7} ctx  ~{:>4.0} tok/s  {}{mark}\n",
                    o.level,
                    gb(o.weights),
                    o.context,
                    o.tok_s,
                    where_(o.placement.as_ref().unwrap(), m)
                )
            }
        }
    }
    for n in &p.notes {
        s += &format!("note: {n}\n");
    }
    s += &format!(
        "(decode speeds are predictions for one user at about 1,000 positions: the GPU's memory rate {}, each class's floor a step and shares of that rate; prompts' a share of it too)\n",
        if m.devices.iter().any(|d| d.stream.is_some()) { "as measured here" } else { "by the maker's figure" },
    );
    s
}

/// `glyd-engine fit MODEL[:LEVEL] [--machine NAME|FILE] [--context N] [--users N]`: the plan, before anything is downloaded.
/// It runs `run`'s checks from the same table (src/support.rs): the model's family and the levels it runs at, its context
/// limits, and the machine's GPU, driver and CPU architecture. Exit 0 with a plan `run` would load; 3 where `run` would refuse
/// (the reason on stderr, after the plan where there is one); 1 where it could not be told.
pub fn fit_cmd(a: &crate::args::Args) -> i32 {
    match fit(a) {
        Ok(()) => 0,
        Err(r) => {
            crate::ui::Ui::new().refuse(&r);
            r.code
        }
    }
}

/// What the loaders' sizing is given of a machine: its main GPU's number, that GPU as a target, the GPUs the dynamic route uses (Ampere or newer) and whether they reach
/// each other directly (the driver's word for this computer, else a datacenter machine's NVLink).
#[allow(clippy::type_complexity)]
pub(crate) fn sizing_inputs(machine: &Machine, probed: bool) -> Option<(usize, Target, Vec<Target>, Box<dyn Fn(&[i32]) -> bool>)> {
    let main = (0..machine.devices.len()).min_by_key(|&i| (std::cmp::Reverse(machine.devices[i].free), i))?;
    let target = Target::of(&machine.devices[main], main as i32, Some(machine.ram_free));
    let devs: Vec<Target> =
        machine.devices.iter().enumerate().filter(|(_, d)| d.cc.0 >= 8).map(|(i, d)| Target::of(d, i as i32, Some(machine.ram_free))).collect();
    let linked = machine.devices.iter().all(|d| d.cc.1 == 0 && (8..=10).contains(&d.cc.0));
    Some((main, target, devs, Box::new(move |g: &[i32]| if probed { crate::dynamic::tp::reach(g) } else { linked })))
}

/// Each level's plan `run` would load for MODEL on `machine`, or the reason it would not (the same functions: `sizing::size`), in the family's order of
/// levels: None where no level of this family is sized by its loader yet, or the machine has no GPU. `probed`: the machine is this computer (its GPUs'
/// reach of each other is asked of the driver), else a datacenter machine's is taken as NVLink's.
#[allow(clippy::too_many_arguments)]
pub(crate) fn exact_options(
    a: &crate::args::Args,
    model: &str,
    config: &ModelConfig,
    level: Option<&'static str>,
    fam: &'static crate::support::Support,
    sh: &Shape,
    machine: &Machine,
    users: usize,
    probed: bool,
    positions: f64,
) -> Option<(Vec<Candidate>, Vec<crate::ui::Refusal>)> {
    let (main, target, devs, reach) = sizing_inputs(machine, probed)?;
    let tries: Vec<&'static str> = match level {
        Some(l) => vec![l],
        None => fam.levels_now(model),
    };
    let (mut refusals, mut options, mut sized) = (Vec::new(), Vec::new(), false);
    for l in tries {
        match sizing::size(a, model, config, l, &devs, &target, &reach, users) {
            Ok(Some(lp)) => {
                sized = true;
                // (a level the model is not had at sizes as the one it is: a local release asked for at hummingbird is its own penguin, listed once)
                let c = candidate_of(lp, sh, machine, main, positions);
                if !options.iter().any(|o: &Candidate| o.level == c.level) {
                    options.push(c);
                }
            }
            Ok(None) => {}
            Err(r) => {
                sized = true;
                options.push(Candidate { level: l, why: Some(r.what.clone()), ..Candidate::speeds(0.0, &machine.devices[main], None) });
                refusals.push(r);
            }
        }
    }
    sized.then_some((options, refusals))
}

/// A model's shape: its config's (parsed once), with its own tensors where its headers can be had (before any download, by ranges), else config.json's estimate
/// scaled to the index's total. `fit` and `run`'s degrade gate both price the ladder's steps on it.
pub fn shape_of(model: &str, config: &ModelConfig) -> Result<Shape, String> {
    let sh = Shape::from_model(config)?;
    Ok(match crate::sizes::Table::find(model) {
        Some(t) => sh.with_table(&t).with_tables(&t, &crate::dynamic::tables::ngram_names(config)),
        None => match index_total(model) {
            Some(total) => sh.scaled_to(total),
            None => sh,
        },
    })
}

/// What `fit` works out of its arguments: the model, the machine, the plan, and the refusal `run` would give (its reason, after the plan is printed).
pub struct Planned {
    pub model: String,
    pub name: String,
    pub family: &'static crate::support::Support,
    pub shape: Shape,
    pub machine: Machine,
    pub plan: Plan,
    pub refusal: Option<crate::ui::Refusal>,
    /// The degrade ladder from the level asked for (the family's first where none is): None where no level of the family is sized by its loader.
    pub ladder: Option<Ladder>,
    /// The positions the decode predictions are for (`Ask::positions`).
    pub positions: f64,
}

/// The words for the keys and values' format, the plan's (`LoadPlan::kv_format`: `fit`'s JSON) and the loaded engine's (`/v1/glyd/info`'s `precision.kv`):
/// "12-bit" where the executor holds them in its lossless 12-bit pages, else "bf16".
pub fn kv_format_of(twelve: bool) -> &'static str {
    if twelve {
        "12-bit"
    } else {
        "bf16"
    }
}

/// Whether the plan of `a` for a model of shape `sh` may hold its keys and values in the executor's 12-bit lossless pages: a server's where GLYD_KV=lossless asks for
/// them (`sizing::twelve_pages`), and its attention kinds have rings the pages take (`Shape::kv12_rings`: heads of 128, 256 or 512 channels, latents; heads of 64
/// and the compressed attention's entries stay bf16), whatever its family; the loaders that take them make the model's tables (src/cli.rs `twelve_bit_pages`).
fn takes_twelve(a: &crate::args::Args, sh: &Shape) -> bool {
    sizing::twelve_pages(a) && !sh.kv12_rings.is_empty()
}

pub fn planned(a: &crate::args::Args) -> Result<Planned, crate::ui::Refusal> {
    use crate::ui::Refusal;
    let given = a.need(1, "MODEL").map_err(|e| Refusal::new(e.msg, "glyd-engine fit MODEL[:LEVEL]"))?;
    let (model, layout) = levels::parse(given, a.has("lossy")).map_err(|e| Refusal::new(e.what, e.fix))?;
    let tagged = levels::split_tag(given).1.is_some() || a.has("lossy");
    let level = tagged.then(|| layout.map_or("penguin", levels::name_of));
    let machine = match a.flag("machine") {
        None => Machine::probe(),
        Some(n) => match Machine::preset(n) {
            Some(m) => m,
            None => Machine::from_json(&std::fs::read_to_string(n).map_err(|e| {
                Refusal::new(format!("--machine {n}: not a preset and not a file: {e}"), format!("The presets: {}", Machine::PRESETS.join(", ")))
            })?)
            .map_err(|e| Refusal::new(format!("{n}: {e}"), ""))?,
        },
    };
    let name = crate::cli::short_name(model);
    // (config.json parsed once, every plan below sized from that one parse)
    let config = crate::support::parsed(&crate::cli::config_text(model)?, &name)?;
    // the family as run treats it: a level it does not run at is refused in run's words
    let fam = match level {
        Some(l) => crate::support::check_in(&config, model, &name, l)?,
        None => crate::support::row(crate::support::family_in(&config, &name)?),
    };
    if fam.levels_now(model).is_empty() {
        crate::support::check_in(&config, model, &name, "penguin")?;
    }
    let sh = shape_of(model, &config).map_err(|e| Refusal::unsupported(format!("{name}: {e}"), ""))?;
    let sh = if takes_twelve(a, &sh) { sh.in_twelve() } else { sh };
    let context: Option<usize> = a.flag("context").map(|c| c.parse()).transpose().map_err(|_| Refusal::new("--context takes a number", "--context 8192"))?;
    crate::support::context_check(&name, fam, sh.max_ctx, context)?;
    let users: usize = a.num("users", 1usize)?;
    let positions = crate::tune!("GLYD_FIT_POSITIONS").and_then(|v| v.parse().ok()).unwrap_or(0);
    let ask = Ask { level, context, users, family: Some(fam), qat: levels::is_qat(model), positions };
    let mut p = plan(&sh, &machine, ask);
    let mut refusal = None;
    // where run's own sizing covers the family: the plan `run` would load at each level, or the refusal it would give (the same functions: `sizing::size`)
    let probed = a.flag("machine").is_none();
    if let Some((options, refusals)) = exact_options(a, model, &config, level, fam, &sh, &machine, users, probed, ask.positions()) {
        let max_ctx = context.unwrap_or(fam.default_ctx.unwrap_or(TARGET_CTX).min(sh.max_ctx)).min(fam.max_ctx.unwrap_or(usize::MAX));
        p.chosen = choose(&options, max_ctx);
        p.options = options;
        p.notes.retain(|n| !n.starts_with("nothing fits"));
        if p.chosen.is_none() {
            p.notes.push("nothing fits this machine: a smaller model, or more memory (or other machines: glyd cluster, ROADMAP M4)".into());
            // (what `run` says of the level it would take: the first one asked, else the lossless one)
            refusal = refusals.into_iter().next();
        }
    }
    // (what this model's own runs here reached at a level, where it ran before: its options' speeds and bands)
    if probed {
        let root = crate::store::cache_root();
        if let Some(d) = machine.devices.iter().max_by_key(|d| d.free) {
            for o in p.options.iter_mut().chain(p.chosen.as_mut()) {
                if let Some((b, n)) = observed_for(&root, model, o.level, &d.name) {
                    (o.tok_s, o.range) = (o.prior * b[1], (o.prior * b[0], o.prior * b[2]));
                    o.basis = format!("measured: {n} replies of this model at this level on this GPU");
                }
            }
        }
    }
    // (the machine's refusal first: `run` checks the GPU before it sizes anything)
    let refusal = crate::support::machine(&machine).or(refusal).or_else(|| {
        p.chosen.is_none().then(|| {
            Refusal::unsupported(
                format!("{name} does not fit this machine at any level {} runs", fam.name),
                "A smaller model, more memory, or less context: --context 4096",
            )
        })
    });
    // the ladder from the level asked for, as `run` takes it: the steps printed with their costs, and the plan it loads (what `fit` shows first of all)
    let asked = level.or_else(|| fam.levels_now(model).first().copied());
    let ladder = asked.and_then(|asked| {
        let (_, target, devs, reach) = sizing_inputs(&machine, probed)?;
        let want = context.unwrap_or_else(|| fam.default_ctx.unwrap_or(LOSSY_CTX).min(sh.max_ctx));
        let size = |l: &'static str, ctx: usize| {
            let q = if ctx == want { a.clone() } else { a.with(&["context", "ctx"], ctx.to_string()) };
            sizing::size(&q, model, &config, l, &devs, &target, &*reach, users)
        };
        let lad = ladder(&sh, &machine, fam, model, asked, want, &size);
        (lad.plan.is_some() || lad.refusal.is_some()).then_some(lad)
    });
    Ok(Planned { model: model.to_string(), name, family: fam, shape: sh, machine, plan: p, refusal, ladder, positions: ask.positions() })
}

fn fit(a: &crate::args::Args) -> Result<(), crate::ui::Refusal> {
    let f = planned(a)?;
    let (model, p) = (f.model.as_str(), &f.plan);
    // each level's quality against the lossless level, as its card says (R14: src/quality.rs)
    let shown: Vec<&str> = p.options.iter().map(|o| o.level).collect();
    let (quality_text, quality_json) = crate::quality::fit_block(model, &shown);
    if a.has("json") {
        let mut v: serde_json::Value = match &p.chosen {
            Some(c) => serde_json::from_str(&c.json(model, &f.machine)).unwrap_or_default(),
            None => serde_json::json!({"schema": SCHEMA, "model": model, "plan": null, "notes": p.notes}),
        };
        v["degrade"] = f.ladder.as_ref().map_or(serde_json::Value::Null, Ladder::json);
        v["positions"] = serde_json::json!(f.positions);
        v["quality"] = quality_json;
        println!("{v}");
    } else {
        print!("{}", text(model, &f.shape, &f.machine, p));
        if let Some(l) = &f.ladder {
            print!("{}", l.report());
        }
        if !shown.is_empty() {
            print!("{quality_text}");
        }
        if let Some(n) = p.chosen.as_ref().and_then(|c| c.load.as_ref()).and_then(|lp| users_note(a.num("users", 1usize).unwrap_or(1), lp)) {
            println!("note: {n}");
        }
        if let Some(n) = p.chosen.as_ref().and_then(|c| remembered(&crate::store::cache_root(), model, c)) {
            println!("note: {n}");
        }
        if let Some(n) = f.machine.devices.iter().find_map(|d| crate::machine::cuda12_note(d.driver, crate::dev::has_kernels12())) {
            println!("note: {n}");
        }
        if f.family.hummingbird_packed_only && p.chosen.as_ref().is_some_and(|c| c.level == "hummingbird") {
            // (by the row's route: the one-GPU route's hummingbird level is a pre-packed download alone; the dynamic route's comes from the experts' codes
            // pack, or is packed here from the experts' usage)
            if f.family.hummingbird_one_gpu() {
                println!("note: the hummingbird level comes as a pre-packed download: glyd/{}-hummingbird", f.name);
            } else if f.family.route.expert_cache == crate::support::ExpertCache::Codes {
                println!(
                    "note: the hummingbird level is a codes pack of the experts beside the weights (DIR/pack): the pre-packed glyd/{}-hummingbird's (condor/tools/download.sh); --pack precise, best or fast plans each",
                    f.name
                );
            } else {
                println!(
                    "note: the hummingbird level is packed ahead: the pre-packed glyd/{}-hummingbird, or glyd-engine calibrate-experts MODEL usage.json, then glyd-engine pack MODEL DIR --layout hummingbird --usage usage.json",
                    f.name
                );
            }
        }
    }
    match f.refusal {
        Some(r) => Err(r),
        None => Ok(()),
    }
}

/// What `fit` says where the plan `lp` holds fewer sequences at once than the `asked` users (a plan never holds silently less than asked): a model in stages
/// over GPUs, or with every layer across them, runs one sequence at a time.
fn users_note(asked: usize, lp: &LoadPlan) -> Option<String> {
    let why = match &lp.spread {
        Some(s) if s.across => ": every layer across GPUs runs one sequence at a time",
        Some(s) if !s.cuts.is_empty() => ": its layers in stages over GPUs run one sequence at a time",
        _ => "",
    };
    (lp.users < asked).then(|| format!("the plan holds {} of the {asked} users asked for at once{why}", lp.users))
}

/// The checkpoint's bytes from model.safetensors.index.json, where it is here or on the Hub.
fn index_total(model: &str) -> Option<f64> {
    let file = "model.safetensors.index.json";
    let text = std::fs::read_to_string(std::path::Path::new(model).join(file))
        .ok()
        .or_else(|| glyd_gpu::save::Source::find(model).ok().and_then(|s| std::fs::read_to_string(s.dir.join(file)).ok()))
        .or_else(|| {
            (!crate::hub::offline() && model.matches('/').count() == 1).then(|| crate::hub::Hub::from_env().text(model, "main", file).ok()).flatten()
        })?;
    num(json::parse(&text).ok()?.get("metadata")?, "total_size")
}

// ---- the plan each path loads ------------------------------------------------------------------------------------------------------------------------------------------------------

/// The loaders' sizing (P2f): the plan `run`, `serve`, `preflight` and `fit` load and print.
pub mod sizing {
    use super::{reserve_of, Kind, LoadPlan, Target, HOST_MARGIN, LOSSY_CTX, LOSSY_MIN_CTX, LOSSY_WORK, QWEN3_WORK};
    use crate::args::Args;
    use crate::cli::{cannot, draft_dir, level, level_name, qat_wren, shape_checks, short_name, supported_in, unrun_fix, users_of};
    use crate::dev::Gpu;
    use crate::ffi::Blas;
    use crate::levels::{self, layout_for};
    use crate::offload::CachePlan;
    use crate::qwen35::Pack35;
    use crate::runtime::config::ModelConfig;
    use crate::store;
    use crate::support::{Across, ExpertCache, RamLinears, Sizing, Support};
    use crate::ui::{gib, Done, Refusal};
    use crate::weights::{packed_bytes, Config, Layout};
    use glyd_gpu::save::Source;
    use std::path::{Path, PathBuf};

    /// The context a chat should have where the GPU has room, and the least worth running.
    pub const CTX_WANT: usize = crate::support::DEFAULT_CTX;
    pub const CTX_MIN: usize = 4096;

    pub fn bits(layout: Layout) -> f64 {
        match layout {
            Layout::Tiered => 10.80,
            Layout::Twelve => 12.04,
            Layout::Compact => 6.80,                // (Candidate A, lossy: run --lossy)
            Layout::Int6 | Layout::Int6Row => 6.53, // (six-bit codes, a bf16 scale a run of 32, a step's unread word: 6 + 0.5 + 0.03; a row layout's the same bytes)
            Layout::Int5 | Layout::Int5Row => 5.53, // (five-bit codes, a bf16 scale a run of 32, a step's unread word: 5 + 0.5 + 0.03)
            Layout::Int4 | Layout::Int4Row => 4.53, // (four-bit codes, a bf16 scale a run of 32, a step's unread word: 4 + 0.5 + 0.03)
            Layout::Int4H => 4.53,                  // (four-bit codes, a 16-bit scale a run of 32, a step's unread word, as Int4)
            Layout::Int3 => 3.53,                   // (three-bit codes, a bf16 scale a run of 32, a step's unread word: 3 + 0.5 + 0.03)
            Layout::Cb3Row => 3.625,                // (three-bit codes, a bf16 scale a run of 32 and eight bf16 entries a row of 1,024: 3 + 0.5 + 0.125)
            Layout::Cb2Row => 2.5625,               // (two-bit codes, a bf16 scale a run of 32 and four bf16 entries a row of 1,024: 2 + 0.5 + 0.0625)
            // (a layer's experts in the recipe's shares: `moe_hb::HB40_THREE_BIT_PERCENT` of them in the three-bit rows and the rest in the two-bit ones, to a hundredth of a
            // bit as the other levels' are said; the directory's own bytes are what a plan uses where it has them)
            Layout::Hummingbird => {
                let three = crate::moe_hb::HB40_THREE_BIT_PERCENT as f64 / 100.0;
                ((three * bits(Layout::Cb3Row) + (1.0 - three) * bits(Layout::Cb2Row)) * 100.0).round() / 100.0
            }
        }
    }

    /// The weights on the GPU in `layout` (every matrix packed, the embedding too), with the norms and tables.
    pub fn weights(cfg: &Config, layout: Layout) -> u64 {
        (cfg.matrix_bytes() as f64 * bits(layout) / 16.0) as u64 + (cfg.layers * cfg.hidden * 4 * 2) as u64 + (cfg.max_pos * cfg.head_dim * 4) as u64
    }

    /// A token's keys and values in the executor's cache, all layers: bf16, or in its 12-bit lossless pages (`twelve`) where they take the layers' heads
    /// (`kv12_token_bytes`; what they spend once is `Once`).
    pub fn kv_per_token(cfg: &Config, twelve: bool) -> u64 {
        let layer = match dense_ring(cfg, twelve) {
            Some(s) => kv12_token_bytes(s.heads, s.channels, s.one),
            None => (2 * cfg.kv_heads * cfg.head_dim * 2) as u64,
        };
        cfg.layers as u64 * layer
    }

    /// A plain dense model's layer's 12-bit ring where its keys and values are in the 12-bit pages (`twelve`) and the pages take its heads; else None (bf16).
    fn dense_ring(cfg: &Config, twelve: bool) -> Option<crate::dynamic::kv12::KvShape> {
        crate::dynamic::kv12::ring_shape(cfg.kv_heads, cfg.head_dim, cfg.heads, 0, false).filter(|_| twelve)
    }

    /// The 12-bit rings of a plain dense model whose keys and values are in its 12-bit pages (`twelve`; none where the pages do not take its heads).
    pub fn dense_rings(cfg: &Config, twelve: bool) -> Vec<crate::dynamic::kv12::KvShape> {
        dense_ring(cfg, twelve).map_or(Vec::new(), |s| vec![s; cfg.layers])
    }

    /// A token's bytes in a ring of the executor's 12-bit lossless pages as the planner sizes them: the pool's own (`kv12::token_bytes`: a sixteenth of the
    /// ring's page, its escape room and its share of the raw slabs; `kv12check` holds it to the pages' own by the library's sizes, never less).
    pub fn kv12_token_bytes(heads: usize, channels: usize, one: bool) -> u64 {
        crate::dynamic::kv12::token_bytes(heads, channels, one)
    }

    /// What a plan's keys and values take once, whatever its users, at a context (`Needs::once`): the 12-bit pages' pools' `kv12::once_bytes` (one sequence's
    /// raw slabs, the unpacked ring, the escape pool's least) for prompt chunks of `chunk`, nothing for bf16 rings (no `rings`); and `fixed`, what the pools take
    /// whatever the users and the context (`rows::SeqRoom`): each group's block 0, which no sequence holds, and where the engine keeps a model's states at block
    /// boundaries, the pool of those copies.
    #[derive(Clone, Debug, Default)]
    pub(crate) struct Once {
        pub rings: Vec<crate::dynamic::kv12::KvShape>,
        pub chunk: usize,
        pub fixed: u64,
    }

    impl Once {
        pub fn at(&self, ctx: usize) -> u64 {
            crate::dynamic::kv12::once_bytes(&self.rings, ctx, self.chunk) + self.fixed
        }
    }

    /// The pools' block 0 of groups whose rings take `bytes` a token in bf16 (`rows::SeqRoom::block`: a block of 16 tokens; a 12-bit page's is smaller).
    pub(crate) fn block_zero(bytes: &[u64]) -> u64 {
        bytes.iter().map(|b| 16 * b).sum()
    }

    /// What a model's sequences take on the executor beside their tokens' keys and values (`rows::SeqRoom`), its states `state` bytes a sequence and its pools' groups
    /// `groups` bytes a token each. A server's (`served`: the scheduler's sequences in pools of blocks): each sequence's states and, where the engine keeps them
    /// (`marks_kept`), a copy at its mark; once, the pool of copies at block boundaries (`snaps::count`) and each group's block 0 (`block_zero`). A run's one
    /// sequence (`run` answers by itself, its rings made at the load): its states, nothing more. A sequence's bytes, and what is taken once.
    pub(crate) fn pools_of(state: u64, groups: &[u64], served: bool) -> (u64, u64) {
        if !served {
            return (state, 0);
        }
        let zero = block_zero(groups);
        if state == 0 || !crate::dynamic::marks_kept() {
            return (state, zero);
        }
        (2 * state, zero + crate::dynamic::snaps::count(state as usize) as u64 * state)
    }

    /// Whether the plan of `a` is a server's, whose sequences the scheduler holds in pools of blocks (`Dyn::serve_rows`): `serve`'s, and `fit`'s where `--users` is
    /// given; `run` answers one prompt at a time by itself.
    pub(crate) fn served(a: &Args) -> bool {
        match a.pos.first().map(String::as_str) {
            Some("serve") => true,
            Some("fit") => a.has("users"),
            _ => false,
        }
    }

    /// The sequences a plan holds at once: a server's `--users` (`served`; 4 where `serve` is given none: `users_of`), else one.
    pub(crate) fn users_asked(a: &Args) -> Done<usize> {
        if served(a) {
            users_of(a)
        } else {
            Ok(1)
        }
    }

    /// The activations of a chunk of `chunk` tokens (the buffers of one forward pass), the decoded matrix a cuBLAS product reads, and the fixed workspaces.
    pub fn work(cfg: &Config, chunk: usize, cublas: bool) -> u64 {
        let (hid, hd) = (cfg.hidden as u64, cfg.head_dim as u64);
        let qkv = (cfg.heads as u64 + 2 * cfg.kv_heads as u64) * hd;
        let per_token = 2 * (hid * 4 + qkv + 2 * cfg.heads as u64 * hd + cfg.kv_heads as u64 * hd + 3 * cfg.inter as u64) + 16;
        let decoded = if cublas { 2 * cfg.inter as u64 * hid * 2 } else { 0 };
        chunk as u64 * per_token + decoded + QWEN3_WORK
    }

    /// The layout for this GPU, and why: the plugin's `best_layout` for a dense model. The tiered layout where it is the fastest (`smallest_is_fastest`) or the only one that fits (2 GiB of the GPU kept
    /// for activations and the cache), else the 12-bit one.
    pub fn best_layout(cc: (i32, i32), total_mem: u64, matrix_bytes: u64, other_bytes: u64) -> (Layout, String) {
        let room = total_mem.saturating_sub(2 << 30);
        let (tiered, twelve) = ((matrix_bytes as f64 * 10.80 / 16.0) as u64 + other_bytes, (matrix_bytes as f64 * 12.04 / 16.0) as u64 + other_bytes);
        if twelve > room && room >= tiered {
            return (
                Layout::Tiered,
                format!(
                    "only the mma layout fits (the smallest: {:.1} GB; mma12 {:.1} GB, room {:.1} GB)",
                    tiered as f64 / 1e9,
                    twelve as f64 / 1e9,
                    room as f64 / 1e9
                ),
            );
        }
        if crate::weights::smallest_is_fastest(cc) {
            return (Layout::Tiered, "Ada: mma, the smallest layout, is as fast as mma12 (measured)".to_string());
        }
        (Layout::Twelve, "mma12, the faster layout, on this GPU (measured)".to_string())
    }

    /// The layout of a run: the one asked for with `--layout` (what the user wants wins, whether or not it fits: the caller's fit check says so), else this GPU's.
    pub fn choose_layout(asked: Option<Layout>, cc: (i32, i32), total_mem: u64, matrix_bytes: u64, other_bytes: u64) -> (Layout, String) {
        match asked {
            Some(l) => (l, "as asked".to_string()),
            None => best_layout(cc, total_mem, matrix_bytes, other_bytes),
        }
    }

    /// `host` bytes in the computer's memory split over the stages that `cuts` make of layers holding `per_layer` bytes each (of what goes there): each stage's
    /// share by its layers', the rounding the last stage's, so that they add up to `host` exactly; all of it the first's where no layer holds any.
    pub fn stage_shares(per_layer: &[u64], cuts: &[usize], host: u64) -> Vec<u64> {
        let bounds: Vec<usize> = std::iter::once(0).chain(cuts.iter().copied()).chain(std::iter::once(per_layer.len())).collect();
        let n = per_layer.len();
        let of: Vec<u128> = bounds.windows(2).map(|w| per_layer[w[0].min(n)..w[1].clamp(w[0].min(n), n)].iter().map(|&b| b as u128).sum()).collect();
        let all: u128 = of.iter().sum();
        let mut out: Vec<u64> = of.iter().map(|&b| if all == 0 { 0 } else { (host as u128 * b / all) as u64 }).collect();
        let given: u64 = out.iter().sum();
        let at = if all == 0 { 0 } else { out.len() - 1 };
        out[at] += host - given;
        out
    }

    // ---- the dynamic route's layers' bytes (moved from src/dynamic/split.rs, which re-exports them) ----

    /// An FP8 matrix's bytes on the GPU, in hundredths of the release's: layout H's plain packs (an HBM GPU's default) 9 bits a value (a patch's row blocks 0-2
    /// in its 7-bit core, the 4th in var: 112.5%); coded packs about 7.1 bits (layout H; layout c 6.75), 90 a bound. ponytail: fixed shares; the matrices' own
    /// sizes (`f8_size`) where a release codes worse.
    pub const PLAIN_PCT: u64 = 113;
    pub const CODED_PCT: u64 = 90;
    /// A bf16 matrix's bytes on the GPU at the penguin level, in hundredths of the release's: the smallest lossless layout (tiered, about 10.8 bits a value)
    /// 72, the 12-bit one (about 12.0 and its exceptions) 78. The 12-bit layout's one-token product reads at about 80% of an H100's rate where the tiered
    /// one's reads at 30-45% (research/fp4-one-h100-2026-10-22), so an HBM GPU holds bf16 matrices in it where they fit.
    pub const TIERED_PCT: u64 = 72;
    pub const TWELVE_PCT: u64 = 78;

    /// The bytes each of the `nl` layers holds on a GPU, and the rest's (embedding, head, final norm): the release's tensors as they are, an FP8 matrix at
    /// `f8_pct` hundredths of its bytes, a bf16 matrix at penguin's 72% (about 11.5 bits a weight, an upper bound), a latent attention's small tensor released
    /// in FP8 at its values in bf16 (twice its bytes: `mla::f8_small_name`). Not counted: the tensors no description loads (a multi-token-prediction layer, a
    /// vision or audio tower).
    pub fn layer_bytes(dir: &Path, nl: usize, f8_pct: u64) -> crate::ffi::Result<(Vec<u64>, u64)> {
        layer_bytes_where(dir, nl, f8_pct, TIERED_PCT, &|_| true)
    }

    /// As `layer_bytes`, of the tensors whose names `keep` keeps.
    pub fn layer_bytes_where(dir: &Path, nl: usize, f8_pct: u64, bf16_pct: u64, keep: &dyn Fn(&str) -> bool) -> crate::ffi::Result<(Vec<u64>, u64)> {
        layer_bytes_with(dir, nl, f8_pct, bf16_pct, keep, true)
    }

    /// `layer_bytes_where`, a hummingbird directory's experts (their files of their own) counted in each layer's bytes only where `experts`.
    pub fn layer_bytes_with(
        dir: &Path,
        nl: usize,
        f8_pct: u64,
        bf16_pct: u64,
        keep: &dyn Fn(&str) -> bool,
        experts: bool,
    ) -> crate::ffi::Result<(Vec<u64>, u64)> {
        let table = crate::sizes::Table::from_dir(dir).map_err(crate::ffi::Error)?;
        let (mut per, base) = layer_bytes_table(&table, nl, f8_pct, bf16_pct, keep);
        if experts {
            per.iter_mut().zip(hb_layer_bytes(dir, nl)?).for_each(|(p, b)| *p += b);
        }
        Ok((per, base))
    }

    /// The bytes of routed experts each of the `nl` layers of a hummingbird directory holds in files of its own (src/hb_pack.rs: by the description's block, decoder layer l's are block
    /// 2 l + 1); zeros for any other directory.
    pub fn hb_layer_bytes(dir: &Path, nl: usize) -> crate::ffi::Result<Vec<u64>> {
        let mut per = vec![0u64; nl];
        if let Some(hb) = crate::hb_pack::HbDir::open(dir)? {
            for (l, bytes) in hb.layer_bytes() {
                if l < nl {
                    per[l] += bytes;
                }
            }
        }
        Ok(per)
    }

    /// `layer_bytes_where` of a table of tensors (src/sizes.rs: a directory's, or the Hub's by ranges before the files are here).
    pub fn layer_bytes_table(table: &crate::sizes::Table, nl: usize, f8_pct: u64, bf16_pct: u64, keep: &dyn Fn(&str) -> bool) -> (Vec<u64>, u64) {
        // (an FP8 matrix whose E8M0 scales are of blocks other than 128 x 128 (DeepSeek-V4.1's 32 x 32), which the library's FP8 products do not take: the loader
        // holds its values in bf16, packed as a bf16 matrix is; one with a scale a row's run (an n-gram table, whose rows are looked up), as the bf16 it decodes to)
        let shapes: std::collections::HashMap<&str, &[u64]> = table.rows.iter().map(|r| (r.name.as_str(), &r.shape[..])).collect();
        let held_bf16 = |r: &crate::sizes::Row| -> Option<bool> {
            let sc = shapes.get(r.name.replace(".weight", ".scale").as_str()).filter(|_| r.name.ends_with(".weight") && r.shape.len() == 2)?;
            let b = crate::formats::scale_block(r.shape[0] as usize, r.shape[1] as usize, sc)?;
            (b != [128, 128]).then_some(b[0] == 1)
        };
        table.per_layer(nl, &|r| keep(&r.name), &|r| {
            let matrix = r.shape.len() == 2 && r.shape.iter().all(|&d| d >= 256);
            match r.dtype.as_str() {
                // (a latent attention's kv_b_proj and indexer products released in FP8: the loader holds their E4M3 values in bf16, `Part::scales`)
                "F8_E4M3" if crate::dynamic::mla::f8_small_name(&r.name) => r.bytes * 2,
                "F8_E4M3" => match held_bf16(r) {
                    Some(true) => r.bytes * 2,
                    Some(false) if matrix => r.bytes * 2 * bf16_pct / 100,
                    Some(false) => r.bytes * 2,
                    None if matrix => r.bytes * f8_pct / 100,
                    None => r.bytes,
                },
                "BF16" if matrix => r.bytes * bf16_pct / 100,
                _ => r.bytes,
            }
        })
    }

    /// The linears the loader of the family the table names `family` keeps in the computer's memory (`Route::ram_linears`; none for a name with no row).
    fn ram_linears(family: &str) -> RamLinears {
        crate::support::row_named(family).map_or(RamLinears::None, |r| r.route.ram_linears)
    }

    /// Each of the model's `nl` layers' bytes that its family's loader can keep in the computer's memory (src/dense_ram.rs: its packed linears there), at the packing `layer_bytes_where` counts: the linears the
    /// family's row names (`Route::ram_linears`: every matrix of a layer's own products the executor's loader packs); zeros for a row that keeps none there.
    pub fn ram_bytes(dir: &Path, nl: usize, family: &str, f8_pct: u64, bf16_pct: u64) -> crate::ffi::Result<Vec<u64>> {
        if ram_linears(family) == RamLinears::None {
            return Ok(vec![0; nl]);
        }
        Ok(ram_bytes_table(&crate::sizes::Table::from_dir(dir).map_err(crate::ffi::Error)?, nl, family, f8_pct, bf16_pct))
    }

    /// `ram_bytes` of a table of tensors (src/sizes.rs: before the files are here too).
    pub fn ram_bytes_table(table: &crate::sizes::Table, nl: usize, family: &str, f8_pct: u64, bf16_pct: u64) -> Vec<u64> {
        layer_bytes_table(table, nl, f8_pct, bf16_pct, &|n| in_ram(family, n)).0
    }

    /// Whether the tensor `name` is one of the matrices `family`'s loader keeps in the computer's memory for a layer there (src/dense_ram.rs).
    pub(crate) fn in_ram(family: &str, name: &str) -> bool {
        linear_in_ram(ram_linears(family), name)
    }

    /// `in_ram` by the linears a loader keeps there (`Route::ram_linears`).
    fn linear_in_ram(which: RamLinears, name: &str) -> bool {
        match which {
            RamLinears::Products => {
                let attention = name.contains(".self_attn.")
                    && !name.contains(".indexer.")
                    && !name.ends_with(".kv_b_proj.weight")
                    && (name.ends_with("_proj.weight") || name.ends_with(".kv_a_proj_with_mqa.weight"));
                let mlp =
                    ["gate_proj", "up_proj", "down_proj", "pre_expert_proj", "post_expert_proj"].iter().any(|x| name.ends_with(&format!(".mlp.{x}.weight")));
                attention || mlp
            }
            RamLinears::None => false,
        }
    }

    /// (the dynamic route's cuts and places, moved from src/dynamic/split.rs, which re-exports them)
    use crate::dynamic::split;

    /// The bytes `layer_bytes` gives and the cut `plan_cuts` makes over `room`: on HBM GPUs (`hbm`) the FP8 matrices in layout H's plain packs and, for a loader
    /// that takes it (`twelve`: Kimi-K3's, whose `Opts::layout` is its bf16 matrices' alone), the bf16 ones in the 12-bit layout where they fit (their faster
    /// products), else the bf16 ones tiered, then the FP8 ones coded too (coded: true third, `Lib::f8h_plain(Some(false))` before the load; 12-bit: true fourth,
    /// `Layout::Twelve` the load's); elsewhere coded and tiered, their only layouts. No cut where nothing fits.
    #[allow(clippy::type_complexity)]
    pub fn fit(
        dir: &Path,
        nl: usize,
        room: &[(i32, u64)],
        hbm: bool,
        twelve: bool,
    ) -> crate::ffi::Result<((Vec<u64>, u64), Option<(Vec<usize>, Vec<i32>)>, bool, bool)> {
        Ok(fit_table_with(&crate::sizes::Table::from_dir(dir).map_err(crate::ffi::Error)?, nl, room, hbm, twelve, &split::hb_layer_bytes(dir, nl)?, &[]))
    }

    /// `fit` of a table of tensors (src/sizes.rs).
    #[allow(clippy::type_complexity)]
    pub fn fit_table(
        table: &crate::sizes::Table,
        nl: usize,
        room: &[(i32, u64)],
        hbm: bool,
        twelve: bool,
    ) -> ((Vec<u64>, u64), Option<(Vec<usize>, Vec<i32>)>, bool, bool) {
        fit_table_with(table, nl, room, hbm, twelve, &[], &[])
    }

    /// `fit_table`, each layer's bytes more by `extra`'s (a hummingbird directory's experts, which no table lists: `split::hb_layer_bytes`), a stage beginning
    /// only at a layer `starts` keeps (`plan_cuts_at`; every layer where it is empty).
    #[allow(clippy::type_complexity)]
    pub fn fit_table_with(
        table: &crate::sizes::Table,
        nl: usize,
        room: &[(i32, u64)],
        hbm: bool,
        twelve: bool,
        extra: &[u64],
        starts: &[bool],
    ) -> ((Vec<u64>, u64), Option<(Vec<usize>, Vec<i32>)>, bool, bool) {
        let bytes = |f8: u64, bf16: u64| {
            let (mut per, base) = split::layer_bytes_table(table, nl, f8, bf16, &|_| true);
            per.iter_mut().zip(extra).for_each(|(p, e)| *p += e);
            (per, base)
        };
        if hbm {
            for (f8, twelve) in [(split::PLAIN_PCT, twelve), (split::PLAIN_PCT, false), (split::CODED_PCT, twelve)] {
                let b = bytes(f8, if twelve { split::TWELVE_PCT } else { split::TIERED_PCT });
                if let Some(c) = plan_cuts_at(&b.0, b.1, room, starts) {
                    return (b, Some(c), f8 == split::CODED_PCT, twelve);
                }
            }
        }
        let coded = bytes(split::CODED_PCT, split::TIERED_PCT);
        let cut = plan_cuts_at(&coded.0, coded.1, room, starts);
        let switched = hbm && cut.is_some();
        (coded, cut, switched, false)
    }

    /// A model whose experts the GPUs do not all hold (Kimi-K3's 1.4 TB on 8x 80 GB): its layers over the GPUs of `room` in order, each layer taking its bytes
    /// besides its routed experts (`dense`) and, where it has experts (`moe`), places for the same number of them, `expert` bytes each, the most that lets every
    /// layer fit (`base`, the rest, half on the first GPU and half on the last); then each GPU's room left over as more places for its own layers, as many as a
    /// layer has at most. The cuts, their GPUs and each stage's places a layer; None where fewer than `least` a layer fit.
    pub fn places_plan(
        dense: &[u64],
        base: u64,
        moe: &[bool],
        room: &[(i32, u64)],
        expert: u64,
        experts: usize,
        least: usize,
    ) -> Option<(Vec<usize>, Vec<i32>, Vec<usize>)> {
        places_plan_at(dense, base, moe, room, expert, experts, least, &[])
    }

    /// `places_plan` where a stage begins only at a layer `may_start` keeps (the first always; every layer where it is empty, as `plan_cuts_at` takes it):
    /// the layers from each such layer to the next kept on one GPU.
    #[allow(clippy::too_many_arguments)]
    pub fn places_plan_at(
        dense: &[u64],
        base: u64,
        moe: &[bool],
        room: &[(i32, u64)],
        expert: u64,
        experts: usize,
        least: usize,
        may_start: &[bool],
    ) -> Option<(Vec<usize>, Vec<i32>, Vec<usize>)> {
        let n = dense.len();
        let firsts: Vec<usize> = (0..n).filter(|&l| l == 0 || may_start.get(l).copied().unwrap_or(true)).collect();
        let group = |i: usize| firsts[i]..firsts.get(i + 1).copied().unwrap_or(n);
        let foot = |i: usize, p: u64| group(i).map(|l| dense[l] + if moe[l] { p * expert } else { 0 }).sum::<u64>();
        // (the groups of layers in order, each GPU filled to its room, with p places a layer: where each stage starts; None where the GPUs do not hold them)
        let assign = |p: u64| -> Option<Vec<usize>> {
            let (mut starts, mut g, mut used) = (vec![0usize], 0usize, base / 2);
            for i in 0..firsts.len() {
                if used + foot(i, p) > room.get(g)?.1 {
                    g += 1;
                    starts.push(firsts[i]);
                    used = 0;
                }
                used += foot(i, p);
                if used > room.get(g)?.1 {
                    return None;
                }
            }
            (used + base - base / 2 <= room[g].1).then_some(starts)
        };
        let (mut lo, mut hi) = (least as u64, experts as u64);
        assign(lo)?;
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            if assign(mid).is_some() {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let starts = assign(lo)?;
        let bounds: Vec<usize> = starts.iter().copied().chain(std::iter::once(n)).collect();
        let stages = starts.len();
        let mut places = Vec::new();
        for (k, w) in bounds.windows(2).enumerate() {
            let held = dense[w[0]..w[1]].iter().sum::<u64>() + if k == 0 { base / 2 } else { 0 } + if k + 1 == stages { base - base / 2 } else { 0 };
            let layers = moe[w[0]..w[1]].iter().filter(|&&m| m).count().max(1) as u64;
            places.push(((room[k].1.saturating_sub(held) / (layers * expert.max(1))) as usize).clamp(lo as usize, experts));
        }
        Some((starts[1..].to_vec(), room[..stages].iter().map(|r| r.0).collect(), places))
    }

    /// `places_plan`, and where it finds no room on one GPU, the last layers' dense matrices in the computer's memory (src/dense_ram.rs; `ram`: each layer's bytes that can be there), as few layers as fit: their bytes off the GPU,
    /// one slot of the largest of them on it. The cuts, their GPUs, each stage's places, how many layers are in the computer's memory, and the bytes that leaves off the GPU (theirs less the slot).
    #[allow(clippy::type_complexity)]
    pub fn places_plan_dense(
        dense: &[u64],
        base: u64,
        moe: &[bool],
        room: &[(i32, u64)],
        expert: u64,
        experts: usize,
        least: usize,
        ram: &[u64],
        may_start: &[bool],
    ) -> Option<(Vec<usize>, Vec<i32>, Vec<usize>, usize, u64)> {
        if let Some((c, d, p)) = places_plan_at(dense, base, moe, room, expert, experts, least, may_start) {
            return Some((c, d, p, 0, 0));
        }
        if room.len() != 1 || ram.len() != dense.len() {
            return None;
        }
        let n = dense.len();
        (1..=n).find_map(|k| {
            let from = n - k;
            let moved: Vec<u64> = (from..n).map(|l| ram[l].min(dense[l])).collect();
            let slot = moved.iter().copied().max().unwrap_or(0);
            if slot == 0 {
                return None;
            }
            let mut d = dense.to_vec();
            d[from..].iter_mut().zip(&moved).for_each(|(x, m)| *x -= m);
            places_plan(&d, base + slot, moe, room, expert, experts, least).map(|(c, dv, p)| (c, dv, p, k, moved.iter().sum::<u64>() - slot))
        })
    }

    /// Where to cut layers of `per` bytes (the rest, `base`, half with the first stage and half with the last) over GPUs with `room` bytes each (ordinal, room):
    /// the one with the most room where it holds the whole model, else consecutive layers over the GPUs in order, each GPU's share of the bytes as its
    /// share of the room. None: they do not hold it.
    pub fn plan_cuts(per: &[u64], base: u64, room: &[(i32, u64)]) -> Option<(Vec<usize>, Vec<i32>)> {
        let total = per.iter().sum::<u64>() + base;
        let best = room.iter().max_by_key(|r| r.1)?;
        if total <= best.1 {
            return Some((Vec::new(), vec![best.0]));
        }
        cut_by_share(per, base, room)
    }

    /// `plan_cuts` where a stage begins only at a layer `starts` keeps (the first always; every layer where it is empty): the layers grouped from each such layer
    /// to the next, the groups cut, a cut at a group's first layer.
    pub fn plan_cuts_at(per: &[u64], base: u64, room: &[(i32, u64)], starts: &[bool]) -> Option<(Vec<usize>, Vec<i32>)> {
        let firsts: Vec<usize> = (0..per.len()).filter(|&l| l == 0 || starts.get(l).copied().unwrap_or(true)).collect();
        let groups: Vec<u64> = firsts.iter().enumerate().map(|(i, &f)| per[f..firsts.get(i + 1).copied().unwrap_or(per.len())].iter().sum()).collect();
        let (cuts, devices) = plan_cuts(&groups, base, room)?;
        Some((cuts.into_iter().map(|c| firsts[c]).collect(), devices))
    }

    /// The layers a stage of the model whose config.json is `c` (parsed once) may begin at, of its `nl`: every one but a layer whose latent attention takes the
    /// picks of the one before it (its indexer `shared`: `lower::latent_moe` keeps the two in one stage), and but a layer at or after which a compressed attention
    /// reads a part of a layer before it (`CompShare`: its entries, picks or candidate blocks; `lower_comp::lower` keeps a source and its readers in one stage);
    /// empty (every one) where the config describes no such layer.
    pub fn stage_starts(c: &ModelConfig, nl: usize) -> Vec<bool> {
        use crate::runtime::kinds::{Body, Mixer};
        let Ok(d) = crate::runtime::describe::describe_config(c) else { return Vec::new() };
        let blocks = &d.desc.blocks;
        let nb = blocks.len();
        let lent = |b: &crate::runtime::kinds::Block| matches!(&b.body, Body::Mixer(Mixer::Mla(m)) if m.indexer.as_ref().is_some_and(|i| i.shared));
        // (a compressed attention's shared parts: the layer of each block it reads, its own layer (a layer's mixer is its first, the blocks before it an n-gram's))
        let layer = |b: usize| blocks[..b].iter().filter(|x| matches!(x.body, Body::Mixer(_))).count();
        let shared: Vec<(usize, usize)> = (0..nb)
            .filter_map(|b| match &blocks[b].body {
                Body::Mixer(Mixer::Compressed(c)) => c.share.map(|sh| (b, sh)),
                _ => None,
            })
            .flat_map(|(b, sh)| [sh.entries, sh.picks, sh.candidates].into_iter().flatten().map(move |s| (layer(s), layer(b))))
            .collect();
        if nl == 0 || nb < nl || (!blocks.iter().any(lent) && shared.is_empty()) {
            return Vec::new();
        }
        let mut starts = vec![true; nl];
        for b in (0..nb).filter(|&b| lent(&blocks[b])) {
            starts[b * nl / nb] = false;
        }
        for (src, reader) in shared {
            starts[(src + 1).min(nl)..(reader + 1).min(nl)].iter_mut().for_each(|s| *s = false);
        }
        starts
    }

    /// The layers over every GPU of `room`, in order, each GPU's share of the bytes as its share of the room (`plan_cuts` where one GPU does not hold them).
    pub(crate) fn cut_by_share(per: &[u64], base: u64, room: &[(i32, u64)]) -> Option<(Vec<usize>, Vec<i32>)> {
        let total = per.iter().sum::<u64>() + base;
        let all = room.iter().map(|r| r.1).sum::<u64>().max(1);
        let (head, tail) = (base / 2, base - base / 2);
        let (mut cuts, mut devices, mut l) = (Vec::new(), Vec::new(), 0usize);
        for (k, &(dev, cap)) in room.iter().enumerate() {
            if l == per.len() {
                break;
            }
            let (start, last) = (l, k + 1 == room.len());
            let target = (total as u128 * cap as u128 / all as u128) as u64;
            let mut held = if devices.is_empty() { head } else { 0 };
            while l < per.len() {
                let add = per[l] + if l + 1 == per.len() { tail } else { 0 };
                if held + add > cap || (!last && held >= target) {
                    break;
                }
                held += add;
                l += 1;
            }
            if l > start {
                if !devices.is_empty() {
                    cuts.push(start);
                }
                devices.push(dev);
            }
        }
        (l == per.len()).then_some((cuts, devices))
    }

    // ---- a codes pack's expert cache that no plan sized (a development check's: src/dynamic/qwen4x.rs `load_cached_stage`, `free_sized`) ----

    /// The GPU memory the places of an expert cache leave stage `st` of the model of description `d`: each of `users` sequences' keys, values and states the
    /// stage's layers hold for the description's context (`rows::stage_holds`; each ring at `max_pos` rows), and the executor's slots and scratch (1.5 GiB, not
    /// measured for this model: the final pass measures it).
    pub fn cache_reserve(d: &crate::dynamic::ModelDesc, users: usize, st: &crate::dynamic::Stage) -> u64 {
        let (rings, states) = crate::dynamic::stage_holds(d, st);
        let ring: u64 = rings.iter().map(|&i| &d.rings[i]).map(|r| (d.max_pos * r.nk * r.d * 2 * (1 + r.values as usize)) as u64).sum();
        let states: u64 = states.iter().map(|&i| d.states[i] as u64).sum();
        users.max(1) as u64 * (ring + states) + CACHE_SCRATCH
    }

    /// The executor's slots and scratch beside an expert cache's places (`cache_reserve`).
    pub const CACHE_SCRATCH: u64 = 3 << 29;

    /// Places a layer of an expert cache: the GPU's `free` bytes less `reserve`, over `layers` layers of places of `place` bytes; at least one, at most `kept` (all
    /// of a layer's).
    pub fn cache_places(free: u64, reserve: u64, layers: usize, place: usize, kept: usize) -> usize {
        ((free.saturating_sub(reserve) / (layers.max(1) * place.max(1)) as u64) as usize).clamp(1, kept.max(1))
    }

    /// Whether a GPU of `free` bytes free holds `bytes` of experts held whole (and their tables of n-grams): else the experts go to an expert cache.
    pub fn resident_fits(bytes: u64, free: u64) -> bool {
        bytes <= free
    }

    /// What a model takes of one GPU at a level, read from its description: every path that holds a model on one GPU is sized from this by `one_gpu`, one rule for all
    /// of them (the weights, a draft model and the engine's work fixed; each sequence's states or rings and its context's keys and values; the reserve).
    pub(crate) struct Needs {
        pub kind: Kind,
        pub family: &'static str,
        pub layout: Layout,
        pub weights: u64,
        pub bf16: u64,
        /// A sequence's bytes besides its tokens' (recurrent states, sliding layers' rings), and a token's keys and values in one sequence.
        pub per_seq: u64,
        pub per_token: u64,
        /// What the keys and values take once whatever the users, at the context (the 12-bit pages' pools': `kv12::once_bytes`; nothing for bf16 rings).
        pub once: Once,
        pub users: usize,
        /// The engine takes fewer sequences where the room holds fewer than `users` of the context (rooms its scheduler gives out), so a context is sized for one;
        /// else every one of them holds it.
        pub fewer: bool,
        pub draft: u64,
        /// The prompt chunks the engine runs, each with its work (activations and workspaces), the first preferred: the plan takes the first the room left holds, else
        /// the last, whose work every plan keeps room for.
        pub chunks: Vec<(usize, u64)>,
        /// The context where none is asked, the least a plan takes and the most; a context the room gives is cut to a multiple of `grain`, every context to whole
        /// pages of `page` tokens.
        pub want: usize,
        pub least: usize,
        pub cap: usize,
        pub grain: usize,
        pub page: usize,
    }

    /// Why a model does not fit one GPU: the bytes it needs at `at` tokens of context (the context asked, else the least), and the longest context that fits (0: none).
    #[derive(Debug)]
    pub(crate) struct Short {
        pub need: u64,
        pub at: usize,
        pub most: usize,
    }

    /// The plan of a model on one GPU with `free` bytes free (`gpu` its name), at the context `asked` (refused where it does not fit), else the most the room holds up
    /// to `want`, not below `least`.
    pub(crate) fn one_gpu(n: &Needs, free: u64, asked: Option<usize>, gpu: &str) -> Result<LoadPlan, Short> {
        let (users, per) = (n.users.max(1) as u64, n.per_token.max(1));
        let reserve = reserve_of(free);
        let fixed = n.weights + n.draft + n.chunks.last().map_or(0, |c| c.1) + reserve;
        let kv = |seqs: u64, ctx: usize| n.kv(seqs, ctx);
        // (the tokens each of `seqs` sequences has room for beside `fixed`; where something is once and grows with the context, the most that fits, found by halves
        // below the bound without it)
        let most = |seqs: u64| -> usize {
            let top = (free.saturating_sub(fixed + n.per_seq * seqs + n.once.fixed) / (per * seqs)) as usize;
            if n.once.rings.is_empty() {
                return top;
            }
            let (mut lo, mut hi) = (0usize, top);
            while lo < hi {
                let mid = lo + (hi - lo).div_ceil(2);
                if fixed + kv(seqs, mid) <= free {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            lo
        };
        // (the tokens each sequence has room for where all `users` hold the context, and where one does)
        let (all, one) = (most(users), most(1));
        let room = if n.fewer { one } else { all };
        let want = n.want.min(n.cap);
        let ctx = match asked {
            Some(a) => a,
            None if n.fewer && all.max(n.least).min(one) >= want => want,
            None if n.fewer => all.max(n.least).min(one) / n.grain * n.grain,
            None if room >= want => want,
            None => room / n.grain * n.grain,
        } / n.page
            * n.page;
        let at = asked.unwrap_or(n.least.min(want));
        if asked.is_some_and(|a| a > n.cap) || ctx < at.min(n.least) || ctx > room {
            let at = if asked.is_some() { at } else { ctx.max(n.least) };
            let most = (room.min(n.cap) / n.page * n.page).min(asked.unwrap_or(usize::MAX));
            // (a scheduler of rooms is refused where one sequence does not fit: its need is one's)
            let seqs = if n.fewer { 1 } else { users };
            return Err(Short { need: fixed + kv(seqs, at), at, most });
        }
        let left = free.saturating_sub(n.weights + n.draft + reserve + kv(users, ctx));
        let chunk = n.chunks.iter().copied().find(|c| c.1 <= left).or(n.chunks.last().copied()).unwrap_or((512, 0));
        let mut p = n.plan_at(ctx, chunk, free, gpu);
        if n.fewer {
            // (a scheduler of rooms in pools of blocks holds what the GPU has left, at most every user's context, at least one's: src/dynamic/rows.rs `page`.
            // On an A10 (23 GB) a dense model of 36 layers of 4,096 for 8 users at 16,384 tokens was planned at 31,867 MiB, the context of all eight; the server took 21,889)
            p.kv = p.kv.min(free.saturating_sub(p.weights + p.draft + p.work + p.reserve)).max(kv(1, ctx));
        }
        Ok(p)
    }

    /// The plan of `n` (its keys and values bf16) or of `twelve` (the same model and level, its keys and values in the executor's 12-bit pages, where
    /// GLYD_KV=lossless asks for them), as `one_gpu` makes each: both are lossless, so the 12-bit pages are taken only where they reserve no more than bf16
    /// for the plan, that is where their plan gives more context than bf16's, or the same in no more bytes for its users (`Needs::kv`: the pages' worst case
    /// beside their tokens, a raw slab for every page of one sequence and the rows a pass unpacks, `kv12::once_bytes`, which one user at a long context pays
    /// for and many users share). Else bf16's plan, so a level is never planned lower for the 12-bit pages than bf16 holds it. A check of the pages against
    /// their own plan asks for the 12-bit plan wherever one fits (`twelve_always`).
    pub(crate) fn one_gpu_pages(n: &Needs, twelve: Option<&Needs>, free: u64, asked: Option<usize>, gpu: &str) -> Result<LoadPlan, Short> {
        let bf16 = one_gpu(n, free, asked, gpu);
        let Some(t) = twelve.filter(|t| t.twelve()) else { return bf16 };
        match (bf16, one_gpu(t, free, asked, gpu)) {
            (Ok(b), Ok(p)) => {
                let seqs = n.users.max(1) as u64;
                Ok(if twelve_always() || p.context > b.context || (p.context == b.context && t.kv(seqs, p.context) <= n.kv(seqs, b.context)) { p } else { b })
            }
            (Err(_), Ok(p)) => Ok(p),
            (bf16, _) => bf16,
        }
    }

    impl Needs {
        /// The keys and values of `seqs` sequences at `ctx` tokens: each one's, and what the 12-bit pages take once whatever the sequences (`Once`).
        pub(crate) fn kv(&self, seqs: u64, ctx: usize) -> u64 {
            seqs * (self.per_seq + ctx as u64 * self.per_token.max(1)) + self.once.at(ctx)
        }

        /// Whether the keys and values are in the executor's 12-bit pages (a ring of them takes the pages: what they spend once is counted).
        pub(crate) fn twelve(&self) -> bool {
            !self.once.rings.is_empty()
        }

        /// The plan at `ctx` tokens of context and the prompt chunk `chunk` (its tokens, its work).
        pub(crate) fn plan_at(&self, ctx: usize, chunk: (usize, u64), free: u64, gpu: &str) -> LoadPlan {
            let users = self.users.max(1) as u64;
            LoadPlan {
                kind: self.kind,
                family: self.family,
                level: level_name(self.layout),
                layout: self.layout,
                context: ctx,
                users: users as usize,
                chunk: chunk.0,
                weights: self.weights,
                bf16: self.bf16,
                kv: self.kv(users, ctx),
                kv12: self.twelve(),
                work: chunk.1,
                draft: self.draft,
                reserve: reserve_of(free),
                free,
                gpu: gpu.to_string(),
                experts: None,
                spread: None,
            }
        }
    }

    /// The refusal of a model that does not fit one GPU (`one_gpu`'s `Short`), in one form for every path: what it needs at what context against what is free, and
    /// what to do: a shorter context where one fits the context asked, else the ways to more room (`alt` first where the caller has one: a smaller download).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn too_big(
        name: &str,
        level: &str,
        gpu: &str,
        free: u64,
        users: usize,
        s: &Short,
        asked: Option<usize>,
        draft: bool,
        alt: Option<String>,
    ) -> Refusal {
        let what = format!(
            "{name} at the {level} level takes about {} of GPU memory{} at {} tokens of context, and the {gpu} has {} free",
            gib(s.need),
            if users > 1 { format!(" for {users} users") } else { String::new() },
            s.at,
            gib(free)
        );
        if asked.is_some() && s.most >= 1024 {
            return Refusal::unsupported(what, format!("Ask for less: --context {}", s.most / 256 * 256));
        }
        let rest = format!(
            "lose programs that use the GPU (nvidia-smi lists them), ask for less context (--context 4096), {}or use a GPU with more memory",
            if draft { "run without the draft model (--no-draft), " } else { "" }
        );
        Refusal::unsupported(what, alt.map_or_else(|| format!("C{rest}"), |a| format!("{a}\nOr c{rest}")))
    }

    /// A dense model's needs on the executor at `layout` for `users` sequences (the dense kind's: one attention and one gated MLP a layer; `family`: its row's name):
    /// its packed weights, its keys and values (bf16, or 12 bits a value in its 12-bit pages: `twelve`) in pages the sequences share (a sequence takes the blocks its
    /// tokens fill, so the context is sized for one), its chunks' work with cuBLAS (`cublas`) or without.
    pub(crate) fn dense_needs(cfg: &Config, family: &'static str, layout: Layout, users: usize, cublas: bool, twelve: bool, served: bool) -> Needs {
        let first = if cublas { 4096 } else { 2048 };
        Needs {
            kind: Kind::Qwen3,
            family,
            layout,
            weights: weights(cfg, layout),
            bf16: cfg.matrix_bytes(),
            per_seq: 0,
            per_token: kv_per_token(cfg, twelve),
            once: Once { rings: dense_rings(cfg, twelve), chunk: first, fixed: pools_of(0, &[kv_per_token(cfg, false)], served).1 },
            users: users.max(1),
            fewer: true,
            draft: 0,
            chunks: [first, 2048, 1024, 512].into_iter().filter(|&c| c <= first).map(|c| (c, work(cfg, c, cublas))).collect(),
            want: CTX_WANT.min(cfg.max_pos),
            least: CTX_MIN,
            cap: cfg.max_pos,
            grain: 256,
            page: 1,
        }
    }

    /// The plan of a dense model of the family `row` on a GPU that has `free` bytes (`gpu` its name), for `users` sequences at once, at the context `asked`, else the
    /// model's own up to `CTX_WANT`, less where the room is short (not below `CTX_MIN`); its keys and values bf16, or in the 12-bit pages where `twelve` asks for
    /// them and they reserve no more for the plan (`one_gpu_pages`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dense_fit(
        cfg: &Config,
        row: &'static Support,
        name: &str,
        gpu: &str,
        free: u64,
        layout: Layout,
        asked: Option<usize>,
        users: usize,
        cublas: bool,
        twelve: bool,
        served: bool,
    ) -> Result<LoadPlan, Refusal> {
        let n = dense_needs(cfg, row.name, layout, users, cublas, false, served);
        let pages = twelve.then(|| dense_needs(cfg, row.name, layout, users, cublas, true, served));
        one_gpu_pages(&n, pages.as_ref(), free, asked, gpu).map_err(|s| too_big(name, level_name(layout), gpu, free, n.users, &s, asked, false, None))
    }

    /// The row of the one family whose configs take the dense sizing today (`Sizing::Dense`): what `qwen3_fit` and `qwen3_plan`, which are given no config's
    /// text, size for.
    fn dense_row() -> &'static Support {
        crate::support::TABLE.iter().find(|r| r.route.sizing == Sizing::Dense).expect("a row takes the dense sizing")
    }

    /// `dense_fit` for the row that takes the dense sizing, its keys and values bf16 (src/cli.rs calls it by this name).
    #[allow(clippy::too_many_arguments)]
    pub fn qwen3_fit(
        cfg: &Config,
        name: &str,
        gpu: &str,
        free: u64,
        layout: Layout,
        asked: Option<usize>,
        users: usize,
        cublas: bool,
    ) -> Result<LoadPlan, Refusal> {
        dense_fit(cfg, dense_row(), name, gpu, free, layout, asked, users, cublas, false, false)
    }

    /// Whether the plan of `a` may put its keys and values in the executor's 12-bit lossless pages (src/dynamic/kvpages.rs): GLYD_KV=lossless asks for them, and the
    /// plan is a server's (`served`: `serve`, or `fit --users`), whose scheduler holds its sequences in pools of blocks, which the pages are (`Dyn::page`); `run`
    /// answers one prompt by itself, in rings of its own. Each path's plan takes them where its description's rings take them and they reserve no more than bf16
    /// (`one_gpu_pages`, `size_dynamic_at`).
    pub(crate) fn twelve_pages(a: &Args) -> bool {
        lossless_asked() && served(a)
    }

    /// Whether GLYD_KV=lossless asks for the executor's 12-bit lossless pages (a test's thread says so for itself alone: `asking_lossless`).
    pub(crate) fn lossless_asked() -> bool {
        #[cfg(test)]
        if let Some(on) = LOSSLESS.with(std::cell::Cell::get) {
            return on;
        }
        std::env::var("GLYD_KV").as_deref() == Ok("lossless")
    }

    #[cfg(test)]
    thread_local! {
        static LOSSLESS: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
        static ALWAYS: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    }

    /// Whether a development build's GLYD_KV12_ALWAYS=1 asks for the 12-bit plan wherever one fits, also where bf16 reserves less: a check that runs the 12-bit
    /// pages holds them to the plan made for them (`usercheck --ctx plan --kv12`), never to bf16's bytes. A test's thread says so for itself alone (`always_twelve`).
    pub(crate) fn twelve_always() -> bool {
        #[cfg(test)]
        if let Some(on) = ALWAYS.with(std::cell::Cell::get) {
            return on;
        }
        crate::tune!("GLYD_KV12_ALWAYS").as_deref() == Some("1")
    }

    /// `f` run with the 12-bit plan asked for wherever one fits (`on`) or not, as this test's thread alone sees it.
    #[cfg(test)]
    pub(crate) fn always_twelve<T>(on: bool, f: impl FnOnce() -> T) -> T {
        ALWAYS.with(|c| c.set(Some(on)));
        let got = f();
        ALWAYS.with(|c| c.set(None));
        got
    }

    /// `f` run with GLYD_KV=lossless asked (`on`) or not, as this test's thread alone sees it: the tests share one process's environment.
    #[cfg(test)]
    pub(crate) fn asking_lossless<T>(on: bool, f: impl FnOnce() -> T) -> T {
        LOSSLESS.with(|c| c.set(Some(on)));
        let got = f();
        LOSSLESS.with(|c| c.set(None));
        got
    }

    /// A dense product for the long prompts (cuBLAS where the computer has it, else the engine's own on Ampere, Ada and Hopper): the decoded matrix and the larger
    /// chunk that goes with it. `cublas`: the target's word on cuBLAS (`Target::cublas`: a preset's or a described machine's), None for this computer (its loader
    /// is asked): a plan for a machine one does not have never turns on the computer that plans it.
    pub(crate) fn dense_product(cc: (i32, i32), cublas: Option<bool>) -> bool {
        cublas.unwrap_or_else(Blas::present) || (matches!(cc.0, 8 | 9) && crate::tune!("GLYD_ENGINE_GEMM").map_or(true, |v| v != "lib"))
    }

    /// The GPU `run` picked, as a plan sees it, and the computer's memory the experts may take.
    pub(crate) fn target(gpu: &Gpu) -> Target {
        Target { ordinal: gpu.ordinal, name: gpu.name.clone(), cc: gpu.cc, total: gpu.total, free: gpu.free, host: crate::offload::host_memory(), cublas: None }
    }

    /// The layout and the fit of a model on a GPU: what `prepare` settles before it downloads a thing, and `preflight` before it says the engine will run the model. `packed`: the layout of a
    /// directory `pack` made, which is the layout there.
    pub(crate) fn qwen3_plan(a: &Args, cfg: &Config, name: &str, gpu: &Gpu, packed: Option<Layout>, users: usize) -> Done<(Layout, String, LoadPlan)> {
        dense_plan_on(a, cfg, name, &target(gpu), packed, users)
    }

    /// `qwen3_plan` on the target `t` (a machine's GPU with its own word on cuBLAS, `Target::cublas`).
    pub(crate) fn dense_plan_on(a: &Args, cfg: &Config, name: &str, t: &Target, packed: Option<Layout>, users: usize) -> Done<(Layout, String, LoadPlan)> {
        dense_plan(a, cfg, dense_row(), name, t, packed, users)
    }

    /// The layout and the plan of a dense model of the family `row` on the GPU `t` (`dense_fit`), with the flags of `a`.
    pub(crate) fn dense_plan(
        a: &Args,
        cfg: &Config,
        row: &'static Support,
        name: &str,
        t: &Target,
        packed: Option<Layout>,
        users: usize,
    ) -> Done<(Layout, String, LoadPlan)> {
        let asked = if a.has("context") || a.has("ctx") { Some(a.either("context", "ctx", 0usize)?) } else { None };
        crate::support::context_check(name, row, cfg.max_pos, asked)?;
        let (layout, why) = match packed {
            Some(l) => (l, "as packed".to_string()),
            None => choose_layout(a.layout()?, t.cc, t.total, cfg.matrix_bytes(), 0),
        };
        // (what the last load of this model here took beyond its plan is planned as already taken: `calibrated`)
        let extra = calibration(name, levels::name_of(layout), &t.name);
        let mut fit =
            dense_fit(cfg, row, name, &t.name, t.free.saturating_sub(extra), layout, asked, users, dense_product(t.cc, t.cublas), twelve_pages(a), served(a))?;
        calibrated(&mut fit, t.free, extra);
        Ok((layout, why, fit))
    }

    /// The bytes of this GPU's free memory the planner takes as already used for `model` at `level`: what the last load took beyond its plan (src/plan.rs
    /// `calibration_in`; the one-GPU paths' only: the others' loads do not count their weights apart).
    pub(crate) fn calibration(model: &str, level: &str, gpu: &str) -> u64 {
        crate::plan::calibration_in(&store::cache_root(), model, level, gpu)
    }

    /// `plan` made for the GPU's free memory less `extra`, said as it is: the free memory as it was, and `extra` a part of the reserve.
    pub(crate) fn calibrated(p: &mut LoadPlan, free: u64, extra: u64) {
        p.free = free;
        p.reserve += extra;
    }

    /// The plan `run` loads MODEL (its config.json, parsed once: `c`) at `level` on the GPU `t`, with the flags of `a`, or the refusal `run` gives: what `fit`
    /// prints and `run`, `serve` and `preflight` size with. `users`: the sequences a server holds at once (the Qwen3 path's; the lossy paths hold one). Ok(None):
    /// no path of `run` loads this family yet (`Sizing::Estimate`), or the dynamic route has no headers of the model's files, and for `fit` the planner's
    /// estimate stands (src/plan.rs).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn size(
        a: &Args,
        model: &str,
        c: &ModelConfig,
        level: &'static str,
        devs: &[Target],
        t: &Target,
        reach: &dyn Fn(&[i32]) -> bool,
        users: usize,
    ) -> Done<Option<LoadPlan>> {
        let name = crate::levels::short_name(model);
        let row = crate::support::row(crate::support::family_in(c, &name)?);
        match row.route.sizing {
            Sizing::Dense => {
                supported_in(&name, c)?;
                let cfg = Config::of(c, "config.json", None).map_err(cannot)?;
                // (penguin at the layout the GPU is fastest in, a lossy level at its own)
                let (_, _, p) = dense_plan(a, &cfg, row, &name, t, levels::layout_of(level).filter(|l| *l != Layout::Tiered), users)?;
                shape_checks(&name, &cfg, &p)?;
                Ok(Some(p))
            }
            Sizing::OneGpu => {
                let Some(layout) = levels::layout_of(level) else { return Ok(None) };
                // (penguin of a family with lossy levels on the dynamic route: its layers cut over the GPUs as their room holds them; its row's name is the route's)
                if layout == Layout::Tiered && row.split_levels.contains(&"penguin") {
                    return size_local_dynamic(a, model, c, row.name, devs, reach);
                }
                // (its layers in stages over the machine's GPUs where its row cuts them at this level and one GPU does not hold it: `lossy_plan_over`)
                lossy_plan_over(a, model, layout_for(model, layout), c, None, t.free, &t.name, None, t.host, devs).map(Some)
            }
            // (the hummingbird level of a row whose experts come from a codes pack: the pack's plan, where its headers and a pack can be had)
            // (a model here with no codes pack beside its weights is refused as `run` refuses it)
            Sizing::Dynamic if level == "hummingbird" && row.route.expert_cache == ExpertCache::Codes => match codes_pack(a, model) {
                Some(pack) => {
                    let Some(table) = crate::sizes::Table::find(model) else { return Ok(None) };
                    codes_plan(a, &name, &table, c, row, &pack, devs, devs.first().and_then(|t| t.host)).map(Some)
                }
                None if local_dir(model).is_some() => Err(no_codes_pack(&name, row.name, model)),
                None => Ok(None),
            },
            Sizing::Dynamic => size_local_dynamic(a, model, c, row.name, devs, reach),
            Sizing::Estimate => Ok(None),
        }
    }

    /// `size_dynamic` of a model by its files' headers: a directory's or the Hub cache's, else the Hub's by ranges before anything is downloaded (src/sizes.rs;
    /// ROADMAP P2b); Ok(None) where the headers cannot be had (offline, the Hub does not answer), and the planner's estimate stands.
    pub(crate) fn size_local_dynamic(
        a: &Args,
        model: &str,
        c: &ModelConfig,
        family: &'static str,
        devs: &[Target],
        reach: &dyn Fn(&[i32]) -> bool,
    ) -> Done<Option<LoadPlan>> {
        let Some(table) = crate::sizes::Table::find(model) else { return Ok(None) };
        // (the files where they are here: a hummingbird directory's experts are not in the table)
        let dir = if Path::new(model).join("config.json").exists() { Some(PathBuf::from(model)) } else { Source::find(model).ok().map(|s| s.dir) };
        let host = devs.first().and_then(|t| t.host);
        size_dynamic(a, &short_name(model), &table, dir.as_deref(), c, family, devs, host, reach).map(Some)
    }

    /// `--expert-cache GIB`: the GPU memory to give the places of the expert cache, in GiB.
    pub(crate) fn cache_flag(a: &Args) -> Done<Option<u64>> {
        let Some(v) = a.flag("expert-cache") else { return Ok(None) };
        match v.parse::<f64>() {
            Ok(g) if g.is_finite() && g > 0.0 => Ok(Some((g * (1u64 << 30) as f64) as u64)),
            _ => Err(Refusal::new(format!("--expert-cache {v}: not a number of GiB"), "Give the GPU memory for the experts, in GiB: --expert-cache 12")),
        }
    }

    /// `--host-memory GIB|auto`: the computer's memory the experts may take, in GiB; None for `auto` (and where it is not given): what the computer has free less a reserve (`HOST_MARGIN`), at most all the experts. Less than all
    /// of them, the Hummingbird level keeps the rest in its packed directory and reads them as the router asks for them (src/offload.rs `Tier`).
    pub fn host_flag(a: &Args) -> Done<Option<u64>> {
        match a.flag("host-memory") {
            None | Some("auto") => Ok(None),
            Some(v) => match v.parse::<f64>() {
                Ok(g) if g.is_finite() && g > 0.0 => Ok(Some((g * (1u64 << 30) as f64) as u64)),
                _ => Err(Refusal::new(
                    format!("--host-memory {v}: not a number of GiB"),
                    "Give the computer's memory the experts may take, in GiB: --host-memory 20 (auto: what the computer has free, less 6 GiB for the system)",
                )),
            },
        }
    }

    /// The bytes of `experts_bytes` of experts the computer's memory holds (`CachePlan::ram`): what `asked` is (`--host-memory`), else what the computer has free (`host`; None: it does not say, and is taken to have enough)
    /// less the reserve; at most all of them.
    pub fn held_in_memory(experts_bytes: u64, asked: Option<u64>, host: Option<u64>) -> u64 {
        match (asked, host) {
            (Some(b), _) => b.min(experts_bytes),
            (None, Some(h)) => h.saturating_sub(HOST_MARGIN).min(experts_bytes),
            (None, None) => experts_bytes,
        }
    }

    /// The expert cache a model of the Nemotron-H layout in the glyd-v7 directory `dir` (at `layout`) gets on a GPU of `free` bytes free, in a computer with `host`
    /// bytes of memory to give: what `run` gives it (`lossy_plan_with` on the directory's own config, parsed once, and packs); None where the GPU holds the model
    /// whole. For `glyd-engine generate --expert-cache auto`.
    pub fn expert_cache_for(dir: &Path, layout: Layout, free: u64, gpu: &str, host: Option<u64>) -> Done<Option<CachePlan>> {
        let c = ModelConfig::load(dir).map_err(|e| Refusal::new(e.msg, ""))?;
        let (generation, name) = (ModelConfig::generation(dir), dir.to_string_lossy().into_owned());
        let packed = dir.join("glyd.json").exists().then_some(dir);
        Ok(lossy_plan_with(&Args::from(["run".to_string(), name.clone()]), &name, layout, &c, generation.as_deref(), free, gpu, packed, host)?.experts)
    }

    /// The model's config as a lossy level reads it (the Qwen3.5 layout, dense or with experts, by its description: Qwen3.5, Qwen3.6 and Qwen3.8, their config says `qwen3_5`; the Nemotron-H layout: `nemotron_h`, at the
    /// kestrel, swift and wren levels; Gemma 4 at penguin and its lossy levels), and what it takes in `layout` of `free` bytes of GPU memory at the context asked for (else up to LOSSY_CTX, as much as fits): refused where
    /// it does not fit at LOSSY_MIN_CTX. A pack is sized in its own layout (`pack_layout`: Nemotron's experts alone are four-bit at wren). `c`: its config.json, parsed once; `packed`: the glyd-v7 directory the model is,
    /// where its packs' bytes are the ones it holds.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn lossy_plan(
        a: &Args,
        model: &str,
        layout: Layout,
        c: &ModelConfig,
        gen: Option<&str>,
        free: u64,
        gpu: &str,
        packed: Option<&Path>,
    ) -> Done<LoadPlan> {
        lossy_plan_with(a, model, layout, c, gen, free, gpu, packed, crate::offload::host_memory())
    }

    /// `lossy_plan` for a computer that has `host` bytes of memory to give (None: it does not say). A model whose description has experts the computer's memory can hold
    /// (`LevelOf::experts`) that does not fit the GPU whole, or that `--expert-cache GIB` asks for, keeps its experts in the computer's memory and a cache of them on the
    /// GPU (`expert_cache_with`); it is refused, with the memory it needs of each, where the GPU or the computer has too little for that.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn lossy_plan_with(
        a: &Args,
        model: &str,
        layout: Layout,
        c: &ModelConfig,
        gen: Option<&str>,
        free: u64,
        gpu: &str,
        packed: Option<&Path>,
        host: Option<u64>,
    ) -> Done<LoadPlan> {
        lossy_plan_over(a, model, layout, c, gen, free, gpu, packed, host, &[])
    }

    /// `lossy_plan_with` on a machine of the GPUs `devs` (`free` and `gpu` the one with the most free memory's, which the one-GPU rule takes): where the model's
    /// row cuts this level's layers over GPUs (`Support::split_levels`) and that GPU does not hold the model whole, its layers in stages over them
    /// (`lossy_stages`); `--split CUTS [--devices GPUS]` cuts them where it says. What `fit`, `run` and `serve` plan.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn lossy_plan_over(
        a: &Args,
        model: &str,
        layout: Layout,
        c: &ModelConfig,
        gen: Option<&str>,
        free: u64,
        gpu: &str,
        packed: Option<&Path>,
        host: Option<u64>,
        devs: &[Target],
    ) -> Done<LoadPlan> {
        // (what the last load of this model here took beyond its plan is planned as already taken: `calibrated`; one GPU's, not a cut's)
        let extra = calibration(&short_name(model), level_name(layout), gpu);
        let mut p = lossy_plan_inner(a, model, layout, c, gen, free.saturating_sub(extra), gpu, packed, host, devs)?;
        if p.spread.is_none() {
            calibrated(&mut p, free, extra);
        }
        Ok(p)
    }

    /// A smaller download of the same model a refusal names where it fits (Gemma 4's wren repo of the QAT checkpoint): its repo, its level, what it is, and its
    /// weights' and draft model's bytes.
    pub(crate) struct Smaller {
        pub repo: String,
        pub level: &'static str,
        pub what: &'static str,
        pub weights: u64,
        pub draft: u64,
    }

    /// What a model's description says its plan at a lossy route's level takes (`lossy_plan_inner`'s one rule reads nothing else of it): its weights at the level
    /// and in bf16; a sequence's recurrent states (copied at its mark where the engine keeps them: `pools_of`) and its sliding layers' rings at the context; a token's
    /// keys and values in the other layers; each in bf16 and in the executor's 12-bit pages where they take its rings (`[bf16, 12-bit]`), with the rings they take
    /// and its pools' groups' bytes a token in bf16 (each group's block 0); its longest context, the prompt chunk its engine runs, and the page its contexts are cut
    /// to; the bytes of a draft model where one is found or comes with the download (the rule drafts only for one user: `serve_rows` holds one sequence then);
    /// a smaller download that a refusal names; and, where it has experts (every description with the Moe kind's blocks), their description as the one
    /// expert-cache rule takes it (`CacheOf`; the reason in words where it cannot be read): the computer's memory holds them and a cache of them is on the GPU.
    pub(crate) struct LevelOf {
        pub kind: Kind,
        pub row: &'static Support,
        pub weights: u64,
        pub bf16: u64,
        pub state: u64,
        pub rings: [u64; 2],
        pub per: [u64; 2],
        pub groups: Vec<u64>,
        pub rings12: Vec<crate::dynamic::kv12::KvShape>,
        pub max_pos: usize,
        pub chunk: usize,
        pub page: usize,
        pub draft: Option<u64>,
        pub smaller: Option<Smaller>,
        pub experts: Option<Result<CacheOf, String>>,
        /// Its layers' bytes, where a stage of the layers loads at this level (`lossy_stages`; None where its loader loads it whole alone).
        pub layers: Option<LayerBytes>,
    }

    /// What cuts a lossy level's layers into stages (`lossy_stages`), by the description: each layer's bytes at the level, all of them (`whole`) and besides its
    /// routed experts (`dense`), the rest the weights less them (the embedding, the head, the final norm), and the description lowered whole, whose rings and
    /// states are each layer's share of a sequence (`rows::layer_rooms`).
    #[derive(Clone, Debug)]
    pub(crate) struct LayerBytes {
        pub whole: Vec<u64>,
        pub dense: Vec<u64>,
        pub low: crate::dynamic::ModelDesc,
    }

    impl LevelOf {
        /// The one-GPU rule's needs (`one_gpu`) of this level for `users` sequences, served (`served`: the scheduler's pools) or not, its keys and values in 12-bit
        /// pages (`twelve`) or bf16, with a draft model of `draft` bytes (0: none).
        pub(crate) fn needs(&self, layout: Layout, users: usize, twelve: bool, served: bool, draft: u64) -> Needs {
            let (state, fixed) = pools_of(self.state, &self.groups, served);
            let t = twelve as usize;
            Needs {
                kind: self.kind,
                family: self.row.name,
                layout,
                weights: self.weights,
                bf16: self.bf16,
                per_seq: state + self.rings[t],
                per_token: self.per[t].max(1),
                once: Once { rings: if twelve { self.rings12.clone() } else { Vec::new() }, chunk: self.chunk, fixed },
                users,
                fewer: true,
                draft,
                chunks: vec![(self.chunk, LOSSY_WORK)],
                want: LOSSY_CTX,
                least: LOSSY_MIN_CTX,
                cap: self.max_pos.max(LOSSY_MIN_CTX),
                grain: 1,
                page: self.page,
            }
        }
    }

    /// The weights of packs `packs` (each at the layout `own` gives its key) and tensors `kept` as they are, as a lossy level holds them, each with its bytes
    /// and its bf16 bytes (named by its first part): a glyd-v7 directory's own bytes where `packed` is one (a matrix it holds as bf16 is packed as it loads:
    /// sized as without one).
    fn packed_items(
        packs: Vec<Pack35>,
        kept: Vec<crate::qwen35::Tensor35>,
        packed: Option<&Path>,
        own: &dyn Fn(&str) -> Layout,
    ) -> Done<Vec<(String, u64, u64)>> {
        let ck = packed.map(|d| glyd_gpu::safetensors::Checkpoint::open(d).map_err(cannot)).transpose()?;
        let mut items = Vec::with_capacity(packs.len() + kept.len());
        for p in packs {
            let b = (p.rows() * p.cols * 2) as u64;
            let saved = ck.as_ref().map_or(0, |ck| p.parts.iter().filter_map(|x| packed_bytes(ck, Pack35::saved(&x.0))).sum::<u64>());
            let w = if saved > 0 {
                saved
            } else if p.packable() {
                (b as f64 * bits(own(&p.key)) / 16.0) as u64
            } else {
                b
            };
            items.push((p.parts.first().map_or_else(String::new, |x| x.0.clone()), w, b));
        }
        for t in kept {
            let b = t.shape.iter().product::<u64>() * if t.dtype == "F32" { 4 } else { 2 };
            items.push((t.name, b, b));
        }
        Ok(items)
    }

    /// Each of `nl` layers' bytes of `items` (`packed_items`: a tensor's layer by its name), all of them and besides its routed experts (a tensor of
    /// `.experts.`); a tensor of no layer is the rest's.
    fn layer_items(items: &[(String, u64, u64)], nl: usize) -> (Vec<u64>, Vec<u64>) {
        let (mut whole, mut dense) = (vec![0u64; nl], vec![0u64; nl]);
        for (n, w, _) in items {
            if let Some(l) = crate::sizes::layer_of(n).filter(|&l| l < nl) {
                whole[l] += w;
                if !n.contains(".experts.") {
                    dense[l] += w;
                }
            }
        }
        (whole, dense)
    }

    /// The level description (`LevelOf`) of the model `c` at `layout`: one reader for every row whose levels the one-GPU route's lossy levels take (`Support::lossy`),
    /// by the model's description (`describe_config`) lowered whole as its loader lowers it (`split::lowered_described`): what a level's save holds of it
    /// (`pack::described`: every matrix in the layout its row's loader packs it in at the level, `Lossy::pack`, a glyd-v7 directory's own bytes where `packed` is
    /// one; the tensors kept as they are), a sequence's states, its rings' keys and values (a sliding window's fixed at the context asked, else LOSSY_CTX, not
    /// below LOSSY_MIN_CTX: `ring_bytes`), its longest context, its row's prompt chunk and page, its row's draft model where one is found or comes with the
    /// download, the smaller download a penguin refusal names (`qat_wren`), its experts where its feed-forwards are the Moe kind's (`experts_of`), and each
    /// layer's bytes for its stages. Refused in the table's words for a level its row does not run (`support::check_in`), in its family's reader's for a config
    /// the engine does not run (`Lossy::reads`), and in its experts' for a level they are not held at.

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn level_of(
        a: &Args,
        model: &str,
        name: &str,
        layout: Layout,
        c: &ModelConfig,
        gen: Option<&str>,
        packed: Option<&Path>,
        asked: Option<usize>,
    ) -> Done<LevelOf> {
        let tag = level_name(layout);
        let row = crate::support::row(crate::support::family_in(c, name)?);
        let Some(lossy) = row.lossy else {
            let rows: Vec<&str> = crate::support::TABLE.iter().filter(|r| r.lossy.is_some()).map(|r| r.name).collect();
            return Err(Refusal::unsupported(
                format!("{name}: the {tag} level on this path takes the {} rows, not the {} one", rows.join(", "), row.name),
                format!("Run it lossless: glyd run {model}"),
            ));
        };
        crate::support::check_in(c, model, name, tag)?;
        let refuse = |e: crate::ffi::Error| Refusal::unsupported(format!("{name}: {}", e.msg), unrun_fix());
        (lossy.reads)(c, gen).map_err(refuse)?;
        let d = crate::runtime::describe::describe_config(c).map_err(refuse)?;
        let max_pos = c.shared.max_position_embeddings.ok_or_else(|| refuse(crate::ffi::Error("no max_position_embeddings".into())))?;
        let low = crate::dynamic::split::lowered_described(&d, c.num("num_hidden_layers").unwrap_or(0), max_pos).map_err(refuse)?;
        // (its experts at this level, refused where they are not held at it)
        let moe = MoeShape::of(&d, c);
        if let Some(Err(e)) = moe.as_ref().map(|x| x.place(layout)) {
            let fix = match layout {
                Layout::Hummingbird => format!("Run it at the swift or kestrel level: glyd run {model}:swift"),
                _ => format!("Run it at kestrel: glyd run {model}:kestrel"),
            };
            return Err(Refusal::unsupported(format!("{name}: {}", e.msg), fix));
        }
        let (packs, kept) = crate::pack::described(&d, &low).map_err(refuse)?;
        let typed = a.pos.get(1).map_or(name, |m| level(m, false).map_or(m.as_str(), |(m, _)| m));
        // (the smaller download a penguin refusal names, its weights at its layout)
        let smaller = match qat_wren(typed).filter(|_| layout == Layout::Tiered) {
            Some(repo) => {
                let at = packed_items(packs.clone(), kept.clone(), None, &|k| lossy.pack.map_or(Layout::Int4H, |f| f(Layout::Int4H, k)))?;
                let weights = at.iter().map(|x| x.1).sum();
                Some(Smaller { repo, level: "wren", what: "Google's QAT checkpoint, bit for bit", weights, draft: lossy.draft })
            }
            None => None,
        };
        let items = packed_items(packs, kept, packed, &|k| lossy.pack.map_or(layout, |f| f(layout, k)))?;
        let (weights, bf16) = items.iter().fold((0, 0), |(w, b), x| (w + x.1, b + x.2));
        let (whole, dense) = layer_items(&items, low.layers.len());
        // (a model not on this computer yet comes as its pre-packed repo, which holds its draft model: planned with it, the larger of the two plans)
        let draft_to_come = packed.is_none() && !a.has("no-draft") && levels::draft_repo(typed).is_some(); // (the one rule that says which downloads bring a draft model: src/levels.rs)
        let draft = (lossy.draft > 0 && (draft_dir(a, packed).is_some() || draft_to_come)).then_some(lossy.draft);
        let (rings, per, groups) = ring_bytes(&low, lossy.chunk, asked.unwrap_or(LOSSY_CTX).max(LOSSY_MIN_CTX));
        let experts = moe.map(|x| x.cache_of(c, layout, weights, &dense).map_err(|e| e.msg));
        Ok(LevelOf {
            kind: lossy.kind,
            row,
            weights,
            bf16,
            state: low.states.iter().sum::<usize>() as u64,
            rings,
            per,
            groups,
            rings12: crate::dynamic::twelve_bit_rings(&low).into_iter().flatten().collect(),
            max_pos: low.max_pos,
            chunk: lossy.chunk,
            page: lossy.page,
            draft,
            smaller,
            experts,
            layers: Some(LayerBytes { whole, dense, low }),
        })
    }

    /// The rings of the lowered description `l` at `ctx` tokens of context and prompt chunks of `chunk`, each in bf16 and at the 12-bit pages' bytes where they
    /// take it (`twelve_bit_rings`; `[bf16, 12-bit]`): a sequence's bytes of the rings of a sliding window (fixed: the window and a chunk each, or the context where
    /// it is shorter, as the pools give them out: src/dynamic/rows.rs `page`), a token's keys and values in the others, and each pools' group's bf16 bytes a token
    /// (the windowed rings', the others'; a group of none left out: `rows::ring_groups`).
    fn ring_bytes(l: &crate::dynamic::ModelDesc, chunk: usize, ctx: usize) -> ([u64; 2], [u64; 2], Vec<u64>) {
        let pages = crate::dynamic::twelve_bit_rings(l);
        let (mut rings, mut per, mut groups) = ([0u64; 2], [0u64; 2], [0u64; 2]);
        for (r, x) in l.rings.iter().enumerate() {
            let bf16 = x.token_bytes() as u64;
            let at = [bf16, pages[r].map_or(bf16, |k| kv12_token_bytes(k.heads, k.channels, k.one))];
            let held = (crate::dynamic::seq_blocks(Some(x.window + chunk), ctx, 16) * 16) as u64;
            for t in 0..2 {
                if x.window > 0 {
                    rings[t] += held * at[t];
                } else {
                    per[t] += at[t];
                }
            }
            groups[(x.window == 0) as usize] += bf16;
        }
        (rings, per, groups.into_iter().filter(|&g| g > 0).collect())
    }

    /// The plan of the lossy route's levels, one rule for every model it takes, by what its description says (`level_of`: its weights, its sequences' bytes, its
    /// draft model, its experts): the context asked (refused where its family takes less), the users a server holds (each a sequence of its own in pools of blocks
    /// that hold up to the context of each: src/dynamic/rows.rs `page`), a draft model only for one user (a drafted engine runs one sequence at a time:
    /// `serve_rows`), the keys and values in the executor's 12-bit pages where a server with no draft model asks for them and they reserve no more than bf16
    /// (`one_gpu_pages`), the one-GPU rule (`one_gpu`); where the GPU does not hold the model whole (or `--expert-cache` asks for a cache), its experts in the
    /// computer's memory and their places on the GPU by the one expert-cache rule (`expert_cache_with`: every user's context first, the engine's work next, the
    /// places what is left), at least a step's choices and as many again a layer (a step's choices where the cache's GPU memory is asked). Refused in one form
    /// (`too_big`), naming a smaller download where one fits, and why the experts in the computer's memory do not help.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn lossy_plan_inner(
        a: &Args,
        model: &str,
        layout: Layout,
        c: &ModelConfig,
        gen: Option<&str>,
        free: u64,
        gpu: &str,
        packed: Option<&Path>,
        host: Option<u64>,
        devs: &[Target],
    ) -> Done<LoadPlan> {
        let name = crate::levels::short_name(model);
        let tag = level_name(layout);
        let asked = if a.has("context") || a.has("ctx") { Some(a.either("context", "ctx", 0usize)?) } else { None };
        let lv = level_of(a, model, &name, layout, c, gen, packed, asked)?;
        crate::support::context_check(&name, lv.row, lv.max_pos, asked)?;
        // (its layers in stages at this level: where its row cuts them and its loader loads a stage, `LevelOf::layers`; the GPUs, this one where none are given)
        let stages = lv.layers.as_ref().filter(|_| lv.row.split_levels.contains(&tag));
        let one = [Target { ordinal: 0, name: gpu.to_string(), cc: (0, 0), total: free, free, host, cublas: None }];
        let devs = if devs.is_empty() { &one[..] } else { devs };
        if a.flag("split").is_some() {
            let Some(cut) = stages else {
                return Err(Refusal::unsupported(
                    format!("{name} ({}) loads on one GPU at {tag}: its layers are not cut over GPUs at this level", lv.row.name),
                    "Leave out --split",
                ));
            };
            return lossy_chosen(a, &name, layout, &lv, cut, c, devs, asked);
        }
        // (served, the executor gives each of --users a sequence of its own in pools of blocks that hold up to the context of each (src/dynamic/rows.rs `page`): the
        // plan counts them. Measured on an A100: Qwen3.5-0.8B served to 8 took 5,054 MiB, planned for one at 3,037)
        let users = users_asked(a)?;
        // (a drafted engine runs one sequence at a time (src/dynamic/rows.rs `serve_rows`): a plan of more users than one holds them with no draft model, and the
        // loader loads none (src/cli.rs `open_lossy` drafts where the plan has a draft model))
        let draft = lv.draft.filter(|_| users == 1).unwrap_or(0);
        // (a server's keys and values in the executor's 12-bit pages where GLYD_KV=lossless asks for them, it has no draft model and its description has rings they
        // take, by its attention kinds: those rings at the pages' bytes a token, the others bf16, and what their pools spend once; taken by `one_gpu_pages`' rule)
        let twelve = twelve_pages(a) && draft == 0 && !lv.rings12.is_empty();
        let (n, pages) = (lv.needs(layout, users, false, served(a), draft), twelve.then(|| lv.needs(layout, users, true, served(a), draft)));
        let fitted = one_gpu_pages(&n, pages.as_ref(), free, asked, gpu);
        // (the flags of the experts in the computer's memory: refused for a description with none)
        let not_here = |flag: &str| {
            Refusal::new(format!("{flag}: {name} has no experts to keep in the computer's memory"), "It is for a model with experts, which it keeps there")
        };
        let cache = cache_flag(a)?;
        if cache.is_some() && lv.experts.is_none() {
            return Err(not_here("--expert-cache"));
        }
        let host_asked = host_flag(a)?;
        if host_asked.is_some() && lv.experts.is_none() {
            return Err(not_here("--host-memory"));
        }
        let mut also = String::new();
        // (a GPU that does not hold it whole, of a machine of several: its layers in stages over them, as the dynamic route cuts them, before its experts go to the
        // computer's memory with places on one GPU; not where --expert-cache asks for one GPU's cache)
        if let (Some(cut), true) = (stages, fitted.is_err() && cache.is_none() && devs.len() > 1) {
            match lossy_stages(&lv, cut, layout, c, devs, asked, host, host_asked) {
                Ok(p) => return Ok(p),
                Err(why) => also = format!("; in stages over its {} GPUs {why}", devs.len()),
            }
        }
        if let Some(src) = lv.experts.as_ref().filter(|_| fitted.is_err() || cache.is_some()) {
            // (its context the one asked, else LOSSY_CTX, every one of its users holding it)
            let ctx = asked.unwrap_or(LOSSY_CTX).min(lv.max_pos.max(LOSSY_MIN_CTX));
            match src.as_ref().map_err(String::clone).and_then(|src| experts_in_ram(src, n.kv(n.users as u64, ctx), layout, free, host, cache, host_asked, gpu))
            {
                Ok(plan) => return Ok(LoadPlan { experts: Some(plan), ..n.plan_at(ctx, n.chunks[0], free, gpu) }),
                Err(why) if cache.is_some() => {
                    return Err(Refusal::unsupported(
                        format!("{name} at the {tag} level with --expert-cache: {why}"),
                        "Close programs that use the GPU (nvidia-smi lists them), ask for a smaller cache, or use a GPU or a computer with more memory",
                    ))
                }
                Err(why) => also = format!("; {why}{also}"),
            }
        }
        fitted.map_err(|s| {
            // (a level that does not fit, of a model with a smaller download that does: that one first, with its draft model, at the shortest context a plan takes)
            let alt = lv.smaller.as_ref().and_then(|d| {
                let need = d.weights + n.per_seq + d.draft + LOSSY_WORK + reserve_of(free) + LOSSY_MIN_CTX as u64 * n.per_token;
                (need <= free)
                    .then(|| format!("glyd run {} fits: the {} level, about {} at {LOSSY_MIN_CTX} tokens of context ({})", d.repo, d.level, gib(need), d.what))
            });
            // (the need is one sequence's: the scheduler takes fewer users where the room holds fewer, `Needs::fewer`)
            let mut r = too_big(&name, tag, gpu, free, 1, &s, asked, draft > 0, alt);
            if !also.is_empty() {
                r.what += &also;
                r.fix = "Close programs that use the GPU (nvidia-smi lists them) or the computer's memory, or use a GPU or a computer with more memory".into();
            }
            r
        })
    }

    /// A lossy level's layers cut into stages over the GPUs `devs` by the dynamic route's rule (`size_dynamic_held`'s): each GPU's room its free memory less the
    /// reserve and the lossy engine's work (each stage is one engine); each layer its bytes at the level and its share of one sequence's keys, values and states
    /// at the context (`rows::layer_rooms`: stages hold one sequence, as the dynamic route's do), the rest (the embedding, the head) half with the first stage
    /// and half with the last, a stage beginning only where one may (`stage_starts`); every expert on the GPUs where they hold them (`plan_cuts_at`), else the
    /// experts in the computer's memory (`host` free, `asked_host` given) and each stage's places of them by the one expert-cache rule (`expert_cache_with` over
    /// the GPUs' rooms), at least a step's choices and as many again a layer. The context the one asked, else LOSSY_CTX (the model's longest where shorter) and
    /// its halves down to LOSSY_MIN_CTX: the longest that fits. The plan, or why not in words with the figures.
    #[allow(clippy::too_many_arguments)]
    fn lossy_stages(
        lv: &LevelOf,
        cut: &LayerBytes,
        layout: Layout,
        c: &ModelConfig,
        devs: &[Target],
        asked: Option<usize>,
        host: Option<u64>,
        asked_host: Option<u64>,
    ) -> Result<LoadPlan, String> {
        let nl = cut.whole.len();
        let room: Vec<(i32, u64)> = devs.iter().map(|t| (t.ordinal, t.free.saturating_sub(LOSSY_WORK + reserve_of(t.free)))).collect();
        let (starts, rest) = (stage_starts(c, nl), lv.weights.saturating_sub(cut.whole.iter().sum()));
        let top = LOSSY_CTX.min(lv.max_pos.max(LOSSY_MIN_CTX));
        let tries: Vec<usize> = match asked {
            Some(x) => vec![x],
            None => std::iter::successors(Some(top), |&x| (x / 2 >= LOSSY_MIN_CTX).then_some(x / 2)).collect(),
        };
        let (free, fixed) = (devs.iter().map(|t| t.free).sum::<u64>(), devs.iter().map(|t| LOSSY_WORK + reserve_of(t.free)).sum::<u64>());
        let mut why = String::new();
        for ctx in tries {
            let kv = crate::dynamic::layer_rooms(&cut.low, ctx, lv.chunk);
            let per: Vec<u64> = cut.whole.iter().zip(&kv).map(|(w, k)| w + k).collect();
            if let Some((cuts, devices)) = plan_cuts_at(&per, rest, &room, &starts) {
                return Ok(staged(lv, cut, layout, ctx, &kv, devs, cuts, devices, None));
            }
            let seq = kv.iter().sum::<u64>();
            let Some(src) = lv.experts.as_ref() else {
                why = format!("it needs about {} of GPU memory at {ctx} tokens of context, and they have {} free", gib(lv.weights + seq + fixed), gib(free));
                continue;
            };
            let src = src.as_ref().map_err(String::clone)?;
            let least = 2 * src.top_k;
            let ask = CacheAsk { least: Some(least), ram_least: Some(least), host_memory: asked_host };
            match expert_cache_with(src, ask, &room, &kv, host) {
                Ok(k) => return Ok(staged(lv, cut, layout, ctx, &kv, devs, k.cuts.clone(), k.devices.clone(), Some((src, k)))),
                Err(CacheShort::Memory { free: h }) => {
                    why = format!(
                        "with its experts in the computer's memory it needs {} of the computer's memory, and this computer has {} free",
                        gib(src.all + HOST_MARGIN),
                        gib(h)
                    )
                }
                Err(CacheShort::Asked { ram, free: h }) => {
                    why = format!("--host-memory {}: the computer has {} free and keeps {} of them for the system", gib(ram), gib(h), gib(HOST_MARGIN))
                }
                Err(_) => {
                    let need = src.base + src.dense.iter().sum::<u64>() + seq + fixed + (src.layers() * least) as u64 * src.place;
                    why = format!(
                        "with its experts in the computer's memory it needs at least {} of GPU memory at {ctx} tokens of context (and {} of the computer's), and they have {} free",
                        gib(need),
                        gib(src.all + HOST_MARGIN),
                        gib(free)
                    )
                }
            }
        }
        Err(why)
    }

    /// `--split CUTS [--devices GPUS]` at a lossy level whose layers are cut (`lossy_plan_inner`): the layers before each cut on the GPUs named in order
    /// (`split::split_args`: GPU 0 for each where none are named; a GPU named twice holds both stages), every expert on them, one sequence at the context asked
    /// (else LOSSY_CTX, the model's longest where shorter); refused in words where the cuts are not inside the layers in order with one GPU more than cuts, a
    /// stage begins at a layer that takes a part of one before it (`stage_starts`), a GPU is not this machine's, or a GPU's stages take more than its room.
    #[allow(clippy::too_many_arguments)]
    fn lossy_chosen(
        a: &Args,
        name: &str,
        layout: Layout,
        lv: &LevelOf,
        cut: &LayerBytes,
        c: &ModelConfig,
        devs: &[Target],
        asked: Option<usize>,
    ) -> Done<LoadPlan> {
        let (cuts, devices) = crate::dynamic::split::split_args(a).map_err(|e| Refusal::new(e.msg, "--split 31 --devices 0,1"))?;
        let nl = cut.whole.len();
        if devices.len() != cuts.len() + 1 || cuts.windows(2).any(|w| w[0] >= w[1]) || cuts.iter().any(|&x| x == 0 || x >= nl) {
            return Err(Refusal::new(
                format!("--split {cuts:?} --devices {devices:?} over {nl} layers: ascending cuts inside the layers, one GPU more than cuts"),
                "--split 31 --devices 0,1: the layers before 31 on GPU 0, the rest on GPU 1",
            ));
        }
        let starts = stage_starts(c, nl);
        if let Some(l) = cuts.iter().copied().find(|&l| !starts.get(l).copied().unwrap_or(true)) {
            let firsts = (1..nl).filter(|&l| starts.get(l).copied().unwrap_or(true)).take(6).map(|l| l.to_string()).collect::<Vec<_>>().join(", ");
            let why = if crate::dynamic::lower_comp::shares_parts(c) {
                crate::dynamic::lower_comp::split_apart_parts(l)
            } else {
                crate::dynamic::lower::split_apart_picks(l)
            };
            return Err(Refusal::unsupported(why, format!("A stage may begin at: {firsts}")));
        }
        if let Some(d) = devices.iter().find(|d| !devs.iter().any(|t| t.ordinal == **d)) {
            return Err(Refusal::new(format!("--devices: there is no GPU {d} here"), "Name the GPUs nvidia-smi lists"));
        }
        let ctx = asked.unwrap_or(LOSSY_CTX.min(lv.max_pos.max(LOSSY_MIN_CTX)));
        let kv = crate::dynamic::layer_rooms(&cut.low, ctx, lv.chunk);
        let rest = lv.weights.saturating_sub(cut.whole.iter().sum());
        let bounds: Vec<usize> = std::iter::once(0).chain(cuts.iter().copied()).chain(std::iter::once(nl)).collect();
        // (each GPU's stages: their layers' bytes and their share of the sequence, the rest half with the first and half with the last)
        for t in devs.iter().filter(|t| devices.contains(&t.ordinal)) {
            let on = devices.iter().filter(|&&d| d == t.ordinal).count() as u64;
            let held: u64 = (0..devices.len())
                .filter(|&k| devices[k] == t.ordinal)
                .map(|k| {
                    let (from, to) = (bounds[k], bounds[k + 1]);
                    (from..to).map(|l| cut.whole[l] + kv[l]).sum::<u64>()
                        + if k == 0 { rest / 2 } else { 0 }
                        + if k + 1 == devices.len() { rest - rest / 2 } else { 0 }
                })
                .sum();
            // (each stage its own engine's work: a GPU named twice holds two)
            let room = t.free.saturating_sub(on * LOSSY_WORK + reserve_of(t.free));
            if held > room {
                return Err(Refusal::unsupported(
                    format!(
                        "{name} at the {} level cut before {cuts:?}: GPU {} takes about {} of its stages' weights and sequence at {ctx} tokens of context, and has {} for them",
                        level_name(layout),
                        t.ordinal,
                        gib(held),
                        gib(room)
                    ),
                    "Cut elsewhere, or leave out --split: the plan cuts the layers as the GPUs hold them",
                ));
            }
        }
        Ok(staged(lv, cut, layout, ctx, &kv, devs, cuts, devices, None))
    }

    /// The plan of a lossy level in stages (`lossy_stages`, `lossy_chosen`): stage k the layers from its cut on `devices[k]`, one sequence of `ctx` tokens (`kv`
    /// each layer's share), each GPU the lossy engine's work and the reserve; every expert on the GPUs, or (`cached`: its source and the rule's plan) in the
    /// computer's memory and each stage's places of them.
    #[allow(clippy::too_many_arguments)]
    fn staged(
        lv: &LevelOf,
        cut: &LayerBytes,
        layout: Layout,
        ctx: usize,
        kv: &[u64],
        devs: &[Target],
        cuts: Vec<usize>,
        devices: Vec<i32>,
        cached: Option<(&CacheOf, Cached)>,
    ) -> LoadPlan {
        let used: Vec<&Target> = devs.iter().filter(|t| devices.contains(&t.ordinal)).collect();
        let seq = kv.iter().sum::<u64>();
        let routed: Vec<u64> = cut.whole.iter().zip(&cut.dense).map(|(w, d)| w.saturating_sub(*d)).collect();
        let (gpu_bytes, experts, places, per_layer, host_bytes) = match &cached {
            Some((src, k)) => (
                k.on_gpus.saturating_sub(seq),
                Some(crate::offload::CachePlan {
                    layers: src.layers(),
                    experts: src.experts,
                    slots: k.places.iter().min().copied().unwrap_or(0),
                    expert_bytes: src.place,
                    in_place: false,
                    ram: k.ram,
                }),
                k.places.clone(),
                k.experts,
                k.all,
            ),
            None => (lv.weights, None, Vec::new(), 0, 0),
        };
        LoadPlan {
            kind: lv.kind,
            family: lv.row.name,
            level: level_name(layout),
            layout,
            context: ctx,
            users: 1,
            chunk: lv.chunk,
            weights: lv.weights,
            bf16: lv.bf16,
            kv: seq,
            kv12: false,
            // (each stage its engine's work: a GPU named twice works for two)
            work: devices.len() as u64 * LOSSY_WORK,
            draft: 0,
            reserve: used.iter().map(|t| reserve_of(t.free)).sum(),
            free: used.iter().map(|t| t.free).sum(),
            gpu: used.iter().map(|t| t.name.as_str()).collect::<Vec<_>>().join(", "),
            experts,
            spread: Some(crate::plan::Spread {
                host_per_stage: stage_shares(&routed, &cuts, host_bytes),
                cuts,
                devices,
                across: false,
                coded: false,
                twelve: false,
                places,
                experts_per_layer: per_layer,
                host_bytes,
                gpu_bytes,
                dense: 0,
                dense_bytes: 0,
                tables: 0,
            }),
        }
    }

    /// A lossy level's experts in the computer's memory (`src`) and their places on its one GPU of `free` bytes free, by the one expert-cache rule
    /// (`expert_cache_with`): its users' keys, values and states `kv` first, the engine's work and the reserve next, the places what is left (or `cache` bytes of
    /// the GPU's, `--expert-cache`), at least a step's choices and as many again a layer (a step's choices for an asked cache); every expert in the computer's
    /// memory (`host` free; None: it does not say, and is taken to have enough), or at the hummingbird level what it holds (`asked_host`: `--host-memory`), the
    /// packed directory the rest, at least a step's choices in each width of each layer. The cache's plan, or why not in words.
    #[allow(clippy::too_many_arguments)]
    fn experts_in_ram(
        src: &CacheOf,
        kv: u64,
        layout: Layout,
        free: u64,
        host: Option<u64>,
        cache: Option<u64>,
        asked_host: Option<u64>,
        gpu: &str,
    ) -> Result<CachePlan, String> {
        let (layers, k) = (src.layers(), src.top_k);
        // (the weights besides the experts: each layer's and the rest)
        let held = src.base + src.dense.iter().sum::<u64>();
        let rest = held + LOSSY_WORK + reserve_of(free) + kv;
        let least = if cache.is_some() { k } else { 2 * k };
        let need = rest + (layers * least) as u64 * src.place;
        if let Some(b) = cache.filter(|&b| rest.saturating_add(b) > free) {
            return Err(format!(
                "a cache of {} for the experts and the {} the rest of the model takes are more than the {gpu}'s {} free",
                gib(b),
                gib(rest),
                gib(free)
            ));
        }
        let room = cache.map_or(free.saturating_sub(LOSSY_WORK + reserve_of(free)), |b| held + kv + b);
        let (host_need, least_ram) = (src.all + HOST_MARGIN, (layers * 2 * k) as u64 * src.place);
        let ask = CacheAsk { least: Some(least), ram_least: Some(2 * k), host_memory: asked_host };
        let mut kvs = vec![0; src.dense.len().max(1)];
        kvs[0] = kv;
        match expert_cache_with(src, ask, &[(0, room)], &kvs, host) {
            Ok(c) => Ok(CachePlan {
                layers,
                experts: src.experts,
                slots: c.places.first().copied().unwrap_or(0),
                expert_bytes: src.place,
                in_place: layout == Layout::Hummingbird,
                ram: c.ram,
            }),
            Err(CacheShort::Gpu) => Err(format!(
                "with its experts in the computer's memory it needs at least {} of GPU memory (and {} of the computer's)",
                gib(need),
                gib(if src.disk { least_ram + HOST_MARGIN } else { host_need })
            )),
            Err(CacheShort::Memory { free: h }) => Err(format!(
                "with its experts in the computer's memory it needs {} of the computer's memory besides at least {} of GPU memory, and this computer has {} free",
                gib(host_need),
                gib(need),
                gib(h)
            )),
            Err(CacheShort::Asked { ram, free: h }) => {
                Err(format!("--host-memory {}: the computer has {} free and keeps {} of them for the system", gib(ram), gib(h), gib(HOST_MARGIN)))
            }
            Err(CacheShort::Held { ram, least }) => Err(format!(
                "with its experts read from the disk it needs at least {} of the computer's memory for them ({} for a step's choices in each width of each layer), and {} are to be given",
                gib(least + HOST_MARGIN),
                gib(least),
                gib(ram)
            )),
        }
    }

    /// The experts of a description as a lossy level holds them (`MoeShape::of`): which layers have a block of the Moe kind, a layer's experts and a token's, an
    /// expert's two matrices' shapes (its first's rows: twice its width inside where it is gated; the width it reads: its latent's where it works in one, else the
    /// model's), and whether the release stores them quantized, each matrix with its scales (`Described::scales`).
    pub(crate) struct MoeShape {
        moe: Vec<bool>,
        experts: usize,
        top_k: usize,
        inter: usize,
        width: usize,
        gated: bool,
        latent: usize,
        scaled: bool,
    }

    impl MoeShape {
        /// The experts of the description `d` of the config `m` (its layers `num_hidden_layers`, each a share of the blocks), each block of the Moe kind's
        /// matrices by its second's shape (`experts_down`: [the width it reads, its width inside]); None where it has none.
        pub(crate) fn of(d: &crate::runtime::describe::Described, m: &ModelConfig) -> Option<MoeShape> {
            use crate::runtime::describe::Source;
            use crate::runtime::kinds::{Body, Ffn};
            let (nl, nb) = (m.num("num_hidden_layers").unwrap_or(0), d.desc.blocks.len().max(1));
            let mut x = MoeShape { moe: vec![false; nl], experts: 0, top_k: 0, inter: 0, width: 0, gated: false, latent: 0, scaled: false };
            let mut seen = false;
            for (b, (blk, roles)) in d.desc.blocks.iter().zip(&d.blocks).enumerate() {
                let Body::Ffn(Ffn::Moe(e)) = &blk.body else { continue };
                if let Some(f) = x.moe.get_mut(b * nl / nb) {
                    *f = true;
                }
                (x.experts, x.top_k, x.gated, x.latent) =
                    (x.experts.max(e.experts), x.top_k.max(e.router.top_k), e.expert.gated, e.latent.as_ref().map_or(0, |l| l.dim));
                let down = roles.iter().find_map(|(r, s)| match s {
                    Source::Experts(v) if r == "experts_down" => v.first().and_then(|t| t.first()).and_then(|n| d.shapes.get(n)),
                    _ => None,
                });
                if let Some(&[w, i, ..]) = down.map(Vec::as_slice) {
                    (x.width, x.inter) = (w, i);
                }
                // (stored quantized: every matrix of an expert, the first's as each's, with its scales)
                let scaled = roles.iter().filter(|(r, _)| r.starts_with("experts_")).all(|(_, s)| match s {
                    Source::Experts(v) => v.first().is_some_and(|e| e.iter().all(|n| d.scales.contains_key(n))),
                    _ => false,
                });
                x.scaled = scaled && (x.scaled || !seen);
                seen = true;
            }
            x.moe.contains(&true).then_some(x)
        }

        /// An expert's bytes at `layout`, a place of the cache: its two matrices as the block layout packs them (`moe::expert_strides`; the second in its row
        /// layout takes the same bytes), at the hummingbird level the mean of the recipe's (its kernels take experts on a latent of 128 to 1,024, a power of two);
        /// at any other layout an error that says so.
        pub(crate) fn place(&self, layout: Layout) -> crate::ffi::Result<u64> {
            let tag = match layout {
                Layout::Hummingbird if !(self.latent.is_power_of_two() && (128..=1024).contains(&self.latent)) => {
                    return crate::ffi::err(format!(
                        "the hummingbird level takes experts on a latent of 128 to 1,024; this model's experts read its width of {}",
                        self.width
                    ))
                }
                Layout::Hummingbird => return crate::moe_hb::expert_bytes_mean(self.experts, self.inter, self.latent),
                Layout::Int6 => glyd_gpu::pack::INT6_TAG,
                Layout::Int5 => glyd_gpu::pack::INT5_TAG,
                Layout::Int4 => glyd_gpu::pack::INT4_TAG,
                other => {
                    return crate::ffi::err(format!(
                        "its experts are held at the kestrel, swift and wren levels on this path, not in the {} layout",
                        other.name()
                    ))
                }
            };
            let up = if self.gated { 2 * self.inter } else { self.inter };
            let ((ud, us), (dd, ds)) = (crate::moe::expert_strides(tag, up, self.width)?, crate::moe::expert_strides(tag, self.width, self.inter)?);
            Ok(ud + us + dd + ds)
        }

        /// An expert's bytes as a loader that packs each expert itself holds it (`ExpertCache::Packed`): its two matrices in each layout of `tags`, the largest
        /// (the layout its scales allow is the load's); an error where the release does not store them quantized with their scales, the form such a loader
        /// packs from.
        pub(crate) fn packed(&self, tags: &[u32]) -> crate::ffi::Result<u64> {
            if !self.scaled {
                return crate::ffi::err("its loader packs experts released quantized, each matrix with its scales, and this model's experts are not");
            }
            let up = if self.gated { 2 * self.inter } else { self.inter };
            let mut most = 0;
            for &tag in tags {
                let ((ud, us), (dd, ds)) = (crate::moe::expert_strides(tag, up, self.width)?, crate::moe::expert_strides(tag, self.width, self.inter)?);
                most = most.max(ud + us + dd + ds);
            }
            Ok(most)
        }

        /// Its experts at `layout` as the one expert-cache rule takes them (`expert_cache_with`): a place's bytes (`place`), every expert's, and the rest of the
        /// model (its `weights` at the level less the experts', as the plan counts them: `LoadPlan::weights_on_gpu`) held whole: each layer's own (`dense`, its
        /// bytes besides its routed experts: `layer_items`; none given, every byte the rest's), the rest what they leave, and the layers a stage may begin at
        /// (`stage_starts`), so the rule cuts them into stages as the dynamic route's; at the hummingbird level the disk holds the experts the computer's memory
        /// does not.
        pub(crate) fn cache_of(&self, m: &ModelConfig, layout: Layout, weights: u64, dense: &[u64]) -> crate::ffi::Result<CacheOf> {
            let (nl, place) = (self.moe.len(), self.place(layout)?);
            let all = (self.moe.iter().filter(|&&f| f).count() * self.experts) as u64 * place;
            let dense = if dense.len() == nl { dense.to_vec() } else { vec![0; nl] };
            Ok(CacheOf {
                base: weights.saturating_sub(all).saturating_sub(dense.iter().sum()),
                dense,
                ram: Vec::new(),
                moe: self.moe.clone(),
                place,
                experts: self.experts,
                top_k: self.top_k,
                all,
                disk: layout == Layout::Hummingbird,
                across: 1,
                starts: stage_starts(m, nl),
            })
        }
    }

    /// An expert cache of `budget` bytes of a GPU's memory and nothing else on it (a development check's `--expert-cache GIB`) for the model of the config `m` at
    /// `layout`: the places a layer the one expert-cache rule gives (`expert_cache_with`), at least `least`, every expert in the computer's memory; None where fewer
    /// than `least` fit; an error where it has no experts.
    pub fn budget_cache(m: &ModelConfig, layout: Layout, budget: u64, least: usize) -> crate::ffi::Result<Option<CachePlan>> {
        let d = crate::runtime::describe::describe_config(m)?;
        let x = MoeShape::of(&d, m).ok_or_else(|| crate::ffi::Error("no experts to cache".into()))?;
        let src = x.cache_of(m, layout, 0, &[])?;
        let ask = CacheAsk { least: Some(least), ram_least: None, host_memory: None };
        Ok(expert_cache_with(&src, ask, &[(0, budget)], &[], None).ok().map(|c| CachePlan {
            layers: src.layers(),
            experts: src.experts,
            slots: c.places.first().copied().unwrap_or(0),
            expert_bytes: src.place,
            in_place: layout == Layout::Hummingbird,
            ram: src.all,
        }))
    }

    /// What an expert cache is planned from, read by one rule whatever the source of its experts and whatever the route (`expert_cache_with`: a release's,
    /// `release_experts`; a loader's own packs, `packed_experts`; a hummingbird directory's, `hb_dir_experts`; a lossy level's, `MoeShape::cache_of`): each
    /// layer's weights besides its experts (`dense`, the rest `base`) and the bytes of them its loader can keep in the computer's memory (`ram`, src/dense_ram.rs),
    /// which layers have experts, a place's bytes (an expert's, their mean where they differ), a layer's experts and a token's, every expert's bytes, whether
    /// those the computer's memory does not hold are read from the disk (`disk`: a hummingbird directory's, the hummingbird level's) or all must be in it, the
    /// GPUs every layer is across (src/dynamic/tp.rs: each holds `dense` and `base`, its share of them already, and a slice of each place, `place` a slice's
    /// bytes; 1 where the layers are cut over the GPUs, each its own), and the layers a stage may begin at (`stage_starts`; every one where empty).
    #[derive(Clone, Debug)]
    pub(crate) struct CacheOf {
        pub dense: Vec<u64>,
        pub base: u64,
        pub ram: Vec<u64>,
        pub moe: Vec<bool>,
        pub place: u64,
        pub experts: usize,
        pub top_k: usize,
        pub all: u64,
        pub disk: bool,
        pub across: usize,
        pub starts: Vec<bool>,
    }

    impl CacheOf {
        /// A token's experts in every layer of experts: the least the places of a layer hold.
        pub(crate) fn token_bytes(&self) -> u64 {
            self.layers() as u64 * self.top_k.max(1) as u64 * self.place
        }

        /// The layers of experts.
        pub(crate) fn layers(&self) -> usize {
            self.moe.iter().filter(|&&m| m).count()
        }
    }

    /// An expert cache's plan (`expert_cache_with`): the cuts, their GPUs, each stage's places a layer, the GPUs' bytes (the weights besides the experts, the
    /// places and the sequences' bytes the layers were given), every expert's bytes, a layer's experts, the experts' bytes the computer's memory holds (the rest
    /// read from the disk), and the last layers whose dense matrices are in the computer's memory, with their bytes.
    #[derive(Clone, Debug)]
    pub(crate) struct Cached {
        pub cuts: Vec<usize>,
        pub devices: Vec<i32>,
        pub places: Vec<usize>,
        pub on_gpus: u64,
        pub all: u64,
        pub experts: usize,
        pub ram: u64,
        pub dense: (usize, u64),
    }

    /// What a route asks of the expert-cache rule beside its room: the least places a layer on the GPUs (None: a token's experts), the least of a layer's
    /// experts in the computer's memory where the disk holds the rest (None: a token's), and the computer's memory `--host-memory` gives the experts.
    #[derive(Clone, Copy, Debug, Default)]
    pub(crate) struct CacheAsk {
        pub least: Option<usize>,
        pub ram_least: Option<usize>,
        pub host_memory: Option<u64>,
    }

    /// Why the expert-cache rule plans no cache, with its figures (each route says it in its own words): the computer's memory does not hold every expert
    /// (`free` of it free); `--host-memory` asks for more than the computer has free less `HOST_MARGIN`; what the computer's memory holds of them where the disk
    /// holds the rest is less than the least (`least` bytes); the GPUs leave too little room for the least places beside the layers' weights and sequences.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum CacheShort {
        Memory { free: u64 },
        Asked { ram: u64, free: u64 },
        Held { ram: u64, least: u64 },
        Gpu,
    }

    /// The one rule of an expert cache, every route's (the dynamic route's penguin and hummingbird, a lossy level's experts in the computer's memory): the experts
    /// of `c` in the computer's memory (`host` free; None: it does not say, and is taken to have enough), less `HOST_MARGIN`; where the disk holds those it does
    /// not (`c.disk`), what it holds (`ask.host_memory`, else what is free), at least `ask.ram_least` of a layer's experts in every layer. On the GPUs of `room`:
    /// each layer's weights and its sequences' bytes `kv` first (a route gives them: each layer's share of one sequence's, and a server's users' beside on the
    /// first layer where one GPU holds them), the places what the room left gives, at least `ask.least` a layer (`split::places_plan_dense`: the last layers'
    /// dense matrices in the computer's memory where a token's experts do not fit beside them on one GPU; a stage begins only where one may, `CacheOf::starts`).
    /// Every layer across the GPUs (`c.across`): each GPU holds its share and the sequences' bytes whole, the least room of them sets the places, every GPU
    /// the same. Why not, with the figures, where the computer's memory or the GPUs hold too little.
    pub(crate) fn expert_cache_with(c: &CacheOf, ask: CacheAsk, room: &[(i32, u64)], kv: &[u64], host: Option<u64>) -> Result<Cached, CacheShort> {
        if let Some(h) = host.filter(|&h| !c.disk && c.all + HOST_MARGIN > h) {
            return Err(CacheShort::Memory { free: h });
        }
        // (the computer's memory for the experts: all of them, else (the disk holding the rest) what it gives, at least the least of each layer's)
        let ram = if c.disk { held_in_memory(c.all, ask.host_memory, host) } else { c.all };
        if let Some(h) = host.filter(|&h| c.disk && ask.host_memory.is_some() && ram.saturating_add(HOST_MARGIN) > h) {
            return Err(CacheShort::Asked { ram, free: h });
        }
        let least_ram = c.layers() as u64 * ask.ram_least.unwrap_or(c.top_k).max(1) as u64 * c.place;
        if c.disk && ram < least_ram {
            return Err(CacheShort::Held { ram, least: least_ram });
        }
        // (no places of no bytes: a source with experts of none is no cache)
        if c.place == 0 {
            return Err(CacheShort::Gpu);
        }
        let mut dense = c.dense.clone();
        dense.iter_mut().zip(kv).for_each(|(d, k)| *d += k);
        let nl = dense.len();
        let least = ask.least.unwrap_or(c.top_k).max(1).min(c.experts.max(1));
        // (every layer across the GPUs: one GPU's share planned on the least room of them, every GPU the same)
        let n = c.across.max(1);
        let one: Vec<(i32, u64)> = room.iter().copied().min_by_key(|r| r.1).into_iter().collect();
        let (at, ram_of) = if n > 1 { (&one[..], &[][..]) } else { (room, &c.ram[..]) };
        let (cuts, devices, places, held, freed) =
            crate::dynamic::split::places_plan_dense(&dense, c.base, &c.moe, at, c.place, c.experts, least, ram_of, &c.starts).ok_or(CacheShort::Gpu)?;
        let bounds: Vec<usize> = std::iter::once(0).chain(cuts.iter().copied()).chain(std::iter::once(nl)).collect();
        let on_gpus = dense.iter().sum::<u64>() - freed
            + c.base
            + bounds.windows(2).zip(&places).map(|(w, &p)| c.moe[w[0]..w[1]].iter().filter(|&&m| m).count() as u64 * p as u64 * c.place).sum::<u64>();
        if n > 1 {
            let p = places.first().copied().unwrap_or(0);
            return Ok(Cached {
                cuts: Vec::new(),
                devices: room.iter().map(|r| r.0).collect(),
                places: vec![p; room.len()],
                on_gpus: n as u64 * on_gpus,
                all: c.all,
                experts: c.experts,
                ram,
                dense: (0, 0),
            });
        }
        Ok(Cached { cuts, devices, places, on_gpus, all: c.all, experts: c.experts, ram, dense: (held, ram_held(&c.ram, &dense, held)) })
    }

    /// `expert_cache_with` on the dynamic route (a token's experts a layer at least; `host` None: none free), or the refusal `run` gives, in its words. `hb`: the
    /// model's row runs at the hummingbird level too (src/support.rs), the way out a refusal for the computer's memory then names; a row without it is not told of one.
    pub(crate) fn expert_cache(c: &CacheOf, room: &[(i32, u64)], name: &str, kv: &[u64], host: Option<u64>, hb: bool) -> Done<Cached> {
        let host = host.unwrap_or(0);
        expert_cache_with(c, CacheAsk::default(), room, kv, Some(host)).map_err(|why| match why {
            CacheShort::Memory { .. } if c.across > 1 => Refusal::unsupported(
                format!(
                    "{name} keeps the {} experts the GPUs do not hold in the computer's memory: they take {}, and {} is free",
                    c.layers() * c.experts,
                    gib(c.all),
                    gib(host)
                ),
                "A machine with that much memory runs it",
            ),
            CacheShort::Memory { .. } => Refusal::unsupported(
                format!("{name} keeps the experts its GPUs do not hold in the computer's memory: they take {}, and {} is free", gib(c.all), gib(host)),
                if hb { "A machine with that much memory runs it, or its hummingbird level, whose experts the disk holds too" } else { "A machine with that much memory runs it" },
            ),
            CacheShort::Asked { .. } | CacheShort::Held { .. } => Refusal::unsupported(
                format!(
                    "{name} at the hummingbird level keeps its experts in the computer's memory and on the disk: they take {}, and {} of memory is free",
                    gib(c.all),
                    gib(host)
                ),
                "A machine with more memory runs it",
            ),
            CacheShort::Gpu if c.across > 1 => Refusal::unsupported(
                format!(
                    "{name}: its weights besides the experts take {} a GPU across {} GPUs, and they leave too little room for its experts",
                    gib(c.dense.iter().sum::<u64>() + c.base + kv.iter().sum::<u64>()),
                    c.across
                ),
                "A machine whose GPUs have more memory free runs it",
            ),
            CacheShort::Gpu => Refusal::unsupported(
                format!(
                    "{name}: its weights besides the experts take {}, its sequences' keys, values and states {}, and the GPUs here leave too little room for its experts ({} for a token's in every layer)",
                    gib(c.dense.iter().sum::<u64>() + c.base),
                    gib(kv.iter().sum()),
                    gib(c.token_bytes())
                ),
                "A machine whose GPUs have more memory free runs it",
            ),
        })
    }

    /// The experts of a row whose loader packs them itself (`ExpertCache::Packed`: each expert in the layouts of `tags`, every one in the computer's memory), from
    /// the Moe kind's shape in the description of `m` (its config, parsed once: `MoeShape`, an expert's bytes `MoeShape::packed`), with the tensors of `table`,
    /// its `nl` layers, its bf16 matrices in the 12-bit layout where `twelve` (an HBM GPU's faster product, a few hundred places fewer). `across` GPUs every
    /// layer is across (src/dynamic/tp.rs; 1: its layers cut): each GPU holds its slice of the rows of each big matrix (the embedding whole, and every matrix
    /// too small or of rows that do not divide), its release's matrices besides the experts as released but bf16's, and a slice of each place.
    pub(crate) fn packed_experts(table: &crate::sizes::Table, m: &ModelConfig, nl: usize, twelve: bool, across: usize, tags: &[u32]) -> Done<CacheOf> {
        let d = crate::runtime::describe::describe_config(m).map_err(cannot)?;
        let x = MoeShape::of(&d, m).ok_or_else(|| cannot("its description has no block of experts"))?;
        let bf16 = if twelve { crate::dynamic::split::TWELVE_PCT } else { crate::dynamic::split::TIERED_PCT };
        let expert = x.packed(tags).map_err(cannot)?;
        let all = (x.moe.iter().filter(|&&f| f).count() * x.experts) as u64 * expert;
        let of = |dense, base, ram, place| CacheOf {
            dense,
            base,
            ram,
            moe: x.moe.clone(),
            place,
            experts: x.experts,
            top_k: x.top_k,
            all,
            disk: false,
            across: across.max(1),
            starts: if across > 1 { Vec::new() } else { stage_starts(m, nl) },
        };
        if across > 1 {
            let n = across as u64;
            let mut held = 0u64;
            for t in &table.rows {
                if routed(&t.name) || matches!(crate::sizes::classify(&t.name, nl), crate::sizes::Part::Vision | crate::sizes::Part::Mtp) {
                    continue;
                }
                let matrix = t.shape.len() == 2 && t.shape.iter().all(|&d| d >= 256);
                let b = if t.dtype == "BF16" && matrix { t.bytes * bf16 / 100 } else { t.bytes };
                let sliced = matrix && !t.name.contains("embed_tokens") && t.shape[0] % (64 * n) == 0 && t.bytes >= 16 << 20;
                held += if sliced { b / n } else { b };
            }
            return Ok(of(vec![0; x.moe.len()], held, Vec::new(), expert / n));
        }
        let (dense, base) = crate::dynamic::split::layer_bytes_table(table, nl, crate::dynamic::split::CODED_PCT, bf16, &|n| !routed(n));
        let family = crate::dynamic::split::family_in(m).unwrap_or("");
        let ram = crate::dynamic::split::ram_bytes_table(table, nl, family, crate::dynamic::split::CODED_PCT, bf16);
        Ok(of(dense, base, ram, expert))
    }

    /// Whether a table's tensor is a routed expert's (a matrix of one, its scales, or a stack of every expert's): by the word every naming convention gives
    /// them (a shared expert's is not one).
    fn routed(name: &str) -> bool {
        name.contains(".experts.")
    }

    /// The bytes the last `held` layers' dense matrices take in the computer's memory (`split::places_plan_dense`'s layers).
    pub(crate) fn ram_held(ram: &[u64], dense: &[u64], held: usize) -> u64 {
        let n = ram.len().min(dense.len());
        (n - held.min(n)..n).map(|l| ram[l].min(dense[l])).sum()
    }

    /// A hummingbird directory's experts (src/hbg_store.rs), for any family (`m`: its config, parsed once): places at the mean of the experts' bytes, those the
    /// computer's memory does not hold read from the expert files as the router asks for them (the disk).
    pub(crate) fn hb_dir_experts(dir: &Path, m: &ModelConfig, nl: usize, twelve: bool) -> Done<CacheOf> {
        let hb = crate::hb_pack::HbDir::open(dir).map_err(cannot)?.ok_or_else(|| cannot(format!("{}: not a hummingbird directory", dir.display())))?;
        let bf16 = if twelve { crate::dynamic::split::TWELVE_PCT } else { crate::dynamic::split::TIERED_PCT };
        let (dense, base) = crate::dynamic::split::layer_bytes_with(dir, nl, crate::dynamic::split::CODED_PCT, bf16, &|_| true, false).map_err(cannot)?;
        let sh = crate::plan::Shape::from_model(m).map_err(cannot)?;
        let mut moe = vec![false; nl];
        let (mut all, mut experts, mut count) = (0u64, 0usize, 0usize);
        for r in &hb.layers {
            if let Some(m) = moe.get_mut(r.block / 2) {
                *m = true;
            }
            all += crate::hbg_store::layer_store_bytes(r).map_err(cannot)?;
            experts = experts.max(r.bits.len());
            count += r.bits.len();
        }
        let family = crate::dynamic::split::family_in(m).unwrap_or("");
        let ram = crate::dynamic::split::ram_bytes(dir, nl, family, crate::dynamic::split::CODED_PCT, bf16).map_err(cannot)?;
        let place = all / count.max(1) as u64;
        Ok(CacheOf { dense, base, ram, moe, place, experts, top_k: sh.topk, all, disk: true, across: 1, starts: stage_starts(m, nl) })
    }

    /// A release's experts as the cache takes them, each the same bytes (src/tiering.rs, R12: DeepSeek-V4's FP4 experts; GLM-5.3-Flash's FP8 ones, each packed on
    /// its own in places of the largest's bytes: about 7 bits a value where the release's bytes, which size the places here, are 8; bf16 ones (Muse Spark's, the
    /// Qwen3.5 layout's) in the lossless tiered layout, about 11 bits of 16, sized at a bf16 matrix's share as held): a place the mean of the experts' bytes as
    /// held, every one in the computer's memory (the disk holds hummingbird directories' experts alone). `m`: its config, parsed once.
    pub(crate) fn release_experts(table: &crate::sizes::Table, m: &ModelConfig, nl: usize, twelve: bool) -> Done<CacheOf> {
        let expert = routed;
        let bf16 = if twelve { crate::dynamic::split::TWELVE_PCT } else { crate::dynamic::split::TIERED_PCT };
        let (dense, base) = crate::dynamic::split::layer_bytes_table(table, nl, crate::dynamic::split::CODED_PCT, bf16, &|n| !expert(n));
        // (each expert as the cache holds it, in a place and in the computer's memory: a bf16 one in the lossless tiered layout, at the share of its bytes the
        // planner gives a bf16 matrix (`bf16`, an upper bound); an FP8 or FP4 one as released)
        // (an expert's matrix, or a stack of every expert's: its last two dimensions a matrix's)
        let bf16_matrix = |r: &crate::sizes::Row| r.dtype == "BF16" && r.shape.len() >= 2 && r.shape.iter().rev().take(2).all(|&d| d >= 256);
        let (tiered, _) = table.per_layer(nl, &|r| expert(&r.name) && bf16_matrix(r), &|r| r.bytes * bf16 / 100);
        let others = crate::sizes::Table { rows: table.rows.iter().filter(|r| expert(&r.name) && !bf16_matrix(r)).cloned().collect() };
        let (released, _) = crate::dynamic::split::layer_bytes_table(&others, nl, 100, 100, &|_| true);
        let per: Vec<u64> = tiered.iter().zip(&released).map(|(a, b)| a + b).collect();
        let sh = crate::plan::Shape::from_model(m).map_err(cannot)?;
        let moe: Vec<bool> = per.iter().map(|&b| b > 0).collect();
        let (all, layers, experts) = (per.iter().sum::<u64>(), moe.iter().filter(|&&m| m).count(), sh.experts.max(1));
        let family = crate::dynamic::split::family_in(m).unwrap_or("");
        let ram = crate::dynamic::split::ram_bytes_table(table, nl, family, crate::dynamic::split::CODED_PCT, bf16);
        let place = all / (layers * experts).max(1) as u64;
        Ok(CacheOf { dense, base, ram, moe, place, experts, top_k: sh.topk, all, disk: false, across: 1, starts: stage_starts(m, nl) })
    }

    /// The plan `run` loads a model of a row whose experts come from a codes pack at the hummingbird level (`ExpertCache::Codes`: src/moe_cb.rs; the pack in
    /// `pack`, its `meta.json` and nothing else read), as `split::open_plan` loads it (src/dynamic/qwen4x.rs `stage`): on the GPU with the most free memory,
    /// the always-on matrices at the lossless tiered layout's share of their bf16 bytes, its context and users the flags' (`codes_plan_with`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn codes_plan(
        a: &Args,
        name: &str,
        table: &crate::sizes::Table,
        c: &ModelConfig,
        row: &'static Support,
        pack: &Path,
        devs: &[Target],
        host: Option<u64>,
    ) -> Done<LoadPlan> {
        let model_max = c.num("max_position_embeddings").unwrap_or(usize::MAX);
        let asked = a.has("context") || a.has("ctx");
        let ctx = a.either("context", "ctx", crate::support::default_context(row, model_max)).map_err(|e| Refusal::new(e.msg, "--context 8192"))?;
        crate::support::context_check(name, row, model_max, asked.then_some(ctx))?;
        let ask = CodesAsk {
            ctx,
            users: users_asked(a)?,
            served: served(a),
            chunk: DYN_CHUNK,
            layout: Layout::Tiered,
            hi: None,
            sides: crate::qwen4x::PackMode::of(pack, false).side(pack).into_iter().collect(),
            cut: false,
            split: None,
            places: None,
            host_memory: host_flag(a)?,
        };
        codes_plan_with(name, table, c, row, pack, devs, host, &ask)
    }

    /// What a codes pack's plan is asked (`codes_plan_with`): the context, the users (a server's: `served`), the prompt chunk, the always-on matrices' layout
    /// (penguin's lossless tiered one for `run`; Condor's server's level, four bits by default) and that of the attention blocks' own matrices and the LM head
    /// where it differs (`hi`: Condor's server's `--hi`), the sidecars over the pack's experts whose copies the GPU's places hold (its mode's, `--side`'s, or
    /// none: `--no-side`), whether the layers may be cut over the GPUs given in their order (Condor's server's `--gpus 0,1`; else the GPU with the most free
    /// memory holds them), the cuts asked (`--split`), the places a layer asked (`--places`), and the computer's memory `--host-memory` gives the experts.
    #[derive(Clone, Debug)]
    pub(crate) struct CodesAsk {
        pub ctx: usize,
        pub users: usize,
        pub served: bool,
        pub chunk: usize,
        pub layout: Layout,
        pub hi: Option<Layout>,
        pub sides: Vec<std::path::PathBuf>,
        pub cut: bool,
        pub split: Option<Vec<usize>>,
        pub places: Option<usize>,
        pub host_memory: Option<u64>,
    }

    /// The share of a matrix's bf16 bytes `layout` packs it in: a block layout's step (64 rows of 16 weights: its codes, its scales' 64 bytes and its base's 4, of
    /// the 2,048 bytes they are in bf16); penguin's own (`split::TIERED_PCT`, `TWELVE_PCT`).
    pub(crate) fn layout_pct(layout: Layout) -> u64 {
        use glyd_gpu::pack::{INT4_STEP_BYTES, INT5_STEP_BYTES, INT6_STEP_BYTES};
        let step = |b: usize| ((b + 68) * 100).div_ceil(2048) as u64;
        match layout {
            Layout::Int6 => step(INT6_STEP_BYTES),
            Layout::Int5 => step(INT5_STEP_BYTES),
            Layout::Int4 => step(INT4_STEP_BYTES),
            Layout::Twelve => crate::dynamic::split::TWELVE_PCT,
            _ => crate::dynamic::split::TIERED_PCT,
        }
    }

    /// A codes pack's experts as the expert cache takes them (`CacheOf`, the one rule's source like every other's): each layer's always-on matrices (the
    /// checkpoint's tensors but its routed experts and the tables its description reads on the host, `on_gpu`, at `pct` of their bf16 bytes; the vision tower
    /// as released with the rest; a table read by rows held there, as saved), every layer's experts the pack's (`cb`, opened as the loader opens it, with its
    /// mode's sidecar), a place the largest expert's
    /// of a layer (`CbPack::place_bytes`), a layer's experts the fewest any layer keeps, all of them in the pack's files mapped, those the computer's memory does
    /// not hold read as the router asks (the disk). `c`: its config, parsed once. `hi`: the share and the tensors of the matrices at a second layout (the
    /// attention blocks' own and the LM head: `Described::attention_and_head`), counted at it.
    pub(crate) fn codes_experts(
        on_gpu: &crate::sizes::Table,
        c: &ModelConfig,
        cb: &crate::cb_pack::CbPack,
        nl: usize,
        pct: u64,
        hi: Option<(u64, &std::collections::BTreeSet<String>)>,
    ) -> Done<CacheOf> {
        // (a table read by rows that a check holds on the GPU is held as saved: its bytes whole)
        let tables: std::collections::HashSet<String> = crate::dynamic::tables::ngram_names(c).into_iter().collect();
        let table = |n: &str| tables.contains(crate::formats::stands_for(n).as_ref());
        let at_hi = |n: &str| hi.is_some_and(|h| h.1.contains(n));
        let (mut dense, mut base) = crate::dynamic::split::layer_bytes_table(on_gpu, nl, 100, pct, &|n| !n.contains(".experts.") && !table(n) && !at_hi(n));
        if let Some((hp, _)) = hi {
            let (more, rest) = crate::dynamic::split::layer_bytes_table(on_gpu, nl, 100, hp, &at_hi);
            dense.iter_mut().zip(&more).for_each(|(d, s)| *d += s);
            base += rest;
        }
        let (saved, rest) = crate::dynamic::split::layer_bytes_table(on_gpu, nl, 100, 100, &table);
        dense.iter_mut().zip(&saved).for_each(|(d, s)| *d += s);
        base += rest;
        let vision: u64 = on_gpu.rows.iter().filter(|r| matches!(crate::sizes::classify(&r.name, nl), crate::sizes::Part::Vision)).map(|r| r.bytes).sum();
        let sh = crate::plan::Shape::from_model(c).map_err(cannot)?;
        Ok(CacheOf {
            dense,
            base: base + vision,
            ram: Vec::new(),
            moe: (0..nl).map(|l| cb.has(l)).collect(),
            place: (0..nl).map(|l| cb.place_bytes(l)).max().unwrap_or(1) as u64,
            experts: (0..nl).filter_map(|l| cb.kept(l).map(<[usize]>::len)).min().unwrap_or(1),
            top_k: sh.topk,
            all: (0..nl).map(|l| cb.layer_bytes(l) as u64).sum(),
            disk: true,
            across: 1,
            starts: stage_starts(c, nl),
        })
    }

    /// A codes pack's plan (`ExpertCache::Codes`), by the one expert-cache rule (`expert_cache_with` of `codes_experts`): the users' keys, values and states at
    /// the context first (`rows::SeqRoom`, a server's users' whole contexts where one GPU holds the model; a run's one sequence, and a model cut over GPUs, one
    /// sequence each part its share of it: `rows::layer_rooms`), the executor's scratch (`CACHE_SCRATCH`) a part and each GPU's floor (`CONTEXT_FLOOR`) next,
    /// the places what is left: at least one a layer (the asked ones where `ask.places` says), at most all a layer keeps. On the GPU with the most free memory,
    /// or the layers cut over the GPUs `devs` in order where `ask.cut` (a GPU named twice shares its memory between its parts; the cuts `ask.split` where
    /// given, each part its places by the same rule). Every expert of the layers is in the computer's memory as the pack's files mapped: `host` free less
    /// `HOST_MARGIN` holds what it holds (`--host-memory` asks less), the files the rest, read as the router asks. Refused in words where the GPUs leave no
    /// place a layer, the computer's memory holds less than a token's experts a layer, or the pack lacks a layer of the model.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn codes_plan_with(
        name: &str,
        table: &crate::sizes::Table,
        c: &ModelConfig,
        row: &'static Support,
        pack: &Path,
        devs: &[Target],
        host: Option<u64>,
        k: &CodesAsk,
    ) -> Done<LoadPlan> {
        let (ctx, chunk) = (k.ctx, k.chunk);
        let (nl, hidden, inter) = (c.num("num_hidden_layers").unwrap_or(0), c.num("hidden_size").unwrap_or(0), c.num("moe_intermediate_size").unwrap_or(0));
        let mut cb = crate::cb_pack::CbPack::open_dims(pack, hidden, inter).map_err(cannot)?;
        for side in &k.sides {
            cb = cb.with_side(side).map_err(cannot)?;
        }
        if let Some(l) = (0..nl).find(|&l| !cb.has(l)) {
            return Err(Refusal::unsupported(
                format!("{name}: its codes pack {} has no experts of layer {l} of the model's {nl}", pack.display()),
                "Give the pack of this model's experts: the release's (condor/tools/download.sh), or one made for it",
            ));
        }
        // (the always-on matrices as the loader holds them; the tables its description reads on the host and the routed experts none of the GPU's)
        let (on_gpu, tables) = crate::dynamic::tables::off_gpu(table, c);
        // (the attention blocks' own matrices and the LM head at `k.hi` where it is another layout: the loader's `Placing::hi`)
        let hi_names = match k.hi.filter(|&h| h != k.layout) {
            Some(_) => Some(crate::runtime::describe::describe_config(c).map_err(cannot)?.attention_and_head()),
            None => None,
        };
        let hi = k.hi.zip(hi_names.as_ref()).map(|(h, n)| (layout_pct(h), n));
        let src = codes_experts(&on_gpu, c, &cb, nl, layout_pct(k.layout), hi)?;
        let always = src.dense.iter().sum::<u64>() + src.base;
        // (the sequences' keys, values and states, the description's as the loader lowers it, by the executor's rule: a server's users each their whole context
        // as its pools hold them where one GPU holds the model; one sequence, each layer its share, where it is cut or run)
        let low = crate::dynamic::split::lowered_of(c, ctx + chunk + 16).map_err(cannot)?;
        let one: Vec<u64> = crate::dynamic::layer_rooms(&low, ctx, chunk);
        let (floor, work) = (crate::plan::CONTEXT_FLOOR, CACHE_SCRATCH);
        let used: Vec<&Target> = if k.cut { devs.iter().collect() } else { devs.iter().max_by_key(|t| t.free).into_iter().collect() };
        if used.is_empty() {
            return Err(Refusal::unsupported(format!("{name}: no GPU to plan it on"), ""));
        }
        // (each part's room: its GPU's free memory less the floor, shared by the parts named on it, less a part's scratch)
        let room: Vec<(i32, u64)> = used
            .iter()
            .map(|t| {
                let on = used.iter().filter(|x| x.ordinal == t.ordinal).count().max(1) as u64;
                (t.ordinal, (t.free.saturating_sub(floor) / on).saturating_sub(work))
            })
            .collect();
        let ask = CacheAsk { least: Some(k.places.unwrap_or(1)), ram_least: None, host_memory: k.host_memory };
        let first = |kv: u64| -> Vec<u64> { std::iter::once(kv).chain(std::iter::repeat_n(0, nl.saturating_sub(1))).collect() };
        let seq = crate::dynamic::SeqRoom::of(&low, ctx, chunk);
        let served_kv = if k.served { seq.plan(k.users, u64::MAX).unwrap_or((k.users, 0)) } else { (1, one.iter().sum()) };
        // (the most users one GPU holds beside a place a layer, where the users asked do not fit: the way out a refusal names)
        let fewer = || {
            (1..k.users).rev().find(|&u| {
                let kv = seq.plan(u, u64::MAX).map_or(0, |p| p.1);
                k.served && !k.cut && expert_cache_with(&src, ask, &room, &first(kv), host).is_ok()
            })
        };
        let planned = match &k.split {
            Some(cuts) => codes_stages(&src, cuts, &room, &one, ask, host).map(|c| ((1, one.iter().sum::<u64>()), c)),
            None => match expert_cache_with(&src, ask, &room, &first(served_kv.1), host) {
                Ok(c) if c.cuts.is_empty() => Ok((served_kv, c)),
                // (more than one GPU's layers: each part holds its share of one sequence)
                Ok(_) => expert_cache_with(&src, ask, &room, &one, host).map(|c| ((1, one.iter().sum()), c)),
                Err(e) => Err(e),
            },
        };
        let ((users, kv), mut cache) = planned.map_err(|why| {
            let free = used.iter().map(|t| t.free).sum::<u64>();
            let gpus = if used.len() == 1 { used[0].name.clone() } else { format!("GPUs {:?}", used.iter().map(|t| t.ordinal).collect::<Vec<_>>()) };
            match why {
                CacheShort::Held { ram, least } => Refusal::unsupported(
                    format!(
                        "{name} at the hummingbird level reads its experts from its pack's files as the router asks, and keeps at least {} of them in the computer's memory (a token's in every layer), where {} are given",
                        gib(least),
                        gib(ram)
                    ),
                    "Give its experts more of the computer's memory (--host-memory GIB), or free some",
                ),
                CacheShort::Asked { ram, free: h } => {
                    Refusal::new(format!("--host-memory {}: the computer has {} free and keeps {} of them for the system", gib(ram), gib(h), gib(HOST_MARGIN)), "")
                }
                CacheShort::Memory { .. } | CacheShort::Gpu => Refusal::unsupported(
                    format!(
                        "{name} at the hummingbird level: its weights besides the experts take {}, its sequences {} at {ctx} tokens, the engine's work {}, and the {gpus} {} {} free: no room for {} a layer for its experts ({} each)",
                        gib(always),
                        gib(if k.split.is_some() { one.iter().sum() } else { served_kv.1 }),
                        gib(work * used.len() as u64),
                        if used.len() == 1 { "has" } else { "have" },
                        gib(free),
                        match k.places {
                            Some(p) if p > 1 => format!("{p} places"),
                            _ => "a place".to_string(),
                        },
                        gib(src.place)
                    ),
                    match fewer() {
                        Some(u) => format!("Ask for fewer users (--users {u}) or less context (--context {}), or use a GPU with more memory free", (ctx / 2).max(1024)),
                        None => "Ask for less context (--context 4096), or use a GPU with more memory free".to_string(),
                    },
                ),
            }
        })?;
        // (the places asked: no more than those)
        if let Some(p) = k.places {
            cache.places.iter_mut().for_each(|x| *x = (*x).min(p));
        }
        let bounds: Vec<usize> = std::iter::once(0).chain(cache.cuts.iter().copied()).chain(std::iter::once(nl)).collect();
        let gpu_bytes = always
            + bounds.windows(2).zip(&cache.places).map(|(w, &p)| src.moe[w[0]..w[1]].iter().filter(|&&m| m).count() as u64 * p as u64 * src.place).sum::<u64>();
        let gpus: Vec<&Target> = cache
            .devices
            .iter()
            .enumerate()
            .filter(|(i, d)| !cache.devices[..*i].contains(d))
            .filter_map(|(_, d)| used.iter().find(|t| t.ordinal == *d).copied())
            .collect();
        let host_per_stage = stage_shares(&(0..nl).map(|l| cb.layer_bytes(l) as u64).collect::<Vec<_>>(), &cache.cuts, src.all);
        Ok(LoadPlan {
            kind: Kind::Dynamic,
            family: row.name,
            level: "hummingbird",
            layout: k.layout,
            context: ctx,
            users,
            chunk,
            weights: always + src.all + tables,
            bf16: 0,
            kv,
            kv12: false,
            work: work * cache.devices.len() as u64,
            draft: 0,
            reserve: floor * gpus.len() as u64,
            free: gpus.iter().map(|t| t.free).sum(),
            gpu: gpus.iter().map(|t| t.name.as_str()).collect::<Vec<_>>().join(", "),
            experts: Some(CachePlan {
                layers: nl,
                experts: src.experts,
                slots: cache.places.iter().min().copied().unwrap_or(0),
                expert_bytes: src.all / (nl * src.experts).max(1) as u64,
                in_place: true,
                ram: cache.ram,
            }),
            spread: Some(crate::plan::Spread {
                cuts: cache.cuts,
                devices: cache.devices,
                across: false,
                coded: false,
                twelve: false,
                places: cache.places,
                experts_per_layer: src.experts,
                host_bytes: src.all,
                host_per_stage,
                gpu_bytes,
                dense: 0,
                dense_bytes: 0,
                tables,
            }),
        })
    }

    /// The expert cache of a codes pack's model cut before each of `cuts` (asked: `--split`), part k on the GPU of `room[k]` (one more than the cuts), each part
    /// planned by the one rule on its own GPU (`expert_cache_with` of its layers: their weights and their share of one sequence `one`, the rest's half with the
    /// first part and half with the last, as the rule holds it), its places what its GPU leaves.
    fn codes_stages(src: &CacheOf, cuts: &[usize], room: &[(i32, u64)], one: &[u64], ask: CacheAsk, host: Option<u64>) -> Result<Cached, CacheShort> {
        let nl = src.dense.len();
        let bounds: Vec<usize> = std::iter::once(0).chain(cuts.iter().copied()).chain(std::iter::once(nl)).collect();
        let parts = bounds.len() - 1;
        let mut out =
            Cached { cuts: cuts.to_vec(), devices: Vec::new(), places: Vec::new(), on_gpus: 0, all: src.all, experts: src.experts, ram: 0, dense: (0, 0) };
        for (i, w) in bounds.windows(2).enumerate() {
            let r = *room.get(i).ok_or(CacheShort::Gpu)?;
            let base = if i == 0 { src.base / 2 } else { 0 } + if i + 1 == parts { src.base - src.base / 2 } else { 0 };
            let part =
                CacheOf { dense: src.dense[w[0]..w[1]].to_vec(), base, ram: Vec::new(), moe: src.moe[w[0]..w[1]].to_vec(), starts: Vec::new(), ..src.clone() };
            let c = expert_cache_with(&part, ask, &[r], &one[w[0]..w[1]], host)?;
            if !c.cuts.is_empty() {
                return Err(CacheShort::Gpu);
            }
            out.devices.push(r.0);
            out.places.push(c.places.first().copied().unwrap_or(0));
            out.on_gpus += c.on_gpus;
            out.ram = c.ram;
        }
        Ok(out)
    }

    /// The codes pack of the model `model` at the hummingbird level (`ExpertCache::Codes`): `--pack DIR` where given, else the one beside its weights (`codes_beside`),
    /// else, for a model of the Hub, the `meta.json` of the pre-packed download's pack (glyd/NAME-hummingbird: `--pack precise`, the default, `best` or `fast`,
    /// each its newest folder of the download, `precise-v2` before `precise`) kept in the engine's cache, nothing else of it read; None where none can be had
    /// (offline, or the Hub has no such pack).
    pub(crate) fn codes_pack(a: &Args, model: &str) -> Option<PathBuf> {
        let given = a.flag("pack").filter(|p| !p.is_empty());
        if let Some(p) = given.filter(|p| Path::new(p).join("meta.json").exists()) {
            return Some(PathBuf::from(p));
        }
        // (a model here: the pack beside its weights alone, as `run` loads it)
        if let Some(d) = local_dir(model) {
            return codes_beside(&d);
        }
        if crate::hub::offline() || model.matches('/').count() != 1 {
            return None;
        }
        let repo = format!("glyd/{}-hummingbird", short_name(model));
        let mode = given.unwrap_or("precise");
        let hub = crate::hub::Hub::from_env();
        for folder in [format!("{mode}-v2"), mode.to_string()] {
            let at = store::cache_root().join("codes").join(&repo).join(&folder);
            if at.join("meta.json").exists() {
                return Some(at);
            }
            if let Ok(text) = hub.text(&repo, "main", &format!("{folder}/meta.json")) {
                std::fs::create_dir_all(&at).ok()?;
                std::fs::write(at.join("meta.json"), text).ok()?;
                return Some(at);
            }
        }
        None
    }

    /// The directory of MODEL where it is here: itself where it holds a config.json, else the Hub cache's copy.
    fn local_dir(model: &str) -> Option<PathBuf> {
        if Path::new(model).join("config.json").exists() {
            Some(PathBuf::from(model))
        } else {
            Source::find(model).ok().map(|s| s.dir)
        }
    }

    /// The refusal of the model `model` (short name `name`) of the row `family` whose experts come from a codes pack at the hummingbird level
    /// (`ExpertCache::Codes`) where none is beside its weights or named (`codes_of`): `run`'s, and `fit`'s of a model here.
    pub fn no_codes_pack(name: &str, family: &str, model: &str) -> Refusal {
        Refusal::unsupported(
            format!(
                "{name} ({family}) at the hummingbird level runs from a codes pack of its experts beside its weights (DIR/pack) or named by --pack, and {model} has none here"
            ),
            "Name its pack (the release's: condor/tools/download.sh): run DIR:hummingbird --pack PACK, or put it in DIR/pack, or serve it with qwenserve CKPT PACK",
        )
    }

    /// The codes pack beside the weights in `dir` (`dir/pack`, its `meta.json` the format's, glyd-x/1), where it has one.
    pub fn codes_beside(dir: &Path) -> Option<PathBuf> {
        codes_at(&dir.join("pack"))
    }

    /// `p` where it is a codes pack (its `meta.json` the format's, glyd-x/1).
    fn codes_at(p: &Path) -> Option<PathBuf> {
        let text = std::fs::read_to_string(p.join("meta.json")).ok()?;
        glyd_gpu::json::parse(&text).ok()?.get("format").and_then(glyd_gpu::json::Value::as_str).filter(|f| *f == "glyd-x/1").map(|_| p.to_path_buf())
    }

    /// The codes pack `--pack DIR` names to `run`, `serve` and `prepare` (`name_pack`), for a model whose directory holds none beside its weights: the
    /// release's download keeps its packs beside `model/` (condor/tools/download.sh: `DIR/model`, `DIR/precise-v2`).
    static PACK: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

    /// `--pack DIR` of `a` as this process's codes pack (`codes_of`), where it names one.
    pub fn name_pack(a: &Args) {
        if let Some(p) = a.flag("pack").filter(|p| !p.is_empty()) {
            let _ = PACK.set(PathBuf::from(p));
        }
    }

    /// The codes pack the experts of the model in `dir` are read from at the hummingbird level: the one `--pack` named (`name_pack`), else the one beside its
    /// weights (`codes_beside`).
    pub fn codes_of(dir: &Path) -> Option<PathBuf> {
        PACK.get().and_then(|p| codes_at(p)).or_else(|| codes_beside(dir))
    }

    /// Each GPU's room for weights on the dynamic route: its free memory less the reserve and the route's work (the KV, the scratch and the graphs together;
    // ponytail: that work is a fixed margin, 3 GiB and 5% with the reserve; the description's own numbers where it proves too little or too much)
    pub(crate) fn rooms(gpus: &[Gpu]) -> Vec<(i32, u64)> {
        gpus.iter().map(|g| (g.ordinal, crate::plan::dynamic_room_of(g.free))).collect()
    }

    /// The table's own name of the family named `family` (`dynamic_family`'s words are these); the empty name for one the table has no row for, which
    /// `size_dynamic` refuses.
    pub(crate) fn static_family(family: &str) -> &'static str {
        crate::support::row_named(family).map_or("", |r| r.name)
    }

    /// Whether the GPUs of `devs` are HBM ones (A100, H100, B200: compute capability 8.0, 9.0, 10.0), whose FP8 matrices take layout H (src/f8.rs `layout_h`).
    pub(crate) fn hbm_of(devs: &[Target]) -> bool {
        devs.iter().any(|t| t.cc.1 == 0 && (8..=10).contains(&t.cc.0))
    }

    /// The plan `run` loads a model of the dynamic route (`dynamic_family`) from `dir` at penguin on the GPUs `devs`: the layers cut over them as their room holds
    /// them (`--split CUTS --devices GPUS` chooses), or every layer across them (src/dynamic/tp.rs; `reach`: whether they reach each other directly), the FP8
    /// matrices coded where only that fits, Kimi-K3's experts kept in the computer's memory (`host` bytes) with places on the GPUs where they do not all fit; or
    /// the refusal `run` gives. What `load_dynamic` loads, `check_dynamic` checks and `fit` prints.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn size_dynamic(
        a: &Args,
        name: &str,
        table: &crate::sizes::Table,
        dir: Option<&Path>,
        c: &ModelConfig,
        family: &'static str,
        devs: &[Target],
        host: Option<u64>,
        reach: &dyn Fn(&[i32]) -> bool,
    ) -> Done<LoadPlan> {
        // (a family the table has no row for is refused here, never sized as another's)
        let row = crate::support::row_named(family).ok_or_else(|| {
            Refusal::unsupported(format!("{name}: the engine has no family named {family:?} to size it as"), "Run a model of a family this engine runs")
        })?;
        size_dynamic_row(a, name, table, dir, c, row, devs, host, reach)
    }

    /// The prompt chunk the dynamic route's engines take (`LoadPlan::chunk`).
    pub const DYN_CHUNK: usize = 256;

    /// `size_dynamic` for the family whose row is `row` (`c`: its config, parsed once): every choice that differs between families is one of the row's fields
    /// (`Support::route`). The context is the one asked, else the row's (`Support::default_ctx`), else the most that fits of LOSSY_CTX and its halves down to
    /// LOSSY_MIN_CTX; the sequences the plan holds are `users_asked`'s.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn size_dynamic_row(
        a: &Args,
        name: &str,
        table: &crate::sizes::Table,
        dir: Option<&Path>,
        c: &ModelConfig,
        row: &'static Support,
        devs: &[Target],
        host: Option<u64>,
        reach: &dyn Fn(&[i32]) -> bool,
    ) -> Done<LoadPlan> {
        let model_max = c.num("max_position_embeddings").unwrap_or(usize::MAX);
        // (--ctx as the one-GPU paths take it, too: the Qwen3.5 layout's lossy levels and its penguin read the same flags)
        let asked = a.has("context") || a.has("ctx");
        let ctx = a.either("context", "ctx", crate::support::default_context(row, model_max)).map_err(|e| Refusal::new(e.msg, "--context 8192"))?;
        crate::support::context_check(name, row, model_max, asked.then_some(ctx))?;
        let tries: Vec<usize> = if asked || row.default_ctx.is_some() {
            vec![ctx]
        } else {
            std::iter::successors(Some(ctx), |&c| (c / 2 >= LOSSY_MIN_CTX).then_some(c / 2)).collect()
        };
        let users = users_asked(a)?;
        let mut first = None;
        for ctx in tries {
            match size_dynamic_at(a, name, table, dir, c, row, devs, host, reach, ctx, users) {
                Ok(p) => return Ok(p),
                Err(r) => {
                    first.get_or_insert(r);
                }
            }
        }
        Err(first.expect("a context was tried"))
    }

    /// `size_dynamic_row` at `ctx` tokens of context for `users` sequences (`c`: the config parsed once). The sequences' keys, values and states are its description's,
    /// lowered as its loader lowers it (`split::lowered_of`), by the executor's own rule (`rows::SeqRoom`, `rows::layer_rooms`): each layer's share of one
    /// sequence's, and a server's least for its users beside, are held with its weights where the layers are cut; on one GPU the whole model is one engine, whose
    /// server's users share pools of what the GPU leaves them (as many as their states leave one sequence's context room); with its experts in the expert cache,
    /// every user's whole context first, the engine's work next and the cache's places what is left (`expert_cache`), the users held or refused in words; a run's
    /// one sequence, and stages and layers across GPUs, hold one sequence, each part its own of it.
    ///
    /// The keys and values bf16, or in the executor's 12-bit lossless pages where GLYD_KV=lossless asks for them for a server (`twelve_pages`) and its description
    /// has rings the pages take in pools (`SeqRoom::pages`, by its attention kinds: heads of 128, 256 or 512, latents; never by its family): both are lossless, so
    /// the pages are taken by `one_gpu_pages`' rule, only where they reserve no more than bf16 for the plan's users (`SeqRoom::most_bytes`: every one holding the
    /// context, with what the pages spend once), or where bf16 does not fit and they do; a check of the pages against their own plan takes them wherever they fit
    /// (`twelve_always`).
    #[allow(clippy::too_many_arguments)]
    fn size_dynamic_at(
        a: &Args,
        name: &str,
        table: &crate::sizes::Table,
        dir: Option<&Path>,
        c: &ModelConfig,
        row: &'static Support,
        devs: &[Target],
        host: Option<u64>,
        reach: &dyn Fn(&[i32]) -> bool,
        ctx: usize,
        users: usize,
    ) -> Done<LoadPlan> {
        // (the sequences' keys, values and states: the description's lowered as the loader lowers it, at the context, for the users. A description the executor does
        // not lower (a shape its kernels do not take, which its loader refuses) is sized by the config's estimate of one sequence, `Shape`, a layer its share)
        let low = crate::dynamic::split::lowered_of(c, ctx + DYN_CHUNK + 16).ok();
        let bf16 = low.as_ref().map(|l| crate::dynamic::SeqRoom::of(l, ctx, DYN_CHUNK));
        let held = |seq: Option<crate::dynamic::SeqRoom>, pages: bool| {
            size_dynamic_held(a, name, table, dir, c, row, devs, host, reach, ctx, users, low.as_ref(), seq, pages)
        };
        let Some(t) = low.as_ref().filter(|_| twelve_pages(a)).and_then(|l| crate::dynamic::SeqRoom::pages(l, ctx, DYN_CHUNK)) else {
            return held(bf16, false);
        };
        let fewer = bf16.as_ref().is_some_and(|b| t.most_bytes(users) <= b.most_bytes(users));
        match (held(bf16, false), held(Some(t), true)) {
            (Ok(b), Ok(p)) => Ok(if twelve_always() || fewer { p } else { b }),
            (Err(_), Ok(p)) => Ok(p),
            (b, _) => b,
        }
    }

    /// `size_dynamic_at` with the sequences' rule `seq` of the lowered description `low` (None: the config's estimate), its keys and values in 12-bit pages where
    /// `pages` (a server's pools with the whole model on one GPU, else refused: elsewhere each part holds one sequence in rings of its own, in bf16).
    #[allow(clippy::too_many_arguments)]
    fn size_dynamic_held(
        a: &Args,
        name: &str,
        table: &crate::sizes::Table,
        dir: Option<&Path>,
        c: &ModelConfig,
        row: &'static Support,
        devs: &[Target],
        host: Option<u64>,
        reach: &dyn Fn(&[i32]) -> bool,
        ctx: usize,
        users: usize,
        low: Option<&crate::dynamic::ModelDesc>,
        seq: Option<crate::dynamic::SeqRoom>,
        pages: bool,
    ) -> Done<LoadPlan> {
        let (family, route) = (row.name, row.route);
        // (the tables its description reads on the host take none of the GPUs' memory: their bytes are the computer's, the checkpoint's files mapped)
        let (table, tables) = crate::dynamic::tables::off_gpu(table, c);
        let table = &table;
        let nl = c.num("num_hidden_layers").unwrap_or(0);
        let room: Vec<(i32, u64)> = devs.iter().map(|t| (t.ordinal, crate::plan::dynamic_room_of(t.free))).collect();
        let hbm = hbm_of(devs);
        // (a hummingbird directory, src/hb_pack.rs: its routed experts are files of their own, which no table of tensors lists)
        let hb_dir = dir.filter(|d| matches!(crate::hb_pack::HbDir::open(d), Ok(Some(_))));
        let hb = hb_dir.is_some();
        // (a family whose layers are not cut over GPUs at this level (`Support::split_levels`) loads on one GPU: the one with the most room)
        let level = if hb { "hummingbird" } else { "penguin" };
        let one_gpu = !row.split_levels.contains(&level);
        let room: Vec<(i32, u64)> = if one_gpu { room.iter().copied().max_by_key(|r| r.1).into_iter().collect() } else { room };
        let hb_experts = match hb_dir {
            Some(d) => crate::dynamic::split::hb_layer_bytes(d, nl).map_err(cannot)?,
            None => Vec::new(),
        };
        // (one sequence's keys, values and states, each layer its share: the description's (`layer_rooms`), else the config's estimate)
        let rooms = match low {
            Some(low) => crate::dynamic::layer_rooms(low, ctx, DYN_CHUNK),
            None => {
                let sh = crate::plan::Shape::from_model(c).map_err(cannot)?;
                vec![((sh.kv_full * ctx as f64 + sh.kv_sliding * ctx.min(sh.window.max(1)) as f64 + sh.state) / nl.max(1) as f64) as u64; nl]
            }
        };
        let one_seq = rooms.iter().sum::<u64>();
        // (the loaders that take penguin's 12-bit layout for their bf16 matrices, which an HBM GPU reads faster where it fits: the row says (`Route::bf16_twelve`);
        // a release with no FP8 matrices takes it on every GPU but Ada's where it fits, as the one-GPU paths choose it (`best_layout`) and as their checks
        // of several sequences at once ran)
        let takes_twelve = route.bf16_twelve;
        let no_f8 = !table.rows.iter().any(|r| r.dtype == "F8_E4M3");
        let twelve_faster = hbm || (no_f8 && !devs.is_empty() && devs.iter().all(|t| !crate::weights::smallest_is_fastest(t.cc)));
        // (a stage begins at a layer that takes nothing from the one before it: `stage_starts`)
        let starts = stage_starts(c, nl);
        // (each layer's bytes with its share of the sequences': one sequence's, each layer its own (`layer_rooms`), and `more` beside them with the first layer)
        let with = |more: u64| -> Vec<u64> {
            let mut kv: Vec<u64> = if rooms.len() == nl {
                rooms.clone()
            } else {
                (0..nl).map(|l| one_seq / nl.max(1) as u64 + (l == 0) as u64 * (one_seq % nl.max(1) as u64)).collect()
            };
            if let Some(k) = kv.first_mut() {
                *k += more;
            }
            kv
        };
        let place = |kv: &[u64]| {
            let extra: Vec<u64> = (0..nl).map(|l| kv[l] + hb_experts.get(l).copied().unwrap_or(0)).collect();
            crate::dynamic::split::fit_table_with(table, nl, &room, twelve_faster, takes_twelve, &extra, &starts)
        };
        // (a server's users are held where one GPU holds the whole model beside their least (`SeqRoom::least_bytes`: one sequence's context and every user's states
        // beside); where it does not, the layers cut over GPUs hold one sequence, as a run's)
        let serving = served(a) && seq.is_some();
        let mut served = serving;
        let mut kv_layers = with(seq.as_ref().filter(|_| serving).map_or(0, |s| s.least_bytes(users).saturating_sub(one_seq)));
        let mut fitted = place(&kv_layers);
        if served && !matches!(&fitted.1, Some((c, d)) if c.is_empty() && d.len() == 1) {
            served = false;
            kv_layers = with(0);
            fitted = place(&kv_layers);
        }
        let ((per, base), cut, coded, mut twelve) = fitted;
        let chosen = a.flag("split").is_some() || a.flag("devices").is_some();
        // (every layer across the GPUs, src/dynamic/tp.rs: by default on datacenter GPUs that reach each other directly, where no split is asked for; a
        // development build's GLYD_TP=1 or 0 says so itself, GLYD_TP_GPUS which GPUs)
        let tp_room: Vec<(i32, u64)> = match crate::tune!("GLYD_TP_GPUS") {
            Some(v) => room.iter().copied().filter(|r| v.split(',').any(|x| x.trim().parse::<i32>().ok() == Some(r.0))).collect(),
            None => room.clone(),
        };
        let tp_devs: Vec<i32> = tp_room.iter().map(|r| r.0).collect();
        let across = route.across != Across::No
            && tp_room.len() > 1
            && !chosen
            && match crate::tune!("GLYD_TP").as_deref() {
                Some("1") => true,
                Some(_) => false,
                None => route.across == Across::Default && hbm && reach(&tp_devs),
            };
        // the expert cache where the GPUs do not hold the model whole, its experts read by one rule whatever their source (`expert_cache`): a hummingbird directory's
        // for any family, its layers cut over the GPUs, never every layer across them; a release's as released (src/tiering.rs, `ExpertCache::Release`: FP4 experts,
        // FP8 ones, lossless bf16 ones; a release of such a row with no experts, the Qwen3.5 layout's dense one, has none to keep there); a family's whose loader packs
        // its experts itself (`ExpertCache::Packed`), every layer across the GPUs where they take it so (`packed_experts` of each GPU's share, `CacheOf::across`)
        let has_experts = table.rows.iter().any(|r| r.name.contains(".experts."));
        let release_cached = route.expert_cache == ExpertCache::Release && has_experts && !hb && cut.is_none() && !chosen;
        let packed = match route.expert_cache {
            ExpertCache::Packed(tags) if !hb && cut.is_none() && !chosen => Some(tags),
            _ => None,
        };
        let across = across && !hb && !release_cached;
        let source = if let Some(dir) = hb_dir.filter(|_| cut.is_none() && !chosen) {
            Some(hb_dir_experts(dir, c, nl, hbm)?)
        } else if release_cached {
            Some(release_experts(table, c, nl, hbm)?)
        } else if let Some(tags) = packed.filter(|_| !across) {
            Some(packed_experts(table, c, nl, hbm, 1, tags)?)
        } else {
            None
        };
        // (a server's users on one GPU beside the expert cache: every user's whole context first (`SeqRoom::plan` of them all, as the pools take them), the engine's
        // work next (`room` keeps it apart), the places what the GPU has left: --users is held, or refused in words with the numbers. Where the places fill more than
        // one GPU, the stages hold one sequence, as a run's)
        let hb = row.levels.contains(&"hummingbird");
        let mut cache = None;
        if let (Some(src), Some(s)) = (&source, seq.as_ref().filter(|_| serving)) {
            let users_kv = |k: usize| with(s.plan(k, u64::MAX).map_or(0, |p| p.1).saturating_sub(one_seq));
            match expert_cache(src, &room, name, &users_kv(users), host, hb) {
                Ok(c) if c.cuts.is_empty() && c.devices.len() == 1 => (served, kv_layers, cache) = (true, users_kv(users), Some(c)),
                Err(_) if room.len() == 1 => {
                    // (one sequence's refusal first, where even one does not fit: the computer's memory, or the GPU's room beside the weights)
                    expert_cache(src, &room, name, &kv_layers, host, hb)?;
                    let most = (1..users).rev().find(|&k| expert_cache(src, &room, name, &users_kv(k), host, hb).is_ok());
                    let gpu = devs.iter().find(|t| t.ordinal == room[0].0);
                    let free = gpu.map_or(0, |t| t.free);
                    let (weights, all) = (src.dense.iter().sum::<u64>() + src.base, s.plan(users, u64::MAX).map_or(0, |p| p.1));
                    return Err(Refusal::unsupported(
                        format!(
                            "{name} at {level}, its experts in the computer's memory, takes about {} of GPU memory for {users} users at {ctx} tokens of context: {} of weights besides the experts, {} of keys, values and states for the users, {} for a token's experts in every layer and {} for the engine's work, and the {} has {} free",
                            gib(weights + all + src.token_bytes() + free.saturating_sub(room[0].1)),
                            gib(weights),
                            gib(all),
                            gib(src.token_bytes()),
                            gib(free.saturating_sub(room[0].1)),
                            gpu.map_or("GPU", |t| t.name.as_str()),
                            gib(free)
                        ),
                        match most {
                            Some(k) => format!("Ask for fewer users (--users {k}) or less context (--context {}), or use a GPU with more memory free", (ctx / 2).max(LOSSY_MIN_CTX)),
                            None => "Ask for less context (--context 4096), or use a GPU with more memory free".to_string(),
                        },
                    ));
                }
                _ => {}
            }
        }
        let cache = match (cache, &source) {
            (Some(c), _) => Some(c),
            (None, Some(src)) => Some(expert_cache(src, &room, name, &kv_layers, host, hb)?),
            // (every layer across the GPUs: each its slice of the big matrices, its sequence's bytes whole, and a slice of each place)
            (None, None) => match packed.filter(|_| across) {
                Some(tags) => Some(expert_cache(&packed_experts(table, c, nl, hbm, tp_room.len(), tags)?, &tp_room, name, &with(0), host, hb)?),
                None => None,
            },
        };
        let held = kv_layers.iter().sum::<u64>();
        // (the weights alone: the sequences' bytes held with them where the layers were cut are counted apart)
        let mut total = (per.iter().sum::<u64>() + base).saturating_sub(held);
        // (the last layers' dense matrices in the computer's memory where a token's experts do not fit beside them on one GPU: how many, their bytes)
        let dense_ram = cache.as_ref().map_or((0, 0), |k| k.dense);
        if let Some(k) = &cache {
            // (the places' plans hold the sequences' bytes with the layers' weights too: every GPU its own across them)
            total = k.on_gpus.saturating_sub(if across { tp_room.len() as u64 * one_seq } else { held });
            twelve = hbm;
        }
        // (what the GPUs would take of the model and of its sequences at the least, for the refusals)
        let need = total + held;
        let (cuts, devices) = if let Some(k) = &cache {
            (k.cuts.clone(), k.devices.clone())
        } else if across {
            // (each GPU its slice of the big matrices and the rest whole: about its share of the bytes and the embedding's and head's)
            // ponytail: the share is an estimate; a GPU that runs out of memory says so at the load
            (Vec::new(), tp_devs.clone())
        } else if chosen {
            let (cuts, devices) = crate::dynamic::split::split_args(a)
                .map_err(|e| Refusal::new(e.msg, "--split 31 --devices 0,1: the layers before 31 on GPU 0, the rest on GPU 1"))?;
            if one_gpu && (!cuts.is_empty() || devices.len() > 1) {
                return Err(Refusal::unsupported(
                    format!("{name} ({family}) loads on one GPU at {level}: its layers are not cut over GPUs"),
                    "Leave out --split; --devices 1 picks the GPU",
                ));
            }
            // (a stage begins only where `stage_starts` says: the loader refuses a layer that takes the picks of another stage's, or reads a compressed part of
            // another stage's layer, after the plan is printed)
            if let Some(l) = cuts.iter().copied().filter(|&l| !starts.get(l).copied().unwrap_or(true)).min() {
                let firsts = (1..nl).filter(|&l| starts.get(l).copied().unwrap_or(true)).take(6).map(|l| l.to_string()).collect::<Vec<_>>().join(", ");
                return Err(if crate::dynamic::lower_comp::shares_parts(c) {
                    Refusal::unsupported(
                        crate::dynamic::lower_comp::split_apart_parts(l),
                        format!("A stage may begin at a layer no layer after it reads a compressed part before: {firsts}"),
                    )
                } else {
                    Refusal::unsupported(
                        crate::dynamic::lower::split_apart_picks(l),
                        format!("A stage may begin at a layer whose indexer is its own: {firsts}"),
                    )
                });
            }
            (cuts, devices)
        } else if one_gpu {
            cut.ok_or_else(|| {
                Refusal::unsupported(
                    format!(
                        "{name} needs about {} of GPU memory at {level} on one GPU (its layers are not cut over GPUs), and the GPU here with the most has {} free",
                        gib(need),
                        gib(devs.iter().map(|t| t.free).max().unwrap_or(0))
                    ),
                    "A GPU with that much memory free runs it (nvidia-smi lists what uses them)",
                )
            })?
        } else {
            cut.ok_or_else(|| {
                Refusal::unsupported(
                    format!(
                        "{name} needs about {} of GPU memory at penguin, and the {} GPU{} here have {} free",
                        gib(need),
                        devs.len(),
                        if devs.len() == 1 { "" } else { "s" },
                        gib(devs.iter().map(|t| t.free).sum::<u64>())
                    ),
                    "A machine whose GPUs have that much memory free runs it (nvidia-smi lists what uses them)",
                )
            })?
        };
        // the sequences: on one GPU the whole model is one engine (`split::Run::One`), a server's users in pools of what the GPU leaves them (`SeqRoom::plan`); a
        // run's one sequence, and stages and layers across GPUs, one sequence at a time, its rings and states as each part loads them (across: every GPU the whole)
        let whole = cuts.is_empty() && devices.len() == 1 && !across;
        if pages && !(whole && served) {
            return Err(Refusal::unsupported(
                format!("{name}: 12-bit KV pages are a server's pools with the whole model on one GPU; this plan holds one sequence in each part"),
                "Its plan in bf16",
            ));
        }
        let (users, kv) = if let Some(seq) = seq.as_ref().filter(|_| whole && served) {
            let gpu = devs.iter().find(|t| t.ordinal == devices[0]);
            let left = gpu.map_or(0, |t| crate::plan::dynamic_room_of(t.free)).saturating_sub(total);
            seq.plan(users, left).ok_or_else(|| {
                Refusal::unsupported(
                    format!(
                        "{name} at {level} takes about {} of GPU memory for {ctx} tokens of context, its weights {} of it, and the {} has {} free",
                        gib(total + seq.least_bytes(users)),
                        gib(total),
                        gpu.map_or("GPU", |t| t.name.as_str()),
                        gib(gpu.map_or(0, |t| t.free))
                    ),
                    "Ask for less context (--context 4096), or use a GPU with more memory free",
                )
            })?
        } else {
            (1, one_seq * if across { devices.len() as u64 } else { 1 })
        };
        let used: Vec<&Target> = devs.iter().filter(|t| devices.contains(&t.ordinal)).collect();
        let (free, work) = (used.iter().map(|t| t.free).sum::<u64>(), used.iter().map(|t| crate::plan::dynamic_work_of(t.free)).sum::<u64>());
        let host_bytes = cache.as_ref().map_or(0, |k| k.all);
        // (the expert cache: its places, the experts' bytes at their mean, the memory and the disk they take)
        let experts = cache.as_ref().map(|k| {
            // (its layers of experts: a hummingbird directory's own; a release's, the layers the table lists experts of)
            let layers = match hb_dir.and_then(|d| crate::hb_pack::HbDir::open(d).ok().flatten()) {
                Some(d) => d.layers.len(),
                None => crate::dynamic::split::layer_bytes_table(table, nl, 100, 100, &|n| n.contains(".experts.")).0.iter().filter(|&&b| b > 0).count(),
            };
            crate::offload::CachePlan {
                layers,
                experts: k.experts,
                slots: k.places.iter().min().copied().unwrap_or(0),
                expert_bytes: k.all / (layers * k.experts).max(1) as u64,
                in_place: true,
                ram: k.ram,
            }
        });
        // (each stage's layers' experts: a hummingbird directory's files, else the table's expert tensors)
        let host_per_stage = crate::plan::sizing::stage_shares(
            &if hb_experts.is_empty() {
                crate::dynamic::split::layer_bytes_table(table, nl, 100, 100, &|n| n.contains(".experts.")).0
            } else {
                hb_experts.clone()
            },
            if across { &[] } else { &cuts },
            host_bytes,
        );
        Ok(LoadPlan {
            kind: Kind::Dynamic,
            family,
            level,
            layout: if twelve { Layout::Twelve } else { Layout::Tiered },
            context: ctx,
            users,
            chunk: DYN_CHUNK,
            weights: total + host_bytes + dense_ram.1 + tables,
            bf16: 0,
            kv,
            kv12: pages,
            work,
            draft: 0,
            reserve: used.len() as u64 * crate::plan::CONTEXT_FLOOR,
            free,
            gpu: used.iter().map(|t| t.name.as_str()).collect::<Vec<_>>().join(", "),
            experts,
            spread: Some(crate::plan::Spread {
                cuts,
                devices,
                across,
                coded,
                twelve,
                places: cache.as_ref().map_or(Vec::new(), |k| k.places.clone()),
                experts_per_layer: cache.as_ref().map_or(0, |k| k.experts),
                host_bytes,
                host_per_stage,
                gpu_bytes: total,
                dense: dense_ram.0,
                dense_bytes: dense_ram.1,
                tables,
            }),
        })
    }

    #[cfg(test)]
    mod fit_tests {
        use super::*;
        use crate::runtime::config::mc;

        fn dense_36l() -> Config {
            Config {
                hidden: 4096,
                layers: 36,
                heads: 32,
                kv_heads: 8,
                head_dim: 128,
                inter: 12288,
                vocab: 151936,
                eps: 1e-6,
                rope_theta: 1e6,
                max_pos: 40960,
                tie: false,
                eos: vec![],
            }
        }

        #[test]
        fn layout_by_gpu() {
            use crate::weights::smallest_is_fastest;
            let gib = 1u64 << 30;
            let m = 16_380_000_000u64; // (the 36-layer dense shape's matrices in bf16)
            let (t, w) = (Layout::Tiered, Layout::Twelve);
            // (the GPU, its class, its memory, the layout it is given for an 8B model)
            for (gpu, cc, mem, want) in [
                ("L4", (8, 9), 22, t),
                ("RTX 4090", (8, 9), 24, t),
                ("L40S", (8, 9), 48, t),
                ("A10", (8, 6), 24, w),
                ("RTX 3090", (8, 6), 24, w),
                ("A100 40 GB", (8, 0), 40, w),
                ("A100 80 GB", (8, 0), 80, w),
                ("H100", (9, 0), 80, w),
                ("GH200", (9, 0), 96, w),
                ("B200", (10, 0), 180, w),
                ("RTX 5090", (12, 0), 32, w),
            ] {
                assert_eq!(best_layout(cc, mem * gib, m, 1_000_000).0, want, "{gpu}");
                assert_eq!(choose_layout(None, cc, mem * gib, m, 1_000_000).0, want, "{gpu} with no flag");
                // --layout wins on every GPU, either way
                assert_eq!(
                    choose_layout(Some(Layout::Tiered), cc, mem * gib, m, 1_000_000),
                    (Layout::Tiered, "as asked".to_string()),
                    "{gpu} asked for the tiered layout"
                );
                assert_eq!(choose_layout(Some(Layout::Twelve), cc, mem * gib, m, 1_000_000).0, Layout::Twelve, "{gpu} asked for the 12-bit layout");
            }
            // a 24 GiB A10 for a 14B-class model: 12.04 bits do not fit, 10.8 do
            let (l, why) = best_layout((8, 6), 24 * gib, 33_000_000_000, 1_000_000);
            assert_eq!(l, Layout::Tiered);
            assert!(why.contains("only the mma layout fits"), "{why}");
            // neither fits (a 32B model on that A10): the GPU's own layout, and the fit check refuses the model
            assert_eq!(best_layout((8, 6), 24 * gib, 64_000_000_000, 1_000_000).0, Layout::Twelve);
            // the knob: only Ada has the smallest layout as the fastest
            assert!(smallest_is_fastest((8, 9)) && !smallest_is_fastest((8, 0)) && !smallest_is_fastest((9, 0)) && !smallest_is_fastest((8, 6)));
        }

        #[test]
        fn the_dense_kind_on_the_gpus() {
            let c = dense_36l();
            // weights near the measured 10.2 GiB of the engine's record
            let w = weights(&c, Layout::Tiered);
            assert!((10 << 30..11_500_000_000u64).contains(&w), "{w}");
            // an L4 (22.5 GiB free): the want of 8,192 tokens, the 4,096 chunk with cuBLAS
            let l4 = qwen3_fit(&c, "dense-36l", "NVIDIA L4", 22 << 30, Layout::Tiered, None, 1, true).unwrap();
            assert_eq!((l4.context, l4.chunk), (8192, 4096), "{l4:?}");
            assert!(l4.gpu_total() < 22 << 30);
            // the whole 40,960-token window fits an L4 too: the lossless cache is small
            assert_eq!(qwen3_fit(&c, "dense-36l", "NVIDIA L4", 22 << 30, Layout::Tiered, Some(40960), 1, true).unwrap().context, 40960);
            // asking for more than fits is refused, with what fits
            let r = qwen3_fit(&c, "dense-36l", "NVIDIA L4", 13 << 30, Layout::Tiered, Some(40960), 1, true).unwrap_err();
            assert!(r.fix.starts_with("Ask for less: --context ") && r.code == crate::ui::CANNOT, "{r}");
            // a smaller GPU: less context than the want, or a smaller chunk
            let small = qwen3_fit(&c, "dense-36l", "NVIDIA L4", 13 << 30, Layout::Tiered, None, 1, true).unwrap();
            assert!(small.context >= CTX_MIN, "{small:?}");
            assert!(small.gpu_total() < 13 << 30, "{small:?}");
            // too small altogether: a refusal that hands the model to vLLM
            assert_eq!(qwen3_fit(&c, "dense-36l", "NVIDIA L4", 8 << 30, Layout::Tiered, None, 1, true).unwrap_err().code, crate::ui::CANNOT);
            // more than the model's own context
            assert!(qwen3_fit(&Config { max_pos: 2048, ..dense_36l() }, "tiny", "NVIDIA L4", 22 << 30, Layout::Tiered, Some(4096), 1, true).is_err());
        }

        /// One rule sizes every one-GPU path (P2f): a context asked that does not fit is refused, at any level, with the longest that does, and that one loads; where
        /// none is asked, the most the room holds (cut to the path's grain and page), and a scheduler of rooms (`fewer`) sizes one sequence's where several do not fit.
        #[test]
        fn one_rule_sizes_every_one_gpu_path() {
            let q35 = crate::runtime::config::ModelConfig::parse(
                &std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_configs.json")).unwrap(),
            )
            .unwrap();
            let text = serde_json::Value::Object(q35.top["Qwen/Qwen3.8-27B"].as_object().unwrap().clone()).to_string();
            let a = |ctx: &str| crate::args::Args::from(["run", "Qwen/Qwen3.8-27B", "--context", ctx].iter().map(|s| s.to_string()));
            let free = 23u64 << 30;
            let r = lossy_plan_with(&a("131072"), "Qwen/Qwen3.8-27B", Layout::Int4, &mc(&text), None, free, "NVIDIA GeForce RTX 4090", None, None).unwrap_err();
            let most = r.fix.strip_prefix("Ask for less: --context ").unwrap_or_else(|| panic!("{r}")).to_string();
            assert!(r.what.contains("at 131072 tokens of context"), "{r}");
            let p = lossy_plan_with(&a(&most), "Qwen/Qwen3.8-27B", Layout::Int4, &mc(&text), None, free, "NVIDIA GeForce RTX 4090", None, None).unwrap();
            assert!(p.context.to_string() == most && p.gpu_total() <= free, "{p:?}");
            // the rule itself: grain and page, and one sequence's room where `fewer`
            let n = |users: usize, fewer: bool, grain: usize, page: usize| Needs {
                kind: Kind::Qwen35,
                family: "Qwen3.5",
                layout: Layout::Int4,
                weights: 1000,
                bf16: 0,
                per_seq: 100,
                per_token: 10,
                once: Once::default(),
                users,
                fewer,
                draft: 0,
                chunks: vec![(2048, 0)],
                want: 1 << 20,
                least: 16,
                cap: 1 << 20,
                grain,
                page,
            };
            let free = reserve_of(0) + 1000 + 100 + 10 * 1000 + 5;
            assert_eq!(one_gpu(&n(1, false, 1, 1), free, None, "g").unwrap().context, 1000);
            assert_eq!(one_gpu(&n(1, false, 256, 1), free, None, "g").unwrap().context, 768);
            assert_eq!(one_gpu(&n(1, false, 1, 16), free, None, "g").unwrap().context, 992);
            // (four users: each a quarter of the room, or, where the engine takes fewer, the least context for one at least)
            assert_eq!(one_gpu(&n(4, false, 1, 1), free, None, "g").unwrap().context, (10_000 + 100 + 5 - 400) / 40);
            assert!(one_gpu(&n(4, false, 1, 1), free, Some(1000), "g").is_err());
            assert_eq!(one_gpu(&n(4, true, 1, 1), free, Some(1000), "g").unwrap().context, 1000);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::config::mc;

    /// The planner and the command line name no family outside the lines listed here (the owner's rule: one operation per layer kind, nothing outside the
    /// descriptions and the support table branches on a family), so the residue cannot grow back: in src/plan.rs and src/cli.rs, besides their tests (an item
    /// after `#[cfg(test)]`, and the tests' module), a line naming a family's row (`Family::`), its typed config (`ConfigG4`, ...), its module (`crate::gemma4::`,
    /// `dynamic::nemotron::`, ...), its plan's engine (`Kind::Gemma4`, ...), its architecture or model type in words (`ForCausalLM`, `.contains("nemotron`,
    /// ...) fails here unless it is listed, each with the reason it stays.
    /// The experts of a row whose loader packs them itself (`ExpertCache::Packed`, Kimi-K3's) are planned from the Moe kind's shape in the description and the
    /// row's layouts (`packed_experts`): on Kimi-K3's and Kimi-K3.1-Pro's released headers, every layer cut over GPUs and every layer across 8, the source is
    /// the one its family's own reader made (`kimi::ConfigKimi`, the loader's `dynamic::kimi::expert_bytes`, the experts' tensors by their names), to the byte;
    /// a release whose experts are not stored with their scales, which that loader does not read, is refused in words.
    #[test]
    fn a_loaders_own_packs_are_planned_from_the_description() {
        use sizing::CacheOf;
        let tags = match crate::support::row(crate::support::Family::Kimi).route.expert_cache {
            crate::support::ExpertCache::Packed(t) => t,
            other => panic!("{other:?}"),
        };
        // (the family's own reader's source, as the planner made it before)
        let by_family = |table: &crate::sizes::Table, m: &ModelConfig, nl: usize, twelve: bool, across: usize| -> CacheOf {
            let c = crate::kimi::ConfigKimi::of(m, "config.json", None).unwrap();
            let bf16 = if twelve { crate::dynamic::split::TWELVE_PCT } else { crate::dynamic::split::TIERED_PCT };
            let expert = crate::dynamic::kimi::expert_bytes(&c).unwrap();
            let e = &c.experts;
            let all = (c.moe.iter().filter(|&&m| m).count() * e.experts) as u64 * expert;
            let of = |dense, base, ram, place| CacheOf {
                dense,
                base,
                ram,
                moe: c.moe.clone(),
                place,
                experts: e.experts,
                top_k: e.top_k,
                all,
                disk: false,
                across: across.max(1),
                starts: if across > 1 { Vec::new() } else { sizing::stage_starts(m, nl) },
            };
            if across > 1 {
                let n = across as u64;
                let mut held = 0u64;
                for t in &table.rows {
                    if t.name.contains(".block_sparse_moe.experts.")
                        || matches!(crate::sizes::classify(&t.name, c.layers), crate::sizes::Part::Vision | crate::sizes::Part::Mtp)
                    {
                        continue;
                    }
                    let matrix = t.shape.len() == 2 && t.shape.iter().all(|&d| d >= 256);
                    let b = if t.dtype == "BF16" && matrix { t.bytes * bf16 / 100 } else { t.bytes };
                    let sliced = matrix && !t.name.contains("embed_tokens") && t.shape[0] % (64 * n) == 0 && t.bytes >= 16 << 20;
                    held += if sliced { b / n } else { b };
                }
                return of(vec![0; c.moe.len()], held, Vec::new(), expert / n);
            }
            let (dense, base) =
                crate::dynamic::split::layer_bytes_table(table, nl, crate::dynamic::split::CODED_PCT, bf16, &|n| !n.contains(".block_sparse_moe.experts."));
            let ram = crate::dynamic::split::ram_bytes_table(table, nl, "Kimi-K3", crate::dynamic::split::CODED_PCT, bf16);
            of(dense, base, ram, expert)
        };
        for release in ["Kimi-K3", "Kimi-K3.1-Pro"] {
            let (text, table) = recorded_release(release);
            let m = mc(&text);
            let nl = m.num("num_hidden_layers").unwrap();
            for (twelve, across) in [(false, 1), (true, 1), (false, 8), (true, 8)] {
                let got = sizing::packed_experts(&table, &m, nl, twelve, across, tags).unwrap_or_else(|r| panic!("{release}: {}", r.what));
                let want = by_family(&table, &m, nl, twelve, across);
                assert_eq!(format!("{got:?}"), format!("{want:?}"), "{release}, 12-bit {twelve}, across {across}");
            }
        }
        // (experts not stored with their scales: the release's config without its quantization)
        let (text, table) = recorded_release("Kimi-K3");
        let mut v: serde_json::Value = serde_json::from_str(&text).unwrap();
        v["text_config"].as_object_mut().unwrap().remove("quantization_config");
        let m = mc(&v.to_string());
        let r = sizing::packed_experts(&table, &m, 93, false, 1, tags).unwrap_err();
        assert!(r.what.contains("packs experts released quantized, each matrix with its scales"), "{}", r.what);
    }

    #[test]
    fn the_planner_and_the_command_line_name_no_family() {
        // (a line that stays, by a part of it, and why)
        let allowed: [(&str, &str, &str); 13] = [
            ("plan.rs", "use crate::qwen35::Pack35;", "the save's pack: every description's matrices (`pack::described`), the type named where it began"),
            ("plan.rs", "kept: Vec<crate::qwen35::Tensor35>,", "the save's kept tensor: every description's, as above"),
            ("plan.rs", "kind: Kind::Qwen3,", "`dense_fit`, the one sizer of the rows sized `Sizing::Dense`: its plans' engine is the plain dense kind's"),
            ("plan.rs", "mode: crate::qwen4x::PackMode::of(pack, false),", "`codes_plan`, the rows whose experts come from a codes pack (`ExpertCache::Codes`): the pack's modes, its format's"),
            ("plan.rs", "pub mode: crate::qwen4x::PackMode,", "as above"),
            ("plan.rs", "sides: crate::qwen4x::PackMode::of(pack, false).side(pack)", "as above: the sidecars the pack's mode names"),
            ("cli.rs", "if plan.kind == Kind::Gemma4 {", "`open_lossy`: the one-GPU route's loader by the plan's engine (its row's `Lossy::kind`), each arm a family's weight map; a seam of that route's loaders is its own piece"),
            ("cli.rs", "crate::dynamic::gemma4::load_of(", "as above"),
            ("cli.rs", "} else if plan.kind == Kind::Qwen3 {", "as above"),
            ("cli.rs", "crate::dynamic::dense::load_of(", "as above"),
            ("cli.rs", "} else if plan.kind == Kind::NemotronH {", "as above"),
            ("cli.rs", "crate::dynamic::nemotron::load_whole_of(", "as above"),
            ("cli.rs", "crate::dynamic::qwen35::", "as above (the arm of the rest: the Qwen3.5 layout's)"),
        ];
        let families = ["gemma", "nemotron", "qwen", "glm", "kimi", "deepseek", "muse"];
        let named = |code: &str| -> bool {
            let low = code.to_ascii_lowercase();
            code.contains("Family::")
                || ["ConfigG4", "ConfigNemotronH", "Config35", "ConfigGlm", "ConfigKimi", "ConfigMuse", "ConfigQx", "ConfigV4"].iter().any(|t| code.contains(t))
                || ["Kind::Gemma4", "Kind::NemotronH", "Kind::Qwen3"].iter().any(|t| code.contains(t))
                || ["gemma4", "nemotron", "qwen35", "glm", "kimi", "muse", "qwen4x", "deepseek_v4", "dense"]
                    .iter()
                    .any(|m| code.contains(&format!("crate::{m}::")) || code.contains(&format!("dynamic::{m}::")))
                || code.contains("ForCausalLM")
                || code.contains("ForConditionalGeneration")
                || code.contains("is_gemma4")
                || families.iter().any(|f| ["contains(\"", "starts_with(\"", "== \"", "!= \""].iter().any(|op| low.contains(&format!("{op}{f}"))))
        };
        let mut found = Vec::new();
        for (file, text) in [("plan.rs", include_str!("plan.rs")), ("cli.rs", include_str!("cli.rs"))] {
            // (the code before the tests' module, an item after `#[cfg(test)]` left out by its braces)
            let code = text.split("\n#[cfg(test)]\nmod tests {").next().unwrap();
            let (mut skip, mut depth, mut opened) = (false, 0i64, false);
            for (i, line) in code.lines().enumerate() {
                let t = line.trim_start();
                if t.starts_with("#[cfg(test)]") {
                    (skip, depth, opened) = (true, 0, false);
                    continue;
                }
                let bare = t.split(" //").next().unwrap_or("");
                if skip {
                    depth += bare.matches('{').count() as i64 - bare.matches('}').count() as i64;
                    opened |= bare.contains('{');
                    if opened && depth <= 0 || !opened && bare.ends_with(';') {
                        skip = false;
                    }
                    continue;
                }
                if t.starts_with("//") || !named(bare) {
                    continue;
                }
                if !allowed.iter().any(|(f, part, _)| *f == file && bare.contains(part)) {
                    found.push(format!("{file}:{}: {}", i + 1, t));
                }
            }
        }
        assert!(
            found.is_empty(),
            "a family named in the planner or the command line (list it here with its reason, or read it from the description or the support row):\n{}",
            found.join("\n")
        );
        assert!(allowed.iter().all(|(_, _, why)| !why.is_empty()));
    }

    /// The rows a step's draws see fit the dynamic route's work many times over at the most users a step takes and the largest vocabulary of the dynamic route's families
    /// (Gemma 4's 262,144): 64 MiB of the 3 GiB the work holds before the GPU's share.
    #[test]
    fn the_draws_rows_are_a_part_of_the_work() {
        let most = draw_rows_bytes(262_144, crate::sched::MAX_USERS);
        assert_eq!(most, 64 << 20);
        assert!(most * 16 <= dynamic_work_of(0), "{most} of {}", dynamic_work_of(0));
    }

    /// The planner's keys and values a token (`Shape::kv_full`, `kv_sliding`) against what the executor's pools take a token (each ring's `token_bytes`, bf16), for a random case of each
    /// attention kind: plain and sliding attention (Gemma 4's), the gated attention beside recurrent layers (Qwen3.5's layout), latent attention with its indexer's sparse keys
    /// (GLM-5.3-Flash's toy case, as glmrandom makes it), compressed entries beside a window and an indexer (DeepSeek-V4's small random case). The same bytes, exactly.
    #[cfg(feature = "dev")]
    #[test]
    fn the_planners_keys_and_values_a_token_are_the_pools() {
        use crate::dynamic::Stage;
        let fixture = |f: &str| std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(f)).unwrap();
        let qwen35: serde_json::Value = serde_json::from_str(&fixture("qwen35_tiny_manifest.json")).unwrap();
        let q35 = qwen35["config"].as_str().unwrap().to_string();
        let glm = fixture("glm5_next_toy_config.json");
        let cases: Vec<(&str, String, crate::dynamic::ModelDesc)> = vec![
            ("plain and sliding attention", crate::logits::GEMMA4_RANDOM_CONFIG.to_string(), {
                crate::dynamic::gemma4::describe(&crate::gemma4::ConfigG4::parse(crate::logits::GEMMA4_RANDOM_CONFIG, None, false).unwrap())
            }),
            ("gated attention beside recurrent layers", q35.clone(), {
                let c = crate::qwen35::Config35::parse(&q35, "qwen35", None).unwrap();
                crate::dynamic::qwen35::describe(&c, &Stage::whole(c.layers)).unwrap()
            }),
            ("latent attention and its indexer's keys", glm.clone(), {
                let c = crate::glm::ConfigGlm::parse(&glm, "GLM-5.3-Flash", None).unwrap();
                crate::dynamic::glm::describe(&c, &Stage::whole(c.layers)).unwrap()
            }),
            ("compressed entries, a window and an indexer", crate::dsv4_ref::SMALL_CONFIG.to_string(), {
                let c = crate::deepseek_v4::ConfigV4::parse(crate::dsv4_ref::SMALL_CONFIG, "random", None).unwrap();
                let vo = crate::dynamic::deepseek_v4::V4Opts::of(crate::weights::Layout::Tiered, 1024);
                let kinds = crate::runtime::describe::describe(&serde_json::from_str(crate::dsv4_ref::SMALL_CONFIG).unwrap()).unwrap();
                crate::dynamic::lower::deepseek_v4(&kinds, vec![1], 1 << 20, &Stage::whole(c.layers), &vo).unwrap()
            }),
        ];
        let mut bad = Vec::new();
        for (kind, text, d) in &cases {
            let sh = Shape::from_config(text).unwrap();
            let full: usize = d.rings.iter().filter(|r| r.window == 0).map(|r| r.token_bytes()).sum();
            let sliding: usize = d.rings.iter().filter(|r| r.window > 0).map(|r| r.token_bytes()).sum();
            eprintln!("{kind}: the pools {full} + {sliding} (sliding) bytes a token, the planner {} + {}", sh.kv_full, sh.kv_sliding);
            if (sh.kv_full, sh.kv_sliding) != (full as f64, sliding as f64) {
                bad.push(format!("{kind}: the pools {full} + {sliding}, the planner {} + {}", sh.kv_full, sh.kv_sliding));
            }
            // (in 12-bit pages: each ring the pages take (kv12::shapes) at the planner's bytes a token there, which kv12check holds to the pages' own; the others as bf16)
            let taken: Vec<Option<crate::dynamic::kv12::KvShape>> = (0..d.rings.len()).map(|r| crate::dynamic::kv12::pages_shape(d, r)).collect();
            let bytes = |i: usize| taken[i].as_ref().map_or(d.rings[i].token_bytes(), |k| sizing::kv12_token_bytes(k.heads, k.channels, k.one) as usize);
            let full12: usize = (0..d.rings.len()).filter(|&i| d.rings[i].window == 0).map(bytes).sum();
            let sliding12: usize = (0..d.rings.len()).filter(|&i| d.rings[i].window > 0).map(bytes).sum();
            eprintln!("{kind}: in 12-bit pages the pools {full12} + {sliding12}, the planner {} + {}", sh.kv12_full, sh.kv12_sliding);
            if (sh.kv12_full, sh.kv12_sliding) != (full12 as f64, sliding12 as f64) {
                bad.push(format!("{kind}: in 12-bit pages the pools {full12} + {sliding12}, the planner {} + {}", sh.kv12_full, sh.kv12_sliding));
            }
        }
        assert!(bad.is_empty(), "{bad:?}");
    }

    /// The rings of each family whose description the 12-bit pages take, as the planner reads them from the config and as the executor's pools take them
    /// (`kv12::pages_shape`): a random case of Gemma 4 (sliding rings of 256 channels, K = V global ones of 512) and the gemma-4-12B release's config, the Qwen3.5
    /// layout's gated attention (and its wide case), the plain dense kind, Kimi-K3's and GLM-5.3-Flash's latents: (heads, channels, one stream, window) each.
    #[cfg(feature = "dev")]
    fn twelve_bit_cases() -> Vec<(&'static str, String, Vec<(usize, usize, bool, usize)>)> {
        use crate::dynamic::Stage;
        let manifest = |f: &str| -> String {
            let v: serde_json::Value = serde_json::from_str(&fixture(f)).unwrap();
            v["config"].as_str().map(String::from).unwrap_or_else(|| v["config"].to_string())
        };
        let rings = |d: &crate::dynamic::ModelDesc| -> Vec<(usize, usize, bool, usize)> {
            let mut v: Vec<_> =
                (0..d.rings.len()).filter_map(|r| crate::dynamic::kv12::pages_shape(d, r)).map(|s| (s.heads, s.channels, s.one, s.window)).collect();
            v.sort_unstable();
            v
        };
        let gemma = |text: &str| rings(&crate::dynamic::gemma4::describe(&crate::gemma4::ConfigG4::parse(text, None, false).unwrap()));
        let q35 = |text: &str| {
            let c = crate::qwen35::Config35::parse(text, "qwen35", None).unwrap();
            rings(&crate::dynamic::qwen35::describe(&c, &Stage::whole(c.layers)).unwrap())
        };
        let dense = |text: &str| {
            let dir = std::env::temp_dir().join(format!("glyd-plan-twelve-dense-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("config.json"), text).unwrap();
            let d = crate::dynamic::dense::lowered(&dir).unwrap_or_else(|e| panic!("{}", e.msg));
            let _ = std::fs::remove_dir_all(&dir);
            rings(&d)
        };
        let kimi: serde_json::Value = serde_json::from_str(&fixture("headers/Kimi-K3.json")).unwrap();
        let kimi = kimi["config"].as_str().unwrap().to_string();
        let kc = crate::kimi::ConfigKimi::parse(&kimi, "Kimi-K3", None).unwrap();
        let kd = crate::dynamic::kimi::describe(&kc, &Stage::whole(kc.layers), &vec![(true, true); kc.layers], false).unwrap();
        let glm = fixture("glm5_next_toy_config.json");
        let gc = crate::glm::ConfigGlm::parse(&glm, "GLM-5.3-Flash", None).unwrap();
        let gd = crate::dynamic::glm::describe(&gc, &Stage::whole(gc.layers)).unwrap();
        // (GLM-5.3's full-size layout, its random case's config: the latent attention on every layer, its indexer's ring bf16)
        let dsa = fixture("glm_moe_dsa_toy_config.json");
        let dk = crate::runtime::describe::describe(&serde_json::from_str(&dsa).unwrap()).unwrap();
        let dd = crate::dynamic::lower::latent_moe(&dk, Vec::new(), 4096, &Stage::whole(dk.desc.blocks.len() / 2)).unwrap_or_else(|e| panic!("{}", e.msg));
        let (g4, g12, q, qw, dn) = (
            crate::logits::GEMMA4_RANDOM_CONFIG.to_string(),
            fixture("gemma4_12b_config.json"),
            manifest("qwen35_tiny_manifest.json"),
            manifest("qwen35_wide_tiny_manifest.json"),
            manifest("dense_tiny_manifest.json"),
        );
        vec![
            ("gemma4 (random)", g4.clone(), gemma(&g4)),
            ("gemma-4-12B", g12.clone(), gemma(&g12)),
            ("qwen35 (random)", q.clone(), q35(&q)),
            ("qwen35-wide (random)", qw.clone(), q35(&qw)),
            ("dense (random)", dn.clone(), dense(&dn)),
            ("Kimi-K3", kimi, rings(&kd)),
            ("glm (random)", glm, rings(&gd)),
            ("glm-dsa (random)", dsa, rings(&dd)),
        ]
    }

    /// The planner's 12-bit rings from a config hold the executor's pools' (`twelve_bit_cases`), every family whose description the pages take, each at the pool's
    /// bytes a token and once-only bytes (`kv12::token_bytes`, `kv12::once_bytes`, the pool's own functions): the load plans of the families whose runs put their
    /// keys and values in the pages (Gemma 4's `ConfigG4::rings12`, the plain dense kind's `sizing::dense_rings`) are the pools' rings exactly; the general plan's
    /// (`Shape::kv12_rings`) every pool ring at least (Kimi-K3's config names its delta-attention layers in its own key, which the general plan does not read: it
    /// counts every layer a latent attention, more bytes and never fewer).
    #[cfg(feature = "dev")]
    #[test]
    fn the_planners_twelve_bit_rings_are_the_pools() {
        let tuples = |v: &[crate::dynamic::kv12::KvShape]| {
            let mut t: Vec<_> = v.iter().map(|s| (s.heads, s.channels, s.one, s.window)).collect();
            t.sort_unstable();
            t
        };
        let cases = twelve_bit_cases();
        for (name, text, pools) in &cases {
            assert!(!pools.is_empty(), "{name}: no ring in 12-bit pages");
            let mut planned = tuples(&Shape::from_config(text).unwrap().kv12_rings);
            eprintln!("{name}: {} rings in 12-bit pages, the general plan counts {}", pools.len(), planned.len());
            for r in pools {
                let at = planned.iter().position(|x| x == r).unwrap_or_else(|| panic!("{name}: the general plan does not count a ring {r:?}"));
                planned.remove(at);
            }
            if name.contains("gemma") {
                assert_eq!(&tuples(&crate::gemma4::ConfigG4::parse(text, None, false).unwrap().rings12()), pools, "{name}: Gemma 4's load plan's rings");
            }
            if name.starts_with("dense") {
                assert_eq!(
                    &tuples(&sizing::dense_rings(&crate::weights::Config::parse(text, "c", None).unwrap(), true)),
                    pools,
                    "{name}: the dense load plan's rings"
                );
            }
        }
        assert_eq!(cases.len(), 8);
    }

    /// The rule (R4b, #332): a plan never promises more tokens than the pool it sizes holds, in bf16 pages and in 12-bit pages. Gemma 4's load plan (the random case's
    /// config and gemma-4-12B's: sliding rings of 256, K = V global ones of 512) and the plain dense kind's (the random case's config and the 36-layer shape), for one user
    /// and eight, served, on GPUs whose memory binds the context: a pool made by the executor's rule (`Dyn::page` of a plan's bytes: what the 12-bit pages spend once
    /// and each group's block 0 off the top, then `rows::pool_blocks` over each group's bytes a token, every sliding group a window and a chunk a sequence) from the
    /// plan's keys and values holds one sequence's planned context (the plan's users share the pool: `fewer`), and the plan's reserve over the executor's
    /// (`Dyn::page`'s 512 MiB) holds the block tables.
    #[cfg(feature = "dev")]
    #[test]
    fn a_plan_never_promises_more_tokens_than_its_pool_holds() {
        use crate::dynamic::kv12;
        let gib = 1u64 << 30;
        // (the pool of a plan: groups (bytes a token, its sliding window or 0) of 12-bit rings where `twelve`, else bf16; what the pages spend once; one sequence's context
        // held, every group)
        let pool_holds = |p: &LoadPlan, groups: &[(u64, usize)], rings: &[kv12::KvShape], zero: u64| -> bool {
            // (the loader's budget: the plan's bytes less what the 12-bit pages spend once and each group's block 0, `rows::SeqRoom::pool_budget`)
            let (once, chunk) = (kv12::once_bytes(rings, p.context, p.chunk) + zero, p.chunk);
            let per: Vec<usize> = groups.iter().map(|g| g.0 as usize).collect();
            let keep: Vec<Option<usize>> = groups.iter().map(|g| (g.1 > 0).then_some(g.1 + chunk)).collect();
            let covers = vec![16; groups.len()];
            let most: Vec<usize> = keep.iter().map(|&k| p.users * crate::dynamic::seq_blocks(k, p.context, 16)).collect();
            let tables = (p.users * groups.len() * p.context.div_ceil(16) * 4) as u64;
            let budget = p.kv.saturating_sub(once) as usize;
            let least: Vec<usize> = keep.iter().map(|&k| crate::dynamic::seq_blocks(k, p.context, 16)).collect();
            let blocks = crate::dynamic::pool_blocks(budget, &per, &covers, &keep, &most, &least);
            tables <= reserve_of(0) - (512 << 20)
                && blocks.iter().zip(&keep).all(|(&b, k)| match k {
                    Some(k) => b >= crate::dynamic::seq_blocks(Some(*k), p.context, 16),
                    None => b * 16 >= p.context,
                })
        };
        let mut checked: Vec<(u32, bool, usize)> = Vec::new();
        for text in [crate::logits::GEMMA4_RANDOM_CONFIG.to_string(), fixture("gemma4_12b_config.json")] {
            let cfg = crate::gemma4::ConfigG4::parse(&text, None, false).unwrap();
            for (twelve, users) in [(false, 1), (true, 1), (false, 8), (true, 8)] {
                for room in [2u64 << 20, 8 << 20, 32 << 20, 64 << 20, 256 << 20, gib, 6 * gib, 16 * gib] {
                    let lv = sizing::level_of(&args(&["run", "g"]), "g", "g", crate::weights::Layout::Tiered, &mc(&text), None, None, Some(LOSSY_CTX)).unwrap();
                    let n = sizing::LevelOf { weights: 0, bf16: 0, ..lv }.needs(crate::weights::Layout::Tiered, users, twelve, true, 0);
                    let free = reserve_of(0) + LOSSY_WORK + room;
                    let Ok(p) = sizing::one_gpu(&n, free, None, "g") else { continue };
                    let bytes = |k: crate::gemma4::KindG4| {
                        let (d, nk, _) = cfg.attn_of(k);
                        let s = kv12::ring_shape(nk, d, cfg.heads, 0, false).filter(|_| twelve);
                        s.map_or((nk * d * 4) as u64, |s| kv12::token_bytes(s.heads, s.channels, s.one))
                    };
                    let count = |k: crate::gemma4::KindG4| cfg.kinds.iter().filter(|&&x| x == k).count() as u64;
                    let groups = [
                        (count(crate::gemma4::KindG4::Sliding) * bytes(crate::gemma4::KindG4::Sliding), cfg.window),
                        (count(crate::gemma4::KindG4::Global) * bytes(crate::gemma4::KindG4::Global), 0),
                    ];
                    let rings = if twelve { cfg.rings12() } else { Vec::new() };
                    assert!(
                        pool_holds(&p, &groups, &rings, sizing::block_zero(&cfg.group_token_bytes())),
                        "Gemma 4 ({} layers), {} pages, {users} users, {room} bytes: {p:?}",
                        cfg.layers,
                        if twelve { "12-bit" } else { "bf16" }
                    );
                    checked.push((cfg.layers as u32, twelve, users));
                }
            }
        }
        let dense_case: serde_json::Value = serde_json::from_str(&fixture("dense_tiny_manifest.json")).unwrap();
        for text in [dense_case["config"].as_str().map(String::from).unwrap_or_else(|| dense_case["config"].to_string()), DENSE_36L.to_string()] {
            let cfg = crate::weights::Config::parse(&text, "c", None).unwrap_or_else(|e| panic!("{}", e.msg));
            for (twelve, users) in [(false, 1), (true, 1), (false, 8), (true, 8)] {
                for room in [2u64 << 20, 8 << 20, 32 << 20, 64 << 20, 256 << 20, gib, 6 * gib, 16 * gib] {
                    let n = sizing::dense_needs(&cfg, "dense", crate::weights::Layout::Tiered, users, false, twelve, true);
                    let free = n.weights + reserve_of(0) + n.chunks.last().unwrap().1 + room;
                    let Ok(p) = sizing::one_gpu(&n, free, None, "g") else { continue };
                    let groups = [(sizing::kv_per_token(&cfg, twelve), 0)];
                    assert!(
                        pool_holds(&p, &groups, &sizing::dense_rings(&cfg, twelve), sizing::block_zero(&[sizing::kv_per_token(&cfg, false)])),
                        "dense ({} layers), {} pages, {users} users, {room} bytes: {p:?}",
                        cfg.layers,
                        if twelve { "12-bit" } else { "bf16" }
                    );
                    checked.push((1000 + cfg.layers as u32, twelve, users));
                }
            }
        }
        checked.dedup();
        // (every config, page format and user count: some plan of each checked)
        assert_eq!(checked.len(), 16, "{checked:?}");
    }

    /// Both formats of the keys and values are lossless, so a plan takes the 12-bit pages GLYD_KV=lossless asks for only where they reserve no more than bf16
    /// (`sizing::one_gpu_pages`). One user at gemma-4-12B's whole context (262,144 tokens) reserves more in them than in bf16 (their worst case beside the tokens:
    /// a raw slab for every page of the sequence and the rows a pass unpacks): on an L4 and an RTX 4080 SUPER the plan is bf16's, its level bf16's (penguin on
    /// the L4, as the plan was before that worst case was counted; swift on the 4080 SUPER, where the 12-bit pages were planned at kestrel in fewer bytes than
    /// their pools take). Eight users share the worst case: on the L4 the 12-bit pages plan more tokens each than bf16, at the same level. The plan says which.
    #[test]
    fn the_twelve_bit_pages_are_planned_only_where_they_reserve_no_more_than_bf16() {
        let text = fixture("gemma4_12b_config.json");
        let cfg = crate::gemma4::ConfigG4::parse(&text, None, false).unwrap();
        let (w, bf16) = cfg.weight_bytes(crate::weights::Layout::Tiered);
        let kv = |twelve: bool, users: usize, ctx: usize| {
            let lv = sizing::level_of(&args(&["run", "g"]), "g", "g", crate::weights::Layout::Tiered, &mc(&text), None, None, Some(ctx)).unwrap();
            assert_eq!((lv.weights, lv.bf16), (w, bf16));
            lv.needs(crate::weights::Layout::Tiered, users, twelve, true, 0).kv(users as u64, ctx)
        };
        assert!(kv(true, 1, 262_144) > kv(false, 1, 262_144) && kv(true, 8, 19_680) < kv(false, 8, 19_680));
        let dir = model_dir("kvformat", "gemma-4-12B", &text);
        let m = dir.to_str().unwrap();
        let plan = |lossless: bool, words: &[&str]| -> LoadPlan {
            sizing::asking_lossless(lossless, || planned(&args(words))).unwrap().plan.chosen.and_then(|c| c.load).expect("a plan")
        };
        for (preset, level) in [("l4", "penguin"), ("rtx4080s", "swift")] {
            let words = ["serve", m, "--machine", preset, "--users", "1", "--no-draft", "--context", "262144"];
            let (b, t) = (plan(false, &words), plan(true, &words));
            assert_eq!(t, b, "{preset}: one user at the whole context is planned as in bf16");
            assert_eq!((t.level, t.context, t.kv_format()), (level, 262_144, "bf16"), "{preset}");
            assert!(t.json(m).contains(r#""kv_format":"bf16""#) && t.text().contains("keys and values in bf16"), "{}", t.text());
        }
        let words = ["serve", m, "--machine", "l4", "--users", "8", "--no-draft"];
        let (b, t) = (plan(false, &words), plan(true, &words));
        assert!(t.kv12 && !b.kv12 && t.level == b.level && t.context > b.context, "bf16 {b:?}, 12-bit {t:?}");
        assert!(t.json(m).contains(r#""kv_format":"12-bit""#) && t.text().contains("keys and values in 12-bit pages"), "{}", t.text());
    }

    /// The lossy route's levels are planned by one rule (`sizing::lossy_plan_inner`, what the description says: `sizing::level_of`), Gemma 4's too: a refusal
    /// gives one user's figure and says it as one user's (the server holds fewer users where the GPU holds fewer: `Needs::fewer`), whatever the users asked; and a
    /// model with no experts to keep in the computer's memory is refused `--expert-cache` and `--host-memory` in words. Gemma 4's own planner, before, said
    /// "for each of 4 users" of one user's figure, and ignored both flags.
    #[test]
    fn a_lossy_levels_refusal_is_one_users_figure_and_its_flags_are_held_to_its_experts() {
        let c = mc(&fixture("gemma4_12b_config.json"));
        let plan = |w: &[&str]| {
            sizing::lossy_plan_with(&args(w), "g", crate::weights::Layout::Tiered, &c, None, 15 << 30, "NVIDIA GeForce RTX 4080 SUPER", None, Some(64 << 30))
        };
        let (one, four) = (plan(&["serve", "g", "--users", "1", "--no-draft"]).unwrap_err(), plan(&["serve", "g", "--users", "4"]).unwrap_err());
        assert!(one.what.starts_with("g at the penguin level takes about ") && four.what == one.what, "{one} / {four}");
        for flag in ["--expert-cache", "--host-memory"] {
            let r = plan(&["serve", "g", flag, "8"]).unwrap_err();
            assert!(r.code == 1 && r.what == format!("{flag}: g has no experts to keep in the computer's memory"), "{flag}: {r}");
        }
        // (the Qwen3.5 layout's, as before: the same words of one user's figure, and the flags refused)
        let q = mc(&json::to_compact(json::parse(&fixture("qwen35_configs.json")).unwrap().get("Qwen/Qwen3.8-27B").unwrap()));
        let qp = |w: &[&str]| sizing::lossy_plan_with(&args(w), "q", crate::weights::Layout::Int4, &q, None, 8 << 30, "NVIDIA L4", None, Some(64 << 30));
        assert_eq!(qp(&["serve", "q", "--users", "4"]).unwrap_err().what, qp(&["serve", "q", "--users", "1"]).unwrap_err().what);
        assert_eq!(qp(&["serve", "q", "--expert-cache", "8"]).unwrap_err().code, 1);
    }

    /// The one rule's last step before it refuses is every description's with experts (#232): the Qwen3.5 layout with experts at a lossy level its GPU does not
    /// hold whole keeps its experts in the computer's memory at the level's block layout and a cache of them on the GPU, as the Nemotron-H layout does. Every
    /// user's keys, values and states first, the engine's work next, the places what is left; refused, with the memory it needs, where the computer's memory
    /// does not hold the experts; `--expert-cache GIB` asks for a cache of that much. (Before, these levels planned no cache: refused, and the flags too.)
    #[test]
    fn the_qwen35_layout_with_experts_keeps_its_experts_in_memory_at_a_lossy_level() {
        let q = json::parse(&fixture("qwen35_configs.json")).unwrap();
        let c = mc(&json::to_compact(q.get("Qwen/Qwen3.5-35B-A3B").unwrap()));
        let plan = |w: &[&str], free: u64, host: u64| {
            sizing::lossy_plan_with(&args(w), "q", crate::weights::Layout::Int4, &c, None, free << 30, "NVIDIA GeForce RTX 4080 SUPER", None, Some(host << 30))
        };
        assert!(plan(&["run", "q"], 40, 56).unwrap().experts.is_none(), "a GPU that holds it whole keeps every expert on it");
        let p = plan(&["run", "q"], 15, 56).unwrap();
        let e = p.experts.expect("its experts in the computer's memory, a cache of them on the GPU");
        let src = sizing::MoeShape::of(&crate::runtime::describe::describe_config(&c).unwrap(), &c)
            .unwrap()
            .cache_of(&c, crate::weights::Layout::Int4, 0, &[])
            .unwrap();
        assert_eq!((e.layers, e.experts, e.expert_bytes, e.host_bytes()), (40, 256, src.place, src.all), "{e:?}");
        assert!(e.slots >= 2 * 8 && e.slots < 256 && !e.in_place && p.gpu_total() <= 15 << 30, "{e:?} {}", p.text());
        // (the block layout's expert: its gate and up rows joined, then its down, codes and scales)
        let ((ud, us), (dd, ds)) = (
            crate::moe::expert_strides(glyd_gpu::pack::INT4_TAG, 1024, 2048).unwrap(),
            crate::moe::expert_strides(glyd_gpu::pack::INT4_TAG, 2048, 512).unwrap(),
        );
        assert_eq!(src.place, ud + us + dd + ds);
        // (served to 8 at 4,096 tokens: their keys, values and states first, the places fewer)
        let s8 = plan(&["serve", "q", "--users", "8", "--context", "4096"], 15, 56).unwrap();
        assert!(s8.users == 8 && s8.experts.unwrap().slots <= e.slots && s8.gpu_total() <= 15 << 30, "{}", s8.text());
        let r = plan(&["run", "q"], 15, 8).unwrap_err();
        assert!(r.what.contains("of the computer's memory besides at least") && r.what.contains("this computer has 8.0 GiB free"), "{r}");
        let asked = plan(&["run", "q", "--expert-cache", "2"], 40, 56).unwrap().experts.unwrap();
        assert_eq!(asked.slots, ((2u64 << 30) / (40 * src.place)) as usize);
    }

    /// A release's bf16 experts in the expert cache are held in the lossless tiered layout (src/dynamic/experts.rs): their places and the computer's memory
    /// they take are counted at a bf16 matrix's share as held (`split::TIERED_PCT`, an upper bound), not at the release's bytes. Qwen3.8-2.4T-A95B's
    /// 4,416 GiB of bf16 experts are about 3,180 GiB held. (Before: 4,416 GiB, more than is held.)
    #[test]
    fn a_releases_bf16_experts_are_counted_as_the_cache_holds_them() {
        let (config, table) = recorded_release("Qwen3.8-2.4T-A95B");
        let c = mc(&config);
        let nl = c.num("num_hidden_layers").unwrap();
        let src = sizing::release_experts(&table, &c, nl, false).unwrap();
        let (released, _) = crate::dynamic::split::layer_bytes_table(&table, nl, 100, 100, &|n| n.contains(".experts."));
        let released: u64 = released.iter().sum();
        assert_eq!(released, 4416 << 30);
        assert!(
            src.all <= released * crate::dynamic::split::TIERED_PCT / 100 && src.all + (1 << 30) > released * crate::dynamic::split::TIERED_PCT / 100,
            "{}",
            src.all
        );
        assert_eq!(src.place, src.all / (src.layers() * src.experts) as u64);
    }

    /// Each stage of a lossy plan in stages within its GPU's room: its layers' bytes besides the experts (each layer's own `dense`, the rest `base` half with the
    /// first stage and half with the last), its places of the experts and its share of the sequence, against the GPU's free memory less the reserve and the
    /// lossy engine's work; the cuts ascending inside the layers, each where a stage may begin (`stage_starts`), the GPUs the machine's, one a stage.
    fn stages_within_rooms(lp: &LoadPlan, c: &ModelConfig, devs: &[Target], layout: crate::weights::Layout) {
        let a = args(&["run", "m"]);
        let lv = sizing::level_of(&a, "m", "m", layout, c, None, None, None).unwrap();
        let (cut, sp) = (lv.layers.as_ref().unwrap(), lp.spread.as_ref().unwrap());
        let nl = cut.whole.len();
        let starts = sizing::stage_starts(c, nl);
        assert!(sp.cuts.windows(2).all(|w| w[0] < w[1]) && sp.cuts.iter().all(|&x| x > 0 && x < nl && starts.get(x).copied().unwrap_or(true)), "{:?}", sp.cuts);
        assert_eq!(sp.devices.len(), sp.cuts.len() + 1);
        let kv = crate::dynamic::layer_rooms(&cut.low, lp.context, lp.chunk);
        assert_eq!((lp.users, lp.kv), (1, kv.iter().sum::<u64>()), "one sequence, each layer its share");
        let src = lv.experts.as_ref().map(|e| e.as_ref().unwrap().clone());
        let (own, base) = match (&src, lp.experts) {
            (Some(s), Some(_)) => (s.dense.clone(), s.base),
            _ => (cut.whole.clone(), lv.weights - cut.whole.iter().sum::<u64>()),
        };
        let bounds: Vec<usize> = std::iter::once(0).chain(sp.cuts.iter().copied()).chain(std::iter::once(nl)).collect();
        for (k, w) in bounds.windows(2).enumerate() {
            let t = devs.iter().find(|t| t.ordinal == sp.devices[k]).expect("a GPU of the machine");
            let places = match (&src, sp.places.get(k)) {
                (Some(s), Some(&p)) => (w[0]..w[1]).filter(|&l| s.moe[l]).count() as u64 * p as u64 * s.place,
                _ => 0,
            };
            let held = (w[0]..w[1]).map(|l| own[l] + kv[l]).sum::<u64>()
                + places
                + if k == 0 { base / 2 } else { 0 }
                + if k + 1 == sp.devices.len() { base - base / 2 } else { 0 };
            let room = t.free - LOSSY_WORK - reserve_of(t.free);
            assert!(held <= room, "stage {k} on GPU {}: {} of {}", t.ordinal, crate::ui::gib(held), crate::ui::gib(room));
        }
        let distinct: std::collections::BTreeSet<i32> = sp.devices.iter().copied().collect();
        assert_eq!(distinct.len(), sp.devices.len(), "a GPU a stage");
    }

    /// #333: Qwen3.8-2.4T-A95B on its reference hardware, eight H100s. At penguin the machine is the limit (its experts held lossless take more than its 1,800 GiB
    /// of memory): refused with the numbers. At wren, swift and kestrel one GPU does not hold it, even with its experts in the computer's memory (88.6, 105.0 and
    /// 121.5 GiB of one GPU's memory): its layers are cut into stages over the eight by the dynamic route's rule (`sizing::lossy_stages`), its experts in the
    /// computer's memory and each stage's places of them, every stage within its GPU's room. `fit` prints that plan (before, each of these levels was refused);
    /// one H100 still refuses them.
    #[test]
    fn the_largest_qwen_release_is_cut_into_stages_at_its_lossy_levels_on_eight_h100s() {
        let (config, table) = recorded_release("Qwen3.8-2.4T-A95B");
        let dir = model_dir("stages", "Qwen3.8-2.4T-A95B", &config);
        table.write(&dir.join(crate::sizes::GGUF_TENSORS));
        let m = dir.to_str().unwrap().to_string();
        let fit = |level: &str, machine: &str| planned(&args(&["fit", &format!("{m}:{level}"), "--machine", machine])).unwrap();
        let r = fit("penguin", "8xh100").refusal.expect("penguin: the machine's memory is the limit");
        assert!(r.what.contains("they take 3444.5 GiB, and 1800.0 GiB is free"), "{r}");
        let mach = Machine::preset("8xh100").unwrap();
        let devs: Vec<Target> = mach.devices.iter().enumerate().map(|(i, d)| Target::of(d, i as i32, Some(mach.ram_free))).collect();
        let c = mc(&config);
        for (level, layout) in [("wren", crate::weights::Layout::Int4), ("swift", crate::weights::Layout::Int5), ("kestrel", crate::weights::Layout::Int6)] {
            let f = fit(level, "8xh100");
            assert!(f.refusal.is_none(), "{level}: {:?}", f.refusal);
            let lp = f.plan.chosen.as_ref().and_then(|c| c.load.clone()).expect("a plan");
            println!("{level}: {}", lp.text());
            let (sp, e) = (lp.spread.as_ref().expect("in stages"), lp.experts.expect("its experts in the computer's memory"));
            assert!(lp.level == level && sp.devices.len() > 1 && !sp.across, "{level}: {}", lp.text());
            assert!(e.slots >= 2 * 10 && e.slots < 512 && e.experts == 512 && e.layers == 92, "{level}: {e:?}");
            assert!(sp.host_bytes == e.host_bytes() && sp.host_bytes + HOST_MARGIN <= mach.ram_free, "{level}: {}", lp.text());
            assert_eq!(sp.host_per_stage.iter().sum::<u64>(), sp.host_bytes);
            assert!(lp.text().contains("stages from layer [0, "), "{}", lp.text());
            stages_within_rooms(&lp, &c, &devs, layout);
            assert!(fit(level, "h100").refusal.is_some(), "{level}: one H100");
        }
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
    }

    /// A lossy level's stages hold every expert where the GPUs hold them all (`plan_cuts_at`, as the dynamic route cuts a model it holds whole), the experts in
    /// the computer's memory only where they do not; `--split CUTS --devices GPUS` cuts where it says, refused in words where the cuts are not the layers', a GPU
    /// is not the machine's, or a GPU's stages take more than its room; a level whose layers its row does not cut (Gemma 4's) refuses `--split`.
    #[test]
    fn a_lossy_level_is_cut_into_stages_where_one_gpu_does_not_hold_it() {
        let q = json::parse(&fixture("qwen35_configs.json")).unwrap();
        let c = mc(&json::to_compact(q.get("Qwen/Qwen3.5-35B-A3B").unwrap()));
        let t = |i: i32, free: u64| Target {
            ordinal: i,
            name: "NVIDIA GeForce RTX 4070".into(),
            cc: (8, 9),
            total: free + (1 << 30),
            free,
            host: Some(64 << 30),
            cublas: None,
        };
        let plan = |w: &[&str], devs: &[Target]| {
            sizing::lossy_plan_over(&args(w), "q", crate::weights::Layout::Int6, &c, None, devs[0].free, &devs[0].name, None, Some(64 << 30), devs)
        };
        // (kestrel: about 27 GiB of weights; one GPU of 20 GiB free does not hold it, two do, every expert on them)
        let two = [t(0, 20 << 30), t(1, 20 << 30)];
        let one = plan(&["run", "q"], &two[..1]).unwrap();
        assert!(one.spread.is_none() && one.experts.is_some(), "one GPU: its experts in the computer's memory, {}", one.text());
        let lp = plan(&["run", "q"], &two).unwrap();
        let sp = lp.spread.as_ref().expect("in stages");
        assert!(lp.experts.is_none() && sp.host_bytes == 0 && sp.places.is_empty() && sp.devices == [0, 1] && sp.cuts.len() == 1, "{}", lp.text());
        assert_eq!(lp.weights_on_gpu(), lp.weights);
        stages_within_rooms(&lp, &c, &two, crate::weights::Layout::Int6);
        // (a GPU that holds it whole: one GPU's plan, as before)
        let big = [t(0, 40 << 30), t(1, 40 << 30)];
        assert!(plan(&["run", "q"], &big).unwrap().spread.is_none());
        // (four GPUs of 5 GiB: not every expert; in the computer's memory, places a stage)
        let four = [t(0, 5 << 30), t(1, 5 << 30), t(2, 5 << 30), t(3, 5 << 30)];
        let lp = plan(&["run", "q"], &four).unwrap();
        assert!(lp.spread.is_some() && lp.experts.is_some(), "{}", lp.text());
        stages_within_rooms(&lp, &c, &four, crate::weights::Layout::Int6);
        // (--split: where it says, both stages on one GPU where it is named twice)
        let chosen = plan(&["run", "q", "--split", "20", "--devices", "0,0"], &big).unwrap();
        let sp = chosen.spread.as_ref().unwrap();
        assert!(sp.cuts == [20] && sp.devices == [0, 0] && chosen.users == 1 && chosen.experts.is_none(), "{}", chosen.text());
        // (two stages on one GPU: two engines' work, one GPU's reserve and free memory)
        assert_eq!((chosen.work, chosen.reserve, chosen.free), (2 * LOSSY_WORK, reserve_of(40 << 30), 40 << 30));
        let bad = |w: &[&str], devs: &[Target], said: &str| {
            let r = plan(w, devs).unwrap_err();
            assert!(r.what.contains(said), "{w:?}: {r}");
        };
        bad(&["run", "q", "--split", "20,10", "--devices", "0,0,0"], &big, "ascending cuts inside the layers");
        bad(&["run", "q", "--split", "40", "--devices", "0,1"], &big, "ascending cuts inside the layers");
        bad(&["run", "q", "--split", "20", "--devices", "0,7"], &big, "there is no GPU 7 here");
        bad(&["run", "q", "--split", "36", "--devices", "0,1"], &two[..], "GPU 0 takes about");
        let g = mc(&fixture("gemma4_12b_config.json"));
        let r = sizing::lossy_plan_over(&args(&["run", "g", "--split", "4"]), "g", crate::weights::Layout::Int6, &g, None, 40 << 30, "G", None, None, &big)
            .unwrap_err();
        assert!(r.what.contains("loads on one GPU at kestrel"), "{r}");
    }

    /// A check that runs the 12-bit pages where a server's plan is bf16's asks for the plan made for them (`sizing::twelve_always`): one user at 32,768 tokens of
    /// gemma-4-12B on an L4 is planned in bf16 for a server (the pages' worst case beside the tokens reserves more), and in 12-bit pages of more bytes for the
    /// check, which the loader's pools then take. Without the pages asked for (no GLYD_KV=lossless) the plan stays bf16's.
    #[test]
    fn a_check_asks_for_the_twelve_bit_plan_where_a_servers_is_bf16() {
        let text = fixture("gemma4_12b_config.json");
        let dir = model_dir("kvalways", "gemma-4-12B", &text);
        let m = dir.to_str().unwrap();
        let words = ["serve", m, "--machine", "l4", "--users", "1", "--no-draft", "--context", "32768"];
        let plan = |lossless: bool, always: bool| -> LoadPlan {
            sizing::always_twelve(always, || sizing::asking_lossless(lossless, || planned(&args(&words))))
                .unwrap()
                .plan
                .chosen
                .and_then(|c| c.load)
                .expect("a plan")
        };
        let (server, check, plain) = (plan(true, false), plan(true, true), plan(false, true));
        assert_eq!((server.kv_format(), server.context), ("bf16", 32_768), "{server:?}");
        assert_eq!((check.kv_format(), check.context, check.level), ("12-bit", 32_768, server.level), "{check:?}");
        assert!(check.kv > server.kv, "the 12-bit pages of one user reserve more: {} against {}", check.kv, server.kv);
        assert_eq!(plain, plan(false, false), "no pages asked for: bf16's plan whatever the check's switch");
        assert_eq!(plain.kv_format(), "bf16");
    }

    fn fixture(name: &str) -> String {
        std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)).unwrap()
    }

    fn qwen35(name: &str) -> Shape {
        let all = json::parse(&fixture("qwen35_configs.json")).unwrap();
        let c = all.get(name).unwrap();
        Shape::from_config(&json::to_compact(c)).unwrap()
    }

    /// The swap area (R4d) is twice the GPU cache it backs, no more than the computer has free less its margin (what the expert store's host tier pinned is not free), and none
    /// where that margin is all there is.
    #[test]
    fn the_swap_area_is_twice_the_cache_within_what_the_host_left() {
        let gib = 1u64 << 30;
        assert_eq!(swap_area(4 * gib, Some(64 * gib)), 8 * gib);
        assert_eq!(swap_area(4 * gib, Some(10 * gib)), 4 * gib, "10 GiB free, 6 kept");
        // (32 GiB free before 20 GiB of experts were pinned: 12 left, 6 kept)
        assert_eq!(swap_area(8 * gib, Some(12 * gib)), 6 * gib);
        assert_eq!(swap_area(4 * gib, Some(5 * gib)), 0);
        assert_eq!(swap_area(3 * gib, None), 6 * gib);
    }

    /// The plain dense kind, large: hidden 5,120, 64 layers of 64 query heads over 8 KV heads of 128, MLP 25,600, a vocabulary of 151,936 (the kind's
    /// architecture key, as the planner tells it apart).
    const DENSE_64L: &str = r#"{"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 5120, "intermediate_size": 25600, "num_hidden_layers": 64, "num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 151936, "tie_word_embeddings": false, "max_position_embeddings": 40960}"#;

    /// The routed experts' bits as released apart from the rest's (Kimi-K3's MXFP4 in its text config's compressed-tensors, DeepSeek-V4's
    /// `expert_dtype`), Kimi-K3's and Muse Spark's names for the experts' width and Kimi-K3's for the choices a token; and hummingbird offered
    /// where its expert bits are below the release's, with every other part as at penguin, and not where the experts are already at its bits.
    #[test]
    fn experts_have_their_own_release_bits_and_hummingbird_applies_below_them() {
        let kimi = r#"{"architectures": ["KimiK3ForConditionalGeneration"], "text_config": {"model_type": "kimi_linear", "hidden_size": 7168, "num_hidden_layers": 4,
            "num_attention_heads": 96, "vocab_size": 163840, "num_experts": 896, "num_experts_per_token": 16, "moe_intermediate_size": 3072, "routed_expert_hidden_size": 3584,
            "first_k_dense_replace": 1, "intermediate_size": 33792, "quantization_config": {"quant_method": "compressed-tensors", "format": "mxfp4-pack-quantized"}}}"#;
        let k = Shape::from_config(kimi).unwrap();
        assert_eq!((k.topk, k.expert_width, k.release_bits, k.expert_bits_released()), (16, 3584, 16.0, 4.25));
        assert!((k.routed - 3.0 * 896.0 * 3.0 * 3072.0 * 3584.0).abs() < 1.0, "{}", k.routed);
        let dsv4 = r#"{"architectures": ["DeepseekV4ForCausalLM"], "model_type": "deepseek_v4", "hidden_size": 4096, "num_hidden_layers": 4, "num_attention_heads": 64,
            "n_routed_experts": 256, "num_experts_per_tok": 6, "moe_intermediate_size": 2048, "expert_dtype": "fp4", "quantization_config": {"quant_method": "fp8"}}"#;
        let d = Shape::from_config(dsv4).unwrap();
        assert_eq!((d.release_bits, d.expert_bits_released(), d.expert_width), (8.0, 4.25, 4096));
        // (V4.1 names the experts' dtype in its quantization_config: the same bits)
        let dsv41 = dsv4.replace(
            r#""expert_dtype": "fp4", "quantization_config": {"quant_method": "fp8"}"#,
            r#""quantization_config": {"quant_method": "fp8", "expert_dtype": "fp4"}"#,
        );
        assert_ne!(dsv41, dsv4);
        assert_eq!(Shape::from_config(&dsv41).unwrap().expert_bits_released(), 4.25);
        let glm = r#"{"model_type": "glm5_next_text", "hidden_size": 4096, "num_hidden_layers": 4, "num_attention_heads": 64, "n_routed_experts": 288,
            "num_experts_per_tok": 8, "moe_intermediate_size": 2048, "quantization_config": {"quant_method": "fp8"}}"#;
        let g = Shape::from_config(glm).unwrap();
        assert_eq!((g.release_bits, g.expert_bits_released()), (8.0, 8.0));
        // the release's bytes: each part at its bits
        assert!((k.release_bytes() - ((k.embed + k.head + k.dense) * 2.0 + k.routed * 4.25 / 8.0)).abs() < 1.0);
        // hummingbird's expert bits by width: 2.99 at 1,024, 2.92 at 3,584 and 4,096
        assert!((hummingbird_expert_bits(1024) - (0.4 * 3.625 + 0.6 * 2.5625)).abs() < 1e-9);
        assert!((2.91..2.93).contains(&hummingbird_expert_bits(4096)) && (2.91..2.93).contains(&hummingbird_expert_bits(3584)));
        let h100 = Machine::preset("h100").unwrap();
        let dev = &h100.devices[0];
        for (s, rest) in [(&k, 12.04), (&d, 8.0), (&g, 8.0)] {
            let (b, e) = level_bits("hummingbird", s, dev, false).unwrap();
            assert_eq!(b, rest);
            assert!((e - hummingbird_expert_bits(s.expert_width)).abs() < 1e-12);
            assert_eq!(level_bits("penguin", s, dev, false).unwrap().1, s.expert_bits_released().min(12.04));
        }
        // the uniform levels are not offered where they would raise the experts' bits (MXFP4 at 4.25 against kestrel's 6.5)
        assert!(level_bits("kestrel", &k, dev, false).is_none() && level_bits("wren", &d, dev, false).is_none());
        // experts released at hummingbird's bits or fewer: nothing to gain
        let low = Shape { expert_release_bits: 2.5, ..d.clone() };
        assert!(level_bits("hummingbird", &low, dev, false).is_none());
        // the one-GPU route's hummingbird (its row's: the Nemotron-H family's): experts on a latent of 1,024 (a power of two from 128 to 1,024, the kernels') take it,
        // experts on a width of 2,688 (no latent) do not; the same shape on the dynamic route's is the described families' rule
        let nh = Shape { arch: "NemotronHForCausalLM".into(), expert_width: 1024, release_bits: 16.0, expert_release_bits: 0.0, ..d.clone() };
        assert!(crate::support::row(crate::support::Family::NemotronH).hummingbird_one_gpu());
        assert_eq!(level_bits("hummingbird", &nh, dev, true), Some((6.0, 2.9)));
        assert!(level_bits("hummingbird", &Shape { expert_width: 2688, ..nh.clone() }, dev, true).is_none());
        assert_ne!(level_bits("hummingbird", &nh, dev, false), Some((6.0, 2.9)));
    }

    /// A described family at hummingbird on a consumer GPU (GLM-5.3-Flash's shape, 288 experts of 4,096 x 2,048 on 45 layers): its experts in RAM with a cache of the busiest on the GPU where the computer's memory holds
    /// them, the rest on the disk where it does not (the disk's share named), refused where the memory holds less than a token's experts in every layer; the cache's floor a token's experts, not a twentieth of them.
    #[test]
    fn hummingbird_puts_a_described_familys_experts_in_ram_and_on_the_disk() {
        let glm = r#"{"model_type": "glm5_next_text", "hidden_size": 4096, "num_hidden_layers": 45, "num_attention_heads": 64, "n_routed_experts": 288, "num_experts_per_tok": 8,
            "moe_intermediate_size": 2048, "first_k_dense_replace": 3, "vocab_size": 154880, "quantization_config": {"quant_method": "fp8"}}"#;
        let sh = Shape::from_config(glm).unwrap();
        let fam = crate::support::row(crate::support::Family::Glm);
        let ask = Ask { level: Some("hummingbird"), family: Some(fam), ..Default::default() };
        let mut m = Machine::preset("rtx4080s").unwrap();
        let experts = sh.routed * hummingbird_expert_bits(sh.expert_width) / 8.0;
        // memory for all of them: experts in RAM, nothing on the disk
        m.ram_free = (experts * 1.3) as u64 + (8 << 30);
        let p = plan(&sh, &m, ask);
        let c = chosen(&p);
        assert!(matches!(c.placement, Some(Placement::ExpertsInRam { disk: 0, .. })) && c.level == "hummingbird", "{p:?}");
        // half of them: the rest on the disk, and slower
        m.ram_free = (experts * 0.5) as u64;
        let q = plan(&sh, &m, ask);
        let Some(Placement::ExpertsInRam { disk, .. }) = chosen(&q).placement else { panic!("{q:?}") };
        assert!(disk > 0 && (disk as f64) < experts * 0.6 && chosen(&q).tok_s < c.tok_s, "{q:?}");
        assert!(text("GLM-5.3-Flash", &sh, &m, &q).contains("on the disk"));
        // less than a token's experts in every layer: refused
        m.ram_free = (sh.routed_active() * hummingbird_expert_bits(sh.expert_width) / 8.0 * 0.5) as u64 + (4 << 30);
        assert!(plan(&sh, &m, ask).chosen.is_none());
        // the penguin level of the same family keeps its experts on the GPUs (its row has no RAM there)
        m.ram_free = 512 << 30;
        let pg = plan(&sh, &m, Ask { level: Some("penguin"), family: Some(fam), ..Default::default() });
        assert!(!pg.options.iter().any(|o| matches!(o.placement, Some(Placement::ExpertsInRam { .. }))), "{pg:?}");
    }

    /// R12: DeepSeek-V4 at penguin keeps its FP4 experts in RAM with a cache of the busiest on the GPU where the GPU does not hold them all and the computer's memory does; with less memory than its experts it is refused at
    /// penguin (the disk tier is the hummingbird level's) and runs at hummingbird with the rest on the disk.
    #[test]
    fn deepseek_v4_keeps_its_fp4_experts_in_ram_at_penguin() {
        let v4 = r#"{"architectures": ["DeepseekV4ForCausalLM"], "model_type": "deepseek_v4", "hidden_size": 4096, "num_hidden_layers": 43, "num_attention_heads": 64,
            "n_routed_experts": 256, "num_experts_per_tok": 6, "moe_intermediate_size": 2048, "vocab_size": 129280, "expert_dtype": "fp4", "quantization_config": {"quant_method": "fp8"}}"#;
        let sh = Shape::from_config(v4).unwrap();
        let fam = crate::support::row(crate::support::Family::DeepSeekV4);
        let mut m = Machine::preset("rtx4090").unwrap();
        m.ram_free = 256 << 30;
        let p = plan(&sh, &m, Ask { level: Some("penguin"), family: Some(fam), ..Default::default() });
        assert!(matches!(chosen(&p).placement, Some(Placement::ExpertsInRam { disk: 0, .. })), "{p:?}");
        m.ram_free = 96 << 30;
        assert!(plan(&sh, &m, Ask { level: Some("penguin"), family: Some(fam), ..Default::default() }).chosen.is_none());
        let hb = plan(&sh, &m, Ask { level: Some("hummingbird"), family: Some(fam), ..Default::default() });
        assert!(matches!(chosen(&hb).placement, Some(Placement::ExpertsInRam { disk, .. }) if disk > 0), "{hb:?}");
    }

    /// R12: GLM-5.3-Flash's FP8 experts (304 GiB) in the computer's memory at penguin where it has room for them, the rest on a 24 GB GPU; with less memory, penguin is not planned (its hummingbird level is,
    /// with experts on the disk too).
    #[test]
    fn glm_keeps_its_fp8_experts_in_ram_at_penguin() {
        let sh = Shape::from_config(include_str!("../tests/fixtures/glm5_next_config.json")).unwrap();
        let fam = crate::support::row(crate::support::Family::Glm);
        let mut m = Machine::preset("rtx4090").unwrap();
        m.ram_free = 512 << 30;
        let p = plan(&sh, &m, Ask { level: Some("penguin"), family: Some(fam), ..Default::default() });
        assert!(matches!(chosen(&p).placement, Some(Placement::ExpertsInRam { disk: 0, .. })), "{p:?}");
        m.ram_free = 128 << 30;
        assert!(plan(&sh, &m, Ask { level: Some("penguin"), family: Some(fam), ..Default::default() }).chosen.is_none());
    }

    /// A config.json that says a number of layers no model has was an allocation of that many entries (a panic, or the process killed for its memory), and is an error.
    #[test]
    fn a_config_with_a_billion_layers_is_an_error() {
        let c = |layers: &str| DENSE_64L.replace("\"num_hidden_layers\": 64", &format!("\"num_hidden_layers\": {layers}"));
        for layers in ["1000000000", "4000000000000000000", "277777777777777777777777777777777777777777777777777"] {
            assert!(Shape::from_config(&c(layers)).unwrap_err().contains("num_hidden_layers"), "{layers}");
        }
        assert_eq!(Shape::from_config(&c("10000")).unwrap().layers, 10_000);
    }

    /// Every family's parser refuses a layer count no model has (each layer is an entry of a list the parser makes: a billion of them was an allocation that took the machine's memory, or a loop of minutes), and a Qwen4x
    /// config whose `full_attention_interval` is 0 (a remainder by it) is read as an interval of 1, as Qwen3.5's is.
    #[test]
    fn every_family_refuses_a_billion_layers() {
        fn with_layers(v: &mut json::Value, layers: u64) {
            if let json::Value::Object(o) = v {
                o.retain(|(k, _)| k != "layer_types" && k != "hybrid_override_pattern");
                for (k, x) in o.iter_mut() {
                    if k == "num_hidden_layers" {
                        *x = json::Value::from(layers);
                    } else {
                        with_layers(x, layers);
                    }
                }
            }
        }
        let big = |text: &str| {
            let mut v = json::parse(text).unwrap();
            with_layers(&mut v, 1_000_000_000);
            json::to_compact(&v)
        };
        let start = std::time::Instant::now();
        let qx = big(&fixture("qwen4x_config.json"));
        assert!(crate::qwen4x::ConfigQx::parse(&qx, "c", None).unwrap_err().to_string().contains("num_hidden_layers"));
        let glm = big(&fixture("glm5_next_tiny_config.json"));
        assert!(crate::glm::ConfigGlm::parse(&glm, "c", None).unwrap_err().to_string().contains("num_hidden_layers"));
        let q35 = big(&json::to_compact(json::parse(&fixture("qwen35_configs.json")).unwrap().get("Qwen/Qwen3.5-9B").unwrap()));
        assert!(crate::qwen35::Config35::parse(&q35, "c", None).unwrap_err().to_string().contains("num_hidden_layers"));
        let dense = r#"{"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3","hidden_size":1024,"num_hidden_layers":1000000000,"num_attention_heads":16,"num_key_value_heads":8,"intermediate_size":3072,"vocab_size":151936,"rms_norm_eps":1e-06,"rope_theta":1000000.0,"max_position_embeddings":40960}"#;
        assert!(crate::weights::Config::parse(dense, "c", None).unwrap_err().to_string().contains("num_hidden_layers"));
        assert!(start.elapsed() < std::time::Duration::from_secs(10), "{:?}", start.elapsed());
        // an interval of 0 in a Qwen4x config without layer_types
        let mut v = json::parse(&fixture("qwen4x_config.json")).unwrap();
        fn no_types(v: &mut json::Value) {
            if let json::Value::Object(o) = v {
                o.retain(|(k, _)| k != "layer_types");
                if let Some(t) = o.iter_mut().find(|(k, _)| k == "text_config") {
                    no_types(&mut t.1);
                    if let json::Value::Object(t) = &mut t.1 {
                        t.retain(|(k, _)| k != "full_attention_interval");
                        t.push(("full_attention_interval".to_string(), json::Value::from(0u64)));
                    }
                }
            }
        }
        no_types(&mut v);
        let _ = crate::qwen4x::ConfigQx::parse(&json::to_compact(&v), "c", None);
    }

    fn chosen(p: &Plan) -> &Candidate {
        p.chosen.as_ref().expect("a plan")
    }

    #[test]
    fn the_shapes_add_up_to_the_models_sizes() {
        // (sizes as the releases give them: the parameter count in the name, or the bf16 matrices gpu/README.md measured)
        let within = |got: f64, want: f64, tol: f64| assert!((got / want - 1.0).abs() < tol, "{got:.3e} vs {want:.3e}");
        within(Shape::from_config(DENSE_64L).unwrap().params(), 32.8e9, 0.04);
        within(qwen35("Qwen/Qwen3.5-9B").params(), 9.0e9, 0.12);
        within(qwen35("Qwen/Qwen3.8-27B").params(), 26.9e9, 0.12);
        let g = Shape::from_config(&fixture("gemma4_12b_config.json")).unwrap();
        within(g.params(), 12.0e9, 0.15);
        assert!(g.kv_sliding > 0.0 && g.window == 1024 && g.kv_full > 0.0);
        let qx = Shape::from_config(&fixture("qwen4x_config.json")).unwrap();
        assert!(qx.is_moe() && qx.experts == 512 && qx.topk == 10 && qx.mtp && qx.state > 0.0);
        // (a mixture of experts' active weights: the rest, and its routed experts' top-k share)
        within(qx.routed_active(), qx.routed * qx.topk as f64 / qx.experts as f64, 1e-9);
        // scaled to an index's total, every part keeps its share
        let s = Shape::from_config(DENSE_64L).unwrap();
        let t = s.clone().scaled_to(2.0 * s.params() * 1.1);
        within(t.params(), s.params() * 1.1, 1e-9);
    }

    #[test]
    fn predictions_match_what_was_measured() {
        // tokens/s measured with one user and a short prompt (CHANGELOG; starlings/benchmarks; condor/README.md)
        let within = |p: &Plan, want: f64, tol: f64| {
            let got = chosen(p).tok_s;
            assert!((got / want - 1.0).abs() < tol, "{got:.1} tok/s, measured {want}");
        };
        let m4080 = Machine::preset("rtx4080s").unwrap();
        let m4090 = Machine::preset("rtx4090").unwrap();
        // Gemma 4 12B at wren (Google's QAT weights, 4.5 bits) on an RTX 4080 SUPER: 82.5
        let g = Shape::from_config(&fixture("gemma4_12b_config.json")).unwrap();
        within(&plan(&g, &m4080, Ask { level: Some("wren"), ..Default::default() }), 82.5, 0.25);
        // Qwen3.8-27B at swift on an RTX 4090: 46.0 (the model has a draft head; the measurement ran without drafting)
        let mut q = qwen35("Qwen/Qwen3.8-27B");
        q.mtp = false;
        within(&plan(&q, &m4090, Ask { level: Some("swift"), ..Default::default() }), 46.0, 0.25);
        // Qwen3.8-Flash-Next at hummingbird on an RTX 4080 SUPER with 64 GB of RAM, experts in RAM, drafting: 91 (precise mode)
        let qx = Shape::from_config(&fixture("qwen4x_config.json")).unwrap();
        let p = plan(&qx, &m4080, Ask { level: Some("hummingbird"), ..Default::default() });
        assert!(matches!(chosen(&p).placement, Some(Placement::ExpertsInRam { .. })), "{p:?}");
        within(&p, 91.0, 0.30);
        // (the large dense shape lossless on an RTX A6000, the tiered layout there: its weights read at the class's lossless share and its keys and values at
        // theirs, and its floor; a release's own measurement on this class is a benchmark's, phase 3)
        let (big, a6000) = (Shape::from_config(DENSE_64L).unwrap(), Machine::preset("a6000").unwrap());
        let (c, d) = (chosen(&plan(&big, &a6000, Ask::default())).clone(), &a6000.devices[0]);
        let (bits, ebits) = level_bits("penguin", &big, d, false).unwrap();
        let (w, kv) = reads_of(&big, bits, ebits, POSITIONS, 1.0);
        let want = 1.0 / ((w * d.consts.wl + kv * d.consts.kv) / d.rate() + d.consts.floor_s(big.layers as f64, big.hidden as f64, 1.0));
        assert!(c.level == "penguin" && (c.prior / want - 1.0).abs() < 1e-9, "{} at {}: {} against {want}", c.level, c.bits, c.prior);
    }

    /// P2c: decode and prefill predicted from the rates of an RTX 4080 SUPER (the preset's stream is the one `glyd-engine rates` measured there, 697 GB/s) against
    /// what `glyd-engine bench --loads 1k:1 --no-draft` measured on one (2026-11-23, three runs each, the median): gemma-4-12B at wren 86.2 tokens/s and kestrel 63.3,
    /// prompts of 936 tokens at 3,570 and 3,580; Qwen3.5-0.8B at kestrel 561.9 and wren 657.9, prompts of 1,002 tokens at 27,073 and 27,372. Each within the fit's own error (below), with
    /// the shapes `fit` reads (the checkpoints' own headers) and the family's path (Qwen3.5's does not draft with its head); the measurement in the band.
    /// FITTED, IN-SAMPLE: `EFF_LOSSY`, `LAYER_S`, `PROMPT_EFF` and `PROMPT_LAYER_S` were set from these very runs, so this pins the constants (`FIT_DECODE`,
    /// `FIT_PREFILL`: a little over the fit's own error on them) and is no evidence that they predict other runs. The held-out measurement (other levels, 4k and
    /// 8k positions, other prompt lengths, each predicted before it is measured) is scripts/planner_heldout_check.sh on the 4080 SUPER, not run yet: the 15% the
    /// planner states for a prediction is that run's to hold.
    /// The fits' own errors on the runs they were fitted on, each a little over the largest: the 4080 SUPER's four runs are within 0.6% (decode) and 0.7%
    /// (prompts) of their figures, the A100's eight within 9.5% (seven of them within 4.7%). A change to a constant that moves a fitted point past these is seen.
    const FIT_DECODE: f64 = 0.02;
    const FIT_PREFILL: f64 = 0.02;
    const FIT_A100: f64 = 0.11;

    #[test]
    fn the_fit_on_the_4080_supers_runs_is_pinned() {
        let gemma: serde_json::Value = serde_json::from_str(&fixture("headers/gemma-4-12B-it.json")).unwrap();
        let groups: Vec<crate::sizes::Group> = gemma["groups"]
            .as_array()
            .unwrap()
            .iter()
            .map(|g| crate::sizes::Group {
                mask: g["mask"].as_str().unwrap().to_string(),
                dtype: g["dtype"].as_str().unwrap().to_string(),
                shape: g["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect(),
                bytes: g["bytes"].as_u64().unwrap(),
                layers: g["layers"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect(),
                experts: g["experts"].as_u64().unwrap() as usize,
            })
            .collect();
        let gemma_text = gemma["config"].as_str().unwrap().to_string();
        let gemma_sh = Shape::from_config(&gemma_text).unwrap().with_table(&crate::sizes::Table::expand(&groups));
        let q35 = json::parse(&fixture("qwen35_configs.json")).unwrap();
        let q08_text = json::to_compact(q35.get("Qwen/Qwen3.5-0.8B").unwrap());
        let q08_sh = Shape::from_config(&q08_text)
            .unwrap()
            .with_table(&table_of(json::parse(&fixture("qwen35_tensors.json")).unwrap().get("Qwen/Qwen3.5-0.8B").unwrap()));
        let m = Machine::preset("rtx4080s").unwrap();
        assert!((m.devices[0].rate() / GB - 697.35).abs() < 0.5, "the preset's stream is the one measured");
        for (sh, text, level, decode, prefill) in [
            (&gemma_sh, &gemma_text, "wren", 86.2, 3570.3),
            (&gemma_sh, &gemma_text, "kestrel", 63.3, 3579.7),
            (&q08_sh, &q08_text, "kestrel", 561.9, 27073.4),
            (&q08_sh, &q08_text, "wren", 657.9, 27372.1),
        ] {
            let fam = crate::support::row(crate::support::family(text, "m").unwrap());
            let c = chosen(&plan(sh, &m, Ask { level: Some(level), family: Some(fam), ..Default::default() })).clone();
            let p = c.prefill.unwrap();
            eprintln!(
                "{level}: decode {:.1} against {decode} ({:+.1}%), prefill {p:.0} against {prefill} ({:+.1}%)",
                c.tok_s,
                100.0 * (c.tok_s / decode - 1.0),
                100.0 * (p / prefill - 1.0)
            );
            assert!((c.tok_s / decode - 1.0).abs() < FIT_DECODE, "{level}: {:.1} tokens/s predicted, {decode} measured", c.tok_s);
            assert!(c.range.0 <= decode && decode <= c.range.1, "{level}: {decode} outside {:?}", c.range);
            assert!((p / prefill - 1.0).abs() < FIT_PREFILL, "{level}: prefill {p:.0} predicted, {prefill} measured");
        }
        // (a GPU whose tensor rate the planner does not know has no prefill prediction)
        let mut odd = m.clone();
        odd.devices[0].tflops = 0.0;
        assert!(chosen(&plan(&gemma_sh, &odd, Ask { level: Some("wren"), ..Default::default() })).prefill.is_none());
    }

    /// `one_gpu_plan` in the layout the run loaded: at penguin the 12-bit one but on Ada (`best_layout`, as the A100 and the A10 loaded it).
    fn loaded(kind: Kind, level: &'static str, cc: [i32; 2]) -> LoadPlan {
        let mut p = one_gpu_plan(kind, level);
        if level == "penguin" && !crate::weights::smallest_is_fastest((cc[0], cc[1])) {
            p.layout = crate::weights::Layout::Twelve;
        }
        p
    }

    /// A plan of `level` with every weight on one GPU, as the calibration records describe one (`features` reads its kind, level and placement alone).
    fn one_gpu_plan(kind: Kind, level: &'static str) -> LoadPlan {
        LoadPlan {
            kind,
            family: "",
            level,
            layout: crate::levels::layout_of(level).unwrap_or(crate::weights::Layout::Tiered),
            context: 32768,
            users: 1,
            chunk: 2048,
            weights: 0,
            bf16: 0,
            kv: 0,
            kv12: false,
            work: 0,
            draft: 0,
            reserve: 0,
            free: 0,
            gpu: String::new(),
            experts: None,
            spread: None,
        }
    }

    /// A recorded header's compact table of tensors (tests/fixtures/headers).
    fn header_groups(v: &serde_json::Value) -> Vec<crate::sizes::Group> {
        v["groups"]
            .as_array()
            .unwrap()
            .iter()
            .map(|g| crate::sizes::Group {
                mask: g["mask"].as_str().unwrap().to_string(),
                dtype: g["dtype"].as_str().unwrap().to_string(),
                shape: g["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect(),
                bytes: g["bytes"].as_u64().unwrap(),
                layers: g["layers"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect(),
                experts: g["experts"].as_u64().unwrap() as usize,
            })
            .collect()
    }

    /// The header-table shapes of gemma-4-12B and Qwen3.5-0.8B (the checkpoints' own, as `fit` reads them).
    fn measured_shapes() -> (Shape, Shape) {
        let gemma: serde_json::Value = serde_json::from_str(&fixture("headers/gemma-4-12B-it.json")).unwrap();
        let gemma_sh = Shape::from_config(gemma["config"].as_str().unwrap()).unwrap().with_table(&crate::sizes::Table::expand(&header_groups(&gemma)));
        let q35 = json::parse(&fixture("qwen35_configs.json")).unwrap();
        let q08_sh = Shape::from_config(&json::to_compact(q35.get("Qwen/Qwen3.5-0.8B").unwrap()))
            .unwrap()
            .with_table(&table_of(json::parse(&fixture("qwen35_tensors.json")).unwrap().get("Qwen/Qwen3.5-0.8B").unwrap()));
        (gemma_sh, q08_sh)
    }

    /// The shipped calibration records (calibration/records.jsonl), with the features the planner reads of them: the RTX 4080 SUPER's runs (two of
    /// 2026-11-23, bench --loads 1k:1 --no-draft, the median of 3; prompts of 936 and 1,002 tokens, 128 made) and the A100 SXM4 40GB's runs at 63bdcdcb
    /// (2026-11-24, bench's gate suite with GLYD_KV=bf16, the median of 3, 128 made: one user and eight on short prompts, one at 1k and 4k; the
    /// prompts of the short loads too short to time a prompt, so their prompt rates are not kept; the rate the A100's probe measured), and the A10's
    /// plancheck runs (15: 8 users, Qwen3.5-0.8B kestrel, gemma-4-12B wren and swift at 1k, 4k, 8k, 1k for eight and 85% of the context). Only
    /// releases of the last six months: the runs of older releases measured beside them are not kept. `GLYD_WRITE_CALIB=1` writes the file anew; the other machines'
    /// `plancheck` records are added to it.
    #[test]
    fn the_shipped_records_are_the_measured_runs() {
        let (gemma_sh, q08_sh) = measured_shapes();
        let mut lines = String::new();
        let (g, q) = ("google/gemma-4-12B-it", "Qwen/Qwen3.5-0.8B");
        type Run<'a> = (&'a str, &'a Shape, Kind, &'static str, &'a str, f64, usize, f64, Option<f64>);
        let a100: Vec<Run> = vec![
            (g, &gemma_sh, Kind::Gemma4, "wren", "short:1", 38.0, 1, 86.972518, None),
            (g, &gemma_sh, Kind::Gemma4, "wren", "short:8", 37.5, 8, 48.504754, None),
            (g, &gemma_sh, Kind::Gemma4, "wren", "1k:1", 935.0, 1, 84.771883, Some(5070.099763)),
            (g, &gemma_sh, Kind::Gemma4, "wren", "4k:1", 3753.0, 1, 83.624592, Some(4436.882222)),
            (q, &q08_sh, Kind::Qwen35, "kestrel", "short:1", 37.0, 1, 454.88434, None),
            (q, &q08_sh, Kind::Qwen35, "kestrel", "short:8", 36.0, 8, 189.34745, None),
            (q, &q08_sh, Kind::Qwen35, "kestrel", "1k:1", 1005.0, 1, 451.127571, Some(20317.583028)),
            (q, &q08_sh, Kind::Qwen35, "kestrel", "4k:1", 4012.0, 1, 439.503543, Some(19963.700121)),
        ];
        let a10: Vec<Run> = vec![
            (q, &q08_sh, Kind::Qwen35, "kestrel", "1k:1", 995.0, 1, 399.443361, Some(17034.02874)),
            (q, &q08_sh, Kind::Qwen35, "kestrel", "4k:1", 4010.0, 1, 385.092413, Some(17844.658907)),
            (q, &q08_sh, Kind::Qwen35, "kestrel", "8k:1", 8005.0, 1, 369.578705, Some(16368.974123)),
            (q, &q08_sh, Kind::Qwen35, "kestrel", "1k:8", 994.5, 8, 152.281159, Some(17351.977334)),
            (q, &q08_sh, Kind::Qwen35, "kestrel", "27852t:1", 27220.0, 1, 300.336779, Some(11598.584925)),
            (g, &gemma_sh, Kind::Gemma4, "wren", "1k:1", 935.0, 1, 57.028112, Some(2178.736128)),
            (g, &gemma_sh, Kind::Gemma4, "wren", "4k:1", 3754.0, 1, 56.254122, Some(2060.675388)),
            (g, &gemma_sh, Kind::Gemma4, "wren", "8k:1", 7511.0, 1, 55.140277, Some(1878.440661)),
            (g, &gemma_sh, Kind::Gemma4, "wren", "1k:8", 935.0, 8, 23.383665, Some(2096.645192)),
            (g, &gemma_sh, Kind::Gemma4, "wren", "27852t:1", 25548.0, 1, 51.465418, Some(1334.308805)),
            (g, &gemma_sh, Kind::Gemma4, "swift", "1k:1", 935.0, 1, 50.202928, Some(2182.203677)),
            (g, &gemma_sh, Kind::Gemma4, "swift", "4k:1", 3754.0, 1, 49.525424, Some(2055.072626)),
            (g, &gemma_sh, Kind::Gemma4, "swift", "8k:1", 7511.0, 1, 48.866016, Some(1883.56461)),
            (g, &gemma_sh, Kind::Gemma4, "swift", "1k:8", 935.0, 8, 22.060678, Some(2095.481739)),
            (g, &gemma_sh, Kind::Gemma4, "swift", "27852t:1", 25548.0, 1, 46.112267, Some(1340.026958)),
        ];
        for (at, gpu, preset, class, cc, sms, stream, ram, tflops, source, runs) in [
            (
                "2026-11-23",
                "NVIDIA GeForce RTX 4080 SUPER",
                "rtx4080s",
                "sm89",
                [8, 9],
                80,
                697.35,
                73.91,
                104.5,
                "glyd-pc: bench --loads 1k:1 --no-draft, the median of 3",
                vec![
                    (g, &gemma_sh, Kind::Gemma4, "wren", "1k:1", 936.0, 1usize, 86.167488, Some(3570.250176)),
                    (g, &gemma_sh, Kind::Gemma4, "kestrel", "1k:1", 936.0, 1, 63.290554, Some(3579.659425)),
                    (q, &q08_sh, Kind::Qwen35, "kestrel", "1k:1", 1002.0, 1, 561.9, Some(27073.4)),
                    (q, &q08_sh, Kind::Qwen35, "wren", "1k:1", 1002.0, 1, 657.9, Some(27372.1)),
                ],
            ),
            (
                "2026-11-24",
                "NVIDIA A100-SXM4-40GB",
                "a100-80g",
                "sm80",
                [8, 0],
                108,
                1274.41,
                61.49,
                312.0,
                "Lambda A100 SXM4 40GB: bench at 63bdcdcb, gate suite, GLYD_KV=bf16, the median of 3",
                a100,
            ),
            (
                "2026-11-24",
                "NVIDIA A10",
                "a6000",
                "sm86",
                [8, 6],
                72,
                518.53,
                116.68,
                125.0,
                "Lambda A10: plancheck at 87023bbc, 8 users, the median of 3",
                a10,
            ),
        ] {
            let m = Machine::preset(preset).unwrap();
            for (model, sh, kind, level, load, prompt, users, decode, prefill) in runs {
                let positions = prompt + 64.0;
                let r = serde_json::json!({
                    "schema": CALIB_SCHEMA, "at": at, "source": source, "gpu": gpu, "cc": cc, "class": class, "driver": null, "sms": sms,
                    "stream_gbs": stream, "measured_rate": true, "ram_gbs": ram, "tflops": tflops, "model": model, "level": level, "load": load, "users": users,
                    "positions": positions, "features": features(sh, &m, 0, &loaded(kind, level, cc), positions, users),
                    "made": 128, "predicted": null, "measured": {"decode": decode, "prefill": prefill, "prompt_tokens": prompt},
                });
                lines += &format!("{r}\n");
            }
        }
        let file = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("calibration/records.jsonl");
        if std::env::var_os("GLYD_WRITE_CALIB").is_some() {
            std::fs::write(&file, &lines).unwrap();
            return;
        }
        let shipped = records_of(CALIBRATION);
        for r in &records_of(&lines) {
            assert!(shipped.contains(r), "{r}");
        }
    }

    /// The share a record's features say its weights were read at.
    fn read_in(f: &serde_json::Value) -> Read {
        match (f["lossless"] == true, f["twelve"] == true) {
            (true, true) => Read::Twelve,
            (true, false) => Read::Tiered,
            _ => Read::Packs,
        }
    }

    /// Records of a class made at figures `c` from the features of real plans (gemma-4-12B, Qwen3.5-0.8B and 9B at two lossy levels and lossless, 1k and 8k
    /// positions, one and eight users): what `fit_consts` reads.
    fn records_at(class: &str, c: Consts) -> Vec<serde_json::Value> {
        let q35 = json::parse(&fixture("qwen35_configs.json")).unwrap();
        let shapes = [
            Shape::from_config(&fixture("gemma4_12b_config.json")).unwrap(),
            Shape::from_config(&json::to_compact(q35.get("Qwen/Qwen3.5-0.8B").unwrap())).unwrap(),
            Shape::from_config(&json::to_compact(q35.get("Qwen/Qwen3.5-9B").unwrap())).unwrap(),
        ];
        let m = Machine::preset("rtx4080s").unwrap();
        let (stream, tflops) = (700.0, 100.0);
        let mut out = Vec::new();
        for sh in &shapes {
            for (level, users, positions) in [("wren", 1, 1088.0), ("kestrel", 1, 8256.0), ("wren", 8, 1088.0), ("kestrel", 1, 1088.0), ("penguin", 1, 1088.0)]
            {
                let f = features(sh, &m, 0, &one_gpu_plan(Kind::Qwen35, level), positions, users);
                let x = |k: &str| f[k].as_f64().unwrap();
                let decode = 1.0 / c.step_s(stream * GB, x("weight_bytes"), x("kv_bytes"), x("layers"), x("hidden"), read_in(&f), users as f64);
                let prefill = 1.0 / c.token_s(x("prompt_flops"), x("layers"), x("attn_bytes"), stream * GB);
                out.push(serde_json::json!({"schema": CALIB_SCHEMA, "class": class, "stream_gbs": stream, "tflops": tflops, "users": users, "features": f,
                    "measured": {"decode": decode, "prefill": prefill}}));
            }
        }
        out
    }

    /// A class with records of its own is a fitted point: its figures are fitted on them (each of its runs predicted within 5%, the fit held a little toward the
    /// others'), its band what each run came to
    /// against the figures fitted without it. A class with none takes the other classes' fit, its band how each class with records fared predicted from the others
    /// (leave-one-class-out): here two classes made at figures 40% apart, so it is wide. No records at all: the defaults and `SPREAD`.
    #[test]
    fn a_class_is_its_own_records_fit_and_an_unseen_one_the_others_with_their_error() {
        let a = Consts { w: 1.0, wl: 1.4, kv: 1.5, layer_s: 35e-6, width_s: 0.0, row_s: 20e-6, w12: 1.0, f: 0.008, prompt_layer_s: 1.1e-6, attn: 0.0 };
        let b = Consts { w: 1.6, wl: 1.1, kv: 6.0, layer_s: 60e-6, width_s: 8e-9, row_s: 50e-6, w12: 1.0, f: 0.02, prompt_layer_s: 2e-6, attn: 3.0 };
        let mut records = records_at("smA", a);
        records.extend(records_at("smB", b));
        let rows = |k: &str| records.iter().filter(|r| r["class"] == k).collect::<Vec<_>>();
        for (k, truth) in [("smA", a), ("smB", b)] {
            let p = prior_from(&records, k);
            assert_eq!((p.source, p.records), (Source::Own, 15), "{k}");
            for r in rows(k) {
                let f = |n: &str| r["features"][n].as_f64().unwrap();
                let (read, users) = (read_in(&r["features"]), r["users"].as_f64().unwrap());
                let (want, got) = (
                    truth.step_s(700e9, f("weight_bytes"), f("kv_bytes"), f("layers"), f("hidden"), read, users),
                    p.consts.step_s(700e9, f("weight_bytes"), f("kv_bytes"), f("layers"), f("hidden"), read, users),
                );
                assert!((got / want - 1.0).abs() < 0.05, "{k}: {got} against {want}");
                let (want, got) = (
                    truth.token_s(f("prompt_flops"), f("layers"), f("attn_bytes"), 700e9),
                    p.consts.token_s(f("prompt_flops"), f("layers"), f("attn_bytes"), 700e9),
                );
                assert!((got / want - 1.0).abs() < 0.05, "{k} prompt: {got} against {want}");
            }
            assert!(p.band[0] < 1.05 && p.band[2] > 0.95 && p.band[2] / p.band[0] < 1.3, "{k}: {:?}", p.band);
        }
        let unseen = prior_from(&records, "smC");
        assert_eq!((unseen.source, unseen.records), (Source::Others, 30));
        assert!(unseen.band[2] / unseen.band[0] > 1.5, "the other classes disagree: {:?}", unseen.band);
        let none = prior_from(&[], "smA");
        assert_eq!((none.source, none.consts, none.band), (Source::Default, DEFAULT_CONSTS, SPREAD));
        // the score: each class from the other's figures (far off) and from its own less the one predicted (close: within 12%, a run of eight rows predicted
        // from the two other models' at 10.2%)
        for (k, dec, pre, loo) in score_classes(&records) {
            assert!(dec.iter().any(|e| e.abs() > 0.2) && pre.iter().any(|e| e.abs() > 0.2), "{k}: {dec:?}");
            assert!(loo.iter().all(|e| e.abs() < 0.12), "{k}: {loo:?}");
        }
        // (the shipped records: the 4080 SUPER and the A100 are fitted points, a class with no record the others' fit)
        assert_eq!(prior_for((8, 9)).source, Source::Own);
        assert_eq!(prior_for((8, 0)).source, Source::Own);
        assert_eq!(prior_for((9, 0)).source, Source::Others);
        assert_eq!(Machine::preset("h100").unwrap().devices[0].prior, prior_for((9, 0)));
    }

    /// The A100's shipped runs (one user and eight; every one at a lossy level), which the RTX 4080 SUPER's figures predicted far too fast: its own figures
    /// predict each within `FIT_A100` (a fitted point, in-sample: this pins the fit and is no evidence of other runs), and the leave-one-class-out score says how
    /// far the other class's figures were (the error an unseen class may expect).
    #[test]
    fn the_a100s_runs_are_its_figures_fit() {
        let records = records_of(CALIBRATION);
        let p = prior_from(&records, "sm80");
        let sm80: Vec<&serde_json::Value> = records.iter().filter(|r| r["class"] == "sm80").collect();
        assert!(!sm80.is_empty() && sm80.iter().all(|r| r["features"]["lossless"] == false), "the shipped A100 records are lossy levels'");
        let (dec, _) = rows_of(&sm80);
        for (x, y) in &dec {
            let t = p.consts.decode_s(x);
            eprintln!("sm80 in-sample: {t:.6} s against {y:.6} ({:+.1}%)", 100.0 * (t / y - 1.0));
            assert!((t / y - 1.0).abs() < FIT_A100, "{t} against {y}");
        }
        // (the 4080 SUPER's figures predict the A100's packs 60-283% too fast)
        let s = score_classes(&records);
        let a100 = s.iter().find(|c| c.0 == "sm80").unwrap();
        assert_eq!(sm80.len(), a100.1.len());
        for (r, e) in sm80.iter().zip(&a100.1) {
            assert!(*e > 0.3, "the 4080 SUPER's figures predict the A100's packs too fast: {} {e}", r["load"]);
        }
    }

    /// A step of several rows reads the weights once for all of them: each row past the first adds its keys' and values' reads and its class's time a layer, no
    /// weights. On the A100 (its records: one user and eight, at 63bdcdcb) eight users at 1k positions are each predicted at 1 / (the one user's step + 7 rows).
    #[test]
    fn a_step_of_several_rows_reads_the_weights_once() {
        let (_, q08) = measured_shapes();
        let m = Machine::preset("a100-80g").unwrap();
        let (c, rate, layers) = (m.devices[0].consts, m.devices[0].rate(), q08.layers as f64);
        assert!(c.row_s > 5e-6, "the A100's records hold steps of eight rows: {c:?}");
        let lp = one_gpu_plan(Kind::Qwen35, "kestrel");
        let (one, eight) = (speed_users(&q08, &m, 0, &lp, 1088.0, 1), speed_users(&q08, &m, 0, &lp, 1088.0, 8));
        let more = 7.0 * (layers * c.row_s + kv_at(&q08, 1088.0) * c.kv / rate);
        assert!((1.0 / eight - 1.0 / one - more).abs() < 1e-9, "{one} {eight}");
        // (the weights' bytes a step reads are the same for one row and eight)
        let (w1, _) = reads_of(&q08, 6.5, 6.5, 1088.0, 1.0);
        let (w8, kv8) = reads_of(&q08, 6.5, 6.5, 1088.0, 8.0);
        assert_eq!((w1, kv8), (w8, 8.0 * kv_at(&q08, 1088.0)));
    }

    /// Users who send their prompts at once each wait in their replies for the prompts read after theirs (`prompt_wait`): on the A100, Qwen3.5-0.8B's
    /// eight users took 127.3 tokens/s each on 1k prompts (994 tokens, 20,206 tokens/s) and 154.5 on short ones; the first, its wait taken out of its 127 steps,
    /// is the second within 1%. A record of such a run is fitted on its steps.
    #[test]
    fn the_prompts_a_user_waits_for_are_not_its_steps() {
        let step = 1.0 / 127.3 - prompt_wait(8, 994.0, 20206.0) / 127.0;
        assert!((1.0 / step / 154.5 - 1.0).abs() < 0.01, "{}", 1.0 / step);
        assert_eq!((prompt_wait(1, 994.0, 20206.0), prompt_wait(8, 994.0, 0.0)), (0.0, 0.0));
        let r = serde_json::json!({"stream_gbs": 1000.0, "users": 8, "made": 128, "features": {"weight_bytes": 1e9, "kv_bytes": 1e8, "layers": 24, "hidden": 1024, "in_vram": true},
            "measured": {"decode": 127.3, "prefill": 20206.0, "prompt_tokens": 994.0}});
        let (dec, _) = rows_of(&[&r]);
        assert!((dec[0].1 - step).abs() < 1e-12 && dec[0].0[5] == 7.0 * 24.0, "{dec:?}");
    }

    /// The prediction is for `Ask::positions` positions (fit's `GLYD_FIT_POSITIONS` in a development build): a longer conversation reads more keys and values a step,
    /// and is predicted slower by exactly their bytes at the GPU's rate (the sliding layers read their window either way).
    #[test]
    fn a_longer_conversation_is_predicted_slower() {
        let g = Shape::from_config(&fixture("gemma4_12b_config.json")).unwrap();
        let m = Machine::preset("rtx4080s").unwrap();
        let at = |positions: usize| chosen(&plan(&g, &m, Ask { level: Some("wren"), positions, ..Default::default() })).prior;
        assert_eq!(at(0), at(1024));
        let (short, long) = (1.0 / at(1024), 1.0 / at(8192));
        let more = g.kv_full * (8192.0 - 1024.0);
        assert!(((long - short) - more * m.devices[0].consts.kv / m.devices[0].rate()).abs() < 1e-9, "{short} {long}");
    }

    #[test]
    fn the_ladder_picks_lossless_then_lossy_then_experts_in_ram() {
        let q9 = qwen35("Qwen/Qwen3.5-9B");
        // a 24 GB card holds the 9B lossless; an 8 GB laptop card takes a lossy level, and says the driver is too old today
        let p = plan(&q9, &Machine::preset("rtx4090").unwrap(), Ask::default());
        assert_eq!((chosen(&p).level, chosen(&p).placement.clone()), ("penguin", Some(Placement::Gpu(0))));
        let p = plan(&q9, &Machine::preset("rtx4060-laptop").unwrap(), Ask::default());
        // (the lossy engines keep 2 GiB of an 8 GB card: swift's 5.8 GiB of weights do not fit beside them, so only wren does)
        assert!(chosen(&p).level != "penguin" && p.options[0].why.is_some(), "{p:?}");
        // (the preset's driver is CUDA 12's: a build with CUDA 13's kernels alone says what it needs; one with CUDA 12's runs on it)
        assert_eq!(
            p.notes.iter().any(|n| n.contains("the engine's kernels need CUDA 13.0 (driver 580 or newer)")),
            !crate::machine::KERNELS12,
            "{:?}",
            p.notes
        );
        // the large dense shape lossless on one 48 GB card (its bf16 needs two); two 24 GB cards hold it at kestrel, split; one 24 GB card at a lower level
        let q32 = Shape::from_config(DENSE_64L).unwrap();
        let p = plan(&q32, &Machine::preset("a6000").unwrap(), Ask::default());
        assert_eq!((chosen(&p).level, chosen(&p).placement.clone()), ("penguin", Some(Placement::Gpu(0))));
        let p = plan(&q32, &Machine::preset("2x3090").unwrap(), Ask::default());
        assert_eq!((chosen(&p).level, chosen(&p).placement.clone()), ("kestrel", Some(Placement::Split(vec![0, 1]))));
        // (one 24 GB card: only wren holds it)
        let p = plan(&q32, &Machine::preset("rtx4090").unwrap(), Ask::default());
        assert_eq!(chosen(&p).level, "wren");
        // the 180B MoE fits no 16 GB card at any level but in RAM; on eight H100s it is lossless, split
        let qx = Shape::from_config(&fixture("qwen4x_config.json")).unwrap();
        let p = plan(&qx, &Machine::preset("rtx4080s").unwrap(), Ask::default());
        assert!(matches!(chosen(&p).placement, Some(Placement::ExpertsInRam { .. })) && chosen(&p).level == "hummingbird", "{p:?}");
        let p = plan(&qx, &Machine::preset("8xh100").unwrap(), Ask::default());
        assert!(chosen(&p).level == "penguin" && matches!(chosen(&p).placement, Some(Placement::Gpu(_) | Placement::Split(_))));
        // the context is the most that fits, to 32,768 (or what is asked); a machine with no GPU has no plan yet
        assert_eq!(chosen(&plan(&q9, &Machine::preset("rtx4090").unwrap(), Ask::default())).context, TARGET_CTX);
        let none = Machine { name: "cpu".into(), devices: vec![], ram_free: 64 << 30, ram_bandwidth: 80.0, cores: 16, arch: "x86_64".into() };
        assert!(plan(&q9, &none, Ask::default()).chosen.is_none());
    }

    #[test]
    fn unified_memory_is_counted_once() {
        // the 180B MoE on machines whose GPU shares the computer's memory: never "experts in RAM" (the same memory, counted twice)
        let qx = Shape::from_config(&fixture("qwen4x_config.json")).unwrap();
        for name in ["dgx-spark", "strix-halo-128g", "m4max-64g"] {
            let m = Machine::preset(name).unwrap();
            assert!(m.devices.iter().all(|d| d.unified), "{name}");
            let p = plan(&qx, &m, Ask::default());
            assert!(!p.options.iter().any(|o| matches!(o.placement, Some(Placement::ExpertsInRam { .. }))), "{name}: {p:?}");
        }
        assert!(!Machine::preset("rtx4080s").unwrap().devices[0].unified);
        let m = Machine::from_json(r#"{"gpus": [{"name": "Some APU", "vendor": "amd", "total_gib": 64, "unified": true}]}"#).unwrap();
        assert!(m.devices[0].unified);
    }

    #[test]
    fn a_plan_for_a_family_is_one_run_would_load() {
        // every preset, every model fixture, every level its family runs at: no option outside the family's levels,
        // placements and context (the table run reads, src/support.rs)
        let q35 = json::parse(&fixture("qwen35_configs.json")).unwrap();
        let mut configs =
            vec![("dense-64l".to_string(), DENSE_64L.to_string()), ("Qwen/Qwen3.5-9B".to_string(), json::to_compact(&q35.as_object().unwrap()[0].1))];
        for (f, m) in [("gemma4_12b_config.json", "google/gemma-4-12B-it"), ("glm5_next_config.json", "zai-org/GLM-5.3-Flash")] {
            configs.push((m.to_string(), fixture(f)));
        }
        // (the mixtures of experts with their experts in the computer's memory: the released configs of the recorded headers)
        for h in ["Kimi-K3", "DeepSeek-V4-Flash-0731"] {
            let v: serde_json::Value = serde_json::from_str(&fixture(&format!("headers/{h}.json"))).unwrap();
            configs.push((h.to_string(), v["config"].as_str().unwrap().to_string()));
        }
        let mut checked = 0;
        for (model, text) in &configs {
            let row = crate::support::row(crate::support::family(text, model).unwrap());
            let sh = Shape::from_config(text).unwrap();
            let qat = levels::is_qat(model);
            for preset in Machine::PRESETS {
                let m = Machine::preset(preset).unwrap();
                for level in std::iter::once(None).chain(row.levels_now(model).into_iter().map(Some)) {
                    let p = plan(&sh, &m, Ask { level, family: Some(row), qat, ..Default::default() });
                    for o in &p.options {
                        assert!(row.allows(o.level), "{model} on {preset}: {} is not a level {} runs at", o.level, row.name);
                        match &o.placement {
                            Some(Placement::Split(_)) => assert!(row.split_levels.contains(&o.level), "{model} on {preset}: split at {}", o.level),
                            Some(Placement::ExpertsInRam { .. }) => {
                                assert!(row.experts_in_ram_levels.contains(&o.level), "{model} on {preset}: experts in RAM at {}", o.level)
                            }
                            _ => {}
                        }
                        if let Some(d) = row.default_ctx {
                            assert!(o.context <= d, "{model} on {preset}: {} tokens where run takes {d}", o.context);
                        }
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 100, "{checked}");
    }

    #[test]
    fn a_machine_reads_from_json_and_presets_name_their_gpus() {
        let m = Machine::from_json(
            r#"{"name": "mine", "ram_free_gib": 30, "gpus": [{"name": "NVIDIA GeForce RTX 3090", "cc": [8, 6], "total_gib": 24, "free_gib": 23}]}"#,
        )
        .unwrap();
        assert_eq!((m.devices.len(), m.devices[0].bandwidth, m.devices[0].cc), (1, 936.0, (8, 6)));
        for p in Machine::PRESETS {
            let m = Machine::preset(p).unwrap();
            assert!(m.devices.iter().all(|d| bandwidth_of(&d.name).is_some()), "{p}");
        }
        assert_eq!(
            (bandwidth_of("NVIDIA GeForce RTX 4080 SUPER"), bandwidth_of("NVIDIA GeForce RTX 4080"), bandwidth_of("NVIDIA L40S"), bandwidth_of("NVIDIA L4")),
            (Some(736.0), Some(717.0), Some(864.0), Some(300.0))
        );
    }

    // ---- the plan is what loads ----------------------------------------------------------------------------------------------------------------------------------------------------------------

    fn args(v: &[&str]) -> crate::args::Args {
        crate::args::Args::from(v.iter().map(|s| s.to_string()))
    }

    /// A directory holding only a config.json, named `name`, which `fit` reads as it reads a model on the disk.
    fn model_dir(tag: &str, name: &str, config: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("glyd-plan-{tag}-{}", std::process::id())).join(name);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), config).unwrap();
        dir
    }

    /// The keys the Qwen3 path reads that the planner does not.
    fn with_rope(config: &str) -> String {
        config.replace("}", ", \"rms_norm_eps\": 1e-6, \"rope_theta\": 1000000}")
    }

    /// The plain dense kind: hidden 4,096, 36 layers of 32 query heads over 8 KV heads of 128, MLP 12,288, the large one's vocabulary.
    const DENSE_36L: &str = r#"{"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 4096, "intermediate_size": 12288, "num_hidden_layers": 36, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 151936, "tie_word_embeddings": false, "max_position_embeddings": 40960, "rms_norm_eps": 1e-6, "rope_theta": 1000000}"#;

    fn gpu_of(d: &Device, ordinal: i32) -> crate::dev::Gpu {
        crate::dev::Gpu { name: d.name.clone(), cc: d.cc, sms: 0, total: d.total, free: d.free, driver: d.driver, count: 1, ordinal }
    }

    /// Each stage's share of what the computer's memory holds goes by its layers' bytes there and adds up to all of it exactly, the rounding the last stage's; none
    /// where nothing is there, all on the first where no layer holds any.
    #[test]
    fn the_stages_shares_of_the_computers_memory_add_up() {
        use sizing::stage_shares;
        assert_eq!(stage_shares(&[1, 1, 2, 2], &[2], 600), vec![200, 400]);
        assert_eq!(stage_shares(&[3, 3, 3], &[1, 2], 100), vec![33, 33, 34]);
        assert_eq!(stage_shares(&[0, 5, 5, 0], &[2], 1001).iter().sum::<u64>(), 1001);
        assert_eq!(stage_shares(&[0, 0], &[1], 9), vec![9, 0]);
        assert_eq!(stage_shares(&[4, 4], &[], 0), vec![0]);
        assert_eq!(stage_shares(&[u64::MAX / 2, u64::MAX / 2], &[1], u64::MAX - 1), vec![(u64::MAX - 1) / 2, (u64::MAX - 1) / 2]);
    }

    /// A plan whose users share pools of blocks holds at most what the GPU has left for them (at least one's context): on an A10 (23 GB) a dense model of 8B parameters served to 8 at
    /// 16,384 tokens was planned at 31,867 MiB, eight full contexts, where the server took 21,889. And its decode reads penguin's weights in the layout it loads
    /// (the 12-bit one off Ada: 11.10 GB measured on the A10 and the A100, where the level's tiered 10.75 bits counted 10.17), at that layout's share.
    #[test]
    fn a_shared_pool_is_planned_within_the_gpu() {
        let dir = model_dir("pool", "dense-36l", DENSE_36L);
        let m = dir.to_str().unwrap();
        for machine in ["l4", "rtx4090", "a6000"] {
            let p = planned(&args(&["serve", m, "--users", "8", "--ctx", "16384", "--machine", machine])).unwrap().plan.chosen.and_then(|c| c.load).unwrap();
            let one = p.kv / 8;
            assert!(p.gpu_total() <= p.free && p.users == 8 && p.context == 16384, "{machine}: {}", p.text());
            assert!(p.kv >= 16384 * crate::plan::sizing::kv_per_token(&crate::weights::Config::parse(DENSE_36L, "c", None).unwrap(), false) && one > 0);
        }
        let sh = Shape::from_config(DENSE_36L).unwrap();
        let m = Machine::preset("a6000").unwrap();
        let mut lp = one_gpu_plan(Kind::Qwen3, "penguin");
        lp.layout = crate::weights::Layout::Twelve;
        let f = features(&sh, &m, 0, &lp, 1088.0, 1);
        assert_eq!((f["lossless"].as_bool(), f["twelve"].as_bool()), (Some(true), Some(true)));
        let (w12, _) = reads_of(&sh, 12.04, 12.04, 1088.0, 1.0);
        assert_eq!(f["weight_bytes"].as_f64(), Some(w12));
        lp.layout = crate::weights::Layout::Tiered;
        assert_eq!(features(&sh, &m, 0, &lp, 1088.0, 1)["twelve"].as_bool(), Some(false));
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
    }

    /// A Qwen3.5 model served to --users N is planned with N sequences, each up to the context (the executor's pools hold up to that, src/dynamic/rows.rs `page`):
    /// on an A100 40GB Qwen3.5-0.8B served to 8 took 5,054 MiB where its plan for one said 3,037. Each server's sequence holds its states twice (a copy at its mark)
    /// and its context, and a server's pools take once the copies of states at block boundaries and a block 0 (`sizing::pools_of`); run's one sequence its states
    /// and its context alone; a GPU too small for N at the full context gives each the context the room leaves them. `fit --users N` plans what `serve` does.
    #[test]
    fn a_qwen35_server_is_planned_for_its_users() {
        let q35: serde_json::Value = serde_json::from_str(&fixture("qwen35_configs.json")).unwrap();
        let dir = model_dir("q35users", "Qwen3.5-0.8B", &q35["Qwen/Qwen3.5-0.8B"].to_string());
        let m = dir.to_str().unwrap();
        let plan_of = |words: &[&str]| planned(&args(words)).unwrap().plan.chosen.and_then(|c| c.load).unwrap();
        let tag = format!("{m}:kestrel");
        let one = plan_of(&["run", &tag, "--machine", "a100-80g"]);
        let eight = plan_of(&["serve", &tag, "--users", "8", "--machine", "a100-80g"]);
        let (s1, s2) = (plan_of(&["serve", &tag, "--users", "1", "--machine", "a100-80g"]), plan_of(&["serve", &tag, "--users", "2", "--machine", "a100-80g"]));
        assert_eq!((one.users, eight.users, eight.context, s1.context), (1, 8, one.context, one.context));
        let each = s2.kv - s1.kv;
        assert_eq!(eight.kv, s1.kv + 7 * each, "each of the eight's states and context");
        assert!(each > one.kv && s1.kv > each, "a server's sequence holds more than run's one (its mark), and its pools something once: {each} {one:?} {s1:?}");
        assert_eq!(eight.gpu_total() - eight.kv, one.gpu_total() - one.kv);
        assert_eq!(plan_of(&["fit", &tag, "--users", "8", "--machine", "a100-80g"]), eight, "fit --users plans the server");
        // (Qwen3.5-9B to 64 users on a 24 GB card: the context the room gives each, within what is free)
        let dir9 = model_dir("q35users", "Qwen3.5-9B", &q35["Qwen/Qwen3.5-9B"].to_string());
        let tag9 = format!("{}:wren", dir9.to_str().unwrap());
        let many = plan_of(&["serve", &tag9, "--users", "64", "--machine", "rtx4090"]);
        assert!(many.users == 64 && many.context < LOSSY_CTX && many.context >= LOSSY_MIN_CTX && many.gpu_total() <= many.free, "{}", many.text());
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
    }

    /// What `fit` prints for each family at each level it runs at, on each NVIDIA preset, is the plan `run` builds from the same GPU (the functions
    /// `run` calls: `sizing::qwen3_plan` for Qwen3, `sizing::lossy_plan_with` for the lossy families), or the refusal `run` gives; and what it plans fits the GPU.
    #[test]
    fn fit_is_the_plan_run_builds() {
        let q35 = json::parse(&fixture("qwen35_configs.json")).unwrap();
        let configs = [
            ("dense-64l", with_rope(DENSE_64L)),
            ("Qwen3.5-9B", json::to_compact(&q35.as_object().unwrap()[0].1)),
            ("gemma-4-12B", fixture("gemma4_12b_config.json")),
        ];
        let mut sized = 0;
        for (name, text) in &configs {
            let dir = model_dir("run", name, text);
            let model = dir.to_str().unwrap();
            let row = crate::support::row(crate::support::family(text, name).unwrap());
            for preset in Machine::PRESETS {
                let m = Machine::preset(preset).unwrap();
                if m.devices.iter().any(|d| d.vendor != Vendor::Nvidia) || m.devices.len() != 1 {
                    continue;
                }
                let d = &m.devices[0];
                let gpu = gpu_of(d, 0);
                for level in row.levels_now(model) {
                    let tag = format!("{model}:{level}");
                    let a = args(&["fit", &tag, "--machine", preset]);
                    let f = planned(&a).unwrap();
                    let run = match row.family {
                        crate::support::Family::Qwen3 => {
                            let cfg = crate::weights::Config::parse(text, "config.json", None).unwrap();
                            let layout = crate::levels::layout_of(level).filter(|l| *l != crate::weights::Layout::Tiered);
                            // (run's plan on the preset's GPU with the preset's word on cuBLAS, as run on that machine plans it; on Windows the name of a path
                            // is the whole path)
                            let t = Target { cublas: d.cublas, ..sizing::target(&gpu) };
                            sizing::dense_plan_on(&args(&["run", model]), &cfg, &crate::cli::short_name(model), &t, layout, 1).map(|p| p.2)
                        }
                        _ => {
                            let layout = crate::levels::layout_for(model, crate::levels::layout_of(level).unwrap());
                            if layout == crate::weights::Layout::Tiered && row.split_levels.contains(&"penguin") {
                                continue; // (the dynamic route's, sized from the files' headers: a directory of a config.json alone has none)
                            }
                            sizing::lossy_plan_with(&args(&["run", model]), model, layout, &mc(text), None, d.free, &d.name, None, Some(m.ram_free))
                        }
                    };
                    match (run, f.plan.options.iter().find(|o| o.level == level)) {
                        (Ok(want), Some(o)) => {
                            let got = o.load.as_ref().unwrap_or_else(|| panic!("{name} {preset} {level}: no loader plan"));
                            assert_eq!(*got, want, "{name} on {preset} at {level}");
                            assert!(got.gpu_total() <= d.free, "{name} on {preset} at {level}: {} on {} free", got.gpu_total(), d.free);
                            assert!(got.context >= MIN_CTX.min(row.max_ctx.unwrap_or(usize::MAX)), "{name} on {preset} at {level}: {}", got.context);
                            sized += 1;
                        }
                        (Err(want), Some(o)) => {
                            assert_eq!(o.why.as_deref(), Some(want.what.as_str()), "{name} on {preset} at {level}");
                            // (the one level asked: fit refuses with the words run has)
                            let r = f.refusal.as_ref().unwrap_or_else(|| panic!("{name} on {preset} at {level}: fit plans what run refuses"));
                            // (`run` checks the GPU and its driver before it sizes anything)
                            let want = crate::support::machine(&m).unwrap_or(want);
                            assert_eq!(
                                (r.what.as_str(), r.fix.as_str(), r.code),
                                (want.what.as_str(), want.fix.as_str(), want.code),
                                "{name} on {preset} at {level}"
                            );
                            sized += 1;
                        }
                        (_, None) => panic!("{name} on {preset} at {level}: no option"),
                    }
                }
            }
        }
        assert!(sized > 60, "{sized}");
    }

    /// The plan holds `users` sequences of the context (a server's pool), which the Qwen3 path sized for one until the plan drove it: four users on a 24 GB card
    /// keep four contexts of cache and still fit.
    #[test]
    fn users_are_in_the_plan() {
        let dir = model_dir("users", "dense-36l", DENSE_36L);
        let one = planned(&args(&["fit", dir.to_str().unwrap(), "--machine", "l4"])).unwrap().plan.chosen.unwrap().load.unwrap();
        let four = planned(&args(&["fit", dir.to_str().unwrap(), "--machine", "l4", "--users", "4"])).unwrap().plan.chosen.unwrap().load.unwrap();
        assert_eq!((one.users, four.users), (1, 4));
        assert_eq!(four.kv, 4 * (four.kv / 4), "four whole contexts");
        assert!(four.kv / four.context as u64 >= 3 * (one.kv / one.context as u64), "the cache of four users per token of context");
        assert!(four.gpu_total() <= four.free && one.gpu_total() <= one.free);
    }

    /// `fit --json` is `glyd-plan/1`: the numbers the text has, under names that stay.
    #[test]
    fn fit_json_is_a_stable_schema() {
        let dir = model_dir("json", "gemma-4-12B", &fixture("gemma4_12b_config.json"));
        let f = planned(&args(&["fit", &format!("{}:kestrel", dir.to_str().unwrap()), "--machine", "rtx4080s"])).unwrap();
        let c = f.plan.chosen.unwrap();
        let v: serde_json::Value = serde_json::from_str(&c.json("google/gemma-4-12B-it", &f.machine)).unwrap();
        for k in [
            "schema",
            "model",
            "family",
            "level",
            "context",
            "users",
            "chunk",
            "gpu",
            "free_bytes",
            "weights_bytes",
            "weights_on_gpu_bytes",
            "kv_bytes",
            "work_bytes",
            "draft_bytes",
            "reserve_bytes",
            "gpu_total_bytes",
            "experts",
            "exact",
            "tokens_per_second",
            "tokens_per_second_p10",
            "tokens_per_second_p90",
            "prefill_tokens_per_second",
        ] {
            assert!(v.get(k).is_some(), "{k} is in glyd-plan/1: {v}");
        }
        assert_eq!((v["schema"].as_str(), v["level"].as_str(), v["exact"].as_bool()), (Some("glyd-plan/1"), Some("kestrel"), Some(true)));
        assert_eq!(v["gpu_total_bytes"].as_u64(), Some(c.load.as_ref().unwrap().gpu_total()));
    }

    /// The bytes of a directory's weight files (the safetensors shards, as the engine reads them onto the GPU).
    #[cfg(feature = "dev")]
    fn shard_bytes(dir: &std::path::Path) -> u64 {
        std::fs::read_dir(dir)
            .unwrap()
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
            .map(|e| e.metadata().unwrap().len())
            .sum()
    }

    /// A GPU under WDDM keeps `WDDM_EXTRA` more free than another (its plans take that much less), and the room a plan has shrinks by it.
    #[test]
    fn a_wddm_gpu_keeps_more_free() {
        let free = 15u64 << 30;
        assert_eq!(reserve_with(free, false), CONTEXT_FLOOR);
        assert_eq!(reserve_with(free, true), CONTEXT_FLOOR + WDDM_EXTRA);
        assert!(reserve_with(free, true) > reserve_with(free, false));
        assert_eq!(free - reserve_with(free, false) - (free - reserve_with(free, true)), WDDM_EXTRA);
    }

    /// The plan's weights against what the loader makes of a model: a tiny Qwen3 and a tiny Qwen3.5-family model (random weights, the shapes of the real
    /// ones: src/logits.rs) packed on the CPU at each layout the paths run, and the bytes the packed directory holds. The plan sizes a lossy level from the
    /// config alone before anything is downloaded (within 3%: a pack's tail words and the small tensors) and from the directory's own bytes once it is packed.
    #[cfg(feature = "dev")] // (the random models are a development build's: src/logits.rs)
    #[test]
    fn the_plan_weighs_what_the_packer_writes() {
        use crate::weights::Layout;
        let root = std::env::temp_dir().join(format!("glyd-plan-pack-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let (q3, q35) = (root.join("dense"), root.join("qwen35"));
        crate::logits::dense_random(&q3).unwrap();
        crate::logits::qwen35_random(&q35).unwrap();
        let near = |got: u64, want: u64, what: &str| assert!((got as f64 / want as f64 - 1.0).abs() < 0.03, "{what}: planned {got}, written {want}");
        // Qwen3 dense, lossless
        let cfg = crate::weights::Config::load(&q3).unwrap();
        let out = root.join("qwen3-packed");
        crate::pack::run_with(q3.to_str().unwrap(), &out, 2, Layout::Tiered, true).unwrap();
        // (the lossless layouts' bytes depend on the values: the plan takes 10.8 bits a weight as the real ones come to, and these random ones are smaller)
        let (planned, written) = (sizing::weights(&cfg, Layout::Tiered), shard_bytes(&out));
        assert!(written as f64 <= planned as f64 * 1.03 && written as f64 >= planned as f64 * 0.8, "Qwen3 tiered: planned {planned}, written {written}");
        // Qwen3.5-family, each lossy level
        let text = std::fs::read_to_string(q35.join("config.json")).unwrap();
        let a = args(&["run", q35.to_str().unwrap()]);
        for layout in [Layout::Int5, Layout::Int4] {
            let out = root.join(format!("qwen35-{}", layout.name()));
            crate::pack::run_with(q35.to_str().unwrap(), &out, 2, layout, true).unwrap();
            let written = shard_bytes(&out);
            let ask = |packed: Option<&std::path::Path>| {
                sizing::lossy_plan_with(&a, q35.to_str().unwrap(), layout, &mc(&text), None, 24 << 30, "NVIDIA RTX 4090", packed, None).unwrap()
            };
            near(ask(None).weights, written, &format!("Qwen3.5-family {}", layout.name()));
            let packed = ask(Some(&out));
            near(packed.weights, written, &format!("Qwen3.5-family {} from its directory", layout.name()));
            assert!(packed.gpu_total() <= packed.free);
        }
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A checkpoint of the tensors of `fixture` (their names, types and shapes, from a real release's header: `tensors`) as a sparse file: the header the planner
    /// reads, no bytes written.
    /// The released tensors of a Qwen3.5-layout model the fixtures list, as `header_only_checkpoint` takes them: Qwen3.5 0.8B's and 9B's index, Qwen3.8-27B's
    /// header fixture (its groups expanded); None for another model.
    fn qwen35_tensors(name: &str) -> Option<String> {
        let tensors: serde_json::Map<String, serde_json::Value> = match name {
            "Qwen3.5-0.8B" | "Qwen3.5-9B" => {
                serde_json::from_str::<serde_json::Value>(&fixture("qwen35_tensors.json")).unwrap()[format!("Qwen/{name}")].as_object()?.clone()
            }
            "Qwen3.8-27B" => {
                let h: serde_json::Value = serde_json::from_str(&fixture("headers/Qwen3.8-27B.json")).unwrap();
                let groups = h["groups"].as_array().unwrap();
                let mut out = serde_json::Map::new();
                for g in groups {
                    let mask = g["mask"].as_str().unwrap();
                    let layers: Vec<String> = match g["layers"].as_array().unwrap() {
                        l if l.is_empty() => vec![String::new()],
                        l => l.iter().map(|x| x.to_string()).collect(),
                    };
                    assert_eq!(g["experts"], 0, "a dense model's groups");
                    for l in layers {
                        out.insert(mask.replace("{L}", &l), serde_json::json!([g["dtype"], g["shape"]]));
                    }
                }
                out
            }
            _ => return None,
        };
        Some(serde_json::json!({ "tensors": tensors }).to_string())
    }

    /// The released tensors of a fixture model of the dynamic route's families whose penguin is planned from its files' headers (GLM-5.3-Flash's, Qwen3.8-Flash-Next's).
    /// A release's config.json and its tensors as its recorded headers give them (tests/fixtures/headers: the Hub's safetensors headers, read by ranges, no weights).
    fn recorded_release(name: &str) -> (String, crate::sizes::Table) {
        let v: serde_json::Value = serde_json::from_str(&fixture(&format!("headers/{name}.json"))).unwrap();
        let u64s = |x: &serde_json::Value| x.as_array().unwrap().iter().map(|n| n.as_u64().unwrap()).collect::<Vec<u64>>();
        let groups: Vec<crate::sizes::Group> = v["groups"]
            .as_array()
            .unwrap()
            .iter()
            .map(|g| crate::sizes::Group {
                mask: g["mask"].as_str().unwrap().to_string(),
                dtype: g["dtype"].as_str().unwrap().to_string(),
                shape: u64s(&g["shape"]),
                bytes: g["bytes"].as_u64().unwrap(),
                layers: u64s(&g["layers"]).into_iter().map(|l| l as usize).collect(),
                experts: g["experts"].as_u64().unwrap() as usize,
            })
            .collect();
        (v["config"].as_str().unwrap().to_string(), crate::sizes::Table::expand(&groups))
    }

    fn released_tensors(name: &str) -> Option<String> {
        match name {
            "GLM-5.3-Flash" => Some(fixture("glm5_next_tensors.json")),
            "Qwen3.8-Flash-Next" => Some(fixture("qwen4x_tensors.json")),
            _ => None,
        }
    }

    fn header_only_checkpoint(dir: &std::path::Path, fixture: &str) {
        let v = json::parse(fixture).unwrap();
        let (mut header, mut at) = (String::from("{"), 0u64);
        for (name, t) in v.get("tensors").and_then(Value::as_object).unwrap() {
            let (dtype, mut shape) = (t.as_array().unwrap()[0].as_str().unwrap(), t.as_array().unwrap()[1].as_u64s().unwrap());
            // (a sparse file where the system makes them, else the shapes' large dimensions cut by 16 so that the file is under a GiB: a Windows runner's disk is not 15 GiB)
            if cfg!(windows) {
                shape.iter_mut().filter(|d| **d >= 256).for_each(|d| *d /= 16);
            }
            let width = match dtype {
                "F64" | "I64" | "U64" => 8,
                "F32" | "I32" | "U32" => 4,
                "F8_E4M3" | "I8" | "U8" => 1,
                _ => 2,
            };
            let bytes = shape.iter().product::<u64>() * width;
            header +=
                &format!("{}\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":{shape:?},\"data_offsets\":[{at},{}]}}", if at == 0 { "" } else { "," }, at + bytes);
            at += bytes;
        }
        header += "}";
        let mut f = std::fs::File::create(dir.join("model.safetensors")).unwrap();
        std::io::Write::write_all(&mut f, &(header.len() as u64).to_le_bytes()).unwrap();
        std::io::Write::write_all(&mut f, header.as_bytes()).unwrap();
        f.set_len(8 + header.len() as u64 + at).unwrap();
    }

    /// A model of the dynamic route whose files are here (GLM-5.3-Flash, from its real header): `fit` prints the plan `run` loads, with the layers cut over the
    /// GPUs of each machine as their room holds them (each stage's bytes and the route's work and reserve inside its GPU's free memory, every layer in exactly
    /// one stage), and refuses where `run` refuses, in its words.
    #[test]
    fn the_dynamic_route_is_planned_from_the_files() {
        let dir = model_dir("dyn", "GLM-5.3-Flash", &fixture("glm5_next_config.json"));
        header_only_checkpoint(&dir, &fixture("glm5_next_tensors.json"));
        let nl = crate::dynamic::split::config_num(&fixture("glm5_next_config.json"), "num_hidden_layers").unwrap();
        let mut planned_ok = 0;
        for preset in ["rtx4080s", "rtx4090", "2x3090", "a100-80g", "8xh100"] {
            let m = Machine::preset(preset).unwrap();
            let f = planned(&args(&["fit", &format!("{}:penguin", dir.to_str().unwrap()), "--machine", preset])).unwrap();
            let devs: Vec<Target> = m.devices.iter().enumerate().map(|(i, d)| Target::of(d, i as i32, Some(m.ram_free))).collect();
            let table = crate::sizes::Table::from_dir(&dir).unwrap();
            let direct = sizing::size_dynamic(
                &args(&["run", dir.to_str().unwrap()]),
                &crate::cli::short_name(dir.to_str().unwrap()),
                &table,
                Some(&dir),
                &mc(&fixture("glm5_next_config.json")),
                "GLM-5.3-Flash",
                &devs,
                Some(m.ram_free),
                &|_| false,
            );
            match (f.plan.chosen.as_ref().and_then(|c| c.load.clone()), direct) {
                (Some(got), Ok(want)) => {
                    assert_eq!(got, want, "{preset}");
                    let sp = got.spread.as_ref().unwrap();
                    assert_eq!(sp.devices.len(), sp.cuts.len() + 1, "{preset}: a GPU for each stage");
                    assert!(sp.cuts.windows(2).all(|w| w[0] < w[1]) && sp.cuts.iter().all(|&c| 0 < c && c < nl), "{preset}: cuts {:?}", sp.cuts);
                    // each stage's bytes (the layers' and its half of the rest) inside its GPU's room
                    // (an HBM GPU's FP8 matrices are plain unless the plan codes them; elsewhere coded is their only layout)
                    let hbm = m.devices.iter().any(|d| d.cc.1 == 0 && (8..=10).contains(&d.cc.0));
                    let f8 = if sp.coded || !hbm { crate::dynamic::split::CODED_PCT } else { crate::dynamic::split::PLAIN_PCT };
                    let bf16 = if sp.twelve { crate::dynamic::split::TWELVE_PCT } else { crate::dynamic::split::TIERED_PCT };
                    let (per, base) = crate::dynamic::split::layer_bytes_where(&dir, nl, f8, bf16, &|_| true).unwrap();
                    let bounds: Vec<usize> = std::iter::once(0).chain(sp.cuts.iter().copied()).chain(std::iter::once(nl)).collect();
                    let mut total = 0;
                    for (k, w) in bounds.windows(2).enumerate() {
                        let held = per[w[0]..w[1]].iter().sum::<u64>()
                            + if k == 0 { base / 2 } else { 0 }
                            + if k + 1 == sp.devices.len() { base - base / 2 } else { 0 };
                        let d = &m.devices[sp.devices[k] as usize];
                        assert!(
                            held <= crate::plan::dynamic_room_of(d.free),
                            "{preset}: stage {k} holds {held} of {} in room",
                            crate::plan::dynamic_room_of(d.free)
                        );
                        total += held;
                    }
                    assert_eq!(total, got.weights_on_gpu(), "{preset}: the stages hold what the plan says");
                    assert!(got.gpu_total() <= got.free, "{preset}");
                    planned_ok += 1;
                }
                (None, Err(want)) => {
                    let r = f.refusal.as_ref().unwrap_or_else(|| panic!("{preset}: fit plans what run refuses"));
                    assert_eq!((r.what.as_str(), r.code), (want.what.as_str(), want.code), "{preset}");
                }
                (a, b) => panic!("{preset}: fit {a:?}, run {:?}", b.map(|p| p.level)),
            }
        }
        assert!(planned_ok >= 3, "{planned_ok}");
    }

    /// Qwen3.8-Flash-Next at penguin is cut over the GPUs as the route cuts any family's layers, its tables read by rows on the host as its description holds
    /// them (src/dynamic/tables.rs: the n-gram tables and the embedding, none of the GPUs' bytes): a model of the release's shapes whose layers two GPUs hold
    /// between them (its table of n-grams cut to one shard of the release's 128) is planned on two H200s in two stages, each inside its GPU's room without the
    /// tables, and `--split` asking that cut is that plan; one H100 refuses it with the memory it needs. The release itself, from its files' headers: its 95 GiB of
    /// n-gram tables and its embedding are the computer's (they were one layer's on a GPU, more than any holds), so eight H100s hold the rest of it; an RTX 4080
    /// SUPER and an L4 refuse it for the GPU memory its experts need at penguin (every one on the GPU: the family's packs are `qwenserve`'s), the tables not in it.
    #[test]
    fn the_newest_qwen_family_is_cut_over_the_gpus_at_penguin() {
        let text = fixture("qwen4x_config.json");
        let nl = crate::dynamic::split::config_num(&text, "num_hidden_layers").unwrap();
        let released = json::parse(&fixture("qwen4x_tensors.json")).unwrap();
        let host: std::collections::HashSet<String> = crate::dynamic::tables::host_names(&mc(&text)).into_iter().collect();
        let size = |v: &Value, keep: &dyn Fn(&str) -> bool| -> u64 {
            let t = v.get("tensors").and_then(Value::as_object).unwrap();
            t.iter()
                .filter(|(n, _)| keep(n))
                .map(|(_, x)| {
                    let a = x.as_array().unwrap();
                    a[1].as_u64s().unwrap().iter().product::<u64>() * if a[0].as_str() == Some("I64") { 8 } else { 2 }
                })
                .sum()
        };
        // (the release's tables on the host: its 128 shards of n-grams and its embedding, 95.4 and 1.2 GiB)
        assert_eq!(host.len(), 129, "{host:?}");
        assert!(host.contains("model.language_model.embed_tokens.weight") && !host.contains("lm_head.weight"));
        let tables = size(&released, &|n| host.contains(n));
        assert_eq!(tables, 128 * 2_500_012 * 160 * 2 + 248_320 * 2_560 * 2);
        let mut cut = released.clone();
        if let Value::Object(top) = &mut cut {
            for (k, v) in top.iter_mut().filter(|(k, _)| k == "tensors") {
                let Value::Object(t) = v else { panic!("{k}") };
                t.retain(|(n, _)| !n.contains(".ngram_embedding.shard_") || n.ends_with(".shard_0.weight"));
            }
        }
        let dir = model_dir("qx-split", "Qwen3.8-Flash-Next", &text);
        header_only_checkpoint(&dir, &json::to_compact(&cut));
        let h200 = r#"{"name": "NVIDIA H200", "cc": [9, 0], "total_gib": 141, "free_gib": 139.5}"#;
        let two = dir.parent().unwrap().join("2xh200.json");
        std::fs::write(&two, format!(r#"{{"name": "2xh200", "gpus": [{h200}, {h200}], "ram_free_gib": 400}}"#)).unwrap();
        let penguin = format!("{}:penguin", dir.to_str().unwrap());
        let fit = |machine: &str, more: &[&str]| {
            let mut w = vec!["fit", penguin.as_str(), "--machine", machine];
            w.extend(more);
            planned(&args(&w)).unwrap()
        };
        let m = Machine::from_json(&std::fs::read_to_string(&two).unwrap()).unwrap();
        let f = fit(two.to_str().unwrap(), &[]);
        let p = f.plan.chosen.as_ref().and_then(|c| c.load.clone()).unwrap_or_else(|| panic!("{:?}", f.refusal));
        let sp = p.spread.as_ref().unwrap();
        assert_eq!((p.family, p.level, sp.devices.as_slice(), sp.cuts.len()), ("Qwen3.8-Flash-Next", "penguin", &[0, 1][..], 1));
        assert_eq!(sp.tables, size(&cut, &|n| host.contains(n)), "the cut model's tables, the computer's");
        let bf16 = if sp.twelve { crate::dynamic::split::TWELVE_PCT } else { crate::dynamic::split::TIERED_PCT };
        let (per, base) = crate::dynamic::split::layer_bytes_where(&dir, nl, crate::dynamic::split::CODED_PCT, bf16, &|n| !host.contains(n)).unwrap();
        let c = sp.cuts[0];
        for (k, held) in [per[..c].iter().sum::<u64>() + base / 2, per[c..].iter().sum::<u64>() + base - base / 2].into_iter().enumerate() {
            assert!(held <= crate::plan::dynamic_room_of(m.devices[k].free), "stage {k} holds {held}");
        }
        assert_eq!(per.iter().sum::<u64>() + base, p.weights_on_gpu(), "the stages hold the model but its tables");
        assert!(p.gpu_total() <= p.free && p.weights == p.weights_on_gpu() + sp.tables);
        // (the cut asked for is the one loaded: the same plan)
        let asked = fit(two.to_str().unwrap(), &["--split", &c.to_string(), "--devices", "0,1"]);
        assert_eq!(asked.plan.chosen.and_then(|x| x.load), Some(p.clone()));
        let r = fit("h100", &[]).refusal.unwrap();
        assert!(r.what.contains("needs about") && r.what.contains("at penguin"), "{r:?}");
        let rel = model_dir("qx-release", "Qwen3.8-Flash-Next", &text);
        header_only_checkpoint(&rel, &json::to_compact(&released));
        let release = |machine: &str| planned(&args(&["fit", &format!("{}:penguin", rel.to_str().unwrap()), "--machine", machine])).unwrap();
        let f = release("8xh100");
        let p = f.plan.chosen.as_ref().and_then(|c| c.load.clone()).unwrap_or_else(|| panic!("{:?}", f.refusal));
        let sp = p.spread.as_ref().unwrap();
        assert!(sp.tables == tables && sp.devices.len() > 1 && p.gpu_total() <= p.free, "{p:?}");
        assert!(p.weights_on_gpu() + tables <= size(&released, &|_| true), "the tables are not among the GPUs' bytes");
        let rest = (size(&released, &|_| true) - tables) as f64 / (1u64 << 30) as f64;
        for machine in ["rtx4080s", "l4"] {
            let r = release(machine).refusal.unwrap_or_else(|| panic!("{machine}: planned"));
            let need = r.what.split("needs about ").nth(1).and_then(|w| w.split(" GiB").next()).and_then(|g| g.parse::<f64>().ok()).unwrap();
            // (the GPU memory it asks is the release's but its tables, 238.7 GiB of bf16 (its experts' 229.7 GiB), packed: not the 335.3 GiB of the files)
            assert!(r.what.contains("of GPU memory at penguin") && need > 0.7 * rest && need < rest + 0.1, "{machine}: {r:?}");
        }
        let _ = (std::fs::remove_dir_all(dir.parent().unwrap()), std::fs::remove_dir_all(rel.parent().unwrap()));
    }

    /// The release's codes pack `name` (tests/fixtures/qwen4x_packs.json: glyd/Qwen3.8-Flash-Next-hummingbird's, each layer's count of each format) written as a
    /// pack's meta.json in `dir`, its signs all +1: the planner reads a pack's formats alone.
    fn release_pack(dir: &std::path::Path, name: &str) {
        let v: Value = json::parse(&fixture("qwen4x_packs.json")).unwrap();
        let p = v.get("packs").and_then(|x| x.get(name)).unwrap_or_else(|| panic!("no pack {name}"));
        let hidden = p.get("hidden").and_then(Value::as_u64).unwrap() as usize;
        let mut layers = serde_json::Map::new();
        for (l, counts) in p.get("layers").and_then(Value::as_array).unwrap().iter().enumerate() {
            let Value::Object(c) = counts else { panic!("layer {l}") };
            let f: Vec<&str> = c.iter().flat_map(|(fmt, n)| std::iter::repeat_n(fmt.as_str(), n.as_u64().unwrap() as usize)).collect();
            layers.insert(l.to_string(), serde_json::json!({"file": format!("layer-{l:03}.safetensors"), "s": vec![1; hidden], "f": f}));
        }
        let mut meta =
            serde_json::json!({"format": "glyd-x/1", "hidden": hidden, "intermediate": p.get("intermediate").and_then(Value::as_u64), "layers": layers});
        if let Some(q) = p.get("q").and_then(Value::as_u64) {
            meta["q"] = q.into();
        }
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("meta.json"), meta.to_string()).unwrap();
    }

    /// The release's codes packs beside its shapes (its config and its files' headers) at the hummingbird level (`ExpertCache::Codes`): `fit` plans the RTX 4080
    /// SUPER (16 GB) and the L4 (24 GB) with each pack (precise-v2, best-v2, fast) on one GPU, the always-on matrices and places a layer of the pack's largest
    /// expert on it, every expert of the model's layers in the computer's memory as the pack's files mapped (the L4's 14 GiB free holds part of them, the files the
    /// rest), within the GPU's free memory; the plan `run` loads (one GPU, its places, the pack's bytes beside: `split::open_plan`). Penguin, every expert on the
    /// GPU, stays refused there, and with no level asked `fit` takes the hummingbird level.
    #[test]
    fn the_newest_qwen_familys_packs_are_planned_on_16_and_24_gb_cards() {
        let text = fixture("qwen4x_config.json");
        let nl = crate::dynamic::split::config_num(&text, "num_hidden_layers").unwrap();
        let dir = model_dir("qx-packs", "Qwen3.8-Flash-Next", &text);
        header_only_checkpoint(&dir, &fixture("qwen4x_tensors.json"));
        let hb = format!("{}:hummingbird", dir.to_str().unwrap());
        for pack in ["precise-v2", "best-v2", "fast"] {
            release_pack(&dir.join("pack"), pack);
            let cb = crate::cb_pack::CbPack::open_dims(&dir.join("pack"), 2560, 640).unwrap();
            let (place, held) = ((0..nl).map(|l| cb.place_bytes(l)).max().unwrap() as u64, (0..nl).map(|l| cb.layer_bytes(l) as u64).sum::<u64>());
            for machine in ["rtx4080s", "l4"] {
                let f = planned(&args(&["fit", &hb, "--machine", machine])).unwrap();
                let p = f.plan.chosen.as_ref().and_then(|c| c.load.clone()).unwrap_or_else(|| panic!("{pack} on {machine}: {:?}", f.refusal));
                let (sp, c) = (p.spread.as_ref().unwrap(), p.experts.unwrap());
                let m = Machine::preset(machine).unwrap();
                assert_eq!(
                    (p.family, p.level, sp.devices.as_slice(), sp.cuts.len(), sp.experts_per_layer),
                    ("Qwen3.8-Flash-Next", "hummingbird", &[0][..], 0, 512)
                );
                assert!(c.slots >= 1 && c.slots < 512 && sp.places == [c.slots], "{pack} on {machine}: {} places a layer", c.slots);
                assert_eq!((sp.host_bytes, sp.gpu_bytes - (nl * c.slots) as u64 * place), (held, p.weights - held - sp.tables), "{pack} on {machine}");
                assert!(p.gpu_total() <= p.free && p.free == m.devices[0].free, "{pack} on {machine}: {p:?}");
                // (one more place a layer does not fit)
                assert!(p.gpu_total() + nl as u64 * place > p.free, "{pack} on {machine}: room for more places");
                assert_eq!(c.ram, held.min(m.ram_free - HOST_MARGIN), "{pack} on {machine}: the computer's memory holds what it can");
                eprintln!(
                    "{pack} on {machine}: {} places a layer of {:.2} MB, {:.2} GiB on the GPU of {:.2} free, {:.2} GiB of experts ({:.2} in memory)",
                    c.slots,
                    place as f64 / 1e6,
                    (p.gpu_total() - p.reserve) as f64 / (1u64 << 30) as f64,
                    p.free as f64 / (1u64 << 30) as f64,
                    held as f64 / (1u64 << 30) as f64,
                    c.ram as f64 / (1u64 << 30) as f64
                );
                let penguin = planned(&args(&["fit", &format!("{}:penguin", dir.to_str().unwrap()), "--machine", machine])).unwrap();
                assert!(penguin.refusal.is_some_and(|r| r.what.contains("needs about")), "{pack} on {machine}: penguin planned");
                let any = planned(&args(&["fit", dir.to_str().unwrap(), "--machine", machine])).unwrap();
                assert_eq!(any.plan.chosen.map(|c| c.level), Some("hummingbird"), "{pack} on {machine}");
            }
        }
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
    }

    /// The release's shapes and its precise-v2 pack beside them (a planning directory: headers, no bytes), its config, its table and its pack.
    fn qx_release(tag: &str) -> (std::path::PathBuf, ModelConfig, crate::sizes::Table, std::path::PathBuf) {
        let text = fixture("qwen4x_config.json");
        let dir = model_dir(tag, "Qwen3.8-Flash-Next", &text);
        header_only_checkpoint(&dir, &fixture("qwen4x_tensors.json"));
        release_pack(&dir.join("pack"), "precise-v2");
        let table = crate::sizes::Table::from_dir(&dir).unwrap();
        (dir.clone(), mc(&text), table, dir.join("pack"))
    }

    /// A GPU of `gib` GiB free, numbered `ordinal`, in a computer of `host` bytes free.
    fn free_gpu(ordinal: i32, gib: u64, host: u64) -> Target {
        Target { ordinal, name: format!("GPU of {gib} GiB"), cc: (8, 9), total: gib << 30, free: gib << 30, host: Some(host), cublas: None }
    }

    /// A codes pack's plan is the one expert-cache rule's (`expert_cache_with` of the pack's source, `codes_experts`, as every other source's): `run`'s places
    /// are the rule's on the GPU's room past its floor and the scratch, beside the users' keys, values and states; a computer whose memory holds less than a
    /// token's experts in every layer is refused by the rule in words (the plan's own rule planned it, reading every expert from the disk); over two GPUs named
    /// (Condor's server's `--gpus 0,1`) the rule cuts the layers where one GPU leaves fewer places, each part its own places and its share of one sequence; the
    /// cut and the places asked (`--split`, `--places`) are held, or refused in the plan's words.
    #[test]
    fn a_codes_packs_plan_is_the_one_expert_cache_rules() {
        let (dir, c, table, pack) = qx_release("qx-rule");
        let row = crate::support::row_named("Qwen3.8-Flash-Next").unwrap();
        let name = "Qwen3.8-Flash-Next";
        let nl = c.num("num_hidden_layers").unwrap();
        let host = 64u64 << 30;
        let l4 = [free_gpu(0, 22, host)];
        let p = sizing::codes_plan(&args(&["serve", name, "--users", "4"]), name, &table, &c, row, &pack, &l4, Some(host)).unwrap();
        let cb = crate::cb_pack::CbPack::open_dims(&pack, 2560, 640).unwrap();
        let src = sizing::codes_experts(&crate::dynamic::tables::off_gpu(&table, &c).0, &c, &cb, nl, crate::dynamic::split::TIERED_PCT, None).unwrap();
        let mut kv = vec![0; nl];
        kv[0] = p.kv;
        let room = [(0, (22u64 << 30) - CONTEXT_FLOOR - sizing::CACHE_SCRATCH)];
        let ask = sizing::CacheAsk { least: Some(1), ..Default::default() };
        let rule = sizing::expert_cache_with(&src, ask, &room, &kv, Some(host)).unwrap();
        let sp = p.spread.as_ref().unwrap();
        assert_eq!((p.users, &sp.places, &sp.cuts, rule.cuts.len()), (4, &rule.places, &Vec::new(), 0));
        assert!(p.gpu_total() <= p.free && rule.places[0] > 1 && rule.places[0] < 512, "{p:?}");
        // (a computer of too little memory for a token's experts in every layer: the rule refuses it)
        let r = sizing::codes_plan(&args(&["run", name]), name, &table, &c, row, &pack, &l4, Some(HOST_MARGIN + (1 << 20))).unwrap_err();
        assert!(r.what.contains("keeps at least") && r.what.contains("in the computer's memory"), "{r:?}");
        // (two GPUs of 8 GiB named: the layers cut over them by the rule, each part its places, one sequence at a time)
        let ask = |split: Option<Vec<usize>>, places: Option<usize>| sizing::CodesAsk {
            ctx: 8192,
            users: 4,
            served: true,
            chunk: 1024,
            layout: crate::weights::Layout::Int4,
            hi: None,
            sides: Vec::new(),
            cut: true,
            split,
            places,
            host_memory: None,
        };
        let two = [free_gpu(0, 8, host), free_gpu(1, 8, host)];
        let plan = |k: &sizing::CodesAsk| sizing::codes_plan_with(name, &table, &c, row, &pack, &two, Some(host), k);
        let cut = plan(&ask(None, None)).unwrap();
        let s = cut.spread.as_ref().unwrap();
        assert!(s.cuts.len() == 1 && s.devices == [0, 1] && s.places.len() == 2 && s.places.iter().all(|&x| x >= 1), "{s:?}");
        assert!(cut.users == 1 && cut.layout == crate::weights::Layout::Int4 && cut.gpu_total() <= cut.free && cut.free == 16 << 30, "{cut:?}");
        // (the cut asked and the places asked held; more places than the GPUs hold refused in the plan's words)
        let at = plan(&ask(Some(vec![30]), Some(3))).unwrap();
        assert_eq!((at.spread.as_ref().unwrap().cuts.clone(), at.spread.as_ref().unwrap().places.clone()), (vec![30], vec![3, 3]));
        let r = plan(&ask(None, Some(500))).unwrap_err();
        assert!(r.what.contains("no room for 500 places a layer"), "{r:?}");
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
    }

    /// GLYD_ENGINE_BATCH_PREFILL (a product's switch: at 0 prompts run one a pass and no finished sequence is kept) is read in one place, which the engine's
    /// steps, the plan's sequences and the checks share (`dynamic::batch_prefill`): `fit --users 8` plans what `serve --users 8` does with it at 0 and at 1,
    /// and each sequence's states have a copy at its mark at 1 alone. On a model with states, the release's shapes with its codes pack (planned as `glyd serve`
    /// and Condor's server plan it).
    #[test]
    fn fit_and_serve_plan_alike_with_the_batch_prefill_switch_at_0_and_1() {
        let (dir, c, table, pack) = qx_release("qx-switch");
        let row = crate::support::row(crate::support::Family::FlashNext);
        let host = 64u64 << 30;
        let devs = [free_gpu(0, 22, host)];
        let model = dir.to_str().unwrap();
        let plan = |cmd: &str, on: bool| {
            crate::dynamic::BATCH_PREFILL_ASKED.with(|a| a.set(Some(on)));
            let a = args(&[cmd, model, "--users", "8", "--context", "4096"]);
            let p = sizing::codes_plan(&a, "m", &table, &c, row, &pack, &devs, Some(host));
            crate::dynamic::BATCH_PREFILL_ASKED.with(|a| a.set(None));
            p.unwrap_or_else(|r| panic!("{cmd} at {on}: {}", r.what))
        };
        let (off, on) = (plan("serve", false), plan("serve", true));
        assert_eq!(plan("fit", false), off, "the switch at 0");
        assert_eq!(plan("fit", true), on, "the switch at 1");
        assert_eq!((off.users, on.users), (8, 8));
        assert!(off.kv < on.kv, "the marks' copies at 1 alone: {} and {}", off.kv, on.kv);
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
    }

    /// Condor's server (`qwenserve`) loads by its plan (`qwen4x::serve_plan`, the planned route's rule): its users and context, the always-on matrices at its
    /// level (four bits: more places than the lossless layout leaves on the same GPU), `--places` held; a GPU named twice shares its memory between its two
    /// parts; a GPU too small, and a cut that does not name a layer for each GPU after the first, refused in words.
    #[test]
    fn condors_server_loads_by_its_plan() {
        let (dir, _, _, pack) = qx_release("qx-serve");
        let host = 64u64 << 30;
        let words = |more: &[&str]| -> crate::args::Args {
            let mut w = vec!["qwenserve", dir.to_str().unwrap(), pack.to_str().unwrap(), "--port", "8080"];
            w.extend(more);
            args(&w)
        };
        let serve_for = |users: usize, a: &crate::args::Args, devs: &[Target], layout: crate::weights::Layout| {
            let form = crate::dynamic::qwen4x::Form { mode: crate::qwen4x::PackMode::Precise, sides: Vec::new(), layout, hi: None };
            crate::dynamic::qwen4x::serve_plan(&dir, &ModelConfig::load(&dir).unwrap(), &pack, &form, a, (32768, users, 1024), devs, Some(host))
        };
        let serve = |a: &crate::args::Args, devs: &[Target], layout: crate::weights::Layout| serve_for(4, a, devs, layout);
        let l4 = [free_gpu(0, 22, host)];
        let p = serve(&words(&[]), &l4, crate::weights::Layout::Int4).unwrap();
        let sp = p.spread.as_ref().unwrap();
        assert_eq!((p.users, p.context, p.chunk, p.layout, p.level, sp.cuts.len()), (4, 32768, 1024, crate::weights::Layout::Int4, "hummingbird", 0));
        assert!(p.gpu_total() <= p.free && sp.places[0] >= 1, "{p:?}");
        let lossless = serve(&words(&[]), &l4, crate::weights::Layout::Tiered).unwrap();
        assert!(lossless.spread.as_ref().unwrap().places[0] < sp.places[0], "{lossless:?}");
        let held = serve(&words(&["--places", "7"]), &l4, crate::weights::Layout::Int4).unwrap();
        assert_eq!(held.spread.as_ref().unwrap().places, [7]);
        // (sixteen users, past the release's eight: the plan holds them where the GPU does (a 48 GiB one), and names the most a 22 GiB one holds)
        assert_eq!(serve_for(16, &words(&[]), &[free_gpu(0, 48, host)], crate::weights::Layout::Int4).unwrap().users, 16);
        let r = serve_for(16, &words(&[]), &l4, crate::weights::Layout::Int4).unwrap_err().msg;
        assert!(r.contains("Ask for fewer users (--users "), "{r}");
        // (one GPU named twice: two parts, each half of its memory past the floor)
        let twice = serve(&words(&["--split", "24"]), &[free_gpu(0, 22, host), free_gpu(0, 22, host)], crate::weights::Layout::Int4).unwrap();
        let t = twice.spread.as_ref().unwrap();
        assert!(t.cuts == [24] && t.devices == [0, 0] && twice.users == 1 && twice.free == 22 << 30 && twice.gpu_total() <= twice.free, "{twice:?}");
        // (too small a GPU, and a cut of no layer for the second GPU: refused in words)
        let r = serve(&words(&[]), &[free_gpu(0, 2, host)], crate::weights::Layout::Int4).unwrap_err().msg;
        assert!(r.contains("no room for a place a layer for its experts") && r.contains("Ask for less context"), "{r}");
        let r = serve(&words(&["--split", "24"]), &l4, crate::weights::Layout::Int4).unwrap_err().msg;
        assert!(r.contains("--split 24: 0 first layers in order"), "{r}");
        // (a GPU that holds one user's context and places but not four's: the refusal names the most users it holds)
        let int4 = crate::weights::Layout::Int4;
        let g = (2..24).find(|&g| serve_for(1, &words(&[]), &[free_gpu(0, g, host)], int4).is_ok()).unwrap();
        let four = (g..24).find(|&g| serve(&words(&[]), &[free_gpu(0, g, host)], int4).is_ok()).unwrap();
        assert!(four > g, "one user and four fit from {g} GiB");
        let r = serve(&words(&[]), &[free_gpu(0, four - 1, host)], int4).unwrap_err().msg;
        assert!(r.contains("Ask for fewer users (--users "), "{r}");
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
    }

    /// Condor's server's `--hi` is planned where the loader puts it (`Placing::hi`): the release's attention blocks' own matrices and its LM head
    /// (`Described::attention_and_head`: the head and each full-attention layer's q, k, v, o and indexer products, nothing of a Gated DeltaNet layer, a gated
    /// residual or the experts) at the second layout's share of their bf16 bytes, every other at the layout's; the default form's plan (four bits, five for
    /// those) holds those bytes more and no more places than four bits alone. The plan opens the sidecars the form names (`--no-side`: none).
    #[test]
    fn condors_server_plans_its_second_layout_and_sidecars() {
        use crate::weights::Layout::{Int4, Int5};
        let (dir, c, table, pack) = qx_release("qx-hi");
        let (nl, every) = (c.num("num_hidden_layers").unwrap(), c.num("full_attention_interval").unwrap());
        let names = crate::runtime::describe::describe_config(&c).unwrap().attention_and_head();
        let products: Vec<&String> = names.iter().filter(|n| n.ends_with("_proj.weight") || *n == "lm_head.weight").collect();
        assert_eq!(products.len(), nl / every * 5 + 1, "{products:?}");
        assert!(names.contains("lm_head.weight") && names.iter().all(|n| n == "lm_head.weight" || n.contains(".self_attn.")), "{names:?}");
        let cb = crate::cb_pack::CbPack::open_dims(&pack, 2560, 640).unwrap();
        let on_gpu = crate::dynamic::tables::off_gpu(&table, &c).0;
        let (p4, p5) = (sizing::layout_pct(Int4), sizing::layout_pct(Int5));
        let always = |hi| {
            let s = sizing::codes_experts(&on_gpu, &c, &cb, nl, p4, hi).unwrap();
            s.dense.iter().sum::<u64>() + s.base
        };
        let matrix = |r: &crate::sizes::Row| r.dtype == "BF16" && r.shape.len() == 2 && r.shape.iter().all(|&d| d >= 256);
        let more: u64 = on_gpu.rows.iter().filter(|r| names.contains(&r.name) && matrix(r)).map(|r| r.bytes * p5 / 100 - r.bytes * p4 / 100).sum();
        assert!(more > 100 << 20 && always(Some((p5, &names))) == always(None) + more, "{more}");
        let host = 64u64 << 30;
        let a = args(&["qwenserve", dir.to_str().unwrap(), pack.to_str().unwrap(), "--port", "8080"]);
        let plan = |hi, sides: Vec<std::path::PathBuf>| {
            let form = crate::dynamic::qwen4x::Form { mode: crate::qwen4x::PackMode::Fast, sides, layout: Int4, hi };
            crate::dynamic::qwen4x::serve_plan(&dir, &c, &pack, &form, &a, (32768, 4, 1024), &[free_gpu(0, 22, host)], Some(host)).unwrap()
        };
        let (with, without) = (plan(Some(Int5), Vec::new()), plan(None, Vec::new()));
        assert_eq!(with.weights, without.weights + more);
        assert!(with.spread.as_ref().unwrap().places[0] <= without.spread.as_ref().unwrap().places[0]);
        // (the Monitor's places, as v0.34.2's server gave them: the plan's places a layer and their GB, a place the pack's largest expert)
        let form = crate::dynamic::qwen4x::Form { mode: crate::qwen4x::PackMode::Fast, sides: Vec::new(), layout: Int4, hi: Some(Int5) };
        let sp = with.spread.as_ref().unwrap();
        let (gb, per) = crate::dynamic::qwen4x::monitor_places(sp, &c, &pack, &form).unwrap();
        let place = (0..nl).map(|l| cb.place_bytes(l)).max().unwrap();
        assert!(per == sp.places[0] && per > 0 && (gb - (per * nl * place) as f64 / 1e9).abs() < 1e-9, "{gb} GB, {per} a layer");
        // (a sidecar the form names is opened: one that is not there is refused, none is the pack's own)
        let r = crate::dynamic::qwen4x::serve_plan(
            &dir,
            &c,
            &pack,
            &crate::dynamic::qwen4x::Form { mode: crate::qwen4x::PackMode::Fast, sides: vec![pack.join("hot")], layout: Int4, hi: None },
            &a,
            (32768, 4, 1024),
            &[free_gpu(0, 22, host)],
            Some(host),
        );
        assert!(r.unwrap_err().msg.contains("meta.json"));
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
    }

    /// The tables read by rows (an n-gram block's: DeepSeek-V4.1's Engram tables, Qwen3.8-Flash-Next's n-gram tables) are not parameters a token reads: `fit`'s
    /// line counts them apart from the active ones, every other model's figure as it was (GLM-5.3, no such tables).
    #[test]
    fn tables_read_by_rows_are_not_active_parameters() {
        let line = |name: &str, preset: &str, text: &str, table: &crate::sizes::Table| -> (Shape, String) {
            let dir = model_dir("tables-active", name, text);
            table.write(&dir.join(crate::sizes::GGUF_TENSORS));
            let f = planned(&args(&["fit", dir.to_str().unwrap(), "--machine", preset])).unwrap();
            let t = super::text(name, &f.shape, &f.machine, &f.plan);
            let _ = std::fs::remove_dir_all(&dir);
            eprintln!("{}", t.lines().next().unwrap());
            (f.shape, t.lines().next().unwrap().to_string())
        };
        let (text, table) = recorded_release("DeepSeek-V4.1-Flash-1017");
        let (sh, l) = line("DeepSeek-V4.1-Flash-1017", "8xh100", &text, &table);
        // (its two Engram tables in FP8, a byte a parameter; their block scales carried)
        assert_eq!(sh.tables, (98_305_579_008u64 + 98_308_270_592) as f64);
        assert!((sh.active() - (sh.params() - sh.routed + sh.routed_active() - sh.tables)).abs() < 1.0);
        let want = format!("({:.1}B of them tables a token reads a few rows of), {:.1}B active", sh.tables / 1e9, sh.active() / 1e9);
        assert!(l.contains(&format!("MoE, {:.1}B parameters {want}", sh.params() / 1e9)) && !l.contains("227.7B active"), "{l}");
        let (sh, l) = line("GLM-5.3", "8xh100", &fixture("glm_moe_dsa_config.json"), &glm53_table());
        assert!(sh.tables == 0.0 && !l.contains("tables"), "{l}");
        assert!(l.contains(&format!("{:.1}B active", (sh.params() - sh.routed + sh.routed_active()) / 1e9)), "{l}");
        // (Qwen3.8-Flash-Next's n-gram tables: the same kind, the same count)
        let qx = fixture("qwen4x_config.json");
        let names = crate::dynamic::tables::ngram_names(&mc(&qx));
        assert!(!names.is_empty() && names.iter().all(|n| n.contains("ngram")), "{names:?}");
        let (dir, ..) = qx_release("tables-active-qx");
        let f = planned(&args(&["fit", dir.to_str().unwrap(), "--machine", "l4"])).unwrap();
        let l = super::text("Qwen3.8-Flash-Next", &f.shape, &f.machine, &f.plan).lines().next().unwrap().to_string();
        eprintln!("{l}");
        assert!(f.shape.tables > 0.0 && l.contains(&format!("{:.1}B active", f.shape.active() / 1e9)), "{l}");
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
    }

    /// A plan never holds silently less than asked: `fit --users 4` of DeepSeek-V4.1 on eight H100s (its layers in stages over GPUs) plans one sequence, and
    /// says so in words; a plan that holds the users asked says nothing.
    #[test]
    fn a_staged_plan_says_it_holds_one_sequence() {
        let (text, table) = recorded_release("DeepSeek-V4.1-Flash-1017");
        let dir = model_dir("staged-users", "DeepSeek-V4.1-Flash-1017", &text);
        table.write(&dir.join(crate::sizes::GGUF_TENSORS));
        let f = planned(&args(&["fit", &format!("{}:penguin", dir.to_str().unwrap()), "--machine", "8xh100", "--users", "4"])).unwrap();
        let lp = f.plan.chosen.as_ref().and_then(|c| c.load.clone()).unwrap_or_else(|| panic!("{:?}", f.refusal));
        assert!(lp.users == 1 && !lp.spread.as_ref().unwrap().cuts.is_empty(), "{lp:?}");
        assert_eq!(
            super::users_note(4, &lp).as_deref(),
            Some("the plan holds 1 of the 4 users asked for at once: its layers in stages over GPUs run one sequence at a time")
        );
        assert_eq!(super::users_note(1, &lp), None);
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
        let (qdir, _, _, _) = qx_release("staged-users-qx");
        let f = planned(&args(&["fit", &format!("{}:hummingbird", qdir.to_str().unwrap()), "--machine", "l4", "--users", "4"])).unwrap();
        let lp = f.plan.chosen.as_ref().and_then(|c| c.load.clone()).unwrap();
        assert_eq!((lp.users, super::users_note(4, &lp)), (4, None));
        let _ = std::fs::remove_dir_all(qdir.parent().unwrap());
    }

    /// The dynamic route reads a family's row, never its name: a row of another name with GLM-5.3-Flash's fields plans GLM-5.3-Flash's checkpoint as its own
    /// row does, on every machine; a row that says no expert cache is refused where the cache was the plan; and a name the table has no row for is refused.
    #[test]
    fn the_dynamic_route_plans_by_the_rows_fields() {
        use crate::support::{ExpertCache, Route, Support};
        let text = fixture("glm5_next_config.json");
        let dir = model_dir("dyn-row", "GLM-5.3-Flash", &text);
        header_only_checkpoint(&dir, &fixture("glm5_next_tensors.json"));
        let glm = crate::support::row_named("GLM-5.3-Flash").unwrap();
        let row = |name: &'static str, route: Route| -> &'static Support {
            Box::leak(Box::new(Support {
                family: glm.family,
                name,
                levels: glm.levels,
                hummingbird_packed_only: glm.hummingbird_packed_only,
                default_ctx: glm.default_ctx,
                max_ctx: glm.max_ctx,
                split_levels: glm.split_levels,
                experts_in_ram_levels: glm.experts_in_ram_levels,
                drafts: glm.drafts,
                route,
                saves: glm.saves,
                lossy: glm.lossy,
            }))
        };
        let another = row("Another", glm.route);
        let table = crate::sizes::Table::from_dir(&dir).unwrap();
        let a = args(&["run", dir.to_str().unwrap()]);
        let size = |r: &'static Support, m: &Machine| {
            let devs: Vec<Target> = m.devices.iter().enumerate().map(|(i, d)| Target::of(d, i as i32, Some(m.ram_free))).collect();
            sizing::size_dynamic_row(&a, "m", &table, Some(&dir), &mc(&text), r, &devs, Some(m.ram_free), &|_| false)
        };
        for preset in ["rtx4080s", "rtx4090", "2x3090", "a100-80g", "8xh100"] {
            let m = Machine::preset(preset).unwrap();
            match (size(glm, &m), size(another, &m)) {
                (Ok(own), Ok(other)) => {
                    assert_eq!((own.family, other.family), ("GLM-5.3-Flash", "Another"));
                    assert_eq!(LoadPlan { family: "", ..own }, LoadPlan { family: "", ..other }, "{preset}");
                }
                (Err(own), Err(other)) => assert_eq!(own.what, other.what, "{preset}"),
                (own, other) => panic!("{preset}: {:?} against {:?}", own.map(|p| p.level), other.map(|p| p.level)),
            }
        }
        // (a tiny release with experts, on one GPU with room for its other weights and a third of its experts, in a computer with memory for every expert:
        // the expert cache is the plan under its family's row and under a row of another name with the same fields; a row whose field says no cache is refused)
        let tiny = std::env::temp_dir().join(format!("glyd-plan-rowcache-{}", std::process::id()));
        crate::hb_pack::tests::release(&tiny, "dsv4_model");
        let ttext = std::fs::read_to_string(tiny.join("config.json")).unwrap();
        let nl = crate::dynamic::split::config_num(&ttext, "num_hidden_layers").unwrap();
        let expert = |n: &str| n.contains(".experts.");
        let (per, _) = crate::dynamic::split::layer_bytes_where(&tiny, nl, 100, 100, &expert).unwrap();
        let (dense, base) =
            crate::dynamic::split::layer_bytes_where(&tiny, nl, crate::dynamic::split::CODED_PCT, crate::dynamic::split::TIERED_PCT, &|n| !expert(n)).unwrap();
        let all: u64 = per.iter().sum();
        let sh = Shape::from_config(&ttext).unwrap();
        let kv = ((sh.kv_full * 16.0 + sh.kv_sliding * 16f64.min(sh.window.max(1) as f64) + sh.state) / nl as f64) as u64 * nl as u64;
        let want = dense.iter().sum::<u64>() + base + kv + all / 3;
        // (the least free memory whose room holds that: the model is a fraction of a MiB, so a step of a MiB would hold all of it)
        let (_, free) = (0..40).fold((0u64, 64u64 << 30), |(lo, hi), _| {
            let mid = lo + (hi - lo) / 2;
            if crate::plan::dynamic_room_of(mid) >= want {
                (lo, mid)
            } else {
                (mid + 1, hi)
            }
        });
        let gpu = Target { ordinal: 0, name: "NVIDIA GeForce RTX 4080 SUPER".into(), cc: (8, 9), total: free, free, host: None, cublas: Some(false) };
        let ttable = crate::sizes::Table::from_dir(&tiny).unwrap();
        let v4 = crate::support::row_named("DeepSeek-V4").unwrap();
        let (renamed, bare) = (row("Renamed", v4.route), row("Bare", Route { expert_cache: ExpertCache::None, ..v4.route }));
        let cache = |r: &'static Support| {
            let a = args(&["run", "tiny", "--context", "16"]);
            sizing::size_dynamic_row(&a, "tiny", &ttable, Some(&tiny), &mc(&ttext), r, std::slice::from_ref(&gpu), Some(all + HOST_MARGIN + (1 << 30)), &|_| {
                false
            })
        };
        let (own, other) = (cache(v4).unwrap(), cache(renamed).unwrap());
        assert!(own.experts.is_some_and(|c| c.host_bytes() == all) && own.spread.as_ref().unwrap().places.iter().all(|&p| p > 0), "{own:?}");
        assert_eq!(LoadPlan { family: "", ..own }, LoadPlan { family: "", ..other });
        assert!(cache(bare).unwrap_err().what.contains("of GPU memory at penguin"));
        std::fs::remove_dir_all(&tiny).unwrap();
        let r = sizing::size_dynamic(&a, "m", &table, Some(&dir), &mc(&text), sizing::static_family("No such family"), &[], None, &|_| false).unwrap_err();
        assert!(r.what.contains("no family named"), "{r}");
        for f in ["Kimi-K3", "GLM-5.3-Flash", "DeepSeek-V4", "Muse Spark", "Nemotron-H", "Qwen3.5", "Gemma 4"] {
            assert_eq!(sizing::static_family(f), f);
        }
    }

    /// A plan's draft gain is its family's row's flag, whichever engine path loads the plan: the same model at the same placement makes `MTP_GAIN` tokens a
    /// step under a row that drafts and one under a row that does not.
    #[test]
    fn a_plans_draft_gain_is_its_family_rows() {
        let (_, q08) = measured_shapes();
        let sh = Shape { mtp: true, ..q08 };
        let m = Machine::preset("a100-80g").unwrap();
        let drafting = crate::support::TABLE.iter().find(|r| r.drafts).unwrap().name;
        let plain = crate::support::TABLE.iter().find(|r| !r.drafts).unwrap().name;
        for kind in [Kind::Dynamic, Kind::Qwen35] {
            let of = |family: &'static str| LoadPlan { family, ..one_gpu_plan(kind, "kestrel") };
            let gain = |family| features(&sh, &m, 0, &of(family), 1000.0, 1)["gain"].as_f64().unwrap();
            assert_eq!((gain(drafting), gain(plain), gain("")), (MTP_GAIN, 1.0, 1.0), "{kind:?}");
            let (fast, slow) = (speed_users(&sh, &m, 0, &of(drafting), 1000.0, 1), speed_users(&sh, &m, 0, &of(plain), 1000.0, 1));
            assert!((fast / slow - MTP_GAIN).abs() < 1e-9, "{kind:?}: {fast} against {slow}");
        }
    }

    /// The matrices the planner counts in the computer's memory for a layer there (its row's `ram_linears`) are the ones the executor's loader puts there: every
    /// matrix of the layer's own products, as its family's lowered description names them (`dynamic::layer_tensors`), on Kimi-K3's released tensors (its
    /// delta and latent attention layers and its dense MLP) and on the random models of Kimi's two shapes and Muse Spark; so a plan frees on the GPU, and
    /// holds in the computer's memory, what the load does. A row that keeps none there counts none (GLM-5.3-Flash's, whose random models load by the same
    /// loader, which keeps none there unless the plan asks; Nemotron-H's, whose loader keeps none there; the Qwen3.5 layout with experts' and GLM-5.3's, whose
    /// loaders hold every expert on its GPU, at penguin, the one level their plans offer).
    #[test]
    fn the_matrices_counted_in_memory_are_the_loaders() {
        use crate::dynamic::{layer_tensors, Stage};
        use std::collections::BTreeSet;
        let by_layer = |table: &crate::sizes::Table, nl: usize, family: &str| -> Vec<BTreeSet<String>> {
            let mut v = vec![BTreeSet::new(); nl];
            for r in table.rows.iter().filter(|r| sizing::in_ram(family, &r.name)) {
                if let Some(l) = crate::sizes::layer_of(&r.name).filter(|&l| l < nl) {
                    v[l].insert(r.name.clone());
                }
            }
            v
        };
        let loaders =
            |d: &crate::dynamic::ModelDesc| -> Vec<BTreeSet<String>> { (0..d.layers.len()).map(|l| layer_tensors(d, l).into_iter().collect()).collect() };
        let h: serde_json::Value = serde_json::from_str(&fixture("headers/Kimi-K3.json")).unwrap();
        let text = h["config"].as_str().unwrap();
        let c = crate::kimi::ConfigKimi::parse(text, "config.json", None).unwrap();
        let kinds = crate::runtime::describe::describe(&serde_json::from_str(text).unwrap()).unwrap();
        let w32 = vec![(true, true); c.layers];
        let low = crate::dynamic::lower::kimi(&kinds, c.eos.clone(), c.max_pos, &Stage::whole(c.layers), &w32, false).unwrap();
        let table = crate::sizes::Table::expand(&header_groups(&h));
        let (got, want) = (by_layer(&table, c.layers, "Kimi-K3"), loaders(&low));
        assert_eq!(got, want);
        // (every layer keeps its attention there, a latent attention layer's too; the bytes are the table's of those tensors)
        let bytes = sizing::ram_bytes_table(&table, c.layers, "Kimi-K3", 90, 72);
        assert!(bytes.iter().all(|&b| b > 0) && c.kinds.iter().any(|&k| k == crate::glm::GlmKind::Mla), "{bytes:?}");
        assert!(sizing::ram_bytes_table(&table, c.layers, "DeepSeek-V4", 90, 72).iter().all(|&b| b == 0));
        #[cfg(feature = "dev")]
        {
            let dir = std::env::temp_dir().join(format!("glyd-plan-ram-{}", std::process::id()));
            for m in ["kimi", "kimi-dsa", "muse"] {
                let _ = std::fs::remove_dir_all(&dir);
                let (family, low) = match m {
                    "muse" => {
                        crate::muse_ref::random(&dir).unwrap();
                        let (kinds, _) = crate::runtime::describe::describe_dir(&dir).unwrap();
                        let nl = crate::muse::ConfigMuse::load(&dir).unwrap().layers;
                        ("Muse Spark", crate::dynamic::muse::lowered(&dir, &kinds, &Stage::whole(nl)).unwrap())
                    }
                    _ => {
                        if m == "kimi" {
                            crate::kimi_ref::random(&dir, 1, true, 8).unwrap();
                        } else {
                            crate::kimi_ref::random_dsa(&dir).unwrap();
                        }
                        let c = crate::kimi::ConfigKimi::load(&dir).unwrap();
                        let w32 = crate::dynamic::kimi::dtypes(&glyd_gpu::safetensors::Checkpoint::open(&dir).unwrap(), &c).unwrap();
                        ("Kimi-K3", crate::dynamic::kimi::lowered(&dir, &c, &Stage::whole(c.layers), &w32, false).unwrap())
                    }
                };
                let table = crate::sizes::Table::from_dir(&dir).unwrap();
                assert_eq!(by_layer(&table, low.layers.len(), family), loaders(&low), "{m}");
            }
            // (GLM-5.3-Flash loads by the same loader, which keeps a layer's matrices there only where the plan says (`Placing::ram`, the cache's `dense`); its
            // row keeps none there, so its plans ask for none and count none, both random models, bf16 and with FP8 experts)
            for fp8 in [false, true] {
                let _ = std::fs::remove_dir_all(&dir);
                crate::glm::random(&dir, fp8, None).unwrap();
                let c = crate::glm::ConfigGlm::load(&dir).unwrap();
                let low = crate::dynamic::glm::lowered(&dir, &c, &Stage::whole(c.layers)).unwrap();
                let table = crate::sizes::Table::from_dir(&dir).unwrap();
                assert!(loaders(&low).iter().all(|l| !l.is_empty()), "every layer multiplies by matrices of its own");
                assert_eq!(by_layer(&table, c.layers, "GLM-5.3-Flash"), vec![BTreeSet::new(); c.layers], "fp8 {fp8}");
                assert!(sizing::ram_bytes_table(&table, c.layers, "GLM-5.3-Flash", 90, 72).iter().all(|&b| b == 0), "fp8 {fp8}");
            }
            // (Nemotron-H loads by the same loader with no layer there (`Placing::default`): its row keeps none there and counts none; its Mamba-2 and
            // attention layers multiply by matrices of their own, a mixture-of-experts layer by its block's alone)
            let _ = std::fs::remove_dir_all(&dir);
            crate::kcheck_nh::random(&dir, 0, 16).unwrap();
            let c = crate::nemotron::ConfigNemotronH::load(&dir).unwrap();
            let low = crate::dynamic::nemotron::lowered(&dir, &c, &Stage::whole(c.layers)).unwrap();
            let own: Vec<bool> = c.kinds.iter().map(|&k| k != crate::nemotron::NhKind::Moe).collect();
            assert!(own.contains(&false) && own.contains(&true));
            assert_eq!(loaders(&low).iter().map(|l| !l.is_empty()).collect::<Vec<_>>(), own);
            let table = crate::sizes::Table::from_dir(&dir).unwrap();
            assert_eq!(by_layer(&table, c.layers, "Nemotron-H"), vec![BTreeSet::new(); c.layers]);
            assert!(sizing::ram_bytes_table(&table, c.layers, "Nemotron-H", 90, 72).iter().all(|&b| b == 0));
            // (the Qwen3.5 layout with experts (the random case qwen35moe) and GLM-5.3 (glm-dsa), each by its family's loader (`qwen35::stage`,
            // `glm::stage_plain`). Their rows keep none of a layer's matrices in the computer's memory and count none; at penguin a plan is the dynamic
            // route's, every expert on the GPU where it holds them (these small models, on each machine); the Qwen3.5 layout's lossy levels are the lossy
            // route's plans, by its description, the same for `fit` and `run`)
            type Lowering = fn(&crate::runtime::describe::Described, Vec<i64>, usize, &Stage) -> crate::ffi::Result<crate::dynamic::ModelDesc>;
            let held: [(&str, fn(&std::path::Path) -> crate::ffi::Result<()>, Lowering, &str); 2] = [
                ("qwen35moe", crate::logits::qwen35moe_random, crate::dynamic::lower::qwen35, "Qwen3.5"),
                ("glm-dsa", crate::logits::glm_dsa_random, crate::dynamic::lower::latent_moe, "GLM-5.3"),
            ];
            for (case, make, lower, family) in held {
                let _ = std::fs::remove_dir_all(&dir);
                make(&dir).unwrap();
                let (kinds, _) = crate::runtime::describe::describe_dir(&dir).unwrap();
                let nl = kinds.desc.blocks.len() / 2;
                let low = lower(&kinds, vec![2], 1024, &Stage::whole(nl)).unwrap();
                let table = crate::sizes::Table::from_dir(&dir).unwrap();
                assert!(loaders(&low).iter().all(|l| !l.is_empty()), "{case}: every layer multiplies by matrices of its own");
                assert_eq!(by_layer(&table, nl, family), vec![BTreeSet::new(); nl], "{case}");
                assert!(sizing::ram_bytes_table(&table, nl, family, 90, 72).iter().all(|&b| b == 0), "{case}");
                let (model, text) = (dir.to_str().unwrap().to_string(), std::fs::read_to_string(dir.join("config.json")).unwrap());
                for preset in ["rtx4080s", "rtx4090", "a100-80g"] {
                    let f = planned(&args(&["fit", &model, "--machine", preset])).unwrap();
                    assert_eq!(f.family.name, family, "{case}");
                    let c = f.plan.chosen.as_ref().unwrap_or_else(|| panic!("{case} {preset}: {:?}", f.plan));
                    let lp = c.load.as_ref().unwrap_or_else(|| panic!("{case} {preset}: no loader plan"));
                    assert!(c.level == "penguin" && lp.kind == Kind::Dynamic && lp.experts.is_none() && lp.context > 0, "{case} {preset}: {lp:?}");
                    assert!(matches!(c.placement, Some(Placement::Gpu(_))), "{case} {preset}: {:?}", c.placement);
                    if case != "qwen35moe" {
                        assert!(f.plan.options.iter().all(|o| o.level == "penguin"), "{case} {preset}: a level its row does not offer");
                        continue;
                    }
                    let d = &Machine::preset(preset).unwrap().devices[0];
                    for level in ["kestrel", "swift", "wren"] {
                        let o = f.plan.options.iter().find(|o| o.level == level).unwrap_or_else(|| panic!("{preset} {level}: no option"));
                        let lp = o.load.as_ref().unwrap_or_else(|| panic!("{preset} {level}: {:?}", o.why));
                        let layout = crate::levels::layout_of(level).unwrap();
                        assert!(o.why.is_none() && lp.kind == Kind::Qwen35 && lp.layout == layout && lp.experts.is_none(), "{preset} {level}: {lp:?}");
                        let r = sizing::lossy_plan_with(&args(&["run", &model]), &model, layout, &mc(&text), None, d.free, &d.name, None, None).unwrap();
                        assert_eq!((r.weights, r.bf16, r.context), (lp.weights, lp.bf16, lp.context), "{preset} {level}: fit's plan is run's");
                    }
                }
            }
            // (#239: the keys, values and states a plan counts are the loader's too, on every family's random case of the dynamic route: the config's description
            // lowered as the loader lowers it before a file is here (`split::lowered_of`) has the rings and states of the loader's own lowering of the directory,
            // each layer's one sequence's bytes the same (`rows::layer_rooms`) and the sequences' rule the same (`rows::SeqRoom`); and a plan's work holds at the
            // least the activations the executor makes for its chunk's rows (each slot's width a row, bf16) and the head's logits)
            for case in DYNAMIC_CASES {
                let _ = std::fs::remove_dir_all(&dir);
                crate::logits::random_model(case, &dir).unwrap();
                let c = crate::runtime::config::ModelConfig::load(&dir).unwrap();
                let (ctx, chunk) = (4096, sizing::DYN_CHUNK);
                let planned = crate::dynamic::split::lowered_of(&c, ctx + chunk + 16).unwrap_or_else(|e| panic!("{case}: {}", e.msg));
                let loaded = loader_desc(case, &dir, ctx + chunk + 16);
                let rings = |d: &crate::dynamic::ModelDesc| d.rings.iter().map(|r| (r.d, r.nk, r.window, r.values, r.ratio)).collect::<Vec<_>>();
                assert_eq!((rings(&planned), &planned.states), (rings(&loaded), &loaded.states), "{case}: the rings and states");
                let rooms = crate::dynamic::layer_rooms(&loaded, ctx, chunk);
                assert_eq!(crate::dynamic::layer_rooms(&planned, ctx, chunk), rooms, "{case}: each layer's sequence");
                assert!(rooms.iter().sum::<u64>() > 0 && rooms.len() == c.num("num_hidden_layers").unwrap(), "{case}: {rooms:?}");
                let (p, l) = (crate::dynamic::SeqRoom::of(&planned, ctx, chunk), crate::dynamic::SeqRoom::of(&loaded, ctx, chunk));
                assert_eq!(format!("{p:?}"), format!("{l:?}"), "{case}: the sequences' rule");
                let f = planned_fit(&dir, "rtx4080s", &["--users", "8"]);
                let lp = f.plan.chosen.as_ref().and_then(|c| c.load.clone()).unwrap_or_else(|| panic!("{case}: {:?}", f.refusal));
                let acts = loaded.widths.iter().map(|&w| (w.max(1) * lp.chunk * 2) as u64).sum::<u64>() + (loaded.vocab * 2) as u64;
                assert!(lp.work >= acts, "{case}: the plan's work {} holds the executor's activations {acts}", lp.work);
            }
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// The random cases of the families on the dynamic route (src/logits.rs `random_model`).
    #[cfg(feature = "dev")]
    const DYNAMIC_CASES: [&str; 10] = ["qwen35", "qwen35moe", "glm", "glm-dsa", "kimi", "kimi-dsa", "deepseek-v4", "muse", "qwen4x", "nemotron"];

    /// The description each family's loader of the dynamic route runs the model in `dir` by (its stage loader's lowering, of the directory: `split::loader`), the
    /// whole model, for sequences of up to `max_pos` positions.
    #[cfg(feature = "dev")]
    fn loader_desc(case: &str, dir: &std::path::Path, max_pos: usize) -> crate::dynamic::ModelDesc {
        use crate::dynamic::Stage;
        let (kinds, _) = crate::runtime::describe::describe_dir(dir).unwrap();
        let d = match case {
            // (the dense layout by its reader's config, the layout with experts by its kinds alone: `qwen35::load_stage`)
            "qwen35" | "qwen35-wide" | "qwen35moe" => match crate::qwen35::Config35::load(dir) {
                Ok(c) => crate::dynamic::qwen35::lowered(dir, &c, &Stage::whole(c.layers)),
                Err(_) => crate::dynamic::lower::qwen35(&kinds, Vec::new(), max_pos, &Stage::whole(kinds.desc.blocks.len() / 2)),
            },
            "glm" => {
                let c = crate::glm::ConfigGlm::load(dir).unwrap();
                crate::dynamic::glm::lowered(dir, &c, &Stage::whole(c.layers))
            }
            "glm-dsa" => crate::dynamic::lower::latent_moe(&kinds, Vec::new(), max_pos, &Stage::whole(kinds.desc.blocks.len() / 2)),
            "kimi" | "kimi-dsa" => {
                let c = crate::kimi::ConfigKimi::load(dir).unwrap();
                let w32 = crate::dynamic::kimi::dtypes(&glyd_gpu::safetensors::Checkpoint::open(dir).unwrap(), &c).unwrap();
                crate::dynamic::kimi::lowered(dir, &c, &Stage::whole(c.layers), &w32, false)
            }
            "deepseek-v4" => {
                let (nl, eos, mp) = crate::dynamic::deepseek_v4::ends(dir).unwrap();
                let vo = crate::dynamic::deepseek_v4::V4Opts::of(crate::weights::Layout::Tiered, max_pos);
                crate::dynamic::lower::deepseek_v4(&kinds, eos, mp, &Stage::whole(nl), &vo)
            }
            "muse" => crate::dynamic::muse::lowered(dir, &kinds, &Stage::whole(crate::muse::ConfigMuse::load(dir).unwrap().layers)),
            "qwen4x" => crate::dynamic::qwen4x::lowered(dir, max_pos),
            "nemotron" => {
                let c = crate::nemotron::ConfigNemotronH::load(dir).unwrap();
                crate::dynamic::nemotron::lowered(dir, &c, &Stage::whole(c.layers))
            }
            other => panic!("{other}: no loader"),
        };
        d.unwrap_or_else(|e| panic!("{case}: {}", e.msg))
    }

    /// `fit` of the model in `dir` at penguin on the preset `machine`, with `more` words.
    #[cfg(feature = "dev")]
    fn planned_fit(dir: &std::path::Path, machine: &str, more: &[&str]) -> Planned {
        let tag = format!("{}:penguin", dir.display());
        let mut w = vec!["fit", tag.as_str(), "--machine", machine];
        w.extend(more);
        planned(&args(&w)).unwrap_or_else(|r| panic!("{}: {r:?}", dir.display()))
    }

    /// The dynamic route's plans count their sequences (#239, item 1): a server's `--users` held in the pools of what the GPU leaves them, by the executor's rule
    /// (`rows::SeqRoom` of the loader's description: its bytes less their states, marks and block 0 hold one sequence's context at the least, every user's at
    /// the most), and within what is free; a run's one sequence, its rings and states as loaded. On every family's random case of the route, on an RTX 4080
    /// SUPER and an L4 (the M2 gate's GPUs) and an A10's room, for 64 users and one. Before, the route's plans counted no keys and no values, and one user.
    #[cfg(feature = "dev")]
    #[test]
    fn the_dynamic_route_plans_its_sequences() {
        let dir = std::env::temp_dir().join(format!("glyd-plan-seqs-{}", std::process::id()));
        for case in DYNAMIC_CASES {
            let _ = std::fs::remove_dir_all(&dir);
            crate::logits::random_model(case, &dir).unwrap();
            for machine in ["rtx4080s", "l4", "a6000"] {
                for users in [64usize, 1] {
                    let f = planned_fit(&dir, machine, &["--users", &users.to_string()]);
                    let lp = f.plan.chosen.as_ref().and_then(|c| c.load.clone()).unwrap_or_else(|| panic!("{case} {machine}: {:?}", f.refusal));
                    assert!(lp.kind == Kind::Dynamic && lp.kv > 0 && lp.users == users, "{case} {machine} {users}: {}", lp.text());
                    assert!(lp.gpu_total() <= lp.free, "{case} {machine} {users}: {}", lp.text());
                    let d = loader_desc(case, &dir, lp.context + lp.chunk + 16);
                    let seq = crate::dynamic::SeqRoom::of(&d, lp.context, lp.chunk);
                    match seq.hold {
                        crate::dynamic::Hold::Pooled => {
                            let budget = seq.pool_budget(lp.users, lp.kv, 0);
                            let most: Vec<usize> = seq.least.iter().map(|&b| b * lp.users).collect();
                            let keep: Vec<Option<usize>> = seq.least.iter().map(|_| None).collect();
                            let blocks = crate::dynamic::pool_blocks(budget, &seq.per, &seq.covers, &keep, &most, &seq.least);
                            assert!(blocks.iter().zip(&seq.least).all(|(b, l)| b >= l), "{case} {machine} {users}: {blocks:?} of {:?}", seq.least);
                        }
                        crate::dynamic::Hold::Own => assert_eq!(seq.own_seqs(lp.users, lp.kv), lp.users, "{case} {machine} {users}"),
                        crate::dynamic::Hold::One => assert_eq!(lp.users, 1, "{case} {machine} {users}"),
                    }
                }
                // (run's one sequence: its rings and states as the engine loads them)
                let f = planned_fit(&dir, machine, &[]);
                let lp = f.plan.chosen.as_ref().and_then(|c| c.load.clone()).unwrap();
                let d = loader_desc(case, &dir, lp.context + lp.chunk + 16);
                let seq = crate::dynamic::SeqRoom::of(&d, lp.context, lp.chunk);
                assert_eq!((lp.users, lp.kv), (1, seq.rings + seq.state), "{case} {machine}: run's one sequence");
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The 12-bit pages by kind on the dynamic route, at the releases' shapes (#332: Qwen3.8-27B's heads of 256, Kimi-K3's latents of 512 + 64, GLM-5.3-Flash's
    /// latents): every ring the pages take is fewer bytes a token in them than in bf16 (`SeqRoom::pages`; the latent's 576 values as 5 heads of 128, one stream),
    /// so 32 users at 8,192 tokens each reserve less in them than in bf16 (`SeqRoom::most_bytes`, the rule's figure), and one user at the context more (what the
    /// pages spend once: a raw slab for every page of the sequence and the rows a pass unpacks).
    #[test]
    fn the_releases_attention_kinds_reserve_less_in_twelve_bit_pages_for_many_users() {
        let q35 = json::parse(&fixture("qwen35_configs.json")).unwrap();
        let kimi: serde_json::Value = serde_json::from_str(&fixture("headers/Kimi-K3.json")).unwrap();
        let cases = [
            ("Qwen3.8-27B", json::to_compact(q35.get("Qwen/Qwen3.8-27B").unwrap())),
            ("Kimi-K3", kimi["config"].as_str().unwrap().to_string()),
            ("GLM-5.3-Flash", fixture("glm5_next_config.json")),
        ];
        let ctx = 8192;
        for (name, text) in cases {
            let c = crate::runtime::config::ModelConfig::parse(&text).unwrap();
            let low = crate::dynamic::split::lowered_of(&c, ctx + sizing::DYN_CHUNK + 16).unwrap_or_else(|e| panic!("{name}: {}", e.msg));
            let b = crate::dynamic::SeqRoom::of(&low, ctx, sizing::DYN_CHUNK);
            let t = crate::dynamic::SeqRoom::pages(&low, ctx, sizing::DYN_CHUNK).unwrap_or_else(|| panic!("{name}: no ring in 12-bit pages"));
            let taken = crate::dynamic::twelve_bit_rings(&low).into_iter().flatten().count();
            eprintln!(
                "{name}: {taken} rings in 12-bit pages, {:?} bytes a token against bf16's {:?}; once {} bytes; 32 users {} against {}, one {} against {}",
                t.per,
                b.per,
                t.once,
                t.most_bytes(32),
                b.most_bytes(32),
                t.most_bytes(1),
                b.most_bytes(1)
            );
            assert!(taken > 0 && t.once > 0 && t.per.iter().zip(&b.per).all(|(x, y)| x <= y) && t.per != b.per, "{name}");
            assert!(t.most_bytes(32) < b.most_bytes(32), "{name}: 32 users reserve less in 12-bit pages");
            assert!(t.most_bytes(1) > b.most_bytes(1), "{name}: one user reserves more in 12-bit pages");
        }
    }

    /// The 12-bit KV pages are planned by kind on every route of the executor (#332), never by family: a server's plan with GLYD_KV=lossless takes them where its
    /// description has rings in pools the pages take (`rows::twelve_bit_rings`: heads of 128, 256 or 512 channels, latents) and they reserve no more than bf16 for its
    /// users (`one_gpu_pages`' rule; a check's `twelve_always` takes them wherever they fit), and says so (`kv_format`). On the random cases of the dynamic route:
    /// 32 users on an L4 and an RTX 4080 SUPER are planned in them where a token is fewer bytes in them than in bf16 (the Qwen3.5 layout's heads of 256, GLM-5.3's
    /// latent of 128), in bf16 where it is more (Kimi-K3's random latent of 192 as 2 heads of 128 and GLM-5.3-Flash's of 64 as one: padded, more than bf16), and
    /// for a check in them, every plan's bytes holding one sequence's context in each group's pool by the executor's rule (`Dyn::page`: the plan's bytes less the
    /// states, block 0 and what the pages spend once, over each group's 12-bit bytes a token); one user at the context in bf16, a check's in 12-bit pages of more
    /// bytes; `run`'s one sequence and plans with nothing asked in bf16; heads of 64 (muse) and the compressed attention (deepseek-v4) never in them. The Qwen3.5
    /// layout's lossy route the same (kestrel). Before, the planner planned 12-bit pages for Gemma 4 and the plain dense kind alone.
    #[cfg(feature = "dev")]
    #[test]
    fn the_twelve_bit_pages_are_planned_by_kind_on_every_route() {
        let dir = std::env::temp_dir().join(format!("glyd-plan-kv12-kinds-{}", std::process::id()));
        let plan = |machine: &str, more: &[&str], lossless: bool, always: bool| -> LoadPlan {
            let f = sizing::always_twelve(always, || sizing::asking_lossless(lossless, || planned_fit(&dir, machine, more)));
            f.plan.chosen.as_ref().and_then(|c| c.load.clone()).unwrap_or_else(|| panic!("{machine} {more:?}: {:?}", f.refusal))
        };
        let mut twelve = Vec::new();
        for case in ["qwen35", "qwen35-wide", "kimi", "kimi-dsa", "glm", "glm-dsa", "muse", "deepseek-v4"] {
            let _ = std::fs::remove_dir_all(&dir);
            crate::logits::random_model(case, &dir).unwrap();
            for machine in ["l4", "rtx4080s"] {
                let users = ["--users", "32"];
                let (b, t, check) = (plan(machine, &users, false, false), plan(machine, &users, true, false), plan(machine, &users, true, true));
                assert!(!b.kv12 && b.kind == Kind::Dynamic, "{case} {machine}: nothing asked, bf16: {}", b.text());
                let d = loader_desc(case, &dir, check.context + check.chunk + 16);
                let Some(pages) = crate::dynamic::SeqRoom::pages(&d, check.context, check.chunk) else {
                    assert!(!t.kv12 && !check.kv12, "{case} {machine}: no ring the pages take, bf16: {}", check.text());
                    assert!(matches!(case, "muse" | "deepseek-v4"), "{case}: no ring in 12-bit pages");
                    continue;
                };
                let bf16 = crate::dynamic::SeqRoom::of(&d, check.context, check.chunk);
                let smaller = pages.per.iter().zip(&bf16.per).all(|(x, y)| x <= y);
                assert_eq!(t.kv12, smaller, "{case} {machine}: 12-bit pages where they reserve less ({:?} against {:?}): {}", pages.per, bf16.per, t.text());
                assert!(check.kv12 && check.kv_format() == "12-bit" && check.text().contains("keys and values in 12-bit pages"), "{case} {machine}");
                assert!(check.json("m").contains(r#""kv_format":"12-bit""#) && check.gpu_total() <= check.free, "{case} {machine}: {}", check.text());
                // (the executor's pools of the plan's bytes: its budget less the pages' once-only bytes, over each group's 12-bit bytes a token)
                let budget = bf16.pool_budget(check.users, check.kv, pages.once);
                let most: Vec<usize> = pages.least.iter().map(|&b| b * check.users).collect();
                let keep: Vec<Option<usize>> = pages.least.iter().map(|_| None).collect();
                let blocks = crate::dynamic::pool_blocks(budget, &pages.per, &pages.covers, &keep, &most, &pages.least);
                assert!(blocks.iter().zip(&pages.least).all(|(b, l)| b >= l), "{case} {machine}: {blocks:?} of {:?}", pages.least);
                // (one user at the context: bf16 for a server, the pages of more bytes for a check; run's one sequence bf16 whatever is asked)
                let (one, one_check) = (plan(machine, &["--users", "1"], true, false), plan(machine, &["--users", "1"], true, true));
                assert!(!one.kv12 && one_check.kv12 && one_check.kv > one.kv, "{case} {machine}: {} / {}", one.text(), one_check.text());
                assert!(!plan(machine, &[], true, true).kv12, "{case} {machine}: run's one sequence");
                if t.kv12 {
                    twelve.push(case);
                }
            }
            // (the Qwen3.5 layout's lossy route, by the same rule)
            if case == "qwen35" {
                let lossy = |lossless: bool| -> LoadPlan {
                    let f = sizing::asking_lossless(lossless, || {
                        planned(&args(&["fit", &format!("{}:kestrel", dir.display()), "--machine", "l4", "--users", "32"]))
                    });
                    f.unwrap().plan.chosen.and_then(|c| c.load).expect("a plan")
                };
                let (b, t) = (lossy(false), lossy(true));
                assert!(b.kind == Kind::Qwen35 && !b.kv12 && t.kv12 && t.level == "kestrel", "{} / {}", b.text(), t.text());
            }
        }
        twelve.dedup();
        assert_eq!(twelve, ["qwen35", "qwen35-wide", "glm-dsa"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The hummingbird level's bits a weight are the recipe's shares of the two row layouts' bits, not a number of one model's experts.
    #[test]
    fn hummingbirds_bits_are_the_recipes_shares() {
        use crate::weights::Layout;
        let three = crate::moe_hb::HB40_THREE_BIT_PERCENT as f64 / 100.0;
        let mean = three * sizing::bits(Layout::Cb3Row) + (1.0 - three) * sizing::bits(Layout::Cb2Row);
        assert!((sizing::bits(Layout::Hummingbird) - mean).abs() <= 0.005, "{mean}");
        assert_eq!(sizing::bits(Layout::Hummingbird), 2.99);
    }

    /// The Qwen3.5 layout's penguin is the dynamic route's (#331): Qwen3.8-27B, from its released tensors' headers, planned as `run` loads it: its layers cut
    /// over two A100 40 GB cards (and two RTX 3090s), whole on one A100 80 GB (an A100's in the 12-bit layout), refused on one 24 GB card; `--ctx` read as
    /// `--context`; and untagged, two A100
    /// 40 GB cards load it at penguin, one 24 GB card at a lossy level.
    #[test]
    fn the_qwen35_layouts_penguin_is_planned_on_the_dynamic_route() {
        let q35 = json::parse(&fixture("qwen35_configs.json")).unwrap();
        let dir = model_dir("q35-penguin", "Qwen3.8-27B", &json::to_compact(q35.get("Qwen/Qwen3.8-27B").unwrap()));
        header_only_checkpoint(&dir, &qwen35_tensors("Qwen3.8-27B").unwrap());
        let a100s = dir.parent().unwrap().join("2xa100-40g.json");
        let a100 = r#"{"name": "NVIDIA A100-SXM4-40GB", "cc": [8, 0], "total_gib": 40, "free_gib": 39.4}"#;
        std::fs::write(&a100s, format!(r#"{{"name": "2xa100-40g", "gpus": [{a100}, {a100}], "ram_free_gib": 200}}"#)).unwrap();
        let (model, a100s) = (dir.to_str().unwrap(), a100s.to_str().unwrap());
        let fit = |model: &str, machine: &str, more: &[&str]| {
            let mut w = vec!["fit", model, "--machine", machine];
            w.extend(more);
            planned(&args(&w)).unwrap()
        };
        let penguin = format!("{model}:penguin");
        // (an A100's bf16 matrices in penguin's 12-bit layout, which its one-token product reads faster, as an RTX A6000's (Ampere, not Ada: the one-GPU
        // paths' choice); two RTX 3090s' in the smallest, the 12-bit one not fitting)
        for (machine, stages, twelve) in [(a100s, 2, true), ("2x3090", 2, false), ("a100-80g", 1, true), ("a6000", 1, true)] {
            let f = fit(&penguin, machine, &[]);
            let p = f.plan.chosen.as_ref().and_then(|c| c.load.clone()).unwrap_or_else(|| panic!("{machine}: {:?}", f.refusal));
            let sp = p.spread.as_ref().unwrap();
            assert_eq!((p.family, p.level, sp.devices.len(), sp.cuts.len()), ("Qwen3.5", "penguin", stages, stages - 1), "{machine}");
            assert_eq!((sp.twelve, p.layout == crate::weights::Layout::Twelve), (twelve, twelve), "{machine}");
            assert!(p.gpu_total() <= p.free, "{machine}");
        }
        let r = fit(&penguin, "rtx4090", &[]).refusal.unwrap();
        assert!(r.what.contains("needs about") && r.what.contains("at penguin"), "{r:?}");
        let p = fit(&penguin, "a100-80g", &["--ctx", "4096"]).plan.chosen.and_then(|c| c.load).unwrap();
        assert_eq!(p.context, 4096, "--ctx is --context");
        assert_eq!(fit(model, a100s, &[]).plan.chosen.unwrap().level, "penguin", "lossless where it fits, over two GPUs");
        assert_ne!(fit(model, "rtx4090", &[]).plan.chosen.unwrap().level, "penguin");
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
    }

    /// The reply a model's first load times (`first_use`): a prompt of 512 tokens of ids past the special ones, 33 made with no end, and the times of its first
    /// token and of the rest; nothing on an engine whose context or vocabulary is too small.
    #[test]
    fn the_first_use_times_one_reply() {
        struct Fake(usize, Vec<usize>);
        impl crate::conv::Gen for Fake {
            fn capacity(&self) -> usize {
                self.0
            }
            fn reply(
                &mut self,
                prompt: &[u32],
                o: &crate::sched::Reply,
                each: &mut dyn FnMut(u32) -> bool,
            ) -> crate::ffi::Result<(Vec<u32>, crate::sched::Finish)> {
                self.1.push(prompt.len());
                assert!(o.ignore_eos && prompt.iter().all(|&t| (1000..32000).contains(&t)));
                std::thread::sleep(std::time::Duration::from_millis(20));
                for _ in 0..o.max_new {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                    each(7);
                }
                Ok((vec![7; o.max_new], crate::sched::Finish::Length))
            }
        }
        let mut e = Fake(32768, Vec::new());
        let (n, first_s, made, decode_s) = first_reply(&mut e, 32000).unwrap();
        assert_eq!((n, made, e.1.clone()), (512, 33, vec![512]));
        assert!(first_s >= 0.02 && decode_s >= 0.03 && decode_s < 1.0, "{first_s} {decode_s}");
        assert!(first_reply(&mut Fake(300, Vec::new()), 32000).is_none() && first_reply(&mut Fake(32768, Vec::new()), 1000).is_none());
    }

    /// P2c: a GPU's rates are measured on its first use and kept in the cache by its name and number; later uses read them and measure nothing; where they cannot be
    /// measured nothing is kept.
    #[test]
    fn the_rates_are_probed_once_and_kept() {
        let root = std::env::temp_dir().join(format!("glyd-plan-rates-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let calls = std::cell::Cell::new(0);
        let measure = || {
            calls.set(calls.get() + 1);
            Some(Probed { stream_gbs: 697.35, link_gbs: 26.9, ram_gbs: 73.9 })
        };
        let gpu = "NVIDIA GeForce RTX 4080 SUPER";
        assert_eq!(rates_in(&root, gpu, 0, &measure).map(|r| r.stream_gbs), Some(697.35));
        assert_eq!(rates_in(&root, gpu, 0, &measure).map(|r| r.ram_gbs), Some(73.9));
        assert_eq!(calls.get(), 1, "measured once");
        let kept: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(root.join("rates/NVIDIA_GeForce_RTX_4080_SUPER-0.json")).unwrap()).unwrap();
        assert_eq!((kept["schema"].as_str(), kept["link_gbs"].as_f64()), (Some(RATES_SCHEMA), Some(26.9)));
        // (another GPU of the computer is its own; a probe that fails keeps nothing)
        assert!(rates_in(&root, gpu, 1, &|| None).is_none() && !root.join("rates/NVIDIA_GeForce_RTX_4080_SUPER-1.json").exists());
        // (a GPU's measured rate is the one its predictions read, not the maker's figure)
        let mut d = Machine::preset("rtx4080s").unwrap().devices.remove(0);
        d.stream = Some(600.0);
        assert!((d.rate() - 600e9).abs() < 1.0);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// P2c: a run's replies are kept with its plan against the prediction at their positions; the runs on a GPU then refine every prediction there (the median of what
    /// they reached against theirs, and their p10-p90 as the band), a model's own runs its own, a run with a draft model only its own; the plan kept anew keeps them.
    #[test]
    fn runs_observed_here_refine_the_prediction() {
        let root = std::env::temp_dir().join(format!("glyd-plan-observed-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let dir = model_dir("observed", "gemma-4-12B", &fixture("gemma4_12b_config.json"));
        let model = format!("{}:kestrel", dir.to_str().unwrap());
        let f = planned(&args(&["fit", &model, "--machine", "rtx4080s"])).unwrap();
        // (run with no draft model: --no-draft)
        let lp = LoadPlan { draft: 0, ..f.plan.chosen.as_ref().unwrap().load.clone().unwrap() };
        let m = f.machine.clone();
        remember_in(&root, &model, &lp, lp.weights, 0);
        let w = watch_of(&root, &model, &lp, f.shape.clone(), m.clone()).unwrap();
        let predicted = speed_at(&f.shape, &m, 0, &lp, 1000.0 + 64.5);
        // (four replies of 1,000 prompt tokens and 129 made, at 0.9, 0.95, 1.0 and 1.05 of the prediction; one too short to say anything)
        for k in [0.9, 0.95, 1.0, 1.05] {
            keep(&w, 1000, 0.25, 129, 128.0 / (predicted * k));
        }
        keep(&w, 10, 0.01, 3, 0.02);
        let kept = recall_in(&root, &model, lp.level, &lp.gpu).unwrap();
        let seen = kept["observed"].as_array().unwrap();
        assert_eq!(seen.len(), 4, "the short reply says nothing");
        assert!((seen[2]["decode_ratio"].as_f64().unwrap() - 1.0).abs() < 1e-9 && seen[0]["prefill_tok_s"].as_f64() == Some(4000.0));
        // the GPU's band from them, and a prediction refined by it
        let (band, prefill) = observed_on(&root, &lp.gpu);
        let b = band.unwrap();
        assert!((b[1] - 0.975).abs() < 1e-9 && b[0] < b[1] && b[1] < b[2] && b[2] <= 1.05, "{b:?}");
        assert!(prefill.is_some());
        let mut d = m.devices[0].clone();
        d.observed = band;
        let c = Candidate::speeds(100.0, &d, None);
        assert!((c.tok_s - 97.5).abs() < 1e-9 && c.range == (100.0 * b[0], 100.0 * b[2]));
        // (no run observed: the prediction in its class's band)
        let b0 = m.devices[0].prior.band;
        assert_eq!(Candidate::speeds(100.0, &m.devices[0], None).range, (100.0 * b0[0], 100.0 * b0[2]));
        // the model's own runs, and the plan kept anew keeps them
        assert_eq!(observed_for(&root, &model, lp.level, &lp.gpu).map(|x| x.0), band);
        remember_in(&root, &model, &lp, lp.weights, 1 << 30);
        assert_eq!(observed_for(&root, &model, lp.level, &lp.gpu).map(|x| x.0), band);
        // a run with a draft model is its own model's, not the GPU's
        let drafted = LoadPlan { level: "wren", draft: 640 << 20, ..lp.clone() };
        remember_in(&root, &model, &drafted, lp.weights, 0);
        let at_wren = speed_at(&f.shape, &m, 0, &drafted, 1064.5);
        keep(&watch_of(&root, &model, &drafted, f.shape.clone(), m.clone()).unwrap(), 1000, 0.25, 129, 128.0 / (at_wren * 1.6));
        assert_eq!(observed_on(&root, &lp.gpu).0, band);
        assert!(observed_for(&root, &model, "wren", &lp.gpu).is_some_and(|b| b.0[1] > 1.5));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The plan that loaded is kept by model, level and GPU, and `fit` says what it took here before and whether the plan it makes now is the same.
    #[test]
    fn the_plan_that_loaded_is_remembered() {
        let dir = model_dir("memo", "gemma-4-12B", &fixture("gemma4_12b_config.json"));
        let model = format!("{}:kestrel", dir.to_str().unwrap());
        let f = planned(&args(&["fit", &model, "--machine", "rtx4080s"])).unwrap();
        let c = f.plan.chosen.unwrap();
        let lp = c.load.clone().unwrap();
        let root = std::env::temp_dir().join(format!("glyd-plan-memo-root-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        assert!(remembered(&root, &model, &c).is_none(), "nothing loaded yet");
        remember_in(&root, &model, &lp, lp.weights, lp.gpu_total() - lp.reserve - (200 << 20));
        let again = recall_in(&root, &model, lp.level, &lp.gpu).unwrap();
        assert_eq!((again["plan"]["context"].as_u64(), again["schema"].as_str()), (Some(lp.context as u64), Some(SCHEMA)));
        assert!(remembered(&root, &model, &c).unwrap().contains("the same plan"));
        // (another context asked for: a different plan)
        let f = planned(&args(&["fit", &model, "--machine", "rtx4080s", "--context", "8192"])).unwrap();
        assert!(remembered(&root, &model, f.plan.chosen.as_ref().unwrap()).unwrap().contains("a different plan"));
        // (another GPU, another level: not the same memory)
        assert!(recall_in(&root, &model, "swift", &lp.gpu).is_none() && recall_in(&root, &model, lp.level, "NVIDIA L4").is_none());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The GPU job's script (scripts/ci_plan_check.py) accepts the report of a load that took what the plan said, from the JSON `fit --json` prints, and fails one that took
    /// more of the GPU than planned or loaded another plan.
    #[test]
    fn the_gpu_check_reads_the_plan_and_the_report() {
        let dir = model_dir("check", "gemma-4-12B", &fixture("gemma4_12b_config.json"));
        let model = format!("{}:kestrel", dir.to_str().unwrap());
        let f = planned(&args(&["fit", &model, "--machine", "rtx4080s"])).unwrap();
        let c = f.plan.chosen.as_ref().unwrap();
        let lp = c.load.as_ref().unwrap();
        let out = std::env::temp_dir().join(format!("glyd-plan-check-{}", std::process::id()));
        std::fs::create_dir_all(&out).unwrap();
        let run = |report: String| {
            std::fs::write(out.join("fit.json"), c.json("m", &f.machine)).unwrap();
            std::fs::write(out.join("loaded.json"), report).unwrap();
            let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../scripts/ci_plan_check.py");
            std::process::Command::new("python3").arg(script).arg(out.join("fit.json")).arg(out.join("loaded.json")).output()
        };
        let Ok(ok) = run(lp.report("m", lp.weights_on_gpu() + 1000, lp.gpu_total() - lp.reserve - (300 << 20))) else { return }; // (no python3 here)
        assert!(ok.status.success(), "{}", String::from_utf8_lossy(&ok.stdout));
        let over = run(lp.report("m", lp.weights_on_gpu(), lp.gpu_total() - lp.reserve + (1 << 30))).unwrap();
        assert!(!over.status.success() && String::from_utf8_lossy(&over.stdout).contains("the load took"), "{}", String::from_utf8_lossy(&over.stdout));
        let mut other = lp.clone();
        other.context = 4096;
        let wrong = run(other.report("m", other.weights_on_gpu(), 0)).unwrap();
        assert!(!wrong.status.success() && String::from_utf8_lossy(&wrong.stdout).contains("context"));
        let _ = std::fs::remove_dir_all(&out);
    }

    /// What `run` decides for MODEL at `level` on `m`, in the order it decides it (src/cli.rs `tagged`, `gpu_pick`'s checks, then the loader's sizing): the refusal,
    /// or the plan it loads (None: a family sized by the planner's estimate alone).
    fn run_decides(model: &str, text: &str, level: &str, m: &Machine) -> Result<Option<LoadPlan>, crate::ui::Refusal> {
        let name = crate::cli::short_name(model);
        crate::support::check(text, model, &name, level)?;
        if let Some(r) = crate::support::machine(m) {
            return Err(r);
        }
        let main = (0..m.devices.len()).min_by_key(|&i| (std::cmp::Reverse(m.devices[i].free), i)).unwrap();
        let target = Target::of(&m.devices[main], main as i32, Some(m.ram_free));
        let devs: Vec<Target> = m.devices.iter().enumerate().filter(|(_, d)| d.cc.0 >= 8).map(|(i, d)| Target::of(d, i as i32, Some(m.ram_free))).collect();
        let linked = m.devices.iter().all(|d| d.cc.1 == 0 && (8..=10).contains(&d.cc.0));
        let a = args(&["run", model]);
        sizing::size(&a, model, &mc(text), crate::levels::LEVELS.iter().map(|l| l.name).find(|n| *n == level).unwrap(), &devs, &target, &|_| linked, 1)
    }

    /// P16: no preset by fixture model by level where `fit` and `run` disagree: for every one, `fit` refuses exactly where `run` does, in `run`'s words and with its
    /// status, and where `run` loads a plan `fit` prints that plan. The models are the families' fixtures, one of a dense architecture the engine does not run, and
    /// Qwen3.8-Flash-Next (Condor's); every machine preset, AMD, Apple, Arm and the multi-GPU ones among them; every level this build has, those a family does not run at too.
    #[test]
    fn fit_and_run_agree_on_every_preset_and_model() {
        let q35 = json::parse(&fixture("qwen35_configs.json")).unwrap();
        let unknown = r#"{"architectures": ["DenseForCausalLM"], "model_type": "dense", "hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8}"#;
        let glm = fixture("glm5_next_config.json");
        let configs = [
            ("dense-64l", with_rope(DENSE_64L)),
            ("Qwen3.5-9B", json::to_compact(&q35.as_object().unwrap()[0].1)),
            ("gemma-4-12B", fixture("gemma4_12b_config.json")),
            ("GLM-5.3-Flash", glm.clone()),
            ("Qwen3.8-Flash-Next", fixture("qwen4x_config.json")),
            ("dense-unknown", unknown.to_string()),
        ];
        let (mut pairs, mut loads, mut refusals) = (0, 0, 0);
        for (name, text) in &configs {
            let dir = model_dir("agree", name, text);
            // (the dynamic route sizes a model from its files' headers (src/sizes.rs, P2b): a directory with a config alone has none, so these get a header-only
            // checkpoint of their release's tensors)
            if let Some(t) = released_tensors(name) {
                header_only_checkpoint(&dir, &t);
            }
            let model = dir.to_str().unwrap();
            for preset in Machine::PRESETS {
                let m = Machine::preset(preset).unwrap();
                for level in crate::levels::names_now() {
                    let tag = format!("{model}:{level}");
                    let (fit, run) = (planned(&args(&["fit", &tag, "--machine", preset])), run_decides(model, text, level, &m));
                    pairs += 1;
                    let same = |a: &crate::ui::Refusal, b: &crate::ui::Refusal| {
                        assert_eq!((&a.what, &a.fix, a.code), (&b.what, &b.fix, b.code), "{name} {preset} {level}")
                    };
                    match (fit, run) {
                        (Err(f), Err(r)) => {
                            same(&f, &r);
                            refusals += 1;
                        }
                        (Ok(f), Err(r)) => {
                            same(f.refusal.as_ref().unwrap_or_else(|| panic!("{name} on {preset} at {level}: fit exits 0 where run refuses: {}", r.what)), &r);
                            refusals += 1;
                        }
                        (Ok(f), Ok(want)) => {
                            assert!(f.refusal.is_none(), "{name} on {preset} at {level}: fit refuses where run loads: {:?}", f.refusal);
                            let got = f.plan.chosen.as_ref().and_then(|c| c.load.clone());
                            assert_eq!(got, want, "{name} on {preset} at {level}");
                            loads += usize::from(got.is_some());
                        }
                        (Err(f), Ok(_)) => panic!("{name} on {preset} at {level}: fit refuses where run loads: {}", f.what),
                    }
                }
            }
        }
        assert!(
            pairs == configs.len() * Machine::PRESETS.len() * crate::levels::names_now().len() && loads > 40 && refusals > 200,
            "{pairs} {loads} {refusals}"
        );
    }

    /// The context limits come from the table: past the family's own limit, or the model's, `fit`, `run`'s sizing and the table give one refusal; the default
    /// context is the table's, whichever path sizes it.
    #[test]
    fn context_limits_and_defaults_are_the_tables() {
        let row = crate::support::row(crate::support::Family::Qwen3);
        let r = crate::support::context_check("dense-36l", row, 40960, Some(100_000)).unwrap_err();
        assert_eq!(
            (r.what.as_str(), r.fix.as_str()),
            ("dense-36l takes at most 40960 tokens of context here, and 100000 were asked for", "Ask for less: --context 40960")
        );
        assert!(
            crate::support::context_check("m", row, 262_144, Some(100_000)).unwrap_err().what.contains("at most 40960"),
            "the tables' reach, not the window"
        );
        assert!(crate::support::context_check("m", row, 8192, None).is_ok() && crate::support::context_check("m", row, 8192, Some(8192)).is_ok());
        assert_eq!(crate::support::default_context(row, 40960), sizing::CTX_WANT);
        assert_eq!(crate::support::default_context(crate::support::row(crate::support::Family::Gemma4), 262_144), LOSSY_CTX);
        assert_eq!(crate::support::default_context(crate::support::row(crate::support::Family::Gemma4), 8192), 8192, "never more than the model has");
        // (fit, the Qwen3 path and the lossy path say the same thing of an impossible context)
        let dir = model_dir("ctx", "dense-36l", DENSE_36L);
        let f = planned(&args(&["fit", dir.to_str().unwrap(), "--machine", "rtx4090", "--context", "100000"])).err().unwrap();
        let a = args(&["run", "dense-36l", "--context", "100000"]);
        let cfg = crate::weights::Config::parse(DENSE_36L, "config.json", None).unwrap();
        let m = Machine::preset("rtx4090").unwrap();
        let r = sizing::qwen3_plan(&a, &cfg, &crate::cli::short_name(dir.to_str().unwrap()), &gpu_of(&m.devices[0], 0), None, 1).unwrap_err();
        assert_eq!((f.what, f.fix, f.code), (r.what, r.fix, r.code));
    }

    /// What the last load took beyond its plan is the next plan's already-used memory: nothing before a load, nothing where it took less than planned or said nothing
    /// of the GPU, the difference where it took more, and at most 1 GiB.
    #[test]
    fn the_planner_is_calibrated_by_the_last_load() {
        let dir = model_dir("calib", "gemma-4-12B", &fixture("gemma4_12b_config.json"));
        let model = format!("{}:kestrel", dir.to_str().unwrap());
        let lp = planned(&args(&["fit", &model, "--machine", "rtx4080s"])).unwrap().plan.chosen.unwrap().load.unwrap();
        let root = std::env::temp_dir().join(format!("glyd-plan-calib-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let planned_used = lp.gpu_total() - lp.reserve;
        let calib = |used: u64| {
            remember_in(&root, &model, &lp, lp.weights, used);
            calibration_in(&root, &model, lp.level, &lp.gpu)
        };
        assert_eq!(calibration_in(&root, &model, lp.level, &lp.gpu), 0, "no load yet");
        assert_eq!(calib(planned_used - (200 << 20)), 0, "took less than planned");
        assert_eq!(calib(0), 0, "said nothing of the GPU");
        assert_eq!(calib(planned_used + (300 << 20)), 300 << 20, "took 300 MiB more");
        assert_eq!(calib(planned_used + (5 << 30)), 1 << 30, "at most a GiB: another program's");
        assert_eq!(calibration_in(&root, &model, "swift", &lp.gpu), 0, "another level");
        let _ = std::fs::remove_dir_all(&root);
        // (and it is the reserve's: the plan says the free memory as it was)
        let mut p = lp.clone();
        let before = (p.gpu_total(), p.free);
        sizing::calibrated(&mut p, lp.free, 300 << 20);
        assert_eq!((p.gpu_total(), p.free, p.reserve), (before.0 + (300 << 20), before.1, lp.reserve + (300 << 20)));
    }

    /// A table of the tensors a fixture lists (name -> [type, shape]): the headers a real checkpoint's files have.
    fn table_of(tensors: &Value) -> crate::sizes::Table {
        let rows = tensors
            .as_object()
            .unwrap()
            .iter()
            .map(|(name, t)| {
                let (dtype, shape) = (t.as_array().unwrap()[0].as_str().unwrap().to_string(), t.as_array().unwrap()[1].as_u64s().unwrap());
                let width = match dtype.as_str() {
                    "F32" => 4,
                    "F8_E4M3" | "I8" | "U8" => 1,
                    _ => 2,
                };
                crate::sizes::Row { name: name.clone(), dtype, bytes: shape.iter().product::<u64>() * width, shape }
            })
            .collect();
        crate::sizes::Table { rows }
    }

    /// P2b: a model's parameters counted from its tensors' headers are the checkpoint's (every weight of the parts a plan counts: the vision tower is left out, the
    /// multi-token-prediction head is counted only where the model drafts with it), where config.json's estimate is not: for the recent families whose headers the
    /// fixtures hold as tensor lists (GLM-5.3-Flash, Qwen3.5 at 0.8B and 9B) within 1%; the recorded headers of Kimi-K3, DeepSeek-V4-Flash and the others, whose
    /// packed experts count by their bytes, are src/sizes.rs's `recorded_headers_are_the_hubs_file_sizes`.
    #[test]
    fn a_shape_from_the_headers_is_the_checkpoints() {
        use crate::sizes::{classify, Part};
        let glm = json::parse(&fixture("glm5_next_tensors.json")).unwrap();
        let q35 = json::parse(&fixture("qwen35_tensors.json")).unwrap();
        let q35c = json::parse(&fixture("qwen35_configs.json")).unwrap();
        let mut cases = vec![("GLM-5.3-Flash", fixture("glm5_next_config.json"), table_of(glm.get("tensors").unwrap()))];
        for m in ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-9B"] {
            cases.push((m, json::to_compact(q35c.get(m).unwrap()), table_of(q35.get(m).unwrap())));
        }
        for (name, config, table) in &cases {
            let base = Shape::from_config(config).unwrap();
            let sh = base.clone().with_table(table);
            let counted: f64 = table
                .rows
                .iter()
                .filter(|r| match classify(&r.name, sh.layers) {
                    Part::Vision => false,
                    Part::Mtp => sh.mtp,
                    _ => true,
                })
                .map(|r| r.params().unwrap_or(0) as f64)
                .sum();
            assert!((sh.params() / counted - 1.0).abs() < 0.01, "{name}: {:.4e} parameters from the headers, {:.4e} in the checkpoint", sh.params(), counted);
            // (a vision tower in the checkpoint is not in the shape: the Qwen3.5 releases have one)
            if name.starts_with("Qwen/Qwen3.5") {
                let all: f64 = table.rows.iter().map(|r| r.params().unwrap_or(0) as f64).sum();
                assert!(table.rows.iter().any(|r| classify(&r.name, sh.layers) == Part::Vision) && sh.params() < all, "{name}");
            }
            // (an MoE keeps its experts apart: the routed share is the table's)
            assert_eq!(sh.is_moe(), base.is_moe(), "{name}");
            if sh.is_moe() {
                let routed: f64 =
                    table.rows.iter().filter(|r| classify(&r.name, sh.layers) == Part::RoutedExperts).map(|r| r.params().unwrap_or(0) as f64).sum();
                assert!(
                    (sh.routed - routed).abs() <= 1e-9 * routed.max(1.0),
                    "{name}: {:.4e} routed from the headers, {routed:.4e} in the checkpoint",
                    sh.routed
                );
            }
        }
        // (a table with none of the parts changes nothing)
        let base = Shape::from_config(&fixture("gemma4_12b_config.json")).unwrap();
        assert_eq!(base.clone().with_table(&crate::sizes::Table::default()).params(), base.params());
    }

    /// A server of a model whose experts are in the expert cache, on one GPU (#239, the planner's rule for every route): every user's keys, values and states at the
    /// context first, as the pools take them (`rows::SeqRoom`, each user's whole context), the engine's work kept apart, the cache's places what is left; `--users`
    /// held (eight here, where the plan held one and its pools then had no block), or refused in words with the numbers where the GPU leaves no place for a token's
    /// experts beside them. GLM-5.3 as released (its headers), on a GPU of the room asked for.
    #[test]
    fn a_cached_models_users_come_before_its_places() {
        let (config, table) = (fixture("glm_moe_dsa_config.json"), glm53_table());
        let nl = crate::dynamic::split::config_num(&config, "num_hidden_layers").unwrap();
        let src = sizing::release_experts(&crate::dynamic::tables::off_gpu(&table, &mc(&config)).0, &mc(&config), nl, false).unwrap();
        let (layers, ctx) = (src.moe.iter().filter(|&&m| m).count() as u64, 4096usize);
        // (eight users' whole contexts as the pools hold them: the executor's rule)
        let c = crate::runtime::config::ModelConfig::parse(&config).unwrap();
        let low = crate::dynamic::split::lowered_of(&c, ctx + sizing::DYN_CHUNK + 16).unwrap();
        let seq = crate::dynamic::SeqRoom::of(&low, ctx, sizing::DYN_CHUNK);
        let eight = seq.plan(8, u64::MAX).unwrap();
        assert!(eight.0 == 8 && eight.1 > seq.least_bytes(1), "{eight:?}");
        let weights = src.dense.iter().sum::<u64>() + src.base;
        // (the least free memory whose room holds `want`)
        let gpu_for = |want: u64| {
            let (_, free) = (0..48).fold((0u64, 1u64 << 40), |(lo, hi), _| {
                let mid = lo + (hi - lo) / 2;
                if crate::plan::dynamic_room_of(mid) >= want {
                    (lo, mid)
                } else {
                    (mid + 1, hi)
                }
            });
            Target { ordinal: 0, name: "NVIDIA A10".into(), cc: (8, 6), total: free, free, host: None, cublas: Some(false) }
        };
        let host = Some(src.all + HOST_MARGIN + (1 << 30));
        let at = ctx.to_string();
        let plan_on = |gpu: &Target, w: &[&str]| {
            sizing::size_dynamic(&args(w), "GLM-5.3", &table, None, &mc(&config), "GLM-5.3", std::slice::from_ref(gpu), host, &|_| false)
        };
        // a GPU whose room holds the weights besides the experts, the eight users and a third of the experts
        let gpu = gpu_for(weights + eight.1 + src.all / 3);
        let room = crate::plan::dynamic_room_of(gpu.free);
        let served = plan_on(&gpu, &["serve", "GLM-5.3", "--context", &at, "--users", "8"]).unwrap();
        // the eight users held, their bytes the pools' for each one's whole context
        assert_eq!((served.users, served.kv, served.context), (8, eight.1, ctx), "{}", served.text());
        // the places what is left of the room after the weights besides the experts and the users' bytes, every layer's alike on one GPU
        let places = ((room - weights - eight.1) / (layers * src.place)) as usize;
        assert!(places > src.top_k && places < src.experts, "{places} of {}", src.experts);
        let sp = served.spread.as_ref().unwrap();
        assert_eq!((sp.cuts.len(), sp.places.clone()), (0, vec![places]), "{}", served.text());
        assert!(served.weights_on_gpu() + served.kv <= room, "{}", served.text());
        // one sequence (a run) takes the places that are left beside it; `fit --users 8` plans what `serve` loads
        let run = plan_on(&gpu, &["run", "GLM-5.3", "--context", &at]).unwrap();
        assert!(run.users == 1 && run.kv < served.kv && run.spread.as_ref().unwrap().places[0] >= places, "{}", run.text());
        let fit = plan_on(&gpu, &["fit", "GLM-5.3", "--context", &at, "--users", "8"]).unwrap();
        assert_eq!(fit, served);
        // a GPU whose room holds the eight users and a token's experts a layer: eight served with the least places, sixty-four refused with the numbers
        let small = gpu_for(weights + eight.1 + src.token_bytes());
        let p = plan_on(&small, &["serve", "GLM-5.3", "--context", &at, "--users", "8"]).unwrap();
        assert_eq!((p.users, p.spread.as_ref().unwrap().places.clone()), (8, vec![src.top_k]), "{}", p.text());
        let e = plan_on(&small, &["serve", "GLM-5.3", "--context", &at, "--users", "64"]).unwrap_err();
        assert!(e.what.contains(&format!("for 64 users at {ctx} tokens of context")) && e.what.contains("keys, values and states for the users"), "{}", e.what);
        assert!(e.fix.contains("Ask for fewer users (--users 8)"), "{}", e.fix);
    }

    /// P2b: a model of the dynamic route is planned before its files are here, from its shards' headers by ranges, and the plan is the one its files give once they
    /// are: GLM-5.3-Flash's tensors (names, types and shapes of the real header, every large dimension cut by 64 so that the shards are megabytes) on a Hub in
    /// miniature, against the same shards in a directory, on each machine that holds them.
    #[test]
    fn a_model_is_planned_from_the_hubs_headers_as_from_its_files() {
        let tensors = json::parse(&fixture("glm5_next_tensors.json")).unwrap();
        let rows: Vec<(String, String, Vec<u64>)> = tensors
            .get("tensors")
            .and_then(Value::as_object)
            .unwrap()
            .iter()
            .map(|(n, t)| {
                let (d, sh) = (t.as_array().unwrap()[0].as_str().unwrap().to_string(), t.as_array().unwrap()[1].as_u64s().unwrap());
                (n.clone(), d, sh.into_iter().map(|x| if x >= 256 { x / 64 } else { x }).collect())
            })
            .collect();
        let refs: Vec<(&str, &str, &[u64])> = rows.iter().map(|(n, d, s)| (n.as_str(), d.as_str(), s.as_slice())).collect();
        let file = crate::sizes::tests::safetensors(&refs);
        let config = fixture("glm5_next_config.json");
        let (_m, hub, hub_dir) =
            crate::hub::tests::mock_with(vec![("config.json".into(), config.clone().into_bytes()), ("model.safetensors".into(), file.clone())]);
        let remote = hub.list("org/model", "main").unwrap();
        let root = std::env::temp_dir().join(format!("glyd-plan-hub-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let remote_table = crate::sizes::Table::from_hub_in(&root, &hub, &remote).unwrap();
        let dir = model_dir("hubplan", "GLM-5.3-Flash", &config);
        std::fs::write(dir.join("model.safetensors"), &file).unwrap();
        let local_table = crate::sizes::Table::from_dir(&dir).unwrap();
        assert_eq!(remote_table, local_table, "the headers by ranges are the files' headers");
        let mut planned_on = 0;
        for preset in ["rtx4080s", "2x3090", "8xh100"] {
            let m = Machine::preset(preset).unwrap();
            let devs: Vec<Target> = m.devices.iter().enumerate().map(|(i, d)| Target::of(d, i as i32, Some(m.ram_free))).collect();
            let plan = |t: &crate::sizes::Table| {
                sizing::size_dynamic(
                    &args(&["run", "GLM-5.3-Flash"]),
                    "GLM-5.3-Flash",
                    t,
                    None,
                    &mc(&config),
                    "GLM-5.3-Flash",
                    &devs,
                    Some(m.ram_free),
                    &|_| false,
                )
            };
            match (plan(&remote_table), plan(&local_table)) {
                (Ok(a), Ok(b)) => {
                    assert_eq!(a, b, "{preset}");
                    planned_on += 1;
                }
                (Err(a), Err(b)) => assert_eq!((a.what, a.fix), (b.what, b.fix), "{preset}"),
                _ => panic!("{preset}: a plan from the headers and one from the files disagree"),
            }
        }
        assert!(planned_on >= 2, "{planned_on}");
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&hub_dir);
    }

    /// The ladder, on a loader's answers made up: nothing to say where the model loads as asked; a shorter context is a step that costs nothing; experts in the computer's
    /// memory a step with its cost, asked about over 30%; a lower level always asked about; and the first refusal where no level loads.
    #[test]
    fn the_ladder_takes_its_steps_in_order_and_prices_them() {
        let dir = model_dir("ladder", "Nemotron-H", &fixture("nemotron_h_config.json"));
        let model = format!("{}:swift", dir.to_str().unwrap());
        let f = planned(&args(&["fit", &model, "--machine", "a100-80g"])).unwrap();
        let (sh, m, fam) = (&f.shape, &f.machine, f.family);
        // (swift's 78 GiB with its experts in the computer's memory on an 80 GB card: made whole here, the loader's answers are made up below)
        let base = LoadPlan { experts: None, ..f.plan.chosen.as_ref().and_then(|c| c.load.clone()).expect("swift loads on 80 GB") };
        let name = dir.to_str().unwrap();
        let go = |asked: &'static str, want: usize, answers: Vec<(&'static str, Result<Option<LoadPlan>, String>)>| {
            let size = |l: &'static str, _ctx: usize| match answers.iter().find(|a| a.0 == l) {
                Some((_, Ok(p))) => Ok(p.clone()),
                Some((_, Err(w))) => Err(crate::ui::Refusal::unsupported(w.clone(), "")),
                None => Err(crate::ui::Refusal::unsupported(format!("no answer for {l}"), "")),
            };
            ladder(sh, m, fam, name, asked, want, &size)
        };
        // as asked: no step, nothing said
        let l = go("swift", base.context, vec![("swift", Ok(Some(base.clone())))]);
        assert!(l.is_as_asked() && l.steps.is_empty() && !l.needs_confirmation() && l.report().is_empty());
        // a shorter context: the loader's, said, no cost, not asked about
        let short = LoadPlan { context: 4096, ..base.clone() };
        let l = go("swift", 32768, vec![("swift", Ok(Some(short.clone())))]);
        assert_eq!(l.steps.iter().map(|s| (s.kind, s.cost_pct, s.confirm)).collect::<Vec<_>>(), vec![(StepKind::ShorterContext, 0.0, false)]);
        assert!(l.text().contains("1. shorter context: 4096 tokens where 32768 were wanted (no change in speed)"), "{}", l.text());
        // experts in the computer's memory: a cache of a few places is far slower than the GPU alone, a cache of most of them is not
        let cache =
            |slots: usize| crate::offload::CachePlan { layers: 40, experts: 512, slots, expert_bytes: 3 << 20, in_place: false, ram: 40 * 512 * (3 << 20) };
        let tight = LoadPlan { experts: Some(cache(26)), ..base.clone() };
        let roomy = LoadPlan { experts: Some(cache(500)), ..base.clone() };
        let l = go("swift", base.context, vec![("swift", Ok(Some(tight)))]);
        assert_eq!(l.steps.len(), 1);
        assert!(
            l.steps[0].kind == StepKind::ExpertsInRam && l.steps[0].cost_pct > CONFIRM_COST_PCT && l.steps[0].confirm && l.needs_confirmation(),
            "{:?}",
            l.steps
        );
        let tight_cost = l.steps[0].cost_pct;
        let l = go("swift", base.context, vec![("swift", Ok(Some(roomy)))]);
        // (a cache of nearly every expert is the cheaper step, though the host and the GPU meeting in every layer still costs: the planner's calibrated sync)
        assert!(l.steps[0].cost_pct < tight_cost && l.steps[0].cost_pct > 0.0, "{:?} against {tight_cost}", l.steps);
        // a lower level where the one asked is refused: asked about, whatever it costs, after the context and the experts at that level in the order
        let lower = LoadPlan { level: "wren", context: 8192, experts: Some(cache(500)), ..base.clone() };
        let l = go("swift", 32768, vec![("swift", Err("it needs 90 GiB".into())), ("wren", Ok(Some(lower)))]);
        assert_eq!(l.steps.iter().map(|s| s.kind).collect::<Vec<_>>(), vec![StepKind::ShorterContext, StepKind::ExpertsInRam, StepKind::LowerLevel]);
        assert!(l.steps[2].confirm && l.needs_confirmation() && l.plan.as_ref().unwrap().level == "wren");
        assert!(l.report().contains("it loads at the wren level") && l.report().contains("not available: the KV cache's type"), "{}", l.report());
        // a shorter context before a lower level: a loader that refuses the context wanted and takes a shorter one (Gemma 4's, which does not shorten it itself) is taken at
        // the longest that fits, and below the floor the lower level is
        let at = |most: usize| {
            let b = base.clone();
            move |l: &'static str, ctx: usize| match l {
                "swift" if ctx <= most => Ok(Some(LoadPlan { context: ctx, ..b.clone() })),
                "wren" => Ok(Some(LoadPlan { level: "wren", ..b.clone() })),
                _ => Err(crate::ui::Refusal::unsupported(format!("{l} at {ctx} tokens"), "")),
            }
        };
        let l = ladder(sh, m, fam, name, "swift", 262144, &at(65536));
        assert_eq!(l.plan.as_ref().map(|p| (p.level, p.context)), Some(("swift", 65536)));
        assert_eq!(l.steps.iter().map(|s| s.kind).collect::<Vec<_>>(), vec![StepKind::ShorterContext]);
        let l = ladder(sh, m, fam, name, "swift", 262144, &at(4096));
        assert_eq!(l.plan.as_ref().map(|p| p.level), Some("wren"), "4096 is below the floor of {PREFERRED_CTX}");
        // dense layers in the computer's memory (the dynamic route's plan, src/dense_ram.rs): a step after the experts, priced by itself (their bytes over the link, every token)
        let spread = |dense: usize, dense_bytes: u64| Spread {
            cuts: vec![],
            devices: vec![0],
            across: false,
            coded: false,
            twelve: false,
            places: vec![26],
            experts_per_layer: 512,
            host_bytes: 40 * 512 * (3 << 20),
            host_per_stage: vec![40 * 512 * (3 << 20)],
            gpu_bytes: base.weights,
            dense,
            dense_bytes,
            tables: 0,
        };
        let without = go("swift", base.context, vec![("swift", Ok(Some(LoadPlan { spread: Some(spread(0, 0)), ..base.clone() })))]);
        assert_eq!(without.steps.iter().map(|s| s.kind).collect::<Vec<_>>(), vec![StepKind::ExpertsInRam]);
        let with = go("swift", base.context, vec![("swift", Ok(Some(LoadPlan { spread: Some(spread(2, 2 << 30)), ..base.clone() })))]);
        assert_eq!(with.steps.iter().map(|s| s.kind).collect::<Vec<_>>(), vec![StepKind::ExpertsInRam, StepKind::DenseInRam]);
        assert_eq!(with.steps[0].cost_pct, without.steps[0].cost_pct, "the experts' step costs what it does without the dense one");
        // (2 GiB over a 25 GB/s link is 86 ms a token more: over 30% of any speed this model has on the card)
        assert!(with.steps[1].cost_pct > CONFIRM_COST_PCT && with.steps[1].confirm, "{:?}", with.steps);
        assert!(with.text().contains("dense layers in the computer's memory: the last 2 layers' matrices (2.0 GiB) there"), "{}", with.text());
        assert_eq!(with.json()["steps"][1]["kind"], "dense_in_ram");
        // nothing loads: the first refusal, no plan
        let l = go("swift", 32768, vec![("swift", Err("first".into())), ("wren", Err("second".into())), ("hummingbird", Err("third".into()))]);
        assert!(l.plan.is_none() && l.refusal.as_ref().is_some_and(|r| r.what == "first") && l.report().contains("no step loads it (first)"));
        // a path not sized by its loader: the ladder says nothing
        let l = go("swift", 32768, vec![("swift", Ok(None))]);
        assert!(l.plan.is_none() && l.steps.is_empty() && l.refusal.is_none());
    }

    /// GLM-5.3's release as its headers list it (tests/fixtures/glm_moe_dsa_tensors.json: its tensors, and its experts' as runs of the same names).
    fn glm53_table() -> crate::sizes::Table {
        let t = json::parse(&fixture("glm_moe_dsa_tensors.json")).unwrap();
        let row = |name: String, v: &Value| {
            let (dtype, shape) = (v.as_array().unwrap()[0].as_str().unwrap().to_string(), v.as_array().unwrap()[1].as_u64s().unwrap());
            let width = match dtype.as_str() {
                "F8_E4M3" => 1,
                "BF16" => 2,
                _ => 4,
            };
            crate::sizes::Row { bytes: shape.iter().product::<u64>() * width, name, dtype, shape }
        };
        let mut rows: Vec<crate::sizes::Row> = t.get("tensors").and_then(Value::as_object).unwrap().iter().map(|(n, v)| row(n.clone(), v)).collect();
        for run in t.get("runs").and_then(Value::as_array).unwrap() {
            let p = run.get("prefix").and_then(Value::as_str).unwrap();
            for e in 0..run.get("count").and_then(Value::as_u64).unwrap() {
                rows.extend(run.get("tensors").and_then(Value::as_object).unwrap().iter().map(|(n, v)| row(format!("{p}{e}.{n}"), v)));
            }
        }
        crate::sizes::Table { rows }
    }

    /// GLM-5.3's release (tests/fixtures/glm_moe_dsa_tensors.json, its headers): each layer's bytes count its latent attention's kv_b_proj and its indexer's
    /// products, released in FP8, at the bf16 the loader holds them as (two bytes a value: 23.9 MB more on a layer with its own indexer, 14.7 MB on one
    /// lent its picks); a stage begins only at a layer whose indexer is its own (layers 0, 1, 2, then every fourth: `stage_starts`), and the cuts over
    /// eight GPUs of 136 GB are at such layers.
    #[test]
    fn glm53_stages_begin_where_an_indexer_picks() {
        let table = glm53_table();
        let (per, _) = crate::dynamic::split::layer_bytes_table(&table, 78, 100, 100, &|_| true);
        let raw = |l: usize| table.rows.iter().filter(|r| crate::sizes::layer_of(&r.name) == Some(l)).map(|r| r.bytes).sum::<u64>();
        assert_eq!((per[2] - raw(2), per[3] - raw(3)), (28672 * 512 + 4096 * 2048 + 128 * 6144, 28672 * 512));
        let text = fixture("glm_moe_dsa_config.json");
        let starts = sizing::stage_starts(&mc(&text), 78);
        let firsts: Vec<usize> = (0..78).filter(|&l| starts[l]).collect();
        assert_eq!(firsts, [0, 1, 2].into_iter().chain((6..78).step_by(4)).collect::<Vec<usize>>());
        let room: Vec<(i32, u64)> = (0..8).map(|g| (g, 136_000_000_000)).collect();
        let (_, cut, _, _) = crate::dynamic::split::fit_table_with(&table, 78, &room, true, false, &[], &starts);
        let (cuts, devices) = cut.expect("eight GPUs of 136 GB hold it");
        assert!(devices.len() > 1 && cuts.iter().all(|&c| starts[c]), "{cuts:?}");
        // (R12, #232: where its GPUs do not hold it, eight of 80 GB (its 704 GiB with its draft layer, 684 of them its experts), its experts in the expert cache as released (its
        // row's `ExpertCache::Release`): every expert in the computer's memory, places for at least a token's on each GPU, each stage beginning at a layer
        // whose indexer is its own (a stage of few layers of experts may hold all of its own); refused with the numbers where the computer's memory does
        // not hold the experts)
        let m = Machine::preset("8xh100").unwrap();
        let devs: Vec<Target> = m.devices.iter().enumerate().map(|(i, d)| Target::of(d, i as i32, Some(m.ram_free))).collect();
        let plan = |host: u64| sizing::size_dynamic(&args(&["run", "GLM-5.3"]), "GLM-5.3", &table, None, &mc(&text), "GLM-5.3", &devs, Some(host), &|_| false);
        let p = plan(m.ram_free).unwrap();
        let (sp, c) = (p.spread.as_ref().unwrap(), p.experts.expect("its experts in the computer's memory"));
        assert!(!sp.cuts.is_empty() && sp.cuts.iter().all(|&l| starts[l]), "{:?}", sp.cuts);
        let some = sp.places.iter().all(|&x| (8..=256).contains(&x)) && sp.places.iter().any(|&x| x < 256);
        assert!((c.experts, c.layers) == (256, 75) && some && sp.host_bytes > 600 << 30, "{c:?} {sp:?}");
        let r = plan(256 << 30).unwrap_err();
        // (the model's 75 layers of experts, 675 GiB: its draft layer's are not the model's)
        assert_eq!(r.what, "GLM-5.3 keeps the experts its GPUs do not hold in the computer's memory: they take 675.2 GiB, and 256.0 GiB is free");
        // (a split asked for at a layer that takes another stage's picks (7: the layer before it, 6, has its indexer's own) is refused by the plan in the loader's words, and one at a
        // layer with its own is not refused for that)
        let cut = |l: usize| {
            let a = args(&["run", "GLM-5.3", "--split", &l.to_string(), "--devices", "0,1"]);
            sizing::size_dynamic(&a, "GLM-5.3", &table, None, &mc(&text), "GLM-5.3", &devs, Some(m.ram_free), &|_| false).err().map(|r| r.what)
        };
        assert_eq!(cut(7).as_deref(), Some("a latent attention model: layer 7 takes the picks of a layer of another stage: a split must keep them together"));
        assert!(!cut(6).is_some_and(|w| w.contains("takes the picks")), "{:?}", cut(6));
    }

    /// DeepSeek-V4.1's release (tests/fixtures/headers/DeepSeek-V4.1-Flash-1017.json, its files' headers) on its reference machine, eight H100s: its Engram tables
    /// (two of 384M rows of 256 in FP8 with their scales, 189 GiB) read by rows on the host, none of the GPUs' bytes; its 40 layers cut over the GPUs only where
    /// no layer after a cut reads a compressed part of a layer before it (its kv sources 2, 8, 14, 20 and the candidate source 20: stages begin at 0, 1, 2, 8,
    /// 14 or 20), and its experts, which the GPUs do not hold so cut (layers 20-39 are one stage), in the expert cache as released, every one in the computer's
    /// memory. A computer of too little memory for its experts refuses it with the numbers.
    #[test]
    fn deepseek_v41_flash_plans_on_eight_h100s() {
        let h: Value = json::parse(&fixture("headers/DeepSeek-V4.1-Flash-1017.json")).unwrap();
        let text = h.get("config").and_then(Value::as_str).unwrap().to_string();
        let groups: Vec<crate::sizes::Group> = h
            .get("groups")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .map(|g| crate::sizes::Group {
                mask: g.get("mask").and_then(Value::as_str).unwrap().to_string(),
                dtype: g.get("dtype").and_then(Value::as_str).unwrap().to_string(),
                shape: g.get("shape").unwrap().as_u64s().unwrap(),
                bytes: g.get("bytes").and_then(Value::as_u64).unwrap(),
                layers: g.get("layers").unwrap().as_u64s().unwrap().into_iter().map(|l| l as usize).collect(),
                experts: g.get("experts").and_then(Value::as_u64).unwrap() as usize,
            })
            .collect();
        let table = crate::sizes::Table::expand(&groups);
        let starts = sizing::stage_starts(&mc(&text), 40);
        let firsts: Vec<usize> = (0..40).filter(|&l| starts[l]).collect();
        assert_eq!(firsts, [0, 1, 2, 8, 14, 20]);
        // (the tables on the host: both layers' rows and their scales, as stored)
        let host: u64 = table.rows.iter().filter(|r| r.name.contains(".engram.embed.")).map(|r| r.bytes).sum();
        assert_eq!(host, 98_305_579_008 + 3_072_049_344 + 98_308_270_592 + 3_072_133_456);
        assert_eq!(crate::dynamic::tables::off_gpu(&table, &mc(&text)).1, host);
        let m = Machine::preset("8xh100").unwrap();
        let devs: Vec<Target> = m.devices.iter().enumerate().map(|(i, d)| Target::of(d, i as i32, Some(m.ram_free))).collect();
        let name = "DeepSeek-V4.1-Flash-1017";
        let plan = |host: u64| sizing::size_dynamic(&args(&["run", name]), name, &table, None, &mc(&text), "DeepSeek-V4.1", &devs, Some(host), &|_| false);
        let p = plan(m.ram_free).unwrap_or_else(|r| panic!("{r:?}"));
        let (sp, c) = (p.spread.as_ref().unwrap(), p.experts.expect("its experts in the computer's memory"));
        assert!(!sp.cuts.is_empty() && sp.cuts.iter().all(|&l| starts[l]), "{:?}", sp.cuts);
        assert!(sp.tables == host && sp.devices.len() == sp.cuts.len() + 1 && sp.places.iter().all(|&x| (6..=384).contains(&x)), "{sp:?}");
        assert!((c.experts, c.layers) == (384, 40) && sp.host_bytes > 260 << 30, "{c:?}");
        assert!(p.gpu_total() <= p.free, "{p:?}");
        // (its experts, 40 layers of 384, 268.9 GiB: the computer's memory must hold them)
        let r = plan(200 << 30).unwrap_err();
        assert!(r.what.starts_with("DeepSeek-V4.1-Flash-1017 keeps the experts its GPUs do not hold in the computer's memory: they take "), "{r:?}");
        // (a split asked for inside the layers that read layer 20's parts is refused in the loader's words; one at 20 is not refused for that)
        let cut = |l: usize| {
            let a = args(&["run", name, "--split", &l.to_string(), "--devices", "0,1"]);
            sizing::size_dynamic(&a, name, &table, None, &mc(&text), "DeepSeek-V4.1", &devs, Some(m.ram_free), &|_| false).err().map(|r| r.what)
        };
        assert_eq!(cut(25), Some(crate::dynamic::lower_comp::split_apart_parts(25)));
        assert!(!cut(20).is_some_and(|w| w.contains("compressed part")), "{:?}", cut(20));
    }

    /// The plan refuses a split by the rule the loader keeps: a stage begins at a layer whose indexer is its own. On GLM-5.3's random case (a layer of its own and one more, then
    /// two that take its picks) `stage_starts` says what the loader says of the stage beginning at each layer, and the words are the loader's.
    #[test]
    fn the_plans_stage_starts_are_the_loaders() {
        use crate::dynamic::{
            lower::{latent_moe, split_apart_picks},
            Stage,
        };
        let text = fixture("glm_moe_dsa_toy_config.json");
        let kinds = crate::runtime::describe::describe(&serde_json::from_str(&text).unwrap()).unwrap();
        let starts = sizing::stage_starts(&mc(&text), 4);
        assert_eq!(starts, [true, true, false, false]);
        for c in 1..4 {
            let loaded = latent_moe(&kinds, vec![1], 4096, &Stage { layers: (c, 4), first: false, last: true });
            assert_eq!(loaded.is_err(), !starts[c], "a stage beginning at layer {c}");
            if let Err(e) = loaded {
                assert_eq!(e.msg, split_apart_picks(c));
            }
        }
    }

    /// A stage of the expert cache's plan begins only where one may (`places_plan_at`: the layers from each such layer to the next on one GPU), and where
    /// every layer may, the plan is `places_plan`'s.
    #[test]
    fn the_caches_stages_begin_where_one_may() {
        let (dense, moe) = (vec![10u64; 8], (0..8).map(|l| l > 0).collect::<Vec<_>>());
        let room = [(0, 100), (1, 100)];
        let free = sizing::places_plan(&dense, 4, &moe, &room, 2, 64, 4);
        assert_eq!(free, Some((vec![4], vec![0, 1], vec![9, 7])));
        assert_eq!(sizing::places_plan_at(&dense, 4, &moe, &room, 2, 64, 4, &[true; 8]), free);
        let may: Vec<bool> = (0..8).map(|l| l != 4).collect();
        let (cuts, _, places) = sizing::places_plan_at(&dense, 4, &moe, &room, 2, 64, 4, &may).expect("two GPUs hold it");
        assert!(cuts.len() == 1 && cuts[0] != 4 && places.iter().all(|&p| p >= 4), "{cuts:?} {places:?}");
    }

    /// The Qwen3.5 layout with experts at its largest release (Qwen3.8-2.4T-A95B, 92 layers of 512 experts in bf16): no machine of the presets holds it at
    /// any level, and each says so with the numbers: at penguin its experts in the computer's memory take more than any has, at the lossy levels the one
    /// GPU's memory it takes. The tensors are its description's, at their shapes (no index of the release's is in the fixtures).
    #[test]
    fn the_largest_qwen35_layout_with_experts_is_refused_with_its_numbers() {
        let q35 = json::parse(&fixture("qwen35_configs.json")).unwrap();
        let text = json::to_compact(q35.get("Qwen/Qwen3.8-2.4T-A95B").unwrap());
        let d = crate::runtime::describe::describe(&serde_json::from_str(&text).unwrap()).unwrap();
        let rows: Vec<crate::sizes::Row> = d
            .names()
            .into_iter()
            .map(|n| {
                let shape: Vec<u64> = d.shapes[&n].iter().map(|&x| x as u64).collect();
                crate::sizes::Row { bytes: shape.iter().product::<u64>() * 2, name: n, dtype: "BF16".into(), shape }
            })
            .collect();
        let table = crate::sizes::Table { rows };
        let all = table.rows.iter().map(|r| r.bytes).sum::<u64>();
        assert!(all > 4_000 << 30, "{} GiB", all >> 30);
        let dir = model_dir("q35-2t", "Qwen3.8-2.4T-A95B", &text);
        for preset in ["a100-80g", "8xh100", "dgx-spark"] {
            let m = Machine::preset(preset).unwrap();
            let devs: Vec<Target> = m.devices.iter().enumerate().map(|(i, d)| Target::of(d, i as i32, Some(m.ram_free))).collect();
            let r = sizing::size_dynamic(&args(&["run", "q"]), "Qwen3.8-2.4T-A95B", &table, None, &mc(&text), "Qwen3.5", &devs, Some(m.ram_free), &|_| false)
                .unwrap_err();
            assert!(r.what.contains("keeps the experts its GPUs do not hold in the computer's memory: they take"), "{preset}: {}", r.what);
            for level in ["kestrel", "swift", "wren"] {
                let layout = crate::levels::layout_of(level).unwrap();
                let model = dir.to_str().unwrap();
                let r = sizing::lossy_plan_with(&args(&["run", model]), model, layout, &mc(&text), None, devs[0].free, &devs[0].name, None, devs[0].host)
                    .unwrap_err();
                assert!(r.what.contains(&format!("at the {level} level takes about")) && r.what.contains("GiB of GPU memory"), "{preset} {level}: {}", r.what);
            }
        }
    }

    /// A snapshot of the planner's own plans (tests/fixtures/plans_p2a.json): each family's fixture, on every preset, at every level this build has, alone and
    /// with --users 4, --context 8192 and 131072; the plan `run` loads, or its refusal in its words. It holds the planner to what it made when the file was last
    /// written, so a change in a plan is seen; it does not compare the planner with any other (the file began as the plans of the per-path planners this one
    /// replaced, and has been written anew since where a plan was meant to change). The recent set (2026-11-25): the plain dense kind at two shapes (36 and 64
    /// layers) and one dense architecture the engine does not run, Qwen3.5 at 0.8B and 9B, Qwen3.8-27B, gemma-4-12B, GLM-5.3-Flash and Qwen3.8-Flash-Next;
    /// and (2026-11-26) the releases larger than one machine, from their recorded headers: GLM-5.3, Qwen3.8-2.4T-A95B, Kimi-K3.1-Pro and DeepSeek-V4.1-Flash.
    /// `GLYD_WRITE_P2A=1` writes the file anew (only where a change in a plan is meant, and the diff of the file is then read).
    #[test]
    fn the_plans_are_the_recorded_ones() {
        let q35 = json::parse(&fixture("qwen35_configs.json")).unwrap();
        let unknown = r#"{"architectures": ["DenseForCausalLM"], "model_type": "dense", "hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8}"#;
        let configs = [
            ("dense-64l", with_rope(DENSE_64L)),
            ("dense-36l", DENSE_36L.to_string()),
            ("Qwen3.5-0.8B", json::to_compact(q35.get("Qwen/Qwen3.5-0.8B").unwrap())),
            ("Qwen3.5-9B", json::to_compact(q35.get("Qwen/Qwen3.5-9B").unwrap())),
            ("Qwen3.8-27B", json::to_compact(q35.get("Qwen/Qwen3.8-27B").unwrap())),
            ("gemma-4-12B", fixture("gemma4_12b_config.json")),
            ("GLM-5.3-Flash", fixture("glm5_next_config.json")),
            ("Qwen3.8-Flash-Next", fixture("qwen4x_config.json")),
            ("dense-unknown", unknown.to_string()),
            ("GLM-5.3", fixture("glm_moe_dsa_config.json")),
            ("Qwen3.8-2.4T-A95B", recorded_release("Qwen3.8-2.4T-A95B").0),
            ("Kimi-K3.1-Pro", recorded_release("Kimi-K3.1-Pro").0),
            ("DeepSeek-V4.1-Flash-1017", recorded_release("DeepSeek-V4.1-Flash-1017").0),
        ];
        let mut got = serde_json::Map::new();
        for (name, text) in &configs {
            let dir = model_dir("p2a", name, text);
            if let Some(t) = released_tensors(name) {
                header_only_checkpoint(&dir, &t);
            }
            // (Qwen3.8-Flash-Next's hummingbird level is its codes pack's beside its weights: the release's precise-v2)
            if *name == "Qwen3.8-Flash-Next" {
                release_pack(&dir.join("pack"), "precise-v2");
            }
            // (the Qwen3.5 layout's penguin is the dynamic route's, planned from the files' headers: the released tensors of each)
            if let Some(t) = qwen35_tensors(name) {
                header_only_checkpoint(&dir, &t);
            }
            // (the releases larger than any one machine's disk, GLM-5.3, Qwen3.8-2.4T-A95B, Kimi-K3.1-Pro and DeepSeek-V4.1-Flash: their tensors from their
            // headers in a planning directory's table of its own (`sizes::GGUF_TENSORS`), no file of their bytes)
            match *name {
                "GLM-5.3" => glm53_table().write(&dir.join(crate::sizes::GGUF_TENSORS)),
                "Qwen3.8-2.4T-A95B" | "Kimi-K3.1-Pro" | "DeepSeek-V4.1-Flash-1017" => recorded_release(name).1.write(&dir.join(crate::sizes::GGUF_TENSORS)),
                _ => {}
            }
            let model = dir.to_str().unwrap();
            let base = dir.parent().unwrap().to_str().unwrap().to_string();
            let variants: Vec<Vec<&str>> = vec![vec![], vec!["--users", "4"], vec!["--context", "8192"], vec!["--context", "131072"]];
            // (a release larger than one machine on the machines that bound it, alone and served to 4: its table of hundreds of thousands of tensors is read at
            // every plan)
            let big = matches!(*name, "GLM-5.3" | "Qwen3.8-2.4T-A95B" | "Kimi-K3.1-Pro" | "DeepSeek-V4.1-Flash-1017");
            let variants: Vec<Vec<&str>> = if big { variants.into_iter().take(2).collect() } else { variants };
            for preset in Machine::PRESETS.into_iter().filter(|p| !big || ["rtx4080s", "l4", "h100", "8xh100"].contains(p)) {
                let nvidia = Machine::preset(preset).unwrap().devices.iter().all(|d| d.vendor == Vendor::Nvidia);
                for level in crate::levels::names_now() {
                    for v in variants.iter().filter(|v| v.is_empty() || nvidia) {
                        let tag = format!("{model}:{level}");
                        let mut words = vec!["fit", tag.as_str(), "--machine", preset];
                        words.extend(v.iter().copied());
                        let said = match planned(&args(&words)) {
                            Ok(f) => match (f.refusal, f.plan.chosen.and_then(|c| c.load)) {
                                (Some(r), _) => serde_json::json!({"refusal": [r.what, r.fix, r.code]}),
                                (None, Some(lp)) => serde_json::from_str(&lp.json(name)).unwrap(),
                                (None, None) => serde_json::json!({"estimate": true}),
                            },
                            Err(r) => serde_json::json!({"refusal": [r.what, r.fix, r.code]}),
                        };
                        let said: serde_json::Value = serde_json::from_str(&said.to_string().replace(&base, "DIR")).unwrap();
                        got.insert(format!("{name} {preset} {level} {}", v.join(" ")).trim_end().to_string(), said);
                    }
                }
            }
        }
        let file = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/plans_p2a.json");
        if std::env::var_os("GLYD_WRITE_P2A").is_some() {
            // (the file is a build's with CUDA 13's kernels alone: one with CUDA 12's runs on more drivers, below)
            if crate::machine::KERNELS12 {
                panic!("write the file on a build with CUDA 13's kernels alone");
            }
            // (a case a line)
            let lines: Vec<String> = got.iter().map(|(k, v)| format!("{}: {v}", serde_json::Value::from(k.as_str()))).collect();
            std::fs::write(&file, format!("{{\n{}\n}}\n", lines.join(",\n"))).unwrap();
            return;
        }
        let want: serde_json::Map<String, serde_json::Value> = serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
        assert_eq!(got.len(), want.len(), "the cases recorded");
        let plans = want.values().filter(|v| v.get("schema").is_some()).count();
        assert!(plans > 300, "{plans} plans recorded");
        // (the file holds a build with CUDA 13's kernels alone. A build that carries CUDA 12's too runs on a preset whose driver is below CUDA 13 and at
        // least this build's floor (the laptop preset's 12.7): there the file's refusal of that driver is this build's plan of the case, which is held
        // only to not refusing the driver)
        let driver_refusal = |v: &serde_json::Value| v["refusal"][0].as_str().is_some_and(|w| w.starts_with("the driver supports CUDA"));
        let admitted = |k: &str| {
            let preset = k.split(' ').nth(1).and_then(Machine::preset);
            preset.is_some_and(|m| {
                m.devices
                    .iter()
                    .filter(|d| d.vendor == Vendor::Nvidia)
                    .all(|d| d.driver < crate::machine::DRIVER_MIN && d.driver >= crate::machine::driver_min(d.cc))
            })
        };
        let mut planned_here = 0;
        for (k, w) in &want {
            if driver_refusal(w) && admitted(k) {
                planned_here += 1;
                assert!(got.get(k).is_some_and(|g| !driver_refusal(g)), "{k}: this build runs on the driver the file's build refused: {:?}", got.get(k));
                continue;
            }
            assert_eq!(got.get(k), Some(w), "{k}");
        }
        assert!(crate::machine::KERNELS12 || planned_here == 0, "{planned_here} cases of drivers this build refuses were not compared");
    }

    /// A machine one does not have says itself whether it has cuBLAS, so its plan never turns on the computer that plans it: every preset says it has none (the
    /// engine never ships cuBLAS; a computer has it where NVIDIA's toolkit is installed), a described machine says so where its JSON does, and this computer's GPU
    /// leaves it to its loader. A preset off Ampere, Ada and Hopper (DGX Spark's GB10) takes the decoded product and its larger chunk only where it says it has cuBLAS.
    #[test]
    fn a_machine_one_does_not_have_says_whether_it_has_cublas() {
        for name in Machine::PRESETS {
            let m = Machine::preset(name).unwrap();
            assert!(m.devices.iter().all(|d| d.cublas == Some(false) && Target::of(d, 0, None).cublas == Some(false)), "{name}");
        }
        let spark = Machine::preset("dgx-spark").unwrap().devices[0].cc;
        assert!(!sizing::dense_product(spark, Some(false)) && sizing::dense_product(spark, Some(true)));
        let said = r#"{"name": "x", "gpus": [{"name": "NVIDIA GB10", "cc": [12, 1], "total_gib": 119, "cublas": true}, {"name": "NVIDIA GB10", "cc": [12, 1], "total_gib": 119}]}"#;
        let m = Machine::from_json(said).unwrap();
        assert_eq!(m.devices.iter().map(|d| d.cublas).collect::<Vec<_>>(), vec![Some(true), Some(false)]);
        assert_eq!(Device::listed("NVIDIA GB10".into(), Vendor::Nvidia, (12, 1), 1, 1, 13000).cublas, None);
    }

    /// `fit` prints the ladder from the level asked for: Gemma 4 12B lossless does not fit a 16 GB card, kestrel does, and the step is a lower level that asks; as JSON too.
    #[test]
    fn fit_shows_every_step_and_the_json_has_them() {
        let dir = model_dir("ladderfit", "gemma-4-12B", &fixture("gemma4_12b_config.json"));
        let f = planned(&args(&["fit", &format!("{}:penguin", dir.to_str().unwrap()), "--machine", "rtx4080s"])).unwrap();
        let l = f.ladder.as_ref().expect("a ladder");
        assert_eq!(l.asked, "penguin");
        let p = l.plan.as_ref().expect("a lower level loads");
        assert!(p.level != "penguin" && l.steps.iter().any(|s| s.kind == StepKind::LowerLevel && s.confirm), "{:?}", l.steps);
        let report = l.report();
        assert!(
            report.starts_with("The ladder, from penguin at ")
                && report.contains("asks first (--yes takes it)")
                && report.contains("not available: the KV cache's type"),
            "{report}"
        );
        let v = l.json();
        assert_eq!((v["as_asked"].as_bool(), v["loads"].as_bool(), v["needs_confirmation"].as_bool()), (Some(false), Some(true), Some(true)));
        assert!(v["steps"].as_array().unwrap().iter().any(|s| s["kind"] == "lower_level" && s["confirm"] == true));
        // (a model that loads as asked has nothing to say)
        let ok = planned(&args(&["fit", &format!("{}:kestrel", dir.to_str().unwrap()), "--machine", "rtx4090"])).unwrap();
        assert!(ok.ladder.as_ref().is_some_and(Ladder::is_as_asked) && ok.ladder.unwrap().report().is_empty());
    }
}
