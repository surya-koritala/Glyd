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
    /// What a kernel streaming the GPU's memory reads, GB/s, as the rate probe measured it on this computer (`rates_of`); None for a preset or a GPU not probed,
    /// whose rate is the maker's figure times `STREAM_OF_SPEC`.
    pub stream: Option<f64>,
    /// The dense bf16 tensor rate (fp32 sums), TFLOPS, from the makers' sheets (`TFLOPS`); 0 where the planner does not know it (no prefill prediction).
    pub tflops: f64,
    /// What decode reached here against the prediction (observed over predicted: p10, p50, p90), from the runs this computer recorded on this GPU (`observed`);
    /// None before any.
    pub observed: Option<[f64; 3]>,
    /// What prompts reached here against the prefill prediction (the median), from the same runs.
    pub observed_prefill: Option<f64>,
    /// The CUDA version the driver supports (13000 for 13.0); 0 where none.
    pub driver: i32,
    /// The GPU shares the computer's memory (Apple Silicon, DGX Spark, Jetson, Strix Halo): its memory is not a tier apart
    /// from RAM, so experts are never "moved to RAM" and the computer's free memory is not counted twice.
    pub unified: bool,
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
            unified: is_unified(&name, vendor),
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

    /// The decode prediction's factor and band on this GPU: p10, p50 and p90 of what runs here reached against their prediction (`observed`), else `SPREAD`.
    pub fn band(&self) -> [f64; 3] {
        self.observed.unwrap_or(SPREAD)
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
        driver,
        unified: is_unified(name, vendor),
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

    /// A machine described in JSON: {"name", "ram_free_gib", "ram_bandwidth", "cores", "gpus": [{"name", "vendor", "cc": [8, 9], "total_gib", "free_gib", "bandwidth", "driver", "unified"}]}.
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
    /// `GLYD_NO_PROBE` is set), and what the runs here reached against their predictions (`observed_on`).
    pub fn probe() -> Machine {
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
    pub window: usize,
    /// Recurrent state per sequence (linear attention, Mamba), bytes.
    pub state: f64,
    /// Latent attention (MLA: `kv_lora_rank`): its cache is one sequence's, never copied, so no states are kept at a prompt's end or at boundaries.
    pub latent_kv: bool,
    /// Bits a weight of the release has (16 for bf16; 8 for FP8; about 4.25 for FP4 releases): every part's but the routed experts'.
    pub release_bits: f64,
    /// Bits a routed expert's weight has as released, where it differs from the rest (Kimi-K3's MXFP4, DeepSeek-V4's FP4 experts:
    /// 4.25); 0: the same as `release_bits`.
    pub expert_release_bits: f64,
    /// The width the routed experts read (the hidden size, or the latent of a latent mixture of experts).
    pub expert_width: usize,
    /// A draft head (multi-token prediction) the engine drafts with.
    pub mtp: bool,
}

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
        let gated = if model_type.contains("nemotron") { 2.0 } else { 3.0 }; // (Nemotron-H's MLPs are relu²: two matrices)
                                                                             // (a latent mixture of experts: the experts work on a narrower vector, projected down and up around them)
        let latent = f(s.latent()).unwrap_or(0.0);
        let expert_width = if latent > 0.0 { latent } else { hf };
        let first_dense = s.first_k_dense_replace.unwrap_or(0);
        // attention
        let (q_lora, kv_lora) = (f(s.q_lora_rank).unwrap_or(0.0), f(s.kv_lora_rank).unwrap_or(0.0));
        let (nope, rope, v_hd) = (f(s.qk_nope_head_dim).unwrap_or(128.0), f(s.qk_rope_head_dim).unwrap_or(64.0), f(s.v_head_dim).unwrap_or(128.0));
        let gate = if s.attn_output_gate.unwrap_or(false) { 2.0 } else { 1.0 };
        let hq = heads as f64;
        let (g_kv, g_hd) = (f(s.num_global_key_value_heads).unwrap_or(kv_heads), f(s.global_head_dim).unwrap_or(hd));
        let attention = |global: bool| -> (f64, f64) {
            if kv_lora > 0.0 {
                let q = if q_lora > 0.0 { hf * q_lora + q_lora * hq * (nope + rope) } else { hf * hq * (nope + rope) };
                (q + hf * (kv_lora + rope) + kv_lora * hq * (nope + v_hd) + hq * v_hd * hf, (kv_lora + rope) * 2.0)
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
        sh.latent_kv = kv_lora > 0.0;
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
                    let (p, kv) = attention(!sliding && pattern.is_none() && s.global_head_dim.is_some());
                    sh.dense += p;
                    if sliding {
                        sh.kv_sliding += kv
                    } else {
                        sh.kv_full += kv
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
        // (FP4 experts beside the rest: DeepSeek-V4's `expert_dtype`, and compressed-tensors' MXFP4 or NVFP4 groups, which Kimi-K3 puts on its
        // routed experts alone and the rest left bf16)
        let format = qc.and_then(|q| q.format.as_deref()).unwrap_or("");
        if s.expert_dtype.as_deref() == Some("fp4") || (quant == "compressed-tensors" && format.contains("fp4")) {
            sh.expert_release_bits = 4.25;
        }
        sh.expert_width = expert_width as usize;
        sh.mtp = s.mtp_layers() > 0;
        Ok(sh)
    }

    /// The same model with its parameters counted from the checkpoint's own tensors (src/sizes.rs: the headers of its files, a directory's or the Hub's by ranges) in
    /// place of an estimate from config.json: each part by its tensors' weights (a packed type's at the release's bits), a vision tower left out, and the
    /// multi-token-prediction head counted only where the model drafts with it. A table with no weights of the parts changes nothing.
    pub fn with_table(mut self, t: &crate::sizes::Table) -> Shape {
        use crate::sizes::Part;
        let p = t.params_by_part(self.layers, self.release_bits.max(1.0));
        let get = |k: Part| p.get(&k).copied().unwrap_or(0.0);
        let dense = get(Part::Attention) + get(Part::Mlp) + get(Part::SharedExperts) + get(Part::Other) + if self.mtp { get(Part::Mtp) } else { 0.0 };
        if get(Part::Embedding) + get(Part::Head) + dense + get(Part::RoutedExperts) <= 0.0 {
            return self;
        }
        (self.embed, self.head, self.dense, self.routed) = (get(Part::Embedding), get(Part::Head), dense, get(Part::RoutedExperts));
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
const POSITIONS: f64 = 1024.0;
/// What a kernel streaming the GPU's memory reads of the maker's figure: 697.35 of 736 GB/s, measured on an RTX 4080 SUPER (`glyd-engine rates`, 512 MiB, 10
/// passes); the rate of a preset or of a GPU the probe has not measured.
pub const STREAM_OF_SPEC: f64 = 0.9475;
/// Of the streamed rate, what decode's reads reach: the lossy levels' packs all of it, the lossless layout (decoded in the product) 0.72; and a decode step's time
/// besides its reads, 35 µs a layer (its kernels' launches and the gaps between them). Measured on an RTX 4080 SUPER at about 1,000 positions: gemma-4-12B at wren
/// 86.2 tokens/s and kestrel 63.3, Qwen3.5-0.8B at kestrel 561.9 and wren 657.9 (a 35 µs layer and the stream's rate predict them within 4%); the lossless figure from
/// Qwen3-32B on an RTX A6000 (11.8 tokens/s).
const EFF_LOSSY: f64 = 1.0;
const EFF_LOSSLESS: f64 = 0.72;
const LAYER_S: f64 = 35e-6;
/// Of the computer's memory bandwidth, what the CPU's expert products reach; the time a mixture-of-experts layer with its experts in RAM adds for the GPU and host
/// to meet besides `LAYER_S` (calibrated on Condor's Qwen3.8-Flash-Next on an RTX 4080 SUPER, 0.20 ms a layer in all).
const EFF_CPU: f64 = 0.70;
const SYNC_PER_MOE_LAYER: f64 = 0.165e-3;
/// A prompt's tokens a second: 2 x the parameters a token multiplies over the GPU's tensor rate (`Device::tflops`) at `PROMPT_EFF` of it, and 1.11 µs a token a
/// layer besides (attention, norms, the recurrent layers' scans). Both from two prompts of 936 and 1,002 tokens on an RTX 4080 SUPER: gemma-4-12B at wren 3,570
/// tokens/s, Qwen3.5-0.8B at kestrel 27,073.
const PROMPT_EFF: f64 = 0.92;
const PROMPT_LAYER_S: f64 = 1.11e-6;
/// The band a prediction is given in where no run on the GPU was observed (p10 and p90 against the prediction): the runs these figures are calibrated on (above, and
/// `predictions_match_what_was_measured`'s) came to 0.91-1.02 of their predictions on the GPUs they were measured on; another GPU's kernels may differ more.
pub const SPREAD: [f64; 3] = [0.90, 1.0, 1.10];
/// The share of expert reads a cache holding a share `f` of the experts serves: experts are used unevenly (Condor: 70-76% at about a quarter).
const HIT_EXPONENT: f64 = 0.23;
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
/// The Qwen3 path's fixed workspaces: the CUDA context and the library's own buffers, cuBLAS's and the pool's, 1,400 MiB (600 + 400 + 400) measured
/// as the path's total before the reserve was one model, less that reserve; a prompt chunk's activations are on top (`sizing::work`).
pub const QWEN3_WORK: u64 = (1400 << 20) - CONTEXT_FLOOR;
/// The dynamic route's work on each GPU: the activations, the scratch, the graphs and the KV cache of its context together, which this route has not
/// split apart yet (3 GiB and 5% of the GPU's free memory in all with the reserve, as it kept before the planner drove it).
pub fn dynamic_work_of(free: u64) -> u64 {
    (3 << 30) - CONTEXT_FLOOR + free / 20
}

/// What a GPU has for the dynamic route's weights: its free memory less the reserve and the route's work.
pub fn dynamic_room_of(free: u64) -> u64 {
    room_of(free).saturating_sub(dynamic_work_of(free))
}

/// Condor's (Qwen3.8-Flash-Next's) memory terms on a GPU, measured on an RTX 4080 SUPER (src/qwen4x_serve.rs): what its first prompt and the first decode use
/// beyond the load (0.886 GiB), a chunk's scratch a row (0.335 MB), the keys and values of a token (28 KB) and the margin kept free (0.3 GiB); the places of
/// its expert cache take the rest.
pub const CONDOR_FIRST_PROMPT_GIB: f64 = 0.886;
pub const CONDOR_CHUNK_GIB_PER_ROW: f64 = 0.000335;
pub const CONDOR_KV_GIB_PER_TOKEN: f64 = 28e-6;
pub const CONDOR_MARGIN_GIB: f64 = 0.3;

/// The GiB of always-on matrices Condor holds at `layout` (measured at i6 with 0.28 GiB left free after the load, i5 and i4 short of that; i6 0.15 more since
/// the drafts' graphs ran out at 0.11-0.15 GiB free on the 4080), and 0.35 more where the attention layers and the LM head are held at a layout of their own.
pub fn condor_always_gib(layout: crate::weights::Layout, hi_differs: bool) -> f64 {
    use crate::weights::Layout;
    let base = match layout {
        Layout::Int4 => 2.6,
        Layout::Int5 => 3.05,
        _ => 3.55,
    };
    base + if hi_differs { 0.35 } else { 0.0 }
}

/// The context the lossy paths take where none is asked, if it fits, and the least a plan takes.
pub const LOSSY_CTX: usize = 32768;
pub const LOSSY_MIN_CTX: usize = 4096;
/// The computer's memory kept for the system and the engine's own besides the experts pinned in it.
pub const HOST_MARGIN: u64 = 6 << 30;
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
/// 2,688); the families the description builder maps (src/hb_pack.rs) keep every other matrix as at penguin.
fn level_bits(level: &str, sh: &Shape, d: &Device) -> Option<(f64, f64)> {
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
        if sh.arch == "NemotronHForCausalLM" {
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
    // (a model with recurrent states and no latent cache keeps a second copy of each sequence's states at its prompt's end, and the pool of them kept at block
    // boundaries, src/dynamic/snaps.rs: as the loader makes them)
    let marked = if sh.state > 0.0 && !sh.latent_kv { sh.state } else { 0.0 };
    let pool = if marked > 0.0 { crate::dynamic::snaps::count(sh.state as usize) as f64 * sh.state } else { 0.0 };
    let kv_at = |ctx: usize| (sh.kv_full * ctx as f64 + sh.kv_sliding * ctx.min(sh.window.max(1)) as f64 + sh.state + marked) * users + pool;
    let same: Vec<usize> = (0..m.devices.len()).filter(|&i| m.devices[i].vendor == d0.vendor).collect();
    // (the draft head counts where the family's path drafts with it; a model no path runs yet is taken to)
    let gain = gain_of(sh, fam.is_none_or(|f| f.drafts));
    let step = sh.layers as f64 * LAYER_S;
    // (every level the family runs, or the one asked)
    let levels: Vec<&'static str> = levels::LEVELS
        .iter()
        .map(|l| l.name)
        .filter(|n| ask.level.map_or(levels::names_now().contains(n), |a| a == *n))
        .filter(|n| fam.is_none_or(|f| f.allows(n)))
        .collect();
    let mut options = Vec::new();
    for &level in &levels {
        let Some((bits, ebits)) = level_bits(level, sh, d0) else { continue };
        let rest = (sh.embed + sh.head + sh.dense) * bits / 8.0;
        let experts = sh.routed * ebits / 8.0;
        let weights = rest + experts;
        let eff = eff_of(level, sh);
        let read = read_of(sh, bits, ebits, POSITIONS);
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
            let generic_hb = level == "hummingbird" && sh.arch != "NemotronHForCausalLM";
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
}

impl Target {
    pub fn of(d: &Device, ordinal: i32, host: Option<u64>) -> Target {
        Target { ordinal, name: d.name.clone(), cc: d.cc, total: d.total, free: d.free, host }
    }
}

/// `fit --json`'s schema: its `schema` member.
pub const SCHEMA: &str = "glyd-plan/1";

/// The engine path that loads a plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Qwen3 dense, lossless (src/model.rs).
    Qwen3,
    /// Qwen3.5, 3.6 and 3.8 dense at a lossy level.
    Qwen35,
    NemotronH,
    Gemma4,
    /// The dynamic route (src/dynamic): Kimi-K3, GLM-5.3-Flash, DeepSeek-V4, Muse Spark and Nemotron-H lossless, over one GPU or several.
    Dynamic,
    /// Condor (`qwenserve`, src/qwen4x_serve.rs): Qwen3.8-Flash-Next with its experts in the computer's memory and a cache of them on each GPU.
    Condor,
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
    /// The bytes the GPUs hold of the model (its weights, and the places' experts).
    pub gpu_bytes: u64,
    /// The last layers whose dense matrices are in the computer's memory (src/dense_ram.rs: one stage, P2d's rung after the experts), and their bytes there.
    pub dense: usize,
    pub dense_bytes: u64,
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
            "kv_bytes": self.kv, "work_bytes": self.work, "draft_bytes": self.draft,
            "reserve_bytes": self.reserve, "gpu_total_bytes": self.gpu_total(),
            "experts": experts,
            "spread": self.spread.as_ref().map(|s| serde_json::json!({
                "cuts": s.cuts, "devices": s.devices, "across": s.across, "coded": s.coded, "twelve": s.twelve,
                "places": s.places, "experts_per_layer": s.experts_per_layer, "host_bytes": s.host_bytes,
            })),
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

    /// The plan in words, for the line `run` says before it loads and `fit` prints.
    pub fn text(&self) -> String {
        let mut s = format!(
            "{} level, {} tokens of context{}: {} of weights on the GPU, {} of cache, {} for the engine's work{}, {} kept free; {} of the {}'s {} free",
            self.level,
            self.context,
            if self.users > 1 { format!(" for each of {} users", self.users) } else { String::new() },
            gb(self.weights_on_gpu()),
            gb(self.kv),
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

/// Bytes a decode step reads at `bits` (`ebits` the routed experts') at `positions` positions: the matrices it reads whole (the head, the embedding where it is
/// tied, the dense ones), a token's routed experts, and the keys and values.
fn read_of(sh: &Shape, bits: f64, ebits: f64, positions: f64) -> f64 {
    ((sh.head.max(if sh.head == 0.0 { sh.embed } else { 0.0 }) + sh.dense) * bits + sh.routed_active() * ebits) / 8.0
        + sh.kv_full * positions
        + sh.kv_sliding * positions.min(sh.window.max(1) as f64)
}

/// Of the streamed rate, what decode's reads reach at `level`.
fn eff_of(level: &str, sh: &Shape) -> f64 {
    if level == "penguin" && sh.release_bits > 8.0 {
        EFF_LOSSLESS
    } else {
        EFF_LOSSY
    }
}

/// A prompt's tokens a second on `d` with every weight on the GPU (`PROMPT_EFF`, `PROMPT_LAYER_S`): None where the GPU's tensor rate is not known.
pub fn prefill_of(sh: &Shape, d: &Device) -> Option<f64> {
    (d.tflops > 0.0).then(|| 1.0 / (2.0 * (sh.dense + sh.routed_active()) / (d.tflops * 1e12 * PROMPT_EFF) + sh.layers as f64 * PROMPT_LAYER_S))
}

/// Decode tokens/s of a plan `run` loads at `positions` positions, from the model's parameters and the machine's rates (the planner's model, for a placement the
/// loader chose), before what runs here observed refines it.
fn speed_at(sh: &Shape, m: &Machine, main: usize, lp: &LoadPlan, positions: f64) -> f64 {
    let d0 = &m.devices[main];
    let Some((bits, ebits)) = level_bits(lp.level, sh, d0) else { return 0.0 };
    let eff = eff_of(lp.level, sh);
    let read = read_of(sh, bits, ebits, positions);
    let gain = gain_of(sh, lp.kind == Kind::Condor);
    let step = sh.layers as f64 * LAYER_S;
    if let Some(sp) = &lp.spread {
        // (the GPUs it is over: stages read in turn, a layer across them reads on all at once; the experts the GPUs do not hold are the CPU's)
        let rates: Vec<f64> = sp.devices.iter().filter_map(|&d| m.devices.get(d as usize)).map(Device::rate).collect();
        let n = rates.len().max(1) as f64;
        let active = sh.routed_active() * ebits / 8.0;
        let hit = if sp.host_bytes > 0 {
            (sp.places.iter().min().copied().unwrap_or(0) as f64 / sp.experts_per_layer.max(1) as f64).min(1.0).powf(HIT_EXPONENT)
        } else {
            1.0
        };
        let on_gpu = read - active * (1.0 - hit);
        let gpu_t = if sp.across {
            on_gpu / (n * rates.iter().copied().fold(f64::MAX, f64::min) * eff)
        } else {
            rates.iter().map(|r| on_gpu / n / (r * eff)).sum::<f64>()
        } + step
            + 0.1e-3 * (n - 1.0);
        let cpu_t = active * (1.0 - hit) / (m.ram_bandwidth * GB * EFF_CPU);
        // (the last layers' dense matrices in the computer's memory, src/dense_ram.rs: each copied to the GPU before it runs, every token)
        let link_t = sp.dense_bytes as f64 / (LINK_GBS * GB);
        return gain / (gpu_t.max(cpu_t) + link_t + if sp.host_bytes > 0 { SYNC_PER_MOE_LAYER * sh.moe_layers as f64 } else { 0.0 });
    }
    match &lp.experts {
        None => gain / (read / (d0.rate() * eff) + step),
        Some(c) => {
            let hit = (c.slots as f64 / c.experts.max(1) as f64).min(1.0).powf(HIT_EXPONENT);
            let active = sh.routed_active() * ebits / 8.0;
            let gpu_t = (read - active * (1.0 - hit)) / (d0.rate() * eff) + step;
            let cpu_t = active * (1.0 - hit) / (m.ram_bandwidth * GB * EFF_CPU);
            gain / (gpu_t.max(cpu_t) + SYNC_PER_MOE_LAYER * sh.moe_layers as f64)
        }
    }
}

fn speed_of(sh: &Shape, m: &Machine, main: usize, lp: &LoadPlan) -> f64 {
    speed_at(sh, m, main, lp, POSITIONS)
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
fn ideal_speed(sh: &Shape, m: &Machine, main: usize, level: &str, drafts: bool) -> f64 {
    let d0 = &m.devices[main];
    let Some((bits, ebits)) = level_bits(level, sh, d0) else { return 0.0 };
    gain_of(sh, drafts) / (read_of(sh, bits, ebits, POSITIONS) / (d0.rate() * eff_of(level, sh)) + sh.layers as f64 * LAYER_S)
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
                    let (ideal, got) = (ideal_speed(sh, m, main, level, fam.drafts), speed_of(sh, m, main, &all_on_gpu));
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
                    let (before, after) = (ideal_speed(sh, m, main, asked, fam.drafts), ideal_speed(sh, m, main, level, fam.drafts));
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
fn candidate_of(lp: LoadPlan, sh: &Shape, m: &Machine, main: usize) -> Candidate {
    let tok_s = speed_of(sh, m, main, &lp);
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
/// `measure` (`costs::probe_rates`: about a second) and kept. None where they cannot be measured.
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
pub fn observed_for(root: &std::path::Path, model: &str, level: &str, gpu: &str) -> Option<[f64; 3]> {
    let v = recall_in(root, model, level, gpu)?;
    band_of(v["observed"].as_array()?.iter().filter_map(|s| s["decode_ratio"].as_f64()).collect())
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
    let kind = if sh.is_moe() {
        format!("MoE, {:.1}B parameters, {:.1}B active", sh.params() / 1e9, (sh.params() - sh.routed + sh.routed_active()) / 1e9)
    } else {
        format!("dense, {:.1}B parameters", sh.params() / 1e9)
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
    let observed = m.devices.iter().any(|d| d.observed.is_some());
    s += &format!(
        "(decode speeds are predictions for one user at about 1,000 positions, from the GPU's memory {}, with the p10-p90 band of {}; prompts' from its tensor rate)\n",
        if m.devices.iter().any(|d| d.stream.is_some()) { "as measured here" } else { "by the maker's figure" },
        if observed { "the runs observed here" } else { "the calibration runs (no run observed here yet)" }
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
    config: &str,
    level: Option<&'static str>,
    fam: &'static crate::support::Support,
    sh: &Shape,
    machine: &Machine,
    users: usize,
    probed: bool,
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
                let c = candidate_of(lp, sh, machine, main);
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

/// A model's shape: its config's, with its own tensors where its headers can be had (before any download, by ranges), else config.json's estimate scaled to the
/// index's total. `fit` and `run`'s degrade gate both price the ladder's steps on it.
pub fn shape_of(model: &str, config: &str) -> Result<Shape, String> {
    let sh = Shape::from_config(config)?;
    Ok(match crate::sizes::Table::find(model) {
        Some(t) => sh.with_table(&t),
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
    let config = crate::cli::config_text(model)?;
    // the family as run treats it: a level it does not run at is refused in run's words
    let fam = match level {
        Some(l) => crate::support::check(&config, model, &name, l)?,
        None => crate::support::row(crate::support::family(&config, &name)?),
    };
    if fam.levels_now(model).is_empty() {
        crate::support::check(&config, model, &name, "penguin")?;
    }
    let sh = shape_of(model, &config).map_err(|e| Refusal::unsupported(format!("{name}: {e}"), ""))?;
    let context: Option<usize> = a.flag("context").map(|c| c.parse()).transpose().map_err(|_| Refusal::new("--context takes a number", "--context 8192"))?;
    crate::support::context_check(&name, fam, sh.max_ctx, context)?;
    let users: usize = a.num("users", 1usize)?;
    let ask = Ask { level, context, users, family: Some(fam), qat: levels::is_qat(model) };
    let mut p = plan(&sh, &machine, ask);
    let mut refusal = None;
    // where run's own sizing covers the family: the plan `run` would load at each level, or the refusal it would give (the same functions: `sizing::size`)
    let probed = a.flag("machine").is_none();
    if let Some((options, refusals)) = exact_options(a, model, &config, level, fam, &sh, &machine, users, probed) {
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
                if let Some(b) = observed_for(&root, model, o.level, &d.name) {
                    (o.tok_s, o.range) = (o.prior * b[1], (o.prior * b[0], o.prior * b[2]));
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
    Ok(Planned { model: model.to_string(), name, family: fam, shape: sh, machine, plan: p, refusal, ladder })
}

fn fit(a: &crate::args::Args) -> Result<(), crate::ui::Refusal> {
    let f = planned(a)?;
    let (model, p) = (f.model.as_str(), &f.plan);
    if a.has("json") {
        let mut v: serde_json::Value = match &p.chosen {
            Some(c) => serde_json::from_str(&c.json(model, &f.machine)).unwrap_or_default(),
            None => serde_json::json!({"schema": SCHEMA, "model": model, "plan": null, "notes": p.notes}),
        };
        v["degrade"] = f.ladder.as_ref().map_or(serde_json::Value::Null, Ladder::json);
        println!("{v}");
    } else {
        print!("{}", text(model, &f.shape, &f.machine, p));
        if let Some(l) = &f.ladder {
            print!("{}", l.report());
        }
        if let Some(n) = p.chosen.as_ref().and_then(|c| remembered(&crate::store::cache_root(), model, c)) {
            println!("note: {n}");
        }
        if let Some(n) = f.machine.devices.iter().find_map(|d| crate::machine::cuda12_note(d.driver, crate::dev::has_kernels12())) {
            println!("note: {n}");
        }
        if f.family.hummingbird_packed_only && p.chosen.as_ref().is_some_and(|c| c.level == "hummingbird") {
            if f.family.family == crate::support::Family::NemotronH {
                println!("note: the hummingbird level comes as a pre-packed download: glyd/{}-hummingbird", f.name);
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

/// The loaders' sizing, moved here from src/cli.rs and src/fit.rs (P2f): the plan `run`, `serve`, `preflight` and `fit` load and print.
pub mod sizing {
    use super::{reserve_of, Kind, LoadPlan, Target, GEMMA_CHUNK, GEMMA_DRAFT_BYTES, HOST_MARGIN, LOSSY_CTX, LOSSY_MIN_CTX, LOSSY_WORK, QWEN3_WORK};
    use crate::args::Args;
    use crate::cli::{cannot, gemma_draft, gemma_wren, level, level_name, shape_checks, short_name, supported, unrun_fix, users_of};
    use crate::dev::Gpu;
    use crate::ffi::Blas;
    use crate::levels::{self, layout_for};
    use crate::nemotron::{pack_layout, ConfigNemotronH, NhKind};
    use crate::offload::CachePlan;
    use crate::qwen35::{Config35, Pack35};
    use crate::store;
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
            Layout::Hummingbird => 2.99, // (the experts of the recipe hb40: 205 of a layer's 512 at 3.625 and the rest at 2.5625; the directory's own bytes are what a plan uses where it has them)
        }
    }

    /// The weights on the GPU in `layout` (every matrix packed, the embedding too), with the norms and tables.
    pub fn weights(cfg: &Config, layout: Layout) -> u64 {
        (cfg.matrix_bytes() as f64 * bits(layout) / 16.0) as u64 + (cfg.layers * cfg.hidden * 4 * 2) as u64 + (cfg.max_pos * cfg.head_dim * 4) as u64
    }

    /// A token's keys and values in the lossless cache, all layers: 12 bits a value and a page's header.
    pub fn kv_per_token(cfg: &Config) -> u64 {
        ((cfg.layers * 2 * cfg.kv_heads * cfg.head_dim) as f64 * 1.5 * 1.06) as u64
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

    /// The parts of a model of `layers` layers split over GPUs (`free`: the GiB free of each part's GPU, a GPU's own shared among the parts on it): each part's first layer after the first part's (`cuts`) as `--split` gives
    /// them, else the split that leaves the most places a layer on the part with the fewest (`room(part, layers)`: the places a layer part `part` has room for with that many layers; the first part holds `least` layers at
    /// least, the n-gram embedding's being there). None where no split leaves a part a place a layer.
    pub fn split_layers(layers: usize, parts: usize, least: usize, room: &dyn Fn(usize, usize) -> f64) -> Option<Vec<usize>> {
        // (at least `s` places a layer on every part: each part takes as many layers as leave it that, the last part the rest)
        let fit = |s: f64| -> Option<Vec<usize>> {
            let (mut cuts, mut at) = (Vec::new(), 0);
            for p in 0..parts - 1 {
                let k = (1..=layers - at - (parts - 1 - p)).rev().find(|&k| room(p, k) >= s)?;
                if p == 0 && k < least {
                    return None;
                }
                at += k;
                cuts.push(at);
            }
            (room(parts - 1, layers - at) >= s).then_some(cuts)
        };
        let (mut lo, mut hi) = (1.0, 1e5);
        fit(lo)?;
        for _ in 0..40 {
            let mid = (lo + hi) / 2.0;
            if fit(mid).is_some() {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        fit(lo)
    }

    /// What Condor's parts are sized from (`qwenserve`, src/qwen4x_serve.rs): the GiB free on its one GPU (`free`) or on each part's (`frees`, a GPU's own shared among the
    /// parts on it), the always-on matrices, a part's own scratch (`fixed`) and the whole reserve of one GPU (`reserve`), the context, the LM head (GiB), the bytes of an
    /// expert's place, and the experts a layer.
    pub struct CondorRoom<'a> {
        pub free: f64,
        pub frees: &'a [f64],
        pub always: f64,
        pub fixed: f64,
        pub reserve: f64,
        pub ctx: usize,
        pub head: f64,
        pub place: usize,
        pub experts: usize,
    }

    /// Why Condor's parts cannot be made: no split leaves a part a place a layer, or the split asked is not one.
    #[derive(Debug, PartialEq)]
    pub enum CondorShort {
        NoSplit,
        BadSplit(Vec<usize>),
    }

    /// Condor's parts over `gpus`: (GPU, its layers, places a layer, GB of places), the first part's first. One GPU: the places take what the reserve leaves (`cache_gb` at
    /// most); several: the layers split as `cuts` gives them, else as `split_layers` evens the places, each part's room its GPU's free memory less its scratch (a context of
    /// its own more where it shares the GPU), its layers' share of the always-on matrices and of the context's caches, and the head on the last.
    #[allow(clippy::too_many_arguments)]
    pub fn condor_parts(
        gpus: &[i32],
        r: &CondorRoom,
        layers: usize,
        with_experts: usize,
        mtp: bool,
        least: usize,
        cuts: Option<Vec<usize>>,
        cache_gb: Option<f64>,
    ) -> Result<Vec<(i32, (usize, usize), usize, f64)>, CondorShort> {
        use crate::plan::CONDOR_KV_GIB_PER_TOKEN as PER_TOKEN;
        let gib = (1u64 << 30) as f64;
        let per_gb = |gb: f64, n: usize| ((gb * 1e9) / (n as f64 * r.place as f64)) as usize;
        if gpus.len() == 1 {
            let fit = (r.free - r.reserve).max(0.5) * gib / 1e9;
            let gb = cache_gb.unwrap_or(fit).min(fit);
            return Ok(vec![(gpus[0], (0, layers), per_gb(gb, with_experts + mtp as usize), gb)]);
        }
        let last = gpus.len() - 1;
        let room_gib = |p: usize, n: usize| {
            r.frees[p]
                - r.fixed
                - if gpus[..p].contains(&gpus[p]) { 0.25 } else { 0.0 }
                - (r.always - r.head + r.ctx as f64 * PER_TOKEN) * n as f64 / layers as f64
                - if p == last { r.head } else { 0.0 }
        };
        let places = |p: usize, n: usize| room_gib(p, n).max(0.5) * gib / ((n + (p == last && mtp) as usize) as f64 * r.place as f64);
        let cuts = match cuts {
            Some(c) => c,
            None => split_layers(layers, gpus.len(), least, &places).ok_or(CondorShort::NoSplit)?,
        };
        let bounds: Vec<usize> = std::iter::once(0).chain(cuts.iter().copied()).chain(std::iter::once(layers)).collect();
        if cuts.len() != last || bounds.windows(2).any(|w| w[0] >= w[1]) || bounds[1] < least {
            return Err(CondorShort::BadSplit(cuts));
        }
        Ok((0..gpus.len())
            .map(|p| {
                let (lo, hi) = (bounds[p], bounds[p + 1]);
                let held = hi - lo + (p == last && mtp) as usize;
                let fit = (places(p, hi - lo) as usize).min(r.experts);
                let slots = match cache_gb.filter(|&g| g > 0.0) {
                    Some(all) => per_gb(all, with_experts + mtp as usize).min(fit),
                    None => fit,
                };
                (gpus[p], (lo, hi), slots, (slots * held * r.place) as f64 / 1e9)
            })
            .collect())
    }

    /// The plan of a Condor load: the always-on matrices and the places of each part's cache are the weights on the GPUs, the context's keys and values, the first prompt's
    /// and a chunk's scratch are the cache and the work, and what is kept free is the reserve (`plan::CONDOR_*`; the pack's experts are in the computer's memory).
    #[allow(clippy::too_many_arguments)]
    pub fn condor_plan(
        layout: Layout,
        always_gib: f64,
        parts: &[(i32, (usize, usize), usize, f64)],
        ctx: usize,
        chunk: usize,
        stream_gib: f64,
        headroom_gib: f64,
        free: f64,
        host_bytes: u64,
        experts: usize,
    ) -> LoadPlan {
        use crate::plan::{CONDOR_CHUNK_GIB_PER_ROW, CONDOR_FIRST_PROMPT_GIB, CONDOR_KV_GIB_PER_TOKEN, CONDOR_MARGIN_GIB};
        let gib = (1u64 << 30) as f64;
        let places: u64 = parts.iter().map(|p| (p.3 * 1e9) as u64).sum();
        let gpu_bytes = (always_gib * gib) as u64 + places;
        let names: Vec<String> = parts.iter().map(|p| crate::dev::Dev::probe(p.0).map_or_else(|_| format!("GPU {}", p.0), |g| g.name)).collect();
        LoadPlan {
            kind: Kind::Condor,
            family: "Qwen3.8-Flash-Next",
            level: crate::levels::name_of(layout),
            layout,
            context: ctx,
            users: 1,
            chunk,
            weights: gpu_bytes + host_bytes,
            bf16: 0,
            kv: (ctx as f64 * CONDOR_KV_GIB_PER_TOKEN * gib) as u64,
            work: ((CONDOR_FIRST_PROMPT_GIB + chunk as f64 * CONDOR_CHUNK_GIB_PER_ROW + stream_gib) * gib) as u64,
            draft: 0,
            reserve: ((CONDOR_MARGIN_GIB + headroom_gib) * gib) as u64,
            free: free as u64,
            gpu: names.join(", "),
            experts: None,
            spread: Some(super::Spread {
                cuts: parts.iter().skip(1).map(|p| (p.1).0).collect(),
                devices: parts.iter().map(|p| p.0).collect(),
                across: false,
                coded: false,
                twelve: false,
                places: parts.iter().map(|p| p.2).collect(),
                experts_per_layer: experts,
                host_bytes,
                gpu_bytes,
                dense: 0,
                dense_bytes: 0,
            }),
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
        Ok(fit_table_with(&crate::sizes::Table::from_dir(dir).map_err(crate::ffi::Error)?, nl, room, hbm, twelve, &split::hb_layer_bytes(dir, nl)?))
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
        fit_table_with(table, nl, room, hbm, twelve, &[])
    }

    /// `fit_table`, each layer's bytes more by `extra`'s (a hummingbird directory's experts, which no table lists: `split::hb_layer_bytes`).
    #[allow(clippy::type_complexity)]
    pub fn fit_table_with(
        table: &crate::sizes::Table,
        nl: usize,
        room: &[(i32, u64)],
        hbm: bool,
        twelve: bool,
        extra: &[u64],
    ) -> ((Vec<u64>, u64), Option<(Vec<usize>, Vec<i32>)>, bool, bool) {
        let bytes = |f8: u64, bf16: u64| {
            let (mut per, base) = split::layer_bytes_table(table, nl, f8, bf16, &|_| true);
            per.iter_mut().zip(extra).for_each(|(p, e)| *p += e);
            (per, base)
        };
        if hbm {
            for (f8, twelve) in [(split::PLAIN_PCT, twelve), (split::PLAIN_PCT, false), (split::CODED_PCT, twelve)] {
                let b = bytes(f8, if twelve { split::TWELVE_PCT } else { split::TIERED_PCT });
                if let Some(c) = plan_cuts(&b.0, b.1, room) {
                    return (b, Some(c), f8 == split::CODED_PCT, twelve);
                }
            }
        }
        let coded = bytes(split::CODED_PCT, split::TIERED_PCT);
        let cut = plan_cuts(&coded.0, coded.1, room);
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
        let n = dense.len();
        let foot = |l: usize, p: u64| dense[l] + if moe[l] { p * expert } else { 0 };
        // (the layers in order, each GPU filled to its room, with p places a layer: where each stage starts; None where the GPUs do not hold them)
        let assign = |p: u64| -> Option<Vec<usize>> {
            let (mut starts, mut g, mut used) = (vec![0usize], 0usize, base / 2);
            for l in 0..n {
                if used + foot(l, p) > room.get(g)?.1 {
                    g += 1;
                    starts.push(l);
                    used = 0;
                }
                used += foot(l, p);
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
    ) -> Option<(Vec<usize>, Vec<i32>, Vec<usize>, usize, u64)> {
        if let Some((c, d, p)) = places_plan(dense, base, moe, room, expert, experts, least) {
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

    /// The places a layer gets out of `budget` bytes of the GPU's memory for the cache: as many as fit, at most all of its experts; None where fewer than `least` fit.
    pub fn places_for(budget: u64, layers: usize, expert_bytes: u64, experts: usize, least: usize) -> Option<usize> {
        let each = (layers as u64).checked_mul(expert_bytes)?;
        let n = (budget.checked_div(each)? as usize).min(experts);
        (n >= least.min(experts)).then_some(n)
    }

    /// The expert cache that `budget` bytes of the GPU's memory buy for the model `c` at `level`: places a layer for each of its mixture-of-experts layers, as many as fit (all of a layer's experts at most); None where
    /// fewer than `least` fit.
    pub fn cache_plan(c: &ConfigNemotronH, level: Layout, budget: u64, least: usize) -> crate::ffi::Result<Option<CachePlan>> {
        let (layers, experts, expert_bytes) = (c.layers_of(NhKind::Moe), c.moe.experts, crate::moe_nh::expert_bytes(c, level)?);
        Ok(places_for(budget, layers, expert_bytes, experts, least).map(|slots| CachePlan {
            layers,
            experts,
            slots,
            expert_bytes,
            in_place: level == Layout::Hummingbird,
            ram: (layers * experts) as u64 * expert_bytes,
        }))
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
        // (the tokens each sequence has room for where all `users` hold the context, and where one does)
        let all = (free.saturating_sub(fixed + n.per_seq * users) / (per * users)) as usize;
        let one = (free.saturating_sub(fixed + n.per_seq) / per) as usize;
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
            return Err(Short { need: fixed + seqs * (n.per_seq + at as u64 * per), at, most });
        }
        let left = free.saturating_sub(n.weights + n.draft + reserve + users * (n.per_seq + ctx as u64 * per));
        let chunk = n.chunks.iter().copied().find(|c| c.1 <= left).or(n.chunks.last().copied()).unwrap_or((512, 0));
        Ok(n.plan_at(ctx, chunk, free, gpu))
    }

    impl Needs {
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
                kv: users * (self.per_seq + ctx as u64 * self.per_token.max(1)),
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
            "{name} at the {level} level takes about {} of GPU memory at {} tokens of context{}, and the {gpu} has {} free",
            gib(s.need),
            s.at,
            if users > 1 { format!(" for each of {users} users") } else { String::new() },
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

    /// A Qwen3 model's needs at `layout` for `users` sequences: its packed weights, its keys and values (12 bits a value), its chunks' work with cuBLAS (`cublas`) or without.
    pub(crate) fn qwen3_needs(cfg: &Config, layout: Layout, users: usize, cublas: bool) -> Needs {
        let first = if cublas { 4096 } else { 2048 };
        Needs {
            kind: Kind::Qwen3,
            family: "Qwen3",
            layout,
            weights: weights(cfg, layout),
            bf16: cfg.matrix_bytes(),
            per_seq: 0,
            per_token: kv_per_token(cfg),
            users: users.max(1),
            fewer: false,
            draft: 0,
            chunks: [first, 2048, 1024, 512].into_iter().filter(|&c| c <= first).map(|c| (c, work(cfg, c, cublas))).collect(),
            want: CTX_WANT.min(cfg.max_pos),
            least: CTX_MIN,
            cap: cfg.max_pos,
            grain: 256,
            page: 1,
        }
    }

    /// The plan of a Qwen3 model on a GPU that has `free` bytes (`gpu` its name), for `users` sequences at once, at the context `asked`, else the model's own up to
    /// `CTX_WANT`, less where the room is short (not below `CTX_MIN`).
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
        let n = qwen3_needs(cfg, layout, users, cublas);
        one_gpu(&n, free, asked, gpu).map_err(|s| too_big(name, level_name(layout), gpu, free, n.users, &s, asked, false, None))
    }

    /// A dense product for the long prompts (cuBLAS where it is, else the engine's own on Ampere, Ada and Hopper): the decoded matrix and the larger chunk that goes with it.
    pub(crate) fn dense_product(cc: (i32, i32)) -> bool {
        Blas::present() || (matches!(cc.0, 8 | 9) && crate::tune!("GLYD_ENGINE_GEMM").map_or(true, |v| v != "lib"))
    }

    /// The GPU `run` picked, as a plan sees it, and the computer's memory the experts may take.
    pub(crate) fn target(gpu: &Gpu) -> Target {
        Target { ordinal: gpu.ordinal, name: gpu.name.clone(), cc: gpu.cc, total: gpu.total, free: gpu.free, host: crate::offload::host_memory() }
    }

    /// The layout and the fit of a model on a GPU: what `prepare` settles before it downloads a thing, and `preflight` before it says the engine will run the model. `packed`: the layout of a
    /// directory `pack` made, which is the layout there.
    pub(crate) fn qwen3_plan(a: &Args, cfg: &Config, name: &str, gpu: &Gpu, packed: Option<Layout>, users: usize) -> Done<(Layout, String, LoadPlan)> {
        plan_for(a, cfg, name, &target(gpu), packed, users)
    }

    pub(crate) fn plan_for(a: &Args, cfg: &Config, name: &str, t: &Target, packed: Option<Layout>, users: usize) -> Done<(Layout, String, LoadPlan)> {
        let asked = if a.has("context") || a.has("ctx") { Some(a.either("context", "ctx", 0usize)?) } else { None };
        crate::support::context_check(name, crate::support::row(crate::support::Family::Qwen3), cfg.max_pos, asked)?;
        let (layout, why) = match packed {
            Some(l) => (l, "as packed".to_string()),
            None => choose_layout(a.layout()?, t.cc, t.total, cfg.matrix_bytes(), 0),
        };
        // (what the last load of this model here took beyond its plan is planned as already taken: `calibrated`)
        let extra = calibration(name, levels::name_of(layout), &t.name);
        let mut fit = qwen3_fit(cfg, name, &t.name, t.free.saturating_sub(extra), layout, asked, users, dense_product(t.cc))?;
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

    /// The plan `run` loads MODEL (its config.json `text`) at `level` on the GPU `t`, with the flags of `a`, or the refusal `run` gives: what `fit` prints
    /// and `run`, `serve` and `preflight` size with. `users`: the sequences a server holds at once (the Qwen3 path's; the lossy paths hold one). Ok(None): this
    /// family's path is not sized by its loader yet, and the planner's estimate stands (src/plan.rs).
    pub(crate) fn size(
        a: &Args,
        model: &str,
        text: &str,
        level: &'static str,
        devs: &[Target],
        t: &Target,
        reach: &dyn Fn(&[i32]) -> bool,
        users: usize,
    ) -> Done<Option<LoadPlan>> {
        use crate::support::Family;
        let name = short_name(model);
        match crate::support::row(crate::support::family(text, &name)?).family {
            Family::Qwen3 => {
                supported(&name, text)?;
                let cfg = Config::parse(text, "config.json", None).map_err(cannot)?;
                let (_, _, p) = plan_for(a, &cfg, &name, t, None, users)?;
                shape_checks(&name, &cfg, &p)?;
                Ok(Some(p))
            }
            Family::Qwen35 | Family::Gemma4 | Family::NemotronH => {
                let Some(layout) = levels::layout_of(level) else { return Ok(None) };
                if layout == Layout::Tiered && crate::nemotron::is_nemotron_h(text) {
                    return size_local_dynamic(a, model, text, "Nemotron-H", devs, reach);
                }
                lossy_plan_with(a, model, layout_for(model, layout), text, None, t.free, &t.name, None, t.host).map(Some)
            }
            Family::Kimi => size_local_dynamic(a, model, text, "Kimi-K3", devs, reach),
            Family::Glm => size_local_dynamic(a, model, text, "GLM-5.3-Flash", devs, reach),
            Family::DeepSeekV4 => size_local_dynamic(a, model, text, "DeepSeek-V4", devs, reach),
            Family::Muse => size_local_dynamic(a, model, text, "Muse Spark", devs, reach),
            Family::FlashNext => Ok(None),
        }
    }

    /// `size_dynamic` of a model by its files' headers: a directory's or the Hub cache's, else the Hub's by ranges before anything is downloaded (src/sizes.rs;
    /// ROADMAP P2b); Ok(None) where the headers cannot be had (offline, the Hub does not answer), and the planner's estimate stands.
    pub(crate) fn size_local_dynamic(
        a: &Args,
        model: &str,
        text: &str,
        family: &'static str,
        devs: &[Target],
        reach: &dyn Fn(&[i32]) -> bool,
    ) -> Done<Option<LoadPlan>> {
        let Some(table) = crate::sizes::Table::find(model) else { return Ok(None) };
        // (the files where they are here: a hummingbird directory's experts are not in the table)
        let dir = if Path::new(model).join("config.json").exists() { Some(PathBuf::from(model)) } else { Source::find(model).ok().map(|s| s.dir) };
        let host = devs.first().and_then(|t| t.host);
        size_dynamic(a, &short_name(model), &table, dir.as_deref(), text, family, devs, host, reach).map(Some)
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

    /// The expert cache a Nemotron model in the glyd-v7 directory `dir` (at `layout`) gets on a GPU of `free` bytes free, in a computer with `host` bytes of memory to give: what `run` gives it (`lossy_plan_with` on the
    /// directory's own config and packs); None where the GPU holds the model whole. For `glyd-engine generate --expert-cache auto`.
    pub fn expert_cache_for(dir: &Path, layout: Layout, free: u64, gpu: &str, host: Option<u64>) -> Done<Option<CachePlan>> {
        let file = |f: &str| std::fs::read_to_string(dir.join(f));
        let text = file("config.json").map_err(|e| Refusal::new(format!("{}: {e}", dir.join("config.json").display()), ""))?;
        let (generation, name) = (file("generation_config.json").ok(), dir.to_string_lossy().into_owned());
        let packed = dir.join("glyd.json").exists().then_some(dir);
        Ok(lossy_plan_with(&Args::from(["run".to_string(), name.clone()]), &name, layout, &text, generation.as_deref(), free, gpu, packed, host)?.experts)
    }

    /// Nemotron 3 Super with its experts in the computer's memory (pinned) and a cache of them on the GPU: the context, and the cache, which takes what the GPU has left besides the rest of the model, the states, the
    /// workspaces and the context's keys and values (or `asked` bytes of it); why not, in words, where the GPU cannot hold even that rest and the least cache (a step's choices and as many again; the choices alone for
    /// an asked cache), or the computer cannot hold the experts. At the Hummingbird level a computer that cannot hold them all (or `--host-memory`, `asked_host`, that gives it less) keeps the rest in the packed directory:
    /// `CachePlan::ram` is what it holds, refused where that is less than a step's choices in each width of each layer.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn offloaded(
        cfg: &ConfigNemotronH,
        layout: Layout,
        weights: u64,
        state: u64,
        per: u64,
        free: u64,
        host: Option<u64>,
        ctx: Option<usize>,
        asked: Option<u64>,
        asked_host: Option<u64>,
        gpu: &str,
    ) -> Result<(usize, CachePlan), String> {
        let (layers, experts, k) = (cfg.layers_of(NhKind::Moe), cfg.moe.experts, cfg.moe.top_k);
        let each = crate::moe_nh::expert_bytes(cfg, layout).map_err(|e| e.msg)?;
        let experts_bytes = (layers * experts) as u64 * each;
        let ctx = ctx.unwrap_or(LOSSY_CTX).min(cfg.max_pos.max(LOSSY_MIN_CTX));
        // (the plan's weights are at `bits` a weight, a little over the experts' exact bytes: the rest of the model is taken as what is left)
        let rest = weights.saturating_sub(experts_bytes) + state + LOSSY_WORK + reserve_of(free) + ctx as u64 * per.max(1);
        let least = if asked.is_some() { k } else { 2 * k };
        let need = rest + (layers * least) as u64 * each;
        let budget = asked.unwrap_or_else(|| free.saturating_sub(rest));
        if asked.is_some_and(|b| rest.saturating_add(b) > free) {
            return Err(format!(
                "a cache of {} for the experts and the {} the rest of the model takes are more than the {gpu}'s {} free",
                gib(budget),
                gib(rest),
                gib(free)
            ));
        }
        let host_need = experts_bytes + HOST_MARGIN;
        // (the Hummingbird level's experts can stay in the packed directory: it needs only the memory for what it holds, and a step's choices in each of a layer's two widths)
        let disk = layout == Layout::Hummingbird;
        let ram = held_in_memory(experts_bytes, asked_host, host);
        let least_ram = if disk { (layers * 2 * k) as u64 * each } else { experts_bytes };
        let Some(plan) = crate::moe_nh::cache_plan(cfg, layout, budget, least).map_err(|e| e.msg)?.filter(|p| p.gpu_bytes() + rest <= free) else {
            return Err(format!(
                "with its experts in the computer's memory it needs at least {} of GPU memory (and {} of the computer's)",
                gib(need),
                gib(if disk { least_ram + HOST_MARGIN } else { host_need })
            ));
        };
        if !disk {
            if let Some(h) = host.filter(|&h| h < host_need) {
                return Err(format!("with its experts in the computer's memory it needs {} of the computer's memory besides at least {} of GPU memory, and this computer has {} free", gib(host_need), gib(need), gib(h)));
            }
            return Ok((ctx, plan));
        }
        if let Some(h) = host.filter(|&h| asked_host.is_some() && ram.saturating_add(HOST_MARGIN) > h) {
            return Err(format!("--host-memory {}: the computer has {} free and keeps {} of them for the system", gib(ram), gib(h), gib(HOST_MARGIN)));
        }
        if ram < least_ram {
            return Err(format!("with its experts read from the disk it needs at least {} of the computer's memory for them ({} for a step's choices in each width of each layer), and {} are to be given", gib(least_ram + HOST_MARGIN), gib(least_ram), gib(ram)));
        }
        Ok((ctx, CachePlan { ram, ..plan }))
    }

    /// The model's config as a lossy level reads it (Qwen3.5, Qwen3.6 and Qwen3.8 dense: their config says `qwen3_5`; Nemotron 3 Super: `nemotron_h`, at the kestrel, swift and wren levels), and what it takes in `layout` of
    /// `free` bytes of GPU memory at the context asked for (else up to LOSSY_CTX, as much as fits): refused where it does not fit at LOSSY_MIN_CTX. A pack is sized in its own layout (`pack_layout`: Nemotron's experts
    /// alone are four-bit at wren). `packed`: the glyd-v7 directory the model is, where its packs' bytes are the ones it holds.
    pub(crate) fn lossy_plan(
        a: &Args,
        model: &str,
        layout: Layout,
        text: &str,
        gen: Option<&str>,
        free: u64,
        gpu: &str,
        packed: Option<&Path>,
    ) -> Done<LoadPlan> {
        lossy_plan_with(a, model, layout, text, gen, free, gpu, packed, crate::offload::host_memory())
    }

    /// `lossy_plan` for a computer that has `host` bytes of memory to give (None: it does not say). A Nemotron model that does not fit the GPU whole, or that `--expert-cache GIB` asks for, keeps its experts in the
    /// computer's memory and a cache of them on the GPU (`offloaded`); it is refused, with the memory it needs of each, where the GPU or the computer has too little for that.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn lossy_plan_with(
        a: &Args,
        model: &str,
        layout: Layout,
        text: &str,
        gen: Option<&str>,
        free: u64,
        gpu: &str,
        packed: Option<&Path>,
        host: Option<u64>,
    ) -> Done<LoadPlan> {
        // (what the last load of this model here took beyond its plan is planned as already taken: `calibrated`)
        let extra = calibration(&short_name(model), level_name(layout), gpu);
        let mut p = lossy_plan_inner(a, model, layout, text, gen, free.saturating_sub(extra), gpu, packed, host)?;
        calibrated(&mut p, free, extra);
        Ok(p)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn lossy_plan_inner(
        a: &Args,
        model: &str,
        layout: Layout,
        text: &str,
        gen: Option<&str>,
        free: u64,
        gpu: &str,
        packed: Option<&Path>,
        host: Option<u64>,
    ) -> Done<LoadPlan> {
        let name = short_name(model);
        let tag = level_name(layout);
        if crate::gemma4::is_gemma4(text) {
            return gemma_plan(a, &name, tag, layout, text, gen, free, gpu, packed);
        }
        // the matrices and the tensors kept as they are, what a sequence holds (its states, a token's KV) and the longest context
        let nemotron = crate::nemotron::is_nemotron_h(text);
        let (packs, kept, state, per, max_pos) = if nemotron {
            if !matches!(layout, Layout::Int6 | Layout::Int5 | Layout::Int4 | Layout::Hummingbird) {
                return Err(Refusal::unsupported(format!("{name}: the {tag} level is not built for Nemotron 3 Super, whose experts are held at the kestrel, swift and wren levels, or at the hummingbird level"), format!("Run it at the wren, swift or kestrel level: glyd run {model}:wren")));
            }
            let cfg = ConfigNemotronH::parse(text, "config.json", gen).map_err(|e| Refusal::unsupported(format!("{name}: {}", e.msg), unrun_fix()))?;
            // (the hummingbird kernels take experts on a latent that is a power of two from 128 to 1,024, Nemotron 3 Super's: src/moe_hb.rs; the
            // experts of a model with no latent, Nemotron 3.5 Lightning's, read its width)
            let l = cfg.moe.latent;
            if layout == Layout::Hummingbird && !(l.is_power_of_two() && (128..=1024).contains(&l)) {
                return Err(Refusal::unsupported(
                    format!("{name}: the hummingbird level takes Nemotron-H experts on a latent of 128 to 1,024 (Nemotron 3 Super's); this model's experts read its width of {}", cfg.hidden),
                    format!("Run it at the swift or kestrel level: glyd run {model}:swift"),
                ));
            }
            (cfg.packs(), cfg.kept(), cfg.state_bytes(), cfg.kv_bytes_per_token(), cfg.max_pos)
        } else {
            if layout == Layout::Hummingbird {
                return Err(Refusal::unsupported(
                    format!("{name}: the hummingbird level is built for Nemotron 3 Super alone"),
                    "Run it at the kestrel, swift or wren level: glyd run MODEL:swift",
                ));
            }
            let cfg = Config35::parse(text, "config.json", gen).map_err(|e| {
                Refusal::unsupported(
                    format!("{name}: the {tag} level runs {} models so far ({})", "Qwen3.5, Qwen3.6, Qwen3.8 and Nemotron 3 Super", e.msg),
                    format!("Run it lossless: glyd run {model}"),
                )
            })?;
            let cfg = match packed {
                Some(d) => cfg.as_saved(d),
                None => cfg,
            };
            (cfg.packs(), cfg.kept(), cfg.state_bytes(), cfg.kv_bytes_per_token(), cfg.max_pos)
        };
        let ck = packed.map(|d| glyd_gpu::safetensors::Checkpoint::open(d).map_err(cannot)).transpose()?;
        let (mut weights, mut bf16) = (0u64, 0u64);
        for p in packs {
            let b = (p.rows() * p.cols * 2) as u64;
            bf16 += b;
            // (a matrix the directory holds as bf16 is packed as it loads: sized as without one, in the layout it is packed in)
            let saved = ck.as_ref().map_or(0, |ck| p.parts.iter().filter_map(|x| packed_bytes(ck, Pack35::saved(&x.0))).sum::<u64>());
            let own = if nemotron { pack_layout(layout, &p.key) } else { layout };
            weights += if saved > 0 {
                saved
            } else if p.packable() {
                (b as f64 * bits(own) / 16.0) as u64
            } else {
                b
            };
        }
        for t in kept {
            let b = t.shape.iter().product::<u64>() * if t.dtype == "F32" { 4 } else { 2 };
            bf16 += b;
            weights += b;
        }
        let asked = if a.has("context") || a.has("ctx") { Some(a.either("context", "ctx", 0usize)?) } else { None };
        let fam = if nemotron { crate::support::Family::NemotronH } else { crate::support::Family::Qwen35 };
        crate::support::context_check(&name, crate::support::row(fam), max_pos, asked)?;
        let n = Needs {
            kind: if nemotron { Kind::NemotronH } else { Kind::Qwen35 },
            family: if nemotron { "Nemotron-H" } else { "Qwen3.5" },
            layout,
            weights,
            bf16,
            per_seq: state,
            per_token: per.max(1),
            users: 1,
            fewer: false,
            draft: 0,
            chunks: vec![(2048, LOSSY_WORK)],
            want: LOSSY_CTX,
            least: LOSSY_MIN_CTX,
            cap: max_pos.max(LOSSY_MIN_CTX),
            grain: 1,
            page: 1,
        };
        let fitted = one_gpu(&n, free, asked, gpu);
        let cache = cache_flag(a)?;
        if cache.is_some() && !nemotron {
            return Err(Refusal::new(format!("--expert-cache: {name} has no experts to keep in the computer's memory"), "It is for Nemotron 3 Super"));
        }
        let host_asked = host_flag(a)?;
        if host_asked.is_some() && !nemotron {
            return Err(Refusal::new(format!("--host-memory: {name} has no experts to keep in the computer's memory"), "It is for Nemotron 3 Super"));
        }
        let mut also = String::new();
        if nemotron && (fitted.is_err() || cache.is_some()) {
            let cfg = ConfigNemotronH::parse(text, "config.json", gen).map_err(|e| Refusal::unsupported(format!("{name}: {}", e.msg), unrun_fix()))?;
            match offloaded(&cfg, layout, weights, state, n.per_token, free, host, asked, cache, host_asked, gpu) {
                Ok((ctx, plan)) => return Ok(LoadPlan { experts: Some(plan), ..n.plan_at(ctx, n.chunks[0], free, gpu) }),
                Err(why) if cache.is_some() => {
                    return Err(Refusal::unsupported(
                        format!("{name} at the {tag} level with --expert-cache: {why}"),
                        "Close programs that use the GPU (nvidia-smi lists them), ask for a smaller cache, or use a GPU or a computer with more memory",
                    ))
                }
                Err(why) => also = format!("; {why}"),
            }
        }
        fitted.map_err(|s| {
            let mut r = too_big(&name, tag, gpu, free, 1, &s, asked, false, None);
            if !also.is_empty() {
                r.what += &also;
                r.fix = "Close programs that use the GPU (nvidia-smi lists them) or the computer's memory, or use a GPU or a computer with more memory".into();
            }
            r
        })
    }

    /// `lossy_plan` for a Gemma 4 dense model: its packed weights (as a glyd-v7 directory holds them, else at `layout`), the sliding layers' rings, a draft model where one is found, and the
    /// context the rest of the GPU holds in the global layers' keys and values.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gemma_plan(
        a: &Args,
        name: &str,
        tag: &str,
        layout: Layout,
        text: &str,
        gen: Option<&str>,
        free: u64,
        gpu: &str,
        packed: Option<&Path>,
    ) -> Done<LoadPlan> {
        if !matches!(layout, Layout::Tiered | Layout::Int6 | Layout::Int5 | Layout::Int4 | Layout::Int4H) {
            return Err(Refusal::unsupported(
                format!("{name}: the {tag} level is not built for Gemma 4"),
                format!("Run it lossless (penguin) or at the kestrel or swift level: glyd run {name}:kestrel"),
            ));
        }
        let cfg = crate::gemma4::ConfigG4::parse(text, gen, false).map_err(|e| Refusal::unsupported(format!("{name}: {}", e.msg), ""))?;
        let ck = packed.map(|d| glyd_gpu::safetensors::Checkpoint::open(d).map_err(cannot)).transpose()?;
        let (weights, bf16) = cfg.weight_bytes(layout, ck.as_ref());
        // (a model not on this computer yet comes as its pre-packed repo, which holds its draft model: planned with it, the larger of the two plans)
        let typed = a.pos.get(1).map_or(name, |m| level(m, false).map_or(m.as_str(), |(m, _)| m));
        let draft_to_come = packed.is_none() && !a.has("no-draft") && short_name(typed).to_ascii_lowercase().starts_with("gemma-4-"); // (ship's `draft_repo`: Google's "-assistant" model of each Gemma 4)
        let draft = if gemma_draft(a, packed).is_some() || draft_to_come { GEMMA_DRAFT_BYTES } else { 0 };
        let asked = if a.has("context") || a.has("ctx") { Some(a.either("context", "ctx", 0usize)?) } else { None };
        crate::support::context_check(name, crate::support::row(crate::support::Family::Gemma4), cfg.max_pos, asked)?;
        // (12-bit lossless pages where the run asks for them and has no draft model, whose executor then pages its cache: src/dynamic/kvpages.rs)
        let twelve = std::env::var("GLYD_KV").as_deref() == Ok("lossless") && a.has("no-draft");
        let (rings, per) = cfg.kv_bytes(GEMMA_CHUNK, asked.unwrap_or(LOSSY_CTX).max(LOSSY_MIN_CTX), twelve);
        // (served with no draft model, the scheduler gives each of --users a room of its own, its sliding layers' rings and its context (src/dynamic/rows.rs): the context not asked for is what the room gives
        // each of them, at least LOSSY_MIN_CTX (where that leaves room for fewer, the server takes fewer))
        let users = if draft == 0 && a.pos.first().is_some_and(|c| c == "serve") { users_of(a)? } else { 1 };
        let n = Needs {
            kind: Kind::Gemma4,
            family: "Gemma 4",
            layout,
            weights,
            bf16,
            per_seq: rings,
            per_token: per.max(1),
            users,
            fewer: true,
            draft,
            chunks: vec![(GEMMA_CHUNK, LOSSY_WORK)],
            want: LOSSY_CTX,
            least: LOSSY_MIN_CTX,
            cap: cfg.max_pos.max(LOSSY_MIN_CTX),
            grain: 1,
            page: 16,
        };
        one_gpu(&n, free, asked, gpu).map_err(|s| {
            // (penguin that does not fit, of a model with a wren repo that does: that repo first, with its draft model, at the shortest context a plan takes)
            let wren = gemma_wren(typed).filter(|_| layout == Layout::Tiered).and_then(|repo| {
                let need =
                    cfg.weight_bytes(Layout::Int4H, None).0 + rings + GEMMA_DRAFT_BYTES + LOSSY_WORK + reserve_of(free) + LOSSY_MIN_CTX as u64 * n.per_token;
                (need <= free).then(|| {
                    format!(
                        "glyd run {repo} fits: the wren level, about {} at {LOSSY_MIN_CTX} tokens of context (Google's QAT checkpoint, bit for bit)",
                        gib(need)
                    )
                })
            });
            too_big(name, tag, gpu, free, 1, &s, asked, draft > 0, wren)
        })
    }

    /// Kimi-K3's expert cache over these GPUs (`split::places_plan`): the cuts, their GPUs, each stage's places a layer, the bytes on the GPUs, the experts' bytes in the computer's
    /// memory, the experts a layer has; refused with the numbers where the computer's memory does not hold every expert or a GPU has room for fewer than a token's experts.
    pub(crate) fn kimi_cache(
        table: &crate::sizes::Table,
        text: &str,
        nl: usize,
        room: &[(i32, u64)],
        name: &str,
        ctx: usize,
        twelve: bool,
        host: Option<u64>,
    ) -> Done<(Vec<usize>, Vec<i32>, Vec<usize>, u64, u64, usize, (usize, u64))> {
        let c = crate::kimi::ConfigKimi::parse(text, "config.json", None).map_err(cannot)?;
        // (its bf16 matrices in the 12-bit layout where `twelve`: an HBM GPU's faster product, a few hundred places fewer)
        let bf16 = if twelve { crate::dynamic::split::TWELVE_PCT } else { crate::dynamic::split::TIERED_PCT };
        let (mut dense, base) =
            crate::dynamic::split::layer_bytes_table(table, nl, crate::dynamic::split::CODED_PCT, bf16, &|n| !n.contains(".block_sparse_moe.experts."));
        // (each layer's cache for the context too: a latent attention's latents, ctx of kv_rank + rope bf16; a delta attention's state, heads 128 x 128 fp32, and its convolution's tail)
        let qd = c.kda.heads * c.kda.dim;
        for (l, d) in dense.iter_mut().enumerate() {
            *d += match c.kinds.get(l) {
                Some(crate::glm::GlmKind::Mla) => (ctx * (c.mla.kv_rank + c.mla.rope) * 2) as u64,
                _ => (c.kda.heads * c.kda.dim * c.kda.dim * 4 + 3 * 3 * qd * 2) as u64,
            };
        }
        let expert = crate::dynamic::kimi::expert_bytes(&c).map_err(cannot)?;
        let e = &c.experts;
        let layers = c.moe.iter().filter(|&&m| m).count();
        let all = (layers * e.experts) as u64 * expert;
        let host = host.unwrap_or(0);
        if all + HOST_MARGIN > host {
            return Err(Refusal::unsupported(
                format!(
                    "{name} keeps the {} experts the GPUs do not hold in the computer's memory: they take {}, and {} is free",
                    layers * e.experts,
                    gib(all),
                    gib(host)
                ),
                "A machine with that much memory runs it",
            ));
        }
        // (no room for a token's experts on one GPU: the last layers' dense matrices in the computer's memory, src/dense_ram.rs)
        let ram = crate::dynamic::split::ram_bytes_table(table, text, nl, "Kimi-K3", crate::dynamic::split::CODED_PCT, bf16).map_err(cannot)?;
        let Some((cuts, devices, places, held, freed)) = crate::dynamic::split::places_plan_dense(&dense, base, &c.moe, room, expert, e.experts, e.top_k, &ram)
        else {
            return Err(Refusal::unsupported(
                format!(
                    "{name}: its weights besides the experts take {}, and the GPUs here leave too little room for its experts",
                    gib(dense.iter().sum::<u64>() + base)
                ),
                "A machine whose GPUs have more memory free runs it",
            ));
        };
        let bounds: Vec<usize> = std::iter::once(0).chain(cuts.iter().copied()).chain(std::iter::once(nl)).collect();
        let on_gpus = dense.iter().sum::<u64>() - freed
            + base
            + bounds.windows(2).zip(&places).map(|(w, &p)| c.moe[w[0]..w[1]].iter().filter(|&&m| m).count() as u64 * p as u64 * expert).sum::<u64>();
        Ok((cuts, devices, places, on_gpus, all, e.experts, (held, ram_held(&ram, &dense, held))))
    }

    /// The bytes the last `held` layers' dense matrices take in the computer's memory (`split::places_plan_dense`'s layers).
    pub(crate) fn ram_held(ram: &[u64], dense: &[u64], held: usize) -> u64 {
        let n = ram.len().min(dense.len());
        (n - held.min(n)..n).map(|l| ram[l].min(dense[l])).sum()
    }

    /// A hummingbird directory's expert cache on the dynamic route (src/hbg_store.rs) where its GPUs do not hold all its experts, for any family: each layer's weights besides its experts and its share of the cache for `ctx` tokens
    /// on the GPUs, places for the busiest experts with the room left (`split::places_plan`, at the mean of the experts' bytes), and every expert in the computer's memory (`host` bytes free, less `HOST_MARGIN`) or, where it holds
    /// less, its share there and the rest read from the expert files as the router asks for them (the disk). The cuts, their GPUs, each stage's places a layer, the GPUs' bytes, all the experts' bytes, the experts a layer, and
    /// the bytes of experts the computer's memory holds; or the refusal.
    #[allow(clippy::type_complexity)]
    pub(crate) fn hb_cache(
        dir: &Path,
        text: &str,
        nl: usize,
        room: &[(i32, u64)],
        name: &str,
        ctx: usize,
        twelve: bool,
        host: Option<u64>,
    ) -> Done<(Vec<usize>, Vec<i32>, Vec<usize>, u64, u64, usize, u64, (usize, u64))> {
        let hb = crate::hb_pack::HbDir::open(dir).map_err(cannot)?.ok_or_else(|| cannot(format!("{}: not a hummingbird directory", dir.display())))?;
        let bf16 = if twelve { crate::dynamic::split::TWELVE_PCT } else { crate::dynamic::split::TIERED_PCT };
        let (mut dense, base) = crate::dynamic::split::layer_bytes_with(dir, nl, crate::dynamic::split::CODED_PCT, bf16, &|_| true, false).map_err(cannot)?;
        // (each layer's share of the cache for the context: the model's KV a token and its recurrent state, over its layers)
        let sh = crate::plan::Shape::from_config(text).map_err(cannot)?;
        let kv = (sh.kv_full * ctx as f64 + sh.kv_sliding * ctx.min(sh.window.max(1)) as f64 + sh.state) / nl.max(1) as f64;
        dense.iter_mut().for_each(|d| *d += kv as u64);
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
        let mean = all / count.max(1) as u64;
        let host = host.unwrap_or(0);
        // (the computer's memory for the experts: all of them, else what it has less the margin; at least a token's experts in every layer)
        let ram = all.min(host.saturating_sub(HOST_MARGIN));
        let token = hb.layers.len() as u64 * sh.topk.max(1) as u64 * mean;
        if ram < token {
            return Err(Refusal::unsupported(
                format!(
                    "{name} at the hummingbird level keeps its experts in the computer's memory and on the disk: they take {}, and {} of memory is free",
                    gib(all),
                    gib(host)
                ),
                "A machine with more memory runs it",
            ));
        }
        // (no room for a token's experts on one GPU: the last layers' dense matrices in the computer's memory, src/dense_ram.rs)
        let family = crate::dynamic::split::family(text).unwrap_or("");
        let rb = crate::dynamic::split::ram_bytes(dir, nl, family, crate::dynamic::split::CODED_PCT, bf16).map_err(cannot)?;
        let Some((cuts, devices, places, held, freed)) = crate::dynamic::split::places_plan_dense(&dense, base, &moe, room, mean, experts, sh.topk.max(1), &rb)
        else {
            return Err(Refusal::unsupported(
                format!(
                    "{name}: its weights besides the experts take {}, and the GPUs here leave too little room for its experts",
                    gib(dense.iter().sum::<u64>() + base)
                ),
                "A machine whose GPUs have more memory free runs it",
            ));
        };
        let bounds: Vec<usize> = std::iter::once(0).chain(cuts.iter().copied()).chain(std::iter::once(nl)).collect();
        let on_gpus = dense.iter().sum::<u64>() - freed
            + base
            + bounds.windows(2).zip(&places).map(|(w, &p)| moe[w[0]..w[1]].iter().filter(|&&m| m).count() as u64 * p as u64 * mean).sum::<u64>();
        Ok((cuts, devices, places, on_gpus, all, experts, ram, (held, ram_held(&rb, &dense, held))))
    }

    /// A release's expert cache on the dynamic route where its experts are each the same bytes (src/tiering.rs, R12: DeepSeek-V4's FP4 experts; GLM-5.3-Flash's FP8 ones, each packed on its own in places of
    /// the largest's bytes: about 7 bits a value where the release's bytes, which size the places here, are 8; Muse Spark's bf16 ones in the lossless tiered layout, about 11 bits of 16) and its GPUs do not hold them all: each layer's weights besides its experts and
    /// its share of the cache for `ctx` tokens on the GPUs, places for the busiest experts with the room left, and every expert in the computer's memory (`host` bytes free, less `HOST_MARGIN`: the disk holds hummingbird
    /// directories' experts alone). The cuts, their GPUs, each stage's places a layer, the GPUs' bytes, all the experts' bytes and the experts a layer; or the refusal.
    #[allow(clippy::type_complexity)]
    pub(crate) fn release_cache(
        table: &crate::sizes::Table,
        text: &str,
        nl: usize,
        room: &[(i32, u64)],
        name: &str,
        ctx: usize,
        twelve: bool,
        host: Option<u64>,
    ) -> Done<(Vec<usize>, Vec<i32>, Vec<usize>, u64, u64, usize, (usize, u64))> {
        let expert = |n: &str| n.contains(".experts.");
        let bf16 = if twelve { crate::dynamic::split::TWELVE_PCT } else { crate::dynamic::split::TIERED_PCT };
        let (mut dense, base) = crate::dynamic::split::layer_bytes_table(table, nl, crate::dynamic::split::CODED_PCT, bf16, &|n| !expert(n));
        let (per, _) = crate::dynamic::split::layer_bytes_table(table, nl, 100, 100, &expert);
        let sh = crate::plan::Shape::from_config(text).map_err(cannot)?;
        let kv = (sh.kv_full * ctx as f64 + sh.kv_sliding * ctx.min(sh.window.max(1)) as f64 + sh.state) / nl.max(1) as f64;
        dense.iter_mut().for_each(|d| *d += kv as u64);
        let moe: Vec<bool> = per.iter().map(|&b| b > 0).collect();
        let (all, layers, experts) = (per.iter().sum::<u64>(), moe.iter().filter(|&&m| m).count(), sh.experts.max(1));
        let mean = all / (layers * experts).max(1) as u64;
        let host = host.unwrap_or(0);
        if all + HOST_MARGIN > host {
            return Err(Refusal::unsupported(
                format!("{name} keeps the experts its GPUs do not hold in the computer's memory: they take {}, and {} is free", gib(all), gib(host)),
                "A machine with that much memory runs it, or its hummingbird level, whose experts the disk holds too",
            ));
        }
        // (no room for a token's experts on one GPU: the last layers' dense matrices in the computer's memory, src/dense_ram.rs)
        let family = crate::dynamic::split::family(text).unwrap_or("");
        let rb = crate::dynamic::split::ram_bytes_table(table, text, nl, family, crate::dynamic::split::CODED_PCT, bf16).map_err(cannot)?;
        let Some((cuts, devices, places, held, freed)) = crate::dynamic::split::places_plan_dense(&dense, base, &moe, room, mean, experts, sh.topk.max(1), &rb)
        else {
            return Err(Refusal::unsupported(
                format!(
                    "{name}: its weights besides the experts take {}, and the GPUs here leave too little room for its experts",
                    gib(dense.iter().sum::<u64>() + base)
                ),
                "A machine whose GPUs have more memory free runs it",
            ));
        };
        let bounds: Vec<usize> = std::iter::once(0).chain(cuts.iter().copied()).chain(std::iter::once(nl)).collect();
        let on_gpus = dense.iter().sum::<u64>() - freed
            + base
            + bounds.windows(2).zip(&places).map(|(w, &p)| moe[w[0]..w[1]].iter().filter(|&&m| m).count() as u64 * p as u64 * mean).sum::<u64>();
        Ok((cuts, devices, places, on_gpus, all, experts, (held, ram_held(&rb, &dense, held))))
    }

    /// `kimi_cache` with every layer across all the GPUs of `room` (src/dynamic/tp.rs): each GPU holds its slice of the rows of each big matrix (the embedding
    /// whole, and every matrix too small or of rows that do not divide), the caches for the context whole, and the same places a layer, each a slice of an
    /// expert's; the least room of the GPUs sets the places.
    pub(crate) fn kimi_cache_tp(
        table: &crate::sizes::Table,
        text: &str,
        room: &[(i32, u64)],
        name: &str,
        ctx: usize,
        twelve: bool,
        host: Option<u64>,
    ) -> Done<(Vec<usize>, Vec<i32>, Vec<usize>, u64, u64, usize)> {
        let c = crate::kimi::ConfigKimi::parse(text, "config.json", None).map_err(cannot)?;
        let n = room.len().max(1) as u64;
        let bf16 = if twelve { crate::dynamic::split::TWELVE_PCT } else { crate::dynamic::split::TIERED_PCT };
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
        let qd = c.kda.heads * c.kda.dim;
        held += c
            .kinds
            .iter()
            .map(|k| match k {
                crate::glm::GlmKind::Mla => (ctx * (c.mla.kv_rank + c.mla.rope) * 2) as u64,
                _ => (c.kda.heads * c.kda.dim * c.kda.dim * 4 + 3 * 3 * qd * 2) as u64,
            })
            .sum::<u64>();
        let e = &c.experts;
        let layers = c.moe.iter().filter(|&&m| m).count() as u64;
        let expert = crate::dynamic::kimi::expert_bytes(&c).map_err(cannot)?;
        let all = layers * e.experts as u64 * expert;
        let host = host.unwrap_or(0);
        if all + HOST_MARGIN > host {
            return Err(Refusal::unsupported(
                format!(
                    "{name} keeps the {} experts the GPUs do not hold in the computer's memory: they take {}, and {} is free",
                    layers as usize * e.experts,
                    gib(all),
                    gib(host)
                ),
                "A machine with that much memory runs it",
            ));
        }
        let least = room.iter().map(|r| r.1).min().unwrap_or(0);
        let places = ((least.saturating_sub(held)) / (layers.max(1) * (expert / n)).max(1)).min(e.experts as u64) as usize;
        if places < e.top_k {
            return Err(Refusal::unsupported(
                format!("{name}: its weights besides the experts take {} a GPU across {n} GPUs, and they leave too little room for its experts", gib(held)),
                "A machine whose GPUs have more memory free runs it",
            ));
        }
        let on_gpus = n * (held + layers * places as u64 * (expert / n));
        Ok((Vec::new(), room.iter().map(|r| r.0).collect(), vec![places; room.len()], on_gpus, all, e.experts))
    }

    /// Each GPU's room for weights on the dynamic route: its free memory less the reserve and the route's work (the KV, the scratch and the graphs together;
    // ponytail: that work is a fixed margin, 3 GiB and 5% with the reserve; the description's own numbers where it proves too little or too much)
    pub(crate) fn rooms(gpus: &[Gpu]) -> Vec<(i32, u64)> {
        gpus.iter().map(|g| (g.ordinal, crate::plan::dynamic_room_of(g.free))).collect()
    }

    /// The table's name of a dynamic family (`dynamic_family`'s words are these).
    pub(crate) fn static_family(family: &str) -> &'static str {
        ["Kimi-K3", "GLM-5.3-Flash", "DeepSeek-V4", "Muse Spark", "Nemotron-H"].into_iter().find(|f| *f == family).unwrap_or("DeepSeek-V4")
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
        text: &str,
        family: &'static str,
        devs: &[Target],
        host: Option<u64>,
        reach: &dyn Fn(&[i32]) -> bool,
    ) -> Done<LoadPlan> {
        let nl = crate::dynamic::split::config_num(text, "num_hidden_layers").unwrap_or(0);
        let room: Vec<(i32, u64)> = devs.iter().map(|t| (t.ordinal, crate::plan::dynamic_room_of(t.free))).collect();
        let hbm = hbm_of(devs);
        let kimi = family == "Kimi-K3";
        // (a hummingbird directory, src/hb_pack.rs: its routed experts are files of their own, which no table of tensors lists)
        let hb_dir = dir.filter(|d| matches!(crate::hb_pack::HbDir::open(d), Ok(Some(_))));
        let hb = hb_dir.is_some();
        let hb_experts = match hb_dir {
            Some(d) => crate::dynamic::split::hb_layer_bytes(d, nl).map_err(cannot)?,
            None => Vec::new(),
        };
        let ((per, base), cut, coded, mut twelve) = crate::dynamic::split::fit_table_with(table, nl, &room, hbm, kimi, &hb_experts);
        let mut total = per.iter().sum::<u64>() + base;
        let chosen = a.flag("split").is_some() || a.flag("devices").is_some();
        let model_max = crate::dynamic::split::config_num(text, "max_position_embeddings").unwrap_or(usize::MAX);
        let row = crate::support::row_named(family).expect("a dynamic family has a row");
        let ctx = a.num("context", crate::support::default_context(row, model_max)).map_err(|e| Refusal::new(e.msg, "--context 8192"))?;
        crate::support::context_check(&name, row, model_max, a.flag("context").map(|_| ctx))?;
        // (every layer across the GPUs, src/dynamic/tp.rs: by default on datacenter GPUs that reach each other directly, where no split is asked for; a
        // development build's GLYD_TP=1 or 0 says so itself, GLYD_TP_GPUS which GPUs)
        let tp_room: Vec<(i32, u64)> = match crate::tune!("GLYD_TP_GPUS") {
            Some(v) => room.iter().copied().filter(|r| v.split(',').any(|x| x.trim().parse::<i32>().ok() == Some(r.0))).collect(),
            None => room.clone(),
        };
        let tp_devs: Vec<i32> = tp_room.iter().map(|r| r.0).collect();
        let across = matches!(family, "Kimi-K3" | "DeepSeek-V4" | "GLM-5.3-Flash" | "Muse Spark")
            && tp_room.len() > 1
            && !chosen
            && match crate::tune!("GLYD_TP").as_deref() {
                Some("1") => true,
                Some(_) => false,
                None => matches!(family, "Kimi-K3" | "DeepSeek-V4") && hbm && reach(&tp_devs),
            };
        // Kimi-K3 where the GPUs do not hold its experts: the expert cache (each GPU's room after the other weights as places for its layers' experts, every
        // expert in the computer's memory)
        // (a hummingbird directory's own expert cache for any family, its layers cut over the GPUs, never every layer across them)
        // (a release whose experts the cache takes, src/tiering.rs: DeepSeek-V4's FP4 experts, GLM-5.3-Flash's FP8 ones, Muse Spark's lossless ones; its cache where its GPUs do not hold it)
        let release_cached = matches!(family, "DeepSeek-V4" | "GLM-5.3-Flash" | "Muse Spark") && !hb && cut.is_none() && !chosen;
        let across = across && !hb && !release_cached;
        let mut ram = None;
        // (the last layers' dense matrices in the computer's memory where a token's experts do not fit beside them on one GPU: how many, their bytes)
        let mut dense_ram = (0usize, 0u64);
        let cache = if let Some(dir) = hb_dir.filter(|_| cut.is_none() && !chosen) {
            let k = hb_cache(dir, text, nl, &room, name, ctx, hbm, host)?;
            (ram, dense_ram) = (Some(k.6), k.7);
            Some((k.0, k.1, k.2, k.3, k.4, k.5))
        } else if release_cached {
            // (R12: its experts in the cache, every one in the computer's memory)
            let k = release_cache(table, text, nl, &room, name, ctx, hbm, host)?;
            (ram, dense_ram) = (Some(k.4), k.6);
            Some((k.0, k.1, k.2, k.3, k.4, k.5))
        } else if kimi && !hb && cut.is_none() && !chosen {
            Some(if across {
                kimi_cache_tp(table, text, &tp_room, name, ctx, hbm, host)?
            } else {
                let k = kimi_cache(table, text, nl, &room, name, ctx, hbm, host)?;
                dense_ram = k.6;
                (k.0, k.1, k.2, k.3, k.4, k.5)
            })
        } else {
            None
        };
        if let Some(k) = &cache {
            total = k.3;
            twelve = hbm;
        }
        let (cuts, devices) = if let Some(k) = &cache {
            (k.0.clone(), k.1.clone())
        } else if across {
            // (each GPU its slice of the big matrices and the rest whole: about its share of the bytes and the embedding's and head's)
            // ponytail: the share is an estimate; a GPU that runs out of memory says so at the load
            (Vec::new(), tp_devs.clone())
        } else if chosen {
            crate::dynamic::split::split_args(a)
                .map_err(|e| Refusal::new(e.msg, "--split 31 --devices 0,1: the layers before 31 on GPU 0, the rest on GPU 1"))?
        } else {
            cut.ok_or_else(|| {
                Refusal::unsupported(
                    format!(
                        "{name} needs about {} of GPU memory at penguin, and the {} GPU{} here have {} free",
                        gib(total),
                        devs.len(),
                        if devs.len() == 1 { "" } else { "s" },
                        gib(devs.iter().map(|t| t.free).sum::<u64>())
                    ),
                    "A machine whose GPUs have that much memory free runs it (nvidia-smi lists what uses them)",
                )
            })?
        };
        let used: Vec<&Target> = devs.iter().filter(|t| devices.contains(&t.ordinal)).collect();
        let (free, work) = (used.iter().map(|t| t.free).sum::<u64>(), used.iter().map(|t| crate::plan::dynamic_work_of(t.free)).sum::<u64>());
        let host_bytes = cache.as_ref().map_or(0, |k| k.4);
        // (a hummingbird directory's cache: its places, the experts' bytes at their mean, the memory and the disk they take)
        let experts = cache.as_ref().zip(ram).map(|(k, ram)| {
            // (its layers of experts: a hummingbird directory's own; a release's, the layers the table lists experts of)
            let layers = match hb_dir.and_then(|d| crate::hb_pack::HbDir::open(d).ok().flatten()) {
                Some(d) => d.layers.len(),
                None => crate::dynamic::split::layer_bytes_table(table, nl, 100, 100, &|n| n.contains(".experts.")).0.iter().filter(|&&b| b > 0).count(),
            };
            crate::offload::CachePlan {
                layers,
                experts: k.5,
                slots: k.2.iter().min().copied().unwrap_or(0),
                expert_bytes: k.4 / (layers * k.5).max(1) as u64,
                in_place: true,
                ram,
            }
        });
        Ok(LoadPlan {
            kind: Kind::Dynamic,
            family,
            level: if hb { "hummingbird" } else { "penguin" },
            layout: if twelve { Layout::Twelve } else { Layout::Tiered },
            context: ctx,
            users: 1,
            chunk: 256,
            weights: total + host_bytes + dense_ram.1,
            bf16: 0,
            kv: 0,
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
                places: cache.as_ref().map_or(Vec::new(), |k| k.2.clone()),
                experts_per_layer: cache.as_ref().map_or(0, |k| k.5),
                host_bytes,
                gpu_bytes: total,
                dense: dense_ram.0,
                dense_bytes: dense_ram.1,
            }),
        })
    }

    #[cfg(test)]
    mod fit_tests {
        use super::*;

        fn qwen3_8b() -> Config {
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
            let m = 16_380_000_000u64; // Qwen3-8B's matrices in bf16
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
        fn qwen3_8b_on_the_gpus() {
            let c = qwen3_8b();
            // weights near the measured 10.2 GiB of the engine's record
            let w = weights(&c, Layout::Tiered);
            assert!((10 << 30..11_500_000_000u64).contains(&w), "{w}");
            // an L4 (22.5 GiB free): the want of 8,192 tokens, the 4,096 chunk with cuBLAS
            let l4 = qwen3_fit(&c, "Qwen3-8B", "NVIDIA L4", 22 << 30, Layout::Tiered, None, 1, true).unwrap();
            assert_eq!((l4.context, l4.chunk), (8192, 4096), "{l4:?}");
            assert!(l4.gpu_total() < 22 << 30);
            // the whole 40,960-token window fits an L4 too: the lossless cache is small
            assert_eq!(qwen3_fit(&c, "Qwen3-8B", "NVIDIA L4", 22 << 30, Layout::Tiered, Some(40960), 1, true).unwrap().context, 40960);
            // asking for more than fits is refused, with what fits
            let r = qwen3_fit(&c, "Qwen3-8B", "NVIDIA L4", 13 << 30, Layout::Tiered, Some(40960), 1, true).unwrap_err();
            assert!(r.fix.starts_with("Ask for less: --context ") && r.code == crate::ui::CANNOT, "{r}");
            // a smaller GPU: less context than the want, or a smaller chunk
            let small = qwen3_fit(&c, "Qwen3-8B", "NVIDIA L4", 13 << 30, Layout::Tiered, None, 1, true).unwrap();
            assert!(small.context >= CTX_MIN, "{small:?}");
            assert!(small.gpu_total() < 13 << 30, "{small:?}");
            // too small altogether: a refusal that hands the model to vLLM
            assert_eq!(qwen3_fit(&c, "Qwen3-8B", "NVIDIA L4", 8 << 30, Layout::Tiered, None, 1, true).unwrap_err().code, crate::ui::CANNOT);
            // more than the model's own context
            assert!(qwen3_fit(&Config { max_pos: 2048, ..qwen3_8b() }, "tiny", "NVIDIA L4", 22 << 30, Layout::Tiered, Some(4096), 1, true).is_err());
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
            let r = lossy_plan_with(&a("131072"), "Qwen/Qwen3.8-27B", Layout::Int4, &text, None, free, "NVIDIA GeForce RTX 4090", None, None).unwrap_err();
            let most = r.fix.strip_prefix("Ask for less: --context ").unwrap_or_else(|| panic!("{r}")).to_string();
            assert!(r.what.contains("at 131072 tokens of context"), "{r}");
            let p = lossy_plan_with(&a(&most), "Qwen/Qwen3.8-27B", Layout::Int4, &text, None, free, "NVIDIA GeForce RTX 4090", None, None).unwrap();
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

    fn fixture(name: &str) -> String {
        std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)).unwrap()
    }

    fn qwen35(name: &str) -> Shape {
        let all = json::parse(&fixture("qwen35_configs.json")).unwrap();
        let c = all.get(name).unwrap();
        Shape::from_config(&json::to_compact(c)).unwrap()
    }

    /// Qwen3-32B's config (Qwen3ForCausalLM), as the Hub has it.
    const QWEN3_32B: &str = r#"{"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 5120, "intermediate_size": 25600, "num_hidden_layers": 64, "num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 151936, "tie_word_embeddings": false, "max_position_embeddings": 40960}"#;

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
            let (b, e) = level_bits("hummingbird", s, dev).unwrap();
            assert_eq!(b, rest);
            assert!((e - hummingbird_expert_bits(s.expert_width)).abs() < 1e-12);
            assert_eq!(level_bits("penguin", s, dev).unwrap().1, s.expert_bits_released().min(12.04));
        }
        // the uniform levels are not offered where they would raise the experts' bits (MXFP4 at 4.25 against kestrel's 6.5)
        assert!(level_bits("kestrel", &k, dev).is_none() && level_bits("wren", &d, dev).is_none());
        // experts released at hummingbird's bits or fewer: nothing to gain
        let low = Shape { expert_release_bits: 2.5, ..d.clone() };
        assert!(level_bits("hummingbird", &low, dev).is_none());
        // Nemotron-H: the kernels' latent (Nemotron 3 Super's 1,024) takes it, Nemotron 3.5 Lightning's width of 2,688 does not
        let nh = Shape { arch: "NemotronHForCausalLM".into(), expert_width: 1024, release_bits: 16.0, expert_release_bits: 0.0, ..d.clone() };
        assert_eq!(level_bits("hummingbird", &nh, dev), Some((6.0, 2.9)));
        assert!(level_bits("hummingbird", &Shape { expert_width: 2688, ..nh }, dev).is_none());
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
        let c = |layers: &str| QWEN3_32B.replace("\"num_hidden_layers\": 64", &format!("\"num_hidden_layers\": {layers}"));
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
        let q3 = r#"{"architectures":["Qwen3ForCausalLM"],"hidden_size":1024,"num_hidden_layers":1000000000,"num_attention_heads":16,"num_key_value_heads":8,"intermediate_size":3072,"vocab_size":151936,"rms_norm_eps":1e-06,"rope_theta":1000000.0,"max_position_embeddings":40960}"#;
        assert!(crate::weights::Config::parse(q3, "c", None).unwrap_err().to_string().contains("num_hidden_layers"));
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
        within(Shape::from_config(QWEN3_32B).unwrap().params(), 32.8e9, 0.04);
        within(qwen35("Qwen/Qwen3.5-9B").params(), 9.0e9, 0.12);
        within(qwen35("Qwen/Qwen3.8-27B").params(), 26.9e9, 0.12);
        let g = Shape::from_config(&fixture("gemma4_12b_config.json")).unwrap();
        within(g.params(), 12.0e9, 0.15);
        assert!(g.kv_sliding > 0.0 && g.window == 1024 && g.kv_full > 0.0);
        let nh = Shape::from_config(&fixture("nemotron_h_config.json")).unwrap();
        within(nh.params(), 120.0e9, 0.12);
        within(nh.params() - nh.routed + nh.routed_active(), 12.0e9, 0.25);
        let qx = Shape::from_config(&fixture("qwen4x_config.json")).unwrap();
        assert!(qx.is_moe() && qx.experts == 512 && qx.topk == 10 && qx.mtp && qx.state > 0.0);
        // scaled to an index's total, every part keeps its share
        let s = Shape::from_config(QWEN3_32B).unwrap();
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
        // Qwen3-32B lossless on an RTX A6000: 11.8 (one sequence)
        within(&plan(&Shape::from_config(QWEN3_32B).unwrap(), &Machine::preset("a6000").unwrap(), Ask::default()), 11.8, 0.25);
    }

    /// P2c: decode and prefill predicted from the rates of an RTX 4080 SUPER (the preset's stream is the one `glyd-engine rates` measured there, 697 GB/s) against
    /// what `glyd-engine bench --loads 1k:1 --no-draft` measured on one (2026-11-23, three runs each, the median): gemma-4-12B at wren 86.2 tokens/s and kestrel 63.3,
    /// prompts of 936 tokens at 3,570 and 3,580; Qwen3.5-0.8B at kestrel 561.9 and wren 657.9, prompts of 1,002 tokens at 27,073 and 27,372. Each within 15%, with
    /// the shapes `fit` reads (the checkpoints' own headers) and the family's path (Qwen3.5's does not draft with its head); the measurement in the band.
    #[test]
    fn predictions_on_the_4080_super_are_within_15_percent() {
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
            assert!((c.tok_s / decode - 1.0).abs() < 0.15, "{level}: {:.1} tokens/s predicted, {decode} measured", c.tok_s);
            assert!(c.range.0 <= decode && decode <= c.range.1, "{level}: {decode} outside {:?}", c.range);
            let p = c.prefill.unwrap();
            assert!((p / prefill - 1.0).abs() < 0.15, "{level}: prefill {p:.0} predicted, {prefill} measured");
        }
        // (a GPU whose tensor rate the planner does not know has no prefill prediction)
        let mut odd = m.clone();
        odd.devices[0].tflops = 0.0;
        assert!(chosen(&plan(&gemma_sh, &odd, Ask { level: Some("wren"), ..Default::default() })).prefill.is_none());
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
        // Qwen3-32B lossless on one 48 GB card (README: bf16 needs two); two 24 GB cards hold it at kestrel, split; one 24 GB card at a lower level
        let q32 = Shape::from_config(QWEN3_32B).unwrap();
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
            vec![("Qwen/Qwen3-32B".to_string(), QWEN3_32B.to_string()), ("Qwen/Qwen3.5-9B".to_string(), json::to_compact(&q35.as_object().unwrap()[0].1))];
        for (f, m) in [
            ("gemma4_12b_config.json", "google/gemma-4-12B-it"),
            ("nemotron_h_config.json", "nvidia/Nemotron-H"),
            ("glm5_next_config.json", "zai-org/GLM-5.3-Flash"),
        ] {
            configs.push((m.to_string(), fixture(f)));
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

    const QWEN3_8B: &str = r#"{"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 4096, "intermediate_size": 12288, "num_hidden_layers": 36, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 151936, "tie_word_embeddings": false, "max_position_embeddings": 40960, "rms_norm_eps": 1e-6, "rope_theta": 1000000}"#;

    fn gpu_of(d: &Device, ordinal: i32) -> crate::dev::Gpu {
        crate::dev::Gpu { name: d.name.clone(), cc: d.cc, sms: 0, total: d.total, free: d.free, driver: d.driver, count: 1, ordinal }
    }

    /// What `fit` prints for each family at each level it runs at, on each NVIDIA preset, is the plan `run` builds from the same GPU (the functions
    /// `run` calls: `sizing::qwen3_plan` for Qwen3, `sizing::lossy_plan_with` for the lossy families), or the refusal `run` gives; and what it plans fits the GPU.
    #[test]
    fn fit_is_the_plan_run_builds() {
        let q35 = json::parse(&fixture("qwen35_configs.json")).unwrap();
        let configs = [
            ("Qwen3-32B", with_rope(QWEN3_32B)),
            ("Qwen3.5-9B", json::to_compact(&q35.as_object().unwrap()[0].1)),
            ("gemma-4-12B", fixture("gemma4_12b_config.json")),
            ("NVIDIA-Nemotron-H", fixture("nemotron_h_config.json")),
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
                            sizing::qwen3_plan(&args(&["run", model]), &cfg, &crate::cli::short_name(model), &gpu, None, 1).map(|p| p.2)
                            // (on Windows the name of a path is the whole path)
                        }
                        _ => {
                            let layout = crate::levels::layout_for(model, crate::levels::layout_of(level).unwrap());
                            if layout == crate::weights::Layout::Tiered && crate::nemotron::is_nemotron_h(text) {
                                continue; // (the dynamic route's: not sized by `size` yet)
                            }
                            sizing::lossy_plan_with(&args(&["run", model]), model, layout, text, None, d.free, &d.name, None, Some(m.ram_free))
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
        let dir = model_dir("users", "Qwen3-8B", QWEN3_8B);
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
        let (q3, q35) = (root.join("qwen3"), root.join("qwen35"));
        crate::logits::qwen3_random(&q3).unwrap();
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
                sizing::lossy_plan_with(&a, q35.to_str().unwrap(), layout, &text, None, 24 << 30, "NVIDIA RTX 4090", packed, None).unwrap()
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
                "F32" => 4,
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
                &fixture("glm5_next_config.json"),
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
        // (no run observed: the prediction itself in the calibration's band)
        assert_eq!(Candidate::speeds(100.0, &m.devices[0], None).range, (100.0 * SPREAD[0], 100.0 * SPREAD[2]));
        // the model's own runs, and the plan kept anew keeps them
        assert_eq!(observed_for(&root, &model, lp.level, &lp.gpu), band);
        remember_in(&root, &model, &lp, lp.weights, 1 << 30);
        assert_eq!(observed_for(&root, &model, lp.level, &lp.gpu), band);
        // a run with a draft model is its own model's, not the GPU's
        let drafted = LoadPlan { level: "wren", draft: 640 << 20, ..lp.clone() };
        remember_in(&root, &model, &drafted, lp.weights, 0);
        let at_wren = speed_at(&f.shape, &m, 0, &drafted, 1064.5);
        keep(&watch_of(&root, &model, &drafted, f.shape.clone(), m.clone()).unwrap(), 1000, 0.25, 129, 128.0 / (at_wren * 1.6));
        assert_eq!(observed_on(&root, &lp.gpu).0, band);
        assert!(observed_for(&root, &model, "wren", &lp.gpu).is_some_and(|b| b[1] > 1.5));
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
        sizing::size(&a, model, text, crate::levels::LEVELS.iter().map(|l| l.name).find(|n| *n == level).unwrap(), &devs, &target, &|_| linked, 1)
    }

    /// P16: no preset by fixture model by level where `fit` and `run` disagree: for every one, `fit` refuses exactly where `run` does, in `run`'s words and with its
    /// status, and where `run` loads a plan `fit` prints that plan. The models are the families' fixtures, one a Llama (a kind the engine does not run), and
    /// Qwen3.8-Flash-Next (Condor's); every machine preset, AMD, Apple, Arm and the multi-GPU ones among them; every level this build has, those a family does not run at too.
    #[test]
    fn fit_and_run_agree_on_every_preset_and_model() {
        let q35 = json::parse(&fixture("qwen35_configs.json")).unwrap();
        let llama = r#"{"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8}"#;
        let glm = fixture("glm5_next_config.json");
        let configs = [
            ("Qwen3-32B", with_rope(QWEN3_32B)),
            ("Qwen3.5-9B", json::to_compact(&q35.as_object().unwrap()[0].1)),
            ("gemma-4-12B", fixture("gemma4_12b_config.json")),
            ("NVIDIA-Nemotron-H", fixture("nemotron_h_config.json")),
            ("GLM-5.3-Flash", glm.clone()),
            ("Qwen3.8-Flash-Next", fixture("qwen4x_config.json")),
            ("Llama-3.1-8B", llama.to_string()),
        ];
        let (mut pairs, mut loads, mut refusals) = (0, 0, 0);
        for (name, text) in &configs {
            let dir = model_dir("agree", name, text);
            // (the dynamic route sizes a model from its files' headers (src/sizes.rs, P2b): a directory with a config alone has none, so these two get a header-only checkpoint)
            if *name == "GLM-5.3-Flash" {
                header_only_checkpoint(&dir, &fixture("glm5_next_tensors.json"));
            }
            if *name == "NVIDIA-Nemotron-H" {
                header_only_checkpoint(&dir, &fixture("nemotron_h_tensors.json"));
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
        let r = crate::support::context_check("Qwen3-8B", row, 40960, Some(100_000)).unwrap_err();
        assert_eq!(
            (r.what.as_str(), r.fix.as_str()),
            ("Qwen3-8B takes at most 40960 tokens of context here, and 100000 were asked for", "Ask for less: --context 40960")
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
        let dir = model_dir("ctx", "Qwen3-8B", QWEN3_8B);
        let f = planned(&args(&["fit", dir.to_str().unwrap(), "--machine", "rtx4090", "--context", "100000"])).err().unwrap();
        let a = args(&["run", "Qwen3-8B", "--context", "100000"]);
        let cfg = crate::weights::Config::parse(QWEN3_8B, "config.json", None).unwrap();
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
    /// fixtures hold (GLM-5.3-Flash, Qwen3.5 at 0.8B and 9B, Nemotron 3 Super) within 1%.
    #[test]
    fn a_shape_from_the_headers_is_the_checkpoints() {
        use crate::sizes::{classify, Part};
        let glm = json::parse(&fixture("glm5_next_tensors.json")).unwrap();
        let q35 = json::parse(&fixture("qwen35_tensors.json")).unwrap();
        let q35c = json::parse(&fixture("qwen35_configs.json")).unwrap();
        let nh = json::parse(&fixture("nemotron_h_tensors.json")).unwrap();
        let mut cases = vec![
            ("GLM-5.3-Flash", fixture("glm5_next_config.json"), table_of(glm.get("tensors").unwrap())),
            ("Nemotron 3 Super", fixture("nemotron_h_config.json"), table_of(nh.get("tensors").unwrap())),
        ];
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
                sizing::size_dynamic(&args(&["run", "GLM-5.3-Flash"]), "GLM-5.3-Flash", t, None, &config, "GLM-5.3-Flash", &devs, Some(m.ram_free), &|_| false)
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
            gpu_bytes: base.weights,
            dense,
            dense_bytes,
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

    /// P2f: every plan the per-path planners made (P2a's: src/cli.rs `size`, `lossy_plan_with`, `gemma_plan`, `size_dynamic` and src/fit.rs `plan`, recorded in
    /// tests/fixtures/plans_p2a.json before they were moved here) is the plan the one planner makes: each family's fixture, on every preset, at every level this
    /// build has, alone and with --users 4, --context 8192 and 131072, and Nemotron-H's --expert-cache and --host-memory; the plan `run` loads, or its refusal in its
    /// words. `GLYD_WRITE_P2A=1` writes the file anew (only where a change in a plan is meant).
    #[test]
    fn the_plans_are_p2as() {
        let q35 = json::parse(&fixture("qwen35_configs.json")).unwrap();
        let llama = r#"{"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8}"#;
        let configs = [
            ("Qwen3-32B", with_rope(QWEN3_32B)),
            ("Qwen3-8B", QWEN3_8B.to_string()),
            ("Qwen3.5-0.8B", json::to_compact(q35.get("Qwen/Qwen3.5-0.8B").unwrap())),
            ("Qwen3.5-9B", json::to_compact(q35.get("Qwen/Qwen3.5-9B").unwrap())),
            ("Qwen3.8-27B", json::to_compact(q35.get("Qwen/Qwen3.8-27B").unwrap())),
            ("gemma-4-12B", fixture("gemma4_12b_config.json")),
            ("NVIDIA-Nemotron-H", fixture("nemotron_h_config.json")),
            ("GLM-5.3-Flash", fixture("glm5_next_config.json")),
            ("Qwen3.8-Flash-Next", fixture("qwen4x_config.json")),
            ("Llama-3.1-8B", llama.to_string()),
        ];
        let mut got = serde_json::Map::new();
        for (name, text) in &configs {
            let dir = model_dir("p2a", name, text);
            if *name == "GLM-5.3-Flash" {
                header_only_checkpoint(&dir, &fixture("glm5_next_tensors.json"));
            }
            if *name == "NVIDIA-Nemotron-H" {
                header_only_checkpoint(&dir, &fixture("nemotron_h_tensors.json"));
            }
            let model = dir.to_str().unwrap();
            let base = dir.parent().unwrap().to_str().unwrap().to_string();
            let mut variants: Vec<Vec<&str>> = vec![vec![], vec!["--users", "4"], vec!["--context", "8192"], vec!["--context", "131072"]];
            if *name == "NVIDIA-Nemotron-H" {
                variants.extend([vec!["--expert-cache", "4"], vec!["--host-memory", "40"]]);
            }
            for preset in Machine::PRESETS {
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
            // (a case a line)
            let lines: Vec<String> = got.iter().map(|(k, v)| format!("{}: {v}", serde_json::Value::from(k.as_str()))).collect();
            std::fs::write(&file, format!("{{\n{}\n}}\n", lines.join(",\n"))).unwrap();
            return;
        }
        let want: serde_json::Map<String, serde_json::Value> = serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
        assert_eq!(got.len(), want.len(), "the cases recorded");
        let plans = want.values().filter(|v| v.get("schema").is_some()).count();
        assert!(plans > 300, "{plans} plans recorded");
        for (k, w) in &want {
            assert_eq!(got.get(k), Some(w), "{k}");
        }
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
