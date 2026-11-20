//! The planner: what a run of MODEL does on a machine, worked out before anything is downloaded. It chooses the level, where
//! the weights, the experts and the KV cache live, the context, and predicts decode tokens/s, from the model's config.json
//! and the machine. Pure functions of explicit inputs (a probed machine, or a preset or JSON description of one), so every
//! branch is a host test. `glyd-engine fit MODEL[:LEVEL] [--machine NAME|FILE]` prints the plan.
//!
//! The order of preference (docs/architecture.md, "Adaptation"): the release's weights exactly (penguin) when they fit the
//! GPUs, else the highest lossy level that does, else (a mixture of experts) its experts in the computer's memory with the
//! CPU computing the ones the GPU does not hold, else a refusal with the numbers. One GPU is preferred to a split.

use crate::levels;
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
    /// Memory bandwidth, GB/s.
    pub bandwidth: f64,
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

fn gpu(name: &str, vendor: Vendor, cc: (i32, i32), total_gib: f64, free_gib: f64, driver: i32) -> Device {
    let gib = |g: f64| (g * GIB) as u64;
    Device {
        name: name.into(),
        vendor,
        cc,
        total: gib(total_gib),
        free: gib(free_gib),
        bandwidth: bandwidth_of(name).unwrap_or(300.0),
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
        Some(Machine { name: name.into(), devices, ram_free: (ram_gib * GIB) as u64, ram_bandwidth: ram_bw, cores })
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
        })
    }

    /// This computer: its GPUs as the driver reports them, its free memory.
    pub fn probe() -> Machine {
        let vendor = if cfg!(glyd_hip) { Vendor::Amd } else { Vendor::Nvidia };
        let devices = crate::dev::Dev::probe_all()
            .map(|gs| {
                gs.into_iter()
                    .map(|g| Device {
                        bandwidth: bandwidth_of(&g.name).unwrap_or(300.0),
                        unified: is_unified(&g.name, vendor),
                        name: g.name,
                        vendor,
                        cc: g.cc,
                        total: g.total,
                        free: g.free,
                        driver: g.driver,
                    })
                    .collect()
            })
            .unwrap_or_default();
        let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
        Machine { name: "this computer".into(), devices, ram_free: crate::offload::host_memory().unwrap_or(8 << 30), ram_bandwidth: 50.0, cores }
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
    /// Bits a weight of the release has (16 for bf16; 8 for FP8; about 4.25 for FP4 releases).
    pub release_bits: f64,
    /// A draft head (multi-token prediction) the engine drafts with.
    pub mtp: bool,
}

impl Shape {
    pub fn params(&self) -> f64 {
        self.embed + self.head + self.dense + self.routed
    }

    pub fn is_moe(&self) -> bool {
        self.experts > 0
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
        let root = json::parse(text)?;
        let t = root.get("text_config").filter(|v| v.as_object().is_some()).unwrap_or(&root);
        let n = |k: &str| num(t, k).or_else(|| num(&root, k));
        let u = |k: &str| n(k).unwrap_or(0.0) as usize;
        let s = |k: &str| t.get(k).or_else(|| root.get(k)).and_then(Value::as_str).map(str::to_string);
        let arch = root.get("architectures").and_then(Value::as_array).and_then(|a| a.first()).and_then(Value::as_str).unwrap_or("").to_string();
        let model_type = s("model_type").unwrap_or_default();
        let (h, layers, heads) = (u("hidden_size"), u("num_hidden_layers"), u("num_attention_heads"));
        if h == 0 || layers == 0 || heads == 0 {
            return Err("config.json has no hidden_size, num_hidden_layers or num_attention_heads".into());
        }
        let (hf, kv_heads) = (h as f64, n("num_key_value_heads").unwrap_or(heads as f64));
        let hd = n("head_dim").filter(|&d| d > 0.0).unwrap_or(hf / heads as f64);
        let vocab = n("vocab_size").unwrap_or(32000.0);
        let tied = t.get("tie_word_embeddings").or_else(|| root.get("tie_word_embeddings")).and_then(Value::as_bool).unwrap_or(false);
        // the layers' kinds: layer_types, Nemotron-H's pattern, else every layer full attention
        let pattern = s("hybrid_override_pattern");
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
        } else if let Some(a) = t.get("layer_types").and_then(Value::as_array) {
            a.iter().map(|x| x.as_str().unwrap_or("full_attention").to_string()).collect()
        } else {
            vec!["full_attention".to_string(); layers]
        };
        let experts = u("num_experts").max(u("num_local_experts")).max(u("n_routed_experts"));
        let topk = u("num_experts_per_tok");
        let moe_inter = n("moe_intermediate_size").unwrap_or(0.0);
        let shared_inter = n("shared_expert_intermediate_size")
            .or_else(|| n("moe_shared_expert_intermediate_size"))
            .unwrap_or(moe_inter * n("n_shared_experts").unwrap_or(0.0));
        let gated = if model_type.contains("nemotron") { 2.0 } else { 3.0 }; // (Nemotron-H's MLPs are relu²: two matrices)
                                                                             // (a latent mixture of experts, Nemotron 3 Super's: the experts work on a narrower vector, projected down and up around them)
        let latent = n("moe_latent_size").unwrap_or(0.0);
        let expert_width = if latent > 0.0 { latent } else { hf };
        let first_dense = u("first_k_dense_replace");
        // attention
        let (q_lora, kv_lora) = (n("q_lora_rank").unwrap_or(0.0), n("kv_lora_rank").unwrap_or(0.0));
        let (nope, rope, v_hd) = (n("qk_nope_head_dim").unwrap_or(128.0), n("qk_rope_head_dim").unwrap_or(64.0), n("v_head_dim").unwrap_or(128.0));
        let gate = if t.get("attn_output_gate").and_then(Value::as_bool).unwrap_or(false) { 2.0 } else { 1.0 };
        let hq = heads as f64;
        let (g_kv, g_hd) = (n("num_global_key_value_heads").unwrap_or(kv_heads), n("global_head_dim").unwrap_or(hd));
        let attention = |global: bool| -> (f64, f64) {
            if kv_lora > 0.0 {
                let q = if q_lora > 0.0 { hf * q_lora + q_lora * hq * (nope + rope) } else { hf * hq * (nope + rope) };
                (q + hf * (kv_lora + rope) + kv_lora * hq * (nope + v_hd) + hq * v_hd * hf, (kv_lora + rope) * 2.0)
            } else {
                let (kv, d) = if global { (g_kv, g_hd) } else { (kv_heads, hd) };
                (hf * hq * d * gate + 2.0 * hf * kv * d + hq * d * hf, 2.0 * kv * d * 2.0)
            }
        };
        let (lk, lv) = (
            n("linear_num_key_heads").unwrap_or(16.0) * n("linear_key_head_dim").unwrap_or(128.0),
            n("linear_num_value_heads").unwrap_or(32.0) * n("linear_value_head_dim").or_else(|| n("linear_key_head_dim")).unwrap_or(128.0),
        );
        let (m_heads, m_hd, m_state, m_groups) =
            (n("mamba_num_heads").unwrap_or(0.0), n("mamba_head_dim").unwrap_or(64.0), n("ssm_state_size").unwrap_or(128.0), n("n_groups").unwrap_or(8.0));
        let mut sh =
            Shape { arch: if arch.is_empty() { model_type.clone() } else { arch }, layers: kinds.len(), hidden: h, experts, topk, ..Default::default() };
        sh.window = u("sliding_window");
        for (i, kind) in kinds.iter().enumerate() {
            match kind.as_str() {
                "linear_attention" => {
                    sh.dense += hf * 2.0 * lk + hf * 2.0 * lv + lv * hf;
                    sh.state += n("linear_num_value_heads").unwrap_or(32.0)
                        * n("linear_key_head_dim").unwrap_or(128.0)
                        * n("linear_value_head_dim").unwrap_or(128.0)
                        * 4.0;
                }
                "mamba" => {
                    let inner = m_heads * m_hd;
                    sh.dense += hf * (2.0 * inner + 2.0 * m_groups * m_state + m_heads) + inner * hf;
                    sh.state += m_heads * m_hd * m_state * 4.0;
                }
                k if k.contains("attention") => {
                    let sliding = k.contains("sliding") && sh.window > 0;
                    let (p, kv) = attention(!sliding && pattern.is_none() && t.get("global_head_dim").is_some());
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
                sh.dense += gated * hf * n("intermediate_size").unwrap_or(4.0 * hf);
            }
        }
        sh.embed = vocab * hf;
        sh.head = if tied { 0.0 } else { vocab * hf };
        sh.max_ctx = u("max_position_embeddings").max(4096);
        let quant = root.get("quantization_config").and_then(|q| q.get("quant_method")).and_then(Value::as_str).unwrap_or("");
        sh.release_bits = match quant {
            "fp8" => 8.0,
            "mxfp4" | "nvfp4" => 4.25,
            _ => 16.0,
        };
        sh.mtp = n("mtp_num_hidden_layers").or_else(|| n("num_nextn_predict_layers")).unwrap_or(0.0) > 0.0;
        Ok(sh)
    }

    /// The same model with its parameters scaled to the checkpoint's bytes (model.safetensors.index.json's total_size), where known.
    pub fn scaled_to(mut self, total_bytes: f64) -> Shape {
        let ours = self.params() * self.release_bits / 8.0;
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
}

#[derive(Clone, Debug, PartialEq)]
pub enum Placement {
    /// Every weight on one GPU (its index in the machine's list).
    Gpu(usize),
    /// Layers split over these GPUs, in proportion to their room.
    Split(Vec<usize>),
    /// A mixture of experts' experts in the computer's memory, a cache of them on the GPU; `hit`: the share of expert reads the cache serves.
    ExpertsInRam { gpu: usize, cache: u64, hit: f64 },
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
    pub tok_s: f64,
    /// Why it does not fit, where it does not.
    pub why: Option<String>,
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
/// Of a GPU's memory bandwidth, what decode reaches (measured: lossy levels 0.76-0.85, lossless about 0.7; tests below).
const EFF_LOSSY: f64 = 0.80;
const EFF_LOSSLESS: f64 = 0.70;
/// Of the computer's memory bandwidth, what the CPU's expert products reach; the time a mixture-of-experts layer with its
/// experts in RAM adds for the GPU and host to meet (calibrated on Condor's Qwen3.8-Flash-Next on an RTX 4080 SUPER).
const EFF_CPU: f64 = 0.70;
const SYNC_PER_MOE_LAYER: f64 = 0.20e-3;
/// The share of expert reads a cache holding a share `f` of the experts serves: experts are used unevenly (Condor: 70-76% at about a quarter).
const HIT_EXPONENT: f64 = 0.23;
/// Tokens a step makes with a draft head that proposes one and is accepted 80-90% of the time, after its cost.
const MTP_GAIN: f64 = 1.7;

/// What a GPU keeps for itself: the engine's workspace and the driver's, and a margin for other programs (1 GiB as llama.cpp's
/// fitting keeps, and 4% of what is free). `run` still sizes with its own constants (cli.rs `rooms`) until the planner drives it.
fn reserve(d: &Device) -> u64 {
    (1.0 * GIB) as u64 + d.free / 25
}

fn room(d: &Device) -> u64 {
    d.free.saturating_sub(reserve(d))
}

/// Bits a weight of each part takes at a level: (everything else, the routed experts). None: the level does not apply.
fn level_bits(level: &str, sh: &Shape, d: &Device) -> Option<(f64, f64)> {
    let lossless_bits = if sh.release_bits <= 8.0 {
        sh.release_bits
    } else if matches!(d.cc, (8, 0) | (9, 0) | (10, 0)) {
        12.04 // (A100, H100, B200 take the 12-bit layout: it is the faster there)
    } else {
        10.75
    };
    let b = match level {
        "penguin" => (lossless_bits, lossless_bits),
        "kestrel" => (6.5, 6.5),
        "swift" => (5.5, 5.5),
        "wren" => (4.5, 4.5),
        "hummingbird" if sh.is_moe() => (6.0, 2.9),
        _ => return None,
    };
    (b.0 < sh.release_bits).then_some(b).or((level == "penguin").then_some(b))
}

/// The plan for `sh` on `m`.
pub fn plan(sh: &Shape, m: &Machine, ask: Ask) -> Plan {
    let mut notes = Vec::new();
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
    let backend = match d0.vendor {
        Vendor::Nvidia if d0.driver >= crate::machine::DRIVER_MIN => format!("CUDA (driver CUDA {}.{})", d0.driver / 1000, d0.driver % 1000 / 10),
        Vendor::Nvidia => {
            notes.push(format!(
                "the driver supports CUDA {}.{}: the engine's kernels need 13.0 (driver 580) until the CUDA 12 build (ROADMAP P6)",
                d0.driver / 1000,
                d0.driver % 1000 / 10
            ));
            "CUDA 12 (planned)".into()
        }
        Vendor::Amd => "ROCm/HIP (development build)".into(),
        Vendor::Apple => "Metal via MLX (planned, ROADMAP V3)".into(),
    };
    if d0.vendor == Vendor::Nvidia && d0.cc.0 < 8 {
        notes.push(format!("{} is older than Ampere: Turing and Volta support is ROADMAP V5", d0.name));
    }
    let users = ask.users.max(1) as f64;
    let max_ctx = ask.context.unwrap_or(TARGET_CTX.min(sh.max_ctx));
    let kv_at = |ctx: usize| (sh.kv_full * ctx as f64 + sh.kv_sliding * ctx.min(sh.window.max(1)) as f64 + sh.state) * users;
    let kv_read = sh.kv_full * POSITIONS + sh.kv_sliding * POSITIONS.min(sh.window.max(1) as f64);
    let same: Vec<usize> = (0..m.devices.len()).filter(|&i| m.devices[i].vendor == d0.vendor).collect();
    let gain = if sh.mtp { MTP_GAIN } else { 1.0 };
    // (wren on a checkpoint that is not quantization-aware is a development build's: offered only where this build runs it, or asked)
    let levels: Vec<&'static str> = levels::LEVELS.iter().map(|l| l.name).filter(|n| ask.level.map_or(levels::names_now().contains(n), |a| a == *n)).collect();
    let mut options = Vec::new();
    for &level in &levels {
        let Some((bits, ebits)) = level_bits(level, sh, d0) else { continue };
        let rest = (sh.embed + sh.head + sh.dense) * bits / 8.0;
        let experts = sh.routed * ebits / 8.0;
        let weights = rest + experts;
        let eff = if level == "penguin" && sh.release_bits > 8.0 { EFF_LOSSLESS } else { EFF_LOSSY };
        let read = ((sh.head.max(if sh.head == 0.0 { sh.embed } else { 0.0 }) + sh.dense) * bits + sh.routed_active() * ebits) / 8.0 + kv_read;
        // the largest context that fits in `budget` bytes once `fixed` are placed
        let ctx_in = |budget: f64, fixed: f64| -> Option<usize> {
            let fits: Vec<usize> = CONTEXTS.iter().copied().filter(|&c| c <= max_ctx && fixed + kv_at(c) <= budget).collect();
            fits.last()
                .copied()
                .or_else(|| (ask.context.is_none() && fixed + kv_at(MIN_CTX) <= budget).then_some(MIN_CTX))
                .filter(|&c| c >= MIN_CTX.min(max_ctx))
        };
        let option = |placement: Option<Placement>, context: usize, tok_s: f64, why: Option<String>| Candidate {
            level,
            bits: if experts > 0.0 && ebits != bits { (rest * 8.0 + experts * 8.0) / (sh.params()) } else { bits },
            weights: weights as u64,
            placement,
            context,
            kv: kv_at(context) as u64,
            tok_s,
            why,
        };
        // one GPU
        if let Some(ctx) = ctx_in(room(d0) as f64, weights) {
            options.push(option(Some(Placement::Gpu(main)), ctx, gain * d0.bandwidth * GB * eff / read, None));
            continue;
        }
        // a split over the GPUs of the vendor
        let total_room: f64 = same.iter().map(|&i| room(&m.devices[i]) as f64).sum();
        if same.len() > 1 {
            if let Some(ctx) = ctx_in(total_room, weights) {
                // (one GPU at a time: each reads its share, in proportion to its room)
                let t: f64 = same.iter().map(|&i| read * room(&m.devices[i]) as f64 / total_room / (m.devices[i].bandwidth * GB * eff)).sum::<f64>()
                    + 0.1e-3 * (same.len() - 1) as f64;
                options.push(option(Some(Placement::Split(same.clone())), ctx, gain / t, None));
                continue;
            }
        }
        // a mixture of experts: the experts in the computer's memory, the rest and a cache of experts on the GPU (not where the GPU
        // already uses the computer's memory: there RAM is the same memory, counted once)
        if sh.is_moe() && !d0.unified {
            let ram = (m.ram_free as f64 - 4.0 * GIB) * 0.9;
            let min_cache = 0.05 * experts;
            if experts > ram {
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
                let gpu_t = (read - active * (1.0 - hit)) / (d0.bandwidth * GB * eff);
                let cpu_t = active * (1.0 - hit) / (m.ram_bandwidth * GB * EFF_CPU);
                let t = gpu_t.max(cpu_t) + SYNC_PER_MOE_LAYER * sh.moe_layers as f64;
                options.push(option(Some(Placement::ExpertsInRam { gpu: main, cache: cache as u64, hit }), ctx, gain / t, None));
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
    // in the levels' order: the first in GPU memory alone with at least PREFERRED_CTX of context, else the first with experts in
    // RAM at that context; only then a shorter context (a model held in 4,096 tokens is a poorer run than one level down at 32,768)
    let in_vram = |o: &&Candidate| matches!(o.placement, Some(Placement::Gpu(_) | Placement::Split(_)));
    let roomy = |o: &&Candidate| o.placement.is_some() && o.context >= PREFERRED_CTX.min(max_ctx);
    let chosen = (options.iter().find(|o| in_vram(o) && roomy(o)))
        .or_else(|| options.iter().find(roomy))
        .or_else(|| options.iter().find(in_vram))
        .or_else(|| options.iter().find(|o| o.placement.is_some()))
        .cloned();
    if chosen.is_none() {
        notes.push("nothing fits this machine: a smaller model, or more memory (or other machines: glyd cluster, ROADMAP M5)".into());
    }
    Plan { backend, chosen, options, notes }
}

// ---- the command ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

fn gb(b: u64) -> String {
    format!("{:.1} GiB", b as f64 / GIB)
}

fn where_(p: &Placement, m: &Machine) -> String {
    match p {
        Placement::Gpu(i) => format!("GPU {i} ({})", m.devices[*i].name),
        Placement::Split(g) => format!("split over GPUs {}", g.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(", ")),
        Placement::ExpertsInRam { cache, hit, .. } => {
            format!("experts in RAM, a {} cache on the GPU (~{:.0}% of expert reads), the CPU computes the rest", gb(*cache), hit * 100.0)
        }
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
        s += &format!("GPU {i}: {} · {} free of {} · {:.0} GB/s\n", d.name, gb(d.free), gb(d.total), d.bandwidth);
    }
    s += &format!("RAM: {} free · {} cores · backend: {}\n", gb(m.ram_free), m.cores, p.backend);
    match &p.chosen {
        Some(c) => {
            s += &format!(
                "Plan: {} ({:.1} bits) · {} · {} tokens of context · ~{:.0} tokens/s\n",
                c.level,
                c.bits,
                where_(c.placement.as_ref().unwrap(), m),
                c.context,
                c.tok_s
            )
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
    s += "(speeds are estimates from memory bandwidth, for a short prompt and one user)\n";
    s
}

/// `glyd-engine fit MODEL[:LEVEL] [--machine NAME|FILE] [--context N] [--users N]`: the plan, before anything is downloaded.
pub fn fit_cmd(a: &crate::args::Args) -> crate::ffi::Result<()> {
    use crate::ffi::{err, Error};
    let given = a.need(1, "MODEL")?;
    let (model, layout) = levels::parse(given, false).map_err(|e| Error(format!("{} ({})", e.what, e.fix)))?;
    let level = levels::split_tag(given).1.and(Some(layout.map_or("penguin", levels::name_of)));
    let machine = match a.flag("machine") {
        None => Machine::probe(),
        Some(n) => match Machine::preset(n) {
            Some(m) => m,
            None => Machine::from_json(
                &std::fs::read_to_string(n).map_err(|e| Error(format!("--machine {n}: not a preset ({}) and not a file: {e}", Machine::PRESETS.join(", "))))?,
            )
            .map_err(|e| Error(format!("{n}: {e}")))?,
        },
    };
    let config = crate::cli::config_text(model).map_err(|r| Error(r.what))?;
    let mut sh = Shape::from_config(&config).map_err(|e| Error(format!("{model}: {e}")))?;
    if let Some(total) = index_total(model) {
        sh = sh.scaled_to(total);
    }
    let ask = Ask {
        level,
        context: a.flag("context").map(|c| c.parse()).transpose().map_err(|_| Error("--context takes a number".into()))?,
        users: a.num("users", 1usize)?,
    };
    let p = plan(&sh, &machine, ask);
    print!("{}", text(model, &sh, &machine, &p));
    if p.chosen.is_none() {
        return err("no plan fits");
    }
    Ok(())
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

    #[test]
    fn the_ladder_picks_lossless_then_lossy_then_experts_in_ram() {
        let q9 = qwen35("Qwen/Qwen3.5-9B");
        // a 24 GB card holds the 9B lossless; an 8 GB laptop card takes a lossy level, and says the driver is too old today
        let p = plan(&q9, &Machine::preset("rtx4090").unwrap(), Ask::default());
        assert_eq!((chosen(&p).level, chosen(&p).placement.clone()), ("penguin", Some(Placement::Gpu(0))));
        let p = plan(&q9, &Machine::preset("rtx4060-laptop").unwrap(), Ask::default());
        assert!(chosen(&p).level != "penguin" && p.options[0].why.is_some(), "{p:?}");
        assert!(p.notes.iter().any(|n| n.contains("CUDA 12 build")));
        // Qwen3-32B lossless on one 48 GB card (README: bf16 needs two); two 24 GB cards hold it at kestrel, split; one 24 GB card at a lower level
        let q32 = Shape::from_config(QWEN3_32B).unwrap();
        let p = plan(&q32, &Machine::preset("a6000").unwrap(), Ask::default());
        assert_eq!((chosen(&p).level, chosen(&p).placement.clone()), ("penguin", Some(Placement::Gpu(0))));
        let p = plan(&q32, &Machine::preset("2x3090").unwrap(), Ask::default());
        assert_eq!((chosen(&p).level, chosen(&p).placement.clone()), ("kestrel", Some(Placement::Split(vec![0, 1]))));
        // (one 24 GB card: only wren holds it, which a release build runs on QAT checkpoints alone)
        let p = plan(&q32, &Machine::preset("rtx4090").unwrap(), Ask::default());
        if levels::PREVIEW {
            assert_eq!(chosen(&p).level, "wren")
        } else {
            assert!(p.chosen.is_none(), "{p:?}")
        }
        // the 180B MoE fits no 16 GB card at any level but in RAM; on eight H100s it is lossless, split
        let qx = Shape::from_config(&fixture("qwen4x_config.json")).unwrap();
        let p = plan(&qx, &Machine::preset("rtx4080s").unwrap(), Ask::default());
        assert!(matches!(chosen(&p).placement, Some(Placement::ExpertsInRam { .. })) && chosen(&p).level == "hummingbird", "{p:?}");
        let p = plan(&qx, &Machine::preset("8xh100").unwrap(), Ask::default());
        assert!(chosen(&p).level == "penguin" && matches!(chosen(&p).placement, Some(Placement::Gpu(_) | Placement::Split(_))));
        // the context is the most that fits, to 32,768 (or what is asked); a machine with no GPU has no plan yet
        assert_eq!(chosen(&plan(&q9, &Machine::preset("rtx4090").unwrap(), Ask::default())).context, TARGET_CTX);
        let none = Machine { name: "cpu".into(), devices: vec![], ram_free: 64 << 30, ram_bandwidth: 80.0, cores: 16 };
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
}
