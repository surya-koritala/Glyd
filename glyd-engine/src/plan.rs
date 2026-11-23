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
        Machine {
            name: "this computer".into(),
            devices,
            ram_free: crate::offload::host_memory().unwrap_or(8 << 30),
            ram_bandwidth: 50.0,
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
    pub tok_s: f64,
    /// Why it does not fit, where it does not.
    pub why: Option<String>,
    /// The plan `run` loads, where this path is sized by the loader's own numbers (src/cli.rs `size`): the figures above are its. None: an
    /// estimate from the model's parameters (a path not yet sized by its loader).
    pub load: Option<LoadPlan>,
}

impl Candidate {
    /// The plan as `fit --json` prints it (`glyd-plan/1`): the loader's own numbers where `load` has them (`exact`), else the planner's estimate.
    pub fn json(&self, model: &str, m: &Machine) -> String {
        if let Some(l) = &self.load {
            let mut v: serde_json::Value = serde_json::from_str(&l.json(model)).unwrap_or_default();
            v["exact"] = true.into();
            v["tokens_per_second"] = serde_json::json!((self.tok_s * 10.0).round() / 10.0);
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
        serde_json::json!({
            "schema": SCHEMA, "model": model, "exact": false, "level": self.level, "context": self.context,
            "gpus": gpu, "weights_bytes": self.weights, "kv_bytes": self.kv, "placement": placement,
            "tokens_per_second": (self.tok_s * 10.0).round() / 10.0,
        })
        .to_string()
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
/// Of a GPU's memory bandwidth, what decode reaches (measured: lossy levels 0.76-0.85, lossless about 0.7; tests below).
const EFF_LOSSY: f64 = 0.80;
const EFF_LOSSLESS: f64 = 0.70;
/// Of the computer's memory bandwidth, what the CPU's expert products reach; the time a mixture-of-experts layer with its
/// experts in RAM adds for the GPU and host to meet (calibrated on Condor's Qwen3.8-Flash-Next on an RTX 4080 SUPER).
const EFF_CPU: f64 = 0.70;
const SYNC_PER_MOE_LAYER: f64 = 0.20e-3;
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
/// as the path's total before the reserve was one model, less that reserve; a prompt chunk's activations are on top (src/fit.rs `work`).
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
    let kv_read = sh.kv_full * POSITIONS + sh.kv_sliding * POSITIONS.min(sh.window.max(1) as f64);
    let same: Vec<usize> = (0..m.devices.len()).filter(|&i| m.devices[i].vendor == d0.vendor).collect();
    let gain = if sh.mtp { MTP_GAIN } else { 1.0 };
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
            load: None,
        };
        // one GPU
        if let Some(ctx) = ctx_in(room(d0) as f64, weights) {
            options.push(option(Some(Placement::Gpu(main)), ctx, gain * d0.bandwidth * GB * eff / read, None));
            continue;
        }
        // a split over the GPUs of the vendor (where the family's path splits at this level)
        let total_room: f64 = same.iter().map(|&i| room(&m.devices[i]) as f64).sum();
        if same.len() > 1 && fam.is_none_or(|f| f.split_levels.contains(&level)) {
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
                let gpu_t = (read - active * (1.0 - hit)) / (d0.bandwidth * GB * eff);
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

/// Decode tokens/s of a plan `run` loads, from the model's parameters and the machine's bandwidths (the planner's model, for a placement the
/// loader chose).
fn speed_of(sh: &Shape, m: &Machine, main: usize, lp: &LoadPlan) -> f64 {
    let d0 = &m.devices[main];
    let Some((bits, ebits)) = level_bits(lp.level, sh, d0) else { return 0.0 };
    let eff = if lp.level == "penguin" && sh.release_bits > 8.0 { EFF_LOSSLESS } else { EFF_LOSSY };
    let kv_read = sh.kv_full * POSITIONS + sh.kv_sliding * POSITIONS.min(sh.window.max(1) as f64);
    let read = ((sh.head.max(if sh.head == 0.0 { sh.embed } else { 0.0 }) + sh.dense) * bits + sh.routed_active() * ebits) / 8.0 + kv_read;
    let gain = if sh.mtp { MTP_GAIN } else { 1.0 };
    if let Some(sp) = &lp.spread {
        // (the GPUs it is over: stages read in turn, a layer across them reads on all at once; the experts the GPUs do not hold are the CPU's)
        let bw: Vec<f64> = sp.devices.iter().filter_map(|&d| m.devices.get(d as usize)).map(|d| d.bandwidth).collect();
        let n = bw.len().max(1) as f64;
        let active = sh.routed_active() * ebits / 8.0;
        let hit = if sp.host_bytes > 0 {
            (sp.places.iter().min().copied().unwrap_or(0) as f64 / sp.experts_per_layer.max(1) as f64).min(1.0).powf(HIT_EXPONENT)
        } else {
            1.0
        };
        let on_gpu = read - active * (1.0 - hit);
        let gpu_t = if sp.across {
            on_gpu / (n * bw.iter().copied().fold(f64::MAX, f64::min) * GB * eff)
        } else {
            bw.iter().map(|b| on_gpu / n / (b * GB * eff)).sum::<f64>()
        } + 0.1e-3 * (n - 1.0);
        let cpu_t = active * (1.0 - hit) / (m.ram_bandwidth * GB * EFF_CPU);
        // (the last layers' dense matrices in the computer's memory, src/dense_ram.rs: each copied to the GPU before it runs, every token)
        let link_t = sp.dense_bytes as f64 / (LINK_GBS * GB);
        return gain / (gpu_t.max(cpu_t) + link_t + if sp.host_bytes > 0 { SYNC_PER_MOE_LAYER * sh.moe_layers as f64 } else { 0.0 });
    }
    match &lp.experts {
        None => gain * d0.bandwidth * GB * eff / read,
        Some(c) => {
            let hit = (c.slots as f64 / c.experts.max(1) as f64).min(1.0).powf(HIT_EXPONENT);
            let active = sh.routed_active() * ebits / 8.0;
            let gpu_t = (read - active * (1.0 - hit)) / (d0.bandwidth * GB * eff);
            let cpu_t = active * (1.0 - hit) / (m.ram_bandwidth * GB * EFF_CPU);
            gain / (gpu_t.max(cpu_t) + SYNC_PER_MOE_LAYER * sh.moe_layers as f64)
        }
    }
}

/// Decode tokens/s of `level` with every weight on the GPU `main` (the speed a run at that level would have if it fitted).
fn ideal_speed(sh: &Shape, m: &Machine, main: usize, level: &str) -> f64 {
    let d0 = &m.devices[main];
    let Some((bits, ebits)) = level_bits(level, sh, d0) else { return 0.0 };
    let eff = if level == "penguin" && sh.release_bits > 8.0 { EFF_LOSSLESS } else { EFF_LOSSY };
    let kv_read = sh.kv_full * POSITIONS + sh.kv_sliding * POSITIONS.min(sh.window.max(1) as f64);
    let read = ((sh.head.max(if sh.head == 0.0 { sh.embed } else { 0.0 }) + sh.dense) * bits + sh.routed_active() * ebits) / 8.0 + kv_read;
    (if sh.mtp { MTP_GAIN } else { 1.0 }) * d0.bandwidth * GB * eff / read
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
                    let (ideal, got) = (ideal_speed(sh, m, main, level), speed_of(sh, m, main, &all_on_gpu));
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
                    let (before, after) = (ideal_speed(sh, m, main, asked), ideal_speed(sh, m, main, level));
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
    Candidate {
        level: lp.level,
        bits: lp.weights as f64 * 8.0 / sh.params().max(1.0),
        weights: lp.weights,
        placement: Some(placement),
        context: lp.context,
        kv: lp.kv,
        tok_s,
        why: None,
        load: Some(lp),
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
    let _ = std::fs::write(file, plan.report(crate::levels::split_tag(model).0, weights_loaded, gpu_used));
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
    s += "(speeds are estimates from memory bandwidth, for a short prompt and one user)\n";
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

/// Each level's plan `run` would load for MODEL on `machine`, or the reason it would not (the same functions: src/cli.rs `size`), in the family's order of
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
        match crate::cli::size(a, model, config, l, &devs, &target, &reach, users) {
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
                options.push(Candidate {
                    level: l,
                    bits: 0.0,
                    weights: 0,
                    placement: None,
                    context: 0,
                    kv: 0,
                    tok_s: 0.0,
                    why: Some(r.what.clone()),
                    load: None,
                });
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
    // where run's own sizing covers the family: the plan `run` would load at each level, or the refusal it would give (the same functions: src/cli.rs `size`)
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
            crate::cli::size(&q, model, &config, l, &devs, &target, &*reach, users)
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
    /// `run` calls: src/cli.rs `plan` for Qwen3, `lossy_plan_with` for the lossy families), or the refusal `run` gives; and what it plans fits the GPU.
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
                            crate::cli::plan(&args(&["run", model]), &cfg, &crate::cli::short_name(model), &gpu, None, 1).map(|p| p.2)
                            // (on Windows the name of a path is the whole path)
                        }
                        _ => {
                            let layout = crate::levels::layout_for(model, crate::levels::layout_of(level).unwrap());
                            if layout == crate::weights::Layout::Tiered && crate::nemotron::is_nemotron_h(text) {
                                continue; // (the dynamic route's: not sized by `size` yet)
                            }
                            crate::cli::lossy_plan_with(&args(&["run", model]), model, layout, text, None, d.free, &d.name, None, Some(m.ram_free))
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
        let (planned, written) = (crate::fit::weights(&cfg, Layout::Tiered), shard_bytes(&out));
        assert!(written as f64 <= planned as f64 * 1.03 && written as f64 >= planned as f64 * 0.8, "Qwen3 tiered: planned {planned}, written {written}");
        // Qwen3.5-family, each lossy level
        let text = std::fs::read_to_string(q35.join("config.json")).unwrap();
        let a = args(&["run", q35.to_str().unwrap()]);
        for layout in [Layout::Int5, Layout::Int4] {
            let out = root.join(format!("qwen35-{}", layout.name()));
            crate::pack::run_with(q35.to_str().unwrap(), &out, 2, layout, true).unwrap();
            let written = shard_bytes(&out);
            let ask = |packed: Option<&std::path::Path>| {
                crate::cli::lossy_plan_with(&a, q35.to_str().unwrap(), layout, &text, None, 24 << 30, "NVIDIA RTX 4090", packed, None).unwrap()
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
            let direct = crate::cli::size_dynamic(
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
        crate::cli::size(&a, model, text, crate::levels::LEVELS.iter().map(|l| l.name).find(|n| *n == level).unwrap(), &devs, &target, &|_| linked, 1)
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
        assert_eq!(crate::support::default_context(row, 40960), crate::fit::CTX_WANT);
        assert_eq!(crate::support::default_context(crate::support::row(crate::support::Family::Gemma4), 262_144), LOSSY_CTX);
        assert_eq!(crate::support::default_context(crate::support::row(crate::support::Family::Gemma4), 8192), 8192, "never more than the model has");
        // (fit, the Qwen3 path and the lossy path say the same thing of an impossible context)
        let dir = model_dir("ctx", "Qwen3-8B", QWEN3_8B);
        let f = planned(&args(&["fit", dir.to_str().unwrap(), "--machine", "rtx4090", "--context", "100000"])).err().unwrap();
        let a = args(&["run", "Qwen3-8B", "--context", "100000"]);
        let cfg = crate::weights::Config::parse(QWEN3_8B, "config.json", None).unwrap();
        let m = Machine::preset("rtx4090").unwrap();
        let r = crate::cli::plan(&a, &cfg, &crate::cli::short_name(dir.to_str().unwrap()), &gpu_of(&m.devices[0], 0), None, 1).unwrap_err();
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
        crate::cli::calibrated(&mut p, lp.free, 300 << 20);
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
                crate::cli::size_dynamic(&args(&["run", "GLM-5.3-Flash"]), "GLM-5.3-Flash", t, None, &config, "GLM-5.3-Flash", &devs, Some(m.ram_free), &|_| {
                    false
                })
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
