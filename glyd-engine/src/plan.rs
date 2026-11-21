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
    /// Bits a weight of the release has (16 for bf16; 8 for FP8; about 4.25 for FP4 releases).
    pub release_bits: f64,
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
        // (a layer takes a list entry below: a number of them no model has is a file that is not one)
        if layers > MAX_LAYERS {
            return Err(format!("config.json: num_hidden_layers {layers}, where no model has more than {MAX_LAYERS}"));
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
    /// The model's family as `run` treats it (src/support.rs): its levels, where its weights may go, its context. None:
    /// any level and placement (the planner's own view, as for a model no path runs yet).
    pub family: Option<&'static crate::support::Support>,
    /// The checkpoint is quantization-aware (`levels::is_qat`): its wren runs in every build.
    pub qat: bool,
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
            Some(Placement::ExpertsInRam { gpu, cache, hit }) => {
                serde_json::json!({"kind": "experts_in_ram", "gpus": [gpu], "cache_bytes": cache, "hit_rate": hit})
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
/// Tokens a step makes with a draft head that proposes one and is accepted 80-90% of the time, after its cost.
const MTP_GAIN: f64 = 1.7;

// ---- the reserve: one model for every path --------------------------------------------------------------------------------------------------------------------------------------------------

/// What a GPU keeps free for itself: the driver's buffers and a margin for the memory other programs take between the plan and
/// the load (768 MiB). Every path sizes with `reserve_of` (`run`, `serve`, `preflight`, `fit`); what an engine takes besides, its
/// activations, workspaces and graphs, is its `work` (below, one constant a path, each measured). A path's reserve and work
/// together are what its loader needed before the planner drove it: nothing that ran then is refused now.
pub const CONTEXT_FLOOR: u64 = 768 << 20;

/// The reserve of a GPU with `free` bytes free (the same on every GPU: other programs do not take more of a larger one).
pub fn reserve_of(_free: u64) -> u64 {
    CONTEXT_FLOOR
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
    let kv_at = |ctx: usize| (sh.kv_full * ctx as f64 + sh.kv_sliding * ctx.min(sh.window.max(1)) as f64 + sh.state) * users;
    let kv_read = sh.kv_full * POSITIONS + sh.kv_sliding * POSITIONS.min(sh.window.max(1) as f64);
    let same: Vec<usize> = (0..m.devices.len()).filter(|&i| m.devices[i].vendor == d0.vendor).collect();
    let gain = if sh.mtp { MTP_GAIN } else { 1.0 };
    // (wren on a checkpoint that is not quantization-aware is a development build's: offered only where this build runs it, or asked)
    let levels: Vec<&'static str> = levels::LEVELS
        .iter()
        .map(|l| l.name)
        .filter(|n| ask.level.map_or(levels::names_now().contains(n), |a| a == *n))
        .filter(|n| fam.is_none_or(|f| f.allows(n, ask.qat)))
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
    let chosen = choose(&options, max_ctx);
    if chosen.is_none() {
        notes.push("nothing fits this machine: a smaller model, or more memory (or other machines: glyd cluster, ROADMAP M5)".into());
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
        return gain / (gpu_t.max(cpu_t) + if sp.host_bytes > 0 { SYNC_PER_MOE_LAYER * sh.moe_layers as f64 } else { 0.0 });
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

/// An option of the ladder from the plan `run` loads at that level.
fn candidate_of(lp: LoadPlan, sh: &Shape, m: &Machine, main: usize) -> Candidate {
    let tok_s = speed_of(sh, m, main, &lp);
    let placement = match (&lp.spread, &lp.experts) {
        (Some(sp), _) if sp.host_bytes > 0 => Placement::ExpertsInRam {
            gpu: sp.devices.first().map_or(main, |&d| d as usize),
            cache: sp.gpu_bytes,
            hit: (sp.places.iter().min().copied().unwrap_or(0) as f64 / sp.experts_per_layer.max(1) as f64).min(1.0).powf(HIT_EXPONENT),
        },
        (Some(sp), _) if sp.devices.len() > 1 => Placement::Split(sp.devices.iter().map(|&d| d as usize).collect()),
        (Some(sp), _) => Placement::Gpu(sp.devices.first().map_or(main, |&d| d as usize)),
        (None, None) => Placement::Gpu(main),
        (None, Some(c)) => {
            Placement::ExpertsInRam { gpu: main, cache: c.gpu_bytes(), hit: (c.slots as f64 / c.experts.max(1) as f64).min(1.0).powf(HIT_EXPONENT) }
        }
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
    let slug: String =
        format!("{}-{level}-{gpu}", crate::levels::split_tag(model).0).chars().map(|c| if c.is_ascii_alphanumeric() || c == '.' { c } else { '_' }).collect();
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
    let main = (0..machine.devices.len()).min_by_key(|&i| (std::cmp::Reverse(machine.devices[i].free), i))?;
    let target = Target::of(&machine.devices[main], main as i32, Some(machine.ram_free));
    // (the GPUs the dynamic route uses: Ampere or newer; they reach each other directly where this computer says so, or as a datacenter machine's NVLink does)
    let devs: Vec<Target> =
        machine.devices.iter().enumerate().filter(|(_, d)| d.cc.0 >= 8).map(|(i, d)| Target::of(d, i as i32, Some(machine.ram_free))).collect();
    let linked = machine.devices.iter().all(|d| d.cc.1 == 0 && (8..=10).contains(&d.cc.0));
    let reach = move |g: &[i32]| if probed { crate::dynamic::tp::reach(g) } else { linked };
    let tries: Vec<&'static str> = match level {
        Some(l) => vec![l],
        None => fam.levels_now(model),
    };
    let (mut refusals, mut options, mut sized) = (Vec::new(), Vec::new(), false);
    for l in tries {
        match crate::cli::size(a, model, config, l, &devs, &target, &reach, users) {
            Ok(Some(lp)) => {
                sized = true;
                options.push(candidate_of(lp, sh, machine, main));
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

/// What `fit` works out of its arguments: the model, the machine, the plan, and the refusal `run` would give (its reason, after the plan is printed).
pub struct Planned {
    pub model: String,
    pub name: String,
    pub family: &'static crate::support::Support,
    pub shape: Shape,
    pub machine: Machine,
    pub plan: Plan,
    pub refusal: Option<crate::ui::Refusal>,
}

pub fn planned(a: &crate::args::Args) -> Result<Planned, crate::ui::Refusal> {
    use crate::ui::Refusal;
    let given = a.need(1, "MODEL").map_err(|e| Refusal::new(e.0, "glyd-engine fit MODEL[:LEVEL]"))?;
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
    let mut sh = Shape::from_config(&config).map_err(|e| Refusal::unsupported(format!("{name}: {e}"), ""))?;
    if let Some(total) = index_total(model) {
        sh = sh.scaled_to(total);
    }
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
            p.notes.push("nothing fits this machine: a smaller model, or more memory (or other machines: glyd cluster, ROADMAP M5)".into());
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
    Ok(Planned { model: model.to_string(), name, family: fam, shape: sh, machine, plan: p, refusal })
}

fn fit(a: &crate::args::Args) -> Result<(), crate::ui::Refusal> {
    let f = planned(a)?;
    let (model, p) = (f.model.as_str(), &f.plan);
    if a.has("json") {
        match &p.chosen {
            Some(c) => println!("{}", c.json(model, &f.machine)),
            None => println!("{}", serde_json::json!({"schema": SCHEMA, "model": model, "plan": null, "notes": p.notes})),
        }
    } else {
        print!("{}", text(model, &f.shape, &f.machine, p));
        if let Some(n) = p.chosen.as_ref().and_then(|c| remembered(&crate::store::cache_root(), model, c)) {
            println!("note: {n}");
        }
        if f.family.hummingbird_packed_only && p.chosen.as_ref().is_some_and(|c| c.level == "hummingbird") {
            println!("note: the hummingbird level comes as a pre-packed download: glyd/{}-hummingbird", f.name);
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
        // (the lossy engines keep 2 GiB of an 8 GB card: swift's 5.8 GiB of weights do not fit beside them, so only wren does, which a release build runs on QAT checkpoints alone)
        if levels::PREVIEW {
            assert!(chosen(&p).level != "penguin" && p.options[0].why.is_some(), "{p:?}");
        } else {
            assert!(p.chosen.is_none() && p.options[0].why.is_some(), "{p:?}");
        }
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
                        assert!(row.allows(o.level, qat), "{model} on {preset}: {} is not a level {} runs at", o.level, row.name);
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
            let direct = crate::cli::size_dynamic(
                &args(&["run", dir.to_str().unwrap()]),
                &dir,
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
            if *name == "GLM-5.3-Flash" {
                header_only_checkpoint(&dir, &fixture("glm5_next_tensors.json"));
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
}
