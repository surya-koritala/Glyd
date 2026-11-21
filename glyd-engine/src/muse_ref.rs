//! The CPU reference of Muse Spark's text model (src/muse.rs), one sequence a token at a time, from the checkpoint's weights: fp32 inside (the products' sums in f64), rounded to bf16 where transformers'
//! modeling_muse_spark.py holds a bf16 tensor (each bf16 operation's result, as eager PyTorch rounds it). Its attention sums in fp32 (transformers' eager attention rounds the scores to bf16 first; vLLM's do
//! not). Equal router scores: the smaller expert first. The GPU's kernels are checked against it; `random` writes a small model of the release's layout. A development build's only.

use crate::ffi::{err, Error, Result};
use crate::muse::{bf, plus_one, residual_gate, ConfigMuse};
use glyd_gpu::safetensors::Checkpoint;
use std::path::Path;

fn io<T>(r: std::io::Result<T>) -> Result<T> {
    r.map_err(|e| Error(e.to_string()))
}

/// A tensor's values as f32 (bf16 or fp32), and its shape.
fn tensor(ck: &Checkpoint, name: &str) -> Result<(Vec<u64>, Vec<f32>)> {
    let (_, t) = ck.get(name).ok_or_else(|| Error(format!("no tensor {name}")))?;
    let (dtype, shape) = (t.dtype.clone(), t.shape.clone());
    let b = io(ck.read(name))?;
    let v = match dtype.as_str() {
        "BF16" => b.chunks(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
        "F32" => b.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        d => return err(format!("{name}: a tensor of {d}")),
    };
    Ok((shape, v))
}

/// Rows of a matrix: W [rows, cols] times x, the sums in f64, each rounded to f32.
fn mul(w: &[f32], cols: usize, x: &[f32]) -> Vec<f32> {
    w.chunks(cols).map(|r| r.iter().zip(x).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>() as f32).collect()
}

/// RMS of x with eps, the f32 factor (x^2's mean in fp32).
fn rs(x: &[f32], eps: f32) -> f32 {
    (x.iter().map(|&v| v * v).sum::<f32>() / x.len() as f32 + eps).powf(-0.5)
}

/// The RMS norm without a weight, rounded once.
fn rms(x: &[f32], eps: f32) -> Vec<f32> {
    let r = rs(x, eps);
    x.iter().map(|&v| bf(v * r)).collect()
}

/// The RMS norm with a weight (its factor for each channel: the weight, or a centered norm's bf16(1 + w)), the product rounded once.
fn rms_w(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let r = rs(x, eps);
    x.iter().zip(w).map(|(&v, &g)| bf(v * r * g)).collect()
}

/// The input-scaled norm: x times its factor (bf16(1 + w)) rounded to bf16, then the RMS norm without a weight.
fn rms_in(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let y: Vec<f32> = x.iter().zip(w).map(|(&v, &g)| bf(v * g)).collect();
    rms(&y, eps)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

struct Layer {
    w_in: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    g: Vec<f32>,
    o: Vec<f32>,
    ga: (Vec<f32>, Vec<f32>),
    w_ff: Vec<f32>,
    router: Vec<f32>,
    bias: Vec<f32>,
    pre: Vec<f32>,
    w_pre: Vec<f32>,
    gu: Vec<f32>,
    down: Vec<f32>,
    w_post_e: Vec<f32>,
    post: Vec<f32>,
    w_post_ff: Vec<f32>,
    gf: (Vec<f32>, Vec<f32>),
    keys: Vec<Vec<f32>>,
    vals: Vec<Vec<f32>>,
}

pub struct RefMuse {
    pub c: ConfigMuse,
    embed: Vec<f32>,
    head: Vec<f32>,
    norm: Vec<f32>,
    layers: Vec<Layer>,
    pub pos: usize,
    /// The step's closest expert choice: the smallest gap, over its layers, between the k-th choice's score (plus bias) and the next one's.
    pub margin: std::cell::Cell<f32>,
}

impl RefMuse {
    pub fn load(dir: &Path) -> Result<RefMuse> {
        let c = ConfigMuse::load(dir)?;
        let ck = io(Checkpoint::open(dir))?;
        let t = |n: &str| -> Result<Vec<f32>> { Ok(tensor(&ck, n)?.1) };
        let mut layers = Vec::new();
        for l in 0..c.layers {
            let p = format!("{}layers.{l}", c.prefix);
            let (a, m) = (format!("{p}.self_attn"), format!("{p}.mlp"));
            layers.push(Layer {
                w_in: plus_one(&t(&format!("{p}.input_layernorm.weight"))?),
                q: t(&format!("{a}.q_proj.weight"))?,
                k: t(&format!("{a}.k_proj.weight"))?,
                v: t(&format!("{a}.v_proj.weight"))?,
                g: t(&format!("{a}.gate_proj.weight"))?,
                o: t(&format!("{a}.o_proj.weight"))?,
                ga: residual_gate(&t(&format!("{p}.post_attention_residual_gate.gate"))?, c.gate_temp),
                w_ff: plus_one(&t(&format!("{p}.pre_feedforward_layernorm.weight"))?),
                router: t(&format!("{m}.gate.weight"))?,
                bias: t(&format!("{m}.gate.e_score_correction_bias"))?,
                pre: t(&format!("{m}.pre_expert_proj.weight"))?,
                w_pre: plus_one(&t(&format!("{m}.pre_expert_norm.weight"))?),
                gu: t(&format!("{m}.experts.gate_up_proj"))?,
                down: t(&format!("{m}.experts.down_proj"))?,
                w_post_e: plus_one(&t(&format!("{m}.experts.post_expert_norm.weight"))?),
                post: t(&format!("{m}.post_expert_proj.weight"))?,
                w_post_ff: plus_one(&t(&format!("{p}.post_feedforward_layernorm.weight"))?),
                gf: residual_gate(&t(&format!("{p}.post_feedforward_residual_gate.gate"))?, c.gate_temp),
                keys: Vec::new(),
                vals: Vec::new(),
            });
        }
        let head = t("lm_head.weight")?;
        Ok(RefMuse {
            embed: t(&format!("{}embed_tokens.weight", c.prefix))?,
            head,
            norm: t(&format!("{}norm.weight", c.prefix))?,
            layers,
            c,
            pos: 0,
            margin: std::cell::Cell::new(f32::INFINITY),
        })
    }

    fn attention(&mut self, l: usize, h: &[f32]) -> Vec<f32> {
        let (c, pos) = (&self.c, self.pos);
        let (d, nh, nk) = (c.hidden, c.heads, c.kv_heads);
        let hd = c.head_dim;
        let ly = &mut self.layers[l];
        let r = |w: &[f32]| -> Vec<f32> { mul(w, d, h).into_iter().map(bf).collect() };
        let (q, k, v, g) = (r(&ly.q), r(&ly.k), r(&ly.v), r(&ly.g));
        // each head normed without a weight; the queries times the scale; the rotary (rotate_half, cos and sin in bf16) on the window's layers
        let prep = |x: &[f32], scale: Option<f32>| -> Vec<f32> {
            let mut o = Vec::with_capacity(x.len());
            for hh in x.chunks(hd) {
                let mut n = rms(hh, c.eps);
                if let Some(s) = scale {
                    n = n.into_iter().map(|v| bf(v * s)).collect();
                }
                if c.sliding[l] {
                    let f = c.freqs();
                    let half = hd / 2;
                    let mut out = vec![0f32; hd];
                    for i in 0..hd {
                        let ang = pos as f32 * f[i % half];
                        let (cs, sn) = (bf((ang as f64).cos() as f32), bf((ang as f64).sin() as f32));
                        let rot = if i < half { -n[i + half] } else { n[i - half] };
                        out[i] = bf(bf(n[i] * cs) + bf(rot * sn));
                    }
                    n = out;
                }
                o.extend(n);
            }
            o
        };
        let q = prep(&q, Some(c.qk_scale));
        ly.keys.push(prep(&k, None));
        ly.vals.push(v);
        let first = if c.sliding[l] { (pos + 1).saturating_sub(c.window) } else { 0 };
        let scale = (hd as f32).powf(-0.5);
        let mut att = Vec::with_capacity(nh * hd);
        for hh in 0..nh {
            let kv = hh / (nh / nk);
            let qh = &q[hh * hd..][..hd];
            let s: Vec<f32> =
                (first..=pos).map(|j| (qh.iter().zip(&ly.keys[j][kv * hd..][..hd]).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>() as f32) * scale).collect();
            let mx = s.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x));
            let e: Vec<f32> = s.iter().map(|&x| (x - mx).exp()).collect();
            let sum: f32 = e.iter().sum();
            let mut o = vec![0f32; hd];
            for (j, &ev) in (first..=pos).zip(&e) {
                for (ov, &vv) in o.iter_mut().zip(&ly.vals[j][kv * hd..][..hd]) {
                    *ov += ev / sum * vv;
                }
            }
            let o: Vec<f32> = o.into_iter().map(bf).collect();
            // the output normed per head, then gated by sigmoid of the gate's product
            let on = rms(&o, c.eps);
            att.extend(on.iter().zip(&g[hh * hd..][..hd]).map(|(&x, &gv)| bf(x * bf(sigmoid(gv)))));
        }
        mul(&ly.o, nh * hd, &att).into_iter().map(bf).collect()
    }

    fn moe(&self, l: usize, h: &[f32]) -> Vec<f32> {
        let (c, ly) = (&self.c, &self.layers[l]);
        let (d, lat, inter) = (c.hidden, c.latent, c.inter);
        let s: Vec<f32> = mul(&ly.router, d, h).iter().map(|&x| sigmoid(x * c.out_mult)).collect();
        let mut ord: Vec<usize> = (0..c.experts).collect();
        ord.sort_by(|&i, &j| (s[j] + ly.bias[j]).total_cmp(&(s[i] + ly.bias[i])).then(i.cmp(&j)));
        if c.top_k < c.experts {
            let gap = (s[ord[c.top_k - 1]] + ly.bias[ord[c.top_k - 1]]) - (s[ord[c.top_k]] + ly.bias[ord[c.top_k]]);
            self.margin.set(self.margin.get().min(gap));
        }
        ord.truncate(c.top_k);
        let tot: f32 = ord.iter().map(|&e| s[e]).sum::<f32>() + 1e-15;
        let x = rms_w(&mul(&ly.pre, d, h).into_iter().map(bf).collect::<Vec<_>>(), &ly.w_pre, c.eps);
        let mut acc = vec![0f32; lat];
        let mut by_expert = ord.clone();
        by_expert.sort_unstable();
        for e in by_expert {
            let w = s[e] / tot * c.route_scale;
            let gu: Vec<f32> = mul(&ly.gu[e * 2 * inter * lat..][..2 * inter * lat], lat, &x).into_iter().map(bf).collect();
            let a: Vec<f32> = (0..inter).map(|i| bf(bf(gu[i] * sigmoid(gu[i])) * gu[inter + i])).collect();
            let dn: Vec<f32> = mul(&ly.down[e * lat * inter..][..lat * inter], inter, &a).into_iter().map(bf).collect();
            let dn = rms_in(&dn, &ly.w_post_e, c.post_eps);
            for (av, &dv) in acc.iter_mut().zip(&dn) {
                *av = bf(*av + bf(dv * w));
            }
        }
        mul(&ly.post, lat, &acc).into_iter().map(bf).collect()
    }

    /// One token at the next position: its logits (bf16 values, soft-capped as transformers does).
    pub fn step(&mut self, token: u32) -> Vec<f32> {
        self.margin.set(f32::INFINITY);
        let (d, eps, pe) = (self.c.hidden, self.c.eps, self.c.post_eps);
        let mut x = rms(&self.embed[token as usize * d..][..d], eps);
        let gate = |x: &[f32], o: &[f32], (a, b): &(Vec<f32>, Vec<f32>)| -> Vec<f32> {
            x.iter().zip(o).zip(a.iter().zip(b)).map(|((&r, &h), (&al, &be))| bf(bf(al * r) + bf(be * h))).collect()
        };
        for l in 0..self.layers.len() {
            let h = rms_w(&x, &self.layers[l].w_in, eps);
            let a = rms(&self.attention(l, &h), pe);
            x = gate(&x, &a, &self.layers[l].ga);
            let h = rms_w(&x, &self.layers[l].w_ff, eps);
            let f = rms_in(&self.moe(l, &h), &self.layers[l].w_post_ff, pe);
            x = gate(&x, &f, &self.layers[l].gf);
        }
        self.pos += 1;
        let y = rms_w(&x, &self.norm, eps);
        let (m, cap) = (self.c.out_mult, self.c.softcap);
        mul(&self.head, d, &y).into_iter().map(|v| bf(bf(bf(bf(bf(v) * m) / cap).tanh()) * cap)).collect()
    }
}

#[cfg(test)]
impl RefMuse {
    /// A model of attention layers only, from each layer's q, k, v, gate and o weights (the runtime's kind tests,
    /// `runtime::reference::ref_tests`).
    pub(crate) fn attention_only(c: ConfigMuse, w: Vec<[Vec<f32>; 5]>) -> RefMuse {
        let layers = w
            .into_iter()
            .map(|[q, k, v, g, o]| Layer {
                w_in: Vec::new(),
                q,
                k,
                v,
                g,
                o,
                ga: (Vec::new(), Vec::new()),
                w_ff: Vec::new(),
                router: Vec::new(),
                bias: Vec::new(),
                pre: Vec::new(),
                w_pre: Vec::new(),
                gu: Vec::new(),
                down: Vec::new(),
                w_post_e: Vec::new(),
                post: Vec::new(),
                w_post_ff: Vec::new(),
                gf: (Vec::new(), Vec::new()),
                keys: Vec::new(),
                vals: Vec::new(),
            })
            .collect();
        RefMuse { c, embed: Vec::new(), head: Vec::new(), norm: Vec::new(), layers, pos: 0, margin: std::cell::Cell::new(f32::INFINITY) }
    }

    /// A model of MoE layers only, from each layer's router, choice bias, pre-expert projection and norm, experts' gate-up and down,
    /// post-expert norm and projection (the norms' weights as the checkpoint holds them).
    pub(crate) fn moe_only(c: ConfigMuse, w: Vec<[Vec<f32>; 8]>) -> RefMuse {
        let mut m = RefMuse::attention_only(c, w.iter().map(|_| Default::default()).collect());
        for (ly, [router, bias, pre, w_pre, gu, down, w_post_e, post]) in m.layers.iter_mut().zip(w) {
            (ly.router, ly.bias, ly.pre, ly.w_pre, ly.gu, ly.down, ly.w_post_e, ly.post) =
                (router, bias, pre, plus_one(&w_pre), gu, down, plus_one(&w_post_e), post);
        }
        m
    }

    pub(crate) fn moe_of(&self, l: usize, h: &[f32]) -> Vec<f32> {
        self.moe(l, h)
    }

    /// Layer `l`'s attention on its normed input at `self.pos`.
    pub(crate) fn attention_of(&mut self, l: usize, h: &[f32]) -> Vec<f32> {
        self.attention(l, h)
    }
}

/// A small Muse Spark of the release's layout in `out` (bf16, the routers fp32; random weights): four layers (the window's, every position's, the window's twice), a window of 32, 16 experts of which 4,
/// a latent of 512.
pub fn random(out: &Path) -> Result<()> {
    let text = r#"{"architectures": ["MuseSparkForConditionalGeneration"], "model_type": "muse_spark", "text_config": {"model_type": "muse_spark_text", "hidden_size": 1024, "num_hidden_layers": 4,
        "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 64, "intermediate_size": 2048, "moe_hidden_size": 512, "moe_intermediate_size": 512, "num_local_experts": 16,
        "num_experts_per_tok": 4, "num_shared_experts": 0, "vocab_size": 4096, "max_position_embeddings": 65536, "sliding_window": 32, "hidden_act": "silu", "attention_bias": false,
        "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "sliding_attention"], "layer_rope_theta": [500000.0, 0, 500000.0, 500000.0],
        "mlp_layer_types": ["sparse", "sparse", "sparse", "sparse"], "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"}, "rms_norm_eps": 1e-05, "post_norm_eps": 1e-08,
        "qk_scale_factor": 5.4730064863875, "output_multiplier": 0.5, "final_logit_softcapping": 20.0, "residual_gate_temperature": 0.3, "routed_scaling_factor": 1.0,
        "bos_token_id": 0, "eos_token_id": 1, "tie_word_embeddings": false}}"#;
    let c = ConfigMuse::parse(text, "random", None)?;
    let mut r = 0x2026_1018_0000_0003u64;
    let mut uni = move |a: f32, b: f32| {
        r ^= r << 13;
        r ^= r >> 7;
        r ^= r << 17;
        a + (b - a) * ((r >> 40) as f32 / (1u64 << 24) as f32)
    };
    let bfw = |x: f32| {
        let b = x.to_bits();
        ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
    };
    let mut t: Vec<(String, &str, Vec<u64>, Vec<u8>)> = Vec::new();
    let put =
        |t: &mut Vec<(String, &str, Vec<u64>, Vec<u8>)>, name: String, shape: Vec<u64>, lo: f32, hi: f32, f32_: bool, uni: &mut dyn FnMut(f32, f32) -> f32| {
            let n: u64 = shape.iter().product();
            let bytes =
                if f32_ { (0..n).flat_map(|_| uni(lo, hi).to_le_bytes()).collect() } else { (0..n).flat_map(|_| bfw(uni(lo, hi)).to_le_bytes()).collect() };
            t.push((name, if f32_ { "F32" } else { "BF16" }, shape, bytes));
        };
    let (d, nh, nk, hd, lat, inter, e, v) =
        (c.hidden as u64, c.heads as u64, c.kv_heads as u64, c.head_dim as u64, c.latent as u64, c.inter as u64, c.experts as u64, c.vocab as u64);
    let a = |cols: u64| (3.0 / cols as f32).sqrt();
    let p0 = c.prefix.clone();
    put(&mut t, format!("{p0}embed_tokens.weight"), vec![v, d], -1.0, 1.0, false, &mut uni);
    put(&mut t, format!("{p0}norm.weight"), vec![d], 0.9, 1.1, false, &mut uni);
    put(&mut t, "lm_head.weight".into(), vec![v, d], -a(d), a(d), false, &mut uni);
    for l in 0..c.layers {
        let p = format!("{p0}layers.{l}");
        for n in ["input_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"] {
            put(&mut t, format!("{p}.{n}.weight"), vec![d], -0.1, 0.1, false, &mut uni);
        }
        for n in ["post_attention_residual_gate", "post_feedforward_residual_gate"] {
            put(&mut t, format!("{p}.{n}.gate"), vec![d], -1.0, 1.0, false, &mut uni);
        }
        for (n, rows) in [("q_proj", nh * hd), ("k_proj", nk * hd), ("v_proj", nk * hd), ("gate_proj", nh * hd)] {
            put(&mut t, format!("{p}.self_attn.{n}.weight"), vec![rows, d], -a(d), a(d), false, &mut uni);
        }
        put(&mut t, format!("{p}.self_attn.o_proj.weight"), vec![d, nh * hd], -a(nh * hd), a(nh * hd), false, &mut uni);
        let m = format!("{p}.mlp");
        put(&mut t, format!("{m}.gate.weight"), vec![e, d], -a(d) * 4.0, a(d) * 4.0, true, &mut uni);
        put(&mut t, format!("{m}.gate.e_score_correction_bias"), vec![e], 0.0, 0.1, true, &mut uni);
        put(&mut t, format!("{m}.pre_expert_proj.weight"), vec![lat, d], -a(d), a(d), false, &mut uni);
        put(&mut t, format!("{m}.pre_expert_norm.weight"), vec![lat], -0.1, 0.1, false, &mut uni);
        put(&mut t, format!("{m}.experts.gate_up_proj"), vec![e, 2 * inter, lat], -a(lat), a(lat), false, &mut uni);
        put(&mut t, format!("{m}.experts.down_proj"), vec![e, lat, inter], -a(inter), a(inter), false, &mut uni);
        put(&mut t, format!("{m}.experts.post_expert_norm.weight"), vec![lat], -0.1, 0.1, false, &mut uni);
        put(&mut t, format!("{m}.post_expert_proj.weight"), vec![d, lat], -a(lat), a(lat), false, &mut uni);
    }
    io(std::fs::create_dir_all(out))?;
    io(std::fs::write(out.join("config.json"), text))?;
    let refs: Vec<(&str, &str, Vec<u64>, &[u8])> = t.iter().map(|(n, d, s, b)| (n.as_str(), *d, s.clone(), &b[..])).collect();
    crate::pack::write_safetensors(&out.join("model.safetensors"), &[("format", "pt".to_string())], &refs)?;
    println!("{}: a random Muse Spark, {} tensors, {:.0} MB", out.display(), t.len(), t.iter().map(|t| t.3.len()).sum::<usize>() as f64 / 1e6);
    Ok(())
}

/// `museref DIR [--tokens N]`: the reference's greedy continuation of a fixed prompt, its best logit a step.
pub fn cmd(a: &crate::args::Args) -> Result<()> {
    let dir = Path::new(a.need(1, "DIR")?);
    let n: usize = a.num("tokens", 64)?;
    let mut m = RefMuse::load(dir)?;
    let mut tok = 2u32;
    let t0 = std::time::Instant::now();
    for i in 0..n {
        let lg = m.step(tok);
        let (best, v) = lg.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b });
        println!("{i:4} token {tok:5} -> {best:5} ({v:.4})");
        tok = if i < 40 { (i as u32 * 37 + 11) % m.c.vocab as u32 } else { best as u32 };
    }
    println!("{n} tokens in {:.1} s", t0.elapsed().as_secs_f64());
    Ok(())
}

/// `musecheck DIR [--tokens N] [--ctx C] [--chunk K] [--no-graph] [--layout L]`: the dynamic engine against this reference on the same tokens (a fixed prompt, then the reference's choices): each step's
/// logits (both soft-capped), then the whole sequence as one prompt in chunks.
pub fn check(a: &crate::args::Args) -> Result<()> {
    let dir = Path::new(a.need(1, "DIR")?);
    let n: usize = a.num("tokens", 80)?;
    let o = crate::dynamic::Opts { ctx: a.num("ctx", 1024)?, chunk: a.num("chunk", 64)?, graph: !a.has("no-graph"), device: 0, layout: a.layout()? };
    let mut r = RefMuse::load(dir)?;
    let mut e = crate::dynamic::muse::load(dir, &o)?;
    println!("{}", e.describe());
    let best = |l: &[f32]| l.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b });
    let (mut toks, mut worst, mut agree, mut tok) = (Vec::new(), 0f32, 0usize, 2u32);
    for i in 0..n {
        toks.push(tok);
        let lr = r.step(tok);
        if i == 0 {
            e.prefill(&[tok])?;
        } else {
            e.step(tok)?;
        }
        let lg = e.logits()?;
        let (br, bg) = (best(&lr), best(&lg));
        let diff = lr.iter().zip(&lg).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        worst = worst.max(diff);
        agree += (br.0 == bg.0) as usize;
        if i < 6 || br.0 != bg.0 || diff > 0.1 {
            println!("{i:4} token {tok:5}: reference {:5} ({:.4}), engine {:5} ({:.4}), largest difference {diff:.4}", br.0, br.1, bg.0, bg.1);
        }
        tok = if i < 40 { (i as u32 * 37 + 11) % r.c.vocab as u32 } else { br.0 as u32 };
    }
    println!("decode: {agree} of {n} steps choose the reference's token; the largest logit difference {worst:.4}");
    e.reset()?;
    e.prefill(&toks)?;
    let lg = e.logits()?;
    let mut r2 = RefMuse::load(dir)?;
    let mut lr = Vec::new();
    for &t in &toks {
        lr = r2.step(t);
    }
    let diff = lr.iter().zip(&lg).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    println!("prompt of {} in chunks of {}: reference {} engine {}, largest logit difference {diff:.4}", toks.len(), o.chunk, best(&lr).0, best(&lg).0);
    Ok(())
}
