//! Each kind against the per-family references it replaces (`qwen35_ref.rs`, `kimi_ref.rs`, `muse_ref.rs`) on the same random weights.
//! Those round to bf16 where vLLM or transformers hold bf16, so the tolerance is `BF16_TOL`.

use super::testing::{assert_close, fill};
use super::{attention, bf16, linear, mla, Libm, Params, BF16_TOL};
use crate::runtime::kinds::*;
use glyd_gpu::safetensors::Checkpoint;

/// A checkpoint tensor as f32 (bf16 or f32): (shape, values).
fn tensor(ck: &Checkpoint, name: &str) -> (Vec<usize>, Vec<f32>) {
    let (_, t) = ck.get(name).unwrap_or_else(|| panic!("no tensor {name}"));
    let (dtype, shape) = (t.dtype.clone(), t.shape.iter().map(|&d| d as usize).collect::<Vec<_>>());
    let b = ck.read(name).unwrap();
    let v = match dtype.as_str() {
        "BF16" => b.chunks(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
        "F32" => b.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        d => panic!("{name}: {d}"),
    };
    (shape, v)
}

fn put(p: &mut Params, ck: &Checkpoint, role: &str, name: &str) {
    let (shape, v) = tensor(ck, name);
    p.put(role, &shape, v);
}

/// Gated DeltaNet's core against `qwen35_ref::mixer_step` (vLLM's decode step), on random products over a few tokens.
#[test]
fn gated_deltanet_against_qwen35_ref() {
    use crate::qwen35_ref as q;
    let d = q::Dims { nk: 2, nv: 4, dk: 16, dv: 8, kernel: 4 };
    let n = d.conv_dim();
    let l = Linear {
        kind: LinearKind::GatedDeltaNet,
        k_heads: 2,
        v_heads: 4,
        k_dim: 16,
        v_dim: 8,
        conv: 4,
        decay: Decay::Softplus,
        gate_act: Act::Silu,
        norm_eps: 1e-6,
        state_f32: true,
    };
    let bfv = |v: Vec<f32>| v.into_iter().map(bf16).collect::<Vec<_>>();
    let conv = bfv(fill("conv", n * 4, -0.5, 0.5));
    let (a_log, dt_bias, norm) = (fill("a_log", 4, -1.0, 0.5), fill("dt_bias", 4, -0.5, 0.5), fill("norm", 8, 0.8, 1.2));
    let lp = q::LinearParams { conv: conv.clone(), a_log: a_log.clone(), dt_bias: dt_bias.clone(), norm: norm.clone(), eps: 1e-6 };
    let mut p = Params::new();
    p.put("conv", &[n, 4], conv);
    p.put("a_log", &[4], a_log);
    p.put("dt_bias", &[4], dt_bias);
    p.put("norm", &[8], norm);
    let (mut theirs, mut ours) = (q::LinearState::new(&d), linear::LinearState::new(&l));
    let (mut want, mut got) = (Vec::new(), Vec::new());
    for t in 0..6 {
        let g = |what: &str, k: usize, lo: f32, hi: f32| bfv(fill(&format!("{what}{t}"), k, lo, hi));
        let (qkv, z, b, a) = (g("qkv", n, -2.0, 2.0), g("z", 32, -2.0, 2.0), g("b", 4, -2.0, 2.0), g("a", 4, -2.0, 2.0));
        want.extend(q::mixer_step(&d, &lp, &mut theirs, &qkv, &z, &b, &a));
        got.extend(linear::core(&l, &p, &mut ours, &qkv, &b, &a, &z).unwrap());
    }
    assert_close(&got, &want, BF16_TOL, "GDN against qwen35_ref");
}

/// Kimi delta attention and Kimi-K3's gated NoPE MLA against `kimi_ref` on its random model (dense MLPs, so it stays small).
#[test]
fn kimi_k3_mixers_against_kimi_ref() {
    use crate::kimi_ref::{random, RefKimi};
    let dir = std::env::temp_dir().join(format!("glyd-runtime-kimi-{}", std::process::id()));
    random(&dir, 8, true, 8).unwrap();
    let mut theirs = RefKimi::load(&dir).unwrap();
    let ck = Checkpoint::open(&dir).unwrap();
    let hidden = theirs.c.hidden;
    let kda = Linear {
        kind: LinearKind::KimiDelta,
        k_heads: 2,
        v_heads: 2,
        k_dim: 128,
        v_dim: 128,
        conv: 4,
        decay: Decay::Bounded(-5.0),
        gate_act: Act::Sigmoid,
        norm_eps: 1e-5,
        state_f32: true,
    };
    let twice = Norm { rounding: Rounding::Twice, ..Norm::rms(1e-6) };
    let m = Mla {
        heads: 2,
        q_lora: Some(128),
        kv_lora: 128,
        nope: 128,
        rope_dim: 64,
        v_dim: 128,
        rope: None,
        q_norm: twice,
        kv_norm: twice,
        gate: Some(OutGate { act: Act::Sigmoid, from: GateFrom::Own, norm: None }),
        scale: 192f32.lpowf(-0.5),
        absorbed: false,
        indexer: None,
    };
    for l in [0usize, 3] {
        let a = format!("language_model.model.layers.{l}.self_attn");
        let mut p = Params::new();
        let is_mla = l == 3;
        if is_mla {
            for (role, n) in [("q_a", "q_a_proj"), ("q_b", "q_b_proj"), ("kv_a", "kv_a_proj_with_mqa"), ("kv_b", "kv_b_proj"), ("g", "g_proj"), ("o", "o_proj")]
            {
                put(&mut p, &ck, role, &format!("{a}.{n}.weight"));
            }
            put(&mut p, &ck, "q_a_norm", &format!("{a}.q_a_layernorm.weight"));
            put(&mut p, &ck, "kv_a_norm", &format!("{a}.kv_a_layernorm.weight"));
        } else {
            for (role, n) in [
                ("q", "q_proj"),
                ("k", "k_proj"),
                ("v", "v_proj"),
                ("beta", "b_proj"),
                ("decay_a", "f_a_proj"),
                ("decay_b", "f_b_proj"),
                ("gate", "g_proj"),
                ("o", "o_proj"),
            ] {
                put(&mut p, &ck, role, &format!("{a}.{n}.weight"));
            }
            let conv: Vec<f32> = ["q", "k", "v"].iter().flat_map(|n| tensor(&ck, &format!("{a}.{n}_conv1d.weight")).1).collect();
            p.put("conv", &[conv.len() / 4, 4], conv);
            put(&mut p, &ck, "a_log", &format!("{a}.A_log"));
            put(&mut p, &ck, "dt_bias", &format!("{a}.dt_bias"));
            put(&mut p, &ck, "norm", &format!("{a}.o_norm.weight"));
        }
        let (mut st, mut cache) = (linear::LinearState::new(&kda), mla::LatentCache::default());
        let (mut want, mut got) = (Vec::new(), Vec::new());
        for t in 0..5u32 {
            let x: Vec<f32> = fill(&format!("x{t}"), hidden, -1.0, 1.0).into_iter().map(bf16).collect();
            want.extend(theirs.mixer_of(l, &x));
            got.extend(if is_mla { mla::step(&m, &p, &mut cache, &x, t).unwrap() } else { linear::step(&kda, &p, &mut st, &x).unwrap() });
        }
        assert_close(&got, &want, BF16_TOL, if is_mla { "Kimi-K3 MLA against kimi_ref" } else { "KDA against kimi_ref" });
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Muse Spark's attention (weightless q/k norm, gated output with a per-head norm, rope and a window on sliding layers only) against
/// `muse_ref`, a sliding and a global layer, past the window.
#[test]
fn muse_attention_against_muse_ref() {
    use crate::muse::ConfigMuse;
    use crate::muse_ref::RefMuse;
    let text = r#"{"architectures": ["MuseSparkForConditionalGeneration"], "model_type": "muse_spark", "text_config": {"model_type": "muse_spark_text", "hidden_size": 128, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32, "intermediate_size": 256, "moe_hidden_size": 64, "moe_intermediate_size": 64, "num_local_experts": 4,
        "num_experts_per_tok": 2, "num_shared_experts": 0, "vocab_size": 256, "max_position_embeddings": 4096, "sliding_window": 3, "hidden_act": "silu", "attention_bias": false,
        "layer_types": ["sliding_attention", "full_attention"], "layer_rope_theta": [500000.0, 0],
        "mlp_layer_types": ["sparse", "sparse"], "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"}, "rms_norm_eps": 1e-05, "post_norm_eps": 1e-08,
        "qk_scale_factor": 5.4730064863875, "output_multiplier": 0.5, "final_logit_softcapping": 20.0, "residual_gate_temperature": 0.3, "routed_scaling_factor": 1.0,
        "bos_token_id": 0, "eos_token_id": 1, "tie_word_embeddings": false}}"#;
    let c = ConfigMuse::parse(text, "test", None).unwrap();
    let (d, nh, nk, hd) = (c.hidden, c.heads, c.kv_heads, c.head_dim);
    let a = (3.0 / d as f32).sqrt();
    let bfv = |v: Vec<f32>| v.into_iter().map(bf16).collect::<Vec<_>>();
    let w: Vec<[Vec<f32>; 5]> = (0..2)
        .map(|l| {
            let g = |n: &str, k: usize, s: f32| bfv(fill(&format!("{l}.{n}"), k, -s, s));
            [
                g("q", nh * hd * d, a),
                g("k", nk * hd * d, a),
                g("v", nk * hd * d, a),
                g("g", nh * hd * d, a),
                g("o", d * nh * hd, (3.0 / (nh * hd) as f32).sqrt()),
            ]
        })
        .collect();
    let mut theirs = RefMuse::attention_only(c.clone(), w.clone());
    let bare = Norm::rms_bare(c.eps);
    for l in 0..2 {
        let sliding = c.sliding[l];
        let at = Attn {
            heads: nh,
            kv_heads: nk,
            head_dim: hd,
            v_head_dim: hd,
            window: sliding.then_some(c.window),
            sinks: false,
            rope: sliding.then(|| Rope::plain(c.rope_theta, hd)),
            q_norm: Some(bare),
            k_norm: Some(bare),
            v_norm: None,
            gate: Some(OutGate { act: Act::Sigmoid, from: GateFrom::Own, norm: Some(bare) }),
            softcap: None,
            bias: false,
            scale: c.qk_scale / (hd as f32).sqrt(),
            k_eq_v: false,
            indexer: None,
            rounding: AttnRounding::Sdpa,
            kv_from: None,
        };
        let mut p = Params::new();
        for (i, (role, shape)) in
            [("q", [nh * hd, d]), ("k", [nk * hd, d]), ("v", [nk * hd, d]), ("g", [nh * hd, d]), ("o", [d, nh * hd])].into_iter().enumerate()
        {
            p.put(role, &shape, w[l][i].clone());
        }
        let mut cache = attention::KvCache::default();
        let (mut want, mut got) = (Vec::new(), Vec::new());
        for t in 0..6u32 {
            let h = bfv(fill(&format!("h{t}"), d, -1.0, 1.0));
            theirs.pos = t as usize;
            want.extend(theirs.attention_of(l, &h));
            got.extend(attention::step(&at, &p, &mut cache, &h, t).unwrap());
        }
        assert_close(&got, &want, BF16_TOL, &format!("Muse attention layer {l} against muse_ref"));
    }
}

/// Kimi-K3's FFNs against `kimi_ref`: the dense situ MLP of layer 0 and the latent MoE of layer 1 (sigmoid + choice bias, top-4 of 16,
/// renormed, the experts MXFP4 in a latent of 128, the summed latent normed with two roundings, a shared expert).
#[test]
fn kimi_k3_ffns_against_kimi_ref() {
    use crate::kimi_ref::{mxfp4, random, RefKimi};
    let dir = std::env::temp_dir().join(format!("glyd-runtime-kimi-moe-{}", std::process::id()));
    random(&dir, 1, false, 2).unwrap();
    let theirs = RefKimi::load(&dir).unwrap();
    let ck = Checkpoint::open(&dir).unwrap();
    let c = &theirs.c;
    let situ = Mlp { act: Act::Situ { beta: c.situ.0, linear: c.situ.1 }, gated: true, bias: false, clamp: None };
    let q = |l: usize| format!("language_model.model.layers.{l}");
    // the dense layer
    let mut p = Params::new();
    for r in ["gate", "up", "down"] {
        put(&mut p, &ck, r, &format!("{}.mlp.{r}_proj.weight", q(0)));
    }
    // the MoE layer
    let f = format!("{}.block_sparse_moe", q(1));
    let mut pm = Params::new();
    put(&mut pm, &ck, "router", &format!("{f}.gate.weight"));
    put(&mut pm, &ck, "choice_bias", &format!("{f}.gate.e_score_correction_bias"));
    put(&mut pm, &ck, "latent_down", &format!("{f}.routed_expert_down_proj.weight"));
    put(&mut pm, &ck, "latent_up", &format!("{f}.routed_expert_up_proj.weight"));
    put(&mut pm, &ck, "latent_norm_out", &format!("{f}.routed_expert_norm.weight"));
    for r in ["gate", "up", "down"] {
        put(&mut pm, &ck, &format!("shared_{r}"), &format!("{f}.shared_experts.{r}_proj.weight"));
    }
    let fp4 = |n: &str| -> (usize, usize, Vec<f32>) {
        let sh = ck.get(&format!("{n}.weight_packed")).unwrap().1.shape.clone();
        let (rows, cols) = (sh[0] as usize, sh[1] as usize * 2);
        (rows, cols, mxfp4(&ck.read(&format!("{n}.weight_packed")).unwrap(), &ck.read(&format!("{n}.weight_scale")).unwrap(), rows, cols).unwrap())
    };
    let e = &c.experts;
    let (mut gu, mut down) = (Vec::new(), Vec::new());
    let (mut inter, mut lat) = (0, 0);
    for x in 0..e.experts {
        let (i, l, w1) = fp4(&format!("{f}.experts.{x}.w1"));
        (inter, lat) = (i, l);
        gu.extend(w1);
        gu.extend(fp4(&format!("{f}.experts.{x}.w3")).2);
        down.extend(fp4(&format!("{f}.experts.{x}.w2")).2);
    }
    pm.put("experts_gate_up", &[e.experts, 2 * inter, lat], gu);
    pm.put("experts_down", &[e.experts, lat, inter], down);
    let m = Moe {
        router: Router {
            score: RouterScore::Sigmoid,
            logit_scale: 1.0,
            bias: true,
            groups: None,
            top_k: e.top_k,
            renorm: e.renorm,
            scale: e.scale,
            hash: false,
        },
        experts: e.experts,
        expert: situ,
        shared: Some(situ),
        shared_gate: false,
        latent: Some(Latent { dim: lat, norm_in: None, expert_norm: None, norm_out: Some(Norm { rounding: Rounding::Twice, ..Norm::rms(c.eps) }) }),
    };
    let (mut want, mut got, mut want_m, mut got_m) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for t in 0..4 {
        let x: Vec<f32> = fill(&format!("x{t}"), c.hidden, -1.0, 1.0).into_iter().map(bf16).collect();
        want.extend(theirs.ffn_of(0, &x));
        got.extend(super::ffn::mlp(&situ, &p, "", &x).unwrap());
        want_m.extend(theirs.ffn_of(1, &x));
        got_m.extend(super::ffn::moe(&m, &pm, &x, 0).unwrap());
    }
    assert_close(&got, &want, BF16_TOL, "Kimi-K3 situ MLP against kimi_ref");
    assert_close(&got_m, &want_m, BF16_TOL, "Kimi-K3 latent MoE against kimi_ref");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Muse Spark's latent MoE against `muse_ref`: sigmoid of the logits times the output multiplier with a choice bias, a (1+w)-normed
/// latent input, each expert's output normed with its input scaled by (1+w), no shared expert.
#[test]
fn muse_moe_against_muse_ref() {
    use crate::muse::ConfigMuse;
    use crate::muse_ref::RefMuse;
    let text = r#"{"architectures": ["MuseSparkForConditionalGeneration"], "model_type": "muse_spark", "text_config": {"model_type": "muse_spark_text", "hidden_size": 128, "num_hidden_layers": 1,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32, "intermediate_size": 256, "moe_hidden_size": 64, "moe_intermediate_size": 48, "num_local_experts": 8,
        "num_experts_per_tok": 3, "num_shared_experts": 0, "vocab_size": 256, "max_position_embeddings": 4096, "sliding_window": 3, "hidden_act": "silu", "attention_bias": false,
        "layer_types": ["full_attention"], "layer_rope_theta": [0],
        "mlp_layer_types": ["sparse"], "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"}, "rms_norm_eps": 1e-05, "post_norm_eps": 1e-08,
        "qk_scale_factor": 5.4730064863875, "output_multiplier": 0.5, "final_logit_softcapping": 20.0, "residual_gate_temperature": 0.3, "routed_scaling_factor": 1.25,
        "bos_token_id": 0, "eos_token_id": 1, "tie_word_embeddings": false}}"#;
    let c = ConfigMuse::parse(text, "test", None).unwrap();
    let (d, lat, inter, ne) = (c.hidden, c.latent, c.inter, c.experts);
    let bfv = |v: Vec<f32>| v.into_iter().map(bf16).collect::<Vec<_>>();
    let g = |n: &str, k: usize, lo: f32, hi: f32| bfv(fill(n, k, lo, hi));
    let a = |cols: usize| (3.0 / cols as f32).sqrt();
    let w = [
        fill("router", ne * d, -4.0 * a(d), 4.0 * a(d)),
        fill("bias", ne, 0.0, 0.1),
        g("pre", lat * d, -a(d), a(d)),
        g("w_pre", lat, -0.1, 0.1),
        g("gu", ne * 2 * inter * lat, -a(lat), a(lat)),
        g("down", ne * lat * inter, -a(inter), a(inter)),
        g("w_post_e", lat, -0.1, 0.1),
        g("post", d * lat, -a(lat), a(lat)),
    ];
    let theirs = RefMuse::moe_only(c.clone(), vec![w.clone()]);
    let mut p = Params::new();
    for (i, (role, shape)) in [
        ("router", vec![ne, d]),
        ("choice_bias", vec![ne]),
        ("latent_down", vec![lat, d]),
        ("latent_norm_in", vec![lat]),
        ("experts_gate_up", vec![ne, 2 * inter, lat]),
        ("experts_down", vec![ne, lat, inter]),
        ("expert_norm", vec![lat]),
        ("latent_up", vec![d, lat]),
    ]
    .into_iter()
    .enumerate()
    {
        p.put(role, &shape, w[i].clone());
    }
    let m = Moe {
        router: Router {
            score: RouterScore::Sigmoid,
            logit_scale: c.out_mult,
            bias: true,
            groups: None,
            top_k: c.top_k,
            renorm: true,
            scale: c.route_scale,
            hash: false,
        },
        experts: ne,
        expert: Mlp { act: Act::Silu, gated: true, bias: false, clamp: None },
        shared: None,
        shared_gate: false,
        latent: Some(Latent {
            dim: lat,
            norm_in: Some(Norm::rms_plus_one(c.eps)),
            expert_norm: Some(Norm { weight: NormWeight::InputPlusOne, ..Norm::rms(c.post_eps) }),
            norm_out: None,
        }),
    };
    let (mut want, mut got) = (Vec::new(), Vec::new());
    for t in 0..4 {
        let h = g(&format!("h{t}"), d, -1.0, 1.0);
        want.extend(theirs.moe_of(0, &h));
        got.extend(super::ffn::moe(&m, &p, &h, 0).unwrap());
    }
    assert_close(&got, &want, BF16_TOL, "Muse latent MoE against muse_ref");
}

/// Kimi-K3's depth mix against `kimi_ref`'s, over one to five sources, at Kimi-K3's eps and another; the scores' norm takes the model's eps (vLLM's score
/// norm's `variance_epsilon`, the config's rms_norm_eps), not a constant: on sources of a small RMS, an eps of 1e-2 gives another mix than 1e-5 in both, each
/// the softmax over the sources of u . v / sqrt(mean v^2 + eps) weighing their sum.
#[test]
fn depth_mix_against_kimi_ref() {
    use crate::kimi_ref::RefKimi;
    let d = 64;
    let bfv = |v: Vec<f32>| v.into_iter().map(bf16).collect::<Vec<_>>();
    let (wn, wp) = (bfv(fill("w_n", d, 0.5, 1.5)), bfv(fill("w_p", d, -1.0, 1.0)));
    let src: Vec<Vec<f32>> = (0..5).map(|i| bfv(fill(&format!("src{i}"), d, -2.0, 2.0))).collect();
    for n in 1..=5 {
        let s: Vec<&[f32]> = src[..n].iter().map(|v| &v[..]).collect();
        for eps in [1e-5, 1e-2] {
            assert_close(&super::depth_mix(&s, &wn, &wp, eps), &RefKimi::mix_of(&s, &wn, &wp, eps), BF16_TOL, &format!("depth mix of {n} at {eps}"));
        }
    }
    let small: Vec<Vec<f32>> = src.iter().enumerate().map(|(i, v)| bfv(v.iter().map(|x| x * 0.02 * (i + 1) as f32).collect())).collect();
    let s: Vec<&[f32]> = small.iter().map(|v| &v[..]).collect();
    let by_hand = |eps: f64| -> Vec<f32> {
        let sc: Vec<f64> = s
            .iter()
            .map(|v| {
                let ms = v.iter().map(|&x| x as f64 * x as f64).sum::<f64>() / d as f64;
                (0..d).map(|i| wn[i] as f64 * wp[i] as f64 * v[i] as f64).sum::<f64>() / (ms + eps).sqrt()
            })
            .collect();
        let m = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let z: f64 = sc.iter().map(|x| (x - m).exp()).sum();
        (0..d).map(|i| s.iter().zip(&sc).map(|(v, x)| (x - m).exp() / z * v[i] as f64).sum::<f64>() as f32).collect()
    };
    let (lo, hi) = (super::depth_mix(&s, &wn, &wp, 1e-5), super::depth_mix(&s, &wn, &wp, 1e-2));
    assert_close(&lo, &by_hand(1e-5), 1e-5, "depth mix at 1e-5 by hand");
    assert_close(&hi, &by_hand(1e-2), 1e-5, "depth mix at 1e-2 by hand");
    assert_close(&RefKimi::mix_of(&s, &wn, &wp, 1e-2), &by_hand(1e-2), BF16_TOL, "kimi_ref's depth mix at 1e-2 by hand");
    assert!(super::testing::rel(&lo, &hi) > 1e-2, "the eps moves the mix of small sources: {:.3e}", super::testing::rel(&lo, &hi));
}

/// A checkpoint as the reference model's tensor source (`model::checkpoint_tensor`).
fn checkpoint_source(ck: &Checkpoint) -> impl Fn(&str) -> Option<(Vec<usize>, Vec<f32>)> + '_ {
    move |name: &str| super::model::checkpoint_tensor(ck, name)
}

/// A whole model from its config.json through the description builder against the per-family reference's logits. Those references
/// round to bf16 after nearly every operation; over a few layers the hidden state drifts by about 2% from the f32 kinds', which moves
/// a router score by about 0.01. Where the reference's closest top-k choice in a step is nearer than `MARGIN`, the two can pick
/// different experts, and attention carries that position's difference to every later one (Kimi-K3's random model: margins of 0.005
/// and 0.014 gave 8% and 5%, margins of 0.024 and 0.036 gave 0.4% and 2%). So the prompt is chosen by the reference alone: the first of
/// a fixed list of three-token prompts whose every step's margin clears `MARGIN`; then every position is bounded by `WHOLE_TOL`. With the
/// routing clear, the bf16 roundings of 4-8 layers leave 3-5% at the third position (measured: Kimi-K3 5.0e-2, Muse Spark 5.3e-2); a
/// wrong norm, residual or routing rule is off by tens of percent (a flipped expert choice alone gave 40-60%).
const WHOLE_TOL: f32 = 8e-2;
const MARGIN: f32 = 0.02;

/// `theirs` makes a fresh reference: its step gives a token's logits and the step's routing margin. The search starts at prompt `first`
/// of the list (the one the reference picked when the test was written; loading a reference is the test's cost) and goes on from there.
fn whole_model(dir: &std::path::Path, theirs: &mut dyn FnMut() -> Box<dyn FnMut(u32) -> (Vec<f32>, f32)>, vocab: usize, first: u32, tol: f32, what: &str) {
    use crate::runtime::{describe::describe, reference::model::RefModel};
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let d = describe(&cfg).unwrap_or_else(|e| panic!("{what}: {}", e.msg));
    let ck = Checkpoint::open(dir).unwrap();
    let src = checkpoint_source(&ck);
    for c in first..first + 64 {
        let toks: Vec<u32> = (0..3).map(|i| ((c * 977 + i * 131 + 7) as usize % vocab) as u32).collect();
        let mut step = theirs();
        let steps: Vec<(Vec<f32>, f32)> = toks.iter().map(|&t| step(t)).collect();
        if steps.iter().any(|s| s.1 < MARGIN) {
            continue;
        }
        let mut m = RefModel::load(&d, &src).unwrap_or_else(|e| panic!("{what}: {}", e.msg));
        let ours = m.forward(&toks, 0).unwrap();
        for (i, (want, _)) in steps.iter().enumerate() {
            assert_close(&ours[i], want, tol, &format!("{what}: prompt {toks:?}, position {i}"));
        }
        return;
    }
    panic!("{what}: no prompt of the list routes clear of rounding");
}

#[test]
fn kimi_k3_model_described() {
    use crate::kimi_ref::{random, RefKimi};
    let dir = std::env::temp_dir().join(format!("glyd-runtime-kimi-model-{}", std::process::id()));
    random(&dir, 1, true, 8).unwrap();
    let vocab = RefKimi::load(&dir).unwrap().c.vocab;
    whole_model(
        &dir,
        &mut || {
            let d2 = dir.clone();
            let mut r = RefKimi::load(&d2).unwrap();
            Box::new(move |t| {
                let l = r.step(t);
                (l, r.margin.get())
            })
        },
        vocab,
        0,
        WHOLE_TOL,
        "Kimi-K3",
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Muse Spark's whole random model, as `whole_model` holds Kimi-K3's, but on the same expert choices: the description's, taken by the per-family
/// reference (`RefMuse::take`) in the place of its routers' own, so no prompt need route clear of rounding (the case's model has routers whose
/// near ties the two references' roundings take either way).
#[test]
fn muse_spark_model_described() {
    use crate::muse_ref::{random, RefMuse};
    use crate::runtime::kinds::{Body, Ffn};
    use crate::runtime::{describe::describe, reference::model::RefModel};
    let dir = std::env::temp_dir().join(format!("glyd-runtime-muse-model-{}", std::process::id()));
    random(&dir).unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let d = describe(&cfg).unwrap_or_else(|e| panic!("{}", e.msg));
    let ck = Checkpoint::open(&dir).unwrap();
    let mut m = RefModel::load(&d, &checkpoint_source(&ck)).unwrap_or_else(|e| panic!("{}", e.msg));
    let moe: Vec<usize> = (0..m.layers.len()).filter(|&b| matches!(m.layers[b].block.body, Body::Ffn(Ffn::Moe(_)))).collect();
    moe.iter().for_each(|&b| m.layers[b].routed = Some(Vec::new()));
    let toks = [7u32, 1108, 2209];
    let ours = m.forward(&toks, 0).unwrap();
    let mut r = RefMuse::load(&dir).unwrap();
    *r.take.borrow_mut() = moe.iter().map(|&b| m.layers[b].routed.take().unwrap().into_iter().map(|c| c.ids).collect()).collect();
    for (i, &t) in toks.iter().enumerate() {
        assert_close(&ours[i], &r.step(t), WHOLE_TOL, &format!("Muse Spark: prompt {toks:?}, position {i}, on the same expert choices"));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// DeepSeek-V4's release format: the random checkpoint of `dsv4_ref` (the release's names, FP8 matrices and FP4 experts with their
/// scales, a ratio of 4 being the overlapping windows with the indexer) is named exactly by the description, and gives `RefV4`'s logits
/// (its simulated quantizations off) on prompts whose expert and indexer choices are clear of rounding.
#[test]
fn deepseek_v4_release_described() {
    use crate::dsv4_ref::{random, RefV4};
    use crate::runtime::describe::describe;
    use std::collections::BTreeSet;
    let dir = std::env::temp_dir().join(format!("glyd-runtime-dsv4-model-{}", std::process::id()));
    random(&dir, None).unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let d = describe(&cfg).unwrap_or_else(|e| panic!("{}", e.msg));
    let ck = Checkpoint::open(&dir).unwrap();
    let names: BTreeSet<String> = ck.names().map(String::from).collect();
    assert_eq!(d.names(), names, "the weight map");
    let vocab = d.desc.vocab;
    whole_model(
        &dir,
        &mut || {
            let d2 = dir.clone();
            let mut r = RefV4::load(&d2, false).unwrap();
            Box::new(move |t| {
                let l = r.step(t);
                let (me, mi) = r.margin.get();
                (l, me.min(mi))
            })
        },
        vocab,
        15,
        WHOLE_TOL,
        "DeepSeek-V4",
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Nemotron 3's description on the reference (Mamba-2, attention with no position embedding, the mixture of experts with a shared expert on the full vector, a layer a block)
/// against the logits the engine recorded for the same random model on an RTX 4080 SUPER at the lossless level (tests/logits/nemotron-h.sm89-geforce-80sm.json.gz; no other implementation
/// made them). The weight map names the checkpoint's tensors but the draft head's, and every row (the prompt's, each decode step's, the long prompt's) is the engine's within
/// what its bf16 values explain: the same best token wherever it has a clear lead.
///
/// Not in the default run: 186 tokens of a model 2,688 wide take minutes on an unoptimized build (`cargo test --release -- --ignored nemotron_model`, 30 s). What runs
/// everywhere is the fixture this reference recorded (tests/logits/nemotron-h.json.gz), which the GPU job holds the engine to.
#[test]
#[ignore = "minutes on an unoptimized build: cargo test --release -- --ignored nemotron_model"]
fn nemotron_model_described_against_the_engines_logits() {
    use crate::logits::{compare, Bound, Fixture};
    use crate::runtime::{describe::describe, reference::model::RefModel};
    use std::collections::BTreeSet;
    let dir = std::env::temp_dir().join(format!("glyd-runtime-nemotron-model-{}", std::process::id()));
    crate::kcheck_nh::random(&dir, 0, 16).unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let d = describe(&cfg).unwrap_or_else(|e| panic!("{}", e.msg));
    let ck = Checkpoint::open(&dir).unwrap();
    let names: BTreeSet<String> = ck.names().filter(|n| !n.starts_with("mtp.")).map(String::from).collect();
    assert_eq!(d.names(), names, "the weight map");
    let fx = Fixture::read(&std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/logits/nemotron-h.sm89-geforce-80sm.json.gz")).unwrap();
    let src = checkpoint_source(&ck);
    let t0 = std::time::Instant::now();
    let mut m = RefModel::load(&d, &src).unwrap_or_else(|e| panic!("{}", e.msg));
    let mut got = vec![m.forward(&fx.prompt, 0).unwrap().pop().unwrap()];
    for (i, &t) in fx.decode.iter().enumerate() {
        got.push(m.forward(&[t], (fx.prompt.len() + i) as u32).unwrap().pop().unwrap());
    }
    let mut m = RefModel::load(&d, &src).unwrap();
    got.push(m.forward(&fx.long, 0).unwrap().pop().unwrap());
    let v = compare(&fx, &Bound::Kl { mean: 1e-4, max: 5e-4, top1: 1.0 }, &got);
    eprintln!(
        "nemotron on the reference: {} rows in {:.1} s, KL mean {:.3e} max {:.3e}, best token {}/{}, largest logit difference {:.4}",
        v.rows,
        t0.elapsed().as_secs_f64(),
        v.mean_kl,
        v.max_kl,
        v.top1.0,
        v.top1.1,
        v.max_dlogit
    );
    let _ = std::fs::remove_dir_all(&dir);
    assert!(v.ok, "{:?}", v.notes);
}

/// The Gated DeltaNet bodies of the `qwen4x-s2` case's draw against themselves with the values the GPU stores in bf16 rounded (the products q, k, v, the
/// gates', and the core's output into `o`), on 32 sequences of a prompt of 8 rows at random (normed) then 2 decode steps, measured as `kindcheck` measures
/// a body (the prompt's rows together, a step's row alone): each at most half the kind's bound (`kindcheck::bound`), so that the kernels have the other
/// half. Drawn with dt_bias about N(0, 1) the state forgot a token in about a step, a step's output was a few query-key products near 0 for random
/// queries and keys, and a last place of q or k moved a step's row by up to 4.3e-2, past the bound of 3e-2 itself (`kindcheck qwen4x-s2 --prompt 8 --steps 2`
/// on an A10: 3.04e-2); drawn about N(-4.6, 1), as the family initializes it, at most 7.6e-3 (the prompt's rows 5.3e-3 to 7.1e-3 on both draws).
#[test]
fn qwen4x_s2_deltanet_floor() {
    use crate::dynamic::kindcheck::{bound, Class};
    use crate::runtime::reference::model::{checkpoint_tensor, RefModel};
    let dir = std::env::temp_dir().join(format!("glyd-deltanet-floor-{}", std::process::id()));
    crate::logits::qwen4x_s2_random(&dir).unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let d = crate::runtime::describe::describe(&cfg).unwrap();
    let ck = Checkpoint::open(&dir).unwrap();
    let m = RefModel::load(&d, &|n: &str| checkpoint_tensor(&ck, n)).unwrap();
    let rel = |a: f32, b: f32| a / b.max(1e-30);
    let mut bodies = 0;
    for (bi, b) in m.layers.iter().enumerate() {
        let Body::Mixer(Mixer::Linear(l)) = &b.block.body else { continue };
        let p = &b.p;
        let (mut prompt, mut steps) = (0f32, 0f32);
        for seq in 0..32 {
            let (mut s0, mut s1) = (linear::LinearState::new(l), linear::LinearState::new(l));
            let (mut pd, mut pm) = (0f32, 0f32);
            for t in 0..10 {
                let mut h = fill(&format!("deltanet floor {seq} {t}"), b.hidden, -1.0, 1.0);
                let rms = (h.iter().map(|x| x * x).sum::<f32>() / b.hidden as f32).sqrt();
                h.iter_mut().for_each(|x| *x = bf16(*x / rms));
                let want = linear::step(l, p, &mut s0, &h).unwrap();
                let prod = |r: &str| -> Vec<f32> { p.prod(r, &h).unwrap().into_iter().map(bf16).collect() };
                let qkv = if p.has_prod("qkv") { prod("qkv") } else { [prod("q"), prod("k"), prod("v")].concat() };
                let o = linear::core(l, p, &mut s1, &qkv, &prod("beta"), &prod("decay"), &prod("gate")).unwrap();
                let got = p.prod("o", &o.into_iter().map(bf16).collect::<Vec<_>>()).unwrap();
                let diff = got.iter().zip(&want).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
                let most = want.iter().fold(0f32, |m, y| m.max(y.abs()));
                if t < 8 {
                    (pd, pm) = (pd.max(diff), pm.max(most));
                } else {
                    steps = steps.max(rel(diff, most));
                }
            }
            prompt = prompt.max(rel(pd, pm));
        }
        println!("qwen4x-s2 block {bi}, Gated DeltaNet: the prompt's rows {prompt:.2e}, a step's row {steps:.2e}");
        let half = bound(Class::DeltaNet) / 2.0;
        assert!(prompt <= half && steps <= half, "block {bi}: the draw's floor (prompt {prompt:e}, steps {steps:e}) past half the kind's bound");
        bodies += 1;
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(bodies > 0, "no Gated DeltaNet body in the qwen4x-s2 draw");
}
