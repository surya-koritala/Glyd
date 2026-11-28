//! Each kind against transformers in float32, on the fixtures `tests/runtime/record.py` writes: a recent model's module at reduced widths,
//! a causal prompt of a few tokens, the same generated weights on both sides.

use super::testing::{assert_close, Fixture};
use super::{attention, ffn, linear, mla, Libm, Params, HF_TOL};
use crate::runtime::kinds::*;

/// The fixture's tensors under their roles.
fn params(f: &Fixture, map: &[(&str, &str)]) -> Params {
    let mut p = Params::new();
    for &(role, name) in map {
        let (shape, v) = f.tensor(name);
        // a conv1d weight [channels, 1, taps] is [channels, taps]
        let shape: Vec<usize> = shape.into_iter().filter(|&d| d != 1 || role != "conv").collect();
        p.put(role, &shape, v);
    }
    p
}

/// The kind on the fixture's prompt, a token at a time, against its output.
fn check(f: &Fixture, what: &str, mut step: impl FnMut(&[f32], u32) -> Vec<f32>) {
    let d = f.usize("hidden");
    let x = f.input();
    let got: Vec<f32> = x.chunks(d).zip(f.positions()).flat_map(|(h, pos)| step(h, pos)).collect();
    assert_close(&got, &f.f32s("output"), HF_TOL, what);
}

fn attn(heads: usize, kv_heads: usize, head_dim: usize, rope: Option<Rope>) -> Attn {
    Attn {
        heads,
        kv_heads,
        head_dim,
        v_head_dim: head_dim,
        window: None,
        sinks: false,
        rope,
        q_norm: None,
        k_norm: None,
        v_norm: None,
        gate: None,
        softcap: None,
        bias: false,
        scale: (head_dim as f32).lpowf(-0.5),
        k_eq_v: false,
        indexer: None,
        rounding: AttnRounding::F32,
        kv_from: None,
        v_scale: 1.0,
    }
}

const QKVO: [(&str, &str); 4] = [("q", "q_proj.weight"), ("k", "k_proj.weight"), ("v", "v_proj.weight"), ("o", "o_proj.weight")];

#[test]
fn qwen38_gated_attention() {
    let f = Fixture::load("qwen38_attention");
    let a = Attn {
        q_norm: Some(Norm::rms_plus_one(1e-6)),
        k_norm: Some(Norm::rms_plus_one(1e-6)),
        gate: Some(OutGate { act: Act::Sigmoid, from: GateFrom::InQuery, norm: None }),
        ..attn(4, 2, 32, Some(Rope::plain(1e7, 8)))
    };
    let mut map = QKVO.to_vec();
    map.extend([("q_norm", "q_norm.weight"), ("k_norm", "k_norm.weight")]);
    let p = params(&f, &map);
    let mut c = attention::KvCache::default();
    check(&f, "Qwen3.8 attention", |h, pos| attention::step(&a, &p, &mut c, h, pos).unwrap());
}

#[test]
fn gemma4_sliding_attention() {
    let f = Fixture::load("gemma4_sliding");
    assert_eq!(f.usize("window"), 4);
    let a = Attn {
        window: Some(4),
        q_norm: Some(Norm::rms(1e-6)),
        k_norm: Some(Norm::rms(1e-6)),
        v_norm: Some(Norm::rms_bare(1e-6)),
        scale: 1.0,
        ..attn(4, 2, 16, Some(Rope::plain(1e4, 16)))
    };
    let mut map = QKVO.to_vec();
    map.extend([("q_norm", "q_norm.weight"), ("k_norm", "k_norm.weight")]);
    let p = params(&f, &map);
    let mut c = attention::KvCache::default();
    check(&f, "gemma-4 sliding attention", |h, pos| attention::step(&a, &p, &mut c, h, pos).unwrap());
    // the window matters on this prompt: without it the output differs
    let full = Attn { window: None, ..a.clone() };
    let mut c = attention::KvCache::default();
    let d = f.usize("hidden");
    let got: Vec<f32> = f.input().chunks(d).zip(f.positions()).flat_map(|(h, pos)| attention::step(&full, &p, &mut c, h, pos).unwrap()).collect();
    assert!(super::testing::rel(&got, &f.f32s("output")) > 1e-2);
}

#[test]
fn gemma4_global_attention() {
    let f = Fixture::load("gemma4_global");
    let a = Attn {
        q_norm: Some(Norm::rms(1e-6)),
        k_norm: Some(Norm::rms(1e-6)),
        v_norm: Some(Norm::rms_bare(1e-6)),
        scale: 1.0,
        k_eq_v: true,
        ..attn(4, 1, 32, Some(Rope::proportional(1e6, 32, 0.25)))
    };
    let p = params(&f, &[("q", "q_proj.weight"), ("k", "k_proj.weight"), ("o", "o_proj.weight"), ("q_norm", "q_norm.weight"), ("k_norm", "k_norm.weight")]);
    let mut c = attention::KvCache::default();
    check(&f, "gemma-4 global attention", |h, pos| attention::step(&a, &p, &mut c, h, pos).unwrap());
}

#[test]
fn qwen38_gated_deltanet() {
    let f = Fixture::load("qwen38_gdn");
    let l = Linear {
        kind: LinearKind::GatedDeltaNet,
        k_heads: 2,
        v_heads: 4,
        k_dim: 16,
        v_dim: 16,
        conv: 4,
        decay: Decay::Softplus,
        gate_act: Act::Silu,
        norm_eps: 1e-6,
        state_f32: true,
    };
    let p = params(
        &f,
        &[
            ("qkv", "in_proj_qkv.weight"),
            ("conv", "conv1d.weight"),
            ("beta", "in_proj_b.weight"),
            ("decay", "in_proj_a.weight"),
            ("gate", "in_proj_z.weight"),
            ("a_log", "A_log"),
            ("dt_bias", "dt_bias"),
            ("norm", "norm.weight"),
            ("o", "out_proj.weight"),
        ],
    );
    let mut st = linear::LinearState::new(&l);
    check(&f, "Qwen3.8 Gated DeltaNet", |h, _| linear::step(&l, &p, &mut st, h).unwrap());
}

#[test]
fn glm53_kimi_delta_attention() {
    let f = Fixture::load("glm53flash_kda");
    let l = Linear {
        kind: LinearKind::KimiDelta,
        k_heads: 4,
        v_heads: 4,
        k_dim: 16,
        v_dim: 16,
        conv: 4,
        decay: Decay::Bounded(-5.0),
        gate_act: Act::Sigmoid,
        norm_eps: 1e-5,
        state_f32: true,
    };
    let p = params(
        &f,
        &[
            ("q", "q_proj.weight"),
            ("k", "k_proj.weight"),
            ("v", "v_proj.weight"),
            ("conv", "conv1d.weight"),
            ("beta", "b_proj.weight"),
            ("decay_a", "forget_gate.f_a_proj.weight"),
            ("decay_b", "forget_gate.f_b_proj.weight"),
            ("a_log", "forget_gate.A_log"),
            ("dt_bias", "forget_gate.dt_bias"),
            ("gate_a", "g_a_proj.weight"),
            ("gate_b", "g_b_proj.weight"),
            ("norm", "o_norm.weight"),
            ("o", "o_proj.weight"),
        ],
    );
    let mut st = linear::LinearState::new(&l);
    check(&f, "GLM-5.3-Flash KDA", |h, _| linear::step(&l, &p, &mut st, h).unwrap());
}

#[test]
fn glm53_latent_attention() {
    let f = Fixture::load("glm53_mla");
    let m = Mla {
        heads: 4,
        q_lora: Some(32),
        kv_lora: 16,
        nope: 8,
        rope_dim: 8,
        v_dim: 8,
        rope: Some(Rope { style: RopeStyle::Interleaved, ..Rope::plain(1e4, 8) }),
        q_norm: Norm::rms(1e-6),
        kv_norm: Norm::rms(1e-6),
        gate: None,
        scale: 16f32.lpowf(-0.5),
        absorbed: false,
        indexer: None,
        head_mix: 0,
    };
    let p = params(
        &f,
        &[
            ("q_a", "q_a_proj.weight"),
            ("q_a_norm", "q_a_layernorm.weight"),
            ("q_b", "q_b_proj.weight"),
            ("kv_a", "kv_a_proj_with_mqa.weight"),
            ("kv_a_norm", "kv_a_layernorm.weight"),
            ("kv_b", "kv_b_proj.weight"),
            ("o", "o_proj.weight"),
        ],
    );
    let mut c = mla::LatentCache::default();
    check(&f, "GLM-5.3 MLA", |h, pos| mla::step(&m, &p, &mut c, h, pos).unwrap());
}

#[test]
fn gemma4_mlp() {
    let f = Fixture::load("gemma4_mlp");
    let m = Mlp { act: Act::GeluTanh, gated: true, bias: false, clamp: None };
    let p = params(&f, &[("gate", "gate_proj.weight"), ("up", "up_proj.weight"), ("down", "down_proj.weight")]);
    check(&f, "gemma-4 MLP", |h, _| ffn::mlp(&m, &p, "", h).unwrap());
}

/// A MoE fixture's experts and shared expert under their roles (`shared` the shared expert's module name).
fn moe_params(f: &Fixture, shared: &str, more: &[(&str, &str)]) -> Params {
    let names = [
        ("router".to_string(), "gate.weight".to_string()),
        ("experts_gate_up".into(), "experts.gate_up_proj".into()),
        ("experts_down".into(), "experts.down_proj".into()),
        ("shared_gate".into(), format!("{shared}.gate_proj.weight")),
        ("shared_up".into(), format!("{shared}.up_proj.weight")),
        ("shared_down".into(), format!("{shared}.down_proj.weight")),
    ];
    let mut map: Vec<(&str, &str)> = names.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    map.extend_from_slice(more);
    params(f, &map)
}

fn silu_mlp(clamp: Option<f32>) -> Mlp {
    Mlp { act: Act::Silu, gated: true, bias: false, clamp: clamp.map(crate::runtime::kinds::Clamp::Gate) }
}

#[test]
fn glm53_moe() {
    let f = Fixture::load("glm53_moe");
    let m = Moe {
        router: Router {
            score: RouterScore::Sigmoid,
            logit_scale: 1.0,
            bias: true,
            groups: Some((4, 2)),
            top_k: 2,
            renorm: true,
            scale: 2.5,
            hash: false,
            routers: 1,
            expert_scale: false,
        },
        experts: 8,
        expert: silu_mlp(None),
        shared: Some(silu_mlp(None)),
        shared_gate: false,
        latent: None,
    };
    let p = moe_params(&f, "shared_experts", &[("choice_bias", "gate.e_score_correction_bias")]);
    check(&f, "GLM-5.3 MoE", |h, _| ffn::moe(&m, &p, h, 0).unwrap());
    // the group limit matters here: without it some token picks other experts
    let free = Moe { router: Router { groups: None, ..m.router.clone() }, ..m.clone() };
    let d = f.usize("hidden");
    let got: Vec<f32> = f.input().chunks(d).flat_map(|h| ffn::moe(&free, &p, h, 0).unwrap()).collect();
    assert!(super::testing::rel(&got, &f.f32s("output")) > 1e-3);
}

#[test]
fn deepseek_v4_moe() {
    for (case, hash) in [("dsv4_hash_moe", true), ("dsv4_moe", false)] {
        let f = Fixture::load(case);
        let m = Moe {
            router: Router {
                score: RouterScore::SqrtSoftplus,
                logit_scale: 1.0,
                bias: !hash,
                groups: None,
                top_k: 2,
                renorm: true,
                scale: 1.5,
                hash,
                routers: 1,
                expert_scale: false,
            },
            experts: 8,
            expert: silu_mlp(Some(1.0)),
            shared: Some(silu_mlp(Some(1.0))),
            shared_gate: false,
            latent: None,
        };
        let mut p = moe_params(&f, "shared_experts", if hash { &[] } else { &[("choice_bias", "gate.e_score_correction_bias")] });
        if hash {
            let t: Vec<f32> = f.j["hash"].as_array().unwrap().iter().flat_map(|r| r.as_array().unwrap().iter().map(|e| e.as_u64().unwrap() as f32)).collect();
            p.put("hash", &[t.len() / 2, 2], t);
        }
        let ids: Vec<u32> = f.j["token_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let mut tok = ids.iter();
        check(&f, case, |h, _| ffn::moe(&m, &p, h, *tok.next().unwrap()).unwrap());
    }
}

#[test]
fn qwen38_moe() {
    let f = Fixture::load("qwen38_moe");
    let m = Moe {
        router: Router {
            score: RouterScore::Softmax,
            logit_scale: 1.0,
            bias: false,
            groups: None,
            top_k: 3,
            renorm: true,
            scale: 1.0,
            hash: false,
            routers: 1,
            expert_scale: false,
        },
        experts: 8,
        expert: silu_mlp(None),
        shared: Some(silu_mlp(None)),
        shared_gate: true,
        latent: None,
    };
    let p = moe_params(&f, "shared_expert", &[("shared_router", "shared_expert_gate.weight")]);
    check(&f, "Qwen3.8 MoE", |h, _| ffn::moe(&m, &p, h, 0).unwrap());
}

/// Each rope scaling's inverse frequencies and attention factor against transformers' `ROPE_INIT_FUNCTIONS`.
#[test]
fn rope_scaling() {
    let f = Fixture::load("rope_scaling");
    let cases = f.j["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 8);
    for c in cases {
        let rp = &c["rope_parameters"];
        let num = |k: &str| rp[k].as_f64().unwrap();
        let opt = |k: &str| rp[k].as_f64();
        let orig = || rp["original_max_position_embeddings"].as_u64().unwrap() as u32;
        let hd = c["head_dim"].as_u64().unwrap() as usize;
        let dim = (hd as f64 * opt("partial_rotary_factor").unwrap_or(1.0)) as usize;
        let max_pos = c["max_position_embeddings"].as_u64().unwrap() as u32;
        let factors = |k: &str| rp[k].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect::<Vec<_>>();
        let scaling = match rp["rope_type"].as_str().unwrap() {
            "linear" => RopeScaling::Linear { factor: num("factor") },
            "dynamic" => RopeScaling::DynamicNtk { factor: num("factor"), max_positions: max_pos },
            "yarn" => RopeScaling::Yarn {
                factor: num("factor"),
                original_max: orig(),
                beta_fast: num("beta_fast"),
                beta_slow: num("beta_slow"),
                mscale: opt("mscale"),
                mscale_all_dim: opt("mscale_all_dim"),
                attention_factor: None,
                truncate: true,
            },
            "llama3" => RopeScaling::Llama3 {
                factor: num("factor"),
                low_freq_factor: num("low_freq_factor"),
                high_freq_factor: num("high_freq_factor"),
                original_max: orig(),
            },
            "longrope" => RopeScaling::LongRope {
                short: factors("short_factor"),
                long: factors("long_factor"),
                original_max: orig(),
                max_positions: max_pos,
                attention_factor: None,
            },
            t => panic!("{t}"),
        };
        let r = Rope { scaling, ..Rope::plain(num("rope_theta"), dim) };
        let (inv, af) = super::rope::inv_freq_at(&r, c["seq_len"].as_u64().unwrap_or(0) as u32);
        let want: Vec<f32> = c["inv_freq"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let name = c["name"].as_str().unwrap();
        assert_eq!(inv.len(), want.len(), "{name}");
        for (i, (a, b)) in inv.iter().zip(&want).enumerate() {
            assert!((a - b).abs() <= 2e-6 * b.abs(), "{name}: frequency {i}: {a} against {b}");
        }
        assert!((af - c["attention_factor"].as_f64().unwrap() as f32).abs() < 1e-6, "{name}: attention factor {af}");
    }
}

/// GLM-5.3's MTP head against `tools/glm_ref.py`'s `mtp_forward` (which rounds to bf16 between steps, hence `BF16_TOL`), its blocks the
/// transformers modules of `glm53_latent_attention` and `glm53_moe`.
#[test]
fn glm53_mtp_head() {
    use super::{mtp::RefMtp, RefLayer, BF16_TOL};
    let f = Fixture::load("glm53_mtp");
    let mla_kind = Mla {
        heads: 4,
        q_lora: Some(32),
        kv_lora: 16,
        nope: 8,
        rope_dim: 8,
        v_dim: 8,
        rope: Some(Rope { style: RopeStyle::Interleaved, ..Rope::plain(1e4, 8) }),
        q_norm: Norm::rms(1e-6),
        kv_norm: Norm::rms(1e-6),
        gate: None,
        scale: 16f32.lpowf(-0.5),
        absorbed: false,
        indexer: None,
        head_mix: 0,
    };
    let moe_kind = Moe {
        router: Router {
            score: RouterScore::Sigmoid,
            logit_scale: 1.0,
            bias: true,
            groups: Some((4, 2)),
            top_k: 2,
            renorm: true,
            scale: 2.5,
            hash: false,
            routers: 1,
            expert_scale: false,
        },
        experts: 8,
        expert: silu_mlp(None),
        shared: Some(silu_mlp(None)),
        shared_gate: false,
        latent: None,
    };
    let block = |body: Body| Block { norm_in: Norm::rms(1e-5), body, norm_out: None, residual: Residual::Add };
    let twice = Norm { rounding: Rounding::Twice, ..Norm::rms(1e-5) };
    let head = MtpHead {
        blocks: vec![block(Body::Mixer(Mixer::Mla(mla_kind))), block(Body::Ffn(Ffn::Moe(moe_kind)))],
        norm_embed: twice,
        norm_hidden: twice,
        final_norm: Norm::rms(1e-5),
        dspark: None,
        input: MtpInput::Joined,
    };
    let pa = params(
        &f,
        &[
            ("q_a", "attn.q_a_proj.weight"),
            ("q_a_norm", "attn.q_a_layernorm.weight"),
            ("q_b", "attn.q_b_proj.weight"),
            ("kv_a", "attn.kv_a_proj_with_mqa.weight"),
            ("kv_a_norm", "attn.kv_a_layernorm.weight"),
            ("kv_b", "attn.kv_b_proj.weight"),
            ("o", "attn.o_proj.weight"),
            ("norm_in", "norm_a"),
        ],
    );
    let pm = params(
        &f,
        &[
            ("router", "moe.gate.weight"),
            ("choice_bias", "moe.gate.e_score_correction_bias"),
            ("experts_gate_up", "moe.experts.gate_up_proj"),
            ("experts_down", "moe.experts.down_proj"),
            ("shared_gate", "moe.shared_experts.gate_proj.weight"),
            ("shared_up", "moe.shared_experts.up_proj.weight"),
            ("shared_down", "moe.shared_experts.down_proj.weight"),
            ("norm_in", "norm_f"),
        ],
    );
    let ph = params(&f, &[("eh_proj", "eh_proj"), ("norm_embed", "enorm"), ("norm_hidden", "hnorm"), ("final_norm", "head_norm")]);
    let layers = vec![RefLayer::new(head.blocks[0].clone(), 64, pa), RefLayer::new(head.blocks[1].clone(), 64, pm)];
    let mut m = RefMtp::new(head, 64, ph, layers).unwrap();
    let (_, embed) = f.tensor("embed");
    let toks: Vec<usize> = f.j["token_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let prev: Vec<f32> = f.input().into_iter().map(super::bf16).collect();
    let got: Vec<f32> = prev
        .chunks(64)
        .enumerate()
        .flat_map(|(j, h)| m.step(0, j as u32, toks[j] as u32, &embed[toks[j] * 64..][..64].iter().map(|&v| super::bf16(v)).collect::<Vec<_>>(), h).unwrap())
        .collect();
    assert_close(&got, &f.f32s("output"), BF16_TOL, "GLM-5.3 MTP head against glm_ref.py");
}

/// DeepSeek-V4's mHC residual rule around a block (an MLP body), through the reference's block: the streams in, the streams out. The fixture is transformers'
/// in float32 (as every fixture here), so the streams' rounding point (the release's `Once`) is f32's here.
#[test]
fn deepseek_v4_mhc() {
    use super::RefLayer;
    let f = Fixture::load("dsv4_mhc");
    let block = Block {
        norm_in: Norm::rms(1e-6),
        body: Body::Ffn(Ffn::Mlp(silu_mlp(Some(10.0)))),
        norm_out: None,
        residual: Residual::Streams(StreamsP {
            streams: 4,
            sinkhorn_iters: 20,
            eps: 1e-6,
            norm_eps: 1e-6,
            rounding: Rounding::F32,
            lagged: false,
            eps_outside: false,
        }),
    };
    let p = params(
        &f,
        &[
            ("hc_fn", "hc.fn"),
            ("hc_base", "hc.base"),
            ("hc_scale", "hc.scale"),
            ("norm_in", "norm.weight"),
            ("gate", "mlp.gate_proj.weight"),
            ("up", "mlp.up_proj.weight"),
            ("down", "mlp.down_proj.weight"),
        ],
    );
    let mut l = RefLayer::new(block, 64, p);
    let mut x = f.input();
    let pos: Vec<u32> = f.positions();
    l.forward(&pos, &vec![0; pos.len()], &[], &mut x).unwrap();
    assert_close(&x, &f.f32s("output"), HF_TOL, "DeepSeek-V4 mHC");
}
