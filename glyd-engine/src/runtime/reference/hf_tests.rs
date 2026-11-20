//! Each kind against transformers in float32, on the fixtures `tests/runtime/record.py` writes: a recent model's module at reduced widths,
//! a causal prompt of a few tokens, the same generated weights on both sides.

use super::testing::{assert_close, Fixture};
use super::{attention, linear, mla, Params, HF_TOL};
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
        scale: (head_dim as f32).powf(-0.5),
        k_eq_v: false,
        indexer: None,
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
        scale: 16f32.powf(-0.5),
        absorbed: false,
        indexer: None,
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
