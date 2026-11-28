//! Descriptions from config.json: whole tiny models against transformers (the config transformers wrote, logits of a prompt), and the
//! released configs of the recent families against their checkpoints' tensor lists.

use super::describe::describe;
use super::reference::model::RefModel;
use super::reference::testing::{assert_close, Fixture};
use std::collections::BTreeSet;
use std::path::PathBuf;

/// The tiny model's description reads exactly the fixture's tensors, and its logits are transformers'.
fn model(case: &str, tol: f32) {
    let f = Fixture::load(case);
    let mut d = describe(&f.j["config"]).unwrap_or_else(|e| panic!("{case}: {}", e.msg));
    let want: BTreeSet<String> = f.j["tensors"].as_object().unwrap().keys().cloned().collect();
    assert_eq!(d.names(), want, "{case}: the weight map");
    // (each shape the description gives is the tensor's: a loader by the kinds and a random model read them)
    for (n, s) in &d.shapes {
        let got: Vec<usize> =
            f.j["tensors"][n]["shape"].as_array().map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect()).unwrap_or_default();
        assert_eq!(&got, s, "{case}: the shape of {n}");
    }
    // (transformers in float32: every bf16 rounding point the description gives as the family's release rounds, off: `ModelDesc::f32_arithmetic`)
    d.desc.f32_arithmetic();
    let src = |n: &str| f.j["tensors"].get(n).map(|_| f.tensor(n));
    let mut m = RefModel::load(&d, &src).unwrap();
    let toks: Vec<u32> = f.j["token_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let got: Vec<f32> = m.forward(&toks, 0).unwrap().concat();
    assert_close(&got, &f.f32s("output"), tol, case);
}

const MODEL_TOL: f32 = 1e-5;

/// The full-size GLM layout's description gives every tensor it names a shape (what a loader by the kinds and a random model of the layout read); `model`
/// holds each to the recorded checkpoint's.
#[test]
fn glm53_model_shapes_every_tensor() {
    let f = Fixture::load("glm53_model");
    let d = describe(&f.j["config"]).unwrap_or_else(|e| panic!("{}", e.msg));
    let missing: Vec<String> = d.names().into_iter().filter(|n| !d.shapes.contains_key(n)).collect();
    assert!(missing.is_empty(), "no shape for {missing:?}");
}

/// The full-size GLM family's shape with an indexer that drops tokens and a layer whose indexer is shared: the recorded fixture's model, its
/// `index_topk` cut to 4 of its prompt's 7 tokens and its second layer's indexer made `shared`. The reference's model lends the first layer's picks
/// to the second (`RefModel::share_into`): each query keeps the same tokens in both, 4 at most; while every token is kept the logits are the
/// recorded ones, and past that they are not. This is NOT a check against transformers: no fixture of this model was recorded
/// (reference/mla.rs says what one takes).
#[test]
fn glm53_model_with_fewer_picks_and_a_shared_layer_runs_on_the_reference() {
    use super::reference::SeqState;
    let f = Fixture::load("glm53_model");
    let mut cfg = f.j["config"].clone();
    cfg["index_topk"] = 4.into();
    cfg["indexer_types"] = serde_json::json!(["full", "shared", "full"]);
    let mut d = describe(&cfg).unwrap_or_else(|e| panic!("{}", e.msg));
    d.desc.f32_arithmetic();
    let src = |n: &str| f.j["tensors"].get(n).map(|_| f.tensor(n));
    let mut m = RefModel::load(&d, &src).unwrap_or_else(|e| panic!("{}", e.msg));
    let toks: Vec<u32> = f.j["token_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let (want, mut got) = (f.f32s("output"), Vec::new());
    let vocab = want.len() / toks.len();
    for (t, &tok) in toks.iter().enumerate() {
        let row = m.forward(&[tok], t as u32).unwrap_or_else(|e| panic!("token {t}: {}", e.msg)).concat();
        // (a layer's mixer is its first block: blocks 0, 2 and 4)
        let picks = |b: usize| match m.layers[b].seqs.get(&0) {
            Some(SeqState::Mla(c)) => c.picks.clone(),
            _ => panic!("block {b} has no latent cache"),
        };
        assert_eq!(picks(0), picks(2), "token {t}: the shared layer's picks are the layer's before it");
        assert_eq!((picks(0).len(), picks(4).len()), ((t + 1).min(4), (t + 1).min(4)), "token {t}");
        if t < 4 {
            assert_close(&row, &want[t * vocab..(t + 1) * vocab], MODEL_TOL, &format!("token {t}, every token kept"));
        }
        got.extend(row);
    }
    let apart = got[4 * vocab..].iter().zip(&want[4 * vocab..]).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    assert!(apart > 1e-4, "the picks dropped no token that mattered: {apart}");
    // (the tokens in one call: a layer runs them all before the next layer, and the shared layer takes each token's own picks, so its rows are the
    // token-by-token ones, bit for bit)
    let whole = RefModel::load(&d, &src).unwrap().forward(&toks, 0).unwrap_or_else(|e| panic!("{}", e.msg)).concat();
    assert!(whole.iter().zip(&got).all(|(a, b)| a.to_bits() == b.to_bits()) && whole.len() == got.len(), "the tokens in one call");
}

#[test]
fn qwen38_dense_model() {
    model("qwen38_model", MODEL_TOL);
}

#[test]
fn qwen38_moe_model() {
    model("qwen38_moe_model", MODEL_TOL);
}

#[test]
fn gemma4_model() {
    model("gemma4_model", MODEL_TOL);
}

#[test]
fn llama_model() {
    model("llama_model", MODEL_TOL);
}

fn fixture(name: &str) -> serde_json::Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name);
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

/// Every Qwen3.5-architecture release's config describes; the 0.8B and 9B weight maps are their checkpoints' tensors less the vision
/// tower and the MTP layer (which the description does not use yet).
#[test]
fn released_qwen_configs() {
    let cfgs = fixture("qwen35_configs.json");
    for (repo, c) in cfgs.as_object().unwrap() {
        let d = describe(c).unwrap_or_else(|e| panic!("{repo}: {}", e.msg));
        let tc = c.get("text_config").unwrap_or(c);
        assert_eq!(d.desc.blocks.len(), 2 * tc["num_hidden_layers"].as_u64().unwrap() as usize, "{repo}");
        assert_eq!(d.mtp_layers, tc["mtp_num_hidden_layers"].as_u64().unwrap_or(0) as usize, "{repo}");
    }
    let tensors = fixture("qwen35_tensors.json");
    for repo in ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-9B"] {
        let d = describe(&cfgs[repo]).unwrap();
        let want: BTreeSet<String> =
            tensors[repo].as_object().unwrap().keys().filter(|n| !n.starts_with("model.visual.") && !n.starts_with("mtp.")).cloned().collect();
        assert_eq!(d.names(), want, "{repo}");
        // (and each tensor's shape as the config makes it is the checkpoint's)
        assert_eq!(d.shapes.keys().cloned().collect::<BTreeSet<_>>(), want, "{repo}: a shape for every tensor");
        for (n, sh) in &d.shapes {
            let got: Vec<usize> = tensors[repo][n][1].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
            assert_eq!(&got, sh, "{repo}: {n}");
        }
    }
}

/// #333: Qwen3.8-2.4T-A95B (`Qwen3_5MoeForCausalLM`, its released config.json in tests/fixtures/qwen35_configs.json) describes with the kinds the library has:
/// the Gated DeltaNet and gated attention of the Qwen3.5 layout, and on every layer a mixture of 512 softmax-routed experts (top 10, renormalised) with the
/// shared expert gated by sigmoid(shared_router x); each tensor's shape said, each expert's matrices and the shared expert's included. It has no other
/// mechanism: what it lacks to run is the Qwen3.5 layout's experts on the executor (its weights, 4.6 TB of bf16, are for a machine of many GPUs).
#[test]
fn qwen38_2_4t_describes_with_the_existing_kinds() {
    use super::kinds::*;
    let cfgs = fixture("qwen35_configs.json");
    let c = &cfgs["Qwen/Qwen3.8-2.4T-A95B"];
    let d = describe(c).unwrap_or_else(|e| panic!("{}", e.msg));
    let n = |k: &str| c[k].as_u64().unwrap() as usize;
    let (layers, hidden) = (n("num_hidden_layers"), n("hidden_size"));
    assert_eq!(d.desc.blocks.len(), 2 * layers);
    for (l, ty) in c["layer_types"].as_array().unwrap().iter().enumerate() {
        match (&d.desc.blocks[2 * l].body, ty.as_str().unwrap()) {
            (Body::Mixer(Mixer::Linear(x)), "linear_attention") => assert_eq!(x.kind, LinearKind::GatedDeltaNet),
            (Body::Mixer(Mixer::Attention(a)), "full_attention") => assert_eq!(a.gate.map(|g| g.from), Some(GateFrom::InQuery)),
            (b, t) => panic!("layer {l} ({t}): {b:?}"),
        }
        let Body::Ffn(Ffn::Moe(m)) = &d.desc.blocks[2 * l + 1].body else { panic!("layer {l}: {:?}", d.desc.blocks[2 * l + 1].body) };
        assert_eq!((m.experts, m.router.top_k, m.router.score, m.router.renorm, m.shared_gate), (512, 10, RouterScore::Softmax, true, true));
    }
    assert_eq!(d.shapes.keys().cloned().collect::<BTreeSet<_>>(), d.names(), "a shape for every tensor described");
    let (mi, si) = (n("moe_intermediate_size"), n("shared_expert_intermediate_size"));
    let p = "model.layers.3.mlp.";
    for (t, want) in [
        ("experts.511.gate_proj.weight", vec![mi, hidden]),
        ("experts.0.down_proj.weight", vec![hidden, mi]),
        ("gate.weight", vec![512, hidden]),
        ("shared_expert.up_proj.weight", vec![si, hidden]),
        ("shared_expert_gate.weight", vec![1, hidden]),
    ] {
        assert_eq!(d.shapes.get(&format!("{p}{t}")), Some(&want), "{t}");
    }
}

#[test]
fn released_gemma4_config() {
    use super::kinds::*;
    let d = describe(&fixture("gemma4_12b_config.json")).unwrap();
    assert_eq!(d.desc.blocks.len(), 96);
    let attn = |i: usize| match &d.desc.blocks[2 * i].body {
        Body::Mixer(Mixer::Attention(a)) => a.clone(),
        b => panic!("{b:?}"),
    };
    let (s, g) = (attn(0), attn(5));
    assert_eq!((s.head_dim, s.kv_heads, s.window, s.k_eq_v), (256, 8, Some(1024), false));
    assert_eq!((g.head_dim, g.kv_heads, g.window, g.k_eq_v), (512, 1, None, true));
    assert_eq!(g.rope.unwrap(), Rope::proportional(1e6, 512, 0.25));
    // (the release's logits are bf16 from the head's product on, through the soft-cap's three operations)
    assert_eq!((d.desc.head.tied, d.desc.head.softcap, d.desc.head.rounding), (true, Some(30.0), Rounding::Twice));
    assert!(d.head.is_none());
}

/// KV sharing and the double-wide MLP (E2B, E4B): the last `num_kv_shared_layers` layers' attentions read the keys and values of the last earlier layer of
/// their type (`Attn::kv_from`, that layer's attention block), name no k, v or k norm (a checkpoint's are left unread), and their MLPs are twice as wide.
#[test]
fn gemma4_kv_sharing_and_the_double_wide_mlp() {
    use super::kinds::*;
    let mut g = fixture("gemma4_12b_config.json");
    let t = &mut g["text_config"];
    let types = [
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
    ];
    t["layer_types"] = serde_json::json!(types);
    (t["num_hidden_layers"], t["num_kv_shared_layers"], t["use_double_wide_mlp"]) = (8.into(), 4.into(), true.into());
    let inter = t["intermediate_size"].as_u64().unwrap() as usize;
    let d = describe(&g).unwrap();
    let from: Vec<Option<usize>> = (0..8)
        .map(|l| match &d.desc.blocks[2 * l].body {
            Body::Mixer(Mixer::Attention(a)) => a.kv_from,
            b => panic!("{b:?}"),
        })
        .collect();
    assert_eq!(from, [None, None, None, None, Some(6), Some(4), Some(6), Some(4)]);
    let p = "model.language_model.layers";
    for l in 0..8 {
        let names: BTreeSet<String> = d.blocks[2 * l].iter().map(|(r, _)| r.clone()).collect();
        assert_eq!(names.contains("k") && names.contains("k_norm"), l < 4, "layer {l}: {names:?}");
        assert_eq!(d.shapes[&format!("{p}.{l}.mlp.gate_proj.weight")][0], if l < 4 { inter } else { 2 * inter }, "layer {l}");
        assert_eq!(d.unread.contains(&format!("{p}.{l}.self_attn.k_proj.")), l >= 4);
    }
    // (the double-wide MLP is on the shared layers alone; with no shared layers there is none)
    g["text_config"]["num_kv_shared_layers"] = 0.into();
    assert!(describe(&g).unwrap().shapes.iter().filter(|(n, _)| n.ends_with("gate_proj.weight")).all(|(_, s)| s[0] == inter));
    // (every layer shared: none makes the keys and values)
    g["text_config"]["num_kv_shared_layers"] = 8.into();
    assert!(describe(&g).is_err());
}

/// A key the builder does not know fails with its name; a setting it does not support says so.
#[test]
fn unknown_keys_are_named() {
    let mut c = fixture("qwen35_configs.json")["Qwen/Qwen3.8-27B"].clone();
    c["text_config"]["attn_logit_softcap"] = 50.0.into();
    let e = describe(&c).unwrap_err().msg;
    assert!(e.contains("attn_logit_softcap"), "{e}");
    let mut g = fixture("gemma4_12b_config.json");
    g["text_config"]["enable_moe_block"] = true.into();
    assert!(describe(&g).unwrap_err().msg.contains("enable_moe_block"));
    let mut g = fixture("gemma4_12b_config.json");
    g["text_config"]["model_type"] = "something_new".into();
    assert!(describe(&g).unwrap_err().msg.contains("something_new"));
}

#[test]
fn glm53_model() {
    model("glm53_model", MODEL_TOL);
}

#[test]
fn glm53_flash_model() {
    model("glm53flash_model", MODEL_TOL);
}

/// GLM-5.3-Flash's released config describes its released checkpoint exactly: every tensor of the language model (the experts from
/// the fixture's runs, the FP8 matrices with their block scales), less the vision tower and the MTP layer.
#[test]
fn released_glm53_flash_config() {
    let d = describe(&fixture("glm5_next_config.json")).unwrap();
    let t = fixture("glm5_next_tensors.json");
    let all = names_with_runs(&t);
    let mtp = format!("model.language_model.layers.{}.", d.desc.blocks.len() / 2);
    let want: BTreeSet<String> = all.into_iter().filter(|n| !n.starts_with("model.visual.") && !n.starts_with(&mtp)).collect();
    let got = d.names();
    let missing: Vec<_> = want.difference(&got).take(8).collect();
    let extra: Vec<_> = got.difference(&want).take(8).collect();
    assert!(missing.is_empty() && extra.is_empty(), "not described: {missing:?}; not in the checkpoint: {extra:?}");
    assert_eq!((d.mtp_layers, d.desc.embed.streams), (1, 4));
}

/// Qwen3.8-Flash-Next's released config (tests/fixtures/qwen4x_config.json) describes its released checkpoint (tests/fixtures/qwen4x_tensors.json, the
/// release's safetensors headers) exactly: every tensor, the MTP layer's as the draft head's, the vision tower's; each at the checkpoint's shape where the
/// description gives one.
#[test]
fn released_qwen38_flash_next_config() {
    let d = describe(&fixture("qwen4x_config.json")).unwrap_or_else(|e| panic!("{}", e.msg));
    let t = fixture("qwen4x_tensors.json");
    let tensors = t["tensors"].as_object().unwrap();
    let have: BTreeSet<String> = tensors.keys().cloned().collect();
    d.check(&have, true).unwrap_or_else(|e| panic!("{}", e.msg));
    for (n, shape) in &d.shapes {
        let got: Vec<usize> = tensors[n][1].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        assert_eq!(&got, shape, "{n}");
    }
}

/// GLM-5.3's released config (`glm_moe_dsa`) describes its released checkpoint exactly: every tensor (the experts from the fixture's runs), its MTP layer
/// as its draft head (DeepSeek-V3's joined form: enorm, hnorm, eh_proj, shared_head.norm, and a latent attention with an indexer of its own and the
/// experts on a plain residual), each at the checkpoint's shape where the description gives one; the FP8 matrices are the ones the checkpoint holds with
/// block scales (the head's too); the indexers' schedule is the config's (a `full` layer every fourth after the first three, `shared` between); the
/// latent attention turns 64 channels of interleaved pairs, its indexer the same first 64 of its keys and queries.
#[test]
fn released_glm53_config() {
    use super::kinds::{Body, Ffn, Mixer, MtpInput, Residual};
    let d = describe(&fixture("glm_moe_dsa_config.json")).unwrap_or_else(|e| panic!("{}", e.msg));
    let t = fixture("glm_moe_dsa_tensors.json");
    let want: BTreeSet<String> = names_with_runs(&t);
    let got = d.names();
    let missing: Vec<_> = want.difference(&got).take(8).collect();
    let extra: Vec<_> = got.difference(&want).take(8).collect();
    assert!(missing.is_empty() && extra.is_empty(), "not described: {missing:?}; not in the checkpoint: {extra:?}");
    let shape = |n: &str| -> Option<Vec<usize>> {
        let s = t["tensors"].get(n).map(|v| v[1].clone()).or_else(|| {
            t["runs"].as_array().unwrap().iter().find_map(|r| {
                let rest = n.strip_prefix(r["prefix"].as_str().unwrap())?;
                r["tensors"].get(rest.split_once('.')?.1).map(|v| v[1].clone())
            })
        })?;
        Some(s.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect())
    };
    for (n, s) in &d.shapes {
        assert_eq!(shape(n).as_ref(), Some(s), "the shape of {n}");
    }
    let scaled: BTreeSet<String> = want.iter().filter_map(|n| n.strip_suffix("_scale_inv").map(str::to_string)).collect();
    let fp8: BTreeSet<String> = d.scales.keys().cloned().collect();
    assert_eq!(fp8, scaled, "the FP8 matrices");
    let mla: Vec<_> = d.desc.blocks.iter().filter_map(|b| if let Body::Mixer(Mixer::Mla(m)) = &b.body { Some(m) } else { None }).collect();
    let shared: Vec<bool> = mla.iter().map(|m| m.indexer.as_ref().unwrap().shared).collect();
    assert_eq!(shared, (0..78).map(|l| l >= 3 && (l - 2) % 4 != 0).collect::<Vec<bool>>());
    let (m, ix) = (mla[0], mla[0].indexer.as_ref().unwrap());
    assert_eq!(
        (m.rope_dim, m.rope.as_ref().map(|r| (r.dim, r.style)), ix.rope, ix.top_k, ix.block),
        (64, Some((64, super::kinds::RopeStyle::Interleaved)), 64, 2048, 1)
    );
    assert_eq!((d.mtp_layers, d.desc.embed.streams, d.desc.head.rounding), (1, 1, super::kinds::Rounding::Once));
    // (the draft head: layer 78's two blocks, its own roles under that layer's name; nothing of the checkpoint left unread)
    let (h, (_, own)) = (&d.desc.mtp[0], &d.mtp[0]);
    assert_eq!((d.desc.mtp.len(), h.input, h.blocks.len()), (1, MtpInput::Joined, 2));
    assert!(matches!(&h.blocks[0].body, Body::Mixer(Mixer::Mla(m)) if m.indexer.as_ref().is_some_and(|i| !i.shared) && m.rope_dim == 64));
    assert!(matches!(&h.blocks[1].body, Body::Ffn(Ffn::Moe(m)) if m.experts == 256) && h.blocks.iter().all(|b| b.residual == Residual::Add));
    let own: Vec<String> = own.iter().flat_map(|(_, s)| s.names()).collect();
    assert_eq!(
        own,
        ["model.layers.78.eh_proj.weight", "model.layers.78.enorm.weight", "model.layers.78.hnorm.weight", "model.layers.78.shared_head.norm.weight"]
    );
    assert!(d.unread.iter().all(|p| !p.starts_with("model.layers.")), "{:?}", d.unread);
}

/// The indexers' schedule where a config lists no `indexer_types`: transformers' GlmMoeDsaConfig makes it from `index_topk_pattern`, else from
/// `index_topk_freq` and `index_skip_topk_offset`; listed, they are the list (the release lists them and the schedule both).
#[test]
fn glm_indexer_types_follow_the_releases_schedule() {
    use super::kinds::{Body, Mixer};
    let shared = |cfg: &serde_json::Value| -> Vec<bool> {
        let d = describe(cfg).unwrap_or_else(|e| panic!("{}", e.msg));
        d.desc.blocks.iter().filter_map(|b| if let Body::Mixer(Mixer::Mla(m)) = &b.body { m.indexer.as_ref().map(|x| x.shared) } else { None }).collect()
    };
    let release = fixture("glm_moe_dsa_config.json");
    let mut cfg = release.clone();
    cfg.as_object_mut().unwrap().remove("indexer_types");
    assert_eq!(shared(&cfg), shared(&release));
    cfg["index_topk_pattern"] = "FSF".into();
    cfg["num_hidden_layers"] = 3.into();
    cfg["mlp_layer_types"] = serde_json::json!(["dense", "sparse", "sparse"]);
    assert_eq!(shared(&cfg), [false, true, false]);
    cfg["index_topk_pattern"] = "SFF".into();
    assert!(describe(&cfg).unwrap_err().msg.contains("first layer's indexer is shared"));
}

/// GLM's logits are its bf16 lm_head's output (modeling_glm5_next.py and modeling_glm_moe_dsa.py: `self.lm_head(hidden_states)` of a bf16 model, not
/// cast): the head rounds once, in both layouts.
#[test]
fn glm_heads_round_the_logits_once() {
    for cfg in [fixture("glm5_next_config.json"), Fixture::load("glm53_model").j["config"].clone()] {
        assert_eq!(describe(&cfg).unwrap().desc.head.rounding, super::kinds::Rounding::Once);
    }
}

/// GLM's form flags: the rotary in interleaved pairs (`rope_interleave`, `indexer_rope_interleave`), the pools' keys gated (`index_kpool_compress`) and the
/// open pool's tokens attended to (`index_kpool_always_select_tail`) are the only forms described; a config that says another is refused, the key and the
/// value the engine runs named, where the flag changes what is computed. Where it does not (GLM-5.3-Flash has no rotary; GLM-5.3 keys a token each), the
/// config describes as with the flag true.
#[test]
fn glm_form_flags_of_another_form_are_refused_in_words() {
    let (dsa, flash) = (fixture("glm_moe_dsa_config.json"), fixture("glm5_next_config.json"));
    let refused = |cfg: &serde_json::Value, key: &str| {
        let e = describe(cfg).err().unwrap_or_else(|| panic!("{key} = false described")).msg;
        assert!(e.contains(&format!("{key} = false")) && e.contains(&format!("the engine runs {key} = true")), "{key}: {e}");
    };
    for key in ["rope_interleave", "indexer_rope_interleave"] {
        let mut c = dsa.clone();
        c[key] = false.into();
        refused(&c, key);
        let mut c = flash.clone();
        c["text_config"][key] = false.into();
        assert_eq!(describe(&c).unwrap().desc, describe(&flash).unwrap().desc, "{key}: GLM-5.3-Flash turns nothing");
    }
    for key in ["index_kpool_compress", "index_kpool_always_select_tail"] {
        let mut c = flash.clone();
        c["text_config"][key] = false.into();
        refused(&c, key);
        let mut c = dsa.clone();
        c[key] = false.into();
        assert_eq!(describe(&c).unwrap().desc, describe(&dsa).unwrap().desc, "{key}: GLM-5.3 pools nothing");
    }
}

/// Keys a GLM config leaves out take transformers' defaults, in the description and in GLM-5.3-Flash's own reader alike (`glm::ConfigGlm`): the pools of
/// 16 (Glm5NextTextConfig's `index_kpool`), the forget gate's lower bound by `glm::gate_bound` (-5 where absent; null and no safe gate: no bound, which the
/// reader refuses), the first three layers dense where neither `mlp_layer_types` nor `first_k_dense_replace` is given. The releases set every one.
#[test]
fn glm_keys_left_out_take_transformers_defaults() {
    use super::kinds::{Body, Decay, Ffn, Mixer};
    let flash = fixture("glm5_next_config.json");
    let indexer_block = |cfg: &serde_json::Value| {
        describe(cfg)
            .unwrap()
            .desc
            .blocks
            .iter()
            .find_map(|b| if let Body::Mixer(Mixer::Mla(m)) = &b.body { m.indexer.as_ref().map(|x| x.block) } else { None })
    };
    let reader = |cfg: &serde_json::Value| crate::glm::ConfigGlm::parse(&cfg.to_string(), "c", None);
    let mut c = flash.clone();
    c["text_config"].as_object_mut().unwrap().remove("index_kpool");
    assert_eq!((indexer_block(&flash), indexer_block(&c), reader(&c).unwrap().indexer.kpool), (Some(4), Some(16), 16));
    // (the gate's bound: the description's decay and the reader's bound, or its refusal, for each way a config can say it)
    let decay = |cfg: &serde_json::Value| {
        describe(cfg).unwrap().desc.blocks.iter().find_map(|b| if let Body::Mixer(Mixer::Linear(l)) = &b.body { Some(l.decay) } else { None }).unwrap()
    };
    let block = |f: &dyn Fn(&mut serde_json::Map<String, serde_json::Value>)| {
        let mut c = flash.clone();
        f(c["text_config"]["linear_attn_config"].as_object_mut().unwrap());
        c
    };
    let flat = |v: Option<serde_json::Value>| {
        let mut c = flash.clone();
        let t = c["text_config"].as_object_mut().unwrap();
        t.remove("linear_attn_config");
        // (the block's other numbers as transformers' flat keys)
        for (k, n) in [("linear_num_heads", 64), ("linear_head_dim", 128), ("linear_conv_kernel_dim", 4)] {
            t.insert(k.into(), n.into());
        }
        if let Some(v) = v {
            t.insert("linear_lower_bound".into(), v);
        }
        c
    };
    for (what, cfg, want) in [
        ("the release's -5", flash.clone(), Some(-5.0)),
        (
            "the block without the key",
            block(&|o| {
                o.remove("gate_lower_bound");
            }),
            Some(-5.0),
        ),
        (
            "the block's key null, its safe gate on",
            block(&|o| {
                o.insert("gate_lower_bound".into(), serde_json::Value::Null);
            }),
            Some(-5.0),
        ),
        (
            "the block's -3",
            block(&|o| {
                o.insert("gate_lower_bound".into(), (-3.0).into());
            }),
            Some(-3.0),
        ),
        (
            "the block's key null, its safe gate off",
            block(&|o| {
                o.insert("gate_lower_bound".into(), serde_json::Value::Null);
                o.insert("safe_gate".into(), false.into());
            }),
            None,
        ),
        ("no block, no flat key", flat(None), Some(-5.0)),
        ("no block, the flat key -2", flat(Some((-2.0).into())), Some(-2.0)),
        ("no block, the flat key null", flat(Some(serde_json::Value::Null)), None),
    ] {
        match want {
            Some(b) => {
                assert_eq!(decay(&cfg), Decay::Bounded(b), "{what}");
                assert_eq!(reader(&cfg).unwrap_or_else(|e| panic!("{what}: {}", e.msg)).kda.bound, b, "{what}");
            }
            None => {
                assert_eq!(decay(&cfg), Decay::Softplus, "{what}");
                assert!(reader(&cfg).unwrap_err().msg.contains("without its lower bound"), "{what}");
            }
        }
    }
    // (the feed-forward's schedule with neither key: three dense layers, then experts)
    let mut c = fixture("glm_moe_dsa_config.json");
    c.as_object_mut().unwrap().retain(|k, _| k != "mlp_layer_types" && k != "first_k_dense_replace");
    let dense: Vec<bool> =
        describe(&c).unwrap().desc.blocks.iter().filter_map(|b| if let Body::Ffn(f) = &b.body { Some(matches!(f, Ffn::Mlp(_))) } else { None }).collect();
    assert_eq!(dense, (0..78).map(|l| l < 3).collect::<Vec<bool>>());
}

#[test]
fn deepseek_v4_model() {
    model("dsv4_model", MODEL_TOL);
}

/// The DeepSeek-V4 fixture exercises the compressed entries and the indexer's choice: without them (a window alone) the logits differ.
#[test]
fn deepseek_v4_compression_matters() {
    use super::kinds::{Body, Mixer};
    let f = Fixture::load("dsv4_model");
    let mut d = describe(&f.j["config"]).unwrap();
    let src = |n: &str| f.j["tensors"].get(n).map(|_| f.tensor(n));
    let toks: Vec<u32> = f.j["token_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let mut keep_all = d.clone();
    for (b, b2) in d.desc.blocks.iter_mut().zip(keep_all.desc.blocks.iter_mut()) {
        if let (Body::Mixer(Mixer::Compressed(c)), Body::Mixer(Mixer::Compressed(c2))) = (&mut b.body, &mut b2.body) {
            c.ratio = 0;
            if let Some(ix) = &mut c2.indexer {
                ix.top_k = 1000;
            }
        }
    }
    let want = f.f32s("output");
    let window_only = RefModel::load(&d, &src).unwrap().forward(&toks, 0).unwrap().concat();
    let every_entry = RefModel::load(&keep_all, &src).unwrap().forward(&toks, 0).unwrap().concat();
    assert!(super::reference::testing::rel(&window_only, &want) > 1e-3);
    assert!(super::reference::testing::rel(&every_entry, &want) > 1e-4);
}

/// Keys a DeepSeek-V4 config leaves out take transformers' DeepseekV4Config defaults (the routed weights times 1.5, the SwiGLU clamp at 10, the first three
/// layers routed by the token): Flash-0731's config, which sets each to that value, describes the same model without them, and the release's hand-written
/// reader (`deepseek_v4::ConfigV4`) reads the same numbers; transformers' flavour with no `mlp_layer_types` routes its first three layers by the token.
#[test]
fn deepseek_v4_keys_left_out_take_transformers_defaults() {
    use super::kinds::{Body, Ffn};
    let h = fixture("headers/DeepSeek-V4-Flash-0731.json");
    let v4: serde_json::Value = serde_json::from_str(h["config"].as_str().unwrap()).unwrap();
    let mut bare = v4.clone();
    for k in ["routed_scaling_factor", "swiglu_limit", "num_hash_layers"] {
        assert!(bare.as_object_mut().unwrap().remove(k).is_some(), "{k}");
    }
    assert_eq!(describe(&bare).unwrap().desc, describe(&v4).unwrap().desc);
    let read = |c: &serde_json::Value| crate::deepseek_v4::ConfigV4::parse(&c.to_string(), "c", None).unwrap_or_else(|e| panic!("{}", e.msg));
    let (a, b) = (read(&v4), read(&bare));
    assert_eq!((b.hash_layers, b.route_scale, b.swiglu_limit), (a.hash_layers, a.route_scale, a.swiglu_limit));
    let mut c = Fixture::load("dsv4_model").j["config"].clone();
    c.as_object_mut().unwrap().remove("mlp_layer_types");
    let hash: Vec<bool> =
        describe(&c).unwrap().desc.blocks.iter().filter_map(|b| if let Body::Ffn(Ffn::Moe(m)) = &b.body { Some(m.router.hash) } else { None }).collect();
    assert_eq!(hash, [true, true, true]);
}

/// A DeepSeek-V4 layer that compresses by 1 is refused in words, in either flavour (no release has one: vLLM and TensorRT-LLM read a 1 as the window alone,
/// transformers maps 0, 4 and 128 only, and this builder's ratio of 1 is DeepSeek-V4.1's compressor of a token); the ratios listed past the model's layers
/// (the draft layers', which take the window alone) are not read.
#[test]
fn deepseek_v4_a_ratio_of_one_is_refused() {
    let h = fixture("headers/DeepSeek-V4-Flash-0731.json");
    let v4: serde_json::Value = serde_json::from_str(h["config"].as_str().unwrap()).unwrap();
    let mut c = v4.clone();
    c["compress_ratios"][2] = 1.into();
    let e = describe(&c).unwrap_err().msg;
    assert!(e.contains("layer 2 compresses by 1, which no DeepSeek-V4 layer does"), "{e}");
    let mut c = v4.clone();
    for i in 43..46 {
        c["compress_ratios"][i] = 1.into();
    }
    assert_eq!(describe(&c).unwrap().desc, describe(&v4).unwrap().desc);
    let mut c = Fixture::load("dsv4_model").j["config"].clone();
    c["compress_rates"]["heavily_compressed_attention"] = 1.into();
    let e = describe(&c).unwrap_err().msg;
    assert!(e.contains("layer 2 compresses by 1"), "{e}");
}

/// A checkpoint's tensor names from a fixture that holds a layer's experts as a run.
fn names_with_runs(t: &serde_json::Value) -> BTreeSet<String> {
    let mut all: BTreeSet<String> = t["tensors"].as_object().unwrap().keys().cloned().collect();
    for run in t["runs"].as_array().unwrap() {
        let p = run["prefix"].as_str().unwrap();
        for e in 0..run["count"].as_u64().unwrap() {
            for n in run["tensors"].as_object().unwrap().keys() {
                all.insert(format!("{p}{e}.{n}"));
            }
        }
    }
    all
}

/// Nemotron 3 Super's released config describes its released checkpoint exactly: every tensor of the backbone and the head (the experts
/// from the fixture's runs), less the draft head; and each role's tensor has the shape its kind says.
#[test]
fn released_nemotron_config() {
    use super::describe::Source;
    use super::kinds::*;
    let cfg = fixture("nemotron_h_config.json");
    let d = describe(&cfg).unwrap_or_else(|e| panic!("{}", e.msg));
    let t = fixture("nemotron_h_tensors.json");
    let want: BTreeSet<String> = names_with_runs(&t).into_iter().filter(|n| !n.starts_with("mtp.")).collect();
    let got = d.names();
    let missing: Vec<_> = want.difference(&got).take(8).collect();
    let extra: Vec<_> = got.difference(&want).take(8).collect();
    assert!(missing.is_empty() && extra.is_empty(), "not described: {missing:?}; not in the checkpoint: {extra:?}");
    assert_eq!((d.desc.blocks.len(), d.mtp_layers, d.desc.head.tied, d.desc.rope.is_none()), (88, 1, false, true));
    let count = |f: &dyn Fn(&Body) -> bool| d.desc.blocks.iter().filter(|b| f(&b.body)).count();
    assert_eq!(count(&|b| matches!(b, Body::Mixer(Mixer::Mamba2(_)))), 40);
    assert_eq!(count(&|b| matches!(b, Body::Ffn(Ffn::Moe(_)))), 40);
    assert_eq!(count(&|b| matches!(b, Body::Mixer(Mixer::Attention(_)))), 8);
    // each role's shape, as the kind's roles say, against the checkpoint's
    let shape = |n: &str| -> Vec<usize> {
        let v = match t["tensors"].get(n) {
            Some(v) => v.clone(),
            None => {
                let run = t["runs"].as_array().unwrap().iter().find(|r| n.starts_with(r["prefix"].as_str().unwrap())).unwrap_or_else(|| panic!("{n}"));
                let rest = &n[run["prefix"].as_str().unwrap().len()..];
                run["tensors"][rest.split_once('.').unwrap().1].clone()
            }
        };
        v[1].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect()
    };
    let h = d.desc.hidden;
    for (b, roles) in d.desc.blocks.iter().zip(&d.blocks) {
        let of = |role: &str| match &roles.iter().find(|(r, _)| r == role).unwrap_or_else(|| panic!("no role {role}")).1 {
            Source::Tensor(n) => shape(n),
            Source::Experts(e) => [vec![e.len()], shape(&e[0][0])].concat(),
            Source::Rows(_) | Source::RowsOf(..) | Source::Transposed(_) => panic!("{role}"),
        };
        assert_eq!(of("norm_in"), [h]);
        match &b.body {
            Body::Mixer(Mixer::Mamba2(m)) => {
                let (inner, cd) = (m.heads * m.head_dim, m.heads * m.head_dim + 2 * m.groups * m.state);
                assert_eq!((inner, cd, m.conv, m.chunk, m.norm_eps), (8192, 10240, 4, 128, 1e-5));
                assert_eq!(of("in"), [inner + cd + m.heads, h]);
                // (the checkpoint's convolution weight has the one input channel of a depthwise convolution as its middle dimension)
                assert_eq!(of("conv"), [cd, 1, m.conv]);
                assert_eq!(of("conv_bias"), [cd]);
                for r in ["a_log", "d", "dt_bias"] {
                    assert_eq!(of(r), [m.heads], "{r}");
                }
                assert_eq!((of("norm"), of("o")), (vec![inner], vec![h, inner]));
            }
            Body::Mixer(Mixer::Attention(a)) => {
                assert!(a.rope.is_none() && a.window.is_none() && !a.bias && a.scale == (128f32).powf(-0.5));
                assert_eq!((of("q"), of("o")), (vec![a.heads * a.head_dim, h], vec![h, a.heads * a.v_head_dim]));
                assert_eq!((of("k"), of("v")), (vec![a.kv_heads * a.head_dim, h], vec![a.kv_heads * a.v_head_dim, h]));
                assert_eq!((a.heads, a.kv_heads, a.head_dim), (32, 2, 128));
            }
            Body::Ffn(Ffn::Moe(m)) => {
                let lat = m.latent.as_ref().unwrap().dim;
                let r = &m.router;
                assert_eq!((m.experts, r.top_k, r.renorm, r.scale, r.bias, r.score, lat), (512, 22, true, 5.0, true, RouterScore::Sigmoid, 1024));
                assert!(!m.expert.gated && m.expert.act == Act::Relu2 && !m.shared_gate && m.shared == Some(m.expert));
                assert_eq!((of("router"), of("choice_bias")), (vec![m.experts, h], vec![m.experts]));
                assert_eq!((of("latent_down"), of("latent_up")), (vec![lat, h], vec![h, lat]));
                assert_eq!((of("experts_up"), of("experts_down")), (vec![m.experts, 2688, lat], vec![m.experts, lat, 2688]));
                assert_eq!((of("shared_up"), of("shared_down")), (vec![5376, h], vec![h, 5376]));
            }
            b => panic!("{b:?}"),
        }
    }
}

/// The same model from the other ways a Nemotron config is written, and what the builder does not take.
#[test]
fn nemotron_config_variants() {
    use super::kinds::*;
    let cfg = fixture("nemotron_h_config.json");
    let base = describe(&cfg).unwrap();
    // the pattern as a list of types (as transformers' config class holds it; Nemotron 3.5's names too)
    for (m, a) in [("mamba", "attention"), ("linear_attention", "full_attention")] {
        let mut c = cfg.clone();
        let pat = c.as_object_mut().unwrap().remove("hybrid_override_pattern").unwrap();
        c["layers_block_type"] = pat
            .as_str()
            .unwrap()
            .chars()
            .map(|x| {
                if x == 'M' {
                    m
                } else if x == '*' {
                    a
                } else {
                    "moe"
                }
            })
            .collect::<Vec<_>>()
            .into();
        let d = describe(&c).unwrap_or_else(|e| panic!("{}", e.msg));
        assert!(d.desc == base.desc && d.names() == base.names());
    }
    // no latent: the experts read the model's width, and there are no projections around them
    let mut c = cfg.clone();
    c["moe_latent_size"] = serde_json::Value::Null;
    let d = describe(&c).unwrap();
    assert!(matches!(&d.desc.blocks[1].body, Body::Ffn(Ffn::Moe(m)) if m.latent.is_none()));
    assert!(!d.names().iter().any(|n| n.contains("latent_proj")) && d.names().len() == base.names().len() - 80);
    // the toy config of the project's GPU check
    let toy = describe(&fixture("nemotron_h_moetoy_config.json")).unwrap_or_else(|e| panic!("{}", e.msg));
    assert_eq!((toy.desc.blocks.len(), toy.desc.hidden, toy.mtp_layers), (6, 256, 0));
    // refusals: a key it does not know, a plain-MLP layer, a setting it does not take, layers that do not add up
    let refused = |edit: &dyn Fn(&mut serde_json::Value), says: &str| {
        let mut c = cfg.clone();
        edit(&mut c);
        let e = describe(&c).err().unwrap_or_else(|| panic!("{says}: described")).msg;
        assert!(e.contains(says), "{e}");
    };
    refused(&|c| c["mamba_extra_gate"] = true.into(), "mamba_extra_gate");
    refused(&|c| c["hybrid_override_pattern"] = "M-M*".into(), "'-'");
    refused(&|c| c["num_hidden_layers"] = 87.into(), "88 layers in the pattern");
    refused(&|c| c["use_conv_bias"] = false.into(), "use_conv_bias");
    refused(&|c| c["mamba_ssm_cache_dtype"] = "bfloat16".into(), "mamba_ssm_cache_dtype");
    refused(&|c| c["n_group"] = 4.into(), "n_group");
    refused(&|c| c["sliding_window"] = 4096.into(), "sliding window");
    refused(&|c| c["norm_eps"] = 1e-6.into(), "differ");
    refused(&|c| c["mlp_hidden_act"] = "tanh".into(), "tanh");
}

/// R3a's acceptance on the four checkpoints whose tensors the header fixtures list (tests/fixtures/headers: the Hub's safetensors headers): each released config.json
/// describes, from that file alone, a weight map whose every tensor is in its checkpoint, and that reads every tensor of the checkpoint but the parts the language
/// model does not run (vision and audio towers, their projectors, the next-token-prediction layers it does not use yet).
#[test]
fn released_configs_describe_their_checkpoints_tensors() {
    for (f, convention, mtp) in [
        ("gemma-4-12B-it", "Gemma", 0),
        ("Kimi-K3", "Kimi linear", 0),
        ("DeepSeek-V4-Flash-0731", "DeepSeek (release)", 1),
        ("Qwen3.8-27B", "Hugging Face (Llama-style)", 1),
    ] {
        let h = fixture(&format!("headers/{f}.json"));
        let cfg: serde_json::Value = serde_json::from_str(h["config"].as_str().unwrap()).unwrap();
        let d = describe(&cfg).unwrap_or_else(|e| panic!("{f}: {}", e.msg));
        d.check(&header_names(&h), true).unwrap_or_else(|e| panic!("{f}: {}", e.msg));
        assert_eq!((d.convention, d.mtp_layers), (convention, mtp), "{f}");
    }
}

/// The naming conventions held to every checkpoint index the fixtures have (the Hub's headers above; Qwen3.5 0.8B and 9B and GLM-5.3-Flash's
/// tensor lists): each names exactly the checkpoint's tensors, its quantized matrices' scales with them, less the parts it says it leaves. And the
/// check finds what differs: a tensor the index lacks, and one it has that nothing reads.
#[test]
fn conventions_name_the_checkpoints_tensors_exactly() {
    let q35 = fixture("qwen35_tensors.json");
    let cfgs = fixture("qwen35_configs.json");
    let mut cases: Vec<(String, serde_json::Value, BTreeSet<String>)> = ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-9B"]
        .iter()
        .map(|r| (r.to_string(), cfgs[*r].clone(), q35[*r].as_object().unwrap().keys().cloned().collect()))
        .collect();
    cases.push(("GLM-5.3-Flash".into(), fixture("glm5_next_config.json"), names_with_runs(&fixture("glm5_next_tensors.json"))));
    for (what, cfg, have) in &cases {
        let d = describe(cfg).unwrap();
        d.check(have, true).unwrap_or_else(|e| panic!("{what}: {}", e.msg));
        // (a tensor the description reads, gone from the index; a tensor no role reads, added to it)
        let some = d.names().into_iter().find(|n| n.contains("layers.0.")).unwrap();
        let mut less = have.clone();
        less.remove(&some);
        let e = d.check(&less, true).unwrap_err().msg;
        assert!(e.contains(&some) && e.contains(d.convention) && e.contains("not in the checkpoint") && e.contains("1 tensor:"), "{what}: {e}");
        let mut more = have.clone();
        more.insert(some.replace(".weight", ".bias").replace("layers.0.", "layers.0.extra."));
        assert!(d.check(&more, true).unwrap_err().msg.contains("not read by its description"), "{what}");
        assert!(d.check(&more, false).is_ok(), "{what}: a loader's check asks the tensors read alone");
    }
}

/// #333: every header fixture there is (tests/fixtures/headers; `sizes::record_header_fixtures` records the recent releases', #333's among them) describes
/// from its config.json (and, where the config does not say how a checkpoint stores something, its index: `describe_held`) a weight map its checkpoint holds and
/// reads it whole but the parts the language model does not run, or is refused in the words `LACKS` gives: the releases whose mechanisms the layer kinds do not
/// have yet (none now: DeepSeek-V4.1's CSA2, Engram and lagged mHC are kind parameters; Kimi-K3.1's sparse indexer and Qwen3.8-2.4T-A95B's stacked experts are
/// kinds the library has).
#[test]
fn every_recorded_release_describes_or_says_what_it_lacks() {
    const LACKS: &[(&str, &str)] = &[];
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/headers");
    let mut files: Vec<String> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
    files.sort();
    assert!(files.len() >= 4, "{files:?}");
    for f in files {
        let name = f.trim_end_matches(".json");
        let h = fixture(&format!("headers/{f}"));
        let cfg: serde_json::Value = serde_json::from_str(h["config"].as_str().unwrap()).unwrap_or_else(|e| panic!("{f}: its config: {e}"));
        // (each held to its own index: where a config does not say how its checkpoint stores something, the index does)
        let have = header_names(&h);
        match (super::describe::describe_held(&cfg, &have), LACKS.iter().find(|l| l.0 == name)) {
            (Ok(d), None) => d.check(&have, true).unwrap_or_else(|e| panic!("{f}: {}", e.msg)),
            (Err(e), Some((_, why))) => assert!(e.msg.contains(why), "{f}: {}", e.msg),
            (Ok(_), Some(_)) => panic!("{f} describes now: take it out of LACKS"),
            (Err(e), None) => panic!("{f}: {}", e.msg),
        }
    }
}

/// A loader's check (`describe_dir`) reads a Hub directory's index: a tensor the description reads is named where the index lacks it, a matrix is there in
/// either stored form, and a directory Glyd packed (its glyd.json) is not held to it.
#[test]
fn a_loader_names_a_missing_tensor_from_the_index() {
    let dir = std::env::temp_dir().join(format!("glyd-describe-dir-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let cfg = Fixture::load("qwen38_model").j["config"].clone();
    std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
    let d = describe(&cfg).unwrap();
    let mut names: Vec<String> = d.names().into_iter().collect();
    let index = |names: &[String]| {
        let map: serde_json::Map<String, serde_json::Value> = names.iter().map(|n| (n.clone(), "model-00001.safetensors".into())).collect();
        std::fs::write(dir.join("model.safetensors.index.json"), serde_json::json!({ "weight_map": map }).to_string()).unwrap();
    };
    index(&names);
    assert_eq!(super::describe::describe_dir(&dir).unwrap().0.names(), d.names());
    let gone = names.iter().position(|n| n.ends_with("layers.1.mlp.up_proj.weight")).unwrap();
    names[gone] = names[gone].replace(".weight", ".weight_packed");
    index(&names);
    assert!(super::describe::describe_dir(&dir).is_ok(), "a matrix packed as compressed-tensors packs it is there");
    names.remove(gone);
    index(&names);
    let e = super::describe::describe_dir(&dir).unwrap_err().msg;
    assert!(e.contains("config.json") && e.contains("layers.1.mlp.up_proj.weight") && e.contains("1 tensor:"), "{e}");
    std::fs::write(dir.join("glyd.json"), "{}").unwrap();
    assert!(super::describe::describe_dir(&dir).is_ok(), "a packed directory's names are its manifest's");
    let _ = std::fs::remove_dir_all(&dir);
}

/// #333, DSpark: DeepSeek-V4-Flash-0731's released config describes its drafter (`kinds::DSpark`): three of the decoder's layers with the window alone
/// (`n_mtp_layers` unsaid: the format's 3, as its checkpoint's mtp.0 to mtp.2), drafting blocks of 5 from main_x of layers 40 to 42, a Markov head of rank 256 and
/// the noise token 128799; its streams mHC's own (V4's, not lagged).
#[test]
fn deepseek_v4_0731s_dspark_drafter_is_described() {
    use super::kinds::*;
    let h = fixture("headers/DeepSeek-V4-Flash-0731.json");
    let d = describe(&serde_json::from_str(h["config"].as_str().unwrap()).unwrap()).unwrap();
    let [head] = &d.desc.mtp[..] else { panic!("{:?}", d.desc.mtp.len()) };
    let ds = head.dspark.as_ref().unwrap();
    assert_eq!((ds.block, ds.noise, ds.targets.clone(), ds.markov_rank, head.blocks.len()), (5, 128_799, vec![40, 41, 42], 256, 6));
    let Body::Mixer(Mixer::Compressed(c)) = &head.blocks[0].body else { panic!() };
    assert_eq!((c.ratio, c.window, c.indexer.is_some(), c.ape, c.q_head_norm), (0, 128, false, true, true));
    assert!(head.blocks.iter().all(|b| matches!(b.residual, Residual::Streams(StreamsP { lagged: false, .. }))));
    // (its tensors mapped: every mtp.* tensor of the checkpoint is a role's or a role's scale, and every role's is in the checkpoint)
    let (names, have) = (d.names(), header_names(&h));
    let drafter: BTreeSet<&String> = have.iter().filter(|n| n.starts_with("mtp.")).collect();
    assert_eq!(drafter.len(), 3 * (29 + 6 * 256) + 10, "three layers of 29 tensors and 256 experts of six, ten of the head");
    assert!(drafter.iter().all(|n| names.contains(*n)), "{:?}", drafter.iter().find(|n| !names.contains(**n)));
    d.check(&have, true).unwrap_or_else(|e| panic!("{}", e.msg));
    let [(blocks, own)] = &d.mtp[..] else { panic!() };
    assert_eq!((blocks.len(), own.len()), (6, 9));
}

/// A DSpark drafter's depth is the config's `n_mtp_layers`, else the format's default (three); a directory is held to its index in every family with the
/// drafter: one whose checkpoint lacks the last draft layer the count reads is refused, its tensors named (DeepSeek-V4-Flash-0731's config, which says no
/// count, and DeepSeek-V4.1's, which says one).
#[test]
fn a_drafters_depth_is_the_configs_else_the_formats_and_a_directory_is_held_to_it() {
    let depth = |d: &super::describe::Described| d.desc.mtp.first().filter(|h| h.dspark.is_some()).map(|h| h.blocks.len() / 2);
    let h = fixture("headers/DeepSeek-V4-Flash-0731.json");
    let v4: serde_json::Value = serde_json::from_str(h["config"].as_str().unwrap()).unwrap();
    assert!(v4.get("n_mtp_layers").is_none() && v4.get("dspark_block_size").is_some());
    assert_eq!(depth(&describe(&v4).unwrap()), Some(3));
    let mut said = v4.clone();
    said["n_mtp_layers"] = 2.into();
    assert_eq!(depth(&describe(&said).unwrap()), Some(2));
    let dir = std::env::temp_dir().join(format!("glyd-describe-draft-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for (what, cfg) in [("DeepSeek-V4", v4), ("DeepSeek-V4.1", fixture("deepseek_v41_config.json"))] {
        let d = describe(&cfg).unwrap();
        let n = depth(&d).unwrap_or_else(|| panic!("{what}: no drafter"));
        let last = format!("mtp.{}.", n - 1);
        std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
        let index = |keep: &dyn Fn(&String) -> bool| {
            let map: serde_json::Map<String, serde_json::Value> =
                d.names().into_iter().filter(|n| keep(n)).map(|n| (n, "model-00001.safetensors".into())).collect();
            std::fs::write(dir.join("model.safetensors.index.json"), serde_json::json!({ "weight_map": map }).to_string()).unwrap();
        };
        index(&|_| true);
        assert_eq!(depth(&super::describe::describe_dir(&dir).unwrap_or_else(|e| panic!("{what}: {}", e.msg)).0), Some(n), "{what}");
        index(&|name| !name.contains(&last));
        let e = super::describe::describe_dir(&dir).unwrap_err().msg;
        assert!(e.contains("config.json") && e.contains("not in the checkpoint") && e.contains(&last), "{what}: {e}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Every tensor of a header fixture by name: its dtype, shape and bytes.
fn header_tensors(h: &serde_json::Value) -> std::collections::BTreeMap<String, (String, Vec<usize>, u64)> {
    let mut out = std::collections::BTreeMap::new();
    for g in h["groups"].as_array().unwrap() {
        let (mask, dtype, bytes) = (g["mask"].as_str().unwrap(), g["dtype"].as_str().unwrap(), g["bytes"].as_u64().unwrap());
        let shape: Vec<usize> = g["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        let layers: Vec<String> = g["layers"].as_array().unwrap().iter().map(|l| l.as_u64().unwrap().to_string()).collect();
        let experts: Vec<String> = (0..g["experts"].as_u64().unwrap()).map(|e| e.to_string()).collect();
        for l in if mask.contains("{L}") { layers } else { vec![String::new()] } {
            for e in if mask.contains("{E}") { experts.clone() } else { vec![String::new()] } {
                out.insert(mask.replace("{L}", &l).replace("{E}", &e), (dtype.to_string(), shape.clone(), bytes));
            }
        }
    }
    out
}

/// Every tensor name of a header fixture: its groups' masks with each of their layers ({L}) and experts ({E}).
fn header_names(h: &serde_json::Value) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for g in h["groups"].as_array().unwrap() {
        let mask = g["mask"].as_str().unwrap();
        let layers: Vec<String> = g["layers"].as_array().unwrap().iter().map(|l| l.as_u64().unwrap().to_string()).collect();
        let experts: Vec<String> = (0..g["experts"].as_u64().unwrap()).map(|e| e.to_string()).collect();
        let ls = if mask.contains("{L}") { layers } else { vec![String::new()] };
        let es = if mask.contains("{E}") { experts } else { vec![String::new()] };
        for l in &ls {
            for e in &es {
                out.insert(mask.replace("{L}", l).replace("{E}", e));
            }
        }
    }
    out
}

/// #333: DeepSeek-V4.1-Flash's weight map, described from its released config alone, is its safetensors index (DeepSeek-V4.1-Flash-1017's file headers,
/// recorded by ranges from the Hub with no token and no weight read: 96,085 tensors in 48 files, 510,286,023,000 bytes, the index's total; the first
/// V4.1-Flash's index names the same tensors). Every tensor the description reads is there, at the shape the description sizes it by (FP4 experts two
/// values a byte), each quantized one with its scales; and every tensor there is read but the vision tower, its aligner, the image delimiters and the
/// routers' image-token biases (`names::DEEPSEEK_V41`'s unread parts): the decoder's 40 layers, the Engram tables of layers 1 and 14, and the three DSpark
/// layers of 128 experts.
#[test]
fn deepseek_v41_flashs_weight_map_is_its_index() {
    let h = fixture("headers/DeepSeek-V4.1-Flash-1017.json");
    assert_eq!((h["repo"].as_str(), h["index_total_size"].as_u64()), (Some("deepseek-ai/DeepSeek-V4.1-Flash-1017"), Some(510_286_023_000)));
    let d = describe(&serde_json::from_str(h["config"].as_str().unwrap()).unwrap()).unwrap();
    let have = header_names(&h);
    assert_eq!(have.len(), 96_085);
    d.check(&have, true).unwrap_or_else(|e| panic!("{}", e.msg));
    let names = d.names();
    let left: Vec<&String> = have.iter().filter(|n| !names.contains(*n)).collect();
    let tower = |n: &str| n.starts_with("vision.") || n.starts_with("aligner.") || n.starts_with("image_") || n.ends_with(".ffn.gate.bias_vl");
    assert!(left.iter().all(|n| tower(n)), "{:?}", left.iter().find(|n| !tower(n)));
    assert_eq!(left.iter().filter(|n| n.ends_with(".ffn.gate.bias_vl")).count(), 43);
    // (each tensor's shape: the header's groups by mask, layer and expert)
    let mut shape = std::collections::BTreeMap::new();
    for g in h["groups"].as_array().unwrap() {
        let sh: Vec<usize> = g["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        let mask = g["mask"].as_str().unwrap().to_string();
        let ls: Vec<String> = g["layers"].as_array().unwrap().iter().map(|l| l.as_u64().unwrap().to_string()).collect();
        for l in if mask.contains("{L}") { ls } else { vec![String::new()] } {
            for e in 0..g["experts"].as_u64().unwrap().max(1) {
                shape.insert(mask.replace("{L}", &l).replace("{E}", &e.to_string()), (g["dtype"].as_str().unwrap().to_string(), sh.clone()));
            }
        }
    }
    let mut sized = 0;
    for (n, want) in &d.shapes {
        let (dtype, got) = &shape[n];
        let want = if dtype == "I8" { vec![want[0], want[1] / 2] } else { want.clone() };
        assert_eq!(got, &want, "{n} ({dtype})");
        sized += 1;
    }
    // (every tensor read is sized but the scales, each beside its matrix)
    assert_eq!(sized + d.scales.len(), names.len());
    for (w, sc) in &d.scales {
        assert!(have.contains(w) && have.contains(sc), "{w}: {sc}");
    }
    // (the Engram tables: FP8 rows of 256 with a scale a run of 32, the primes past 16,000,000 counted)
    for (l, rows) in [(1, 384_006_168usize), (14, 384_016_682)] {
        assert_eq!(shape[&format!("layers.{l}.engram.embed.weight")], ("F8_E4M3".to_string(), vec![rows, 256]));
        assert_eq!(shape[&format!("layers.{l}.engram.embed.scale")], ("F8_E8M0".to_string(), vec![rows, 8]));
    }
}

/// #333, Kimi-K3.1: its sparse attention (`text_config.dsa_config`, the keys of vLLM's `KimiLinearDSAConfig`; the release's own, at its own numbers, is held to
/// its index by `kimi_k31_pros_weight_map_is_its_index`) on Kimi-K3's released config: each latent attention layer it names takes the sparse attention's indexer of a key a token in the gated form, its
/// keys RMS-normed at the model's eps (LayerNorm-ed with a bias where `index_k_norm_type` says so), and reads the indexer's tensors; the delta attention layers
/// are as they were. What is not described is refused: an indexer rotary, picks shared across layers, a layer that is no latent attention.
#[test]
fn kimi_k31s_sparse_attention_is_the_gated_indexer_with_its_key_norm() {
    use super::kinds::*;
    let h = fixture("headers/Kimi-K3.json");
    let base: serde_json::Value = serde_json::from_str(h["config"].as_str().unwrap()).unwrap();
    let full: Vec<u64> = base["text_config"]["linear_attn_config"]["full_attn_layers"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
    let eps = base["text_config"]["rms_norm_eps"].as_f64().unwrap() as f32;
    let with = |edit: &dyn Fn(&mut serde_json::Value)| {
        let mut c = base.clone();
        c["text_config"]["dsa_config"] = serde_json::json!({"dsa_layers": full, "index_topk": 2048, "index_head_dim": 64, "index_num_heads": 16,
            "index_rope_dim": 0, "index_k_norm_type": "rmsnorm", "indexer_use_fp8": false, "index_topk_freq": 1, "index_skip_topk_offset": 1});
        edit(&mut c["text_config"]["dsa_config"]);
        describe(&c)
    };
    let plain = describe(&base).unwrap();
    for (norm, kind) in [("rmsnorm", NormKind::Rms), ("layernorm", NormKind::Layer)] {
        let d = with(&|x| x["index_k_norm_type"] = norm.into()).unwrap_or_else(|e| panic!("{norm}: {}", e.msg));
        let mut latent = 0;
        for (b, was) in d.desc.blocks.iter().zip(&plain.desc.blocks) {
            match (&b.body, &was.body) {
                (Body::Mixer(Mixer::Mla(m)), Body::Mixer(Mixer::Mla(w))) => {
                    let ix = m.indexer.as_ref().unwrap();
                    assert_eq!((ix.heads, ix.head_dim, ix.top_k, ix.block, ix.shared), (16, 64, 2048, 1, false));
                    assert_eq!(ix.form, IndexForm::Gated { key: Norm { kind, ..Norm::rms(eps) } }, "{norm}");
                    assert_eq!(Mla { indexer: None, ..m.clone() }, *w, "the latent attention otherwise as Kimi-K3's");
                    latent += 1;
                }
                (x, y) => assert_eq!(x, y),
            }
        }
        assert_eq!(latent, full.len());
        let names = d.names();
        let l = full[0] - 1;
        for t in ["wq_b.weight", "wk.weight", "weights_proj.weight", "k_norm.weight"] {
            assert!(names.contains(&format!("language_model.model.layers.{l}.self_attn.indexer.{t}")), "{norm}: {t}");
        }
        assert_eq!(names.contains(&format!("language_model.model.layers.{l}.self_attn.indexer.k_norm.bias")), kind == NormKind::Layer, "{norm}");
        assert_eq!(names.len(), plain.names().len() + full.len() * if kind == NormKind::Layer { 5 } else { 4 }, "{norm}");
    }
    for (edit, want) in [
        (&(|x: &mut serde_json::Value| x["index_rope_dim"] = 64.into()) as &dyn Fn(&mut serde_json::Value), "index_rope_dim"),
        (&|x: &mut serde_json::Value| x["index_topk_freq"] = 2.into(), "index_topk_freq"),
        (&|x: &mut serde_json::Value| x["dsa_layers"] = serde_json::json!([1]), "layer 1"),
        (&|x: &mut serde_json::Value| x["index_k_norm_type"] = "l2".into(), "rmsnorm or layernorm"),
        (&|x: &mut serde_json::Value| x["index_share"] = true.into(), "index_share"),
    ] {
        let e = with(edit).unwrap_err().msg;
        assert!(e.contains(want), "{want}: {e}");
    }
}

/// Kimi-K3's matrices' shapes as the builder makes them from the config are the released checkpoint's (the header fixture's index): each layer's delta
/// attention's (q, k, v, the full-rank gate, the forget gate's two, beta, o), its latent attention's (q_a, q_b, the gate, kv_a, kv_b, o), the dense MLP's,
/// the router's, the latent's two projections and the shared experts' three, the embedding and the head, and every routed expert's three, whose codes as
/// released are two to a byte (a matrix of `[rows, cols]` is stored `[rows, cols / 2]`, its scales beside it): every matrix of the text model. The lowering
/// to the executor sizes every product by them and names its tensors with them; the planner sizes an expert by them (src/plan.rs `MoeShape`).
#[test]
fn kimi_k3s_matrices_shapes_are_the_released_checkpoints() {
    let h = fixture("headers/Kimi-K3.json");
    let d = describe(&serde_json::from_str(h["config"].as_str().unwrap()).unwrap()).unwrap();
    let mut header = std::collections::BTreeMap::new();
    for g in h["groups"].as_array().unwrap() {
        let (mask, shape) = (g["mask"].as_str().unwrap(), g["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect::<Vec<_>>());
        let layers: Vec<u64> = g["layers"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
        let experts = g["experts"].as_u64().unwrap() as usize;
        if layers.is_empty() {
            header.insert(mask.to_string(), shape);
        } else if !mask.contains("{E}") {
            header.extend(layers.iter().map(|l| (mask.replace("{L}", &l.to_string()), shape.clone())));
        } else {
            for l in &layers {
                let m = mask.replace("{L}", &l.to_string());
                header.extend((0..experts).map(|e| (m.replace("{E}", &e.to_string()), shape.clone())));
            }
        }
    }
    let mut packed = 0;
    for (n, sh) in &d.shapes {
        let stored = match d.scales.contains_key(n) {
            true => {
                packed += 1;
                vec![sh[0], sh[1] / 2]
            }
            false => sh.clone(),
        };
        assert_eq!(header.get(n), Some(&stored), "{n}");
    }
    assert_eq!(packed, 92 * 896 * 3, "every routed expert's three matrices");
    let scales: std::collections::BTreeSet<&String> = d.scales.values().collect();
    let matrices = header.iter().filter(|(n, s)| s.len() == 2 && s[0] > 1 && n.starts_with("language_model.") && !scales.contains(n));
    let unsaid: Vec<&String> = matrices.filter(|(n, _)| !d.shapes.contains_key(*n)).map(|x| x.0).collect();
    assert!(unsaid.is_empty(), "matrices of the text model with no shape said: {unsaid:?}");
}

/// Kimi-K3's experts as released (MXFP4 in groups of 32): each expert matrix is its packed codes and its scales; a quantization the description does not read is
/// refused with what differs, never read as bf16.
#[test]
fn kimi_k3s_packed_experts_are_named_and_other_packings_refused() {
    let h = fixture("headers/Kimi-K3.json");
    let cfg: serde_json::Value = serde_json::from_str(h["config"].as_str().unwrap()).unwrap();
    let d = describe(&cfg).unwrap();
    let w = "language_model.model.layers.1.block_sparse_moe.experts.7.w2.weight_packed";
    assert_eq!(d.scales.get(w).map(String::as_str), Some("language_model.model.layers.1.block_sparse_moe.experts.7.w2.weight_scale"));
    assert_eq!(d.scales.len(), 92 * 896 * 3, "the routed experts' matrices alone are packed");
    let refused = |edit: &dyn Fn(&mut serde_json::Value), want: &str| {
        let mut c = cfg.clone();
        edit(&mut c["text_config"]["quantization_config"]);
        let e = describe(&c).unwrap_err().msg;
        assert!(e.contains("quantization_config") && e.contains(want), "{e}");
    };
    refused(&|q| q["config_groups"]["group_0"]["weights"]["group_size"] = 16.into(), "groups of 32");
    refused(&|q| q["format"] = "nvfp4-pack-quantized".into(), "nvfp4");
    refused(&|q| q["ignore"] = serde_json::json!(["re:.*self_attn.*", "re:.*lm_head.*"]), "shared_experts");
}

/// Kimi-K3's layout's tensor at its place in a layer (or in the model), as a config's numbers make it: its dtype and shape. The experts' codes are two to a byte
/// and their scales a byte a group of 32. None for a name the layout has no tensor of.
fn kimi_tensor(c: &serde_json::Value, name: &str) -> Option<(&'static str, Vec<usize>)> {
    let t = &c["text_config"];
    let n = |k: &str| t[k].as_u64().unwrap() as usize;
    let (d, ne, lat, mi, inter, heads) =
        (n("hidden_size"), n("num_experts"), n("routed_expert_hidden_size"), n("moe_intermediate_size"), n("intermediate_size"), n("num_attention_heads"));
    let la = &t["linear_attn_config"];
    let (kh, kd, conv) = (la["num_heads"].as_u64()? as usize, la["head_dim"].as_u64()? as usize, la["short_conv_kernel_size"].as_u64()? as usize);
    let (qd, ql, kvl, vh) = (kh * kd, n("q_lora_rank"), n("kv_lora_rank"), n("v_head_dim"));
    let (nope, rope, shared) = (n("qk_nope_head_dim"), n("qk_rope_head_dim"), n("num_shared_experts") * mi);
    let (b, f, u) = ("BF16", "F32", "U8");
    let rest = name.strip_prefix("language_model.")?;
    let (layer, rest) = match rest.strip_prefix("model.layers.") {
        Some(r) => {
            let (l, r) = r.split_once('.')?;
            (Some(l.parse::<usize>().ok()?), r)
        }
        None => (None, rest),
    };
    let latent = layer.is_some_and(|l| la["full_attn_layers"].as_array().unwrap().iter().any(|x| x.as_u64() == Some(l as u64 + 1)));
    let big = if latent { heads * vh } else { qd };
    if let Some(r) = rest.strip_prefix("block_sparse_moe.experts.") {
        let (matrix, what) = r.split_once('.')?.1.split_once('.')?;
        let (rows, cols) = if matrix == "w2" { (lat, mi) } else { (mi, lat) };
        return match what {
            "weight_packed" => Some((u, vec![rows, cols / 2])),
            "weight_scale" => Some((u, vec![rows, cols / 32])),
            _ => None,
        };
    }
    Some(match rest {
        "lm_head.weight" | "model.embed_tokens.weight" => (b, vec![n("vocab_size"), d]),
        "model.norm.weight" | "model.output_attn_res_norm.weight" => (b, vec![d]),
        "model.output_attn_res_proj.weight" => (b, vec![1, d]),
        "input_layernorm.weight" | "post_attention_layernorm.weight" | "mlp_res_norm.weight" | "self_attention_res_norm.weight" => (b, vec![d]),
        "mlp_res_proj.weight" | "self_attention_res_proj.weight" => (b, vec![1, d]),
        "mlp.gate_proj.weight" | "mlp.up_proj.weight" => (b, vec![inter, d]),
        "mlp.down_proj.weight" => (b, vec![d, inter]),
        "block_sparse_moe.gate.weight" => (b, vec![ne, d]),
        "block_sparse_moe.gate.e_score_correction_bias" => (f, vec![ne]),
        "block_sparse_moe.routed_expert_down_proj.weight" => (b, vec![lat, d]),
        "block_sparse_moe.routed_expert_up_proj.weight" => (b, vec![d, lat]),
        "block_sparse_moe.routed_expert_norm.weight" => (b, vec![lat]),
        "block_sparse_moe.shared_experts.gate_proj.weight" | "block_sparse_moe.shared_experts.up_proj.weight" => (b, vec![shared, d]),
        "block_sparse_moe.shared_experts.down_proj.weight" => (b, vec![d, shared]),
        // (the delta attention's and the latent attention's, which both have a g_proj and an o_proj)
        "self_attn.q_proj.weight" | "self_attn.k_proj.weight" | "self_attn.v_proj.weight" => (b, vec![qd, d]),
        "self_attn.g_proj.weight" => (b, vec![big, d]),
        "self_attn.o_proj.weight" => (b, vec![d, big]),
        "self_attn.b_proj.weight" => (b, vec![kh, d]),
        "self_attn.f_a_proj.weight" => (b, vec![kd, d]),
        "self_attn.f_b_proj.weight" => (b, vec![qd, kd]),
        "self_attn.dt_bias" => (f, vec![qd]),
        // (as released: the delta heads' size, not the heads' number, which the loader reads the first of)
        "self_attn.A_log" | "self_attn.o_norm.weight" => (f, vec![kd]),
        "self_attn.q_conv1d.weight" | "self_attn.k_conv1d.weight" | "self_attn.v_conv1d.weight" => (f, vec![qd, 1, conv]),
        "self_attn.q_a_proj.weight" => (b, vec![ql, d]),
        "self_attn.q_a_layernorm.weight" => (b, vec![ql]),
        "self_attn.q_b_proj.weight" => (b, vec![heads * (nope + rope), ql]),
        "self_attn.kv_a_proj_with_mqa.weight" => (b, vec![kvl + rope, d]),
        "self_attn.kv_a_layernorm.weight" => (b, vec![kvl]),
        "self_attn.kv_b_proj.weight" => (b, vec![heads * (nope + vh), kvl]),
        // (the sparse attention's indexer, where the config has one)
        "self_attn.indexer.wq_b.weight" => {
            (b, vec![t["dsa_config"]["index_num_heads"].as_u64()? as usize * t["dsa_config"]["index_head_dim"].as_u64()? as usize, ql])
        }
        "self_attn.indexer.wk.weight" => (b, vec![t["dsa_config"]["index_head_dim"].as_u64()? as usize, d]),
        "self_attn.indexer.weights_proj.weight" => (b, vec![t["dsa_config"]["index_num_heads"].as_u64()? as usize, d]),
        "self_attn.indexer.k_norm.weight" | "self_attn.indexer.k_norm.bias" => (b, vec![t["dsa_config"]["index_head_dim"].as_u64()? as usize]),
        _ => return None,
    })
}

/// #333, Kimi-K3.1-Pro (released 2026-11-02; wrapper model type `kimi_k31`, text model Kimi-K3's `kimi_linear`): its released config.json describes, by its
/// keys alone, a weight map that is its safetensors index (497,340 tensors in 96 files, 1,561,217,377,280 bytes, the index's total: its file headers recorded
/// by ranges, no weight read). Every tensor the description reads is there at the dtype and shape the config's numbers make it, and every tensor there is read
/// but the vision tower and its projector (168, which the language model does not run). It differs from Kimi-K3 (headers/Kimi-K3.json) in 120 tensors, the
/// sparse indexer's five (wq_b, wk, weights_proj, a key norm's weight and bias) on each of the 24 latent attention layers, and in nothing else: the
/// description is Kimi-K3's but for `dsa_config`, whose keys are the indexer's heads (32 of 128), keys (LayerNorm-ed, with a bias), 2,048 picks, no rotary and
/// no shared picks.
#[test]
fn kimi_k31_pros_weight_map_is_its_index() {
    use super::kinds::*;
    let (h, h3) = (fixture("headers/Kimi-K3.1-Pro.json"), fixture("headers/Kimi-K3.json"));
    assert_eq!(h["repo"].as_str(), Some("moonshotai/Kimi-K3.1-Pro"));
    let (cfg, cfg3): (serde_json::Value, serde_json::Value) =
        (serde_json::from_str(h["config"].as_str().unwrap()).unwrap(), serde_json::from_str(h3["config"].as_str().unwrap()).unwrap());
    assert_eq!((cfg["model_type"].as_str(), cfg3["model_type"].as_str()), (Some("kimi_k31"), Some("kimi_k3")));
    let (d, d3) = (describe(&cfg).unwrap_or_else(|e| panic!("{}", e.msg)), describe(&cfg3).unwrap());
    let (t, t3) = (header_tensors(&h), header_tensors(&h3));
    let have: BTreeSet<String> = t.keys().cloned().collect();
    assert_eq!((t.len(), t3.len()), (497_340, 497_220));
    assert_eq!(
        t.values().map(|x| x.2).sum::<u64>(),
        1_561_217_377_280,
        "the index's total size (its metadata, read by a range request: the 59 MB index is over the recorder's text limit)"
    );
    d.check(&have, true).unwrap_or_else(|e| panic!("{}", e.msg));
    // (what it does not read: the vision tower's 27 blocks of six tensors and its three, and the projector's three)
    let names = d.names();
    let left: Vec<&String> = have.iter().filter(|n| !names.contains(*n)).collect();
    assert_eq!(left.len(), 27 * 6 + 3 + 3);
    assert!(
        left.iter().all(|n| n.starts_with("vision_tower.") || n.starts_with("mm_projector.")),
        "{:?}",
        left.iter().find(|n| !n.starts_with("vision_tower."))
    );
    assert_eq!((names.len(), d.mtp_layers, d.scales.len()), (497_172, 0, 92 * 896 * 3));
    // (every tensor read at its dtype and shape: the config's numbers, and the description's where it says them)
    for n in &names {
        let (dtype, shape, _) = &t[n];
        let want = kimi_tensor(&cfg, n).unwrap_or_else(|| panic!("{n}: no tensor of the layout has this name"));
        assert_eq!((dtype.as_str(), shape), (want.0, &want.1), "{n}");
        // (a routed expert's matrix: its codes two to a byte, `[rows, cols / 2]`)
        if let Some(s) = d.shapes.get(n) {
            let stored = if d.scales.contains_key(n) { vec![s[0], s[1] / 2] } else { s.clone() };
            assert_eq!(&stored, shape, "{n}: the description's shape");
        }
    }
    // (against Kimi-K3: the tensors it adds, each of the 24 latent attention layers' indexer, and the rest the same at the same dtype and shape)
    let full: Vec<usize> =
        cfg["text_config"]["linear_attn_config"]["full_attn_layers"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize - 1).collect();
    assert_eq!((full.len(), cfg["text_config"]["dsa_config"]["dsa_layers"].as_array().unwrap().len()), (24, 24));
    let added: BTreeSet<&String> = have.iter().filter(|n| !t3.contains_key(*n)).collect();
    let want: BTreeSet<String> = full
        .iter()
        .flat_map(|l| {
            ["wq_b.weight", "wk.weight", "weights_proj.weight", "k_norm.weight", "k_norm.bias"]
                .map(|x| format!("language_model.model.layers.{l}.self_attn.indexer.{x}"))
        })
        .collect();
    assert_eq!(added, want.iter().collect::<BTreeSet<_>>());
    assert!(t3.keys().all(|n| t.contains_key(n)) && t3.iter().all(|(n, v)| &t[n] == v), "the other tensors are Kimi-K3's");
    let names3 = d3.names();
    assert_eq!(names.iter().filter(|n| !names3.contains(*n)).collect::<BTreeSet<_>>(), want.iter().collect::<BTreeSet<_>>());
    assert!(names3.iter().all(|n| names.contains(n)));
    // (the description: Kimi-K3's but for the indexer on the layers `dsa_config` names; the config's other differences (the wrapper's name, the ignore list's two
    // more entries for tensors that are bf16 in both) leave it as it was)
    let mut plain = cfg.clone();
    plain["text_config"].as_object_mut().unwrap().remove("dsa_config");
    let p = describe(&plain).unwrap();
    assert!(p.desc == d3.desc && p.names() == d3.names() && p.shapes == d3.shapes && p.scales == d3.scales, "without dsa_config it is Kimi-K3's");
    let ix = Indexer {
        heads: 32,
        head_dim: 128,
        top_k: 2048,
        block: 1,
        shared: false,
        form: IndexForm::Gated { key: Norm { kind: NormKind::Layer, ..Norm::rms(cfg["text_config"]["rms_norm_eps"].as_f64().unwrap() as f32) } },
        rope: 0,
        rope_style: None,
        q_in: None,
        pos_scale: false,
    };
    let mut latent = 0;
    for (b, was) in d.desc.blocks.iter().zip(&d3.desc.blocks) {
        match (&b.body, &was.body) {
            (Body::Mixer(Mixer::Mla(m)), Body::Mixer(Mixer::Mla(w))) => {
                assert_eq!((m.indexer.as_ref(), &Mla { indexer: None, ..m.clone() }), (Some(&ix), w));
                latent += 1;
            }
            (x, y) => assert_eq!(x, y),
        }
    }
    assert_eq!(latent, 24);
}

/// #333, Qwen3.8-2.4T-A95B's weight map, described from its released config.json (the Qwen3.5 layout with experts, `Qwen3_5MoeForCausalLM`), is its safetensors
/// index (its 213 files' headers, recorded by ranges with no weight read: 1,609 tensors, 4,892,365,451,008 bytes, the index's total). The config does not
/// say how a checkpoint stores the experts: the description's form for one with no index names each expert's three matrices a tensor (as the layout's earlier
/// releases are saved), and this release holds each layer's 512 experts as two stacked tensors (`mlp.experts.gate_up_proj` [512, 4096, 8192] and
/// `down_proj` [512, 8192, 2048], transformers' own form), so a description held to its index (`describe_held`, which a directory's loader is given) names those.
/// Every tensor it reads is there at the shape it says; every tensor there is read but the draft layer (`mtp.*`, 19 tensors: its front's norms and product
/// and one full-attention layer with its 512 experts, which this layout's description counts and does not read, its `mtp` empty).
#[test]
fn qwen38_2_4ts_weight_map_is_its_index() {
    use super::describe::{describe_held, Source};
    let h = fixture("headers/Qwen3.8-2.4T-A95B.json");
    assert_eq!((h["repo"].as_str(), h["index_total_size"].as_u64()), (Some("Qwen/Qwen3.8-2.4T-A95B"), Some(4_892_365_451_008)));
    let cfg: serde_json::Value = serde_json::from_str(h["config"].as_str().unwrap()).unwrap();
    let (t, have) = (header_tensors(&h), header_names(&h));
    assert_eq!((t.len(), t.values().map(|x| x.2).sum::<u64>()), (1609, 4_892_365_451_008));
    assert!(t.values().all(|x| x.0 == "BF16"));
    // (the config alone: each expert's matrices a tensor, 92 layers of 512 experts of three, none of which the checkpoint has)
    let alone = describe(&cfg).unwrap();
    let e = alone.check(&have, true).unwrap_err().msg;
    assert!(e.contains(&format!("{} tensors", 92 * 512 * 3)), "{e}");
    let d = describe_held(&cfg, &have).unwrap_or_else(|e| panic!("{}", e.msg));
    d.check(&have, true).unwrap_or_else(|e| panic!("{}", e.msg));
    assert_eq!(d.desc.blocks, alone.desc.blocks, "the same kinds, the experts' stored form apart");
    let names = d.names();
    assert_eq!(names.len(), 1609 - 19);
    let left: Vec<&String> = have.iter().filter(|n| !names.contains(*n)).collect();
    assert!(left.len() == 19 && left.iter().all(|n| n.starts_with("mtp.")), "{left:?}");
    assert_eq!((d.mtp_layers, d.desc.mtp.len()), (1, 0));
    // (every tensor read at its shape: a shape said for each, and the header's)
    assert_eq!(d.shapes.keys().cloned().collect::<BTreeSet<_>>(), names, "a shape for every tensor described");
    for (n, s) in &d.shapes {
        assert_eq!(&t[n].1, s, "{n}");
    }
    let role = |l: usize, r: &str| match d.blocks[2 * l + 1].iter().find(|(x, _)| x == r) {
        Some((_, Source::Tensor(n))) => n.clone(),
        other => panic!("layer {l} {r}: {other:?}"),
    };
    assert_eq!(role(91, "experts_gate_up"), "model.layers.91.mlp.experts.gate_up_proj");
    assert_eq!((d.shapes[&role(0, "experts_gate_up")].clone(), d.shapes[&role(0, "experts_down")].clone()), (vec![512, 4096, 8192], vec![512, 8192, 2048]));
    // (a directory is held to its own index: this release's names read stacked, a per-expert checkpoint's (the random case of the layout) per expert, a directory
    // with no index as the config alone says)
    let dir = std::env::temp_dir().join(format!("glyd-describe-stored-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let held = |cfg: &serde_json::Value, names: &BTreeSet<String>| {
        std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
        let map: serde_json::Map<String, serde_json::Value> = names.iter().map(|n| (n.clone(), "model-00001.safetensors".into())).collect();
        std::fs::write(dir.join("model.safetensors.index.json"), serde_json::json!({ "weight_map": map }).to_string()).unwrap();
        super::describe::describe_dir(&dir).map(|x| x.0)
    };
    let from_dir = held(&cfg, &have).unwrap_or_else(|e| panic!("{}", e.msg));
    assert_eq!((from_dir.names(), from_dir.shapes), (names, d.shapes.clone()));
    let tiny = fixture("qwen35moe_tiny_config.json");
    let per_expert = describe(&tiny).unwrap();
    let from_dir = held(&tiny, &per_expert.names()).unwrap_or_else(|e| panic!("{}", e.msg));
    assert!(from_dir.names() == per_expert.names() && from_dir.shapes == per_expert.shapes);
    assert!(matches!(from_dir.blocks[1].iter().find(|(r, _)| r == "experts_gate_up"), Some((_, Source::Experts(e))) if e.len() == 32));
    let _ = std::fs::remove_dir_all(&dir);
}

/// #333, Qwen3.8-2.4T-A95B's stacked experts as the lossy route's packer reads them (`stacked_rows`): each of the 92 x 512 experts' three matrices the
/// config alone names (the names its save gives them) is its rows of the layer's stacked tensor in its recorded headers, gate then up then the next expert,
/// every row of the 92 x 2 stacked tensors one matrix's; the release's lossy plan reads each from those rows (`pack::plan`). A description held
/// to a checkpoint of a tensor an expert's matrix gives none, and one whose experts do not cover the stacked tensor's rows is refused.
#[test]
fn qwen38_2_4ts_stacked_experts_are_rows_of_its_tensors() {
    use super::describe::{describe_held, stacked_rows};
    let h = fixture("headers/Qwen3.8-2.4T-A95B.json");
    let cfg: serde_json::Value = serde_json::from_str(h["config"].as_str().unwrap()).unwrap();
    let (t, have) = (header_tensors(&h), header_names(&h));
    let (alone, held) = (describe(&cfg).unwrap(), describe_held(&cfg, &have).unwrap());
    let rows = stacked_rows(&alone, &held).unwrap();
    assert_eq!(rows.len(), 92 * 512 * 3);
    let at = |n: &str| rows[n].clone();
    let gu = "model.layers.7.mlp.experts.gate_up_proj".to_string();
    assert_eq!(at("model.layers.7.mlp.experts.0.gate_proj.weight"), (gu.clone(), 0, 2048));
    assert_eq!(at("model.layers.7.mlp.experts.0.up_proj.weight"), (gu.clone(), 2048, 4096));
    assert_eq!(at("model.layers.7.mlp.experts.511.up_proj.weight"), (gu, 511 * 4096 + 2048, 512 * 4096));
    assert_eq!(at("model.layers.91.mlp.experts.300.down_proj.weight"), ("model.layers.91.mlp.experts.down_proj".into(), 300 * 8192, 301 * 8192));
    // (every row of every stacked tensor one matrix's)
    let mut covered: std::collections::BTreeMap<&str, usize> = Default::default();
    for (tensor, a, b) in rows.values() {
        *covered.entry(tensor.as_str()).or_default() += b - a;
    }
    assert_eq!(covered.len(), 92 * 2);
    for (n, r) in covered {
        assert_eq!(r as u64, t[n].1[0] as u64 * t[n].1[1] as u64, "{n}");
    }
    // (a checkpoint of a tensor an expert's matrix: none; a stacked tensor the experts do not cover: refused)
    assert!(stacked_rows(&alone, &alone).unwrap().is_empty());
    let mut short = held.clone();
    short.shapes.insert("model.layers.0.mlp.experts.down_proj".into(), vec![512, 8192 + 64, 2048]);
    assert!(stacked_rows(&alone, &short).unwrap_err().msg.contains("expert 0's matrices take 8192 of its 8256 rows"));
    let qwen35 = crate::support::row(crate::support::Family::Qwen35).saves.unwrap();
    let mc = crate::runtime::config::ModelConfig::from_value(&cfg).unwrap();
    let plan = crate::pack::plan(&mc, glyd_gpu::save::Layout::Int6, &qwen35, Some(&held), &|_| false, &|_| None).unwrap();
    assert!(plan.contains(
        r#"{"linear": ["model.layers.7.mlp.experts.511.gate_proj", "model.layers.7.mlp.experts.511.up_proj"], "from": [{"tensor": "model.layers.7.mlp.experts.gate_up_proj", "rows": [2093056, 2095104]}, {"tensor": "model.layers.7.mlp.experts.gate_up_proj", "rows": [2095104, 2097152]}]}"#
    ));
    assert!(!plan.contains(r#""from": ["model.layers.7.mlp.experts.511.gate_proj""#));
}

/// Every config of tests/fixtures describes from its own file, the toy and tiny ones as the released (Ibis's Qwen3.8-Flash-Next too: 48 layers of two blocks, and
/// the n-gram embedding's block before its second layer).
#[test]
fn every_fixture_config_describes() {
    for (f, blocks) in [
        ("gemma4_12b_config.json", 96),
        ("glm5_next_config.json", 90),
        ("glm5_next_tiny_config.json", 10),
        ("glm5_next_toy_config.json", 16),
        ("nemotron_h_config.json", 88),
        ("nemotron_h_moetoy_config.json", 6),
        ("qwen4x_config.json", 97),
    ] {
        let d = describe(&fixture(f)).unwrap_or_else(|e| panic!("{f}: {}", e.msg));
        assert_eq!(d.desc.blocks.len(), blocks, "{f}");
    }
    for (repo, c) in fixture("qwen35_configs.json").as_object().unwrap() {
        describe(c).unwrap_or_else(|e| panic!("{repo}: {}", e.msg));
    }
}

/// Every model config the fixtures hold, by a name: tests/fixtures' (each entry of qwen35_configs.json apart), the headers' and the tiny models' of
/// tests/runtime.
fn fixture_configs() -> Vec<(String, serde_json::Value)> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");
    let files = |dir: &str| -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(root.join(dir)).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
        v.sort();
        v
    };
    let mut out = Vec::new();
    for f in files("fixtures").into_iter().filter(|f| f.ends_with(".json") && f.contains("config") && !f.contains("generation_config")) {
        let v = fixture(&f);
        if f == "qwen35_configs.json" {
            out.extend(v.as_object().unwrap().iter().map(|(repo, c)| (format!("{f}:{repo}"), c.clone())));
        } else {
            out.push((f, v));
        }
    }
    for f in files("fixtures/headers") {
        let h = fixture(&format!("headers/{f}"));
        out.push((format!("headers/{f}"), serde_json::from_str(h["config"].as_str().unwrap()).unwrap()));
    }
    for f in files("runtime").into_iter().filter(|f| f.ends_with("_model.json")) {
        out.push((format!("runtime/{f}"), Fixture::load(f.trim_end_matches(".json")).j["config"].clone()));
    }
    out
}

/// A description as two digests: its layer kinds (with the draft layers it counts), and its weight map (the tensors, their roles, scales and shapes).
fn digests(d: &super::describe::Described) -> (String, String) {
    use sha2::{Digest, Sha256};
    let h = |s: String| Sha256::digest(s.as_bytes()).iter().take(8).map(|b| format!("{b:02x}")).collect::<String>();
    (h(format!("{:?} {}", d.desc, d.mtp_layers)), h(format!("{:?}", (&d.embed, &d.final_norm, &d.head, &d.blocks, &d.extra, &d.scales, &d.shapes))))
}

/// Each fixture config's description, from its config.json alone, is the one recorded in tests/fixtures/descriptions.json (its blocks, its tensors and the
/// digests of its kinds and weight map), and a config that is refused is refused with the words recorded: a change to the builder that changes any
/// fixture's description fails here, by name. GLYD_BLESS_DESCRIPTIONS=1 writes the record anew (for a change meant to alter one).
#[test]
fn fixture_descriptions_are_unchanged() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/descriptions.json");
    let mut now = serde_json::Map::new();
    for (name, cfg) in fixture_configs() {
        let mut entry = match describe(&cfg) {
            Ok(d) => {
                let (kinds, weights) = digests(&d);
                serde_json::json!({ "blocks": d.desc.blocks.len(), "tensors": d.names().len(), "kinds": kinds, "weights": weights })
            }
            Err(e) => serde_json::json!({ "refused": e.msg }),
        };
        // (and what the planner and the refusals read of the same file: the shape fit prices, the family run takes it as)
        let text = cfg.to_string();
        entry["plan"] = match crate::plan::Shape::from_config(&text) {
            Ok(s) => {
                use sha2::{Digest, Sha256};
                Sha256::digest(format!("{s:?}").as_bytes()).iter().take(8).map(|b| format!("{b:02x}")).collect::<String>().into()
            }
            Err(e) => format!("refused: {e}").into(),
        };
        entry["family"] = match crate::support::family(&text, "m") {
            Ok(f) => format!("{f:?}").into(),
            Err(r) => format!("refused: {}", r.what).into(),
        };
        now.insert(name, entry);
    }
    if std::env::var_os("GLYD_BLESS_DESCRIPTIONS").is_some() {
        std::fs::write(&path, serde_json::to_string_pretty(&now).unwrap() + "\n").unwrap();
        return;
    }
    let was: serde_json::Map<String, serde_json::Value> = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    for (name, e) in &now {
        assert_eq!(was.get(name), Some(e), "{name}: its description is not the one recorded (tests/fixtures/descriptions.json)");
    }
    let gone: Vec<_> = was.keys().filter(|k| !now.contains_key(*k)).collect();
    assert!(gone.is_empty(), "recorded, and no fixture now: {gone:?}");
    // (every config describes)
    let refused: Vec<_> = now.iter().filter(|(_, e)| e.get("refused").is_some()).map(|(n, _)| n.as_str()).collect();
    assert!(refused.is_empty(), "{refused:?}");
}

/// Nemotron 3's released configs (Nano, 3.5 Lightning) say `moe_shared_expert_overlap: true`, a schedule of the shared experts beside the exchange between GPUs,
/// not arithmetic: the description is the same whichever it says.
#[test]
fn nemotron_h_shared_expert_overlap_is_read_either_way() {
    let mut cfg = fixture("nemotron_h_config.json");
    let off = format!("{:#?}", describe(&cfg).unwrap().desc);
    cfg["moe_shared_expert_overlap"] = serde_json::Value::Bool(true);
    let on = describe(&cfg).unwrap_or_else(|e| panic!("{}", e.msg));
    assert_eq!(format!("{:#?}", on.desc), off);
}

/// Nemotron 3.5 Lightning's released config (the recorded header, tests/fixtures/headers/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.json) says
/// `mtp_layers_block_type` (["attention", "moe"]: the layers of its multi-token-prediction head, one step of `num_nextn_predict_layers` 1). The head refused it for that
/// key. It describes: 52 layers (23 Mamba-2, 6 attention, 23 mixtures of experts with no latent), the draft head counted and its tensors (`mtp.*`) left, every other tensor of the
/// checkpoint read at the shape the config makes it; and what the key says changes nothing of the description (absent, another list, the older string form).
#[test]
fn nemotron_3_5_lightning_describes_and_leaves_its_draft_head() {
    use super::kinds::*;
    let h = fixture("headers/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.json");
    let cfg: serde_json::Value = serde_json::from_str(h["config"].as_str().unwrap()).unwrap();
    assert_eq!(cfg["mtp_layers_block_type"], serde_json::json!(["attention", "moe"]));
    let d = describe(&cfg).unwrap_or_else(|e| panic!("{}", e.msg));
    let count = |f: &dyn Fn(&Body) -> bool| d.desc.blocks.iter().filter(|b| f(&b.body)).count();
    assert_eq!(d.desc.blocks.len(), 52);
    assert_eq!(count(&|b| matches!(b, Body::Mixer(Mixer::Mamba2(_)))), 23);
    assert_eq!(count(&|b| matches!(b, Body::Mixer(Mixer::Attention(_)))), 6);
    assert_eq!(count(&|b| matches!(b, Body::Ffn(Ffn::Moe(m)) if m.experts == 128 && m.latent.is_none())), 23);
    // (the head: counted, none in the description, its tensors the unread `mtp.`)
    assert_eq!((d.mtp_layers, d.desc.mtp.len(), d.mtp.len(), d.unread.as_slice()), (1, 0, 0, &["mtp.".to_string()][..]));
    let have = header_names(&h);
    d.check(&have, true).unwrap_or_else(|e| panic!("{}", e.msg));
    let (names, draft): (Vec<&String>, Vec<&String>) = have.iter().partition(|n| !n.starts_with("mtp."));
    assert_eq!(
        draft.len(),
        4 + 2 * 128 + 2 + 2 + 4 + 2,
        "the head's tensors: eh_proj, enorm, hnorm, final_layernorm; 128 experts of two; the router's two; the shared expert's two; q k v o; two norms"
    );
    assert!(draft.iter().all(|n| !d.names().contains(*n)) && names.iter().all(|n| d.names().contains(*n)));
    // (each tensor read, at the header's shape)
    let mut shape = std::collections::BTreeMap::new();
    for g in h["groups"].as_array().unwrap() {
        let sh: Vec<usize> = g["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        let mask = g["mask"].as_str().unwrap().to_string();
        let ls: Vec<String> = g["layers"].as_array().unwrap().iter().map(|l| l.as_u64().unwrap().to_string()).collect();
        for l in if mask.contains("{L}") { ls } else { vec![String::new()] } {
            for e in 0..g["experts"].as_u64().unwrap().max(1) {
                shape.insert(mask.replace("{L}", &l).replace("{E}", &e.to_string()), sh.clone());
            }
        }
    }
    assert_eq!(d.shapes.keys().cloned().collect::<BTreeSet<_>>(), d.names(), "a shape for every tensor described");
    for (n, want) in &d.shapes {
        assert_eq!(&shape[n], want, "{n}");
    }
    // (the key is read and says nothing the description uses)
    let same = format!("{:?}", (&d.desc, &d.blocks, &d.shapes, d.mtp_layers));
    for edit in [
        |c: &mut serde_json::Value| drop(c.as_object_mut().unwrap().remove("mtp_layers_block_type")),
        |c: &mut serde_json::Value| c["mtp_layers_block_type"] = serde_json::json!(["linear_attention", "mlp"]),
        |c: &mut serde_json::Value| {
            c.as_object_mut().unwrap().remove("mtp_layers_block_type");
            c["mtp_hybrid_override_pattern"] = "*E".into();
        },
    ] {
        let mut c = cfg.clone();
        edit(&mut c);
        let e = describe(&c).unwrap_or_else(|e| panic!("{}", e.msg));
        assert_eq!(format!("{:?}", (&e.desc, &e.blocks, &e.shapes, e.mtp_layers)), same);
    }
}

/// The tiny Qwen3.8-Flash-Next of tests/fixtures/qwen4x_tiny_manifest.json (the tensors transformers saved for that config: Ibis's random model of the `qwen4x`
/// logits case is made from it): its config describes exactly those tensors, each at its saved shape.
#[test]
fn ibis_tiny_config_describes_its_tensors() {
    let m = fixture("qwen4x_tiny_manifest.json");
    let cfg: serde_json::Value = serde_json::from_str(m["config"].as_str().unwrap()).unwrap();
    let d = describe(&cfg).unwrap_or_else(|e| panic!("{}", e.msg));
    let saved: std::collections::BTreeMap<String, Vec<usize>> = m["tensors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| (t[0].as_str().unwrap().to_string(), t[2].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect()))
        .collect();
    assert_eq!(d.names(), saved.keys().cloned().collect::<BTreeSet<_>>());
    assert_eq!(d.shapes, saved, "each tensor's shape as the config makes it");
    // (8 layers of two blocks, the n-gram embedding's before the second; four streams; the head's collapse is the final mixer)
    use super::kinds::*;
    assert_eq!((d.desc.blocks.len(), d.desc.embed.streams, d.desc.head.collapse, d.mtp_layers), (17, 4, Collapse::GatedMix, 0));
    let Body::Ngram(n) = &d.desc.blocks[2].body else { panic!("{:?}", d.desc.blocks[2].body) };
    assert_eq!((n.order, n.heads, n.dim, n.base, n.seed, n.index, n.eos, n.conv, n.dilation), (3, 8, 256, 1000, 1234, 0, 2000, 4, 3));
    let Body::Mixer(Mixer::Attention(a)) = &d.desc.blocks[7].body else { panic!("{:?}", d.desc.blocks[7].body) };
    let ix = a.indexer.as_ref().unwrap();
    assert_eq!((ix.heads, ix.head_dim, ix.top_k, ix.block, a.rope.as_ref().unwrap().dim), (4, 128, 8, 4, 64));
    assert!(d
        .desc
        .blocks
        .iter()
        .all(|b| matches!(b.residual, Residual::GatedStreams(GatedStreamsP { streams: 4, lowrank: 64 })) || matches!(b.body, Body::Ngram(_))));
}

/// The vision tower kind of a multimodal config (`describe::with_tower`): the language model alone (`language_model_only`) and a config with no tower describe
/// none; a tower whose features go into the text's layers (deepstack), a text rotary whose M-RoPE sections are not interleaved, an image token outside the
/// vocabulary, rows of another width than the text's, and a key the builder does not know are refused in words, each naming what it is.
#[test]
fn a_vision_tower_is_described_or_refused_in_words() {
    let rel = fixture("qwen4x_config.json");
    let edited = |f: &dyn Fn(&mut serde_json::Value)| {
        let mut c = rel.clone();
        f(&mut c);
        describe(&c)
    };
    let d = edited(&|c| c["language_model_only"] = true.into()).unwrap();
    assert!(d.desc.vision.is_none() && d.vision.is_none() && d.unread == ["model.visual."]);
    let d = edited(&|c| {
        c.as_object_mut().unwrap().remove("vision_config");
    })
    .unwrap();
    assert!(d.desc.vision.is_none());
    for (f, words) in [
        (
            Box::new(|c: &mut serde_json::Value| c["vision_config"]["deepstack_visual_indexes"] = serde_json::json!([5, 11]))
                as Box<dyn Fn(&mut serde_json::Value)>,
            "deepstack",
        ),
        (Box::new(|c: &mut serde_json::Value| c["text_config"]["rope_parameters"]["mrope_interleaved"] = false.into()), "interleaved M-RoPE"),
        (Box::new(|c: &mut serde_json::Value| c["image_token_id"] = 300_000.into()), "image_token_id"),
        (Box::new(|c: &mut serde_json::Value| c["vision_config"]["out_hidden_size"] = 2048.into()), "out_hidden_size 2048"),
        (Box::new(|c: &mut serde_json::Value| c["vision_config"]["window_size"] = 112.into()), "window_size"),
        (Box::new(|c: &mut serde_json::Value| c["vision_config"]["num_position_embeddings"] = 2000.into()), "a square table"),
    ] {
        let e = edited(&*f).err().unwrap_or_else(|| panic!("{words}: described")).msg;
        assert!(e.contains(words), "{words}: {e}");
    }
}

/// Ibis's released config (tests/fixtures/qwen4x_config.json) describes every tensor Ibis's engine loads (src/qwen4x.rs: each layer's matrices, the gated
/// residuals' and the final mixer's, the n-gram embedding's products; the MTP layer's, its front's and its final mixer's as the draft head) and the vision
/// tower as the release's own engine read it (src/vision.rs: 27 blocks of 16 heads over 1,152, 4,304 neurons, a 48 x 48 position table, patches of 3 x 2 x
/// 16 x 16 merged 2 x 2 into rows of 2,560, image token 248,056), nothing left; and what it does not take is refused in words.
#[test]
fn ibis_released_config() {
    use super::kinds::*;
    let cfg = fixture("qwen4x_config.json");
    let d = describe(&cfg).unwrap();
    let c = crate::qwen4x::ConfigQx::parse(&cfg.to_string(), "x", None).unwrap();
    let names = d.names();
    let mut loads: Vec<String> = vec!["lm_head.weight".into(), format!("{}embed_tokens.weight", c.prefix)];
    let pack = |p: crate::qwen35::Pack35| p.parts.into_iter().map(|x| x.0).collect::<Vec<_>>();
    for l in 0..c.layers {
        loads.extend(c.layer_packs(l).into_iter().flat_map(pack));
        for g in ["attn_hyper_connection", "mlp_hyper_connection"] {
            let m = format!("{}.{g}", c.layer_name(l));
            loads.extend(c.gr_packs(&m).into_iter().flat_map(pack));
            loads.extend([format!("{m}.hc_norm.weight"), format!("{m}.block_inject_weight.weight")]);
        }
    }
    loads.extend(c.gr_packs(&c.mixer_name(false)).into_iter().flat_map(pack));
    loads.extend(c.ple_pack().into_iter().flat_map(pack));
    // (the draft head: the MTP layer's matrices, norms and gated residuals as a layer's, its front, its own final mixer)
    let head = c.layer_name(c.layers);
    let packs = c.layer_packs(c.layers);
    loads.extend(packs.into_iter().flat_map(pack));
    for g in ["attn_hyper_connection", "mlp_hyper_connection"] {
        let m = format!("{head}.{g}");
        loads.extend(c.gr_packs(&m).into_iter().flat_map(pack));
        loads.extend([format!("{m}.hc_norm.weight"), format!("{m}.block_inject_weight.weight")]);
    }
    for n in ["q_norm", "k_norm", "indexer.q_layernorm", "indexer.k_layernorm"] {
        loads.push(format!("{head}.self_attn.{n}.weight"));
    }
    loads.extend(c.gr_packs(&c.mixer_name(true)).into_iter().flat_map(pack));
    for n in ["pre_fc_norm_embedding", "pre_fc_norm_hidden", "fc_embedding", "fc_hidden", "hyper_connection_mixer.hc_norm"] {
        loads.push(format!("mtp.{n}.weight"));
    }
    let missing: Vec<&String> = loads.iter().filter(|n| !names.contains(*n)).collect();
    assert!(missing.is_empty(), "{missing:?}");
    assert_eq!((d.desc.blocks.len(), d.mtp_layers, d.desc.hidden, d.desc.vocab), (97, 1, 2560, 248320));
    assert!(d.unread.is_empty(), "{:?}", d.unread);
    let v = d.desc.vision.as_ref().expect("the release's vision tower");
    assert_eq!(
        (v.blocks, v.hidden, v.heads, v.head_dim(), v.inter, v.act, v.patch_dim(), v.merge, v.out, v.table, v.theta, v.image_token),
        (27, 1152, 16, 72, 4304, Act::GeluTanh, 1536, 2, 2560, 48, 10_000.0, 248056)
    );
    let roles = d.vision.as_ref().unwrap();
    assert_eq!((roles.own.len(), roles.blocks.len(), roles.tensor(Some(26), "qkv").unwrap()), (9, 27, "model.visual.blocks.26.attn.qkv.weight"));
    assert_eq!(names.iter().filter(|n| n.starts_with("model.visual.")).count(), 9 + 27 * 12);
    // (the head: one indexed attention and its experts on the model's four streams, its front of two products and its own mixer)
    let h = &d.desc.mtp[0];
    assert_eq!((h.blocks.len(), h.input, d.mtp.len(), d.mtp[0].0.len()), (2, MtpInput::Streams, 1, 2));
    assert!(matches!(&h.blocks[0].body, Body::Mixer(Mixer::Attention(a)) if a.indexer.is_some()) && matches!(h.blocks[1].body, Body::Ffn(Ffn::Moe(_))));
    assert_eq!((d.shapes["mtp.pre_fc_norm_hidden.weight"].clone(), d.shapes["mtp.fc_hidden.weight"].clone()), (vec![4 * 2560], vec![2560, 2560]));
    // (the n-gram table: 128 shards of the 16 tables' rows padded to 128)
    let shards = names.iter().filter(|n| n.contains("ngram_embedding.shard_")).count();
    let rows: usize = d.shapes.iter().filter(|(n, _)| n.contains("ngram_embedding.shard_")).map(|(_, s)| s[0]).sum();
    assert_eq!((shards, rows % 128, d.shapes[&format!("{}layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight", c.prefix)][1]), (128, 0, 160));
    let refused = |edit: &dyn Fn(&mut serde_json::Value), says: &str| {
        let mut x = cfg.clone();
        edit(&mut x["text_config"]);
        let e = describe(&x).err().unwrap_or_else(|| panic!("{says}: described")).msg;
        assert!(e.contains(says), "{e}");
    };
    refused(&|t| t["indexer_kv_heads"] = 2.into(), "indexer_kv_heads");
    refused(&|t| t["mtp"]["layer_types"] = serde_json::json!(["linear_attention"]), "MTP layer of type linear_attention");
    refused(&|t| t["mtp"]["mtp_use_hidden_state_from_layer"] = 40.into(), "hidden state of a layer before the last");
    refused(&|t| t["indexer_budget"] = 2049.into(), "pools of 4");
    refused(&|t| _ = t.as_object_mut().unwrap().remove("indexer_head_dim"), "some are missing");
    refused(&|t| t["ple_layer_ids"] = serde_json::json!([4]), "ple_layer_ids 4");
    refused(&|t| t["hc_count"] = 1.into(), "1 streams");
    refused(&|t| t["number_of_conv_states"] = 1.into(), "number_of_conv_states");
    refused(&|t| t["qk_norm_type"] = "layer".into(), "qk_norm_type");
    refused(&|t| t["ngram_vocab_size_base"] = (1u64 << 50).into(), "an n-gram embedding");
    refused(&|t| t["heads_per_ngram"] = u64::MAX.into(), "an n-gram embedding");
    // (no n-gram layer: no eos needed, one conv state)
    let mut x = cfg.clone();
    x["text_config"]["ple_layer_ids"] = serde_json::json!([]);
    let plain = describe(&x).unwrap();
    assert!(plain.desc.blocks.len() == 96 && !plain.desc.blocks.iter().any(|b| matches!(b.body, Body::Ngram(_))));
}

/// Ibis's model from its config.json alone, on the CPU reference of the layer kinds, against Ibis's own engine: the random tiny Qwen3.8-Flash-Next of the `qwen4x`
/// logits case (src/logits.rs `qwen4x_random`) and the rows its engine recorded on an RTX 4080 SUPER at kestrel (tests/fixtures/qwen4x_ibis_rows.json.gz, that engine's
/// record: no logits case reads it, as the family is held to this reference and not to that engine's bits), where the random
/// model's experts are on the six-bit grid kestrel holds them on and the engine's other matrices are lossless: the engine runs this model's weights, its activations
/// bf16. The two implementations share no code. The prompt of 24 and the 12 decode steps (the prompt of 150 takes 50 s more here, and measured KL 4.0e-5): KL mean 8.5e-6, max 5.0e-5, the
/// best token 6 of 6 where it leads clearly, the largest logit difference 0.097 (measured 2026-11-23). It counts: with the indexer keeping every token the KL mean is
/// 1.8e-4 and the best token 4 of 6; with the n-gram hash's seed off by one, 1.1e-2 and 0 of 6.
#[cfg(feature = "dev")]
#[test]
fn ibis_model_on_the_reference_against_ibis_engine() {
    use super::kinds::{Body, Mixer};
    use super::reference::model::checkpoint_tensor;
    use crate::logits::{compare, Bound, Fixture};
    let dir = std::env::temp_dir().join(format!("glyd-qwen4x-ref-{}", std::process::id()));
    crate::logits::qwen4x_random(&dir).unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let d = describe(&cfg).unwrap_or_else(|e| panic!("{}", e.msg));
    let ck = glyd_gpu::safetensors::Checkpoint::open(&dir).unwrap();
    let mut fx = Fixture::read(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4x_ibis_rows.json.gz")).unwrap();
    // (the long prompt's row left out)
    fx.rows.pop();
    fx.long.clear();
    let src = |n: &str| checkpoint_tensor(&ck, n);
    let run = |d: &super::describe::Described| {
        let mut m = RefModel::load(d, &src).unwrap();
        let mut rows = vec![m.forward(&fx.prompt, 0).unwrap().pop().unwrap()];
        for (i, &t) in fx.decode.iter().enumerate() {
            rows.push(m.forward(&[t], (fx.prompt.len() + i) as u32).unwrap().pop().unwrap());
        }
        rows
    };
    let v = compare(&fx, &Bound::Kl { mean: 5e-5, max: 2e-4, top1: 1.0 }, &run(&d));
    assert!(v.ok && v.max_dlogit < 0.2, "KL mean {:.3e} max {:.3e}, largest logit difference {:.4}: {:?}", v.mean_kl, v.max_kl, v.max_dlogit, v.notes);
    let mut every = d.clone();
    for b in every.desc.blocks.iter_mut() {
        if let Body::Mixer(Mixer::Attention(a)) = &mut b.body {
            a.indexer.as_mut().unwrap().top_k = 1 << 20;
        }
    }
    let w = compare(&fx, &Bound::Kl { mean: 5e-5, max: 2e-4, top1: 1.0 }, &run(&every));
    assert!(w.mean_kl > 1e-4, "the indexer's picks do not show: KL mean {:.3e}", w.mean_kl);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A DeepSeek-V4.1 config of the release's keys (vLLM's and llama.cpp's readings of it: `compress_ratios`, the kv, index and candidate sources,
/// the Engram keys), its values small and made up for the test (the release's own config.json is #333's to record). Layers 0 and 1 the window
/// alone; 2 and 3 ratio 2 (both sources 2); 4 to 7 ratio 1 (kv sources 4 and 6, index sources 4, 5 and 6, the candidate source 4); Engram before
/// layers 1 and 5; one draft layer.
fn v41_config() -> serde_json::Value {
    let n = super::kinds::Ngram {
        order: 3,
        heads: 1,
        dim: 8,
        base: 11,
        vocab: 24,
        seed: 0,
        index: 0,
        eos: 0,
        streams: 2,
        conv: 0,
        dilation: 1,
        norm: super::kinds::Norm::rms(1e-6),
        form: super::kinds::NgramForm::Engram,
        table: Default::default(),
    };
    let rows: Vec<u64> = (0..2).map(|i| super::reference::ngram::tables(&super::kinds::Ngram { index: i, ..n }).0.iter().sum()).collect();
    let mut c: serde_json::Value = serde_json::from_str(
        r#"{"architectures": ["DeepseekV41ForCausalLM"], "model_type": "deepseek_v41",
        "quantization_config": {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [32, 32], "activation_scheme": "dynamic", "scale_fmt": "ue8m0",
            "expert_dtype": "fp4"},
        "vision_config": {"num_hidden_layers": 2},
        "text_config": {"model_type": "deepseek_v41_text", "vocab_size": 32, "hidden_size": 16, "num_hidden_layers": 8, "rms_norm_eps": 1e-6,
            "hidden_act": "silu", "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8, "qk_rope_head_dim": 4, "q_lora_rank": 12,
            "o_lora_rank": 4, "o_groups": 2, "sliding_window": 3, "index_n_heads": 2, "index_head_dim": 8, "index_topk": 2,
            "n_routed_experts": 4, "num_experts_per_tok": 2, "n_shared_experts": 1, "moe_intermediate_size": 6, "scoring_func": "sqrtsoftplus",
            "norm_topk_prob": true, "routed_scaling_factor": 1.5, "swiglu_limit": 10.0, "gate_temp": 1.0, "num_hash_layers": 0,
            "hc_mult": 2, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6, "rope_theta": 10000.0, "compress_rope_theta": 160000.0,
            "rope_scaling": {"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 64, "beta_fast": 32, "beta_slow": 1},
            "max_position_embeddings": 256, "compress_ratios": [0, 0, 2, 2, 1, 1, 1, 1, 0],
            "kv_source_layer_ids": [2, 4, 6], "index_source_layer_ids": [2, 4, 5, 6],
            "candidate_source_layer_id": 4, "candidate_block_size": 2, "candidate_topk_blocks": 1,
            "engram_layer_ids": [1, 5], "engram_max_ngram_size": 3, "engram_n_heads": 1, "engram_head_dim": 4, "engram_vocab_size": 11,
            "engram_compressed_vocab_size": 24, "engram_pad_token_id": 0,
            "num_nextn_predict_layers": 1, "dspark_block_size": 5, "dspark_noise_token_id": 31, "dspark_target_layer_ids": [5, 6, 7],
            "dspark_markov_rank": 4, "dspark_n_routed_experts": 3, "dspark_num_experts_per_tok": 1}}"#,
    )
    .unwrap();
    c["text_config"]["engram_num_embeddings"] = serde_json::json!(rows);
    c
}

/// #333, DeepSeek-V4.1 describes with the kinds there are, through V4.1's three parameters: CSA2's sharing (each compressing layer's entries and
/// picks from the most recent source at or below it, the candidate blocks from layer 4 to the index sources after it), lagged mHC (the head's
/// collapse the last block's weights), Engram before layers 1 and 5 (numpy's multipliers seeded by 10007 times the layer); the weight map reads the
/// sources' compressors and indexers alone, each quantized matrix with its scale, and leaves the drafter, the vision tower and the image tokens'
/// router bias. What it does not describe is refused.
#[test]
fn deepseek_v41_describes_with_csa2_lagged_mhc_and_engram() {
    use super::kinds::*;
    let cfg = v41_config();
    let d = describe(&cfg).unwrap_or_else(|e| panic!("{}", e.msg));
    assert_eq!((d.desc.blocks.len(), d.mtp_layers, d.desc.head.collapse), (2 * 8 + 2, 1, Collapse::LastPre));
    let comp = |b: usize| match &d.desc.blocks[b].body {
        Body::Mixer(Mixer::Compressed(c)) => c.clone(),
        x => panic!("block {b}: {x:?}"),
    };
    // (blocks: 0 attn, 1 moe, 2 engram, 3 attn of layer 1, 4 moe, then layers 2-4 two blocks each, 11 engram, 12 attn of layer 5, ...)
    let attn = [0usize, 3, 5, 7, 9, 12, 14, 16];
    for (l, &b) in attn.iter().enumerate() {
        let c = comp(b);
        assert!(!c.ape && !c.q_head_norm && matches!(d.desc.blocks[b].residual, Residual::Streams(StreamsP { lagged: true, .. })), "layer {l}");
        assert_eq!(c.ratio, [0, 0, 2, 2, 1, 1, 1, 1][l], "layer {l}");
    }
    let share = |l: usize| comp(attn[l]).share;
    assert_eq!(share(0), None);
    assert_eq!(share(2), Some(CompShare { entries: None, picks: None, candidates: None, publish: None }));
    assert_eq!(share(3), Some(CompShare { entries: Some(attn[2]), picks: Some(attn[2]), candidates: None, publish: None }));
    assert_eq!(share(4), Some(CompShare { entries: None, picks: None, candidates: None, publish: Some((2, 1)) }));
    assert_eq!(share(5), Some(CompShare { entries: Some(attn[4]), picks: None, candidates: Some(attn[4]), publish: None }));
    assert_eq!(share(6), Some(CompShare { entries: None, picks: None, candidates: Some(attn[4]), publish: None }));
    assert_eq!(share(7), Some(CompShare { entries: Some(attn[6]), picks: Some(attn[6]), candidates: None, publish: None }));
    for (b, layer, place) in [(2usize, 1u64, 0usize), (11, 5, 1)] {
        let Body::Ngram(n) = &d.desc.blocks[b].body else { panic!("block {b}: {:?}", d.desc.blocks[b].body) };
        assert_eq!((n.form, n.seed, n.index, n.conv, n.streams, n.vocab), (NgramForm::Engram, 10007 * layer, place, 0, 2, 24));
    }
    let names = d.names();
    let has = |n: &str| names.contains(n);
    assert!(has("layers.2.attn.compressor.wgate.weight") && !has("layers.6.attn.compressor.wgate.weight") && has("layers.6.attn.compressor.wkv.weight"));
    assert!(!has("layers.3.attn.compressor.wkv.weight") && !has("layers.3.attn.indexer.wq_b.weight"));
    assert!(has("layers.5.attn.indexer.wq_b.weight") && !has("layers.5.attn.indexer.wk.weight") && has("layers.4.attn.indexer.wk.weight"));
    assert!(has("layers.1.engram.embed.weight") && has("layers.5.engram.wkv.weight") && !has("hc_head_fn"));
    assert_eq!(d.scales.get("layers.1.engram.embed.weight").map(String::as_str), Some("layers.1.engram.embed.scale"));
    assert_eq!(d.scales.get("layers.0.ffn.experts.3.w2.weight").map(String::as_str), Some("layers.0.ffn.experts.3.w2.scale"));
    let mut ck: BTreeSet<String> = names.iter().cloned().chain(d.scales.values().cloned()).collect();
    ck.extend(["layers.3.ffn.gate.bias_vl", "mtp.0.attn.wq_a.weight", "vision.blocks.0.w.weight", "aligner.w.weight"].map(String::from));
    d.check(&ck, true).unwrap_or_else(|e| panic!("{}", e.msg));
    let refused = |edit: &dyn Fn(&mut serde_json::Value), want: &str| {
        let mut c = cfg.clone();
        edit(&mut c["text_config"]);
        let e = describe(&c).unwrap_err().msg;
        assert!(e.contains(want), "{want}: {e}");
    };
    // (its DSpark drafter: one layer of the window's attention and its own experts, on the lagged streams)
    let [h] = &d.desc.mtp[..] else { panic!("{:?}", d.desc.mtp) };
    let ds = h.dspark.as_ref().unwrap();
    assert_eq!((ds.block, ds.noise, ds.targets.clone(), ds.markov_rank, h.blocks.len()), (5, 31, vec![5, 6, 7], 4, 2));
    let (Body::Mixer(Mixer::Compressed(c)), Body::Ffn(Ffn::Moe(m))) = (&h.blocks[0].body, &h.blocks[1].body) else { panic!("{:?}", h.blocks) };
    assert_eq!((c.ratio, c.indexer.is_some(), c.share, m.experts, m.router.top_k), (0, false, None, 3, 1));
    assert!(matches!(h.blocks[0].residual, Residual::Streams(StreamsP { lagged: true, .. })));
    refused(&|t| t["compress_ratios"][2] = 4.into(), "compress_ratios");
    refused(&|t| t["gate_temp"] = 0.0.into(), "gate temperature");
    let mut hot = cfg.clone();
    hot["text_config"]["gate_temp"] = 2.0.into();
    let Body::Ffn(Ffn::Moe(m)) = &describe(&hot).unwrap().desc.blocks[1].body else { panic!() };
    assert_eq!(m.router.logit_scale, 0.5, "the logits over the gate temperature");
    refused(&|t| t["kv_source_layer_ids"] = serde_json::json!([2, 6]), "layer 4");
    refused(&|t| t["candidate_source_layer_id"] = 2.into(), "the candidate source 2");
    refused(&|t| t["engram_num_embeddings"][1] = 7.into(), "engram_num_embeddings");
}

/// DeepSeek-V4.1's sharing layouts past the release's are refused when described, in words that name the layer and its source, never computed: a source
/// of another ratio, and an index source that keeps no entries of its own whose kv source is no index source (its indexer keys would be read from a layer
/// that keeps none; before, such a layout described and failed when run).
#[test]
fn deepseek_v41_share_layouts_not_described_are_refused_in_words() {
    let with = |kv: serde_json::Value, idx: serde_json::Value| {
        let mut c = v41_config();
        c["text_config"]["kv_source_layer_ids"] = kv;
        c["text_config"]["index_source_layer_ids"] = idx;
        describe(&c).map(|_| ()).map_err(|e| e.msg)
    };
    use serde_json::json;
    assert_eq!(with(json!([2, 4, 6]), json!([2, 4, 5, 6])), Ok(()));
    let e = with(json!([2, 4, 6]), json!([2, 4, 5, 7])).unwrap_err();
    assert!(e.contains("index source 7 takes its entries from kv source 6, which is not in index_source_layer_ids"), "{e}");
    let e = with(json!([2, 6]), json!([2, 4, 5, 6])).unwrap_err();
    assert!(e.contains("layer 4 compresses by 1, and its source in kv_source_layer_ids, layer 2 (the most recent at or below it), by 2"), "{e}");
    let e = with(json!([3, 4, 6]), json!([2, 4, 5, 6])).unwrap_err();
    assert!(e.contains("layer 2 compresses by 2, and kv_source_layer_ids [3, 4, 6] has no layer at or below it"), "{e}");
}

/// DeepSeek-V4.1-Flash's configuration as published (tests/fixtures/deepseek_v41_config.json: the release's config.json, DeepSeek-V4.1-Flash-1017's, which
/// is the first DeepSeek-V4.1-Flash's byte for byte; its rms_norm_eps 1e-20, its inference code's norm_eps too): 40 layers (2 window-only, 18 of ratio 2, 20
/// of ratio 1), kv sources 2, 8, 14 and 20, index sources those and 24 to 36, the candidate source 20; Engram before layers 1 and 14, whose tables are the
/// primes past 16,000,000 that engram_num_embeddings counts (384,006,168 and 384,016,682 rows); three DSpark layers of 128 experts.
#[test]
fn deepseek_v41_flashs_published_config_describes() {
    use super::kinds::*;
    let d = describe(&fixture("deepseek_v41_config.json")).unwrap_or_else(|e| panic!("{}", e.msg));
    assert_eq!(d.desc.blocks.len(), 2 * 40 + 2);
    assert_eq!(d.desc.final_norm.eps, 1e-20);
    // (the release's rounding points: its RMSNorms' outputs rounded once, the streams' too; Engram's gate in fp32; the head's product in fp32)
    assert_eq!(
        (d.desc.final_norm.rounding, d.desc.blocks[0].norm_in.rounding, d.desc.blocks[1].norm_in.rounding),
        (Rounding::Once, Rounding::Once, Rounding::Once)
    );
    assert_eq!((d.desc.blocks[2].norm_in.rounding, d.desc.head.rounding), (Rounding::F32, Rounding::F32));
    assert!(
        matches!(d.desc.blocks[2].body, Body::Ngram(_))
            && matches!(d.desc.blocks[0].residual, Residual::Streams(StreamsP { rounding: Rounding::Once, lagged: true, .. }))
    );
    assert_eq!(d.fp8_block, Some(32));
    // (the release's file: the header fixture's config, recorded from the Hub)
    let h = fixture("headers/DeepSeek-V4.1-Flash-1017.json");
    assert_eq!(serde_json::from_str::<serde_json::Value>(h["config"].as_str().unwrap()).unwrap(), fixture("deepseek_v41_config.json"));
    // (the block of each layer's attention: two blocks a layer, Engram's before layers 1 and 14)
    let attn = |l: usize| 2 * l + (l >= 1) as usize + (l >= 14) as usize;
    let comp = |l: usize| match &d.desc.blocks[attn(l)].body {
        Body::Mixer(Mixer::Compressed(c)) => c.clone(),
        x => panic!("layer {l}: {x:?}"),
    };
    let sh = |l: usize| comp(l).share.unwrap();
    assert_eq!(comp(0).share, None);
    assert_eq!((sh(2).entries, sh(2).picks), (None, None));
    assert_eq!((sh(13).entries, sh(13).picks, comp(13).ratio), (Some(attn(8)), Some(attn(8)), 2));
    assert_eq!((sh(20).entries, sh(20).picks, sh(20).publish, comp(20).ratio), (None, None, Some((8, 2048)), 1));
    assert_eq!((sh(24).entries, sh(24).picks, sh(24).candidates), (Some(attn(20)), None, Some(attn(20))));
    assert_eq!((sh(39).entries, sh(39).picks, sh(39).candidates), (Some(attn(20)), Some(attn(36)), None));
    for (l, place) in [(1usize, 0usize), (14, 1)] {
        let Body::Ngram(n) = &d.desc.blocks[attn(l) - 1].body else { panic!("layer {l}") };
        assert_eq!((n.order, n.heads, n.dim, n.vocab, n.seed, n.index, n.eos), (4, 8, 3 * 8 * 256, 99_092, 10007 * l as u64, place, 2));
    }
    let [h] = &d.desc.mtp[..] else { panic!() };
    let Body::Ffn(Ffn::Moe(m)) = &h.blocks[1].body else { panic!() };
    assert_eq!((h.blocks.len(), m.experts, m.router.top_k, h.dspark.as_ref().unwrap().targets.clone()), (6, 128, 3, vec![37, 38, 39]));
    let Body::Ffn(Ffn::Moe(m)) = &d.desc.blocks[1].body else { panic!() };
    assert_eq!((m.experts, m.router.top_k, m.router.logit_scale, m.router.scale, m.expert.clamp), (384, 6, 1.0, 1.5, Some(super::kinds::Clamp::Gate(10.0))));
}

/// DeepSeek-V4.1's Engram keys that are no model's are refused by name, on the path a user's config.json takes: a list of layers that is not one (it was read as
/// no Engram layers, and the model described without them), an entry that is not a layer, an order under 2 (it underflowed), no heads or no head width (a
/// division by zero), a vocabulary past any model's (the builder would keep finding primes).
#[test]
fn deepseek_v41s_engram_keys_that_are_no_models_are_refused() {
    let with = |key: &str, v: serde_json::Value| {
        let mut c = v41_config();
        let tc = if c.get("text_config").is_some() { &mut c["text_config"] } else { &mut c };
        assert!(tc.get(key).is_some(), "{key} is a key of the fixture");
        tc[key] = v;
        describe(&c).err().unwrap_or_else(|| panic!("{key}: described")).msg
    };
    for (key, v, says) in [
        ("engram_layer_ids", serde_json::json!("1, 5"), "engram_layer_ids"),
        ("engram_layer_ids", serde_json::json!([1, "five"]), "engram_layer_ids"),
        ("engram_max_ngram_size", serde_json::json!(0), "engram_max_ngram_size 0"),
        ("engram_max_ngram_size", serde_json::json!(1), "engram_max_ngram_size 1"),
        ("engram_max_ngram_size", serde_json::json!(65), "engram_max_ngram_size 65"),
        ("engram_n_heads", serde_json::json!(0), "engram_n_heads 0"),
        ("engram_head_dim", serde_json::json!(0), "engram_head_dim 0"),
        ("engram_vocab_size", serde_json::json!(1u64 << 41), "engram_vocab_size"),
        ("engram_compressed_vocab_size", serde_json::json!(0), "engram_compressed_vocab_size 0"),
    ] {
        let e = with(key, v.clone());
        assert!(e.contains(says), "{key} = {v}: {e}");
    }
}

/// A smoke test, with no expected value: the tiny DeepSeek-V4.1 of `v41_config`, its tensors made at random by role, runs on the CPU reference as a model
/// (CSA2's parts shared from block to block, the mHC weights handed on, Engram's tables read through a compressed vocabulary), the same tokens give the same
/// logits twice, and every one is finite. It would pass with those parts wired wrongly end to end: what holds each part's values is its own test by hand
/// (`reference::comp`, `reference::ngram`, the streams' in `reference`), and nothing here holds the whole model's logits to another implementation's.
#[test]
fn deepseek_v41s_tiny_model_runs_on_the_reference() {
    use super::describe::Source;
    use super::reference::testing::fill;
    let d = describe(&v41_config()).unwrap();
    let (h, n, i, e) = (16usize, 2usize, 6usize, 4usize);
    let shape = |role: &str, b: usize| -> Vec<usize> {
        let engram = matches!(d.desc.blocks[b].body, super::kinds::Body::Ngram(_));
        match role {
            "norm_in" if engram => vec![n * h],
            "norm_in" => vec![h],
            "q_a" => vec![12, h],
            "q_a_norm" => vec![12],
            "q_b" | "idx_q_b" => vec![16, 12],
            "kv" if engram => vec![(n + 1) * h, 8],
            "kv" | "comp_kv" | "comp_gate" | "idx_k" => vec![8, if role == "idx_k" { 8 } else { h }],
            "kv_norm" | "comp_norm" | "idx_k_norm" => vec![8],
            "o_a" => vec![8, 8],
            "o_b" => vec![h, 8],
            "sinks" => vec![2],
            "idx_weights" => vec![2, h],
            "hc_fn" => vec![(2 + n) * n, n * h],
            "hc_base" => vec![(2 + n) * n],
            "hc_scale" => vec![3],
            "router" => vec![e, h],
            "choice_bias" => vec![e],
            "experts_gate_up" | "shared_gate" | "shared_up" => vec![i, h],
            "experts_down" | "shared_down" => vec![h, i],
            "ngram" => vec![0, 4],
            "norm_key" => vec![n * h],
            r => panic!("no shape for role {r}"),
        }
    };
    let mut t: std::collections::HashMap<String, (Vec<usize>, Vec<f32>)> = std::collections::HashMap::new();
    for (b, roles) in d.blocks.iter().enumerate() {
        for (role, src) in roles {
            let names = match src {
                Source::Tensor(x) | Source::RowsOf(x, _) | Source::Transposed(x) => vec![x.clone()],
                Source::Rows(v) => v.clone(),
                Source::Experts(v) => v.iter().flatten().cloned().collect(),
            };
            for x in names {
                let mut s = shape(role, b);
                if role == "ngram" {
                    let super::kinds::Body::Ngram(ng) = &d.desc.blocks[b].body else { unreachable!() };
                    s[0] = super::reference::ngram::tables(ng).0.iter().sum::<u64>() as usize;
                }
                let (lo, hi) = if role.contains("norm") { (0.5, 1.5) } else { (-0.4, 0.4) };
                let v = fill(&format!("v41.{x}"), s.iter().product(), lo, hi);
                t.insert(x, (s, v));
            }
        }
    }
    t.insert(d.embed.clone(), (vec![32, h], fill("v41.embed", 32 * h, -1.0, 1.0)));
    t.insert(d.head.clone().unwrap(), (vec![32, h], fill("v41.head", 32 * h, -0.5, 0.5)));
    t.insert(d.final_norm.clone(), (vec![h], fill("v41.norm", h, 0.5, 1.5)));
    let src = |x: &str| t.get(x).cloned();
    let run = || {
        let mut m = RefModel::load(&d, &src).unwrap_or_else(|e| panic!("{}", e.msg));
        // (the compressed vocabulary's map, the loader's from the tokenizer: 32 ids onto 24)
        for b in [2, 11] {
            m.layers[b].p.put("ngram_map", &[32], (0..32).map(|x| (x % 24) as f32).collect());
        }
        let tokens: Vec<u32> = (0..12).map(|x| (x * 5 + 3) % 32).collect();
        m.forward(&tokens, 0).unwrap_or_else(|e| panic!("{}", e.msg))
    };
    let (a, b) = (run(), run());
    assert_eq!(a, b);
    assert!(a.len() == 12 && a.iter().flatten().all(|v| v.is_finite()));
}

/// #549, Domyn Edge: the released Domyn-Edge-Base (its recorded header) describes as its two sorts of layers: 12 over the whole context with no positions
/// and heads 256 wide, 24 with rotary positions, heads 64 wide and a window of 1,025 (the query's own position and 1,024 before it); every layer's queries
/// normed with no weight and its keys with one, each part between two norms; the head is the embedding. Every shape the description gives is the
/// checkpoint's, and the checkpoint is read whole.
#[test]
fn domyn_edge_base_describes_its_two_sorts_of_layers() {
    use super::kinds::{Body, Ffn, Mixer, NormWeight};
    let h = fixture("headers/Domyn-Edge-Base.json");
    let cfg: serde_json::Value = serde_json::from_str(h["config"].as_str().unwrap()).unwrap();
    let d = describe(&cfg).unwrap_or_else(|e| panic!("{}", e.msg));
    assert_eq!((d.desc.blocks.len(), d.desc.head.tied, d.head.is_none(), d.desc.hidden), (72, true, true, 2560));
    let (mut whole, mut windowed) = (0, 0);
    for (i, b) in d.desc.blocks.iter().enumerate() {
        assert!(b.norm_out.is_some(), "block {i}: a norm after the part");
        match &b.body {
            Body::Mixer(Mixer::Attention(a)) => {
                assert_eq!((a.heads, a.kv_heads, a.bias, a.sinks), (32, 8, false, false), "block {i}");
                assert_eq!((a.q_norm.unwrap().weight, a.k_norm.unwrap().weight), (NormWeight::None, NormWeight::Plain), "block {i}");
                assert_eq!(a.scale, (a.head_dim as f32).powf(-0.5), "block {i}");
                match (a.head_dim, a.window, a.rope.is_some()) {
                    (256, None, false) => whole += 1,
                    (64, Some(1025), true) => windowed += 1,
                    other => panic!("block {i}: {other:?}"),
                }
            }
            Body::Ffn(Ffn::Mlp(_)) => {}
            _ => panic!("block {i} is neither attention nor an MLP"),
        }
    }
    assert_eq!((whole, windowed), (12, 24));
    // (every tensor read has the checkpoint's shape, and no q norm is read: it has no weight)
    let table = crate::sizes::Table::expand(&crate::sizes::groups_of(&h).unwrap());
    let have = header_names(&h);
    assert_eq!(d.names().len(), have.len());
    assert!(!d.names().iter().any(|n| n.contains("q_norm")));
    for r in &table.rows {
        let want: Vec<usize> = r.shape.iter().map(|&x| x as usize).collect();
        assert_eq!(d.shapes.get(&r.name), Some(&want), "{}", r.name);
    }
    d.check(&have, true).unwrap_or_else(|e| panic!("{}", e.msg));
}

/// A small config of the Domyn Edge layout: `sorts` gives each layer as (whole context: head 8, no positions) or (windowed: head 4, rotary, `left` keys
/// before the query's own).
fn domyn_tiny(sorts: &[bool], left: i64) -> serde_json::Value {
    let per = |whole: serde_json::Value, windowed: serde_json::Value| -> Vec<serde_json::Value> {
        sorts.iter().map(|&w| if w { whole.clone() } else { windowed.clone() }).collect()
    };
    serde_json::json!({
        "architectures": ["DomynEdgeForCausalLM"], "model_type": "domynedge", "hidden_size": 32, "num_hidden_layers": sorts.len(), "vocab_size": 61,
        "num_attention_heads": 4, "num_key_value_heads": 2, "intermediate_size": 48, "hidden_act": "silu", "rms_norm_eps": 1e-6, "tie_word_embeddings": true,
        "max_position_embeddings": 4096, "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0}, "sliding_window": left + 1,
        "position_layers": per("nope".into(), "rope".into()),
        "window_sizes": per(serde_json::json!([-1, 0]), serde_json::json!([left, 0])),
        "head_sizes": per(8.into(), 4.into()),
        "layer_types": per("full_attention".into(), "sliding_attention".into()),
    })
}

/// A setting of the Domyn Edge layout the builder does not know is refused in words that name the list, the layer and the value, and a list of another
/// length than the layers is refused.
#[test]
fn domyn_edge_refuses_what_it_does_not_know() {
    let ok = domyn_tiny(&[true, false], 2);
    describe(&ok).unwrap_or_else(|e| panic!("{}", e.msg));
    let with = |path: &str, i: usize, v: serde_json::Value| {
        let mut c = ok.clone();
        c[path][i] = v;
        describe(&c).err().map(|e| e.msg).unwrap_or_default()
    };
    let e = with("position_layers", 1, "learned".into());
    assert!(e.contains("position_layers[1]") && e.contains("learned"), "{e}");
    let e = with("window_sizes", 1, serde_json::json!([2, 1]));
    assert!(e.contains("window_sizes[1]") && e.contains("[2,1]"), "{e}");
    let e = with("layer_types", 0, "sliding_attention".into());
    assert!(e.contains("layer_types[0]") && e.contains("does not agree"), "{e}");
    let e = with("window_sizes", 1, serde_json::json!([5, 0]));
    assert!(e.contains("a window of 6") && e.contains("sliding_window says 3"), "{e}");
    let mut short = ok.clone();
    short["head_sizes"] = serde_json::json!([8]);
    let e = describe(&short).unwrap_err().msg;
    assert!(e.contains("head_sizes has 1 entries for 2 layers"), "{e}");
    let mut more = ok.clone();
    more["new_gate"] = true.into();
    more["other_width"] = 3.into();
    let e = describe(&more).unwrap_err().msg;
    assert!(e.contains("config keys new_gate, other_width are not understood"), "{e}");
}

/// The Domyn Edge layout on the CPU reference, a random model of each small config (every tensor of the shape its description gives): it runs, a token at
/// a time gives the rows a whole prompt gives, and the window is the layout's. With every layer windowed to the query's own position and 2 before it, two
/// layers see 4 positions back: the last row of a prompt of six does not change, to the bit, when the first token does, and it changes when the second
/// does. With one layer over the whole context it changes with the first token too.
#[test]
fn domyn_edge_on_the_reference_keeps_its_windows() {
    let run = |cfg: &serde_json::Value, toks: &[u32]| -> Vec<Vec<f32>> {
        let d = describe(cfg).unwrap_or_else(|e| panic!("{}", e.msg));
        let src = |n: &str| {
            let shape = d.shapes.get(n)?.clone();
            let len: usize = shape.iter().product();
            // (a fixed draw a tensor: its name's bytes seed it; a norm's weights near 1)
            let mut s = n.bytes().fold(0x9E37_79B9_7F4A_7C15u64, |a, b| (a ^ b as u64).wrapping_mul(0x0100_0000_01B3));
            let norm = n.contains("norm");
            let v = (0..len)
                .map(|_| {
                    s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                    let u = ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5;
                    if norm {
                        1.0 + 0.2 * u
                    } else {
                        0.6 * u
                    }
                })
                .collect();
            Some((shape, v))
        };
        let mut m = RefModel::load(&d, &src).unwrap_or_else(|e| panic!("{}", e.msg));
        m.forward(toks, 0).unwrap_or_else(|e| panic!("{}", e.msg))
    };
    let (a, first, second): ([u32; 6], [u32; 6], [u32; 6]) = ([5, 9, 13, 17, 21, 25], [44, 9, 13, 17, 21, 25], [5, 50, 13, 17, 21, 25]);
    let windowed = domyn_tiny(&[false, false], 2);
    let rows = run(&windowed, &a);
    assert_eq!((rows.len(), rows[5].len()), (6, 61));
    assert!(rows.iter().flatten().all(|x| x.is_finite()) && rows[5].iter().any(|&x| x != rows[5][0]), "finite, and not one value");
    assert_eq!(rows[5], run(&windowed, &first)[5], "four positions back is as far as two windowed layers see");
    assert_ne!(rows[5], run(&windowed, &second)[5], "the second token is inside the two layers' reach");
    let mixed = domyn_tiny(&[false, true], 2);
    assert_ne!(run(&mixed, &a)[5], run(&mixed, &first)[5], "a layer over the whole context sees the first token");
    // (a wider window is another answer: the window is read, not ignored)
    assert_ne!(rows[5], run(&domyn_tiny(&[false, false], 4), &a)[5]);
}

/// #549, Granite Swash: the released granite-swash-2b (its recorded header) describes as 24 layers of Llama-style parts with Granite's multipliers (the
/// embedding x 12, each part's output x 0.28, the scores x 1/128, the logits / 10), a sink a head on every layer, and a window of 128 on the layers its
/// `layer_types` calls sliding; every shape is the checkpoint's and the checkpoint is read whole, its sinks among the tensors read.
#[test]
fn granite_swash_2b_describes_its_multipliers_windows_and_sinks() {
    use super::kinds::{Body, Ffn, Mixer, Residual};
    let h = fixture("headers/granite-swash-2b.json");
    let cfg: serde_json::Value = serde_json::from_str(h["config"].as_str().unwrap()).unwrap();
    let d = describe(&cfg).unwrap_or_else(|e| panic!("{}", e.msg));
    let types: Vec<&str> = cfg["layer_types"].as_array().unwrap().iter().map(|t| t.as_str().unwrap()).collect();
    assert_eq!((d.desc.blocks.len(), d.desc.head.tied, d.desc.embed.scale, d.desc.head.scale), (48, true, 12.0, 0.1));
    for (i, b) in d.desc.blocks.iter().enumerate() {
        assert_eq!(b.residual, Residual::AddScaled(0.28), "block {i}");
        assert!(b.norm_out.is_none(), "block {i}");
        match &b.body {
            Body::Mixer(Mixer::Attention(a)) => {
                assert_eq!((a.heads, a.kv_heads, a.head_dim, a.sinks, a.scale, a.rope.is_some()), (20, 4, 128, true, 0.0078125, true), "block {i}");
                assert_eq!(a.window, (types[i / 2] == "sliding_attention").then_some(128), "layer {}", i / 2);
                assert!(a.q_norm.is_none() && a.k_norm.is_none());
            }
            Body::Ffn(Ffn::Mlp(_)) => {}
            _ => panic!("block {i}"),
        }
    }
    assert!(types.contains(&"full_attention") && types.contains(&"sliding_attention"));
    let table = crate::sizes::Table::expand(&crate::sizes::groups_of(&h).unwrap());
    let have = header_names(&h);
    assert_eq!(d.names().iter().filter(|n| n.ends_with("self_attn.sinks")).count(), 24);
    for r in &table.rows {
        let want: Vec<usize> = r.shape.iter().map(|&x| x as usize).collect();
        assert_eq!(d.shapes.get(&r.name), Some(&want), "{}", r.name);
    }
    d.check(&have, true).unwrap_or_else(|e| panic!("{}", e.msg));
}

/// A small config of the dense Granite Swash layout with `keys` over the class's defaults.
fn granite_tiny(keys: serde_json::Value) -> serde_json::Value {
    let mut c = serde_json::json!({
        "architectures": ["GraniteSWAForCausalLM"], "model_type": "granite_swa", "hidden_size": 32, "num_hidden_layers": 2, "vocab_size": 61,
        "num_attention_heads": 4, "num_key_value_heads": 2, "intermediate_size": 48, "hidden_act": "silu", "max_position_embeddings": 4096, "rope_theta": 10000.0,
    });
    c.as_object_mut().unwrap().extend(keys.as_object().unwrap().clone());
    c
}

/// The Granite Swash config class's defaults are the description's where a config is silent: norms' eps 1e-5, the head tied, every multiplier 1, a window
/// of 128 on every layer but each fourth from the first, the model's rotary on every layer. A `layer_rope_theta` gives a layer its own base, and none at 0.
/// What the builder does not know is refused by name.
#[test]
fn granite_swash_defaults_are_its_config_classs() {
    use super::kinds::{Body, Mixer, Residual};
    let attn = |d: &super::describe::Described, layer: usize| match &d.desc.blocks[2 * layer].body {
        Body::Mixer(Mixer::Attention(a)) => a.clone(),
        _ => panic!(),
    };
    let d = describe(&granite_tiny(serde_json::json!({"num_hidden_layers": 5}))).unwrap_or_else(|e| panic!("{}", e.msg));
    assert_eq!((d.desc.final_norm.eps, d.desc.head.tied, d.desc.embed.scale, d.desc.head.scale), (1e-5, true, 1.0, 1.0));
    assert_eq!(d.desc.blocks[0].residual, Residual::Add);
    assert_eq!((0..5).map(|l| attn(&d, l).window).collect::<Vec<_>>(), [None, Some(128), Some(128), Some(128), None]);
    assert_eq!((attn(&d, 0).scale, attn(&d, 0).kv_heads), (1.0, 2));
    let d = describe(&granite_tiny(serde_json::json!({"layer_rope_theta": [0, 500.0]}))).unwrap_or_else(|e| panic!("{}", e.msg));
    assert!(attn(&d, 0).rope.is_none() && attn(&d, 1).rope.is_some() && attn(&d, 1).rope != describe(&granite_tiny(serde_json::json!({}))).unwrap().desc.rope);
    for (keys, word) in [
        (serde_json::json!({"attention_bias": true}), "attention_bias"),
        (serde_json::json!({"layer_types": ["full_attention", "global"]}), "layer_types[1]"),
        (serde_json::json!({"layer_types": ["full_attention"]}), "layer_types has 1 entries for 2 layers"),
        (serde_json::json!({"logits_scaling": 0}), "logits_scaling = 0"),
        (serde_json::json!({"num_attention_heads": 5}), "heads of one width"),
        (serde_json::json!({"shared_intermediate_size": 8}), "config key shared_intermediate_size is not understood"),
    ] {
        let e = describe(&granite_tiny(keys)).unwrap_err().msg;
        assert!(e.contains(word), "{word}: {e}");
    }
}

/// The dense Granite Swash layout on the CPU reference, a random model of each small config. The logits are divided by `logits_scaling`: by 4 they are
/// half of what they are by 2, to the bit. A sink takes part in the softmax: with every sink far below the scores the answer is the one of a model with
/// no sinks at all, and with the sinks at the scores' size it is another. A window of 3 on both layers leaves the last row of six unchanged, to the bit,
/// when the first token changes.
#[test]
fn granite_swash_on_the_reference_scales_sinks_and_windows() {
    let run = |cfg: &serde_json::Value, sink: f32, sinks: bool, toks: &[u32]| -> Vec<f32> {
        let mut d = describe(cfg).unwrap_or_else(|e| panic!("{}", e.msg));
        if !sinks {
            for b in d.desc.blocks.iter_mut() {
                if let super::kinds::Body::Mixer(super::kinds::Mixer::Attention(a)) = &mut b.body {
                    a.sinks = false;
                }
            }
        }
        let src = |n: &str| {
            let shape = d.shapes.get(n)?.clone();
            let len: usize = shape.iter().product();
            if n.ends_with(".sinks") {
                return Some((shape, vec![sink; len]));
            }
            let mut s = n.bytes().fold(0x9E37_79B9_7F4A_7C15u64, |a, b| (a ^ b as u64).wrapping_mul(0x0100_0000_01B3));
            let norm = n.contains("norm");
            let v = (0..len)
                .map(|_| {
                    s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                    let u = ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5;
                    if norm {
                        1.0 + 0.2 * u
                    } else {
                        0.6 * u
                    }
                })
                .collect();
            Some((shape, v))
        };
        let mut m = RefModel::load(&d, &src).unwrap_or_else(|e| panic!("{}", e.msg));
        m.forward(toks, 0).unwrap_or_else(|e| panic!("{}", e.msg)).pop().unwrap()
    };
    let (a, first): ([u32; 6], [u32; 6]) = ([5, 9, 13, 17, 21, 25], [44, 9, 13, 17, 21, 25]);
    let base = serde_json::json!({"embedding_multiplier": 4.0, "residual_multiplier": 0.5, "attention_multiplier": 0.25, "layer_types": ["full_attention", "full_attention"]});
    let with = |more: serde_json::Value| {
        let mut k = base.clone();
        k.as_object_mut().unwrap().extend(more.as_object().unwrap().clone());
        granite_tiny(k)
    };
    let by2 = run(&with(serde_json::json!({"logits_scaling": 2.0})), 0.0, true, &a);
    let by4 = run(&with(serde_json::json!({"logits_scaling": 4.0})), 0.0, true, &a);
    assert!(by2.iter().all(|x| x.is_finite()) && by2.iter().any(|&x| x != by2[0]));
    assert_eq!(by4, by2.iter().map(|x| x / 2.0).collect::<Vec<f32>>(), "the logits over logits_scaling");
    let c = with(serde_json::json!({"logits_scaling": 2.0}));
    assert_eq!(run(&c, -1e9, true, &a), run(&c, 0.0, false, &a), "a sink far below the scores is no sink");
    assert_ne!(by2, run(&c, 0.0, false, &a), "a sink at the scores' size takes its share of the softmax");
    let windowed = with(serde_json::json!({"layer_types": ["sliding_attention", "sliding_attention"], "sliding_window": 3}));
    assert_eq!(run(&windowed, 0.0, true, &a), run(&windowed, 0.0, true, &first), "two layers of three positions see four back");
    assert_ne!(run(&c, 0.0, true, &a), run(&c, 0.0, true, &first), "the whole context sees the first token");
}

/// #549, Mellum 2: the released Mellum2.1-12B-A2.5B-Instruct (its recorded header) describes as 28 layers of Qwen3-style attention (32 query and 4
/// key heads of 128, queries and keys normed with a weight) over the whole context or a window of 1,024 as `layer_types` says, each sort with its own
/// rotary (yarn on the full layers, plain on the sliding ones), and a mixture of 64 experts a layer: the softmax of the router's logits, the best 8
/// divided by their sum, no shared expert. Every shape is the checkpoint's and the checkpoint is read whole.
#[test]
fn mellum_21_describes_its_layer_types_rotaries_and_experts() {
    use super::kinds::{Body, Ffn, Mixer, RouterScore};
    let h = fixture("headers/Mellum2.1-12B-A2.5B-Instruct.json");
    let cfg: serde_json::Value = serde_json::from_str(h["config"].as_str().unwrap()).unwrap();
    let d = describe(&cfg).unwrap_or_else(|e| panic!("{}", e.msg));
    let types: Vec<&str> = cfg["layer_types"].as_array().unwrap().iter().map(|t| t.as_str().unwrap()).collect();
    assert_eq!((d.desc.blocks.len(), d.desc.head.tied, d.head.is_some()), (56, false, true));
    let (mut full, mut sliding) = (None, None);
    for (i, b) in d.desc.blocks.iter().enumerate() {
        match &b.body {
            Body::Mixer(Mixer::Attention(a)) => {
                assert_eq!((a.heads, a.kv_heads, a.head_dim, a.sinks, a.bias), (32, 4, 128, false, false), "block {i}");
                assert!(a.q_norm.is_some() && a.k_norm.is_some(), "block {i}");
                let is_sliding = types[i / 2] == "sliding_attention";
                assert_eq!(a.window, is_sliding.then_some(1024), "layer {}", i / 2);
                let seen = if is_sliding { &mut sliding } else { &mut full };
                assert!(seen.is_none() || *seen == a.rope, "one rotary a layer type");
                *seen = a.rope.clone();
            }
            Body::Ffn(Ffn::Moe(m)) => {
                assert_eq!((m.experts, m.router.top_k, m.router.renorm, m.router.score == RouterScore::Softmax), (64, 8, true, true), "block {i}");
                assert!(m.shared.is_none() && !m.shared_gate && m.latent.is_none(), "block {i}");
            }
            _ => panic!("block {i}"),
        }
    }
    assert!(full.is_some() && sliding.is_some() && full != sliding, "the full layers' rotary is not the sliding layers'");
    let table = crate::sizes::Table::expand(&crate::sizes::groups_of(&h).unwrap());
    let have = header_names(&h);
    assert_eq!(d.names().len(), have.len());
    for r in &table.rows {
        let want: Vec<usize> = r.shape.iter().map(|&x| x as usize).collect();
        assert_eq!(d.shapes.get(&r.name), Some(&want), "{}", r.name);
    }
    d.check(&have, true).unwrap_or_else(|e| panic!("{}", e.msg));
}

/// A small config of the Mellum layout with `keys` over the class's defaults.
fn mellum_tiny(keys: serde_json::Value) -> serde_json::Value {
    let mut c = serde_json::json!({
        "architectures": ["MellumForCausalLM"], "model_type": "mellum", "hidden_size": 32, "num_hidden_layers": 2, "vocab_size": 61,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "intermediate_size": 48, "moe_intermediate_size": 16, "num_experts": 6,
        "num_experts_per_tok": 2, "max_position_embeddings": 4096,
    });
    c.as_object_mut().unwrap().extend(keys.as_object().unwrap().clone());
    c
}

/// The Mellum config class's defaults are the description's where a config is silent (every layer full attention and experts, the head not tied, the
/// class's two rotaries), a "dense" layer is a gated MLP of `intermediate_size`, the draft layers are counted under either of their keys, and what the
/// builder does not know is refused by name.
#[test]
fn mellum_defaults_are_its_config_classs() {
    use super::kinds::{Body, Ffn, Mixer};
    let d = describe(&mellum_tiny(serde_json::json!({}))).unwrap_or_else(|e| panic!("{}", e.msg));
    assert!(!d.desc.head.tied && d.desc.final_norm.eps == 1e-6);
    assert!(d.desc.blocks.iter().step_by(2).all(|b| matches!(&b.body, Body::Mixer(Mixer::Attention(a)) if a.window.is_none())));
    assert!(d.desc.blocks.iter().skip(1).step_by(2).all(|b| matches!(&b.body, Body::Ffn(Ffn::Moe(m)) if m.router.renorm && m.router.top_k == 2)));
    let d =
        describe(&mellum_tiny(serde_json::json!({"mlp_layer_types": ["dense", "sparse"], "layer_types": ["sliding_attention", "full_attention"]}))).unwrap();
    assert!(matches!(&d.desc.blocks[1].body, Body::Ffn(Ffn::Mlp(_))) && matches!(&d.desc.blocks[3].body, Body::Ffn(Ffn::Moe(_))));
    assert!(matches!(&d.desc.blocks[0].body, Body::Mixer(Mixer::Attention(a)) if a.window == Some(1024)));
    assert!(
        d.names().iter().any(|n| n.ends_with("layers.0.mlp.gate_proj.weight"))
            && d.names().iter().any(|n| n.ends_with("layers.1.mlp.experts.5.down_proj.weight"))
    );
    for key in ["num_mtp_layers", "num_nextn_predict_layers"] {
        assert_eq!(describe(&mellum_tiny(serde_json::json!({key: 1}))).unwrap().mtp_layers, 1, "{key}");
    }
    for (keys, word) in [
        (serde_json::json!({"attention_bias": true}), "attention_bias"),
        (serde_json::json!({"layer_types": ["full_attention", "global"]}), "layer_types[1]"),
        (serde_json::json!({"mlp_layer_types": ["sparse", "latent"]}), "mlp_layer_types[1]"),
        (serde_json::json!({"rope_parameters": {"rope_type": "default", "rope_theta": 1e4}}), "is not a layer type"),
        (
            serde_json::json!({"layer_types": ["sliding_attention", "full_attention"], "rope_parameters": {"full_attention": {"rope_type": "default", "rope_theta": 1e4}}}),
            "has no sliding_attention",
        ),
        (serde_json::json!({"shared_expert_intermediate_size": 8}), "config key shared_expert_intermediate_size is not understood"),
    ] {
        let e = describe(&mellum_tiny(keys)).unwrap_err().msg;
        assert!(e.contains(word), "{word}: {e}");
    }
}

/// The Mellum layout on the CPU reference, a random model of each small config: it runs; the experts kept a token count (two of six is not three of
/// six); a dense layer is another model than a sparse one; and two layers windowed to three positions do not see a token five back, to the bit, while
/// full layers do.
#[test]
fn mellum_on_the_reference_routes_and_windows() {
    let run = |cfg: &serde_json::Value, toks: &[u32]| -> Vec<f32> {
        let d = describe(cfg).unwrap_or_else(|e| panic!("{}", e.msg));
        let src = |n: &str| {
            let shape = d.shapes.get(n)?.clone();
            let len: usize = shape.iter().product();
            let mut s = n.bytes().fold(0x9E37_79B9_7F4A_7C15u64, |a, b| (a ^ b as u64).wrapping_mul(0x0100_0000_01B3));
            let norm = n.contains("norm");
            let v = (0..len)
                .map(|_| {
                    s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                    let u = ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5;
                    if norm {
                        1.0 + 0.2 * u
                    } else {
                        0.6 * u
                    }
                })
                .collect();
            Some((shape, v))
        };
        let mut m = RefModel::load(&d, &src).unwrap_or_else(|e| panic!("{}", e.msg));
        m.forward(toks, 0).unwrap_or_else(|e| panic!("{}", e.msg)).pop().unwrap()
    };
    let (a, first): ([u32; 6], [u32; 6]) = ([5, 9, 13, 17, 21, 25], [44, 9, 13, 17, 21, 25]);
    let base = mellum_tiny(serde_json::json!({}));
    let row = run(&base, &a);
    assert!(row.len() == 61 && row.iter().all(|x| x.is_finite()) && row.iter().any(|&x| x != row[0]));
    assert_ne!(row, run(&mellum_tiny(serde_json::json!({"num_experts_per_tok": 3})), &a), "three experts a token is another answer than two");
    assert_ne!(row, run(&mellum_tiny(serde_json::json!({"mlp_layer_types": ["dense", "sparse"]})), &a), "a dense layer is another model");
    assert_ne!(row, run(&base, &first), "full layers see the first token");
    let windowed = mellum_tiny(serde_json::json!({"layer_types": ["sliding_attention", "sliding_attention"], "sliding_window": 3}));
    assert_eq!(run(&windowed, &a), run(&windowed, &first), "two layers of three positions see four back");
}

/// MiMo V2: the released MiMo-V2.6-Flash-RL (its recorded header) describes as 48 layers of two sorts as `hybrid_layer_pattern` says: 9 over the whole
/// context (64 query heads over 4 key heads, a rotary of base 10,000,000) and 39 sliding (8 key heads, the query's own position and 128 before it, a sink
/// a head, base 10,000); queries and keys 192 wide and values 128, the first 64 channels of a head rotated, every value times 0.707. The first layer's
/// feed-forward is a gated MLP and the other 47 are 256 experts routed by a sigmoid with a choice bias, the best 8 divided by their sum, no shared
/// expert. Its three draft layers are draft heads, each a sliding attention and a dense MLP. The fused query, key and value matrix is cut a shard at a
/// time; every shape is the checkpoint's (an expert's 4-bit matrix two values a byte), every FP8 tensor of the header has its scales named and no other
/// has, and the checkpoint is read whole but the audio encoder, the speech embeddings and the vision tower.
#[test]
fn mimo_v26_flash_describes_its_two_sorts_of_layers_experts_and_draft_heads() {
    use super::describe::Source;
    use super::kinds::{Body, Ffn, Mixer, MtpInput, RouterScore};
    let h = fixture("headers/MiMo-V2.6-Flash-RL.json");
    let cfg: serde_json::Value = serde_json::from_str(h["config"].as_str().unwrap()).unwrap();
    let d = describe(&cfg).unwrap_or_else(|e| panic!("{}", e.msg));
    let pattern: Vec<u64> = cfg["hybrid_layer_pattern"].as_array().unwrap().iter().map(|t| t.as_u64().unwrap()).collect();
    assert_eq!((d.desc.blocks.len(), d.desc.head.tied, d.desc.final_norm.eps, d.desc.embed.scale, d.fp8_block), (96, false, 1e-6, 1.0, Some(128)));
    let (mut full, mut sliding, mut dense, mut sparse) = (0, 0, 0, 0);
    for (i, b) in d.desc.blocks.iter().enumerate() {
        assert!(b.norm_out.is_none(), "block {i}");
        match &b.body {
            Body::Mixer(Mixer::Attention(a)) => {
                let swa = pattern[i / 2] == 1;
                assert_eq!((a.heads, a.kv_heads, a.head_dim, a.v_head_dim), (64, if swa { 8 } else { 4 }, 192, 128), "layer {}", i / 2);
                assert_eq!((a.window, a.sinks, a.v_scale, a.bias), (swa.then_some(129), swa, 0.707, false), "layer {}", i / 2);
                assert!(a.q_norm.is_none() && a.k_norm.is_none() && a.gate.is_none(), "layer {}", i / 2);
                let r = a.rope.as_ref().unwrap();
                assert_eq!((r.theta, r.dim), (if swa { 1e4 } else { 1e7 }, 64), "layer {}", i / 2);
                *if swa { &mut sliding } else { &mut full } += 1;
            }
            Body::Ffn(Ffn::Mlp(m)) => {
                assert!(i == 1 && m.gated && !m.bias, "block {i}");
                dense += 1;
            }
            Body::Ffn(Ffn::Moe(m)) => {
                assert_eq!((m.experts, m.router.top_k, m.router.renorm, m.router.bias, m.router.scale), (256, 8, true, true, 1.0), "block {i}");
                assert!(m.router.score == RouterScore::Sigmoid && m.router.groups.is_none() && m.shared.is_none() && m.latent.is_none(), "block {i}");
                sparse += 1;
            }
            _ => panic!("block {i}"),
        }
    }
    assert_eq!((full, sliding, dense, sparse), (9, 39, 1, 47));
    // (the draft layers: three heads of the joined form, each a sliding layer's attention and a dense MLP)
    assert_eq!((d.desc.mtp.len(), d.mtp.len(), d.mtp_layers), (3, 3, 3));
    for head in &d.desc.mtp {
        assert!(head.input == MtpInput::Joined && head.dspark.is_none() && head.blocks.len() == 2);
        assert!(matches!(&head.blocks[0].body, Body::Mixer(Mixer::Attention(a)) if a.window == Some(129) && a.sinks && a.kv_heads == 8 && a.v_scale == 0.707));
        assert!(matches!(&head.blocks[1].body, Body::Ffn(Ffn::Mlp(_))));
    }
    assert!(d.names().contains("model.mtp.layers.2.eh_proj.weight") && d.names().contains("model.mtp.layers.0.pre_mlp_layernorm.weight"));
    // (the fused matrix, a shard at a time: 16 query heads, then the shard's key heads, then its value heads; a sliding layer's shard holds two of each)
    let cut = |block: usize, role: &str| match &d.blocks[block].iter().find(|(r, _)| r == role).unwrap().1 {
        Source::RowsOf(n, rows) => (n.clone(), rows.clone()),
        other => panic!("{role}: {other:?}"),
    };
    let qkv = |l: usize| format!("model.layers.{l}.self_attn.qkv_proj.weight");
    assert_eq!(cut(0, "q"), (qkv(0), vec![(0, 3072), (3392, 6464), (6784, 9856), (10176, 13248)]));
    assert_eq!(cut(0, "k"), (qkv(0), vec![(3072, 3264), (6464, 6656), (9856, 10048), (13248, 13440)]));
    assert_eq!(cut(0, "v"), (qkv(0), vec![(3264, 3392), (6656, 6784), (10048, 10176), (13440, 13568)]));
    assert_eq!(cut(2, "k"), (qkv(1), vec![(3072, 3456), (6784, 7168), (10496, 10880), (14208, 14592)]));
    assert_eq!(cut(2, "v").1[3], (14592, 14848));
    // every tensor the description reads, at the checkpoint's shape and in the stored format the description says
    let have = header_names(&h);
    let tensors = header_tensors(&h);
    let (mut fp8, mut fp4, mut sinks) = (0, 0, 0);
    let scale_names: BTreeSet<&String> = d.scales.values().collect();
    for n in d.names() {
        let (dtype, shape, _) = &tensors[&n];
        match (dtype.as_str(), d.scales.get(&n)) {
            ("F8_E4M3", Some(s)) => {
                assert_eq!((s, d.shapes.get(&n)), (&format!("{n}_scale_inv"), Some(shape)), "{n}");
                fp8 += 1;
            }
            // (an expert's matrix: two 4-bit values a byte along a row, a byte's scale a run of 32)
            ("U8", Some(s)) if n.ends_with(".weight") => {
                let want = &d.shapes[&n];
                assert_eq!((s, shape), (&format!("{n}_scale"), &vec![want[0], want[1] / 2]), "{n}");
                assert_eq!(tensors[s].1, [want[0], want[1] / 32], "{s}");
                fp4 += 1;
            }
            (_, None) if scale_names.contains(&n) => {}
            (_, None) => {
                assert!(dtype == "BF16" || n.ends_with("e_score_correction_bias"), "{n}: {dtype} with no scales named");
                assert_eq!(d.shapes.get(&n), Some(shape), "{n}");
                sinks += n.ends_with("attention_sink_bias") as usize;
            }
            (t, s) => panic!("{n}: {t}, scales {s:?}"),
        }
    }
    // (48 + 3 fused matrices and the three matrices of the dense MLPs of layer 0 and the draft layers; 47 layers of 256 experts of three matrices)
    assert_eq!((fp8, fp4, sinks), (51 + 4 * 3, 47 * 256 * 3, 39 + 3));
    // (a fused matrix's scales start again at every shard: 4 shards of ceil(3,392 / 128) rows of blocks, where the whole matrix's rows are 106 blocks)
    assert_eq!(tensors["model.layers.0.self_attn.qkv_proj.weight_scale_inv"].1, [4 * 27, 32]);
    assert_eq!(d.unread, ["audio_encoder.", "speech_embeddings.", "visual."]);
    assert!(have.iter().any(|n| n.starts_with("visual.")) && !d.names().iter().any(|n| n.starts_with("visual.") || n.starts_with("audio_encoder.")));
    d.check(&have, true).unwrap_or_else(|e| panic!("{}", e.msg));
}

/// A small config of the MiMo V2 layout with `keys` over it: two layers, the first over the whole context with a dense MLP, the second sliding with
/// experts; queries and keys of 8 and values of 4, the fused matrix in two shards.
fn mimo_tiny(keys: serde_json::Value) -> serde_json::Value {
    let mut c = serde_json::json!({
        "architectures": ["MiMoV2ForCausalLM"], "model_type": "mimo_v2", "hidden_size": 32, "num_hidden_layers": 2, "vocab_size": 61,
        "layernorm_epsilon": 1e-6, "hidden_act": "silu", "max_position_embeddings": 4096, "attention_projection_layout": "fused_qkv",
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "v_head_dim": 4, "rope_theta": 10000.0,
        "swa_num_attention_heads": 4, "swa_num_key_value_heads": 4, "swa_head_dim": 8, "swa_v_head_dim": 4, "swa_rope_theta": 100.0,
        "partial_rotary_factor": 0.5, "hybrid_layer_pattern": [0, 1], "sliding_window_size": 2, "add_swa_attention_sink_bias": true,
        "attention_value_scale": 0.5, "moe_layer_freq": [0, 1], "intermediate_size": 48, "moe_intermediate_size": 16, "n_routed_experts": 6,
        "num_experts_per_tok": 2, "scoring_func": "sigmoid", "topk_method": "noaux_tc", "norm_topk_prob": true, "n_group": 1, "topk_group": 1,
    });
    c.as_object_mut().unwrap().extend(keys.as_object().unwrap().clone());
    c
}

/// The MiMo V2 builder reads a config as the release's code does: a window of `sliding_window_size` and the query's own position, a sink where a
/// sort's key says, values as wide as the head and not scaled where unsaid, a matrix a product where the layout is "split" or unsaid, no draft heads
/// unless counted, a quantization_config's formats named as scales; and what it does not know is refused by name.
#[test]
fn mimo_v2_reads_its_config_as_the_releases_code_does() {
    use super::describe::Source;
    use super::kinds::{Body, Ffn, Mixer};
    let attn = |d: &super::describe::Described, layer: usize| match &d.desc.blocks[2 * layer].body {
        Body::Mixer(Mixer::Attention(a)) => a.clone(),
        _ => panic!(),
    };
    let d = describe(&mimo_tiny(serde_json::json!({}))).unwrap_or_else(|e| panic!("{}", e.msg));
    let (a, b) = (attn(&d, 0), attn(&d, 1));
    assert_eq!((a.window, a.sinks, a.kv_heads, a.v_head_dim, a.v_scale), (None, false, 2, 4, 0.5));
    assert_eq!((b.window, b.sinks, b.kv_heads, b.v_head_dim, b.v_scale), (Some(3), true, 4, 4, 0.5));
    assert_eq!((a.rope.as_ref().unwrap().theta, b.rope.as_ref().unwrap().theta, a.rope.as_ref().unwrap().dim), (1e4, 100.0, 4));
    assert!(matches!(&d.desc.blocks[1].body, Body::Ffn(Ffn::Mlp(_))) && matches!(&d.desc.blocks[3].body, Body::Ffn(Ffn::Moe(m)) if m.experts == 6));
    assert!(d.desc.mtp.is_empty() && d.scales.is_empty() && d.fp8_block.is_none());
    // (two shards of 2 query heads, a key head and a value head on the first layer: 16 + 8 + 4 rows each)
    assert_eq!(d.shapes["model.layers.0.self_attn.qkv_proj.weight"], [56, 32]);
    assert!(matches!(&d.blocks[0].iter().find(|(r, _)| r == "k").unwrap().1, Source::RowsOf(_, rows) if rows[..] == [(16, 24), (44, 52)]));
    assert_eq!(d.shapes["model.layers.1.self_attn.attention_sink_bias"], [4]);
    // (unsaid: the values as wide as the head and not scaled, a matrix a product, no sinks)
    let mut plain = mimo_tiny(serde_json::json!({}));
    for key in ["v_head_dim", "swa_v_head_dim", "attention_value_scale", "attention_projection_layout", "add_swa_attention_sink_bias"] {
        plain.as_object_mut().unwrap().remove(key);
    }
    let p = describe(&plain).unwrap_or_else(|e| panic!("{}", e.msg));
    assert_eq!((attn(&p, 1).v_head_dim, attn(&p, 1).v_scale, attn(&p, 1).sinks), (8, 1.0, false));
    assert_eq!(p.shapes["model.layers.1.self_attn.v_proj.weight"], [32, 32]);
    assert!(!p.names().iter().any(|n| n.contains("qkv_proj") || n.contains("attention_sink_bias")));
    // (a draft layer is a head of the sliding sort; a quantization_config names the FP8 matrices' scales and the experts', and leaves out what it lists)
    let quant = serde_json::json!({"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128], "store_dtype": "mxfp4", "mxfp4_block_size": 32,
        "activation_scheme": "dynamic", "ignored_layers": ["model.layers.0.self_attn.o_proj", "model.decoder.self_attn.o_proj"]});
    let q = describe(&mimo_tiny(serde_json::json!({"num_nextn_predict_layers": 1, "quantization_config": quant}))).unwrap_or_else(|e| panic!("{}", e.msg));
    assert_eq!((q.desc.mtp.len(), q.fp8_block), (1, Some(128)));
    assert!(matches!(&q.desc.mtp[0].blocks[0].body, Body::Mixer(Mixer::Attention(a)) if a.window == Some(3) && a.sinks && a.kv_heads == 4));
    let scale = |n: &str| q.scales.get(n).map(String::as_str);
    assert_eq!(scale("model.layers.0.self_attn.qkv_proj.weight"), Some("model.layers.0.self_attn.qkv_proj.weight_scale_inv"));
    assert_eq!(scale("model.layers.1.mlp.experts.5.up_proj.weight"), Some("model.layers.1.mlp.experts.5.up_proj.weight_scale"));
    assert_eq!(scale("model.mtp.layers.0.mlp.down_proj.weight"), Some("model.mtp.layers.0.mlp.down_proj.weight_scale_inv"));
    assert_eq!(scale("model.layers.1.self_attn.o_proj.weight"), Some("model.layers.1.self_attn.o_proj.weight_scale_inv"));
    for left in ["model.layers.0.self_attn.o_proj.weight", "model.mtp.layers.0.self_attn.o_proj.weight", "model.layers.1.mlp.gate.weight", "lm_head.weight"] {
        assert_eq!(scale(left), None, "{left}");
    }
    let quant_with = |key: &str, v: serde_json::Value| {
        let mut q = quant.clone();
        q[key] = v;
        serde_json::json!({"quantization_config": q})
    };
    for (keys, word) in [
        (serde_json::json!({"attention_bias": true}), "attention_bias"),
        (serde_json::json!({"hidden_act": "gelu"}), "hidden_act"),
        (serde_json::json!({"hybrid_layer_pattern": [0, 2]}), "hybrid_layer_pattern[1] = 2"),
        (serde_json::json!({"hybrid_layer_pattern": [0]}), "hybrid_layer_pattern has 1 entries for 2 layers"),
        (serde_json::json!({"moe_layer_freq": 1}), "no list moe_layer_freq"),
        (serde_json::json!({"attention_projection_layout": "fused_kv"}), "attention_projection_layout fused_kv"),
        (serde_json::json!({"swa_num_key_value_heads": 1}), "a fused qkv_proj of 2 shards"),
        (serde_json::json!({"swa_num_key_value_heads": 3}), "not a whole number of heads a key head"),
        (serde_json::json!({"swa_head_dim": null}), "no swa_head_dim"),
        (serde_json::json!({"sliding_window_size": null}), "no sliding_window_size"),
        (serde_json::json!({"sliding_window": 5}), "sliding_window = 5"),
        (serde_json::json!({"rope_parameters": {"rope_type": "yarn", "rope_theta": 10000.0, "factor": 4.0}}), "rope_parameters.rope_type yarn"),
        (serde_json::json!({"rope_parameters": {"rope_type": "default", "rope_theta": 500.0}}), "rope_parameters.rope_theta = 500.0"),
        (serde_json::json!({"scoring_func": "softmax"}), "scoring_func"),
        (serde_json::json!({"topk_method": "greedy"}), "topk_method"),
        (serde_json::json!({"moe_router_dtype": "float16"}), "moe_router_dtype float16"),
        (serde_json::json!({"n_shared_experts": 1}), "n_shared_experts"),
        (serde_json::json!({"routed_scaling_factor": 2.5}), "routed_scaling_factor = 2.5"),
        (serde_json::json!({"hybrid_block_size": 8}), "hybrid_block_size"),
        (serde_json::json!({"attention_value_scale": 0.0}), "attention_value_scale"),
        (serde_json::json!({"rms_norm_eps": 1e-5}), "config key rms_norm_eps is not understood"),
        (serde_json::json!({"layernorm_epsilon": null}), "no layernorm_epsilon"),
        (quant_with("fmt", "e5m2".into()), "where the description reads fp8 e4m3"),
        (quant_with("weight_block_size", serde_json::Value::Null), "weight_block_size"),
        (quant_with("store_dtype", "nvfp4".into()), "store_dtype"),
        (quant_with("mxfp4_block_size", 16.into()), "mxfp4_block_size"),
        (quant_with("kv_cache_scheme", "static".into()), "config key kv_cache_scheme is not understood"),
    ] {
        let e = describe(&mimo_tiny(keys)).unwrap_err().msg;
        assert!(e.contains(word), "{word}: {e}");
    }
    // (the release writes `sliding_window` as the size; transformers' earlier class means the size and the query's own position by it)
    for w in [2, 3] {
        assert_eq!(attn(&describe(&mimo_tiny(serde_json::json!({"sliding_window": w}))).unwrap(), 1).window, Some(3));
    }
}

/// The MiMo V2 layout on the CPU reference, a random model of each small config (values half as wide as the queries and keys, the fused matrix in
/// two shards). The values' factor is read: at 0.5 the answer is not the one at 1, and it is, to the bit, the one of a model at 1 whose `o_proj` is
/// halved (the product is linear in the values, and a half is exact: the fold the release's loader makes). A sink takes part in the softmax: far
/// below the scores it is no sink, at the scores' size it is another answer. The fused matrix is its shards' rows: a model of a matrix a product,
/// given the rows the description cuts, answers the same to the bit. Two sliding layers of the query's own position and 2 before it do not see a
/// token five back, and a layer over the whole context does.
#[test]
fn mimo_v2_on_the_reference_scales_values_keeps_sinks_and_cuts_the_fused_matrix() {
    // (a fixed draw a tensor: its name's bytes seed it; a norm's weights near 1)
    fn draw(n: &str, len: usize) -> Vec<f32> {
        let mut s = n.bytes().fold(0x9E37_79B9_7F4A_7C15u64, |a, b| (a ^ b as u64).wrapping_mul(0x0100_0000_01B3));
        let norm = n.contains("norm");
        (0..len)
            .map(|_| {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                let u = ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5;
                if norm {
                    1.0 + 0.2 * u
                } else {
                    0.6 * u
                }
            })
            .collect()
    }
    // (`sink`: every sink's value, None a model with no sinks; `o_x`: a factor on every `o_proj`)
    let run = |cfg: &serde_json::Value, sink: Option<f32>, o_x: f32, toks: &[u32]| -> Vec<f32> {
        let mut d = describe(cfg).unwrap_or_else(|e| panic!("{}", e.msg));
        if sink.is_none() {
            for b in d.desc.blocks.iter_mut() {
                if let super::kinds::Body::Mixer(super::kinds::Mixer::Attention(a)) = &mut b.body {
                    a.sinks = false;
                }
            }
        }
        let fused = describe(&mimo_tiny(serde_json::json!({"hybrid_layer_pattern": cfg["hybrid_layer_pattern"]}))).unwrap();
        let src = |n: &str| {
            let shape = d.shapes.get(n)?.clone();
            let len: usize = shape.iter().product();
            if n.ends_with(".attention_sink_bias") {
                return Some((shape, vec![sink.unwrap_or(0.0); len]));
            }
            // (a split layout's product: its rows of the fused matrix the same layer of the fused layout holds, shard by shard)
            if let Some((at, part)) = n.strip_suffix("_proj.weight").and_then(|x| x.rsplit_once(".self_attn.")).filter(|(_, p)| ["q", "k", "v"].contains(p)) {
                let whole = format!("{at}.self_attn.qkv_proj.weight");
                let rows = fused.shapes[&whole][0];
                let w = draw(&whole, rows * 32);
                // (two shards; a shard's rows are its 2 query heads', then its keys', then its values', the keys twice as wide as the values)
                let rest = rows / 2 - 16;
                let (q, k, v) = (16, 2 * rest / 3, rest / 3);
                let (from, many) = match part {
                    "q" => (0, q),
                    "k" => (q, k),
                    _ => (q + k, v),
                };
                let cut: Vec<f32> = (0..2).flat_map(|i| w[(i * rows / 2 + from) * 32..(i * rows / 2 + from + many) * 32].to_vec()).collect();
                assert_eq!(cut.len(), len, "{n}");
                return Some((shape, cut));
            }
            let x = if n.ends_with(".o_proj.weight") { o_x } else { 1.0 };
            Some((shape, draw(n, len).into_iter().map(|v| v * x).collect()))
        };
        let mut m = RefModel::load(&d, &src).unwrap_or_else(|e| panic!("{}", e.msg));
        m.forward(toks, 0).unwrap_or_else(|e| panic!("{}", e.msg)).pop().unwrap()
    };
    let (a, first): ([u32; 6], [u32; 6]) = ([5, 9, 13, 17, 21, 25], [44, 9, 13, 17, 21, 25]);
    let base = mimo_tiny(serde_json::json!({}));
    let row = run(&base, Some(0.0), 1.0, &a);
    assert!(row.len() == 61 && row.iter().all(|x| x.is_finite()) && row.iter().any(|&x| x != row[0]));
    // the values' factor
    let unscaled = mimo_tiny(serde_json::json!({"attention_value_scale": 1.0}));
    assert_ne!(row, run(&unscaled, Some(0.0), 1.0, &a), "values times 0.5 are another answer than values as they are");
    assert_eq!(row, run(&unscaled, Some(0.0), 0.5, &a), "values times 0.5 are o_proj times 0.5");
    // the sinks (the second layer's)
    assert_eq!(run(&base, Some(-1e9), 1.0, &a), run(&base, None, 1.0, &a), "a sink far below the scores is no sink");
    assert_ne!(row, run(&base, None, 1.0, &a), "a sink at the scores' size takes its share of the softmax");
    // the fused matrix, shard by shard
    assert_eq!(
        row,
        run(&mimo_tiny(serde_json::json!({"attention_projection_layout": "split"})), Some(0.0), 1.0, &a),
        "a matrix a product, of the fused one's rows"
    );
    // the window
    assert_ne!(row, run(&base, Some(0.0), 1.0, &first), "a layer over the whole context sees the first token");
    let windowed = mimo_tiny(serde_json::json!({"hybrid_layer_pattern": [1, 1]}));
    assert_eq!(run(&windowed, Some(0.0), 1.0, &a), run(&windowed, Some(0.0), 1.0, &first), "two layers of three positions see four back");
    assert_ne!(
        run(&windowed, Some(0.0), 1.0, &a),
        run(&mimo_tiny(serde_json::json!({"hybrid_layer_pattern": [1, 1], "sliding_window_size": 5})), Some(0.0), 1.0, &a)
    );
}

/// LFM: the released LFM3-1.7B (its recorded header) describes as 24 layers of a mixer and the gated MLP, each part between two norms: 13 gated short
/// convolutions of 7 taps with no biases, 6 attentions over the whole context and 5 with a window of 256; every attention 16 heads over 4 of 128, its
/// queries and keys normed with a weight, its output gated a head; the full layers' rotary the release's reading of "default" with a partial factor of
/// 0.5 (the whole head paired, the first 32 of the 64 pairs turning, exponents over 128, base 1,000,000), the sliding layers' plain at 10,000. The head
/// is the embedding, the last norm the checkpoint's `embedding_norm`. Every shape the description gives is the checkpoint's, and it is read whole.
#[test]
fn lfm3_describes_its_three_sorts_of_layers() {
    use super::kinds::{Act, Body, Ffn, GateFrom, Mixer, NormWeight, OutGate, Rope, ShortConv};
    let h = fixture("headers/LFM3-1.7B.json");
    let cfg: serde_json::Value = serde_json::from_str(h["config"].as_str().unwrap()).unwrap();
    let d = describe(&cfg).unwrap_or_else(|e| panic!("{}", e.msg));
    assert_eq!((d.desc.blocks.len(), d.desc.head.tied, d.head.is_none(), d.desc.hidden, d.desc.vocab), (48, true, true, 2048, 128_000));
    assert_eq!((d.final_norm.as_str(), d.desc.final_norm.eps, d.desc.embed.norm), ("model.embedding_norm.weight", 1e-5, None));
    let (mut conv, mut full, mut sliding) = (Vec::new(), Vec::new(), Vec::new());
    for (i, b) in d.desc.blocks.iter().enumerate() {
        assert!(b.norm_out.is_some() && b.norm_in.eps == 1e-5, "block {i}: a norm before the part and one after");
        match &b.body {
            Body::Mixer(Mixer::ShortConv(c)) => {
                assert_eq!((*c, c.state(2048)), (ShortConv { taps: 7, bias: false }, 6 * 2048), "block {i}");
                conv.push(i / 2);
            }
            Body::Mixer(Mixer::Attention(a)) => {
                assert_eq!((a.heads, a.kv_heads, a.head_dim, a.v_head_dim, a.bias, a.sinks), (16, 4, 128, 128, false, false), "block {i}");
                assert_eq!((a.q_norm.unwrap().weight, a.k_norm.unwrap().weight, a.v_norm), (NormWeight::Plain, NormWeight::Plain, None), "block {i}");
                assert_eq!(a.gate, Some(OutGate { act: Act::Sigmoid, from: GateFrom::Head, norm: None }), "block {i}");
                assert_eq!(a.scale, 128f32.powf(-0.5), "block {i}");
                let r = a.rope.clone().unwrap();
                match a.window {
                    None => {
                        assert_eq!(r, Rope::proportional(1e6, 128, 0.5), "block {i}");
                        assert_eq!((r.dim, r.freqs, r.freq_dim), (128, 32, 128), "block {i}");
                        full.push(i / 2);
                    }
                    Some(256) => {
                        assert_eq!(r, Rope::plain(1e4, 128), "block {i}");
                        sliding.push(i / 2);
                    }
                    w => panic!("block {i}: a window of {w:?}"),
                }
            }
            Body::Ffn(Ffn::Mlp(m)) => assert!(i % 2 == 1 && m.gated && m.act == Act::Silu && !m.bias, "block {i}"),
            _ => panic!("block {i} is no kind of the layout"),
        }
    }
    assert_eq!((conv.len(), full, sliding), (13, vec![2, 6, 10, 14, 18, 22], vec![4, 8, 12, 16, 20]));
    assert_eq!(conv, [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]);
    // (every tensor read has the checkpoint's shape: the convolution's [2048, 1, 7], its input product's [6144, 2048], a gate's [16, 2048])
    let table = crate::sizes::Table::expand(&crate::sizes::groups_of(&h).unwrap());
    let have = header_names(&h);
    assert_eq!((d.names().len(), have.len()), (286, 286));
    for r in &table.rows {
        let want: Vec<usize> = r.shape.iter().map(|&x| x as usize).collect();
        assert_eq!(d.shapes.get(&r.name), Some(&want), "{}", r.name);
    }
    for (n, s) in [("conv.conv", vec![2048, 1, 7]), ("conv.in_proj", vec![6144, 2048]), ("conv.out_proj", vec![2048, 2048])] {
        assert_eq!(d.shapes.get(&format!("model.layers.0.{n}.weight")), Some(&s), "{n}");
    }
    assert_eq!(d.shapes.get("model.layers.2.self_attn.gate_proj.weight"), Some(&vec![16, 2048]));
    d.check(&have, true).unwrap_or_else(|e| panic!("{}", e.msg));
}

/// A small config of the LFM layout: heads of 8 (4 over 2), a convolution of 3 taps, the layers' sorts as given, and `keys` over them.
fn lfm_tiny(sorts: &[&str], keys: serde_json::Value) -> serde_json::Value {
    let mut c = serde_json::json!({
        "architectures": ["Lfm3ForCausalLM"], "model_type": "lfm3", "hidden_size": 32, "num_hidden_layers": sorts.len(), "vocab_size": 61,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "intermediate_size": 48, "norm_eps": 1e-5, "tie_word_embeddings": true,
        "max_position_embeddings": 4096, "conv_L_cache": 3, "conv_bias": false, "sliding_window": 3, "use_sandwich_norm": true, "layer_types": sorts,
        "rope_parameters": {"full_attention": {"rope_type": "default", "rope_theta": 1000000.0, "partial_rotary_factor": 0.5},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}},
    });
    c.as_object_mut().unwrap().extend(keys.as_object().unwrap().clone());
    c
}

/// The LFM builder reads its config as the release's code does: what a config leaves unsaid is the class's (the layers' sorts by their number, the two
/// rotaries, a window of 256, 7 taps, the gate and the second norms on, the head tied); a rotary written "proportional" is the one written "default"
/// with the same partial factor; each switch takes away what it names and its tensors with it; and what the builder does not take is refused in words
/// that name the key and the value.
#[test]
fn lfm3_reads_its_config_as_the_releases_code_does() {
    use super::kinds::{Body, Mixer, Rope};
    let ok = lfm_tiny(&["conv", "full_attention", "sliding_attention"], serde_json::json!({}));
    let base = describe(&ok).unwrap_or_else(|e| panic!("{}", e.msg));
    let attn = |d: &super::describe::Described, b: usize| match &d.desc.blocks[b].body {
        Body::Mixer(Mixer::Attention(a)) => a.clone(),
        other => panic!("block {b}: {other:?}"),
    };
    assert_eq!((attn(&base, 2).rope, attn(&base, 4).rope, attn(&base, 4).window), (Some(Rope::proportional(1e6, 8, 0.5)), Some(Rope::plain(1e4, 8)), Some(3)));
    // (the class's own: only the model type and the sizes that keep it small said)
    let bare = serde_json::json!({"model_type": "lfm3", "hidden_size": 32, "num_hidden_layers": 9, "vocab_size": 61, "intermediate_size": 48});
    let d = describe(&bare).unwrap_or_else(|e| panic!("{}", e.msg));
    let sorts: Vec<&str> = (0..9)
        .map(|l| match &d.desc.blocks[2 * l].body {
            Body::Mixer(Mixer::ShortConv(c)) => {
                assert_eq!((c.taps, c.bias), (7, false));
                "conv"
            }
            Body::Mixer(Mixer::Attention(a)) if a.window == Some(256) => "sliding",
            Body::Mixer(Mixer::Attention(a)) if a.window.is_none() => "full",
            other => panic!("layer {l}: {other:?}"),
        })
        .collect();
    assert_eq!(sorts, ["conv", "conv", "full", "conv", "sliding", "conv", "full", "conv", "sliding"]);
    let (f, s) = (attn(&d, 4), attn(&d, 8));
    assert_eq!((f.heads, f.kv_heads, f.head_dim, f.gate.is_some(), d.desc.head.tied, d.desc.final_norm.eps), (16, 4, 2, true, true, 1e-5));
    assert_eq!((f.rope, s.rope), (Some(Rope::proportional(1e6, 2, 0.5)), Some(Rope::plain(1e4, 2))));
    assert!(d.desc.blocks.iter().all(|b| b.norm_out.is_some()));
    // ("proportional" written out is the same description)
    let mut prop = ok.clone();
    prop["rope_parameters"]["full_attention"]["rope_type"] = "proportional".into();
    assert_eq!(describe(&prop).unwrap().desc, base.desc);
    // (the switches: no gate, no second norms, biases on the convolution's parts, an untied head; the older tie key over the newer)
    let off = describe(&lfm_tiny(&["conv", "full_attention"], serde_json::json!({"use_attention_gate": false, "use_sandwich_norm": false}))).unwrap();
    assert!(attn(&off, 2).gate.is_none() && off.desc.blocks.iter().all(|b| b.norm_out.is_none()));
    assert!(!off.names().iter().any(|n| n.contains("gate_proj") || n.contains("post_")), "{:?}", off.names());
    let biased = describe(&lfm_tiny(&["conv"], serde_json::json!({"conv_bias": true, "tie_word_embeddings": false}))).unwrap();
    assert!(matches!(biased.desc.blocks[0].body, Body::Mixer(Mixer::ShortConv(c)) if c.bias && c.taps == 3));
    for n in ["conv.in_proj.bias", "conv.conv.bias", "conv.out_proj.bias"] {
        assert!(biased.shapes.contains_key(&format!("model.layers.0.{n}")), "{n}");
    }
    assert_eq!(biased.head.as_deref(), Some("lm_head.weight"));
    assert!(!describe(&lfm_tiny(&["conv"], serde_json::json!({"tie_embedding": false}))).unwrap().desc.head.tied);
    // (the refusals)
    let refused = |sorts: &[&str], keys: serde_json::Value| describe(&lfm_tiny(sorts, keys)).err().map(|e| e.msg).unwrap_or_default();
    let all = ["conv", "full_attention", "sliding_attention"];
    let e = refused(&["conv", "mamba"], serde_json::json!({}));
    assert!(e.contains("layer_types[1]") && e.contains("mamba") && e.contains("conv, full_attention or sliding_attention"), "{e}");
    let e = refused(&all, serde_json::json!({"layer_types": ["conv"]}));
    assert!(e.contains("layer_types has 1 entries for 3 layers"), "{e}");
    let e = refused(&all, serde_json::json!({"rope_parameters": {"full_attention": {"rope_theta": 1e6}}}));
    assert!(e.contains("rope_parameters has no sliding_attention"), "{e}");
    let e = refused(&all, serde_json::json!({"rope_parameters": {"conv": {"rope_theta": 1e6}}}));
    assert!(e.contains("rope_parameters.conv is not an attention layer type"), "{e}");
    let yarn = serde_json::json!({"rope_type": "yarn", "rope_theta": 1e6, "partial_rotary_factor": 0.5, "factor": 2.0, "original_max_position_embeddings": 64});
    let e = refused(&all, serde_json::json!({"rope_parameters": {"full_attention": yarn, "sliding_attention": {"rope_theta": 1e4}}}));
    assert!(e.contains("rope_parameters.full_attention") && e.contains("rope_type yarn with partial_rotary_factor 0.5"), "{e}");
    let e = refused(
        &all,
        serde_json::json!({"rope_parameters": {"full_attention": {"rope_type": "dynamic", "rope_theta": 1e6, "factor": 2.0}, "sliding_attention": {"rope_theta": 1e4}}}),
    );
    assert!(e.contains("rope_type dynamic is not supported"), "{e}");
    let e = refused(&all, serde_json::json!({"sliding_window": null}));
    assert!(e.contains("sliding_window = null"), "{e}");
    describe(&lfm_tiny(&["conv", "full_attention"], serde_json::json!({"sliding_window": null}))).unwrap_or_else(|e| panic!("{}", e.msg));
    let e = refused(&all, serde_json::json!({"attention_bias": true}));
    assert!(e.contains("attention_bias = true is not supported"), "{e}");
    let e = refused(&all, serde_json::json!({"conv_L_cache": 0}));
    assert!(e.contains("conv_L_cache = 0"), "{e}");
    let e = refused(&all, serde_json::json!({"rms_norm_eps": 1e-6}));
    assert!(e.contains("norm_eps") && e.contains("rms_norm_eps") && e.contains("differ"), "{e}");
    let e = refused(&all, serde_json::json!({"num_key_value_heads": 3}));
    assert!(e.contains("4 query heads to 3 KV heads"), "{e}");
    let e = refused(&all, serde_json::json!({"block_ffn_dim_multiplier": 1.0, "hidden_act": "gelu"}));
    assert!(e.contains("config keys block_ffn_dim_multiplier, hidden_act are not understood"), "{e}");
}

/// The LFM layout on the CPU reference, a random model of each small config (every tensor of the shape its description gives). The convolution is causal
/// and remembers exactly its taps: with two conv layers of 3 taps a row reads 4 tokens back, so the last row of a prompt of seven does not change, to the
/// bit, when the second token does, and changes when the third does; no row changes when a later token does. A prompt at once and the same prompt a token
/// at a time (the convolutions' states and the attentions' caches kept between the calls) are the same bits, in a model of all three sorts of layers.
/// The gate a head changes the answer, the second norms change it, and a convolution of other taps is another answer.
#[test]
fn lfm3_on_the_reference_is_causal_keeps_its_taps_and_steps_as_it_prompts() {
    let load = |cfg: &serde_json::Value| -> RefModel {
        let d = describe(cfg).unwrap_or_else(|e| panic!("{}", e.msg));
        let src = |n: &str| {
            let shape = d.shapes.get(n)?.clone();
            let len: usize = shape.iter().product();
            // (a fixed draw a tensor: its name's bytes seed it; a norm's weights near 1)
            let mut s = n.bytes().fold(0x9E37_79B9_7F4A_7C15u64, |a, b| (a ^ b as u64).wrapping_mul(0x0100_0000_01B3));
            let norm = n.contains("norm");
            let v = (0..len)
                .map(|_| {
                    s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                    let u = ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5;
                    if norm {
                        1.0 + 0.2 * u
                    } else {
                        0.6 * u
                    }
                })
                .collect();
            Some((shape, v))
        };
        RefModel::load(&d, &src).unwrap_or_else(|e| panic!("{}", e.msg))
    };
    let run = |cfg: &serde_json::Value, toks: &[u32]| load(cfg).forward(toks, 0).unwrap_or_else(|e| panic!("{}", e.msg));
    let none = serde_json::json!({});
    let a: [u32; 7] = [5, 9, 13, 17, 21, 25, 29];
    let with = |i: usize, t: u32| {
        let mut b = a;
        b[i] = t;
        b
    };
    // (the taps' memory: two conv layers of 3 taps reach 2 + 2 tokens back and no further)
    let convs = lfm_tiny(&["conv", "conv"], none.clone());
    let rows = run(&convs, &a);
    assert_eq!((rows.len(), rows[6].len()), (7, 61));
    assert!(rows.iter().flatten().all(|x| x.is_finite()) && rows[6].iter().any(|&x| x != rows[6][0]), "finite, and not one value");
    assert_eq!(rows[6], run(&convs, &with(1, 44))[6], "five tokens back is past two layers of three taps");
    assert_ne!(rows[6], run(&convs, &with(2, 44))[6], "four tokens back is inside them");
    let one = lfm_tiny(&["conv"], none.clone());
    let r1 = run(&one, &a);
    assert_eq!(r1[6], run(&one, &with(3, 44))[6], "three tokens back is past one layer of three taps");
    assert_ne!(r1[6], run(&one, &with(4, 44))[6]);
    // (other taps, another answer; one tap reads its own token alone)
    assert_ne!(r1[6], run(&lfm_tiny(&["conv"], serde_json::json!({"conv_L_cache": 5})), &a)[6]);
    let single = lfm_tiny(&["conv"], serde_json::json!({"conv_L_cache": 1}));
    assert_eq!(run(&single, &a)[6], run(&single, &with(5, 44))[6]);
    // (causal: a later token changes no earlier row, in a model of all three sorts; and a row over the whole context sees the first token)
    let sorts = ["conv", "full_attention", "conv", "sliding_attention"];
    let mixed = lfm_tiny(&sorts, none.clone());
    let rows = run(&mixed, &a);
    let later = run(&mixed, &with(6, 44));
    assert_eq!(rows[..6], later[..6], "a later token changes no earlier row");
    assert_ne!(rows[6], later[6]);
    assert_ne!(rows[6], run(&mixed, &with(0, 44))[6], "a layer over the whole context sees the first token");
    // (a prompt at once, then two more tokens; and every token alone, in order: the same bits)
    let mut m = load(&mixed);
    let mut by_step = Vec::new();
    for (t, &tok) in a.iter().enumerate() {
        by_step.extend(m.forward(&[tok], t as u32).unwrap_or_else(|e| panic!("{}", e.msg)));
    }
    assert_eq!(by_step, rows, "a token at a time is the prompt");
    let mut m = load(&mixed);
    let mut parts = m.forward(&a[..5], 0).unwrap();
    parts.extend(m.forward(&a[5..], 5).unwrap());
    assert_eq!(parts, rows, "a prompt, then its next tokens");
    // (the gate a head and the second norms are read, not ignored)
    assert_ne!(rows[6], run(&lfm_tiny(&sorts, serde_json::json!({"use_attention_gate": false})), &a)[6], "the gate changes the answer");
    assert_ne!(rows[6], run(&lfm_tiny(&sorts, serde_json::json!({"use_sandwich_norm": false})), &a)[6], "the second norms change it");
    assert_ne!(rows[6], run(&lfm_tiny(&sorts, serde_json::json!({"conv_bias": true})), &a)[6], "the convolution's biases change it");
}

/// Ling: the released Ling-3.2-air (its recorded header) describes as 42 layers on a plain residual. Its mixers are 35 Kimi delta attentions (32 heads of
/// 128, a convolution of 4 taps a product, the bounded decay at -5, a sigmoid-gated norm) and 7 latent attentions, the last layer of every six (32
/// heads, a latent of 512, 128 + 64 query channels of which the 64 turn in interleaved pairs at 6,000,000, values of 128, a sigmoid gate of one value a
/// head). Its first two feed-forwards are gated MLPs and the other 40 are 512 experts in 8 groups of which 4 are kept, the best 8 divided by their sum
/// and times 2.5, beside a shared MLP; the experts' SwiGLU limit holds the activated gate, 4 on the last seven layers, and the shared MLP's 5 then 7 on
/// the last eight. Its draft layer is a draft head of a latent attention and experts with no limit. Every shape is the checkpoint's, every FP8 tensor
/// of the header has its scales named and no other has, and the checkpoint is read whole but the vision tower and its projector.
#[test]
fn ling_3_2_air_describes_its_two_mixers_its_experts_and_its_draft_head() {
    use super::kinds::{Act, Body, Clamp, Decay, Ffn, GateFrom, LinearKind, Mixer, MtpInput, OutGate, Rope, RopeStyle, RouterScore};
    let h = fixture("headers/Ling-3.2-air.json");
    let cfg: serde_json::Value = serde_json::from_str(h["config"].as_str().unwrap()).unwrap();
    let d = describe(&cfg).unwrap_or_else(|e| panic!("{}", e.msg));
    assert_eq!((d.desc.blocks.len(), d.desc.head.tied, d.desc.final_norm.eps, d.desc.embed.scale, d.fp8_block), (84, false, 1e-6, 1.0, Some(128)));
    assert_eq!((d.desc.hidden, d.desc.vocab, d.convention), (2560, 157184, "Ling"));
    let rope = Rope { style: RopeStyle::Interleaved, ..Rope::plain(6e6, 64) };
    assert_eq!(d.desc.rope, Some(rope.clone()));
    let gate = Some(OutGate { act: Act::Sigmoid, from: GateFrom::Head, norm: None });
    let (mut delta, mut latent, mut dense, mut sparse) = (0, 0, 0, 0);
    for (i, b) in d.desc.blocks.iter().enumerate() {
        let l = i / 2;
        assert!(b.norm_out.is_none() && b.residual == super::kinds::Residual::Add, "block {i}");
        match &b.body {
            Body::Mixer(Mixer::Linear(x)) => {
                assert!(l % 6 != 5, "layer {l}");
                assert_eq!((x.kind, x.k_heads, x.v_heads, x.k_dim, x.v_dim, x.conv), (LinearKind::KimiDelta, 32, 32, 128, 128, 4), "layer {l}");
                assert_eq!((x.decay, x.gate_act, x.norm_eps), (Decay::Bounded(-5.0), Act::Sigmoid, 1e-6), "layer {l}");
                delta += 1;
            }
            Body::Mixer(Mixer::Mla(x)) => {
                assert!(l % 6 == 5, "layer {l}");
                assert_eq!((x.heads, x.q_lora, x.kv_lora, x.nope, x.rope_dim, x.v_dim), (32, None, 512, 128, 64, 128), "layer {l}");
                assert_eq!((x.rope.as_ref(), x.gate, x.scale, x.kv_norm.eps), (Some(&rope), gate, 192f32.powf(-0.5), 1e-6), "layer {l}");
                assert!(x.indexer.is_none(), "layer {l}");
                latent += 1;
            }
            Body::Ffn(Ffn::Mlp(m)) => {
                assert!(l < 2 && m.gated && !m.bias && m.act == Act::Silu && m.clamp.is_none(), "layer {l}");
                dense += 1;
            }
            Body::Ffn(Ffn::Moe(m)) => {
                let r = &m.router;
                assert_eq!((m.experts, r.top_k, r.groups, r.renorm, r.bias, r.scale), (512, 8, Some((8, 4)), true, true, 2.5), "layer {l}");
                assert!(r.score == RouterScore::Sigmoid && !r.hash && m.latent.is_none() && !m.shared_gate, "layer {l}");
                // (the limits a layer, as the config's two lists say them)
                let limit = |key: &str| cfg["text_config"][key][l].as_f64().filter(|x| *x != 0.0).map(|x| Clamp::Activated(x as f32));
                assert_eq!(
                    (m.expert.clamp, m.shared.unwrap().clamp),
                    (limit("expert_swiglu_limit_list"), limit("share_expert_swiglu_limit_list")),
                    "layer {l}"
                );
                assert_eq!(m.expert.clamp, (l >= 35).then_some(Clamp::Activated(4.0)), "layer {l}");
                assert_eq!(m.shared.unwrap().clamp, (l >= 34).then_some(Clamp::Activated(if l >= 40 { 7.0 } else { 5.0 })), "layer {l}");
                sparse += 1;
            }
            _ => panic!("block {i}"),
        }
    }
    assert_eq!((delta, latent, dense, sparse), (35, 7, 2, 40));
    // (the draft layer: one head of the joined form, a latent attention with its gate a head and the experts, past the limit lists)
    assert_eq!((d.desc.mtp.len(), d.mtp.len(), d.mtp_layers), (1, 1, 1));
    let head = &d.desc.mtp[0];
    assert!(head.input == MtpInput::Joined && head.dspark.is_none() && head.blocks.len() == 2);
    assert!(matches!(&head.blocks[0].body, Body::Mixer(Mixer::Mla(x)) if x.gate == gate && x.heads == 32));
    assert!(matches!(&head.blocks[1].body, Body::Ffn(Ffn::Moe(m)) if m.expert.clamp.is_none() && m.shared.unwrap().clamp.is_none() && m.experts == 512));
    for n in ["model.layers.42.eh_proj.weight", "model.layers.42.final_layernorm.weight", "model.layers.42.attention.g_proj.weight"] {
        assert!(d.names().contains(n), "{n}");
    }
    assert_eq!((d.embed.as_str(), d.final_norm.as_str(), d.head.as_deref()), ("model.word_embeddings.weight", "model.norm.weight", Some("lm_head.weight")));
    // every tensor the description reads, at the checkpoint's shape and in the stored format the description says
    let have = header_names(&h);
    let tensors = header_tensors(&h);
    let scale_names: BTreeSet<&String> = d.scales.values().collect();
    let (mut fp8, mut plain) = (0, 0);
    for n in d.names() {
        let (dtype, shape, _) = &tensors[&n];
        match (dtype.as_str(), d.scales.get(&n)) {
            // (an FP8 matrix: its scales a block of 128 x 128, the last row and column of blocks what is left)
            ("F8_E4M3", Some(s)) => {
                assert_eq!((s, d.shapes.get(&n)), (&format!("{n}_scale_inv"), Some(shape)), "{n}");
                assert_eq!(tensors[s].1, [shape[0].div_ceil(128), shape[1].div_ceil(128)], "{s}");
                fp8 += 1;
            }
            (_, None) if scale_names.contains(&n) => {}
            (_, None) => {
                // (not FP8: the bf16 tensors, and the fp32 ones the release keeps so: the choice biases, A_log, dt_bias)
                let fp32 = n.ends_with("expert_bias") || n.ends_with("A_log") || n.ends_with("dt_bias");
                assert_eq!(dtype, if fp32 { "F32" } else { "BF16" }, "{n}: with no scales named");
                assert_eq!(d.shapes.get(&n), Some(shape), "{n}");
                plain += 1;
            }
            (t, s) => panic!("{n}: {t}, scales {s:?}"),
        }
    }
    // (35 delta attentions of six FP8 products and 7 latent attentions of four, the draft layer's of five with its gate; two dense MLPs and 41 shared
    // ones of three matrices; 41 layers of 512 experts of three)
    assert_eq!(fp8, 35 * 6 + 7 * 4 + 5 + 2 * 3 + 41 * 3 + 41 * 512 * 3);
    assert_eq!(d.names().len(), 2 * fp8 + plain);
    // (as released, bf16: the model's latent attentions' gates and every delta attention's beta, the routers, a draft head's joined product)
    for n in
        ["model.layers.5.attention.g_proj.weight", "model.layers.0.attention.b_proj.weight", "model.layers.2.mlp.gate.weight", "model.layers.42.eh_proj.weight"]
    {
        assert!(!d.scales.contains_key(n) && tensors[n].0 == "BF16", "{n}");
    }
    assert_eq!(tensors["model.layers.42.attention.g_proj.weight"].0, "F8_E4M3");
    assert_eq!(d.shapes["model.layers.5.attention.g_proj.weight"], [32, 2560]);
    assert_eq!(d.shapes["model.layers.0.attention.k_conv1d.weight"], [4096, 1, 4]);
    assert_eq!(d.unread, ["model.visual.", "linear_proj."]);
    assert!(have.iter().any(|n| n.starts_with("model.visual.")) && have.contains("linear_proj.0.weight"));
    assert!(!d.names().iter().any(|n| n.starts_with("model.visual.") || n.starts_with("linear_proj.")));
    d.check(&have, true).unwrap_or_else(|e| panic!("{}", e.msg));
    // (its layers have a loader of the dynamic route, the latent attention's with plain residuals; what the executor lacks of them, its lowering names)
    assert!(crate::dynamic::split::loader(&d).is_ok());
    let e = crate::dynamic::split::lowered_described(&d, 42, 4096).err().map(|e| e.msg).unwrap_or_default();
    assert!(e.contains("grouped choices"), "{e}");
}

/// A small config of the Ling layout in its wrapper with `keys` over its text config: five layers in groups of two (layers 1 and 3 latent attentions,
/// and layer 4, past the last whole group), the first layer's feed-forward dense and the others 8 experts in 2 groups; a limit on the activated gate
/// on the last two layers.
fn ling_tiny(keys: serde_json::Value) -> serde_json::Value {
    let mut text = serde_json::json!({
        "model_type": "bailing_hybrid", "hidden_size": 32, "num_hidden_layers": 5, "vocab_size": 61, "rms_norm_eps": 1e-6, "max_position_embeddings": 4096,
        "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 8, "short_conv_kernel_size": 3, "layer_group_size": 2,
        "no_kda_lora": true, "kda_safe_gate": true, "kda_lower_bound": -5.0,
        "q_lora_rank": null, "kv_lora_rank": 12, "qk_nope_head_dim": 8, "qk_rope_head_dim": 4, "v_head_dim": 6, "rope_theta": 6000000,
        "mrope_section": [0, 1, 1], "partial_rotary_factor": 0.5, "use_mla_nope": false, "gated_attention_proj_granularity_type": "head_wise",
    });
    // (the feed-forward's keys, and the earlier attention's that the layout does not read)
    let ffn = serde_json::json!({
        "intermediate_size": 48, "first_k_dense_replace": 1, "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 16,
        "moe_shared_expert_intermediate_size": 16, "n_group": 2, "topk_group": 1, "routed_scaling_factor": 2.5, "norm_topk_prob": true,
        "score_function": "sigmoid", "router_dtype": "fp32", "moe_router_enable_expert_bias": true,
        "expert_swiglu_limit_list": [0, 0, 0, 0.05, 0.05], "share_expert_swiglu_limit_list": [0, 0, 0, 0.02, 0.04],
        "use_qk_norm": true, "group_norm_size": 1, "linear_silu": true, "rotary_dim": 64,
    });
    for more in [ffn, keys] {
        text.as_object_mut().unwrap().extend(more.as_object().unwrap().clone());
    }
    serde_json::json!({
        "architectures": ["BailingMoeV3VLForConditionalGeneration"], "model_type": "bailing_moe_v3_vl", "tie_word_embeddings": false, "norm_query_embeds": false,
        "image_token_id": 60, "vision_config": {"model_type": "qwen3_moe_vit", "depth": 2}, "text_config": text,
    })
}

/// The Ling builder reads a config as the release's code does: the latent attentions the last layer of each group and every layer past the last whole
/// group; a gate a head or a channel; the queries one product or a low rank; the bounded decay or the softplus one; a limit a layer where its list has
/// one that is not 0; draft layers as heads past the lists; a quantization_config's FP8 matrices named with their scales. A key whose default the
/// release's code and transformers' give differently is asked for, and what the builder does not know is refused by name.
#[test]
fn ling_reads_its_config_as_the_releases_code_does() {
    use super::describe::Source;
    use super::kinds::{Body, Clamp, Decay, Ffn, GateFrom, Mixer};
    let ok = |keys: serde_json::Value| describe(&ling_tiny(keys)).unwrap_or_else(|e| panic!("{}", e.msg));
    let d = ok(serde_json::json!({}));
    let sorts: Vec<bool> = d.desc.blocks.iter().step_by(2).map(|b| matches!(b.body, Body::Mixer(Mixer::Mla(_)))).collect();
    assert_eq!(sorts, [false, true, false, true, true], "the last of each group of two, and the layer past the last whole group");
    assert!(matches!(&d.desc.blocks[0].body, Body::Mixer(Mixer::Linear(x)) if x.decay == Decay::Bounded(-5.0) && (x.k_heads, x.k_dim, x.conv) == (2, 8, 3)));
    assert!(matches!(&d.desc.blocks[1].body, Body::Ffn(Ffn::Mlp(m)) if m.clamp.is_none()));
    let moe = |d: &super::describe::Described, l: usize| match &d.desc.blocks[2 * l + 1].body {
        Body::Ffn(Ffn::Moe(m)) => m.clone(),
        b => panic!("layer {l}: {b:?}"),
    };
    assert_eq!(
        (moe(&d, 2).expert.clamp, moe(&d, 3).expert.clamp, moe(&d, 4).shared.unwrap().clamp),
        (None, Some(Clamp::Activated(0.05)), Some(Clamp::Activated(0.04)))
    );
    assert_eq!((moe(&d, 1).router.groups, moe(&d, 1).router.renorm, moe(&d, 1).router.scale), (Some((2, 1)), true, 2.5));
    assert_eq!(d.shapes["model.layers.1.attention.g_proj.weight"], [2, 32]);
    assert_eq!(d.shapes["model.layers.0.attention.g_proj.weight"], [16, 32]);
    assert_eq!(d.shapes["model.layers.1.attention.q_proj.weight"], [24, 32]);
    assert!(d.desc.mtp.is_empty() && d.scales.is_empty() && d.fp8_block.is_none() && d.desc.rope.as_ref().unwrap().mrope.is_none());
    assert!(
        matches!(&d.blocks[0].iter().find(|(r, _)| r == "conv").unwrap().1, Source::Rows(v) if v.len() == 3 && v[1] == "model.layers.0.attention.k_conv1d.weight")
    );
    // (the text model alone, with no wrapper: the same description)
    let alone = describe(&ling_tiny(serde_json::json!({}))["text_config"]).unwrap_or_else(|e| panic!("{}", e.msg));
    assert_eq!((format!("{:?}", alone.desc), alone.names()), (format!("{:?}", d.desc), d.names()));
    // (a gate a channel; the queries in a low rank; the softplus decay; one expert a token is not divided by itself; whole groups alone)
    let wide = ok(serde_json::json!({"gated_attention_proj_granularity_type": "element_wise"}));
    assert!(matches!(&wide.desc.blocks[2].body, Body::Mixer(Mixer::Mla(x)) if x.gate.unwrap().from == GateFrom::Own));
    assert_eq!(wide.shapes["model.layers.1.attention.g_proj.weight"], [12, 32]);
    let low = ok(serde_json::json!({"q_lora_rank": 10}));
    assert!(matches!(&low.desc.blocks[2].body, Body::Mixer(Mixer::Mla(x)) if x.q_lora == Some(10)));
    assert_eq!(
        (low.shapes["model.layers.1.attention.q_b_proj.weight"].clone(), low.shapes["model.layers.1.attention.q_a_layernorm.weight"].clone()),
        (vec![24, 10], vec![10])
    );
    assert!(!low.names().contains("model.layers.1.attention.q_proj.weight") && !d.names().iter().any(|n| n.contains("q_a_proj")));
    assert!(matches!(&ok(serde_json::json!({"kda_safe_gate": false})).desc.blocks[0].body, Body::Mixer(Mixer::Linear(x)) if x.decay == Decay::Softplus));
    assert!(!moe(&ok(serde_json::json!({"num_experts_per_tok": 1})), 1).router.renorm);
    assert_eq!(
        ok(serde_json::json!({"num_hidden_layers": 4})).desc.blocks.iter().step_by(2).filter(|b| matches!(b.body, Body::Mixer(Mixer::Mla(_)))).count(),
        2
    );
    // (lists shorter than the layers, and nulls: no limit there; a draft layer is a head past the lists, a latent attention and the experts)
    let short = ok(serde_json::json!({"expert_swiglu_limit_list": [null, 3], "share_expert_swiglu_limit_list": null, "num_nextn_predict_layers": 2}));
    assert_eq!((moe(&short, 1).expert.clamp, moe(&short, 2).expert.clamp, moe(&short, 4).shared.unwrap().clamp), (Some(Clamp::Activated(3.0)), None, None));
    assert_eq!((short.desc.mtp.len(), short.mtp.len(), short.mtp_layers), (2, 2, 2));
    assert!(
        matches!(&short.desc.mtp[1].blocks[0].body, Body::Mixer(Mixer::Mla(_)))
            && matches!(&short.desc.mtp[1].blocks[1].body, Body::Ffn(Ffn::Moe(m)) if m.expert.clamp.is_none())
    );
    assert!(short.names().contains("model.layers.6.final_layernorm.weight") && short.shapes["model.layers.5.eh_proj.weight"] == [32, 64]);
    // (a quantization_config names the FP8 products' scales and leaves out what it lists; the convolutions, the routers' biases and a head's own roles have none)
    let quant = serde_json::json!({"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128], "activation_scheme": "dynamic",
        "modules_to_not_convert": ["model.visual", "lm_head", "model.word_embeddings", "model.layers.0.attention.b_proj", "model.layers.1.attention.g_proj",
            "model.layers.1.mlp.gate"]});
    let mut q = ling_tiny(serde_json::json!({"num_nextn_predict_layers": 1}));
    q["quantization_config"] = quant.clone();
    let q = describe(&q).unwrap_or_else(|e| panic!("{}", e.msg));
    assert_eq!(q.fp8_block, Some(128));
    for held in [
        "model.layers.0.attention.f_proj.weight",
        "model.layers.0.attention.g_proj.weight",
        "model.layers.2.attention.b_proj.weight",
        "model.layers.1.attention.dense.weight",
        "model.layers.3.attention.g_proj.weight",
        "model.layers.5.attention.g_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.1.mlp.shared_experts.up_proj.weight",
        "model.layers.4.mlp.experts.7.down_proj.weight",
    ] {
        assert_eq!(q.scales.get(held), Some(&format!("{held}_scale_inv")), "{held}");
    }
    for left in [
        "model.layers.0.attention.b_proj.weight",
        "model.layers.1.attention.g_proj.weight",
        "model.layers.1.mlp.gate.weight",
        "model.layers.2.mlp.gate.weight",
        "model.layers.0.attention.q_conv1d.weight",
        "model.layers.0.attention.o_norm.weight",
        "model.layers.5.eh_proj.weight",
        "model.word_embeddings.weight",
        "lm_head.weight",
    ] {
        assert!(q.names().contains(left) && !q.scales.contains_key(left), "{left}");
    }
    let refused = |c: serde_json::Value| describe(&c).err().map(|e| e.msg).unwrap_or_default();
    for (keys, word) in [
        (serde_json::json!({"layer_group_size": null}), "no layer_group_size"),
        (serde_json::json!({"layer_group_size": 0}), "layer_group_size = 0"),
        (serde_json::json!({"first_k_dense_replace": null}), "no first_k_dense_replace"),
        (serde_json::json!({"routed_scaling_factor": null}), "no routed_scaling_factor"),
        (serde_json::json!({"rope_theta": null}), "no rope_theta"),
        (serde_json::json!({"kda_safe_gate": null}), "no kda_safe_gate"),
        (serde_json::json!({"kda_lower_bound": null}), "kda_lower_bound = None"),
        (serde_json::json!({"kda_lower_bound": 5.0}), "kda_lower_bound = Some(5.0)"),
        (serde_json::json!({"gated_attention_proj_granularity_type": null}), "no gated_attention_proj_granularity_type"),
        (serde_json::json!({"gated_attention_proj_granularity_type": "group_wise"}), "gated_attention_proj_granularity_type group_wise"),
        (serde_json::json!({"no_kda_lora": false}), "no_kda_lora is not true"),
        (serde_json::json!({"no_kda_lora": null}), "no_kda_lora is not true"),
        (serde_json::json!({"use_mla_nope": true}), "use_mla_nope"),
        (serde_json::json!({"rope_interleave": false}), "rope_interleave"),
        (serde_json::json!({"rope_scaling": {"type": "yarn", "factor": 4.0}}), "rope_scaling"),
        (serde_json::json!({"num_key_value_heads": 1}), "num_key_value_heads"),
        (serde_json::json!({"qk_rope_head_dim": 3}), "qk_rope_head_dim = 3"),
        (serde_json::json!({"mrope_section": [1, 1, 1]}), "mrope_section = [1,1,1]"),
        (serde_json::json!({"mrope_section": [0, 2, 0]}), "mrope_section"),
        (serde_json::json!({"router_dtype": "bf16"}), "router_dtype"),
        (serde_json::json!({"score_function": "softmax"}), "score_function"),
        (serde_json::json!({"moe_router_enable_expert_bias": false}), "moe_router_enable_expert_bias"),
        (serde_json::json!({"norm_topk_prob": false}), "norm_topk_prob"),
        (serde_json::json!({"n_group": 3}), "n_group = 3"),
        (serde_json::json!({"topk_group": 3}), "topk_group = 3"),
        (serde_json::json!({"num_shared_experts": 0}), "num_shared_experts = 0"),
        (serde_json::json!({"expert_swiglu_limit_list": [0, -1]}), "expert_swiglu_limit_list[1] = -1"),
        (serde_json::json!({"share_expert_swiglu_limit_list": 5}), "share_expert_swiglu_limit_list = 5"),
        (serde_json::json!({"use_bias": true}), "use_bias"),
        (serde_json::json!({"hidden_act": "gelu"}), "hidden_act"),
        (serde_json::json!({"use_nGPT": false, "value_norm": false}), "config keys use_nGPT, value_norm are not understood"),
        (serde_json::json!({"layer_types": ["full_attention"]}), "config key layer_types is not understood"),
    ] {
        let e = refused(ling_tiny(keys));
        assert!(e.contains(word), "{word}: {e}");
    }
    // (the wrapper's own keys; a quantization of another format)
    for (key, v, word) in [
        ("norm_query_embeds", serde_json::json!(true), "norm_query_embeds = true"),
        ("audio_config", serde_json::json!({}), "config key audio_config is not understood"),
        ("quantization_config", serde_json::json!({"quant_method": "fp8", "fmt": "e4m3"}), "weight_block_size"),
        ("quantization_config", serde_json::json!({"quant_method": "awq", "weight_block_size": [128, 128]}), "quantization awq"),
    ] {
        let mut c = ling_tiny(serde_json::json!({}));
        c[key] = v;
        let e = refused(c);
        assert!(e.contains(word), "{word}: {e}");
    }
    // (no delta attention, no key of it asked: groups of one are latent attentions throughout)
    let all = ok(serde_json::json!({"layer_group_size": 1, "kda_safe_gate": null, "no_kda_lora": null}));
    assert!(all.desc.blocks.iter().step_by(2).all(|b| matches!(b.body, Body::Mixer(Mixer::Mla(_)))));
}

/// The Ling layout on the CPU reference, a random model of each small config (every tensor of the shape its description gives). A prompt at once and
/// the same prompt a token at a time (the delta attentions' states and convolutions and the latent attentions' caches kept between the calls) are the
/// same bits, and a later token changes no earlier row. The gate a head is read: with every `g_proj` of the latent attentions at zero each head's gate
/// is a half, and the answer is, to the bit, the one of the model with no gate whose `dense` is halved. The limits are read, and on the activated gate:
/// a limit that bites is another answer than none, a limit far above the activations is none, to the bit, and the same limit held on the gate before
/// its activation (another family's form) is another answer. The decay's two forms differ.
#[test]
fn ling_on_the_reference_steps_as_it_prompts_gates_a_head_and_limits_the_activated_gate() {
    use super::kinds::{Body, Clamp, Ffn, Mixer};
    // (a fixed draw a tensor: its name's bytes seed it; a norm's weights near 1)
    fn draw(n: &str, len: usize) -> Vec<f32> {
        let mut s = n.bytes().fold(0x9E37_79B9_7F4A_7C15u64, |a, b| (a ^ b as u64).wrapping_mul(0x0100_0000_01B3));
        let norm = n.contains("norm");
        (0..len)
            .map(|_| {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                let u = ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5;
                if norm {
                    1.0 + 0.2 * u
                } else {
                    0.6 * u
                }
            })
            .collect()
    }
    // (`edit`: a change to the description before it is loaded; `gate`: a value for every latent attention's `g_proj`, None as drawn; `dense_x`: a
    // factor on every latent attention's output product)
    let load = |cfg: &serde_json::Value, edit: &dyn Fn(&mut super::kinds::ModelDesc), gate: Option<f32>, dense_x: f32| -> RefModel {
        let mut d = describe(cfg).unwrap_or_else(|e| panic!("{}", e.msg));
        edit(&mut d.desc);
        let latent: BTreeSet<String> = d
            .desc
            .blocks
            .iter()
            .zip(&d.blocks)
            .filter(|(b, _)| matches!(b.body, Body::Mixer(Mixer::Mla(_))))
            .flat_map(|(_, r)| r.iter().filter(|(role, _)| role == "g").flat_map(|(_, s)| s.names()))
            .collect();
        let src = |n: &str| {
            let shape = d.shapes.get(n)?.clone();
            let len: usize = shape.iter().product();
            let v = match gate {
                Some(g) if latent.contains(n) => vec![g; len],
                _ => draw(n, len),
            };
            let x = if n.ends_with(".attention.dense.weight") { dense_x } else { 1.0 };
            Some((shape, v.into_iter().map(|v| v * x).collect()))
        };
        RefModel::load(&d, &src).unwrap_or_else(|e| panic!("{}", e.msg))
    };
    let same = |_: &mut super::kinds::ModelDesc| {};
    let run = |cfg: &serde_json::Value, toks: &[u32]| load(cfg, &same, None, 1.0).forward(toks, 0).unwrap_or_else(|e| panic!("{}", e.msg));
    let a: [u32; 7] = [5, 9, 13, 17, 21, 25, 29];
    let base = ling_tiny(serde_json::json!({}));
    let rows = run(&base, &a);
    assert_eq!((rows.len(), rows[6].len()), (7, 61));
    assert!(rows.iter().flatten().all(|x| x.is_finite()) && rows[6].iter().any(|&x| x != rows[6][0]), "finite, and not one value");
    // (causal, and the whole context is seen: the states and the caches carry the first token to the last row)
    let mut later = a;
    later[6] = 44;
    assert_eq!(rows[..6], run(&base, &later)[..6], "a later token changes no earlier row");
    let mut first = a;
    first[0] = 44;
    assert_ne!(rows[6], run(&base, &first)[6], "the last row reads the first token");
    // (a token at a time, and a prompt then its next tokens: the same bits)
    let mut m = load(&base, &same, None, 1.0);
    let mut by_step = Vec::new();
    for (t, &tok) in a.iter().enumerate() {
        by_step.extend(m.forward(&[tok], t as u32).unwrap_or_else(|e| panic!("{}", e.msg)));
    }
    assert_eq!(by_step, rows, "a token at a time is the prompt");
    let mut m = load(&base, &same, None, 1.0);
    let mut parts = m.forward(&a[..4], 0).unwrap();
    parts.extend(m.forward(&a[4..], 4).unwrap());
    assert_eq!(parts, rows, "a prompt, then its next tokens");
    // the gate a head
    let ungated = |d: &mut super::kinds::ModelDesc| {
        for b in d.blocks.iter_mut() {
            if let Body::Mixer(Mixer::Mla(x)) = &mut b.body {
                x.gate = None;
            }
        }
    };
    let halved = load(&base, &same, Some(0.0), 1.0).forward(&a, 0).unwrap();
    assert_eq!(halved, load(&base, &ungated, None, 0.5).forward(&a, 0).unwrap(), "gates of a half are the output product halved");
    assert_ne!(rows[6], halved[6], "the gate as drawn is not a half");
    assert_ne!(rows[6], run(&ling_tiny(serde_json::json!({"gated_attention_proj_granularity_type": "element_wise"})), &a)[6]);
    // the limits, on the activated gate
    let none = ling_tiny(serde_json::json!({"expert_swiglu_limit_list": null, "share_expert_swiglu_limit_list": null}));
    let free = run(&none, &a);
    assert_ne!(rows[6], free[6], "a limit that bites is another answer");
    assert_eq!(
        free,
        run(&ling_tiny(serde_json::json!({"expert_swiglu_limit_list": [1e9, 1e9, 1e9, 1e9, 1e9], "share_expert_swiglu_limit_list": [0, 1e9]})), &a)
    );
    assert_ne!(free[6], run(&ling_tiny(serde_json::json!({"share_expert_swiglu_limit_list": null})), &a)[6], "the routed experts' limit alone bites");
    assert_ne!(free[6], run(&ling_tiny(serde_json::json!({"expert_swiglu_limit_list": null})), &a)[6], "the shared MLP's limit alone bites");
    let before = |d: &mut super::kinds::ModelDesc| {
        for b in d.blocks.iter_mut() {
            if let Body::Ffn(Ffn::Moe(m)) = &mut b.body {
                m.expert.clamp = m.expert.clamp.map(|c| Clamp::Gate(c.limit()));
                m.shared = m.shared.map(|s| super::kinds::Mlp { clamp: s.clamp.map(|c| Clamp::Gate(c.limit())), ..s });
            }
        }
    };
    assert_ne!(rows[6], load(&base, &before, None, 1.0).forward(&a, 0).unwrap()[6], "the gate held before its activation is another function");
    // the decay's two forms
    assert_ne!(rows[6], run(&ling_tiny(serde_json::json!({"kda_safe_gate": false})), &a)[6]);
}

/// ERNIE 5.5's config as released (its text model's keys: the release is text in, text out), with `keys` over them.
fn ernie_config(keys: serde_json::Value) -> serde_json::Value {
    let mut ratios = vec![128; 43];
    for l in [7, 14, 21, 28, 35, 42] {
        ratios[l] = -2;
    }
    let mut c = serde_json::Map::new();
    for part in [
        serde_json::json!({
            "architectures": ["Ernie5_5_MoeForCausalLM"], "model_type": "ernie5_5_moe", "torch_dtype": "bfloat16", "transformers_version": "4.46.1",
            "use_cache": true, "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0, "max_position_embeddings": 1048576,
            "hidden_size": 4096, "num_hidden_layers": 43, "vocab_size": 201216, "rms_norm_eps": 1e-6, "tie_word_embeddings": false,
            "hidden_act": "situ", "activation_situ_beta": 4.0, "activation_situ_linear_beta": 25.0, "gated_linear_unit": true, "attention_bias": false,
            "use_bias": false, "compress_ratios": ratios, "sliding_window": 128, "compress_rope_theta": 160000.0, "csa_dense_mode": true,
            "experimental_attention_variant": "dsv4_hybrid", "use_mla": false, "num_attention_heads": 64, "num_key_value_heads": 1, "head_dim": 512,
            "qk_nope_head_dim": 448, "qk_rope_head_dim": 64, "qk_pos_emb_head_dim": 64, "v_head_dim": 512, "kv_lora_rank": 512, "q_lora_rank": 1024,
        }),
        serde_json::json!({
            "o_lora_rank": 1024, "o_groups": 8, "use_qk_norm": true, "qk_norm_type": "per_head", "use_attn_merger": true, "use_vha_attention": true,
            "use_gated_attn": true, "gated_attn_act": "sigmoid", "gated_attn_use_q_lora": true, "add_full_attention_sink_bias": true,
            "hybrid_mla_attention": "mqa_dsa", "hybrid_mla_num_attention_heads": 64, "hybrid_mla_num_key_value_heads": 64, "hybrid_mla_q_lora_rank": 1024,
            "hybrid_mla_kv_lora_rank": 512, "hybrid_mla_qk_nope_head_dim": 192, "hybrid_mla_qk_rope_head_dim": 64, "hybrid_mla_v_head_dim": 256,
            "mqa_split_kv_b_proj": true, "index_n_heads": 64, "index_head_dim": 128, "index_topk": 2048, "use_fast_hadamard": true,
            "rope_theta": 8000000, "rotary_base": 8000000, "rotary_percent": 1.0, "rope_type": "rope", "rope_interleave": true, "hca_rope_type": "yarn",
            "rope_scaling": {"beta_fast": 32, "beta_slow": 1, "factor": 16, "original_max_position_embeddings": 65536, "type": "yarn"},
        }),
        serde_json::json!({
            "enable_hyper_connections": true, "hc_mult": 4, "hc_eps": 1e-6, "hc_sinkhorn_iters": 20,
            "first_k_dense_replace": 1, "intermediate_size": 14336, "moe_intermediate_size": 2048, "moe_latent_size": 2048, "latent_moe_use_norm": true,
            "moe_layer_freq": 1, "moe_group_experts": false, "moe_shared_expert_gate": false, "moe_split_feature_routing": true,
            "moe_token_dispatcher_type": "deepep", "n_routed_experts": 512, "n_shared_experts": 1, "num_experts_per_tok": 10, "n_group": 1, "topk_group": 1,
            "norm_topk_prob": true, "scoring_func": "sigmoid", "topk_method": "noaux_tc", "routed_scaling_factor": 3.0,
            "routed_scaling_factor_learnable": true, "multimax": "lm_head", "num_nextn_predict_layers": 0, "use_dense_mtp": false,
            "mtp_shared_last_layer": true, "dspark_block_size": 7, "dspark_markov_rank": 256, "dspark_noise_token_id": 1024,
            "dspark_target_layer_ids": [40, 41, 42],
        }),
        keys,
    ] {
        c.extend(part.as_object().unwrap().clone());
    }
    serde_json::Value::Object(c)
}

/// ERNIE 5.5's released config describes: every key of it read, 37 layers of the compressed attention and 6 of the latent one (each with its gate from
/// the query latent and its heads' mix; the latent ones with sinks, their window beside the indexer's picks and the indexer's own pairing), a dense
/// feed-forward on the first layer and experts in a latent on the rest (two routers, a scale an expert), four residual streams whose norm has its
/// epsilon outside the root, the corrected logits, a drafter of three stages whose targets are read after their layers. The counts are the config's:
/// a model of other layers says its own, and one with no drafter's keys has none. What it still refuses is the executor's to say: each part the
/// executor has no operation for, by its name, and no loader.
#[test]
fn ernie_5_5s_released_config_is_read_and_described_part_by_part() {
    use super::describe::keys_named;
    use super::kinds::{Act, Body, Collapse, Ffn, GateFrom, IndexForm, Mixer, OutGate, Residual, RopeScaling, RopeStyle, Rounding, RouterScore};
    let c = ernie_config(serde_json::json!({}));
    assert_eq!(c.as_object().unwrap().len(), 97, "every key of the release's config.json");
    let d = describe(&c).unwrap_or_else(|e| panic!("{}", e.msg));
    assert_eq!((d.desc.blocks.len(), d.desc.hidden, d.desc.vocab, d.convention, d.mtp_layers), (86, 4096, 201216, "ERNIE", 0));
    let gate = Some(OutGate { act: Act::Sigmoid, from: GateFrom::Latent, norm: None });
    let (mut comp, mut latent, mut dense, mut sparse) = (0, 0, 0, 0);
    for (i, b) in d.desc.blocks.iter().enumerate() {
        let l = i / 2;
        let Residual::Streams(s) = b.residual else { panic!("block {i}") };
        assert_eq!((s.streams, s.sinkhorn_iters, s.eps, s.norm_eps, s.rounding, s.lagged, s.eps_outside), (4, 20, 1e-6, 1e-6, Rounding::Once, false, true));
        assert!(b.norm_out.is_none() && b.norm_in == super::kinds::Norm::rms(1e-6), "block {i}");
        match &b.body {
            Body::Mixer(Mixer::Compressed(x)) => {
                assert!(l % 7 != 0 || l == 0, "layer {l}");
                assert_eq!((x.heads, x.head_dim, x.q_lora, x.o_groups, x.o_lora, x.window, x.ratio, x.overlap), (64, 512, 1024, 8, 1024, 128, 128, false));
                assert!(x.indexer.is_none() && x.ape && x.q_head_norm && x.share.is_none() && x.gate == gate && x.head_mix == 16, "layer {l}");
                assert_eq!((x.rope.theta, x.rope.dim, x.rope.style), (160_000.0, 64, RopeStyle::Interleaved), "layer {l}");
                assert!(
                    matches!(x.rope.scaling, RopeScaling::Yarn { factor, original_max: 65536, attention_factor: Some(a), .. } if factor == 16.0 && a == 1.0)
                );
                comp += 1;
            }
            Body::Mixer(Mixer::Mla(x)) => {
                assert!(l % 7 == 0 && l > 0, "layer {l}");
                assert_eq!((x.heads, x.q_lora, x.kv_lora, x.nope, x.rope_dim, x.v_dim), (64, Some(1024), 512, 192, 64, 256), "layer {l}");
                assert_eq!((x.gate, x.scale, x.head_mix, x.sinks, x.window), (gate, 256f32.powf(-0.5), 16, true, Some(128)), "layer {l}");
                let r = x.rope.as_ref().unwrap();
                assert_eq!((r.theta, r.dim, r.style, &r.scaling), (8e6, 64, RopeStyle::Interleaved, &RopeScaling::None), "layer {l}");
                let ix = x.indexer.as_ref().unwrap();
                assert_eq!((ix.heads, ix.head_dim, ix.top_k, ix.block, ix.shared, ix.rope), (64, 128, 2048, 1, false, 64), "layer {l}");
                assert_eq!((ix.form, ix.rope_style), (IndexForm::LAYER_KEYS, Some(RopeStyle::Half)), "layer {l}");
                latent += 1;
            }
            Body::Ffn(Ffn::Mlp(m)) => {
                assert!(l == 0 && m.gated && !m.bias && m.clamp.is_none() && m.act == Act::Situ { beta: 4.0, linear: 25.0 }, "layer {l}");
                dense += 1;
            }
            Body::Ffn(Ffn::Moe(m)) => {
                let r = &m.router;
                assert_eq!((m.experts, r.top_k, r.routers, r.expert_scale, r.bias, r.renorm, r.scale), (512, 10, 2, true, true, true, 1.0), "layer {l}");
                assert!(r.score == RouterScore::Sigmoid && !r.hash && r.groups.is_none() && !m.shared_gate && m.shared == Some(m.expert), "layer {l}");
                assert_eq!(m.expert.act, Act::Situ { beta: 4.0, linear: 25.0 }, "layer {l}");
                let lt = m.latent.as_ref().unwrap();
                assert!(lt.dim == 2048 && lt.norm_in.is_none() && lt.expert_norm.is_none() && lt.norm_out == Some(super::kinds::Norm::rms(1e-6)), "layer {l}");
                sparse += 1;
            }
            _ => panic!("block {i}"),
        }
    }
    assert_eq!((comp, latent, dense, sparse), (37, 6, 1, 42));
    let h = d.desc.head;
    assert!(!h.tied && h.correction && h.rounding == Rounding::Twice && h.softcap.is_none() && h.scale == 1.0);
    assert_eq!(h.collapse, Collapse::Weighted { eps: 1e-6, norm_eps: 1e-6 });
    assert_eq!(d.desc.embed.streams, 4);
    // (the drafter: three stages of the window alone with the compressing layers' rotary and the experts, its targets read after their layers)
    assert_eq!((d.desc.mtp.len(), d.mtp.len()), (1, 1));
    let head = &d.desc.mtp[0];
    let ds = head.dspark.as_ref().unwrap();
    assert_eq!(
        (ds.block, ds.noise, ds.targets.clone(), ds.markov_rank, ds.confidence, ds.after, head.blocks.len()),
        (7, 1024, vec![40, 41, 42], 256, true, true, 6)
    );
    for (i, b) in head.blocks.iter().enumerate() {
        match &b.body {
            Body::Mixer(Mixer::Compressed(x)) => {
                assert!(i % 2 == 0 && x.ratio == 0 && x.window == 128 && x.gate == gate && x.head_mix == 16 && x.rope.theta == 160_000.0, "draft block {i}")
            }
            Body::Ffn(Ffn::Moe(m)) => assert!(i % 2 == 1 && m.router.routers == 2 && m.router.expert_scale && m.experts == 512, "draft block {i}"),
            b => panic!("draft block {i}: {b:?}"),
        }
    }
    assert_eq!((d.embed.as_str(), d.final_norm.as_str(), d.head.as_deref()), ("model.embed_tokens.weight", "model.norm.weight", Some("lm_head.weight")));
    assert!(d.scales.is_empty() && d.fp8_block.is_none() && d.unread.is_empty());
    // (the counts are the config's: a model of other layers says its own)
    let small = describe(&ernie_config(serde_json::json!({
        "num_hidden_layers": 3, "compress_ratios": [128, -2, -2], "first_k_dense_replace": 2, "dspark_target_layer_ids": [2], "n_routed_experts": 16,
    })))
    .unwrap_or_else(|e| panic!("{}", e.msg));
    let sorts: Vec<&str> = small
        .desc
        .blocks
        .iter()
        .map(|b| match &b.body {
            Body::Mixer(Mixer::Compressed(_)) => "compressed",
            Body::Mixer(Mixer::Mla(_)) => "latent",
            Body::Ffn(Ffn::Mlp(_)) => "dense",
            Body::Ffn(Ffn::Moe(m)) if m.experts == 16 => "experts",
            b => panic!("{b:?}"),
        })
        .collect();
    assert_eq!(sorts, ["compressed", "dense", "latent", "dense", "latent", "experts"]);
    assert_eq!(small.desc.mtp[0].blocks.len(), 2, "a drafter of one stage");
    // (no drafter: the keys of one are not asked for)
    let mut bare = ernie_config(serde_json::json!({}));
    for key in ["dspark_block_size", "dspark_markov_rank", "dspark_noise_token_id", "dspark_target_layer_ids"] {
        bare.as_object_mut().unwrap().remove(key);
    }
    let bare = describe(&bare).unwrap_or_else(|e| panic!("{}", e.msg));
    assert!(bare.desc.mtp.is_empty() && bare.mtp.is_empty() && !bare.names().iter().any(|n| n.starts_with("model.mtp.")));
    // (what the builder itself still refuses: a form the reader does not compute, in words)
    for (keys, want) in [
        (serde_json::json!({"index_topk": 128}), "index_topk = 128 is not above sliding_window (128)"),
        (serde_json::json!({"num_attention_heads": 62, "o_groups": 2}), "62 and 64 heads, where the heads' mix has a rank of a quarter of them"),
    ] {
        let e = describe(&ernie_config(keys.clone())).expect_err("refused").msg;
        assert!(e.starts_with("ernie5_5_moe: ") && e.contains(want) && keys_named(&e).is_empty(), "{keys}: {e}");
    }
    // (described, not run: the executor names the first part it has no operation for, and has no loader)
    let e = crate::dynamic::split::loader(&d).err().map(|e| e.msg).unwrap_or_default();
    assert_eq!(e, crate::dynamic::split::STORED_FORM_NOT_RUN);
    assert!(e.ends_with("which the executor does not run yet"));
    let Err(e) = crate::dynamic::split::lowered_described(&d, 43, 4096) else { panic!("lowered") };
    assert_eq!(e.msg, crate::dynamic::split::STORED_FORM_NOT_RUN);
}

/// ERNIE-5.5-Flash-Preview's released header: the description reads every one of its 70,826 tensors, the drafter's too, each at the shape the
/// checkpoint stores it in and in the type it is released in; the matrices stored transposed, the scalars stored apart and the two routers are named
/// by their source's form.
#[test]
fn ernie_5_5_flash_preview_describes_its_whole_checkpoint() {
    use super::describe::Source;
    let h = fixture("headers/ERNIE-5.5-Flash-Preview-BF16.json");
    let cfg: serde_json::Value = serde_json::from_str(h["config"].as_str().unwrap()).unwrap();
    assert_eq!(cfg, ernie_config(serde_json::json!({})), "the release's config is the one the tests of the builder read");
    let d = describe(&cfg).unwrap_or_else(|e| panic!("{}", e.msg));
    let (have, tensors) = (header_names(&h), header_tensors(&h));
    assert_eq!((have.len(), d.names().len()), (70_826, 70_826));
    d.check(&have, true).unwrap_or_else(|e| panic!("{}", e.msg));
    let fp32 = |n: &str| {
        let ends = ["e_score_correction_bias", "attn_sink", "compressor.ape"];
        n.contains("hyper_connection.") || n.contains("mhc_contract.") || n.contains("block_sparse_moe.gate.") || ends.iter().any(|e| n.ends_with(e))
    };
    for n in d.names() {
        let (dtype, shape, _) = &tensors[&n];
        assert_eq!(d.shapes.get(&n), Some(shape), "{n}");
        assert_eq!(dtype, if fp32(&n) { "F32" } else { "BF16" }, "{n}");
    }
    let role = |b: usize, r: &str| d.blocks[b].iter().find(|(x, _)| x == r).unwrap_or_else(|| panic!("block {b}: no {r}")).1.clone();
    assert_eq!(role(0, "hc_fn"), Source::Transposed("model.layers.0.self_attention_hyper_connection.mapping_proj.weight".into()));
    assert_eq!(d.shapes["model.layers.0.self_attention_hyper_connection.mapping_proj.weight"], [16384, 24]);
    let alphas = ["alpha_pre", "alpha_post", "alpha_res"].map(|a| format!("model.layers.3.mlp_hyper_connection.{a}")).to_vec();
    assert_eq!(role(7, "hc_scale"), Source::Rows(alphas));
    let routers = ["weight", "weight_1"].map(|w| format!("model.layers.3.block_sparse_moe.gate.{w}")).to_vec();
    assert_eq!(role(7, "router"), Source::Rows(routers));
    assert_eq!(role(7, "expert_scale"), Source::Tensor("model.layers.3.block_sparse_moe.gate.routed_scaling_factor_param".into()));
    assert_eq!(role(14, "sinks"), Source::Tensor("model.layers.7.self_attn.core_attention.softmax_offset".into()));
    assert_eq!(role(12, "sinks"), Source::Tensor("model.layers.6.self_attn.core_attention.attn_sink".into()));
    assert_eq!(role(12, "o_a"), Source::Tensor("model.layers.6.self_attn.linear_o_group_proj".into()));
    assert_eq!(
        (d.shapes["model.layers.6.self_attn.gate_proj.weight"].clone(), d.shapes["model.layers.7.self_attn.gate_proj.weight"].clone()),
        (vec![8192, 1024], vec![16384, 1024])
    );
    let extra = |r: &str| d.extra.iter().find(|(x, _)| x == r).unwrap().1.clone();
    assert_eq!(extra("out_hc_fn"), Source::Transposed("model.mhc_contract.hc_head_fn".into()));
    assert_eq!(extra("logit_ranges"), Source::Tensor("model.lm_head.multimax_ranges".into()));
    let (stages, own) = &d.mtp[0];
    assert_eq!(stages.len(), 6);
    let own_of = |r: &str| own.iter().find(|(x, _)| x == r).unwrap().1.clone();
    assert_eq!(own_of("markov_embed"), Source::Transposed("model.mtp.markov_head.markov_w1.weight".into()));
    assert_eq!(d.shapes["model.mtp.markov_head.markov_w1.weight"], [256, 201216]);
    assert_eq!(own_of("main_proj"), Source::Tensor("model.mtp.feature_projection.weight".into()));
    assert_eq!(d.shapes["model.mtp.feature_projection.weight"], [4096, 12288]);
    assert!(stages[4].iter().any(|(r, s)| r == "kv" && *s == Source::Tensor("model.mtp.2.self_attn.linear_kv_proj.weight".into())));
    assert!(!stages[4].iter().any(|(r, _)| r.starts_with("comp_")), "a stage's attention is the window's: no compressor");
}

/// The ERNIE builder holds each key to what its reader computes: a setting of another form is refused by its key before the parts are, a key nobody
/// knows by its name, and a list of ratios that is not the layers', a ratio that is no sort of layer, or a width that is not the form's in words.
#[test]
fn ernie_5_5_holds_every_key_to_the_form_its_reader_computes() {
    use super::describe::keys_named;
    let refusal = |keys: serde_json::Value| describe(&ernie_config(keys)).expect_err("refused").msg;
    for (keys, want) in [
        (serde_json::json!({"hidden_act": "silu"}), "hidden_act = \"silu\" is not supported (only \"situ\")"),
        (serde_json::json!({"multimax": "none"}), "multimax = \"none\" is not supported"),
        (serde_json::json!({"moe_split_feature_routing": false}), "moe_split_feature_routing = false is not supported"),
        (serde_json::json!({"routed_scaling_factor_learnable": false}), "routed_scaling_factor_learnable = false is not supported"),
        (serde_json::json!({"latent_moe_use_norm": false}), "latent_moe_use_norm = false is not supported"),
        (serde_json::json!({"n_group": 2}), "n_group = 2 is not supported"),
        (serde_json::json!({"use_gated_attn": false}), "use_gated_attn = false is not supported"),
        (serde_json::json!({"gated_attn_act": "silu"}), "gated_attn_act = \"silu\" is not supported"),
        (serde_json::json!({"use_vha_attention": false}), "use_vha_attention = false is not supported"),
        (serde_json::json!({"add_full_attention_sink_bias": false}), "add_full_attention_sink_bias = false is not supported"),
        (serde_json::json!({"enable_hyper_connections": false}), "enable_hyper_connections = false is not supported"),
        (serde_json::json!({"scoring_func": "softmax"}), "scoring_func = \"softmax\" is not supported"),
        (serde_json::json!({"rope_interleave": false}), "rope_interleave = false is not supported"),
        (serde_json::json!({"use_mla": true}), "use_mla = true is not supported"),
        (serde_json::json!({"csa_dense_mode": false}), "csa_dense_mode = false is not supported"),
        (serde_json::json!({"num_key_value_heads": 8}), "num_key_value_heads = 8 is not supported (only 1)"),
        (serde_json::json!({"tie_word_embeddings": true}), "tie_word_embeddings = true is not supported"),
        (serde_json::json!({"qk_nope_head_dim": 400}), "qk_nope_head_dim = 400 is not supported (only 448, head_dim less qk_rope_head_dim"),
        (serde_json::json!({"v_head_dim": 256}), "v_head_dim = 256 is not supported (only 512, head_dim: the keys are the values)"),
        (serde_json::json!({"kv_lora_rank": 256}), "kv_lora_rank = 256 is not supported (only 512"),
        (serde_json::json!({"rotary_base": 10000}), "rotary_base is not rope_theta (8000000)"),
        (serde_json::json!({"rotary_percent": 0.5}), "rotary_percent is not supported (only 1.0)"),
        (serde_json::json!({"rope_scaling": {"type": "linear", "factor": 2.0}}), "rope_scaling of type linear is not supported (only yarn)"),
        (serde_json::json!({"hybrid_mla_num_key_value_heads": 1}), "hybrid_mla_num_key_value_heads is not supported (only hybrid_mla_num_attention_heads, 64)"),
        (serde_json::json!({"compress_ratios": [128, -2]}), "2 compress_ratios for 43 layers"),
        (serde_json::json!({"num_hidden_layers": 2, "compress_ratios": [128, 4], "dspark_target_layer_ids": [1]}), "layer 1 has compress ratio 4 (only 128"),
        (serde_json::json!({"num_nextn_predict_layers": 1}), "num_nextn_predict_layers = 1 is not supported (only 0"),
        (serde_json::json!({"num_nextn_predict_layers": null}), "num_nextn_predict_layers = 1 is not supported (only 0"),
        (serde_json::json!({"num_experts_per_tok": 600}), "600 of 512 experts a token"),
        (serde_json::json!({"activation_situ_beta": 0.0}), "activation_situ_beta = 0"),
        (serde_json::json!({"dspark_target_layer_ids": [40, 41, 43]}), "dspark_target_layer_ids names layer 43 of 43"),
        (serde_json::json!({"dspark_markov_rank": null}), "a DSpark drafter without its dspark_block_size, dspark_markov_rank"),
        (serde_json::json!({"hc_sinkhorn_iters": null}), "no hc_sinkhorn_iters"),
        (serde_json::json!({"index_topk": null}), "no index_topk"),
    ] {
        let e = refusal(keys.clone());
        assert!(e.starts_with("ernie5_5_moe: ") && e.contains(want), "{keys}: {e}");
        assert!(!e.contains("its config is read"), "{keys}: refused at the key, before the parts: {e}");
    }
    // (a key no builder knows is named, with every other one, before the parts are)
    let e = refusal(serde_json::json!({"attn_merger_rank": 16, "vha_rank": 16}));
    assert_eq!(keys_named(&e), ["attn_merger_rank", "vha_rank"], "{e}");
}

/// A small config of the ERNIE layout with `keys` over it: three layers of hidden 32 on two residual streams, the first and the last of the compressed
/// attention (4 heads of 8, a window of 4, compressing by 128) and the second a latent attention (4 heads, an indexer keeping 6 tokens before the
/// window); a dense feed-forward on the first layer and 8 experts in a latent of 12 on the others; a drafter of two stages.
fn ernie_tiny(keys: serde_json::Value) -> serde_json::Value {
    let mut c = ernie_config(serde_json::json!({
        "hidden_size": 32, "num_hidden_layers": 3, "vocab_size": 61, "compress_ratios": [128, -2, 128], "sliding_window": 4,
        "num_attention_heads": 4, "head_dim": 8, "qk_nope_head_dim": 4, "qk_rope_head_dim": 4, "qk_pos_emb_head_dim": 4, "v_head_dim": 8, "kv_lora_rank": 8,
        "q_lora_rank": 12, "o_lora_rank": 6, "o_groups": 2,
        "hybrid_mla_num_attention_heads": 4, "hybrid_mla_num_key_value_heads": 4, "hybrid_mla_q_lora_rank": 10, "hybrid_mla_kv_lora_rank": 12,
        "hybrid_mla_qk_nope_head_dim": 4, "hybrid_mla_qk_rope_head_dim": 4, "hybrid_mla_v_head_dim": 6,
        "index_n_heads": 2, "index_head_dim": 8, "index_topk": 6,
        "hc_mult": 2, "first_k_dense_replace": 1, "intermediate_size": 48, "moe_intermediate_size": 16, "moe_latent_size": 12,
        "n_routed_experts": 8, "num_experts_per_tok": 2,
        "dspark_block_size": 3, "dspark_markov_rank": 4, "dspark_noise_token_id": 60, "dspark_target_layer_ids": [1, 2],
    }));
    c.as_object_mut().unwrap().extend(keys.as_object().unwrap().clone());
    c
}

/// The ERNIE layout on the CPU reference: a random model of the small config, every tensor of the shape its description gives and in the form the
/// checkpoint stores it (the mixing matrices transposed, the three scales apart, the two routers apart). Over a prompt of 135 tokens, past the first
/// closed window of 128 and past the latent layer's window and picks: a prompt at once, a token at a time and a prompt then its next tokens are the same
/// bits, and a later token changes no earlier row; the compressed layers each hold their one entry and the latent layer's last query reads its six picks
/// and its window of four. Each part is read, to the bit where a value of it is the layer without it: gates of zero are the output products halved; a
/// mix whose V is zero, sinks far below the scores, a correction whose factors are zero and scales of one are the description without the part; and the
/// window, the second router and the place of the norm's epsilon each change the answer. The drafter loads and drafts its block, its base logits bf16.
#[test]
fn ernie_on_the_reference_runs_both_sorts_of_layer_and_reads_each_part() {
    use super::describe::Described;
    use super::kinds::{Body, Ffn, Mixer, Residual};
    use super::reference::{bf16, mtp::RefDSpark, SeqState};
    // (a fixed draw a tensor: its name's bytes seed it; a norm's weights near 1)
    fn draw(n: &str, len: usize) -> Vec<f32> {
        let mut s = n.bytes().fold(0x9E37_79B9_7F4A_7C15u64, |a, b| (a ^ b as u64).wrapping_mul(0x0100_0000_01B3));
        let norm = n.contains("norm");
        (0..len)
            .map(|_| {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                let u = ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5;
                if norm {
                    1.0 + 0.2 * u
                } else {
                    0.6 * u
                }
            })
            .collect()
    }
    // (`set`: for the tensors whose names end so, a factor on the draw and a value added: (0, c) is every value c)
    type Set<'a> = &'a [(&'a str, f32, f32)];
    fn tensor(d: &Described, set: Set, n: &str) -> Option<(Vec<usize>, Vec<f32>)> {
        let shape = d.shapes.get(n)?.clone();
        let (a, b) = set.iter().find(|(end, _, _)| n.ends_with(end)).map_or((1.0, 0.0), |x| (x.1, x.2));
        Some((shape.clone(), draw(n, shape.iter().product()).into_iter().map(|v| a * v + b).collect()))
    }
    let cfg = ernie_tiny(serde_json::json!({}));
    let described = |edit: &dyn Fn(&mut Described)| {
        let mut d = describe(&cfg).unwrap_or_else(|e| panic!("{}", e.msg));
        edit(&mut d);
        d
    };
    let load = |edit: &dyn Fn(&mut Described), set: Set| -> RefModel {
        let d = described(edit);
        RefModel::load(&d, &|n: &str| tensor(&d, set, n)).unwrap_or_else(|e| panic!("{}", e.msg))
    };
    let same = |_: &mut Described| {};
    let toks: Vec<u32> = (0..135u32).map(|t| (t * 7 + 3) % 59).collect();
    let run = |edit: &dyn Fn(&mut Described), set: Set| load(edit, set).forward(&toks, 0).unwrap_or_else(|e| panic!("{}", e.msg));
    let mut m = load(&same, &[]);
    let rows = m.forward(&toks, 0).unwrap_or_else(|e| panic!("{}", e.msg));
    assert!(rows.len() == 135 && rows.iter().flatten().all(|x| x.is_finite() && *x == bf16(*x)), "bf16 logits");
    // (both sorts ran past their first choices: an entry of the first 128 tokens in each compressed layer, six picks and the window in the latent one)
    for b in [0, 4] {
        let Some(SeqState::Comp(c)) = m.layers[b].seqs.get(&0) else { panic!("block {b}") };
        assert_eq!((c.comp.entries.len(), c.win.len()), (1, 135), "block {b}");
    }
    let Some(SeqState::Mla(c)) = m.layers[2].seqs.get(&0) else { panic!("block 2") };
    assert_eq!((c.picks.len(), &c.picks[6..]), (10, &[131usize, 132, 133, 134][..]));
    assert!(c.picks[..6].iter().all(|&e| e < 131) && c.ikey.len() == 135);
    // (the stored forms: a mixing matrix is the checkpoint's turned, the scales the three scalars in their order, the routers one above the other)
    let d = described(&same);
    let stored = draw("model.layers.0.self_attention_hyper_connection.mapping_proj.weight", 64 * 8);
    let fn0 = m.layers[0].p.vec("hc_fn").unwrap();
    assert!((0..8).all(|i| (0..64).all(|j| fn0[i * 64 + j] == stored[j * 8 + i])));
    let alphas: Vec<f32> = ["alpha_pre", "alpha_post", "alpha_res"].iter().map(|a| draw(&format!("model.layers.1.mlp_hyper_connection.{a}"), 1)[0]).collect();
    assert_eq!(m.layers[3].p.vec("hc_scale").unwrap(), &alphas[..]);
    let router = m.layers[3].p.vec("router").unwrap();
    assert_eq!((router.len(), &router[8 * 32..]), (2 * 8 * 32, &draw("model.layers.1.block_sparse_moe.gate.weight_1", 8 * 32)[..]));
    // (a later token changes no earlier row, and the last row reads the first token)
    let mut other = toks.clone();
    other[134] = 58;
    assert_eq!(load(&same, &[]).forward(&other, 0).unwrap()[..134], rows[..134]);
    other = toks.clone();
    other[0] = 44;
    assert_ne!(load(&same, &[]).forward(&other, 0).unwrap()[134], rows[134], "the last row reads the first token");
    // (a token at a time, and a prompt then its next tokens: the same bits)
    let mut m = load(&same, &[]);
    let by_step: Vec<Vec<f32>> = toks.iter().enumerate().flat_map(|(t, &tok)| m.forward(&[tok], t as u32).unwrap_or_else(|e| panic!("{}", e.msg))).collect();
    assert_eq!(by_step, rows, "a token at a time is the prompt");
    let mut m = load(&same, &[]);
    let mut parts = m.forward(&toks[..100], 0).unwrap();
    parts.extend(m.forward(&toks[100..], 100).unwrap());
    assert_eq!(parts, rows, "a prompt, then its next tokens");
    // the gates: of zero they are a half, the output products halved
    let ungated = |d: &mut Described| {
        for b in d.desc.blocks.iter_mut() {
            match &mut b.body {
                Body::Mixer(Mixer::Mla(x)) => x.gate = None,
                Body::Mixer(Mixer::Compressed(x)) => x.gate = None,
                _ => {}
            }
        }
    };
    let halved = run(&same, &[("self_attn.gate_proj.weight", 0.0, 0.0)]);
    assert_eq!(halved, run(&ungated, &[("self_attn.o_proj.weight", 0.5, 0.0)]), "gates of a half are the output products halved");
    assert_ne!(halved[134], rows[134], "the gates as drawn are not a half");
    // the heads' mix
    let unmixed = |d: &mut Described| {
        for b in d.desc.blocks.iter_mut() {
            match &mut b.body {
                Body::Mixer(Mixer::Mla(x)) => x.head_mix = 0,
                Body::Mixer(Mixer::Compressed(x)) => x.head_mix = 0,
                _ => {}
            }
        }
    };
    let plain = run(&unmixed, &[]);
    assert_eq!(run(&same, &[("self_attn.vha_postmix_V", 0.0, 0.0)]), plain, "a mix whose V is zero is no mix");
    assert_ne!(plain[134], rows[134]);
    // the latent attention's sinks, and its window
    let latent = |f: &'static dyn Fn(&mut super::kinds::Mla)| {
        move |d: &mut Described| {
            for b in d.desc.blocks.iter_mut() {
                if let Body::Mixer(Mixer::Mla(x)) = &mut b.body {
                    f(x);
                }
            }
        }
    };
    let sinkless = run(&latent(&|x| x.sinks = false), &[]);
    assert_eq!(run(&same, &[("core_attention.softmax_offset", 0.0, -1e30)]), sinkless, "sinks far below the scores are no sinks");
    assert_ne!(sinkless[134], rows[134]);
    assert_ne!(run(&latent(&|x| x.window = None), &[])[134], rows[134], "the window is read beside the picks");
    // the logits' correction
    let uncorrected = run(&|d: &mut Described| d.desc.head.correction = false, &[]);
    assert_eq!(run(&same, &[("lm_head.multimax_ts", 0.0, 0.0)]), uncorrected, "a correction whose factors are zero is none");
    assert_ne!(uncorrected[134], rows[134]);
    // the routers: scales of one are no scale; the second router is read
    let experts = |f: &'static dyn Fn(&mut super::kinds::Router)| {
        move |d: &mut Described| {
            for b in d.desc.blocks.iter_mut() {
                if let Body::Ffn(Ffn::Moe(x)) = &mut b.body {
                    f(&mut x.router);
                }
            }
        }
    };
    let unscaled = run(&experts(&|r| r.expert_scale = false), &[]);
    assert_eq!(run(&same, &[("gate.routed_scaling_factor_param", 0.0, 1.0)]), unscaled, "scales of one are no scale");
    assert_ne!(unscaled[134], rows[134]);
    assert_ne!(run(&same, &[("block_sparse_moe.gate.weight_1", 0.0, 0.0)])[134], rows[134], "the second router is read");
    // the streams' norm: its epsilon inside the root is another model
    let inside = |d: &mut Described| {
        for b in d.desc.blocks.iter_mut() {
            if let Residual::Streams(s) = &mut b.residual {
                s.eps_outside = false;
            }
        }
    };
    // (at an epsilon that is something beside the streams: at the release's 1e-6 the two forms part below the bf16 of the logits)
    let coarse = ernie_tiny(serde_json::json!({"rms_norm_eps": 0.05}));
    let at_coarse = |edit: &dyn Fn(&mut Described)| {
        let mut d = describe(&coarse).unwrap_or_else(|e| panic!("{}", e.msg));
        edit(&mut d);
        RefModel::load(&d, &|n: &str| tensor(&d, &[], n)).unwrap().forward(&toks[..12], 0).unwrap_or_else(|e| panic!("{}", e.msg))
    };
    assert_ne!(at_coarse(&inside)[11], at_coarse(&same)[11]);
    // the drafter: its two stages and its own roles load (the Markov embedding turned), and a block of three is drafted from a context of six positions
    let src = |n: &str| tensor(&d, &[], n);
    let mut drafter = RefDSpark::load(&d, &src).unwrap_or_else(|e| panic!("{}", e.msg));
    assert_eq!((drafter.blocks.len(), drafter.p.t["markov_embed"].shape.clone()), (4, vec![61, 4]));
    let w1 = draw("model.mtp.markov_head.markov_w1.weight", 4 * 61);
    assert!((0..61).all(|v| (0..4).all(|r| drafter.p.vec("markov_embed").unwrap()[v * 4 + r] == w1[r * 61 + v])));
    let ctx: Vec<Vec<f32>> = (0..6).map(|t| draw(&format!("context {t}"), 2 * 32)).collect();
    let (embed, head) = (src("model.embed_tokens.weight").unwrap().1, src("lm_head.weight").unwrap().1);
    let out = drafter.draft(&ctx, 5, &embed, &head).unwrap_or_else(|e| panic!("{}", e.msg));
    assert_eq!((out.tokens.len(), out.base.len(), out.confidence.len()), (3, 3, 3));
    assert!(out.base.iter().flatten().all(|x| x.is_finite() && *x == bf16(*x)), "the drafter's base logits take the model's head's arithmetic");
    assert!(out.logits.iter().flatten().all(|x| x.is_finite()) && out.confidence.iter().all(|c| (0.0..=1.0).contains(c)));
}

/// Mistral Large as released (Mistral-Large-4-1T-A52B's config and header): 72 layers, each a latent attention with an indexer, then a gated MLP (the
/// first layer) or 128 experts and a shared MLP. Every fourth layer has no positions (no shared key part, keys of 128 from the latent); the others
/// rotate 64 channels by YaRN with learned factors. Every layer scales its queries by position. Layers 39, 43, ... 71 attend to layer 35's latents and
/// have no key-value latent product; from layer 17 two layers in four take the picks of the layer before. The checkpoint is in Mistral's names, its
/// products FP8 with `qscale_weight` scales but those the config leaves out; the vision encoder, its adapter and the draft model are left unread.
#[test]
fn mistral_large_describes_its_latent_attentions_their_indexers_and_its_experts() {
    use super::kinds::{Act, Body, Ffn, IndexForm, Mixer, Mla, NormKind, Rope, RopeScaling, RopeStyle, RouterScore};
    let h = fixture("headers/Mistral-Large-4-1T-A52B.json");
    let cfg: serde_json::Value = serde_json::from_str(h["config"].as_str().unwrap()).unwrap();
    let d = describe(&cfg).unwrap_or_else(|e| panic!("{}", e.msg));
    assert_eq!((d.desc.blocks.len(), d.desc.head.tied, d.desc.final_norm.eps, d.desc.embed.scale, d.fp8_block), (144, false, 1e-5, 1.0, Some(128)));
    assert_eq!((d.desc.hidden, d.desc.vocab, d.convention, d.desc.head.softcap), (6144, 204800, "Mistral (consolidated)", Some(30.0)));
    assert!(d.desc.mtp.is_empty() && d.mtp.is_empty() && d.mtp_layers == 0 && d.desc.vision.is_none());
    let yarn = RopeScaling::Yarn {
        factor: 2048.0,
        original_max: 512,
        beta_fast: 32.0,
        beta_slow: 1.0,
        mscale: None,
        mscale_all_dim: None,
        attention_factor: Some(1.0),
        truncate: true,
    };
    let rope = Rope { style: RopeStyle::Interleaved, scaling: yarn, ..Rope::plain(1e4, 64) };
    assert_eq!(d.desc.rope, Some(rope.clone()));
    let (mut nope, mut lent, mut shared, mut dense, mut sparse) = (0, 0, 0, 0, 0);
    for (i, b) in d.desc.blocks.iter().enumerate() {
        let l = i / 2;
        assert!(b.norm_out.is_none() && b.residual == super::kinds::Residual::Add && b.norm_in.eps == 1e-5, "block {i}");
        match &b.body {
            Body::Mixer(Mixer::Mla(x)) => {
                let turns = l % 4 != 3;
                assert_eq!((x.heads, x.q_lora, x.kv_lora, x.v_dim, x.scale, x.gate), (64, None, 512, 128, 128f32.powf(-0.5), None), "layer {l}");
                assert_eq!((x.nope, x.rope_dim, x.rope.as_ref()), if turns { (64, 64, Some(&rope)) } else { (128, 0, None) }, "layer {l}");
                assert_eq!((x.pos_scale, x.rope_learned, x.kv_norm.eps), (true, turns, 1e-5), "layer {l}");
                assert_eq!(x.kv_from, (l > 35 && !turns).then_some(70), "layer {l}: layer 35's block");
                let ix = x.indexer.as_ref().unwrap();
                assert_eq!((ix.heads, ix.head_dim, ix.top_k, ix.block, ix.form), (32, 128, 2048, 1, IndexForm::LAYER_KEYS), "layer {l}");
                assert_eq!((ix.shared, ix.rope, ix.pos_scale), (l >= 17 && matches!(l % 4, 1 | 2), if turns { 64 } else { 0 }, true), "layer {l}");
                assert!(matches!(ix.q_in, Some(n) if n.kind == NormKind::Layer && n.eps == 1e-6), "layer {l}");
                assert_eq!(x.not_run(), Some(Mla::POS_SCALE_NOT_RUN));
                (nope, lent, shared) = (nope + !turns as usize, lent + x.kv_from.is_some() as usize, shared + ix.shared as usize);
            }
            Body::Ffn(Ffn::Mlp(m)) => {
                assert!(l == 0 && m.gated && !m.bias && m.act == Act::Silu && m.clamp.is_none(), "layer {l}");
                dense += 1;
            }
            Body::Ffn(Ffn::Moe(m)) => {
                let r = &m.router;
                assert_eq!(
                    (m.experts, r.top_k, r.groups, r.renorm, r.bias, r.scale, r.score),
                    (128, 4, None, true, false, 1.0, RouterScore::Softmax),
                    "layer {l}"
                );
                assert!(!r.hash && m.latent.is_none() && !m.shared_gate && m.shared.is_some() && m.expert.clamp.is_none(), "layer {l}");
                sparse += 1;
            }
            _ => panic!("block {i}"),
        }
    }
    assert_eq!((nope, lent, shared, dense, sparse), (18, 9, 28, 1, 71));
    assert_eq!((d.embed.as_str(), d.final_norm.as_str(), d.head.as_deref()), ("tok_embeddings.weight", "norm.weight", Some("output.weight")));
    // every tensor the description reads, at the checkpoint's shape and in the stored format the description says
    let have = header_names(&h);
    let tensors = header_tensors(&h);
    let scale_names: BTreeSet<&String> = d.scales.values().collect();
    let (mut fp8, mut plain) = (0, 0);
    for n in d.names() {
        let (dtype, shape, _) = &tensors[&n];
        match (dtype.as_str(), d.scales.get(&n)) {
            // (an FP8 matrix: its scales a block of 128 x 128, under Mistral's name beside it)
            ("F8_E4M3", Some(s)) => {
                assert_eq!((s, d.shapes.get(&n)), (&format!("{}.qscale_weight", n.strip_suffix(".weight").unwrap()), Some(shape)), "{n}");
                assert_eq!((tensors[s].0.as_str(), &tensors[s].1[..]), ("F32", &[shape[0].div_ceil(128), shape[1].div_ceil(128)][..]), "{s}");
                fp8 += 1;
            }
            (_, None) if scale_names.contains(&n) => {}
            (_, None) => {
                assert_eq!(dtype, "BF16", "{n}: with no scales named");
                assert_eq!(d.shapes.get(&n), Some(shape), "{n}");
                plain += 1;
            }
            (t, s) => panic!("{n}: {t}, scales {s:?}"),
        }
    }
    // (72 layers' query and output products, 44 indexers' query products, the dense MLP's three matrices, 71 shared MLPs' and 71 layers of 128 experts')
    assert_eq!(fp8, 72 * 2 + 44 + 3 + 71 * 3 + 71 * 128 * 3);
    assert_eq!(d.names().len(), 2 * fp8 + plain);
    // (bf16 as released: the key-value latent's two products, the indexer's key product and head weights, the routers, the scales by position)
    for n in [
        "layers.0.attention.wkv_a_with_mqa.weight",
        "layers.0.attention.wkv_b.weight",
        "layers.0.attention.indexer.wk.weight",
        "layers.0.attention.indexer.weights_proj.weight",
        "layers.1.gate.weight",
        "layers.0.attention.ssmax_scaling.s",
        "lerope.alpha",
    ] {
        assert!(d.names().contains(n) && !d.scales.contains_key(n) && tensors[n].0 == "BF16", "{n}");
    }
    // (the shapes the layers' sorts make: a layer with no positions has no shared key part and wider keys; a layer that shares has no latent product)
    assert_eq!(d.shapes["layers.0.attention.wkv_a_with_mqa.weight"], [576, 6144]);
    assert_eq!(d.shapes["layers.3.attention.wkv_a_with_mqa.weight"], [512, 6144]);
    assert_eq!((&d.shapes["layers.0.attention.wkv_b.weight"], &d.shapes["layers.3.attention.wkv_b.weight"]), (&vec![12288, 512], &vec![16384, 512]));
    assert_eq!((&d.shapes["layers.5.attention.indexer.wq_b.weight"], &d.shapes["lerope.alpha"]), (&vec![4096, 6144], &vec![32]));
    for l in 0..72 {
        let own = !(l > 35 && l % 4 == 3);
        let full = !(l >= 17 && matches!(l % 4, 1 | 2));
        assert_eq!(d.names().contains(&format!("layers.{l}.attention.wkv_a_with_mqa.weight")), own, "layer {l}");
        assert_eq!(d.names().contains(&format!("layers.{l}.attention.kv_a_norm.weight")), own, "layer {l}");
        assert_eq!(d.names().contains(&format!("layers.{l}.attention.indexer.q_input_norm.bias")), full, "layer {l}");
        assert_eq!(d.blocks[2 * l].iter().any(|(r, _)| r == "rope_alpha"), l % 4 != 3, "layer {l}");
    }
    assert_eq!(d.unread, ["vision_encoder.", "pre_mm_projector_norm.", "patch_merger.", "vision_language_adapter.", "mtp_dspark."]);
    for left in ["vision_encoder.patch_conv.weight", "vision_language_adapter.w_in.weight", "mtp_dspark.backbone.mask_embedding"] {
        assert!(have.contains(left) && !d.names().contains(left), "{left}");
    }
    d.check(&have, true).unwrap_or_else(|e| panic!("{}", e.msg));
    // (described, not run: the executor names what it lacks)
    let e = crate::dynamic::split::loader(&d).err().map(|e| e.msg).unwrap_or_default();
    assert_eq!(e, format!("its layers hold {}", Mla::POS_SCALE_NOT_RUN));
}

/// A small config of the Mistral Large layout in its wrapper with `keys` over its text config: eight layers, every fourth without positions, the
/// last attending to layer 3's latents; indexers of two heads of 8 keeping 4 tokens, layers 2, 5 and 6 taking the picks of the layer before; the first
/// layer's feed-forward dense and the others 8 experts of which 2 a token; YaRN over 8 positions by 4.
fn mistral_tiny(keys: serde_json::Value) -> serde_json::Value {
    let mut text = serde_json::json!({
        "model_type": "mistral_large4_text", "hidden_size": 32, "num_hidden_layers": 8, "vocab_size": 61, "rms_norm_eps": 1e-5,
        "max_position_embeddings": 4096, "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 4, "hidden_act": "silu", "attention_bias": false,
        "kv_lora_rank": 12, "qk_nope_head_dim": 8, "qk_rope_head_dim": 4, "qk_head_dim": 12, "v_head_dim": 6,
        "rope_parameters": {"rope_type": "yarn", "rope_theta": 10000.0, "factor": 4, "original_max_position_embeddings": 8, "beta_fast": 32, "beta_slow": 1,
            "attention_factor": 1.0},
        "no_rope_layers": [1, 1, 1, 0, 1, 1, 1, 0], "no_rope_layer_interval": 4, "kv_source_layers": [0, 1, 2, 3, 4, 5, 6, 3],
        "layer_types": ["indexed_attention", "indexed_attention", "indexed_attention", "indexed_attention", "indexed_attention", "indexed_attention",
            "indexed_attention", "indexed_attention"],
        "index_n_heads": 2, "index_head_dim": 8, "index_topk": 4, "indexer_types": ["full", "full", "shared", "full", "full", "shared", "shared", "full"],
        "intermediate_size": 48, "first_k_dense_replace": 1, "mlp_layer_types": ["dense", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse"],
        "n_routed_experts": 8, "n_shared_experts": 1, "num_experts_per_tok": 2, "moe_intermediate_size": 16, "n_group": 1, "topk_group": 1,
        "routed_scaling_factor": 1.0, "final_logit_softcapping": 30.0, "tie_word_embeddings": false, "attention_dropout": 0.0, "use_cache": true,
    });
    text.as_object_mut().unwrap().extend(keys.as_object().unwrap().clone());
    serde_json::json!({
        "architectures": ["MistralLarge4ForConditionalGeneration"], "model_type": "mistral_large4", "tie_word_embeddings": false, "image_token_index": 10,
        "projector_hidden_act": "gelu", "vision_feature_layer": -1, "multimodal_projector_bias": false, "spatial_merge_size": 2,
        "transformers_weights": "consolidated.safetensors.index.json", "vision_config": {"model_type": "pixtral", "num_hidden_layers": 2}, "text_config": text,
    })
}

/// The Mistral Large builder reads a config as transformers' config class does: the layers without positions from the list or from the interval; each
/// layer's latents its own or an earlier layer's; the indexers' types from the list or the schedule; the dense layers from the list or the count;
/// the class's defaults where a key is absent; a quantization_config's FP8 products named with Mistral's scales. What the builder does not support,
/// and what the references read differently, is refused in words.
#[test]
fn mistral_large_reads_its_config_as_transformers_does() {
    use super::kinds::{Body, Ffn, Mixer, Mla, RopeScaling};
    let ok = |keys: serde_json::Value| describe(&mistral_tiny(keys)).unwrap_or_else(|e| panic!("{}", e.msg));
    let mla = |d: &super::describe::Described, l: usize| -> Mla {
        match &d.desc.blocks[2 * l].body {
            Body::Mixer(Mixer::Mla(x)) => x.clone(),
            b => panic!("layer {l}: {b:?}"),
        }
    };
    let d = ok(serde_json::json!({}));
    assert_eq!((0..8).map(|l| mla(&d, l).rope.is_some()).collect::<Vec<_>>(), [true, true, true, false, true, true, true, false]);
    assert_eq!((0..8).map(|l| mla(&d, l).kv_from).collect::<Vec<_>>(), [None, None, None, None, None, None, None, Some(6)], "layer 3's mixer is block 6");
    assert_eq!((0..8).map(|l| mla(&d, l).indexer.unwrap().shared).collect::<Vec<_>>(), [false, false, true, false, false, true, true, false]);
    assert_eq!((mla(&d, 0).nope, mla(&d, 0).rope_dim, mla(&d, 3).nope, mla(&d, 3).rope_dim), (8, 4, 12, 0));
    assert_eq!(
        (mla(&d, 0).scale, mla(&d, 3).scale, mla(&d, 0).indexer.unwrap().rope, mla(&d, 3).indexer.unwrap().rope),
        (12f32.powf(-0.5), 12f32.powf(-0.5), 4, 0)
    );
    assert!(
        matches!(&d.desc.blocks[1].body, Body::Ffn(Ffn::Mlp(_))) && d.desc.blocks.iter().skip(3).step_by(2).all(|b| matches!(b.body, Body::Ffn(Ffn::Moe(_))))
    );
    assert_eq!((&d.shapes["layers.0.attention.wkv_a_with_mqa.weight"], &d.shapes["layers.3.attention.wkv_a_with_mqa.weight"]), (&vec![16, 32], &vec![12, 32]));
    assert_eq!((&d.shapes["layers.0.attention.wkv_b.weight"], &d.shapes["layers.7.attention.wkv_b.weight"]), (&vec![28, 12], &vec![36, 12]));
    assert_eq!((&d.shapes["layers.0.attention.wq.weight"], &d.shapes["layers.3.attention.wq.weight"]), (&vec![24, 32], &vec![24, 32]));
    assert!(!d.names().contains("layers.7.attention.wkv_a_with_mqa.weight") && !d.names().contains("layers.7.attention.kv_a_norm.weight"));
    assert!(d.names().contains("layers.7.attention.indexer.wk.weight") && !d.names().contains("layers.2.attention.indexer.wk.weight"));
    assert!(d.names().contains("lerope.alpha") && d.scales.is_empty() && d.fp8_block.is_none());
    assert_eq!(d.shapes["layers.1.experts.7.w3.weight"], [16, 32]);
    // (the text model alone, with no wrapper: the same description)
    let alone = describe(&mistral_tiny(serde_json::json!({}))["text_config"]).unwrap_or_else(|e| panic!("{}", e.msg));
    assert_eq!((format!("{:?}", alone.desc), alone.names()), (format!("{:?}", d.desc), d.names()));
    // (the config class's defaults: every (interval)-th layer without positions, 4 where unsaid; every layer its own latents and indexer; the first
    // layer dense; the rotary YaRN over 512 positions by 2048 with no magnitude factor; eps 1e-5; a cap of 30; the head its own matrix)
    let bare = ok(serde_json::json!({"no_rope_layers": null, "no_rope_layer_interval": null, "kv_source_layers": null, "indexer_types": null,
        "mlp_layer_types": null, "first_k_dense_replace": null, "rope_parameters": null, "rms_norm_eps": null, "final_logit_softcapping": null,
        "layer_types": null, "routed_scaling_factor": null, "n_group": null, "topk_group": null, "n_shared_experts": null, "head_dim": null, "qk_head_dim": null}));
    assert_eq!((0..8).map(|l| mla(&bare, l).rope.is_some()).collect::<Vec<_>>(), [true, true, true, false, true, true, true, false]);
    assert!((0..8).all(|l| mla(&bare, l).kv_from.is_none() && !mla(&bare, l).indexer.unwrap().shared));
    assert!(matches!(&bare.desc.blocks[1].body, Body::Ffn(Ffn::Mlp(_))) && matches!(&bare.desc.blocks[3].body, Body::Ffn(Ffn::Moe(_))));
    assert!(
        matches!(bare.desc.rope.as_ref().unwrap().scaling, RopeScaling::Yarn { factor, original_max: 512, attention_factor: Some(a), .. } if factor == 2048.0 && a == 1.0)
    );
    assert_eq!((bare.desc.final_norm.eps, bare.desc.head.softcap, bare.desc.head.tied), (1e-5, Some(30.0), false));
    let third = ok(serde_json::json!({"no_rope_layers": null, "no_rope_layer_interval": 3, "kv_source_layers": null}));
    assert_eq!((0..8).map(|l| mla(&third, l).rope.is_some()).collect::<Vec<_>>(), [true, true, false, true, true, false, true, true]);
    // (the indexers' schedule where the list is absent, as transformers makes it: a pattern, or a frequency and an offset)
    let pat = ok(serde_json::json!({"indexer_types": null, "index_topk_pattern": "FFSFFSSF"}));
    assert_eq!((format!("{:?}", pat.desc), pat.names()), (format!("{:?}", d.desc), d.names()));
    let freq = ok(serde_json::json!({"indexer_types": null, "index_topk_freq": 2, "index_skip_topk_offset": 3}));
    assert_eq!((0..8).map(|l| mla(&freq, l).indexer.unwrap().shared).collect::<Vec<_>>(), [false, false, false, true, false, true, false, true]);
    // (the dense layers by the count; the older name of the experts' count; a tied head; YaRN's factor on every channel squares into the scores')
    let two = ok(serde_json::json!({"mlp_layer_types": null, "first_k_dense_replace": 2}));
    assert!(matches!(&two.desc.blocks[3].body, Body::Ffn(Ffn::Mlp(_))) && two.names().contains("layers.1.feed_forward.w1.weight"));
    let named = ok(serde_json::json!({"num_experts": 4}));
    assert!(matches!(&named.desc.blocks[3].body, Body::Ffn(Ffn::Moe(m)) if m.experts == 4) && !named.names().contains("layers.1.experts.4.w1.weight"));
    let tied = ok(serde_json::json!({"tie_word_embeddings": true}));
    assert!(tied.desc.head.tied && tied.head.is_none());
    let ms = ok(serde_json::json!({"rope_parameters": {"rope_type": "yarn", "rope_theta": 10000.0, "factor": 4, "original_max_position_embeddings": 8,
        "mscale": 1.0, "mscale_all_dim": 1.0}}));
    let f = 0.1 * 4f64.ln() + 1.0;
    assert_eq!(mla(&ms, 0).scale, 12f32.powf(-0.5) * (f * f) as f32);
    // (a quantization_config names the FP8 products' scales, Mistral's `qscale_weight`, and leaves out what it lists by transformers' module names)
    let mut q = mistral_tiny(serde_json::json!({}));
    q["quantization_config"] = serde_json::json!({"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [128, 128],
        "modules_to_not_convert": ["model.vision_tower", "model.multi_modal_projector", "lm_head", "kv_b_proj", "indexer.wk", "indexer.weights_proj",
            "kv_a_proj_with_mqa"]});
    let q = describe(&q).unwrap_or_else(|e| panic!("{}", e.msg));
    assert_eq!(q.fp8_block, Some(128));
    for held in [
        "layers.0.attention.wq.weight",
        "layers.7.attention.wo.weight",
        "layers.3.attention.indexer.wq_b.weight",
        "layers.0.feed_forward.w2.weight",
        "layers.1.shared_experts.w1.weight",
        "layers.7.experts.7.w3.weight",
    ] {
        assert_eq!(q.scales.get(held), Some(&held.replace(".weight", ".qscale_weight")), "{held}");
    }
    for left in [
        "layers.0.attention.wkv_a_with_mqa.weight",
        "layers.7.attention.wkv_b.weight",
        "layers.0.attention.indexer.wk.weight",
        "layers.0.attention.indexer.weights_proj.weight",
        "layers.1.gate.weight",
        "layers.0.attention.kv_a_norm.weight",
        "layers.0.attention.ssmax_scaling.b",
        "lerope.alpha",
        "tok_embeddings.weight",
        "output.weight",
    ] {
        assert!(q.names().contains(left) && !q.scales.contains_key(left), "{left}");
    }
    assert!(q.names().contains("layers.0.attention.wq.qscale_weight") && q.names().len() == d.names().len() + q.scales.len());
    let refused = |c: serde_json::Value| describe(&c).err().map(|e| e.msg).unwrap_or_default();
    for (keys, word) in [
        (serde_json::json!({"q_lora_rank": 8}), "q_lora_rank = 8 is not supported"),
        (serde_json::json!({"hidden_act": "gelu"}), "hidden_act"),
        (serde_json::json!({"attention_bias": true}), "attention_bias"),
        (serde_json::json!({"num_key_value_heads": 1}), "num_key_value_heads"),
        (serde_json::json!({"qk_rope_head_dim": 3}), "qk_rope_head_dim = 3"),
        (serde_json::json!({"layer_types": ["indexed_attention", "full_attention"]}), "layer_types is not 8 layers of indexed_attention"),
        (serde_json::json!({"no_rope_layers": [1, 1, 1]}), "no_rope_layers = [1,1,1] is not a value a layer"),
        (serde_json::json!({"no_rope_layers": [1, 1, 1, 0, 1, 1, 2, 0]}), "no_rope_layers[6] = 2"),
        (serde_json::json!({"no_rope_layers": null, "no_rope_layer_interval": 0}), "no_rope_layer_interval = 0"),
        (serde_json::json!({"rope_parameters": {"rope_type": "axial", "rope_theta": 10000.0}}), "rope_type axial is not supported"),
        (serde_json::json!({"rope_parameters": 5}), "rope_parameters = 5"),
        (serde_json::json!({"kv_source_layers": [0, 1, 2]}), "kv_source_layers = [0,1,2] is not a layer's number a layer"),
        (serde_json::json!({"kv_source_layers": [0, 1, 2, 3, 4, 5, 7, 7]}), "kv_source_layers[6] = 7 is not an earlier layer"),
        (serde_json::json!({"kv_source_layers": [0, 1, 2, 3, 4, 5, 6, 4]}), "a layer without positions attending to the latents of one with"),
        (serde_json::json!({"kv_source_layers": [0, 0, 1, 3, 4, 5, 6, 7]}), "kv_source_layers[2] = 1 is not an earlier layer that makes its own latents"),
        (serde_json::json!({"kv_source_layers": [0, 0, 2, 3, 4, 4, 0, 7]}), "with layer 4 lending its latents between them"),
        (serde_json::json!({"indexer_types": ["shared", "full"]}), "the first layer's indexer is shared"),
        (serde_json::json!({"indexer_types": ["full", "full"]}), "indexer_types has 2 entries for 8 layers"),
        (serde_json::json!({"indexer_types": ["full", "pooled"]}), "an indexer of type"),
        (serde_json::json!({"index_head_dim": 2}), "an indexer of 2 heads of 2"),
        (serde_json::json!({"mlp_layer_types": ["dense", "latent", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse"]}), "mlp_layer_types[1]"),
        (serde_json::json!({"n_group": 2}), "n_group = 2"),
        (serde_json::json!({"topk_group": 2}), "topk_group = 2"),
        (serde_json::json!({"n_shared_experts": 0}), "n_shared_experts = 0"),
        (serde_json::json!({"num_experts_per_tok": 9}), "num_experts_per_tok = 9 of 8 experts"),
        (serde_json::json!({"final_logit_softcapping": 0.0}), "final_logit_softcapping = 0"),
        (serde_json::json!({"scoring_func": "sigmoid", "swiglu_limit": 7.0}), "config keys scoring_func, swiglu_limit are not understood"),
    ] {
        let e = refused(mistral_tiny(keys));
        assert!(e.contains(word), "{word}: {e}");
    }
    // (the wrapper's own keys; a quantization of another format)
    for (key, v, word) in [
        ("audio_config", serde_json::json!({}), "config key audio_config is not understood"),
        ("quantization_config", serde_json::json!({"quant_method": "fp8"}), "weight_block_size"),
        ("quantization_config", serde_json::json!({"quant_method": "awq", "weight_block_size": [128, 128]}), "quantization awq"),
    ] {
        let mut c = mistral_tiny(serde_json::json!({}));
        c[key] = v;
        let e = refused(c);
        assert!(e.contains(word), "{word}: {e}");
    }
    // (described, not run: the planner's lowering and the loader name what the executor lacks)
    let Err(e) = crate::dynamic::split::lowered_described(&d, 8, 4096) else { panic!("a scale by position lowered") };
    assert!(e.msg.contains("queries are scaled by their position, which the executor does not run yet"), "{}", e.msg);
}

/// The Mistral Large layout on the CPU reference, a random model of the small config (every tensor of the shape its description gives). A prompt at
/// once, the same prompt a token at a time and a prompt then its next tokens are the same bits (the latent caches, the indexers' keys, the lent picks
/// and the lent latents kept between the calls), and a later token changes no earlier row. Each of the layout's forms is read: scales by position of
/// zero with biases of one are, to the bit, the model with no scale by position, and the scales as drawn are another answer; learned factors of zero
/// are the rotary as it is, to the bit; the lent latents, the lent picks, the positions and the cap each change the answer.
#[test]
fn mistral_large_on_the_reference_steps_as_it_prompts_and_reads_each_of_its_forms() {
    use super::kinds::{Body, Mixer, ModelDesc};
    // (a fixed draw a tensor: its name's bytes seed it; a norm's weights and a scale by position's bias near 1)
    fn draw(n: &str, len: usize) -> Vec<f32> {
        let mut s = n.bytes().fold(0x9E37_79B9_7F4A_7C15u64, |a, b| (a ^ b as u64).wrapping_mul(0x0100_0000_01B3));
        let near_one = (n.contains("norm") && !n.ends_with(".bias")) || n.ends_with("ssmax_scaling.b");
        (0..len)
            .map(|_| {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                let u = ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5;
                if near_one {
                    1.0 + 0.2 * u
                } else {
                    0.6 * u
                }
            })
            .collect()
    }
    // (`edit`: a change to the description before it is loaded; `set`: a value for every element of the tensors whose names end so, in the draw's place)
    let load = |cfg: &serde_json::Value, edit: &dyn Fn(&mut ModelDesc), set: &[(&str, f32)]| -> RefModel {
        let mut d = describe(cfg).unwrap_or_else(|e| panic!("{}", e.msg));
        edit(&mut d.desc);
        let src = |n: &str| {
            let shape = d.shapes.get(n)?.clone();
            let len: usize = shape.iter().product();
            let v = match set.iter().find(|(end, _)| n.ends_with(end)) {
                Some(&(_, x)) => vec![x; len],
                None => draw(n, len),
            };
            Some((shape, v))
        };
        RefModel::load(&d, &src).unwrap_or_else(|e| panic!("{}", e.msg))
    };
    let same = |_: &mut ModelDesc| {};
    let on_mla = |d: &mut ModelDesc, f: &dyn Fn(&mut super::kinds::Mla)| {
        for b in d.blocks.iter_mut() {
            if let Body::Mixer(Mixer::Mla(x)) = &mut b.body {
                f(x);
            }
        }
    };
    let run = |cfg: &serde_json::Value, toks: &[u32]| load(cfg, &same, &[]).forward(toks, 0).unwrap_or_else(|e| panic!("{}", e.msg));
    let a: [u32; 9] = [5, 9, 13, 17, 21, 25, 29, 33, 37];
    let base = mistral_tiny(serde_json::json!({}));
    let rows = run(&base, &a);
    assert_eq!((rows.len(), rows[8].len()), (9, 61));
    assert!(rows.iter().flatten().all(|x| x.is_finite() && x.abs() < 30.0) && rows[8].iter().any(|&x| x != rows[8][0]), "finite, under the cap, not one value");
    // (causal)
    let mut later = a;
    later[8] = 44;
    assert_eq!(rows[..8], run(&base, &later)[..8], "a later token changes no earlier row");
    // (a token at a time, and a prompt then its next tokens: the same bits)
    let mut m = load(&base, &same, &[]);
    let mut by_step = Vec::new();
    for (t, &tok) in a.iter().enumerate() {
        by_step.extend(m.forward(&[tok], t as u32).unwrap_or_else(|e| panic!("{}", e.msg)));
    }
    assert_eq!(by_step, rows, "a token at a time is the prompt");
    let mut m = load(&base, &same, &[]);
    let mut parts = m.forward(&a[..5], 0).unwrap();
    parts.extend(m.forward(&a[5..], 5).unwrap());
    assert_eq!(parts, rows, "a prompt, then its next tokens");
    // the scale by position: a factor of one is the model without it, the attention's and the indexer's; as drawn it is another answer, and it is
    // the position that scales (the first row, whose factor is the bias, is the one with every scale at zero)
    let unscaled = |d: &mut ModelDesc| {
        on_mla(d, &|x| {
            x.pos_scale = false;
            x.indexer.as_mut().unwrap().pos_scale = false;
        })
    };
    let ones = load(&base, &same, &[("ssmax_scaling.s", 0.0), ("ssmax_scaling.b", 1.0)]).forward(&a, 0).unwrap();
    assert_eq!(ones, load(&base, &unscaled, &[]).forward(&a, 0).unwrap(), "a factor of one");
    assert_ne!(rows[8], ones[8], "the scales as drawn");
    let flat = load(&base, &same, &[("ssmax_scaling.s", 0.0)]).forward(&a, 0).unwrap();
    assert_eq!(rows[0], flat[0], "position 0: the bias alone");
    assert_ne!(rows[8], flat[8], "position 8: the scale times ln 9");
    // the learned frequencies: factors of zero are the rotary as it is
    let plain_rope = |d: &mut ModelDesc| on_mla(d, &|x| x.rope_learned = false);
    let zero = load(&base, &same, &[("lerope.alpha", 0.0)]).forward(&a, 0).unwrap();
    assert_eq!(zero, load(&base, &plain_rope, &[]).forward(&a, 0).unwrap(), "exp(0) = 1");
    assert_ne!(rows[8], zero[8], "the factors as drawn");
    // the positions are read (the rotary and the scales): the same tokens further on are another answer
    assert_ne!(rows[8], load(&base, &same, &[]).forward(&a, 7).unwrap()[8]);
    // the lent latents and the lent picks are read: every layer its own is another answer, and the indexers bite (keeping every token is another)
    assert_ne!(rows[8], run(&mistral_tiny(serde_json::json!({"kv_source_layers": null})), &a)[8], "layer 7 on layer 3's latents");
    assert_ne!(rows[8], run(&mistral_tiny(serde_json::json!({"indexer_types": null})), &a)[8], "layers 2, 5 and 6 on the picks before them");
    let all = run(&mistral_tiny(serde_json::json!({"index_topk": 64})), &a);
    assert_eq!(rows[..4], all[..4], "four tokens kept of four");
    assert_ne!(rows[8], all[8], "four tokens kept of nine");
    // the cap, cap tanh(logit / cap): in f32 and with a cap among the logits' own sizes, the head's product through the formula
    let head = |cap: Option<f32>| {
        move |d: &mut ModelDesc| {
            d.head.softcap = cap;
            d.head.rounding = super::kinds::Rounding::F32;
        }
    };
    let uncapped = load(&base, &head(None), &[]).forward(&a, 0).unwrap();
    let capped = load(&base, &head(Some(0.05)), &[]).forward(&a, 0).unwrap();
    let want: Vec<f32> = uncapped[8].iter().map(|&x| 0.05 * (x / 0.05).tanh()).collect();
    assert!(super::reference::testing::rel(&capped[8], &want) < 1e-5 && super::reference::testing::rel(&capped[8], &uncapped[8]) > 1e-2);
}

/// MiniMax M3 as released (MiniMax-M3.1-Flash's config and header): 60 layers of one attention and a feed-forward. Every attention 64 query heads over 4
/// of 128, its queries and keys normed a head times (1 + weight), the first 64 channels of a head rotated at a base of 10,000,000, and the block-max
/// indexer: 4 index heads of 128, one a key head, keeping 16 blocks of 128 positions, the query's own among them. The first three layers a dense MLP of
/// 12,288, the other 57 a mixture of 128 experts of 3,072 with one shared expert, 4 a token by the sigmoid with a choice bias, their sum doubled; the
/// gpt-oss SwiGLU at 1.702 and a limit of 7 everywhere. The products FP8 with a byte's scale a run of 32, the experts NVFP4 with their runs' scales and
/// the scales' own scale, the routers fp32; the vision tower, its projectors and the experts' input scales are left unread, and the checkpoint is read whole.
#[test]
fn minimax_m3_1_flash_describes_its_sparse_attention_its_experts_and_its_stored_formats() {
    use super::kinds::{Act, Body, Clamp, Ffn, IndexForm, Mixer, Norm, Rope, RouterScore};
    let h = fixture("headers/MiniMax-M3.1-Flash.json");
    let cfg: serde_json::Value = serde_json::from_str(h["config"].as_str().unwrap()).unwrap();
    assert!(cfg["text_config"].get("model_type").is_none() && cfg["model_type"] == "minimax_m3_vl");
    let d = describe(&cfg).unwrap_or_else(|e| panic!("{}", e.msg));
    let pn = Norm::rms_plus_one(1e-6);
    let mlp = super::kinds::Mlp { act: Act::SwigluOai { alpha: 1.702 }, gated: true, bias: false, clamp: Some(Clamp::Gate(7.0)) };
    assert_eq!((d.desc.blocks.len(), d.desc.head.tied, d.desc.final_norm, d.desc.embed.scale, d.fp8_block), (120, false, pn, 1.0, None));
    assert_eq!((d.desc.hidden, d.desc.vocab, d.desc.rope.as_ref()), (6144, 200_064, Some(&Rope::plain(1e7, 64))));
    assert!(d.desc.mtp.is_empty() && d.desc.vision.is_none() && d.mtp_layers == 0);
    let (mut attentions, mut dense, mut sparse) = (0, 0, 0);
    for (i, b) in d.desc.blocks.iter().enumerate() {
        assert!(b.norm_in == pn && b.norm_out.is_none() && b.residual == super::kinds::Residual::Add, "block {i}");
        match &b.body {
            Body::Mixer(Mixer::Attention(a)) => {
                assert_eq!((a.heads, a.kv_heads, a.head_dim, a.v_head_dim, a.scale), (64, 4, 128, 128, 128f32.powf(-0.5)), "layer {}", i / 2);
                assert_eq!(
                    (a.q_norm, a.k_norm, a.v_norm, a.gate, a.window, a.sinks, a.bias),
                    (Some(pn), Some(pn), None, None, None, false, false),
                    "layer {}",
                    i / 2
                );
                // (the first 64 channels of a head, paired (i, i + 32), 32 angles over a span of 64)
                assert_eq!(a.rope.as_ref(), Some(&Rope::plain(1e7, 64)), "layer {}", i / 2);
                let ix = a.indexer.as_ref().unwrap();
                assert_eq!((ix.heads, ix.head_dim, ix.top_k, ix.block, ix.shared), (4, 128, 16 * 128, 128, false), "layer {}", i / 2);
                assert_eq!(ix.form, IndexForm::Max { norm: pn, local: 1 }, "layer {}", i / 2);
                attentions += 1;
            }
            Body::Ffn(Ffn::Mlp(m)) => {
                assert!(i < 6 && *m == mlp, "block {i}: {m:?}");
                assert_eq!(m.not_run(), Some(Act::SWIGLU_OAI_NOT_RUN));
                dense += 1;
            }
            Body::Ffn(Ffn::Moe(m)) => {
                assert_eq!(
                    (m.experts, m.router.top_k, m.router.renorm, m.router.bias, m.router.scale, m.router.logit_scale),
                    (128, 4, true, true, 2.0, 1.0),
                    "block {i}"
                );
                assert!(
                    m.router.score == RouterScore::Sigmoid && m.router.groups.is_none() && !m.router.hash && m.latent.is_none() && !m.shared_gate,
                    "block {i}"
                );
                assert_eq!((m.expert, m.shared), (mlp, Some(mlp)), "block {i}");
                sparse += 1;
            }
            _ => panic!("block {i}"),
        }
    }
    assert_eq!((attentions, dense, sparse), (60, 3, 57));
    // every tensor the description reads, at the checkpoint's shape and in the stored format the description says
    let have = header_names(&h);
    let tensors = header_tensors(&h);
    let (mut fp8, mut fp4, mut plain) = (0, 0, 0);
    let scale_names: BTreeSet<&String> = d.scales.values().collect();
    for n in d.names() {
        let (dtype, shape, _) = &tensors[&n];
        match (dtype.as_str(), d.scales.get(&n)) {
            // (an expert's runs' scales, FP8 themselves, and their own scale: one fp32 number a matrix)
            ("F8_E4M3", Some(g)) if n.ends_with(".weight_scale") => {
                assert_eq!((g, &tensors[g].0, tensors[g].1.len()), (&n.replace(".weight_scale", ".weight_global_scale"), &"F32".to_string(), 0), "{n}");
            }
            // (a product: FP8, a byte's scale a run of 32 columns of a row)
            ("F8_E4M3", Some(s)) => {
                assert_eq!((s, d.shapes.get(&n)), (&format!("{n}_scale_inv"), Some(shape)), "{n}");
                assert_eq!((tensors[s].0.as_str(), &tensors[s].1), ("U8", &vec![shape[0], shape[1] / 32]), "{s}");
                fp8 += 1;
            }
            // (an expert's matrix: two 4-bit values a byte along a row, an FP8 scale a run of 16)
            ("U8", Some(s)) if n.ends_with(".weight_packed") => {
                let want = &d.shapes[&n];
                assert_eq!((s, shape), (&n.replace(".weight_packed", ".weight_scale"), &vec![want[0], want[1] / 2]), "{n}");
                assert_eq!((tensors[s].0.as_str(), &tensors[s].1), ("F8_E4M3", &vec![want[0], want[1] / 16]), "{s}");
                fp4 += 1;
            }
            (_, None) if scale_names.contains(&n) => {}
            (_, None) => {
                let router = n.ends_with("block_sparse_moe.gate.weight") || n.ends_with("e_score_correction_bias");
                assert_eq!(dtype, if router { "F32" } else { "BF16" }, "{n} with no scales named");
                assert_eq!(d.shapes.get(&n), Some(shape), "{n}");
                plain += 1;
            }
            (t, s) => panic!("{n}: {t}, scales {s:?}"),
        }
    }
    // (60 layers of q, k, v, o and the indexer's two products, three dense MLPs and 57 shared experts of three matrices; 57 layers of 128 experts of three;
    // 60 layers of two norms and four head norms, 57 routers and their biases, the embedding, the last norm and the head)
    assert_eq!((fp8, fp4, plain), (60 * 6 + 3 * 3 + 57 * 3, 57 * 128 * 3, 60 * 6 + 57 * 2 + 3));
    let l = "language_model.model.layers.59";
    assert_eq!(d.shapes[&format!("{l}.self_attn.index_q_proj.weight")], [4 * 128, 6144]);
    assert_eq!(d.shapes[&format!("{l}.self_attn.index_k_proj.weight")], [128, 6144]);
    assert_eq!(d.shapes[&format!("{l}.block_sparse_moe.experts.127.w2.weight_packed")], [6144, 3072]);
    assert_eq!(
        d.scales[&format!("{l}.block_sparse_moe.shared_experts.up_proj.weight")],
        format!("{l}.block_sparse_moe.shared_experts.up_proj.weight_scale_inv")
    );
    assert!(!d.scales.contains_key(&format!("{l}.block_sparse_moe.gate.weight")) && !d.scales.contains_key("language_model.lm_head.weight"));
    // (left unread: the vision tower and its projectors, and each expert matrix's input scale)
    assert_eq!(d.unread, ["vision_tower.", "multi_modal_projector.", "patch_merge_mlp.", "*.input_global_scale"]);
    assert_eq!(have.iter().filter(|n| n.ends_with(".input_global_scale")).count(), 57 * 128 * 3);
    assert!(["vision_tower.", "multi_modal_projector.", "patch_merge_mlp."].iter().all(|p| have.iter().any(|n| n.starts_with(p))));
    assert!(!d.names().iter().any(|n| n.ends_with(".input_global_scale") || !n.starts_with("language_model.")));
    assert_eq!((have.len(), d.names().len()), (89_632, 67_221));
    d.check(&have, true).unwrap_or_else(|e| panic!("{}", e.msg));
}

/// A small config of the MiniMax M3 layout with `keys` over its text model's (a key `sparse`: over its `sparse_attention_config`'s; `top`: over the
/// wrapper's): three layers of 4 query heads over 2 of 8 with half a head rotated, the first and the last sparse (2 index heads of 8 keeping 2 blocks of 2
/// positions, the query's own among them), the first a dense MLP and the others 6 experts of 2 a token with a shared expert.
fn minimax_tiny(keys: serde_json::Value) -> serde_json::Value {
    let sparse = serde_json::json!({
        "use_sparse_attention": true, "sparse_index_dim": 8, "sparse_num_index_heads": 2, "sparse_topk_blocks": 2, "sparse_block_size": 2,
        "sparse_score_type": "max", "sparse_init_block": 0, "sparse_local_block": 1, "sparse_attention_freq": [1, 0, 1],
        "sparse_disable_index_value": [1, 1, 1],
    });
    let mut text = serde_json::json!({
        "hidden_size": 32, "intermediate_size": 16, "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
        "vocab_size": 61, "max_position_embeddings": 4096, "rms_norm_eps": 1e-6, "use_gemma_norm": true, "attention_output_gate": false,
        "rope_theta": 10000000, "rotary_dim": 4, "partial_rotary_factor": 0.5, "hidden_act": "swigluoai", "use_qk_norm": true,
        "tie_word_embeddings": false, "dense_intermediate_size": 48, "shared_intermediate_size": 16, "num_local_experts": 6, "num_experts_per_tok": 2,
    });
    text.as_object_mut().unwrap().extend(
        serde_json::json!({
            "n_shared_experts": 1, "scoring_func": "sigmoid", "use_routing_bias": true, "moe_layer_freq": [0, 1, 1], "qk_norm_type": "per_head",
            "swiglu_alpha": 1.702, "swiglu_limit": 7.0, "routed_scaling_factor": 2.0, "sparse_attention_config": sparse,
        })
        .as_object()
        .unwrap()
        .clone(),
    );
    let mut c = serde_json::json!({
        "architectures": ["MiniMaxM3SparseForConditionalGeneration"], "model_type": "minimax_m3_vl", "image_token_index": 200025, "torch_dtype": "bfloat16",
        "text_config": text,
    });
    let mut keys = keys.as_object().unwrap().clone();
    if let Some(s) = keys.remove("sparse") {
        c["text_config"]["sparse_attention_config"].as_object_mut().unwrap().extend(s.as_object().unwrap().clone());
    }
    if let Some(t) = keys.remove("top") {
        c.as_object_mut().unwrap().extend(t.as_object().unwrap().clone());
    }
    c["text_config"].as_object_mut().unwrap().extend(keys);
    c
}

/// The MiniMax M3 builder reads a config as the release's code does: the norms times the weight where `use_gemma_norm` is unsaid, the whole head rotated at
/// 10,000 where the rotary's keys are, every layer a mixture where `moe_layer_freq` is, no shared expert, no choice bias and an unscaled sum where their
/// keys are, no indexer without a `sparse_attention_config` and none on a layer its list says 0; a quantization_config's formats named as scales; and
/// what it does not know, or what the references do not agree on, is refused by name.
#[test]
fn minimax_m3_reads_its_config_as_the_releases_code_does() {
    use super::kinds::{Act, Attn, Body, Clamp, Ffn, IndexForm, Mixer, Moe, Norm, Rope};
    let attn = |d: &super::describe::Described, layer: usize| -> Attn {
        match &d.desc.blocks[2 * layer].body {
            Body::Mixer(Mixer::Attention(a)) => a.clone(),
            _ => panic!(),
        }
    };
    let moe = |d: &super::describe::Described, layer: usize| -> Moe {
        match &d.desc.blocks[2 * layer + 1].body {
            Body::Ffn(Ffn::Moe(m)) => m.clone(),
            b => panic!("layer {layer}: {b:?}"),
        }
    };
    let d = describe(&minimax_tiny(serde_json::json!({}))).unwrap_or_else(|e| panic!("{}", e.msg));
    let pn = Norm::rms_plus_one(1e-6);
    let (a, b) = (attn(&d, 0), attn(&d, 1));
    assert_eq!((a.heads, a.kv_heads, a.head_dim, a.q_norm, a.rope.as_ref()), (4, 2, 8, Some(pn), Some(&Rope::plain(1e7, 4))));
    let ix = a.indexer.as_ref().unwrap();
    assert_eq!((ix.heads, ix.head_dim, ix.top_k, ix.block, ix.form), (2, 8, 4, 2, IndexForm::Max { norm: pn, local: 1 }));
    assert!(b.indexer.is_none() && attn(&d, 2).indexer == a.indexer);
    assert!(matches!(&d.desc.blocks[1].body, Body::Ffn(Ffn::Mlp(m)) if m.act == Act::SwigluOai { alpha: 1.702 } && m.clamp == Some(Clamp::Gate(7.0))));
    let m = moe(&d, 1);
    assert_eq!((m.experts, m.router.top_k, m.router.bias, m.router.scale, m.shared, m.expert.clamp), (6, 2, true, 2.0, Some(m.expert), Some(Clamp::Gate(7.0))));
    assert!(d.scales.is_empty() && d.fp8_block.is_none());
    let l = |n: usize, rest: &str| format!("language_model.model.layers.{n}.{rest}");
    assert_eq!((&d.shapes[&l(0, "self_attn.index_q_proj.weight")], &d.shapes[&l(0, "self_attn.index_k_norm.weight")]), (&vec![16, 32], &vec![8]));
    assert_eq!((&d.shapes[&l(0, "mlp.gate_proj.weight")], &d.shapes[&l(2, "block_sparse_moe.experts.5.w3.weight")]), (&vec![48, 32], &vec![16, 32]));
    assert_eq!(d.shapes[&l(1, "block_sparse_moe.shared_experts.down_proj.weight")], [32, 16]);
    assert!(!d.names().iter().any(|n| n.starts_with(&l(1, "self_attn.index_"))) && d.names().contains(&l(1, "block_sparse_moe.e_score_correction_bias")));
    assert_eq!((d.embed.as_str(), d.head.as_deref()), ("language_model.model.embed_tokens.weight", Some("language_model.lm_head.weight")));
    // (unsaid: the norms times their weights, the whole head rotated at 10,000, every layer a mixture with no shared expert, no choice bias, an unscaled sum)
    let mut plain = minimax_tiny(serde_json::json!({}));
    for key in [
        "use_gemma_norm",
        "rotary_dim",
        "partial_rotary_factor",
        "rope_theta",
        "moe_layer_freq",
        "n_shared_experts",
        "shared_intermediate_size",
        "use_routing_bias",
        "routed_scaling_factor",
        "sparse_attention_config",
        "dense_intermediate_size",
        "use_qk_norm",
        "attention_output_gate",
    ] {
        plain["text_config"].as_object_mut().unwrap().remove(key);
    }
    let p = describe(&plain).unwrap_or_else(|e| panic!("{}", e.msg));
    let (a, m) = (attn(&p, 0), moe(&p, 0));
    assert_eq!((a.q_norm, a.rope.as_ref(), a.indexer.as_ref(), p.desc.final_norm), (Some(Norm::rms(1e-6)), Some(&Rope::plain(1e4, 8)), None, Norm::rms(1e-6)));
    assert_eq!((m.shared, m.router.bias, m.router.scale), (None, false, 1.0));
    assert!(!p.names().iter().any(|n| n.contains("index_") || n.contains("shared_experts") || n.contains("e_score_correction_bias") || n.contains(".mlp.")));
    // (a head's width unsaid is the model's over its heads)
    let mut wide = minimax_tiny(serde_json::json!({"rotary_dim": 8, "partial_rotary_factor": 1.0}));
    wide["text_config"].as_object_mut().unwrap().remove("head_dim");
    assert_eq!(attn(&describe(&wide).unwrap_or_else(|e| panic!("{}", e.msg)), 1).head_dim, 8);
    // (the release's quantization_config: the products FP8 with a scale a run of 32, but the modules it leaves out; the experts' codes, their runs' scales
    // and the scales' own scale)
    let quant = serde_json::json!({"quant_method": "mxfp8", "activation_scheme": "dynamic", "weight_block_size": [1, 32], "group_size": 16,
        "moe_quant_algo": "NVFP4", "moe_quant_format": "nvfp4-pack-quantized",
        "ignored_layers": ["language_model.model.layers.1.block_sparse_moe.gate", "lm_head", "model.embed_tokens", "vision_tower", "model.layers.2.self_attn.o_proj"]});
    let q = describe(&minimax_tiny(serde_json::json!({"top": {"quantization_config": quant}}))).unwrap_or_else(|e| panic!("{}", e.msg));
    let scale = |n: &str| q.scales.get(n).map(String::as_str);
    for w in ["self_attn.q_proj", "self_attn.index_k_proj", "mlp.down_proj"] {
        assert_eq!(scale(&l(0, &format!("{w}.weight"))), Some(l(0, &format!("{w}.weight_scale_inv")).as_str()), "{w}");
    }
    assert_eq!(
        scale(&l(1, "block_sparse_moe.shared_experts.gate_proj.weight")),
        Some(l(1, "block_sparse_moe.shared_experts.gate_proj.weight_scale_inv").as_str())
    );
    let e = l(1, "block_sparse_moe.experts.3.w1");
    assert!(q.names().contains(&format!("{e}.weight_packed")) && !q.names().contains(&format!("{e}.weight")));
    assert_eq!(scale(&format!("{e}.weight_packed")), Some(format!("{e}.weight_scale").as_str()));
    assert_eq!(scale(&format!("{e}.weight_scale")), Some(format!("{e}.weight_global_scale").as_str()));
    assert_eq!(q.shapes[&format!("{e}.weight_packed")], [16, 32]);
    for left in
        [l(2, "self_attn.o_proj.weight"), l(1, "block_sparse_moe.gate.weight"), l(0, "input_layernorm.weight"), "language_model.lm_head.weight".to_string()]
    {
        assert_eq!(scale(&left), None, "{left}");
    }
    assert_eq!((q.fp8_block, q.desc.blocks.len()), (None, 6));
    let quant_with = |key: &str, v: serde_json::Value| {
        let mut q = quant.clone();
        q[key] = v;
        serde_json::json!({"top": {"quantization_config": q}})
    };
    let sparse = |key: &str, v: serde_json::Value| serde_json::json!({"sparse": {key: v}});
    for (keys, word) in [
        (serde_json::json!({"use_qk_norm": false}), "use_qk_norm"),
        (serde_json::json!({"qk_norm_type": "per_layer"}), "qk_norm_type per_layer"),
        (serde_json::json!({"qk_norm_type": null}), "no qk_norm_type"),
        (serde_json::json!({"attention_output_gate": true}), "attention_output_gate"),
        (serde_json::json!({"num_key_value_heads": 3}), "not whole heads a key head"),
        (serde_json::json!({"partial_rotary_factor": 0.25}), "partial_rotary_factor = 0.25"),
        (serde_json::json!({"rotary_dim": 3}), "rotary_dim = 3"),
        (serde_json::json!({"rope_scaling": {"type": "yarn", "factor": 4.0}}), "rope_scaling"),
        (serde_json::json!({"hidden_act": "silu"}), "hidden_act"),
        (serde_json::json!({"swiglu_limit": 0.0}), "swiglu_limit = 0"),
        (serde_json::json!({"swiglu_alpha": null}), "no swiglu_alpha"),
        (serde_json::json!({"scoring_func": "softmax"}), "scoring_func"),
        (serde_json::json!({"shared_intermediate_size": 32}), "shared_intermediate_size = 32"),
        (serde_json::json!({"moe_layer_freq": [0, 1]}), "moe_layer_freq has 2 entries for 3 layers"),
        (serde_json::json!({"num_experts_per_tok": 7}), "num_experts_per_tok = 7"),
        (serde_json::json!({"dense_intermediate_size": null}), "no dense_intermediate_size"),
        (serde_json::json!({"intermediate_size": null}), "no intermediate_size"),
        (serde_json::json!({"num_mtp_modules": 1}), "config key num_mtp_modules is not understood"),
        (serde_json::json!({"sparse_attention_config": 4}), "sparse_attention_config = 4 is not an object"),
        (sparse("sparse_score_type", "lse".into()), "sparse_score_type lse"),
        (sparse("sparse_init_block", 1.into()), "sparse_init_block = 1"),
        (sparse("sparse_init_tokens", 4.into()), "config key sparse_init_tokens is not understood"),
        (sparse("sparse_num_index_heads", 4.into()), "sparse_num_index_heads = 4 for 2 key heads"),
        (sparse("sparse_disable_index_value", serde_json::json!([1, 1, 0])), "does not say 1 for the sparse layer 2"),
        (sparse("sparse_disable_index_value", serde_json::Value::Null), "does not say 1 for the sparse layer 0"),
        (sparse("sparse_local_block", 3.into()), "keeping 2 blocks of 2, 3 of them the query's own"),
        (sparse("sparse_index_dim", 2.into()), "under a rotary of 4"),
        (sparse("use_sparse_attention", false.into()), "use_sparse_attention"),
        (sparse("sparse_attention_freq", serde_json::json!([1, 1])), "sparse_attention_freq has 2 entries for 3 layers"),
        (sparse("training_stage", "dash".into()), "config key training_stage is not understood"),
        (serde_json::json!({"top": {"num_reward_heads": 2}}), "num_reward_heads = 2"),
        (serde_json::json!({"top": {"dspark_block_size": 7}}), "config key dspark_block_size is not understood"),
        (quant_with("quant_method", "fp8".into()), "where the description reads mxfp8"),
        (quant_with("weight_block_size", serde_json::json!([128, 128])), "weight_block_size [128,128] is not supported"),
        (quant_with("moe_quant_algo", "MXFP4".into()), "the experts' stored format is read as NVFP4"),
        (quant_with("moe_quant_format", serde_json::Value::Null), "the experts' stored format is read as NVFP4"),
        (quant_with("group_size", 32.into()), "group_size"),
        (quant_with("activation_scheme", "static".into()), "activation_scheme"),
        (quant_with("kv_cache_quant_algo", "FP8".into()), "config key kv_cache_quant_algo is not understood"),
    ] {
        let e = describe(&minimax_tiny(keys)).unwrap_err().msg;
        assert!(e.contains(word), "{word}: {e}");
    }
    // (a sparse layer's indexer with no block of its own always kept, and a layer said dense whose flag of the indexer's values is then nothing)
    let free = describe(&minimax_tiny(sparse("sparse_local_block", 0.into()))).unwrap();
    assert_eq!(attn(&free, 0).indexer.unwrap().form, IndexForm::Max { norm: pn, local: 0 });
    assert!(describe(&minimax_tiny(sparse("sparse_disable_index_value", serde_json::json!([1, 0, 1])))).is_ok());
    // (the wrapper's type alone chooses the builder: another wrapper whose text model names no type has none)
    let other = describe(&minimax_tiny(serde_json::json!({"top": {"model_type": "minimax_m2_vl"}}))).unwrap_err().msg;
    assert_eq!(other, "model_type minimax_m2_vl has no description builder yet");
}

/// The MiniMax M3 layout on the CPU reference, a random model of each small config. It is causal, reads the whole context, and steps as it prompts (the
/// indexer's keys are cached a token, its blocks are the positions'). While a query sees no more blocks than its indexer keeps, the sparse layers are the
/// attention without an indexer, to the bit; from the first position of the third block on they are not, at every row; and an indexer that keeps every
/// block is no indexer at any length. The SwiGLU's gain and limit are read, and it is not SiLU's gated pair: the up is shifted. The experts' sum is
/// scaled: at 2 it is, to the bit, the one of a model at 1 whose experts' second matrices are doubled, and not the one whose shared expert's is too.
#[test]
fn minimax_m3_on_the_reference_keeps_blocks_a_key_head_and_shifts_the_up() {
    use super::kinds::{Act, Body, Ffn, ModelDesc};
    // (a fixed draw a tensor: its name's bytes seed it; a norm's weights, which are added to 1, near 0)
    fn draw(n: &str, len: usize) -> Vec<f32> {
        let mut s = n.bytes().fold(0x9E37_79B9_7F4A_7C15u64, |a, b| (a ^ b as u64).wrapping_mul(0x0100_0000_01B3));
        let x = if n.contains("norm") { 0.2 } else { 0.6 };
        (0..len)
            .map(|_| {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                x * (((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5)
            })
            .collect()
    }
    // (`edit`: a change to the description before it is loaded; `w2_x`: a factor on every routed expert's second matrix; `shared_x`: on every shared expert's)
    let load = |cfg: &serde_json::Value, edit: &dyn Fn(&mut ModelDesc), w2_x: f32, shared_x: f32| -> RefModel {
        let mut d = describe(cfg).unwrap_or_else(|e| panic!("{}", e.msg));
        edit(&mut d.desc);
        let src = |n: &str| {
            let shape = d.shapes.get(n)?.clone();
            let x = match n {
                _ if n.ends_with(".w2.weight") => w2_x,
                _ if n.ends_with(".shared_experts.down_proj.weight") => shared_x,
                _ => 1.0,
            };
            Some((shape.clone(), draw(n, shape.iter().product()).into_iter().map(|v| v * x).collect()))
        };
        RefModel::load(&d, &src).unwrap_or_else(|e| panic!("{}", e.msg))
    };
    let same = |_: &mut ModelDesc| {};
    let run = |cfg: &serde_json::Value, toks: &[u32]| load(cfg, &same, 1.0, 1.0).forward(toks, 0).unwrap_or_else(|e| panic!("{}", e.msg));
    let a: [u32; 9] = [5, 9, 13, 17, 21, 25, 29, 33, 37];
    let base = minimax_tiny(serde_json::json!({}));
    let rows = run(&base, &a);
    assert_eq!((rows.len(), rows[8].len()), (9, 61));
    assert!(rows.iter().flatten().all(|x| x.is_finite()) && rows[8].iter().any(|&x| x != rows[8][0]), "finite, and not one value");
    // (causal, and the context is read: the middle layer attends to every token)
    let mut later = a;
    later[8] = 44;
    assert_eq!(rows[..8], run(&base, &later)[..8], "a later token changes no earlier row");
    let mut first = a;
    first[0] = 44;
    assert_ne!(rows[8], run(&base, &first)[8], "the last row reads the first token");
    // (a token at a time, and a prompt then its next tokens: the same bits)
    let mut m = load(&base, &same, 1.0, 1.0);
    let mut by_step = Vec::new();
    for (t, &tok) in a.iter().enumerate() {
        by_step.extend(m.forward(&[tok], t as u32).unwrap_or_else(|e| panic!("{}", e.msg)));
    }
    assert_eq!(by_step, rows, "a token at a time is the prompt");
    let mut m = load(&base, &same, 1.0, 1.0);
    let mut parts = m.forward(&a[..5], 0).unwrap();
    parts.extend(m.forward(&a[5..], 5).unwrap());
    assert_eq!(parts, rows, "a prompt, then its next tokens");
    // the indexer: 2 blocks of 2 positions kept
    let sparse = |key: &str, v: serde_json::Value| minimax_tiny(serde_json::json!({"sparse": {key: v}}));
    let full = run(&sparse("sparse_attention_freq", serde_json::json!([0, 0, 0])), &a);
    for t in 0..9 {
        assert_eq!(rows[t] == full[t], t < 4, "row {t}: two blocks hold the first four positions");
    }
    assert_eq!(run(&sparse("sparse_topk_blocks", 5.into()), &a), full, "an indexer that keeps every block is none");
    assert_ne!(run(&sparse("sparse_topk_blocks", 4.into()), &a)[8], full[8], "four of five blocks are not the context");
    assert_eq!(run(&sparse("sparse_topk_blocks", 4.into()), &a)[..8], full[..8], "four blocks hold the first eight positions");
    assert_ne!(run(&sparse("sparse_local_block", 2.into()), &a), rows, "the own block and the one before it are another choice than the own and the best");
    // the SwiGLU
    assert_ne!(rows[8], run(&minimax_tiny(serde_json::json!({"swiglu_alpha": 1.0})), &a)[8], "the sigmoid's gain is read");
    assert_ne!(rows[8], run(&minimax_tiny(serde_json::json!({"swiglu_limit": 0.05})), &a)[8], "a limit that bites is another answer");
    let silu = |d: &mut ModelDesc| {
        for b in d.blocks.iter_mut() {
            match &mut b.body {
                Body::Ffn(Ffn::Mlp(m)) => m.act = Act::Silu,
                Body::Ffn(Ffn::Moe(m)) => {
                    m.expert.act = Act::Silu;
                    m.shared = m.shared.map(|s| super::kinds::Mlp { act: Act::Silu, ..s });
                }
                _ => {}
            }
        }
    };
    let gain_one = minimax_tiny(serde_json::json!({"swiglu_alpha": 1.0}));
    assert_ne!(
        run(&gain_one, &a)[8],
        load(&gain_one, &silu, 1.0, 1.0).forward(&a, 0).unwrap()[8],
        "at a gain of 1 the gate is SiLU's and the up is still shifted"
    );
    // the experts' sum, scaled; the shared expert is not
    let unscaled = minimax_tiny(serde_json::json!({"routed_scaling_factor": 1.0}));
    assert_ne!(rows[8], run(&unscaled, &a)[8], "the experts' sum doubled is another answer");
    assert_eq!(rows, load(&unscaled, &same, 2.0, 1.0).forward(&a, 0).unwrap(), "the experts' sum doubled is their second matrices doubled");
    assert_ne!(rows[8], load(&unscaled, &same, 2.0, 2.0).forward(&a, 0).unwrap()[8], "the shared expert is added as it is");
}

/// Hy4 Flash's config as released (the top level with its text model's and its tower's keys), with `text` over the text model's keys and `top` over the top
/// level's. The lists a layer and the n-gram tables' sizes are made as the release has them: a layer of latent attention with the indexer first and on
/// every fourth after layer 1, sliding ones between; a dense feed-forward on the first layer; the tables' sizes the 64 primes past 10,312,000 in order,
/// 16 a layer of the memory.
fn hy4_config(text: serde_json::Value, top: serde_json::Value) -> serde_json::Value {
    let sparse = |l: usize| l == 0 || l % 4 == 1;
    let prime = |n: &u64| (2..).take_while(|i| i * i <= *n).all(|i| n % i != 0);
    let primes: Vec<u64> = (10_312_000u64..).filter(prime).take(64).collect();
    let tables: serde_json::Map<String, serde_json::Value> =
        [2, 12, 22, 32].iter().zip(primes.chunks(16)).map(|(l, p)| (l.to_string(), serde_json::json!([p[..8], p[8..]]))).collect();
    let mut t = serde_json::Map::new();
    for part in [
        serde_json::json!({
            "architectures": ["HYV4FlashForCausalLM"], "attention_bias": false, "attention_dropout": 0.0, "bitwise_backward_align": true, "bos_token_id": 120000,
            "compressed_indexer_candidate_topk": 2048, "compressed_indexer_compress_ratio": 4, "compressed_indexer_group_topk": 512,
            "compressed_indexer_head_dim": 128, "compressed_indexer_key_pooling": "gated", "compressed_indexer_num_attention_heads": 16,
            "compressed_indexer_reuse_fine_indexer": false, "compressed_indexer_skip_fine_select": true, "compressed_indexer_stage2_query_block_size": 8192,
            "compressed_indexer_teacher_pooling_type": "max", "compressed_indexer_triton_pooling_min_groups": 1, "compressed_indexer_use_triton_pooling": true,
            "dsa_skip_attn_post_norm": false, "enable_ihc": true, "enable_lm_head_fp32": true, "engram_embed_random_seed": 1234, "engram_fp8_embedding": false,
            "engram_isolate_fffd_namespace": false, "engram_layer_ids": [2, 12, 22, 32], "engram_max_ngram_size": 3, "engram_n_embed_per_ngram": 768,
            "engram_n_head_per_ngram": 8, "engram_pad_id": 120002, "engram_seed": 0, "engram_tokenizer_name_or_path": null, "engram_vocab_map_path": null,
            "engram_vocab_size": [10312000, 10312000], "eos_token_id": 120025, "fast_op_fp32_linear_gemm": true, "first_k_dense_replace": 1,
        }),
        serde_json::json!({
            "full_nope": false, "gate_groups": 1, "gate_lora_op": "none", "gate_lora_rank": null, "gated_mla": true, "gating_type": "elementwise", "hc_eps": 1e-06,
            "hc_magnitude": 2.0, "hc_mult": 4, "head_dim": 64, "hidden_act": "silu", "hidden_size": 3072, "ihc_kernel_backend": "naive", "index_head_dim": 128,
            "index_n_heads": 16, "index_topk": 2048, "initializer_range": 0.006, "intermediate_size": 13824, "kv_lora_rank": 512, "learnable_sink": true,
            "learnable_sink_init": 0.0, "max_position_embeddings": 1048576, "model_type": "hy_v4_flash", "moe_aux_loss_coeff": 0.0, "moe_intermediate_size": 1536,
            "moe_router_enable_expert_bias": true, "moe_router_use_sigmoid": true, "n_group": 1, "n_routed_experts": 512, "n_shared_experts": 1,
            "norm_topk_prob": true, "num_attention_heads": 32, "num_experts_per_tok": 16, "num_hidden_layers": 42, "num_key_value_heads": 32,
        }),
        serde_json::json!({
            "num_nextn_predict_layers": 0, "o_groups": 1, "o_lora_op": "none", "o_lora_rank": null, "output_router_logits": false, "pad_token_id": 120002,
            "q_lora_rank": 1024, "qk_head_dim": 256, "qk_nope_head_dim": 192, "qk_norm": true, "qk_rope_head_dim": 64, "return_bos_score": true,
            "return_max_logit": true, "rms_norm_eps": 1e-05, "routed_expert_hidden_size": 1536, "routed_scaling_factor": 3.998, "sandwich_norm": true,
            "sliding_window": 512, "swa_num_attention_heads": 64, "swa_qk_head_dim": 256, "swa_qk_nope_head_dim": 192, "swa_qk_rope_head_dim": 64,
            "swa_v_head_dim": 256, "swiglu_limit": 10.0, "tie_word_embeddings": false, "topk_group": 1, "use_cache": true, "use_compressed_indexer": true,
            "use_dsa": true, "use_grouped_mm": false, "use_mla": true, "v_head_dim": 256, "vocab_size": 120832, "window_size": [511, 0],
        }),
        serde_json::json!({
            "layer_types": (0..42).map(|l| if sparse(l) { "deepseek_sparse_attention" } else { "sliding_attention" }).collect::<Vec<_>>(),
            "swa_pattern": (0..42).map(|l| !sparse(l) as usize).collect::<Vec<_>>(),
            "mlp_layer_types": (0..42).map(|l| if l == 0 { "dense" } else { "sparse" }).collect::<Vec<_>>(),
            "indexer_types": vec!["full"; 42],
            "engram_layer_head_vocab_sizes": tables,
            "rope_parameters": {
                "sliding_attention": {"mrope_interleaved": true, "mrope_section": [12, 10, 10], "rope_theta": 10000, "rope_type": "default"},
                "deepseek_sparse_attention": {"mrope_interleaved": true, "mrope_section": [12, 10, 10], "rope_theta": 10000000, "rope_type": "default"},
            },
        }),
        text,
    ] {
        t.extend(part.as_object().unwrap().clone());
    }
    let vision = serde_json::json!({
        "add_patchemb_bias": true, "anyres_vit_max_image_size": 2048, "attention_backend": "hf_align_sdpa", "attention_dropout": 0.0, "bias_activation_fusion": true,
        "cat_extra_token": 0, "hidden_act": "gelu", "hidden_dropout": 0.0, "hidden_size": 1152, "img_max_token_num": 4096, "intermediate_size": 4304,
        "interpolate_mode": "bilinear", "learnable_mlp_pooling_size": 0, "max_image_size": 2048, "max_vit_seq_len": 16384, "min_image_size": 512,
        "model_type": "hy_v4_flash", "num_attention_heads": 16, "num_channels": 3, "num_hidden_layers": 27, "num_key_value_heads": 16, "out_hidden_size": 3072,
        "packed_seq_alignment": 128, "patch_size": 16, "perceive_post_norm": false, "perceive_pre_norm": true, "projector_type": "rmsnorm_mlp_adapter",
        "remove_prenorm": true, "resize_resolution": 2048, "rms_norm_eps": 1e-05, "rope_2d_percent": 1.0, "rope_2d_theta": 10000.0, "spatial_merge_size": 2,
        "spatial_patch_size": 1, "temporal_patch_size": 1, "text_hidden_size": 3072, "video_max_image_size": 768, "video_min_image_size": 256,
        "video_spatial_compression": false, "video_temporal_compression": false, "vision_full_attention": false,
    });
    let mut c = serde_json::json!({
        "architectures": ["HYV4FlashForConditionalGeneration"], "dtype": "bfloat16", "image_end_token_id": 120119, "image_newline_token_id": 120121,
        "image_start_token_id": 120118, "image_token_id": 120120, "model_type": "hy_v4_flash", "processor_class": "HYV4FlashProcessor", "spatial_merge_size": 2,
        "transformers_version": "5.23.0", "video_end_token_id": 120123, "video_start_token_id": 120122, "video_token_id": 120124, "vision_start_token_id": 120118,
        "dfly_block_size": 6, "dfly_mtp_fusion": "slot0", "dfly_target_index_topk": 2048, "dfly_context_local_window": null, "dfly_enable_hidden_correction": true,
        "dfly_num_hidden_layers": 5, "dfly_mask_token_id": 120009, "dfly_target_index_layer": [25, 29, 33, 37, 41], "dfly_target_kv_layers": [25, 29, 33, 37, 41],
    });
    c.as_object_mut().unwrap().extend(top.as_object().unwrap().clone());
    c["text_config"] = serde_json::Value::Object(t);
    c["vision_config"] = vision;
    c
}

/// Hy4 Flash's released config has a builder that reads every key of it, the text model's and the top level's, and says what it found: 12 layers of
/// latent attention with the compressed indexer and 30 in a window, a dense feed-forward on the first layer and experts on the rest, four residual
/// streams, the n-gram memory on four layers, a drafter of five. It is then refused: each part that no layer kind says is named with its tensors, and
/// no key is left unread.
#[test]
fn hy4_flashs_released_config_is_read_and_refused_at_the_parts_no_kind_says() {
    use super::describe::{keys_named, HY4_LACKS};
    let none = || serde_json::json!({});
    let c = hy4_config(none(), none());
    let keys = |v: &serde_json::Value| v.as_object().unwrap().len();
    assert_eq!((keys(&c), keys(&c["text_config"]), keys(&c["vision_config"])), (25, 110, 41), "every key of the release's config.json");
    let e = describe(&c).expect_err("refused").msg;
    let read =
        "hy_v4_flash: its config is read (12 layers of latent attention with a compressed indexer and 30 in a window of 512 tokens, 1 dense feed-forward \
                and 41 of 512 experts, 4 residual streams, an n-gram memory on 4 layers, a drafter of 5 layers), and it is not described yet.";
    assert!(e.starts_with(read), "{e}");
    assert_eq!(HY4_LACKS.len(), 6);
    for part in HY4_LACKS {
        assert!(e.contains(part), "{part}: {e}");
    }
    for tensor in [
        "learnable_sink_param",
        "q_b_proj",
        "compressed_indexer.token_key_projection",
        "key_gate_projection",
        "intra_group_ape",
        "head_weight_projection",
        "hc_pre.hc_fn",
        "lookup_table",
        "lookup_per_head_norms",
        "unified_per_hc_scales",
        "mtp_fusion.gate",
        "hidden_correction",
        "confidence_head",
    ] {
        assert!(e.contains(tensor), "{tensor}: {e}");
    }
    assert!(keys_named(&e).is_empty() && !e.contains("has no description builder yet"), "{e}");
    // (the counts are the config's: a model of other layers says its own; a layer's names as the readers fold them; one layer for every draft layer's picks)
    let small = hy4_config(
        serde_json::json!({
            "num_hidden_layers": 3, "layer_types": ["sparse", "sliding_attention", "indexed_attention"], "swa_pattern": [0, 1, 0], "indexer_types": null,
            "mlp_layer_types": ["dense", "dense", "sparse"], "first_k_dense_replace": 2, "n_routed_experts": 32, "sliding_window": 64, "window_size": [63, 0],
            "engram_layer_ids": [3], "engram_layer_head_vocab_sizes": {"3": vec![vec![10_312_007; 8]; 2]},
        }),
        serde_json::json!({"dfly_num_hidden_layers": 1, "dfly_target_index_layer": 2, "dfly_target_kv_layers": [2]}),
    );
    let e = describe(&small).expect_err("refused").msg;
    let read = "(2 layers of latent attention with a compressed indexer and 1 in a window of 64 tokens, 2 dense feed-forward and 1 of 32 experts, \
                4 residual streams, an n-gram memory on 1 layers, a drafter of 1 layers)";
    assert!(e.contains(read), "{e}");
    // (the text model alone, its keys at the top level, with no memory: no drafter's keys are asked for, nor the tables')
    let mut flat = c["text_config"].clone();
    for key in ["engram_layer_ids", "engram_layer_head_vocab_sizes", "engram_vocab_size"] {
        flat[key] = serde_json::Value::Null;
    }
    let e = describe(&flat).expect_err("refused").msg;
    assert!(e.contains("an n-gram memory on 0 layers, a drafter of 0 layers), and it is not described yet"), "{e}");
}

/// The Hy4 builder holds each key to what the family's readers compute: a setting of another form is refused by its key before the parts are, a key
/// nobody knows by its name (the top level's and the text model's together), and a list that is not the layers', a sort that is no layer's, a width
/// that is not the form's, a table or a drafter's layer that cannot be, in words.
#[test]
fn hy4_flash_holds_every_key_to_the_form_its_readers_compute() {
    use super::describe::keys_named;
    let refusal = |text: serde_json::Value, top: serde_json::Value| describe(&hy4_config(text, top)).expect_err("refused").msg;
    for (text, want) in [
        (serde_json::json!({"hidden_act": "gelu"}), "hidden_act = \"gelu\" is not supported (only \"silu\")"),
        (serde_json::json!({"gated_mla": false}), "gated_mla = false is not supported"),
        (serde_json::json!({"gating_type": "headwise"}), "gating_type = \"headwise\" is not supported (only \"elementwise\")"),
        (serde_json::json!({"learnable_sink": false}), "learnable_sink = false is not supported"),
        (serde_json::json!({"enable_ihc": false}), "enable_ihc = false is not supported"),
        (serde_json::json!({"use_compressed_indexer": false}), "use_compressed_indexer = false is not supported"),
        (serde_json::json!({"compressed_indexer_key_pooling": "mean"}), "compressed_indexer_key_pooling = \"mean\" is not supported (only \"gated\")"),
        (serde_json::json!({"compressed_indexer_skip_fine_select": false}), "compressed_indexer_skip_fine_select = false is not supported"),
        (serde_json::json!({"index_head_dim": 64}), "index_head_dim = 64 is not supported (only 128)"),
        (serde_json::json!({"o_lora_rank": 256}), "o_lora_rank = 256 is not supported (only null)"),
        (serde_json::json!({"qk_norm": false}), "qk_norm = false is not supported"),
        (serde_json::json!({"full_nope": true}), "full_nope = true is not supported"),
        (serde_json::json!({"use_mla": false}), "use_mla = false is not supported"),
        (serde_json::json!({"n_group": 2}), "n_group = 2 is not supported (only 1)"),
        (serde_json::json!({"moe_router_use_sigmoid": false}), "moe_router_use_sigmoid = false is not supported"),
        (serde_json::json!({"swa_qk_rope_head_dim": 32}), "swa_qk_rope_head_dim = 32 is not supported (only 64, qk_rope_head_dim"),
        (serde_json::json!({"qk_head_dim": 128}), "qk_head_dim = 128 is not supported (only 256, qk_nope_head_dim + qk_rope_head_dim)"),
        (serde_json::json!({"head_dim": 256}), "head_dim = 256 is not supported (only 64, qk_rope_head_dim"),
        (serde_json::json!({"num_key_value_heads": 1}), "num_key_value_heads = 1 is not supported (only 32, num_attention_heads"),
        (serde_json::json!({"layer_types": ["sparse", "sliding_attention"]}), "2 layer_types for 42 layers"),
        (serde_json::json!({"num_hidden_layers": 2, "layer_types": ["sparse", "full_attention"]}), "layer 1 is of type \"full_attention\" (only deepseek_sparse_attention"),
        (serde_json::json!({"swa_pattern": vec![0; 42]}), "swa_pattern is not layer_types'"),
        (serde_json::json!({"indexer_types": (0..42).map(|l| if l == 5 { "shared" } else { "full" }).collect::<Vec<_>>()}), "indexer_types is not \"full\" for each of the 42"),
        (serde_json::json!({"window_size": [512, 0]}), "window_size = [512,0] is not supported (only [511, 0]"),
        (
            serde_json::json!({"rope_parameters": {"rope_theta": 10000, "rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 4096}}),
            "rope_type yarn is not supported (only default)",
        ),
        (serde_json::json!({"rope_parameters": {"sliding_attention": {"rope_theta": 10000}}}), "rope_parameters has no rotary for the deepseek_sparse_attention layers"),
        (serde_json::json!({"rope_parameters": {"rope_theta": 10000, "mrope_section": [16, 8, 4]}}), "mrope_section [16, 8, 4] is not the 32 angles of qk_rope_head_dim"),
        (serde_json::json!({"rope_parameters": {"mrope_section": [12, 10, 10], "mrope_interleaved": false}}), "mrope_interleaved = false is not supported"),
        (serde_json::json!({"rope_parameters": {"sparse": {"rope_theta": 10000}, "sliding_attention": {}, "rope_theta": 10}}), "rope_parameters has rope_theta beside"),
        (serde_json::json!({"compressed_indexer_compress_ratio": 3}), "compressed_indexer_compress_ratio = 3 is not supported (only 2, 4, 8 or 16"),
        (
            serde_json::json!({"index_topk": 1024}),
            "index_topk = 1024 and compressed_indexer_candidate_topk = 2048, where both are compressed_indexer_group_topk x compressed_indexer_compress_ratio (2048)",
        ),
        (serde_json::json!({"hc_mult": 0}), "0 residual streams"),
        (serde_json::json!({"mlp_layer_types": ["dense"]}), "1 mlp_layer_types for 42 layers"),
        (serde_json::json!({"first_k_dense_replace": 3}), "first_k_dense_replace = 3, where mlp_layer_types has 1 dense layers"),
        (serde_json::json!({"num_experts_per_tok": 600}), "600 of 512 experts a token"),
        (serde_json::json!({"routed_scaling_factor": null}), "no routed_scaling_factor"),
        (serde_json::json!({"num_nextn_predict_layers": 1}), "num_nextn_predict_layers = 1 is not supported (only 0"),
        (serde_json::json!({"num_nextn_predict_layers": null}), "no num_nextn_predict_layers"),
        (serde_json::json!({"engram_layer_ids": [0, 12]}), "engram_layer_ids names layer 0: the layers numbered from 1, 1 to 42"),
        (serde_json::json!({"engram_layer_ids": [2, 43]}), "engram_layer_ids names layer 43"),
        (serde_json::json!({"engram_layer_ids": [2, 2]}), "names a layer twice"),
        (serde_json::json!({"engram_layer_ids": [3]}), "engram_layer_head_vocab_sizes has no tables for layer 3 (2 orders of 8 sizes)"),
        (serde_json::json!({"engram_vocab_size": [10312000, 10400000]}), "layer 2's table of 10312177 rows is not past engram_vocab_size (10400000)"),
        (serde_json::json!({"engram_n_head_per_ngram": 7}), "an n-gram memory of orders up to 3, 7 heads an order over 768 values"),
        (serde_json::json!({"engram_pad_id": 120832}), "engram_pad_id 120832 of 120832"),
    ] {
        let e = refusal(text.clone(), serde_json::json!({}));
        assert!(e.starts_with("hy_v4_flash") && e.contains(want), "{text}: {e}");
        assert!(!e.contains("its config is read"), "{text}: refused at the key, before the parts: {e}");
    }
    // (the drafter's keys, the top level's)
    for (top, want) in [
        (serde_json::json!({"dfly_mtp_fusion": "concat"}), "dfly_mtp_fusion = \"concat\" is not supported (only \"slot0\")"),
        (serde_json::json!({"dfly_enable_hidden_correction": false}), "dfly_enable_hidden_correction = false is not supported"),
        (serde_json::json!({"dfly_enable_ihc": true}), "dfly_enable_ihc = true is not supported"),
        (serde_json::json!({"dfly_block_size": null}), "a drafter without its dfly_block_size"),
        (serde_json::json!({"dfly_mask_token_id": 120832}), "or its dfly_mask_token_id (a token of the vocabulary)"),
        (serde_json::json!({"dfly_target_index_topk": 1024}), "dfly_target_index_topk = 1024 is not supported (only index_topk, 2048"),
        (serde_json::json!({"dfly_target_index_layer": [25]}), "dfly_target_index_layer is not a layer of the model for each of the 5 draft layers"),
        (
            serde_json::json!({"dfly_target_kv_layers": [25, 29, 33, 37, 40]}),
            "dfly_target_kv_layers names layer 40: a layer of latent attention with the indexer, 1 to 41",
        ),
        (serde_json::json!({"dfly_target_kv_layers": [25, 25, 33, 37, 41]}), "dfly_target_kv_layers names a layer twice"),
        (serde_json::json!({"dfly_attn_post_proj_norm": "rms"}), "dfly_attn_post_proj_norm is not true or false"),
    ] {
        let e = refusal(serde_json::json!({}), top.clone());
        assert!(e.starts_with("hy_v4_flash: ") && e.contains(want), "{top}: {e}");
        assert!(!e.contains("its config is read"), "{top}: refused at the key, before the parts: {e}");
    }
    // (a key no builder knows is named, the top level's and the text model's, with every other one, before the parts are)
    let e = refusal(serde_json::json!({"attn_merger_rank": 16}), serde_json::json!({"dfly_new_setting": 1, "quantization_config": {}}));
    assert_eq!(keys_named(&e), ["dfly_new_setting", "quantization_config", "attn_merger_rank"], "{e}");
}
