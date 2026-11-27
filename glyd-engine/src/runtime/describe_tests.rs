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
            Source::Rows(_) => panic!("{role}"),
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

/// Every config of tests/fixtures describes from its own file, the toy and tiny ones as the released (Condor's Qwen3.8-Flash-Next too: 48 layers of two blocks, and
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

/// The tiny Qwen3.8-Flash-Next of tests/fixtures/qwen4x_tiny_manifest.json (the tensors transformers saved for that config: Condor's random model of the `qwen4x`
/// logits case is made from it): its config describes exactly those tensors, each at its saved shape.
#[test]
fn condors_tiny_config_describes_its_tensors() {
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

/// Condor's released config (tests/fixtures/qwen4x_config.json) describes every tensor Condor's engine loads (src/qwen4x.rs: each layer's matrices, the gated
/// residuals' and the final mixer's, the n-gram embedding's products; the MTP layer's, its front's and its final mixer's as the draft head) and the vision
/// tower as the release's own engine read it (src/vision.rs: 27 blocks of 16 heads over 1,152, 4,304 neurons, a 48 x 48 position table, patches of 3 x 2 x
/// 16 x 16 merged 2 x 2 into rows of 2,560, image token 248,056), nothing left; and what it does not take is refused in words.
#[test]
fn condors_released_config() {
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

/// Condor's model from its config.json alone, on the CPU reference of the layer kinds, against Condor's own engine: the random tiny Qwen3.8-Flash-Next of the `qwen4x`
/// logits case (src/logits.rs `qwen4x_random`) and the rows its engine recorded on an RTX 4080 SUPER at kestrel (tests/fixtures/qwen4x_condor_rows.json.gz, that engine's
/// record: no logits case reads it, as the family is held to this reference and not to that engine's bits), where the random
/// model's experts are on the six-bit grid kestrel holds them on and the engine's other matrices are lossless: the engine runs this model's weights, its activations
/// bf16. The two implementations share no code. The prompt of 24 and the 12 decode steps (the prompt of 150 takes 50 s more here, and measured KL 4.0e-5): KL mean 8.5e-6, max 5.0e-5, the
/// best token 6 of 6 where it leads clearly, the largest logit difference 0.097 (measured 2026-11-23). It counts: with the indexer keeping every token the KL mean is
/// 1.8e-4 and the best token 4 of 6; with the n-gram hash's seed off by one, 1.1e-2 and 0 of 6.
#[cfg(feature = "dev")]
#[test]
fn condors_model_on_the_reference_against_condors_engine() {
    use super::kinds::{Body, Mixer};
    use super::reference::model::checkpoint_tensor;
    use crate::logits::{compare, Bound, Fixture};
    let dir = std::env::temp_dir().join(format!("glyd-qwen4x-ref-{}", std::process::id()));
    crate::logits::qwen4x_random(&dir).unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let d = describe(&cfg).unwrap_or_else(|e| panic!("{}", e.msg));
    let ck = glyd_gpu::safetensors::Checkpoint::open(&dir).unwrap();
    let mut fx = Fixture::read(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4x_condor_rows.json.gz")).unwrap();
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
    assert_eq!((m.experts, m.router.top_k, m.router.logit_scale, m.router.scale, m.expert.clamp), (384, 6, 1.0, 1.5, Some(10.0)));
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
                Source::Tensor(x) => vec![x.clone()],
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
