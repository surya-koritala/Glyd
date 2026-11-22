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
    let d = describe(&f.j["config"]).unwrap_or_else(|e| panic!("{case}: {}", e.msg));
    let want: BTreeSet<String> = f.j["tensors"].as_object().unwrap().keys().cloned().collect();
    assert_eq!(d.names(), want, "{case}: the weight map");
    let src = |n: &str| f.j["tensors"].get(n).map(|_| f.tensor(n));
    let mut m = RefModel::load(&d, &src).unwrap();
    let toks: Vec<u32> = f.j["token_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let got: Vec<f32> = m.forward(&toks, 0).unwrap().concat();
    assert_close(&got, &f.f32s("output"), tol, case);
}

const MODEL_TOL: f32 = 1e-5;

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
    assert_eq!((d.desc.head.tied, d.desc.head.softcap), (true, Some(30.0)));
    assert!(d.head.is_none());
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
    for (f, unread, mtp) in [
        ("gemma-4-12B-it", &["model.embed_audio.", "model.embed_vision.", "model.vision_embedder.", "model.vision_tower.", "model.audio_tower."][..], 0),
        ("Kimi-K3", &["vision_tower.", "mm_projector."][..], 0),
        ("DeepSeek-V4-Flash-0731", &["mtp."][..], 1),
        ("Qwen3.8-27B", &["model.visual.", "mtp."][..], 1),
    ] {
        let h = fixture(&format!("headers/{f}.json"));
        let cfg: serde_json::Value = serde_json::from_str(h["config"].as_str().unwrap()).unwrap();
        let d = describe(&cfg).unwrap_or_else(|e| panic!("{f}: {}", e.msg));
        let (all, got) = (header_names(&h), d.names());
        let extra: Vec<_> = got.difference(&all).take(6).collect();
        assert!(extra.is_empty(), "{f}: described but not in the checkpoint: {extra:?}");
        let missing: Vec<_> = all.difference(&got).filter(|n| !unread.iter().any(|p| n.starts_with(p))).take(6).collect();
        assert!(missing.is_empty(), "{f}: in the checkpoint but not described: {missing:?}");
        assert_eq!(d.mtp_layers, mtp, "{f}");
    }
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
