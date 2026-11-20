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
    let d = describe(&f.j["config"]).unwrap_or_else(|e| panic!("{case}: {}", e.0));
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
        let d = describe(c).unwrap_or_else(|e| panic!("{repo}: {}", e.0));
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
    let e = describe(&c).unwrap_err().0;
    assert!(e.contains("attn_logit_softcap"), "{e}");
    let mut g = fixture("gemma4_12b_config.json");
    g["text_config"]["enable_moe_block"] = true.into();
    assert!(describe(&g).unwrap_err().0.contains("enable_moe_block"));
    let mut g = fixture("gemma4_12b_config.json");
    g["text_config"]["model_type"] = "something_new".into();
    assert!(describe(&g).unwrap_err().0.contains("something_new"));
}
