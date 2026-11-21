//! Llama 3.x as a description (F1a, #166): the released configs of Llama 3.2 1B and 3B, 3.1 8B and 3.3 70B (config files alone, from a public mirror; no checkpoint) describe with no per-model code, each at
//! the layers, heads, rope scaling and tying its release has, and the weight maps of the 8B and the 70B are exactly the tensors of the Hub's safetensors index. Adding the next standard family (Mistral, Qwen 2, Phi-4,
//! Granite, OLMo, SmolLM) is this: a config and its index, and a test like this one.

use glyd_engine::runtime::describe::describe;
use glyd_engine::runtime::kinds::*;
use std::collections::BTreeSet;
use std::path::PathBuf;

fn fixture(name: &str) -> serde_json::Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/llama").join(name);
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn index_names(name: &str) -> BTreeSet<String> {
    fixture(name)["tensors"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect()
}

/// (layers, heads, kv heads, head dim, tied, factor of the rope scaling) of each release.
const RELEASES: [(&str, usize, usize, usize, usize, bool, f64); 4] = [
    ("meta-llama/Llama-3.2-1B-Instruct", 16, 32, 8, 64, true, 32.0),
    ("meta-llama/Llama-3.2-3B-Instruct", 28, 24, 8, 128, true, 32.0),
    ("meta-llama/Llama-3.1-8B-Instruct", 32, 32, 8, 128, false, 8.0),
    ("meta-llama/Llama-3.3-70B-Instruct", 80, 64, 8, 128, false, 8.0),
];

#[test]
fn every_released_llama_config_describes() {
    let cfgs = fixture("configs.json");
    for (repo, layers, heads, kv, hd, tied, factor) in RELEASES {
        let d = describe(&cfgs[repo]).unwrap_or_else(|e| panic!("{repo}: {}", e.0));
        // a decoder layer is two blocks: the attention's, then the gated MLP's
        assert_eq!(d.desc.blocks.len(), 2 * layers, "{repo}");
        for (i, b) in d.desc.blocks.iter().enumerate() {
            match (&b.body, i % 2) {
                (Body::Mixer(Mixer::Attention(a)), 0) => {
                    assert_eq!((a.heads, a.kv_heads, a.head_dim), (heads, kv, hd), "{repo} layer {}", i / 2);
                    assert!(
                        a.window.is_none() && a.q_norm.is_none() && a.k_norm.is_none() && !a.bias && a.gate.is_none(),
                        "{repo}: Llama's attention is plain"
                    );
                    match a.rope.as_ref().map(|r| &r.scaling) {
                        Some(RopeScaling::Llama3 { factor: f, low_freq_factor, high_freq_factor, original_max }) => {
                            assert_eq!((*f, *low_freq_factor, *high_freq_factor, *original_max), (factor, 1.0, 4.0, 8192), "{repo}");
                        }
                        other => panic!("{repo}: rope scaling {other:?}"),
                    }
                }
                (Body::Ffn(Ffn::Mlp(_)), 1) => {}
                (body, _) => panic!("{repo} block {i}: {body:?}"),
            }
        }
        assert_eq!((d.desc.vocab, d.desc.head.tied, d.head.is_none()), (128256, tied, tied), "{repo}");
        assert_eq!((d.mtp_layers, d.desc.embed.streams), (0, 1), "{repo}");
    }
}

#[test]
fn the_weight_maps_are_the_hubs_safetensors_index() {
    let cfgs = fixture("configs.json");
    for (repo, index) in [("meta-llama/Llama-3.1-8B-Instruct", "llama31_8b_index.json"), ("meta-llama/Llama-3.3-70B-Instruct", "llama33_70b_index.json")] {
        let want = index_names(index);
        let got = describe(&cfgs[repo]).unwrap().names();
        let missing: Vec<_> = want.difference(&got).take(8).collect();
        let extra: Vec<_> = got.difference(&want).take(8).collect();
        assert!(missing.is_empty() && extra.is_empty(), "{repo}: not described: {missing:?}; not in the checkpoint: {extra:?}");
    }
}

/// The tied releases have no `lm_head`: their head is the embedding.
#[test]
fn the_tied_releases_read_the_embedding_as_the_head() {
    let cfgs = fixture("configs.json");
    for repo in ["meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"] {
        let d = describe(&cfgs[repo]).unwrap();
        assert!(d.head.is_none() && !d.names().contains("lm_head.weight") && d.names().contains(&d.embed), "{repo}");
    }
}

/// A key the description does not know is an error that names it (a model with a feature this description lacks is not run as if it had none).
#[test]
fn an_unknown_key_is_refused_by_name() {
    let mut c = fixture("configs.json")["meta-llama/Llama-3.1-8B-Instruct"].clone();
    c["sliding_window_pattern"] = serde_json::json!(6);
    let e = describe(&c).unwrap_err();
    assert!(e.0.contains("sliding_window_pattern"), "{}", e.0);
}
