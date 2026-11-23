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
        // (and each tensor's shape as the config makes it is the checkpoint's)
        assert_eq!(d.shapes.keys().cloned().collect::<BTreeSet<_>>(), want, "{repo}: a shape for every tensor");
        for (n, sh) in &d.shapes {
            let got: Vec<usize> = tensors[repo][n][1].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
            assert_eq!(&got, sh, "{repo}: {n}");
        }
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

/// The naming conventions held to every checkpoint index the fixtures have (the Hub's headers above; Qwen3.5 0.8B and 9B, GLM-5.3-Flash and Nemotron 3
/// Super's tensor lists): each names exactly the checkpoint's tensors, its quantized matrices' scales with them, less the parts it says it leaves. And the
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
    cases.push(("Nemotron 3 Super".into(), fixture("nemotron_h_config.json"), names_with_runs(&fixture("nemotron_h_tensors.json"))));
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

/// A loader's check (`describe_dir`) reads a Hub directory's index: a tensor the description reads is named where the index lacks it, a matrix is there in
/// either stored form, and a directory Glyd packed (its glyd.json) is not held to it.
#[test]
fn a_loader_names_a_missing_tensor_from_the_index() {
    let dir = std::env::temp_dir().join(format!("glyd-describe-dir-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let cfg = Fixture::load("llama_model").j["config"].clone();
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

/// Every config of tests/fixtures describes from its own file, the toy and tiny ones as the released: all but Condor's (Qwen3.8-Flash-Next, `qwen4_exp_text`), whose
/// indexer, n-gram embedding and gated residuals are kinds of R3d (#140), and which is refused in words meanwhile.
#[test]
fn every_fixture_config_describes() {
    for (f, blocks) in [
        ("gemma4_12b_config.json", 96),
        ("glm5_next_config.json", 90),
        ("glm5_next_tiny_config.json", 10),
        ("glm5_next_toy_config.json", 16),
        ("nemotron_h_config.json", 88),
        ("nemotron_h_moetoy_config.json", 6),
    ] {
        let d = describe(&fixture(f)).unwrap_or_else(|e| panic!("{f}: {}", e.msg));
        assert_eq!(d.desc.blocks.len(), blocks, "{f}");
    }
    for (repo, c) in fixture("qwen35_configs.json").as_object().unwrap() {
        describe(c).unwrap_or_else(|e| panic!("{repo}: {}", e.msg));
    }
    let e = describe(&fixture("qwen4x_config.json")).unwrap_err().msg;
    assert!(e.contains("qwen4_exp_text"), "{e}");
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
    // (every config but Condor's describes: its kinds are R3d's, #140)
    let refused: Vec<_> = now.iter().filter(|(_, e)| e.get("refused").is_some()).map(|(n, _)| n.as_str()).collect();
    assert_eq!(refused, ["qwen4x_config.json"]);
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
