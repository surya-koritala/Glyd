//! GLY-362: where to cut a model's layers between two machines, from the model's own files and
//! each GPU's free memory. The engine's `prepare` is still the final check of a stage's fit.
use std::{collections::BTreeMap, fs, io::Read, path::Path};

/// What each part of a model needs on a GPU, in bytes.
#[derive(Debug, PartialEq)]
pub struct Shape {
    /// Each layer's weights, plus its keys and values for the context (full-attention layers) or
    /// its recurrent state (linear-attention layers), for every session.
    pub layers: Vec<u64>,
    /// The first stage's embedding; the last stage's head.
    pub first: u64,
    pub last: u64,
}

type Result<T> = std::result::Result<T, String>;

/// Room left on each GPU for the engine's activations and scratch.
// ponytail: a flat 1 GiB; the engine's prepare is the real check.
pub const RESERVE: u64 = 1 << 30;

/// A safetensors file's tensors and their byte sizes, from its header alone.
fn tensors(path: &Path) -> Result<Vec<(String, u64)>> {
    let mut file = fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut length = [0; 8];
    file.read_exact(&mut length).map_err(|e| e.to_string())?;
    let length = u64::from_le_bytes(length);
    if length > 64 << 20 {
        return Err(format!("{}: a {length}-byte header", path.display()));
    }
    let mut header = vec![0; length as usize];
    file.read_exact(&mut header).map_err(|e| e.to_string())?;
    let header: BTreeMap<String, serde_json::Value> =
        serde_json::from_slice(&header).map_err(|e| e.to_string())?;
    let payload = file
        .metadata()
        .map_err(|e| e.to_string())?
        .len()
        .checked_sub(length + 8)
        .ok_or("truncated safetensors file")?;
    header
        .into_iter()
        .filter(|(name, _)| name != "__metadata__")
        .map(|(name, t)| {
            let at = t["data_offsets"]
                .as_array()
                .filter(|a| a.len() == 2)
                .ok_or("invalid tensor offsets")?;
            let start = at[0].as_u64().ok_or("invalid tensor start")?;
            let end = at[1].as_u64().ok_or("invalid tensor end")?;
            if start > end || end > payload {
                return Err(format!("{name}: tensor outside safetensors data"));
            }
            Ok((name, end - start))
        })
        .collect()
}

fn sum(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b).ok_or("model size overflow".into())
}
fn product(values: &[u64]) -> Result<u64> {
    values.iter().try_fold(1u64, |n, v| {
        n.checked_mul(*v).ok_or("model size overflow".into())
    })
}

/// The model's shape for `context` tokens and `sessions` requests at once.
pub fn read(model: &Path, context: u32, sessions: u32) -> Result<Shape> {
    let config: serde_json::Value = serde_json::from_slice(
        &fs::read(model.join("config.json")).map_err(|e| format!("config.json: {e}"))?,
    )
    .map_err(|e| e.to_string())?;
    let text = config.get("text_config").unwrap_or(&config);
    let number = |key: &str| {
        text[key]
            .as_u64()
            .filter(|&n| n > 0)
            .ok_or(format!("config.json: no positive {key}"))
    };
    let count = number("num_hidden_layers")?;
    if !(2..=4096).contains(&count) || context == 0 || !(1..=2).contains(&sessions) {
        return Err("unsupported layer count, context or session count".into());
    }
    let count = count as usize;
    let mut layers = vec![0u64; count];
    let (mut first, mut last) = (0, 0);
    let mut files: Vec<_> = fs::read_dir(model)
        .map_err(|e| e.to_string())?
        .collect::<std::io::Result<Vec<_>>>()
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    files.sort();
    for file in &files {
        for (name, bytes) in tensors(file)? {
            // The text model's layers; vision and draft (MTP) towers are not a stage's.
            if name.contains("visual") || name.contains("vision") || name.contains("mtp") {
                continue;
            }
            if let Some(rest) = name.split("layers.").nth(1)
                && let Some(i) = rest.split('.').next().and_then(|i| i.parse::<usize>().ok())
            {
                let layer = layers
                    .get_mut(i)
                    .ok_or(format!("{name}: past the config's layers"))?;
                *layer = sum(*layer, bytes)?;
            } else if name.contains("embed_tokens") {
                first = sum(first, bytes)?;
            } else if name.contains("lm_head") {
                last = sum(last, bytes)?;
            }
        }
    }
    if first == 0 || layers.contains(&0) {
        return Err("missing embedding or layer weights".into());
    }
    // Tied output embeddings occupy the head stage too.
    if last == 0 {
        if text["tie_word_embeddings"] == true || config["tie_word_embeddings"] == true {
            last = first;
        } else {
            return Err("missing output head weights".into());
        }
    }
    // Runtime state per layer: bf16 keys and values of the full-attention layers for the
    // context, the linear-attention layers' fp32 delta and bf16 conv states.
    let kinds = text["layer_types"].as_array();
    let kv = product(&[
        4,
        number("num_key_value_heads")?,
        number("head_dim")?,
        u64::from(context),
    ])?;
    let state = match (
        number("linear_num_value_heads"),
        number("linear_num_key_heads"),
        number("linear_key_head_dim"),
        number("linear_value_head_dim"),
    ) {
        (Ok(nv), Ok(nk), Ok(dk), Ok(dv)) => sum(
            product(&[nv, dk, dv, 4])?,
            product(&[6, sum(product(&[2, nk, dk])?, product(&[nv, dv])?)?])?,
        )?,
        _ => 0,
    };
    for (i, layer) in layers.iter_mut().enumerate() {
        let full =
            kinds.is_none_or(|k| k.get(i).and_then(|k| k.as_str()) != Some("linear_attention"));
        if !full && state == 0 {
            return Err("missing linear attention dimensions".into());
        }
        *layer = sum(
            *layer,
            product(&[u64::from(sessions), if full { kv } else { state }])?,
        )?;
    }
    // Every stage reservation (at most two sessions) must remain representable.
    let total = layers
        .iter()
        .try_fold(sum(first, last)?, |n, &v| sum(n, v))?;
    product(&[2, sum(total, RESERVE)?])?;
    Ok(Shape {
        layers,
        first,
        last,
    })
}

/// The cut (layers 0..cut on the first GPU, cut..all on the second) that fits both, each with
/// `reserve` bytes to spare, using their memory most evenly. None if no cut fits.
pub fn cut(shape: &Shape, free: [u64; 2], reserve: u64) -> Option<u32> {
    let total = shape
        .layers
        .iter()
        .try_fold(0u64, |n, &v| n.checked_add(v))?;
    let mut head = 0u64;
    let mut best: Option<(f64, u32)> = None;
    for c in 1..shape.layers.len() {
        head = head.checked_add(shape.layers[c - 1])?;
        let need = [
            shape.first.checked_add(head)?.checked_add(reserve)?,
            shape.last.checked_add(total - head)?.checked_add(reserve)?,
        ];
        if need[0] > free[0] || need[1] > free[1] {
            continue;
        }
        let worst = (need[0] as f64 / free[0] as f64).max(need[1] as f64 / free[1] as f64);
        if best.is_none_or(|(w, _)| worst < w) {
            best = Some((worst, c as u32));
        }
    }
    best.map(|(_, c)| c)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn malformed_headers_and_overflow_cannot_understate_memory() {
        let file = tempfile::NamedTempFile::new().unwrap();
        for offsets in [
            serde_json::json!([4, 2]),
            serde_json::json!([0, 9]),
            serde_json::json!([0]),
            serde_json::json!([0, "8"]),
        ] {
            let header = serde_json::to_vec(&serde_json::json!({"model.layers.0.a":
                {"data_offsets": offsets}}))
            .unwrap();
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend(header);
            bytes.extend([0; 8]);
            fs::write(file.path(), bytes).unwrap();
            assert!(tensors(file.path()).is_err());
        }
        assert!(product(&[u64::MAX, 2]).is_err());
        assert!(sum(u64::MAX, 1).is_err());
        assert_eq!(
            cut(
                &Shape {
                    layers: vec![u64::MAX, 1],
                    first: 1,
                    last: 1
                },
                [u64::MAX; 2],
                RESERVE
            ),
            None
        );
        let model = tempfile::tempdir().unwrap();
        fs::write(
            model.path().join("config.json"),
            br#"{"num_hidden_layers":18446744073709551615}"#,
        )
        .unwrap();
        assert!(read(model.path(), 4096, 1).is_err());
    }
    fn safetensors(path: &Path, tensors: &[(&str, u64)]) {
        let mut at = 0;
        let mut header = serde_json::Map::new();
        for (name, bytes) in tensors {
            header.insert(
                name.to_string(),
                serde_json::json!({"dtype": "U8", "shape": [bytes], "data_offsets": [at, at + bytes]}),
            );
            at += bytes;
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.extend(vec![0; at as usize]);
        fs::write(path, file).unwrap();
    }
    #[test]
    fn reads_layers_and_cuts_where_both_fit_most_evenly() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("config.json"),
            serde_json::to_vec(&serde_json::json!({"text_config": {"num_hidden_layers": 4,
                "num_key_value_heads": 1, "head_dim": 2, "layer_types": ["linear_attention",
                "full_attention", "linear_attention", "full_attention"],
                "linear_num_value_heads": 1, "linear_num_key_heads": 1, "linear_key_head_dim": 1,
                "linear_value_head_dim": 1}}))
            .unwrap(),
        )
        .unwrap();
        safetensors(
            &dir.path().join("model-1.safetensors"),
            &[
                ("model.embed_tokens.weight", 50),
                ("model.layers.0.a", 100),
                ("model.layers.1.a", 100),
            ],
        );
        safetensors(
            &dir.path().join("model-2.safetensors"),
            &[
                ("model.layers.2.a", 100),
                ("model.layers.3.a", 100),
                ("lm_head.weight", 50),
                ("model.visual.blocks.0.a", 999),
                ("mtp.layers.0.a", 999),
            ],
        );
        // context 10, one session: kv = 2*1*2*2*10 = 80 per full layer; state = 4 + 3*3*2 = 22.
        let shape = read(dir.path(), 10, 1).unwrap();
        assert_eq!(
            shape,
            Shape {
                layers: vec![122, 180, 122, 180],
                first: 50,
                last: 50
            }
        );
        // Equal GPUs cut in the middle; a bigger first GPU takes more layers.
        assert_eq!(cut(&shape, [400, 400], 0), Some(2));
        assert_eq!(cut(&shape, [600, 300], 0), Some(3));
        assert_eq!(cut(&shape, [100, 100], 0), None);
        // The reserve counts on both.
        assert_eq!(cut(&shape, [400, 400], 100), None);
        // Two sessions double the runtime state.
        assert_eq!(
            read(dir.path(), 10, 2).unwrap().layers,
            vec![144, 260, 144, 260]
        );
        // A tied head needs its own embedding allocation on the second stage.
        safetensors(
            &dir.path().join("model-2.safetensors"),
            &[("model.layers.2.a", 100), ("model.layers.3.a", 100)],
        );
        assert!(
            read(dir.path(), 10, 1).is_err(),
            "an untied missing head is incomplete"
        );
        let config_path = dir.path().join("config.json");
        let mut config: serde_json::Value =
            serde_json::from_slice(&fs::read(&config_path).unwrap()).unwrap();
        config["text_config"]["tie_word_embeddings"] = true.into();
        fs::write(&config_path, serde_json::to_vec(&config).unwrap()).unwrap();
        let tied = read(dir.path(), 10, 1).unwrap();
        assert_eq!((tied.first, tied.last), (50, 50));
    }
}
