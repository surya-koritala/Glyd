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
    Ok(header
        .into_iter()
        .filter(|(name, _)| name != "__metadata__")
        .filter_map(|(name, t)| {
            let at = t["data_offsets"].as_array()?;
            Some((
                name,
                at.get(1)?.as_u64()?.checked_sub(at.first()?.as_u64()?)?,
            ))
        })
        .collect())
}

/// The model's shape for `context` tokens and `sessions` requests at once.
pub fn read(model: &Path, context: u32, sessions: u32) -> Result<Shape> {
    let config: serde_json::Value = serde_json::from_slice(
        &fs::read(model.join("config.json")).map_err(|e| format!("config.json: {e}"))?,
    )
    .map_err(|e| e.to_string())?;
    let text = config.get("text_config").unwrap_or(&config);
    let number = |key: &str| text[key].as_u64().ok_or(format!("config.json: no {key}"));
    let count = number("num_hidden_layers")? as usize;
    let mut layers = vec![0u64; count];
    let (mut first, mut last) = (0, 0);
    let mut files: Vec<_> = fs::read_dir(model)
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok().map(|e| e.path()))
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
                *layers
                    .get_mut(i)
                    .ok_or(format!("{name}: past the config's layers"))? += bytes;
            } else if name.contains("embed_tokens") {
                first += bytes;
            } else if name.contains("lm_head") {
                last += bytes;
            }
        }
    }
    // Runtime state per layer: bf16 keys and values of the full-attention layers for the
    // context, the linear-attention layers' fp32 delta and bf16 conv states.
    let kinds = text["layer_types"].as_array();
    let kv = 2 * number("num_key_value_heads")? * number("head_dim")? * 2 * u64::from(context);
    let state = match (
        number("linear_num_value_heads"),
        number("linear_num_key_heads"),
        number("linear_key_head_dim"),
        number("linear_value_head_dim"),
    ) {
        (Ok(nv), Ok(nk), Ok(dk), Ok(dv)) => nv * dk * dv * 4 + 3 * (2 * nk * dk + nv * dv) * 2,
        _ => 0,
    };
    for (i, layer) in layers.iter_mut().enumerate() {
        let full =
            kinds.is_none_or(|k| k.get(i).and_then(|k| k.as_str()) != Some("linear_attention"));
        *layer += u64::from(sessions) * if full { kv } else { state };
    }
    Ok(Shape {
        layers,
        first,
        last,
    })
}

/// The cut (layers 0..cut on the first GPU, cut..all on the second) that fits both, each with
/// `reserve` bytes to spare, using their memory most evenly. None if no cut fits.
pub fn cut(shape: &Shape, free: [u64; 2], reserve: u64) -> Option<u32> {
    let total: u64 = shape.layers.iter().sum();
    let mut head = 0;
    let mut best: Option<(f64, u32)> = None;
    for c in 1..shape.layers.len() {
        head += shape.layers[c - 1];
        let need = [
            shape.first + head + reserve,
            shape.last + total - head + reserve,
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
    }
}
