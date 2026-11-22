//! `glyd-engine pack MODEL OUT --layout hummingbird --codes DIR`: Nemotron 3 Super at the Hummingbird level (Linear GLY-98, GLY-134), a glyd-v7 directory (layout `hb`) whose routed experts are the neuron rows of a calibration
//! export (research/lossy-2026/nemotron_hbr.py's `--export DIR`, glyd-gpu's hbexport.rs) packed as they are in c3r or c2r by the bits the export gives each expert, with the signs of each layer's rotation beside them, and every other
//! matrix (Mamba-2's, attention's, the latent projections', the shared experts', the embedding and the head) six-bit as at wren, from the bf16 checkpoint MODEL. `toy_export` makes a random export for a tiny model
//! (`glyd-engine hbtoy`), for the tests and the pod job that run the layer through the expert cache against all its experts on the GPU.

use crate::ffi::{err, Error, Result};
use crate::nemotron::{ConfigNemotronH, NhKind};
use crate::pack::plan_puts;
use glyd_gpu::codebook::{cb_strides, pack_codebook_rows};
use glyd_gpu::hbexport::HbExport;
use glyd_gpu::safetensors::Checkpoint;
use glyd_gpu::save::{self, Layout, Source};
use std::path::Path;
use std::time::Instant;

fn io<T>(r: std::io::Result<T>) -> Result<T> {
    r.map_err(|e| Error(e.to_string()))
}

/// The save plan of the model `c` at the Hummingbird level: its matrices but the routed experts', each six-bit (the linear says so: `"layout": "i6"`), its small tensors copied as the checkpoint has them (`f32` says which the
/// checkpoint holds as F32: cast to bf16, as the saver holds nothing else), and for each mixture-of-experts layer (the export's layer k is the model's k-th) its experts' two matrices from the export (`up` and `downT`)
/// and the signs of its rotation.
pub fn plan(c: &ConfigNemotronH, f32: &dyn Fn(&str) -> bool) -> String {
    let dense: Vec<_> = c.packs().into_iter().filter(|p| !p.key.contains(".mixer.experts.")).collect();
    let mut puts = plan_puts(&dense, &c.kept(), f32, &|_| Some(Layout::Int6));
    for (l, &(_, k)) in c.schedule().iter().enumerate().filter(|(_, (kind, _))| *kind == NhKind::Moe) {
        for e in 0..c.moe.experts {
            for (name, matrix) in [("up_proj", "up"), ("down_proj", "downT")] {
                puts.push(format!(
                    "{{\"codebook\": \"{}layers.{l}.mixer.experts.{e}.{name}\", \"layer\": {k}, \"expert\": {e}, \"matrix\": \"{matrix}\"}}",
                    c.prefix
                ));
            }
        }
        puts.push(format!("{{\"signs\": \"{}layers.{l}.mixer.glyd_rot_signs\", \"layer\": {k}}}", c.prefix));
    }
    format!("{{\"puts\": [{}]}}", puts.join(", "))
}

/// The export held to the model: the shape of the experts' rows, their number, the layers (the export's k-th is the model's k-th mixture-of-experts layer, by its block).
pub fn check(c: &ConfigNemotronH, ex: &HbExport) -> Result<()> {
    let m = &c.moe;
    if (ex.rows, ex.cols, ex.experts) != (m.inter, m.latent, m.experts) {
        return err(format!(
            "{}: {} experts of {} neuron rows of {} values, where the model has {} of {} of {}",
            ex.dir.display(),
            ex.experts,
            ex.rows,
            ex.cols,
            m.experts,
            m.inter,
            m.latent
        ));
    }
    let blocks: Vec<usize> = (0..c.layers).filter(|&l| c.kinds[l] == NhKind::Moe).collect();
    if ex.layers.len() != blocks.len() {
        return err(format!("{}: {} mixture-of-experts layers, where the model has {}", ex.dir.display(), ex.layers.len(), blocks.len()));
    }
    for (k, (l, b)) in ex.layers.iter().zip(&blocks).enumerate() {
        if l.block != *b {
            return err(format!(
                "{}: layer {k} of the export is block {}, where the model's {k}-th mixture-of-experts layer is block {b}",
                ex.dir.display(),
                l.block
            ));
        }
    }
    Ok(())
}

/// `model` (a Nemotron 3 Super checkpoint in bf16) packed at the Hummingbird level into `out` from the export in `export`, on `threads` threads: the whole directory is written (a directory already holding a pack is left as it is).
pub fn run(model: &str, out: &Path, threads: usize, export: &Path) -> Result<()> {
    let t0 = Instant::now();
    let source = io(Source::find(model))?;
    let c = ConfigNemotronH::load(&source.dir)?;
    let ex = io(HbExport::open(export))?;
    check(&c, &ex)?;
    if out.join("glyd.json").exists() {
        eprintln!("{}: already packed, keeping its files", out.display());
        return Ok(());
    }
    let ck = io(Checkpoint::open(&source.dir))?;
    let f32 = |n: &str| ck.get(n).is_some_and(|(_, t)| t.dtype == "F32");
    let (twos, threes) = ex.bit_counts();
    let s = io(save::save_hummingbird(&source, out, threads, 5 << 30, &plan(&c, &f32), export))?;
    eprintln!(
        "glyd-v7 at the hummingbird level: {} packs checked ({} of them the export's, {twos} experts at 2 bits and {threes} at 3), {} tensors hashed, {:.2} GB of bf16 read, {} shards ({:.1} s)",
        s.checked,
        s.calibrated,
        s.hashed,
        s.bytes as f64 / 1e9,
        s.shards,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

/// A random export for the model `c` (a tiny one: the files hold every matrix in memory) in `out`, as nemotron_hbr.py writes one: each layer's experts at 3 bits for `share` percent of them (spread over the layer) and at 2 for the
/// rest, random codes, scales of both signs and ascending codebooks in units of the scale, signs of 1 and 0, from `seed`. The matrices are the export's own; nothing is a fit of the model's weights.
pub fn toy_export(c: &ConfigNemotronH, out: &Path, seed: u64, share: usize) -> Result<()> {
    let (m, blocks) = (&c.moe, (0..c.layers).filter(|&l| c.kinds[l] == NhKind::Moe).collect::<Vec<_>>());
    if m.latent % 32 != 0 || share > 100 {
        return err(format!("a latent of {} and {share} percent of the experts at 3 bits", m.latent));
    }
    io(std::fs::create_dir_all(out))?;
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = move || {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    let uniform = |v: u64| (v >> 40) as f32 / (1u64 << 24) as f32;
    let bf = |v: f32| ((v.to_bits() + 0x7FFF + ((v.to_bits() >> 16) & 1)) >> 16) as u16;
    let mut recs = Vec::new();
    for (k, &block) in blocks.iter().enumerate() {
        // (3 bits for the experts where the running share of `share` percent steps up: that many of a layer's, spread over it)
        let bits: Vec<u32> = (0..m.experts).map(|e| if (e + 1) * share / 100 > e * share / 100 { 3 } else { 2 }).collect();
        let mut owned: Vec<(String, &str, Vec<u64>, Vec<u8>)> = Vec::new();
        for (e, &b) in bits.iter().enumerate() {
            for matrix in ["up", "downT"] {
                let (rows, cols) = (m.inter, m.latent);
                let codes: Vec<u8> = (0..rows * cols).map(|_| (next() % (1 << b)) as u8).collect();
                let scales: Vec<u16> = (0..rows * cols / 32).map(|_| bf((uniform(next()) - 0.15) * 0.1)).collect();
                let mut book: Vec<u16> = (0..rows << b).map(|_| bf(uniform(next()) * 2.0 - 1.0)).collect();
                book.chunks_mut(1 << b).for_each(|r| r.sort_by(|p, q| f32::from_bits((*p as u32) << 16).total_cmp(&f32::from_bits((*q as u32) << 16))));
                let p = pack_codebook_rows(&codes, &scales, &book, b, rows, cols).ok_or_else(|| Error("a toy matrix that does not pack".into()))?;
                let (cs, _, _) = cb_strides(b, cols);
                owned.push((format!("e{e:03}.{matrix}.codes"), "U8", vec![rows as u64, cs as u64], p.codes));
                owned.push((format!("e{e:03}.{matrix}.scales"), "BF16", vec![rows as u64, cols as u64 / 32], p.scales));
                owned.push((format!("e{e:03}.{matrix}.codebook"), "BF16", vec![rows as u64, 1 << b], p.codebook));
            }
        }
        let name = format!("layer-{k:03}.safetensors");
        let tensors: Vec<(&str, &str, Vec<u64>, &[u8])> = owned.iter().map(|(n, d, s, b)| (n.as_str(), *d, s.clone(), &b[..])).collect();
        crate::pack::write_safetensors(&out.join(&name), &[("format", "pt".to_string())], &tensors)?;
        let signs: Vec<String> = (0..m.latent).map(|_| (next() & 1).to_string()).collect();
        let size = io(std::fs::metadata(out.join(&name)))?.len();
        recs.push(format!(
            "{{\"moe_layer\":{k},\"block\":{block},\"seed\":{},\"signs_u8\":[{}],\"expert_bits\":[{}],\"file\":\"{name}\",\"bytes\":{size},\"sha256\":\"\"}}",
            k * 8 + 2,
            signs.join(","),
            bits.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
        ));
    }
    let meta = format!(
        "{{\"format\":\"{}\",\"recipe\":\"toy\",\"complete\":true,\"model\":{{\"repo\":null,\"revision\":null}},\"rows\":{},\"row_length\":{},\"block\":32,\"experts\":{},\"moe_layers\":{},\"layers\":[{}]}}",
        glyd_gpu::hbexport::FORMAT,
        m.inter,
        m.latent,
        m.experts,
        blocks.len(),
        recs.join(",")
    );
    io(std::fs::write(out.join("meta.json"), meta))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::moe_hb::read_layer;
    use crate::nemotron::tests::checkpoint;
    use crate::weights::{v7_manifest, Layout as L};
    use glyd_gpu::hbexport::Matrix;

    /// The tiny model `nemotron_ref.py moetoy` builds (16 experts of 128 neurons over a latent of 128; blocks 1, 3 and 5 are mixture-of-experts layers).
    const MOETOY: &str = include_str!("../tests/fixtures/nemotron_h_moetoy_config.json");

    /// The tiny model packed at the Hummingbird level from a random export (`share` percent of the experts at three bits) into `dir/out`, in shards of about `shard_bytes` (a small number splits an expert's tensors between two
    /// shards, as a shard of 5 GB does now and then): for the tests of other modules that read a directory. Returns the directory and the config.
    pub(crate) fn packed_toy(dir: &Path, shard_bytes: u64, share: usize) -> (std::path::PathBuf, ConfigNemotronH) {
        let _ = std::fs::remove_dir_all(dir);
        let c = ConfigNemotronH::parse(MOETOY, "moetoy", None).unwrap();
        let (src, export, out) = (dir.join("src"), dir.join("export"), dir.join("out"));
        checkpoint(&src, MOETOY, &c);
        toy_export(&c, &export, 11, share).unwrap();
        let source = save::Source::find(src.to_str().unwrap()).unwrap();
        let ck = Checkpoint::open(&source.dir).unwrap();
        let f32 = |n: &str| ck.get(n).is_some_and(|(_, t)| t.dtype == "F32");
        save::save_hummingbird(&source, &out, 2, shard_bytes, &plan(&c, &f32), &export).unwrap();
        (out, c)
    }

    /// The tiny model packed at the Hummingbird level from a random export (`toy_export`): a glyd-v7 directory at layout `hb` that glyd-gpu's verify decodes to the sha256 of every pack, whose tensors the engine's check takes (the
    /// codebooks and the signs among them), whose layers read back to the export's matrices bit for bit in two groups by bits with each expert's group and place, whose dense matrices are six-bit; and what is refused: an export of
    /// other shape, other layers, an incomplete one, a directory already packed is kept.
    #[test]
    fn a_tiny_model_is_packed_at_the_hummingbird_level_and_read_back() {
        let dir = std::env::temp_dir().join(format!("glyd-engine-hbpack-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let c = ConfigNemotronH::parse(MOETOY, "moetoy", None).unwrap();
        let (src, export, out) = (dir.join("src"), dir.join("export"), dir.join("out"));
        checkpoint(&src, MOETOY, &c);
        toy_export(&c, &export, 7, 40).unwrap();
        let ex = HbExport::open(&export).unwrap();
        check(&c, &ex).unwrap();
        let (twos, threes) = ex.bit_counts();
        assert!(twos > 0 && threes > 0 && twos + threes == 3 * 16, "{twos} + {threes}");
        run(src.to_str().unwrap(), &out, 2, &export).unwrap();
        // the directory
        let (layout, manifest) = v7_manifest(&out).unwrap().unwrap();
        assert_eq!(layout, L::Hummingbird);
        assert_eq!((manifest.get("level").and_then(|v| v.as_str()), manifest.get("format").and_then(|v| v.as_str())), (Some("hummingbird"), Some("glyd-v7")));
        let v = save::verify(&out, 2, &save::Decoder::Cpu).unwrap();
        assert_eq!(v.unchecked, 0);
        let ck = Checkpoint::open(&out).unwrap();
        let have: Vec<(String, String, Vec<u64>)> =
            ck.names().map(|n| (n.to_string(), ck.get(n).unwrap().1.dtype.clone(), ck.get(n).unwrap().1.shape.clone())).collect();
        c.check(&have, &c.saved(&ck)).unwrap();
        // the layers read back
        for (k, &l) in [1usize, 3, 5].iter().enumerate() {
            let h = read_layer(&ck, &manifest, &c, l).unwrap();
            assert_eq!(h.signs, ex.layers[k].signs);
            let counts: Vec<usize> = h.groups.iter().map(|g| g.as_ref().map_or(0, |g| g.experts)).collect();
            assert_eq!(counts.iter().sum::<usize>(), 16);
            let mut next = [0usize; 2];
            for e in 0..16 {
                let b = ex.layers[k].expert_bits[e];
                let g = b as usize - 2;
                assert_eq!((h.group_of[e], h.local_of[e] as usize), (g as u8, next[g]), "layer {l} expert {e}");
                let host = h.groups[g].as_ref().unwrap();
                let (up, down) = (ex.matrix(k, e, Matrix::Up).unwrap(), ex.matrix(k, e, Matrix::DownT).unwrap());
                let p = host.place;
                let (cs, ss, bs) = cb_strides(b, 128);
                let (d, m) = (&host.data[next[g] * p.data..(next[g] + 1) * p.data], &host.meta[next[g] * p.meta..(next[g] + 1) * p.meta]);
                assert_eq!(&d[..128 * cs], &up.codes[..]);
                assert_eq!(&d[p.down_codes..p.down_codes + 128 * cs], &down.codes[..]);
                assert_eq!(&m[..128 * ss], &up.scales[..]);
                assert_eq!(&m[p.up_book..p.up_book + 128 * bs], &up.codebook[..]);
                assert_eq!(&m[p.down_scales..p.down_scales + 128 * ss], &down.scales[..]);
                assert_eq!(&m[p.down_book..p.down_book + 128 * bs], &down.codebook[..]);
                next[g] += 1;
            }
        }
        // a dense matrix of a layer is six-bit, as at wren
        let fc1 = c.layer_packs(1).into_iter().next().unwrap();
        let p = crate::weights::read_pack(&ck, Some(&(layout, manifest.clone())), &fc1, crate::nemotron::pack_layout(layout, &fc1.key)).unwrap();
        assert!(matches!(p.words, crate::ffi::Words::Tiered { tiers, .. } if tiers[0] >> 24 == glyd_gpu::pack::INT6_TAG));
        // an old reader of the directory: the engine's generic reader refuses an expert's pack by its layout
        let e = crate::weights::read_pack(&ck, Some(&(L::Int6, manifest.clone())), &c.expert_packs(1)[0], L::Int6).err().expect("refused").msg;
        assert!(e.contains("c3r") || e.contains("c2r"), "{e}");
        // the plan of `run` for this directory is sized by its own bytes at the level: a GPU with a megabyte more than the engine's 2 GiB of work has no room for the tiny model (the level is named in the refusal), one with 64 holds it whole
        let gib = 1u64 << 30;
        let said = crate::cli::expert_cache_for(&out, L::Hummingbird, 2 * gib + (1 << 20), "a GPU", Some(64 * gib)).unwrap_err().to_string();
        assert!(said.contains("at the hummingbird level takes about"), "{said}");
        assert!(crate::cli::expert_cache_for(&out, L::Hummingbird, 2 * gib + (64 << 20), "a GPU", Some(64 * gib)).unwrap().is_none());
        // a directory already packed is left as it is
        let before = std::fs::metadata(out.join("glyd.json")).unwrap().modified().unwrap();
        run(src.to_str().unwrap(), &out, 2, &export).unwrap();
        assert_eq!(std::fs::metadata(out.join("glyd.json")).unwrap().modified().unwrap(), before);
        // refused: an export of another shape, of fewer layers, of other blocks, an incomplete one
        let says = |what: &str, r: Result<()>| {
            let e = r.unwrap_err().msg;
            assert!(e.contains(what), "{what:?} not in {e:?}");
        };
        let other = ConfigNemotronH { moe: crate::nemotron::MoeDims { experts: 8, ..c.moe.clone() }, ..c.clone() };
        says("16 experts of 128 neuron rows of 128 values, where the model has 8", check(&other, &ex));
        let short = ConfigNemotronH { kinds: vec![NhKind::Mamba, NhKind::Moe, NhKind::Attention, NhKind::Moe, NhKind::Mamba, NhKind::Mamba], ..c.clone() };
        says("3 mixture-of-experts layers, where the model has 2", check(&short, &ex));
        let shifted = ConfigNemotronH { kinds: vec![NhKind::Moe, NhKind::Mamba, NhKind::Attention, NhKind::Moe, NhKind::Mamba, NhKind::Moe], ..c.clone() };
        says("layer 0 of the export is block 1, where the model's 0-th mixture-of-experts layer is block 0", check(&shifted, &ex));
        let meta = std::fs::read_to_string(export.join("meta.json")).unwrap();
        std::fs::write(export.join("meta.json"), meta.replace("\"complete\":true", "\"complete\":false")).unwrap();
        says("not complete", run(src.to_str().unwrap(), &dir.join("out2"), 2, &export));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
