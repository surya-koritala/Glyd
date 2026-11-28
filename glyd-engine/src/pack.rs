//! `glyd-engine pack MODEL OUT`: the checkpoint packed by `glyd-gpu`'s save: a glyd-v7 directory for a lossy level and for penguin of Gemma 4 and the plain dense kind, each
//! matrix packed, the embedding too; otherwise glyd-v3 (the 12-bit layout) or glyd-v4 (each matrix in the smaller of the two layouts) with the embedding packed beside it in
//! `engine.safetensors`, so that every matrix the engine holds is compressed. A glyd-v7 save's plan is made from the model's description for every family (`plan`: each
//! block's products by its kind, the tensors it keeps by their roles), as the support table's row of its family says the family is saved (`support::Saves`).

use crate::dynamic::ModelDesc;
use crate::ffi::{err, Error, Result};
use crate::qwen35::{Pack35, Tensor35};
use crate::runtime::config::ModelConfig;
use crate::runtime::describe::{Described, Source as Of};
use crate::support::{Family, Saves};
use glyd_gpu::pack::{pack_smallest_checked, pack_twelve_checked, Packed};
use glyd_gpu::safetensors::Checkpoint;
use glyd_gpu::save::{self, Layout, Source};
use std::path::Path;
use std::time::Instant;

fn io<T>(r: std::io::Result<T>) -> Result<T> {
    r.map_err(|e| Error(e.to_string()))
}

/// A safetensors file of the named U8 / I32 tensors (bytes as given), with `metadata` pairs.
pub fn write_safetensors(path: &Path, metadata: &[(&str, String)], tensors: &[(&str, &str, Vec<u64>, &[u8])]) -> Result<()> {
    use std::io::Write;
    let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
    let mut h = String::from("{");
    if !metadata.is_empty() {
        h.push_str("\"__metadata__\":{");
        h.push_str(&metadata.iter().map(|(k, v)| format!("\"{}\":\"{}\"", esc(k), esc(v))).collect::<Vec<_>>().join(","));
        h.push_str("},");
    }
    let mut at = 0u64;
    let mut parts = Vec::new();
    for (name, dtype, shape, bytes) in tensors {
        let end = at + bytes.len() as u64;
        parts.push(format!(
            "\"{}\":{{\"dtype\":\"{dtype}\",\"shape\":[{}],\"data_offsets\":[{at},{end}]}}",
            esc(name),
            shape.iter().map(u64::to_string).collect::<Vec<_>>().join(",")
        ));
        at = end;
    }
    h.push_str(&parts.join(","));
    h.push('}');
    while (8 + h.len()) % 8 != 0 {
        h.push(' ');
    }
    let tmp = path.with_extension("tmp");
    let mut f = io(std::fs::File::create(&tmp))?;
    io(f.write_all(&(h.len() as u64).to_le_bytes()))?;
    io(f.write_all(h.as_bytes()))?;
    for (_, _, _, bytes) in tensors {
        io(f.write_all(bytes))?;
    }
    io(f.sync_all())?;
    io(std::fs::rename(&tmp, path))
}

/// Each matrix a checkpoint holds as rows of a tensor of experts stacked: its name, that tensor's and its rows there (`describe::stacked_rows`).
type Rows = std::collections::BTreeMap<String, (String, usize, usize)>;

/// What a level's save holds of a model the executor loads by its layer kinds, in the order the save puts it (`walk`'s, every pack before every tensor):
/// the lossy route's sizer's (`plan::sizing::lossy_plan_inner`, through `dynamic::qwen35::saved`), as the plan `plan` makes of the same description. A
/// tensor kept is F32 where the loader holds it in fp32 (`lowered`'s small tensors that `dynamic::small_f32` says, and a choice bias).
pub(crate) fn described(d: &Described, lowered: &ModelDesc) -> Result<(Vec<Pack35>, Vec<Tensor35>)> {
    let small: std::collections::HashMap<&str, usize> = lowered.small.iter().enumerate().flat_map(|(i, s)| s.split('|').map(move |n| (n, i))).collect();
    let f32 = |n: &str, role: &str| small.get(n).map_or(role == "choice_bias", |&i| crate::dynamic::small_f32(lowered, i));
    let groups = walk(d, lowered.layers.len(), &|n: &str| d.shapes.get(n).cloned(), false, &f32)?;
    let packs = groups.iter().flat_map(|g| g.packs.iter().map(|(p, _)| p.clone())).collect();
    Ok((packs, groups.into_iter().flat_map(|g| g.kept).collect()))
}

/// What a save holds of one layer of the model (or, the last, of the model's own tensors: the embedding, the head, the final norm): its packs, each with
/// what its layout is chosen by (`Own`), and its tensors kept as saved, in the order the save puts them.
pub(crate) struct Group {
    pub packs: Vec<(Pack35, Own)>,
    pub kept: Vec<Tensor35>,
}

/// What a pack's layout at a level is chosen by (`Own::layout`): a routed expert's matrix (`moe`: its block's number among the model's blocks of experts;
/// `expert`: its number there; `down`: its second matrix; `relu2`: the experts' activation is relu squared), or any other matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Own {
    Expert { moe: usize, expert: usize, down: bool, relu2: bool },
    Rest,
}

impl Own {
    /// The pack's layout at `level` where it is not the level's own: a routed expert's second matrix whose activation is relu squared in the level's row
    /// layout (92% of its inputs are zero after relu squared: src/moe_rows.rs reads its rows for the ones that are not; as `dynamic::experts` packs it; a
    /// development build's GLYD_ENGINE_DENSE_DOWN keeps the block layout); every other matrix but the routed experts' in `form.rest_at_wren` at wren and
    /// hummingbird where the family's row says one.
    fn layout(self, level: Layout, form: &Saves) -> Option<Layout> {
        let own = match self {
            Own::Expert { down: true, relu2: true, .. } if crate::tune!("GLYD_ENGINE_DENSE_DOWN").is_none() => level.row_of().unwrap_or(level),
            Own::Expert { .. } => level,
            Own::Rest => form.rest_at_wren.filter(|_| matches!(level, Layout::Int4 | Layout::Hummingbird)).unwrap_or(level),
        };
        (own != level).then_some(own)
    }
}

/// What a save of the model `d` (of `layers` layers, its blocks as many a layer) holds, layer by layer and then the model's own tensors (`Group`), from its
/// description alone. The matrices: each block's products (`kinds::Body::products`: q, k and v one pack; gate and up), at the place its weight map names
/// the first of them (a product whose roles are cut from one stored matrix's rows is that matrix), and a block of experts' routed experts, expert by expert, each expert's matrices as its roles name them (its gate and up one pack
/// where its roles stack them); after the last layer the embedding and the head. Every other tensor a block's roles name is kept as saved: a layer's
/// blocks' norms (`norm_in`, `norm_out`), then the residual streams' mixing tensors (`hc_*`), then each block's other tensors in its roles' order and the
/// parts of its matrices the library does not pack (`Pack35::packable`), each after its own block's tensors or, `parts_last`, after all of them; the
/// model's own: the final norm, its other roles, then the parts of the embedding and the head the library does not pack. Refused in words: a block of a
/// kind with no products of this form, a product's bias, a routed expert's matrix the library does not pack (a block level's experts are read packed), a
/// block of experts the kernels do not run (`dynamic::experts::cfg`). `shape`: each tensor's [rows, cols, ...]; `f32`: whether the loader holds a
/// tensor (of a role) in fp32, the dtype a tensor kept is given (the sizer's).
pub(crate) fn walk(
    d: &Described,
    layers: usize,
    shape: &dyn Fn(&str) -> Option<Vec<usize>>,
    parts_last: bool,
    f32: &dyn Fn(&str, &str) -> bool,
) -> Result<Vec<Group>> {
    use crate::runtime::kinds::{Act, Body, Ffn};
    let nb = d.desc.blocks.len();
    if layers == 0 || !nb.is_multiple_of(layers) || d.blocks.len() != nb {
        return err(format!("{nb} blocks ({} with roles) over {layers} layers", d.blocks.len()));
    }
    let per = nb / layers;
    let dims = |n: &str| shape(n).ok_or_else(|| Error(format!("the description has no shape for {n}")));
    let pack = |members: &[String]| -> Result<Pack35> {
        let cols = *dims(&members[0])?.get(1).ok_or_else(|| Error(format!("{}: not a matrix", members[0])))?;
        let parts = members.iter().map(|m| Ok((m.clone(), dims(m)?[0]))).collect::<Result<Vec<_>>>()?;
        Ok(Pack35 { key: Pack35::saved(&members[0]).to_string(), parts, cols })
    };
    let parts_of =
        |p: &Pack35| p.parts.iter().map(|(n, r)| Tensor35 { name: n.clone(), dtype: "BF16", shape: vec![*r as u64, p.cols as u64] }).collect::<Vec<_>>();
    let kept_of = |n: &str, role: &str| -> Result<Tensor35> {
        Ok(Tensor35 { name: n.to_string(), dtype: if f32(n, role) { "F32" } else { "BF16" }, shape: dims(n)?.iter().map(|&x| x as u64).collect() })
    };
    let (mut groups, mut moes) = (Vec::with_capacity(layers + 1), 0);
    for l in 0..layers {
        let (mut packs, mut norms, mut streams, mut own, mut parts) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for b in l * per..(l + 1) * per {
            let (blk, roles) = (&d.desc.blocks[b], &d.blocks[b]);
            let products = blk.body.products().ok_or_else(|| Error(format!("block {b}: {:?}, a kind no save of this packer holds", blk.body)))?;
            let moe = match &blk.body {
                Body::Ffn(Ffn::Moe(m)) => {
                    crate::dynamic::experts::cfg(m, d.desc.hidden, 0).map_err(|e| e.context(&format!("block {b}")))?;
                    Some(m)
                }
                _ => None,
            };
            let names = |r: &str| roles.iter().find(|(x, _)| x == r).map(|(_, s)| s.names());
            let product = |r: &str| products.iter().position(|p| p.contains(&r));
            let (mut bk, mut bp, mut made, mut experts_done) = (Vec::new(), Vec::new(), vec![false; products.len()], false);
            for (r, src) in roles {
                if let (Some(m), true) = (moe, r.starts_with("experts_")) {
                    if std::mem::replace(&mut experts_done, true) {
                        continue;
                    }
                    // (every routed expert's matrices, expert by expert, each role's in its order)
                    let each = roles
                        .iter()
                        .filter(|(r, _)| r.starts_with("experts_"))
                        .map(|(r, s)| match s {
                            Of::Experts(v) if v.len() == m.experts => Ok((r.as_str(), v)),
                            other => err(format!(
                                "block {b}: the experts' role {r} is {other:?}, where {} experts' tensors are, a tensor (or its rows) an expert",
                                m.experts
                            )),
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let relu2 = m.expert.act == Act::Relu2;
                    for e in 0..m.experts {
                        for (r, v) in &each {
                            let p = pack(&v[e])?;
                            if !p.packable() {
                                return err(format!("{}: {} rows of {} columns, which the library does not pack", p.key, p.rows(), p.cols));
                            }
                            packs.push((p, Own::Expert { moe: moes, expert: e, down: *r == "experts_down", relu2 }));
                        }
                    }
                    moes += 1;
                    continue;
                }
                match product(r) {
                    // (a product at its first role's place: every role of it the block has, stacked)
                    Some(k) if !std::mem::replace(&mut made[k], true) => {
                        // (roles cut from one stored matrix's rows, MiMo's fused q, k and v: that matrix once, as the lowering's product is)
                        let mut members: Vec<String> = products[k].iter().filter_map(|x| names(x)).flatten().collect();
                        members.dedup();
                        match pack(&members)? {
                            p if p.packable() => packs.push((p, Own::Rest)),
                            p => bp.extend(parts_of(&p)),
                        }
                    }
                    Some(_) => {}
                    None if r.strip_suffix("_bias").is_some_and(|x| product(x).is_some()) => {
                        return err(format!("block {b}: {r}, a product's bias, which no save of this packer holds"));
                    }
                    None => {
                        for n in src.names() {
                            let t = kept_of(&n, r)?;
                            match r.as_str() {
                                "norm_in" | "norm_out" => norms.push(t),
                                x if x.starts_with("hc_") => streams.push(t),
                                _ => bk.push(t),
                            }
                        }
                    }
                }
            }
            own.push(bk);
            parts.push(bp);
        }
        let mut kept = norms;
        kept.append(&mut streams);
        if parts_last {
            kept.extend(own.into_iter().flatten().chain(parts.into_iter().flatten()));
        } else {
            kept.extend(own.into_iter().zip(parts).flat_map(|(o, p)| o.into_iter().chain(p)));
        }
        groups.push(Group { packs, kept });
    }
    // (the model's own: the embedding and the head where it is not the embedding, the final norm, its other roles)
    let (mut packs, mut kept, mut parts) = (Vec::new(), vec![kept_of(&d.final_norm, "final_norm")?], Vec::new());
    for (key, n) in std::iter::once(("embed", &d.embed)).chain(d.head.as_ref().map(|h| ("lm_head", h))) {
        match pack(std::slice::from_ref(n))? {
            p if p.packable() => packs.push((Pack35 { key: key.into(), ..p }, Own::Rest)),
            p => parts.extend(parts_of(&p)),
        }
    }
    for (r, src) in &d.extra {
        kept.extend(src.names().iter().map(|n| kept_of(n, r)).collect::<Result<Vec<_>>>()?);
    }
    kept.append(&mut parts);
    groups.push(Group { packs, kept });
    Ok(groups)
}

/// The save plan (glyd-gpu's `--plan` JSON) of the model of config `c` at `level`, saved as the family's row says (`form`), from its description alone
/// (`walk`): one linear for each pack, its members under their own names and read from the checkpoint's (or, where `held` holds a layer's experts stacked
/// in one tensor, from their rows there: `describe::stacked_rows`; the save is then the one of the same weights held a tensor an expert's matrix), each in
/// its layout where that is not the level's (`Own::layout`); one copy for each tensor kept, cast to bf16 where the checkpoint holds it as F32 (`f32`) but
/// where `form` keeps F32; every pack before every tensor, or each layer's then the next's (`form.by_layer`). `shape`: each tensor's as the checkpoint
/// holds it, where the description gives none.
pub(crate) fn plan(
    c: &ModelConfig,
    level: Layout,
    form: &Saves,
    held: Option<&Described>,
    f32: &dyn Fn(&str) -> bool,
    shape: &dyn Fn(&str) -> Option<Vec<usize>>,
) -> Result<String> {
    planned(c, level, form, held, f32, shape, false)
}

/// The save plan at the hummingbird level from a calibration export of the routed experts (glyd-gpu's `save_hummingbird`, src/hbpack.rs): `plan`'s at
/// hummingbird of every matrix but the routed experts' (each in its own layout, the row's: `Own::layout`) and of the tensors kept, then each block of
/// experts' matrices from the export, expert by expert (`{"codebook": PACK, "layer": K, "expert": E, "matrix": "up" | "downT"}`, K the block's number
/// among the model's blocks of experts, `downT` the export's of the second matrix), and the signs of its rotation (`MODULE.glyd_rot_signs`, MODULE the
/// experts' module path).
#[cfg(feature = "dev")]
pub(crate) fn hb_plan(c: &ModelConfig, form: &Saves, f32: &dyn Fn(&str) -> bool, shape: &dyn Fn(&str) -> Option<Vec<usize>>) -> Result<String> {
    planned(c, Layout::Hummingbird, form, None, f32, shape, true)
}

/// `plan`, or (`export`) `hb_plan`.
fn planned(
    c: &ModelConfig,
    level: Layout,
    form: &Saves,
    held: Option<&Described>,
    f32: &dyn Fn(&str) -> bool,
    shape: &dyn Fn(&str) -> Option<Vec<usize>>,
    export: bool,
) -> Result<String> {
    let d = crate::runtime::describe::describe_config(c)?;
    let rows = match held {
        Some(h) => crate::runtime::describe::stacked_rows(&d, h)?,
        None => Rows::new(),
    };
    let layers = c.num("num_hidden_layers").ok_or_else(|| Error("config.json: no num_hidden_layers".into()))?;
    let groups = walk(&d, layers, &|n: &str| d.shapes.get(n).cloned().or_else(|| shape(n)), form.parts_last, &|_, _| false)?;
    let q = |n: &str| format!("\"{n}\"");
    let from = |n: &str| match rows.get(n) {
        Some((t, a, b)) => format!("{{\"tensor\": {}, \"rows\": [{a}, {b}]}}", q(t)),
        None => q(Pack35::saved(n)),
    };
    let linear = |(p, own): &(Pack35, Own)| {
        let names = p.parts.iter().map(|x| q(Pack35::saved(&x.0))).collect::<Vec<_>>().join(", ");
        let froms = p.parts.iter().map(|x| from(&x.0)).collect::<Vec<_>>().join(", ");
        let layout = own.layout(level, form).map_or(String::new(), |l| format!(", \"layout\": {}", q(l.name())));
        format!("{{\"linear\": [{names}], \"from\": [{froms}]{layout}}}")
    };
    let copy = |t: &Tensor35| format!("{{\"copy\": {}{}}}", q(&t.name), if f32(&t.name) && !form.f32_kept { ", \"cast\": \"BF16\"" } else { "" });
    // (from an export, the routed experts are its: after everything else, each block's experts then its rotation's signs)
    let packs = |g: &Group| g.packs.iter().filter(|(_, own)| !export || *own == Own::Rest).map(linear).collect::<Vec<_>>();
    let mut puts: Vec<String> = if form.by_layer {
        groups.iter().flat_map(|g| packs(g).into_iter().chain(g.kept.iter().map(copy))).collect()
    } else {
        groups.iter().flat_map(packs).chain(groups.iter().flat_map(|g| g.kept.iter().map(copy))).collect()
    };
    if export {
        let experts: Vec<(&Pack35, usize, usize, bool)> = groups
            .iter()
            .flat_map(|g| g.packs.iter())
            .filter_map(|(p, own)| match *own {
                Own::Expert { moe, expert, down, .. } => Some((p, moe, expert, down)),
                Own::Rest => None,
            })
            .collect();
        for (i, &(p, k, e, down)) in experts.iter().enumerate() {
            let matrix = if down { "downT" } else { "up" };
            puts.push(format!("{{\"codebook\": {}, \"layer\": {k}, \"expert\": {e}, \"matrix\": \"{matrix}\"}}", q(&p.key)));
            if experts.get(i + 1).is_none_or(|x| x.1 != k) {
                let module = p.key.split(".experts.").next().unwrap_or(&p.key);
                puts.push(format!("{{\"signs\": {}, \"layer\": {k}}}", q(&format!("{module}.glyd_rot_signs"))));
            }
        }
    }
    Ok(format!("{{\"puts\": [{}]}}", puts.join(", ")))
}

/// The save plan of the level `level` for the model in `dir` (`plan`, from its description), where the support table's row of its family says how the packer
/// saves it (`support::Saves`); else none (the GPU library's own plan, which knows no such model: its refusal says so). A config the description builder
/// refuses, or a kind the plan does not save, is an error that says why. `config`: the directory's config.json parsed once. Where there is none, or its keys do
/// not parse, there is no plan, but for a GLM-5.3-Flash release's (its names tell it): `glm`, its error, which names the key.
pub(crate) fn level_plan(dir: &Path, level: Layout, config: Option<&ModelConfig>, glm: Option<Error>) -> Result<Option<String>> {
    let Some(c) = config else { return glm.map_or(Ok(None), Err) };
    let Some(form) = saves_of(c) else { return Ok(None) };
    let label = dir.join("config.json").display().to_string();
    // (the checkpoint's own form of its experts, by its index: a tensor an expert's matrix, or each layer's experts stacked)
    let (held, _) = crate::runtime::describe::describe_dir_of(dir, c)?;
    let ck = io(Checkpoint::open(dir))?;
    let f32 = |n: &str| ck.get(n).is_some_and(|(_, t)| t.dtype == "F32");
    let shape = |n: &str| ck.get(n).map(|(_, t)| t.shape.iter().map(|&x| x as usize).collect());
    plan(c, level, &form, Some(&held), &f32, &shape).map(Some).map_err(|e| e.context(&label))
}

/// How the packer saves the model of config `c`: its family's row's `saves` (src/support.rs), where it has one.
fn saves_of(c: &ModelConfig) -> Option<Saves> {
    c.family().and_then(|f| crate::support::row(f).saves)
}

pub fn run(model: &str, out: &Path, threads: usize, layout: Layout) -> Result<()> {
    run_with(model, out, threads, layout, false)
}

/// `run`, saying nothing where `quiet` (a caller with a bar of its own).
pub fn run_with(model: &str, out: &Path, threads: usize, layout: Layout, quiet: bool) -> Result<()> {
    let source = io(Source::find(model))?;
    let (config, glm) = match ModelConfig::load_or_name(&source.dir) {
        Ok(c) => (Some(c), None),
        Err((e, family)) => (None, (family == Some(Family::Glm)).then_some(e)),
    };
    run_of(&source, config.as_ref(), glm, out, threads, layout, quiet)
}

/// `run_with` of the model `source`, its config.json parsed once (`config`; `glm`: where its keys do not parse, a GLM-5.3-Flash release's error: `level_plan`).
pub fn run_of(source: &Source, config: Option<&ModelConfig>, glm: Option<Error>, out: &Path, threads: usize, layout: Layout, quiet: bool) -> Result<()> {
    // (a level the family's row does not run is refused in `run`'s words: a save of it no level reads)
    let row = config.and_then(ModelConfig::family).map(crate::support::row);
    if let Some(r) = row.and_then(|s| s.pack_refusal(layout, &source.repo, &crate::levels::short_name(&source.repo))) {
        return err(r.to_string());
    }
    // (penguin of a family whose row says so, a Gemma 4 model's and a plain dense one's, is a glyd-v7 save too: the executor reads its packs as saved)
    let saved_penguin = config.and_then(saves_of).is_some_and(|s| s.penguin);
    let t0 = Instant::now();
    let mut have = out.join("glyd.json").exists();
    if have && crate::weights::earlier_format(out).is_some() {
        // (a directory of the tiered layout of glyd 0.27 and before, which the library no longer reads: the packs of this tool are made again; its KV tables and any other file stay)
        if !quiet {
            eprintln!("{}: {}: packing it again", out.display(), crate::weights::EARLIER_FORMAT);
        }
        for e in io(std::fs::read_dir(out))? {
            let name = io(e)?.file_name().to_string_lossy().into_owned();
            if name == "glyd.json"
                || name == "engine.safetensors"
                || name == "model.safetensors.index.json"
                || (name.starts_with("model") && name.ends_with(".safetensors"))
            {
                io(std::fs::remove_file(out.join(&name)))?;
            }
        }
        have = false;
    }
    if have {
        if !quiet {
            eprintln!("{}: already packed, keeping its files", out.display());
        }
    } else {
        // (a lossy level of a Qwen3.5 layout's model (dense or with experts), a NemotronH or a GLM-5.3-Flash one: the loader's own packs and tensors, in the
        // checkpoint's names)
        // (and penguin of a Gemma 4 model or a plain dense one: the executor reads the packs as saved, a glyd-v7 directory of level penguin)
        let planned =
            if matches!(layout, Layout::Int6 | Layout::Int5 | Layout::Int4 | Layout::Int4H | Layout::Int3) || (layout == Layout::Tiered && saved_penguin) {
                level_plan(&source.dir, layout, config, glm)?
            } else {
                None
            };
        let s = match planned {
            Some(plan) if layout == Layout::Tiered => io(save::save_penguin(source, out, threads, 5 << 30, &plan))?,
            Some(plan) => io(save::save_planned(source, out, threads, 5 << 30, layout, &plan))?,
            None => io(save::save(source, out, threads, 5 << 30, true, layout))?,
        };
        if !quiet {
            eprintln!(
                "{}: {} packs checked, {} tensors hashed, {:.2} GB of bf16 read, {} shards ({:.1} s)",
                match layout {
                    Layout::Twelve => "glyd-v3",
                    Layout::Int6 | Layout::Int5 | Layout::Int4 | Layout::Int4H | Layout::Int3 => "glyd-v7",
                    Layout::Tiered if saved_penguin => "glyd-v7",
                    _ => "glyd-v4",
                },
                s.checked,
                s.hashed,
                s.bytes as f64 / 1e9,
                s.shards,
                t0.elapsed().as_secs_f64()
            );
        }
    }
    if matches!(layout, Layout::Int6 | Layout::Int5 | Layout::Int4 | Layout::Int4H | Layout::Int3) || saved_penguin {
        return Ok(()); // (a glyd-v7 save holds the embedding packed: nothing of the engine's sidecar)
    }
    // the embedding
    let t1 = Instant::now();
    let ck = io(Checkpoint::open(&source.dir))?;
    let (f, t) = ck.get("model.embed_tokens.weight").ok_or_else(|| Error("no model.embed_tokens.weight".into()))?;
    if t.dtype != "BF16" || t.shape.len() != 2 {
        return err(format!("model.embed_tokens.weight: {} {:?}, where a bf16 matrix is wanted", t.dtype, t.shape));
    }
    let (rows, cols) = (t.shape[0] as usize, t.shape[1] as usize);
    if rows % 64 != 0 || cols % 16 != 0 {
        return err(format!("the embedding is {rows} x {cols}: this layout does not take that shape"));
    }
    let mut w = vec![0u16; rows * cols];
    {
        // SAFETY: u16's bytes, as many as its elements take; the file's are little-endian, as the host's.
        let bytes = unsafe { std::slice::from_raw_parts_mut(w.as_mut_ptr() as *mut u8, w.len() * 2) };
        io(f.read_into(t, bytes))?;
    }
    let what = || Error("the embedding decoded to other bits than its weights".into());
    let le = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    type Bufs<'a> = Vec<(&'a str, &'a str, Vec<u8>)>;
    // (the smallest layout packs the embedding in the smaller of the two, as it does each matrix of the save; the sidecar's metadata says which)
    let packed = match layout {
        Layout::Tiered | Layout::Compact => pack_smallest_checked(&w, rows, cols).ok_or_else(what)?,
        Layout::Int6
        | Layout::Int5
        | Layout::Int4
        | Layout::Int4H
        | Layout::Int3
        | Layout::Int6Row
        | Layout::Int5Row
        | Layout::Int4Row
        | Layout::Cb3Row
        | Layout::Cb2Row
        | Layout::Hummingbird => unreachable!("returned above"),
        Layout::Twelve => Packed::Twelve(pack_twelve_checked(&w, rows, cols).ok_or_else(what)?),
    };
    let (meta, mut bufs, used): (String, Bufs, Layout) = match packed {
        Packed::Tiered(p) => {
            let base = le(&p.block_base);
            (
                format!("{{\"layout\":\"mma\",\"shape\":[{rows},{cols}],\"{}\":[{},{},{}]}}", crate::weights::WORDS_KEY, p.tiers[0], p.tiers[1], p.tiers[2]),
                vec![("embed.glyd_data", "U8", p.data), ("embed.glyd_blocks", "U8", p.blocks), ("embed.glyd_block_base", "I32", base)],
                Layout::Tiered,
            )
        }
        Packed::Twelve(p) => {
            let (exc, base) = (le(&p.exc), le(&p.exc_base));
            (
                format!("{{\"layout\":\"mma12\",\"shape\":[{rows},{cols}],\"hb\":{}}}", p.hb()),
                vec![("embed.glyd_data", "U8", p.data), ("embed.glyd_exc", "I32", exc), ("embed.glyd_exc_base", "I32", base)],
                Layout::Twelve,
            )
        }
    };
    drop(w);
    let tensors: Vec<(&str, &str, Vec<u64>, &[u8])> =
        bufs.iter_mut().map(|(n, d, b)| (*n, *d, vec![b.len() as u64 / if *d == "I32" { 4 } else { 1 }], &b[..])).collect();
    write_safetensors(&out.join("engine.safetensors"), &[("format", "glyd-engine-v1".to_string()), ("embed", meta)], &tensors)?;
    if !quiet {
        eprintln!(
            "embedding {rows} x {cols} packed ({}) and checked ({:.1} s); wrote {}",
            used.name(),
            t1.elapsed().as_secs_f64(),
            out.join("engine.safetensors").display()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glm::ConfigGlm;
    use crate::nemotron::{pack_layout, ConfigNemotronH};
    use crate::qwen35::Config35;
    use glyd_gpu::safetensors::File;

    /// The family's row's form of its saves.
    fn form(f: Family) -> Saves {
        crate::support::row(f).saves.unwrap()
    }

    /// The plan of a family's own lists of its packs and tensors (its reader's, `Config35::packs` and `kept`), to hold the description's to: one linear for
    /// each pack, its parts fused under the first one's name, one copy for each tensor (an F32 one cast).
    fn listed(packs: &[Pack35], kept: &[Tensor35], f32: &dyn Fn(&str) -> bool) -> String {
        let q = |n: &str| format!("\"{n}\"");
        let linear = |p: &Pack35| {
            let names = p.parts.iter().map(|x| q(Pack35::saved(&x.0))).collect::<Vec<_>>().join(", ");
            format!("{{\"linear\": [{names}], \"from\": [{names}]}}")
        };
        let copy = |t: &Tensor35| format!("{{\"copy\": {}{}}}", q(&t.name), if f32(&t.name) { ", \"cast\": \"BF16\"" } else { "" });
        format!("{{\"puts\": [{}]}}", packs.iter().map(linear).chain(kept.iter().map(copy)).collect::<Vec<_>>().join(", "))
    }

    /// The plan of a real config whose checkpoint's tensors (name, dtype, shape) a fixture lists: their dtypes cast as the row says, their shapes where the
    /// description gives none.
    fn plan_of(text: &str, level: Layout, family: Family, tensors: &[(String, String, Vec<u64>)]) -> String {
        let t: std::collections::HashMap<&str, (&str, Vec<usize>)> =
            tensors.iter().map(|(n, d, s)| (n.as_str(), (d.as_str(), s.iter().map(|&x| x as usize).collect()))).collect();
        let (f32, shape) = (|n: &str| t.get(n).is_some_and(|x| x.0 == "F32"), |n: &str| t.get(n).map(|x| x.1.clone()));
        plan(&ModelConfig::parse(text).unwrap(), level, &form(family), None, &f32, &shape).unwrap_or_else(|e| panic!("{}", e.msg))
    }

    /// What the engine writes, the crate's reader reads back: names, dtypes, shapes, bytes and the metadata.
    #[test]
    fn safetensors_round_trip() {
        let dir = std::env::temp_dir().join(format!("glyd-engine-st-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.safetensors");
        let a = vec![1u8, 2, 3, 4, 5];
        let b: Vec<u8> = [7i32, -8, 9].iter().flat_map(|x| x.to_le_bytes()).collect();
        write_safetensors(
            &p,
            &[("format", "x".to_string()), ("meta", "{\"a\":1}".to_string())],
            &[("a", "U8", vec![5], &a[..]), ("b", "I32", vec![3], &b[..])],
        )
        .unwrap();
        let f = File::open(&p).unwrap();
        assert_eq!(f.tensors.len(), 2);
        assert_eq!(f.read(f.get("a").unwrap()).unwrap(), a);
        assert_eq!(f.get("b").unwrap().shape, vec![3]);
        assert_eq!(f.read(f.get("b").unwrap()).unwrap(), b);
        assert!(f.metadata.iter().any(|(k, v)| k == "meta" && v == "{\"a\":1}"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The Qwen3.5 plan, made from the description, names every pack part and every kept tensor of the dense configs once, in the checkpoint's names, and
    /// nothing of the vision tower or the MTP head: byte for byte the plan of the dense layout's own lists (`Config35::packs` and `kept`, which the packer read
    /// before it read the description), so a dense model's save is the same files it was.
    #[test]
    fn qwen35_plan_lists_what_the_loader_reads() {
        let all = glyd_gpu::json::parse(include_str!("../tests/fixtures/qwen35_configs.json")).unwrap();
        let have = glyd_gpu::json::parse(include_str!("../tests/fixtures/qwen35_tensors.json")).unwrap();
        for name in ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-27B", "Qwen/Qwen3.6-27B", "Qwen/Qwen3.8-27B"] {
            let text = glyd_gpu::json::to_compact(all.get(name).unwrap());
            let c = Config35::parse(&text, name, None).unwrap();
            let tensors = have.get(name).and_then(|t| t.as_object());
            let f32 = |n: &str| tensors.is_some_and(|t| t.iter().any(|(k, v)| k == n && v.as_array().unwrap()[0].as_str() == Some("F32")));
            let made = plan(&ModelConfig::parse(&text).unwrap(), Layout::Int6, &form(Family::Qwen35), None, &f32, &|_| None).unwrap();
            assert_eq!(made, listed(&c.packs(), &c.kept(), &f32), "{name}");
            let plan = glyd_gpu::json::parse(&made).unwrap();
            let (mut linear, mut copy) = (Vec::new(), Vec::new());
            for it in plan.get("puts").unwrap().as_array().unwrap() {
                match it.get("linear") {
                    Some(l) => {
                        let m: Vec<&str> = l.as_array().unwrap().iter().map(|x| x.as_str().unwrap()).collect();
                        assert_eq!(it.get("from").unwrap(), l, "{name}");
                        linear.extend(m.into_iter().map(String::from));
                    }
                    None => copy.push((it.get("copy").unwrap().as_str().unwrap().to_string(), it.get("cast").is_some())),
                }
            }
            let parts: Vec<String> = c.packs().iter().flat_map(|p| p.parts.iter().map(|x| Pack35::saved(&x.0).to_string())).collect();
            assert_eq!(linear, parts, "{name}");
            let kept: Vec<String> = c.kept().into_iter().map(|t| t.name).collect();
            assert_eq!(copy.iter().map(|c| c.0.clone()).collect::<Vec<_>>(), kept, "{name}");
            let all: std::collections::HashSet<_> = linear.iter().map(|l| format!("{l}.weight")).chain(kept.iter().cloned()).collect();
            assert_eq!(all.len(), linear.len() + kept.len(), "{name}: a tensor twice");
            assert!(all.iter().all(|n| n.starts_with(c.prefix) || n == "lm_head.weight") && !all.iter().any(|n| skipped_name(n)), "{name}");
            if let Some(t) = tensors {
                assert!(all.iter().all(|n| t.iter().any(|(k, _)| k == n)), "{name}: a tensor the checkpoint lacks");
                assert!(copy.iter().filter(|c| c.1).count() > 0, "{name}: no F32 tensor cast");
            }
        }
    }

    /// The Qwen3.5 layout with experts packs by its description (the random case's config and the two releases' of the fixtures): every tensor the description
    /// reads once (none of the MTP head's or the vision tower's), each expert's gate and up as one pack and its down as another, the shared expert's two packs,
    /// the routers and the shared expert's gate copied as saved, and every other matrix and small tensor as the dense layout names them (no MLP's).
    #[test]
    fn the_qwen35_layout_with_experts_packs_by_its_description() {
        let all = glyd_gpu::json::parse(include_str!("../tests/fixtures/qwen35_configs.json")).unwrap();
        let cases = [
            ("the random case", include_str!("../tests/fixtures/qwen35moe_tiny_config.json").to_string()),
            ("Qwen3.5-35B-A3B", glyd_gpu::json::to_compact(all.get("Qwen/Qwen3.5-35B-A3B").unwrap())),
            ("Qwen3.8-2.4T-A95B", glyd_gpu::json::to_compact(all.get("Qwen/Qwen3.8-2.4T-A95B").unwrap())),
        ];
        for (what, text) in &cases {
            let mc = ModelConfig::parse(text).unwrap();
            let d = crate::runtime::describe::describe_config(&mc).unwrap();
            let plan = glyd_gpu::json::parse(&plan(&mc, Layout::Int6, &form(Family::Qwen35), None, &|_| false, &|_| None).unwrap()).unwrap();
            let (mut linear, mut copy) = (Vec::new(), Vec::new());
            for it in plan.get("puts").unwrap().as_array().unwrap() {
                match it.get("linear") {
                    Some(l) => linear.push(l.as_array().unwrap().iter().map(|x| format!("{}.weight", x.as_str().unwrap())).collect::<Vec<_>>()),
                    None => copy.push(it.get("copy").unwrap().as_str().unwrap().to_string()),
                }
            }
            let named: std::collections::BTreeSet<String> = linear.iter().flatten().chain(&copy).cloned().collect();
            assert_eq!(named.len(), linear.iter().map(Vec::len).sum::<usize>() + copy.len(), "{what}: a tensor twice");
            assert_eq!(named, d.names(), "{what}");
            assert!(!named.iter().any(|n| n.starts_with("mtp.") || n.contains(".visual.") || n.contains(".mlp.gate_proj.")), "{what}");
            let (e, layers) = (mc.shared.experts(), d.desc.blocks.len() / 2);
            let expert =
                |l: &Vec<String>, tail: &[&str]| l.len() == tail.len() && l.iter().zip(tail).all(|(n, t)| n.contains(".mlp.experts.") && n.ends_with(t));
            let gate_up = linear.iter().filter(|l| expert(l, &[".gate_proj.weight", ".up_proj.weight"])).count();
            let down = linear.iter().filter(|l| expert(l, &[".down_proj.weight"])).count();
            assert_eq!((gate_up, down, e > 0), (layers * e, layers * e, true), "{what}");
            let shared = linear.iter().filter(|l| l[0].contains(".mlp.shared_expert.")).count();
            assert_eq!(shared, 2 * layers, "{what}");
            for tail in [".mlp.gate.weight", ".mlp.shared_expert_gate.weight"] {
                assert_eq!(copy.iter().filter(|n| n.ends_with(tail)).count(), layers, "{what}: {tail}");
            }
        }
    }

    /// The NemotronH plan, made from the description, names every pack part (every expert's two matrices) and every kept tensor of the pinned checkpoint once, in
    /// its names, in the order of the family's own lists (`ConfigNemotronH::packs` and `kept`, which the packer read before), and nothing of the MTP head; its one
    /// fp32 class is cast.
    #[test]
    fn nemotron_plan_lists_what_the_loader_reads() {
        let c = crate::nemotron::tests::parse();
        let have = crate::nemotron::tests::tensors();
        let text = include_str!("../tests/fixtures/nemotron_h_config.json");
        let plan = glyd_gpu::json::parse(&plan_of(text, Layout::Int5, Family::NemotronH, &have)).unwrap();
        let (mut linear, mut copy) = (Vec::new(), Vec::new());
        for it in plan.get("puts").unwrap().as_array().unwrap() {
            match it.get("linear") {
                Some(l) => {
                    assert_eq!(it.get("from").unwrap(), l);
                    linear.extend(l.as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()));
                }
                None => copy.push((it.get("copy").unwrap().as_str().unwrap().to_string(), it.get("cast").is_some())),
            }
        }
        let parts: Vec<String> = c.packs().iter().flat_map(|p| p.parts.iter().map(|x| Pack35::saved(&x.0).to_string())).collect();
        assert_eq!(linear, parts);
        assert_eq!(linear.len(), 40 * 2 + 8 * 4 + 40 * (4 + 1024) + 2);
        assert_eq!(copy.iter().map(|c| c.0.clone()).collect::<Vec<_>>(), c.kept().into_iter().map(|t| t.name).collect::<Vec<_>>());
        assert_eq!(copy.iter().filter(|c| c.1).count(), 40);
        assert!(copy.iter().filter(|c| c.1).all(|c| c.0.ends_with("e_score_correction_bias")));
        // the split by kind of tensor: packed, the matrices (Mamba-2's two a layer, attention's four, a mixture of experts' four and its 512 experts' two, the embedding and the head); kept, the routers
        // (bf16, used in fp32) and everything small: the norms, Mamba-2's convolution, A_log, D, dt_bias and gated norm
        let (packed, kept): (Vec<&str>, Vec<&str>) = (linear.iter().map(String::as_str).collect(), copy.iter().map(|c| c.0.as_str()).collect());
        let count = |names: &[&str], suffix: &str| names.iter().filter(|n| n.ends_with(suffix)).count();
        for (suffix, n) in [
            ("mixer.in_proj", 40),
            ("mixer.out_proj", 40),
            ("mixer.q_proj", 8),
            ("mixer.k_proj", 8),
            ("mixer.v_proj", 8),
            ("mixer.o_proj", 8),
            ("mixer.fc1_latent_proj", 40),
            ("mixer.fc2_latent_proj", 40),
            ("shared_experts.up_proj", 40),
            ("shared_experts.down_proj", 40),
            ("embeddings", 1),
            ("lm_head", 1),
        ] {
            assert_eq!((count(&packed, suffix), count(&kept, &format!("{suffix}.weight"))), (n, 0), "{suffix} packed");
        }
        assert_eq!((count(&packed, ".up_proj") - 40, count(&packed, ".down_proj") - 40), (20_480, 20_480));
        for (suffix, n) in [
            ("mixer.gate.weight", 40),
            ("mixer.gate.e_score_correction_bias", 40),
            ("mixer.A_log", 40),
            ("mixer.D", 40),
            ("mixer.dt_bias", 40),
            ("mixer.conv1d.weight", 40),
            ("mixer.conv1d.bias", 40),
            ("mixer.norm.weight", 40),
            ("norm_f.weight", 1),
        ] {
            assert_eq!((count(&kept, suffix), count(&packed, suffix.trim_end_matches(".weight"))), (n, 0), "{suffix} kept");
        }
        assert_eq!(count(&kept, ".norm.weight") - 40, 88);
        let all: std::collections::HashSet<String> = linear.iter().map(|l| format!("{l}.weight")).chain(copy.iter().map(|c| c.0.clone())).collect();
        assert_eq!(all.len(), linear.len() + copy.len(), "a tensor twice");
        let names: std::collections::HashSet<&str> = have.iter().map(|t| t.0.as_str()).collect();
        assert!(all.iter().all(|n| names.contains(n.as_str()) && !crate::nemotron::skipped(n)), "a tensor the checkpoint lacks, or the MTP head's");
    }

    /// The plan at each level on the pinned checkpoint's config: at kestrel and linnet no linear names a layout (the save's holds every pack, the plan as it was); at wren the 40,960 experts' matrices (112.7 billion
    /// weights) are the save's, four-bit codes, and the other 258 packs (7.84 billion) say \"i6\": six-bit codes. Qwen3.5's plan names no layout at any level.
    #[test]
    fn nemotron_plan_puts_the_experts_in_four_bit_codes_and_the_rest_in_six_at_wren() {
        let c = crate::nemotron::tests::parse();
        let have = crate::nemotron::tests::tensors();
        for level in [Layout::Int6, Layout::Int5, Layout::Int4] {
            let text = plan_of(include_str!("../tests/fixtures/nemotron_h_config.json"), level, Family::NemotronH, &have);
            let plan = glyd_gpu::json::parse(&text).unwrap();
            let linears: Vec<(&str, Option<&str>)> = plan
                .get("puts")
                .and_then(|p| p.as_array())
                .unwrap()
                .iter()
                .filter_map(|it| it.get("linear").map(|l| (l.as_array().unwrap()[0].as_str().unwrap(), it.get("layout").map(|x| x.as_str().unwrap()))))
                .collect();
            assert_eq!(linears.len(), c.packs().len());
            let named = linears.iter().filter(|l| l.1.is_some()).count();
            let down = |name: &str| name.contains(".mixer.experts.") && name.ends_with(".down_proj");
            if level != Layout::Int4 {
                // (kestrel and linnet: the level's layout for every pack but the experts' down matrices, whose plan names the row layout of the level's bits)
                let rows = level.row_of().unwrap().name();
                assert_eq!(named, 20_480, "{level:?}");
                assert!(linears.iter().all(|(name, own)| *own == if down(name) { Some(rows) } else { None }), "{level:?}");
                continue;
            }
            assert_eq!((linears.iter().filter(|l| l.0.contains(".mixer.experts.")).count(), named), (40_960, 258 + 20_480));
            for (name, own) in &linears {
                assert_eq!(
                    *own,
                    if down(name) {
                        Some("i4r")
                    } else if name.contains(".mixer.experts.") {
                        None
                    } else {
                        Some("i6")
                    },
                    "{name}"
                );
            }
            let own = |name: &str| linears.iter().find(|l| l.0 == name).unwrap().1;
            assert_eq!(
                (own("backbone.embeddings"), own("lm_head"), own("backbone.layers.1.mixer.shared_experts.up_proj")),
                (Some("i6"), Some("i6"), Some("i6"))
            );
        }
        // (a dense Qwen3.5's plan: byte for byte what it was, no layout in it)
        let (all, tensors) = (glyd_gpu::json::parse(include_str!("../tests/fixtures/qwen35_configs.json")).unwrap(), "Qwen/Qwen3.5-9B");
        let q = ModelConfig::parse(&glyd_gpu::json::to_compact(all.get(tensors).unwrap())).unwrap();
        assert!(!plan(&q, Layout::Int4, &form(Family::Qwen35), None, &|_| false, &|_| None).unwrap().contains("layout"));
    }

    /// A tiny NemotronH checkpoint (the random tiny model's config: Mamba-2, attention and mixture-of-experts layers, every matrix in whole blocks; with a tensor of the MTP head) packed by `run_with` at each of
    /// the three levels the engine runs: a glyd-v7 directory under the source's names, whose packs the loader's checks accept and `read_pack` reads as the packer makes them of the bf16 tensors (the fused q, k, v too),
    /// each in the layout `pack_layout` gives it (the level's layout at kestrel and linnet but for the experts' down matrices, in the row layout of its bits; at wren the experts four-bit, their down matrices in the
    /// four-bit row layout, and the rest six-bit, the entries naming them), and which glyd-gpu's verify decodes to the sha256 of its glyd.json. A reader that takes the level's layout for every pack refuses the packs of
    /// another layout by their tag.
    #[test]
    fn nemotron_saves_a_v7_directory_the_loader_reads() {
        use crate::ffi::Words;
        use crate::weights::{check_v7, read_pack, saved_parts, v7_manifest, WORDS_KEY};
        let dir = std::env::temp_dir().join(format!("glyd-engine-nh-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (src, text) = (dir.join("src"), include_str!("../tests/fixtures/nemotron_h_moetoy_config.json"));
        let c = ConfigNemotronH::parse(text, "moetoy", None).unwrap();
        crate::nemotron::tests::checkpoint(&src, text, &c);
        let bf16 = Checkpoint::open(&src).unwrap();
        for asked in [Layout::Int6, Layout::Int5, Layout::Int4] {
            let out = dir.join(asked.name());
            run_with(src.to_str().unwrap(), &out, 2, asked, true).unwrap();
            // the saved directory: a glyd-v7 one of the layout asked, of the same model under the same names, with every pack the loader reads and every tensor it keeps (and nothing of the MTP head)
            let (layout, m) = v7_manifest(&out).unwrap().unwrap();
            assert_eq!(layout, asked);
            let saved_config = ConfigNemotronH::load(&out).unwrap();
            assert_eq!((saved_config.prefix, saved_config.kinds.clone()), (c.prefix, c.kinds.clone()));
            let saved = Checkpoint::open(&out).unwrap();
            assert!(saved.get("mtp.layers.0.norm.weight").is_none());
            let mut layouts = std::collections::BTreeSet::new();
            for p in c.packs() {
                let key = Pack35::saved(&p.parts[0].0);
                let e = m.get("packs").and_then(|x| x.get(key)).unwrap_or_else(|| panic!("no pack {key}"));
                let lens = ["data", "blocks", "block_base"].map(|s| saved.get(&format!("{key}.glyd_{s}")).unwrap().1.bytes);
                let (shape, words) = (e.get("shape").and_then(|s| s.as_u64s()).unwrap(), e.get(WORDS_KEY).and_then(|s| s.as_u64s()).unwrap());
                // the pack's own layout: its entry names it, and its words, shape and buffers are that layout's
                let own = pack_layout(asked, &p.key);
                assert_eq!(e.get("layout").and_then(|l| l.as_str()), Some(own.name()), "{key}");
                layouts.insert(own.name());
                check_v7(key, own, p.rows(), p.cols, &shape, &words, lens).unwrap();
                // (a reader that holds the level's layout for every pack: right for a pack in it, a refusal by the tag for another)
                assert_eq!(check_v7(key, layout, p.rows(), p.cols, &shape, &words, lens).is_ok(), own == layout, "{key}");
                // (q, k and v are one pack, under q's name; the pack is the packer's of the bf16 tensors, buffer for buffer)
                assert_eq!(saved_parts(&saved, &p), vec![(key, p.rows())], "{key}");
                let (got, want) = (read_pack(&saved, Some(&(layout, m.clone())), &p, own).unwrap(), read_pack(&bf16, None, &p, own).unwrap());
                assert!(got.data == want.data && got.b1 == want.b1 && got.b2 == want.b2 && (got.rows, got.cols) == (want.rows, want.cols), "{}", p.key);
                assert!(matches!((got.words, want.words), (Words::Tiered { tiers: a, .. }, Words::Tiered { tiers: b, .. }) if a == b), "{}", p.key);
            }
            // (the level's layout and the row layout of its bits at kestrel and linnet, the experts' four-bit layout, its row layout and the six-bit one at wren)
            assert_eq!(layouts.len(), if asked == Layout::Int4 { 3 } else { 2 }, "{layouts:?}");
            for t in c.kept() {
                assert_eq!(saved.get(&t.name).unwrap().1.dtype, "BF16", "{}", t.name);
            }
            // the loader's check of the directory: every matrix held saved, every small tensor there (the bias, fp32 in the source, bf16 here)
            let held = c.saved(&saved);
            assert_eq!(held.len(), c.packs().iter().map(|p| p.parts.len()).sum::<usize>());
            let have: Vec<(String, String, Vec<u64>)> =
                saved.files.iter().flat_map(|f| f.tensors.iter().map(|(n, t)| (n.clone(), t.dtype.clone(), t.shape.clone()))).collect();
            assert_eq!(c.check(&have, &held).unwrap(), 0);
            assert!(c.check(&have, &Default::default()).is_err(), "the matrices are not bf16 tensors there");
            c.check_layers(&have, &[0, 1, 2], &held).unwrap();
            let v = save::verify(&out, 2, &save::Decoder::Cpu).unwrap();
            assert_eq!((v.packed, v.hashed), (c.packs().iter().map(|p| p.parts.len()).sum::<usize>(), c.kept().len()));
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A model of the Qwen3.5 layout (the logits checks' random one) packed at kestrel by `run_with`, as `glyd prepare MODEL:kestrel` and a run pack it: a glyd-v7
    /// directory whose merged packs (q, k and v; the gated delta rule's qkv with z; gate and up) glyd-gpu's verify takes as the save's own, every pack decoded to
    /// the sha256 of its glyd.json (`glyd verify` refused it: its qkv and z were not taken for one group).
    #[cfg(feature = "dev")]
    #[test]
    fn the_qwen35_layout_at_kestrel_verifies() {
        let dir = std::env::temp_dir().join(format!("glyd-engine-q35-i6-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (src, out) = (dir.join("src"), dir.join("i6"));
        crate::logits::qwen35_random(&src).unwrap();
        run_with(src.to_str().unwrap(), &out, 4, Layout::Int6, true).unwrap();
        let c = Config35::load(&src).unwrap();
        assert!(c.packs().iter().any(|p| p.parts.len() == 2 && p.parts[1].0.ends_with("linear_attn.in_proj_z.weight")));
        let v = save::verify(&out, 4, &save::Decoder::Cpu).unwrap();
        assert_eq!(v.packed, c.packs().iter().map(|p| p.parts.len()).sum::<usize>());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The GLM-5.3-Flash plan names every pack part and every kept tensor of the pinned checkpoint once, in its names, and nothing of the vision tower or the MTP layer; no tensor is cast (the fp32 ones are saved fp32), and
    /// the split by kind of tensor is the checkpoint's: the FP8 matrices and the bf16 ones are all packs, the small tensors and the indexer copies.
    #[test]
    fn glm_plan_lists_what_the_loader_reads() {
        let c = crate::glm::tests::parse();
        let have = crate::glm::tests::tensors();
        let plan = glyd_gpu::json::parse(&plan_of(include_str!("../tests/fixtures/glm5_next_config.json"), Layout::Int6, Family::Glm, &have)).unwrap();
        let (mut linear, mut copy, mut fused) = (Vec::new(), Vec::new(), 0);
        for it in plan.get("puts").unwrap().as_array().unwrap() {
            match it.get("linear") {
                Some(l) => {
                    assert_eq!(it.get("from").unwrap(), l);
                    let m: Vec<String> = l.as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect();
                    fused += (m.len() > 1) as usize;
                    linear.extend(m);
                }
                None => {
                    assert!(it.get("cast").is_none() && it.get("from").is_none());
                    copy.push(it.get("copy").unwrap().as_str().unwrap().to_string());
                }
            }
        }
        let parts: Vec<String> = c.packs().iter().flat_map(|p| p.parts.iter().map(|x| Pack35::saved(&x.0).to_string())).collect();
        assert_eq!(linear, parts);
        assert_eq!(linear.len(), 36_786);
        // (a gate and an up as one: 3 dense MLPs, 42 shared experts and 12,096 experts; q, k and v as one in the 34 Kimi delta layers)
        assert_eq!(fused, 3 + 42 + 12_096 + 34);
        assert_eq!(copy, c.kept().into_iter().map(|t| t.name).collect::<Vec<_>>());
        assert_eq!(copy.len(), 748);
        let count = |names: &[String], suffix: &str| names.iter().filter(|n| n.ends_with(suffix)).count();
        for (suffix, n) in [
            ("self_attn.q_proj", 34),
            ("self_attn.k_proj", 34),
            ("self_attn.b_proj", 34),
            ("self_attn.f_b_proj", 34),
            ("self_attn.kv_b_proj", 11),
            ("self_attn.kv_a_proj_with_mqa", 11),
            ("embed_tokens", 1),
            ("lm_head", 1),
        ] {
            assert_eq!((count(&linear, suffix), count(&copy, &format!("{suffix}.weight"))), (n, 0), "{suffix} packed");
        }
        // (the first expert's gate and up in each of the 42 layers; every down: the dense MLPs', the shared experts' and the experts')
        assert_eq!(
            (count(&linear, "mlp.experts.0.gate_proj") + count(&linear, "mlp.experts.0.up_proj"), count(&linear, ".down_proj")),
            (2 * 42, 3 + 42 + 42 * 288)
        );
        for (suffix, n) in [
            ("mlp.gate.weight", 42),
            ("mlp.gate.e_score_correction_bias", 42),
            ("self_attn.A_log", 34),
            ("self_attn.dt_bias", 34),
            ("self_attn.q_conv1d.weight", 34),
            ("hc_attn_fn", 45),
            ("hc_ffn_base", 45),
            ("indexer.wq_b.weight", 11),
            ("indexer.weights_proj.weight", 11),
            ("indexer.index_kpool_compress_gate", 11),
            ("norm.weight", 45 * 2 + 11 * 2 + 34 + 11 + 1),
        ] {
            assert_eq!((count(&copy, suffix), count(&linear, suffix.trim_end_matches(".weight"))), (n, 0), "{suffix} kept");
        }
        let all: std::collections::HashSet<String> = linear.iter().map(|l| format!("{l}.weight")).chain(copy.iter().cloned()).collect();
        assert_eq!(all.len(), linear.len() + copy.len(), "a tensor twice");
        let names: std::collections::HashSet<&str> = have.iter().map(|t| t.0.as_str()).collect();
        assert!(all.iter().all(|n| names.contains(n.as_str()) && !c.skipped(n)), "a tensor the checkpoint lacks, or the MTP layer's or the vision tower's");
    }

    /// A tiny GLM-5.3-Flash checkpoint laid out as the real one is (Kimi delta attention and latent attention layers, a dense MLP and mixture-of-experts layers, the experts, the MLPs and the latent attention's projections as
    /// E4M3 bytes with block scales that do not divide into whole blocks, the rest bf16, fp32 parameters, the hyper-connections, an indexer; with a tensor of the MTP layer and one of the vision tower) saved by its row's
    /// plan (`level_plan`; `run_with` refuses those levels, which the row does not run) in each of two block layouts: a glyd-v7 directory under the source's names, whose packs the loader's checks accept and which are, buffer for buffer, the packer's of the bf16 the matrices dequantise to
    /// (the same checkpoint with them dequantised first, and the FP8 one packed as the loader packs a bf16 directory), whose small tensors are the source's bytes (the fp32 ones fp32), and which glyd-gpu's verify decodes
    /// to the sha256 of its glyd.json. A config whose experts the kernels do not run (another activation) is refused in words.
    #[test]
    fn glm_saves_a_v7_directory_the_loader_reads() {
        use crate::weights::{check_v7, read_pack, saved_parts, v7_manifest, WORDS_KEY};
        let dir = std::env::temp_dir().join(format!("glyd-engine-glm-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let text = crate::glm::tests::tiny_text();
        let c = ConfigGlm::parse(text, "tiny", None).unwrap();
        let words = crate::glm::tests::checkpoint(&dir, text, &c);
        let (fp8, bf16) = (Checkpoint::open(&dir.join("fp8")).unwrap(), Checkpoint::open(&dir.join("bf16")).unwrap());
        let n_parts = c.packs().iter().map(|p| p.parts.len()).sum::<usize>();
        // (the source: FP8 matrices, their scales, and what the engine leaves out)
        assert!(fp8.names().filter(|n| n.ends_with("_scale_inv")).count() > 20 && bf16.names().all(|n| !n.ends_with("_scale_inv")));
        let src: Vec<(String, String, Vec<u64>)> =
            fp8.files.iter().flat_map(|f| f.tensors.iter().map(|(n, t)| (n.clone(), t.dtype.clone(), t.shape.clone()))).collect();
        assert_eq!(c.check(&src, &Default::default()).unwrap(), 3);
        let source = Source::find(dir.join("fp8").to_str().unwrap()).unwrap();
        let config = ModelConfig::load(&source.dir).unwrap();
        for asked in [Layout::Int6, Layout::Int5] {
            let out = dir.join(asked.name());
            let plan = level_plan(&source.dir, asked, Some(&config), None).unwrap().unwrap();
            save::save_planned(&source, &out, 2, 5 << 30, asked, &plan).unwrap();
            // the saved directory: a glyd-v7 one of the layout asked, of the same model under the same names, with every pack the loader reads and every tensor it keeps, and nothing of the MTP layer, the vision tower or the scales
            let (layout, m) = v7_manifest(&out).unwrap().unwrap();
            assert_eq!(layout, asked);
            let saved_config = ConfigGlm::load(&out).unwrap();
            assert_eq!((saved_config.prefix, saved_config.kinds.clone(), saved_config.fp8_block), (c.prefix, c.kinds.clone(), c.fp8_block));
            let saved = Checkpoint::open(&out).unwrap();
            assert!(saved.names().all(|n| !c.skipped(n) && !n.ends_with("_scale_inv")));
            for p in c.packs() {
                let key = Pack35::saved(&p.parts[0].0);
                let e = m.get("packs").and_then(|x| x.get(key)).unwrap_or_else(|| panic!("no pack {key}"));
                let lens = ["data", "blocks", "block_base"].map(|s| saved.get(&format!("{key}.glyd_{s}")).unwrap().1.bytes);
                check_v7(
                    key,
                    layout,
                    p.rows(),
                    p.cols,
                    &e.get("shape").and_then(|s| s.as_u64s()).unwrap(),
                    &e.get(WORDS_KEY).and_then(|s| s.as_u64s()).unwrap(),
                    lens,
                )
                .unwrap();
                // (a gate and an up, or q, k and v, are one pack under the first's name)
                assert_eq!(saved_parts(&saved, &p), vec![(key, p.rows())], "{key}");
                let got = read_pack(&saved, Some(&(layout, m.clone())), &p, asked).unwrap();
                for want in [read_pack(&bf16, None, &p, asked).unwrap(), read_pack(&fp8, None, &p, asked).unwrap()] {
                    assert!(got.data == want.data && got.b1 == want.b1 && got.b2 == want.b2 && (got.rows, got.cols) == (want.rows, want.cols), "{}", p.key);
                }
                // and the FP8 source is read as the bf16 the test worked out apart
                for (name, rows) in &p.parts {
                    let mut w = vec![0u16; rows * p.cols];
                    fp8.read_bf16(name, &mut w).unwrap();
                    assert!(w == words[name], "{name}");
                }
            }
            // the tensors kept are the source's bytes: bf16 ones bf16, fp32 ones fp32 (not cast)
            for t in c.kept() {
                assert_eq!(saved.get(&t.name).unwrap().1.dtype, t.dtype, "{}", t.name);
                assert!(saved.read(&t.name).unwrap() == fp8.read(&t.name).unwrap(), "{}", t.name);
            }
            assert!(c.kept().iter().filter(|t| t.dtype == "F32").count() >= 20);
            // the loader's check of the directory: every matrix held saved, every small tensor there
            let held = c.saved(&saved);
            assert_eq!(held.len(), n_parts);
            let have: Vec<(String, String, Vec<u64>)> =
                saved.files.iter().flat_map(|f| f.tensors.iter().map(|(n, t)| (n.clone(), t.dtype.clone(), t.shape.clone()))).collect();
            assert_eq!(c.check(&have, &held).unwrap(), 0);
            assert!(c.check(&have, &Default::default()).is_err(), "the matrices are not tensors there");
            let v = save::verify(&out, 2, &save::Decoder::Cpu).unwrap();
            assert_eq!((v.packed, v.hashed), (n_parts, c.kept().len()));
        }
        // a lossless layout is not an FP8 model's: refused in words
        let e = run_with(dir.join("fp8").to_str().unwrap(), &dir.join("lossless"), 2, Layout::Tiered, true).unwrap_err().msg;
        assert!(e.contains("not a family this packer knows"), "{e:?}");
        // a config of GLM-5.3-Flash that the engine does not run says why, and is not taken for another model
        let bad = dir.join("refused");
        std::fs::create_dir_all(&bad).unwrap();
        std::fs::write(bad.join("config.json"), text.replace("\"silu\"", "\"gelu\"")).unwrap();
        std::fs::copy(dir.join("fp8").join("model.safetensors"), bad.join("model.safetensors")).unwrap();
        let e = level_plan(&bad, Layout::Int6, ModelConfig::load(&bad).ok().as_ref(), None).unwrap_err().msg;
        assert!(e.contains("the kernels take SwiGLU, GeGLU, relu squared and SiTU"), "{e:?}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    fn skipped_name(n: &str) -> bool {
        crate::qwen35::skipped(n)
    }

    /// A tiny Qwen3.5 checkpoint (with a vision tensor and an MTP one) written to `dir/src`: its source, its config and its save plan.
    fn tiny_qwen35(dir: &Path) -> (Source, Config35, String) {
        let _ = std::fs::remove_dir_all(dir);
        let src = dir.join("src");
        std::fs::create_dir_all(&src).unwrap();
        // (the heads the loader runs, whose description the packer reads: attention heads of 256, linear heads of 128, a convolution of 4)
        let config = r#"{"architectures": ["Qwen3_5ForConditionalGeneration"], "tie_word_embeddings": false, "text_config": {"model_type": "qwen3_5_text", "hidden_size": 64, "intermediate_size": 64, "num_hidden_layers": 2, "layer_types": ["linear_attention", "full_attention"], "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 256, "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 1, "linear_num_value_heads": 2, "linear_value_head_dim": 128, "max_position_embeddings": 128, "vocab_size": 64, "rms_norm_eps": 1e-6, "eos_token_id": 1, "rope_parameters": {"rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 0.25, "mrope_section": [16, 8, 8]}}}"#;
        std::fs::write(src.join("config.json"), config).unwrap();
        let c = Config35::parse(config, "tiny", None).unwrap();
        let mut want = c.kept();
        want.extend(
            c.packs().iter().flat_map(|p| p.parts.iter().map(|x| Tensor35 { name: x.0.clone(), dtype: "BF16", shape: vec![x.1 as u64, p.cols as u64] })),
        );
        want.push(Tensor35 { name: "model.visual.patch_embed.weight".into(), dtype: "BF16", shape: vec![64, 64] });
        want.push(Tensor35 { name: "mtp.fc.weight".into(), dtype: "BF16", shape: vec![64, 64] });
        let data: Vec<(String, &str, Vec<u64>, Vec<u8>)> = want
            .iter()
            .map(|t| {
                let n: u64 = t.shape.iter().product();
                // (bf16 values in a small range; the two F32 parameters as the checkpoint holds them)
                let f = |i: u64| ((((i * 2654435761) >> 7) % 400) as f32 / 100.0 - 2.0).to_bits();
                let bytes = if t.dtype == "F32" {
                    (0..n).flat_map(|i| f(i).to_le_bytes()).collect()
                } else {
                    (0..n).flat_map(|i| ((f(i) >> 16) as u16).to_le_bytes()).collect()
                };
                (t.name.clone(), t.dtype, t.shape.clone(), bytes)
            })
            .collect();
        let tensors: Vec<(&str, &str, Vec<u64>, &[u8])> = data.iter().map(|(n, d, s, b)| (n.as_str(), *d, s.clone(), &b[..])).collect();
        write_safetensors(&src.join("model.safetensors"), &[("format", "pt".to_string())], &tensors).unwrap();
        let source = Source::find(src.to_str().unwrap()).unwrap();
        let ck = Checkpoint::open(&src).unwrap();
        let f32 = |n: &str| ck.get(n).is_some_and(|(_, t)| t.dtype == "F32");
        let plan = plan(&ModelConfig::parse(config).unwrap(), Layout::Int6, &form(Family::Qwen35), None, &f32, &|_| None).unwrap();
        (source, c, plan)
    }

    /// A tiny Qwen3.5 checkpoint saved through the plan in each block layout: a glyd-v7 directory the loader's checks accept, under the source's names.
    #[test]
    fn qwen35_saves_a_v7_directory_the_loader_reads() {
        let dir = std::env::temp_dir().join(format!("glyd-engine-q35-{}", std::process::id()));
        let (source, _, plan) = tiny_qwen35(&dir);
        for asked in [Layout::Int6, Layout::Int4, Layout::Int3] {
            let out = dir.join(asked.name());
            save::save_planned(&source, &out, 2, 5 << 30, asked, &plan).unwrap();
            // the saved directory: a glyd-v7 one of the layout asked that keeps the source's names, with every pack the loader reads and every tensor it keeps
            let (layout, m) = crate::weights::v7_manifest(&out).unwrap().unwrap();
            assert_eq!(layout, asked);
            let c = Config35::load(&out).unwrap();
            assert_eq!(c.prefix, "model.language_model.");
            let saved = Checkpoint::open(&out).unwrap();
            for p in c.packs() {
                let key = Pack35::saved(&p.parts[0].0);
                let e = m.get("packs").and_then(|x| x.get(key)).unwrap_or_else(|| panic!("no pack {key}"));
                let lens = ["data", "blocks", "block_base"].map(|s| saved.get(&format!("{key}.glyd_{s}")).unwrap().1.bytes);
                crate::weights::check_v7(
                    key,
                    layout,
                    p.rows(),
                    p.cols,
                    &e.get("shape").and_then(|s| s.as_u64s()).unwrap(),
                    &e.get(crate::weights::WORDS_KEY).and_then(|s| s.as_u64s()).unwrap(),
                    lens,
                )
                .unwrap();
            }
            for t in c.kept() {
                assert_eq!(saved.get(&t.name).unwrap().1.dtype, "BF16", "{}", t.name);
            }
            assert!(saved.get("model.visual.patch_embed.weight").is_none() && saved.get("mtp.fc.weight").is_none());
            let held: std::collections::HashSet<String> = c.packs().iter().flat_map(|p| p.parts.iter().map(|x| Pack35::saved(&x.0).to_string())).collect(); // (a fused pack holds its parts)
            let have: Vec<(String, String, Vec<u64>)> =
                saved.files.iter().flat_map(|f| f.tensors.iter().map(|(n, t)| (n.clone(), t.dtype.clone(), t.shape.clone()))).collect();
            c.check(&have, &held).unwrap();
        }
        // `run_with` (what `glyd run` and `glyd-engine pack` call) takes the same plan from the config
        let out = dir.join("run");
        run_with(dir.join("src").to_str().unwrap(), &out, 2, Layout::Int5, true).unwrap();
        assert_eq!(crate::weights::v7_manifest(&out).unwrap().unwrap().0, Layout::Int5);
        assert_eq!(Config35::load(&out).unwrap().prefix, "model.language_model.");
        assert!(Checkpoint::open(&out).unwrap().get("model.visual.patch_embed.weight").is_none());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The same tiny checkpoint saved, in each block layout, from calibrated codes and scales (spreads no fit makes) for every pack but the embedding's: glyd.json records the calibration, the loader's checks accept
    /// every pack and `read_v7` reads each coded pack, whose saved bytes are exactly what glyd-gpu's packer makes of its codes and scales.
    #[test]
    fn qwen35_saves_calibrated_codes_the_loader_reads() {
        let dir = std::env::temp_dir().join(format!("glyd-engine-q35-codes-{}", std::process::id()));
        let (source, c, plan) = tiny_qwen35(&dir);
        let calibration = r#"{"method": "gptq-act", "data": [], "seqs": 512, "tokens": 2048}"#;
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for (layout, q) in [(Layout::Int6, 31i8), (Layout::Int5, 15)] {
            let (codes, out) = (dir.join(format!("codes-{q}")), dir.join(format!("out-{q}")));
            std::fs::create_dir_all(&codes).unwrap();
            std::fs::write(codes.join("calibration.json"), calibration).unwrap();
            // a pack's codes and scales, its parts' stacked (the embedding's pack has none: it is packed from its weights)
            let coded: Vec<(Pack35, Vec<i8>, Vec<u16>)> = c
                .packs()
                .into_iter()
                .filter(|p| p.key != "embed")
                .map(|p| {
                    let code = (0..p.rows() * p.cols).map(|_| (next() % (2 * q as u64 + 1)) as i8 - q).collect();
                    let scale = (0..p.rows() * p.cols / 32).map(|_| next() as u16 & 0xBF7F | 0x3000).collect();
                    (p, code, scale)
                })
                .collect();
            assert!(coded.len() + 1 <= c.packs().len() && !coded.is_empty());
            let mut held = Vec::new();
            for (p, code, scale) in &coded {
                let (mut at_c, mut at_s) = (0, 0);
                for (name, r) in &p.parts {
                    let m = Pack35::saved(name);
                    held.push((
                        format!("{m}.codes"),
                        "I8",
                        vec![*r as u64, p.cols as u64],
                        code[at_c..at_c + r * p.cols].iter().map(|&c| c as u8).collect::<Vec<u8>>(),
                    ));
                    held.push((
                        format!("{m}.scales"),
                        "BF16",
                        vec![*r as u64, p.cols as u64 / 32],
                        scale[at_s..at_s + r * p.cols / 32].iter().flat_map(|s| s.to_le_bytes()).collect(),
                    ));
                    (at_c, at_s) = (at_c + r * p.cols, at_s + r * p.cols / 32);
                }
            }
            let tensors: Vec<(&str, &str, Vec<u64>, &[u8])> = held.iter().map(|(n, d, s, b)| (n.as_str(), *d, s.clone(), &b[..])).collect();
            write_safetensors(&codes.join("model.safetensors"), &[("format", "pt".to_string())], &tensors).unwrap();
            let s = save::save_calibrated(&source, &out, 2, 5 << 30, layout, &plan, &codes).unwrap();
            assert_eq!(s.calibrated, coded.iter().map(|c| c.0.parts.len()).sum::<usize>());
            let (found, m) = crate::weights::v7_manifest(&out).unwrap().unwrap();
            assert_eq!(found, if q == 31 { crate::weights::Layout::Int6 } else { crate::weights::Layout::Int5 });
            assert_eq!(m.get("calibration"), Some(&glyd_gpu::json::parse(calibration).unwrap()));
            let saved = Checkpoint::open(&out).unwrap();
            for p in Config35::load(&out).unwrap().packs() {
                let key = Pack35::saved(&p.parts[0].0);
                let e = m.get("packs").and_then(|x| x.get(key)).unwrap_or_else(|| panic!("no pack {key}"));
                let lens = ["data", "blocks", "block_base"].map(|s| saved.get(&format!("{key}.glyd_{s}")).unwrap().1.bytes);
                crate::weights::check_v7(
                    key,
                    found,
                    p.rows(),
                    p.cols,
                    &e.get("shape").and_then(|s| s.as_u64s()).unwrap(),
                    &e.get(crate::weights::WORDS_KEY).and_then(|s| s.as_u64s()).unwrap(),
                    lens,
                )
                .unwrap();
            }
            for (p, code, scale) in &coded {
                let want = if q == 31 {
                    glyd_gpu::pack::pack_int6_runs_checked(code, scale, p.rows(), p.cols)
                } else {
                    glyd_gpu::pack::pack_int5_runs_checked(code, scale, p.rows(), p.cols)
                }
                .unwrap();
                let key = Pack35::saved(&p.parts[0].0);
                crate::weights::read_v7(&saved, &m, found, &[(key, p.rows())], p.cols).unwrap();
                assert!(
                    saved.read(&format!("{key}.glyd_data")).unwrap() == want.data && saved.read(&format!("{key}.glyd_blocks")).unwrap() == want.blocks,
                    "{key}"
                );
            }
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// sha256 in hex.
    #[cfg(feature = "dev")]
    fn sha(bytes: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Each file of the save in `out` by name, its sha256: glyd.json with the glyd that saved it and its source (a path of this run) written alike, so that a save
    /// is its packs' and tensors' bytes, their order and their entries.
    #[cfg(feature = "dev")]
    fn save_digests(out: &Path) -> std::collections::BTreeMap<String, String> {
        use glyd_gpu::json::Value;
        let mut files = std::collections::BTreeMap::new();
        for e in std::fs::read_dir(out).unwrap() {
            let p = e.unwrap().path();
            let name = p.file_name().unwrap().to_string_lossy().to_string();
            let mut bytes = std::fs::read(&p).unwrap();
            if name == "glyd.json" {
                let mut m = glyd_gpu::json::parse(std::str::from_utf8(&bytes).unwrap()).unwrap();
                if let Value::Object(o) = &mut m {
                    for (k, v) in o.iter_mut() {
                        match k.as_str() {
                            "glyd" => *v = Value::String("VERSION".into()),
                            "source" => *v = Value::Object(vec![("repo".into(), Value::String("MODEL".into())), ("revision".into(), Value::Null)]),
                            _ => {}
                        }
                    }
                }
                bytes = glyd_gpu::json::to_python(&m, 1).into_bytes();
            }
            files.insert(name, sha(&bytes));
        }
        files
    }

    /// The models of the saves the packer writes: each family's random model of the logits checks (`logits::random_model`'s case, or its maker's form) and the
    /// layouts each is packed at, those it is refused at among them (the levels a family is not saved at, a level its row does not run, and Int4H where no codes
    /// times a scale give the weights). GLM-5.3-Flash's row runs no block level: its lossy saves are refused (`a_level_its_row_does_not_run_is_not_packed`).
    #[cfg(feature = "dev")]
    fn saved_models() -> Vec<(&'static str, Vec<Layout>)> {
        use Layout::*;
        let lossy = || vec![Tiered, Int6, Int5, Int4, Int4H, Int3];
        let mut v = vec![("dense", vec![Tiered, Twelve, Compact, Int6, Int5, Int4, Int4H, Int3])];
        for c in ["qwen35", "qwen35-wide", "qwen35moe", "qwen35moe-st", "gemma4", "gemma4-kv", "nemotron", "nemotron-latent"] {
            v.push((c, lossy()));
        }
        for c in ["glm", "glm-fp8"] {
            v.push((c, vec![Tiered]));
        }
        for c in ["glm-dsa", "kimi", "muse", "deepseek-v4", "qwen4x"] {
            v.push((c, vec![Tiered, Int6]));
        }
        v
    }

    /// `pack` saves a family at the levels its row runs alone (src/support.rs `levels`): at another it refuses in the words `run` gives, before anything is
    /// read or written, whatever the family. GLM-5.3-Flash's row runs penguin and hummingbird: its block levels' saves (whose kv_b was the earlier engine's
    /// form, which nothing loads) are refused; the three-bit layout, no level, is packed where the row runs a block level (the plain dense kind's) and refused
    /// where it runs none.
    #[cfg(feature = "dev")]
    #[test]
    fn a_level_its_row_does_not_run_is_not_packed() {
        let root = std::env::temp_dir().join(format!("glyd-pack-levels-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let (glm, dense) = (root.join("glm"), root.join("dense"));
        crate::logits::random_model("glm", &glm).unwrap();
        crate::logits::random_model("dense", &dense).unwrap();
        let row = crate::support::row(Family::Glm);
        for (layout, level) in [(Layout::Int6, "kestrel"), (Layout::Int5, "linnet"), (Layout::Int4, "wren"), (Layout::Int4H, "wren"), (Layout::Int3, "i3")] {
            let out = root.join(format!("glm-{}", layout.name()));
            let e = run_with(glm.to_str().unwrap(), &out, 2, layout, true).unwrap_err().msg;
            let want = format!(
                "glm (GLM-5.3-Flash) runs at the penguin and hummingbird levels in this engine, not {level}: Run it lossless: glyd run {}",
                glm.display()
            );
            assert_eq!(e, want, "{level}");
            assert_eq!(row.pack_refusal(layout, &glm.display().to_string(), "glm").unwrap().to_string(), want);
            assert!(!out.exists(), "nothing written at {level}");
        }
        let dense_row = crate::support::row(Family::Qwen3);
        assert!(dense_row.pack_refusal(Layout::Int3, "d", "d").is_none() && dense_row.pack_refusal(Layout::Hummingbird, "d", "d").is_some());
        run_with(dense.to_str().unwrap(), &root.join("dense-i3"), 2, Layout::Int3, true).unwrap();
        // (every row: a level it does not run refused in run's words, one it runs not refused for its level)
        for r in &crate::support::TABLE {
            for layout in [Layout::Tiered, Layout::Int6, Layout::Int5, Layout::Int4, Layout::Hummingbird] {
                let level = crate::levels::name_of(layout);
                assert_eq!(r.pack_refusal(layout, "m", "m").map(|x| x.what), r.refusal("m", "m", level).map(|x| x.what), "{} {level}", r.name);
                assert_eq!(r.pack_refusal(layout, "m", "m").is_none(), r.allows(level), "{} {level}", r.name);
            }
        }
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// Every save the packer writes is the one recorded (tests/fixtures/pack_saves.json): each of `saved_models` packed by `run_with` at each of its layouts, every
    /// file of the save byte for byte (its sha256: `save_digests`) and the plan text it saves by (`level_plan`'s, at the levels a plan makes); a model refused is
    /// refused in the words recorded. And the plan text of each real config the packer's own tests read (the Qwen3.5 releases', Nemotron 3 Super's at each of its
    /// levels, GLM-5.3-Flash's, Gemma 4 12B's), by its sha256. A change that alters any of them fails here, by case and layout. A save meant to change (a new
    /// format version; a save made right) is recorded anew by running this test alone with GLYD_BLESS_PACKS=1, and its commit says which lines changed and why.
    /// GLYD_PACK_PLANS=DIR writes each plan text to DIR, to compare two trees' plans line by line.
    #[cfg(feature = "dev")]
    #[test]
    fn every_save_is_the_one_recorded() {
        use serde_json::{json, Map, Value};
        let root = std::env::temp_dir().join(format!("glyd-pack-saves-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let dump = std::env::var_os("GLYD_PACK_PLANS").map(std::path::PathBuf::from);
        if let Some(d) = &dump {
            std::fs::create_dir_all(d).unwrap();
        }
        let plan_digest = |name: &str, text: &str| {
            if let Some(d) = &dump {
                std::fs::write(d.join(format!("{name}.json")), text).unwrap();
            }
            sha(text.as_bytes())
        };
        let (mut saves, mut plans) = (Map::new(), Map::new());
        for (case, layouts) in saved_models() {
            let model = root.join(case);
            match case {
                "nemotron-latent" => crate::kcheck_nh::random(&model, 1024, 16).unwrap(),
                "glm-fp8" => crate::glm::random(&model, true, None).unwrap(),
                c => crate::logits::random_model(c, &model).unwrap(),
            }
            let config = ModelConfig::load(&model).ok();
            let words = |e: String| e.replace(&model.display().to_string(), "MODEL").replace(&root.display().to_string(), "ROOT");
            for layout in layouts {
                // (by `--layout`'s word: the compact layout's name in glyd.json is the tiered one's)
                let word = if layout == Layout::Compact { "a" } else { layout.name() };
                let name = format!("{case} {word}");
                let out = root.join(format!("{case}-{word}"));
                let entry = match run_with(model.to_str().unwrap(), &out, 4, layout, true) {
                    Ok(()) => {
                        let mut e = json!({ "files": save_digests(&out) });
                        // (the plan text, where the save is a planned one: a lossy level's, or penguin's glyd-v7)
                        let planned = matches!(layout, Layout::Int6 | Layout::Int5 | Layout::Int4 | Layout::Int4H | Layout::Int3)
                            || crate::weights::v7_manifest(&out).ok().flatten().is_some();
                        if planned {
                            let text = level_plan(&model, layout, config.as_ref(), None).unwrap().expect("a planned save's plan");
                            e["plan"] = Value::String(plan_digest(&format!("{case}-{word}"), &text));
                        }
                        e
                    }
                    Err(e) => json!({ "refused": words(e.msg) }),
                };
                let _ = std::fs::remove_dir_all(&out);
                saves.insert(name, entry);
            }
            if case == "nemotron-latent" {
                // (and at the hummingbird level from a random export of its experts, as packcheck's hummingbird line packs it: src/hbpack.rs)
                let (export, out) = (root.join("export"), root.join("nemotron-latent-hb"));
                let c = ConfigNemotronH::load(&model).unwrap();
                crate::hbpack::toy_export(&c, &export, 1, 40).unwrap();
                crate::hbpack::run(model.to_str().unwrap(), &out, 4, &export).unwrap();
                let text = crate::hbpack::plan(&model).unwrap();
                saves.insert("nemotron-latent hb".into(), json!({ "files": save_digests(&out), "plan": plan_digest("nemotron-latent-hb", &text) }));
                let _ = std::fs::remove_dir_all(&out);
            }
            let _ = std::fs::remove_dir_all(&model);
        }
        // (the real configs' plans, from their configs, the tensors' dtypes the fixtures hold, and their shapes where the description gives none)
        let all = glyd_gpu::json::parse(include_str!("../tests/fixtures/qwen35_configs.json")).unwrap();
        let have = glyd_gpu::json::parse(include_str!("../tests/fixtures/qwen35_tensors.json")).unwrap();
        for (name, v) in all.as_object().unwrap() {
            let text = glyd_gpu::json::to_compact(v);
            let tensors = have.get(name).and_then(|t| t.as_object());
            let f32 = |n: &str| tensors.is_some_and(|t| t.iter().any(|(k, v)| k == n && v.as_array().unwrap()[0].as_str() == Some("F32")));
            let made = plan(&ModelConfig::parse(&text).unwrap(), Layout::Int6, &form(Family::Qwen35), None, &f32, &|_| None).map_err(|e| e.msg);
            plans.insert(name.clone(), Value::String(made.map_or_else(|e| format!("refused: {e}"), |t| plan_digest(&name.replace('/', "_"), &t))));
        }
        let index = |t: &[(String, String, Vec<u64>)]| -> std::collections::HashMap<String, (String, Vec<usize>)> {
            t.iter().map(|(n, d, s)| (n.clone(), (d.clone(), s.iter().map(|&x| x as usize).collect()))).collect()
        };
        for (fixture, text, tensors, family, levels) in [
            (
                "nemotron_h_config.json",
                include_str!("../tests/fixtures/nemotron_h_config.json"),
                crate::nemotron::tests::tensors(),
                Family::NemotronH,
                &[Layout::Int6, Layout::Int5, Layout::Int4, Layout::Int3][..],
            ),
            ("glm5_next_config.json", include_str!("../tests/fixtures/glm5_next_config.json"), crate::glm::tests::tensors(), Family::Glm, &[Layout::Int6][..]),
            ("gemma4_12b_config.json", include_str!("../tests/fixtures/gemma4_12b_config.json"), Vec::new(), Family::Gemma4, &[Layout::Tiered][..]),
        ] {
            let t = index(&tensors);
            let (f32, shape) = (|n: &str| t.get(n).is_some_and(|x| x.0 == "F32"), |n: &str| t.get(n).map(|x| x.1.clone()));
            for &level in levels {
                let text =
                    plan(&ModelConfig::parse(text).unwrap(), level, &form(family), None, &f32, &shape).unwrap_or_else(|e| panic!("{fixture}: {}", e.msg));
                let name = if family == Family::NemotronH { format!("{fixture} {}", level.name()) } else { fixture.to_string() };
                let dumped = match family {
                    Family::NemotronH => format!("nemotron-real-{}", level.name()),
                    Family::Glm => "glm-real".into(),
                    _ => "gemma4-real".into(),
                };
                plans.insert(name, Value::String(plan_digest(&dumped, &text)));
            }
        }
        let _ = std::fs::remove_dir_all(&root);
        let now = json!({ "saves": saves, "plans": plans });
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pack_saves.json");
        if std::env::var_os("GLYD_BLESS_PACKS").is_some() {
            std::fs::write(&path, serde_json::to_string_pretty(&now).unwrap() + "\n").unwrap();
            return;
        }
        let was: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let mut differ = Vec::new();
        for part in ["saves", "plans"] {
            let (a, b) = (was[part].as_object().unwrap(), now[part].as_object().unwrap());
            differ.extend(
                b.iter()
                    .filter(|(k, v)| a.get(*k) != Some(*v))
                    .map(|(k, v)| format!("{part}: {k}: {v} where {} is recorded", a.get(k).unwrap_or(&Value::Null))),
            );
            differ.extend(a.keys().filter(|k| !b.contains_key(*k)).map(|k| format!("{part}: {k}: recorded, and not made now")));
        }
        assert!(differ.is_empty(), "the saves are not the ones recorded (tests/fixtures/pack_saves.json):\n{}", differ.join("\n"));
    }

    /// A plain dense model (the logits checks' random one) packed at penguin is a glyd-v7 save the executor reads as saved: each matrix of its lowered description (q, k
    /// and v under q's name; gate and up under gate's; the embedding, the head) a lossless pack under its first member's name at the description's rows and columns, each small
    /// tensor copied as bf16, and no sidecar of the earlier dense engine's.
    #[cfg(feature = "dev")]
    #[test]
    fn a_dense_model_at_penguin_is_a_glyd_v7_save_of_its_description() {
        let dir = std::env::temp_dir().join(format!("glyd-pack-dense-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (src, out) = (dir.join("src"), dir.join("out"));
        crate::logits::dense_random(&src).unwrap();
        run_with(src.to_str().unwrap(), &out, 2, Layout::Tiered, true).unwrap();
        let (layout, m) = crate::weights::v7_manifest(&out).unwrap().expect("a glyd-v7 save");
        assert_eq!(layout, Layout::Tiered);
        assert!(!out.join("engine.safetensors").exists());
        let d = crate::dynamic::dense::lowered(&out).unwrap();
        assert_eq!(d.mats.len(), 4 * 4 + 2, "four matrices a layer, the embedding and the head");
        let ck = Checkpoint::open(&out).unwrap();
        for it in &d.mats {
            let key = it.members[0].trim_end_matches(".weight");
            crate::weights::read_lossless(&ck, &m, key, it.rows, it.cols).unwrap_or_else(|e| panic!("{key}: {}", e.msg));
        }
        for n in &d.small {
            assert_eq!(ck.get(n).unwrap_or_else(|| panic!("no {n}")).1.dtype, "BF16", "{n}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// MiMo's small model (the logits checks' random one) packed at kestrel is a glyd-v7 save the executor's lowering reads as saved: each layer's fused q-k-v
    /// matrix, which the description cuts into the three roles, is one pack under its own name at its stored rows (once, not a time a role), and every
    /// other matrix of the lowered description a pack under its first member's name; each routed expert's two packs are there too.
    #[cfg(feature = "dev")]
    #[test]
    fn a_fused_matrix_cut_in_shards_is_saved_once_as_stored() {
        let dir = std::env::temp_dir().join(format!("glyd-pack-mimo-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (src, out) = (dir.join("src"), dir.join("out"));
        crate::logits::mimo_random(&src).unwrap();
        run_with(src.to_str().unwrap(), &out, 2, Layout::Int6, true).unwrap();
        let (layout, m) = crate::weights::v7_manifest(&out).unwrap().expect("a glyd-v7 save");
        assert_eq!(layout, Layout::Int6);
        let d = crate::dynamic::dense::lowered(&out).unwrap();
        let ck = Checkpoint::open(&out).unwrap();
        let mut fused = Vec::new();
        for it in &d.mats {
            let key = it.members[0].trim_end_matches(".weight");
            // (a merge of two tensors, the dense MLP's gate and up, is one pack under the first's name)
            let p = crate::weights::read_v7(&ck, &m, Layout::Int6, &[(key, it.rows)], it.cols).unwrap_or_else(|e| panic!("{key}: {}", e.msg));
            assert_eq!((p.rows, p.cols), (it.rows, it.cols), "{key}");
            if key.ends_with("qkv_proj") {
                fused.push(it.rows);
            }
        }
        assert_eq!(fused, [2176, 2816, 2816, 2176]);
        let packs = m.get("packs").and_then(|p| p.as_object()).unwrap();
        assert_eq!(packs.iter().filter(|(k, _)| k.contains("qkv_proj")).count(), 4);
        assert_eq!(packs.iter().filter(|(k, _)| k.contains(".mlp.experts.")).count(), 3 * 8 * 2);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
