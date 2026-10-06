//! Model files (safetensors, GGUF): every tensor whose element type this knows is
//! cut into streams of symbols (a float's exponent, sign and mantissa; a quantised
//! block's scales and codes) that are coded with tables chosen for the tensor, under
//! a context where one pays; everything else (the header, the padding, a tensor of a
//! type this does not know, a tensor too small to pay for its tables) is kept and
//! goes through the level the caller asked for. Every byte of the file is
//! accounted for, so the file comes back exactly, whatever it holds.
//!
//! Envelope (integers are varints unless said otherwise):
//!
//! ```text
//! "GLYDWGT1"  original_len  n_regions  kept_len  kept_stream_len
//! per region: gap (kept bytes since the end of the one before)  len  kind (u8)  payload_len
//! the kept stream: a stream of any level holding the kept bytes in file order
//! the region payloads, back to back
//!
//! a region's payload:
//!   variant (u8)  n_params  params  n_streams (u8)
//!   per stream: mode (u8): 0 raw | 1 constant (+ the symbol) | 2 bits | 3 coded (+ its tables)
//!   n_chunks  per chunk: payload length
//!   the chunks: per stream: length, bytes; then the CRC-32C of the chunk's bytes (u32)
//! ```
//!
//! A chunk is the bytes of up to `CHUNK_BYTES` of whole units. `src/weights/rans.rs`
//! gives the coding of a stream, `kinds.rs` the streams of each kind.

pub(crate) mod kinds;
mod parse;
pub(crate) mod rans;

use crate::error::{CodecError, Result};
use crate::record::{get_varint, put_varint};
use kinds::{layout, Kind, Spec, StreamSpec};
use rans::{DecTables, Tables};
use std::cell::RefCell;
use std::sync::OnceLock;

pub(crate) const MAGIC: &[u8; 8] = b"GLYDWGT1";

fn bad(msg: &'static str) -> CodecError {
    CodecError::CorruptedBitstream(msg)
}

/// Whether `input` is a file this mode reads the tensors of.
pub fn is_model_file(input: &[u8]) -> bool {
    parse::is_gguf(input) || crate::safetensors::is_safetensors(input)
}

// ------------------------------------------------------------------ stream modes

#[derive(Clone, Debug, PartialEq, Eq)]
enum Mode {
    Raw,
    Const(u8),
    Bits1,
    Rans(Tables),
}

fn write_mode(m: &Mode, out: &mut Vec<u8>) {
    match m {
        Mode::Raw => out.push(0),
        Mode::Const(v) => {
            out.push(1);
            out.push(*v);
        }
        Mode::Bits1 => out.push(2),
        Mode::Rans(t) => {
            out.push(3);
            rans::write_tables(t, out);
        }
    }
}

fn read_mode(src: &[u8], pos: &mut usize, s: &StreamSpec) -> Result<Mode> {
    let tag = *src.get(*pos).ok_or(bad("weights: truncated stream modes"))?;
    *pos += 1;
    Ok(match tag {
        0 => Mode::Raw,
        1 => {
            let v = *src.get(*pos).ok_or(bad("weights: truncated stream modes"))?;
            *pos += 1;
            if v as usize >= s.alphabet.max(2) && !s.bits1 {
                return Err(bad("weights: constant symbol"));
            }
            Mode::Const(v)
        }
        2 if s.bits1 => Mode::Bits1,
        3 if !s.bits1 => Mode::Rans(rans::read_tables(src, pos, s.alphabet, s.nctx())?),
        _ => return Err(bad("weights: stream mode")),
    })
}

// ------------------------------------------------------------------ encoding

/// A region's coding, chosen: its spec and the mode of every stream.
struct Plan {
    spec: Spec,
    layout: Vec<StreamSpec>,
    modes: Vec<Mode>,
    lookup: Option<Vec<u8>>,
    /// Bytes the region is expected to take.
    est: usize,
}

fn chunks_of<'a>(data: &'a [u8], spec: &Spec) -> Vec<&'a [u8]> {
    data.chunks(spec.chunk_units() * spec.unit_bytes()).collect()
}

/// Histograms (per stream, `nctx * alphabet` counts) of one chunk under `layout`.
fn hists_of(spec: &Spec, layout: &[StreamSpec], lookup: Option<&[u8]>, chunk: &[u8], scratch: &mut Vec<Vec<u8>>) -> Vec<Vec<u32>> {
    scratch.resize_with(layout.len(), Vec::new);
    kinds::split(spec, chunk, scratch);
    let aux = if layout.iter().any(|s| matches!(s.ctx, kinds::CtxSpec::Aux(_))) { kinds::aux(spec, lookup.unwrap(), &[&scratch[0][..], &scratch[1][..]]) } else { Vec::new() };
    let mut out = Vec::with_capacity(layout.len());
    for (s, sp) in layout.iter().enumerate() {
        if sp.bits1 {
            out.push(Vec::new());
            continue;
        }
        let nctx = sp.nctx();
        let mut h = vec![0u32; nctx * sp.alphabet];
        match kinds::contexts(layout, s, scratch, &aux) {
            None => {
                for &x in &scratch[s] {
                    h[x as usize] += 1;
                }
            }
            Some(ctx) => {
                for (&x, &c) in scratch[s].iter().zip(&ctx) {
                    h[(c as usize).min(nctx - 1) * sp.alphabet + x as usize] += 1;
                }
            }
        }
        out.push(h);
    }
    out
}

fn add_hists(total: &mut Vec<Vec<u32>>, more: Vec<Vec<u32>>) {
    if total.is_empty() {
        *total = more;
        return;
    }
    for (t, m) in total.iter_mut().zip(more) {
        for (a, b) in t.iter_mut().zip(m) {
            *a += b;
        }
    }
}

/// The modes of the streams for histograms taken over `sampled` of `all` units, and the
/// bytes the region would take.
fn choose_modes(layout: &[StreamSpec], hists: &[Vec<u32>], scale: f64, total_units: usize) -> (Vec<Mode>, f64) {
    let mut modes = Vec::with_capacity(layout.len());
    let mut bits = 0.0;
    for (sp, h) in layout.iter().zip(hists) {
        if sp.bits1 {
            modes.push(Mode::Bits1);
            bits += total_units as f64;
            continue;
        }
        let nctx = sp.nctx();
        let n: u64 = h.iter().map(|&x| x as u64).sum();
        // marginal over contexts
        let mut marg = vec![0u32; sp.alphabet];
        for c in 0..nctx {
            for s in 0..sp.alphabet {
                marg[s] = marg[s].saturating_add(h[c * sp.alphabet + s]);
            }
        }
        let nnz = marg.iter().filter(|&&x| x > 0).count();
        if n == 0 {
            modes.push(Mode::Raw);
            continue;
        }
        if nnz == 1 {
            modes.push(Mode::Const(marg.iter().position(|&x| x > 0).unwrap() as u8));
            bits += 16.0;
            continue;
        }
        // Many contexts' tables are many cache lines a symbol: coarser ones, which cost nothing visible.
        let (t, dbits, tbits) = rans::plan(h, nctx, sp.alphabet, if nctx > 64 { 10 } else { rans::MAX_BITS });
        let est = dbits * scale + tbits;
        let raw = 8.0 * n as f64 * scale;
        if est < 0.99 * raw {
            bits += est;
            modes.push(Mode::Rans(t));
        } else {
            bits += raw;
            modes.push(Mode::Raw);
        }
    }
    (modes, bits)
}

/// The exponent most bf16 elements of `chunk` have.
fn bf16_emode(chunk: &[u8]) -> u8 {
    let mut h = [0u32; 256];
    for c in chunk.chunks_exact(2).take(1 << 16) {
        h[(u16::from_le_bytes([c[0], c[1]]) >> 7) as usize & 0xFF] += 1;
    }
    (0..256).max_by_key(|&i| h[i]).unwrap_or(0) as u8
}

/// The block scales of a Q8_0 chunk that many blocks have, by how many.
fn q8_classes(sample: &[&[u8]]) -> Vec<u8> {
    let mut count = vec![0u32; 1 << 16];
    let mut blocks = 0usize;
    for chunk in sample {
        for b in chunk.chunks_exact(34) {
            count[u16::from_le_bytes([b[0], b[1]]) as usize] += 1;
            blocks += 1;
        }
    }
    let mut ds: Vec<usize> = (0..1 << 16).filter(|&d| count[d] as usize >= (blocks / 4096).max(4)).collect();
    ds.sort_by_key(|&d| std::cmp::Reverse(count[d]));
    ds.truncate(255);
    let covered: usize = ds.iter().map(|&d| count[d] as usize).sum();
    if covered * 10 < blocks * 3 {
        return Vec::new();
    }
    ds.iter().flat_map(|&d| (d as u16).to_le_bytes()).collect()
}

/// Entropy of the sign-and-mantissa byte of the bf16 elements of `sample`, in bits.
fn bf16_sm_entropy(sample: &[&[u8]]) -> f64 {
    let mut h = [0u64; 256];
    let mut n = 0u64;
    for chunk in sample {
        for c in chunk.chunks_exact(2).take(1 << 18) {
            h[((c[1] & 0x80) | (c[0] & 0x7F)) as usize] += 1;
            n += 1;
        }
    }
    h.iter().filter(|&&x| x > 0).map(|&x| -(x as f64) * (x as f64 / n as f64).log2()).sum::<f64>() / n.max(1) as f64
}

fn candidates(spec: &Spec, sample: &[&[u8]]) -> Vec<Spec> {
    match spec.kind {
        Kind::Bf16 => {
            let emode = bf16_emode(sample[0]);
            let mut v = vec![Spec::new(Kind::Bf16)];
            // the mantissa before says something only where the mantissas sit on a grid
            let flags: &[u8] = if bf16_sm_entropy(sample) < 7.7 { &[2, 3] } else { &[2] };
            for &f in flags {
                v.push(Spec { kind: Kind::Bf16, variant: 1, params: vec![f, emode] });
            }
            v
        }
        Kind::Q8_0 => {
            let mut v = vec![Spec::new(Kind::Q8_0), Spec { kind: Kind::Q8_0, variant: 2, params: Vec::new() }];
            let classes = q8_classes(sample);
            if !classes.is_empty() {
                v.push(Spec { kind: Kind::Q8_0, variant: 1, params: classes });
            }
            v
        }
        _ => vec![spec.clone()],
    }
}

/// Estimated bytes of `chunks` under `spec` (the sample's streams scaled to `total_units`), and
/// the modes that estimate chose.
fn estimate(spec: &Spec, chunks: &[&[u8]], total_units: usize) -> Option<(Vec<Mode>, f64, Vec<StreamSpec>, Option<Vec<u8>>)> {
    let l = layout(spec)?;
    let lookup = if spec.kind == Kind::Q8_0 && spec.variant == 1 { kinds::scale_classes(&spec.params) } else { None };
    let mut hists = Vec::new();
    let mut scratch = Vec::new();
    let mut units = 0usize;
    for c in chunks {
        add_hists(&mut hists, hists_of(spec, &l, lookup.as_deref(), c, &mut scratch));
        units += c.len() / spec.unit_bytes();
    }
    let scale = total_units as f64 / units.max(1) as f64;
    let (modes, bits) = choose_modes(&l, &hists, scale, total_units);
    Some((modes, bits, l, lookup))
}

/// The coding of one run of a file (`None` when it would not pay), by the histograms of all
/// its chunks. The layout among a kind's variants is chosen on a sample.
fn plan_region(data: &[u8], spec0: &Spec) -> Option<Plan> {
    let ub = spec0.unit_bytes();
    let total_units = data.len() / ub;
    let chunks = chunks_of(data, spec0);
    // what the variants are tried on: up to three pieces of the run, a quarter of a chunk or so each
    let piece = (spec0.chunk_units() / 2).max(1) * ub;
    let sample: Vec<&[u8]> = if chunks.len() <= 3 { chunks.iter().map(|c| &c[..c.len().min(piece)]).collect() } else { (0..3).map(|i| { let c = chunks[i * (chunks.len() - 1) / 2]; &c[..c.len().min(piece)] }).collect() };
    let cands = candidates(spec0, &sample);
    let spec = if cands.len() == 1 {
        cands[0].clone()
    } else {
        // The first candidate is the cheapest to decode: another has to take 2% fewer bytes to be chosen.
        let mut best: Option<(f64, Spec)> = None;
        for (k, c) in cands.into_iter().enumerate() {
            if let Some((_, bits, _, _)) = estimate(&c, &sample, total_units) {
                let bits = if k == 0 { bits } else { bits * 1.002 };
                if best.as_ref().map_or(true, |b| bits < b.0) {
                    best = Some((bits, c));
                }
            }
        }
        best?.1
    };
    // the histograms of the whole run under the chosen layout
    let l = layout(&spec)?;
    let lookup = if spec.kind == Kind::Q8_0 && spec.variant == 1 { kinds::scale_classes(&spec.params) } else { None };
    let mut hists = Vec::new();
    let mut scratch = Vec::new();
    for c in &chunks {
        add_hists(&mut hists, hists_of(&spec, &l, lookup.as_deref(), c, &mut scratch));
    }
    let (modes, bits) = choose_modes(&l, &hists, 1.0, total_units);
    // chunk headers: a length and a checksum, and a length for every stream
    let overhead = chunks.len() * (4 + 3 * l.len()) + 64;
    let est = (bits / 8.0) as usize + overhead;
    if est as f64 >= data.len() as f64 * 0.985 {
        return None;
    }
    Some(Plan { spec, layout: l, modes, lookup, est })
}

/// One chunk of a run, coded.
fn encode_chunk(plan: &Plan, chunk: &[u8], scratch: &mut Vec<Vec<u8>>, out: &mut Vec<u8>) {
    let l = &plan.layout;
    scratch.resize_with(l.len(), Vec::new);
    kinds::split(&plan.spec, chunk, scratch);
    let aux = if l.iter().any(|s| matches!(s.ctx, kinds::CtxSpec::Aux(_))) { kinds::aux(&plan.spec, plan.lookup.as_deref().unwrap(), &[&scratch[0][..], &scratch[1][..]]) } else { Vec::new() };
    let mut body = Vec::new();
    for (s, m) in plan.modes.iter().enumerate() {
        body.clear();
        match m {
            Mode::Raw | Mode::Bits1 => body.extend_from_slice(&scratch[s]),
            Mode::Const(_) => {}
            Mode::Rans(t) => {
                let ctx = kinds::contexts(l, s, scratch, &aux);
                rans::encode(&scratch[s], ctx.as_deref(), t, &mut body);
            }
        }
        put_varint(out, body.len() as u64);
        out.extend_from_slice(&body);
    }
    out.extend_from_slice(&crate::format::crc32c(chunk).to_le_bytes());
}

thread_local! {
    static SCRATCH: RefCell<Vec<Vec<u8>>> = const { RefCell::new(Vec::new()) };
    static ARENA: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}


/// `input` as an envelope, its model-file tensors coded and the rest compressed by `kept_level`;
/// `false` (nothing written) for anything else, or where it would not pay.
pub(crate) fn wrap(input: &[u8], output: &mut Vec<u8>, kept_level: impl Fn(&[u8], &mut Vec<u8>)) -> bool {
    if crate::in_part() || input.len() < 64 << 10 {
        return false;
    }
    let Some(cands) = parse::candidates(input) else { return false };
    let run_bytes: usize = cands.iter().map(|c| c.len).sum();
    if run_bytes * 4 < input.len() {
        return false;
    }
    // Phase 1: the coding of each run, runs in parallel.
    let plans: Vec<std::sync::Mutex<Option<Plan>>> = cands.iter().map(|_| std::sync::Mutex::new(None)).collect();
    let _ = crate::par_units::<()>(cands.len(), |i| {
        let c = &cands[i];
        let p = crate::as_part(|| plan_region(&input[c.start..c.start + c.len], &c.spec));
        *plans[i].lock().unwrap() = p;
        Ok(())
    });
    let plans: Vec<Option<Plan>> = plans.into_iter().map(|m| m.into_inner().unwrap()).collect();
    let coded_bytes: usize = cands.iter().zip(&plans).filter(|(_, p)| p.is_some()).map(|(c, _)| c.len).sum();
    let est_bytes: usize = plans.iter().flatten().map(|p| p.est).sum();
    if coded_bytes * 10 < input.len() || est_bytes + coded_bytes / 50 >= coded_bytes {
        return false;
    }
    // Phase 2: the chunks, in parallel.
    struct Task {
        region: usize,
        at: usize,
        len: usize,
    }
    let mut tasks = Vec::new();
    for (ri, (c, p)) in cands.iter().zip(&plans).enumerate() {
        if let Some(p) = p {
            let step = p.spec.chunk_units() * p.spec.unit_bytes();
            let mut at = 0;
            while at < c.len {
                tasks.push(Task { region: ri, at, len: step.min(c.len - at) });
                at += step;
            }
        }
    }
    let results: Vec<std::sync::Mutex<Vec<u8>>> = tasks.iter().map(|_| std::sync::Mutex::new(Vec::new())).collect();
    let _ = crate::par_units::<()>(tasks.len(), |ti| {
        let t = &tasks[ti];
        let c = &cands[t.region];
        let p = plans[t.region].as_ref().unwrap();
        let chunk = &input[c.start + t.at..c.start + t.at + t.len];
        let mut out = Vec::with_capacity(t.len / 2 + 64);
        SCRATCH.with_borrow_mut(|s| encode_chunk(p, chunk, s, &mut out));
        *results[ti].lock().unwrap() = out;
        Ok(())
    });
    let results: Vec<Vec<u8>> = results.into_iter().map(|m| m.into_inner().unwrap()).collect();
    // The regions' payloads, and the bytes they leave.
    let mut payloads: Vec<(usize, Vec<u8>)> = Vec::new(); // (candidate index, payload)
    let mut at_task = 0;
    for (ri, p) in plans.iter().enumerate() {
        let Some(p) = p else { continue };
        let mut n = 0;
        while at_task + n < tasks.len() && tasks[at_task + n].region == ri {
            n += 1;
        }
        let chunks = &results[at_task..at_task + n];
        at_task += n;
        let mut pay = Vec::with_capacity(chunks.iter().map(|c| c.len()).sum::<usize>() + 256);
        pay.push(p.spec.variant);
        put_varint(&mut pay, p.spec.params.len() as u64);
        pay.extend_from_slice(&p.spec.params);
        pay.push(p.layout.len() as u8);
        for m in &p.modes {
            write_mode(m, &mut pay);
        }
        put_varint(&mut pay, chunks.len() as u64);
        for c in chunks {
            put_varint(&mut pay, c.len() as u64);
        }
        for c in chunks {
            pay.extend_from_slice(c);
        }
        payloads.push((ri, pay));
    }
    let mut kept = Vec::with_capacity(input.len() - coded_bytes);
    let mut table = Vec::new();
    let mut last = 0usize;
    for (ri, pay) in &payloads {
        let c = &cands[*ri];
        kept.extend_from_slice(&input[last..c.start]);
        put_varint(&mut table, (c.start - last) as u64);
        put_varint(&mut table, c.len as u64);
        table.push(c.spec.kind as u8);
        put_varint(&mut table, pay.len() as u64);
        last = c.start + c.len;
    }
    kept.extend_from_slice(&input[last..]);
    let mut kept_stream = Vec::with_capacity(kept.len() / 3 + 64);
    crate::as_part(|| kept_level(&kept, &mut kept_stream));
    let before = output.len();
    output.extend_from_slice(MAGIC);
    put_varint(output, input.len() as u64);
    put_varint(output, payloads.len() as u64);
    put_varint(output, kept.len() as u64);
    put_varint(output, kept_stream.len() as u64);
    output.extend_from_slice(&table);
    output.extend_from_slice(&kept_stream);
    for (_, pay) in &payloads {
        output.extend_from_slice(pay);
    }
    if output.len() - before >= input.len() {
        output.truncate(before);
        return false;
    }
    true
}

// ------------------------------------------------------------------ decoding

struct RegionHdr<'a> {
    start: usize,
    len: usize,
    spec: Spec,
    layout: Vec<StreamSpec>,
    modes: Vec<Mode>,
    chunk_lens: Vec<usize>,
    chunks: &'a [u8],
    /// The decoding tables of the coded streams, built when a worker first needs them.
    tables: OnceLock<Vec<Option<DecTables>>>,
}

pub(crate) fn original_len(compressed: &[u8]) -> Option<usize> {
    if compressed.len() < 10 || &compressed[..8] != MAGIC {
        return None;
    }
    let mut pos = 8;
    get_varint(compressed, &mut pos).ok().and_then(|v| usize::try_from(v).ok())
}

/// The file an envelope holds, its kept bytes decoded by `kept`; `None` for anything that is not one.
pub(crate) fn unwrap(compressed: &[u8], kept: impl FnOnce(&[u8]) -> Result<Vec<u8>>) -> Option<Result<Vec<u8>>> {
    if compressed.len() < 9 || &compressed[..8] != MAGIC {
        return None;
    }
    Some(unwrap_envelope(compressed, kept))
}

fn unwrap_envelope(compressed: &[u8], kept: impl FnOnce(&[u8]) -> Result<Vec<u8>>) -> Result<Vec<u8>> {
    let mut pos = 8usize;
    let original = usize::try_from(get_varint(compressed, &mut pos)?).map_err(|_| bad("weights: length"))?;
    let n_regions = get_varint(compressed, &mut pos)? as usize;
    let kept_len = get_varint(compressed, &mut pos)? as usize;
    let kept_stream_len = get_varint(compressed, &mut pos)? as usize;
    // A stream of constants is a handful of bytes for a megabyte; nothing real is a hundred thousand times smaller.
    if n_regions > compressed.len() || kept_len > original || original > 1 << 44 || original as u128 > (compressed.len() as u128 + 4096) * 200_000 {
        return Err(bad("weights: envelope"));
    }
    struct Row {
        gap: usize,
        len: usize,
        kind: u8,
        payload: usize,
    }
    let mut rows = Vec::with_capacity(n_regions);
    let (mut at, mut coded, mut gaps, mut payload_total) = (0usize, 0usize, 0usize, 0usize);
    for _ in 0..n_regions {
        let gap = get_varint(compressed, &mut pos)? as usize;
        let len = get_varint(compressed, &mut pos)? as usize;
        let kind = *compressed.get(pos).ok_or(bad("weights: truncated table"))?;
        pos += 1;
        let payload = get_varint(compressed, &mut pos)? as usize;
        at = at.checked_add(gap).and_then(|a| a.checked_add(len)).ok_or(bad("weights: region table"))?;
        gaps = gaps.checked_add(gap).ok_or(bad("weights: region table"))?;
        coded = coded.checked_add(len).ok_or(bad("weights: region table"))?;
        payload_total = payload_total.checked_add(payload).ok_or(bad("weights: region table"))?;
        rows.push(Row { gap, len, kind, payload });
    }
    if at > original || gaps > kept_len || coded > original || kept_len + coded != original {
        return Err(bad("weights: region table"));
    }
    let kept_stream = compressed.get(pos..pos.checked_add(kept_stream_len).ok_or(bad("weights: kept stream"))?).ok_or(bad("weights: truncated kept stream"))?;
    pos += kept_stream_len;
    if compressed.len() - pos != payload_total {
        return Err(bad("weights: payloads"));
    }
    let kept = kept(kept_stream)?;
    if kept.len() != kept_len {
        return Err(bad("weights: kept length"));
    }
    let mut out = Vec::new();
    out.try_reserve_exact(original).map_err(|_| bad("weights: output too large"))?;
    out.resize(original, 0);
    // regions' headers, and the kept bytes laid between them
    let mut hdrs: Vec<RegionHdr> = Vec::with_capacity(n_regions);
    let (mut at, mut kat) = (0usize, 0usize);
    for r in &rows {
        out[at..at + r.gap].copy_from_slice(&kept[kat..kat + r.gap]);
        at += r.gap;
        kat += r.gap;
        let pay = &compressed[pos..pos + r.payload];
        pos += r.payload;
        hdrs.push(read_region(pay, r.kind, at, r.len)?);
        at += r.len;
    }
    out[at..].copy_from_slice(&kept[kat..]);
    // the chunks, in parallel
    struct Task {
        region: usize,
        out_at: usize,
        units: usize,
        payload: (usize, usize),
    }
    let mut tasks = Vec::new();
    for (ri, h) in hdrs.iter().enumerate() {
        let step = h.spec.chunk_units();
        let ub = h.spec.unit_bytes();
        let mut p = 0usize;
        let mut done = 0usize;
        for (ci, &cl) in h.chunk_lens.iter().enumerate() {
            let units = step.min(h.len / ub - done);
            if units == 0 || ci * step != done {
                return Err(bad("weights: chunks"));
            }
            tasks.push(Task { region: ri, out_at: h.start + done * ub, units, payload: (p, cl) });
            p += cl;
            done += units;
        }
        if done * ub != h.len || p != h.chunks.len() {
            return Err(bad("weights: chunks"));
        }
    }
    let base = out.as_mut_ptr() as usize;
    crate::par_units(tasks.len(), |ti| -> Result<()> {
        let t = &tasks[ti];
        let h = &hdrs[t.region];
        let tabs = h.tables.get_or_init(|| h.modes.iter().map(|m| if let Mode::Rans(tb) = m { Some(DecTables::new(tb)) } else { None }).collect());
        // Tasks cover disjoint ranges of `out`.
        let dst = unsafe { std::slice::from_raw_parts_mut((base + t.out_at) as *mut u8, t.units * h.spec.unit_bytes()) };
        let payload = &h.chunks[t.payload.0..t.payload.0 + t.payload.1];
        ARENA.with_borrow_mut(|s| decode_chunk(h, tabs, payload, t.units, dst, s))
    })?;
    Ok(out)
}

fn read_region<'a>(pay: &'a [u8], kind: u8, start: usize, len: usize) -> Result<RegionHdr<'a>> {
    let kind = Kind::from_code(kind).ok_or(bad("weights: region kind"))?;
    let mut pos = 0usize;
    let variant = *pay.first().ok_or(bad("weights: truncated region"))?;
    pos += 1;
    let np = get_varint(pay, &mut pos)? as usize;
    let params = pay.get(pos..pos.checked_add(np).ok_or(bad("weights: region"))?).ok_or(bad("weights: truncated region"))?.to_vec();
    pos += np;
    let spec = Spec { kind, variant, params };
    let ub = spec.unit_bytes();
    if ub == 0 || len % ub != 0 {
        return Err(bad("weights: region length"));
    }
    if kind == Kind::Q8_0 && variant == 1 && kinds::scale_classes(&spec.params).is_none() {
        return Err(bad("weights: region parameters"));
    }
    let l = layout(&spec).ok_or(bad("weights: region layout"))?;
    let ns = *pay.get(pos).ok_or(bad("weights: truncated region"))? as usize;
    pos += 1;
    if ns != l.len() {
        return Err(bad("weights: stream count"));
    }
    let mut modes = Vec::with_capacity(ns);
    for s in &l {
        modes.push(read_mode(pay, &mut pos, s)?);
    }
    let nchunks = get_varint(pay, &mut pos)? as usize;
    if nchunks > pay.len() {
        return Err(bad("weights: chunk count"));
    }
    let mut chunk_lens = Vec::with_capacity(nchunks);
    for _ in 0..nchunks {
        chunk_lens.push(get_varint(pay, &mut pos)? as usize);
    }
    Ok(RegionHdr { start, len, spec, layout: l, modes, chunk_lens, chunks: &pay[pos..], tables: OnceLock::new() })
}

/// Bytes between the buffers of a chunk's streams: no two are a multiple of 4 KB apart, or the loop that
/// writes one while it reads another waits on stores that look as though they overlap (the page offsets
/// of an array and of the symbols made from it must differ: a third of the time otherwise).
fn stagger(k: usize) -> usize {
    96 + 64 * k
}

fn decode_chunk(h: &RegionHdr, tabs: &[Option<DecTables>], payload: &[u8], units: usize, dst: &mut [u8], arena: &mut Vec<u8>) -> Result<()> {
    use kinds::{AuxCtx, Bf16Ctx, CtxSpec, OtherCtx};
    let l = &h.layout;
    let crc_at = payload.len().checked_sub(4).ok_or(bad("weights: truncated chunk"))?;
    // The streams side by side in one buffer, then room for a class of every unit.
    let ns: Vec<usize> = l.iter().map(|sp| if sp.bits1 { units.div_ceil(8) } else { units * sp.per_unit }).collect();
    let mut offs = Vec::with_capacity(l.len() + 1);
    let mut total = 0usize;
    for (k, &n) in ns.iter().enumerate() {
        offs.push(total);
        total += n + stagger(k);
    }
    offs.push(total);
    total += units + stagger(l.len());
    if arena.len() < total {
        arena.resize(total, 0);
    }
    let mut pos = 0usize;
    for s in 0..l.len() {
        let len = get_varint(payload, &mut pos)? as usize;
        let end = pos.checked_add(len).filter(|&e| e <= crc_at).ok_or(bad("weights: truncated stream"))?;
        let data = &payload[pos..end];
        pos = end;
        let nsym = ns[s];
        let (left, right) = arena.split_at_mut(offs[s]);
        let (out, rest) = right.split_at_mut(nsym);
        // the room for the units' classes, after the streams
        let cls_at = offs[l.len()] - offs[s] - nsym;
        let cls = &mut rest[cls_at..cls_at + units];
        let prior = |k: usize| &left[offs[k]..offs[k] + ns[k]];
        match &h.modes[s] {
            Mode::Raw | Mode::Bits1 => {
                if data.len() != nsym {
                    return Err(bad("weights: stream length"));
                }
                out.copy_from_slice(data);
            }
            Mode::Const(v) => {
                if !data.is_empty() {
                    return Err(bad("weights: stream length"));
                }
                out.fill(*v);
            }
            Mode::Rans(_) => {
                let t = tabs[s].as_ref().ok_or(bad("weights: tables"))?;
                match &l[s].ctx {
                    CtxSpec::None => rans::decode(data, nsym, t, &rans::NoCtx, out)?,
                    CtxSpec::Other(src) => rans::decode(data, nsym, t, &OtherCtx(prior(*src)), out)?,
                    CtxSpec::Aux(sh) => {
                        // the class of each block, from its scale's two bytes
                        let lookup = kinds::scale_classes(&h.spec.params).ok_or(bad("weights: region parameters"))?;
                        kinds::aux_into(&lookup, prior(0), prior(1), cls);
                        rans::decode(data, nsym, t, &AuxCtx(cls, *sh), out)?
                    }
                    CtxSpec::Bf16 { emode, prev, exp } => {
                        kinds::bf16_classes_into(prior(0), *emode, *exp, cls);
                        rans::decode(data, nsym, t, &Bf16Ctx { classes: cls, pm: if *prev { 0x7E } else { 0 } }, out)?
                    }
                    CtxSpec::Q8Max => rans::decode(data, nsym, t, &kinds::Q8Ctx, out)?,
                    CtxSpec::Nibbles(p) => rans::decode(data, nsym, t, &kinds::NibbleCtx(*p as usize - 1), out)?,
                    CtxSpec::Extremes { period, lo, hi } => rans::decode(data, nsym, t, &kinds::ExtremeCtx { mask: *period as usize - 1, lo: *lo, hi: *hi }, out)?,
                }
            }
        }
    }
    if pos != crc_at {
        return Err(bad("weights: chunk length"));
    }
    let refs: Vec<&[u8]> = (0..l.len()).map(|k| &arena[offs[k]..offs[k] + ns[k]]).collect();
    kinds::join(&h.spec, &refs, units, dst);
    let want = u32::from_le_bytes(payload[crc_at..].try_into().unwrap());
    let got = crate::format::crc32c(dst);
    if want != got {
        return Err(CodecError::ChecksumMismatch { expected: want, computed: got });
    }
    Ok(())
}
