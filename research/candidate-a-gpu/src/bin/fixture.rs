//! CI-only matrix preparation; no device execution.
use glyd_candidate_a_gpu::{expanded, pack_rounded, unpack};
use glyd_gpu::pack;
use glyd_lossy_study::approximate;
use std::{fs, path::Path};

fn write_words(path: &Path, words: &[u16]) -> std::io::Result<()> {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    fs::write(path, bytes)
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 { return Err("usage: fixture INPUT.bf16 ROWS COLS OUT_DIR".into()); }
    let rows: usize = args[2].parse()?;
    let cols: usize = args[3].parse()?;
    let raw = fs::read(&args[1])?;
    if rows == 0 || cols == 0 || rows > 1 << 20 || cols > 1 << 20
        || rows.checked_mul(cols).filter(|n| *n <= 1 << 31).and_then(|n| n.checked_mul(2)) != Some(raw.len()) {
        return Err("matrix dimensions or input length invalid".into());
    }
    let words: Vec<u16> = raw.chunks_exact(2)
        .map(|b| approximate(u16::from_le_bytes([b[0],b[1]]),3)).collect();
    drop(raw);
    let packed = pack_rounded(&words,rows,cols)?;
    if unpack(&packed)? != words { return Err("rounded-value reconstruction differs".into()); }
    let original = expanded(&packed)?;
    if original != pack::pack_tiered_checked(&words,rows,cols).ok_or("base pack check failed")? {
        return Err("expanded pack differs from the base reference".into());
    }
    let out = Path::new(&args[4]);
    fs::create_dir(out)?;
    write_words(&out.join("rounded.bf16"),&words)?;
    fs::write(out.join("compact.data"),&packed.data)?;
    fs::write(out.join("expanded.data"),&original.data)?;
    fs::write(out.join("blocks.bin"),&packed.blocks)?;
    fs::write(out.join("bases.i32"),packed.block_base.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())?;
    let metadata=format!(
        "{{\"rows\":{rows},\"cols\":{cols},\"compact\":{},\"tiers\":[{},{},{}],\"base_tiers\":[{},{},{}],\"weight_buffer_bytes\":{},\"symbol_bytes\":12,\"base_buffer_bytes\":{},\"exact_after_rounding\":true}}\n",
        packed.compact(),packed.tiers[0],packed.tiers[1],packed.tiers[2],
        original.tiers[0],original.tiers[1],original.tiers[2],
        packed.resident_bytes()-12,pack::tiered_bytes(&original)-12);
    fs::write(out.join("pack.json"),metadata)?;
    Ok(())
}
fn main() {
    if let Err(err)=run() { eprintln!("{err}"); std::process::exit(1); }
}
