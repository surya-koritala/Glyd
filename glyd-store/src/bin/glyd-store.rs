//! glyd-store: a store that compresses across the objects it holds.
//!
//!   glyd-store DIR --put FILE...        each file kept as a delta against the stored object
//!                                       it most resembles when that pays; small files in packs
//!   glyd-store DIR --get ID -o FILE     an object back
//!   glyd-store DIR --get ID --content   a gzip object's content, its stream not re-created
//!   glyd-store DIR --restore OUT        every live object back, into OUT/<id>; prints id and name
//!   glyd-store DIR --find NAME          the id of the latest object put under a name
//!   glyd-store DIR --delete ID          a tombstone; the bytes stay while a live chain needs them
//!   glyd-store DIR --rebase ID          an object read often stored alone again (one decode)
//!   glyd-store DIR --compact            free what no live object needs
//!   glyd-store DIR --verify             every object read back and checked
//!   glyd-store DIR --stats              the objects and the bytes
//!   glyd-store DIR --s3 s3://bucket/prefix ...   objects in S3 (or an S3-compatible service); metadata in DIR
//!   glyd-store DIR [--s3 ...] --rebuild     DIR made anew from the objects (a lost metadata directory)
//!   glyd-store DIR --ultra | --cold ... the level objects stored alone take
//!   glyd-store --audit DIR|s3://bucket/prefix [--sample N]
//!                                       what the store would save there, from a sample, in dollars a year
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

fn usage() {
    eprintln!("{}", include_str!("glyd-store.rs").lines().take_while(|l| l.starts_with("//!")).map(|l| l.trim_start_matches("//!").trim_start_matches(' ')).collect::<Vec<_>>().join("\n"));
}

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut store_dir: Option<String> = None;
    let mut output_path: Option<String> = None;
    let mut inputs: Vec<String> = Vec::new();
    let mut store_put = false;
    let mut store_get: Option<u32> = None;
    let mut store_content = false;
    let mut store_stats = false;
    let mut store_find: Option<String> = None;
    let mut store_delete: Option<u32> = None;
    let mut store_compact = false;
    let mut store_rebuild = false;
    let mut store_verify = false;
    let mut store_restore: Option<String> = None;
    let mut store_rebase: Option<u32> = None;
    let mut store_s3: Option<String> = None;
    let mut audit: Option<String> = None;
    let mut sample: usize = 200;
    let (mut ultra, mut cold) = (false, false);
    let need = |i: usize, what: &str| -> String {
        args.get(i + 1).cloned().unwrap_or_else(|| {
            eprintln!("Error: {} requires {what}", args[i]);
            std::process::exit(1);
        })
    };
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                usage();
                return Ok(());
            }
            "--put" => store_put = true,
            "--get" => {
                store_get = Some(need(i, "an object id").parse().unwrap_or_else(|_| {
                    eprintln!("Error: --get requires an object id");
                    std::process::exit(1);
                }));
                i += 1;
            }
            "--content" => store_content = true,
            "--stats" => store_stats = true,
            "--version" | "-V" => {
                println!("glyd-store {}", env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            "--find" => {
                store_find = Some(need(i, "a name"));
                i += 1;
            }
            "--delete" => {
                store_delete = Some(need(i, "an object id").parse().unwrap_or_else(|_| {
                    eprintln!("Error: --delete requires an object id");
                    std::process::exit(1);
                }));
                i += 1;
            }
            "--compact" => store_compact = true,
            "--rebuild" => store_rebuild = true,
            "--verify" => store_verify = true,
            "--restore" => {
                store_restore = Some(need(i, "a directory"));
                i += 1;
            }
            "--rebase" => {
                store_rebase = Some(need(i, "an object id").parse().unwrap_or_else(|_| {
                    eprintln!("Error: --rebase requires an object id");
                    std::process::exit(1);
                }));
                i += 1;
            }
            "--s3" => {
                store_s3 = Some(need(i, "an s3://bucket/prefix url"));
                i += 1;
            }
            "--audit" => {
                audit = Some(need(i, "a directory or an s3:// url"));
                i += 1;
            }
            "--sample" => {
                sample = need(i, "a count").parse().unwrap_or_else(|_| {
                    eprintln!("Error: --sample requires a count");
                    std::process::exit(1);
                });
                i += 1;
            }
            "-o" | "--output" => {
                output_path = Some(need(i, "an output filename"));
                i += 1;
            }
            "--ultra" | "-19" => ultra = true,
            "--cold" | "-C" => cold = true,
            other => {
                if other.starts_with('-') {
                    eprintln!("Unknown option: {other}");
                    usage();
                    std::process::exit(1);
                } else if store_dir.is_none() {
                    store_dir = Some(other.to_string());
                } else {
                    inputs.push(other.to_string());
                }
            }
        }
        i += 1;
    }
    if let Some(target) = audit {
        return run_audit(&target, sample);
    }
    let Some(store_dir) = store_dir else {
        usage();
        std::process::exit(1);
    };
    {
        let dir = store_dir;
        let mut store = match (store_s3, store_rebuild) {
            (Some(url), false) => glyd_store::Store::open_with(&dir, Box::new(glyd_store::S3Backend::new(&url)?))?,
            (None, false) => glyd_store::Store::open(&dir)?,
            (s3, true) => {
                let t = Instant::now();
                let (store, failures) = match s3 {
                    Some(url) => glyd_store::Store::rebuild_with(&dir, Box::new(glyd_store::S3Backend::new(&url)?))?,
                    None => glyd_store::Store::rebuild(&dir)?,
                };
                for (id, e) in &failures {
                    eprintln!("object {id}: {e}");
                }
                eprintln!("rebuilt: {} objects in the index, {} unreadable, {:.0} s", store.entries().len(), failures.len(), t.elapsed().as_secs_f64());
                store
            }
        };
        if cold {
            store.set_level(glyd_store::Level::Cold);
        } else if ultra {
            store.set_level(glyd_store::Level::Ultra);
        }
        if store_put {
            for f in &inputs {
                // Read into a buffer the store keeps as its cached copy:
                // one copy of the bytes, not two.
                let data = std::fs::read(f)?;
                let len = data.len();
                let t = Instant::now();
                let id = store.put_vec(f, data)?;
                let e = &store.entries()[id as usize];
                eprintln!("{:>6}  {:>10} -> {:>10} B  {}  {:.0} MB/s  {}", id, e.raw_len, e.stored_len, e.base.map_or("alone".to_string(), |b| format!("delta against {} (depth {})", b, e.depth)), len as f64 / t.elapsed().as_secs_f64() / 1e6, f);
            }
        }
        if let Some(id) = store_get {
            if store_content {
                write_out(&output_path, &store.get_content(id)?)?;
            } else {
                let mut out: Box<dyn Write> = match output_path {
                    Some(ref p) if p != "-" => Box::new(io::BufWriter::with_capacity(1 << 20, std::fs::File::create(p)?)),
                    _ => Box::new(io::stdout()),
                };
                store.get_to(id, &mut out)?;
                out.flush()?;
            }
        }
        if let Some(name) = store_find {
            match store.id_of(&name) {
                Some(id) => println!("{id}"),
                None => {
                    eprintln!("not in the store: {name}");
                    std::process::exit(1);
                }
            }
        }
        if let Some(id) = store_delete {
            store.delete(id)?;
        }
        if let Some(id) = store_rebase {
            store.rebase(id)?;
        }
        if store_compact {
            let freed = store.compact()?;
            eprintln!("compacted: {freed} B freed");
        }
        if let Some(out) = store_restore {
            // In id order, so each chain is decoded once, its objects kept
            // for the ones built on them.
            std::fs::create_dir_all(&out)?;
            let t = Instant::now();
            let mut bytes = 0u64;
            let live: Vec<(u32, String)> = store.entries().iter().filter(|e| !e.deleted && !e.name.starts_with("pack of ")).map(|e| (e.id, e.name.clone())).collect();
            for (id, name) in &live {
                let data = store.get(*id)?;
                bytes += data.len() as u64;
                std::fs::write(Path::new(&out).join(id.to_string()), &data)?;
                println!("{id}\t{name}");
            }
            eprintln!("restored: {} objects, {} bytes, {:.0} MB/s", live.len(), bytes, bytes as f64 / t.elapsed().as_secs_f64() / 1e6);
        }
        if store_verify {
            let (ok, failed) = store.verify();
            for (id, e) in &failed {
                eprintln!("object {id}: {e}");
            }
            eprintln!("verified: {ok} objects ok, {} failed", failed.len());
            if !failed.is_empty() {
                std::process::exit(1);
            }
        }
        if store_stats {
            for e in store.entries() {
                let how = if e.deleted { "deleted".to_string() } else if let Some((p, i)) = e.pack { format!("in pack {p} at {i}") } else { e.base.map_or("alone".to_string(), |b| format!("delta against {} (depth {})", b, e.depth)) };
                println!("{:>6}  {:>12}  {:>12}  {:<28} {}", e.id, e.raw_len, e.stored_len, how, e.name);
            }
            let (raw, stored) = store.stats();
            let live = store.entries().iter().filter(|e| !e.deleted).count();
            println!("{} objects ({} live): {} B raw, {} B on disk ({:.2}x)", store.entries().len(), live, raw, stored, raw as f64 / stored.max(1) as f64);
        }
        return Ok(());
    }
}

/// The objects under `target`, a directory or s3://bucket/prefix:
/// (name, bytes) in name order, and how to read one.
fn list_objects(target: &str) -> io::Result<(Vec<(String, u64)>, Box<dyn Fn(&str) -> io::Result<Vec<u8>>>)> {
    if target.starts_with("s3://") {
        let s3 = glyd_store::S3Backend::new(target)?;
        let mut out = s3.list()?;
        out.retain(|(_, n)| *n > 0);
        return Ok((out, Box::new(move |key| glyd_store::Backend::read(&s3, key))));
    }
    fn walk(dir: &Path, out: &mut Vec<(String, u64)>) -> io::Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out)?;
            } else if let Ok(m) = entry.metadata() {
                if m.len() > 0 {
                    out.push((path.to_string_lossy().to_string(), m.len()));
                }
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    walk(Path::new(target), &mut out)?;
    out.sort();
    Ok((out, Box::new(|name| std::fs::read(name))))
}

/// `zstd -3` bytes of `data` through the zstd CLI, when it is installed.
fn zstd3_len(data: &[u8]) -> Option<usize> {
    use std::process::{Command, Stdio};
    let mut child = Command::new("zstd").args(["-3", "-q", "-c", "-T0"]).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn().ok()?;
    let mut stdin = child.stdin.take()?;
    let data = data.to_vec();
    let writer = std::thread::spawn(move || stdin.write_all(&data));
    let out = child.wait_with_output().ok()?;
    writer.join().ok()?.ok()?;
    if out.status.success() { Some(out.stdout.len()) } else { None }
}

/// The store on a sample of the objects under `target`, scaled to all
/// of them, priced at S3 Standard's list rate.
fn run_audit(target: &str, sample: usize) -> io::Result<()> {
    const PRICE_TB_YEAR: f64 = 0.023 * 12.0 * 1000.0;
    let (all, read_object) = list_objects(target)?;
    if all.is_empty() {
        eprintln!("nothing under {target}");
        std::process::exit(1);
    }
    let total: u64 = all.iter().map(|(_, n)| n).sum();
    // The sample: runs of consecutive objects (versions of a thing sit
    // together by name) at eight places spread over the listing.
    let n = sample.min(all.len());
    let picked: Vec<&(String, u64)> = if n >= all.len() {
        all.iter().collect()
    } else {
        let runs = 8.min(n);
        let per = n / runs;
        let mut v = Vec::with_capacity(n);
        for r in 0..runs {
            let start = r * (all.len() - per) / (runs - 1).max(1);
            v.extend(all[start..start + per].iter());
        }
        v
    };
    let dir = std::env::temp_dir().join(format!("glyd-audit-{}", std::process::id()));
    let mut store = glyd_store::Store::open(&dir)?;
    let zstd = zstd3_len(b"probe").is_some();
    let (mut raw, mut alone_z, mut alone_g) = (0u64, 0u64, 0u64);
    eprintln!("{target}: {} objects, {:.1} GB; sampling {} of them{}", all.len(), total as f64 / 1e9, picked.len(), if zstd { " against zstd -3" } else { " (no zstd CLI: Glyd --max alone stands in for today's codec)" });
    let t = Instant::now();
    for (i, (name, _)) in picked.iter().enumerate() {
        let data = match read_object(name) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping {name}: {e}");
                continue;
            }
        };
        raw += data.len() as u64;
        if zstd {
            alone_z += zstd3_len(&data).unwrap_or(data.len()) as u64;
        }
        let mut c = Vec::new();
        glyd::compress_parallel_into_max(&data, &mut c);
        alone_g += c.len() as u64;
        store.put(name, &data)?;
        if (i + 1) % 20 == 0 {
            eprintln!("  {} of {} ({:.0} MB/s)", i + 1, picked.len(), raw as f64 / t.elapsed().as_secs_f64() / 1e6);
        }
    }
    store.flush()?;
    let (_, stored) = store.stats();
    let deltas = store.entries().iter().filter(|e| e.base.is_some()).count();
    let packs = store.entries().iter().filter(|e| e.name.starts_with("pack of ")).count();
    drop(store);
    let _ = std::fs::remove_dir_all(&dir);
    let today = if zstd { alone_z } else { alone_g };
    let scale = total as f64 / raw.max(1) as f64;
    let tb = |b: u64| b as f64 * scale / 1e12;
    let size = |t: f64| if t >= 1.0 { format!("{t:.2} TB") } else { format!("{:.1} GB", t * 1000.0) };
    let money = |d: f64| if d >= 100.0 { format!("${d:.0}") } else { format!("${d:.2}") };
    println!("{target}");
    println!("  {} objects, {} raw; sample of {} objects, {}", all.len(), size(total as f64 / 1e12), picked.len(), size(raw as f64 / 1e12));
    println!("  {:<34} {:>10} stored  {:>10} a year", if zstd { "zstd -3, each object alone:" } else { "Glyd --max, each object alone:" }, size(tb(today)), money(tb(today) * PRICE_TB_YEAR));
    if zstd {
        println!("  {:<34} {:>10} stored  {:>10} a year", "Glyd --max, each object alone:", size(tb(alone_g)), money(tb(alone_g) * PRICE_TB_YEAR));
    }
    println!("  {:<34} {:>10} stored  {:>10} a year", "Glyd store (across the objects):", size(tb(stored)), money(tb(stored) * PRICE_TB_YEAR));
    println!("  {:.2}x fewer bytes than {}; {} of {} sampled objects stored as deltas, {} packs of small ones", today as f64 / stored.max(1) as f64, if zstd { "zstd -3" } else { "Glyd alone" }, deltas, picked.len(), packs);
    println!("  saving about {} a year at S3 Standard's list price ($23/TB-month), scaled from the sample", money((tb(today) - tb(stored)) * PRICE_TB_YEAR));
    Ok(())
}

fn write_out(output_path: &Option<String>, data: &[u8]) -> io::Result<()> {
    match output_path {
        Some(p) if p != "-" => std::fs::write(p, data),
        _ => {
            io::stdout().write_all(data)?;
            io::stdout().flush()
        }
    }
}
