// One small object with a dictionary: where its bytes go (header,
// sub-header, sections, tables reused or not), and the same with zstd's
// trained content as Glyd's dictionary content.
use glyd::format::{BlockHeader, COMPACT_MARKER, FLAG_RAW_UNCOMPRESSED, HEADER_SIZE};
use glyd::v7_encode::{payload_layout, payload_layout_compact};
use std::io::Read;
fn anatomy(name: &str, dict: &glyd::Dict, objects: &[&[u8]]) {
    let (mut raw, mut out, mut hdr, mut sub, mut secs, mut rawblocks, mut reuse_lit, mut reuse_seq, mut n) = (0usize, 0usize, 0usize, 0usize, [0usize; 5], 0usize, 0usize, 0usize, 0usize);
    for o in objects {
        let mut c = Vec::new();
        glyd::compress_with_dict(dict, o, &mut c);
        raw += o.len(); out += c.len();
        let v9 = c[0] == COMPACT_MARKER;
        let (h, hl): (BlockHeader, usize) = if v9 { BlockHeader::read_compact(&c).unwrap() } else { (unsafe { std::ptr::read_unaligned(c.as_ptr() as *const BlockHeader) }, HEADER_SIZE) };
        hdr += hl; n += 1;
        if h.flags & FLAG_RAW_UNCOMPRESSED != 0 { rawblocks += 1; continue; }
        let payload = &c[hl..hl + h.payload_len()];
        let l = if v9 { payload_layout_compact(payload, false) } else { payload_layout(payload) }.unwrap();
        sub += payload.len() - l.sub.sizes.iter().map(|&s| s as usize).sum::<usize>();
        for i in 0..5 { secs[i] += l.sub.sizes[i] as usize; }
        if l.sub.reuse & 1 != 0 { reuse_lit += 1; }
        if l.sub.reuse & 2 != 0 { reuse_seq += 1; }
    }
    println!("{name}: ratio {:.2}; per object {:.0} B = header {:.0} + subheader {:.0} + lit {:.0} + ll {:.0} + ml {:.0} + off {:.0} + extra {:.0}; raw blocks {rawblocks}/{n}, lit table reused {reuse_lit}, seq tables reused {reuse_seq}",
        raw as f64 / out as f64, out as f64 / n as f64, hdr as f64 / n as f64, sub as f64 / n as f64, secs[0] as f64 / n as f64, secs[1] as f64 / n as f64, secs[2] as f64 / n as f64, secs[3] as f64 / n as f64, secs[4] as f64 / n as f64);
}
fn main() {
    let file = std::env::args().nth(1).unwrap_or("corpus/ext/gharchive.json".into());
    let size: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(4096);
    let mut f = std::fs::File::open(&file).unwrap();
    let mut d = vec![0u8; 64 << 20];
    let n = f.read(&mut d).unwrap(); d.truncate(n);
    let objects: Vec<&[u8]> = d.chunks(size).take(2000).collect();
    let samples: Vec<&[u8]> = d[32 << 20..].chunks(size).take(2000).collect();
    let zdict = zstd::dict::from_samples(&samples, 110 * 1024).unwrap();
    // zstd's dictionary: a header (magic, id, entropy tables) then raw content; the content is the tail.
    let zcontent = &zdict[zdict.len() - 100 * 1024..];
    anatomy("Glyd trainer", &glyd::Dict::train(&samples, 110 * 1024), &objects);
    anatomy("zstd content", &glyd::Dict::from_content(zcontent, &samples), &objects);
    anatomy("no tables (content only, tables from content)", &glyd::Dict::from_content(zcontent, &[]), &objects);
}
