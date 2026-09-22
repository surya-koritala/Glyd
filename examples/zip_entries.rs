// Scratch: each local entry of a zip, and whether preflate opens it.
use preflate_rs::{preflate_whole_deflate_stream, PreflateConfig};
fn main() {
    let f = std::env::args().nth(1).expect("a zip");
    let d = std::fs::read(&f).unwrap();
    let le16 = |p: usize| u16::from_le_bytes(d[p..p + 2].try_into().unwrap()) as usize;
    let le32 = |p: usize| u32::from_le_bytes(d[p..p + 4].try_into().unwrap()) as usize;
    let mut at = 0usize;
    let (mut n, mut ok, mut fail) = (0, 0, 0);
    let mut errors: std::collections::HashMap<String, usize> = Default::default();
    while at + 30 <= d.len() && &d[at..at + 4] == b"PK\x03\x04" {
        let (flags, method, csize) = (le16(at + 6), le16(at + 8), le32(at + 18));
        let (nlen, xlen) = (le16(at + 26), le16(at + 28));
        let name = String::from_utf8_lossy(&d[at + 30..at + 30 + nlen]).to_string();
        let data = at + 30 + nlen + xlen;
        n += 1;
        let mut end = data + csize;
        if method == 8 {
            match preflate_whole_deflate_stream(&d[data..], &PreflateConfig::default()) {
                Ok((r, t)) => {
                    ok += 1;
                    end = data + r.compressed_size;
                    if n <= 5 { println!("{name}: flags {flags:#x} csize {csize} -> ok, compressed {} plain {} corrections {}", r.compressed_size, t.text().len(), r.corrections.len()); }
                }
                Err(e) => {
                    fail += 1;
                    let key = format!("{:?}", e.exit_code());
                    *errors.entry(key.clone()).or_default() += 1;
                    if fail <= 5 { println!("{name}: flags {flags:#x} csize {csize} -> ERROR {key}: {}", e.message()); }
                    if csize == 0 { end = (data..d.len() - 3).find(|&p| &d[p..p + 4] == b"PK\x03\x04").unwrap_or(d.len()); }
                }
            }
        }
        // past a data descriptor
        let next = (end..(end + 32).min(d.len().saturating_sub(3))).find(|&p| &d[p..p + 4] == b"PK\x03\x04").unwrap_or(end);
        at = next;
    }
    println!("{n} entries, {ok} open, {fail} fail; errors {errors:?}");
}
