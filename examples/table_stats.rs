// How often the ultra level's blocks carry fresh entropy tables (bytes
// spent on tables per file), from the sub-headers of the output.
use glyd::format::{BlockHeader, HEADER_SIZE, FLAG_RAW_UNCOMPRESSED};
use glyd::v7_encode::payload_layout;

fn main() {
    let mut files: Vec<_> = std::fs::read_dir("corpus").unwrap().filter_map(|e| e.ok()).map(|e| e.path())
        .filter(|p| p.is_file() && !p.file_name().unwrap().to_string_lossy().starts_with("enwik")).collect();
    files.sort();
    for p in files {
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        let d = std::fs::read(&p).unwrap();
        let mut c = Vec::new();
        glyd::compress_into_ultra(&d, &mut c);
        let (mut blocks, mut raw, mut lit_new, mut seq_new, mut lit_raw, mut seq_raw, mut cursor) = (0, 0, 0, 0, 0, 0, 0usize);
        while cursor < c.len() {
            let h: BlockHeader = unsafe { std::ptr::read_unaligned(c[cursor..].as_ptr() as *const BlockHeader) };
            let len = h.payload_len();
            blocks += 1;
            if h.flags & FLAG_RAW_UNCOMPRESSED != 0 { raw += 1; } else {
                let l = payload_layout(&c[cursor + HEADER_SIZE..cursor + HEADER_SIZE + len]).unwrap();
                if l.sub.coded & 1 == 0 { lit_raw += 1; } else if l.sub.reuse & 1 == 0 { lit_new += 1; }
                if l.sub.coded & 0b1110 != 0b1110 { seq_raw += 1; } else if l.sub.reuse & 2 == 0 { seq_new += 1; }
            }
            cursor += HEADER_SIZE + len;
        }
        println!("{name:10} {blocks:4} blocks ({raw} raw): literal tables written {lit_new:4} (raw literals {lit_raw:3}); sequence tables written {seq_new:4} (some raw {seq_raw:3}); {:.0} B/block compressed", c.len() as f64 / blocks as f64);
    }
}
