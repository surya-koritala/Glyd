// Scratch: over a Parquet file, how many snappy pages the reference
// compressor's port writes back byte for byte, by build; and the
// bytes of every codec's pages. `resnappy_probe <file.parquet>`.
use std::collections::BTreeMap;

fn main() {
    let f = std::env::args().nth(1).expect("a Parquet file");
    let data = std::fs::read(&f).unwrap();
    let (chunks, created_by) = glyd::parquet::chunks(&data).expect("not a Parquet file, or a footer this reader cannot walk");
    println!("{}: {} B, {} chunks, written by {created_by:?}", f.rsplit('/').next().unwrap(), data.len(), chunks.len());
    let mut by_codec: BTreeMap<String, (usize, usize, usize)> = BTreeMap::new();
    let mut by_build: BTreeMap<String, usize> = BTreeMap::new();
    let (mut snappy_pages, mut reproduced, mut snappy_bytes, mut reproduced_bytes, mut kept) = (0usize, 0usize, 0usize, 0usize, 0usize);
    for c in &chunks {
        let pages = glyd::parquet::pages(&data, c).unwrap_or_else(|| panic!("chunk {}/{}: page headers", c.row_group, c.column));
        for p in &pages {
            let e = by_codec.entry(format!("{:?}", c.codec)).or_insert((0, 0, 0));
            e.0 += 1;
            e.1 += p.compressed_len;
            e.2 += p.uncompressed_len;
            if c.codec == glyd::parquet::Codec::Zstd {
                let stream = p.compressed(&data);
                *by_build.entry(match glyd::rezstd::reproduce(stream) { Some((_, b)) => format!("zstd {:?}/{:?}/{:?}{}", b.version, b.level, b.writer, if b.checksum { "+ck" } else { "" }), None => "zstd: NOT reproduced".to_string() }).or_insert(0) += 1;
                continue;
            }
            if c.codec != glyd::parquet::Codec::Snappy {
                continue;
            }
            let stream = p.compressed(&data);
            snappy_pages += 1;
            snappy_bytes += stream.len();
            match glyd::resnappy::reproduce(stream) {
                Some((_, build)) => {
                    reproduced += 1;
                    reproduced_bytes += stream.len();
                    *by_build.entry(format!("{:?}/{}", build.hash, build.table_bits)).or_insert(0) += 1;
                }
                None => kept += stream.len(),
            }
        }
    }
    for (codec, (n, comp, raw)) in &by_codec {
        println!("  {codec}: {n} pages, {comp} B compressed, {raw} B raw ({:.2}x)", *raw as f64 / (*comp).max(1) as f64);
    }
    if snappy_pages > 0 {
        println!("  snappy pages reproduced: {reproduced} of {snappy_pages} ({} of {} B; {} B kept as they are)", reproduced_bytes, snappy_bytes, kept);
    }
    println!("  by build: {by_build:?}");
}
