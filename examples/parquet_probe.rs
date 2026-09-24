// Scratch: over a Parquet file with snappy pages, the bytes the codec
// sees three ways: the pages as they are, opened (raw), and opened and
// modeled; each compressed at --max. `parquet_probe <file.parquet>`.
use std::collections::BTreeMap;

fn main() {
    let f = std::env::args().nth(1).expect("a Parquet file");
    let data = std::fs::read(&f).unwrap();
    let (chunks, created_by) = glyd::parquet::chunks(&data).expect("not a Parquet file this reader walks");
    let (mut closed, mut raw, mut modeled, mut recipes) = (Vec::new(), Vec::new(), Vec::new(), 0usize);
    let mut kinds: BTreeMap<String, (usize, usize, usize)> = BTreeMap::new();
    let mut streams: BTreeMap<String, (Vec<u8>, Vec<u8>)> = BTreeMap::new();
    let (mut pages, mut unmodeled) = (0usize, 0usize);
    for c in &chunks {
        for p in glyd::parquet::pages(&data, c).unwrap() {
            let stream = p.compressed(&data);
            closed.extend_from_slice(stream);
            if c.codec != glyd::parquet::Codec::Snappy {
                continue;
            }
            let Some((plain, _)) = glyd::resnappy::reproduce(stream) else { continue };
            pages += 1;
            raw.extend_from_slice(&plain);
            match glyd::parquet::model(&plain, c, &p) {
                Some((recipe, m)) => {
                    assert_eq!(glyd::parquet::unmodel(&recipe, &m).as_deref(), Some(&plain[..]), "page at {} comes back", p.header_at);
                    let name = format!("type {} kind {} {}", c.physical_type, recipe[0], if p.kind == 2 { "dict" } else { "data" });
                    let e = kinds.entry(name.clone()).or_insert((0, 0, 0));
                    e.0 += 1;
                    e.1 += plain.len();
                    e.2 += m.len();
                    recipes += recipe.len();
                    modeled.extend_from_slice(&m);
                    let st = streams.entry(name).or_insert((Vec::new(), Vec::new()));
                    st.0.extend_from_slice(&plain);
                    st.1.extend_from_slice(&m);
                }
                None => {
                    unmodeled += 1;
                    if matches!(p.encoding, 2 | 8) {
                        let why = glyd::parquet::index_page_diagnosis(&plain, c, &p);
                        let why = if why.starts_with("encode differs") { format!("encode differs, width {}", why.rsplit(' ').next().unwrap()) } else { why };
                        *kinds.entry(format!("  why not: {why}")).or_insert((0, 0, 0)) = { let e = kinds.get(&format!("  why not: {why}")).copied().unwrap_or((0, 0, 0)); (e.0 + 1, e.1 + plain.len(), e.2) };
                    }
                    let name = format!("type {} enc {} {} (as is)", c.physical_type, p.encoding, if p.kind == 2 { "dict" } else { "data" });
                    let e = kinds.entry(name.clone()).or_insert((0, 0, 0));
                    e.0 += 1;
                    e.1 += plain.len();
                    e.2 += plain.len();
                    modeled.extend_from_slice(&plain);
                    let st = streams.entry(name).or_insert((Vec::new(), Vec::new()));
                    st.0.extend_from_slice(&plain);
                    st.1.extend_from_slice(&plain);
                }
            }
        }
    }
    let size = |b: &[u8]| {
        let mut out = Vec::new();
        glyd::compress_into_max(b, &mut out);
        out.len()
    };
    println!("{}: {} B, written by {created_by:?}; {pages} snappy pages, {unmodeled} left as they are, recipes {recipes} B", f.rsplit('/').next().unwrap(), data.len());
    println!("  closed pages {} B -> --max {} B", closed.len(), size(&closed));
    println!("  raw pages    {} B -> --max {} B", raw.len(), size(&raw));
    println!("  modeled      {} B -> --max {} B", modeled.len(), size(&modeled));
    for (name, (n, r, m)) in &kinds {
        match streams.get(name) {
            Some((rs, ms)) => println!("    {name}: {n} pages, raw {r} B -> --max {} B; modeled {m} B -> --max {} B", size(rs), size(ms)),
            None => println!("    {name}: {n} pages, raw {r} B"),
        }
    }
}
