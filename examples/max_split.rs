// Scratch: where the max level's time goes on a file, one core: the
// long-distance pass, the dfast parse, the entropy coding, the rest.
use std::time::Instant;
fn main() {
    let f = std::env::args().nth(1).unwrap();
    let d = std::fs::read(&f).unwrap();
    let n = d.len();
    let t = Instant::now();
    let far = glyd::ldm::Matches::find(&d, true);
    let t_ldm = t.elapsed().as_secs_f64();
    let mut tables = glyd::v7_encode::DfastTables::new();
    tables.clear_for(n);
    // GLYD_BITS="17,16": the table sizes to measure with.
    if let Some((l, s)) = std::env::var("GLYD_BITS").ok().and_then(|v| v.split_once(',').map(|(a, b)| (a.parse().unwrap(), b.parse().unwrap()))) {
        tables.set_bits(l, s);
    }
    let mut scratch = glyd::v7_encode::EncScratch::new();
    let (mut seqs, mut lits, mut payload) = (Vec::new(), Vec::new(), Vec::new());
    let mut prev = glyd::v7_encode::Tables::none();
    let (mut t_parse, mut t_enc, mut out) = (0.0f64, 0.0f64, 0usize);
    let mut off = 0usize;
    while off < n {
        let len = (n - off).min(glyd::format::MAX_BLOCK_SIZE);
        seqs.clear(); lits.clear();
        let mut reps = [1u32, 4, 8];
        let t = Instant::now();
        glyd::v7_encode::find_sequences_dfast_far(&d, off, len, &mut tables, &far, &mut reps, &mut seqs, &mut lits, &mut scratch);
        t_parse += t.elapsed().as_secs_f64();
        let t = Instant::now();
        payload.clear();
        glyd::v7_encode::encode_block_coded(&lits, 0, &mut prev, &mut scratch, true, &mut payload);
        t_enc += t.elapsed().as_secs_f64();
        out += payload.len() + 19;
        off += len;
    }
    let t_all = t_ldm + t_parse + t_enc;
    let c = vec![0u8; out];
    println!("{}: {} MB, --max {} B ({:.2}x) in {:.2} s = {:.0} MB/s; split: ldm {:.2} s ({:.0}%), parse {:.2} s ({:.0}%), entropy {:.2} s ({:.0}%); far matches {}",
        f.rsplit('/').next().unwrap(), n >> 20, c.len(), n as f64 / c.len() as f64, t_all, n as f64 / t_all / 1e6, t_ldm, t_ldm / t_all * 100.0, t_parse, t_parse / t_all * 100.0, t_enc, t_enc / t_all * 100.0, far.list.len());
}
