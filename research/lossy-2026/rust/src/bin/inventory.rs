use glyd_lossy_study::{approximate, decode, encode};
use std::{env, fs::{self,File}, io::{BufReader,Read,Seek,SeekFrom}, time::Instant};

#[derive(Default,Debug)]
struct Stats {
    values: u64, packed_bytes: u64, blocks: u64, raw_blocks: u64,
    dictionary_bytes: u64, field_bytes: u64, exception_bytes: u64, raw_bytes: u64,
    changed: u64, histogram: Vec<u64>,
}

fn measure<R: Read>(reader: &mut R, bytes: u64) -> Result<Stats,Box<dyn std::error::Error>> {
    if bytes % 2 != 0 { return Err("BF16 tensor has a partial word".into()); }
    let mut stats = Stats { packed_bytes: 28, histogram: vec![0;65536], ..Stats::default() };
    let mut remaining = bytes;
    let mut data = [0u8;2048];
    while remaining != 0 {
        let n = remaining.min(data.len() as u64) as usize;
        reader.read_exact(&mut data[..n])?;
        let words: Vec<u16> = data[..n].chunks_exact(2).map(|b|u16::from_le_bytes([b[0],b[1]])).collect();
        let packed = encode(&words,3,1024)?;
        let restored = decode(&packed)?;
        if restored.len()!=words.len() || restored.iter().zip(&words).any(|(&a,&b)| a!=approximate(b,3)) {
            return Err("Candidate A reconstruction changed".into());
        }
        stats.packed_bytes += (packed.len()-28) as u64;
        stats.blocks += 1;
        let block = &packed[28..];
        if block[2] == 0 {
            stats.raw_blocks += 1;
            stats.raw_bytes += n as u64;
        } else {
            stats.dictionary_bytes += block[4] as u64;
            stats.exception_bytes += 2*u16::from_le_bytes([block[6],block[7]]) as u64;
            stats.field_bytes += u32::from_le_bytes(block[8..12].try_into()?) as u64;
        }
        for (&a,&b) in words.iter().zip(&restored) {
            stats.changed += (a!=b) as u64;
            stats.histogram[b as usize] += 1;
        }
        stats.values += words.len() as u64;
        remaining -= n as u64;
    }
    if stats.packed_bytes != 28+12*stats.blocks+stats.dictionary_bytes+stats.field_bytes+stats.exception_bytes+stats.raw_bytes {
        return Err("encoded byte accounting mismatch".into());
    }
    Ok(stats)
}

impl Stats {
    fn json(&self, seconds: f64) -> String {
        let entropy = if self.values == 0 { 0.0 } else {
            self.histogram.iter().filter(|&&n|n>0).map(|&n| {
                let p=n as f64/self.values as f64; -p*p.log2()
            }).sum::<f64>()
        };
        format!(concat!("{{\"fraction_bits\":3,\"block_values\":1024,\"values\":{},",
            "\"source_bytes\":{},\"packed_bytes\":{},\"tensor_header_bytes\":28,",
            "\"block_header_bytes\":{},\"dictionary_bytes\":{},\"field_bytes\":{},",
            "\"exception_bytes\":{},\"raw_bytes\":{},\"raw_blocks\":{},\"changed_values\":{},",
            "\"empirical_iid_entropy_bits\":{},\"exact_after_rounding\":true,\"seconds\":{:.6}}}\n"),
            self.values,2*self.values,self.packed_bytes,12*self.blocks,self.dictionary_bytes,self.field_bytes,
            self.exception_bytes,self.raw_bytes,self.raw_blocks,self.changed,entropy,seconds)
    }
}

fn main() -> Result<(),Box<dyn std::error::Error>> {
    let a: Vec<String> = env::args().collect();
    if a.len()==3 && a[1]=="--rounding-reference" {
        let data: Vec<u8>=(0..=u16::MAX).flat_map(|w|approximate(w,3).to_le_bytes()).collect();
        fs::write(&a[2],data)?; return Ok(());
    }
    if a.len()!=5 { return Err("usage: glyd-lossy-inventory FILE OFFSET BYTES OUTPUT.json".into()); }
    let offset: u64 = a[2].parse()?;
    let bytes: u64 = a[3].parse()?;
    let file = File::open(&a[1])?;
    if offset.checked_add(bytes).ok_or("tensor range overflow")? > file.metadata()?.len() {
        return Err("tensor extends past input".into());
    }
    let mut reader = BufReader::with_capacity(1<<20,file);
    reader.seek(SeekFrom::Start(offset))?;
    let start=Instant::now();
    let stats = measure(&mut reader,bytes)?;
    fs::write(&a[4],stats.json(start.elapsed().as_secs_f64()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn whole_tensor_accounting_matches_one_shot_codec() {
        let words: Vec<u16> = (0..=u16::MAX).collect();
        for n in [0,1,31,1023,1024,1025,65536] {
            let data: Vec<u8> = words[..n].iter().flat_map(|w|w.to_le_bytes()).collect();
            let stats=measure(&mut &data[..],data.len() as u64).unwrap();
            assert_eq!(stats.packed_bytes,encode(&words[..n],3,1024).unwrap().len() as u64);
            assert_eq!(stats.histogram.iter().sum::<u64>(),n as u64);
        }
        assert!(measure(&mut &[0u8][..],1).is_err());
        assert!(measure(&mut &[0u8][..],2).is_err());
    }
    #[test]
    fn streaming_throughput_estimate() {
        // CI-only estimate for the complete inventory's CPU budget; no GPU claim.
        let values=1<<22;
        let data: Vec<u8> = (0..values).flat_map(|i| ((0x3900 | (i%2048)) as u16).to_le_bytes()).collect();
        let start=Instant::now();
        let stats=measure(&mut &data[..],data.len() as u64).unwrap();
        let seconds=start.elapsed().as_secs_f64();
        assert_eq!(stats.values,values as u64);
        eprintln!("inventory synthetic throughput: {:.2} MiB/s; 51.75 GiB serial estimate {:.1} min",data.len() as f64/seconds/1048576.0,55_562_855_904.0/data.len() as f64*seconds/60.0);
    }
}
