//! Experimental resident representation of fixed Candidate A values.
//! Matrix ordering and exponent streams are the existing first-party representation.
//! Only the four zero fraction bits are removed. No rounding occurs in this module.
use glyd_gpu::pack::{self, Tiered};

pub const TAG: u32 = 3;
pub const STEP_BYTES: usize = 768;

#[derive(Clone, Debug)]
pub struct Packed {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<u8>,
    pub blocks: Vec<u8>,
    pub block_base: Vec<i32>,
    pub tiers: [u32; 3],
}

impl Packed {
    pub fn compact(&self) -> bool { self.tiers[0] >> 24 == TAG }
    /// Device weight buffers and symbol words; activation/cache/workspace bytes are separate.
    pub fn resident_bytes(&self) -> usize {
        self.data.len() + self.blocks.len() + 4 * self.block_base.len() + 12
    }
}

fn shape(rows: usize, cols: usize, count: usize) -> bool {
    rows != 0 && cols != 0 && rows % 64 == 0 && cols % 16 == 0 && rows.checked_mul(cols) == Some(count)
}

fn nibble(byte: u8) -> u8 { ((byte >> 4) & 14) | (byte & 1) }
fn byte(nibble: u8) -> u8 { ((nibble & 14) << 4) | (nibble & 1) }

/// Accept already-rounded Candidate A values. Preserve non-finite payloads with an
/// unchanged tiered tensor where any such payload needs the four omitted bits.
pub fn pack_rounded(words: &[u16], rows: usize, cols: usize) -> Result<Packed, String> {
    if !shape(rows, cols, words.len()) { return Err("unsupported matrix shape".into()); }
    if words.iter().any(|w| w & 0x7f80 != 0x7f80 && w & 15 != 0) {
        return Err("finite input is not a Candidate A rounded value".into());
    }
    let base = pack::pack_tiered_checked(words, rows, cols).ok_or("base representation check failed")?;
    let use_compact = words.iter().all(|w| w & 15 == 0);
    let mut data = Vec::with_capacity(if use_compact { words.len() / 1024 * STEP_BYTES } else { 0 });
    if use_compact {
        for source in base.data.chunks_exact(1280) {
            data.extend_from_slice(&source[..256]);
            // Two adjacent significance bytes become two nibbles. Lane/half order is unchanged.
            for pair in source[256..].chunks_exact(2) {
                if pair[0] & 30 != 0 || pair[1] & 30 != 0 { return Err("nonzero omitted field".into()); }
                data.push(nibble(pair[0]) | nibble(pair[1]) << 4);
            }
        }
    } else { data = base.data; }
    let mut tiers = base.tiers;
    if use_compact { tiers[0] = (tiers[0] & 0x00ff_ffff) | TAG << 24; }
    Ok(Packed { rows, cols, data, blocks: base.blocks, block_base: base.block_base, tiers })
}

/// Host verification only. The device path expands significance in registers.
pub fn expanded(p: &Packed) -> Result<Tiered, String> {
    let count = p.rows.checked_mul(p.cols).ok_or("matrix count overflow")?;
    if !shape(p.rows, p.cols, count) { return Err("unsupported matrix shape".into()); }
    let step = match p.tiers[0] >> 24 { TAG => STEP_BYTES, pack::TIERED_TAG => 1280, _ => return Err("unknown representation".into()) };
    if p.data.len() != count / 1024 * step || p.block_base.len() != count / 1024 + 1 {
        return Err("matrix buffer length mismatch".into());
    }
    let mut data = Vec::with_capacity(count / 1024 * 1280);
    if p.compact() {
        for source in p.data.chunks_exact(STEP_BYTES) {
            data.extend_from_slice(&source[..256]);
            for &value in &source[256..] { data.extend_from_slice(&[byte(value), byte(value >> 4)]); }
        }
    } else { data.extend_from_slice(&p.data); }
    if !pack::tiered_blocks_hold(&data, &p.blocks, &p.block_base) {
        return Err("invalid exponent buffers".into());
    }
    let mut tiers = p.tiers;
    tiers[0] = (tiers[0] & 0x00ff_ffff) | pack::TIERED_TAG << 24;
    Ok(Tiered { rows: p.rows, cols: p.cols, data, blocks: p.blocks.clone(), block_base: p.block_base.clone(), tiers })
}

pub fn unpack(p: &Packed) -> Result<Vec<u16>, String> {
    let base = expanded(p)?;
    let mut words = vec![0; base.rows * base.cols];
    if !pack::unpack_tiered(&base, &mut words) { return Err("invalid matrix encoding".into()); }
    Ok(words)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glyd_lossy_study::approximate;

    #[test]
    fn every_compactable_bf16_pattern_and_matrix_order() {
        let mut words: Vec<u16> = (0..=u16::MAX).map(|x| approximate(x,3)).filter(|w| w & 15 == 0).collect();
        words.resize(words.len().div_ceil(1024)*1024, 0x8000);
        let p = pack_rounded(&words,64,words.len()/64).unwrap();
        assert!(p.compact());
        assert_eq!(unpack(&p).unwrap(),words);
        let original = pack::pack_tiered(&words,p.rows,p.cols);
        assert_eq!(expanded(&p).unwrap(),original);
        assert_eq!(original.data.len()-p.data.len(),words.len()/2);
        assert_eq!(pack::tiered_bytes(&original) as usize-p.resident_bytes(),words.len()/2);
    }

    #[test]
    fn every_bf16_pattern_keeps_nonfinite_payloads() {
        let words: Vec<u16> = (0..=u16::MAX).map(|x| approximate(x,3)).collect();
        let p = pack_rounded(&words,64,1024).unwrap();
        assert!(!p.compact());
        assert_eq!(unpack(&p).unwrap(),words);
    }

    #[test]
    fn shapes_and_existing_exponent_streams() {
        for (rows,cols) in [(64,16),(128,64),(192,80),(256,256)] {
            let words: Vec<u16> = (0..rows*cols).map(|i| {
                let exponent = [0,1,120,121,122,123,124,125,126,127,128,254][(i*7+i/31)%12];
                ((i%2) as u16)<<15 | exponent<<7 | ((i%8) as u16)<<4
            }).collect();
            let p = pack_rounded(&words,rows,cols).unwrap();
            assert!(p.compact()); assert_eq!(unpack(&p).unwrap(),words);
        }
    }

    #[test]
    fn invalid_input_and_buffers_are_refused() {
        assert!(pack_rounded(&[0;1024],63,16).is_err());
        assert!(pack_rounded(&[0;1024],64,0).is_err());
        assert!(pack_rounded(&[0x3f81;1024],64,16).is_err());
        let p = pack_rounded(&[0x3f80;1024],64,16).unwrap();
        let mut truncated = p.clone(); truncated.data.pop(); assert!(unpack(&truncated).is_err());
        let mut index = p.clone(); index.block_base[1] = i32::MAX; assert!(unpack(&index).is_err());
        let mut tag = p; tag.tiers[0] |= 0xff00_0000; assert!(unpack(&tag).is_err());
    }
}
