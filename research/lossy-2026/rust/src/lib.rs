//! Private reference representation for approximated BF16 weights. No runtime integration.
//! A block chooses a small exponent dictionary and fixed-width value fields, with full
//! rounded BF16 words for exceptions. Every byte, including dictionaries and headers, counts.
const MAGIC: &[u8; 8] = b"GLYDL27\x01";
const HEADER: usize = 28;
const BLOCK_HEADER: usize = 12;

/// Round to the nearest BF16 value retaining m fraction bits, with ties to even.
/// Preserve signed zero, infinities and NaN payloads. Finite overflow saturates.
pub fn approximate(w: u16, m: u8) -> u16 {
    assert!(m <= 7);
    if m == 7 || (w & 0x7f80) == 0x7f80 { return w; }
    let shift = 7 - m;
    let step = 1u32 << shift;
    let r = ((w as u32 + step / 2 - 1 + ((w as u32 >> shift) & 1)) & !(step - 1)) as u16;
    if (r & 0x7fff) >= 0x7f80 { (w & 0x8000) | (0x7f80 - step as u16) } else { r }
}

const fn crc_table() -> [u32; 256] {
    let mut table = [0; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut b = 0;
        while b < 8 { c = (c >> 1) ^ (0xedb88320u32 & 0u32.wrapping_sub(c & 1)); b += 1; }
        table[i] = c; i += 1;
    }
    table
}
const CRC_TABLE: [u32; 256] = crc_table();
fn crc(words: &[u16]) -> u32 {
    let mut c = !0u32;
    for w in words { for b in w.to_le_bytes() { c = (c >> 8) ^ CRC_TABLE[((c as u8) ^ b) as usize]; } }
    !c
}

fn put_bits(out: &mut [u8], pos: usize, width: usize, value: u32) {
    let first = pos / 8; let shift = pos % 8;
    let value = (value & ((1u32 << width) - 1)) << shift;
    for b in 0..(shift + width + 7)/8 { out[first+b] |= (value >> (8*b)) as u8; }
}
fn get_bits(data: &[u8], pos: usize, width: usize) -> u32 {
    let first = pos / 8; let shift = pos % 8;
    let mut value = 0;
    for b in 0..(shift + width + 7)/8 { value |= (data[first+b] as u32) << (8*b); }
    (value >> shift) & ((1u32 << width)-1)
}
fn u16le(v: &mut Vec<u8>, x: usize) { v.extend_from_slice(&(x as u16).to_le_bytes()); }
fn u32le(v: &mut Vec<u8>, x: usize) { v.extend_from_slice(&(x as u32).to_le_bytes()); }
fn block_header(n: usize, mode: u8, bits: u8, palette: usize, escapes: usize, payload: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(BLOCK_HEADER + payload + palette + 2*escapes);
    u16le(&mut v,n); v.extend_from_slice(&[mode,bits,palette as u8,0]);
    u16le(&mut v,escapes); u32le(&mut v,payload); v
}
fn encode_block(words: &[u16], m: u8) -> Vec<u8> {
    let mut freq = [0usize; 255];
    for w in words { let e = ((w >> 7) & 255) as usize; if e < 255 { freq[e] += 1; } }
    let mut ranked: Vec<usize> = (0..255).filter(|&e| freq[e] > 0).collect();
    ranked.sort_by_key(|&e| (std::cmp::Reverse(freq[e]),e));
    let mut best_bytes = BLOCK_HEADER + 2*words.len();
    let mut chosen = None;
    for ib in 2..=4u8 {
        let np = ranked.len().min((1usize << ib)-1);
        let ne = words.len() - ranked[..np].iter().map(|&e| freq[e]).sum::<usize>();
        let width = 1 + m as usize + ib as usize;
        let nb = (width*words.len()+7)/8;
        let bytes = BLOCK_HEADER + np + nb + 2*ne;
        if bytes < best_bytes { best_bytes = bytes; chosen = Some((ib,np,ne,nb,width)); }
    }
    let Some((ib,np,ne,nb,width)) = chosen else {
        let mut out = block_header(words.len(),0,0,0,0,2*words.len());
        for w in words { out.extend_from_slice(&w.to_le_bytes()); }
        return out;
    };
    let sentinel = (1usize << ib)-1;
    let mut lookup = [sentinel;256];
    for (i,&e) in ranked[..np].iter().enumerate() { lookup[e] = i; }
    let mut out = block_header(words.len(),1,ib,np,ne,nb);
    out.extend(ranked[..np].iter().map(|&e| e as u8));
    let at = out.len(); out.resize(at+nb,0);
    for (i,&w) in words.iter().enumerate() {
        let idx = lookup[((w>>7)&255) as usize];
        let significant = ((w >> 15) << m) | ((w & 127) >> (7-m));
        put_bits(&mut out[at..at+nb],i*width,width,((significant as u32)<<ib)|idx as u32);
        if idx == sentinel { out.extend_from_slice(&w.to_le_bytes()); }
    }
    debug_assert_eq!(out.len(), best_bytes);
    out
}

#[cfg(test)]
fn encode_block_reference(words: &[u16], m: u8) -> Vec<u8> {
    let mut best = block_header(words.len(),0,0,0,0,words.len()*2);
    for w in words { best.extend_from_slice(&w.to_le_bytes()); }
    let mut freq = [0usize; 255];
    for w in words { let e = ((w >> 7) & 255) as usize; if e < 255 { freq[e] += 1; } }
    let mut ranked: Vec<usize> = (0..255).filter(|&e| freq[e] > 0).collect();
    ranked.sort_by_key(|&e| (std::cmp::Reverse(freq[e]),e));
    for ib in 2..=4u8 {
        let sentinel = (1usize << ib)-1;
        let palette = &ranked[..ranked.len().min(sentinel)];
        let mut lookup = [sentinel;256];
        for (i,&e) in palette.iter().enumerate() { lookup[e]=i; }
        let width = 1 + m as usize + ib as usize;
        let mut fields = vec![0u8; (width*words.len()+7)/8];
        let mut exceptions = Vec::new();
        for (i,&w) in words.iter().enumerate() {
            let idx = lookup[((w>>7)&255) as usize];
            let significant = ((w >> 15) << m) | ((w & 127) >> (7-m));
            put_bits(&mut fields,i*width,width,((significant as u32)<<ib)|idx as u32);
            if idx == sentinel { exceptions.extend_from_slice(&w.to_le_bytes()); }
        }
        let bytes = BLOCK_HEADER + palette.len() + fields.len() + exceptions.len();
        if bytes < best.len() {
            best = block_header(words.len(),1,ib,palette.len(),exceptions.len()/2,fields.len());
            best.extend(palette.iter().map(|&e|e as u8));
            best.extend_from_slice(&fields); best.extend_from_slice(&exceptions);
        }
    }
    best
}

/// Encode rounded weights; m=7 is an exact control, including all BF16 bit patterns.
pub fn encode(words: &[u16], m: u8, block: usize) -> Result<Vec<u8>, String> {
    if m>7 || !(32..=4096).contains(&block) || !block.is_power_of_two() { return Err("invalid precision or block size".into()); }
    let rounded: Vec<u16> = words.iter().map(|&w|approximate(w,m)).collect();
    let mut v = Vec::with_capacity(HEADER+2*words.len());
    v.extend_from_slice(MAGIC); v.extend_from_slice(&[m,0]); u16le(&mut v,block);
    v.extend_from_slice(&(words.len() as u64).to_le_bytes());
    v.extend_from_slice(&crc(&rounded).to_le_bytes()); v.extend_from_slice(&0u32.to_le_bytes());
    for b in rounded.chunks(block) { v.extend_from_slice(&encode_block(b,m)); }
    Ok(v)
}

struct Reader<'a> { data: &'a [u8], at: usize }
impl<'a> Reader<'a> {
    fn take(&mut self,n: usize) -> Result<&'a [u8],String> {
        let end = self.at.checked_add(n).ok_or("length overflow")?;
        let v = self.data.get(self.at..end).ok_or("truncated representation")?;
        self.at=end; Ok(v)
    }
    fn byte(&mut self) -> Result<u8,String> { Ok(self.take(1)?[0]) }
    fn short(&mut self) -> Result<usize,String> { Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as usize) }
    fn int(&mut self) -> Result<u32,String> { Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap())) }
}

pub fn decode(data: &[u8]) -> Result<Vec<u16>,String> {
    let mut r = Reader{data,at:0};
    if r.take(8)? != MAGIC { return Err("not the study representation".into()); }
    let m=r.byte()?;
    if m>7 || r.byte()? != 0 { return Err("invalid precision or flags".into()); }
    let block=r.short()?;
    if !(32..=4096).contains(&block) || !block.is_power_of_two() { return Err("invalid block size".into()); }
    let count=u64::from_le_bytes(r.take(8)?.try_into().unwrap());
    if count > data.len().saturating_mul(8) as u64 { return Err("impossible value count".into()); }
    let count=count as usize;
    let checksum=r.int()?;
    if r.int()? != 0 { return Err("invalid reserved field".into()); }
    let mut out=Vec::with_capacity(count);
    while out.len()<count {
        let n=r.short()?; let mode=r.byte()?; let ib=r.byte()?; let np=r.byte()? as usize;
        if r.byte()? != 0 { return Err("invalid block flags".into()); }
        let ne=r.short()?; let nb=r.int()? as usize;
        if n==0 || n>block || n>count-out.len() { return Err("invalid block count".into()); }
        if mode==0 {
            if ib!=0 || np!=0 || ne!=0 || nb!=2*n { return Err("invalid raw block".into()); }
            for b in r.take(nb)?.chunks_exact(2) { out.push(u16::from_le_bytes(b.try_into().unwrap())); }
        } else if mode==1 {
            if !(2..=4).contains(&ib) || np >= (1usize<<ib) || ne>n { return Err("invalid dictionary block".into()); }
            let width=1+m as usize+ib as usize;
            if nb!=(n*width+7)/8 { return Err("invalid field bytes".into()); }
            let palette=r.take(np)?;
            let mut seen=[false;256];
            for &e in palette { if e==255 || seen[e as usize] { return Err("invalid exponent dictionary".into()); } seen[e as usize]=true; }
            let fields=r.take(nb)?; let exceptions=r.take(2*ne)?;
            if (n*width)%8 != 0 && fields[nb-1] >> ((n*width)%8) != 0 { return Err("nonzero padding".into()); }
            let sentinel=(1usize<<ib)-1; let mut used=0;
            for i in 0..n {
                let v=get_bits(fields,i*width,width);
                let idx=(v as usize)&sentinel;
                if idx==sentinel {
                    if used>=ne { return Err("missing exception".into()); }
                    out.push(u16::from_le_bytes(exceptions[2*used..2*used+2].try_into().unwrap())); used+=1;
                } else {
                    let e=*palette.get(idx).ok_or("invalid exponent index")? as u16;
                    let sig=(v>>ib) as u16;
                    out.push(((sig>>m)<<15)|(e<<7)|((sig&((1<<m)-1))<<(7-m)));
                }
            }
            if used!=ne { return Err("unused exceptions".into()); }
        } else { return Err("invalid block mode".into()); }
    }
    if r.at != data.len() || crc(&out)!=checksum { return Err("trailing bytes or checksum mismatch".into()); }
    Ok(out)
}

pub fn as_float(w: u16) -> f64 { f32::from_bits((w as u32)<<16) as f64 }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn optimized_encoder_preserves_reference_bytes() {
        let all: Vec<u16> = (0..=u16::MAX).collect();
        let mut seed = 0x6c796431u32;
        let random: Vec<u16> = (0..8193).map(|_| {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            (seed as u16 & 0x83ff) | 0x3a00
        }).collect();
        for m in 0..=7 {
            for source in [&all, &random, &vec![0x7fc1;1025], &vec![0;1025]] {
                let rounded: Vec<u16> = source.iter().map(|&w| approximate(w,m)).collect();
                for block in [32,256,1024,4096] { for row in rounded.chunks(block) {
                    assert_eq!(encode_block(row,m),encode_block_reference(row,m));
                }}
                let mut c = !0u32;
                for w in &rounded { for b in w.to_le_bytes() {
                    c ^= b as u32;
                    for _ in 0..8 { c = (c >> 1) ^ (0xedb88320u32 & 0u32.wrapping_sub(c & 1)); }
                }}
                assert_eq!(crc(&rounded),!c);
            }
        }
    }
    #[test]
    fn byte_fields_match_independent_bit_order() {
        for width in 1..=12 { for shift in 0..8 {
            for value in 0..(1u32 << width) {
                let mut expected = [0u8;3];
                for bit in 0..width { expected[(shift+bit)/8] |= (((value>>bit)&1) as u8) << ((shift+bit)%8); }
                let mut got = [0u8;3]; put_bits(&mut got,shift,width,value);
                assert_eq!(got,expected); assert_eq!(get_bits(&got,shift,width),value);
            }
        }}
    }
    #[test]
    fn every_bf16_pattern_round_trips_after_approximation() {
        let words: Vec<u16>=(0..=u16::MAX).collect();
        for m in 0..=7 {
            let packed=encode(&words,m,256).unwrap();
            let got=decode(&packed).unwrap();
            let want: Vec<u16>=words.iter().map(|&w|approximate(w,m)).collect();
            assert_eq!(got,want);
            if m==7 { assert_eq!(got,words); }
            for (&a,&b) in words.iter().zip(&got) {
                if a&0x7f80!=0x7f80 { assert!(b&0x7f80!=0x7f80); }
                else { assert_eq!(a,b); }
            }
        }
    }
    #[test]
    fn rounding_ties_sign_and_overflow() {
        assert_eq!(approximate(0x3f84,4),0x3f80);
        assert_eq!(approximate(0x3f8c,4),0x3f90);
        assert_eq!(approximate(0xbf84,4),0xbf80);
        assert_eq!(approximate(0xbf8c,4),0xbf90);
        assert_eq!(approximate(0x7f7f,0),0x7f00);
        assert_eq!(approximate(0xff7f,0),0xff00);
        for m in 0..=7 { assert_eq!(approximate(0x8000,m),0x8000); assert_eq!(approximate(0,m),0); }
    }
    #[test]
    fn dictionary_tail_and_exceptions() {
        let mut w: Vec<u16>=(0..2051).map(|i|0x3c80|((i%128) as u16)|(((i%2) as u16)<<15)).collect();
        w[11]=0x7fc3; w[200]=0x0001; w[513]=0xff80;
        for m in 0..=7 { for block in [32,128,1024,4096] {
            let b=encode(&w,m,block).unwrap();
            assert_eq!(decode(&b).unwrap(),w.iter().map(|&x|approximate(x,m)).collect::<Vec<_>>());
            if block==1024 { assert!(b.len()<w.len()*2); }
        }}
        assert_eq!(decode(&encode(&[],3,256).unwrap()).unwrap(),Vec::<u16>::new());
    }
    #[test]
    fn malformed_inputs_are_rejected() {
        let b=encode(&vec![0x3c80;256],3,128).unwrap();
        for end in 0..b.len() { assert!(decode(&b[..end]).is_err()); }
        let mut bad=b.clone(); bad.push(0); assert!(decode(&bad).is_err());
        for offset in [0,8,9,10,20,24,28,30,31,32,33,34,36,b.len()-1] {
            let mut bad=b.clone(); bad[offset]^=0x80; assert!(decode(&bad).is_err(),"offset {offset}");
        }
        let mut bad=b.clone(); bad[12..20].copy_from_slice(&u64::MAX.to_le_bytes()); assert!(decode(&bad).is_err());
        assert!(encode(&[],8,128).is_err()); assert!(encode(&[],3,7).is_err());
    }
}
