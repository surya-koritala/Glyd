//! The pixels of a PNG or a JPEG as RGB8, the engine's own decoding (no image library): PNG's filtered scanlines of any bit depth and color type, interlaced or not, inflated by flate2;
//! baseline and progressive JPEG (8-bit samples, one or three components, whole sampling factors) decoded the way libjpeg-turbo does by default (the integer inverse DCT, the smooth
//! chroma upsampling, the fixed-point YCbCr conversion), so that an image gives the pixels PIL gives the reference's processor. Alpha is dropped and gray repeated (PIL's convert("RGB")).

use std::io::Read;

/// Pixels a decoded image may have (a phone's 48 M; PIL refuses past twice its 89 M).
pub const MAX_PIXELS: usize = 50_000_000;

/// (width, height, RGB8 rows) of a PNG or a JPEG.
pub fn rgb(bytes: &[u8]) -> Result<(usize, usize, Vec<u8>), String> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        png(bytes)
    } else if bytes.starts_with(&[0xFF, 0xD8]) {
        jpeg(bytes)
    } else {
        Err("an image that is neither PNG nor JPEG".into())
    }
}

fn size_ok(w: usize, h: usize) -> Result<(), String> {
    if w == 0 || h == 0 || w * h > MAX_PIXELS {
        return Err(format!("an image of {w} x {h} pixels: at most {} M pixels", MAX_PIXELS / 1_000_000));
    }
    Ok(())
}

// ---- PNG -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

fn png(b: &[u8]) -> Result<(usize, usize, Vec<u8>), String> {
    let bad = || "a PNG that could not be read".to_string();
    let be = |p: &[u8]| u32::from_be_bytes([p[0], p[1], p[2], p[3]]) as usize;
    let (mut head, mut palette, mut z): (Option<&[u8]>, &[u8], Vec<u8>) = (None, &[], Vec::new());
    let mut at = 8;
    while at + 12 <= b.len() {
        let len = be(&b[at..]);
        let data = b.get(at + 8..at + 8 + len).ok_or_else(bad)?;
        match &b[at + 4..at + 8] {
            b"IHDR" if len >= 13 => head = Some(data),
            b"PLTE" => palette = data,
            b"IDAT" => z.extend_from_slice(data),
            b"IEND" => break,
            _ => {}
        }
        at += 12 + len;
    }
    let h0 = head.ok_or_else(bad)?;
    let (w, h, depth, color, interlace) = (be(h0), be(&h0[4..]), h0[8] as usize, h0[9], h0[12]);
    size_ok(w, h)?;
    let chans = match (color, depth) {
        (0, 1 | 2 | 4 | 8 | 16) => 1,
        (3, 1 | 2 | 4 | 8) => 1,
        (2, 8 | 16) => 3,
        (4, 8 | 16) => 2,
        (6, 8 | 16) => 4,
        _ => return Err(format!("a PNG of color type {color} at depth {depth}")),
    };
    if interlace > 1 {
        return Err(bad());
    }
    let bits = chans * depth;
    let bpp = (bits / 8).max(1);
    let passes: &[(usize, usize, usize, usize)] =
        if interlace == 1 { &[(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)] } else { &[(0, 0, 1, 1)] };
    let want: usize = passes
        .iter()
        .map(|&(x0, y0, dx, dy)| {
            let (pw, ph) = (w.saturating_sub(x0).div_ceil(dx), h.saturating_sub(y0).div_ceil(dy));
            if pw == 0 {
                0
            } else {
                ph * (1 + (pw * bits).div_ceil(8))
            }
        })
        .sum();
    let mut raw = Vec::with_capacity(want);
    flate2::read::ZlibDecoder::new(&z[..]).take(want as u64).read_to_end(&mut raw).map_err(|_| bad())?;
    if raw.len() < want {
        return Err(bad());
    }
    // a sample of a line: 16-bit ones by their high byte (PIL's), those under 8 bits as they are (scaled below where they are gray)
    let sample = |line: &[u8], i: usize| -> u8 {
        match depth {
            8 => line[i],
            16 => line[2 * i],
            d => (line[i * d / 8] >> (8 - d - i * d % 8)) & ((1u16 << d) - 1) as u8,
        }
    };
    let gray = |v: u8| if depth < 8 { (v as usize * 255 / ((1 << depth) - 1)) as u8 } else { v };
    let mut out = vec![0u8; w * h * 3];
    let mut pos = 0;
    for &(x0, y0, dx, dy) in passes {
        let (pw, ph) = (w.saturating_sub(x0).div_ceil(dx), h.saturating_sub(y0).div_ceil(dy));
        if pw == 0 || ph == 0 {
            continue;
        }
        let stride = (pw * bits).div_ceil(8);
        let mut prev = vec![0u8; stride];
        for py in 0..ph {
            let f = raw[pos];
            let mut line = raw[pos + 1..pos + 1 + stride].to_vec();
            pos += 1 + stride;
            for i in 0..stride {
                let (a, up, c) = if i >= bpp { (line[i - bpp] as i16, prev[i] as i16, prev[i - bpp] as i16) } else { (0, prev[i] as i16, 0) };
                let add = match f {
                    0 => 0,
                    1 => a,
                    2 => up,
                    3 => (a + up) / 2,
                    4 => {
                        let p = a + up - c;
                        let (pa, pb, pc) = ((p - a).abs(), (p - up).abs(), (p - c).abs());
                        if pa <= pb && pa <= pc {
                            a
                        } else if pb <= pc {
                            up
                        } else {
                            c
                        }
                    }
                    _ => return Err(bad()),
                };
                line[i] = line[i].wrapping_add(add as u8);
            }
            for px in 0..pw {
                let p = match color {
                    0 | 4 => [gray(sample(&line, px * chans)); 3],
                    3 => {
                        let k = sample(&line, px) as usize * 3;
                        palette.get(k..k + 3).map_or([0; 3], |c| [c[0], c[1], c[2]])
                    }
                    _ => [sample(&line, px * chans), sample(&line, px * chans + 1), sample(&line, px * chans + 2)],
                };
                let o = ((y0 + py * dy) * w + x0 + px * dx) * 3;
                out[o..o + 3].copy_from_slice(&p);
            }
            prev = line;
        }
    }
    Ok((w, h, out))
}

// ---- JPEG ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// The order a block's 64 coefficients come in (the zigzag), as positions in the 8 x 8 block.
const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29,
    22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// A table of the scan's variable-length codes (a DHT segment's): codes of up to 9 bits looked up at once, longer ones by their length's first and last code.
#[derive(Clone)]
struct Codes {
    fast: Vec<(u8, u8)>,
    maxcode: [i32; 17],
    first: [i32; 17],
    index: [usize; 17],
    vals: Vec<u8>,
}

impl Codes {
    fn new(counts: &[u8; 16], vals: Vec<u8>) -> Result<Codes, String> {
        let mut c = Codes { fast: vec![(0, 0); 512], maxcode: [-1; 17], first: [0; 17], index: [0; 17], vals };
        let (mut code, mut k) = (0i32, 0usize);
        for len in 1..=16 {
            let n = counts[len - 1] as usize;
            c.index[len] = k;
            c.first[len] = code;
            for _ in 0..n {
                if code >= 1 << len {
                    return Err("a JPEG code table that does not fit its lengths".into());
                }
                if len <= 9 {
                    let lo = (code as usize) << (9 - len);
                    for e in &mut c.fast[lo..lo + (1 << (9 - len))] {
                        *e = (len as u8, *c.vals.get(k).ok_or("a JPEG code table with fewer values than codes")?);
                    }
                }
                code += 1;
                k += 1;
            }
            if n > 0 {
                c.maxcode[len] = code - 1;
            }
            code <<= 1;
        }
        Ok(c)
    }
}

/// The bits of a scan: 0xFF 0x00 read as 0xFF; a marker stops them (zeros after it, as libjpeg reads a scan that ends early), and is kept for a restart.
struct Bits<'a> {
    d: &'a [u8],
    at: usize,
    acc: u64,
    n: u32,
    marker: Option<u8>,
}

impl Bits<'_> {
    fn fill(&mut self) {
        while self.n <= 56 {
            let mut byte = 0u64;
            if self.marker.is_none() && self.at < self.d.len() {
                let b = self.d[self.at];
                if b == 0xFF {
                    match self.d.get(self.at + 1) {
                        Some(0) => {
                            byte = 0xFF;
                            self.at += 2;
                        }
                        m => self.marker = Some(m.copied().unwrap_or(0xD9)),
                    }
                } else {
                    byte = b as u64;
                    self.at += 1;
                }
            }
            self.acc |= byte << (56 - self.n);
            self.n += 8;
        }
    }

    fn peek(&mut self, k: u32) -> u32 {
        self.fill();
        (self.acc >> (64 - k)) as u32
    }

    fn skip(&mut self, k: u32) {
        self.acc <<= k;
        self.n -= k;
    }

    fn get(&mut self, k: u32) -> u32 {
        if k == 0 {
            return 0;
        }
        let v = self.peek(k);
        self.skip(k);
        v
    }

    /// The value of a `s`-bit difference (its sign in its first bit; a size past 16, a broken file's, read as 16).
    fn extend(&mut self, s: u32) -> i32 {
        let s = s.min(16);
        if s == 0 {
            return 0;
        }
        let v = self.get(s) as i32;
        if v < 1 << (s - 1) {
            v - (1 << s) + 1
        } else {
            v
        }
    }

    fn decode(&mut self, t: &Codes) -> Result<u8, String> {
        let p = self.peek(16) as i32;
        let (len, v) = t.fast[(p >> 7) as usize];
        if len > 0 {
            self.skip(len as u32);
            return Ok(v);
        }
        for l in 10..=16 {
            let code = p >> (16 - l);
            if code <= t.maxcode[l] {
                self.skip(l as u32);
                return t.vals.get(t.index[l] + (code - t.first[l]) as usize).copied().ok_or_else(|| "a JPEG code past its table".to_string());
            }
        }
        Err("a JPEG with a code that is in no table".into())
    }

    /// A restart: the bits left dropped, the data read on past the RSTn marker that ends the interval (to the next marker of any kind, which is passed where it is a restart).
    fn restart(&mut self) {
        let mut p = self.at;
        while p + 1 < self.d.len() {
            if self.d[p] == 0xFF && self.d[p + 1] != 0 && self.d[p + 1] != 0xFF {
                if (0xD0..=0xD7).contains(&self.d[p + 1]) {
                    p += 2;
                }
                break;
            }
            p += 1;
        }
        (self.at, self.acc, self.n, self.marker) = (p, 0, 0, None);
    }
}

struct Comp {
    id: u8,
    h: usize,
    v: usize,
    tq: usize,
    /// its blocks across and down (whole MCUs), its coefficients (natural order, 64 a block), the quantization table it was read with (latched at its first scan)
    bw: usize,
    bh: usize,
    coef: Vec<i16>,
    q: Option<[i32; 64]>,
    dc: usize,
    ac: usize,
    pred: i32,
}

fn jpeg(b: &[u8]) -> Result<(usize, usize, Vec<u8>), String> {
    let bad = || "a JPEG that could not be read".to_string();
    let be16 = |p: usize| -> Result<usize, String> { Ok(((*b.get(p).ok_or_else(bad)? as usize) << 8) | *b.get(p + 1).ok_or_else(bad)? as usize) };
    let mut qt: [Option<[i32; 64]>; 4] = [None; 4];
    let (mut dct, mut act): (Vec<Option<Codes>>, Vec<Option<Codes>>) = (vec![None; 4], vec![None; 4]);
    let mut comps: Vec<Comp> = Vec::new();
    let (mut w, mut h, mut progressive, mut restart) = (0usize, 0usize, false, 0usize);
    let (mut adobe, mut jfif): (Option<u8>, bool) = (None, false);
    let (mut hmax, mut vmax, mut mcux, mut mcuy) = (1, 1, 0, 0);
    let (mut at, mut scans) = (2, 0);
    loop {
        // the next marker (fill bytes 0xFF skipped); a file that ends after a scan with no EOI ends there (libjpeg's warning, not an error)
        while b.get(at) != Some(&0xFF) {
            if at >= b.len() {
                if scans > 0 {
                    break;
                }
                return Err(bad());
            }
            at += 1;
        }
        if at >= b.len() {
            break;
        }
        while b.get(at) == Some(&0xFF) {
            at += 1;
        }
        let m = *b.get(at).ok_or_else(bad)?;
        at += 1;
        match m {
            0xD8 | 0x01 | 0xD0..=0xD7 => continue,
            0xD9 => break,
            _ => {}
        }
        let len = be16(at)?;
        let seg = b.get(at + 2..at + len).ok_or_else(bad)?;
        at += len;
        match m {
            0xC0..=0xC2 => {
                if seg.len() < 6 || seg[0] != 8 {
                    return Err("a JPEG whose samples are not 8 bits deep".into());
                }
                if !comps.is_empty() {
                    // (a second frame header: refused, so a file cannot ask for the frame's buffers again and again)
                    return Err(bad());
                }
                progressive = m == 0xC2;
                h = (seg[1] as usize) << 8 | seg[2] as usize;
                w = (seg[3] as usize) << 8 | seg[4] as usize;
                size_ok(w, h)?;
                let n = seg[5] as usize;
                if n != 1 && n != 3 {
                    return Err(format!("a JPEG of {n} components: send an RGB or gray one"));
                }
                for i in 0..n {
                    let c = seg.get(6 + 3 * i..9 + 3 * i).ok_or_else(bad)?;
                    let (hs, vs) = ((c[1] >> 4) as usize, (c[1] & 15) as usize);
                    if !(1..=4).contains(&hs) || !(1..=4).contains(&vs) || c[2] > 3 {
                        return Err(bad());
                    }
                    comps.push(Comp { id: c[0], h: hs, v: vs, tq: c[2] as usize, bw: 0, bh: 0, coef: Vec::new(), q: None, dc: 0, ac: 0, pred: 0 });
                }
                hmax = comps.iter().map(|c| c.h).max().unwrap_or(1);
                vmax = comps.iter().map(|c| c.v).max().unwrap_or(1);
                if comps.iter().any(|c| hmax % c.h != 0 || vmax % c.v != 0) {
                    return Err("a JPEG with sampling factors that do not divide".into());
                }
                (mcux, mcuy) = (w.div_ceil(8 * hmax), h.div_ceil(8 * vmax));
                for c in &mut comps {
                    (c.bw, c.bh) = (mcux * c.h, mcuy * c.v);
                    c.coef = vec![0; c.bw * c.bh * 64];
                }
            }
            0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF => {
                return Err("a JPEG of a kind this engine does not read (lossless, hierarchical or arithmetic-coded)".into())
            }
            0xC4 => {
                let mut p = 0;
                while p + 17 <= seg.len() {
                    let (class, id) = (seg[p] >> 4, (seg[p] & 15) as usize);
                    let counts: [u8; 16] = seg[p + 1..p + 17].try_into().map_err(|_| bad())?;
                    let n: usize = counts.iter().map(|&x| x as usize).sum();
                    let vals = seg.get(p + 17..p + 17 + n).ok_or_else(bad)?.to_vec();
                    if id > 3 {
                        return Err(bad());
                    }
                    let t = Some(Codes::new(&counts, vals)?);
                    if class == 0 {
                        dct[id] = t
                    } else {
                        act[id] = t
                    }
                    p += 17 + n;
                }
            }
            0xDB => {
                let mut p = 0;
                while p < seg.len() {
                    let (wide, id) = (seg[p] >> 4 == 1, (seg[p] & 15) as usize);
                    let step = if wide { 2 } else { 1 };
                    let body = seg.get(p + 1..p + 1 + 64 * step).ok_or_else(bad)?;
                    if id > 3 {
                        return Err(bad());
                    }
                    let mut t = [0i32; 64];
                    for k in 0..64 {
                        t[ZIGZAG[k]] = if wide { (body[2 * k] as i32) << 8 | body[2 * k + 1] as i32 } else { body[k] as i32 };
                    }
                    qt[id] = Some(t);
                    p += 1 + 64 * step;
                }
            }
            0xDD => restart = (*seg.first().ok_or_else(bad)? as usize) << 8 | *seg.get(1).ok_or_else(bad)? as usize,
            0xE0 if seg.starts_with(b"JFIF\0") => jfif = true,
            0xEE if seg.starts_with(b"Adobe") && seg.len() >= 12 => adobe = Some(seg[11]),
            0xDC => return Err("a JPEG with a DNL marker".into()),
            0xDA => {
                if comps.is_empty() {
                    return Err(bad());
                }
                let ns = *seg.first().ok_or_else(bad)? as usize;
                let tail = seg.get(1 + 2 * ns..1 + 2 * ns + 3).ok_or_else(bad)?;
                let (ss, se, ah, al) = (tail[0] as usize, tail[1] as usize, tail[2] >> 4, tail[2] & 15);
                let mut sc = Vec::with_capacity(ns);
                for i in 0..ns {
                    let (id, t) = (seg[1 + 2 * i], seg[2 + 2 * i]);
                    let ci = comps.iter().position(|c| c.id == id).ok_or_else(bad)?;
                    let c = &mut comps[ci];
                    (c.dc, c.ac, c.pred) = ((t >> 4) as usize & 3, (t & 15) as usize & 3, 0);
                    if c.q.is_none() {
                        c.q = Some(qt[c.tq].ok_or("a JPEG with no quantization table for a component")?);
                    }
                    sc.push(ci);
                }
                if se > 63 || ss > se || (progressive && ((ss == 0) != (se == 0) || (ss > 0 && ns != 1))) || (!progressive && (ss != 0 || se != 63)) {
                    return Err(bad());
                }
                at = scan(b, at, &mut comps, &sc, (ss, se, ah as u32, al as u32), progressive, restart, (mcux, mcuy), (w, h, hmax, vmax), &dct, &act)?;
                scans += 1;
            }
            _ => {}
        }
    }
    if comps.iter().any(|c| c.q.is_none()) {
        return Err(bad());
    }
    // each component's samples: its blocks dequantized and turned back, then brought to the image's size
    let planes: Vec<Vec<u8>> = comps.iter().map(|c| upsample(&samples(c), c, (w, h, hmax, vmax))).collect();
    let mut out = vec![0u8; w * h * 3];
    // (RGB samples where libjpeg reads them so: no JFIF marker, and Adobe's transform 0, or no Adobe marker and the components named R, G, B; else YCbCr)
    let rgb_data = comps.len() == 3 && !jfif && (adobe == Some(0) || (adobe.is_none() && comps.iter().map(|c| c.id).eq([b'R', b'G', b'B'])));
    if comps.len() == 1 {
        for (o, &y) in out.chunks_exact_mut(3).zip(&planes[0]) {
            o.copy_from_slice(&[y; 3]);
        }
    } else if rgb_data {
        for (i, o) in out.chunks_exact_mut(3).enumerate() {
            o.copy_from_slice(&[planes[0][i], planes[1][i], planes[2][i]]);
        }
    } else {
        // libjpeg's tables: R = Y + 1.402 Cr, G = Y - 0.34414 Cb - 0.71414 Cr, B = Y + 1.772 Cb, in 16-bit fixed point, each rounded as it has them
        let fix = |x: f64| (x * 65536.0 + 0.5) as i64;
        let (rr, gb, gr, bb, half) = (fix(1.40200), fix(0.34414), fix(0.71414), fix(1.77200), 1i64 << 15);
        let clamp = |x: i64| x.clamp(0, 255) as u8;
        for (i, o) in out.chunks_exact_mut(3).enumerate() {
            let (y, cb, cr) = (planes[0][i] as i64, planes[1][i] as i64 - 128, planes[2][i] as i64 - 128);
            let r = y + ((rr * cr + half) >> 16);
            let g = y + ((-gb * cb + half - gr * cr) >> 16);
            let bl = y + ((bb * cb + half) >> 16);
            o.copy_from_slice(&[clamp(r), clamp(g), clamp(bl)]);
        }
    }
    Ok((w, h, out))
}

/// One scan from `at` (after its header) into the components' coefficients; where the next segment starts.
#[allow(clippy::too_many_arguments)]
fn scan(
    b: &[u8],
    at: usize,
    comps: &mut [Comp],
    sc: &[usize],
    (ss, se, ah, al): (usize, usize, u32, u32),
    progressive: bool,
    restart: usize,
    (mcux, mcuy): (usize, usize),
    (w, h, hmax, vmax): (usize, usize, usize, usize),
    dct: &[Option<Codes>],
    act: &[Option<Codes>],
) -> Result<usize, String> {
    let mut bits = Bits { d: b, at, acc: 0, n: 0, marker: None };
    let table = |t: &[Option<Codes>], i: usize| t[i].clone().ok_or_else(|| "a JPEG scan with a code table it does not define".to_string());
    let tabs: Vec<(Option<Codes>, Option<Codes>)> = sc
        .iter()
        .map(|&ci| {
            Ok::<_, String>((
                if ss == 0 && ah == 0 { Some(table(dct, comps[ci].dc)?) } else { None },
                if se > 0 { Some(table(act, comps[ci].ac)?) } else { None },
            ))
        })
        .collect::<Result<_, _>>()?;
    let mut eobrun = 0u32;
    // the blocks of each MCU: an interleaved scan's MCU has each component's h x v blocks; a scan of one component has its blocks one at a time, as many as its samples cover
    let (units_x, units_y) = if sc.len() == 1 {
        let c = &comps[sc[0]];
        ((w * c.h).div_ceil(hmax).div_ceil(8), (h * c.v).div_ceil(vmax).div_ceil(8))
    } else {
        (mcux, mcuy)
    };
    let total = units_x * units_y;
    for u in 0..total {
        if restart > 0 && u > 0 && u % restart == 0 {
            bits.restart();
            eobrun = 0;
            for &ci in sc {
                comps[ci].pred = 0;
            }
        }
        let (ux, uy) = (u % units_x, u / units_x);
        for (k, &ci) in sc.iter().enumerate() {
            let c = &mut comps[ci];
            let (nh, nv) = if sc.len() == 1 { (1, 1) } else { (c.h, c.v) };
            for by in 0..nv {
                for bx in 0..nh {
                    let (x, y) = if sc.len() == 1 { (ux, uy) } else { (ux * c.h + bx, uy * c.v + by) };
                    let base = (y * c.bw + x) * 64;
                    let blk = &mut c.coef[base..base + 64];
                    if !progressive {
                        let s = bits.decode(tabs[k].0.as_ref().ok_or("a JPEG scan with no table")?)? as u32;
                        c.pred = c.pred.wrapping_add(bits.extend(s));
                        blk[0] = c.pred as i16;
                        let t = tabs[k].1.as_ref().ok_or("a JPEG scan with no table")?;
                        let mut i = 1;
                        while i < 64 {
                            let rs = bits.decode(t)?;
                            let (r, s) = ((rs >> 4) as usize, (rs & 15) as u32);
                            if s == 0 {
                                if r != 15 {
                                    break;
                                }
                                i += 16;
                                continue;
                            }
                            i += r;
                            if i > 63 {
                                return Err("a JPEG block past its 64 values".into());
                            }
                            blk[ZIGZAG[i]] = bits.extend(s) as i16;
                            i += 1;
                        }
                    } else if ss == 0 {
                        if ah == 0 {
                            let s = bits.decode(tabs[k].0.as_ref().ok_or("a JPEG scan with no table")?)? as u32;
                            c.pred = c.pred.wrapping_add(bits.extend(s));
                            blk[0] = c.pred.wrapping_shl(al) as i16;
                        } else if bits.get(1) == 1 {
                            blk[0] |= (1i32 << al) as i16;
                        }
                    } else {
                        let t = tabs[k].1.as_ref().ok_or("a JPEG scan with no table")?;
                        if ah == 0 {
                            ac_first(&mut bits, t, blk, ss, se, al, &mut eobrun)?;
                        } else {
                            ac_refine(&mut bits, t, blk, ss, se, al, &mut eobrun)?;
                        }
                    }
                }
            }
        }
    }
    // the scan's end: past its data to the next marker that is not a restart
    let mut p = bits.at;
    loop {
        match (b.get(p), b.get(p + 1)) {
            (Some(0xFF), Some(0x00 | 0xD0..=0xD7 | 0xFF)) => p += if b[p + 1] == 0xFF { 1 } else { 2 },
            (Some(0xFF), Some(_)) | (None, _) | (Some(_), None) => break,
            _ => p += 1,
        }
    }
    Ok(p)
}

/// A progressive scan's first pass of AC coefficients `ss..=se` of a block (shifted by `al`), with its end-of-band run.
fn ac_first(bits: &mut Bits, t: &Codes, blk: &mut [i16], ss: usize, se: usize, al: u32, eobrun: &mut u32) -> Result<(), String> {
    if *eobrun > 0 {
        *eobrun -= 1;
        return Ok(());
    }
    let mut k = ss;
    while k <= se {
        let rs = bits.decode(t)?;
        let (r, s) = ((rs >> 4) as u32, (rs & 15) as u32);
        if s != 0 {
            k += r as usize;
            if k > se {
                return Err("a JPEG block past its band".into());
            }
            blk[ZIGZAG[k]] = bits.extend(s).wrapping_mul(1 << al) as i16;
        } else if r == 15 {
            k += 15;
        } else {
            *eobrun = (1 << r) + bits.get(r) - 1;
            break;
        }
        k += 1;
    }
    Ok(())
}

/// A progressive scan's later pass: one more bit of the coefficients `ss..=se` already nonzero, and the ones that become nonzero now (of the size 1 << al).
fn ac_refine(bits: &mut Bits, t: &Codes, blk: &mut [i16], ss: usize, se: usize, al: u32, eobrun: &mut u32) -> Result<(), String> {
    let (p1, m1) = ((1i32 << al) as i16, (-1i32 << al) as i16);
    let mut k = ss;
    let refine = |bits: &mut Bits, v: &mut i16| {
        if bits.get(1) == 1 && *v & p1 == 0 {
            *v = v.wrapping_add(if *v >= 0 { p1 } else { m1 });
        }
    };
    if *eobrun == 0 {
        while k <= se {
            let rs = bits.decode(t)?;
            let (mut r, s) = ((rs >> 4) as i32, rs & 15);
            let mut val = 0;
            if s != 0 {
                val = if bits.get(1) == 1 { p1 } else { m1 };
            } else if r != 15 {
                *eobrun = (1 << r) + bits.get(r as u32);
                break;
            }
            while k <= se {
                let z = ZIGZAG[k];
                if blk[z] != 0 {
                    refine(bits, &mut blk[z]);
                } else {
                    r -= 1;
                    if r < 0 {
                        break;
                    }
                }
                k += 1;
            }
            if val != 0 && k <= se {
                blk[ZIGZAG[k]] = val;
            }
            k += 1;
        }
    }
    if *eobrun > 0 {
        while k <= se {
            let z = ZIGZAG[k];
            if blk[z] != 0 {
                refine(bits, &mut blk[z]);
            }
            k += 1;
        }
        *eobrun -= 1;
    }
    Ok(())
}

/// A component's samples (its blocks across x 8 by its blocks down x 8): each block dequantized and through libjpeg's integer inverse DCT (jidctint's, 13-bit constants, 2 bits kept
/// between the passes), clamped as its range table does.
fn samples(c: &Comp) -> Vec<u8> {
    const C: [i64; 12] = [2446, 3196, 4433, 6270, 7373, 9633, 12299, 15137, 16069, 16819, 20995, 25172];
    let [f0298, f0390, f0541, f0765, f0899, f1175, f1501, f1847, f1961, f2053, f2562, f3072] = C;
    let q = c.q.unwrap_or([1; 64]);
    let stride = c.bw * 8;
    let mut out = vec![0u8; stride * c.bh * 8];
    // the even and odd halves of one 1-D pass on (x0..x7), the eight outputs before their shift
    let pass = |x: [i64; 8]| -> [i64; 8] {
        let z1 = (x[2] + x[6]) * f0541;
        let (t2, t3) = (z1 - x[6] * f1847, z1 + x[2] * f0765);
        let (t0, t1) = ((x[0] + x[4]) << 13, (x[0] - x[4]) << 13);
        let (t10, t13, t11, t12) = (t0 + t3, t0 - t3, t1 + t2, t1 - t2);
        let (o0, o1, o2, o3) = (x[7], x[5], x[3], x[1]);
        let (z1, z2, z3, z4) = (o0 + o3, o1 + o2, o0 + o2, o1 + o3);
        let z5 = (z3 + z4) * f1175;
        let (o0, o1, o2, o3) = (o0 * f0298, o1 * f2053, o2 * f3072, o3 * f1501);
        let (z1, z2, z3, z4) = (-z1 * f0899, -z2 * f2562, -z3 * f1961 + z5, -z4 * f0390 + z5);
        let (o0, o1, o2, o3) = (o0 + z1 + z3, o1 + z2 + z4, o2 + z2 + z3, o3 + z1 + z4);
        [t10 + o3, t11 + o2, t12 + o1, t13 + o0, t13 - o0, t12 - o1, t11 - o2, t10 - o3]
    };
    let limit = |v: i64| -> u8 { ((((v & 1023) ^ 512) - 512) + 128).clamp(0, 255) as u8 };
    for by in 0..c.bh {
        for bx in 0..c.bw {
            let blk = &c.coef[(by * c.bw + bx) * 64..][..64];
            let mut ws = [0i64; 64];
            for col in 0..8 {
                let x: [i64; 8] = std::array::from_fn(|r| blk[r * 8 + col] as i64 * q[r * 8 + col] as i64);
                let y = if x[1..].iter().all(|&v| v == 0) { [x[0] << 2; 8] } else { pass(x).map(|v| (v + (1 << 10)) >> 11) };
                for r in 0..8 {
                    ws[r * 8 + col] = y[r];
                }
            }
            for row in 0..8 {
                let x: [i64; 8] = std::array::from_fn(|i| ws[row * 8 + i]);
                let o = &mut out[(by * 8 + row) * stride + bx * 8..][..8];
                if x[1..].iter().all(|&v| v == 0) {
                    o.fill(limit((x[0] + 16) >> 5));
                } else {
                    for (d, v) in o.iter_mut().zip(pass(x)) {
                        *d = limit((v + (1 << 17)) >> 18);
                    }
                }
            }
        }
    }
    out
}

/// A component's samples brought to the image's w x h (`hmax`, `vmax` its largest sampling factors) as libjpeg-turbo does with smooth upsampling on: 2:1 across (where wider than 2
/// samples), 2:1 down, or both, each output 3/4 its nearest sample and 1/4 the next nearest (both ways: 9/16, 3/16, 3/16, 1/16) with libjpeg's alternating rounding; the same size as it
/// is; any other whole factor by repetition. The row above the first and the one below the last repeat them.
fn upsample(s: &[u8], c: &Comp, (w, h, hmax, vmax): (usize, usize, usize, usize)) -> Vec<u8> {
    let (fx, fy) = (hmax / c.h, vmax / c.v);
    let stride = c.bw * 8;
    let (cw, ch) = ((w * c.h).div_ceil(hmax), (h * c.v).div_ceil(vmax));
    let at = |x: usize, y: usize| s[y * stride + x] as i32;
    // (2:1 across: an output's sum of 3 x this column's and 1 x its neighbour's (the column itself at the edges), plus `lo` for a left output and `hi` for a right one, shifted)
    let across = |col: &dyn Fn(usize) -> i32, ox: usize, (lo, hi, shift): (i32, i32, u32)| -> i32 {
        let j = ox / 2;
        if ox % 2 == 1 {
            (col(j) * 3 + col(if j + 1 < cw { j + 1 } else { j }) + hi) >> shift
        } else {
            (col(j) * 3 + col(j.saturating_sub(1)) + lo) >> shift
        }
    };
    let mut out = vec![0u8; w * h];
    for oy in 0..h {
        let iy = oy / fy;
        // (the next nearest row: the one above for an upper output row, below for a lower one)
        let far = if oy % 2 == 1 { (iy + 1).min(ch - 1) } else { iy.saturating_sub(1) };
        for ox in 0..w {
            let v = match (fx, fy) {
                (2, 1) if cw > 2 => across(&|x| at(x, iy), ox, (1, 2, 2)),
                (1, 2) => (at(ox, iy) * 3 + at(ox, far) + if oy % 2 == 1 { 2 } else { 1 }) >> 2,
                (2, 2) if cw > 2 => across(&|x| at(x, iy) * 3 + at(x, far), ox, (8, 7, 4)),
                _ => at(ox / fx, iy),
            };
            out[oy * w + ox] = v as u8;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A PNG of `lines` (each line's bytes, filtered here with each filter in turn; an interlaced one's the passes' lines in order) at `depth` and `color`, a 4-color palette beside.
    fn png_of(w: usize, h: usize, depth: u8, color: u8, interlace: u8, lines: &[Vec<u8>], bpp: usize) -> Vec<u8> {
        let mut raw = Vec::new();
        let mut prev: Vec<u8> = Vec::new();
        for (i, l) in lines.iter().enumerate() {
            let f = (i % 5) as u8;
            if prev.len() != l.len() {
                prev = vec![0; l.len()];
            }
            raw.push(f);
            for k in 0..l.len() {
                let (a, b, c) = (if k >= bpp { l[k - bpp] as i16 } else { 0 }, prev[k] as i16, if k >= bpp { prev[k - bpp] as i16 } else { 0 });
                let p = match f {
                    0 => 0,
                    1 => a,
                    2 => b,
                    3 => (a + b) / 2,
                    _ => {
                        let q = a + b - c;
                        let (pa, pb, pc) = ((q - a).abs(), (q - b).abs(), (q - c).abs());
                        if pa <= pb && pa <= pc {
                            a
                        } else if pb <= pc {
                            b
                        } else {
                            c
                        }
                    }
                };
                raw.push(l[k].wrapping_sub(p as u8));
            }
            prev = l.clone();
        }
        let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        z.write_all(&raw).unwrap();
        let z = z.finish().unwrap();
        let chunk = |t: &[u8], d: &[u8]| [&(d.len() as u32).to_be_bytes()[..], t, d, &[0; 4]].concat();
        let mut head = Vec::new();
        head.extend((w as u32).to_be_bytes());
        head.extend((h as u32).to_be_bytes());
        head.extend([depth, color, 0, 0, interlace]);
        [
            &b"\x89PNG\r\n\x1a\n"[..],
            &chunk(b"IHDR", &head),
            &chunk(b"PLTE", &[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]),
            &chunk(b"IDAT", &z),
            &chunk(b"IEND", &[]),
        ]
        .concat()
    }

    #[test]
    fn a_jpeg_with_two_frame_headers_is_refused() {
        // SOI, then the same SOF0 (16 x 16, three components) twice: the second must not add components or buffers
        let sof = [0xFF, 0xC0, 0x00, 0x11, 0x08, 0x00, 0x10, 0x00, 0x10, 0x03, 0x01, 0x11, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01];
        let b = [&[0xFF, 0xD8][..], &sof, &sof].concat();
        assert!(rgb(&b).is_err());
        let huge = [0xFF, 0xC0, 0x00, 0x11, 0x08, 0xFF, 0xFF, 0xFF, 0xFF, 0x03, 0x01, 0x11, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01];
        assert!(rgb(&[&[0xFF, 0xD8][..], &huge].concat()).unwrap_err().contains("at most"));
    }

    #[test]
    fn pngs_of_each_kind() {
        let (w, h) = (13usize, 7usize);
        let px = |x: usize, y: usize| [(x * 19) as u8, (y * 37) as u8, (x * y * 5) as u8];
        let want: Vec<u8> = (0..h).flat_map(|y| (0..w).flat_map(move |x| px(x, y))).collect();
        // RGB 8-bit, every filter
        let lines: Vec<Vec<u8>> = (0..h).map(|y| (0..w).flat_map(|x| px(x, y)).collect()).collect();
        assert_eq!(rgb(&png_of(w, h, 8, 2, 0, &lines, 3)).unwrap(), (w, h, want.clone()));
        // RGBA 16-bit: the high bytes, alpha dropped
        let lines: Vec<Vec<u8>> = (0..h)
            .map(|y| {
                (0..w)
                    .flat_map(|x| {
                        let p = px(x, y);
                        [p[0], 1, p[1], 2, p[2], 3, 9, 9]
                    })
                    .collect()
            })
            .collect();
        assert_eq!(rgb(&png_of(w, h, 16, 6, 0, &lines, 8)).unwrap().2, want);
        // Adam7 interlaced RGB
        let mut lines = Vec::new();
        for (x0, y0, dx, dy) in [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)] {
            if x0 < w {
                lines.extend((y0..h).step_by(dy).map(|y| (x0..w).step_by(dx).flat_map(|x| px(x, y)).collect::<Vec<u8>>()));
            }
        }
        assert_eq!(rgb(&png_of(w, h, 8, 2, 1, &lines, 3)).unwrap().2, want);
        // 2-bit gray (0, 85, 170, 255) and a 4-bit palette
        let g = |x: usize, y: usize| ((x + y) % 4) as u8;
        let lines: Vec<Vec<u8>> =
            (0..h).map(|y| (0..w.div_ceil(4)).map(|b| (0..4).fold(0u8, |a, i| a << 2 | if 4 * b + i < w { g(4 * b + i, y) } else { 0 })).collect()).collect();
        assert_eq!(rgb(&png_of(w, h, 2, 0, 0, &lines, 1)).unwrap().2, (0..h).flat_map(|y| (0..w).flat_map(move |x| [g(x, y) * 85; 3])).collect::<Vec<u8>>());
        let lines: Vec<Vec<u8>> =
            (0..h).map(|y| (0..w.div_ceil(2)).map(|b| (0..2).fold(0u8, |a, i| a << 4 | if 2 * b + i < w { g(2 * b + i, y) } else { 0 })).collect()).collect();
        assert_eq!(
            rgb(&png_of(w, h, 4, 3, 0, &lines, 1)).unwrap().2,
            (0..h)
                .flat_map(|y| (0..w).flat_map(move |x| {
                    let k = g(x, y) * 30;
                    [10 + k, 20 + k, 30 + k]
                }))
                .collect::<Vec<u8>>()
        );
        assert!(rgb(b"\x89PNG\r\n\x1a\n").is_err());
    }

    /// JPEGs PIL made (baseline 4:2:2, baseline 4:2:0 with restarts every 3 blocks, progressive 4:2:0): the pixels PIL decodes from them (libjpeg-turbo), by their SHA-256; a file cut
    /// in half fails or reads what it has, never panics.
    #[test]
    fn jpegs_as_pil_decodes_them() {
        use sha2::{Digest, Sha256};
        for (f, w, h, sha) in [
            ("baseline_422.jpg", 17, 9, "5a53ba32acccd0c3cffa00ca83e76f008f1b170d5bf3adb13e270e52054a1338"),
            ("restarts_420.jpg", 17, 9, "01b0bf325462ab172ef9cce25925968be1cdc8a37bdd5ddadfc9e0934026f90c"),
            ("progressive_420.jpg", 64, 48, "44657c0a401ce6341040d2f51f7c64b444e7edae0612f24ecbd377a6e5f03e61"),
        ] {
            let bytes = std::fs::read(format!("{}/tests/fixtures/images/{f}", env!("CARGO_MANIFEST_DIR"))).unwrap();
            let (gw, gh, px) = rgb(&bytes).unwrap();
            let got: String = Sha256::digest(&px).iter().map(|x| format!("{x:02x}")).collect();
            assert_eq!((gw, gh, got.as_str()), (w, h, sha), "{f}");
            let _ = rgb(&bytes[..bytes.len() / 2]);
        }
    }
}
