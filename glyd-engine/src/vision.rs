//! Qwen3.8-Flash-Next's images: the processor's preprocessing (transformers' Qwen2VLImageProcessor as the checkpoint's preprocessor_config.json sets it) and the vision tower
//! (Qwen4ExpVisionModel) on the GPU, our own implementation of their math. An image (PNG or JPEG, from a data: URL or http(s)) is decoded to RGB (src/pixels.rs: PIL's pixels),
//! resized so that each side is a multiple of 32 and its pixels within the bounds (bicubic, antialiased, a pass a side in 8-bit steps, as torchvision resizes uint8 images), normalized
//! ((x - 127.5) / 127.5) and cut into 16x16 patches, 2x2 of them a block in order (each patch twice over: the checkpoint's patches are two frames deep). The tower: the patches' product
//! (the 3D convolution) and bias, the learned 48x48 position table resampled (bilinear, corners aligned) to the image's grid; 27 blocks of LayerNorm, attention (16 heads of 72, a 2D
//! rotary of the patch's row and column, over every patch of the image) and a GELU MLP; the merger (LayerNorm, a block's 4 patches as one row, two products with a GELU between) into one
//! row of 2560 a block: the image's tokens in the text (kernels/vision.cu beside the engine's dense products). In the text the image's tokens take M-RoPE positions (`positions`).

use crate::dev::{copy_h2d_async, memset_async, Buf, Dev, Kernel};
use crate::ffi::{err, Error, Result};
use crate::kern::Kern;
use crate::launch;
use crate::mapped::Mapping;
use glyd_gpu::json::{self, Value};
use glyd_gpu::safetensors::Checkpoint;
use std::collections::HashMap;
use std::net::{IpAddr, ToSocketAddrs};
use std::path::Path;
use std::sync::{Arc, OnceLock};

macro_rules! kname {
    ($n:literal) => {
        $n
    };
}

/// The shapes this tower is written for (Qwen3.8-Flash-Next's vision_config): kernels/vision.cu has them as constants.
const HIDDEN: usize = 1152;
const HEADS: usize = 16;
const HEAD_DIM: usize = 72;
const INTER: usize = 4304;
/// The MLP's neurons padded with zero rows (and the second product's columns) to the products' step of 128.
const INTER_PAD: usize = 4352;
const PATCH: usize = 16;
const MERGE: usize = 2;
const FRAMES: usize = 2;
const PATCH_DIM: usize = 3 * FRAMES * PATCH * PATCH;
/// A side of an image, a multiple of this: a block of 2x2 patches.
const FACTOR: usize = PATCH * MERGE;
/// Bytes an image's file may be (its pixels: `pixels::MAX_PIXELS`).
const MAX_BYTES: usize = 64 << 20;

/// How images are read (preprocessor_config.json's bounds, `--image-pixels` lowering the top) and the text's image token; set by a server whose checkpoint has a vision tower.
pub static PREP: OnceLock<Prep> = OnceLock::new();

#[derive(Clone, Debug)]
pub struct Prep {
    pub min_pixels: usize,
    pub max_pixels: usize,
    /// The token an image's places in the text are (`<|image_pad|>`), one a row of the tower's output.
    pub image_token: u32,
}

/// An image as the tower reads it: its grid of patches (rows, columns: even), the patches (bf16 [rows cols, 3 x 2 x 16 x 16], in 2x2 blocks), and a key of the bytes it came from.
pub struct Image {
    pub gh: usize,
    pub gw: usize,
    pub patches: Vec<u16>,
    pub key: u64,
}

impl Image {
    /// Its tokens in the text: a 2x2 block of patches each.
    pub fn tokens(&self) -> usize {
        self.gh * self.gw / (MERGE * MERGE)
    }

    /// Its tokens' grid (rows, columns).
    pub fn grid(&self) -> (usize, usize) {
        (self.gh / MERGE, self.gw / MERGE)
    }
}

impl Prep {
    /// The checkpoint's image bounds and token where it has a vision tower this engine runs (config.json's vision_config, the tower's tensors, preprocessor_config.json); None where it has
    /// none. `max_pixels` lowers the top bound (the processor's 16.8 M pixels are 16,384 tokens an image).
    pub fn load(dir: &Path, max_pixels: Option<usize>) -> Result<Option<Prep>> {
        let Some(v) = vision_config(dir)? else { return Ok(None) };
        let p = dir.join("preprocessor_config.json");
        let pc = match std::fs::read_to_string(&p) {
            Ok(t) => json::parse(&t).map_err(|e| Error(format!("{}: {e}", p.display())))?,
            Err(_) => return err(format!("{}: missing, where the checkpoint has a vision tower", p.display())),
        };
        let num = |k: &str| pc.get(k).and_then(Value::as_u64).map(|x| x as usize);
        let size = |k: &str| pc.get("size").and_then(|s| s.get(k)).and_then(Value::as_u64).map(|x| x as usize);
        if num("patch_size") != Some(PATCH) || num("merge_size") != Some(MERGE) || num("temporal_patch_size").is_some_and(|t| t != FRAMES) {
            return err(format!("{}: patches other than 16 x 16 in blocks of 2 x 2: not supported", p.display()));
        }
        let halves = |k: &str| {
            pc.get(k)
                .and_then(Value::as_array)
                .is_some_and(|a| a.len() == 3 && a.iter().all(|x| matches!(x, Value::Number(n) if n.parse::<f64>().ok() == Some(0.5))))
        };
        if !halves("image_mean") || !halves("image_std") {
            return err(format!("{}: a mean or deviation other than 0.5: not supported", p.display()));
        }
        let (lo, hi) = (size("shortest_edge").unwrap_or(65536), size("longest_edge").unwrap_or(16_777_216));
        let hi = max_pixels.map_or(hi, |m| m.min(hi)).max(lo);
        Ok(Some(Prep { min_pixels: lo, max_pixels: hi, image_token: v.image_token }))
    }

    /// An image's bytes as the tower reads them: decoded, resized, normalized, cut into patches.
    pub fn image(&self, bytes: &[u8]) -> Result<Image> {
        let (w, h, rgb) = crate::pixels::rgb(bytes).map_err(Error)?;
        let (nh, nw) = smart_resize(h, w, self.min_pixels, self.max_pixels)?;
        let rgb = resize(&rgb, w, h, nw, nh);
        let (gh, gw) = (nh / PATCH, nw / PATCH);
        let mut patches = Vec::with_capacity(gh * gw * PATCH_DIM);
        let px = |c: usize, y: usize, x: usize| bf16((rgb[(y * nw + x) * 3 + c] as f32 - 127.5) / 127.5);
        for br in 0..gh / MERGE {
            for bc in 0..gw / MERGE {
                for ir in 0..MERGE {
                    for ic in 0..MERGE {
                        let (r, c) = ((br * MERGE + ir) * PATCH, (bc * MERGE + ic) * PATCH);
                        for ch in 0..3 {
                            for _ in 0..FRAMES {
                                for y in 0..PATCH {
                                    patches.extend((0..PATCH).map(|x| px(ch, r + y, c + x)));
                                }
                            }
                        }
                    }
                }
            }
        }
        use std::hash::{Hash, Hasher};
        let mut k = std::collections::hash_map::DefaultHasher::new();
        (bytes, self.min_pixels, self.max_pixels).hash(&mut k);
        Ok(Image { gh, gw, patches, key: k.finish() })
    }
}

/// What the engine reads of config.json's vision part: the image token, the tower's depth (its other shapes checked against the constants above). None: the checkpoint has no tower.
struct VisionCfg {
    image_token: u32,
    depth: usize,
}

fn vision_config(dir: &Path) -> Result<Option<VisionCfg>> {
    let p = dir.join("config.json");
    let t = std::fs::read_to_string(&p).map_err(|e| Error(format!("{}: {e}", p.display())))?;
    let c = json::parse(&t).map_err(|e| Error(format!("{}: {e}", p.display())))?;
    let Some(v) = c.get("vision_config").filter(|v| matches!(v, Value::Object(_))) else { return Ok(None) };
    if c.get("language_model_only").and_then(Value::as_bool) == Some(true) {
        return Ok(None);
    }
    let u = |k: &str| v.get(k).and_then(Value::as_u64).map(|x| x as usize);
    let want = [
        ("hidden_size", HIDDEN),
        ("num_heads", HEADS),
        ("intermediate_size", INTER),
        ("out_hidden_size", 2560),
        ("patch_size", PATCH),
        ("spatial_merge_size", MERGE),
        ("num_position_embeddings", 2304),
    ];
    for (k, x) in want {
        if u(k) != Some(x) {
            return err(format!("{}: vision_config.{k} {:?}, where this engine's tower has {x}", p.display(), u(k)));
        }
    }
    if v.get("hidden_act").and_then(Value::as_str) != Some("gelu_pytorch_tanh")
        || v.get("deepstack_visual_indexes").and_then(Value::as_array).is_some_and(|a| !a.is_empty())
    {
        return err(format!("{}: a vision tower with another activation or deepstack layers: not supported", p.display()));
    }
    let image_token = c.get("image_token_id").and_then(Value::as_u64).ok_or_else(|| Error(format!("{}: no image_token_id", p.display())))? as u32;
    Ok(Some(VisionCfg { image_token, depth: u("depth").unwrap_or(27) }))
}

/// f32 to bf16, round to nearest even.
fn bf16(x: f32) -> u16 {
    let b = x.to_bits();
    ((b.wrapping_add(0x7fff + ((b >> 16) & 1))) >> 16) as u16
}

// ---- the bytes of an image ------------------------------------------------------------------------------------------------------------------------------------------------------------

/// The bytes an image URL names: a data: URL's (base64), or an http(s) URL's body (30 s, 64 MB at most, up to 5 redirects). Other schemes (a file on
/// this computer) are refused, and so is any host that is not a public address (this computer, the LAN, link-local, a cloud's metadata service): any
/// client of the API names the URL, and the server must not fetch on its behalf what the client could not reach itself.
pub fn fetch(url: &str) -> Result<Vec<u8>> {
    let url = url.trim();
    if let Some(rest) = url.strip_prefix("data:") {
        let (meta, data) = rest.split_once(',').ok_or_else(|| Error("a data URL with no data".into()))?;
        if !meta.ends_with(";base64") {
            return err("a data URL that is not base64");
        }
        return base64(data);
    }
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        let shown: String = url.chars().take(40).collect();
        return err(format!("an image URL that is neither data: nor http(s): {shown}"));
    }
    // (redirects followed here, one at a time, so that every hop's host is checked)
    // (the addresses a name resolves to are checked where the connection is made, not in an earlier lookup: a name whose DNS answers a public address to the check and a private one to the connection (DNS rebinding) is not let through)
    let agent = ureq::Agent::with_parts(
        ureq::Agent::config_builder()
            .timeout_global(Some(std::time::Duration::from_secs(30)))
            .max_redirects(0)
            .user_agent(concat!("glyd-engine/", env!("CARGO_PKG_VERSION")))
            .build(),
        ureq::unversioned::transport::DefaultConnector::new(),
        PublicOnly(ureq::unversioned::resolver::DefaultResolver::default()),
    );
    let mut at = url.to_string();
    for _ in 0..6 {
        public_host(&at)?;
        let mut r = agent.get(&at).call().map_err(|e| Error(format!("the image at {at}: {e}")))?;
        if matches!(r.status().as_u16(), 301 | 302 | 303 | 307 | 308) {
            let to = r.headers().get("location").and_then(|v| v.to_str().ok()).unwrap_or("");
            if !(to.starts_with("http://") || to.starts_with("https://")) {
                return err(format!("the image at {at}: a redirect to {}", to.chars().take(40).collect::<String>()));
            }
            at = to.to_string();
            continue;
        }
        return r.body_mut().with_config().limit(MAX_BYTES as u64).read_to_vec().map_err(|e| Error(format!("the image at {at}: {e}")));
    }
    err(format!("the image at {url}: more than 5 redirects"))
}

/// A resolver that answers with the addresses of `R` only where all of them are public (`public_ip`): the connection goes to an address that was checked, since the connection takes its address from here.
#[derive(Debug)]
struct PublicOnly<R>(R);

impl<R: ureq::unversioned::resolver::Resolver> ureq::unversioned::resolver::Resolver for PublicOnly<R> {
    fn resolve(
        &self,
        uri: &ureq::http::Uri,
        config: &ureq::config::Config,
        timeout: ureq::unversioned::transport::NextTimeout,
    ) -> std::result::Result<ureq::unversioned::resolver::ResolvedSocketAddrs, ureq::Error> {
        let addrs = self.0.resolve(uri, config, timeout)?;
        match addrs.iter().find(|a| !public_ip(a.ip())) {
            Some(a) => Err(ureq::Error::Io(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                format!("{} resolves to {}, an address on this computer or its network (only public addresses are fetched)", uri.host().unwrap_or(""), a.ip()),
            ))),
            None => Ok(addrs),
        }
    }
}

/// Refuses an http(s) URL unless every address its host resolves to is public (`public_ip`).
fn public_host(url: &str) -> Result<()> {
    let https = url.starts_with("https://");
    let rest = url.split_once("://").map_or(url, |(_, r)| r);
    let hostport = rest.split(['/', '?', '#']).next().unwrap_or("");
    let hostport = hostport.rsplit_once('@').map_or(hostport, |(_, h)| h);
    let default = if https { 443 } else { 80 };
    let (host, port) = if let Some(h) = hostport.strip_prefix('[') {
        let (h, p) = h.split_once(']').ok_or_else(|| Error(format!("a bad host in {url}")))?;
        (h, p.strip_prefix(':').map_or(Ok(default), |p| p.parse::<u16>()).map_err(|_| Error(format!("a bad port in {url}")))?)
    } else if let Some((h, p)) = hostport.rsplit_once(':') {
        (h, p.parse::<u16>().map_err(|_| Error(format!("a bad port in {url}")))?)
    } else {
        (hostport, default)
    };
    if host.is_empty() {
        return err(format!("no host in {url}"));
    }
    let addrs: Vec<IpAddr> = (host, port).to_socket_addrs().map_err(|e| Error(format!("the host of {url}: {e}")))?.map(|a| a.ip()).collect();
    if addrs.is_empty() || addrs.iter().any(|&ip| !public_ip(ip)) {
        return err(format!("the image at {url}: a host on this computer or its network (only public addresses are fetched)"));
    }
    Ok(())
}

/// A public unicast address: not loopback, private, link-local, carrier-grade NAT, multicast, broadcast, reserved, documentation or unspecified.
pub fn public_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v) => {
            let o = v.octets();
            !(v.is_unspecified()
                || v.is_loopback()
                || v.is_private()
                || v.is_link_local()
                || v.is_broadcast()
                || v.is_multicast()
                || v.is_documentation()
                || (o[0] == 100 && (o[1] & 0xC0) == 64)
                || o[0] >= 240
                || (o[0] == 192 && o[1] == 0 && o[2] == 0)
                || (o[0] == 198 && (o[1] & 0xFE) == 18))
        }
        IpAddr::V6(v) => {
            if let Some(m) = v.to_ipv4_mapped() {
                return public_ip(IpAddr::V4(m));
            }
            let s = v.segments();
            !(v.is_unspecified()
                || v.is_loopback()
                || v.is_multicast()
                || (s[0] & 0xFE00) == 0xFC00
                || (s[0] & 0xFFC0) == 0xFE80
                || (s[0] == 0x2001 && s[1] == 0x0DB8)
                || v.to_ipv4().is_some())
        }
    }
}

/// Standard base64 (padding and white space allowed).
fn base64(s: &str) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    let (mut acc, mut bits) = (0u32, 0u32);
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' | b' ' | b'\n' | b'\r' | b'\t' => continue,
            _ => return err("an image's base64 with a character outside it"),
        };
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
        if out.len() > MAX_BYTES {
            return err(format!("an image of more than {} MB", MAX_BYTES >> 20));
        }
    }
    Ok(out)
}

/// The processor's smart_resize: each side rounded (half to even) to a multiple of 32, then scaled down or up, keeping the aspect, where the pixels are past the bounds.
fn smart_resize(h: usize, w: usize, min_pixels: usize, max_pixels: usize) -> Result<(usize, usize)> {
    if h.max(w) as f64 / h.min(w) as f64 > 200.0 {
        return err(format!("an image of {w} x {h}: its sides may differ 200 times at most"));
    }
    let f = FACTOR as f64;
    let (hf, wf) = (h as f64, w as f64);
    let (mut hb, mut wb) = ((hf / f).round_ties_even() * f, (wf / f).round_ties_even() * f);
    if hb * wb > max_pixels as f64 {
        let beta = (hf * wf / max_pixels as f64).sqrt();
        hb = f.max((hf / beta / f).floor() * f);
        wb = f.max((wf / beta / f).floor() * f);
    } else if hb * wb < min_pixels as f64 {
        let beta = (min_pixels as f64 / (hf * wf)).sqrt();
        hb = (hf * beta / f).ceil() * f;
        wb = (wf * beta / f).ceil() * f;
    }
    Ok((hb as usize, wb as usize))
}

/// The taps of each output pixel of an axis `inp` long resized to `out` (bicubic, a = -0.5, widened by the scale where it shrinks, as torchvision's antialiased resize has them): the first
/// input pixel and the weights, in int16 at the precision (bits) the largest weight allows.
fn taps(inp: usize, out: usize) -> (Vec<(usize, Vec<i16>)>, u32) {
    let cubic = |x: f64| -> f64 {
        let (a, x) = (-0.5, x.abs());
        if x < 1.0 {
            ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
        } else if x < 2.0 {
            ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a
        } else {
            0.0
        }
    };
    let scale = inp as f64 / out as f64;
    let support = if scale >= 1.0 { 2.0 * scale } else { 2.0 };
    let most = (support as f32).ceil() as i64 * 2 + 1;
    let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    let mut rows = Vec::with_capacity(out);
    let mut top = 0.0f64;
    for i in 0..out {
        let center = scale * (i as f64 + 0.5);
        let xmin = ((center - support + 0.5) as i64).max(0);
        let xsize = (((center + support + 0.5) as i64).min(inp as i64) - xmin).clamp(0, most);
        let mut w: Vec<f64> = (0..xsize).map(|j| cubic((j as f64 + xmin as f64 - center + 0.5) * invscale)).collect();
        let total: f64 = w.iter().sum();
        if total != 0.0 {
            w.iter_mut().for_each(|x| *x /= total);
        }
        top = w.iter().cloned().fold(top, f64::max);
        rows.push((xmin as usize, w));
    }
    let mut prec = 0u32;
    while prec < 22 && ((0.5 + top * (1u64 << (prec + 1)) as f64) as i64) < (1 << 15) {
        prec += 1;
    }
    let q = rows
        .into_iter()
        .map(|(x0, w)| {
            (
                x0,
                w.iter()
                    .map(|&v| {
                        let v = v * (1u64 << prec) as f64;
                        (if v < 0.0 { -0.5 + v } else { 0.5 + v }) as i16
                    })
                    .collect(),
            )
        })
        .collect();
    (q, prec)
}

/// RGB8 rows (w x h) resized to nw x nh: the width first, then the height, each a pass of `taps` rounded to 8 bits (a side that keeps its length is not resampled), as torchvision resizes
/// a uint8 image (bicubic, antialiased: the processor's).
fn resize(rgb: &[u8], w: usize, h: usize, nw: usize, nh: usize) -> Vec<u8> {
    // (a pass along an axis of `inp` pixels to `out`: `lines` lines, (the lines' stride, the pixels' step) in bytes in and out)
    let pass = |src: &[u8], inp: usize, out: usize, lines: usize, (si, ti): (usize, usize), (so, to): (usize, usize)| -> Vec<u8> {
        let (taps, prec) = taps(inp, out);
        let mut dst = vec![0u8; lines * out * 3];
        for line in 0..lines {
            for (o, (x0, wt)) in taps.iter().enumerate() {
                for c in 0..3 {
                    let acc = wt.iter().enumerate().fold(1i64 << (prec - 1), |a, (j, &k)| a + src[line * si + (x0 + j) * ti + c] as i64 * k as i64);
                    dst[line * so + o * to + c] = (acc >> prec).clamp(0, 255) as u8;
                }
            }
        }
        dst
    };
    let mut cur = rgb.to_vec();
    if nw != w {
        cur = pass(&cur, w, nw, h, (w * 3, 3), (nw * 3, 3));
    }
    if nh != h {
        cur = pass(&cur, h, nh, nw, (3, nw * 3), (3, nw * 3));
    }
    cur
}

// ---- positions in the text ------------------------------------------------------------------------------------------------------------------------------------------------------------

/// `ids` with each image's place (the one `image` token a chat template writes for it, in order) made as long as the image's tokens; the places and the images as many.
pub fn expand(ids: &[u32], image: u32, imgs: &[Arc<Image>]) -> Result<Vec<u32>> {
    let places = ids.iter().filter(|&&t| t == image).count();
    if places != imgs.len() {
        return err(format!("{} images and {places} places for them in the prompt (an <|image_pad|> in its text?)", imgs.len()));
    }
    let mut out = Vec::with_capacity(ids.len() + imgs.iter().map(|i| i.tokens()).sum::<usize>());
    let mut k = 0;
    for &t in ids {
        if t == image {
            out.extend(std::iter::repeat_n(image, imgs[k].tokens()));
            k += 1;
        } else {
            out.push(t);
        }
    }
    Ok(out)
}

/// The M-RoPE position (t, h, w) of each row of `ids` and the position after the last: a run of text from position p takes (p, p, p), (p + 1, ..) and so on; a run of `image` tokens (an image
/// of `grids`, in order: its tokens' rows and columns) from p takes (p, p + row, p + col) in the image's row order, and the text after it starts at p + the larger of rows and columns (the
/// reference's get_rope_index). Text alone: every row (p, p, p).
pub fn positions(ids: &[u32], image: u32, grids: &[(usize, usize)]) -> Result<(Vec<[u32; 3]>, u32)> {
    let mut out = Vec::with_capacity(ids.len());
    let (mut p, mut i, mut g) = (0u32, 0usize, 0usize);
    while i < ids.len() {
        if ids[i] != image {
            out.push([p; 3]);
            p += 1;
            i += 1;
            continue;
        }
        let run = ids[i..].iter().take_while(|&&t| t == image).count();
        let &(rows, cols) = grids.get(g).ok_or_else(|| Error(format!("the prompt has more image places than its {} images", grids.len())))?;
        if run != rows * cols {
            return err(format!("image {}'s place in the prompt is {run} tokens, where it has {}", g + 1, rows * cols));
        }
        for r in 0..rows {
            for c in 0..cols {
                out.push([p, p + r as u32, p + c as u32]);
            }
        }
        p += rows.max(cols) as u32;
        i += run;
        g += 1;
    }
    if g != grids.len() {
        return err(format!("{} images and {g} places for them in the prompt", grids.len()));
    }
    Ok((out, p))
}

/// The rotary's cos and sin (bf16, `half` of each a row) of M-RoPE positions: frequency i (theta^(-2i / (2 half))) turns by the position's t, h or w as i mod 3 is 0, 1 or 2 (the checkpoint's
/// interleaved sections of 11, 11 and 10); a row (p, p, p) is the plain rotary's row p, bit for bit (`weights::rope_tables`).
pub fn rope_rows(theta: f64, half: usize, rows: &[[u32; 3]]) -> (Vec<u16>, Vec<u16>) {
    let inv: Vec<f32> = (0..half).map(|i| 1.0f32 / (theta as f32).powf((2 * i) as f32 / (2 * half) as f32)).collect();
    let (mut cos, mut sin) = (Vec::with_capacity(rows.len() * half), Vec::with_capacity(rows.len() * half));
    for r in rows {
        for (i, &f) in inv.iter().enumerate() {
            let a = r[i % 3] as f32 * f;
            cos.push(bf16(a.cos()));
            sin.push(bf16(a.sin()));
        }
    }
    (cos, sin)
}

// ---- the tower ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

struct KernVis {
    ln: Kernel,
    bias: Kernel,
    embed: Kernel,
    rope: Kernel,
    attn: Kernel,
    pad: Kernel,
}

/// vis_attn's dynamic shared memory: 320 rows of 88 bf16.
const ATTN_SMEM: u32 = 320 * 88 * 2;

/// The tower: its tensors read from the checkpoint's mapped files for each image (no GPU memory between images), its kernels.
pub struct Tower {
    src: HashMap<String, (Arc<Mapping>, usize, usize)>,
    depth: usize,
    k: KernVis,
}

/// Bytes of the weights' buffer: the largest of a block's matrices (the second MLP product's columns padded, its rows as read beside), the merger's, and the patch product's with the
/// position table.
fn weight_elems() -> (usize, usize, usize) {
    let block = 4 * HIDDEN + 3 * HIDDEN * HIDDEN + 3 * HIDDEN + HIDDEN * HIDDEN + HIDDEN + 2 * INTER_PAD * HIDDEN + INTER_PAD + HIDDEN + HIDDEN * INTER;
    let merger = 2 * HIDDEN + 16 * HIDDEN * HIDDEN + 4 * HIDDEN + 4 * HIDDEN * 2560 + 2560;
    let embed = HIDDEN * PATCH_DIM + HIDDEN + 2304 * HIDDEN;
    (block, merger, embed)
}

/// The weights' buffer: the largest set, and room for each tensor's start at a multiple of 256 bytes.
fn weight_bytes() -> usize {
    let (b, m, e) = weight_elems();
    2 * b.max(m).max(e) + (64 << 10)
}

impl Tower {
    /// The tower of the checkpoint at `dir` (None where it has none), its tensors' files mapped, checked for every tensor and shape it reads.
    pub fn open(dir: &Path, dev: &Dev) -> Result<Option<Tower>> {
        let Some(cfg) = vision_config(dir)? else { return Ok(None) };
        let ck = Checkpoint::open(dir).map_err(|e| Error(e.to_string()))?;
        let mut maps: HashMap<std::path::PathBuf, Arc<Mapping>> = HashMap::new();
        let mut src = HashMap::new();
        let mut want: Vec<(String, Vec<u64>)> = vec![
            ("patch_embed.proj.weight".into(), vec![HIDDEN as u64, 3, FRAMES as u64, PATCH as u64, PATCH as u64]),
            ("patch_embed.proj.bias".into(), vec![HIDDEN as u64]),
            ("pos_embed.weight".into(), vec![2304, HIDDEN as u64]),
            ("merger.norm.weight".into(), vec![HIDDEN as u64]),
            ("merger.norm.bias".into(), vec![HIDDEN as u64]),
            ("merger.linear_fc1.weight".into(), vec![4608, 4608]),
            ("merger.linear_fc1.bias".into(), vec![4608]),
            ("merger.linear_fc2.weight".into(), vec![2560, 4608]),
            ("merger.linear_fc2.bias".into(), vec![2560]),
        ];
        let (h, i) = (HIDDEN as u64, INTER as u64);
        for l in 0..cfg.depth {
            for (n, s) in [
                ("norm1.weight", vec![h]),
                ("norm1.bias", vec![h]),
                ("norm2.weight", vec![h]),
                ("norm2.bias", vec![h]),
                ("attn.qkv.weight", vec![3 * h, h]),
                ("attn.qkv.bias", vec![3 * h]),
                ("attn.proj.weight", vec![h, h]),
                ("attn.proj.bias", vec![h]),
                ("mlp.linear_fc1.weight", vec![i, h]),
                ("mlp.linear_fc1.bias", vec![i]),
                ("mlp.linear_fc2.weight", vec![h, i]),
                ("mlp.linear_fc2.bias", vec![h]),
            ] {
                want.push((format!("blocks.{l}.{n}"), s));
            }
        }
        for (n, shape) in want {
            let name = format!("model.visual.{n}");
            let (f, t) = ck.get(&name).ok_or_else(|| Error(format!("{}: no tensor {name} (the vision tower's)", dir.display())))?;
            if t.dtype != "BF16" || t.shape != shape {
                return err(format!("{name}: {} {:?}, where BF16 {shape:?} is wanted", t.dtype, t.shape));
            }
            let map = match maps.get(&f.path) {
                Some(m) => m.clone(),
                None => {
                    let m = Arc::new(Mapping::open(&f.path)?);
                    maps.insert(f.path.clone(), m.clone());
                    m
                }
            };
            if (t.offset + t.bytes) as usize > map.len {
                return err(format!("{name}: past the end of {}", f.path.display()));
            }
            src.insert(n, (map, t.offset as usize, t.bytes as usize));
        }
        let attn = dev.kernel(kname!("vis_attn"))?;
        attn.allow_smem(ATTN_SMEM)?;
        let k = KernVis {
            ln: dev.kernel(kname!("vis_ln"))?,
            bias: dev.kernel(kname!("vis_bias"))?,
            embed: dev.kernel(kname!("vis_embed"))?,
            rope: dev.kernel(kname!("vis_rope"))?,
            attn,
            pad: dev.kernel(kname!("vis_pad"))?,
        };
        Ok(Some(Tower { src, depth: cfg.depth, k }))
    }

    /// GPU memory an image of `patches` patches takes while the tower runs: the weights' buffer and the activations.
    pub fn bytes(patches: usize) -> usize {
        weight_bytes() + patches * (2 * (2 * HIDDEN + INTER_PAD + 3 * HEADS * 80) + 40) + (1 << 20)
    }

    /// Tensor `n`'s bytes copied to `dst` on the stream (from the mapped file: the copy has read them when it returns).
    unsafe fn put(&self, dev: &Dev, dst: u64, n: &str) -> Result<()> {
        let (m, off, bytes) = self.src.get(n).ok_or_else(|| Error(format!("no tensor {n}")))?;
        copy_h2d_async(dst, (m.ptr as *const u8).add(*off) as *const _, *bytes, dev.stream)
    }

    /// The image's rows for the text: bf16 [tokens, 2560], its 2x2 blocks in order.
    pub fn run(&self, dev: &Dev, kern: &Kern, img: &Image) -> Result<Vec<u16>> {
        let n = img.gh * img.gw;
        if img.patches.len() != n * PATCH_DIM || n == 0 || img.gh % MERGE != 0 || img.gw % MERGE != 0 {
            return err("an image whose patches are not its grid's");
        }
        let s = dev.stream;
        let be = weight_elems().0;
        let wb = Buf::new(weight_bytes())?;
        let (x, h) = (Buf::new(n * HIDDEN * 2)?, Buf::new(n * HIDDEN * 2)?);
        let big = Buf::new(n * INTER_PAD.max(PATCH_DIM) * 2)?;
        let qkv: Vec<Buf> = (0..3).map(|_| Buf::new(HEADS * n * 80 * 2)).collect::<Result<_>>()?;
        // the patches' grid: each patch's (row, column) for the rotary, its 4 taps into the 48 x 48 position table and their weights (the reference's, in fp32)
        let side = 48usize;
        let axis = |i: usize, size: usize| -> ([usize; 2], [f32; 2]) {
            let src = i as f32 * (side - 1) as f32 / (size - 1).max(1) as f32;
            let fl = src.floor();
            let t = |o: f32| ((fl + o) as i64).clamp(0, side as i64 - 1) as usize;
            let w = |o: f32| (1.0 - (src - fl - o).abs()).max(0.0);
            ([t(0.0), t(1.0)], [w(0.0), w(1.0)])
        };
        let (mut pos, mut idx, mut wt) = (Vec::with_capacity(2 * n), Vec::with_capacity(4 * n), Vec::with_capacity(4 * n));
        for br in 0..img.gh / MERGE {
            for bc in 0..img.gw / MERGE {
                for ir in 0..MERGE {
                    for ic in 0..MERGE {
                        let (r, c) = (br * MERGE + ir, bc * MERGE + ic);
                        pos.extend([r as i32, c as i32]);
                        let ((hi, hw), (wi, ww)) = (axis(r, img.gh), axis(c, img.gw));
                        for a in 0..2 {
                            for b in 0..2 {
                                idx.push((hi[a] * side + wi[b]) as i32);
                                wt.push(hw[a] * ww[b]);
                            }
                        }
                    }
                }
            }
        }
        // the rotary's cos and sin of each row or column position on each of its 18 frequencies (theta 10,000 over 36 dims), fp32 as the reference makes them
        let inv: Vec<f32> = (0..18).map(|j| 1.0f32 / 10000f32.powf((2 * j) as f32 / 36.0)).collect();
        let rot: Vec<f32> = (0..img.gh.max(img.gw))
            .flat_map(|p| {
                inv.iter().flat_map(move |&f| {
                    let a = p as f32 * f;
                    [a.cos(), a.sin()]
                })
            })
            .collect();
        let (pos_b, idx_b, wt_b, rot_b) = (Buf::new(pos.len() * 4)?, Buf::new(idx.len() * 4)?, Buf::new(wt.len() * 4)?, Buf::new(rot.len() * 4)?);
        let (ni, ci, hi, inter) = (n as i32, HIDDEN as i32, HIDDEN as i64, INTER_PAD as i64);
        let elems = |rows: usize, cols: usize| (rows * cols) as i64;
        let grid = |total: i64| [(total as u64).div_ceil(256).min(8192) as u32, 1, 1];
        // SAFETY: the buffers above, of the sizes the kernels and products read and write; the host's vectors are read by the copies before they return; synchronized before the read.
        unsafe {
            copy_h2d_async(pos_b.ptr, pos.as_ptr() as *const _, pos.len() * 4, s)?;
            copy_h2d_async(idx_b.ptr, idx.as_ptr() as *const _, idx.len() * 4, s)?;
            copy_h2d_async(wt_b.ptr, wt.as_ptr() as *const _, wt.len() * 4, s)?;
            copy_h2d_async(rot_b.ptr, rot.as_ptr() as *const _, rot.len() * 4, s)?;
            copy_h2d_async(big.ptr, img.patches.as_ptr() as *const _, img.patches.len() * 2, s)?;
            // the patches' product, its bias and the positions
            let (pw, pb, table) = (wb.ptr, wb.ptr + (HIDDEN * PATCH_DIM * 2) as u64, wb.ptr + ((HIDDEN * PATCH_DIM + HIDDEN) * 2) as u64);
            self.put(dev, pw, "patch_embed.proj.weight")?;
            self.put(dev, pb, "patch_embed.proj.bias")?;
            self.put(dev, table, "pos_embed.weight")?;
            kern.gemm_fill(s, big.ptr, pw, h.ptr, ni, ci, PATCH_DIM as i32)?;
            launch!(self.k.embed, [n as u32, 1, 1], [256, 1, 1], s; h.ptr, pb, table, idx_b.ptr, wt_b.ptr, x.ptr, ci)?;
            // the blocks: a block's weights at fixed places in the buffer (the MLP's padding rows and columns zero), copied in turn
            memset_async(wb.ptr, 0, 2 * be, s)?;
            let mut at = 0u64;
            let mut take = |elems: usize| {
                let p = wb.ptr + at;
                at += (elems * 2).next_multiple_of(256) as u64;
                p
            };
            let (n1w, n1b, n2w, n2b) = (take(HIDDEN), take(HIDDEN), take(HIDDEN), take(HIDDEN));
            let (qw, qb, ow, ob) = (take(3 * HIDDEN * HIDDEN), take(3 * HIDDEN), take(HIDDEN * HIDDEN), take(HIDDEN));
            let (f1w, f1b, f2w, f2b, f2raw) = (take(INTER_PAD * HIDDEN), take(INTER_PAD), take(HIDDEN * INTER_PAD), take(HIDDEN), take(HIDDEN * INTER));
            if at as usize > wb.bytes {
                return err("the vision weights' buffer is too small");
            }
            let sl2 = (1.0 / (HEAD_DIM as f32).sqrt()) * std::f32::consts::LOG2_E;
            for l in 0..self.depth {
                let p = |t: &str| format!("blocks.{l}.{t}");
                for (dst, t) in [
                    (n1w, "norm1.weight"),
                    (n1b, "norm1.bias"),
                    (n2w, "norm2.weight"),
                    (n2b, "norm2.bias"),
                    (qw, "attn.qkv.weight"),
                    (qb, "attn.qkv.bias"),
                    (ow, "attn.proj.weight"),
                    (ob, "attn.proj.bias"),
                    (f1w, "mlp.linear_fc1.weight"),
                    (f1b, "mlp.linear_fc1.bias"),
                    (f2raw, "mlp.linear_fc2.weight"),
                    (f2b, "mlp.linear_fc2.bias"),
                ] {
                    self.put(dev, dst, &p(t))?;
                }
                launch!(self.k.pad, grid(elems(HIDDEN, INTER_PAD)), [256, 1, 1], s; f2raw, f2w, INTER as i32, INTER_PAD as i32, elems(HIDDEN, INTER_PAD))?;
                // attention: x += proj(attn(rotary(qkv(ln1(x)))))
                launch!(self.k.ln, [n as u32, 1, 1], [256, 1, 1], s; x.ptr, n1w, n1b, h.ptr, ci, 1e-6f32)?;
                kern.gemm_fill(s, h.ptr, qw, big.ptr, ni, 3 * ci, ci)?;
                launch!(self.k.bias, grid(elems(n, 3 * HIDDEN)), [256, 1, 1], s; big.ptr, qb, 0u64, 3 * ci, 0i32, elems(n, 3 * HIDDEN))?;
                launch!(self.k.rope, [n as u32, 1, 1], [256, 1, 1], s; big.ptr, pos_b.ptr, rot_b.ptr, qkv[0].ptr, qkv[1].ptr, qkv[2].ptr, ni)?;
                launch!(self.k.attn, [(n as u32).div_ceil(64), HEADS as u32, 1], [128, 1, 1], smem ATTN_SMEM, s; qkv[0].ptr, qkv[1].ptr, qkv[2].ptr, h.ptr, ni, sl2)?;
                kern.gemm_fill(s, h.ptr, ow, big.ptr, ni, ci, ci)?;
                launch!(self.k.bias, grid(elems(n, HIDDEN)), [256, 1, 1], s; big.ptr, ob, x.ptr, ci, 0i32, n as i64 * hi)?;
                // the MLP: x += fc2(gelu(fc1(ln2(x))))
                launch!(self.k.ln, [n as u32, 1, 1], [256, 1, 1], s; x.ptr, n2w, n2b, h.ptr, ci, 1e-6f32)?;
                kern.gemm_fill(s, h.ptr, f1w, big.ptr, ni, INTER_PAD as i32, ci)?;
                launch!(self.k.bias, grid(elems(n, INTER_PAD)), [256, 1, 1], s; big.ptr, f1b, 0u64, INTER_PAD as i32, 1i32, n as i64 * inter)?;
                kern.gemm_fill(s, big.ptr, f2w, h.ptr, ni, ci, INTER_PAD as i32)?;
                launch!(self.k.bias, grid(elems(n, HIDDEN)), [256, 1, 1], s; h.ptr, f2b, x.ptr, ci, 0i32, n as i64 * hi)?;
            }
            // the merger: ln, a block's 4 patches as one row of 4608, fc1, GELU (erf), fc2
            let m = n / (MERGE * MERGE);
            let mut at = 0u64;
            let mut take = |elems: usize| {
                let p = wb.ptr + at;
                at += (elems * 2).next_multiple_of(256) as u64;
                p
            };
            let (nw_, nb_, w1, b1, w2, b2) = (take(HIDDEN), take(HIDDEN), take(4608 * 4608), take(4608), take(2560 * 4608), take(2560));
            if at as usize > wb.bytes {
                return err("the vision weights' buffer is too small");
            }
            for (dst, t) in [
                (nw_, "merger.norm.weight"),
                (nb_, "merger.norm.bias"),
                (w1, "merger.linear_fc1.weight"),
                (b1, "merger.linear_fc1.bias"),
                (w2, "merger.linear_fc2.weight"),
                (b2, "merger.linear_fc2.bias"),
            ] {
                self.put(dev, dst, t)?;
            }
            launch!(self.k.ln, [n as u32, 1, 1], [256, 1, 1], s; x.ptr, nw_, nb_, h.ptr, ci, 1e-6f32)?;
            kern.gemm_fill(s, h.ptr, w1, big.ptr, m as i32, 4608, 4608)?;
            launch!(self.k.bias, grid(elems(m, 4608)), [256, 1, 1], s; big.ptr, b1, 0u64, 4608i32, 2i32, elems(m, 4608))?;
            kern.gemm_fill(s, big.ptr, w2, x.ptr, m as i32, 2560, 4608)?;
            launch!(self.k.bias, grid(elems(m, 2560)), [256, 1, 1], s; x.ptr, b2, 0u64, 2560i32, 0i32, elems(m, 2560))?;
        }
        dev.sync()?;
        let mut out = vec![0u16; img.tokens() * 2560];
        x.read(0, &mut out)?;
        Ok(out)
    }
}

/// `qxvision MODEL IMAGE... [--pixels N] [--out DIR] [--rgb=1]` (a development check): each image through the preprocessing and the tower alone, its patches (DIR/NAME.patches.bf16, [patches,
/// 1536]), its grid (DIR/NAME.json) and its rows (DIR/NAME.rows.bf16, [tokens, 2560]) written for research/qwen-data/qwen_ref.py's `vision` comparison; the tower's time. --rgb=1: the decoded
/// pixels alone (DIR/NAME.WxH.rgb, no GPU), to set beside PIL's.
#[cfg(feature = "dev")]
pub fn check(a: &crate::args::Args) -> Result<()> {
    let dir = Path::new(a.need(1, "MODEL")?);
    let pixels = a.flag("pixels").map(|v| v.parse::<usize>().map_err(|_| Error(format!("--pixels {v}: a number")))).transpose()?;
    let prep = Prep::load(dir, pixels)?.ok_or_else(|| Error(format!("{}: no vision tower", dir.display())))?;
    let gpu = if a.has("rgb") {
        None
    } else {
        let dev = Dev::new(0)?;
        let kern = Kern::load(&dev)?;
        let tower = Tower::open(dir, &dev)?.ok_or_else(|| Error("no vision tower".into()))?;
        Some((dev, kern, tower))
    };
    let out = Path::new(a.flag("out").unwrap_or("."));
    std::fs::create_dir_all(out).map_err(|e| Error(format!("{}: {e}", out.display())))?;
    for path in a.pos.iter().skip(2) {
        let bytes =
            if path.contains("://") || path.starts_with("data:") { fetch(path)? } else { std::fs::read(path).map_err(|e| Error(format!("{path}: {e}")))? };
        let name = Path::new(path).file_stem().map_or("image".into(), |s| s.to_string_lossy().to_string());
        let Some((dev, kern, tower)) = gpu.as_ref() else {
            let t = std::time::Instant::now();
            match crate::pixels::rgb(&bytes) {
                Ok((w, h, px)) => {
                    eprintln!("{path}: {w} x {h} decoded in {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
                    std::fs::write(out.join(format!("{name}.{w}x{h}.rgb")), px).map_err(|e| Error(e.to_string()))?;
                }
                Err(e) => eprintln!("{path}: {e}"),
            }
            continue;
        };
        let t0 = std::time::Instant::now();
        let img = prep.image(&bytes)?;
        let prep_ms = t0.elapsed().as_secs_f64() * 1e3;
        let free0 = dev.mem_info()?.0;
        let t1 = std::time::Instant::now();
        let rows = tower.run(&dev, &kern, &img)?;
        let ms = t1.elapsed().as_secs_f64() * 1e3;
        let bytes_of = |v: &[u16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        std::fs::write(out.join(format!("{name}.patches.bf16")), bytes_of(&img.patches)).map_err(|e| Error(e.to_string()))?;
        std::fs::write(out.join(format!("{name}.rows.bf16")), bytes_of(&rows)).map_err(|e| Error(e.to_string()))?;
        std::fs::write(out.join(format!("{name}.json")), format!("{{\"grid_thw\":[1,{},{}],\"tokens\":{}}}\n", img.gh, img.gw, img.tokens()))
            .map_err(|e| Error(e.to_string()))?;
        eprintln!(
            "{path}: {} x {} patches, {} tokens; preprocessing {prep_ms:.1} ms, the tower {ms:.1} ms; {:.0} MB of GPU memory wanted ({:.0} MB free before)",
            img.gh,
            img.gw,
            img.tokens(),
            Tower::bytes(img.gh * img.gw) as f64 / 1e6,
            free0 as f64 / 1e6
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A resolver that answers what it was given: the lookup of a name whose DNS changes its answer between the check and the connection.
    #[derive(Debug)]
    struct Fixed(Vec<&'static str>);

    impl ureq::unversioned::resolver::Resolver for Fixed {
        fn resolve(
            &self,
            _: &ureq::http::Uri,
            _: &ureq::config::Config,
            _: ureq::unversioned::transport::NextTimeout,
        ) -> std::result::Result<ureq::unversioned::resolver::ResolvedSocketAddrs, ureq::Error> {
            let mut v = ureq::unversioned::resolver::ArrayVec::from_fn(|_| "0.0.0.0:1".parse().unwrap());
            for a in &self.0 {
                v.push(format!("{a}:80").parse().unwrap());
            }
            Ok(v)
        }
    }

    /// The connection takes its addresses from the resolver that checks them: a name that resolves to a private address (the answer a rebinding DNS gives the second time), or to a public one and a private one together,
    /// is refused where the connection is made; a public one is let through as it came.
    #[test]
    fn the_address_that_is_connected_to_is_the_address_that_was_checked() {
        use ureq::unversioned::resolver::Resolver;
        let uri: ureq::http::Uri = "http://rebind.example/x.png".parse().unwrap();
        let cfg = ureq::config::Config::default();
        let ask = |ips: Vec<&'static str>| {
            PublicOnly(Fixed(ips)).resolve(
                &uri,
                &cfg,
                ureq::unversioned::transport::NextTimeout {
                    after: ureq::unversioned::transport::time::Duration::NotHappening,
                    reason: ureq::Timeout::Global,
                    ..Default::default()
                },
            )
        };
        for ips in [vec!["127.0.0.1"], vec!["169.254.169.254"], vec!["10.0.0.5"], vec!["93.184.216.34", "192.168.1.1"], vec!["192.168.1.1", "93.184.216.34"]] {
            let e = ask(ips.clone()).map(|_| ()).unwrap_err().to_string();
            assert!(e.contains("only public addresses"), "{ips:?}: {e}");
        }
        let ok = ask(vec!["93.184.216.34", "8.8.8.8"]).unwrap();
        assert_eq!(ok.iter().map(|a| a.to_string()).collect::<Vec<_>>(), ["93.184.216.34:80", "8.8.8.8:80"]);
        // and a URL of an address that is this computer's own is refused by `fetch` itself
        for url in
            ["http://127.0.0.1:1/a.png", "http://[::1]:1/a.png", "http://169.254.169.254/latest/meta-data", "http://localhost:9/a.png", "file:///etc/passwd"]
        {
            assert!(fetch(url).is_err(), "{url}");
        }
    }

    #[test]
    fn only_public_hosts_are_fetched() {
        for ip in [
            "127.0.0.1",
            "10.1.2.3",
            "172.16.5.5",
            "192.168.1.1",
            "169.254.169.254",
            "100.64.0.1",
            "0.0.0.0",
            "224.0.0.1",
            "240.0.0.1",
            "192.0.0.1",
            "198.18.0.1",
            "::1",
            "::",
            "fe80::1",
            "fc00::1",
            "fd12::1",
            "ff02::1",
            "::ffff:127.0.0.1",
            "::ffff:10.0.0.1",
            "2001:db8::1",
        ] {
            assert!(!public_ip(ip.parse().unwrap()), "{ip}");
        }
        for ip in ["8.8.8.8", "93.184.216.34", "2606:4700::1111", "2a00:1450:4001::1"] {
            assert!(public_ip(ip.parse().unwrap()), "{ip}");
        }
        assert!(public_host("http://localhost/a.png").is_err());
        assert!(public_host("http://127.0.0.1:8080/a.png").is_err());
        assert!(public_host("http://[::1]/a.png").is_err());
        assert!(public_host("http://user@169.254.169.254/latest/meta-data").is_err());
        assert!(public_host("http://10.0.0.1:99999/x").is_err());
        assert!(public_host("http:///x").is_err());
        assert!(fetch("file:///etc/passwd").is_err());
    }

    #[test]
    fn base64_as_the_standard_has_it() {
        assert_eq!(base64("aGVsbG8gd29ybGQ=").unwrap(), b"hello world");
        assert_eq!(base64("aGVs\nbG8").unwrap(), b"hello");
        assert!(base64("a*b").is_err());
    }

    #[test]
    fn smart_resize_as_the_processor_has_it() {
        // (values from transformers' smart_resize at factor 32, 65536 to 16777216 pixels; and with the top at 1048576)
        assert_eq!(smart_resize(480, 640, 65536, 16_777_216).unwrap(), (480, 640));
        assert_eq!(smart_resize(100, 100, 65536, 16_777_216).unwrap(), (256, 256));
        assert_eq!(smart_resize(48, 80, 65536, 16_777_216).unwrap(), (224, 352));
        assert_eq!(smart_resize(3000, 4000, 65536, 1_048_576).unwrap(), (864, 1152));
        assert_eq!(smart_resize(1080, 1920, 65536, 16_777_216).unwrap(), (1088, 1920));
        assert!(smart_resize(1, 300, 65536, 16_777_216).is_err());
    }

    #[test]
    fn a_resize_that_keeps_a_flat_image_flat() {
        let rgb: Vec<u8> = (0..37 * 23).flat_map(|_| [200u8, 10, 99]).collect();
        for (nw, nh) in [(64, 32), (16, 64), (37, 96)] {
            let out = resize(&rgb, 37, 23, nw, nh);
            assert_eq!(out.len(), nw * nh * 3);
            assert!(out.chunks(3).all(|p| p == [200, 10, 99]), "{nw} x {nh}");
        }
    }

    #[test]
    fn positions_of_text_and_an_image() {
        // text 0..3, an image of 2 x 3 tokens from position 3 (its rows and columns from there), then text from 3 + 3
        let ids = [1, 2, 3, 9, 9, 9, 9, 9, 9, 4, 5];
        let (p, next) = positions(&ids, 9, &[(2, 3)]).unwrap();
        assert_eq!(&p[..3], &[[0; 3], [1; 3], [2; 3]]);
        assert_eq!(&p[3..9], &[[3, 3, 3], [3, 3, 4], [3, 3, 5], [3, 4, 3], [3, 4, 4], [3, 4, 5]]);
        assert_eq!(&p[9..], &[[6; 3], [7; 3]]);
        assert_eq!(next, 8);
        assert!(positions(&ids, 9, &[(2, 2)]).is_err());
        assert!(positions(&ids, 9, &[]).is_err());
        assert_eq!(positions(&[5, 6], 9, &[]).unwrap(), (vec![[0; 3], [1; 3]], 2));
        // the template's one place an image, made as long as its tokens
        let img = |gh, gw| Arc::new(Image { gh, gw, patches: Vec::new(), key: 0 });
        assert_eq!(expand(&[1, 9, 2, 9, 3], 9, &[img(2, 4), img(4, 2)]).unwrap(), [1, 9, 9, 2, 9, 9, 3]);
        assert!(expand(&[1, 9, 9], 9, &[img(2, 2)]).is_err());
    }

    #[test]
    fn rope_rows_of_text_are_the_plain_tables() {
        let (c0, s0) = crate::weights::rope_tables(1e7, 64, 40);
        let rows: Vec<[u32; 3]> = (0..40).map(|p| [p; 3]).collect();
        let (c, s) = rope_rows(1e7, 32, &rows);
        assert_eq!((c, s), (c0, s0));
    }
}
