use std::slice;
use crate::format::MAX_BLOCK_SIZE;

/// Return safe upper bound for compressed destination buffer.
#[no_mangle]
pub extern "C" fn glyd_max_compressed_len(src_len: usize) -> usize {
    let num_blocks = (src_len + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
    src_len + (num_blocks.max(1) * 1024) + 64
}

/// Return version string.
#[no_mangle]
pub extern "C" fn glyd_version() -> *const std::ffi::c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const std::ffi::c_char
}

/// Compress sequentially using single-core engine.
///
/// Returns: Number of compressed bytes written to `dst`, or negative on error:
///   -1: Destination buffer too small
///   -2: Null pointer passed
#[no_mangle]
pub unsafe extern "C" fn glyd_compress(
    src: *const u8,
    src_len: usize,
    dst: *mut u8,
    dst_capacity: usize,
) -> isize {
    if src.is_null() || dst.is_null() {
        return -2;
    }

    let input = slice::from_raw_parts(src, src_len);
    let mut out_vec = Vec::with_capacity(dst_capacity);
    crate::compress_into(input, &mut out_vec);

    if out_vec.len() > dst_capacity {
        return -1;
    }

    std::ptr::copy_nonoverlapping(out_vec.as_ptr(), dst, out_vec.len());
    out_vec.len() as isize
}

/// Compress in parallel across all CPU cores using Rayon.
///
/// Returns: Number of compressed bytes written to `dst`, or negative on error:
///   -1: Destination buffer too small
///   -2: Null pointer passed
#[no_mangle]
pub unsafe extern "C" fn glyd_compress_parallel(
    src: *const u8,
    src_len: usize,
    dst: *mut u8,
    dst_capacity: usize,
) -> isize {
    if src.is_null() || dst.is_null() {
        return -2;
    }

    let input = slice::from_raw_parts(src, src_len);
    let mut out_vec = Vec::with_capacity(dst_capacity);
    crate::compress_parallel_into(input, &mut out_vec);

    if out_vec.len() > dst_capacity {
        return -1;
    }

    std::ptr::copy_nonoverlapping(out_vec.as_ptr(), dst, out_vec.len());
    out_vec.len() as isize
}

/// Compress sequentially using the max level (format v7, entropy coded).
///
/// Returns: Number of compressed bytes written to `dst`, or negative on error:
///   -1: Destination buffer too small
///   -2: Null pointer passed
#[no_mangle]
pub unsafe extern "C" fn glyd_compress_max(
    src: *const u8,
    src_len: usize,
    dst: *mut u8,
    dst_capacity: usize,
) -> isize {
    if src.is_null() || dst.is_null() {
        return -2;
    }

    let input = slice::from_raw_parts(src, src_len);
    let mut out_vec = Vec::with_capacity(dst_capacity);
    crate::compress_into_max(input, &mut out_vec);

    if out_vec.len() > dst_capacity {
        return -1;
    }

    std::ptr::copy_nonoverlapping(out_vec.as_ptr(), dst, out_vec.len());
    out_vec.len() as isize
}

/// Compress sequentially using the ultra level (format v7 on the optimal
/// parse: denser than the max level, much slower to produce, same decoder).
///
/// Returns: Number of compressed bytes written to `dst`, or negative on error:
///   -1: Destination buffer too small
///   -2: Null pointer passed
#[no_mangle]
pub unsafe extern "C" fn glyd_compress_ultra(
    src: *const u8,
    src_len: usize,
    dst: *mut u8,
    dst_capacity: usize,
) -> isize {
    if src.is_null() || dst.is_null() {
        return -2;
    }

    let input = slice::from_raw_parts(src, src_len);
    let mut out_vec = Vec::with_capacity(dst_capacity);
    crate::compress_into_ultra(input, &mut out_vec);

    if out_vec.len() > dst_capacity {
        return -1;
    }

    std::ptr::copy_nonoverlapping(out_vec.as_ptr(), dst, out_vec.len());
    out_vec.len() as isize
}

/// Compress in parallel across all CPU cores using the ultra level.
///
/// Returns: Number of compressed bytes written to `dst`, or negative on error:
///   -1: Destination buffer too small
///   -2: Null pointer passed
#[no_mangle]
pub unsafe extern "C" fn glyd_compress_ultra_parallel(
    src: *const u8,
    src_len: usize,
    dst: *mut u8,
    dst_capacity: usize,
) -> isize {
    if src.is_null() || dst.is_null() {
        return -2;
    }

    let input = slice::from_raw_parts(src, src_len);
    let mut out_vec = Vec::with_capacity(dst_capacity);
    crate::compress_parallel_into_ultra(input, &mut out_vec);

    if out_vec.len() > dst_capacity {
        return -1;
    }

    std::ptr::copy_nonoverlapping(out_vec.as_ptr(), dst, out_vec.len());
    out_vec.len() as isize
}

/// Compress in parallel across all CPU cores using the max level (format v7,
/// entropy coded).
///
/// Returns: Number of compressed bytes written to `dst`, or negative on error:
///   -1: Destination buffer too small
///   -2: Null pointer passed
#[no_mangle]
pub unsafe extern "C" fn glyd_compress_max_parallel(
    src: *const u8,
    src_len: usize,
    dst: *mut u8,
    dst_capacity: usize,
) -> isize {
    if src.is_null() || dst.is_null() {
        return -2;
    }

    let input = slice::from_raw_parts(src, src_len);
    let mut out_vec = Vec::with_capacity(dst_capacity);
    crate::compress_parallel_into_max(input, &mut out_vec);

    if out_vec.len() > dst_capacity {
        return -1;
    }

    std::ptr::copy_nonoverlapping(out_vec.as_ptr(), dst, out_vec.len());
    out_vec.len() as isize
}

/// The fast level into a caller's buffer (as `glyd_compress`).
#[no_mangle]
pub unsafe extern "C" fn glyd_compress_fast(src: *const u8, src_len: usize, dst: *mut u8, dst_capacity: usize) -> isize {
    if src.is_null() || dst.is_null() {
        return -2;
    }
    let mut out_vec = Vec::with_capacity(dst_capacity);
    crate::compress_into_fast(slice::from_raw_parts(src, src_len), &mut out_vec);
    if out_vec.len() > dst_capacity {
        return -1;
    }
    std::ptr::copy_nonoverlapping(out_vec.as_ptr(), dst, out_vec.len());
    out_vec.len() as isize
}

/// The turbo level into a caller's buffer (as `glyd_compress`).
#[no_mangle]
pub unsafe extern "C" fn glyd_compress_turbo(src: *const u8, src_len: usize, dst: *mut u8, dst_capacity: usize) -> isize {
    if src.is_null() || dst.is_null() {
        return -2;
    }
    let mut out_vec = Vec::with_capacity(dst_capacity);
    crate::compress_into_turbo(slice::from_raw_parts(src, src_len), &mut out_vec);
    if out_vec.len() > dst_capacity {
        return -1;
    }
    std::ptr::copy_nonoverlapping(out_vec.as_ptr(), dst, out_vec.len());
    out_vec.len() as isize
}

/// Decompress sequentially using single-core engine with checksum validation.
///
/// Returns: Number of uncompressed bytes written to `dst`, or negative on error:
///   -1: Output buffer too small
///   -2: Corrupted bitstream or checksum mismatch
///   -3: Null pointer passed
#[no_mangle]
pub unsafe extern "C" fn glyd_decompress(
    src: *const u8,
    src_len: usize,
    dst: *mut u8,
    dst_capacity: usize,
) -> isize {
    if src.is_null() || dst.is_null() {
        return -3;
    }

    let compressed = slice::from_raw_parts(src, src_len);
    let dst_slice = slice::from_raw_parts_mut(dst, dst_capacity);

    match crate::decompress_into(compressed, dst_slice) {
        Ok(written) => written as isize,
        Err(crate::error::CodecError::OutputBufferTooSmall { .. }) => -1,
        Err(_) => -2,
    }
}

/// Decompress in parallel across all CPU cores with checksum validation.
///
/// Returns: Number of uncompressed bytes written to `dst`, or negative on error:
///   -1: Output buffer too small
///   -2: Corrupted bitstream or checksum mismatch
///   -3: Null pointer passed
#[no_mangle]
pub unsafe extern "C" fn glyd_decompress_parallel(
    src: *const u8,
    src_len: usize,
    dst: *mut u8,
    dst_capacity: usize,
) -> isize {
    if src.is_null() || dst.is_null() {
        return -3;
    }

    let compressed = slice::from_raw_parts(src, src_len);
    let dst_slice = slice::from_raw_parts_mut(dst, dst_capacity);

    match crate::decompress_parallel_into(compressed, dst_slice) {
        Ok(written) => written as isize,
        Err(crate::error::CodecError::OutputBufferTooSmall { .. }) => -1,
        Err(_) => -2,
    }
}


// ---------------------------------------------------------------------------
// The allocating API (v0.9.3): one call per operation, the output
// allocated by the library and released with `glyd_free`. Levels and
// modes by number; the store and packs. Language bindings sit on this.
// ---------------------------------------------------------------------------

/// Levels for `glyd_compress2`.
pub const GLYD_LEVEL_DEFAULT: i32 = 0;
pub const GLYD_LEVEL_FAST: i32 = 1;
pub const GLYD_LEVEL_TURBO: i32 = 2;
pub const GLYD_LEVEL_MAX: i32 = 3;
pub const GLYD_LEVEL_ULTRA: i32 = 4;
pub const GLYD_LEVEL_COLD: i32 = 5;

/// A library-allocated buffer handed to the caller: exactly `len`
/// bytes, freed by `glyd_free(ptr, len)`.
unsafe fn hand_out(v: Vec<u8>, out: *mut *mut u8, out_len: *mut usize) -> i32 {
    let b = v.into_boxed_slice();
    let len = b.len();
    *out = Box::into_raw(b) as *mut u8;
    *out_len = len;
    0
}

/// Release a buffer from `glyd_compress2`, `glyd_decompress2`,
/// `glyd_store_get`, `glyd_pack` or `glyd_unpack_object`.
#[no_mangle]
pub unsafe extern "C" fn glyd_free(ptr: *mut u8, len: usize) {
    if !ptr.is_null() {
        drop(Box::from_raw(slice::from_raw_parts_mut(ptr, len)));
    }
}

/// Compress `src` at `level` (GLYD_LEVEL_*), in record mode when
/// `records` is nonzero, on all cores when `threads` is 0 or more than 1
/// and on one when it is 1. Returns 0 and the output in `out` /
/// `out_len`, or -1 for a bad argument.
#[no_mangle]
pub unsafe extern "C" fn glyd_compress2(src: *const u8, src_len: usize, level: i32, records: i32, threads: i32, out: *mut *mut u8, out_len: *mut usize) -> i32 {
    if (src.is_null() && src_len > 0) || out.is_null() || out_len.is_null() {
        return -1;
    }
    let input = if src_len == 0 { &[][..] } else { slice::from_raw_parts(src, src_len) };
    let multi = threads != 1;
    let mut v = Vec::with_capacity(src_len / 3 + 1024);
    match (level, records != 0, multi) {
        (GLYD_LEVEL_DEFAULT, false, true) => crate::compress_parallel_into(input, &mut v),
        (GLYD_LEVEL_DEFAULT, false, false) => crate::compress_into(input, &mut v),
        (GLYD_LEVEL_FAST, false, true) => crate::compress_parallel_into_fast(input, &mut v),
        (GLYD_LEVEL_FAST, false, false) => crate::compress_into_fast(input, &mut v),
        (GLYD_LEVEL_TURBO, false, true) => crate::compress_parallel_into_turbo(input, &mut v),
        (GLYD_LEVEL_TURBO, false, false) => crate::compress_into_turbo(input, &mut v),
        (GLYD_LEVEL_MAX, false, true) => crate::compress_parallel_into_max(input, &mut v),
        (GLYD_LEVEL_MAX, false, false) => crate::compress_into_max(input, &mut v),
        (GLYD_LEVEL_ULTRA, false, true) => crate::compress_parallel_into_ultra(input, &mut v),
        (GLYD_LEVEL_ULTRA, false, false) => crate::compress_into_ultra(input, &mut v),
        (GLYD_LEVEL_COLD, false, true) => crate::compress_parallel_into_cold(input, &mut v),
        (GLYD_LEVEL_COLD, false, false) => crate::compress_into_cold(input, &mut v),
        (GLYD_LEVEL_DEFAULT, true, _) => crate::compress_records_with(input, &mut v, crate::compress_into),
        (GLYD_LEVEL_FAST, true, _) => crate::compress_records_with(input, &mut v, crate::compress_into_fast),
        (GLYD_LEVEL_TURBO, true, _) => crate::compress_records_with(input, &mut v, crate::compress_into_turbo),
        (GLYD_LEVEL_MAX, true, _) => crate::compress_records_into_max(input, &mut v),
        (GLYD_LEVEL_ULTRA, true, _) => crate::compress_records_into_ultra(input, &mut v),
        (GLYD_LEVEL_COLD, true, _) => crate::compress_records_into_cold(input, &mut v),
        _ => return -1,
    }
    hand_out(v, out, out_len)
}

/// Decompress any Glyd stream (every level, record mode, packs, the
/// cold level) into a library-allocated buffer. Returns 0, -1 for a bad
/// argument, -2 for a corrupt stream (a base-mode stream needs
/// `glyd_decompress_with_base`).
#[no_mangle]
pub unsafe extern "C" fn glyd_decompress2(src: *const u8, src_len: usize, out: *mut *mut u8, out_len: *mut usize) -> i32 {
    if (src.is_null() && src_len > 0) || out.is_null() || out_len.is_null() {
        return -1;
    }
    let input = if src_len == 0 { &[][..] } else { slice::from_raw_parts(src, src_len) };
    match crate::decompress_parallel(input) {
        Ok(v) => hand_out(v, out, out_len),
        Err(_) => -2,
    }
}

/// The decompressed size of a stream, or -1 when it is not one.
#[no_mangle]
pub unsafe extern "C" fn glyd_decompressed_len(src: *const u8, src_len: usize) -> i64 {
    if src.is_null() && src_len > 0 {
        return -1;
    }
    let input = if src_len == 0 { &[][..] } else { slice::from_raw_parts(src, src_len) };
    match crate::decompressed_len(input) {
        Ok(n) => n as i64,
        Err(_) => -1,
    }
}

/// Base mode: `src` compressed against `base` (max level, or ultra when
/// `ultra` is nonzero); decoding needs the same base.
#[no_mangle]
pub unsafe extern "C" fn glyd_compress_with_base(base: *const u8, base_len: usize, src: *const u8, src_len: usize, ultra: i32, out: *mut *mut u8, out_len: *mut usize) -> i32 {
    if base.is_null() || src.is_null() || out.is_null() || out_len.is_null() {
        return -1;
    }
    let mut v = Vec::new();
    crate::compress_with_base(slice::from_raw_parts(base, base_len), slice::from_raw_parts(src, src_len), &mut v, ultra != 0);
    hand_out(v, out, out_len)
}

#[no_mangle]
pub unsafe extern "C" fn glyd_decompress_with_base(base: *const u8, base_len: usize, src: *const u8, src_len: usize, out: *mut *mut u8, out_len: *mut usize) -> i32 {
    if base.is_null() || src.is_null() || out.is_null() || out_len.is_null() {
        return -1;
    }
    match crate::decompress_with_base(slice::from_raw_parts(base, base_len), slice::from_raw_parts(src, src_len)) {
        Ok(v) => hand_out(v, out, out_len),
        Err(_) => -2,
    }
}

/// Many small objects as one stream with an index (`objs[i]` of
/// `lens[i]` bytes), at `level` (GLYD_LEVEL_MAX, _ULTRA or _COLD).
#[no_mangle]
pub unsafe extern "C" fn glyd_pack(objs: *const *const u8, lens: *const usize, n: usize, level: i32, out: *mut *mut u8, out_len: *mut usize) -> i32 {
    if (objs.is_null() || lens.is_null()) && n > 0 || out.is_null() || out_len.is_null() {
        return -1;
    }
    let ptrs = if n == 0 { &[][..] } else { slice::from_raw_parts(objs, n) };
    let lens = if n == 0 { &[][..] } else { slice::from_raw_parts(lens, n) };
    let objects: Vec<&[u8]> = ptrs.iter().zip(lens).map(|(&p, &l)| if l == 0 { &[][..] } else { slice::from_raw_parts(p, l) }).collect();
    let level: fn(&[u8], &mut Vec<u8>) = match level {
        GLYD_LEVEL_MAX => crate::compress_into_max,
        GLYD_LEVEL_ULTRA => crate::compress_into_ultra,
        GLYD_LEVEL_COLD => crate::compress_into_cold,
        _ => return -1,
    };
    let mut v = Vec::new();
    crate::compress_pack(&objects, &mut v, level);
    hand_out(v, out, out_len)
}

/// Object `i` of a pack.
#[no_mangle]
pub unsafe extern "C" fn glyd_unpack_object(pack: *const u8, pack_len: usize, i: usize, out: *mut *mut u8, out_len: *mut usize) -> i32 {
    if pack.is_null() || out.is_null() || out_len.is_null() {
        return -1;
    }
    match crate::decompress_pack_object(slice::from_raw_parts(pack, pack_len), i) {
        Ok(v) => hand_out(v, out, out_len),
        Err(_) => -2,
    }
}

/// The object count of a pack, or -1.
#[no_mangle]
pub unsafe extern "C" fn glyd_pack_len(pack: *const u8, pack_len: usize) -> i64 {
    if pack.is_null() {
        return -1;
    }
    crate::pack_len(slice::from_raw_parts(pack, pack_len)).map_or(-1, |n| n as i64)
}
