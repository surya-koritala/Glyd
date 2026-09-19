use std::slice;
use crate::format::MAX_BLOCK_SIZE;

/// Return safe upper bound for compressed destination buffer.
#[no_mangle]
pub extern "C" fn alatirok_max_compressed_len(src_len: usize) -> usize {
    let num_blocks = (src_len + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
    src_len + (num_blocks.max(1) * 1024) + 64
}

/// Return version string.
#[no_mangle]
pub extern "C" fn alatirok_version() -> *const std::ffi::c_char {
    concat!("0.1.0", "\0").as_ptr() as *const std::ffi::c_char
}

/// Compress sequentially using single-core engine.
///
/// Returns: Number of compressed bytes written to `dst`, or negative on error:
///   -1: Destination buffer too small
///   -2: Null pointer passed
#[no_mangle]
pub unsafe extern "C" fn alatirok_compress(
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
pub unsafe extern "C" fn alatirok_compress_parallel(
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
pub unsafe extern "C" fn alatirok_compress_max(
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

/// Compress in parallel across all CPU cores using the max level (format v7,
/// entropy coded).
///
/// Returns: Number of compressed bytes written to `dst`, or negative on error:
///   -1: Destination buffer too small
///   -2: Null pointer passed
#[no_mangle]
pub unsafe extern "C" fn alatirok_compress_max_parallel(
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

/// Decompress sequentially using single-core engine with checksum validation.
///
/// Returns: Number of uncompressed bytes written to `dst`, or negative on error:
///   -1: Output buffer too small
///   -2: Corrupted bitstream or checksum mismatch
///   -3: Null pointer passed
#[no_mangle]
pub unsafe extern "C" fn alatirok_decompress(
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
pub unsafe extern "C" fn alatirok_decompress_parallel(
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
