//! The store's C ABI (`include/glyd.h`, the `glyd_store_*` functions):
//! built into `libglyd_store`, which carries the codec's ABI as well.
use std::slice;

/// A store (`glyd_store_*`): metadata at `dir`, objects there too, or
/// in `s3_url` (s3://bucket/prefix, through the AWS CLI) when given.
pub struct GlydStore(crate::Store);

const GLYD_LEVEL_MAX: i32 = 3;
const GLYD_LEVEL_ULTRA: i32 = 4;
const GLYD_LEVEL_COLD: i32 = 5;

/// A library-allocated buffer for the caller, released by `glyd_free`
/// (the codec's, in the same library).
unsafe fn hand_out(v: Vec<u8>, out: *mut *mut u8, out_len: *mut usize) -> i32 {
    let b = v.into_boxed_slice();
    let len = b.len();
    *out = Box::into_raw(b) as *mut u8;
    *out_len = len;
    0
}

#[no_mangle]
pub unsafe extern "C" fn glyd_store_open(dir: *const std::ffi::c_char, s3_url: *const std::ffi::c_char) -> *mut GlydStore {
    if dir.is_null() {
        return std::ptr::null_mut();
    }
    let dir = std::ffi::CStr::from_ptr(dir).to_string_lossy().to_string();
    let store = if s3_url.is_null() {
        crate::Store::open(&dir)
    } else {
        let url = std::ffi::CStr::from_ptr(s3_url).to_string_lossy().to_string();
        crate::S3Cli::new(&url).and_then(|b| crate::Store::open_with(&dir, Box::new(b)))
    };
    match store {
        Ok(s) => Box::into_raw(Box::new(GlydStore(s))),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Close a store (the open pack is written).
#[no_mangle]
pub unsafe extern "C" fn glyd_store_close(store: *mut GlydStore) {
    if !store.is_null() {
        drop(Box::from_raw(store));
    }
}

/// Store an object under `name`; its id, or -1.
#[no_mangle]
pub unsafe extern "C" fn glyd_store_put(store: *mut GlydStore, name: *const std::ffi::c_char, data: *const u8, len: usize) -> i64 {
    if store.is_null() || name.is_null() || (data.is_null() && len > 0) {
        return -1;
    }
    let name = std::ffi::CStr::from_ptr(name).to_string_lossy().to_string();
    let data = if len == 0 { &[][..] } else { slice::from_raw_parts(data, len) };
    (*store).0.put(&name, data).map_or(-1, |id| id as i64)
}

/// Object `id` back: 0, -1 for a bad argument, -2 when the store refuses it.
#[no_mangle]
pub unsafe extern "C" fn glyd_store_get(store: *mut GlydStore, id: u32, out: *mut *mut u8, out_len: *mut usize) -> i32 {
    if store.is_null() || out.is_null() || out_len.is_null() {
        return -1;
    }
    match (*store).0.get(id) {
        Ok(v) => hand_out(v, out, out_len),
        Err(_) => -2,
    }
}

/// The latest live object under `name`, or -1.
#[no_mangle]
pub unsafe extern "C" fn glyd_store_id_of(store: *mut GlydStore, name: *const std::ffi::c_char) -> i64 {
    if store.is_null() || name.is_null() {
        return -1;
    }
    let name = std::ffi::CStr::from_ptr(name).to_string_lossy();
    (*store).0.id_of(&name).map_or(-1, |id| id as i64)
}

#[no_mangle]
pub unsafe extern "C" fn glyd_store_delete(store: *mut GlydStore, id: u32) -> i32 {
    if store.is_null() {
        return -1;
    }
    if (*store).0.delete(id).is_ok() { 0 } else { -2 }
}

/// Bytes freed, or -1.
#[no_mangle]
pub unsafe extern "C" fn glyd_store_compact(store: *mut GlydStore) -> i64 {
    if store.is_null() {
        return -1;
    }
    (*store).0.compact().map_or(-1, |n| n as i64)
}

#[no_mangle]
pub unsafe extern "C" fn glyd_store_flush(store: *mut GlydStore) -> i32 {
    if store.is_null() {
        return -1;
    }
    if (*store).0.flush().is_ok() { 0 } else { -2 }
}

#[no_mangle]
pub unsafe extern "C" fn glyd_store_rebase(store: *mut GlydStore, id: u32) -> i32 {
    if store.is_null() {
        return -1;
    }
    if (*store).0.rebase(id).is_ok() { 0 } else { -2 }
}

/// Objects that failed verification (every live object read back), or -1.
#[no_mangle]
pub unsafe extern "C" fn glyd_store_verify(store: *mut GlydStore) -> i64 {
    if store.is_null() {
        return -1;
    }
    (*store).0.verify().1.len() as i64
}

/// The store's raw bytes and bytes on disk.
#[no_mangle]
pub unsafe extern "C" fn glyd_store_stats(store: *mut GlydStore, raw: *mut u64, stored: *mut u64) -> i32 {
    if store.is_null() || raw.is_null() || stored.is_null() {
        return -1;
    }
    let (r, s) = (*store).0.stats();
    *raw = r;
    *stored = s;
    0
}

/// The level objects stored alone take from now on (GLYD_LEVEL_MAX,
/// _ULTRA or _COLD).
#[no_mangle]
pub unsafe extern "C" fn glyd_store_set_level(store: *mut GlydStore, level: i32) -> i32 {
    if store.is_null() {
        return -1;
    }
    let level = match level {
        GLYD_LEVEL_MAX => crate::Level::Max,
        GLYD_LEVEL_ULTRA => crate::Level::Ultra,
        GLYD_LEVEL_COLD => crate::Level::Cold,
        _ => return -1,
    };
    (*store).0.set_level(level);
    0
}

/// The number of objects the store knows (ids run 0..count).
#[no_mangle]
pub unsafe extern "C" fn glyd_store_count(store: *mut GlydStore) -> i64 {
    if store.is_null() {
        return -1;
    }
    (*store).0.entries().len() as i64
}
