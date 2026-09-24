pub mod error;
pub mod format;
pub mod huffman;
pub mod finder;
pub mod fallback;
#[cfg(target_arch = "x86_64")]
pub mod x86_decompress;
#[cfg(target_arch = "x86_64")]
pub mod x86_compress;
#[cfg(target_arch = "x86_64")]
pub mod x86_checksum;
#[cfg(target_arch = "aarch64")]
pub mod neon_decompress;
#[cfg(target_arch = "aarch64")]
pub mod neon_checksum;
pub mod streaming;
pub mod c_api;
pub mod bits;
pub mod huff8;
pub mod tans;
pub mod v7_format;
pub mod v7_encode;
pub mod v7_decode;
pub mod v7_ultra;
pub mod cm;
pub mod shape;
pub mod mmap;
#[cfg(feature = "deflate")]
pub mod deflate;
pub mod reflate;
pub mod jpg;
pub mod jpeg;
pub mod fixlog;
pub mod split;
pub mod record;
pub mod ldm;
pub mod dict;
pub use dict::Dict;
pub use shape::ShapeDict;

pub use streaming::{GlydReader, GlydWriter};
pub use format::compute_checksum;

use error::{CodecError, Result};
use finder::{new_table, Dense, HashTable, Lzav, Mode, Turbo};
use format::*;
use v7_format::LOCAL_WINDOW;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Threads the parallel paths use: `set_threads`, else the machine's.
static THREADS: AtomicUsize = AtomicUsize::new(0);

/// Use `n` threads for the parallel paths (0: the machine's count).
pub fn set_threads(n: usize) {
    THREADS.store(n, Ordering::Relaxed);
}

pub(crate) fn threads() -> usize {
    match THREADS.load(Ordering::Relaxed) {
        0 => std::thread::available_parallelism().map_or(1, |t| t.get()),
        n => n,
    }
}

thread_local! {
    /// Set on a thread while it compresses a part of an object: a unit
    /// of a parallel stream, a record unit, a trial sample, an opened
    /// container's plain text. A part is never opened as a container of
    /// its own: its output is laid into a stream whose decoder reads
    /// plain blocks.
    static PART: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// `f` with this thread marked as compressing a part (see `PART`).
pub(crate) fn as_part<T>(f: impl FnOnce() -> T) -> T {
    let old = PART.with(|p| p.replace(true));
    let r = f();
    PART.with(|p| p.set(old));
    r
}

pub(crate) fn in_part() -> bool {
    PART.with(|p| p.get())
}

/// `f(i)` for every `i < n`, on up to `threads()` scoped threads that
/// each take the next unit as they finish one and exit when none is
/// left (no idle thread spins, so the CPU time is the work's). The
/// first error stops the rest and is returned.
pub(crate) fn par_units<E: Send>(n: usize, f: impl Fn(usize) -> std::result::Result<(), E> + Sync) -> std::result::Result<(), E> {
    let workers = threads().min(n);
    if workers <= 1 {
        return (0..n).try_for_each(f);
    }
    let next = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let run = || -> std::result::Result<(), E> {
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= n || stop.load(Ordering::Relaxed) {
                return Ok(());
            }
            if let Err(e) = f(i) {
                stop.store(true, Ordering::Relaxed);
                return Err(e);
            }
        }
    };
    std::thread::scope(|s| {
        let handles: Vec<_> = (1..workers).map(|_| s.spawn(run)).collect();
        let mut result = run();
        for h in handles {
            if let Err(e) = h.join().expect("a worker panicked") {
                if result.is_ok() {
                    result = Err(e);
                }
            }
        }
        result
    })
}
use std::cell::RefCell;

struct CompressScratch {
    tokens: Vec<u8>,
    offsets: Vec<u8>,
    extras: Vec<u8>,
    literals: Vec<u8>,
    table: Box<HashTable>,
}

thread_local! {
    static COMPRESS_SCRATCH: RefCell<CompressScratch> = RefCell::new(CompressScratch {
        tokens: Vec::with_capacity(4096),
        offsets: Vec::with_capacity(8192),
        extras: Vec::with_capacity(1024),
        literals: Vec::with_capacity(MAX_BLOCK_SIZE + 64),
        table: new_table(),
    });
}

thread_local! {
    /// The previous v7 block's entropy tables. Reuse across blocks needs
    /// the blocks of a chain decoded in order on one thread: the
    /// sequential path does that, and the parallel path decodes each unit
    /// (a run of chained blocks) whole on one thread. Reset at the start
    /// of every call and unit (so a stream whose first block asks for
    /// reuse fails the same way on any thread) and at every
    /// FLAG_CHAIN_RESET block.
    static V7_TABLES: RefCell<v7_decode::DecTables<'static>> = RefCell::new(v7_decode::DecTables::none());
}

#[inline(always)]
pub(crate) fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("bmi2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Run the match finder over one block of `full_input` with parse `P`.
#[inline(always)]
fn find_block<P: Mode>(
    full_input: &[u8],
    block_start: usize,
    block_len: usize,
    table: &mut HashTable,
    tokens: &mut Vec<u8>,
    offsets: &mut Vec<u8>,
    extras: &mut Vec<u8>,
    literals: &mut Vec<u8>,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe {
                x86_compress::compress_chained_avx2::<P>(
                    full_input, block_start, block_len, table, tokens, offsets, extras, literals,
                );
            }
            return;
        }
    }
    fallback::compress_chained_fallback::<P>(
        full_input, block_start, block_len, table, tokens, offsets, extras, literals,
    );
}

/// The dense retry is opt-in (GLYD_DENSE=1). GOAL3 section 3: it bought
/// x-ray 1.08 for 23 ms of compression and 2.7 ms of decode, which the speed
/// target cannot afford; liblz4 gets 1.01 there at memcpy speed. The fast
/// level (GOAL3 Tier S3) is the intended replacement. Cached, never per block.
fn dense_enabled() -> bool {
    use std::sync::OnceLock;
    static D: OnceLock<bool> = OnceLock::new();
    *D.get_or_init(|| std::env::var("GLYD_DENSE").is_ok())
}

/// Streams would not save at least 4% of the chunk.
#[inline(always)]
fn not_worth_it(chunk_len: usize, tokens: &[u8], offsets: &[u8], extras: &[u8], literals: &[u8]) -> bool {
    tokens.len() + offsets.len() + extras.len() + literals.len() >= chunk_len - (chunk_len / 25)
}

/// Parse one block: the ordinary parse first, then the dense retry when it
/// would have been stored raw. Returns the flag bits describing the result.
#[inline(always)]
fn parse_block(
    full_input: &[u8],
    block_start: usize,
    block_len: usize,
    table: &mut HashTable,
    tokens: &mut Vec<u8>,
    offsets: &mut Vec<u8>,
    extras: &mut Vec<u8>,
    literals: &mut Vec<u8>,
) -> u16 {
    find_block::<Lzav>(full_input, block_start, block_len, table, tokens, offsets, extras, literals);
    if !not_worth_it(block_len, tokens, offsets, extras, literals) {
        return FLAG_COMPRESSED;
    }
    if !dense_enabled() {
        // Keep the ordinary parse whenever it beats a raw store at all.
        let payload = tokens.len() + offsets.len() + extras.len() + literals.len();
        return if payload < block_len { FLAG_COMPRESSED } else { FLAG_RAW_UNCOMPRESSED };
    }
    tokens.clear();
    offsets.clear();
    extras.clear();
    literals.clear();
    find_block::<Dense>(full_input, block_start, block_len, table, tokens, offsets, extras, literals);
    if not_worth_it(block_len, tokens, offsets, extras, literals) {
        return FLAG_RAW_UNCOMPRESSED;
    }
    FLAG_DENSE
}

/// Append one block (header plus payload) for `chunk`, given its streams and
/// the parse result flags (raw, ordinary or dense).
fn write_block(
    chunk: &[u8],
    parse_flags: u16,
    chain_flag: u16,
    tokens: &[u8],
    offsets: &[u8],
    extras: &[u8],
    literals: &[u8],
    output: &mut Vec<u8>,
) {
    let checksum = format::crc32c(chunk);

    if (parse_flags & FLAG_RAW_UNCOMPRESSED) != 0 {
        let header = BlockHeader {
            magic: MAGIC,
            version: CURRENT_VERSION,
            flags: FLAG_RAW_UNCOMPRESSED | FLAG_CRC32C | chain_flag,
            checksum,
            uncompressed_len: chunk.len() as u32,
            token_count: 0,
            token_bytes: 0,
            offset_bytes: 0,
            extras_bytes: 0,
            literal_len: chunk.len() as u32,
        };
        output.extend_from_slice(header_bytes(&header));
        output.extend_from_slice(chunk);
        return;
    }

    let header = BlockHeader {
        magic: MAGIC,
        version: CURRENT_VERSION,
        flags: parse_flags | FLAG_CRC32C | chain_flag,
        checksum,
        uncompressed_len: chunk.len() as u32,
        token_count: tokens.len() as u32,
        token_bytes: tokens.len() as u32,
        offset_bytes: offsets.len() as u32,
        extras_bytes: extras.len() as u32,
        literal_len: literals.len() as u32,
    };
    output.extend_from_slice(header_bytes(&header));
    output.extend_from_slice(tokens);
    output.extend_from_slice(offsets);
    output.extend_from_slice(extras);
    output.extend_from_slice(literals);
}

#[inline(always)]
fn header_bytes(header: &BlockHeader) -> &[u8] {
    unsafe { std::slice::from_raw_parts(header as *const BlockHeader as *const u8, HEADER_SIZE) }
}

/// Whether a coded block of `len` bytes takes the compact (v9) framing.
#[inline(always)]
fn compact_block(len: usize) -> bool {
    len <= COMPACT_MAX
}

/// Append a coded block: the v8 header and payload, or the v9 (compact)
/// ones for a small block.
fn write_coded_block(chunk: &[u8], n_seq: usize, n_lit: usize, chain_flag: u16, payload: &[u8], output: &mut Vec<u8>) {
    let compact = compact_block(chunk.len());
    let header = BlockHeader {
        magic: MAGIC,
        version: if compact { VERSION_V9 } else { VERSION_V8 },
        flags: FLAG_COMPRESSED | FLAG_CRC32C | FLAG_LL0_REP | chain_flag,
        checksum: format::crc32c(chunk),
        uncompressed_len: chunk.len() as u32,
        token_count: n_seq as u32,
        token_bytes: payload.len() as u32,
        offset_bytes: 0,
        extras_bytes: 0,
        literal_len: n_lit as u32,
    };
    if compact {
        header.write_compact(output);
    } else {
        output.extend_from_slice(header_bytes(&header));
    }
    output.extend_from_slice(payload);
}

/// Compress a single block (at most MAX_BLOCK_SIZE bytes) with no history
/// into a pre-allocated vector.
pub fn compress_block_into(chunk: &[u8], output: &mut Vec<u8>) {
    assert!(chunk.len() <= MAX_BLOCK_SIZE, "block exceeds MAX_BLOCK_SIZE");
    COMPRESS_SCRATCH.with(|scratch_cell| {
        let mut scratch = scratch_cell.borrow_mut();
        let CompressScratch {
            ref mut tokens,
            ref mut offsets,
            ref mut extras,
            ref mut literals,
            ref mut table,
        } = *scratch;

        tokens.clear();
        offsets.clear();
        extras.clear();
        literals.clear();
        finder::init_table(table, chunk);
        let flags = parse_block(chunk, 0, chunk.len(), table, tokens, offsets, extras, literals);
        write_block(chunk, flags, FLAG_CHAIN_RESET, tokens, offsets, extras, literals, output);
    });
}

/// Compress an arbitrary byte slice sequentially using a single core.
pub fn compress(input: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(input.len() / 2 + 1024);
    compress_into(input, &mut output);
    output
}

/// Compress an input slice into a destination vector with cross-block history lookback.
pub fn compress_into(input: &[u8], output: &mut Vec<u8>) {
    let mut table = new_table();
    finder::init_table(&mut table, input);
    let mut tokens = Vec::with_capacity(4096);
    let mut offsets = Vec::with_capacity(8192);
    let mut extras = Vec::with_capacity(1024);
    let mut literals = Vec::with_capacity(MAX_BLOCK_SIZE);

    let mut offset = 0;
    while offset < input.len() {
        let chunk_len = (input.len() - offset).min(MAX_BLOCK_SIZE);
        let chunk = &input[offset..offset + chunk_len];

        tokens.clear();
        offsets.clear();
        extras.clear();
        literals.clear();

        let flags = parse_block(input, offset, chunk_len, &mut table, &mut tokens, &mut offsets, &mut extras, &mut literals);

        let chain_flag = if offset == 0 { FLAG_CHAIN_RESET } else { 0 };
        write_block(chunk, flags, chain_flag, &tokens, &offsets, &extras, &literals, output);
        offset += chunk_len;
    }
}

/// Turbo level: the default finder at minimum match 8, emitted as
/// FLAG_TURBO blocks. Fewer tokens, so decode is ~18% faster than the
/// default at ~6% less ratio. Blocks the parse cannot shrink by 4% are
/// stored raw.
pub fn compress_into_turbo(input: &[u8], output: &mut Vec<u8>) {
    let mut table = new_table();
    finder::init_table(&mut table, input);
    let mut tokens = Vec::new();
    let mut offsets = Vec::new();
    let mut extras = Vec::new();
    let mut literals = Vec::new();

    let mut offset = 0;
    while offset < input.len() {
        let chunk_len = (input.len() - offset).min(MAX_BLOCK_SIZE);
        let chunk = &input[offset..offset + chunk_len];
        tokens.clear();
        offsets.clear();
        extras.clear();
        literals.clear();
        find_block::<Turbo>(input, offset, chunk_len, &mut table, &mut tokens, &mut offsets, &mut extras, &mut literals);
        let flags = if not_worth_it(chunk_len, &tokens, &offsets, &extras, &literals) {
            FLAG_RAW_UNCOMPRESSED
        } else {
            FLAG_TURBO
        };
        let chain_flag = if offset == 0 { FLAG_CHAIN_RESET } else { 0 };
        write_block(chunk, flags, chain_flag, &tokens, &offsets, &extras, &literals, output);
        offset += chunk_len;
    }
}

/// Turbo level, all cores.
pub fn compress_parallel_into_turbo(input: &[u8], output: &mut Vec<u8>) {
    compress_parallel_with(input, output, compress_into_turbo, PARALLEL_UNIT_V6)
}

/// Fast level (GOAL3 S3): LZ4-class finder, minimum match 5 (FLAG_DENSE
/// blocks), same container. Blocks the finder cannot shrink by 4% are
/// stored raw.
pub fn compress_into_fast(input: &[u8], output: &mut Vec<u8>) {
    let mut table: Box<finder::FastTable> =
        vec![0u32; finder::FAST_HASH_SIZE].into_boxed_slice().try_into().unwrap();
    let mut tokens = Vec::new();
    let mut offsets = Vec::new();
    let mut extras = Vec::new();
    let mut literals = Vec::new();

    let mut offset = 0;
    while offset < input.len() {
        let chunk_len = (input.len() - offset).min(MAX_BLOCK_SIZE);
        let chunk = &input[offset..offset + chunk_len];
        tokens.clear();
        offsets.clear();
        extras.clear();
        literals.clear();
        {
            let mut out = finder::Streams::new(Dense::MIN_MATCH, &mut tokens, &mut offsets, &mut extras, &mut literals);
            unsafe { finder::find_matches_fast::<Dense>(input, offset, chunk_len, &mut table, &mut out) };
        }
        let flags = if not_worth_it(chunk_len, &tokens, &offsets, &extras, &literals) {
            FLAG_RAW_UNCOMPRESSED
        } else {
            FLAG_DENSE
        };
        let chain_flag = if offset == 0 { FLAG_CHAIN_RESET } else { 0 };
        write_block(chunk, flags, chain_flag, &tokens, &offsets, &extras, &literals, output);
        offset += chunk_len;
    }
}

/// Compress across all CPU cores in parallel into a pre-allocated destination vector.
pub fn compress_parallel_into(input: &[u8], output: &mut Vec<u8>) {
    compress_parallel_with(input, output, compress_into, PARALLEL_UNIT_V6)
}

/// Fast level, all cores.
pub fn compress_parallel_into_fast(input: &[u8], output: &mut Vec<u8>) {
    compress_parallel_with(input, output, compress_into_fast, PARALLEL_UNIT_V6)
}

/// Max level: format v7 (entropy-coded sequences and literals) on the
/// double-fast parse, whose 2 MB window spans the blocks of one call.
/// Blocks the coder cannot shrink are stored raw (as v6 raw blocks,
/// which every decoder reads).
pub fn compress_into_max(input: &[u8], output: &mut Vec<u8>) {
    #[cfg(feature = "deflate")]
    if deflate::wrap(input, output, compress_into_max, compress_into_max) {
        return;
    }
    compress_max_plain(input, output)
}

/// The max level on the bytes as they are: no container opened.
pub(crate) fn compress_max_plain(input: &[u8], output: &mut Vec<u8>) {
    compress_max_from(input, 0, 0, Parse::Dfast, None, false, output, None)
}

/// Bytes a sequential max stream gathers before handing them to its
/// flush hook: a few blocks, so the writer thread stays busy.
const FLUSH_BYTES: usize = 1 << 20;

/// The max level (`long` as `compress_into_max_long`) with the output
/// handed to `sink` in pieces as the blocks are made, so that a caller
/// on one core can write while the parse goes on (the CLI does, on a
/// second thread). A container is compressed whole (its opened form is
/// judged against the closed one) and handed over at the end. The
/// bytes are exactly `compress_into_max`'s / `compress_into_max_long`'s.
pub fn compress_max_to(input: &[u8], long: bool, mut sink: impl FnMut(Vec<u8>)) {
    if deflate::is_container(input) && !in_part() {
        let mut out = Vec::new();
        if long { compress_into_max_long(input, &mut out) } else { compress_into_max(input, &mut out) }
        return sink(out);
    }
    let mut out = Vec::with_capacity(2 * FLUSH_BYTES);
    let mut flush = |o: &mut Vec<u8>| sink(std::mem::replace(o, Vec::with_capacity(2 * FLUSH_BYTES)));
    compress_max_from(input, 0, 0, Parse::Dfast, None, long, &mut out, Some(&mut flush));
    sink(out);
}

/// The max level with the long-distance matcher: repeats up to 128 MB
/// back (past the parse's 8 MB) found in a pass before the parse, as
/// `zstd --long`. Fewer bytes where content repeats across an input
/// (JSON events, logs), at the pass's cost in time; the same stream
/// format, so every decoder reads it.
pub fn compress_into_max_long(input: &[u8], output: &mut Vec<u8>) {
    #[cfg(feature = "deflate")]
    if deflate::wrap(input, output, compress_into_max_long, compress_into_max_long) {
        return;
    }
    compress_max_from(input, 0, 0, Parse::Dfast, None, true, output, None)
}

/// Ultra level: format v7 on the optimal parse (`v7_ultra`): the same
/// decoder and window, denser output, an order of magnitude slower to
/// produce.
pub fn compress_into_ultra(input: &[u8], output: &mut Vec<u8>) {
    #[cfg(feature = "deflate")]
    if deflate::wrap(input, output, compress_into_ultra, compress_into_ultra) {
        return;
    }
    compress_max_from(input, 0, 0, Parse::Ultra, None, true, output, None)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Parse {
    Dfast,
    Ultra,
}

/// The id a dictionary is named by in the blocks compressed with it: its
/// checksum, with 0 (no dictionary) reserved.
pub fn dict_id(dict: &[u8]) -> u32 {
    let id = compute_checksum(dict);
    if id == 0 { 1 } else { id }
}

/// Max level with a prepared dictionary (`Dict`): its content is the
/// window's first bytes, so matches may reach into it from the first
/// block on, and its entropy tables are the "previous block's" for the
/// first block, so an object that reuses them writes none. Every block
/// carries the dictionary's id and `decompress_with_dict` must be given
/// the same dictionary. Blocks stored raw carry no id. Sequential decode
/// only.
pub fn compress_with_dict(dict: &Dict, input: &[u8], output: &mut Vec<u8>) {
    // The input is parsed in place: the dictionary's content is history
    // outside it, found through the dictionary's own seeded tables.
    compress_max_from(input, 0, dict.id(), Parse::Dfast, Some(dict), false, output, None);
}

/// The max level over `full[start..]`, with `full[..start]` (a dictionary
/// named `dict_id`, or nothing) as history: indexed before the first
/// block, so the parse's window reaches into it.
fn compress_max_from(full: &[u8], start: usize, dict_id: u32, parse: Parse, dict: Option<&Dict>, long: bool, output: &mut Vec<u8>, mut flush: Option<&mut dyn FnMut(&mut Vec<u8>)>) {
    thread_local! {
        /// The parse's 2 MB of tables, allocated once per thread and
        /// cleared at the start of every call (the parallel path makes
        /// one per 256 KB chunk), so the output depends only on the
        /// input and the dictionary, not on what the thread compressed
        /// before (see `DfastTables::new`).
        static DFAST: RefCell<Box<v7_encode::DfastTables>> = RefCell::new(v7_encode::DfastTables::new());
        /// The ultra parse's 20 MB, likewise.
        static ULTRA: RefCell<Option<Box<v7_ultra::UltraState>>> = RefCell::new(None);
        /// The per-block buffers, kept across calls: a call on a small
        /// object would otherwise spend a fifth of its time growing them.
        static WORK: RefCell<Work> = RefCell::new(Work::default());
    }
    #[derive(Default)]
    struct Work {
        seqs: Vec<v7_encode::Sequence>,
        literals: Vec<u8>,
        scratch: v7_encode::EncScratch,
        payload: Vec<u8>,
        part_seqs: Vec<v7_encode::Sequence>,
        part_lits: Vec<u8>,
    }
    let mut work = WORK.with(|w| std::mem::take(&mut *w.borrow_mut()));
    let Work { seqs, literals, scratch, payload, part_seqs, part_lits } = &mut work;
    // A prepared dictionary's tables stand as the previous block's.
    let mut prev = dict.map_or_else(v7_encode::Tables::none, |d| d.tables());
    match parse {
        Parse::Dfast => DFAST.with_borrow_mut(|t| {
            // Tables sized to the input (a small object clears little);
            // `full[..start]` is a dictionary's content laid before the
            // input (the ultra path) and is indexed here, while the max
            // level reads a dictionary through the dictionary's own tables.
            t.clear_for(full.len());
            // A base region before the input is history the matcher
            // reaches; the tables take its last few megabytes only.
            t.seed_range(full, start.saturating_sub(BASE_SEED), start);
        }),
        Parse::Ultra => ULTRA.with_borrow_mut(|t| {
            let t = t.get_or_insert_with(v7_ultra::UltraState::new);
            // With a dictionary, an object of up to a block starts from
            // the dictionary's snapshot (its content indexed once) and
            // is priced by the dictionary's tables.
            match dict.map(|d| d.ultra_snapshot()).filter(|s| full.len() <= s.len) {
                Some(snap) => t.restore(snap),
                None => {
                    t.clear(full.len());
                    // A base region before the input: the tree reaches
                    // its last window only, the matcher pass the rest.
                    t.start_at(start.saturating_sub(LOCAL_WINDOW as usize));
                }
            }
            if let Some(d) = dict {
                t.seed_stats(&d.tables());
            }
        }),
    }
    // Far matches (past the local finders' 8 MB) over the whole input,
    // found once here when `long`; an input that fits the local window
    // has none. The max level gives the pass up on inputs with few far
    // repeats.
    let far = if long && full.len() > LOCAL_WINDOW as usize && dict.is_none() {
        ldm::Matches::find(full, matches!(parse, Parse::Dfast) && start == 0)
    } else {
        ldm::Matches { list: Vec::new() }
    };
    let mut offset = start;
    let mut first = true;
    let mut splitter = split::Splitter::new();
    // One block: the chunk at `offset`, its parse, and the header flags.
    let emit = |chunk: &[u8], seqs: &[v7_encode::Sequence], literals: &[u8], first: bool, prev: &mut v7_encode::Tables, scratch: &mut v7_encode::EncScratch, payload: &mut Vec<u8>, output: &mut Vec<u8>| {
        payload.clear();
        let compact = compact_block(chunk.len());
        v7_encode::encode_block_with(seqs, literals, dict_id, prev, scratch, compact, payload);
        // A dictionary stream's first block continues the dictionary's
        // window and tables: no reset.
        let chain_flag = if first && dict.is_none() { FLAG_CHAIN_RESET } else { 0 };
        if payload.len() + coded_header_len(compact, chunk.len(), payload.len(), seqs.len(), literals.len()) >= chunk.len() {
            *prev = v7_encode::Tables::none();
            write_block(chunk, FLAG_RAW_UNCOMPRESSED, chain_flag, &[], &[], &[], &[], output);
        } else {
            write_coded_block(chunk, seqs.len(), literals.len(), chain_flag, payload, output);
        }
    };
    while offset < full.len() {
        // The double-fast level cuts a block short where the bytes'
        // statistics change (`split::block_len`); the optimal parse
        // cuts on its sequences (`v7_ultra::split_points`).
        let chunk_len = match parse {
            Parse::Dfast => splitter.next(full, offset, MAX_BLOCK_SIZE),
            Parse::Ultra => (full.len() - offset).min(MAX_BLOCK_SIZE),
        };
        let chunk = &full[offset..offset + chunk_len];
        seqs.clear();
        literals.clear();
        let mut reps = [1u32, 4, 8]; // encode_block's Reps starts fresh per block
        match parse {
            Parse::Dfast => {
                DFAST.with_borrow_mut(|t| match dict {
                    Some(d) => v7_encode::find_sequences_dfast_dict(full, offset, chunk_len, t, d.finder(), &mut reps, seqs, literals, scratch),
                    None => v7_encode::find_sequences_dfast_far(full, offset, chunk_len, t, &far, &mut reps, seqs, literals, scratch),
                });
                // The dfast parse wrote its codes into the scratch as it
                // went; encode from those.
                payload.clear();
                let compact = compact_block(chunk_len);
                v7_encode::encode_block_coded(literals, dict_id, &mut prev, scratch, compact, payload);
                let chain_flag = if first && dict.is_none() { FLAG_CHAIN_RESET } else { 0 };
                let n_seq = scratch.count();
                if payload.len() + coded_header_len(compact, chunk_len, payload.len(), n_seq, literals.len()) >= chunk_len {
                    prev = v7_encode::Tables::none();
                    write_block(chunk, FLAG_RAW_UNCOMPRESSED, chain_flag, &[], &[], &[], &[], output);
                } else {
                    write_coded_block(chunk, n_seq, literals.len(), chain_flag, payload, output);
                }
                first = false;
                // Blocks so far handed on (the CLI's writer thread),
                // so the write overlaps the parse.
                if let Some(f) = flush.as_mut() {
                    if output.len() >= FLUSH_BYTES {
                        f(output);
                    }
                }
            }
            Parse::Ultra => {
                ULTRA.with_borrow_mut(|t| v7_ultra::find_sequences_ultra_far(full, offset, chunk_len, t.as_mut().unwrap(), reps, &far, seqs, literals));
                // The parse may be split into blocks with their own tables
                // where its statistics change (`v7_ultra::split_points`).
                let mut cuts = v7_ultra::split_points(seqs, literals);
                cuts.push(seqs.len());
                let (mut at, mut lit_at, mut pos) = (0usize, 0usize, 0usize);
                for cut in cuts {
                    part_seqs.clear();
                    part_seqs.extend_from_slice(&seqs[at..cut]);
                    let lit_end = lit_at + part_seqs.iter().map(|q| q.lit_len as usize).sum::<usize>();
                    let len: usize = part_seqs.iter().map(|q| (q.lit_len + q.match_len) as usize).sum();
                    if part_seqs.last().map_or(true, |q| q.match_len != 0) {
                        part_seqs.push(v7_encode::Sequence { lit_len: 0, match_len: 0, offset: 0 });
                    }
                    part_lits.clear();
                    part_lits.extend_from_slice(&literals[lit_at..lit_end]);
                    emit(&chunk[pos..pos + len], part_seqs, part_lits, first, &mut prev, scratch, payload, output);
                    first = false;
                    at = cut;
                    lit_at = lit_end;
                    pos += len;
                }
                debug_assert_eq!(pos, chunk_len);
            }
        }
        offset += chunk_len;
    }
    WORK.with(|w| *w.borrow_mut() = work);
}

/// Max level, all cores: units of a core's share (at least 8 MB), so
/// that reads scale across cores as writes do.
pub fn compress_parallel_into_max(input: &[u8], output: &mut Vec<u8>) {
    #[cfg(feature = "deflate")]
    if deflate::wrap(input, output, compress_parallel_into_max, compress_parallel_into_max) {
        return;
    }
    compress_parallel_with(input, output, compress_into_max, PARALLEL_UNIT_MAX)
}

/// `compress_into_max_long` on all cores (units as `compress_parallel_into_max`).
pub fn compress_parallel_into_max_long(input: &[u8], output: &mut Vec<u8>) {
    #[cfg(feature = "deflate")]
    if deflate::wrap(input, output, compress_parallel_into_max_long, compress_parallel_into_max_long) {
        return;
    }
    compress_parallel_with(input, output, compress_into_max_long, PARALLEL_UNIT_MAX)
}

/// Max level, all cores, dense: units of `PARALLEL_UNIT_LARGEST` (the
/// far matcher's reach, the long search on), each parsed in stripes on every core, so the
/// bytes are one core's whatever the core count — 5–9% fewer than
/// `compress_parallel_into_max` on files of a few hundred MB, the
/// same on files of gigabytes — at the price of reads that scale only
/// with the units (a 512 MB file decodes on 4 cores, not 8). For
/// objects written once and read rarely: the store's.
pub fn compress_into_max_dense(input: &[u8], output: &mut Vec<u8>) {
    #[cfg(feature = "deflate")]
    if deflate::wrap(input, output, compress_into_max_dense, compress_into_max_dense) {
        return;
    }
    if input.len() <= PARALLEL_UNIT_MAX || threads() == 1 {
        return as_part(|| compress_into_max_long(input, output));
    }
    let _ = compress_max_stream(input, |part| {
        output.extend_from_slice(part);
        Ok(())
    });
}

/// How far back a stripe's tables are seeded: 2 MB reproduces the
/// sequential parse to 0.03% (8 MB, the local finder's whole reach,
/// to 0.0%, at a fifth less speed).
const STRIPE_SEED: usize = 2 << 20;
/// Bytes per stripe: enough that seeding is a small part of the work.
const STRIPE: usize = 16 << 20;

/// The max level over all cores with the ratio of one core, its output
/// handed to `sink` in order as it is made. Units of
/// `PARALLEL_UNIT_LARGEST` are streams of their own (a chain reset at
/// the first block), as `compress_parallel_with` makes them, but the
/// units are never shrunk to give every core one: a unit's far matches
/// are found by one core (one unit ahead of the parse) and its blocks
/// are then parsed in `STRIPE`-sized stripes by all of them, each
/// stripe's tables seeded with the `STRIPE_SEED` bytes before it. The
/// output is within a percent of the sequential level's, whatever the
/// core count, where units of a tenth the size lost 5-14%.
pub fn compress_max_stream(input: &[u8], mut sink: impl FnMut(&[u8]) -> std::io::Result<()> + Send) -> std::io::Result<()> {
    thread_local! {
        static DFAST: RefCell<Box<v7_encode::DfastTables>> = RefCell::new(v7_encode::DfastTables::new());
        static WORK: RefCell<(Vec<v7_encode::Sequence>, Vec<u8>, v7_encode::EncScratch, Vec<u8>)> = RefCell::new(Default::default());
    }
    let units: Vec<&[u8]> = if input.is_empty() { vec![input] } else { input.chunks(PARALLEL_UNIT_LARGEST).collect() };
    let fars: Vec<std::sync::OnceLock<ldm::Matches>> = units.iter().map(|_| std::sync::OnceLock::new()).collect();
    // Tasks in the order threads take them: every unit's far pass
    // first (one core each, the rest start on the first unit's stripes
    // as soon as its matches are in), then the stripes in order.
    #[derive(Clone, Copy)]
    enum Task {
        Far(usize),
        Stripe { unit: usize, from: usize, to: usize, index: usize },
    }
    let mut tasks: Vec<Task> = (0..units.len()).map(Task::Far).collect();
    let mut stripe_count = 0usize;
    for (u, unit) in units.iter().enumerate() {
        let mut from = 0usize;
        loop {
            let to = (from + STRIPE).min(unit.len());
            tasks.push(Task::Stripe { unit: u, from, to, index: stripe_count });
            stripe_count += 1;
            if to >= unit.len() {
                break;
            }
            from = to;
        }
    }
    // Stripe i's output goes into slot i; the writer takes slots in
    // order as they fill.
    let slots: Vec<std::sync::Mutex<Option<Vec<u8>>>> = (0..stripe_count).map(|_| std::sync::Mutex::new(None)).collect();
    let ready = std::sync::Condvar::new();
    let done = std::sync::Mutex::new(0usize);
    let error: std::sync::Mutex<Option<std::io::Error>> = std::sync::Mutex::new(None);
    let far_of = |u: usize| -> &ldm::Matches {
        // A stripe whose unit's matches are not in yet finds them
        // itself (never on the thread order above; harmless if so).
        fars[u].get_or_init(|| if units[u].len() > LOCAL_WINDOW as usize { ldm::Matches::find(units[u], true) } else { ldm::Matches { list: Vec::new() } })
    };
    std::thread::scope(|scope| {
        let writer = scope.spawn(|| {
            for slot in slots.iter() {
                let out = loop {
                    if let Some(out) = slot.lock().unwrap().take() {
                        break out;
                    }
                    let guard = done.lock().unwrap();
                    let _guard = ready.wait(guard).unwrap();
                };
                if let Err(e) = sink(&out) {
                    *error.lock().unwrap() = Some(e);
                    return;
                }
            }
        });
        let _ = par_units::<()>(tasks.len(), |i| {
            match tasks[i] {
                Task::Far(u) => {
                    far_of(u);
                }
                Task::Stripe { unit: u, from, to, index } => {
                    let unit = units[u];
                    let far = far_of(u);
                    let mut out = Vec::with_capacity((to - from) / 2 + 1024);
                    WORK.with_borrow_mut(|(seqs, literals, scratch, payload)| {
                        DFAST.with_borrow_mut(|t| {
                            t.clear_for(unit.len());
                            t.seed_range(unit, from.saturating_sub(STRIPE_SEED), from);
                            let mut prev = v7_encode::Tables::none();
                            let mut offset = from;
                            let mut splitter = split::Splitter::new();
                            while offset < to {
                                let chunk_len = splitter.next(&unit[..to], offset, MAX_BLOCK_SIZE);
                                let chunk = &unit[offset..offset + chunk_len];
                                seqs.clear();
                                literals.clear();
                                let mut reps = [1u32, 4, 8];
                                v7_encode::find_sequences_dfast_far(unit, offset, chunk_len, t, far, &mut reps, seqs, literals, scratch);
                                payload.clear();
                                let compact = compact_block(chunk_len);
                                v7_encode::encode_block_coded(literals, 0, &mut prev, scratch, compact, payload);
                                let chain_flag = if offset == 0 { FLAG_CHAIN_RESET } else { 0 };
                                let n_seq = scratch.count();
                                if payload.len() + coded_header_len(compact, chunk_len, payload.len(), n_seq, literals.len()) >= chunk_len {
                                    prev = v7_encode::Tables::none();
                                    write_block(chunk, FLAG_RAW_UNCOMPRESSED, chain_flag, &[], &[], &[], &[], &mut out);
                                } else {
                                    write_coded_block(chunk, n_seq, literals.len(), chain_flag, payload, &mut out);
                                }
                                offset += chunk_len;
                            }
                        });
                    });
                    *slots[index].lock().unwrap() = Some(out);
                    let mut d = done.lock().unwrap();
                    *d += 1;
                    ready.notify_all();
                }
            }
            Ok(())
        });
        let _ = writer.join();
    });
    match error.into_inner().unwrap() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Ultra level, all cores: 16 MB units parsed on their own (0.7% less
/// dense than the sequential level on Silesia; the window does not reach
/// across units, and each decodes on its own).
pub fn compress_parallel_into_ultra(input: &[u8], output: &mut Vec<u8>) {
    #[cfg(feature = "deflate")]
    if deflate::wrap(input, output, compress_parallel_into_ultra, compress_parallel_into_ultra) {
        return;
    }
    compress_parallel_with(input, output, compress_into_ultra, PARALLEL_UNIT_ULTRA)
}

/// Ultra level with a prepared dictionary; `decompress_with_dict` reads it.
pub fn compress_with_dict_ultra(dict: &Dict, input: &[u8], output: &mut Vec<u8>) {
    let mut joined = Vec::with_capacity(dict.content().len() + input.len());
    joined.extend_from_slice(dict.content());
    joined.extend_from_slice(input);
    compress_max_from(&joined, dict.content().len(), dict.id(), Parse::Ultra, Some(dict), false, output, None);
}

/// `level` over units of at least `smallest` bytes on all cores
/// (`format::parallel_unit`), each unit a chain of its own. An input of
/// one unit or less is compressed sequentially.
fn compress_parallel_with(input: &[u8], output: &mut Vec<u8>, level: fn(&[u8], &mut Vec<u8>), smallest: usize) {
    let unit = parallel_unit(input.len(), threads(), smallest);
    if input.len() <= unit {
        as_part(|| level(input, output));
        return;
    }

    let chunks: Vec<&[u8]> = input.chunks(unit).collect();
    let compressed_chunks: Vec<std::sync::Mutex<Vec<u8>>> = chunks.iter().map(|_| std::sync::Mutex::new(Vec::new())).collect();
    let _ = par_units::<()>(chunks.len(), |i| {
        let mut chunk_out = Vec::with_capacity(chunks[i].len() / 2 + 1024);
        as_part(|| level(chunks[i], &mut chunk_out));
        *compressed_chunks[i].lock().unwrap() = chunk_out;
        Ok(())
    });

    let total_len: usize = compressed_chunks.iter().map(|c| c.lock().unwrap().len()).sum();
    output.reserve(total_len);
    for chunk in compressed_chunks {
        output.extend_from_slice(&chunk.into_inner().unwrap());
    }
}

/// `compress_parallel_with`, the units handed to `sink` in order as
/// each finishes (from a writer thread, so writing overlaps the
/// compressing) instead of gathered into one buffer: the CLI's memory
/// is a few units of output, not the file's. `level` is a sequential
/// level (`compress_into_max`, ...).
pub fn compress_stream(input: &[u8], level: fn(&[u8], &mut Vec<u8>), smallest: usize, mut sink: impl FnMut(&[u8]) -> std::io::Result<()> + Send) -> std::io::Result<()> {
    let unit = parallel_unit(input.len(), threads(), smallest);
    let chunks: Vec<&[u8]> = if input.is_empty() { vec![input] } else { input.chunks(unit).collect() };
    let n = chunks.len();
    // Unit i's output goes into slot i; the writer takes slots in order
    // as they fill, waiting on the condition variable.
    let slots: Vec<std::sync::Mutex<Option<Vec<u8>>>> = (0..n).map(|_| std::sync::Mutex::new(None)).collect();
    let ready = std::sync::Condvar::new();
    let done = std::sync::Mutex::new(0usize);
    let error: std::sync::Mutex<Option<std::io::Error>> = std::sync::Mutex::new(None);
    std::thread::scope(|scope| {
        let writer = scope.spawn(|| {
            for slot in slots.iter() {
                let out = loop {
                    if let Some(out) = slot.lock().unwrap().take() {
                        break out;
                    }
                    let guard = done.lock().unwrap();
                    let _guard = ready.wait(guard).unwrap();
                };
                if let Err(e) = sink(&out) {
                    *error.lock().unwrap() = Some(e);
                    return;
                }
            }
        });
        let _ = par_units::<()>(n, |i| {
            let mut out = Vec::with_capacity(chunks[i].len() / 2 + 1024);
            as_part(|| level(chunks[i], &mut out));
            *slots[i].lock().unwrap() = Some(out);
            let mut d = done.lock().unwrap();
            *d += 1;
            ready.notify_all();
            Ok(())
        });
        let _ = writer.join();
    });
    match error.into_inner().unwrap() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Compress across all CPU cores in parallel using Rayon.
pub fn compress_parallel(input: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(input.len() / 2 + 1024);
    compress_parallel_into(input, &mut output);
    output
}

/// Read and validate the block header at `cursor`. Returns the header and
/// the cursor of the next block. Nothing in the payload is touched.
#[inline(always)]
fn parse_header(compressed: &[u8], cursor: usize) -> Result<(BlockHeader, usize)> {
    parse_header_at(compressed, cursor).map(|(h, _, next)| (h, next))
}

/// `parse_header`, also returning where the payload starts (the header's
/// length depends on the block's version: 19 bytes for v9, 32 otherwise).
/// `cursor + COMPACT_HEADER_MIN <= compressed.len()` is the caller's check.
#[inline(always)]
fn parse_header_at(compressed: &[u8], cursor: usize) -> Result<(BlockHeader, usize, usize)> {
    let head = &compressed[cursor..];
    let (header, used) = match BlockHeader::read(head) {
        Some(h) => h,
        None if head[0] != COMPACT_MARKER && head.len() >= 4 && u32::from_le_bytes([head[0], head[1], head[2], head[3]]) != MAGIC => return Err(CodecError::InvalidMagic),
        None => return Err(CodecError::CorruptedBitstream("Truncated block header")),
    };
    if header.version != CURRENT_VERSION && !is_coded_version(header.version) {
        return Err(CodecError::UnsupportedVersion(header.version));
    }
    if !header.is_plausible() {
        return Err(CodecError::CorruptedBitstream("Implausible block header"));
    }
    let start = cursor + used;
    let next = start + header.payload_len();
    if next > compressed.len() {
        return Err(CodecError::CorruptedBitstream("Truncated compressed block payload"));
    }
    Ok((header, start, next))
}

/// Decode one block's payload into `dst` (the block's output region plus
/// padding), with the match window starting at `buffer_start`.
#[inline(always)]
unsafe fn decode_block(
    header: &BlockHeader,
    payload: &[u8],
    dst: &mut [u8],
    buffer_start: *const u8,
    avx2: bool,
    ext: Option<&[u8]>,
    tables: Option<&mut v7_decode::DecTables<'_>>,
) -> Result<()> {
    let uncomp_len = header.uncompressed_len as usize;
    if uncomp_len > dst.len() {
        return Err(CodecError::OutputBufferTooSmall { required: uncomp_len, provided: dst.len() });
    }
    if (header.flags & FLAG_RAW_UNCOMPRESSED) != 0 {
        dst[..uncomp_len].copy_from_slice(&payload[..uncomp_len]);
        return Ok(());
    }
    if is_coded_version(header.version) {
        let v8 = header.version >= VERSION_V8;
        let compact = header.version == VERSION_V9;
        // `payload` may run past the block (see the callers); the block's
        // own bytes, plus for a compact block the readable bytes after
        // them that stand in for its padding.
        let len = header.payload_len();
        let prepadded = compact && payload.len() >= len + bits::PAD;
        let payload = if prepadded { &payload[..len + bits::PAD] } else { &payload[..len] };
        let (n_seq, n_lit) = (header.token_count as usize, header.literal_len as usize);
        let ll0 = header.flags & FLAG_LL0_REP != 0;
        return v7_decode::with_scratch(|scratch| {
            // A dictionary stream's tables are the caller's (borrowing the
            // dictionary's); otherwise the thread's, reset at a chain start.
            if let Some(t) = tables {
                return v7_decode::decode_block(payload, v8, compact, prepadded, n_seq, n_lit, dst, buffer_start, uncomp_len, t, scratch, ext, ll0).map(|_| ());
            }
            V7_TABLES.with(|t| {
                let mut t = t.borrow_mut();
                if (header.flags & FLAG_CHAIN_RESET) != 0 {
                    *t = v7_decode::DecTables::none();
                }
                v7_decode::decode_block(payload, v8, compact, prepadded, n_seq, n_lit, dst, buffer_start, uncomp_len, &mut t, scratch, ext, ll0).map(|_| ())
            })
        });
    }
    if (header.flags & FLAG_HUFF_TOKENS) != 0 {
        return Err(CodecError::CorruptedBitstream("Huffman token blocks not supported yet"));
    }
    let token_count = header.token_count as usize;
    let token_bytes = header.token_bytes as usize;
    if token_bytes != token_count {
        return Err(CodecError::CorruptedBitstream("Token section length mismatch"));
    }
    let mut c = 0usize;
    let tokens = payload.as_ptr().add(c);
    c += token_bytes;
    let offsets = payload.as_ptr().add(c);
    let offsets_len = header.offset_bytes as usize;
    c += offsets_len;
    let extras = payload.as_ptr().add(c);
    let extras_len = header.extras_bytes as usize;
    c += extras_len;
    let literals = &payload[c..c + header.literal_len as usize];
    let dense = (header.flags & FLAG_DENSE) != 0;
    let turbo = (header.flags & FLAG_TURBO) != 0;
    let min_match = if dense { MIN_MATCH_LEN_DENSE } else if turbo { MIN_MATCH_LEN_TURBO } else { MIN_MATCH_LEN };
    // Each machine takes its own decoder below; the others' inputs go unused.
    let _ = (avx2, min_match);
    let (table, esc): (&[u32; 256], usize) = if dense {
        (&TOKEN_TABLE_DENSE, ESCAPE_BASE_MATCH_DENSE)
    } else if turbo {
        (&TOKEN_TABLE_TURBO, ESCAPE_BASE_MATCH_TURBO)
    } else {
        (&TOKEN_TABLE, ESCAPE_BASE_MATCH)
    };

    #[cfg(target_arch = "x86_64")]
    {
        if avx2 {
            x86_decompress::decompress_avx2(
                tokens, token_count, offsets, offsets_len, extras, extras_len,
                literals, dst, buffer_start, uncomp_len, table, esc,
            )?;
            return Ok(());
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        neon_decompress::decompress_neon(
            tokens, token_count, offsets, offsets_len, extras, extras_len,
            payload.len() - (c - extras_len), literals, dst, buffer_start, uncomp_len, table, esc,
        )?;
        return Ok(());
    }
    #[allow(unreachable_code)]
    let _ = (table, esc);
    #[allow(unreachable_code)]
    fallback::decompress_fallback_raw(
        tokens, token_count, offsets, offsets_len, extras, extras_len,
        literals, dst, buffer_start, uncomp_len, min_match,
    )?;
    Ok(())
}

/// The decompressed size of `compressed`, from its block headers (every
/// header is validated; the payloads are not read).
pub fn decompressed_len(compressed: &[u8]) -> Result<usize> {
    #[cfg(feature = "deflate")]
    if let Some(original) = deflate::original_len(compressed) {
        return Ok(original);
    }
    if let Some((original, _)) = jpeg::parse(compressed) {
        return Ok(original);
    }
    if let Some(units) = records_envelope(compressed) {
        return Ok(units.iter().map(|u| u.len).sum());
    }
    if let Some(units) = cold_envelope(compressed) {
        return Ok(units.iter().map(|u| u.len).sum());
    }
    if is_pack(compressed) {
        return Ok(pack_envelope(compressed)?.0.iter().sum());
    }
    if let Some((_, units)) = base_envelope(compressed) {
        return Ok(units.iter().map(|u| u.len).sum());
    }
    #[cfg(feature = "deflate")]
    if let Some(r) = embedded_envelopes(compressed) {
        return r.map(|v| v.len());
    }
    total_uncompressed_len(compressed)
}

// ---------------------------------------------------------------------------
// Packs: many small objects compressed as one stream — record mode where
// it pays, so a thousand events cost what they cost as a file, not as a
// thousand objects — with an index of their lengths, so any one of them
// is the pack decoded and sliced. Envelope:
//
//   "GLYDPACK" n_objects, index_len (varints), the index (the lengths
//   as zigzag deltas, compressed at the max level), then the objects'
//   concatenation compressed by `level` in record mode.
// ---------------------------------------------------------------------------

const PACK_MAGIC: &[u8; 8] = b"GLYDPACK";

/// `objects` compressed together at `level` (a sequential level:
/// `compress_into_max`, `compress_into_ultra`, `compress_into_cold`),
/// record mode where the transform pays. Packs of 1-4 MB decode in
/// under a millisecond per megabyte; `decompress_pack_object` is the
/// pack decoded and one object sliced out.
pub fn compress_pack(objects: &[&[u8]], output: &mut Vec<u8>, level: fn(&[u8], &mut Vec<u8>)) {
    let total: usize = objects.iter().map(|o| o.len()).sum();
    let mut joined = Vec::with_capacity(total);
    let mut lengths = Vec::with_capacity(objects.len() * 2);
    let mut last = 0i64;
    for o in objects {
        joined.extend_from_slice(o);
        record::put_varint(&mut lengths, record::zigzag(o.len() as i64 - last));
        last = o.len() as i64;
    }
    let mut index = Vec::new();
    compress_into_max(&lengths, &mut index);
    output.extend_from_slice(PACK_MAGIC);
    put_varint(output, objects.len() as u32);
    put_varint(output, index.len() as u32);
    output.extend_from_slice(&index);
    as_part(|| records_with(&joined, output, level, PARALLEL_UNIT_MAX));
}

/// The objects' lengths and the payload of a pack.
fn pack_envelope(compressed: &[u8]) -> Result<(Vec<usize>, &[u8])> {
    if compressed.len() < 10 || &compressed[..8] != PACK_MAGIC {
        return Err(CodecError::CorruptedBitstream("not a pack"));
    }
    let mut pos = 8usize;
    let n = get_varint(compressed, &mut pos).ok_or(CodecError::CorruptedBitstream("pack: count"))? as usize;
    let index_len = get_varint(compressed, &mut pos).ok_or(CodecError::CorruptedBitstream("pack: index"))? as usize;
    let index = compressed.get(pos..pos.checked_add(index_len).ok_or(CodecError::CorruptedBitstream("pack: index"))?).ok_or(CodecError::CorruptedBitstream("pack: index"))?;
    pos += index_len;
    if n > index.len() * 64 + 64 {
        return Err(CodecError::CorruptedBitstream("pack: count"));
    }
    let deltas = decompress(index)?;
    let mut lengths = Vec::with_capacity(n);
    let (mut at, mut last) = (0usize, 0i64);
    for _ in 0..n {
        let d = record::get_varint(&deltas, &mut at)?;
        last = last.wrapping_add(record::unzigzag(d));
        if last < 0 {
            return Err(CodecError::CorruptedBitstream("pack: length"));
        }
        lengths.push(last as usize);
    }
    Ok((lengths, &compressed[pos..]))
}

/// Whether `compressed` is a pack.
pub fn is_pack(compressed: &[u8]) -> bool {
    compressed.len() >= 10 && &compressed[..8] == PACK_MAGIC
}

/// A pack's objects back to back (what `decompress` returns for one).
fn decompress_pack_joined(compressed: &[u8]) -> Result<Vec<u8>> {
    let (lengths, stream) = pack_envelope(compressed)?;
    let all = decompress(stream)?;
    if lengths.iter().sum::<usize>() != all.len() {
        return Err(CodecError::CorruptedBitstream("pack: lengths"));
    }
    Ok(all)
}

/// Every object of a pack.
pub fn decompress_pack(compressed: &[u8]) -> Result<Vec<Vec<u8>>> {
    let (lengths, stream) = pack_envelope(compressed)?;
    let all = decompress(stream)?;
    if lengths.iter().sum::<usize>() != all.len() {
        return Err(CodecError::CorruptedBitstream("pack: lengths"));
    }
    let mut at = 0usize;
    Ok(lengths
        .iter()
        .map(|&n| {
            at += n;
            all[at - n..at].to_vec()
        })
        .collect())
}

/// Object `i` of a pack: the pack decoded and sliced.
pub fn decompress_pack_object(compressed: &[u8], i: usize) -> Result<Vec<u8>> {
    let (lengths, stream) = pack_envelope(compressed)?;
    if i >= lengths.len() {
        return Err(CodecError::CorruptedBitstream("pack: no such object"));
    }
    let all = decompress(stream)?;
    let start: usize = lengths[..i].iter().sum();
    let end = start + lengths[i];
    if end > all.len() {
        return Err(CodecError::CorruptedBitstream("pack: lengths"));
    }
    Ok(all[start..end].to_vec())
}

/// The object count of a pack.
pub fn pack_len(compressed: &[u8]) -> Result<usize> {
    Ok(pack_envelope(compressed)?.0.len())
}

// ---------------------------------------------------------------------------
// The cold level: context mixing (`cm`) over 32 MB units, each coded
// from an empty model so that units decode in parallel. A unit's
// stream has no framing of its own; the envelope carries its length and
// checksum, and a decoder that does not get the checksum back reports
// corruption. Envelope:
//
//   "GLYDCOLD" n_units, then per unit: len, stream_len (varints),
//   checksum (u32 LE); then the units' streams back to back.
// ---------------------------------------------------------------------------

const COLD_MAGIC: &[u8; 8] = b"GLYDCOLD";
const COLD_UNIT: usize = 32 << 20;

struct ColdUnit<'a> {
    len: usize,
    checksum: u32,
    stream: &'a [u8],
}

fn cold_envelope(compressed: &[u8]) -> Option<Vec<ColdUnit<'_>>> {
    if compressed.len() < 9 || &compressed[..8] != COLD_MAGIC {
        return None;
    }
    let mut pos = 8usize;
    let n = get_varint(compressed, &mut pos)? as usize;
    if n > compressed.len() {
        return None;
    }
    let mut heads = Vec::with_capacity(n);
    for _ in 0..n {
        let len = get_varint(compressed, &mut pos)? as usize;
        let slen = get_varint(compressed, &mut pos)? as usize;
        let checksum = u32::from_le_bytes(compressed.get(pos..pos + 4)?.try_into().unwrap());
        pos += 4;
        if len > COLD_UNIT {
            return None;
        }
        heads.push((len, slen, checksum));
    }
    let mut units = Vec::with_capacity(n);
    for (len, slen, checksum) in heads {
        let stream = compressed.get(pos..pos.checked_add(slen)?)?;
        pos += slen;
        units.push(ColdUnit { len, checksum, stream });
    }
    if pos != compressed.len() {
        return None;
    }
    Some(units)
}

fn cold_units(input: &[u8]) -> Vec<&[u8]> {
    if input.is_empty() {
        return vec![input];
    }
    input.chunks(COLD_UNIT).collect()
}

fn write_cold(output: &mut Vec<u8>, units: &[&[u8]], streams: &[Vec<u8>]) {
    output.extend_from_slice(COLD_MAGIC);
    put_varint(output, units.len() as u32);
    for (u, c) in units.iter().zip(streams) {
        put_varint(output, u.len() as u32);
        put_varint(output, c.len() as u32);
        output.extend_from_slice(&compute_checksum(u).to_le_bytes());
    }
    for c in streams {
        output.extend_from_slice(c);
    }
}

/// The cold level: context mixing, the smallest output and 1-2 MB/s
/// each way on one core; for what is stored for years and read rarely.
/// Units of 32 MB coded one after the other; `compress_parallel_into_cold`
/// codes them on all cores. Every decoder reads the result.
pub fn compress_into_cold(input: &[u8], output: &mut Vec<u8>) {
    #[cfg(feature = "deflate")]
    if deflate::wrap(input, output, compress_into_cold, compress_into_cold) {
        return;
    }
    let units = cold_units(input);
    let streams: Vec<Vec<u8>> = units
        .iter()
        .map(|u| {
            let mut c = Vec::with_capacity(u.len() / 4 + 64);
            cm::encode(u, &mut c);
            c
        })
        .collect();
    write_cold(output, &units, &streams);
}

/// The cold level, all cores (a thread holds 400 MB of model and unit).
pub fn compress_parallel_into_cold(input: &[u8], output: &mut Vec<u8>) {
    #[cfg(feature = "deflate")]
    if deflate::wrap(input, output, compress_parallel_into_cold, compress_parallel_into_cold) {
        return;
    }
    let units = cold_units(input);
    let slots: Vec<std::sync::Mutex<Vec<u8>>> = units.iter().map(|_| std::sync::Mutex::new(Vec::new())).collect();
    let _ = par_units::<()>(units.len(), |i| {
        let mut c = Vec::with_capacity(units[i].len() / 4 + 64);
        cm::encode(units[i], &mut c);
        *slots[i].lock().unwrap() = c;
        Ok(())
    });
    let streams: Vec<Vec<u8>> = slots.into_iter().map(|m| m.into_inner().unwrap()).collect();
    write_cold(output, &units, &streams);
}

/// The cold level in record mode: the typed columns, then context mixing.
pub fn compress_records_into_cold(input: &[u8], output: &mut Vec<u8>) {
    #[cfg(feature = "deflate")]
    if deflate::wrap(input, output, compress_records_into_cold, compress_records_into_cold) {
        return;
    }
    if !records_pay(records_trial(input), compress_into_cold) {
        return as_part(|| compress_parallel_into_cold(input, output));
    }
    records_units(input, output, compress_into_cold)
}

/// The units of a cold envelope decoded into `dst` in parallel, each
/// checked against its checksum.
fn cold_into(units: &[ColdUnit<'_>], dst: &mut [u8]) -> Result<usize> {
    let total: usize = units.iter().map(|u| u.len).sum();
    if dst.len() < total {
        return Err(CodecError::OutputBufferTooSmall { required: total, provided: dst.len() });
    }
    let mut offsets = Vec::with_capacity(units.len());
    let mut at = 0usize;
    for u in units {
        offsets.push(at);
        at += u.len;
    }
    let base = dst.as_mut_ptr() as usize;
    par_units(units.len(), |i| -> Result<()> {
        let (u, off) = (&units[i], offsets[i]);
        // Units cover disjoint ranges of `dst`.
        let out = unsafe { std::slice::from_raw_parts_mut((base + off) as *mut u8, u.len) };
        cm::decode(u.stream, out);
        if compute_checksum(out) != u.checksum {
            return Err(CodecError::CorruptedBitstream("cold envelope: checksum"));
        }
        Ok(())
    })?;
    Ok(total)
}

fn decompress_cold(units: &[ColdUnit<'_>]) -> Result<Vec<u8>> {
    let total: usize = units.iter().map(|u| u.len).sum();
    let mut out = vec![0u8; total];
    cold_into(units, &mut out)?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// Base (delta) mode: a new version of an object compressed against the
// old one. The input is cut into `BASE_UNIT`s; each is parsed with a
// region of the base laid before it as history (the region around the
// unit's own position, `BASE_SLACK` each way, so a version whose
// content has drifted by less than that finds it), the long-distance
// matcher reaching all of it and the local finder its last megabytes.
// The decoder reads the region in place. Envelope:
//
//   "GLYDBASE" base id (u64 LE: the base's length and checksum)
//   n_units, then per unit: base_start, base_len, new_len, stream_len
//   (varints), then the units' streams back to back.
// ---------------------------------------------------------------------------

const BASE_MAGIC: &[u8; 8] = b"GLYDBASE";
const BASE_UNIT: usize = 32 << 20;
const BASE_SLACK: usize = 32 << 20;
/// A region ends this far before the base's end: the decoder's copies
/// read a little past it.
const BASE_TAIL: usize = 64;
/// Bytes of the region before the input the local finder is seeded with.
const BASE_SEED: usize = 8 << 20;

/// What identifies a base: its length and checksum.
pub fn base_id(base: &[u8]) -> u64 {
    (base.len() as u64) << 32 | compute_checksum(base) as u64
}

struct BaseUnit<'a> {
    base_start: usize,
    base_len: usize,
    len: usize,
    stream: &'a [u8],
}

fn base_envelope(compressed: &[u8]) -> Option<(u64, Vec<BaseUnit<'_>>)> {
    if compressed.len() < 16 || &compressed[..8] != BASE_MAGIC {
        return None;
    }
    let id = u64::from_le_bytes(compressed[8..16].try_into().unwrap());
    let mut pos = 16usize;
    let n = get_varint64(compressed, &mut pos)? as usize;
    if n > compressed.len() {
        return None;
    }
    let mut heads = Vec::with_capacity(n);
    for _ in 0..n {
        let base_start = get_varint64(compressed, &mut pos)? as usize;
        let base_len = get_varint64(compressed, &mut pos)? as usize;
        let len = get_varint64(compressed, &mut pos)? as usize;
        let slen = get_varint64(compressed, &mut pos)? as usize;
        // What the encoder writes; a corrupt header cannot ask for more.
        if len > BASE_UNIT || base_len > BASE_UNIT + 2 * BASE_SLACK {
            return None;
        }
        heads.push((base_start, base_len, len, slen));
    }
    let mut units = Vec::with_capacity(n);
    for (base_start, base_len, len, slen) in heads {
        let stream = compressed.get(pos..pos.checked_add(slen)?)?;
        pos += slen;
        units.push(BaseUnit { base_start, base_len, len, stream });
    }
    if pos != compressed.len() {
        return None;
    }
    Some((id, units))
}

/// The base's sparse anchors (`ldm::sparse_anchors`) sorted by hash: a
/// unit's own anchors looked up here say where in the base its content
/// is.
fn base_map(base: &[u8]) -> Vec<(u64, u64)> {
    const CHUNK: usize = 16 << 20;
    let n = (base.len() + CHUNK - 1) / CHUNK;
    let parts: Vec<std::sync::Mutex<Vec<(u64, u64)>>> = (0..n).map(|_| std::sync::Mutex::new(Vec::new())).collect();
    let _ = par_units::<()>(n, |i| {
        let (a, b) = (i * CHUNK, ((i + 1) * CHUNK + 32).min(base.len()));
        let mut v = Vec::with_capacity((b - a) >> ldm::MAP_BITS);
        ldm::sparse_anchors(&base[a..b], a as u64, &mut v);
        *parts[i].lock().unwrap() = v;
        Ok(())
    });
    let mut map: Vec<(u64, u64)> = parts.into_iter().flat_map(|m| m.into_inner().unwrap()).collect();
    map.sort_unstable();
    map
}

/// Bins of the base a region's choice is made in.
const BASE_BIN: usize = 4 << 20;
/// An anchor found at more places in the base than this says nothing
/// about where a unit's content is; fewer share one hit between them.
const MAP_AMBIGUOUS: usize = 64;

/// The region of the base for a unit at `a..b` of the input: the
/// window of `len` bytes holding the most of the unit's anchors (by
/// `map`), when the unit's anchors are found there at all; else the
/// base around the unit's own position (content that has not moved).
fn base_region(base_end: usize, map: &[(u64, u64)], anchors: &[(u64, u64)], a: usize, b: usize) -> (usize, usize) {
    let len = (b - a + 2 * BASE_SLACK).min(base_end);
    let positional = {
        let r0 = a.saturating_sub(BASE_SLACK).min(base_end);
        (r0, (r0 + len).min(base_end))
    };
    let bins = base_end / BASE_BIN + 1;
    // Hits in 1/MAP_AMBIGUOUS-ths: an anchor at k places is 1/k at each.
    let mut hits = vec![0u64; bins];
    let mut total = 0u64;
    for &(h, _) in anchors {
        let lo = map.partition_point(|e| e.0 < h);
        let hi = map.partition_point(|e| e.0 <= h);
        let k = hi - lo;
        if k == 0 || k > MAP_AMBIGUOUS {
            continue;
        }
        for &(_, pos) in &map[lo..hi] {
            hits[(pos as usize).min(base_end) / BASE_BIN] += (MAP_AMBIGUOUS / k) as u64;
        }
        total += MAP_AMBIGUOUS as u64;
    }
    if total < 16 * MAP_AMBIGUOUS as u64 {
        return positional;
    }
    // The window of `len` bytes (whole bins) with the most hits, ties
    // to the earliest; the region starts at its first bin.
    let span = (len / BASE_BIN).max(1);
    let (mut best, mut best_at, mut sum) = (0u64, 0usize, 0u64);
    for i in 0..bins {
        sum += hits[i];
        if i >= span {
            sum -= hits[i - span];
        }
        if sum > best {
            best = sum;
            best_at = i + 1 - span.min(i + 1);
        }
    }
    {
        // The window shrunk to the bins holding nearly all of its hits
        // (content that did not move needs no slack), never under the
        // unit plus TIGHT_SLACK a side: the region copied and indexed
        // per unit is then the unit's own size, not three times it.
        // Kernel pair on 16 cores: 15% less time and 1.2% fewer bytes.
        let min_span = ((b - a + 2 * TIGHT_SLACK) / BASE_BIN).max(1);
        if span > min_span && best > 0 {
            let lo = best_at;
            let hi = (best_at + span).min(bins);
            let need = best * TIGHT_KEEP / 100;
            let (mut ti, mut tj) = (lo, hi);
            let mut j = lo;
            let mut sum = 0u64;
            for i in lo..hi {
                while j < hi && (sum < need || j - i < min_span) {
                    sum += hits[j];
                    j += 1;
                }
                if sum >= need && j - i >= min_span && j - i < tj - ti {
                    ti = i;
                    tj = j;
                }
                if j == hi && sum < need {
                    break;
                }
                sum -= hits[i];
            }
            let r0 = (ti * BASE_BIN).min(base_end);
            return (r0, (tj * BASE_BIN).min(base_end));
        }
    }
    let r0 = (best_at * BASE_BIN).min(base_end);
    (r0, (r0 + len).min(base_end))
}
/// A region keeps at least this share of its window's hits when shrunk,
/// and at least this much slack a side of the unit.
const TIGHT_KEEP: u64 = 97;
const TIGHT_SLACK: usize = 8 << 20;

/// `input` compressed against `base` at the max level (`ultra` for the
/// ultra level): the output decodes only with the same base
/// (`decompress_with_base`). A version of a dump, a source tree or an
/// image costs a few percent of what it costs alone.
pub fn compress_with_base(base: &[u8], input: &[u8], output: &mut Vec<u8>, ultra: bool) {
    #[cfg(feature = "deflate")]
    if !in_part() && deflate::is_container(input) {
        if let Some(opened) = deflate::open(input) {
            // Both sides opened: the delta is between the plain texts.
            deflate::envelope(input.len(), &opened.recipe, output);
            let base_plain = deflate::open(base).map(|o| o.plain);
            return as_part(|| compress_with_base(base_plain.as_deref().unwrap_or(base), &opened.plain, output, ultra));
        }
    }
    compress_with_base_index(base, &BaseIndex::new(base), input, output, ultra)
}

/// A base prepared once for several `compress_with_base_index` calls
/// against it: its coarse map, which a call on a base of gigabytes and
/// an input of megabytes (a trial sample) spends most of its time on.
pub struct BaseIndex {
    map: Vec<(u64, u64)>,
}

impl BaseIndex {
    pub fn new(base: &[u8]) -> BaseIndex {
        BaseIndex { map: base_map(base) }
    }
}

/// `compress_with_base` for a store that keeps the base opened: the
/// input is opened if it is a container and its plain text compressed
/// against `base_plain` (the base's opened content and its index, made
/// once and kept), else the input as it is against `base` and its
/// index. Opening a base of thousands of gzip members on every put
/// against it was half of such a put.
pub fn compress_with_base_plain(base: &[u8], base_index: &BaseIndex, base_plain: Option<(&[u8], &BaseIndex)>, input: &[u8], anchors: Option<&[(u64, u64)]>, output: &mut Vec<u8>, ultra: bool) {
    #[cfg(feature = "deflate")]
    if !in_part() && deflate::is_container(input) {
        if let Some(opened) = deflate::open(input) {
            deflate::envelope(input.len(), &opened.recipe, output);
            let (b, bi) = base_plain.unwrap_or((base, base_index));
            // The opened text has its own positions: its anchors are not the input's.
            return as_part(|| compress_with_base_index(b, bi, &opened.plain, output, ultra));
        }
    }
    compress_with_base_anchored(base, base_index, input, anchors, output, ultra)
}

/// `compress_with_base` with the base's index made beforehand, on the
/// bytes as they are: no container is opened (`compress_with_base`
/// opens one before it gets here).
pub fn compress_with_base_index(base: &[u8], index: &BaseIndex, input: &[u8], output: &mut Vec<u8>, ultra: bool) {
    compress_with_base_anchored(base, index, input, None, output, ultra)
}

/// `compress_with_base_index` with the input's sparse anchors
/// (`ldm::sparse_anchors` over the whole input, positions absolute)
/// given when the caller has them already: the store's fingerprints
/// are those anchors, and the region choice scanned the input again.
pub fn compress_with_base_anchored(base: &[u8], index: &BaseIndex, input: &[u8], anchors: Option<&[(u64, u64)]>, output: &mut Vec<u8>, ultra: bool) {
    let parse = if ultra { Parse::Ultra } else { Parse::Dfast };
    let units: Vec<(usize, usize)> = (0..input.len().max(1)).step_by(BASE_UNIT).map(|a| (a, (a + BASE_UNIT).min(input.len()))).collect();
    let base_end = base.len().saturating_sub(BASE_TAIL);
    // Each unit's region: where the base holds its content, by a coarse
    // map of the base, so content that moved farther than the slack
    // (a table that grew, a file added early in an archive) is found.
    let map = &index.map;
    let slots: Vec<std::sync::Mutex<((usize, usize), Vec<u8>)>> = units.iter().map(|_| std::sync::Mutex::new(((0, 0), Vec::new()))).collect();
    let _ = par_units::<()>(units.len(), |i| {
        let (a, b) = units[i];
        let mut own = Vec::new();
        let unit_anchors: &[(u64, u64)] = match anchors {
            Some(all) => {
                let lo = all.partition_point(|&(_, p)| (p as usize) < a);
                let hi = all.partition_point(|&(_, p)| (p as usize) < b);
                &all[lo..hi]
            }
            None => {
                ldm::sparse_anchors(&input[a..b], 0, &mut own);
                &own
            }
        };
        let (r0, r1) = base_region(base_end, map, unit_anchors, a, b);
        // The region and the unit laid out together, in a buffer each
        // thread keeps across units: allocated and page-faulted anew
        // per unit, the copy was a quarter of a version's put.
        thread_local! {
            static FULL: RefCell<Vec<u8>> = RefCell::new(Vec::new());
        }
        let mut out = Vec::with_capacity((b - a) / 8 + 1024);
        FULL.with_borrow_mut(|full| {
            full.clear();
            full.extend_from_slice(&base[r0..r1]);
            full.extend_from_slice(&input[a..b]);
            compress_max_from(full, r1 - r0, 0, parse, None, true, &mut out, None);
        });
        *slots[i].lock().unwrap() = ((r0, r1), out);
        Ok(())
    });
    output.extend_from_slice(BASE_MAGIC);
    output.extend_from_slice(&base_id(base).to_le_bytes());
    put_varint(output, units.len() as u32);
    let streams: Vec<((usize, usize), Vec<u8>)> = slots.into_iter().map(|m| m.into_inner().unwrap()).collect();
    for (&(a, b), ((r0, r1), s)) in units.iter().zip(&streams) {
        put_varint64(output, *r0 as u64);
        put_varint64(output, (r1 - r0) as u64);
        put_varint64(output, (b - a) as u64);
        put_varint64(output, s.len() as u64);
    }
    for (_, s) in &streams {
        output.extend_from_slice(s);
    }
}

fn put_varint64(out: &mut Vec<u8>, mut v: u64) {
    while v >= 128 {
        out.push((v & 127) as u8 | 128);
        v >>= 7;
    }
    out.push(v as u8);
}

fn get_varint64(src: &[u8], pos: &mut usize) -> Option<u64> {
    let (mut v, mut shift) = (0u64, 0u32);
    loop {
        let b = *src.get(*pos)?;
        *pos += 1;
        v |= ((b & 127) as u64) << shift;
        if b < 128 {
            return Some(v);
        }
        shift += 7;
        if shift > 63 {
            return None;
        }
    }
}

/// The units of a base envelope decoded into `dst` (disjoint ranges),
/// each reading its region of `base` as history before it.
fn base_units_into(base: &[u8], units: &[BaseUnit<'_>], dst: &mut [u8]) -> Result<usize> {
    let total: usize = units.iter().map(|u| u.len).sum();
    if dst.len() < total {
        return Err(CodecError::OutputBufferTooSmall { required: total, provided: dst.len() });
    }
    let mut offsets = Vec::with_capacity(units.len());
    let mut at = 0usize;
    for u in units {
        offsets.push(at);
        at += u.len;
    }
    let ptr = dst.as_mut_ptr() as usize;
    par_units(units.len(), |i| -> Result<()> {
        let (u, off) = (&units[i], offsets[i]);
        let end = u.base_start.checked_add(u.base_len).ok_or(CodecError::CorruptedBitstream("base envelope: region"))?;
        if end + BASE_TAIL > base.len() && u.base_len > 0 {
            return Err(CodecError::CorruptedBitstream("base envelope: region past the base"));
        }
        let region = &base[u.base_start..end];
        // Units cover disjoint ranges of `dst`.
        let out = unsafe { std::slice::from_raw_parts_mut((ptr + off) as *mut u8, u.len) };
        let n = decompress_sequential_impl(u.stream, out, 0, None, true, Some(region))?;
        if n != u.len {
            return Err(CodecError::CorruptedBitstream("base envelope: unit length"));
        }
        Ok(())
    })?;
    Ok(total)
}

/// The content of a gzip or zlib object stored opened (`--max` and up):
/// what `gunzip` prints — a gzip's members one after the other, a
/// tar.gz's tar — without re-creating the deflate stream, so at the
/// plain decode speed. An object that is not one stream of content (a
/// zip, a PDF, a tar of gzips) or one stored closed is an error.
pub fn decompress_content(compressed: &[u8]) -> Result<Vec<u8>> {
    #[cfg(feature = "deflate")]
    if let Some(r) = deflate::content(compressed, decompress_parallel) {
        return r;
    }
    Err(CodecError::CorruptedBitstream("not an opened gzip or zlib object"))
}

/// `decompress_content` for an object compressed against `base`.
pub fn decompress_content_with_base(base: &[u8], compressed: &[u8]) -> Result<Vec<u8>> {
    #[cfg(feature = "deflate")]
    if let Some(r) = deflate::content_with_base(base, compressed, decompress_with_base) {
        return r;
    }
    Err(CodecError::CorruptedBitstream("not an opened gzip or zlib object"))
}

/// Decode `compressed` (a `compress_with_base` output) with its base.
pub fn decompress_with_base(base: &[u8], compressed: &[u8]) -> Result<Vec<u8>> {
    #[cfg(feature = "deflate")]
    if let Some(r) = deflate::unwrap_with_base(base, compressed, decompress_with_base) {
        return r;
    }
    let (id, units) = base_envelope(compressed).ok_or(CodecError::CorruptedBitstream("not a base envelope"))?;
    if id != base_id(base) {
        return Err(CodecError::CorruptedBitstream("base envelope: not this base"));
    }
    let total: usize = units.iter().map(|u| u.len).sum();
    let mut out = vec![0u8; total + PADDING * 2];
    let n = base_units_into(base, &units, &mut out)?;
    out.truncate(n);
    Ok(out)
}

/// `decompress_stream` for a base envelope: batches of units decoded
/// against `base` into one reused buffer, handed to `sink` in order.
pub fn decompress_stream_with_base(base: &[u8], compressed: &[u8], mut sink: impl FnMut(&[u8]) -> std::io::Result<()>) -> std::io::Result<()> {
    let codec = |e: CodecError| std::io::Error::new(std::io::ErrorKind::InvalidData, e);
    #[cfg(feature = "deflate")]
    if deflate::original_len(compressed).is_some() {
        return sink(&decompress_with_base(base, compressed).map_err(codec)?);
    }
    let (id, units) = base_envelope(compressed).ok_or_else(|| codec(CodecError::CorruptedBitstream("not a base envelope")))?;
    if id != base_id(base) {
        return Err(codec(CodecError::CorruptedBitstream("base envelope: not this base")));
    }
    let workers = threads();
    let mut buf: Vec<u8> = Vec::new();
    let mut at = 0usize;
    while at < units.len() {
        let mut end = at + 1;
        let mut total = units[at].len;
        while end < units.len() && end - at < workers.max(2) && total < STREAM_BATCH {
            total += units[end].len;
            end += 1;
        }
        buf.resize(total + PADDING * 2, 0);
        base_units_into(base, &units[at..end], &mut buf).map_err(codec)?;
        sink(&buf[..total])?;
        at = end;
    }
    Ok(())
}

/// Whether `compressed` needs a base to decode (`decompress_with_base`).
pub fn needs_base(compressed: &[u8]) -> bool {
    compressed.len() >= 16 && &compressed[..8] == BASE_MAGIC
}

// ---------------------------------------------------------------------------
// Record mode: `record::transform` turns record-shaped text (logs, table
// dumps) into typed column streams, which the ordinary levels then
// compress. The input is cut into units at line boundaries; each unit
// is transformed on its own (its own dictionaries) when a trial on its
// first megabytes shows the transform pays, and units compress and
// rebuild in parallel. The output is an envelope around the units'
// Glyd streams, and every decoder reads it (`record::inverse`).

/// The envelope: magic; number of units; per unit its original length,
/// its stream's length and whether it is a record image (1) or the text
/// itself (0); then the streams back to back.
const RECORDS_MAGIC: &[u8; 8] = b"GLYDRECS";
/// Text per unit (cut at a line end), the granule of parallel rebuild.
const RECORDS_UNIT: usize = 32 << 20;
/// Bytes of a unit the record-mode decision is made on: up to 4 MB, an
/// eighth of a small input (a pack), at least 256 KB.
const RECORDS_TRIAL: usize = 4 << 20;

fn records_trial(input: &[u8]) -> &[u8] {
    &input[..input.len().min(RECORDS_TRIAL).min((input.len() / 8).max(256 << 10))]
}

struct RecordUnit<'a> {
    len: usize,
    image: bool,
    stream: &'a [u8],
}

/// The units of a record-mode output, or None when `compressed` is not one.
fn records_envelope(compressed: &[u8]) -> Option<Vec<RecordUnit<'_>>> {
    if compressed.len() < 9 || &compressed[..8] != RECORDS_MAGIC {
        return None;
    }
    let mut pos = 8usize;
    let n = get_varint(compressed, &mut pos)? as usize;
    if n > compressed.len() {
        return None;
    }
    let mut units = Vec::with_capacity(n);
    let mut sizes = Vec::with_capacity(n);
    for _ in 0..n {
        let len = get_varint(compressed, &mut pos)? as usize;
        let clen = get_varint(compressed, &mut pos)? as usize;
        // A unit is at most RECORDS_UNIT; a corrupt header cannot ask for more.
        if len > RECORDS_UNIT {
            return None;
        }
        let image = *compressed.get(pos)?;
        pos += 1;
        sizes.push((len, clen, image != 0));
    }
    for (len, clen, image) in sizes {
        let stream = compressed.get(pos..pos.checked_add(clen)?)?;
        pos += clen;
        units.push(RecordUnit { len, image, stream });
    }
    if pos != compressed.len() {
        return None;
    }
    Some(units)
}

/// `level` in record mode: units of `input` are transformed when they are
/// record-shaped and the transform pays, else compressed as they are;
/// the units run in parallel, so `level` should be a sequential one
/// (`compress_into_max`, `compress_into_ultra`). Input whose first
/// 4 MB the transform does not pay on (JSON, binaries) goes through the
/// plain parallel path instead, whose units reach 128 MB (the
/// long-distance matcher's window; record units are 32 MB).
/// `decompress`, `decompress_into` and the parallel decoders read the
/// result.
pub fn compress_records_with(input: &[u8], output: &mut Vec<u8>, level: fn(&[u8], &mut Vec<u8>)) {
    #[cfg(feature = "deflate")]
    if deflate::wrap(input, output, |p, o| compress_records_with(p, o, level), |p, o| compress_records_with(p, o, level)) {
        return;
    }
    records_with(input, output, level, PARALLEL_UNIT_MAX)
}

/// Whether the transform pays on `sample`: its image must compress 5%
/// smaller than the sample itself. Decided at the max level whatever
/// the level: the question is the data's shape, and the cold level
/// would spend seconds on the trial.
fn records_pay(sample: &[u8], _level: fn(&[u8], &mut Vec<u8>)) -> bool {
    match record::transform(sample) {
        Some(image) => {
            let (mut a, mut b) = (Vec::new(), Vec::new());
            as_part(|| {
                compress_into_max(sample, &mut a);
                compress_into_max(&image, &mut b);
            });
            b.len() * 100 < a.len() * 95
        }
        None => false,
    }
}

fn records_with(input: &[u8], output: &mut Vec<u8>, level: fn(&[u8], &mut Vec<u8>), smallest: usize) {
    if !records_pay(records_trial(input), level) {
        compress_parallel_with(input, output, level, smallest);
        return;
    }
    records_units(input, output, level)
}

/// Record mode over units cut at line ends, `level` on each unit's
/// image (or the unit itself where the transform does not pay).
fn records_units(input: &[u8], output: &mut Vec<u8>, level: fn(&[u8], &mut Vec<u8>)) {
    // Units cut at line ends.
    let mut units: Vec<&[u8]> = Vec::new();
    let mut at = 0usize;
    while at < input.len() {
        let mut end = (at + RECORDS_UNIT).min(input.len());
        if end < input.len() {
            match input[at..end].iter().rposition(|&b| b == b'\n') {
                Some(p) if p > 0 => end = at + p + 1,
                _ => {}
            }
        }
        units.push(&input[at..end]);
        at = end;
    }
    if units.is_empty() {
        units.push(&input[..0]);
    }
    let slots: Vec<std::sync::Mutex<(bool, Vec<u8>)>> = units.iter().map(|_| std::sync::Mutex::new((false, Vec::new()))).collect();
    let _ = par_units::<()>(units.len(), |i| {
        let unit = units[i];
        let mut out = Vec::with_capacity(unit.len() / 3 + 1024);
        let image = records_pay(records_trial(unit), level) && match record::transform(unit) {
            Some(image) => {
                as_part(|| level(&image, &mut out));
                true
            }
            None => false,
        };
        if !image {
            as_part(|| level(unit, &mut out));
        }
        *slots[i].lock().unwrap() = (image, out);
        Ok(())
    });
    let compressed: Vec<(bool, Vec<u8>)> = slots.into_iter().map(|m| m.into_inner().unwrap()).collect();
    output.extend_from_slice(RECORDS_MAGIC);
    put_varint(output, units.len() as u32);
    for (u, (image, c)) in units.iter().zip(&compressed) {
        put_varint(output, u.len() as u32);
        put_varint(output, c.len() as u32);
        output.push(*image as u8);
    }
    for (_, c) in &compressed {
        output.extend_from_slice(c);
    }
}

/// Max level in record mode (see `compress_records_with`).
pub fn compress_records_into_max(input: &[u8], output: &mut Vec<u8>) {
    #[cfg(feature = "deflate")]
    if deflate::wrap(input, output, compress_records_into_max, compress_records_into_max) {
        return;
    }
    if !records_pay(records_trial(input), compress_into_max) {
        return as_part(|| compress_parallel_into_max(input, output));
    }
    records_units(input, output, compress_into_max)
}

/// Record mode with the long search (`compress_into_max_long`) in every unit.
pub fn compress_records_into_max_long(input: &[u8], output: &mut Vec<u8>) {
    #[cfg(feature = "deflate")]
    if deflate::wrap(input, output, compress_records_into_max_long, compress_records_into_max_long) {
        return;
    }
    if !records_pay(records_trial(input), compress_into_max) {
        return as_part(|| compress_parallel_into_max_long(input, output));
    }
    records_units(input, output, compress_into_max_long)
}

/// Record mode at the max level with the long search, dense
/// (`compress_into_max_dense`) where the transform does not pay.
pub fn compress_records_into_max_dense(input: &[u8], output: &mut Vec<u8>) {
    #[cfg(feature = "deflate")]
    if deflate::wrap(input, output, compress_records_into_max_dense, compress_records_into_max_dense) {
        return;
    }
    if !records_pay(records_trial(input), compress_into_max) {
        return as_part(|| compress_into_max_dense(input, output));
    }
    records_units(input, output, compress_into_max_long)
}

/// Ultra level in record mode.
pub fn compress_records_into_ultra(input: &[u8], output: &mut Vec<u8>) {
    #[cfg(feature = "deflate")]
    if deflate::wrap(input, output, compress_records_into_ultra, compress_records_into_ultra) {
        return;
    }
    records_with(input, output, compress_into_ultra, PARALLEL_UNIT_ULTRA)
}

// Decode a record-mode stream into `dst`: every unit in parallel, its
// inner stream then the rebuild.
thread_local! {
    // A record unit's image and text, kept across units.
    static RECORD_BUFS: RefCell<(Vec<u8>, Vec<u8>)> = RefCell::new((Vec::new(), Vec::new()));
}

fn records_into(units: &[RecordUnit<'_>], dst: &mut [u8]) -> Result<usize> {
    let total: usize = units.iter().map(|u| u.len).sum();
    if dst.len() < total {
        return Err(CodecError::OutputBufferTooSmall { required: total, provided: dst.len() });
    }
    let mut offsets = Vec::with_capacity(units.len());
    let mut at = 0usize;
    for u in units {
        offsets.push(at);
        at += u.len;
    }
    let base = dst.as_mut_ptr() as usize;
    par_units(units.len(), |i| -> Result<()> {
        let (u, off) = (&units[i], offsets[i]);
        // Units cover disjoint ranges of `dst`.
        let out = unsafe { std::slice::from_raw_parts_mut((base + off) as *mut u8, u.len) };
        if u.image {
            // The image and the rebuilt text in buffers this thread keeps.
            RECORD_BUFS.with_borrow_mut(|(image, text)| -> Result<()> {
                let n = decompressed_len(u.stream)?;
                image.resize(n + PADDING * 2, 0);
                let n = decompress_into(u.stream, image)?;
                image.truncate(n);
                record::inverse_into(image, text)?;
                if text.len() != u.len {
                    return Err(CodecError::CorruptedBitstream("record envelope: unit length"));
                }
                out.copy_from_slice(text);
                Ok(())
            })?;
        } else {
            let n = decompress_into(u.stream, out)?;
            if n != u.len {
                return Err(CodecError::CorruptedBitstream("record envelope: unit length"));
            }
        }
        Ok(())
    })?;
    Ok(total)
}

fn decompress_records(units: &[RecordUnit<'_>]) -> Result<Vec<u8>> {
    let total: usize = units.iter().map(|u| u.len).sum();
    let mut out = vec![0u8; total];
    records_into(units, &mut out)?;
    Ok(out)
}


/// Walk every header, validating framing, and return the total output size.
fn total_uncompressed_len(compressed: &[u8]) -> Result<usize> {
    let mut total = 0usize;
    let mut cursor = 0usize;
    while cursor + COMPACT_HEADER_MIN <= compressed.len() {
        let (header, next) = parse_header(compressed, cursor)?;
        total += header.uncompressed_len as usize;
        cursor = next;
    }
    if cursor != compressed.len() {
        return Err(CodecError::CorruptedBitstream("Trailing unparsed bytes or truncated block header"));
    }
    Ok(total)
}

/// Streams written by v0.12.0 and v0.13.0 could hold a unit that was
/// itself opened as a container: an envelope where a block was due
/// (`as_part` keeps that from happening now). Such a stream is read
/// around it: runs of blocks decoded as they are, each envelope's inner
/// blocks taken until they hold the plain text its recipe needs, then
/// closed. `None` for any other stream.
#[cfg(feature = "deflate")]
fn embedded_envelopes(compressed: &[u8]) -> Option<Result<Vec<u8>>> {
    let mut cursor = 0usize;
    while cursor + COMPACT_HEADER_MIN <= compressed.len() {
        match parse_header(compressed, cursor) {
            Ok((_, next)) => cursor = next,
            Err(_) if deflate::legacy_magic(&compressed[cursor..]) => return Some(read_around_envelopes(compressed)),
            Err(_) => return None,
        }
    }
    None
}

#[cfg(feature = "deflate")]
fn read_around_envelopes(compressed: &[u8]) -> Result<Vec<u8>> {
    let bad = || CodecError::CorruptedBitstream("an embedded envelope that does not close");
    let mut out = Vec::new();
    let (mut cursor, mut run) = (0usize, 0usize);
    while cursor < compressed.len() {
        if deflate::legacy_magic(&compressed[cursor..]) {
            if run < cursor {
                out.extend_from_slice(&decompress(&compressed[run..cursor])?);
            }
            let (inner_at, need) = deflate::embedded_head(&compressed[cursor..]).ok_or_else(bad)?;
            let (start, mut end, mut total) = (cursor + inner_at, cursor + inner_at, 0usize);
            while total < need {
                let (header, next) = parse_header(compressed, end)?;
                total += header.uncompressed_len as usize;
                end = next;
            }
            if total != need {
                return Err(bad());
            }
            let plain = decompress(&compressed[start..end])?;
            out.extend_from_slice(&deflate::embedded_close(&compressed[cursor..], &plain).ok_or_else(bad)?);
            cursor = end;
            run = end;
        } else {
            cursor = parse_header(compressed, cursor)?.1;
        }
    }
    if run < cursor {
        out.extend_from_slice(&decompress(&compressed[run..cursor])?);
    }
    Ok(out)
}

/// Decompress an entire SIMD-stream payload sequentially into a freshly allocated vector.
pub fn decompress(compressed: &[u8]) -> Result<Vec<u8>> {
    #[cfg(feature = "deflate")]
    if let Some(r) = deflate::unwrap(compressed, decompress_parallel) {
        return r;
    }
    if needs_base(compressed) {
        return Err(CodecError::CorruptedBitstream("a base envelope: decode with its base"));
    }
    if let Some(units) = records_envelope(compressed) {
        return decompress_records(&units);
    }
    if let Some(units) = cold_envelope(compressed) {
        return decompress_cold(&units);
    }
    if is_pack(compressed) {
        return decompress_pack_joined(compressed);
    }
    #[cfg(feature = "deflate")]
    if let Some(r) = embedded_envelopes(compressed) {
        return r;
    }
    let total = total_uncompressed_len(compressed)?;
    let mut output = vec![0u8; total + PADDING * 2];
    let written = decompress_into(compressed, &mut output)?;
    output.truncate(written);
    Ok(output)
}

/// Decompress a stream made by `compress_with_dict` with the same `dict`.
pub fn decompress_with_dict(dict: &Dict, compressed: &[u8]) -> Result<Vec<u8>> {
    let total = total_uncompressed_len(compressed)?;
    let mut buf = vec![0u8; total + PADDING * 2];
    let written = decompress_with_dict_into(dict, compressed, &mut buf)?;
    buf.truncate(written);
    Ok(buf)
}

/// `decompress_with_dict` into a caller's buffer of at least the output's
/// size. The dictionary's content is read in place: matches that reach
/// before the output take their bytes from it.
pub fn decompress_with_dict_into(dict: &Dict, compressed: &[u8], dst: &mut [u8]) -> Result<usize> {
    decompress_sequential_impl(compressed, dst, 0, Some(dict), true, Some(dict.content()))
}

/// The dictionary id a v7 block names; None for raw blocks (which need
/// none) and payloads too short to hold a sub-header (rejected later).
fn block_dict_id(header: &BlockHeader, payload: &[u8]) -> Option<u32> {
    if !is_coded_version(header.version) || (header.flags & FLAG_RAW_UNCOMPRESSED) != 0 {
        return None;
    }
    if header.version == VERSION_V9 {
        v7_format::SubHeader::compact_dict_id(payload)
    } else {
        v7_format::SubHeader::parse(payload).map(|s| s.dict_id)
    }
}

fn decompress_sequential(compressed: &[u8], dst: &mut [u8], verify: bool) -> Result<usize> {
    decompress_sequential_impl(compressed, dst, 0, None, verify, None)
}

/// Decode every block, in order, into `dst[dst_offset0..]`; the window
/// starts at `dst[0]`, so `dst[..dst_offset0]` (a dictionary) is history
/// the first block may match into. Every v7 block must name
/// `expected_dict` (0: none). Returns the bytes written.
/// Every block in order into `dst[dst_offset0..]`, with a dictionary
/// (`dict`: its id is what every coded block must name, its tables are
/// the first block's "previous", and with `ext` its content is history
/// outside the buffer) or without.
fn decompress_sequential_impl(compressed: &[u8], dst: &mut [u8], dst_offset0: usize, dict: Option<&Dict>, verify: bool, ext: Option<&[u8]>) -> Result<usize> {
    let mut tables = dict.map(|d| d.dec_tables());
    let expected_dict = dict.map(|d| d.id());
    if dict.is_none() {
        V7_TABLES.with_borrow_mut(|t| *t = v7_decode::DecTables::none());
    }
    let mut cursor = 0usize;
    let mut dst_offset = dst_offset0;
    let buffer_start = dst.as_ptr();
    let avx2 = has_avx2();
    let expected_dict = expected_dict.unwrap_or(0);

    while cursor + COMPACT_HEADER_MIN <= compressed.len() {
        let (header, start, next) = parse_header_at(compressed, cursor)?;
        let uncomp_len = header.uncompressed_len as usize;
        if dst_offset + uncomp_len > dst.len() {
            return Err(CodecError::OutputBufferTooSmall {
                required: dst_offset + uncomp_len,
                provided: dst.len(),
            });
        }
        // The slice runs up to `PAD` bytes past the payload when the
        // buffer has them: a compact block is then decoded in place.
        let payload = &compressed[start..(next + bits::PAD).min(compressed.len())];
        if block_dict_id(&header, payload).is_some_and(|id| id != expected_dict) {
            return Err(CodecError::CorruptedBitstream("dictionary id mismatch"));
        }
        let dst_slice = &mut dst[dst_offset..];
        unsafe {
            decode_block(&header, payload, dst_slice, buffer_start, avx2, ext, tables.as_mut())?;
        }
        if verify {
            let actual = format::block_checksum(header.flags, &dst[dst_offset..dst_offset + uncomp_len]);
            if actual != header.checksum {
                return Err(CodecError::ChecksumMismatch { expected: header.checksum, computed: actual });
            }
        }
        cursor = next;
        dst_offset += uncomp_len;
    }

    if cursor != compressed.len() {
        return Err(CodecError::CorruptedBitstream("Trailing unparsed bytes or truncated block"));
    }
    Ok(dst_offset - dst_offset0)
}

/// Decompress into a pre-allocated buffer sequentially with checksum validation.
pub fn decompress_into(compressed: &[u8], dst: &mut [u8]) -> Result<usize> {
    #[cfg(feature = "deflate")]
    if let Some(r) = deflate::unwrap(compressed, decompress_parallel) {
        let out = r?;
        if dst.len() < out.len() {
            return Err(CodecError::OutputBufferTooSmall { required: out.len(), provided: dst.len() });
        }
        dst[..out.len()].copy_from_slice(&out);
        return Ok(out.len());
    }
    if needs_base(compressed) {
        return Err(CodecError::CorruptedBitstream("a base envelope: decode with its base"));
    }
    if let Some(units) = records_envelope(compressed) {
        return records_into(&units, dst);
    }
    if let Some(units) = cold_envelope(compressed) {
        return cold_into(&units, dst);
    }
    if is_pack(compressed) {
        let all = decompress_pack_joined(compressed)?;
        if dst.len() < all.len() {
            return Err(CodecError::OutputBufferTooSmall { required: all.len(), provided: dst.len() });
        }
        dst[..all.len()].copy_from_slice(&all);
        return Ok(all.len());
    }
    #[cfg(feature = "deflate")]
    if let Some(r) = embedded_envelopes(compressed) {
        let out = r?;
        if dst.len() < out.len() {
            return Err(CodecError::OutputBufferTooSmall { required: out.len(), provided: dst.len() });
        }
        dst[..out.len()].copy_from_slice(&out);
        return Ok(out.len());
    }
    decompress_sequential(compressed, dst, true)
}

/// Decompress into pre-allocated buffer without verifying checksum (raw codec speed).
pub fn decompress_into_raw(compressed: &[u8], dst: &mut [u8]) -> Result<usize> {
    if let Some(units) = cold_envelope(compressed) {
        return cold_into(&units, dst);
    }
    decompress_sequential(compressed, dst, false)
}

struct BlockInfo {
    block_offset: usize,
    block_size: usize,
    uncomp_offset: usize,
    uncomp_len: usize,
}

struct ParallelUnit {
    first_block_idx: usize,
    block_count: usize,
    uncomp_offset: usize,
    uncomp_len: usize,
}

/// Decompress in parallel across all CPU cores into a freshly allocated vector.
pub fn decompress_parallel(compressed: &[u8]) -> Result<Vec<u8>> {
    #[cfg(feature = "deflate")]
    if let Some(r) = deflate::unwrap(compressed, decompress_parallel) {
        return r;
    }
    if needs_base(compressed) {
        return Err(CodecError::CorruptedBitstream("a base envelope: decode with its base"));
    }
    if let Some(units) = records_envelope(compressed) {
        return decompress_records(&units);
    }
    if let Some(units) = cold_envelope(compressed) {
        return decompress_cold(&units);
    }
    if is_pack(compressed) {
        return decompress_pack_joined(compressed);
    }
    #[cfg(feature = "deflate")]
    if let Some(r) = embedded_envelopes(compressed) {
        return r;
    }
    let total = total_uncompressed_len(compressed)?;
    let mut output = vec![0u8; total + PADDING * 2];
    let written = decompress_parallel_into(compressed, &mut output)?;
    output.truncate(written);
    Ok(output)
}

/// Dictionary streams (`compress_with_dict`) are sequential-only: a unit
/// here has no history before it, so any v7 block naming a dictionary is
/// rejected up front.
/// The blocks of a plain stream and its units (a unit starts at a
/// chain reset and decodes on its own), with the total decoded length.
fn scan_units(compressed: &[u8]) -> Result<(Vec<BlockInfo>, Vec<ParallelUnit>, usize)> {
    let mut blocks = Vec::new();
    let mut units: Vec<ParallelUnit> = Vec::new();
    let mut cursor = 0usize;
    let mut total_uncomp = 0usize;

    while cursor + COMPACT_HEADER_MIN <= compressed.len() {
        let (header, start, next) = parse_header_at(compressed, cursor)?;
        if block_dict_id(&header, &compressed[start..next]).is_some_and(|id| id != 0) {
            return Err(CodecError::CorruptedBitstream("dictionary streams are sequential-only"));
        }
        let uncomp_len = header.uncompressed_len as usize;
        let block_idx = blocks.len();
        blocks.push(BlockInfo {
            block_offset: cursor,
            block_size: next - cursor,
            uncomp_offset: total_uncomp,
            uncomp_len,
        });
        if (header.flags & FLAG_CHAIN_RESET) != 0 || units.is_empty() {
            units.push(ParallelUnit {
                first_block_idx: block_idx,
                block_count: 1,
                uncomp_offset: total_uncomp,
                uncomp_len,
            });
        } else {
            let u = units.last_mut().unwrap();
            u.block_count += 1;
            u.uncomp_len += uncomp_len;
        }
        cursor = next;
        total_uncomp += uncomp_len;
    }

    if cursor != compressed.len() {
        return Err(CodecError::CorruptedBitstream("Trailing unparsed bytes or truncated block"));
    }
    Ok((blocks, units, total_uncomp))
}

/// Decode `units` (their output starting at `base` of the stream) into
/// `dst`, in parallel; `dst[0]` holds the byte at `base`.
fn decode_units(compressed: &[u8], blocks: &[BlockInfo], units: &[ParallelUnit], base: usize, dst: &mut [u8], verify: bool) -> Result<()> {
    decode_units_to(compressed, blocks, units, base, dst, verify, None).map_err(|e| match e {
        StreamError::Codec(c) => c,
        StreamError::Io(_) => unreachable!("no sink"),
    })
}

enum StreamError {
    Codec(CodecError),
    Io(std::io::Error),
}

impl From<CodecError> for StreamError {
    fn from(e: CodecError) -> Self {
        StreamError::Codec(e)
    }
}

/// `decode_units`, and with `sink` each unit handed to it in order as
/// soon as it is decoded, by the thread that decoded it while the
/// others go on: the write of a unit overlaps the decode of the next
/// ones and copies from a buffer still in cache.
fn decode_units_to(compressed: &[u8], blocks: &[BlockInfo], units: &[ParallelUnit], base: usize, dst: &mut [u8], verify: bool, sink: Option<&std::sync::Mutex<&mut (dyn FnMut(&[u8]) -> std::io::Result<()> + Send)>>) -> std::result::Result<(), StreamError> {
    let output_ptr = dst.as_mut_ptr() as usize;
    let avx2 = has_avx2();
    let next_write = AtomicUsize::new(0);
    let io_error: std::sync::Mutex<Option<std::io::Error>> = std::sync::Mutex::new(None);
    // Set by a unit that failed (a corrupted block, a sink error): the
    // units after it stop waiting for their turn, which would never
    // come. Without it a flipped byte in one unit hung the decoder.
    let failed = std::sync::atomic::AtomicBool::new(false);
    let decode_unit = |i: usize| -> Result<()> {
        let unit = &units[i];
        V7_TABLES.with_borrow_mut(|t| *t = v7_decode::DecTables::none());
        let unit_buffer_start = (output_ptr + unit.uncomp_offset - base) as *const u8;
        for i in 0..unit.block_count {
            let b = &blocks[unit.first_block_idx + i];
            let block_slice = &compressed[b.block_offset..(b.block_offset + b.block_size + bits::PAD).min(compressed.len())];
            // Parsed once already in the scan above.
            let (header, start, _) = parse_header_at(block_slice, 0)?;
            let payload = &block_slice[start..];
            // Blocks cover disjoint output ranges, so these slices never alias.
            let dst_slice = unsafe {
                let ptr = (output_ptr + b.uncomp_offset - base) as *mut u8;
                std::slice::from_raw_parts_mut(ptr, b.uncomp_len)
            };
            unsafe {
                decode_block(&header, payload, dst_slice, unit_buffer_start, avx2, None, None)?;
            }
            if verify {
                let actual = format::block_checksum(header.flags, &dst_slice[..b.uncomp_len]);
                if actual != header.checksum {
                    return Err(CodecError::ChecksumMismatch { expected: header.checksum, computed: actual });
                }
            }
        }
        if let Some(sink) = sink {
            // Units before this one first: they were claimed before it
            // and finish about as soon, so the wait is short.
            while next_write.load(Ordering::Acquire) != i {
                if failed.load(Ordering::Acquire) {
                    return Err(CodecError::CorruptedBitstream("an earlier unit failed"));
                }
                std::thread::yield_now();
            }
            let out = unsafe { std::slice::from_raw_parts(unit_buffer_start, unit.uncomp_len) };
            let r = (sink.lock().unwrap())(out);
            next_write.store(i + 1, Ordering::Release);
            if let Err(e) = r {
                *io_error.lock().unwrap() = Some(e);
                return Err(CodecError::CorruptedBitstream("the sink failed"));
            }
        }
        Ok(())
    };
    let r = par_units(units.len(), |i| -> Result<()> {
        let r = decode_unit(i);
        if r.is_err() {
            failed.store(true, Ordering::Release);
        }
        r
    });
    if let Some(e) = io_error.lock().unwrap().take() {
        return Err(StreamError::Io(e));
    }
    r.map_err(StreamError::Codec)
}

fn decompress_parallel_impl(compressed: &[u8], dst: &mut [u8], verify: bool) -> Result<usize> {
    let (blocks, units, total_uncomp) = scan_units(compressed)?;
    if dst.len() < total_uncomp {
        return Err(CodecError::OutputBufferTooSmall { required: total_uncomp, provided: dst.len() });
    }
    if units.len() <= 1 {
        return decompress_sequential(compressed, dst, verify);
    }
    decode_units(compressed, &blocks, &units, 0, dst, verify)?;
    Ok(total_uncomp)
}

/// Bytes of output a streaming batch holds at least (units are added
/// until the batch reaches this, or `threads()` of them).
const STREAM_BATCH: usize = 256 << 20;

/// Batches decoded one at a time into two buffers taking turns: the
/// sink writes one on its own thread while the next is decoded, so
/// the write of a batch (a page-cache copy, a socket) overlaps the
/// decode instead of following it. `decode` fills the buffer, sized
/// `len + 2 * PADDING`, and says how many bytes are the batch's.
fn stream_batches<B>(batches: Vec<B>, decode: impl Fn(&B, &mut Vec<u8>) -> Result<usize>, len: impl Fn(&B) -> usize, mut sink: impl FnMut(&[u8]) -> std::io::Result<()> + Send) -> std::io::Result<()>
where
    B: Sync,
{
    let codec = |e: CodecError| std::io::Error::new(std::io::ErrorKind::InvalidData, e);
    if batches.len() <= 1 {
        let mut buf = Vec::new();
        for b in &batches {
            buf.resize(len(b) + PADDING * 2, 0);
            let n = decode(b, &mut buf).map_err(codec)?;
            sink(&buf[..n])?;
        }
        return Ok(());
    }
    std::thread::scope(|s| {
        let (full, filled) = std::sync::mpsc::sync_channel::<(Vec<u8>, usize)>(1);
        let (empty, emptied) = std::sync::mpsc::channel::<Vec<u8>>();
        let writer = s.spawn(move || -> std::io::Result<()> {
            for (buf, n) in filled {
                sink(&buf[..n])?;
                let _ = empty.send(buf);
            }
            Ok(())
        });
        let mut spare = 2usize;
        let mut result = Ok(());
        for b in &batches {
            let mut buf = if spare > 0 {
                spare -= 1;
                Vec::new()
            } else {
                match emptied.recv() {
                    Ok(b) => b,
                    Err(_) => break, // the writer stopped: its error is below
                }
            };
            buf.resize(len(b) + PADDING * 2, 0);
            match decode(b, &mut buf) {
                Ok(n) => {
                    if full.send((buf, n)).is_err() {
                        break;
                    }
                }
                Err(e) => {
                    result = Err(codec(e));
                    break;
                }
            }
        }
        drop(full);
        let written = writer.join().expect("the writer does not panic");
        result.and(written)
    })
}

/// Decode `compressed` a batch of units at a time into buffers reused
/// in turn, handing each batch to `sink` in order (with checksums
/// verified): the memory is two batches, not the whole output, after
/// the first batches no output page is touched for the first time,
/// which is what a whole-output decode spends most of its extra CPU
/// on, and the sink runs beside the decode (`stream_batches`). A
/// codec error comes back as `InvalidData`.
pub fn decompress_stream(compressed: &[u8], mut sink: impl FnMut(&[u8]) -> std::io::Result<()> + Send) -> std::io::Result<()> {
    let codec = |e: CodecError| std::io::Error::new(std::io::ErrorKind::InvalidData, e);
    #[cfg(feature = "deflate")]
    if let Some(r) = deflate::unwrap(compressed, decompress_parallel) {
        return sink(&r.map_err(codec)?);
    }
    let workers = threads();
    if let Some(units) = records_envelope(compressed) {
        let mut batches = Vec::new();
        let mut at = 0usize;
        while at < units.len() {
            let mut end = at + 1;
            let mut total = units[at].len;
            while end < units.len() && end - at < workers.max(2) && total < STREAM_BATCH {
                total += units[end].len;
                end += 1;
            }
            batches.push((at, end, total));
            at = end;
        }
        return stream_batches(batches, |&(at, end, total), buf| records_into(&units[at..end], buf).map(|_| total), |b| b.2, sink);
    }
    if is_pack(compressed) {
        let all = decompress_pack_joined(compressed).map_err(codec)?;
        return sink(&all);
    }
    if let Some(units) = cold_envelope(compressed) {
        let mut batches = Vec::new();
        let mut at = 0usize;
        while at < units.len() {
            let mut end = at + 1;
            let mut total = units[at].len;
            while end < units.len() && end - at < workers.max(2) && total < STREAM_BATCH {
                total += units[end].len;
                end += 1;
            }
            batches.push((at, end, total));
            at = end;
        }
        return stream_batches(batches, |&(at, end, total), buf| { buf.truncate(total); cold_into(&units[at..end], buf).map(|_| total) }, |b| b.2, sink);
    }
    #[cfg(feature = "deflate")]
    if let Some(r) = embedded_envelopes(compressed) {
        return sink(&r.map_err(codec)?);
    }
    let (blocks, units, total_uncomp) = scan_units(compressed).map_err(codec)?;
    if units.len() <= 1 {
        let mut buf = vec![0u8; total_uncomp + PADDING * 2];
        let n = decompress_sequential(compressed, &mut buf, true).map_err(codec)?;
        return sink(&buf[..n]);
    }
    // Batches of a few units per core into one reused buffer; within a
    // batch every unit goes to the sink as it is decoded.
    let mut buf: Vec<u8> = Vec::new();
    let sink: &mut (dyn FnMut(&[u8]) -> std::io::Result<()> + Send) = &mut sink;
    let mut at = 0usize;
    while at < units.len() {
        let mut end = at + 1;
        while end < units.len() && end - at < 4 * workers.max(2) && units[end].uncomp_offset - units[at].uncomp_offset < STREAM_BATCH {
            end += 1;
        }
        let base = units[at].uncomp_offset;
        let total = units[end - 1].uncomp_offset + units[end - 1].uncomp_len - base;
        buf.resize(total + PADDING * 2, 0);
        let shared: std::sync::Mutex<&mut (dyn FnMut(&[u8]) -> std::io::Result<()> + Send)> = std::sync::Mutex::new(&mut *sink);
        match decode_units_to(compressed, &blocks, &units[at..end], base, &mut buf, true, Some(&shared)) {
            Ok(()) => {}
            Err(StreamError::Codec(e)) => return Err(codec(e)),
            Err(StreamError::Io(e)) => return Err(e),
        }
        at = end;
    }
    Ok(())
}

/// Decompress in parallel across all CPU cores into a pre-allocated buffer with checksum verification.
pub fn decompress_parallel_into(compressed: &[u8], dst: &mut [u8]) -> Result<usize> {
    #[cfg(feature = "deflate")]
    if let Some(r) = deflate::unwrap(compressed, decompress_parallel) {
        let out = r?;
        if dst.len() < out.len() {
            return Err(CodecError::OutputBufferTooSmall { required: out.len(), provided: dst.len() });
        }
        dst[..out.len()].copy_from_slice(&out);
        return Ok(out.len());
    }
    if needs_base(compressed) {
        return Err(CodecError::CorruptedBitstream("a base envelope: decode with its base"));
    }
    if let Some(units) = records_envelope(compressed) {
        return records_into(&units, dst);
    }
    if let Some(units) = cold_envelope(compressed) {
        return cold_into(&units, dst);
    }
    if is_pack(compressed) {
        return decompress_into(compressed, dst);
    }
    #[cfg(feature = "deflate")]
    if let Some(r) = embedded_envelopes(compressed) {
        let out = r?;
        if dst.len() < out.len() {
            return Err(CodecError::OutputBufferTooSmall { required: out.len(), provided: dst.len() });
        }
        dst[..out.len()].copy_from_slice(&out);
        return Ok(out.len());
    }
    decompress_parallel_impl(compressed, dst, true)
}

/// Decompress in parallel across all CPU cores into a pre-allocated buffer without verifying checksum (raw codec speed).
pub fn decompress_parallel_into_raw(compressed: &[u8], dst: &mut [u8]) -> Result<usize> {
    if let Some(units) = cold_envelope(compressed) {
        return cold_into(&units, dst);
    }
    decompress_parallel_impl(compressed, dst, false)
}
