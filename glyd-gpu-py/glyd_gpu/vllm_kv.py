"""Glyd's lossless KV cache in vLLM 0.30 (kv: lossless): an attention backend of the plugin, 12 bits a value where bf16 has
16, every value read back bit for bit.

    vllm serve MODEL --quantization glyd --additional-config '{"glyd": {"kv": "lossless"}}'          (or GLYD_KV=lossless)
    ... '{"glyd": {"fraction": 0, "kv": "lossless"}}'                                                       (bf16 weights, the KV cache compressed)

The backend is registered as AttentionBackendEnum.CUSTOM and selected for the full-attention layers (AttentionConfig.backend_per_kind),
so vLLM's allocator sizes the cache from the size its spec says (customize_spec); the scheduler, block allocator and prefix hashing are vLLM's own,
untouched (prefix caching and chunked prefill work). The kernels are in gpu/glyd_gpu.cu and glyd_gpu.h (glyd_gpu_kv_*); what they need of
a model is made once (kvcal.py).

- Decode steps run Glyd's attention kernel, in vLLM's FULL decode CUDA graphs (FULL_AND_PIECEWISE, its default): a sequence's output does not depend
  on the other rows nor on the batch's padding.
- Prompts, the later steps of a chunked one and a prefix hit's new tokens run vLLM's own FlashAttention, so a row's outputs and prompt logprobs are what vLLM's bf16
  gives, bit for bit; FlashAttention is vLLM's installed one, called, not copied. A step with such rows waits for nothing on the host.
- exact (the plugin's option, with kv): every row, decode rows too, runs that way, in one FlashAttention call as vLLM's own (refused where that takes
  over 5% of the GPU's memory): the logits are vLLM's bf16 bit for bit, eager. For verifying bounded configurations.
- verify (the plugin's option): a bf16 shadow of every layer's cache, and after every write what the step touched compared with it, bit for bit:
  GlydKVVerifyError at the first value that differs. It holds a second copy of the cache, so a small one (--num-gpu-blocks-override 600) and a small run.
- Reserves: for content that does not store at 12 bits the cache keeps two reserves of its own, shares of the KV cache sized from the model's tables
  (GLYD_KV_RESERVE_1 and GLYD_KV_RESERVE_2, percent; GLYD_KV_RESERVE_1_UNITS and GLYD_KV_RESERVE_2_UNITS their sizes; for measuring), allocated at load so that
  vLLM's profile counts them. A request is let in when the second has room for what its prompt will take (vllm_sched, vllm_admit: it waits as it
  does for KV blocks; a prompt that could never fit is refused with a 400); where it runs out the scheduler preempts the requests that need it. The
  engine does not stop for what a prompt holds, and no request that stock vLLM serves is failed for it. The reserves' use and counts are kept (kv_status).
- Not in this release, refused with why: speculative decoding, sliding window, attention sinks, ALiBi, logit soft
  cap, MLA, LoRA, KV connectors and CPU offload, tensor and pipeline parallel, context parallel, fp8 KV caches,
  head sizes but 128 and 256, GPUs but Ampere, Ada and Hopper (the prompt rows' FlashAttention is vLLM's own version, 2, and 3 on Hopper, scheduled as vLLM's builder does).
- A head of 256 channels is two of the pages' heads of 128 (the library's rule): the tables (kvcal) and the pools count the pages' heads.
- Hybrid models (Qwen3.5's linear-attention layers beside its full-attention ones): the full-attention layers' cache is this one, the rest vLLM's own, all in vLLM's one pool of blocks. A block (vLLM's manager's)
  is block_size / 16 of the kernels' 16-token pages; the scheduler hands the pages of every block a step takes from the free pool over to be cleared before vLLM's worker zeroes or copies into it (a block's
  pages may hold the pools' chunks and slabs of an earlier life: vLLM zeroes a hybrid model's new attention blocks, and another group may take it), at the start of the step's execute_model.
  Under VLLM_BATCH_INVARIANT=1 a sequence's decode outputs are the same at any batch (the weights, with kv, must be bf16 or exact: Glyd's fused kernels are not).

vLLM's internals it relies on, beyond vllm_plugin's (vLLM 0.30; the entry point refuses another minor release):
- register_backend and AttentionBackendEnum.CUSTOM (v1.attention.backends.registry); AttentionConfig.backend_per_kind;
- v1.attention.backend: AttentionBackend (get_name, get_impl_cls, get_builder_cls, get_supported_kernel_block_sizes,
  supported_kv_cache_layouts, customize_spec, supports_* and validate_configuration's checks), AttentionImpl (forward,
  do_kv_cache_update, process_weights_after_loading), AttentionMetadata, AttentionMetadataBuilder (build, _cudagraph_support,
  _init_reorder_batch_threshold), AttentionCGSupport, CommonAttentionMetadata (query_start_loc and its CPU copy, seq_lens and its CPU
  upper bound, block_table_tensor, slot_mapping, num_reqs, num_actual_tokens, max_query_len, max_seq_len), AttentionType;
- v1.attention.backends.utils.split_decodes_and_prefills; v1.kv_cache_interface's AttentionSpec (dtype, state_content_bytes) and
  KVCacheLayout; quantization.kv_cache.BaseKVCacheMethod (the layer's tables);
- v1.attention.backends.fa_utils: flash_attn_varlen_func, get_flash_attn_version and get_scheduler_metadata (the prompt call, and FlashAttention 3's schedule); v1.attention.backends.utils.get_num_attention_heads_from_layers;
- the Attention layer's num_heads, num_kv_heads, head_size, head_size_v, layer_name, sliding_window, attn_type, _k_scale and _v_scale;
- vllm.envs.VLLM_BATCH_INVARIANT, and the scheduler's max_num_seqs, the model's max_model_len from vLLM's config;
- the scheduler (vllm_sched, named by SchedulerConfig.scheduler_cls, a class vLLM does not list as public): v1.core.sched.scheduler.Scheduler and async_scheduler.AsyncScheduler (running, waiting and skipped_waiting, requests, schedule,
  _preempt_request, finish_requests, update_from_output, kv_cache_manager.get_block_ids and evict_blocks, block_pool.blocks[i].ref_cnt), v1.core.sched.request_queue's queues, v1.request.Request and RequestStatus, v1.engine's EngineCoreOutput,
  EngineCoreOutputs and FinishReason; and, for the prompt check (vllm_admit), v1.engine.input_processor.InputProcessor._validate_model_input and vllm.exceptions.VLLMValidationError;
- v1.worker.gpu_worker.warmup_kernels (the V2 runner's warmup of its kernels, with steps of dummy requests whose prompts are the token ids 0, 1, 2 ...), wrapped, and only for a model whose tables count tokens
  the cache cannot store at 12 bits (skip_warmup_writes: the KV writes of those steps are skipped);
- v1.worker.gpu_worker.Worker.execute_model, wrapped (release_early): the blocks handed over are cleared first, before the model runner zeroes new blocks and copies Mamba states (hybrid models).
Under the Apache License 2.0 (LICENSE), as the rest of Glyd."""
import dataclasses
import functools
import os
import threading
import types

import numpy as np
import torch
from vllm import envs
from vllm.config import get_current_vllm_config, get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import KVCacheLayout

from . import _lib, kvcal

log = init_logger("vllm.glyd")
D = 128  # the channels of a page's head: the kernels' (a model's head of 256 is two of them)
HEAD_SIZES = (128, 256)  # the model head sizes held
PT = 16  # tokens a page
MAX_GROUP = 8  # query heads to a KV head the decode kernel takes
BACKEND_PATH = "glyd_gpu.vllm_kv.GlydBackend"
ND, CH = 16, 64  # the chunks of the escape pool a page's header names, and the entries in a chunk (gpu/glyd_gpu.cu)
STORE_PERCENT = 1.3  # the escape pool's share of the KV cache, percent (GLYD_KV_RESERVE_1), where the tables have no sweep
STORE_MARGIN = 1.0  # the pool is this much over what the tables' sweep found a whole cache of the worst token to take (a token's pages, run through the model, took 0.5 to 1.0 times the sweep's figure on the models measured; where more is asked of it, pages are raw)
STORE_MIN = 1024  # chunks the pool has at least
STORE_MIN_PERCENT, STORE_MAX_PERCENT = 0.5, 15.0  # the share the sweep may set
SECOND_K, SECOND_K_NONE = 1.5, 0.25  # the raw pool holds the planes of this many layer-pages of every block of the cache (RAW_K: a model with tokens whose pages are raw; RAW_K_NONE: a model with none)
SECOND_MIN = 256  # slabs the raw pool has at least
AUX = _lib.KV_AUX  # places of each of the status's rings at the least
AUX_STEPS = 3  # the steps' worth of blocks a ring holds: the scheduler reads it after each step, with one step pipelined behind it, one more of slack
OVERHEAD, OVERHEAD_PER_TOKEN = 300 << 20, 70 << 10  # bytes of activations and the like vLLM's profile takes off the memory before the cache, an estimate (0.86 GiB for Qwen3-8B at 8,704 tokens a step on an L4; the cache is sized after the weights load, the pool here, before)
_STATE = types.SimpleNamespace(bay=None, store_percent=STORE_PERCENT, second_percent=0.0, store_chunks=None, second_lumps=None, tables=None, path="", verify=False, exact=False, layers=0, max_leaves=0, max_seqs=0, invariant=False, warming=0, scheduler=False, reuse=False, early=False)  # this process's: the arena (None: what a page's header leaves), the escape pool's share, the raw pool's, their sizes in chunks and slabs where they are given (None: from the shares), the tables, verify, exact, layers numbered, the plan's: the model's most pages a sequence, the most sequences, batch invariance, vLLM's kernel warmup under way, whether the scheduler that fails what holds a lost page is the engine's, whether it hands every block a step takes over to be cleared first (a hybrid model: reuse), whether the worker clears them at the start of its execute_model (early)
_EXTRA = {}  # device index: the escape pool (_lib.Escape)
_TWIN = {}  # verify: layer name -> (keys, values) int16 [blocks * 16, H, D], the bf16 bits as written
LOST = [0]  # requests the scheduler has failed for a lost page
RELEASE_MAX = 4096  # pages cleared in one step at the most (a hybrid model's: all of them, before the step's other writes)
_RELEASE = []  # the pages (16 tokens) the scheduler has handed over to be cleared in every layer (of free blocks that keep pages in the pools, and of a hybrid model's blocks a step takes), under _RELEASE_LOCK
_RELEASE_LOCK = threading.Lock()
RELEASED = [0]  # blocks cleared so far
_CACHES = {}  # layer index -> (its cache tensor, its owner tag, KV heads): what a clear of a block needs of each layer
_VERIFIED = [0, 0]  # verify: values compared, pages checked
_WORK = {}  # device index: the bf16 staging cache the prefill and extension rows run FlashAttention on
_DECODE_WS = {}  # device index: the decode kernel's partial sums (uint8), one buffer for every layer and every captured graph (they run one after another on a stream)
_SHARES = {}  # (device, kv heads, batch rounded up to a power of two): the most CTAs the decode kernel splits a (sequence, head) over
_MONITORS = {}  # device index: the write kernel's status words and their host copy


class GlydKVVerifyError(RuntimeError):
    """verify: a page decodes to a value other than the one written."""


def leaf_heads(H, head_size):
    """The heads of 128 channels a page holds for H KV heads of head_size (128 or 256: two a head)."""
    return H * head_size // D


def leaf_bytes(H, bay):
    """A page's bytes (16 tokens of H heads of 128 channels) with an arena of `bay` entries: glyd_gpu.h's (kv_page_bytes)."""
    hdr = (16 * H + 4 * (1 + ND + bay) + 127) // 128 * 128
    return hdr + H * 6144


def free_bay(H):
    """The arena's entries a page of H heads holds without growing: what its header leaves up to a multiple of 128 bytes (15 for 8 heads)."""
    return (leaf_bytes(H, 0) - H * 6144 - 16 * H) // 4 - 1 - ND


def bay_of(H):
    """The arena of a page of H heads: GLYD_KV_SPARE's, else the header's."""
    return _STATE.bay if _STATE.bay is not None else free_bay(H)


def store_percent(meta, H=None):
    """The escape pool's share of the KV cache, in percent, from a model's tables (the sweep's chunks a layer-page takes where a whole cache is filled with the worst token's pages, at the default arena),
    POOL_MARGIN over, between POOL_MIN_PERCENT and POOL_MAX_PERCENT; POOL_PERCENT where the tables have none (older ones)."""
    p = (meta or {}).get("store")
    if not p:
        return STORE_PERCENT
    H = H or meta["heads"]
    pct = 100.0 * p["worst_chunks_per_layer_leaf"] * STORE_MARGIN * (CH * 4 + 4) / leaf_bytes(H, free_bay(H))
    return min(STORE_MAX_PERCENT, max(STORE_MIN_PERCENT, round(pct, 2)))


def hard_tokens(meta):
    """How many tokens of the vocabulary have pages that need more entries than a page holds (the tables' sweep): 0 where none."""
    p = (meta or {}).get("store") or {}
    return int(p.get("hard") or 0)


def second_percent(meta, H=None):
    """The raw pool's share of the KV cache, in percent: planes for RAW_K layer-pages of every block (RAW_K_NONE for a model with no such token), at GLYD_KV_RESERVE_2's if set."""
    m = meta or {}
    H = H or m["heads"]
    k = SECOND_K if hard_tokens(m) else SECOND_K_NONE
    return round(100.0 * k * H * 2048 / (max(1, int(m.get("layers", 1))) * leaf_bytes(H, free_bay(H))), 2)


def cell_bytes(H, bay, kv_heads=None):
    """The bytes vLLM's page grid gives a (KV head, token) for this page of H heads of 128 channels: state_content_bytes, 386 for 8 heads of 128 and arena 15, 772 for 4 KV heads of 256 (8 of
    the page's). A page is 16 of them a KV head; where they do not divide the page the last of each is slack (the stride between pages is the grid's)."""
    return -(-leaf_bytes(H, bay) // (16 * (kv_heads or H)))


# ---- the write kernel's status: read without waiting --------------------------------------------------------------------------
class Monitor:
    """A device's status words (glyd_gpu.h: [0] entries that found no place in a page (it was made raw), [1] the most entries any page was given, [2] pages that used their arena, [3] entries appended, [4] the escape
    pool's chunks taken now, [5] the most taken at once, [6] pages that took a chunk, [7] stale directory words, [8] the raw pool's slabs taken now, [9] the most taken at once, [10] pages made raw, [11] pages made void
    (no slab: what they hold is not what was written), [12] the ring's head, [13] the raw ring's head, then the two rings: the blocks of the void pages, the blocks that have a raw page) and a host copy, `host` (pinned), that the last layer's write refreshes in the step's own
    stream (end_of_step: a memcpy node in a CUDA graph's replay), so that it is final for a step when the step's output is on the host: the scheduler's lost() reads it, and observe() logs from it."""

    def __init__(self, device, aux=AUX):
        self.aux = int(aux)  # places of each of the two rings
        self.dev = torch.zeros(_lib.kv_status_words(self.aux), dtype=torch.int32, device=device)
        self.host = torch.zeros(_lib.kv_status_words(self.aux), dtype=torch.int32).pin_memory()
        self.np = self.host.numpy()
        self.lost_bits = None  # two bits a block, 2 b while one of its pages is void and 2 b + 1 while one is raw (made with the pools: make_escape), so that the rings name a block once
        self.index = device.index
        self.warned = 0  # the share of the escape pool (50, 75, 90 %) that has been taken at once and logged
        self.warned_second = 0  # the same for the raw pool's slabs
        self.seen = 0  # the ring's entries lost() has taken
        self.seen_second = 0  # the raw ring's entries raw_new() has taken
        self.logged_gap = 0  # void pages logged

    def end_of_step(self):
        at = _lib.KV_AUX_AT  # (the ring first, then the counters with its head, in the stream's order: a head the host reads is of a ring that has been copied)
        self.host[at:].copy_(self.dev[at:], non_blocking=True)
        self.host[:at].copy_(self.dev[:at], non_blocking=True)

    def lost(self):
        """(the blocks of the pages made void since the last call, whether the ring overflowed: more pages than it holds, so that some blocks are not named): of the steps that have reached the host."""
        head = int(self.np[_lib.KV_AUX_HEAD])
        if head == self.seen:
            return set(), False
        n = head - self.seen
        first = self.seen if n <= self.aux else head - self.aux
        blocks = {int(self.np[_lib.KV_AUX_AT + (k % self.aux)]) - 1 for k in range(first, head)}
        self.seen = head
        return blocks, n > self.aux

    def second_new(self):
        """The blocks that have had a raw page made since the last call (the raw ring: the blocks of the steps that have reached the host; more than it holds: the latest of them)."""
        head = int(self.np[_lib.KV_SECOND_AUX_HEAD])
        if head == self.seen_second:
            return set()
        first = max(self.seen_second, head - self.aux)
        at = _lib.kv_second_aux_at(self.aux)
        blocks = {int(self.np[at + (k % self.aux)]) - 1 for k in range(first, head)}
        if head - self.seen_second > self.aux:
            log.error("glyd kv: %d blocks used the lossless cache's second extra memory since the scheduler last looked, more than the %d it tracks: the older ones keep it until their blocks are written again", head - self.seen_second, self.aux)
        self.seen_second = head
        return blocks

    def taken(self):
        """(the raw pool's slabs, the escape pool's chunks) taken as of the latest host copy (the last step that has reached the host)."""
        return int(self.np[8]), int(self.np[4])

    def observe(self):
        """Logs from the latest host copy: the pools' use (50, 75 and 90 % of the escape pool's chunks or the raw pool's slabs taken at once), and pages made void. Called each step from the metadata builder."""
        st = self.np
        store = _EXTRA.get(self.index)
        nc = store.chunks if store is not None else 0
        ns = store.lumps if store is not None else 0
        for share in (90, 75, 50):  # (the highest crossed since the last line, once)
            if nc and share > self.warned and int(st[5]) * 100 >= share * nc:
                self.warned = share
                text, args = "glyd kv: %d%% of the lossless cache's first extra memory has been in use at once", [int(st[5]) * 100 // nc]
                # dev: the variable that sets it
                text, args = text + "; GLYD_KV_RESERVE_1 sets its share of the KV cache (percent, now %g)", args + [_STATE.store_percent]
                # end dev
                log.warning(text, *args)
                break
        for share in (90, 75, 50):
            if ns and share > self.warned_second and int(st[9]) * 100 >= share * ns:
                self.warned_second = share
                text, args = ("glyd kv: %d%% of the lossless cache's second extra memory has been in use at once: requests that would take more wait for room, and where it runs out the requests that need it are preempted and run again "
                              "when there is room"), [int(st[9]) * 100 // ns]
                # dev: the variable that sets it
                text, args = text + "; GLYD_KV_RESERVE_2 sets its share of the KV cache (percent, now %g)", args + [_STATE.second_percent]
                # end dev
                log.warning(text, *args)
                break
        if int(st[11]) > self.logged_gap:
            log.error("glyd kv: the lossless cache's second extra memory ran out %d times (%d%% in use): the requests that need it are preempted%s", int(st[11]), int(st[8]) * 100 // max(1, ns),
                      "" if _STATE.scheduler else ", but the engine's scheduler is not Glyd's: the values of those blocks are not what was written")
            self.logged_gap = int(st[11])

    def read(self):
        """The status now (waits for the device)."""
        torch.cuda.synchronize(self.dev.device)
        return self.dev.tolist()


def _index(device):
    """A device's index (an int, a torch.device, None: the current one)."""
    if isinstance(device, int):
        return device
    i = torch.device(device).index if device is not None else None
    return torch.cuda.current_device() if i is None else i


def aux_places(vc=None):
    """The places of each of a status's rings for vLLM's config: RING_STEPS steps' worth of the blocks a step can name (a block for each 16 tokens of its batch, and two for each request), at least RING."""
    vc = vc if vc is not None else get_current_vllm_config_or_none()
    if vc is None:
        return AUX
    sc = vc.scheduler_config
    return max(AUX, AUX_STEPS * (cdiv(int(sc.max_num_batched_tokens), PT) + 2 * int(sc.max_num_seqs)) + 64)


def monitor(device):
    i = _index(device)
    m = _MONITORS.get(i)
    if m is None:
        m = _MONITORS[i] = Monitor(torch.device("cuda", i), aux_places())
        log.info("glyd kv: tracking up to %d blocks a step", m.aux)
    return m


def gap_blocks(device, blocks):
    """verify: bool [n]: of the block ids `blocks` (int64), those with a page that is void now (what they hold is not what was written: the scheduler fails the requests that hold them)."""
    lb = monitor(device).lost_bits
    if lb is None:
        return torch.zeros(blocks.shape, dtype=torch.bool, device=blocks.device)
    b = 2 * blocks
    w = lb[torch.clamp(b >> 5, max=lb.numel() - 1)]
    return (b + 1 < lb.numel() * 32) & (((w >> (b & 31).to(torch.int32)) & 1) != 0)


def kv_status():
    """{device: the status words now} of this process (the entries that found no place, the most any page was given, the escape pool's and the raw pool's use ...: the Monitor's); "pool": {device: its chunks},
    "raw": {device: its slabs}; verify's counts "verified" (values compared, pages checked). For tests and logs."""
    return {**{d: m.read() for d, m in _MONITORS.items()}, "store": {d: e.chunks for d, e in _EXTRA.items()}, "second": {d: e.lumps for d, e in _EXTRA.items()}, "verified": list(_VERIFIED)}


def leaves_of(blocks, ratio=1):
    """The kernels' 16-token pages of vLLM's blocks of ratio x 16 tokens: block b is pages b ratio .. b ratio + ratio - 1 (vLLM's block table splits it so)."""
    return [b * ratio + j for b in blocks for j in range(ratio)] if ratio > 1 else list(blocks)


def release(leaves):
    """The scheduler's hand-over: pages (16 tokens) whose content is not to be read again, to be cleared (in every layer, before the next step's writes) so that the chunks and slabs
    they hold go back to the pools."""
    with _RELEASE_LOCK:
        _RELEASE.extend(leaves)


def release_pending():
    """Clears the pages handed over in every layer, on the current stream (up to RELEASE_MAX a call; a hybrid model's all of them): called at the start of a step's execute_model (release_early), or else
    where its metadata is built, both before its forward, so before its writes (and, in a CUDA graph's step, before the replay). Before every layer has written once, a layer whose cache tensor is not known
    holds nothing to clear: a hybrid model's pages are cleared in the layers that are known (what is handed over is never kept for later: by then another may hold it), anyone else's wait."""
    if torch.cuda.is_current_stream_capturing():  # (a graph capture must not hold a clear: it would replay)
        return 0
    if len(_CACHES) < _STATE.layers and not _STATE.reuse:
        return 0
    with _RELEASE_LOCK:
        n = len(_RELEASE) if _STATE.reuse else RELEASE_MAX
        blocks = _RELEASE[:n]
        del _RELEASE[:n]
    if not blocks or not _CACHES:
        return 0
    staged = {}
    for li in sorted(_CACHES):
        cache, tag, H = _CACHES[li]
        dev = cache.device
        if dev not in staged:
            staged[dev] = torch.tensor(blocks, dtype=torch.int32).pin_memory().to(dev, non_blocking=True)
        _lib.kv_release(cache, cache.stride(0), cache.shape[0], staged[dev], H, bay_of(H), _EXTRA.get(_index(dev)), tag, monitor(dev).dev, monitor(dev).lost_bits)
    RELEASED[0] += len(blocks)
    if not _STATE.reuse:
        log.info("glyd kv: %d blocks that no request holds were released (%d so far)", len(blocks), RELEASED[0])
    return len(blocks)


def release_early():
    """vLLM's Worker.execute_model wrapped so that a hybrid model's pages handed over are cleared first: before the model runner zeroes the step's new blocks and copies Mamba states into them, which would otherwise
    come before the clear (vLLM zeroes a hybrid model's new attention blocks, and a block another group takes may have held this cache's pages). True where it is (or was) wrapped."""
    try:
        from vllm.v1.worker import gpu_worker
    except ImportError:
        return False
    W = getattr(gpu_worker, "Worker", None)
    orig = getattr(W, "execute_model", None)
    if orig is None:
        return False
    if not getattr(orig, "_glyd_releases", False):
        @functools.wraps(orig)
        def execute_model(self, *a, **k):
            if _RELEASE and _STATE.reuse:  # (a hybrid model's; anyone else's where the step's metadata is built, as before)
                release_pending()
            return orig(self, *a, **k)

        execute_model._glyd_releases = True
        W.execute_model = execute_model
    _STATE.early = True
    return True


def make_extra(device, H):
    """The device's escape pool and raw pool, made once, at load (so vLLM's memory profile counts them): _STATE.pool_percent and _STATE.raw_percent of the KV cache, which vLLM sizes only after the weights have
    loaded, so estimated here: the memory it may use less what is in use now and OVERHEAD for its profile's activations; at least POOL_MIN chunks and RAW_MIN slabs (a slab: H 2,048 bytes, H the pages' heads of 128 channels).
    A hybrid model's pool is shared with its other layers' states: the shares are of the whole pool (more than the attention layers take)."""
    i = _index(device)
    e = _EXTRA.get(i)
    if e is None:
        vc = get_current_vllm_config()
        free, total = torch.cuda.mem_get_info(i)
        budget = max(0.0, getattr(vc.cache_config, "gpu_memory_utilization", 0.9) * total - (total - free) - OVERHEAD - OVERHEAD_PER_TOKEN * vc.scheduler_config.max_num_batched_tokens)
        chunks = max(STORE_MIN, int(_STATE.store_percent / 100.0 * budget / (CH * 4 + 4))) if _STATE.store_percent > 0 else 0
        lumps = max(SECOND_MIN, int(_STATE.second_percent / 100.0 * budget / (H * 2048))) if _STATE.second_percent > 0 else 0
        chunks = _STATE.store_chunks if _STATE.store_chunks is not None else chunks
        lumps = _STATE.second_lumps if _STATE.second_lumps is not None else lumps
        e = _EXTRA[i] = _lib.Extra(chunks, torch.device("cuda", i), lumps, H)
        L = max(1, int(getattr(_STATE.tables, "meta", {}).get("layers", 1)))
        blocks = int(1.5 * budget / (L * leaf_bytes(H, bay_of(H)))) + 4096  # (the blocks vLLM will make, estimated and some more: a block past them is named in the ring each time its page is lost)
        monitor(device).lost_bits = torch.zeros((2 * blocks + 31) // 32, dtype=torch.int32, device=torch.device("cuda", i))
        log.info("glyd kv: extra memory %.1f MiB (%g%% of the KV cache's estimated %.2f GiB) and %.1f MiB (%g%%)", e.buf.numel() / 2**20, _STATE.store_percent, budget / 2**30,
                 (e.raw.numel() if e.raw is not None else 0) / 2**20, _STATE.second_percent)
    return e


EXACT_BUDGET = 0.05  # exact's staging cache, the share of the GPU's memory it may take


WORK_ROWS = 64  # rows of a step whose rounding up to a whole block the staging cache has room for; past them a step's rows are staged in turns (a group more)
GROUPS_READ = 3  # a step whose rows the host's bound puts in this many groups or more reads, once, at its first staged layer, how many the plan made: the host's bound counts every row's blocks as its own, so where rows share a prefix most of its groups are empty, and each empty group is a launch of three kernels a layer


def work_blocks(vc, exact=False):
    """The staging cache's blocks for vLLM's config: a step's tokens, each request's a whole number of blocks (max_num_batched_tokens, and one over for each of up to STAGING_ROWS requests: a step of
    more fresh prompts than that is staged in two or more groups, a FlashAttention call each), and at least one request of the longest context (max-model-len: its context decoded and its tokens
    after it; a step's rows that do not fit together are staged in turns); exact: every row of a step at once, as vLLM's FlashAttention is called on its cache (max_num_seqs requests of the longest context)."""
    mb, ms, mml = vc.scheduler_config.max_num_batched_tokens, vc.scheduler_config.max_num_seqs, vc.model_config.max_model_len
    blocks = max(cdiv(mb, PT) + min(ms, mb, WORK_ROWS), cdiv(mml, PT))
    # dev: GLYD_KV_STAGING_BLOCKS: at least this many, for measuring
    blocks = max(blocks, int(os.environ.get("GLYD_KV_STAGING_BLOCKS") or 0))
    # end dev
    return max(blocks, ms * cdiv(mml, PT)) if exact else blocks


def work(device, H=None, hd=D):
    """The device's bf16 staging cache [blocks, 16, H, 2 hd] (hd: the model's head size): an ordinary paged cache (vLLM's LBNHC strides) that prefill and extension rows' keys and values are copied into (a
    context's pages decoded into it first) and FlashAttention runs on. Made once, at load, where vLLM's memory profile counts it (staging_blocks); exact's must fit
    EXACT_BUDGET of the GPU's memory or it is refused."""
    i = _index(device)
    s = _WORK.get(i)
    if s is None:
        vc = get_current_vllm_config()
        blocks = work_blocks(vc, _STATE.exact)
        need = blocks * PT * H * 2 * hd * 2
        total = torch.cuda.get_device_properties(i).total_memory
        if _STATE.exact and need > EXACT_BUDGET * total:
            ms, mml = vc.scheduler_config.max_num_seqs, vc.model_config.max_model_len
            raise ValueError(f"glyd: exact mode needs a bf16 copy of the KV cache of {ms} requests of up to {mml:,} tokens at once, {need / 2**30:.1f} GiB, more than "
                             f"{EXACT_BUDGET * 100:.0f}% of this GPU: serve with a smaller --max-num-seqs or --max-model-len (exact is for verifying bounded configurations)")
        s = _WORK[i] = torch.empty(blocks, PT, H, 2 * hd, dtype=torch.bfloat16, device=torch.device("cuda", i))
    return s


INVARIANT_SHARES = 8  # under VLLM_BATCH_INVARIANT the decode kernel splits a (sequence, head) over at most this many CTAs, whatever the batch (the partials' memory: max_num_seqs x heads x this)


def decode_shares(device, H, B):
    """The most CTAs the decode kernel splits a (sequence, kv head) over for B decode rows (rounded up to a power of two, so a batch and the batch padded for a CUDA graph have one): the library's
    (enough for 6 waves of the GPU, at least 16 pages a CTA (4 where the batch has few rows, on a GPU of 100 SMs or more), at most what the longest sequence the model holds needs); with batch invariance a number of its own that does not depend on B.
    A sequence's output's bits depend on its length and this."""
    i = _index(device)
    bb = 1 << max(0, (B - 1).bit_length())
    key = (i, H, 0 if _STATE.invariant else bb)
    s = _SHARES.get(key)
    if s is None:
        s = _SHARES[key] = min(INVARIANT_SHARES, -(-_STATE.max_leaves // 16)) if _STATE.invariant else _lib.kv_decode_shares(bb, H, _STATE.max_leaves, i)
    return s


def decode_workspace(device, H, G, hd=D):
    """The decode kernel's partial sums, made at load for the largest batch (so vLLM's memory profile counts it; the same memory every step, which a CUDA graph needs): the most any batch of up to
    max_num_seqs rows takes. Grown, not replaced, if a layer of other shape comes after."""
    i = _index(device)
    need = 0
    for e in range(max(1, _STATE.max_seqs - 1).bit_length() + 1):
        need = max(need, _lib.kv_decode_workspace_bytes(1 << e, H, G, decode_shares(i, H, 1 << e), hd))
    w = _DECODE_WS.get(i)
    if w is None or w.numel() < need:
        w = _DECODE_WS[i] = torch.zeros(need, dtype=torch.uint8, device=torch.device("cuda", i))  # (zero-filled: the decode kernel's counters)
        log.info("glyd kv: decode workspace of %.1f MiB (up to %d rows)", need / 2**20, _STATE.max_seqs)
    return w


def work_prepare(bt, lens, qsl, ntok, wc, w, cap):
    """The part of the staging plan that does not depend on how the rows go in groups, from tensors that are on the device already (nothing here waits for it): bt int32 [n, >= w], the rows' pages; lens
    int32 [n], the tokens each holds in all (context and query); qsl int32 [n + 1], the rows' query starts from 0; the numbers the host knows: ntok = qsl[-1], wc the most whole pages of context a row has, w
    the most pages a row has (its context and tokens), cap the staging cache's blocks. What each row's pages are (whole pages of context, the partly written one, its own blocks), where each token goes
    among them, and the pages of all the contexts sorted (rows that share a prefix share its pages: stably, so that a page's rows are together in order)."""
    dev = bt.device
    n = lens.numel()
    kl, qs = lens.to(torch.int64), qsl.to(torch.int64)
    ctx = kl - (qs[1:] - qs[:-1])
    nfull = ctx >> 4
    blocks = (kl + 15) >> 4
    P = {"bt": bt, "n": n, "kl": kl, "qs": qs, "ctx": ctx, "nfull": nfull, "blocks": blocks, "own": blocks - nfull, "rows": torch.arange(n, device=dev), "col": torch.arange(w, device=dev)[None, :], "wc": wc, "w": w,
         "cap": cap, "nl": min(n * wc, cap), "dev": dev}
    P["L"] = P["nl"] + n
    t = torch.arange(ntok, device=dev)
    P["trow"] = torch.searchsorted(qs[1:].contiguous(), t, right=True)  # (the row of each token)
    P["tpos"] = ctx[P["trow"]] + t - qs[P["trow"]]
    P["part"] = (ctx & 15) != 0
    P["ppage"] = bt.gather(1, nfull.clamp(max=bt.shape[1] - 1)[:, None])[:, 0]  # (a row's partly written context page)
    if wc:
        big = torch.iinfo(torch.int32).max
        flat = torch.where(P["col"][:, :wc] < nfull[:, None], bt[:, :wc], big).reshape(-1)  # (every whole page of a context, anything else past the end of the sort)
        srt, order = torch.sort(flat, stable=True)
        same = torch.zeros(srt.numel(), dtype=torch.bool, device=dev)
        same[1:] = srt[1:] == srt[:-1]  # (the same page as the one before it in the sort)
        P.update(srt=srt, order=order, srow=order // wc, real=srt != big, same=same)
    return P


def work_next(P, a):
    """Where the group of rows that starts at row a (a scalar on the device) ends: the first row that does not fit the staging cache with the rows from a on (at least one row), a scalar on the device: rows
    from a on in turn while the distinct whole pages of their contexts (a page's first place among the rows from a on, counted for its row) and their own blocks (the partly written last page of a context, the
    tokens) fit cap."""
    n, rows = P["n"], P["rows"]
    if P["wc"]:
        v = P["real"] & (P["srow"] >= a)
        first = v.clone()
        first[1:] &= ~(v[:-1] & P["same"][1:])
        new = torch.zeros(n, dtype=torch.int64, device=P["dev"]).scatter_add_(0, P["srow"], first.to(torch.int64))
    else:
        new = torch.zeros(n, dtype=torch.int64, device=P["dev"])
    used = torch.cumsum(torch.where(rows >= a, new + P["own"], 0), 0)
    return torch.clamp(torch.maximum(torch.searchsorted(used, P["cap"], right=True), a + 1), max=n)


def work_group(P, a, b):
    """The staging plan of the group of rows [a, b) (scalars on the device): every array is of the whole step's size and what is not the group's is empty. A dict: `cu` int32 [n + 1], the query starts with the
    rows outside the group of no tokens (at the group's start or end), `lens` int32 [n] (0 outside), `table` int32 [n, w] (a row's staging blocks: its whole pages' by their order among the group's distinct
    ones, then its own after them, rows' own in order; 0 past its last and outside), `slots` int64 [ntok] (a token's slot in the staging cache, -1 outside), and for the gather kernel src int32 [L] (the
    cache's pages to decode, -1: none), dst (their staging blocks), valid (the tokens of each that are context: 16 for a whole page), L = min(n wc, cap) + n: the whole pages first, the partly written last
    pages of the rows' contexts after (a place a row); `rows` (a, b)."""
    n, wc, w, nl, L, dev, rows, col = P["n"], P["wc"], P["w"], P["nl"], P["L"], P["dev"], P["rows"], P["col"]
    ing = (rows >= a) & (rows < b)
    src = torch.full((L + 1,), -1, dtype=torch.int32, device=dev)  # (one place more: where what is not a page's first is put)
    if wc:
        vg = P["real"] & (P["srow"] >= a) & (P["srow"] < b)
        fg = vg.clone()
        fg[1:] &= ~(vg[:-1] & P["same"][1:])
        rank_s = torch.cumsum(fg, 0, dtype=torch.int32) - 1  # (the group's pages' order among its distinct ones)
        u = rank_s[-1] + 1
        rank = torch.nn.functional.pad(torch.empty_like(rank_s).scatter_(0, P["order"], rank_s).view(n, wc), (0, w - wc))
        src.scatter_(0, torch.where(fg & (rank_s < nl), rank_s, L).to(torch.int64), P["srt"])
    else:
        u = torch.zeros((), dtype=torch.int32, device=dev)
        rank = torch.zeros(n, w, dtype=torch.int32, device=dev)
    own_g = torch.where(ing, P["own"], 0)
    ostart = torch.cumsum(own_g, 0) - own_g
    tab = torch.where(col < P["nfull"][:, None], rank.to(torch.int64), torch.where(col < P["blocks"][:, None], u.to(torch.int64) + ostart[:, None] + col - P["nfull"][:, None], 0))
    tab = torch.where(ing[:, None], tab, 0)
    cells = torch.where(ing[P["trow"]], tab[P["trow"], P["tpos"] >> 4] * PT + (P["tpos"] & 15), -1)
    src[nl:L] = torch.where(P["part"] & ing, P["ppage"], -1)
    dst = torch.arange(L, device=dev, dtype=torch.int32)
    dst[nl:L] = (u.to(torch.int64) + ostart).to(torch.int32)
    valid = torch.full((L,), PT, dtype=torch.int32, device=dev)
    valid[nl:L] = (P["ctx"] & 15).to(torch.int32)
    qs = P["qs"]
    lo, hi = qs.gather(0, a.view(1)), qs.gather(0, b.view(1))  # (a tensor as an index would be read on the host)
    return {"cu": torch.minimum(torch.maximum(qs, lo), hi).to(torch.int32), "lens": torch.where(ing, P["kl"], 0).to(torch.int32), "table": tab.to(torch.int32), "cells": cells, "src": src[:L], "dst": dst, "valid": valid, "rows": (a, b)}


def work_plan(bt, lens, qsl, G, ntok, wc, w, cap):
    """The staging plan of a step's rows with a prompt to run (a prompt, a chunk, a prefix hit's new tokens) in G groups (a FlashAttention call each, the staging cache reused; the last ones empty where
    sharing leaves fewer than the host's bound G): a list of stage_group's dicts. Everything is device operations that wait for nothing."""
    P = work_prepare(bt, lens, qsl, ntok, wc, w, cap)
    a = torch.zeros((), dtype=torch.int64, device=bt.device)
    out = []
    for _ in range(G):
        b = work_next(P, a)
        out.append(work_group(P, a, b))
        a = b
    return out


# ---- the layer's tables ---------------------------------------------------------------------------------------------------------
class GlydKVCacheMethod(BaseKVCacheMethod):
    """A decoder Attention layer's quantization method when kv is lossless: no scales (there are none), the layer's two base tables
    (glyd_kbase, glyd_vbase: uint8 [KV heads, channels]) put on its buffers once its weights have loaded."""

    def __init__(self, config):
        super().__init__(config)

    def create_weights(self, layer):
        if layer.sliding_window is not None or layer.attn_type != AttentionType.DECODER or layer.head_size not in HEAD_SIZES or layer.head_size_v != layer.head_size:
            raise ValueError(f"glyd: kv lossless does not hold {layer.layer_name} (head size {layer.head_size}, sliding window {layer.sliding_window}, {layer.attn_type}) yet: serve with kv off")
        if layer.num_heads // layer.num_kv_heads > MAX_GROUP:
            raise ValueError(f"glyd: kv lossless takes up to {MAX_GROUP} query heads to a KV head; {layer.layer_name} has {layer.num_heads // layer.num_kv_heads}")

    def process_weights_after_loading(self, layer):
        if getattr(layer, "glyd_kbase", None) is not None:  # (vLLM calls it again after the load)
            return
        t = _STATE.tables
        try:
            k, v = t.layer(layer.layer_name)
        except KeyError:
            raise ValueError(f"glyd: the KV tables {_STATE.path} have no layer {layer.layer_name}: they are for another model; make tables for this one (python -m glyd_gpu kvcal MODEL)") from None
        hp = leaf_heads(layer.num_kv_heads, layer.head_size)
        if k.shape != (hp, D):
            raise ValueError(f"glyd: the KV tables' {layer.layer_name} are {k.shape}, this layer's [{hp}, {D}] (tensor parallel needs its own tables)")
        dev = layer._k_scale.device
        layer.register_buffer("glyd_kbase", torch.from_numpy(np.ascontiguousarray(k)).to(dev), persistent=False)
        layer.register_buffer("glyd_vbase", torch.from_numpy(np.ascontiguousarray(v)).to(dev), persistent=False)
        layer.glyd_layer = _STATE.layers % 255  # (the pools' owner tag of this layer's pages: 0 to 254)
        layer.glyd_index = _STATE.layers  # (its number among the layers: the last one's write ends a step for the monitor)
        _STATE.layers += 1
        monitor(dev)  # (made now: never in a graph's memory)
        make_extra(dev, hp)


def skip_warmup_writes():
    """vLLM's V2 runner warms its kernels up before it serves (vllm.v1.worker.gpu.warmup) with steps of dummy requests whose prompts are the token ids 0, 1, 2 ..., written to the KV cache as any
    request's are. For a model whose first ids are rows of the embedding matrix that were never trained (Mistral's 768 control tokens, id 0 `<unk>` among them) the keys and values of those
    pages are all exceptions: they would take the raw pool's slabs before the first request. The pages of dummy requests are never read again, so where the base tables count tokens whose
    pages are raw the wrapped warmup_kernels sets `_STATE.warming` and do_kv_cache_update writes nothing meanwhile (CUDA graph capture, which comes after, writes padding slots: skipped as ever).
    True where the warmup is wrapped, False where this vLLM has no such function."""
    try:
        from vllm.v1.worker import gpu_worker
    except ImportError:
        return False
    orig = getattr(gpu_worker, "warmup_kernels", None)
    if orig is None:
        return False
    if getattr(orig, "_glyd_skips_writes", False):
        return True

    @functools.wraps(orig)
    def warmup_kernels(*a, **k):
        _STATE.warming += 1
        try:
            return orig(*a, **k)
        finally:
            _STATE.warming -= 1

    warmup_kernels._glyd_skips_writes = True
    gpu_worker.warmup_kernels = warmup_kernels
    return True


# ---- the backend ------------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class WorkGroup:
    """Rows of a step that run through the staging cache together (FlashAttention on bf16 values: their context's pages decoded into it, their own tokens put after): the rows the plan put in a group, which
    only the device knows, so every array is of the step's staged rows and tokens and what is not the group's is empty (stage_plan)."""

    tok0: int  # the staged rows' tokens in the step's query, after the decode rows' (tok0 .. tok1): all of them, in every group
    tok1: int
    cu: torch.Tensor  # int32 [rows + 1]: the rows' query starts (the rows outside the group of no tokens)
    lens: torch.Tensor  # int32 [rows]: the tokens each row holds in all, context and query (0: not this group's)
    max_q: int  # (bounds over all the staged rows)
    max_k: int
    cells: torch.Tensor  # int64 [tokens]: a token's slot in the staging cache (-1: not this group's)
    table: torch.Tensor  # int32 [rows, blocks]: the staging blocks of each row
    src: torch.Tensor | None = None  # int32 [places]: the cache's pages the rows' contexts are in (-1: no page in this place)
    dst: torch.Tensor | None = None  # int32 [places]: the staging blocks they are decoded into
    valid: torch.Tensor | None = None  # int32 [places]: the tokens of each that are context (a context's last page may be partly its)
    sched: torch.Tensor | None = None  # FlashAttention 3's scheduler metadata of the group (Hopper; none under VLLM_BATCH_INVARIANT, as vLLM's own builder leaves it)
    shares: int = 0  # the splits FlashAttention is given (vLLM's max_num_splits for a step of this size: GlydMetadataBuilder.splits)


@dataclasses.dataclass
class GlydMetadata(AttentionMetadata):
    """A step's: the requests in vLLM's order (decodes first: one query token), and for the rows that run through the staging cache their groups."""

    num_actual_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    block_table: torch.Tensor  # int32 [requests, blocks]
    seq_lens: torch.Tensor  # int32 [requests]
    groups: list = dataclasses.field(default_factory=list)  # StageGroup: the staged rows, in the order they come
    n_groups: torch.Tensor | None = None  # int64 scalar on the device: the groups the plan made that have rows (set where the host's bound is GROUPS_READ groups or more; `groups` is then made by the step's first staged layer)
    g_real: int | None = None  # n_groups, read once by the step's first staged layer
    lazy: tuple | None = None  # (the plan's shared part, the groups' boundaries, a function that makes a StageGroup of a group's plan): what the first staged layer makes `groups` from


class GlydBackend(AttentionBackend):
    """Attention over Glyd's 12-bit KV pages (AttentionBackendEnum.CUSTOM)."""

    forward_includes_kv_cache_update = False
    supported_dtypes = [torch.bfloat16]
    supported_kv_cache_dtypes = ["auto", "bfloat16"]

    @classmethod
    def customize_spec(cls, spec):
        """The page the layer's cache holds: uint8 cells of cell_bytes(heads, arena) a (head, token), so vLLM's allocator divides the GPU's
        memory by Glyd's 49,408-byte page where bf16's is 65,536 (8 KV heads of 128, or 4 of 256)."""
        if spec.state_content_bytes is not None:
            return spec
        hp = leaf_heads(spec.num_kv_heads, spec.head_size)
        return dataclasses.replace(spec, dtype=torch.uint8, state_content_bytes=cell_bytes(hp, bay_of(hp), spec.num_kv_heads))

    @staticmethod
    def get_name():
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        return GlydImpl

    @staticmethod
    def get_builder_cls():
        return GlydMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [PT]  # (a manager block of 32, 64 ... is that many 16-token pages)

    @classmethod
    def supported_kv_cache_layouts(cls):
        return (KVCacheLayout.LBNHC, KVCacheLayout.LBHNC, KVCacheLayout.BLNHC, KVCacheLayout.BLHNC)  # (each page one contiguous run)

    @classmethod
    def supports_head_size(cls, head_size):
        return head_size in HEAD_SIZES

    @classmethod
    def supports_compute_capability(cls, capability):
        return capability.major in (8, 9)  # (Ampere, Ada and Hopper: the prefill rows' FlashAttention is vLLM's own version, 3 on Hopper)

    @classmethod
    def supports_attn_type(cls, attn_type):
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_batch_invariance(cls):
        return True  # (a sequence's decode plan is its own, the staging route's FlashAttention runs with one split, as vLLM's: tested at batch 1 and 8)

    @classmethod
    def supports_kv_connector(cls):
        return False


class GlydMetadataBuilder(AttentionMetadataBuilder[GlydMetadata]):
    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE  # (decode steps: vLLM's FULL graphs; the rest, its piecewise graphs around the attention layers, which run eager)

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)
        self.monitor = monitor(device)
        self.cap = work_blocks(vllm_config, _STATE.exact)  # (the staging cache's blocks: the groups' size)
        self.fa3 = self.fa3_graph = None  # FlashAttention 3's schedule (Hopper): the arguments vLLM's own builder gives get_scheduler_metadata, but for the rows and the page
        from vllm.v1.attention.backends.fa_utils import get_flash_attn_version

        hd = getattr(kv_cache_spec, "head_size", D)
        if get_flash_attn_version(head_size=hd) == 3:
            from vllm.v1.attention.backends.fa_utils import get_scheduler_metadata
            from vllm.v1.attention.backends.utils import get_num_attention_heads_from_layers

            self.sched_fn = get_scheduler_metadata
            mc = vllm_config.model_config
            self.fa3 = dict(num_heads_q=get_num_attention_heads_from_layers(vllm_config, layer_names) or mc.get_num_attention_heads(vllm_config.parallel_config), num_heads_kv=kv_cache_spec.num_kv_heads,
                            headdim=hd, qkv_dtype=torch.bfloat16, page_size=PT, causal=True, window_size=(-1, -1))
            cc = vllm_config.compilation_config
            if cc.cudagraph_mode.has_full_cudagraphs():  # (vLLM's builder: the bound on the splits that its graphs are captured with, for the steps that fit them)
                self.fa3_graph = (vllm_config.attention_config.flash_attn_max_num_splits_for_cuda_graph, cc.max_cudagraph_capture_size)

    def shares(self, ntok):
        """The splits vLLM's builder gives FlashAttention (its max_num_splits) for a step of ntok tokens: 1 under VLLM_BATCH_INVARIANT; with version 3 and the full CUDA graphs on, the bound its graphs are captured with
        for a step that fits them; else 0 (the kernel's own choice)."""
        if envs.VLLM_BATCH_INVARIANT:
            return 1
        if self.fa3_graph is not None and self.fa3_graph[1] is not None and ntok <= self.fa3_graph[1]:
            return self.fa3_graph[0]
        return 0

    def schedule(self, p, max_q, max_k, shares=0):
        """FlashAttention 3's scheduler metadata of a group of staged rows, as vLLM's builder makes it for a batch of those rows (the group's starts and lengths, every staged row in it: those outside the group
        have no tokens), at the step's longest query and context and with its splits; none where the version is 2 and under VLLM_BATCH_INVARIANT (vLLM's builder leaves it out there)."""
        if self.fa3 is None or envs.VLLM_BATCH_INVARIANT:
            return None
        return self.sched_fn(batch_size=p["lens"].numel(), max_seqlen_q=max_q, max_seqlen_k=max_k, cache_seqlens=p["lens"], cu_seqlens_q=p["cu"], num_splits=shares, **self.fa3)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        """A step's metadata. Nothing here waits for the GPU: the lengths it reads on the host are vLLM's CPU copies (the query starts, and the upper bound of the sequence lengths, which is exact
        for the rows with a prompt to run), and where a row's pages go in the staging cache is planned on the device from the block table that is there (stage_plan)."""
        cam = common_attn_metadata
        if _RELEASE and not (_STATE.early and _STATE.reuse):  # (a hybrid model's are cleared where the step's execute_model starts)
            release_pending()
        self.monitor.observe()
        nd, npf, ndt, _ = split_decodes_and_prefills(cam, decode_threshold=1)
        if _STATE.exact:  # (every row through the staging cache, in one FlashAttention call as vLLM's own)
            nd, npf, ndt = 0, cam.num_reqs, 0
        md = GlydMetadata(num_actual_tokens=cam.num_actual_tokens, num_decodes=nd, num_decode_tokens=ndt, num_prefills=npf, block_table=cam.block_table_tensor, seq_lens=cam.seq_lens)
        if npf:
            ub = cam.seq_lens_cpu_upper_bound
            qs = cam.query_start_loc_cpu[nd : nd + npf + 1]
            qlen = (qs[1:] - qs[:-1]).tolist()
            klen = (ub[nd : nd + npf] if ub is not None else cam.seq_lens[nd : nd + npf].cpu()).tolist()  # (exact for rows outside async speculation)
            bt, lens, qsl = cam.block_table_tensor[nd : nd + npf], cam.seq_lens[nd : nd + npf], cam.query_start_loc[nd : nd + npf + 1] - cam.query_start_loc[nd]
            r0, G = 0, 0
            while r0 < npf:  # (the groups the rows need at the most: rows in turn while the blocks each needs, as if no page were shared, fit the staging cache; sharing makes the plan's groups larger, so fewer)
                used, r1 = 0, r0
                while r1 < npf and (r1 == r0 or used + cdiv(klen[r1], PT) <= self.cap):
                    used += cdiv(klen[r1], PT)
                    r1 += 1
                if used > self.cap:
                    raise RuntimeError(f"glyd kv: a request of {klen[r0]:,} tokens needs {used:,} blocks of the prompt workspace, which has {self.cap:,} (sized by max-model-len)")
                G += 1
                r0 = r1
            ntok = sum(qlen)
            wc = max((k - q) // PT for k, q in zip(klen, qlen))
            ctx = wc > 0 or any(k > q for k, q in zip(klen, qlen))
            w_ = max(cdiv(k, PT) for k in klen)
            ns = self.shares(cam.num_actual_tokens)

            def make(p):
                return WorkGroup(tok0=0, tok1=ntok, cu=p["cu"], lens=p["lens"], max_q=max(qlen), max_k=max(klen), cells=p["cells"], table=p["table"], sched=self.schedule(p, max(qlen), max(klen), ns), shares=ns,
                                  **({"src": p["src"], "dst": p["dst"], "valid": p["valid"]} if ctx else {}))

            if G < GROUPS_READ:  # (at most one empty group: made whole)
                md.groups = [make(p) for p in work_plan(bt, lens, qsl, G, ntok, wc, w_, self.cap)]
            else:  # (the bound allows many groups and sharing leaves most of them empty: where each ends is found now, a few operations a group, and the groups the step has rows in are made by its first staged layer, which reads how many)
                P = work_prepare(bt, lens, qsl, ntok, wc, w_, self.cap)
                a, bounds = torch.zeros((), dtype=torch.int64, device=bt.device), []
                for _ in range(G):
                    b = work_next(P, a)
                    bounds.append((a, b))
                    a = b
                md.n_groups = torch.stack([b > a for a, b in bounds]).sum()
                md.lazy = (P, bounds, make)
        return md


class GlydImpl(AttentionImpl):
    supports_dcp = False
    supports_pcp = False

    def __init__(self, num_heads, head_size, scale, num_kv_heads=None, alibi_slopes=None, sliding_window=None, kv_cache_dtype="auto", logits_soft_cap=None,
                 attn_type=AttentionType.DECODER, kv_sharing_target_layer_name=None, **kwargs):
        if alibi_slopes is not None or sliding_window is not None or logits_soft_cap or kwargs.get("sinks") is not None or kv_sharing_target_layer_name is not None:
            raise ValueError("glyd kv: lossless does not hold ALiBi, a sliding window, a logit soft cap, attention sinks or KV sharing between layers yet: serve with kv off")
        if head_size not in HEAD_SIZES or attn_type != AttentionType.DECODER:
            raise ValueError(f"glyd kv: lossless holds decoder attention of head size 128 or 256, not {attn_type} of {head_size}")
        self.num_heads, self.head_size, self.scale = num_heads, head_size, float(scale)
        self.num_kv_heads = num_kv_heads or num_heads
        self.hp = leaf_heads(self.num_kv_heads, head_size)  # (the page's heads of 128 channels)
        self.group = num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func, get_flash_attn_version

        self.fa = flash_attn_varlen_func
        picked = get_flash_attn_version(head_size=head_size)
        if picked not in (2, 3):
            raise ValueError(f"glyd kv: lossless runs prefill on FlashAttention 2 (3 on Hopper); vLLM picked version {picked} here")
        self.fa_version = picked  # (vLLM's own: 3 on Hopper, scheduled by the builder as vLLM's is: the staged rows' calls are the calls vLLM's own backend makes on the same rows)

    def process_weights_after_loading(self, act_dtype):
        if act_dtype != torch.bfloat16:
            raise ValueError(f"glyd kv: lossless holds bf16 activations, not {act_dtype}")
        dev = torch.cuda.current_device()
        work(dev, self.num_kv_heads, self.head_size)
        monitor(dev)
        release_early()
        vc = get_current_vllm_config()
        _STATE.max_leaves, _STATE.max_seqs, _STATE.invariant = cdiv(vc.model_config.max_model_len, PT), vc.scheduler_config.max_num_seqs, bool(envs.VLLM_BATCH_INVARIANT)
        if _STATE.invariant:  # (a sequence's CTAs from its own length: not from the number of rows the step has)
            _lib.kv_floor_set(1)
        decode_workspace(dev, self.num_kv_heads, self.group, self.head_size)
        if hard_tokens(getattr(_STATE.tables, "meta", None)):  # (vLLM's warmup steps use the ids 0, 1, 2 ...)
            skip_warmup_writes()

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        if kv_cache.numel() == 0 or slot_mapping.numel() == 0 or _STATE.warming:
            return
        mon = monitor(kv_cache.device)
        c = _CACHES.get(layer.glyd_index)
        if (c is None or c[0].data_ptr() != kv_cache.data_ptr()) and not torch.cuda.is_current_stream_capturing():  # (the layer's cache as a step runs it: vLLM's profiling and capture runs use caches of their own)
            _CACHES[layer.glyd_index] = (kv_cache, layer.glyd_layer, self.hp)
        _lib.kv_write(key, value, slot_mapping, kv_cache, kv_cache.stride(0), layer.glyd_kbase, layer.glyd_vbase, bay_of(self.hp), _EXTRA.get(_index(kv_cache.device)), layer.glyd_layer, mon.dev, mon.lost_bits)
        if layer.glyd_index == _STATE.layers - 1:  # (the step's last write: the status goes to the host in the step's own stream, final when the step's output is there: what the scheduler reads)
            mon.end_of_step()
        if _STATE.verify:
            self._verify(layer, key, value, kv_cache, slot_mapping)

    def _verify(self, layer, key, value, kv_cache, cells):
        """verify: this step's tokens into the layer's bf16 shadow, then the pages they are in decoded and every token of them written so far compared with it."""
        H, D = self.num_kv_heads, self.head_size
        sh = _TWIN.get(layer.layer_name)
        if sh is None:
            try:
                sh = _TWIN[layer.layer_name] = tuple(torch.zeros(kv_cache.shape[0] * PT, H, D, dtype=torch.int16, device=kv_cache.device) for _ in range(2))
            except torch.OutOfMemoryError as e:
                raise RuntimeError(f"glyd kv: verify keeps a bf16 copy of each layer's cache ({kv_cache.shape[0]} blocks here): serve a small cache (--num-gpu-blocks-override 600) ({e})") from e
        n = cells.numel()
        live = cells >= 0
        s = cells[live]
        sh[0][s] = key[:n].view(torch.int16)[live]
        sh[1][s] = value[:n].view(torch.int16)[live]
        leaves, inv = torch.unique(s >> 4, return_inverse=True)
        last = torch.full((leaves.numel(),), -1, dtype=torch.int64, device=s.device).scatter_reduce_(0, inv, s & 15, "amax")
        ok = torch.empty(leaves.numel(), PT, H, D, dtype=torch.int16, device=s.device)
        ov = torch.empty_like(ok)
        _lib.kv_unpack(kv_cache, kv_cache.stride(0), leaves.to(torch.int32), layer.glyd_kbase, layer.glyd_vbase, bay_of(self.hp), _EXTRA.get(_index(kv_cache.device)), ok, ov)
        mask = (torch.arange(PT, device=s.device)[None, :] <= last[:, None])[:, :, None, None] & ~gap_blocks(kv_cache.device, leaves.to(torch.int64))[:, None, None, None]
        bad = ((ok != sh[0].view(-1, PT, H, D)[leaves]) | (ov != sh[1].view(-1, PT, H, D)[leaves])) & mask
        nbad = int(bad.sum())
        _VERIFIED[0] += int(mask.sum()) * H * D * 2
        _VERIFIED[1] += leaves.numel()
        if nbad:
            p, t, h, c = bad.nonzero()[0].tolist()
            raise GlydKVVerifyError(f"glyd kv verify: {nbad} values of {layer.layer_name} decode to other bits than were written (first: block {int(leaves[p])}, token {t}, head {h}, channel {c})")

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale=None, output_block_scale=None):
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not supported by glyd kv")
        md = attn_metadata
        if md is None:  # (vLLM's profiling run)
            return output.fill_(0)
        ndt = md.num_decode_tokens
        if ndt:  # (a row with no query token, graph padding, has no output: the rows are the first ndt)
            i = _index(kv_cache.device)
            _lib.kv_decode(query[:ndt], kv_cache, kv_cache.stride(0), layer.glyd_kbase, layer.glyd_vbase, md.block_table[:ndt], md.seq_lens[:ndt], decode_shares(i, self.num_kv_heads, ndt), self.num_kv_heads, self.group,
                           bay_of(self.hp), _EXTRA.get(i), self.scale, output[:ndt], _DECODE_WS[i])
        if md.lazy is not None and md.g_real is None:  # (the step's one wait on the host, here where the GPU has only the plan's boundaries and this layer's projections to do first, and not in the metadata: how many of the bound's groups have rows)
            P, bounds, make = md.lazy
            md.g_real = int(md.n_groups)
            md.groups = [make(work_group(P, a, b)) for a, b in bounds[: md.g_real]]
        for g in md.groups:
            self._work(layer, md, g, query, key, value, kv_cache, output)
        return output

    def _work(self, layer, md, g, query, key, value, kv_cache, output):
        """A group of rows through the staging cache: their contexts' pages decoded into it (a prefix hit's, the chunks before this one), their own keys and values put after, FlashAttention on it
        as it runs on vLLM's own cache (the same arguments): the same kernel on the same bf16 values, so a row's outputs are vLLM's bit for bit."""
        ndt = md.num_decode_tokens
        a, b = ndt + g.tok0, ndt + g.tok1
        H, D = self.num_kv_heads, self.head_size
        st = work(query.device, H, D)
        if g.src is not None:
            _lib.kv_gather(kv_cache, kv_cache.stride(0), g.src, g.dst, layer.glyd_kbase, layer.glyd_vbase, bay_of(self.hp), _EXTRA.get(_index(kv_cache.device)), st)
            if _STATE.verify:
                self._verify_gather(layer, g, st)
        _lib.kv_work_put(key[a:b], value[a:b], g.cells, st)
        shape = (g.lens.numel(), H)
        self.fa(q=query[a:b], k=st[..., :D], v=st[..., D:], out=output[a:b], cu_seqlens_q=g.cu, max_seqlen_q=g.max_q, seqused_k=g.lens, max_seqlen_k=g.max_k,
                softmax_scale=self.scale, causal=True, alibi_slopes=None, window_size=[-1, -1], block_table=g.table, softcap=0, scheduler_metadata=g.sched, fa_version=self.fa_version,
                q_descale=None, k_descale=layer._k_scale.expand(shape), v_descale=layer._v_scale.expand(shape), num_splits=g.shares, s_aux=None)

    def _verify_gather(self, layer, g, st):
        """verify: the context pages decoded into the staging cache against the layer's bf16 shadow (the values as they were written), every token of them that is context."""
        H, D = self.num_kv_heads, self.head_size
        sh = _TWIN.get(layer.layer_name)
        if sh is None:
            raise GlydKVVerifyError(f"glyd kv verify: {layer.layer_name} reads blocks that were never written")
        keep = g.src >= 0  # (the places of the plan's lists that have a page)
        leaves = g.src[keep].to(torch.int64)
        got = st[g.dst[keep].to(torch.int64)].view(torch.int16)  # [pages, 16, H, 2 D]
        mask = (torch.arange(PT, device=leaves.device)[None, :] < g.valid[keep][:, None])[:, :, None, None] & ~gap_blocks(leaves.device, leaves)[:, None, None, None]
        bad = ((got[..., :D] != sh[0].view(-1, PT, H, D)[leaves]) | (got[..., D:] != sh[1].view(-1, PT, H, D)[leaves])) & mask
        nbad = int(bad.sum())
        _VERIFIED[0] += int(mask.sum()) * H * D * 2
        _VERIFIED[1] += leaves.numel()
        if nbad:
            p, t, h, c = bad.nonzero()[0].tolist()
            raise GlydKVVerifyError(f"glyd kv verify: {nbad} values of {layer.layer_name}'s context, as decoded for a prompt, are other bits than were written (first: block {int(leaves[p])}, token {t}, head {h}, channel {c})")


def register():
    """The backend as AttentionBackendEnum.CUSTOM (in every process vLLM starts); False where another plugin holds the slot."""
    c = AttentionBackendEnum.CUSTOM
    if c.is_overridden() and c.get_path() != BACKEND_PATH:
        return False
    register_backend(c, BACKEND_PATH)
    return True


def use_tables(path, verify=False, exact=False):
    """Loads the tables file; verify: every write checked (the shadow cache), and every context decoded into the staging cache; exact: every row's attention through vLLM's own FlashAttention (the staging cache). GLYD_KV_SPARE sets a block's own spare (0 to
    1023, for measuring), GLYD_KV_RESERVE_1 and GLYD_KV_RESERVE_2 the first and second reserves' shares of the KV cache (percent), GLYD_KV_RESERVE_1_UNITS and GLYD_KV_RESERVE_2_UNITS their sizes in units."""
    t = kvcal.Tables(path)
    bay = None
    # dev: GLYD_KV_SPARE sets a block's own spare (0 to 1023, for measuring)
    if os.environ.get("GLYD_KV_SPARE", "").strip():
        bay = int(os.environ["GLYD_KV_SPARE"])
        if not 0 <= bay <= 1023:
            raise ValueError(f"glyd: GLYD_KV_SPARE {bay}: 0 to 1023")
        log.warning("glyd kv: GLYD_KV_SPARE is %d, not the default", bay)
    # end dev
    store = STORE_PERCENT if bay is not None else store_percent(t.meta)
    # dev: GLYD_KV_RESERVE_1 sets the first extra memory's share of the KV cache (percent)
    if os.environ.get("GLYD_KV_RESERVE_1", "").strip():
        store = float(os.environ["GLYD_KV_RESERVE_1"])
        if not 0.0 <= store <= 50.0:
            raise ValueError(f"glyd: GLYD_KV_RESERVE_1 {store:g}: 0 to 50 (percent of the KV cache)")
        log.warning("glyd kv: GLYD_KV_RESERVE_1 sets the first extra memory to %g%% of the KV cache, not %g%%", store, store_percent(t.meta) if bay is None else STORE_PERCENT)
    # end dev
    raw = second_percent(t.meta)
    # dev: GLYD_KV_RESERVE_2 the second's
    if os.environ.get("GLYD_KV_RESERVE_2", "").strip():
        raw = float(os.environ["GLYD_KV_RESERVE_2"])
        if not 0.0 <= raw <= 50.0:
            raise ValueError(f"glyd: GLYD_KV_RESERVE_2 {raw:g}: 0 to 50 (percent of the KV cache)")
        log.warning("glyd kv: GLYD_KV_RESERVE_2 sets the second extra memory to %g%% of the KV cache, not %g%%", raw, second_percent(t.meta))
    # end dev
    _STATE.store_chunks = _STATE.second_lumps = None
    # dev: the extra memories' sizes themselves, for measuring
    for name, key in (("store_chunks", "GLYD_KV_RESERVE_1_UNITS"), ("second_lumps", "GLYD_KV_RESERVE_2_UNITS")):
        v = os.environ.get(key, "").strip()
        setattr(_STATE, name, int(v) if v else None)
        if v:
            if not 0 <= int(v) <= 1 << 30:
                raise ValueError(f"glyd: {key} {v}: 0 to 2^30")
            log.warning("glyd kv: the %s extra memory is %d (%s), not a share of the KV cache", "first" if name == "store_chunks" else "second", int(v), key)
    # end dev
    _STATE.tables, _STATE.bay, _STATE.store_percent, _STATE.second_percent, _STATE.path, _STATE.verify, _STATE.exact = t, bay, store, raw, path, verify, exact
    log.info("glyd kv: extra memory of %g%% and %g%% of the KV cache", store, raw)
    if hard_tokens(t.meta):
        log.info("glyd kv: %d tokens of this model's vocabulary need the lossless cache's second extra memory", hard_tokens(t.meta))
    return t
