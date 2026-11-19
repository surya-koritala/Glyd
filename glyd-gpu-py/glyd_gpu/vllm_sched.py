"""The scheduler of kv: lossless: requests wait for the room they need, and are never failed for content the cache could not store.

A prompt's needs are predicted before it runs (vllm_admit: the tables' sweep of every token), and the request is let in only when the cache's second reserve has room for them, as vLLM keeps a request
waiting when its KV blocks are short; the others in the queue go on. Where a step's tokens still exhaust the reserve (the prompt took more than predicted, or the tokens it generates), the requests that need what it could not
store are preempted as vLLM preempts for blocks (their blocks freed, the request back in the queue, its prompt and what it has generated recomputed when there is room), one at a time after the others that were running have finished. Only a request that
cannot fit even alone ends with an error: a prompt that could not is refused when it is made (vllm_admit, a 400); one that outgrows the reserve alone by what it generates, with the error finish.
What a request held of the reserve is given back as soon as its blocks are free. GLYD_KV_ADMIT=off turns the prediction off, here and where the API server validates a prompt (for measuring: every request is let in
whatever it needs, and those that exhaust the reserve are preempted).

A block of vLLM's (its manager's) is block_size / 16 of the cache's 16-token pages, the unit the status's rings name and the clears take (a hybrid model's attention blocks hold hundreds of tokens). Where vLLM zeroes
the KV blocks a step takes (kv_cache_config.needs_kv_cache_zeroing: a hybrid model, whose pool of blocks its linear-attention or state-space layers share), every block a step takes from the free pool is handed over to be
cleared at the start of the step, before vLLM zeroes it or another group writes it: a block's pages may hold the reserves' chunks and slabs of an earlier life, which the zeroed page would no longer name.

vLLM's scheduler classes are not a public interface; this one subclasses vLLM 0.30's own (Scheduler, or AsyncScheduler where async scheduling is on) and overrides schedule, update_from_output and add_request's queue.
It relies on: SchedulerConfig.scheduler_cls; Scheduler.running, waiting, skipped_waiting (v1.core.sched.request_queue's queues), requests, max_num_running_reqs, num_waiting_for_streaming_input, prev_step_scheduled_req_ids,
_preempt_request (with drop_stale_output), finish_requests, update_from_output, schedule and kv_cache_config (kv_cache_groups' kv_cache_spec, needs_kv_cache_zeroing); Request's status, num_tokens, num_computed_tokens,
num_in_flight_tokens, all_token_ids, prompt_embeds, client_index and request_id; kv_cache_manager (get_block_ids, evict_blocks, reset_prefix_cache, block_pool.blocks[i].ref_cnt and block_pool.get_new_blocks, wrapped
on the instance); v1.kv_cache_interface.AttentionSpec; RequestStatus; EngineCoreOutput, EngineCoreOutputs and FinishReason."""
import importlib
import os
import time

import numpy as np
from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.request import RequestStatus

from . import vllm_admit, vllm_kv

log = init_logger("vllm.glyd")
MESSAGE = "glyd kv: the lossless cache cannot hold this request, even alone: send a shorter prompt, or serve with kv off"
MAX_TRIES = 6  # exhaustions of the reserve a request may cause before it ends with an error (the reserve is shared; a request that is still there after this many waits does not fit it)
WAITING = (RequestStatus.WAITING, RequestStatus.PREEMPTED)


class Pred:
    """What a request's tokens are predicted to take of the second reserve: `cum[b]` of the blocks before block b, `total` of all, for the `n` tokens it was made from."""

    __slots__ = ("n", "cum", "total")

    def __init__(self, n, blocks):
        self.n = n
        self.cum = np.concatenate([[0], np.cumsum(blocks, dtype=np.int64)])
        self.total = int(self.cum[-1])

    def remaining(self, done):
        """The raw pages of the blocks past the first `done` tokens (a block that is partly written counts whole)."""
        return self.total - int(self.cum[min(done // 16, len(self.cum) - 1)])


class LostLeaves:
    """The scheduler's behaviour: before a step, requests whose predicted raw pages the pool has no room for yet are kept out of the queues the step takes from (and a prompt that could never fit is ended with an error);
    after it, the requests that hold a page that was lost (vllm_kv.Monitor.lost) are preempted, those that cannot fit alone ended with an error; and the blocks that have had a raw page (Monitor.raw_new) or a lost one are handed
    over to be cleared (vllm_kv.release) once no request holds them."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._glyd_pending = set()  # blocks that have had a raw or a lost page and are not free yet
        self._glyd_cache = {}  # request id -> (tokens it was predicted for, its Pred or None: no raw page)
        self._glyd_run = {}  # request id -> Pred: the heavy requests let in (their raw pages are counted while they run)
        self._glyd_after = {}  # request id -> the ids of the requests it waits for (running when it caused an exhaustion)
        self._glyd_tries = {}  # request id -> the exhaustions it was held for
        self._glyd_errors = []  # requests ended with an error that the next output carries
        self._glyd_ever_held = set()  # ids of the requests that were held (for the count)
        self._glyd_predictor = vllm_admit.predictor(vllm_kv._STATE.tables)
        # dev: GLYD_KV_ADMIT=off: no prediction (for measuring)
        if os.environ.get("GLYD_KV_ADMIT", "").strip().lower() == "off":
            self._glyd_predictor = None
        # end dev
        self.glyd_stats = {"held": 0, "preempted": 0, "refused": 0, "failed": 0}  # requests that were kept waiting, preempted (each time), ended for never fitting, ended alone
        self._glyd_ratio, self._glyd_new = 1, []  # the pages a block (its attention group's block_size / 16); the blocks taken from the free pool since the last step (where they are handed over)
        kcc = getattr(self, "kv_cache_config", None)
        if kcc is not None:
            from vllm.v1.kv_cache_interface import AttentionSpec

            sizes = {g.kv_cache_spec.block_size for g in kcc.kv_cache_groups if isinstance(g.kv_cache_spec, AttentionSpec)}
            if len(sizes) == 1:
                self._glyd_ratio = max(1, sizes.pop() // vllm_kv.PT)
            if getattr(kcc, "needs_kv_cache_zeroing", False):
                self._glyd_tap()

    def _glyd_tap(self):
        """Every block the pool hands out from now on is noted (block_pool.get_new_blocks, wrapped on this pool): handed over to be cleared at the start of the step that takes it."""
        bp = self.kv_cache_manager.block_pool
        orig = bp.get_new_blocks
        new = self._glyd_new

        def get_new_blocks(*a, **k):
            out = orig(*a, **k)
            new.extend(b.block_id for b in out)
            return out

        bp.get_new_blocks = get_new_blocks
        vllm_kv._STATE.reuse = True

    def _glyd_leaves(self, blocks):
        return vllm_kv.leaves_of(blocks, self._glyd_ratio)

    # ---- before a step -------------------------------------------------------------------------------------------------------------
    def schedule(self, *args, **kwargs):
        held = self._glyd_hold()
        try:
            return super().schedule(*args, **kwargs)
        finally:
            self._glyd_unhold(held)
            if self._glyd_new:  # (the blocks this step takes: cleared before it writes them)
                vllm_kv.release(self._glyd_leaves(self._glyd_new))
                self._glyd_new.clear()

    def _glyd_store(self):
        """(slabs of the raw pool, slabs taken now): of this process's pools, the taken as the latest host copy of the status has them."""
        lumps = sum(e.lumps for e in vllm_kv._EXTRA.values())
        taken = sum(m.taken()[0] for m in vllm_kv._MONITORS.values())
        return lumps, taken

    def _glyd_pred(self, r):
        """The Pred of a request's tokens, None where none of its pages is predicted raw (made once for as many tokens as it has)."""
        n = r.num_tokens
        c = self._glyd_cache.get(r.request_id)
        if c is not None and c[0] == n:
            return c[1]
        p = None
        if getattr(r, "prompt_embeds", None) is None:
            ids = np.fromiter(r.all_token_ids, dtype=np.int64, count=n)
            b = self._glyd_predictor.blocks(ids)
            if b.any():
                p = Pred(n, b)
        self._glyd_cache[r.request_id] = (n, p)
        return p

    def _glyd_forget(self):
        """Drops what is kept for requests that are gone, and the reservations of those that are not running."""
        live = self.requests
        for d in (self._glyd_cache, self._glyd_after, self._glyd_tries):
            for rid in [k for k in d if k not in live]:
                del d[rid]
        self._glyd_ever_held.intersection_update(live)
        for rid in [k for k in self._glyd_run if k not in live or live[k].status != RequestStatus.RUNNING]:
            del self._glyd_run[rid]

    def _glyd_hold(self):
        """Takes out of the waiting queues the requests the pool has no room for: [(request, queue, the request after it that stays)], in the order they were in. A request is let in (and its raw pages counted against the pool)
        when they fit what the running ones have taken and will take; after the first that does not fit, none that is predicted to take raw pages is let in before it (the queue's order), but the ones that take none are."""
        if self._glyd_predictor is None and not self._glyd_after:
            return []
        if not (self.waiting or self.skipped_waiting):  # (nothing to decide: what is kept for requests that are gone is dropped now and then)
            if len(self._glyd_cache) > 2 * len(self.requests) + 64:
                self._glyd_forget()
            return []
        self._glyd_forget()
        free = self.max_num_running_reqs - len(self.running) - getattr(self, "num_waiting_for_streaming_input", 0)
        if free <= 0:
            return []
        lumps, taken = self._glyd_store()
        in_use = 0
        if self.running:  # (nothing running: whatever the status counts is not in use by a request that can be waited for)
            in_use = taken
            for rid, p in self._glyd_run.items():
                r = self.requests[rid]
                in_use += p.remaining(max(0, r.num_computed_tokens - getattr(r, "num_in_flight_tokens", 0)))
        avail, blocked, held, passed, lim = lumps - in_use, False, [], 0, 4 * free + 64
        for q in (self.skipped_waiting, self.waiting):  # (the order vLLM takes them in)
            items = _head(q, lim + 1)  # (the requests this step can reach, and the one after them: where those held at their end go back)
            pending = []  # held, with no request after them that stays yet
            for r in items[:lim]:
                if passed >= free:
                    break
                if r.status not in WAITING:
                    continue
                hold = self._glyd_wait(r)
                p = self._glyd_pred(r) if self._glyd_predictor is not None else None
                if p is not None:
                    if p.total > vllm_admit.NEVER * lumps:
                        self._glyd_refuse(r, p)
                        continue
                    need = min(p.total, lumps)
                    if hold or blocked or need > avail:
                        hold = True
                    else:
                        avail -= need
                        self._glyd_run[r.request_id] = p
                if hold:
                    blocked = True
                    held.append([r, q, None])
                    pending.append(held[-1])
                else:
                    passed += 1
                    for h in pending:
                        h[2] = r
                    pending.clear()
            else:
                r = items[lim] if len(items) > lim else None
            for h in pending:
                h[2] = r  # (the request the scan stopped at, or the one after the last it looked at: None at the end of the queue)
        for r, q, _ in held:
            q.remove_request(r)
        for r, _, _ in held:
            if r.request_id not in self._glyd_ever_held:
                self._glyd_ever_held.add(r.request_id)
                self.glyd_stats["held"] += 1
        return held

    def _glyd_wait(self, r):
        """Whether the request waits for others to finish (it caused an exhaustion while they ran, or came after one that did): the ones of them not finished yet."""
        after = self._glyd_after.get(r.request_id)
        if not after:
            return False
        after = {x for x in after if x in self.requests}
        if after:
            self._glyd_after[r.request_id] = after
            return True
        del self._glyd_after[r.request_id]
        return False

    def _glyd_refuse(self, r, p):
        """A prompt that could never fit the reserve, even alone, that the API server did not refuse: ended with the error finish (the output of this step carries it)."""
        failed = self.finish_requests([r.request_id], RequestStatus.FINISHED_ERROR)
        self._glyd_errors += failed
        self.glyd_stats["refused"] += len(failed)
        log.error("glyd kv: request %s ended with an error: its %d tokens are predicted to need %d of the lossless cache's second extra memory, which holds %d", r.request_id, r.num_tokens, p.total, self._glyd_store()[0])

    def _glyd_unhold(self, held):
        """The held requests back in their queues where they were: before the request that was after them (the first of those still queued: else at the front), at the end if none was. A priority queue puts them in its order."""
        by_queue = {}
        for h in held:
            by_queue.setdefault(id(h[1]), (h[1], []))[1].append(h)
        for q, hs in by_queue.values():
            if not hasattr(q, "insert"):  # (a priority queue: its own order)
                for r, _, _ in hs:
                    q.add_request(r)
                continue
            gone = []
            for r, _, anchor in hs:
                if anchor is None:
                    q.append(r)
                    continue
                try:
                    q.insert(q.index(anchor), r)
                except ValueError:  # (the one it was before was scheduled: it goes to the front)
                    gone.append(r)
            for r in reversed(gone):
                q.appendleft(r)

    # ---- after a step --------------------------------------------------------------------------------------------------------------
    def update_from_output(self, scheduler_output, model_runner_output):
        for m in vllm_kv._MONITORS.values():
            self._glyd_pending |= {p // self._glyd_ratio for p in m.second_new()}  # (the ring names pages: the blocks they are in)
        self._glyd_exhausted(self._glyd_pending)
        out = super().update_from_output(scheduler_output, model_runner_output)
        self._glyd_release(self._glyd_pending)
        errors, self._glyd_errors = self._glyd_errors, []
        for r in errors:
            eco = out.get(r.client_index)
            if eco is None:
                eco = out[r.client_index] = EngineCoreOutputs()
            eco.outputs.append(EngineCoreOutput(request_id=r.request_id, new_token_ids=[], finish_reason=FinishReason.ERROR, stop_reason=MESSAGE))
        return out

    def _glyd_exhausted(self, pending):
        """The requests that hold a page the pools had no room for (a block the status ring names; every running request where the ring overflowed): preempted, and held back until the requests that were running with them have
        finished, one after another; one that is alone twice, or that has caused MAX_TRIES, ends with an error. Their blocks are out of the prefix cache, and handed over to be cleared once they are free."""
        blocks, overflow = set(), False
        for m in vllm_kv._MONITORS.values():
            b, o = m.lost()
            blocks |= b
            overflow |= o
        blocks = {b // self._glyd_ratio for b in blocks if b >= 0}  # (the rings name pages: the blocks they are in)
        if not blocks and not overflow:
            return
        pending |= blocks
        hit = []
        for r in self.running:
            try:
                held = [b for g in self.kv_cache_manager.get_block_ids(r.request_id) for b in g]
            except KeyError:
                continue
            if overflow or not blocks.isdisjoint(held):
                hit.append(r)
        if blocks:
            self.kv_cache_manager.evict_blocks(blocks)
        ids = {r.request_id for r in hit}
        rest = [r.request_id for r in self.running if r.request_id not in ids]  # (running, with no lost page)
        others = {rid for rid in rest if rid in self._glyd_run} or set(rest)  # (those of them predicted to hold some of the reserve, else any: the reserve is held by one that was not predicted)
        fail, wait = [], []
        for r in hit:
            t = self._glyd_tries[r.request_id] = self._glyd_tries.get(r.request_id, 0) + 1
            (fail if t > MAX_TRIES or (t >= 2 and len(hit) == 1 and not others) else wait).append(r)
        if fail:
            failed = self.finish_requests([r.request_id for r in fail], RequestStatus.FINISHED_ERROR)
            self._glyd_errors += failed
            vllm_kv.LOST[0] += len(failed)
            self.glyd_stats["failed"] += len(failed)
            log.error("glyd kv: %d requests ended with an error: they exhausted the lossless cache's second extra memory and are the only ones that hold any of it (or have done so %d times)", len(failed), MAX_TRIES)
        if wait:
            now = time.monotonic()
            gone = {r.request_id for r in wait}
            self.running = [r for r in self.running if r.request_id not in gone]
            for i, r in enumerate(wait):
                self._glyd_after[r.request_id] = others | {x.request_id for x in wait[:i]}
            for r in reversed(wait):  # (the first of them first in the queue)
                self._preempt_request(r, now, drop_stale_output=True)
                self.prev_step_scheduled_req_ids.discard(r.request_id)
                self._glyd_run.pop(r.request_id, None)
            self.glyd_stats["preempted"] += len(wait)
            log.warning("glyd kv: the lossless cache's second extra memory was full: %d requests preempted (%d blocks lost%s), they run again one after another as the requests that hold it finish", len(wait), len(blocks), ", more than are tracked: every running request" if overflow else "")
        if overflow:  # (the ring did not name every block: any cached block may hold a lost page, so none may be hit again)
            self.kv_cache_manager.reset_prefix_cache()

    def _glyd_release(self, pending):
        """The blocks in `pending` that no request holds now: out of the prefix cache (so that no request can hit a cleared page) and handed over to be cleared."""
        if not pending:
            return
        blocks = self.kv_cache_manager.block_pool.blocks
        pending.difference_update({b for b in pending if not 0 < b < len(blocks)})
        free = {b for b in pending if blocks[b].ref_cnt == 0}
        if free:
            pending -= free
            self.kv_cache_manager.evict_blocks(free)
            vllm_kv.release(self._glyd_leaves(sorted(free)))


def _head(q, n):
    """The first n requests of a queue, in its order."""
    out = []
    for r in q:
        out.append(r)
        if len(out) >= n:
            break
    return out


def make(*args, vllm_config=None, **kwargs):
    """The engine's scheduler (SchedulerConfig.scheduler_cls names this function): vLLM's own for the config, or the class the user asked for, with LostPages."""
    vc = vllm_config if vllm_config is not None else kwargs["vllm_config"]
    base = (vc.additional_config.get("glyd") or {}).get("scheduler_cls")
    if base:
        mod, _, name = base.rpartition(".")
        cls = getattr(importlib.import_module(mod), name)
    elif vc.scheduler_config.async_scheduling:
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler as cls
    else:
        from vllm.v1.core.sched.scheduler import Scheduler as cls
    vllm_kv._STATE.scheduler = True
    return type("Glyd" + cls.__name__, (LostLeaves, cls), {})(*args, vllm_config=vc, **kwargs)
