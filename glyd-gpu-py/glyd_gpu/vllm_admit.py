"""A prompt the lossless KV cache could never hold is refused where vLLM validates its input, with a client error (400) and why, as one longer than the model's context is; the scheduler (vllm_sched) keeps the others waiting for room.

Each process that takes requests decides on its own, from what it can read itself: the model's tables (kvcal: the cache directory, by the model and the tables' version, which every process of a host sees) and the size of the KV cache vLLM
reports to it once the engine is up. Nothing is handed over between processes: any number of API servers, in any environment, refuse the same prompts. Where it cannot read the tables, one line is logged and the engine's scheduler ends such a prompt
with an error finish instead.

vLLM's internals it relies on: v1.engine.input_processor.InputProcessor._validate_model_input (wrapped, in every process: after vLLM's own checks), its vllm_config (additional_config, cache_config.num_gpu_blocks and block_size),
and vllm.exceptions.VLLMValidationError (the 400).
Under the Apache License 2.0 (LICENSE), as the rest of Glyd."""
import functools
import os

from vllm.logger import init_logger

from . import kvcal

log = init_logger("vllm.glyd")

NEVER = 1.5  # a prompt is refused where what it will need is over this many times what the cache holds (the prediction is 0.5 to 1.2 times what is taken on the models measured)
MESSAGE = ("glyd kv: this prompt needs more room in the lossless KV cache than it has ({need:,} against {room:,}): "
           "send a shorter prompt, or serve with kv off")
_PRED = {}  # tables path -> its kvcal.Predictor (None: the file has no sweep of exceptions)


def bay(heads):
    """The block's own spare as the engine's process has it (GLYD_KV_SPARE's, else the default)."""
    from . import vllm_kv

    a = ""
    # dev: GLYD_KV_SPARE
    a = os.environ.get("GLYD_KV_SPARE", "").strip()
    # end dev
    return int(a) if a else vllm_kv.free_bay(heads)


def predictor(tables):
    """The kvcal.Predictor of a loaded kvcal.Tables (made once a file); None where its sweep has no per-token figures (a file of an earlier version) or the model has no token that needs the second reserve."""
    from . import vllm_kv

    if tables is None:
        return None
    p = _PRED.get(tables.path, False)
    if p is False:
        meta = tables.meta
        p = None
        if tables.exc is None:
            log.warning("glyd kv: the tables %s were made by an earlier version: requests are not held back for room before they run, only preempted where the cache runs out; make new tables "
                        "(python -m glyd_gpu kvcal MODEL)", tables.path)
        if tables.exc is not None and meta.get("heads"):
            p = kvcal.Predictor(tables.exc, meta["heads"], bay(meta["heads"]) + vllm_kv.ND * vllm_kv.CH)
            if not p.active:
                p = None
        _PRED[tables.path] = p
    return p


def room(vc, meta):
    """What the engine's second reserve holds, from what a process that takes requests knows: the share the tables set (GLYD_KV_RESERVE_2's if set) of the KV cache, which vLLM has sized and reports (cache_config.num_gpu_blocks).
    The engine sized it before vLLM's memory profile, from an estimate of the cache: this is within a few percent of its figure (-9% to +7% on the five models measured). GLYD_KV_RESERVE_2_UNITS names the size itself. None where the cache's size is not known yet."""
    from . import vllm_kv

    # dev: GLYD_KV_RESERVE_2_UNITS names the size itself
    v = os.environ.get("GLYD_KV_RESERVE_2_UNITS", "").strip()
    if v:
        return int(v)
    # end dev
    cc = vc.cache_config
    if not cc.num_gpu_blocks:
        return None
    H, L = meta["heads"], max(1, int(meta.get("layers", 1)))
    pct = vllm_kv.second_percent(meta)
    # dev: GLYD_KV_RESERVE_2
    pct = float(os.environ.get("GLYD_KV_RESERVE_2", "").strip() or pct)
    # end dev
    if pct <= 0:
        return 0
    kv_bytes = cc.num_gpu_blocks * max(1, cc.block_size // vllm_kv.PT) * L * vllm_kv.leaf_bytes(H, bay(H))
    return max(vllm_kv.SECOND_MIN, int(pct / 100.0 * kv_bytes / (H * 2048)))


def front(ip):
    """(the predictor, the tables' meta) for an InputProcessor's server, made once; None where kv lossless is off for it or nothing can be predicted (the model has no token that needs the second reserve, the tables cannot be read: logged)."""
    st = ip.__dict__.get("_glyd_front", False)
    if st is not False:
        return st
    st = None
    ac = (getattr(ip.vllm_config, "additional_config", None) or {}).get("glyd") or {}
    path = (ac.get("kv_tables") or {}).get("path")
    off = False
    # dev: GLYD_KV_ADMIT=off turns the prediction off (for measuring)
    off = os.environ.get("GLYD_KV_ADMIT", "").strip().lower() == "off"
    # end dev
    if path and not off:
        try:
            t = kvcal.Tables(path)
            p = predictor(t)
            st = (p, t.meta) if p is not None else None
        except Exception as e:
            log.warning("glyd kv: this process cannot read the KV tables %s (%s: %s): a prompt that could never fit the cache is ended by the engine with an error finish, not refused with a 400", path, type(e).__name__, e)
    ip.__dict__["_glyd_front"] = st
    return st


def check(ip, ids):
    """What a prompt (token ids) is predicted to need; refused (vllm.exceptions.VLLMValidationError) where that is over NEVER times what the cache holds. Nothing is refused where the cache's size is not known."""
    f = front(ip)
    if f is None:
        return 0
    p, meta = f
    need = int(p.blocks(ids).sum())
    if need:
        have = ip.__dict__.get("_glyd_room")
        if have is None:
            have = ip.__dict__["_glyd_room"] = room(ip.vllm_config, meta)
        if have is not None and need > NEVER * have:
            from vllm.exceptions import VLLMValidationError

            raise VLLMValidationError(MESSAGE.format(need=need, room=have), parameter="prompt", value=len(ids))
    return need


def install_front():
    """In every process: a prompt that could never fit is refused where vLLM validates the input, before it is queued (a server without kv lossless is not touched)."""
    from vllm.v1.engine.input_processor import InputProcessor

    orig = InputProcessor._validate_model_input
    if getattr(orig, "_glyd_checks", False):
        return True

    @functools.wraps(orig)
    def _validate_model_input(self, prompt_input, prompt_type):
        orig(self, prompt_input, prompt_type)
        if prompt_type == "decoder" and prompt_input.get("type") != "embeds" and prompt_input.get("prompt_token_ids") is not None:
            check(self, prompt_input["prompt_token_ids"])

    _validate_model_input._glyd_checks = True
    InputProcessor._validate_model_input = _validate_model_input
    return True
