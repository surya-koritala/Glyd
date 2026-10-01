# Diagnostic only (not shipped): GLYD_PROBE_GEFORCE=1 the plugin reads this GPU as GeForce Ada; GLYD_PROBE=1 each packed
# module's memory while the weights load (torch allocated / reserved, the driver's free, the allocator's retries and OOMs);
# GLYD_PROBE_SNAP=1 the allocator's blocks when vLLM measures "Model loading took" (slack: block size less requested);
# GLYD_PROBE_EMPTY=1 torch.cuda.empty_cache() after each packed module (an experiment of the fix).
import importlib.abc
import importlib.machinery
import os
import sys

E = os.environ.get


def _mem(tag):
    import torch

    s = torch.cuda.memory_stats()
    free = torch.cuda.mem_get_info()[0] >> 20
    print(f"[probe] {tag}: alloc {torch.cuda.memory_allocated() >> 20} MiB, reserved {torch.cuda.memory_reserved() >> 20}, "
          f"free {free}, retries {s.get('num_alloc_retries', 0)}, ooms {s.get('num_ooms', 0)}, "
          f"peak alloc {torch.cuda.max_memory_allocated() >> 20}, peak reserved {torch.cuda.max_memory_reserved() >> 20}",
          file=sys.stderr, flush=True)


def _snap(tag):
    import collections
    import torch

    n = slack = act = req = 0
    by = collections.Counter()
    segs = collections.Counter()
    for seg in torch.cuda.memory_snapshot():
        segs[(seg["segment_pool_id"] != (0, 0), seg.get("is_expandable", False))] += 1
        for b in seg["blocks"]:
            if b["state"] == "active_allocated":
                n += 1
                act += b["size"]
                r = b.get("requested_size", b["size"])
                req += r
                slack += b["size"] - r
                by[(b["size"] >> 20, (b["size"] - r) >> 10)] += 1
    print(f"[probe] {tag}: {n} live blocks, {act / 2**20:.1f} MiB in blocks, {req / 2**20:.1f} MiB requested, slack {slack / 2**20:.1f} MiB",
          file=sys.stderr, flush=True)
    free = collections.Counter()
    for seg in torch.cuda.memory_snapshot():
        for b in seg["blocks"]:
            if b["state"] == "inactive":
                free[b["size"] >> 20] += 1
    tot = sum(k * v for k, v in free.items())
    print(f"[probe] cached free blocks: {sum(free.values())} blocks, {tot} MiB; by MiB x count: " + ", ".join(f"{k}x{v}" for k, v in sorted(free.items(), key=lambda kv: -kv[0] * kv[1])[:14]), file=sys.stderr, flush=True)
    top = sorted(by.items(), key=lambda kv: -kv[0][1] * kv[1])[:12]
    print("[probe] (block MiB, slack KiB) x count: " + ", ".join(f"({a},{b})x{c}" for (a, b), c in top), file=sys.stderr, flush=True)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in ("glyd.gpu.vllm_plugin", "vllm.utils.mem_utils", "glyd.gpu.kernels"):
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None:
            return None
        run = spec.loader.exec_module

        def exec_module(module):
            run(module)
            if name == "glyd.gpu.kernels":
                if E("GLYD_PROBE_CHUNKS"):  # "hist,pack": the packers' chunk sizes (an experiment)
                    hc, pc = (int(x) for x in E("GLYD_PROBE_CHUNKS").split(","))
                    module.HIST_CHUNK, module.PACK_CHUNK = hc, pc
                    module.pack_mma.__defaults__ = (None, pc)
                    print(f"glyd test: HIST_CHUNK {hc}, PACK_CHUNK {pc}", file=sys.stderr, flush=True)
                return
            if name == "vllm.utils.mem_utils":
                if E("GLYD_PROBE_SNAP"):
                    cls = module.DeviceMemoryProfiler
                    exit0 = cls.__exit__

                    def exit1(self, *a):
                        _snap("at the end of loading")
                        _mem("at the end of loading")
                        return exit0(self, *a)

                    cls.__exit__ = exit1
                return
            if E("GLYD_PROBE_GEFORCE"):
                module._gpu = lambda d: 1089
                print("glyd test: the plugin reads this GPU as GeForce Ada (1089)", file=sys.stderr, flush=True)
            if E("GLYD_PROBE") or E("GLYD_PROBE_EMPTY"):
                for cls in (module.GlydLinearMethod, module.GlydMoEMethod):
                    f = cls.process_weights_after_loading

                    def wrap(self, layer, f=f):
                        import torch

                        name = getattr(layer, "prefix", None) or getattr(layer, "layer_name", "?")
                        r = f(self, layer)
                        if E("GLYD_PROBE_EMPTY"):
                            torch.cuda.empty_cache()
                        if E("GLYD_PROBE"):
                            _mem(f"packed {name}")
                        return r

                    cls.process_weights_after_loading = wrap

        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _Finder())
