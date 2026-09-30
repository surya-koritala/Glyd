# The test's GeForce emulation, in every Python process started with this directory on PYTHONPATH: Glyd's vLLM plugin
# reads the GPU as GeForce Ada (code 1089: glyd_gpu.h's GLYD_GPU_GEFORCE + 89) instead of the library's own reading of
# the device's name (an L4: 3089). The plugin takes its routes by that code (a prompt past 512 tokens decoded ahead for
# cuBLAS, as on an RTX 4080) and sizes its scratch buffer by them at load; the library's own kernel choices stay the L4's.
import importlib.abc
import importlib.machinery
import sys


class _GeForce(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "glyd.gpu.vllm_plugin":
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None:
            return None
        run = spec.loader.exec_module

        def exec_module(module):
            run(module)
            module._gpu = lambda d: 1089
            print("glyd test: the plugin reads this GPU as GeForce Ada (1089)", file=sys.stderr, flush=True)

        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _GeForce())
