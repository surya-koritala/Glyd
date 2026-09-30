"""vLLM's entry point for the "glyd" quantization method (vllm.general_plugins: glyd), in every process vLLM starts,
whatever it serves: where vLLM is the minor release the plugin is tested with, glyd.gpu.vllm_plugin.register(); else
one line in the log, and a "glyd" method that says why when it is asked for (--quantization glyd). It never raises.
Under the Business Source License 1.1 (LICENSE-glyd-gpu), as the rest of Glyd's GPU code."""
import logging

TESTED = "0.30"  # vLLM's minor release the plugin is tested with (glyd[vllm] pins it)


def tested(version):
    """Whether vLLM's version is of the minor release the plugin is tested with (0.30.0, 0.30.1rc1, 0.30.2.dev3 ...)."""
    return str(version).split("+")[0].split(".")[:2] == TESTED.split(".")


def register():
    log = logging.getLogger("vllm.glyd")
    try:
        import vllm

        version = getattr(vllm, "__version__", "?")
    except Exception as e:  # (no vLLM to register with)
        log.warning("glyd: vLLM did not import (%s: %s): the glyd quantization method is not registered", type(e).__name__, e)
        return
    if not tested(version):
        log.warning("glyd: the glyd quantization method is tested with vLLM %s, and this is vLLM %s: it is not loaded", TESTED, version)
        _refusing(f"the glyd quantization method is tested with vLLM {TESTED}; this is vLLM {version}")
        return
    try:
        from . import vllm_plugin

        vllm_plugin.register()
    except Exception as e:  # (a vLLM of the release whose internals moved, or a glyd without its library)
        log.warning("glyd: the glyd quantization method did not load in vLLM %s (%s: %s)", version, type(e).__name__, e)
        _refusing(f"the glyd quantization method did not load in vLLM {version} ({type(e).__name__}: {e})")


def _refusing(why):
    """A "glyd" quantization method that refuses with why when a model asks for it; nothing where vLLM's registry is
    not as 0.30 has it."""
    try:
        from vllm.model_executor.layers.quantization import register_quantization_config
        from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
    except Exception:
        return

    class Refused(QuantizationConfig):
        def get_name(self):
            return "glyd"

        def get_supported_act_dtypes(self):
            return []

        @classmethod
        def get_min_capability(cls):
            return 80

        @staticmethod
        def get_config_filenames():
            return []

        @classmethod
        def from_config(cls, config):
            raise ValueError(f'glyd: {why}: pip install "glyd[vllm]" for the vLLM it is tested with')

        def get_quant_method(self, layer, prefix):
            return None

    try:
        register_quantization_config("glyd")(Refused)
    except Exception:
        pass
