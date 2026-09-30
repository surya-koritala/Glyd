"""check_vllm.py's child with exact mode's refusal under torch.compile lifted, for one experiment: does exact mode
compiled give vLLM's compiled bf16 bits when inductor runs deterministic? argv: SPEC.json-string OUT.json CHECK_DIR"""
import json
import sys

import glyd.gpu.vllm_plugin as vp
from vllm.config import get_current_vllm_config_or_none

resolve = vp.GlydConfig.resolve


def lifted(self):
    vc = get_current_vllm_config_or_none()
    eager, vc.model_config.enforce_eager = vc.model_config.enforce_eager, True  # (the refusal's test alone)
    try:
        resolve(self)
    finally:
        vc.model_config.enforce_eager = eager


vp.GlydConfig.resolve = lifted
sys.path.insert(0, sys.argv[3])
import check_vllm  # noqa: E402

json.dump(check_vllm.child(json.loads(sys.argv[1])), open(sys.argv[2], "w"))
