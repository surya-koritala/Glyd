#!/bin/bash
# After m6_spec.sh, m6_bi.sh and m6_tight.sh: m6_spec.sh again (its runs done before skipped): the draft packed by Glyd,
# refused by the memory check before its fix (PyTorch's cached memory not counted free; a draft sized with the
# target's embeddings), run again.
while pgrep -f "m6_spec.sh|m6_bi.sh|m6_tight.sh" > /dev/null; do sleep 30; done
cd ~/vllm-work && bash m6_spec.sh
