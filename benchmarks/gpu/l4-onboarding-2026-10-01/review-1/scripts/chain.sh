#!/bin/bash
# the review round: a smoke run (Qwen3-1.7B), then the 24 GB, 16 GB and 8 GB acceptance runs, one after the other under the box lock
~/onb/acc.sh smoke3 ~/accept-onb/work-smoke --wheel /home/ubuntu/onb/wheels/glyd-0.26.0rc3-py3-none-manylinux_2_28_x86_64.whl --rust-glyd /home/ubuntu/onb/glyd-rust --model Qwen/Qwen3-1.7B --webui none
~/onb/acc.sh a24 ~/accept-onb/work-a24 --wheel /home/ubuntu/onb/wheels/glyd-0.26.0rc3-py3-none-manylinux_2_28_x86_64.whl --rust-glyd /home/ubuntu/onb/glyd-rust --webui none
~/onb/acc.sh a16 ~/accept-onb/work-a24 --wheel /home/ubuntu/onb/wheels/glyd-0.26.0rc3-py3-none-manylinux_2_28_x86_64.whl --rust-glyd /home/ubuntu/onb/glyd-rust --card 4080s --webui all --pip-refusal
~/onb/acc.sh a8 ~/accept-onb/work-a24 --wheel /home/ubuntu/onb/wheels/glyd-0.26.0rc3-py3-none-manylinux_2_28_x86_64.whl --rust-glyd /home/ubuntu/onb/glyd-rust --card 8gb
