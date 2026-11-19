#!/bin/bash
# the v0.28.0rc1 acceptance: the cli flow (no GPU: no lock), then the 16 GB card, the 24 GB card and the served flow one at a time, each under the lock on its own and a minute between them
R=$HOME/rel028/rust/glyd
REL=https://raw.githubusercontent.com/surya-koritala/Glyd/release-0.28.0/scripts/install.sh
W=/opt/dlami/nvme/rel028/work
mkdir -p ~/accept-028/results
touch ~/.glyd-busy
( cd ~/rel028/co/gpu/vllm && HF_HOME=$HOME/hf bash acceptance.sh --work /opt/dlami/nvme/rel028/work-cli --flow cli --rust-glyd $R > ~/accept-028/cli.out 2>&1; rc=$?
  rm -rf ~/accept-028/results/cli; mkdir -p ~/accept-028/results/cli; cp -r /opt/dlami/nvme/rel028/work-cli/summary.txt /opt/dlami/nvme/rel028/work-cli/logs ~/accept-028/results/cli/; echo "== cli rc=$rc" >> ~/accept-028/cli.out )
~/rel028/acc.sh 16gb-card $W --rust-glyd $R --install-url $REL --card 4080s --webui all --pip-refusal
sleep 60
~/rel028/acc.sh 24gb $W --rust-glyd $R --install-url $REL --webui none
sleep 60
~/rel028/acc.sh 16gb-card-served-script $W --rust-glyd $R --served --prerelease --install-url $REL --card 4080s --webui all --pip-refusal
echo CHAIN DONE > ~/accept-028/chain.done
pgrep -f '[f]lock .*glyd-box.lock' > /dev/null || rm -f ~/.glyd-busy
