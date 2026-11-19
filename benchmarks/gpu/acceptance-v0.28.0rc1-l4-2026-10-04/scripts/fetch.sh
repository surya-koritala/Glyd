#!/bin/bash
# fetch.sh [NAME...]: the runs' results from the box into rel028acc/runs/NAME (the files the box named .log are .txt here, as in the v0.27.0 record)
S=/private/tmp/claude-501/-Users-suryakoritala-alatirok/a9feedc0-e067-4d76-bca6-e4b6b9538345/scratchpad
IP=$(bash $S/devbox status | awk '{print $2}')
mkdir -p $S/rel028acc/runs
for n in "$@"; do
  rm -rf $S/rel028acc/runs/$n
  ssh -i ~/.ssh/glyd-dev.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$IP "tar -C ~/accept-028/results -cf - $n" | tar -C $S/rel028acc/runs -xf - 2>/dev/null
  scp -q -i ~/.ssh/glyd-dev.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$IP:accept-028/$n.out $S/rel028acc/runs/$n/acceptance.out 2>/dev/null
  for f in $(find $S/rel028acc/runs/$n -name '*.log'); do mv "$f" "${f%.log}.txt"; done
  rm -f $S/rel028acc/runs/$n/logs/page.html $S/rel028acc/runs/$n/logs/webui.pid
  echo "$n: $(grep -c '^ok' $S/rel028acc/runs/$n/summary.txt) ok, $(grep -c '^FAIL' $S/rel028acc/runs/$n/summary.txt) FAIL, last: $(tail -1 $S/rel028acc/runs/$n/summary.txt | cut -c1-80)"
done
