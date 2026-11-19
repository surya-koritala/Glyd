#!/bin/bash
# One token-link profile Mac(worker) -> PC(coordinator): restart both daemons, approve, run, save record, print stats.
set -euo pipefail
name=$1 basis=$2 gap=$3 keep=$4
W=~/glyd-compute/target/release/starlings; S=~/starlings-link
B=/home/surya-koritala/glyd-scratch/starlings/target/release/starlings; C=/home/surya-koritala/glyd-scratch/starlings/cluster/coord
WN=d45c43fd-1bf3-47b5-a881-05bc450bbd69
L="\"limits\":{\"version\":1,\"payload_bytes\":16384,\"ping_samples\":100,\"inter_ping_gap_ms\":$gap,\"minimum_interval_ms\":30000,\"token_profile\":{\"gap_basis\":\"$basis\",\"keepalive_ms\":$keep}}"
$W --state-dir $S/worker cluster stop >/dev/null 2>&1 || true
ssh glyd-pc "$B --state-dir $C cluster stop >/dev/null 2>&1 || true; sleep 1; umask 077; (setsid nohup $B --state-dir $C daemon >> $C/../coord.log 2>&1 < /dev/null &); sleep 2
echo '{\"version\":1,\"listen\":\"192.168.2.11:0\",$L}' > $C/probe-server.json
d=\$($B --state-dir $C cluster probe-digest $C/probe-server.json | tail -1)
$B --state-dir $C cluster probe-listen $C/probe-server.json --approve \$d" > $S/listen.out
port=$(grep -oE ':[0-9]+$' $S/listen.out | tr -d :)
umask 077
(nohup $W --state-dir $S/worker daemon >> $S/worker.log 2>&1 < /dev/null &); sleep 4
echo "{\"version\":1,\"destination\":\"192.168.2.11:$port\",\"source\":null,$L}" > $S/worker/probe-route.json
d=$($W --state-dir $S/worker cluster probe-digest $S/worker/probe-route.json | tail -1)
$W --state-dir $S/worker cluster probe $S/worker/probe-route.json --approve $d
for i in $(seq 1 40); do sleep 1; $W --state-dir $S/worker cluster links $WN 2>&1 | grep -q '"completed_samples":100\|"error":"[a-z]' && break; done
$W --state-dir $S/worker cluster links-record $WN $S/records/$name.jsonl
python3 - $S/records/$name.jsonl <<'PY'
import json,sys
r=json.loads(open(sys.argv[1]).readline())
def find(o):
    if isinstance(o,dict):
        if isinstance(o.get('samples'),list): return o
        for v in o.values():
            x=find(v)
            if x: return x
    if isinstance(o,list):
        for v in o:
            x=find(v)
            if x: return x
t=find(r); s=t['samples']
rtt=sorted(int(x['rtt_us']) for x in s if x.get('rtt_us') is not None)
n=len(rtt)
print(sys.argv[1].split('/')[-1], 'n',n,'p50 %.2f p90 %.2f p99 %s max %.2f mean %.2f ms'%(rtt[n//2]/1e3, rtt[int(n*.9)]/1e3, '%.2f'%(rtt[98]/1e3) if n>=100 else 'null', rtt[-1]/1e3, sum(rtt)/n/1e3), 'error', t.get('error'))
PY
