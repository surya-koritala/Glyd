#!/bin/bash
# Every 2 min for up to 8 h: an 8x A100 80 GB ($22.32/h; the cheaper of the 8-GPU boxes, and the last run's) with capacity, launched once; then wait for its IP, upload the bundle (kdc12/: the engine and
# GPU library built for this branch, job.sh, inner.sh, the requests) to ~/dc, start job.sh, pull ~/w/out every 2 min into out/, terminate at DONE or after
# 2 h 55 min, and a terminator of its own at 3 h however this script ends (owner 2026-10-22: mind the credits). The key is read from ~/.lambda/api_key per call, never printed. launched.txt: the launch's answer.
S=$(cd "$(dirname "$0")" && pwd); cd "$S"; B=${B:?the bundle directory}
api() { curl -s -u "$(tr -d '[:space:]' < ~/.lambda/api_key):" "$@"; }
I=""
for i in $(seq 1 240); do
  pick=$(api https://cloud.lambdalabs.com/api/v1/instance-types | python3 -I -c "
import json,sys
d=json.load(sys.stdin)['data']
for k in ['gpu_8x_a100_80gb_sxm4']:
    r=[x['name'] for x in d.get(k,{}).get('regions_with_capacity_available',[])]
    if r: print(k, r[0]); break")
  if [ -n "$pick" ]; then
    set -- $pick
    out=$(api -H 'Content-Type: application/json' -d "{\"region_name\":\"$2\",\"instance_type_name\":\"$1\",\"ssh_key_names\":[\"glyd-lambda\"],\"quantity\":1,\"name\":\"glyd-tp-ll\"}" https://cloud.lambdalabs.com/api/v1/instance-operations/launch)
    echo "$(date '+%m-%d %H:%M %Z') $1 $2 $out" >> launched.txt
    I=$(echo "$out" | grep -o '"instance_ids":\["[0-9a-f]*' | grep -o '[0-9a-f]*$')
    [ -n "$I" ] && break
  fi
  sleep 120
done
[ -n "$I" ] || { echo "$(date '+%m-%d %H:%M %Z') no capacity in 8 h" >> launched.txt; exit 3; }
echo "$(date +%T) instance $I ($1)"
# (the 3 h stop whatever happens here: its own process, the key read when it fires)
nohup bash -c "sleep 10800; curl -s -u \"\$(tr -d '[:space:]' < ~/.lambda/api_key):\" -X POST https://cloud.lambdalabs.com/api/v1/instance-operations/terminate -H 'Content-Type: application/json' -d '{\"instance_ids\":[\"$I\"]}' > /dev/null; echo \"\$(date +%T) deadline terminate $I\"" >> deadline.log 2>&1 < /dev/null &
echo $! > deadline.pid
done_() { api -X POST https://cloud.lambdalabs.com/api/v1/instance-operations/terminate -H "Content-Type: application/json" -d "{\"instance_ids\":[\"$I\"]}" | python3 -I -c "import sys,json; d=json.load(sys.stdin); print('$(date +%T) terminated:', [i['id'][:8]+' '+i['status'] for i in d['data']['terminated_instances']])"; exit "${1:-0}"; }
T0=$(date +%s); IP=""
while true; do
  read -r IP ST < <(api "https://cloud.lambdalabs.com/api/v1/instances/$I" | python3 -I -c 'import sys,json; d=json.load(sys.stdin)["data"]; print(d.get("ip") or "-", d.get("status"))' 2>/dev/null)
  [ "$ST" = active ] && [ "$IP" != "-" ] && break
  [ $(( $(date +%s) - T0 )) -gt 1200 ] && { echo "not active in 20 min ($ST)"; done_ 4; }
  sleep 20
done
echo "$IP" > ip.txt
K="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o ServerAliveInterval=60 -o LogLevel=ERROR -o BatchMode=yes -i $HOME/.ssh/glyd-lambda"
for _ in $(seq 1 60); do $K "ubuntu@$IP" true < /dev/null 2>/dev/null && break; sleep 10; done
echo "$(date +%T) ssh $IP"
rsync -a -e "$K" "$B/" "ubuntu@$IP:dc/" || { echo "upload failed"; done_ 1; }
$K "ubuntu@$IP" "chmod +x ~/dc/*.sh ~/dc/glyd-engine; nohup bash ~/dc/job.sh > ~/job.out 2>&1 < /dev/null &" < /dev/null > /dev/null 2>&1 &
sleep 15; kill $! 2>/dev/null
echo "$(date +%T) job started"
mkdir -p out
while true; do
  sleep 120
  rsync -a -e "$K" "ubuntu@$IP:w/out/" out/ 2>/dev/null
  [ -e out/DONE ] && break
  [ $(( $(date +%s) - T0 )) -gt 10500 ] && { echo "2 h 55 min cap"; break; }
done
echo "$(date +%T) results: $(ls out | wc -l) files"
done_ 0
