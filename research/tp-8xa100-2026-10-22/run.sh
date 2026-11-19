#!/bin/bash
# The 8-GPU run once poll10.sh has launched it (its id in launched11.txt): wait for an IP, upload kdc11/ to ~/dc, start job.sh, pull ~/w/out every 2 min
# (out11/), terminate at DONE or after 4 h. The key is read from ~/.lambda/api_key per call, never printed.
S=$(cd "$(dirname "$0")" && pwd); cd "$S"
api() { curl -s -u "$(tr -d '[:space:]' < ~/.lambda/api_key):" "$@"; }
I=$(grep -o '"instance_ids":\["[0-9a-f]*' launched11.txt | tail -1 | grep -o '[0-9a-f]*$')
[ -n "$I" ] || { echo "no instance id in launched11.txt"; exit 4; }
echo "$(date +%T) instance $I"
done_() { api -X POST https://cloud.lambdalabs.com/api/v1/instance-operations/terminate -H "Content-Type: application/json" -d "{\"instance_ids\":[\"$I\"]}" | python3 -I -c "import sys,json; d=json.load(sys.stdin); print('$(date +%T) terminated:', [i['id'][:8]+' '+i['status'] for i in d['data']['terminated_instances']])"; exit "${1:-0}"; }
T0=$(date +%s); IP=""
while true; do
  read -r IP ST < <(api "https://cloud.lambdalabs.com/api/v1/instances/$I" | python3 -I -c 'import sys,json; d=json.load(sys.stdin)["data"]; print(d.get("ip") or "-", d.get("status"))' 2>/dev/null)
  [ "$ST" = active ] && [ "$IP" != "-" ] && break
  [ $(( $(date +%s) - T0 )) -gt 1200 ] && { echo "not active in 20 min ($ST)"; done_ 4; }
  sleep 20
done
echo "$IP" > kimi11-ip.txt
SSH=(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o ServerAliveInterval=60 -o LogLevel=ERROR -o BatchMode=yes -i ~/.ssh/glyd-lambda "ubuntu@$IP")
for _ in $(seq 1 60); do "${SSH[@]}" true < /dev/null 2>/dev/null && break; sleep 10; done
echo "$(date +%T) ssh $IP"
rsync -a -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o BatchMode=yes -i $HOME/.ssh/glyd-lambda" kdc11/ "ubuntu@$IP:dc/" || { echo "upload failed"; done_ 1; }
"${SSH[@]}" "chmod +x ~/dc/*.sh ~/dc/glyd-engine; nohup bash ~/dc/job.sh > ~/job.out 2>&1 < /dev/null &" < /dev/null > /dev/null 2>&1 &
sleep 15; kill $! 2>/dev/null
echo "$(date +%T) job started"
mkdir -p out11
while true; do
  sleep 120
  rsync -a -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o BatchMode=yes -i $HOME/.ssh/glyd-lambda" "ubuntu@$IP:w/out/" out11/ 2>/dev/null
  [ -e out11/DONE ] && break
  [ $(( $(date +%s) - T0 )) -gt 14400 ] && { echo "4 h cap"; break; }
done
echo "$(date +%T) results: $(ls out11 | wc -l) files"
done_ 0
