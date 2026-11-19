#!/bin/bash
# One Lambda H100 SXM (or PCIe, or A100 SXM) for job.sh: launch, upload dc/ (lib + engine), job.sh and inner.sh, start, pull session.log each minute,
# terminate at DONE, after 15 min with no IP, or at 60 min. The key is read from ~/.lambda/api_key per call, never printed.
S=$(cd "$(dirname "$0")" && pwd); cd "$S"
api() { curl -s -u "$(tr -d '[:space:]' < ~/.lambda/api_key):" "$@"; }
I=""
for try in $(seq 1 30); do  # the regions with capacity, each in turn; capacity comes and goes, so up to 30 rounds a minute apart
  for TR in $(api https://cloud.lambda.ai/api/v1/instance-types | python3 -c 'import sys,json; d=json.load(sys.stdin)["data"]
for t in ["gpu_1x_h100_sxm5","gpu_1x_h100_pcie","gpu_1x_a100_sxm4"]:
  for x in d.get(t,{}).get("regions_with_capacity_available",[]): print(t+":"+x["name"])'); do
    T=${TR%%:*}; REG=${TR#*:}
    r=$(api -X POST https://cloud.lambda.ai/api/v1/instance-operations/launch -H "Content-Type: application/json" -d "{\"region_name\":\"$REG\",\"instance_type_name\":\"$T\",\"ssh_key_names\":[\"glyd-lambda\"],\"name\":\"moe-slots\"}")
    I=$(echo "$r" | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"]["instance_ids"][0])' 2>/dev/null)
    [ -n "$I" ] && { echo "$(date +%T) $T $REG"; break 2; }
    echo "$(date +%T) $T $REG: $(echo "$r" | python3 -c 'import sys,json; print(json.load(sys.stdin)["error"]["code"])' 2>/dev/null)"
  done
  sleep 60
done
[ -n "$I" ] || { echo "no instance in 30 rounds"; exit 4; }
echo "$(date +%T) instance $I" | tee instance.txt
done_() { api -X POST https://cloud.lambda.ai/api/v1/instance-operations/terminate -H "Content-Type: application/json" -d "{\"instance_ids\":[\"$I\"]}" | python3 -c "import sys,json; d=json.load(sys.stdin); print('$(date +%T) terminated:', [i['id'][:8]+' '+i['status'] for i in d['data']['terminated_instances']])"; exit "${1:-0}"; }
T0=$(date +%s); IP=""
while true; do
  read -r IP ST < <(api "https://cloud.lambda.ai/api/v1/instances/$I" | python3 -c 'import sys,json; d=json.load(sys.stdin)["data"]; print(d.get("ip") or "-", d.get("status"))' 2>/dev/null)
  [ "$ST" = active ] && [ "$IP" != "-" ] && break
  [ $(( $(date +%s) - T0 )) -gt 900 ] && { echo "not active in 15 min ($ST)"; done_ 4; }
  sleep 20
done
echo "$IP" > ip
SSH=(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o ServerAliveInterval=60 -o LogLevel=ERROR -o BatchMode=yes -i ~/.ssh/glyd-lambda "ubuntu@$IP")
for _ in $(seq 1 40); do "${SSH[@]}" true < /dev/null 2>/dev/null && break; sleep 10; done
echo "$(date +%T) ssh $IP"
"${SSH[@]}" "mkdir -p ~/dc ~/w" < /dev/null
for f in dc/libglyd_gpu_cuda13.so dc/glyd-engine job.sh inner.sh; do "${SSH[@]}" "cat > ~/$f" < "$f" || { echo "upload $f failed"; done_ 1; }; done
"${SSH[@]}" "chmod +x ~/job.sh ~/inner.sh ~/dc/glyd-engine; nohup bash ~/job.sh > ~/job.out 2>&1 < /dev/null &" < /dev/null > /dev/null 2>&1 &
sleep 15; kill $! 2>/dev/null
echo "$(date +%T) job started"
while true; do
  sleep 60
  "${SSH[@]}" "cat ~/w/out/session.log 2>/dev/null | tail -n 5; ls ~/w/out" < /dev/null > run.log 2>/dev/null
  grep -q "^DONE$" run.log && break
  [ $(( $(date +%s) - T0 )) -gt 3600 ] && { echo "60 min cap"; break; }
done
"${SSH[@]}" "cd ~/w && tar czf - out" < /dev/null > results.tgz 2>/dev/null
echo "$(date +%T) results: $(tar tzf results.tgz 2>/dev/null | wc -l) files"
done_ 0
