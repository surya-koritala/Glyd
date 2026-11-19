#!/bin/bash
# every 2 min: an 8x H100 (or 8x A100 80 GB) with capacity, launched once; its id and type in launched11.txt
S=$(dirname "$0")
for i in $(seq 1 180); do
  pick=$(curl -s -u "$(cat ~/.lambda/api_key):" https://cloud.lambdalabs.com/api/v1/instance-types | python3 -I -c "
import json,sys
d=json.load(sys.stdin)['data']
for k in ['gpu_8x_h100_sxm5','gpu_8x_a100_80gb_sxm4']:
    r=[x['name'] for x in d.get(k,{}).get('regions_with_capacity_available',[])]
    if r: print(k, r[0]); break")
  if [ -n "$pick" ]; then
    set -- $pick
    out=$(curl -s -u "$(cat ~/.lambda/api_key):" -H 'Content-Type: application/json' -d "{\"region_name\":\"$2\",\"instance_type_name\":\"$1\",\"ssh_key_names\":[\"glyd-lambda\"],\"quantity\":1,\"name\":\"glyd-kimi-tp\"}" https://cloud.lambdalabs.com/api/v1/instance-operations/launch)
    echo "$(date '+%H:%M %Z') $1 $2 $out" >> $S/launched11.txt
    echo "$out" | grep -q instance_ids && exit 0
  fi
  sleep 120
done
echo "$(date '+%H:%M %Z') no capacity in 6 h" >> $S/launched11.txt
