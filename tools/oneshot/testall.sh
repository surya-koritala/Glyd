#!/bin/bash
# Every reply in a PC folder through the browser test: testall.sh REMOTE_DIR LOCAL_DIR
set -u
D=$(cd "$(dirname "$0")" && pwd); R=$1; L=$2; mkdir -p "$L"
scp -q "glyd-pc:$R/*.json" "$L/" 2>/dev/null
for j in "$L"/*.json; do
  n=$(basename "$j" .json); [ -f "$L/$n.result" ] && continue
  python3 -c "
import json,re,sys
c=json.load(open('$j'))['choices'][0]['message']['content']; m=re.search(r'\`\`\`html\n(.*?)\`\`\`',c,re.S)
open('$L/$n.html','w').write(m.group(1) if m else ''); sys.exit(0 if m else 1)" || { echo "$n: no complete html block" > "$L/$n.result"; continue; }
  echo "$n: $(node "$D/gametest.js" "$L/$n.html" "$L/shots-$n" 2>&1 | tail -1)" > "$L/$n.result"
done
cat "$L"/*.result
