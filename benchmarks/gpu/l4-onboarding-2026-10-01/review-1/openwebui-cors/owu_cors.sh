#!/bin/bash
# S14: what Open WebUI 0.11.4 does with WEBUI_AUTH=False for a page from another origin and for a rebinding Host.
# CPU only (no model server: the connection points at a closed port). Output: ~/onb/owu-cors/*.txt
set -u
IMG=ghcr.io/open-webui/open-webui:v0.11.4
OUT=${1:-$HOME/onb/owu-cors}
mkdir -p "$OUT"
docker rm -f owu-a owu-b >/dev/null 2>&1

docker run --rm --entrypoint sh $IMG -c 'grep -rn "CORS_ALLOW_ORIGIN" /app/backend/open_webui/env.py /app/backend/open_webui/main.py /app/backend/open_webui/socket/main.py 2>/dev/null | head -40; echo ---; grep -n "WEBUI_AUTH\b" /app/backend/open_webui/env.py | head; echo ---; grep -rn "def signin" -A30 /app/backend/open_webui/routers/auths.py | grep -n "WEBUI_AUTH\|admin\|first" | head' > "$OUT/source.txt" 2>&1

run() { # name port extra-docker-args...
  local name=$1 port=$2; shift 2
  docker run -d --name "$name" -p 127.0.0.1:$port:8080 -e WEBUI_AUTH=False -e ENABLE_PERSISTENT_CONFIG=False \
    -e OPENAI_API_BASE_URL=http://127.0.0.1:9/v1 -e OPENAI_API_KEY=none "$@" $IMG >/dev/null || return 1
  for i in $(seq 1 150); do curl -sf http://127.0.0.1:$port/health >/dev/null && { echo "$name: healthy after $((i*2)) s"; return 0; }; sleep 2; done
  echo "$name: no /health"; docker logs --tail 30 "$name"; return 1
}

hdr() { grep -i "^HTTP/\|^access-control\|^vary\|^content-type" | tr -d '\r'; }

probe() { # port label
  local port=$1 label=$2 u=http://127.0.0.1:$1
  {
    echo "=== $label ==="
    echo "--- signin, no Origin (a program on this computer)"
    curl -si -X POST -H 'Content-Type: application/json' -d '{"email":"","password":""}' $u/api/v1/auths/signin | tee "$OUT/$label.signin.raw" | hdr
    python3 - "$OUT/$label.signin.raw" <<'EOF'
import json, sys
raw = open(sys.argv[1], "rb").read().decode("utf-8", "replace")
body = raw.split("\r\n\r\n", 1)[-1]
try:
    j = json.loads(body)
    print("body: token", "yes" if j.get("token") else "no", "| role", j.get("role"), "| name", j.get("name"))
except Exception as e:
    print("body not JSON:", body[:200])
EOF
    echo "--- signin with Origin: http://evil.example (a script of another page)"
    curl -si -X POST -H 'Origin: http://evil.example' -H 'Content-Type: application/json' -d '{"email":"","password":""}' $u/api/v1/auths/signin | hdr
    echo "--- preflight from http://evil.example"
    curl -si -X OPTIONS -H 'Origin: http://evil.example' -H 'Access-Control-Request-Method: POST' -H 'Access-Control-Request-Headers: content-type,authorization' $u/api/v1/auths/signin | hdr
    echo "--- signin with the page's own Origin http://localhost:3000"
    curl -si -X POST -H 'Origin: http://localhost:3000' -H 'Content-Type: application/json' -d '{"email":"","password":""}' $u/api/v1/auths/signin | hdr
    echo "--- signin with Host: evil.example:3000 (DNS rebinding: the browser sends the page's own name as Host and no Origin on a same-origin POST)"
    curl -si -X POST -H 'Host: evil.example:3000' -H 'Content-Type: application/json' -d '{"email":"","password":""}' $u/api/v1/auths/signin | hdr
    echo "--- socket.io handshake with Origin: http://evil.example"
    curl -si -H 'Origin: http://evil.example' "$u/ws/socket.io/?EIO=4&transport=polling" | hdr
    echo "--- socket.io handshake with Origin: http://localhost:3000"
    curl -si -H 'Origin: http://localhost:3000' "$u/ws/socket.io/?EIO=4&transport=polling" | hdr
    echo "--- the admin session's reach: GET /api/v1/functions/ and /api/v1/tools/ with the no-auth token"
    tok=$(python3 - "$OUT/$label.signin.raw" <<'EOF'
import json, sys
raw = open(sys.argv[1], "rb").read().decode("utf-8", "replace")
print(json.loads(raw.split("\r\n\r\n", 1)[-1]).get("token", ""))
EOF
)
    for p in /api/v1/functions/ /api/v1/tools/ /api/v1/users/; do
      printf '%s -> ' "$p"; curl -s -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer $tok" $u$p
    done
    echo "--- the same, with Origin: http://evil.example, ACAO shows whether the page may read it"
    curl -si -H 'Origin: http://evil.example' -H "Authorization: Bearer $tok" $u/api/v1/functions/ | hdr
  } 2>&1 | tee "$OUT/$label.txt"
}

run owu-a 3101 && probe 3101 default
docker logs owu-a 2>&1 | grep -i "cors\|WARNING" | head -5 > "$OUT/default.logs.txt"
docker rm -f owu-a >/dev/null 2>&1

run owu-b 3102 -e 'CORS_ALLOW_ORIGIN=http://localhost:3000;http://127.0.0.1:3000' && probe 3102 cors-set
docker logs owu-b 2>&1 | grep -i "cors\|WARNING" | head -5 > "$OUT/cors-set.logs.txt"
docker rm -f owu-b >/dev/null 2>&1
echo done > "$OUT/DONE"
