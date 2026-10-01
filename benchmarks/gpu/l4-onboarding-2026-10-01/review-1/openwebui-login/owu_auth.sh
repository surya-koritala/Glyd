#!/bin/bash
# Open WebUI 0.11.4 with its login on (no WEBUI_AUTH=False): what the first visit and the API do. CPU only, no model server.
set -u
IMG=ghcr.io/open-webui/open-webui:v0.11.4
OUT=${1:-$HOME/onb/owu-auth}
mkdir -p "$OUT"; docker rm -f owu-c >/dev/null 2>&1; docker volume rm owu-c-data >/dev/null 2>&1
docker run -d --name owu-c -p 127.0.0.1:3103:8080 -e OPENAI_API_BASE_URL=http://127.0.0.1:9/v1 -e OPENAI_API_KEY=none -e ENABLE_PERSISTENT_CONFIG=False \
  -e 'CORS_ALLOW_ORIGIN=http://localhost:3000;http://127.0.0.1:3000' -v owu-c-data:/app/backend/data $IMG >/dev/null
for i in $(seq 1 150); do curl -sf http://127.0.0.1:3103/health >/dev/null && { echo "healthy after $((i*2)) s"; break; }; sleep 2; done
u=http://127.0.0.1:3103
code() { curl -s -o /dev/null -w '%{http_code}' "$@"; }
{
echo "--- no token: GET /api/models -> $(code $u/api/models)"
echo "--- signin with empty credentials -> $(code -X POST -H 'Content-Type: application/json' -d '{"email":"","password":""}' $u/api/v1/auths/signin)"
echo "--- config, no token (does it say signup is open?)"; curl -s $u/api/config | python3 -c 'import json,sys; j=json.load(sys.stdin); print({k: j.get(k) for k in ("onboarding","auth","features")})' 2>&1 | cut -c1-300
echo "--- the first signup (what a person does on first visit)"
curl -s -X POST -H 'Content-Type: application/json' -d '{"name":"Acceptance","email":"acceptance@example.com","password":"Pw-1234567890","profile_image_url":"/user.png"}' $u/api/v1/auths/signup | tee "$OUT/signup1.raw" | python3 -c 'import json,sys; j=json.load(sys.stdin); print({k: (v if k != "token" else "<token>") for k, v in j.items() if k != "profile_image_url"})'
echo "--- a second signup"
curl -s -X POST -H 'Content-Type: application/json' -d '{"name":"Other","email":"other@example.com","password":"Pw-1234567890","profile_image_url":"/user.png"}' $u/api/v1/auths/signup | python3 -c 'import json,sys; j=json.load(sys.stdin); print({k: (v if k != "token" else "<token>") for k, v in j.items() if k != "profile_image_url"})' 2>&1 | cut -c1-300
tok=$(python3 -c 'import json; print(json.load(open("'$OUT'/signup1.raw"))["token"])')
echo "--- with the token: GET /api/models -> $(code -H "Authorization: Bearer $tok" $u/api/models)"
echo "--- /api/v1/auths/ (who am I)"; curl -s -H "Authorization: Bearer $tok" $u/api/v1/auths/ | python3 -c 'import json,sys; j=json.load(sys.stdin); print({k: j.get(k) for k in ("role","email","name")})'
echo "--- signin with the account's credentials -> $(code -X POST -H 'Content-Type: application/json' -d '{"email":"acceptance@example.com","password":"Pw-1234567890"}' $u/api/v1/auths/signin)"
echo "--- CORS: evil origin POST signin"; curl -si -X POST -H 'Origin: http://evil.example' -H 'Content-Type: application/json' -d '{"email":"","password":""}' $u/api/v1/auths/signin | grep -i "^HTTP/\|^access-control" | tr -d '\r'
echo "--- CORS: preflight from evil"; curl -si -X OPTIONS -H 'Origin: http://evil.example' -H 'Access-Control-Request-Method: POST' -H 'Access-Control-Request-Headers: content-type,authorization' $u/api/v1/auths/signin | grep -i "^HTTP/\|^access-control" | tr -d '\r'
echo "--- CORS: own origin POST signin (bad credentials)"; curl -si -X POST -H 'Origin: http://localhost:3000' -H 'Content-Type: application/json' -d '{"email":"","password":""}' $u/api/v1/auths/signin | grep -i "^HTTP/\|^access-control" | tr -d '\r'
} 2>&1 | tee "$OUT/probe.txt"
docker logs owu-c 2>&1 | grep -iE "secret|error|WARNING" | head -5 > "$OUT/logs.txt"
docker rm -f owu-c >/dev/null 2>&1; docker volume rm owu-c-data >/dev/null 2>&1
echo done > "$OUT/DONE"
