#!/bin/bash
# The README's two curl commands, as written there, against the server on localhost:8000 (the server up, bash serve.sh).
#   bash curl_test.sh
echo '$ curl localhost:8000/v1/models'
curl -s localhost:8000/v1/models | python3 -c 'import sys, json; d = json.load(sys.stdin)["data"][0]; print({k: d[k] for k in ("id", "max_model_len")})'
echo
echo '$ curl localhost:8000/v1/chat/completions ...'
curl -s localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen3-8B",
  "messages": [{"role": "user", "content": "What is lossless compression? Answer in one sentence. /no_think"}],
  "max_tokens": 100}'
echo
