# The background service

`glyd service` keeps one program running that owns the API port, loads a model when a request names it, unloads it after it has been idle, and holds more
than one model at a time. After `install.sh` on Linux with an NVIDIA GPU it is a user unit of your account: it answers on `127.0.0.1:11434` from then on and at
every login, with no command typed (`GLYD_NO_SERVICE=1` in front of `sh` leaves it out; `glyd service uninstall` takes it out).

```
glyd pull qwen3                      # download a model (by name or Hub id)
curl localhost:11434/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3","messages":[{"role":"user","content":"hi"}]}'    # loads it; the answer streams as with glyd serve
glyd ps                              # what is loaded, for how long, and when it is unloaded
glyd stop qwen3                      # unload it now (glyd stop --all)
glyd run qwen3                       # a chat through the service (its model, no second copy on the GPU)
glyd service install | uninstall | status | stop
```

The service is a supervisor: each model is a `glyd serve` of its own on a loopback port, so a model has the server's behaviour as it is (constrained decoding, tools, the
lossless KV cache) and leaves the GPU by its program ending, which frees all of its memory. The service passes the request through and streams the answer back.

| | |
|---|---|
| `--port N` | The port. Fixed (11434, where Ollama's clients also look). A port in use is an error that names the program that has it, never a move to another port. `glyd ps`, `stop` and `run` read `--port N` or `GLYD_SERVICE_PORT=N`. |
| `--keep-alive T` | How long a model stays loaded after its last request: `90s`, `5m` (the default), `2h`, `0` (unload at once), `never`. |
| `--max-loaded N` | Models held at once (2). A request for another unloads the least recently used one that is idle; if every one is busy the request is answered 503. A load that fails (no room on the GPU) is tried again once with the idle models unloaded. |
| `--context N`, `--users N` | Passed to each model's `glyd serve`. |
| `--host H` | An address of your choice, with the warning `glyd serve` gives for one that is not this computer's own. |

## What is open

- It listens on 127.0.0.1 unless `--host` says otherwise. On its own address it refuses a Host that is not a loopback name and an Origin that is not its own, as `glyd serve` does.
- It loads only models that are downloaded already (`glyd pull`): a request cannot make it fetch a model, and it takes no path.
- `GLYD_API_KEY` (in the unit's environment) guards `/v1`, as for `glyd serve`.
- `glyd stop` and `glyd service stop` carry a MAC (HMAC-SHA256, good for one request and a minute) made with a secret in a file only your account can read
  (`~/.local/state/glyd/service.secret`, mode 0600; the secret is never sent), so another user of the computer cannot stop your models. The same secret lets `glyd ps`, `stop` and
  `run` know that the program on the port is your service, for that port, before they send it a key or a prompt: a program that took the port first cannot prove it (and `glyd run` then runs a copy of its own).
- Each model's program binds a port the system chooses (port 0) and says it in its log, so no other program can take it between a choice and the bind.
- Without an API key, another user of a shared computer who can reach 127.0.0.1 can send the models prompts, as with any server on a loopback port. On a shared computer set `GLYD_API_KEY`.

## Files

Each model's program writes `~/.local/state/glyd/logs/service-MODEL-TIME.log`; when a model does not come up, the answer to the request says why, with the log's last lines.

`glyd service install` writes `~/.config/systemd/user/glyd.service` (Linux; `loginctl enable-linger $USER` starts it at boot rather than at login) or
`~/Library/LaunchAgents/com.getglyd.glyd.plist` (macOS). There is no Windows service yet: `glyd service` runs it in a terminal.
