# One-shot and release checks (Qwen Hummingbird, 2026-10-14)

- `loops.py URL N OUTDIR prompt.txt '{"name": {sampling}}'`: the one-shot game prompt N times per setting; finish, loop flag, closed html.
- `testall.sh REMOTE_DIR LOCAL_DIR`: every reply through `gametest.js` (headless Chrome: page errors, title, scripted play, #autoplay
  screenshots). Look at the screenshots: error-free is not playable. `findjs.js FILE`: the first syntax error. `sheet.js`: screenshots side by side.
- `prep.py` + `run_all.sh`: teacher-forced scoring (`"score_from"` on qwenserve's JSON-lines protocol) of a human-written file through each build.
- `cancel_test.py URL`: stop, conversation reuse, thinking. `verify.sh`: the clean-install release chain. `http_restart.sh`: the PC's dev server.
- `capture_clock.js` + `makegif.py`: a page's frames on a virtual clock, then a GIF.
Paths inside are the PC's (~/glyd-scratch) and the Mac's playwright-core; adjust when reusing.
