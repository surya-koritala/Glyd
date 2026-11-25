- A quality bar for the lossy levels (R14): `glyd-engine quality` measures any OpenAI-compatible server on one fixed suite: next-token KL after 192
  prefixes of four fixed texts, and small reasoning, code and long-context task sets (40, 40 and 4 a length) made from a seed, with known answers and
  no dataset. `quality --card` takes a pack's card against the lossless level of the same model and holds it to the level's bar (kestrel: mean KL
  0.02, top-1 94%, a task 5 points; swift 0.05, 91%, 7.5; wren 0.08, 86%, 10; hummingbird 0.12, 90%, 10): publish or refuse. `ship` writes the card
  into the pack (`quality.json`, and a Quality section of its README) and does not upload a lossy pack without a passing one. `glyd fit` shows each
  level's figures, or "not measured". `quality --table` sets cards side by side by bits, for comparisons with GGUF, AWQ and FP8 at equal bits
  (llama.cpp's log-probabilities are read as well). No card has been measured yet. docs/quality.md.
