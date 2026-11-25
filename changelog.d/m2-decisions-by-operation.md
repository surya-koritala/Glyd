- The description executor decides whether an engine holds several sequences, runs a verify pass, and which state types it reports from the operations of
  the model's description, not from the family's name. Qwen3.8-Flash-Next on the executor now serves several users at once, its sequences batched in one
  pass (before, it was refused a second sequence). GPU check not run yet.
