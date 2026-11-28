- MiMo's layout runs on the engine's executor in bf16: attention heads of 192 with values of 128, the values times the family's factor, the three attention
  products as one matrix stored in shards, sliding layers with a sink a head, and experts routed by sigmoid scores with a choice bias. A small random model
  of the layout is a case of the logits checks and of `kindcheck`, held to the CPU reference.
- MiMo's draft layers are not read.
- An attention layer with heads of 192, or with values narrower than its queries and keys, runs on the attention kind's own kernels and keeps its keys and
  values in bf16 pages; narrower values with an output gate are refused by name.
