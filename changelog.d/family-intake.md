- `glyd-engine intake MODEL` (a development build): the first step of adding a model family, or of meeting a new release of one. It reads a Hub
  repository's config and file headers by ranges (no weights), tries to describe the model, and prints in one pass every config key that is not
  understood, every tensor not read or missing, a new family's keys and tensor groups, and signs of the layer kinds it needs; `--record` writes
  the header the family's tests hold. A config with several keys the builder does not know now names all of them, not the first. The guide has a
  page, "Adding a model family".
