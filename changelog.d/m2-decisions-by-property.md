- `glyd fit`, `run` and `serve` size a model by the fields of its family's row in the table of families (the layout its matrices take, whether its layers
  run across the GPUs, how its expert cache is sized, which matrices may wait in the computer's memory, whether it drafts with its own head), not by the
  family's name; a name the table has no row for is refused instead of being sized as another family. No plan of a family the engine runs changes.
  Part of #239.
