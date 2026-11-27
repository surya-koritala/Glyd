# Adding a model family

Glyd does not have a file per model. A family is a *description builder*: a function that reads a release's `config.json` and says which
layer kinds its blocks are and which tensor of the checkpoint each part reads. One executor runs every description, so a family that is
described runs at every level, with several users, through the planner and the server, with no code of its own. This page is the routine
for adding one, and for meeting a new release of a family that is already here.

It needs a development build of the engine (`cargo build -p glyd-engine`; the commands below are not in a release).

## 1. Intake: what the release is, against what the engine has

```bash
glyd-engine intake ORG/NAME
```

It reads the repository's `config.json` and the headers of its safetensors files by ranges. Nothing of the weights is downloaded, so a model
of a terabyte is read in seconds. `MODEL` can also be a directory, or a recorded header (below). It prints one report:

- **what the model is**: its model type, its architecture's name, how many tensors it has and their size;
- **where it describes**: the layer kinds of its blocks, then every tensor of the checkpoint the description does not read and every tensor
  it reads that the checkpoint lacks, grouped by name with the layer and expert numbers masked (`model.layers.{L}.mlp.experts.{E}.up.weight`);
- **where a family's builder refuses it**: the refusal, and *every* config key the builder does not understand, not the first one;
- **where no builder takes its model type** (a new family): every key of the text model's config that says something of its shape, with its
  value, and every group of the checkpoint's tensors with its type and shape. This is what a builder is written from;
- **signs of the layer kinds it needs**: words in its config keys and tensor names (`n_routed_experts`, `kv_lora_rank`, `short_conv_kernel_size`),
  each with whether the engine has the kind. A sign is a word in a name. It says where to look in the release's own code; it is not a reading
  of that code.

It ends with status 0 where the model describes and its checkpoint is read whole, and 3 where it does not.

For a release of a family that is here, a clean report means there is nothing to do. A key that is not understood, or a tensor that is not
read, is the whole of the work: go to step 2 with that key.

## 2. The builder

The builders are in `glyd-engine/src/runtime/describe.rs`; `describe_config_in` chooses one by the model type. A builder:

- reads **every** key of the config through the getters of `Keys`. A key that nobody read and that could change what a step computes is
  refused by name: a release that adds a setting is stopped at its config, not found later as a wrong answer. Keys that cannot change the
  math (ids, dtypes, training settings) are listed once, in `INERT`;
- gives each key the **default the release's own code gives it** (transformers' configuration class, or the release's modelling file), and
  refuses a value it does not support (`expect`) in words that say which key and which value;
- returns the blocks as layer kinds (`runtime/kinds.rs`: the mixers Attention, Mla, Compressed, Linear and Mamba2, the feed-forwards Mlp and
  Moe, Ngram, PerLayer, and the residual forms), and for each block the checkpoint's tensors by role (`runtime/names.rs`).

Write it from the release's own code and its paper, in Glyd's terms. Where the family is one of another engine's too, read how that engine
does it to check an understanding, not to copy it.

A family whose layers are all kinds the engine has needs the builder and its names, nothing else. A layer the engine has no kind for is a
new kind (step 5).

## 3. Record the header, and the tests that come with it

```bash
glyd-engine intake ORG/NAME --record glyd-engine/tests/fixtures/headers
```

writes `NAME.json`: the config, the files the Hub listed and the table of tensors in a compact form (a few hundred kilobytes for a model of
half a million tensors). Committed, it is held by the tests from then on:

- every record in that folder describes, every tensor of its checkpoint is read or is a part the description names as left (a vision tower the
  language model does not run), and every tensor it reads is there. This test reads the folder: a new record needs no line;
- its table of tensors adds up to the files' sizes, so the planner's sizes for the family are the release's. This one lists its records
  (`HEADER_FIXTURES` in `src/sizes.rs`): add the new one's name.

A gated repository is recorded with `HF_TOKEN` set, and its record is kept out of the tree where its licence asks for that.

## 4. A random model of the layout, held to the CPU reference

The engine's answer for a family is held to a second implementation: the CPU reference (`runtime/reference`), one plain function a kind in
32-bit floats, itself held to transformers where transformers has the family. A family is proven on a small random model of its layout, once,
and not model by model:

```bash
glyd-engine logitsmodel CASE DIR      # a random model of the case's layout
glyd-engine kindcheck CASE            # every layer kind of it: the executor's kernels against the CPU reference
glyd-engine logitscheck CASE          # its logits against the recorded fixture
```

Add the family as a case beside the others in `src/logits.rs`, with a config small enough to make in seconds and with every option the
released configs use turned on.

## 5. A layer the engine has no kind for

A new kind is added to `runtime/kinds.rs` as data, with its function in the CPU reference first: that is the definition every backend is
held to. On the GPU it starts in a plain form, built from the operations the executor has (the packed-matrix product, norms, activations, adds),
so that the family runs and is correct; a kernel of its own comes later, where a measurement says it pays.

## 6. One run on a GPU

`glyd run ORG/NAME` on one card: the plan `glyd fit` prints is the one that loads, and the first answer is a sensible one. Then the family's
line in the table of what the engine runs (`glyd-engine/src/support.rs`), from which this guide's [Models](models.md) page is made.

## What a pull request for a family holds

1. the builder and its names;
2. the recorded header of at least one release;
3. the random model's case and its fixture;
4. the family's line in `support.rs`;
5. a changelog entry (`changelog.d/`).
