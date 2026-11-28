# Engine design notes

Design documents and implementation plans written while the engine was developed (October 2026). They record decisions and
measurements behind the code; the code is the reference where they differ. The current design is
[docs/design/runtime.md](../../../docs/design/runtime.md) and [REFERENCE.md](../../REFERENCE.md).

- `kernel-libraries.md`, `one-kernel-build.md`: which operations use NVIDIA's libraries and which Glyd's own kernels; the GPU library folded into the engine.
- `specs/`: `2026-10-05-fp8-lossless-design.md` (lossless FP8 for the vLLM plugin; its plan is `plans/2026-10-05-fp8-lossless.md`) and
  `2026-10-06-engine-qwen35-design.md` (Qwen3.5 on the engine: built, and since moved onto the shared executor; the kernels' rounding points cite it).
