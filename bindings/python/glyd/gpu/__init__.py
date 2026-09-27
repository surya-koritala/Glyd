"""Glyd on the GPU: a bf16 model's weights held compressed in GPU memory,
bit for bit, and multiplied from there: kernels.py (the packs and the
products), _lib.py (the kernels from the prebuilt library, through its C
API), model.py (the modules in place of nn.Linear and nn.Embedding).
Under the Business Source License 1.1 (LICENSE-glyd-gpu), as the rest of
Glyd's GPU code.
"""
