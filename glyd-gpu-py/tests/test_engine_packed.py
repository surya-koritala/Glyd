"""The Hub repo of a model at a level (engine.packed_repo): `glyd/NAME-LEVEL`, the rule glyd-engine's cli.rs packed_repo has and the publish workflow names its repos by. What needs no GPU.

    python test_engine_packed.py          (or pytest test_engine_packed.py)"""
from glyd_gpu import engine


def test_packed_repo():
    for model, level, repo in (
        ("Qwen/Qwen3.5-9B", "linnet", "glyd/Qwen3.5-9B-linnet"),
        ("Qwen/Qwen3.8-27B", "kestrel", "glyd/Qwen3.8-27B-kestrel"),
        ("Qwen/Qwen3.5-9B", "penguin", "glyd/Qwen3.5-9B-penguin"),
        ("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", "hummingbird", "glyd/nemotron-3-hummingbird"),
    ):
        assert engine.packed_repo(model, level) == repo


if __name__ == "__main__":
    test_packed_repo()
    print("test_packed_repo ok")
