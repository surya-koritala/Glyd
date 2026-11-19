import os, sys
from vllm.entrypoints.cli import serve as S
from vllm.entrypoints.cli.main import main

if __name__ == "__main__":
    tmp = os.environ.pop("GLYD_TEST_TMP_API", "")
    if tmp:  # the API servers are started after the engine's process: they get this TMPDIR, the engine keeps the parent's
        Orig = S.APIServerProcessManager

        class Mgr(Orig):
            def __init__(self, *a, **k):
                os.environ["TMPDIR"] = tmp
                super().__init__(*a, **k)

        S.APIServerProcessManager = Mgr
    sys.argv = ["vllm"] + sys.argv[1:]
    main()
