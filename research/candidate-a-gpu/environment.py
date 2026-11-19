"""Validate the immutable image and research ABI without changing packages."""
import argparse
import ctypes
import importlib.metadata
import json
from pathlib import Path
import platform

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--expected",type=Path)
    ap.add_argument("--library",type=Path,required=True)
    ap.add_argument("--gpu",action="store_true")
    args=ap.parse_args()
    import torch
    if platform.python_version_tuple()[:2] != ("3","12") or torch.__version__.split("+")[0] != "2.8.0":
        raise ValueError("unexpected base Python or PyTorch")
    record={"python":platform.python_version(),"torch":torch.__version__,"torch_cuda":torch.version.cuda,
            "numpy":importlib.metadata.version("numpy")}
    if args.expected and record != json.loads(args.expected.read_text()):
        raise ValueError("prepared runtime identity mismatch")
    from validate import api
    funcs=api(args.library)
    a,b=ctypes.c_size_t(),ctypes.c_size_t()
    if funcs["workspace"](0,16,1,0,ctypes.byref(a),ctypes.byref(b))!=1:
        raise ValueError("ABI invalid-dimension check failed")
    if args.gpu:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()!=(8,9) or torch.cuda.device_count()!=1:
            raise ValueError("one SM89 GPU required for this batch")
        record={**record,"gpu":torch.cuda.get_device_name(),"cc":list(torch.cuda.get_device_capability()),
                "total_memory":torch.cuda.get_device_properties(0).total_memory}
    args.out.write_text(json.dumps(record,indent=2)+"\n")

if __name__=="__main__":
    main()
