"""Prepared GPU matrix validation. Isolated matrices are not model serving benchmarks."""
import argparse
import ctypes as C
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from protocol import POLICY, TOKENS, check_inputs, complete_report, digest

PTR, I64, SIZE = C.c_void_p, C.c_int64, C.c_size_t

def bind(lib, name, types):
    fn = getattr(lib,name)
    fn.argtypes, fn.restype = types, C.c_int
    return fn

def api(path):
    lib = C.CDLL(str(path.resolve()))
    return {
        "workspace":bind(lib,"glyd_candidate_a_workspace",[I64,I64,I64,I64,C.POINTER(SIZE),C.POINTER(SIZE)]),
        "unpack":bind(lib,"glyd_candidate_a_unpack",[PTR]*4+[I64,I64,PTR,PTR]),
        "product":bind(lib,"glyd_candidate_a_product",[PTR]*4+[I64,I64,PTR,I64,I64,PTR,PTR,PTR,SIZE,PTR,PTR]),
        "base_unpack":bind(lib,"glyd_gpu_mma_unpack",[PTR]*4+[I64,I64,I64,PTR,I64,PTR]),
        "base_workspace":bind(lib,"glyd_gpu_mma_gemm_workspace",[I64,I64,I64,C.POINTER(SIZE)]),
        "base_big_workspace":bind(lib,"glyd_gpu_mma_gemm_x1_workspace",[I64,I64,I64,I64,C.POINTER(SIZE)]),
        "base_product":bind(lib,"glyd_gpu_mma_gemm",[PTR]*4+[I64,I64,PTR,I64,PTR,PTR,PTR,SIZE,PTR,PTR]),
        "base_big_product":bind(lib,"glyd_gpu_mma_gemm_x1",[PTR]*4+[I64,I64,PTR,I64,PTR,PTR,I64,PTR,SIZE,PTR,PTR]),
    }

def ok(code):
    if code != 0:
        raise RuntimeError(f"CUDA function returned {code}")

def fp8_product(x, wbf, bias, reference, rotate, timing):
    """Diagnostic FP8 baseline: per-tensor e4m3 weights and activations through cuBLASLt with fast
    accumulation. Activation quantization is excluded, so this bounds FP8 product time from below."""
    import torch
    f8=torch.float8_e4m3fn
    sw=(wbf.float().abs().amax().clamp(min=1e-30)/448).reshape(())
    sx=(x.float().abs().amax().clamp(min=1e-30)/448).reshape(())
    w8=(wbf.float()/sw).to(f8)
    x8=(x.float()/sx).to(f8)
    out=torch.empty((x.shape[0],wbf.shape[0]),dtype=torch.bfloat16,device="cuda")
    def mm(w):
        torch._scaled_mm(x8,w.t(),scale_a=sx,scale_b=sw,bias=bias,out_dtype=torch.bfloat16,use_fast_accum=True,out=out)
    mm(w8)
    torch.cuda.synchronize()
    delta=out.float()-reference
    nrms=float(delta.square().mean().sqrt())/max(float(reference.square().mean().sqrt()),1e-30)
    fw,rotation=rotate((w8,))
    ms=timing(lambda i: mm(fw[i][0]),len(fw))
    return {"fp8_ms":ms,"fp8_normalized_rms_error":nrms,"fp8_rotation":rotation,
            "fp8_scaling":"per-tensor e4m3 weights and activations, cuBLASLt fast accumulation; activation quantization excluded"}

def case_run(root, case, output):
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((8,0),(8,9)):
        raise RuntimeError("prepared SM80/SM89 GPU required")
    for key in os.environ:
        if key.startswith("GLYD_GPU_"):
            raise RuntimeError("unrecorded kernel tuning environment")
    torch.manual_seed(20261007)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    funcs = api(root/"libcandidate_a.so")
    directory = root/"fixtures"/case["id"]
    meta = json.loads((directory/"pack.json").read_text())
    rows,cols = case["shape"]
    tiers = (C.c_uint32*3)(*meta["tiers"])
    base_tiers = (C.c_uint32*3)(*meta["base_tiers"])
    def stream():
        return torch.cuda.current_stream().cuda_stream
    def load(name,dtype,device):
        return torch.frombuffer(bytearray((directory/name).read_bytes()),dtype=dtype).clone().to(device)
    def ptr(t):
        return t.data_ptr() if t is not None else None
    def stats():
        torch.cuda.synchronize()
        free,total = torch.cuda.mem_get_info()
        return {"allocated":torch.cuda.memory_allocated(),"reserved":torch.cuda.memory_reserved(),
                "peak_allocated":torch.cuda.max_memory_allocated(),
                "peak_reserved":torch.cuda.max_memory_reserved(),"device_free":free,"device_total":total}
    def guarded(count):
        buf = torch.full((count+32,),0x5a5a,dtype=torch.int16,device="cuda")
        return buf,buf[16:-16].view(torch.bfloat16)
    def guards(buf):
        if not bool((buf[:16]==0x5a5a).all()) or not bool((buf[-16:]==0x5a5a).all()):
            raise RuntimeError("output guard corrupted")
    def rotate(tensors):
        # Copies spanning POLICY["rotation_bytes"]: each timed operation reads weights evicted from L2.
        nbytes = sum(t.numel()*t.element_size() for t in tensors)
        k = min(POLICY["max_copies"], max(2, -(-POLICY["rotation_bytes"]//max(1,nbytes))))
        return [tuple(tensors)]+[tuple(t.clone() for t in tensors) for _ in range(k-1)], {"copies":k,"bytes":k*nbytes}
    def timing(fn, k):
        ops = max(POLICY["repetitions"], k)
        for _ in range(POLICY["warmup"]):
            for i in range(k):
                fn(i)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for i in range(ops):
                fn(i % k)
        graph.replay()
        torch.cuda.synchronize()
        samples=[]
        for _ in range(POLICY["timing_groups"]):
            a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record()
            graph.replay()
            b.record()
            b.synchronize()
            samples.append(a.elapsed_time(b)/ops)
        del graph
        return samples
    def compare(y,ref,exact_required):
        observed=y.reshape_as(ref).float()
        delta=observed-ref
        maximum=float(delta.abs().max())
        ref_max=float(ref.abs().max())
        rms=float(delta.square().mean().sqrt())
        ref_rms=float(ref.square().mean().sqrt())
        nrms=rms/max(ref_rms,1e-30)
        exact=torch.equal(y.reshape_as(ref).view(torch.int16),ref.to(torch.bfloat16).view(torch.int16))
        passed=bool(torch.isfinite(observed).all()) and maximum<=POLICY["max_abs_error_fraction_of_reference_max"]*ref_max+POLICY["max_abs_error_absolute_allowance"] and nrms<=POLICY["max_normalized_rms_error"]
        if exact_required:
            passed=passed and exact
        if not passed:
            raise RuntimeError(f"reference failed: {case['id']} max={maximum} nrms={nrms}")
        return maximum,nrms,exact,ref_max

    torch.cuda.init()
    warm = torch.zeros(1,device="cuda")
    del warm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = stats()
    data = load("compact.data",torch.uint8,"cuda")
    blocks = load("blocks.bin",torch.uint8,"cuda")
    bases = load("bases.i32",torch.int32,"cuda")
    weights_held = stats()
    if sum(t.numel()*t.element_size() for t in (data,blocks,bases)) != meta["weight_buffer_bytes"]:
        raise RuntimeError("device weight byte accounting differs")
    expected = load("rounded.bf16",torch.int16,"cpu").reshape(rows,cols)
    unpack_buf,unpacked = guarded(rows*cols)
    if meta["compact"]:
        ok(funcs["unpack"](ptr(data),ptr(blocks),ptr(bases),tiers,rows,cols,ptr(unpacked),stream()))
    else:
        if funcs["unpack"](ptr(data),ptr(blocks),ptr(bases),tiers,rows,cols,ptr(unpacked),stream()) != 1:
            raise RuntimeError("compact API accepted a fallback tag")
        ok(funcs["base_unpack"](ptr(data),ptr(blocks),ptr(bases),tiers,cols,0,rows,ptr(unpacked),0,stream()))
    torch.cuda.synchronize()
    if not torch.equal(unpacked.view(torch.int16).reshape(rows,cols).cpu(),expected):
        raise RuntimeError("decoded weight bits differ")
    guards(unpack_buf)
    del unpacked,unpack_buf
    result = {"case":case["id"],"shape":case["shape"],"unpack_bit_exact":True,
              "weight_buffer_bytes":meta["weight_buffer_bytes"],"host_symbol_bytes":12,
              "compact_weight_allocator_delta":weights_held["allocated"]-before["allocated"],
              "memory_before":before,"memory_weights":weights_held,
              "products":[],"complete":False}
    output.write_text(json.dumps(result,indent=2)+"\n")
    if not case["products"]:
        result["complete"]=True
        output.write_text(json.dumps(result,indent=2)+"\n")
        return
    for bad in ((0,cols,1,0),(rows,0,1,0),(rows,cols,0,0),(rows,cols,2049,0),(rows+1,cols,1,0),
                (rows,cols,1,1),(rows,cols,128,3),(rows,cols,128,-1)):
        w,n = SIZE(),SIZE()
        if funcs["workspace"](*bad,C.byref(w),C.byref(n)) != 1:
            raise RuntimeError("invalid product dimensions or variant accepted")

    for tokens in TOKENS:
        for has_bias in (False,True):
            work_bytes,counter_bytes = SIZE(),SIZE()
            code=funcs["workspace"](rows,cols,tokens,0,C.byref(work_bytes),C.byref(counter_bytes))
            if tokens>64 and cols%64:
                if code != 1:
                    raise RuntimeError("unaligned large product was not refused")
                result["products"].append({"tokens":tokens,"bias":has_bias,"status":"expected-refusal"})
                continue
            ok(code)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            compact_start=stats()
            if case["exact_product"]:
                x=((torch.arange(tokens*cols,device="cuda",dtype=torch.int32)*5)%7-3).to(torch.bfloat16).reshape(tokens,cols)
                bias=((torch.arange(rows,device="cuda")%5)-2).to(torch.bfloat16) if has_bias else None
            else:
                x=torch.randn(tokens,cols,device="cuda",dtype=torch.float32).to(torch.bfloat16)
                bias=(torch.randn(rows,device="cuda")*0.1).to(torch.bfloat16) if has_bias else None
            ybuf,y=guarded(tokens*rows)
            work=torch.empty(max(1,work_bytes.value),dtype=torch.uint8,device="cuda")
            counters=torch.zeros(counter_bytes.value//4,dtype=torch.int32,device="cuda")
            def product(bufs,variant,out,work,work_size,counters,offset=0):
                d,bl,ba = bufs
                return funcs["product"](ptr(d),ptr(bl),ptr(ba),tiers,rows,cols,ptr(x)+offset,tokens,variant,
                                        ptr(bias),ptr(out),ptr(work),work_size,ptr(counters),stream())
            def repeatable(variant,out,work,work_size,counters):
                ok(product((data,blocks,bases),variant,out,work,work_size,counters))
                torch.cuda.synchronize()
                first=out.view(torch.int16).cpu()
                for _ in range(3):
                    ok(product((data,blocks,bases),variant,out,work,work_size,counters))
                    torch.cuda.synchronize()
                    if not torch.equal(out.view(torch.int16).cpu(),first):
                        raise RuntimeError("product changed bits on repetition")
                    if bool(counters.any()):
                        raise RuntimeError("product counters did not reset")
                return first
            if product((data,blocks,bases),0,y,work,work_bytes.value,counters,offset=2) != 1:
                raise RuntimeError("misaligned input was not refused")
            first=repeatable(0,y,work,work_bytes.value,counters)
            guards(ybuf)
            compact_memory=stats()
            # Reference buffers and timing copies below are excluded from compact-only peak accounting.
            wbf=expected.view(torch.bfloat16).to("cuda")
            reference=x.float() @ wbf.float().T
            if bias is not None:
                reference.add_(bias.float())
            max_error,nrms,exact,ref_max=compare(y,reference,case["exact_product"])
            cw,compact_rotation=rotate((data,blocks,bases))
            compact_ms=timing(lambda i: ok(product(cw[i],0,y,work,work_bytes.value,counters)),len(cw))
            torch.cuda.synchronize()
            if not torch.equal(y.view(torch.int16).cpu(),first):
                raise RuntimeError("graph replay changed product bits")
            guards(ybuf)
            variants={}
            if tokens>64:
                for v in POLICY["big_variants"]:
                    vw,vn = SIZE(),SIZE()
                    ok(funcs["workspace"](rows,cols,tokens,v,C.byref(vw),C.byref(vn)))
                    vwork=torch.empty(max(1,vw.value),dtype=torch.uint8,device="cuda")
                    vcount=torch.zeros(vn.value//4,dtype=torch.int32,device="cuda")
                    vbuf,vy=guarded(tokens*rows)
                    vfirst=repeatable(v,vy,vwork,vw.value,vcount)
                    vmax,vnrms,vexact,_=compare(vy,reference,case["exact_product"])
                    vms=timing(lambda i: ok(product(cw[i],v,vy,vwork,vw.value,vcount)),len(cw))
                    torch.cuda.synchronize()
                    if not torch.equal(vy.view(torch.int16).cpu(),vfirst):
                        raise RuntimeError("graph replay changed variant product bits")
                    guards(vbuf)
                    variants[str(v)]={"repeat_bit_exact":True,"reference_pass":True,"max_abs_error":vmax,
                                      "normalized_rms_error":vnrms,"integer_or_reference_bit_exact":vexact,
                                      "bit_equal_to_default":bool(torch.equal(vfirst,first)),
                                      "workspace_bytes":vw.value,"ms":vms}
                    del vwork,vcount,vbuf,vy,vfirst
            bf_y=torch.empty((tokens,rows),dtype=torch.bfloat16,device="cuda")
            bw,bf16_rotation=rotate((wbf,))
            def bf_product(i):
                w=bw[i][0]
                if bias is None:
                    torch.mm(x,w.T,out=bf_y)
                else:
                    torch.addmm(bias,x,w.T,out=bf_y)
            bf_ms=timing(bf_product,len(bw))
            del bw
            expanded=load("expanded.data",torch.uint8,"cuda")
            ew,expanded_rotation=rotate((expanded,blocks,bases))
            base_ws=SIZE()
            if tokens<=64:
                ok(funcs["base_workspace"](rows,cols,tokens,C.byref(base_ws)))
            else:
                ok(funcs["base_big_workspace"](rows,cols,tokens,0,C.byref(base_ws)))
            base_work=torch.empty(max(1,base_ws.value),dtype=torch.uint8,device="cuda")
            base_counts=torch.zeros(((tokens+127)//128)*(rows//64),dtype=torch.int32,device="cuda")
            def base_product(i):
                e,bl,ba=ew[i]
                args=[ptr(e),ptr(bl),ptr(ba),base_tiers,rows,cols,ptr(x),tokens,ptr(bias),ptr(bf_y)]
                if tokens>64:
                    args.append(0)
                args += [ptr(base_work),base_ws.value,ptr(base_counts),stream()]
                ok(funcs["base_product" if tokens<=64 else "base_big_product"](*args))
            expanded_ms=timing(base_product,len(ew))
            del ew
            base_error,base_nrms,base_exact,_=compare(bf_y,reference,case["exact_product"])
            parent_difference=float((bf_y.float()-first.view(torch.bfloat16).reshape(tokens,rows).to("cuda").float()).abs().max())
            try:
                fp8=fp8_product(x,wbf,bias,reference,rotate,timing)
            except Exception as e:
                fp8={"fp8_error":f"{type(e).__name__}: {e}"[:500]}
            decoded=torch.empty_like(wbf)
            def unpack(i):
                d,bl,ba=cw[i]
                ok(funcs["unpack"](ptr(d),ptr(bl),ptr(ba),tiers,rows,cols,ptr(decoded),stream()))
            unpack_ms=timing(unpack,len(cw))
            row={"tokens":tokens,"bias":has_bias,"status":"pass","reference_pass":True,
                 "repeat_bit_exact":True,"integer_or_reference_bit_exact":exact,
                 "reference_max":ref_max,"max_abs_error":max_error,"normalized_rms_error":nrms,
                 "base_max_abs_error":base_error,"base_normalized_rms_error":base_nrms,
                 "base_reference_bit_exact":base_exact,
                 "max_difference_from_expanded_product":parent_difference,
                 "workspace_bytes":work_bytes.value,"counter_bytes":counter_bytes.value,
                 "memory_compact_start":compact_start,"memory_compact_peak":compact_memory,
                 "rotation":{"compact":compact_rotation,"expanded":expanded_rotation,"bf16":bf16_rotation},
                 "compact_ms":compact_ms,"expanded_ms":expanded_ms,"bf16_ms":bf_ms,"unpack_ms":unpack_ms,
                 **fp8}
            if variants:
                row["variants"]=variants
            result["products"].append(row)
            output.write_text(json.dumps(result,indent=2)+"\n")
            print(json.dumps({"case":case["id"],"tokens":tokens,"bias":has_bias,"pass":True,"fp8":"fp8_ms" in fp8}),flush=True)
            del x,bias,y,ybuf,work,counters,first,wbf,reference,bf_y,expanded,base_work,base_counts,decoded,cw
    result["complete"]=True
    output.write_text(json.dumps(result,indent=2)+"\n")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bundle",type=Path,required=True)
    ap.add_argument("--out",type=Path)
    ap.add_argument("--case")
    ap.add_argument("--check-inputs",action="store_true")
    args=ap.parse_args()
    manifest=check_inputs(args.bundle)
    if args.check_inputs:
        print(json.dumps({"validated_inputs":len(manifest["cases"])}))
        return
    if args.out is None:
        raise ValueError("--out required")
    args.out.mkdir(parents=True,exist_ok=True)
    if args.case:
        matches=[c for c in manifest["cases"] if c["id"]==args.case]
        if len(matches)!=1:
            raise ValueError("unknown case")
        case_run(args.bundle,matches[0],args.out/(args.case+".json"))
        return
    start=time.monotonic()
    rows=[]
    for case in manifest["cases"]:
        subprocess.run([sys.executable,__file__,"--bundle",str(args.bundle),"--out",str(args.out),
                        "--case",case["id"]],check=True,timeout=600)
        rows.append(json.loads((args.out/(case["id"]+".json")).read_text()))
    report=complete_report(manifest,rows)
    report.update({"manifest_sha256":digest(args.bundle/"cases.json"),"elapsed_s":time.monotonic()-start,
                   "timing":"CUDA event intervals over replayed graphs; five groups of ten operations"})
    (args.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"complete":True,"cases":len(rows)}))

if __name__=="__main__":
    main()
