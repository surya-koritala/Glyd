"""The library's routes (glyd_gpu.cu's route code, compiled on the host) against check_capi.py's rule (main_route,
split_rule), every GPU code (compute capabilities x classes x WITH_SPLIT), both layouts, a grid of O, K and M, under
several environments; each route's last checked by the library's own routes at last and last + 1."""
import ast, itertools, os, re, subprocess, sys, tempfile, types

T = sys.argv[1]  # the tree
cu, h = open(f"{T}/gpu/glyd_gpu.cu").read(), open(f"{T}/gpu/glyd_gpu.h").read()
a, b = cu.index("struct RouteMins {"), cu.index("// The current device as the routes take it")
c, d = cu.index("static int route_run("), cu.index("GLYD_GPU_API int glyd_gpu_mma_route(")
defs = "".join(f"#define {k} {v}\n" for k, v in re.findall(r"#define (GLYD_GPU_\w+) (\(?-?[0-9][0-9 <()xX]*\)?)\s", h) if not k.startswith("GLYD_GPU_API"))
prog = "#include <cstdint>\n#include <cstdlib>\n#include <cstdio>\n#include <cctype>\n#include <cstring>\n#include <cerrno>\n#include <algorithm>\n#include <initializer_list>\n"
prog += "enum { cudaErrorInvalidValue = 1 };\n" + defs + cu[a:b] + cu[c:d] + r'''
int main() {  // stdin: twelve gpu O K M; stdout: route last sms
    int t; long long g, O, K, M;
    while (scanf("%d %lld %lld %lld %lld", &t, &g, &O, &K, &M) == 5) {
        int r, r2 = -1, r3 = -1; int64_t last, l2;
        route_run(t, g, O, K, M, &r, &last);
        if (last < INT64_MAX - 1) { route_run(t, g, O, K, last, &r2, &l2); route_run(t, g, O, K, last + 1, &r3, &l2); }
        printf("%d %lld %lld %d %d\n", r, (long long)last, (long long)split_sms(t, g, O, K, M), r2, r3);
    }
}
'''
cap = open(f"{T}/gpu/check_capi.py").read()
tree = ast.parse(cap)
fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("main_ahead", "main_dec", "main_route", "split_rule")]
kern = {}
exec(compile(ast.Module([n for n in ast.parse(open(f"{T}/bindings/python/glyd/gpu/kernels.py").read()).body if isinstance(n, ast.Assign)
                         and {x.id for t_ in n.targets for x in ast.walk(t_) if isinstance(x, ast.Name)} <= {"DECODE", "GEMM", "MID", "WG", "BIG", "AHEAD", "SPLIT", "GEFORCE", "A10", "L4", "L40S", "PCIE", "GH200", "H100", "WITH_SPLIT"}], []), "kernels.py", "exec"), kern)
with tempfile.TemporaryDirectory() as tmp:
    open(f"{tmp}/r.cpp", "w").write(prog)
    subprocess.run(["c++", "-std=c++20", "-O2", "-o", f"{tmp}/r", f"{tmp}/r.cpp"], check=True)
    ENVS = [{}, {"GLYD_SPLIT_MIN": "-1"}, {"GLYD_SPLIT_MIN": "100", "GLYD_SPLIT_MAX": "3000", "GLYD_SPLIT_SMS": "6"}, {"GLYD_DEC_MIN": "1000"},
            {"GLYD_WG_MIN": "33", "GLYD_WG_MAX": "2048", "GLYD_MID_MIN": "40"}]
    total = 0
    for env in ENVS:
        e = {k: v for k, v in os.environ.items() if not k.startswith("GLYD_")} | env
        ns = {"g": types.SimpleNamespace(**{k: kern[k] for k in kern if k.isupper()}), "os": types.SimpleNamespace(environ=e)}
        ns.update(MID_MIN=int(e.get("GLYD_MID_MIN", 17)), DEC_MIN=int(e.get("GLYD_DEC_MIN", 0)) or None, WG_MIN=int(e.get("GLYD_WG_MIN", 17)), WG_MAX=int(e.get("GLYD_WG_MAX", 1024)))
        ns.update(SPLIT_MIN=int(e.get("GLYD_SPLIT_MIN", 0)), SPLIT_MAX=int(e.get("GLYD_SPLIT_MAX", 0)), SPLIT_SMS=int(e.get("GLYD_SPLIT_SMS", 0)))
        exec(compile(ast.Module(fns, []), "check_capi.py", "exec"), ns)
        codes = [cc + cls + fl for cc in (70, 75, 80, 86, 87, 89, 90, 100, 103, 120) for cls in (0, 1000, 2000, 3000, 4000, 5000, 6000, *([kern["H100"]] if "H100" in kern else [])) for fl in (0, kern["WITH_SPLIT"])]
        OK = [(512, 1024), (512, 1040), (4096, 4096), (5120, 5120), (5056, 8192), (8192, 5056), (131072, 1024), (25600, 5120)]
        Ms = sorted({*range(0, 2100), *range(2100, 10000, 11), 2559, 2560, 3071, 3072, 4096, 4097, 6143, 6144, 8192, 8193, 20000, 1 << 40})
        rows = [(t, gpu, O, K, M) for t in (0, 1) for gpu in codes for O, K in OK for M in Ms]
        out = subprocess.run([f"{tmp}/r"], input="".join(f"{t} {gpu} {O} {K} {M}\n" for t, gpu, O, K, M in rows), capture_output=True, text=True, env=e, check=True).stdout.split("\n")
        bad = 0
        for (t, gpu, O, K, M), line in zip(rows, out):
            r, last, sms, r2, r3 = map(int, line.split())
            code, asked = gpu & ~kern["WITH_SPLIT"], bool(gpu & kern["WITH_SPLIT"])
            want_sms = ns["split_rule"](code, bool(t), O, K, M) if asked else 0
            want = kern["SPLIT"] if want_sms else ns["main_route"](code, bool(t), K, M)
            ok = r == want and sms == want_sms and (last >= (1 << 62) or (r2 == r and r3 != r)) and last >= M
            if not ok:
                bad += 1
                if bad <= 5:
                    print("MISMATCH", env, dict(twelve=t, gpu=gpu, O=O, K=K, M=M), "got", (r, last, sms, r2, r3), "want", (want, want_sms))
        total += len(rows)
        print(f"{env or 'no GLYD_ variables'}: {len(rows)} routes, {bad} against check_capi's rule")
    print(f"{total} routes in all")
