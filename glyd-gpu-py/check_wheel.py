#!/usr/bin/env python3
"""The gate a glyd-gpu wheel passes before it is published (release.yml runs it on every wheel): what the wheel holds, and what must
not be in it. Pure Python (a demangler, c++filt, where a library is read; cuobjdump where it is at hand): it reads the zip and its ELF files.

    python check_wheel.py WHEEL... [--version V] [--platform manylinux_2_28_x86_64] [--source glyd_gpu] [--header include/glyd_gpu.h]
                          [--license ../gpu/LICENSE] [--sums libs/libglyd_gpu_cuda12.so.sha256 ...] [--no-libs] [--engine-source ../glyd-engine] [--table-header ../gpu/glyd_gpu.h] [--summary]
    python check_wheel.py --lib LIBRARY... [--header include/glyd_gpu.h] [--table-header ../gpu/glyd_gpu.h] [--summary]   (a library alone: gpu-lib job)
    python check_wheel.py --engine PROGRAM... [--engine-source ../glyd-engine] [--table-header ../gpu/glyd_gpu.h] [--summary]   (the engine's program alone: engine job)
    python check_wheel.py --engine-source ../glyd-engine [--summary]   (the engine's Rust sources alone: ci.yml)

  files      only the files a wheel is made of: the compiled modules, the two CUDA libraries, the engine's program, the public header, the
             chat page, the bundled text (compressed), the licence and the metadata; and of Python source only __main__.py (runpy runs a
             module from its code object) and _trace.py (torch.compile traces what it reaches); no .c, .pyx, .pyc, .cu, .md, no second header
  stripped   no symbol table and no debug sections in a compiled file; a module exports only its PyInit_ function, a library only the
             public header's eight functions and glyd_gpu_table, which gives the package every other function (PUBLIC below)
  no PTX     the libraries are the bytes the gpu-lib job listed with cuobjdump (--sums: sha256 of each); where cuobjdump is on PATH it
             is run on them here too (no PTX, machine code for sm_80, 86, 89, 90a, 100, 120). cuobjdump is required: a gate that cannot
             read the machine code fails (the one place it may be absent is a wheel whose libraries and engine --sums vouches for)
  names      every kernel, device function, device variable and type of a library has a neutral name (gen_names.py's: gk_N, gd_N, gv_N,
             gt_N), in its host code and in the machine code's symbols (cuobjdump -symbols, where it is at hand); no name in a library
             carries a method word (IDENT_DENY); the only tuning variable a library names is GLYD_SPLIT_MIN (LIB_ENV_OK)
  header     glyd_gpu.h is include/glyd_gpu.h (--header), declares the eight functions, and says nothing of how the library is built (told())
  docstrings none of the docstrings of the sources (--source) is in the compiled module that was made of them
  method     no string of the sources (--source), no Python source, page or description of the wheel, and no plain text of a compiled module says
             how the weights or the KV cache are held or how a kernel works (METHOD, PHRASES below): public text says what, never
             how. Cython keeps most of a module's strings compressed, so they are read where they come from, in the sources, and
             in the table that holds them (decoded here: string_table(), STABLE below)
  hardened   no identifier of the sources carries a method word (ident_problems(): IDENT_DENY; a name that is another library's, a format's or
             the public API's is IDENT_KEEP); no string of them names a variable that tunes the code (env_problems(): ENV_OK is what a release
             build reads); no table of a compiled module, decoded, holds the name of one (INTERNAL_ENV) or a word of STABLE. What the sources
             hold between `# dev` and `# end dev` is not compiled into a release build (setup.py), so it is not read here
  engine     glyd-engine, the engine's program (Rust, the CUDA driver and the library opened at run time): executable in the wheel, for the wheel's
             architecture, stripped, exporting nothing, needing the system's libraries alone; machine code for sm_80, 86, 89, 90a, 100, 120 and no
             PTX and no line or debug information, every kernel and function of it under a neutral name (cuobjdump reads the cubins: the names of their
             sections and symbols are gen_names.py's, or the compiler's own), no method word or phrase (METHOD, PHRASES, STABLE) in any string of it, no
             development command or flag, internal count or detail a wording could say (ENGINE_DENY), no variable of a development build named (INTERNAL_ENV;
             ENGINE_ENV_OK is what its release build reads; its messages may say ENGINE_SAYS, the way to vLLM), no path of the machine that built it, and with --engine-source no name of a kernel of its
             sources (gen_names.py reads them) and with --table-header none of the library's functions that its table hides; and with --engine-source no
             string of the Rust sources of the release build (a module declared under #[cfg(feature = "dev")], an item under it, a #[cfg(test)] module and
             tune!("NAME") are not compiled into it) that says a method word
  metadata   no dependency without an extra (torch, transformers and vLLM are glyd-gpu[gpu] and glyd-gpu[vllm]), BUSL-1.1, the licence file

Every line it prints carries a tag, [file], [export], [symbols], [docstring], [words], [kernel], [name], [env], [table], [identifier], [dev], [path] or [meta];
--summary adds the count of each. Exit status 1 if anything is wrong, with a line for each."""
import argparse
import ast
import collections
import email.parser
import glob
import hashlib
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile

PYTHON_SOURCE = {"__main__", "_trace"}  # glyd-gpu-py/setup.py's SOURCE
LIBS = ("libglyd_gpu_cuda12.so", "libglyd_gpu_cuda13.so")
ARCHS = {"x86_64": 62, "aarch64": 183}  # ELF e_machine
SASS = {"sm_80", "sm_86", "sm_89", "sm_90a", "sm_100", "sm_120"}
EXTRAS = {"gpu", "vllm"}
ENGINE = "glyd-engine"  # the engine's program, beside the libraries: `glyd run` and `glyd serve` start it where it runs the model
FILES = {"chat.html", "kvcorpus.bin", "glyd_gpu.h", "__main__.py", "_trace.py", ENGINE, *LIBS}
MODULE = re.compile(r"(?P<mod>\w+)\.(?:cpython-3\d+t?-(?:x86_64|aarch64)-linux-gnu|abi3)\.so$")
DIST = re.compile(r"glyd_gpu-[^/]+\.dist-info/(METADATA|WHEEL|RECORD|entry_points\.txt|top_level\.txt|licenses/LICENSE)$")
MAX_BYTES = 90 * 2**20  # PyPI refuses a file past 100 MB
# what the public header does not say: the names of the routes and the GPU classes, tile and ring sizes, the numbers of a layout, the
# environment variables that tune the library, any number in hex
TOLD = r"route|\bring|split|warp|\bsms?\b|\btile|\bsteps?\b|ahead|variant|geforce|\ba10\b|\bl4\b|\bh100|gh200|pcie|cublas|\b(?:64|128|256|1280|1536)\b|_(?:min|max)\b|uint8_t|0x[0-9a-f]"
# what no string, page or plain text of the wheel says: the bit counts and names of the layouts, the pools of an exception scheme and its reserves,
# the fields of a value, a kernel's tiles, warps, TMA, wgmma, cp.async and mbarriers, the SMs set apart, a ring of slots, a route's name. The names
# a user types stay (the layouts mma and mma12, the GLYD_* variables). PHRASES: said in any file, as no name is made of them; WORDS: said in a
# string that is text (a name, or a key as 'tiers' or 'arena', is not).
PHRASES = r"\b12[- ]bits?\b|cp\.async|done counters?|green context|split byte|(?:first|second) reserve|tiered layout|bits a (?:weight|value)"
WORDS = (r"\btier(?:s|ed)?\b|\bexponents?\b|\bmantissa|\breserves?\b|\barena|\bcalibrat|\bescapes?\b|\bhuffman|\bentropy\b|\bwarps?\b|\btiles?\b|wgmma|mbarrier|\bstaging\b|\bshadow\b|"
         r"\bring\b|\bslots?\b|\bslabs?\b|\bpools?\b|\bpages\b|\broutes?\b|(?-i:\b(?:TMA|SMs?|CTAs?|SPLIT)\b)")
METHOD = re.compile(PHRASES + "|" + WORDS, re.I)
PLAIN = re.compile(PHRASES, re.I)
# NVIDIA's own messages, in every program that links the CUDA runtime statically, word for word (they say nothing of Glyd's code)
VENDOR_TEXT = {"API call not permitted since the green context associated with the stream has been destroyed"}

# the engine's program: the variables its release build reads (the cache's folder, the library's path, the API's key: each is in the README), what its
# dynamic table may name besides the system's libraries (none), a word of STABLE in the text of a library of the program that is that library's own, the
# names of the format's keys that carry a method word (glyd.json's: the packs' words), the Rust sources of the engine that make its release build
ENGINE_ENV_OK = {"GLYD_CACHE", "GLYD_GPU_LIB", "GLYD_API_KEY", "GLYD_ENGINE_ASKED", "GLYD_IPV4"}  # (GLYD_ENGINE_ASKED=1: glyd run's word that the user set GLYD_ENGINE=1, which skips the list of measured GPUs; GLYD_IPV4=1: the download over IPv4 alone, for a network whose IPv6 does not reach the Hub's storage)
ENGINE_SAYS = ("GLYD_ENGINE=0",)  # what its messages may say besides: the way to vLLM, where the engine does not do what a request asks (it reads no variable of that name)
ENGINE_KEEP = {"tiers"}
ENGINE_FOREIGN = (re.compile(r"exponent must be an integer"),  # (minijinja's own message for the template language's pow)
                  re.compile(r"[\w./~-]*registry/src/[\w./~-]+|/rustc/[0-9a-f]{40}/[\w./-]+|[\w.~-]*registry\\src\\[\w.\\~-]+|\\rustc\\[0-9a-f]{40}\\[\w.\\-]+"),  # (the paths of the crates' and the standard library's sources, as their panics name them; a Windows build's with backslashes)
                  re.compile(r"escape sequence|pattern at escape|without escape|global thread pool|, slot=|Kamu SM|ring::signature|is Ring|minijinja::filters::escape|slot table length doesn't overflow|a small enough slot|\bRing\b"),  # (the regex crates', onig's, rayon's, ring's, a derived Debug's, a root certificate's and the Unicode tables' own words)
                  re.compile(r"glyd-(?:engine|gpu)[/\\]src[/\\]\w+\.rs"),  # (the file a panic names: the source's own, a file name is no wording)
                  re.compile(r"calibration"))  # (glyd.json's key, save.rs KEYS: a word of the format every lossy save's manifest carries, not a wording of how; "calibrate", "calibrated" and the rest stay refused)
SYSTEM_LIB = re.compile(r"lib(c|m|dl|pthread|rt)\.so\.[0-9]+|libgcc_s\.so\.1|ld-linux-(x86-64|aarch64)\.so\.[0-9]+")
WINDOWS_DLL = re.compile(r"(?i)(?:kernel32|ntdll|advapi32|ws2_32|bcrypt|bcryptprimitives|userenv|ole32|oleaut32|shell32|user32|secur32|crypt32|ncrypt|iphlpapi|dbghelp|api-ms-win-[\w-]+)\.dll")
PE_ARCHS = {"x86_64": 0x8664}  # (a PE file's COFF machine)
PE_LIB_EXPORTS = {"NvOptimusEnablementCuda"}  # (CUDA's static runtime exports it on Windows: a laptop's hint that the NVIDIA GPU runs the program)
WIN_BUILD_PATH = re.compile(r"(?<![\w])[A-Za-z]:\\(?:[^\\\x00-\x1f\"<>|]{1,80}\\)+")
FATBIN = b"\x50\xed\x55\xba"  # (the magic of a CUDA fat binary, which the engine holds as bytes: its kernels)
# What the engine's program may not say, in any wording: the development commands and flags (a release build has none, and a message that tells a user to run one is wrong; rangeworker
# ships, rangecheck does not), the keys of
# a development build's /stats and the other counts of the lossless cache, and the phrases that told how in words METHOD has none of (the cache's extra memory, a calibration's passes, a
# layout's block shape). The kernels' own names are checked in the cubins, by their neutral form, and in the program against the sources'.
ENGINE_DENY = re.compile(
    r"kvcal|kvtest|rowcheck|rowprofile|topkcheck|calcheck|\bkcheck\b|kcheck35|dump35|g4(?:check|fatest|prefill|score|spec|tok)\b|i4xcheck|glmfmtcheck|qx(?:bench|cb|host|layers|prod|toy|tq|vision)\b|skinnybench|prodbench|prodcheck|fp4bench|l2check|pkcheck|moecheck|nhkernels|nhlayers|nhtoy|nhdecode|nhrandom|dsv4random|muserandom|museref|musecheck|f8moecheck|hubslice|rangecheck|fp4stat|f8stat|dsv4prof|dsv4ref|dsv4check|dsv4prompt|dlrate|dsv4layers|dsv4pack|dsv4gen|kimirandom|kimiref|kimicheck|kimilayers|tpcheck|tpbench|tpgather|dcfill|dcreplay|glmkernels|glmlayers|glmtoy|glmrandom|glmdyn|glmmtpreal|glmreal|glmmtp|glmspec|glmprof|glmdequant|glmdecode|glmserve|glmcbtest|glmcbblock|cb2check|cb2prefill|sampletime|nhoffload|\bcpuexperts\b|hbcheck|hbtoy|diskcheck|offload-mode|offload-window|offload-promote|cpu-pin|recompute|glyd-engine (?:pack|generate|check|score|bench|info|stress|bandwidth|chat|ship|ceiling)\b|no-graph|ignore-eos|profile-prefill|attn-decode|"
    r"lost_blocks|heavy_blocks|taken_back|ended_for_loss|values_in_lost|values_differ|values_checked|extra memory|compress(?:es)? (?:too )?poorly|multiple of (?:8|16|64)\b|columns of 16|exceptions? past|\bpass \d of \d", re.I)
# the build machine's own paths (a release build remaps them: release.yml's --remap-path-prefix), as a program holds them in the places of its panics
BUILD_PATH = re.compile(r"(?<![\w.~-])/(?:opt|home|Users|github|root|__w|runner|workspace|build|builds|src|tmp)/[\w./-]*")
# the names the sections and symbols of a cubin may have besides the neutral ones: the compiler's own words (`.nv.info.gk_3`, `.text.gk_3`, `.nv.constant0.gk_3`, `.nv.capmerc.text.gk_3`, ...)
CUBIN_OWN = {"nv", "text", "info", "shared", "local", "global", "rel", "rela", "merc", "capmerc", "debug_frame", "callgraph", "compat", "action", "note", "tkinfo", "cuinfo", "shstrtab", "strtab",
             "symtab", "symtab_shndx", "reserved", "reservedSmem", "cap", "offset0",
             "optimizer"}  # (ptxas's .nv.constant.optimizer section on sm_100 and later: a kernel's immediates hoisted into a constant bank)

# ---- what a reader of the binaries learns ---------------------------------------------------------------------------------------
TABLE = "glyd_gpu_table"  # (glyd_gpu.h: the package's way to every other function of the library)
# what a release library exports, by name: the public header's eight functions and the table's entry point
PUBLIC = {"glyd_gpu_api_version", "glyd_gpu_error_string", "glyd_gpu_mma_linear_workspace", "glyd_gpu_mma_linear", "glyd_gpu_mma12_linear_workspace", "glyd_gpu_mma12_linear",
          "glyd_gpu_mma_unpack", "glyd_gpu_mma12_unpack", TABLE}
NEUTRAL = re.compile(r"g[kdvtn]_\d+")  # (gen_names.py's names: a kernel, a device function, a device variable, a type or namespace)
# the CUDA runtime's, the C and C++ libraries' and the loader's own names, which no source of ours gives
OWN = re.compile(r"\$?(?:__internal_\d+_\$)?__cuda|\.nv|__nv_|cuda[A-Z]|cu[A-Z]|_ZSt|_ZNSt|_ZNKSt|_Zd[al]|_Zn[aw]|_ZT[ISV]|__cxa|__gxx|__aarch64|_Unwind|_ITM|__libc|__stack|__errno|__ctype|__isoc|__f?xstat|__getdelim|__dso|__tls|__bss|_edata|_end$|_fini|_init|GLIBC|GCC_|CXXABI|lib[\w+.-]+\.so|ld-linux|__gmon")
CXX = {"void", "bool", "char", "short", "int", "long", "float", "double", "unsigned", "signed", "const", "true", "false", "__nv_bfloat16", "CUtensorMap_st", "CUstream_st", "std", "int4", "uint4"}
# the variables a release build reads: what the public docs and records tell a user to set (GLYD_KV, GLYD_CACHE, GLYD_COMPILE, GLYD_LAYOUT, GLYD_EXACT,
# GLYD_VERIFY, GLYD_FRACTION, the library's path GLYD_GPU_LIB, and the two that turn a route off, GLYD_SPLIT_MIN=-1 and GLYD_MOE_DECODE_MIN=-1), the one that
# says which engine runs a model (GLYD_ENGINE=0: vLLM; GLYD_ENGINE_ASKED=1, the Rust glyd's and the tool's word to the engine that the user set GLYD_ENGINE=1), the API key of a served model (GLYD_API_KEY: the environment is where a user is told to put it, a flag shows in `ps`)
# and three that are no tuning (GLYD_LOCAL_ONLY: glyd run tells the page's server it is local; GLYD_DEBUG: the traceback of a failure; GLYD_FORWARDED: a glyd that handed a command
# to another says so). A library reads GLYD_SPLIT_MIN alone
ENV_OK = {"GLYD_KV", "GLYD_CACHE", "GLYD_COMPILE", "GLYD_COMPILE_MAX", "GLYD_GPU_LIB", "GLYD_LAYOUT", "GLYD_EXACT", "GLYD_VERIFY", "GLYD_FRACTION", "GLYD_SPLIT_MIN", "GLYD_MOE_DECODE_MIN",
          "GLYD_LOCAL_ONLY", "GLYD_DEBUG", "GLYD_ENGINE", "GLYD_ENGINE_ASKED", "GLYD_FORWARDED", "GLYD_API_KEY"}
LIB_ENV_OK = {"GLYD_SPLIT_MIN"}
# the tuning variables of a development build, which no release build reads: a name of one is in no binary of the wheel (as a substring: the strings
# of a compiled module are run together in its table)
INTERNAL_ENV = re.compile(r"GLYD_(?:KV_(?:RESERVE|SPARE|ADMIT|LONG|STAGING|FLOOR|PIPE|FUSE|PPC|LEAN|WAVES|TABLES|SESSION)|AHEAD|WG_|MID_MIN|DEC_MIN|PACK_(?:EMBED|TOWERS)|SPLIT_(?:MAX|SMS|SLOTS)|MOE_DENSE|MOE_ACTS|GPU_(?:STAGE|MOE|GEMM|NO_SPLIT|WITH_SPLIT|ROUTE|GEFORCE|XMAX|PF|PATCH_B|NA|BIG_|STEP|TMID|TMT2))")
# what no identifier of the sources carries (a part of its name between underscores or case changes), and what stays: a name that is another
# library's (vLLM's and torch's), a format's (glyd.json's tiers) or the public API's
IDENT_DENY = {"tier", "tiers", "tiered", "mantissa", "escape", "escapes", "arena", "staging", "shadow", "ring", "rings", "slot", "slots", "slab", "slabs", "pool", "pools",
              "pages", "route", "routes", "tile", "tiles", "warp", "warps", "sms", "ahead", "wg", "split", "splits", "stage", "page", "raw", "void", "heavy", "mid", "big", "sm", "calibrate"}
IDENT_BARE = {"split", "stage", "page", "raw", "void", "heavy", "mid", "big"}  # (a word of the libraries we call, as a whole name: str.split, ...)
IDENT_SUB = ("mantissa", "arena", "slab", "staging", "shadow", "huffman", "entropy", "wgmma", "mbarrier", "exponent")  # (inside any word: no word of English holds them)
IDENT_KEEP = {"__slots__", "BaseException", "BaseKVCacheMethod", "BrokenPipeError", "c_void_p", "ChatPage", "FusedMoEQuantConfig", "get_fused_moe_quant_config", "LinearBase", "LinearMethodBase",
              "OnlineMoEMethodBase", "UnquantizedFusedMoEMethod", "RawDescriptionHelpFormatter", "requires_calibration", "trust_remote_code", "flash_attn_max_num_splits_for_cuda_graph",
              "split_decodes_and_prefills", "supports_head_size", "slot_mapping", "page_size", "block_pool", "num_splits", "tiers", "_cuda_getCurrentRawStream"}
# what the tables of a compiled module may not hold, as plain substrings (its strings are run together there: a word of a name is a word of a text)
STABLE = re.compile(r"tiered|mantissa|exponent|huffman|entropy|arena|staging|wgmma|mbarrier|cp\.async|done counter|green context|(?:first|second) reserve|12[- ]bits?\b|" + INTERNAL_ENV.pattern, re.I)


def told(header):
    """What a header says that it does not."""
    return re.findall(TOLD, header, flags=re.I)


def method(text, plain=False):
    """The method words text says (the names of GLYD_* variables, which a user types, left out); plain: only the phrases no name is made of."""
    return sorted({m.group().lower() for m in (PLAIN if plain else METHOD).finditer(re.sub(r"\bGLYD_\w+", " ", text))})


def tokens(name):
    """The parts of an identifier, lower-cased (between underscores, and where the case changes)."""
    return [p.lower() for p in re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).split("_") if p]


def method_name(name):
    """The method words an identifier carries: its parts that are IDENT_DENY's (a bare word of IDENT_BARE as a whole name is another library's) and a
    word of IDENT_SUB inside any of them."""
    t = tokens(name)
    if len(t) == 1 and t[0] in IDENT_BARE:
        return []
    return sorted((set(t) & IDENT_DENY) | {w for w in IDENT_SUB if w in name.lower()})


def lzss(src, start, cap=1 << 21):
    """What Cython's LZSS decompression (its string table: __Pyx_DecompressString_LZSS) makes of src from start: the bytes it gives until the
    data stops making sense (the table's text, then what follows it, which is not text)."""
    pos, out = start, bytearray()
    try:
        while True:
            flags = src[pos] | 0xFF00
            pos += 1
            while flags & 0x100:
                if flags & 1:
                    out.append(src[pos])
                    pos += 1
                else:
                    lo, hi = src[pos], src[pos + 1]
                    pos += 2
                    if not lo & 0x80:
                        off, ml = lo, hi
                    elif not hi & 0x80:
                        off, ml = 0x80 + (((hi << 2) & 0x180) | (lo & 0x7F)), hi & 0x1F
                    else:
                        off, ml = 0x80 + ((hi & 0x7F) << 7 | (lo & 0x7F)), src[pos]
                        pos += 1
                    ml += 3
                    ref = len(out) - off - ml
                    if ref < 0:
                        return bytes(out)
                    out += out[ref : ref + ml]
                if len(out) >= cap:
                    return bytes(out)
                flags >>= 1
    except IndexError:
        return bytes(out)


_TABLE_START = re.compile(rb"[\x07\x0f\x17\x1f\x27\x2f\x37\x3f\x47\x4f\x57\x5f\x67\x6f\x77\x7f\x87\x8f\x97\x9f\xa7\xaf\xb7\xbf\xc7\xcf\xd7\xdf\xe7\xef\xf7\xff]")


def string_table(data, marker, names=()):
    """The text of a compiled module's string table (Cython's: its names and its strings, sorted and run together), read from the module's bytes:
    None where it is not found. The table is stored compressed (LZSS), where the module's bytes show only the first of each run of text, and
    the module does not say where it starts or where it ends: every byte that can start a group (its three first items literals) is tried, and
    of the texts that hold marker (a string that is in the table: the module's own source path) the one that has most of names (the identifiers of
    its source) is the table; it ends where the names do, a little later (past the text, the decoder reads what follows the table, which is
    not text: binary tables, and a text made of the pieces of the table's own that holds words no one wrote)."""
    best = None
    for m in _TABLE_START.finditer(data):
        out = lzss(data, m.start())
        if marker in out:
            text = out.decode("utf-8", "replace")
            at = sorted(p for p in (text.find(x) for x in names) if p >= 0)
            head = text[:4096]
            texty = sum(c.isprintable() or c in "\t\n\r" for c in head) / max(1, len(head))
            score = (len(at), texty > 0.5, len(text))
            if best is None or score > best[0]:
                best = (score, text, at)
    if best is None:
        return None
    _, text, at = best
    if len(at) >= 10:
        return text[: at[int(len(at) * 0.98)] + 2000]
    return text


def source_names(text):
    """The identifiers and the string constants that are names or keys, of a module's source: what its table must hold."""
    out = set()
    for n in ast.walk(ast.parse(text)):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
        elif isinstance(n, ast.keyword) and n.arg:
            out.add(n.arg)
        elif isinstance(n, ast.alias):
            out.add(n.name.split(".")[-1])
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and re.fullmatch(r"[A-Za-z_]\w{5,}", n.value):
            out.add(n.value)
    return {x for x in out if len(x) >= 6 and not x.startswith("__")}


def shipped(text):
    """text as a release build compiles it: without its dev blocks (the lines from `# dev` or `# dev: ...` to `# end dev`, where a module reads the
    tuning variables; glyd-gpu-py/setup.py's strip_dev), blank."""
    out, inside = [], False
    for line in text.splitlines(keepends=True):
        mark = line.strip()
        if re.fullmatch(r"# dev(:.*)?", mark):
            inside = True
        elif mark == "# end dev":
            inside = False
        out.append("\n" if inside or mark == "# end dev" else line)
    return "".join(out)


def parse(path):
    """The syntax tree of a source of the package as a release build compiles it."""
    return ast.parse(shipped(open(path, encoding="utf-8").read()))


def modules_of(source):
    return sorted(glob.glob(os.path.join(source, "*.py")))


def docstrings(source):
    """{module: the docstrings of its module, classes and functions}, from the sources in the directory source."""
    out = {}
    for f in modules_of(source):
        tree = parse(f)
        out[os.path.basename(f)[:-3]] = [d for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and (d := ast.get_docstring(n, clean=False))]
    return out


def doc_ids(tree):
    return {id(n.body[0].value) for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.body and isinstance(n.body[0], ast.Expr)}


def prose(source):
    """{module: the strings of its source that are text}, from the sources in the directory source: not its docstrings (the compiled module has none), and not a name or a key ('tiers', 'arena')."""
    out = {}
    for f in modules_of(source):
        tree = parse(f)
        docs = doc_ids(tree)
        out[os.path.basename(f)[:-3]] = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs and not re.fullmatch(r"[\w.]*", n.value)]
    return out


def ident_problems(source):
    """The identifiers of the sources (every name, attribute, argument, keyword, definition and import of a module, and every string that is a name
    or a key) that carry a method word."""
    bad = []
    for f in modules_of(source):
        tree = parse(f)
        docs = doc_ids(tree)
        found = {}
        for n in ast.walk(tree):
            names = []
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.append(n.name)
            elif isinstance(n, ast.Name):
                names.append(n.id)
            elif isinstance(n, ast.Attribute):
                names.append(n.attr)
            elif isinstance(n, ast.arg):
                names.append(n.arg)
            elif isinstance(n, ast.keyword) and n.arg:
                names.append(n.arg)
            elif isinstance(n, ast.alias):
                names += [n.name.split(".")[-1]] + ([n.asname] if n.asname else [])
            elif isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs and re.fullmatch(r"[A-Za-z_]\w*", n.value):
                names.append(n.value)
            for x in names:
                found.setdefault(x, n.lineno)
        for x, line in sorted(found.items()):
            if x in IDENT_KEEP or x.startswith("GLYD_") or (x.startswith("__") and x.endswith("__")):
                continue
            words = method_name(x)
            if words:
                bad.append(f"{os.path.basename(f)}:{line}: [identifier] {x} carries {words}")
    return bad


def env_problems(source):
    """The variables a string of the sources (as a release build compiles them) names that a release build does not read."""
    bad = []
    for f in modules_of(source):
        tree = parse(f)
        docs = doc_ids(tree)
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs:
                bad += [f"{os.path.basename(f)}:{n.lineno}: [env] {v} is named in a release build" for v in sorted(set(re.findall(r"GLYD_[A-Z0-9_]+", n.value)) - ENV_OK)]
    return bad


def names_of(mangled):
    """The identifiers that mangled names (_Z...) spell, demangled by c++filt (or llvm-cxxfilt): None where neither is at hand."""
    tool = shutil.which("c++filt") or shutil.which("llvm-cxxfilt")
    if not tool:
        return None
    out = subprocess.run([tool], input="\n".join(mangled), capture_output=True, text=True).stdout
    return {w for w in re.findall(r"[A-Za-z_]\w*", out) if w not in CXX}


def name_problems(label, runs, neutral=True):
    """What names (runs of printable characters of a library, or the symbols of its machine code) say that they should not: a code name that is
    not neutral, a method word in any name (neutral=False, the engine's kernels: only the method word)."""
    bad = []
    mangled = sorted({m for r in runs for m in re.findall(r"_Z[A-Za-z0-9_]+", r) if not OWN.match(m)})
    if mangled:
        words = names_of(mangled)
        if words is None:  # (no demangler here: the name each one starts with, which its length says, must be neutral; stderr says so)
            print(f"{label}: no c++filt: the mangled names are read for the function's own name alone", file=sys.stderr)
            words = set()
            for m in mangled:
                head = re.match(r"_Z[NK]*(\d+)", m)
                words.add(m[head.end() : head.end() + int(head[1])] if head else m)
        if neutral:
            bad += [f"{label}: [kernel] {w} is a name of the code, not a neutral one" for w in sorted(words) if not NEUTRAL.fullmatch(w) and not OWN.match(w)]
        else:
            bad += [f"{label}: [name] {w} says {x!r}" for w in sorted(words) if not OWN.match(w) for x in method_name(w)]
    plain = sorted(r for r in runs if re.fullmatch(r"[A-Za-z_$.][\w$.]{4,}", r) and not r.startswith("GLYD_") and not OWN.match(r))
    bad += [f"{label}: [name] {r[:80]} says {w!r}" for r in plain for w in method_name(r)]
    # a text (a run with a space, as a message is): the phrases of PHRASES, and the library whose handle the host's products take
    bad += [f"{label}: [words] {r[:80]!r} says {w!r}" for r in sorted(runs) if " " in r and r not in VENDOR_TEXT for w in sorted(set(method(r, plain=True)) | set(re.findall(r"(?i)cublas", r)))]
    return bad


def fatbin_blanked(data):
    """data with the library's machine code (.nv_fatbin, compressed: its bytes are not text, and a run of them that looks like a name is chance) blanked; the names of its kernels,
    functions and variables are read from cuobjdump -symbols (cubin_problems)."""
    shoff = struct.unpack_from("<Q", data, 0x28)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    secs = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize) for i in range(shnum)]
    out = bytearray(data)
    for s in secs:
        start = secs[shstrndx][4] + s[0]
        if data[start : data.index(b"\0", start)] == b".nv_fatbin":
            out[s[4] : s[4] + s[5]] = b" " * s[5]
    return bytes(out)


def lib_problems(label, data, expected=PUBLIC, env_ok=LIB_ENV_OK):
    """What a reader of a library learns that it should not: an export outside expected, a kernel, function or type that is not under a neutral
    name, a method word in a name, a name of a tuning variable that env_ok does not have."""
    bad = []
    if data[:2] == b"MZ":  # (a Windows DLL: the system's DLLs alone imported; CUDA's static runtime adds its one export)
        try:
            exports, imports, functions = pe(data)[4:]
        except (ValueError, struct.error) as e:
            return [f"{label}: [file] {e}"]
        exports = [e for e in exports if e not in PE_LIB_EXPORTS]  # (allowed, not required)
        bad += [f"{label}: [file] needs {n}: the system's DLLs alone are looked for at load time" for n in imports if not WINDOWS_DLL.fullmatch(n)]
        text = os_names_blanked(fatbins_blanked(data).decode("latin-1"), functions)
    else:
        try:
            exports = elf(data)[3]
        except ValueError:
            return [f"{label}: [file] not an ELF64 little-endian file"]
        text = fatbin_blanked(data).decode("latin-1")
    bad += [f"{label}: [export] {e} is exported" for e in sorted(set(exports) - expected)]
    bad += [f"{label}: [export] {e} is not exported" for e in sorted(expected - set(exports))]
    bad += name_problems(label, set(re.findall(r"[\x20-\x7e]{5,}", text)))
    bad += [f"{label}: [env] {v}, a variable that tunes the code, is named" for v in sorted(set(re.findall(r"GLYD_[A-Z0-9_]+", text)) - env_ok)]
    return bad


def no_tool(label, what):
    """The line for a gate that cannot read machine code: cuobjdump is required (a gate that skips what it cannot read passes anything)."""
    return f"{label}: [symbols] the machine code of {what} was not read: cuobjdump is not on PATH, and the gate needs it (PTX, the architectures and the names of the kernels are read with it)"


def cubin_problems(label, path, cuobjdump, neutral=True):
    """The names in the machine code of a library (cuobjdump -symbols: its kernels', device functions' and variables'), where they could be hidden from a
    read of the file's bytes (a compressed fatbin)."""
    out = subprocess.run([cuobjdump, "-symbols", path], capture_output=True, text=True).stdout
    names = {line.split()[-1] for line in out.splitlines() if line.startswith("STT_")}
    return name_problems(label, {n for x in names for n in x.split("$") if n}, neutral)


def elf(data):
    """(machine, the section names, the types of the sections, the symbols the file exports) of an ELF64 little-endian file."""
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise ValueError("not an ELF64 little-endian file")
    machine = struct.unpack_from("<H", data, 18)[0]
    shoff = struct.unpack_from("<Q", data, 0x28)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    secs = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize) for i in range(shnum)]  # name, type, flags, addr, offset, size, link ...

    def text(base, i):
        return data[base + i : data.index(b"\0", base + i)].decode()

    names = {text(secs[shstrndx][4], s[0]) for s in secs}
    exports = []
    for s in secs:
        if s[1] == 11:  # the dynamic symbols
            strings = secs[s[6]][4]
            for k in range(s[5] // 24):
                name, info, _, shndx, _, _ = struct.unpack_from("<IBBHQQ", data, s[4] + 24 * k)
                if name and shndx and info >> 4 in (1, 2):  # defined, global or weak
                    exports.append(text(strings, name))
    return machine, names, {s[1] for s in secs}, exports


def pe(data):
    """(machine, the section names, the COFF symbols' count, the debug directory's CodeView paths, the names the file exports, the DLLs it imports) of a PE32+ file (Windows x64)."""
    if data[:2] != b"MZ" or len(data) < 0x40:
        raise ValueError("not a PE file")
    off = struct.unpack_from("<I", data, 0x3C)[0]
    if data[off : off + 4] != b"PE\0\0" or struct.unpack_from("<H", data, off + 24)[0] != 0x20B:
        raise ValueError("not a PE32+ file")
    machine, nsec = struct.unpack_from("<HH", data, off + 4)
    nsym = struct.unpack_from("<I", data, off + 16)[0]
    opt = off + 24
    dirs = [struct.unpack_from("<II", data, opt + 112 + 8 * i) for i in range(16)]  # export, import, ..., debug (6)
    secs = []
    for i in range(nsec):
        h = opt + struct.unpack_from("<H", data, off + 20)[0] + 40 * i
        vsize, va, rsize, roff = struct.unpack_from("<IIII", data, h + 8)
        secs.append((data[h : h + 8].rstrip(b"\0").decode("latin-1"), va, max(vsize, rsize), roff))

    def at(rva):
        for _, va, size, roff in secs:
            if va <= rva < va + size:
                return roff + rva - va
        raise ValueError(f"not a PE32+ file: an address ({rva:#x}) in no section")

    def text(o):
        return data[o : data.index(b"\0", o)].decode("latin-1")

    exports, imports, debug, functions = [], [], [], []
    if dirs[0][1]:
        e = at(dirs[0][0])
        n, names = struct.unpack_from("<I", data, e + 24)[0], at(struct.unpack_from("<I", data, e + 32)[0])
        exports = [text(at(struct.unpack_from("<I", data, names + 4 * k)[0])) for k in range(n)]
    if dirs[1][1]:
        d = at(dirs[1][0])
        while struct.unpack_from("<I", data, d + 12)[0]:
            imports.append(text(at(struct.unpack_from("<I", data, d + 12)[0])))
            t = struct.unpack_from("<I", data, d)[0] or struct.unpack_from("<I", data, d + 16)[0]  # (the names' table, else the addresses')
            while t and (entry := struct.unpack_from("<Q", data, at(t))[0]):
                if not entry >> 63:  # (by name: a hint, then the name)
                    functions.append(text(at(entry & 0x7FFFFFFF) + 2))
                t += 8
            d += 20
    if dirs[6][1]:
        d = at(dirs[6][0])
        for k in range(dirs[6][1] // 28):
            typ, ptr = struct.unpack_from("<I", data, d + 28 * k + 12)[0], struct.unpack_from("<I", data, d + 28 * k + 24)[0]
            if typ == 2 and data[ptr : ptr + 4] == b"RSDS":  # (CodeView: the PDB's path)
                debug.append(text(ptr + 24))
    return machine, [x[0] for x in secs], nsym, debug, exports, imports, functions


def os_names_blanked(text, functions):
    """text with the names of the functions a PE file imports from the system's DLLs blanked: Windows' own (IsValidCodePage), not the program's wording."""
    for f in sorted(set(functions), key=len, reverse=True):
        text = text.replace(f, " " * len(f))
    return text


def pe_read_only(data):
    """The bytes of a PE file's sections that hold initialized data and no code (.rdata and the like), one after another (elf_read_only's twin)."""
    off = struct.unpack_from("<I", data, 0x3C)[0]
    nsec, opt_size = struct.unpack_from("<H", data, off + 6)[0], struct.unpack_from("<H", data, off + 20)[0]
    out = []
    for i in range(nsec):
        h = off + 24 + opt_size + 40 * i
        rsize, roff = struct.unpack_from("<II", data, h + 16)
        if struct.unpack_from("<I", data, h + 36)[0] & 0x20000040 == 0x40:  # initialized data, not executable
            out.append(bytes(data[roff : roff + rsize]))
    return b"\n".join(out)


def fatbins_blanked(data):
    """data with every CUDA fat binary in it blanked (a PE library's .nv_fatb: compressed bytes, not text)."""
    out = bytearray(data)
    for i, n in fatbins(data):
        out[i : i + n] = b" " * n
    return bytes(out)


def sass_problems(label, path, cuobjdump):
    """No PTX in a library, and machine code for exactly the architectures of SASS."""
    bad = []
    if re.search(r"\.ptx", subprocess.run([cuobjdump, "--list-ptx", path], capture_output=True, text=True).stdout):
        bad.append(f"{label}: [symbols] holds PTX")
    # (an older cuobjdump names a newer toolkit's sm_90a sm_90: the gpu-lib job, with the toolkit that built it, tells them apart)
    got = {a.rstrip("a") for a in re.findall(r"\bsm_\d+a?\b", subprocess.run([cuobjdump, "--list-elf", path], capture_output=True, text=True).stdout)}
    if got != {a.rstrip("a") for a in SASS}:
        bad.append(f"{label}: [symbols] holds machine code for {sorted(got)}, not {sorted(SASS)}")
    return bad


def table_problems(label, data, header):
    """A library's table has an entry for every function header declares (its 32-bit id, FNV-1a of the name without glyd_gpu_, and the word after it, zero)."""
    bad = []
    for fn in sorted(set(re.findall(r"\b(glyd_gpu_\w+)\(", re.sub(r"/\*.*?\*/", "", open(header).read(), flags=re.S))) - {TABLE}):
        h = 2166136261
        for c in fn[len("glyd_gpu_") :].encode():
            h = ((h ^ c) * 16777619) & 0xFFFFFFFF
        if struct.pack("<II", h or 1, 0) not in data:
            bad.append(f"{label}: [export] the table has no entry for {fn}")
    return bad


def needed(data):
    """The libraries the dynamic section of an ELF64 little-endian file names (DT_NEEDED)."""
    shoff = struct.unpack_from("<Q", data, 0x28)[0]
    shentsize, shnum, _ = struct.unpack_from("<HHH", data, 0x3A)
    secs = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize) for i in range(shnum)]
    out = []
    for s in secs:
        if s[1] == 6:  # the dynamic section
            strings = secs[s[6]][4]
            for k in range(s[5] // 16):
                tag, val = struct.unpack_from("<qQ", data, s[4] + 16 * k)
                if tag == 0:
                    break
                if tag == 1:
                    out.append(data[strings + val : data.index(b"\0", strings + val)].decode())
    return out


def fatbins(data):
    """The CUDA fat binaries held in data as bytes (the engine's kernels): [(offset, length)]. A header is the magic, a version (1), its own size (16) and
    the size of what follows."""
    out, at = [], 0
    while (i := data.find(FATBIN, at)) >= 0:
        if i + 16 <= len(data):
            version, head, size = struct.unpack_from("<HHQ", data, i + 4)
            if version == 1 and head == 16 and size > 0 and i + head + size <= len(data):
                out.append((i, head + size))
                at = i + head + size
                continue
        at = i + 4
    return out


def elf_read_only(data):
    """The bytes of the sections of an ELF64 little-endian file that are loaded and are not code (.rodata and the like: where a program's strings are), one after another. The instructions of a
    program have strings' pieces in them too (an x86-64 compiler puts a name it compares in them, 8 bytes at a time): those are not text, and a short word is chance in them."""
    shoff = struct.unpack_from("<Q", data, 0x28)[0]
    shentsize, shnum, _ = struct.unpack_from("<HHH", data, 0x3A)
    out = []
    for i in range(shnum):
        _, typ, flags, _, offset, size, *_ = struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize)
        if flags & 2 and not flags & 4 and typ != 8:  # SHF_ALLOC, not SHF_EXECINSTR, not NOBITS
            out.append(bytes(data[offset : offset + size]))
    return b"\n".join(out)


def elf_names(data):
    """The names of the sections and of the symbols of an ELF64 little-endian file (a cubin's: its kernels' `.text.NAME`, `.nv.info.NAME`, ... and NAME)."""
    shoff = struct.unpack_from("<Q", data, 0x28)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    secs = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize) for i in range(shnum)]

    def text(base, i):
        return data[base + i : data.index(b"\0", base + i)].decode("latin-1")

    out = {text(secs[shstrndx][4], s[0]) for s in secs}
    for s in secs:
        if s[1] == 2:  # the symbol table
            for k in range(s[5] // 24):
                name = struct.unpack_from("<I", data, s[4] + 24 * k)[0]
                if name:
                    out.add(text(secs[s[6]][4], name))
    return {n for n in out if n}


def cubin_name_problems(label, names):
    """What the names of a cubin's sections and symbols say that they should not: a part that is not a neutral name (gen_names.py's), not the compiler's own word
    (CUBIN_OWN) and not a number, and any debug or line section but the frame's (a build with -lineinfo or -G)."""
    bad, mangled = [], set()
    for n in sorted(names):
        if re.search(r"debug(?!_frame)", n) or n.startswith(".nv_debug"):
            bad.append(f"{label}: [symbols] has the section {n}: line or debug information")
        if n.startswith(("$__internal_", "__nv_", "__cuda")):
            continue  # (the compiler's own functions: `$__internal_3_$__cuda_sm20_div_u64`)
        parts = [t for t in re.split(r"[.$]", n) if t]
        mangled |= {t for t in parts if t.startswith("_Z")}  # (a C++ name that is not inlined: read by what it spells)
        stray = [t for t in parts if not t.startswith("_Z") and not NEUTRAL.fullmatch(t) and t not in CUBIN_OWN and not re.fullmatch(r"constant\d*|\d+", t)]
        if stray:
            bad.append(f"{label}: [kernel] {n} says {stray}, not a neutral name: a release build's kernels, functions and types are gk_N, gd_N, gv_N and gt_N")
    return bad + name_problems(label, mangled)


def engine_cubin_problems(label, fatbin, directory, cuobjdump):
    """The machine code of a fat binary the engine holds, each cubin extracted by cuobjdump (-xelf) and its names read: neutral, and no debug section."""
    subprocess.run([cuobjdump, "-xelf", "all", fatbin], cwd=directory, capture_output=True, text=True)
    cubins = sorted(f for f in os.listdir(directory) if f.endswith(".cubin"))
    if not cubins:
        return [f"{label}: [symbols] cuobjdump read no machine code from the program's fat binary"]
    bad = []
    for f in cubins:
        with open(os.path.join(directory, f), "rb") as fh:
            data = fh.read()
        try:
            bad += cubin_name_problems(f"{label}: {f.rsplit('.', 2)[1]}", elf_names(data))
        except (ValueError, struct.error):
            bad.append(f"{label}: [symbols] {f} is not an ELF64 file")
    return bad


def internal_names(header):
    """The names of the library's functions that its public header does not declare (gpu/glyd_gpu.h's, with `glyd_gpu_` and without; a name of 8 letters or more with an underscore): what the
    library's table hides from a reader of the library, which a program that calls them by name would say."""
    fns = set(re.findall(r"\b(glyd_gpu_\w+)\(", re.sub(r"/\*.*?\*/", "", open(header).read(), flags=re.S))) - PUBLIC
    return {n for f in fns for n in (f, f[len("glyd_gpu_") :]) if "_" in n and len(n) >= 8}


def engine_kernel_names(root):
    """The names of the kernels of the engine's CUDA sources (root: the glyd-engine directory), as gpu/gen_names.py reads them: a release build's program names none of them."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("gen_names", os.path.join(root, "..", "gpu", "gen_names.py"))
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    src = "".join(open(os.path.join(root, "kernels", f), encoding="utf-8").read() for f in ("gemm_idx.h", "engine.cu"))
    return set(gen.names(src)["k"])


def engine_problems(label, data, arch=None, cuobjdump=None, vouched=False, kernels=(), internal=()):
    """What the engine's program (a Rust ELF with its kernels as a fat binary in it) says or does that it should not: not for the architecture, a symbol
    table or debug sections, an export, a library other than the system's, no machine code, PTX, line information, a kernel or function that is not under a neutral name, a method
    word or phrase or a word of STABLE (but the dependencies' own), a development command or flag or internal count (ENGINE_DENY), a variable that tunes the code, the build machine's
    path, a kernel of `kernels` or a function of `internal` named. cuobjdump (None: the one on PATH; False: not run, for the tests' stand-ins for the machine code) is required unless
    `vouched` (the bytes are those the engine job checked)."""
    bad = []
    windows = data[:2] == b"MZ"
    if windows:  # (a Windows program: the system's DLLs alone imported, the C runtime linked in; a PDB named by its file name alone)
        try:
            machine, sections, nsym, debug, exports, imports, functions = pe(data)
        except (ValueError, struct.error) as e:
            return [f"{label}: [file] {e}"]
        if arch and machine != PE_ARCHS[arch]:
            bad.append(f"{label}: [file] is for PE machine {machine:#x}, not {arch}'s")
        if nsym or any(re.search(r"[\\/:]", p) for p in debug):
            bad.append(f"{label}: [symbols] has a symbol table, or names its debug file by a path: {debug}")
        bad += [f"{label}: [export] {e} is exported" for e in sorted(exports)]
        bad += [f"{label}: [file] needs {n}: the system's DLLs alone are looked for at load time" for n in imports if not WINDOWS_DLL.fullmatch(n)]
    else:
        try:
            machine, sections, types, exports = elf(data)
        except ValueError:
            return [f"{label}: [file] not an ELF64 little-endian file"]
        if arch and machine != ARCHS[arch]:
            bad.append(f"{label}: [file] is for ELF machine {machine}, not {arch}'s")
        if 2 in types or any(s.startswith((".debug", ".zdebug")) for s in sections):
            bad.append(f"{label}: [symbols] has a symbol table or debug sections")
        bad += [f"{label}: [export] {e} is exported" for e in sorted(exports)]
        bad += [f"{label}: [file] needs {n}: the system's libraries alone are looked for at load time" for n in needed(data) if not SYSTEM_LIB.fullmatch(n)]
    fats = fatbins(data)
    if not fats:
        bad.append(f"{label}: [symbols] holds no machine code for the GPU (built without nvcc?)")
    blank = bytearray(data)  # (the machine code is compressed: its bytes are not text, and a run of them that looks like a word is chance)
    for i, n in fats:
        blank[i : i + n] = b" " * n
    if windows:  # (the names of the functions it imports from the system's DLLs: Windows' own, not the program's wording)
        blank = bytearray(os_names_blanked(bytes(blank).decode("latin-1"), functions).encode("latin-1"))
    text = blank.decode("latin-1")
    masked = text
    for rx in ENGINE_FOREIGN:
        masked = rx.sub(" ", masked)
    # the phrases anywhere; the words in the program's data (its strings), where they are text: in its instructions a short word is a piece of a name it compares, or chance (the Rust sources are read for those)
    data_text = (pe_read_only if windows else elf_read_only)(blank).decode("latin-1")
    for rx in ENGINE_FOREIGN:
        data_text = rx.sub(" ", data_text)
    # (a short run with no space and no lower-case letter, such as "G@SM", is the bytes of a number in a table, not text)
    runs = "\n".join(r for r in re.findall(r"[\x20-\x7e]{4,}", data_text) if len(r) >= 8 or " " in r or re.search(r"[a-z]", r))
    bad += [f"{label}: [words] holds the text {w!r}" for w in sorted(set(method(masked, plain=True)) | set(method(runs)))]
    # where each word of the program's strings is: the run it is in, so a failure names the string and not only the word
    bad += [f"{label}: [words] {w!r} is in {r[:160]!r}" for w in sorted(set(method(runs))) for r in runs.split("\n") if w in method(r)][:20]
    bad += [f"{label}: [table] holds {w!r}" for w in sorted({m.group().lower() for m in STABLE.finditer(masked)})]
    bad += [f"{label}: [dev] holds {w!r}, a development build's or a wording of how" for w in sorted({m.group().lower() for m in ENGINE_DENY.finditer(masked)})]
    bad += [f"{label}: [path] holds {p!r}, a path of the machine that built it (release.yml remaps them)" for p in sorted({"/".join(m.group().split("/")[:4]) for m in BUILD_PATH.finditer(text)})]
    bad += [f"{label}: [path] holds {p!r}, a path of the machine that built it (remap it: --remap-path-prefix)" for p in sorted({"\\".join(m.group().split("\\")[:3]) for m in WIN_BUILD_PATH.finditer(text)})]
    bad += [f"{label}: [kernel] names {n}, a kernel of its sources (a release build names none)" for n in sorted(kernels) if n in masked]  # (as a substring: a program's strings are run together)
    bad += [f"{label}: [name] names {n}, a function of the library that its table hides" for n in sorted(internal) if n in masked]
    # each GLYD_ is a name's start (strings run together). Where the code compares a name, an x86-64 compiler puts it in the instructions as 8-byte immediates (GLYD_CAC,
    # then HE): a name's first eight bytes are the allowed name's as well
    ok = ENGINE_ENV_OK
    names = set()
    for m in re.finditer(r"GLYD_", text):
        if text.startswith(ENGINE_SAYS, m.start()) or any(text.startswith(a, m.start()) for a in ok):
            continue
        name = text[m.start() : m.start() + 48].split("\0")[0]
        # loaded by movabs (REX.W 0x48/0x49, then 0xb8-0xbf): 8 bytes a piece, joined while the next instruction is another movabs (the byte after a piece is the next
        # instruction's, often 0x48, an 'H', which is not the name's)
        i, loaded = m.start(), False
        if i >= 2 and text[i - 2] in "HI" and "\xb8" <= text[i - 1] <= "\xbf":
            name, loaded = "", True
            while True:
                name += text[i : i + 8]
                i += 8
                if not (text[i : i + 1] in ("H", "I") and "\xb8" <= text[i + 1 : i + 2] <= "\xbf"):
                    break
                i += 2
        word = re.match(r"[A-Z0-9_]*", name).group()
        if (loaded or len(word) <= 8) and any(a.startswith(word) for a in ok):  # (an allowed name's immediates, or its first eight bytes; a longer name is its own)
            continue
        names.add(name)
    bad += [f"{label}: [env] {v}, a variable that tunes the code, is named" for v in sorted(names)]
    if cuobjdump is None:
        cuobjdump = shutil.which("cuobjdump")
    if fats and cuobjdump:
        with tempfile.TemporaryDirectory() as d:
            for k, (i, n) in enumerate(fats):
                p = os.path.join(d, f"engine{k}.fatbin")
                with open(p, "wb") as f:
                    f.write(data[i : i + n])
                bad += sass_problems(label, p, cuobjdump)
                bad += engine_cubin_problems(label, p, d, cuobjdump)
    elif fats and cuobjdump is None and not vouched:
        bad.append(no_tool(label, "the program"))
    return bad


# the Rust sources of the engine as its release build compiles them: no module declared under #[cfg(feature = "dev")], no item or arm under it, no tests (the
# `#[cfg(test)] mod tests` tail of a file; any other #[cfg(test)] item is read), no tune!("NAME") (a development build's variable: None in a release build)
DEV_ATTR = re.compile(r'#\[cfg\(feature = "dev"\)\]')
RUST_STR = re.compile(r'(?:b?r(#*)")|(?:b?")')
RUST_CHAR = re.compile(r"'(?:\\[^']*|[^\\'])'")


def rust_skip(src, i):
    """The index after the item, statement or match arm that starts at i (after its attribute): its first `;` or `,` outside any bracket, or the end of its first
    `{ ... }` block (and the `,` after it); a `let` goes on to its `;`, and a block that an `else` follows takes the `else`'s too."""
    n = len(src)
    head = re.compile(r"\s*(?:pub(?:\([^)]*\))?\s+)?(\w+)").match(src, i)
    let = bool(head) and head.group(1) == "let"
    depth = 0
    while i < n:
        c = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            i = n if j < 0 else j + 2
        elif c == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            i = j + 1
        elif c == "'" and RUST_CHAR.match(src, i):
            i = RUST_CHAR.match(src, i).end()
        elif c in "([{":
            depth += 1
            i += 1
        elif c in ")]}":
            depth -= 1
            i += 1
            if depth == 0 and c == "}":
                after = re.compile(r"\s*else\b").match(src, i)
                if after:
                    i = after.end()
                elif not let:
                    m = re.compile(r"\s*,").match(src, i)
                    return m.end() if m else i
        elif depth == 0 and c in ";,":
            return i + 1
        else:
            i += 1
    return n


def rust_block(src, i):
    """The index after the `{ ... }` block that starts at src[i] (the strings, characters and comments in it left out of the count of its braces)."""
    n, depth = len(src), 0
    while i < n:
        c = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            i = n if j < 0 else j + 2
        elif c == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            i = j + 1
        elif c == "'" and RUST_CHAR.match(src, i):
            i = RUST_CHAR.match(src, i).end()
        else:
            depth += (c == "{") - (c == "}")
            i += 1
            if depth == 0 and c == "}":
                return i
    return n


def rust_shipped(text):
    """text (a Rust source of the engine) as a release build compiles it, with the lines of what is left where they were."""
    cut = re.search(r"^#\[cfg\(test\)\]\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+\w+", text, re.M)
    if cut:
        text = text[: cut.start()] + re.sub(r"[^\n]", " ", text[cut.start() :])
    out, pos = [], 0
    for m in DEV_ATTR.finditer(text):
        if m.start() < pos:
            continue
        out.append(text[pos : m.start()])
        end = rust_skip(text, m.end())
        out.append(re.sub(r"[^\n]", " ", text[m.start() : end]))
        pos = end
    out.append(text[pos:])
    text = re.sub(r'(?:crate::|glyd_engine::)?tune!\(\s*"[A-Z0-9_]+"\s*\)', "None", "".join(out))
    # `cfg!(feature = "dev") && args.has("flag")` is false in a release build, and `if cfg!(feature = "dev") { dev } else { release }` is its else block alone
    text = re.sub(r'cfg!\(feature = "dev"\)\s*&&\s*[\w.]+\.has\("[\w-]+"\)', "false", text)
    while m := re.search(r'\bif\s+cfg!\(feature = "dev"\)\s*\{', text):
        end = rust_block(text, m.end() - 1)
        text = text[: m.start()] + re.sub(r"[^\n]", " ", text[m.start() : end]) + text[end:]
    return text


def rust_strings(src):
    """[(line, text)] of the string literals of a Rust source (byte strings and raw strings too), not the comments."""
    out, i, n, line = [], 0, len(src), 1
    while i < n:
        c = src[i]
        if c == "\n":
            line += 1
            i += 1
        elif src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
        elif src.startswith("/*", i):
            depth, i = 1, i + 2
            while i < n and depth:
                if src.startswith("/*", i):
                    depth, i = depth + 1, i + 2
                elif src.startswith("*/", i):
                    depth, i = depth - 1, i + 2
                else:
                    line += src[i] == "\n"
                    i += 1
        else:
            m = RUST_STR.match(src, i)
            if m and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == "_")):
                if m.group(1) is not None:  # r"..." and r#"..."#
                    end = src.find('"' + m.group(1), m.end())
                    end = n if end < 0 else end
                    out.append((line, src[m.end() : end]))
                    line += src.count("\n", m.end(), end)
                    i = end + 1 + len(m.group(1))
                else:
                    j, buf = m.end(), []
                    while j < n and src[j] != '"':
                        if src[j] == "\\" and src[j + 1] == "\n":  # a line continued: the next line's indentation is not the text
                            j += 2
                            line += 1
                            while j < n and src[j] in " \t\n":
                                line += src[j] == "\n"
                                j += 1
                        elif src[j] == "\\":
                            buf.append(src[j : j + 2])
                            j += 2
                        else:
                            line += src[j] == "\n"
                            buf.append(src[j])
                            j += 1
                    out.append((line, "".join(buf)))
                    i = j + 1
            elif c == "'":
                m2 = RUST_CHAR.match(src, i)
                i = m2.end() if m2 else i + 1
            else:
                i += 1
    return out


def engine_dev_modules(srcdir):
    """The files of the engine's modules that only a development build has: `#[cfg(feature = "dev")] mod NAME;` of lib.rs and main.rs."""
    mods = set()
    for f in ("lib.rs", "main.rs"):
        with open(os.path.join(srcdir, f), encoding="utf-8") as fh:
            mods |= {m.group(1) + ".rs" for m in re.finditer(DEV_ATTR.pattern + r"\s*(?:pub\s+)?mod\s+(\w+);", fh.read())}
    return mods


def engine_source_problems(root):
    """What the strings of the engine's release build say (root: the glyd-engine directory): of every string literal of its Rust sources, a name or key that carries
    a method word (IDENT_DENY, but a format's key: ENGINE_KEEP), a text that says one (METHOD; the names of inline format arguments are not in the binary), a variable
    that tunes the code (ENGINE_ENV_OK is what it reads), a development command or flag, an internal count or a wording of how (ENGINE_DENY); and the page it serves."""
    srcdir = os.path.join(root, "src")
    skip = engine_dev_modules(srcdir)
    bad = []
    for f in sorted(glob.glob(os.path.join(srcdir, "*.rs"))):
        base = os.path.basename(f)
        if base in skip:
            continue
        with open(f, encoding="utf-8") as fh:
            strings = rust_strings(rust_shipped(fh.read()))
        for line, s in strings:
            deny = sorted({m.group().lower() for m in ENGINE_DENY.finditer(s)})
            if deny:
                bad.append(f"engine src/{base}:{line}: [dev] a string says {deny}: {s[:70]!r}")
            if re.fullmatch(r"[\w.]*", s):  # (a name or a key, not a text)
                if re.fullmatch(r"GLYD_[A-Z0-9_]+", s) and s not in ENGINE_ENV_OK | {"GLYD_SOURCE_COMMIT"}:  # (a variable read by its name, std::env::var("GLYD_..."); the build's commit is option_env!'s, its value alone in the binary)
                    bad.append(f"engine src/{base}:{line}: [env] {s} is named in a release build")
                    continue
                words = method_name(s) if re.fullmatch(r"[A-Za-z_]\w*", s) and s not in ENGINE_KEEP else []
                if words:
                    bad.append(f"engine src/{base}:{line}: [identifier] {s} carries {words}")
                continue
            said = s
            for phrase in ENGINE_SAYS:
                said = said.replace(phrase, " ")
            bad += [f"engine src/{base}:{line}: [env] {v} is named in a release build" for v in sorted(set(re.findall(r"GLYD_[A-Z0-9_]+", said)) - ENGINE_ENV_OK)]
            say = method(re.sub(r"\{[A-Za-z_][\w.]*(?::[^}]*)?\}", "{}", s))
            if say:
                bad.append(f"engine src/{base}:{line}: [words] a string says {say}: {s[:70]!r}")
    page = os.path.join(root, "data", "chat.html")
    if os.path.exists(page):
        with open(page, encoding="utf-8") as fh:
            bad += [f"engine data/chat.html: [words] says {w!r}" for w in method(fh.read())]
    return bad


def check_engine(path, root=None, cuobjdump=None, table_header=None):
    """The problems of the engine's program alone (the engine job's), and of its sources where root is given (the names of their kernels are not the program's), and of what it names of the library
    where table_header (gpu/glyd_gpu.h) is given. cuobjdump is required."""
    with open(path, "rb") as f:
        bad = engine_problems(os.path.basename(path), f.read(), None, cuobjdump, kernels=engine_kernel_names(root) if root else (), internal=internal_names(table_header) if table_header else ())
    return bad + (engine_source_problems(root) if root else [])


def check_lib(path, header=None, table_header=None, cuobjdump=None):
    """The problems of a library alone (the gpu-lib job's); cuobjdump (None: the one on PATH; False: not run, for the tests) is required."""
    if cuobjdump is None:
        cuobjdump = shutil.which("cuobjdump")  # (None again where it is missing: a line below)
    label = os.path.basename(path)
    data = open(path, "rb").read()
    bad = lib_problems(label, data)
    windows = data[:2] == b"MZ"
    try:
        if windows:
            _, _, nsym, debug, _, _, _ = pe(data)
            if nsym or debug:
                bad.append(f"{label}: [symbols] has a symbol table or a debug directory")
        else:
            machine, sections, types, _ = elf(data)
            if 2 in types or any(s.startswith((".debug", ".zdebug")) for s in sections):
                bad.append(f"{label}: [symbols] has a symbol table or debug sections")
    except (ValueError, struct.error):
        pass
    if table_header:
        bad += table_problems(label, data, table_header)
    if cuobjdump and windows:  # (cuobjdump reads a Linux host binary, not a DLL: the fat binaries it holds are read on their own)
        with tempfile.TemporaryDirectory() as d:
            for k, (i, n) in enumerate(fatbins(data)):
                f = os.path.join(d, f"lib{k}.fatbin")
                with open(f, "wb") as fh:
                    fh.write(data[i : i + n])
                bad += sass_problems(label, f, cuobjdump)
                bad += cubin_problems(label, f, cuobjdump)
            if not fatbins(data):
                bad.append(f"{label}: [symbols] holds no machine code for the GPU")
    elif cuobjdump:
        bad += sass_problems(label, path, cuobjdump)
        bad += cubin_problems(label, path, cuobjdump)
    elif cuobjdump is None:
        bad.append(no_tool(label, "the library"))
    return bad


def check(path, version=None, platform=None, source=None, header=None, licence=None, sums=None, libs=True, cuobjdump=None, engine_source=None, engine=None, table_header=None):
    """The problems of the wheel at path: a list of lines (empty: it passes). cuobjdump: the tool's path (default: the one on PATH; False: not run, for the tests'
    stand-ins). Where there is none the machine code of a library or of the engine is not read here, and is accepted only where `sums` (sha256sum's lines: the
    gpu-lib and engine jobs', which read it with cuobjdump) vouch for the very bytes. engine: whether the wheel has the engine's program (default: where it has the
    libraries); engine_source: the glyd-engine directory, whose Rust sources are read as its release build compiles them, and whose kernels' names the program
    does not hold; table_header: gpu/glyd_gpu.h, whose internal functions the program does not name."""
    if cuobjdump is None:
        cuobjdump = shutil.which("cuobjdump")  # (None again where it is missing)
    known = {}  # sha256sum's lines: the digest, then the file's path
    for f in sums or []:
        for line in open(f):
            if line.strip():
                known[os.path.basename(line.split()[-1])] = line.split()[0]
    bad = []
    name = os.path.basename(path)
    m = re.fullmatch(r"glyd_gpu-(?P<v>[^-]+)-(?P<py>cp3\d+t?)-(?P<abi>[^-]+)-(?P<plat>[^-]+)\.whl", name)
    if not m:
        return [f"{name}: [file] not a glyd_gpu wheel's file name"]
    if version and m["v"] != version:
        bad.append(f"{name}: [file] version {m['v']}, not {version}")
    if platform and m["plat"] != platform:
        bad.append(f"{name}: [file] platform {m['plat']}, not {platform}")
    arch = "aarch64" if "aarch64" in m["plat"] else "x86_64"
    if os.path.getsize(path) > MAX_BYTES:
        bad.append(f"{name}: [file] {os.path.getsize(path)} bytes, past PyPI's limit")
    z = zipfile.ZipFile(path)
    members = [i.filename for i in z.infolist() if not i.is_dir()]
    modules = {}
    for f in members:
        d, _, base = f.rpartition("/")
        module = MODULE.fullmatch(base) if d == "glyd_gpu" else None
        if module:
            modules[module["mod"]] = f
        elif not (d == "glyd_gpu" and base in FILES) and not DIST.fullmatch(f):
            bad.append(f"{name}: [file] {f} is not a file of a glyd-gpu wheel")
    sources = sorted(f for f in members if f.endswith(".py"))
    if sources != [f"glyd_gpu/{s}.py" for s in sorted(PYTHON_SOURCE)]:
        bad.append(f"{name}: [file] Python source {sources}, not {sorted(PYTHON_SOURCE)}")
    present = set(f.rpartition("/")[2] for f in members)
    if libs:
        bad += [f"{name}: [file] no {lib}" for lib in LIBS if lib not in present]
    else:
        bad += [f"{name}: [file] {lib} where there is none to be" for lib in LIBS if lib in present]
    engine = libs if engine is None else engine
    if engine and ENGINE not in present:
        bad.append(f"{name}: [file] no {ENGINE}")
    elif not engine and ENGINE in present:
        bad.append(f"{name}: [file] {ENGINE} where there is none to be")
    for need in ("glyd_gpu.h", "chat.html", "kvcorpus.bin"):
        if need not in present:
            bad.append(f"{name}: [file] no {need}")
    docs, names = {}, {}
    if source:
        docs = docstrings(source)
        expected = set(docs) - PYTHON_SOURCE
        if set(modules) != expected:
            bad.append(f"{name}: [file] compiled modules {sorted(modules)}, the sources have {sorted(expected)}")
        names = {os.path.basename(f)[:-3]: source_names(shipped(open(f, encoding="utf-8").read())) for f in modules_of(source)}
    for mod, f in sorted(modules.items()) + [(None, f"glyd_gpu/{lib}") for lib in LIBS if f"glyd_gpu/{lib}" in members]:
        data = z.read(f)
        try:
            machine, sections, types, exports = elf(data)
        except ValueError as e:
            bad.append(f"{name}: [file] {f}: {e}")
            continue
        if machine != ARCHS[arch]:
            bad.append(f"{name}: [file] {f} is for ELF machine {machine}, not {arch}'s")
        if 2 in types or any(s.startswith((".debug", ".zdebug")) for s in sections):
            bad.append(f"{name}: [symbols] {f} has a symbol table or debug sections")
        if mod:
            if set(exports) != {"PyInit_" + ("glyd_gpu" if mod == "__init__" else mod)}:  # (the package's: named for the package)
                bad.append(f"{name}: [export] {f} exports {sorted(exports)}")
            if source and any(d.strip()[:40].encode() in data for d in docs.get(mod, []) if len(d.strip()) >= 40):
                bad.append(f"{name}: [docstring] {f} holds a docstring of its source")
            bad += [f"{name}: [words] {f} holds the text {w!r}" for w in method(data.decode("latin-1"), plain=True)]
            table = string_table(data, f"glyd_gpu/{mod}.py".encode(), names.get(mod, ()))
            text = (table or "") + "\n" + data.decode("latin-1")
            bad += [f"{name}: [table] {f} holds {w!r}" for w in sorted({m.group().lower() for m in STABLE.finditer(text)})]
            have = names.get(mod)
            if have and sum(x in text for x in have) < 0.9 * len(have):  # (the reader is blind where it finds less than the source has: the gate would pass anything)
                bad.append(f"{name}: [table] the names of {mod}.py are not in {f} where this reads them ({sum(x in text for x in have)} of {len(have)})")
        else:
            bad += lib_problems(f"{name}: {f}", data)
    if header:
        fns = set(re.findall(r"\b(glyd_gpu_\w+)\(", re.sub(r"/\*.*?\*/", "", open(header).read(), flags=re.S)))
        if fns | {TABLE} != PUBLIC:
            bad.append(f"{name}: [export] {header} declares {sorted(fns ^ (PUBLIC - {TABLE}))} other than the eight functions")
    if source:  # (a compiled module's strings are compressed in it: read where they come from)
        for mod, texts in prose(source).items():
            bad += [f"{name}: [words] a string of {mod}.py says {w!r}" for w in sorted({w for t in texts for w in method(t)})]
        bad += [f"{name}: {line}" for line in ident_problems(source) + env_problems(source)]
    for f in members:
        if (f.startswith("glyd_gpu/") and f.endswith((".py", ".html"))) or f.endswith(".dist-info/METADATA"):  # (the page of the wheel at PyPI is its METADATA)
            bad += [f"{name}: [words] {f} says {w!r}" for w in method(z.read(f).decode("utf-8", "replace"))]
    eng = f"glyd_gpu/{ENGINE}"
    if engine and eng in members:
        if not (z.getinfo(eng).external_attr >> 16) & 0o111:  # (an installer sets a file's mode from here)
            bad.append(f"{name}: [file] {ENGINE} is not executable in the wheel")
        bad += engine_problems(f"{name}: {eng}", z.read(eng), arch, cuobjdump, known.get(ENGINE) == hashlib.sha256(z.read(eng)).hexdigest(),
                               engine_kernel_names(engine_source) if engine_source else (), internal_names(table_header) if table_header else ())
    if engine_source:
        bad += [f"{name}: {line}" for line in engine_source_problems(engine_source)]
    h = "glyd_gpu/glyd_gpu.h"
    if h in members:
        text = z.read(h).decode()
        if header and text != open(header).read():
            bad.append(f"{name}: [file] {h} is not {header}")
        bad += [f"{name}: [words] the header says {w!r}" for w in sorted(set(told(text)))]
    if sums:
        for lib in LIBS:
            if f"glyd_gpu/{lib}" in members and known.get(lib) != hashlib.sha256(z.read(f"glyd_gpu/{lib}")).hexdigest():
                bad.append(f"{name}: [file] {lib} is not the library the gpu-lib job checked (sha256)")
        if engine and eng in members and known.get(ENGINE) != hashlib.sha256(z.read(eng)).hexdigest():
            bad.append(f"{name}: [file] {ENGINE} is not the program the engine job checked (sha256)")
    if libs and cuobjdump:
        with tempfile.TemporaryDirectory() as d:
            for lib in LIBS:
                if f"glyd_gpu/{lib}" in members:
                    p = os.path.join(d, lib)
                    open(p, "wb").write(z.read(f"glyd_gpu/{lib}"))
                    bad += sass_problems(f"{name}: {lib}", p, cuobjdump)
                    bad += cubin_problems(f"{name}: {lib}", p, cuobjdump)
    elif libs and cuobjdump is None:
        bad += [no_tool(f"{name}: {lib}", "the library") for lib in LIBS if f"glyd_gpu/{lib}" in members and known.get(lib) != hashlib.sha256(z.read(f"glyd_gpu/{lib}")).hexdigest()]
    meta = [f for f in members if f.endswith(".dist-info/METADATA")]
    if meta:
        md = email.parser.Parser().parsestr(z.read(meta[0]).decode())
        plain = [r for r in (md.get_all("Requires-Dist") or []) if "extra ==" not in r]
        if plain:
            bad.append(f"{name}: [meta] dependencies without an extra: {plain}")
        if set(md.get_all("Provides-Extra") or []) != EXTRAS:
            bad.append(f"{name}: [meta] extras {md.get_all('Provides-Extra')}, not {sorted(EXTRAS)}")
        if md["Name"] != "glyd-gpu" or md.get("License-Expression") != "BUSL-1.1" or "LICENSE" not in (md.get_all("License-File") or []):
            bad.append(f"{name}: [meta] Name, License-Expression or License-File is not glyd-gpu's, BUSL-1.1 and LICENSE")
        if md["Version"] != m["v"]:
            bad.append(f"{name}: [meta] METADATA's version {md['Version']}")
    if licence:
        lic = [f for f in members if f.endswith("licenses/LICENSE")]
        if not lic or z.read(lic[0]) != open(licence, "rb").read():
            bad.append(f"{name}: [meta] the licence is not {licence}")
    return bad


def counts(bad):
    """{tag: how many of the lines carry it}."""
    return dict(sorted(collections.Counter((re.search(r"\[(\w+)\]", line) or [None, "other"])[1] for line in bad).items()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("wheels", nargs="*")
    ap.add_argument("--lib", nargs="+", help="libraries alone (the gpu-lib job's): names, exports, machine code")
    ap.add_argument("--engine", nargs="+", help="the engine's program alone (the engine job's): what it is, its strings, its machine code")
    ap.add_argument("--engine-source", help="the glyd-engine directory: its Rust sources are read as its release build compiles them (with --engine, or the wheels)")
    ap.add_argument("--version")
    ap.add_argument("--platform")
    ap.add_argument("--source", help="the directory of the package's sources (glyd_gpu/): its modules, and their docstrings")
    ap.add_argument("--header", help="the public header the wheel's is")
    ap.add_argument("--table-header", help="the header that declares every function (gpu/glyd_gpu.h): with --lib, the table has an entry for each; with --engine or a wheel, the program names none of those the public header does not")
    ap.add_argument("--license", dest="licence")
    ap.add_argument("--sums", nargs="*", help="sha256 files of the libraries the gpu-lib job checked")
    ap.add_argument("--no-libs", action="store_true", help="a wheel built without the CUDA libraries and the engine (CI's compile check)")
    ap.add_argument("--no-engine", action="store_true", help="a wheel with the libraries but not the engine's program")
    ap.add_argument("--summary", action="store_true", help="end with the count of each kind of line")
    a = ap.parse_args(argv)
    if sum(map(bool, (a.wheels, a.lib, a.engine))) > 1 or not (a.wheels or a.lib or a.engine or a.engine_source):
        ap.error("wheels, or --lib libraries, or --engine programs, or --engine-source alone")
    bad = []
    if a.engine_source and not (a.wheels or a.lib or a.engine):  # (the Rust sources alone: ci.yml's)
        bad = engine_source_problems(a.engine_source)
        print(f"{'FAIL' if bad else 'ok  '} {a.engine_source}")
    for w in a.lib or a.engine or a.wheels:
        if a.lib:
            problems = check_lib(w, a.header, a.table_header)
        elif a.engine:
            problems = check_engine(w, a.engine_source, table_header=a.table_header)
        else:
            problems = check(w, a.version, a.platform, a.source, a.header, a.licence, a.sums, not a.no_libs, engine_source=a.engine_source, engine=not (a.no_libs or a.no_engine), table_header=a.table_header)
        print(f"{'FAIL' if problems else 'ok  '} {os.path.basename(w)}")
        bad += problems
    for line in bad:
        print("  " + line)
    if a.summary:
        print(f"  {len(bad)} lines: " + ", ".join(f"{k} {v}" for k, v in counts(bad).items()))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
