# Glyd quant names (draft, 2026-10-15, owner: whole numbers in the names)

Named like GGUF's types: the code bits in the name, the true bits per weight in the table (a scale per run of 32 weights adds 0.5).
Public text gives these numbers and nothing about how the codes are stored.

| Name     | Level        | What it is                                   | Bits/weight, true          | Lossless?                       |
|----------|--------------|----------------------------------------------|----------------------------|---------------------------------|
| GLYD-P   | Penguin      | lossless compression of the bf16 weights      | ~10.8 (model dependent)    | yes, bit for bit                |
| GLYD-K6  | Kestrel      | 6-bit codes, a scale per 32                   | 6.5                        | no                              |
| GLYD-S5  | Swift        | 5-bit codes, a scale per 32                   | 5.5                        | no                              |
| GLYD-W4  | Wren         | 4-bit codes, a scale per 32                   | 4.5                        | no                              |
| GLYD-X4  | (QAT models) | exact 4-bit: a 4-bit-trained checkpoint kept bit for bit | 4.5             | yes vs the QAT release, not vs bf16 |
| GLYD-HB3 | Hummingbird  | MoE: experts by use (precise mode)            | experts ~2.7, rest 6.5     | no                              |
| GLYD-HB2 | Hummingbird  | MoE: experts by use (fast mode)               | experts ~2.1, rest 6.5     | no                              |

Every release lists, per name: size on disk, mean KLD and 99.9% KLD vs bf16 (wikitext-2 test, 512-token chunks, the same way
llama.cpp's --kl-divergence tables are made), PPL ratio, top-1 agreement, tok/s on named cards.

## Hugging Face registration (needs owner approval: it is a public PR to huggingface.js)
An entry in packages/tasks/src/model-libraries.ts so repos with `library_name: glyd` get a library tag, a "Use this model" snippet
and a page listing every Glyd model. Downloads: the per-model repos (glyd/Qwen3.5-9B-kestrel ...) have config.json and glyd.json at
the root and already count; the Qwen Flash repo has neither at the root (counts 0) until the small root config.json is uploaded
(download.sh already asks for it, cea26db). So the rule below counts config.json, as the default does:

    glyd: {
        prettyLabel: "Glyd",
        repoName: "glyd",
        repoUrl: "https://github.com/surya-koritala/glyd-qwen-3.8-flash",   // or the main Glyd repo, owner to pick
        docsUrl: "https://getglyd.com",
        filter: false,
        countDownloads: `path:"config.json"`,
    },

Then every glyd/* card gets `library_name: glyd` in its YAML.
