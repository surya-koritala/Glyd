//! The byte-level BPE pre-tokenizers of GGUF files (`tokenizer.ggml.pre`): for each, the regular expressions the text is split with, in order, whether a word that is a token whole is not
//! merged, and whether a text is given the beginning-of-sequence token where the file does not say. The expressions are the ones llama.cpp's llama-vocab.cpp (MIT licence) tokenizes with, which
//! is what a GGUF's vocabulary was made for; a name this does not have is refused by `gguf_tok::files` (so are the two whose expressions only name a hand-written splitter there, Kimi-K2's
//! and AFMoE's digits).
//! Generated from that file's tables: a name is one of its `tokenizer_pre` strings, its expressions the type's `regex_exprs`.

/// (names, regular expressions in order, ignore_merges, add_bos)
pub const PRE: &[(&[&str], &[&str], bool, bool)] = &[
    (
        &["bailingmoe", "bailingmoe2", "llada-moe"],
        &[r#"'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"#],
        false,
        false,
    ),
    (&["bloom"], &[r#" ?[^(\s|.,!?…。，、।۔،)]+"#], false, false),
    (
        &["chatglm-bpe", "glm4", "glm5"],
        &[
            r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#,
        ],
        false,
        false,
    ),
    (&["codeshell"], &[r#"\p{N}"#, r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"#], false, false),
    (&["command-r"], &[r#"\p{N}"#, r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"#], false, false),
    (
        &["dbrx"],
        &[
            r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#,
        ],
        false,
        false,
    ),
    (&["deepseek-coder"], &[r#"[\r\n]"#, r#"\s?\p{L}+"#, r#"\s?\p{P}+"#, r#"[一-龥ࠀ-一가-퟿]+"#, r#"\p{N}"#], false, false),
    (
        &["deepseek-llm"],
        &[
            r#"[\r\n]"#,
            r#"\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+"#,
            r#"\s?[!-/:-~！-／：-～‘-‟　-。]+"#,
            r#"\s+$"#,
            r#"[一-龥ࠀ-一가-퟿]+"#,
            r#"\p{N}+"#,
        ],
        false,
        false,
    ),
    (
        &["deepseek-v3"],
        &[
            r#"\p{N}{1,3}"#,
            r#"[一-龥぀-ゟ゠-ヿ]+"#,
            r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"##,
        ],
        false,
        false,
    ),
    (
        &["default"],
        &[r#"[\p{P}\$\+<=>\^~\|]+"#, r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"#, r#"\p{N}+"#, r#"[0-9][0-9][0-9]"#],
        false,
        false,
    ),
    (&["exaone"], &[r#"\p{N}"#, r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"#], false, false),
    (
        &["exaone-moe"],
        &[
            r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?(?:\p{L}\p{M}*(?: \p{L}\p{M}*)*)+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]?|\s*[\r\n]|\s+(?!\S)|\s+"#,
        ],
        false,
        false,
    ),
    (
        &["falcon"],
        &[r#"[\p{P}\$\+<=>\^~\|`]+"#, r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"#, r#"[0-9][0-9][0-9]"#],
        false,
        false,
    ),
    (
        &["flint"],
        &[
            r#"\p{N}{1,3}"#,
            r#"[一-龥぀-ゟ゠-ヿ]+"#,
            r#"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#,
        ],
        false,
        false,
    ),
    (
        &[
            "gpt-2",
            "phi-2",
            "jina-es",
            "jina-de",
            "gigachat",
            "jina-v2-es",
            "jina-v2-de",
            "a.x-4.0",
            "mellum",
            "modern-bert",
            "jina-v1-en",
            "jina-v2-code",
            "roberta-bpe",
            "exaone4",
        ],
        &[r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"#],
        false,
        false,
    ),
    (
        &["gpt-4o", "llama4", "kanana2", "talkie", "inkling2"],
        &[
            r#"[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))*((?=[\p{L}\p{M}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))+((?=[\p{L}\p{M}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"#,
        ],
        false,
        false,
    ),
    (&["gpt3-finnish"], &[r#" ?[^(\s|.,!?…。，、।۔،)]+"#], false, false),
    (&["granite-docling"], &[r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"#], false, false),
    (
        &["grok-2"],
        &[r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#],
        false,
        false,
    ),
    (
        &["hunyuan"],
        &[r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#],
        false,
        false,
    ),
    (
        &["hunyuan-dense"],
        &[
            r#"\p{N}{1,3}"#,
            r#"[一-龥぀-ゟ゠-ヿ]+"#,
            r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"##,
        ],
        false,
        false,
    ),
    (
        &["hy_v4"],
        &[
            r#"\p{N}{1,3}"#,
            r#"[一-龥぀-ゟ゠-ヿ]+"#,
            r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"##,
        ],
        false,
        false,
    ),
    (&["jais"], &[r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"#], false, false),
    (
        &["jais-2"],
        &[
            r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s{512}(?!\S)|\s{256}(?!\S)|\s{128}(?!\S)|\s{64}(?!\S)|\s{32}(?!\S)|\s{16}(?!\S)|\s{8}(?!\S)|\s{4}(?!\S)|\s{1,2}(?!\S)|\s{1}"#,
        ],
        false,
        false,
    ),
    (
        &["joyai-llm"],
        &[
            r#"\p{N}{1,3}"#,
            r#"[一-龥぀-ゟ゠-ヿ]+"#,
            r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"##,
        ],
        false,
        false,
    ),
    (
        &["k2-horizon"],
        &[
            r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?(?:\p{L}|\p{M}|\u200C|\u200D)+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#,
        ],
        false,
        false,
    ),
    (
        &["laguna"],
        &[
            r#"[^\n]+|[\n]+"#,
            r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#,
        ],
        false,
        false,
    ),
    (
        &["llama3", "llama-v3", "llama-bpe", "falcon3", "falcon-h1", "midm-2.0", "jina-v5-nano", "lfm2"],
        &[
            r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#,
        ],
        true,
        true,
    ),
    (&["mellum2"], &[r#"\p{N}"#, r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"#], false, false),
    (&["minerva-7b"], &[r#"\p{N}"#, r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"#], false, false),
    (
        &["minicpm5"],
        &[
            r#"\p{N}{1,3}"#,
            r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}+| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#,
        ],
        true,
        false,
    ),
    (
        &["minimax-m2"],
        &[
            r#"[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))*((?=[\p{L}\p{M}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))+((?=[\p{L}\p{M}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"#,
        ],
        false,
        false,
    ),
    (&["mpt"], &[r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"#], false, false),
    (&["olmo"], &[r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"#], false, false),
    (&["poro-chat"], &[r#" ?[^(\s|.,!?…。，、।۔،)]+"#], false, false),
    (
        &["qwen2", "deepseek-r1-qwen", "kormo", "f2llmv2", "domyn-edge", "megrez"],
        &[r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#],
        false,
        false,
    ),
    (
        &["qwen35"],
        &[
            r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#,
        ],
        false,
        false,
    ),
    (&["refact"], &[r#"\p{N}"#, r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"#], false, false),
    (
        &["seed-coder"],
        &[r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1}| ?[^\s\p{L}\p{N}\r\n]+|\s*[\r\n]+|\s+(?!\S)|\s+"#],
        false,
        false,
    ),
    (
        &["smaug-bpe"],
        &[
            r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#,
        ],
        false,
        false,
    ),
    (&["smollm"], &[r#"\p{N}"#, r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"#], false, false),
    (
        &["solar-open"],
        &[r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#],
        false,
        false,
    ),
    (&["soprano"], &[r#"\p{N}"#, r#"\s+|[\p{L}\p{N}_]+|[^\p{L}\p{N}_\s]+\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)"#], false, false),
    (
        &["spark2_5"],
        &[
            r#"\p{N}{1,3}"#,
            r#"[一-龥぀-ゟ゠-ヿ]+"#,
            r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+|[\r\n]|\s+(?!\S)|\s+"##,
            r#"\p{N}"#,
        ],
        false,
        false,
    ),
    (
        &["stablelm2"],
        &[r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#],
        false,
        false,
    ),
    (&["starcoder"], &[r#"\p{N}"#, r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"#], false, false),
    (&["superbpe"], &[r#"\p{N}+"#, r#"(?=(\d{3})+(?!\d))"#], false, false),
    (
        &["tiny_aya", "cohere2moe"],
        &[
            r#"\d{1,3}(?=(?:\d{3})*\b)"#,
            r#"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"#,
        ],
        false,
        false,
    ),
    (&["trillion"], &[r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"#], false, false),
    (&["ufakzeka"], &[r#"[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#], false, false),
    (&["viking"], &[r#" ?[^(\s|.,!?…。，、।۔،)]+"#, r#"\p{N}"#], false, false),
    (
        &["youtu"],
        &[
            r#"[가-힣ㄱ-ㆎ]+|[！…“”‘’—：；，、-〿︰-﹏]+|[ㄅ-ㄯ]+|[一-龥぀-ゟ゠-ヿ]+"#,
            r#"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"#,
        ],
        true,
        false,
    ),
];
