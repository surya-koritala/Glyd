//! What this build of the engine runs (ROADMAP P16): one table of model families, the levels each runs at, where its
//! weights may go and the context it takes, read by `fit` (src/plan.rs) and by `run`/`serve` (src/cli.rs `tagged`) and
//! their refusals, so `fit` never plans a run that `run` refuses. The machine's side (vendor, compute capability, driver,
//! CPU architecture) is `machine`, on the same constants `machine::gpu_refusal` uses.

use crate::machine::{driver_min, driver_need, ARCHS, MAX_CONTEXT};
use crate::plan::{Kind, Machine, Vendor, GEMMA_CHUNK, GEMMA_DRAFT_BYTES, LOSSY_CHUNK};
use crate::runtime::config::ModelConfig;
use crate::ui::Refusal;
use crate::weights::Layout;

/// A family of models the engine tells apart by its config.json.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// The plain dense kind (every pre-norm dense decoder the description builder's `llama` reading takes: `runtime::describe::LLAMA`), on the executor
    /// (src/dynamic/dense.rs), lossless or at the lossy levels. The variant keeps the name of the first release that ran on it.
    Qwen3,
    /// The Qwen3.5 layout (3.5, 3.6 and 3.8), dense (Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration) or with experts (Qwen3_5MoeForCausalLM,
    /// Qwen3_5MoeForConditionalGeneration): penguin on the dynamic route (one GPU, or its layers cut over several; the experts in the expert cache where the
    /// GPUs do not hold them), the lossy levels on one GPU or, where one does not hold it, its layers cut into stages over the machine's GPUs by the same rule
    /// (src/plan.rs `sizing::lossy_stages`, each stage its layers' packs on its GPU: src/dynamic/qwen35.rs `stage`), both layouts packed and sized by their
    /// description (src/pack.rs `described`), the experts on the GPUs or, where they do not hold them, in the computer's memory with places of them a stage.
    Qwen35,
    NemotronH,
    /// Gemma 4 dense (12B, 31B); the 26B-A4B mixture of experts and the E models are refused.
    Gemma4,
    Kimi,
    Glm,
    /// GLM-5.3 (`glm_moe_dsa`: the latent attention on every layer, its indexer its own or shared, plain residuals; FP8 as released): penguin on the dynamic
    /// route (src/dynamic/glm.rs `load_plain`), on one GPU or its layers cut over several, a stage's experts on its GPU or, where the GPUs do not hold them,
    /// in the expert cache as released (FP8 experts each packed on its own: src/tiering.rs).
    GlmDsa,
    DeepSeekV4,
    /// DeepSeek-V4.1 (`deepseek_v41`: CSA2's compressed attention sharing its parts across layers, lagged mHC, Engram): penguin on the dynamic route, on one
    /// GPU or its layers cut over several (a stage begins where no layer after it reads a compressed part of a layer before: `plan::sizing::stage_starts`;
    /// lagged mHC's weights handed over with the streams), its experts on the GPUs or, where they do not hold them, in the expert cache as released (FP4
    /// experts, as DeepSeek-V4's: src/tiering.rs).
    DeepSeekV41,
    Muse,
    /// Qwen3.8-Flash-Next (the Qwen 4 preview, `qwen4_exp`): penguin on the dynamic route, every expert on the GPUs: one, or its layers cut over several
    /// (src/dynamic/qwen4x.rs `stage`, src/dynamic/split.rs); hummingbird on one GPU from a codes pack of its experts beside its weights (`DIR/pack`, the release's
    /// precise, best and fast packs' format: src/moe_cb.rs), every expert in the computer's memory and places of them on the GPU (`ExpertCache::Codes`).
    /// `qwenserve CKPT PACK --port` serves a pack from anywhere, on one GPU or `--gpus 0,1`.
    FlashNext,
    /// The families whose description is the whole of them: layers of one attention and one feed-forward (a gated MLP or a mixture of experts) that the dense
    /// lowering takes as described (`dynamic::lower::dense`), with no reader, loader or sizing of their own. Domyn Edge, Granite Swash and Mellum 2 are the
    /// first (`runtime::describe::LAYERS`); a family joins by its description builder and its model type in that list. One GPU, at every block level: sized
    /// from the description as every one-GPU row is (src/plan.rs `sizing::level_of`), loaded by the dense loader (`dynamic::dense::load_of`). LFM is one: a
    /// layer's mixer the gated short convolution in the attention's place on most layers, whose state a sequence (each conv layer's channels' last values,
    /// fp32) the description's lowering gives the plan and the scheduler; its convolution's biases, which no release has on, are refused by name. MiMo is
    /// one in its layout (queries and keys of 192 with values of 128 times a factor, the three products one matrix in shards, sliding layers with sinks,
    /// sigmoid-routed experts with a choice bias): as released its fused matrices are FP8 whose block scales start again at every shard and its experts
    /// packed 4-bit codes, two stored forms the dense loader dequantises to bf16 as it reads them, by the scales the description names
    /// (`weights::item_words_as`), and then packs at the plan's level; its draft layers are not read.
    Layers,
}

/// One family's row.
#[derive(Debug)]
pub struct Support {
    pub family: Family,
    pub name: &'static str,
    /// The levels `run` takes this family at, in every build.
    pub levels: &'static [&'static str],
    /// The hummingbird level comes as a pre-packed download or directory only (its experts' codes are made apart).
    pub hummingbird_packed_only: bool,
    /// The context `run` takes where none is asked: Some(n), n tokens (less where the model's own limit is lower);
    /// None: the most that fits, to `LOSSY_CTX`.
    pub default_ctx: Option<usize>,
    /// The most context this path runs, besides the model's own limit.
    pub max_ctx: Option<usize>,
    /// The levels at which the weights may be split over the GPUs of one machine.
    pub split_levels: &'static [&'static str],
    /// The levels at which a mixture of experts may keep its experts in the computer's memory.
    pub experts_in_ram_levels: &'static [&'static str],
    /// The family's own multi-token-prediction head drafts on the executor (src/dynamic/head.rs, Qwen3.8-Flash-Next's): a decode step makes more than one token. The
    /// planner counts that gain; the served paths do not draft with the head yet (its verify pass takes experts held whole on the GPU).
    pub drafts: bool,
    /// How the planner sizes the family's path and what its loader takes (src/plan.rs `sizing`).
    pub route: Route,
    /// How `glyd-engine pack` saves the family (src/pack.rs `plan`, made from the model's description), at the levels it runs (`levels`; another is refused:
    /// `pack_refusal`); None: it saves no level of it.
    pub saves: Option<Saves>,
    /// What the one-GPU route's lossy levels take of it besides its model's description: the rows sized on one GPU (`Sizing::OneGpu`) have one, no other.
    pub lossy: Option<Lossy>,
}

/// What the one-GPU route's lossy levels take of a row besides the model's description, from which src/plan.rs `sizing::level_of` reads the rest (its weights by
/// kind at the level, its sequences' rings and states, its experts, its longest context): every field a row's, no branch on a family there.
#[derive(Clone, Copy, Debug)]
pub struct Lossy {
    /// The engine that loads its plan (src/cli.rs `open_lossy`).
    pub kind: crate::plan::Kind,
    /// The prompt chunk its engine runs (`LoadPlan::chunk`; a drafted round's k + 1 rows fit in it), and the page its contexts are cut to, in tokens.
    pub chunk: usize,
    pub page: usize,
    /// The GPU bytes kept for a draft model of its own, loaded beside it where one is found or comes with the download (src/cli.rs `draft_dir`, src/levels.rs
    /// `draft_repo`); 0: it drafts with none.
    pub draft: u64,
    /// The layout its loader packs a matrix in at a level (the matrix by its save's key, `Pack35::key`), where it is not always the level's (None): src/nemotron.rs
    /// `pack_layout`, the Nemotron-H layout's loader holding its matrices besides the routed experts at six bits at wren.
    pub pack: Option<fn(Layout, &str) -> Layout>,
    /// Its config read as its family's reader reads it before a plan is made: the reader's words where the engine does not run the model (Ok where the
    /// description and its lowering are the whole reading: `described`).
    pub reads: fn(&ModelConfig, Option<&str>) -> crate::ffi::Result<()>,
    /// Whether its loader holds a model's experts lossless at penguin, every one on the GPU (the dense engine's, src/dynamic/dense.rs): the planner then sizes
    /// them at that level and plans none in the computer's memory there. Elsewhere experts are held at the block levels on this route.
    pub lossless_experts: bool,
}

/// A config the description and its lowering read whole (they refuse in words what the engine does not run).
fn described(_: &ModelConfig, _: Option<&str>) -> crate::ffi::Result<()> {
    Ok(())
}

/// The Nemotron-H layout's reader (src/nemotron.rs `ConfigNemotronH::of`: a convolution without its bias, ...).
fn nemotron_reads(c: &ModelConfig, generation: Option<&str>) -> crate::ffi::Result<()> {
    crate::nemotron::ConfigNemotronH::of(c, "config.json", generation).map(drop)
}

/// Gemma 4's reader (src/gemma4.rs `ConfigG4::of`: the heads it runs, the kinds it does not yet).
fn gemma4_reads(c: &ModelConfig, generation: Option<&str>) -> crate::ffi::Result<()> {
    crate::gemma4::ConfigG4::of(c, generation, false).map(drop)
}

/// How `glyd-engine pack` saves a family at a lossy level (and at penguin where it says so): one plan for every family, made from the model's description
/// (src/pack.rs `plan`). The plain form: each layer's matrices packed in the order its weight map names them, every pack before every tensor kept, a
/// tensor the checkpoint holds as F32 cast to bf16, a layer's matrices the library does not pack kept beside their own block's tensors, every matrix in the
/// level's layout but the routed experts' second matrices where their activation is relu squared (the row layout, src/moe_rows.rs). Where a family's
/// published saves differ from that, its row says how: they are read back by those bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Saves {
    /// penguin too: a glyd-v7 directory of the lossless layout, which the executor reads as saved (else the lossy levels alone).
    pub penguin: bool,
    /// Each layer's packs then its tensors kept, layer by layer (glyd/gemma-4-*).
    pub by_layer: bool,
    /// A tensor the checkpoint holds as F32 saved as F32, not cast to bf16.
    pub f32_kept: bool,
    /// A layer's matrices the library does not pack kept after the tensors of all its blocks, not each after its own block's.
    pub parts_last: bool,
    /// The layout of every matrix but the routed experts' at wren and hummingbird (Nemotron 3: six-bit codes, its routed experts alone at the level's).
    pub rest_at_wren: Option<crate::weights::Layout>,
}

/// The plain form (`Saves`).
const SAVED: Saves = Saves { penguin: false, by_layer: false, f32_kept: false, parts_last: false, rest_at_wren: None };

/// What a family's loader takes, as the planner reads it: every branch of `plan::sizing` that differs between families reads a field here, so a
/// family of the same layer kinds is a row, not an edit there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Route {
    pub sizing: Sizing,
    /// The loader holds bf16 matrices in penguin's 12-bit layout where that fits and reads faster (else tiered alone).
    pub bf16_twelve: bool,
    pub across: Across,
    pub expert_cache: ExpertCache,
    pub ram_linears: RamLinears,
}

/// Which sizing `plan::sizing::size` gives a family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sizing {
    /// No path of `run` loads it yet: the planner's estimate stands.
    Estimate,
    /// The dense kind on the executor, on one GPU at every level (`dense_fit`).
    Dense,
    /// One GPU at the lossy levels (`lossy_plan`), and the dynamic route at a level its layers split at (`Support::split_levels`).
    OneGpu,
    /// The dynamic route at every level, over one GPU or several (`size_dynamic`).
    Dynamic,
}

/// Whether a family's layers run across the GPUs of one machine, each GPU a slice of every matrix (src/dynamic/tp.rs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Across {
    No,
    /// Where a development build asks for it (GLYD_TP=1).
    Asked,
    /// And where nothing is asked, on HBM GPUs that reach each other directly.
    Default,
}

/// The cache of experts a family's loader keeps on the GPUs where they do not hold every expert, by what sizes a place.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertCache {
    None,
    /// An expert as the loader packs it, in the layouts of these tags (glyd_gpu::pack's), each place the largest of its bytes in them (the layout its scales
    /// allow is the load's), from the Moe kind's shape in the model's description and the experts as it records them released (src/plan.rs `packed_experts`).
    Packed(&'static [u32]),
    /// An expert's bytes as released, each the same (`release_experts`: src/plan.rs).
    Release,
    /// A codes pack of the experts beside the release's weights at the hummingbird level (src/cb_pack.rs, src/moe_cb.rs): places of the pack's largest expert
    /// a layer, every expert the pack's bytes in the computer's memory, its files mapped (`codes_plan`).
    Codes,
}

/// The linears of a layer a loader can keep in the computer's memory (src/dense_ram.rs), copied to the GPU before the layer runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RamLinears {
    None,
    /// Every matrix the executor's loader packs for a layer's own products (src/dynamic/mod.rs `Placing::ram` keeps each of them there): the attention's
    /// projections (a delta attention's or a latent one's: not a latent attention's kv_b nor its indexer's, small tensors it holds as saved), a dense MLP's
    /// three, and the experts' latent projections where they are the layer's products (a block of experts keeps its own matrices on the GPU).
    Products,
}

/// A family on one GPU with nothing kept off it: what the other rows differ from.
const ONE_GPU: Route = Route { sizing: Sizing::OneGpu, bf16_twelve: false, across: Across::No, expert_cache: ExpertCache::None, ram_linears: RamLinears::None };

/// The most context the lossy paths take where none is asked (one constant: the planner's, src/plan.rs).
pub use crate::plan::LOSSY_CTX;
/// The context the plain dense kind's path and the dynamic engine take where none is asked (src/plan.rs `sizing::CTX_WANT`, `sizing::size_dynamic`).
pub const DEFAULT_CTX: usize = 8192;

/// Lossless, or a block level of every weight.
const PENGUIN_LOSSY: &[&str] = &["penguin", "kestrel", "linnet", "wren"];
const NONE: &[&str] = &[];
const PENGUIN: &[&str] = &["penguin"];
/// Lossless, or the routed experts at hummingbird's bits (a directory `glyd-engine pack --layout hummingbird --usage` makes: src/hb_pack.rs).
const PENGUIN_HB: &[&str] = &["penguin", "hummingbird"];
const HB: &[&str] = &["hummingbird"];
/// A loader's own pack of an expert released in four-bit floats with their scales (MXFP4): the FP4 layout with its scales packed where they allow (FP4R),
/// else as released (src/dynamic/experts.rs `pack_into_host`).
const FP4_PACKS: &[u32] = &[glyd_gpu::pack::FP4R_TAG, glyd_gpu::pack::FP4_TAG];

pub static TABLE: [Support; 12] = [
    Support {
        family: Family::Qwen3,
        name: "plain dense",
        levels: PENGUIN_LOSSY,
        hummingbird_packed_only: false,
        default_ctx: Some(DEFAULT_CTX),
        max_ctx: Some(MAX_CONTEXT),
        split_levels: NONE,
        experts_in_ram_levels: NONE,
        drafts: false,
        route: Route { sizing: Sizing::Dense, ..ONE_GPU },
        saves: Some(Saves { penguin: true, ..SAVED }),
        lossy: None,
    },
    Support {
        family: Family::Qwen35,
        name: "Qwen3.5",
        levels: PENGUIN_LOSSY,
        hummingbird_packed_only: false,
        default_ctx: None,
        max_ctx: None,
        // (its loader loads a stage of its layers at every level: src/dynamic/qwen35.rs `stage`)
        split_levels: PENGUIN_LOSSY,
        // (the layout with experts: at penguin, where its GPUs do not hold them)
        experts_in_ram_levels: PENGUIN,
        drafts: false,
        route: Route { bf16_twelve: true, expert_cache: ExpertCache::Release, ..ONE_GPU },
        saves: Some(SAVED),
        lossy: Some(Lossy { kind: Kind::Qwen35, chunk: LOSSY_CHUNK, page: 1, draft: 0, pack: None, reads: described, lossless_experts: false }),
    },
    Support {
        family: Family::NemotronH,
        name: "Nemotron-H",
        levels: &["penguin", "kestrel", "linnet", "wren", "hummingbird"],
        hummingbird_packed_only: true,
        default_ctx: None,
        max_ctx: None,
        split_levels: PENGUIN,
        experts_in_ram_levels: &["kestrel", "linnet", "wren", "hummingbird"],
        drafts: false,
        route: ONE_GPU,
        saves: Some(Saves { rest_at_wren: Some(crate::weights::Layout::Int6), ..SAVED }),
        lossy: Some(Lossy {
            kind: Kind::NemotronH,
            chunk: LOSSY_CHUNK,
            page: 1,
            draft: 0,
            pack: Some(crate::nemotron::pack_layout),
            reads: nemotron_reads,
            lossless_experts: false,
        }),
    },
    Support {
        family: Family::Gemma4,
        name: "Gemma 4",
        levels: PENGUIN_LOSSY,
        hummingbird_packed_only: false,
        default_ctx: None,
        max_ctx: None,
        split_levels: NONE,
        experts_in_ram_levels: NONE,
        drafts: false,
        route: ONE_GPU,
        saves: Some(Saves { penguin: true, by_layer: true, ..SAVED }),
        lossy: Some(Lossy {
            kind: Kind::Gemma4,
            chunk: GEMMA_CHUNK,
            page: 16,
            draft: GEMMA_DRAFT_BYTES,
            pack: None,
            reads: gemma4_reads,
            lossless_experts: false,
        }),
    },
    Support {
        family: Family::Kimi,
        name: "Kimi-K3",
        levels: PENGUIN_HB,
        hummingbird_packed_only: true,
        default_ctx: Some(DEFAULT_CTX),
        max_ctx: None,
        split_levels: PENGUIN_HB,
        experts_in_ram_levels: PENGUIN_HB,
        drafts: false,
        route: Route {
            sizing: Sizing::Dynamic,
            bf16_twelve: true,
            across: Across::Default,
            expert_cache: ExpertCache::Packed(FP4_PACKS),
            ram_linears: RamLinears::Products,
        },
        saves: None,
        lossy: None,
    },
    Support {
        family: Family::Glm,
        name: "GLM-5.3-Flash",
        levels: PENGUIN_HB,
        hummingbird_packed_only: true,
        default_ctx: Some(DEFAULT_CTX),
        max_ctx: None,
        split_levels: PENGUIN_HB,
        experts_in_ram_levels: PENGUIN_HB,
        drafts: false,
        route: Route { sizing: Sizing::Dynamic, across: Across::Asked, expert_cache: ExpertCache::Release, ..ONE_GPU },
        saves: Some(Saves { f32_kept: true, parts_last: true, ..SAVED }),
        lossy: None,
    },
    Support {
        family: Family::GlmDsa,
        name: "GLM-5.3",
        levels: PENGUIN,
        hummingbird_packed_only: false,
        default_ctx: Some(DEFAULT_CTX),
        max_ctx: None,
        split_levels: PENGUIN,
        experts_in_ram_levels: PENGUIN,
        drafts: false,
        route: Route { sizing: Sizing::Dynamic, expert_cache: ExpertCache::Release, ..ONE_GPU },
        saves: None,
        lossy: None,
    },
    Support {
        family: Family::DeepSeekV4,
        name: "DeepSeek-V4",
        levels: PENGUIN_HB,
        hummingbird_packed_only: true,
        default_ctx: Some(DEFAULT_CTX),
        max_ctx: None,
        split_levels: PENGUIN_HB,
        experts_in_ram_levels: PENGUIN_HB,
        drafts: false,
        route: Route { sizing: Sizing::Dynamic, across: Across::Default, expert_cache: ExpertCache::Release, ..ONE_GPU },
        saves: None,
        lossy: None,
    },
    Support {
        family: Family::DeepSeekV41,
        name: "DeepSeek-V4.1",
        levels: PENGUIN,
        hummingbird_packed_only: false,
        default_ctx: Some(DEFAULT_CTX),
        max_ctx: None,
        split_levels: PENGUIN,
        experts_in_ram_levels: PENGUIN,
        drafts: false,
        route: Route { sizing: Sizing::Dynamic, expert_cache: ExpertCache::Release, ..ONE_GPU },
        saves: None,
        lossy: None,
    },
    Support {
        family: Family::Muse,
        name: "Muse Spark",
        levels: PENGUIN_HB,
        hummingbird_packed_only: true,
        default_ctx: Some(DEFAULT_CTX),
        max_ctx: None,
        split_levels: PENGUIN_HB,
        experts_in_ram_levels: PENGUIN_HB,
        drafts: false,
        route: Route { sizing: Sizing::Dynamic, across: Across::Asked, expert_cache: ExpertCache::Release, ram_linears: RamLinears::Products, ..ONE_GPU },
        saves: None,
        lossy: None,
    },
    Support {
        family: Family::FlashNext,
        name: "Qwen3.8-Flash-Next",
        levels: PENGUIN_HB,
        hummingbird_packed_only: true,
        default_ctx: Some(DEFAULT_CTX),
        max_ctx: None,
        split_levels: PENGUIN,
        experts_in_ram_levels: HB,
        drafts: true,
        route: Route { sizing: Sizing::Dynamic, expert_cache: ExpertCache::Codes, ..ONE_GPU },
        saves: None,
        lossy: None,
    },
    Support {
        family: Family::Layers,
        name: "described layers",
        levels: PENGUIN_LOSSY,
        hummingbird_packed_only: false,
        default_ctx: None,
        max_ctx: None,
        split_levels: NONE,
        experts_in_ram_levels: NONE,
        drafts: false,
        route: ONE_GPU,
        saves: Some(SAVED),
        // (nothing of its own: the description and the dense lowering are the whole reading, the dense loader the engine)
        lossy: Some(Lossy { kind: Kind::Qwen3, chunk: LOSSY_CHUNK, page: 1, draft: 0, pack: None, reads: described, lossless_experts: true }),
    },
];

/// The row of `family`.
pub fn row(family: Family) -> &'static Support {
    TABLE.iter().find(|s| s.family == family).expect("every family has a row")
}

/// The row by the name the table gives it (`Support::name`: "Kimi-K3", "Gemma 4").
pub fn row_named(name: &str) -> Option<&'static Support> {
    TABLE.iter().find(|s| s.name == name)
}

/// The context `row`'s path takes where none is asked (the model's own limit where lower): the table's, else the lossy paths' most.
pub fn default_context(row: &Support, model_max: usize) -> usize {
    row.default_ctx.unwrap_or(LOSSY_CTX).min(model_max)
}

/// The refusal for a context asked for beyond what `row`'s path runs or the model has (`model_max`): the one `fit`, `run`, `serve` and `preflight` all give.
pub fn context_check(name: &str, row: &Support, model_max: usize, asked: Option<usize>) -> Result<(), Refusal> {
    let most = model_max.min(row.max_ctx.unwrap_or(usize::MAX));
    match asked {
        Some(c) if c > most => Err(Refusal::unsupported(
            format!("{name} takes at most {most} tokens of context here, and {c} were asked for"),
            format!("Ask for less: --context {most}"),
        )),
        _ => Ok(()),
    }
}

impl Support {
    /// Whether `run` takes this family at `level` (`model`: the name asked; every build runs a family's levels since v0.34.0, wren
    /// included, which a release build ran on quantization-aware checkpoints alone before).
    pub fn runs_at(&self, level: &str, _model: &str) -> bool {
        self.allows(level)
    }

    /// Whether `level` is one of this family's.
    pub fn allows(&self, level: &str) -> bool {
        self.levels.contains(&level)
    }

    /// The refusal `run` gives this family at `level` where it does not run it there (`check`'s words); `model`: the name asked, `name`: its short name.
    pub fn refusal(&self, model: &str, name: &str, level: &str) -> Option<Refusal> {
        if self.runs_at(level, model) {
            return None;
        }
        let now = self.levels_now(model);
        if now.is_empty() {
            return Some(Refusal::unsupported(
                format!("{name} ({}) does not run with glyd run yet", self.name),
                "Run it with Ibis: ibis/README.md (glyd run takes it with ROADMAP P8)",
            ));
        }
        let fix = match now[0] {
            "penguin" => format!("Run it lossless: glyd run {model}"),
            l => format!("Run it at the {l} level: glyd run {model}:{l}"),
        };
        let list = match now.as_slice() {
            [one] => format!("the {one} level"),
            [rest @ .., last] => format!("the {} and {last} levels", rest.join(", ")),
            [] => unreachable!(),
        };
        Some(Refusal::unsupported(format!("{name} ({}) runs at {list} in this engine, not {level}", self.name), fix))
    }

    /// The refusal of `glyd-engine pack` in `layout` where its level is not one this family runs (`refusal`, `run`'s words): a save no level of it reads. A
    /// layout's level is its name's (`levels::name_of`); the three-bit layout, which is no level, packs where the family runs a block level (its codes are
    /// the block levels' at three bits).
    pub fn pack_refusal(&self, layout: Layout, model: &str, name: &str) -> Option<Refusal> {
        if layout == Layout::Int3 && ["kestrel", "linnet", "wren"].iter().any(|l| self.allows(l)) {
            return None;
        }
        self.refusal(model, name, crate::levels::name_of(layout))
    }

    /// Whether its hummingbird level is the one-GPU route's (src/moe_hb.rs: its experts on a latent the kernels take, every other matrix six-bit, a pre-packed
    /// directory: src/hbpack.rs), not the dynamic route's (src/hb_pack.rs, src/moe_cb.rs).
    pub fn hummingbird_one_gpu(&self) -> bool {
        self.lossy.is_some() && self.allows("hummingbird")
    }

    /// The levels this build runs this family at, for `model`.
    pub fn levels_now(&self, model: &str) -> Vec<&'static str> {
        self.levels.iter().copied().filter(|l| self.runs_at(l, model)).collect()
    }
}

/// A config.json's text parsed once, or the refusal `run` gives where a key is of another type than the engine reads it as. `name`: the model's short name, for the words.
pub fn parsed(cfg_text: &str, name: &str) -> Result<ModelConfig, Refusal> {
    ModelConfig::parse(cfg_text).map_err(|e| Refusal::unsupported(format!("{name}: {}", e.msg), ""))
}

/// The family of a config.json, or the refusal `run` gives for it. `name`: the model's short name, for the words.
pub fn family(cfg_text: &str, name: &str) -> Result<Family, Refusal> {
    family_in(&parsed(cfg_text, name)?, name)
}

/// `family` of a config.json parsed once: the family by its names, `runtime::config`'s one rule.
pub fn family_in(c: &ModelConfig, name: &str) -> Result<Family, Refusal> {
    if c.family() == Some(Family::Gemma4) {
        // (the 26B-A4B mixture of experts and the E models are refused whatever the level)
        if let Err(e) = crate::gemma4::ConfigG4::of(c, None, false) {
            if e.msg.contains("not in this engine yet") {
                return Err(Refusal::unsupported(format!("{name}: {}", e.msg), "Run Gemma 4 12B or 31B: glyd run google/gemma-4-12B-it"));
            }
        }
    }
    let arch = c.arch.as_str();
    match c.family() {
        Some(f) => Ok(f),
        None if arch.is_empty() => Err(Refusal::unsupported(
            format!("{name}: its config.json does not say what kind of model it is"),
            "Run a model whose config.json names its architecture, as google/gemma-4-12B-it's does",
        )),
        // (what is missing is said, not a list of what runs: the description of this architecture's layers. The Python tool hands the model to vLLM on this
        // refusal's status and prints its first line as it is)
        _ => Err(Refusal::unsupported(
            format!("{name} is a {arch} model, which this engine has no description of yet"),
            crate::machine::instead(
                "Other models run with the vLLM-based glyd, which is installed here",
                &format!("Pick a model of a family this engine runs (glyd run Qwen/Qwen3.8-27B), or {}", crate::machine::FULL_TOOL),
            ),
        )),
    }
}

/// The row `run` uses for `cfg_text` at `level`, or its refusal. `model`: the name asked, without its level (`base` in
/// the fix); `name`: its short name.
pub fn check(cfg_text: &str, model: &str, name: &str, level: &str) -> Result<&'static Support, Refusal> {
    check_in(&parsed(cfg_text, name)?, model, name, level)
}

/// `check` of a config.json parsed once.
pub fn check_in(c: &ModelConfig, model: &str, name: &str, level: &str) -> Result<&'static Support, Refusal> {
    let s = row(family_in(c, name)?);
    s.refusal(model, name, level).map_or(Ok(s), Err)
}

/// The refusal `run` gives on `m`'s GPU with the most free memory (the one a run uses), or None where it runs there:
/// the same conditions as `machine::gpu_refusal` for a run that was asked for, and the vendors and the CPU-only machine
/// this build has no backend for.
pub fn machine(m: &Machine) -> Option<Refusal> {
    let Some(d) = m.devices.iter().max_by_key(|d| (d.free, std::cmp::Reverse(d.name.len()))) else {
        return Some(Refusal::unsupported(
            "no GPU was found: the engine's CPU backend is not built yet",
            "See ROADMAP V4; on this computer glyd compresses files: glyd --help",
        ));
    };
    match d.vendor {
        Vendor::Amd => {
            return Some(Refusal::unsupported(
                format!("{}: AMD GPUs run in development builds only (GLYD_HIP=1)", d.name),
                "See ROADMAP V1 for AMD in releases; CONTRIBUTING.md builds it",
            ))
        }
        Vendor::Apple => return Some(Refusal::unsupported(format!("{}: Apple GPUs are not supported yet", d.name), "See ROADMAP V3 (Metal through MLX)")),
        Vendor::Nvidia => {}
    }
    if d.cc.0 < 8 {
        return Some(Refusal::unsupported(
            format!("{} (compute capability {}.{}) is older than the engine runs on", d.name, d.cc.0, d.cc.1),
            "glyd run needs the Ampere generation or newer: RTX 30 and 40 series, A10, A100, L4, L40S, H100 (Turing: ROADMAP V5)",
        ));
    }
    if !ARCHS.contains(&m.arch.as_str()) {
        return Some(Refusal::unsupported(
            format!("the engine has not run on a {} computer yet", m.arch),
            format!("It runs on {} computers", ARCHS.join(" and ")),
        ));
    }
    if d.driver < driver_min(d.cc) {
        return Some(Refusal::unsupported(
            format!("the driver supports CUDA {}.{}: the engine's kernels need {}", d.driver / 1000, d.driver % 1000 / 10, driver_need(d.cc)),
            format!("Update the NVIDIA driver to {} or newer (https://www.nvidia.com/drivers)", crate::machine::driver_series(driver_min(d.cc))),
        ));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use glyd_gpu::json::{self, Value};

    fn fixture(name: &str) -> String {
        std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)).unwrap()
    }

    /// A plain dense decoder's config (the kind's: `model_type` is what the description builder reads, no release is meant by it).
    const DENSE: &str = r#"{"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 36, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128}"#;

    #[test]
    fn every_family_is_told_apart() {
        assert_eq!(family(DENSE, "d").unwrap(), Family::Qwen3);
        assert_eq!(
            row(Family::Qwen3).name,
            crate::dynamic::dense::FAMILY,
            "one name for the dense kind: the table's row, the executor's description, a plan's family"
        );
        // (the dense kind by any model type the description builder's `llama` reading takes, under any architecture's name)
        assert_eq!(family(r#"{"architectures": ["AnyNameForCausalLM"], "model_type": "mistral"}"#, "d").unwrap(), Family::Qwen3);
        let q35: Value = json::parse(&fixture("qwen35_configs.json")).unwrap();
        let first = q35.as_object().unwrap()[0].1.clone();
        assert_eq!(family(&json::to_compact(&first), "q").unwrap(), Family::Qwen35);
        assert_eq!(family(&fixture("nemotron_h_config.json"), "n").unwrap(), Family::NemotronH);
        assert_eq!(family(&fixture("gemma4_12b_config.json"), "g").unwrap(), Family::Gemma4);
        assert_eq!(family(&fixture("glm5_next_config.json"), "g").unwrap(), Family::Glm);
        assert_eq!(family(&fixture("qwen4x_config.json"), "x").unwrap(), Family::FlashNext);
        assert_eq!(family(r#"{"model_type": "deepseek_v4", "architectures": ["DeepseekV4ForCausalLM"]}"#, "d").unwrap(), Family::DeepSeekV4);
        // (a kind with no description: what is missing is said, and no family is listed)
        let r = family(r#"{"architectures": ["OtherForCausalLM"], "model_type": "other"}"#, "other").unwrap_err();
        assert!(r.what == "other is a OtherForCausalLM model, which this engine has no description of yet" && r.code == crate::ui::CANNOT, "{r:?}");
        assert!(!r.fix.contains("Qwen3 model") && !r.fix.contains("Qwen3-"), "{r:?}");
        let r = family("{}", "nameless").unwrap_err();
        assert!(r.what.contains("does not say what kind of model it is") && !r.fix.contains("Qwen3"), "{r:?}");
    }

    #[test]
    fn levels_follow_the_table() {
        // (the dense kind's one line: the row's levels, and a refusal's words, on a made-up name)
        let r = check(DENSE, "org/dense", "dense", "hummingbird").unwrap_err();
        assert_eq!(
            (r.what.as_str(), r.fix.as_str()),
            ("dense (plain dense) runs at the penguin, kestrel, linnet and wren levels in this engine, not hummingbird", "Run it lossless: glyd run org/dense")
        );
        assert!(check(DENSE, "org/dense", "dense", "penguin").is_ok() && check(DENSE, "org/dense", "dense", "linnet").is_ok());
        let n = fixture("nemotron_h_config.json");
        for l in ["penguin", "kestrel", "linnet", "hummingbird"] {
            assert!(check(&n, "nvidia/N", "N", l).is_ok(), "{l}");
        }
        assert!(check(&n, "nvidia/N", "N", "wren").is_ok(), "wren runs in every build");
        // (the Qwen3.5 layout: penguin and every lossy level; never hummingbird, which is for experts)
        let q35: Value = json::parse(&fixture("qwen35_configs.json")).unwrap();
        let q38 = json::to_compact(q35.get("Qwen/Qwen3.8-27B").unwrap());
        for l in ["penguin", "kestrel", "linnet", "wren"] {
            assert!(check(&q38, "Qwen/Qwen3.8-27B", "Qwen3.8-27B", l).is_ok(), "{l}");
        }
        assert!(check(&q38, "Qwen/Qwen3.8-27B", "Qwen3.8-27B", "hummingbird").is_err());
        assert_eq!(
            row(Family::Qwen35).split_levels,
            ["penguin", "kestrel", "linnet", "wren"],
            "its layers cut over the GPUs at penguin and at every lossy level, as the dynamic route cuts them"
        );
        // (Qwen3.8-Flash-Next: penguin on the dynamic route, its layers cut over the GPUs as the route cuts them; hummingbird from the codes pack beside its weights,
        // on one GPU)
        let x = fixture("qwen4x_config.json");
        for l in ["penguin", "hummingbird"] {
            assert!(check(&x, "Qwen/X", "X", l).is_ok(), "{l}");
        }
        let r = check(&x, "Qwen/X", "X", "kestrel").unwrap_err();
        assert_eq!(r.what, "X (Qwen3.8-Flash-Next) runs at the penguin and hummingbird levels in this engine, not kestrel", "{r:?}");
        let fx = row(Family::FlashNext);
        assert!(fx.split_levels == ["penguin"] && fx.route.sizing == Sizing::Dynamic && fx.route.expert_cache == ExpertCache::Codes);
        assert!(fx.experts_in_ram_levels == ["hummingbird"] && fx.hummingbird_packed_only);
        assert_eq!(crate::dynamic::split::family(&x), Some("Qwen3.8-Flash-Next"), "opened by the dynamic route from its release's directory");
        let g = fixture("gemma4_12b_config.json");
        assert!(check(&g, "google/gemma-4-12B-it", "gemma-4-12B-it", "wren").is_ok(), "a QAT checkpoint's wren runs in every build");
        assert!(check(&g, "google/gemma-4-12B-it", "gemma-4-12B-it", "hummingbird").is_err());
    }

    /// The Qwen3.5 layout with experts (`Qwen3_5MoeForCausalLM`, `Qwen3_5MoeForConditionalGeneration`) is the Qwen3.5 family's, not a row of its own: the
    /// dynamic route opens it (its layers cut over the GPUs at penguin) by the loader of the gated delta rule, which reads its layers by its description; the
    /// tree's random case and the two releases of the fixtures alike, at every level of the row (its lossy levels sized by its description: src/plan.rs).
    #[test]
    fn the_qwen35_layout_with_experts_is_the_familys() {
        let q35: Value = json::parse(&fixture("qwen35_configs.json")).unwrap();
        let cases = [
            ("the random case", fixture("qwen35moe_tiny_config.json")),
            ("Qwen3.5-35B-A3B", json::to_compact(q35.get("Qwen/Qwen3.5-35B-A3B").unwrap())),
            ("Qwen3.8-2.4T-A95B", json::to_compact(q35.get("Qwen/Qwen3.8-2.4T-A95B").unwrap())),
        ];
        for (what, text) in &cases {
            assert_eq!(family(text, "q").unwrap(), Family::Qwen35, "{what}");
            for level in PENGUIN_LOSSY {
                assert!(check(text, "Qwen/q", "q", level).is_ok(), "{what} {level}");
            }
            assert_eq!(crate::dynamic::split::family(text), Some("Qwen3.5"), "{what}: opened by the dynamic route from its release's directory");
            let d = crate::runtime::describe::describe(&serde_json::from_str(text).unwrap()).unwrap();
            assert!(d.desc.blocks.iter().any(|b| matches!(b.body, crate::runtime::kinds::Body::Ffn(crate::runtime::kinds::Ffn::Moe(_)))), "{what}");
            assert!(crate::dynamic::split::loader(&d).is_ok(), "{what}");
        }
    }

    /// GLM-5.3 (`glm_moe_dsa`) is a family of its own: penguin on the dynamic route, its layers cut over the GPUs there, opened by the loader of the latent
    /// attention with plain residuals (src/dynamic/glm.rs `stage_plain`).
    #[test]
    fn glm53_runs_on_the_dynamic_route() {
        let g = fixture("glm_moe_dsa_config.json");
        assert_eq!(family(&g, "g").unwrap(), Family::GlmDsa);
        assert!(check(&g, "zai-org/GLM-5.3", "GLM-5.3", "penguin").is_ok());
        let r = check(&g, "zai-org/GLM-5.3", "GLM-5.3", "kestrel").unwrap_err();
        assert_eq!(r.what, "GLM-5.3 (GLM-5.3) runs at the penguin level in this engine, not kestrel", "{r:?}");
        assert_eq!((row(Family::GlmDsa).split_levels, row(Family::GlmDsa).route.sizing), (PENGUIN, Sizing::Dynamic));
        assert_eq!(crate::dynamic::split::family(&g), Some("GLM-5.3"), "opened by the dynamic route from its release's directory");
        let d = crate::runtime::describe::describe(&serde_json::from_str(&g).unwrap()).unwrap();
        assert!(crate::dynamic::split::loader(&d).is_ok());
    }

    /// DeepSeek-V4.1 (`deepseek_v41`, its release's config) is a family of its own: penguin on the dynamic route, its layers cut over GPUs, its experts there or
    /// in the expert cache as released, opened by the compressed attention's loader, its layers and end tokens by its description (a nested text config).
    #[test]
    fn deepseek_v41_runs_on_the_dynamic_route() {
        let v = fixture("deepseek_v41_config.json");
        assert_eq!(family(&v, "v").unwrap(), Family::DeepSeekV41);
        assert!(check(&v, "deepseek-ai/DeepSeek-V4.1-Flash-1017", "DeepSeek-V4.1-Flash-1017", "penguin").is_ok());
        let r = check(&v, "deepseek-ai/DeepSeek-V4.1-Flash-1017", "DeepSeek-V4.1-Flash-1017", "hummingbird").unwrap_err();
        assert_eq!(r.what, "DeepSeek-V4.1-Flash-1017 (DeepSeek-V4.1) runs at the penguin level in this engine, not hummingbird", "{r:?}");
        let r = row(Family::DeepSeekV41);
        assert_eq!((r.split_levels, r.experts_in_ram_levels, r.route.expert_cache, r.route.sizing), (PENGUIN, PENGUIN, ExpertCache::Release, Sizing::Dynamic));
        assert_eq!(crate::dynamic::split::family(&v), Some("DeepSeek-V4.1"), "opened by the dynamic route from its release's directory");
        let d = crate::runtime::describe::describe(&serde_json::from_str(&v).unwrap()).unwrap();
        assert!(crate::dynamic::split::loader(&d).is_ok());
    }

    /// What the one-GPU route's lossy levels read of a row besides the description is the row's (`Lossy`): the rows sized on one GPU have it and no other row;
    /// Gemma 4's engine runs prompts in chunks of 512 tokens, cuts its contexts to pages of 16 and drafts with a draft model of 640 MiB; the Nemotron-H layout's
    /// loader packs some matrices in layouts of its own, the others' every matrix at the level; and only Nemotron-H's hummingbird level is the one-GPU route's.
    #[test]
    fn a_row_says_what_the_lossy_levels_take_beside_the_description() {
        for r in &TABLE {
            assert_eq!(r.lossy.is_some(), r.route.sizing == Sizing::OneGpu, "{}", r.name);
        }
        let l = |f: Family| row(f).lossy.unwrap();
        let (g, n, q) = (l(Family::Gemma4), l(Family::NemotronH), l(Family::Qwen35));
        assert_eq!((g.kind, g.chunk, g.page, g.draft), (Kind::Gemma4, 512, 16, 640 << 20));
        assert_eq!((n.kind, n.chunk, n.page, n.draft), (Kind::NemotronH, 2048, 1, 0));
        assert_eq!((q.kind, q.chunk, q.page, q.draft), (Kind::Qwen35, 2048, 1, 0));
        assert!(n.pack.is_some() && g.pack.is_none() && q.pack.is_none());
        let pack = n.pack.unwrap();
        assert_eq!(pack(Layout::Int4, "backbone.layers.1.mixer.in_proj"), Layout::Int6, "its matrices besides the routed experts six-bit at wren");
        assert_eq!(pack(Layout::Int4, "backbone.layers.1.mixer.experts.0.up_proj"), Layout::Int4);
        assert_eq!(TABLE.iter().filter(|r| r.hummingbird_one_gpu()).map(|r| r.family).collect::<Vec<_>>(), [Family::NemotronH]);
        // (a family's reader refuses in its words what the engine does not run: Nemotron-H's convolution without its bias)
        let nh = fixture("nemotron_h_config.json").replace("\"use_conv_bias\": true", "\"use_conv_bias\": false");
        assert!((n.reads)(&parsed(&nh, "n").unwrap(), None).unwrap_err().msg.contains("a convolution without its bias"));
        assert!((g.reads)(&parsed(&fixture("gemma4_12b_config.json"), "g").unwrap(), None).is_ok());
    }

    #[test]
    fn the_machine_side_matches_gpu_refusal() {
        let ok = Machine::preset("rtx4080s").unwrap();
        assert!(machine(&ok).is_none());
        for (name, word) in [("rx7900xtx", "AMD"), ("m4max-64g", "Apple")] {
            let r = machine(&Machine::preset(name).unwrap()).unwrap_or_else(|| panic!("{name}"));
            assert!(r.what.contains(word) || r.fix.contains(word), "{name}: {r:?}");
            assert_eq!(r.code, crate::ui::CANNOT);
        }
        // (a DGX Spark is an aarch64 computer, which the engine runs on as on an x86_64 one; an architecture it is not built for is refused)
        assert!(machine(&Machine::preset("dgx-spark").unwrap()).is_none());
        let mut other = Machine::preset("rtx4080s").unwrap();
        other.arch = "riscv64".into();
        let r = machine(&other).unwrap();
        assert!(r.what.contains("riscv64") && r.fix.contains("x86_64 and aarch64"), "{r:?}");
        // (a driver below this build's floor, read from the build: CUDA 12.7 where it carries CUDA 13's kernels alone, 12.1 where it carries CUDA 12's
        // too, which run from 12.2; the refusal names the driver's version and the floor's)
        let mut old = Machine::preset("rtx4080s").unwrap();
        let (cc, floor) = (old.devices[0].cc, crate::machine::driver_min(old.devices[0].cc));
        old.devices[0].driver = if floor > 12070 { 12070 } else { floor - 10 };
        let r = machine(&old).unwrap();
        let said = format!("CUDA {}.{}", old.devices[0].driver / 1000, old.devices[0].driver % 1000 / 10);
        assert!(r.what.contains(&said) && r.what.contains(&crate::machine::driver_need(cc)), "{r:?}");
        // (and a driver at the floor runs: CUDA 12.7 does where the build carries CUDA 12's kernels)
        old.devices[0].driver = floor;
        assert!(machine(&old).is_none());
        if crate::machine::KERNELS12 {
            old.devices[0].driver = 12070;
            assert!(machine(&old).is_none());
        }
    }
}
