//! glyd-engine: Glyd's own inference engine (the families of src/support.rs on CUDA).
//!
//!   glyd-engine run MODEL[:LEVEL] [--prompt TEXT] [--no-think] [--context N] [--users N] [--port N] [--host H] [--temp T] [--layout auto|mma|mma12] [--expert-cache GIB] [--host-memory GIB]
//!                                                             what `glyd run` starts: the model made ready on the first run (downloaded, packed for this GPU,
//!                                                             its KV tables made), then a chat in the terminal and at one URL, or one answer with --prompt
//!                                                             LEVEL: penguin (lossless, the default), kestrel (about 6.5 bits per weight; --lossy is the same), swift (about 5.5), wren (about 4.5)
//!                                                             or hummingbird (a mixture of experts' routed experts at about 3)
//!                                                             --expert-cache GIB: Nemotron 3 Super with its experts in the computer's memory and a cache of them in GIB GiB of the GPU's (without it, the cache is what the GPU has left when it cannot hold the model)
//!                                                             --cpu-threads N: the threads that compute the experts that are not on the GPU (the computer's cores less two)
//!                                                             --host-memory GIB|auto: the computer's memory the experts may take (auto: what it has free, less 6 GiB for the system); the Hummingbird level keeps the rest of its
//!                                                             experts in the packed directory, on the SSD, and reads one when the router asks for it
//!   glyd-engine serve MODEL [--port N] [--host H] [--context N] [--users N] [--host-memory GIB]   the OpenAI-compatible API and the chat page
//!   glyd-engine fetch MODEL                                      the model's files downloaded into the Hugging Face cache (no GPU)
//!   glyd-engine prepare MODEL [--layout ...]                     downloaded and packed for this GPU, the KV tables left for the first run
//!   glyd-engine doctor | login | supports MODEL                  what this computer has for it; a Hugging Face token saved; whether the engine runs MODEL (exit 3: no)
//!   glyd-engine preflight [MODEL[:LEVEL]] [--costs]               whether the engine will certainly run MODEL on this computer, said before anything is downloaded (exit 3: no, and why on stderr); --costs: its cost profile
//!   glyd-engine costs MODEL[:LEVEL] [--predict [--rates FILE]]    the model's cost profile at the level (what each layer holds, what a token reads; docs/cost-profile.md), and predicted decode speed
//!   glyd-engine rates [--dir DIR]                                 this machine's memory tiers measured: the GPU's memory, the link to it, the computer's memory, the disk under DIR
//!   glyd-engine fit MODEL[:LEVEL] [--machine NAME|FILE] [--context N] [--users N]
//!                                                             the plan before anything is downloaded: level, placement, context, predicted tokens/s (src/plan.rs)
//!   glyd-engine rangeworker --model MODEL[:LEVEL] --layers A..B [--ctx N] [--chunk N] [--head-rows N] [--sessions N] [--device D]
//!                                                             one range of a Qwen3.5 model's layers on this GPU, its frames on the standard input and output: one model across
//!                                                             processes or machines (docs/range-worker.md, glyd-range/1)
//!   glyd-engine --version
//!
//! A development build (the default features) has the checks and the measurements too: src/devcmd.rs.

use glyd_engine::entry::Dev;

#[cfg(feature = "dev")]
mod devcmd;

fn main() {
    #[cfg(feature = "dev")]
    let dev: Dev = Some(&devcmd::run);
    #[cfg(not(feature = "dev"))]
    let dev: Dev = None;
    // the whole command line is the library's (src/entry.rs), which `glyd` runs too where the engine is linked
    glyd_engine::entry::main("glyd-engine", std::env::args().skip(1).collect(), dev)
}
