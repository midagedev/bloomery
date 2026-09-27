//! bloomery-gpu-glm5next — the GLM-5.3-Flash program: a body over the
//! runtime's layer walk (`runtime::sched`) whose layers are the programs
//! their descriptions name (`runtime::layer`). It has no device code of its
//! own: the KDA delta rule, the latent appends, the q8_0 gemvs and the head
//! are `bloomery-gpu`'s, the hyper-connections, the absorbed attention, the
//! GLM router, the shared expert and the host handoff `bloomery-gpu-deepseek41`'s.
//! Every routed expert runs on the host tier.
//!
//! - [`body`]: the load, the stores, the step's buffers and [`body::Body`]
//!   behind `GpuModel`;
//! - `program`: the step's walk and its host leg's port;
//! - `kda`, `mla`: the two mixers; `ffn`: the dense block and the routed
//!   block around its host leg;
//! - `host`: the host tier's routed stacks;
//! - [`forced`]: one layer alone on given streams, for the gates.
//!
//! Built with `cargo oxide`, as the device crates it links are.

pub mod body;
mod ffn;
pub mod forced;
mod host;
mod kda;
mod mla;
mod program;

pub use body::{Body, CHECKPOINT_EVERY, Glm5nextModel, prompt};
pub use host::GlmHost;
pub use program::{layer_launches, step_launches};
