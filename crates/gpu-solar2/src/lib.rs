//! bloomery-gpu-solar2 — the Solar-Open2 program: a body over the runtime's
//! layer walk (`runtime::sched`) whose layers are the programs their
//! descriptions name (`runtime::layer`): `Gqa` (NoPE, out gate by its own
//! projection) at every fourth layer from 0, `DeltaRule(Kda)` elsewhere, a
//! routed block with one shared expert on every layer. It has no device code
//! and no copy of another family's launches: the KDA mixer is
//! `bloomery_gpu::linear::mixer_kda`, the GQA layer
//! `bloomery_gpu::arch::hybrid::gqa_layer`, the tier shell
//! `bloomery_gpu::host::shell`, the router `sigmoid_320_8` and the gated
//! block `bloomery-gpu-deepseek41`'s.
//!
//! What only Solar's architecture forces stays here: the layer walk over its
//! two mixer kinds, its stores (a KDA recurrent store a linear layer, K/V
//! planes a GQA layer), its typed-site projections of the KDA mixer
//! ([`kda`]), and the step and the prompt batch over the same buffers.
//!
//! - [`body`]: the load, the stores, the buffers, [`body::Body`];
//! - [`step`]: the one-token step's walk and its host leg;
//! - [`prefill`]: the prompt batch over `m` rows, bit for bit the steps;
//! - [`kda`]: the KDA projections over typed sites ([`mixer_kda::KdaProj`]);
//! - [`ffn`]: the routed block with its shared expert around the host leg.
//! - [`tier`]: the expert tier on the second card, the common leg tier over Solar's shape.
//!
//! [`mixer_kda::KdaProj`]: bloomery_gpu::linear::mixer_kda::KdaProj

pub mod body;
pub mod ffn;
pub mod kda;
pub mod prefill;
pub mod step;
pub mod tier;

pub use body::{Body, Solar2Model, StepInput, set_taps};
pub use prefill::set_prefill_group;
pub use tier::NODE_FAMILY;
