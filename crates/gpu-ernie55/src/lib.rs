//! bloomery-gpu-ernie55: the ERNIE 5.5 Flash Preview program over `bloomery-gpu`'s kernels, host code only. The model
//! description is `model::arch::ernie55`.
//!
//! The body ([`body::Body`] behind `GpuModel`) walks the parts each layer kind runs, each behind a narrow trait
//! ([`body::Mixer`], [`body::RoutedFfn`], [`body::Trunk`], [`body::HeadMix`]) that [`parts::load`] builds from the
//! file. It loads by a placement plan ([`body::Body::open_placed`]): every layer's dense part and the stores on the
//! stage card with each routed layer's card share of its experts, the rest on the common host tier. A prompt runs as
//! the pass's units ([`pass::prompt`]), each hosted layer's host experts read once for a unit's rows through the host
//! tier's batch port, and under `BLOOMERY_XSTREAM=admit` the unit's most-routed host experts move into the card's pool
//! first (`stream`); a load opened with no pass steps it.

pub mod attn_hca;
pub mod attn_mla;
pub mod body;
pub mod head;
pub mod moe;
pub mod parts;
pub mod pass;
mod placed;
mod rooms;
mod step;
mod stream;
pub mod trunk;

pub use body::{Body, Cx, HC, HeadMix, Mixer, Parts, RoutedFfn, Shape, SubIn, TapRows, Trunk};
pub use pass::prompt;
pub use placed::PASS_ROWS;
