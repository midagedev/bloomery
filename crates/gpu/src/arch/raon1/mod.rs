//! `raon1` — Raon-1-mini on one card: the chain ([`body`]), Raon_mHC's
//! streams ([`mhc`]), the SiTU feed-forward blocks and the 128/6 router
//! ([`ffn`]), the RDA layer ([`rda`]) and the gated latent layer ([`mla`]);
//! the passes of several rows — a verify, a prompt call's chunks — and the
//! taps a block draft reads ([`rows`]), over the RDA stores' lanes
//! ([`state`]); and a prompt's units of more rows than a pass, which run the
//! projections and the experts as GEMMs ([`wide`]). The model description is
//! `model::arch::raon1`.

mod body;
pub(crate) mod ffn;
mod mhc;
pub mod mla;
pub mod rda;
mod rows;
mod state;
#[cfg(test)]
mod synth;
mod wide;

pub use body::Body;
pub use ffn::sigmoid_128_6;
pub use mhc::{HC_OUT, STREAMS};
