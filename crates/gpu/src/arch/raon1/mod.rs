//! `raon1` — Raon-1's two dialects on one card: the chain ([`body`]), Raon_mHC's
//! streams ([`mhc`], the split dialect's scale route and the Nano file's
//! folded chart one), the SiTU and SwiGLU feed-forward blocks and the 128/6
//! router ([`crate::arch::routed_ffn`]), the RDA layer ([`rda`], whose
//! load-time folds split the Nano file's fused stems) and the gated latent
//! layer ([`crate::arch::gated_mla`], whose wide arm is [`mla`]); the row
//! gathers both dialects' tables take ([`gather`]), the Nano dialect's
//! per-layer input ([`ple`]);
//! the passes of several rows — a verify, a prompt call's chunks — and the
//! taps a block draft reads ([`rows`]), over the RDA stores' lanes
//! ([`state`]); and a prompt's units of more rows than a pass, which run the
//! projections and the experts as GEMMs ([`wide`]). The model description is
//! `model::arch::raon1`.

mod body;
mod gather;
mod mhc;
mod mla;
mod ple;
pub mod rda;
mod rows;
mod state;
#[cfg(test)]
mod synth;
mod wide;

pub use crate::route_body::sigmoid_128_6;
pub use body::Body;
pub use mhc::{CHART_LEN, ChartVecs, HC_MIX, HC_OUT, MhcKernels, STREAMS, SiteVecs, parts_len};
pub use ple::PleKernels;
