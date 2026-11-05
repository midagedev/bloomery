//! `kimi_k3` — Kimi K3 on the card: the step ([`body`]) over the delta layers ([`kda`]), the
//! gated latent layers (the core Raon-1 shares, `crate::arch::raon1::mla`) and the routed block
//! ([`moe`]), the cross-layer residual's mix ([`crate::attn_res`]) between them, and the state
//! book of the step ([`state`]). The model description is `model::arch::kimi_k3`.

mod body;
mod kda;
pub mod moe;
mod state;

pub use body::{Body, TAP_STAGES};
