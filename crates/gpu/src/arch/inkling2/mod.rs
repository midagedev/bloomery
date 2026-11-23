//! `inkling2` — Inkling 2 on the card: the whole-card step ([`body`]) over the visits of the
//! looped trunk — per visit the cross-layer residual mixes (`crate::attn_res`), the looped core's
//! message controllers (`crate::loop_control`), the residual short convs
//! (`crate::linear::conv_res`), the projection epilogues and the f16 append (`crate::qk_clip`)
//! and the rel-bias attention with its per-query sink (`crate::rel_bias`), then the routed block
//! on the common card leg — and the state book of the step ([`state`]), one store set a visit.
//! The model description is `model::arch::inkling2`.

mod body;
mod seat;
mod state;

pub use body::{Body, PAD_LOGIT, TAP_STAGES};
pub use state::StoreHost;
