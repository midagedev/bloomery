//! The nemotron_h family's half of the GPU engine: [`Body`], the decode chain
//! of a dense hybrid of Mamba-2 layers, grouped-query attention layers with
//! no rope and squared-ReLU MLP layers (the description is
//! `model::arch::nemotron_h`; the dialect's authority is llama.cpp's
//! `src/models/nemotron-h.cpp`).
//!
//! One layer is a pre-norm mixer and a residual add, by the layer's kind
//! (read from the file's per-layer arrays at load, not from its index):
//!
//! - Mamba-2: norm, the `ssm_in` projection split into `z`, `xBC` and `dt`
//!   (three views of one output), the scan step ([`crate::mamba2`]), the
//!   `ssm_out` projection;
//! - attention: norm, `q`, `k` and `v` projections, `k` and `v` appended to
//!   the layer's head-major planes, the group-5 flash pass, the output
//!   projection;
//! - MLP: norm, `ffn_up`, the squared ReLU, `ffn_down`.
//!
//! The sites run the launch their file type selects ([`crate::site`]): the
//! tensors this body reads are Q5_0, Q8_0, Q4_K and Q6_K, and any other type is
//! refused by name at load. The token embedding is a Q5_0 table and the
//! output head a Q8_0 one ([`crate::head::Head`]).
//!
//! Every per-step value (the token, the position, the key heads' row words)
//! lives in device words refreshed before the enqueue or replay, so one
//! captured graph serves every position. The body holds one sequence in one
//! slot; a prompt is one step per token.

mod body;
mod step;

pub use body::{Body, DecodeInput};
