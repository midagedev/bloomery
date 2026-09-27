//! The model-free K-quant expert family: one row walk per plane layout, one
//! super-block decoder per format, and the expert-select entries over them.
//!
//! - [`walk`]: Walk A, the walk Q4_K, Q5_K and Q8_0 share (lane map, q4 plane,
//!   iteration pairs, rounding), generic over [`walk::SbDecode`]. Its Q4_K
//!   decoder is `cores::q4k_sb_decode`, so the Q4_K instance of any entry
//!   here is the arithmetic of this crate's Q4_K gemvs bit for bit.
//! - [`q5k`]: Q5_K's decoder (the qh plane's bit as a code's fifth bit).
//! - [`q8_0`]: Q8_0's decoder over the file's 34-byte blocks, eight to a
//!   Walk A super-block.
//! - [`sel`]: the down `_sel` and the gate·up with its rule as a launch
//!   argument, generic bodies and their Q5_K and Q8_0 entries.
//! - [`act`]: the gate·up rules.
//!
//! No entry here names a model: a caller passes its stacks' rows, experts
//! and super-blocks. Which types a card runs is the caller's program
//! to declare; this family only adds the kernels.

pub mod act;
pub mod q5k;
pub mod q8_0;
pub mod sel;
pub mod walk;

pub use act::Act;
pub use q5k::Q5k;
pub use q8_0::Q8_0;
pub use sel::{GateUpAct, KquantKernels, SelDown, walk_a_planes};
pub use walk::{Q4k, SbDecode};
