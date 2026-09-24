//! The qwen3moe (Qwen3-30B-A3B) half of the GPU engine: kernels whose
//! constants are this architecture's and nothing else's. The shape kernels it
//! runs over — the NEOX norm/rope/append, the grouped-query flash, the Q6_K
//! down `_sel` — live in the crate root beside the runtime.

pub mod router;
