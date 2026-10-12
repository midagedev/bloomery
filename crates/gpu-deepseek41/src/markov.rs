//! The DSpark draft's Markov head lives with the common draft's head,
//! [`bloomery_gpu::dspark::head`]; this path names it for the callers that
//! reach it from here.

pub use bloomery_gpu::dspark::head::{MARKOV_RANK, MARKOV_ROW_WORDS, MarkovKernels, MarkovWeights};
