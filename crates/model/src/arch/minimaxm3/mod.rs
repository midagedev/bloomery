//! `minimax-m3` — MiniMax-M3. The file's keys (`hparams`), every tensor's role (`roles`), the
//! typed description (`spec`), the tensor names a step reads (`names`), the file planned onto a
//! machine (`place`) and the host tier's view of a routed layer (`host`).
//!
//! Every layer is a GQA layer with a per-head q/k norm and a partial NeoX rope. Attention and the
//! feed-forward block are separate facts. The first `leading_dense_block_count` layers run a dense
//! `swigluoai` block and the rest route by a sigmoid noaux_tc router over `swigluoai` experts
//! beside an ungated shared one. Absent `attention.sparse_layers`, a layer attends the blocks its
//! indexer picks (MiniMax Sparse Attention) exactly when the file carries that indexer: a routed
//! layer always does, a dense-block layer may, and a layer without one attends every position.
//! Present, the array names the sparse-attention layers. The dialect's authority is llama.cpp's
//! `src/models/minimax-m3.cpp`.

pub mod host;
pub mod hparams;
pub mod names;
pub mod place;
pub mod roles;
pub mod spec;
