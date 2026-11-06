//! `kimi_k3` — Kimi K3 and K3.1 (the architecture words `kimi-k3` and `kimi-k31`, one registry row:
//! K3.1 is K3 whose latent layers also carry a lightning indexer, [`hparams::Indexer`]). The file's keys (`hparams`), every tensor's role (`roles`), the typed
//! description (`spec`), the tensor names a step reads (`names`), the file planned onto a
//! machine's cache bytes (`place`) and the synthetic gate fixture cut from the released file
//! (`fixture`).
//!
//! The trunk mixes with Kimi delta attention (KDA) layers and gated latent-attention (MLA)
//! layers, the MLA layers being those whose `attention.head_count_kv` is 1 and the KDA layers
//! those whose is 0. A KDA layer is the Kimi delta rule whose decay is bounded by
//! `kda.gate_lower_bound` and whose output gate is one full-width projection (`ssm_g`). An MLA
//! layer rotates nothing: the `rope.dimension_count` key values beside the latent are keys no
//! rope turns, and its heads' outputs are gated by a projection of the layer input. The first
//! layer is a dense `situ` block, the rest route by a sigmoid noaux_tc router over `situ`
//! experts that run in a latent (`expert_latent_length`) beside a shared stack, and every
//! sublayer reads a softmax mix of the banked block inputs and the running sum (AttnRes). The
//! dialect's authority is llama.cpp's `src/models/kimi-k3.cpp`.

pub mod fixture;
pub mod hparams;
pub mod names;
pub mod place;
pub mod roles;
pub mod spec;
