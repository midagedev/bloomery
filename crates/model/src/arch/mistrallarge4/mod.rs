//! `mistral-large4`: Mistral Large 4 (1T-A52B), a text decoder of one
//! repeated layer. Every layer is MLA with no query latent — the hidden row
//! projects to every head whole, the latent's per-head up-projections absorb
//! into the query, and the cached row is `[latent | rope]`, the rope part
//! zero-width on the NoPE layers the file names one in four — with SSMax on
//! the query, its positions picked by a lightning indexer over index keys
//! projected from the layer input (a shared-indexer layer reusing the
//! previous full layer's top-k), and the KV-sharing layers the file names
//! reading another layer's cache and storing nothing. The rope is YaRN at
//! LeRoPE factors, net magnitude 1; the router is softmax top-4 of 128 with
//! one shared expert; the first layer is dense; the head softcaps its logits.
//! The file's keys (`hparams`), every tensor's role (`roles`), the tensor
//! names (`names`), the typed description and the context bound
//! (`spec`), and the load-time row moves the indexer tensors need (`rows`,
//! the family's one model-only fact). No program runs the architecture yet;
//! the dialect's authority is llama.cpp's `src/models/mistral-large4.cpp`.

pub mod hparams;
pub mod names;
pub mod roles;
pub mod rows;
pub mod spec;
