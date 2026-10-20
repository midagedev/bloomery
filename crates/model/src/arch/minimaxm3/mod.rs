//! `minimax-m3` — MiniMax-M3. The file's keys (`hparams`), every tensor's role (`roles`), the
//! typed description (`spec`), the tensor names a step reads (`names`) and the file planned onto a
//! machine's cache bytes (`place`).
//!
//! Every layer is a GQA layer with a per-head q/k norm and a partial NeoX rope. The leading
//! layers run a dense `swigluoai` block and attend every position; the rest route by a sigmoid
//! noaux_tc router over `swigluoai` experts beside an ungated shared one, and attend only the
//! blocks of positions their own indexer picks (MiniMax Sparse Attention). The dialect's
//! authority is llama.cpp's `src/models/minimax-m3.cpp`.

pub mod hparams;
pub mod names;
pub mod place;
pub mod roles;
pub mod spec;
