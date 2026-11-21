//! `inkling2` — Inkling 2 (the Mini and the Small, one program). The file's keys (`hparams`),
//! every tensor's role (`roles`), the typed description (`spec`) and the tensor names a step
//! reads (`names`). No placement of its own yet and no fixture: the rounds that add them carry
//! this module forward.
//!
//! The trunk is grouped-query attention whose score path carries the positions through a learned
//! relative-position bias — no rope anywhere — with a per-(head, query) sink column in the
//! softmax denominator, per-head channel scales and whole-row L2 clips on q, k and the bias
//! projection, V RMS-normed with no gain, and causal depthwise short convolutions on the k and v
//! paths before the cache and on both sublayers' outputs. A windowed layer (the pattern's `true`)
//! shares one attention input norm; a windowless layer gives q, k and v an input norm each and
//! input-dependent per-head conv taps. Every sublayer reads the softmax mix of the banked block
//! inputs and the running stream ([`models::Residual::AttnResMix`]), every layer routes by the
//! biased sigmoid router over `situ` experts with no shared one, and the head softcaps its logits
//! then cuts them to the unpadded vocabulary.
//!
//! The Mini loops: `[core_start, core_end)` runs `passes` times, the visits of a core layer
//! sharing its weights, each visit its own cache slot, and each pass past the first restarting
//! from the prelude output and reading the previous pass's banked blocks through a gated
//! rank-`gate_rank` message controller a core layer carries ×4 (q, k, v, ffn). The Small is the
//! same file without the loop.
//!
//! Two spellings of every loop fact reach the one description ([`hparams::Dialect`]): the
//! unsloth files' `inkling2` word (the `loop.core_*` keys and `blk.N.loop_{attn,k,v,mlp}_*`
//! stems) and llama.cpp master's renamed `inkling2_loop` word (the `loop.start/end/count` keys
//! and `blk.N.loop_msg_query` + `blk.N.loop_{q,k,v,ffn}_ctrl_*`). The dialect's authority is
//! llama.cpp's `src/models/inkling2.cpp` (and `inkling2-loop.cpp` for the second spelling).

pub mod hparams;
pub mod names;
pub mod roles;
pub mod spec;
