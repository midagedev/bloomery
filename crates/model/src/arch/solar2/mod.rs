//! `solar-open2` — Solar-Open2. The file's keys (`hparams`), every tensor's role (`roles`), the
//! typed description (`spec`), the tensor names a step reads (`names`) and the host tier's view of a
//! routed layer (`host`).
//!
//! The trunk is Kimi delta attention (KDA) layers with a NoPE GQA layer between them, the GQA
//! layers being those whose `attention.head_count_kv` is not 0. A GQA layer multiplies its
//! attention output by the sigmoid of a projection of its own (`attn_gate`). Every layer routes by
//! a sigmoid noaux_tc router over SwiGLU experts beside an ungated shared one, on a plain
//! residual. The dialect's authority is llama.cpp's `solar-open2` loader: the file is converted
//! with its tensor layout, which differs from ik's KDA layout in `ssm_a`'s dims.

pub mod host;
pub mod hparams;
pub mod names;
pub mod place;
pub mod roles;
pub mod spec;
