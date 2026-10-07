//! One server binary's seats: each module is one model the server serves —
//! the flags it takes, the records it prints, its cache and slot rules —
//! behind [`bloomery_gpu_gates::bind::Seat`], with a `run(&args)` a binary
//! calls once. The per-model binaries (`bloomery-serve-ds41`,
//! `bloomery-serve-qwen38`) include their one seat directly; the one-binary
//! server (`bloomery-serve --model ds41|qwen38|glm|qwen3`) includes this
//! registry and is one call into the seat the model names.
//!
//! A seat's module is its contract's owner: read it for the flags, the
//! records, the keep rule and the prompt cache. [`glm`] is the GLM-5.3-Flash
//! seat, opened as `generate_glm5next` opens the model; [`qwen3`] the
//! single-card Qwen seat (qwen3moe, qwen35moe), opened as
//! `generate_qwen3moe` opens them. `drafted` is the
//! seats' shared MTP draft driver, one file a seat's per-model binary
//! includes beside the seat; [`ctx`] the seats' shared `--ctx` rules (the
//! search bisection, the expert-margin guard, and the slot count a set
//! `--ctx` buys through `ctx::slots_of`: one slot at the whole context
//! while `--parallel` names none), included the same way.

#[cfg(feature = "deepseek41")]
mod drafted;
#[cfg(feature = "deepseek41")]
pub mod ds41;
#[cfg(feature = "glm5next")]
pub mod glm;
// The seats' shared `--ctx` default rule: the search bisection and the
// expert-margin guard (qwen38's, which the glm seat lifts).
#[cfg(any(feature = "deepseek41", feature = "glm5next"))]
pub mod ctx;
// The qwen38 seat runs no V4.1 code, but the server surface it sits on —
// `bind` and the `serve` and `sampler` crates — is scoped to `deepseek41`,
// as its per-model binary's doc says.
#[cfg(feature = "deepseek41")]
pub mod qwen38;
// The qwen3 seat sits behind the same server-surface feature.
#[cfg(feature = "deepseek41")]
pub mod qwen3;
// The seats' one owner of a round run as passes (`step_rows_one_pass`).
#[cfg(feature = "deepseek41")]
pub mod rounds;
// The decide seat: a decision model's backbone on the card, its head on the host.
#[cfg(feature = "clef")]
pub mod decide;
