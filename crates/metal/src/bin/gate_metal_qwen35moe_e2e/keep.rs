//! Prompt helpers the pass clauses share.
//!
//! `k_keep` compared checkpoints to ik's d1k prompt.
//! PIN(2026-10-28): removed, user decision, Metal verifies MLX only; the GGUF path runs unverified

use bloomery_metal::arch::qwen3moe::session::Session;
use runtime::{Out, Target, Want};

use super::says;

pub(super) fn prompt_bits(s: &mut Session, ids: &[u32]) -> Result<(Vec<u32>, u32), String> {
    prompt_bits_marked(s, ids, &[])
}

/// [`prompt_bits`] with a mark at each of `marks`, the way a server's chat
/// turn passes them.
pub(super) fn prompt_bits_marked(
    s: &mut Session,
    ids: &[u32],
    marks: &[u32],
) -> Result<(Vec<u32>, u32), String> {
    match Target::prompt_marked(s, ids, marks, Want::Logits).map_err(says)? {
        Out::Logits { argmax, row } => Ok((row.iter().map(|v| v.to_bits()).collect(), argmax)),
        Out::Argmax(_) => Err("a Want::Logits prompt returned the argmax alone".into()),
    }
}
