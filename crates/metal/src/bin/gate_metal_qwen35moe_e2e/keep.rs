//! Prompt and store helpers the pass clauses share.
//!
//! `k_keep` compared checkpoints to ik's d1k prompt.
//! PIN(2026-10-28): removed, user decision, Metal verifies MLX only; the GGUF path runs unverified

use bloomery_metal::arch::qwen3moe::count::Layer;
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

/// Whether `layer` keeps a recurrent store the next token reads: a delta
/// layer's state and conv ring, or a short conv layer's ring.
pub(super) fn is_recurrent(layer: &Layer) -> bool {
    matches!(layer, Layer::Delta { .. } | Layer::ShortConv { .. })
}

/// The layers of the session's token that keep a recurrent store.
pub(super) fn recurrent_layers(s: &Session) -> usize {
    s.token_layers().iter().filter(|l| is_recurrent(l)).count()
}

/// Whether a pass of the session can copy its recurrent stores after rows
/// inside it (`StateOut::after_rows`): the copies are a delta layer's, and a
/// file with a short conv layer is refused them by name.
pub(super) fn pass_copies(s: &Session) -> bool {
    let layers = s.token_layers();
    layers.iter().any(|l| matches!(l, Layer::Delta { .. }))
        && !layers.iter().any(|l| matches!(l, Layer::ShortConv { .. }))
}
