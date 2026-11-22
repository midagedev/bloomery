//! `generate_metal --plan` for a qwen35moe file: what it asks of this Mac,
//! read from its header and tensor table alone.
//!
//! The frame — the header read (whole file or header alone), the device's and
//! the host's room, the byte lines, the render and its exit codes — is the
//! crate's own [`crate::plan`], generic over the family's verdict; this
//! module states the qwen3moe family's ([`Verdict` for `Outcome`]: the arch
//! line from the description the check read, the refusals, the checked
//! file's bytes) and the cache rows of an open that asked for none, which
//! the family's room rule cuts.

use std::path::Path;

use gguf::Split;

use super::check::{Bytes, Outcome};
use super::session::SessionError;

pub use crate::plan::*;

/// The cache rows of an open of `model` that asked for none: the smaller of
/// `cap` and the file's `context_length`, cut to the largest cache whose
/// session fits the device's free bytes when that default's does not
/// ([`super::check::unasked_ctx`]). A file with no `context_length` count is
/// refused by name.
pub fn unasked_ctx(split: &Split, cap: usize, model: &Path) -> Result<Unasked, String> {
    let train = split
        .arch_get_u64("context_length")
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| {
            format!(
                "{}: {} is absent or not a count",
                model.display(),
                split.arch_key("context_length")
            )
        })?;
    let default = cap.min(train);
    let free = device_room().ok().map(|d| d.free);
    let ctx = free.map_or(default, |free| {
        super::check::unasked_ctx(split, default, free)
    });
    Ok(Unasked { ctx, default, free })
}

/// The qwen3moe family's plan verdict: the arch line from the description
/// the check read, its refusals, and the checked file's bytes.
impl crate::plan::Verdict for Outcome {
    type Error = SessionError;

    fn spec_line(&self) -> Option<String> {
        self.spec.as_ref().map(|s| {
            format!(
                "arch={} layers={} (delta {}, conv {}, attention {}) hidden={} vocab={} \
                 experts={} picks={} trained_ctx={}",
                s.arch,
                s.layers,
                s.delta,
                s.conv,
                s.attention,
                s.hidden,
                s.vocab,
                s.experts,
                s.picks,
                s.ctx_train
            )
        })
    }

    fn items(&self) -> &[crate::plan::Item<SessionError>] {
        &self.items
    }

    fn checked_bytes(&self) -> Option<Result<Bytes, String>> {
        self.checked
            .as_ref()
            .map(|checked| checked.bytes().map_err(|e| e.to_string()))
    }
}
