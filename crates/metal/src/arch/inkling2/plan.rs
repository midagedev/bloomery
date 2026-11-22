//! `generate_metal --plan` for an Inkling 2 file: what it asks of this Mac,
//! read from its header and tensor table alone. The frame (the header read,
//! the device's and the host's room, the render and its exit codes) is the
//! crate's own [`crate::plan`]; this module states the family's verdict and
//! the cache rows of an open that asked for none.

use std::path::Path;

use gguf::Split;

use super::check::{Checked, Outcome, Refusal, check, check_room};
use crate::plan::{Header, HostMemory, Item, Printed, Verdict, device_room, render};
use crate::session::room::{Bytes, SessionBytes};

impl Verdict for Outcome {
    type Error = Refusal;

    fn spec_line(&self) -> Option<String> {
        self.spec_line.clone()
    }

    fn items(&self) -> &[Item<Refusal>] {
        &self.items
    }

    fn checked_bytes(&self) -> Option<Result<Bytes, String>> {
        self.checked.as_ref().map(|c| Ok(c.bytes()))
    }
}

/// An Inkling 2 file as the room rule reads it.
pub struct File<'a>(pub &'a Split);

impl SessionBytes for File<'_> {
    type Checked = Checked;
    type Error = Refusal;

    fn check_at(&self, rows: usize) -> Option<Checked> {
        check(self.0, rows).checked
    }

    fn bytes_at(checked: &Checked, _every: u32) -> Result<Bytes, Refusal> {
        Ok(checked.bytes())
    }

    fn ctx(checked: &Checked) -> usize {
        checked.ctx
    }

    fn refuse_room(detail: String) -> Refusal {
        Refusal(detail)
    }
}

/// The cache rows of an open that asked for none: the smaller of `cap` and
/// the file's `context_length`, cut to the largest cache whose session fits
/// the device's free bytes. A file with no `context_length` count is refused
/// by name.
///
/// # Errors
/// The missing count, by its key.
pub fn unasked_ctx(split: &Split, cap: usize, model: &Path) -> Result<usize, String> {
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
    Ok(device_room().ok().map_or(default, |d| {
        crate::session::room::unasked_ctx(&File(split), default, 0, d.free)
    }))
}

/// The plan of `model`, whose header `header` holds, at `ctx` rows (`None`:
/// the unasked default under `cap`): its lines and its exit code.
///
/// # Errors
/// A file with no `context_length`.
pub fn plan_of(
    model: &Path,
    header: &Header,
    ctx: Option<usize>,
    cap: usize,
) -> Result<Printed, String> {
    let ctx = match ctx {
        Some(c) => c,
        None => unasked_ctx(header.split(), cap, model)?,
    };
    let device = device_room();
    let free = device.as_ref().map(|d| d.free).map_err(Clone::clone);
    let outcome = check_room(header.split(), ctx, free);
    Ok(render(
        model,
        header.data,
        ctx,
        &outcome,
        &device,
        &HostMemory::probe(),
    ))
}
