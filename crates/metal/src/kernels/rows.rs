//! What the row launches share: the check of a strided window. A row launch's
//! windows hold `rows` rows, `stride` elements apart, each `width` wide; a
//! window may be longer than the last row's end, and a stride shorter than the
//! row would overlap the rows.

use bloomery_device::{DeviceError, Elems};

use super::{range, word};

/// A strided window holds the last row's end, and the stride is not shorter
/// than a row. The extent also fits the kernel's 32-bit index.
pub(crate) fn check_strided(
    what: &'static str,
    name: &str,
    held: Elems,
    rows: usize,
    stride: usize,
    width: usize,
) -> Result<(), DeviceError> {
    if stride < width {
        return Err(range(
            what,
            format!("the {name} stride {stride} is shorter than the row's {width}"),
        ));
    }
    let need = rows
        .checked_sub(1)
        .and_then(|n| n.checked_mul(stride))
        .and_then(|n| n.checked_add(width))
        .ok_or_else(|| {
            range(
                what,
                format!("{rows} rows of {name} stride {stride} do not fit"),
            )
        })?;
    word(what, &format!("the {name} extent"), need)?;
    if held.get() < need {
        return Err(range(
            what,
            format!(
                "the {name} window holds {} elements, the rows need {need}",
                held.get()
            ),
        ));
    }
    Ok(())
}
