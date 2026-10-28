//! What the row launches share: the check of a strided window. A row launch's
//! windows hold `rows` rows, `stride` elements apart, each `width` wide; a
//! window may be longer than the last row's end, and a stride shorter than the
//! row would overlap the rows.

use bloomery_device::{DeviceError, Elems};

use super::range;

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
    // The refusal is built on the error path only: a launch that passes
    // allocates nothing.
    if u32::try_from(need).is_err() {
        return Err(range(
            what,
            format!("the {name} extent of {need} does not fit the kernel's 32-bit word"),
        ));
    }
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

#[cfg(test)]
mod tests {
    use bloomery_device::Elems;

    use super::check_strided;

    /// A strided window's extent is held to the kernel's 32-bit index: the
    /// last extent that fits passes, and the first past it is refused by
    /// name with the extent.
    ///
    /// Mutant: delete the `u32::try_from(need)` refusal.
    #[test]
    fn a_strided_extent_past_the_kernels_word_is_refused_by_name() {
        let top = u32::MAX as usize;
        assert!(check_strided("test", "x", Elems(top), 2, top - 1, 1).is_ok());
        let said = check_strided("test", "x", Elems(top + 1), 2, top, 1)
            .unwrap_err()
            .to_string();
        assert!(
            said.contains("the x extent of 4294967296 does not fit the kernel's 32-bit word"),
            "{said}"
        );
    }
}
