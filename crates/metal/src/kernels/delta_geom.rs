use crate::geom::{Geometry, KernelRefused};
use crate::geometry;
use crate::instance::HOST;

/// Taps of the causal conv. A shim: the build's host instance's delta axes.
/// Callers left: the arch, the e2e and kernel gates, their oracles and the
/// probes.
pub const CONV_TAPS: usize = HOST.delta_built().conv_taps as usize;

/// Positions one step's position window may name.
pub const PASS_ROWS: usize = 8;

/// Slots of the conv ring of a conv of `taps` taps: the predecessors it
/// reads (the taps except the current token, none for a conv of no taps)
/// plus the positions of one pass.
#[must_use]
pub const fn ring_rows(taps: usize) -> usize {
    taps.saturating_sub(1) + PASS_ROWS
}

/// Slots of the conv ring. A shim: the build's host instance's. Callers left:
/// the arch, the e2e and kernel gates and their oracles.
pub const RING_ROWS: usize = ring_rows(CONV_TAPS);

/// Why a lane-ring launch of some rows over some lanes cannot be taken.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RingRefusal {
    /// No rows.
    NoRows,
    /// More rows than lanes: two rows would end in one lane.
    Lanes { rows: usize, lanes: usize },
    /// More rows than the conv ring rolls back ([`PASS_ROWS`]): a slot a kept
    /// row reads would hold a dropped row's input.
    RollBack { rows: usize },
}

impl core::fmt::Display for RingRefusal {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            Self::NoRows => write!(f, "a lane ring of no rows"),
            Self::Lanes { rows, lanes } => write!(
                f,
                "a lane ring of {rows} rows needs {rows} lanes; the state holds {lanes}"
            ),
            Self::RollBack { rows } => write!(
                f,
                "a lane ring of {rows} rows; the conv ring rolls back at most {PASS_ROWS}"
            ),
        }
    }
}

/// Whether a lane ring of `rows` rows fits `lanes` lanes: each row ends in a
/// lane of its own, and the conv ring, whose slots are addressed by position,
/// rolls back at most [`PASS_ROWS`] positions. The one owner of the rule: the
/// launch and the verify's bookkeeping both ask it.
///
/// # Errors
///
/// [`RingRefusal`] naming what the lanes or the conv ring cannot serve.
pub const fn ring_fits(rows: usize, lanes: usize) -> Result<(), RingRefusal> {
    if rows == 0 {
        return Err(RingRefusal::NoRows);
    }
    if rows > lanes {
        return Err(RingRefusal::Lanes { rows, lanes });
    }
    if rows > PASS_ROWS {
        return Err(RingRefusal::RollBack { rows });
    }
    Ok(())
}

/// Copies of the state lane and the conv ring one rows launch may take after
/// rows inside it: a prompt call's first and last message start.
pub const SNAPS: usize = 2;

/// The geometry the delta kernels are written for, held for `g`'s delta axes
/// (the axes its kernels are compiled at, [`crate::instance::Instance::delta_built`]):
/// the kernels' own `static_assert`s, stated where the host sizes its launches.
///
/// # Errors
///
/// [`KernelRefused`] naming the axis the delta step cannot take.
pub fn holds(g: &Geometry) -> Result<(), KernelRefused> {
    let d = &g.delta;
    if d.ring_rows != 11 {
        return Err(KernelRefused::new(
            "delta",
            "conv taps",
            d.conv_taps,
            "4 (the conv ring holds 11 slots)",
        ));
    }
    if d.value_heads == 0 || !d.value_heads.is_multiple_of(2) {
        return Err(KernelRefused::new(
            "delta",
            "value heads",
            d.value_heads,
            "an even count above 0 (a prepared row is whole 16-byte words)",
        ));
    }
    if d.head_cols != 128 || d.head_cols / geometry::PARTIALS_PER_HEAD != 16 {
        return Err(KernelRefused::new(
            "delta",
            "head columns",
            d.head_cols,
            "128 (16 columns a block)",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_lane_ring_fits_as_many_rows_as_lanes_up_to_the_roll_back() {
        for lanes in 1..=PASS_ROWS + 2 {
            for rows in 0..=PASS_ROWS + 3 {
                let want = if rows == 0 {
                    Err(RingRefusal::NoRows)
                } else if rows > lanes {
                    Err(RingRefusal::Lanes { rows, lanes })
                } else if rows > PASS_ROWS {
                    Err(RingRefusal::RollBack { rows })
                } else {
                    Ok(())
                };
                assert_eq!(
                    ring_fits(rows, lanes),
                    want,
                    "{rows} rows over {lanes} lanes"
                );
            }
        }
    }

    #[test]
    fn a_refused_lane_ring_names_the_rows_and_the_lanes() {
        let lanes = ring_fits(3, 2).unwrap_err().to_string();
        assert!(
            lanes.contains("needs 3 lanes") && lanes.contains("holds 2"),
            "{lanes}"
        );
        let roll = ring_fits(PASS_ROWS + 1, PASS_ROWS + 1)
            .unwrap_err()
            .to_string();
        assert!(roll.contains("rolls back at most 8"), "{roll}");
        assert_eq!(PASS_ROWS, 8, "the text above names the roll-back");
    }
}
