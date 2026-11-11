//! The port of a chain with no host leg, shared by every family's walk.

use crate::ChainError;
use runtime::sched::{Overlap, Port, PortKind, Refused};
use std::marker::PhantomData;

const WHAT: &str = "chain::NoLeg";

/// The port of a chain with no host leg: it exchanges nothing, and opens one unit of `1..=rows`
/// columns, `rows` the backend's arena's.
pub(crate) struct NoLeg<E> {
    rows: usize,
    _error: PhantomData<fn() -> E>,
}

impl<E> NoLeg<E> {
    pub(crate) fn new(rows: usize) -> NoLeg<E> {
        NoLeg {
            rows,
            _error: PhantomData,
        }
    }
}

impl<E: From<ChainError>> Port for NoLeg<E> {
    type Error = E;
    const KIND: PortKind = PortKind::Step;

    /// Refuses by name more than one unit (no leg to overlap across units) and more columns than
    /// the arena holds rows.
    fn open(&mut self, o: Overlap) -> Result<(), E> {
        if o.units != 1 || !(1..=self.rows).contains(&o.cols) {
            return Err(ChainError::shape(
                "chain::NoLeg::open",
                format!(
                    "{} units of {} columns; a chain with no host leg walks one unit of \
                     1..={} columns (the arena's rows)",
                    o.units, o.cols, self.rows
                ),
            )
            .into());
        }
        Ok(())
    }

    fn refused(why: Refused) -> E {
        ChainError::shape(WHAT, why.to_string()).into()
    }
}
