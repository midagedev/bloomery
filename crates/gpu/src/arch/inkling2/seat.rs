//! Inkling 2's body behind the traits the common seats drive a draft through: [`Rollback`], which
//! takes back the positions its conv ring and window rings still hold, and [`Slots`], which the
//! serve seats' draft owners ask of every body they serve — this one holds a single sequence, so
//! a second is refused by name.

use super::body::Body;
use crate::Gpu;
use crate::GpuError;
use crate::model::{Rollback, Slots};

const ROLLBACK: &str = "inkling2::Body::rollback";
const SLOTS: &str = "inkling2::Body::slots";

impl Rollback for Body {
    /// Back to `pos`, no device work: every store is positional (a visit's K and V by position,
    /// its conv ring by `position mod rows`, the rest per token), and the next step or pass
    /// writes a position's rows before any launch reads them. The depth is measured from the
    /// written end ([`Body::rollback_floor`]): the conv ring holds [`Body::rollback_reach`]
    /// positions below it ([`runtime::stores::PASS_ROWS`]), and a position below that is refused
    /// by name. The written end stays, so a second rollback is as deep as the first and the span
    /// between.
    fn rollback(&mut self, pos: u32) -> Result<(), GpuError> {
        if self.held == u32::MAX {
            return Err(GpuError::shape(
                ROLLBACK,
                format!(
                    "a rollback to position {pos}: the stores stand nowhere (a call that failed \
                     part-way: reset)"
                ),
            ));
        }
        if pos > self.held {
            return Err(GpuError::shape(
                ROLLBACK,
                format!(
                    "a rollback to position {pos} past the {} the stores stand at",
                    self.held
                ),
            ));
        }
        let floor = self.rollback_floor();
        if pos < floor {
            return Err(GpuError::shape(
                ROLLBACK,
                format!(
                    "a rollback to position {pos} from {}: the conv ring holds {} positions \
                     below the written end, down to {floor}",
                    self.held,
                    self.rollback_reach()
                ),
            ));
        }
        self.held = pos;
        Ok(())
    }
}

/// The body holds one sequence: its visits' stores are one context's. The serve seats' draft
/// owners ([`Slots`]) select and count slots of every body; here slot 0 is the only one, so a
/// select of it is a no-op and a second resident sequence is refused by name.
impl Slots for Body {
    type Seq = ();

    /// Refused by name: a second resident sequence is never made. The seat refuses `--parallel`
    /// above one before any plan.
    fn new_seq(&mut self, _gpu: &Gpu) -> Result<(), GpuError> {
        Err(GpuError::shape(
            SLOTS,
            "the inkling2 body holds one sequence: no second resident sequence is made",
        ))
    }

    /// Refused by name: no sequence is ever parked ([`Slots::new_seq`]), so there is nothing to
    /// exchange.
    fn swap_seq(&mut self, _gpu: &Gpu, _seq: &mut ()) -> Result<(), GpuError> {
        Err(GpuError::shape(
            SLOTS,
            "the inkling2 body holds one sequence: nothing to exchange",
        ))
    }

    /// The one sequence's stores.
    fn seq_bytes(&self) -> usize {
        self.store_bytes()
    }
}
