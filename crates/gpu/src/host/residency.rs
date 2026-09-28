//! The host set a placed load read in and, when asked, locked: a load
//! artifact the host tier holds for the model's lifetime, declared before
//! the host computation whose mappings its lock's spans are pages of.

use crate::GpuError;
use bloomery_levers::HostCfg;
use gguf::Split;
use model::placement::host_lock::{HostLock, HostSet, Walk};
use model::placement::{ExpertList, ModelTensor, Plan};
use model::r8file::{HostR8, R8Source};

/// What a placed load did to its plan's host set: populated it, locked it,
/// both or neither ([`HostCfg`]), in the files the host tier reads
/// ([`HostR8`]). Holds the lock, when there is one, for as long as it lives;
/// its owner keeps the split's mappings alive longer.
pub struct HostResidency {
    set: HostSet,
    populate: Option<Walk>,
    lock: Option<HostLock>,
}

impl HostResidency {
    /// The host set of `plan` over `split` — the host segments whose tensor
    /// `keep` selects — populated, then locked, as `cfg` asks. Populating
    /// first makes the lock a walk over resident pages. When the host tier
    /// reads the r8 sidecar ([`HostR8::at_load`] under `cfg.r8`, the same
    /// reading and the same mapping a V4.1 host tier's build takes), the
    /// stacks it holds are walked in its mapping and their source pages are
    /// not; everything else is the source's.
    pub fn at_load(
        split: &Split,
        plan: &Plan<'_>,
        keep: impl Fn(&ModelTensor) -> bool,
        cfg: HostCfg,
    ) -> Result<HostResidency, GpuError> {
        HostResidency::at_load_with(split, plan, keep, cfg, &[])
    }

    /// [`HostResidency::at_load`] of a set that also holds `extra`, experts
    /// of card segments the host must serve too ([`HostSet::of_with`]:
    /// adaptive residency's churn pool).
    pub fn at_load_with(
        split: &Split,
        plan: &Plan<'_>,
        keep: impl Fn(&ModelTensor) -> bool,
        cfg: HostCfg,
        extra: &[(usize, ExpertList)],
    ) -> Result<HostResidency, GpuError> {
        const WHAT: &str = "HostResidency::at_load";
        let r8 = HostR8::at_load(split, cfg.r8)?;
        let src = R8Source::of(split, &r8)?;
        let set = HostSet::of_with(src, plan, keep, extra).map_err(|e| GpuError::plan(WHAT, e))?;
        let populate = cfg
            .populate
            .then(|| set.populate(src))
            .transpose()
            .map_err(|e| GpuError::plan(WHAT, e))?;
        let lock = cfg
            .lock
            .then(|| HostLock::lock(src, &set))
            .transpose()
            .map_err(|e| GpuError::plan(WHAT, e))?;
        Ok(HostResidency {
            set,
            populate,
            lock,
        })
    }

    /// The host set the load walked.
    #[must_use]
    pub fn set(&self) -> &HostSet {
        &self.set
    }

    /// The populate walk; `None` with `BLOOMERY_HOST_POPULATE=0`.
    #[must_use]
    pub fn populated(&self) -> Option<&Walk> {
        self.populate.as_ref()
    }

    /// The lock; `None` unless `BLOOMERY_HOST_LOCK=1`.
    #[must_use]
    pub fn lock(&self) -> Option<&HostLock> {
        self.lock.as_ref()
    }
}
