//! Shared host-tier records for server loads and request spans.

use crate::record;

/// A server's `tier census` records, each the span since the census it last kept.
#[derive(Clone, Copy, Debug, Default)]
pub struct CensusSpan(Option<bloomery_gpu::host::census::TierCensus>);

impl CensusSpan {
    /// The `tier census` record, phase `phase`, of the span from the census last kept to `now`,
    /// on stderr, and `now` kept: nothing at a first census or on a load whose host counts no
    /// reads (`now` `None`).
    pub fn mark(&mut self, phase: &str, now: Option<bloomery_gpu::host::census::TierCensus>) {
        let Some(now) = now else {
            return;
        };
        if let Some(last) = self.0.replace(now) {
            record::require(record::groups::host_tier::CENSUS, "CensusSpan::mark");
            record::census_record(phase, &now.since(&last)).eprint();
        }
    }

    /// Before a prompt call on the host tier `hybrid` (`None` on a load with none): the `tier
    /// census` record of the steps since the census last kept.
    pub fn steps(
        &mut self,
        hybrid: Option<&bloomery_gpu::host::HostTier<bloomery_gpu::host::run::HostRun>>,
    ) {
        self.mark("steps", hybrid.and_then(record::census_of));
    }

    /// After the prompt call: its `tier census` record, then the load's `nvtier` record.
    pub fn prompt(
        &mut self,
        hybrid: Option<&bloomery_gpu::host::HostTier<bloomery_gpu::host::run::HostRun>>,
    ) {
        self.mark("prompt", hybrid.and_then(record::census_of));
        eprint_nvtier(hybrid.and_then(|h| h.nvme_tier()));
    }
}

/// The `nvtier` record ([`record::nvtier_of`]) of a server's load on stderr; nothing for a load that
/// attached no tier.
pub fn eprint_nvtier(tier: Option<&std::sync::Arc<bloomery_gpu::host::nvtier::NvTier>>) {
    record::require(record::groups::host_tier::ARENA, "host_tier::eprint_nvtier");
    if let Some(r) = record::nvtier_of(tier.map(|t| &**t)) {
        r.eprint();
    }
}
