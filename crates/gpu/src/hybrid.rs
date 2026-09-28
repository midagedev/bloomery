//! The hybrid MoE boundary's names as the engine's callers use them: the
//! host tier itself is [`crate::host`] ([`HostTier`], its step and batch
//! ports); [`Hybrid`] is the tier's name at the call sites that have not
//! moved to it. What stays here is V2-Lite's own lever: the prefix `[0, n_l)`
//! of every MoE layer on the card ([`HybridConfig`], [`SlotMap::prefix`]); a
//! V4.1 card holds its plan's `ExpertList` per layer — the id prefix or a hot
//! list's ranked ids.

use crate::GpuError;
use gguf::Split;
use std::sync::OnceLock;

pub use crate::host::batch::{BatchKey, BatchPort, ServeTimes};
pub use crate::host::page::{HandoffLayout, MAX_ROWS, PageError, PageLayout};
pub use crate::host::residency::HostResidency;
pub use crate::host::slots::{HOST, Slot, SlotMap, TIER};
pub use crate::host::step::{Boundary, BoundaryShape, Chain, HandoffTarget, RELEASE, StepPort};
pub use crate::host::{
    BEGIN_GROUP, HostExperts, HostTier, HybridStats, HybridWords, Poison, PoisonKind, PoisonMark,
    Refusal, name_refusal, refuse_expert_tiers,
};

/// The host tier under its call sites' name.
pub type Hybrid<H> = HostTier<H>;

// ----------------------------------------------------------------- levers

/// The V2-Lite hybrid load's lever, as read from the environment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Levers {
    /// `BLOOMERY_HYBRID_NL`: experts per MoE layer kept on the card; `None`
    /// when unset.
    pub n_l: Option<usize>,
}

/// The lever, read once per process (a `OnceLock`: the environment is read at
/// first use and never again).
pub fn levers() -> Result<Levers, GpuError> {
    static LEVERS: OnceLock<Result<Levers, String>> = OnceLock::new();
    LEVERS
        .get_or_init(read_levers)
        .clone()
        .map_err(|detail| GpuError::shape("hybrid::levers", detail))
}

fn read_levers() -> Result<Levers, String> {
    let n_l = match std::env::var("BLOOMERY_HYBRID_NL") {
        Ok(v) => Some(
            v.trim()
                .parse::<usize>()
                .map_err(|e| format!("BLOOMERY_HYBRID_NL={v:?}: {e}"))?,
        ),
        Err(std::env::VarError::NotPresent) => None,
        Err(e) => return Err(format!("BLOOMERY_HYBRID_NL: {e}")),
    };
    Ok(Levers { n_l })
}

/// A V2-Lite (deepseek2) hybrid load's parameter: experts `[0, n_l)` of
/// every MoE layer on the card and the rest on the host. A V4.1 load takes
/// its card set from the plan instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HybridConfig {
    pub n_l: usize,
}

impl HybridConfig {
    /// What the lever asks of `file`'s model: `None` when `BLOOMERY_HYBRID_NL`
    /// is unset or keeps every expert on the card — today's all-card path — and
    /// an error when it asks for more experts than a layer has.
    pub fn from_levers(file: &Split) -> Result<Option<HybridConfig>, GpuError> {
        let l = levers()?;
        let Some(n_l) = l.n_l else {
            return Ok(None);
        };
        let what = "HybridConfig::from_levers";
        let n_expert = file
            .arch_get_u64("expert_count")
            .ok_or(GpuError::metadata(what, "expert_count"))?;
        let n_expert = usize::try_from(n_expert)
            .map_err(|_| GpuError::shape(what, format!("expert_count {n_expert}")))?;
        if n_l > n_expert {
            return Err(GpuError::shape(
                what,
                format!("BLOOMERY_HYBRID_NL={n_l} keeps more experts than a layer's {n_expert}"),
            ));
        }
        Ok((n_l < n_expert).then_some(HybridConfig { n_l }))
    }
}
