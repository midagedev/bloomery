//! `gate_nvtier` — the NVMe expert tier's RAM arena on the real Qwen3.8
//! file: a load whose room forces the split dial to give the tier an arena
//! against the same load with every host expert resident, through the
//! clauses of the tier's design (nvtier R2):
//!
//! - `bits` (clause 1): both arms pin `BLOOMERY_CARD_BUDGET`, the context
//!   and `BLOOMERY_RESIDENCY` to one explicit word; 512 prompt positions
//!   (one prompt call) then 96 greedy steps — every greedy token equal and
//!   each step's logits row bit for bit the all-RAM arm's.
//! - `counters` (clause 2): misses > 0, fills > 0, evictions > 0, the
//!   arena's resident bytes at or under its budget at every step, and no
//!   buffered read on the box's direct-capable mount
//!   ([`NvTierStats::buffered_reads`]).
//! - `audit` (clause 3, `--audit`): just before compute, every listed
//!   NVMe-tier pick's slot is `memcmp`d against a fresh buffered read of
//!   the same file range at the six sampled offsets of
//!   [`audit_offsets`] — a torn or zeroed span refuses by name.
//! - `refuse` (clause 4): the room under the tier's floor
//!   (`HostRoomFloor`), the direct-read probe of a mount without direct IO,
//!   and an arena budget under one slot's span, each by name.
//! - `residency` (clause 5): the paged plan's unset rule resolves `mid`
//!   with no churn pool ([`residency38`]'s paged branch), every arena victim
//!   lands with no `host_serves` refusal, and `unresident` and `faulting`
//!   are 0 on the paged arms — bug catchers, structurally unreachable once
//!   the seam serves victims from the arena.
//!
//! PIN(2026-10-08): the page-cache design's two named refusals are dropped
//! — a paged tier under `BLOOMERY_HOST_LOCK=1` (the lock walk pins model
//! pages the tier no longer flushes) and an r8 resident copy under a paged
//! tier (`MADV_DONTNEED` cannot zero a copy the arena never advises) —
//! because the arena never advises the model mapping, which the
//! `advice` helper below holds: the arena's returned pages and the model
//! mapping's are disjoint ranges, and a mutant that madvises a model page
//! turns the check red.
//!
//! The FAIL-first mutants of each clause are named in the design's §7.6
//! table; this gate's body runs them at R2b, when the Qwen3.8 body's placed
//! load attaches the tier (its build site is the body's own).

use std::path::Path;

use bloomery_gpu_gates::{GateError, exit_with};
use gguf::Split;
use model::arch::qwen35moe::place::{Experts, PlanInputs};

/// The model the gates open (`BLOOMERY_REF_MODEL` under the qwen4exp
/// profile).
const MODEL: &str = refset::arch::qwen4exp::MODEL;
/// Prompt positions: one prompt call above the stream floor. The R2b arms
/// read this; the body's placed load attaches the tier then.
#[allow(dead_code, reason = "the R2b arms read it")]
const PROMPT: usize = 512;
/// Greedy steps after the prompt, the R2b arms'.
#[allow(dead_code, reason = "the R2b arms read it")]
const STEPS: usize = 96;
/// The paged arm's room, a 32 GB machine's.
const ROOM: u64 = 27 << 30;

fn main() -> std::process::ExitCode {
    exit_with("gate_nvtier", run())
}

fn run() -> Result<(), GateError> {
    let levers = bloomery_levers::at_main(&[
        bloomery_levers::HOST_POPULATE,
        bloomery_levers::HOST_LOCK,
        bloomery_levers::CARD_DONTNEED,
    ])?;
    let _ = levers.host();
    let audit = std::env::args().any(|a| a == "--audit");
    let path = Path::new(MODEL);
    let split = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut inputs = PlanInputs::describe(&split)?;
    // The paged arm's room, set as given (no environment between the arms);
    // the all-RAM arm plans on the machine's own reading, R2b's second load.
    inputs.room = (ROOM, model::placement::workstation::HostRead::Given);
    let machine = model::arch::qwen35moe::place::machine_for_experts(
        model::placement::workstation::RTX_3090,
        inputs.spec.layers.len(),
        bloomery_gpu::arch::qwen3moe::ubatch::ubatch_for(3072)? as u64,
        Experts::Card,
    );
    let plan = inputs.plan_with(
        &machine,
        3072,
        &model::placement::PlanLevers::default(),
        Experts::Card,
    )?;
    if plan.host.nvme_expert_bytes == 0 || plan.host.nvme_arena_bytes == 0 {
        return Err(format!(
            "the paged arm's plan pages {} B with an arena of {} B: the dial gave the room no \
             tier — check the floor and the room",
            plan.host.nvme_expert_bytes, plan.host.nvme_arena_bytes
        )
        .into());
    }
    let tier = bloomery_gpu::model::nvme_tier(&plan, &std::sync::Arc::new(split), false)?
        .ok_or("the paged plan built no arena")?;
    println!(
        "nvtier: arena {} B, {} paged experts, audit {}",
        tier.budget(),
        plan.host.nvme_expert_bytes,
        audit
    );
    // R2b: the placed load attaches the tier (the body's build site), then
    // the arms run and every clause above reads its own instrument. Until
    // then the gate proves the plan side and the tier's own build here.
    let stats = tier.stats();
    println!(
        "nvtier: counters at rest — misses {}, fills {}, evictions {}, resident {} B",
        stats.misses, stats.fills, stats.evictions, stats.resident_bytes
    );
    Ok(())
}

/// The offsets an audit reads of one part's span, the `--audit` arm's: the first and the last
/// page of it, the part's own first and last bytes, and one page between —
/// six `memcmp` sites a torn or zeroed slot cannot all pass.
#[allow(dead_code, reason = "the R2b audit arm reads it")]
fn audit_offsets(at: u64, len: usize) -> [usize; 6] {
    let page = engram::direct::DIRECT_ALIGN;
    let mid = (at as usize + len / 2) & !(page - 1);
    [
        0,
        page,
        len.saturating_sub(page),
        len / 2,
        mid.saturating_sub(at as usize),
        len.saturating_sub(1),
    ]
}

/// Whether the arena's own ranges — the spans its evictions return — and
/// the model mapping's ranges are disjoint: the arena never advises the
/// model mapping, so a lock's pinned pages and an r8 copy's pages stay
/// theirs. A mutant that madvises a model page names an arena range that
/// overlaps the mapping's, and this turns red.
#[allow(dead_code, reason = "the R2b advice clause reads it")]
fn advice_disjoint(arena: &[(usize, usize)], model: &[(usize, usize)]) -> bool {
    arena
        .iter()
        .all(|&(a0, a1)| model.iter().all(|&(m0, m1)| a1 <= m0 || m1 <= a0))
}

#[cfg(test)]
mod tests {
    use super::{advice_disjoint, audit_offsets};
    use bloomery_gpu::host::run::NvTier;
    use engram::direct::DIRECT_ALIGN;

    /// The audit's sites touch the span's two edge pages, its own ends and
    /// a middle page: whatever a torn fill leaves, one site reads it.
    #[test]
    fn audit_sites_cover_both_edge_pages_and_the_middle() {
        let page = DIRECT_ALIGN;
        for (at, len) in [(0, page), (7 * page + 13, 3 * page), (page + 1, 2 * page)] {
            let sites = audit_offsets(at as u64, len);
            assert_eq!(sites.len(), 6);
            assert!(sites.iter().all(|&s| s < len), "{at}+{len}: {sites:?}");
            assert!(sites.contains(&0) && sites.contains(&(len - 1)));
            // The first page and the last page each name a site.
            assert!(sites.iter().any(|&s| s < page), "{at}+{len}: {sites:?}");
            assert!(
                sites.iter().any(|&s| s + page >= len),
                "{at}+{len}: {sites:?}"
            );
        }
    }

    /// Disjoint arena and mapping ranges pass; an arena range that names a
    /// model page — the mutant — does not.
    #[test]
    fn the_arena_never_advises_the_model_mapping() {
        let model = [(0x1000_0000, 0x2000_0000), (0x3000_0000, 0x4000_0000)];
        assert!(advice_disjoint(
            &[(0x0000_1000, 0x1000_0000), (0x2000_0000, 0x3000_0000)],
            &model
        ));
        // The mutant: an eviction that returns a model page.
        assert!(!advice_disjoint(&[(0x1800_0000, 0x2800_0000)], &model));
        assert!(!advice_disjoint(&[(0x0, 0x9000_0000)], &model));
        // Touching edges are disjoint.
        assert!(advice_disjoint(&[(0x2000_0000, 0x3000_0000)], &model));
    }

    /// The tier's own name is reachable from the gate's crate set, so the
    /// R2b arms read its counters without a path change.
    #[test]
    fn the_tiers_stats_are_reachable() {
        fn _of(t: &NvTier) -> bloomery_gpu::host::run::NvTierStats {
            t.stats()
        }
        let _ = _of;
    }
}
