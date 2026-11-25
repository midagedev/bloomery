//! What the two-card tier gates share (`gate_glm5next_twocard`, `gate_solar2_twocard`, `gate_mistrallarge4_twocard`):
//! plan (b′) against the same plan with every tier segment on the stage card, the legs' reads, and the per-layer
//! counters of what the tier was sent.
//! The reference of a two-card run is the stage card opened alone over the union of its experts and the tier's
//! ([`model::placement::union_plan`]): the tier's slots computed by the stage card's own launches, so the two-card run must read the same
//! tokens and logits rows bit for bit ([`same`]). A gate includes it by `#[path]`, with `#[allow(dead_code)]` for the
//! helpers it does not read.

use std::time::Instant;

use bloomery_gpu::host::batch::TierBatchStats;
use bloomery_gpu::host::run::HostRun;
use bloomery_gpu::hybrid::Hybrid;
use bloomery_gpu_gates::GateError;

/// Every token, kept count and logits row a leg read, in order.
#[derive(Default)]
pub struct Leg {
    pub tokens: Vec<u32>,
    pub logits: Vec<Vec<f32>>,
    pub kept: Vec<usize>,
}

/// What a span of a run sent the tier: its hits per layer from the
/// tier's first layer, and the batch services it enqueued.
pub struct Sent {
    pub first: usize,
    pub hits: Vec<u64>,
    pub stats: TierBatchStats,
}

/// The tier's per-layer hits and batch counters of `hybrid` since load;
/// `None` without a tier.
pub fn tier_counts(hybrid: &Hybrid<HostRun>) -> Option<Sent> {
    hybrid.tiers().first().map(|t| Sent {
        first: t.set().layers().start,
        hits: t.stats().layer_hits,
        stats: hybrid
            .tier_batch_stats()
            .first()
            .copied()
            .unwrap_or_default(),
    })
}

/// The counters `after` less `before`: what the span between sent.
pub fn sent_between(before: Option<Sent>, after: Option<Sent>) -> Option<Sent> {
    let (b, a) = (before?, after?);
    Some(Sent {
        first: a.first,
        hits: a.hits.iter().zip(&b.hits).map(|(a, b)| a - b).collect(),
        stats: TierBatchStats {
            served: a.stats.served - b.stats.served,
            settles: a.stats.settles - b.stats.settles,
            settle_early: a.stats.settle_early - b.stats.settle_early,
            settle_ns: a.stats.settle_ns - b.stats.settle_ns,
            skipped: a.stats.skipped - b.stats.skipped,
        },
    })
}

/// The counters of two spans together.
pub fn plus(a: Option<Sent>, b: Option<Sent>) -> Option<Sent> {
    let (a, b) = (a?, b?);
    Some(Sent {
        first: a.first,
        hits: a.hits.iter().zip(&b.hits).map(|(a, b)| a + b).collect(),
        stats: TierBatchStats {
            served: a.stats.served + b.stats.served,
            settles: a.stats.settles + b.stats.settles,
            settle_early: a.stats.settle_early + b.stats.settle_early,
            settle_ns: a.stats.settle_ns + b.stats.settle_ns,
            skipped: a.stats.skipped + b.stats.skipped,
        },
    })
}

/// Refused unless `hybrid`'s tier `Gpu` is the expert tier card of plan (b′), the second card in
/// view by usable bytes ([`bloomery_gpu_gates::tier::opened_on_rank`]); prints the load line,
/// `what` its name, `t0` the load's start and `stage` the stage card's name and resident bytes.
pub fn tier_on_card(
    what: &str,
    t0: Instant,
    stage: (&str, usize),
    hybrid: &Hybrid<HostRun>,
) -> Result<(), GateError> {
    let tier = hybrid
        .tiers()
        .first()
        .ok_or("the (b′) load holds no tier card")?;
    let tier_gpu = tier.gpu()?;
    bloomery_gpu_gates::tier::opened_on_rank("tier", tier_gpu, 1)?;
    println!(
        "{what} in {:.1} s: stage {} ({} B), tier {} experts on {} ({} B)",
        t0.elapsed().as_secs_f64(),
        stage.0,
        stage.1,
        tier.set().experts(),
        tier_gpu.device_name()?,
        tier.resident_bytes(),
    );
    Ok(())
}

/// Refused unless `hybrid`'s tier is the one expert tier of plan (b′) on a node (`BLOOMERY_TIER_NODES`), a
/// card of the node's census; prints the load line as [`tier_on_card`] does, `what` its name, `t0` the load's
/// start and `stage` the stage card's name and resident bytes.
pub fn tier_on_node(
    what: &str,
    t0: Instant,
    stage: (&str, usize),
    hybrid: &Hybrid<HostRun>,
) -> Result<(), GateError> {
    let tier = hybrid
        .tiers()
        .first()
        .ok_or("the (b′) load holds no tier card")?;
    let (addr, device) = tier
        .node()
        .zip(tier.node_device())
        .ok_or("the load's tier is a card of this process, not a node's")?;
    println!(
        "{what} in {:.1} s: stage {} ({} B), tier {} experts on node {addr}'s {} ({} B)",
        t0.elapsed().as_secs_f64(),
        stage.0,
        stage.1,
        tier.set().experts(),
        device.name,
        tier.resident_bytes(),
    );
    Ok(())
}

/// Whether leg `got` is `want` bit for bit; prints the first difference.
pub fn same(what: &str, want: &Leg, got: &Leg) -> bool {
    if want.tokens != got.tokens || want.kept != got.kept {
        let at = want
            .tokens
            .iter()
            .zip(&got.tokens)
            .position(|(a, b)| a != b);
        println!(
            "FAIL {what}: tokens apart at {at:?} ({} and {} tokens), kept {:?} against the \
             reference's {:?}",
            got.tokens.len(),
            want.tokens.len(),
            got.kept,
            want.kept
        );
        return false;
    }
    if want.logits.len() != got.logits.len() {
        println!(
            "FAIL {what}: {} logits rows, the reference read {}",
            got.logits.len(),
            want.logits.len()
        );
        return false;
    }
    for (i, (w, g)) in want.logits.iter().zip(&got.logits).enumerate() {
        let first = w
            .iter()
            .zip(g)
            .position(|(a, b)| a.to_bits() != b.to_bits());
        if w.len() != g.len() || first.is_some() {
            println!("FAIL {what}: logits row {i}: first logit apart {first:?}");
            return false;
        }
    }
    println!(
        "ok {what}: {} tokens, {} logits rows, {} passes' kept counts bit for bit the \
         reference's",
        got.tokens.len(),
        got.logits.len(),
        got.kept.len()
    );
    true
}

/// The precondition of leg `what`: it alone sent every layer of
/// `tier_layers` at least one routed slot, and, when `batch`, the tier
/// served its batch; prints its counters.
pub fn sent_every(what: &str, sent: Option<&Sent>, tier_layers: &[usize], batch: bool) -> bool {
    let Some(sent) = sent else {
        println!("FAIL precondition {what}: the load holds no tier card");
        return false;
    };
    let hit = |l: usize| {
        l.checked_sub(sent.first)
            .and_then(|i| sent.hits.get(i))
            .copied()
            .unwrap_or(0)
    };
    let missing: Vec<usize> = tier_layers
        .iter()
        .copied()
        .filter(|&l| hit(l) == 0)
        .collect();
    let st = sent.stats;
    println!(
        "{what} tier: {} batch services, {} settles ({} early, {:.2} ms waited), hits per \
         layer {:?}",
        st.served,
        st.settles,
        st.settle_early,
        st.settle_ns as f64 / 1e6,
        tier_layers.iter().map(|&l| hit(l)).collect::<Vec<_>>()
    );
    let served = !batch || st.served > 0;
    if missing.is_empty() && served {
        println!(
            "ok precondition {what}: every tier layer was sent a routed slot{}",
            if batch {
                ", the tier served the batch"
            } else {
                ""
            }
        );
        true
    } else {
        println!(
            "FAIL precondition {what}: tier layers {missing:?} were sent no routed slot, {} \
             batch services; the leg proves nothing there",
            st.served
        );
        false
    }
}
