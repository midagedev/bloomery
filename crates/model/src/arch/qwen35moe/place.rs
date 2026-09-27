//! A qwen4exp file planned onto a machine: the one path from the file's
//! headers to a plan that keeps its invariants, as `glm5next::place` is for
//! GLM-5.3-Flash. The hyperparameters, every tensor's role, the typed
//! description and each layer's recurrent and cache bytes ([`KvLayout`]) are
//! read once ([`PlanInputs::read`]); a file with a feature the engine does not
//! run is refused there by the coverage check; then the placement
//! ([`PlanInputs::plan`]) keeps every routed expert on the host
//! ([`placement::plan_host_routed`]: no card expert kernel serves the file's
//! routed stacks yet) and the rest where its role says — the trunk, the
//! hyper-connections, the PLE site's projections, the router, the shared
//! expert and the head on the card, the PLE table in the file, its rows
//! gathered by the host.
//!
//! The attention layers select their positions by the mean-pool indexer at
//! every position the program runs, so no count of positions is refused for
//! the attention's sake; the one bound is the kernels' position argument
//! ([`KERNEL_POSITIONS`]). A card that cannot hold the cache at a context is
//! the plan's own violation ([`Violation::CardOver`]).

use gguf::Split;
use models::ModelSpec;

use super::hparams::{Hparams, Kind};
use super::{roles, spec};
use crate::arch::chat_of;
use crate::arch::coverage;
use crate::placement::workstation::{CONTEXT, CardSpec, GRANULE, MARGIN, SCRATCH, host};
use crate::placement::{
    self, Card, KvBytes, Machine, ModelTensors, PlacementError, Plan, PlanLevers, Unimplemented,
    Violation,
};

const F16_BYTES: u64 = 2;
const F32_BYTES: u64 = 4;

/// The widest call a later call may roll back into, in positions: the GDN
/// conv ring keeps `conv − 1 + PASS_ROWS` inputs and the PLE conv ring its
/// reach plus as many. The card body binds it to its kernels' own constant.
pub const PASS_ROWS: usize = 8;

/// The positions the selector's kernels take: the QSA passes take the
/// context as a `u32` launch argument and the selected flash reads `u32`
/// cache rows, so a context past this is one those launches cannot name.
pub const KERNEL_POSITIONS: u64 = u32::MAX as u64;

/// What a plan of a qwen4exp file is made from, read from its headers.
#[derive(Debug)]
pub struct PlanInputs {
    /// The file's hyperparameters.
    pub hp: Hparams,
    /// Every tensor with its role, and the model's layer and expert counts.
    pub model: ModelTensors,
    /// The typed model description the coverage check reads.
    pub spec: ModelSpec,
    /// Each layer's recurrent and cache bytes.
    pub kv: KvLayout,
}

/// Why a plan was refused.
#[derive(Debug, thiserror::Error)]
pub enum PlaceError {
    /// The placement could not be built.
    #[error(transparent)]
    Placement(#[from] PlacementError),
    /// The plan was built and breaks these invariants, every one of them.
    #[error("the plan breaks its invariants: {}", joined(.0))]
    Broken(Vec<Violation>),
    /// A context of no position, or of more than the kernels take.
    #[error(
        "ctx_max {ctx_max}: a context of 1 to {KERNEL_POSITIONS} positions (the QSA passes \
         take the context as a u32 launch argument, the selected flash reads u32 cache rows)"
    )]
    Positions { ctx_max: u64 },
}

/// The violations, `; `-separated.
fn joined(broken: &[Violation]) -> String {
    let list: Vec<String> = broken.iter().map(ToString::to_string).collect();
    list.join("; ")
}

impl PlanInputs {
    /// `split`'s hyperparameters, then its tensors' roles, then its
    /// description, the first that fails being the error; then the file is
    /// refused if it has a feature the engine does not run
    /// ([`PlacementError::Unimplemented`], every one listed).
    pub fn read(split: &Split) -> Result<PlanInputs, PlacementError> {
        let inputs = PlanInputs::describe(split)?;
        let missing = inputs.unimplemented();
        if missing.is_empty() {
            Ok(inputs)
        } else {
            Err(PlacementError::Unimplemented(missing))
        }
    }

    /// [`PlanInputs::read`] without the refusal of unimplemented features.
    pub fn describe(split: &Split) -> Result<PlanInputs, PlacementError> {
        let hp = Hparams::read(split)?;
        let model = roles::classify(split, &hp)?;
        let chat = chat_of(split, None, None)?;
        let spec = spec::spec_of(&hp, &model, chat)?;
        let kv = KvLayout::of(&hp);
        Ok(PlanInputs {
            hp,
            model,
            spec,
            kv,
        })
    }

    /// Every feature of the file the engine does not run, as
    /// [`coverage::check`] lists them. Empty for a file the program runs.
    pub fn unimplemented(&self) -> Vec<Unimplemented> {
        coverage::check(&self.spec, &self.model)
    }

    /// The placement of the file on `machine` at `ctx_max` positions under
    /// the placement's `levers`, every routed expert on the host; refused
    /// past [`KERNEL_POSITIONS`], when it cannot be built, or when it breaks
    /// an invariant.
    pub fn plan<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
    ) -> Result<Plan<'a>, PlaceError> {
        if ctx_max == 0 || ctx_max > KERNEL_POSITIONS {
            return Err(PlaceError::Positions { ctx_max });
        }
        let plan = placement::plan_host_routed(&self.model, machine, ctx_max, &self.kv, levers)?;
        let broken = plan.violations();
        if broken.is_empty() {
            Ok(plan)
        } else {
            Err(PlaceError::Broken(broken))
        }
    }
}

/// The machine a qwen4exp plan runs on: `card` runs every one of `layers`,
/// the head and the token embedding table whole (the file's q8_0 rows, which
/// the card gathers), with this workstation's context, scratch, margin and
/// host tier.
#[must_use]
pub fn machine(card: CardSpec, layers: usize) -> Machine {
    Machine {
        cards: vec![Card {
            name: card.name.to_string(),
            usable_bytes: card.usable_bytes(),
            context_bytes: CONTEXT,
            scratch_bytes: SCRATCH,
            margin_bytes: MARGIN,
            granule_bytes: GRANULE,
            layers: 0..layers,
            head: true,
            token_embedding: true,
        }],
        host: host(),
    }
}

/// One file's per-layer bytes beside the weights, all on the card:
/// - a GDN layer: its recurrent state (`v_heads` heads of `state × state`
///   f32) and its conv ring (`conv − 1 + PASS_ROWS` rows of the conv's
///   channels, `2·k_heads·state + v_heads·state`, in f32), both fixed by the
///   file;
/// - the PLE site's layer, beside that: the PLE conv ring
///   (`(taps − 1)·dilation + PASS_ROWS` rows of `streams · n_embd` f32);
/// - an attention layer: a position's K and V (`2 · kv_heads · head_dim` f16)
///   and its raw indexer key (`idx_dim` f16), and a pooled key (`idx_dim`
///   f16) per pool of `ratio` positions, the last pool counted whole.
#[derive(Clone, Debug)]
pub struct KvLayout {
    /// Per layer: its kind.
    kinds: Vec<Kind>,
    /// A GDN layer's state and conv ring, in bytes.
    recurrent: u64,
    /// The PLE site's layer and its conv ring's bytes.
    ple: Option<(usize, u64)>,
    /// An attention layer's bytes a position.
    row: u64,
    /// A pooled key's bytes (none on a qwen35moe file).
    pooled: Option<u64>,
    /// Per layer: its pool ratio (0 on a GDN layer or without a selector).
    ratios: Vec<u64>,
}

impl KvLayout {
    /// The layout `hp` describes.
    #[must_use]
    pub fn of(hp: &Hparams) -> KvLayout {
        let exp = hp.exp.as_ref();
        let raw_key = exp.map_or(0, |e| e.idx_dim as u64 * F16_BYTES);
        KvLayout {
            kinds: hp.kinds.clone(),
            recurrent: recurrent_bytes(hp.v_heads, hp.k_heads, hp.state, hp.conv),
            ple: exp.and_then(|e| e.ple).map(|p| {
                (
                    p.layer,
                    ple_ring_bytes(p.conv, p.ngram, streams(hp), hp.n_embd),
                )
            }),
            row: kv_row_bytes(hp.n_head_kv, hp.head_dim) + raw_key,
            pooled: exp.map(|e| e.idx_dim as u64 * F16_BYTES),
            ratios: (0..hp.n_layer)
                .map(|l| exp.map_or(0, |e| e.ratios[l] as u64))
                .collect(),
        }
    }
}

/// The hyper-connection streams of a qwen4exp file (1 without).
fn streams(hp: &Hparams) -> usize {
    hp.exp.as_ref().map_or(1, |e| e.hc_streams)
}

/// Rows of the GDN conv ring: the conv's reach and a pass.
#[must_use]
pub const fn conv_ring_rows(conv: usize) -> usize {
    conv - 1 + PASS_ROWS
}

/// Rows of the PLE conv ring: the reach of `taps` taps `dilation` apart and a
/// pass.
#[must_use]
pub const fn ple_ring_rows(taps: usize, dilation: usize) -> usize {
    (taps - 1) * dilation + PASS_ROWS
}

/// A GDN layer's state and conv ring over `v_heads` value heads and
/// `k_heads` key heads of `state` values with a `conv`-tap conv, in bytes.
fn recurrent_bytes(v_heads: usize, k_heads: usize, state: usize, conv: usize) -> u64 {
    let (v, k, d) = (v_heads as u64, k_heads as u64, state as u64);
    let channels = 2 * k * d + v * d;
    v * d * d * F32_BYTES + conv_ring_rows(conv) as u64 * channels * F32_BYTES
}

/// The PLE conv ring of `taps` taps `dilation` apart over `streams` streams
/// of `n_embd`, in bytes.
fn ple_ring_bytes(taps: usize, dilation: usize, streams: usize, n_embd: usize) -> u64 {
    ple_ring_rows(taps, dilation) as u64 * (streams * n_embd) as u64 * F32_BYTES
}

/// A GQA position's K and V in f16.
fn kv_row_bytes(kv_heads: usize, head_dim: usize) -> u64 {
    2 * (kv_heads * head_dim) as u64 * F16_BYTES
}

impl KvBytes for KvLayout {
    fn layer_bytes(&self, layer: usize, ctx_max: u64) -> u64 {
        let ple = match self.ple {
            Some((site, bytes)) if site == layer => bytes,
            _ => 0,
        };
        let own = match self.kinds.get(layer) {
            Some(Kind::DeltaRule) => self.recurrent,
            Some(Kind::Attention) => {
                let pooled = match (self.pooled, self.ratios.get(layer)) {
                    (Some(bytes), Some(&ratio)) if ratio > 0 => ctx_max.div_ceil(ratio) * bytes,
                    _ => 0,
                };
                ctx_max * self.row + pooled
            }
            None => 0,
        };
        own + ple
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Kind, KvBytes, KvLayout, conv_ring_rows, kv_row_bytes, ple_ring_bytes, ple_ring_rows,
        recurrent_bytes,
    };

    /// Qwen3.8's sizes: a GDN layer holds 48 heads of 128 × 128 f32 and
    /// eleven conv rows of 10,240 f32 channels; the PLE ring 17 rows of four
    /// 2,560-value streams in f32 (the card's `ple::ring_len(4)` values); an
    /// attention position 2 × 2 × 256 f16 of K and V and a 128-value raw
    /// indexer key, and a 128-value pooled key per four positions.
    #[test]
    fn qwen38_layer_bytes() {
        assert_eq!(conv_ring_rows(4), 11);
        assert_eq!(ple_ring_rows(4, 3), 17);
        assert_eq!(recurrent_bytes(48, 16, 128, 4), 3_145_728 + 11 * 10_240 * 4);
        assert_eq!(ple_ring_bytes(4, 3, 4, 2560), 17 * 10_240 * 4);
        assert_eq!(kv_row_bytes(2, 256), 2048);
        let kv = KvLayout {
            kinds: vec![Kind::DeltaRule, Kind::DeltaRule, Kind::Attention],
            recurrent: recurrent_bytes(48, 16, 128, 4),
            ple: Some((1, ple_ring_bytes(4, 3, 4, 2560))),
            row: kv_row_bytes(2, 256) + 256,
            pooled: Some(256),
            ratios: vec![0, 0, 4],
        };
        assert_eq!(kv.layer_bytes(0, 4096), 3_596_288);
        assert_eq!(kv.layer_bytes(1, 4096), 3_596_288 + 696_320);
        assert_eq!(kv.layer_bytes(2, 4096), 4096 * 2304 + 1024 * 256);
        assert_eq!(kv.layer_bytes(2, 4097), 4097 * 2304 + 1025 * 256);
        assert_eq!(kv.layer_bytes(3, 4096), 0);
    }
}
