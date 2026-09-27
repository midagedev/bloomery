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

pub use runtime::stores::{PASS_ROWS, conv_ring_rows, ple_ring_rows};
use runtime::stores::{kv_row_bytes, ple_ring_bytes, recurrent_bytes, selecting_bytes};

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
/// - an attention layer: a position's K and V (`2 · kv_heads · head_dim` f16);
///   with the selector, also its raw indexer key (`idx_dim` f16) and a
///   pooled key (`idx_dim` f16) per pool of `ratio` positions, the last pool
///   counted whole (`runtime::stores::selecting_bytes`).
#[derive(Clone, Debug)]
pub struct KvLayout {
    /// Per layer: its kind.
    kinds: Vec<Kind>,
    /// A GDN layer's state and conv ring, in bytes.
    recurrent: u64,
    /// The PLE site's layer and its conv ring's bytes.
    ple: Option<(usize, u64)>,
    /// An attention layer's K and V heads and their width.
    kv_heads: usize,
    head_dim: usize,
    /// The selector's key width and each layer's pool (0 on a GDN layer);
    /// `None` on a qwen35moe file.
    select: Option<(usize, Vec<usize>)>,
}

impl KvLayout {
    /// The layout `hp` describes.
    #[must_use]
    pub fn of(hp: &Hparams) -> KvLayout {
        let exp = hp.exp.as_ref();
        KvLayout {
            kinds: hp.kinds.clone(),
            recurrent: recurrent_bytes(hp.v_heads, hp.k_heads, hp.state, hp.conv),
            ple: exp.and_then(|e| e.ple).map(|p| {
                (
                    p.layer,
                    ple_ring_bytes(p.conv, p.ngram, streams(hp), hp.n_embd),
                )
            }),
            kv_heads: hp.n_head_kv,
            head_dim: hp.head_dim,
            select: exp.map(|e| (e.idx_dim, e.ratios.clone())),
        }
    }
}

/// The hyper-connection streams of a qwen4exp file (1 without).
fn streams(hp: &Hparams) -> usize {
    hp.exp.as_ref().map_or(1, |e| e.hc_streams)
}

impl KvBytes for KvLayout {
    fn layer_bytes(&self, layer: usize, ctx_max: u64) -> u64 {
        let ple = match self.ple {
            Some((site, bytes)) if site == layer => bytes,
            _ => 0,
        };
        let own = match self.kinds.get(layer) {
            Some(Kind::DeltaRule) => self.recurrent,
            Some(Kind::Attention) => match &self.select {
                Some((idx_dim, pools)) => {
                    let pool = pools.get(layer).copied().filter(|&p| p > 0).expect(
                        "KvLayout: an attention layer's pool (Hparams::read refuses 0 there)",
                    );
                    let ctx = usize::try_from(ctx_max).expect("KvLayout: a context of usize");
                    selecting_bytes(self.kv_heads, self.head_dim, *idx_dim, pool, ctx)
                }
                None => ctx_max * kv_row_bytes(self.kv_heads, self.head_dim),
            },
            None => 0,
        };
        own + ple
    }
}

#[cfg(test)]
mod tests {
    use super::{Kind, KvBytes, KvLayout, ple_ring_bytes, recurrent_bytes};

    /// Qwen3.8's layers by kind (the sizes themselves are
    /// `runtime::stores`'s, pinned there): a GDN layer's state and conv ring,
    /// the PLE ring beside it on its layer, an attention layer's K/V, raw and
    /// pooled keys at a context of whole pools and one past.
    #[test]
    fn qwen38_layer_bytes() {
        let kv = KvLayout {
            kinds: vec![Kind::DeltaRule, Kind::DeltaRule, Kind::Attention],
            recurrent: recurrent_bytes(48, 16, 128, 4),
            ple: Some((1, ple_ring_bytes(4, 3, 4, 2560))),
            kv_heads: 2,
            head_dim: 256,
            select: Some((128, vec![0, 0, 4])),
        };
        assert_eq!(kv.layer_bytes(0, 4096), 3_596_288);
        assert_eq!(kv.layer_bytes(1, 4096), 3_596_288 + 696_320);
        assert_eq!(kv.layer_bytes(2, 4096), 4096 * 2304 + 1024 * 256);
        assert_eq!(kv.layer_bytes(2, 4097), 4097 * 2304 + 1025 * 256);
        assert_eq!(kv.layer_bytes(3, 4096), 0);
    }
}
