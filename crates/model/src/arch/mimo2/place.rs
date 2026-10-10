//! A mimo2 file planned onto a machine: the one path from the file's headers
//! to a plan that keeps its invariants, as `glm5next::place` is for GLM. The
//! hyperparameters, every tensor's role, the typed description and each
//! layer's KV bytes ([`KvLayout`]) are read once ([`PlanInputs::read`], which
//! refuses a file whose next-token layers carry tensors — the draft that
//! reads them is a later round's — and one with a feature no program runs,
//! every one listed); then the placement ([`PlanInputs::plan`]) keeps every
//! routed expert on the host ([`placement::plan_host_routed`]: this program
//! runs no card expert), every layer's KV planes full whatever its window,
//! and refuses by name a machine that hangs an expert tier card and a
//! context of no position.

use std::num::NonZeroUsize;

use bloomery_placement::slots::{SeqTerms, Stores};
use gguf::Split;
use models::ModelSpec;

use super::hparams::Hparams;
use super::{roles, spec};
use crate::arch::chat_of;
use crate::arch::coverage;
use crate::placement::{
    self, KvBytes, Machine, ModelTensors, PlacementError, Plan, PlanLevers, Unimplemented,
    Violation, checked, joined,
};

const F16_BYTES: u64 = 2;

/// What a plan of a mimo2 file is made from, read from its headers.
#[derive(Debug)]
pub struct PlanInputs {
    /// The file's hyperparameters.
    pub hp: Hparams,
    /// Every tensor with its role, and the model's layer and expert counts.
    pub model: ModelTensors,
    /// The typed model description the coverage check reads.
    pub spec: ModelSpec,
    /// Each trunk layer's KV bytes.
    pub kv: KvLayout,
}

/// Why a plan was refused.
#[derive(Debug, thiserror::Error)]
pub enum PlaceError {
    /// The placement could not be built — a machine that hangs an expert
    /// tier card among its refusals ([`PlacementError::HostRoutedTier`]):
    /// this plan keeps every routed expert on the host, so the tier would
    /// hold nothing.
    #[error(transparent)]
    Placement(#[from] PlacementError),
    /// The plan was built and breaks these invariants, every one of them.
    #[error("the plan breaks its invariants: {}", joined(.0))]
    Broken(Vec<Violation>),
    /// A context of no position: no plane, nothing to serve.
    #[error("ctx_max 0: a plan serves at least one position")]
    CtxZero,
}

impl PlanInputs {
    /// `split`'s hyperparameters, then its tensors' roles, then its
    /// description, the first that fails being the error; then a file whose
    /// next-token layers carry tensors is refused by the next-token count's
    /// name (the draft that reads them is a later round's; this plan loads
    /// the trunk without them), and one with a feature the engine does not
    /// run ([`PlacementError::Unimplemented`], every one listed).
    pub fn read(split: &Split) -> Result<PlanInputs, PlacementError> {
        let inputs = PlanInputs::describe(split)?;
        if inputs.hp.n_layer > inputs.hp.n_trunk {
            return Err(PlacementError::Metadata {
                key: "nextn_predict_layers".to_string(),
                detail: format!(
                    "is {}; the draft that reads the next-token layers is a later round's, and \
                     this plan loads the trunk without them",
                    inputs.hp.n_layer - inputs.hp.n_trunk
                ),
            });
        }
        let missing = inputs.unimplemented();
        if missing.is_empty() {
            Ok(inputs)
        } else {
            Err(PlacementError::Unimplemented(missing))
        }
    }

    /// [`PlanInputs::read`] without the refusals: the file as it is, for a
    /// gate that plans or describes a file whatever the coverage list holds.
    pub fn describe(split: &Split) -> Result<PlanInputs, PlacementError> {
        let hp = Hparams::read(split)?;
        let model = roles::classify(split, &hp)?;
        let chat = chat_of(split, spec::TOOLS, Some(models::ReasoningFormat::ThinkSpan))?;
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
    /// the placement's `levers`, every routed expert on the host and none on
    /// a card; refused by name a context of no position, and as the
    /// placement refuses a machine that hangs an expert tier card (it would
    /// hold nothing) and a context whose KV planes do not fit the card.
    pub fn plan<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
    ) -> Result<Plan<'a>, PlaceError> {
        self.plan_with_slots(machine, ctx_max, levers, NonZeroUsize::MIN)
    }

    /// The placement of a load that serves `slots` resident sequences of
    /// `ctx_max` positions each: every layer's KV planes counted `slots`
    /// times on the card ([`SeqTerms::slots_of`] over
    /// [`PlanInputs::seq_terms`]), refused as [`PlanInputs::plan`] refuses.
    /// A sequence holds nothing beside its stores, so the card's kv class is
    /// `slots` sequences' stores.
    pub fn plan_with_slots<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
        slots: NonZeroUsize,
    ) -> Result<Plan<'a>, PlaceError> {
        if ctx_max == 0 {
            return Err(PlaceError::CtxZero);
        }
        let kv = self.seq_terms().slots_of(slots.get() as u64);
        let plan = placement::plan_host_routed(&self.model, machine, ctx_max, &kv, levers)?;
        checked(plan).map_err(PlaceError::Broken)
    }

    /// What one resident sequence of a load of the file holds on its card
    /// ([`SeqTerms`]): its stores over the trunk's layers ([`KvLayout`]), no
    /// draft and nothing beside them — the body's sequence is its layers'
    /// KV planes alone.
    #[must_use]
    pub fn seq_terms(&self) -> SeqTerms<'_> {
        SeqTerms {
            layers: Stores {
                kv: &self.kv,
                count: self.hp.n_trunk,
            },
            draft: None,
            beside: 0,
        }
    }
}

/// One file's per-layer KV bytes: every trunk layer, whatever its window
/// kind, holds a plane of every position — its KV-head count's key and value
/// rows in f16 — for the attention design keeps full planes with no ring; a
/// layer's window bounds what a query reads, not what the layer holds.
#[derive(Clone, Debug)]
pub struct KvLayout {
    /// Per trunk layer: its KV-head count (`attention.head_count_kv`).
    kv_heads: Vec<usize>,
    /// One KV head's key and value rows a position, in bytes: the key and
    /// value head widths (`attention.key_length`,
    /// `attention.value_length`) in f16.
    row: u64,
}

impl KvLayout {
    /// The layout `hp` describes.
    #[must_use]
    pub fn of(hp: &Hparams) -> KvLayout {
        KvLayout {
            kv_heads: hp.kv_heads[..hp.n_trunk].to_vec(),
            row: (hp.head_k + hp.head_v) as u64 * F16_BYTES,
        }
    }

    /// The bytes of layers `layers` at `ctx_max` positions: what a card that
    /// runs them holds beside its weights ([`KvBytes::layer_bytes`] summed).
    #[must_use]
    pub fn bytes(&self, layers: std::ops::Range<usize>, ctx_max: u64) -> u64 {
        layers.map(|l| self.layer_bytes(l, ctx_max)).sum()
    }
}

impl KvBytes for KvLayout {
    fn layer_bytes(&self, layer: usize, ctx_max: u64) -> u64 {
        self.kv_heads
            .get(layer)
            .map_or(0, |&kv| ctx_max * kv as u64 * self.row)
    }
}

#[cfg(test)]
mod tests {
    use gguf::Split;

    use super::super::hparams::tests::{keys, tensors};
    use super::{F16_BYTES, KvBytes, KvLayout, PlanInputs, SeqTerms, Stores};
    use crate::arch::synthetic::{V, header_shaped};

    /// A layer's plane: its own KV-head count's key and value rows in f16 a
    /// position — a window layer's plane is as full as a full layer's, and at
    /// 9 positions, past the small header's window of 8, a ring would already
    /// differ — and a layer past the trunk nothing. The widths are the
    /// `hparams` tests' small header's: key 16 and value 8 values, 2 KV heads
    /// on a full layer and 1 on a window one.
    #[test]
    fn layer_bytes() {
        let kv = KvLayout {
            kv_heads: vec![2, 1, 1, 2],
            row: (16 + 8) as u64 * F16_BYTES,
        };
        assert_eq!(kv.row, 48);
        assert_eq!(kv.layer_bytes(0, 9), 9 * 2 * 48);
        assert_eq!(kv.layer_bytes(1, 9), 9 * 48);
        assert_eq!(kv.layer_bytes(4, 9), 0);
        assert_eq!(kv.bytes(0..4, 1), 6 * 48);
        assert_eq!(kv.bytes(0..4, 9), 9 * 6 * 48);
    }

    /// One sequence's terms are the layout's stores over the trunk with
    /// nothing beside them, and `slots` sequences count each layer `slots`
    /// times — the card's kv class of a load that serves them.
    #[test]
    fn seq_terms_count_every_sequence() {
        let kv = KvLayout {
            kv_heads: vec![2, 1, 1, 2],
            row: (16 + 8) as u64 * F16_BYTES,
        };
        let terms = SeqTerms {
            layers: Stores { kv: &kv, count: 4 },
            draft: None,
            beside: 0,
        };
        assert_eq!(terms.bytes(9, 1), kv.bytes(0..4, 9));
        assert_eq!(terms.bytes(9, 3), 3 * kv.bytes(0..4, 9));
        assert_eq!(terms.plan_kv(9, 3), 3 * kv.bytes(0..4, 9));
        let three = terms.slots_of(3);
        for l in 0..4 {
            assert_eq!(three.layer_bytes(l, 9), 3 * kv.layer_bytes(l, 9));
        }
        assert_eq!(three.layer_bytes(4, 9), 0);
    }

    /// A file whose next-token layers carry tensors is refused at
    /// [`PlanInputs::read`] by the next-token count's name — the draft that
    /// reads them is a later round's, and this plan loads the trunk without
    /// them — before the coverage list; the same file without its next-token
    /// layer is refused by that list alone (the synthetic head width is one no
    /// row of the mimo2 body runs), so the count is the named refusal.
    #[test]
    fn next_token_layers_are_refused_at_read() {
        let err = read_inputs("mimo2-place-nextn", &keys(), &tensors())
            .expect_err("a file with a next-token layer is refused");
        assert!(
            err.contains("metadata nextn_predict_layers: is 1")
                && err.contains("the draft that reads the next-token layers is a later round's"),
            "{err}"
        );
        let tensors: Vec<(String, Vec<u64>)> = tensors()
            .into_iter()
            .filter(|(n, _)| !n.starts_with("blk.4."))
            .collect();
        let err = read_inputs("mimo2-place-trunk", &trunk_keys(), &tensors)
            .expect_err("a trunk-only file is refused by the coverage list");
        assert!(
            err.contains("feature(s) of this file are not implemented"),
            "{err}"
        );
    }

    /// The small header of the `hparams` tests' keys with its next-token
    /// layer taken out: four trunk blocks, the per-layer arrays four long.
    fn trunk_keys() -> Vec<(&'static str, V)> {
        keys()
            .into_iter()
            .map(|(k, v)| match k {
                "block_count" => ("block_count", V::U32(4)),
                "nextn_predict_layers" => ("nextn_predict_layers", V::U32(0)),
                "attention.head_count_kv" => ("attention.head_count_kv", V::I32s(vec![2, 1, 1, 2])),
                "attention.sliding_window_pattern" => (
                    "attention.sliding_window_pattern",
                    V::I32s(vec![0, 1, 1, 0]),
                ),
                _ => (k, v),
            })
            .collect()
    }

    /// [`PlanInputs::read`] of a synthetic header, or its error's text.
    fn read_inputs(
        tag: &str,
        kv: &[(&'static str, V)],
        tensors: &[(String, Vec<u64>)],
    ) -> Result<PlanInputs, String> {
        let global = [
            ("tokenizer.ggml.pre", V::Str("qwen2")),
            ("tokenizer.chat_template", V::Str("{{ messages }}")),
        ];
        let path = header_shaped(tag, "mimo2", kv, &global, tensors);
        let split = Split::open(&path).expect("the synthetic header opens");
        let read = PlanInputs::read(&split).map_err(|e| e.to_string());
        let _ = std::fs::remove_file(&path);
        read
    }
}
