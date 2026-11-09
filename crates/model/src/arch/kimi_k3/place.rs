//! A Kimi K3 file planned onto a machine: the common hybrid owner's plan ([`place_hybrid`]) over
//! this family's reading. What each layer holds is read off the description (a KDA layer's state
//! and conv ring, a latent layer's plane of cached rows), and the routed experts' bytes are the
//! file's by their stacks' types. The plan serves the placed load: an expert tier card beside the
//! stage card ([`tier_places_bytes`] of card buffers on the stage, [`tier_batch`] reserved on the
//! tier card and the host) and the common host run's NVMe arena for the experts the host's room
//! cannot hold ([`PlanInputs::with_nvme_tier`]). The step's plan keeps no prompt-pass bytes
//! ([`PlanInputs::prompt_reserve`] is 0) and the tier's run and page are one token
//! ([`TIER_CHUNK`], [`TIER_ROWS`]). A load that runs the placed prompt pass plans its rows apart:
//! the card driver states the pass's bytes, its tier places among them, as the prompt reserve,
//! and the tier's batch at the pass's rows ([`tier_batch_at`]). The body runs every layer of the
//! description ([`Program::KimiK3Body`]); what is left to the coverage check is the template's
//! tool-call parser, the chat surface's, so [`PlanInputs::read`] refuses that one item and a
//! driver that feeds ids and renders no chat reads the file by [`PlanInputs::read_without_chat`].
//!
//! A K3.1 file plans with the indexer's key row beside each latent layer's cached row (the
//! description's selector, [`place_hybrid::KvLayout`]), at any context the stores fit: the body's
//! latent layers attend the rows their indexer picks past its top-k (`gpu/src/arch/kimi_k3/index.rs`),
//! and up to it the selection is every position, whose result is the dense latent attention. A
//! driver that opens the file with a context reads it by [`PlanInputs::read_without_chat_for`].

use gguf::Split;
use models::Need;

use super::hparams::Hparams;
use super::spec;
use crate::arch::place_hybrid;
use crate::placement::kernels;
use crate::placement::workstation::{self, TierBatchBytes};
use crate::placement::{PlacementError, Unimplemented};

pub use place_hybrid::{KvLayout, PlaceError};

/// What a plan of a Kimi K3 file is made from, read from its headers.
pub type PlanInputs = place_hybrid::PlanInputs<Hparams>;

/// The engine the plan's refusals name.
const ENGINE: &str = "Kimi K3";

/// Tokens one run of the tier's card leg covers: the body walks a prompt one id a step, so a
/// run is the step's one token.
pub const TIER_CHUNK: u64 = 1;

/// Rows of the tier's page and of the host tier's batch port: the step's one.
pub const TIER_ROWS: u64 = 1;

/// The stage card's bytes a tiered load makes beside the plan's terms: the slot map's tier view
/// (a place per expert per layer) and the step's tier places (`n_used` for each of
/// [`TIER_ROWS`] rows), four bytes each.
#[must_use]
pub fn tier_places_bytes(hp: &Hparams) -> u64 {
    4 * (hp.n_layer * hp.experts.n_expert + TIER_ROWS as usize * hp.experts.n_used) as u64
}

/// The expert tier's batch bytes ([`TierBatchBytes`]) the machine reserves on the tier card and
/// the host: the experts run in the latent, so the tier's rows and its staging are the latent's
/// width, through the card leg in runs of [`TIER_CHUNK`] over [`TIER_ROWS`] tokens. Refused by
/// name: a geometry the common leg's bytes do not count.
pub fn tier_batch(hp: &Hparams) -> Result<TierBatchBytes, PlacementError> {
    tier_batch_at(hp, TIER_ROWS)
}

/// [`tier_batch`] for a load whose tier batch holds `rows` rows, a prompt pass's, run through the
/// card leg in runs of [`TIER_CHUNK`].
pub fn tier_batch_at(hp: &Hparams, rows: u64) -> Result<TierBatchBytes, PlacementError> {
    workstation::tier_leg_batch_bytes(
        hp.latent as u64,
        hp.experts.ff as u64,
        hp.experts.n_used as u64,
        TIER_CHUNK,
        rows,
    )
}

impl PlanInputs {
    /// [`PlanInputs::describe`], then the file refused with every feature the engine does not run
    /// ([`PlacementError::Unimplemented`]): the template's tool-call parser.
    pub fn read(split: &Split) -> Result<PlanInputs, PlacementError> {
        PlanInputs::describe(split)?.runnable()
    }

    /// [`PlanInputs::read`] for a driver that renders no chat: the file refused with every feature
    /// the engine does not run but the tool-call parser of its template, which only a request with
    /// tools reads.
    pub fn read_without_chat(split: &Split) -> Result<PlanInputs, PlacementError> {
        let inputs = PlanInputs::describe(split)?;
        let parser = Need::ToolParser.to_string();
        let missing: Vec<Unimplemented> = inputs
            .unimplemented()
            .into_iter()
            .filter(|item| item.feature != parser)
            .collect();
        if missing.is_empty() {
            Ok(inputs)
        } else {
            Err(PlacementError::Unimplemented(missing))
        }
    }

    /// [`PlanInputs::read_without_chat`] for a driver that opens the file with a context of `ctx`
    /// positions, which [`PlanInputs::check_ctx`] takes whatever its length.
    pub fn read_without_chat_for(split: &Split, ctx: u64) -> Result<PlanInputs, PlacementError> {
        let inputs = PlanInputs::read_without_chat(split)?;
        inputs.check_ctx(ctx)?;
        Ok(inputs)
    }

    /// A context of `ctx` positions as [`Hparams::check_ctx`] takes it: every context, a K3.1 file's
    /// past its indexer's top-k included. A driver that opens the file calls it beside
    /// [`PlanInputs::plan`].
    pub fn check_ctx(&self, ctx: u64) -> Result<(), PlacementError> {
        self.hp.check_ctx(ctx)
    }

    /// The file as it is, for a gate that describes it whatever the coverage list holds, and the
    /// host's room and the arena's given bytes. The roles and the description are the registry's
    /// reader's ([`spec::read`]), so the plan and the description of a file are one reading.
    /// Refused by name, before any load: a dense site whose q8_1 input column is wider than the
    /// card's activation holds ([`kernels::check_q8_act_width`]).
    pub fn describe(split: &Split) -> Result<PlanInputs, PlacementError> {
        let hp = Hparams::read(split)?;
        let read = spec::read(split)?;
        read.tensors
            .tensors
            .iter()
            .try_for_each(kernels::check_q8_act_width)?;
        let tier = tier_places_bytes(&hp);
        let rooms = hp.pos_layout();
        Ok(PlanInputs::new(ENGINE, hp, read.tensors, read.spec)?
            .with_rooms(rooms)
            .with_tier_reserve(tier)
            .with_nvme_tier())
    }
}

#[cfg(test)]
mod tests {
    use gguf::Split;
    use models::Need;

    use super::super::hparams::tests::{N_EXP, TOP_K, keys, keys31, tensors, tensors31, typed};
    use super::{PlaceError, PlanInputs};
    use crate::arch::synthetic::{Ty, V, header_typed};
    use crate::placement::workstation::{self, A6000, RTX_3090};
    use crate::placement::{KvBytes, PlacementError, PlanLevers};

    /// The small header's file with `over` typed as the released set's other stacks, opened; each
    /// test names its own file (the tests run in threads of one process).
    fn opened(name: &str, over: &[(&str, Ty)]) -> Split {
        let global = [
            ("tokenizer.ggml.pre", V::Str("kimi-k2")),
            ("tokenizer.chat_template", V::Str("{{ messages }}")),
        ];
        let path = header_typed(name, "kimi-k3", &keys(), &global, &typed(&tensors(), over));
        let split = Split::open(&path).expect("the synthetic header opens");
        let _ = std::fs::remove_file(&path);
        split
    }

    /// The small K3.1 header's file, opened.
    fn opened31(name: &str) -> Split {
        opened31_at(name, TOP_K as u32, N_EXP as u32)
    }

    /// The small K3.1 header's file with the indexer's `top_k` and `experts` routed experts,
    /// opened: 8 experts with the 2 kept are the router instance the body is built for, so that
    /// file's coverage list is the template's parser alone.
    fn opened31_at(name: &str, top_k: u32, experts: u32) -> Split {
        let global = [
            ("tokenizer.ggml.pre", V::Str("kimi-k2")),
            ("tokenizer.chat_template", V::Str("{{ messages }}")),
        ];
        let kv: Vec<_> = keys31()
            .into_iter()
            .map(|(k, v)| match k {
                "attention.indexer.top_k" => (k, V::U32(top_k)),
                "expert_count" => (k, V::U32(experts)),
                _ => (k, v),
            })
            .collect();
        let stacks: Vec<(String, Vec<u64>)> = tensors31()
            .into_iter()
            .map(|(n, mut dims)| {
                if n.ends_with("_exps.weight") {
                    dims[2] = u64::from(experts);
                } else if n.ends_with("ffn_gate_inp.weight") {
                    dims[1] = u64::from(experts);
                } else if n.ends_with("exp_probs_b.bias") {
                    dims[0] = u64::from(experts);
                }
                (n, dims)
            })
            .collect();
        let path = header_typed(name, "kimi-k31", &kv, &global, &typed(&stacks, &[]));
        let split = Split::open(&path).expect("the synthetic header opens");
        let _ = std::fs::remove_file(&path);
        split
    }

    /// The small header's file, described.
    fn described(name: &str) -> PlanInputs {
        PlanInputs::describe(&opened(name, &[])).expect("the synthetic header describes")
    }

    /// A card's kv class is the layers' stores [derived from the header's widths, apart from the
    /// reader]: a KDA layer 2 heads of 128 x 128 f32 (131,072 B) + 11 ring rows (3 + 8 of a
    /// pass) of the q, k and v channels, 3 x 2 x 128 f32 (33,792 B) + a lane's u32 stamp =
    /// 164,868 B whatever the context; a latent layer a row of 512 + 64 f16 values (1,152 B) a
    /// position, 64 positions 73,728 B. At 64 positions the 3 KDA and 1 latent layers hold
    /// 3 x 164,868 + 73,728 = 568,332 B, a layer past the model none, and nothing rotates, so no
    /// rope row. Mutants (`.mutants/kimik3place-*.patch`): the state's bytes halved in
    /// `runtime::stores::recurrent_bytes`, the unrotated values left out of a latent row.
    #[test]
    fn the_layers_stores_are_the_headers_widths() {
        let inputs = described("kimi-k3-place-kv");
        let kv: &super::KvLayout = &inputs.kv;
        for l in [0, 1, 3] {
            assert_eq!(kv.layer_bytes(l, 1), 164_868, "layer {l}");
            assert_eq!(kv.layer_bytes(l, 1 << 20), 164_868, "layer {l}");
        }
        assert_eq!(kv.layer_bytes(2, 1), 1_152);
        assert_eq!(kv.layer_bytes(2, 64), 73_728);
        assert_eq!(kv.layer_bytes(4, 64), 0);
        assert_eq!(kv.bytes(0..4, 64), 3 * 164_868 + 73_728);
        assert_eq!(kv.stage_position_bytes(64), 0, "no layer rotates");
    }

    /// A routed stack is the file's bytes by its type [derived from the header's widths, apart
    /// from the reader]: an expert of the mini's 256 x 256 stack is 12,800 B in IQ1_S and 16,896 B
    /// in IQ2_XXS, so an expert (three stacks) is 38,400 B all IQ1_S, 46,592 B with the gate and
    /// up IQ2_XXS and the down IQ1_S, and 42,496 B with the down IQ2_XXS.
    #[test]
    fn an_expert_is_its_stacks_bytes_by_type() {
        let over = [
            ("blk.2.ffn_gate_exps.weight", Ty::Iq2Xxs),
            ("blk.2.ffn_up_exps.weight", Ty::Iq2Xxs),
            ("blk.3.ffn_down_exps.weight", Ty::Iq2Xxs),
        ];
        let inputs = PlanInputs::describe(&opened("kimi-k3-place-experts", &over))
            .expect("the synthetic header describes");
        let expert = |layer: usize| -> u64 {
            let experts = ["gate", "up", "down"].map(|stack| {
                let name = format!("blk.{layer}.ffn_{stack}_exps.weight");
                let t = inputs
                    .model
                    .tensors
                    .iter()
                    .find(|t| t.name == name)
                    .unwrap_or_else(|| panic!("{name} is in the file"));
                assert_eq!(t.dims[2], 4, "{name}: the stack's experts");
                t.file_bytes / 4
            });
            experts.iter().sum()
        };
        assert_eq!(expert(1), 38_400);
        assert_eq!(expert(2), 46_592);
        assert_eq!(expert(3), 42_496);
    }

    /// The coverage check lists what the body is not built for. The released file's list is the
    /// template's tool-call parser alone (`kimi_k3_places_on_the_box`); the small header's router
    /// of 4 experts is no instance of the body's (896 and 8), so it lists that router at each
    /// routed layer beside the parser. [`PlanInputs::read`] refuses with the whole list, [`PlanInputs::plan`]
    /// makes the plan whatever the list holds, [`PlanInputs::read_without_chat`] leaves the
    /// parser out of the refusal, and a context of no position is refused by its own name.
    /// Mutant: `Program::KimiK3Body` left out of a coverage row's list lists that row's layers.
    #[test]
    fn the_coverage_list_is_what_the_body_is_not_built_for() {
        let inputs = described("kimi-k3-place-refused");
        let parser = Need::ToolParser.to_string();
        let items = inputs.unimplemented();
        assert!(
            items
                .iter()
                .any(|i| i.layer.is_none() && i.feature == parser)
        );
        let shapes: Vec<&str> = items
            .iter()
            .filter(|i| i.feature != parser)
            .map(|i| i.feature.as_str())
            .collect();
        assert_eq!(shapes.len(), 3, "{shapes:?}: the router of layers 1 to 3");
        assert!(
            shapes
                .iter()
                .all(|f| f.starts_with("router: sigmoid, 4 experts, top 2"))
        );
        match PlanInputs::read(&opened("kimi-k3-place-read", &[])) {
            Err(PlacementError::Unimplemented(got)) => assert_eq!(got, items),
            other => panic!("{other:?}, not the coverage refusal"),
        }
        match PlanInputs::read_without_chat(&opened("kimi-k3-place-chatless", &[])) {
            Err(PlacementError::Unimplemented(got)) => {
                assert_eq!(got.len(), items.len() - 1);
                assert!(got.iter().all(|i| i.feature != parser));
            }
            other => panic!("{other:?}, not the refusal without the parser"),
        }
        let machine = workstation::plan_gate(4);
        inputs
            .plan(&machine, 64, &PlanLevers::default())
            .expect("the gate machine plans the small file");
        assert!(matches!(
            inputs.plan(&machine, 0, &PlanLevers::default()),
            Err(PlaceError::CtxZero)
        ));
    }

    /// The tier's bytes are the header's counts [derived apart from the reader]: the stage card
    /// keeps a place per expert per layer for the slot map's tier view and `n_used` places for the
    /// step's row, four bytes each, so the mini's 4 layers of 4 experts, 2 kept, are
    /// 4 x (4 x 4 + 2) = 72 B, which the inputs carry as the tier reserve and a tiered machine's
    /// prompt batch adds to the (zero) prompt reserve. A machine that hangs an expert tier card
    /// passes the body's tier check, and meets the placement's own refusal of a tier with nothing
    /// to hold (the mini's experts all fit the stage card); the inputs of a body that serves no
    /// tier are refused by the tier's name.
    #[test]
    fn a_tier_card_passes_the_bodys_tier_check() {
        let inputs = described("kimi-k3-place-tier");
        assert_eq!(super::tier_places_bytes(&inputs.hp), 72);
        assert_eq!(inputs.tier_reserve, Some(72));
        assert_eq!(inputs.prompt_reserve, 0);
        let batch = super::tier_batch(&inputs.hp).expect("the mini's geometry has leg bytes");
        let machine = workstation::plan_tiers(4, A6000, &[RTX_3090], None, batch)
            .expect("the tier machine plans");
        assert_eq!(inputs.prompt_batch_bytes(&machine).expect("tiered"), 72);
        match inputs.plan(&machine, 64, &PlanLevers::default()) {
            Err(PlaceError::Placement(PlacementError::IdleTier { tier: 0, .. })) => {}
            other => panic!("{other:?}, not the placement's idle-tier refusal"),
        }
        let mut untiered = inputs;
        untiered.tier_reserve = None;
        match untiered.plan(&machine, 64, &PlanLevers::default()) {
            Err(PlaceError::Tier { card, tier }) => assert_eq!((card.as_str(), tier), ("3090", 0)),
            other => panic!("{other:?}, not the tier refusal"),
        }
    }

    /// A dense FFN past the card's q8_1 activation is refused at the description by the tensor's
    /// name and the width, before any load (the released file's dense down is 33,792 wide and
    /// passes; the card's `Q8Act` refused it only after the file was read): the small header's
    /// dense block widened to the cap passes and one super-block past it is the refusal. Mutant:
    /// `kernels::Q8_ACT_MAX_K` left at 24,576, which refuses the 33,792 of the released file.
    #[test]
    fn a_dense_ffn_past_the_q8_activation_is_refused_by_name() {
        use super::super::hparams::tests::E;
        use crate::placement::kernels::Q8_ACT_MAX_K;
        let wide = |name: &str, ff: u64| -> Split {
            let mut kv = keys();
            let key = kv
                .iter_mut()
                .find(|(k, _)| *k == "feed_forward_length")
                .expect("the dense width key");
            key.1 = V::U32(u32::try_from(ff).expect("a u32 width"));
            let dense: Vec<(String, Vec<u64>)> = tensors()
                .into_iter()
                .map(|(n, d)| match n.as_str() {
                    "blk.0.ffn_gate.weight" | "blk.0.ffn_up.weight" => (n, vec![E, ff]),
                    "blk.0.ffn_down.weight" => (n, vec![ff, E]),
                    _ => (n, d),
                })
                .collect();
            let global = [
                ("tokenizer.ggml.pre", V::Str("kimi-k2")),
                ("tokenizer.chat_template", V::Str("{{ messages }}")),
            ];
            let path = header_typed(name, "kimi-k3", &kv, &global, &typed(&dense, &[]));
            let split = Split::open(&path).expect("the synthetic header opens");
            let _ = std::fs::remove_file(&path);
            split
        };
        let widest = Q8_ACT_MAX_K as u64;
        PlanInputs::describe(&wide("kimi-k3-place-wide-ok", 33_792))
            .expect("the released file's dense down, 33,792 wide, describes");
        PlanInputs::describe(&wide("kimi-k3-place-wide-cap", widest))
            .expect("the widest column the engine runs describes");
        match PlanInputs::describe(&wide("kimi-k3-place-wide-past", widest + 256)) {
            Err(PlacementError::Tensor { name, detail }) => {
                assert_eq!(name, "blk.0.ffn_down.weight");
                for word in ["q6_K", "34048", "33792"] {
                    assert!(detail.contains(word), "{word} missing from: {detail}");
                }
            }
            other => panic!("{other:?}, not the width's refusal by tensor"),
        }
    }

    /// A K3.1 latent layer holds the key row of its indexer beside its cached row: 128 f16 values
    /// (256 B) a position [derived from the indexer's key width, apart from the reader], on that
    /// layer alone, so at 64 positions the layer's 1,152 + 256 B a position are 90,112 B and the
    /// 3 delta layers' state is K3's. Mutant: the key row left out of the layout, or put on the
    /// delta layers.
    #[test]
    fn a_k31_latent_layer_holds_the_index_key_row() {
        let inputs = PlanInputs::describe(&opened31("kimi-k31-place-kv"))
            .expect("the synthetic K3.1 header describes");
        let kv: &super::KvLayout = &inputs.kv;
        for l in [0, 1, 3] {
            assert_eq!(kv.layer_bytes(l, 64), 164_868, "layer {l}");
        }
        assert_eq!(kv.layer_bytes(2, 1), 1_152 + 256);
        assert_eq!(kv.layer_bytes(2, 64), 64 * (1_152 + 256));
        assert_eq!(kv.bytes(0..4, 64), 3 * 164_868 + 90_112);
        assert_eq!(kv.layer_bytes(4, 64), 0);
    }

    /// The coverage check lists no selection for a K3.1 file: the description asks for the
    /// indexer's two needs at the latent layer alone and the body runs them, at a context of its
    /// top-k (the selection is every position) and past it (the rows the indexer picks). At the
    /// body's router instance (8 experts, 2 kept) the K3.1 header is read whole but for the
    /// template's parser. Mutants: the selection's row (`spec::AVAILABLE`) removed, the bound past
    /// top-k put back.
    #[test]
    fn a_k31_selection_is_run_at_every_context() {
        let inputs = PlanInputs::describe(&opened31("kimi-k31-place-listed"))
            .expect("the synthetic K3.1 header describes");
        let asked: Vec<_> = models::needs(&inputs.spec)
            .into_iter()
            .filter(|(n, _)| matches!(n, Need::StreamIndex { .. } | Need::IndexKeyFromInput))
            .collect();
        assert_eq!(asked.len(), 2, "{asked:?}");
        assert!(asked.iter().all(|(_, l)| *l == Some(2)), "{asked:?}");
        let selected: Vec<_> = inputs
            .unimplemented()
            .into_iter()
            .filter(|i| i.feature.contains("index"))
            .collect();
        assert!(selected.is_empty(), "the selection is listed: {selected:?}");
        PlanInputs::read_without_chat(&opened31_at("kimi-k31-place-read", TOP_K as u32, 8))
            .expect("a K3.1 header at the body's router instance reads whole");
        for ctx in [TOP_K, TOP_K + 1, 1 << 20] {
            inputs
                .check_ctx(ctx)
                .unwrap_or_else(|e| panic!("a context of {ctx} runs: {e}"));
        }
        let k3 = described("kimi-k3-place-ctx");
        k3.check_ctx(1 << 20).expect("a K3 file has no bound");
    }

    /// A K3.1 context past the top-k plans, and its stores are the plan's bytes [derived from the
    /// header's widths, apart from the reader]: a latent layer holds its cached row, 512 + 64 f16
    /// values (1,152 B), and its index key row, 128 f16 values (256 B), a position, so 1,408 B a
    /// position and 369,098,752 B at 262,144 positions; the 3 delta layers' 164,868 B each whatever
    /// the context. The gate machine plans the small file at `top_k + 1` and at 262,144 positions
    /// with its card's kv class those bytes. Mutants: the bound past top-k put back (no plan), the
    /// key row left out of the layout (1,152 B a position).
    #[test]
    fn a_k31_context_past_top_k_plans_its_stores() {
        const CTX_256K: u64 = 262_144;
        let inputs = PlanInputs::describe(&opened31("kimi-k31-place-256k"))
            .expect("the synthetic K3.1 header describes");
        let kv: &super::KvLayout = &inputs.kv;
        assert_eq!(kv.layer_bytes(2, CTX_256K), 369_098_752);
        assert_eq!(kv.bytes(0..4, CTX_256K), 3 * 164_868 + 369_098_752);
        let machine = workstation::plan_gate(4);
        for ctx in [TOP_K + 1, CTX_256K] {
            inputs
                .check_ctx(ctx)
                .unwrap_or_else(|e| panic!("a context of {ctx} runs: {e}"));
            let plan = inputs
                .plan(&machine, ctx, &PlanLevers::default())
                .unwrap_or_else(|e| panic!("the gate machine plans a context of {ctx}: {e}"));
            let held: u64 = plan.cards.iter().map(|c| c.kv_bytes).sum();
            assert_eq!(held, 3 * 164_868 + ctx * 1_408, "ctx {ctx}");
            assert_eq!(held, kv.bytes(0..4, ctx), "ctx {ctx}");
        }
    }

    /// The by-position buffers a K3.1 latent layer declares are its stores' rows [derived from the
    /// header's widths, apart from the reader]: the cached row (512 + 64 f16 values, 1,152 B) a
    /// step reads through the indexer's picks, so a `Select` of the indexer's `top_k` rows, and
    /// the index key row (128 f16 values, 256 B) the indexer scores whole, a `Scan`; a delta layer
    /// declares none. Their bytes are the layer's own at every context, 1,408 B a position, which
    /// at the released file's 24 latent layers are 33,792 B a position. A K3 file has no
    /// indexer: its one buffer is the cached row, a `Scan` (card-only). Mutants: the key row
    /// declared `Select` (the host-able bytes move), the cached row of a K3 file declared `Select`,
    /// the key row left out of the declaration.
    #[test]
    fn the_declared_rooms_are_the_positional_rows() {
        use runtime::stores::Access;
        const RELEASED_LATENT_LAYERS: u64 = 24;
        let inputs = PlanInputs::describe(&opened31("kimi-k31-place-rooms"))
            .expect("the synthetic K3.1 header describes");
        let rooms = inputs.kv.rooms().expect("a K3.1 file declares its rooms");
        for l in [0, 1, 3, 4] {
            assert!(rooms.buffers(l).is_empty(), "layer {l}");
        }
        let [cache, keys] = rooms.buffers(2) else {
            panic!(
                "{:?}: the cached row and the index key row",
                rooms.buffers(2)
            );
        };
        assert_eq!((cache.row, cache.per, cache.runs), (1_152, 1, 1));
        assert_eq!(cache.access, Access::Select { rows: TOP_K });
        assert_eq!((keys.row, keys.per, keys.runs), (256, 1, 1));
        assert_eq!(keys.access, Access::Scan);
        for ctx in [1, 64, 262_144] {
            for l in 0..5 {
                let layer = KvBytes::layer_bytes(&inputs.kv, l, ctx);
                let held = if l == 2 { layer } else { 0 };
                assert_eq!(rooms.bytes(l, ctx), held, "layer {l} at {ctx}");
            }
        }
        assert_eq!(rooms.bytes(2, 1), 1_408);
        assert_eq!(RELEASED_LATENT_LAYERS * 1_408, 33_792);
        assert_eq!(
            rooms.host_able(2, 64),
            64 * 1_152,
            "the cached row is host-able alone"
        );

        let k3 = described("kimi-k3-place-rooms");
        let rooms = k3.kv.rooms().expect("a K3 file declares its rooms");
        let [cache] = rooms.buffers(2) else {
            panic!("{:?}: the cached row alone", rooms.buffers(2));
        };
        assert_eq!((cache.row, cache.access), (1_152, Access::Scan));
        assert_eq!(
            rooms.host_able(2, 64),
            0,
            "a K3 file has nothing a host page may hold"
        );
        assert_eq!(rooms.bytes(2, 64), KvBytes::layer_bytes(&k3.kv, 2, 64));
    }

    /// The plan reads the declaration end to end: with no floor every term is at the cap (the
    /// card's kv class is `a_k31_context_past_top_k_plans_its_stores`'), and with a floor of 8,192
    /// positions at a context of 262,144 the latent layer's rooms count 8,192 x 1,408 =
    /// 11,534,336 B [derived], the delta layers' states and every byte past the rooms staying at
    /// the cap; under `BLOOMERY_KV_HOST=all` the cached row leaves the card (8,192 x 256 B stay),
    /// and a K3 file under `all` has no host-able room, so its bytes stay. Mutants: the rooms left
    /// out of the layout (the floor changes nothing), `rooms()` not forwarded by the wrapper.
    #[test]
    fn a_k31_plan_counts_its_rooms_at_the_floor() {
        use bloomery_levers::KvHost;
        const CTX: u64 = 262_144;
        const FLOOR: u64 = 8_192;
        const PAGE: u64 = 2 << 20;
        let machine = workstation::plan_gate(4);
        let held = |inputs: &PlanInputs, levers: &PlanLevers| -> u64 {
            let plan = inputs
                .plan(&machine, CTX, levers)
                .unwrap_or_else(|e| panic!("the gate machine plans {CTX}: {e}"));
            plan.cards.iter().map(|c| c.kv_bytes).sum()
        };
        let fixed = 3 * 164_868;
        let mut k31 = PlanInputs::describe(&opened31("kimi-k31-place-floor"))
            .expect("the synthetic K3.1 header describes");
        k31.kv_host_page = Some(PAGE);
        let at_floor = PlanLevers {
            kv_floor: Some(FLOOR),
            ..PlanLevers::default()
        };
        assert_eq!(held(&k31, &PlanLevers::default()), fixed + CTX * 1_408);
        assert_eq!(held(&k31, &at_floor), fixed + FLOOR * 1_408);
        let all = PlanLevers {
            kv_host: KvHost::All,
            ..PlanLevers::default()
        };
        assert_eq!(held(&k31, &all), fixed + CTX * 256);
        let mut k3 = described("kimi-k3-place-floor");
        k3.kv_host_page = Some(PAGE);
        let k3_cap = fixed + CTX * 1_152;
        assert_eq!(held(&k3, &PlanLevers::default()), k3_cap);
        assert_eq!(
            held(&k3, &all),
            k3_cap,
            "a scanned buffer never leaves the card"
        );
        assert_eq!(held(&k3, &at_floor), fixed + FLOOR * 1_152);
    }

    /// A K3.1 plan under `BLOOMERY_KV_HOST` counts the host-able cached row's pages against its
    /// host room [derived by hand, the fixture's one latent layer, page 2 MiB]: at 262,144
    /// positions the cached run is 144 whole pages (301,989,888 B), at a floor of 8,192 it is 5
    /// pages (10,485,760 B), so `spill` sends 291,504,128 B to the host and `all` 301,989,888 B,
    /// the key run never; `off` and a plan with no floor count none. A host backing with no page
    /// is refused by name before any term is read, and a plan with the page set adds the rooms'
    /// page rounding and the device's spare page to card 0's stage rows: at 100,000 positions
    /// the cached run is 143,360 B and the key run 1,662,976 B past their rows, and the spare
    /// page 2,097,152 B. Mutants: the host count left out of `plan_with_slots`, the floor's pages
    /// not taken off, the padding left out of the stage rows.
    #[test]
    fn a_k31_plan_counts_its_host_pages_and_padding() {
        use bloomery_levers::KvHost;
        const PAGE: u64 = 2 << 20;
        const CTX: u64 = 262_144;
        let machine = workstation::plan_gate(4);
        let mut inputs = PlanInputs::describe(&opened31("kimi-k31-place-host"))
            .expect("the synthetic K3.1 header describes");
        let levers = |host, floor| PlanLevers {
            kv_host: host,
            kv_floor: floor,
            ..PlanLevers::default()
        };
        let host_bytes = |inputs: &PlanInputs, levers: &PlanLevers, ctx| -> u64 {
            inputs
                .plan(&machine, ctx, levers)
                .unwrap_or_else(|e| panic!("the gate machine plans {ctx}: {e}"))
                .host
                .kv_host_bytes
        };
        inputs.kv_host_page = Some(PAGE);
        assert_eq!(
            host_bytes(&inputs, &levers(KvHost::Spill, Some(8_192)), CTX),
            291_504_128
        );
        assert_eq!(
            host_bytes(&inputs, &levers(KvHost::All, None), CTX),
            301_989_888
        );
        assert_eq!(host_bytes(&inputs, &levers(KvHost::Spill, None), CTX), 0);
        assert_eq!(
            host_bytes(&inputs, &levers(KvHost::Off, Some(8_192)), CTX),
            0
        );

        inputs.kv_host_page = None;
        let refused = inputs
            .plan(&machine, CTX, &levers(KvHost::Spill, Some(8_192)))
            .expect_err("a host backing with no page is refused");
        assert!(
            refused.to_string().contains("BLOOMERY_KV_HOST"),
            "{refused}"
        );

        let kv = |inputs: &PlanInputs| -> u64 {
            let plan = inputs
                .plan(&machine, 100_000, &levers(KvHost::Off, Some(8_192)))
                .expect("the gate machine plans 100,000");
            plan.cards[0].kv_bytes
        };
        let bare = kv(&inputs);
        inputs.room_page = Some(PAGE);
        assert_eq!(kv(&inputs) - bare, 143_360 + 1_662_976 + PAGE);
    }

    /// The rounding lands on the card that holds the room's layer, with that card's spare page,
    /// and on no other [derived by hand, page 2 MiB, 100,000 positions]: the fixture's one latent
    /// layer (layer 2) on the second of two stage cards (layers 0..2 and 2..4) adds 143,360 B +
    /// 1,662,976 B + the spare page 2,097,152 B to card 1's cache term and nothing to card 0's,
    /// which holds no room. Mutant: the rounding counted on card 0 only (`kvc-k31-padding-card-0`).
    #[test]
    fn a_k31_plan_counts_each_cards_room_padding() {
        use bloomery_levers::KvHost;
        const PAGE: u64 = 2 << 20;
        let mut machine = workstation::plan_gate(4);
        let mut second = workstation::plan_on(A6000, 4).cards.remove(0);
        second.head = false;
        second.token_embedding = false;
        second.layers = 2..4;
        machine.cards[0].layers = 0..2;
        machine.cards.push(second);
        let mut inputs = PlanInputs::describe(&opened31("kimi-k31-place-cards"))
            .expect("the synthetic K3.1 header describes");
        let levers = PlanLevers {
            kv_floor: Some(8_192),
            kv_host: KvHost::Off,
            ..PlanLevers::default()
        };
        let kv = |inputs: &PlanInputs| -> Vec<u64> {
            inputs
                .plan(&machine, 100_000, &levers)
                .unwrap_or_else(|e| panic!("the two-card machine plans 100,000: {e}"))
                .cards
                .iter()
                .map(|c| c.kv_bytes)
                .collect()
        };
        let bare = kv(&inputs);
        inputs.room_page = Some(PAGE);
        let padded = kv(&inputs);
        assert_eq!(padded[0] - bare[0], 0, "card 0 holds no room");
        assert_eq!(padded[1] - bare[1], 143_360 + 1_662_976 + PAGE);
    }

    /// A driver that opens a K3.1 file with a context reads it by `read_without_chat_for`: at the
    /// released file's top-k of 2,048 a context of 2,048, of 2,049 and of 262,144 positions opens.
    /// Mutant: the bound past top-k put back in the read.
    #[test]
    fn a_driver_context_past_top_k_opens_as_the_file_is_read() {
        const RELEASED_TOP_K: u32 = 2048;
        for ctx in [2048, 2049, 262_144] {
            let name = format!("kimi-k31-place-ctx-{ctx}");
            PlanInputs::read_without_chat_for(&opened31_at(&name, RELEASED_TOP_K, 8), ctx)
                .unwrap_or_else(|e| panic!("a context of {ctx} opens: {e}"));
        }
    }

    /// The plan's description is the registry's: one reading of the file, whose chat surface is
    /// the Kimi K2 pre-tokenizer and the file's template, with no tool-call parser named and the
    /// reasoning sections the template spells ([`spec::REASONING`]).
    #[test]
    fn the_plans_description_is_the_registrys() {
        let split = opened("kimi-k3-place-chat", &[]);
        let inputs = PlanInputs::describe(&split).expect("describes");
        let read = crate::arch::spec(&split).expect("the registry reads");
        assert_eq!(inputs.spec, read.spec);
        let chat = &inputs.spec.chat;
        assert_eq!(chat.pre, "kimi-k2");
        assert_eq!(chat.template.as_deref(), Some("{{ messages }}"));
        assert!(chat.tools.is_none());
        assert_eq!(chat.reasoning, Some(super::spec::REASONING));
    }
}
