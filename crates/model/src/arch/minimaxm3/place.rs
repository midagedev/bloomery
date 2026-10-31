//! An M3 file planned onto a machine: the common hybrid owner's plan ([`place_hybrid`]) over this
//! family's reading. What each layer holds is read off the description: every layer's K and V
//! planes, and for a sparse-attention layer ([`models::Selector::BlockMax`]) an f16 index key of
//! the indexer's width for every position ([`KvLayout`]), whichever feed-forward block that layer
//! runs. The routed experts' bytes are the routed layers' stacks. The card experts' formats are
//! the common leg's. A Q6_K down beside a gate and up the leg runs is a card layer; the down reads
//! the leg's `Q8Act` columns. The expert tier card a machine names (plan (b′): the A6000 the
//! stage, the 3090 its tier) holds the next ids of each card layer on that leg, the NVFP4 stacks
//! in their planes; its prompt-batch bytes are [`tier_batch`]. The prompt batch's card buffers are
//! the plan's reserve ([`prompt_reserve_bytes`]) and a tiered load's copies of the tier places
//! beside them the tier reserve ([`tier_places_bytes`]); the stage card's own tier places sit in
//! the scratch term the load holds its made bytes to. The family's own is the reading
//! ([`PlanInputs::describe`]) and the engine's name in the refusals.

use gguf::Split;
use runtime::prompt::group_sets;

use super::hparams::Hparams;
use super::{roles, spec};
use crate::arch::chat_of;
use crate::arch::names::routed_stacks;
use crate::arch::place_hybrid::{
    self, ring_forms_planned, ring_pass_bytes_in, route_bytes, stream_ring_bytes,
    tier_places_bytes, wide_bytes,
};
use crate::placement::PlacementError;
use crate::placement::accepts::{self, Program};
use crate::placement::kernels::{self, DownAct, DownForms};
use crate::placement::workstation::{self, TierBatchBytes};

pub use place_hybrid::{KvLayout, PlaceError};

/// What a plan of an M3 file is made from, read from its headers.
pub type PlanInputs = place_hybrid::PlanInputs<Hparams>;

/// The engine the plan's refusals name.
const ENGINE: &str = "MiniMax-M3";

/// The rows a unit of the body's prompt batch holds. A unit reads each hosted expert its rows pick
/// once and runs every (expert, row) slot through the NVFP4 column tile. At 128 rows the host's
/// read of a unit's experts and its tile instructions take about the same time; at 256 the read
/// halves a row while the instructions a row stay, so the host leg is its multiply-adds and more
/// rows, or one union over a group's units, no longer shorten it. Two units of 256 still fill the
/// default prompt group of a 512-id call, whose second unit's attention runs under the first's host
/// leg. The host union takes up to [`crate::moe::UNION_MAX_COLS`] columns.
pub const PROMPT_ROWS: usize = 256;

/// The rows one flash launch of the prompt batch runs at most: the batch's partials hold this many
/// rows of the dense layers' decode flash (a row's partials are the largest of any flash the batch
/// launches), and the sparse layers' launches take as many rows as those partials hold.
pub const FLASH_ROWS: usize = 16;

/// The decode flash's segments a query head's keys are cut into, as the kernels cut them; the body
/// holds this equal to the kernels' by assertion.
pub const FLASH_SEGMENTS: usize = 80;

const _: () = assert!(FLASH_ROWS <= PROMPT_ROWS && PROMPT_ROWS <= crate::moe::UNION_MAX_COLS);

/// The card bytes the body's prompt batch makes beside the plan's scratch term for a file described
/// by `hp`, at [`PROMPT_ROWS`] rows. The plan keeps them out of the expert budget
/// ([`place_hybrid::PlanInputs::prompt_reserve`]) and the body holds its made bytes to the same
/// figure (`Body::prompt_bytes`). The terms, in four-byte words a row unless said:
///
/// - the batch's rows: the normed rows and the sub-layers' deltas (`2 · embd`), the query
///   projection with its normed and turned heads and the flash's output (`3 · q_width`), the key and
///   value rows (`2 · kv_width`), the index heads' projection and turned heads (`2 · idx`) and the
///   index key row with its value row (`2 · idx_d`);
/// - the flash's partials, for [`FLASH_ROWS`] rows of `n_head` query heads of [`FLASH_SEGMENTS`]
///   segments, each a head-wide `Σ exp·V` row and a `(max, Σ exp)` pair; the sparse layers' listed
///   flash cuts a row into fewer segments than the dense layers' decode flash, so the dense
///   layers' size every launch;
/// - the selection's block ids (`msa.heads · top_k`) and key count; the block scores are the
///   scratch term's, as they grow with the context;
/// - the unit's buffers of the walk's first group set (the rows `x` and `x1`, the positions and key
///   counts) and the host sums (`embd`); the sets a larger prompt group adds are made out of the
///   card's margin;
/// - the routed blocks' rows: the normed rows, the shared expert's output and the base (`3 · embd`),
///   the router's scores (`n_expert`), ids and weights (`2 · n_used`), the card experts' sum and the
///   sum plus the base (`2 · embd`) and places (`n_used`), the block input of the widest block
///   (`max(shared_ff, dense_ff)`), and each GEMM block's gate and up rows (`2 · shared_ff`, and for
///   the dense layers `2 · dense_ff` with their normed and output rows `2 · embd`);
/// - the card experts' leg buffers ([`kernels::leg_bytes`], in both of the down forms a leg holds);
/// - the wide forms of the rows at each width a GEMM reads, both forms where the width allows them
///   and the f32 rows of the blocks' SwiGLU input (a bound: the load makes the ones its sites'
///   types read), and a one-expert route for each of the batch's and the two blocks' GEMMs.
#[must_use]
pub fn prompt_reserve_bytes(hp: &Hparams) -> u64 {
    let r = PROMPT_ROWS;
    let embd = hp.n_embd;
    let (q_width, kv_width) = (hp.n_head * hp.head_dim, hp.n_head_kv * hp.head_dim);
    let (idx, idx_d) = (hp.msa.heads * hp.msa.d, hp.msa.d);
    let (experts, shared_ff) = (&hp.experts, hp.experts.shared_ff());
    let dense_ff = hp.dense_ff.unwrap_or(0);
    let batch_rows = 2 * embd + 3 * q_width + 2 * kv_width + 2 * idx + 2 * idx_d;
    let selection = hp.msa.heads * hp.msa.top_k + 1;
    let units = group_sets(1) * (2 * embd + 2) + embd;
    let routed = 3 * embd
        + experts.n_expert
        + 2 * experts.n_used
        + 2 * embd
        + experts.n_used
        + shared_ff.max(dense_ff)
        + 2 * shared_ff;
    let dense = if dense_ff > 0 {
        2 * embd + 2 * dense_ff
    } else {
        0
    };
    let partials = r.min(FLASH_ROWS) * hp.n_head * FLASH_SEGMENTS * (hp.head_dim + 2);
    let mut forms = wide_bytes(r, embd) + wide_bytes(r, q_width);
    forms += 2 * wide_bytes(r, embd) + wide_bytes(r, shared_ff) + 4 * r * shared_ff;
    let mut routes = 2 * route_bytes(r);
    if dense_ff > 0 {
        forms += wide_bytes(r, dense_ff) + 4 * r * dense_ff;
        routes += route_bytes(r);
    }
    let leg = kernels::leg_bytes(
        r as u64,
        experts.n_used as u64,
        embd as u64,
        experts.ff as u64,
        DownForms::of(DownAct::Q8Act).with(DownAct::Q8Blocks32),
    ) as usize;
    let words = r * (batch_rows + selection + units + routed + dense) + partials;
    (4 * words + forms + routes + leg) as u64
}

/// Tokens one run of the tier's card leg covers (the tier's `CHUNK`, which holds itself equal to
/// this): a block of a prompt unit's rows runs in runs of at most this many.
pub const TIER_CHUNK: u64 = 16;

/// The expert tier's prompt-batch bytes ([`TierBatchBytes`]) for the file `inputs` describes and
/// blocks of `cols` rows (the body's prompt unit): the tier card's staging for a block, one run of
/// the card leg's buffers over [`TIER_CHUNK`] of its rows in the forms the leg's routed downs read
/// (an NVFP4 down's 32-value blocks, a K- or i-quant down's 128-value form) and the host's rows
/// and places. The plan reserves them on the tier card and the host, and the load checks its
/// allocations against them.
pub fn tier_batch(inputs: &PlanInputs, cols: u64) -> Result<TierBatchBytes, PlacementError> {
    let hp = &inputs.hp;
    workstation::try_tier_leg_batch_bytes_of(
        hp.n_embd as u64,
        hp.experts.ff as u64,
        hp.experts.n_used as u64,
        (TIER_CHUNK, cols),
        accepts::leg_down_forms(Program::MiniMaxM3Body, &inputs.model),
        false,
    )
}

/// The slots of each half of the expert stream's ring a streaming plan keeps room for
/// ([`stream_bytes`]) [derived: at the ring pass's route GEMMs the stream rule's crossover is about
/// 18 columns of a walk, and a walk of the default group (two units of [`PROMPT_ROWS`] rows, four
/// picks a row over 128 experts) averages 16 columns an expert, so the experts past it are about a
/// quarter of a layer's host experts, under 32; a half of 32 NVFP4 experts is about 1 GB].
pub const STREAM_HALF_SLOTS: u64 = 32;

/// The card bytes the prompt calls' expert stream takes on an M3 load (`BLOOMERY_XSTREAM=split`):
/// the common rule ([`stream_ring_bytes`]) over the routed layers' three stacks at
/// [`STREAM_HALF_SLOTS`] a half, beside the ring pass's GEMM buffers at [`PROMPT_ROWS`] rows in the
/// forms the routed stacks read ([`ring_pass_bytes_in`], [`ring_forms_planned`]). Refused by name as the rule refuses.
pub fn stream_bytes(inputs: &PlanInputs) -> Result<u64, PlacementError> {
    let hp = &inputs.hp;
    let half = STREAM_HALF_SLOTS;
    let pass = ring_pass_bytes_in(
        PROMPT_ROWS,
        hp.experts.n_used,
        hp.n_embd,
        hp.experts.ff,
        half as usize,
        ring_forms_planned(&inputs.model, routed_stacks),
    ) as u64;
    stream_ring_bytes(
        &inputs.model,
        routed_stacks,
        (hp.experts.n_expert as u64, half),
        pass,
    )
}

impl PlanInputs {
    /// `self` for a load whose prompt calls stream their host experts (`BLOOMERY_XSTREAM=split`):
    /// the prompt reserve grows by [`stream_bytes`] ([`place_hybrid::PlanInputs::streaming`]), so
    /// the stream's ring finds its room on a card the plan otherwise fills with experts. Refused by
    /// name as [`stream_bytes`] refuses.
    pub fn with_stream(self) -> Result<PlanInputs, PlacementError> {
        let ring = stream_bytes(&self)?;
        Ok(self.streaming(ring))
    }

    /// [`PlanInputs::describe`], then the file refused with every feature the engine does not run
    /// ([`PlacementError::Unimplemented`]). The reader refuses a next-token block by its tensors.
    pub fn read(split: &Split) -> Result<PlanInputs, PlacementError> {
        PlanInputs::describe(split)?.runnable()
    }

    /// The file as it is, for a gate that plans or describes it whatever the coverage list holds,
    /// and the host's room and the arena's given bytes. The prompt batch's buffers are the plan's
    /// reserve beside its terms ([`prompt_reserve_bytes`], and [`tier_places_bytes`] on a machine
    /// that names a tier card); the step's come out of its scratch term. The body's host run is
    /// the common one, so the plan pages the host experts the room cannot hold
    /// ([`PlanInputs::with_nvme_tier`]).
    pub fn describe(split: &Split) -> Result<PlanInputs, PlacementError> {
        let hp = Hparams::read(split)?;
        let model = roles::classify(split, &hp)?;
        // No parser is declared: see `spec`.
        let spec = spec::spec_of(&hp, chat_of(split, None, None)?)?;
        let reserve = prompt_reserve_bytes(&hp);
        let tier = tier_places_bytes(PROMPT_ROWS, hp.experts.n_used);
        Ok(PlanInputs::new(ENGINE, hp, model, spec)?
            .with_prompt_reserve(reserve)
            .with_tier_reserve(tier)
            .with_nvme_tier())
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use gguf::Split;

    use super::super::hparams::tests::{
        IDX_D, LAYERS, LEAD, keys, keys_shaped, tensors, tensors_shaped,
    };
    use super::super::hparams::{Attn, FfnKind, Kind};
    use super::{PROMPT_ROWS, PlaceError, PlanInputs, TIER_CHUNK, stream_bytes, tier_batch};
    use crate::arch::place_hybrid::ring_pass_bytes;
    use crate::arch::synthetic::{V, header_shaped, host_run_experts};
    use crate::placement::{PlacementError, PlanLevers, workstation};

    /// The small header's global keys: the pre-tokenizer and a template the description reads.
    fn global() -> [(&'static str, V); 2] {
        [
            ("tokenizer.ggml.pre", V::Str("minimax-m2")),
            ("tokenizer.chat_template", V::Str("{{ messages }}")),
        ]
    }

    /// A header, as the small file the `hparams` tests describe; each test names its own file (the
    /// tests run in threads of one process).
    fn written(name: &str) -> std::path::PathBuf {
        header_shaped(name, "minimax-m3", &keys(), &global(), &tensors())
    }

    /// The small header's file, described.
    fn described(name: &str) -> PlanInputs {
        let path = written(name);
        let split = Split::open(&path).expect("the synthetic header opens");
        let inputs = PlanInputs::describe(&split).expect("the synthetic header describes");
        let _ = std::fs::remove_file(&path);
        inputs
    }

    /// [`described`], its routed stacks rows a host kernel runs ([`host_run_experts`]): a header
    /// a plan places.
    fn runnable(name: &str) -> PlanInputs {
        let mut inputs = described(name);
        host_run_experts(&mut inputs.model);
        inputs
    }

    /// A header reads into the plan's inputs; the file is refused, every missing feature listed,
    /// until a program runs it. A file the reader refuses is that refusal, not a coverage one.
    #[test]
    fn the_inputs_describe_a_file_and_the_coverage_refuses_it() {
        let path = written("mm3-place");
        let split = Split::open(&path).expect("the synthetic header opens");
        let described = PlanInputs::describe(&split).expect("the header describes");
        let refused = PlanInputs::read(&split);
        let _ = std::fs::remove_file(&path);
        assert_eq!(described.model.layers, 5);
        assert_eq!(
            described.hp.kind(1),
            Kind {
                attn: Attn::Full,
                ffn: FfnKind::Dense
            }
        );
        assert_eq!(
            described.hp.kind(2),
            Kind {
                attn: Attn::Sparse,
                ffn: FfnKind::Routed
            }
        );
        // The batch's buffers at PROMPT_ROWS = 256 rows [derived from the header's widths, apart
        // from the formula]: the f32 words 4 B × (256 × 1,527 + 16 × 8 × 80 × 18) = 2,300,928 B
        // (a row's 1,527 words: batch rows 672, selection 7, unit 194, routed block 430, dense
        // block 224; the partials 184,320 words, 16 rows of 8 query heads), the wide forms
        // 176,128 B, three routes 6,288 B (4 B × (2 × 256 + 2 × 5 + 2) each) and the card leg
        // 180,224 B. Mutants: the partials counted at the batch's 256 rows, the card leg or the
        // routes left out, the reserve left 0, the unit left at 128 rows.
        assert_eq!(
            described.prompt_reserve,
            2_300_928 + 176_128 + 6_288 + 180_224
        );
        match refused {
            Err(PlacementError::Unimplemented(items)) => {
                assert!(
                    items.iter().any(|u| u.feature.contains("swigluoai")),
                    "{items:?}"
                );
            }
            other => panic!("the coverage refusal: {other:?}"),
        }
        let bad = header_shaped(
            "mm3-place-bad",
            "minimax-m3",
            &[("block_count", V::U32(5))],
            &global(),
            &tensors(),
        );
        let split = Split::open(&bad).expect("the synthetic header opens");
        let err = PlanInputs::read(&split).expect_err("a header without keys");
        let _ = std::fs::remove_file(&bad);
        assert!(matches!(err, PlacementError::Metadata { .. }), "{err:?}");
    }

    /// A streaming plan keeps the stream's room [derived from the header's widths, apart from the
    /// formula]: the ring's two halves of 32 slots, a slot the widest routed layer's three F32
    /// stacks per expert (64 × 24 × 4 B each, 18,432 B), 1,179,648 B; the ring pass's buffers at
    /// 256 rows of 2 picks (512 slots): the rows' 32-value form 20,480 B, the gate and up rows
    /// 98,304 B, their SwiGLU's 32-value form 40,960 B and the route over 32 experts 2,376 B. At
    /// M3.1's widths (6,144 wide, 3,072 an expert, 4 picks) the pass is 31,068,552 B. A plan that
    /// does not stream keeps none of it. Mutants: `with_stream` leaves the reserve as it was, the
    /// ring counted at one half, the pass's gate and up rows left out.
    #[test]
    fn a_streaming_plan_keeps_the_streams_room() {
        let inputs = described("mm3-place-stream");
        let base = inputs.prompt_reserve;
        let pass = 20_480 + 98_304 + 40_960 + 2_376;
        assert_eq!(ring_pass_bytes(PROMPT_ROWS, 2, 64, 24, 32), pass);
        assert_eq!(
            stream_bytes(&inputs).expect("the routed stacks"),
            1_179_648 + pass as u64
        );
        let streaming = inputs.with_stream().expect("the routed stacks");
        assert_eq!(streaming.prompt_reserve, base + 1_179_648 + pass as u64);
        assert_eq!(
            ring_pass_bytes(PROMPT_ROWS, 4, 6_144, 3_072, 32),
            31_068_552
        );
    }

    /// The probe's plan and the open's plan keep the same room under the stream: the run sizes the
    /// ring once from the file it read and plans `streaming` of those bytes; the open describes the
    /// file again and plans `streaming` of the run's bytes. Both take the plain plan's batch bytes
    /// plus the ring's [derived from the header's widths, apart from the formula]: two halves of 32
    /// slots, a slot an expert's three Q8_0 stacks ([`runnable`]: the gate and up 24 rows of 64
    /// values, 1,632 B each, the down 64 rows of 32, 2,176 B), 348,160 B, and the pass's buffers,
    /// 162,120 B, 510,280 B, and they plan the same context and expert counts. Mutant: `streaming` leaves the reserve as it was (both equal the
    /// plain plan's, the clause naming the ring's bytes reddens).
    #[test]
    fn the_probe_and_the_open_keep_the_same_stream_room() {
        let machine = workstation::plan_gate(LAYERS);
        let plain = runnable("mm3-place-plain");
        let ring = stream_bytes(&plain).expect("the routed stacks");
        assert_eq!(ring, 348_160 + 162_120);
        let probe = runnable("mm3-place-probe").streaming(ring);
        let open = runnable("mm3-place-open").streaming(ring);
        let room = |inputs: &PlanInputs| {
            inputs
                .prompt_batch_bytes(&machine)
                .expect("the gate machine names no tier")
        };
        assert_eq!(room(&probe), room(&open));
        assert_eq!(room(&probe), room(&plain) + ring);
        let levers = PlanLevers::default();
        let slots = NonZeroUsize::new(1).expect("a slot");
        let probe_plan = probe
            .plan_with_slots(&machine, 64, &levers, slots)
            .expect("the small header plans");
        let open_plan = open
            .plan_with_slots(&machine, 64, &levers, slots)
            .expect("the small header plans");
        assert_eq!(probe_plan.ctx_max, open_plan.ctx_max);
        assert_eq!(probe_plan.cards[0].experts, open_plan.cards[0].experts);
        assert_eq!(probe_plan.host.experts, open_plan.host.experts);
    }

    /// The card's kv class is the layers' stores [derived from the header's widths, apart from the
    /// reader]: 2 key heads × (16 + 16) × 2 B = 128 B of K and V a position, and the 3 sparse
    /// layers (2, 3 and 4) 16 f16 index-key values = 32 B more; at 64 positions the 2 dense layers
    /// are 8,192 B each and the 3 sparse 10,240 B each, 47,104 B, a slot each time. The stage keeps
    /// one table of the 8 rotated values in f32 (32 B) a position, 2,048 B at 64, whatever the
    /// slots. A context of no position is refused by name. Mutant: the index keys left out of the
    /// layout (the card holds 40,960 B), or the stage rows counted a slot.
    #[test]
    fn the_cards_kv_class_is_the_layers_stores() {
        let inputs = runnable("mm3-place-kv");
        assert_eq!(inputs.kv.bytes(0..LAYERS, 64), 47_104);
        let machine = workstation::plan_gate(LAYERS);
        let levers = PlanLevers::default();
        let slots = |n| NonZeroUsize::new(n).expect("a slot");
        for n in [1, 3] {
            let plan = inputs
                .plan_with_slots(&machine, 64, &levers, slots(n))
                .expect("the small header plans");
            let want = n as u64 * 47_104;
            assert_eq!(plan.cards[0].kv_bytes, want, "{n} slots");
            assert_eq!(want, n as u64 * inputs.kv.bytes(0..LAYERS, 64), "{n} slots");
            // The rope table and the index selector's sink row, a position each.
            let rows = 8 * 4 + IDX_D * 2;
            assert_eq!(plan.cards[0].stage_rows_bytes, rows * 64, "{n} slots");
        }
        assert!(matches!(
            inputs.plan(&machine, 0, &levers),
            Err(PlaceError::CtxZero)
        ));
    }

    /// A tiered machine retains the stage's prompt reserve and the tier places beside it.
    /// The fixture's experts fit on the stage, so placement refuses the idle tier rather
    /// than rejecting the description as a body without tier support.
    #[test]
    fn a_tier_card_is_planned_with_the_leg_batch() {
        const COLS: u64 = 3 * TIER_CHUNK;
        let inputs = described("mm3-place-tier");
        let tiered = workstation::plan_bp(
            LAYERS,
            None,
            tier_batch(&inputs, COLS).expect("valid tier geometry"),
        );
        let plain = workstation::plan_gate(LAYERS);
        let places = 3 * 4 * 256 * 2;
        assert_eq!(inputs.tier_reserve, Some(places));
        assert_eq!(
            inputs.prompt_batch_bytes(&plain).expect("a plain machine"),
            inputs.prompt_reserve
        );
        assert_eq!(
            inputs
                .prompt_batch_bytes(&tiered)
                .expect("a tiered machine"),
            inputs.prompt_reserve + places
        );
        match inputs.plan(&tiered, 64, &PlanLevers::default()) {
            Err(PlaceError::Placement(PlacementError::IdleTier { tier: 0, .. })) => {}
            other => panic!("{other:?}, not the placement's idle-tier refusal"),
        }
    }

    /// Index keys follow attention and expert bytes the feed-forward block. A header whose every
    /// layer attends sparsely, the dense lead included, holds an index key on each of those layers
    /// and the same expert bytes as the coupled header. Mutant: the index key tied to the routed
    /// block, or the dense lead's stacks counted as experts.
    #[test]
    fn index_keys_follow_attention_and_expert_bytes_the_block() {
        let flags = vec![true; LAYERS];
        let path = header_shaped(
            "mm3-place-split",
            "minimax-m3",
            &keys_shaped(LAYERS, LEAD, Some(flags.clone())),
            &global(),
            &tensors_shaped(LAYERS, LEAD, Some(&flags)),
        );
        let split = Split::open(&path).expect("the synthetic header opens");
        let mut inputs = PlanInputs::describe(&split).expect("the split header describes");
        host_run_experts(&mut inputs.model);
        let _ = std::fs::remove_file(&path);
        let row = 2 * (16 + 16) * 2;
        let index = IDX_D * 2;
        assert_eq!(
            inputs.kv.bytes(0..LAYERS, 64),
            LAYERS as u64 * (row + index) * 64
        );
        assert!(inputs.model.tensors.iter().all(|t| {
            let routed = matches!(
                t.role,
                crate::placement::Role::RoutedExperts | crate::placement::Role::SharedExpert
            );
            !routed || t.layer.is_some_and(|l| l >= LEAD)
        }));
        let coupled = runnable("mm3-place-coupled");
        let machine = workstation::plan_gate(LAYERS);
        let levers = PlanLevers::default();
        let slots = NonZeroUsize::new(1).expect("a slot");
        let split_plan = inputs
            .plan_with_slots(&machine, 64, &levers, slots)
            .expect("the split header plans");
        let coupled_plan = coupled
            .plan_with_slots(&machine, 64, &levers, slots)
            .expect("the coupled header plans");
        let experts = |plan: &crate::placement::Plan| {
            plan.cards.iter().map(|c| c.expert_bytes).sum::<u64>()
                + plan.host.expert_bytes
                + plan.host.nvme_expert_bytes
        };
        assert_eq!(experts(&split_plan), experts(&coupled_plan));
        assert!(experts(&split_plan) > 0);
        assert_eq!(
            split_plan.cards[0].kv_bytes,
            coupled_plan.cards[0].kv_bytes + (LEAD as u64) * index * 64
        );
    }
}
