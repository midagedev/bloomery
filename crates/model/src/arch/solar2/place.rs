//! A Solar-Open2 file planned onto a machine: the common hybrid owner's plan
//! ([`place_hybrid`]) over this family's reading. What each layer holds is read off the
//! description (a KDA layer's state, a NoPE GQA layer's planes), and the card experts' formats are
//! the common leg's; the expert tier card a machine names holds the next ids of each card layer on
//! that leg. The family's own is the reading.

use std::collections::BTreeSet;

use gguf::Split;
use models::{Mixer, ModelSpec};

use super::hparams::Hparams;
use super::{roles, spec};
use crate::arch::chat_of;
use crate::arch::place_hybrid;
use crate::placement::accepts::{Program, leg_down_forms};
use crate::placement::kernels::q8_act_bytes;
use crate::placement::workstation::TierBatchBytes;
use crate::placement::{ModelTensors, PlacementError, workstation};

pub use place_hybrid::{KvLayout, PlaceError};

/// What a plan of a Solar-Open2 file is made from, read from its headers.
pub type PlanInputs = place_hybrid::PlanInputs<Hparams>;

/// The engine the plan's refusals name.
const ENGINE: &str = "Solar-Open2";

/// The rows the body's prompt batch buffers are made for: the host union's columns.
pub const PROMPT_ROWS: usize = crate::moe::UNION_MAX_COLS;

/// The flash's partial segments of a query head and the rows its partials span, as the card's
/// batch scratch sizes them; the body holds them equal to the kernels' by assertion.
pub const FLASH_SEGMENTS: usize = 80;
/// See [`FLASH_SEGMENTS`].
pub const PARTIAL_ROWS: usize = 8;
/// The columns a GEMM tile holds (the route's tile table) and the values a 32-value form's step
/// covers; the body holds them equal to the kernels' by assertion.
pub const GEMM_TILE_COLS: usize = 64;
/// See [`GEMM_TILE_COLS`].
pub const GEMM32_STEP: usize = 64;

/// Tokens a chunk of the tier's prompt-batch service runs at most: the card leg's launch group of
/// the stage card's card experts, which the body binds to its own constant.
pub const TIER_CHUNK: u64 = 8;

/// The expert tier's prompt-batch bytes ([`TierBatchBytes`]) for a described file:
/// the tier card's staging, the leg's chunk scratch and the host's rows and places, for blocks of
/// the host union's columns. The plan reserves them on the tier card and the host, and the load
/// checks its allocations against them.
pub fn tier_batch(inputs: &PlanInputs) -> Result<TierBatchBytes, PlacementError> {
    let hp = &inputs.hp;
    workstation::tier_leg_batch_bytes(
        hp.n_embd as u64,
        hp.experts.ff as u64,
        hp.experts.n_used as u64,
        TIER_CHUNK,
        crate::moe::UNION_MAX_COLS as u64,
        leg_down_forms(Program::Solar2Body, &inputs.model),
    )
}

/// The card bytes a tiered load makes beside [`prompt_reserve_bytes`]: per routed slot of the
/// prompt batch's `rows` rows (`n_used` a row), the tier place, the copy of it the join reads and
/// the copy of the slot's route weight, three four-byte words.
#[must_use]
pub fn tier_places_bytes(hp: &Hparams, rows: usize) -> u64 {
    3 * 4 * (rows * hp.experts.n_used) as u64
}

/// Card bytes of the q8 blocks of 32 of `cols` columns of `k` values (Q8_0 and Q5_0 sites' input,
/// `GemmAct32`): per column `16 + 2 + 2` words a step of [`GEMM32_STEP`] values.
const fn act32_bytes(cols: usize, k: usize) -> usize {
    4 * cols * 20 * k.div_ceil(GEMM32_STEP)
}

/// The most card bytes the forms a GEMM reads take at `k` values a column, `cols` columns: both,
/// where the width allows them. The load makes the ones its sites' types read.
const fn wide_bytes(cols: usize, k: usize) -> usize {
    let q128 = if k.is_multiple_of(256) {
        q8_act_bytes(cols as u64, k as u64) as usize
    } else {
        0
    };
    let q32 = if k.is_multiple_of(32) {
        act32_bytes(cols, k)
    } else {
        0
    };
    q128 + q32
}

/// Card bytes of the GEMM route's table of `cols` slots over one expert: the slot list, the tiles
/// (a [`GEMM_TILE_COLS`] column tile each, and the expert's), the two counts and the all-zero ids.
const fn route_bytes(cols: usize) -> usize {
    4 * (cols + 2 * (cols / GEMM_TILE_COLS + 1) + 2 + cols)
}

/// The most card bytes the single-column activation forms of one step take at width `k`: the q8_1
/// form of the K-quants and the q8 blocks of 32 of the Q8_0 and Q5_0 sites.
const fn acts_bytes(k: usize) -> usize {
    let blocks = k / 32;
    q8_act_bytes(1, k as u64) as usize + 4 * (256 * blocks.div_ceil(32) + 2 * blocks)
}

/// The card bytes the body's prompt batch makes beside the plan's terms for a file described by `hp`
/// and `spec`, at `rows` positions a unit (the batch is made once, whatever the context). The plan
/// keeps them out of the expert budget ([`place_hybrid::PlanInputs::prompt_reserve`]) and the body
/// holds its made bytes to the same figure (`Body::open_placed`). With `R = rows` and four bytes a
/// value, the terms are:
///
/// - the step's buffers at `R` rows: the residual rows `x`, `x1`, `xn` and the delta rows
///   (`4 · embd`); a KDA layer's (`2 · channels` for the qkv and conv rows, `2 · d` for the gate
///   low-rank rows, `2 · v_heads` for beta, `5 · v` for f, z, decay, o and gated, with `channels =
///   (2 · k_heads + v_heads) · d` and `v = v_heads · d`); a GQA layer's (`3 · q_width` for q, the
///   out gate and the attention, `2 · kv_width` for k and v); the flash's partials for
///   [`PARTIAL_ROWS`] rows of [`FLASH_SEGMENTS`] segments; the position and key-count rows, the
///   word record; and one column of each activation form at the widths the mixers read;
/// - the wide forms of the rows at `embd`, and at each other width the output projections and the
///   low-rank halves read (`d`, `v`, `q_width`);
/// - the host sums (`embd` a row), the KDA qkv window's staging (`2 · k_heads · d + v_heads · d` a
///   row) and the route of the one-expert table;
/// - the routed blocks' rows: the normed rows, the shared expert's output and the base
///   (`3 · embd`), the router's scores (`n_expert`), ids and weights (`2 · n_used`), the card
///   experts' batch (the leg's buffers, [`workstation::tier_chunk_scratch_bytes`]; the sum and
///   the sum plus the base, `2 · embd`; the places, `n_used`) and the shared expert over the rows
///   (its route, the forms at `embd` and at the shared width, its gate and up rows, `2 ·
///   shared_ff`).
///
/// The wide forms are a bound: both forms at each width that allows them, where the load makes
/// the ones its sites' types read. The units a prefill group past one adds are made out of the
/// card's margin.
pub fn prompt_reserve_bytes(
    hp: &Hparams,
    spec: &ModelSpec,
    model: &ModelTensors,
    rows: usize,
) -> Result<u64, PlacementError> {
    let (embd, r) = (hp.n_embd, rows);
    let (mut heads, mut kv_heads, mut head) = (hp.n_head, 0, hp.head_dim);
    let (mut k_heads, mut v_heads, mut d) = (0, 0, 0);
    for layer in &spec.layers {
        match &layer.mixer {
            Mixer::Gqa(g) => {
                heads = heads.max(g.heads as usize);
                kv_heads = kv_heads.max(g.kv_heads as usize);
                head = head.max(g.head_dim as usize).max(g.value_dim as usize);
            }
            Mixer::DeltaRule(k) => {
                k_heads = k_heads.max(k.k_heads as usize);
                v_heads = v_heads.max(k.v_heads as usize);
                d = d.max(k.d as usize);
            }
            _ => {}
        }
    }
    let (q_width, kv_width) = (heads * head, kv_heads * head);
    let (v, channels) = (v_heads * d, (2 * k_heads + v_heads) * d);
    let (experts, shared_ff) = (&hp.experts, hp.experts.shared_ff());
    let step_rows =
        4 * embd + 2 * channels + 2 * d + 2 * v_heads + 5 * v + 3 * q_width + 2 * kv_width;
    let partials = r.min(PARTIAL_ROWS) * heads * FLASH_SEGMENTS * (head + 2);
    let words = 2 * r + 1 + kv_heads;
    let host_rows = embd + 2 * k_heads * d + v_heads * d;
    let routed_rows = 3 * embd + experts.n_expert + 2 * experts.n_used + 2 * embd + experts.n_used;
    let f32s = r * (step_rows + host_rows + routed_rows + 2 * shared_ff) + partials + words;
    let others: BTreeSet<usize> = [d, v, q_width].into_iter().filter(|&k| k != embd).collect();
    let acts: usize = others.iter().chain([&embd]).map(|&k| acts_bytes(k)).sum();
    let wide: usize = others.iter().map(|&k| wide_bytes(r, k)).sum::<usize>()
        + 2 * wide_bytes(r, embd)
        + wide_bytes(r, shared_ff);
    let downs = leg_down_forms(Program::Solar2Body, model);
    let card_leg = workstation::TierLegScratch {
        n_embd: embd as u64,
        ff: experts.ff as u64,
        n_used: experts.n_used as u64,
        chunk: r as u64,
        downs,
    }
    .bytes(downs)?;
    ((4 * f32s + acts + wide + 2 * route_bytes(r)) as u64)
        .checked_add(card_leg)
        .ok_or_else(|| PlacementError::Metadata {
            key: "solar2 prompt reserve".into(),
            detail: "bytes overflow u64".into(),
        })
}

impl PlanInputs {
    /// [`PlanInputs::describe`], then the file refused with every feature the engine does not run
    /// ([`PlacementError::Unimplemented`]). The reader refuses a next-token block by its key.
    pub fn read(split: &Split) -> Result<PlanInputs, PlacementError> {
        PlanInputs::describe(split)?.runnable()
    }

    /// The file as it is, for a gate that plans or describes it whatever the coverage list holds,
    /// and the host's room and the arena's given bytes.
    pub fn describe(split: &Split) -> Result<PlanInputs, PlacementError> {
        let hp = Hparams::read(split)?;
        let model = roles::classify(split, &hp)?;
        // No parser is declared: see `spec`.
        let spec = spec::spec_of(&hp, chat_of(split, None, None)?)?;
        let reserve = prompt_reserve_bytes(&hp, &spec, &model, PROMPT_ROWS)?;
        let tier = tier_places_bytes(&hp, PROMPT_ROWS);
        Ok(PlanInputs::new(ENGINE, hp, model, spec)?
            .with_prompt_reserve(reserve)
            .with_tier_reserve(tier))
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use gguf::Split;

    use super::super::hparams::tests::{LAYERS, keys, tensors};
    use super::{
        PROMPT_ROWS, PlaceError, PlanInputs, prompt_reserve_bytes, tier_batch, tier_places_bytes,
    };
    use crate::arch::synthetic::{V, header_shaped};
    use crate::placement::{PlacementError, PlanLevers, Violation, workstation};

    /// The small header's file, described; each test names its own file (the tests run in
    /// threads of one process).
    fn described(name: &str) -> PlanInputs {
        let global = [
            ("tokenizer.ggml.pre", V::Str("solar-open")),
            ("tokenizer.chat_template", V::Str("{{ messages }}")),
        ];
        let path = header_shaped(name, "solar-open2", &keys(), &global, &tensors());
        let split = Split::open(&path).expect("the synthetic header opens");
        let inputs = PlanInputs::describe(&split).expect("the synthetic header describes");
        let _ = std::fs::remove_file(&path);
        inputs
    }

    /// The card's kv class is the layers' stores [derived from the header's widths, apart from the
    /// reader]: the 2 GQA layers (0 and 4) 2 key heads × (16 + 16) × 2 B = 128 B a position, the 6
    /// KDA layers 4 heads of 16 × 16 f32 (4,096 B) + 11 ring rows of 3 × 4 × 16 f32 (8,448 B) + a
    /// lane's u32 stamp = 12,548 B; at 64 positions 2 × 8,192 + 6 × 12,548 = 91,672 B, a slot each
    /// time. A context of no position is refused first. Mutant: the stamp left out of the state.
    #[test]
    fn the_cards_kv_class_is_the_layers_stores() {
        let inputs = described("so2-place-kv");
        let machine = workstation::plan_gate(LAYERS);
        let levers = PlanLevers::default();
        let slots = |n| NonZeroUsize::new(n).expect("a slot");
        for n in [1, 3] {
            let plan = inputs
                .plan_with_slots(&machine, 64, &levers, slots(n))
                .expect("the small header plans");
            assert_eq!(plan.cards[0].kv_bytes, n as u64 * 91_672, "{n} slots");
        }
        assert!(matches!(
            inputs.plan(&machine, 0, &levers),
            Err(PlaceError::CtxZero)
        ));
    }

    #[test]
    fn unified_room_still_refuses_an_unserved_nvme_tier() {
        use crate::placement::{Role, UnifiedPool};
        let mut inputs = described("so2-place-pool-room");
        for tensor in &mut inputs.model.tensors {
            if tensor.role == Role::RoutedExperts {
                tensor.dims[1] *= 1024;
                tensor.file_bytes *= 1024;
            }
        }
        let mut machine = workstation::plan_gate(LAYERS);
        machine.cards[0].context_bytes = 0;
        machine.cards[0].scratch_bytes = 0;
        machine.cards[0].margin_bytes = 0;
        machine.cards[0].reserves.clear();
        let levers = PlanLevers::default();
        inputs.room = (u64::MAX, workstation::HostRead::Given);
        let whole = inputs
            .plan(&machine, 64, &levers)
            .expect("the ample host fits");
        let pool = whole.cards[0].bytes()
            + inputs.prompt_reserve
            + workstation::HostNeed::of(&whole, 0).bytes()
            - 1;
        machine.host_mut().usable_bytes = pool;
        machine.host_mut().reserves.clear();
        *machine.unified_mut() = Some(UnifiedPool { bytes: pool });
        match inputs.plan(&machine, 64, &levers) {
            Err(PlaceError::NoNvmeTier { paged, .. }) => assert!(paged > 0),
            other => panic!("the unserved tier must be refused by name: {other:?}"),
        }
    }

    /// A machine that names an expert tier card is planned, the stage card keeping the tier's
    /// places of the prompt batch beside its prompt reserve: the reserve a tiered load makes is
    /// the plain one plus the places, join places and join weights of `PROMPT_ROWS` rows of the
    /// header's used experts, 12 B a slot, and the plan reaches the placement (which refuses the
    /// tier idle: the header's experts all fit the stage card). A body that serves no tier
    /// ([`PlanInputs::tier_reserve`] unset) refuses the machine by name. Mutant: the places left
    /// out of the tiered reserve, or the tier reserve left out of the description (the plan stops
    /// at [`PlaceError::Tier`]).
    #[test]
    fn a_tier_card_is_planned_with_the_batch_places_beside_the_reserve() {
        let inputs = described("so2-place-tier");
        let tiered = workstation::plan_bp(LAYERS, None, tier_batch(&inputs).expect("tier bytes"));
        let plain = workstation::plan_gate(LAYERS);
        let places = 12 * 512 * inputs.hp.experts.n_used as u64;
        assert_eq!(tier_places_bytes(&inputs.hp, PROMPT_ROWS), places);
        assert_eq!(
            inputs.reserve_on(&plain).expect("a plain machine"),
            inputs.prompt_reserve
        );
        assert_eq!(
            inputs.reserve_on(&tiered).expect("a tiered machine"),
            inputs.prompt_reserve + places
        );
        // The small header's experts all fit the stage card, so the plan passes the body's tier
        // check and meets the placement's own refusal of a tier with nothing to hold; the real
        // file's plan puts experts on the tier (`solar_open2_meta`).
        match inputs.plan(&tiered, 64, &PlanLevers::default()) {
            Err(PlaceError::Placement(PlacementError::IdleTier { tier: 0, .. })) => {}
            other => panic!("{other:?}, not the placement's idle-tier refusal"),
        }
        let mut untiered = inputs.clone();
        untiered.tier_reserve = None;
        match untiered.plan(&tiered, 64, &PlanLevers::default()) {
            Err(e @ PlaceError::Tier { tier: 0, .. }) => assert_eq!(
                e.to_string(),
                "the machine has the expert tier card 3090 (tier 0); the program this plan is \
                 for hangs no expert tier"
            ),
            other => panic!("{other:?}, not the tier refusal"),
        }
    }

    /// A card that holds the plan's terms but not the prompt batch's reserve beside them is
    /// refused by name with the bytes it is short, and one byte more of room plans: the reserve is
    /// the one the description gives ([`prompt_reserve_bytes`] at [`PROMPT_ROWS`] rows), kept
    /// out of the expert budget, so a card with no room left beside the dense terms holds no
    /// expert. Mutant: the reserve left out of the plan's call (the card plans, and its load stops
    /// at the allocation).
    #[test]
    fn a_host_only_mixture_keeps_the_exact_prompt_reserve_boundary() {
        use crate::placement::{ExpertPart, Role};
        use gguf::GgmlType::{Q5_K, Q8_0};

        let mut kv = keys();
        for (key, value) in &mut kv {
            if matches!(*key, "embedding_length" | "expert_feed_forward_length") {
                *value = V::U32(256);
            }
        }
        let mut ts = tensors();
        for (name, dims) in &mut ts {
            let tail = name.rsplit('.').nth(1).unwrap_or("");
            if matches!(
                tail,
                "ffn_gate_exps"
                    | "ffn_up_exps"
                    | "ffn_down_exps"
                    | "ffn_gate_shexp"
                    | "ffn_up_shexp"
                    | "ffn_down_shexp"
            ) {
                dims[0] = 256;
                dims[1] = 256;
            } else if tail == "attn_output" {
                dims[1] = 256;
            } else if matches!(
                tail,
                "token_embd"
                    | "output"
                    | "output_norm"
                    | "attn_norm"
                    | "ffn_norm"
                    | "attn_q"
                    | "attn_k"
                    | "attn_v"
                    | "attn_gate"
                    | "ssm_f_a"
                    | "ssm_g_a"
                    | "ssm_beta"
                    | "ffn_gate_inp"
            ) {
                dims[0] = 256;
            }
        }
        let global = [
            ("tokenizer.ggml.pre", V::Str("solar-open")),
            ("tokenizer.chat_template", V::Str("{{ messages }}")),
        ];
        let path = header_shaped("so2-place-mixed-reserve", "solar-open2", &kv, &global, &ts);
        let split = Split::open(&path).expect("the mixed fixture opens");
        let mut inputs = PlanInputs::describe(&split).expect("valid routed geometry");
        std::fs::remove_file(path).expect("remove the header fixture");
        for t in inputs
            .model
            .tensors
            .iter_mut()
            .filter(|t| t.role == Role::RoutedExperts)
        {
            t.ty = Q5_K;
            t.file_bytes = t.dims.iter().product::<u64>() / t.ty.blck_size().unwrap()
                * t.ty.type_size().unwrap();
        }
        let admitted = prompt_reserve_bytes(&inputs.hp, &inputs.spec, &inputs.model, PROMPT_ROWS)
            .expect("the admitted prompt reserve");
        let admitted_tier = tier_batch(&inputs).expect("the admitted tier reserve");
        let down = inputs
            .model
            .tensors
            .iter_mut()
            .find(|t| {
                t.role == Role::RoutedExperts
                    && t.layer == Some(0)
                    && ExpertPart::of_name(&t.name) == Some(ExpertPart::Down)
            })
            .expect("layer zero's down");
        down.ty = Q8_0;
        down.file_bytes = down.dims.iter().product::<u64>() / down.ty.blck_size().unwrap()
            * down.ty.type_size().unwrap();
        let reserve = prompt_reserve_bytes(&inputs.hp, &inputs.spec, &inputs.model, PROMPT_ROWS)
            .expect("prompt reserve");
        assert_eq!(
            reserve, admitted,
            "the host-only Q8_0 down adds no card form"
        );
        assert_eq!(
            tier_batch(&inputs).expect("mixed tier reserve"),
            admitted_tier
        );
        inputs.prompt_reserve = reserve;
        let ample = workstation::plan_gate(LAYERS);
        let placed = inputs
            .plan(&ample, 64, &PlanLevers::default())
            .expect("the mixed plan");
        assert_eq!(
            placed.n_l[0], 0,
            "Q8_0 stays on the host under Solar's rule"
        );
        assert!(
            placed.n_l[1..].iter().all(|&n| n > 0),
            "other layers remain card candidates"
        );
        let levers = PlanLevers::default();
        let mut machine = workstation::plan_gate(LAYERS);
        machine.cards[0].usable_bytes = machine.cards[0].margin_bytes;
        let short = loop {
            match inputs.plan(&machine, 64, &levers) {
                Err(PlaceError::Broken(v)) => {
                    let over = v
                        .iter()
                        .find_map(|v| match v {
                            Violation::CardOver { total, limit, .. } => Some(total - limit),
                            _ => None,
                        })
                        .unwrap_or_else(|| panic!("{v:?}, no card over its limit"));
                    machine.cards[0].usable_bytes += over;
                }
                Err(PlaceError::PromptOver {
                    reserve: got,
                    left,
                    short,
                    ..
                }) => {
                    assert_eq!((got, short), (reserve, reserve - left));
                    break short;
                }
                other => panic!("{other:?}, not the reserve's refusal"),
            }
        };
        machine.cards[0].usable_bytes += short - 1;
        assert!(matches!(
            inputs.plan(&machine, 64, &levers),
            Err(PlaceError::PromptOver { short: 1, .. })
        ));
        machine.cards[0].usable_bytes += 1;
        let plan = inputs
            .plan(&machine, 64, &levers)
            .expect("the card holds the terms and the reserve");
        assert!(plan.violations().is_empty());
    }
}
