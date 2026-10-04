//! qwen4exp MTP metadata gate: what `arch::qwen35moe::mtp::mtp_of` reads from
//! unsloth's two Qwen3.8-Flash-Next MTP draft files against the target's
//! header. One contract, one test; it prints what it compared, then fails
//! with the whole list of what differs.
//!
//! `hw_`: needs the target's shards and both draft files on the box
//! (`just gate-qwen4exp-meta`). Headers only: seconds.

#[allow(
    dead_code,
    reason = "this gate renders one layer; the rest serve the other meta gates"
)]
#[path = "common/spec_view.rs"]
mod spec_view;

use std::fmt::Write as _;

use gguf::Split;
use model::arch::models::{DraftSpec, MtpDraft, MtpSource};

/// The target: unsloth's `UD-Q4_K_XL`.
const Q38: &str = "/models/Qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf";
/// The draft that uses the target's embedding and output matrix.
const SHARED: &str = "/models/Qwen3.8-Flash-Next/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf";
/// The draft that carries its own.
const OWN: &str = "/models/Qwen3.8-Flash-Next/mtp-Qwen3.8-Flash-Next-Q8_0.gguf";

// PIN(2026-09-28): the draft's description, both files alike but for the source line [the header
// dump: block_count 49, nextn_predict_layers 1, compress_ratios[48] = 0 of 49 values, blk.48 GQA
// 24/2 x 256 with the gate in attn_q, 512/10 experts of 640 with a gated shared expert of 640].
const VIEW: &[&str] = &[
    "index 48 hidden 2560 vocab 248320 rms_eps bits 0x358637bd",
    "hc Some(HcSpec { streams: 4, kind: Gated { rank: 320 } }) input Streams head_norm HcHead head_rows Full",
    "gqa 24/2 x 256 rope Imrope { sections: [11, 11, 10, 0] } 64 base 10000000 yarn - qk_norm true out_gate true || moe 512/10 ff 640 swiglu None Softmax bias false norm true x1 hash false shared 640 swiglu None gate true || hc",
];

// PIN(2026-09-28): each file's tensor bytes, the header excluded [the header dump's TOTAL lines:
// 32 tensors and 34]; the difference is the unshared file's token_embd and output, each Q8_0
// [2560, 248320] of 675,430,400 B.
const SHARED_BYTES: u64 = 2_775_621_632;
const OWN_BYTES: u64 = 4_126_482_432;
const MATRIX_BYTES: u64 = 675_430_400;

/// The description's lines, the source's apart.
fn view(m: &MtpDraft) -> Vec<String> {
    vec![
        format!(
            "index {} hidden {} vocab {} rms_eps bits {:#010x}",
            m.index,
            m.hidden,
            m.vocab,
            m.rms_eps.to_bits()
        ),
        format!(
            "hc {:?} input {:?} head_norm {:?} head_rows {:?}",
            m.hc, m.input, m.head_norm, m.head_rows
        ),
        spec_view::layer_line(&m.layer),
    ]
}

/// The two drafts read against the Qwen3.8 target: index 48, dense; the
/// shared file borrows both matrices and the other neither; each file's
/// tensor bytes, and their difference the unshared file's own two matrices.
#[test]
#[ignore = "needs the Qwen3.8-Flash-Next shards and MTP files on the box (just gate-qwen4exp-meta)"]
fn hw_qwen4exp_mtp_spec() {
    let mut o = String::new();
    let mut b: Vec<String> = Vec::new();
    let target = Split::open(Q38).unwrap_or_else(|e| panic!("open {Q38}: {e}"));
    let spec = model::arch::spec(&target)
        .unwrap_or_else(|e| panic!("spec of {Q38}: {e}"))
        .spec;
    let mut read = |path: &str| -> (Split, MtpDraft) {
        let draft = Split::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
        let d = model::arch::qwen35moe::mtp::mtp_of(&draft, &target, &spec)
            .unwrap_or_else(|e| panic!("mtp_of {path}: {e}"));
        let DraftSpec::Mtp(m) = d else {
            panic!("{path}: not an MTP draft")
        };
        let m = *m;
        let _ = writeln!(o, "file {path}: {:?}", m.source);
        spec_view::compare(&mut o, &mut b, path, &view(&m), VIEW);
        (draft, m)
    };
    let (_, shared) = read(SHARED);
    let (own_file, own) = read(OWN);
    let mut check = |what: String, ok: bool| {
        let _ = writeln!(o, "{what}: {}", if ok { "PASS" } else { "FAIL" });
        if !ok {
            b.push(what);
        }
    };
    let source = |m: &MtpDraft| match &m.source {
        MtpSource::File {
            first_shard,
            bytes,
            borrows,
        } => (first_shard.clone(), *bytes, *borrows),
        MtpSource::InFile { layer } => {
            panic!("a draft file read as in the target's, layer {layer}")
        }
    };
    let (s_path, s_bytes, s_borrows) = source(&shared);
    let (o_path, o_bytes, o_borrows) = source(&own);
    check(
        format!("shared first shard {} is {SHARED}", s_path.display()),
        s_path.as_os_str() == SHARED,
    );
    check(
        format!("unshared first shard {} is {OWN}", o_path.display()),
        o_path.as_os_str() == OWN,
    );
    check(
        format!("shared borrows {s_borrows:?}: both"),
        s_borrows.embedding && s_borrows.head,
    );
    check(
        format!("unshared borrows {o_borrows:?}: neither"),
        !o_borrows.embedding && !o_borrows.head,
    );
    check(
        format!("shared tensor bytes {s_bytes} = {SHARED_BYTES}"),
        s_bytes == SHARED_BYTES,
    );
    check(
        format!("unshared tensor bytes {o_bytes} = {OWN_BYTES}"),
        o_bytes == OWN_BYTES,
    );
    check(
        format!(
            "difference {} = 2 x {MATRIX_BYTES}",
            o_bytes.wrapping_sub(s_bytes)
        ),
        o_bytes.checked_sub(s_bytes) == Some(2 * MATRIX_BYTES),
    );
    let matrices: Vec<u64> = ["token_embd.weight", "output.weight"]
        .iter()
        .map(|n| own_file.find(n).map_or(0, |(_, t)| t.nbytes))
        .collect();
    check(
        format!("the unshared file's token_embd and output {matrices:?} = the difference"),
        matrices == [MATRIX_BYTES, MATRIX_BYTES]
            && o_bytes.checked_sub(s_bytes) == Some(matrices.iter().sum()),
    );
    println!("{o}");
    assert!(
        b.is_empty(),
        "{} pin(s) differ:\n  {}",
        b.len(),
        b.join("\n  ")
    );
}

// PIN(2026-10-01): the shared draft's card terms [derived from the header dump's table: the Q8_0
// tensors but the injects in two planes of their file bytes, the F32 ones but the indexer's norms,
// the two injects widened to F32 [10240, 4]; 98,715,648 B dense, 2,673,868,800 B of routed experts,
// 2,772,584,448 resident; the uploads in the file's order, then the injects, through a 2 MiB-granule
// heap of their own: 2,785,017,856 B, rounding 12,433,408. The store 2 x 2 x 256 f16 = 2,048 B a
// position. The head's map 248,320 x 4 = 993,280 B, with the full head and with a list of 40,960
// rows alike: a list's rows are the target's `output`, read in place].
const DRAFT_DENSE: u64 = 98_715_648;
const DRAFT_EXPERTS: u64 = 2_673_868_800;
const DRAFT_ROUNDING: u64 = 12_433_408;
const DRAFT_KV_ROW: u64 = 2048;
const DRAFT_MAP: u64 = 993_280;
const LIST_ROWS: u32 = 40_960;
// The largest context each card holds with the draft (full head), as predicted [derived: the plan
// test's CARD_MAX_CTX budget less 2,785,017,856 B of draft granules, over 28,416 + 2,048 B a
// position plus the draft program's arena, whose dense flash partials grow 198,144 B a 64-key
// segment (3,096 B a position) — the same budget over 33,560 B a position, its 4.86 MB of fixed
// rows beside the granules; the last pool counted whole]. Printed beside the boundary the test
// finds; the clause holds the plan to that boundary.
const CARD_MAX_CTX_MTP: [(&str, u64); 2] = [("A6000", 1_194_276), ("3090", 431_309)];

/// The plan with the shared draft (`place::PlanInputs::plan_mtp`): on each
/// card at 4,096 and 32,768 positions the target's plan is `plan`'s field
/// for field, the draft's holds its 512 experts on the card with the bytes
/// derived above, and the headroom is the target's less the draft's; the
/// boundary context with the draft breaks with `CardOver`; a list of 40,960
/// rows over the target's tokenizer plans as the full head does: under
/// either expert rule (`plan_mtp_with`, `Experts::Host` and `Experts::Card`)
/// at 4,096, 4,352 and 32,768 positions, the two plans' rows, cards, host
/// and expert counts, the draft's, the map, the arena and the headroom are
/// equal; the unshared file and a list of another tokenizer are refused by
/// name.
#[test]
#[ignore = "needs the Qwen3.8-Flash-Next shards and MTP files on the box (just gate-qwen4exp-meta)"]
fn hw_qwen4exp_mtp_plan() {
    use model::arch::models::HeadRows;
    use model::arch::qwen35moe::place::{
        self, KERNEL_POSITIONS, MtpInputs, PlaceError, PlanInputs, vocab_sha256,
    };
    use model::placement::workstation::{A6000, RTX_3090};
    use model::placement::{PlanLevers, Violation};

    let mut o = String::new();
    let mut b: Vec<String> = Vec::new();
    let mut check = |o: &mut String, what: String, ok: bool| {
        let _ = writeln!(o, "{what}: {}", if ok { "PASS" } else { "FAIL" });
        if !ok {
            b.push(what);
        }
    };
    let target = Split::open(Q38).unwrap_or_else(|e| panic!("open {Q38}: {e}"));
    let draft = Split::open(SHARED).unwrap_or_else(|e| panic!("open {SHARED}: {e}"));
    let inputs = PlanInputs::describe(&target).unwrap_or_else(|e| panic!("describe: {e}"));
    let full = MtpInputs::read(&draft, &target, &inputs, HeadRows::Full)
        .unwrap_or_else(|e| panic!("MtpInputs::read {SHARED}: {e}"));
    let digest = vocab_sha256(&target).unwrap_or_else(|e| panic!("vocab_sha256: {e}"));
    let ids: Vec<u32> = (0..LIST_ROWS).map(|i| i * 6).collect();
    let list = MtpInputs::read(
        &draft,
        &target,
        &inputs,
        HeadRows::List {
            ids: ids.into(),
            digest,
        },
    )
    .unwrap_or_else(|e| panic!("MtpInputs::read with a list: {e}"));
    let levers = PlanLevers::default();
    let view = |p: &model::placement::Plan<'_>| {
        format!(
            "{:?}",
            (&p.rows, &p.cards, &p.host, p.nvme_bytes, &p.n_l, p.ctx_max)
        )
    };
    for card in [A6000, RTX_3090] {
        let machine = place::machine(card, inputs.hp.n_layer, place::UBATCH_PLANNED);
        for ctx in [4096u64, 32_768] {
            let plain = inputs
                .plan(&machine, ctx, &levers)
                .unwrap_or_else(|e| panic!("{} ctx {ctx}: plan: {e}", card.name));
            for (label, m) in [("full", &full), ("list", &list)] {
                let with = match inputs.plan_mtp(&machine, ctx, &levers, m) {
                    Ok(p) => p,
                    Err(e) => {
                        check(
                            &mut o,
                            format!("{} ctx {ctx} {label}: plan_mtp refused: {e}", card.name),
                            false,
                        );
                        continue;
                    }
                };
                let d = &with.draft.cards[0];
                let got = (
                    d.dense_bytes,
                    d.expert_bytes,
                    d.rounding_bytes,
                    d.kv_bytes,
                    with.map_bytes,
                );
                let want = (
                    DRAFT_DENSE,
                    DRAFT_EXPERTS,
                    DRAFT_ROUNDING,
                    ctx * DRAFT_KV_ROW,
                    DRAFT_MAP,
                );
                let headroom = plain.cards[0].headroom_bytes
                    - i128::from(with.draft_card_bytes() + with.arena_bytes);
                check(
                    &mut o,
                    format!(
                        "{} ctx {ctx} {label}: the target's plan is plan()'s ({}); draft \
                         (dense, experts, rounding, kv, map) {got:?} (want {want:?}); n_l {:?}; \
                         headroom {} = the target's {} less {}",
                        card.name,
                        view(&with.plan) == view(&plain),
                        with.draft.n_l,
                        with.headroom_bytes,
                        plain.cards[0].headroom_bytes,
                        with.draft_card_bytes() + with.arena_bytes
                    ),
                    view(&with.plan) == view(&plain)
                        && got == want
                        && with.draft.n_l == [512]
                        && with.headroom_bytes == headroom,
                );
            }
        }
        for experts in [place::Experts::Host, place::Experts::Card] {
            for ctx in [4096u64, 4352, 32_768] {
                let machine = place::machine_for_experts(
                    card,
                    inputs.hp.n_layer,
                    place::UBATCH_PLANNED.min(ctx),
                    experts,
                );
                let plans = [&full, &list]
                    .map(|m| inputs.plan_mtp_with(&machine, ctx, &levers, m, experts));
                let [Ok(f), Ok(l)] = &plans else {
                    let why: Vec<String> = plans
                        .iter()
                        .filter_map(|p| p.as_ref().err().map(ToString::to_string))
                        .collect();
                    check(
                        &mut o,
                        format!(
                            "{} ctx {ctx} {experts:?}: plan_mtp_with refused: {why:?}",
                            card.name
                        ),
                        false,
                    );
                    continue;
                };
                let terms = |p: &place::MtpPlan<'_>| {
                    (
                        view(&p.plan),
                        view(&p.draft),
                        p.map_bytes,
                        p.arena_bytes,
                        p.headroom_bytes,
                    )
                };
                let same = terms(f) == terms(l);
                let card_experts = |p: &place::MtpPlan<'_>| p.plan.n_l.iter().sum::<u64>();
                check(
                    &mut o,
                    format!(
                        "{} ctx {ctx} {experts:?}: the list's plan is the full head's, rows, cards, \
                         host, expert counts, draft, map, arena and headroom ({same}; card \
                         experts {} and {}, map {} and {}, headroom {} and {})",
                        card.name,
                        card_experts(f),
                        card_experts(l),
                        f.map_bytes,
                        l.map_bytes,
                        f.headroom_bytes,
                        l.headroom_bytes
                    ),
                    same,
                );
            }
        }
        let predicted = CARD_MAX_CTX_MTP
            .iter()
            .find(|(n, _)| *n == card.name)
            .map_or(0, |&(_, c)| c);
        let Ok(at) = inputs.plan_mtp(&machine, 4096, &levers, &full) else {
            check(
                &mut o,
                format!("{}: no draft plan to find the boundary from", card.name),
                false,
            );
            continue;
        };
        let (c, spec) = (&at.plan.cards[0], &machine.cards[0]);
        let fixed = c.dense_bytes
            + c.expert_bytes
            + c.rounding_bytes
            + c.scratch_bytes
            + c.context_bytes
            + at.draft_card_bytes()
            - at.draft.cards[0].kv_bytes;
        let limit = at.plan.usable_bytes(spec) - spec.margin_bytes;
        let kv = |ctx: u64| -> u64 {
            (0..inputs.hp.n_layer)
                .map(|l| model::placement::KvBytes::layer_bytes(&inputs.kv, l, ctx))
                .sum::<u64>()
                + ctx * DRAFT_KV_ROW
                + model::arch::qwen35moe::place::mtp_arena_bytes(u64::from(full.draft.vocab))
        };
        let (mut lo, mut hi) = (1u64, KERNEL_POSITIONS);
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            if fixed + kv(mid) <= limit {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let at_max = inputs.plan_mtp(&machine, lo, &levers, &full).is_ok();
        let past = matches!(
            inputs.plan_mtp(&machine, lo + 1, &levers, &full),
            Err(PlaceError::Broken(v)) if v.iter().any(|x| matches!(x, Violation::CardOver { .. }))
        );
        check(
            &mut o,
            format!(
                "{} with the draft holds ctx {lo} ({at_max}; predicted {predicted}) and breaks \
                 with CardOver at {} ({past})",
                card.name,
                lo + 1
            ),
            at_max && past,
        );
    }
    let own = Split::open(OWN).unwrap_or_else(|e| panic!("open {OWN}: {e}"));
    let refused = MtpInputs::read(&own, &target, &inputs, HeadRows::Full)
        .err()
        .map_or("read".to_string(), |e| e.to_string());
    check(
        &mut o,
        format!("the unshared file is refused by name: {refused}"),
        refused.contains("tensor token_embd.weight: is in the draft file"),
    );
    let other = MtpInputs::read(
        &draft,
        &target,
        &inputs,
        HeadRows::List {
            ids: vec![0u32, 1].into(),
            digest: [0; 32],
        },
    )
    .err()
    .map_or("read".to_string(), |e| e.to_string());
    check(
        &mut o,
        format!("a list of another tokenizer is refused by name: {other}"),
        other.contains("vocab_sha256 0000"),
    );
    println!("{o}");
    assert!(
        b.is_empty(),
        "{} pin(s) differ:\n  {}",
        b.len(),
        b.join("\n  ")
    );
}

// PIN(2026-10-02): plan (b′) (`place::machine_bp`, `--place bp`) with the shared draft (full head)
// at 4,096 positions and U 4,096: the A6000 keeps plan (a)'s drafted row at its budget less the tier
// join's card rows (`place::card_tier_join_bytes`) byte for byte — 242 on the first 42 layers, 241
// on the rest, 11,610 experts, 72 fewer than the row before the join's 244/18/243 — the draft's card
// bytes the A6000's "MTP draft" reserve, which the tier's budget does not see; the split's
// derivation is qwen4exp_meta's BP_PLAIN [derived: the tier's budget is BP_PLAIN's, 22,826,941,416
// B; the same spread beside 242/241: both cards 393 on the first 30 layers and 392 on the rest, so
// the tier holds 151 on layers 0-29 and 42-47 and 150 on 30-41, 7,236 experts, 22,676,889,600 B of
// experts and 146,415,616 B of rounding; the host 119 or 120 a layer, 5,730 experts]. Row: the
// A6000's (high, at_high, low), both cards' (high, at_high), the tier's experts, expert bytes,
// rounding.
// PIN(2026-10-04): re-pinned for the decode flash's fixed segment count (`flash_gqa::SEGMENTS`, 80
// at every cache height): the draft program's arena (`place::mtp_arena_bytes`) grows 3,170,304 B at
// ctx 4,096 (64 -> 80 segments of 49,536 f32), so the A6000's "MTP draft" reserve grows by it and
// plan (a)'s drafted row, qwen4exp_meta's CARD_PLANS A6000 4k U 4,096 row, loses one expert
// (244/17/243): the A6000 242 on the first 41 layers, 241 on the rest, 11,609 experts; both cards'
// spread keeps the tier's 7,236 experts, bytes and rounding, now 393 on the first 29: the tier holds
// 151 on layers 0-28 and 41-47 and 150 on 29-40, the host 5,731 [predicted: one expert off the
// A6000 (3,170,304 B against 3,072,000 B an expert and the granules' slack) and the tier's terms
// unchanged; the box's planner on this tree: as predicted, the old pins red on exactly the A6000's
// split and the tier's].
const BP_DRAFTED: (u64, usize, u64, u64, usize, u64, u64, u64) =
    (242, 41, 241, 393, 29, 7_236, 22_676_889_600, 146_415_616);

/// Plan (b′) of the Qwen3.8 file with the shared draft (`place::machine_bp`
/// with `MtpInputs::card_bytes` as the A6000's reserve, `plan_mtp_with`
/// under `Experts::Card`): the A6000's counts, card bytes and headroom plan
/// (a)'s drafted plan's, its reserve the draft's card bytes, the draft's
/// plan (a)'s; the 3090 tier's per layer both cards' totals less the
/// A6000's, its expert and rounding bytes as predicted; a plan with the
/// draft on a machine that reserves nothing for it, or another amount, is
/// refused by name.
#[test]
#[ignore = "needs the Qwen3.8-Flash-Next shards and MTP files on the box (just gate-qwen4exp-meta)"]
fn hw_qwen4exp_bp_mtp_plan() {
    use model::arch::models::HeadRows;
    use model::arch::qwen35moe::place::{
        self, Experts, MtpInputs, PlaceError, PlanInputs, machine_bp, tier_batch,
    };
    use model::placement::PlanLevers;
    use model::placement::workstation::A6000;

    let mut o = String::new();
    let mut b: Vec<String> = Vec::new();
    let mut check = |o: &mut String, what: String, ok: bool| {
        let _ = writeln!(o, "{what}: {}", if ok { "PASS" } else { "FAIL" });
        if !ok {
            b.push(what);
        }
    };
    let target = Split::open(Q38).unwrap_or_else(|e| panic!("open {Q38}: {e}"));
    let draft = Split::open(SHARED).unwrap_or_else(|e| panic!("open {SHARED}: {e}"));
    let inputs = PlanInputs::describe(&target).unwrap_or_else(|e| panic!("describe: {e}"));
    let mtp = MtpInputs::read(&draft, &target, &inputs, HeadRows::Full)
        .unwrap_or_else(|e| panic!("MtpInputs::read {SHARED}: {e}"));
    let (ctx, u) = (4096u64, 4096u64);
    let n = inputs.hp.n_layer;
    let levers = PlanLevers::default();
    let batch = tier_batch(&inputs.hp, u);
    let bytes = mtp
        .card_bytes(ctx)
        .unwrap_or_else(|e| panic!("card_bytes: {e}"));
    let machine = machine_bp(n, u, Some(bytes), batch);
    // Plan (a)'s rule at the A6000's budget less the tier join's scratch
    // (`place::card_tier_join_bytes`), which plan (b′) adds to its scratch.
    let mut a_machine = place::machine_for_experts(A6000, n, u, Experts::Card);
    a_machine.cards[0].scratch_bytes += place::card_tier_join_bytes(u);
    let a = inputs
        .plan_mtp_with(&a_machine, ctx, &levers, &mtp, Experts::Card)
        .unwrap_or_else(|e| panic!("plan (a) with the draft: {e}"));
    let view = |p: &model::placement::Plan<'_>| {
        format!(
            "{:?}",
            (&p.rows, &p.cards, &p.host, p.nvme_bytes, &p.n_l, p.ctx_max)
        )
    };
    let (high, at_high, low, t_high, t_at_high, t_experts, t_bytes, t_rounding) = BP_DRAFTED;
    match inputs.plan_mtp_with(&machine, ctx, &levers, &mtp, Experts::Card) {
        Err(e) => check(
            &mut o,
            format!("plan (b′) with the draft refused: {e}"),
            false,
        ),
        Ok(m) => {
            let plan = &m.plan;
            let stage: Vec<u64> = (0..n)
                .map(|l| if l < at_high { high } else { low })
                .collect();
            let total: Vec<u64> = (0..n)
                .map(|l| if l < t_at_high { t_high } else { t_high - 1 })
                .collect();
            let tier_want: Vec<u64> = total.iter().zip(&stage).map(|(t, s)| t - s).collect();
            let tier = plan.tier_n_l.first().cloned().unwrap_or_default();
            let (s, sa, t) = (&plan.cards[0], &a.plan.cards[0], &plan.cards[1]);
            let stage_same = (s.dense_bytes, s.expert_bytes, s.rounding_bytes, s.kv_bytes)
                == (
                    sa.dense_bytes,
                    sa.expert_bytes,
                    sa.rounding_bytes,
                    sa.kv_bytes,
                )
                && plan.n_l == a.plan.n_l
                && m.headroom_bytes == a.headroom_bytes
                && view(&m.draft) == view(&a.draft)
                && s.reserve_bytes == m.draft_card_bytes() + m.arena_bytes
                && s.reserve_bytes == bytes;
            check(
                &mut o,
                format!(
                    "the A6000 {high} on the first {at_high}, {low} on the rest ({}: {:?}), plan \
                     (a)'s drafted counts, card bytes, headroom and draft plan, its reserve {} the \
                     draft's {bytes} ({stage_same}); headroom {} ((a) {})",
                    plan.n_l == stage,
                    plan.n_l,
                    s.reserve_bytes,
                    m.headroom_bytes,
                    a.headroom_bytes
                ),
                plan.n_l == stage && stage_same,
            );
            let held: u64 = tier.iter().sum();
            check(
                &mut o,
                format!(
                    "the 3090 tier: both cards {t_high} on the first {t_at_high}, {} on the rest, \
                     so the tier {tier:?} (want {tier_want:?}); {held} experts (want {t_experts}), \
                     expert bytes {} (want {t_bytes}), rounding {} (want {t_rounding}); host {} \
                     experts; tier headroom {}",
                    t_high - 1,
                    t.expert_bytes,
                    t.rounding_bytes,
                    plan.host.experts,
                    t.headroom_bytes
                ),
                tier == tier_want
                    && held == t_experts
                    && (t.expert_bytes, t.rounding_bytes) == (t_bytes, t_rounding)
                    && plan.host.experts
                        == n as u64 * inputs.model.experts - plan.n_l.iter().sum::<u64>() - held,
            );
        }
    }
    for (label, reserve) in [("none", None), ("one byte more", Some(bytes + 1))] {
        let wrong = machine_bp(n, u, reserve, batch);
        let got = inputs.plan_mtp_with(&wrong, ctx, &levers, &mtp, Experts::Card);
        check(
            &mut o,
            format!(
                "the draft on a tier machine reserving {label}: {:?}",
                got.as_ref().err().map(ToString::to_string)
            ),
            matches!(got, Err(PlaceError::DraftReserve { .. })),
        );
    }
    println!("{o}");
    assert!(
        b.is_empty(),
        "{} clause(s) failed:\n  {}",
        b.len(),
        b.join("\n  ")
    );
}
