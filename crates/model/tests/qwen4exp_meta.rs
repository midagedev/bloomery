//! qwen4exp metadata gate: what the header reader `arch::qwen35moe` resolves
//! from the Qwen3.8-Flash-Next file, and what the coverage check lists for it
//! — the parts no program in this tree runs yet, the work queue of the rounds
//! that build one. One contract, one test; it prints what it compared, then
//! fails with the whole list of what differs.
//!
//! `hw_`: needs the four shards on the box (`just gate-qwen4exp-meta`).
//! Headers only: seconds.

#[path = "common/spec_fail_first.rs"]
mod spec_fail_first;
#[path = "common/spec_view.rs"]
mod spec_view;

use std::collections::BTreeMap;
use std::fmt::Write as _;

use gguf::Split;

/// The file: unsloth's `UD-Q4_K_XL`. Its first shard holds the header and no
/// tensor.
const Q38: &str = "/models/Qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf";

// PIN(2026-09-27): the file's description as `arch::qwen35moe::spec` reads it, line by line (the key
// values checked against the header dump and llama.cpp's loader, src/models/qwen4exp.cpp:26-147).
const VIEW: &[&str] = &[
    "arch Qwen4Exp",
    "hidden 2560 vocab 248320 ctx_train 262144",
    "rms_eps bits 0x358637bd",
    "layers 48 mtp 0",
    "hc Some(HcSpec { streams: 4, kind: Gated { rank: 320 } })",
    "engram Some(EngramSpec { heads: 8, max_ngram: 3, key_length: 160, rule: Ple { eos: 248044, image: Some(248056), conv: 4, dilation: 3 } })",
    "chat pre qwen35 template bytes 9993 tools None reasoning None",
    "[0,2,4-6,8-10,12-14,16-18,20-22,24-26,28-30,32-34,36-38,40-42,44-46] delta Gdn { khead_map: Tiled, gate: Sigmoid } k 16 v 48 d 128 conv 4 || moe 512/10 ff 640 swiglu None Softmax bias false norm true x1 hash false shared 640 swiglu None gate true || hc",
    "[1] delta Gdn { khead_map: Tiled, gate: Sigmoid } k 16 v 48 d 128 conv 4 || moe 512/10 ff 640 swiglu None Softmax bias false norm true x1 hash false shared 640 swiglu None gate true || hc [Ple]",
    "[3,7,11,15,19,23,27,31,35,39,43,47] gqa 24/2 x 256 rope Imrope { sections: [11, 11, 10, 0] } 64 base 10000000 yarn - qk_norm true out_gate true | TokenPool { heads: 4, d: 128, top_k: 2048, pool: 4, rule: Mean { rope: Rope { mode: Imrope { sections: [11, 11, 10, 0] }, dims: 64, base: 10000000.0, yarn: None } } } || moe 512/10 ff 640 swiglu None Softmax bias false norm true x1 hash false shared 640 swiglu None gate true || hc",
];

// PIN(2026-09-27): the keys the reader took a default for: the router's constants, which the file
// does not carry.
const DEFAULTS: &[&str] = &[
    "expert_gating_func = softmax (qwen4exp.cpp:992-1001)",
    "expert_weights_norm = true (qwen4exp.cpp:992-1001)",
    "expert_weights_scale = 1 (qwen4exp.cpp:992-1001)",
];

// PIN(2026-09-27): every tensor by role and type (1,224 in all, none in the first shard); the PLE
// table is the site's, the output hyper-connection head the head's.
const TENSORS: &[&str] = &[
    "attn bf16 24",
    "attn f32 264",
    "attn q8_0 156",
    "hc f32 192",
    "hc q8_0 192",
    "router f32 48",
    "shexp f32 48",
    "shexp q8_0 144",
    "engram_dense q8_0 2",
    "engram_gain f32 4",
    "engram_table iq4_nl 1",
    "routed q4_K 94",
    "routed q5_1 43",
    "routed q5_K 2",
    "routed q8_0 5",
    "token_embd q8_0 1",
    "head f32 1",
    "head q8_0 3",
];

// PIN(2026-09-27): the coverage check's list (feature: layers). No program runs qwen4exp, so a
// type any program's pin reads (the q8_0 attention matrices) is not an item. The routed q5_1 and
// q8_0 stacks are not items either: the check counts a card format (`CardFormat::of_routed`), not
// a card expert kernel for it, and no card `_sel` reads either type yet. No pin reads the PLE
// table's type. The gated-residual hyper-connections, the mean-pool indexer and the PLE site are
// needs of their own, which no row for mHC, the token pool or the engram covers.
// PIN(2026-09-27): the router and the GQA flash rows left the list when the shape registry gained
// their serving rows (shapes p1-4: the `_512` router and the `_256_p4` flash entries serve them).
// PIN(2026-09-27): five items left the list with GLM-5.3-Flash's first program, whose rows and type
// pins the whole-tree listing now counts: the q8_0 output head, token embedding and hyper-connection
// fn (its pins read q8_0), the recurrent-state slot (`gpu-glm5next/src/kda.rs`) and the program that
// runs delta-rule and attention layers in one trunk (`gpu-glm5next/src/program.rs`).
// PIN(2026-09-27): twelve items left the list with qwen4exp's own program (`Body38`, q38prog), which
// the check now holds the file to alone: its rows cover the GDN, the shared expert, the gated
// hyper-connections and their head, the PLE site, the QK norm and rope, the output gate, the
// mean-pool indexer, the image placeholder's refusal; its type pins read the q8_0 and bf16
// attention matrices; the routed q5_K stacks are served on the host (`CardFormat::of`). The two
// left are the chat surface's, which the program does not use (`Body38`'s `ALLOWED`).
// PIN(2026-09-30): the pre-tokenizer item left with the tokenizer's qwen35 pre-tokenizer
// (`crates/tokenizer/src/pretok.rs`); the tool-call parser is the one item left.
const COVERAGE: &[&str] = &["a tool-call parser for this template"];

#[test]
#[ignore = "needs the Qwen3.8-Flash-Next shards on the box (just gate-qwen4exp-meta)"]
fn hw_qwen4exp_spec() {
    let mut o = String::new();
    let mut b = Vec::new();
    let split = Split::open(Q38).unwrap_or_else(|e| panic!("open {Q38}: {e}"));
    let read = model::arch::spec(&split).unwrap_or_else(|e| panic!("spec of {Q38}: {e}"));
    let _ = writeln!(o, "file {Q38}");
    spec_view::compare(
        &mut o,
        &mut b,
        "description",
        &spec_view::view(&read.spec),
        VIEW,
    );
    spec_view::compare(&mut o, &mut b, "defaults taken", &read.defaults, DEFAULTS);
    let mut by_role: BTreeMap<(String, String), usize> = BTreeMap::new();
    for t in &read.tensors.tensors {
        *by_role
            .entry((t.role.to_string(), t.ty.to_string()))
            .or_default() += 1;
    }
    let tensors: Vec<String> = by_role
        .iter()
        .map(|((role, ty), n)| format!("{role} {ty} {n}"))
        .collect();
    let _ = writeln!(o, "tensors {}", read.tensors.tensors.len());
    spec_view::compare(
        &mut o,
        &mut b,
        "tensors by role and type",
        &tensors,
        TENSORS,
    );
    let list = model::arch::coverage::check(&read.spec, &read.tensors);
    spec_view::compare(
        &mut o,
        &mut b,
        "coverage",
        &spec_view::items(&list),
        COVERAGE,
    );
    for (at, want) in [
        (
            "gpu/src/ple.rs (gate, conv; Body38's host rows)",
            "a PLE site: a gate of 2560-value rows, a conv of 4 taps 3 apart: 1",
        ),
        (
            "gpu/src/arch/qwen3moe/program38.rs shared (q38.rs q38_shared_add)",
            "shared expert, ff 640, with a sigmoid gate: 0-47",
        ),
    ] {
        spec_fail_first::fail_first(&mut o, &mut b, &read.spec, &read.tensors, at, want);
    }
    let err = model::placement::PlacementError::Unimplemented(list);
    let _ = writeln!(o, "as the engine refuses it: {err}");
    println!("{o}");
    assert!(
        b.is_empty(),
        "{} pin(s) differ:\n  {}",
        b.len(),
        b.join("\n  ")
    );
}

// PIN(2026-09-27): the plan's card and host figures [derived from the header dump: every
// non-routed tensor but the PLE table on the card in its CardFormat (q8_0 two planes of the file's
// bytes, f32 as is, bf16 widened to f32), its buffers through a 2 MiB-granule heap in the file's
// tensor order; the routed stacks, 1,224 − 1,080 = 144 of them, and the PLE table on the host in the
// file's bytes]. The cache at 4,096 and 32,768 positions: 36 GDN layers of 4 · 3,145,728 + 16 B of
// state and 11 · 10,240 f32 of conv ring, the PLE ring 17 · 10,240 f32, 12 attention layers of
// 2,304 B a position and 256 B a pool of four.
const CARD_DENSE: u64 = 5_544_906_240;
const CARD_ROUNDING: u64 = 398_422_528;
const HOST_EXPERTS: u64 = 77_017_907_200;
// PIN(2026-09-28): the PLE table on the host, in the host set a load reads in, where the NVMe tier
// held it [derived: 320,001,536 IQ4_NL rows of 160 values, 160 / 32 · 18 = 90 B a row; the bytes the
// NVMe figure pinned, moved to the host's tables, the NVMe tier now 0].
const HOST_TABLES: u64 = 28_800_138_240;
// PIN(2026-09-28): KV_AT and CARD_MAX_CTX past the verify's delta lanes [derived: each GDN layer's
// state keeps runtime::stores::DELTA_LANES = 4 lanes with a u32 stamp each, 3 · 3,145,728 + 16 =
// 9,437,200 B past the one lane, 339,739,200 B over the 36 layers; KV_AT was 246,554,624 and
// 1,061,298,176, CARD_MAX_CTX 1,520,312 and 619,339, each now 339,739,200 / 28,416 = 11,955.9
// positions lower, rounded to 11,956 (one fewer when the old boundary's slack passed 26,000 B)].
const KV_AT: [(u64, u64); 2] = [(4096, 586_293_824), (32_768, 1_401_037_376)];
// The largest context each card holds, as predicted [derived: (usable − margin − the dense
// granules − context − scratch − 469,901,888 B of recurrent bytes) over 28,416 B a position, the
// last pool counted whole]. Printed beside the boundary the test finds from the plan's own totals;
// the clause holds the placement to that boundary, not to this figure. PIN(2026-09-28): the
// scratch counts the ubatch walk's at U = 4,096 [derived: CARD_SCRATCH's 4,096 row − 64 MiB =
// 2,770,817,024 B over 28,416 B a position, 97,509.05 positions, rounded up to 97,510].
const CARD_MAX_CTX: [(&str, u64); 2] = [("A6000", 1_410_846), ("3090", 509_873)];
// PIN(2026-09-28): the card's scratch, the m = 1 scratch and the ubatch walk's at 4,096
// positions [derived: 64 MiB + place::ubatch_scratch_bytes(4096) = 67,108,864 + 4,096 ·
// 668,277 + 32 MiB].
// PIN(2026-09-28): and at the U the load runs, which `place::machine` now takes: at 512 positions
// [derived: 67,108,864 + 512 · 668,277 + 33,554,432 = 442,821,120].
const CARD_SCRATCH: [(u64, u64); 2] = [(4096, 2_837_925_888), (512, 442_821_120)];

// PIN(2026-09-29): the card scratch of a card-experts machine
// (`place::machine_for_experts` under `Experts::Card`): the host rows' plus
// the ubatch walk's card route [derived:
// `place::card_route_scratch_bytes`: at 4,096 positions
// 2,048 · 170,360 + 4,096 · 10,240 = 390,840,320; at 512
// 512 · 170,360 + 512 · 10,240 = 92,467,200 — the run-token bound's
// derivation is `place::CARD_ROUTE_RUN_TOKEN_BYTES`'s].
const CARD_ROUTE_SCRATCH: [(u64, u64); 2] = [(4096, 3_228_766_208), (512, 535_288_320)];

/// The card scratch [`CARD_SCRATCH`] pins at ubatch `u` (a host-routed plan).
fn scratch_at(u: u64) -> u64 {
    CARD_SCRATCH
        .iter()
        .find(|&&(x, _)| x == u)
        .map_or_else(|| panic!("CARD_SCRATCH has no row for U {u}"), |&(_, b)| b)
}

/// The card scratch [`CARD_ROUTE_SCRATCH`] pins at ubatch `u`.
fn card_scratch_at(u: u64) -> u64 {
    CARD_ROUTE_SCRATCH
        .iter()
        .find(|&&(x, _)| x == u)
        .map_or_else(
            || panic!("CARD_ROUTE_SCRATCH has no row for U {u}"),
            |&(_, b)| b,
        )
}

/// The plan of the Qwen3.8 file from its headers (`arch::qwen35moe::place`):
/// `PlanInputs::read` refuses it with the coverage list; `describe` reads
/// it, and on each card at 4,096 and 32,768 positions the plan places every
/// tensor once — the routed stacks whole and the PLE table on the host,
/// everything else on the card — with the card, host and cache bytes
/// predicted and the host's and the NVMe tier's totals those of its rows;
/// the host set of the plan holds the PLE table's pages once, beside the
/// routed stacks' (`HostSet::of`, headers only); every name the program
/// reads is in the file on the layers that carry it; a context of 0 or past
/// `u32` positions is refused by name, and one past what a card holds breaks
/// the plan with `CardOver`.
#[test]
#[ignore = "needs the Qwen3.8-Flash-Next shards on the box (just gate-qwen4exp-meta)"]
fn hw_qwen4exp_plan() {
    use model::arch::qwen35moe::hparams::Kind;
    use model::arch::qwen35moe::names::{self, Sub};
    use model::arch::qwen35moe::place::{self, KERNEL_POSITIONS, PlaceError, PlanInputs};
    use model::placement::host_lock::{HostFile, HostSet, page_bytes};
    use model::placement::workstation::{A6000, RTX_3090};
    use model::placement::{Device, KvBytes, ModelTensor, PlanLevers, Role, Violation};
    use model::r8file::R8Source;

    let mut o = String::new();
    let mut b: Vec<String> = Vec::new();
    let mut check = |o: &mut String, what: String, ok: bool| {
        let _ = writeln!(o, "{what}: {}", if ok { "PASS" } else { "FAIL" });
        if !ok {
            b.push(what);
        }
    };
    let split = Split::open(Q38).unwrap_or_else(|e| panic!("open {Q38}: {e}"));
    let refused = match PlanInputs::read(&split) {
        Err(model::placement::PlacementError::Unimplemented(list)) => spec_view::items(&list),
        other => panic!("PlanInputs::read: want the coverage refusal, got {other:?}"),
    };
    check(
        &mut o,
        format!(
            "read refuses the file with the coverage list: {} items (want {})",
            refused.len(),
            COVERAGE.len()
        ),
        refused
            .iter()
            .map(String::as_str)
            .eq(COVERAGE.iter().copied()),
    );
    let inputs = PlanInputs::describe(&split).unwrap_or_else(|e| panic!("describe: {e}"));
    let hp = &inputs.hp;
    let file: std::collections::HashSet<&str> = inputs
        .model
        .tensors
        .iter()
        .map(|t| t.name.as_str())
        .collect();
    let mut missing = Vec::new();
    let mut want = |name: String| {
        if !file.contains(name.as_str()) {
            missing.push(name);
        }
    };
    for n in [
        names::token_embd(),
        names::output(),
        names::output_hc_norm(),
        names::output_hc_down(),
        names::output_hc_up(),
        names::per_layer_token_embd(),
    ] {
        want(n);
    }
    let ple = hp.exp.as_ref().and_then(|e| e.ple).map(|p| p.layer);
    for (l, kind) in hp.kinds.iter().enumerate() {
        for sub in [Sub::Attn, Sub::Ffn] {
            want(names::hc_norm(l, sub));
            want(names::hc_down(l, sub));
            want(names::hc_up(l, sub));
            want(names::hc_inject(l, sub));
        }
        for f in [
            names::ffn_gate_inp,
            names::ffn_gate_inp_shexp,
            names::ffn_gate_shexp,
            names::ffn_up_shexp,
            names::ffn_down_shexp,
            names::ffn_gate_exps,
            names::ffn_up_exps,
            names::ffn_down_exps,
        ] {
            want(f(l));
        }
        let mixer: &[fn(usize) -> String] = match kind {
            Kind::DeltaRule => &[
                names::attn_qkv,
                names::attn_gate,
                names::ssm_conv1d,
                names::ssm_dt_bias,
                names::ssm_a,
                names::ssm_alpha,
                names::ssm_beta,
                names::ssm_norm,
                names::ssm_out,
            ],
            Kind::Attention => &[
                names::attn_q,
                names::attn_k,
                names::attn_v,
                names::attn_q_norm,
                names::attn_k_norm,
                names::attn_output,
                names::indexer_q_proj,
                names::indexer_k_proj,
                names::indexer_q_norm,
                names::indexer_k_norm,
            ],
        };
        for f in mixer {
            want(f(l));
        }
        if ple == Some(l) {
            for f in [
                names::ple_key,
                names::ple_value,
                names::ple_norm_key,
                names::ple_norm_query,
                names::ple_norm_conv,
                names::ple_conv1d,
            ] {
                want(f(l));
            }
        }
    }
    check(
        &mut o,
        format!("every name the program reads is in the file: missing {missing:?}"),
        missing.is_empty(),
    );
    let levers = PlanLevers::default();
    for card in [A6000, RTX_3090] {
        let machine = place::machine(card, hp.n_layer, place::UBATCH_PLANNED);
        for (ctx, kv) in KV_AT {
            let plan = match inputs.plan(&machine, ctx, &levers) {
                Ok(p) => p,
                Err(e) => {
                    check(
                        &mut o,
                        format!("{} ctx {ctx}: plan refused: {e}", card.name),
                        false,
                    );
                    continue;
                }
            };
            let mut once = vec![0usize; inputs.model.tensors.len()];
            let mut misplaced = Vec::new();
            for r in &plan.rows {
                once[r.tensor] += 1;
                let t = &inputs.model.tensors[r.tensor];
                let want = match t.role {
                    Role::RoutedExperts | Role::EngramTable => Device::Host,
                    _ => Device::Card(0),
                };
                if r.segments.len() != 1 || r.segments[0].device != want {
                    misplaced.push(t.name.clone());
                }
            }
            let placed_once = once.iter().all(|&n| n == 1);
            let c = &plan.cards[0];
            let figures = [
                ("card dense", c.dense_bytes, CARD_DENSE),
                ("card rounding", c.rounding_bytes, CARD_ROUNDING),
                ("card experts", c.expert_bytes, 0),
                ("card kv", c.kv_bytes, kv),
                (
                    "card scratch",
                    c.scratch_bytes,
                    scratch_at(place::UBATCH_PLANNED),
                ),
                ("host experts", plan.host.expert_bytes, HOST_EXPERTS),
                ("host tables", plan.host.table_bytes, HOST_TABLES),
                ("nvme", plan.nvme_bytes, 0),
            ];
            let bytes_ok = figures.iter().all(|&(_, got, want)| got == want);
            let shown: Vec<String> = figures
                .iter()
                .map(|(what, got, want)| format!("{what} {got} (want {want})"))
                .collect();
            check(
                &mut o,
                format!(
                    "{} ctx {ctx}: every tensor placed once {placed_once}, by its role (misplaced \
                     {misplaced:?}); n_l all 0 {}; {}; headroom {}",
                    card.name,
                    plan.n_l.iter().all(|&n| n == 0),
                    shown.join(", "),
                    c.headroom_bytes
                ),
                placed_once && misplaced.is_empty() && plan.n_l.iter().all(|&n| n == 0) && bytes_ok,
            );
            // The host's and the NVMe tier's totals re-derived from the rows, whichever step of
            // the placement wrote them.
            let (mut host_rows, mut nvme_rows) = (0u64, 0u64);
            for seg in plan.rows.iter().flat_map(|r| &r.segments) {
                match seg.device {
                    Device::Host => host_rows += seg.resident_bytes,
                    Device::Nvme => nvme_rows += seg.resident_bytes,
                    Device::Card(_) | Device::Unused => {}
                }
            }
            let h = &plan.host;
            let terms = h.expert_bytes + h.table_bytes + h.shadow_bytes + h.reserve_bytes;
            let headroom = i128::from(machine.host.usable_bytes) - i128::from(terms);
            check(
                &mut o,
                format!(
                    "{} ctx {ctx}: the host's rows {host_rows} = experts + tables {}; the NVMe \
                     tier's rows {nvme_rows} = {}; host headroom {} = usable − its terms {headroom}",
                    card.name,
                    h.expert_bytes + h.table_bytes,
                    plan.nvme_bytes,
                    h.headroom_bytes
                ),
                host_rows == h.expert_bytes + h.table_bytes
                    && nvme_rows == plan.nvme_bytes
                    && h.headroom_bytes == headroom,
            );
        }
        let predicted = CARD_MAX_CTX
            .iter()
            .find(|(n, _)| *n == card.name)
            .map_or(0, |&(_, c)| c);
        // The boundary from the plan's own totals: the card's granules, scratch and
        // context, and the cache the layout gives each context, against usable − margin.
        let Ok(at) = inputs.plan(&machine, KV_AT[0].0, &levers) else {
            check(
                &mut o,
                format!("{}: no plan to find the boundary from", card.name),
                false,
            );
            continue;
        };
        let (c, spec) = (&at.cards[0], &machine.cards[0]);
        let fixed =
            c.dense_bytes + c.expert_bytes + c.rounding_bytes + c.scratch_bytes + c.context_bytes;
        let limit = at.usable_bytes(spec) - spec.margin_bytes;
        let kv = |ctx: u64| -> u64 { (0..hp.n_layer).map(|l| inputs.kv.layer_bytes(l, ctx)).sum() };
        let (mut lo, mut hi) = (1u64, KERNEL_POSITIONS);
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            if fixed + kv(mid) <= limit {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let max = lo;
        let at_max = inputs.plan(&machine, max, &levers).is_ok();
        let past = matches!(
            inputs.plan(&machine, max + 1, &levers),
            Err(PlaceError::Broken(v)) if v.iter().any(|x| matches!(x, Violation::CardOver { .. }))
        );
        check(
            &mut o,
            format!(
                "{} holds ctx {max} ({at_max}; predicted {predicted}) and breaks with CardOver \
                 at {} ({past})",
                card.name,
                max + 1
            ),
            at_max && past,
        );
    }
    let machine = place::machine(A6000, hp.n_layer, place::UBATCH_PLANNED);
    // The plan's host set, from the headers: the PLE table's pages once, beside the routed
    // stacks', each run grown to whole pages and merged per shard.
    match inputs.plan(&machine, KV_AT[0].0, &levers) {
        Ok(plan) => {
            let src = R8Source::rows(&split);
            let set = |keep: &dyn Fn(&ModelTensor) -> bool| {
                HostSet::of(src, &plan, keep).unwrap_or_else(|e| panic!("HostSet::of: {e}"))
            };
            let all = set(&|_| true);
            let stacks = set(&|t| t.role != Role::EngramTable);
            let page = page_bytes().unwrap_or_else(|e| panic!("page_bytes: {e}"));
            let name = names::per_layer_token_embd();
            let (s, info) = split
                .find(&name)
                .unwrap_or_else(|| panic!("{name} is not in the file"));
            let at = split
                .shard(s)
                .unwrap_or_else(|| panic!("shard {s} of {name}"))
                .data_base()
                + info.offset;
            let ple = at / page..(at + info.nbytes).div_ceil(page);
            let runs = |set: &HostSet| -> Vec<std::ops::Range<u64>> {
                set.runs()
                    .into_iter()
                    .find(|(f, _)| *f == HostFile::Shard(s))
                    .map(|(_, r)| r.to_vec())
                    .unwrap_or_default()
            };
            let held = runs(&all)
                .iter()
                .any(|r| r.start <= ple.start && ple.end <= r.end);
            let shared: u64 = runs(&stacks)
                .iter()
                .map(|r| r.end.min(ple.end).saturating_sub(r.start.max(ple.start)))
                .sum();
            let want = stacks.pages() + (ple.end - ple.start) - shared;
            check(
                &mut o,
                format!(
                    "the host set holds the PLE table's pages {ple:?} of shard {s} ({held}), once: \
                     {} pages = the routed stacks' {} + the table's {} − {shared} shared (want \
                     {want})",
                    all.pages(),
                    stacks.pages(),
                    ple.end - ple.start
                ),
                held && all.pages() == want,
            );
        }
        Err(e) => check(
            &mut o,
            format!("no plan to build the host set from: {e}"),
            false,
        ),
    }
    for ctx in [0, KERNEL_POSITIONS + 1] {
        let got = inputs.plan(&machine, ctx, &levers);
        let named = matches!(&got, Err(PlaceError::Positions { ctx_max }) if *ctx_max == ctx);
        let text = got.err().map_or("planned".to_string(), |e| e.to_string());
        check(&mut o, format!("ctx {ctx}: {text}"), named);
    }
    println!("{o}");
    assert!(
        b.is_empty(),
        "{} clause(s) failed:\n  {}",
        b.len(),
        b.join("\n  ")
    );
}

// PIN(2026-09-28): the card rule's plan (`place::Experts::Card`) [derived: the id prefix spread
// over the 43 layers whose gate, up and down are q4_K, q4_K and q5_1 (`place::card_routed`), an
// expert 3,072,000 B on the card as in the file — a gate and an up of 640 rows of ten 144 B
// super-blocks, 921,600 B each, a down of 2,560 rows of twenty 24 B blocks, 1,228,800 B — each
// stack's buffer in whole 2 MiB granules after the dense ones (CARD_DENSE + CARD_ROUNDING); the
// budget usable − the cache − context − scratch at U (CARD_SCRATCH) − margin, with the draft less
// its 2,785,017,856 B of granules and 2,048 B a position of store (the full head). The spread stops
// at the first expert that passes it, so the first `at_high` eligible layers, ascending, hold one
// more than the rest. Rows: (card, ctx, with the draft, U, high, at_high, low, card expert bytes,
// card rounding bytes).
// PIN(2026-09-28): the ubatch arena in the budget, at the U the load runs (`place::machine`'s
// `ubatch`): the rows before it counted 64 MiB of scratch, not the 2,770,817,024 B arena of
// U = 4,096 every plan carries (A6000 4k 323 a layer, 32k 317/316, with the draft 302 and 296/295;
// 3090 4k 130/129, 32k 123/122, with the draft 108/107 and 101/100). The budgets [derived: usable
// 50,952,404,992 (A6000) or 25,350,373,376 (3090) − kv 586,293,824 (4k) or 1,401,037,376 (32k) −
// context 536,870,912 − scratch 2,837,925,888 (U 4,096) or 442,821,120 (U 512) − margin
// 1,073,741,824 − the draft 2,793,406,464 (4k) or 2,852,126,720 (32k)]: A6000 4k 45,917,572,544 /
// 48,312,677,312, with the draft 43,124,166,080 / 45,519,270,848; A6000 32k 45,102,828,992 /
// 47,497,933,760, with the draft 42,250,702,272 / 44,645,807,040; 3090 4k 20,315,540,928 /
// 22,710,645,696, with the draft 17,522,134,464 / 19,917,239,232; 3090 32k 19,500,797,376 /
// 21,895,902,144, with the draft 16,648,670,656 / 19,043,775,424 (U 4,096 / U 512). U 512 frees
// 2,395,104,768 B, 779.7 experts at 3,072,000 B, 763 of them past the granules on the A6000 at 4k
// (17.7 a layer).
type CardPlanRow = (&'static str, u64, bool, u64, u64, usize, u64, u64, u64);
// PIN(2026-09-29): every row re-pinned for the ubatch walk's card route, whose
// scratch a card-experts machine now counts beside the ubatch's
// (`place::machine_for_experts`, `place::card_route_scratch_bytes`): 390,840,320 B
// at U 4,096 and 92,467,200 B at 512 off every budget above [derived: the
// budgets the rows were pinned at before the route, less its scratch, the
// spread replayed over the 43 eligible layers — three stacks each (921,600 +
// 921,600 + 1,228,800 B an expert) in whole 2 MiB granules past CARD_DENSE +
// CARD_ROUNDING — by a replica that first reproduced all sixteen old rows
// exactly, so its rows are the pins — its rounding column counted only the
// experts' granules, and each row's rounding is the dense part's 398,422,528 B
// plus those, as the host plans' rows carry; the biggest moves are the A6000 4k rows
// at U 4,096 (303/302 → 299/298, 149 experts off the card) and the 3090 32k
// row at U 4,096 (103/102 → 98, 176 experts), the smallest the rows whose
// budget slack absorbed the term].
// PIN(2026-09-30): the eight draft rows re-pinned for the MTP program's arena,
// which the draft's plan now reserves beside its card bytes
// (`place::mtp_arena_bytes`, 25,479,044 B at ctx 4,096 and 114,247,556 B at
// 32,768 with the full 248,320-row head): each row's card experts plus
// rounding fall by that arena to within the 2 MiB granules the spread fills
// (25,165,824 B at 4k, 111,149,056–115,343,360 B at 32k) [derived]; the
// values are the planner's on the box (the lead's landing window C),
// which the old pins held red on this tree and green before the arena term.
// PIN(2026-10-01): the eight draft rows re-pinned for the store walk's arena, which the draft
// program's arena now holds beside its eight-row buffers (`place::mtp_arena_bytes`, 24,877,352 B
// more at every context: 50,356,396 B at ctx 4,096 and 139,124,908 B at 32,768): each row's budget
// falls by those bytes [derived: the spread replayed over the 43 eligible layers — three stacks of
// 921,600 + 921,600 + 1,228,800 B an expert in whole 2 MiB granules past CARD_DENSE + CARD_ROUNDING
// — by a replica that first reproduced all eight old draft rows exactly from the budgets above
// (the 09-28 budgets less the 09-29 card route's scratch and the 09-30 arena); 4 to 12 experts off
// each card, 6 on the A6000 at 4k and U 4,096 (11,925 -> 11,919)].
// PIN(2026-10-01): every row re-pinned for the card experts on all 48 layers: the card reads a
// q5_K gate and up (`kq_gate_up_act_q5k`) and a q8_0 down (`q8_0_gemv_sel32`), so layers 2, 4,
// 30, 46 and 47 join the spread, their experts in the file's bytes — layer 2's gate and up of 640
// rows of ten 176 B super-blocks, 1,126,400 B each, and its down of 2,560 rows of twenty 34 B
// blocks, 1,740,800 B (3,993,600 B an expert); layers 4, 30, 46 and 47 the q4_K gate and up and
// that down (3,584,000 B); the rest 3,072,000 B as before [derived: the spread replayed over the
// 48 layers, each stack in whole 2 MiB granules past CARD_DENSE + CARD_ROUNDING, within the
// budgets above (the 09-28 budgets less the 09-29 card route's scratch, the draft rows also less
// the 09-30 and 10-01 arena), by a replica that first reproduced all sixteen old rows exactly; the
// eligible layers ascending, so the first `at_high` of the 48 hold `high`: 12,841 -> 12,568
// experts on the A6000 at 4k and U 4,096, 262/261 a layer; the card's expert bytes move by the
// three sizes and their granules, the budget's slack].
// PIN(2026-10-04): the eight draft rows re-pinned for the decode flash's fixed segment count
// (`flash_gqa::SEGMENTS`, 80 segments at every cache height): the draft program's arena
// (`place::mtp_arena_bytes`, 49,536 f32 a segment) grows 3,170,304 B at ctx 4,096 (64 -> 80
// segments) and shrinks 85,598,208 B at 32,768 (512 -> 80), so each row's card experts plus
// rounding move by the opposite of that within the 2 MiB granules' slack [derived; predicted one
// expert off each 4k row and about 27.9 (85,598,208 / 3,072,000 B) onto each 32k row]. The box's
// planner on this tree, the old pins red on exactly the eight draft rows: the 4k rows one expert
// off each, experts plus rounding -4,194,304 B (-6,291,456 B on the A6000 at U 512), the splits
// 244/17/243, 262/15/261, 73/35/72, 92/15/91; the 32k rows +26 (A6000, U 4,096), +13 (A6000, U
// 512), +43 (3090, U 4,096) and +21 (3090, U 512) experts, experts plus rounding +81,788,928 to
// +88,080,384 B, the splits 238/12/237, 255/40/254, 68/2/67, 86/5/85 — the count past the estimate
// is where the granules each layer's three stacks round to fall, the bytes within two granules of
// the arena's move.
const CARD_PLANS: [CardPlanRow; 16] = [
    (
        "A6000",
        4_096,
        false,
        4_096,
        262,
        40,
        261,
        39_385_907_200,
        594_162_176,
    ),
    (
        "A6000",
        4_096,
        false,
        512,
        280,
        28,
        279,
        42_056_192_000,
        616_620_544,
    ),
    (
        "A6000",
        4_096,
        true,
        4_096,
        244,
        17,
        243,
        36_607_078_400,
        522_961_408,
    ),
    (
        "A6000",
        4_096,
        true,
        512,
        262,
        15,
        261,
        39_308_595_200,
        514_187_776,
    ),
    (
        "A6000",
        32_768,
        false,
        4_096,
        258,
        8,
        257,
        38_685_388_800,
        478_888_448,
    ),
    (
        "A6000",
        32_768,
        false,
        512,
        275,
        37,
        274,
        41_332_224_000,
        526_893_568,
    ),
    (
        "A6000",
        32_768,
        true,
        4_096,
        238,
        12,
        237,
        35_689_164_800,
        570_556_928,
    ),
    (
        "A6000",
        32_768,
        true,
        512,
        255,
        40,
        254,
        38_332_928_000,
        615_342_592,
    ),
    (
        "3090",
        4_096,
        false,
        4_096,
        93,
        5,
        92,
        13_855_948_800,
        522_088_960,
    ),
    (
        "3090",
        4_096,
        false,
        512,
        110,
        38,
        109,
        16_515_072_000,
        557_806_080,
    ),
    (
        "3090",
        4_096,
        true,
        4_096,
        73,
        35,
        72,
        10_940_108_800,
        589_996_544,
    ),
    (
        "3090",
        4_096,
        true,
        512,
        92,
        15,
        91,
        13_736_243_200,
        486_605_312,
    ),
    (
        "3090",
        32_768,
        false,
        4_096,
        87,
        29,
        86,
        13_027_123_200,
        535_122_432,
    ),
    (
        "3090",
        32_768,
        false,
        512,
        105,
        22,
        104,
        15_713_280_000,
        539_611_648,
    ),
    (
        "3090",
        32_768,
        true,
        4_096,
        68,
        2,
        67,
        10_084_659_200,
        573_030_912,
    ),
    (
        "3090",
        32_768,
        true,
        512,
        86,
        5,
        85,
        12_802_969_600,
        547_463_680,
    ),
];
// PIN(2026-09-28): the routed stacks no card expert kernel of the program reads, which keep their
// layers' experts on the host [the header dump: layer 2's gate and up q5_K and its down q8_0, the
// downs of layers 4, 30, 46 and 47 q8_0, every q8_0 down rows of 640 values].
// PIN(2026-10-01): none: the card experts read the q5_K gate and up and the q8_0 down
// (`place::card_routed`), so every layer's stacks are the card's [the same header dump].
const HOST_ONLY: [(usize, &str); 0] = [];
/// The draft that uses the target's embedding and output matrix.
const SHARED: &str = "/models/Qwen3.8-Flash-Next/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf";

/// The card rule's plan of the Qwen3.8 file (`place::PlanInputs::plan_with`
/// under `Experts::Card`): the stacks it leaves on the host, each named with
/// its reason; on each card at 4,096 and 32,768 positions, alone and beside
/// the shared MTP draft (`plan_mtp_with`), each layer's card count, the card's
/// expert and rounding bytes and the host's as predicted, each routed stack's
/// card segment its layer's id prefix in the file's words and the rest on
/// the host, and the draft's plan the host-routed one's; a card budget that
/// leaves no expert makes the host-routed plan of the same levers, field for
/// field.
#[test]
#[ignore = "needs the Qwen3.8-Flash-Next shards and the shared MTP file on the box (just gate-qwen4exp-meta)"]
fn hw_qwen4exp_card_plan() {
    use model::arch::models::HeadRows;
    use model::arch::qwen35moe::place::{self, Experts, MtpInputs, PlanInputs};
    use model::placement::workstation::{A6000, CONTEXT, MARGIN, RTX_3090};
    use model::placement::{CardFormat, Device, Format, Plan, PlanLevers, Role};

    let mut o = String::new();
    let mut b: Vec<String> = Vec::new();
    let mut check = |o: &mut String, what: String, ok: bool| {
        let _ = writeln!(o, "{what}: {}", if ok { "PASS" } else { "FAIL" });
        if !ok {
            b.push(what);
        }
    };
    let split = Split::open(Q38).unwrap_or_else(|e| panic!("open {Q38}: {e}"));
    let inputs = PlanInputs::describe(&split).unwrap_or_else(|e| panic!("describe: {e}"));
    let draft = Split::open(SHARED).unwrap_or_else(|e| panic!("open {SHARED}: {e}"));
    let mtp = MtpInputs::read(&draft, &split, &inputs, HeadRows::Full)
        .unwrap_or_else(|e| panic!("MtpInputs::read {SHARED}: {e}"));
    let mut host_only = inputs.host_only();
    host_only.sort_by(|x, y| (x.layer, &x.tensor).cmp(&(y.layer, &y.tensor)));
    let named: Vec<(usize, &str)> = host_only
        .iter()
        .map(|h| (h.layer, h.tensor.as_str()))
        .collect();
    let reasons = host_only
        .iter()
        .all(|h| h.why.contains("no card expert kernel"));
    for h in &host_only {
        let _ = writeln!(o, "host only: layer {} {}: {}", h.layer, h.tensor, h.why);
    }
    check(
        &mut o,
        format!(
            "the stacks the card experts do not read, by name with their reason ({reasons}): \
             {named:?}"
        ),
        named == HOST_ONLY && reasons,
    );
    let off: Vec<usize> = HOST_ONLY.iter().map(|&(l, _)| l).collect();
    let per_expert = |t: &model::placement::ModelTensor| t.file_bytes / inputs.model.experts;
    let view = |p: &Plan<'_>| {
        format!(
            "{:?}",
            (
                &p.rows,
                &p.cards,
                &p.host,
                p.nvme_bytes,
                &p.n_l,
                p.ctx_max,
                p.card_budget
            )
        )
    };
    let levers = PlanLevers::default();
    for (name, ctx, with_draft, u, high, at_high, low, experts, rounding) in CARD_PLANS {
        let card = if name == A6000.name { A6000 } else { RTX_3090 };
        let machine = place::machine_for_experts(card, inputs.hp.n_layer, u, Experts::Card);
        let got = if with_draft {
            inputs
                .plan_mtp_with(&machine, ctx, &levers, &mtp, Experts::Card)
                .map(|m| {
                    let host = inputs.plan_mtp(&machine, ctx, &levers, &mtp).ok();
                    let same = host.is_some_and(|h| view(&h.draft) == view(&m.draft));
                    (m.plan, same)
                })
        } else {
            inputs
                .plan_with(&machine, ctx, &levers, Experts::Card)
                .map(|p| (p, true))
        };
        let (plan, draft_same) = match got {
            Ok(p) => p,
            Err(e) => {
                check(
                    &mut o,
                    format!("{name} ctx {ctx} draft {with_draft} U {u}: refused: {e}"),
                    false,
                );
                continue;
            }
        };
        let mut want = Vec::with_capacity(plan.n_l.len());
        let mut eligible = 0usize;
        for l in 0..plan.n_l.len() {
            if off.contains(&l) {
                want.push(0);
            } else {
                want.push(if eligible < at_high { high } else { low });
                eligible += 1;
            }
        }
        let mut segs_bad = Vec::new();
        for r in &plan.rows {
            let t = &inputs.model.tensors[r.tensor];
            if t.role != Role::RoutedExperts {
                continue;
            }
            let n = plan.n_l[t.layer.unwrap_or(0)];
            let host_ok = r.segments.last().is_some_and(|s| {
                s.device == Device::Host
                    && s.resident_bytes == (inputs.model.experts - n) * per_expert(t)
            });
            let card_ok = match r.segments.as_slice() {
                [_] => n == 0,
                [c, _] => {
                    c.device == Device::Card(0)
                        && c.format == Format::Card(CardFormat::KQuant)
                        && c.resident_bytes == n * per_expert(t)
                        && c.experts.as_ref().and_then(|e| e.as_prefix()) == Some(n)
                }
                _ => false,
            };
            if !(host_ok && card_ok) {
                segs_bad.push(t.name.clone());
            }
        }
        let c = &plan.cards[0];
        let kv = KV_AT
            .iter()
            .find(|&&(x, _)| x == ctx)
            .map_or(0, |&(_, k)| k);
        let figures = [
            ("card dense", c.dense_bytes, CARD_DENSE),
            ("card experts", c.expert_bytes, experts),
            ("card rounding", c.rounding_bytes, rounding),
            ("card kv", c.kv_bytes, kv),
            (
                "host experts",
                plan.host.expert_bytes,
                HOST_EXPERTS - experts,
            ),
            ("card scratch", c.scratch_bytes, card_scratch_at(u)),
            ("host tables", plan.host.table_bytes, HOST_TABLES),
            ("nvme", plan.nvme_bytes, 0),
        ];
        let bytes_ok = figures.iter().all(|&(_, g, w)| g == w);
        let shown: Vec<String> = figures
            .iter()
            .map(|(what, g, w)| format!("{what} {g} (want {w})"))
            .collect();
        let held: u64 = plan.n_l.iter().sum();
        check(
            &mut o,
            format!(
                "{name} ctx {ctx} draft {with_draft} U {u}: n_l {high} on the first {at_high} of the {eligible}, \
                 {low} on the rest, 0 on {off:?} ({}; {held} experts); segments off the rule \
                 {segs_bad:?}; the draft's plan the host-routed one's {draft_same}; {}; headroom {}",
                plan.n_l == want,
                shown.join(", "),
                c.headroom_bytes
            ),
            plan.n_l == want && segs_bad.is_empty() && draft_same && bytes_ok,
        );
    }
    for card in [A6000, RTX_3090] {
        let machine = place::machine(card, inputs.hp.n_layer, place::UBATCH_PLANNED);
        let floor = CARD_DENSE
            + CARD_ROUNDING
            + KV_AT[0].1
            + CONTEXT
            + scratch_at(place::UBATCH_PLANNED)
            + MARGIN;
        let levers = PlanLevers {
            card_budget_bytes: Some(floor),
        };
        let pair = (
            inputs.plan_with(&machine, KV_AT[0].0, &levers, Experts::Card),
            inputs.plan_with(&machine, KV_AT[0].0, &levers, Experts::Host),
        );
        let (ok, text) = match pair {
            (Ok(c), Ok(h)) => (
                c.n_l.iter().all(|&n| n == 0) && view(&c) == view(&h),
                format!(
                    "card experts {}, the host-routed plan's field for field {}",
                    c.cards[0].experts,
                    view(&c) == view(&h)
                ),
            ),
            (c, h) => (false, format!("card {:?} host {:?}", c.err(), h.err())),
        };
        check(
            &mut o,
            format!(
                "{} ctx {} under a card budget of its floor {floor}: {text}",
                card.name, KV_AT[0].0
            ),
            ok,
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

/// The Q3 file: unsloth's `UD-Q3_K_XL`. Its first shard holds the header and
/// no tensor.
const Q3: &str =
    "/models/Qwen3.8-Flash-Next-UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf";

// PIN(2026-10-06): the Q3 file's dense card part and routed share [derived from its headers, as
// CARD_DENSE's derivation reads the Q4 file's: every non-routed tensor but the PLE table on the
// card in its CardFormat, the head one Q6_K word buffer of 521,472,000 B (248,320 rows of ten
// 210 B super-blocks) in place of the Q4 file's two Q8_0 planes of 675,430,400 B, every other
// non-routed tensor the same type and size in both files; the routed stacks, 43 layers of an
// IQ3_XXS gate and up (627,200 B each: 640 rows of ten 98 B super-blocks) with an IQ4_NL down
// (921,600 B: 2,560 rows of twenty 18 B blocks), 2,176,000 B an expert, layers 4, 30, 46 and 47
// that gate and up with a Q8_0 down (1,740,800 B), 2,995,200 B, and layer 2 an IQ4_XS gate and
// up (870,400 B each) with that down, 3,481,600 B; 512 experts a layer].
const CARD_DENSE_Q3: u64 = 5_390_947_840;
const HOST_EXPERTS_Q3: u64 = 55_823_564_800;
// PIN(2026-10-06): the dense part's granules at no expert [derived: CARD_ROUNDING's heap walk
// over the Q3 file's dense uploads, the head's smaller buffer].
const CARD_ROUNDING_Q3: u64 = 397_191_680;

// PIN(2026-10-06): the card rule's plan of the Q3 file (`hw_qwen4exp_card_plan`'s clauses over
// it, every stack a card type since `card_routed` reads the i-quants) [derived by a replica that
// first reproduced all sixteen CARD_PLANS rows of the Q4 file exactly, then replayed the spread
// over the same 48 layers with the per-layer expert bytes 2,176,000 / 2,995,200 / 3,481,600 B,
// each stack in whole 2 MiB granules past CARD_DENSE_Q3 + CARD_ROUNDING_Q3, within CARD_PLANS'
// own budgets: the machine terms are none of the file's — the same usable, cache, context,
// scratch and route scratch, margin and draft reserve, the draft's own plan unchanged beside the
// target (it borrows the target's matrices)]. PIN(2026-10-07): the draft reserve the spread fills
// within now holds the Q6_K head's 68,736 B of activation bytes as the CardOver total does
// (`MtpInputs::arena_bytes`, round q3seat); every drafted row here left 400,148 B or more of slack
// before it, so no row moves [derived: the q3seat replica reproduced all eight drafted rows'
// expert bytes, the least slack the 3090's at ctx 4,096 and U 512].
const CARD_PLANS_Q3: [CardPlanRow; 16] = [
    (
        "A6000",
        4_096,
        false,
        4_096,
        365,
        1,
        364,
        39_689_241_600,
        444_786_176,
    ),
    (
        "A6000",
        4_096,
        false,
        512,
        388,
        21,
        387,
        42_242_585_600,
        586_282_496,
    ),
    (
        "A6000",
        4_096,
        true,
        4_096,
        338,
        4,
        337,
        36_753_254_400,
        532_840_960,
    ),
    (
        "A6000",
        4_096,
        true,
        512,
        362,
        28,
        361,
        39_423_027_200,
        553_714_176,
    ),
    (
        "A6000",
        32_768,
        false,
        4_096,
        355,
        41,
        354,
        38_688_921_600,
        629_314_048,
    ),
    (
        "A6000",
        32_768,
        false,
        512,
        381,
        48,
        381,
        41_540_582_400,
        470_396_416,
    ),
    (
        "A6000",
        32_768,
        true,
        4_096,
        328,
        46,
        327,
        35_755_980_800,
        657_699_328,
    ),
    (
        "A6000",
        32_768,
        true,
        512,
        355,
        7,
        354,
        38_614_118_400,
        490_207_744,
    ),
    (
        "3090",
        4_096,
        false,
        4_096,
        128,
        41,
        127,
        13_939_020_800,
        590_878_208,
    ),
    (
        "3090",
        4_096,
        false,
        512,
        154,
        5,
        153,
        16_694_656_000,
        530_083_328,
    ),
    (
        "3090",
        4_096,
        true,
        4_096,
        103,
        22,
        102,
        11_171_097_600,
        512_966_144,
    ),
    (
        "3090",
        4_096,
        true,
        512,
        128,
        17,
        127,
        13_885_977_600,
        492_926_464,
    ),
    (
        "3090",
        32_768,
        false,
        4_096,
        121,
        29,
        120,
        13_148_876_800,
        565_230_080,
    ),
    (
        "3090",
        32_768,
        false,
        512,
        146,
        21,
        145,
        15_857_228_800,
        553_815_552,
    ),
    (
        "3090",
        32_768,
        true,
        4_096,
        94,
        34,
        93,
        10_216_755_200,
        592_796_160,
    ),
    (
        "3090",
        32_768,
        true,
        512,
        119,
        40,
        118,
        12_955_571_200,
        548_820_480,
    ),
];

/// The card rule's plan of the Q3 file: `hw_qwen4exp_card_plan`'s clauses
/// over it — the coverage list the Q4 file's (the i-quant stacks and the
/// Q6_K head no items), the shared MTP draft read against it with its
/// borrowed head the Q6_K form, every stack the card's, and the rows of
/// [`CARD_PLANS_Q3`] with the same segment checks.
#[test]
#[ignore = "needs the UD-Q3_K_XL shards and the shared MTP file on the box (just gate-qwen4exp-meta)"]
fn hw_qwen4exp_q3_card_plan() {
    use model::arch::models::HeadRows;
    use model::arch::qwen35moe::mtp::BorrowedHead;
    use model::arch::qwen35moe::place::{self, Experts, MtpInputs, PlanInputs};
    use model::placement::workstation::{A6000, CONTEXT, MARGIN, RTX_3090};
    use model::placement::{CardFormat, Device, Format, Plan, PlanLevers, Role};

    let mut o = String::new();
    let mut b: Vec<String> = Vec::new();
    let mut check = |o: &mut String, what: String, ok: bool| {
        let _ = writeln!(o, "{what}: {}", if ok { "PASS" } else { "FAIL" });
        if !ok {
            b.push(what);
        }
    };
    let split = Split::open(Q3).unwrap_or_else(|e| panic!("open {Q3}: {e}"));
    let inputs = PlanInputs::describe(&split).unwrap_or_else(|e| panic!("describe: {e}"));
    let list = spec_view::items(&inputs.unimplemented());
    check(
        &mut o,
        format!("the coverage list is the Q4 file's: {list:?}"),
        list.iter().map(String::as_str).eq(COVERAGE.iter().copied()),
    );
    let draft = Split::open(SHARED).unwrap_or_else(|e| panic!("open {SHARED}: {e}"));
    let mtp = MtpInputs::read(&draft, &split, &inputs, HeadRows::Full)
        .unwrap_or_else(|e| panic!("MtpInputs::read {SHARED} over {Q3}: {e}"));
    check(
        &mut o,
        format!(
            "the borrowed head is the Q6_K form: {:?}",
            mtp.borrowed_head
        ),
        mtp.borrowed_head == BorrowedHead::Q6K,
    );
    let host_only = inputs.host_only();
    check(
        &mut o,
        format!("the stacks the card experts do not read: {host_only:?}"),
        host_only.is_empty(),
    );
    let per_expert = |t: &model::placement::ModelTensor| t.file_bytes / inputs.model.experts;
    let view = |p: &Plan<'_>| {
        format!(
            "{:?}",
            (
                &p.rows,
                &p.cards,
                &p.host,
                p.nvme_bytes,
                &p.n_l,
                p.ctx_max,
                p.card_budget
            )
        )
    };
    let levers = PlanLevers::default();
    for (name, ctx, with_draft, u, high, at_high, low, experts, rounding) in CARD_PLANS_Q3 {
        let card = if name == A6000.name { A6000 } else { RTX_3090 };
        let machine = place::machine_for_experts(card, inputs.hp.n_layer, u, Experts::Card);
        let got = if with_draft {
            inputs
                .plan_mtp_with(&machine, ctx, &levers, &mtp, Experts::Card)
                .map(|m| {
                    let host = inputs.plan_mtp(&machine, ctx, &levers, &mtp).ok();
                    let same = host.is_some_and(|h| view(&h.draft) == view(&m.draft));
                    (m.plan, same)
                })
        } else {
            inputs
                .plan_with(&machine, ctx, &levers, Experts::Card)
                .map(|p| (p, true))
        };
        let (plan, draft_same) = match got {
            Ok(p) => p,
            Err(e) => {
                check(
                    &mut o,
                    format!("{name} ctx {ctx} draft {with_draft} U {u}: refused: {e}"),
                    false,
                );
                continue;
            }
        };
        let want: Vec<u64> = (0..plan.n_l.len())
            .map(|l| if l < at_high { high } else { low })
            .collect();
        let mut segs_bad = Vec::new();
        for r in &plan.rows {
            let t = &inputs.model.tensors[r.tensor];
            if t.role != Role::RoutedExperts {
                continue;
            }
            let n = plan.n_l[t.layer.unwrap_or(0)];
            let host_ok = r.segments.last().is_some_and(|s| {
                s.device == Device::Host
                    && s.resident_bytes == (inputs.model.experts - n) * per_expert(t)
            });
            let card_ok = match r.segments.as_slice() {
                [_] => n == 0,
                [c, _] => {
                    c.device == Device::Card(0)
                        && c.format == Format::Card(CardFormat::KQuant)
                        && c.resident_bytes == n * per_expert(t)
                        && c.experts.as_ref().and_then(|e| e.as_prefix()) == Some(n)
                }
                _ => false,
            };
            if !(host_ok && card_ok) {
                segs_bad.push(t.name.clone());
            }
        }
        let c = &plan.cards[0];
        let kv = KV_AT
            .iter()
            .find(|&&(x, _)| x == ctx)
            .map_or(0, |&(_, k)| k);
        let figures = [
            ("card dense", c.dense_bytes, CARD_DENSE_Q3),
            ("card experts", c.expert_bytes, experts),
            ("card rounding", c.rounding_bytes, rounding),
            ("card kv", c.kv_bytes, kv),
            (
                "host experts",
                plan.host.expert_bytes,
                HOST_EXPERTS_Q3 - experts,
            ),
            ("card scratch", c.scratch_bytes, card_scratch_at(u)),
            ("host tables", plan.host.table_bytes, HOST_TABLES),
            ("nvme", plan.nvme_bytes, 0),
        ];
        let bytes_ok = figures.iter().all(|&(_, g, w)| g == w);
        let shown: Vec<String> = figures
            .iter()
            .map(|(what, g, w)| format!("{what} {g} (want {w})"))
            .collect();
        check(
            &mut o,
            format!(
                "{name} ctx {ctx} draft {with_draft} U {u}: n_l {high} on the first {at_high}, \
                 {low} on the rest ({}); segments off the rule {segs_bad:?}; the draft's plan \
                 the host-routed one's {draft_same}; {}; headroom {}",
                plan.n_l == want,
                shown.join(", "),
                c.headroom_bytes
            ),
            plan.n_l == want && segs_bad.is_empty() && draft_same && bytes_ok,
        );
    }
    for card in [A6000, RTX_3090] {
        let machine = place::machine(card, inputs.hp.n_layer, place::UBATCH_PLANNED);
        let floor = CARD_DENSE_Q3
            + CARD_ROUNDING_Q3
            + KV_AT[0].1
            + CONTEXT
            + scratch_at(place::UBATCH_PLANNED)
            + MARGIN;
        let levers = PlanLevers {
            card_budget_bytes: Some(floor),
        };
        let pair = (
            inputs.plan_with(&machine, KV_AT[0].0, &levers, Experts::Card),
            inputs.plan_with(&machine, KV_AT[0].0, &levers, Experts::Host),
        );
        let (ok, text) = match pair {
            (Ok(c), Ok(h)) => (
                c.n_l.iter().all(|&n| n == 0) && view(&c) == view(&h),
                format!(
                    "card experts {}, the host-routed plan's field for field {}",
                    c.cards[0].experts,
                    view(&c) == view(&h)
                ),
            ),
            (c, h) => (false, format!("card {:?} host {:?}", c.err(), h.err())),
        };
        check(
            &mut o,
            format!(
                "{} ctx {} under a card budget of its floor {floor}: {text}",
                card.name, KV_AT[0].0
            ),
            ok,
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

/// Plan (b′)'s row: the A6000's counts (`high` on the first `at_high`
/// layers, `low` on the rest), both cards' totals a layer (`total_high` on
/// the first `total_at_high`, one fewer on the rest), the tier's experts,
/// expert bytes and rounding bytes.
type BpRow = (u64, usize, u64, u64, usize, u64, u64, u64);
// PIN(2026-10-02): plan (b′) (`place::machine_bp`, `--place bp`), plain, at 4,096 positions and
// U 4,096. The split [derived: per routed slot of a decode pass the A6000's card leg reads an expert
// of 3,072,000 B, ~5.3 µs at 575 GB/s, the 3090's ~4.4 µs at 700 GB/s, the host's union 23.6 µs
// (rig-log 09-30#q38res-hit); residency holds the tier's ids away for the load's life, so the
// tier's share of the routed mass is its ids' — ~k/512 under the id prefix — whatever the A6000
// keeps, and an expert the A6000 gave up past its budget would go to the host (the tier is at its
// own), at ~4.5 A6000 slots' time; so both cards plan to their budgets and the A6000 keeps plan (a)'s
// row at its budget less the tier join's unit-wide card rows, places and ranks
// (`place::card_tier_join_bytes`, 210,124,800 B at U 4,096), byte for byte: 12,534 experts, 262 on
// the first 6 layers and 261 on the rest, 34 fewer than the row before the join's 12,568 (the
// drafted row loses 72)]. The tier [derived: budget 25,350,373,376 usable − 536,870,912 context −
// 67,108,864 scratch − 845,710,360 tier prompt batch (`place::tier_batch` at U 4,096: staging
// 496,730,112, the block route's 348,980,248, `place::tier_route_scratch_bytes`, which holds the
// run's down rows the pack reads, 209,715,200, and the block's ranks, 163,840) − 1,073,741,824
// margin = 22,826,941,416 B; the spread over all 48 layers beside the A6000's counts, the layer with
// the fewest on both cards first, each stack in whole 2 MiB granules: both cards 412 on the first
// 42 layers and 411 on the rest, the tier 150 on layers 0-5 and 42-47 and 151 on 6-41, 7,236
// experts, 22,674,944,000 B of experts and 148,361,216 B of rounding (22,823,305,216 B, 3,636,200
// under the budget); the host 100 or 101 a layer, 4,806 experts: the run's down rows move 68 tier
// experts to the host]. The tier's spread and the host's count are what the box's planner printed
// in q38tier2b's red run, the A6000's split their remainder.
const BP_PLAIN: BpRow = (262, 6, 261, 412, 42, 7_236, 22_674_944_000, 148_361_216);

/// Plan (b′) of the Qwen3.8 file without the draft (`place::machine_bp`,
/// `plan_with` under `Experts::Card`): the A6000's counts and card bytes
/// plan (a)'s, the 3090 tier's per layer both cards' totals less the
/// A6000's, its expert and rounding bytes and the host's as predicted, each
/// routed stack in three segments — the A6000's id prefix, the tier's next
/// ids, the host's rest; the host-routed rule on the tier machine and a
/// draft reserve on a plan without the draft are refused by name.
#[test]
#[ignore = "needs the Qwen3.8-Flash-Next shards on the box (just gate-qwen4exp-meta)"]
fn hw_qwen4exp_bp_plan() {
    use model::arch::qwen35moe::place::{
        self, Experts, MTP_RESERVE, PlaceError, PlanInputs, machine_bp, tier_batch,
    };
    use model::placement::workstation::A6000;
    use model::placement::{Device, PlanLevers};

    let mut o = String::new();
    let mut b: Vec<String> = Vec::new();
    let mut check = |o: &mut String, what: String, ok: bool| {
        let _ = writeln!(o, "{what}: {}", if ok { "PASS" } else { "FAIL" });
        if !ok {
            b.push(what);
        }
    };
    let split = Split::open(Q38).unwrap_or_else(|e| panic!("open {Q38}: {e}"));
    let inputs = PlanInputs::describe(&split).unwrap_or_else(|e| panic!("describe: {e}"));
    let (ctx, u) = (4096u64, 4096u64);
    let n = inputs.hp.n_layer;
    let levers = PlanLevers::default();
    let batch = tier_batch(&inputs.hp, u);
    let machine = machine_bp(n, u, None, batch);
    // Plan (a)'s rule at the A6000's budget less the tier join's scratch
    // (`place::card_tier_join_bytes`), which plan (b′) adds to its scratch.
    let mut a_machine = place::machine_for_experts(A6000, n, u, Experts::Card);
    a_machine.cards[0].scratch_bytes += place::card_tier_join_bytes(u);
    let a = inputs
        .plan_with(&a_machine, ctx, &levers, Experts::Card)
        .unwrap_or_else(|e| panic!("plan (a): {e}"));
    let (high, at_high, low, t_high, t_at_high, t_experts, t_bytes, t_rounding) = BP_PLAIN;
    match inputs.plan_with(&machine, ctx, &levers, Experts::Card) {
        Err(e) => check(&mut o, format!("plan (b′) refused: {e}"), false),
        Ok(plan) => {
            let stage: Vec<u64> = (0..n)
                .map(|l| if l < at_high { high } else { low })
                .collect();
            let total: Vec<u64> = (0..n)
                .map(|l| if l < t_at_high { t_high } else { t_high - 1 })
                .collect();
            let tier_want: Vec<u64> = total.iter().zip(&stage).map(|(t, s)| t - s).collect();
            let tier = plan.tier_n_l.first().cloned().unwrap_or_default();
            let (s, sa, t) = (&plan.cards[0], &a.cards[0], &plan.cards[1]);
            let stage_same = (s.dense_bytes, s.expert_bytes, s.rounding_bytes, s.kv_bytes)
                == (
                    sa.dense_bytes,
                    sa.expert_bytes,
                    sa.rounding_bytes,
                    sa.kv_bytes,
                )
                && plan.n_l == a.n_l
                && s.headroom_bytes == sa.headroom_bytes;
            check(
                &mut o,
                format!(
                    "the A6000 {high} on the first {at_high}, {low} on the rest ({}: {:?}), plan \
                     (a)'s counts and card bytes ({stage_same}); headroom {}",
                    plan.n_l == stage,
                    plan.n_l,
                    s.headroom_bytes
                ),
                plan.n_l == stage && stage_same,
            );
            let held: u64 = tier.iter().sum();
            check(
                &mut o,
                format!(
                    "the 3090 tier: both cards {t_high} on the first {t_at_high}, {} on the rest, \
                     so the tier {tier:?} (want {tier_want:?}); {held} experts (want {t_experts}), \
                     card experts {} (want {t_experts}), expert bytes {} (want {t_bytes}), \
                     rounding {} (want {t_rounding}), dense {}, kv {}; headroom {}",
                    t_high - 1,
                    t.experts,
                    t.expert_bytes,
                    t.rounding_bytes,
                    t.dense_bytes,
                    t.kv_bytes,
                    t.headroom_bytes
                ),
                tier == tier_want
                    && held == t_experts
                    && t.experts == t_experts
                    && (t.expert_bytes, t.rounding_bytes, t.dense_bytes, t.kv_bytes)
                        == (t_bytes, t_rounding, 0, 0),
            );
            let host_experts =
                n as u64 * inputs.model.experts - plan.n_l.iter().sum::<u64>() - held;
            let host_bytes = HOST_EXPERTS - s.expert_bytes - t.expert_bytes;
            check(
                &mut o,
                format!(
                    "the host {} experts (want {host_experts}), {} B (want {host_bytes}); \
                     headroom {}",
                    plan.host.experts, plan.host.expert_bytes, plan.host.headroom_bytes
                ),
                plan.host.experts == host_experts && plan.host.expert_bytes == host_bytes,
            );
            let mut off_rule = Vec::new();
            for r in &plan.rows {
                let tensor = &inputs.model.tensors[r.tensor];
                let Some(l) = tensor
                    .layer
                    .filter(|_| tensor.role == model::placement::Role::RoutedExperts)
                else {
                    continue;
                };
                let per = tensor.file_bytes / inputs.model.experts;
                let (ns, nt) = (plan.n_l[l], tier.get(l).copied().unwrap_or(0));
                let ok = match r.segments.as_slice() {
                    [c, k, h] => {
                        c.device == Device::Card(0)
                            && c.experts.as_ref().and_then(|e| e.as_prefix()) == Some(ns)
                            && k.device == Device::Card(1)
                            && k.experts.as_ref().is_some_and(|e| {
                                e.ids().iter().map(|&i| u64::from(i)).eq(ns..ns + nt)
                            })
                            && k.resident_bytes == nt * per
                            && h.device == Device::Host
                            && h.resident_bytes == (inputs.model.experts - ns - nt) * per
                    }
                    _ => false,
                };
                if !ok {
                    off_rule.push(tensor.name.clone());
                }
            }
            check(
                &mut o,
                format!(
                    "each routed stack: the A6000's prefix, the tier's next ids, the host's rest; \
                     off the rule {off_rule:?}"
                ),
                off_rule.is_empty(),
            );
        }
    }
    let host = inputs.plan_with(&machine, ctx, &levers, Experts::Host);
    check(
        &mut o,
        format!(
            "the host-routed rule on the tier machine: {:?}",
            host.as_ref().err()
        ),
        matches!(host, Err(PlaceError::TierHost { .. })),
    );
    let drafted = machine_bp(n, u, Some(1), batch);
    let reserve = inputs.plan_with(&drafted, ctx, &levers, Experts::Card);
    check(
        &mut o,
        format!(
            "a plain plan on a machine reserving \"{MTP_RESERVE}\": {:?}",
            reserve.as_ref().err().map(ToString::to_string)
        ),
        matches!(reserve, Err(PlaceError::DraftReserve { .. })),
    );
    println!("{o}");
    assert!(
        b.is_empty(),
        "{} clause(s) failed:\n  {}",
        b.len(),
        b.join("\n  ")
    );
}
