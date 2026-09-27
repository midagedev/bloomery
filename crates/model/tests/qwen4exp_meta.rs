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
const COVERAGE: &[&str] = &[
    "pre-tokenizer qwen35",
    "a tool-call parser for this template",
];

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
// the clause holds the placement to that boundary, not to this figure.
const CARD_MAX_CTX: [(&str, u64); 2] = [("A6000", 1_508_356), ("3090", 607_383)];

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
        let machine = place::machine(card, hp.n_layer);
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
    let machine = place::machine(A6000, hp.n_layer);
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
