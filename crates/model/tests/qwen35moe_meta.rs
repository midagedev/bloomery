//! qwen35moe metadata gate: what the header reader `arch::qwen35moe` resolves
//! from a Qwen3.6-35B-A3B file, and what the coverage check lists for it —
//! the parts no program in this tree runs yet, the work queue of the rounds
//! that build one, and which layers a prompt call runs at every position
//! (every one: the GQA layers keep their planes, the GDN layers their state).
//! One contract, one test; it prints what it compared, then fails with the
//! whole list of what differs.
//!
//! `hw_`: needs both files on the box (`just gate-qwen35moe-meta`). Headers
//! only: seconds.

#[path = "common/spec_fail_first.rs"]
mod spec_fail_first;
#[path = "common/spec_view.rs"]
mod spec_view;

use std::fmt::Write as _;

use gguf::Split;

/// The two files: the lmstudio `Q4_K_M` the next model runs from, and
/// unsloth's `UD-Q4_K_XL`, whose Q8_0 and Q5_K tensors are the format items.
const Q4KM: &str = "/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-Q4_K_M.gguf";
const UD: &str = "/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf";

// PIN(2026-09-27): both files' description as `arch::qwen35moe::spec` reads it, line by line, but
// the chat line — docs/research/modelspec-design.md §2b and §5 are what these were checked against.
// PIN(2026-09-27): the delta line re-pinned when the GDN kind gained its output gate's activation
// (SiLU here); the coverage lists below are unchanged.
const VIEW: &[&str] = &[
    "arch Qwen35Moe",
    "hidden 2048 vocab 248320 ctx_train 262144",
    "rms_eps bits 0x358637bd",
    "layers 40 mtp 0",
    "hc None",
    "engram None",
    "[0-2,4-6,8-10,12-14,16-18,20-22,24-26,28-30,32-34,36-38] delta Gdn { khead_map: Tiled, gate: Silu } k 16 v 32 d 128 conv 4 || moe 256/8 ff 512 swiglu None Softmax bias false norm true x1 hash false shared 512 swiglu None gate true || plain",
    "[3,7,11,15,19,23,27,31,35,39] gqa 16/2 x 256 rope Imrope { sections: [11, 11, 10, 0] } 64 base 10000000 yarn - qk_norm true out_gate true || moe 256/8 ff 512 swiglu None Softmax bias false norm true x1 hash false shared 512 swiglu None gate true || plain",
];

// PIN(2026-09-27): each file's chat line (its template's bytes differ).
const CHAT_Q4KM: &str = "chat pre qwen35 template bytes 7764 tools None reasoning None";
const CHAT_UD: &str = "chat pre qwen35 template bytes 8057 tools None reasoning None";

// PIN(2026-09-27): the keys the reader took a default for, on both files.
const DEFAULTS: &[&str] = &[
    "expert_gating_func = softmax (qwen35moe.cpp:499-508)",
    "expert_weights_norm = true (qwen35moe.cpp:499-508)",
    "expert_weights_scale = 1 (qwen35moe.cpp:499-508)",
];

// PIN(2026-09-27): the coverage check's list for the Q4_K_M file (feature: layers): every tensor
// is Q4_K, Q6_K or F32, so no format item. The router (256/8 with the shared expert's gate row)
// and the GQA flash (head 256, group 8) are not items: the shape table's rows serve them
// (`FAIL_FIRST` shows each come back without its row).
// PIN(2026-09-27): six items left both lists with the rows of what Body35 runs, keyed to its
// program: the SiLU-gated GDN (`plan.rs` delta_shape), the sigmoid-gated shared expert of 512
// (`moe_fits`), the QK norm and rope of head 256 (IMROPE sections covering the 64 turned values),
// the output gate, the recurrent-state slot and the mixed trunk; gate-gpu-qwen35moe-e2e runs each.
// The chat surface's two items stay: the program does not run them.
// PIN(2026-09-30): the pre-tokenizer item left both lists with the tokenizer's qwen35 pre-tokenizer
// (`crates/tokenizer/src/pretok.rs`), which gate-tokenizer holds to the reference's split.
const COVERAGE_Q4KM: &[&str] = &["a tool-call parser for this template"];

// PIN(2026-09-27): the same for the UD-Q4_K_XL file, with its Q8_0 and Q5_K format items, which
// Body35 does not read.
// PIN(2026-10-03): the type pins split by body (Body35 its own program, its pins its sites' types,
// gpu/src/arch/qwen3moe/body35.rs PROJ, EMBED, HEAD_TY, ROUTED, ROUTED_DOWN): Body35 launches q8_0
// at the head, the embedding and every attention and delta-rule matrix, so those three items left;
// it joins each shared expert into its layer's routed stacks, of one type, q4_K gate and up and
// q4_K or q6_K down, so this file's q8_0 shared experts on every layer are two items. Red on the
// tree before the split: it lists the three q8_0 items and not the shared expert's two.
const COVERAGE_UD: &[&str] = &[
    "q5_K routed experts on a card: 0-33,35-37",
    "q8_0 shared expert down, joined into the routed stack (the body reads q4_K and q6_K): 0-39",
    "q8_0 shared expert gate and up, joined into the routed stacks (the body reads q4_K): 0-39",
    "a tool-call parser for this template",
];

// PIN(2026-09-27): each shape-table row of AVAILABLE the files use, and the item it covers on
// them — the coverage change's FAIL-first, on the Q4_K_M file; with them two of Body35's own rows.
// PIN(2026-09-27): the delta-rule and output-gate rows joined when Body35's rows were keyed.
const FAIL_FIRST: &[(&str, &str)] = &[
    (
        "models/src/shape.rs ROUTERS, the Qwen3moe and Qwen35moe bodies",
        "router: softmax, 256 experts, top 8, the shared expert's gate as one more row: 0-39",
    ),
    (
        "models/src/shape.rs GQA",
        "GQA flash, head 256, group 8: 3,7,11,15,19,23,27,31,35,39",
    ),
    (
        "gpu/src/arch/qwen3moe/plan.rs delta_shape (Body35's delta layers)",
        "delta rule GDN: d 128, 16 k-heads and 32 v-heads (tiled), conv 4: 0-2,4-6,8-10,12-14,16-18,20-22,24-26,28-30,32-34,36-38",
    ),
    (
        "gpu/src/arch/qwen3moe/dispatch.rs gated_256 (Body35's output gate)",
        "attention output gate: sigmoid, interleaved with q: 3,7,11,15,19,23,27,31,35,39",
    ),
];

#[test]
#[ignore = "needs the Qwen3.6 files on the box (just gate-qwen35moe-meta)"]
fn hw_qwen35moe_spec() {
    let mut o = String::new();
    let mut b = Vec::new();
    for (path, chat, coverage) in [(Q4KM, CHAT_Q4KM, COVERAGE_Q4KM), (UD, CHAT_UD, COVERAGE_UD)] {
        let split = Split::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
        let read = model::arch::spec(&split).unwrap_or_else(|e| panic!("spec of {path}: {e}"));
        let _ = writeln!(o, "file {path}");
        let mut view: Vec<&str> = VIEW.to_vec();
        view.insert(6, chat);
        spec_view::compare(
            &mut o,
            &mut b,
            "description",
            &spec_view::view(&read.spec),
            &view,
        );
        spec_view::compare(&mut o, &mut b, "defaults taken", &read.defaults, DEFAULTS);
        let list = model::arch::coverage::check(&read.spec, &read.tensors);
        spec_view::compare(
            &mut o,
            &mut b,
            "coverage",
            &spec_view::items(&list),
            coverage,
        );
        if path == Q4KM {
            for (at, want) in FAIL_FIRST {
                spec_fail_first::fail_first(&mut o, &mut b, &read.spec, &read.tensors, at, want);
            }
        }
        let short: Vec<String> = (0..)
            .zip(&read.spec.layers)
            .filter(|(_, l)| !runtime::state::every_position(l))
            .map(|(i, _): (usize, _)| format!("layer {i}: its stores are windows alone"))
            .collect();
        spec_view::compare(
            &mut o,
            &mut b,
            "layers a prompt call runs at fewer than every position",
            &short,
            &[],
        );
        let err = model::placement::PlacementError::Unimplemented(list);
        let _ = writeln!(o, "as the engine refuses it: {err}");
    }
    println!("{o}");
    assert!(
        b.is_empty(),
        "{} pin(s) differ:\n  {}",
        b.len(),
        b.join("\n  ")
    );
}
