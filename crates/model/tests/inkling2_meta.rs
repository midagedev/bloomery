//! inkling2 metadata gate: what the header reader `arch::inkling2` resolves from the two Inkling 2
//! files on the box — the Mini's looped `inkling2` (unsloth's UD-Q4_K_XL) and the Small's loopless
//! one (unsloth's UD-Q2_K_XL, six shards) — what the coverage check lists for them (the parts no
//! program in this tree runs yet, the work queue of the rounds that build one), and the tensors
//! by role and type. One contract, one test a file; each prints what it compared, then fails with
//! the whole list of what differs.
//!
//! `hw_`: needs the files on the box (`just gate-inkling2-meta`). Headers only: seconds.

#[path = "common/spec_view.rs"]
mod spec_view;

use std::collections::BTreeMap;
use std::fmt::Write as _;

use gguf::Split;

/// The Mini: unsloth's UD-Q4_K_XL, one shard, the loop of the core [6, 18) three times.
const MINI: &str = "/models/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q4_K_XL.gguf";

/// The Small: unsloth's UD-Q2_K_XL, six shards, no loop.
const SMALL: &str =
    "/models/Inkling-2-Small-GGUF/UD-Q2_K_XL/Inkling-2-Small-UD-Q2_K_XL-00001-of-00006.gguf";

// PIN(2026-11-21): the Mini's description as `arch::inkling2::spec` reads it, line by line (the
// key values checked against the UD-Q4_K_XL header: 24 stored layers of 48 visits, heads 24/8 of
// 128, 128 experts of 1152 at the top 8 scaled 8, the bias bank 16 x 1024, the clips 22.627417
// and 8, the log-scaled positions at floor 8192 alpha 0.1, four taps, the windowed layers at
// 1024, the windowless at 5, 11, 17 and 23 and their repeats, the loop of [6, 18) three times at
// rank 64, the head softcapped at 30 and cut to the unpadded 200058 of 201024).
const MINI_VIEW: &[&str] = &[
    "arch Inkling2",
    "hidden 2304 vocab 201024 ctx_train 1048576",
    "rms_eps bits 0x358637bd",
    "layers 24 mtp 0",
    "hc None",
    "engram None",
    "chat pre inkling2 template bytes 6868 tools None reasoning None",
    "loop core [6, 18) passes 3 rank 64",
    "logit scale 1 softcap 30 cut 200058",
    "[0-4,6-10,12-16,18-22] relbias 24/8 x 128 d_rel 16 extent 1024 clips 22.627417/8 log 8192/0.1 taps 4 win 1024 gate Proj || moe 128/8 ff 1152 situ 4/25 Sigmoid bias true norm true x8 hash false || attn-res-mix 6/1",
    "[5,11,17,23] relbias 24/8 x 128 d_rel 16 extent 1024 clips 22.627417/8 log 8192/0.1 taps 4 dyn gate Proj || moe 128/8 ff 1152 situ 4/25 Sigmoid bias true norm true x8 hash false || attn-res-mix 6/3",
];

// PIN(2026-11-21): the Mini takes no key default; its `defaults` are the description row's facts,
// each with its source (the router's f32 logits and selected-sum norm, the value norm, the mix's
// fixed epsilon, the convolutions' scale, the sink denominator, the head's cut, flash attention
// off), in the row's order.
const MINI_DEFAULTS: &[&str] = &[
    "router norm = selected-sum (B:65: expert_weights_norm_eps = 1e-20 unconditional)",
    "router logits = f32 (expert_router_f32) (B:64: the router scores run in f32)",
    "router bias = selection only (M:690-696: exp_probs_b steers the choice, the kept weights are the unbiased scores)",
    "V = gainless per-head RMS, always (B:406: the value rows are normed whatever the file's `value_norm` key says)",
    "mix eps = 1e-6 fixed, scale 1/sqrt(hidden) (B:217,222: the block attention residual's states are normed at a fixed epsilon and scored at 1/sqrt(hidden))",
    "conv output = (x + conv(x)) / sqrt(taps + 1), twice the taps where the k/v convs are dynamic (B:282: every convolution's scale counts its taps (and the dynamic taps))",
    "softmax denominator = 1 + sum(exp(s - t)) (B:429-438: a per-(head, query) sink logit t adds a zero column)",
    "head = softcap then the unpadded columns (B:503-510: the softcap runs on the padded logits, the cut after it)",
    "flash attention = off (B:88, llama-graph.cpp:3201: use_flash_attn needs no kq_b, and every layer carries one)",
];

// PIN(2026-11-21): the keys of the Mini the reader does not read: the rope it never turns, the
// keys' second spelling the reference abandoned, the dense width no layer holds and the imatrix
// calibration's.
const MINI_UNREAD: &[&str] = &[
    "inkling2.feed_forward_length",
    "inkling2.attention.key_length_swa",
    "inkling2.attention.value_length_swa",
    "inkling2.rope.dimension_count",
    "general.sampling.temp",
    "general.sampling.top_k",
    "general.sampling.top_p",
    "general.sampling.min_p",
];

// PIN(2026-11-21): every tensor of the Mini by role and type (734 in all), [derived] from the
// header dump's per-name table: the attention rows Q8_0 but the sink projection BF16 and the
// small matrices F32; the loop's controllers and queries F32; the routed stacks NVFP4 on the
// first twenty layers and Q8_0 on the last four, their per-expert scales unread F32.
const MINI_TENSORS: &[&str] = &[
    "attn bf16 32",
    "attn f32 345",
    "attn q8_0 144",
    "ffn_norm f32 72",
    "head f32 2",
    "head q8_0 1",
    "routed nvfp4 40",
    "routed q8_0 8",
    "router f32 48",
    "token_embd f32 1",
    "token_embd q8_0 1",
    "unread f32 40",
];

// PIN(2026-11-21): the Small's description: the same program at twice the width, no loop, the
// GPT-4o spelling of the pre-tokenizer, 40 windowed layers of 48 and the windowless at every
// sixth from 5. The log-scaled positions run at the file's floor 128000 — `attention.log_scale_floor`,
// the one key the reference reads — where the checkpoint's config.json says 8192: the description
// carries the reference's value, the divergence recorded here.
const SMALL_VIEW: &[&str] = &[
    "arch Inkling2",
    "hidden 4608 vocab 201024 ctx_train 1048576",
    "rms_eps bits 0x358637bd",
    "layers 48 mtp 0",
    "hc None",
    "engram None",
    "chat pre gpt-4o template bytes 27995 tools None reasoning None",
    "logit scale 1 softcap 30 cut 200058",
    "[0-4,6-10,12-16,18-22,24-28,30-34,36-40,42-46] relbias 48/8 x 128 d_rel 16 extent 1024 clips 22.627417/8 log 128000/0.1 taps 4 win 1024 gate Proj || moe 384/8 ff 2304 situ 4/25 Sigmoid bias true norm true x8 hash false || attn-res-mix 6/1",
    "[5,11,17,23,29,35,41,47] relbias 48/8 x 128 d_rel 16 extent 1024 clips 22.627417/8 log 128000/0.1 taps 4 dyn gate Proj || moe 384/8 ff 2304 situ 4/25 Sigmoid bias true norm true x8 hash false || attn-res-mix 6/3",
];

const SMALL_DEFAULTS: &[&str] = &[
    "router norm = selected-sum (B:65: expert_weights_norm_eps = 1e-20 unconditional)",
    "router logits = f32 (expert_router_f32) (B:64: the router scores run in f32)",
    "router bias = selection only (M:690-696: exp_probs_b steers the choice, the kept weights are the unbiased scores)",
    "V = gainless per-head RMS, always (B:406: the value rows are normed whatever the file's `value_norm` key says)",
    "mix eps = 1e-6 fixed, scale 1/sqrt(hidden) (B:217,222: the block attention residual's states are normed at a fixed epsilon and scored at 1/sqrt(hidden))",
    "conv output = (x + conv(x)) / sqrt(taps + 1), twice the taps where the k/v convs are dynamic (B:282: every convolution's scale counts its taps (and the dynamic taps))",
    "softmax denominator = 1 + sum(exp(s - t)) (B:429-438: a per-(head, query) sink logit t adds a zero column)",
    "head = softcap then the unpadded columns (B:503-510: the softcap runs on the padded logits, the cut after it)",
    "flash attention = off (B:88, llama-graph.cpp:3201: use_flash_attn needs no kq_b, and every layer carries one)",
];

const SMALL_UNREAD: &[&str] = &[
    "inkling2.feed_forward_length",
    "inkling2.d_rel",
    "inkling2.rel_extent",
    "inkling2.rel_extent_swa",
    "inkling2.shortconv_kernel",
    "inkling2.log_scaling_n_floor",
    "inkling2.log_scaling_alpha",
    "inkling2.unpadded_vocab_size",
    "inkling2.qk_clip_threshold",
    "inkling2.rel_clip_threshold",
    "inkling2.attn_res_normalize_values",
    "inkling2.mlp_gate_softcap",
    "inkling2.mlp_up_softcap",
    "inkling2.value_norm",
];

const SMALL_TENSORS: &[&str] = &[
    "attn bf16 48",
    "attn f32 415",
    "attn q6_K 120",
    "attn q8_0 168",
    "ffn_norm f32 144",
    "head f32 2",
    "head q6_K 1",
    "routed iq2_xs 40",
    "routed iq3_xxs 48",
    "routed iq4_xs 8",
    "router f32 96",
    "token_embd f32 1",
    "token_embd q5_K 1",
];

/// One file's description, defaults, unread keys, tensors and coverage items: printed, each
/// compared against its pins.
fn check(path: &str, view: &[&str], defaults: &[&str], unread_pins: &[&str], tensors: &[&str]) {
    let mut o = String::new();
    let mut b = Vec::new();
    let split = Split::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let read = model::arch::spec(&split).unwrap_or_else(|e| panic!("spec of {path}: {e}"));
    let _ = writeln!(o, "file {path}");
    spec_view::compare(
        &mut o,
        &mut b,
        "description",
        &spec_view::view(&read.spec),
        view,
    );
    spec_view::compare(&mut o, &mut b, "defaults taken", &read.defaults, defaults);
    let unread = model::arch::inkling2::hparams::unread_keys(&split);
    spec_view::compare(&mut o, &mut b, "keys not read", &unread, unread_pins);
    let mut by_role: BTreeMap<(String, String), usize> = BTreeMap::new();
    for t in &read.tensors.tensors {
        *by_role
            .entry((t.role.to_string(), t.ty.to_string()))
            .or_default() += 1;
    }
    let got: Vec<String> = by_role
        .iter()
        .map(|((role, ty), n)| format!("{role} {ty} {n}"))
        .collect();
    let _ = writeln!(o, "tensors {}", read.tensors.tensors.len());
    spec_view::compare(&mut o, &mut b, "tensors by role and type", &got, tensors);
    let list = model::arch::coverage::check(&read.spec, &read.tensors);
    let _ = writeln!(o, "coverage (the work queue, printed not pinned)");
    for line in spec_view::items(&list) {
        let _ = writeln!(o, "  {line}");
    }
    println!("{o}");
    assert!(
        b.is_empty(),
        "{} pin(s) differ:\n  {}",
        b.len(),
        b.join("\n  ")
    );
}

#[test]
#[ignore = "needs the Inkling-2-Mini UD-Q4_K_XL file on the box"]
fn hw_inkling2_spec_mini() {
    check(MINI, MINI_VIEW, MINI_DEFAULTS, MINI_UNREAD, MINI_TENSORS);
}

#[test]
#[ignore = "needs the Inkling-2-Small UD-Q2_K_XL shards on the box"]
fn hw_inkling2_spec_small() {
    check(
        SMALL,
        SMALL_VIEW,
        SMALL_DEFAULTS,
        SMALL_UNREAD,
        SMALL_TENSORS,
    );
}
