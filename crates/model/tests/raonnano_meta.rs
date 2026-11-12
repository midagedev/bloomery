//! raon1nano metadata gate: what the header reader `arch::raon1` resolves from the Raon-1-nano-preview file (the
//! reader serves both dialects; this is the nano one's own pins), what the coverage check lists for it, and the
//! stores its layers hold. One contract, one test a file; each prints what it compared, then fails with the whole
//! list of what differs.
//!
//! `hw_`: needs the file on the box (`just gate-raonnano-meta`). Headers only: seconds.

#[path = "common/spec_view.rs"]
mod spec_view;

use std::collections::BTreeMap;
use std::fmt::Write as _;

use gguf::Split;

/// The Q8_0 file (KRAFTON's conversion, one shard): the family's real file and its own fixture. Every weight Q8_0
/// but the small mixers' tensors (f16 and f32).
const Q8: &str = "/data/models/Raon-1-nano-gguf/raon-1-nano-Q8_0.gguf";

// PIN(2026-11-13): the file's description as `arch::raon1::spec` reads it, line by line (the key values checked
// against the Q8_0 header dump, 24 layers = 18 delta + 6 latent at 3, 7, .., 23; `attention.head_count_kv` is 1 at
// the latent layers and 0 elsewhere, and a dense FFN sits in every layer: no `leading_dense_block_count` key). The
// delta layers have 16 heads of 128 (their own count, half the latent layers' 32 query heads over a 512-value
// latent with 64 unrotated key values). The chart is the folded parametrization; the head is the token table.
const VIEW: &[&str] = &[
    "arch Raon1Nano",
    "hidden 1536 vocab 259840 ctx_train 262144",
    "rms_eps bits 0x358637bd",
    "layers 24 mtp 0",
    "hc Some(HcSpec { streams: 4, kind: Chart { eps: 1e-5, magnitude: 1.0, collapse: Mean } })",
    "engram None",
    "chat pre raon template bytes 6735 tools Some(QwenXml) reasoning Some(ThinkSpan { open: \"<think>\", close: \"</think>\" })",
    "[0-2,4-6,8-10,12-14,16-18,20-22] delta Rda { gate: Sigmoid { lower_bound: -5.0 } } k 16 v 16 d 128 conv 4 || dense 6144 swiglu None || hc [PerLayerEmbed { width: 256 }]",
    "[3,7,11,15,19,23] latent h 32 q 512 kv 512 Absorbed { qk: 192, v: 128 } rope - qhn false out Plain win None sinks false pe 64 gate Proj || dense 6144 swiglu None || hc [PerLayerEmbed { width: 256 }]",
];

// PIN(2026-11-13): the file's own keys state no activation, per-layer input layout, dense lead or tie: each is the
// description row's, and the row's line names it.
const DEFAULTS: &[&str] = &[
    "ffn act = SwiGLU (llama.cpp src/models/raon1nano.cpp:152-154: one LLM_FFN_SWIGLU on ffn_up's [gate | up] rows)",
    "per-layer input = token table, layout 2 (llama.cpp src/models/raon1nano.cpp:367-383: p = (RMSNorm(W_mp·e) + √n_ple·table[t])/√2; hub modeling_raon1_nano.py:495-502)",
    "leading_dense_block_count = 24 (absent; a dense FFN in every layer, src/models/raon1nano.cpp:23-24)",
    "head = tied (the file carries no output tensor and LG's config ties the word embeddings (hub modeling_raon1_nano.py: tie_word_embeddings))",
];

// PIN(2026-11-13): the keys the reader does not read: listed, not refused. The file carries none beyond its own.
const UNREAD: &[&str] = &[];

// PIN(2026-11-13): every tensor of the Q8_0 file by role and type (497 in all): the attention 204 = the fused
// q|k|v|g stems, the conv tap, the decay's fused rows, the norms and the two latent dialects' tensors in f16 54
// (18 each of `ssm_conv1d_qkv`, `ssm_f_a`, `ssm_f_b`), f32 90 (24 `attn_norm`, 18 `ssm_dt`, the latent norms and
// gates), q8_0 60 (18 each of `attn_qkv`, `attn_output`, `ssm_g`); the hyper-connections f16 48 (24 each of the
// two charts) and f32 96; the dense block q8_0 48 (24 each of `ffn_up`, `ffn_down`); the per-layer input's table
// q8_0 1, its projection and norm f32 among `engram_dense`, its per-layer weights f16 1; the token table q8_0 1
// and the tied head's norm f32 1.
const TENSORS: &[&str] = &[
    "attn f16 54",
    "attn f32 90",
    "attn q8_0 60",
    "dense_ffn q8_0 48",
    "engram_dense f16 1",
    "engram_dense f32 25",
    "engram_dense q8_0 48",
    "engram_table q8_0 1",
    "ffn_norm f32 24",
    "hc f16 48",
    "hc f32 96",
    "head f32 1",
    "token_embd q8_0 1",
];

// PIN(2026-11-13): the coverage check lists nothing for the Q8_0 file: the body's program runs every kind it names
// (the delta rule's fused stems, the folded chart, the per-layer input, the dense swiglu block, the latent
// attention with its unrotated values and output gate, the tied head).
const COVERAGE: &[&str] = &[];

#[test]
#[ignore = "needs the Raon-1-nano-preview Q8_0 file on the box"]
fn hw_raonnano_spec_q8_0() {
    let mut o = String::new();
    let mut b = Vec::new();
    let split = Split::open(Q8).unwrap_or_else(|e| panic!("open {Q8}: {e}"));
    let read = model::arch::spec(&split).unwrap_or_else(|e| panic!("spec of {Q8}: {e}"));
    let _ = writeln!(o, "file {Q8}");
    spec_view::compare(
        &mut o,
        &mut b,
        "description",
        &spec_view::view(&read.spec),
        VIEW,
    );
    spec_view::compare(&mut o, &mut b, "defaults taken", &read.defaults, DEFAULTS);
    let unread = model::arch::raon1::hparams::unread_keys(&split);
    spec_view::compare(&mut o, &mut b, "keys not read", &unread, UNREAD);
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
    spec_view::compare(&mut o, &mut b, "tensors by role and type", &got, TENSORS);
    let list = model::arch::coverage::check(&read.spec, &read.tensors);
    spec_view::compare(
        &mut o,
        &mut b,
        "coverage",
        &spec_view::items(&list),
        COVERAGE,
    );
    println!("{o}");
    assert!(
        b.is_empty(),
        "{} pin(s) differ:\n  {}",
        b.len(),
        b.join("\n  ")
    );
}

// PIN(2026-11-13): the stores of the file's layers [derived] from the description alone: a delta layer 16 heads x
// 128 x 128 f32 (1,048,576 B) + the conv ring 11 rows x 3 x 2048 f32 (270,336 B) + one lane's u32 stamp, 18 layers;
// a latent layer a row of (512 + 64) f16 values = 1,152 B a position, 6 layers (6,912 B a position); no rope rows
// (NoPE). Mutants: the stamp or the pass rows of the ring left out, the unrotated values left out of the row, a
// window on a latent layer.
#[test]
#[ignore = "needs the Raon-1-nano-preview Q8_0 file on the box"]
fn hw_raonnano_stores() {
    use model::arch::raon1::place::PlanInputs;
    use model::placement::KvBytes;

    const CTX: u64 = 32_768;
    const DELTA_LAYERS: u64 = 18;
    const DELTA_STATE: u64 = 16 * 128 * 128 * 4 + 11 * 3 * 2048 * 4 + 4;
    const LATENT_POSITION: u64 = 6 * (512 + 64) * 2;

    let mut o = String::new();
    let mut b = Vec::new();
    let split = Split::open(Q8).unwrap_or_else(|e| panic!("open {Q8}: {e}"));
    let inputs =
        PlanInputs::describe(&split).unwrap_or_else(|e| panic!("plan inputs of {Q8}: {e}"));
    let layers = inputs.spec.layers.len();
    let _ = writeln!(o, "file {Q8}, {layers} layers, ctx {CTX}");
    row(
        &mut o,
        &mut b,
        "kv bytes a position over the trunk",
        inputs.kv.bytes(0..layers, 1),
        DELTA_LAYERS * DELTA_STATE + LATENT_POSITION,
    );
    row(
        &mut o,
        &mut b,
        "kv bytes at the context",
        inputs.kv.bytes(0..layers, CTX),
        DELTA_LAYERS * DELTA_STATE + LATENT_POSITION * CTX,
    );
    row(
        &mut o,
        &mut b,
        "bytes a position of the latent layers",
        LATENT_POSITION,
        6_912,
    );
    row(
        &mut o,
        &mut b,
        "stage rows bytes",
        inputs.kv.stage_position_bytes(CTX),
        0,
    );
    println!("{o}");
    assert!(
        b.is_empty(),
        "{} pin(s) differ:\n  {}",
        b.len(),
        b.join("\n  ")
    );
}

/// One compared value: a line of the printed table, and an entry of `bad` when it differs.
fn row<T: PartialEq + std::fmt::Debug>(
    out: &mut String,
    bad: &mut Vec<String>,
    what: &str,
    got: T,
    want: T,
) {
    let mark = if got == want { "ok " } else { "BAD" };
    let _ = writeln!(out, "  {mark} {what:<38} {got:?}");
    if got != want {
        bad.push(format!("{what}: got {got:?}, want {want:?}"));
    }
}
