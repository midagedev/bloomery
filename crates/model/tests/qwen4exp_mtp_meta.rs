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
