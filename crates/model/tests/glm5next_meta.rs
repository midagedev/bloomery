//! glm5next metadata gate: what the header reader `arch::glm5next` resolves
//! from the GLM-5.3-Flash file, and what the coverage check lists for it —
//! the parts no program in this tree runs yet, the work queue of the rounds
//! that build one. One contract, one test; it prints what it compared, then
//! fails with the whole list of what differs.
//!
//! `hw_`: needs the six shards on the box (`just gate-glm5next-meta`).
//! The HF map gate reads the real checkpoint headers and small value oracle too
//! (`just gate-glm5next-hf-map`, on the set `just dump-ref-glm5next-hf-map` writes).

#[path = "common/spec_view.rs"]
mod spec_view;

#[path = "common/spec_fail_first.rs"]
mod spec_fail_first;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use gguf::Split;
use hf::quant_config::Config;
use hf::safetensors::Dtype;
use model::arch::glm5next::hf::{Geometry, Layout, map};

#[path = "common/glm_hf.rs"]
mod hf_support;

use hf_support::{apply, gguf_table, gunzip, logical, table};

/// The file: unsloth's `UD-Q4_K_XL`, the quant whose routed stacks fit the
/// host tier. Its first shard holds the header and no tensor.
const GLM: &str = "/models/GLM-5.3-Flash-UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf";

// PIN(2026-09-27): the file's description as `arch::glm5next::spec` reads it, line by line (the key
// values checked against the header dump and ik's glm5next loader, src/llama-hparams.cpp).
// PIN(2026-09-27): the `hc` and latent lines re-pinned when `HcSpec` gained its kind (mHC or
// gated-residual) and `TokenPool` its rule (learned or mean pools); the values are the same.
// PIN(2026-09-27): tools None -> Some(GlmXml): the spec declares the `<tool_call>` markup
// `crates/serve` parses (`glmxml`), which the reader had left undeclared.
const VIEW: &[&str] = &[
    "arch Glm5Next",
    "hidden 4096 vocab 154880 ctx_train 1048576",
    "rms_eps bits 0x3727c5ac",
    "layers 45 mtp 1",
    "hc Some(HcSpec { streams: 4, kind: Mhc { sinkhorn: 20, eps: 1e-6, mix: Own, collapse: Mean } })",
    "engram None",
    "chat pre glm4 template bytes 10648 tools Some(GlmXml) reasoning Some(ThinkSpan)",
    "[0-2] delta Kda { gate: Sigmoid { lower_bound: -5.0 }, beta_scale: 1.0 } k 64 v 64 d 128 conv 4 || dense 12288 swiglu Some(10.0) || hc",
    "[3,7,11,15,19,23,27,31,35,39,43] latent h 64 q 1536 kv 512 Absorbed { qk: 256, v: 256 } rope - qhn false out Plain win None sinks false | TokenPool { heads: 32, d: 128, top_k: 2048, pool: 4, rule: Learned { key_eps: 1e-6 } } || moe 288/8 ff 2048 swiglu Some(10.0) Sigmoid bias true norm true x2.5 hash false shared 2048 swiglu Some(10.0) gate false || hc",
    "[4-6,8-10,12-14,16-18,20-22,24-26,28-30,32-34,36-38,40-42,44] delta Kda { gate: Sigmoid { lower_bound: -5.0 }, beta_scale: 1.0 } k 64 v 64 d 128 conv 4 || moe 288/8 ff 2048 swiglu Some(10.0) Sigmoid bias true norm true x2.5 hash false shared 2048 swiglu Some(10.0) gate false || hc",
];

// PIN(2026-09-27): the next-token layer, carried and not run: ik numbers it 45, the first past the
// trunk (n_layer_kv_from_start).
// PIN(2026-09-27): re-pinned when `TokenPool` gained its rule; the values are the same.
const MTP: &[&str] = &[
    "mtp 45: latent h 64 q 1536 kv 512 Absorbed { qk: 256, v: 256 } rope - qhn false out Plain win None sinks false | TokenPool { heads: 32, d: 128, top_k: 2048, pool: 4, rule: Learned { key_eps: 1e-6 } } || moe 288/8 ff 2048 swiglu Some(10.0) Sigmoid bias true norm true x2.5 hash false shared 2048 swiglu Some(10.0) gate false || plain",
];

// PIN(2026-09-27): the file carries every key the reader reads, so it takes no default.
const DEFAULTS: &[&str] = &[];

// PIN(2026-09-27): the keys the reader does not read: listed, not refused.
const UNREAD: &[&str] = &["general.sampling.top_p", "general.sampling.temp"];

// PIN(2026-09-27): every tensor by role and type (1,412 in all, none in the first shard).
const TENSORS: &[&str] = &[
    "attn f32 315",
    "attn q8_0 405",
    "hc f32 180",
    "hc q8_0 90",
    "router f32 84",
    "ffn_norm f32 45",
    "shexp q8_0 126",
    "dense_ffn q8_0 9",
    "routed q4_K 82",
    "routed q5_K 41",
    "routed q6_K 3",
    "token_embd q8_0 1",
    "head f32 1",
    "head q8_0 1",
    "unused f32 13",
    "unused q4_K 2",
    "unused q5_K 1",
    "unused q8_0 13",
];

// PIN(2026-09-27): the coverage check's list (feature: layers). The glm5next body
// (`gpu-glm5next`) runs every need of the trunk and reads every type the file holds (its routed
// stacks on the host tier, which refuses a type with no host kernel), so nothing is listed.
const COVERAGE: &[&str] = &[];

#[test]
#[ignore = "needs the GLM-5.3-Flash shards on the box (just gate-glm5next-meta)"]
fn hw_glm5next_spec() {
    let mut o = String::new();
    let mut b = Vec::new();
    let split = Split::open(GLM).unwrap_or_else(|e| panic!("open {GLM}: {e}"));
    let read = model::arch::spec(&split).unwrap_or_else(|e| panic!("spec of {GLM}: {e}"));
    let _ = writeln!(o, "file {GLM}");
    spec_view::compare(
        &mut o,
        &mut b,
        "description",
        &spec_view::view(&read.spec),
        VIEW,
    );
    let n_trunk = read.spec.layers.len();
    let mtp: Vec<String> = (n_trunk..)
        .zip(&read.spec.mtp)
        .map(|(l, layer)| format!("mtp {l}: {}", spec_view::layer_line(layer)))
        .collect();
    spec_view::compare(&mut o, &mut b, "next-token layers", &mtp, MTP);
    spec_view::compare(&mut o, &mut b, "defaults taken", &read.defaults, DEFAULTS);
    let unread = model::arch::glm5next::hparams::unread_keys(&split);
    spec_view::compare(&mut o, &mut b, "keys not read", &unread, UNREAD);
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
            "gpu-deepseek41/src/hc.rs HC_STREAMS (hc_pre_q8_0)",
            "hyper-connections of 4 streams: 0-44",
        ),
        (
            "gpu-glm5next/src/ffn.rs (the shared expert)",
            "shared expert, ff 2048: 3-44",
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

/// The glm5next stems whose values the map gate compares, each with the comparison: every mapped
/// tensor of these stems must be in the oracle's `gguf.safetensors`, and no other.
///
/// | stem | from | compared |
/// |---|---|---|
/// | `ssm_a` | `A_log`, [`Layout::NegExp`] | F32, one unit in the last place |
/// | `ssm_dt.bias` | `dt_bias` | F32, byte-equal |
/// | `ssm_conv1d_{q,k,v}.weight` | `{q,k,v}_conv1d` | F32, byte-equal |
/// | `exp_probs_b.bias` | `e_score_correction_bias` | F32, byte-equal |
/// | `hc_{attn,ffn}_{base,scale}.weight` | `hc_{attn,ffn}_{base,scale}` | BF16, the GGUF's F32 rounded |
const VALUES: &[&str] = &[
    "ssm_a",
    "ssm_dt.bias",
    "ssm_conv1d_q.weight",
    "ssm_conv1d_k.weight",
    "ssm_conv1d_v.weight",
    "exp_probs_b.bias",
    "hc_attn_base.weight",
    "hc_attn_scale.weight",
    "hc_ffn_base.weight",
    "hc_ffn_scale.weight",
];

/// Every layer of nvidia/GLM-5.3-Flash-NVFP4 (the shard headers of the map oracle,
/// `refset::arch::glm5next::HF_MAP`) maps to the names and dims of unsloth's UD-Q4_K_XL table
/// (`tensors.tsv`) for layers 0 to 44 and the unlayered tensors; every layer is in both; and the
/// [`VALUES`] tensors range-read from both (`hf.safetensors`, `gguf.safetensors`), all of them,
/// are equal after the map's layout as the table there states.
#[test]
#[ignore = "just gate-glm5next-hf-map ($BLOOMERY_DATA/ref-glm5next-hf-map)"]
fn hw_glm_map_matches_the_unsloth_gguf() {
    use refset::arch::glm5next::{HF_MAP, HF_MAP_SET};
    use refset::hfmap::{GGUF, HF, HfMapSet, TENSORS};
    let set = HfMapSet::open(&HF_MAP.path(HF_MAP_SET), &HF_MAP).unwrap_or_else(|e| panic!("{e}"));
    let hf = concat!(env!("CARGO_MANIFEST_DIR"), "/../hf/tests/fixtures");
    let config = Config::parse(
        &gunzip(&format!(
            "{hf}/quant-configs/nvidia__GLM-5.3-Flash-NVFP4__config.json.gz"
        )),
        Some(&gunzip(&format!(
            "{hf}/quant-configs/nvidia__GLM-5.3-Flash-NVFP4__hf_quant_config.json.gz"
        ))),
    )
    .unwrap();
    let mut all = Vec::new();
    for header in set.headers() {
        all.extend(table(&gunzip(header.to_str().unwrap())));
    }
    let tensors = logical(&config, &all);
    let got = map(
        hf_support::GLM,
        tensors.iter().map(|(n, s)| (n.as_str(), s.as_slice())),
    )
    .unwrap();
    let dims: BTreeMap<String, Vec<usize>> = got
        .iter()
        .map(|(n, m)| (n.clone(), m.dims.clone()))
        .collect();
    let tsv = std::fs::read_to_string(set.dir.join(TENSORS)).unwrap();
    /// A layer's tensor: its layer and its stem.
    fn blk(n: &str) -> Option<(usize, &str)> {
        let (l, stem) = n.strip_prefix("blk.")?.split_once('.')?;
        Some((l.parse().unwrap(), stem))
    }
    let want = gguf_table(&tsv, |n| {
        blk(n).is_none_or(|(l, _)| l < hf_support::GLM.layers)
    });
    assert_eq!(dims.len(), want.len());
    assert_eq!(dims, want);
    let missing: Vec<usize> = (0..hf_support::GLM.layers)
        .filter(|&l| !want.keys().any(|n| blk(n).is_some_and(|b| b.0 == l)))
        .collect();
    assert!(
        missing.is_empty(),
        "layers {missing:?} are not in the oracle"
    );
    println!("names and dims equal: {} tensors", dims.len());

    let small = hf::safetensors::Safetensors::open(&set.dir.join(HF)).unwrap();
    let gguf = hf::safetensors::Safetensors::open(&set.dir.join(GGUF)).unwrap();
    let held: BTreeSet<&str> = gguf.names().collect();
    let required: BTreeSet<&str> = got
        .keys()
        .map(String::as_str)
        .filter(|n| blk(n).is_some_and(|(_, stem)| VALUES.contains(&stem)))
        .collect();
    let absent: Vec<&str> = VALUES
        .iter()
        .copied()
        .filter(|v| !required.iter().any(|n| blk(n).is_some_and(|b| b.1 == *v)))
        .collect();
    assert!(
        absent.is_empty(),
        "the map gives no {absent:?} tensor to compare"
    );
    let unheld: Vec<_> = required.difference(&held).collect();
    assert!(
        unheld.is_empty(),
        "the oracle holds no values of {unheld:?}"
    );
    let extra: Vec<_> = held.difference(&required).collect();
    assert!(
        extra.is_empty(),
        "the oracle holds {extra:?}, which VALUES does not compare"
    );
    let (mut equal, mut rounded, mut last_place) = (0, 0, 0);
    for name in required {
        let m = &got[name];
        let raw = small.raw(&m.source).unwrap();
        let want = gguf.raw(name).unwrap().bytes;
        match raw.dtype {
            // The converter's `torch.exp` is not correctly rounded, so `-exp` may differ in
            // the last place: `ssm_a` is held to one unit there.
            Dtype::F32 if m.layout != Layout::Same => {
                let out = apply(&m.layout, raw.bytes, raw.shape, 4);
                assert_eq!(out.len(), want.len(), "{name}: converted byte count");
                for (&o, &w) in out.as_chunks::<4>().0.iter().zip(want.as_chunks::<4>().0) {
                    let o = i32::from_le_bytes(o);
                    let w = i32::from_le_bytes(w);
                    assert!((o - w).abs() <= 1, "{name}: {o:#x} against {w:#x}");
                    last_place += usize::from(o != w);
                }
                equal += 1;
            }
            Dtype::F32 => {
                assert_eq!(
                    apply(&m.layout, raw.bytes, raw.shape, 4),
                    want,
                    "{name} from {}",
                    m.source
                );
                equal += 1;
            }
            // NVIDIA's export stores these BF16 where the GGUF's source kept F32: each BF16
            // value is the GGUF's F32 rounded to nearest even.
            Dtype::Bf16 => {
                let out = apply(&m.layout, raw.bytes, raw.shape, 2);
                let round = |b: [u8; 4]| {
                    let x = u32::from_le_bytes(b);
                    ((x + 0x7fff + ((x >> 16) & 1)) >> 16) as u16
                };
                let want: Vec<u8> = want
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .flat_map(|&b| round(b).to_le_bytes())
                    .collect();
                assert_eq!(out, want, "{name} from {}", m.source);
                rounded += 1;
            }
            d => panic!("{}: {}", m.source, d.name()),
        }
    }
    println!(
        "F32 equal: {equal} tensors, {last_place} ssm_a values one unit apart; \
         BF16 equal to the GGUF's F32 rounded: {rounded}"
    );
}
