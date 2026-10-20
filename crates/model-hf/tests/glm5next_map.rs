//! The GLM-5.3-Flash checkpoint map gate: `model_hf::arch::glm5next::map` against the real checkpoint
//! headers and a small value oracle (`just gate-glm5next-hf-map`, on the set
//! `just dump-ref-glm5next-hf-map` writes).

use std::collections::{BTreeMap, BTreeSet};

use hf::quant_config::Config;
use hf::safetensors::Dtype;
use model_hf::arch::glm5next::{Geometry, Layout, map};

#[path = "common/glm_hf.rs"]
mod hf_support;

use hf_support::{apply, gguf_table, gunzip, logical, table};

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
