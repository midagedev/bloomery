//! What the fixtures of the hybrid families (Mamba-2 and attention layers
//! under one trunk: `nemotron_h`, `nemotron_h_moe`) share: the trunk's key
//! rules and F32 constants, the layer-map checks that keep the source's
//! kinds and every (tensor stem, type) pair, and the comparison of a written
//! fixture's trunk with the source's. A family's own file holds what its
//! architecture forces on top (its feed-forward kind, its experts). The map
//! check serves any family whose layers have kinds (`solar-open2`'s KDA and
//! GQA layers use it without the Mamba-2 trunk).

use std::collections::BTreeMap;
use std::fmt::Debug;

use gguf::Split;

use super::{FixtureError, KeyRule, split_layer};
use crate::arch::hybrid::{ATTN_NORM, SSM_A, SSM_CONV_BIAS, SSM_D, SSM_DT_BIAS, SSM_NORM, Ssm};

/// The scan's constants of one family: the real file's medians of `ssm_a`,
/// `ssm_d` and `ssm_dt.bias` [measured per family]. `ssm_a` is stored
/// `−exp(A_log)`, so it is negative and `dA = exp(dt·A)` stays below 1 for any
/// `dt > 0`; the heads span several decades and a constant is the median.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Scan {
    pub a: f32,
    pub d: f32,
    pub dt_bias: f32,
}

/// The value of an F32 trunk tensor by its name past `blk.N.`: the norms'
/// gains 1, including the grouped `ssm_norm` (a `[d_inner / groups, groups]`
/// matrix, still a gain), the conv's bias 0, and the scan's three vectors
/// ([`Scan`]); every head shares each constant. `None` is a leaf the trunk
/// does not name.
pub(crate) fn trunk_const(leaf: &str, scan: &Scan) -> Option<f32> {
    let rules: [(&str, f32); 7] = [
        (ATTN_NORM, 1.0),
        ("output_norm.weight", 1.0),
        (SSM_NORM, 1.0),
        (SSM_CONV_BIAS, 0.0),
        (SSM_A, scan.a),
        (SSM_D, scan.d),
        (SSM_DT_BIAS, scan.dt_bias),
    ];
    rules.iter().find(|(n, _)| *n == leaf).map(|&(_, v)| v)
}

/// The rule of an architecture key (past `<arch>.`) the hybrid trunk reads:
/// the copy list is every such key that is not a function of the map;
/// `nextn_predict_layers` may be present and must be 0 (the readers run no
/// next-token layer). `None` is a key the trunk does not name.
pub(crate) fn trunk_key_rule(suffix: &str) -> Option<KeyRule> {
    const COPY: [&str; 15] = [
        "context_length",
        "embedding_length",
        "vocab_size",
        "attention.head_count",
        "attention.layer_norm_rms_epsilon",
        "attention.layer_norm_epsilon",
        "attention.key_length",
        "attention.value_length",
        "rope.dimension_count",
        "rope.scaling.finetuned",
        "ssm.conv_kernel",
        "ssm.state_size",
        "ssm.group_count",
        "ssm.inner_size",
        "ssm.time_step_rank",
    ];
    Some(match suffix {
        "block_count" => KeyRule::BlockCount,
        "attention.head_count_kv" | "feed_forward_length" => KeyRule::PerLayer,
        "nextn_predict_layers" => KeyRule::Zero("the reader runs no next-token layer"),
        s if COPY.contains(&s) => KeyRule::Copy,
        _ => return None,
    })
}

/// A [`FixtureError::Mismatch`] of the layer map.
pub(crate) fn map_error(detail: String) -> FixtureError {
    FixtureError::Mismatch {
        what: "layer map".to_string(),
        detail,
    }
}

/// Every (tensor stem, type name) of `split`'s layer tensors with the first
/// layer that holds the pair, for the layers `keep` accepts.
fn pairs(split: &Split, keep: impl Fn(usize) -> bool) -> BTreeMap<(String, String), usize> {
    let mut out = BTreeMap::new();
    for (_, t) in split.iter_tensors() {
        if let Some((l, leaf)) = split_layer(&t.name)
            && keep(l)
        {
            out.entry((leaf.to_string(), t.ty.to_string())).or_insert(l);
        }
    }
    out
}

/// `layers` keeps the source's facts: one layer of each of `kept` kinds (of
/// `kinds`, the source's per layer), and every (stem, type) pair any source
/// layer holds. A quant type that appears in the source on one layer only
/// (`attn_v` Q8_0 on two of four, `ffn_down` Q4_K on eight of seventeen) is a
/// kernel dispatch the gate would not reach if the map dropped it, and a kind
/// the map drops is a launch the fixture never makes. Each refusal is a
/// [`FixtureError::Mismatch`] on the layer map, naming the kind, or the pair
/// and the first source layer that holds it.
pub(crate) fn check_map<K: Copy + Debug + PartialEq>(
    kinds: &[K],
    kept: &[K],
    source: &Split,
    layers: &[usize],
) -> Result<(), FixtureError> {
    let at_map: Vec<K> = layers
        .iter()
        .filter_map(|&l| kinds.get(l).copied())
        .collect();
    for kind in kept {
        if !at_map.contains(kind) {
            return Err(map_error(format!(
                "{layers:?} holds no {kind:?} layer (kinds {at_map:?}): one of each is kept"
            )));
        }
    }
    let held = pairs(source, |l| layers.contains(&l));
    if let Some(((leaf, ty), l)) = pairs(source, |_| true)
        .into_iter()
        .find(|(pair, _)| !held.contains_key(pair))
    {
        return Err(map_error(format!(
            "{layers:?} holds no {leaf} of type {ty}, which source layer {l} does"
        )));
    }
    Ok(())
}

/// The header facts of a hybrid family's `Hparams` that a fixture keeps: the
/// trunk's globals, the Mamba-2 widths and the per-layer arrays.
pub(crate) struct Trunk<'a, K> {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_ctx: usize,
    pub n_vocab: usize,
    pub n_head: usize,
    pub key_len: usize,
    pub value_len: usize,
    pub rms_eps: f32,
    pub ssm: &'a Ssm,
    pub kv: &'a [usize],
    /// `feed_forward_length` per layer where the family's reader keeps it
    /// (the Nano's kind marker); `None` where the layers' widths are held
    /// elsewhere (the Bolt's experts).
    pub ff: Option<&'a [usize]>,
    pub kinds: &'a [K],
}

/// `fixture`'s trunk is `layers`' source layers' (`source`'s): its layer
/// count, each layer's KV head count, feed-forward length and kind, and the
/// source's globals and Mamba-2 widths. The first difference is a
/// [`FixtureError::Mismatch`] naming it.
pub(crate) fn check_trunk<K: Copy + Debug + PartialEq>(
    layers: &[usize],
    fixture: &Trunk<'_, K>,
    source: &Trunk<'_, K>,
) -> Result<(), FixtureError> {
    let (fx, src) = (fixture, source);
    let bad = |what: String, detail: String| FixtureError::Mismatch { what, detail };
    if fx.n_layer != layers.len() {
        return Err(bad(
            "layer count".into(),
            format!("{}, not {}", fx.n_layer, layers.len()),
        ));
    }
    for (f, &l) in layers.iter().enumerate() {
        let what = format!("layer {f} (source {l})");
        let ff = fx
            .ff
            .zip(src.ff)
            .map(|(a, b)| ("feed-forward length", a[f], b[l]));
        for (key, a, b) in std::iter::once(("kv heads", fx.kv[f], src.kv[l])).chain(ff) {
            if a != b {
                return Err(bad(what, format!("{key} {a}, the source's {b}")));
            }
        }
        if fx.kinds[f] != src.kinds[l] {
            return Err(bad(
                what,
                format!("kind {:?}, the source's {:?}", fx.kinds[f], src.kinds[l]),
            ));
        }
    }
    for (key, a, b) in [
        ("n_embd", fx.n_embd, src.n_embd),
        ("n_ctx", fx.n_ctx, src.n_ctx),
        ("n_vocab", fx.n_vocab, src.n_vocab),
        ("n_head", fx.n_head, src.n_head),
        ("key_len", fx.key_len, src.key_len),
        ("value_len", fx.value_len, src.value_len),
    ] {
        if a != b {
            return Err(bad(key.into(), format!("{a}, want {b}")));
        }
    }
    if fx.ssm != src.ssm {
        return Err(bad(
            "ssm".into(),
            format!("{:?}, the source's {:?}", fx.ssm, src.ssm),
        ));
    }
    if fx.rms_eps.to_bits() != src.rms_eps.to_bits() {
        return Err(bad(
            "attention.layer_norm_rms_epsilon".into(),
            format!("{}, the source's {}", fx.rms_eps, src.rms_eps),
        ));
    }
    Ok(())
}
