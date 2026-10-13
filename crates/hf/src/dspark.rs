//! A DSpark draft directory's geometry: `config.json` (and the `d2t` tensor of a reduced
//! vocabulary) read into the runtime's [`DsparkShape`], and the dense backbone's own numbers
//! ([`Backbone`]).
//!
//! Two layouts of the same draft are read:
//! - **speculators** (`algorithm: dspark`, RedHatAI's Qwen3.6 draft): `aux_hidden_state_layer_ids`,
//!   `block_size`, `draft_vocab_size`, `mask_token_id`, `markov_rank`, `enable_confidence_head` at
//!   the top and the backbone's numbers in `transformer_layer_config`.
//! - **DFlash** (MiniCPM's draft): `dflash_config` holds `target_layer_ids`, `mask_token_id` and the
//!   heads' switches, and the backbone's numbers sit at the top.
//!
//! # Tap convention
//!
//! [`DsparkShape::taps`] is "the residual row entering layer `L`" (HF `hidden_states[L]`).
//! - speculators writes `aux_hidden_state_layer_ids` in that convention: vLLM's layer loop captures
//!   index `L` after layer `L - 1` ran (`vllm/model_executor/models/qwen3_next.py`:
//!   `self._maybe_add_hidden_state(aux_hidden_states, layer_idx + 1, hidden_states, residual)`),
//!   and its speculators reader hands the same ids to the target and `id - 1` to the draft
//!   (`vllm/transformers_utils/configs/speculators/algos.py`, `update_dspark`:
//!   `pre_trained_config["target_layer_ids"] = [i - 1 for i in aux_layer_ids]`). The ids are taps
//!   as they stand.
//! - DFlash's `target_layer_ids` is the other convention, "leaving layer `L`":
//!   `hidden_states[layer_id + offset]` with `offset = 1` (z-lab/dflash `dflash/model.py`,
//!   `extract_context_feature`), which speculators' converter restates (`src/speculators/convert/
//!   dflash/converter.py`: "z-lab reads hidden_states[layer_id + 1] ... while speculators uses the
//!   layer id directly"). This reader adds 1.
//!
//! # Row 0
//!
//! [`DsparkShape::row0_drafts`] is the file's `sample_from_anchor`: with it, block slot `k`
//! predicts token `p + k + 1` and row 0 (the token at the target's position) drafts id 0 (speculators
//! `src/speculators/models/dspark/core.py`: "With sample_from_anchor=True (DSpark default), slot k
//! predicts token p+k+1"); without it row 0 is only the anchor (DFlash). The key sits at the top of a
//! speculators config and in `dflash_config` of the DFlash layout, and a file that states none is
//! read as DSpark's default, true (vLLM `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py`:
//! `getattr(hf_config, "sample_from_anchor", True)`).
//!
//! # The id map
//!
//! `d2t` holds offsets: `target_id = draft_id + d2t[draft_id]` (speculators
//! `src/speculators/train/vocab_mapping.py`: "Store offset: target_token_id = draft_idx +
//! draft_to_target[draft_idx]"; vLLM `vllm/model_executor/models/qwen3_dspark.py`:
//! `return draft_ids + self.draft_id_to_target_id[draft_ids]`).

use runtime::dspark::{DraftVocab, DsparkError, DsparkShape};
use serde_json::Value;

/// Why a draft's config or id map was refused.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("dspark config: not JSON: {0}")]
    Json(String),
    #[error("dspark config: `{0}` is missing or null")]
    Missing(&'static str),
    #[error("dspark config: `{key}` is {got}; want {want}")]
    Type {
        key: &'static str,
        got: String,
        want: &'static str,
    },
    #[error(
        "dspark config: neither `dflash_config` (a DFlash layout) nor `aux_hidden_state_layer_ids` \
         (a speculators layout) is present"
    )]
    Layout,
    #[error(
        "dspark config: {draft} draft ids of the target's {target} need the `d2t` map, which was not given"
    )]
    NoMap { draft: usize, target: usize },
    #[error("dspark config: `d2t` holds {got} offsets, `draft_vocab_size` is {want}")]
    MapLen { got: usize, want: usize },
    #[error("dspark config: `d2t` sends draft id {index} to {value}, which is no u32 id")]
    MapValue { index: usize, value: i64 },
    #[error(
        "dspark config: `layer_types` mixes sliding and full attention; the dense layers share one window"
    )]
    MixedLayers,
    #[error("dspark config: {0}")]
    Unsupported(String),
    #[error(transparent)]
    Shape(#[from] DsparkError),
}

/// The dense backbone's numbers, as the file's config states them.
#[derive(Clone, Debug, PartialEq)]
pub struct Backbone {
    /// Decoder layers.
    pub layers: usize,
    /// The draft's own hidden width.
    pub hidden: usize,
    /// Query heads.
    pub heads: usize,
    /// Key/value heads.
    pub kv_heads: usize,
    /// Width of one head; the rotation covers all of it.
    pub head_dim: usize,
    /// The MLP's inner width.
    pub intermediate: usize,
    /// The RMS norms' epsilon.
    pub eps: f32,
    /// The rotary base.
    pub rope_theta: f64,
    /// How many context rows back each layer reads: `None` is the whole context.
    pub window: Option<usize>,
    /// Whether a block row attends to the block's earlier rows only (`true`), or to every row of
    /// the block (`false`).
    pub block_causal: bool,
}

struct Config {
    root: Value,
}

impl Config {
    fn parse(text: &str) -> Result<Config, Error> {
        let root: Value = serde_json::from_str(text).map_err(|e| Error::Json(e.to_string()))?;
        if !root.is_object() {
            return Err(Error::Json("the top level is not an object".into()));
        }
        Ok(Config { root })
    }

    /// The DFlash layout's switches, when that layout is the file's.
    fn dflash(&self) -> Option<&Value> {
        self.root.get("dflash_config").filter(|v| v.is_object())
    }

    /// The object that holds the backbone's numbers.
    fn backbone(&self) -> Result<&Value, Error> {
        if self.dflash().is_some() {
            return Ok(&self.root);
        }
        self.root
            .get("transformer_layer_config")
            .filter(|v| v.is_object())
            .ok_or(Error::Missing("transformer_layer_config"))
    }

    fn layout(&self) -> Result<Layout, Error> {
        if self.dflash().is_some() {
            Ok(Layout::Dflash)
        } else if self.root.get("aux_hidden_state_layer_ids").is_some() {
            Ok(Layout::Speculators)
        } else {
            Err(Error::Layout)
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Layout {
    Speculators,
    Dflash,
}

fn kind(v: &Value) -> String {
    v.to_string()
}

fn field<'a>(obj: &'a Value, key: &'static str) -> Result<&'a Value, Error> {
    obj.get(key)
        .filter(|v| !v.is_null())
        .ok_or(Error::Missing(key))
}

fn uint(obj: &Value, key: &'static str) -> Result<u64, Error> {
    let v = field(obj, key)?;
    v.as_u64().ok_or_else(|| Error::Type {
        key,
        got: kind(v),
        want: "an unsigned integer",
    })
}

fn size(obj: &Value, key: &'static str) -> Result<usize, Error> {
    let n = uint(obj, key)?;
    usize::try_from(n).map_err(|_| Error::Type {
        key,
        got: n.to_string(),
        want: "a size",
    })
}

fn flag(obj: &Value, key: &'static str, default: bool) -> Result<bool, Error> {
    match obj.get(key).filter(|v| !v.is_null()) {
        None => Ok(default),
        Some(v) => v.as_bool().ok_or_else(|| Error::Type {
            key,
            got: kind(v),
            want: "a boolean",
        }),
    }
}

/// `key` of `dflash` when that layout holds it there, else of the top level.
fn switch<'a>(cfg: &'a Config, key: &'static str) -> Option<&'a Value> {
    cfg.dflash()
        .and_then(|d| d.get(key))
        .filter(|v| !v.is_null())
        .or_else(|| cfg.root.get(key).filter(|v| !v.is_null()))
}

/// The layers' attention reach: the window all of them share, or `None` for the whole context.
fn window_of(cfg: &Config) -> Result<Option<usize>, Error> {
    let bb = cfg.backbone()?;
    let use_swa = match cfg.dflash() {
        Some(d) => flag(d, "use_swa", false)?,
        None => false,
    };
    let sliding = match bb.get("layer_types").filter(|v| !v.is_null()) {
        Some(Value::Array(types)) => {
            let n = types
                .iter()
                .filter(|t| t.as_str() == Some("sliding_attention"))
                .count();
            match n {
                0 => use_swa,
                n if n == types.len() => true,
                _ => return Err(Error::MixedLayers),
            }
        }
        Some(other) => {
            return Err(Error::Type {
                key: "layer_types",
                got: kind(other),
                want: "an array",
            });
        }
        None => use_swa || flag(bb, "use_sliding_window", false)?,
    };
    if !sliding {
        return Ok(None);
    }
    let w = match cfg.dflash().and_then(|d| d.get("swa_window_size")) {
        Some(v) if !v.is_null() => v,
        _ => field(bb, "sliding_window")?,
    };
    let n = w.as_u64().ok_or_else(|| Error::Type {
        key: "sliding_window",
        got: kind(w),
        want: "an unsigned integer",
    })?;
    Ok(Some(usize::try_from(n).map_err(|_| Error::Type {
        key: "sliding_window",
        got: n.to_string(),
        want: "a size",
    })?))
}

/// The geometry `config_json` states, with the `d2t` offsets of a reduced vocabulary when the
/// draft's head scores fewer ids than the target has.
///
/// The shape is checked ([`DsparkShape::check`]) before it is returned: a map that is not
/// injective or that leaves the target's vocabulary is a named error here, not a wrong id later.
pub fn shape(config_json: &str, d2t: Option<&[i64]>) -> Result<DsparkShape, Error> {
    let cfg = Config::parse(config_json)?;
    let layout = cfg.layout()?;
    let bb = cfg.backbone()?;
    let dflash = cfg.dflash();

    let block = size(&cfg.root, "block_size")?;
    let taps: Vec<usize> = match layout {
        Layout::Speculators => ids(&cfg.root, "aux_hidden_state_layer_ids")?,
        // "leaving layer L" is "entering layer L + 1".
        Layout::Dflash => ids(dflash.unwrap_or(&cfg.root), "target_layer_ids")?
            .into_iter()
            .map(|l| l + 1)
            .collect(),
    };
    // The target's hidden width: the file's `target_hidden_size`, else the draft's own.
    let hidden = match cfg.root.get("target_hidden_size").filter(|v| !v.is_null()) {
        Some(_) => size(&cfg.root, "target_hidden_size")?,
        None => size(bb, "hidden_size")?,
    };
    let n_target = size(bb, "vocab_size")?;
    let draft_rows = match cfg.root.get("draft_vocab_size").filter(|v| !v.is_null()) {
        Some(_) => size(&cfg.root, "draft_vocab_size")?,
        None => n_target,
    };
    let vocab = if draft_rows == n_target {
        DraftVocab::Full { n: n_target }
    } else {
        let d2t = d2t.ok_or(Error::NoMap {
            draft: draft_rows,
            target: n_target,
        })?;
        if d2t.len() != draft_rows {
            return Err(Error::MapLen {
                got: d2t.len(),
                want: draft_rows,
            });
        }
        let to_target = d2t
            .iter()
            .enumerate()
            .map(|(d, &off)| {
                let value = d as i64 + off;
                u32::try_from(value).map_err(|_| Error::MapValue { index: d, value })
            })
            .collect::<Result<Vec<u32>, Error>>()?;
        DraftVocab::Reduced {
            to_target,
            n_target,
        }
    };
    let mask_id = u32::try_from(uint(dflash.unwrap_or(&cfg.root), "mask_token_id")?)
        .map_err(|_| Error::Unsupported("`mask_token_id` is no u32 id".into()))?;
    let markov_rank = match switch(&cfg, "markov_rank") {
        Some(v) => v
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| Error::Type {
                key: "markov_rank",
                got: kind(v),
                want: "an unsigned integer",
            })?,
        None => return Err(Error::Missing("markov_rank")),
    };
    let confidence = match switch(&cfg, "enable_confidence_head") {
        Some(v) => v.as_bool().ok_or_else(|| Error::Type {
            key: "enable_confidence_head",
            got: kind(v),
            want: "a boolean",
        })?,
        None => false,
    };
    let row0_drafts = match switch(&cfg, "sample_from_anchor") {
        Some(v) => v.as_bool().ok_or_else(|| Error::Type {
            key: "sample_from_anchor",
            got: kind(v),
            want: "a boolean",
        })?,
        None => true,
    };
    let shape = DsparkShape {
        block,
        row0_drafts,
        taps,
        hidden,
        vocab,
        mask_id,
        window: window_of(&cfg)?,
        markov_rank,
        confidence,
    };
    shape.check()?;
    Ok(shape)
}

fn ids(obj: &Value, key: &'static str) -> Result<Vec<usize>, Error> {
    let Value::Array(items) = field(obj, key)? else {
        return Err(Error::Type {
            key,
            got: kind(&obj[key]),
            want: "an array",
        });
    };
    items
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| Error::Type {
                    key,
                    got: kind(v),
                    want: "an unsigned integer",
                })
        })
        .collect()
}

/// The dense backbone's numbers `config_json` states. A rotation of less than the whole head, a
/// rotary type other than `default` and an activation other than `silu` are refused by name: the
/// card's dense layer holds no other.
pub fn backbone(config_json: &str) -> Result<Backbone, Error> {
    let cfg = Config::parse(config_json)?;
    let layout = cfg.layout()?;
    let bb = cfg.backbone()?;
    let head_dim = size(bb, "head_dim")?;
    if let Some(act) = bb.get("hidden_act").and_then(Value::as_str)
        && act != "silu"
    {
        return Err(Error::Unsupported(format!(
            "`hidden_act` is {act:?}; the dense layer's MLP is SiLU"
        )));
    }
    let rope = bb.get("rope_parameters").filter(|v| v.is_object());
    if let Some(ty) = rope
        .and_then(|r| r.get("rope_type"))
        .and_then(Value::as_str)
        && ty != "default"
    {
        return Err(Error::Unsupported(format!(
            "`rope_type` is {ty:?}; the dense layer rotates by the default type"
        )));
    }
    if let Some(dim) = rope.and_then(|r| r.get("rope_dim")).and_then(Value::as_u64)
        && dim != head_dim as u64
    {
        return Err(Error::Unsupported(format!(
            "`rope_dim` is {dim} of a head of {head_dim}; the dense layer rotates the whole head"
        )));
    }
    let theta = match rope.and_then(|r| r.get("rope_theta")) {
        Some(v) if !v.is_null() => v,
        _ => field(bb, "rope_theta")?,
    };
    let rope_theta = theta.as_f64().ok_or_else(|| Error::Type {
        key: "rope_theta",
        got: kind(theta),
        want: "a number",
    })?;
    let eps = field(bb, "rms_norm_eps")?
        .as_f64()
        .ok_or_else(|| Error::Type {
            key: "rms_norm_eps",
            got: kind(&bb["rms_norm_eps"]),
            want: "a number",
        })?;
    let window = window_of(&cfg)?;
    let block_causal = match layout {
        // Training masks a block's rows causally in the sliding layers unless
        // `sliding_window_non_causal` says otherwise (speculators
        // `models/dflash/attention.py`: `non_causal = sliding_window is None or
        // sliding_window_non_causal`); a full-attention layer is bidirectional.
        Layout::Speculators => {
            window.is_some() && !flag(&cfg.root, "sliding_window_non_causal", false)?
        }
        // vLLM's DFlash backbone: `dflash_config.causal` when given, else causal exactly in the
        // sliding layers (`qwen3_dflash.py`, `_dflash_layer_causal`).
        Layout::Dflash => match cfg.dflash().and_then(|d| d.get("causal")) {
            Some(v) if !v.is_null() => v.as_bool().ok_or_else(|| Error::Type {
                key: "causal",
                got: kind(v),
                want: "a boolean",
            })?,
            _ => window.is_some(),
        },
    };
    Ok(Backbone {
        layers: size(bb, "num_hidden_layers")?,
        hidden: size(bb, "hidden_size")?,
        heads: size(bb, "num_attention_heads")?,
        kv_heads: size(bb, "num_key_value_heads")?,
        head_dim,
        intermediate: size(bb, "intermediate_size")?,
        eps: eps as f32,
        rope_theta,
        window,
        block_causal,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const REDHAT: &str = include_str!("../tests/fixtures/dspark-redhat-qwen3.6-config.json");
    const MINICPM: &str = include_str!("../tests/fixtures/dspark-minicpm-v-4.7-config.json");

    /// An injective offset map over 32000 draft ids in a vocabulary of 248320: draft id `d` goes to
    /// `7 * d` (offsets `6 * d`), the shape of a real `d2t` (rising, with gaps).
    fn d2t() -> Vec<i64> {
        (0..32000).map(|d| 6 * d).collect()
    }

    #[test]
    fn the_speculators_layout_reads_the_redhat_qwen36_draft() {
        let map = d2t();
        let s = shape(REDHAT, Some(&map)).unwrap();
        assert_eq!(s.block, 8);
        // `sample_from_anchor: true`: row 0 drafts id 0, a block of 8 drafts 8 ids.
        assert!(s.row0_drafts);
        assert_eq!(s.ids_per_block(), 8);
        // Entering layers, as the file states them.
        assert_eq!(s.taps, [2, 10, 20, 30, 37]);
        // `target_hidden_size` is null: the target's width is the draft's own.
        assert_eq!(s.hidden, 2048);
        assert_eq!(s.mask_id, 248077);
        assert_eq!(s.window, Some(2048));
        assert_eq!(s.markov_rank, 256);
        assert!(s.confidence);
        let DraftVocab::Reduced {
            to_target,
            n_target,
        } = &s.vocab
        else {
            panic!("32000 of 248320 ids is a reduced head: {:?}", s.vocab);
        };
        assert_eq!((to_target.len(), *n_target), (32000, 248320));
        // `target_id = draft_id + d2t[draft_id]`.
        assert_eq!(
            (to_target[0], to_target[1], to_target[31999]),
            (0, 7, 223993)
        );
        assert_eq!(s.vocab.target(3), Ok(21));
        // A proposal against body35's 8-row verify holds 7 ids.
        assert_eq!(s.width(8), 7);
    }

    #[test]
    fn the_dflash_layout_reads_the_minicpm_draft_and_moves_its_taps_to_entering() {
        let s = shape(MINICPM, None).unwrap();
        assert_eq!(s.block, 15);
        // The file states no `sample_from_anchor`: DSpark's default.
        assert!(s.row0_drafts);
        // `target_layer_ids` [3, 11, 19, 27, 35] leave those layers: the rows entering the next.
        assert_eq!(s.taps, [4, 12, 20, 28, 36]);
        assert_eq!(s.hidden, 2048);
        assert_eq!(s.mask_id, 248076);
        // Full attention, no window; the head scores the target's own ids.
        assert_eq!(s.window, None);
        assert_eq!(s.vocab, DraftVocab::Full { n: 248144 });
        assert_eq!(s.markov_rank, 256);
        assert!(s.confidence);
        assert_eq!(s.width(8), 7);
    }

    #[test]
    fn a_file_without_sample_from_anchor_drafts_only_from_the_mask_rows() {
        let map = d2t();
        let anchor = REDHAT.replace(
            "\"sample_from_anchor\": true",
            "\"sample_from_anchor\": false",
        );
        let s = shape(&anchor, Some(&map)).unwrap();
        assert!(!s.row0_drafts);
        // Row 0 is the anchor only: 8 rows draft 7 ids.
        assert_eq!((s.ids_per_block(), s.rows(7)), (7, 8));
    }

    #[test]
    fn the_id_map_is_refused_by_name_when_absent_short_outside_or_repeating() {
        assert!(matches!(
            shape(REDHAT, None),
            Err(Error::NoMap {
                draft: 32000,
                target: 248320
            })
        ));
        let short: Vec<i64> = d2t()[..31999].to_vec();
        assert!(matches!(
            shape(REDHAT, Some(&short)),
            Err(Error::MapLen {
                got: 31999,
                want: 32000
            })
        ));
        let mut outside = d2t();
        outside[5] = 248320;
        assert!(matches!(
            shape(REDHAT, Some(&outside)),
            Err(Error::Shape(DsparkError::Map { index: 5, .. }))
        ));
        let mut negative = d2t();
        negative[5] = -6;
        assert!(matches!(
            shape(REDHAT, Some(&negative)),
            Err(Error::MapValue {
                index: 5,
                value: -1
            })
        ));
        let mut repeat = d2t();
        repeat[9] = repeat[8] - 1;
        assert!(matches!(
            shape(REDHAT, Some(&repeat)),
            Err(Error::Shape(DsparkError::Map { index: 9, .. }))
        ));
    }

    #[test]
    fn a_config_that_states_neither_layout_or_lacks_a_field_is_refused_by_name() {
        assert!(matches!(shape("{}", None), Err(Error::Layout)));
        assert!(matches!(shape("[1]", None), Err(Error::Json(_))));
        let no_mask = REDHAT.replace("\"mask_token_id\"", "\"mask_token_ids\"");
        assert!(matches!(
            shape(&no_mask, Some(&d2t())),
            Err(Error::Missing("mask_token_id"))
        ));
        // A mask id past the target's vocabulary is the shape's own refusal.
        let past = REDHAT.replace("248077", "248320");
        assert!(matches!(
            shape(&past, Some(&d2t())),
            Err(Error::Shape(DsparkError::Mask { id: 248320, .. }))
        ));
        let mixed = REDHAT.replacen("sliding_attention", "full_attention", 1);
        assert!(matches!(
            shape(&mixed, Some(&d2t())),
            Err(Error::MixedLayers)
        ));
    }

    #[test]
    fn the_backbone_numbers_and_the_block_mask_follow_each_file() {
        let r = backbone(REDHAT).unwrap();
        assert_eq!(
            (
                r.layers,
                r.hidden,
                r.heads,
                r.kv_heads,
                r.head_dim,
                r.intermediate
            ),
            (5, 2048, 16, 2, 256, 6144)
        );
        assert_eq!((r.rope_theta, r.eps, r.window), (1e7, 1e-6, Some(2048)));
        // Five sliding layers and `sliding_window_non_causal: false`: a block row sees the block's
        // earlier rows only.
        assert!(r.block_causal);
        let non_causal = REDHAT.replace(
            "\"sliding_window_non_causal\": false",
            "\"sliding_window_non_causal\": true",
        );
        assert!(!backbone(&non_causal).unwrap().block_causal);
        let m = backbone(MINICPM).unwrap();
        assert_eq!(
            (
                m.layers,
                m.hidden,
                m.heads,
                m.kv_heads,
                m.head_dim,
                m.intermediate
            ),
            (5, 2048, 8, 2, 256, 6144)
        );
        assert_eq!((m.rope_theta, m.window, m.block_causal), (1e7, None, false));
    }

    #[test]
    fn a_backbone_the_dense_layer_does_not_hold_is_refused_by_name() {
        let gelu = REDHAT.replace("\"hidden_act\": \"silu\"", "\"hidden_act\": \"gelu\"");
        assert!(matches!(backbone(&gelu), Err(Error::Unsupported(e)) if e.contains("gelu")));
        let yarn = REDHAT.replace("\"rope_type\": \"default\"", "\"rope_type\": \"yarn\"");
        assert!(matches!(backbone(&yarn), Err(Error::Unsupported(e)) if e.contains("yarn")));
        let partial = MINICPM.replace("\"rope_dim\": 256", "\"rope_dim\": 64");
        assert!(matches!(backbone(&partial), Err(Error::Unsupported(e)) if e.contains("rope_dim")));
    }
}
