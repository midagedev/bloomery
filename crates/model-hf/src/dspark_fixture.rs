//! The synthetic DSpark draft fixture: a Hugging Face directory (`config.json` and
//! `model.safetensors`) a family's gate runs its draft clauses on, written from a target's facts
//! alone — no trained draft, no real file. The loader the gates read it through is `dspark_hf`
//! (`crates/gpu-gates/src/bin/shared/dspark_hf.rs`), whose geometry parser (`hf::dspark`) is the
//! pure half this module's tests round-trip through.
//!
//! What a draft states about its target it borrows from [`Target`]: the target's hidden width
//! (one tap row's and the head's rows'), its vocabulary (the draft's ids) and its layer count
//! (every tap below it). What is the draft's own is [`Draft`]: the dense backbone's widths, the
//! taps, the block, the mask id and the head's rows. [`Vocab`] picks the head's shape — the
//! speculator's reduced vocabulary with its own `lm_head`, `embed_tokens` and `d2t` map
//! ([`Vocab::Reduced`]), the target's own ids borrowing its `output.weight` and
//! `token_embd.weight` rows ([`Vocab::Full`]), and the refused-by-shape twin of the reduced one
//! whose stated vocabulary falls one id short ([`Vocab::Mismatched`]): a seat's fit check
//! refuses it before any weight is read, which is the cross-target refusal a gate proves on
//! fixtures.
//!
//! `config.json` carries every key the loader reads and no other, in the speculators layout
//! (`aux_hidden_state_layer_ids`, `block_size`, `target_hidden_size`, `draft_vocab_size`,
//! `mask_token_id`, `markov_rank`, `enable_confidence_head`, `sample_from_anchor`,
//! `sliding_window_non_causal` at the top, the backbone's numbers in
//! `transformer_layer_config`), as the speculators format's published drafts state them.
//! `model.safetensors` carries the tensors the loader opens: bf16 matrices and per-layer norm
//! gains, f32 gains and confidence head, and the reduced vocabulary's `d2t` as i64 offsets.
//!
//! The weights are drawn like the GGUF fixtures' ([`model::fixture`]): each tensor's stream is cut
//! into chunks keyed by (seed, tensor name, chunk index), every matrix element a standard normal
//! of deviation `1/√k` over its input width `k` (a projection of a unit-RMS row has unit RMS),
//! the norm gains 1, the confidence bias 0. The id map is the real files' shape — rising, with
//! gaps: draft id `d` is target id `d · stride`, the stride the stated vocabulary holds per
//! draft row.
//!
//! A family's own drafts — their taps, widths and mask id, facts of its target — and the set it
//! writes beside its fixture belong to the family's own `arch` module.
//!
//! [`rename_target`] writes the companion the "fits by shapes, not by names" clause reads: the
//! target's own shards with `general.name` and `general.basename` replaced, every tensor byte
//! for byte, so a draft loads beside a target whose name is not the one it was written for.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use gguf::write::{Layout, TensorDecl, Writer};
use gguf::{Split, Value};
use safetensors::Dtype;

use model::fixture::dspark::RENAMED_DIR;
use model::fixture::{CHUNK_TARGET, FixtureError, Rng, chunk_rng, io_err, meta};

/// The Markov rank every DSpark head the card runs holds
/// (`bloomery_gpu::dspark::head::MARKOV_RANK`); the writer takes no other.
pub const MARKOV_RANK: usize = 256;

/// The values per head the dense layer's kernels serve; the writer takes no other.
const HEAD_DIM: usize = 256;

/// What a draft states about its target, read from the target's own header ([`Target::of`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Target {
    /// The target's hidden width: one tap row's and the head's rows'.
    pub hidden: usize,
    /// The target's vocabulary: the draft's ids.
    pub n_vocab: usize,
    /// The target's layers: every tap is below this.
    pub n_layers: usize,
}

impl Target {
    /// The target facts `split` (a fixture, its first shard) states: its `embedding_length`,
    /// `vocab_size` (the tokenizer's row count when the header carries one, as every fixture
    /// does) and `block_count`. A header naming none of them is refused by name.
    pub fn of(split: &Split) -> Result<Target, FixtureError> {
        let g = split.shard(0).ok_or_else(|| FixtureError::Mismatch {
            what: "the DSpark target".into(),
            detail: "the split holds no shard".into(),
        })?;
        let arch = g.architecture().unwrap_or_default();
        let uint = |key: String| -> Result<usize, FixtureError> {
            let v = g.value(&key).ok_or_else(|| meta(&key, "is absent"))?;
            let n = match v {
                Value::U8(v) => u64::from(*v),
                Value::U16(v) => u64::from(*v),
                Value::U32(v) => u64::from(*v),
                Value::U64(v) => *v,
                Value::I8(v) if *v >= 0 => u64::try_from(*v).unwrap_or(u64::MAX),
                Value::I16(v) if *v >= 0 => u64::try_from(*v).unwrap_or(u64::MAX),
                Value::I32(v) if *v >= 0 => u64::try_from(*v).unwrap_or(u64::MAX),
                Value::I64(v) if *v >= 0 => u64::try_from(*v).unwrap_or(u64::MAX),
                other => return Err(meta(&key, format!("{other:?} is not an unsigned integer"))),
            };
            usize::try_from(n).map_err(|_| meta(&key, format!("{n} is past usize")))
        };
        let n_vocab = match g.value("tokenizer.ggml.tokens") {
            Some(Value::Array(tokens)) => tokens.len(),
            _ => uint(format!("{arch}.vocab_size"))?,
        };
        Ok(Target {
            hidden: uint(format!("{arch}.embedding_length"))?,
            n_vocab,
            n_layers: uint(format!("{arch}.block_count"))?,
        })
    }
}

/// The draft's own geometry: everything a synthetic draft picks for itself, held to the loader's
/// rules at [`Draft::check`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Draft {
    /// The dense backbone's decoder layers.
    pub layers: usize,
    /// The draft's own hidden width; its tables' rows are the target's [`Target::hidden`] wide.
    pub hidden: usize,
    /// Query heads, of `HEAD_DIM` values each.
    pub heads: usize,
    /// Key/value heads.
    pub kv_heads: usize,
    /// The MLP's inner width.
    pub inter: usize,
    /// The target layers whose entering rows the draft reads, strictly rising.
    pub taps: Vec<usize>,
    /// Tokens one block pass drafts.
    pub block: usize,
    /// The token the block's rows past the first embed.
    pub mask_id: u32,
    /// How far back the draft's attention reads; `None` is the whole context.
    pub window: Option<usize>,
    /// Whether the file carries a confidence head.
    pub confidence: bool,
    /// The reduced vocabulary the draft's head scores (a full one scores the target's own).
    pub rows: usize,
}

impl Draft {
    /// The draft's own rules, the loader's: whole 32-value Q8_0 rows, whole heads of
    /// `HEAD_DIM`, a query group the ring pass serves, taps strictly rising and below the
    /// target's layers, a block that drafts, a mask inside the vocabulary, a window of
    /// positions, and rows the stated vocabulary holds a rising stride of. A full-vocabulary
    /// draft's own width is the target's, the borrowed table's rows' width.
    pub fn check(&self, target: &Target) -> Result<(), FixtureError> {
        let bad = |what: &str, detail: String| FixtureError::Mismatch {
            what: format!("the DSpark draft's {what}"),
            detail,
        };
        for (what, v) in [("hidden", self.hidden), ("inter", self.inter)] {
            if v == 0 || v % 32 != 0 {
                return Err(bad(
                    what,
                    format!("{v}: the Q8_0 rows run in blocks of 32 values"),
                ));
            }
        }
        if self.kv_heads == 0 || !self.heads.is_multiple_of(self.kv_heads) {
            return Err(bad(
                "heads",
                format!(
                    "{} query over {} key heads: not whole groups",
                    self.heads, self.kv_heads
                ),
            ));
        }
        let group = self.heads / self.kv_heads;
        if group != 8 && !group.is_multiple_of(4) {
            return Err(bad(
                "heads",
                format!(
                    "a group of {group} query heads a key head: the ring pass serves 8 or a \
                     multiple of 4"
                ),
            ));
        }
        for (what, rows) in [
            ("query", self.heads * HEAD_DIM),
            ("key", self.kv_heads * HEAD_DIM),
        ] {
            if rows % 32 != 0 {
                return Err(bad(what, format!("{rows} rows: not blocks of 32 values")));
            }
        }
        if self.taps.is_empty() || self.taps.windows(2).any(|w| w[0] >= w[1]) {
            return Err(bad(
                "taps",
                format!("{:?}: at least one, strictly rising", self.taps),
            ));
        }
        if let Some(&tap) = self.taps.iter().find(|&&t| t >= target.n_layers) {
            return Err(bad(
                "taps",
                format!("tap {tap} beside a target of {} layers", target.n_layers),
            ));
        }
        if self.block < 2 {
            return Err(bad(
                "block",
                format!("{}: a block drafts at least one id", self.block),
            ));
        }
        if usize::try_from(self.mask_id).map_or(true, |m| m >= target.n_vocab) {
            return Err(bad(
                "mask id",
                format!("{} beside a target of {} ids", self.mask_id, target.n_vocab),
            ));
        }
        if self.window == Some(0) {
            return Err(bad("window", "0 positions".into()));
        }
        Ok(())
    }
}

/// Which head a synthetic draft carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vocab {
    /// The draft's own `lm_head`, `embed_tokens` and `d2t` map over [`Draft::rows`] of the
    /// target's ids: the speculator's shape.
    Reduced,
    /// The target's own ids: no `lm_head`, no `embed_tokens`, no map — the draft borrows the
    /// target's `output.weight` in place and its `token_embd.weight` rows, so its own hidden
    /// width is the target's.
    Full,
    /// [`Vocab::Reduced`] with a stated vocabulary one id short of the target's: the shape a
    /// seat's fit check refuses by name before any weight is read.
    Mismatched,
}

impl Vocab {
    /// The vocabulary the draft's `config.json` states.
    pub fn n_target(self, target: &Target) -> usize {
        match self {
            Self::Reduced | Self::Full => target.n_vocab,
            Self::Mismatched => target.n_vocab - 1,
        }
    }

    /// The head's rows: the draft's own for a reduced shape, the target's for a full one.
    pub fn rows(self, draft: &Draft, target: &Target) -> usize {
        match self {
            Self::Full => target.n_vocab,
            Self::Reduced | Self::Mismatched => draft.rows,
        }
    }
}

/// One tensor of the draft's file.
struct Tensor {
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
    /// The matrix rule's input width, `None` for a fixed fill.
    k: Option<usize>,
    /// The id map's stride, set on `d2t` alone.
    stride: Option<usize>,
}

/// The tensors `vocab`'s draft carries, in file order: the loader's own list, no other.
fn tensors(target: &Target, draft: &Draft, vocab: Vocab) -> Result<Vec<Tensor>, FixtureError> {
    let n_target = vocab.n_target(target);
    let rows = vocab.rows(draft, target);
    let map_stride = match vocab {
        Vocab::Full => None,
        _ => Some(stride(n_target, rows)?),
    };
    let (h, taps_width) = (draft.hidden, draft.taps.len() * target.hidden);
    let matrix = |name: String, rows: usize, k: usize| Tensor {
        name,
        dtype: Dtype::Bf16,
        shape: vec![rows, k],
        k: Some(k),
        stride: None,
    };
    let gain = |name: String, dtype: Dtype, n: usize| Tensor {
        name,
        dtype,
        shape: vec![n],
        k: None,
        stride: None,
    };
    let mut t = Vec::new();
    for l in 0..draft.layers {
        t.push(gain(
            format!("layers.{l}.input_layernorm.weight"),
            Dtype::Bf16,
            h,
        ));
        t.push(gain(
            format!("layers.{l}.post_attention_layernorm.weight"),
            Dtype::Bf16,
            h,
        ));
        t.push(gain(
            format!("layers.{l}.self_attn.q_norm.weight"),
            Dtype::Bf16,
            HEAD_DIM,
        ));
        t.push(gain(
            format!("layers.{l}.self_attn.k_norm.weight"),
            Dtype::Bf16,
            HEAD_DIM,
        ));
        t.push(matrix(
            format!("layers.{l}.self_attn.q_proj.weight"),
            draft.heads * HEAD_DIM,
            h,
        ));
        t.push(matrix(
            format!("layers.{l}.self_attn.k_proj.weight"),
            draft.kv_heads * HEAD_DIM,
            h,
        ));
        t.push(matrix(
            format!("layers.{l}.self_attn.v_proj.weight"),
            draft.kv_heads * HEAD_DIM,
            h,
        ));
        t.push(matrix(
            format!("layers.{l}.self_attn.o_proj.weight"),
            h,
            draft.heads * HEAD_DIM,
        ));
        t.push(matrix(
            format!("layers.{l}.mlp.gate_proj.weight"),
            draft.inter,
            h,
        ));
        t.push(matrix(
            format!("layers.{l}.mlp.up_proj.weight"),
            draft.inter,
            h,
        ));
        t.push(matrix(
            format!("layers.{l}.mlp.down_proj.weight"),
            h,
            draft.inter,
        ));
    }
    if vocab != Vocab::Full {
        t.push(matrix("lm_head.weight".into(), rows, h));
        t.push(matrix("embed_tokens.weight".into(), n_target, h));
    }
    t.push(matrix("fc.weight".into(), h, taps_width));
    t.push(gain("hidden_norm.weight".into(), Dtype::F32, h));
    t.push(gain("norm.weight".into(), Dtype::F32, h));
    t.push(matrix(
        "markov_head.markov_w1.weight".into(),
        n_target,
        MARKOV_RANK,
    ));
    t.push(matrix(
        "markov_head.markov_w2.weight".into(),
        rows,
        MARKOV_RANK,
    ));
    if vocab != Vocab::Full {
        t.push(Tensor {
            name: "d2t".into(),
            dtype: Dtype::I64,
            shape: vec![rows],
            k: None,
            stride: map_stride,
        });
    }
    if draft.confidence {
        t.push(Tensor {
            name: "confidence_head.proj.weight".into(),
            dtype: Dtype::F32,
            shape: vec![1, h + MARKOV_RANK],
            k: Some(h + MARKOV_RANK),
            stride: None,
        });
        t.push(gain("confidence_head.proj.bias".into(), Dtype::F32, 1));
    }
    Ok(t)
}

/// The draft's `config.json`, every key the loader reads and no other.
fn config(target: &Target, draft: &Draft, vocab: Vocab) -> String {
    let n_target = vocab.n_target(target);
    let windowed = draft.window.is_some();
    let mut bb = serde_json::Map::new();
    bb.insert("vocab_size".into(), (n_target as u64).into());
    bb.insert("hidden_size".into(), (draft.hidden as u64).into());
    bb.insert("intermediate_size".into(), (draft.inter as u64).into());
    bb.insert("num_hidden_layers".into(), (draft.layers as u64).into());
    bb.insert("num_attention_heads".into(), (draft.heads as u64).into());
    bb.insert("num_key_value_heads".into(), (draft.kv_heads as u64).into());
    bb.insert("head_dim".into(), (HEAD_DIM as u64).into());
    bb.insert("hidden_act".into(), "silu".into());
    bb.insert("rms_norm_eps".into(), 1e-6.into());
    let mut rope = serde_json::Map::new();
    rope.insert("rope_theta".into(), 10_000_000.0.into());
    rope.insert("rope_type".into(), "default".into());
    bb.insert("rope_parameters".into(), serde_json::Value::Object(rope));
    if let Some(window) = draft.window {
        bb.insert(
            "layer_types".into(),
            serde_json::Value::Array(
                (0..draft.layers)
                    .map(|_| "sliding_attention".into())
                    .collect(),
            ),
        );
        bb.insert("sliding_window".into(), (window as u64).into());
    }
    let mut root = serde_json::Map::new();
    root.insert(
        "aux_hidden_state_layer_ids".into(),
        serde_json::Value::Array(draft.taps.iter().map(|&t| (t as u64).into()).collect()),
    );
    root.insert("block_size".into(), (draft.block as u64).into());
    root.insert(
        "target_hidden_size".into(),
        if draft.hidden == target.hidden {
            serde_json::Value::Null
        } else {
            (target.hidden as u64).into()
        },
    );
    root.insert(
        "draft_vocab_size".into(),
        if vocab == Vocab::Full {
            serde_json::Value::Null
        } else {
            (draft.rows as u64).into()
        },
    );
    root.insert("mask_token_id".into(), (u64::from(draft.mask_id)).into());
    root.insert("markov_rank".into(), (MARKOV_RANK as u64).into());
    root.insert("enable_confidence_head".into(), draft.confidence.into());
    root.insert("sample_from_anchor".into(), true.into());
    if windowed {
        root.insert("sliding_window_non_causal".into(), false.into());
    }
    root.insert(
        "transformer_layer_config".into(),
        serde_json::Value::Object(bb),
    );
    format!(
        "{}\n",
        serde_json::to_string_pretty(&serde_json::Value::Object(root))
            .expect("a config of plain values renders")
    )
}

/// `x` rounded to bf16, nearest even.
fn bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let rounded = bits + 0x7fff + ((bits >> 16) & 1);
    (rounded >> 16) as u16
}

/// A standard normal from the stream's next two uniforms (Box–Muller); the uniform is never 0,
/// so the logarithm is finite.
fn normal(r: &mut Rng) -> f64 {
    let u = |r: &mut Rng| ((r.step() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    let (a, b) = (u(r), u(r));
    (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
}

/// One tensor's bytes: `t`'s rule over its values, in chunks keyed by (seed, name, chunk), so
/// any chunk size writes the same file. The gains and the confidence bias are fixed fills
/// (`k` `None`); every matrix element is a normal of deviation `1/√k`; the map's offsets are
/// `d · stride − d`.
fn bytes(seed: u64, t: &Tensor) -> Vec<u8> {
    let n: usize = t.shape.iter().product();
    let value = t.dtype.size();
    let mut out = Vec::with_capacity(n * value);
    let chunk = (CHUNK_TARGET / value).max(1);
    let mut at = 0;
    let mut part = 0;
    while at < n {
        let end = (at + chunk).min(n);
        let mut r = chunk_rng(seed, &t.name, part);
        for i in at..end {
            let v = match (t.k, t.dtype) {
                // A projection of a unit-RMS row has unit RMS.
                (Some(k), _) => normal(&mut r) / (k as f64).sqrt(),
                (None, Dtype::I64) => {
                    let stride = t.stride.expect("the writer sets the map's stride");
                    let d = i as u64;
                    ((d * stride as u64).wrapping_sub(d)) as f64
                }
                (None, _) => 1.0,
            };
            match t.dtype {
                Dtype::Bf16 => out.extend(bf16_bits(v as f32).to_le_bytes()),
                Dtype::F32 => out.extend((v as f32).to_le_bytes()),
                Dtype::I64 => out.extend((v as i64).to_le_bytes()),
                _ => unreachable!("the tensor list names bf16, f32 and i64 only"),
            }
        }
        at = end;
        part += 1;
    }
    out
}

/// The map's stride: the stated vocabulary's `rows`-ths, at least 2 (the real maps rise with
/// gaps) and the last row inside the stated vocabulary.
fn stride(n_target: usize, rows: usize) -> Result<usize, FixtureError> {
    let stride = n_target / rows.max(1);
    if stride < 2 || rows.saturating_sub(1) * stride >= n_target {
        return Err(FixtureError::Mismatch {
            what: "the DSpark draft's id map".into(),
            detail: format!("{rows} rows of {n_target} ids leave no rising stride with gaps"),
        });
    }
    Ok(stride)
}

/// Write the `vocab` draft of `draft` beside `target` into `out` (which must not exist):
/// `config.json` and `model.safetensors`, into `<out>.tmp.<pid>` first, each synced, renamed
/// once complete. The bytes are a function of `seed`, `draft` and `target` alone.
pub fn write_dir(
    out: &Path,
    target: &Target,
    draft: &Draft,
    vocab: Vocab,
    seed: u64,
) -> Result<(), FixtureError> {
    draft.check(target)?;
    if vocab == Vocab::Full && draft.hidden != target.hidden {
        return Err(FixtureError::Mismatch {
            what: "the DSpark draft's hidden width".into(),
            detail: format!(
                "{}: a full-vocabulary draft borrows the target's table, whose rows are {} wide",
                draft.hidden, target.hidden
            ),
        });
    }
    if vocab == Vocab::Mismatched && target.n_vocab == 0 {
        return Err(FixtureError::Mismatch {
            what: "the DSpark draft's stated vocabulary".into(),
            detail: "the target holds no id to fall short of".into(),
        });
    }
    let list = tensors(target, draft, vocab)?;
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent).map_err(|e| io_err(parent, "create the directory", e))?;
    }
    let tmp = tmp_of(out);
    std::fs::create_dir(&tmp).map_err(|e| match e.kind() {
        io::ErrorKind::AlreadyExists => FixtureError::Exists { path: tmp.clone() },
        _ => io_err(&tmp, "create the directory", e),
    })?;
    let written = write_files(&tmp, target, draft, vocab, &list, seed);
    match written {
        Ok(()) => {
            File::open(out.parent().unwrap_or(Path::new(".")))
                .and_then(|d| d.sync_all())
                .map_err(|e| io_err(out, "fsync", e))?;
            model::fileio::rename_noreplace(&tmp, out).map_err(|e| match e.kind() {
                io::ErrorKind::AlreadyExists => FixtureError::Exists {
                    path: out.to_path_buf(),
                },
                _ => io_err(out, "rename", e),
            })
        }
        Err(error) => abandon(error, &tmp),
    }
}

/// The two files of one draft, into `tmp`.
fn write_files(
    tmp: &Path,
    target: &Target,
    draft: &Draft,
    vocab: Vocab,
    list: &[Tensor],
    seed: u64,
) -> Result<(), FixtureError> {
    let cfg = File::create(tmp.join("config.json"))
        .map_err(|e| io_err(&tmp.join("config.json"), "create", e))?;
    let mut cfg = BufWriter::new(cfg);
    cfg.write_all(config(target, draft, vocab).as_bytes())
        .map_err(|e| io_err(&tmp.join("config.json"), "write", e))?;
    cfg.flush()
        .map_err(|e| io_err(&tmp.join("config.json"), "flush", e))?;
    let decls: Vec<_> = list
        .iter()
        .map(|t| {
            let raw = bytes(seed, t);
            (t.name.as_str(), t.dtype.name(), t.shape.as_slice(), raw)
        })
        .collect();
    let refs: Vec<(&str, &str, &[usize], &[u8])> = decls
        .iter()
        .map(|(name, dtype, shape, raw)| (*name, *dtype, *shape, raw.as_slice()))
        .collect();
    let f = File::create(tmp.join("model.safetensors"))
        .map_err(|e| io_err(&tmp.join("model.safetensors"), "create", e))?;
    let mut f = BufWriter::new(f);
    f.write_all(&safetensors::write(&refs))
        .map_err(|e| io_err(&tmp.join("model.safetensors"), "write", e))?;
    f.flush()
        .map_err(|e| io_err(&tmp.join("model.safetensors"), "flush", e))?;
    f.get_ref()
        .sync_all()
        .map_err(|e| io_err(&tmp.join("model.safetensors"), "fsync", e))?;
    Ok(())
}

/// `<path>.tmp.<pid>`, the path a writer stages in.
fn tmp_of(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(format!(".tmp.{}", std::process::id()));
    PathBuf::from(s)
}

/// A failed write's cleanup: the error, with a failed cleanup of `tmp` (a file or a directory)
/// named beside it.
fn abandon<T>(error: FixtureError, tmp: &Path) -> Result<T, FixtureError> {
    let cleanup = std::fs::remove_dir_all(tmp).or_else(|e| {
        if e.kind() == io::ErrorKind::NotFound {
            Ok(())
        } else {
            std::fs::remove_file(tmp)
        }
    });
    match cleanup {
        Ok(()) => Err(error),
        Err(cleanup) => Err(FixtureError::Abandoned {
            error: Box::new(error),
            tmp: tmp.to_path_buf(),
            cleanup,
        }),
    }
}

/// Bytes a tensor-copy part holds.
const COPY_PART: usize = 16 << 20;

/// Write the renamed target copy beside `first`'s split: every shard of the set under
/// `<first's directory>/renamed/`, its `general.name` and `general.basename` replaced by `name`
/// and `basename`, every metadata key else and every tensor byte for byte. An output that
/// exists is refused, shard by shard.
pub fn rename_target(
    first: &Path,
    name: &str,
    basename: &str,
) -> Result<Vec<PathBuf>, FixtureError> {
    let split = Split::open(first).map_err(FixtureError::Load)?;
    let out_dir = first.parent().unwrap_or(Path::new(".")).join(RENAMED_DIR);
    let mut written = Vec::new();
    for i in 0..split.shard_count() {
        let (Some(g), Some(path)) = (split.shard(i), split.shard_path(i)) else {
            return Err(FixtureError::Mismatch {
                what: "the renamed copy".into(),
                detail: format!("the split holds no shard {i}"),
            });
        };
        let file = path.file_name().ok_or_else(|| FixtureError::Mismatch {
            what: "the renamed copy".into(),
            detail: format!("{} names no file", path.display()),
        })?;
        let out = out_dir.join(file);
        std::fs::create_dir_all(&out_dir)
            .map_err(|e| io_err(&out_dir, "create the directory", e))?;
        let tmp = tmp_of(&out);
        match copy_shard(g, &tmp, name, basename) {
            Ok(()) => model::fileio::rename_noreplace(&tmp, &out).map_err(|e| match e.kind() {
                io::ErrorKind::AlreadyExists => FixtureError::Exists { path: out.clone() },
                _ => io_err(&out, "rename", e),
            })?,
            Err(error) => return abandon(error, &tmp),
        }
        written.push(out);
    }
    Ok(written)
}

/// One shard of [`rename_target`]: its metadata with the two name keys replaced, its tensors
/// copied byte for byte in declaration order.
fn copy_shard(g: &gguf::Gguf, out: &Path, name: &str, basename: &str) -> Result<(), FixtureError> {
    let kvs: Vec<(String, Value)> = g
        .iter_kv()
        .map(|(k, v)| {
            let v = match k {
                "general.name" => Value::String(name.to_string()),
                "general.basename" => Value::String(basename.to_string()),
                _ => v.clone(),
            };
            (k.to_string(), v)
        })
        .collect();
    let decls: Vec<TensorDecl> = g
        .iter_tensors()
        .map(|t| TensorDecl {
            name: t.name.clone(),
            dims: t.dims.clone(),
            type_id: t.ty.file_id(),
            nbytes: t.nbytes,
        })
        .collect();
    let layout = Layout::new(&kvs, decls).map_err(|e| FixtureError::Write {
        path: out.to_path_buf(),
        source: e,
    })?;
    let file = File::create(out).map_err(|e| io_err(out, "create", e))?;
    let werr = |source| FixtureError::Write {
        path: out.to_path_buf(),
        source,
    };
    let mut w = Writer::new(BufWriter::with_capacity(1 << 20, file), layout).map_err(werr)?;
    for t in g.iter_tensors() {
        let data = g.data(t).map_err(FixtureError::Load)?;
        w.tensor_parts(&t.name, data.chunks(COPY_PART))
            .map_err(werr)?;
    }
    let file = w
        .finish()
        .map_err(werr)?
        .into_inner()
        .map_err(|e| io_err(out, "flush", e.into_error()))?;
    file.sync_all().map_err(|e| io_err(out, "fsync", e))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Draft, MARKOV_RANK, Target, Vocab, stride, write_dir};
    use model::fixture::FixtureError;
    use std::path::{Path, PathBuf};

    /// A target small enough to write twice in a test, with a fixture target's facts in
    /// kind: four layers, a hidden width of whole Q8_0 blocks, a vocabulary a stride fits.
    fn tiny_target() -> Target {
        Target {
            hidden: 64,
            n_vocab: 8192,
            n_layers: 4,
        }
    }

    /// A tiny draft of the speculator's shape: its block, window and confidence head.
    fn tiny_draft(target: &Target) -> Draft {
        Draft {
            layers: 2,
            hidden: target.hidden,
            heads: 8,
            kv_heads: 2,
            inter: 128,
            taps: vec![1, 2, 3],
            block: 8,
            mask_id: 4095,
            window: Some(64),
            confidence: true,
            rows: 1024,
        }
    }

    /// A fresh directory under the temp root, removed first.
    fn dir(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("bloomery-dspark-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        p
    }

    /// The tensors `vocab`'s draft carries, by name and shape, for the set equality: the
    /// writer's file holds these and no other.
    fn want_tensors(target: &Target, draft: &Draft, vocab: Vocab) -> Vec<(String, Vec<usize>)> {
        let (rows, n_target) = (vocab.rows(draft, target), vocab.n_target(target));
        let (h, q, kv) = (draft.hidden, draft.heads * 256, draft.kv_heads * 256);
        let mut t: Vec<(String, Vec<usize>)> = (0..draft.layers)
            .flat_map(|l| {
                [
                    (format!("layers.{l}.input_layernorm.weight"), vec![h]),
                    (
                        format!("layers.{l}.post_attention_layernorm.weight"),
                        vec![h],
                    ),
                    (format!("layers.{l}.self_attn.q_norm.weight"), vec![256]),
                    (format!("layers.{l}.self_attn.k_norm.weight"), vec![256]),
                    (format!("layers.{l}.self_attn.q_proj.weight"), vec![q, h]),
                    (format!("layers.{l}.self_attn.k_proj.weight"), vec![kv, h]),
                    (format!("layers.{l}.self_attn.v_proj.weight"), vec![kv, h]),
                    (format!("layers.{l}.self_attn.o_proj.weight"), vec![h, q]),
                    (
                        format!("layers.{l}.mlp.gate_proj.weight"),
                        vec![draft.inter, h],
                    ),
                    (
                        format!("layers.{l}.mlp.up_proj.weight"),
                        vec![draft.inter, h],
                    ),
                    (
                        format!("layers.{l}.mlp.down_proj.weight"),
                        vec![h, draft.inter],
                    ),
                ]
                .into_iter()
            })
            .collect();
        if vocab != Vocab::Full {
            t.push(("lm_head.weight".into(), vec![rows, h]));
            t.push(("embed_tokens.weight".into(), vec![n_target, h]));
        }
        t.push(("fc.weight".into(), vec![h, 3 * target.hidden]));
        t.push(("hidden_norm.weight".into(), vec![h]));
        t.push(("norm.weight".into(), vec![h]));
        t.push((
            "markov_head.markov_w1.weight".into(),
            vec![n_target, MARKOV_RANK],
        ));
        t.push((
            "markov_head.markov_w2.weight".into(),
            vec![rows, MARKOV_RANK],
        ));
        if vocab != Vocab::Full {
            t.push(("d2t".into(), vec![rows]));
        }
        if draft.confidence {
            t.push((
                "confidence_head.proj.weight".into(),
                vec![1, h + MARKOV_RANK],
            ));
            t.push(("confidence_head.proj.bias".into(), vec![1]));
        }
        t
    }

    /// One written draft: its file holds exactly the tensors of its shape, each at its shape,
    /// the map's offsets are the stride's, and the whole directory is the same bytes written
    /// twice. The config's own round trip — the loader's parse of it — is the bin's test, beside
    /// the crate that holds the parser.
    fn check_written(path: &Path, target: &Target, draft: &Draft, vocab: Vocab) {
        let st = safetensors::Safetensors::open(&path.join("model.safetensors")).expect("opens");
        let mut got: Vec<(String, Vec<usize>)> = st
            .names()
            .map(|n| {
                let e = st.entry(n).expect("a listed name has an entry");
                (n.to_string(), e.shape.clone())
            })
            .collect();
        got.sort();
        let mut want = want_tensors(target, draft, vocab);
        want.sort();
        assert_eq!(got, want, "{vocab:?}");
        if vocab != Vocab::Full {
            let rows = vocab.rows(draft, target);
            let stride = stride(vocab.n_target(target), rows).expect("the map's stride");
            let d2t = st.i64s("d2t").expect("the map");
            for (d, &off) in d2t.iter().enumerate() {
                assert_eq!(
                    off,
                    (d as u64 * stride as u64) as i64 - d as i64,
                    "draft id {d}"
                );
            }
        }
        // The same bytes twice: the file is a function of the seed and the shape alone.
        let again = path.with_extension("again");
        write_dir(&again, target, draft, vocab, 7).expect("the second write");
        for name in ["config.json", "model.safetensors"] {
            let (a, b) = (
                std::fs::read(path.join(name)).expect("the first"),
                std::fs::read(again.join(name)).expect("the second"),
            );
            assert_eq!(a, b, "{name}");
        }
        let _ = std::fs::remove_dir_all(&again);
    }

    #[test]
    fn the_reduced_draft_holds_the_loaders_tensors() {
        let (target, draft) = (tiny_target(), tiny_draft(&tiny_target()));
        let path = dir("reduced");
        write_dir(&path, &target, &draft, Vocab::Reduced, 7).expect("the write");
        check_written(&path, &target, &draft, Vocab::Reduced);
        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn the_full_draft_holds_the_loaders_tensors() {
        let (target, draft) = (tiny_target(), tiny_draft(&tiny_target()));
        let path = dir("full");
        write_dir(&path, &target, &draft, Vocab::Full, 7).expect("the write");
        check_written(&path, &target, &draft, Vocab::Full);
        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn the_mismatched_draft_states_one_id_short_of_the_target() {
        let (target, draft) = (tiny_target(), tiny_draft(&tiny_target()));
        let path = dir("mismatched");
        write_dir(&path, &target, &draft, Vocab::Mismatched, 7).expect("the write");
        check_written(&path, &target, &draft, Vocab::Mismatched);
        let _ = std::fs::remove_dir_all(&path);
    }

    /// The write refuses a directory that exists, and the rules refuse what the loader would:
    /// a tap past the target's layers, a width between Q8_0 blocks, a query group the ring pass
    /// does not serve, a mask past the vocabulary, a full draft of another width than the
    /// target's.
    #[test]
    fn the_writer_refuses_what_the_loader_refuses() {
        let (target, draft) = (tiny_target(), tiny_draft(&tiny_target()));
        let path = dir("refuse");
        write_dir(&path, &target, &draft, Vocab::Reduced, 7).expect("the write");
        let again = write_dir(&path, &target, &draft, Vocab::Reduced, 7);
        assert!(
            matches!(again, Err(FixtureError::Exists { .. })),
            "{again:?}"
        );
        let _ = std::fs::remove_dir_all(&path);

        let bad = |d: Draft| {
            d.check(&target).expect_err("refused");
        };
        let mut tap = tiny_draft(&tiny_target());
        tap.taps = vec![1, 4];
        bad(tap);
        let mut width = tiny_draft(&tiny_target());
        width.hidden = 48;
        bad(width);
        let mut group = tiny_draft(&tiny_target());
        group.heads = 6;
        bad(group);
        let mut mask = tiny_draft(&tiny_target());
        mask.mask_id = 8192;
        bad(mask);
        let mut window = tiny_draft(&tiny_target());
        window.window = Some(0);
        bad(window);
        let mut block = tiny_draft(&tiny_target());
        block.block = 1;
        bad(block);
        let mut own = tiny_draft(&tiny_target());
        own.hidden = 96;
        own.heads = 4;
        assert!(write_dir(&dir("own-width"), &target, &own, Vocab::Full, 7).is_err());
        let _ = std::fs::remove_dir_all(dir("own-width"));
    }
}
