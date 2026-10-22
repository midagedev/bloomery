//! The loader of a DSpark draft kept as a Hugging Face directory (`config.json` and `model.safetensors`),
//! for the binaries that run one beside a target: Qwen3.6's RedHat speculator today.
//!
//! [`open`] reads the geometry through `hf::dspark`, the weights through `safetensors` (a mapping:
//! the 1.9 GB file is never read into memory at once), and builds what the card-side owner takes:
//! [`DenseLayers`] (its matrices Q8_0, quantized from the file's bf16 on the host), [`FcWeights`], the
//! head ([`HeadWeights`] with its own Q8_0 `lm_head`) and the host embedding table. The Markov tables
//! travel as the file's bf16, two values to a word. Nothing of the bf16 is kept on the card.
//!
//! [`card_bytes`] is the same load's card size read from the file's header alone, for the placement
//! plan: a seat pushes it as the draft's reserve ([`RESERVE`]) on the target's stage card, which the
//! planner then refuses by that name when the card cannot hold it.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use bloomery_gpu::dspark::dense::{Bf16Source, DenseDims, DenseLayers, DenseSpec};
use bloomery_gpu::dspark::head::{HeadTail, HeadWeights, MarkovTail};
use bloomery_gpu::dspark::{
    EmbedSource, FcWeights, HeadSource, HostEmbed, InPlace, Q8Rows, TargetHead,
};
use bloomery_gpu::{DeviceTensor, Gpu, GpuError};
use bloomery_gpu_gates::GateError;
use cuda_core::{CudaContext, DeviceBuffer};
use gguf::Split;
use hf::dspark as config;
use model::arch::names;
use model::embed::EmbedTable;
use runtime::dspark::{DraftVocab, DsparkShape, HeadKind};
use safetensors::Safetensors;

/// The name the draft's card bytes carry as a reserve of the target's plan.
pub const RESERVE: &str = "dspark draft";

const WHAT: &str = "dspark_hf";

/// Everything a draft directory builds: the arguments of `DsparkBody::new`, in its order.
pub struct Parts {
    pub shape: DsparkShape,
    pub layers: DenseLayers,
    pub fc: FcWeights,
    pub head: HeadWeights,
    pub embed: EmbedSource,
}

/// The target a draft is opened beside: the facts the draft's own shape is held to — its ids,
/// its layers and its hidden width. A draft over the target's own ids borrows the target's
/// head and embedding rows beside these facts ([`Borrowed`]); a reduced one borrows nothing.
#[derive(Clone, Copy)]
pub struct Target {
    /// The target's vocabulary: the draft's ids.
    pub n_vocab: usize,
    /// The target's layers: every tap is below this.
    pub n_layers: usize,
    /// The target's hidden width: one tap row's width and the head's rows'.
    pub hidden: usize,
}

/// What a full-vocabulary draft borrows of its target: the head in place and the embedding
/// table's rows. A draft over the target's own ids carries neither an `lm_head` nor an
/// `embed_tokens` of its own; a reduced one carries both and borrows nothing.
pub struct Borrowed<'a> {
    /// The target's file, for its `token_embd.weight` rows.
    pub split: Arc<Split>,
    /// The target's resident Q6_K `output.weight`.
    pub head: &'a DeviceTensor<u32>,
    /// The CUDA context the target's buffers live in: the draft's `Gpu` must be on it.
    pub ctx: &'a Arc<CudaContext>,
}

/// A draft's safetensors file as the layers' weight source.
pub struct HfFile {
    st: Safetensors,
}

impl HfFile {
    /// Map `dir/model.safetensors`.
    pub fn open(dir: &Path) -> Result<HfFile, GateError> {
        let path = dir.join("model.safetensors");
        Ok(HfFile {
            st: Safetensors::open(&path).map_err(|e| format!("{}: {e}", path.display()))?,
        })
    }

    /// The file's header and tensors.
    pub fn st(&self) -> &Safetensors {
        &self.st
    }

    /// Tensor `name` widened to f32.
    pub fn f32s(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>), GateError> {
        let t = self.st.tensor(name).map_err(|e| format!("{WHAT}: {e}"))?;
        Ok((t.data, t.shape))
    }
}

impl Bf16Source for HfFile {
    fn bf16(&self, name: &str) -> Result<(&[u16], &[usize]), GpuError> {
        self.st.bf16(name).map_err(|e| GpuError::Shape {
            what: WHAT,
            detail: e.to_string(),
        })
    }
}

/// `dir/config.json`'s text.
fn config_text(dir: &Path) -> Result<String, GateError> {
    let path = dir.join("config.json");
    std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()).into())
}

/// The shape `dir` states: its config and, for a reduced head, the file's `d2t` offsets.
pub fn shape(dir: &Path, file: &HfFile) -> Result<DsparkShape, GateError> {
    let d2t = if file.st.entry("d2t").is_some() {
        Some(file.st.i64s("d2t").map_err(|e| format!("{WHAT}: {e}"))?)
    } else {
        None
    };
    Ok(config::shape(&config_text(dir)?, d2t.as_deref())?)
}

/// Whether `shape` serves a target of `n_vocab` ids, `n_layers` layers and hidden width `hidden`:
/// its ids, its mask and every tap fit the target. Refused by name, before any weight is read.
pub fn fits(
    shape: &DsparkShape,
    n_vocab: usize,
    n_layers: usize,
    hidden: usize,
) -> Result<(), GateError> {
    let n_target = shape.vocab.n_target();
    if n_target != n_vocab {
        return Err(format!(
            "{WHAT}: the draft is over the target's {n_target} ids; the target holds {n_vocab}"
        )
        .into());
    }
    if shape.mask_id as usize >= n_vocab {
        return Err(format!(
            "{WHAT}: mask id {} beside a target of {n_vocab} ids",
            shape.mask_id
        )
        .into());
    }
    if let Some(&tap) = shape.taps.iter().find(|&&t| t >= n_layers) {
        return Err(format!("{WHAT}: tap {tap} beside a target of {n_layers} layers").into());
    }
    if shape.hidden != hidden {
        return Err(format!(
            "{WHAT}: tap rows of {} values; the target's hidden width is {hidden}",
            shape.hidden
        )
        .into());
    }
    Ok(())
}

/// The dense layers' spec: the config's numbers, checked against the tensors' shapes.
pub fn dense_spec(
    dir: &Path,
    file: &HfFile,
    ctx: usize,
) -> Result<(DenseSpec, DenseDims), GateError> {
    let bb = config::backbone(&config_text(dir)?)?;
    let dims = DenseDims::read(file)?;
    let want = DenseDims {
        hidden: bb.hidden,
        n_q: bb.heads,
        n_kv: bb.kv_heads,
        head_dim: bb.head_dim,
        inter: bb.intermediate,
    };
    if dims != want {
        return Err(format!("{WHAT}: the tensors are {dims:?}, the config says {want:?}").into());
    }
    Ok((
        DenseSpec {
            layers: bb.layers,
            eps: bb.eps,
            rope_theta: bb.rope_theta as f32,
            window: bb.window,
            block_causal: bb.block_causal,
            ctx,
        },
        dims,
    ))
}

fn expect_shape(file: &HfFile, name: &str, want: &[usize]) -> Result<(), GateError> {
    let e = file
        .st
        .entry(name)
        .ok_or_else(|| format!("{WHAT}: the file holds no {name}"))?;
    if e.shape != want {
        return Err(format!("{WHAT}: {name} is {:?}, wanted {want:?}", e.shape).into());
    }
    Ok(())
}

/// The file's bf16 tensor `name` as little-endian u32 words, two values to a word, `rows` rows of
/// `cols` values (`cols` even): the layout of the Markov tables.
fn pair_words(file: &HfFile, name: &str, rows: usize, cols: usize) -> Result<Vec<u32>, GateError> {
    expect_shape(file, name, &[rows, cols])?;
    let raw = file.st.raw(name).map_err(|e| format!("{WHAT}: {e}"))?;
    Ok(raw
        .bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect())
}

/// An f32 vector as a one-row tensor.
fn row_f32(gpu: &Gpu, v: &[f32]) -> Result<DeviceTensor<f32>, GateError> {
    Ok(DeviceTensor::upload(gpu.stream(), v, 1, v.len())?)
}

/// Open the draft in `dir` on `gpu`, for contexts of up to `ctx` positions. A draft over the
/// target's own ids is refused by name: it borrows the target's head and embedding
/// ([`open_for_target`]).
pub fn open(gpu: &Gpu, dir: &Path, ctx: usize) -> Result<Parts, GateError> {
    open_with(gpu, dir, ctx, None)
}

/// The shape the draft in `dir` states, read from its config and its file's header alone: the
/// seat asks it before it opens the target, to know whether the draft borrows the target's head
/// ([`shape`]'s [`DraftVocab`]).
#[allow(
    dead_code,
    reason = "the reader is shared by the draft gate and the serve seat; each opens the part it reads"
)]
pub fn shape_of(dir: &Path) -> Result<DsparkShape, GateError> {
    let file = HfFile::open(dir)?;
    shape(dir, &file)
}

/// Open the draft in `dir` beside `target`, for contexts of up to `ctx` positions: a full-vocabulary
/// draft borrows the target's Q6_K head in place and its `token_embd.weight` rows, a reduced one
/// loads its own `lm_head` and embedding. The draft's ids, mask and taps are held to the target
/// before any weight is read.
#[allow(
    dead_code,
    reason = "the reader is shared by the draft gate and the serve seat; each opens the part it reads"
)]
pub fn open_for_target(
    gpu: &Gpu,
    dir: &Path,
    ctx: usize,
    target: &Target,
    borrowed: Option<&Borrowed<'_>>,
) -> Result<Parts, GateError> {
    open_with(gpu, dir, ctx, Some((target, borrowed)))
}

fn open_with(
    gpu: &Gpu,
    dir: &Path,
    ctx: usize,
    target: Option<(&Target, Option<&Borrowed<'_>>)>,
) -> Result<Parts, GateError> {
    let file = HfFile::open(dir)?;
    let shape = shape(dir, &file)?;
    if let Some((t, _)) = target {
        fits(&shape, t.n_vocab, t.n_layers, t.hidden)?;
    }
    let (spec, dims) = dense_spec(dir, &file, ctx)?;
    let hidden = dims.hidden;
    let taps = shape.taps.len();
    let rows = shape.vocab.rows();
    let n_target = shape.vocab.n_target();

    let (layers, source, embed) = match shape.vocab {
        DraftVocab::Full { .. } => {
            let Some((_, Some(t))) = target else {
                return Err(format!(
                    "{WHAT}: a draft over the target's own {n_target} ids borrows the target's \
                     head, which the seat holds; this loader opens drafts with their own lm_head"
                )
                .into());
            };
            for own in ["lm_head.weight", "embed_tokens.weight"] {
                if file.st.entry(own).is_some() {
                    return Err(format!(
                        "{WHAT}: a full-vocabulary draft carries {own}; a draft over the target's \
                         own ids borrows both its head and its embedding"
                    )
                    .into());
                }
            }
            let table = EmbedTable::new(Arc::clone(&t.split), &names::token_embd(), hidden)
                .map_err(|e| format!("{WHAT}: {e}"))?;
            let layers = DenseLayers::load_target_embed(gpu, &spec, &file, table, shape.mask_id)?;
            if t.ctx.cu_ctx() != gpu.context().cu_ctx() {
                return Err(format!(
                    "{WHAT}: the target's head is on another CUDA context; a borrowed head runs \
                     only on the target's own device"
                )
                .into());
            }
            // SAFETY: the target's model owns `head` and outlives the draft (the seat holds both in
            // one session), and the check above holds the draft's `Gpu` to the head's context, so
            // the address a captured graph bakes in stays alive and in place.
            let source = HeadSource::Target(TargetHead::Q6KInPlace(unsafe {
                InPlace::new(t.head, t.ctx)
            }));
            (layers, source, EmbedSource::Target)
        }
        DraftVocab::Reduced { .. } => {
            let layers = DenseLayers::load(gpu, &spec, &file)?;
            expect_shape(&file, "lm_head.weight", &[rows, hidden])?;
            let (lm_bits, _) = file.bf16("lm_head.weight")?;
            let own = Q8Rows::from_bf16(gpu, lm_bits, rows, hidden, &|r| r)?;
            // The draft's own embedding table stays on the host: the owner gathers the rows a
            // pass needs.
            expect_shape(&file, "embed_tokens.weight", &[n_target, hidden])?;
            let (embed_bits, _) = file.bf16("embed_tokens.weight")?;
            let embed = EmbedSource::OwnHost(HostEmbed {
                rows: n_target,
                cols: hidden,
                bf16: embed_bits.to_vec(),
            });
            (layers, HeadSource::Own(own), embed)
        }
    };

    // fc over the taps, then the hidden norm.
    expect_shape(&file, "fc.weight", &[hidden, taps * shape.hidden])?;
    let (fc_bits, _) = file.bf16("fc.weight")?;
    let fc = Q8Rows::from_bf16(gpu, fc_bits, hidden, taps * shape.hidden, &|r| r)?;
    let (norm, _) = file.f32s("hidden_norm.weight")?;
    let fc = FcWeights {
        fc,
        norm: row_f32(gpu, &norm)?,
        eps: spec.eps,
    };

    // The head: the final norm, the projection (borrowed or the draft's own), the Markov tables
    // and, for a reduced vocabulary, the id map.
    let (head_norm, _) = file.f32s("norm.weight")?;
    let HeadKind::Markov { rank, confidence } = shape.head else {
        return Err(format!(
            "{WHAT}: a {:?} head: this loader reads Markov drafts",
            shape.head
        )
        .into());
    };
    let w1 = pair_words(&file, "markov_head.markov_w1.weight", n_target, rank)?;
    let w2 = pair_words(&file, "markov_head.markov_w2.weight", rows, rank)?;
    let stream = gpu.stream();
    let to_target = match &shape.vocab {
        DraftVocab::Reduced { to_target, .. } => Some(DeviceBuffer::from_host(stream, to_target)?),
        DraftVocab::Full { .. } => None,
    };
    let conf = if confidence {
        expect_shape(&file, "confidence_head.proj.weight", &[1, hidden + rank])?;
        expect_shape(&file, "confidence_head.proj.bias", &[1])?;
        let (w, _) = file.f32s("confidence_head.proj.weight")?;
        let (b, _) = file.f32s("confidence_head.proj.bias")?;
        Some((DeviceTensor::upload(stream, &w, 1, hidden + rank)?, b[0]))
    } else {
        None
    };
    let head = HeadWeights {
        norm: row_f32(gpu, &head_norm)?,
        eps: spec.eps,
        source,
        tail: HeadTail::Markov(MarkovTail {
            markov_w1: DeviceTensor::upload(stream, &w1, n_target, rank / 2)?,
            markov_w2: DeviceTensor::upload(stream, &w2, rows, rank / 2)?,
            to_target,
            conf,
        }),
    };
    Ok(Parts {
        shape,
        layers,
        fc,
        head,
        embed,
    })
}

/// The card bytes [`open`] puts on `dir`'s reduced-vocabulary draft at `ctx` positions, from the
/// file's header and config alone. A full-vocabulary draft's bytes need its target
/// ([`card_bytes_for_target`]).
pub fn card_bytes(dir: &Path, ctx: usize) -> Result<u64, GateError> {
    let file = HfFile::open(dir)?;
    let shape = shape(dir, &file)?;
    if matches!(shape.vocab, DraftVocab::Full { .. }) {
        return Err(format!(
            "{WHAT}: a full-vocabulary draft's card bytes need the target: its head and embedding \
             are the target's"
        )
        .into());
    }
    let (spec, dims) = dense_spec(dir, &file, ctx)?;
    card_bytes_of(&shape, &dims, &spec)
}

/// The card bytes [`open_for_target`] puts on `dir`'s draft at `ctx` positions, from the file's
/// header and config alone: the dense layers (with the target embedding's staging rows for a full
/// vocabulary), fc, the head's own projection and norm, the Markov tables, the id map (a reduced
/// vocabulary) and the confidence head. A borrowed head and the target's table are the target's.
#[allow(
    dead_code,
    reason = "the reader is shared by the draft gate and the serve seat; each opens the part it reads"
)]
pub fn card_bytes_for_target(dir: &Path, ctx: usize) -> Result<u64, GateError> {
    let file = HfFile::open(dir)?;
    let shape = shape(dir, &file)?;
    let (spec, dims) = dense_spec(dir, &file, ctx)?;
    card_bytes_of(&shape, &dims, &spec)
}

fn card_bytes_of(
    shape: &DsparkShape,
    dims: &DenseDims,
    spec: &DenseSpec,
) -> Result<u64, GateError> {
    let q8 = |rows: usize, k: usize| (rows * (k + k / 16)) as u64;
    let rows = shape.vocab.rows();
    let n_target = shape.vocab.n_target();
    let HeadKind::Markov { rank, confidence } = shape.head else {
        return Err(format!(
            "{WHAT}: a {:?} head: this loader reads Markov drafts",
            shape.head
        )
        .into());
    };
    let h = dims.hidden;
    let fc = q8(h, shape.taps.len() * shape.hidden);
    let markov = 4 * ((n_target + rows) * (rank / 2)) as u64;
    let conf = if confidence {
        4 * (h + rank + 1) as u64
    } else {
        0
    };
    let norms = 4 * (2 * h) as u64;
    let (layers, head, map) = match shape.vocab {
        DraftVocab::Full { .. } => (dims.card_bytes_target_embed(spec), 0, 0),
        DraftVocab::Reduced { .. } => (dims.card_bytes(spec), q8(rows, h), 4 * rows as u64),
    };
    Ok(layers + fc + head + markov + map + conf + norms)
}

/// `BLOOMERY_DSPARK_MODEL` as a full-vocabulary draft's HF directory: `config.json` and
/// `model.safetensors` beside each other. Refused by name when unset or not a directory.
#[allow(
    dead_code,
    reason = "the reader is shared by the draft gate and the serve seat; each opens the part it reads"
)]
pub fn dir_from_env() -> Result<PathBuf, GateError> {
    let dir = match std::env::var_os("BLOOMERY_DSPARK_MODEL") {
        Some(p) if !p.is_empty() => PathBuf::from(p),
        _ => {
            return Err(
                "BLOOMERY_DRAFT=dspark: BLOOMERY_DSPARK_MODEL unset — export the HF DSpark \
                 draft's directory (`config.json` and `model.safetensors`), as the dspark gates do"
                    .into(),
            );
        }
    };
    for name in ["config.json", "model.safetensors"] {
        if !dir.join(name).is_file() {
            return Err(format!(
                "BLOOMERY_DRAFT=dspark: {} holds no {name}; BLOOMERY_DSPARK_MODEL is an HF \
                 draft's directory",
                dir.display()
            )
            .into());
        }
    }
    Ok(dir)
}

/// The directory the RedHat speculator is read from on the box.
pub fn default_dir() -> PathBuf {
    PathBuf::from("/models/Qwen3.6-35B-A3B/dspark-redhat")
}
