//! The loader of a DSpark draft kept as a Hugging Face directory (`config.json` and `model.safetensors`),
//! for the binaries that run one beside a target: Qwen3.6's RedHat speculator today.
//!
//! [`open`] reads the geometry through `hf::dspark`, the weights through `hf::safetensors` (a mapping:
//! the 1.9 GB file is never read into memory at once), and builds what the card-side owner takes:
//! [`DenseLayers`] (its matrices Q8_0, quantized from the file's bf16 on the host), [`FcWeights`], the
//! head ([`HeadWeights`] with its own Q8_0 `lm_head`) and the host embedding table. The Markov tables
//! travel as the file's bf16, two values to a word. Nothing of the bf16 is kept on the card.
//!
//! [`card_bytes`] is the same load's card size read from the file's header alone, for the placement
//! plan: a seat pushes it as the draft's reserve ([`RESERVE`]) on the target's stage card, which the
//! planner then refuses by that name when the card cannot hold it.

use std::path::{Path, PathBuf};

use bloomery_gpu::dspark::dense::{Bf16Source, DenseDims, DenseLayers, DenseSpec};
use bloomery_gpu::dspark::head::HeadWeights;
use bloomery_gpu::dspark::{EmbedSource, FcWeights, HeadSource, HostEmbed, Q8Rows};
use bloomery_gpu::{DeviceTensor, Gpu, GpuError};
use bloomery_gpu_gates::GateError;
use cuda_core::DeviceBuffer;
use hf::dspark as config;
use hf::safetensors::Safetensors;
use runtime::dspark::{DraftVocab, DsparkShape};

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

/// Open the draft in `dir` on `gpu`, for contexts of up to `ctx` positions.
pub fn open(gpu: &Gpu, dir: &Path, ctx: usize) -> Result<Parts, GateError> {
    let file = HfFile::open(dir)?;
    let shape = shape(dir, &file)?;
    let (spec, dims) = dense_spec(dir, &file, ctx)?;
    let hidden = dims.hidden;
    let taps = shape.taps.len();
    let layers = DenseLayers::load(gpu, &spec, &file)?;

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

    // The head: the final norm, the draft's own projection, the Markov tables and the id map.
    let rows = shape.vocab.rows();
    let n_target = shape.vocab.n_target();
    if !matches!(shape.vocab, DraftVocab::Reduced { .. }) {
        return Err(format!(
            "{WHAT}: a draft over the target's own {n_target} ids borrows the target's head, which \
             the seat holds; this loader opens drafts with their own lm_head"
        )
        .into());
    }
    expect_shape(&file, "lm_head.weight", &[rows, hidden])?;
    let (lm_bits, _) = file.bf16("lm_head.weight")?;
    let own = Q8Rows::from_bf16(gpu, lm_bits, rows, hidden, &|r| r)?;
    let (head_norm, _) = file.f32s("norm.weight")?;
    let rank = shape.markov_rank;
    let w1 = pair_words(&file, "markov_head.markov_w1.weight", n_target, rank)?;
    let w2 = pair_words(&file, "markov_head.markov_w2.weight", rows, rank)?;
    let stream = gpu.stream();
    let to_target = match &shape.vocab {
        DraftVocab::Reduced { to_target, .. } => Some(DeviceBuffer::from_host(stream, to_target)?),
        DraftVocab::Full { .. } => None,
    };
    let conf = if shape.confidence {
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
        source: HeadSource::Own(own),
        markov_w1: DeviceTensor::upload(stream, &w1, n_target, rank / 2)?,
        markov_w2: DeviceTensor::upload(stream, &w2, rows, rank / 2)?,
        to_target,
        conf,
    };

    // The draft's own embedding table stays on the host: the owner gathers the rows a pass needs.
    expect_shape(&file, "embed_tokens.weight", &[n_target, hidden])?;
    let (embed_bits, _) = file.bf16("embed_tokens.weight")?;
    let embed = EmbedSource::OwnHost(HostEmbed {
        rows: n_target,
        cols: hidden,
        bf16: embed_bits.to_vec(),
    });
    Ok(Parts {
        shape,
        layers,
        fc,
        head,
        embed,
    })
}

/// The card bytes [`open`] puts on `dir`'s draft at `ctx` positions, from the file's header and config
/// alone: the dense layers ([`DenseDims::card_bytes`]), fc, the head's own projection and norm, the
/// Markov tables, the id map and the confidence head. The embedding table is the host's.
pub fn card_bytes(dir: &Path, ctx: usize) -> Result<u64, GateError> {
    let file = HfFile::open(dir)?;
    let shape = shape(dir, &file)?;
    let (spec, dims) = dense_spec(dir, &file, ctx)?;
    let q8 = |rows: usize, k: usize| (rows * (k + k / 16)) as u64;
    let rows = shape.vocab.rows();
    let n_target = shape.vocab.n_target();
    let (h, rank) = (dims.hidden, shape.markov_rank);
    let fc = q8(h, shape.taps.len() * shape.hidden);
    let head = q8(rows, h);
    let markov = 4 * ((n_target + rows) * (rank / 2)) as u64;
    let map = 4 * rows as u64;
    let conf = if shape.confidence {
        4 * (h + rank + 1) as u64
    } else {
        0
    };
    let norms = 4 * (2 * h) as u64;
    Ok(dims.card_bytes(&spec) + fc + head + markov + map + conf + norms)
}

/// The directory the RedHat speculator is read from on the box.
pub fn default_dir() -> PathBuf {
    PathBuf::from("/models/Qwen3.6-35B-A3B/dspark-redhat")
}
