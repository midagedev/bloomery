//! The MTP draft layer of a session: the sidecar's `mtp.*` tensors opened as
//! the layer after the file's last, and one draft step over it.
//!
//! The sidecar is a folder of MLX's 4-bit export of the checkpoint's
//! next-token layer (`mlx-community/Qwen3.8-27B-MTP-4bit`): `config.json` and
//! one `model.safetensors` (or `mtp/model.safetensors`), tensors named with no
//! prefix. MLX's own folders strip them, so the main folder never holds them.
//! A draft step is
//!
//! ```text
//! x      = fc · [ pre_fc_norm_embedding(embed(t)) | pre_fc_norm_hidden(h) ]
//! x      = layer(x)                      attention + dense FFN, its own K/V
//! token  = argmax(output · norm(x))      the target's head, the draft's norm
//! ```
//!
//! `h` is the target's hidden row of the position before `t`, and the join
//! takes it as the row before the target's output norm or after it
//! ([`HiddenRow`]): the two readings of the checkpoint's text differ, and the
//! draft's agreement with the target on real text says which is the layer's.
//!
//! The layer is the file's own attention layer with a dense FFN and a plain
//! residual, over a different set of tensors: its layout is the first such
//! layer's of the file ([`Seed`]) with the sidecar's names, so its buffers
//! (the P1 dispatch, the gains, the K/V planes) are the ones a layer of the
//! file has, made by the same loader, and its ops are the session's own
//! ([`MetalOps`] over [`walk::layer`]). While a draft step encodes, the
//! arena's layer list and head norm are the draft's (a swap of two handles
//! the encode undoes whatever it returns): the walk's readers and hand-ons
//! then see a one-layer chain whose head norm is the draft's, as they would
//! for a file of that layer alone. The step's scratch (the residual, the P1
//! output, the logits) is the arena's; a draft step runs between settled
//! steps and a token running ahead is dropped first, as a pass drops it.
//!
//! The draft's K/V rows are its own: a step at position `p` appends row `p`
//! and attends rows `0..=p`, so a caller that wants the layer to see the
//! prompt feeds every position in order. The step's record slot is its
//! position's: the pick and the fault word are read before the next step.
//!
//! Not here: the policy (when to draft, how many), filling the draft's rows
//! from a prompt walk without the head, and keeping the hidden row of a chain
//! step. The hidden row is the host's to set or tap between settled steps.

use core::sync::atomic::Ordering;
use core::time::Duration;
use std::path::{Path, PathBuf};
use std::time::Instant;

use bloomery_device::{Buffer, Elems, Stream};
use chain::arch::qwen3moe::ops::{ChainOps, LayerCtx};
use chain::arch::qwen3moe::walk;
use gguf::TensorInfo;
use gguf::quant::GgmlType;
use model::part::Part;
use safetensors::{Dtype, Safetensors};

use super::super::arena::{
    self, Arena, LayerBufs, PartSource, load_layer, load_site, load_values, slot_of,
};
use super::super::check::{AttnLayout, FfnLayout, LayerLayout, Layout, MixerLayout, Site, Values};
use super::super::ops::{Feed, MetalOps};
use super::{STAMP_WAIT, Session, SessionError, refuse};
use crate::buffer::MetalBuffer;
use crate::kernels::{EmbedParams, HiddenRow, Kernels, MatvecParams, Mode, MtpJoinParams, Quant};
use crate::residency::Alloc;
use crate::stream::GpuInterval;
use chain::arch::qwen3moe::plan::LayerPlan;

const WHAT: &str = "qwen35moe mtp";

/// The sidecar's text: the checkpoint class its config names.
const MODEL_TYPE: &str = "qwen3_5_mtp";

/// MLX's affine groups: 4-bit codes, 64 values a scale and bias.
const BITS: u64 = 4;
const GROUP: usize = 64;

/// The record slot's layer of an embed fault: layer 0 of the draft.
const LAYER: u32 = 0;

fn mtp_refuse(detail: impl Into<String>) -> SessionError {
    SessionError::Shape {
        what: WHAT,
        detail: detail.into(),
    }
}

/// What a session keeps of its file's layout to shape the draft layer after:
/// the first attention layer with a dense FFN and a plain residual, and its
/// place in the plan.
pub(super) struct Seed {
    /// The layer's index in the file: the plan the draft's walk reads.
    layer: usize,
    attn: AttnLayout,
    ffn: FfnLayout,
    hidden: usize,
    vocab: usize,
    eps: f32,
}

impl Seed {
    /// The seed of a checked file, or `None` for a file with no such layer.
    pub(super) fn of(layout: &Layout) -> Option<Seed> {
        layout.layers.iter().enumerate().find_map(|(i, l)| {
            let MixerLayout::Attention(a) = &l.mixer else {
                return None;
            };
            let dense = l.ffn.gate_inp.is_none() && l.ffn.shared.is_none();
            (dense && l.post.is_none()).then(|| Seed {
                layer: i,
                attn: a.clone(),
                ffn: l.ffn.clone(),
                hidden: layout.hidden,
                vocab: layout.vocab,
                eps: layout.eps,
            })
        })
    }

    /// The draft layer's layout: the seed's, with the sidecar's names.
    fn layout(&self) -> Result<LayerLayout, SessionError> {
        let mut a = self.attn.clone();
        if a.gate || a.bias.is_some() || a.p1.sites.len() != 3 || a.window.is_some() {
            return Err(mtp_refuse(
                "the file's attention layer has a projection of its own for the output gate, a \
                 bias or a window; the draft layer is the gated-query attention with none",
            ));
        }
        a.p1.norm.name = "layers.0.input_layernorm.weight".to_owned();
        for (site, stem) in a.p1.sites.iter_mut().zip(["q_proj", "k_proj", "v_proj"]) {
            site.name = format!("layers.0.self_attn.{stem}");
        }
        a.out.name = "layers.0.self_attn.o_proj".to_owned();
        match (a.q_norm.as_mut(), a.k_norm.as_mut()) {
            (Some(q), Some(k)) => {
                q.name = "layers.0.self_attn.q_norm.weight".to_owned();
                k.name = "layers.0.self_attn.k_norm.weight".to_owned();
            }
            _ => {
                return Err(mtp_refuse(
                    "the file's attention layer applies no q/k head norm; the draft layer does",
                ));
            }
        }
        let mut f = self.ffn.clone();
        f.norm.name = "layers.0.post_attention_layernorm.weight".to_owned();
        f.gate.name = "layers.0.mlp.gate_proj".to_owned();
        f.up.name = "layers.0.mlp.up_proj".to_owned();
        f.down.name = "layers.0.mlp.down_proj".to_owned();
        Ok(LayerLayout {
            mixer: MixerLayout::Attention(a),
            ffn: f,
            post: None,
        })
    }

    /// The linears and vectors the draft layer holds, by sidecar name, with
    /// the shape each must have: what [`Sidecar::check`] holds the file to.
    fn expect(&self, layer: &LayerLayout) -> Expect {
        let MixerLayout::Attention(a) = &layer.mixer else {
            unreachable!("the draft layer is an attention layer");
        };
        let mut linears: Vec<(String, usize, usize, Quant)> = Vec::new();
        for s in
            a.p1.sites
                .iter()
                .chain([&a.out, &layer.ffn.gate, &layer.ffn.up, &layer.ffn.down])
        {
            linears.push((s.name.clone(), s.rows, s.cols, s.quant));
        }
        let mut vectors: Vec<(String, usize)> = Vec::new();
        for v in [&a.p1.norm, &layer.ffn.norm]
            .into_iter()
            .chain(a.q_norm.iter())
            .chain(a.k_norm.iter())
        {
            vectors.push((v.name.clone(), v.len));
        }
        Expect {
            hidden: self.hidden,
            vocab: self.vocab,
            linears,
            vectors,
        }
    }
}

/// What the sidecar must hold for the file it joins.
struct Expect {
    hidden: usize,
    vocab: usize,
    /// Each linear's name (its `.weight`, `.scales` and `.biases` follow), rows,
    /// columns and the weight type the layer's launches read it in.
    linears: Vec<(String, usize, usize, Quant)>,
    /// Each vector's name and length.
    vectors: Vec<(String, usize)>,
}

/// The tensors of the sidecar's own beside the layer's: the fusion
/// projection, and the three norms the join and the head read.
const FC: &str = "fc";
const NORM: &str = "norm.weight";
const PRE_EMBEDDING: &str = "pre_fc_norm_embedding.weight";
const PRE_HIDDEN: &str = "pre_fc_norm_hidden.weight";

/// An opened sidecar: its config's facts and its tensors, mapped.
pub(super) struct Sidecar {
    st: Safetensors,
    path: PathBuf,
    hidden: usize,
    vocab: usize,
}

impl Sidecar {
    /// The sidecar folder (or the `model.safetensors` file) at `path`: its
    /// config read and its tensors mapped. A folder that is not a qwen3_5_mtp
    /// export of 4-bit affine groups of 64, or a config that omits what a
    /// refusal needs, is refused by name.
    pub(super) fn open(path: &Path) -> Result<Sidecar, SessionError> {
        let (config, file) = if path.is_dir() {
            let direct = path.join("model.safetensors");
            let nested = path.join("mtp").join("model.safetensors");
            let file = if direct.is_file() {
                direct
            } else if nested.is_file() {
                nested
            } else {
                return Err(mtp_refuse(format!(
                    "{}: no model.safetensors or mtp/model.safetensors in the sidecar folder",
                    path.display()
                )));
            };
            (path.join("config.json"), file)
        } else {
            let dir = path.parent().unwrap_or_else(|| Path::new("."));
            (dir.join("config.json"), path.to_path_buf())
        };
        let text = std::fs::read_to_string(&config)
            .map_err(|e| mtp_refuse(format!("{}: {e}", config.display())))?;
        let json = safetensors::parse_unique("config.json", &text)
            .map_err(|e| mtp_refuse(format!("{}: {e}", config.display())))?;
        let model_type = json.get("model_type").and_then(|v| v.as_str());
        if model_type != Some(MODEL_TYPE) {
            return Err(mtp_refuse(format!(
                "{}: model_type {model_type:?}, not {MODEL_TYPE:?}: this is no MTP sidecar of the \
                 Qwen3.5 family",
                config.display()
            )));
        }
        let quant = json.get("quantization");
        let q = |k: &str| quant.and_then(|q| q.get(k));
        if q("bits").and_then(|v| v.as_u64()) != Some(BITS)
            || q("group_size").and_then(|v| v.as_u64()) != Some(GROUP as u64)
            || q("mode").and_then(|v| v.as_str()) != Some("affine")
        {
            return Err(mtp_refuse(format!(
                "{}: quantization {quant:?}; the draft reads MLX's affine 4-bit groups of {GROUP}",
                config.display()
            )));
        }
        let text_config = json
            .get("text_config")
            .ok_or_else(|| mtp_refuse(format!("{}: no text_config", config.display())))?;
        let count = |key: &str| -> Result<usize, SessionError> {
            text_config
                .get(key)
                .and_then(|v| v.as_u64())
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| {
                    mtp_refuse(format!("{}: text_config has no {key}", config.display()))
                })
        };
        let layers = count("mtp_num_hidden_layers")?;
        if layers != 1 {
            return Err(mtp_refuse(format!(
                "{}: {layers} draft layers; the draft this loads is one layer",
                config.display()
            )));
        }
        if text_config.get("num_experts").is_some() {
            return Err(mtp_refuse(format!(
                "{}: a routed draft layer (num_experts in text_config); the draft this loads \
                 is a dense FFN",
                config.display()
            )));
        }
        let (hidden, vocab) = (count("hidden_size")?, count("vocab_size")?);
        let st =
            Safetensors::open(&file).map_err(|e| mtp_refuse(format!("{}: {e}", file.display())))?;
        Ok(Sidecar {
            st,
            path: file,
            hidden,
            vocab,
        })
    }

    /// The shape of the linear `base`: its rows and columns, read from the
    /// packed weight's shape.
    fn shape(&self, base: &str) -> Result<(usize, usize), SessionError> {
        let w = self.st.entry(&format!("{base}.weight")).ok_or_else(|| {
            mtp_refuse(format!("{}: no tensor {base}.weight", self.path.display()))
        })?;
        match w.shape.as_slice() {
            [rows, words] if w.dtype == Dtype::U32 => Ok((*rows, words * 8)),
            other => Err(mtp_refuse(format!(
                "{base}.weight is {} of shape {other:?}, not a packed U32 matrix",
                w.dtype.name()
            ))),
        }
    }

    /// Whether the sidecar is the file `expect` describes, every difference
    /// named: its hidden size and vocabulary, the exact set of its tensors
    /// (a gated delta layer's, or a router's, is another layer kind), and the
    /// shape and type of each.
    fn check(&self, expect: &Expect) -> Result<(), SessionError> {
        if self.hidden != expect.hidden {
            return Err(mtp_refuse(format!(
                "{}: hidden size {}, the model's is {}: a sidecar of another model",
                self.path.display(),
                self.hidden,
                expect.hidden
            )));
        }
        if self.vocab != expect.vocab {
            return Err(mtp_refuse(format!(
                "{}: vocabulary {}, the model's is {}: a sidecar of another model",
                self.path.display(),
                self.vocab,
                expect.vocab
            )));
        }
        let mut held: Vec<String> = Vec::new();
        for (base, ..) in &expect.linears {
            for suffix in ["weight", "scales", "biases"] {
                held.push(format!("{base}.{suffix}"));
            }
        }
        for suffix in ["weight", "scales", "biases"] {
            held.push(format!("{FC}.{suffix}"));
        }
        for (name, _) in &expect.vectors {
            held.push(name.clone());
        }
        held.extend([NORM, PRE_EMBEDDING, PRE_HIDDEN].map(str::to_owned));
        for name in self.st.names() {
            if !held.iter().any(|h| h == name) {
                let kind = if name.contains("linear_attn") {
                    "a gated delta layer's tensor"
                } else if name.contains("switch_mlp") || name.contains("mlp.gate.") {
                    "a routed FFN's tensor"
                } else {
                    "a tensor the draft layer does not have"
                };
                return Err(mtp_refuse(format!(
                    "{}: {name} is {kind}; the draft layer is attention with a dense FFN",
                    self.path.display()
                )));
            }
        }
        for name in &held {
            if self.st.entry(name).is_none() {
                return Err(mtp_refuse(format!(
                    "{}: no tensor {name}",
                    self.path.display()
                )));
            }
        }
        let fc = (FC.to_owned(), expect.hidden, 2 * expect.hidden, Quant::A4);
        for (base, rows, cols, quant) in expect.linears.iter().chain([&fc]) {
            if *quant != Quant::A4 {
                return Err(mtp_refuse(format!(
                    "{base}: the file's layer reads this projection as {}; the sidecar's groups \
                     are MLX's affine 4-bit (A4)",
                    quant.name()
                )));
            }
            let shape = self.shape(base)?;
            if shape != (*rows, *cols) {
                return Err(mtp_refuse(format!(
                    "{}: {base} is {} rows of {} columns, the model's layer needs {rows} of {cols}",
                    self.path.display(),
                    shape.0,
                    shape.1
                )));
            }
            let groups = vec![*rows, cols / GROUP];
            for suffix in ["scales", "biases"] {
                let e = self.st.entry(&format!("{base}.{suffix}"));
                if e.is_none_or(|e| e.dtype != Dtype::Bf16 || e.shape != groups) {
                    return Err(mtp_refuse(format!(
                        "{}: {base}.{suffix} is not BF16 of shape {groups:?}",
                        self.path.display()
                    )));
                }
            }
        }
        for (name, len) in expect.vectors.iter().cloned().chain(
            [NORM, PRE_EMBEDDING, PRE_HIDDEN]
                .into_iter()
                .map(|n| (n.to_owned(), expect.hidden)),
        ) {
            let e = self.st.entry(&name);
            if e.is_none_or(|e| e.dtype != Dtype::Bf16 || e.shape != [len]) {
                return Err(mtp_refuse(format!(
                    "{}: {name} is not a BF16 vector of {len} values",
                    self.path.display()
                )));
            }
        }
        Ok(())
    }

    /// The f32 values of the BF16 vector `name`, read byte-wise: a sidecar's
    /// small vectors may sit off a 2-byte boundary in the file.
    fn vector(&self, name: &str) -> Result<Vec<f32>, SessionError> {
        let raw = self
            .st
            .raw(name)
            .map_err(|e| mtp_refuse(format!("{name}: {e}")))?;
        if raw.dtype != Dtype::Bf16 {
            return Err(mtp_refuse(format!(
                "{name}: a {} tensor where BF16 is expected",
                raw.dtype.name()
            )));
        }
        let (pairs, rest) = raw.bytes.as_chunks::<2>();
        if !rest.is_empty() {
            return Err(mtp_refuse(format!("{name}: an odd byte count")));
        }
        Ok(pairs
            .iter()
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16))
            .collect())
    }
}

/// MLX's affine 4-bit tensor as the file's MLX_A4 blocks: each group of 64
/// values is its 8 little-endian words of codes, then its scale and its bias
/// as BF16 bits (36 bytes), a row's groups in order, rows in order.
/// `words` is the packed weight (`rows * cols / 8` words), `scales` and
/// `biases` each `rows * cols / 64` BF16 values, as safetensors stores them.
fn a4_blocks(
    name: &str,
    rows: usize,
    cols: usize,
    words: &[u8],
    scales: &[u8],
    biases: &[u8],
) -> Result<Vec<u8>, SessionError> {
    let groups = rows * (cols / GROUP);
    if !cols.is_multiple_of(GROUP)
        || words.len() != groups * 32
        || scales.len() != groups * 2
        || biases.len() != groups * 2
    {
        return Err(mtp_refuse(format!(
            "{name}: {rows} rows of {cols} columns are not {groups} groups of {GROUP} in {} \
             bytes of codes, {} of scales and {} of biases",
            words.len(),
            scales.len(),
            biases.len()
        )));
    }
    let mut out = Vec::with_capacity(groups * 36);
    for g in 0..groups {
        out.extend_from_slice(&words[g * 32..(g + 1) * 32]);
        out.extend_from_slice(&scales[g * 2..g * 2 + 2]);
        out.extend_from_slice(&biases[g * 2..g * 2 + 2]);
    }
    Ok(out)
}

impl PartSource for Sidecar {
    /// A name a layout gives: a linear's base name is its MLX_A4 blocks, a
    /// vector's full name its F32 values.
    fn part(&self, name: &str) -> Result<Part<'static>, SessionError> {
        let scales = format!("{name}.scales");
        if self.st.entry(&scales).is_some() {
            let (rows, cols) = self.shape(name)?;
            let get = |suffix: &str| -> Result<&[u8], SessionError> {
                self.st
                    .raw(&format!("{name}.{suffix}"))
                    .map(|r| r.bytes)
                    .map_err(|e| mtp_refuse(format!("{name}.{suffix}: {e}")))
            };
            let bytes = a4_blocks(
                name,
                rows,
                cols,
                get("weight")?,
                get("scales")?,
                get("biases")?,
            )?;
            let info = TensorInfo {
                name: name.to_owned(),
                dims: vec![cols as u64, rows as u64],
                ty: GgmlType::MLX_A4,
                offset: 0,
                nbytes: bytes.len() as u64,
            };
            return Ok(Part::made(info, bytes, None));
        }
        let values = self.vector(name)?;
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let info = TensorInfo {
            name: name.to_owned(),
            dims: vec![values.len() as u64],
            ty: GgmlType::F32,
            offset: 0,
            nbytes: bytes.len() as u64,
        };
        Ok(Part::made(info, bytes, None))
    }
}

/// The draft layer's buffers and the join's scratch.
struct MtpBufs {
    /// The draft layer alone: swapped into the arena while a step encodes.
    layers: Vec<LayerBufs>,
    /// The draft's final norm gain, swapped in as the arena's head norm.
    head_norm: MetalBuffer<f32>,
    /// The fusion projection `[hidden][2 · hidden]`, A4.
    fc: MetalBuffer<u8>,
    gain_emb: MetalBuffer<f32>,
    gain_hid: MetalBuffer<f32>,
    /// The token's embedding row.
    emb: MetalBuffer<f32>,
    /// The hidden row the next step joins, written by the host.
    hid: MetalBuffer<f32>,
    /// The join's output: the embedding half, then the hidden half.
    join: MetalBuffer<f32>,
    /// The draft token's id, written by the host.
    id: MetalBuffer<u32>,
}

/// An opened draft: its buffers, and where in the plan its layer's shape is.
pub(super) struct Mtp {
    bufs: MtpBufs,
    /// The plan layer the draft's walk reads.
    plan_layer: usize,
    eps: f32,
    /// The bytes the sidecar's weights occupy on the device.
    weight_bytes: u64,
    /// The bytes of the draft layer's K/V planes.
    kv_bytes: u64,
}

/// What [`Session::open_mtp`] made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MtpOpened {
    /// The device bytes of the draft's weights and norm gains.
    pub weight_bytes: u64,
    /// The device bytes of the draft layer's K/V planes.
    pub kv_bytes: u64,
}

/// A draft step's result: the pick, at the position the step was run at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Drafted {
    pub id: u32,
    pub pos: u32,
}

/// The draft step's own time, repeated at one position: the command buffer's
/// GPU seconds and the host's wall from encode to the settled wait, per step.
#[derive(Clone, Debug, PartialEq)]
pub struct DraftTimes {
    pub gpu_ms: Vec<f64>,
    pub wall_ms: Vec<f64>,
}

impl DraftTimes {
    /// The median of `v`, in the units it holds.
    #[must_use]
    pub fn median(v: &[f64]) -> f64 {
        let mut s = v.to_vec();
        s.sort_by(f64::total_cmp);
        match s.len() {
            0 => f64::NAN,
            n if n % 2 == 1 => s[n / 2],
            n => (s[n / 2 - 1] + s[n / 2]) / 2.0,
        }
    }
}

impl Session {
    /// Open the MTP sidecar at `path` as the draft layer of this session: its
    /// config and tensors checked against the file's own attention layer, its
    /// weights copied to the device and registered resident, its K/V planes
    /// made at the session's cache length. A sidecar of another model (a
    /// hidden size or vocabulary that differs, another layer kind, other
    /// shapes or types) is refused by name before any buffer is made. The
    /// session's own bytes and programs are unchanged by it; a second open
    /// is refused.
    pub fn open_mtp(&mut self, path: &Path) -> Result<MtpOpened, SessionError> {
        if self.mtp.is_some() {
            return Err(mtp_refuse("a sidecar is open; one draft layer a session"));
        }
        let seed = self.seed.as_ref().ok_or_else(|| {
            mtp_refuse(
                "the file has no attention layer with a dense FFN and a plain residual to shape \
                 the draft layer after",
            )
        })?;
        let layout = seed.layout()?;
        let side = Sidecar::open(path)?;
        side.check(&seed.expect(&layout))?;
        let hidden = seed.hidden;
        let residency = self
            .stream
            .residency()
            .ok_or_else(|| refuse("the stream has no residency set"))?;
        let staged = residency.stage();
        let alloc = Alloc::new(&self.device, &staged);
        let geom = &self.arena.geom;
        let layer = load_layer(alloc, &side, geom, &layout)?;
        let kv_bytes = match &layer.mixer {
            arena::MixerBufs::Attention(a) => bytes_of(&a.k_cache) + bytes_of(&a.v_cache),
            _ => 0,
        };
        let fc_site = Site {
            name: FC.to_owned(),
            cols: 2 * hidden,
            rows: hidden,
            quant: Quant::A4,
        };
        let fc = load_site(alloc, &side, geom, &fc_site)?;
        let vector = |name: &str| {
            load_values(
                alloc,
                &side,
                geom,
                &Values {
                    name: name.to_owned(),
                    len: hidden,
                    ty: GgmlType::F32,
                },
            )
        };
        let head_norm = vector(NORM)?;
        let gain_emb = vector(PRE_EMBEDDING)?;
        let gain_hid = vector(PRE_HIDDEN)?;
        let scratch = |n: usize| -> Result<MetalBuffer<f32>, SessionError> {
            let mut b = alloc.buffer::<f32>(Elems(n))?;
            b.zero();
            Ok(b)
        };
        let (emb, hid, join) = (scratch(hidden)?, scratch(hidden)?, scratch(2 * hidden)?);
        let mut id = alloc.buffer::<u32>(Elems(1))?;
        id.zero();
        let weight_bytes = bytes_of(&fc)
            + bytes_of(&head_norm)
            + bytes_of(&gain_emb)
            + bytes_of(&gain_hid)
            + layer_bytes(&layer);
        let bufs = MtpBufs {
            layers: vec![layer],
            head_norm,
            fc,
            gain_emb,
            gain_hid,
            emb,
            hid,
            join,
            id,
        };
        staged.commit();
        let (plan_layer, eps) = (seed.layer, seed.eps);
        self.mtp = Some(Mtp {
            bufs,
            plan_layer,
            eps,
            weight_bytes,
            kv_bytes,
        });
        Ok(MtpOpened {
            weight_bytes,
            kv_bytes,
        })
    }

    /// Whether a sidecar is open.
    #[must_use]
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    fn mtp_ref(&self) -> Result<&Mtp, SessionError> {
        self.mtp
            .as_ref()
            .ok_or_else(|| mtp_refuse("no sidecar is open (`open_mtp` first)"))
    }

    /// The hidden row of the draft's next step, written by the host: `row`
    /// is the target's hidden row before its output norm, `hidden` values.
    pub fn set_mtp_hidden(&mut self, row: &[f32]) -> Result<(), SessionError> {
        let m = self.mtp_ref()?;
        if row.len() != m.bufs.hid.len().get() {
            return Err(mtp_refuse(format!(
                "a hidden row of {} values; the model's is {}",
                row.len(),
                m.bufs.hid.len().get()
            )));
        }
        self.settle_for_read()?;
        let m = self.mtp.as_mut().expect("checked above");
        m.bufs.hid.write(row)?;
        Ok(())
    }

    /// The draft's next hidden row taken from the residual the last settled
    /// step left in the arena: the row before the output norm, the position
    /// `pos - 1` the session stands at. A token running ahead is refused:
    /// the residual is its own.
    pub fn tap_mtp_hidden(&mut self) -> Result<(), SessionError> {
        self.mtp_ref()?;
        self.settle_for_read()?;
        // Nothing runs ahead (checked by `settle_for_read`) and the last step
        // settled before it returned, so the copy reads the residual it left.
        let row = self.arena.x.pages().to_vec();
        let m = self.mtp.as_mut().expect("checked above");
        m.bufs.hid.write(&row)?;
        Ok(())
    }

    /// Wait for the stream, and refuse a token running ahead, a pending
    /// verify and a poisoned session.
    fn settle_for_read(&self) -> Result<(), SessionError> {
        if let Some(fault) = self.poisoned {
            return Err(SessionError::Poisoned(fault));
        }
        self.not_pending("a draft step")?;
        if self.ahead.is_some() {
            return Err(mtp_refuse(
                "a token runs ahead; a host-fed step or a settled pass drops it first",
            ));
        }
        self.stream.sync()?;
        Ok(())
    }

    /// One draft step at position `pos` with the token `id`: the join of its
    /// embedding with the hidden row set or tapped before it (read as
    /// `hidden`), the fusion projection, the draft layer (appending K/V row
    /// `pos` and attending rows `0..=pos`), the draft's norm and the target's
    /// head. The pick is returned and its logits row is
    /// [`Session::logits_row`]. A token running ahead is dropped first; a
    /// fault poisons the session as a step's does.
    pub fn draft_step(
        &mut self,
        id: u32,
        pos: u32,
        hidden: HiddenRow,
    ) -> Result<Drafted, SessionError> {
        self.draft_buffer(id, pos, hidden).map(|(d, _)| d)
    }

    /// [`Session::draft_step`] and the command buffer's stamps.
    fn draft_buffer(
        &mut self,
        id: u32,
        pos: u32,
        hidden: HiddenRow,
    ) -> Result<(Drafted, f64), SessionError> {
        self.mtp_ref()?;
        if let Some(fault) = self.poisoned {
            return Err(SessionError::Poisoned(fault));
        }
        self.not_pending("a draft step")?;
        if pos as usize >= self.arena.ctx {
            return Err(mtp_refuse(format!(
                "a draft step at position {pos}: the cache holds {} rows",
                self.arena.ctx
            )));
        }
        if let Some(a) = self.ahead.take() {
            self.drop_ahead(a)?;
        }
        let mtp = self.mtp.as_mut().expect("checked above");
        mtp.bufs.id.pages().store(0, id, Ordering::Relaxed)?;
        let seed_plan: &LayerPlan = self
            .plans
            .get(mtp.plan_layer)
            .ok_or_else(|| mtp_refuse("the draft layer's plan is not in the file's"))?;
        let mut recording = self.stream.begin()?;
        let mut enc = recording.token_encoder()?;
        let encoded = encode_draft(
            &mut enc,
            &mut self.arena,
            &self.kernels,
            mtp,
            seed_plan,
            pos,
            hidden,
        );
        enc.end();
        encoded?;
        let stamps = recording.stamps();
        let signalled = self.stream.submit_signalled(recording)?;
        let settled = self.stream.settle(signalled)?;
        let slot = slot_of(pos);
        if let Some(fault) = self.arena.faults[slot].read_settled(&settled)? {
            self.poisoned = Some(fault);
            return Err(SessionError::Fault(fault));
        }
        let gpu_s = match stamps.interval(STAMP_WAIT)? {
            GpuInterval::Ran(s) => s,
            GpuInterval::StillRunning => {
                return Err(mtp_refuse(
                    "the draft step's buffer still runs after its settle",
                ));
            }
        };
        let id = self.arena.step_id.pages().load(slot, Ordering::Relaxed)?;
        Ok((Drafted { id, pos }, gpu_s))
    }

    /// `n` draft steps at one position over one hidden row, each its own
    /// command buffer and wait: the buffer's GPU milliseconds and the host's
    /// wall milliseconds of each, in order. Every step does the work a step
    /// at that position does (it appends the same K/V row over itself), so the
    /// times are those of a draft at that depth.
    pub fn draft_step_times(
        &mut self,
        id: u32,
        pos: u32,
        hidden: HiddenRow,
        n: usize,
    ) -> Result<DraftTimes, SessionError> {
        let mut times = DraftTimes {
            gpu_ms: Vec::with_capacity(n),
            wall_ms: Vec::with_capacity(n),
        };
        for _ in 0..n {
            let started = Instant::now();
            let (_, gpu_s) = self.draft_buffer(id, pos, hidden)?;
            let wall: Duration = started.elapsed();
            times.gpu_ms.push(gpu_s * 1e3);
            times.wall_ms.push(wall.as_secs_f64() * 1e3);
        }
        Ok(times)
    }

    /// The sidecar's resident bytes, for a caller that prints them.
    #[must_use]
    pub fn mtp_bytes(&self) -> Option<MtpOpened> {
        self.mtp.as_ref().map(|m| MtpOpened {
            weight_bytes: m.weight_bytes,
            kv_bytes: m.kv_bytes,
        })
    }
}

/// The device bytes of the weights of one layer of buffers: its mixer's and
/// its FFN's.
fn layer_bytes(l: &LayerBufs) -> u64 {
    let arena::MixerBufs::Attention(a) = &l.mixer else {
        return 0;
    };
    let mut n = bytes_of(&a.out) + bytes_of(&a.p1.norm);
    n += a.p1.parts.iter().map(|p| bytes_of(&p.weights)).sum::<u64>();
    n += a
        .q_norm
        .iter()
        .chain(a.k_norm.iter())
        .map(bytes_of)
        .sum::<u64>();
    n + bytes_of(&l.ffn.gate) + bytes_of(&l.ffn.up) + bytes_of(&l.ffn.down) + bytes_of(&l.ffn.norm)
}

/// The bytes buffer `b` holds.
fn bytes_of<T: bloomery_device::Elem>(b: &MetalBuffer<T>) -> u64 {
    b.pages().len().get() as u64 * core::mem::size_of::<T>() as u64
}

/// The draft step's groups on `enc`: the embedding of the token, the join, the
/// fusion projection, the draft layer's walk and the head, each closed by a
/// barrier. The arena's layer list and head norm are the draft's for the
/// walk and are put back whatever the walk returns.
fn encode_draft(
    enc: &mut crate::encode::ComputeEncoder<'_>,
    arena: &mut Arena,
    kernels: &Kernels,
    mtp: &mut Mtp,
    plan: &LayerPlan,
    pos: u32,
    hidden: HiddenRow,
) -> Result<(), SessionError> {
    let slot = slot_of(pos);
    let h = arena.hidden;
    fn all(
        b: &MetalBuffer<f32>,
    ) -> Result<
        bloomery_device::Window<'_, f32, crate::device::MetalDevice>,
        bloomery_device::DeviceError,
    > {
        b.window(Elems(0), b.len())
    }
    {
        let b = &mut mtp.bufs;
        kernels.embed(
            enc,
            EmbedParams {
                quant: arena.embed_quant,
                table: arena.embed.window(Elems(0), arena.embed.len())?,
                vocab: arena.vocab,
                cols: h,
                ids: b.id.window(Elems(0), b.id.len())?,
                out: b.emb.window_mut(Elems(0), Elems(h))?,
                fault: &arena.faults[slot],
                layer: LAYER,
            },
        )?;
        enc.memory_barrier();
        kernels.mtp_join(
            enc,
            MtpJoinParams {
                cols: h,
                emb: all(&b.emb)?,
                hid: all(&b.hid)?,
                gain_emb: all(&b.gain_emb)?,
                gain_hid: all(&b.gain_hid)?,
                gain_out: all(&arena.head_norm)?,
                hidden,
                y: b.join.window_mut(Elems(0), Elems(2 * h))?,
                eps: mtp.eps,
                fault: &arena.faults[slot],
                layer: LAYER,
            },
        )?;
        enc.memory_barrier();
        kernels.matvec(
            enc,
            MatvecParams {
                quant: Quant::A4,
                rows: h,
                cols: 2 * h,
                weights: b.fc.window(Elems(0), b.fc.len())?,
                x: all(&b.join)?,
                y: arena.x.window_mut(Elems(0), Elems(h))?,
                mode: Mode::Store,
            },
        )?;
        enc.memory_barrier();
    }
    std::mem::swap(&mut arena.layers, &mut mtp.bufs.layers);
    std::mem::swap(&mut arena.head_norm, &mut mtp.bufs.head_norm);
    let walked = {
        let mut ops = MetalOps::new(enc, arena, kernels, pos, Feed::Host, true);
        let cx = LayerCtx {
            layer: 0,
            plan,
            m: 1,
            last: true,
        };
        walk::layer(&mut ops, cx).and_then(|()| ops.head(1))
    };
    std::mem::swap(&mut arena.layers, &mut mtp.bufs.layers);
    std::mem::swap(&mut arena.head_norm, &mut mtp.bufs.head_norm);
    walked
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A packed weight, scales and biases of `rows` x `cols`, drawn from a
    /// counter, and the value MLX's `dequantize` gives each place
    /// (`scale · code + bias`, the low nibble of word `i / 8` first).
    fn drawn(rows: usize, cols: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<f32>) {
        let groups = rows * (cols / GROUP);
        let mut words = Vec::with_capacity(groups * 32);
        for i in 0..groups * 8 {
            let w: u32 = (i as u32)
                .wrapping_mul(0x9E37_79B1)
                .rotate_left(i as u32 % 31);
            words.extend_from_slice(&w.to_le_bytes());
        }
        let bf = |x: f32| ((x.to_bits() >> 16) as u16).to_le_bytes();
        let scale = |g: usize| 0.0625 * (1 + g % 5) as f32;
        let bias = |g: usize| -0.5 + 0.125 * (g % 7) as f32;
        let (mut scales, mut biases) = (Vec::new(), Vec::new());
        for g in 0..groups {
            scales.extend_from_slice(&bf(scale(g)));
            biases.extend_from_slice(&bf(bias(g)));
        }
        let mut want = Vec::with_capacity(rows * cols);
        for i in 0..rows * cols {
            let g = i / GROUP;
            let word = u32::from_le_bytes(words[(i / 8) * 4..(i / 8) * 4 + 4].try_into().unwrap());
            let code = ((word >> (4 * (i % 8))) & 0xf) as f32;
            want.push(scale(g) * code + bias(g));
        }
        (words, scales, biases, want)
    }

    /// The MLX_A4 blocks decode, through the file reader's own `dequant_row`,
    /// to the values MLX's affine dequantize gives the stored planes: the
    /// repack moves bytes and changes no value. Mutant: scale and bias written
    /// in the other order (every value moves).
    #[test]
    fn the_repacked_blocks_decode_to_mlx_s_dequantize() {
        let (rows, cols) = (3, 128);
        let (words, scales, biases, want) = drawn(rows, cols);
        let blocks = a4_blocks("t", rows, cols, &words, &scales, &biases).unwrap();
        assert_eq!(blocks.len(), rows * cols / GROUP * 36);
        let mut got = vec![0f32; rows * cols];
        gguf::quant::dequant_row(GgmlType::MLX_A4, &blocks, &mut got).unwrap();
        assert_eq!(got, want);
    }

    #[test]
    fn a_plane_of_another_length_is_refused_by_name() {
        let (words, scales, biases, _) = drawn(2, 64);
        let e = a4_blocks(
            "layers.0.mlp.up_proj",
            2,
            64,
            &words[..words.len() - 4],
            &scales,
            &biases,
        )
        .unwrap_err();
        assert!(e.to_string().contains("layers.0.mlp.up_proj"), "{e}");
        let e = a4_blocks("t", 2, 60, &words, &scales, &biases).unwrap_err();
        assert!(e.to_string().contains("groups of 64"), "{e}");
    }

    /// A sidecar written to a temporary folder: `tensors` by name, dtype,
    /// shape and bytes, and a config of `text` for its `config.json`.
    fn folder(tag: &str, config: &str, tensors: &[(&str, &str, &[usize], &[u8])]) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("bloomery-mtp-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), config).unwrap();
        std::fs::write(dir.join("model.safetensors"), safetensors::write(tensors)).unwrap();
        dir
    }

    fn config(hidden: usize, vocab: usize, extra: &str, bits: u64) -> String {
        format!(
            r#"{{"model_type": "qwen3_5_mtp",
                "quantization": {{"group_size": 64, "bits": {bits}, "mode": "affine"}},
                "text_config": {{"hidden_size": {hidden}, "vocab_size": {vocab},
                                 "mtp_num_hidden_layers": 1 {extra}}}}}"#
        )
    }

    /// Every refusal names what differs: another model's hidden size or
    /// vocabulary, another layer kind (a gated delta tensor, a router), another
    /// quantization, and a model type that is no MTP sidecar.
    #[test]
    fn a_sidecar_of_another_model_is_refused_by_name() {
        let expect = Expect {
            hidden: 64,
            vocab: 100,
            linears: vec![],
            vectors: vec![],
        };
        let vec64 = vec![0u8; 128];
        let norms: Vec<(&str, &str, &[usize], &[u8])> = vec![
            ("norm.weight", "BF16", &[64], &vec64),
            ("pre_fc_norm_embedding.weight", "BF16", &[64], &vec64),
            ("pre_fc_norm_hidden.weight", "BF16", &[64], &vec64),
        ];
        let open = |tag: &str, cfg: String, extra: &[(&str, &str, &[usize], &[u8])]| {
            let all: Vec<_> = norms.iter().chain(extra).copied().collect();
            let dir = folder(tag, &cfg, &all);
            let r = Sidecar::open(&dir);
            let _ = std::fs::remove_dir_all(&dir);
            r
        };
        // Another hidden size and another vocabulary.
        let side = open("hidden", config(128, 100, "", 4), &[]).unwrap();
        let e = side.check(&expect).unwrap_err().to_string();
        assert!(
            e.contains("hidden size 128") && e.contains("another model"),
            "{e}"
        );
        let side = open("vocab", config(64, 99, "", 4), &[]).unwrap();
        let e = side.check(&expect).unwrap_err().to_string();
        assert!(e.contains("vocabulary 99"), "{e}");
        // Another layer kind: a gated delta tensor, and a routed FFN's.
        let one = vec![0u8; 8];
        let delta: [(&str, &str, &[usize], &[u8]); 1] =
            [("layers.0.linear_attn.A_log", "BF16", &[4], &one)];
        let side = open("delta", config(64, 100, "", 4), &delta).unwrap();
        let e = side.check(&expect).unwrap_err().to_string();
        assert!(
            e.contains("layers.0.linear_attn.A_log") && e.contains("gated delta"),
            "{e}"
        );
        let routed: [(&str, &str, &[usize], &[u8]); 1] =
            [("layers.0.mlp.switch_mlp.up_proj.weight", "BF16", &[4], &one)];
        let side = open("routed", config(64, 100, "", 4), &routed).unwrap();
        let e = side.check(&expect).unwrap_err().to_string();
        assert!(e.contains("routed FFN"), "{e}");
        // A routed config, another quantization, another model type.
        let e = open(
            "experts",
            config(64, 100, r#", "num_experts": 256"#, 4),
            &[],
        )
        .err()
        .unwrap()
        .to_string();
        assert!(e.contains("routed draft layer"), "{e}");
        let e = open("bits", config(64, 100, "", 8), &[])
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("affine 4-bit"), "{e}");
        let e = open(
            "type",
            config(64, 100, "", 4).replace("qwen3_5_mtp", "qwen3_5"),
            &[],
        )
        .err()
        .unwrap()
        .to_string();
        assert!(e.contains("no MTP sidecar"), "{e}");
    }

    /// A folder with no weights file is refused by the names it looked for.
    #[test]
    fn a_folder_with_no_weights_is_refused_by_name() {
        let dir = std::env::temp_dir().join(format!("bloomery-mtp-empty-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let e = Sidecar::open(&dir).err().unwrap().to_string();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(e.contains("mtp/model.safetensors"), "{e}");
    }
}
