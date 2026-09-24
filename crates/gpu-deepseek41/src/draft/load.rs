//! The draft's weights on its card, each tensor in the format its kernel
//! reads ([`dspark::card_format`]):
//!
//! - Q8_0, F32 and the router's bf16 gate (widened to f32) through the GPU
//!   loader ([`Weights::load_where`]), which refuses and checks every upload
//!   by `CardFormat`'s arithmetic;
//! - the routed MXFP4 stacks through [`MxStack::upload`], in
//!   `bloomery_gpu::mxfp4`'s two-plane layout;
//! - the Markov weights as the file's bf16 words ([`CardFormat::Bf16Raw`]),
//!   which `ds41_markov` widens on the card;
//! - from the target's file, the head's Q6_K projection (`output.weight`,
//!   [`Borrow::Copied`]) and one embedding row, the mask token's
//!   ([`Borrow::RowSource`]: the block's other rows are read per pass), as
//!   the file stores it ([`EmbdRows`]: bf16, or Q3_K super-blocks).
//!
//! The file is checked against [`dspark::inventory`] before anything is
//! uploaded: an absent, misshapen or unread tensor is refused by name.

use bloomery_gpu::weights::{DevWeight, Weights};
use bloomery_gpu::{DeviceTensor, GpuError};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use gguf::quant::GgmlType;
use model::arch::deepseek41::names as target_names;
use model::arch::dspark::{self, Borrow, DraftHparams, DraftTensor, Group, names};
use model::placement::CardFormat;

use crate::experts_mxfp4::{MxStack, N_EXPERT, N_USED};
use crate::markov::MARKOV_ROW_WORDS;

const WHAT: &str = "draft::load";

/// Values of one Q3_K super-block, and its bytes.
const Q3_K_BLOCK: usize = 256;
const Q3_K_BYTES: usize = 110;

/// How the target's `token_embd` rows sit in its file, and so in the draft's
/// buffers and images: the file's bytes, a row's words from its first byte.
/// The card decodes them with the target's own glue entries — bf16 widened
/// exactly, Q3_K by `elem::q3k_embed_value` — so a draft row and the target's
/// row of the same id are the same f32.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbdRows {
    /// Two bf16 to a word, the low half first ([`CardFormat::Bf16Raw`]).
    Bf16,
    /// Q3_K super-blocks, 110 bytes each ([`CardFormat::KQuant`]); the last
    /// word of a row with an odd super-block count is zero-padded.
    Q3K,
}

impl EmbdRows {
    /// The row format of a `token_embd` of type `ty` with rows of `n_embd`
    /// values; `None` for any other type, and for a Q3_K row that is not
    /// whole super-blocks.
    #[must_use]
    pub fn of(ty: GgmlType, n_embd: usize) -> Option<EmbdRows> {
        match ty {
            GgmlType::BF16 => Some(EmbdRows::Bf16),
            GgmlType::Q3_K if n_embd.is_multiple_of(Q3_K_BLOCK) => Some(EmbdRows::Q3K),
            _ => None,
        }
    }

    /// The file bytes of one row of `n_embd` values.
    #[must_use]
    pub fn row_bytes(self, n_embd: usize) -> usize {
        match self {
            EmbdRows::Bf16 => 2 * n_embd,
            EmbdRows::Q3K => n_embd / Q3_K_BLOCK * Q3_K_BYTES,
        }
    }

    /// The u32 words one row takes in a buffer or an image.
    #[must_use]
    pub fn row_words(self, n_embd: usize) -> usize {
        self.row_bytes(n_embd).div_ceil(4)
    }

    /// The card format the mask row is held in.
    fn card_format(self) -> CardFormat {
        match self {
            EmbdRows::Bf16 => CardFormat::Bf16Raw,
            EmbdRows::Q3K => CardFormat::KQuant,
        }
    }

    /// The file type these rows are.
    fn ty(self) -> GgmlType {
        match self {
            EmbdRows::Bf16 => GgmlType::BF16,
            EmbdRows::Q3K => GgmlType::Q3_K,
        }
    }
}

/// One draft layer's routed experts.
pub struct LayerExperts {
    pub gate: MxStack,
    pub up: MxStack,
    pub down: MxStack,
}

/// How one tensor sits on the card, for the load table.
#[derive(Clone, Debug)]
pub struct Loaded {
    pub name: String,
    /// The draft group; `None` for a tensor borrowed from the target.
    pub group: Option<Group>,
    /// The layout: a `CardFormat` label, or `mxfp4_planes`.
    pub format: &'static str,
    /// The device buffers in bytes, in allocation order: read from the
    /// buffers, except an MXFP4 stack's two planes, which are derived from
    /// its shape (codes 16 B and one scale byte per 32 values).
    pub buffers: Vec<u64>,
}

/// The draft's weights on its card. See the module comment.
pub struct DraftWeights {
    hp: DraftHparams,
    /// Every draft tensor the GPU loader formats, by name.
    dense: Weights,
    /// The target's `output.weight`.
    head: Weights,
    markov_w1: DeviceTensor<u32>,
    markov_w2: DeviceTensor<u32>,
    /// The mask token's embedding row, the file's bytes as words
    /// ([`EmbdRows`]).
    mask_row: DeviceTensor<u32>,
    embd: EmbdRows,
    experts: Vec<LayerExperts>,
    table: Vec<Loaded>,
}

/// The file bytes of tensor `name` of `split` (any shard).
fn file_bytes<'a>(split: &'a Split, name: &str) -> Result<&'a [u8], GpuError> {
    let (s, t) = split.find(name).ok_or_else(|| GpuError::Tensor {
        what: WHAT,
        name: name.to_string(),
        need: "in the file",
    })?;
    let g = split.shard(s).ok_or(GpuError::State {
        what: WHAT,
        missing: "a shard the split names",
    })?;
    Ok(g.data(t)?)
}

/// Little-endian bytes as u32 words; `b.len()` a multiple of 4.
fn words(b: &[u8]) -> Vec<u32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

/// Little-endian bytes as u32 words into `out`, the last word zero-padded
/// when `b.len()` is not a multiple of 4; `out.len() == b.len().div_ceil(4)`.
pub(super) fn pack_words(b: &[u8], out: &mut [u32]) {
    for (d, c) in out.iter_mut().zip(b.chunks(4)) {
        let mut w = [0u8; 4];
        w[..c.len()].copy_from_slice(c);
        *d = u32::from_le_bytes(w);
    }
}

/// The mask token's row `bytes` (one row of `n_embd` values in `embd`'s
/// format), uploaded as words and checked against its card format's
/// arithmetic.
fn upload_mask_row(
    stream: &CudaStream,
    name: &str,
    bytes: &[u8],
    embd: EmbdRows,
    n_embd: usize,
) -> Result<DeviceTensor<u32>, GpuError> {
    let shape = |detail: String| GpuError::Shape { what: WHAT, detail };
    let words = embd.row_words(n_embd);
    let size = embd
        .card_format()
        .resident_bytes(embd.ty(), n_embd as u64, 1)
        .ok_or_else(|| shape(format!("{name}: a row of {n_embd} has no {embd:?} layout")))?;
    if bytes.len() != embd.row_bytes(n_embd) || size != 4 * words as u64 {
        return Err(shape(format!(
            "{name}: a row of {} bytes in {size} card bytes; {embd:?} rows of {n_embd} are {} \
             bytes in {} words",
            bytes.len(),
            embd.row_bytes(n_embd),
            words
        )));
    }
    let mut w = vec![0u32; words];
    pack_words(bytes, &mut w);
    let t = DeviceTensor::upload(stream, &w, 1, words)?;
    if t.buf().num_bytes() as u64 != size {
        return Err(shape(format!(
            "{name}: uploaded {} bytes, the row's card format has {size}",
            t.buf().num_bytes()
        )));
    }
    Ok(t)
}

/// `rows` rows of `k` bf16 values in [`CardFormat::Bf16Raw`], uploaded and
/// checked against the format's arithmetic.
fn upload_bf16_raw(
    stream: &CudaStream,
    name: &str,
    bytes: &[u8],
    k: u64,
    rows: u64,
) -> Result<DeviceTensor<u32>, GpuError> {
    let shape = |detail: String| GpuError::Shape { what: WHAT, detail };
    let size = CardFormat::Bf16Raw
        .resident_bytes(GgmlType::BF16, k, rows)
        .ok_or_else(|| shape(format!("{name}: {rows} rows of {k} have no Bf16Raw layout")))?;
    let bytes = usize::try_from(size)
        .ok()
        .and_then(|n| bytes.get(..n))
        .ok_or_else(|| shape(format!("{name}: fewer than {size} bytes")))?;
    let (rows, cols) = (
        usize::try_from(rows).map_err(|_| shape(format!("{name}: {rows} rows")))?,
        usize::try_from(k / 2).map_err(|_| shape(format!("{name}: {k} values")))?,
    );
    let t = DeviceTensor::upload(stream, &words(bytes), rows, cols)?;
    if t.buf().num_bytes() as u64 != size {
        return Err(shape(format!(
            "{name}: uploaded {} bytes, Bf16Raw has {size}",
            t.buf().num_bytes()
        )));
    }
    Ok(t)
}

/// An MXFP4 stack `[k, rows_per_expert, n_experts]` of the draft file.
fn upload_stack(stream: &CudaStream, split: &Split, t: &DraftTensor) -> Result<MxStack, GpuError> {
    let d = |i: usize| {
        usize::try_from(t.dims[i]).map_err(|_| GpuError::Shape {
            what: WHAT,
            detail: format!("{}: dim {i} = {}", t.name, t.dims[i]),
        })
    };
    MxStack::upload(stream, file_bytes(split, &t.name)?, d(2)?, d(1)?, d(0)?)
}

/// An MXFP4 stack's two planes in bytes, derived from its shape.
fn stack_buffers(s: &MxStack) -> Vec<u64> {
    let blocks = (s.n_experts() * s.rows_per_expert() * s.k() / 32) as u64;
    vec![16 * blocks, blocks]
}

fn format_label(f: CardFormat) -> &'static str {
    match f {
        CardFormat::KQuant => "kquant",
        CardFormat::Q5_0 => "q5_0",
        CardFormat::Q5_1 => "q5_1",
        CardFormat::Q8_0Planes => "q8_0_planes",
        CardFormat::F32 => "f32",
        CardFormat::Bf16AsF32 => "bf16_as_f32",
        CardFormat::Bf16Raw => "bf16_raw",
    }
}

/// A tensor the load expected and does not hold.
fn missing(name: String, need: &'static str) -> GpuError {
    GpuError::Tensor {
        what: WHAT,
        name,
        need,
    }
}

/// The draft's shape against what its kernels take: the router's 128/3 and
/// the Markov rank.
fn check_kernels(hp: &DraftHparams) -> Result<(), GpuError> {
    if hp.experts.n_expert != N_EXPERT || hp.experts.n_used != N_USED {
        return Err(GpuError::Shape {
            what: WHAT,
            detail: format!(
                "the router is {}/{}; the draft kernels take {N_EXPERT}/{N_USED}",
                hp.experts.n_expert, hp.experts.n_used
            ),
        });
    }
    if hp.markov_rank != 2 * MARKOV_ROW_WORDS {
        return Err(GpuError::Shape {
            what: WHAT,
            detail: format!(
                "the Markov rank is {}; ds41_markov takes {}",
                hp.markov_rank,
                2 * MARKOV_ROW_WORDS
            ),
        });
    }
    Ok(())
}

/// What the load gathers besides the two `Weights`, tensor by tensor.
struct Uploads {
    table: Vec<Loaded>,
    w1: Option<DeviceTensor<u32>>,
    w2: Option<DeviceTensor<u32>>,
    /// Per layer: gate, up, down.
    stacks: Vec<[Option<MxStack>; 3]>,
    mask_row: Option<(DeviceTensor<u32>, EmbdRows)>,
}

impl Uploads {
    /// Draft tensor `t`: uploaded here unless the GPU loader already holds
    /// it in `dense`; its row of the table either way.
    fn draft_tensor(
        &mut self,
        stream: &CudaStream,
        draft: &Split,
        dense: &Weights,
        t: &DraftTensor,
    ) -> Result<(), GpuError> {
        let (format, buffers) = match dspark::card_format(t) {
            Some(CardFormat::Bf16Raw) => {
                let bytes = file_bytes(draft, &t.name)?;
                let w = upload_bf16_raw(stream, &t.name, bytes, t.dims[0], t.dims[1])?;
                let b = vec![w.buf().num_bytes() as u64];
                if t.name == names::markov_w1() {
                    self.w1 = Some(w);
                } else if t.name == names::markov_w2() {
                    self.w2 = Some(w);
                } else {
                    return Err(missing(t.name.clone(), "a Bf16Raw reader"));
                }
                ("bf16_raw", b)
            }
            Some(f) => {
                let dw = dense
                    .get(&t.name)
                    .ok_or_else(|| missing(t.name.clone(), "resident"))?;
                let b = dw.buffer_bytes().into_iter().map(|b| b as u64).collect();
                (format_label(f), b)
            }
            None if t.ty == GgmlType::MXFP4 => {
                let l = t.block.ok_or_else(|| missing(t.name.clone(), "a block"))?;
                let slot = [names::ffn_gate_exps(l), names::ffn_up_exps(l)]
                    .iter()
                    .position(|n| *n == t.name)
                    .unwrap_or(2);
                let s = upload_stack(stream, draft, t)?;
                let b = stack_buffers(&s);
                self.stacks[l][slot] = Some(s);
                ("mxfp4_planes", b)
            }
            None => return Err(missing(t.name.clone(), "a card format")),
        };
        self.table.push(Loaded {
            name: t.name.clone(),
            group: Some(t.group),
            format,
            buffers,
        });
        Ok(())
    }

    /// Borrowed tensor `b`: the head already in `head`, or the mask
    /// token's embedding row, uploaded here.
    fn borrowed(
        &mut self,
        stream: &CudaStream,
        target: &Split,
        head: &Weights,
        b: &dspark::Borrowed,
        mask_token: u32,
    ) -> Result<(), GpuError> {
        let (format, buffers) = match (b.borrow, b.card_format()) {
            (Borrow::Copied, Some(f)) => {
                let dw = head
                    .get(&b.name)
                    .ok_or_else(|| missing(b.name.clone(), "resident"))?;
                let v = dw.buffer_bytes().into_iter().map(|b| b as u64).collect();
                (format_label(f), v)
            }
            (Borrow::RowSource, Some(f)) => {
                let n_embd = usize::try_from(b.dims[0]).map_err(|_| GpuError::Shape {
                    what: WHAT,
                    detail: format!("{}: rows of {}", b.name, b.dims[0]),
                })?;
                let embd = EmbdRows::of(b.ty, n_embd)
                    .filter(|e| e.card_format() == f)
                    .ok_or_else(|| missing(b.name.clone(), "a card format for its use"))?;
                let (row, _) = embedding_row(target, &b.name, n_embd, mask_token)?;
                let t = upload_mask_row(stream, &b.name, row, embd, n_embd)?;
                let v = vec![t.buf().num_bytes() as u64];
                self.mask_row = Some((t, embd));
                let label = match embd {
                    EmbdRows::Bf16 => "bf16_raw (mask row)",
                    EmbdRows::Q3K => "q3k_row (mask row)",
                };
                (label, v)
            }
            _ => return Err(missing(b.name.clone(), "a card format for its use")),
        };
        self.table.push(Loaded {
            name: b.name.clone(),
            group: None,
            format,
            buffers,
        });
        Ok(())
    }
}

impl DraftWeights {
    /// Check `draft` against the inventory with `target`'s borrowed tensors,
    /// then upload every tensor the draft reads on `stream`'s card. Refuses,
    /// before any upload, a file the inventory refuses and a draft whose
    /// router or Markov rank is not the kernels'. Load-time only.
    pub fn load(
        stream: &CudaStream,
        draft: &Split,
        target: &Split,
    ) -> Result<DraftWeights, GpuError> {
        let hp = DraftHparams::read(draft).map_err(|e| GpuError::plan(WHAT, e))?;
        let inv = dspark::inventory(draft, &hp, target).map_err(|e| GpuError::plan(WHAT, e))?;
        inv.check().map_err(|e| GpuError::plan(WHAT, e))?;
        check_kernels(&hp)?;
        stream.context().bind_to_thread()?;
        let want = dspark::tensors(&hp);
        let by_loader: Vec<&DraftTensor> = want
            .iter()
            .filter(|t| dspark::card_format(t).is_some_and(|f| CardFormat::of(t.ty) == Some(f)))
            .collect();
        let dense = Weights::load_where(stream, draft, |n| by_loader.iter().any(|t| t.name == n))?;
        let head = Weights::load_where(stream, target, |n| n == target_names::output())?;
        let mut up = Uploads {
            table: Vec::with_capacity(want.len() + inv.borrowed.len()),
            w1: None,
            w2: None,
            stacks: (0..hp.n_layer).map(|_| [None, None, None]).collect(),
            mask_row: None,
        };
        for t in &want {
            up.draft_tensor(stream, draft, &dense, t)?;
        }
        for b in &inv.borrowed {
            up.borrowed(stream, target, &head, b, hp.mask_token)?;
        }
        let experts = std::mem::take(&mut up.stacks)
            .into_iter()
            .enumerate()
            .map(|(l, [g, u, d])| match (g, u, d) {
                (Some(gate), Some(up), Some(down)) => Ok(LayerExperts { gate, up, down }),
                _ => Err(missing(names::ffn_gate_exps(l), "all three stacks")),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (mask_row, embd) = up
            .mask_row
            .ok_or_else(|| missing(target_names::token_embd(), "loaded"))?;
        Ok(DraftWeights {
            markov_w1: up.w1.ok_or_else(|| missing(names::markov_w1(), "loaded"))?,
            markov_w2: up.w2.ok_or_else(|| missing(names::markov_w2(), "loaded"))?,
            mask_row,
            embd,
            hp,
            dense,
            head,
            experts,
            table: up.table,
        })
    }

    /// The draft file's hyperparameters.
    #[must_use]
    pub fn hp(&self) -> &DraftHparams {
        &self.hp
    }

    /// Every tensor on the card, draft tensors in `dspark::tensors` order and
    /// then the borrowed ones.
    #[must_use]
    pub fn table(&self) -> &[Loaded] {
        &self.table
    }

    /// Layer `l`'s routed experts.
    #[must_use]
    pub fn experts(&self, l: usize) -> Option<&LayerExperts> {
        self.experts.get(l)
    }

    /// `markov_w1` and `markov_w2`: `n_vocab` rows of `MARKOV_ROW_WORDS` words each.
    #[must_use]
    pub fn markov(&self) -> (&DeviceTensor<u32>, &DeviceTensor<u32>) {
        (&self.markov_w1, &self.markov_w2)
    }

    /// The mask token's embedding row, [`EmbdRows::row_words`] words.
    #[must_use]
    pub fn mask_row(&self) -> &DeviceTensor<u32> {
        &self.mask_row
    }

    /// How the target's embedding rows sit in its file: the mask row's
    /// format, and every block's `id_last` row's.
    #[must_use]
    pub fn embd_rows(&self) -> EmbdRows {
        self.embd
    }

    /// The target's head projection, resident as the GPU loader holds it.
    #[must_use]
    pub fn head(&self) -> &Weights {
        &self.head
    }

    /// Draft tensor `name` as the GPU loader holds it; `None` for a tensor
    /// the loader does not hold (the MXFP4 stacks, the Markov words).
    #[must_use]
    pub(crate) fn dense(&self, name: &str) -> Option<&DevWeight> {
        self.dense.get(name)
    }

    /// The Q8_0 planes of draft tensor `name`, `rows` rows of `k` values.
    pub fn q8(
        &self,
        name: &str,
        k: usize,
        rows: usize,
    ) -> Result<(&DeviceTensor<u32>, &DeviceTensor<u16>), GpuError> {
        match self.dense.get(name) {
            Some(DevWeight::Q8_0 { qs, d, k: wk }) if *wk == k && d.rows() == rows => Ok((qs, d)),
            Some(DevWeight::Q8_0 { d, k: wk, .. }) => Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "{name} is {} rows of {wk}, the draft reads {rows} of {k}",
                    d.rows()
                ),
            }),
            _ => Err(GpuError::Tensor {
                what: WHAT,
                name: name.to_string(),
                need: "Q8_0 planes",
            }),
        }
    }

    /// The f32 plane of draft tensor `name` (a gain, a bias, a widened bf16
    /// matrix), `len` values in all.
    pub fn f32(&self, name: &str, len: usize) -> Result<&DeviceTensor<f32>, GpuError> {
        match self.dense.get(name) {
            Some(DevWeight::F32 { w, .. }) if w.buf().len() == len => Ok(w),
            Some(DevWeight::F32 { w, .. }) => Err(GpuError::Shape {
                what: WHAT,
                detail: format!("{name} holds {} f32, the draft reads {len}", w.buf().len()),
            }),
            _ => Err(GpuError::Tensor {
                what: WHAT,
                name: name.to_string(),
                need: "an f32 plane",
            }),
        }
    }

    /// [`DraftWeights::f32`]'s buffer.
    pub fn gain(&self, name: &str, len: usize) -> Result<&DeviceBuffer<f32>, GpuError> {
        Ok(self.f32(name, len)?.buf())
    }
}

/// Row `id` of the embedding `name` of `split`, `k` values: its file bytes
/// and their format, the row stride taken from the tensor's type. A type
/// with no [`EmbdRows`] format is refused by name.
pub(super) fn embedding_row<'a>(
    split: &'a Split,
    name: &str,
    k: usize,
    id: u32,
) -> Result<(&'a [u8], EmbdRows), GpuError> {
    let (_, t) = split
        .find(name)
        .ok_or_else(|| missing(name.to_string(), "in the file"))?;
    let embd = EmbdRows::of(t.ty, k).ok_or_else(|| GpuError::Shape {
        what: WHAT,
        detail: format!(
            "{name} is {}; the draft reads bf16 or Q3_K rows of {k}",
            t.ty
        ),
    })?;
    let bytes = file_bytes(split, name)?;
    let row = embd.row_bytes(k);
    let past = || GpuError::Shape {
        what: WHAT,
        detail: format!("{name}: row {id} past its {} bytes", bytes.len()),
    };
    let at = row.checked_mul(id as usize).ok_or_else(past)?;
    let end = at.checked_add(row).ok_or_else(past)?;
    Ok((bytes.get(at..end).ok_or_else(past)?, embd))
}
