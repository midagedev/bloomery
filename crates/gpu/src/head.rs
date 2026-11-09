//! The GPU output head: `result_norm → lm_head → argmax` as one capturable
//! sequence over resident scratch (docs/gpu-design.md decisions 4 and 6). No
//! kernels of its own: after `rms_norm`, a K-quant head uses the shared
//! q8_1 quantizer and its format's gemv, then `argmax_fault`; an NVFP4 head
//! the site's launch for its width ([`site::reads_at`]: `site::kgemv` on the
//! q8_1 rows, `site::nvfp4_sel32_gemv` on the 32-value rows), its token-major
//! logits copied into the argmax's layout past one row, then `argmax_fault`.
//! Q8_0, BF16 and E4M3 heads read f32 activations through the shared gemvs and
//! use `argmax_finite_fault`. The quantized path refuses non-finite inputs;
//! the f32-activation path refuses non-finite logits, using for `m > 1` rows
//! `argmax_rows_finite_fault`, whose last block to finish copies the word, so
//! the readback carries every row's raise.
//!
//! A head carries `m` rows (1..=8), fixed at construction: the decode step
//! samples one position (`m = 1`), a k-token step every row. Each row's
//! normed vector, logits and token are bit for bit that row's `m = 1` head —
//! the norm and quantizer work per row, the Q6_K gemv keeps each column's
//! own accumulation order, and the argmax walks each row alone. The input
//! buffer is exposed mutable so the assembly round can have the last block's
//! residual store write straight into it, and `set_input` is the gate's way
//! in.
//!
//! A head's [`HeadNorm`] is its body's choice, never a guess from the file:
//! [`HeadNorm::Rms`] normalizes by `output_norm.weight` first,
//! [`HeadNorm::Mixed`] projects its input as it stands — a body whose own
//! last launch is the final norm (a hyper-connection head's mix) writes the
//! normed rows into the input. Its file carries no `output_norm`: the plan
//! refuses a qwen4exp file that does by name (the tensor has no role there,
//! `model::arch::qwen35moe::roles`), and a Mixed head reads none.
//!
//! The argmax also carries the card's fault (crate::fault) out: the readback
//! buffer holds the `m` tokens, then the first-layer word and that layer's
//! site mask, one copy, and a raised word turns the readback into
//! [`GpuError::Fault`] instead of a token.
//!
//! A plan that carries no head on a card leaves `output_norm.weight` and
//! `output.weight` on the host, read in place from the file
//! ([`HostHead`], [`Head::host`]): the head is one row and the chain's last
//! launch writes its input as for a card head, the head enqueues nothing, and
//! the readback ([`Head::token`], [`Head::logits_to_host`]) waits for the
//! stream, refuses a raised fault, copies the row to the host and runs the
//! final norm, the projection and the argmax there, over the host's workers.
//! The card holds no readback words of such a head, so a chain that reads
//! them on the card ([`Head::copy_tokens`]) is refused by name.

use crate::Gpu;
use crate::GpuError;
use crate::fault::{Fault, FaultSink, LAYER_HEAD};
use crate::kquant::SelDown;
use crate::q5::Q8Blocks32;
use crate::q8f32::{E4m3GemvArgs, E4m3Weight, GemvOut};
use crate::quant_sel::QuantSelKernels;
use crate::site::{self, Form, KGemvKernels, SiteTy};
use crate::tensor::{DeviceTensor, Q8Act};
use crate::token_major::TokenMajor;
use crate::weights::{DevWeight, HEAD_TENSOR, Weights};
use cuda_core::{CudaEvent, CudaStream, DeviceBuffer, PinnedHostBuffer};
use gguf::{Split, TensorInfo};
use model::Tensor2;
use model::arch::names::output_norm;
use model::ops::{f32_tensor, matmul_q, rms_norm};
use model::placement::kernels::{self, HeadEntry};
use std::sync::{Arc, Mutex, PoisonError};

/// The lm_head weight (`output.weight`) in a format a gemv here reads: a
/// K-quant words over q8_1 rows, NVFP4's planes over q8_1 or 32-value rows,
/// E4M3 or Q8_0 planes, or BF16's words over the f32 rows.
enum OutW<'a> {
    Q6K(&'a DeviceTensor<u32>),
    Q5K(&'a DeviceTensor<u32>),
    Q4K(&'a DeviceTensor<u32>),
    Q8_0 {
        qs: &'a DeviceTensor<u32>,
        d: &'a DeviceTensor<u16>,
    },
    Bf16(&'a DeviceTensor<u32>),
    E4m3(E4m3Weight<'a>),
    /// NVFP4's planes and the tensor's scale, read as the site's launch for `form` reads them
    /// ([`site::reads_at`] at the head's width); `rows` is the vocabulary.
    Nvfp4 {
        rows: usize,
        form: Form,
    },
}

impl OutW<'_> {
    /// Its rows: the vocabulary.
    fn rows(&self) -> usize {
        match self {
            OutW::Q6K(w) | OutW::Q5K(w) | OutW::Q4K(w) | OutW::Bf16(w) => w.rows(),
            OutW::Q8_0 { d, .. } => d.rows(),
            OutW::E4m3(w) => w.qs.rows(),
            OutW::Nvfp4 { rows, .. } => *rows,
        }
    }

    /// Whether its gemv reads f32 rows: the head then holds
    /// no q8_1 form and its argmax refuses a non-finite logit.
    fn reads_f32(&self) -> bool {
        matches!(self, OutW::Q8_0 { .. } | OutW::Bf16(_) | OutW::E4m3(_))
    }

    /// Whether its gemv reads the q8_1 rows the head quantizes its normed rows into: a K-quant
    /// head, and an NVFP4 one at a width of whole super-blocks (an NVFP4 head of another width
    /// quantizes into the 32-value rows).
    fn reads_q8_1(&self) -> bool {
        match self {
            OutW::Nvfp4 { form, .. } => *form == Form::Q8x128,
            other => !other.reads_f32(),
        }
    }

    /// Whether its gemv writes token-major columns that a copy interleaves
    /// into the argmax's layout past one row: the `_sel` Q5_K gemv and NVFP4's.
    fn copies_rows(&self) -> bool {
        matches!(self, OutW::Q5K(_) | OutW::Nvfp4 { .. })
    }
}

/// The head's q8_1 form, refused by name on a head that holds none (its
/// projection reads the f32 rows).
fn held(act: &mut Option<Q8Act>) -> Result<&mut Q8Act, GpuError> {
    act.as_mut().ok_or_else(|| {
        GpuError::state("Head::enqueue", "a q8_1 form (a K-quant lm_head holds one)")
    })
}

/// The lm_head weight (`output.weight`) and its row width. The type is
/// checked, not assumed — a file whose lm_head is another format needs a
/// different gemv, and a silent plane reuse would misread its bytes: a word
/// plane runs by the kernel table's head entry for its type (`kernels::head`).
fn head_out_w(w: &Weights) -> Result<(OutW<'_>, usize), GpuError> {
    match w.get(HEAD_TENSOR) {
        Some(DevWeight::KQuant { ty, w, k }) => match kernels::head(*ty) {
            Some(HeadEntry::Q6k) => Ok((OutW::Q6K(w), *k)),
            Some(HeadEntry::Q5k) => Ok((OutW::Q5K(w), *k)),
            Some(HeadEntry::Q4k) => Ok((OutW::Q4K(w), *k)),
            Some(HeadEntry::Q8_0 | HeadEntry::Bf16 | HeadEntry::Nvfp4) | None => {
                Err(GpuError::shape(
                    "head_out_w",
                    format!(
                        "{HEAD_TENSOR} is {ty} words, which no head entry reads (kernels::head)"
                    ),
                ))
            }
        },
        Some(DevWeight::Q8_0 { qs, d, k }) => Ok((OutW::Q8_0 { qs, d }, *k)),
        Some(DevWeight::Bf16 { w, k }) => Ok((OutW::Bf16(w), *k)),
        Some(DevWeight::Mxfp8 { k, .. } | DevWeight::Fp8B128 { k, .. }) => {
            Ok((OutW::E4m3(w.e4m3("head_out_w", HEAD_TENSOR)?), *k))
        }
        Some(DevWeight::Nvfp4 { qs, k, .. }) => {
            let form = site::reads_at(SiteTy::Nvfp4, *k);
            Ok((
                OutW::Nvfp4 {
                    rows: qs.rows(),
                    form,
                },
                *k,
            ))
        }
        Some(_) => Err(GpuError::tensor(
            "head_out_w",
            HEAD_TENSOR,
            "K-quant words, NVFP4, Q8_0 or E4M3 planes, or BF16 words",
        )),
        None => Err(GpuError::tensor(
            "head_out_w",
            HEAD_TENSOR,
            "resident — load Weights with globals",
        )),
    }
}

/// The Q6_K lm_head's word plane, for a chain that folds the projection
/// into its own launch ([`Head::enqueue_with_tail`]); another format is
/// refused by name.
fn head_q6k_w(w: &Weights) -> Result<&DeviceTensor<u32>, GpuError> {
    match head_out_w(w)?.0 {
        OutW::Q6K(w) => Ok(w),
        OutW::Q5K(_)
        | OutW::Q4K(_)
        | OutW::Q8_0 { .. }
        | OutW::Bf16(_)
        | OutW::E4m3(_)
        | OutW::Nvfp4 { .. } => Err(GpuError::tensor(
            "head_q6k_w",
            HEAD_TENSOR,
            "a Q6_K word plane: the tail this chain hands the head reads Q6_K rows",
        )),
    }
}

/// What stands between a head's input and its projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadNorm {
    /// The final RMS norm by `output_norm.weight`, then the projection.
    Rms,
    /// None: the input is the normed rows, which the body's own last launch
    /// wrote (a hyper-connection head's mix).
    Mixed,
}

/// A head's final norm with what it writes: the normed rows under
/// [`HeadNorm::Rms`], nothing under [`HeadNorm::Mixed`].
enum Final {
    Rms { normed: DeviceBuffer<f32> },
    Mixed,
}

/// Refuse the fault pair at the end of a head readback. The seeded (0, 0)
/// pair names a readback no argmax ever wrote — no `FaultSite` code is 0,
/// and a raise always ORs its bit into the mask, so no launch can store it —
/// and is a state error, not a fault; every other pair is
/// [`Fault::from_words`]'s answer, a raised pair [`GpuError::Fault`].
fn refuse_readback_fault(word: u32, sites: u32) -> Result<(), GpuError> {
    if word == 0 && sites == 0 {
        return Err(GpuError::state(
            "Head::tokens",
            "the head's readback was never written: no argmax launched for this row",
        ));
    }
    if let Some(fault) = Fault::from_words(word, sites) {
        return Err(GpuError::fault("Head::tokens", fault));
    }
    Ok(())
}

/// Enqueue the final norm `fin` names over `m` rows of `hidden` in `x`, its
/// gain the tensor `gain.1` of the weights `gain.0`, and return the rows the
/// projection reads: its normed rows, or `x` itself.
fn norm_rows<'a>(
    gpu: &Gpu,
    gain: (&Weights, &str),
    fin: &'a mut Final,
    (eps, hidden, m): (f32, usize, usize),
    x: &'a DeviceBuffer<f32>,
) -> Result<&'a DeviceBuffer<f32>, GpuError> {
    match fin {
        Final::Rms { normed } => {
            let g = crate::model::f32_gain(gain.0, gain.1)?;
            gpu.elem()
                .enqueue_rms_norm(gpu.stream(), x, g, eps, hidden, m, normed)?;
            Ok(normed)
        }
        Final::Mixed => Ok(x),
    }
}

/// The launches [`Head::enqueue`] records past the final norm, by the type of
/// the `output.weight` it projects: the q8_1 quantization of a K-quant
/// projection's input rows, the projection, a Q5_K projection's token-major
/// copy past one row, and the argmax. A body whose captured walk holds a head
/// counts it through [`HeadLaunches::at`], so the count follows the kernels
/// the head launches and not a copy of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadLaunches {
    quantizes: bool,
    copies_rows: bool,
}

impl HeadLaunches {
    /// The launches of the head over `out`'s `output.weight`.
    ///
    /// # Errors
    /// A weight no head entry reads, or none resident (as [`Head::with_gain`]).
    pub fn of(out: &Weights) -> Result<HeadLaunches, GpuError> {
        let (w, _) = head_out_w(out)?;
        Ok(HeadLaunches {
            quantizes: !w.reads_f32(),
            copies_rows: w.copies_rows(),
        })
    }

    /// The launches of a head of `m` rows past its final norm.
    #[must_use]
    pub fn at(self, m: usize) -> usize {
        usize::from(self.quantizes) + 1 + usize::from(self.copies_rows && m > 1) + 1
    }
}

/// The output head a plan reads on the host: `output_norm.weight` and `output.weight` of the
/// file, which the plan leaves where they lie ([`Head::host`]). One row: the decode step's.
pub struct HostHead {
    file: Arc<Split>,
    /// The shard `output.weight` lies in: the two tensors need not share one.
    shard: usize,
    out: TensorInfo,
    gain: Vec<f32>,
    eps: f32,
    n_vocab: usize,
    /// The projection's scratch and the row it was last made from. One lock per readback, taken
    /// by the one thread that reads the head.
    state: Mutex<HostState>,
}

/// What a [`HostHead`] reads into and keeps between readbacks, sized once at open.
struct HostState {
    /// The input row as copied from the card.
    x: Tensor2,
    /// The input row `normed` and `logits` were made from; they hold it only while `held`.
    from: Vec<f32>,
    held: bool,
    normed: Vec<f32>,
    logits: Vec<f32>,
}

impl HostHead {
    /// The head of `file` with the final norm's epsilon `eps`. Refused by name: either tensor
    /// missing from the file, `output.weight` of other than two dimensions, a vocabulary of no
    /// rows, and a row width other than the gain's.
    pub fn open(file: Arc<Split>, eps: f32) -> Result<HostHead, GpuError> {
        const WHAT: &str = "HostHead::open";
        let norm = output_norm();
        let (gain_shard, gain_info) = file
            .find(&norm)
            .ok_or_else(|| GpuError::tensor(WHAT, norm.clone(), "in the file"))?;
        let gain_file = file
            .shard(gain_shard)
            .ok_or_else(|| GpuError::shape(WHAT, format!("no shard {gain_shard} in the file")))?;
        let gain = f32_tensor(gain_file, gain_info)?;
        let (shard, out) = file
            .find(HEAD_TENSOR)
            .map(|(s, t)| (s, t.clone()))
            .ok_or_else(|| GpuError::tensor(WHAT, HEAD_TENSOR, "in the file"))?;
        let &[k, n_vocab] = out.dims.as_slice() else {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{HEAD_TENSOR} is {:?}, want a width and a vocabulary",
                    out.dims
                ),
            ));
        };
        if n_vocab == 0 || usize::try_from(k).ok() != Some(gain.len()) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{HEAD_TENSOR} is {k} values wide over {n_vocab} rows, the norm gain is {}",
                    gain.len()
                ),
            ));
        }
        let n_vocab = usize::try_from(n_vocab)
            .map_err(|_| GpuError::shape(WHAT, format!("{n_vocab} rows past usize")))?;
        let hidden = gain.len();
        Ok(HostHead {
            file,
            shard,
            out,
            gain,
            eps,
            n_vocab,
            state: Mutex::new(HostState {
                x: Tensor2 {
                    ne0: hidden,
                    ne1: 1,
                    data: vec![0.0; hidden],
                },
                from: vec![0.0; hidden],
                held: false,
                normed: vec![0.0; hidden],
                logits: vec![0.0; n_vocab],
            }),
        })
    }

    /// `read` over the normed row and the logits of the card's input row `x`. The row is copied
    /// down, and the projection (the final RMS norm by the gain, then `output.weight`'s own type,
    /// `model::ops::matmul_q`, which quantizes the row into the format that type reads) runs only
    /// when it is not the row the held result was made from, so a second readback of one step is
    /// a copy. Refused by name: a row of another width than the head's and the projection's own
    /// refusals.
    fn with_row<R>(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        read: impl FnOnce(&[f32], &[f32]) -> R,
    ) -> Result<R, GpuError> {
        const WHAT: &str = "Head::tokens";
        if x.len() != self.gain.len() {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a row of {} values, the head takes {}",
                    x.len(),
                    self.gain.len()
                ),
            ));
        }
        let mut guard = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let st = &mut *guard;
        x.copy_to_host(stream, &mut st.x.data)?;
        let same =
            st.x.data
                .iter()
                .zip(&st.from)
                .all(|(a, b)| a.to_bits() == b.to_bits());
        if !(st.held && same) {
            st.held = false;
            let normed = rms_norm(&st.x, &self.gain, self.eps);
            let gguf = self.file.shard(self.shard).ok_or_else(|| {
                GpuError::shape(WHAT, format!("no shard {} in the file", self.shard))
            })?;
            let logits = matmul_q(gguf, &self.out, &normed)?;
            if (logits.ne0, logits.ne1) != (self.n_vocab, 1) {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "the projection wrote {} x {}, the head's row is {} logits",
                        logits.ne0, logits.ne1, self.n_vocab
                    ),
                ));
            }
            st.normed.copy_from_slice(&normed.data);
            st.logits.copy_from_slice(&logits.data);
            st.from.copy_from_slice(&st.x.data);
            st.held = true;
        }
        Ok(read(&st.normed, &st.logits))
    }
}

/// The most rows one projection of a host head takes ([`Head::host_logits_rows`]).
pub const HOST_HEAD_ROWS: usize = 128;

impl HostHead {
    /// The logits of `rows` token-major rows at the start of the card's `x`, row-major (`rows`
    /// rows of `n_vocab`): the rows copied down, the final RMS norm by the gain on each, then one
    /// projection of every row by `output.weight`'s own type ([`matmul_q`], each column's dot the
    /// one-row call's), so the weight is read once for all of them. Leaves the held row of
    /// [`HostHead::with_row`] as it was. Refused by name: no rows, more than [`HOST_HEAD_ROWS`],
    /// a buffer shorter than the rows, and the projection's own refusals.
    fn logits_rows(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        rows: usize,
    ) -> Result<Vec<f32>, GpuError> {
        const WHAT: &str = "Head::host_logits_rows";
        let hidden = self.gain.len();
        if rows == 0 || rows > HOST_HEAD_ROWS || x.len() < rows * hidden {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{rows} rows of {hidden} from a buffer of {} values: 1 to {HOST_HEAD_ROWS} \
                     rows a projection",
                    x.len()
                ),
            ));
        }
        let mut data = vec![0.0; rows * hidden];
        let window = crate::tensor::Window::<f32>::of(x, 0, rows * hidden)?;
        window.copy_to_host(stream, &mut data)?;
        let x = Tensor2 {
            ne0: hidden,
            ne1: rows,
            data,
        };
        let normed = rms_norm(&x, &self.gain, self.eps);
        let gguf = self
            .file
            .shard(self.shard)
            .ok_or_else(|| GpuError::shape(WHAT, format!("no shard {} in the file", self.shard)))?;
        let mut logits = matmul_q(gguf, &self.out, &normed)?;
        if (logits.ne0, logits.ne1) != (self.n_vocab, rows) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the projection wrote {} x {}, the head's rows are {rows} of {} logits",
                    logits.ne0, logits.ne1, self.n_vocab
                ),
            ));
        }
        Ok(std::mem::take(&mut logits.data))
    }
}

/// The token of `logits`, one row: the first index of the largest value, the card argmax's rule.
/// A logit that is not finite is refused by name, as the card's argmax over f32 rows refuses it.
fn host_argmax(logits: &[f32]) -> Result<u32, GpuError> {
    const WHAT: &str = "Head::tokens";
    let mut best: Option<(usize, f32)> = None;
    for (i, &v) in logits.iter().enumerate() {
        if !v.is_finite() {
            return Err(GpuError::shape(
                WHAT,
                format!("logit {i} is {v}: the host head's argmax refuses a non-finite logit"),
            ));
        }
        if best.is_none_or(|(_, b)| v > b) {
            best = Some((i, v));
        }
    }
    let (i, _) = best.ok_or_else(|| GpuError::shape(WHAT, "a head of no logits"))?;
    u32::try_from(i).map_err(|_| GpuError::shape(WHAT, format!("token {i} past u32")))
}

/// The resident output head for `m` rows: input `m * hidden` f32 (row-major,
/// one row per token), the normed rows, their q8_1 quantization, `n_vocab * m`
/// logits (`logits[v*m + c]`, the gemv's layout) and `m` argmax tokens. Every
/// buffer is allocated at construction and `enqueue` touches addresses only,
/// so a graph that captures it serves every input written into `x`.
pub struct Head {
    fin: Final,
    /// The gain tensor's name under [`HeadNorm::Rms`]: [`output_norm`], or
    /// the one the body's [`Head::with_gain`] named.
    gain: String,
    eps: f32,
    hidden: usize,
    n_vocab: usize,
    m: usize,
    /// The chain's input — the last block's output vector. The assembly round
    /// writes here through `input_mut`; the gate through `set_input`.
    x: DeviceBuffer<f32>,
    /// The q8_1 form a K-quant projection reads; f32-activation heads hold none.
    act: Option<Q8Act>,
    logits: DeviceBuffer<f32>,
    /// The `m` argmax tokens, then the fault's first-layer word and that
    /// layer's site mask as the argmax found them. Seeded with the (0, 0)
    /// pair, which no raise can store: `tokens` refuses that pair as an
    /// unwritten readback rather than decode it as a fault.
    token_out: DeviceBuffer<u32>,
    /// `token_out`'s pinned twin and the event behind its last copy
    /// ([`Head::copy_tokens`]): the readback of a run with later work queued
    /// behind it.
    copied: PinnedHostBuffer<u32>,
    copied_at: CudaEvent,
    /// The ticket count of an f32-activation head of `m > 1` rows (its argmax's blocks
    /// each draw one), zero between launches; `None` for every other head.
    rows_done: Option<DeviceBuffer<u32>>,
    /// A Q5_K head's gemv: the `_sel` kernels and the one expert's id.
    q5k: Option<Q5kHead>,
    /// An NVFP4 head's kernels and forms.
    nvfp4: Option<Nvfp4Head>,
    /// The head of a plan that reads it on the host ([`Head::host`]): the card holds the input
    /// row alone, and the normed rows, logits and readback words here stay as allocated.
    host: Option<HostHead>,
}

/// The kernels a Q5_K head's projection launches: the `_sel` Q5_K gemv, which
/// reads the matrix as a stack of one expert (`one`: the expert's id, a slot
/// each of the `m` rows). Past one row it writes token-major (`cols[c ·
/// n_vocab + v]`), which `token_major` copies into the interleaved layout the
/// argmax reads (`logits[v · m + c]`).
struct Q5kHead {
    kernels: QuantSelKernels,
    one: DeviceBuffer<u32>,
    rows: Option<TokenRows>,
}

/// The kernels an NVFP4 head's projection launches (the site's: [`site::kgemv`],
/// [`site::nvfp4_sel32_gemv`]), the 32-value form of its normed rows at a width of no whole
/// super-blocks (the head's q8_1 form holds the others'), and past one row the logits' copy.
struct Nvfp4Head {
    kernels: KGemvKernels,
    act32: Option<Q8Blocks32>,
    rows: Option<TokenRows>,
}

/// A head of more than one row whose gemv writes token-major logits: the gemv's columns and the
/// copy into the interleaved layout the argmax reads.
struct TokenRows {
    cols: DeviceBuffer<f32>,
    token_major: TokenMajor,
}

impl Head {
    /// The decode head, `m = 1`: [`Head::with_m`] with one row.
    pub fn new(gpu: &Gpu, w: &Weights, eps: f32) -> Result<Head, GpuError> {
        Head::with_m(gpu, w, eps, 1)
    }

    /// Cross-check the two weights against each other and allocate the
    /// scratch for `m` rows (1..=8). Load-time only.
    pub fn with_m(gpu: &Gpu, w: &Weights, eps: f32, m: usize) -> Result<Head, GpuError> {
        Head::with_norm(gpu, w, eps, m, HeadNorm::Rms)
    }

    /// The decode head of a body whose final norm is `norm`: [`Head::new`]
    /// for [`HeadNorm::Rms`].
    pub fn of(gpu: &Gpu, w: &Weights, eps: f32, norm: HeadNorm) -> Result<Head, GpuError> {
        Head::with_norm(gpu, w, eps, 1, norm)
    }

    /// [`Head::with_m`] with the final norm `norm`: under
    /// [`HeadNorm::Mixed`] the width is the projection's rows' and no
    /// `output_norm` is read. Load-time only.
    pub fn with_norm(
        gpu: &Gpu,
        w: &Weights,
        eps: f32,
        m: usize,
        norm: HeadNorm,
    ) -> Result<Head, GpuError> {
        Head::with_gain(gpu, (w, &output_norm()), w, eps, m, norm)
    }

    /// [`Head::with_norm`] over two weight sets: the norm's gain is the tensor
    /// `gain.1` of `gain.0`, the projection `output.weight` of `out` — a draft
    /// layer's own norm in front of the target's matrix, which it reads where
    /// it sits ([`Head::enqueue_over`] takes the same two sets). Load-time
    /// only.
    pub fn with_gain(
        gpu: &Gpu,
        gain: (&Weights, &str),
        out: &Weights,
        eps: f32,
        m: usize,
        norm: HeadNorm,
    ) -> Result<Head, GpuError> {
        const WHAT: &str = "Head::with_gain";
        if !(1..=8).contains(&m) {
            return Err(GpuError::shape(
                WHAT,
                format!("need 1 <= m <= 8, got m={m}"),
            ));
        }
        let stream = gpu.stream();
        let (out_w, k) = head_out_w(out)?;
        let hidden = match norm {
            HeadNorm::Rms => crate::model::f32_gain(gain.0, gain.1)?.len(),
            HeadNorm::Mixed => k,
        };
        if k != hidden {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{HEAD_TENSOR} rows are {k} values wide, the norm \
                 gain is {hidden}"
                ),
            ));
        }
        let n_vocab = out_w.rows();
        if n_vocab == 0 {
            return Err(GpuError::shape(WHAT, format!("{HEAD_TENSOR} has no rows")));
        }
        let rows_done = if out_w.reads_f32() && m > 1 {
            Some(DeviceBuffer::zeroed(stream, 1)?)
        } else {
            None
        };
        let token_rows = || -> Result<Option<TokenRows>, GpuError> {
            if m == 1 {
                return Ok(None);
            }
            Ok(Some(TokenRows {
                cols: DeviceBuffer::zeroed(stream, m * n_vocab)?,
                token_major: TokenMajor::load(gpu.context())?,
            }))
        };
        let q5k = match out_w {
            OutW::Q5K(_) => Some(Q5kHead {
                kernels: QuantSelKernels::load(gpu.context(), gpu.fault_word())?,
                one: DeviceBuffer::zeroed(stream, m)?,
                rows: token_rows()?,
            }),
            _ => None,
        };
        let nvfp4 = match out_w {
            OutW::Nvfp4 { form, .. } => Some(Nvfp4Head {
                kernels: KGemvKernels::load(gpu)?,
                act32: (form == Form::Q8Blocks32)
                    .then(|| Q8Blocks32::new(stream, hidden, m))
                    .transpose()?,
                rows: token_rows()?,
            }),
            _ => None,
        };
        let x = DeviceBuffer::zeroed(stream, m * hidden)?;
        let fin = match norm {
            HeadNorm::Rms => Final::Rms {
                normed: DeviceBuffer::zeroed(stream, m * hidden)?,
            },
            HeadNorm::Mixed => Final::Mixed,
        };
        Ok(Head {
            fin,
            gain: gain.1.to_string(),
            eps,
            hidden,
            n_vocab,
            m,
            x,
            act: if out_w.reads_q8_1() {
                Some(Q8Act::with_k(stream, m, hidden)?)
            } else {
                None
            },
            logits: DeviceBuffer::zeroed(stream, m * n_vocab)?,
            token_out: DeviceBuffer::from_host(stream, &vec![0u32; m + 2])?,
            copied: PinnedHostBuffer::zeroed(stream.context(), m + 2)?,
            copied_at: stream.context().new_event(None)?,
            rows_done,
            q5k,
            nvfp4,
            host: None,
        })
    }

    /// The decode head, `m = 1`, of a plan that leaves the head on the host: the input row on
    /// the card, written by the chain's last launch, and everything past it `host`'s
    /// ([`Head::enqueue`] launches nothing; the readback runs the norm, the projection and the
    /// argmax on the host). Load-time only.
    pub fn host(gpu: &Gpu, host: HostHead) -> Result<Head, GpuError> {
        let stream = gpu.stream();
        let (hidden, n_vocab) = (host.gain.len(), host.n_vocab);
        // The host reads the norm and the projection into its own rows: the fields the card's head
        // reads stay one value each, and a caller that would read the normed rows on the card is
        // refused by name (`Head::reads_on_host`).
        Ok(Head {
            fin: Final::Rms {
                normed: DeviceBuffer::zeroed(stream, 1)?,
            },
            gain: output_norm(),
            eps: host.eps,
            hidden,
            n_vocab,
            m: 1,
            x: DeviceBuffer::zeroed(stream, hidden)?,
            act: None,
            logits: DeviceBuffer::zeroed(stream, 1)?,
            token_out: DeviceBuffer::from_host(stream, &[0u32; 3])?,
            copied: PinnedHostBuffer::zeroed(stream.context(), 3)?,
            copied_at: stream.context().new_event(None)?,
            rows_done: None,
            q5k: None,
            nvfp4: None,
            host: Some(host),
        })
    }

    /// The name of the gain tensor the final norm reads ([`output_norm`], or
    /// what [`Head::with_gain`] was given); a [`HeadNorm::Mixed`] head reads
    /// none.
    pub fn gain(&self) -> &str {
        &self.gain
    }

    /// The hidden width the head's norm and projection take.
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// The vocabulary size, the logit count the projection writes per row.
    pub fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    /// The rows this head was built for.
    pub fn m(&self) -> usize {
        self.m
    }

    /// Device bytes of the scratch (the weights are counted by their owner).
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        let normed = match &self.fin {
            Final::Rms { normed } => normed.num_bytes(),
            Final::Mixed => 0,
        };
        self.x.num_bytes()
            + normed
            + self.logits.num_bytes()
            + self.token_out.num_bytes()
            + self.rows_done.as_ref().map_or(0, DeviceBuffer::num_bytes)
            + self.q5k.as_ref().map_or(0, |q| {
                q.one.num_bytes() + q.rows.as_ref().map_or(0, |r| r.cols.num_bytes())
            })
            + self.nvfp4.as_ref().map_or(0, |n| {
                n.kernels.device_bytes()
                    + n.act32
                        .as_ref()
                        .map_or(0, |a| a.q.num_bytes() + a.s8.num_bytes() + a.d8.num_bytes())
                    + n.rows.as_ref().map_or(0, |r| r.cols.num_bytes())
            })
            + self.act.as_ref().map_or(0, Q8Act::device_bytes)
    }

    /// Enqueue normalization, projection and argmax for the rows in `x`.
    /// K-quant projections quantize the rows to q8_1; other formats read
    /// f32 rows. The argmax copies fault words after the tokens.
    /// Allocation-free and asynchronous, so capture records the same body.
    pub fn enqueue(&mut self, gpu: &Gpu, w: &Weights) -> Result<(), GpuError> {
        self.enqueue_over(gpu, w, w)
    }

    /// [`Head::enqueue`] with `epilogue` run over the logits between the
    /// projection and the argmax, for a chain whose logits take a function of
    /// their own (a softcap): the projection and the argmax are those of the
    /// file's `output.weight` type, and the argmax reads what `epilogue`
    /// leaves. The logits are `n_vocab · m` values as the projection writes
    /// them. Allocation-free and asynchronous when `epilogue` is, so capture
    /// records the same body.
    pub fn enqueue_with_epilogue<E>(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        mut epilogue: E,
    ) -> Result<(), GpuError>
    where
        E: FnMut(&mut DeviceBuffer<f32>) -> Result<(), GpuError>,
    {
        self.enqueue_over_with(gpu, w, w, &mut epilogue)
    }

    /// [`Head::enqueue`] with the norm's gain read from `gain` and the
    /// projection from `out`: the two sets [`Head::with_gain`] was made over.
    pub fn enqueue_over(
        &mut self,
        gpu: &Gpu,
        gain: &Weights,
        out: &Weights,
    ) -> Result<(), GpuError> {
        self.enqueue_over_with(gpu, gain, out, &mut |_: &mut DeviceBuffer<f32>| Ok(()))
    }

    /// The one body of the head's launches past the input row: the norm, the
    /// projection of `out`'s type, `epilogue` over the logits, the argmax.
    fn enqueue_over_with(
        &mut self,
        gpu: &Gpu,
        gain: &Weights,
        out: &Weights,
        epilogue: &mut dyn FnMut(&mut DeviceBuffer<f32>) -> Result<(), GpuError>,
    ) -> Result<(), GpuError> {
        if self.host.is_some() {
            // The host holds the norm's gain and the projection: the readback runs them.
            return Ok(());
        }
        let stream = gpu.stream();
        let shape = (self.eps, self.hidden, self.m);
        let rows = norm_rows(
            gpu,
            (gain, self.gain.as_str()),
            &mut self.fin,
            shape,
            &self.x,
        )?;
        let fault = gpu.fault_sink(LAYER_HEAD);
        // Whether the projection read f32 rows, whose argmax refuses a
        // non-finite logit.
        let finite_argmax = match head_out_w(out)?.0 {
            OutW::Q6K(out_w) => {
                let act = held(&mut self.act)?;
                gpu.enqueue_quantize_q8_1_head(rows, act)?;
                gpu.enqueue_gemv_q6k(out_w, &*act, &mut self.logits)?;
                false
            }
            OutW::Q5K(out_w) => {
                let act = held(&mut self.act)?;
                gpu.enqueue_quantize_q8_1_head(rows, act)?;
                let q = self.q5k.as_mut().ok_or_else(|| {
                    GpuError::state(
                        "Head::enqueue",
                        "a Q5_K head's kernels (made with the head)",
                    )
                })?;
                let down = SelDown {
                    w: out_w,
                    act: &*act,
                    sel: &q.one,
                    n_slots: self.m,
                    rows_per_expert: out_w.rows(),
                };
                match q.rows.as_mut() {
                    None => {
                        q.kernels
                            .enqueue_gemv_q5k_sel(stream, &down, fault, &mut self.logits)?
                    }
                    Some(r) => {
                        q.kernels
                            .enqueue_gemv_q5k_sel(stream, &down, fault, &mut r.cols)?;
                        // `logits[v · m + c] = cols[c · n_vocab + v]`: the
                        // copy of an `m`-row, `n_vocab`-column matrix with
                        // the two counts as `token_major`'s `rows` and `m`.
                        r.token_major.enqueue(
                            stream,
                            &r.cols,
                            self.m,
                            self.n_vocab,
                            &mut self.logits,
                        )?;
                    }
                }
                false
            }
            OutW::Q4K(out_w) => {
                let act = held(&mut self.act)?;
                gpu.enqueue_quantize_q8_1_head(rows, act)?;
                gpu.enqueue_gemv_q4k(out_w, &*act, &mut self.logits)?;
                false
            }
            OutW::Nvfp4 { form, .. } => {
                let q = self.nvfp4.as_mut().ok_or_else(|| {
                    GpuError::state(
                        "Head::enqueue",
                        "an NVFP4 head's kernels (made with the head)",
                    )
                })?;
                let key = (SiteTy::Nvfp4, out, HEAD_TENSOR);
                // Past one row the gemv's token-major columns land beside the logits and a copy
                // interleaves them.
                let y = match q.rows.as_mut() {
                    Some(r) => &mut r.cols,
                    None => &mut self.logits,
                };
                if form == Form::Q8Blocks32 {
                    let act = q.act32.as_mut().ok_or_else(|| {
                        GpuError::state(
                            "Head::enqueue",
                            "a Q8Blocks32 form (an NVFP4 head of a width past whole super-blocks)",
                        )
                    })?;
                    gpu.q5().enqueue_quantize_q8(stream, rows, act, fault)?;
                    site::nvfp4_sel32_gemv(gpu, &q.kernels, key, act, fault, y)?;
                } else {
                    let act = held(&mut self.act)?;
                    gpu.enqueue_quantize_q8_1_head(rows, act)?;
                    site::kgemv(gpu, &q.kernels, key, &*act, fault, y)?;
                }
                if let Some(r) = q.rows.as_mut() {
                    r.token_major.enqueue(
                        stream,
                        &r.cols,
                        self.m,
                        self.n_vocab,
                        &mut self.logits,
                    )?;
                }
                false
            }
            OutW::Q8_0 { qs, d } => {
                gpu.q8f32()
                    .enqueue_q8_0_gemv(stream, qs, d, rows, self.m, &mut self.logits)?;
                true
            }
            OutW::E4m3(w) => {
                gpu.q8f32().enqueue_e4m3_gemv(
                    stream,
                    E4m3GemvArgs {
                        w,
                        x: rows,
                        m: self.m,
                        out: GemvOut::RowMajor,
                        y: &mut self.logits,
                    },
                )?;
                true
            }
            OutW::Bf16(out_w) => {
                gpu.q8f32()
                    .enqueue_bf16_gemv(stream, out_w, rows, self.m, &mut self.logits)?;
                true
            }
        };
        epilogue(&mut self.logits)?;
        if finite_argmax {
            return self.argmax_finite(gpu, fault);
        }
        if self.m == 1 {
            gpu.elem().enqueue_argmax_fault(
                stream,
                &self.logits,
                self.n_vocab,
                fault,
                &mut self.token_out,
            )?;
        } else {
            gpu.elem().enqueue_argmax_rows_fault(
                stream,
                &self.logits,
                self.n_vocab,
                self.m,
                fault,
                &mut self.token_out,
            )?;
        }
        Ok(())
    }

    /// The argmax after a gemv over f32 rows, which refuses
    /// a non-finite logit: `argmax_finite_fault`, or `argmax_rows_finite_fault`
    /// past one row, the fault word copied after the tokens.
    fn argmax_finite(&mut self, gpu: &Gpu, fault: FaultSink) -> Result<(), GpuError> {
        let stream = gpu.stream();
        match &mut self.rows_done {
            None => gpu.elem().enqueue_argmax_finite_fault(
                stream,
                &self.logits,
                self.n_vocab,
                fault,
                &mut self.token_out,
            ),
            Some(done) => gpu.elem().enqueue_argmax_rows_finite_fault(
                stream,
                &self.logits,
                (self.n_vocab, self.m),
                fault,
                &mut self.token_out,
                done,
            ),
        }
    }

    /// Enqueue the head with its projection and argmax handed to `tail`, for
    /// an architecture whose chain folds the two into one launch: the same
    /// rms_norm and q8_1 quantization [`Head::enqueue`] runs, then `tail(act,
    /// out_w, fault, logits, token_out)` over the quantized rows, the Q6_K
    /// lm_head, the head's fault sink and the two outputs the readbacks read
    /// (`logits` as `logits_to_host` takes them, `token_out` as the `m`
    /// tokens then the fault's two words). `tail` must write both as
    /// `enqueue`'s gemv and argmax do; [`Head::token`] and
    /// [`Head::logits_to_host`] are unchanged.
    /// Pure enqueues, so the same body is what a capture records. Public
    /// because an architecture crate's chain writes its own tail (a fused
    /// projection the shared one holds no launch of); a logit epilogue over
    /// the head's own projection is [`Head::enqueue_with_epilogue`]'s.
    pub fn enqueue_with_tail<F>(&mut self, gpu: &Gpu, w: &Weights, tail: F) -> Result<(), GpuError>
    where
        F: FnOnce(
            &Q8Act,
            &DeviceTensor<u32>,
            FaultSink,
            &mut DeviceBuffer<f32>,
            &mut DeviceBuffer<u32>,
        ) -> Result<(), GpuError>,
    {
        let shape = (self.eps, self.hidden, self.m);
        let rows = norm_rows(gpu, (w, self.gain.as_str()), &mut self.fin, shape, &self.x)?;
        let act = held(&mut self.act)?;
        gpu.enqueue_quantize_q8_1_head(rows, act)?;
        tail(
            &*act,
            head_q6k_w(w)?,
            gpu.fault_sink(LAYER_HEAD),
            &mut self.logits,
            &mut self.token_out,
        )
    }

    /// Write the chain's input from the host. Synchronizing copy on the
    /// engine stream; never inside a capture.
    pub fn set_input(&mut self, gpu: &Gpu, host: &[f32]) -> Result<(), GpuError> {
        if host.len() != self.m * self.hidden {
            return Err(GpuError::shape(
                "Head::set_input",
                format!(
                    "{} values, the head takes {} rows of {}",
                    host.len(),
                    self.m,
                    self.hidden
                ),
            ));
        }
        self.x.copy_from_host(gpu.stream(), host)?;
        Ok(())
    }

    /// The chain's input buffer — the write target of the step's last launch
    /// before the head, so that launch's output lands directly in the head.
    /// Public because an architecture crate's chain writes it too.
    pub fn input_mut(&mut self) -> &mut DeviceBuffer<f32> {
        &mut self.x
    }

    /// The chain's input row as it stands (the running residual of the
    /// layers a chain has run), the read side of [`Head::input_mut`].
    pub fn input(&self) -> &DeviceBuffer<f32> {
        &self.x
    }

    /// The normed rows of the last run (the dump's `result_norm` tap; under
    /// [`HeadNorm::Mixed`] the input, `result_embd`). Blocking read;
    /// gate/debug use.
    pub fn normed_to_host(&self, gpu: &Gpu) -> Result<Vec<f32>, GpuError> {
        if let Some(h) = &self.host {
            return self.host_row(gpu, h, |normed, _| normed.to_vec());
        }
        Ok(self.normed().to_host_vec(gpu.stream())?)
    }

    /// `read` over the normed row and the logits of the input row as it stands, a host head's
    /// readback: the engine stream waited for, a raised fault refused, the row copied and
    /// projected on the host unless the held projection is of it.
    fn host_row<R>(
        &self,
        gpu: &Gpu,
        h: &HostHead,
        read: impl FnOnce(&[f32], &[f32]) -> R,
    ) -> Result<R, GpuError> {
        const WHAT: &str = "Head::tokens";
        let stream = gpu.stream();
        crate::host::await_done(WHAT, || stream.query())?;
        if let Some(fault) = gpu.fault()? {
            return Err(GpuError::fault(WHAT, fault));
        }
        h.with_row(stream, &self.x, read)
    }

    /// The logits of `rows` token-major rows at the start of `x` (a pass's final mix), row-major,
    /// projected on the host at once, when the head is the host's ([`HOST_HEAD_ROWS`] rows at
    /// most); `None` for a card head, whose rows a caller runs through [`Head::enqueue`] one at a
    /// time. Waits for the stream and refuses a raised fault first, as the host head's readback
    /// does.
    pub fn host_logits_rows(
        &self,
        gpu: &Gpu,
        x: &DeviceBuffer<f32>,
        rows: usize,
    ) -> Result<Option<Vec<f32>>, GpuError> {
        const WHAT: &str = "Head::host_logits_rows";
        let Some(h) = &self.host else {
            return Ok(None);
        };
        let stream = gpu.stream();
        crate::host::await_done(WHAT, || stream.query())?;
        if let Some(fault) = gpu.fault()? {
            return Err(GpuError::fault(WHAT, fault));
        }
        h.logits_rows(stream, x, rows).map(Some)
    }

    /// The head is read on the host: its normed rows and logits are not on the card.
    pub(crate) fn reads_on_host(&self) -> bool {
        self.host.is_some()
    }

    /// [`Head::normed_to_host`]'s rows on the card, for a launch on the
    /// engine stream that reads them where they sit.
    pub(crate) fn normed(&self) -> &DeviceBuffer<f32> {
        match &self.fin {
            Final::Rms { normed } => normed,
            Final::Mixed => &self.x,
        }
    }

    /// The logits of the last run (`n_vocab * m` f32, `[v*m + c]`). Blocking
    /// read; gate/debug use.
    pub fn logits_to_host(&self, gpu: &Gpu) -> Result<Vec<f32>, GpuError> {
        if let Some(h) = &self.host {
            return self.host_row(gpu, h, |_, logits| logits.to_vec());
        }
        Ok(self.logits.to_host_vec(gpu.stream())?)
    }

    /// The logits of the last run on the card (`n_vocab * m` f32, `[v*m + c]`).
    pub(crate) fn logits(&self) -> &DeviceBuffer<f32> {
        &self.logits
    }

    /// [`Head::logits_to_host`] into `out`, which holds exactly `n_vocab * m`
    /// f32: a caller reading every token reuses one buffer. Blocking read.
    pub fn logits_into_host(&self, gpu: &Gpu, out: &mut [f32]) -> Result<(), GpuError> {
        if let Some(h) = &self.host {
            return self.host_row(gpu, h, |_, logits| {
                if out.len() != logits.len() {
                    return Err(GpuError::shape(
                        "Head::logits_into_host",
                        format!("a buffer of {} f32 for {} logits", out.len(), logits.len()),
                    ));
                }
                out.copy_from_slice(logits);
                Ok(())
            })?;
        }
        if out.len() != self.logits.len() {
            return Err(GpuError::shape(
                "Head::logits_into_host",
                format!(
                    "a buffer of {} f32 for {} logits",
                    out.len(),
                    self.logits.len()
                ),
            ));
        }
        self.logits.copy_to_host(gpu.stream(), out)?;
        Ok(())
    }

    /// The argmax token of the last run's first row: a blocking read — the
    /// decode step's single synchronize-shaped retrieval. A fault raised by
    /// any launch before the argmax is [`GpuError::Fault`], not a token.
    pub fn token(&self, gpu: &Gpu) -> Result<u32, GpuError> {
        Ok(self.tokens(gpu)?[0])
    }

    /// The readback words on the card: the `m` argmax tokens, then the fault
    /// words ([`Head::tokens`]). A step chained behind a run reads its token
    /// from word 0; a round of several runs copies them on the stream into one
    /// buffer of that layout and reads it back once ([`read_tokens`]).
    #[must_use]
    pub fn token_words(&self) -> &DeviceBuffer<u32> {
        &self.token_out
    }

    /// Enqueue the copy of the last run's readback words to the host on the
    /// card's stream, for [`Head::copied_token`]: the readback of a run with
    /// later work queued behind it. Never inside a capture.
    pub fn copy_tokens(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        if self.host.is_some() {
            return Err(GpuError::state(
                "Head::copy_tokens",
                "readback words on the card (this head reads on the host)",
            ));
        }
        let stream = gpu.stream();
        // SAFETY: `copied` is this head's pinned twin of `token_out`, as long
        // as it; the host reads it only once `copied_at`, recorded right
        // below, has passed ([`Head::copied_token`]), and the head's drop
        // waits for that event before freeing it.
        unsafe {
            self.token_out
                .copy_to_pinned_host_async(stream, &mut self.copied)?
        };
        self.copied_at.record(stream)?;
        Ok(())
    }

    /// The first row's argmax [`Head::copy_tokens`] copied, once the copy
    /// lands (within [`crate::host::ENGINE_BOUND`], past it
    /// [`GpuError::Stalled`]); a raised fault or a readback no argmax wrote
    /// is refused as [`Head::tokens`] refuses it.
    pub fn copied_token(&self) -> Result<u32, GpuError> {
        crate::host::await_done("Head::copied_token", || self.copied_at.query())?;
        let words = self.copied.as_slice();
        let (Some(&token), [.., word, sites]) = (words.first(), words) else {
            return Err(GpuError::state(
                "Head::copied_token",
                "the readback copy holds no fault words",
            ));
        };
        refuse_readback_fault(*word, *sites)?;
        Ok(token)
    }

    /// The argmax tokens of the last run, one per row ([`read_tokens`] of
    /// [`Head::token_words`]).
    pub fn tokens(&self, gpu: &Gpu) -> Result<Vec<u32>, GpuError> {
        match &self.host {
            Some(h) => Ok(vec![
                self.host_row(gpu, h, |_, logits| host_argmax(logits))??,
            ]),
            None => read_tokens(gpu, &self.token_out),
        }
    }
}

/// The tokens of `words`, a buffer laid out as a head's argmax words
/// ([`Head::token_words`]): tokens, then the fault word and its site mask.
/// Blocking read: the engine stream waited for within
/// [`crate::host::ENGINE_BOUND`] ([`crate::host::await_done`], past it
/// [`GpuError::Stalled`]), then the words copied back; a raised fault is
/// [`GpuError::Fault`], and a readback no argmax ever wrote a state error,
/// as [`Head::token`].
pub fn read_tokens(gpu: &Gpu, words: &DeviceBuffer<u32>) -> Result<Vec<u32>, GpuError> {
    let stream = gpu.stream();
    crate::host::await_done("Head::tokens", || stream.query())?;
    let mut out = words.to_host_vec(stream)?;
    let (Some(sites), Some(word)) = (out.pop(), out.pop()) else {
        return Err(GpuError::state(
            "Head::tokens",
            "the readback buffer holds no fault words",
        ));
    };
    refuse_readback_fault(word, sites)?;
    Ok(out)
}

impl Drop for Head {
    fn drop(&mut self) {
        // The pinned twin outlives a copy still writing it. A failure on the
        // drop path is unreportable and ignored.
        let _ = self.copied_at.synchronize();
    }
}

/// One rank's pick over its part of the vocabulary: `id`, the card's argmax
/// of `logits` (one row), with its value, as `(value, id)` with `id` counted
/// from `vocab_at`, the part's first id in the whole vocabulary. An id past
/// the row, or past `u32` from `vocab_at`, is refused by name.
pub fn local_pick(logits: &[f32], id: u32, vocab_at: u32) -> Result<(f32, u32), GpuError> {
    const WHAT: &str = "head::local_pick";
    let v = *logits.get(id as usize).ok_or_else(|| {
        GpuError::shape(WHAT, format!("pick {id} past the row's {}", logits.len()))
    })?;
    let at = vocab_at
        .checked_add(id)
        .ok_or_else(|| GpuError::shape(WHAT, format!("pick {id} from {vocab_at} past u32")))?;
    Ok((v, at))
}

/// The step's token from every rank's pick `(value, id)`, `id` in the whole
/// vocabulary: the largest value, and of equal values the lowest id, which is
/// the one-card argmax's rule (the first index of the largest value) over the
/// ranks' parts in vocabulary order. No pick, and a value that is not finite
/// (a part's argmax refuses one), are refused by name.
pub fn merge(picks: &[(f32, u32)]) -> Result<u32, GpuError> {
    const WHAT: &str = "head::merge";
    let mut best: Option<(f32, u32)> = None;
    for &(v, id) in picks {
        if !v.is_finite() {
            return Err(GpuError::shape(WHAT, format!("a pick of {v} at id {id}")));
        }
        best = match best {
            Some((bv, bid)) if bv > v || (bv == v && bid < id) => Some((bv, bid)),
            _ => Some((v, id)),
        };
    }
    best.map(|(_, id)| id)
        .ok_or_else(|| GpuError::shape(WHAT, "no rank's pick"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fault::{FAULT_NONE, FaultSite};
    use gguf::Split;
    use gguf::quant::{GgmlType, dequant_row};
    use gguf::write::{Layout, TensorDecl, Writer};

    /// Bytes of one Q5_K super-block.
    const Q5K_BLOCK: usize = 176;

    /// Next value of a xorshift64 stream.
    fn next(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// `rows` rows of `k` values of Q5_K as the file's block bytes: random
    /// codes, and each block's `d` and `dmin` a small finite f16, so no dot is
    /// NaN or infinite.
    fn q5k_rows(rows: usize, k: usize) -> Vec<u8> {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut out = vec![0u8; rows * (k / 256) * Q5K_BLOCK];
        for block in out.as_chunks_mut::<Q5K_BLOCK>().0 {
            for chunk in block.chunks_mut(8) {
                let v = next(&mut state).to_le_bytes();
                chunk.copy_from_slice(&v[..chunk.len()]);
            }
            for scale in block[..4].chunks_mut(2) {
                let (exp, mant) = (5 + next(&mut state) % 4, next(&mut state) & 0x3ff);
                let bits = u16::try_from(exp << 10 | mant).expect("an f16");
                scale.copy_from_slice(&bits.to_le_bytes());
            }
        }
        out
    }

    /// A Q5_K `output.weight` is read by the head as the file's rows: each
    /// logit is the host's dot of the dequantized row with the normed input,
    /// within the q8_1 rounding of that input, and the argmax is the card's
    /// own logits' maximum.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_a_q5k_head_projects_the_rows_it_reads() {
        const K: usize = 512;
        const N_VOCAB: usize = 600;
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let rows = q5k_rows(N_VOCAB, K);
        let gain: Vec<u8> = (0..K).flat_map(|_| 1.0f32.to_le_bytes()).collect();
        let path = write_file(
            "q5k-head",
            &[
                (
                    HEAD_TENSOR,
                    vec![K as u64, N_VOCAB as u64],
                    GgmlType::Q5_K,
                    &rows,
                ),
                (&output_norm(), vec![K as u64], GgmlType::F32, &gain),
            ],
        );
        let dir = path.parent().expect("the file's directory").to_path_buf();
        let split = Split::open(&path).expect("the file opens");
        let w = Weights::load_where(gpu.stream(), &split, |_| true).expect("resident");
        let _ = std::fs::remove_dir_all(&dir);

        let mut head = Head::new(&gpu, &w, 1e-6).expect("a Q5_K head");
        assert_eq!((head.hidden(), head.n_vocab()), (K, N_VOCAB));
        let x: Vec<f32> = (0..K)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
            .collect();
        head.set_input(&gpu, &x).expect("the input");
        head.enqueue(&gpu, &w).expect("the head");
        let normed = head.normed_to_host(&gpu).expect("the normed row");
        let logits = head.logits_to_host(&gpu).expect("the logits");
        assert_eq!(logits.len(), N_VOCAB);
        let row_bytes = K / 256 * Q5K_BLOCK;
        let mut wv = vec![0.0f32; K];
        for (r, &got) in logits.iter().enumerate() {
            dequant_row(
                GgmlType::Q5_K,
                &rows[r * row_bytes..(r + 1) * row_bytes],
                &mut wv,
            )
            .expect("a Q5_K row");
            let terms = wv
                .iter()
                .zip(&normed)
                .map(|(&a, &b)| f64::from(a) * f64::from(b));
            let want: f64 = terms.clone().sum();
            let scale = terms.map(|t| t * t).sum::<f64>().sqrt();
            assert!(
                (f64::from(got) - want).abs() <= 0.03 * scale + 1e-6,
                "row {r}: the card's {got}, the host's {want} (the row's own scale {scale})"
            );
        }
        let best = (0..N_VOCAB)
            .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
            .expect("a vocabulary");
        assert_eq!(head.token(&gpu).expect("the token") as usize, best);

        // A head of three rows: row `c`'s logits and token are the one-row
        // head's on that row, bit for bit (the `_sel` gemv is token-major and
        // the copy puts it in the argmax's layout).
        const M: usize = 3;
        let inputs: Vec<Vec<f32>> = (0..M)
            .map(|c| {
                (0..K)
                    .map(|i| (((i + 7 * c) * 37 % 101) as f32 - 50.0 + c as f32) / 25.0)
                    .collect()
            })
            .collect();
        let mut multi = Head::with_m(&gpu, &w, 1e-6, M).expect("a Q5_K head of three rows");
        multi
            .set_input(&gpu, &inputs.concat())
            .expect("the three rows");
        multi.enqueue(&gpu, &w).expect("the head of three rows");
        let got = multi.logits_to_host(&gpu).expect("the logits");
        let tokens = multi.tokens(&gpu).expect("the tokens");
        assert_eq!((got.len(), tokens.len()), (M * N_VOCAB, M));
        for (c, input) in inputs.iter().enumerate() {
            head.set_input(&gpu, input).expect("the input");
            head.enqueue(&gpu, &w).expect("the head");
            let want = head.logits_to_host(&gpu).expect("the logits");
            for (v, &w1) in want.iter().enumerate() {
                assert_eq!(
                    got[v * M + c].to_bits(),
                    w1.to_bits(),
                    "row {c}, logit {v}: the head of three rows, the one-row head's"
                );
            }
            assert_eq!(tokens[c], head.token(&gpu).expect("the token"), "row {c}");
        }
    }

    /// The readback's seeded (0, 0) pair is an unwritten readback, not a
    /// fault: the error names the state, with no device fault in it.
    #[test]
    fn an_unwritten_readback_pair_is_a_state_error_not_a_fault() {
        let err = refuse_readback_fault(0, 0).unwrap_err();
        assert!(
            matches!(err, GpuError::State { .. }),
            "want a state error for the unwritten pair, got {err:?}"
        );
        assert_eq!(
            err.to_string(),
            "Head::tokens: the head's readback was never written: no argmax launched for this row"
        );
    }

    /// A raised pair still refuses as the fault it names, and a clean word
    /// passes.
    #[test]
    fn a_raised_readback_pair_still_refuses_as_its_fault() {
        let fault = Fault::at(3, FaultSite::CachePos);
        match refuse_readback_fault(fault.word(), fault.sites) {
            Err(GpuError::Fault { fault: raised, .. }) => assert_eq!(raised, fault),
            other => panic!("want the raised fault, got {other:?}"),
        }
        assert!(refuse_readback_fault(FAULT_NONE, 0).is_ok());
    }

    /// Bytes of one Q6_K super-block.
    const Q6K_BLOCK: usize = 210;

    /// `rows` rows of `k` values of Q6_K as the file's block bytes: random codes and scales, and
    /// each block's `d` (its last two bytes) a small finite f16, so no dot is NaN or infinite.
    fn q6k_rows(rows: usize, k: usize) -> Vec<u8> {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut out = vec![0u8; rows * (k / 256) * Q6K_BLOCK];
        for block in out.as_chunks_mut::<Q6K_BLOCK>().0 {
            for chunk in block[..Q6K_BLOCK - 2].chunks_mut(8) {
                let v = next(&mut state).to_le_bytes();
                chunk.copy_from_slice(&v[..chunk.len()]);
            }
            let (exp, mant) = (5 + next(&mut state) % 4, next(&mut state) & 0x3ff);
            let bits = u16::try_from(exp << 10 | mant).expect("an f16");
            block[Q6K_BLOCK - 2..].copy_from_slice(&bits.to_le_bytes());
        }
        out
    }

    /// A GGUF file of `tensors` (name, dims, type, bytes) in a fresh temp directory: the file's
    /// path, whose parent the caller removes.
    fn write_file(tag: &str, tensors: &[(&str, Vec<u64>, GgmlType, &[u8])]) -> std::path::PathBuf {
        let decls: Vec<TensorDecl> = tensors
            .iter()
            .map(|(name, dims, ty, bytes)| TensorDecl {
                name: (*name).to_string(),
                dims: dims.clone(),
                type_id: ty.as_u32(),
                nbytes: bytes.len() as u64,
            })
            .collect();
        let dir = std::env::temp_dir().join(format!("bloomery-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("a temp dir");
        let path = dir.join("head.gguf");
        let layout = Layout::new(&[], decls.clone()).expect("a layout");
        let file = std::io::BufWriter::new(std::fs::File::create(&path).expect("the file"));
        let mut writer = Writer::new(file, layout).expect("a writer");
        for (decl, (_, _, _, bytes)) in decls.iter().zip(tensors) {
            writer.tensor(&decl.name, bytes).expect("a tensor");
        }
        writer.finish().expect("a finished file");
        path
    }

    /// A host head reads the rows it is handed: the normed row is the input scaled by the gain
    /// and the rms (the gain applied), each logit is the dot of the dequantized Q6_K row with
    /// that row within the q8 rounding of the activation, the token is the first largest logit,
    /// and the card holds no readback words to copy.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_a_host_head_projects_the_rows_it_reads() {
        const K: usize = 512;
        const N_VOCAB: usize = 600;
        const EPS: f32 = 1e-6;
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let rows = q6k_rows(N_VOCAB, K);
        let gain: Vec<f32> = (0..K).map(|i| 0.5 + (i % 7) as f32 / 10.0).collect();
        let gain_bytes: Vec<u8> = gain.iter().flat_map(|g| g.to_le_bytes()).collect();
        let path = write_file(
            "host-head",
            &[
                (
                    HEAD_TENSOR,
                    vec![K as u64, N_VOCAB as u64],
                    GgmlType::Q6_K,
                    &rows,
                ),
                (
                    "output_norm.weight",
                    vec![K as u64],
                    GgmlType::F32,
                    &gain_bytes,
                ),
            ],
        );
        let split = Arc::new(Split::open(&path).expect("the file opens"));
        let w = Weights::load_where(gpu.stream(), &split, |n| n == output_norm())
            .expect("the gain resident");
        let host = HostHead::open(Arc::clone(&split), EPS).expect("a host head");
        let mut head = Head::host(&gpu, host).expect("a host head");
        assert_eq!(
            (head.hidden(), head.n_vocab()),
            (K, N_VOCAB),
            "the host head takes its width from the gain and its vocabulary from the rows"
        );
        let x: Vec<f32> = (0..K)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
            .collect();
        head.set_input(&gpu, &x).expect("the input");
        head.enqueue(&gpu, &w).expect("the head launches nothing");

        let normed = head.normed_to_host(&gpu).expect("the normed row");
        let mean_sq = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / K as f64;
        let inv = 1.0 / (mean_sq + f64::from(EPS)).sqrt();
        for i in 0..K {
            let want = f64::from(x[i]) * f64::from(gain[i]) * inv;
            assert!(
                (f64::from(normed[i]) - want).abs() <= 1e-5 * want.abs().max(1.0),
                "normed[{i}] is {}, the gain-scaled row is {want}",
                normed[i]
            );
        }
        let logits = head.logits_to_host(&gpu).expect("the logits");
        assert_eq!(
            logits.len(),
            N_VOCAB,
            "one logit a row of the file's output tensor"
        );
        let row_bytes = K / 256 * Q6K_BLOCK;
        let mut wv = vec![0.0f32; K];
        for (r, &got) in logits.iter().enumerate() {
            dequant_row(
                GgmlType::Q6_K,
                &rows[r * row_bytes..(r + 1) * row_bytes],
                &mut wv,
            )
            .expect("a Q6_K row");
            let terms = wv
                .iter()
                .zip(&normed)
                .map(|(&a, &b)| f64::from(a) * f64::from(b));
            let want: f64 = terms.clone().sum();
            let scale = terms.map(|t| t * t).sum::<f64>().sqrt();
            assert!(
                (f64::from(got) - want).abs() <= 0.03 * scale + 1e-6,
                "row {r}: the host's {got}, the dequantized dot {want} (the row's own scale {scale})"
            );
        }
        let best = (0..N_VOCAB).fold(0, |b, i| if logits[i] > logits[b] { i } else { b });
        assert_eq!(
            head.token(&gpu).expect("the token") as usize,
            best,
            "the token is the first largest logit of the row the host read back"
        );
        assert_eq!(
            head.tokens(&gpu).expect("the tokens"),
            vec![best as u32],
            "a one-row head has one token, the same pick"
        );
        assert!(
            head.copy_tokens(&gpu).is_err(),
            "a host head has no readback words on the card"
        );
        let _ = std::fs::remove_dir_all(path.parent().expect("the temp dir"));
    }

    /// The host head's argmax is the first index of the largest logit, ties included, and a
    /// logit that is not finite is refused by name rather than ranked.
    #[test]
    fn the_host_argmax_is_the_first_largest_and_refuses_a_non_finite_logit() {
        assert_eq!(
            host_argmax(&[1.0, 3.0, 2.0, 3.0, -1.0]).unwrap(),
            1,
            "the first of two equal maxima wins"
        );
        assert_eq!(
            host_argmax(&[-4.0, -2.0, -3.0, -2.0]).unwrap(),
            1,
            "an all-negative row ranks by value, ties to the first"
        );
        assert_eq!(
            host_argmax(&[0.0; 5]).unwrap(),
            0,
            "a flat row is a tie of every index, the first wins"
        );
        assert_eq!(
            host_argmax(&[7.0]).unwrap(),
            0,
            "a one-logit row has one index to name"
        );
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let err = host_argmax(&[1.0, bad, 2.0]).unwrap_err().to_string();
            assert!(
                err.contains("logit 1") && err.contains("non-finite"),
                "{bad} must be refused naming its index and that it is not finite: {err}"
            );
        }
        assert!(
            host_argmax(&[]).is_err(),
            "an empty row has no largest logit to name"
        );
    }

    /// A host head of a file that lacks a tensor or whose tensors disagree on the width is
    /// refused by name at the open, not at the first token.
    #[test]
    fn a_host_head_of_a_file_without_its_tensors_is_refused_by_name() {
        const K: usize = 256;
        let rows = q6k_rows(8, K);
        let gain = vec![0u8; 4 * K];
        let short = vec![0u8; 4 * (K / 2)];
        let head = |k: u64| (HEAD_TENSOR, vec![k, 8], GgmlType::Q6_K, rows.as_slice());
        let open = |tag: &str, tensors: &[(&str, Vec<u64>, GgmlType, &[u8])]| {
            let path = write_file(tag, tensors);
            let split = Arc::new(Split::open(&path).expect("the file opens"));
            let err = HostHead::open(split, 1e-6).err().map(|e| e.to_string());
            let _ = std::fs::remove_dir_all(path.parent().expect("the temp dir"));
            err
        };
        let no_gain = open("host-head-no-gain", &[head(K as u64)]).expect("a refusal");
        assert!(
            no_gain.contains("output_norm.weight") && no_gain.contains("in the file"),
            "a file without the gain is refused naming it and where it was looked for: {no_gain}"
        );
        let no_out = open(
            "host-head-no-out",
            &[("output_norm.weight", vec![K as u64], GgmlType::F32, &gain)],
        )
        .expect("a refusal");
        assert!(
            no_out.contains(HEAD_TENSOR),
            "a file without the output rows is refused naming them: {no_out}"
        );
        let wide = open(
            "host-head-wide",
            &[
                head(K as u64),
                (
                    "output_norm.weight",
                    vec![(K / 2) as u64],
                    GgmlType::F32,
                    &short,
                ),
            ],
        )
        .expect("a refusal");
        assert!(
            wide.contains(&format!("{K} values wide"))
                && wide.contains(&format!("gain is {}", K / 2)),
            "a gain of another width than the rows is refused naming both widths: {wide}"
        );
        let whole = open(
            "host-head-whole",
            &[
                head(K as u64),
                ("output_norm.weight", vec![K as u64], GgmlType::F32, &gain),
            ],
        );
        assert!(whole.is_none(), "a whole head opens: {whole:?}");
    }

    /// The merge of every split of a row into N = 1, 2 and 4 parts, each part's pick its
    /// [`super::local_pick`], is the row's first index of its largest value, ties across parts and
    /// within one included, whatever order the picks come in; no pick and a pick that is not
    /// finite are refused.
    #[test]
    fn the_merge_of_local_picks_is_the_whole_rows_argmax() {
        use super::{local_pick, merge};
        let rows: [&[f32]; 4] = [
            &[1.0, 3.0, 2.0, 3.0, 0.5, -1.0, 3.0, 2.5],
            &[-4.0, -2.0, -3.0, -2.0, -9.0, -2.5, -7.0, -8.0],
            &[0.0; 8],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0],
        ];
        // The one-card argmax: the first index of the largest value.
        let argmax = |row: &[f32]| {
            row.iter()
                .enumerate()
                .fold(0, |b, (i, &x)| if x > row[b] { i } else { b }) as u32
        };
        for row in rows {
            let want = argmax(row);
            for n in [1usize, 2, 4] {
                let part = row.len() / n;
                let mut picks: Vec<(f32, u32)> = row
                    .chunks(part)
                    .enumerate()
                    .map(|(r, c)| local_pick(c, argmax(c), (r * part) as u32).unwrap())
                    .collect();
                assert_eq!(merge(&picks).unwrap(), want, "{row:?} in {n} parts");
                picks.reverse();
                assert_eq!(
                    merge(&picks).unwrap(),
                    want,
                    "{row:?} in {n} parts, reversed"
                );
            }
        }
        assert!(local_pick(&[], 0, 0).is_err() && local_pick(&[1.0], 1, 0).is_err());
        assert!(
            local_pick(&[1.0], 0, u32::MAX).is_ok()
                && local_pick(&[1.0, 2.0], 1, u32::MAX).is_err()
        );
        assert!(merge(&[]).is_err());
        assert!(merge(&[(1.0, 0), (f32::NAN, 1)]).is_err());
        assert!(merge(&[(f32::INFINITY, 0)]).is_err());
    }
}
