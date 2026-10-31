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
use cuda_core::DeviceBuffer;
use model::arch::names::output_norm;
use model::placement::kernels::{self, HeadEntry};

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
    /// The ticket count of an f32-activation head of `m > 1` rows (its argmax's blocks
    /// each draw one), zero between launches; `None` for every other head.
    rows_done: Option<DeviceBuffer<u32>>,
    /// A Q5_K head's gemv: the `_sel` kernels and the one expert's id.
    q5k: Option<Q5kHead>,
    /// An NVFP4 head's kernels and forms.
    nvfp4: Option<Nvfp4Head>,
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
            rows_done,
            q5k,
            nvfp4,
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

    /// [`Head::enqueue`] with the norm's gain read from `gain` and the
    /// projection from `out`: the two sets [`Head::with_gain`] was made over.
    pub fn enqueue_over(
        &mut self,
        gpu: &Gpu,
        gain: &Weights,
        out: &Weights,
    ) -> Result<(), GpuError> {
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
        match head_out_w(out)?.0 {
            OutW::Q6K(out_w) => {
                let act = held(&mut self.act)?;
                gpu.enqueue_quantize_q8_1_head(rows, act)?;
                gpu.enqueue_gemv_q6k(out_w, &*act, &mut self.logits)?;
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
            }
            OutW::Q4K(out_w) => {
                let act = held(&mut self.act)?;
                gpu.enqueue_quantize_q8_1_head(rows, act)?;
                gpu.enqueue_gemv_q4k(out_w, &*act, &mut self.logits)?;
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
            }
            OutW::Q8_0 { qs, d } => {
                gpu.q8f32()
                    .enqueue_q8_0_gemv(stream, qs, d, rows, self.m, &mut self.logits)?;
                return self.argmax_finite(gpu, fault);
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
                return self.argmax_finite(gpu, fault);
            }
            OutW::Bf16(out_w) => {
                gpu.q8f32()
                    .enqueue_bf16_gemv(stream, out_w, rows, self.m, &mut self.logits)?;
                return self.argmax_finite(gpu, fault);
            }
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
    /// Pure enqueues, so the same body is what a capture records.
    pub(crate) fn enqueue_with_tail<F>(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        tail: F,
    ) -> Result<(), GpuError>
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

    /// The normed rows of the last run (the dump's `result_norm` tap; under
    /// [`HeadNorm::Mixed`] the input, `result_embd`). Blocking read;
    /// gate/debug use.
    pub fn normed_to_host(&self, gpu: &Gpu) -> Result<Vec<f32>, GpuError> {
        Ok(self.normed().to_host_vec(gpu.stream())?)
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
        Ok(self.logits.to_host_vec(gpu.stream())?)
    }

    /// The logits of the last run on the card (`n_vocab * m` f32, `[v*m + c]`).
    pub(crate) fn logits(&self) -> &DeviceBuffer<f32> {
        &self.logits
    }

    /// [`Head::logits_to_host`] into `out`, which holds exactly `n_vocab * m`
    /// f32: a caller reading every token reuses one buffer. Blocking read.
    pub fn logits_into_host(&self, gpu: &Gpu, out: &mut [f32]) -> Result<(), GpuError> {
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

    /// The argmax tokens of the last run, one per row. Blocking read: the
    /// engine stream waited for within [`crate::host::ENGINE_BOUND`]
    /// ([`crate::host::await_done`], past it [`GpuError::Stalled`]), then
    /// the words copied back; a raised fault is [`GpuError::Fault`], and a
    /// readback no argmax ever wrote a state error, as [`Head::token`].
    pub fn tokens(&self, gpu: &Gpu) -> Result<Vec<u32>, GpuError> {
        let stream = gpu.stream();
        crate::host::await_done("Head::tokens", || stream.query())?;
        let mut out = self.token_out.to_host_vec(stream)?;
        let (Some(sites), Some(word)) = (out.pop(), out.pop()) else {
            return Err(GpuError::state(
                "Head::tokens",
                "the readback buffer holds no fault words",
            ));
        };
        refuse_readback_fault(word, sites)?;
        Ok(out)
    }
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
        let decls = vec![
            TensorDecl {
                name: HEAD_TENSOR.to_string(),
                dims: vec![K as u64, N_VOCAB as u64],
                type_id: GgmlType::Q5_K.as_u32(),
                nbytes: rows.len() as u64,
            },
            TensorDecl {
                name: output_norm(),
                dims: vec![K as u64],
                type_id: GgmlType::F32.as_u32(),
                nbytes: gain.len() as u64,
            },
        ];
        let dir = std::env::temp_dir().join(format!("bloomery-q5k-head-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("a temp dir");
        let path = dir.join("head.gguf");
        let layout = Layout::new(&[], decls.clone()).expect("a layout");
        let file = std::io::BufWriter::new(std::fs::File::create(&path).expect("the file"));
        let mut writer = Writer::new(file, layout).expect("a writer");
        writer.tensor(&decls[0].name, &rows).expect("the rows");
        writer.tensor(&decls[1].name, &gain).expect("the gain");
        writer.finish().expect("a finished file");
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
}
