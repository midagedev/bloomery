//! Inkling 2's DSpark draft, the one opener of it for the binaries that run one beside the Mini:
//! `gate_inkling2_draft` (in the target's shadow) and the inkling2 seat of `bloomery-serve`
//! (through the serve seats' one owner of a served draft, `serve_seats::served::ServedSpark`).
//! [`Draft`] is the draft's statement and its weights, in either form: a checkpoint folder (the
//! target's `config.json`, which states the draft under `mtp_config` (`hf::dspark::inkling2`), and
//! `mtp.safetensors`, every tensor BF16) or a GGUF file of llama.cpp's DSpark convention (its keys
//! state the draft, held against the target's header). [`load`] is the one opener of both: the
//! common owner ([`AnchorDraft`]) over Inkling 2's layers ([`CtxDenseLayers`]: the fused context and
//! the anchor form of the recurrent head, its correction folded through the target's resident
//! output matrix, read in place), each form through its own weight source. [`Draft::reserve`] is
//! what a plan sets aside for the draft before any load. The anchor's embedding rows come from a GGUF file's header
//! ([`anchor_of_file`]) or from a checkpoint folder's part source ([`anchor_of_folder`]).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use bloomery_gpu::arch::inkling2::Body;
use bloomery_gpu::dspark::anchor_draft::AnchorDraft;
use bloomery_gpu::dspark::ctx_dense::{
    AnchorEmbed, AnchorTable, CtxDenseLayers, CtxSpec, HostRows,
};
use bloomery_gpu::dspark::dense::{Bf16Source, GgufSource, gguf_f32};
use bloomery_gpu::dspark::{DsparkBody, InPlace, TargetHead};
use bloomery_gpu::model::FIT_RESERVE;
use bloomery_gpu::weights::DevWeight;
use bloomery_gpu::{Gpu, GpuError, GpuModel};
use bloomery_gpu_gates::GateError;
use cuda_core::CudaContext;
use gguf::{GgmlType, Split};
use hf::dspark::inkling2 as config;
use model::arch::inkling2::hparams::Hparams;
use model::arch::inkling2::names;
use model::arch::models::LoopSpec;
use model::arch::names::{output, token_embd};
use model::embed::EmbedTable;
use refset::arch::draft::DsparkForm;
use safetensors::Safetensors;

const WHAT: &str = "inkling2 draft";

/// What a plan sets aside for the draft's working buffers, the owner's and the layers': the
/// head's scratch at a block of 8 rows among them.
const WORKING: u64 = 64 << 20;

/// The DSpark draft of the Inkling 2 seat: the common card draft over Inkling 2's layers.
pub type Ink2Dspark = app::dspark::CardDspark<CtxDenseLayers>;

/// The draft's statement and its weights' file.
pub struct Draft {
    /// The draft as its target's config (or the GGUF's keys) states it.
    pub d: config::Draft,
    /// The file the weights are read from: the folder's `mtp.safetensors`, or the GGUF.
    pub file: PathBuf,
    /// That file's length: what a plan sets aside for the weights.
    bytes: u64,
    weights: Weights,
}

/// The form the draft's weights are in.
enum Weights {
    /// The folder's `mtp.safetensors`.
    Folder(Safetensors),
    /// The GGUF file.
    Gguf(Split),
}

/// The folder's BF16 tensors, by the file's names.
struct Mtp<'a> {
    st: &'a Safetensors,
}

impl Bf16Source for Mtp<'_> {
    fn bf16(&self, name: &str) -> Result<(&[u16], &[usize]), GpuError> {
        self.st.bf16(name).map_err(|e| GpuError::Shape {
            what: WHAT,
            detail: e.to_string(),
        })
    }
}

impl Draft {
    /// The draft the folder `folder` states: its `config.json` (the target's whole config) read
    /// by `hf::dspark::inkling2`, its `mtp.safetensors` opened.
    pub fn read(folder: &Path) -> Result<Draft, GateError> {
        let at = folder.join("config.json");
        let text = std::fs::read_to_string(&at).map_err(|e| format!("{}: {e}", at.display()))?;
        let d = config::read(&text).map_err(|e| format!("{WHAT}: {}: {e}", at.display()))?;
        let file = folder.join("mtp.safetensors");
        let st =
            Safetensors::open(&file).map_err(|e| format!("{WHAT}: {}: {e}", file.display()))?;
        Ok(Draft {
            d,
            bytes: length(&file)?,
            file,
            weights: Weights::Folder(st),
        })
    }

    /// The draft the GGUF file `file` states over the target `hp` describes: its keys read by
    /// `hf::dspark::inkling2` and its tensors checked against the statement. A draft whose width,
    /// vocabulary, cap or taps are not the target's, a file that is not this draft's and a tensor
    /// the file lacks or holds in a type the load does not read are refused by name.
    pub fn read_gguf(file: &Path, hp: &Hparams) -> Result<Draft, GateError> {
        let split = Split::open(file).map_err(|e| format!("{WHAT}: {}: {e}", file.display()))?;
        let d = config::read_gguf(&split, target_of(hp)?)
            .map_err(|e| format!("{WHAT}: {}: {e}", file.display()))?;
        config::check_gguf(&d, &split).map_err(|e| format!("{WHAT}: {}: {e}", file.display()))?;
        Ok(Draft {
            d,
            bytes: length(file)?,
            file: file.to_path_buf(),
            weights: Weights::Gguf(split),
        })
    }

    /// The draft `form` names: a folder's or a GGUF file's (`hp` describes the target, which the
    /// GGUF's keys are held against).
    pub fn open(form: &DsparkForm, hp: &Hparams) -> Result<Draft, GateError> {
        match form {
            DsparkForm::Folder(folder) => Draft::read(folder),
            DsparkForm::Gguf(file) => Draft::read_gguf(file, hp),
        }
    }

    /// The target's visits the draft taps, as its config names them.
    #[must_use]
    pub fn taps(&self) -> &[usize] {
        &self.d.shape.taps
    }

    /// The card bytes a plan sets aside for the draft beside a target at `ctx` positions whose
    /// prompt pass holds `pass_rows` rows: [`Draft::reserve_fixed`] and [`Draft::reserve_per_pos`]
    /// a position. An upper bound for a folder, since no header states the Q8_0 the load makes of the BF16.
    #[must_use]
    pub fn reserve(&self, (ctx, pass_rows, hidden): (usize, usize, usize)) -> u64 {
        self.reserve_fixed((pass_rows, hidden)) + self.reserve_per_pos() * ctx as u64
    }

    /// The part of [`Draft::reserve`] no context moves: the weights' file whole (the card holds
    /// a folder's in less, since no header states the Q8_0 the load makes of the BF16), the fold's
    /// matrix and bias over the target's ids, the draft's taps (a prompt pass's rows of every
    /// tap) and the working buffers, with the load's own free margin ([`FIT_RESERVE`]) beside.
    #[must_use]
    pub fn reserve_fixed(&self, (pass_rows, hidden): (usize, usize)) -> u64 {
        let file = self.bytes;
        let ids = self.d.shape.vocab.n_target() as u64;
        let fold = ids * self.d.rank as u64 * 2 + ids * 4;
        let taps = (self.taps().len() * pass_rows.max(8) * hidden) as u64 * 4;
        file + fold + taps + WORKING + FIT_RESERVE as u64
    }

    /// The part of [`Draft::reserve`] a position of context adds: each global layer's context
    /// planes, two a layer (its keys and values). A windowed layer's planes are a constant its
    /// window fixes, which the working allowance ([`WORKING`]) covers.
    #[must_use]
    pub fn reserve_per_pos(&self) -> u64 {
        let b = &self.d.backbone;
        b.windowed.iter().filter(|&&w| !w).count() as u64
            * (b.kv_heads * b.head_dim / 2) as u64
            * 4
            * 2
    }
}

/// The length of `file`, refused by name where it cannot be read.
fn length(file: &Path) -> Result<u64, GateError> {
    std::fs::metadata(file)
        .map(|m| m.len())
        .map_err(|e| format!("{WHAT}: {}: {e}", file.display()).into())
}

/// The target's facts a GGUF draft is held to, from the target's own header.
pub fn target_of(hp: &Hparams) -> Result<config::Target, GateError> {
    let loop_ = hp
        .loop_
        .map(|l| -> Result<LoopSpec, GateError> {
            let w = |v: usize| u32::try_from(v).map_err(|_| format!("{WHAT}: loop {v}"));
            Ok(LoopSpec {
                core_start: w(l.core_start)?,
                core_end: w(l.core_end)?,
                passes: w(l.passes)?,
                gate_rank: w(l.gate_rank)?,
            })
        })
        .transpose()?;
    config::Target::new(
        hp.n_embd,
        (hp.n_vocab, hp.n_vocab_unpadded),
        hp.softcap,
        hp.n_layer,
        loop_,
    )
    .map_err(|e| format!("{WHAT}: the target's header: {e}").into())
}

/// The block's anchor row from a GGUF target: its embedding table and its embedding norm's gain.
///
/// # Errors
///
/// A tensor the file does not hold in the form the draft reads.
pub fn anchor_of_file(split: &Arc<Split>, hp: &Hparams) -> Result<AnchorEmbed, GpuError> {
    let table = EmbedTable::new(Arc::clone(split), &token_embd(), hp.n_embd)
        .map_err(|e| GpuError::shape(WHAT, format!("token_embd: {e}")))?;
    let gain = gguf_f32(split, names::TOKEN_EMBD_NORM, &[hp.n_embd])?;
    Ok(AnchorEmbed {
        table: AnchorTable::File(table),
        gain,
        eps: hp.rms_eps,
    })
}

/// The block's anchor row from a checkpoint folder: the embedding table and its norm's gain as
/// the folder's part source makes them, read whole on the host.
///
/// # Errors
///
/// A tensor the source refuses, or in a form the draft does not read.
#[cfg(feature = "checkpoint")]
pub fn anchor_of_folder<S: model::part::Source>(
    source: &S,
    hp: &Hparams,
) -> Result<AnchorEmbed, GpuError> {
    use model::part::Take;

    let read = |name: &str| -> Result<(GgmlType, Vec<u8>), GpuError> {
        let planned = source
            .plan(|n, _, _| {
                Ok::<Take, std::convert::Infallible>(if n == name {
                    Take::Whole
                } else {
                    Take::Skip
                })
            })
            .map_err(|e| GpuError::shape(WHAT, format!("{name}: {e}")))?;
        let planned = planned.into_iter().next().ok_or_else(|| {
            GpuError::shape(WHAT, format!("{name}: the source plans no part of it"))
        })?;
        let part = source
            .read(&planned)
            .map_err(|e| GpuError::shape(WHAT, format!("{name}: {e}")))?;
        let mut bytes = Vec::new();
        for span in part.spans() {
            bytes.extend_from_slice(span);
        }
        let ty = part.info.ty;
        source
            .release(&planned)
            .map_err(|e| GpuError::shape(WHAT, format!("{name}: {e}")))?;
        Ok((ty, bytes))
    };
    let (ty, bytes) = read(&token_embd())?;
    let rows = HostRows::new(ty, hp.n_embd, bytes)?;
    let (ty, bytes) = read(names::TOKEN_EMBD_NORM)?;
    if ty != GgmlType::F32 || bytes.len() != hp.n_embd * 4 {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "{} is {ty} of {} bytes; the draft reads {} f32",
                names::TOKEN_EMBD_NORM,
                bytes.len(),
                hp.n_embd
            ),
        ));
    }
    let gain: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    Ok(AnchorEmbed {
        table: AnchorTable::Rows(rows),
        gain,
        eps: hp.rms_eps,
    })
}

/// The draft beside the target `m`, running on `gpu`: its layers, the fused context and the
/// head, which reads `m`'s resident output matrix in place over the target's ids and folds the
/// correction through it (`m` outlives the draft; `gpu` is on `m`'s card, whose primary context
/// the two share). `anchor` is the block's anchor row from the target, `ctx` the context the
/// draft's planes hold.
pub fn load(
    m: &GpuModel<Body>,
    gpu: &Gpu,
    anchor: AnchorEmbed,
    d: &Draft,
    ctx: usize,
) -> Result<DsparkBody<CtxDenseLayers>, GpuError> {
    let b = &d.d.backbone;
    let window = b
        .window
        .ok_or_else(|| GpuError::shape(WHAT, "the draft states no window"))?;
    let draft = AnchorDraft {
        shape: &d.d.shape,
        layers: CtxSpec {
            layers: b.layers,
            eps: b.eps,
            rope_theta: b.rope_theta as f32,
            window,
            windowed: b.windowed.clone(),
            ctx,
        },
        hidden: b.hidden,
        rank: d.d.rank,
        padded_vocab: d.d.target.padded_vocab,
        softcap: d.d.target.softcap,
    };
    let output = target_head(m, d.d.shape.vocab.n_target(), gpu)?;
    match &d.weights {
        Weights::Folder(st) => draft.load(gpu, &Mtp { st }, anchor, output),
        Weights::Gguf(split) => {
            let names = config::gguf_name;
            draft.load(gpu, &GgufSource::new(split, &names), anchor, output)
        }
    }
}

/// The target's resident output matrix as the draft's head borrows it: in place, over the first
/// `rows` rows (the target's ids).
fn target_head(m: &GpuModel<Body>, rows: usize, gpu: &Gpu) -> Result<TargetHead, GpuError> {
    let ctx: &Arc<CudaContext> = gpu.context();
    // SAFETY: `m` owns the output matrix and outlives the draft (the caller drops the draft
    // first), and the draft runs on the target's card, whose primary context the two share, so
    // the windows' addresses stay allocated and in place for every launch and captured graph.
    unsafe {
        match m.weights().get(&output()) {
            Some(DevWeight::Q8_0 { qs, d, .. }) => Ok(TargetHead::Q8_0InPlace {
                qs: InPlace::rows(qs, rows, ctx)?,
                d: InPlace::rows(d, rows, ctx)?,
            }),
            Some(DevWeight::Bf16 { w, .. }) => {
                Ok(TargetHead::Bf16InPlace(InPlace::rows(w, rows, ctx)?))
            }
            Some(_) => Err(GpuError::shape(
                WHAT,
                "the target's output.weight is neither a resident Q8_0 nor a BF16 matrix, the \
                 two forms the draft's head reads in place",
            )),
            None => Err(GpuError::shape(
                WHAT,
                "the target holds no resident output.weight",
            )),
        }
    }
}

/// The served draft over the target `m`: [`load`] on `gpu`, behind the serve seats' adapter
/// ([`app::dspark::CardDspark`]) for the card `card`. The calling thread is bound to the
/// target's context again before this returns, the load failed or not.
pub fn open(
    m: &GpuModel<Body>,
    gpu: Gpu,
    card: &'static str,
    anchor: AnchorEmbed,
    d: &Draft,
    ctx: usize,
) -> Result<Ink2Dspark, GateError> {
    let (free, _) = gpu.mem_info()?;
    let loaded = load(m, &gpu, anchor, d, ctx);
    m.gpu()
        .context()
        .bind_to_thread()
        .map_err(GateError::from)?;
    let body = loaded?;
    let own = body.own_held();
    let held = body.layers().held(&own);
    let taken = free.saturating_sub(gpu.mem_info()?.0);
    Ok(app::dspark::CardDspark::new::<Body>(
        body,
        gpu,
        Arc::clone(m.gpu().context()),
        card,
        d.d.shape.clone(),
        held,
        taken,
    )?)
}
