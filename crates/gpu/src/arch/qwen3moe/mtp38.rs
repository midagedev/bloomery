//! Qwen3.8's MTP draft layer resident on the target's card ([`Mtp38`]): its
//! weights, its store and the reduced head's rows, opened once at load from
//! the plan `model::arch::qwen35moe::place::PlanInputs::plan_mtp` made, and
//! its program ([`Mtp38::run`]).
//!
//! - The draft file's tensors the plan puts on the card are uploaded by the
//!   placement loader from the draft's plan rows, each checked against its
//!   segment's bytes; the two injects are decoded to f32 and filed as the
//!   derived weights `place::mtp_widened` names; the router and the shared
//!   expert's gate are joined as the target's are (`plan38::router`).
//! - `token_embd` and `output` are the target's: nothing is uploaded for
//!   them. The body holds their names and the device addresses they had at
//!   open — the embedding's two Q8_0 planes, the head's two Q8_0 planes or
//!   its one Q6_K word plane; the target's `Weights`, which `GpuModel` drops
//!   after the body, own the buffers, and a walk reads them through the
//!   `&Weights` it is handed ([`Mtp38::borrowed`]), which refuses by name an
//!   address that moved.
//! - The store is every position's K and V, `[n_kv][ctx][head]` f16 each,
//!   the target's attention layers' K/V planes ([`KvPlanes`]).
//! - The head's row → id map, one `u32` a vocabulary id whatever the head
//!   (so a list's plan is the full head's): with a row list its first rows
//!   are the list's ids, which the head reads in place from the target's
//!   `output`; with the full head no walk reads it.
//!
//! The program ([`Mtp38::run`]) is ik's MTP graph (`build_qwen4exp.cpp`,
//! `is_mtp`) over `m` rows (1..=[`MTP_ROWS`]; captured at 1..=[`MTP_GRAPH_ROWS`]):
//! row `t` is a token at position `pos0 + t` and a target hidden row (the
//! four streams after the target's last layer, `res_hc`), or the draft's own
//! last row ([`MtpFeed::Own`], ik's scheme A: its token the draft's argmax,
//! its hidden the draft's own streams). Every launch is an entry the
//! target's chain has, but two of its own (`crate::mtp`), the routed
//! experts' `q8_0_gemv_sel_f32` and a listed head's `q8_0_gemv_ids` (one
//! row) or `q8_0_gemv_ids_mcol`, and over a Q6_K `output` its quantize and
//! `q6k_gemv_ids` pair (`crate::q6k_ids`):
//! - the target's embedding rows (`embed_rows_q8_0`, the borrowed
//!   `token_embd`), the input pack (`mtp_input`: `rms(e)·enorm` beside
//!   `rms(h)·hnorm` over all four streams, a row's four `[e | h_s]`), and
//!   `nextn.eh_proj` over the `4·m` packed columns into the streams, in runs
//!   of eight columns (`q8_0_gemv_mcol`, token-major);
//! - the attention site's mix, then dense gated GQA over the store: q (with
//!   each head's gate), k and v, the q/k norm, turn and append
//!   (`head_norm_neox_append_256`), the flash over every stored position
//!   below the row's (`gqa_flash_seg_mma_256_p4`, `gqa_flash_merge_256`: the
//!   file's `compress_ratios[48]` is 0, and ik runs the layer dense), the gate
//!   and the output projection;
//! - the feed-forward site's mix (the attention's combine first), the joined
//!   router, the routed slots' places over the identity map (every expert
//!   is the card's), the routed gate and up and down through
//!   `q8_0_gemv_sel_f32` (the stacks are Q8_0 planes, and the down's 640
//!   values are no multiple of `Q8Act`'s 256), SwiGLU between, the slots'
//!   weighted sum (`q38_card_acc` over 512 card experts), the shared expert
//!   and its gated sum (`q38_shared_add`);
//! - the head site's mix (`nextn.hc_head_*`, the block's combine first, so
//!   the streams are then the layer's output, `l_out`), the head's
//!   projection — a Q8_0 `output`'s planes ([`MtpHead::Full`]) or its rows
//!   of the list read in place through the map (`q8_0_gemv_ids`, or
//!   `q8_0_gemv_ids_mcol` past one row; [`MtpHead::Rows`]); a Q6_K one
//!   quantizes the head's input rows to q8_1 in the arena first (one launch
//!   a Q8_0 walk has not), then reads its rows, full or listed, through
//!   `q6k_gemv_ids`/`_mcol` — and
//!   `argmax_p_rows_fault`, which names each
//!   row's token and the draft's largest probability among the head's rows;
//! - the last row's token and streams copied beside the arena, where the
//!   next [`MtpFeed::Own`] walk reads them.
//!
//! The store is by position: a walk appends its rows' keys before its
//! attention reads them, and one may start at or below the end of the last
//! walk of the sequence, never past it (it would read keys no walk of the
//! sequence wrote). A new sequence ([`Mtp38::forget`], the body's reset)
//! holds no position and no own row.
//!
//! A store walk ([`MtpMode::Store`], the prompt's warmup) stops after the
//! append: a later row reads a row's keys and values and nothing else of
//! it. It runs over up to [`MTP_STORE_ROWS`] rows at once through the
//! launches the ubatch walk (`wide38`) runs, in its own buffers:
//! - the embedding rows and the input pack, as above;
//! - the pack quantized to 32-value q8 blocks (`quantize_gemm32`) and
//!   `eh_proj` as one 32-value GEMM (`gemm_q8_0p`) over the `4·m` packed
//!   columns, through a one-expert table;
//! - the attention site's wide mix (`HcWideKernels::enqueue_mix`);
//! - the mix quantized, q (with each head's gate), k and v as three GEMMs
//!   over the rows, then the norm and append.
//!
//! Each op computes a row's values from that row's inputs alone, so a
//! row's keys and values do not depend on the walk it lands in or on its
//! neighbours. Against an [`MtpMode::Eager`] walk, which reads the f32
//! rows, the three projections read q8 activations: the two stores agree to
//! the error of that quantization and are not bit-equal.
//!
//! Launches of a walk at `m` rows: [`walk_launches`]. The arena is the
//! program's, allocated once beside the body ([`Mtp38::arm`]) and counted
//! apart from the plan's bytes ([`Mtp38::arena_bytes`]).

use super::body::ATTN_SCALE_256;
use super::plan38::{HcSite, geo, router};
use super::program38::{Ctx38, Kernels38, MMA, q8};
use super::router::{RouterDims, RouterOut};
use super::scratch::{IN_IDS, IN_POS0, Inbox, KvPlanes, f32_view, param_view, put_input};
use super::scratch38::{PASS_ROWS, VERIFY_ROWS};
use crate::fault::Fault;
use crate::flash_gqa::{GqaArgs, partials_ms_len, partials_v_len_256};
use crate::gemm::{Gemm32Args, Gemm32Weight, GemmAct32, GemmInput, GemmRoute};
use crate::graph::Graph;
use crate::hc_gated::{Before, HcScratch, HcWideScratch, SiteWeights, WideMixArgs};
use crate::host::handoff::Places;
use crate::model::lookup::{f32_gain, f32_tensor};
use crate::mtp::{ArgmaxPArgs, MtpInputArgs};
use crate::q6k_ids::{Q6kIdsArgs, Q6kIdsKernels};
use crate::q8f32::{GemvOut, Q8_0GemvIdsArgs, Q8_0GemvMcolArgs, Q8_0SelArgs};
use crate::q38::{CardAccArgs, EmbedQ8Args, OutGateArgs, SharedAddArgs};
use crate::rope_neox::PartialNeoxArgs;
use crate::tensor::Q8Act;
use crate::weights::{DevWeight, Weights};
use crate::{DeviceTensor, Gpu, GpuError};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use gguf::quant::{GgmlType, dequant_row};
use model::arch::models::{HeadRows, MtpSource};
use model::placement::Plan;
use runtime::hc_gated::Geometry;
use std::mem::ManuallyDrop;

const WHAT: &str = "qwen4exp Mtp38";

/// A matrix of the target's the draft reads in place: its name and its two
/// Q8_0 planes' device addresses at open.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BorrowedPlanes {
    pub name: String,
    /// The code plane's and the scale plane's `CUdeviceptr`.
    pub planes: [u64; 2],
}

/// The target's `output` the draft reads in place: its Q8_0 planes' device
/// addresses at open, or its Q6_K word plane's with the row width and rows
/// the gemv reads.
#[derive(Clone, Debug, PartialEq, Eq)]
enum BorrowedHead {
    Q8_0(BorrowedPlanes),
    Q6K {
        name: String,
        plane: u64,
        k: usize,
        rows: usize,
    },
}

/// The head's `output` of the target as the form the draft was opened
/// beside, its addresses re-checked ([`Mtp38::borrowed`]): what a walk's
/// projection launches.
pub enum HeadOut<'w> {
    Q8_0 {
        qs: &'w DeviceTensor<u32>,
        d: &'w DeviceTensor<u16>,
    },
    Q6K {
        w: &'w DeviceTensor<u32>,
    },
}

/// The head's row → vocabulary id map, one word a vocabulary id: with a row
/// list its first `rows` words are the list's ids (each a row of the
/// target's `output`) and the rest 0; with the full head every word is 0
/// and no walk reads it.
struct HeadMap {
    ids: DeviceBuffer<u32>,
    /// The list's rows; `None` for the full head.
    rows: Option<usize>,
}

/// The MTP layer on the target's card. See the module comment.
pub struct Mtp38 {
    /// The draft's own tensors, the widened injects and the joined router.
    w: Weights,
    store: KvPlanes,
    head: HeadMap,
    /// `token_embd`, then `output` in the form its walk launches.
    embd: BorrowedPlanes,
    head_w: BorrowedHead,
    /// The Q6_K head's row-map gemv, loaded with the body.
    q6k_ids: Q6kIdsKernels,
    /// The layer's `blk.` index in the draft file.
    index: u32,
    ctx: usize,
    /// The program's arena, made once the body beside it is loaded
    /// ([`Mtp38::arm`]); outside the plan's bytes ([`Mtp38::arena_bytes`]).
    a: Option<MtpArena>,
    /// The captured walks, by rows, feed and head.
    graphs: Vec<(MtpKey, Graph)>,
    /// The rows of the last walk, which the readbacks and the next own row
    /// read; `None` before a walk of the sequence and after a store walk.
    last: Option<(usize, MtpHead)>,
    /// Positions of this sequence the store holds: the last walk's end. A
    /// walk may start at or below it, never past it.
    held: usize,
}

/// One sequence's side of the draft ([`Body38`]'s
/// [`Slots`](crate::model::Slots)): its store, the walks captured over it
/// and the store's own record of the positions it holds for the sequence.
/// The captures are declared first: they address the store, and drop while
/// it is alive. The program's arena stays the model's one — a walk's
/// buffers are scratch, rewritten within the walk before anything reads
/// them, save the own row an [`MtpFeed::Own`] walk reads, which no call
/// reads across a select: a chain's refresh walk rewrites it before any own
/// walk of the chain does.
pub(super) struct DraftSeq38 {
    graphs: Vec<(MtpKey, Graph)>,
    store: KvPlanes,
    last: Option<(usize, MtpHead)>,
    held: usize,
}

impl Mtp38 {
    /// The draft `mtp` describes, resident on `gpu` as `plan`'s draft plan
    /// places it, reading `target_w`'s `token_embd` and `output` (the head's
    /// listed rows among them); each segment's file pages leave the page
    /// cache once uploaded when `card_dontneed`. `slots` is the count of
    /// resident sequences the plan was made for
    /// ([`PlanInputs::plan_mtp_with_slots`](model::arch::qwen35moe::place::PlanInputs)):
    /// the plan's store term counts them all, this load's one sequence among
    /// them. Refused by name: a draft that is not the target's shape, a
    /// draft file that carries its own matrices, a borrowed matrix absent or
    /// not of its read form (`token_embd` Q8_0, `output` Q8_0 or Q6_K, each
    /// `[hidden, vocab]`), a listed id at or past the vocabulary, an
    /// upload whose bytes are not the plan's.
    #[allow(
        clippy::too_many_arguments,
        reason = "the card, the target's weights, the draft's file, inputs and plan, the page-cache release and the planned slots (rust-quality R8)"
    )]
    pub fn open(
        gpu: &Gpu,
        target_w: &Weights,
        draft_file: &Split,
        mtp: &model::arch::qwen35moe::place::MtpInputs,
        plan: &model::arch::qwen35moe::place::MtpPlan<'_>,
        card_dontneed: bool,
        slots: usize,
    ) -> Result<Mtp38, GpuError> {
        let d = &mtp.draft;
        let (hidden, vocab) = (d.hidden as usize, d.vocab as usize);
        if hidden != geo::HIDDEN {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the layer is {hidden} values wide; the kernels take {}",
                    geo::HIDDEN
                ),
            ));
        }
        match &d.source {
            MtpSource::File { borrows, .. } if borrows.embedding && borrows.head => {}
            other => {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "source {other:?}: the load borrows the target's token_embd and output \
                         (a shared draft file)"
                    ),
                ));
            }
        }
        let ctx = usize::try_from(plan.plan.ctx_max)
            .ok()
            .filter(|&c| c > 0)
            .ok_or_else(|| GpuError::shape(WHAT, format!("ctx_max {}", plan.plan.ctx_max)))?;
        gpu.context().bind_to_thread()?;
        let stream = gpu.stream();
        let embd = borrowed_planes(
            target_w,
            model::arch::qwen35moe::names::token_embd(),
            hidden,
            vocab,
        )?;
        let head_w = borrowed_head(
            target_w,
            model::arch::qwen35moe::names::output(),
            hidden,
            vocab,
        )?;
        let mut w = Weights::load_placed(
            stream,
            draft_file,
            &file_rows(&plan.draft),
            0,
            card_dontneed,
        )?;
        for t in &mtp.tensors {
            if let model::arch::qwen35moe::place::MtpLoad::Widened(_) = t.load {
                let name = model::arch::qwen35moe::place::mtp_widened(d.index, t.stem);
                let dw = widened(stream, draft_file, t, d.index)?;
                held_as_planned(&plan.draft, &name, dw.resident_bytes())?;
                w.insert_derived(name, dw)?;
            }
        }
        let (gate, shared) = (
            format!("blk.{}.ffn_gate_inp.weight", d.index),
            format!("blk.{}.ffn_gate_inp_shexp.weight", d.index),
        );
        w.join_rows(stream, &[&gate, &shared], router(d.index as usize))?;
        let head = match &d.head_rows {
            HeadRows::Full => HeadMap {
                ids: DeviceBuffer::zeroed(stream, vocab)?,
                rows: None,
            },
            HeadRows::List { ids, .. } => {
                if let Some(&id) = ids.iter().find(|&&id| id as usize >= vocab) {
                    return Err(GpuError::shape(
                        WHAT,
                        format!("the head's list names id {id}, past the vocabulary's {vocab}"),
                    ));
                }
                if ids.is_empty() || ids.len() > vocab {
                    return Err(GpuError::shape(
                        WHAT,
                        format!(
                            "the head's list holds {} rows; a list holds 1 to the vocabulary's \
                             {vocab}",
                            ids.len()
                        ),
                    ));
                }
                let mut map = vec![0u32; vocab];
                map[..ids.len()].copy_from_slice(ids);
                HeadMap {
                    ids: DeviceBuffer::from_host(stream, &map)?,
                    rows: Some(ids.len()),
                }
            }
        };
        let n = geo::N_KV * ctx * geo::HEAD;
        // The draft's store is f16: the qwen38 family's stores carry no q8_0
        // form.
        let store = KvPlanes::F16 {
            k: DeviceBuffer::zeroed(stream, n)?,
            v: DeviceBuffer::zeroed(stream, n)?,
        };
        stream.synchronize()?;
        let body = Mtp38 {
            w,
            store,
            head,
            embd,
            head_w,
            q6k_ids: Q6kIdsKernels::load(gpu.context())?,
            index: d.index,
            ctx,
            a: None,
            graphs: Vec::new(),
            last: None,
            held: 0,
        };
        let c = &plan.draft.cards[0];
        let got = [
            body.w.resident_bytes() as u64,
            body.store.bytes() as u64,
            body.map_bytes(),
        ];
        let want = [
            plan.draft_resident_bytes(),
            c.kv_bytes / slots as u64,
            plan.map_bytes,
        ];
        if got != want {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "resident weights, store and row map hold {got:?} bytes; the plan has {want:?}"
                ),
            ));
        }
        Ok(body)
    }

    /// Device bytes the draft holds: its weights, its store and the row map;
    /// the borrowed matrices are the target's.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.w.resident_bytes() + self.store.bytes() + self.map_bytes() as usize
    }

    fn map_bytes(&self) -> u64 {
        (self.head.ids.len() * std::mem::size_of::<u32>()) as u64
    }

    /// The target's matrices the draft reads, with the device addresses
    /// each holds at open: `token_embd`'s and a Q8_0 head's two planes, a
    /// Q6_K head's word plane and a zero.
    #[must_use]
    pub fn borrowed_planes(&self) -> [BorrowedPlanes; 2] {
        let head = match &self.head_w {
            BorrowedHead::Q8_0(b) => b.clone(),
            BorrowedHead::Q6K { name, plane, .. } => BorrowedPlanes {
                name: name.clone(),
                planes: [*plane, 0],
            },
        };
        [self.embd.clone(), head]
    }

    /// `token_embd` and `output` of `target`, the weights the draft was
    /// opened beside, the head in the form its walk launches. Refused by
    /// name when either is absent or not where it was at open.
    pub fn borrowed<'w>(
        &self,
        target: &'w Weights,
    ) -> Result<(&'w DevWeight, HeadOut<'w>), GpuError> {
        let embd = at_open(target, &self.embd)?;
        let head = match &self.head_w {
            BorrowedHead::Q8_0(b) => {
                let DevWeight::Q8_0 { qs, d, .. } = at_open(target, b)? else {
                    return Err(GpuError::tensor(WHAT, b.name.clone(), "Q8_0 planes"));
                };
                HeadOut::Q8_0 { qs, d }
            }
            BorrowedHead::Q6K {
                name,
                plane,
                k,
                rows,
            } => match target.get(name) {
                Some(DevWeight::KQuant {
                    ty: GgmlType::Q6_K,
                    w,
                    k: kw,
                }) if *kw == *k && w.rows() == *rows && w.buf().cu_deviceptr() == *plane => {
                    HeadOut::Q6K { w }
                }
                _ => {
                    return Err(GpuError::shape(
                        WHAT,
                        format!(
                            "{name} is not the Q6_K word plane of {rows} rows of {k} values at \
                             {plane:#x} the draft was opened beside"
                        ),
                    ));
                }
            },
        };
        Ok((embd, head))
    }

    /// The draft's own weights: its file's tensors by name, the widened
    /// injects and the joined router.
    #[must_use]
    pub fn weights(&self) -> &Weights {
        &self.w
    }

    /// The row → vocabulary id map (one word a vocabulary id) and the list's
    /// rows, its first words; `None` for the full head.
    #[must_use]
    pub fn head_map(&self) -> Option<(&DeviceBuffer<u32>, usize)> {
        self.head.rows.map(|n| (&self.head.ids, n))
    }

    /// The layer's `blk.` index in the draft file.
    #[must_use]
    pub fn index(&self) -> u32 {
        self.index
    }

    /// Positions the store holds.
    #[must_use]
    pub fn ctx(&self) -> usize {
        self.ctx
    }

    /// The store's keys and values at positions `0..n`, position-major:
    /// `[n][n_kv][head]` f16 bits each (the store's own layout is
    /// `[n_kv][ctx][head]`). Blocking; gate use.
    pub fn store_host(
        &self,
        stream: &CudaStream,
        n: usize,
    ) -> Result<(Vec<u16>, Vec<u16>), GpuError> {
        if n > self.ctx {
            return Err(GpuError::shape(
                WHAT,
                format!("positions 0..{n} of a store of {}", self.ctx),
            ));
        }
        let (ctx, head) = (self.ctx, geo::HEAD);
        let plane = |b: &DeviceBuffer<u16>| -> Result<Vec<u16>, GpuError> {
            let all = b.to_host_vec(stream)?;
            Ok((0..n)
                .flat_map(|p| (0..geo::N_KV).map(move |kv| (kv * ctx + p) * head))
                .flat_map(|at| &all[at..at + head])
                .copied()
                .collect())
        };
        let (k, v) = self.store.f16(WHAT)?;
        Ok((plane(k)?, plane(v)?))
    }

    /// The store's rows `rows` (`store_host`'s position-major shape, both
    /// planes) at positions `0..n` put back, the store counted at `n`: the
    /// side of a sequence state adopted by a put-back
    /// (`Body38::put_draft_rows`). Refused by name when the rows do not
    /// cover `n` positions. Blocking.
    pub(super) fn adopt_store(
        &mut self,
        stream: &CudaStream,
        n: usize,
        rows: &(Vec<u16>, Vec<u16>),
    ) -> Result<(), GpuError> {
        let want = n * geo::N_KV * geo::HEAD;
        if n > self.ctx || rows.0.len() != want || rows.1.len() != want {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the store's rows at 0..{n} of a store of {}: {} and {} values",
                    self.ctx,
                    rows.0.len(),
                    rows.1.len()
                ),
            ));
        }
        let (ctx, head) = (self.ctx, geo::HEAD);
        let plane = |rows: &[u16], buf: &mut DeviceBuffer<u16>| -> Result<(), GpuError> {
            let mut all = buf.to_host_vec(stream)?;
            for p in 0..n {
                for kv in 0..geo::N_KV {
                    let at = (p * geo::N_KV + kv) * head;
                    let to = (kv * ctx + p) * head;
                    all[to..to + head].copy_from_slice(&rows[at..at + head]);
                }
            }
            buf.copy_from_host(stream, &all)?;
            Ok(())
        };
        let (k, v) = self.store.f16_mut(WHAT)?;
        plane(&rows.0, k)?;
        plane(&rows.1, v)?;
        self.held = n;
        Ok(())
    }

    /// A new sequence's side of the draft ([`Body38`]'s
    /// [`Slots`](crate::model::Slots)): a zeroed store no walk has captured
    /// over, no positions held. Load-time allocation.
    pub(super) fn new_seq(&self, stream: &CudaStream) -> Result<DraftSeq38, GpuError> {
        let n = geo::N_KV * self.ctx * geo::HEAD;
        // The draft's store is f16: the qwen38 family's stores carry no q8_0
        // form.
        Ok(DraftSeq38 {
            graphs: Vec::new(),
            store: KvPlanes::F16 {
                k: DeviceBuffer::zeroed(stream, n)?,
                v: DeviceBuffer::zeroed(stream, n)?,
            },
            last: None,
            held: 0,
        })
    }

    /// Exchange the live sequence's side of the draft with `seq`'s: pointer
    /// moves only, so each sequence's captures keep addressing its own
    /// store.
    pub(super) fn swap_slot(&mut self, seq: &mut DraftSeq38) {
        std::mem::swap(&mut self.store, &mut seq.store);
        std::mem::swap(&mut self.graphs, &mut seq.graphs);
        std::mem::swap(&mut self.last, &mut seq.last);
        std::mem::swap(&mut self.held, &mut seq.held);
    }

    /// Device bytes one sequence's side of the draft holds: its store.
    pub(super) fn seq_bytes(&self) -> usize {
        self.store.bytes()
    }
}

/// A Q8_0 weight's two planes' device addresses; `None` for another format.
fn planes(dw: &DevWeight) -> Option<[u64; 2]> {
    match dw {
        DevWeight::Q8_0 { qs, d, .. } => Some([qs.buf().cu_deviceptr(), d.buf().cu_deviceptr()]),
        _ => None,
    }
}

/// `name` of the target's weights, the Q8_0 matrix at the addresses `b`
/// records; the refusal names what moved.
fn at_open<'w>(target: &'w Weights, b: &BorrowedPlanes) -> Result<&'w DevWeight, GpuError> {
    let dw = target
        .get(&b.name)
        .ok_or_else(|| GpuError::tensor(WHAT, b.name.clone(), "resident"))?;
    match planes(dw) {
        Some(p) if p == b.planes => Ok(dw),
        p => Err(GpuError::shape(
            WHAT,
            format!(
                "{} sits at {p:?}, the draft was opened beside {:?}",
                b.name, b.planes
            ),
        )),
    }
}

/// `name` of the target's weights `w`, resident as a Q8_0 `[hidden, vocab]`
/// matrix: its planes' addresses.
fn borrowed_planes(
    w: &Weights,
    name: String,
    hidden: usize,
    vocab: usize,
) -> Result<BorrowedPlanes, GpuError> {
    let dw = w
        .get(&name)
        .ok_or_else(|| GpuError::tensor(WHAT, name.clone(), "resident on the target's card"))?;
    let shape = (dw.k(), dw.rows());
    match planes(dw) {
        Some(planes) if shape == (hidden, vocab) => Ok(BorrowedPlanes { name, planes }),
        _ => Err(GpuError::shape(
            WHAT,
            format!(
                "the target's {name} is {} rows of {} values in another format than Q8_0; the \
                 draft reads a Q8_0 [{hidden}, {vocab}] matrix",
                shape.1, shape.0
            ),
        )),
    }
}

/// `name` of the target's weights `w`, resident as the head's read form over
/// a `[hidden, vocab]` matrix: its Q8_0 planes, or its Q6_K word plane with
/// the row width and rows the gemv reads.
fn borrowed_head(
    w: &Weights,
    name: String,
    hidden: usize,
    vocab: usize,
) -> Result<BorrowedHead, GpuError> {
    let dw = w
        .get(&name)
        .ok_or_else(|| GpuError::tensor(WHAT, name.clone(), "resident on the target's card"))?;
    match dw {
        DevWeight::Q8_0 { .. } if (dw.k(), dw.rows()) == (hidden, vocab) => {
            Ok(BorrowedHead::Q8_0(borrowed_planes(w, name, hidden, vocab)?))
        }
        DevWeight::KQuant {
            ty: GgmlType::Q6_K,
            w: plane,
            k,
        } if (*k, plane.rows()) == (hidden, vocab) => Ok(BorrowedHead::Q6K {
            plane: plane.buf().cu_deviceptr(),
            k: *k,
            rows: plane.rows(),
            name,
        }),
        _ => {
            let (k, rows) = (dw.k(), dw.rows());
            Err(GpuError::shape(
                WHAT,
                format!(
                    "the target's {name} is {rows} rows of {k} values in another format than \
                     Q8_0 or Q6_K; the draft reads a [{hidden}, {vocab}] matrix of one of them"
                ),
            ))
        }
    }
}

/// `draft`'s plan with only its file tensors' rows: the derived ones
/// (`derived.*`) are this module's uploads, not the loader's.
fn file_rows<'a>(draft: &Plan<'a>) -> Plan<'a> {
    let mut p = draft.clone();
    p.rows
        .retain(|r| !p.model.tensors[r.tensor].name.starts_with("derived."));
    p
}

/// Refuse by name a derived upload `name` of `bytes` device bytes that is
/// not the draft plan's one card segment of it.
fn held_as_planned(draft: &Plan<'_>, name: &str, bytes: usize) -> Result<(), GpuError> {
    let planned = draft
        .rows
        .iter()
        .find(|r| draft.model.tensors[r.tensor].name == name)
        .map(|r| r.segments.iter().map(|s| s.resident_bytes).sum::<u64>());
    if planned == Some(bytes as u64) {
        Ok(())
    } else {
        Err(GpuError::shape(
            WHAT,
            format!("{name} holds {bytes} device bytes; the draft's plan has {planned:?}"),
        ))
    }
}

/// Tensor `t` of layer `index` of `draft`, a Q8_0 matrix, decoded to f32 on
/// the host and uploaded: rows × k f32, `k` the file's row width.
fn widened(
    stream: &CudaStream,
    draft: &Split,
    t: &model::arch::qwen35moe::place::MtpTensor,
    index: u32,
) -> Result<DevWeight, GpuError> {
    let name = t.name(index);
    let (shard, info) = draft
        .find(&name)
        .ok_or_else(|| GpuError::tensor(WHAT, name.clone(), "in the draft file"))?;
    if info.ty != GgmlType::Q8_0 {
        return Err(GpuError::shape(
            WHAT,
            format!("{name} is {}; the load widens a Q8_0 tensor", info.ty),
        ));
    }
    let g = draft
        .shard(shard)
        .ok_or_else(|| GpuError::shape(WHAT, format!("{name} names shard {shard}")))?;
    let k = info.dims[0] as usize;
    let rows = info.dims[1..].iter().product::<u64>() as usize;
    let mut vals = vec![0.0f32; rows * k];
    dequant_row(info.ty, g.data(info)?, &mut vals).map_err(::model::ModelError::from)?;
    Ok(DevWeight::F32 {
        w: DeviceTensor::upload(stream, &vals, rows, k)?,
        k,
    })
}

// ------------------------------------------------------------ the program

/// The most rows one walk of the draft takes: the m-column kernels' width.
pub const MTP_ROWS: usize = PASS_ROWS;

/// The most rows a captured walk takes: a verify's, whose kept rows a
/// refresh runs.
pub const MTP_GRAPH_ROWS: usize = VERIFY_ROWS;

/// The most rows one store walk ([`MtpMode::Store`]) takes: the columns of
/// its GEMMs, and the rows of the store arena it runs in.
pub const MTP_STORE_ROWS: usize = 64;

/// Values of a row's streams, the target's hidden row.
const WIDE: usize = geo::STREAMS * geo::HIDDEN;

/// Columns one `eh_proj` launch takes.
const EH_COLS: usize = 8;

/// A walk's launches at `m` rows (the module doc's list, in order): the
/// embedding and the pack; `eh_proj` in runs of [`EH_COLS`] columns; the
/// attention site's mix (3), q, k and v, the norm and append, the flash's
/// two, the gate and the output projection; the feed-forward site's mix
/// (3), the router and the places, the routed gate, up, SwiGLU and down,
/// the slots' sum, the shared expert's four and its gated sum; the head
/// site's mix (3), the projection — a Q6_K head's quantize launch before
/// it, `q6k_head` — and the argmax; the two copies of the last row. A Q6_K
/// walk holds one launch more than a Q8_0 one.
#[must_use]
pub fn walk_launches(m: usize, q6k_head: bool) -> usize {
    2 + (4 * m).div_ceil(EH_COLS)
        + (3 + 3 + 1 + 2 + 1 + 1)
        + (3 + 2 + 4 + 1 + 4 + 1)
        + (3 + 1 + 1)
        + usize::from(q6k_head)
        + 2
}

/// Which projection the draft's head runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MtpHead {
    /// The target's `output`: every token of the vocabulary.
    Full,
    /// The load's row list: the target's `output` rows of its ids, read in
    /// place through the head's map.
    Rows,
}

/// Where a walk's hidden rows come from.
#[derive(Clone, Copy, Debug)]
pub enum MtpHidden<'a> {
    /// `m · 4 · 2560` values written from the host.
    Host(&'a [f32]),
    /// The target's streams after its last walk of `walk`'s arena, from row
    /// `first` on; that walk must have run its head, whose mix applies the
    /// last layer's combine (a step, a verify, a prompt's last pass or
    /// ubatch).
    Target { walk: TargetRows, first: usize },
}

/// The target arena a [`MtpHidden::Target`] feed reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetRows {
    /// The step's one row.
    Step,
    /// A pass's or a verify's rows.
    Pass,
    /// A ubatch walk's rows.
    Ubatch,
}

/// The positions a target arena's rows hold, row `r` position `first + r`:
/// what the call that last wrote the arena left there, less the positions a
/// cut took back since. No rows: the arena holds no position of the
/// sequence. A walk that reads an arena names the positions its rows must
/// hold ([`MtpHidden::Target`]); without this record it would read a stale
/// arena silently.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Held38 {
    first: u32,
    rows: u32,
}

impl Held38 {
    /// No position.
    pub const NONE: Held38 = Held38 { first: 0, rows: 0 };

    /// Rows `0 .. rows` at positions `first ..`.
    #[must_use]
    pub const fn at(first: u32, rows: u32) -> Held38 {
        Held38 { first, rows }
    }

    /// The positions from `pos` on taken back.
    pub(super) fn cut(&mut self, pos: u32) {
        self.rows = self.rows.min(pos.saturating_sub(self.first));
    }

    /// The rows it holds, and the first position they hold.
    #[must_use]
    pub const fn parts(self) -> (u32, u32) {
        (self.first, self.rows)
    }

    /// Whether rows `row .. row + m` hold positions `at .. at + m`.
    #[must_use]
    pub const fn holds(self, row: usize, m: usize, at: u32) -> bool {
        row + m <= self.rows as usize && self.first as usize + row == at as usize
    }

    /// The positions held, for a refusal's words.
    fn shown(self) -> String {
        match self.rows {
            0 => "no position".to_string(),
            r => format!("positions {}..{}", self.first, self.first + r),
        }
    }
}

/// What a sequence state carries of the draft beside the target's own
/// stores ([`Body38::draft_rows`]): the store's K and V rows below its
/// `held` positions ([`Mtp38::held`], position-major as `store_host`
/// gathers them), and the step's and the pass's arena rows with the
/// positions they hold ([`Held38`]) — the target's hidden rows a draft's
/// waiting rows read when it rejoins the sequence. The ubatch walk's rows
/// are not carried: after a resume that arena holds no position, and a walk
/// that names it is refused.
#[derive(Clone, Debug)]
pub struct DraftRows38 {
    pub(super) held: u32,
    pub(super) store: (Vec<u16>, Vec<u16>),
    pub(super) step: (Held38, Vec<f32>),
    pub(super) pass: (Held38, Vec<f32>),
}

impl DraftRows38 {
    /// The host bytes it holds.
    #[must_use]
    pub fn bytes(&self) -> usize {
        (self.store.0.len() + self.store.1.len()) * size_of::<u16>()
            + (self.step.1.len() + self.pass.1.len()) * size_of::<f32>()
    }
}

/// A walk's rows.
#[derive(Clone, Copy, Debug)]
pub enum MtpFeed<'a> {
    /// `tokens` at positions `pos0 ..`, each beside its hidden row.
    Rows {
        tokens: &'a [u32],
        pos0: u32,
        hidden: MtpHidden<'a>,
    },
    /// One row at `pos0`: the last walk's last row's token and streams.
    Own { pos0: u32 },
}

/// The target's streams a [`MtpHidden::Target`] feed copies from, and the
/// positions its rows hold ([`Held38`]): the buffer and the record together,
/// the body's `mtp_target` resolving both over its arenas.
#[derive(Clone, Copy)]
pub(super) struct Target38<'a> {
    pub(super) buf: &'a DeviceBuffer<f32>,
    pub(super) held: Held38,
}

/// How a walk runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MtpMode {
    /// Enqueued launch by launch.
    Eager,
    /// Replayed from its capture, captured on first use.
    Graph,
    /// Enqueued launch by launch through the store's append and no
    /// further, up to [`MTP_STORE_ROWS`] rows through the wide launches
    /// (module doc): the rows' keys and values, which are all a later row
    /// reads of them (a row's input is its token and the target's hidden
    /// row, never the draft's output). No flash, feed-forward block or head,
    /// no readback and no own row: the prompt's warmup. A row's stored bits
    /// are a function of that row's inputs alone, within the q8
    /// activations' error of an [`MtpMode::Eager`] walk's.
    Store,
}

/// A captured walk's key: its rows, whether it reads its own last row, its
/// head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MtpKey {
    m: usize,
    own: bool,
    head: MtpHead,
}

/// A walk's readback: each row's token and the draft's probability of it
/// among the head's rows.
#[derive(Clone, Debug, PartialEq)]
pub struct MtpDraft {
    pub tokens: Vec<u32>,
    pub p: Vec<f32>,
}

/// A gate's taps of an eager walk: the streams `eh_proj` wrote (`m · 4 ·
/// 2560`), the router's logits, ids and weights, `slots` and `logits` a row
/// as the router writes them, the routed slots' and the shared expert's
/// SwiGLU outputs (the downs' inputs, `m · 10 · 640` and `m · 640`), and
/// every [`MtpNode`]'s rows, token-major.
#[derive(Clone, Debug)]
pub struct MtpTaps {
    pub eh: Vec<f32>,
    pub logits: Vec<f32>,
    pub ids: Vec<u32>,
    pub weights: Vec<f32>,
    pub routed_h: Vec<f32>,
    pub shared_h: Vec<f32>,
    /// The router's logits and slots a row.
    pub logits_row: usize,
    pub slots_row: usize,
    /// Each node of [`MtpNode::ALL`] in order: `m` rows of its width.
    pub nodes: Vec<(MtpNode, Vec<f32>)>,
}

/// A value the walk writes between `eh_proj` and the head, which armed taps
/// copy as the walk writes it: where a comparison with ik's graph finds the
/// first node that leaves its band.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MtpNode {
    /// The attention site's mix: the attention's input.
    AttnIn,
    /// The attention's heads after the output gate, before the output
    /// projection.
    AttnGated,
    /// The output projection: the attention block's output.
    AttnOut,
    /// The streams after the attention's combine.
    AttnCombined,
    /// The feed-forward site's mix: the router's and the experts' input.
    FfnIn,
    /// The routed slots' weighted sum.
    Routed,
    /// The shared expert's down projection, before its gate.
    Shared,
    /// The block's output: the routed sum and the gated shared expert.
    FfnOut,
    /// The head site's mix: the head projection's input.
    HeadIn,
}

impl MtpNode {
    /// Every node, in the walk's order.
    pub const ALL: [MtpNode; 9] = [
        MtpNode::AttnIn,
        MtpNode::AttnGated,
        MtpNode::AttnOut,
        MtpNode::AttnCombined,
        MtpNode::FfnIn,
        MtpNode::Routed,
        MtpNode::Shared,
        MtpNode::FfnOut,
        MtpNode::HeadIn,
    ];

    /// The node's values a row.
    #[must_use]
    pub fn width(self) -> usize {
        match self {
            MtpNode::AttnGated => geo::ATTN,
            MtpNode::AttnCombined => WIDE,
            _ => geo::HIDDEN,
        }
    }

    fn at(self) -> usize {
        MtpNode::ALL
            .iter()
            .position(|&n| n == self)
            .expect("ALL lists every node")
    }
}

/// The taps' device side.
struct TapBufs {
    eh: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    ids: DeviceBuffer<u32>,
    weights: DeviceBuffer<f32>,
    routed_h: DeviceBuffer<f32>,
    shared_h: DeviceBuffer<f32>,
    /// [`MtpNode::ALL`]'s rows, [`MTP_ROWS`] of each node's width.
    nodes: Vec<DeviceBuffer<f32>>,
}

impl TapBufs {
    /// Copy the first `m` rows of `src`, `node`'s values, into its tap;
    /// enqueued on `stream`.
    fn node(
        &mut self,
        stream: &CudaStream,
        node: MtpNode,
        src: &DeviceBuffer<f32>,
        m: usize,
    ) -> Result<(), GpuError> {
        let n = m * node.width();
        // SAFETY: `src` holds MTP_ROWS rows of the node's width (the arena's
        // buffer the node names) and the tap as many; m <= MTP_ROWS; the
        // windows live for this copy.
        let (mut dst, src) =
            unsafe { (f32_view(&self.nodes[node.at()], 0, n), f32_view(src, 0, n)) };
        dst.copy_from_device_async(&src, stream)?;
        Ok(())
    }
}

/// A store walk's own buffers for up to [`MTP_STORE_ROWS`] rows,
/// token-major (module doc): the hidden rows, the embedding with its
/// positions and key counts, the pack and its q8 blocks over the `4·m`
/// packed columns, the streams, the wide mix's scratch, the mix and its q8
/// blocks, q with its gates, the turned q, k and v, and the two one-expert
/// tables (`4·m` and `m` slots).
struct StoreArena {
    h: DeviceBuffer<f32>,
    emb: DeviceBuffer<f32>,
    pos: DeviceBuffer<u32>,
    n_keys: DeviceBuffer<u32>,
    pack: DeviceBuffer<f32>,
    pack_q: GemmAct32,
    res: DeviceBuffer<f32>,
    hc: HcWideScratch,
    mixed: DeviceBuffer<f32>,
    mixed_q: GemmAct32,
    qg: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    packed: GemmRoute,
    rows: GemmRoute,
}

impl StoreArena {
    fn new(stream: &CudaStream, geometry: Geometry) -> Result<StoreArena, GpuError> {
        let r = MTP_STORE_ROWS;
        let f = |n: usize| DeviceBuffer::<f32>::zeroed(stream, n);
        let u = |n: usize| DeviceBuffer::<u32>::zeroed(stream, n);
        let (h, cols) = (geo::HIDDEN, geo::STREAMS * r);
        Ok(StoreArena {
            h: f(r * WIDE)?,
            emb: f(r * h)?,
            pos: u(r)?,
            n_keys: u(r)?,
            pack: f(r * 2 * WIDE)?,
            pack_q: GemmAct32::new(stream, cols, 2 * h)?,
            res: f(r * WIDE)?,
            hc: HcWideScratch::new(stream, geometry, r)?,
            mixed: f(r * h)?,
            mixed_q: GemmAct32::new(stream, r, h)?,
            qg: f(r * geo::Q_ROWS)?,
            q: f(r * geo::ATTN)?,
            k: f(r * geo::KV)?,
            v: f(r * geo::KV)?,
            packed: GemmRoute::new(stream, cols, 1)?,
            rows: GemmRoute::new(stream, r, 1)?,
        })
    }

    fn bytes(&self) -> usize {
        let f32s = [
            &self.h,
            &self.emb,
            &self.pack,
            &self.res,
            &self.mixed,
            &self.qg,
            &self.q,
            &self.k,
            &self.v,
        ];
        f32s.iter().map(|b| b.num_bytes()).sum::<usize>()
            + self.pos.num_bytes()
            + self.n_keys.num_bytes()
            + self.pack_q.bytes()
            + self.hc.bytes()
            + self.mixed_q.bytes()
            + self.packed.bytes()
            + self.rows.bytes()
    }
}

/// The program's buffers for up to [`MTP_ROWS`] rows, token-major, and a
/// store walk's for up to [`MTP_STORE_ROWS`] (`st`).
struct MtpArena {
    /// The walk's record — `pos0`, then the rows' tokens — and its windows.
    inbox: Inbox,
    pos0: ManuallyDrop<DeviceBuffer<u32>>,
    /// The hidden rows a [`MtpFeed::Rows`] walk reads, and the last row's
    /// token and streams a [`MtpFeed::Own`] walk reads.
    h_in: DeviceBuffer<f32>,
    own_id: DeviceBuffer<u32>,
    own_h: DeviceBuffer<f32>,
    emb: DeviceBuffer<f32>,
    pos: DeviceBuffer<u32>,
    n_keys: DeviceBuffer<u32>,
    pack: DeviceBuffer<f32>,
    /// The streams: `eh_proj`'s output, then each combine in place.
    res: DeviceBuffer<f32>,
    mixed: DeviceBuffer<f32>,
    y: DeviceBuffer<f32>,
    hc: HcScratch,
    qg: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    part_v: DeviceBuffer<f32>,
    part_ms: DeviceBuffer<f32>,
    flash: DeviceBuffer<f32>,
    attn: DeviceBuffer<f32>,
    route: RouterOut,
    /// The routed slots' places (ten a row) and the identity map they are
    /// read through.
    sel: DeviceBuffer<u32>,
    id_map: DeviceBuffer<u32>,
    ffn_x: DeviceBuffer<f32>,
    rg: DeviceBuffer<f32>,
    ru: DeviceBuffer<f32>,
    rh: DeviceBuffer<f32>,
    rdown: DeviceBuffer<f32>,
    acc: DeviceBuffer<f32>,
    sh_g: DeviceBuffer<f32>,
    sh_u: DeviceBuffer<f32>,
    sh_h: DeviceBuffer<f32>,
    sh_y: DeviceBuffer<f32>,
    head_x: DeviceBuffer<f32>,
    /// The head's input rows quantized to q8_1: a Q6_K head's projection
    /// reads them ([`MTP_ROWS`] columns of the hidden width); `None` for a
    /// Q8_0 head, whose projection reads the f32 rows.
    head_act: Option<Q8Act>,
    /// The head's logits, `[row][m]` as the gemv writes them, for the full
    /// vocabulary.
    logits: DeviceBuffer<f32>,
    /// The argmax's readback (`2·rows + 2` words), its ticket count and the
    /// stand-in map word of a full head.
    out: DeviceBuffer<u32>,
    done: DeviceBuffer<u32>,
    no_map: DeviceBuffer<u32>,
    /// A window chain's readback: each walk's last row's id and probability
    /// bits (two words a walk), then the last walk's fault word and site
    /// mask — read once a window ([`Mtp38::run_chain`]).
    chain: DeviceBuffer<u32>,
    taps: Option<TapBufs>,
    st: StoreArena,
    vocab: usize,
}

impl MtpArena {
    fn new(
        stream: &CudaStream,
        dims: RouterDims,
        vocab: usize,
        head_act: bool,
    ) -> Result<MtpArena, GpuError> {
        let r = MTP_ROWS;
        let f = |n: usize| DeviceBuffer::<f32>::zeroed(stream, n);
        let u = |n: usize| DeviceBuffer::<u32>::zeroed(stream, n);
        let h = geo::HIDDEN;
        let inbox = Inbox::new(stream, IN_IDS + MTP_STORE_ROWS)?;
        // SAFETY: word IN_POS0 of the inbox's IN_IDS + MTP_STORE_ROWS device
        // words, and the inbox moves into the arena beside the window (a
        // move of the handle, not of the allocation), where it outlives it.
        let pos0 = unsafe { param_view::<u32>(inbox.dev(), IN_POS0, 1) };
        let geometry = Geometry::new(geo::STREAMS as u32, geo::RANK as u32, h as u32)
            .map_err(|e| GpuError::shape(WHAT, e.to_string()))?;
        let ids: Vec<u32> = (0..geo::EXPERTS as u32).collect();
        Ok(MtpArena {
            inbox,
            pos0,
            h_in: f(r * WIDE)?,
            own_id: u(1)?,
            own_h: f(WIDE)?,
            emb: f(r * h)?,
            pos: u(r)?,
            n_keys: u(r)?,
            pack: f(r * 2 * WIDE)?,
            res: f(r * WIDE)?,
            mixed: f(r * h)?,
            y: f(r * h)?,
            hc: HcScratch::new(stream, geometry)?,
            qg: f(r * geo::Q_ROWS)?,
            k: f(r * geo::KV)?,
            v: f(r * geo::KV)?,
            q: f(r * geo::ATTN)?,
            part_v: f(partials_v_len_256(r, geo::N_HEAD))?,
            part_ms: f(partials_ms_len(r, geo::N_HEAD))?,
            flash: f(r * geo::ATTN)?,
            attn: f(r * geo::ATTN)?,
            route: RouterOut::with_tokens(stream, dims, r)?,
            sel: u(r * geo::N_USED)?,
            id_map: DeviceBuffer::from_host(stream, &ids)?,
            ffn_x: f(r * h)?,
            rg: f(r * geo::N_USED * geo::FF)?,
            ru: f(r * geo::N_USED * geo::FF)?,
            rh: f(r * geo::N_USED * geo::FF)?,
            rdown: f(r * geo::N_USED * h)?,
            acc: f(r * h)?,
            sh_g: f(r * geo::FF)?,
            sh_u: f(r * geo::FF)?,
            sh_h: f(r * geo::FF)?,
            sh_y: f(r * h)?,
            head_x: f(r * h)?,
            head_act: head_act.then(|| Q8Act::with_k(stream, r, h)).transpose()?,
            logits: f(r * vocab)?,
            out: u(2 * r + 2)?,
            done: u(1)?,
            no_map: u(1)?,
            chain: u(2 * MTP_GRAPH_ROWS + 2)?,
            taps: None,
            st: StoreArena::new(stream, geometry)?,
            vocab,
        })
    }

    fn bytes(&self) -> usize {
        let f32s = [
            &self.h_in,
            &self.own_h,
            &self.emb,
            &self.pack,
            &self.res,
            &self.mixed,
            &self.y,
            &self.hc.xn,
            &self.hc.dpart,
            &self.hc.ipart,
            &self.hc.lo,
            &self.hc.wgt,
            &self.qg,
            &self.k,
            &self.v,
            &self.q,
            &self.part_v,
            &self.part_ms,
            &self.flash,
            &self.attn,
            &self.ffn_x,
            &self.rg,
            &self.ru,
            &self.rh,
            &self.rdown,
            &self.acc,
            &self.sh_g,
            &self.sh_u,
            &self.sh_h,
            &self.sh_y,
            &self.head_x,
            &self.logits,
        ];
        let act = self.head_act.as_ref().map_or(0, Q8Act::device_bytes);
        let u32s = [
            &self.own_id,
            &self.pos,
            &self.n_keys,
            &self.sel,
            &self.id_map,
            &self.out,
            &self.done,
            &self.no_map,
            &self.chain,
        ];
        f32s.iter().map(|b| b.num_bytes()).sum::<usize>()
            + u32s.iter().map(|b| b.num_bytes()).sum::<usize>()
            + act
            + self.inbox.bytes()
            + self.route.bytes()
            + self.st.bytes()
            + self.taps.as_ref().map_or(0, |t| {
                t.eh.num_bytes() + t.logits.num_bytes() + t.ids.num_bytes()
            })
    }
}

/// The draft layer's names, read once a walk.
struct Names {
    attn_site: HcSite,
    ffn_site: HcSite,
    head_site: HcSite,
    enorm: String,
    hnorm: String,
    eh: String,
    q: String,
    k: String,
    v: String,
    q_norm: String,
    k_norm: String,
    out: String,
    router: String,
    gate_exps: String,
    up_exps: String,
    down_exps: String,
    gate_sh: String,
    up_sh: String,
    down_sh: String,
}

impl Names {
    fn of(index: u32) -> Names {
        let b = |stem: &str| format!("blk.{index}.{stem}");
        let site = |sub: &str| HcSite {
            norm: b(&format!("hc_{sub}_norm.weight")),
            down: b(&format!("hc_{sub}_down.weight")),
            up: b(&format!("hc_{sub}_up.weight")),
            inject: Some(model::arch::qwen35moe::place::mtp_widened(
                index,
                &format!("hc_{sub}_inject.weight"),
            )),
        };
        Names {
            attn_site: site("attn"),
            ffn_site: site("ffn"),
            head_site: HcSite {
                norm: b("nextn.hc_head_norm.weight"),
                down: b("nextn.hc_head_down.weight"),
                up: b("nextn.hc_head_up.weight"),
                inject: None,
            },
            enorm: b("nextn.enorm.weight"),
            hnorm: b("nextn.hnorm.weight"),
            eh: b("nextn.eh_proj.weight"),
            q: b("attn_q.weight"),
            k: b("attn_k.weight"),
            v: b("attn_v.weight"),
            q_norm: b("attn_q_norm.weight"),
            k_norm: b("attn_k_norm.weight"),
            out: b("attn_output.weight"),
            router: router(index as usize),
            gate_exps: b("ffn_gate_exps.weight"),
            up_exps: b("ffn_up_exps.weight"),
            down_exps: b("ffn_down_exps.weight"),
            gate_sh: b("ffn_gate_shexp.weight"),
            up_sh: b("ffn_up_shexp.weight"),
            down_sh: b("ffn_down_shexp.weight"),
        }
    }
}

/// What a walk reads besides the draft: the card, the target's weights,
/// the kernels, the norms' epsilon and the rope table.
pub(super) struct MtpCtx<'a> {
    pub(super) gpu: &'a Gpu,
    pub(super) tw: &'a Weights,
    pub(super) k: &'a Kernels38,
    pub(super) eps: f32,
    pub(super) table: &'a DeviceBuffer<f32>,
}

impl Mtp38 {
    /// The program's arena beside the loaded body: the router's dims, the
    /// vocabulary the head writes and the rope table's rows, which must
    /// cover the store's positions, held to `arena` device bytes — the plan's
    /// count ([`model::arch::qwen35moe::place::mtp_arena_bytes`] and, for a
    /// Q6_K head,
    /// [`model::arch::qwen35moe::place::mtp_head_act_bytes`]). Load-time
    /// only.
    pub(super) fn arm(
        &mut self,
        stream: &CudaStream,
        dims: RouterDims,
        vocab: usize,
        rope_rows: usize,
        arena: u64,
    ) -> Result<(), GpuError> {
        if self.ctx > rope_rows {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a store of {} positions over a rope table of {rope_rows} rows",
                    self.ctx
                ),
            ));
        }
        let q6k = matches!(self.head_w, BorrowedHead::Q6K { .. });
        let a = MtpArena::new(stream, dims, vocab, q6k)?;
        let got = a.bytes() as u64;
        if got != arena {
            return Err(GpuError::shape(
                WHAT,
                format!("the program's arena holds {got} device bytes; the plan counts {arena}"),
            ));
        }
        self.a = Some(a);
        stream.synchronize()?;
        Ok(())
    }

    /// Device bytes of the program's arena, which the plan does not count.
    #[must_use]
    pub fn arena_bytes(&self) -> usize {
        self.a.as_ref().map_or(0, MtpArena::bytes)
    }

    /// Arm (or disarm) the taps of the eager walks ([`MtpTaps`]): `eh_proj`'s
    /// streams, the router's logits, ids and weights, the SwiGLU outputs and
    /// every [`MtpNode`]. The captured walks are dropped first (a
    /// capture records what it was given). Load-time allocation; gate use.
    pub fn set_taps(&mut self, stream: &CudaStream, on: bool) -> Result<(), GpuError> {
        self.graphs.clear();
        let a = self.arena()?;
        a.taps = if on {
            Some(TapBufs {
                eh: DeviceBuffer::zeroed(stream, MTP_ROWS * WIDE)?,
                logits: DeviceBuffer::zeroed(stream, a.route.logits.len())?,
                ids: DeviceBuffer::zeroed(stream, a.route.ids.len())?,
                weights: DeviceBuffer::zeroed(stream, a.route.weights.len())?,
                routed_h: DeviceBuffer::zeroed(stream, a.rh.len())?,
                shared_h: DeviceBuffer::zeroed(stream, a.sh_h.len())?,
                nodes: MtpNode::ALL
                    .iter()
                    .map(|n| DeviceBuffer::zeroed(stream, MTP_ROWS * n.width()))
                    .collect::<Result<_, _>>()?,
            })
        } else {
            None
        };
        Ok(())
    }

    /// A new sequence: no last walk to read an own row from, and no position
    /// the store holds for it. The store's rows stay; a walk writes its rows'
    /// keys before its attention reads them, and none may start past the
    /// count.
    pub(super) fn forget(&mut self) {
        self.last = None;
        self.held = 0;
    }

    /// A cut of the target to `pos`: the store holds no position past it —
    /// the rows past it belong to the branch the cut dropped, and a walk
    /// writes its own rows' keys before its attention reads them.
    pub(super) fn cut(&mut self, pos: u32) {
        self.held = self.held.min(pos as usize);
    }

    /// Positions of the current sequence the store holds: a walk may start
    /// at or below it, never past it.
    #[must_use]
    pub fn held(&self) -> usize {
        self.held
    }

    fn arena(&mut self) -> Result<&mut MtpArena, GpuError> {
        self.a
            .as_mut()
            .ok_or(GpuError::state(WHAT, "the program's arena (Mtp38::arm)"))
    }

    fn arena_ref(&self) -> Result<&MtpArena, GpuError> {
        self.a
            .as_ref()
            .ok_or(GpuError::state(WHAT, "the program's arena (Mtp38::arm)"))
    }

    /// The captured walks' node counts, by rows, feed and head.
    #[must_use]
    pub fn graph_nodes(&self) -> Vec<(usize, bool, MtpHead, usize)> {
        self.graphs
            .iter()
            .map(|(k, g)| (k.m, k.own, k.head, g.node_count()))
            .collect()
    }

    /// Run one walk of the program (module doc) over `feed`'s rows into
    /// `head`, eager or replayed, and read its tokens back. `target` is the
    /// target's streams a [`MtpHidden::Target`] feed copies from. Refused by
    /// name before anything moves: rows outside 1..=[`MTP_ROWS`] (a captured
    /// walk 1..=[`MTP_GRAPH_ROWS`]), a hidden slice of other than the rows'
    /// values, a token past the vocabulary, positions past the store, an
    /// own row before any walk of this sequence, a walk starting past the
    /// positions the store holds for the sequence, a row-list head on a
    /// full-head load, a captured walk with the taps armed, a store walk
    /// ([`MtpMode::Store`] reads nothing back). A fault a launch raised is
    /// [`GpuError::Fault`], not a token.
    pub(super) fn run(
        &mut self,
        c: &MtpCtx<'_>,
        target: Option<Target38<'_>>,
        feed: MtpFeed<'_>,
        head: MtpHead,
        mode: MtpMode,
    ) -> Result<MtpDraft, GpuError> {
        let stream = c.gpu.stream();
        refuse_store(
            mode,
            "a walk with a head to read back (MtpMode::Store writes the store alone)",
        )?;
        let m = self.run_walk(c, target, feed, head, mode)?;
        let a = self.arena_ref()?;
        read_draft(a, stream, m)
    }

    /// One walk of the program over `feed`'s rows into `head`, eager,
    /// replayed from its capture or through the store's append alone
    /// ([`MtpMode::Store`]), with no readback: [`Mtp38::run`]'s checks and
    /// launches, the arena's last-walk bookkeeping moved and the walk's rows
    /// returned. A store walk leaves no last walk: an own row after it, and
    /// the readbacks of its rows, are refused by name. A fault the walk
    /// raised stays on the fault word, which the next readback names
    /// ([`read_draft`], a chain's).
    pub(super) fn run_walk(
        &mut self,
        c: &MtpCtx<'_>,
        target: Option<Target38<'_>>,
        feed: MtpFeed<'_>,
        head: MtpHead,
        mode: MtpMode,
    ) -> Result<usize, GpuError> {
        let stream = c.gpu.stream();
        if head == MtpHead::Rows && self.head.rows.is_none() {
            return Err(GpuError::state(
                WHAT,
                "a row-list head: this draft was opened with the full head",
            ));
        }
        let ctx = self.ctx;
        let taps_on = self.arena_ref()?.taps.is_some();
        if mode == MtpMode::Graph && taps_on {
            return Err(GpuError::state(WHAT, "the taps off for a captured walk"));
        }
        let (m, own, pos0) = match feed {
            MtpFeed::Rows { tokens, pos0, .. } => (tokens.len(), false, pos0),
            MtpFeed::Own { pos0 } => (1, true, pos0),
        };
        let most = match mode {
            MtpMode::Eager => MTP_ROWS,
            MtpMode::Graph => MTP_GRAPH_ROWS,
            MtpMode::Store => MTP_STORE_ROWS,
        };
        if !(1..=most).contains(&m) {
            return Err(GpuError::shape(
                WHAT,
                format!("a walk of {m} rows; the {mode:?} walk takes 1..={most}"),
            ));
        }
        if pos0 as usize + m > ctx {
            return Err(GpuError::shape(
                WHAT,
                format!("{m} rows from position {pos0}, in a store of {ctx}"),
            ));
        }
        let (vocab, held, walked) = {
            let vocab = self.arena_ref()?.vocab;
            (vocab, self.held, self.last.is_some())
        };
        if own && !walked {
            return Err(GpuError::state(WHAT, "a walk before the draft's own row"));
        }
        if pos0 as usize > held {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a walk from position {pos0}: the draft's store holds this sequence's \
                     positions below {held}, and the walk would read the keys between"
                ),
            ));
        }
        let mut src = None;
        if let MtpFeed::Rows { tokens, hidden, .. } = feed {
            super::refuse_past_vocab(WHAT, tokens, vocab)?;
            match hidden {
                MtpHidden::Host(v) if v.len() != m * WIDE => {
                    return Err(GpuError::shape(
                        WHAT,
                        format!("{} hidden values for {m} rows of {WIDE}", v.len()),
                    ));
                }
                MtpHidden::Host(_) => {}
                MtpHidden::Target { walk, first } => {
                    let t = target.ok_or(GpuError::state(WHAT, "the target's streams to copy"))?;
                    if (first + m) * WIDE > t.buf.len() {
                        return Err(GpuError::shape(
                            WHAT,
                            format!(
                                "rows {first}..{} of the target's {} streams' rows",
                                first + m,
                                t.buf.len() / WIDE
                            ),
                        ));
                    }
                    let at = pos0.checked_sub(1).ok_or_else(|| {
                        GpuError::shape(
                            WHAT,
                            format!(
                                "the {walk:?} arena's rows for a walk at position 0: the target \
                                 holds no row before it (a host zero row is position 0's)"
                            ),
                        )
                    })?;
                    if !t.held.holds(first, m, at) {
                        return Err(GpuError::shape(
                            WHAT,
                            format!(
                                "rows {first}..{} of the {walk:?} arena as positions {at}..{} for \
                                 a walk from {pos0}: the arena holds {}",
                                first + m,
                                at as usize + m,
                                t.held.shown()
                            ),
                        ));
                    }
                    src = Some((t.buf, first));
                }
            }
        }
        let a = self.arena()?;
        match feed {
            MtpFeed::Rows { tokens, hidden, .. } => {
                put_input(a.inbox.host_mut()?, tokens, pos0)?;
                a.inbox.upload(stream, IN_IDS + m)?;
                // SAFETY: `h_in` holds MTP_ROWS · WIDE values and the store
                // arena's `h` MTP_STORE_ROWS · WIDE, and m is at most the
                // mode's rows (checked above); the window lives for this copy.
                let mut h = unsafe {
                    match mode {
                        MtpMode::Store => f32_view(&a.st.h, 0, m * WIDE),
                        MtpMode::Eager | MtpMode::Graph => f32_view(&a.h_in, 0, m * WIDE),
                    }
                };
                match (hidden, src) {
                    (MtpHidden::Host(v), _) => h.copy_from_host(stream, v)?,
                    (MtpHidden::Target { .. }, Some((t, first))) => {
                        // SAFETY: rows first .. first + m lie inside `t` (checked
                        // above); the window lives for this copy.
                        let src = unsafe { f32_view(t, first * WIDE, m * WIDE) };
                        h.copy_from_device_async(&src, stream)?;
                    }
                    (MtpHidden::Target { .. }, None) => {
                        return Err(GpuError::state(WHAT, "the target's streams to copy"));
                    }
                }
            }
            MtpFeed::Own { .. } => {
                put_input(a.inbox.host_mut()?, &[], pos0)?;
                a.inbox.upload(stream, IN_IDS)?;
            }
        }
        let key = MtpKey { m, own, head };
        match mode {
            MtpMode::Eager => self.walk(c, key)?,
            MtpMode::Store => self.store_walk(c, m, own)?,
            MtpMode::Graph => {
                let mut graphs = std::mem::take(&mut self.graphs);
                let found = graphs.iter().position(|(k, _)| *k == key);
                let r = match found {
                    Some(i) => Ok(i),
                    None => Graph::capture(stream, |_| self.walk(c, key)).map(|g| {
                        graphs.push((key, g));
                        graphs.len() - 1
                    }),
                };
                let launched = r.and_then(|i| graphs[i].1.launch(stream));
                self.graphs = graphs;
                launched?;
            }
        }
        // A store walk wrote no own row, no streams past its attention's
        // input and no head: nothing of it is there to read.
        self.last = (mode != MtpMode::Store).then_some((m, head));
        self.held = pos0 as usize + m;
        Ok(m)
    }

    /// One window's chain — the draft's proposal, one readback: `refresh`'s
    /// walk (a [`MtpFeed::Rows`] feed, its rows the target's kept rows with
    /// the target's hidden rows), whose last row proposes the first id, then
    /// `own` walks of [`MtpFeed::Own`], each reading the walk before it on
    /// the card and proposing one id more, into `head`. Each walk's last
    /// row's id and probability bits are copied into the arena's chain
    /// buffer as it ends and the whole chain is read back once, the last
    /// walk's fault word with it: the tokens are the proposal's ids in
    /// order (1 + `own` of them). Refused as [`Mtp38::run`] refuses each
    /// walk, before the first one moves anything the chain's feeds name.
    pub(super) fn run_chain(
        &mut self,
        c: &MtpCtx<'_>,
        target: Option<Target38<'_>>,
        refresh: MtpFeed<'_>,
        own: usize,
        head: MtpHead,
        mode: MtpMode,
    ) -> Result<MtpDraft, GpuError> {
        let stream = c.gpu.stream();
        refuse_store(
            mode,
            "a chain's walks with a head (MtpMode::Store writes the store alone)",
        )?;
        let MtpFeed::Rows { tokens, pos0, .. } = refresh else {
            return Err(GpuError::shape(
                WHAT,
                format!("a chain's refresh {refresh:?}: the rows the target kept"),
            ));
        };
        if own >= MTP_GRAPH_ROWS {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{own} own walks; a chain proposes at most {}",
                    MTP_GRAPH_ROWS - 1
                ),
            ));
        }
        // Every walk's positions, checked before the first moves anything:
        // the refresh's rows, then one own row at each position after them.
        let (m, end) = (tokens.len(), pos0 as usize + tokens.len());
        if end + own > self.ctx {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a chain of {} rows from position {pos0}: its last own row runs past a store \
                     of {}",
                    m + own,
                    self.ctx
                ),
            ));
        }
        let feeds: Vec<MtpFeed<'_>> = std::iter::once(refresh)
            .chain((0..own).map(|i| {
                MtpFeed::Own {
                    pos0: u32::try_from(end + i)
                        .expect("a chain's positions lie below its store (checked above)"),
                }
            }))
            .collect();
        let mut walks = Vec::with_capacity(feeds.len());
        for (i, feed) in feeds.iter().enumerate() {
            let m = self.run_walk(c, target, *feed, head, mode)?;
            let a = self.arena_ref()?;
            // SAFETY: words m − 1 (the last row's id) and 2m − 1 (its
            // probability's bits) of the readback's 2·MTP_ROWS + 2 words,
            // and words 2i and 2i + 1 of the chain's 2·MTP_GRAPH_ROWS + 2
            // (i + 1 <= MTP_GRAPH_ROWS walks); the windows live for these
            // copies.
            let (mut id, mut p) = unsafe {
                (
                    param_view::<u32>(&a.chain, 2 * i, 1),
                    param_view::<u32>(&a.chain, 2 * i + 1, 1),
                )
            };
            // SAFETY: as above, of `out`.
            let (w_id, w_p) = unsafe {
                (
                    param_view::<u32>(&a.out, m - 1, 1),
                    param_view::<u32>(&a.out, 2 * m - 1, 1),
                )
            };
            id.copy_from_device_async(&w_id, stream)?;
            p.copy_from_device_async(&w_p, stream)?;
            walks.push(m);
        }
        // The last walk's fault word and site mask close the chain: the word
        // is the card's own, so a fault any walk of the chain raised is on
        // it (the argmax kernel copies it after its launch).
        let m = *walks.last().expect("a chain runs its refresh");
        let a = self.arena_ref()?;
        // SAFETY: words 2m and 2m + 1 of the readback, and the chain's last
        // two words; the windows live for these copies.
        let (mut word, mut sites) = unsafe {
            (
                param_view::<u32>(&a.chain, 2 * MTP_GRAPH_ROWS, 1),
                param_view::<u32>(&a.chain, 2 * MTP_GRAPH_ROWS + 1, 1),
            )
        };
        // SAFETY: as above, of `out`.
        let (w_word, w_sites) = unsafe {
            (
                param_view::<u32>(&a.out, 2 * m, 1),
                param_view::<u32>(&a.out, 2 * m + 1, 1),
            )
        };
        word.copy_from_device_async(&w_word, stream)?;
        sites.copy_from_device_async(&w_sites, stream)?;
        let out = a.chain.to_host_vec(stream)?;
        let n = walks.len();
        let (Some(&word), Some(&sites)) =
            (out.get(2 * MTP_GRAPH_ROWS), out.get(2 * MTP_GRAPH_ROWS + 1))
        else {
            return Err(GpuError::state(WHAT, "a chain readback of 2n + 2 words"));
        };
        if let Some(fault) = Fault::from_words(word, sites) {
            return Err(GpuError::fault(WHAT, fault));
        }
        // Walk i's id and probability bits sit at words 2i and 2i + 1.
        Ok(MtpDraft {
            tokens: out[..2 * n].iter().step_by(2).copied().collect(),
            p: out[1..2 * n]
                .iter()
                .step_by(2)
                .map(|&b| f32::from_bits(b))
                .collect(),
        })
    }

    /// The walk itself: every launch of the module doc's list, enqueued on
    /// the card's stream (what a capture records).
    fn walk(&mut self, c: &MtpCtx<'_>, key: MtpKey) -> Result<(), GpuError> {
        let MtpKey { m, own, head } = key;
        let (embd, head_out) = self.borrowed(c.tw)?;
        let Mtp38 {
            w,
            store,
            head: hm,
            index,
            ctx,
            a,
            q6k_ids,
            ..
        } = self;
        let a = a
            .as_mut()
            .ok_or(GpuError::state(WHAT, "the program's arena (Mtp38::arm)"))?;
        let (gpu, k) = (c.gpu, c.k);
        let stream = gpu.stream();
        let sink = gpu.layer_sink(*index as usize)?;
        let n = Names::of(*index);
        let cx = Ctx38 {
            gpu,
            w,
            k,
            eps: c.eps,
            table: c.table,
            ctx: *ctx,
        };
        let capturing = crate::capturing(stream)?;
        let (h, ff, used) = (geo::HIDDEN, geo::FF, geo::N_USED);

        // The embedding rows and the input pack.
        let DevWeight::Q8_0 { qs: eqs, d: ed, .. } = embd else {
            return Err(GpuError::tensor(WHAT, "token_embd.weight", "Q8_0 planes"));
        };
        // SAFETY: words IN_IDS .. IN_IDS + m of the inbox's IN_IDS +
        // MTP_STORE_ROWS words (m <= MTP_ROWS <= MTP_STORE_ROWS); the window
        // lives for the embedding's enqueue.
        let ids = unsafe { param_view::<u32>(a.inbox.dev(), IN_IDS, m) };
        k.q38.enqueue_embed_rows(
            stream,
            EmbedQ8Args {
                qs: eqs,
                d: ed,
                ids: if own { &a.own_id } else { &ids },
                pos0: &a.pos0,
                first: 0,
                fault: sink,
                y: &mut a.emb,
                pos: &mut a.pos,
                n_keys: &mut a.n_keys,
            },
        )?;
        k.mtp.enqueue_mtp_input(
            stream,
            MtpInputArgs {
                e: &a.emb,
                h: if own { &a.own_h } else { &a.h_in },
                enorm: f32_gain(w, &n.enorm)?,
                hnorm: f32_gain(w, &n.hnorm)?,
                hidden: h,
                eps: c.eps,
                m,
                fault: sink,
                out: &mut a.pack,
            },
        )?;
        let (eqs, ed) = q8(w, &n.eh)?;
        let cols = geo::STREAMS * m;
        let mut at = 0;
        while at < cols {
            let run = EH_COLS.min(cols - at);
            // SAFETY: columns at .. at + run of the pack (2·hidden values a
            // column) and of the streams (hidden a column) lie inside the
            // arena's MTP_ROWS rows (cols = 4m <= 4·MTP_ROWS); the windows
            // live for this enqueue.
            let (x, mut y) = unsafe {
                (
                    f32_view(&a.pack, at * 2 * h, run * 2 * h),
                    f32_view(&a.res, at * h, run * h),
                )
            };
            gpu.q8f32().enqueue_q8_0_gemv_mcol(
                stream,
                Q8_0GemvMcolArgs {
                    qs: eqs,
                    d: ed,
                    x: &x,
                    m: run,
                    out: GemvOut::TokenMajor,
                    y: &mut y,
                },
            )?;
            at += run;
        }
        if let (Some(t), false) = (a.taps.as_mut(), capturing) {
            // SAFETY: the first m rows of the tap and of the streams, both
            // MTP_ROWS · WIDE values; the windows live for this copy.
            let (mut dst, src) =
                unsafe { (f32_view(&t.eh, 0, m * WIDE), f32_view(&a.res, 0, m * WIDE)) };
            dst.copy_from_device_async(&src, stream)?;
        }

        // The attention site.
        cx.mix(
            &n.attn_site,
            &mut a.res,
            Before::Plain,
            m,
            sink,
            &mut a.hc,
            &mut a.mixed,
        )?;
        if let (Some(t), false) = (a.taps.as_mut(), capturing) {
            t.node(stream, MtpNode::AttnIn, &a.mixed, m)?;
        }
        cx.q8_gemv(&n.q, &a.mixed, m, &mut a.qg)?;
        cx.q8_gemv(&n.k, &a.mixed, m, &mut a.k)?;
        cx.q8_gemv(&n.v, &a.mixed, m, &mut a.v)?;
        // The draft's store is f16: the family's shared append runs its f16
        // arm.
        let (kc, vc) = store.f16_mut("qwen38::mtp")?;
        k.neox.enqueue_head_norm_neox_append_256(
            stream,
            PartialNeoxArgs {
                qg: &a.qg,
                q: &mut a.q,
                k: &mut a.k,
                v: &a.v,
                gq: f32_gain(w, &n.q_norm)?,
                gk: f32_gain(w, &n.k_norm)?,
                table: c.table,
                pos: &a.pos,
                eps: c.eps,
                n_head: geo::N_HEAD,
                n_kv: geo::N_KV,
                ctx: *ctx,
                m,
                fault: sink,
                cache_k: kc,
                cache_v: vc,
            },
        )?;
        let (kc, vc) = store.f16("qwen38::mtp")?;
        k.flash.enqueue_pass_256_p4(
            stream,
            GqaArgs {
                q: &a.q,
                kc,
                vc,
                n_keys: &a.n_keys,
                scale: ATTN_SCALE_256,
                n_kv: geo::N_KV,
                ctx: *ctx,
                m,
                part_v: &mut a.part_v,
                part_ms: &mut a.part_ms,
                fault: sink,
                y: &mut a.flash,
            },
            geo::N_HEAD,
            MMA,
        )?;
        k.q38.enqueue_out_gate(
            stream,
            OutGateArgs {
                attn: &a.flash,
                qg: &a.qg,
                n_head: geo::N_HEAD,
                m,
                fault: sink,
                y: &mut a.attn,
            },
        )?;
        cx.q8_gemv(&n.out, &a.attn, m, &mut a.y)?;
        if let (Some(t), false) = (a.taps.as_mut(), capturing) {
            t.node(stream, MtpNode::AttnGated, &a.attn, m)?;
            t.node(stream, MtpNode::AttnOut, &a.y, m)?;
        }

        // The feed-forward site: the router, the routed slots, the shared
        // expert.
        cx.mix(
            &n.ffn_site,
            &mut a.res,
            Before::Combine { y: &a.y },
            m,
            sink,
            &mut a.hc,
            &mut a.ffn_x,
        )?;
        if let (Some(t), false) = (a.taps.as_mut(), capturing) {
            t.node(stream, MtpNode::AttnCombined, &a.res, m)?;
            t.node(stream, MtpNode::FfnIn, &a.ffn_x, m)?;
        }
        k.router.enqueue_fused(
            stream,
            f32_tensor(w, &n.router)?,
            &a.ffn_x,
            m,
            sink,
            &mut a.route,
        )?;
        if let (Some(t), false) = (a.taps.as_mut(), capturing) {
            t.logits.copy_from_device_async(&a.route.logits, stream)?;
            t.ids.copy_from_device_async(&a.route.ids, stream)?;
            t.weights.copy_from_device_async(&a.route.weights, stream)?;
        }
        let pitch = a.route.dims().slots();
        k.handoff.enqueue_places_cols(
            stream,
            &Places {
                ids: &a.route.ids,
                map: &a.id_map,
                row_off: 0,
                n_expert: geo::EXPERTS,
            },
            pitch,
            m,
            sink,
            &mut a.sel,
        )?;
        let slots = m * used;
        for (name, out) in [(&n.gate_exps, &mut a.rg), (&n.up_exps, &mut a.ru)] {
            let (qs, d) = q8(w, name)?;
            gpu.q8f32().enqueue_q8_0_gemv_sel_f32(
                stream,
                &Q8_0SelArgs {
                    qs,
                    d,
                    x: &a.ffn_x,
                    sel: &a.sel,
                    n_slots: slots,
                    rows_per_expert: ff,
                    slots_per_col: used,
                },
                sink,
                out,
            )?;
        }
        gpu.elem()
            .enqueue_swiglu(stream, &a.rg, &a.ru, slots * ff, &mut a.rh)?;
        let (qs, d) = q8(w, &n.down_exps)?;
        gpu.q8f32().enqueue_q8_0_gemv_sel_f32(
            stream,
            &Q8_0SelArgs {
                qs,
                d,
                x: &a.rh,
                sel: &a.sel,
                n_slots: slots,
                rows_per_expert: h,
                slots_per_col: 1,
            },
            sink,
            &mut a.rdown,
        )?;
        k.q38.enqueue_card_acc(
            stream,
            CardAccArgs {
                down: &a.rdown,
                w: &a.route.weights,
                sel: &a.sel,
                n: h,
                m,
                n_card: geo::EXPERTS,
                fault: sink,
                acc: &mut a.acc,
            },
        )?;
        cx.q8_gemv(&n.gate_sh, &a.ffn_x, m, &mut a.sh_g)?;
        cx.q8_gemv(&n.up_sh, &a.ffn_x, m, &mut a.sh_u)?;
        gpu.elem()
            .enqueue_swiglu(stream, &a.sh_g, &a.sh_u, ff * m, &mut a.sh_h)?;
        cx.q8_gemv(&n.down_sh, &a.sh_h, m, &mut a.sh_y)?;
        k.q38.enqueue_shared_add(
            stream,
            SharedAddArgs {
                hsum: &a.acc,
                sh: &a.sh_y,
                w: &a.route.weights,
                slot: used,
                slots: pitch,
                n: h,
                m,
                fault: sink,
                y: &mut a.y,
            },
        )?;
        if let (Some(t), false) = (a.taps.as_mut(), capturing) {
            t.node(stream, MtpNode::Routed, &a.acc, m)?;
            t.node(stream, MtpNode::Shared, &a.sh_y, m)?;
            t.node(stream, MtpNode::FfnOut, &a.y, m)?;
            t.routed_h.copy_from_device_async(&a.rh, stream)?;
            t.shared_h.copy_from_device_async(&a.sh_h, stream)?;
        }

        // The head: its site's mix (the block's combine first), the
        // projection and the argmax.
        cx.mix(
            &n.head_site,
            &mut a.res,
            Before::Combine { y: &a.y },
            m,
            sink,
            &mut a.hc,
            &mut a.head_x,
        )?;
        if let (Some(t), false) = (a.taps.as_mut(), capturing) {
            t.node(stream, MtpNode::HeadIn, &a.head_x, m)?;
        }
        let full_rows = match &head_out {
            HeadOut::Q8_0 { d, .. } => d.rows(),
            HeadOut::Q6K { w } => w.rows(),
        };
        let (rows, map) = match head {
            MtpHead::Full => (full_rows, None),
            MtpHead::Rows => (
                hm.rows
                    .ok_or(GpuError::state(WHAT, "a row list for the head"))?,
                Some(&hm.ids),
            ),
        };
        // SAFETY: the logits hold MTP_ROWS · vocab values and rows <= vocab
        // (the full head's, or a list no longer than the vocabulary, refused
        // at open otherwise), m <= MTP_ROWS; the window lives for the
        // projection and the argmax.
        let mut lg = unsafe { f32_view(&a.logits, 0, rows * m) };
        match head_out {
            HeadOut::Q8_0 { qs, d } => match map {
                None => gpu
                    .q8f32()
                    .enqueue_q8_0_gemv(stream, qs, d, &a.head_x, m, &mut lg)?,
                Some(ids) => gpu.q8f32().enqueue_q8_0_gemv_ids(
                    stream,
                    Q8_0GemvIdsArgs {
                        qs,
                        d,
                        ids,
                        rows,
                        x: &a.head_x,
                        m,
                        y: &mut lg,
                    },
                    sink,
                )?,
            },
            HeadOut::Q6K { w: out_w } => {
                let act = a.head_act.as_mut().ok_or_else(|| {
                    GpuError::state(WHAT, "the head's q8_1 activation (a Q6_K head's arena)")
                })?;
                gpu.enqueue_quantize_q8_1_head(&a.head_x, act)?;
                q6k_ids.enqueue_gemv_q6k_ids(
                    stream,
                    Q6kIdsArgs {
                        w: out_w,
                        act,
                        map,
                        no_map: &a.no_map,
                        rows,
                        m,
                        y: &mut lg,
                        fault: sink,
                    },
                )?;
            }
        }
        k.mtp.enqueue_argmax_p_rows_fault(
            stream,
            ArgmaxPArgs {
                x: &lg,
                n: rows,
                m,
                map,
                no_map: &a.no_map,
                vocab: a.vocab,
                fault: sink,
                out: &mut a.out,
                done: &mut a.done,
            },
        )?;

        // The last row, for the next own walk.
        // SAFETY: word m − 1 of the readback's 2·MTP_ROWS + 2 and row m − 1
        // of the streams' MTP_ROWS rows (1 <= m <= MTP_ROWS); the windows
        // live for the two copies.
        let (id, row) = unsafe {
            (
                param_view::<u32>(&a.out, m - 1, 1),
                f32_view(&a.res, (m - 1) * WIDE, WIDE),
            )
        };
        a.own_id.copy_from_device_async(&id, stream)?;
        a.own_h.copy_from_device_async(&row, stream)?;
        Ok(())
    }

    /// A store walk over `m` rows (module doc): the embedding and the pack,
    /// `eh_proj`, the attention site's mix, q, k and v, the norm and
    /// append, every launch over the rows at once in the store arena, on
    /// the card's stream. The rows' tokens are the inbox's and their hidden
    /// rows the store arena's `h`, or the draft's own last row when `own`.
    fn store_walk(&mut self, c: &MtpCtx<'_>, m: usize, own: bool) -> Result<(), GpuError> {
        let (embd, _) = self.borrowed(c.tw)?;
        let Mtp38 {
            w,
            store,
            index,
            ctx,
            a,
            ..
        } = self;
        let a = a
            .as_mut()
            .ok_or(GpuError::state(WHAT, "the program's arena (Mtp38::arm)"))?;
        let (gpu, k) = (c.gpu, c.k);
        let stream = gpu.stream();
        let sink = gpu.layer_sink(*index as usize)?;
        let n = Names::of(*index);
        let (h, cols) = (geo::HIDDEN, geo::STREAMS * m);
        let s = &mut a.st;

        // The embedding rows and the input pack.
        let DevWeight::Q8_0 { qs: eqs, d: ed, .. } = embd else {
            return Err(GpuError::tensor(WHAT, "token_embd.weight", "Q8_0 planes"));
        };
        // SAFETY: words IN_IDS .. IN_IDS + m of the inbox's IN_IDS +
        // MTP_STORE_ROWS words (m <= MTP_STORE_ROWS); the window lives for
        // the embedding's enqueue.
        let ids = unsafe { param_view::<u32>(a.inbox.dev(), IN_IDS, m) };
        k.q38.enqueue_embed_rows(
            stream,
            EmbedQ8Args {
                qs: eqs,
                d: ed,
                ids: if own { &a.own_id } else { &ids },
                pos0: &a.pos0,
                first: 0,
                fault: sink,
                y: &mut s.emb,
                pos: &mut s.pos,
                n_keys: &mut s.n_keys,
            },
        )?;
        k.mtp.enqueue_mtp_input(
            stream,
            MtpInputArgs {
                e: &s.emb,
                h: if own { &a.own_h } else { &s.h },
                enorm: f32_gain(w, &n.enorm)?,
                hnorm: f32_gain(w, &n.hnorm)?,
                hidden: h,
                eps: c.eps,
                m,
                fault: sink,
                out: &mut s.pack,
            },
        )?;

        // `eh_proj` over the packed columns into the streams.
        k.g32
            .enqueue_quantize_gemm32(stream, &s.pack, cols, &mut s.pack_q, sink)?;
        k.gemm
            .enqueue_route_dense(stream, cols, &mut s.packed, gpu.unlabelled_sink())?;
        let (qs, d) = q8(w, &n.eh)?;
        k.g32.enqueue_gemm32(
            stream,
            Gemm32Args {
                w: Gemm32Weight::Q8_0Plane { qs, d },
                rows_per_expert: qs.rows(),
                act: &s.pack_q,
                route: &s.packed,
                input: GemmInput::PerSlot,
                y: &mut s.res,
            },
        )?;

        // The attention site's mix, then q, k and v over the rows.
        k.gemm
            .enqueue_route_dense(stream, m, &mut s.rows, gpu.unlabelled_sink())?;
        let site = &n.attn_site;
        let (down_qs, down_d) = q8(w, &site.down)?;
        let (up_qs, up_d) = q8(w, &site.up)?;
        let inject = match &site.inject {
            Some(t) => Some(f32_tensor(w, t)?),
            None => None,
        };
        k.hcw.enqueue_mix(
            stream,
            WideMixArgs {
                res: &mut s.res,
                before: Before::Plain,
                w: SiteWeights {
                    gamma: f32_gain(w, &site.norm)?,
                    down_qs,
                    down_d,
                    up_qs,
                    up_d,
                    inject,
                },
                eps: c.eps,
                m,
                fault: sink,
                scratch: &mut s.hc,
                gemm: &k.g32,
                dense: &s.rows,
                mixed: &mut s.mixed,
            },
        )?;
        k.g32
            .enqueue_quantize_gemm32(stream, &s.mixed, m, &mut s.mixed_q, sink)?;
        for (name, y) in [(&n.q, &mut s.qg), (&n.k, &mut s.k), (&n.v, &mut s.v)] {
            let (qs, d) = q8(w, name)?;
            k.g32.enqueue_gemm32(
                stream,
                Gemm32Args {
                    w: Gemm32Weight::Q8_0Plane { qs, d },
                    rows_per_expert: qs.rows(),
                    act: &s.mixed_q,
                    route: &s.rows,
                    input: GemmInput::PerSlot,
                    y,
                },
            )?;
        }
        // The draft's store is f16: the family's shared append runs its f16
        // arm.
        let (kc, vc) = store.f16_mut("qwen38::mtp")?;
        k.neox.enqueue_head_norm_neox_append_256(
            stream,
            PartialNeoxArgs {
                qg: &s.qg,
                q: &mut s.q,
                k: &mut s.k,
                v: &s.v,
                gq: f32_gain(w, &n.q_norm)?,
                gk: f32_gain(w, &n.k_norm)?,
                table: c.table,
                pos: &s.pos,
                eps: c.eps,
                n_head: geo::N_HEAD,
                n_kv: geo::N_KV,
                ctx: *ctx,
                m,
                fault: sink,
                cache_k: kc,
                cache_v: vc,
            },
        )
    }

    /// The streams of the last walk's rows: the layer's output, `l_out`.
    /// Blocking; gate use.
    pub fn l_out(&self, stream: &CudaStream) -> Result<Vec<f32>, GpuError> {
        let a = self.arena_ref()?;
        let (m, _) = self.last.ok_or(GpuError::state(WHAT, "a walk to read"))?;
        let mut v = a.res.to_host_vec(stream)?;
        v.truncate(m * WIDE);
        Ok(v)
    }

    /// The last walk's head logits, `[row][m]` as the gemv writes them, and
    /// the head's rows. Blocking; gate use.
    pub fn logits(&self, stream: &CudaStream) -> Result<(Vec<f32>, usize), GpuError> {
        let a = self.arena_ref()?;
        let (m, head) = self.last.ok_or(GpuError::state(WHAT, "a walk to read"))?;
        let rows = match (head, self.head.rows) {
            (MtpHead::Rows, Some(n)) => n,
            _ => a.vocab,
        };
        let mut v = a.logits.to_host_vec(stream)?;
        v.truncate(rows * m);
        Ok((v, rows))
    }

    /// The last eager walk's taps. Blocking; refused when they are not
    /// armed.
    pub fn taps(&self, stream: &CudaStream) -> Result<MtpTaps, GpuError> {
        let a = self.arena_ref()?;
        let (m, _) = self.last.ok_or(GpuError::state(WHAT, "a walk to read"))?;
        let t = a
            .taps
            .as_ref()
            .ok_or(GpuError::state(WHAT, "armed taps (Mtp38::set_taps)"))?;
        let dims = a.route.dims();
        let mut eh = t.eh.to_host_vec(stream)?;
        eh.truncate(m * WIDE);
        let nodes = MtpNode::ALL
            .iter()
            .zip(&t.nodes)
            .map(|(&n, b)| {
                let mut v = b.to_host_vec(stream)?;
                v.truncate(m * n.width());
                Ok((n, v))
            })
            .collect::<Result<_, GpuError>>()?;
        let mut routed_h = t.routed_h.to_host_vec(stream)?;
        routed_h.truncate(m * geo::N_USED * geo::FF);
        let mut shared_h = t.shared_h.to_host_vec(stream)?;
        shared_h.truncate(m * geo::FF);
        Ok(MtpTaps {
            eh,
            logits: t.logits.to_host_vec(stream)?,
            ids: t.ids.to_host_vec(stream)?,
            weights: t.weights.to_host_vec(stream)?,
            routed_h,
            shared_h,
            logits_row: dims.logits(),
            slots_row: dims.slots(),
            nodes,
        })
    }
}

/// Refuse by name a store walk ([`MtpMode::Store`]) where the head's rows
/// are read back, `why` naming the call.
fn refuse_store(mode: MtpMode, why: &'static str) -> Result<(), GpuError> {
    if mode == MtpMode::Store {
        return Err(GpuError::state(WHAT, why));
    }
    Ok(())
}

/// The readback of a walk of `m` rows: the tokens, then the probabilities'
/// bits, then the fault word and its site mask. Blocking; a raised fault is
/// [`GpuError::Fault`].
fn read_draft(a: &MtpArena, stream: &CudaStream, m: usize) -> Result<MtpDraft, GpuError> {
    let out = a.out.to_host_vec(stream)?;
    let (Some(&word), Some(&sites)) = (out.get(2 * m), out.get(2 * m + 1)) else {
        return Err(GpuError::state(WHAT, "a readback of 2m + 2 words"));
    };
    if let Some(fault) = Fault::from_words(word, sites) {
        return Err(GpuError::fault(WHAT, fault));
    }
    Ok(MtpDraft {
        tokens: out[..m].to_vec(),
        p: out[m..2 * m].iter().map(|&b| f32::from_bits(b)).collect(),
    })
}

// A walk's rows fit the eh_proj run and the m-column kernels; a store
// walk's inbox holds an eager walk's rows too.
const _: () = assert!(
    MTP_ROWS <= 8 && MTP_GRAPH_ROWS <= MTP_ROWS && EH_COLS == 8 && MTP_ROWS <= MTP_STORE_ROWS
);
