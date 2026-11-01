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
//!   open — the embedding's two Q8_0 planes or BF16 words, the head's two
//!   Q8_0 planes, BF16 words or one Q6_K word plane; the target's `Weights`, which `GpuModel` drops
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
//! last row ([`DraftFeed::Own`], ik's scheme A: its token the draft's argmax,
//! its hidden the draft's own streams). Every launch is an entry the
//! target's chain has, but two of its own (`crate::mtp`), the routed
//! experts' expert-select gemv and a listed head's `q8_0_gemv_ids` (one
//! row) or `q8_0_gemv_ids_mcol`, over a BF16 `output` `bf16_gemv_ids`, and
//! over a Q6_K `output` its quantize and `q6k_gemv_ids` pair
//! (`crate::q6k_ids`):
//! - the target's embedding rows (`embed_rows_q8_0` or `embed_rows_bf16`,
//!   the borrowed `token_embd`), the input pack (`mtp_input`: `rms(e)·enorm` beside
//!   `rms(h)·hnorm` over all four streams, a row's four `[e | h_s]`), and
//!   `nextn.eh_proj` over the `4·m` packed columns into the streams, in runs
//!   of eight columns (the common dense site, `site::resident_gemv_cols`:
//!   `q8_0_gemv_mcol` for Q8_0, the BF16 gemv or tile for BF16, token-major);
//! - the attention site's mix, then dense gated GQA over the store: q (with
//!   each head's gate), k and v, the q/k norm, turn and append
//!   (`head_norm_neox_append_256`), the flash over every stored position
//!   below the row's (`gqa_flash_seg_mma_256_p4`, `gqa_flash_merge_256`: the
//!   file's `compress_ratios[48]` is 0, and ik runs the layer dense), the gate
//!   and the output projection;
//! - the feed-forward site's mix (the attention's combine first), the joined
//!   router, the routed slots' places over the identity map (every expert
//!   is the card's), the routed gate and up and down through the
//!   expert-select gemv of the stacks' form (`site::resident_sel_f32`:
//!   `q8_0_gemv_sel_f32` over Q8_0 planes, `fp8_b128_gemv_sel_f32` over
//!   FP8_B128 ones; the f32 rows, since the down's 640 values are no multiple
//!   of `Q8Act`'s 256), SwiGLU between, the slots'
//!   weighted sum (`ds41_ffn_card_acc_10` over 512 card experts), the shared expert
//!   and its gated sum (`q38_shared_add`);
//! - the head site's mix (`nextn.hc_head_*`, the block's combine first, so
//!   the streams are then the layer's output, `l_out`), the head's
//!   projection — a Q8_0 `output`'s planes ([`DraftHead::Full`]) or its rows
//!   of the list read in place through the map (`q8_0_gemv_ids`, or
//!   `q8_0_gemv_ids_mcol` past one row; [`DraftHead::Rows`]); a Q6_K one
//!   quantizes the head's input rows to q8_1 in the arena first (one launch
//!   a Q8_0 walk has not), then reads its rows, full or listed, through
//!   `q6k_gemv_ids`/`_mcol` — and
//!   `argmax_p_rows_fault`, which names each
//!   row's token and the draft's largest probability among the head's rows;
//! - the last row's token and streams copied beside the arena, where the
//!   next [`DraftFeed::Own`] walk reads them.
//!
//! The store is by position: a walk appends its rows' keys before its
//! attention reads them, and one may start at or below the end of the last
//! walk of the sequence, never past it (it would read keys no walk of the
//! sequence wrote). A new sequence ([`Mtp38::forget`], the body's reset)
//! holds no position and no own row.
//!
//! A store walk ([`WalkMode::Store`], the prompt's warmup) stops after the
//! append: a later row reads a row's keys and values and nothing else of
//! it, so the walk computes them alone. It runs over up to
//! [`MTP_STORE_ROWS`] rows at once through the launches the ubatch walk
//! (`wide38`) runs, in its own buffers:
//! - the embedding rows, a media prompt call's span rows copied over the
//!   rows of their positions (the row the target consumed there in place of
//!   the image token's embedding, widened exactly), and the input pack, as
//!   above;
//! - the pack quantized to 32-value q8 blocks (`quantize_gemm32`) and
//!   `eh_proj` as one 32-value GEMM (`gemm_q8_0p`) over the `4·m` packed
//!   columns, through a one-expert table;
//! - the attention site's wide mix (`HcWideKernels::enqueue_mix`);
//! - the mix quantized, k and v as two GEMMs over the rows, then the key
//!   heads' norm and append alone (`head_norm_neox_append_256` at no query
//!   head): no query or gate is projected, normed or turned.
//!
//! Each op computes a row's values from that row's inputs alone, so a
//! row's keys and values do not depend on the walk it lands in or on its
//! neighbours. A store walk of several resident sequences' rows (the
//! shared `GpuModel::draft_store_slots`, a round's waiting rows) rests on it: each
//! sequence's rows are checked and staged on that sequence as a block of
//! the store arena's rows ([`StoreBlock`]), the launches run once over
//! every block, the embedding a block at its own positions, and each
//! block's norm and append go into its own sequence's store. Against an
//! [`WalkMode::Eager`] walk, which reads the f32
//! rows, the three projections read q8 activations: the two stores agree to
//! the error of that quantization and are not bit-equal.
//!
//! Launches of a walk at `m` rows: [`walk_launches`]. The arena is the
//! program's, allocated once beside the body ([`Mtp38::arm`]) and counted
//! apart from the plan's bytes ([`Mtp38::arena_bytes`]).

use super::body::ATTN_SCALE_256;
use super::body::KvQ8;
use super::plan38::{HcSite, geo, router};
use super::program38::{Ctx38, Kernels38, MMA};
use super::router::gated;
use super::scratch::{
    Append256, IN_IDS, IN_POS0, Inbox, KvPlanes, cells_view, param_view, put_input,
};
use super::scratch38::{PASS_ROWS, VERIFY_ROWS};
use crate::card_slots::CardAcc;
use crate::checkpoint::bytes::{Fail, Put, Take};
use crate::checkpoint::saved::Spans;
use crate::draft::{
    ChainBlock, DraftFeed, DraftHead, Feed, Held, Hidden, Proposal, WalkBounds, WalkMode,
    chain_rows, refuse_rows,
};
use crate::fault::Fault;
use crate::flash_gqa::{partials_ms_len, partials_v_len_256};
use crate::gemm::{GemmAct32, GemmRoute};
use crate::graph::Graph;
use crate::hc_gated::{
    Before, HcMatrix, HcScratch, HcWideScratch, SiteWeights, Streams, WideMixArgs,
};
use crate::host::handoff::Places;
use crate::media::{CallRows, widen_bf16};
use crate::model::lookup::{f32_gain, f32_tensor};
use crate::mtp::{ArgmaxPArgs, MtpInputArgs};
use crate::q6k_ids::{Q6kIdsArgs, Q6kIdsKernels};
use crate::q8f32::{
    Bf16GemvIdsArgs, E4m3GemvArgs, E4m3Scales, E4m3Weight, GemvOut, Mxfp8GemvIdsArgs,
    Q8_0GemvIdsArgs, SelArgs,
};
use crate::q38::{EmbedArgs, EmbedRows, OutGateArgs, SLOTS, SharedAddArgs, W_PITCH};
use crate::room::{Roomed, SharedPages};
use crate::route_softmax::{RouterDims, RouterOut};
use crate::site::{self, WideIn, WideKernels};
use crate::tensor::{Q8Act, Window, WindowMut};
use crate::weights::{DevWeight, Weights};
use crate::{DeviceTensor, Gpu, GpuError};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use gguf::quant::{GgmlType, dequant_row};
use model::arch::models::MtpSource;
use model::arch::models::headpick::HeadSet;
use model::placement::Plan;
use runtime::hc_gated::Geometry;
use std::sync::Arc;

const WHAT: &str = "qwen4exp Mtp38";

/// A target matrix read in place: its name and Q8_0 or MXFP8 code and scale
/// addresses at open, or its BF16 words and a zero
/// ([`planes`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BorrowedPlanes {
    pub name: String,
    /// The code plane's and the scale plane's `CUdeviceptr`.
    pub planes: [u64; 2],
}

/// The target's `output` read in place: its Q8_0 or MXFP8 planes, BF16
/// words, or Q6_K word plane's device addresses, with the
/// row width and rows the gemv reads.
#[derive(Clone, Debug, PartialEq, Eq)]
enum BorrowedHead {
    Q8_0(BorrowedPlanes),
    Mxfp8(BorrowedPlanes),
    Bf16(BorrowedPlanes),
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
    Mxfp8 {
        qs: &'w DeviceTensor<u32>,
        e: &'w DeviceTensor<u32>,
    },
    Q6K {
        w: &'w DeviceTensor<u32>,
    },
    Bf16 {
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
    names: Names,
    /// The draft's own tensors, the widened injects and the joined router.
    w: Weights,
    store: Roomed<KvPlanes>,
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
    /// The rows of the last walk, which the readbacks read; `None` before
    /// a walk of the sequence and after a store walk.
    last: Option<(usize, DraftHead)>,
    /// The next position of the own input in the shared arena; a slot exchange,
    /// cut, store walk or state adoption requires a Rows refresh before Own.
    own_at: Option<usize>,
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
/// them, save the own row a [`DraftFeed::Own`] walk reads, which no call
/// may read across a select: the exchange invalidates it, and a Rows refresh
/// must rewrite it before an own walk.
pub(super) struct DraftSeq38 {
    graphs: Vec<(MtpKey, Graph)>,
    store: Roomed<KvPlanes>,
    last: Option<(usize, DraftHead)>,
    held: usize,
}

impl DraftSeq38 {
    /// Positions of the sequence the store holds ([`Mtp38::held`]'s, parked).
    pub(super) fn held(&self) -> usize {
        self.held
    }
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
    /// not of its read form (`token_embd` Q8_0 or BF16, `output` Q8_0, Q6_K
    /// or BF16, each `[hidden, vocab]`), a listed id at or past the
    /// vocabulary, an upload whose bytes are not the plan's.
    #[allow(
        clippy::too_many_arguments,
        reason = "the card, the target's weights, the draft's file, inputs and plan, the page-cache release and the planned slots (rust-quality R8)"
    )]
    pub fn open(
        gpu: &Gpu,
        target_w: &Weights,
        (draft_file, parts): (&Split, Option<&dyn crate::model::PartsLoad>),
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
        let rows = file_rows(&plan.draft);
        let mut w = match parts {
            Some(parts) => {
                let reading = model::placement::workstation::host_available_read()
                    .map_err(|e| GpuError::plan(WHAT, e))?;
                let (need, room) = parts.host_need(&rows, 0, 0, reading.0)?;
                need.check(&reading).map_err(|e| GpuError::plan(WHAT, e))?;
                parts.load(stream, &rows, 0, card_dontneed, room)?
            }
            None => Weights::load_placed(stream, draft_file, &rows, 0, card_dontneed)?,
        };
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
            HeadSet::Full => HeadMap {
                ids: DeviceBuffer::zeroed(stream, vocab)?,
                rows: None,
            },
            HeadSet::List { ids, .. } => {
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
        // The draft's store in the target's cache format (the plan's), its
        // rows backed as the walks reach them ([`Mtp38::run_walk`]).
        let kv = KvQ8::of(mtp.kv_q8());
        let store = KvPlanes::roomed(&gpu.stream_handle(), (geo::N_KV, ctx, geo::HEAD), kv)?;
        stream.synchronize()?;
        let body = Mtp38 {
            names: Names::of(d.index, &w),
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
            own_at: None,
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
            BorrowedHead::Q8_0(b) | BorrowedHead::Mxfp8(b) | BorrowedHead::Bf16(b) => b.clone(),
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
        let embd = at_open(target, &self.embd, WHAT)?;
        let head = match &self.head_w {
            BorrowedHead::Q8_0(b) => {
                let DevWeight::Q8_0 { qs, d, .. } = at_open(target, b, WHAT)? else {
                    return Err(GpuError::tensor(WHAT, b.name.clone(), "Q8_0 planes"));
                };
                HeadOut::Q8_0 { qs, d }
            }
            BorrowedHead::Mxfp8(b) => {
                let DevWeight::Mxfp8 { qs, e, .. } = at_open(target, b, WHAT)? else {
                    return Err(GpuError::tensor(WHAT, b.name.clone(), "MXFP8 planes"));
                };
                HeadOut::Mxfp8 { qs, e }
            }
            BorrowedHead::Bf16(b) => {
                let DevWeight::Bf16 { w, .. } = at_open(target, b, WHAT)? else {
                    return Err(GpuError::tensor(WHAT, b.name.clone(), "BF16 words"));
                };
                HeadOut::Bf16 { w }
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
            let heads = (0..geo::N_KV)
                .map(|kv| {
                    let at = kv * ctx * head * size_of::<u16>();
                    Ok(Window::<u16>::of(b, at, n * head)?.to_host_vec(stream)?)
                })
                .collect::<Result<Vec<_>, GpuError>>()?;
            Ok((0..n)
                .flat_map(|p| heads.iter().map(move |h| &h[p * head..(p + 1) * head]))
                .flatten()
                .copied()
                .collect())
        };
        let (k, v) = self.store.f16(WHAT)?;
        Ok((plane(k)?, plane(v)?))
    }

    /// The store's rows of the positions below `n` as a sequence state
    /// carries them ([`KvPlanes::spans`]), in the store's format, so a prefix
    /// of the state moves only its own rows.
    pub(super) fn store_spans(&mut self, n: usize) -> Result<Vec<Spans<'_>>, GpuError> {
        self.store.spans((geo::N_KV, self.ctx, geo::HEAD), n)
    }

    /// Back the store's rows of the positions below `n` before a state copy
    /// moves them ([`Room::fit`](crate::room::Room::fit)).
    pub(super) fn fit_store(&mut self, n: usize) -> Result<(), GpuError> {
        self.store.fit(n)
    }

    /// The store's rows of the first `n` positions taken from `from`,
    /// another sequence's side of the draft ([`Roomed::share`]), the store
    /// counted at `n` positions ([`Mtp38::adopt`]): its pages below them
    /// mapped, the page where they end copied. Refused by name, nothing
    /// counted, past the positions `from`'s store holds. Never inside a
    /// capture.
    pub(super) fn share_store(
        &mut self,
        from: &DraftSeq38,
        n: usize,
    ) -> Result<SharedPages, GpuError> {
        if n > from.held {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a share of {n} positions from a store that holds {}",
                    from.held
                ),
            ));
        }
        let moved = self.store.share(&from.store, n)?;
        self.adopt(n)?;
        Ok(moved)
    }

    /// Device bytes of the live sequence's store pages another sequence's
    /// store maps too.
    pub(super) fn store_shared(&self) -> usize {
        self.store.shared()
    }

    /// The store counted at `n` positions, whose rows a sequence state's
    /// put-back wrote ([`Mtp38::store_spans`]): the side of the state adopted
    /// (`Body38::put_draft_rows`). Refused by name past the store.
    pub(super) fn adopt(&mut self, n: usize) -> Result<(), GpuError> {
        if n > self.ctx {
            return Err(GpuError::shape(
                WHAT,
                format!("{n} positions adopted by a store of {}", self.ctx),
            ));
        }
        self.held = n;
        self.own_at = None;
        Ok(())
    }

    /// A new sequence's side of the draft ([`Body38`]'s
    /// [`Slots`](crate::model::Slots)): a zeroed store no walk has captured
    /// over, no positions held. Load-time allocation.
    pub(super) fn new_seq(&self, stream: &Arc<CudaStream>) -> Result<DraftSeq38, GpuError> {
        // The live store's format: the load's one choice.
        let dims = (geo::N_KV, self.ctx, geo::HEAD);
        Ok(DraftSeq38 {
            graphs: Vec::new(),
            store: KvPlanes::roomed(stream, dims, self.store.format())?,
            last: None,
            held: 0,
        })
    }

    /// A sequence's end: the memory of the store's pages past the rows its
    /// walks reached given back (`Room::settle`).
    pub(super) fn settle(&mut self) -> Result<(), GpuError> {
        self.store.settle()
    }

    /// Device bytes the live sequence's store holds: its mapped pages, and
    /// its layout at the load's context.
    pub(super) fn store_room(&self) -> (usize, usize) {
        (self.store.mapped(), self.store.bytes())
    }

    /// Device bytes the live sequence's store holds once every page is
    /// backed.
    pub(super) fn store_whole(&self) -> usize {
        self.store.whole()
    }

    /// Exchange the live sequence's side of the draft with `seq`'s: pointer
    /// moves only, so each sequence's captures keep addressing its own
    /// store. The shared own input is invalidated; a Rows refresh must precede Own.
    pub(super) fn swap_slot(&mut self, seq: &mut DraftSeq38) {
        std::mem::swap(&mut self.store, &mut seq.store);
        std::mem::swap(&mut self.graphs, &mut seq.graphs);
        std::mem::swap(&mut self.last, &mut seq.last);
        std::mem::swap(&mut self.held, &mut seq.held);
        self.own_at = None;
    }

    /// Device bytes one sequence's side of the draft holds: its store.
    pub(super) fn seq_bytes(&self) -> usize {
        self.store.bytes()
    }
}

/// Q8_0 or MXFP8 code and scale addresses, or BF16 words and a zero.
/// Returns `None` for another format.
pub(super) fn planes(dw: &DevWeight) -> Option<[u64; 2]> {
    match dw {
        DevWeight::Q8_0 { qs, d, .. } => Some([qs.buf().cu_deviceptr(), d.buf().cu_deviceptr()]),
        DevWeight::Mxfp8 { qs, e, .. } => Some([qs.buf().cu_deviceptr(), e.buf().cu_deviceptr()]),
        DevWeight::Bf16 { w, .. } => Some([w.buf().cu_deviceptr(), 0]),
        _ => None,
    }
}

/// `name` of the target's weights, at the addresses [`super::mtp35::addresses`]
/// records in `b`. `what` names the caller in the refusal.
pub(super) fn at_open<'w>(
    target: &'w Weights,
    b: &BorrowedPlanes,
    what: &'static str,
) -> Result<&'w DevWeight, GpuError> {
    let dw = target
        .get(&b.name)
        .ok_or_else(|| GpuError::tensor(what, b.name.clone(), "resident"))?;
    match super::mtp35::addresses(dw) {
        Some(p) if p == b.planes => Ok(dw),
        p => Err(GpuError::shape(
            what,
            format!(
                "{} sits at {p:?}, the draft was opened beside {:?}",
                b.name, b.planes
            ),
        )),
    }
}

/// Addresses of a resident Q8_0, MXFP8 or BF16 `[hidden, vocab]` matrix.
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
                "the target's {name} is {} rows of {} values; the draft requires a \
                 Q8_0, MXFP8 or BF16 [{hidden}, {vocab}] matrix",
                shape.1, shape.0
            ),
        )),
    }
}

/// `name` of the target's weights `w`, resident as the head's read form over
/// a `[hidden, vocab]` matrix: its Q8_0 or MXFP8 planes, BF16 words, or Q6_K
/// word plane with the row width and rows the gemv reads.
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
        DevWeight::Mxfp8 { .. } if (dw.k(), dw.rows()) == (hidden, vocab) => Ok(
            BorrowedHead::Mxfp8(borrowed_planes(w, name, hidden, vocab)?),
        ),
        DevWeight::Bf16 { .. } if (dw.k(), dw.rows()) == (hidden, vocab) => {
            Ok(BorrowedHead::Bf16(borrowed_planes(w, name, hidden, vocab)?))
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
                     Q8_0, MXFP8, Q6_K or BF16; the draft reads a [{hidden}, {vocab}] matrix of one of \
                     them"
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

/// The program's [`WalkBounds::WALK_ROWS`]: the m-column kernels' width.
pub const MTP_ROWS: usize = PASS_ROWS;

/// The program's [`WalkBounds::GRAPH_ROWS`]: a verify's rows, whose kept rows
/// a refresh runs.
pub const MTP_GRAPH_ROWS: usize = VERIFY_ROWS;

/// The program's [`WalkBounds::STORE_ROWS`] ([`WalkMode::Store`]): the columns
/// of its GEMMs, and the rows of the store arena it runs in. A store walk
/// enqueues launch by launch through the store's append and no further, up to
/// these rows through the wide launches (module doc): the rows' keys and
/// values, which are all a later row reads of them. A row's stored bits are
/// a function of that row's inputs alone, within the q8 activations' error
/// of an [`WalkMode::Eager`] walk's.
pub const MTP_STORE_ROWS: usize = 64;

impl WalkBounds for Mtp38 {
    const WALK_ROWS: usize = MTP_ROWS;
    const GRAPH_ROWS: usize = MTP_GRAPH_ROWS;
    const STORE_ROWS: usize = MTP_STORE_ROWS;
}

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

/// The target arena a [`Hidden::Target`] feed reads: the target's streams
/// after its last walk of that arena, from the row the feed names on; that
/// walk must have left them final, the last layer's combine applied: the
/// head's mix applies it over the rows it covers (a step, a verify, a
/// prompt's last pass or ubatch, a split walk's last unit), and a prompt
/// call with a tap applies it alone to every other row it walks. A prompt
/// call without a tap records only the rows its head or checkpoint requests
/// as final: a ubatch's requested suffix (`wide38`'s `WideParts::kept`),
/// every row of a pass with a head or finals, and none of a headless pass
/// with no finals. MTP refuses any row outside that completed range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetRows {
    /// The step's one row.
    Step,
    /// A pass's or a verify's rows.
    Pass,
    /// A ubatch walk's rows.
    Ubatch,
}

/// What a sequence state carries of the draft beside its positional rows,
/// which hold the store's ([`Mtp38::store_spans`]): the positions the store
/// holds ([`Mtp38::held`]), and the step's and the pass's arena rows with the
/// positions they hold ([`Held`]) — the target's hidden rows a draft's
/// waiting rows read when it rejoins the sequence. The ubatch walk's rows
/// are not carried: after a resume that arena holds no position, and a walk
/// that names it is refused. Beside them, for each point the state carries
/// that has one, the target's hidden row at the position before the point:
/// a put-back as the point's prefix rejoins the draft there.
#[derive(Clone, Debug)]
pub struct DraftRows38 {
    pub(super) held: u32,
    pub(super) step: (Held, Vec<f32>),
    pub(super) pass: (Held, Vec<f32>),
    pub(super) final_rows: [Held; 2],
    pub(super) point_rows: Vec<(u32, Vec<f32>)>,
}

impl DraftRows38 {
    /// The host bytes it holds.
    #[must_use]
    pub fn bytes(&self) -> usize {
        (self.step.1.len() + self.pass.1.len()) * size_of::<f32>()
            + self.point_rows.iter().map(|(_, r)| r.len()).sum::<usize>() * size_of::<f32>()
    }

    /// Its byte form ([`crate::checkpoint::bytes`]): the store's positions
    /// (u32); the step's and then the pass's arena, each its record (the
    /// first position and the rows, u32) and its values (a count, then the
    /// f32s); the two final-row records; the point rows (a count, then each
    /// point, u32, and its row as an array). The put back checks every
    /// length against the load ([`super::body38`]'s `draft_fits`).
    pub(super) fn put(&self, w: &mut Put<'_>) -> Result<(), Fail> {
        w.one(self.held)?;
        for (held, v) in [&self.step, &self.pass] {
            put_held(w, *held)?;
            w.array(v)?;
        }
        for &h in &self.final_rows {
            put_held(w, h)?;
        }
        w.count(self.point_rows.len())?;
        for (p, row) in &self.point_rows {
            w.one(*p)?;
            w.array(row)?;
        }
        Ok(())
    }

    /// What [`DraftRows38::put`] wrote.
    pub(super) fn take(r: &mut Take<'_>) -> Result<DraftRows38, Fail> {
        let held = r.one("the draft store's positions")?;
        let step = (take_held(r)?, r.array("the step arena's rows")?);
        let pass = (take_held(r)?, r.array("the pass arena's rows")?);
        let final_rows = [take_held(r)?, take_held(r)?];
        let n = r.count("the point rows")?;
        let mut point_rows = Vec::new();
        for _ in 0..n {
            let p = r.one("a point row")?;
            point_rows.push((p, r.array("a point row")?));
        }
        Ok(DraftRows38 {
            held,
            step,
            pass,
            final_rows,
            point_rows,
        })
    }
}

/// Refused by name unless rows `first .. first + m` of the `walk` arena, of
/// `rows` rows whose records are `held` and `final_rows`, hold the target's
/// final hidden rows of the positions `pos0 - 1 ..` ([`Held::check_read`]).
pub(super) fn check_target_rows(
    what: &'static str,
    walk: TargetRows,
    (held, final_rows, rows): (Held, Held, usize),
    (first, m): (usize, usize),
    pos0: u32,
) -> Result<(), GpuError> {
    held.check_read(what, walk, rows, (first, m), pos0)?;
    let at = pos0 - 1;
    if !at
        .checked_sub(final_rows.parts().0)
        .is_some_and(|r| final_rows.holds(r as usize, m, at))
    {
        return Err(GpuError::shape(
            what,
            format!(
                "the {walk:?} arena's final hidden rows hold {}, requested positions {at}..{}",
                final_rows.shown(),
                u64::from(at) + m as u64
            ),
        ));
    }
    Ok(())
}

fn put_held(w: &mut Put<'_>, h: Held) -> Result<(), Fail> {
    let (first, rows) = h.parts();
    w.vals(&[first, rows])
}

fn take_held(r: &mut Take<'_>) -> Result<Held, Fail> {
    let first = r.one("an arena's record")?;
    Ok(Held::at(first, r.one("an arena's record")?))
}

/// The target's streams a [`Hidden::Target`] feed copies from, and the
/// positions its rows hold ([`Held`]): the buffer and the record together,
/// the body's `mtp_target` resolving both over its arenas.
#[derive(Clone, Copy)]
pub(super) struct Target38<'a> {
    pub(super) buf: &'a DeviceBuffer<f32>,
    pub(super) held: Held,
    pub(super) final_rows: Held,
}

/// One sequence's rows of a store walk ([`Mtp38::store_rows`]): rows
/// `at .. at + m` of the store arena, at positions `pos0 ..` of their
/// sequence. A walk of one sequence is one block at row 0; a store walk of
/// several sequences' rows (`GpuModel::draft_store_slots`) lays a block a
/// sequence, each appended to its own store ([`Mtp38::append_store`]).
#[derive(Clone, Copy, Debug)]
pub struct StoreBlock<'a> {
    at: usize,
    m: usize,
    pos0: u32,
    /// The media prompt call's span rows by position ([`Feed::media`]), which
    /// stand in for the embedding rows of their positions; `None` on a text
    /// call.
    media: Option<&'a CallRows<'a>>,
}

/// A walk's hidden rows `hidden` copied into `h`, a row for each of its
/// rows: the host's, or the target's streams `src` from their row `first`.
fn copy_hidden(
    stream: &CudaStream,
    h: &mut DeviceBuffer<f32>,
    hidden: Hidden<'_, TargetRows>,
    src: Option<(&DeviceBuffer<f32>, usize)>,
) -> Result<(), GpuError> {
    match (hidden, src) {
        (Hidden::Host(v), _) => h.copy_from_host(stream, v)?,
        (Hidden::Target { .. }, Some((t, first))) => {
            let src = Window::<f32>::of(t, first * WIDE * size_of::<f32>(), h.len())?;
            h.copy_from_device_async(&src, stream)?;
        }
        (Hidden::Target { .. }, None) => {
            return Err(GpuError::state(WHAT, "the target's streams to copy"));
        }
    }
    Ok(())
}

/// A walk's rows as [`Mtp38::checked`] passed them: their count and first
/// position, the call's rows (`None` for the draft's own row) and the
/// target's streams their hidden rows copy from, with the row they start at.
struct Checked<'f, 't> {
    m: usize,
    pos0: u32,
    rows: Option<Feed<'f, TargetRows>>,
    src: Option<(&'t DeviceBuffer<f32>, usize)>,
}

/// A captured walk's key: its rows, whether it reads its own last row, its
/// head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MtpKey {
    m: usize,
    own: bool,
    head: DraftHead,
}

/// One sequence's rows of a walk ([`Mtp38::walk_in`]): arena rows `at .. at
/// + m`, their ids where [`IdsAt`] names them and their first position the
/// inbox word `pos0`.
#[derive(Clone, Copy, Debug)]
struct WalkRows {
    at: usize,
    m: usize,
    ids: IdsAt,
    pos0: usize,
}

/// Where a walk's rows' ids sit on the card: inbox words from a word on,
/// the own id the last walk copied beside the arena (one row), or one word
/// of the readback of several sequences' chains, the id a sequence's walk
/// before proposed (one row, [`walk_rows`]).
#[derive(Clone, Copy, Debug)]
enum IdsAt {
    Inbox(usize),
    Own,
    Chains(usize),
}

/// The inbox's first word of the positions a walk of several sequences'
/// chains reads: walk `j`'s block `i` at `CHAIN_POS + j·MTP_ROWS + i`, past
/// the walk's ids.
const CHAIN_POS: usize = IN_IDS + MTP_ROWS;

/// The most readback pairs one walk of several sequences' chains fills, the
/// store arena's position words less the fault word and its site mask
/// ([`Mtp38::read_chains`]).
pub(super) const CHAIN_PAIRS: usize = MTP_STORE_ROWS / 2 - 1;

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
        let (mut dst, src) = unsafe {
            (
                cells_view(&self.nodes[node.at()], 0, n),
                cells_view(src, 0, n),
            )
        };
        dst.copy_from_device_async(&src, stream)?;
        Ok(())
    }
}

/// A store walk's own buffers for up to [`MTP_STORE_ROWS`] rows,
/// token-major (module doc): the hidden rows, the embedding with its
/// positions and key counts, the pack and its q8 blocks over the `4·m`
/// packed columns, the streams, the wide mix's scratch, the mix and its q8
/// blocks, k and v, and the two one-expert tables (`4·m` and `m` slots).
struct StoreArena {
    h: DeviceBuffer<f32>,
    emb: DeviceBuffer<f32>,
    /// The rows' positions; between store walks, which write them before
    /// any read, a round of several sequences' chains' readback
    /// ([`Mtp38::chain_out`]: [`CHAIN_PAIRS`] pairs, then the fault word and
    /// its site mask).
    pos: DeviceBuffer<u32>,
    n_keys: DeviceBuffer<u32>,
    pack: DeviceBuffer<f32>,
    pack_q: GemmAct32,
    res: DeviceBuffer<f32>,
    hc: HcWideScratch,
    mixed: DeviceBuffer<f32>,
    mixed_q: GemmAct32,
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
            hc: HcWideScratch::new(stream, geometry, r, false)?,
            mixed: f(r * h)?,
            mixed_q: GemmAct32::new(stream, r, h)?,
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
    pos0: WindowMut<'static, u32>,
    /// The hidden rows a [`DraftFeed::Rows`] walk reads, and the last row's
    /// token and streams a [`DraftFeed::Own`] walk reads.
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

/// The draft layer's names, resolved once when its weights are loaded.
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
    fn of(index: u32, w: &Weights) -> Names {
        let b = |stem: &str| format!("blk.{index}.{stem}");
        let site = |sub: &str| HcSite {
            norm: b(&format!("hc_{sub}_norm.weight")),
            down: b(&format!("hc_{sub}_down.weight")),
            up: b(&format!("hc_{sub}_up.weight")),
            inject: Some({
                let stem = format!("hc_{sub}_inject.weight");
                let name = b(&stem);
                if w.get(&name).is_some() {
                    name
                } else {
                    model::arch::qwen35moe::place::mtp_widened(index, &stem)
                }
            }),
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
/// the kernels, the norms' epsilon and the rotation table of the sequence
/// it walks (a media load's own, the load's plain one on a text load).
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
        // The walk's dense sites read the f32 rows or q8 planes: a K-quant
        // draft layer is not projected by name (the target's sites are,
        // `site::ByName`), and is refused here rather than at a walk.
        let n = &self.names;
        for name in [
            &n.eh, &n.q, &n.k, &n.v, &n.out, &n.gate_sh, &n.up_sh, &n.down_sh,
        ] {
            let ty = site::resident_type(&self.w, WHAT, name)?;
            if !site::BY_NAME.contains(&ty) {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "{name} is {ty}: the draft layer's walks read the f32 rows and the q8 \
                         planes of their sites, and none projects a K-quant site by name"
                    ),
                ));
            }
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

    pub(super) fn poison_scratch(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        let a = self.arena()?;
        for b in [
            &mut a.emb,
            &mut a.pack,
            &mut a.res,
            &mut a.mixed,
            &mut a.qg,
            &mut a.q,
            &mut a.k,
            &mut a.v,
            &mut a.part_v,
            &mut a.part_ms,
            &mut a.flash,
            &mut a.attn,
            &mut a.y,
            &mut a.ffn_x,
            &mut a.rg,
            &mut a.ru,
            &mut a.rh,
            &mut a.rdown,
            &mut a.acc,
            &mut a.sh_g,
            &mut a.sh_u,
            &mut a.sh_h,
            &mut a.sh_y,
            &mut a.head_x,
            &mut a.logits,
            &mut a.hc.xn,
            &mut a.hc.dpart,
            &mut a.hc.ipart,
            &mut a.hc.lo,
            &mut a.hc.wgt,
            &mut a.route.logits,
            &mut a.route.probs,
            &mut a.route.weights,
        ] {
            b.copy_from_host(stream, &vec![f32::NAN; b.len()])?;
        }
        for b in [
            &mut a.pos,
            &mut a.n_keys,
            &mut a.sel,
            &mut a.route.ids,
            &mut a.out,
            &mut a.chain,
        ] {
            b.copy_from_host(stream, &vec![u32::MAX; b.len()])?;
        }
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
        self.own_at = None;
        self.held = 0;
    }

    /// A cut of the target to `pos`: the store holds no position past it —
    /// the rows past it belong to the branch the cut dropped, and a walk
    /// writes its own rows' keys before its attention reads them.
    pub(super) fn cut(&mut self, pos: u32) {
        self.held = self.held.min(pos as usize);
        self.own_at = None;
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

    /// Capture every walk a window's chain replays on this sequence that is
    /// not captured yet: a refresh of 1 to [`MTP_GRAPH_ROWS`] rows and the
    /// own walk, into the opened head; how many it captured. Nothing is
    /// launched. Load-time; refused by name with the taps armed, as a
    /// captured walk is.
    pub(super) fn capture_walks(&mut self, c: &MtpCtx<'_>) -> Result<usize, GpuError> {
        if self.arena_ref()?.taps.is_some() {
            return Err(GpuError::state(WHAT, "the taps off for a captured walk"));
        }
        let head = if self.head.rows.is_some() {
            DraftHead::Rows
        } else {
            DraftHead::Full
        };
        let refresh = (1..=MTP_GRAPH_ROWS).map(|m| MtpKey {
            m,
            own: false,
            head,
        });
        let own = MtpKey {
            m: 1,
            own: true,
            head,
        };
        let mut captured = 0;
        for key in refresh.chain([own]) {
            if self.graphs.iter().any(|(k, _)| *k == key) {
                continue;
            }
            let graph = Graph::capture(c.gpu.stream(), |_| self.walk(c, key))?;
            self.graphs.push((key, graph));
            captured += 1;
        }
        Ok(captured)
    }

    /// The captured walks' node counts, by rows, feed and head.
    #[must_use]
    pub fn graph_nodes(&self) -> Vec<(usize, bool, DraftHead, usize)> {
        self.graphs
            .iter()
            .map(|(k, g)| (k.m, k.own, k.head, g.node_count()))
            .collect()
    }

    /// Run one walk of the program (module doc) over `feed`'s rows into
    /// `head`, eager or replayed, and read its tokens back. `target` is the
    /// target's streams a [`Hidden::Target`] feed copies from. Refused by
    /// name before anything moves: rows outside 1..=[`MTP_ROWS`] (a captured
    /// walk 1..=[`MTP_GRAPH_ROWS`]), a hidden slice of other than the rows'
    /// values, a token past the vocabulary, positions past the store, an
    /// own row before any walk of this sequence, a walk starting past the
    /// positions the store holds for the sequence, a row-list head on a
    /// full-head load, a captured walk with the taps armed, a store walk
    /// ([`WalkMode::Store`] reads nothing back), media rows ([`call_rows`]).
    /// A fault a launch raised is [`GpuError::Fault`], not a
    /// token.
    pub(super) fn run(
        &mut self,
        c: &MtpCtx<'_>,
        target: Option<Target38<'_>>,
        feed: DraftFeed<'_, TargetRows>,
        head: DraftHead,
        mode: WalkMode,
    ) -> Result<Proposal, GpuError> {
        let stream = c.gpu.stream();
        refuse_store(
            mode,
            "a walk with a head to read back (WalkMode::Store writes the store alone)",
        )?;
        let m = self.run_walk(c, target, feed, head, mode)?;
        let a = self.arena_ref()?;
        read_draft(a, stream, m)
    }

    /// One walk of the program over `feed`'s rows into `head`, eager,
    /// replayed from its capture or through the store's append alone
    /// ([`WalkMode::Store`]), with no readback: [`Mtp38::run`]'s checks and
    /// launches, the arena's last-walk bookkeeping moved and the walk's rows
    /// returned. A store walk leaves no last walk: an own row after it, and
    /// the readbacks of its rows, are refused by name. A fault the walk
    /// raised stays on the fault word, which the next readback names
    /// ([`read_draft`], a chain's).
    pub(super) fn run_walk(
        &mut self,
        c: &MtpCtx<'_>,
        target: Option<Target38<'_>>,
        feed: DraftFeed<'_, TargetRows>,
        head: DraftHead,
        mode: WalkMode,
    ) -> Result<usize, GpuError> {
        let stream = c.gpu.stream();
        head.check_opened(WHAT, self.head.rows.is_some())?;
        let taps_on = self.arena_ref()?.taps.is_some();
        if mode == WalkMode::Graph && taps_on {
            return Err(GpuError::state(WHAT, "the taps off for a captured walk"));
        }
        let w = self.checked(target, feed, mode)?;
        let (m, pos0) = (w.m, w.pos0);
        self.own_at = None;
        self.store.fit(pos0 as usize + m)?;
        self.store.own_from(pos0 as usize)?;
        if mode == WalkMode::Store {
            let b = StoreBlock {
                at: 0,
                m,
                pos0,
                media: w.rows.and_then(|f| f.media),
            };
            self.stage_hidden(stream, &w, b)?;
            let ids = w.rows.map_or(&[][..], |f| f.tokens);
            self.store_rows(c, ids, &[b], w.rows.is_none())?;
            self.append_store(c, b)?;
            return Ok(m);
        }
        let own = w.rows.is_none();
        let a = self.arena()?;
        match w.rows {
            Some(f) => {
                put_input(a.inbox.host_mut()?, f.tokens, pos0)?;
                a.inbox.upload(stream, IN_IDS + m)?;
                let mut h = WindowMut::<f32>::of_mut(&mut a.h_in, 0, m * WIDE)?;
                copy_hidden(stream, &mut h, f.hidden, w.src)?;
            }
            None => {
                put_input(a.inbox.host_mut()?, &[], pos0)?;
                a.inbox.upload(stream, IN_IDS)?;
            }
        }
        let key = MtpKey { m, own, head };
        if mode == WalkMode::Graph {
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
        } else {
            self.walk(c, key)?;
        }
        self.last = Some((m, head));
        self.held = pos0 as usize + m;
        self.own_at = Some(self.held);
        Ok(m)
    }

    /// `feed`'s rows checked for a walk in `mode` on this sequence, before
    /// anything moves ([`Mtp38::run_walk`]'s refusals): rows outside the
    /// mode's 1..=rows, positions past the store, an own row before any
    /// walk of this sequence, a walk starting past the positions the store
    /// holds, an own row at any position other than its recorded next position,
    /// a token past the vocabulary, a host hidden slice of other than
    /// the rows' values, target rows that do not hold the positions before
    /// the rows' ([`Held::check_read`]) or were not completed by the last layer,
    /// media rows outside a store walk or of another width ([`call_rows`]).
    fn checked<'f, 't>(
        &self,
        target: Option<Target38<'t>>,
        feed: DraftFeed<'f, TargetRows>,
        mode: WalkMode,
    ) -> Result<Checked<'f, 't>, GpuError> {
        let ctx = self.ctx;
        let rows = call_rows(feed, mode)?;
        let (m, own, pos0) = (feed.count(), rows.is_none(), feed.pos0());
        refuse_rows::<Mtp38>(WHAT, mode, m)?;
        if pos0 as usize + m > ctx {
            return Err(GpuError::shape(
                WHAT,
                format!("{m} rows from position {pos0}, in a store of {ctx}"),
            ));
        }
        let (vocab, held) = {
            let vocab = self.arena_ref()?.vocab;
            (vocab, self.held)
        };
        if own && self.own_at != Some(pos0 as usize) {
            return Err(match self.own_at {
                None => GpuError::state(
                    WHAT,
                    "a walk before the draft's own row (Rows refresh required)",
                ),
                Some(at) => GpuError::shape(
                    WHAT,
                    format!("the draft's own row is for position {at}, requested position {pos0}"),
                ),
            });
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
        if let Some(f) = rows {
            let (tokens, hidden) = (f.tokens, f.hidden);
            super::refuse_past_vocab(WHAT, tokens, vocab)?;
            match hidden {
                Hidden::Host(v) if v.len() != m * WIDE => {
                    return Err(GpuError::shape(
                        WHAT,
                        format!("{} hidden values for {m} rows of {WIDE}", v.len()),
                    ));
                }
                Hidden::Host(_) => {}
                Hidden::Target { walk, first } => {
                    let t = target.ok_or(GpuError::state(WHAT, "the target's streams to copy"))?;
                    check_target_rows(
                        WHAT,
                        walk,
                        (t.held, t.final_rows, t.buf.len() / WIDE),
                        (first, m),
                        pos0,
                    )?;
                    src = Some((t.buf, first));
                }
            }
        }
        Ok(Checked { m, pos0, rows, src })
    }

    /// One sequence's rows of a store walk of several sequences' rows
    /// ([`GpuModel::draft_store_slots`](crate::GpuModel::draft_store_slots),
    /// which holds the walk to the store arena's [`MTP_STORE_ROWS`]), staged
    /// at rows `at ..` of the store arena: `feed` checked as a store walk
    /// into `head` on this sequence, the live one ([`Mtp38::checked`]), and
    /// its hidden rows copied in. Refused by name as [`Mtp38::run_walk`]
    /// refuses a store walk; nothing but the arena's scratch moves.
    pub(super) fn stage_store<'f>(
        &mut self,
        stream: &CudaStream,
        target: Option<Target38<'_>>,
        feed: Feed<'f, TargetRows>,
        head: DraftHead,
        at: usize,
    ) -> Result<StoreBlock<'f>, GpuError> {
        head.check_opened(WHAT, self.head.rows.is_some())?;
        let w = self.checked(target, DraftFeed::Rows(feed), WalkMode::Store)?;
        self.store.fit(w.pos0 as usize + w.m)?;
        self.store.own_from(w.pos0 as usize)?;
        let b = StoreBlock {
            at,
            m: w.m,
            pos0: w.pos0,
            media: feed.media,
        };
        self.stage_hidden(stream, &w, b)?;
        Ok(b)
    }

    /// The hidden rows of `w`, a store walk's checked rows, copied into rows
    /// `b.at ..` of the store arena's `h`; nothing for the own row, which the
    /// walk reads in place.
    fn stage_hidden(
        &mut self,
        stream: &CudaStream,
        w: &Checked<'_, '_>,
        b: StoreBlock<'_>,
    ) -> Result<(), GpuError> {
        let Some(f) = w.rows else {
            return Ok(());
        };
        let a = self.arena()?;
        let mut h =
            WindowMut::<f32>::of_mut(&mut a.st.h, b.at * WIDE * size_of::<f32>(), b.m * WIDE)?;
        copy_hidden(stream, &mut h, f.hidden, w.src)
    }

    /// One window's chain — the draft's proposal, one readback: `refresh`'s
    /// walk (a [`DraftFeed::Rows`] feed, its rows the target's kept rows with
    /// the target's hidden rows), whose last row proposes the first id, then
    /// `own` walks of [`DraftFeed::Own`], each reading the walk before it on
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
        refresh: DraftFeed<'_, TargetRows>,
        own: usize,
        head: DraftHead,
        mode: WalkMode,
    ) -> Result<Proposal, GpuError> {
        let stream = c.gpu.stream();
        refuse_store(
            mode,
            "a chain's walks with a head (WalkMode::Store writes the store alone)",
        )?;
        let Some(f) = call_rows(refresh, mode)? else {
            return Err(GpuError::shape(
                WHAT,
                format!("a chain's refresh {refresh:?}: the rows the target kept"),
            ));
        };
        let (tokens, pos0) = (f.tokens, f.pos0);
        // Every walk's positions, checked before the first moves anything:
        // the refresh's rows, then one own row at each position after them.
        let end = self.chain_end(pos0, tokens.len(), own)?;
        let feeds: Vec<DraftFeed<'_, TargetRows>> = std::iter::once(refresh)
            .chain((0..own).map(|i| {
                DraftFeed::Own {
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
        // The own rows are the proposal's, and no later walk reads them: the
        // store holds the sequence through the refresh, where the next
        // refresh starts at or below its end (a verify's kept rows, or the
        // step's row when no verify takes the proposal).
        self.held = end;
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
        Ok(Proposal {
            tokens: out[..2 * n].iter().step_by(2).copied().collect(),
            p: out[1..2 * n]
                .iter()
                .step_by(2)
                .map(|&b| f32::from_bits(b))
                .collect(),
        })
    }

    /// The end of a chain's refresh of `m` rows from `pos0`, its `own` own
    /// walks checked to fit a chain ([`MTP_GRAPH_ROWS`]) and the store: one
    /// own row at each position after the refresh. Refused by name past
    /// either.
    fn chain_end(&self, pos0: u32, m: usize, own: usize) -> Result<usize, GpuError> {
        if own >= MTP_GRAPH_ROWS {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{own} own walks; a chain proposes at most {}",
                    MTP_GRAPH_ROWS - 1
                ),
            ));
        }
        let end = pos0 as usize + m;
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
        Ok(end)
    }

    /// One sequence's chain in a round of several sequences' chains, checked
    /// and staged on this sequence, the live one, as the round's block
    /// `block` at arena rows `at ..`, its readback pairs from `pair`:
    /// `refresh`'s rows — the target's kept rows with its hidden rows, as
    /// [`Mtp38::run_chain`] takes them — then `own` own walks into `head`.
    /// Each walk of the round runs over every block it reaches
    /// ([`Mtp38::chain_in`], [`Mtp38::attend`], [`Mtp38::chain_out`]),
    /// launch by launch whatever `mode` names: its launches address every
    /// block's store. Refused by name as `run_chain` refuses the chain, and
    /// past the round's rows ([`MTP_ROWS`]) or pairs ([`CHAIN_PAIRS`]);
    /// nothing but the arena's scratch and the inbox's host words moves, and
    /// the store's pages are mapped through the chain's last own row.
    pub(super) fn stage_chain(
        &mut self,
        stream: &CudaStream,
        target: Option<Target38<'_>>,
        refresh: Feed<'_, TargetRows>,
        (own, head, mode): (usize, DraftHead, WalkMode),
        (block, at, pair): (usize, usize, usize),
    ) -> Result<ChainBlock, GpuError> {
        refuse_store(
            mode,
            "a chain's walks with a head (WalkMode::Store writes the store alone)",
        )?;
        head.check_opened(WHAT, self.head.rows.is_some())?;
        let w = self.checked(target, DraftFeed::Rows(refresh), mode)?;
        let (m, pos0) = (w.m, w.pos0);
        let end = self.chain_end(pos0, m, own)?;
        if block >= MTP_ROWS || at + m > MTP_ROWS || pair + 1 + own > CHAIN_PAIRS {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "block {block}'s {m} rows from row {at} and {} pairs from pair {pair}: a \
                     round of chains walks at most {MTP_ROWS} rows and reads back {CHAIN_PAIRS} \
                     pairs",
                    1 + own
                ),
            ));
        }
        self.store.fit(end + own)?;
        self.store.own_from(pos0 as usize)?;
        let a = self.arena()?;
        let host = a.inbox.host_mut()?;
        host[IN_IDS + at..IN_IDS + at + m].copy_from_slice(refresh.tokens);
        host[CHAIN_POS + block] = pos0;
        for j in 1..=own {
            host[CHAIN_POS + j * MTP_ROWS + block] =
                u32::try_from(end + j - 1).expect("a chain's positions lie below its store");
        }
        let mut h = WindowMut::<f32>::of_mut(&mut a.h_in, at * WIDE * size_of::<f32>(), m * WIDE)?;
        copy_hidden(stream, &mut h, refresh.hidden, w.src)?;
        Ok(ChainBlock {
            block,
            at,
            m,
            pos0,
            own,
            pair,
        })
    }

    /// The round's inbox words, every block's ids and every walk's
    /// positions ([`Mtp38::stage_chain`]), copied to the card ahead of its
    /// walks.
    pub(super) fn upload_chains(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        self.arena()?
            .inbox
            .upload(stream, CHAIN_POS + MTP_GRAPH_ROWS * MTP_ROWS)
    }

    /// Walk `j` of a round of `blocks`' chains up to its attention
    /// ([`Mtp38::walk_in`]) over the blocks it reaches ([`chain_rows`]).
    pub(super) fn chain_in(
        &mut self,
        c: &MtpCtx<'_>,
        blocks: &[ChainBlock],
        j: usize,
    ) -> Result<(), GpuError> {
        let rows: Vec<WalkRows> = walk_rows(blocks, j).map(|(_, r)| r).collect();
        self.walk_in(c, &rows, false)
    }

    /// Walk `j` of a round of `blocks`' chains after its attention
    /// ([`Mtp38::walk_out`]), then, for each block it reaches, the last
    /// row's id and probability bits copied into the block's pair for the
    /// walk, and, when the block's chain walks on, its streams into the row
    /// the next walk reads them at; the round's last walk's fault word and
    /// site mask close the readback ([`Mtp38::read_chains`]).
    pub(super) fn chain_out(
        &mut self,
        c: &MtpCtx<'_>,
        blocks: &[ChainBlock],
        j: usize,
        head: DraftHead,
    ) -> Result<(), GpuError> {
        let rows: Vec<(ChainBlock, WalkRows)> = walk_rows(blocks, j).collect();
        let m = rows.last().map_or(0, |(_, r)| r.at + r.m);
        self.walk_out(c, m, head)?;
        let stream = c.gpu.stream();
        let a = self.arena()?;
        let word = size_of::<u32>();
        let mut next = 0;
        for (b, r) in &rows {
            let last = r.at + r.m - 1;
            let pair = 2 * (b.pair + j);
            let w_id = Window::<u32>::of(&a.out, last * word, 1)?;
            WindowMut::<u32>::of_mut(&mut a.st.pos, pair * word, 1)?
                .copy_from_device_async(&w_id, stream)?;
            let w_p = Window::<u32>::of(&a.out, (m + last) * word, 1)?;
            WindowMut::<u32>::of_mut(&mut a.st.pos, (pair + 1) * word, 1)?
                .copy_from_device_async(&w_p, stream)?;
            if b.own > j {
                let src = Window::<f32>::of(&a.res, last * WIDE * size_of::<f32>(), WIDE)?;
                let mut dst =
                    WindowMut::<f32>::of_mut(&mut a.h_in, next * WIDE * size_of::<f32>(), WIDE)?;
                dst.copy_from_device_async(&src, stream)?;
                next += 1;
            }
        }
        if blocks.iter().any(|b| b.own > j) {
            return Ok(());
        }
        // The round's last walk: its fault word is the card's own, so a fault
        // any walk of the round raised is on it.
        let w_word = Window::<u32>::of(&a.out, 2 * m * word, 1)?;
        WindowMut::<u32>::of_mut(&mut a.st.pos, 2 * CHAIN_PAIRS * word, 1)?
            .copy_from_device_async(&w_word, stream)?;
        let w_sites = Window::<u32>::of(&a.out, (2 * m + 1) * word, 1)?;
        WindowMut::<u32>::of_mut(&mut a.st.pos, (2 * CHAIN_PAIRS + 1) * word, 1)?
            .copy_from_device_async(&w_sites, stream)?;
        Ok(())
    }

    /// A round of several sequences' chains ended for `b`'s sequence, the
    /// live one: its store holds the sequence through the refresh, as
    /// [`Mtp38::run_chain`] leaves it. The arena's rows are the round's, so
    /// no walk of the sequence is read back and an own row needs a Rows
    /// refresh first.
    pub(super) fn end_chain(&mut self, b: &ChainBlock) {
        self.held = b.end();
        self.last = None;
        self.own_at = None;
    }

    /// A round of `blocks`' chains read back once ([`Mtp38::chain_out`]'s
    /// pairs): each block's proposal, its 1 + own ids and their
    /// probabilities, in `blocks`' order. A fault any walk of the round
    /// raised is [`GpuError::Fault`]: the word is the card's own, which the
    /// argmax copies after its launch.
    pub(super) fn read_chains(
        &self,
        stream: &CudaStream,
        blocks: &[ChainBlock],
    ) -> Result<Vec<Proposal>, GpuError> {
        let out = self.arena_ref()?.st.pos.to_host_vec(stream)?;
        let (Some(&word), Some(&sites)) = (out.get(2 * CHAIN_PAIRS), out.get(2 * CHAIN_PAIRS + 1))
        else {
            return Err(GpuError::state(
                WHAT,
                "a round's readback of 2·CHAIN_PAIRS + 2 words",
            ));
        };
        if let Some(fault) = Fault::from_words(word, sites) {
            return Err(GpuError::fault(WHAT, fault));
        }
        blocks
            .iter()
            .map(|b| {
                let w = out
                    .get(2 * b.pair..2 * (b.pair + b.pairs()))
                    .ok_or(GpuError::state(WHAT, "a block's pairs inside the readback"))?;
                Ok(Proposal {
                    tokens: w.iter().step_by(2).copied().collect(),
                    p: w[1..]
                        .iter()
                        .step_by(2)
                        .map(|&b| f32::from_bits(b))
                        .collect(),
                })
            })
            .collect()
    }

    /// The walk itself: every launch of the module doc's list, enqueued on
    /// the card's stream (what a capture records): the rows' launches up to
    /// k and v ([`Mtp38::walk_in`]), the attention over this sequence's
    /// store ([`Mtp38::attend`]) and the rest through the argmax
    /// ([`Mtp38::walk_out`]), then the last row's token and streams copied
    /// beside the arena, where the next own walk reads them.
    fn walk(&mut self, c: &MtpCtx<'_>, key: MtpKey) -> Result<(), GpuError> {
        let MtpKey { m, own, head } = key;
        let rows = WalkRows {
            at: 0,
            m,
            ids: if own {
                IdsAt::Own
            } else {
                IdsAt::Inbox(IN_IDS)
            },
            pos0: IN_POS0,
        };
        self.walk_in(c, &[rows], own)?;
        self.attend(c, 0, m)?;
        self.walk_out(c, m, head)?;
        let stream = c.gpu.stream();
        let a = self.arena()?;
        // SAFETY: word m − 1 of the readback's 2·MTP_ROWS + 2 and row m − 1
        // of the streams' MTP_ROWS rows (1 <= m <= MTP_ROWS); the windows
        // live for the two copies.
        let (id, row) = unsafe {
            (
                param_view::<u32>(&a.out, m - 1, 1),
                cells_view(&a.res, (m - 1) * WIDE, WIDE),
            )
        };
        a.own_id.copy_from_device_async(&id, stream)?;
        a.own_h.copy_from_device_async(&row, stream)?;
        Ok(())
    }

    /// A walk's launches before its attention, over `rows`, arena rows
    /// `0 ..` in order: each one's embedding at its own first position, then
    /// over all of them the input pack — the hidden rows `own_h`'s, the
    /// draft's own last row, or `h_in`'s — `eh_proj`, the attention site's
    /// mix and q, k and v.
    fn walk_in(&mut self, c: &MtpCtx<'_>, rows: &[WalkRows], own_h: bool) -> Result<(), GpuError> {
        let (embd, _) = self.borrowed(c.tw)?;
        let Mtp38 {
            w, index, ctx, a, ..
        } = self;
        let a = a
            .as_mut()
            .ok_or(GpuError::state(WHAT, "the program's arena (Mtp38::arm)"))?;
        let (gpu, k) = (c.gpu, c.k);
        let stream = gpu.stream();
        let sink = gpu.layer_sink(*index as usize)?;
        let n = &self.names;
        let cx = Ctx38 {
            gpu,
            w,
            k,
            eps: c.eps,
            table: c.table,
            ctx: *ctx,
        };
        let capturing = crate::capturing(stream)?;
        let h = geo::HIDDEN;
        let m = rows.last().map_or(0, |r| r.at + r.m);

        // The embedding rows, a sequence's at a time, and the input pack.
        let table = EmbedRows::of_weight(embd).ok_or_else(|| {
            GpuError::tensor(WHAT, "token_embd.weight", "Q8_0 planes or BF16 words")
        })?;
        let word = size_of::<u32>();
        for r in rows {
            // SAFETY: the rows' ids — words `at .. at + m` of the inbox's
            // IN_IDS + MTP_STORE_ROWS, the own id's one word, or one word of
            // the chains' readback — and their first position's inbox word,
            // each inside its buffer ([`WalkRows`]); the windows live for the
            // embedding's enqueue.
            let (ids, pos0) = unsafe {
                (
                    match r.ids {
                        IdsAt::Inbox(at) => param_view::<u32>(a.inbox.dev(), at, r.m),
                        IdsAt::Own => param_view::<u32>(&a.own_id, 0, 1),
                        IdsAt::Chains(at) => param_view::<u32>(&a.st.pos, at, 1),
                    },
                    param_view::<u32>(a.inbox.dev(), r.pos0, 1),
                )
            };
            let mut y = WindowMut::<f32>::of_mut(&mut a.emb, r.at * h * size_of::<f32>(), r.m * h)?;
            let mut pos = WindowMut::<u32>::of_mut(&mut a.pos, r.at * word, r.m)?;
            let mut n_keys = WindowMut::<u32>::of_mut(&mut a.n_keys, r.at * word, r.m)?;
            k.q38.enqueue_embed_rows(
                stream,
                EmbedArgs {
                    table,
                    ids: &ids,
                    pos0: &pos0,
                    first: 0,
                    fault: sink,
                    y: &mut y,
                    pos: &mut pos,
                    n_keys: &mut n_keys,
                },
            )?;
        }
        k.mtp.enqueue_mtp_input(
            stream,
            MtpInputArgs {
                e: &a.emb,
                h: if own_h { &a.own_h } else { &a.h_in },
                enorm: f32_gain(w, &n.enorm)?,
                hnorm: f32_gain(w, &n.hnorm)?,
                hidden: h,
                eps: c.eps,
                m,
                streams: geo::STREAMS,
                fault: sink,
                out: &mut a.pack,
            },
        )?;
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
                    cells_view(&a.pack, at * 2 * h, run * 2 * h),
                    cells_view(&a.res, at * h, run * h),
                )
            };
            cx.dense_gemv(&n.eh, &x, run, &mut y)?;
            at += run;
        }
        if let (Some(t), false) = (a.taps.as_mut(), capturing) {
            // SAFETY: the first m rows of the tap and of the streams, both
            // MTP_ROWS · WIDE values; the windows live for this copy.
            let (mut dst, src) = unsafe {
                (
                    cells_view(&t.eh, 0, m * WIDE),
                    cells_view(&a.res, 0, m * WIDE),
                )
            };
            dst.copy_from_device_async(&src, stream)?;
        }

        // The attention site: its mix, then q, k and v.
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
        cx.dense_gemv(&n.q, &a.mixed, m, &mut a.qg)?;
        cx.dense_gemv(&n.k, &a.mixed, m, &mut a.k)?;
        cx.dense_gemv(&n.v, &a.mixed, m, &mut a.v)?;
        Ok(())
    }

    /// The attention of arena rows `at .. at + m`, one sequence's rows of
    /// the walk [`Mtp38::walk_in`] enqueued, over this sequence's store (the
    /// live one's): the q/k norm, turn and append at their positions, then
    /// the flash over every stored position below each row's.
    pub(super) fn attend(&mut self, c: &MtpCtx<'_>, at: usize, m: usize) -> Result<(), GpuError> {
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
        let n = &self.names;
        let (f, word) = (size_of::<f32>(), size_of::<u32>());
        let qg = Window::<f32>::of(&a.qg, at * geo::Q_ROWS * f, m * geo::Q_ROWS)?;
        let mut q = WindowMut::<f32>::of_mut(&mut a.q, at * geo::ATTN * f, m * geo::ATTN)?;
        let mut kw = WindowMut::<f32>::of_mut(&mut a.k, at * geo::KV * f, m * geo::KV)?;
        let v = Window::<f32>::of(&a.v, at * geo::KV * f, m * geo::KV)?;
        let pos = Window::<u32>::of(&a.pos, at * word, m)?;
        let n_keys = Window::<u32>::of(&a.n_keys, at * word, m)?;
        let mut y = WindowMut::<f32>::of_mut(&mut a.flash, at * geo::ATTN * f, m * geo::ATTN)?;
        // SAFETY: the per-sequence embedding supplies distinct cache-valid positions.
        unsafe {
            store.append_256(
                &k.neox,
                stream,
                Append256 {
                    qg: &qg,
                    q: &mut q,
                    k: &mut kw,
                    v: &v,
                    gq: f32_gain(w, &n.q_norm)?,
                    gk: f32_gain(w, &n.k_norm)?,
                    table: c.table,
                    pos: &pos,
                    eps: c.eps,
                    n_head: geo::N_HEAD,
                    n_kv: geo::N_KV,
                    ctx: *ctx,
                    m,
                    fault: sink,
                },
            )
        }?;
        store.flash_256(
            &k.flash,
            stream,
            super::scratch::FlashPass {
                q: &q,
                n_keys: &n_keys,
                scale: ATTN_SCALE_256,
                n_kv: geo::N_KV,
                ctx: *ctx,
                m,
                part_v: &mut a.part_v,
                part_ms: &mut a.part_ms,
                fault: sink,
                y: &mut y,
            },
            geo::N_HEAD,
            super::plan::Flash::Quads,
            MMA,
        )
    }

    /// A walk's launches after its attention, over its `m` rows: the gate
    /// and the output projection, the feed-forward site, the head site's mix,
    /// the projection into `head` and the argmax.
    fn walk_out(&mut self, c: &MtpCtx<'_>, m: usize, head: DraftHead) -> Result<(), GpuError> {
        let (_, head_out) = self.borrowed(c.tw)?;
        let Mtp38 {
            w,
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
        let n = &self.names;
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
        cx.dense_gemv(&n.out, &a.attn, m, &mut a.y)?;
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
            gated::RouterWeight::of(w, WHAT, &n.router)?,
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
            crate::site::resident_sel_f32(
                gpu,
                w,
                name,
                &SelArgs {
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
        crate::site::resident_sel_f32(
            gpu,
            w,
            &n.down_exps,
            &SelArgs {
                x: &a.rh,
                sel: &a.sel,
                n_slots: slots,
                rows_per_expert: h,
                slots_per_col: 1,
            },
            sink,
            &mut a.rdown,
        )?;
        let sum = CardAcc {
            down: &a.rdown,
            w: &a.route.weights,
            sel: &a.sel,
            n: h,
            m,
            n_card: geo::EXPERTS,
            n_used: SLOTS,
            w_pitch: W_PITCH,
            fault: sink,
        };
        k.slots.enqueue_card_acc(stream, &sum, &mut a.acc)?;
        cx.dense_gemv(&n.gate_sh, &a.ffn_x, m, &mut a.sh_g)?;
        cx.dense_gemv(&n.up_sh, &a.ffn_x, m, &mut a.sh_u)?;
        gpu.elem()
            .enqueue_swiglu(stream, &a.sh_g, &a.sh_u, ff * m, &mut a.sh_h)?;
        cx.dense_gemv(&n.down_sh, &a.sh_h, m, &mut a.sh_y)?;
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
            HeadOut::Mxfp8 { qs, .. } => qs.rows(),
            HeadOut::Q6K { w } | HeadOut::Bf16 { w } => w.rows(),
        };
        let (rows, map) = match head {
            DraftHead::Full => (full_rows, None),
            DraftHead::Rows => (
                hm.rows
                    .ok_or(GpuError::state(WHAT, "a row list for the head"))?,
                Some(&hm.ids),
            ),
        };
        // SAFETY: the logits hold MTP_ROWS · vocab values and rows <= vocab
        // (the full head's, or a list no longer than the vocabulary, refused
        // at open otherwise), m <= MTP_ROWS; the window lives for the
        // projection and the argmax.
        let mut lg = unsafe { cells_view(&a.logits, 0, rows * m) };
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
            HeadOut::Mxfp8 { qs, e } => match map {
                None => gpu.q8f32().enqueue_e4m3_gemv(
                    stream,
                    E4m3GemvArgs {
                        w: E4m3Weight {
                            qs,
                            e,
                            scales: E4m3Scales::Mxfp8,
                        },
                        x: &a.head_x,
                        m,
                        out: GemvOut::RowMajor,
                        y: &mut lg,
                    },
                )?,
                Some(ids) => gpu.q8f32().enqueue_mxfp8_gemv_ids(
                    stream,
                    Mxfp8GemvIdsArgs {
                        qs,
                        e,
                        ids,
                        rows,
                        x: &a.head_x,
                        m,
                        y: &mut lg,
                    },
                    sink,
                )?,
            },
            HeadOut::Bf16 { w } => match map {
                None => gpu
                    .q8f32()
                    .enqueue_bf16_gemv(stream, w, &a.head_x, m, &mut lg)?,
                Some(ids) => gpu.q8f32().enqueue_bf16_gemv_ids(
                    stream,
                    Bf16GemvIdsArgs {
                        w,
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
        Ok(())
    }

    /// The launches of a store walk before its append (module doc) over
    /// `blocks`' rows, rows `0 ..` of the store arena in order, every launch
    /// once over them all but the embedding, one a block at its own
    /// positions with its media rows over theirs ([`splice_media`]): the
    /// embedding and the pack, `eh_proj`, the attention
    /// site's mix, k and v, on the card's stream. The rows' tokens are
    /// `ids`, the blocks' in order, and their hidden rows the store arena's
    /// `h`; or, when `own`, one block of the draft's own last row. Each row's
    /// values are its inputs' alone, so a block's are those of a walk of it
    /// alone; [`Mtp38::append_store`] writes each block into its sequence's
    /// store.
    pub(super) fn store_rows(
        &mut self,
        c: &MtpCtx<'_>,
        ids: &[u32],
        blocks: &[StoreBlock<'_>],
        own: bool,
    ) -> Result<(), GpuError> {
        let (embd, _) = self.borrowed(c.tw)?;
        let Mtp38 { w, index, a, .. } = self;
        let a = a
            .as_mut()
            .ok_or(GpuError::state(WHAT, "the program's arena (Mtp38::arm)"))?;
        let (gpu, k) = (c.gpu, c.k);
        let stream = gpu.stream();
        let sink = gpu.layer_sink(*index as usize)?;
        let n = &self.names;
        let m = blocks.last().map_or(0, |b| b.at + b.m);
        if !own && ids.len() != m {
            return Err(GpuError::shape(
                WHAT,
                format!("{} ids for a store walk of {m} rows", ids.len()),
            ));
        }
        let (h, cols) = (geo::HIDDEN, geo::STREAMS * m);

        // The rows' tokens, the input record's first position 0: each
        // block's embedding takes its first position as its offset.
        put_input(a.inbox.host_mut()?, ids, 0)?;
        a.inbox.upload(stream, IN_IDS + ids.len())?;
        let s = &mut a.st;

        // The embedding rows, a block at a time, and the input pack.
        let table = EmbedRows::of_weight(embd).ok_or_else(|| {
            GpuError::tensor(WHAT, "token_embd.weight", "Q8_0 planes or BF16 words")
        })?;
        let word = size_of::<u32>();
        for b in blocks {
            let block_ids = Window::<u32>::of(a.inbox.dev(), (IN_IDS + b.at) * word, b.m)?;
            let mut y = WindowMut::<f32>::of_mut(&mut s.emb, b.at * h * size_of::<f32>(), b.m * h)?;
            let mut pos = WindowMut::<u32>::of_mut(&mut s.pos, b.at * word, b.m)?;
            let mut n_keys = WindowMut::<u32>::of_mut(&mut s.n_keys, b.at * word, b.m)?;
            k.q38.enqueue_embed_rows(
                stream,
                EmbedArgs {
                    table,
                    ids: if own { &a.own_id } else { &block_ids },
                    pos0: &a.pos0,
                    first: b.pos0 as usize,
                    fault: sink,
                    y: &mut y,
                    pos: &mut pos,
                    n_keys: &mut n_keys,
                },
            )?;
        }
        for b in blocks {
            splice_media(stream, &mut s.emb, b, h)?;
        }
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
                streams: geo::STREAMS,
                fault: sink,
                out: &mut s.pack,
            },
        )?;

        // `eh_proj` over the packed columns into the streams.
        let tw: &Weights = w;
        let dense = WideKernels {
            gemm: &k.gemm,
            g32: Some(&k.g32),
        };
        site::resident_quantize(
            gpu,
            &dense,
            (tw, &[&n.eh]),
            (&s.pack, cols),
            (None, &mut s.pack_q),
            sink,
        )?;
        k.gemm
            .enqueue_route_dense(stream, cols, &mut s.packed, gpu.unlabelled_sink())?;
        let x = WideIn {
            q128: None,
            q32: Some(&s.pack_q),
            f32: Some(&s.pack),
        };
        site::resident_gemm(gpu, &dense, (tw, &n.eh), &x, &s.packed, cols, &mut s.res)?;

        // The attention site's mix, then k and v over the rows.
        k.gemm
            .enqueue_route_dense(stream, m, &mut s.rows, gpu.unlabelled_sink())?;
        let site = &n.attn_site;
        let inject = match &site.inject {
            Some(t) => Some(f32_tensor(w, t)?),
            None => None,
        };
        k.hcw.enqueue_mix(
            stream,
            WideMixArgs {
                res: Streams::F32(&mut s.res),
                before: Before::Plain,
                w: SiteWeights {
                    gamma: f32_gain(w, &site.norm)?,
                    down: HcMatrix::of(w, WHAT, &site.down)?,
                    up: HcMatrix::of(w, WHAT, &site.up)?,
                    inject,
                },
                eps: c.eps,
                m,
                fault: sink,
                scratch: &mut s.hc,
                gemm: &k.g32,
                q8f32: gpu.q8f32(),
                dense: &s.rows,
                mixed: &mut s.mixed,
                out: None,
            },
        )?;
        site::resident_quantize(
            gpu,
            &dense,
            (tw, &[&n.k, &n.v]),
            (&s.mixed, m),
            (None, &mut s.mixed_q),
            sink,
        )?;
        let x = WideIn {
            q128: None,
            q32: Some(&s.mixed_q),
            f32: Some(&s.mixed),
        };
        for (name, y) in [(&n.k, &mut s.k), (&n.v, &mut s.v)] {
            site::resident_gemm(gpu, &dense, (tw, name), &x, &s.rows, m, y)?;
        }
        Ok(())
    }

    /// The keys and values of `b`'s rows, which [`Mtp38::store_rows`] ran,
    /// the keys normed and turned, appended to this sequence's store (the
    /// live one's) at their positions by the append at no query head: the
    /// store then holds the sequence through the block's last row, and no
    /// walk is there to read (a store walk leaves no own row, no streams past
    /// its attention's input and no head).
    pub(super) fn append_store(
        &mut self,
        c: &MtpCtx<'_>,
        b: StoreBlock<'_>,
    ) -> Result<(), GpuError> {
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
        let s = &mut a.st;
        let (gpu, k) = (c.gpu, c.k);
        let sink = gpu.layer_sink(*index as usize)?;
        let n = &self.names;
        let (kv_at, kv_n) = (b.at * geo::KV * size_of::<f32>(), b.m * geo::KV);
        let mut kw = WindowMut::<f32>::of_mut(&mut s.k, kv_at, kv_n)?;
        let v = Window::<f32>::of(&s.v, kv_at, kv_n)?;
        let pos = Window::<u32>::of(&s.pos, b.at * size_of::<u32>(), b.m)?;
        // No row of a store walk has its query read: the launch runs the key
        // heads alone (no query head) and reads and writes no query value,
        // so the live walk's query buffers stand in for its query arguments.
        // SAFETY: the per-sequence embedding supplies distinct cache-valid positions.
        unsafe {
            store.append_256(
                &k.neox,
                gpu.stream(),
                Append256 {
                    qg: &a.qg,
                    q: &mut a.q,
                    k: &mut kw,
                    v: &v,
                    gq: f32_gain(w, &n.q_norm)?,
                    gk: f32_gain(w, &n.k_norm)?,
                    table: c.table,
                    pos: &pos,
                    eps: c.eps,
                    n_head: 0,
                    n_kv: geo::N_KV,
                    ctx: *ctx,
                    m: b.m,
                    fault: sink,
                },
            )?;
        }
        self.last = None;
        self.own_at = None;
        self.held = b.pos0 as usize + b.m;
        Ok(())
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
            (DraftHead::Rows, Some(n)) => n,
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

/// The rows walk `j` of a round of `blocks`' chains runs ([`chain_rows`]):
/// walk 0 each block's refresh at its inbox ids, a later walk each block's
/// one own row, its id the one the block's walk before proposed
/// ([`Mtp38::chain_out`]); each block's position the inbox word
/// [`Mtp38::stage_chain`] wrote for it.
fn walk_rows(blocks: &[ChainBlock], j: usize) -> impl Iterator<Item = (ChainBlock, WalkRows)> + '_ {
    chain_rows(blocks, j).map(move |(b, at, m)| {
        let rows = if j == 0 {
            WalkRows {
                at,
                m,
                ids: IdsAt::Inbox(IN_IDS + at),
                pos0: CHAIN_POS + b.block,
            }
        } else {
            WalkRows {
                at,
                m,
                ids: IdsAt::Chains(2 * (b.pair + j - 1)),
                pos0: CHAIN_POS + j * MTP_ROWS + b.block,
            }
        };
        (b, rows)
    })
}

/// Refuse by name a store walk ([`WalkMode::Store`]) where the head's rows
/// are read back, `why` naming the call.
fn refuse_store(mode: WalkMode, why: &'static str) -> Result<(), GpuError> {
    if mode == WalkMode::Store {
        return Err(GpuError::state(WHAT, why));
    }
    Ok(())
}

/// The rows `feed` walks in `mode`, the one place a walk's rows are taken
/// from it: `None` for the own row, which has no rows of a call's. Media
/// rows (a media prompt call's spans, [`Feed::media`]) are read by a store
/// walk alone, the prompt's warmup, and only at the embedding's width: a
/// walk elsewhere, or one that dropped them, would walk another feed than the
/// target's, so either is refused by name before the walk enqueues anything.
fn call_rows(
    feed: DraftFeed<'_, TargetRows>,
    mode: WalkMode,
) -> Result<Option<Feed<'_, TargetRows>>, GpuError> {
    let DraftFeed::Rows(f) = feed else {
        return Ok(None);
    };
    match f.media {
        Some(_) if mode != WalkMode::Store => Err(GpuError::shape(
            WHAT,
            "a walk with media rows outside a store walk: only a media prompt call's store walks \
             read them",
        )),
        Some(rows) if rows.n_embd() != geo::HIDDEN => Err(GpuError::shape(
            WHAT,
            format!(
                "media rows of {} values a row; the draft's embedding rows are {}",
                rows.n_embd(),
                geo::HIDDEN
            ),
        )),
        _ => Ok(Some(f)),
    }
}

/// `b`'s media rows copied over the embedding rows of their positions in
/// `emb` (rows of `h` values, the block's from its row `b.at`), each run of
/// consecutive span positions one copy, the bf16 rows widened exactly
/// ([`widen_bf16`]); nothing for a block with none. On `stream` behind the
/// block's embedding, which wrote the image token's rows there.
fn splice_media(
    stream: &CudaStream,
    emb: &mut DeviceBuffer<f32>,
    b: &StoreBlock<'_>,
    h: usize,
) -> Result<(), GpuError> {
    let Some(media) = b.media else {
        return Ok(());
    };
    let rows: Vec<Option<&[u16]>> = (b.pos0..).take(b.m).map(|p| media.row(p)).collect();
    let mut at = b.at;
    let mut run = Vec::new();
    for part in rows.chunk_by(|x, y| x.is_some() == y.is_some()) {
        if part[0].is_some() {
            run.resize(part.len() * h, 0.0);
            for (row, out) in part.iter().flatten().zip(run.chunks_exact_mut(h)) {
                widen_bf16(row, out);
            }
            WindowMut::<f32>::of_mut(emb, at * h * size_of::<f32>(), run.len())?
                .copy_from_host(stream, &run)?;
        }
        at += part.len();
    }
    Ok(())
}

/// The readback of a walk of `m` rows: the tokens, then the probabilities'
/// bits, then the fault word and its site mask. Blocking; a raised fault is
/// [`GpuError::Fault`].
fn read_draft(a: &MtpArena, stream: &CudaStream, m: usize) -> Result<Proposal, GpuError> {
    let out = a.out.to_host_vec(stream)?;
    let (Some(&word), Some(&sites)) = (out.get(2 * m), out.get(2 * m + 1)) else {
        return Err(GpuError::state(WHAT, "a readback of 2m + 2 words"));
    };
    if let Some(fault) = Fault::from_words(word, sites) {
        return Err(GpuError::fault(WHAT, fault));
    }
    Ok(Proposal {
        tokens: out[..m].to_vec(),
        p: out[m..2 * m].iter().map(|&b| f32::from_bits(b)).collect(),
    })
}

// A walk's rows fit the eh_proj run and the m-column kernels; a store
// walk's inbox holds an eager walk's rows too.
const _: () = assert!(
    MTP_ROWS <= 8 && MTP_GRAPH_ROWS <= MTP_ROWS && EH_COLS == 8 && MTP_ROWS <= MTP_STORE_ROWS
);
// A round of chains' positions fit the inbox beside its ids.
const _: () = assert!(CHAIN_POS + MTP_GRAPH_ROWS * MTP_ROWS <= IN_IDS + MTP_STORE_ROWS);

#[cfg(test)]
mod tests {
    use super::{TargetRows, call_rows, geo};
    use crate::draft::{DraftFeed, Feed, Hidden, WalkMode};
    use crate::media::CallRows;

    /// A walk reads an arena's rows only where both its record and its final
    /// rows hold the positions before the walk's.
    #[test]
    fn target_rows_are_read_where_their_final_rows_hold_them() {
        use super::{Held, TargetRows, check_target_rows};

        let check = |rows, pos0| {
            check_target_rows(
                "test",
                TargetRows::Pass,
                (Held::at(9, 4), Held::at(10, 3), 8),
                rows,
                pos0,
            )
        };
        check((1, 2), 11).expect("rows 1 and 2 hold the final positions 10 and 11");
        for (rows, pos0, says) in [
            ((0, 2), 11, "as positions 10..12"),
            ((0, 2), 10, "final hidden rows hold positions 10..13"),
            ((3, 2), 13, "as positions 12..14"),
            ((7, 2), 17, "2 rows from row 7"),
            ((1, 2), 0, "at position 0"),
        ] {
            let why = check(rows, pos0).unwrap_err().to_string();
            assert!(why.contains(says), "{rows:?} from {pos0}: {why}");
        }
    }

    /// Every carried draft row and its position survive the byte form.
    #[test]
    fn draft_rows_go_to_bytes_and_back() {
        use super::{DraftRows38, Held, Put, Take};

        let rows = DraftRows38 {
            held: 7,
            step: (Held::at(6, 1), vec![1.25, -2.5]),
            pass: (Held::at(3, 4), vec![3.5, 4.25, -5.0, 6.5]),
            final_rows: [Held::at(6, 1), Held::at(4, 3)],
            point_rows: vec![(2, vec![7.5, -8.25]), (5, vec![9.0, 10.5])],
        };
        let mut b = Vec::new();
        let mut w = Put::new(&mut b);
        rows.put(&mut w).unwrap();
        w.flush().unwrap();
        let mut input = b.as_slice();
        let mut r = Take::new(&mut input);
        let back = DraftRows38::take(&mut r).unwrap();
        r.end().unwrap();
        assert_eq!(back.held, rows.held, "the draft store keeps its position");
        assert_eq!(
            back.step, rows.step,
            "the step arena keeps its positions and rows"
        );
        assert_eq!(
            back.pass, rows.pass,
            "the pass arena keeps its positions and rows"
        );
        assert_eq!(
            back.final_rows, rows.final_rows,
            "both arenas keep their final row ranges"
        );
        assert_eq!(
            back.point_rows, rows.point_rows,
            "every point keeps its position and hidden row"
        );
    }

    /// Media rows are a store walk's alone, at the embedding's width: a store
    /// walk takes them, an eager or a captured walk refuses them by name
    /// rather than walk without them, and rows of another width are refused
    /// by name in any mode. A feed with none gives its rows, and the own row
    /// has none. Mutant: the store walk refused as before (no media prompt
    /// call), or the width left unchecked.
    #[test]
    fn media_rows_are_a_store_walks_alone_at_the_embedding_width() {
        let at = 0..1;
        let (wide, narrow) = (vec![0u16; geo::HIDDEN], [0u16; 2]);
        let media = CallRows::new(5, geo::HIDDEN, [(&at, &wide[..])]).expect("one whole row");
        let other = CallRows::new(5, 2, [(&at, &narrow[..])]).expect("one whole row");
        let feed = |media| {
            DraftFeed::Rows(Feed {
                tokens: &[1],
                pos0: 0,
                hidden: Hidden::Target {
                    walk: TargetRows::Step,
                    first: 0,
                },
                media,
            })
        };
        assert!(
            matches!(call_rows(feed(Some(&media)), WalkMode::Store), Ok(Some(f)) if f.media.is_some()),
            "a store walk reads the media rows"
        );
        for mode in [WalkMode::Eager, WalkMode::Graph] {
            let e = call_rows(feed(Some(&media)), mode).expect_err("media outside a store walk");
            assert!(e.to_string().contains("outside a store walk"), "{e}");
        }
        let e = call_rows(feed(Some(&other)), WalkMode::Store).expect_err("another width");
        assert!(
            e.to_string().contains("media rows of 2 values a row"),
            "{e}"
        );
        assert!(
            matches!(call_rows(feed(None), WalkMode::Eager), Ok(Some(f)) if f.tokens == [1]),
            "a feed with no media"
        );
        assert!(
            matches!(
                call_rows(DraftFeed::Own { pos0: 3 }, WalkMode::Graph),
                Ok(None)
            ),
            "the own row has no rows"
        );
    }
}
