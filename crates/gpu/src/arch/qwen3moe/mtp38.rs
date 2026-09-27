//! Qwen3.8's MTP draft layer resident on the target's card ([`Mtp38`]): its
//! weights, its store and the reduced head's rows, opened once at load from
//! the plan `model::arch::qwen35moe::place::PlanInputs::plan_mtp` made. No
//! step runs here; the draft's program reads what this holds.
//!
//! - The draft file's tensors the plan puts on the card are uploaded by the
//!   placement loader from the draft's plan rows, each checked against its
//!   segment's bytes; the two injects are decoded to f32 and filed as the
//!   derived weights `place::mtp_widened` names; the router and the shared
//!   expert's gate are joined as the target's are (`plan38::router`).
//! - `token_embd` and `output` are the target's: nothing is uploaded for
//!   them. The body holds their names and the device addresses they had at
//!   open; the target's `Weights`, which `GpuModel` drops after the body,
//!   own the buffers, and a walk reads them through the `&Weights` it is
//!   handed ([`Mtp38::borrowed`]), which refuses by name an address that
//!   moved.
//! - The store is every position's K and V, `[n_kv][ctx][head]` f16 each,
//!   the target's attention layers' K/V planes ([`KvPlanes`]).
//! - With a row list, the head's rows of the target's `output.weight` are
//!   gathered from the target file on the host into one Q8_0 matrix
//!   (`place::MTP_HEAD_ROWS`), uploaded once, beside the row → id map; with
//!   the full head, nothing.

use super::plan38::{geo, router};
use super::scratch::KvPlanes;
use crate::weights::{DevWeight, Q8Block, Weights, q8_0_planes};
use crate::{DeviceTensor, Gpu, GpuError};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use gguf::quant::{GgmlType, dequant_row};
use model::arch::models::{HeadRows, MtpSource};
use model::placement::Plan;

const WHAT: &str = "qwen4exp Mtp38";

/// A matrix of the target's the draft reads in place: its name and its two
/// Q8_0 planes' device addresses at open.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BorrowedPlanes {
    pub name: String,
    /// The code plane's and the scale plane's `CUdeviceptr`.
    pub planes: [u64; 2],
}

/// The reduced head: `output.weight`'s rows of the list, one Q8_0 matrix in
/// `w` under `place::MTP_HEAD_ROWS`, and each row's vocabulary id.
struct HeadMap {
    ids: DeviceBuffer<u32>,
    rows: usize,
}

/// The MTP layer on the target's card. See the module comment.
pub struct Mtp38 {
    /// The draft's own tensors, the widened injects, the joined router and,
    /// with a row list, the head's rows.
    w: Weights,
    store: KvPlanes,
    head: Option<HeadMap>,
    /// `token_embd`, then `output`.
    borrowed: [BorrowedPlanes; 2],
    /// The layer's `blk.` index in the draft file.
    index: u32,
    ctx: usize,
}

impl Mtp38 {
    /// The draft `mtp` describes, resident on `gpu` as `plan`'s draft plan
    /// places it, reading `target_w`'s `token_embd` and `output` and
    /// gathering the head's rows from `target_file`; each segment's file
    /// pages leave the page cache once uploaded when `card_dontneed`.
    /// Refused by name: a draft that is not the target's shape, a draft
    /// file that carries its own matrices, a borrowed matrix absent or not
    /// Q8_0 `[hidden, vocab]`, an upload whose bytes are not the plan's.
    pub fn open(
        gpu: &Gpu,
        target_file: &Split,
        target_w: &Weights,
        draft_file: &Split,
        mtp: &model::arch::qwen35moe::place::MtpInputs,
        plan: &model::arch::qwen35moe::place::MtpPlan<'_>,
        card_dontneed: bool,
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
        let borrowed = [
            model::arch::qwen35moe::names::token_embd(),
            model::arch::qwen35moe::names::output(),
        ]
        .map(|name| borrowed_planes(target_w, name, hidden, vocab));
        let [embd, out] = borrowed;
        let borrowed = [embd?, out?];
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
            HeadRows::Full => None,
            HeadRows::List { ids, .. } => {
                let dw = gathered_rows(stream, target_file, ids, hidden, vocab)?;
                held_as_planned(
                    &plan.draft,
                    model::arch::qwen35moe::place::MTP_HEAD_ROWS,
                    dw.resident_bytes(),
                )?;
                w.insert_derived(model::arch::qwen35moe::place::MTP_HEAD_ROWS.to_string(), dw)?;
                Some(HeadMap {
                    ids: DeviceBuffer::from_host(stream, ids)?,
                    rows: ids.len(),
                })
            }
        };
        let n = geo::N_KV * ctx * geo::HEAD;
        let store = KvPlanes {
            k: DeviceBuffer::zeroed(stream, n)?,
            v: DeviceBuffer::zeroed(stream, n)?,
        };
        stream.synchronize()?;
        let body = Mtp38 {
            w,
            store,
            head,
            borrowed,
            index: d.index,
            ctx,
        };
        let c = &plan.draft.cards[0];
        let got = [
            body.w.resident_bytes() as u64,
            body.store.bytes() as u64,
            body.map_bytes(),
        ];
        let want = [plan.draft_resident_bytes(), c.kv_bytes, plan.map_bytes];
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
        self.head
            .as_ref()
            .map_or(0, |h| (h.ids.len() * std::mem::size_of::<u32>()) as u64)
    }

    /// The target's matrices the draft reads, with their addresses at open.
    #[must_use]
    pub fn borrowed_planes(&self) -> &[BorrowedPlanes; 2] {
        &self.borrowed
    }

    /// `token_embd` and `output` of `target`, the weights the draft was
    /// opened beside. Refused by name when either is absent or its planes are
    /// not where they were at open.
    pub fn borrowed<'w>(
        &self,
        target: &'w Weights,
    ) -> Result<(&'w DevWeight, &'w DevWeight), GpuError> {
        let get = |b: &BorrowedPlanes| -> Result<&'w DevWeight, GpuError> {
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
        };
        Ok((get(&self.borrowed[0])?, get(&self.borrowed[1])?))
    }

    /// The draft's own weights: its file's tensors by name, the widened
    /// injects, the joined router and the head's rows.
    #[must_use]
    pub fn weights(&self) -> &Weights {
        &self.w
    }

    /// The row → vocabulary id map and its rows; `None` for the full head.
    #[must_use]
    pub fn head_map(&self) -> Option<(&DeviceBuffer<u32>, usize)> {
        self.head.as_ref().map(|h| (&h.ids, h.rows))
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
}

/// A Q8_0 weight's two planes' device addresses; `None` for another format.
fn planes(dw: &DevWeight) -> Option<[u64; 2]> {
    match dw {
        DevWeight::Q8_0 { qs, d, .. } => Some([qs.buf().cu_deviceptr(), d.buf().cu_deviceptr()]),
        _ => None,
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

/// The rows `ids` of `target`'s `output.weight`, a Q8_0 `[hidden, vocab]`
/// matrix, gathered on the host in the list's order and uploaded as one
/// derived Q8_0 matrix.
fn gathered_rows(
    stream: &CudaStream,
    target: &Split,
    ids: &[u32],
    hidden: usize,
    vocab: usize,
) -> Result<DevWeight, GpuError> {
    let name = model::arch::qwen35moe::names::output();
    let (shard, info) = target
        .find(&name)
        .ok_or_else(|| GpuError::tensor(WHAT, name.clone(), "in the target file"))?;
    if info.ty != GgmlType::Q8_0 || info.dims != [hidden as u64, vocab as u64] {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "{name} is {} {:?}; the head's rows are gathered from a Q8_0 [{hidden}, {vocab}] \
                 matrix",
                info.ty, info.dims
            ),
        ));
    }
    let g = target
        .shard(shard)
        .ok_or_else(|| GpuError::shape(WHAT, format!("{name} names shard {shard}")))?;
    let data = g.data(info)?;
    let row = hidden / 32 * 34;
    let mut blocks: Vec<Q8Block> = Vec::with_capacity(ids.len() * hidden / 32);
    for &id in ids {
        let at = id as usize * row;
        let bytes = data.get(at..at + row).ok_or_else(|| {
            GpuError::shape(
                WHAT,
                format!("{name}: row {id} runs past its {} bytes", data.len()),
            )
        })?;
        blocks.extend(bytes.as_chunks::<34>().0.iter().map(Q8Block::from_bytes));
    }
    let (qs, d) = q8_0_planes(&blocks);
    Ok(DevWeight::Q8_0Derived {
        qs: DeviceTensor::upload(stream, &qs, ids.len(), hidden / 4)?,
        d: DeviceTensor::upload(stream, &d, ids.len(), hidden / 32)?,
        k: hidden,
    })
}
