//! The launch of the routed gate-up pair and the folded down.
//!
//! Both are one dispatch. The pair's grid height is the slot. The fold maps
//! every slot's blocks as one K' and adds the residual in the same kernel.
//! A launch whose shape is not that one is refused by name before anything
//! is encoded.

use bloomery_device::{DeviceError, Elems, Window, WindowMut};

use crate::buffer::MetalRaw;
use crate::device::MetalDevice;
use crate::encode::{Binding, ComputeEncoder};
use crate::fault::MetalFaultWord;

use super::geometry::{BLOCK_VALUES, ROWS_PER_SIMD_GROUP, SIMD_WIDTH};
use super::{Quant, args_bytes, check_window_len, word};

/// Slots in one launch: eight routed experts and the shared expert.
pub const EXPERT_SLOTS: usize = 9;
/// Ids a launch reads. The shared slot has none.
pub const ROUTED_IDS: usize = 8;
/// The slot that reads the shared expert.
pub const SHARED_SLOT: usize = 8;
/// The expert intermediate width. The pair's rows and the fold's K are this.
pub const EXPERT_K: usize = 512;

/// SIMD groups in an expert threadgroup. Hardcoded: the pipelines are not
/// retiled when a mat-vec launch asks for another count.
const EXPERT_NSG: usize = 2;
const EXPERT_THREADS: usize = SIMD_WIDTH * EXPERT_NSG;
/// Blocks of K': nine slots, two blocks of [`EXPERT_K`] each.
const FOLDED_BLOCKS: usize = EXPERT_SLOTS * (EXPERT_K / BLOCK_VALUES);
const _: () = assert!(FOLDED_BLOCKS == 18);

/// The shared expert, or a refusal when its quant is not the routed experts'.
#[derive(Clone, Copy)]
pub enum SharedPlan<'w> {
    /// One expert, in the routed experts' quant.
    Same(Window<'w, u8, MetalDevice>),
    /// The shared expert's quant differs. The launch is refused by name.
    Apart,
}

/// One gate-up pair: nine slots, each `silu(gate) * up` over `cols` inputs.
pub struct PairParams<'w> {
    /// The weight type. A layer's experts are one type.
    pub quant: Quant,
    /// The slots. A launch runs [`EXPERT_SLOTS`].
    pub slots: usize,
    /// The expert intermediate width. A launch runs [`EXPERT_K`].
    pub k: usize,
    /// The input columns, a positive multiple of 256.
    pub cols: usize,
    /// Experts in the stack. An id at or past this is `ExpertId`.
    pub n_experts: usize,
    /// Eight routed ids, int32. Empty is refused by name.
    pub ids: Window<'w, i32, MetalDevice>,
    /// The gate stack: `n_experts` experts of `k` by `cols`.
    pub gate: Window<'w, u8, MetalDevice>,
    /// The up stack, the gate's shape.
    pub up: Window<'w, u8, MetalDevice>,
    /// The shared expert's gate, one expert.
    pub shared_gate: Window<'w, u8, MetalDevice>,
    /// The shared expert's up, one expert.
    pub shared_up: Window<'w, u8, MetalDevice>,
    /// The input, `cols` values.
    pub x: Window<'w, f32, MetalDevice>,
    /// The nine slot outputs. Slot `s` starts at `s * out_stride`.
    pub y: WindowMut<'w, f32, MetalDevice>,
    /// Floats from one slot's output to the next. Below [`EXPERT_K`] the
    /// slots overlap and the launch is refused.
    pub out_stride: usize,
    /// The fault word an id past the stack raises into.
    pub fault: &'w MetalFaultWord,
    /// The layer a raise is recorded in.
    pub layer: u32,
}

/// One folded down: `y += sum_s w_s * down_s · h_s` over nine slots.
pub struct FoldParams<'w, 'h, 'y, 'buf> {
    /// The weight type. The shared expert is the same type, or [`SharedPlan::Apart`].
    pub quant: Quant,
    /// The shared expert's weights, or a refusal.
    pub shared: SharedPlan<'w>,
    /// The output rows.
    pub rows: usize,
    /// The slots. A launch runs [`EXPERT_SLOTS`].
    pub slots: usize,
    /// The expert width. A launch runs [`EXPERT_K`].
    pub k: usize,
    /// Experts in the stack.
    pub n_experts: usize,
    /// Eight routed ids, int32. The shared slot has none.
    pub ids: Window<'w, i32, MetalDevice>,
    /// The down stack: `n_experts` experts of `rows` by `k`.
    pub down: Window<'w, u8, MetalDevice>,
    /// Nine activation vectors of `k`, concatenated.
    pub h: Window<'h, f32, MetalDevice>,
    /// Nine route weights, one per slot, applied as the block is loaded.
    pub w: Window<'w, f32, MetalDevice>,
    /// The residual. The kernel adds into it. A shared borrow: the binding
    /// reads the window, and a caller can pair it with a read view of the
    /// same buffer when the ranges overlap.
    pub y: &'y WindowMut<'buf, f32, MetalDevice>,
    /// The fault word an id past the stack raises into.
    pub fault: &'w MetalFaultWord,
    /// The layer a raise is recorded in.
    pub layer: u32,
}

fn refuse_pair(detail: String) -> DeviceError {
    DeviceError::Range {
        what: "pair",
        detail,
    }
}

fn refuse_fold(detail: String) -> DeviceError {
    DeviceError::Range {
        what: "fold",
        detail,
    }
}

fn refuse_slots(what: &'static str, slots: usize) -> Result<(), DeviceError> {
    if slots == EXPERT_SLOTS {
        return Ok(());
    }
    Err(DeviceError::Range {
        what,
        detail: format!("a launch runs {EXPERT_SLOTS} slots, not {slots}"),
    })
}

fn refuse_k(what: &'static str, k: usize) -> Result<(), DeviceError> {
    if k == EXPERT_K {
        return Ok(());
    }
    Err(DeviceError::Range {
        what,
        detail: format!("the expert width is {k}, not {EXPERT_K}"),
    })
}

fn refuse_stack(what: &'static str, n_experts: usize) -> Result<(), DeviceError> {
    if n_experts == 0 {
        return Err(DeviceError::Range {
            what,
            detail: "the expert stack is empty".to_string(),
        });
    }
    Ok(())
}

fn refuse_ids(what: &'static str, ids: Elems) -> Result<(), DeviceError> {
    if ids.get() == 0 {
        return Err(DeviceError::Range {
            what,
            detail: "the id list is empty".to_string(),
        });
    }
    Ok(())
}

fn weight_bytes(quant: Quant, rows: usize, nblocks: usize, experts: usize) -> Option<usize> {
    rows.checked_mul(nblocks)?
        .checked_mul(experts)?
        .checked_mul(quant.block_bytes())
}

fn aligned(
    what: &'static str,
    name: &str,
    window: Window<'_, u8, MetalDevice>,
) -> Result<(), DeviceError> {
    let start = window.byte_range()?.start.get();
    if start.is_multiple_of(2) {
        return Ok(());
    }
    Err(DeviceError::Range {
        what,
        detail: format!(
            "the {name} weights start {start} bytes into their buffer: the kernels read 16-bit words"
        ),
    })
}

fn hidden_overlaps_output(
    h: Window<'_, f32, MetalDevice>,
    y: &WindowMut<'_, f32, MetalDevice>,
) -> Result<bool, DeviceError> {
    if !MetalRaw::same_buffer(*h.raw(), *y.raw()) {
        return Ok(false);
    }
    let h = h.byte_range()?;
    let y = y.byte_range()?;
    Ok(h.start < y.end && y.start < h.end)
}

fn pack_words(words: &[u32]) -> Vec<u8> {
    let mut out = vec![0u8; words.len() * 4];
    for (chunk, word) in out.as_chunks_mut::<4>().0.iter_mut().zip(words) {
        *chunk = word.to_ne_bytes();
    }
    out
}

struct PairShape {
    nblocks: u32,
    rows: u32,
    n_experts: u32,
    stride: u32,
    groups_x: usize,
}

fn check_pair(p: &PairParams<'_>) -> Result<PairShape, DeviceError> {
    refuse_slots("pair", p.slots)?;
    refuse_k("pair", p.k)?;
    refuse_stack("pair", p.n_experts)?;
    refuse_ids("pair", p.ids.len())?;
    if p.out_stride < EXPERT_K {
        return Err(refuse_pair("the slot outputs overlap".to_string()));
    }
    if !p.cols.is_multiple_of(BLOCK_VALUES) || p.cols == 0 {
        return Err(refuse_pair(format!(
            "{} columns are not whole blocks of {BLOCK_VALUES}",
            p.cols
        )));
    }
    let nblocks = p.cols / BLOCK_VALUES;
    let stack = weight_bytes(p.quant, EXPERT_K, nblocks, p.n_experts).ok_or_else(|| {
        refuse_pair(format!(
            "{} experts of {} columns wrap",
            p.n_experts, p.cols
        ))
    })?;
    let shared = weight_bytes(p.quant, EXPERT_K, nblocks, 1)
        .ok_or_else(|| refuse_pair("one shared expert wraps".to_string()))?;
    let y_len = (EXPERT_SLOTS - 1)
        .checked_mul(p.out_stride)
        .and_then(|n| n.checked_add(EXPERT_K))
        .ok_or_else(|| refuse_pair("the output length wraps".to_string()))?;
    check_window_len("pair", "gate", p.gate.len(), stack)?;
    check_window_len("pair", "up", p.up.len(), stack)?;
    check_window_len("pair", "shared gate", p.shared_gate.len(), shared)?;
    check_window_len("pair", "shared up", p.shared_up.len(), shared)?;
    check_window_len("pair", "x", p.x.len(), p.cols)?;
    check_window_len("pair", "ids", p.ids.len(), ROUTED_IDS)?;
    check_window_len("pair", "y", p.y.len(), y_len)?;
    aligned("pair", "gate", p.gate)?;
    aligned("pair", "up", p.up)?;
    aligned("pair", "shared gate", p.shared_gate)?;
    aligned("pair", "shared up", p.shared_up)?;
    Ok(PairShape {
        nblocks: word("pair", "the block count", nblocks)?,
        rows: word("pair", "the row count", EXPERT_K)?,
        n_experts: word("pair", "the expert count", p.n_experts)?,
        stride: word("pair", "the output stride", p.out_stride)?,
        groups_x: EXPERT_K / EXPERT_NSG,
    })
}

struct FoldShape {
    rows: u32,
    n_experts: u32,
    groups_x: usize,
}

fn check_fold<'w>(
    p: &FoldParams<'w, '_, '_, '_>,
) -> Result<(FoldShape, Window<'w, u8, MetalDevice>), DeviceError> {
    let shared = match p.shared {
        SharedPlan::Apart => {
            return Err(refuse_fold(
                "the shared expert is SharedPlan::Apart".to_string(),
            ));
        }
        SharedPlan::Same(window) => window,
    };
    refuse_slots("fold", p.slots)?;
    refuse_k("fold", p.k)?;
    refuse_stack("fold", p.n_experts)?;
    refuse_ids("fold", p.ids.len())?;
    if hidden_overlaps_output(p.h, p.y)? {
        return Err(refuse_fold(
            "the output overlaps the hidden states".to_string(),
        ));
    }
    if p.rows == 0 {
        return Err(refuse_fold("no rows".to_string()));
    }
    let nblocks = EXPERT_K / BLOCK_VALUES;
    let stack = weight_bytes(p.quant, p.rows, nblocks, p.n_experts)
        .ok_or_else(|| refuse_fold(format!("{} experts of {} rows wrap", p.n_experts, p.rows)))?;
    let shared_bytes = weight_bytes(p.quant, p.rows, nblocks, 1)
        .ok_or_else(|| refuse_fold("one shared expert wraps".to_string()))?;
    let hidden = EXPERT_SLOTS * EXPERT_K;
    check_window_len("fold", "down", p.down.len(), stack)?;
    check_window_len("fold", "shared down", shared.len(), shared_bytes)?;
    check_window_len("fold", "hidden", p.h.len(), hidden)?;
    check_window_len("fold", "route", p.w.len(), EXPERT_SLOTS)?;
    check_window_len("fold", "ids", p.ids.len(), ROUTED_IDS)?;
    check_window_len("fold", "y", p.y.len(), p.rows)?;
    aligned("fold", "down", p.down)?;
    aligned("fold", "shared down", shared)?;
    let rows_per_group = EXPERT_NSG * ROWS_PER_SIMD_GROUP;
    Ok((
        FoldShape {
            rows: word("fold", "the row count", p.rows)?,
            n_experts: word("fold", "the expert count", p.n_experts)?,
            groups_x: p.rows.div_ceil(rows_per_group),
        },
        shared,
    ))
}

impl super::Kernels {
    /// Encode one gate-up pair. Nine slots, one dispatch; nothing is encoded
    /// for a launch that is refused.
    #[must_use = "a refused launch encoded nothing"]
    pub fn pair(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: PairParams<'_>,
    ) -> Result<(), DeviceError> {
        let shape = check_pair(&p)?;
        let pipe = match p.quant {
            Quant::Q4K => &self.pair_q4,
            Quant::Q6K => &self.pair_q6,
        };
        let args = pack_words(&[
            shape.rows,
            shape.nblocks,
            shape.n_experts,
            p.layer,
            shape.stride,
        ]);
        encoder.set_pipeline(pipe);
        encoder.set_binding(0, Binding::of(&p.gate)?)?;
        encoder.set_binding(1, Binding::of(&p.up)?)?;
        encoder.set_binding(2, Binding::of(&p.shared_gate)?)?;
        encoder.set_binding(3, Binding::of(&p.shared_up)?)?;
        encoder.set_binding(4, Binding::of(&p.x)?)?;
        encoder.set_binding(5, Binding::of(&p.ids)?)?;
        encoder.set_binding(6, Binding::of_mut(&p.y)?)?;
        encoder.set_binding(7, p.fault.binding())?;
        encoder.set_binding(8, Binding::bytes(&args)?)?;
        encoder.dispatch_grid(shape.groups_x, EXPERT_SLOTS, EXPERT_THREADS)
    }

    /// Encode one folded down. The route weights scale the load, and the
    /// residual add is the store; nothing is encoded for a launch that is
    /// refused.
    #[must_use = "a refused launch encoded nothing"]
    pub fn fold(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: FoldParams<'_, '_, '_, '_>,
    ) -> Result<(), DeviceError> {
        let (shape, shared) = check_fold(&p)?;
        let pipe = match p.quant {
            Quant::Q4K => &self.fold_q4,
            Quant::Q6K => &self.fold_q6,
        };
        let args = args_bytes([shape.rows, shape.n_experts, p.layer, 0]);
        encoder.set_pipeline(pipe);
        encoder.set_binding(0, Binding::of(&p.down)?)?;
        encoder.set_binding(1, Binding::of(&shared)?)?;
        encoder.set_binding(2, Binding::of(&p.h)?)?;
        encoder.set_binding(3, Binding::of(&p.w)?)?;
        encoder.set_binding(4, Binding::of(&p.ids)?)?;
        encoder.set_binding(5, Binding::of_mut(p.y)?)?;
        encoder.set_binding(6, p.fault.binding())?;
        encoder.set_binding(7, Binding::bytes(&args)?)?;
        encoder.dispatch(shape.groups_x * EXPERT_THREADS, EXPERT_THREADS)
    }
}
