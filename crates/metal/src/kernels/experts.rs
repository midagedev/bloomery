//! The launch of the routed gate-up pair and the folded down.
//!
//! Both are one dispatch. The pair's grid height is the slot, and it norms
//! its input as it loads it. The fold maps every slot's blocks as one K' and
//! adds the residual in the same kernel.
//! A launch whose shape is not that one is refused by name before anything
//! is encoded.

use bloomery_device::{DeviceError, Elems, Window, WindowMut};

use crate::buffer::MetalRaw;
use crate::device::MetalDevice;
use crate::encode::{Binding, Bindings, ComputeEncoder};
use crate::fault::MetalFaultWord;
use crate::geom::ExpertsGeom;
use crate::library::{Library, Pipeline};

use super::{Quant, args_bytes, check_window_len, pack, word};

const PAIR: &str = "pair";
const FOLD: &str = "fold";

/// The shared expert, or a refusal when its quant is not the routed experts'.
#[derive(Clone, Copy)]
pub enum SharedPlan<'w> {
    /// One expert, in the routed experts' quant.
    Same(Window<'w, u8, MetalDevice>),
    /// The shared expert's quant differs. The launch is refused by name.
    Apart,
    /// The layer has no shared expert: the slots are the routed picks, and
    /// the instance's experts have no shared slot.
    None,
}

/// One gate-up pair: the instance's slots (the routed picks, then the
/// shared expert's when the instance has one), each
/// `silu(gate · x̂) * (up · x̂)` over `cols` inputs, `x̂ = x * gain / sqrt(mean(x^2) + eps)`. A norm that finds a value
/// that is not finite, or a zero scale, raises the norm site of `layer` on
/// `fault` and writes NaN to its rows.
pub struct PairParams<'w> {
    /// The weight type. A layer's experts are one type.
    pub quant: Quant,
    /// The slots. A launch runs the instance's (`ExpertsGeom::slots`).
    pub slots: usize,
    /// The expert intermediate width. A launch runs the instance's
    /// (`ExpertsGeom::k`).
    pub k: usize,
    /// The input columns, a positive multiple of the type's block values.
    pub cols: usize,
    /// Experts in the stack. An id at or past this is `ExpertId`.
    pub n_experts: usize,
    /// Eight routed ids, int32. Empty is refused by name.
    pub ids: Window<'w, i32, MetalDevice>,
    /// The gate stack: `n_experts` experts of `k` by `cols`.
    pub gate: Window<'w, u8, MetalDevice>,
    /// The up stack, the gate's shape.
    pub up: Window<'w, u8, MetalDevice>,
    /// The shared expert's gate, one expert: given exactly when the
    /// instance's experts have a shared slot.
    pub shared_gate: Option<Window<'w, u8, MetalDevice>>,
    /// The shared expert's up, one expert, given with the gate.
    pub shared_up: Option<Window<'w, u8, MetalDevice>>,
    /// The input, `cols` values, before the norm.
    pub x: Window<'w, f32, MetalDevice>,
    /// The norm's gain, `cols` values.
    pub gain: Window<'w, f32, MetalDevice>,
    /// The epsilon added to the mean of the squares.
    pub eps: f32,
    /// The slot outputs. Slot `s` starts at `s * out_stride`.
    pub y: WindowMut<'w, f32, MetalDevice>,
    /// Floats from one slot's output to the next. Below the expert width the
    /// slots overlap and the launch is refused.
    pub out_stride: usize,
    /// The fault word an id past the stack or a norm that cannot take its
    /// input raises into.
    pub fault: &'w MetalFaultWord,
    /// The layer a raise is recorded in.
    pub layer: u32,
}

/// One folded down: `y += sum_s w_s * down_s · h_s` over the instance's
/// slots.
pub struct FoldParams<'w, 'h, 'y, 'buf> {
    /// The weight type. The shared expert is the same type, or [`SharedPlan::Apart`].
    pub quant: Quant,
    /// The shared expert's weights, or a refusal.
    pub shared: SharedPlan<'w>,
    /// The output rows.
    pub rows: usize,
    /// The slots. A launch runs the instance's (`ExpertsGeom::slots`).
    pub slots: usize,
    /// The expert width. A launch runs the instance's (`ExpertsGeom::k`).
    pub k: usize,
    /// Experts in the stack.
    pub n_experts: usize,
    /// Eight routed ids, int32. The shared slot has none.
    pub ids: Window<'w, i32, MetalDevice>,
    /// The down stack: `n_experts` experts of `rows` by `k`.
    pub down: Window<'w, u8, MetalDevice>,
    /// One activation vector of `k` a slot, concatenated.
    pub h: Window<'h, f32, MetalDevice>,
    /// One route weight a slot, applied as the block is loaded.
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

fn refuse_pair(detail: impl Into<String>) -> DeviceError {
    super::range(PAIR, detail)
}

fn refuse_fold(detail: impl Into<String>) -> DeviceError {
    super::range(FOLD, detail)
}

fn refuse_slots(g: &ExpertsGeom, what: &'static str, slots: usize) -> Result<(), DeviceError> {
    if slots == g.slots {
        return Ok(());
    }
    Err(super::range(
        what,
        format!("a launch runs {} slots, not {slots}", g.slots),
    ))
}

fn refuse_k(g: &ExpertsGeom, what: &'static str, k: usize) -> Result<(), DeviceError> {
    if k == g.k {
        return Ok(());
    }
    Err(super::range(
        what,
        format!("the expert width is {k}, not {}", g.k),
    ))
}

fn refuse_stack(what: &'static str, n_experts: usize) -> Result<(), DeviceError> {
    if n_experts == 0 {
        return Err(super::range(what, "the expert stack is empty"));
    }
    Ok(())
}

fn refuse_ids(what: &'static str, ids: Elems) -> Result<(), DeviceError> {
    if ids.get() == 0 {
        return Err(super::range(what, "the id list is empty"));
    }
    Ok(())
}

fn weight_bytes(quant: Quant, rows: usize, nblocks: usize, experts: usize) -> Option<usize> {
    super::weight_bytes(quant, rows, nblocks)?.checked_mul(experts)
}

fn aligned(
    what: &'static str,
    name: &str,
    window: Window<'_, u8, MetalDevice>,
) -> Result<(), DeviceError> {
    let start = window.byte_range()?.start.get();
    super::check_weights_align(what, name, start, super::WeightAlign::Half)
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

struct PairShape {
    nblocks: u32,
    rows: u32,
    n_experts: u32,
    stride: u32,
    groups_x: usize,
}

fn check_pair(
    g: &ExpertsGeom,
    p: &PairParams<'_>,
    simdgroups: usize,
    rows_per_simd_group: usize,
) -> Result<PairShape, DeviceError> {
    refuse_slots(g, PAIR, p.slots)?;
    refuse_k(g, PAIR, p.k)?;
    refuse_stack(PAIR, p.n_experts)?;
    refuse_ids(PAIR, p.ids.len())?;
    if p.out_stride < g.k {
        return Err(refuse_pair("the slot outputs overlap".to_string()));
    }
    if !p.cols.is_multiple_of(p.quant.values_per_block()) || p.cols == 0 {
        return Err(refuse_pair(format!(
            "{} columns are not whole blocks of {}",
            p.cols,
            p.quant.values_per_block()
        )));
    }
    let nblocks = p.cols / p.quant.values_per_block();
    let stack = weight_bytes(p.quant, g.k, nblocks, p.n_experts).ok_or_else(|| {
        refuse_pair(format!(
            "{} experts of {} columns wrap",
            p.n_experts, p.cols
        ))
    })?;
    let shared_bytes = weight_bytes(p.quant, g.k, nblocks, 1)
        .ok_or_else(|| refuse_pair("one shared expert wraps".to_string()))?;
    let shared = match (p.shared_gate, p.shared_up, g.has_shared) {
        (Some(gate), Some(up), true) => Some((gate, up)),
        (None, None, false) => None,
        (None, None, true) => {
            return Err(refuse_pair(
                "no shared expert, and the experts run a shared slot".to_string(),
            ));
        }
        (Some(_), Some(_), false) => {
            return Err(refuse_pair(
                "a shared expert, and the experts run no shared slot".to_string(),
            ));
        }
        _ => {
            return Err(refuse_pair(
                "a shared expert's gate without its up, or its up without its gate".to_string(),
            ));
        }
    };
    let y_len = (g.slots - 1)
        .checked_mul(p.out_stride)
        .and_then(|n| n.checked_add(g.k))
        .ok_or_else(|| refuse_pair("the output length wraps".to_string()))?;
    check_window_len(PAIR, "gate", p.gate.len(), stack)?;
    check_window_len(PAIR, "up", p.up.len(), stack)?;
    if let Some((gate, up)) = shared {
        check_window_len(PAIR, "shared gate", gate.len(), shared_bytes)?;
        check_window_len(PAIR, "shared up", up.len(), shared_bytes)?;
        aligned(PAIR, "shared gate", gate)?;
        aligned(PAIR, "shared up", up)?;
    }
    check_window_len(PAIR, "x", p.x.len(), p.cols)?;
    check_window_len(PAIR, "gain", p.gain.len(), p.cols)?;
    check_window_len(PAIR, "ids", p.ids.len(), g.routed_ids)?;
    check_window_len(PAIR, "y", p.y.len(), y_len)?;
    aligned(PAIR, "gate", p.gate)?;
    aligned(PAIR, "up", p.up)?;
    Ok(PairShape {
        nblocks: word(PAIR, "the block count", nblocks)?,
        rows: word(PAIR, "the row count", g.k)?,
        n_experts: word(PAIR, "the expert count", p.n_experts)?,
        stride: word(PAIR, "the output stride", p.out_stride)?,
        // A SIMD group owns the geometry's weight rows, a gate and an up to
        // an output.
        groups_x: g.k.div_ceil(simdgroups * rows_per_simd_group / 2),
    })
}

struct FoldShape {
    rows: u32,
    n_experts: u32,
    groups_x: usize,
}

fn check_fold<'w>(
    g: &ExpertsGeom,
    p: &FoldParams<'w, '_, '_, '_>,
    simdgroups: usize,
    rows_per_simd_group: usize,
) -> Result<(FoldShape, Window<'w, u8, MetalDevice>), DeviceError> {
    let shared = match (p.shared, g.has_shared) {
        (SharedPlan::Apart, _) => {
            return Err(refuse_fold(
                "the shared expert is SharedPlan::Apart".to_string(),
            ));
        }
        (SharedPlan::Same(window), true) => Some(window),
        (SharedPlan::None, false) => None,
        (SharedPlan::Same(_), false) => {
            return Err(refuse_fold(
                "a shared expert, and the experts run no shared slot".to_string(),
            ));
        }
        (SharedPlan::None, true) => {
            return Err(refuse_fold(
                "no shared expert, and the experts run a shared slot".to_string(),
            ));
        }
    };
    refuse_slots(g, FOLD, p.slots)?;
    refuse_k(g, FOLD, p.k)?;
    refuse_stack(FOLD, p.n_experts)?;
    refuse_ids(FOLD, p.ids.len())?;
    if hidden_overlaps_output(p.h, p.y)? {
        return Err(refuse_fold(
            "the output overlaps the hidden states".to_string(),
        ));
    }
    if p.rows == 0 {
        return Err(refuse_fold("no rows".to_string()));
    }
    let nblocks = g.k / p.quant.values_per_block();
    let stack = weight_bytes(p.quant, p.rows, nblocks, p.n_experts)
        .ok_or_else(|| refuse_fold(format!("{} experts of {} rows wrap", p.n_experts, p.rows)))?;
    let shared_bytes = weight_bytes(p.quant, p.rows, nblocks, 1)
        .ok_or_else(|| refuse_fold("one shared expert wraps".to_string()))?;
    let hidden = g.slots * g.k;
    check_window_len(FOLD, "down", p.down.len(), stack)?;
    if let Some(shared) = shared {
        check_window_len(FOLD, "shared down", shared.len(), shared_bytes)?;
        aligned(FOLD, "shared down", shared)?;
    }
    check_window_len(FOLD, "hidden", p.h.len(), hidden)?;
    check_window_len(FOLD, "route", p.w.len(), g.slots)?;
    check_window_len(FOLD, "ids", p.ids.len(), g.routed_ids)?;
    check_window_len(FOLD, "y", p.y.len(), p.rows)?;
    aligned(FOLD, "down", p.down)?;
    let rows_per_group = simdgroups * rows_per_simd_group;
    Ok((
        FoldShape {
            rows: word(FOLD, "the row count", p.rows)?,
            n_experts: word(FOLD, "the expert count", p.n_experts)?,
            groups_x: p.rows.div_ceil(rows_per_group),
        },
        // An instance with no shared slot binds the stack in its place,
        // which the kernel does not read.
        shared.unwrap_or(p.down),
    ))
}

/// The pair and the fold, one pipeline per quant.
pub(crate) struct Pipelines {
    pair_q4: Pipeline,
    pair_q6: Pipeline,
    pair_a4: Pipeline,
    pair_q8_0: Pipeline,
    pair_q5k: Pipeline,
    fold_q4: Pipeline,
    fold_q6: Pipeline,
    fold_a4: Pipeline,
    fold_q8_0: Pipeline,
    fold_q5k: Pipeline,
}

impl Pipelines {
    /// `simdgroups` is the mat-vec function constant: the pair and the fold
    /// are mat-vec-shaped kernels whose threadgroup width is the same count.
    pub(crate) fn new(library: &Library, simdgroups: u32) -> Result<Self, DeviceError> {
        Ok(Self {
            pair_q4: library.pipeline_with_simdgroups("q4k_pair_swiglu", simdgroups)?,
            pair_q6: library.pipeline_with_simdgroups("q6k_pair_swiglu", simdgroups)?,
            pair_a4: library.pipeline_with_simdgroups("a4_pair_swiglu", simdgroups)?,
            pair_q8_0: library.pipeline_with_simdgroups("q8_0_pair_swiglu", simdgroups)?,
            pair_q5k: library.pipeline_with_simdgroups("q5k_pair_swiglu", simdgroups)?,
            fold_q4: library.pipeline_with_simdgroups("q4k_down_fold", simdgroups)?,
            fold_q6: library.pipeline_with_simdgroups("q6k_down_fold", simdgroups)?,
            fold_a4: library.pipeline_with_simdgroups("a4_down_fold", simdgroups)?,
            fold_q8_0: library.pipeline_with_simdgroups("q8_0_down_fold", simdgroups)?,
            fold_q5k: library.pipeline_with_simdgroups("q5k_down_fold", simdgroups)?,
        })
    }

    fn pair(&self, quant: Quant) -> &Pipeline {
        match quant {
            Quant::Q4K => &self.pair_q4,
            Quant::Q6K => &self.pair_q6,
            Quant::A4 => &self.pair_a4,
            Quant::Q8_0 => &self.pair_q8_0,
            Quant::Q5K => &self.pair_q5k,
        }
    }

    fn fold(&self, quant: Quant) -> &Pipeline {
        match quant {
            Quant::Q4K => &self.fold_q4,
            Quant::Q6K => &self.fold_q6,
            Quant::A4 => &self.fold_a4,
            Quant::Q8_0 => &self.fold_q8_0,
            Quant::Q5K => &self.fold_q5k,
        }
    }
}

impl super::Kernels {
    /// Encode one gate-up pair. The instance's slots, one dispatch; nothing
    /// is encoded for a launch that is refused.
    #[must_use = "a refused launch encoded nothing"]
    pub fn pair(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: PairParams<'_>,
    ) -> Result<(), DeviceError> {
        let shape = check_pair(
            &self.geom.experts,
            &p,
            self.simdgroups,
            self.rows_per_simd_group,
        )?;
        let args = args_bytes([
            shape.rows,
            shape.nblocks,
            shape.n_experts,
            p.layer,
            shape.stride,
            p.eps.to_bits(),
        ]);
        encoder.set_pipeline(self.experts.pair(p.quant));
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.gate)?)?;
        slot.set(Binding::of(&p.up)?)?;
        // An instance with no shared slot binds the stacks in their places,
        // which the kernel does not read.
        slot.set(Binding::of(p.shared_gate.as_ref().unwrap_or(&p.gate))?)?;
        slot.set(Binding::of(p.shared_up.as_ref().unwrap_or(&p.up))?)?;
        slot.set(Binding::of(&p.x)?)?;
        slot.set(Binding::of(&p.gain)?)?;
        slot.set(Binding::of(&p.ids)?)?;
        slot.set(Binding::of_mut(&p.y)?)?;
        slot.set(p.fault.binding())?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.dispatch_groups(
            shape.groups_x,
            self.geom.experts.slots,
            self.threads_per_group(),
        )
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
        let (shape, shared) = check_fold(
            &self.geom.experts,
            &p,
            self.simdgroups,
            self.rows_per_simd_group,
        )?;
        let args = args_bytes([shape.rows, shape.n_experts, p.layer, 0]);
        encoder.set_pipeline(self.experts.fold(p.quant));
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.down)?)?;
        slot.set(Binding::of(&shared)?)?;
        slot.set(Binding::of(&p.h)?)?;
        slot.set(Binding::of(&p.w)?)?;
        slot.set(Binding::of(&p.ids)?)?;
        slot.set(Binding::of_mut(p.y)?)?;
        slot.set(p.fault.binding())?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.dispatch_groups(shape.groups_x, 1, self.threads_per_group())
    }
}
