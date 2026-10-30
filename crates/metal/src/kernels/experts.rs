//! The launch of the routed gate-up pair and the folded down.
//!
//! Both are one dispatch. The pair's grid height is the slot, and it norms
//! its input as it loads it. The fold maps every slot's blocks as one K' and
//! adds the residual in the same kernel. A shared expert of another type than
//! the stacks' (a pair of [`APART_GATE_UP`] or [`APART_DOWN`]) is the same dispatch over an entry of two
//! types: the stacks' and the shared expert's.
//! A launch whose shape is not that one is refused by name before anything
//! is encoded.

use bloomery_device::{DeviceError, Elems, Window, WindowMut};

use crate::buffer::MetalRaw;
use crate::device::MetalDevice;
use crate::encode::{Binding, Bindings, ComputeEncoder};
use crate::fault::MetalFaultWord;
use crate::geom::ExpertsGeom;
use crate::library::{Library, Pipeline};

use super::{LoadNorm, Quant, args_bytes, check_window_len, pack, word};

const PAIR: &str = "pair";
const FOLD: &str = "fold";

/// The (gate and up stacks, shared gate and up) type pairs whose shared
/// expert is read in a type of its own: the pair's entries and the grouped
/// GEMM's pair tile. A pair of types outside the table is refused by name.
pub const APART_GATE_UP: [(Quant, Quant); 2] =
    [(Quant::Q4K, Quant::Q8_0), (Quant::Q5K, Quant::Q8_0)];

/// The (down stack, shared down) type pairs whose shared expert is read in a
/// type of its own: the fold's entries and the grouped GEMM's plain tile.
pub const APART_DOWN: [(Quant, Quant); 2] = [(Quant::Q5K, Quant::Q8_0), (Quant::Q6K, Quant::Q8_0)];

/// Whether a shared expert of type `shared` beside stacks of type `stack` has
/// an entry: the same type, or a pair of the launch's table
/// ([`APART_GATE_UP`] for the gate-up launches, [`APART_DOWN`] for the down
/// launches).
#[must_use]
pub fn shared_entry(stack: Quant, shared: Quant, gate_up: bool) -> bool {
    let table = if gate_up { &APART_GATE_UP } else { &APART_DOWN };
    stack == shared || table.contains(&(stack, shared))
}

/// The shared expert, in the routed experts' quant or in one of its own.
#[derive(Clone, Copy)]
pub enum SharedPlan<'w> {
    /// One expert, in the routed experts' quant.
    Same(Window<'w, u8, MetalDevice>),
    /// One expert in a quant of its own. A pair outside [`APART_DOWN`] is refused
    /// by name.
    Apart {
        quant: Quant,
        weights: Window<'w, u8, MetalDevice>,
    },
    /// The layer has no shared expert: the slots are the routed picks, and
    /// the instance's experts have no shared slot.
    None,
}

/// One gate-up pair: the instance's slots (the routed picks, then the
/// shared expert's when the instance has one), each
/// `silu(gate · x̂) * (up · x̂)` over `cols` inputs, `x̂ = x * gain / sqrt(mean(x^2) + eps)`. A norm that finds a value
/// that is not finite, or a zero scale, raises the norm site of `layer` on
/// `fault` and writes NaN to its rows. A [`LoadNorm::Scaled`] launch reads
/// `x · gain` and the scale its join wrote; the rows are the same bits.
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
    /// The type of the shared expert's gate and up: `quant`, or the second of
    /// a pair of [`APART_GATE_UP`]. An instance with no shared slot takes `quant`.
    pub shared_quant: Quant,
    /// The shared expert's gate, one expert: given exactly when the
    /// instance's experts have a shared slot.
    pub shared_gate: Option<Window<'w, u8, MetalDevice>>,
    /// The shared expert's up, one expert, given with the gate.
    pub shared_up: Option<Window<'w, u8, MetalDevice>>,
    /// The input, `cols` values: before the norm, or `x · gain` for a scaled
    /// launch.
    pub x: Window<'w, f32, MetalDevice>,
    /// The norm: its gain, or the scale of the join that applied it.
    pub norm: LoadNorm<'w>,
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
    /// The weight type of the stacks.
    pub quant: Quant,
    /// The shared expert's weights and type.
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
    quant: Quant,
    window: Window<'_, u8, MetalDevice>,
) -> Result<(), DeviceError> {
    let start = window.byte_range()?.start.get();
    super::check_weights_align(what, name, start, quant.weight_align())
}

/// An activation window the kernels read as `float4`s: its first byte is a
/// multiple of 16 in its buffer.
fn float4_aligned(
    what: &'static str,
    name: &str,
    window: Window<'_, f32, MetalDevice>,
) -> Result<(), DeviceError> {
    let start = window.byte_range()?.start.get();
    super::check_weights_align(what, name, start, super::WeightAlign::Float4)
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
    if !shared_entry(p.quant, p.shared_quant, true) {
        return Err(refuse_pair(format!(
            "stacks of {} and a shared expert of {}: no entry",
            p.quant.name(),
            p.shared_quant.name()
        )));
    }
    for quant in [p.quant, p.shared_quant] {
        if !p.cols.is_multiple_of(quant.values_per_block()) || p.cols == 0 {
            return Err(refuse_pair(format!(
                "{} columns are not whole blocks of {}",
                p.cols,
                quant.values_per_block()
            )));
        }
    }
    let nblocks = p.cols / p.quant.values_per_block();
    let stack = weight_bytes(p.quant, g.k, nblocks, p.n_experts).ok_or_else(|| {
        refuse_pair(format!(
            "{} experts of {} columns wrap",
            p.n_experts, p.cols
        ))
    })?;
    let shared_bytes = weight_bytes(
        p.shared_quant,
        g.k,
        p.cols / p.shared_quant.values_per_block(),
        1,
    )
    .ok_or_else(|| refuse_pair("one shared expert wraps".to_string()))?;
    if !g.has_shared && p.shared_quant != p.quant {
        return Err(refuse_pair(format!(
            "a shared expert of {}, and the experts run no shared slot",
            p.shared_quant.name()
        )));
    }
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
        aligned(PAIR, "shared gate", p.shared_quant, gate)?;
        aligned(PAIR, "shared up", p.shared_quant, up)?;
    }
    check_window_len(PAIR, "x", p.x.len(), p.cols)?;
    p.norm.check(PAIR, p.cols)?;
    check_window_len(PAIR, "ids", p.ids.len(), g.routed_ids)?;
    check_window_len(PAIR, "y", p.y.len(), y_len)?;
    aligned(PAIR, "gate", p.quant, p.gate)?;
    aligned(PAIR, "up", p.quant, p.up)?;
    float4_aligned(PAIR, "x", p.x)?;
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
) -> Result<(FoldShape, Quant, Window<'w, u8, MetalDevice>), DeviceError> {
    if let SharedPlan::Apart { quant, .. } = p.shared
        && !shared_entry(p.quant, quant, false)
    {
        return Err(refuse_fold(format!(
            "stacks of {} and a shared expert of {}: no entry",
            p.quant.name(),
            quant.name()
        )));
    }
    let shared = match (p.shared, g.has_shared) {
        (SharedPlan::Same(window), true) => Some((p.quant, window)),
        (SharedPlan::Apart { quant, weights }, true) => Some((quant, weights)),
        (SharedPlan::None, false) => None,
        (SharedPlan::Same(_) | SharedPlan::Apart { .. }, false) => {
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
    let hidden = g.slots * g.k;
    check_window_len(FOLD, "down", p.down.len(), stack)?;
    if let Some((quant, shared)) = shared {
        if !g.k.is_multiple_of(quant.values_per_block()) {
            return Err(refuse_fold(format!(
                "the expert width {} is not whole blocks of {}",
                g.k,
                quant.values_per_block()
            )));
        }
        let shared_bytes = weight_bytes(quant, p.rows, g.k / quant.values_per_block(), 1)
            .ok_or_else(|| refuse_fold("one shared expert wraps".to_string()))?;
        check_window_len(FOLD, "shared down", shared.len(), shared_bytes)?;
        aligned(FOLD, "shared down", quant, shared)?;
    }
    check_window_len(FOLD, "hidden", p.h.len(), hidden)?;
    check_window_len(FOLD, "route", p.w.len(), g.slots)?;
    check_window_len(FOLD, "ids", p.ids.len(), g.routed_ids)?;
    check_window_len(FOLD, "y", p.y.len(), p.rows)?;
    aligned(FOLD, "down", p.quant, p.down)?;
    float4_aligned(FOLD, "hidden", p.h)?;
    let rows_per_group = simdgroups * rows_per_simd_group;
    // An instance with no shared slot binds the stack in its place, which the
    // kernel does not read.
    let (shared_quant, shared_window) = shared.unwrap_or((p.quant, p.down));
    Ok((
        FoldShape {
            rows: word(FOLD, "the row count", p.rows)?,
            n_experts: word(FOLD, "the expert count", p.n_experts)?,
            groups_x: p.rows.div_ceil(rows_per_group),
        },
        shared_quant,
        shared_window,
    ))
}

/// The pair and the fold, one pipeline per quant.
pub(crate) struct Pipelines {
    pair_q4: Pipeline,
    pair_q6: Pipeline,
    pair_a4: Pipeline,
    pair_a8: Pipeline,
    pair_q8_0: Pipeline,
    pair_q5k: Pipeline,
    /// The pair over a scaled input, one per type: stacks and a shared
    /// expert of that one type.
    pair_scaled: Vec<(Quant, Pipeline)>,
    fold_q4: Pipeline,
    fold_q6: Pipeline,
    fold_a4: Pipeline,
    fold_a8: Pipeline,
    fold_q8_0: Pipeline,
    fold_q5k: Pipeline,
    /// The entries of two types: a pair of [`APART_GATE_UP`] for the pair, of
    /// [`APART_DOWN`] for the fold.
    pair_apart: Vec<((Quant, Quant), Pipeline)>,
    fold_apart: Vec<((Quant, Quant), Pipeline)>,
}

impl Pipelines {
    /// `simdgroups` is the mat-vec function constant: the pair and the fold
    /// are mat-vec-shaped kernels whose threadgroup width is the same count.
    pub(crate) fn new(library: &Library, simdgroups: u32) -> Result<Self, DeviceError> {
        let apart = |kind: &str, table: &[(Quant, Quant)]| {
            table
                .iter()
                .map(|&(stack, shared)| {
                    let name = format!("{}_{kind}_apart_{}", stack.name(), shared.name());
                    Ok((
                        (stack, shared),
                        library.pipeline_with_simdgroups(&name, simdgroups)?,
                    ))
                })
                .collect::<Result<Vec<_>, DeviceError>>()
        };
        Ok(Self {
            pair_q4: library.pipeline_with_simdgroups("q4k_pair_swiglu", simdgroups)?,
            pair_q6: library.pipeline_with_simdgroups("q6k_pair_swiglu", simdgroups)?,
            pair_a4: library.pipeline_with_simdgroups("a4_pair_swiglu", simdgroups)?,
            pair_a8: library.pipeline_with_simdgroups("a8_pair_swiglu", simdgroups)?,
            pair_q8_0: library.pipeline_with_simdgroups("q8_0_pair_swiglu", simdgroups)?,
            pair_q5k: library.pipeline_with_simdgroups("q5k_pair_swiglu", simdgroups)?,
            pair_scaled: [
                Quant::Q4K,
                Quant::Q6K,
                Quant::A4,
                Quant::A8,
                Quant::Q8_0,
                Quant::Q5K,
            ]
            .into_iter()
            .map(|quant| {
                let name = format!("{}_pair_swiglu_scaled", quant.name());
                Ok((quant, library.pipeline_with_simdgroups(&name, simdgroups)?))
            })
            .collect::<Result<Vec<_>, DeviceError>>()?,
            fold_q4: library.pipeline_with_simdgroups("q4k_down_fold", simdgroups)?,
            fold_q6: library.pipeline_with_simdgroups("q6k_down_fold", simdgroups)?,
            fold_a4: library.pipeline_with_simdgroups("a4_down_fold", simdgroups)?,
            fold_a8: library.pipeline_with_simdgroups("a8_down_fold", simdgroups)?,
            fold_q8_0: library.pipeline_with_simdgroups("q8_0_down_fold", simdgroups)?,
            fold_q5k: library.pipeline_with_simdgroups("q5k_down_fold", simdgroups)?,
            pair_apart: apart("pair_swiglu", &APART_GATE_UP)?,
            fold_apart: apart("down_fold", &APART_DOWN)?,
        })
    }

    /// The pair's pipeline for stacks of `quant` and a shared expert of
    /// `shared` over a raw or a `scaled` input, or a named refusal for a type
    /// outside the tables.
    fn pair(&self, quant: Quant, shared: Quant, scaled: bool) -> Result<&Pipeline, DeviceError> {
        if scaled {
            if shared != quant {
                return Err(super::range(
                    PAIR,
                    format!(
                        "stacks of {} and a shared expert of {}: no scaled entry",
                        quant.name(),
                        shared.name()
                    ),
                ));
            }
            return self
                .pair_scaled
                .iter()
                .find(|(q, _)| *q == quant)
                .map(|(_, pipeline)| pipeline)
                .ok_or_else(|| {
                    super::range(PAIR, format!("{} has no scaled entry", quant.name()))
                });
        }
        if shared != quant {
            return Self::apart(&self.pair_apart, quant, shared, PAIR);
        }
        Self::of(quant, PAIR, |q| match q {
            Quant::Q4K => Some(&self.pair_q4),
            Quant::Q6K => Some(&self.pair_q6),
            Quant::A4 => Some(&self.pair_a4),
            Quant::A8 => Some(&self.pair_a8),
            Quant::Q8_0 => Some(&self.pair_q8_0),
            Quant::Q5K => Some(&self.pair_q5k),
            Quant::F32 => None,
        })
    }

    /// The fold's pipeline, as [`Pipelines::pair`].
    fn fold(&self, quant: Quant, shared: Quant) -> Result<&Pipeline, DeviceError> {
        if shared != quant {
            return Self::apart(&self.fold_apart, quant, shared, FOLD);
        }
        Self::of(quant, FOLD, |q| match q {
            Quant::Q4K => Some(&self.fold_q4),
            Quant::Q6K => Some(&self.fold_q6),
            Quant::A4 => Some(&self.fold_a4),
            Quant::A8 => Some(&self.fold_a8),
            Quant::Q8_0 => Some(&self.fold_q8_0),
            Quant::Q5K => Some(&self.fold_q5k),
            Quant::F32 => None,
        })
    }

    fn of<'a>(
        quant: Quant,
        what: &'static str,
        pick: impl FnOnce(Quant) -> Option<&'a Pipeline>,
    ) -> Result<&'a Pipeline, DeviceError> {
        pick(quant).ok_or_else(|| super::range(what, format!("{} has no entry", quant.name())))
    }

    fn apart<'a>(
        table: &'a [((Quant, Quant), Pipeline)],
        quant: Quant,
        shared: Quant,
        what: &'static str,
    ) -> Result<&'a Pipeline, DeviceError> {
        table
            .iter()
            .find(|(pair, _)| *pair == (quant, shared))
            .map(|(_, pipeline)| pipeline)
            .ok_or_else(|| {
                super::range(
                    what,
                    format!(
                        "stacks of {} and a shared expert of {}: no entry",
                        quant.name(),
                        shared.name()
                    ),
                )
            })
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
            p.norm.eps().to_bits(),
        ]);
        encoder.set_pipeline(
            self.experts
                .pair(p.quant, p.shared_quant, p.norm.scaled())?,
        );
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.gate)?)?;
        slot.set(Binding::of(&p.up)?)?;
        // An instance with no shared slot binds the stacks in their places,
        // which the kernel does not read.
        slot.set(Binding::of(p.shared_gate.as_ref().unwrap_or(&p.gate))?)?;
        slot.set(Binding::of(p.shared_up.as_ref().unwrap_or(&p.up))?)?;
        slot.set(Binding::of(&p.x)?)?;
        slot.set(Binding::of(p.norm.window())?)?;
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
        let (shape, shared_quant, shared) = check_fold(
            &self.geom.experts,
            &p,
            self.simdgroups,
            self.rows_per_simd_group,
        )?;
        let args = args_bytes([shape.rows, shape.n_experts, p.layer, 0]);
        encoder.set_pipeline(self.experts.fold(p.quant, shared_quant)?);
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
