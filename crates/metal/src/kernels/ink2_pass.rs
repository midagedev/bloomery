//! The launches of Inkling 2's prompt pass: the step kernels' rules over a
//! chunk of rows (`kernels/ink2_pass.metal`: the clip and append, the
//! rel-bias attention, the router, the route table of the picks, the slots'
//! stored sum) and the grouped
//! experts' scaled tiles over the pass's route table
//! (`kernels/gemm_routed.metal`'s `gemm_routed_ink2_*`). Every shape they
//! check is `inkling2_geom`'s.

use bloomery_device::{DeviceError, Window, WindowMut};

use crate::device::MetalDevice;
use crate::encode::{Binding, Bindings, ComputeEncoder};
use crate::fault::MetalFaultWord;
use crate::inkling2_geom::{D_REL, EXPERTS, HEAD, HIDDEN, KV_HEADS, QUERY_HEADS, TOP_K};
use crate::library::{Library, Pipeline};

use super::nvfp4::NvfpLayout;
use super::routed::{ROUTE_TILE, ROUTED_ROWS, TABLE_THREADS, route_table_words, route_tiles};
use super::{
    Kernels, WeightAlign, args_bytes, check_weights_align, check_window_len, pack, range, word,
};

/// The pipelines of the pass's kernels.
pub(crate) struct Pipelines {
    qk_rows: Pipeline,
    rel_prefill: Pipeline,
    router_rows: Pipeline,
    table: Pipeline,
    combine_store: Pipeline,
    /// The scaled grouped tiles: NVFP4 planes' pair and plain, MXFP8 rows' and
    /// MXFP8 quads'.
    nv4p: [Pipeline; 2],
    mx8r: [Pipeline; 2],
    mx8q: [Pipeline; 2],
}

impl Pipelines {
    pub(crate) fn new(library: &Library) -> Result<Self, DeviceError> {
        Ok(Self {
            qk_rows: library.pipeline("qk_clip_append_rows")?,
            rel_prefill: library.pipeline("rel_prefill_f16")?,
            router_rows: library.pipeline("router_sigmoid_rows")?,
            table: library.pipeline("ink2_route_table")?,
            combine_store: library.pipeline("combine_rows_store")?,
            nv4p: [
                library.pipeline("gemm_routed_ink2_nv4p_pair_f16")?,
                library.pipeline("gemm_routed_ink2_nv4p_plain_f16")?,
            ],
            mx8r: [
                library.pipeline("gemm_routed_ink2_mx8r_pair_f16")?,
                library.pipeline("gemm_routed_ink2_mx8r_plain_f16")?,
            ],
            mx8q: [
                library.pipeline("gemm_routed_ink2_mx8q_pair_f16")?,
                library.pipeline("gemm_routed_ink2_mx8q_plain_f16")?,
            ],
        })
    }
}

/// The rows of the clip's launch: a SIMD group a row of the q, k, v and
/// relative heads, eight a threadgroup.
const CLIP_GROUPS: usize = (2 * (QUERY_HEADS + KV_HEADS)).div_ceil(8);

/// The rows of a prefill threadgroup: four SIMD groups of eight.
const PREFILL_ROWS: usize = 32;
const PREFILL_THREADS: usize = 128;

/// The router's and the combine's threadgroups.
const ROUTER_THREADS: usize = 256;
const COMBINE_THREADS: usize = 256;

/// The GEMM threadgroup's threads.
const GEMM_THREADS: usize = super::GEMM_THREADS;

const CLIP_WHAT: &str = "qk clip rows";

/// The clip and append over `rows` rows from position `pos`.
pub struct QkClipRowsParams<'w> {
    pub rows: usize,
    /// The q path's rows, `rows` of `qr_stride`: each row's `24 · 128` query
    /// values from its start and its `24 · 16` relative values from `r_at`.
    pub qr: WindowMut<'w, f32, MetalDevice>,
    pub qr_stride: usize,
    pub r_at: usize,
    /// The key and value rows, `rows` of `8 · 128` each.
    pub k: WindowMut<'w, f32, MetalDevice>,
    pub v: WindowMut<'w, f32, MetalDevice>,
    pub gq: Window<'w, f32, MetalDevice>,
    pub gr: Window<'w, f32, MetalDevice>,
    pub cache_k: WindowMut<'w, u16, MetalDevice>,
    pub cache_v: WindowMut<'w, u16, MetalDevice>,
    /// The planes' rows: a ring's own, a plane's the context.
    pub ctx: usize,
    pub pos: u32,
    /// Each row's position factor on a windowless visit; none windowed.
    pub taus: Option<Window<'w, f32, MetalDevice>>,
    pub v_eps: f32,
    pub qk_thresh: f32,
    pub r_thresh: f32,
    pub faults: &'w MetalFaultWord,
    pub layer: u32,
}

const REL_WHAT: &str = "rel prefill";

/// The rel-bias attention over `rows` rows from position `from`.
pub struct RelPrefillParams<'w> {
    pub rows: usize,
    pub from: u32,
    /// The planes' rows: a ring's own, a plane's the context.
    pub plane: usize,
    /// The context the chunk's last position is bounded by.
    pub ctx: usize,
    /// The last window of a row's keys attended; 0 for all.
    pub window: usize,
    pub scale: f32,
    /// The query rows and the relative rows, `q_stride` apart.
    pub q: Window<'w, f32, MetalDevice>,
    pub r: Window<'w, f32, MetalDevice>,
    pub q_stride: usize,
    /// The bias table, `16 · 1024`.
    pub bias: Window<'w, f32, MetalDevice>,
    /// The sinks, 24 a row, `sink_stride` apart.
    pub sinks: Window<'w, f32, MetalDevice>,
    pub sink_stride: usize,
    pub cache_k: Window<'w, u16, MetalDevice>,
    pub cache_v: Window<'w, u16, MetalDevice>,
    /// The output, `24 · 128` a row, `out_stride` apart.
    pub out: WindowMut<'w, f32, MetalDevice>,
    pub out_stride: usize,
    pub faults: &'w MetalFaultWord,
    pub layer: u32,
}

const ROUTER_WHAT: &str = "router sigmoid rows";

/// The biased sigmoid router over `rows` rows of the hidden width.
pub struct RouterRowsParams<'w> {
    pub rows: usize,
    pub top_k: usize,
    /// The router's rows, `128 · hidden` f32.
    pub weights: Window<'w, f32, MetalDevice>,
    /// The inputs, `rows · hidden`.
    pub x: Window<'w, f32, MetalDevice>,
    pub bias: Window<'w, f32, MetalDevice>,
    /// The scores, `rows · 128`.
    pub probs: WindowMut<'w, f32, MetalDevice>,
    /// The kept ids and weights, `rows · top_k` each.
    pub ids: WindowMut<'w, u32, MetalDevice>,
    pub route_w: WindowMut<'w, f32, MetalDevice>,
    pub scale: f32,
    pub faults: &'w MetalFaultWord,
    pub layer: u32,
}

const TABLE_WHAT: &str = "inkling2 route table";

/// The route table of a pass's picks: every slot a routed pick, no shared
/// group.
pub struct Ink2TableParams<'w> {
    /// Slots in the pass, `rows * TOP_K`.
    pub slots: usize,
    /// Every slot's expert (the router's or the host's ids).
    pub ids: Window<'w, u32, MetalDevice>,
    /// The slot at each place of the sorted list.
    pub sorted: WindowMut<'w, u32, MetalDevice>,
    /// The place of each slot in the sorted list.
    pub inv: WindowMut<'w, u32, MetalDevice>,
    /// The tiles: [`route_table_words`] words over the stack's experts.
    pub tiles: WindowMut<'w, u32, MetalDevice>,
    pub fault: &'w MetalFaultWord,
    pub layer: u32,
}

const COMBINE_WHAT: &str = "combine rows store";

/// `x[row] = y[place of slot 0] + ... + y[place of slot per_row - 1]`.
pub struct CombineStoreParams<'w> {
    pub rows: usize,
    pub per_row: usize,
    pub cols: usize,
    pub y: Window<'w, f32, MetalDevice>,
    pub y_stride: usize,
    pub inv: Window<'w, u32, MetalDevice>,
    pub x: WindowMut<'w, f32, MetalDevice>,
    pub x_stride: usize,
}

const ROUTED_WHAT: &str = "ink2 routed gemm";

/// One scaled grouped GEMM over the pass's route table.
pub struct Ink2RoutedParams<'w> {
    /// The stack's layout: NVFP4 planes or MXFP8 rows.
    pub layout: NvfpLayout,
    /// The SITU pair over the fused gate·up stack, or the plain down.
    pub pair: bool,
    /// The input columns K.
    pub cols: usize,
    /// The outputs an expert: the pair's gate rows (the up rows follow
    /// them in the stack), the down's rows.
    pub rows: usize,
    /// The stack: every expert's rows, expert `e`'s at `e` times its bytes.
    pub stack: Window<'w, u8, MetalDevice>,
    /// Each expert's global scale.
    pub scales: Window<'w, f32, MetalDevice>,
    /// The activations: the pass's rows (`gather`) or the places' own.
    pub x: Window<'w, f32, MetalDevice>,
    pub x_rows: usize,
    pub x_stride: usize,
    pub gather: bool,
    /// The output, one row a place.
    pub y: WindowMut<'w, f32, MetalDevice>,
    pub y_stride: usize,
    /// The pass's slots and the slots a pass row holds.
    pub slots: usize,
    pub per_row: usize,
    pub sorted: Window<'w, u32, MetalDevice>,
    /// Each slot's route weight (the plain tile's scale reads it).
    pub route_w: Window<'w, f32, MetalDevice>,
    pub tiles: Window<'w, u32, MetalDevice>,
    pub gate_beta: f32,
    pub value_beta: f32,
    pub faults: &'w MetalFaultWord,
    pub layer: u32,
}

impl Kernels {
    /// Encode the clip and append over rows: one dispatch, a threadgroup of
    /// eight heads' rows by the chunk's rows.
    #[must_use = "a refused launch encoded nothing"]
    pub fn qk_clip_rows(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: QkClipRowsParams<'_>,
    ) -> Result<(), DeviceError> {
        let refuse = |d: String| range(CLIP_WHAT, d);
        if p.rows == 0 || p.ctx == 0 {
            return Err(refuse(format!("{} rows over a plane of {}", p.rows, p.ctx)));
        }
        for (name, v) in [
            ("v_eps", p.v_eps),
            ("qk threshold", p.qk_thresh),
            ("r threshold", p.r_thresh),
        ] {
            if !(v.is_finite() && v > 0.0) {
                return Err(refuse(format!("{name} {v} is not finite and positive")));
            }
        }
        let qn = QUERY_HEADS * HEAD;
        let rn = QUERY_HEADS * D_REL;
        if p.r_at < qn || p.qr_stride < p.r_at + rn {
            return Err(refuse(format!(
                "relative values at {} of a q stride of {}: the query values take {qn}, the \
                 relative {rn}",
                p.r_at, p.qr_stride
            )));
        }
        let last = (p.rows - 1) * p.qr_stride;
        check_window_len(CLIP_WHAT, "qr", p.qr.len(), last + p.r_at + rn)?;
        check_window_len(CLIP_WHAT, "k", p.k.len(), p.rows * KV_HEADS * HEAD)?;
        check_window_len(CLIP_WHAT, "v", p.v.len(), p.rows * KV_HEADS * HEAD)?;
        check_window_len(CLIP_WHAT, "gq", p.gq.len(), qn)?;
        check_window_len(CLIP_WHAT, "gr", p.gr.len(), QUERY_HEADS * D_REL)?;
        let plane = KV_HEADS * p.ctx * HEAD;
        check_window_len(CLIP_WHAT, "key plane", p.cache_k.len(), plane)?;
        check_window_len(CLIP_WHAT, "value plane", p.cache_v.len(), plane)?;
        if let Some(t) = &p.taus {
            check_window_len(CLIP_WHAT, "taus", t.len(), p.rows)?;
        }
        let args = args_bytes([
            word(CLIP_WHAT, "ctx", p.ctx)?,
            p.pos,
            p.layer,
            p.v_eps.to_bits(),
            p.qk_thresh.to_bits(),
            p.r_thresh.to_bits(),
            u32::from(p.taus.is_some()),
            word(CLIP_WHAT, "the q stride", p.qr_stride)?,
            word(CLIP_WHAT, "the kv stride", KV_HEADS * HEAD)?,
            word(CLIP_WHAT, "rows", p.rows)?,
            word(CLIP_WHAT, "the relative values' start", p.r_at)?,
            0,
        ]);
        encoder.set_pipeline(&self.ink2_pass.qk_rows);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of_mut(&p.qr)?)?;
        slot.set(Binding::of_mut(&p.k)?)?;
        slot.set(Binding::of_mut(&p.v)?)?;
        slot.set(Binding::of(&p.gq)?)?;
        slot.set(Binding::of(&p.gr)?)?;
        slot.set(Binding::of_mut(&p.cache_k)?)?;
        slot.set(Binding::of_mut(&p.cache_v)?)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.set(p.faults.binding())?;
        // A windowed launch reads no factor: the gains stand in the slot.
        slot.set(match &p.taus {
            Some(t) => Binding::of(t)?,
            None => Binding::of(&p.gq)?,
        })?;
        slot.dispatch_groups(CLIP_GROUPS, p.rows, 256)
    }

    /// Encode the rel-bias attention over rows: one dispatch of
    /// `⌈3 · rows / 32⌉ × 8` threadgroups.
    #[must_use = "a refused launch encoded nothing"]
    pub fn rel_prefill(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: RelPrefillParams<'_>,
    ) -> Result<(), DeviceError> {
        let refuse = |d: String| range(REL_WHAT, d);
        if p.rows == 0 || p.plane == 0 {
            return Err(refuse(format!(
                "{} rows over a plane of {}",
                p.rows, p.plane
            )));
        }
        if !(p.scale.is_finite() && p.scale > 0.0) {
            return Err(refuse(format!("a scale of {}", p.scale)));
        }
        let last = (p.rows - 1) * p.q_stride;
        check_window_len(REL_WHAT, "q", p.q.len(), last + QUERY_HEADS * HEAD)?;
        check_window_len(REL_WHAT, "r", p.r.len(), last + QUERY_HEADS * D_REL)?;
        check_window_len(
            REL_WHAT,
            "bias",
            p.bias.len(),
            D_REL * crate::inkling2_geom::EXTENT,
        )?;
        check_window_len(
            REL_WHAT,
            "sinks",
            p.sinks.len(),
            (p.rows - 1) * p.sink_stride + QUERY_HEADS,
        )?;
        let plane = KV_HEADS * p.plane * HEAD;
        check_window_len(REL_WHAT, "key plane", p.cache_k.len(), plane)?;
        check_window_len(REL_WHAT, "value plane", p.cache_v.len(), plane)?;
        check_window_len(
            REL_WHAT,
            "out",
            p.out.len(),
            (p.rows - 1) * p.out_stride + QUERY_HEADS * HEAD,
        )?;
        let args = args_bytes([
            word(REL_WHAT, "rows", p.rows)?,
            p.from,
            word(REL_WHAT, "the plane", p.plane)?,
            word(REL_WHAT, "the window", p.window)?,
            p.layer,
            p.scale.to_bits(),
            word(REL_WHAT, "the q stride", p.q_stride)?,
            word(REL_WHAT, "the sink stride", p.sink_stride)?,
            word(REL_WHAT, "the out stride", p.out_stride)?,
            word(REL_WHAT, "ctx", p.ctx)?,
        ]);
        encoder.set_pipeline(&self.ink2_pass.rel_prefill);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.q)?)?;
        slot.set(Binding::of(&p.r)?)?;
        slot.set(Binding::of(&p.bias)?)?;
        slot.set(Binding::of(&p.sinks)?)?;
        slot.set(Binding::of(&p.cache_k)?)?;
        slot.set(Binding::of(&p.cache_v)?)?;
        slot.set(Binding::of_mut(&p.out)?)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.set(p.faults.binding())?;
        let groups = (p.rows * (QUERY_HEADS / KV_HEADS)).div_ceil(PREFILL_ROWS);
        slot.dispatch_groups(groups, KV_HEADS, PREFILL_THREADS)
    }

    /// Encode the router over rows: one threadgroup a row.
    #[must_use = "a refused launch encoded nothing"]
    pub fn router_sigmoid_rows(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: RouterRowsParams<'_>,
    ) -> Result<(), DeviceError> {
        let refuse = |d: String| range(ROUTER_WHAT, d);
        if p.rows == 0 || p.top_k == 0 || p.top_k > TOP_K {
            return Err(refuse(format!("{} rows keeping {}", p.rows, p.top_k)));
        }
        check_window_len(ROUTER_WHAT, "weights", p.weights.len(), EXPERTS * HIDDEN)?;
        check_window_len(ROUTER_WHAT, "x", p.x.len(), p.rows * HIDDEN)?;
        check_window_len(ROUTER_WHAT, "bias", p.bias.len(), EXPERTS)?;
        check_window_len(ROUTER_WHAT, "probs", p.probs.len(), p.rows * EXPERTS)?;
        check_window_len(ROUTER_WHAT, "ids", p.ids.len(), p.rows * p.top_k)?;
        check_window_len(
            ROUTER_WHAT,
            "weights out",
            p.route_w.len(),
            p.rows * p.top_k,
        )?;
        let args = args_bytes([
            word(ROUTER_WHAT, "top_k", p.top_k)?,
            p.layer,
            p.scale.to_bits(),
        ]);
        encoder.set_pipeline(&self.ink2_pass.router_rows);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.weights)?)?;
        slot.set(Binding::of(&p.x)?)?;
        slot.set(Binding::of(&p.bias)?)?;
        slot.set(Binding::of_mut(&p.probs)?)?;
        slot.set(Binding::of_mut(&p.ids)?)?;
        slot.set(Binding::of_mut(&p.route_w)?)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.set(p.faults.binding())?;
        slot.dispatch_groups(p.rows, 1, ROUTER_THREADS)
    }

    /// Encode the slots' stored sum: one threadgroup a row.
    #[must_use = "a refused launch encoded nothing"]
    pub fn combine_rows_store(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: CombineStoreParams<'_>,
    ) -> Result<(), DeviceError> {
        let refuse = |d: String| range(COMBINE_WHAT, d);
        if p.rows == 0 || p.per_row == 0 || p.cols == 0 {
            return Err(refuse(format!("{} rows of {} slots", p.rows, p.per_row)));
        }
        for (name, n) in [
            ("cols", p.cols),
            ("y stride", p.y_stride),
            ("x stride", p.x_stride),
        ] {
            if !n.is_multiple_of(4) {
                return Err(refuse(format!("the {name} {n} is not a multiple of four")));
            }
        }
        let slots = p.rows * p.per_row;
        check_window_len(
            COMBINE_WHAT,
            "y",
            p.y.len(),
            (slots - 1) * p.y_stride + p.cols,
        )?;
        check_window_len(
            COMBINE_WHAT,
            "x",
            p.x.len(),
            (p.rows - 1) * p.x_stride + p.cols,
        )?;
        check_window_len(COMBINE_WHAT, "inv", p.inv.len(), slots)?;
        let args = args_bytes([
            word(COMBINE_WHAT, "cols", p.cols)?,
            word(COMBINE_WHAT, "per_row", p.per_row)?,
            word(COMBINE_WHAT, "the x stride", p.x_stride)?,
            word(COMBINE_WHAT, "the y stride", p.y_stride)?,
        ]);
        encoder.set_pipeline(&self.ink2_pass.combine_store);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.y)?)?;
        slot.set(Binding::of(&p.inv)?)?;
        slot.set(Binding::of_mut(&p.x)?)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.dispatch_groups(p.rows, 1, COMBINE_THREADS)
    }

    /// Encode the route table of a pass's picks: one threadgroup counts,
    /// scans and places every slot over the stack's experts, in the shared
    /// table's format (`route_table`'s), which the scaled tiles walk.
    #[must_use = "a refused launch encoded nothing"]
    pub fn ink2_route_table(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: Ink2TableParams<'_>,
    ) -> Result<(), DeviceError> {
        let refuse = |d: String| range(TABLE_WHAT, d);
        if p.slots == 0 || !p.slots.is_multiple_of(TOP_K) {
            return Err(refuse(format!(
                "{} slots are not whole rows of {TOP_K} picks",
                p.slots
            )));
        }
        check_window_len(TABLE_WHAT, "ids", p.ids.len(), p.slots)?;
        check_window_len(TABLE_WHAT, "sorted", p.sorted.len(), p.slots)?;
        check_window_len(TABLE_WHAT, "inv", p.inv.len(), p.slots)?;
        check_window_len(
            TABLE_WHAT,
            "tiles",
            p.tiles.len(),
            route_table_words(p.slots, EXPERTS),
        )?;
        if !p.tiles.byte_range()?.start.get().is_multiple_of(16) {
            return Err(refuse(
                "the tiles window does not start on a table entry".into(),
            ));
        }
        let args = args_bytes([
            word(TABLE_WHAT, "the slot count", p.slots)?,
            word(TABLE_WHAT, "the slots a row", TOP_K)?,
            word(TABLE_WHAT, "the expert count", EXPERTS)?,
            word(
                TABLE_WHAT,
                "the table's capacity",
                route_tiles(p.slots, EXPERTS),
            )?,
            p.layer,
        ]);
        encoder.set_pipeline(&self.ink2_pass.table);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.ids)?)?;
        slot.set(Binding::of_mut(&p.sorted)?)?;
        slot.set(Binding::of_mut(&p.inv)?)?;
        slot.set(Binding::of_mut(&p.tiles)?)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.set(p.fault.binding())?;
        slot.dispatch_groups(1, 1, TABLE_THREADS)
    }

    /// Encode a scaled grouped GEMM over the route table's tiles.
    #[must_use = "a refused launch encoded nothing"]
    pub fn gemm_routed_ink2(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: Ink2RoutedParams<'_>,
    ) -> Result<(), DeviceError> {
        let refuse = |d: String| range(ROUTED_WHAT, d);
        let (family, vpb, block) = match p.layout {
            NvfpLayout::Planes => (&self.ink2_pass.nv4p, 16, 9),
            NvfpLayout::Mxfp8Rows => (&self.ink2_pass.mx8r, 32, 33),
            NvfpLayout::Mxfp8Quads => (&self.ink2_pass.mx8q, 128, 132),
            other => return Err(refuse(format!("a stack of {other:?}: no grouped tile"))),
        };
        if p.cols == 0 || !p.cols.is_multiple_of(32) || !p.cols.is_multiple_of(vpb) {
            return Err(refuse(format!("{} columns are not whole K steps", p.cols)));
        }
        let tile = if p.pair { ROUTED_ROWS / 2 } else { ROUTED_ROWS };
        if p.rows == 0 || !p.rows.is_multiple_of(tile) {
            return Err(refuse(format!(
                "{} outputs are not whole tiles of {tile}",
                p.rows
            )));
        }
        if p.slots == 0 || p.per_row == 0 || !p.slots.is_multiple_of(p.per_row) {
            return Err(refuse(format!(
                "{} slots of rows of {}",
                p.slots, p.per_row
            )));
        }
        let nb = p.cols / vpb;
        let stack_rows = if p.pair { 2 * p.rows } else { p.rows };
        let experts = p.scales.len().get();
        if experts != EXPERTS {
            return Err(refuse(format!(
                "{experts} scales; the stack holds {EXPERTS}"
            )));
        }
        check_window_len(
            ROUTED_WHAT,
            "stack",
            p.stack.len(),
            experts * stack_rows * nb * block,
        )?;
        if p.layout == NvfpLayout::Planes {
            // A tile reads a group's codes as one 8-byte word.
            let start = p.stack.byte_range()?.start.get();
            if !start.is_multiple_of(8) || !(stack_rows * nb * block).is_multiple_of(8) {
                return Err(refuse(format!(
                    "the stack starts {start} bytes into its buffer, an expert {} bytes: the \
                     planes' code words are 8-byte words",
                    stack_rows * nb * block
                )));
            }
        }
        if p.layout == NvfpLayout::Mxfp8Quads {
            // A tile reads a block's codes as 32-bit words; an expert and a
            // row are whole 132-byte blocks.
            check_weights_align(
                ROUTED_WHAT,
                "stack",
                p.stack.byte_range()?.start.get(),
                WeightAlign::Word,
            )?;
        }
        let x_rows = if p.gather { p.x_rows } else { p.slots };
        check_window_len(
            ROUTED_WHAT,
            "x",
            p.x.len(),
            (x_rows - 1) * p.x_stride + p.cols,
        )?;
        check_window_len(
            ROUTED_WHAT,
            "y",
            p.y.len(),
            (p.slots - 1) * p.y_stride + p.rows,
        )?;
        check_window_len(ROUTED_WHAT, "sorted", p.sorted.len(), p.slots)?;
        check_window_len(ROUTED_WHAT, "route weights", p.route_w.len(), p.slots)?;
        let max_tiles = route_tiles(p.slots, EXPERTS);
        check_window_len(ROUTED_WHAT, "tiles", p.tiles.len(), 4 * (1 + max_tiles))?;
        debug_assert_eq!(ROUTE_TILE, 64);
        let args = args_bytes([
            word(ROUTED_WHAT, "cols", p.cols)?,
            word(ROUTED_WHAT, "the x stride", p.x_stride)?,
            word(ROUTED_WHAT, "the y stride", p.y_stride)?,
            p.layer,
            word(ROUTED_WHAT, "experts", experts)?,
            word(ROUTED_WHAT, "the stack blocks an expert", stack_rows * nb)?,
            u32::from(p.gather),
            word(ROUTED_WHAT, "the outputs an expert", p.rows)?,
        ]);
        let iargs = args_bytes([
            p.gate_beta.to_bits(),
            p.value_beta.to_bits(),
            word(ROUTED_WHAT, "per_row", p.per_row)?,
            0,
        ]);
        encoder.set_pipeline(&family[usize::from(!p.pair)]);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.stack)?)?;
        slot.set(Binding::of(&p.x)?)?;
        slot.set(Binding::of_mut(&p.y)?)?;
        slot.set(Binding::of(&p.tiles)?)?;
        slot.set(Binding::of(&p.sorted)?)?;
        slot.set(Binding::of(&p.route_w)?)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.set(p.faults.binding())?;
        slot.set(Binding::of(&p.scales)?)?;
        slot.set(Binding::bytes(pack(&iargs))?)?;
        slot.dispatch_groups(max_tiles, p.rows / tile, GEMM_THREADS)
    }
}
