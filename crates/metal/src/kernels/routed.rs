//! The launches of the prompt pass's routing and grouped experts: the router's
//! logits over rows, the selection of every row's slots, the route table, the
//! grouped GEMM over the table's tiles, the combine, the head-norm scale and a
//! dense FFN's SwiGLU over rows. The kernels are `kernels/route.metal` and
//! `kernels/gemm_routed.metal`.
//!
//! A pass of `rows` rows has `rows * ExpertsGeom::slots` slots: a row's picks
//! and, for an instance whose router has a shared row, the shared expert. The
//! route table sorts the slots by expert, with the shared expert the group
//! after the routed ones when there is one; the grouped GEMM walks the table's
//! tiles of at most [`ROUTE_TILE`] slots, each against its group's weights.

use bloomery_device::{DeviceError, Window, WindowMut};

use crate::device::MetalDevice;
use crate::encode::{Binding, Bindings, ComputeEncoder};
use crate::fault::MetalFaultWord;
use crate::library::{Library, Pipeline};

use super::gemm::{GEMM_ACTS, GEMM_ROWS, GEMM_THREADS, Operand, simdgroup_grid};
use super::matvec::Quant;
use super::router_geom;
use super::rows::check_strided;
use super::{
    Kernels, WeightAlign, args_bytes, check_weights_align, check_window_len, pack, range,
    weight_bytes, word,
};

/// The slots a tile of the grouped GEMM holds at most: the route table's cut
/// and the GEMM's activation tile (`ROUTE_TILE`, `ROUTED_ACTS`).
pub const ROUTE_TILE: usize = 64;

/// The weight rows of a grouped GEMM's tile (`ROUTED_ROWS`): the plain tile's
/// rows, and the pair's two tensors of half as many outputs.
pub const ROUTED_ROWS: usize = 64;

/// The logits row stride a pass keeps: the router's rows rounded up to a
/// multiple of four floats (the GEMM's output rows are four-aligned). A shim:
/// the build's host instance's (`RouterGeom::logit_stride`). Callers left:
/// the arch and the kernel gate.
pub const LOGIT_STRIDE: usize = router_geom::logit_stride(router_geom::ROWS);

/// The threads of the route table's one threadgroup (`ROUTE_THREADS`).
pub const TABLE_THREADS: usize = 1024;

/// The threads of the combine's threadgroup a row.
const COMBINE_THREADS: usize = 256;

/// The threads of the SwiGLU's threadgroup: a run of columns of one row.
const SWIGLU_THREADS: usize = 256;

/// The table's entries for `slots` slots over `groups` groups: every group's
/// tiles are full but its last, so the count is at most the slots' tiles plus
/// one partial tile a group. The groups are the instance's routed experts and
/// its shared expert when it has one (`RouterGeom::rows`), so the bound counts
/// no group the instance lacks.
#[must_use]
pub fn route_tiles(slots: usize, groups: usize) -> usize {
    slots.div_ceil(ROUTE_TILE) + groups
}

/// The words of the table: a header entry of four words, then one entry of
/// four words (group, first place, slots, 0) a tile.
#[must_use]
pub fn route_table_words(slots: usize, groups: usize) -> usize {
    4 * (1 + route_tiles(slots, groups))
}

/// The grouped GEMM's pipelines of one weight type: the pair and the plain
/// tile, in each operand.
struct RoutedFamily {
    pair: [Pipeline; 2],
    plain: [Pipeline; 2],
}

impl RoutedFamily {
    fn new(library: &Library, stem: &str) -> Result<Self, DeviceError> {
        let one = |kind: &str, operand: &str| {
            library.pipeline(&format!("gemm_routed_{stem}_{kind}_{operand}"))
        };
        Ok(Self {
            pair: [one("pair", "f16")?, one("pair", "f32")?],
            plain: [one("plain", "f16")?, one("plain", "f32")?],
        })
    }
}

/// A (stacks, shared expert) type pair's tile, in each operand.
type ApartTile = ((Quant, Quant), [Pipeline; 2]);

/// The `kind` tile of each pair of `table`, in each operand.
fn apart_tiles(
    library: &Library,
    table: &[(Quant, Quant)],
    kind: &str,
) -> Result<Vec<ApartTile>, DeviceError> {
    table
        .iter()
        .map(|&(stack, shared)| {
            let one = |operand: &str| {
                library.pipeline(&format!(
                    "gemm_routed_{}_apart_{}_{kind}_{operand}",
                    stack.name(),
                    shared.name()
                ))
            };
            Ok(((stack, shared), [one("f16")?, one("f32")?]))
        })
        .collect()
}

/// The pipelines of the pass's routing and grouped experts.
pub(crate) struct RoutedPipelines {
    q4k: RoutedFamily,
    q6k: RoutedFamily,
    a4: RoutedFamily,
    a8: RoutedFamily,
    q8_0: RoutedFamily,
    q5k: RoutedFamily,
    /// The pair tile of a shared expert of another type than the stacks, a
    /// pair of [`super::APART_GATE_UP`], and the plain tile of a pair of
    /// [`super::APART_DOWN`]: the simdgroup program's, whichever program the
    /// other families come from.
    apart_pair: Vec<ApartTile>,
    apart_plain: Vec<ApartTile>,
    f32w: Pipeline,
    select: Pipeline,
    table: Pipeline,
    combine: Pipeline,
    headnorm: Pipeline,
    swiglu: Pipeline,
}

impl RoutedPipelines {
    /// The pipelines of `library`, the grouped GEMM's from its GEMMs' library
    /// ([`Library::gemm_library`]).
    pub(crate) fn new(library: &Library) -> Result<Self, DeviceError> {
        let gemm = library.gemm_library();
        Ok(Self {
            q4k: RoutedFamily::new(gemm, "q4k")?,
            q6k: RoutedFamily::new(gemm, "q6k")?,
            a4: RoutedFamily::new(gemm, "a4")?,
            a8: RoutedFamily::new(gemm, "a8")?,
            q8_0: RoutedFamily::new(gemm, "q8_0")?,
            q5k: RoutedFamily::new(gemm, "q5k")?,
            apart_pair: apart_tiles(library, &super::APART_GATE_UP, "pair")?,
            apart_plain: apart_tiles(library, &super::APART_DOWN, "plain")?,
            f32w: library.pipeline("gemm_f32w_store_f32")?,
            select: library.pipeline("route_rows_select")?,
            table: library.pipeline("route_table")?,
            combine: library.pipeline("combine_rows")?,
            headnorm: library.pipeline("headnorm_rows")?,
            swiglu: library.pipeline("swiglu_rows")?,
        })
    }

    /// The tiles for stacks of `quant` and shared tensors of `shared`, the
    /// pair tile or the plain, one a pipeline per operand; `None` for a type
    /// or a pair with no entry.
    fn tiles(&self, quant: Quant, shared: Quant, pair: bool) -> Option<&[Pipeline; 2]> {
        if shared != quant {
            let table = if pair {
                &self.apart_pair
            } else {
                &self.apart_plain
            };
            return table
                .iter()
                .find(|(types, _)| *types == (quant, shared))
                .map(|(_, tiles)| tiles);
        }
        let family = match quant {
            Quant::Q4K => &self.q4k,
            Quant::Q6K => &self.q6k,
            Quant::A4 => &self.a4,
            Quant::A8 => &self.a8,
            Quant::Q8_0 => &self.q8_0,
            Quant::Q5K => &self.q5k,
            Quant::F32 => return None,
        };
        Some(if pair { &family.pair } else { &family.plain })
    }
}

const fn operand_at(operand: Operand) -> usize {
    match operand {
        Operand::F16 => 0,
        Operand::F32 => 1,
    }
}

const GEMM_WHAT: &str = "f32 gemm";

/// The router's dense GEMM over f32 weight rows: `y = W . x`, both operands
/// f32. `rows` weight rows over `cols` columns, `acts` activation rows.
pub struct F32GemmParams<'w> {
    /// The weight rows N: the output's columns.
    pub rows: usize,
    /// The input columns K, a multiple of 16.
    pub cols: usize,
    /// The activation rows R: the output's rows.
    pub acts: usize,
    /// The weights, `rows * cols` values.
    pub weights: Window<'w, f32, MetalDevice>,
    /// The activations, `acts` rows of `cols` values.
    pub x: Window<'w, f32, MetalDevice>,
    /// The f32 elements between the activation rows, at least `cols`.
    pub x_stride: usize,
    /// The output, `acts` rows of `rows` values.
    pub y: WindowMut<'w, f32, MetalDevice>,
    /// The f32 elements between the output rows, at least `rows` and a
    /// multiple of four.
    pub y_stride: usize,
    /// The fault word a raise lands in.
    pub fault: &'w MetalFaultWord,
    /// The layer a raise is recorded in.
    pub layer: u32,
}

const SELECT_WHAT: &str = "route rows select";

/// The selection of every row's slots from its router logits.
pub struct RouteSelectParams<'w> {
    /// Rows in the pass. Zero is refused.
    pub rows: usize,
    /// Routed experts a row keeps: 1 to the router's most.
    pub top_k: usize,
    /// The logits, the router's rows (`RouterGeom::rows`) values a row,
    /// `logit_stride` apart.
    pub logits: Window<'w, f32, MetalDevice>,
    /// Elements between rows of `logits`.
    pub logit_stride: usize,
    /// The ids, `top_k + RouterGeom::shared` a row: the routed picks, then the
    /// shared id when the router has a shared row.
    pub ids: WindowMut<'w, u32, MetalDevice>,
    /// The weights, as many a row as the ids: the softmax of the picks, then
    /// the shared expert's sigmoid gate when the router has a shared row.
    pub weights: WindowMut<'w, f32, MetalDevice>,
    /// The fault word a raise lands in.
    pub fault: &'w MetalFaultWord,
    /// The layer a raise is recorded in.
    pub layer: u32,
}

const TABLE_WHAT: &str = "route table";

/// The route table of a pass: the slots sorted by expert.
pub struct RouteTableParams<'w> {
    /// Slots in the pass, `rows * per_row`.
    pub slots: usize,
    /// Slots a row holds: the instance's (`ExpertsGeom::slots`), the picks and
    /// the shared expert when it has one.
    pub per_row: usize,
    /// The routed stack's experts; the shared expert, when the instance has
    /// one, is the group after them.
    pub experts: usize,
    /// Every slot's group (the select's ids).
    pub ids: Window<'w, u32, MetalDevice>,
    /// The slot at each place of the sorted list.
    pub sorted: WindowMut<'w, u32, MetalDevice>,
    /// The place of each slot in the sorted list.
    pub inv: WindowMut<'w, u32, MetalDevice>,
    /// The tiles: [`route_table_words`] words.
    pub tiles: WindowMut<'w, u32, MetalDevice>,
    /// The fault word a raise lands in.
    pub fault: &'w MetalFaultWord,
    /// The layer a raise is recorded in.
    pub layer: u32,
}

/// What a grouped GEMM computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoutedKind {
    /// The SwiGLU pair of the gate and the up, scaled by the slot's route
    /// weight: `y[p][o] = silu(gate . x) * (up . x) * w`. A tile is 32
    /// outputs.
    Pair,
    /// The down: `y[p][n] = down . x[p]`. A tile is 64 outputs.
    Plain,
}

impl RoutedKind {
    /// The outputs a tile covers.
    const fn tile(self) -> usize {
        match self {
            Self::Pair => ROUTED_ROWS / 2,
            Self::Plain => ROUTED_ROWS,
        }
    }
}

const ROUTED_WHAT: &str = "routed gemm";

/// One grouped GEMM over the route table's tiles.
pub struct RoutedParams<'w> {
    /// The weight type of the stacks.
    pub quant: Quant,
    /// The weight type of the shared tensors: `quant`, or the second of a
    /// pair of the kind's table ([`super::APART_GATE_UP`] for the pair tile,
    /// [`super::APART_DOWN`] for the plain). An instance with no shared group
    /// takes `quant`.
    pub shared_quant: Quant,
    /// The pair or the plain tile.
    pub kind: RoutedKind,
    /// The operand type the GEMM computes in.
    pub operand: Operand,
    /// The input columns K, a multiple of the type's block values.
    pub cols: usize,
    /// The outputs of each expert: the pair's rows of the gate (a multiple of
    /// 32), the down's rows (a multiple of 64).
    pub rows: usize,
    /// The routed stack's experts.
    pub experts: usize,
    /// The stack: the gate's, or the down's.
    pub stack: Window<'w, u8, MetalDevice>,
    /// The up's stack, for the pair.
    pub stack_up: Option<Window<'w, u8, MetalDevice>>,
    /// The shared expert's tensor, the gate's or the down's: given exactly
    /// when the instance's router has a shared group.
    pub shared: Option<Window<'w, u8, MetalDevice>>,
    /// The shared expert's up, for the pair of an instance that has one.
    pub shared_up: Option<Window<'w, u8, MetalDevice>>,
    /// The activations: the pass's rows (`gather`) or the places' own.
    pub x: Window<'w, f32, MetalDevice>,
    /// Rows of `x`.
    pub x_rows: usize,
    /// The f32 elements between the activation rows, at least `cols`.
    pub x_stride: usize,
    /// Whether an activation row is the slot's pass row, read through the
    /// sorted list, or the place itself.
    pub gather: bool,
    /// The output, one row a place.
    pub y: WindowMut<'w, f32, MetalDevice>,
    /// The f32 elements between the output rows, at least `rows`.
    pub y_stride: usize,
    /// The slots in the pass: the output's rows.
    pub slots: usize,
    /// The table's entries the grid covers: [`route_tiles`].
    pub max_tiles: usize,
    /// The sorted slot list.
    pub sorted: Window<'w, u32, MetalDevice>,
    /// The route weights, one a slot (the pair's epilogue scales by them).
    pub route_w: Window<'w, f32, MetalDevice>,
    /// The table.
    pub tiles: Window<'w, u32, MetalDevice>,
    /// The fault word a raise lands in.
    pub fault: &'w MetalFaultWord,
    /// The layer a raise is recorded in.
    pub layer: u32,
}

const COMBINE_WHAT: &str = "combine rows";

/// `x[row] += y[place of slot 0] + ... + y[place of slot per_row - 1]`.
pub struct CombineParams<'w> {
    /// Rows in the pass.
    pub rows: usize,
    /// Slots a row holds: the instance's (`ExpertsGeom::slots`).
    pub per_row: usize,
    /// The columns of a row, a multiple of four.
    pub cols: usize,
    /// The slots' outputs, one row a place.
    pub y: Window<'w, f32, MetalDevice>,
    /// Rows of `y`: the pass's slots.
    pub y_rows: usize,
    /// Elements between rows of `y`, a multiple of four.
    pub y_stride: usize,
    /// The place of each slot.
    pub inv: Window<'w, u32, MetalDevice>,
    /// The residual, added into.
    pub x: WindowMut<'w, f32, MetalDevice>,
    /// Elements between rows of `x`, a multiple of four.
    pub x_stride: usize,
}

const SWIGLU_WHAT: &str = "swiglu rows";

/// A dense FFN's two projections combined,
/// `gate[row][c] = silu(gate[row][c]) * up[row][c]`, in place over the gate's
/// rows.
pub struct SwigluRowsParams<'w> {
    /// Rows in the pass.
    pub rows: usize,
    /// The columns of a row: the FFN's width.
    pub cols: usize,
    /// The gate projection's rows, which the activation replaces.
    pub gate: WindowMut<'w, f32, MetalDevice>,
    /// The up projection's rows, the gate's shape.
    pub up: Window<'w, f32, MetalDevice>,
    /// The f32 elements between the rows of both.
    pub stride: usize,
}

const HEADNORM_WHAT: &str = "headnorm rows";

/// The gated delta numerator times each head's norm scale, in place.
pub struct HeadNormRowsParams<'w> {
    /// Rows in the pass.
    pub rows: usize,
    /// The numerator, the delta net's value heads of its head columns a row
    /// (`DeltaGeom::value_heads`, `DeltaGeom::head_cols`).
    pub g: WindowMut<'w, f32, MetalDevice>,
    /// Elements between rows of `g`.
    pub g_stride: usize,
    /// The heads' partial sums of squares, the value heads' rows of the
    /// column blocks a row (`DeltaGeom::partials_stride`).
    pub partials: Window<'w, f32, MetalDevice>,
    /// Elements between rows of `partials`.
    pub partials_stride: usize,
    /// The head norm's epsilon.
    pub eps: f32,
    /// The fault word a raise lands in.
    pub fault: &'w MetalFaultWord,
    /// The layer a raise is recorded in.
    pub layer: u32,
}

fn four_aligned(what: &'static str, name: &str, start: usize) -> Result<(), DeviceError> {
    check_weights_align(what, name, start, WeightAlign::Float4)
}

impl Kernels {
    /// Encode the router's GEMM over f32 weight rows: one dispatch, the
    /// dense GEMM's tile with both operands f32. A shape the kernel does not
    /// take or a window that is not that shape is refused by name and
    /// nothing is encoded.
    #[must_use = "a refused launch encoded nothing"]
    pub fn gemm_f32(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: F32GemmParams<'_>,
    ) -> Result<(), DeviceError> {
        let refuse = |detail: String| range(GEMM_WHAT, detail);
        if p.rows == 0 || p.cols == 0 || p.acts == 0 {
            return Err(refuse(format!(
                "{} weight rows by {} columns by {} activation rows is empty",
                p.rows, p.cols, p.acts
            )));
        }
        if !p.cols.is_multiple_of(16) {
            return Err(refuse(format!(
                "{} columns are not whole 16-value blocks",
                p.cols
            )));
        }
        if !p.y_stride.is_multiple_of(4) || p.y_stride < p.rows {
            return Err(refuse(format!(
                "the y stride {} is not a multiple of four at least the {} weight rows",
                p.y_stride, p.rows
            )));
        }
        check_window_len(GEMM_WHAT, "weights", p.weights.len(), p.rows * p.cols)?;
        check_strided(GEMM_WHAT, "x", p.x.len(), p.acts, p.x_stride, p.cols)?;
        check_strided(GEMM_WHAT, "y", p.y.len(), p.acts, p.y_stride, p.rows)?;
        four_aligned(GEMM_WHAT, "y", p.y.byte_range()?.start.get())?;
        let args = args_bytes([
            word(GEMM_WHAT, "the column count", p.cols)?,
            word(GEMM_WHAT, "the weight row count", p.rows)?,
            word(GEMM_WHAT, "the activation row count", p.acts)?,
            word(GEMM_WHAT, "the x stride", p.x_stride)?,
            word(GEMM_WHAT, "the y stride", p.y_stride)?,
            p.layer,
        ]);
        encoder.set_pipeline(&self.routed.f32w);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.weights)?)?;
        slot.set(Binding::of(&p.x)?)?;
        slot.set(Binding::of_mut(&p.y)?)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.set(p.fault.binding())?;
        // The entry is the dense GEMM's simdgroup tile on either path: its
        // grid is that tile's.
        let (x, y) = simdgroup_grid(p.rows.div_ceil(GEMM_ROWS), p.acts.div_ceil(GEMM_ACTS));
        slot.dispatch_groups(x, y, GEMM_THREADS)
    }

    /// Encode the selection of every row's slots from its logits: one
    /// threadgroup a row. A shape the kernel does not take or a window that
    /// is not that shape is refused by name and nothing is encoded.
    #[must_use = "a refused launch encoded nothing"]
    pub fn route_rows_select(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: RouteSelectParams<'_>,
    ) -> Result<(), DeviceError> {
        let refuse = |detail: String| range(SELECT_WHAT, detail);
        if p.rows == 0 {
            return Err(refuse("rows of 0".to_string()));
        }
        let router = &self.geom.router;
        if p.top_k == 0 || p.top_k > router.top_k_max {
            return Err(refuse(format!(
                "top_k {}; the selection takes 1 to {}",
                p.top_k, router.top_k_max
            )));
        }
        check_strided(
            SELECT_WHAT,
            "logits",
            p.logits.len(),
            p.rows,
            p.logit_stride,
            router.rows,
        )?;
        let per_row = p.top_k + router.shared;
        let slots = p
            .rows
            .checked_mul(per_row)
            .ok_or_else(|| refuse(format!("{} rows of {per_row} slots wrap", p.rows)))?;
        check_window_len(SELECT_WHAT, "ids", p.ids.len(), slots)?;
        check_window_len(SELECT_WHAT, "weights", p.weights.len(), slots)?;
        let args = args_bytes([
            word(SELECT_WHAT, "top_k", p.top_k)?,
            p.layer,
            word(SELECT_WHAT, "the logit stride", p.logit_stride)?,
        ]);
        encoder.set_pipeline(&self.routed.select);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.logits)?)?;
        slot.set(Binding::of_mut(&p.ids)?)?;
        slot.set(Binding::of_mut(&p.weights)?)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.set(p.fault.binding())?;
        slot.dispatch_groups(p.rows, 1, self.router_threads())
    }

    /// Encode the route table of a pass: one threadgroup counts, scans and
    /// places every slot. A shape the kernel does not take or a window that
    /// is not that shape is refused by name and nothing is encoded.
    #[must_use = "a refused launch encoded nothing"]
    pub fn route_table(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: RouteTableParams<'_>,
    ) -> Result<(), DeviceError> {
        let refuse = |detail: String| range(TABLE_WHAT, detail);
        if p.slots == 0 || p.per_row == 0 || !p.slots.is_multiple_of(p.per_row) {
            return Err(refuse(format!(
                "{} slots of rows of {}: the slots are not whole rows",
                p.slots, p.per_row
            )));
        }
        let router = &self.geom.router;
        if p.per_row != self.geom.experts.slots {
            return Err(refuse(format!(
                "{} slots a row; the instance's rows hold {}",
                p.per_row, self.geom.experts.slots
            )));
        }
        if p.experts + router.shared != router.rows {
            return Err(refuse(format!(
                "{} routed experts and {} shared groups; the table's groups are the router's {} rows",
                p.experts, router.shared, router.rows
            )));
        }
        check_window_len(TABLE_WHAT, "ids", p.ids.len(), p.slots)?;
        check_window_len(TABLE_WHAT, "sorted", p.sorted.len(), p.slots)?;
        check_window_len(TABLE_WHAT, "inv", p.inv.len(), p.slots)?;
        let max_tiles = route_tiles(p.slots, router.rows);
        check_window_len(
            TABLE_WHAT,
            "tiles",
            p.tiles.len(),
            route_table_words(p.slots, router.rows),
        )?;
        four_aligned(TABLE_WHAT, "tiles", p.tiles.byte_range()?.start.get())?;
        let args = args_bytes([
            word(TABLE_WHAT, "the slot count", p.slots)?,
            word(TABLE_WHAT, "the slots a row", p.per_row)?,
            word(TABLE_WHAT, "the expert count", p.experts)?,
            word(TABLE_WHAT, "the table's capacity", max_tiles)?,
            p.layer,
        ]);
        encoder.set_pipeline(&self.routed.table);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.ids)?)?;
        slot.set(Binding::of_mut(&p.sorted)?)?;
        slot.set(Binding::of_mut(&p.inv)?)?;
        slot.set(Binding::of_mut(&p.tiles)?)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.set(p.fault.binding())?;
        slot.dispatch_groups(1, 1, TABLE_THREADS)
    }

    /// Encode a grouped GEMM over the route table's tiles: the grid is the
    /// table's capacity by the tiles of one expert's outputs, and a
    /// threadgroup past the table's tile count returns. A shape the kernel
    /// does not take or a window that is not that shape is refused by name
    /// and nothing is encoded.
    #[must_use = "a refused launch encoded nothing"]
    pub fn gemm_routed(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: RoutedParams<'_>,
    ) -> Result<(), DeviceError> {
        let refuse = |detail: String| range(ROUTED_WHAT, detail);
        let vpb = p.quant.values_per_block();
        if p.cols == 0 || !p.cols.is_multiple_of(vpb) {
            return Err(refuse(format!(
                "{} columns are not whole blocks of {vpb}",
                p.cols
            )));
        }
        let tile = p.kind.tile();
        if p.rows == 0 || !p.rows.is_multiple_of(tile) {
            return Err(refuse(format!(
                "{} outputs a expert are not whole tiles of {tile}",
                p.rows
            )));
        }
        if p.slots == 0 || p.x_rows == 0 || p.max_tiles == 0 {
            return Err(refuse(format!(
                "{} slots over {} activation rows and {} tiles is empty",
                p.slots, p.x_rows, p.max_tiles
            )));
        }
        let pair = p.kind == RoutedKind::Pair;
        let tiles = self
            .routed
            .tiles(p.quant, p.shared_quant, pair)
            .ok_or_else(|| {
                refuse(format!(
                    "stacks of {} and a shared expert of {}: no entry",
                    p.quant.name(),
                    p.shared_quant.name()
                ))
            })?;
        if !p.cols.is_multiple_of(p.shared_quant.values_per_block()) {
            return Err(refuse(format!(
                "{} columns are not whole blocks of {}",
                p.cols,
                p.shared_quant.values_per_block()
            )));
        }
        let nblocks = p.cols / vpb;
        let one = weight_bytes(p.quant, p.rows, nblocks)
            .ok_or_else(|| refuse(format!("{} rows of {} columns wrap", p.rows, p.cols)))?;
        let shared_one = weight_bytes(
            p.shared_quant,
            p.rows,
            p.cols / p.shared_quant.values_per_block(),
        )
        .ok_or_else(|| refuse(format!("{} rows of {} columns wrap", p.rows, p.cols)))?;
        let stack_bytes = one
            .checked_mul(p.experts)
            .ok_or_else(|| refuse(format!("{} experts of {one} bytes wrap", p.experts)))?;
        let has_shared = self.geom.experts.has_shared;
        if pair != p.stack_up.is_some() || p.shared_up.is_some() != (pair && p.shared.is_some()) {
            return Err(refuse(
                "the pair takes the up's stack, and the shared up when it has a shared expert; the plain tile takes neither"
                    .to_string(),
            ));
        }
        match (p.shared.is_some(), has_shared) {
            (true, true) | (false, false) => {}
            (false, true) => {
                return Err(refuse(
                    "no shared expert, and the table has a shared group".to_string(),
                ));
            }
            (true, false) => {
                return Err(refuse(
                    "a shared expert, and the table has no shared group".to_string(),
                ));
            }
        }
        if p.shared.is_none() && p.shared_quant != p.quant {
            return Err(refuse(format!(
                "a shared type of {}, and no shared expert",
                p.shared_quant.name()
            )));
        }
        check_window_len(ROUTED_WHAT, "stack", p.stack.len(), stack_bytes)?;
        let mut windows = vec![("stack", p.stack)];
        if let Some(shared) = p.shared {
            check_window_len(ROUTED_WHAT, "shared", shared.len(), shared_one)?;
            windows.push(("shared", shared));
        }
        if let Some(up) = p.stack_up {
            check_window_len(ROUTED_WHAT, "up stack", up.len(), stack_bytes)?;
            windows.push(("up stack", up));
        }
        if let Some(shared_up) = p.shared_up {
            check_window_len(ROUTED_WHAT, "shared up", shared_up.len(), shared_one)?;
            windows.push(("shared up", shared_up));
        }
        for (name, w) in &windows {
            check_weights_align(
                ROUTED_WHAT,
                name,
                w.byte_range()?.start.get(),
                WeightAlign::Half,
            )?;
        }
        check_strided(ROUTED_WHAT, "x", p.x.len(), p.x_rows, p.x_stride, p.cols)?;
        check_strided(ROUTED_WHAT, "y", p.y.len(), p.slots, p.y_stride, p.rows)?;
        check_window_len(ROUTED_WHAT, "sorted", p.sorted.len(), p.slots)?;
        check_window_len(ROUTED_WHAT, "route weights", p.route_w.len(), p.slots)?;
        check_window_len(ROUTED_WHAT, "tiles", p.tiles.len(), 4 * (1 + p.max_tiles))?;
        let args = args_bytes([
            word(ROUTED_WHAT, "the column count", p.cols)?,
            word(ROUTED_WHAT, "the x stride", p.x_stride)?,
            word(ROUTED_WHAT, "the y stride", p.y_stride)?,
            p.layer,
            word(ROUTED_WHAT, "the expert count", p.experts)?,
            word(ROUTED_WHAT, "the stack blocks an expert", p.rows * nblocks)?,
            u32::from(p.gather),
            word(ROUTED_WHAT, "the outputs an expert", p.rows)?,
        ]);
        encoder.set_pipeline(&tiles[operand_at(p.operand)]);
        let mut slot = Bindings::new(encoder);
        // An instance with no shared group binds the stack in the shared
        // tensors' places, which the kernel does not read.
        let shared = p.shared.as_ref().unwrap_or(&p.stack);
        slot.set(Binding::of(&p.stack)?)?;
        slot.set(Binding::of(p.stack_up.as_ref().unwrap_or(&p.stack))?)?;
        slot.set(Binding::of(shared)?)?;
        slot.set(Binding::of(p.shared_up.as_ref().unwrap_or(shared))?)?;
        slot.set(Binding::of(&p.x)?)?;
        slot.set(Binding::of_mut(&p.y)?)?;
        slot.set(Binding::of(&p.tiles)?)?;
        slot.set(Binding::of(&p.sorted)?)?;
        slot.set(Binding::of(&p.route_w)?)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.set(p.fault.binding())?;
        slot.dispatch_groups(p.max_tiles, p.rows / tile, GEMM_THREADS)
    }

    /// Encode the combine of a pass: one threadgroup a row. A shape the
    /// kernel does not take or a window that is not that shape is refused by
    /// name and nothing is encoded.
    #[must_use = "a refused launch encoded nothing"]
    pub fn combine_rows(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: CombineParams<'_>,
    ) -> Result<(), DeviceError> {
        let refuse = |detail: String| range(COMBINE_WHAT, detail);
        if p.rows == 0 || p.per_row == 0 || p.cols == 0 {
            return Err(refuse(format!(
                "{} rows of {} slots over {} columns is empty",
                p.rows, p.per_row, p.cols
            )));
        }
        for (name, n) in [
            ("columns", p.cols),
            ("y stride", p.y_stride),
            ("x stride", p.x_stride),
        ] {
            if !n.is_multiple_of(4) {
                return Err(refuse(format!(
                    "the {name} {n} is not a multiple of four floats"
                )));
            }
        }
        let slots = p.rows * p.per_row;
        if p.y_rows != slots {
            return Err(refuse(format!(
                "{} output rows for {} rows of {} slots",
                p.y_rows, p.rows, p.per_row
            )));
        }
        check_strided(COMBINE_WHAT, "y", p.y.len(), p.y_rows, p.y_stride, p.cols)?;
        check_strided(COMBINE_WHAT, "x", p.x.len(), p.rows, p.x_stride, p.cols)?;
        check_window_len(COMBINE_WHAT, "inv", p.inv.len(), slots)?;
        four_aligned(COMBINE_WHAT, "y", p.y.byte_range()?.start.get())?;
        four_aligned(COMBINE_WHAT, "x", p.x.byte_range()?.start.get())?;
        let args = args_bytes([
            word(COMBINE_WHAT, "the column count", p.cols)?,
            word(COMBINE_WHAT, "the slots a row", p.per_row)?,
            word(COMBINE_WHAT, "the x stride", p.x_stride)?,
            word(COMBINE_WHAT, "the y stride", p.y_stride)?,
        ]);
        encoder.set_pipeline(&self.routed.combine);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.y)?)?;
        slot.set(Binding::of(&p.inv)?)?;
        slot.set(Binding::of_mut(&p.x)?)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.dispatch_groups(p.rows, 1, COMBINE_THREADS)
    }

    /// Encode the SwiGLU of a dense FFN's rows: a run of columns of one row a
    /// threadgroup. A shape the kernel does not take or a window that is not
    /// that shape is refused by name and nothing is encoded.
    #[must_use = "a refused launch encoded nothing"]
    pub fn swiglu_rows(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: SwigluRowsParams<'_>,
    ) -> Result<(), DeviceError> {
        if p.rows == 0 || p.cols == 0 {
            return Err(range(
                SWIGLU_WHAT,
                format!("{} rows of {} columns is empty", p.rows, p.cols),
            ));
        }
        check_strided(SWIGLU_WHAT, "gate", p.gate.len(), p.rows, p.stride, p.cols)?;
        check_strided(SWIGLU_WHAT, "up", p.up.len(), p.rows, p.stride, p.cols)?;
        let args = args_bytes([
            word(SWIGLU_WHAT, "the column count", p.cols)?,
            word(SWIGLU_WHAT, "the stride", p.stride)?,
        ]);
        encoder.set_pipeline(&self.routed.swiglu);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of_mut(&p.gate)?)?;
        slot.set(Binding::of(&p.up)?)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.dispatch_groups(p.cols.div_ceil(SWIGLU_THREADS), p.rows, SWIGLU_THREADS)
    }

    /// Encode the head-norm scale over rows: a threadgroup a (row, head). A
    /// shape the kernel does not take or a window that is not that shape is
    /// refused by name and nothing is encoded.
    #[must_use = "a refused launch encoded nothing"]
    pub fn headnorm_rows(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: HeadNormRowsParams<'_>,
    ) -> Result<(), DeviceError> {
        if p.rows == 0 {
            return Err(range(HEADNORM_WHAT, "rows of 0"));
        }
        let delta = &self.geom.delta;
        check_strided(
            HEADNORM_WHAT,
            "g",
            p.g.len(),
            p.rows,
            p.g_stride,
            delta.value_heads * delta.head_cols,
        )?;
        check_strided(
            HEADNORM_WHAT,
            "partials",
            p.partials.len(),
            p.rows,
            p.partials_stride,
            delta.partials_stride,
        )?;
        let args = [
            word(HEADNORM_WHAT, "the g stride", p.g_stride)?,
            word(HEADNORM_WHAT, "the partials stride", p.partials_stride)?,
            p.eps.to_bits(),
            p.layer,
        ]
        .map(u32::to_ne_bytes);
        encoder.set_pipeline(&self.routed.headnorm);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of_mut(&p.g)?)?;
        slot.set(Binding::of(&p.partials)?)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.set(p.fault.binding())?;
        slot.dispatch_groups(p.rows, delta.value_heads, delta.head_cols)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::Geometry;
    use crate::instance::REGISTERED;

    /// The tiles of a table whose group `g` holds `counts[g]` slots.
    fn tiles_of(counts: &[usize]) -> usize {
        counts.iter().map(|n| n.div_ceil(ROUTE_TILE)).sum()
    }

    #[test]
    fn the_table_bound_counts_the_groups_the_instance_has() {
        for inst in REGISTERED {
            let g = Geometry::of(inst);
            let groups = g.router.experts + usize::from(g.experts.has_shared);
            assert_eq!(
                g.router.rows, groups,
                "{}: the router's rows are the table's groups",
                inst.name
            );
            for slots in [g.experts.slots, 1000 * g.experts.slots] {
                assert_eq!(
                    route_tiles(slots, g.router.rows),
                    slots.div_ceil(ROUTE_TILE) + groups,
                    "{}: {slots} slots, the slots' tiles and a partial tile a group",
                    inst.name
                );
                assert_eq!(
                    route_table_words(slots, g.router.rows),
                    4 * (1 + route_tiles(slots, g.router.rows)),
                    "{}: {slots} slots, a header entry and an entry a tile",
                    inst.name
                );
            }
            // The tables with the most tiles: one slot in every group, and a
            // full tile and one more slot in every group.
            let ones = vec![1; groups];
            let over = vec![ROUTE_TILE + 1; groups];
            for counts in [ones, over] {
                let slots = counts.iter().sum();
                assert!(
                    tiles_of(&counts) <= route_tiles(slots, g.router.rows),
                    "{}: {slots} slots over {groups} groups fit the bound",
                    inst.name
                );
            }
        }
    }
}
