//! The quantized kernels' host side: the pipelines of the Q4_K, Q6_K, A4 and
//! A8 mat-vecs, the Q4_K, A4 and A8 embeddings, and the launch of each.
//!
//! A launch takes a parameter struct of windows and the shape they are for,
//! checks every window's length against the shape before it encodes anything
//! (a window that is not the shape's is a named [`bloomery_device::DeviceError`], never a
//! partial read or write), and encodes one dispatch on the caller's encoder.
//! The kernels themselves are
//! `kernels/{norm,matvec_q4k,matvec_q6k,matvec_a4,matvec_a8,embed}.metal`.
//! Decode attention is `kernels/attention.metal`.
//!
//! Three mat-vecs live here, one concept each: the plain mat-vec of one
//! tensor ([`matvec`]), the segmented mat-vec of up to four tensors of one
//! type and one width in one dispatch ([`seg`]), and the head-normed residual
//! mat-vec of a gated delta net's out projection ([`headnorm`]).
//! The dense GEMM of R activation rows ([`gemm`]) is one more family, plain
//! and segmented the same way: `kernels/gemm.metal`.
//! The router ([`router`]) is one more pipeline: `kernels/router.metal`, its ticket `kernels/ticket.metal`.
//! The routed gate-up pair and the folded down are one dispatch each.
//! The gated delta step ([`delta`]) writes the numerator and the partials that
//! mat-vec reads. Its kernel is `kernels/delta.metal`.
//! The head ([`head`]) is two more: the logits mat-vec and the argmax over them, `kernels/head.metal`.
//! The short conv ([`shortconv`]) of a gated-conv mixer is `kernels/shortconv.metal`; the
//! sandwich residual is `kernels/norm.metal`'s `norm_add_rows`, and the one-row join that
//! hands its next launch the normed input is each type's `<q>_norm_add_next`
//! ([`Kernels::norm_add_next`]), read by the scaled seg, pair and head entries ([`LoadNorm`]);
//! a plain residual hands the same input on with `<q>_norm_next` ([`Kernels::norm_next`]).

mod attention;
use crate::attention_geom;
mod delta;
use crate::delta_geom;
mod embed;
mod experts;
use crate::experts_geom;
mod gemm;
use crate::geometry;
mod head;
mod headnorm;
mod matvec;
mod norm;
mod routed;
pub mod router;
use crate::router_geom;
mod rows;
mod seg;
mod shortconv;
mod shortconv_geom;

pub use attention::AttentionPrefillParams;
pub use attention::AttentionRowsParams;
pub use attention::{
    AttentionParams, AttnCache, HEAD_DIM, KV_HEADS, ONE_PASS_KEYS, QG_STRIDE, QUERY_HEADS, ROT,
    SCALE, SEGMENT_CAP, SEGMENT_KEYS, TILE, f16_len, out_len, partials_len, q8_code_len,
    q8_scale_len, qg_len, segments_for_keys,
};
pub use attention::{AttnLayer, segments_for_window};
pub use delta::DeltaParams;
pub use delta::DeltaRowStrides;
pub use delta::DeltaRowsParams;
pub use delta::DeltaSnaps;
pub use delta::HeadMap;
pub use delta::Q_SCALE;
pub use delta::channels;
pub use delta::key_of;
pub use delta::prepared_elems;
pub use delta::ring_elems;
pub use delta::state_elems;
pub use delta::state_index;
pub use delta_geom::CONV_TAPS;
pub use delta_geom::PASS_ROWS;
pub use delta_geom::RING_ROWS;
pub use delta_geom::SNAPS;
pub use embed::EmbedParams;
pub use experts::{APART_DOWN, APART_GATE_UP, FoldParams, PairParams, SharedPlan, shared_entry};
pub use experts_geom::{EXPERT_K, EXPERT_SLOTS, ROUTED_IDS, SHARED_SLOT};
pub use gemm::{
    GEMM_ACTS, GEMM_ROWS, GEMM_THREADS, GemmMode, GemmParams, GemmSegParams, GemmX, GemmXParams,
    Operand,
};
pub use geometry::{
    BLOCK_VALUES, HEAD_COLS, MAX_SEGMENTS, PARTIALS_PER_HEAD, ROWS_PER_GROUP, ROWS_PER_SIMD_GROUP,
    SIMD_GROUPS_PER_GROUP, THREADS_PER_GROUP, VALUE_HEADS,
};
pub use head::{HEAD_K, HEAD_SIMD_GROUPS, HEAD_THREADS, HeadParams};
pub use headnorm::HeadNormParams;
pub use matvec::{LoadNorm, MatvecParams, Mode, NormOnLoad, Quant};
pub use norm::NormRowsParams;
pub use norm::{NormAddNextParams, NormAddRowsParams, NormNextParams};
pub use routed::{
    CombineParams, F32GemmParams, HeadNormRowsParams, LOGIT_STRIDE, ROUTE_TILE, ROUTED_ROWS,
    RouteSelectParams, RouteTableParams, RoutedKind, RoutedParams, SwigluRowsParams, TABLE_THREADS,
    route_table_words, route_tiles,
};
pub use seg::{SegParams, Segment};
pub use shortconv::{ShortConvRowsParams, ShortConvStepParams};
pub use shortconv_geom::MAX_SLOTS as SHORTCONV_MAX_SLOTS;

use bloomery_device::DeviceError;

use crate::device::MetalDevice;
pub use crate::geom::Geometry;
use crate::library::{GemmPath, Library, Pipeline};

/// The pipelines of one mat-vec type: the plain entry, the entry that norms its
/// input as it loads it, the one that adds into its output, the segmented
/// entry, its scaled form, and the two hand-ons that give a scaled entry its
/// normed input in this type's lane order, the sandwich join's and the plain
/// residual's.
struct MatvecPipelines {
    store: Pipeline,
    norm: Pipeline,
    residual: Pipeline,
    seg: Pipeline,
    seg_scaled: Pipeline,
    norm_add_next: Pipeline,
    norm_next: Pipeline,
}

impl MatvecPipelines {
    fn new(library: &Library, names: [&str; 7], simdgroups: u32) -> Result<Self, DeviceError> {
        Ok(Self {
            store: library.pipeline_with_simdgroups(names[0], simdgroups)?,
            norm: library.pipeline_with_simdgroups(names[1], simdgroups)?,
            residual: library.pipeline_with_simdgroups(names[2], simdgroups)?,
            seg: library.pipeline_with_simdgroups(names[3], simdgroups)?,
            seg_scaled: library.pipeline_with_simdgroups(names[4], simdgroups)?,
            norm_add_next: library.pipeline(names[5])?,
            norm_next: library.pipeline(names[6])?,
        })
    }

    /// The segmented entry of one type.
    pub(crate) fn seg(&self) -> &Pipeline {
        &self.seg
    }

    /// The segmented entry over a scaled input.
    pub(crate) fn seg_scaled(&self) -> &Pipeline {
        &self.seg_scaled
    }

    /// The sandwich join whose next launch is of this type.
    pub(crate) fn norm_add_next(&self) -> &Pipeline {
        &self.norm_add_next
    }

    /// The plain residual's hand-on whose next launch is of this type.
    pub(crate) fn norm_next(&self) -> &Pipeline {
        &self.norm_next
    }
}

/// The compute pipelines of the quantized kernels.
pub struct Kernels {
    q4k: MatvecPipelines,
    q6k: MatvecPipelines,
    a4: MatvecPipelines,
    a8: MatvecPipelines,
    q8_0: MatvecPipelines,
    q5k: MatvecPipelines,
    /// The f32 weights' one mat-vec entry, the segmented one
    /// ([`Quant::dense`]).
    f32_seg: Pipeline,
    /// The dense GEMM of each weight type: every epilogue and operand,
    /// plain and segmented (`gemm.metal`, or `gemm_tensor.metal` on the
    /// tensor-op path).
    q4k_gemm: gemm::GemmPipelines,
    q6k_gemm: gemm::GemmPipelines,
    a4_gemm: gemm::GemmPipelines,
    a8_gemm: gemm::GemmPipelines,
    q8_0_gemm: gemm::GemmPipelines,
    q5k_gemm: gemm::GemmPipelines,
    /// The cast of the half-row GEMMs' activations (`gemm.metal`'s
    /// `gemm_cast_rows`): the engine library's on either path.
    gemm_cast: Pipeline,
    /// The program the dense and the grouped GEMMs run: the library's
    /// [`Library::gemm_path`].
    gemm_path: GemmPath,
    /// The gated delta net's out projection: the Q4_K entry (A4's, A8's,
    /// Q8_0's and Q5_K's are `headnorm_a4`, `headnorm_a8`, `headnorm_q8_0`
    /// and `headnorm_q5k`; Q6_K has none).
    headnorm: Pipeline,
    headnorm_a4: Pipeline,
    headnorm_a8: Pipeline,
    headnorm_q8_0: Pipeline,
    headnorm_q5k: Pipeline,
    embed_q4k: Pipeline,
    embed_a4: Pipeline,
    embed_a8: Pipeline,
    embed_q8_0: Pipeline,
    embed_q5k: Pipeline,
    router: Pipeline,
    /// The RMS norm over rows.
    norm_rows: Pipeline,
    /// The sandwich residual: the norm of rows added into the stream.
    norm_add_rows: Pipeline,
    /// The short conv: the step, a chunk's rows and the ring commit.
    shortconv: shortconv::Pipelines,
    /// The pass's routing, grouped experts and row glue (`route.metal`,
    /// `gemm_routed.metal`).
    routed: routed::RoutedPipelines,
    experts: experts::Pipelines,
    /// The gated delta step.
    delta: Pipeline,
    /// The gated delta step over rows, and the ring commit after it.
    delta_rows: delta::RowsPipelines,
    attention: attention::AttentionPipelines,
    /// The output head's logits dispatch: the Q6_K entry (A4's, A8's, Q8_0's
    /// and Q5_K's are `head_logits_a4`, `head_logits_a8`,
    /// `head_logits_q8_0` and `head_logits_q5k`; Q4_K has none).
    head_logits_q6k: Pipeline,
    head_logits_a4: Pipeline,
    head_logits_a8: Pipeline,
    head_logits_q8_0: Pipeline,
    head_logits_q5k: Pipeline,
    /// The logits dispatch over the last join's scaled input, of the same
    /// types.
    head_logits_q6k_scaled: Pipeline,
    head_logits_a4_scaled: Pipeline,
    head_logits_a8_scaled: Pipeline,
    head_logits_q8_0_scaled: Pipeline,
    head_logits_q5k_scaled: Pipeline,
    /// The output head's argmax over the logits, the one entry of every type.
    head_pick: Pipeline,
    /// The launch geometry of the library's instance: the numbers every
    /// launch sizes its grid and checks its windows by.
    geom: Geometry,
    /// The SIMD groups in a mat-vec's threadgroup: the kernels' function
    /// constant, and the geometry of every launch.
    simdgroups: usize,
    /// The output rows a mat-vec's SIMD group owns. The number is a constant
    /// of the kernels' source (`geometry::ROWS_PER_SIMD_GROUP` writes it),
    /// not a function constant: a variant count is another source, and the
    /// launches' geometry follows the count the library was built with.
    rows_per_simd_group: usize,
    /// The threads in the router's threadgroup, and its threadgroup count:
    /// the router's own geometry, its source constant by way of
    /// `router_geom`, followed by the launch.
    router_threads: usize,
    router_groups: usize,
}

impl Kernels {
    /// The pipelines of the device's own library.
    pub fn new(device: &MetalDevice) -> Result<Self, DeviceError> {
        Self::from_library(device.library())
    }

    /// The pipelines of `library`: a missing kernel is a named error.
    pub fn from_library(library: &Library) -> Result<Self, DeviceError> {
        Self::with_simdgroups(library, SIMD_GROUPS_PER_GROUP)
    }

    /// The pipelines of `library` with `simdgroups` SIMD groups in a
    /// threadgroup. The count is the kernels' function constant, so each count
    /// is its own pipeline, and every launch's geometry follows it.
    pub fn with_simdgroups(library: &Library, simdgroups: usize) -> Result<Self, DeviceError> {
        Self::with_geometry(
            library,
            geometry::ROWS_PER_SIMD_GROUP,
            simdgroups,
            router_geom::SIMD_GROUPS,
        )
    }

    /// The pipelines of a geometry variant: `rows` is the row count the
    /// kernels' source was built with (the library's `#define`, which
    /// [`geometry::ROWS_PER_SIMD_GROUP`] writes for the shipped source),
    /// `simdgroups` the function constant of the mat-vec pipelines, and
    /// `router_sg` the router's own SIMD-group count, another source
    /// constant. The probe compiles variant sources and names their counts
    /// here; the launches' geometry follows what they name. A `rows` or
    /// `router_sg` the library's source does not define is refused by name.
    /// The numbers of every other launch are the library's instance's
    /// ([`Geometry::of`]).
    pub fn with_geometry(
        library: &Library,
        rows: usize,
        simdgroups: usize,
        router_sg: usize,
    ) -> Result<Self, DeviceError> {
        let count = u32::try_from(simdgroups).map_err(|_| DeviceError::Range {
            what: "matvec",
            detail: format!("{simdgroups} SIMD groups do not fit the kernel's word"),
        })?;
        if count == 0 {
            return Err(DeviceError::Range {
                what: "matvec",
                detail: "a threadgroup of no SIMD groups".to_string(),
            });
        }
        // A threadgroup holds at most the device's 1024 threads, and a SIMD
        // group is 32 of them.
        if simdgroups > geometry::MAX_SIMD_GROUPS_PER_GROUP {
            return Err(DeviceError::Range {
                what: "matvec",
                detail: format!(
                    "{simdgroups} SIMD groups in a threadgroup, the most is {}",
                    geometry::MAX_SIMD_GROUPS_PER_GROUP
                ),
            });
        }
        // The launches' grids and the buffers' lengths are the library's
        // instance's, read from the same row its kernels were compiled at.
        let geom = Geometry::of(library.instance());
        // The counts are the source's `#define`s: a count the library was not
        // compiled with would launch threadgroups over rows no kernel writes.
        let (built_rows, built_router_sg) = library.geometry();
        if built_rows != Some(rows) {
            return Err(DeviceError::Range {
                what: "matvec",
                detail: format!(
                    "{rows} rows a SIMD group; the library's source defines {built_rows:?}"
                ),
            });
        }
        if built_router_sg != Some(router_sg) {
            return Err(DeviceError::Range {
                what: "router",
                detail: format!(
                    "{router_sg} SIMD groups; the library's source defines {built_router_sg:?}"
                ),
            });
        }
        if ![2, 4, 8].contains(&rows) {
            return Err(DeviceError::Range {
                what: "matvec",
                detail: format!("{rows} rows a SIMD group, the kernels take 2, 4 or 8"),
            });
        }
        if router_sg == 0 || router_sg > geometry::MAX_SIMD_GROUPS_PER_GROUP {
            return Err(DeviceError::Range {
                what: "router",
                detail: format!(
                    "{router_sg} SIMD groups in a threadgroup, the kernels take 1 to {}",
                    geometry::MAX_SIMD_GROUPS_PER_GROUP
                ),
            });
        }
        let router_threads = router_geom::SIMD_WIDTH * router_sg;
        Ok(Self {
            q4k: MatvecPipelines::new(
                library,
                [
                    "q4k_matvec",
                    "q4k_matvec_norm",
                    "q4k_matvec_resid",
                    "q4k_matvec_seg",
                    "q4k_matvec_seg_scaled",
                    "q4k_norm_add_next",
                    "q4k_norm_next",
                ],
                count,
            )?,
            q6k: MatvecPipelines::new(
                library,
                [
                    "q6k_matvec",
                    "q6k_matvec_norm",
                    "q6k_matvec_resid",
                    "q6k_matvec_seg",
                    "q6k_matvec_seg_scaled",
                    "q6k_norm_add_next",
                    "q6k_norm_next",
                ],
                count,
            )?,
            a4: MatvecPipelines::new(
                library,
                [
                    "a4_matvec",
                    "a4_matvec_norm",
                    "a4_matvec_resid",
                    "a4_matvec_seg",
                    "a4_matvec_seg_scaled",
                    "a4_norm_add_next",
                    "a4_norm_next",
                ],
                count,
            )?,
            a8: MatvecPipelines::new(
                library,
                [
                    "a8_matvec",
                    "a8_matvec_norm",
                    "a8_matvec_resid",
                    "a8_matvec_seg",
                    "a8_matvec_seg_scaled",
                    "a8_norm_add_next",
                ],
                count,
            )?,
            q8_0: MatvecPipelines::new(
                library,
                [
                    "q8_0_matvec",
                    "q8_0_matvec_norm",
                    "q8_0_matvec_resid",
                    "q8_0_matvec_seg",
                    "q8_0_matvec_seg_scaled",
                    "q8_0_norm_add_next",
                    "q8_0_norm_next",
                ],
                count,
            )?,
            q5k: MatvecPipelines::new(
                library,
                [
                    "q5k_matvec",
                    "q5k_matvec_norm",
                    "q5k_matvec_resid",
                    "q5k_matvec_seg",
                    "q5k_matvec_seg_scaled",
                    "q5k_norm_add_next",
                    "q5k_norm_next",
                ],
                count,
            )?,
            f32_seg: library.pipeline_with_simdgroups("f32_matvec_seg", count)?,
            q4k_gemm: gemm::GemmPipelines::new(
                library.gemm_library(),
                "gemm_q4k",
                library.gemm_path(),
            )?,
            q6k_gemm: gemm::GemmPipelines::new(
                library.gemm_library(),
                "gemm_q6k",
                library.gemm_path(),
            )?,
            a4_gemm: gemm::GemmPipelines::new(
                library.gemm_library(),
                "gemm_a4",
                library.gemm_path(),
            )?,
            a8_gemm: gemm::GemmPipelines::new(
                library.gemm_library(),
                "gemm_a8",
                library.gemm_path(),
            )?,
            q8_0_gemm: gemm::GemmPipelines::new(
                library.gemm_library(),
                "gemm_q8_0",
                library.gemm_path(),
            )?,
            q5k_gemm: gemm::GemmPipelines::new(
                library.gemm_library(),
                "gemm_q5k",
                library.gemm_path(),
            )?,
            gemm_cast: library.pipeline("gemm_cast_rows")?,
            gemm_path: library.gemm_path(),
            headnorm: library.pipeline_with_simdgroups("q4k_matvec_headnorm_resid", count)?,
            headnorm_a4: library.pipeline_with_simdgroups("a4_matvec_headnorm_resid", count)?,
            headnorm_a8: library.pipeline_with_simdgroups("a8_matvec_headnorm_resid", count)?,
            headnorm_q8_0: library.pipeline_with_simdgroups("q8_0_matvec_headnorm_resid", count)?,
            headnorm_q5k: library.pipeline_with_simdgroups("q5k_matvec_headnorm_resid", count)?,
            embed_q4k: library.pipeline("embed_q4k")?,
            embed_a4: library.pipeline("embed_a4")?,
            embed_a8: library.pipeline("embed_a8")?,
            embed_q8_0: library.pipeline("embed_q8_0")?,
            embed_q5k: library.pipeline("embed_q5k")?,
            router: library.pipeline("router35")?,
            norm_rows: library.pipeline("norm_rows")?,
            norm_add_rows: library.pipeline("norm_add_rows")?,
            shortconv: shortconv::Pipelines::new(library)?,
            routed: routed::RoutedPipelines::new(library)?,
            experts: experts::Pipelines::new(library, count)?,
            delta: library.pipeline("gdn_step")?,
            delta_rows: delta::RowsPipelines::new(library)?,
            attention: attention::AttentionPipelines::new(library)?,
            head_logits_q6k: library.pipeline_with_simdgroups("q6k_head_logits", count)?,
            head_logits_a4: library.pipeline_with_simdgroups("a4_head_logits", count)?,
            head_logits_a8: library.pipeline_with_simdgroups("a8_head_logits", count)?,
            head_logits_q8_0: library.pipeline_with_simdgroups("q8_0_head_logits", count)?,
            head_logits_q5k: library.pipeline_with_simdgroups("q5k_head_logits", count)?,
            head_logits_q6k_scaled: library
                .pipeline_with_simdgroups("q6k_head_logits_scaled", count)?,
            head_logits_a4_scaled: library
                .pipeline_with_simdgroups("a4_head_logits_scaled", count)?,
            head_logits_a8_scaled: library
                .pipeline_with_simdgroups("a8_head_logits_scaled", count)?,
            head_logits_q8_0_scaled: library
                .pipeline_with_simdgroups("q8_0_head_logits_scaled", count)?,
            head_logits_q5k_scaled: library
                .pipeline_with_simdgroups("q5k_head_logits_scaled", count)?,
            // The argmax's SIMD-group count, not the mat-vec count this
            // function was given. The constant is the pipeline's function
            // constant.
            head_pick: library.pipeline_with_simdgroups(
                "logits_argmax",
                u32::try_from(head::HEAD_SIMD_GROUPS)
                    .expect("the head's SIMD-group count fits u32"),
            )?,
            geom,
            simdgroups,
            rows_per_simd_group: rows,
            router_threads,
            // The router's split is one, so a threadgroup count is the rows
            // over the SIMD groups the source names.
            router_groups: geom.router.rows.div_ceil(router_sg),
        })
    }

    /// The launch geometry of the library's instance.
    #[must_use]
    pub fn geom(&self) -> &Geometry {
        &self.geom
    }

    /// The program the dense and the grouped GEMMs run.
    #[must_use]
    pub fn gemm_path(&self) -> GemmPath {
        self.gemm_path
    }

    /// The SIMD groups in a mat-vec's threadgroup.
    #[must_use]
    pub fn simdgroups(&self) -> usize {
        self.simdgroups
    }

    /// The output rows a mat-vec's SIMD group owns.
    #[must_use]
    pub fn rows_per_simd_group(&self) -> usize {
        self.rows_per_simd_group
    }

    /// The threads in a mat-vec's threadgroup.
    #[must_use]
    pub fn threads_per_group(&self) -> usize {
        geometry::SIMD_WIDTH * self.simdgroups
    }

    /// The threads in the router's threadgroup, the geometry this object's
    /// library was built with.
    #[must_use]
    pub fn router_threads(&self) -> usize {
        self.router_threads
    }

    /// The router's threadgroup count.
    #[must_use]
    pub fn router_groups(&self) -> usize {
        self.router_groups
    }
}

/// A named range error. Launch modules build every refusal through this.
pub(crate) fn range(what: &'static str, detail: impl Into<String>) -> DeviceError {
    DeviceError::Range {
        what,
        detail: detail.into(),
    }
}

/// How a kernel reads a weight window.
pub(crate) enum WeightAlign {
    /// The kernel loads `float4` from the window.
    Float4,
    /// The kernel loads 32-bit words from the window.
    Word,
    /// The kernel loads 16-bit words from the window.
    Half,
}

/// `start` is the window's first byte in its buffer. A misaligned window is
/// refused by name: the message names the word the kernel loads.
pub(crate) fn check_weights_align(
    what: &'static str,
    name: &str,
    start: usize,
    kind: WeightAlign,
) -> Result<(), DeviceError> {
    match kind {
        WeightAlign::Float4 => {
            if start.is_multiple_of(16) {
                return Ok(());
            }
            Err(range(
                what,
                format!(
                    "the {name} window starts {start} bytes into its buffer and is not 16-byte aligned"
                ),
            ))
        }
        WeightAlign::Word => {
            if start.is_multiple_of(4) {
                return Ok(());
            }
            Err(range(
                what,
                format!(
                    "the {name} weights start {start} bytes into their buffer: the kernels read 32-bit words"
                ),
            ))
        }
        WeightAlign::Half => {
            if start.is_multiple_of(2) {
                return Ok(());
            }
            Err(range(
                what,
                format!(
                    "the {name} weights start {start} bytes into their buffer: the kernels read 16-bit words"
                ),
            ))
        }
    }
}

/// The bytes of `rows` rows of `nblocks` quantized blocks. `None` when the
/// product does not fit. An expert stack multiplies the result by the expert
/// count.
pub(crate) fn weight_bytes(quant: Quant, rows: usize, nblocks: usize) -> Option<usize> {
    rows.checked_mul(nblocks)?.checked_mul(quant.block_bytes())
}

/// The bytes a kernel reads: `words` in order, each already in the machine's
/// byte order.
pub(crate) fn pack<const N: usize>(words: &[[u8; 4]; N]) -> &[u8] {
    words.as_flattened()
}

/// A kernel's argument words, each in the machine's byte order. [`pack`] is
/// the bytes.
pub(crate) fn args_bytes<const N: usize>(words: [u32; N]) -> [[u8; 4]; N] {
    words.map(u32::to_ne_bytes)
}

/// A count as the kernel's 32-bit word, refused by name when it does not fit.
pub(crate) fn word(what: &'static str, name: &str, n: usize) -> Result<u32, DeviceError> {
    u32::try_from(n).map_err(|_| {
        range(
            what,
            format!("{name} of {n} does not fit the kernel's 32-bit word"),
        )
    })
}

/// A kernel window's length against the length the launch's shape needs,
/// refused by name when they differ: `name` says which window. (The
/// contract's [`bloomery_device::check_len`] is the other rule: a host slice
/// against the window it is copied to or from.)
pub(crate) fn check_window_len(
    what: &'static str,
    name: &str,
    got: bloomery_device::Elems,
    want: usize,
) -> Result<(), DeviceError> {
    if got.get() == want {
        return Ok(());
    }
    Err(range(
        what,
        format!(
            "the {name} window holds {} elements, the shape needs {want}",
            got.get()
        ),
    ))
}
