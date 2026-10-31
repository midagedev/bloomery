//! A weight site's type picks its launch: the one owner of which kernel a
//! dense projection of a file's type runs, shared by the bodies that read a
//! site in more than one type.
//!
//! A site is one projection of a layer — `y = W · x` over a unit's columns,
//! token-major (`y[c · rows + r]`). Its [`SiteTy`] is the file tensor's type,
//! read once at load from the file's header ([`file_site`], before any byte
//! is uploaded) and checked against the resident weight ([`site`]); a type no
//! launch here reads is refused by name, with the site.
//!
//! The launches by type and arm:
//! - [`gemv`], a unit of at most eight columns over the f32 rows: Q8_0 the
//!   q8f32 gemv (`q8_0_gemv` at one column, `q8_0_gemv_mcol` token-major
//!   past it, the two bit for bit equal on a column), F32 `f32_gemv` at one
//!   column and the F32 tile past it (`f32_tile_gemm`, token-major, bit for
//!   bit `f32_gemv` on every row and column), MXFP8 (`GgmlType::MXFP8`) the
//!   q8f32 MXFP8 gemv (`mxfp8_gemv` at one column, `mxfp8_gemv_mcol`
//!   token-major past it, bit for bit equal on a column), FP8_B128
//!   (`GgmlType::FP8_B128`) the same walk's FP8_B128 entries
//!   (`fp8_b128_gemv`, `fp8_b128_gemv_mcol`). BF16 uses `bf16_gemv`
//!   token-major over widened words, bit for bit the F32 entries.
//!   A K-quant site's gemv is [`kgemv`]. A Q5_0 site's is [`blocks32_gemv`]. An
//!   NVFP4 site's is [`kgemv`] at a width of whole q8_1 super-blocks of 256
//!   values and [`nvfp4_sel32_gemv`] at another ([`reads_at`]).
//! - [`kgemv`], a site at most eight columns over the q8_1 rows: Q3_K, Q4_K
//!   and Q6_K their gemvs, row-major (`m` outputs a row; a Q6_K row of an odd
//!   super-block count the down `_sel` over one expert, one column); Q5_K
//!   and IQ4_XS the down `_sel` over a stack of one expert, token-major (a slot
//!   a column); NVFP4 `nvfp4_gemv` over its planes times the tensor's scale,
//!   token-major. [`kgemv_order`] is this family's layout and is
//!   `None` for Q5_0. Where a body copies a row-major output token-major,
//!   and its fused groups of Q4_K launches, stay the body's.
//! - [`nvfp4_sel32_gemv`], an NVFP4 site of a width that is no multiple of 256,
//!   at most eight columns over the `Q8Blocks32` rows: `nvfp4_gemv_sel32` over
//!   the planes as one expert times the tensor's scale, token-major.
//! - [`blocks32_gemv`], a Q5_0 site at most eight columns over the
//!   `Q8Blocks32` rows: `enqueue_gemv_q5_0` over the packed weight, the site
//!   as one expert (`row0 = 0`), row-major (`y[r · m + c]`). That row-major
//!   layout is this function's, not [`kgemv_order`]'s.
//! - [`gemm`], a wide unit through a route table: a K-quant the grouped
//!   int8 GEMM over the q8_1 blocks of 128 values ([`GemmAct`]), Q8_0 the
//!   32-value GEMM over the q8 blocks of 32 values ([`GemmAct32`]) on the
//!   q8f32 planes, Q5_0 the same GEMM on the packed weight the gemv reads
//!   (`gemm_q5_0p`, on the unit's one-expert table only), F32 the F32 tile
//!   over the f32 rows (one column a slot, on the unit's one-expert table
//!   only), MXFP8 and FP8_B128 the same tile over their planes
//!   (`mxfp8_tile_gemm`, `fp8_b128_tile_gemm`, the values decoded as they
//!   are staged), BF16 the same tile over widened words (`bf16_tile_gemm`).
//!   On a compute capability 12 device, BF16 and the E4M3 formats instead use
//!   BF16 tensor-core operands
//!   with F32 accumulation (`bf16_mma_gemm`, `mxfp8_mma_gemm`, `fp8_b128_mma_gemm`);
//!   a cell whose row or column holds an exceptional operand replays the exact
//!   F32 tile within that kernel, and every other cell keeps its MMA value.
//!   A site's weight is the resident tensor of its name, rows of it (a part
//!   of a weight joined at load), or, for a K-quant, the words a caller
//!   holds ([`SiteWeight`]). NVFP4 the 32-value GEMM over its planes
//!   (`Gemm32Weight::Nvfp4Planes`, the tensor's scale applied at the store) on
//!   the q8 blocks of 32 of the unit's one-expert table, at any width of the
//!   format's 64-value blocks; a caller quantizes them as a Q8_0 site's.
//! - [`gated`], a wide gate·up·down block over one table: the gate and up
//!   [`gemm`]s, the gate·up rule's SwiGLU written into the form the down's
//!   type reads (the down's activation), and the down [`gemm`] over those
//!   rows, one column a slot.
//!
//! Which activation form a site reads is [`SiteTy::reads`] on the gemv arm
//! ([`reads_at`] at its width) and [`reads_wide`] on the wide arm: a body
//! quantizes a unit's rows into exactly the forms its sites read.
//!
//! A body that names a weight and leaves its type to the file calls a site
//! by name: [`resident_gemv`] at one column, [`resident_gemv_cols`] at up to
//! eight, and on the wide arm [`resident_gemm`] and [`resident_gemm_rows`]
//! (over rows of a joined weight). Each launches
//! what the typed call launches for the site type of the weight's resident
//! form, which the kernel table's dense entry decides
//! ([`of_resident`]); [`resident_site`] checks that type and shape
//! at load. A body whose sites include the K-quants and NVFP4, which read the
//! q8_1 rows of their input, projects by name through [`ByName`]: the same
//! decision, the quantization and the token-major copy the type needs, and
//! an output that is a part of a joined layout.

use crate::gemm::{
    Gemm32Args, Gemm32Kernels, Gemm32Weight, GemmAct, GemmAct32, GemmArgs, GemmInput, GemmKernels,
    GemmRoute, GemmWeight, MmaForm, ProductOut,
};
use crate::kquant::{Act, SelDown};
use crate::model::fit::{act_col_bytes, act32_col_bytes};
use crate::mxfp4::Nvfp4Kernels;
use crate::mxfp4_sel::{Nvfp4Down, Nvfp4SelKernels, Nvfp4Stack};
use crate::q5::Q8Blocks32;
use crate::q6k_sel::Q6kSelKernels;
use crate::q8f32::{
    Bf16GemvMcolArgs, E4m3GemvArgs, E4m3Scales, E4m3Weight, GemvOut, Q8_0GemvMcolArgs, SelArgs,
};
use crate::quant_sel::QuantSelKernels;
use crate::tensor::{Pitched, Q8Act, TensorWindow, WindowMut, dtod_2d};
use crate::token_major::TokenMajor;
use crate::weights::{DevWeight, Weights};
use crate::{DeviceTensor, FaultSink, Gpu, GpuError};
use chain::site::list;
use cuda_core::{DeviceBuffer, DeviceCopy};
use gguf::quant::GgmlType;
use model::placement::kernels;

pub use chain::site::{SiteTy, file_site};

/// The activation form a site's launch reads: the kernel table's fact.
pub use model::placement::kernels::SiteAct as Form;

/// The activation form a site of family `ty` reads over rows of `k` values: [`SiteTy::reads`],
/// the kernel table's, except NVFP4 at a width the q8_1 super-blocks of 256 values do not
/// divide, which reads the 32-value [`Form::Q8Blocks32`] its one-expert `_sel` gemv takes
/// ([`nvfp4_sel32_gemv`]). A body picks a gemv-arm site's launch and the forms it quantizes into
/// by this; the wide arm's form is [`reads_wide`].
#[must_use]
pub fn reads_at(ty: SiteTy, k: usize) -> Form {
    match kernels::dense(ty.ggml()) {
        Some(entry) => entry.reads_at(k as u64),
        None => ty.reads(),
    }
}

/// The activation form a site of family `ty` reads on the wide arm ([`gemm`]): [`SiteTy::reads`],
/// except NVFP4, whose GEMM reads the q8 blocks of 32 ([`Form::Q8x32`]) at any width of its
/// 64-value blocks while its gemv reads the q8_1 rows ([`reads_at`]). [`WideActs::wants`] and the
/// form a [`gated`] block's SwiGLU writes follow this, so the arm's forms have one owner.
#[must_use]
pub fn reads_wide(ty: SiteTy) -> Form {
    kernels::dense(ty.ggml()).map_or_else(|| ty.reads(), kernels::SiteEntry::reads_wide)
}

/// Where [`kgemv`] writes output `r` of column `c` of a site of `rows` rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    /// `y[r · m + c]`, the `m` columns of a row together.
    RowMajor,
    /// `y[c · rows + r]`.
    TokenMajor,
}

/// The family of a resident weight: the kernel table's dense entry for the
/// file type its form holds ([`SiteTy::of_ggml`]; an F32 plane is the f32
/// site whatever type it was widened from). `None` for a form no dense entry
/// reads: a stack in the file's blocks of a type whose entry reads another
/// layout (Q8_0's planes), a derived weight and an NVFP4 stack.
/// The resident form is this crate's, so the map is here, beside the
/// launches that read it, and not on the chain's [`SiteTy`].
#[must_use]
pub fn of_resident(w: &DevWeight) -> Option<SiteTy> {
    let ty = match w {
        DevWeight::KQuant { ty, .. } => {
            return SiteTy::of_ggml(*ty).filter(|s| s.kquant() || *s == SiteTy::Iq4Xs);
        }
        DevWeight::Q5_0 { .. } => GgmlType::Q5_0,
        DevWeight::Q5_1 { .. } => GgmlType::Q5_1,
        DevWeight::Q8_0 { .. } => GgmlType::Q8_0,
        DevWeight::F32 { .. } => GgmlType::F32,
        DevWeight::Nvfp4 { .. } => GgmlType::NVFP4,
        DevWeight::Mxfp8 { .. } => GgmlType::MXFP8,
        DevWeight::Fp8B128 { .. } => GgmlType::FP8_B128,
        DevWeight::Bf16 { .. } => GgmlType::BF16,
        DevWeight::Q8_0Derived { .. } | DevWeight::Nvfp4Stack { .. } => return None,
    };
    SiteTy::of_ggml(ty)
}

/// How [`kgemv`] lays a site out: Q3_K, Q4_K and Q6_K row-major, Q5_K, IQ4_XS and NVFP4 token-major.
/// `None` for a type [`kgemv`] does not run: Q8_0, F32, MXFP8, FP8_B128 and BF16 ([`gemv`]), Q5_0
/// ([`blocks32_gemv`], whose row-major layout `y[r · m + c]` is that function's own), and the MLX
/// affine kinds, a backend's families no card launch reads.
#[must_use]
pub fn kgemv_order(ty: SiteTy) -> Option<Order> {
    match ty {
        SiteTy::Q3K | SiteTy::Q4K | SiteTy::Q6K => Some(Order::RowMajor),
        SiteTy::Q5K | SiteTy::Iq4Xs | SiteTy::Nvfp4 => Some(Order::TokenMajor),
        SiteTy::Q5_0
        | SiteTy::Q8_0
        | SiteTy::F32
        | SiteTy::Mxfp8
        | SiteTy::Fp8B128
        | SiteTy::Bf16
        | SiteTy::MlxA4
        | SiteTy::MlxA8
        | SiteTy::MlxA4h
        | SiteTy::MlxA8h => None,
    }
}

/// The grouped GEMM's type of a K-quant, the one the kernel table's GEMM entry for the file type names
/// (`kernels::gemm`); `None` for Q5_0, Q8_0, F32, MXFP8, FP8_B128, BF16 and NVFP4, whose wide GEMM is the
/// 32-value family's ([`gemm`]).
#[must_use]
pub fn gemm_weight(ty: SiteTy) -> Option<GemmWeight> {
    kernels::gemm(ty.ggml()).and_then(GemmWeight::of_entry)
}

/// The resident f32 vector `name` of `len` values, as its name: a vector of another length is refused by name as
/// `what`'s.
pub fn vector(
    w: &Weights,
    what: &'static str,
    name: String,
    len: usize,
) -> Result<String, GpuError> {
    let got = w.f32_buf(what, &name)?.len();
    if got != len {
        return Err(GpuError::shape(
            what,
            format!("{name} holds {got} values; the layer reads {len}"),
        ));
    }
    Ok(name)
}

/// The resident weight of site `name`: of type `ty`, `rows` rows of `k`
/// values — the type its file entry was read as ([`file_site`]); any other
/// variant or shape is refused by name ([`resident_site`]).
pub fn site(
    w: &Weights,
    what: &'static str,
    name: &str,
    shape: (usize, usize),
    ty: SiteTy,
) -> Result<(), GpuError> {
    resident_site(w, what, name, shape, &[ty]).map(drop)
}

/// The family of the resident weight `name` ([`of_resident`]), `rows`
/// rows (every expert's of a stack) of `k` values: a family outside
/// `allowed`, a form no site launch reads, another shape or a weight that is
/// not resident is refused by name as `what`'s.
pub fn resident_site(
    w: &Weights,
    what: &'static str,
    name: &str,
    (rows, k): (usize, usize),
    allowed: &[SiteTy],
) -> Result<SiteTy, GpuError> {
    let t = w.resident(what, name)?;
    let (r, wk) = (t.rows(), t.k());
    match of_resident(t) {
        Some(ty) if allowed.contains(&ty) && (r, wk) == (rows, k) => Ok(ty),
        Some(ty) => Err(GpuError::shape(
            what,
            format!(
                "{name} is resident as {ty} {r} rows of {wk}; the site takes {} {rows} of {k}",
                list(allowed)
            ),
        )),
        None => Err(GpuError::shape(
            what,
            format!(
                "{name} is resident in a form no site launch reads; the site takes {}",
                list(allowed)
            ),
        )),
    }
}

/// The types a dense site of a resident weight launches by name on both
/// arms, the gemv over `m` columns ([`resident_gemv_cols`]) and the wide
/// GEMM ([`resident_gemm`]): those whose launch reads the f32 rows or the q8
/// blocks of 32 ([`SiteTy::reads`]; a unit test holds the list to it).
pub(crate) const BY_NAME: [SiteTy; 5] = [
    SiteTy::Q8_0,
    SiteTy::F32,
    SiteTy::Mxfp8,
    SiteTy::Fp8B128,
    SiteTy::Bf16,
];

/// The types a dense site launches by name through [`ByName`] over up to
/// [`BY_NAME_COLS`] columns: [`BY_NAME`] and the families that read the q8_1
/// blocks of 128 (the K-quants, IQ4_XS, NVFP4), every dense family but Q5_0. A unit
/// test holds the list to `placement::accepts::dense_projected`.
pub(crate) const PROJECTED: [SiteTy; 11] = [
    SiteTy::Q3K,
    SiteTy::Q4K,
    SiteTy::Q5K,
    SiteTy::Q6K,
    SiteTy::Iq4Xs,
    SiteTy::Q8_0,
    SiteTy::F32,
    SiteTy::Nvfp4,
    SiteTy::Mxfp8,
    SiteTy::Fp8B128,
    SiteTy::Bf16,
];

/// The resident K-quant words of `name`, or a named refusal.
fn kq<'w>(
    w: &'w Weights,
    what: &'static str,
    name: &str,
) -> Result<&'w crate::DeviceTensor<u32>, GpuError> {
    match w.get(name) {
        Some(DevWeight::KQuant { w, .. }) => Ok(w),
        Some(_) => Err(GpuError::tensor(what, name, "a K-quant word plane")),
        None => Err(GpuError::tensor(what, name, "resident")),
    }
}

/// `y = W · x` for site `name` of type `ty` over the first `m` (1..=8)
/// columns of the f32 rows `x`, token-major (module doc): Q8_0, F32,
/// MXFP8, FP8_B128 and BF16. A K-quant site is
/// [`kgemv`] and a Q5_0 site is [`blocks32_gemv`]; each is refused here by
/// name.
pub fn gemv(
    gpu: &Gpu,
    g32: &Gemm32Kernels,
    (ty, w, name): (SiteTy, &Weights, &str),
    x: &DeviceBuffer<f32>,
    m: usize,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let src = (Some(ty), w, name);
    launch(gpu, gpu.stream(), Some(g32), src, x, m, y)
}

/// [`gemv`] for the resident weight `name` at the site type of its resident
/// form ([`of_resident`]) over the first `m` (1..=8) columns of the
/// f32 rows `x`, token-major: a body whose plan picks each site's type per
/// tensor. `g32` is read only by an F32 site past one column, which is
/// refused by name without it.
pub fn resident_gemv_cols(
    gpu: &Gpu,
    g32: Option<&Gemm32Kernels>,
    (w, name): (&Weights, &str),
    x: &DeviceBuffer<f32>,
    m: usize,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    resident_gemv_cols_on(gpu, gpu.stream(), g32, (w, name), x, m, y)
}

/// [`resident_gemv_cols`] on `stream`, ordered by the caller before its consumers.
pub fn resident_gemv_cols_on(
    gpu: &Gpu,
    stream: &cuda_core::CudaStream,
    g32: Option<&Gemm32Kernels>,
    (w, name): (&Weights, &str),
    x: &DeviceBuffer<f32>,
    m: usize,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    launch(gpu, stream, g32, (None, w, name), x, m, y)
}

/// `y = W · x` at one column for the resident weight `name`, by the site
/// type of its resident form ([`resident_gemv_cols`] at `m` = 1): the
/// one-column gemv of a model that names a projection and leaves its format
/// to the file.
pub fn resident_gemv(
    gpu: &Gpu,
    w: &Weights,
    name: &str,
    x: &DeviceBuffer<f32>,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    resident_gemv_cols(gpu, None, (w, name), x, 1, y)
}

/// The expert-select gemv over the f32 rows (`a`, [`SelArgs`]) of the
/// resident stack `name`, by its resident form: Q8_0 planes
/// `q8_0_gemv_sel_f32`, FP8_B128 planes `fp8_b128_gemv_sel_f32`, each slot's
/// rows bit for bit the form's one-column gemv of that expert's rows. Any
/// other form is refused by name.
pub fn resident_sel_f32(
    gpu: &Gpu,
    w: &Weights,
    name: &str,
    a: &SelArgs<'_>,
    fault: FaultSink,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    const WHAT: &str = "site::resident_sel_f32";
    let (q, stream) = (gpu.q8f32(), gpu.stream());
    match w.resident(WHAT, name)? {
        DevWeight::Q8_0 { qs, d, .. } => q.enqueue_q8_0_gemv_sel_f32(stream, (qs, d), a, fault, y),
        DevWeight::Fp8B128 { .. } => {
            q.enqueue_e4m3_gemv_sel_f32(stream, w.e4m3(WHAT, name)?, a, fault, y)
        }
        DevWeight::Nvfp4Stack { .. } => {
            q.enqueue_nvfp4_gemv_sel_f32(stream, w.nvfp4_stack(WHAT, name)?, a, fault, y)
        }
        _ => Err(GpuError::tensor(
            WHAT,
            name,
            "a q8_0, fp8_b128 or nvfp4 stack",
        )),
    }
}

/// The site type of the resident weight `name` ([`of_resident`]): a
/// missing weight, or a form no site launch reads, is refused by name as
/// `what`'s.
pub(crate) fn resident_type(
    w: &Weights,
    what: &'static str,
    name: &str,
) -> Result<SiteTy, GpuError> {
    resident_ty(what, (None, w.resident(what, name)?, name))
}

/// The site type of the resident weight `name` ([`of_resident`]),
/// which must be `declared` when the caller names one: another type, or a
/// form no site launch reads, is refused by name as `what`'s.
fn resident_ty(
    what: &'static str,
    (declared, w, name): (Option<SiteTy>, &DevWeight, &str),
) -> Result<SiteTy, GpuError> {
    match (of_resident(w), declared) {
        (Some(ty), None) => Ok(ty),
        (Some(ty), Some(d)) if ty == d => Ok(ty),
        (Some(ty), Some(d)) => Err(GpuError::shape(
            what,
            format!("{name} is resident as {ty}; the site takes {d}"),
        )),
        (None, _) => Err(GpuError::shape(
            what,
            format!("{name} is resident in a form no site launch reads"),
        )),
    }
}

/// The gemv of resident weight `name` over the first `m` (1..=8) columns
/// of the f32 rows `x`, token-major, at the site type of its resident form, which must be
/// `declared` when the caller names one ([`resident_ty`]). `g32` is read
/// only by an F32 site past one column.
fn launch(
    gpu: &Gpu,
    stream: &cuda_core::CudaStream,
    g32: Option<&Gemm32Kernels>,
    (declared, w, name): (Option<SiteTy>, &Weights, &str),
    x: &DeviceBuffer<f32>,
    m: usize,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    const WHAT: &str = "site::gemv";
    let weight = w.resident(WHAT, name)?;
    let ty = resident_ty(WHAT, (declared, weight, name))?;
    match ty.reads() {
        Form::F32 | Form::Q8x32 => with_planes(WHAT, (ty, weight, name), None, |p| {
            gemv_planes(gpu, stream, g32, p, x, m, y)
        }),
        Form::Q8x128 => Err(GpuError::shape(
            WHAT,
            format!("{name} is {ty}: its gemv is `site::kgemv`, on the q8_1 rows"),
        )),
        Form::Q8Blocks32 => Err(GpuError::shape(
            WHAT,
            format!("{name} is {ty}: its gemv is `site::blocks32_gemv`, on the Q8Blocks32 rows"),
        )),
    }
}

/// `y = W · x` for the planes `p` over the first `m` (1..=8) columns of the
/// f32 rows `x`, token-major: Q8_0 `q8_0_gemv` at one column and
/// `q8_0_gemv_mcol` past it, F32 `f32_gemv` and past one column the F32 tile
/// (`g32`), BF16 `bf16_gemv`, MXFP8 and FP8_B128 their gemvs.
fn gemv_planes(
    gpu: &Gpu,
    stream: &cuda_core::CudaStream,
    g32: Option<&Gemm32Kernels>,
    p: Planes<'_>,
    x: &DeviceBuffer<f32>,
    m: usize,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let q = gpu.q8f32();
    match p {
        Planes::Q8_0(qs, d) if m == 1 => q.enqueue_q8_0_gemv(stream, qs, d, x, 1, y),
        Planes::Q8_0(qs, d) => q.enqueue_q8_0_gemv_mcol(
            stream,
            Q8_0GemvMcolArgs {
                qs,
                d,
                x,
                m,
                out: GemvOut::TokenMajor,
                y,
            },
        ),
        // One column runs `f32_gemv`, a warp a row: the tile gives the same bits in one block
        // for a site of 32 rows or fewer.
        Planes::F32(w) if m == 1 => q.enqueue_f32_gemv(stream, w, x, 1, y),
        Planes::F32(w) => g32
            .ok_or(GpuError::state("site::gemv", "the F32 tile's kernels"))?
            .enqueue_f32_tile(stream, w, x, m, y),
        Planes::Bf16(w) => q.enqueue_bf16_gemv_mcol(
            stream,
            Bf16GemvMcolArgs {
                w,
                x,
                m,
                out: GemvOut::TokenMajor,
                y,
            },
        ),
        Planes::E4m3(w) => q.enqueue_e4m3_gemv(
            stream,
            E4m3GemvArgs {
                w,
                x,
                m,
                out: GemvOut::TokenMajor,
                y,
            },
        ),
    }
}

/// Rows `at .. at + n` of a weight a site reads, `(at, n)`; all of them at
/// `None`.
type Part = Option<(usize, usize)>;

/// A plane of a dense weight as a launch reads it: the resident plane, or a
/// window of its rows.
enum View<'t, T> {
    All(&'t DeviceTensor<T>),
    Rows(TensorWindow<'t, T>),
}

impl<T> std::ops::Deref for View<'_, T> {
    type Target = DeviceTensor<T>;

    fn deref(&self) -> &DeviceTensor<T> {
        match self {
            View::All(t) => t,
            View::Rows(w) => w,
        }
    }
}

/// Plane `t` of site `name`, or its rows `at .. at + n` at `part`: an empty
/// part or one past the plane's rows is refused by name as `what`'s.
fn view<'t, T: DeviceCopy>(
    what: &'static str,
    (name, part): (&str, Part),
    t: &'t DeviceTensor<T>,
) -> Result<View<'t, T>, GpuError> {
    let Some((at, n)) = part else {
        return Ok(View::All(t));
    };
    if n == 0 || at.checked_add(n).is_none_or(|end| end > t.rows()) {
        return Err(GpuError::shape(
            what,
            format!(
                "{name}: rows {at} .. {at} + {n} of a weight of {} rows",
                t.rows()
            ),
        ));
    }
    let off = at
        .checked_mul(t.cols())
        .and_then(|v| v.checked_mul(size_of::<T>()))
        .ok_or_else(|| GpuError::shape(what, format!("{name}: row {at}'s byte overflows")))?;
    DeviceTensor::window_of(t.buf(), off, n, t.cols()).map(View::Rows)
}

/// The planes of a dense site whose launch reads the f32 rows or q8 blocks
/// of 32 (Q8_0, F32, MXFP8, FP8_B128, BF16).
enum Planes<'p> {
    Q8_0(&'p DeviceTensor<u32>, &'p DeviceTensor<u16>),
    F32(&'p DeviceTensor<f32>),
    Bf16(&'p DeviceTensor<u32>),
    E4m3(E4m3Weight<'p>),
}

/// `f` over the planes of the resident weight `name` as a site of type
/// `ty` reads them ([`Planes`]), each plane's rows `part` (all of them at
/// `None`). A form that is not the one the type's dense entry reads, a type
/// with no such planes and a part past the rows are refused by name as
/// `what`'s.
fn with_planes<R>(
    what: &'static str,
    (ty, w, name): (SiteTy, &DevWeight, &str),
    part: Part,
    f: impl FnOnce(Planes<'_>) -> Result<R, GpuError>,
) -> Result<R, GpuError> {
    let at = (name, part);
    match (ty, w) {
        (SiteTy::Q8_0, DevWeight::Q8_0 { qs, d, .. }) => {
            f(Planes::Q8_0(&*view(what, at, qs)?, &*view(what, at, d)?))
        }
        (SiteTy::F32, DevWeight::F32 { w, .. }) => f(Planes::F32(&*view(what, at, w)?)),
        (SiteTy::Bf16, DevWeight::Bf16 { w, .. }) => f(Planes::Bf16(&*view(what, at, w)?)),
        (SiteTy::Mxfp8, DevWeight::Mxfp8 { qs, e, .. })
        | (SiteTy::Fp8B128, DevWeight::Fp8B128 { qs, e, .. }) => {
            let scales = if ty == SiteTy::Mxfp8 {
                E4m3Scales::Mxfp8
            } else {
                E4m3Scales::Fp8B128
            };
            let (qs, e) = (view(what, at, qs)?, view(what, at, e)?);
            f(Planes::E4m3(E4m3Weight {
                qs: &qs,
                e: &e,
                scales,
            }))
        }
        _ => Err(GpuError::shape(
            what,
            format!("{name} is resident in a form a {ty} site's planes are not"),
        )),
    }
}

/// The most columns a [`Q8Act`] holds, so the most a [`kgemv`] runs.
const KGEMV_COLS: usize = 8;

/// The kernels a site's [`kgemv`] launches beyond the `Gpu`'s own: the
/// K-quant `_sel` family, whose Q5_K down entry over a stack of one expert is
/// the Q5_K gemv, the Q6_K `_sel` entry, which a Q6_K row of an odd
/// super-block count runs the same way, NVFP4's gemv and its `_sel` down
/// ([`nvfp4_sel32_gemv`]), and that one expert's id for each of
/// [`KGEMV_COLS`] slots.
pub struct KGemvKernels {
    kq: QuantSelKernels,
    q6: Q6kSelKernels,
    nv: Nvfp4Kernels,
    nvsel: Nvfp4SelKernels,
    one: DeviceBuffer<u32>,
}

impl KGemvKernels {
    /// The device bytes the kernels hold beside their modules: the one
    /// expert's ids.
    pub fn device_bytes(&self) -> usize {
        self.one.num_bytes()
    }

    /// Load the `_sel` family into `gpu`'s context, raising into its fault
    /// word, and zero the ids. Load-time only.
    pub fn load(gpu: &Gpu) -> Result<KGemvKernels, GpuError> {
        Ok(KGemvKernels {
            kq: QuantSelKernels::load(gpu.context(), gpu.fault_word())?,
            q6: Q6kSelKernels::load(gpu.context(), gpu.fault_word())?,
            nv: Nvfp4Kernels::load(gpu.context())?,
            nvsel: Nvfp4SelKernels::load(gpu.context(), gpu.fault_word())?,
            one: DeviceBuffer::zeroed(gpu.stream(), KGEMV_COLS)?,
        })
    }
}

/// `y = W · act` for site `name` of type `ty` over the `act.m()` columns of
/// the q8_1 rows `act`, laid out as [`kgemv_order`] says: Q4_K, Q6_K
/// and Q3_K their gemvs, row-major; Q5_K and IQ4_XS the down `_sel` over the site as
/// one expert, token-major, an id past it raising on `fault`; NVFP4 its gemv
/// times the tensor's scale, token-major. Q8_0, F32, MXFP8, FP8_B128 and BF16 are [`gemv`];
/// Q5_0 is [`blocks32_gemv`]. Each is refused here by name.
pub fn kgemv(
    gpu: &Gpu,
    kk: &KGemvKernels,
    (ty, w, name): (SiteTy, &Weights, &str),
    act: &Q8Act,
    fault: FaultSink,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    kgemv_on(gpu, gpu.stream(), kk, (ty, w, name), act, fault, y)
}

/// [`kgemv`] on `stream`, which the caller orders before the output's readers.
#[allow(
    clippy::too_many_arguments,
    reason = "the card, the stream, the kernels, the site, the activation, the sink and the output (rust-quality R8)"
)]
pub fn kgemv_on(
    gpu: &Gpu,
    stream: &cuda_core::CudaStream,
    kk: &KGemvKernels,
    (ty, w, name): (SiteTy, &Weights, &str),
    act: &Q8Act,
    fault: FaultSink,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    const WHAT: &str = "site::kgemv";
    match ty {
        SiteTy::Q4K => gpu.enqueue_gemv_q4k_on(stream, kq(w, WHAT, name)?, act, y),
        // An odd super-block count leaves the rows unaligned to the words the
        // plain gemv reads: the `_sel` entry addresses rows by byte, so the
        // site runs as a stack of one expert, one column (where its layout
        // is the row-major one).
        SiteTy::Q6K if act.n_sb() % 2 == 1 => {
            if act.m() != 1 {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "{name} is Q6_K at an odd {} super-blocks a row: one column, got {}",
                        act.n_sb(),
                        act.m()
                    ),
                ));
            }
            let wt = kq(w, WHAT, name)?;
            kk.q6
                .enqueue_gemv_q6k_sel(stream, wt, act, &kk.one, 1, wt.rows(), y)
        }
        SiteTy::Q6K => gpu.enqueue_gemv_q6k_on(stream, kq(w, WHAT, name)?, act, y),
        SiteTy::Q3K => gpu.enqueue_gemv_q3k_on(stream, kq(w, WHAT, name)?, act, y),
        SiteTy::Q5K => {
            let wt = kq(w, WHAT, name)?;
            kk.kq.enqueue_gemv_q5k_sel(
                stream,
                &SelDown {
                    w: wt,
                    act,
                    sel: &kk.one,
                    n_slots: act.m(),
                    rows_per_expert: wt.rows(),
                },
                fault,
                y,
            )
        }
        SiteTy::Iq4Xs => {
            let wt = kq(w, WHAT, name)?;
            kk.kq.enqueue_gemv_sel(
                stream,
                GgmlType::IQ4_XS,
                &SelDown {
                    w: wt,
                    act,
                    sel: &kk.one,
                    n_slots: act.m(),
                    rows_per_expert: wt.rows(),
                },
                fault,
                y,
            )
        }
        SiteTy::Nvfp4 => kk.nv.enqueue_gemv(stream, w.nvfp4(WHAT, name)?, act, y),
        SiteTy::Q8_0 | SiteTy::F32 | SiteTy::Mxfp8 | SiteTy::Fp8B128 | SiteTy::Bf16 => {
            Err(GpuError::shape(
                WHAT,
                format!("{name} is {ty}: its gemv is `site::gemv`, on the f32 rows"),
            ))
        }
        SiteTy::Q5_0 => Err(GpuError::shape(
            WHAT,
            format!("{name} is {ty}: its gemv is `site::blocks32_gemv`, on the Q8Blocks32 rows"),
        )),
        // A backend's family, outside the card kernel table: no card path
        // reads a site of it.
        SiteTy::MlxA4 | SiteTy::MlxA8 | SiteTy::MlxA4h | SiteTy::MlxA8h => Err(GpuError::shape(
            WHAT,
            format!("{name} is {ty}: no card launch reads it"),
        )),
    }
}

/// `y = W · act` for Q5_0 site `name` over the `act.m()` columns of the
/// `Q8Blocks32` rows `act`, row-major (`y[r · m + c]`). The weight is the
/// packed plane [`Weights::load_where`] uploads (`pack_q5_0`, the one
/// owner). The site is one expert: `row0 = 0`, `n_rows` the plane's rows,
/// `col0 = 0`, `y0 = 0`. A missing packed weight, another type, or a plane
/// the quantizer's columns do not match is a named error.
pub fn blocks32_gemv(
    gpu: &Gpu,
    (ty, w, name): (SiteTy, &Weights, &str),
    act: &Q8Blocks32,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    const WHAT: &str = "site::blocks32_gemv";
    let SiteTy::Q5_0 = ty else {
        return Err(GpuError::shape(
            WHAT,
            format!("{name} is {ty}: `site::blocks32_gemv` runs a Q5_0 site"),
        ));
    };
    let Some(DevWeight::Q5_0 { w: wt, .. }) = w.get(name) else {
        return Err(GpuError::tensor(WHAT, name, "a packed Q5_0 weight"));
    };
    gpu.q5()
        .enqueue_gemv_q5_0(gpu.stream(), wt, act, 0, wt.rows(), 0, act.m(), y, 0)
}

/// `y = W · act` for NVFP4 site `name` whose row width is no multiple of 256 ([`reads_at`]) over
/// the `act.m()` (1..=8) columns of the `Q8Blocks32` rows `act`, token-major (`y[c · rows + r]`):
/// `nvfp4_gemv_sel32` over the planes as a stack of one expert, every column's id 0, times the
/// tensor's scale. A site of another type, a weight that is not NVFP4 planes, a width other than
/// the activation's and more columns than the ids hold are refused by name.
pub fn nvfp4_sel32_gemv(
    gpu: &Gpu,
    kk: &KGemvKernels,
    (ty, w, name): (SiteTy, &Weights, &str),
    act: &Q8Blocks32,
    fault: FaultSink,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    const WHAT: &str = "site::nvfp4_sel32_gemv";
    let SiteTy::Nvfp4 = ty else {
        return Err(GpuError::shape(
            WHAT,
            format!("{name} is {ty}: `site::nvfp4_sel32_gemv` runs an NVFP4 site"),
        ));
    };
    let DevWeight::Nvfp4 { qs, e, scales, .. } = w.resident(WHAT, name)? else {
        return Err(GpuError::tensor(WHAT, name, "an NVFP4 weight"));
    };
    let m = act.m();
    if m > KGEMV_COLS {
        return Err(GpuError::shape(
            WHAT,
            format!("{name}: {m} columns; the site's ids hold {KGEMV_COLS}"),
        ));
    }
    kk.nvsel.enqueue_sel32(
        gpu.stream(),
        &Nvfp4Down {
            w: Nvfp4Stack { qs, e, scales },
            act,
            sel: &kk.one,
            n_slots: m,
            rows_per_expert: qs.rows(),
            slots_per_col: m,
        },
        fault,
        y,
    )
}

/// The most columns a [`ByName`] projection runs: the K-quant gemv's.
pub const BY_NAME_COLS: usize = KGEMV_COLS;

/// The launches a [`ByName::gemv`] of a whole weight of type `ty` over `m`
/// columns makes beyond the one an f32-row site makes: a K-quant, IQ4_XS or
/// NVFP4 site quantizes its columns into q8_1 rows first, and a type whose gemv is
/// row-major ([`kgemv_order`]) copies its result token-major past one column.
/// A body counts its captured walk's launches by this and the weights'
/// resident types.
#[must_use]
pub fn by_name_extra(ty: SiteTy, m: usize) -> usize {
    if ty.reads() != Form::Q8x128 {
        return 0;
    }
    1 + usize::from(m > 1 && kgemv_order(ty) == Some(Order::RowMajor))
}

/// The launches a [`ByName`] projection makes beyond the f32-row gemv: the
/// K-quant `_sel` family and the row-major copy into token-major rows.
pub struct ByNameKernels {
    kgemv: KGemvKernels,
    tm: TokenMajor,
}

impl ByNameKernels {
    /// Load the K-quant family and the token-major copy into `gpu`'s
    /// context. Load-time only.
    pub fn load(gpu: &Gpu) -> Result<ByNameKernels, GpuError> {
        Ok(ByNameKernels {
            kgemv: KGemvKernels::load(gpu)?,
            tm: TokenMajor::load(gpu.context())?,
        })
    }
}

/// Where a [`ByName`] projection's output goes: column `c`, row `r` to
/// `y[c · stride + at + r]`. A whole weight into its own buffer is
/// `stride = rows, at = 0` ([`Out::whole`]); a part of a stack the consumer
/// reads joined is its row offset `at` in the joined stride.
pub struct Out<'y> {
    pub y: &'y mut DeviceBuffer<f32>,
    pub at: usize,
    pub stride: usize,
}

impl<'y> Out<'y> {
    /// The weight's own rows, token-major, from the start of `y`.
    pub fn whole(y: &'y mut DeviceBuffer<f32>) -> Out<'y> {
        Out {
            y,
            at: 0,
            stride: 0,
        }
    }

    /// Rows `at ..` of each `stride`-value column of `y`.
    pub fn part(y: &'y mut DeviceBuffer<f32>, at: usize, stride: usize) -> Out<'y> {
        Out { y, at, stride }
    }
}

/// The one by-name dense projection for a body that names a weight and
/// leaves its type to the file: any dense family of the kernel table but
/// Q5_0 over `1 ..= `[`BY_NAME_COLS`] columns, token-major, into a whole
/// buffer or a part of a joined layout. The f32-row families are
/// [`resident_gemv_cols`]'s launches; a K-quant or NVFP4 site quantizes the
/// columns into the q8_1 form of its width ([`ByName::for_sites`] made one a
/// width and column count), runs [`kgemv`] and copies a row-major result
/// token-major ([`kgemv_order`]). Each column is the site's one-column launch
/// on that row bit for bit.
pub struct ByName {
    acts: Vec<Q8Act>,
    /// A row-major gemv's output, or a part before its strided copy:
    /// `rows · columns` values at the widest site.
    stage: DeviceBuffer<f32>,
    /// A row-major gemv's token-major rows before a strided copy.
    rows: DeviceBuffer<f32>,
}

impl ByName {
    /// The forms and staging for the K-quant and NVFP4 sites among `names`
    /// (resident in `w`) at each column count `1 ..= cols` (`cols` at most
    /// [`BY_NAME_COLS`]); a body of only f32-row sites holds none. Load-time
    /// only.
    pub fn for_sites<'n>(
        gpu: &Gpu,
        w: &Weights,
        names: impl IntoIterator<Item = &'n str>,
        cols: usize,
    ) -> Result<ByName, GpuError> {
        const WHAT: &str = "site::ByName::for_sites";
        if !(1..=BY_NAME_COLS).contains(&cols) {
            return Err(GpuError::shape(
                WHAT,
                format!("{cols} columns; a by-name unit runs 1..={BY_NAME_COLS}"),
            ));
        }
        let stream = gpu.stream();
        let mut acts: Vec<Q8Act> = Vec::new();
        let mut widest = 0usize;
        for name in names {
            let weight = w.resident(WHAT, name)?;
            let Some(ty) = of_resident(weight) else {
                continue;
            };
            widest = widest.max(weight.rows());
            if ty.reads() != Form::Q8x128 {
                continue;
            }
            let k = weight.k();
            for m in 1..=cols {
                if !acts.iter().any(|a| a.k() == k && a.m() == m) {
                    acts.push(if m == 1 {
                        Q8Act::with_k(stream, 1, k)?
                    } else {
                        Q8Act::with_slots(stream, m, k)?
                    });
                }
            }
        }
        Ok(ByName {
            acts,
            stage: DeviceBuffer::zeroed(stream, (widest * cols).max(1))?,
            rows: DeviceBuffer::zeroed(stream, (widest * cols).max(1))?,
        })
    }

    /// Device bytes of the forms and the staging.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.acts.iter().map(Q8Act::device_bytes).sum::<usize>()
            + self.stage.num_bytes()
            + self.rows.num_bytes()
    }

    /// `out = W · x` for the resident weight `name`, by the site type of its
    /// resident form, over the first `m` (1..=[`BY_NAME_COLS`]) columns of
    /// the f32 rows `x`; a fault of the quantization raises into layer
    /// `layer`'s sink. `k` is read only by a K-quant, IQ4_XS or NVFP4 site, `g32`
    /// by an F32 site past one column ([`resident_gemv_cols`]). A Q5_0 site, a
    /// form the load made no q8_1 rows for, or a part past the output's
    /// stride is refused by name.
    #[allow(
        clippy::too_many_arguments,
        reason = "the projection, the card, the kernels, the site, the layer, the input, its columns and the output (rust-quality R8)"
    )]
    pub fn gemv(
        &mut self,
        gpu: &Gpu,
        kernels: (Option<&ByNameKernels>, Option<&Gemm32Kernels>),
        site: (&Weights, &str),
        layer: usize,
        x: &DeviceBuffer<f32>,
        m: usize,
        out: Out<'_>,
    ) -> Result<(), GpuError> {
        let fault = gpu.layer_sink(layer)?;
        self.gemv_on(gpu, gpu.stream(), kernels, site, fault, x, m, out)
    }

    /// [`ByName::gemv`] on `stream`, which the caller orders before the
    /// output's readers, a fault of the quantization raising into `fault`.
    /// Two projections on two streams at once hold a `ByName` each: a form
    /// and the staging are one projection's.
    #[allow(
        clippy::too_many_arguments,
        reason = "the card, the stream, the kernels, the site, the sink, the rows, their count and the output (rust-quality R8)"
    )]
    pub fn gemv_on(
        &mut self,
        gpu: &Gpu,
        stream: &cuda_core::CudaStream,
        (k, g32): (Option<&ByNameKernels>, Option<&Gemm32Kernels>),
        (w, name): (&Weights, &str),
        fault: FaultSink,
        x: &DeviceBuffer<f32>,
        m: usize,
        out: Out<'_>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "site::ByName::gemv";
        let weight = w.resident(WHAT, name)?;
        let ty = resident_ty(WHAT, (None, weight, name))?;
        let rows = weight.rows();
        let Out { y, at, stride } = out;
        let stride = if stride == 0 { rows } else { stride };
        if !(1..=BY_NAME_COLS).contains(&m) || at + rows > stride {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{name}: {m} columns of {rows} rows at {at} of a {stride}-value column; \
                     a by-name unit runs 1..={BY_NAME_COLS} columns inside the stride"
                ),
            ));
        }
        let whole = at == 0 && stride == rows;
        match ty.reads() {
            Form::F32 | Form::Q8x32 => {
                if whole {
                    return launch(gpu, stream, g32, (None, w, name), x, m, y);
                }
                if m == 1 {
                    let mut win = WindowMut::<f32>::of_mut(y, at * size_of::<f32>(), rows)?;
                    return launch(gpu, stream, g32, (None, w, name), x, 1, &mut win);
                }
                launch(gpu, stream, g32, (None, w, name), x, m, &mut self.stage)?;
                scatter(stream, &self.stage, (rows, m), y, (at, stride))
            }
            Form::Q8x128 => {
                let k = k.ok_or(GpuError::state(WHAT, "the by-name kernels"))?;
                let wk = weight.k();
                let act = self
                    .acts
                    .iter_mut()
                    .find(|a| a.k() == wk && a.m() == m)
                    .ok_or_else(|| {
                        GpuError::shape(
                            WHAT,
                            format!("{name}: no q8_1 form of {m} columns of {wk} values was made"),
                        )
                    })?;
                gpu.enqueue_quantize_q8_1_on(stream, x, act, fault)?;
                let key = (ty, w, name);
                let row_major = kgemv_order(ty) == Some(Order::RowMajor) && m > 1;
                if m == 1 && !whole {
                    let mut win = WindowMut::<f32>::of_mut(y, at * size_of::<f32>(), rows)?;
                    return kgemv_on(gpu, stream, &k.kgemv, key, act, fault, &mut win);
                }
                if !row_major && whole {
                    return kgemv_on(gpu, stream, &k.kgemv, key, act, fault, y);
                }
                if row_major {
                    kgemv_on(gpu, stream, &k.kgemv, key, act, fault, &mut self.stage)?;
                    if whole {
                        return k.tm.enqueue(stream, &self.stage, rows, m, y);
                    }
                    k.tm.enqueue(stream, &self.stage, rows, m, &mut self.rows)?;
                    return scatter(stream, &self.rows, (rows, m), y, (at, stride));
                }
                kgemv_on(gpu, stream, &k.kgemv, key, act, fault, &mut self.stage)?;
                scatter(stream, &self.stage, (rows, m), y, (at, stride))
            }
            Form::Q8Blocks32 => Err(GpuError::shape(
                WHAT,
                format!(
                    "{name} is {ty}: its activation is the Q8Blocks32 form, which no by-name unit holds"
                ),
            )),
        }
    }
}

/// `m` columns of `rows` values, token-major in `src`, into `dst` at `at` of
/// each `stride`-value column: one strided copy.
///
/// # Errors
/// The copy's launch error.
pub fn scatter(
    stream: &cuda_core::CudaStream,
    src: &DeviceBuffer<f32>,
    (rows, m): (usize, usize),
    dst: &mut DeviceBuffer<f32>,
    (at, stride): (usize, usize),
) -> Result<(), GpuError> {
    let f = size_of::<f32>();
    dtod_2d(
        stream,
        (src, 0),
        (dst, at * f),
        Pitched {
            width: rows * f,
            rows: m,
            dev_pitch: rows * f,
            host_pitch: stride * f,
        },
    )
}

/// Where a site reads its weight: the resident tensor of a name
/// (`Weights`), rows of it, or the K-quant or IQ4_XS words a caller holds — a
/// window of experts laid out as the resident stack of the site's type, kept
/// outside `Weights`. Each carries the site's name, which a refusal names.
/// Words under a type with no grouped GEMM, and rows of a site whose
/// launch reads neither the f32 rows nor q8 blocks of 32, are refused by
/// name ([`gemm`], [`gated`], [`resident_gemm_rows`]).
#[derive(Clone, Copy)]
pub enum SiteWeight<'a> {
    /// The resident weight of this name.
    Resident(&'a str),
    /// Rows `at .. at + n` of the resident weight `name`: one part of a
    /// weight joined at load (`Weights::join_rows`).
    Part { name: &'a str, at: usize, n: usize },
    /// The words of site `name`, `w`.
    Words {
        name: &'a str,
        w: &'a DeviceTensor<u32>,
    },
}

impl<'a> SiteWeight<'a> {
    /// The site's name.
    #[must_use]
    pub fn name(self) -> &'a str {
        match self {
            SiteWeight::Resident(name)
            | SiteWeight::Part { name, .. }
            | SiteWeight::Words { name, .. } => name,
        }
    }

    /// Words under a type with no grouped GEMM, and rows of a type whose
    /// launch reads neither the f32 rows nor q8 blocks of 32, refused by
    /// name as `what`.
    fn check(self, what: &'static str, ty: SiteTy) -> Result<(), GpuError> {
        match self {
            SiteWeight::Words { name, .. } if gemm_weight(ty).is_none() => Err(GpuError::shape(
                what,
                format!(
                    "{name} is {ty} on words a caller holds; its grouped GEMM reads no such words"
                ),
            )),
            SiteWeight::Part { name, .. } if !BY_NAME.contains(&ty) => Err(GpuError::shape(
                what,
                format!(
                    "{name} is {ty}: only a site on the f32 rows or q8 blocks of 32 launches \
                         rows of its weight"
                ),
            )),
            _ => Ok(()),
        }
    }

    /// The resident weight's name and the rows of it a site reads, all of
    /// them at `None`; words a caller holds are refused by name as `what`.
    fn rows(self, what: &'static str) -> Result<(&'a str, Part), GpuError> {
        match self {
            SiteWeight::Resident(name) => Ok((name, None)),
            SiteWeight::Part { name, at, n } => Ok((name, Some((at, n)))),
            SiteWeight::Words { name, .. } => Err(GpuError::shape(
                what,
                format!("{name} is on words a caller holds; this site reads a resident weight"),
            )),
        }
    }
}

/// The activation forms of a wide unit's rows a [`gemm`] picks from: the
/// q8_1 blocks of 128 values, the q8 blocks of 32, and the f32 rows. A form
/// no site of the call reads may be absent.
pub struct WideIn<'a> {
    pub q128: Option<&'a GemmAct>,
    pub q32: Option<&'a GemmAct32>,
    pub f32: Option<&'a DeviceBuffer<f32>>,
}

/// The kernels a wide site launches: the grouped GEMM over q8_1 blocks of
/// 128 always, the 32-value kernels when the load holds them — a load whose
/// sites are all K-quants holds none, and a Q8_0 or F32 site, or a SwiGLU into
/// q8 blocks of 32, is refused by name there.
pub struct WideKernels<'a> {
    pub gemm: &'a GemmKernels,
    pub g32: Option<&'a Gemm32Kernels>,
}

impl WideKernels<'_> {
    /// The 32-value kernels, or a named refusal naming `missing`, what the
    /// site needs them for.
    fn g32(&self, what: &'static str, missing: &'static str) -> Result<&Gemm32Kernels, GpuError> {
        self.g32.ok_or(GpuError::state(what, missing))
    }
}

/// Which forms of one wide input the sites of a unit read: q8_1 blocks of 128
/// for a K-quant site, q8 blocks of 32 for a Q8_0, Q5_0 or NVFP4 site, the f32
/// rows for an F32, MXFP8, FP8_B128 or BF16 site ([`reads_wide`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WideForms {
    pub q128: bool,
    pub q32: bool,
    pub f32: bool,
}

impl WideForms {
    /// No form.
    pub const NONE: WideForms = WideForms::q8(false, false);

    /// The quantized forms `q128` and `q32`, and no f32 rows.
    #[must_use]
    pub const fn q8(q128: bool, q32: bool) -> WideForms {
        WideForms {
            q128,
            q32,
            f32: false,
        }
    }

    /// `self` without the f32 rows, for a unit's input: its f32 rows are the
    /// caller's own buffer ([`WideActs::input`]), so the unit holds none.
    #[must_use]
    pub const fn without_f32(self) -> WideForms {
        WideForms { f32: false, ..self }
    }

    /// The forms either of `self` and `other` holds.
    #[must_use]
    pub const fn union(self, other: WideForms) -> WideForms {
        WideForms {
            q128: self.q128 | other.q128,
            q32: self.q32 | other.q32,
            f32: self.f32 | other.f32,
        }
    }

    /// Device bytes of `cols` columns of `k` values in these forms
    /// ([`WideActs::bytes`] of a [`WideActs::new`] over them).
    #[must_use]
    pub fn bytes(self, cols: usize, k: usize) -> usize {
        cols * (usize::from(self.q128) * act_col_bytes(k)
            + usize::from(self.q32) * act32_col_bytes(k)
            + usize::from(self.f32) * k * size_of::<f32>())
    }
}

/// One wide input's forms, each held when a site of the unit reads it: q8_1
/// blocks of 128 for a K-quant site, q8 blocks of 32 for a Q8_0, Q5_0 or NVFP4
/// site, and the f32 rows of a unit whose rows a SwiGLU writes (a down that
/// reads f32). An input's f32 rows are the caller's ([`WideActs::input`]).
pub struct WideActs {
    pub q128: Option<GemmAct>,
    pub q32: Option<GemmAct32>,
    pub f32: Option<DeviceBuffer<f32>>,
}

impl WideActs {
    /// The wide forms a site of type `ty` reads ([`reads_wide`]): q8_1 blocks
    /// of 128 for a K-quant site; q8 blocks of 32 (`GemmAct32`) for a Q8_0 site,
    /// for an NVFP4 site, whose gemv reads the q8_1 rows ([`reads_at`]) and
    /// whose GEMM reads these, and for a Q5_0 site, whose gemv reads
    /// `Q8Blocks32` and whose GEMM reads the same blocks in the wide layout;
    /// the f32 rows for a site that reads those.
    #[must_use]
    pub fn wants(ty: SiteTy) -> WideForms {
        let reads = reads_wide(ty);
        WideForms {
            q128: reads == Form::Q8x128,
            q32: matches!(reads, Form::Q8x32 | Form::Q8Blocks32),
            f32: reads == Form::F32,
        }
    }

    /// The wide forms any of `tys` reads ([`WideActs::wants`] over them).
    #[must_use]
    pub fn wants_of(tys: &[SiteTy]) -> WideForms {
        tys.iter()
            .fold(WideForms::NONE, |acc, &t| acc.union(WideActs::wants(t)))
    }

    /// `cols` columns of `k` values in the forms `forms` asks for. Load-time
    /// only.
    pub fn new(
        stream: &cuda_core::CudaStream,
        forms: WideForms,
        cols: usize,
        k: usize,
    ) -> Result<WideActs, GpuError> {
        Ok(WideActs {
            q128: forms
                .q128
                .then(|| GemmAct::new(stream, cols, k))
                .transpose()?,
            q32: forms
                .q32
                .then(|| GemmAct32::new(stream, cols, k))
                .transpose()?,
            f32: forms
                .f32
                .then(|| DeviceBuffer::zeroed(stream, cols * k))
                .transpose()?,
        })
    }

    pub fn bytes(&self) -> usize {
        self.q128.as_ref().map_or(0, GemmAct::bytes)
            + self.q32.as_ref().map_or(0, GemmAct32::bytes)
            + self.f32.as_ref().map_or(0, DeviceBuffer::num_bytes)
    }

    /// The forms `x`'s first `m` columns are quantized into for sites of
    /// types `tys` ([`WideActsMut::quantize`]).
    #[allow(
        clippy::too_many_arguments,
        reason = "the card, the kernels, the sink, the rows, their count and the site types (rust-quality R8)"
    )]
    pub fn quantize(
        &mut self,
        gpu: &Gpu,
        k: &WideKernels<'_>,
        sink: FaultSink,
        x: &DeviceBuffer<f32>,
        m: usize,
        tys: &[SiteTy],
    ) -> Result<(), GpuError> {
        self.views_mut().quantize(gpu, k, sink, x, m, tys)
    }

    /// The forms as views.
    pub(crate) fn views_mut(&mut self) -> WideActsMut<'_> {
        WideActsMut {
            q128: self.q128.as_mut(),
            q32: self.q32.as_mut(),
        }
    }

    /// The forms beside the f32 rows `f32` a site picks from.
    pub fn input<'a>(&'a self, f32: &'a DeviceBuffer<f32>) -> WideIn<'a> {
        WideIn {
            q128: self.q128.as_ref(),
            q32: self.q32.as_ref(),
            f32: Some(f32),
        }
    }

    /// The forms a gated block's SwiGLU writes: each this unit holds, the f32
    /// rows among them when its downs read those ([`WideActs::wants_of`]).
    pub fn swiglu_rows(&mut self) -> SwigluRows<'_> {
        SwigluRows {
            q128: self.q128.as_mut(),
            q32: self.q32.as_mut(),
            f32: self.f32.as_mut(),
        }
    }
}

/// One wide input's quantized forms as views: those of a [`WideActs`], or of
/// an arena that holds them as plain buffers, so one launch sequence runs over
/// either.
pub(crate) struct WideActsMut<'a> {
    pub q128: Option<&'a mut GemmAct>,
    pub q32: Option<&'a mut GemmAct32>,
}

impl WideActsMut<'_> {
    /// The forms `x`'s first `m` columns are quantized into for sites of
    /// types `tys`: each form one of them reads, and nothing else. A form a
    /// site reads that the unit does not hold, or q8 blocks of 32 on a load
    /// with no 32-value kernels, is refused by name.
    #[allow(
        clippy::too_many_arguments,
        reason = "the card, the kernels, the sink, the rows, their count and the site types (rust-quality R8)"
    )]
    pub fn quantize(
        &mut self,
        gpu: &Gpu,
        k: &WideKernels<'_>,
        sink: FaultSink,
        x: &DeviceBuffer<f32>,
        m: usize,
        tys: &[SiteTy],
    ) -> Result<(), GpuError> {
        const WHAT_Q: &str = "site::WideActsMut::quantize";
        let want = WideActs::wants_of(tys);
        if want.q128 {
            let a = self.q128.as_deref_mut().ok_or(GpuError::state(
                WHAT_Q,
                "the unit's q8_1 blocks of 128 (a K-quant site reads them)",
            ))?;
            gpu.enqueue_quantize_gemm(x, m, a, sink)?;
        }
        if want.q32 {
            let a = self.q32.as_deref_mut().ok_or(GpuError::state(
                WHAT_Q,
                "the unit's q8 blocks of 32 (a Q8_0, Q5_0 or NVFP4 site reads them)",
            ))?;
            k.g32(
                WHAT_Q,
                "the 32-value kernels q8 blocks of 32 are quantized by",
            )?
            .enqueue_quantize_gemm32(gpu.stream(), x, m, a, sink)?;
        }
        Ok(())
    }

    /// The forms beside the f32 rows `f32` a site picks from.
    pub fn input<'a>(&'a self, f32: &'a DeviceBuffer<f32>) -> WideIn<'a> {
        WideIn {
            q128: self.q128.as_deref(),
            q32: self.q32.as_deref(),
            f32: Some(f32),
        }
    }
}

/// `y = W · x` for site `src` of type `ty`, `rows` rows an expert, over the
/// slots `route` holds, `input` picking each slot's column (module doc). A
/// 32-value GEMM over the unit's one-expert table must find it filled for
/// exactly `m` slots, and an F32 site runs only there (one column a slot);
/// a form the site reads that `x` lacks is refused by name. `w` holds the
/// resident weights `src` may name. An NVFP4 site's GEMM reads the q8 blocks
/// of 32 ([`WideIn::q32`]), not the q8_1 blocks of 128 its gemv reads
/// ([`reads_at`]); [`WideActs::wants`] names them, and a caller quantizes them
/// for it as for a Q8_0 site.
#[allow(
    clippy::too_many_arguments,
    reason = "one site's kernels, weight, rows, inputs, table, column rule, width and output (rust-quality R8)"
)]
pub fn gemm(
    gpu: &Gpu,
    k: &WideKernels<'_>,
    site: (SiteTy, &Weights, SiteWeight<'_>),
    rows: usize,
    x: &WideIn<'_>,
    table: (&GemmRoute, GemmInput),
    m: usize,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    gemm_into(gpu, k, site, rows, x, table, m, y.into()).map(drop)
}

/// [`gemm`] into `y`'s cells. A BF16 `y` is stored by the dense MMA tile's
/// BF16 store alone (an MXFP8 site's W8A8 product and a BF16 site's product
/// over its held BF16 form), each cell the F32 product's rounded to nearest
/// even: false, with nothing launched, for a site whose product has no BF16
/// store. Always true for an F32 `y`.
#[allow(
    clippy::too_many_arguments,
    reason = "one site's kernels, weight, rows, inputs, table, column rule, width and output (rust-quality R8)"
)]
fn gemm_into(
    gpu: &Gpu,
    k: &WideKernels<'_>,
    (ty, w, src): (SiteTy, &Weights, SiteWeight<'_>),
    rows: usize,
    x: &WideIn<'_>,
    (route, input): (&GemmRoute, GemmInput),
    m: usize,
    y: ProductOut<'_>,
) -> Result<bool, GpuError> {
    const WHAT: &str = "site::gemm";
    // `y`'s F32 cells; a product with no BF16 store returns false, launching
    // nothing, for a BF16 `y`.
    macro_rules! f32_cells {
        ($y:expr) => {
            match $y {
                ProductOut::F32(y) => y,
                ProductOut::Bf16(_) => return Ok(false),
            }
        };
    }
    let stream = gpu.stream();
    let name = src.name();
    let missing = |form: &'static str| {
        GpuError::shape(
            WHAT,
            format!("{name} is {ty} and reads {form}, which the unit did not quantize"),
        )
    };
    let dense = || {
        if route.filled() == Some(m) && input == GemmInput::PerSlot {
            Ok(())
        } else {
            Err(GpuError::shape(
                WHAT,
                format!(
                    "{name} is {ty}: it runs on the unit's one-expert table filled for its {m} \
                     slots, one column a slot; the table is filled for {:?}",
                    route.filled()
                ),
            ))
        }
    };
    src.check(WHAT, ty)?;
    let launched = match ty {
        SiteTy::Nvfp4 => {
            let y = f32_cells!(y);
            dense()?;
            let DevWeight::Nvfp4 { qs, e, scales, .. } = w.resident(WHAT, name)? else {
                return Err(GpuError::tensor(WHAT, name, "an NVFP4 weight"));
            };
            k.g32(
                WHAT,
                "the 32-value kernels an NVFP4 site runs on (this load holds none)",
            )?
            .enqueue_gemm32(
                stream,
                Gemm32Args {
                    w: Gemm32Weight::Nvfp4Planes(Nvfp4Stack { qs, e, scales }),
                    rows_per_expert: rows,
                    act: x.q32.ok_or_else(|| missing("q8 blocks of 32"))?,
                    route,
                    input,
                    y,
                },
            )
        }
        SiteTy::Q3K | SiteTy::Q4K | SiteTy::Q5K | SiteTy::Q6K | SiteTy::Iq4Xs => {
            let y = f32_cells!(y);
            k.gemm.enqueue_gemm(
                stream,
                GemmArgs {
                    ty: gemm_weight(ty)
                        .ok_or(GpuError::state(WHAT, "the selected site's GEMM type"))?,
                    w: match src {
                        SiteWeight::Words { w, .. } => w,
                        _ => kq(w, WHAT, name)?,
                    },
                    rows_per_expert: rows,
                    act: x.q128.ok_or_else(|| missing("q8_1 blocks of 128"))?,
                    route,
                    input,
                    y,
                },
            )
        }
        SiteTy::Q8_0 | SiteTy::F32 | SiteTy::Mxfp8 | SiteTy::Fp8B128 | SiteTy::Bf16 => {
            dense()?;
            let (site, part) = src.rows(WHAT)?;
            let weight = w.resident(WHAT, site)?;
            let g32 = || {
                k.g32(
                    WHAT,
                    "the 32-value kernels a Q8_0, F32 or E4M3 site runs on (this load holds none)",
                )
            };
            let f32_rows = || x.f32.ok_or_else(|| missing("f32 rows"));
            return with_planes(WHAT, (ty, weight, site), part, |p| {
                match p {
                    Planes::Q8_0(qs, d) => g32()?.enqueue_gemm32(
                        stream,
                        Gemm32Args {
                            w: Gemm32Weight::Q8_0Plane { qs, d },
                            rows_per_expert: rows,
                            act: x.q32.ok_or_else(|| missing("q8 blocks of 32"))?,
                            route,
                            input,
                            y: f32_cells!(y),
                        },
                    ),
                    Planes::F32(t) => {
                        g32()?.enqueue_f32_tile(stream, t, f32_rows()?, m, f32_cells!(y))
                    }
                    Planes::E4m3(t) if gpu.q8f32().dense_mma => match (x.q32, f32_rows()?) {
                        (Some(act), rows) if crate::gemm::mxfp8_w8a8_reads(&t, act, rows, m) => {
                            g32()?.enqueue_mxfp8_w8a8(stream, t, rows, act, m, y)
                        }
                        (_, rows) => g32()?.enqueue_e4m3_mma(stream, t, rows, m, f32_cells!(y)),
                    },
                    Planes::E4m3(t) => {
                        g32()?.enqueue_e4m3_tile(stream, t, f32_rows()?, m, f32_cells!(y))
                    }
                    Planes::Bf16(t) => match (x.q32, f32_rows()?) {
                        (Some(act), rows)
                            if gpu.q8f32().dense_mma && act.holds(MmaForm::Bf16, rows, m) =>
                        {
                            g32()?.enqueue_bf16_mma_cols(stream, t, rows, act, m, y)
                        }
                        (_, rows) => {
                            gpu.q8f32()
                                .enqueue_bf16_prompt(stream, t, rows, m, f32_cells!(y))
                        }
                    },
                }
                .map(|()| true)
            });
        }
        SiteTy::Q5_0 => {
            let y = f32_cells!(y);
            dense()?;
            let Some(DevWeight::Q5_0 { w: packed, .. }) = w.get(name) else {
                return Err(GpuError::tensor(WHAT, name, "a packed Q5_0 weight"));
            };
            k.g32(
                WHAT,
                "the 32-value kernels a Q5_0 site runs on (this load holds none)",
            )?
            .enqueue_gemm32(
                stream,
                Gemm32Args {
                    w: Gemm32Weight::Q5_0Packed(packed),
                    rows_per_expert: rows,
                    act: x.q32.ok_or_else(|| missing("q8 blocks of 32"))?,
                    route,
                    input,
                    y,
                },
            )
        }
        // A backend's family, outside the card kernel table: no card path
        // reads a site of it.
        SiteTy::MlxA4 | SiteTy::MlxA8 | SiteTy::MlxA4h | SiteTy::MlxA8h => Err(GpuError::shape(
            WHAT,
            format!("{name} is {ty}: no card launch reads it"),
        )),
    };
    launched.map(|()| true)
}

/// [`gemm`] for the resident weight `name` at the site type of its resident
/// form ([`of_resident`]) on the unit's one-expert table `route`,
/// filled for the first `m` slots, token-major: a body whose plan picks each
/// dense site's type per tensor. A site reads the form its type reads from
/// `x` ([`resident_quantize`] writes the q8 blocks of 32).
pub fn resident_gemm(
    gpu: &Gpu,
    k: &WideKernels<'_>,
    (w, name): (&Weights, &str),
    x: &WideIn<'_>,
    route: &GemmRoute,
    m: usize,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let rows = w.resident("site::resident_gemm", name)?.rows();
    let src = SiteWeight::Resident(name);
    wide_by_name(gpu, k, (w, src, rows), x, route, m, y.into()).map(drop)
}

/// [`resident_gemm`] storing each output as BF16 bits, the F32 output rounded
/// to nearest even, where the site's product has a BF16 store
/// ([`gemm_into`]): false, with nothing launched, where it has none.
pub(crate) fn resident_gemm_bf16(
    gpu: &Gpu,
    k: &WideKernels<'_>,
    (w, name): (&Weights, &str),
    x: &WideIn<'_>,
    route: &GemmRoute,
    m: usize,
    y: &mut DeviceBuffer<u16>,
) -> Result<bool, GpuError> {
    let rows = w.resident("site::resident_gemm_bf16", name)?.rows();
    let src = SiteWeight::Resident(name);
    wide_by_name(gpu, k, (w, src, rows), x, route, m, y.into())
}

/// A resident gate projection followed by the shared sigmoid norm, its
/// output `tail.y` then quantized for the resident sites `readers` into
/// `out` ([`resident_quantize`]). BF16 and native W8A8 MXFP8 tensor-core
/// sites finish the norm in their product's store epilogue. The MXFP8
/// epilogue also writes the output's MXFP8 form, so it runs only when that
/// form is all the readers take from `out` ([`only_form`]).
#[allow(
    clippy::too_many_arguments,
    reason = "the projection's operands, its norm tail and the output's readers"
)]
pub(crate) fn resident_gemm_norm_gate(
    gpu: &Gpu,
    k: &WideKernels<'_>,
    (w, name): (&Weights, &str),
    x: &WideIn<'_>,
    route: &GemmRoute,
    m: usize,
    (norm, z, tail): (
        &crate::linear::norm_gate::NormGateKernels,
        &mut DeviceBuffer<f32>,
        crate::linear::norm_gate::NormGateTail<'_>,
    ),
    (readers, (q128, out)): (&[&str], (Option<&mut GemmAct>, &mut GemmAct32)),
) -> Result<(), GpuError> {
    use crate::linear::norm_gate::{NormGateArgs, NormGateTail};
    let what = "site::resident_gemm_norm_gate";
    let weight = w.resident(what, name)?;
    let rows = weight.rows();
    let NormGateTail {
        o,
        w: gain,
        eps,
        fault,
        y,
    } = tail;
    let tail = NormGateTail {
        o,
        w: gain,
        eps,
        fault,
        y: &mut *y,
    };
    let mxfp8 = match weight {
        DevWeight::Mxfp8 { .. } => Some(w.e4m3(what, name)?),
        _ => None,
    };
    match (weight, mxfp8, x.q32, x.f32) {
        (DevWeight::Bf16 { w: packed, .. }, _, Some(act), Some(input))
            if gpu.q8f32().dense_mma && act.holds(MmaForm::Bf16, input, m) =>
        {
            k.g32(what, "the BF16 prompt kernels")?
                .enqueue_bf16_mma_norm_gate(gpu.stream(), packed, input, act, m, z, tail)?;
        }
        (_, Some(planes), Some(act), Some(input))
            if gpu.q8f32().dense_mma
                && only_form(gpu, w, readers, out) == Some(MmaForm::Mxfp8)
                && crate::gemm::mxfp8_w8a8_reads(&planes, act, input, m) =>
        {
            return k
                .g32(what, "the MXFP8 prompt kernels")?
                .enqueue_mxfp8_w8a8_norm_gate(
                    gpu.stream(),
                    crate::gemm::Mxfp8NormGateArgs {
                        w: planes,
                        x: input,
                        act,
                        n_cols: m,
                        z,
                        tail,
                        out,
                    },
                );
        }
        _ => {
            tail.check_rows(what, rows, m, z.len())?;
            resident_gemm(gpu, k, (w, name), x, route, m, z)?;
            norm.enqueue_norm_gate_sigmoid(
                gpu.stream(),
                NormGateArgs {
                    o,
                    z,
                    w: gain,
                    eps,
                    n_v: rows / crate::linear::HEAD,
                    m,
                    fault,
                    y: tail.y,
                },
            )?;
        }
    }
    resident_quantize(gpu, k, (w, readers), (&*y, m), (q128, out), fault)
}

/// [`resident_gemm`] over rows `at .. at + n` of the resident weight `name`
/// alone ([`SiteWeight::Part`]), their `n` outputs of each slot token-major
/// into `y`, each bit for bit the whole weight's for its row and slot.
pub fn resident_gemm_rows(
    gpu: &Gpu,
    k: &WideKernels<'_>,
    (w, name, (at, n)): (&Weights, &str, (usize, usize)),
    x: &WideIn<'_>,
    route: &GemmRoute,
    m: usize,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let src = SiteWeight::Part { name, at, n };
    wide_by_name(gpu, k, (w, src, n), x, route, m, y.into()).map(drop)
}

/// [`gemm_into`] of site `src`, `rows` rows, at the site type of its
/// resident form, on the one-expert table `route` filled for `m` slots.
fn wide_by_name(
    gpu: &Gpu,
    k: &WideKernels<'_>,
    (w, src, rows): (&Weights, SiteWeight<'_>, usize),
    x: &WideIn<'_>,
    route: &GemmRoute,
    m: usize,
    y: ProductOut<'_>,
) -> Result<bool, GpuError> {
    let ty = resident_type(w, "site::resident_gemm", src.name())?;
    let table = (route, GemmInput::PerSlot);
    gemm_into(gpu, k, (ty, w, src), rows, x, table, m, y)
}

/// Whether a site of type `ty` copies its input's `form` on the dense MMA
/// tile when the input holds it ([`GemmAct32::with_form`]): a BF16 site the
/// BF16 form; on a native block-scaled build an MXFP8 site the MXFP8 form.
fn copies_form(gpu: &Gpu, ty: SiteTy, form: MmaForm) -> bool {
    let reader = match form {
        MmaForm::Bf16 => SiteTy::Bf16,
        MmaForm::Mxfp8 if cfg!(bloomery_nvfp4_mma) => SiteTy::Mxfp8,
        MmaForm::Mxfp8 => return false,
    };
    gpu.q8f32().dense_mma && ty == reader
}

/// Whether a site of one of the resident weights `names` copies `form`
/// ([`copies_form`] of its resident type).
#[must_use]
pub(crate) fn reads_form(gpu: &Gpu, w: &Weights, names: &[&str], form: MmaForm) -> bool {
    names.iter().any(|n| {
        w.get(n)
            .and_then(of_resident)
            .is_some_and(|ty| copies_form(gpu, ty, form))
    })
}

/// The forms of the first `m` columns of the f32 rows `x` that a site of one
/// of the resident weights `names` reads ([`WideActs::wants`] of its type,
/// [`of_resident`]): the q8_1 blocks of 128 into `q128` for a K-quant site,
/// the q8 blocks of 32 into `q32` for a Q8_0, Q5_0 or NVFP4 site, what
/// [`resident_gemm`] reads; and each [`MmaForm`] that `q32` holds room for and
/// a site reads ([`reads_form`], [`write_forms`]). A form no site reads is
/// left marked as another rows', so no later site reads it; q8_1 blocks a
/// site reads that `q128` does not hold are refused by name.
pub fn resident_quantize(
    gpu: &Gpu,
    k: &WideKernels<'_>,
    (w, names): (&Weights, &[&str]),
    (x, m): (&DeviceBuffer<f32>, usize),
    (q128, q32): (Option<&mut GemmAct>, &mut GemmAct32),
    sink: FaultSink,
) -> Result<(), GpuError> {
    if resident_wants(w, names).q128 {
        let a = q128.ok_or(GpuError::state(
            "site::resident_quantize",
            "the rows' q8_1 blocks of 128 (a K-quant site reads them)",
        ))?;
        gpu.enqueue_quantize_gemm(x, m, a, sink)?;
    }
    if resident_wants(w, names).q32 {
        k.g32(
            "site::resident_quantize",
            "the 32-value kernels a site's q8 blocks of 32 are quantized by",
        )?
        .enqueue_quantize_gemm32(gpu.stream(), x, m, q32, sink)?;
    }
    write_forms(k, gpu.stream(), (x, m), q32, |f| {
        reads_form(gpu, w, names, f)
    })
}

/// The one [`MmaForm`] the sites of the resident weights `readers` take
/// from `act`, when it is all they take: a reader copies it
/// ([`reads_form`]), and none reads a q8 form or another form `act` holds
/// room for. An epilogue that writes this form beside its f32
/// rows leaves the readers nothing to quantize.
#[must_use]
pub(crate) fn only_form(
    gpu: &Gpu,
    w: &Weights,
    readers: &[&str],
    act: &GemmAct32,
) -> Option<MmaForm> {
    let wants = resident_wants(w, readers);
    if wants.q128 || wants.q32 {
        return None;
    }
    let mut read = [MmaForm::Bf16, MmaForm::Mxfp8]
        .into_iter()
        .filter(|&f| act.has(f) && reads_form(gpu, w, readers, f));
    let form = read.next()?;
    read.next().is_none().then_some(form)
}

/// The wide forms the sites of the resident weights `names` read
/// ([`WideActs::wants`] of each resident type, [`of_resident`]).
#[must_use]
pub fn resident_wants(w: &Weights, names: &[&str]) -> WideForms {
    names
        .iter()
        .filter_map(|n| w.get(n).and_then(of_resident))
        .fold(WideForms::NONE, |acc, ty| acc.union(WideActs::wants(ty)))
}

/// The quantized forms of one wide input a body's resident dense sites read
/// ([`resident_quantize`]): the q8_1 blocks of 128 when a K-quant site reads
/// the rows, and the 32-value scratch, its q8 blocks of 32 held only when a
/// Q8_0, Q5_0 or NVFP4 site reads them and room for each [`MmaForm`] a site
/// copies ([`GemmAct32::with_form`]). The rule is the one a plan counts the
/// rows by (`placement::kernels::wide_act_bytes`).
pub struct DenseActs {
    pub q128: Option<GemmAct>,
    pub q32: GemmAct32,
}

impl DenseActs {
    /// `cols` columns of `k` values in the quantized forms of `wants` (its
    /// readers', [`resident_wants`]), with room for `forms`. Load-time only.
    pub fn new(
        stream: &cuda_core::CudaStream,
        wants: WideForms,
        (cols, k): (usize, usize),
        forms: &[MmaForm],
    ) -> Result<DenseActs, GpuError> {
        let q32 = forms.iter().try_fold(
            GemmAct32::with_planes(stream, cols, k, wants.q32)?,
            |act, &f| act.with_form(stream, f),
        )?;
        Ok(DenseActs {
            q128: wants
                .q128
                .then(|| GemmAct::new(stream, cols, k))
                .transpose()?,
            q32,
        })
    }

    pub fn bytes(&self) -> usize {
        self.q128.as_ref().map_or(0, GemmAct::bytes) + self.q32.bytes()
    }

    /// The forms beside the f32 rows `f32` a site picks from.
    pub fn input<'a>(&'a self, f32: &'a DeviceBuffer<f32>) -> WideIn<'a> {
        WideIn {
            q128: self.q128.as_ref(),
            q32: Some(&self.q32),
            f32: Some(f32),
        }
    }

    /// The quantized forms as [`resident_quantize`] writes them.
    pub fn out(&mut self) -> (Option<&mut GemmAct>, &mut GemmAct32) {
        (self.q128.as_mut(), &mut self.q32)
    }
}

/// Each form `q32` holds room for of the first `m` columns of the f32 rows
/// `x`: written (`enqueue_form_cols`) when a site reads it (`reads`), else
/// marked as another rows'.
fn write_forms(
    k: &WideKernels<'_>,
    stream: &cuda_core::CudaStream,
    (x, m): (&DeviceBuffer<f32>, usize),
    q32: &mut GemmAct32,
    reads: impl Fn(MmaForm) -> bool,
) -> Result<(), GpuError> {
    for form in [MmaForm::Bf16, MmaForm::Mxfp8] {
        if !q32.has(form) {
            continue;
        }
        if !reads(form) {
            q32.forget(form);
            continue;
        }
        k.g32(
            "site::write_forms",
            "the 32-value kernels a site's MMA form is written by",
        )?
        .enqueue_form_cols(stream, form, x, m, q32)?;
    }
    Ok(())
}

/// One gated block over a table: the three sites, their shapes, the gate·up
/// rule between them. The gate and up read the input rows through `input`;
/// the down reads the SwiGLU rows, slot `s` its own column `s`
/// ([`GemmInput::PerSlot`]).
pub struct Gated<'a> {
    pub w: &'a Weights,
    pub route: &'a GemmRoute,
    /// Gate, up and down: each site's type and weight.
    pub gate: (SiteTy, SiteWeight<'a>),
    pub up: (SiteTy, SiteWeight<'a>),
    pub down: (SiteTy, SiteWeight<'a>),
    /// Rows an expert of the gate and up: the SwiGLU's values a column.
    pub ff: usize,
    /// Rows an expert of the down.
    pub hidden: usize,
    pub rule: Act,
    pub input: GemmInput,
}

impl<'a> Gated<'a> {
    /// One of the block's sites as [`gemm`] takes it: type, weights, weight.
    fn at(&self, (ty, src): (SiteTy, SiteWeight<'a>)) -> (SiteTy, &'a Weights, SiteWeight<'a>) {
        (ty, self.w, src)
    }

    /// What the block refuses whatever the rows: words under a type with no
    /// grouped GEMM, and a rule the down's SwiGLU launch lacks
    /// ([`swiglu_launch`]) — checked before the first launch, so a block
    /// refused here has launched nothing.
    fn check(&self) -> Result<(), GpuError> {
        for (ty, src) in [self.gate, self.up, self.down] {
            src.check("site::gated", ty)?;
        }
        swiglu_launch((self.down.0, self.down.1.name()), self.rule).map(|_| ())
    }
}

/// Where a [`gated`] block's SwiGLU writes its rows: the form the down's type
/// reads on the wide arm ([`reads_wide`]). A form the down does not read may
/// be absent; a form it reads that is absent is refused by name.
///
/// A unit that holds a [`WideActs`] takes them from it
/// ([`WideActs::swiglu_rows`]), which holds every form its downs read. A
/// caller whose buffers are cut by width, outside a `WideActs`, states each
/// form it holds ([`SwigluRows::held`]).
pub struct SwigluRows<'a> {
    q128: Option<&'a mut GemmAct>,
    q32: Option<&'a mut GemmAct32>,
    f32: Option<&'a mut DeviceBuffer<f32>>,
}

impl<'a> SwigluRows<'a> {
    /// The rows over the buffers a caller holds outside a [`WideActs`], one
    /// argument a form: a form left `None` is one the block's down must not
    /// read.
    #[must_use]
    pub fn held(
        q128: Option<&'a mut GemmAct>,
        q32: Option<&'a mut GemmAct32>,
        f32: Option<&'a mut DeviceBuffer<f32>>,
    ) -> SwigluRows<'a> {
        SwigluRows { q128, q32, f32 }
    }
}

/// A [`gated`] block's rows: the input in the forms the gate and up read,
/// the gate and up outputs, the SwiGLU's rows and the down's output.
pub struct GatedIo<'a> {
    pub x: &'a WideIn<'a>,
    pub gate: &'a mut DeviceBuffer<f32>,
    pub up: &'a mut DeviceBuffer<f32>,
    pub rows: SwigluRows<'a>,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// The gated block `blk` over `m` input columns: the gate and up GEMMs, the
/// rule's SwiGLU over the `n_slots` slot columns they wrote, and the down GEMM
/// over those `n_slots` columns into `io.y`. On a one-expert table
/// `n_slots == m`; on an expert table it is the slots of the `m` columns.
/// The launches are those of the sites' types ([`gemm`] and the SwiGLU's
/// quantizer into the down's form); a form a site reads that `io` lacks is
/// refused by name, and words a site cannot launch ([`SiteWeight`]) before
/// the first launch.
pub fn gated(
    gpu: &Gpu,
    k: &WideKernels<'_>,
    blk: &Gated<'_>,
    (m, n_slots): (usize, usize),
    io: GatedIo<'_>,
    fault: FaultSink,
) -> Result<(), GpuError> {
    blk.check()?;
    let GatedIo {
        x,
        gate,
        up,
        mut rows,
        y,
    } = io;
    let table = (blk.route, blk.input);
    gemm(gpu, k, blk.at(blk.gate), blk.ff, x, table, m, gate)?;
    gemm(gpu, k, blk.at(blk.up), blk.ff, x, table, m, up)?;
    swiglu(gpu, k, blk, (&*gate, &*up), n_slots, &mut rows, fault)?;
    let h = WideIn {
        q128: rows.q128.as_deref(),
        q32: rows.q32.as_deref(),
        f32: rows.f32.as_deref(),
    };
    gemm(
        gpu,
        k,
        blk.at(blk.down),
        blk.hidden,
        &h,
        (blk.route, GemmInput::PerSlot),
        n_slots,
        y,
    )
}

/// The launch a gated block's SwiGLU makes into the form its down's type
/// reads, under the gate·up rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SwigluLaunch {
    /// `GemmKernels::enqueue_swiglu_act_quant` into q8_1 blocks of 128,
    /// under any rule (`SiluMul` is the launch `enqueue_swiglu_quant`
    /// makes).
    Q128,
    /// `Gemm32Kernels::enqueue_swiglu_quant32` into q8 blocks of 32 under
    /// `SiluMul`: its entry has no rule switch.
    Q32Plain,
    /// `Gemm32Kernels::enqueue_swiglu_act_quant32` into q8 blocks of 32
    /// under any other rule.
    Q32Rule,
    /// `ElemKernels::enqueue_swiglu` into f32 rows under `SiluMul`.
    F32,
    /// `ElemKernels::enqueue_swiglu_rule` into f32 rows under any other
    /// rule.
    F32Rule,
}

/// The launch of a SwiGLU into the form down site `(ty, name)` reads on the
/// wide arm ([`reads_wide`]) under `rule`; a rule the form's launch lacks is
/// refused by name.
fn swiglu_launch((ty, name): (SiteTy, &str), rule: Act) -> Result<SwigluLaunch, GpuError> {
    match (reads_wide(ty), rule) {
        (
            Form::Q8x128,
            Act::SiluMul | Act::SwigluClamp { .. } | Act::SwigluOai { .. } | Act::Situ { .. },
        ) => Ok(SwigluLaunch::Q128),
        // `SiluMul` keeps the plain entry, which carries no rule switch; the
        // rule's entry writes the same bytes under it.
        (Form::Q8x32, Act::SiluMul) => Ok(SwigluLaunch::Q32Plain),
        (Form::Q8x32, Act::SwigluClamp { .. } | Act::SwigluOai { .. } | Act::Situ { .. }) => {
            Ok(SwigluLaunch::Q32Rule)
        }
        // `SiluMul` keeps the plain entry; the rule's entry writes the same
        // values under it.
        (Form::F32, Act::SiluMul) => Ok(SwigluLaunch::F32),
        (Form::F32, Act::SwigluClamp { .. } | Act::SwigluOai { .. } | Act::Situ { .. }) => {
            Ok(SwigluLaunch::F32Rule)
        }
        (Form::Q8Blocks32, _) => Err(GpuError::shape(
            "site::swiglu",
            format!(
                "{name} is {ty}: its SwiGLU rows are Q8Blocks32, which the wide GEMM writes none of"
            ),
        )),
    }
}

/// The rule's SwiGLU of the first `cols` columns of the gate and up rows
/// (`blk.ff` values each) into `rows`, in the form `blk`'s down reads: one
/// launch with the quantizer for a q8 form, the f32 rows otherwise, then the
/// [`MmaForm`] of them the down copies when `rows.q32` holds room for it.
fn swiglu(
    gpu: &Gpu,
    k: &WideKernels<'_>,
    blk: &Gated<'_>,
    (g, u): (&DeviceBuffer<f32>, &DeviceBuffer<f32>),
    cols: usize,
    rows: &mut SwigluRows<'_>,
    fault: FaultSink,
) -> Result<(), GpuError> {
    const WHAT: &str = "site::swiglu";
    let (ty, src) = blk.down;
    let name = src.name();
    let stream = gpu.stream();
    let lacks = |form: &'static str| {
        GpuError::shape(
            WHAT,
            format!("{name} is {ty} and reads {form}, which the SwiGLU rows lack"),
        )
    };
    match swiglu_launch((ty, name), blk.rule)? {
        SwigluLaunch::Q128 => {
            let a = rows
                .q128
                .as_deref_mut()
                .ok_or_else(|| lacks("q8_1 blocks of 128"))?;
            k.gemm
                .enqueue_swiglu_act_quant(stream, g, u, blk.rule, cols, a, fault)
        }
        SwigluLaunch::Q32Plain => {
            let a = rows
                .q32
                .as_deref_mut()
                .ok_or_else(|| lacks("q8 blocks of 32"))?;
            k.g32(
                WHAT,
                "the 32-value kernels a SwiGLU into q8 blocks of 32 runs on (this load holds none)",
            )?
            .enqueue_swiglu_quant32(stream, g, u, cols, a, fault)
        }
        SwigluLaunch::Q32Rule => {
            let a = rows
                .q32
                .as_deref_mut()
                .ok_or_else(|| lacks("q8 blocks of 32"))?;
            k.g32(
                WHAT,
                "the 32-value kernels a SwiGLU into q8 blocks of 32 runs on (this load holds none)",
            )?
            .enqueue_swiglu_act_quant32(stream, g, u, blk.rule, cols, a, fault)
        }
        SwigluLaunch::F32 => {
            let y = rows.f32.as_deref_mut().ok_or_else(|| lacks("f32 rows"))?;
            gpu.elem().enqueue_swiglu(stream, g, u, cols * blk.ff, y)?;
            match rows.q32.as_deref_mut() {
                Some(q32) => write_forms(k, stream, (&*y, cols), q32, |f| copies_form(gpu, ty, f)),
                None => Ok(()),
            }
        }
        SwigluLaunch::F32Rule => {
            let y = rows.f32.as_deref_mut().ok_or_else(|| lacks("f32 rows"))?;
            gpu.elem()
                .enqueue_swiglu_rule(stream, blk.rule, (g, u), cols * blk.ff, (fault, y))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Form, Order, SiteTy, SwigluLaunch, WideActs, WideForms, by_name_extra, file_site,
        gemm_weight, kgemv_order, reads_at, reads_wide, swiglu_launch,
    };
    use crate::gemm::GemmWeight;
    use crate::kquant::Act;
    use gguf::Split;
    use gguf::quant::GgmlType;
    use gguf::write::{Layout, TensorDecl};
    use model::placement::kernels;

    /// A site's activation form is the kernel table's for its type at any width, except NVFP4,
    /// whose gemv reads whole q8_1 super-blocks of 256 values: at a width they do not divide (the
    /// 2688 and 3712 of a hidden and a shared width of 64 blocks) it reads the 32-value rows its
    /// one-expert `_sel` gemv takes.
    ///
    /// FAIL-first: `SiteEntry::reads_at` answering `reads()` for every width reads NVFP4 at 2688
    /// as the q8_1 rows.
    #[test]
    fn an_nvfp4_site_reads_the_form_its_width_takes() {
        for &ty in SiteTy::ALL {
            for k in [64usize, 256, 2688, 3712, 4096] {
                let want = if ty == SiteTy::Nvfp4 && !k.is_multiple_of(256) {
                    Form::Q8Blocks32
                } else {
                    ty.reads()
                };
                assert_eq!(reads_at(ty, k), want, "{ty} at {k}");
            }
        }
        assert_eq!(reads_at(SiteTy::Nvfp4, 2688), Form::Q8Blocks32);
        assert_eq!(reads_at(SiteTy::Nvfp4, 4096), Form::Q8x128);
    }

    /// The wide arm's form is the type's own except NVFP4's: its GEMM reads the q8 blocks of 32
    /// where its gemv reads the q8_1 rows, so a unit that holds an NVFP4 site holds the q8 blocks
    /// of 32 and no q8_1 blocks of 128 for it, at any width of its 64-value blocks.
    ///
    /// FAIL-first: `reads_wide` answering `ty.reads()` for NVFP4 wants the q8_1 blocks of 128 and
    /// none of 32.
    #[test]
    fn an_nvfp4_site_wants_the_q32_rows_on_the_wide_arm() {
        for &ty in SiteTy::ALL {
            let want = if ty == SiteTy::Nvfp4 {
                Form::Q8x32
            } else {
                ty.reads()
            };
            assert_eq!(reads_wide(ty), want, "{ty}");
        }
        assert_eq!(WideActs::wants(SiteTy::Nvfp4), WideForms::q8(false, true));
        assert_eq!(
            WideActs::wants_of(&[SiteTy::Nvfp4, SiteTy::Nvfp4, SiteTy::Nvfp4]),
            WideForms::q8(false, true)
        );
        // A K-quant gate and up beside an NVFP4 down hold both quantized forms.
        assert_eq!(
            WideActs::wants_of(&[SiteTy::Q4K, SiteTy::Nvfp4]),
            WideForms::q8(true, true)
        );
    }

    /// A by-name projection's extra launches follow the type: none for a site
    /// that reads the f32 rows or the q8 blocks of 32, the quantization for a
    /// K-quant, IQ4_XS or NVFP4 site, and the token-major copy past one
    /// column for the row-major types alone (Q3_K, Q4_K, Q6_K).
    #[test]
    fn a_by_name_projection_counts_its_quantization_and_its_row_major_copy() {
        for (ty, one, rows) in [
            (SiteTy::Q8_0, 0, 0),
            (SiteTy::F32, 0, 0),
            (SiteTy::Bf16, 0, 0),
            (SiteTy::Mxfp8, 0, 0),
            (SiteTy::Fp8B128, 0, 0),
            (SiteTy::Q3K, 1, 2),
            (SiteTy::Q4K, 1, 2),
            (SiteTy::Q6K, 1, 2),
            (SiteTy::Q5K, 1, 1),
            (SiteTy::Iq4Xs, 1, 1),
            (SiteTy::Nvfp4, 1, 1),
        ] {
            assert_eq!(by_name_extra(ty, 1), one, "{ty} at one column");
            assert_eq!(by_name_extra(ty, 4), rows, "{ty} at four columns");
        }
    }

    /// Every family is the kernel table's dense row of one file type: a type
    /// has a family exactly where the table names a dense entry for it (F16,
    /// which loads widened to f32, is the f32 site of its widened rows), the
    /// family maps back to that type and is named by it in upper case, its
    /// grouped GEMM is the one the table names for the type, it is a
    /// `kgemv` exactly where it reads the q8_1 rows of 128 values, it
    /// launches by name (`BY_NAME`) exactly where it reads the f32 rows or
    /// the q8 blocks of 32, and by projection (`PROJECTED`) exactly where the
    /// table names a dense entry but Q5_0's.
    #[test]
    fn every_family_is_the_tables_dense_row_of_one_type() {
        // PIN(2026-10-21): consolidate the three site pins, preserving their assertions.
        let own = [
            GgmlType::F8E4M3_ROWS,
            GgmlType::MXFP8,
            GgmlType::NVFP4_ROWS,
            GgmlType::FP8_B128,
        ];
        for ty in gguf::QROWS.iter().map(|r| r.ty).chain(own) {
            let family = SiteTy::of_ggml(ty);
            if ty == GgmlType::F16 {
                assert_eq!(
                    family,
                    Some(SiteTy::F32),
                    "{ty}: the f32 site of its widened rows"
                );
                assert_eq!(kernels::dense(ty), None, "{ty}");
                continue;
            }
            assert_eq!(family.is_some(), kernels::dense(ty).is_some(), "{ty}");
            if let Some(s) = family {
                assert_eq!(s.ggml(), ty, "{s}");
            }
        }
        let expected = [
            (SiteTy::Q3K, GgmlType::Q3_K, "Q3_K", Some(GemmWeight::Q3K)),
            (SiteTy::Q4K, GgmlType::Q4_K, "Q4_K", Some(GemmWeight::Q4K)),
            (SiteTy::Q5K, GgmlType::Q5_K, "Q5_K", Some(GemmWeight::Q5K)),
            (SiteTy::Q6K, GgmlType::Q6_K, "Q6_K", Some(GemmWeight::Q6K)),
            (SiteTy::Q5_0, GgmlType::Q5_0, "Q5_0", None),
            (SiteTy::Q8_0, GgmlType::Q8_0, "Q8_0", None),
            (SiteTy::F32, GgmlType::F32, "F32", None),
            (SiteTy::Nvfp4, GgmlType::NVFP4, "NVFP4", None),
            (SiteTy::Mxfp8, GgmlType::MXFP8, "MXFP8", None),
            (SiteTy::Fp8B128, GgmlType::FP8_B128, "FP8_B128", None),
            (SiteTy::Bf16, GgmlType::BF16, "BF16", None),
            (
                SiteTy::Iq4Xs,
                GgmlType::IQ4_XS,
                "IQ4_XS",
                Some(GemmWeight::Iq4Xs),
            ),
        ];
        assert_eq!(SiteTy::ALL.len(), expected.len());
        for (&s, (want, ty, label, grouped)) in SiteTy::ALL.iter().zip(expected) {
            assert_eq!(s, want);
            assert_eq!(s.ggml(), ty);
            assert_eq!(s.to_string(), label);
            assert_eq!(gemm_weight(s), grouped);
            assert_eq!(kgemv_order(s).is_some(), s.reads() == Form::Q8x128);
            let by_name = model::placement::accepts::dense_by_name(ty);
            assert_eq!(super::BY_NAME.contains(&s), by_name.is_some(), "{s}");
            if let Some(entry) = by_name {
                assert_eq!(SiteTy::of_entry(entry), Some(s), "{s}");
            }
            let projected = model::placement::accepts::dense_projected(ty);
            assert_eq!(super::PROJECTED.contains(&s), projected.is_some(), "{s}");
            if let Some(entry) = projected {
                assert_eq!(SiteTy::of_entry(entry), Some(s), "{s}");
            }
        }
        assert_eq!(SiteTy::Q5_0.reads(), Form::Q8Blocks32);
        assert_eq!(kgemv_order(SiteTy::Q5_0), None);
        assert_eq!(SiteTy::Nvfp4.reads(), Form::Q8x128);
        assert_eq!(kgemv_order(SiteTy::Nvfp4), Some(Order::TokenMajor));
        assert_eq!(SiteTy::Mxfp8.reads(), Form::F32);
        assert_eq!(kgemv_order(SiteTy::Iq4Xs), Some(Order::TokenMajor));
    }

    /// A site's type is read from the file's header alone: a Q8_0 matrix of
    /// two rows of 64 is read as Q8_0 where Q8_0 is launched, and refused by
    /// name where only K-quants are; a Q3_K, a Q5_K, a Q5_0, an IQ4_XS, an MXFP8
    /// (bloomery's id, sized by the split) and a BF16 one are read as theirs; an
    /// IQ3_S one (a type no site launch reads), a matrix of another shape and a
    /// type outside the allowed list are refused by name.
    ///
    /// PIN(2026-10-27): the refused type with no launch was IQ4_XS; it is a site family now (`SiteTy::Iq4Xs`), so IQ3_S stands in and IQ4_XS joins the read types.
    #[test]
    fn a_site_type_comes_from_the_header_and_a_type_with_no_launch_is_refused() {
        let decl = |name: &str, ty: GgmlType, bytes| TensorDecl {
            name: name.to_owned(),
            dims: vec![256, 2],
            type_id: ty.as_u32(),
            nbytes: bytes,
        };
        let layout = Layout::new(
            &[],
            vec![
                decl("shared", GgmlType::IQ4_XS, 2 * 136),
                TensorDecl {
                    name: "q8".to_owned(),
                    dims: vec![64, 2],
                    type_id: GgmlType::Q8_0.as_u32(),
                    nbytes: 2 * 2 * 34,
                },
                decl("q3", GgmlType::Q3_K, 2 * 110),
                decl("q5", GgmlType::Q5_K, 2 * 176),
                decl("q50", GgmlType::Q5_0, 2 * 8 * 22),
                decl("mx", GgmlType::MXFP8, 2 * 8 * 33),
                decl("bf", GgmlType::BF16, 2 * 256 * 2),
                decl("unsupported", GgmlType::IQ3_S, 2 * 110),
            ],
        )
        .unwrap();
        let file = Split::header_only("sites.gguf".into(), &layout).unwrap();
        for (name, k, want) in [
            ("q8", 64, SiteTy::Q8_0),
            ("q3", 256, SiteTy::Q3K),
            ("q5", 256, SiteTy::Q5K),
            ("q50", 256, SiteTy::Q5_0),
            ("mx", 256, SiteTy::Mxfp8),
            ("bf", 256, SiteTy::Bf16),
        ] {
            assert_eq!(
                file_site(&file, "test", name, (2, k), SiteTy::ALL).ok(),
                Some(want),
                "{name}"
            );
        }
        assert_eq!(
            file_site(&file, "test", "shared", (2, 256), &super::PROJECTED).ok(),
            Some(SiteTy::Iq4Xs)
        );
        for (name, shape, allowed, want) in [
            (
                "q8",
                (2, 64),
                &[SiteTy::Q4K, SiteTy::Q6K][..],
                "q8 is q8_0 in the file",
            ),
            (
                "shared",
                (2, 256),
                &[SiteTy::Q4K][..],
                "shared is iq4_xs in the file",
            ),
            (
                "unsupported",
                (2, 256),
                SiteTy::ALL,
                "unsupported is iq3_s in the file",
            ),
            ("q8", (4, 64), SiteTy::ALL, "it takes 4 rows of 64"),
            ("shared", (1, 256), SiteTy::ALL, "it takes 1 rows of 256"),
            ("shared", (2, 512), SiteTy::ALL, "it takes 2 rows of 512"),
        ] {
            let Err(chain::ChainError::Shape { what, detail }) =
                file_site(&file, "test", name, shape, allowed)
            else {
                panic!("{name} {shape:?}: not refused as a shape error");
            };
            assert_eq!(what, "test");
            assert!(detail.contains(want), "{name}: {detail}");
        }
    }

    /// The SwiGLU's launch is the down's form under the rule: every K-quant
    /// one launch into q8_1 blocks of 128, Q8_0 and NVFP4 (whose wide GEMM
    /// reads the q8 blocks of 32) the plain 32-value entry under `SiluMul`
    /// (the rule's entry for the clamp and the OAI form), F32, BF16 and MXFP8
    /// the plain f32 rows under `SiluMul` and the rule's f32 rows under the
    /// clamp, the OAI form and SiTU. Mutants: the f32 rule arm dropped (the
    /// rule refused on an f32 down, as before the MXFP8 blocks); the form
    /// read as `ty.reads()`, which sends an NVFP4 down to the q8_1 rows.
    #[test]
    fn the_swiglu_launch_is_the_downs_form_under_the_rule() {
        let clamp = Act::SwigluClamp { limit: 7.0 };
        let oai = Act::SwigluOai {
            alpha: crate::kquant::act::SWIGLU_OAI_ALPHA,
            limit: 7.0,
        };
        let launch = |ty, rule| swiglu_launch((ty, "down"), rule);
        for ty in [
            SiteTy::Q3K,
            SiteTy::Q4K,
            SiteTy::Q5K,
            SiteTy::Q6K,
            SiteTy::Iq4Xs,
        ] {
            for rule in [Act::SiluMul, clamp, oai] {
                assert_eq!(launch(ty, rule).ok(), Some(SwigluLaunch::Q128), "{ty}");
            }
        }
        assert_eq!(
            launch(SiteTy::Q8_0, Act::SiluMul).ok(),
            Some(SwigluLaunch::Q32Plain)
        );
        for rule in [clamp, oai] {
            assert_eq!(launch(SiteTy::Q8_0, rule).ok(), Some(SwigluLaunch::Q32Rule));
        }
        assert_eq!(
            launch(SiteTy::Nvfp4, Act::SiluMul).ok(),
            Some(SwigluLaunch::Q32Plain)
        );
        for rule in [clamp, oai] {
            assert_eq!(
                launch(SiteTy::Nvfp4, rule).ok(),
                Some(SwigluLaunch::Q32Rule)
            );
        }
        let situ = Act::Situ {
            beta: 1.0,
            linear_beta: crate::kquant::act::SITU_LINEAR_BETA,
        };
        for ty in [SiteTy::F32, SiteTy::Bf16, SiteTy::Mxfp8] {
            assert_eq!(
                launch(ty, Act::SiluMul).ok(),
                Some(SwigluLaunch::F32),
                "{ty}"
            );
            for rule in [clamp, oai, situ] {
                assert_eq!(
                    launch(ty, rule).ok(),
                    Some(SwigluLaunch::F32Rule),
                    "{ty} {rule:?}"
                );
            }
        }
        for rule in [Act::SiluMul, clamp, oai] {
            match launch(SiteTy::Q5_0, rule) {
                Err(e) => {
                    let msg = e.to_string();
                    assert!(
                        msg.contains("down is Q5_0") && msg.contains("Q8Blocks32"),
                        "{msg}"
                    );
                }
                Ok(l) => panic!("a Q5_0 down launched {l:?}"),
            }
        }
    }

    #[test]
    fn a_bf16_site_reads_the_f32_rows_and_is_a_gemv() {
        assert_eq!(SiteTy::of_ggml(GgmlType::BF16), Some(SiteTy::Bf16));
        assert!(SiteTy::ALL.contains(&SiteTy::Bf16));
        assert_eq!(SiteTy::Bf16.reads(), Form::F32);
        assert_eq!(kgemv_order(SiteTy::Bf16), None);
        assert_eq!(gemm_weight(SiteTy::Bf16), None);
        assert_eq!(
            WideActs::wants(SiteTy::Bf16),
            WideForms {
                q128: false,
                q32: false,
                f32: true,
            }
        );
        assert_eq!(SiteTy::Bf16.to_string(), "BF16");
    }

    /// A unit holds the forms its sites read, f32 rows included: the SwiGLU
    /// rows of a down that reads f32 (MXFP8, BF16, F32, FP8_B128) are held by
    /// the `WideActs` that `wants_of` the down sizes, and counted in its bytes,
    /// so a caller never supplies them ad hoc. An input's f32 rows are the
    /// caller's (`without_f32`). Mutant: `wants` ignoring `Form::F32`, which
    /// puts the MXFP8, BF16 and F32 rows below out of their forms.
    #[test]
    fn a_down_that_reads_f32_wants_f32_rows_in_the_units_forms_and_bytes() {
        let q8 = WideForms::q8;
        let f32_rows = WideForms {
            q128: false,
            q32: false,
            f32: true,
        };
        for &ty in SiteTy::ALL {
            let w = WideActs::wants(ty);
            assert_eq!(w.f32, reads_wide(ty) == Form::F32, "{ty}");
            assert_eq!(w.q128, reads_wide(ty) == Form::Q8x128, "{ty}");
        }
        for ty in [SiteTy::Mxfp8, SiteTy::Bf16, SiteTy::F32] {
            assert_eq!(WideActs::wants_of(&[ty]), f32_rows, "{ty}");
            assert_eq!(WideActs::wants_of(&[ty]).without_f32(), WideForms::NONE);
        }
        // A mixed set holds the union: a K-quant gate·up beside an MXFP8 down.
        assert_eq!(
            WideActs::wants_of(&[SiteTy::Q4K, SiteTy::Mxfp8, SiteTy::Q8_0]),
            WideForms {
                q128: true,
                q32: true,
                f32: true,
            }
        );
        assert_eq!(WideActs::wants_of(&[]), WideForms::NONE);
        assert_eq!(
            WideActs::wants_of(&[SiteTy::Q4K, SiteTy::Q6K]),
            q8(true, false)
        );
        // The f32 rows are cols · k values: the bytes `WideActs::new` allocates.
        let (cols, k) = (3, 256);
        assert_eq!(f32_rows.bytes(cols, k), cols * k * 4);
        assert_eq!(q8(false, false).bytes(cols, k), 0);
        assert_eq!(
            f32_rows.union(q8(true, false)).bytes(cols, k),
            cols * k * 4 + q8(true, false).bytes(cols, k)
        );
    }
}
