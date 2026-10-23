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
//!   A K-quant site's gemv is [`kgemv`]. A Q5_0 site's is [`blocks32_gemv`].
//! - [`kgemv`], a site at most eight columns over the q8_1 rows: Q3_K, Q4_K
//!   and Q6_K their gemvs, row-major (`m` outputs a row; a Q6_K row of an odd
//!   super-block count the down `_sel` over one expert, one column); Q5_K
//!   the K-quant down `_sel` over a stack of one expert, token-major (a slot
//!   a column); NVFP4 `nvfp4_gemv` over its planes times the tensor's scale,
//!   token-major. [`kgemv_order`] is this family's layout and is
//!   `None` for Q5_0. Where a body copies a row-major output token-major,
//!   and its fused groups of Q4_K launches, stay the body's.
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
//!   On GB10, BF16 and the E4M3 formats instead use BF16 tensor-core operands
//!   with F32 accumulation (`bf16_mma_gemm`, `mxfp8_mma_gemm`, `fp8_b128_mma_gemm`);
//!   exceptional operands replay the exact F32 tile within that kernel.
//!   A site's weight is the resident tensor of its name, rows of it (a part
//!   of a weight joined at load), or, for a K-quant, the words a caller
//!   holds ([`SiteWeight`]). NVFP4 has no wide GEMM and is refused here by
//!   name.
//! - [`gated`], a wide gate·up·down block over one table: the gate and up
//!   [`gemm`]s, the gate·up rule's SwiGLU written into the form the down's
//!   type reads (the down's activation), and the down [`gemm`] over those
//!   rows, one column a slot.
//!
//! Which activation form a site reads is [`SiteTy::reads`]: a body
//! quantizes a unit's rows into exactly the forms its sites read.
//!
//! A body that names a weight and leaves its type to the file calls a site
//! by name: [`resident_gemv`] at one column, [`resident_gemv_cols`] at up to
//! eight, [`resident_gemv_rows`] over rows of a joined weight, and on the
//! wide arm [`resident_gemm`] and [`resident_gemm_rows`]. Each launches
//! what the typed call launches for the site type of the weight's resident
//! form, which the kernel table's dense entry decides
//! ([`of_resident`]); [`resident_site`] checks that type and shape
//! at load.

use crate::gemm::{
    Gemm32Args, Gemm32Kernels, Gemm32Weight, GemmAct, GemmAct32, GemmArgs, GemmInput, GemmKernels,
    GemmRoute, GemmWeight,
};
use crate::kquant::{Act, SelDown};
use crate::mxfp4::Nvfp4Kernels;
use crate::q5::Q8Blocks32;
use crate::q6k_sel::Q6kSelKernels;
use crate::q8f32::{
    Bf16GemvMcolArgs, E4m3GemvArgs, E4m3Scales, E4m3Weight, GemvOut, Q8_0GemvMcolArgs, SelArgs,
};
use crate::quant_sel::QuantSelKernels;
use crate::tensor::{Q8Act, TensorWindow};
use crate::weights::{DevWeight, Weights};
use crate::{DeviceTensor, FaultSink, Gpu, GpuError};
use chain::site::list;
use cuda_core::{DeviceBuffer, DeviceCopy};
use gguf::quant::GgmlType;
use model::placement::kernels;

pub use chain::site::{SiteTy, file_site};

/// The activation form a site's launch reads: the kernel table's fact.
pub use model::placement::kernels::SiteAct as Form;

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
/// layout (Q8_0's planes, an i-quant), a derived weight and an NVFP4 stack.
/// The resident form is this crate's, so the map is here, beside the
/// launches that read it, and not on the chain's [`SiteTy`].
#[must_use]
pub fn of_resident(w: &DevWeight) -> Option<SiteTy> {
    let ty = match w {
        DevWeight::KQuant { ty, .. } => {
            return SiteTy::of_ggml(*ty).filter(|s| s.kquant());
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

/// How [`kgemv`] lays a site of type `ty` out: Q3_K, Q4_K and Q6_K row-major, Q5_K and NVFP4 token-major.
/// `None` for a type [`kgemv`] does not run: Q8_0, F32, MXFP8, FP8_B128 and BF16 ([`gemv`]), and Q5_0
/// ([`blocks32_gemv`], whose row-major layout `y[r · m + c]` is that function's own).
#[must_use]
pub fn kgemv_order(ty: SiteTy) -> Option<Order> {
    match ty {
        SiteTy::Q3K | SiteTy::Q4K | SiteTy::Q6K => Some(Order::RowMajor),
        SiteTy::Q5K | SiteTy::Nvfp4 => Some(Order::TokenMajor),
        SiteTy::Q5_0
        | SiteTy::Q8_0
        | SiteTy::F32
        | SiteTy::Mxfp8
        | SiteTy::Fp8B128
        | SiteTy::Bf16 => None,
    }
}

/// The grouped GEMM's type of a K-quant, the one the kernel table's GEMM entry for the file type names
/// (`kernels::gemm`); `None` for Q5_0, Q8_0, F32, MXFP8, FP8_B128 and BF16, whose wide GEMM is the 32-value
/// family's ([`gemm`]), and NVFP4, which has none.
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
    let src = (Some(ty), w, SiteWeight::Resident(name));
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
pub(crate) fn resident_gemv_cols_on(
    gpu: &Gpu,
    stream: &cuda_core::CudaStream,
    g32: Option<&Gemm32Kernels>,
    (w, name): (&Weights, &str),
    x: &DeviceBuffer<f32>,
    m: usize,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    launch(
        gpu,
        stream,
        g32,
        (None, w, SiteWeight::Resident(name)),
        x,
        m,
        y,
    )
}

/// [`resident_gemv_cols`] over rows `at .. at + n` of the resident weight
/// `name` alone ([`SiteWeight::Part`]), their `n` outputs of each column
/// token-major into `y`, each bit for bit the whole weight's for its row
/// and column.
pub fn resident_gemv_rows(
    gpu: &Gpu,
    g32: Option<&Gemm32Kernels>,
    (w, name, (at, n)): (&Weights, &str, (usize, usize)),
    x: &DeviceBuffer<f32>,
    m: usize,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    launch(
        gpu,
        gpu.stream(),
        g32,
        (None, w, SiteWeight::Part { name, at, n }),
        x,
        m,
        y,
    )
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

/// The gemv of site `src` over the first `m` (1..=8) columns of the f32 rows
/// `x`, token-major, at the site type of its resident form, which must be
/// `declared` when the caller names one ([`resident_ty`]). `g32` is read
/// only by an F32 site past one column.
fn launch(
    gpu: &Gpu,
    stream: &cuda_core::CudaStream,
    g32: Option<&Gemm32Kernels>,
    (declared, w, src): (Option<SiteTy>, &Weights, SiteWeight<'_>),
    x: &DeviceBuffer<f32>,
    m: usize,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    const WHAT: &str = "site::gemv";
    let (name, part) = src.rows(WHAT)?;
    let weight = w.resident(WHAT, name)?;
    let ty = resident_ty(WHAT, (declared, weight, name))?;
    match ty.reads() {
        Form::F32 | Form::Q8x32 => with_planes(WHAT, (ty, weight, name), part, |p| {
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
/// super-block count runs the same way, NVFP4's gemv, and that one expert's
/// id for each of [`KGEMV_COLS`] slots.
pub struct KGemvKernels {
    kq: QuantSelKernels,
    q6: Q6kSelKernels,
    nv: Nvfp4Kernels,
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
            one: DeviceBuffer::zeroed(gpu.stream(), KGEMV_COLS)?,
        })
    }
}

/// `y = W · act` for site `name` of type `ty` over the `act.m()` columns of
/// the q8_1 rows `act`, laid out as [`kgemv_order`] says: Q4_K, Q6_K
/// and Q3_K their gemvs, row-major; Q5_K the down `_sel` over the site as
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
    const WHAT: &str = "site::kgemv";
    match ty {
        SiteTy::Q4K => gpu.enqueue_gemv_q4k(kq(w, WHAT, name)?, act, y),
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
                .enqueue_gemv_q6k_sel(gpu.stream(), wt, act, &kk.one, 1, wt.rows(), y)
        }
        SiteTy::Q6K => gpu.enqueue_gemv_q6k(kq(w, WHAT, name)?, act, y),
        SiteTy::Q3K => gpu.enqueue_gemv_q3k(kq(w, WHAT, name)?, act, y),
        SiteTy::Q5K => {
            let wt = kq(w, WHAT, name)?;
            kk.kq.enqueue_gemv_q5k_sel(
                gpu.stream(),
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
        SiteTy::Nvfp4 => kk
            .nv
            .enqueue_gemv(gpu.stream(), w.nvfp4(WHAT, name)?, act, y),
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
    let wt = match w.get(name) {
        Some(DevWeight::Q5_0 { w, .. }) => w,
        _ => return Err(GpuError::tensor(WHAT, name, "a packed Q5_0 weight")),
    };
    gpu.q5()
        .enqueue_gemv_q5_0(gpu.stream(), wt, act, 0, wt.rows(), 0, act.m(), y, 0)
}

/// Where a site reads its weight: the resident tensor of a name
/// (`Weights`), rows of it, or the K-quant words a caller holds — a window
/// of experts laid out as the resident stack of the site's type, kept
/// outside `Weights`. Each carries the site's name, which a refusal names.
/// Words under any other type than a K-quant's, and rows of a site whose
/// launch reads neither the f32 rows nor q8 blocks of 32, are refused by
/// name ([`gemm`], [`gated`], [`resident_gemv_rows`]).
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

    /// Words under a type that is not a K-quant's, and rows of a type whose
    /// launch reads neither the f32 rows nor q8 blocks of 32, refused by
    /// name as `what`.
    fn check(self, what: &'static str, ty: SiteTy) -> Result<(), GpuError> {
        match self {
            SiteWeight::Words { name, .. } if !ty.kquant() => Err(GpuError::shape(
                what,
                format!("{name} is {ty} on words a caller holds; only a K-quant's words launch"),
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

/// One wide input's quantized forms, each held when a site of the unit reads
/// it: q8_1 blocks of 128 for a K-quant site, q8 blocks of 32 for a Q8_0 or
/// Q5_0 site.
pub struct WideActs {
    pub q128: Option<GemmAct>,
    pub q32: Option<GemmAct32>,
}

impl WideActs {
    /// The wide forms a site of type `ty` reads, `(q128, q32)`: q8_1 blocks of
    /// 128 for a K-quant site; q8 blocks of 32 (`GemmAct32`) for a Q8_0 site
    /// and for a Q5_0 site, whose gemv reads `Q8Blocks32` and whose GEMM
    /// reads the same blocks in the wide layout.
    #[must_use]
    pub fn wants(ty: SiteTy) -> (bool, bool) {
        (
            ty.reads() == Form::Q8x128,
            matches!(ty.reads(), Form::Q8x32 | Form::Q8Blocks32),
        )
    }

    /// The wide forms any of `tys` reads ([`WideActs::wants`] over them).
    #[must_use]
    pub fn wants_of(tys: &[SiteTy]) -> (bool, bool) {
        tys.iter().fold((false, false), |(a, b), &t| {
            let (x, y) = WideActs::wants(t);
            (a | x, b | y)
        })
    }

    /// `cols` columns of `k` values in the forms `(q128, q32)` asks for.
    /// Load-time only.
    pub fn new(
        stream: &cuda_core::CudaStream,
        (q128, q32): (bool, bool),
        cols: usize,
        k: usize,
    ) -> Result<WideActs, GpuError> {
        Ok(WideActs {
            q128: q128.then(|| GemmAct::new(stream, cols, k)).transpose()?,
            q32: q32.then(|| GemmAct32::new(stream, cols, k)).transpose()?,
        })
    }

    pub fn bytes(&self) -> usize {
        self.q128.as_ref().map_or(0, GemmAct::bytes) + self.q32.as_ref().map_or(0, GemmAct32::bytes)
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

    /// The forms a gated block's SwiGLU writes beside the f32 rows `f32`, if
    /// its down reads them.
    pub fn swiglu_rows<'a>(&'a mut self, f32: Option<&'a mut DeviceBuffer<f32>>) -> SwigluRows<'a> {
        SwigluRows {
            q128: self.q128.as_mut(),
            q32: self.q32.as_mut(),
            f32,
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
        let (q128, q32) = WideActs::wants_of(tys);
        if q128 {
            let a = self.q128.as_deref_mut().ok_or(GpuError::state(
                WHAT_Q,
                "the unit's q8_1 blocks of 128 (a K-quant site reads them)",
            ))?;
            gpu.enqueue_quantize_gemm(x, m, a, sink)?;
        }
        if q32 {
            let a = self.q32.as_deref_mut().ok_or(GpuError::state(
                WHAT_Q,
                "the unit's q8 blocks of 32 (a Q8_0 or Q5_0 site reads them)",
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
/// resident weights `src` may name.
#[allow(
    clippy::too_many_arguments,
    reason = "one site's kernels, weight, rows, inputs, table, column rule, width and output (rust-quality R8)"
)]
pub fn gemm(
    gpu: &Gpu,
    k: &WideKernels<'_>,
    (ty, w, src): (SiteTy, &Weights, SiteWeight<'_>),
    rows: usize,
    x: &WideIn<'_>,
    (route, input): (&GemmRoute, GemmInput),
    m: usize,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    const WHAT: &str = "site::gemm";
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
    match ty {
        SiteTy::Nvfp4 => Err(no_wide_gemm(WHAT, name, ty)),
        SiteTy::Q3K | SiteTy::Q4K | SiteTy::Q5K | SiteTy::Q6K => k.gemm.enqueue_gemm(
            stream,
            GemmArgs {
                ty: gemm_weight(ty).ok_or(GpuError::state(WHAT, "a K-quant's GEMM type"))?,
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
        ),
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
            with_planes(WHAT, (ty, weight, site), part, |p| match p {
                Planes::Q8_0(qs, d) => g32()?.enqueue_gemm32(
                    stream,
                    Gemm32Args {
                        w: Gemm32Weight::Q8_0Plane { qs, d },
                        rows_per_expert: rows,
                        act: x.q32.ok_or_else(|| missing("q8 blocks of 32"))?,
                        route,
                        input,
                        y,
                    },
                ),
                Planes::F32(t) => g32()?.enqueue_f32_tile(stream, t, f32_rows()?, m, y),
                Planes::E4m3(t) if gpu.q8f32().dense_mma => {
                    g32()?.enqueue_e4m3_mma(stream, t, f32_rows()?, m, y)
                }
                Planes::E4m3(t) => g32()?.enqueue_e4m3_tile(stream, t, f32_rows()?, m, y),
                Planes::Bf16(t) => {
                    gpu.q8f32()
                        .enqueue_bf16_prompt::<false>(stream, t, f32_rows()?, m, y)
                }
            })
        }
        SiteTy::Q5_0 => {
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
    }
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
    wide_by_name(gpu, k, (w, src, rows), x, route, m, y)
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
    wide_by_name(gpu, k, (w, src, n), x, route, m, y)
}

/// [`gemm`] of site `src`, `rows` rows, at the site type of its resident
/// form, on the one-expert table `route` filled for `m` slots.
fn wide_by_name(
    gpu: &Gpu,
    k: &WideKernels<'_>,
    (w, src, rows): (&Weights, SiteWeight<'_>, usize),
    x: &WideIn<'_>,
    route: &GemmRoute,
    m: usize,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let ty = resident_type(w, "site::resident_gemm", src.name())?;
    let table = (route, GemmInput::PerSlot);
    gemm(gpu, k, (ty, w, src), rows, x, table, m, y)
}

/// The q8 blocks of 32 of the first `m` columns of the f32 rows `x` into
/// `q32` when a site of one of the resident weights `names` reads them
/// ([`WideActs::wants`] of its type, [`of_resident`]): what
/// [`resident_gemm`] reads for a Q8_0 or Q5_0 site.
pub fn resident_quantize(
    gpu: &Gpu,
    k: &WideKernels<'_>,
    (w, names): (&Weights, &[&str]),
    (x, m): (&DeviceBuffer<f32>, usize),
    q32: &mut GemmAct32,
    sink: FaultSink,
) -> Result<(), GpuError> {
    let reads_q32 = |n: &&str| {
        w.get(n)
            .and_then(of_resident)
            .is_some_and(|ty| WideActs::wants(ty).1)
    };
    if names.iter().any(reads_q32) {
        k.g32(
            "site::resident_quantize",
            "the 32-value kernels a site's q8 blocks of 32 are quantized by",
        )?
        .enqueue_quantize_gemm32(gpu.stream(), x, m, q32, sink)?;
    }
    Ok(())
}

/// A site of type `ty` on the wide arm, which has no GEMM for it, refused by
/// name as `what`'s.
fn no_wide_gemm(what: &'static str, name: &str, ty: SiteTy) -> GpuError {
    GpuError::shape(
        what,
        format!("{name} is {ty}: it has a gemv (`site::kgemv`) and no wide GEMM"),
    )
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

    /// What the block refuses whatever the rows: words under a type that is
    /// not a K-quant's, and a rule the down's SwiGLU launch lacks
    /// ([`swiglu_launch`]) — checked before the first launch, so a block
    /// refused here has launched nothing.
    fn check(&self) -> Result<(), GpuError> {
        for (ty, src) in [self.gate, self.up, self.down] {
            src.check("site::gated", ty)?;
            if ty == SiteTy::Nvfp4 {
                return Err(no_wide_gemm("site::gated", src.name(), ty));
            }
        }
        swiglu_launch((self.down.0, self.down.1.name()), self.rule).map(|_| ())
    }
}

/// Where a [`gated`] block's SwiGLU writes its rows: the form the down's type
/// reads ([`SiteTy::reads`]). A form the down does not read may be absent.
pub struct SwigluRows<'a> {
    pub q128: Option<&'a mut GemmAct>,
    pub q32: Option<&'a mut GemmAct32>,
    pub f32: Option<&'a mut DeviceBuffer<f32>>,
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
    /// `ElemKernels::enqueue_swiglu` into f32 rows: `SiluMul` alone.
    F32,
}

/// The launch of a SwiGLU into the form down site `(ty, name)` reads under
/// `rule`; a rule the form's launch lacks is refused by name.
fn swiglu_launch((ty, name): (SiteTy, &str), rule: Act) -> Result<SwigluLaunch, GpuError> {
    match (ty.reads(), rule) {
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
        (Form::F32, Act::SiluMul) => Ok(SwigluLaunch::F32),
        (Form::F32, Act::SwigluClamp { limit }) => Err(GpuError::shape(
            "site::swiglu",
            format!(
                "{name} is {ty}: its SwiGLU rows are f32 and only silu(gate) · up writes them, \
                 not the clamp at {limit}"
            ),
        )),
        (Form::F32, Act::SwigluOai { limit, .. }) => Err(GpuError::shape(
            "site::swiglu",
            format!(
                "{name} is {ty}: its SwiGLU rows are f32 and only silu(gate) · up writes them, \
                 not swigluoai at {limit}"
            ),
        )),
        (Form::F32, Act::Situ { beta, .. }) => Err(GpuError::shape(
            "site::swiglu",
            format!(
                "{name} is {ty}: its SwiGLU rows are f32 and only silu(gate) · up writes them, \
                 not situ at {beta}"
            ),
        )),
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
/// launch with the quantizer for a q8 form, the f32 rows otherwise.
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
            gpu.elem().enqueue_swiglu(stream, g, u, cols * blk.ff, y)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Form, Order, SiteTy, SwigluLaunch, WideActs, file_site, gemm_weight, kgemv_order,
        swiglu_launch,
    };
    use crate::gemm::GemmWeight;
    use crate::kquant::Act;
    use gguf::Split;
    use gguf::quant::GgmlType;
    use gguf::write::{Layout, TensorDecl, Writer};
    use model::placement::kernels;
    use std::fs::File;
    use std::io::BufWriter;

    /// A site's type is read from the file's header alone: a Q8_0 matrix of
    /// two rows of 64 is read as Q8_0 where Q8_0 is launched, and refused by
    /// name where only K-quants are; a Q3_K, a Q5_K, a Q5_0 and an MXFP8 one
    /// (bloomery's id, sized by the split) are read as theirs; an IQ4_XS one (a type no site launch reads) and a matrix of
    /// another shape are refused by name.
    #[test]
    fn a_site_type_comes_from_the_header_and_a_type_with_no_launch_is_refused() {
        let dir = std::env::temp_dir().join(format!("bloomery-site-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{e}"));
        let path = dir.join("m.gguf");
        let decls = [
            ("q8", vec![64, 2], 8u32, 2 * 2 * 34),
            ("iq", vec![256, 1], 23, 136),
            ("q3", vec![256, 2], 11, 2 * 110),
            ("q5", vec![256, 2], 13, 2 * 176),
            ("q50", vec![64, 2], 6, 2 * 2 * 22),
            ("mx", vec![128, 2], GgmlType::MXFP8.as_u32(), 2 * 4 * 33),
            ("bf", vec![64, 2], GgmlType::BF16.as_u32(), 2 * 64 * 2),
        ];
        let tensors = decls
            .iter()
            .map(|(name, dims, type_id, nbytes)| TensorDecl {
                name: (*name).to_string(),
                dims: dims.clone(),
                type_id: *type_id,
                nbytes: *nbytes,
            })
            .collect();
        let layout = Layout::new(&[], tensors).unwrap_or_else(|e| panic!("{e}"));
        let file = BufWriter::new(File::create(&path).unwrap_or_else(|e| panic!("{e}")));
        let mut w = Writer::new(file, layout).unwrap_or_else(|e| panic!("{e}"));
        for (name, _, _, nbytes) in &decls {
            w.tensor(name, &vec![0u8; *nbytes as usize])
                .unwrap_or_else(|e| panic!("{e}"));
        }
        w.finish().unwrap_or_else(|e| panic!("{e}"));
        let split = Split::open(&path).unwrap_or_else(|e| panic!("{e}"));
        let all = [
            SiteTy::Q3K,
            SiteTy::Q4K,
            SiteTy::Q5K,
            SiteTy::Q6K,
            SiteTy::Q5_0,
            SiteTy::Q8_0,
            SiteTy::F32,
            SiteTy::Nvfp4,
            SiteTy::Mxfp8,
            SiteTy::Bf16,
        ];
        for (name, k, want) in [
            ("q8", 64, SiteTy::Q8_0),
            ("q3", 256, SiteTy::Q3K),
            ("q5", 256, SiteTy::Q5K),
            ("q50", 64, SiteTy::Q5_0),
            ("mx", 128, SiteTy::Mxfp8),
            ("bf", 64, SiteTy::Bf16),
        ] {
            let got = file_site(&split, "test", name, (2, k), &all);
            assert_eq!(got.ok(), Some(want), "{name}");
        }
        let refused = [
            (
                "q8",
                (2, 64),
                &[SiteTy::Q4K, SiteTy::Q6K][..],
                "q8 is q8_0 in the file",
            ),
            ("iq", (1, 256), &all[..], "iq is iq4_xs in the file"),
            ("q8", (4, 64), &all[..], "it takes 4 rows of 64"),
        ];
        for (name, shape, allowed, want) in refused {
            match file_site(&split, "test", name, shape, allowed) {
                Err(e) => assert!(e.to_string().contains(want), "{name}: {e}"),
                Ok(t) => panic!("{name} {shape:?} read as {t}"),
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The SwiGLU's launch is the down's form under the rule: every K-quant
    /// one launch into q8_1 blocks of 128, Q8_0 the plain 32-value entry
    /// under `SiluMul` (the rule's entry for the clamp and the OAI form), F32
    /// and BF16 the f32 rows under `SiluMul` alone, the other rules refused
    /// with the site's name.
    #[test]
    fn the_swiglu_launch_is_the_downs_form_under_the_rule() {
        let clamp = Act::SwigluClamp { limit: 7.0 };
        let oai = Act::SwigluOai {
            alpha: crate::kquant::act::SWIGLU_OAI_ALPHA,
            limit: 7.0,
        };
        let launch = |ty, rule| swiglu_launch((ty, "down"), rule);
        for ty in [SiteTy::Q3K, SiteTy::Q4K, SiteTy::Q5K, SiteTy::Q6K] {
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
        for ty in [SiteTy::F32, SiteTy::Bf16] {
            assert_eq!(
                launch(ty, Act::SiluMul).ok(),
                Some(SwigluLaunch::F32),
                "{ty}"
            );
        }
        for (rule, named) in [(clamp, "clamp"), (oai, "swigluoai")] {
            match launch(SiteTy::F32, rule) {
                Err(e) => {
                    let msg = e.to_string();
                    assert!(msg.contains("down is F32") && msg.contains(named), "{msg}");
                }
                Ok(l) => panic!("an F32 down under {named} launched {l:?}"),
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

    /// Every family is the kernel table's dense row of one file type: a type
    /// has a family exactly where the table names a dense entry for it (F16,
    /// which loads widened to f32, is the f32 site of its widened rows), the
    /// family maps back to that type and is named by it in upper case, its
    /// K-quant GEMM is the one the table names for the type, it is a
    /// `kgemv` exactly where it reads the q8_1 rows of 128 values, and it
    /// launches by name (`BY_NAME`) exactly where it reads the f32 rows or
    /// the q8 blocks of 32.
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
        }
        assert_eq!(SiteTy::Q5_0.reads(), Form::Q8Blocks32);
        assert_eq!(kgemv_order(SiteTy::Q5_0), None);
        assert_eq!(SiteTy::Nvfp4.reads(), Form::Q8x128);
        assert_eq!(kgemv_order(SiteTy::Nvfp4), Some(Order::TokenMajor));
        assert_eq!(SiteTy::Mxfp8.reads(), Form::F32);
    }

    #[test]
    fn a_bf16_site_reads_the_f32_rows_and_is_a_gemv() {
        assert_eq!(SiteTy::of_ggml(GgmlType::BF16), Some(SiteTy::Bf16));
        assert!(SiteTy::ALL.contains(&SiteTy::Bf16));
        assert_eq!(SiteTy::Bf16.reads(), Form::F32);
        assert_eq!(kgemv_order(SiteTy::Bf16), None);
        assert_eq!(gemm_weight(SiteTy::Bf16), None);
        assert_eq!(WideActs::wants(SiteTy::Bf16), (false, false));
        assert_eq!(SiteTy::Bf16.to_string(), "BF16");
    }
}
