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
//!   past it, the two bit for bit equal on a column), F32 the F32 tile
//!   (`f32_tile_gemm`, token-major, bit for bit `f32_gemv` on every row and
//!   column). A K-quant site's gemv is [`kgemv`].
//! - [`kgemv`], a K-quant site at most eight columns over the q8_1 rows:
//!   Q3_K, Q4_K and Q6_K their gemvs, row-major (`m` outputs a row); Q5_K
//!   the K-quant down `_sel` over a stack of one expert, token-major (a slot
//!   a column). Where a body copies a row-major output token-major, and its
//!   fused groups of Q4_K launches, stay the body's.
//! - [`gemm`], a wide unit through a route table: a K-quant the grouped
//!   int8 GEMM over the q8_1 blocks of 128 values ([`GemmAct`]), Q8_0 the
//!   32-value GEMM over the q8 blocks of 32 values ([`GemmAct32`]) on the
//!   q8f32 planes, F32 the F32 tile over the f32 rows (one column a slot, on
//!   the unit's one-expert table only).
//!
//! Which activation form a site reads is [`SiteTy::reads`]: a body
//! quantizes a unit's rows into exactly the forms its sites read.

use crate::gemm::{
    Gemm32Args, Gemm32Kernels, Gemm32Weight, GemmAct, GemmAct32, GemmArgs, GemmInput, GemmKernels,
    GemmRoute, GemmWeight,
};
use crate::kquant::{KquantKernels, SelDown};
use crate::q8f32::{GemvOut, Q8_0GemvMcolArgs};
use crate::tensor::Q8Act;
use crate::weights::{DevWeight, Weights};
use crate::{FaultSink, Gpu, GpuError};
use cuda_core::DeviceBuffer;
use gguf::Split;
use gguf::quant::GgmlType;
use std::fmt;

/// The launch family a site's file type selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SiteTy {
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8_0,
    F32,
}

/// The activation form a site's launch reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    /// q8_1 blocks of 128 values (the K-quant gemv's `Q8Act`, the grouped
    /// GEMM's `GemmAct`).
    Q8x128,
    /// The f32 rows on the gemv arm; q8 blocks of 32 values (`GemmAct32`) on
    /// the wide arm.
    Q8x32,
    /// The f32 rows on both arms.
    F32,
}

/// Where a gemv writes output `r` of column `c` of a site of `rows` rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    /// `y[r · m + c]`, the `m` columns of a row together.
    RowMajor,
    /// `y[c · rows + r]`.
    TokenMajor,
}

impl SiteTy {
    /// The family of a file tensor of type `ty`; `None` for a type no launch
    /// here reads.
    #[must_use]
    pub fn of_ggml(ty: GgmlType) -> Option<SiteTy> {
        match ty {
            GgmlType::Q3_K => Some(SiteTy::Q3K),
            GgmlType::Q4_K => Some(SiteTy::Q4K),
            GgmlType::Q5_K => Some(SiteTy::Q5K),
            GgmlType::Q6_K => Some(SiteTy::Q6K),
            GgmlType::Q8_0 => Some(SiteTy::Q8_0),
            GgmlType::F32 => Some(SiteTy::F32),
            _ => None,
        }
    }

    /// Whether the type is one of the four K-quants.
    #[must_use]
    pub fn kquant(self) -> bool {
        self.gemm_weight().is_some()
    }

    /// The grouped GEMM's type of a K-quant; `None` for Q8_0 and F32.
    #[must_use]
    pub fn gemm_weight(self) -> Option<GemmWeight> {
        match self {
            SiteTy::Q3K => Some(GemmWeight::Q3K),
            SiteTy::Q4K => Some(GemmWeight::Q4K),
            SiteTy::Q5K => Some(GemmWeight::Q5K),
            SiteTy::Q6K => Some(GemmWeight::Q6K),
            SiteTy::Q8_0 | SiteTy::F32 => None,
        }
    }

    /// How a K-quant site's [`kgemv`] lays out its columns: Q3_K, Q4_K and
    /// Q6_K row-major, Q5_K token-major; `None` for a type whose gemv is
    /// [`gemv`].
    #[must_use]
    pub fn kgemv_order(self) -> Option<Order> {
        match self {
            SiteTy::Q3K | SiteTy::Q4K | SiteTy::Q6K => Some(Order::RowMajor),
            SiteTy::Q5K => Some(Order::TokenMajor),
            SiteTy::Q8_0 | SiteTy::F32 => None,
        }
    }

    /// The activation form the site's launch reads.
    #[must_use]
    pub fn reads(self) -> Form {
        match self {
            SiteTy::Q3K | SiteTy::Q4K | SiteTy::Q5K | SiteTy::Q6K => Form::Q8x128,
            SiteTy::Q8_0 => Form::Q8x32,
            SiteTy::F32 => Form::F32,
        }
    }
}

impl fmt::Display for SiteTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            SiteTy::Q3K => "Q3_K",
            SiteTy::Q4K => "Q4_K",
            SiteTy::Q5K => "Q5_K",
            SiteTy::Q6K => "Q6_K",
            SiteTy::Q8_0 => "Q8_0",
            SiteTy::F32 => "F32",
        })
    }
}

/// The type of site `name` in `file`, a matrix of `rows` rows (every expert's
/// of a stack) of `k` values, read from its header entry before any upload:
/// a type outside `allowed`, another shape or a missing tensor is refused by
/// name with the site.
pub fn file_site(
    file: &Split,
    what: &'static str,
    name: &str,
    (rows, k): (usize, usize),
    allowed: &[SiteTy],
) -> Result<SiteTy, GpuError> {
    let (_, t) = file
        .find(name)
        .ok_or_else(|| GpuError::tensor(what, name, "in the file"))?;
    let got_k = t.dims.first().copied().unwrap_or(0);
    let got_rows: u64 = t.dims.iter().skip(1).product();
    if usize::try_from(got_k).ok() != Some(k) || usize::try_from(got_rows).ok() != Some(rows) {
        return Err(GpuError::shape(
            what,
            format!(
                "{name} is {:?} in the file; the site takes {rows} rows of {k}",
                t.dims
            ),
        ));
    }
    match SiteTy::of_ggml(t.ty).filter(|ty| allowed.contains(ty)) {
        Some(ty) => Ok(ty),
        None => Err(GpuError::shape(
            what,
            format!(
                "{name} is {} in the file; this site launches {}",
                t.ty,
                list(allowed)
            ),
        )),
    }
}

/// The resident weight of site `name`: of type `ty`, `rows` rows of `k`
/// values — the type its file entry was read as ([`file_site`]); any other
/// variant or shape is refused by name.
pub fn site(
    w: &Weights,
    what: &'static str,
    name: &str,
    (rows, k): (usize, usize),
    ty: SiteTy,
) -> Result<(), GpuError> {
    let got = match w.get(name) {
        None => return Err(GpuError::tensor(what, name, "resident")),
        Some(DevWeight::KQuant { ty: t, w: p, k: wk }) => SiteTy::of_ggml(*t)
            .filter(|s| s.kquant())
            .map(|s| (s, p.rows(), *wk)),
        Some(DevWeight::Q8_0 { d, k: wk, .. }) => Some((SiteTy::Q8_0, d.rows(), *wk)),
        Some(DevWeight::F32 { w: p, k: wk }) => Some((SiteTy::F32, p.rows(), *wk)),
        Some(_) => None,
    };
    match got {
        Some(g) if g == (ty, rows, k) => Ok(()),
        Some((t, r, wk)) => Err(GpuError::shape(
            what,
            format!(
                "{name} is resident as {t} {r} rows of {wk}; the site takes {ty} {rows} of {k}"
            ),
        )),
        None => Err(GpuError::shape(
            what,
            format!("{name} is resident in a form no site launch reads; the site takes {ty}"),
        )),
    }
}

/// `allowed` as a list of names.
fn list(allowed: &[SiteTy]) -> String {
    allowed
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

/// The resident Q8_0 planes of `name`, or a named refusal.
fn q8<'w>(
    w: &'w Weights,
    what: &'static str,
    name: &str,
) -> Result<(&'w crate::DeviceTensor<u32>, &'w crate::DeviceTensor<u16>), GpuError> {
    match w.get(name) {
        Some(DevWeight::Q8_0 { qs, d, .. }) => Ok((qs, d)),
        Some(_) => Err(GpuError::tensor(what, name, "Q8_0 (the q8f32 planes)")),
        None => Err(GpuError::tensor(what, name, "resident")),
    }
}

/// The resident F32 plane of `name`, or a named refusal.
fn f32w<'w>(
    w: &'w Weights,
    what: &'static str,
    name: &str,
) -> Result<&'w crate::DeviceTensor<f32>, GpuError> {
    match w.get(name) {
        Some(DevWeight::F32 { w, .. }) => Ok(w),
        Some(_) => Err(GpuError::tensor(what, name, "F32")),
        None => Err(GpuError::tensor(what, name, "resident")),
    }
}

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
/// columns of the f32 rows `x`, token-major (module doc). A K-quant site is
/// the body's launch and is refused here by name.
pub fn gemv(
    gpu: &Gpu,
    g32: &Gemm32Kernels,
    (ty, w, name): (SiteTy, &Weights, &str),
    x: &DeviceBuffer<f32>,
    m: usize,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    const WHAT: &str = "site::gemv";
    let stream = gpu.stream();
    match ty {
        SiteTy::Q8_0 => {
            let (qs, d) = q8(w, WHAT, name)?;
            let q = gpu.q8f32();
            if m == 1 {
                q.enqueue_q8_0_gemv(stream, qs, d, x, 1, y)
            } else {
                q.enqueue_q8_0_gemv_mcol(
                    stream,
                    Q8_0GemvMcolArgs {
                        qs,
                        d,
                        x,
                        m,
                        out: GemvOut::TokenMajor,
                        y,
                    },
                )
            }
        }
        SiteTy::F32 => g32.enqueue_f32_tile(stream, f32w(w, WHAT, name)?, x, m, y),
        SiteTy::Q3K | SiteTy::Q4K | SiteTy::Q5K | SiteTy::Q6K => Err(GpuError::shape(
            WHAT,
            format!("{name} is {ty}: a K-quant site's gemv is `site::kgemv`, on the q8_1 rows"),
        )),
    }
}

/// The most columns a [`Q8Act`] holds, so the most a [`kgemv`] runs.
const KGEMV_COLS: usize = 8;

/// The kernels a K-quant site's [`kgemv`] launches beyond the `Gpu`'s own:
/// the K-quant `_sel` family, whose Q5_K down entry over a stack of one
/// expert is the Q5_K gemv, and that one expert's id for each of
/// [`KGEMV_COLS`] slots.
pub struct KGemvKernels {
    kq: KquantKernels,
    one: DeviceBuffer<u32>,
}

impl KGemvKernels {
    /// Load the `_sel` family into `gpu`'s context, raising into its fault
    /// word, and zero the ids. Load-time only.
    pub fn load(gpu: &Gpu) -> Result<KGemvKernels, GpuError> {
        Ok(KGemvKernels {
            kq: KquantKernels::load(gpu.context(), gpu.fault_word())?,
            one: DeviceBuffer::zeroed(gpu.stream(), KGEMV_COLS)?,
        })
    }
}

/// `y = W · act` for K-quant site `name` of type `ty` over the `act.m()`
/// columns of the q8_1 rows `act`, laid out as [`SiteTy::kgemv_order`]
/// says: Q4_K, Q6_K and Q3_K their gemvs, row-major; Q5_K the down `_sel`
/// over the site as one expert, token-major, an id past it raising on
/// `fault`. Any other type's gemv is [`gemv`], refused here by name.
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
        SiteTy::Q8_0 | SiteTy::F32 => Err(GpuError::shape(
            WHAT,
            format!("{name} is {ty}: its gemv is `site::gemv`, on the f32 rows"),
        )),
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

/// The kernels a wide site launches.
pub struct WideKernels<'a> {
    pub gemm: &'a GemmKernels,
    pub g32: &'a Gemm32Kernels,
}

/// `y = W · x` for site `name` of type `ty`, `rows` rows an expert, over the
/// slots `route` holds, `input` picking each slot's column (module doc). A
/// 32-value GEMM over the unit's one-expert table must find it filled for
/// exactly `m` slots, and an F32 site runs only there (one column a slot);
/// a form the site reads that `x` lacks is refused by name.
#[allow(
    clippy::too_many_arguments,
    reason = "one site's kernels, weight, rows, inputs, table, column rule, width and output (rust-quality R8)"
)]
pub fn gemm(
    gpu: &Gpu,
    k: &WideKernels<'_>,
    (ty, w, name): (SiteTy, &Weights, &str),
    rows: usize,
    x: &WideIn<'_>,
    (route, input): (&GemmRoute, GemmInput),
    m: usize,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    const WHAT: &str = "site::gemm";
    let stream = gpu.stream();
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
    match ty {
        SiteTy::Q3K | SiteTy::Q4K | SiteTy::Q5K | SiteTy::Q6K => k.gemm.enqueue_gemm(
            stream,
            GemmArgs {
                ty: ty
                    .gemm_weight()
                    .ok_or(GpuError::state(WHAT, "a K-quant's GEMM type"))?,
                w: kq(w, WHAT, name)?,
                rows_per_expert: rows,
                act: x.q128.ok_or_else(|| missing("q8_1 blocks of 128"))?,
                route,
                input,
                y,
            },
        ),
        SiteTy::Q8_0 => {
            dense()?;
            let (qs, d) = q8(w, WHAT, name)?;
            k.g32.enqueue_gemm32(
                stream,
                Gemm32Args {
                    w: Gemm32Weight::Q8_0Plane { qs, d },
                    rows_per_expert: rows,
                    act: x.q32.ok_or_else(|| missing("q8 blocks of 32"))?,
                    route,
                    input,
                    y,
                },
            )
        }
        SiteTy::F32 => {
            dense()?;
            let rows = x.f32.ok_or_else(|| missing("f32 rows"))?;
            k.g32
                .enqueue_f32_tile(stream, f32w(w, WHAT, name)?, rows, m, y)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{SiteTy, file_site};
    use gguf::Split;
    use gguf::write::{Layout, TensorDecl, Writer};
    use std::fs::File;
    use std::io::BufWriter;

    /// A site's type is read from the file's header alone: a Q8_0 matrix of
    /// two rows of 64 is read as Q8_0 where Q8_0 is launched, and refused by
    /// name where only K-quants are; a Q3_K and a Q5_K one are read as
    /// theirs; an IQ4_XS one (a type no site launch reads) and a matrix of
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
            SiteTy::Q8_0,
            SiteTy::F32,
        ];
        for (name, k, want) in [
            ("q8", 64, SiteTy::Q8_0),
            ("q3", 256, SiteTy::Q3K),
            ("q5", 256, SiteTy::Q5K),
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
            ("q8", (4, 64), &all[..], "the site takes 4 rows of 64"),
        ];
        for (name, shape, allowed, want) in refused {
            match file_site(&split, "test", name, shape, allowed) {
                Err(e) => assert!(e.to_string().contains(want), "{name}: {e}"),
                Ok(t) => panic!("{name} {shape:?} read as {t}"),
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
