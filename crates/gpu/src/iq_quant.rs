//! IQ1_S and IQ2_XXS quantization on the card: `gguf::iq_quant`'s rules (ggml's
//! `quantize_row_iq1_s_impl` / `quantize_row_iq2_xxs_impl`) run with every
//! operation spelled as its `.rn` instruction ([`Rn`]), so a row's bytes are
//! the host's and ggml's.
//!
//! Three launches a batch of rows: the source rows widened to f32
//! (`iq_src_f32`: MXFP4, BF16 or F16 file bytes; F32 rows are uploaded as
//! they are), one thread per 32-value sub-block running the sub-block's
//! search (`iq_quant_subs`, four words a sub-block: the rule's two code
//! words, the scale's bits and the status), and one thread per 256-value
//! super-block reading its eight sub-blocks into the block's `d` and codes
//! (`iq_quant_finish`, a record of [`RECORD_WORDS`] words: `d | status << 16`,
//! then IQ2_XXS's sixteen code words or IQ1_S's eight index words and four
//! `qh` pairs). The host packs the records into the file's blocks.
//!
//! Nothing inside a sub-block runs in parallel: ggml's searches keep the
//! first candidate under a strict comparison, so each search runs serially
//! in ggml's order in one thread. The neighbour tables ([`GridTables`]) are
//! uploaded once per [`IqQuant`].

use crate::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::{add_rn_f32, div_rn_f32, mul_rn_f32, sqrt_rn_f32};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use gguf::iq_quant::{
    Fp, GridTables, IqKind, IqQuantError, QK, SUB, SUBS, Sub, Tables, iq1_s_finish, iq1_s_sub,
    iq2_xxs_finish, iq2_xxs_sub, sigma2_iq1_s, sigma2_iq2_xxs,
};
use gguf::quant::{GgmlType, KVALUES_MXFP4, e8m0_to_f32_half, half_to_f32};
use std::sync::Arc;

/// Words of one super-block's record.
pub const RECORD_WORDS: usize = 17;
/// Words of one sub-block's result.
const SUB_WORDS: usize = 4;
/// Threads per block of every entry here.
const BLOCK: u32 = 256;
/// [`Sub::status`] the sub kernel sets when the importance row of its
/// sub-block lies past the uploaded matrix.
pub const STATUS_WEIGHTS: u32 = 16;

/// `kind` argument values.
const KIND_IQ2_XXS: u32 = 0;
const KIND_IQ1_S: u32 = 1;
/// `src` argument values of `iq_src_f32`.
const SRC_MXFP4: u32 = 0;
const SRC_BF16: u32 = 1;
const SRC_F16: u32 = 2;

/// The card's arithmetic: one `.rn` instruction an operation, so nothing is
/// contracted into a fused multiply-add. Device only.
pub struct Rn;

impl Fp for Rn {
    #[inline(always)]
    fn mul(a: f32, b: f32) -> f32 {
        mul_rn_f32(a, b)
    }
    #[inline(always)]
    fn add(a: f32, b: f32) -> f32 {
        add_rn_f32(a, b)
    }
    #[inline(always)]
    fn sub(a: f32, b: f32) -> f32 {
        add_rn_f32(a, -b)
    }
    #[inline(always)]
    fn div(a: f32, b: f32) -> f32 {
        div_rn_f32(a, b)
    }
    #[inline(always)]
    fn sqrt(a: f32) -> f32 {
        sqrt_rn_f32(a)
    }
}

/// Byte `at` of a word slice, little-endian. `at / 4 < w.len()` by the caller.
#[inline(always)]
unsafe fn byte_at(w: &[u32], at: usize) -> u32 {
    // SAFETY: the caller's bound.
    (unsafe { *w.get_unchecked(at >> 2) } >> (8 * (at & 3))) & 0xff
}

/// Value `i` of MXFP4 rows: ggml's `kvalues_mxfp4[q] · e8m0_half(e)`, exact.
#[inline(always)]
unsafe fn mxfp4_value(w: &[u32], i: usize) -> f32 {
    let (b, j) = (i / 32, i % 32);
    let base = 17 * b;
    // SAFETY: the caller's bound: block b's 17 bytes are inside w.
    let (e, q) = unsafe { (byte_at(w, base), byte_at(w, base + 1 + (j & 15))) };
    let nib = if j < 16 { q & 15 } else { q >> 4 };
    mul_rn_f32(
        f32::from(KVALUES_MXFP4[nib as usize]),
        e8m0_to_f32_half(e as u8),
    )
}

#[cuda_module]
mod iq_quant_kernels {
    use super::*;

    /// `x[i]` = value `i` of `n` source values: MXFP4 blocks, BF16 or F16
    /// halves, packed in `w`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), requires = (x.len() >= n))]
    pub fn iq_src_f32(w: &[u32], src: u32, n: u32, mut x: DisjointSlice<f32>) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        let need = if src == SRC_MXFP4 {
            (17 * (i / 32 + 1)).div_ceil(4)
        } else {
            i / 2 + 1
        };
        if w.len() < need {
            return;
        }
        // SAFETY: w holds `need` words, which cover value i's bytes.
        let v = unsafe {
            if src == SRC_MXFP4 {
                mxfp4_value(w, i)
            } else {
                let h = (*w.get_unchecked(i / 2) >> (16 * (i & 1))) & 0xffff;
                if src == SRC_BF16 {
                    f32::from_bits(h << 16)
                } else {
                    half_to_f32(h as u16)
                }
            }
        };
        // SAFETY: i < n <= x.len() by the contract; thread i alone writes it.
        unsafe { *x.get_unchecked_mut(i) = v };
    }

    /// Sub-block `s` of the batch `x` (rows of `n_per_row` values, the
    /// first the tensor's row `row0`): its search with the importance row
    /// of its expert (`qw[(row / rows_per_expert) · n_per_row ..]`), into
    /// `out[4s .. 4s + 4]` = `w0, w1, scale bits, status`.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (x.len() >= n_sub * 32, out.len() >= n_sub * 4)
    )]
    pub fn iq_quant_subs(
        kind: u32,
        x: &[f32],
        qw: &[f32],
        codes: &[u32],
        map: &[i32],
        neighbours: &[u32],
        n_per_row: u32,
        row0: u32,
        rows_per_expert: u32,
        n_sub: u32,
        mut out: DisjointSlice<u32>,
    ) {
        let s = thread::index_1d().get();
        if s >= n_sub as usize || n_per_row == 0 || rows_per_expert == 0 {
            return;
        }
        let e0 = s * SUB;
        let n = n_per_row as usize;
        let row = row0 as usize + e0 / n;
        let q0 = (row / rows_per_expert as usize) * n + e0 % n;
        let r = if q0 + SUB > qw.len() {
            Sub {
                w0: 0,
                w1: 0,
                scale: 0.0,
                status: STATUS_WEIGHTS,
            }
        } else {
            let mut xb = [0.0f32; SUB];
            let mut wb = [0.0f32; SUB];
            for i in 0..SUB {
                // SAFETY: e0 + i < 32·n_sub <= x.len() by the contract;
                // q0 + i < qw.len() by the check above.
                unsafe {
                    xb[i] = *x.get_unchecked(e0 + i);
                    wb[i] = *qw.get_unchecked(q0 + i);
                }
            }
            let h0 = (s / SUBS) * QK;
            let xbl = &x[h0..h0 + QK];
            let t = Tables {
                codes,
                map,
                neighbours,
            };
            if kind == KIND_IQ1_S {
                iq1_s_sub::<Rn>(&xb, &wb, sigma2_iq1_s::<Rn>(xbl), t)
            } else {
                iq2_xxs_sub::<Rn>(&xb, &wb, sigma2_iq2_xxs::<Rn>(xbl), t)
            }
        };
        // SAFETY: 4s + 3 < 4·n_sub <= out.len() by the contract; thread s
        // alone writes its four words.
        unsafe {
            *out.get_unchecked_mut(SUB_WORDS * s) = r.w0;
            *out.get_unchecked_mut(SUB_WORDS * s + 1) = r.w1;
            *out.get_unchecked_mut(SUB_WORDS * s + 2) = r.scale.to_bits();
            *out.get_unchecked_mut(SUB_WORDS * s + 3) = r.status;
        }
    }

    /// Super-block `h`'s record from its eight sub-blocks' results.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (subs.len() >= n_super * 32, out.len() >= n_super * 17)
    )]
    pub fn iq_quant_finish(kind: u32, subs: &[u32], n_super: u32, mut out: DisjointSlice<u32>) {
        let h = thread::index_1d().get();
        if h >= n_super as usize {
            return;
        }
        let mut sb = [Sub {
            w0: 0,
            w1: 0,
            scale: 0.0,
            status: 0,
        }; SUBS];
        let mut status = 0u32;
        for (ib, v) in sb.iter_mut().enumerate() {
            let at = SUB_WORDS * (SUBS * h + ib);
            // SAFETY: at + 3 < 32·n_super <= subs.len() by the contract.
            unsafe {
                v.w0 = *subs.get_unchecked(at);
                v.w1 = *subs.get_unchecked(at + 1);
                v.scale = f32::from_bits(*subs.get_unchecked(at + 2));
                v.status = *subs.get_unchecked(at + 3);
            }
            status |= v.status;
        }
        let mut rec = [0u32; RECORD_WORDS];
        if kind == KIND_IQ1_S {
            let (d, qs, qh) = iq1_s_finish::<Rn>(&sb);
            rec[0] = u32::from(d) | (status << 16);
            rec[1..1 + SUBS].copy_from_slice(&qs);
            for p in 0..SUBS / 2 {
                rec[1 + SUBS + p] = u32::from(qh[2 * p]) | (u32::from(qh[2 * p + 1]) << 16);
            }
        } else {
            let (d, q2) = iq2_xxs_finish::<Rn>(&sb);
            rec[0] = u32::from(d) | (status << 16);
            rec[1..].copy_from_slice(&q2);
        }
        for (k, &v) in rec.iter().enumerate() {
            // SAFETY: 17h + k < 17·n_super <= out.len() by the contract;
            // thread h alone writes its record.
            unsafe { *out.get_unchecked_mut(RECORD_WORDS * h + k) = v };
        }
    }
}

/// A batch the card could not quantize as ggml would, or a launch that failed.
#[derive(Debug)]
pub enum IqQuantCardError {
    Gpu(GpuError),
    Block(IqQuantError),
}

impl std::fmt::Display for IqQuantCardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IqQuantCardError::Gpu(e) => write!(f, "{e}"),
            IqQuantCardError::Block(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for IqQuantCardError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            IqQuantCardError::Gpu(e) => Some(e),
            IqQuantCardError::Block(e) => Some(e),
        }
    }
}

impl From<GpuError> for IqQuantCardError {
    fn from(e: GpuError) -> IqQuantCardError {
        IqQuantCardError::Gpu(e)
    }
}

impl From<cuda_core::LaunchContractError> for IqQuantCardError {
    fn from(e: cuda_core::LaunchContractError) -> IqQuantCardError {
        IqQuantCardError::Gpu(GpuError::from(e))
    }
}

impl From<cuda_core::DriverError> for IqQuantCardError {
    fn from(e: cuda_core::DriverError) -> IqQuantCardError {
        IqQuantCardError::Gpu(GpuError::from(e))
    }
}

/// One grid's neighbour tables on the card.
struct CardTables {
    codes: DeviceBuffer<u32>,
    map: DeviceBuffer<i32>,
    neighbours: DeviceBuffer<u32>,
}

/// The quantizer of one type on one card: its kernels and its grid's tables.
pub struct IqQuant {
    module: iq_quant_kernels::LoadedModule,
    kind: IqKind,
    tables: CardTables,
    times: std::cell::Cell<IqQuantTimes>,
}

/// Seconds [`IqQuant::quantize`] spent in each term, summed over its calls
/// ([`IqQuant::times`]): the host's copy of the source into words, the
/// uploads, the three kernels (to the stream's sync), the records' download
/// and their packing into block bytes.
#[derive(Clone, Copy, Debug, Default)]
pub struct IqQuantTimes {
    pub host_words: f64,
    pub upload: f64,
    pub widen: f64,
    pub search: f64,
    pub finish: f64,
    pub download: f64,
    pub pack: f64,
}

/// The `kind` argument of a type.
fn kind_arg(kind: IqKind) -> u32 {
    match kind {
        IqKind::Iq2Xxs => KIND_IQ2_XXS,
        IqKind::Iq1S => KIND_IQ1_S,
    }
}

impl IqQuant {
    /// Load the kernels into `ctx` and upload `kind`'s tables. Load-time only.
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &CudaStream,
        kind: IqKind,
    ) -> Result<IqQuant, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; every launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(iq_quant_kernels, ctx)? };
        let t = GridTables::new(kind);
        let tables = CardTables {
            codes: DeviceBuffer::from_host(stream, &t.codes)?,
            map: DeviceBuffer::from_host(stream, &t.map)?,
            neighbours: DeviceBuffer::from_host(stream, &t.neighbours)?,
        };
        Ok(IqQuant {
            module,
            kind,
            tables,
            times: std::cell::Cell::new(IqQuantTimes::default()),
        })
    }

    /// The type this quantizer writes.
    pub fn kind(&self) -> IqKind {
        self.kind
    }

    /// The time in each term of every [`IqQuant::quantize`] call so far.
    pub fn times(&self) -> IqQuantTimes {
        self.times.get()
    }

    /// Rows `row0 ..` of a tensor of `n_per_row` columns and
    /// `rows_per_expert` rows an expert, given as `src` bytes of type `ty`
    /// (F32, F16, BF16 or MXFP4, whole rows), quantized with the importance
    /// matrix `qw` (`n_per_row` values an expert, on the card) into the
    /// file's block bytes. A super-block the rule refuses is
    /// [`IqQuantCardError::Block`], its index counted from the batch's first.
    #[allow(
        clippy::too_many_arguments,
        reason = "one batch's geometry: type, rows, expert size, importance"
    )]
    pub fn quantize(
        &self,
        stream: &CudaStream,
        ty: GgmlType,
        src: &[u8],
        n_per_row: usize,
        row0: usize,
        rows_per_expert: usize,
        qw: &DeviceBuffer<f32>,
    ) -> Result<Vec<u8>, IqQuantCardError> {
        let what = "iq_quant::quantize";
        if n_per_row == 0 || !n_per_row.is_multiple_of(QK) {
            return Err(IqQuantCardError::Block(IqQuantError::RowLength {
                n: n_per_row,
            }));
        }
        if rows_per_expert == 0 {
            return Err(GpuError::shape(what, "rows_per_expert = 0".to_string()).into());
        }
        let (n, per) = match ty {
            GgmlType::F32 => (src.len() / 4, 4.0),
            GgmlType::F16 | GgmlType::BF16 => (src.len() / 2, 2.0),
            GgmlType::MXFP4 => (src.len() / 17 * 32, 17.0 / 32.0),
            other => {
                return Err(GpuError::shape(
                    what,
                    format!("source type {other:?}: F32, F16, BF16 or MXFP4"),
                )
                .into());
            }
        };
        if n % n_per_row != 0 || (n as f64 * per) as usize != src.len() {
            return Err(GpuError::shape(
                what,
                format!(
                    "{} source bytes are not whole rows of {n_per_row} {ty:?} values",
                    src.len()
                ),
            )
            .into());
        }
        let mut tm = self.times.get();
        let mut t = std::time::Instant::now();
        let mut lap = |stream: &CudaStream, term: &mut f64| -> Result<(), GpuError> {
            stream.synchronize()?;
            *term += t.elapsed().as_secs_f64();
            t = std::time::Instant::now();
            Ok(())
        };
        let x: DeviceBuffer<f32> = if ty == GgmlType::F32 {
            let v: Vec<f32> = src
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            lap(stream, &mut tm.host_words)?;
            let x = DeviceBuffer::from_host(stream, &v)?;
            lap(stream, &mut tm.upload)?;
            x
        } else {
            let (whole, tail) = src.as_chunks::<4>();
            let mut words: Vec<u32> = Vec::with_capacity(src.len().div_ceil(4));
            words.extend(whole.iter().map(|b| u32::from_le_bytes(*b)));
            if !tail.is_empty() {
                let mut a = [0u8; 4];
                a[..tail.len()].copy_from_slice(tail);
                words.push(u32::from_le_bytes(a));
            }
            lap(stream, &mut tm.host_words)?;
            let w = DeviceBuffer::from_host(stream, &words)?;
            lap(stream, &mut tm.upload)?;
            let mut x = DeviceBuffer::<f32>::zeroed(stream, n)?;
            let src_arg = match ty {
                GgmlType::MXFP4 => SRC_MXFP4,
                GgmlType::BF16 => SRC_BF16,
                _ => SRC_F16,
            };
            let n32 = launch_u32(what, "n", n)?;
            let cfg = LaunchConfig1D::new(n32.div_ceil(BLOCK), BLOCK, 0);
            let prep = self.module.prepare_iq_src_f32(cfg)?;
            self.module
                .iq_src_f32(stream, &prep, &w, src_arg, n32, &mut x)?;
            lap(stream, &mut tm.widen)?;
            x
        };
        let n_sub = n / SUB;
        let n_super = n / QK;
        let mut subs = DeviceBuffer::<u32>::zeroed(stream, SUB_WORDS * n_sub)?;
        let mut recs = DeviceBuffer::<u32>::zeroed(stream, RECORD_WORDS * n_super)?;
        let kind = kind_arg(self.kind);
        let n_sub32 = launch_u32(what, "n_sub", n_sub)?;
        let cfg = LaunchConfig1D::new(n_sub32.div_ceil(BLOCK), BLOCK, 0);
        let prep = self.module.prepare_iq_quant_subs(cfg)?;
        self.module.iq_quant_subs(
            stream,
            &prep,
            kind,
            &x,
            qw,
            &self.tables.codes,
            &self.tables.map,
            &self.tables.neighbours,
            launch_u32(what, "n_per_row", n_per_row)?,
            launch_u32(what, "row0", row0)?,
            launch_u32(what, "rows_per_expert", rows_per_expert)?,
            n_sub32,
            &mut subs,
        )?;
        lap(stream, &mut tm.search)?;
        let n_super32 = launch_u32(what, "n_super", n_super)?;
        let cfg = LaunchConfig1D::new(n_super32.div_ceil(BLOCK), BLOCK, 0);
        let prep = self.module.prepare_iq_quant_finish(cfg)?;
        self.module
            .iq_quant_finish(stream, &prep, kind, &subs, n_super32, &mut recs)?;
        lap(stream, &mut tm.finish)?;
        let recs = recs.to_host_vec(stream)?;
        lap(stream, &mut tm.download)?;
        let out = pack_records(self.kind, &recs);
        lap(stream, &mut tm.pack)?;
        self.times.set(tm);
        out
    }
}

/// The file's blocks from the finish kernel's records; the first record
/// with a status is refused by name.
pub fn pack_records(kind: IqKind, recs: &[u32]) -> Result<Vec<u8>, IqQuantCardError> {
    let n_super = recs.len() / RECORD_WORDS;
    let mut out = Vec::with_capacity(n_super * kind.block_bytes());
    for (block, rec) in recs.as_chunks::<RECORD_WORDS>().0.iter().enumerate() {
        let status = rec[0] >> 16;
        if status != 0 {
            return Err(IqQuantCardError::Block(IqQuantError::Status {
                block,
                status,
            }));
        }
        out.extend_from_slice(&(rec[0] as u16).to_le_bytes());
        let body = match kind {
            IqKind::Iq2Xxs => &rec[1..],
            IqKind::Iq1S => &rec[1..1 + SUBS + SUBS / 2],
        };
        for w in body {
            out.extend_from_slice(&w.to_le_bytes());
        }
    }
    Ok(out)
}
