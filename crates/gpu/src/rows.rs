//! A row table's rows on the card: the one owner of reading a call's rows of
//! a table held on the host or the NVMe tier, uploading them as read and
//! decoding them there, whatever the table's row format. A model's code
//! names the rows (its hash, its history) and gives the decoded values their
//! meaning; this module neither knows nor cares which model asks.
//!
//! [`RowFormat`] names a row's encoding: IQ4_NL blocks (an f16 scale and 16
//! code bytes a 32 values, `kvalues_iq4nl`), bloomery's F8 rows (one E4M3
//! byte a value at the tensor's scale) or its NVFP4 rows (a UE4M3 scale byte
//! and 8 code bytes a 16 values at the tensor's scale). Rows arrive
//! [`Laid::Rows`], as the file holds them, or, for NVFP4 rows a split table
//! stores apart, [`Laid::Planes`]: every code byte of the read, then every
//! scale byte. The host decode ([`RowFormat::decode`]) and the card's
//! ([`RowKernels::enqueue`]) give the same bits: each value is one exact
//! product of the code's value and the block scale (IQ4_NL: the f16 scale and
//! the code; F8: the doubled UE4M3 half and the sign; NVFP4: the UE4M3 half
//! and the doubled E2M1 code) times the tensor scale where there is one,
//! rounded once. A span the format leaves undefined (an E4M3 NaN, a UE4M3 byte
//! that is no scale, a scaled value outside f32's range) is refused by name
//! on the host before anything is enqueued; IQ4_NL defines every block.
//!
//! [`RowRead`] is the NVMe tier's reader: engram's prefetcher over the table,
//! whose helper advises and copies a call's rows off the calling thread, in
//! parts over the worker pool's threads ([`HelperOptions::spread`]). It holds
//! a prompt call's later reads ahead, each keyed by its first position, its
//! positions and its row ids, so a fill takes only the read that is its own
//! and drains a stale one before it reads.

use crate::GpuError;
use crate::launch_u32;
use crate::q8f32::{ByteBlocks, Mxfp8Bytes};
use crate::tensor::WindowMut;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::mul_rn_f32;
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use engram::RowTables;
use engram::prefetch::{FillMode, HelperOptions, Prefetcher, caller_sibling};
use engram::rows::{PageAdvice, RunTable, SplitTable};
use gguf::quant::{E4m3Rows, Nvfp4Rows, dequant_row};
use gguf::{GgmlType, Split};
use model::part::StoredRows;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

const WHAT: &str = "row table";

/// Threads a block of the decode launches.
const BLOCK: u32 = 256;

/// How a table's rows encode their values.
pub enum RowFormat {
    /// GGML's IQ4_NL blocks.
    Iq4Nl,
    /// bloomery's F8 rows ([`GgmlType::F8E4M3_ROWS`]) at the table's scale.
    F8(Box<E4m3Rows>),
    /// bloomery's NVFP4 rows ([`GgmlType::NVFP4_ROWS`]) at the table's scale.
    Nvfp4(Box<Nvfp4Rows>),
}

/// How a read's bytes lie: whole rows one after another, or the two planes
/// of NVFP4 rows a split table stores (every code byte, then every scale
/// byte).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Laid {
    Rows,
    Planes,
}

impl RowFormat {
    /// The format of a table of type `ty`, `scale` its tensor scale where
    /// the type has one; `None` for a type no row decode takes.
    pub fn of(
        ty: GgmlType,
        scale: impl FnOnce() -> Result<f32, GpuError>,
    ) -> Result<Option<RowFormat>, GpuError> {
        Ok(match ty {
            GgmlType::IQ4_NL => Some(RowFormat::Iq4Nl),
            GgmlType::F8E4M3_ROWS => Some(RowFormat::F8(Box::new(E4m3Rows::new(scale()?)))),
            GgmlType::NVFP4_ROWS => Some(RowFormat::Nvfp4(Box::new(Nvfp4Rows::new(scale()?)))),
            _ => None,
        })
    }

    /// One unit of the format: its bytes and its values.
    fn unit(&self) -> (usize, usize) {
        let row = match self {
            RowFormat::Iq4Nl => GgmlType::IQ4_NL,
            RowFormat::F8(_) => GgmlType::F8E4M3_ROWS,
            RowFormat::Nvfp4(_) => GgmlType::NVFP4_ROWS,
        }
        .row()
        .expect("every row format has a registered GGML type");
        (
            usize::try_from(row.bytes).expect("registered block byte counts fit usize"),
            usize::try_from(row.blck).expect("registered block value counts fit usize"),
        )
    }

    /// The values `bytes` bytes laid as `laid` hold: whole units, and planes
    /// of NVFP4 rows alone; anything else refused by name.
    fn values(&self, laid: Laid, bytes: usize) -> Result<usize, GpuError> {
        let (unit, values) = self.unit();
        if laid == Laid::Planes && !matches!(self, RowFormat::Nvfp4(_)) {
            return Err(GpuError::shape(
                WHAT,
                "two planes of rows other than NVFP4 rows",
            ));
        }
        if !bytes.is_multiple_of(unit) {
            return Err(GpuError::shape(
                WHAT,
                format!("{bytes} B of rows of {unit}-byte units"),
            ));
        }
        Ok(bytes / unit * values)
    }

    /// `raw`, rows laid as `laid`, decoded into `e`, one value each.
    pub fn decode(&self, raw: &[u8], laid: Laid, e: &mut [f32]) -> Result<(), GpuError> {
        let values = self.values(laid, raw.len())?;
        let held = e.len();
        let e = e
            .get_mut(..values)
            .ok_or_else(|| GpuError::shape(WHAT, format!("{values} values into {held} decoded")))?;
        match (self, laid) {
            (RowFormat::Iq4Nl, _) => dequant_row(GgmlType::IQ4_NL, raw, e),
            (RowFormat::F8(f8), _) => f8.decode(raw, e),
            (RowFormat::Nvfp4(fp4), Laid::Rows) => fp4.decode(raw, e),
            (RowFormat::Nvfp4(fp4), Laid::Planes) => {
                let (codes, scales) = raw.split_at(values / 2);
                fp4.decode_planes(codes, scales, e)
            }
        }
        .map_err(|err| GpuError::from(model::ModelError::from(err)))
    }

    /// [`RowFormat::decode`]'s refusals of `raw` alone, by the same names.
    fn check(&self, raw: &[u8], laid: Laid) -> Result<(), GpuError> {
        let values = self.values(laid, raw.len())?;
        match (self, laid) {
            (RowFormat::Iq4Nl, _) => Ok(()),
            (RowFormat::F8(f8), _) => f8.check(raw),
            (RowFormat::Nvfp4(fp4), Laid::Rows) => fp4.check(raw, values),
            (RowFormat::Nvfp4(fp4), Laid::Planes) => {
                let (codes, scales) = raw.split_at(values / 2);
                fp4.check_planes(codes, scales, values)
            }
        }
        .map_err(|err| GpuError::shape(WHAT, err.to_string()))
    }
}

/// The value of FP4 code byte `q`'s low nibble then its high under half scale
/// `half`: `(half·v)·scale`, `v` the doubled E2M1 value through the shared
/// nibble decode ([`crate::cores::table16_word`] over
/// [`crate::mxfp4::E2M1`]).
#[inline(always)]
fn fp4_pair(q: u8, half: f32, scale: f32) -> [f32; 2] {
    let (lo, hi) = crate::cores::table16_word(u32::from(q), crate::mxfp4::E2M1);
    let v = |c: u32| mul_rn_f32(mul_rn_f32(half, f32::from(c as u8 as i8)), scale);
    [v(lo), v(hi)]
}

#[cuda_module]
mod row_kernels {
    use super::*;

    /// Group `g` of NVFP4 rows held as two planes in `planes` (the
    /// `groups·8` code bytes, then the `groups` scale bytes) decoded into
    /// `e[at + 16g ..]` as `gguf::quant::Nvfp4Rows` decodes it: value `2i`
    /// the low nibble of code byte `i` and `2i + 1` its high, each
    /// `(h·v)·scale`, `h` half the UE4M3 scale byte and `v` the doubled E2M1
    /// code, `h·v` exact and its product with the tensor scale the one
    /// rounding. The host refused the planes' undefined scales and values
    /// first ([`RowKernels::enqueue`]).
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (planes.len() >= 9 * groups, e.len() >= at + 16 * groups)
    )]
    pub fn rows_nvfp4_planes(
        planes: &[u8],
        groups: u32,
        scale: f32,
        at: u32,
        mut e: DisjointSlice<f32>,
    ) {
        let g = thread::index_1d().get();
        if g >= groups as usize {
            return;
        }
        let base = at as usize + 16 * g;
        // SAFETY: g < groups, so the scale byte 8·groups + g < 9·groups lies
        // in planes by the launch contract.
        let byte = unsafe { *planes.get_unchecked(8 * groups as usize + g) };
        let half = crate::mxfp4::ue4m3_half(u32::from(byte));
        for i in 0..8 {
            ::cuda_device::thread::__unroll_config::<0>();
            // SAFETY: code byte 8g + i < 8·groups lies in planes by the launch
            // contract.
            let q = unsafe { *planes.get_unchecked(8 * g + i) };
            let [lo, hi] = fp4_pair(q, half, scale);
            // SAFETY: value at + 16g + 2i < at + 16·groups lies in e by the
            // launch contract; thread g alone writes its group's 16 values.
            unsafe { *e.get_unchecked_mut(base + 2 * i) = lo };
            // SAFETY: value at + 16g + 2i + 1 < at + 16·groups, likewise.
            unsafe { *e.get_unchecked_mut(base + 2 * i + 1) = hi };
        }
    }

    /// Group `g` of NVFP4 rows as the file holds them (9 bytes a group: the
    /// UE4M3 scale byte, then 8 code bytes) decoded into `e[at + 16g ..]`
    /// as [`rows_nvfp4_planes`] decodes its planes.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (raw.len() >= 9 * groups, e.len() >= at + 16 * groups)
    )]
    pub fn rows_nvfp4(raw: &[u8], groups: u32, scale: f32, at: u32, mut e: DisjointSlice<f32>) {
        let g = thread::index_1d().get();
        if g >= groups as usize {
            return;
        }
        let base = at as usize + 16 * g;
        // SAFETY: g < groups, so byte 9g < 9·groups lies in raw by the launch
        // contract.
        let half = crate::mxfp4::ue4m3_half(u32::from(unsafe { *raw.get_unchecked(9 * g) }));
        for i in 0..8 {
            ::cuda_device::thread::__unroll_config::<0>();
            // SAFETY: byte 9g + 1 + i < 9·groups lies in raw by the launch
            // contract.
            let q = unsafe { *raw.get_unchecked(9 * g + 1 + i) };
            let [lo, hi] = fp4_pair(q, half, scale);
            // SAFETY: value at + 16g + 2i < at + 16·groups lies in e by the
            // launch contract; thread g alone writes its group's 16 values.
            unsafe { *e.get_unchecked_mut(base + 2 * i) = lo };
            // SAFETY: value at + 16g + 2i + 1 < at + 16·groups, likewise.
            unsafe { *e.get_unchecked_mut(base + 2 * i + 1) = hi };
        }
    }

    /// Block `b` of IQ4_NL rows (18 bytes: the f16 scale `d`, then 16 code
    /// bytes) decoded into `e[at + 32b ..]` as `gguf::quant::dequant_row`
    /// decodes it: value `j` the low nibble of code byte `j` and `16 + j` its
    /// high, each `d·kvalues_iq4nl[code]` rounded once.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (raw.len() >= 18 * blocks, e.len() >= at + 32 * blocks)
    )]
    pub fn rows_iq4nl(raw: &[u8], blocks: u32, at: u32, mut e: DisjointSlice<f32>) {
        let b = thread::index_1d().get();
        if b >= blocks as usize {
            return;
        }
        let (src, base) = (18 * b, at as usize + 32 * b);
        // SAFETY: bytes 18b .. 18b + 18 lie in raw (18·blocks bytes) by the
        // launch contract.
        let byte = |i: usize| u32::from(unsafe { *raw.get_unchecked(src + i) });
        let d = crate::cores::half_to_f32((byte(0) | (byte(1) << 8)) as u16);
        for w in 0..4 {
            ::cuda_device::thread::__unroll_config::<0>();
            let at = 2 + 4 * w;
            let word = byte(at) | (byte(at + 1) << 8) | (byte(at + 2) << 16) | (byte(at + 3) << 24);
            let (lo, hi) = crate::iq::iq4_word(word);
            for k in 0..4 {
                ::cuda_device::thread::__unroll_config::<0>();
                let v = |c: u32| mul_rn_f32(d, f32::from((c >> (8 * k)) as u8 as i8));
                // SAFETY: value at + 32b + 4w + k lies below at + 32·blocks
                // in e by the launch contract; thread b alone writes its
                // block's 32 values.
                unsafe { *e.get_unchecked_mut(base + 4 * w + k) = v(lo) };
                // SAFETY: value at + 32b + 16 + 4w + k, likewise.
                unsafe { *e.get_unchecked_mut(base + 16 + 4 * w + k) = v(hi) };
            }
        }
    }

    /// Value `i` of F8 rows (one E4M3FN byte a value) decoded into `e[at + i]`
    /// as `gguf::quant::E4m3Rows` decodes it: the byte's value (MXFP8's byte
    /// rule, [`Mxfp8Bytes::value`], exact) times the tensor scale, rounded
    /// once. The host refused the NaN bytes and the
    /// values outside f32's range first ([`RowKernels::enqueue`]).
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (raw.len() >= n, e.len() >= at + n)
    )]
    pub fn rows_f8(raw: &[u8], n: u32, scale: f32, at: u32, mut e: DisjointSlice<f32>) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        // SAFETY: i < n lies in raw by the launch contract.
        let b = u32::from(unsafe { *raw.get_unchecked(i) });
        let v = mul_rn_f32(Mxfp8Bytes::value(b), scale);
        // SAFETY: at + i < at + n lies in e by the launch contract; thread i
        // alone writes it.
        unsafe { *e.get_unchecked_mut(at as usize + i) = v };
    }
}

/// The loaded row decode module. Owns no stream: each enqueue takes the
/// caller's, so its copies and launches order with the rest of its walk.
pub struct RowKernels {
    module: row_kernels::LoadedModule,
}

impl RowKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<RowKernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the
        // module above; the launchers check its launch contracts.
        let module = unsafe { crate::shared_module!(row_kernels, ctx)? };
        Ok(RowKernels { module })
    }

    /// `raw`, a read's rows of `format` laid as `laid`, decoded into the
    /// values of `e` from value `at` on, through `staging`, which receives
    /// the bytes as read: one copy and one launch (a thread a unit: an
    /// NVFP4 group, an IQ4_NL block, an F8 value) for the read, or, for
    /// rows past `staging`, one of each for every part of whole units it
    /// holds, in order on `stream`. Planes are one copy: planes past
    /// `staging` are refused. The rows' refusals ([`RowFormat::decode`]'s)
    /// are made before anything is enqueued. The copies are from pageable
    /// memory: `raw` is free once this returns.
    pub fn enqueue(
        &self,
        stream: &CudaStream,
        (format, laid, raw): (&RowFormat, Laid, &[u8]),
        staging: &mut DeviceBuffer<u8>,
        e: &mut DeviceBuffer<f32>,
        at: usize,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "rows enqueue";
        let values = format.values(laid, raw.len())?;
        if at.checked_add(values).is_none_or(|end| end > e.len()) {
            return Err(GpuError::shape(
                WHAT,
                format!("{values} values from {at} into {} decoded", e.len()),
            ));
        }
        format.check(raw, laid)?;
        if raw.is_empty() {
            return Ok(());
        }
        let (unit, per) = format.unit();
        let step = match laid {
            Laid::Planes if raw.len() > staging.len() => {
                return Err(GpuError::shape(
                    WHAT,
                    format!("{} B of planes into {} B staged", raw.len(), staging.len()),
                ));
            }
            Laid::Planes => raw.len(),
            Laid::Rows => staging.len() / unit * unit,
        };
        if step == 0 {
            return Err(GpuError::shape(
                WHAT,
                format!("{unit}-byte units into {} B staged", staging.len()),
            ));
        }
        for (k, part) in raw.chunks(step).enumerate() {
            let units = part.len() / unit;
            let from = launch_u32(WHAT, "at", at + k * step / unit * per)?;
            let grid = launch_u32(WHAT, "grid", units.div_ceil(BLOCK as usize))?;
            let n = launch_u32(WHAT, "units", units)?;
            WindowMut::<u8>::of_mut(staging, 0, part.len())?.copy_from_host(stream, part)?;
            let cfg = LaunchConfig1D::new(grid, BLOCK, 0);
            match (format, laid) {
                (RowFormat::Nvfp4(fp4), Laid::Planes) => {
                    let prep = self.module.prepare_rows_nvfp4_planes(cfg)?;
                    self.module.rows_nvfp4_planes(
                        stream,
                        &prep,
                        staging,
                        n,
                        fp4.scale(),
                        from,
                        e,
                    )?;
                }
                (RowFormat::Nvfp4(fp4), Laid::Rows) => {
                    let prep = self.module.prepare_rows_nvfp4(cfg)?;
                    self.module
                        .rows_nvfp4(stream, &prep, staging, n, fp4.scale(), from, e)?;
                }
                (RowFormat::Iq4Nl, _) => {
                    let prep = self.module.prepare_rows_iq4nl(cfg)?;
                    self.module.rows_iq4nl(stream, &prep, staging, n, from, e)?;
                }
                (RowFormat::F8(f8), _) => {
                    let prep = self.module.prepare_rows_f8(cfg)?;
                    self.module
                        .rows_f8(stream, &prep, staging, n, f8.scale(), from, e)?;
                }
            }
        }
        Ok(())
    }
}

/// A row table's reads from the NVMe tier: engram's prefetcher over the
/// table, whose helper advises a read's rows and copies them, so the calling
/// thread takes neither the advice nor a fault; a large read in parts over
/// the worker pool's threads ([`HelperOptions::spread`]). A prompt call's
/// later reads wait in order ([`RowRead::read_ahead`]); each fill hands the
/// helper the next while its own rows are decoded and computed. A fill that
/// waits for its rows at once and finds none read ahead reads them with
/// [`Prefetcher::read_now`]: on the calling thread when the helper would
/// read them alone, so no thread wakes for them.
pub(crate) struct RowRead {
    rows: Prefetcher,
    laid: Laid,
    /// A prompt call's reads not yet filled, in order; while `flight` holds,
    /// the front's rows are on the helper.
    pub(crate) ahead: VecDeque<RowBatch>,
    /// The row ids of the batch last taken off `ahead`, the storage the next
    /// read ahead fills first: a decode step's read ahead allocates nothing.
    pub(crate) spare_ids: Vec<u32>,
    pub(crate) flight: bool,
    /// The started fill's rows already copied ([`RowRead::read_now`]).
    read: bool,
}

/// One read of a prompt call as [`RowRead::read_ahead`] holds it: its first
/// position, its positions and its row ids.
pub(crate) struct RowBatch {
    pos: u32,
    n: usize,
    pub(crate) ids: Vec<u32>,
}

impl RowRead {
    /// The reader of `file`'s table `name` (`rows` rows), or of its rows
    /// where `stored` holds them (a checkpoint folder's value plane and
    /// optional block-scale plane, [`RunTable`]), for reads of up to
    /// `max_rows` rows, `step_rows` a one-position read; its helper on the
    /// SMT sibling of the loading thread's core when that thread is pinned,
    /// else floating. The calling thread's waits for rows go to
    /// [`crate::waits::ple`].
    pub(crate) fn open(
        file: &Arc<Split>,
        stored: Option<&StoredRows>,
        name: &str,
        (max_rows, step_rows, rows): (usize, usize, usize),
    ) -> Result<RowRead, GpuError> {
        let table: Arc<dyn RowTables> = match stored {
            None => Arc::new(
                SplitTable::open(Arc::clone(file), name).map_err(|e| GpuError::plan(WHAT, e))?,
            ),
            Some(s) => {
                let open = |p: &model::part::StoredPlane| {
                    let table = RunTable::open(name, p.row_bytes, &p.runs, PageAdvice::Random)
                        .map_err(|e| GpuError::plan(WHAT, e))?;
                    if table.rows() != rows as u64 {
                        return Err(GpuError::shape(
                            WHAT,
                            format!(
                                "{name}: source plane has {} rows, the table {rows}",
                                table.rows()
                            ),
                        ));
                    }
                    Ok(table)
                };
                let values = open(&s.values)?;
                match &s.block_scales {
                    None => Arc::new(values),
                    Some(scales) => Arc::new(vec![values, open(scales)?]),
                }
            }
        };
        let laid = match table.count() {
            1 => Laid::Rows,
            _ => Laid::Planes,
        };
        let planes = table.count();
        let reader = Prefetcher::over(
            table,
            &[max_rows; 2][..planes],
            HelperOptions::spread(FillMode::Touch, caller_sibling(), false),
        )
        .map_err(|e| GpuError::plan(WHAT, e))?;
        Ok(RowRead {
            rows: reader,
            laid,
            ahead: VecDeque::with_capacity(1),
            spare_ids: Vec::with_capacity(step_rows),
            flight: false,
            read: false,
        })
    }

    /// How the reader's copies lay the rows.
    #[must_use]
    pub(crate) fn laid(&self) -> Laid {
        self.laid
    }

    /// The planes a read asks for: one id list a plane.
    fn planes(&self) -> usize {
        match self.laid {
            Laid::Rows => 1,
            Laid::Planes => 2,
        }
    }

    /// The helper's read taken back, the calling thread's wait timed.
    fn wait(&mut self) -> Result<(), GpuError> {
        let t = Instant::now();
        let r = self.rows.wait();
        crate::waits::ple(t.elapsed());
        r.map_err(|e| GpuError::plan(WHAT, e))
    }

    /// The batch on the helper taken back unread: a call that stopped
    /// before its fills took it, or a new prompt call.
    fn drain(&mut self) -> Result<(), GpuError> {
        if std::mem::take(&mut self.flight) {
            self.wait()?;
        }
        Ok(())
    }

    /// The front batch off `ahead`, its row-id storage kept for the next
    /// read ahead.
    fn recycle_front(&mut self) {
        if let Some(batch) = self.ahead.pop_front() {
            self.spare_ids = batch.ids;
        }
    }

    /// Every read held ahead dropped, the one on the helper drained first:
    /// a new prompt call's reads start from none.
    pub(crate) fn restart(&mut self) -> Result<(), GpuError> {
        self.drain()?;
        self.recycle_front();
        self.ahead.clear();
        Ok(())
    }

    /// A read held ahead, `n` positions at `pos`, its row ids written by
    /// `name` into storage of `len` ids (the spare kept from the last read
    /// when there is one). A read `name` refuses drops every read held ahead:
    /// its fill names it again and refuses it by name.
    pub(crate) fn read_ahead(
        &mut self,
        (pos, n, len): (u32, usize, usize),
        name: impl FnOnce(&mut [u32]) -> bool,
    ) -> bool {
        let mut ids = std::mem::take(&mut self.spare_ids);
        ids.resize(len, 0);
        if !name(&mut ids) {
            self.ahead.clear();
            return false;
        }
        self.ahead.push_back(RowBatch { pos, n, ids });
        true
    }

    /// The front read held ahead on the helper, when there is one.
    pub(crate) fn start_front(&mut self) -> Result<(), GpuError> {
        if let Some(next) = self.ahead.front() {
            self.rows
                .submit(std::iter::repeat_n(next.ids.as_slice(), self.planes()))
                .map_err(|e| GpuError::plan(WHAT, e))?;
            self.flight = true;
        }
        Ok(())
    }

    /// The fill of `n` positions at `pos` naming `ids` taken in: the read
    /// held ahead when it is this fill's, else every read held ahead
    /// dropped. Whether the fill's rows are then on the helper.
    fn take_in(&mut self, pos: u32, n: usize, ids: &[u32]) -> Result<bool, GpuError> {
        self.read = false;
        let own = self
            .ahead
            .front()
            .is_some_and(|b| b.pos == pos && b.n == n && b.ids == ids);
        if own {
            self.recycle_front();
        } else {
            self.restart()?;
        }
        Ok(self.flight)
    }

    /// `ids`, the rows of the fill of `n` positions at `pos`, on the helper:
    /// the read held ahead when it is this fill's, else submitted now.
    /// [`RowRead::finish`] takes them.
    pub(crate) fn start(&mut self, pos: u32, n: usize, ids: &[u32]) -> Result<(), GpuError> {
        if !self.take_in(pos, n, ids)? {
            self.rows
                .submit(std::iter::repeat_n(ids, self.planes()))
                .map_err(|e| GpuError::plan(WHAT, e))?;
            self.flight = true;
        }
        Ok(())
    }

    /// [`RowRead::start`] for a fill that waits for its rows at once: the
    /// read held ahead when it is this fill's, else the rows read now
    /// ([`Prefetcher::read_now`]). [`RowRead::finish`] takes them.
    pub(crate) fn read_now(&mut self, pos: u32, n: usize, ids: &[u32]) -> Result<(), GpuError> {
        if !self.take_in(pos, n, ids)? {
            let planes = self.planes();
            let t = Instant::now();
            let r = self.rows.read_now(std::iter::repeat_n(ids, planes));
            crate::waits::ple(t.elapsed());
            r.map_err(|e| GpuError::plan(WHAT, e))?;
            self.read = true;
        }
        Ok(())
    }

    /// The started fill's rows ([`RowRead::start`], [`RowRead::read_now`])
    /// copied, returned as read; then the next read held ahead goes to the
    /// helper, to run while this one is decoded and computed. The bytes stay
    /// until the next read is taken.
    pub(crate) fn finish(&mut self) -> Result<&[u8], GpuError> {
        let (helper, read) = (
            std::mem::take(&mut self.flight),
            std::mem::take(&mut self.read),
        );
        if helper {
            self.wait()?;
        } else if !read {
            return Err(GpuError::state(
                WHAT,
                "a row read finished before it started",
            ));
        }
        self.start_front()?;
        Ok(self.rows.filled())
    }
}
