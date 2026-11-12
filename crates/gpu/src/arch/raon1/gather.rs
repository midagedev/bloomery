//! The row gathers a walk makes from a resident table: the token rows the
//! streams hold a copy of a stream each, and the per-layer input's table
//! rows ([`super::ple`]). One owner of the table's resident form — a
//! K-quant's words, which `ElemKernels::enqueue_embed_rows_kquant` decodes,
//! or the Q8_0 planes `Q38Kernels::enqueue_embed_rows` does — so a table of
//! either form gathers through one call, and a form neither reads is
//! refused by name.

use crate::DeviceTensor;
use crate::Gpu;
use crate::GpuError;
use crate::q38::{EmbedArgs, EmbedRows, Q38Kernels};
use crate::weights::{DevWeight, Weights};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::quant::GgmlType;

/// A resident row table a gather reads.
pub(super) enum RowsTable<'a> {
    /// Q3_K, Q4_K, Q5_K or Q6_K words.
    KQuant {
        ty: GgmlType,
        w: &'a DeviceTensor<u32>,
    },
    /// The q8f32 code and scale planes.
    Q8_0 {
        qs: &'a DeviceTensor<u32>,
        d: &'a DeviceTensor<u16>,
    },
}

impl<'a> RowsTable<'a> {
    /// The resident table `name`: a K-quant or Q8_0 one, refused by name
    /// otherwise (an F32 or F16 table the load widens has no row decode).
    pub(super) fn of(
        w: &'a Weights,
        what: &'static str,
        name: &str,
    ) -> Result<RowsTable<'a>, GpuError> {
        match w.get(name) {
            Some(DevWeight::KQuant { ty, w, .. }) => Ok(RowsTable::KQuant { ty: *ty, w }),
            Some(DevWeight::Q8_0 { qs, d, .. }) => Ok(RowsTable::Q8_0 { qs, d }),
            Some(_) => Err(GpuError::tensor(
                what,
                name,
                "a K-quant or Q8_0 table; no row decode reads another form",
            )),
            None => Err(GpuError::tensor(what, name, "resident")),
        }
    }
}

/// The `ids.len()` rows of `table` into `y` (a row's width each,
/// token-major), their positions and live key counts into `pos` and
/// `n_keys`, `pos0` the first position of the input the ids are a window
/// of. Asynchronous, allocation-free, capturable.
pub(super) fn enqueue_rows(
    gpu: &Gpu,
    kq: &Q38Kernels,
    table: &RowsTable<'_>,
    (ids, pos0): (&DeviceBuffer<u32>, &DeviceBuffer<u32>),
    (y, pos, n_keys): (
        &mut DeviceBuffer<f32>,
        &mut DeviceBuffer<u32>,
        &mut DeviceBuffer<u32>,
    ),
) -> Result<(), GpuError> {
    let stream: &CudaStream = gpu.stream();
    match table {
        RowsTable::KQuant { ty, w } => gpu.elem().enqueue_embed_rows_kquant(
            stream,
            *ty,
            crate::elem::EmbedRowsArgs {
                w,
                ids,
                pos0,
                first: 0,
                y,
                pos,
                n_keys,
            },
        ),
        RowsTable::Q8_0 { qs, d } => kq.enqueue_embed_rows(
            stream,
            EmbedArgs {
                table: EmbedRows::Q8_0 { qs, d },
                ids,
                pos0,
                first: 0,
                fault: gpu.unlabelled_sink(),
                y,
                pos,
                n_keys,
            },
        ),
    }
}
