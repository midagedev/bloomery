//! What every tower's chain over card buffers does the same way: reading a weight tensor out of
//! the encoder file and uploading it, the tap sink a gate hands an encode to read intermediate
//! tensors, the rows an encode returns, and the branch GEMM a tap needs without its fused
//! epilogue. A tower's own file ([`crate::encoder`], [`crate::qwen3vl`]) holds what its
//! architecture forces: the order of its launches and the names of its taps.

use crate::gemm_bf16::{Epilogue, GemmArgs, GemmKernels};
use bloomery_gpu::{GpuError, TensorWindow, Window};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::{GgmlType, Gguf};

/// Where an encode's intermediate tensors go when a caller asks for them (a gate). `wants` is
/// asked before each named point; a `true` synchronizes the stream and hands the tensor over
/// as bf16 bits, `cols` values per row. The names are each tower's own (listed in its module
/// doc), the oracle's file stems or graph node names a gate maps them to.
pub trait TapSink {
    fn wants(&self, name: &str) -> bool;
    fn take(&mut self, name: &str, cols: usize, bits: Vec<u16>);
}

/// The sink of a plain encode.
pub(crate) struct NoTaps;

impl TapSink for NoTaps {
    fn wants(&self, _: &str) -> bool {
        false
    }
    fn take(&mut self, _: &str, _: usize, _: Vec<u16>) {}
}

/// An encode's result: the rows the text model reads in place of an image's tokens (`out_dim` bf16
/// values each, reading order), a window of the tower's own buffer that lives until its next
/// call, and the chain launches it took.
pub struct Encoded<'a> {
    pub rows: TensorWindow<'a, u16>,
    pub launches: usize,
}

/// Hand the first `rows` rows of `buf` (`cols` values each) to `taps` under `name` when it asks
/// for them.
pub(crate) fn tap(
    stream: &CudaStream,
    taps: &mut dyn TapSink,
    name: &str,
    buf: &DeviceBuffer<u16>,
    rows: usize,
    cols: usize,
) -> Result<(), GpuError> {
    if taps.wants(name) {
        let v = to_host(stream, buf, rows * cols)?;
        taps.take(name, cols, v);
    }
    Ok(())
}

/// The first `len` values of `buf`, after the stream's work so far.
pub(crate) fn to_host(
    stream: &CudaStream,
    buf: &DeviceBuffer<u16>,
    len: usize,
) -> Result<Vec<u16>, GpuError> {
    stream.synchronize()?;
    Ok(Window::<u16>::of(buf, 0, len)?.to_host_vec(stream)?)
}

/// A branch GEMM without its fused epilogue, for a tap only: `a · bᵀ (+ bias)` of shape
/// `(m, n, k)` into `c`, the buffer the fused launch after it overwrites in full, handed to
/// `taps` under `name`.
#[allow(
    clippy::type_complexity,
    reason = "the operand triple and the shape triple of one GEMM, named at the call"
)]
pub(crate) fn side(
    gemm: &GemmKernels,
    stream: &CudaStream,
    taps: &mut dyn TapSink,
    name: &str,
    (a, b, bias): (
        &DeviceBuffer<u16>,
        &DeviceBuffer<u16>,
        Option<&DeviceBuffer<f32>>,
    ),
    (m, n, k): (usize, usize, usize),
    c: &mut DeviceBuffer<u16>,
) -> Result<(), GpuError> {
    if !taps.wants(name) {
        return Ok(());
    }
    gemm.enqueue(
        stream,
        GemmArgs {
            a,
            b,
            bias,
            epilogue: Epilogue::None,
            resid: None,
            m,
            n,
            k,
            c,
        },
    )?;
    tap(stream, taps, name, c, m, n)
}

pub(crate) fn tensor_bytes<'a>(
    file: &'a Gguf,
    name: &str,
    what: &'static str,
) -> Result<&'a [u8], GpuError> {
    let t = file.find(name).ok_or_else(|| GpuError::Tensor {
        what,
        name: name.to_string(),
        need: "in the encoder file",
    })?;
    Ok(file.data(t)?)
}

pub(crate) fn up_bf16(
    stream: &CudaStream,
    file: &Gguf,
    name: &str,
    what: &'static str,
) -> Result<DeviceBuffer<u16>, GpuError> {
    Ok(DeviceBuffer::from_host(
        stream,
        &read_bf16(file, name, what)?,
    )?)
}

pub(crate) fn up_f32(
    stream: &CudaStream,
    file: &Gguf,
    name: &str,
    what: &'static str,
) -> Result<DeviceBuffer<f32>, GpuError> {
    Ok(DeviceBuffer::from_host(
        stream,
        &read_f32(file, name, what)?,
    )?)
}

pub(crate) fn read_bf16(file: &Gguf, name: &str, what: &'static str) -> Result<Vec<u16>, GpuError> {
    Ok(tensor_bytes(file, name, what)?
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes(*c))
        .collect())
}

pub(crate) fn read_f32(file: &Gguf, name: &str, what: &'static str) -> Result<Vec<f32>, GpuError> {
    Ok(tensor_bytes(file, name, what)?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

/// A matrix of the file as bf16 bits: a bf16 tensor as it is; an f32 or f16 one only when every
/// value round-trips through bf16 exactly (the converter keeps some kernels out of the narrow
/// type, and an f16 export's values are not generally bf16's), else refused naming the tensor
/// and the first value that does not.
pub(crate) fn read_bf16_narrowed(
    file: &Gguf,
    name: &str,
    what: &'static str,
) -> Result<Vec<u16>, GpuError> {
    let ty = file
        .find(name)
        .ok_or_else(|| GpuError::Tensor {
            what,
            name: name.to_string(),
            need: "in the encoder file",
        })?
        .ty;
    let values: Vec<f32> = match ty {
        GgmlType::BF16 => return read_bf16(file, name, what),
        GgmlType::F32 => read_f32(file, name, what)?,
        GgmlType::F16 => tensor_bytes(file, name, what)?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| gguf::quant::half_to_f32(u16::from_le_bytes(*c)))
            .collect(),
        other => {
            return Err(GpuError::Shape {
                what,
                detail: format!("tensor {name} is {other}; a matrix is bf16, f16 or f32"),
            });
        }
    };
    values
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let bits = crate::f32_bf16(v);
            if crate::bf16_f32(bits).to_bits() == v.to_bits() {
                Ok(bits)
            } else {
                Err(GpuError::Shape {
                    what,
                    detail: format!(
                        "tensor {name} value {v:e} at index {i} does not round-trip through bf16; the tower keeps its matrices in bf16"
                    ),
                })
            }
        })
        .collect()
}
