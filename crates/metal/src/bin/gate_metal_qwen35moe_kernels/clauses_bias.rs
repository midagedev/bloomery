//! The bias clause: an attention layer's q, k and v biases added to the P1
//! output's rows as the host's f32 add, the gap between strided rows left
//! alone, and the shapes and windows the launch refuses.

use bloomery_device::conformance::{Clause, Verdict, why};
use bloomery_device::{Buffer, Device, Elems};
use bloomery_metal::MetalBuffer;
use bloomery_metal::encode::ComputeEncoder;
use bloomery_metal::kernels::BiasRowsParams;

use super::Harness;
use super::common::{BadArg, refuse_launches, still};
use super::oracle;

/// What a gap between rows holds before the launch: no value of a row.
const GAP: f32 = 7.5;

/// `(rows, cols, stride)` of the launches held bit-equal: the step's one row
/// of a 7B's q, k and v, a pass's rows at a stride past the row, a row whose
/// float4s leave the last threadgroup part live, and the smallest row.
const SHAPES: [(usize, usize, usize); 4] =
    [(1, 4608, 4608), (5, 4608, 4864), (3, 1028, 1032), (1, 4, 4)];

/// `rows` rows of `cols` values `stride` apart, biased: the buffer after the
/// launch against the host's add, bit for bit, every gap value untouched.
fn biased(h: &Harness<'_>, (rows, cols, stride): (usize, usize, usize)) -> Result<(), String> {
    let (device, stream) = (h.device, h.stream);
    let seed = (rows * 31 + cols) as u64;
    let bias = oracle::activations(cols, seed, 3.0);
    let mut y = vec![GAP; rows * stride];
    for r in 0..rows {
        let row = oracle::activations(cols, seed + 1 + r as u64, 5.0);
        y[r * stride..r * stride + cols].copy_from_slice(&row);
    }
    let mut y_buffer = device.upload(stream, &y).map_err(why)?;
    // Values past the bias's end, which a launch that reads past it adds into
    // the gap beside the row.
    let held: Vec<f32> = bias.iter().copied().chain([1000f32; 8]).collect();
    let bias_buffer = device.upload(stream, &held).map_err(why)?;
    let extent = (rows - 1) * stride + cols;
    h.encode(|encoder| {
        h.kernels.bias_rows(
            encoder,
            BiasRowsParams {
                rows,
                cols,
                y: y_buffer.window_mut(Elems(0), Elems(extent))?,
                y_stride: stride,
                bias: bias_buffer.window(Elems(0), Elems(cols))?,
            },
        )
    })?;
    let mut got = vec![0f32; rows * stride];
    device
        .read(stream, y_buffer.whole().map_err(why)?, &mut got)
        .map_err(why)?;
    for (at, (got, before)) in got.iter().zip(&y).enumerate() {
        let (r, i) = (at / stride, at % stride);
        let want = if i < cols { before + bias[i] } else { *before };
        if got.to_bits() != want.to_bits() {
            return Err(format!(
                "{rows} rows of {cols} at stride {stride}: row {r} value {i} is {got} ({:#010x}), the host's add {want} ({:#010x})",
                got.to_bits(),
                want.to_bits()
            ));
        }
    }
    Ok(())
}

/// The buffers of the refusal cases: rows the launch must leave alone, and
/// the bias.
struct Bufs {
    y: MetalBuffer<f32>,
    bias: MetalBuffer<f32>,
}

/// A launch of the refusal cases: its shape and where its two windows lie.
#[derive(Clone, Copy)]
struct Launch {
    rows: usize,
    cols: usize,
    stride: usize,
    y_at: usize,
    y_len: usize,
    bias_at: usize,
    bias_len: usize,
}

/// The well-formed launch every case changes one thing of: 2 rows of 16
/// values.
const WELL_FORMED: Launch = Launch {
    rows: 2,
    cols: 16,
    stride: 16,
    y_at: 0,
    y_len: 32,
    bias_at: 0,
    bias_len: 16,
};

/// The case `what`: `launch` refused by name with `needle`.
fn refusal<'a>(
    h: &'a Harness<'_>,
    what: &'static str,
    needle: &'static str,
    launch: Launch,
) -> BadArg<'a, Bufs> {
    BadArg::new(
        what,
        needle,
        Box::new(move |b: &mut Bufs, encoder: &mut ComputeEncoder<'_>| {
            h.kernels.bias_rows(
                encoder,
                BiasRowsParams {
                    rows: launch.rows,
                    cols: launch.cols,
                    y: b.y.window_mut(Elems(launch.y_at), Elems(launch.y_len))?,
                    y_stride: launch.stride,
                    bias: b
                        .bias
                        .window(Elems(launch.bias_at), Elems(launch.bias_len))?,
                },
            )
        }),
    )
}

/// Every launch the kernel does not take is refused by name and writes
/// nothing: no rows, a row of values that are not whole float4s, a stride
/// shorter than the row or not whole float4s, a `y` window shorter than the
/// rows, a bias of another length, and a window off a 16-byte boundary.
fn refusals(h: &Harness<'_>) -> Result<usize, String> {
    let (device, stream) = (h.device, h.stream);
    let mut bufs = Bufs {
        y: device.upload(stream, &[1f32; 64]).map_err(why)?,
        bias: device.upload(stream, &[1f32; 64]).map_err(why)?,
    };
    let w = WELL_FORMED;
    let cases = vec![
        refusal(h, "no rows", "is empty", Launch { rows: 0, ..w }),
        refusal(
            h,
            "a row of ten values",
            "float4s",
            Launch {
                cols: 10,
                bias_len: 10,
                ..w
            },
        ),
        refusal(
            h,
            "a stride shorter than the row",
            "stride",
            Launch { stride: 8, ..w },
        ),
        refusal(
            h,
            "a stride of whole floats",
            "float4s",
            Launch {
                stride: 18,
                y_len: 34,
                ..w
            },
        ),
        refusal(
            h,
            "a y window one value short",
            "y window holds",
            Launch { y_len: 31, ..w },
        ),
        refusal(
            h,
            "a bias window of another length",
            "bias window holds",
            Launch { bias_len: 12, ..w },
        ),
        refusal(
            h,
            "a y window at the second value",
            "y window starts",
            Launch { y_at: 1, ..w },
        ),
        refusal(
            h,
            "a bias window at the second value",
            "bias window starts",
            Launch { bias_at: 1, ..w },
        ),
    ];
    refuse_launches(h, "bias rows", &mut bufs, cases, |h, b| {
        still(h, &b.y, |v| *v == 1.0)
    })
}

pub fn bias_rows(h: &Harness<'_>) -> Clause {
    for shape in SHAPES {
        biased(h, shape)?;
    }
    let refused = refusals(h)?;
    Ok(Verdict::Noted(format!(
        "{} launches bit-equal to the host's add, {refused} refusals",
        SHAPES.len()
    )))
}
