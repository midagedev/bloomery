use super::*;
use bloomery_gpu::Gpu;
use cuda_core::DeviceBuffer;
use gguf::quant::{f32_to_f16_bits, half_to_f32};

#[test]
#[ignore = "opens the card for the F32 audio primitives"]
fn hw_convolution_and_element_rules() -> Result<(), GpuError> {
    let gpu = Gpu::new()?;
    let stream = gpu.stream();
    let ops = Ops::load(&gpu)?;
    let up = |x: &[f32]| DeviceBuffer::from_host(stream, x);
    let mut y = DeviceBuffer::<f32>::zeroed(stream, 6)?;

    let x = up(&[-2.0, 1.0 / 3.0, 0.75, 2.0, f32::NAN, f32::INFINITY])?;
    let prep = ops.module.prepare_pointwise(config(6))?;
    ops.module
        .pointwise(stream, &prep, &x, 6, -0.5, 0.5, 1, &mut y)?;
    let got = y.to_host_vec(stream)?;
    assert_eq!(
        &got[..4],
        &[-0.5, 0.333_984_38, 0.5, 0.5],
        "clamp and BF16 rounding"
    );
    assert!(
        got[4].is_nan() && got[5].is_infinite(),
        "non-finite clamp input"
    );

    let mut x = up(&[-1.0, 0.0, 1.0])?;
    let prep = ops.module.prepare_unary(config(3))?;
    ops.module
        .unary(stream, &prep, 3, -f32::MAX, f32::MAX, 1, &mut x)?;
    let got = x.to_host_vec(stream)?;
    for (value, input) in got.iter().zip([-1.0_f32, 0.0, 1.0]) {
        assert!(
            (value - input / (1.0 + (-input).exp())).abs() < 2e-7,
            "SiLU"
        );
    }

    let x = up(&[1.0003; 9])?;
    let w = up(&[0.10003; 9])?;
    let bias = up(&[0.0])?;
    let mut out = DeviceBuffer::<f32>::zeroed(stream, 4)?;
    let prep = ops.module.prepare_conv2d(config(4))?;
    ops.module
        .conv2d(stream, &prep, &x, &w, &bias, 3, 3, 1, 1, 2, 2, &mut out)?;
    let a = half_to_f32(f32_to_f16_bits(1.0003));
    let b = half_to_f32(f32_to_f16_bits(0.10003));
    for value in out.to_host_vec(stream)? {
        assert!(
            (value - 4.0 * a * b).abs() < 1e-7,
            "F16 im2col and padded corners"
        );
    }

    let x = up(&[1.0, 3.0, f32::INFINITY, 1.0])?;
    let gain = up(&[2.0, 1.0])?;
    let prep = ops.module.prepare_channel_norm(config(2))?;
    ops.module
        .channel_norm(stream, &prep, &x, &gain, 2, 2, 1e-6, ops.sink, &mut out)?;
    let got = out.to_host_vec(stream)?;
    assert_eq!(got[0], 0.0, "channel ReLU");
    assert!(
        (got[1] - 1.0 / 1.000001_f32.sqrt()).abs() < 1e-7,
        "channel LayerNorm"
    );
    assert!(got[2].is_nan() && got[3].is_nan(), "non-finite channel row");
    assert_eq!(
        bloomery_gpu::fault::read(gpu.fault_word())?.and_then(|f| f.site()),
        Some(bloomery_gpu::FaultSite::F32Product)
    );
    gpu.clear_fault()?;

    let x = up(&[2.0, 3.0, 0.0, 3.0_f32.ln()])?;
    let prep = ops.module.prepare_glu(config(2))?;
    ops.module
        .glu(stream, &prep, &x, 1, 2, ops.sink, &mut out)?;
    let got = out.to_host_vec(stream)?;
    assert!(
        (got[0] - 1.0).abs() < 1e-7 && (got[1] - 2.25).abs() < 1e-6,
        "split GLU"
    );

    let x = up(&[1.0, 2.0, 3.0])?;
    let w = up(&[1.0, 2.0, 3.0, 4.0, 5.0])?;
    let bias = up(&[0.5])?;
    let mut out = DeviceBuffer::<f32>::zeroed(stream, 3)?;
    let prep = ops.module.prepare_depthwise(config(3))?;
    ops.module
        .depthwise(stream, &prep, &x, &w, &bias, 3, 1, 5, &mut out)?;
    assert_eq!(
        out.to_host_vec(stream)?,
        vec![5.5, 14.5, 26.5],
        "causal depthwise padding"
    );
    stream.synchronize()?;
    Ok(())
}
