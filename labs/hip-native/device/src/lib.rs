//! Rust AMDGPU operator probe. This is not a model engine or a serving seat.
#![no_std]
#![feature(abi_gpu_kernel, link_llvm_intrinsics, core_intrinsics)]
#![allow(internal_features)]

use core::panic::PanicInfo;

mod selected;

#[panic_handler]
fn panic(_: &PanicInfo<'_>) -> ! {
    loop {}
}

unsafe extern "C" {
    #[link_name = "llvm.amdgcn.workitem.id.x"]
    fn item_x() -> u32;
    #[link_name = "llvm.amdgcn.workgroup.id.x"]
    fn group_x() -> u32;
}

#[inline(always)]
fn index() -> usize {
    // SAFETY: these intrinsics are called only by AMDGPU kernels launched in 64-thread workgroups.
    unsafe { group_x() as usize * 64 + item_x() as usize }
}

#[inline(always)]
fn half(bits: u16) -> f32 {
    let sign = (u32::from(bits & 0x8000)) << 16;
    let exp = (bits >> 10) & 31;
    let mut frac = u32::from(bits & 1023);
    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign);
        }
        let mut e = 113_u32;
        while frac & 1024 == 0 {
            frac <<= 1;
            e -= 1;
        }
        return f32::from_bits(sign | (e << 23) | ((frac & 1023) << 13));
    }
    if exp == 31 {
        return f32::from_bits(sign | 0x7f800000 | (frac << 13));
    }
    f32::from_bits(sign | ((u32::from(exp) + 112) << 23) | (frac << 13))
}

#[inline(always)]
unsafe fn q4(block: *const u8, i: usize) -> f32 {
    // SAFETY: caller supplies a complete 144-byte Q4_K block and i < 256.
    unsafe {
        let d = half(u16::from_le_bytes([*block, *block.add(1)]));
        let dmin = half(u16::from_le_bytes([*block.add(2), *block.add(3)]));
        let s = block.add(4);
        let j = i / 32;
        let (scale, min) = if j < 4 {
            (*s.add(j) & 63, *s.add(j + 4) & 63)
        } else {
            (
                (*s.add(j + 4) & 15) | ((*s.add(j - 4) >> 6) << 4),
                (*s.add(j + 4) >> 4) | ((*s.add(j) >> 6) << 4),
            )
        };
        let packed = *block.add(16 + (i / 64) * 32 + i % 32);
        let code = if i % 64 < 32 {
            packed & 15
        } else {
            packed >> 4
        };
        core::intrinsics::fmaf32(code as f32, d * scale as f32, -(dmin * min as f32))
    }
}

/// Copies n f32 values. Host allocation sizes cover n and source/destination do not alias.
///
/// # Safety
/// Both pointers cover n values and the caller launches 64-thread workgroups with distinct destination owners.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn bloomery_hip_copy(src: *const f32, dst: *mut f32, n: u32) {
    let i = index();
    if i < n as usize {
        // SAFETY: host checks allocation bounds; each workitem owns dst[i].
        unsafe {
            *dst.add(i) = *src.add(i);
        }
    }
}

/// Q4_K matrix times F32 activation columns. The host validates sizes and K is a positive multiple of 256.
///
/// # Safety
/// Weights cover rows*(K/256)*144 bytes, x covers cols*K values, y covers rows*cols values, and output does not alias input.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn bloomery_hip_q4_matmul(
    weights: *const u8,
    x: *const f32,
    y: *mut f32,
    k: u32,
    rows: u32,
    cols: u32,
) {
    let i = index();
    if i >= rows as usize * cols as usize {
        return;
    }
    let row = i % rows as usize;
    let col = i / rows as usize;
    let row_bytes = k as usize / 256 * 144;
    let mut sum = 0.0_f32;
    let mut j = 0_usize;
    // SAFETY: the host checked weights rows*row_bytes, x cols*k and y rows*cols; each workitem owns y[i].
    unsafe {
        while j < k as usize {
            let w = q4(weights.add(row * row_bytes + j / 256 * 144), j % 256);
            sum += w * *x.add(col * k as usize + j);
            j += 1;
        }
        *y.add(i) = sum;
    }
}

/// RMS norm of n values with one weight per value. Host buffers cover n and the epsilon is positive.
///
/// # Safety
/// All three pointers cover n values, output does not alias input and the caller launches 64-thread workgroups.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn bloomery_hip_rms_norm(
    x: *const f32,
    weights: *const f32,
    y: *mut f32,
    n: u32,
    eps: f32,
) {
    let i = index();
    if i >= n as usize {
        return;
    }
    let mut sum = 0.0_f32;
    let mut j = 0_usize;
    // SAFETY: host buffers cover n; each workitem owns one y value.
    unsafe {
        while j < n as usize {
            let v = *x.add(j);
            sum += v * v;
            j += 1;
        }
        let inverse = 1.0 / core::intrinsics::sqrtf32(sum / n as f32 + eps);
        *y.add(i) = *x.add(i) * inverse * *weights.add(i);
    }
}

/// Elementwise SwiGLU. The host buffers cover n and the two inputs do not alias output.
///
/// # Safety
/// All pointers cover n values and the caller launches 64-thread workgroups with a unique output owner per value.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn bloomery_hip_swiglu(
    gate: *const f32,
    up: *const f32,
    y: *mut f32,
    n: u32,
) {
    let i = index();
    if i < n as usize {
        // SAFETY: host checked n; this workitem owns y[i]. exp2 implements the device exponential.
        unsafe {
            let g = *gate.add(i);
            let exp = core::intrinsics::exp2f32(-g * core::f32::consts::LOG2_E);
            *y.add(i) = (g / (1.0 + exp)) * *up.add(i);
        }
    }
}

/// Adds n residual values. Each workitem updates one distinct output value.
///
/// # Safety
/// Both pointers cover n values, do not alias, and the caller launches 64-thread workgroups.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn bloomery_hip_residual(x: *const f32, y: *mut f32, n: u32) {
    let i = index();
    if i < n as usize {
        // SAFETY: both buffers hold n values; this workitem alone writes y[i].
        unsafe {
            *y.add(i) += *x.add(i);
        }
    }
}
