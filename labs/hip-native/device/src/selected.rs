//! Original Q4_K lane walk with an AMD wave32 and integer-dot shim.
use core::sync::atomic::{AtomicU32, Ordering};

unsafe extern "C" {
    #[link_name = "llvm.amdgcn.ds.bpermute"]
    fn bpermute(index: u32, word: u32) -> u32;
    #[link_name = "llvm.amdgcn.sdot4"]
    fn sdot4(a: u32, b: u32, c: i32, clamp: bool) -> i32;
    #[link_name = "llvm.amdgcn.wavefrontsize"]
    fn wavefront_size() -> u32;
}

#[inline(always)]
fn dp4a_s32(a: u32, b: u32, c: i32) -> i32 {
    // SAFETY: four signed bytes per input, no saturation, the original dp4a contract.
    unsafe { sdot4(a, b, c, false) }
}

#[inline(always)]
fn add_rn_f32(a: f32, b: f32) -> f32 {
    a + b
}

#[inline(always)]
fn mul_rn_f32(a: f32, b: f32) -> f32 {
    a * b
}

#[inline(always)]
fn fma_rn_f32(a: f32, b: f32, c: f32) -> f32 {
    core::intrinsics::fmaf32(a, b, c)
}

#[inline(always)]
fn q4k_coeff(d: f32, dmin: f32, sc: i32, mi: i32) -> (f32, f32) {
    // CUDA contracts this coefficient; qdot::card_q4k_lane_walk spells its FMA.
    (
        d * sc as f32,
        fma_rn_f32(8.0 * d, sc as f32, -(dmin * mi as f32)),
    )
}

include!(concat!(env!("OUT_DIR"), "/q4k_core.rs"));

#[inline(always)]
fn warp_sum(mut value: f32, lane: usize) -> f32 {
    for stride in [16, 8, 4, 2, 1] {
        // SAFETY: the launch is wave32 and every lane of this row executes the same exchange.
        let partner = unsafe { bpermute(((lane ^ stride) * 4) as u32, value.to_bits()) };
        value = add_rn_f32(value, f32::from_bits(partner));
    }
    value
}

#[inline(always)]
unsafe fn expert_fault(fault: *mut u32, layer: u32) {
    // SAFETY: caller owns the aligned global fault allocation through 1 + layer.
    unsafe {
        (&*fault.cast::<AtomicU32>()).fetch_min((layer << 8) | 7, Ordering::Relaxed);
        (&*fault.add(1 + layer as usize).cast::<AtomicU32>()).fetch_or(1 << 7, Ordering::Relaxed);
    }
}

/// Reports the compiled wave width before the host permits a selected launch.
///
/// # Safety
/// `out` holds one u32; the host launches a single thread.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn bloomery_hip_wave_size(out: *mut u32) {
    // SAFETY: this single thread owns the one-word output.
    unsafe { *out = wavefront_size() };
}

/// Selected Q4_K down: the original row walk, Q8 planes and 256-thread launch.
///
/// # Safety
/// Host validates positive dimensions, n_sb <= 80, iters=ceil(n_sb/4), weight words
/// experts*rows*36*n_sb, q slots*256*iters, s8 slots*8*n_sb, d8 slots*2*n_sb,
/// ids slots, y slots*rows, and fault 65537 words with layer <= 65535. All pointers
/// are aligned and live, y/fault do not alias input, each block has 256 threads,
/// and the compiled wave width is 32, checked by the host before this launch.
#[allow(
    clippy::too_many_arguments,
    reason = "kernel entry: the original device ABI takes flat arguments (rust-quality R8)"
)]
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn bloomery_hip_q4_sel(
    w: *const u32,
    q: *const u32,
    s8: *const i32,
    d8: *const f32,
    sel: *const u32,
    n_experts: u32,
    rows_per_expert: u32,
    n_slots: u32,
    n_sb: u32,
    iters: u32,
    fault: *mut u32,
    layer: u32,
    y: *mut f32,
) {
    // SAFETY: these register intrinsics run in the specified 256-thread AMDGPU workgroup.
    let (t, block) = unsafe { (super::item_x() as usize, super::group_x() as usize) };
    let row = block * 8 + t / 32;
    if row >= n_slots as usize * rows_per_expert as usize {
        return;
    }
    let lane = t & 31;
    let slot = row / rows_per_expert as usize;
    // SAFETY: the row guard bounds slot; every lane in this wave reads the same id.
    let id = unsafe { *sel.add(slot) };
    if id >= n_experts {
        if id != u32::MAX && lane == 0 {
            // SAFETY: the host checked the full fault buffer and layer range.
            unsafe { expert_fault(fault, layer) };
        }
        return;
    }
    let row_abs = id as usize * rows_per_expert as usize + row % rows_per_expert as usize;
    // SAFETY: host validated these allocation spans, and the id/row guards bound the core's row and column.
    let (w, q, s8, d8) = unsafe {
        (
            core::slice::from_raw_parts(
                w,
                n_experts as usize * rows_per_expert as usize * 36 * n_sb as usize,
            ),
            core::slice::from_raw_parts(q, n_slots as usize * 256 * iters as usize),
            core::slice::from_raw_parts(s8, n_slots as usize * 8 * n_sb as usize),
            core::slice::from_raw_parts(d8, n_slots as usize * 2 * n_sb as usize),
        )
    };
    let partial = q4k_row_dot_1col(w, q, s8, d8, n_sb as usize, iters, row_abs, slot, lane);
    let sum = warp_sum(partial, lane);
    if lane == 0 {
        // SAFETY: the row guard bounds y; only lane zero of its owning wave writes it.
        unsafe { *y.add(row) = sum };
    }
}
