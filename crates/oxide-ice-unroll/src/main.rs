use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// Minimal `#[unroll]` + `usize` counter reproducer: a `while` loop with a
    /// `usize` counter whose body shifts the counter (`i << 1`, where the
    /// literal `1` is `i32`, so the shift has mixed 64/32-bit operands).
    /// `out[tid]` would be the sum of `i << 1` for `i` in `0..8` (= 56).
    ///
    /// This kernel ICEs device codegen (see REPRO.md). The workaround is a
    /// `u32` counter (variant (b) in REPRO.md, verified on GPU).
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn unroll_usize(mut out: DisjointSlice<u32>) {
        let tid = thread::index_1d();
        if let Some(out_elem) = out.get_mut(tid) {
            let mut acc: u32 = 0;
            let mut i: usize = 0;
            #[unroll]
            while i < 8 {
                acc = acc.wrapping_add((i << 1) as u32);
                i += 1;
            }
            *out_elem = acc;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    const N: usize = 256;
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;

    // SAFETY: this package owns the embedded device bundle produced for the
    // kernels module above.
    let module = unsafe { kernels::load(&ctx)? };
    let prepared = module.prepare_unroll_usize(LaunchConfig1D::new(1, 256, 0))?;
    module.unroll_usize(&stream, &prepared, &mut out_dev)?;

    let out_host = out_dev.to_host_vec(&stream)?;
    let errors = (0..N).filter(|&i| out_host[i] != 56).count();

    if errors == 0 {
        println!("PASSED: all {} elements correct", N);
    } else {
        eprintln!("FAILED: {} errors", errors);
        std::process::exit(1);
    }
    Ok(())
}
