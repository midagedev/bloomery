// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shrink probe s5: the minimal form without an accumulator — the shift
//! result is written straight to the output slice. See REPRO.md.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(domain = 1, block = (32, 1, 1))]
    pub fn ice(mut out: DisjointSlice<u32>) {
        let tid = thread::index_1d();
        if let Some(o) = out.get_mut(tid) {
            let mut i: usize = 0;
            #[unroll]
            while i < 1 {
                *o = ((i + 1) << 2u32) as u32;
                i += 1;
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    const N: usize = 32;
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;

    // SAFETY: this package owns the embedded device bundle produced for the
    // kernels module above.
    let module = unsafe { kernels::load(&ctx)? };
    let prepared = module.prepare_ice(LaunchConfig1D::new(1, 32, 0))?;
    module.ice(&stream, &prepared, &mut out_dev)?;

    let out_host = out_dev.to_host_vec(&stream)?;
    let errors = out_host.iter().filter(|&v| *v != 4).count();
    if errors == 0 {
        println!("PASSED: all {} elements == 4", N);
    } else {
        eprintln!("FAILED: {} wrong elements", errors);
        std::process::exit(1);
    }
    Ok(())
}
