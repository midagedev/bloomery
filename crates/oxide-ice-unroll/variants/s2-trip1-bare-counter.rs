// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal reproducer for the cuda-oxide `#[unroll]` ICE
//! (`APInt::shl: bitwidth mismatch`), cargo-oxide 0.2.1 @ b9847e95.
//!
//! **This crate intentionally does not compile.** That is the bug being
//! reproduced: a `#[unroll]`-annotated `while` loop whose counter is `usize`
//! (64-bit) and whose body shifts a `u32` value by a counter-derived amount.
//! Full unrolling materializes the counter as 64-bit `mir.constant` literals;
//! the mixed-width `mir.shl` (legal Rust: `u32 << usize`) then reaches the
//! constant folder, which calls `APInt::shl` on operands of different widths
//! and panics the codegen driver.
//!
//! Workaround: make the counter `u32` (see `variants/b-u32-counter.rs`) or
//! narrow the shift amount (`variants/e2-u32-amount.rs`). Every variant in
//! `variants/` replaces this file verbatim (`cp variants/<v>.rs src/main.rs`)
//! — see REPRO.md for the full intervention table and recorded outputs.
//!
//! Run (on the reproducing toolchain): `cargo oxide build --arch sm_86`

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
            let mut acc: u32 = 0;
            #[unroll]
            while i < 1 {
                acc |= 1u32 << i;
                i += 1;
            }
            *o = acc;
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
    // 1<<0 | 1<<2 | 1<<4 | 1<<6 == 0b01010101 == 1
    let errors = out_host.iter().filter(|&v| *v != 85).count();
    if errors == 0 {
        println!("PASSED: all {} elements == 1", N);
    } else {
        eprintln!("FAILED: {} wrong elements", errors);
        std::process::exit(1);
    }
    Ok(())
}
