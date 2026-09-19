// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal reproducer for the cuda-oxide `#[unroll]` device-codegen ICE
//! (`APInt::shl: bitwidth mismatch`), cargo-oxide 0.2.1 @ b9847e95.
//!
//! **This crate intentionally does not compile.** That is the bug being
//! reproduced: a `#[unroll]`-annotated `while` loop whose counter is `usize`
//! (64-bit) and whose body shifts a `u32` value by a counter-derived amount.
//! Rust legalizes mixed-width shifts (`u32 << usize`), and mir-lower's
//! `convert_shift` handles them — but the *constant folder* does not. The
//! unroll pass materializes the counter as 64-bit `mir.constant` literals
//! (full unroll) or lets its cleanup SCCP speculate the loop phi at its
//! initial constant (partial unroll), and the then-constant mixed-width
//! `mir.shl` reaches `MirShlOp::check_fold`
//! (`dialect-mir/src/const_fold.rs:200`), which calls `APInt::shl` on
//! operands of different widths and panics the codegen driver:
//!
//! ```text
//! error: [rustc_codegen_cuda] Internal compiler error in device codegen:
//! assertion `left == right` failed: APInt::shl: bitwidth mismatch (32 vs 64)
//! ```
//!
//! Workaround: make the counter `u32` (`variants/b-u32-counter.rs`, verified
//! to build AND run correctly on sm_86) or route the shift amount through an
//! `as` cast (`variants/e2-u32-amount.rs`) — the cast blocks constant
//! propagation, so the folder never sees both operands constant.
//!
//! Every variant in `variants/` replaces this file verbatim
//! (`cp variants/<v>.rs src/main.rs`); REPRO.md holds the full intervention
//! table with recorded outputs.
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
            #[unroll]
            while i < 1 {
                *o = 1u32 << i;
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
    let errors = out_host.iter().filter(|&v| *v != 1).count();
    if errors == 0 {
        println!("PASSED: all {} elements == 1", N);
    } else {
        eprintln!("FAILED: {} wrong elements", errors);
        std::process::exit(1);
    }
    Ok(())
}
