# How bloomery uses cuda-oxide

bloomery is an LLM inference engine for MoE models that splits the experts between GPU and CPU. Its GPU kernels are Rust, compiled with [cuda-oxide](https://github.com/NVIDIA/cuda-rust). This page shows how a larger project is put together on it: how the kernels are laid out, how the host launches them and how the build is pinned. It is written for people who have finished a first cuda-oxide kernel and want to see the next step.

bloomery is still in active development, and this page describes the tree as it is now. Everything here runs on Ampere (sm_86) only.

## Size

The device code lives in three engine crates (`gpu`, `gpu-deepseek41`, `gpu-vision`): 69 `#[cuda_module]` modules with 274 `#[kernel]` entries, plus a few test kernels in `gpu-gates`. The kernels include quantized matrix-vector and matrix-matrix products (k-quants, int8 tensor-core GEMM), flash attention for decode and prefill, MoE routing, RoPE and norms. Every one of them is Rust.

## Layout

| Crate | Holds |
|---|---|
| `crates/gpu` | kernels shared by every model, the device-callable cores (`cores.rs`), CUDA graphs, the fault word |
| `crates/gpu-deepseek41` | kernels only DeepSeek-V4.1 runs |
| `crates/gpu-vision` | the vision encoder's kernels |
| `crates/gpu-gates` | the binaries and the GPU gates (tests) |

Three rules came from experience:

- **One `#[cuda_module]` per operation file.** For example, `rope_neox.rs` holds the module `rope_neox_kernels`, its launch arguments and its host launcher. Writing a new op touches its own file only.
- **Model-specific kernels live in their own crate.** A build that does not need them does not compile them.
- **Every kernel name carries a model prefix** (`ds41_…`, `dflash_…`). cuda-oxide derives a kernel's host symbol from the entry name alone, not from its crate or module, so two crates that declare the same entry name fail to link.

Arithmetic that several kernels share is written once, as `#[inline(always)] pub fn` in `crates/gpu/src/cores.rs` (for example `q3k_row_dot`). A kernel in another crate calls it as `bloomery_gpu::cores::q3k_row_dot`, and it inlines into the same instructions.

## A kernel

This is `head_norm_neox_append` from `crates/gpu/src/rope_neox.rs`, trimmed. It normalizes one attention head, applies RoPE and appends the key and value to the cache, one block per (token, head).

```rust
#[cuda_module]
mod rope_neox_kernels {
    #[kernel]
    #[launch_bounds(64)]
    #[launch_contract(
        domain = 1,
        block = (64, 1, 1),
        requires = (
            q.len() >= m * n_head * 128,
            table.len() >= ctx * 128,
            pos.len() >= m,
            cache_k.len() >= n_kv * ctx * 128,
            // …one line per buffer
        )
    )]
    pub fn head_norm_neox_append(
        gq: &[f32], gk: &[f32], table: &[f32], pos: &[u32], v: &[f32],
        eps: f32, n_head: u32, n_kv: u32, ctx: u32, m: u32,
        fault: FaultSink,
        mut q: DisjointSlice<f32>,
        mut k: DisjointSlice<f32>,
        mut cache_k: DisjointSlice<u16>,
        mut cache_v: DisjointSlice<u16>,
    ) {
        static mut WSUM: SharedArray<f64, 2> = SharedArray::UNINIT;
        let b = thread::blockIdx_x() as usize;
        // …
        // SAFETY: t < m <= pos.len() by the launch contract.
        let p = unsafe { *pos.get_unchecked(t) } as usize;
        // …
        let mut acc = f64::from(x0 * x0) + f64::from(x1 * x1);
        acc += warp::shuffle_xor_f64(acc, 16);
        // … down to 1, then one slot per warp in WSUM and a barrier
        thread::sync_threads();
        // …
    }
}
```

What we use and why:

- **`#[launch_contract(requires = …)]` does the bounds proof.** It lists the length of every buffer as a product of the kernel's scalar arguments. The host checks it before the launch, so the body can use `get_unchecked`. Each `unsafe` block has a `// SAFETY:` comment that points at the contract line it relies on.
- **`#[launch_bounds]` matches the block size.** It caps the registers ptxas may use for that block size.
- **Outputs are `DisjointSlice`.** Each thread writes only its own positions, and the SAFETY comment says which ones.
- **Shared memory is a `static mut SharedArray`**, reached through `SharedArray::as_raw_mut_ptr`.
- **Warp reductions use `cuda_device::warp`.** Here the sum of squares is taken in f64 to match the reference implementation's order.

`requires` accepts only `+ - *`, so a count that needs a division or a ceiling is passed in as its own scalar argument. `crates/gpu/src/lib.rs` explains how `n_sb`, `half_it` and `quad_it` do this for the quantized products.

## Launching it

The host side of each module is a small struct that owns the loaded module. It does not own a stream: every call takes the engine's stream.

```rust
pub struct RopeNeoxKernels {
    module: rope_neox_kernels::LoadedModule,
}

impl RopeNeoxKernels {
    pub fn load(ctx: &Arc<CudaContext>) -> Result<RopeNeoxKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { rope_neox_kernels::load(ctx)? };
        Ok(RopeNeoxKernels { module })
    }

    pub fn enqueue_head_norm_neox_append(
        &self, stream: &CudaStream, args: NeoxArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_head_norm_neox_append";
        // 1. our own checks first, so an error names the argument
        //    ("cache_k.len() 1024 < 4096") and the entry point
        // 2. usize -> u32 without `as`: a value that does not fit is an error
        let grid = launch_u32(what, "grid", m * (n_head + n_kv))?;
        // 3. prepare checks the launch contract, then the launch is enqueued
        let prep = self
            .module
            .prepare_head_norm_neox_append(LaunchConfig1D::new(grid, THREADS_U32, 0))?;
        self.module.head_norm_neox_append(
            stream, &prep, gq, gk, table, pos, v, eps, n_head, n_kv, ctx, m, fault, q, k,
            cache_k, cache_v,
        )?;
        Ok(())
    }
}
```

Modules are loaded once, when the model loads. The launchers allocate nothing, so a whole decode step can be captured into a CUDA graph and replayed. `crates/gpu/src/graph.rs` does capture, instantiate and launch through the driver bindings in `cuda_core::sys`.

## When a kernel meets bad input

A kernel cannot panic, and we do not want it to write a plausible value either. Every kernel that can meet undefined input (a NaN activation, a position past the cache) takes a `FaultSink` argument: a device address plus the layer number, passed by value. On bad input the kernel raises a fault code, keeps its memory accesses in bounds and writes NaN instead of a plausible result. The output head copies the fault word next to the token it writes, so the step's single readback carries it for free. The host then returns a named error. See `crates/gpu/src/fault.rs`.

The raise itself is a few lines of `ptx_asm!` (`red.relaxed.gpu.global.min.u32`), which names the global address space directly.

## Build and pinning

- **Only `cargo oxide` builds device crates.** A plain `cargo build` of such a crate compiles, but produces a binary without the device code.
- **cuda-oxide is pinned by git revision.** `Cargo.toml` declares the upstream revision (the URL still reads NVlabs/cuda-oxide, which GitHub redirects to [NVIDIA/cuda-rust](https://github.com/NVIDIA/cuda-rust), the repository's new name), and a `[patch]` section takes the source from [our fork](https://github.com/midagedev/cuda-oxide) (`bloomery` branch), which is that revision plus a few small patches on their way upstream (`THIRD_PARTY_NOTICES.md` lists them). `just deny` fails if either revision floats. The Rust nightly moves only when this pin moves.
- **One codegen backend per revision.** On the build machine each revision's backend sits in its own directory with a `source-rev.txt`. `tools/box.sh` sets `CUDA_OXIDE_BACKEND` to it and stops when the file is missing or names another revision.
- **Host code gets the same CPU flags as a plain cargo build.** `cargo oxide` sets its own `CARGO_ENCODED_RUSTFLAGS`, which hides `.cargo/config.toml`. So `.cargo/cuda-oxide.toml` repeats `-C target-cpu=znver3` in `extra-rustflags`, and `just check-rustflags` fails when the two files disagree.

## Checking the compiled code

Register use and spills do not change any output bit, only speed, so the bit-exact tests cannot see them. We check them at build time:

- **`just ptx-scan <bin>`** prints, for every kernel entry, its registers, shared memory, ptxas spill bytes and the driver JIT's local bytes, with an md5 of its PTX. A change that should not touch device code (a rename, host-only work) must leave this table identical.
- **`just gate-ptx-spill`** compares every entry's spill and local bytes with the pins in `tools/ref/ptx-shapes.tsv`. Any change, up or down, a new entry without a pin, or a pin whose entry disappeared is a failure. A non-zero pin carries a comment that gives its reason.

## Upstream

A few small fixes found along the way went upstream: [#1314](https://github.com/NVIDIA/cuda-rust/pull/1314) and [#1321](https://github.com/NVIDIA/cuda-rust/pull/1321) (merged), [#1329](https://github.com/NVIDIA/cuda-rust/pull/1329) and [#1346](https://github.com/NVIDIA/cuda-rust/pull/1346) (open).

## Where to start reading

- `crates/gpu/src/rope_neox.rs`: one module, two kernels, a host launcher. Small enough to read in one sitting.
- `crates/gpu/src/cores.rs`: device functions shared across crates.
- `crates/gpu/src/fault.rs`: the fault word and its `ptx_asm!` raise.
- `crates/gpu/src/graph.rs`: graph capture and replay through the driver API.
- `crates/gpu-deepseek41/src/lib.rs`: why a model's kernels get their own crate.
- [`docs/BUILD.md`](BUILD.md): toolchain versions and build commands.

Questions and corrections are welcome as issues.
