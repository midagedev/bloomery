# Native Rust HIP feasibility lab

This isolated lab compiles Rust kernels for `amdgcn-amd-amdhsa`, loads their ELF through HIP, and checks the scalar probes against Bloomery's GGUF dequantizer with FP64 accumulation and the selected Q4_K port against the existing card-rule scalar oracle. It calls no libllama, ggml model forward, or llama-server process. Both host and device code are Rust. The nested workspace is excluded from the default CUDA workspace.

The validation target is one RX 7900 GRE (`gfx1100`), Rust nightly 2026-08-28, its LLVM backend and the local HIP runtime. Rust's AMDGPU target is tier 3: this is a feasibility result, not a supported release backend. The original CUDA model chains, seats, kernels, scheduling and MTP remain unchanged.

## What runs

- Device-buffer allocation and synchronized HIP transfers, plus a Rust GPU copy kernel with exact roundtrip equality.
- A scalar Q4_K × F32 matrix kernel on synthetic shapes and real `blk.1.ssm_out.weight` rows from Qwen3.8-27B IQ3_S. Matrix rows use GGUF Q4_K's 144 bytes per 256 weights. Scales and fused dequantization follow `crates/gguf/src/quant.rs`.
- A 256-wide synthetic dense FFN unit: RMSNorm, separate gate/up projections, SwiGLU, down projection and residual. Its intermediate values stay on the GPU. It does not include attention, GDN state, routed experts or a complete transformer block.
- The original selected Q4_K down row walk with Q8 activation planes, 256-thread blocks and logical warp32 reduction. Host-side `qdot::card_q8_1_scalar` prepares the inputs; device quantization is a later step.
- Three repeated launches use already allocated buffers. The reported zero count covers allocations explicitly issued by this host, not internal allocations by the driver.

The scalar probes retain fixed NMSE <= 1e-6 and finite outputs. The selected Q4_K port requires bitwise equality with the existing `qdot::card_q4k_dot_row_scalar` oracle for finite activations, NaN propagation for refused activation scales, and exact fault words, masks and skipped-slot sentinels. This is host-oracle/HIP equality, not a measured CUDA/HIP or model-token comparison. No throughput is claimed.

## Verify

The lab owner is [Burntgogi](https://github.com/Burntgogi). Install the repository's pinned
nightly with rust-src, rustfmt and clippy, Python 3.11 or later, and a HIP runtime exposing
`libamdhip64.so`. The verified target is `gfx1100`; no other card is claimed.

From a clean, committed checkout:

```sh
export HIP_LIB_DIR=/opt/rocm/lib
export LD_LIBRARY_PATH="$HIP_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
labs/hip-native/verify.sh /tmp/hip-evidence
# Optional real Q4_K rows: exact packed bytes, K and row count.
labs/hip-native/verify.sh /tmp/hip-real-evidence \
  --q4-rows /path/to/q4_k_rows.bin --k 6144 --rows 64
```

The output directory must be new and outside the checkout. An SDK outside `/opt/rocm` may
need additional runtime library directories. `verify.sh` checks formatting, host tests,
host and device clippy with warnings denied, builds both release binaries, and runs the
numerical cases under a 900-second bound per command. It compiles the standalone Rust
AMDGPU device with `cargo -Zbuild-std=core`, never the existing CUDA device crates.

`result.json` records the clean HEAD commit, whole-repository and lab Git tree hashes,
release binary SHA-256 hashes, Rust/LLVM version, target CPU, actual device name, HIP runtime
and driver versions, fixture hash when present, command logs and each case. A failed check
returns nonzero and writes a failed result. Attach the JSON from the PR's exact head; a new
push requires a new run. Logs and generated artifacts stay out of the source tree.

The default run has three synthetic matrix shapes, one dense FFN case and eleven selected
Q4_K geometries. The optional fixture adds matrix-vector, 32-column and selected-Q4_K cases;
it must contain a positive multiple of four rows for the four-expert view. Use packed rows of a Q4_K tensor obtained
through the existing GGUF reader (`gguf::Split`/`TensorInfo`), with no conversion of its
weight bits. Record the original file SHA-256, tensor name and row range alongside the
result; do not upload model weights in the PR. Case results establish numerical operator
checks only, with no timing claim.

Scalar kernels use 64-thread workgroups. The selected port uses 256 threads (eight logical
32-lane rows), and a one-thread probe checks the compiled wave size before any selected launch.
Kernel signatures and argument storage must remain identical. Context/module/stream/buffer handles stay on their
owning thread, and synchronization precedes destruction.

## Original selected Q4_K port

The arithmetic baseline is Bloomery `9fc2cc0d67f9a5b1d71e10663ba43873431c25e5`;
the whole-file drift pin is reviewed against main `6d4113c1`.
`crates/gpu/src/q4k_sel.rs::q4k_gemv_sel` owns selection and launch geometry;
`crates/gpu/src/cores.rs::q4k_row_dot_1col` owns the arithmetic. `device/build.rs`
extracts nine original core function bodies verbatim, retaining the two-iteration
accumulation rule. All nine bodies and the selected entry are unchanged from the
arithmetic baseline; the intervening core additions are unrelated IQ lookup helpers.
A pinned whole-file FNV-1a fingerprint makes unsupported source drift fail at build
time. The fingerprint is a drift check; verification records SHA-256 for
`cores.rs`, the selected entry, the existing host oracle, and the generated Rust source.
The generated source lives in `target/amdgcn-amd-amdhsa/release/build/.../out/q4k_core.rs`
and is copied to the evidence directory for inspection.

The AMD shim supplies signed non-saturating dot4, the 16/8/4/2/1 XOR reduction,
explicit rounded add/multiply/FMA and fault atomics. One coefficient is spelled explicitly:
`cdb = fma(8*d, sc, -(dmin*mi))`, the CUDA contraction already specified by the existing
`qdot` host mirror. The ABI wrapper replaces CUDA slices and launch annotations with
host-validated pointers; the selected row/slot/id relation is unchanged. No CUDA source
or normal engine owner is modified.

For `n_sb = K/256`, a column holds `256*ceil(n_sb/4)` packed Q8 code words, `8*n_sb` signed sums,
and `2*n_sb` scales. `host/src/selected.rs` packs the oracle's codes through the original
`q4_slot` permutation. Its unused tail words are poisoned. Selected slot `s` always uses
activation column `s`, including when two slots select the same expert. A HOST id
(`u32::MAX`) skips its output without fault. Other ids outside the expert stack leave the
output untouched and atomically set the earliest-layer/site word and the layer's ExpertId
mask. Faults remain sticky across launches.

The eleven synthetic K values are 256, 512, 768, 1024, 1280, 1792, 2048, 2304, 2816,
6144 and 20480; rows per expert rotate through 7/8/9. Each geometry runs nine launches:
three repeats, new ids, three invalid-id launches at layers 9/3/9, an isolated HOST skip,
and a refused-scale/zero/subnormal-column case. This covers partial super-block groups, paired
accumulations, incomplete row blocks, duplicate ids, changed ids, unused activation
columns, exact fault masks and NaN propagation. It does not exercise graph capture.
The real fixture's rows are viewed as four experts solely to test the selected operator;
this does not make the source tensor or model routed.

The host quantizer owns the non-finite refusal in this experiment. The down kernel
propagates its NaN scale without claiming a GPU quantizer's QuantColumn fault. GPU
quantization, gate/up, attention/GDN, graph replay and a full model forward are later work.
The lab is not connected to `Gpu`, `GpuModel`, an app session or `bloomery-serve`. Gate one
complete model forward against llama.cpp HIP on identical model bits before designing a
common backend boundary with the maintainers.

The verifier rejects an inherited `CARGO_BUILD_TARGET` and explicitly selects the
`rustc -vV` host triple for host tests, lint and build. Its executable comes from
that triple's output directory, so a Cargo config target cannot select an older
`target/release` binary.
