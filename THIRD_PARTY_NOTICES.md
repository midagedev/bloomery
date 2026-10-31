# Third-party notices

## ggml / llama.cpp / ik_llama.cpp

`crates/tokenizer/src/collapse_table.rs` is generated from `src/unicode-data.cpp` of the reference
(ik_llama.cpp, which carries llama.cpp's file). Kernels and algorithms elsewhere in this repository were
written by reading ggml's and ik_llama.cpp's code; where one was taken, the source comment names the file.

```
MIT License

Copyright (c) 2023-2024 The ggml authors (https://github.com/ggml-org/ggml/blob/master/AUTHORS)
Copyright (c) 2023-2024 The llama.cpp authors (https://github.com/ggml-org/llama.cpp/blob/master/AUTHORS)
Copyright (c) 2024-2025 The ik_llama.cpp authors (https://github.com/ikawrakow/ik_llama.cpp/blob/main/AUTHORS)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## llama.cpp (ggml-metal)

`crates/metal/kernels/matvec_q4k.metal` and `crates/metal/kernels/matvec_q6k.metal` are adapted from the Metal
kernels `kernel_mul_mv_q4_K_f32` and `kernel_mul_mv_q6_K_f32` of llama.cpp (`ggml/src/ggml-metal/kernels/mul_mv.metal`,
commit `36a73916e`): the lane maps, the block loops, the nibble and 6-bit unpacking, the row loop and the SIMD-group
reduction are llama.cpp's; each file's header says what differs. The segmented entries of the two files and the
head-normed entry of the Q4_K file are the same body with the segment's rows or the input's head scale in place of the
one tensor's rows: the parts named above are the adapted ones. The licence is the one above, the ggml authors' MIT
licence, whose text applies to those two files.

`crates/metal/kernels/experts.metal` is adapted from those same two llama.cpp kernels
(`kernel_mul_mv_q4_K_f32` and `kernel_mul_mv_q6_K_f32` in `ggml/src/ggml-metal/kernels/mul_mv.metal`,
commit `36a73916e`): the lane map, the block loop, the nibble and 6-bit unpacking and the SIMD-group
reduction. The pair's gate and up rows, the slot select, the folded K' and the route weight on the
load are this engine's. The licence is the one above, the ggml authors' MIT licence, whose text
applies to that file.

The `Q8_0` and `Q5K` owners in `crates/metal/kernels/quant.metal` are adapted from llama.cpp's
`kernel_mul_mv_q8_0_f32` and `kernel_mul_mv_q5_K_f32` (the same file, commit `36a73916e`): the lane
maps, the block strides, Q5_K's fifth-bit masks and the sum associations. The bodies the owners
instantiate are the ones named above. The licence is the one above, the ggml authors' MIT licence,
whose text applies to those two owners.

## MLX

`crates/metal/src/bin/metal_gemm_time/probe_mlx_qmm.metal` is MLX's affine `qmm_t` kernel for 4-bit, group-64
weights, reduced to the one instantiation the `metal_gemm_time` probe launches as its reference arm. It is taken from
[ml-explore/mlx](https://github.com/ml-explore/mlx) v0.32.1 (`mlx/backend/metal/kernels/quantized.h`,
`steel/gemm/mma.h`, `steel/gemm/loader.h`); the file's header names what was taken out. It is compiled by the probe
only and never enters the engine's kernels.

`crates/metal/src/bin/metal_attention_time/mlx.rs` holds MLX's decode attention kernels `sdpa_vector`,
`sdpa_vector_2pass_1` and `sdpa_vector_2pass_2`, ported to read the engine's cache rows, as the `metal_attention_time`
probe's reference arm. They are taken from ml-explore/mlx v0.32.3 (`mlx/backend/metal/kernels/sdpa_vector.h`); the
file's header names what differs. They are compiled by the probe only and never enter the engine's kernels.

`crates/metal/kernels/gemm_tensor.metal` (the prompt pass's GEMMs on Metal 4's tensor ops) follows the design of MLX's
NAX quantized matmul, v0.32.1 (`mlx/backend/metal/kernels/quantized_nax.h`, `steel/gemm/nax.h`): the weights
dequantized into threadgroup memory a K step, the 64-by-64 tile over a 2-by-2 grid of SIMD groups, the `matmul2d`
descriptor with the weight operand transposed. It is written anew over this crate's own staging and quant owners and
copies no MLX code; the rule that picks it, `MatrixUnits` in `crates/metal/src/library.rs`, is MLX's
`is_nax_available` (`mlx/backend/metal/device.cpp`). Both files' comments name the MLX lines they follow.

MLX is under the MIT licence, Copyright (c) 2023 Apple Inc.:

> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
> documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
> rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
> permit persons to whom the Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or substantial portions of
> the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
> WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
> COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
> OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## cuda-oxide

`crates/oxide-ice-unroll` (a compiler-bug reproducer, excluded from the workspace) carries NVIDIA's
Apache-2.0 headers from cuda-oxide. The GPU crates depend on cuda-oxide (Apache-2.0) as a pinned git revision.

`Cargo.toml` takes cuda-oxide from [NVIDIA/cuda-rust](https://github.com/NVIDIA/cuda-rust) (cuda-oxide lives under
`cuda-oxide/` there) at revision `9df9e7cef561e03a066230b3bb433b3d8285da81`, as published: bloomery carries no fork (the
fork's last pinned revision is the tag `pin/87bc9518` of [midagedev/cuda-oxide](https://github.com/midagedev/cuda-oxide)).
One patch of the backend source is carried in `tools/oxide-backend/cuda-rust-1648.patch` and applied when
`tools/oxide-backend.sh` builds the backend: it is
[NVIDIA/cuda-rust#1648](https://github.com/NVIDIA/cuda-rust/pull/1648) by juliankang4 (dialect-cute's verifier indexes
its provenance facts by carrier and skips a story whose family has no operation in the scope), unchanged, until the pin
holds it. What earlier
revisions needed from a fork is done in bloomery's own tree: the `launch_contract`
`requires` bounds are written with integer literals (a compile-time assertion beside each kernel holds the constant the
literal stands for), and `tools/oxide-host.sh` chooses the host CPU flags of `cargo oxide` by target triple. The range
`for` unrolling the earlier fork carried is upstream
([NVIDIA/cuda-rust#1346](https://github.com/NVIDIA/cuda-rust/pull/1346), merged).

## cuda-core (cutile-rs)

The GPU crates depend on `cuda-core` (Apache-2.0), the host crate cutile-rs and cuda-oxide share, for streams, events and the module loader. It comes from the same repository and revision as cuda-oxide above (`cuda-host` takes it by path there, so a crates.io copy would be a second crate). Its `.oxart` container parser, crates.io `oxide-artifacts`, is the copy the module loader links.

## Unicode Character Database

`crates/tokenizer/src/letter_case_table.rs` is generated from Unicode 14.0.0's `UnicodeData.txt`.

UNICODE LICENSE V3

COPYRIGHT AND PERMISSION NOTICE

Copyright © 1991-2026 Unicode, Inc.

NOTICE TO USER: Carefully read the following legal agreement. BY DOWNLOADING, INSTALLING, COPYING OR OTHERWISE
USING DATA FILES, AND/OR SOFTWARE, YOU UNEQUIVOCALLY ACCEPT, AND AGREE TO BE BOUND BY, ALL OF THE TERMS AND
CONDITIONS OF THIS AGREEMENT. IF YOU DO NOT AGREE, DO NOT DOWNLOAD, INSTALL, COPY, DISTRIBUTE OR USE THE DATA FILES
OR SOFTWARE.

Permission is hereby granted, free of charge, to any person obtaining a copy of data files and any associated
documentation (the "Data Files") or software and any associated documentation (the "Software") to deal in the Data
Files or Software without restriction, including without limitation the rights to use, copy, modify, merge, publish,
distribute, and/or sell copies of the Data Files or Software, and to permit persons to whom the Data Files or Software
are furnished to do so, provided that either (a) this copyright and permission notice appear with all copies of the
Data Files or Software, or (b) this copyright and permission notice appear in associated Documentation.

THE DATA FILES AND SOFTWARE ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF THIRD PARTY
RIGHTS. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR HOLDERS INCLUDED IN THIS NOTICE BE LIABLE FOR ANY CLAIM, OR ANY
SPECIAL INDIRECT OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THE DATA FILES OR SOFTWARE.

Except as contained in this notice, the name of a copyright holder shall not be used in advertising or otherwise to
promote the sale, use or other dealings in these Data Files or Software without prior written authorization of the
copyright holder.
