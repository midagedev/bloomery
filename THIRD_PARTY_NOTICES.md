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

## cuda-oxide

`crates/oxide-ice-unroll` (a compiler-bug reproducer, excluded from the workspace) carries NVIDIA's
Apache-2.0 headers from cuda-oxide. The GPU crates depend on cuda-oxide (Apache-2.0) as a pinned git revision.

`Cargo.toml` declares the dependency on [NVIDIA/cuda-rust](https://github.com/NVIDIA/cuda-rust) (cuda-oxide lives under
`cuda-oxide/` there) at revision `1691162ab9204ef21b0329c2e8059815420dfd5a`
and a `[patch]` section takes the source from our fork, [midagedev/cuda-oxide](https://github.com/midagedev/cuda-oxide),
branch `bloomery`: that revision plus the patches below. Every patch is meant for upstream and leaves the fork once
upstream has it; with none left, the `[patch]` section goes away.

| Patch | Upstream |
|---|---|
| [`6f30fa11`](https://github.com/midagedev/cuda-oxide/commit/6f30fa1150deca8798abc3c3e1ab20779e0ce836) feat(cuda-macros): let `requires` name unsigned integer constants | not yet proposed |
| [`361737ce`](https://github.com/midagedev/cuda-oxide/commit/361737ce49963829d844625b30df365c9ee1a5b3) fix(cuda-macros): name the macro call when a cuda_module finds no kernels | not yet proposed |
| [`e5eddb3b`](https://github.com/midagedev/cuda-oxide/commit/e5eddb3bf8a9cf8f26b9d32201af9c9a730cfc83) feat(unroll): recognize range `for` loops | proposed upstream as [NVIDIA/cuda-rust#1346](https://github.com/NVIDIA/cuda-rust/pull/1346) (open) |
| [`4da6c138`](https://github.com/midagedev/cuda-oxide/commit/4da6c138448538ca12218cb80924c070417ab419) fix(mir-lower): give an aggregate niche payload's leaves typed slots | not yet proposed |

## cuda-core (cutile-rs)

The GPU crates depend on `cuda-core` (Apache-2.0), the host crate cutile-rs and cuda-oxide share, for streams, events and the module loader. It comes from the same repository, revision and `[patch]` as cuda-oxide above (`cuda-host` takes it by path there, so a crates.io copy would be a second crate). Its `.oxart` container parser, crates.io `oxide-artifacts`, is the copy the module loader links.
