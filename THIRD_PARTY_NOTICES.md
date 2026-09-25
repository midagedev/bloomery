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

`Cargo.toml` declares the dependency on NVlabs/cuda-oxide at revision `b9847e9515ed3a23096f22567d3eaf0a6e3e440c`
and a `[patch]` section takes the source from our fork, [midagedev/cuda-oxide](https://github.com/midagedev/cuda-oxide),
branch `bloomery`: that revision plus the patches below. Every patch is meant for upstream and leaves the fork once
upstream has it; with none left, the `[patch]` section goes away.

| Patch | Upstream |
|---|---|
| (none yet — the branch is the pinned revision itself) | |

## cuda-core (cutile-rs)

The GPU crates depend on `cuda-core` (crates.io, Apache-2.0), cutile-rs's host crate, for streams, events and the module loader. Its `.oxart` container parser is the copy the module loader links.
