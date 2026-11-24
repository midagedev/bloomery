# `#[verify]` on a real inference engine's kernels

cuda-oxide's preview Lean 4 verifier ([post](https://nihalpasham.github.io/posts/provably-safe-gpu-kernels/),
[#1851](https://github.com/NVIDIA/cuda-rust/pull/1851)) asks people to "try to break it". bloomery is an LLM
inference engine whose kernels are all cuda-oxide Rust, and its pin `ab2b6738` is the commit that brought `#[verify]`.
So we tried it on our own kernels. This is what we saw, compile-only, on one workstation (x86_64, CUDA 13.3, LLVM 21,
Lean v4.33.1), at pin `ab2b6738`. The rejections, the Lean time and the raw-pointer case below reproduce unchanged on cuda-rust main
`5e5c77b19` (after #1857's hardening).

## Why it fits us

Every engine kernel already carries a `#[launch_contract]`, and most of them argue their bounds by hand:

```rust
#[launch_contract(domain = 1, block = (256, 1, 1), requires = (y.len() >= n))]
pub fn scale_rows(mut y: DisjointSlice<f32>, n: u32, s: f32) {
    let i = thread::index_1d().get();
    if i >= n as usize { return; }
    // SAFETY: i < n <= y.len() by the launch contract; thread i owns entry i.
    unsafe {
        let v = *y.get_unchecked_mut(i);
        *y.get_unchecked_mut(i) = mul_rn_f32(v, s);
    }
}
```

That SAFETY comment is exactly the verification condition Lean proves for `safe_vecadd`. The census of the 632 engine
kernels (a line-level classifier, so a barrier inside a helper is not seen):

| Class | Kernels | `get_unchecked` | `unsafe` blocks |
|---|---|---|---|
| A: global slices only | 303 | 542 | 480 |
| B: shared memory or barriers | 107 | 74 | 237 |
| C: warp intrinsics, mma, raw pointers, asm | 222 | 944 | 1292 |

## What passed

Seven kernels were ported: three of class A (`add`, `scale_rows`, `softcap_rows`), two of B (`rms_norm`,
`ssmax_rows`), two of C (`argmax_rows`, `q4k_gemv`).

All three class A kernels pass `no_oob_idx`, `disjoint_writes` and `writes_all_elems` after two changes:

- **The extent must be a slice length.** 297 of the 303 class A contracts are `requires = (y.len() >= n)` with a scalar
  `n`; the verifier reads only `a.len() >= b.len()`, so `n` goes and the kernel walks `y.len()`.
- **`mul_rn_f32` / `add_rn_f32` are not modeled** (`nvvm.scalar_arithmetic`, rejected). We use `.rn` to stop FMA
  fusion where a sum order is load-bearing, so this one matters for numerics, not only for convenience.

The result is what the post promises. The proven checks are gone (`add`: 39 → 29 PTX instructions, 2 → 0 `trap`),
registers and spills are equal, and the SASS differs only by the extent's width (`LDCU` → `LDCU.64`), one `UMOV` and
NOP padding. Verification costs +1.3–2.3 s per kernel uncached, +0.03 s on a proof-cache hit.

Deliberate bugs on small kernels were all caught, with the right theorem named: an off-by-one tail guard, two threads
writing one slot, a neighbour write, a missing barrier, a barrier under `if tid < 128`, a read before the barrier, a
slot index one past the end (8 of 8).

## Where it stopped

- **B and C kernels of real shape are rejected while correct.** `rms_norm`, `ssmax_rows`, `argmax_rows` and `q4k_gemv`
  fail with `Not a definitional equality: validSymFunc mirFunc symTable = true`, which names no kernel and no Rust
  line. The pin's own `block_reduce` passes. Because the correct base already fails, bugs in real-shaped kernels cannot
  be tested at all.
- **Lean time grows fast with reads per loop iteration.** N reads of `x` per iteration: N=1 passes in 5 s; N=2, 3, 4, 6
  fail after 29, 63, 107, 223 s. Our `rms_norm` (20 reads per iteration) ran 16.7 minutes and 2.3 GB before we stopped
  it.
- **A block-uniform early `return` before `sync_threads()` fails `uniform_barriers`**, even when the condition is a
  kernel parameter.
- **Ballot and atomics are not modeled** (`nvvm.vote_sync_ballot`, `nvvm.atomic_rmw`), so our fault word and our argmax
  are out of scope for now.
- **A raw-pointer read is outside the proof but the log says proved.** A kernel that reads `x[i]` and also
  `*x.as_ptr().add(i + 1_000_000)` builds with `#[verify(no_oob_idx(x, y))]`; the log prints
  `proved: no_oob_idx(param 0, param 1)` and the PTX keeps `ld.global.nc.b32 [%rd1+4000000]`. A `get_unchecked` read at
  the same offset is rejected, and a raw-pointer write is rejected by `disjoint_writes`.

We are preparing a fix for the raw-pointer case. Each limit has a small repro crate that builds on its own against the
pin.

## What we take from it

For class A the trade is already good: the hand-written SAFETY argument becomes a checked proof at no runtime cost. We
will move class A kernels over as the scalar-extent contract and `.rn` arithmetic land, and look at B and C again when
the verifier accepts real-shaped reductions.

Thanks to Nihal Pasham and the cuda-rust maintainers for shipping this in the open.

*The trial was run by an AI agent (Claude) under the maintainer's direction; every number above is from its build logs.*
