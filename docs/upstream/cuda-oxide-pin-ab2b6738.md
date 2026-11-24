# cuda-oxide pin move: 9df9e7ce → ab2b6738

bloomery is an LLM inference engine whose CUDA kernels are written in Rust with
[cuda-oxide](https://github.com/NVIDIA/cuda-rust). On 2026-11-24 it moved its pin from `9df9e7ce` to `ab2b6738`
(commit `678784d14`). This note lists what the move brought, measured on our cards.

The pin is taken straight from NVIDIA/cuda-rust: **no fork, no `[patch]`, no backend patch.** The previous pin
needed one local patch (the verifier fix below); this one is built as published.

## Scale

- 660 kernel entries in the RTX 5090 (sm_120a) spill table, across the engine and gate binaries.
- Four PTX targets pinned per entry (spill bytes, JIT local bytes): sm_86, sm_120a (RTX 5090), sm_121a and sm_121
  (DGX Spark / GB10).
- `generate_ds41` alone carries 437 PTX entries; `gate_kquant` 372.

## Upstream changes this pin brought in

### #1648: the CuTe verifier is linear again (fixes #1647)

Issue [#1647](https://github.com/NVIDIA/cuda-rust/issues/1647) (midagedev): `verify_cute_semantics` ran on every
module and took cubic time in the block count, even with no CuTe operation in the function.
[#1648](https://github.com/NVIDIA/cuda-rust/pull/1648) (juliankang4) fixed it.

| Probe: N blocks, one `u32` block argument each, no CuTe op | N = 50 | N = 800 | N = 8000 |
|---|---|---|---|
| before (`9df9e7ce`) | 0.09 s | 248 s | — |
| after (#1648) | 0.001 s | 0.006 s | 0.050 s |

Cold `cargo oxide build --bin generate_ds41`:

| Backend | Wall |
|---|---|
| rev `0143af5c` (before the verifier existed) | 388 s |
| `9df9e7ce` as published | did not finish in 1700 s |
| `9df9e7ce` + #1648 (our carried patch) | 480 s |

PTX was unchanged by the fix (`generate_ds41` 437 entries, `gate_kquant` 372). With `ab2b6738` the patch, its
checksum and the patch step in `tools/oxide-backend.sh` are gone.

### #1728: shared statics named by identity

[#1728](https://github.com/NVIDIA/cuda-rust/pull/1728) (praveenperera) names `.shared`/`.global` symbols by a hash
of the static's mangled identity, not by a walk-order counter. Before it, a host-only change (two stats fields in
another crate) renumbered the `.visible .shared __shared_mem_N` declarations of an unrelated bundle, so a PTX
comparison moved where no kernel had moved. Now our move proofs (`just canon-check`) see only real changes.

### #1668: native f32 max/min

[#1668](https://github.com/NVIDIA/cuda-rust/pull/1668) (CanReader) lowers `f32::max`/`f32::min` to one
`max.f32`/`min.f32`. Before it, each was four instructions (`setp.nan` + `setp.ge` + two `selp`). In the five PTX
modules of `gate_clef_serve`: the four-instruction form went from 1247 sites to 0, `max.f32` from 1409 to 2695,
`min.f32` from 0 to 8. In `gqa_prefill_flash` that was 32 sequences per warp tile.

### #1661: 32-bit shared-memory pointers

[#1661](https://github.com/NVIDIA/cuda-rust/pull/1661) (nihalpasham) emits `p3:32:32:32` with `--nvptx-short-ptr`.
Its effect on our register pressure was mixed, and we pinned every move:

| Table | Entries whose spill fell | Bytes | Entries whose spill rose | Bytes |
|---|---|---|---|---|
| sm_120a (RTX 5090) | 16 | −1348 | 10 | +648 |
| sm_121a (GB10) | 17 | −1304 | 9 | +600 |

## Speed: no regression

Same-lease A/B on one RTX 5090 (`tools/ref/depth-serve.sh`, new pin against the old one, interleaved rounds),
prompt-pass tok/s at P = 512 / 4096:

| Model | P = 512 | P = 4096 |
|---|---|---|
| Inkling 2 Mini UD-Q4_K_XL | +0.37 % | +0.26 % |
| Inkling 2 Mini, official checkpoint folder | +0.45 % | +0.26 % |
| Qwen3.8-27B Q4_K_M | +0.95 % | +0.85 % |

Decode was flat. Every effect is inside the ruler at the round count (about ±0.8–1.0 %), so we read this as **no
regression**, not as a gain. The models were picked because they run the kernels whose spill rose the most.

## What we found and reported on the new pin

The new cuda-core (#1603/#1604) frees and retires buffers on a `cuda-resource-retirement` thread. Two effects
showed up in our gates:

- **Late `cuMemFree`** (ledger row 57): a dropped `DeviceBuffer` is freed a moment later on the retirement thread.
  The card's free bytes lag the drop, and the late free can take the driver lock while it waits on a shut stream.
  We added `Gpu::settle_frees`. Reported with a reproducer on
  [#1852](https://github.com/NVIDIA/cuda-rust/issues/1852).
- **Safe async ops inside a stream capture** (ledger row 56): `zero_async` and `copy_from_device_async` queue a
  lease whose 1 ms event poll invalidates a capture that lasts longer than one poll (`end_capture` rc 901).
  Reproduced on the RTX 5090 in a scratch crate; we route those calls through a raw async path for now.

Both rows, with repro steps, are in [`nvlabs-ledger.md`](nvlabs-ledger.md).

## Thanks

To the cuda-rust maintainers and to the authors of the PRs above: juliankang4, praveenperera, CanReader and
nihalpasham.

*This note was drafted with an AI assistant (Claude) from the repo's ledger, cards and gate output.*
