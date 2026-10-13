# Vision handoff

Worker swap. Nothing here is a pull request. Do not push `main`. The lead
opens any PR. This branch was not on the remote before this push, so
there is no Shingler review and no CI for it.

## Goal and queue

0.3.3 MiniCPM-V images on `mcvisw`. The measured seat is in `ef10dc8b`.

Done on this branch:

- Tile cap is 70 tokens (1120 patches). A larger tile is the tower's
  existing buffer refusal. The slice plan is unchanged.
- `ImageNumber`, `PlaceholderCount`, and `ImageLast` are HTTP 400.
  `EmptySpan` stays 500.
- A build without the `vision` feature still compiles `vision_seat`
  (`deepseek41` exports it). That `open_minicpm` refuses by name
  (`--mmproj: this build takes no images (built without the vision
  feature)`) and does not call `encoder_seat`.
- The one-span test checks `ids.spans.len() == 1` and `ids.spans[0] ==
  1..1 + span`. Full mac-static, no base, remote rc 0, wall 186 s.
  fmt-check 0, lint 41 equal to the ratchet 41, combos 42 shapes and
  0 red, check-recipes 0, every other check-* 0.
- `vis_attn_group` and `vis_mean_groups` are pinned at 0 0 on this
  branch (`5375a338`). Do not open a pin-only PR.
- Qwen3 KV flash confirm is scored. The default stays `f16`.

Still open, in this order:

1. Rebase onto current `origin/main` and re-measure. See gotchas.
2. Image-byte replay hash, in one change with the `V41Layers` override.
   Do not apply the unapplied Append/Block patch.
3. MiniCPM draft serving, with unitier. Do not edit `served.rs` alone.
4. Nemotron header reader only after the lead says the image list is
   closed or blocked. Nano 30B-A3B first. No weights and no GPU for
   that reader.

Closed, do not rerun: the six-arm KV sitting, the n-gram-beside-MTP
analysis (confirm was not run), and both flash profiles below.

## Branch

- Branch: `mcvisw`
- Rebase parent the static measured: `5375a338` (then 0 behind
  `acfb220a`, 17 ahead).
- Seat commit: `ef10dc8b`. The static rsynced that diff on top of
  `5375a338`. This handoff file was not in that run. It does not change
  Rust.
- When this file was written, `origin/main` was `f1a3e9e4`, 4 commits
  ahead of `acfb220a`. Those commits revert the record-token helper
  and touch `vision_seat.rs`, `record.rs`, `generate.rs`, and
  `tools/gate-paths.tsv`. This branch still calls `record::name_token`.
  Main has `.replace(' ', "_")` again.

## Pull requests

None. `mcvisw` had no upstream. No Shingler status. No CI.

## Runs

No Mac run and no workstation job is in flight. The track lock for
this worktree was absent, and the pool slots checked at wrap-up were
free. No local box hold. Do not remove another track's lock.

| what | result | log |
|---|---|---|
| Full mac-static of the seat diff | remote rc 0, wall 186 s, lint 41 | `/tmp/mcvisw-cfg-static-4.log` (`EXIT:0`) |
| Earlier static, before the one-span edit | remote rc 1, lint 43, combos 0 red, check-recipes 1 | `/tmp/mcvisw-cfg-static-3.log` |
| First static | remote rc 1, `hp` used after move, fmt-check 1 | `/tmp/mcvisw-cfg-static.log` |
| Survey that took no slot | exit 75 in 5 s | `/tmp/mcvisw-cfg-static-2.log` |
| Flash rerun, one queued job | rc 0, decide-below, 0.435 ms | `/tmp/q3kvdepth-flash-rerun.log`, job `s58` |
| First flash profile | rc 0, decide-below, 0.471 ms | `/tmp/q3kvdepth-flash-s54.log`, job `s54` |

`s54.rc` and `s58.rc` are 0. Do not use `/tmp/q3kvdepth-flash.log` (it
was truncated). Do not trust `s53.rc` (1, from a dropped session).

The flash card is `docs/cards/q3kvdepth-flash.card` on the detached
profile tree at `ce422577`, not in this checkout. The profiled binary
prefix is `c5f0719654b1`. Same binary both arms. Last replay, depth
16386, 48 launches, grid 320x1x1, no scalar `gqa_flash_seg`:

- First: `gqa_flash_seg_mma` 2598.6 µs minus `gqa_flash_seg_mma_q8`
  2127.2 µs = 0.471 ms. Control window agrees.
- Rerun: 2564.4 µs minus 2129.2 µs = 0.435 ms. Control replay is
  2565.1 µs minus 2132.8 µs. Merge in that window is 364.4 µs and
  349.5 µs and is not the term.

Band was 0.90..1.26 ms. Both readings are decide-below. Leave `f16`.
Do not fit a new byte rate. The 16K step gap from the sitting is
+0.289 ms (printed effect −3.86 %). That sitting is closed.

## Decisions and rules

User decisions, 05:35 KST. Defaults are the side that measures faster.
The 0.3.3 cut is aggressive. These choices are closed:

- MiniCPM-V tile bound is 70 tokens.
- `ImageNumber` is HTTP 400.
- The replay signature is a 32-byte SHA-256 of the submitted image
  bytes. No image hashes the empty bytes. A capture stores the hash it
  was built with and refuses a different one by name. Not wired.
- MiniCPM draft serving is in 0.3.3, coordinated with unitier.
- Nemotron: Nano 30B-A3B first. Shared experts are 2 by tensor shape,
  otherwise refuse by name. 4B is in 0.3.3 and has no type tag yet; do
  not call it dense or MoE. MTP is not stripped; the one block runs as
  a draft. Chat mode is the file default. Context is the file length
  and the planner may shrink it. Weights are an existing GGUF Q4_K,
  then NVFP4 inside 0.3.3. The fixture follows the checkpoint license.
  The repo stores a hash only.
- `QWEN3_KV` lost at 16K, so `f16` stays. A later win auto-applies only
  under the winning condition. That picker is not implemented.

Rules:

- This machine edits, runs git, and reads. It does not build or test.
  No `cargo build/test/check/clippy/run`, no `rustc`, no `just` recipe
  that builds or tests locally. Static checks go through the main
  checkout's `tools/mac-remote.sh` (`--tree` this worktree). Do not use
  this branch's copy of that script. GPU work goes through `tools/box.sh`
  with `BLOOMERY_BOX_QUEUE=1`. One GPU job, no preemption. Print the
  predicted wall before submit. A timed run needs its prediction card.
- Pool floor is 15 GB. Do not lower it. Do not pin a Mac when another
  pool Mac is at or above 15 GB. The last static was not pinned: one
  Mac was over the floor and warm with both slots free, the preferred
  Mac had both slots taken, and the third was under 15 GB.
- One Mac run per branch. Never remove another track's lock. Exit 75
  is the floor, an unreachable pool, or contention. The shell echo
  makes the local status 0; trust the `EXIT:` line.
- Do not raise `tools/lint-ratchet.txt` (41). Do not change check
  criteria. A `check-recipes` rc 1 from `hold over the queue`, `the
  batch hold` (rc 124), or `boxq` `int-cancel` is a load flake.
- Two trees never share a remote directory. Do not edit a tree while
  a Mac remote is syncing it.
- Hold owner names are letters, digits, and underscore. `vision_mc`
  stays down.
- Workers do not push `main`, tag, or open a PR. This swap was the
  order to push `mcvisw`.
- New repo text is English. Do not put host names, addresses, home
  paths, or pool Mac names in commits or in this file.
- Untracked role notes on the main checkout (`specs/vision/`,
  `specs/q3kvdepth/`, `specs/nemotron/`) are not in this branch. Do not
  move that checkout.

## Gotchas

- `Hparams` is not `Copy`. `Media::new` takes it by value. Compute
  `minicpmv::card::of` before that move. The first static failed here
  (`E0382`) and the rerun was clean.
- `open_minicpm` has 6 parameters. The vision body sizes the tower for
  `MINICPM_TILE_TOKENS` and seats `a.place.unwrap_or(Place::A)`.
  `vision::arch::open` still refuses `Projector::Minicpmv47` by name,
  because image ids come from the text tokenizer. That refusal stays.
  The seat path is `open_minicpm` in `serve_seats/qwen3.rs`.
- `[1..1 + span]` is `clippy::single_range_in_vec_init` plus a
  generated-warning line (the old lint 43). Do not collect the range
  into indices. Do not bump the ratchet.
- mac-static does not run the host tests. `Clip::new(Some("minicpmv4_7"))`
  in the seat test opens a fixture only under `cargo test`.
- `gpu-vision` `open()` sizes MiniCPM as `PATCHES_PER_TOKEN * limits.max()`.
  The seat's 16 is the same figure, not a second constant to invent.
- `DraftLayers::replayable` is parameterless on the older tree and
  defaults to `Ok(())`. On main, `V41Layers` overrides it for
  `Rule::Reference`. A signature change has to update that override in
  the same commit. Do not pass a fake zero hash.
- `served.rs` refuses an image under `BLOOMERY_DRAFT=dspark`. Leave it
  until unitier shares the draft path.
- The rebase conflict in `tools/gate-paths.tsv` kept both
  `gate-gpu-ds41-rope` and `gate-gpu-minicpm-text`. Main has since
  reverted the rope row. On the next rebase, keep `gate-gpu-minicpm-text`
  and take main's rope row.
- `name_token` on this branch will conflict with main's revert. Take
  main's `.replace(' ', "_")`. Keep the 70-token seat.
- A cold or freshly rebased Mac target is slower than the 186 s wall.
  The last prediction used before that run was 20 min, bound 45.
- The flash byte-rate model fails inside `gqa_flash_seg_mma_q8`. Do not
  publish a replacement GB/s. Scalar `gqa_flash_seg` on f16 is the
  wrong pass. `gqa_flash_merge` is recorded and not judged.

## Next 3 steps

1. Rebase `mcvisw` onto `origin/main` (`f1a3e9e4` or later). Keep the
   70-token seat, the 400 mapping, the named no-vision refusal, and the
   index span assert. Take main's `.replace(' ', "_")` and main's
   gate-paths rope row. Keep `gate-gpu-minicpm-text`. Then one
   mac-static of this worktree from the main checkout's
   `tools/mac-remote.sh`. Do not pin unless both other pool Macs are
   still under 15 GB. Do not start it while a lock for this track is
   live.
2. Leave the image-byte hash unwired until the hash source and the
   `V41Layers` `replayable` override land together. Leave MiniCPM draft
   serving for unitier. Do not change `served.rs` in this branch alone.
3. Do not start the Nemotron header reader, the n-gram confirm, or
   another GPU job unless the lead assigns it. `QWEN3_KV` stays `f16`.
   Ask the lead before any PR. The static above is not a CI result.
