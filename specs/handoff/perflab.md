# perflab handoff

2026-10-14. No open perflab pull request. No ws job and no Mac static is running. No box hold and no Mac track lock for `bloomery-perflab` or `bloomery-warmw2`.

## Goal and queue

Continuous perflab resident. Ranked work is `specs/perflab/queue.md` on the bloomery checkout (untracked `specs/`, that checkout is detached). This branch does not contain that board. Summary:

| Rank | Id | State |
|---|---|---|
| — | PF-4 | Closed. No nsys. Kernel `q8_0_gemv`, serial bytes 6,994,952,192. No existing m=1 rate more than about 1% under 567–701 GB/s |
| — | PF-1 | Closed, miss. No lever, no PR |
| 1 | PF-6 | W2 applied on `perflab-warmw2`. mac-static ok. No measurable nvtier/warmprof confirm. Do not take ws |
| 2 | PF-2 | Spark only. Full vs shipped MTP head, 6.0–7.0 ms/window [derived]. Do not change the default |
| 3 | PF-5 | Spark only. `sparkrouter`, −0.07…−0.22 ms. Not run |
| — | PF-3 | Lead decision. Not a run |

Nothing in that rank is implementable as code plus mac-static without a Spark or ws. The last concrete step freed stale Mac `target/` directories so other roles can static (below).

## Pull requests

No pull request for `perflab` or `perflab-warmw2`. Shingler has not reviewed this work. CI has not run on it.

## Branches

- `perflab` product tip before this handoff: `dac9a5e16df4ac5112a4eeef2d82feec715179f5`. Six commits ahead of merge-base `bd29e9c92485704b6affaa90d45462a767befeb2`, 37 behind `origin/main` `f1a3e9e44a8909f4c25fa216818cb6b04400d728`. Not rebased. Upstream was `origin/main`; the push of this handoff sets the upstream to `perflab` and must not update `main`.
- `perflab-warmw2` `dce4b4072deae35cf94e2b3b52b7bfd0203bd2e2`, parent `5c98cc707644f745d02ac95000ad35698f6a4615`. W2 (`ea291288`) with the keep-both `HostTier::new` resolution. Not rebased onto current `origin/main`. Original `warmw*` tips were not moved.
- `perflab-warmw3` exists locally at the W3 seam `343beb39` (parent was `82ce6f10`). It only drops the section (`warm_close` returns `Ok(None)`). Not pushed from here unless the push of WIP includes it; it was created in an earlier session and was not rewritten.

Product commits on `perflab`, oldest first: `9e62dceb` opt-axes, `68ef4532` m>1 tile entry, `4241c931` rustfmt, `981a6cec` PF-1 card, `d41654b1` first time miss, `dac9a5e1` PF-4 byte note.

## Runs and logs

No in-flight run.

PF-1 remeasure, A6000, one queued job, record started 2026-10-14 06:05:23 finished 06:11:14, rc 0 (stamps have no zone). `gate_qwen35moe_verify` PASS in 17 s (decode 520, pass5 617, pass8 629). `time prompt`, kind=prefill, passes=1:

| arm | ms |
|---|---|
| tile | 13.3714, 13.3654, 13.3684, 13.3750 |
| `_sel` | 13.2735, 13.2676, 13.2598, 13.2691 |

Median tile 13.3699. Median `_sel` 13.26835. Difference −0.10155 ms. Pairs −0.0979, −0.0978, −0.1086, −0.1059. decide-below. The earlier sitting was −0.079 ms. Tile prefill nodes 520,605,609,613,617,621,625,629. `_sel` prefill nodes 520,545,549,553,557,561,565,569. The local follower stopped after the fourth tile row; the job log on the box has `PF1 phase=done`. Scored copy on the board tree: `specs/perflab/evidence/pf1-sitting.md` and `pf1-sitting.log`.

W2 mac-static of `perflab-warmw2`: rc 0, total 203 s, fmt-check 0, lint 0 (ratchet 41), combos 0, every check-* 0. Local scratch log was `/tmp/warmw2-mac.log`.

## Rules

- This machine does not build or test. No local cargo, rustc, or just recipe that builds or tests. Mac static via `tools/mac-remote.sh`. GPU via `tools/box.sh` from the perflab worktree, one job, predicted wall printed first. `cargo oxide` must appear in a queued build command.
- Do not push `main`. Do not open a pull request unless the lead asks. A `git push` with no refspec on `perflab` used to target `origin/main`.
- Do not move the bloomery checkout. It is detached at `488131a7`. Untracked `specs/` there is the live board, including `specs/perflab/NOW` and `queue.md`.
- Lead 05:35: a passing measurement proposes that side as the unset default. A miss proposes nothing. Do not flip a lever to manufacture the winning arm. PF-1 missed, so no default change.
- Lead 06:23 and 07:28: PF-6 has no measurable confirm. Do not take ws for it.
- New repo text is English. Do not put host names, account homes, or pool aliases in committed text.

## Gotchas

- `generate_qwen3moe --time` with `-n 1` refuses (no counted step). The PF-1 card's `time prompt` row is `--tokens 760,6511,314,9338,369 --prefill pass -n 1` without `--time`, wrapped by `time-gate.sh` so the lease still takes `docs/cards/pf1-tile-m5.card`.
- The `_sel` arm is a second release build. Force off only `dispatch.rs` `if m > 1 && ffn_tiles(f)` and `if m > 1 && ffn_tiles(n)`. Restore the file before the job exits. Do not commit the flip.
- The queue client's tail can end before the last job lines are flushed. A done job's rows are on the box job log.
- W2 resolution keeps `unserved: None`, `call_open: OpenCall::Closed`, `swap_streams: Vec::new()`, `selected: 0`, and the no-machine prefix on both `keep_rows` and `keep_rows_by`. `Kind::Warm` and the W3 restore are absent. `gate-gpu-nvtier` and `gate-gpu-swap` are pass/fail gates, not that confirm.
- One pool Mac's data volume Available was 9 GB. Six stale unlocked `target/` directories older than a day were removed (18 GB): `bloomery-macremote`, `bloomery-pr37`, `bloomery-s0runner`, `bloomery-w4ctx`, `bloomery-w3q35a`, `bloomery-w3h2d`. `df -g` then showed Available 24 GB on `/` and on the Data volume. Clones were left in place. Live locks were none.

## Next

1. Do not take ws. Do not submit `gate-gpu-nvtier` or a warmprof confirm until `Kind::Warm` and the W3 restore exist and a card names the measured term.
2. Leave PF-2 and PF-5 until a Spark exists. Do not change those defaults.
3. If the lead adds a rank that is code plus mac-static only, implement it on `perflab` or a `perflab-*` branch, run mac-static, and stop at a READY tip. A miss still proposes no default.
