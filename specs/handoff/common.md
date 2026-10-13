# common handoff (2026-10-14)

The common worker stopped here so a new session can take the branch. No main push.
The code tips below were already pushed. This file is the only new commit on `ringr1`.

## Current goal and queue

0.3.3 common queue, two open pull requests. Both Shingler must-fix rounds are folded.
No GPU until quant finishes q36int timings on ws. `(pg)` stays with the user.
q36leg R2 waits on R1 CardLeg. q35hybpp stays unwritten. Do not start q36legr1,
esplitpin, q36leg R3–R4, ring R3/R4, Nemotron, or ubgemm phase 2.

## Open PRs

### #129 ringr1 — CLEAN, merge state CLEAN

https://github.com/midagedev/bloomery/pull/129

- Product tip Shingler approved: `2d841e3d1f686fe4093e7443b982ec0303b0692c`
- Review: github-actions APPROVED at 2026-10-13T23:05:26Z. Verdict CLEAN.
  Both must-fix items are resolved (one allocator through `new_seq_for`; the
  second-cut clause resets the temporary live body before the swap back).
- CI at wrap-up: label, contrib-check t0, gate, diagnostics, review, post all SUCCESS.
  mergeable MERGEABLE. mergeStateStatus CLEAN.
- Non-blocking, not folded: ring kernel resource evidence (`flash_gqa.rs` append
  remainder and `rope_neox.rs`). No card run is attached. Leave it.
- This handoff commit sits on top of `2d841e3d` and will move the PR head.
  It is docs only. Do not treat it as a product change. A synchronize event
  retriggers Shingler.

### #128 w3q35f2f — verdict CLEAN, GitHub still CHANGES_REQUESTED

https://github.com/midagedev/bloomery/pull/128

- Tip: `e81b1f8d80f3f5a92b5dc07cc63c91719fcba818` (matches origin).
- Review: github-actions COMMENTED at 2026-10-13T23:14:04Z on that sha.
  Verdict text is CLEAN. The no-table policy is `Tables::key` / `Tables::dims`.
  The Q8_0 head comment says the gemv reads f32 rows directly.
- The review state is COMMENTED, not APPROVED, so `reviewDecision` was still
  CHANGES_REQUESTED. mergeable MERGEABLE. mergeStateStatus BLOCKED.
  `contrib-check t0` was IN_PROGRESS at wrap-up. Do not re-push this tip.
- Non-blocking, not folded: occupancy and precision proof for the q32 gemv,
  and a direct NaN/infinity fault gate. No GPU. Leave them.

## Branches and tips

- origin/main: `f1a3e9e44a8909f4c25fa216818cb6b04400d728`
- `ringr1` worktree `/workspace/repo/bloomery-ringr1`. Code parent `2d841e3d`.
  Based on `acfb220afe80`. A merge-tree with `f1a3e9e4` was clean (only
  `tools/gate-paths.tsv` overlaps, different hunks). Not rebased onto `f1a3e9e4`.
- `w3q35f2f` worktree `/workspace/repo/bloomery-w3q35f2f`. Tip `e81b1f8d` on
  `acfb220afe80`. Merge-tree with `f1a3e9e4` was clean. Not rebased.
- The main worktree `/workspace/repo/bloomery` is detached. Do not move it.
  Untracked notes there: `specs/common/NOW`, `specs/common/HANDOFF.md`.
  This file supersedes them.
- Unsafe pins: ring `gpu 1747 194 10`. w3 `gpu 1766 196 10` (NVMe 1746 plus
  +20 blocks and +2 fns). Lint ratchet stays 41. No new unsafe in the last folds.
- Eight untracked oxide dumps sit in the ringr1 worktree root
  (`bloomery_gpu*.ll`, `bloomery_gpu*.ptx`, `bloomery_gpu_deepseek41.*`).
  Do not commit them. Do not delete them.

## In-flight runs and logs

Nothing of this worker is running. No ws job was started. The PR watcher is stopped.
Do not start another watcher unless asked.

Mac statics that count (remote rc is the line in the log, not a wrapper's exit):

- `~/.cache/bloomery/ring-fold-ms6.log` — `2d841e3d` vs `f1a3e9e4`. remote rc 0,
  wall 127 s, lint 41. `mac-static: ok`.
- `~/.cache/bloomery/ring-fold-ms5.log` — `e51d357f` vs `acfb220a`. remote rc 0,
  wall 206 s, lint 41.
- `~/.cache/bloomery/w3-tables-ms.log` — `e81b1f8d` vs `f1a3e9e4`. remote rc 0,
  wall 121 s, lint 41. `mac-static: ok`.
- `~/.cache/bloomery/w3-rebase-ms.log` — `08170128` vs `acfb220a`. remote rc 0,
  wall 224 s, lint 41.

Do not rerun. Known reds, already fixed or not code:

- `ring-fold-ms4.log`: `32abd039` remote rc 1. fmt `place.rs` and `at: ref mut`
  in `prefill.rs`. Fixed by `e51d357f`.
- `ring-fold-ms3.log`, `ring-fold-ms2.log`, `w3-gatepaths-ms2.log`: exit 75
  before a slot (pool under 15 integer GB).
- `w3-gatepaths-ms3.log`: remote rc 1 only because the mac-remote self-test
  took a slot after 1 s. Not a product red.

Local `~/.cache/bloomery/mac-pool` was empty. `~/.cache/bloomery/locks` had no
files. A read-only lock survey at wrap-up found no track locks. No ws lock
was held. Finished mac-remote runs remove their own track lock.

## Decisions and rules

- A default follows the measured winner. 0.3.3 scope is aggressive. An
  unmeasured flip stays off. Do not edit `(pg)`. Do not drop a gate. Do not
  bump the lint ratchet except for new unsafe that actually landed.
- Cost rule (06:00 KST): implement and measure. Read the failing line, not the
  whole log. Do not re-run a check whose cause is known.
- This machine does not build. No cargo build/test/check/clippy/run, no rustc,
  no just recipe that builds or tests. Statics go through
  `/workspace/repo/bloomery/tools/mac-remote.sh --tree <worktree> <base>`.
  GPU goes through `tools/box.sh` onto ws. Do not set `CARGO_TARGET_DIR`.
  Do not use the local cargo-oxide.
- Pool floor is 15 integer GB. A pin bypasses the floor. The two machines the
  lead named at 07:28 were under it. At most one mac-remote per branch.
  Do not remove another track's lock. Do not cat another track's owner file.
- Commit only on `ringr1` and `w3q35f2f`, as midagedev. New commits, never amend
  a pushed commit. No new Co-Authored-By. `git commit -F` a file under `/tmp`.
- Rebase by naming the commit from the branch worktree. `git -C` always.
  A box launch from a dirty tree stamps `BLOOMERY_GIT_COMMIT` dirty.
- Push of these feature branches was ordered, with `--force-with-lease` when
  the update is not a fast-forward. Never push main.
- Specs in the main tree stay untracked. Do not commit them onto a branch
  except this handoff file, which the lead asked to push.

## Gotchas

- `enqueue_group` matches `&mut Body`, so the written end is `at: high`.
  `at: ref mut high` does not compile.
- The second-cut host assert in `place.rs` must stay on one line or rustfmt fails.
- `Body.at` is the greatest possibly written end. `cut_keeps` uses
  `self.at.max(pos)`. Reset stays `at = 0`. `new_seq` and `new_seq_rows`
  both call `new_seq_for`. The e2e second-cut clause calls `ChainBody::reset`
  on the temporary live body before swapping the parked sequence back.
- `quant32_quad` returns `(word, lane-local quad sum, scale, refused)`.
  The i32 is not the octet sum. Refused is `(0, 0, NaN, true)`. Instruction
  order is not PTX-identical. Do not claim that it is.
- `Tables::key` and `Tables::dims` are the no-table defaults. Engram, PLE and
  the toy table still override them. MapTables keeps its own `key`.
- On acfb, `Machine` has `nodes`. `.host()` and `.unified()` are methods.
- `gh` colors JSON even with `NO_COLOR=1`. Strip ANSI before jq. `set -o pipefail`.
- mac-remote maps ssh rc 255 to local exit 75. A wrapper that does not
  `exit $rc` can report harness exit 0. Trust `remote rc` and `wrapper_exit=`
  in the log. A nested `remote rc 0` inside check-recipes can be a self-test stub.
- Integer GB under 15 is skipped even when a decimal reading looks like 15.
  Do not lower `POOL_MIN_GB`.
- Shingler `review.yml` is `pull_request_target` on opened, synchronize,
  reopened, and ready_for_review. CLEAN on #128 was posted as COMMENTED, so
  GitHub does not show it as an approval.
- Do not set `BLOOMERY_CARD` together with `BLOOMERY_GATE_CARD=any` (exit 64).
- One compile at a time. Do not touch a live other-track slot.

## Next 3 steps

1. Leave #129's product commit at `2d841e3d`. The lead merges. Do not push main.
   If this handoff commit is the PR head, it is docs only.
2. Leave #128 at `e81b1f8d`. The verdict text is CLEAN. Wait for `contrib-check t0`.
   Do not fold the non-blocking notes. Do not re-push.
3. No GPU, and no new mac-static of a tree whose remote rc is already 0, until
   a new must-fix lands or quant releases ws.
