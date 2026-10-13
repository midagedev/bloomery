# toktape handoff (2026-10-14)

Worker replaced. No Mac or ws lock is held. Nothing of this track is running.

## Goal and queue

Fold Shingler's MUST_FIX on bloomery #134, then stop. Quiet-Mac is next only when a pool Mac has at least 15 GB free. Do not start it in the same breath as a #134 static, and do not run it on `loadcapreg`.

Queue, in order:

1. #134 is pushed and Shingler CLEAN on `455c7435`. mac-static of that fold did not run.
2. Quiet-Mac walls on their own branch. Pair `7f3448bb` vs `3bc2a172`. Local worktree `/workspace/repo/bloomery-quietmac`, branch `quietmac`, tip `3bc2a172`, not pushed, no commits of ours.
3. Nemotron stays waiting. `ggtiming` stays parked (`e0905e88` on `bd29e9c9`, `/workspace/repo/bloomery-ggtiming`). Hub deploy stays untouched.

## Open PRs

- https://github.com/midagedev/bloomery/pull/134 — OPEN. Branch `loadcapreg`. Head at the CLEAN review was `455c7435fda48bd4f128a89c080821f73876ca2c`. Base of the scoped static is `89d8eebb` (do not retarget that base onto later main). Shingler: MUST_FIX on `786f7c96` (run 38470583519), then CLEAN / APPROVED on `455c7435` (submitted 2026-10-13T22:37:47Z). CI on `455c7435`: `label`, `contrib-check t0`, `gate`, `diagnostics`, `review`, `post` all SUCCESS. This handoff commit is a later push and will schedule another review. Do not force-push.
- https://github.com/midagedev/toktape/pull/1 — OPEN. `feed-cache-120` at `547655ec`. Tests were 9 passed. Not deployed. Saved Cloudflare login expired 2026-08-27. Do not copy it. Do not run wrangler.

Merged and closed for this track: bloomery #114, #115, #123, #124. Do not fold #124 again.

## Branch and tip

- Branch to continue: `loadcapreg` in `/workspace/repo/bloomery-loadcapreg`. Parent of the fold is `786f7c96`. The fold commit is `455c7435` (`tools: the synthetic load rejects a non-finite cap`). The tip after this handoff is the commit that adds this file.
- `quietmac` is local only, at `3bc2a172`, clean. Do not push it unless the walls produce a commit the lead asked for.
- Main checkout `/workspace/repo/bloomery` is branch `macauto`. Do not move it. `specs/toktape/` there is untracked and is not on this branch.

## In-flight runs and logs

No mac-remote, no `tools/box.sh`, no slot, no track lock. Last `--locks`: d1-mac both slots free, no track locks; imagoworks-hckim both slots free, no track locks; hckim-mbp not reachable in 5 s, so the command's rc was 75. That 75 is the unreachable Mac, not a lost run. Disk GB was not remeasured in that pass.

Earlier the same morning, before those slots freed: d1-mac 9 GB, hckim-mbp 13 GB, imagoworks-hckim 203 GB with both slots held (`bloomery-placea`, `bloomery-ringr1`) and load `139.35 54.15 40.05`. Probe again before any static. Do not pin a Mac under the 15 GB floor.

Logs:

- `/tmp/loadcapreg-mac.log` — scoped mac-static of `786f7c96` vs `89d8eebb` on d1-mac. Remote rc 0, wall 114 s (setup 19, run 92), mac-static 90 s, check-levers rc 0 in 22 s, check-recipes rc 0 in 90 s, lint 41=41. Do not re-run it.
- Fold FAIL-first of `455c7435` was a temp dir on d1-mac, removed after. Load `2.77 2.70 3.27`. No cargo, no slot. Real: `selftest: self-test ok`, `ok load-ready`, both rc 0. Mutant that drops the three checks and restores `cpus or 1` accepts `nan` and returns `20.0`. Write-up: `specs/toktape/evidence/loadcap-failfirst.md` on the main tree (untracked).
- Quiet-Mac pin self-test in a temp tree failed `pure-crates` and `combos` because that tree was not a workspace. The three "does not pin the commit word" lines did not fire on the real script and did fire on the mutant. `/tmp/quietmac-ff` was removed. Note: `specs/toktape/evidence/quietmac-failfirst.md` on the main tree. The full-tree re-run has not happened.
- A `grep -l` of `/tmp/mac-remote.59LCQ7/*` has been running since 2026-10-13 23:09 under a separate cursor shell. It holds no pool lock. Left alone.

## Decisions and rules

- This machine does not build or test. No cargo, rustc, or local `just` build/test. Statics go through `tools/mac-remote.sh` from the main tree (pool script). GPU only via `tools/box.sh` on ws. ws stays with quant. No new ws job.
- `POOL_MIN_GB` is 15. A pin runs even under that floor. Do not pin to dodge the floor. `--locks` exits 75 if any pool Mac is unreachable; printed lines for the Macs that answered are still valid.
- Commit only on this track's branches. Never push main, never force-push, never amend. One Mac run per branch. Do not take another track's lock or slot. Do not `--clean` unless asked.
- `BLOOMERY_SELFTEST_LOAD` and `BLOOMERY_SELFTEST_NCPU` are registry rows owned by `tools/bloomery/selftest.py` (`Site::Env`). A `tools/levers-direct.txt` line does not clear a `tools/` `BLOOMERY_*` name. The rename on `f2d4b217` (`LOADCAP_*`) is not the fix.
- The fold: synthetic load must be finite and nonnegative; synthetic CPU count must be positive; a non-finite cap raises. Unset synthetic load still uses the host load and `os.cpu_count() or 1`. PIN `2026-10-13` on the scale formula stays. Idle 20 at load 80 / 10 cpus prints 160. Do not scale `gone()` or `wait_requests`. Product cancel wait 150 and poll 5 stay.
- Quiet-Mac is a measurement of the landed tools stack, not a new API. Walls: check-recipes, full mac-static, HEAD-move lint (warm lint, empty commit in a scratch clone, lint again; claim 247 → 0). Known figure: check-recipes 98 s at `3bc2a172`, from about 1065 s. Exclusive quiet means both slots of one Mac free, no other cargo, and the other slot held. Do not substitute today's main for the pair.
- Public numbers need their record. No cross-engine ratio. Do not invent load, ncpu, timings, or D1 rows. Pool alias only in notes. No credentials, no `user@host`, no LAN or tailnet addresses.
- Do not edit `notes-final.md` or the published 0.3.2 release. Do not deploy the hub. Do not raise `tools/lint-ratchet.txt` (it is 41).

## Gotchas

- `check-recipes` rc 1 from mac-remote's own self-test (`run slot 1 of 2 taken after 1 s` vs the assertion's `after 0 s`) is outside the loadcapreg diff. The retry printed `after 0 s` and exited 0. Do not edit `tools/mac-remote.sh` for it.
- ssh rc 75 and a missing pid are load or session loss, not a product finding, except the two self-test cap races already fixed.
- Sharing a Mac makes check-recipes and gpu-gate hold-released flakes. A quiet wall measured beside another cargo is not a number.
- `mac-check.sh --self-test` calls `recipes.py` `pure-crates` and `combos`. A copied subset of the tree fails those. Run it from a full checkout.
- `bloomery-tx3land` on d1-mac was a warm clone at `3bc2a172` with a target. Confirm it is still that sha and still unlocked before using it as a seed. `cp -c` only. Do not write the seed tree.
- The board generator rewrites `specs/leader/BOARD.md` from role NOW files. Re-read the toktape row before any edit. Do not edit other roles.
- `just wt-retire ctxcards` refuses while this session's cwd is inside `/workspace/repo/bloomery-ctxcards`.

## Next 3 steps

1. Leave #134 at the CLEAN sha unless this handoff commit's new review asks for a change. Do not re-run the `786f7c96` static. A mac-static of the fold, if the lead still wants it, is scoped vs `89d8eebb`, one free slot, only on a Mac with at least 15 GB free.
2. Quiet-Mac full-tree `tools/mac-check.sh --self-test` from `/workspace/repo/bloomery-quietmac` (`3bc2a172`), only when a Mac has at least 15 GB free. Then the three walls on `7f3448bb` vs `3bc2a172` when that Mac is quiet. Hold the other slot for the measurement. No ws.
3. Do not deploy https://github.com/midagedev/toktape/pull/1. Nemotron and `ggtiming` stay parked. No push to main.
