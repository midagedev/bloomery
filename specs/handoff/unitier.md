# unitier handoff

Paused 2026-10-14. The worker is stopped. No Mac lock and no ws hold is held.
Nothing is running. Do not start a build on this Linux machine.

## Goal and queue

Unitier, 0.3.3. Two open pull requests, both Shingler `MUST_FIX` on the current head.

1. `#135` `les1` — part-parallel fills, resident-first one-column hits, T1 counters. The first must-fix is folded. A second must-fix is open.
2. `#142` `placea` — a one-card plan asked for a DSpark draft refuses by name. Static is green. Shingler requested changes.

Behind those, and not started: prefetch stays a lever and defaults on only if a measurement wins. Resident-first becomes the default path only if it wins, and it merges with allocation still present. No predictor. No prefetch counters. No ws while quant holds timings. `tilekern` (`dcb5da17`) is unreached. The Q6_K activation assert and the dispatch same-lease A/B stay undone.

`#126` (`dsfullr`) is merged. Do not rerun its eight real-tier gates. `#112` is the docs-only LES plan. Do not treat it as a pull request to update.

## Open pull requests

### #135 `les1`

- https://github.com/midagedev/bloomery/pull/135
- Worktree `/workspace/repo/bloomery-les1`. Branch `les1` matches `origin/les1`.
- Tip `9aad2c8e409565a78f6fdc88b76b320454579588`. Tree clean.
- GitHub says `MERGEABLE` and `BLOCKED` (review). `reviewDecision` is `CHANGES_REQUESTED`.
- Checks on this tip, all `SUCCESS`: `label`, `contrib-check t0`, `gate`, `diagnostics`, `review`, `post`.
- Card label from the review: `needs-gpu`. No card run is in the pull request. Do not take ws for it.
- Pushed with `--force-with-lease` from `f5fa7a17` after a rebase onto `acfb220a`. `origin/main` at this pause is `f1a3e9e44a8909f4c25fa216818cb6b04400d728`. The pull request still merges. It was not rebased again.

Folded already (review `5510330813` on `f5fa7a17`, `MUST_FIX`, two items):

- `HostLayer::prepare_step_ids` in `crates/model/src/moe.rs`.
- `Stats::note_claim` in `crates/gpu/src/host/nvtier.rs`.
- `9aad2c8e` is only the rustfmt wrap of `note_claim`'s `fetch_sub`.

Open (review `5511063938`, submitted 2026-10-13T23:01:07Z, head `9aad2c8e`, `MUST_FIX`, must_fix=2, warn=1, run `38479662371`):

1. One owner for pinned-fill completion. `nvtier.rs` around 2741 and 3089 both wait for every part, map the failing unit to its expert, free failed picks, and publish with the pin under the books lock. The review asks for `settle_fill` shared by `Inflight::settle` and `NvTier::step_finish`.
2. One pin payload. `nvtier.rs` around 1447 repeats `moe.rs` around 2175. The review asks to delete `HeldClaim` and store `PinToken`.

Non-blocking, not done: a fitting one-column plan with early hits adds dispatches, and the reused job takes more read-buffer locks. Needs a same-lease decode A/B. No ws.

The review body has the replacement it wants. Check that against the tree before applying it. Do not paste it blind. The carve constructor already follows main: `ArenaKind`, `Result<NvTier, GpuError>`, `step` after `reader`. Do not put `ArenaMem::new(map)` back.

### #142 `placea`

- https://github.com/midagedev/bloomery/pull/142
- Worktree `/workspace/repo/bloomery-placea`. Branch `placea` matches `origin/placea`.
- Tip `2a07573b9396244c3ce19282e6cb0a3dc8d8122a`. Tree clean. The lead rebased and opened the pull request. This pause did not push it.
- GitHub says `MERGEABLE` and `BLOCKED`. `reviewDecision` is `CHANGES_REQUESTED`.
- Same six checks `SUCCESS` on this tip.
- Card label: `needs-gpu`. No card run. The clause is a host test.
- Ready-for-Shingler comment: https://github.com/midagedev/bloomery/pull/142#issuecomment-6156683713

Open (review `5510874863`, submitted 2026-10-13T22:36:09Z, head `2a07573b`, `MUST_FIX`, must_fix=3, warn=1, run `38477905957`):

1. Plan (b) still drops `--draft`. `gate_load_v41.rs` around 229 accepts the flag and builds `plan_b` without a reserve. The review asks for `check_tiers` on an empty tier list with a tier-0 draft, then `plan_b`.
2. `plan_a_draft` repeats the draft-off-plan rule. The review asks it to call `plan_tiers` (`workstation.rs` around 260 vs 483).
3. The new `plan_bp` reserve assertion repeats the field-for-field pin. Keep `place_a_names_the_draft_or_refuses` as the one test. Drop the extra assertion around line 1181.

Non-blocking: a parsed `--plan a --draft` probe. Reverting the CLI arm to `plan_a` leaves the unit test green.

Do not change the idle-card draft under place (a). That path never asks the plan. `plan_a_draft(None)` stays `plan_a`. `Some(0)` in the gate is a sentinel byte count, not a real file size.

## Branch and tip

| Branch | Worktree | Tip | Remote |
|---|---|---|---|
| `res46` | `/workspace/repo/bloomery-res46` | this commit | pushed for this file |
| `les1` | `/workspace/repo/bloomery-les1` | `9aad2c8e409565a78f6fdc88b76b320454579588` | matches `origin/les1` |
| `placea` | `/workspace/repo/bloomery-placea` | `2a07573b9396244c3ce19282e6cb0a3dc8d8122a` | matches `origin/placea` |

`res46` is the session worktree. Its older commits are the residency climb, not the unitier code. The unitier code is `les1` and `placea`.

The bloomery checkout at `/workspace/repo/bloomery` is detached at `488131a7`. Leave it. Untracked `specs/` and `docs/cards/` live there. Do not commit them. Do not push `main`.

## In-flight runs

None. A pool survey at this pause showed six slots free and no track locks. `~/.cache/bloomery/box-hold/` has no hold file. No ws job was started in this stretch.

Logs, scrubbed, in the untracked tree (not on `les1`, `placea`, or `res46`):

- `/workspace/repo/bloomery/specs/unitier/evidence/placea/mac-static-3.log` — `2a07573b` vs `acfb220a`, remote rc 0, wall 201 s, lint 41.
- `/workspace/repo/bloomery/specs/unitier/evidence/les1/mac-static-13.log` — `80bab993`, remote rc 1, fmt only (`fetch_sub` wrap), wall 181 s. Lint 41 and every other step rc 0.
- `/workspace/repo/bloomery/specs/unitier/evidence/les1/mac-static-14.log` — `9aad2c8e` vs `acfb220a`, remote rc 0, wall 122 s, lint 41.

Older, do not rerun: `les1` mac-static-11 and mac-static-12 are `f5fa7a17` only (remote rc 0). `placea` mac-static-2 is `6beebb80` vs `5c98cc70`. mac-test-1 is not a pass (`--exact` matched nothing). mac-test-2 is the red parent, remote rc 101. mac-test-3 is the fix, remote rc 0, 1 passed. mac-static-1 lost ssh and has no step rc.

## Rules

- This machine edits, runs git, and reads. No `cargo build`, `test`, `check`, `clippy`, or `run`. No `rustc`. No `just` recipe that builds or tests here. Static checks go through `/workspace/repo/bloomery/tools/mac-remote.sh` (the pool script on current main, not a branch copy). GPU work goes through `tools/box.sh` on ws.
- `cargo fmt` on this host is the wrong nightly. Apply a Mac rustfmt diff by hand.
- `POOL_MIN_GB` stays 15. Do not lower it. On 2026-10-14 07:28 KST the first two pool Macs were under that floor, so the statics were pinned to the third. Re-survey before the next pin. A pin bypasses the floor, so do not pin a Mac that is under 15 GB unless the lead names it.
- One queued or running mac-remote per branch. ssh rc 255 can leave the remote run and the track lock alive. Do not start a second until `--locks` shows that track's lock gone. `timeout 20` on `--locks` is the safe survey. Do not run `just mac-locks --clean`. Do not remove another track's lock or another owner's hold.
- Print a predicted wall before a Mac or ws job. Trust the log line `mac-remote: wall … remote rc N`, not a wrapper exit. A pipeline without `pipefail` can exit 0 when ssh returns 255.
- Commit before mac-remote. The logged HEAD is the commit. Uncommitted edits rsync and lie about the sha.
- Do not push `main`. A push to `les1` or `placea` is a `synchronize` event: Shingler reviews again, and an in-progress review of that pull request is cancelled.
- Lint ratchet stays 41. Do not bump `tools/lint-ratchet.txt`.
- No hostname, LAN, BMC, tailnet, credential, or `user@host` in a public file. Pool Macs are mac1, mac2, mac3 in alphabetical config order. Scrub `grok-bot-vm-*` to `local`.
- `gh api graphql` prints ANSI when stdout is redirected. Use `gh api --jq` or strip the escapes.
- HANDOFF under the untracked `specs/unitier/` wins on SHAs when this file and that tree disagree. This file is the pause note.

## Gotchas

- `mac-static` does not run `#[test]` bodies. A red unit test stays green there. The place-a red was a real `cargo test` on a Mac. `--exact` plus the bare function name matches nothing and exits 0.
- `plan_a` used to ignore a draft ask. `plan_tiers` with empty tiers already returned `DraftOffPlan`. The hole was plan (a), and Shingler now says plan (b) is the same hole.
- The screen fixture `crates/serve/tests/fixtures/screen-start-qwen38.log` must keep the nvtier keys at `=0` or `records.py` fails. T1 keys are `step_miss_layers`, `step_fill_wall_ns`, `step_hits_under_read_ns`, and record key `step_run_ns`.
- Rank-3 rebase conflict was only `NvTier::carve`. Main returns `NvTier` from an `ArenaKind` match. The caller wraps `Some`.
- `dirt()` allows a dirty tree. The evidence sha will be wrong if you static before committing.
- Do not `ps | rg` a pattern that matches its own command line.
- Evidence logs are untracked on the detached checkout. They are not in the pull requests.

## Next 3 steps

1. Fold `#135` review `5511063938`: `settle_fill`, and `PinToken` instead of `HeldClaim`. mac-static the new tip against `origin/main`. Push `les1` (force-with-lease only if you rebased). Leave the decode A/B.
2. Fold `#142` review `5510874863`: plan (b) refuses a draft, `plan_a_draft` calls `plan_tiers`, one reserve pin. mac-static. Push `placea` without rewriting the lead's tip unless you rebased. Leave the idle-card path alone.
3. Do not flip prefetch or resident-first, and do not take ws, until a measurement is assigned. Leave `tilekern`, the Q6_K assert, and the dispatch A/B.
