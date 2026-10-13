# Quant handoff

Paused for a replacement session. No Mac track lock and no workstation hold remain from this worker. No run of this worker is in flight. Nothing was pushed to main.

## Current goal and queue

The goal of this stretch is the Qwen3.6 DSpark serve wiring on `q36int`, and the staging gate on `q36gate` (PR #125). The Q6_K down launcher on `q6down` is carried here and is not the active confirm.

Do not start these. They stay queued, one confirm at a time, after a lead call:

- `gate-gpu-qwen3-serve`
- `gate-gpu-serve-start-ds41`
- image refusal
- `--parallel 2` refusal
- the load `draft=dspark` record group
- the 139 gate list (a `card_leg.rs` gate-paths row is held; do not launch it)
- FAIL-first mutants for staging, losslessness, and the acceptance floor
- any rebase of `q6down`, `q36int`, or `q36gate`

The widths timings confirm is already green. Do not relaunch it.

## Open PRs

PR #125 is open: https://github.com/midagedev/bloomery/pull/125

- Title: gpu-gates: gate_qwen35_dspark checks staging before the draft, and clauses 1–2 sit on DenseLayers
- Head: `75e25358085a91e6a1afaa074784ca0e5ac255ea` on `q36gate`. `origin/q36gate` matches that tip. The worktree is clean.
- Shingler: APPROVED on that commit at 2026-10-13T22:43:58Z. The review says the greedy duplication is fixed, both binaries import the shared helper, and the verdict is CLEAN.
- CI on that head: label, contrib-check t0, gate, diagnostics, review, and post are COMPLETED / SUCCESS.
- Non-blocking and still unrun: FAIL-first evidence for staging (`gate_qwen35_dspark.rs` around line 722) and for losslessness and the acceptance floor (around line 981). Do not run those mutants unless the lead asks.
- Re-review was requested in https://github.com/midagedev/bloomery/pull/125#issuecomment-6156669024. That comment says the bloomery assistant wrote the nudge and this worker wrote the fold.
- Merge is the lead's. This worker did not merge.

No pull request exists for `q36int` or `q6down`. This wrap-up did not open one.

## Branches and tips

`origin/main` at wrap-up was `f1a3e9e44a8909f4c25fa216818cb6b04400d728`. Do not rebase onto it unless a merge conflicts and the lead asks.

| Branch | Tip | Upstream | Merge-base with origin/main | Role |
| --- | --- | --- | --- | --- |
| `q36gate` | `75e25358085a91e6a1afaa074784ca0e5ac255ea` | matches `origin/q36gate` | `89d8eebbd31d1dc0499789d41649356e92f2004f` | PR #125. Shared greedy fold. Pushed before this wrap-up. |
| `q36int` | `97fb58f670821b194c15f0ad61aff643e64f16f4` | `origin/q36int` (first push, this wrap-up) | `31abb2d5d3c1143edb9a7bdac68bf91aadef73e9` | Serve wiring. Timings green. Not PR-ready. |
| `q6down` | this commit | `origin/q6down` (first push, this wrap-up) | `31abb2d5d3c1143edb9a7bdac68bf91aadef73e9` | Q6_K down launcher plus this handoff. Not PR-ready. |

`q36int` history under the tip, oldest first: `a3c605d9` WIP(handoff), `9425e98e` replay refusal, `590c1930` rustfmt, `7997559f` experts=host and usage exit 64, `66b355db` helper cap, `97fb58f6` draft-width record. The tip commit is "gpu-gates: the qwen3 seat registers the draft width record". It calls `drafted::serves_widths` from the qwen3 seat and lists `groups::drafted::WIDTH` on `BLOOMERY_SERVE_QWEN3`.

`q6down` product commits under this handoff: `5bbba92e` (the common card leg runs a Q6_K down) and `248b207f83cf073fbf660487f2ac277d14a46eb1` (the Q6_K down launcher takes the caller's fault sink, and the hand launch names Q5_K and Q6_K). The static that finished on the pre-rebase tip does not cover `248b207f`.

## In-flight runs and logs

None. At wrap-up every Mac slot was free and no track lock was present. The workstation hold directory was empty. Local pid files for older Mac and gate runs were dead. The local build lock file was not held. Do not clean another track's lock.

Closed logs live only as untracked notes on the main checkout, under `specs/quant/evidence/`. They are not in this commit. Do not scrub a closed log into a commit, and do not re-run a log that already has an EXIT line.

- `specs/quant/evidence/q36gate/mac-static-greedy.log` — fold `75e25358` versus parent `c42ecc66`. EXIT:0. Predicted 420 s. Measured wall 137 s. Lint warning count 41, equal to the ratchet. Remote rc 0.
- `specs/quant/evidence/q36gate/mac-static-held2.log` — `c42ecc66`. EXIT:0. Wall 123 s. Predicted 260 s.
- `specs/quant/evidence/q36gate/mac-static-held.log` — EXIT:75. SSH loss, not a product failure.
- `specs/quant/evidence/q36gate/mac-static-89d8.log` — EXIT:101. `CardDspark::new` arity. Fixed later by `c42ecc66`. Do not relaunch.
- `specs/quant/evidence/q36int/timings-widths.log` — attempt 3 on `97fb58f6`. EXIT:0. Job s57 rc 0. HTTP 200. `draft_n` 112. `draft_n_accepted` 31. `predicted_n` 64. Predicted wall 300 s. Wrapper about 69 s. Release build 10.67 s. A poll line after EXIT:0 is the client losing a poll. It is not a red confirm.
- `specs/quant/evidence/q36int/timings.log` — attempt 2 on `66b355db`. EXIT:1. HTTP 500. The seat had not registered `mtp_width`. Superseded by attempt 3.
- `specs/quant/evidence/q36int/timings-invoke.log` — attempt 1. EXIT:1 before load. Two model names. Superseded.
- Cards: `specs/quant/evidence/q36int/timings.card.md` matches attempt 3. `specs/quant/evidence/q36int/push.card.md` is stale: it said not to push. The branch is now on origin. The "not PR-ready" reasons in that card still hold.

## Decisions and rules

- This machine edits, uses git, and reads docs. It does not build or test. No local cargo, rustc, or just recipe that builds or tests. Statics go through the main tree's remote runner, one run per branch. GPU confirms go through the workstation queue, one job at a time. Print the predicted wall before either. Exit 75 is reachability or SSH loss, not a red test. A wrapper exit of 0 does not prove the inner EXIT line.
- Commit only on `q6down`, `q36int`, and `q36gate`. Never push main, land on main, tag, or open a pull request unless the lead asks. The lead opens pull requests and merges after Shingler approves. Keep branches small. Do not report a branch ready while a WIP subject, a red static, or an unproven serve confirm is on it.
- Do not move the main checkout. Untracked `specs/quant/` notes on that checkout stay untracked. This file is the committed handoff.
- Quantization and kernel defaults follow a measured winner. A held gate or floor still moves only after that measurement and a lead call. This stretch moved no default. Floor E stays at least 2.0. The lint ratchet stays 41. Equal to 41 is green. Do not invent a Cargo feature. Do not lower the pool free-space floor. Do not edit check criteria because a self-test timed out or missed a pid. Do not edit the remote runner or the "after 0 s" self-test.
- Accept a slot k count when the kept count is greater than k. The timings scale note (an earlier generate arm proposed 119 and accepted 18) is not a cutoff. Attempt 3 accepted 31 of 112 and passed.
- Held, do not decide: the per-image replay signature (`specs/quant/evidence/q36int/replay-per-image.patch` stays unapplied), the `card_leg.rs` gate-paths row, and anything the lead has marked out of this role (0.3.2 is already tagged, board publish, the depth item, IQ2_K).
- `DenseLayers::replayable` refuses only when both staged images are empty. Do not revert that. `CardDspark::new` takes the held flag and the taken count.
- The shared greedy helper is `crates/gpu-gates/src/bin/shared/greedy.rs` on `q36gate`. `gate_qwen35_dspark` and `gate_qwen35moe_verify` include it. The e2e binary keeps its own model-level greedy. `greedy_tapped` stays in the verify gate.
- Fold comments say the bloomery assistant wrote the nudge and this worker wrote the fold.
- New repository text is English. Do not put hosts, network addresses, home paths, tokens, or pool aliases in this file, in NOW, or in a commit.
- Do not remove another track's lock. Do not start a second remote run for a branch that already has one. Do not edit a tree whose remote run is still live.

## Gotchas

- The workstation runner exports `BLOOMERY_REF_MODEL` from the default profile. A serve confirm that also passes `-m` must unset `BLOOMERY_REF_MODEL` around the gate script, or the server sees two model names and exits before load. The unset wraps that invocation. The gate script does not re-export the variable.
- `drafted::print_widths` requires the `mtp_width` record. The qwen3 seat must both call `drafted::serves_widths` and list `groups::drafted::WIDTH`. The call alone panics when the list lacks the group. `ds41` and `qwen38` already do both. Do not add other record groups unless a later panic names them.
- `print_widths` prints only in Cost mode when passes are greater than 0. A green confirm does not require an `mtp_width` line in the log. The pass bar is HTTP 200, `draft_n` greater than 0, and `draft_n_accepted` an integer no greater than `draft_n`. There is no tokens-per-second cutoff.
- The six older gate files on `q36int` (`dense.rs`, `dspark/mod.rs`, `gate_qwen35_dspark.rs`, `gate-paths.tsv`, `gpu-gate.sh`, `gate-batch.sh`) are not the #125 fold. Do not treat a push of `q36int` as an update to #125.
- `a3c605d9` on `q36int` still has a WIP subject and a Co-Authored-By trailer. The wrap-up pushed that history as it stood. Do not call the branch PR-ready. Do not rewrite it unless the lead asks.
- `q36int` and `q6down` were never on origin before this wrap-up. Both pushes were new branches, not force-pushes.

## Next 3 steps

1. Leave #125 for the lead to merge. The head `75e25358085a91e6a1afaa074784ca0e5ac255ea` is approved and the listed checks are green. Do not start another fold unless a merge into current main conflicts.
2. Treat `origin/q36int` at `97fb58f670821b194c15f0ad61aff643e64f16f4` as pushed and not PR-ready. The lead decides the WIP subject, the trailer, and the older gate files versus #125 before any pull request.
3. Keep the unrun serve confirms queued, one at a time, and do not relaunch the widths timings confirm. Do not launch the 139. `q6down` is on origin with this handoff and is not PR-ready until a static covers `248b207f83cf073fbf660487f2ac277d14a46eb1`.
