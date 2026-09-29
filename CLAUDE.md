# bloomery — agent context

@AGENTS.md

The file above is this repo's working contract and the source of truth. This file holds only what applies to Claude sessions.
No fact is written in two places.

## Tracking

Open work has one owner, `docs/plan-triage.md` (user decision 2026-09-27, `docs/rebuild.md` §7 decision 2).
Triage items are never moved or copied into MUL. The self-hosted tracker's **MUL** project holds only stage milestones,
machine changes and upstream work. It is `GADAK_HOME=$HOME/.gadak gadak --workspace gdk`, and
on the Mac use `/opt/homebrew/bin/gadak` (the dev build on PATH cannot read the mirror schema).
Comments are `gadak --workspace gdk comment <KEY> "<body>"` — not `issue comment`.
One issue per stage (MUL-1 to MUL-5), machine changes in MUL-6, upstream in MUL-7.
A round's detail (sitting tables, [derived] predictions, round and sitting names, per-commit history) lives in this
repo's docs. A [rig-log](https://github.com/midagedev/rig-log) `log/` section holds only what was measured on this machine,
how much, and what it showed, and links here for the detail; the triage item or the issue links that section.
Do not open `TODO.md`.

## Delegation

Rounds that can be written as a spec (implementation, gate authoring, investigation, reports) go to GLM-5.3 through the
`outsource` skill, at `--effort max` on the claude-code harness, one worktree per track (user, 2026-09-29).
Investigation-only rounds may go to agy. Opus subagents (Agent tool, `model:"opus"` stated) are for the exceptions:
vision, multi-turn cause narrowing, re-judging a verdict that disagrees with its instrument.
Round operation (waves) is `docs/plan.md` 「라운드 운영」, open items (cards, triage) are `docs/plan-triage.md`, and the prediction
and proof rules are "Derive first, measure the gap" in `AGENTS.md`. Copy the relevant clauses of `AGENTS.md` into the spec —
never assume the delegate reads that file. A delegate's report is not evidence: before reading it, count the tool calls in its
transcript and read `git status` of its worktree. After a round, the lead re-runs the gates under its own ownership.
Commits and pushes are lead-only.

## Writing to rig-log

Measurement records go in `log/` of `~/repo/rig-log`. That repo's rules are not loaded automatically in this session,
so read `~/repo/rig-log/CLAUDE.md` before writing a record. Four things that are easy to miss:

- **It is a public repo.** Before committing, grep for `192.168`, addresses in the `100.` range, `.ts.net` and `admin`. Never
  include BMC addresses or credentials, or the nvidia-bug-report archive from the box's `/root` (it contains the hostname).
- Write only what was measured; when something is wrong, do not delete it — strike it through and correct it.
- Disclose AI help in prose; no Claude badge.
- rig-log commits pass `tools/check-log.py` (its pre-commit hook); that file's header holds the limits.
