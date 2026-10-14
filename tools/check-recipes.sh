#!/usr/bin/env bash
# justfile 레시피 점검 — 맥과 리눅스 호스트 나이틀리에서 돈다(grep뿐, 빌드 없음).
# 막는 실패: 게이트 줄 뒤에 붙은 `||`가 종료 코드를 삼키는 것. ef9e579가 `cargo test … || echo "TIMED OUT"`로
# 열세 게이트 전부를 "빨강이어도 0"으로 만들었다(2026-09-20, tools/gate.sh 머리말 참조).
# 게이트의 종료 코드는 tools/gate.sh가 소유한다 — 시험을 돌리는 레시피 줄에 `||`가 있으면 빨강.
set -euo pipefail
# The self-tests below run gate-batch.sh, which reads BLOOMERY_BOX_ENV as its caller's box env; a batch lane
# that runs this check passes its card there. A static check's verdict does not depend on the caller's box env.
unset BLOOMERY_BOX_ENV
JF="$(cd "$(dirname "$0")/.." && pwd)/justfile"
bad=$(grep -nE '(cargo (oxide )?test|tools/gate\.sh).*\|\|' "$JF" || true)
if [ -n "$bad" ]; then
  echo "check-recipes: a test-running recipe line carries '||' — it swallows the gate's exit code:" >&2
  echo "$bad" >&2
  exit 1
fi
raw=$(grep -nE "box\.sh '.*cargo (oxide )?test" "$JF" || true)
if [ -n "$raw" ]; then
  echo "check-recipes: bare 'cargo test' in a box recipe — route it through tools/gate.sh (bound + exit code):" >&2
  echo "$raw" >&2
  exit 1
fi
# GPU 게이트 락의 소유자는 tools/gpu-gate.sh 하나다. 레시피가 락을 직접 잡으면 그 줄의 바이너리는 상한 없이 돈다
# — 매달린 GPU 게이트 하나가 락을 쥐면 그 락을 기다리는 트랙이 전부 선다.
lock=$(grep -nE '^[^#]*bloomery-gate\.lock' "$JF" || true)
if [ -n "$lock" ]; then
  echo "check-recipes: a recipe takes the GPU gate lock itself — run the binary through tools/gpu-gate.sh (lock + bound + exit code):" >&2
  echo "$lock" >&2
  exit 1
fi
# A host binary's bound and exit code belong to tools/host-gate.sh. A recipe that runs a
# target/release binary behind its own `timeout` spells the bound and the timeout's exit code again,
# and prints nothing when the bound is hit.
bare=$(grep -nE '^[^#]*timeout[^#]*target/release/' "$JF" || true)
if [ -n "$bare" ]; then
  echo "check-recipes: a recipe runs a target/release binary behind its own timeout — run it through tools/host-gate.sh (bound + exit code):" >&2
  echo "$bare" >&2
  exit 1
fi
# A recipe that reads a draft's counts (`accepts=`) pins what the draft verifies: each drafted command
# (`BLOOMERY_DRAFT=<draft>`) carries `BLOOMERY_MTP_WIDTH=` right before it. Unpinned, the width chooser's
# `cost` mode decides on host wall time, and under load the run verifies only its warm-up passes while
# the counts clause still passes.
unpinned=$(grep -nE '^[^#]*accepts=' "$JF" | grep -E '(^|[^=_[:alnum:]])BLOOMERY_DRAFT=' |
  sed -E 's/BLOOMERY_MTP_WIDTH=[^ ]+ BLOOMERY_DRAFT=//g' | grep -E 'BLOOMERY_DRAFT=' | cut -c1-160 || true)
if [ -n "$unpinned" ]; then
  echo "check-recipes: a recipe reads draft counts with a drafted command whose width is not pinned — put BLOOMERY_MTP_WIDTH=fixed right before its BLOOMERY_DRAFT=:" >&2
  echo "$unpinned" >&2
  exit 1
fi
# Every recipe's cargo targets, features and runner binaries against the workspace (cargo metadata, no
# build): a `--bin`, `--test` or `-p` that names nothing, a feature the package lacks, a gpu-gate.sh or
# host-gate.sh name the recipe does not build, a named script that is not in the tree, a gate-* recipe
# with no cargo target, a build of a shipped bin (tools/release/build.sh's BINS) with a feature list
# but that file's. The parser is tools/recipes.py, the one tools/affected-gates.sh uses; its own
# tests (--self-test) run here too, so the parser the lead's batch lists come from is tested wherever
# this check runs.
# The check of the recipes against the workspace is the first of the parallel blocks below (blk_recipescheck).
# The blocks below are independent — each runs in its own subshell on fixtures in its own temp
# directory — so they run as parallel background jobs and report in this file's order after a barrier:
# each block's own output (its last line where the serial form printed only that), its full output and
# its named error on failure, and the first red block's own rc ends the check. The wall is the slowest block's:
# ~140 s with the Mac at load 65-100 (measured 2026-10-12; ~780 s there before the long self-tests, recipes.py
# and gate-batch.sh, ran their sections as parallel processes). The barrier holds every assertion the serial
# form held.
B=$(mktemp -d)
trap 'rm -rf "$B"' EXIT

# blk_NAME: one block. Its stdout is what the serial form printed; a failure prints its named error to
# stderr and returns nonzero. `$(dirname "$0")` still names this script's directory inside a function.
blk_recipescheck() {
  python3 "$(dirname "$0")/recipes.py" check
}
blk_selftest() {
  if ! python3 "$(dirname "$0")/recipes.py" --self-test; then
    echo "check-recipes: the recipes.py self-test failed" >&2
    return 1
  fi
}
blk_smoke() {
  if ! "$(dirname "$0")/gate-batch.sh" --smoke --dry-run > /dev/null; then
    echo "check-recipes: the gate-batch smoke run failed" >&2
    return 1
  fi
}
blk_cardorder() {
  local classes rc
  classes=$(mktemp) || return 1
  "$(dirname "$0")/gate-batch.sh" --classes > "$classes" || { rm -f "$classes"; return 1; }
  python3 - "$JF" "$classes" << 'PY'
import json, re, subprocess, sys

jf, classes = sys.argv[1], sys.argv[2]
dump = json.loads(subprocess.run(["just", "--justfile", jf, "--dump", "--dump-format", "json"],
                                 check=True, capture_output=True, text=True).stdout)
recipes, assigns = dump["recipes"], dump["assignments"]
PRECHECK = "tools/ref/card-precheck.sh"
BUILD = re.compile(r"\bcargo (?:oxide )?(?:build|run|test)\b|\bjust build-[A-Za-z0-9_-]+")


def body(name):
    """The recipe's text with the justfile's string variables in place; a parameter and a computed variable (`arch`) stay {{}}."""
    out = []
    for frags in recipes[name]["body"]:
        line = ""
        for f in frags:
            if isinstance(f, str):
                line += f
            elif len(f) == 1 and f[0][0] == "variable" and f[0][1] in assigns and isinstance(assigns[f[0][1]]["value"], str):
                line += assigns[f[0][1]]["value"]
            else:
                line += "{{}}"
        out.append(line)
    return "\n".join(out)


def dep_builds(name, seen):
    for d in recipes[name]["dependencies"]:
        if d["recipe"] not in seen:
            seen.add(d["recipe"])
            if BUILD.search(body(d["recipe"])):
                yield d["recipe"]
            yield from dep_builds(d["recipe"], seen)


bad, via_dep, n_timed, n_pre = [], [], 0, 0
with open(classes, encoding="utf-8") as fh:
    for row in fh:
        f = row.rstrip("\n").split("\t")
        if len(f) != 5 or f[0] not in recipes:
            sys.exit(f"check-recipes: gate-batch.sh --classes printed a row it should not: {row!r}")
        if f[1] != "T":
            continue
        n_timed += 1
        text = body(f[0])
        build, pre = BUILD.search(text), text.find(PRECHECK)
        if build and (pre < 0 or pre > build.start()):
            bad.append(f"  {f[0]}: `{build.group(0)}` runs before {PRECHECK} ({'after it' if pre >= 0 else 'absent'})")
        elif build:
            n_pre += 1
        deps = list(dep_builds(f[0], set()))
        if deps:
            via_dep.append(f"{f[0]} ({', '.join(deps)})")
run = sorted(n for n in recipes if n.startswith("gate-") and re.search(r"\bcargo (?:oxide )?run\b", body(n)))
if run:
    print("check-recipes: a gate-* recipe runs its binary with `cargo run` — no bound and no runner's exit code; "
          "build it and run it through tools/host-gate.sh or tools/gpu-gate.sh:", file=sys.stderr)
    print("\n".join(f"  {n}" for n in run), file=sys.stderr)
    sys.exit(1)
if bad:
    print("check-recipes: a timed recipe builds before its card check — put {{precheck}} first in its box "
          "command (a forgotten card then costs no build):", file=sys.stderr)
    print("\n".join(bad), file=sys.stderr)
    sys.exit(1)
print(f"check-recipes: {n_timed} timed recipes, {n_pre} build in their box command after the card precheck")
if via_dep:
    print(f"check-recipes: {len(via_dep)} timed recipes build in a just dependency before their box command, "
          f"where the precheck cannot come first: {'; '.join(via_dep)}")
PY
  rc=$?
  rm -f "$classes"
  return "$rc"
}
# The card and lease stub tests (tools/ref/card-tests/run.sh): card.py, lease_take's refusals,
# lease-hold.sh and gpu-ab.py's card check, against a copy of that code and a lock file of their own.
blk_cardtests() {
  if ! cards=$("$(dirname "$0")/ref/card-tests/run.sh" 2>&1); then
    echo "$cards" >&2
    echo "check-recipes: the card tests failed" >&2
    return 1
  fi
  echo "${cards##*$'\n'}"
}
# lease.sh's own self-test: lease_gpu_idle (the take's timing-card check) and the witness tag, on a
# stub nvidia-smi in a temp dir, no card, no lock, no /proc.
blk_lease() {
  if ! lse=$(bash "$(dirname "$0")/ref/lease.sh" --self-test 2>&1); then
    echo "$lse" >&2
    echo "check-recipes: the lease self-test failed" >&2
    return 1
  fi
  echo "${lse##*$'\n'}"
}
# box-tracks.sh deletes remote track directories: its selection (only the names given, never a new stale
# the reader did not see, a bare --remove refused) is tested on fixed input, no ssh.
blk_boxtracks() {
  if ! bt=$(bash "$(dirname "$0")/box-tracks.sh" --self-test 2>&1); then
    echo "$bt" >&2
    echo "check-recipes: the box-tracks self-test failed" >&2
    return 1
  fi
  echo "${bt##*$'\n'}"
}
# load-groups.sh decides which arms of a depth runner share one load, and their order: its grouping and
# rotation are tested on fixed keys, no process started.
blk_loadgroups() {
  if ! lg=$(bash "$(dirname "$0")/ref/load-groups.sh" --self-test 2>&1); then
    echo "$lg" >&2
    echo "check-recipes: the load-groups self-test failed" >&2
    return 1
  fi
  echo "${lg##*$'\n'}"
}
# lcpp-fit.sh builds the depth runners' llama.cpp fit arms and reads what the fit chose: its flag
# rewrite, its --help probe (stub binaries) and its column (fixture output) are tested here, no box.
blk_lcppfit() {
  if ! lf=$(bash "$(dirname "$0")/ref/lcpp-fit.sh" --self-test 2>&1); then
    echo "$lf" >&2
    echo "check-recipes: the lcpp-fit self-test failed" >&2
    return 1
  fi
  echo "${lf##*$'\n'}"
}
# cold-blocks.sh is the depth runners' cold tag and engine-block planner: its bound, its flag rewrite, its
# order refusal and its block plan on fixed arms are tested here, no box.
blk_coldblocks() {
  if ! cb=$(bash "$(dirname "$0")/ref/cold-blocks.sh" --self-test 2>&1); then
    echo "$cb" >&2
    echo "check-recipes: the cold-blocks self-test failed" >&2
    return 1
  fi
  echo "${cb##*$'\n'}"
}
# lever-arms.sh is the depth runners' lever-arm list check and their pre-lease `--levers` probe (lever_probe): its
# words and its args after `--`, no box.
blk_leverarms() {
  if ! la=$(bash "$(dirname "$0")/ref/lever-arms.sh" --self-test 2>&1); then
    echo "$la" >&2
    echo "check-recipes: the lever-arms self-test failed" >&2
    return 1
  fi
  echo "${la##*$'\n'}"
}
# slots-arm.sh is the depth runners' aggregate arm (<arm>@BLOOMERY_GEN_SLOTS=N): its N, its plain twin, the
# environment's refusal, its time pass and residency clauses and the plain tables' labels, no box.
blk_slotsarm() {
  if ! sat=$(bash "$(dirname "$0")/ref/slots-arm.sh" --self-test 2>&1); then
    echo "$sat" >&2
    echo "check-recipes: the slots-arm self-test failed" >&2
    return 1
  fi
  echo "${sat##*$'\n'}"
}
# lcpp-warm.sh is the depth runners' llama-server arms: its flag translation (against V4.1's LCPP_CLI_FLAGS),
# its refusals, the context, the --help probe and a stub server's start, requests, checks and stop (python3
# and curl, 127.0.0.1) are tested here, no box.
blk_lcppwarm() {
  if ! lw=$(bash "$(dirname "$0")/ref/lcpp-warm.sh" --self-test 2>&1); then
    echo "$lw" >&2
    echo "check-recipes: the lcpp-warm self-test failed" >&2
    return 1
  fi
  echo "${lw##*$'\n'}"
}
# q38-3090-server.sh is the lone-3090 Qwen3.8 server run (the discovery search's M4): its refusals (a set lever, a card that
# is not the one 3090, a relative or existing out dir), the chats' records and the stop by the server's own pid, against a stub
# gate runner and server (python3, curl, 127.0.0.1), are tested here, no box.
blk_q38srv() {
  if ! q38s=$(bash "$(dirname "$0")/ref/q38-3090-server.sh" --self-test 2>&1); then
    echo "$q38s" >&2
    echo "check-recipes: the q38-3090-server self-test failed" >&2
    return 1
  fi
  echo "${q38s##*$'\n'}"
}
# mac-check.sh runs the check and lint recipes' box commands on the Mac: its derivation from those
# recipes, its refusals, the ratchet and the prerequisite checks (a fake HOME) are tested here, no cargo.
blk_maccheck() {
  if ! mc=$(bash "$(dirname "$0")/mac-check.sh" --self-test 2>&1); then
    echo "$mc" >&2
    echo "check-recipes: the mac-check self-test failed" >&2
    return 1
  fi
  echo "${mc##*$'\n'}"
}
# gate-batch.sh's placement rules (the v41-load lane, a batch or a solo recipe's arm as an item, a cold
# build's times row) on a fixture justfile and fixture times rows, no box.
blk_gatebatch() {
  if ! gbt=$(bash "$(dirname "$0")/gate-batch.sh" --self-test 2>&1); then
    echo "$gbt" >&2
    echo "check-recipes: the gate-batch self-test failed" >&2
    return 1
  fi
  echo "${gbt##*$'\n'}"
}
# gpu-gate.sh's card choice, its lease refusals and the V4.1 load lock, against lock files of its own and
# a stub nvidia-smi (flock and timeout stand-ins where the host has none, as on the Mac), no card.
blk_gpugate() {
  # The full transcript stays for the proc-leg read after the barrier. The block's own line is still the
  # last line, as the other blocks print.
  local ggt rc=0
  ggt=$(bash "$(dirname "$0")/gpu-gate.sh" --self-test 2>&1) || rc=$?
  printf '%s\n' "$ggt" > "$B/gpugate.full"
  if [ "$rc" != 0 ]; then
    printf '%s\n' "$ggt" >&2
    echo "check-recipes: the gpu-gate self-test failed" >&2
    return 1
  fi
  echo "${ggt##*$'\n'}"
}
# stack-watch.sh (gpu-gate.sh's BLOOMERY_GATE_STACKS): output passed through, a quiet stub dumped and ended by
# its comm among the spawned command's descendants only, a quiet command with no such process left alone.
blk_stackwatch() {
  if ! swt=$(bash "$(dirname "$0")/ref/stack-watch.sh" --self-test 2>&1); then
    echo "$swt" >&2
    echo "check-recipes: the stack-watch self-test failed" >&2
    return 1
  fi
  echo "${swt##*$'\n'}"
}
# ptx-scan.sh's global load widths, legacy columns, entry boundaries and scan readers on synthetic PTX, no box.
blk_ptxscan() {
  if ! pts=$(bash "$(dirname "$0")/ptx-scan.sh" --self-test 2>&1); then
    echo "$pts" >&2
    echo "check-recipes: the ptx-scan self-test failed" >&2
    return 1
  fi
  echo "${pts##*$'\n'}"
}
# ptx-spill-check.sh's table reading (a process-substitution table read for every binary, a binary with no
# pinned row) and its verdict lines, through a stub ptx-scan.sh on fixed scans, no build.
blk_ptxspill() {
  if ! psc=$(bash "$(dirname "$0")/ptx-spill-check.sh" --self-test 2>&1); then
    echo "$psc" >&2
    echo "check-recipes: the ptx-spill-check self-test failed" >&2
    return 1
  fi
  echo "${psc##*$'\n'}"
}
# scan-args.sh is the three scan recipes' refusal of a cargo feature given where the scan's own words go
# (ptx-scan, sass-scan, lds-scan): its cases on a fixed feature list and the real crates/gpu-gates table.
blk_scanargs() {
  if ! sa=$(bash "$(dirname "$0")/scan-args.sh" --self-test 2>&1); then
    echo "$sa" >&2
    echo "check-recipes: the scan-args self-test failed" >&2
    return 1
  fi
  echo "${sa##*$'\n'}"
}
# narrow-scan.sh is `just narrow`'s box orchestration: the runner it ships to each side (one build per feature list, the
# scanner embedded verbatim), the cut of the box's output into per-bin scan logs, the one-run-at-a-time lock on the
# persistent remote dirs, its usage refusals, and the drift check against the ptx-scan recipe's build lines. No box.
blk_narrowscan() {
  if ! nsc=$(bash "$(dirname "$0")/narrow-scan.sh" --self-test 2>&1); then
    echo "$nsc" >&2
    echo "check-recipes: the narrow-scan self-test failed" >&2
    return 1
  fi
  echo "${nsc##*$'\n'}"
}
# lds-scan.sh's rows (its PTX and SASS counts over a fixture binary, stub extractor, ptxas and cuobjdump),
# its filter, its failed-scan banner and its usage refusal, no box.
blk_ldsscan() {
  if ! ls=$(bash "$(dirname "$0")/lds-scan.sh" --self-test 2>&1); then
    echo "$ls" >&2
    echo "check-recipes: the lds-scan self-test failed" >&2
    return 1
  fi
  echo "${ls##*$'\n'}"
}
# mutant-run.sh's kill, survive, not-built and no-Compiling verdicts, its restore checks (a broken copy, an
# edit during the run, a TERM) and its refusals, in a temp git repo with a fake gate, no box.
blk_mutantrun() {
  if ! mrt=$(bash "$(dirname "$0")/mutant-run.sh" --self-test 2>&1); then
    echo "$mrt" >&2
    echo "check-recipes: the mutant-run self-test failed" >&2
    return 1
  fi
  echo "${mrt##*$'\n'}"
}
# mac-static.sh runs the round loop's lanes (fmt-check, lint, the scoped combos and the check scripts in
# parallel): its lane orchestration, its rc/wall capture and its red-step naming are tested against
# /bin/true, /bin/false and sleep stubs, no cargo, no check script.
blk_macstatic() {
  if ! mst=$(bash "$(dirname "$0")/mac-static.sh" --self-test 2>&1); then
    echo "$mst" >&2
    echo "check-recipes: the mac-static self-test failed" >&2
    return 1
  fi
  echo "${mst##*$'\n'}"
}
# mac-remote.sh sends a tree to the second Mac and runs mac-static there: its argument, config-file, track-name,
# dirty-tree, lock and ledger rules, and (on macOS) the whole flow against a stub ssh that runs the "remote" in a
# fake home, are tested here, no remote and no cargo.
blk_macremote() {
  if ! mrm=$(bash "$(dirname "$0")/mac-remote.sh" --self-test 2>&1); then
    echo "$mrm" >&2
    echo "check-recipes: the mac-remote self-test failed" >&2
    return 1
  fi
  echo "${mrm##*$'\n'}"
}
# box.sh's hold on this side (~/.cache/bloomery/box-hold/<owner>, the header): its refusal before any rsync or ssh, the owner's and a
# read's pass, the `--hold up|down` rules and a looping caller stopping at its next call, on a scratch copy under a fake HOME with
# ssh and rsync stubs, no box.
blk_boxsh() {
  if ! bxs=$(bash "$(dirname "$0")/box.sh" --self-test 2>&1); then
    echo "$bxs" >&2
    echo "check-recipes: the box.sh self-test failed" >&2
    return 1
  fi
  echo "${bxs##*$'\n'}"
}
# seed-target.sh clones a warm tree's target/ into a new worktree and restores the seed's mtime on the files that are equal:
# its refusals (a DST with a target, a SRC without one, two volumes, a host that cannot clone) and the mtime pass (only
# equal files) are tested on two scratch trees, no cargo.
blk_seedtarget() {
  if ! stt=$(bash "$(dirname "$0")/seed-target.sh" --self-test 2>&1); then
    echo "$stt" >&2
    echo "check-recipes: the seed-target self-test failed" >&2
    return 1
  fi
  echo "${stt##*$'\n'}"
}
# wt.sh adds a worktree and seeds it from the nearest warm tree, and retires one through mac-disk: the refusals of `new` (an
# existing path, an existing branch, an add that fails or creates nothing, each with the repo unchanged and the next step
# not run, and a mutant of each guard that does run it), the seed choice, the read-only seed, and the in-flight rule of
# `retire`, on scratch git repos with stub seed-target and mac-disk, no cargo, no box.
blk_wt() {
  if ! wts=$(bash "$(dirname "$0")/wt.sh" --self-test 2>&1); then
    echo "$wts" >&2
    echo "check-recipes: the wt self-test failed" >&2
    return 1
  fi
  echo "${wts##*$'\n'}"
}
# carry.sh is the release's carry check (tools/release/carry.tsv against the commit released): its carried, lacking,
# pending, unknown-hash and malformed-row verdicts in a temp git repo, no box. The release-build recipe runs it before
# its box command: a recipe that stops calling it ends the check silently, so the call is held here.
blk_carry() {
  local recipe carry_at box_at
  if ! cy=$(bash "$(dirname "$0")/release/carry.sh" --self-test 2>&1); then
    echo "$cy" >&2
    echo "check-recipes: the carry self-test failed" >&2
    return 1
  fi
  recipe=$(awk '/^release-build /{f=1; next} f && /^[^ ]/{exit} f' "$JF")
  carry_at=$(grep -n 'tools/release/carry\.sh' <<< "$recipe" | head -n1 | cut -d: -f1 || true)
  box_at=$(grep -n 'tools/box\.sh' <<< "$recipe" | head -n1 | cut -d: -f1 || true)
  if [ -z "$carry_at" ] || [ -z "$box_at" ] || [ "$carry_at" -gt "$box_at" ]; then
    echo "check-recipes: the release-build recipe must run tools/release/carry.sh before its tools/box.sh line (carry line '${carry_at:-none}', box line '${box_at:-none}')" >&2
    return 1
  fi
  echo "${cy##*$'\n'}"
}
# build.sh's host-code scan (an instruction past x86-64-v3 outside a runtime-dispatched function) on fixture listings: the
# release check runs only on the box, so its scan is held here.
blk_releasebuild() {
  if ! rb=$(bash "$(dirname "$0")/release/build.sh" --self-test 2>&1); then
    echo "$rb" >&2
    echo "check-recipes: the release build.sh self-test failed" >&2
    return 1
  fi
  echo "${rb##*$'\n'}"
}
# Every Python tool's own tests, on the Mac and the Linux host nightly (seconds in all): a self-test that no check runs rots. A tool
# that grows one is listed here, and the comparison below fails on one that is not. The tools run as
# parallel jobs (they are independent processes on their own temp fixtures); their failures land in one
# file and are reported together.
blk_pytools() {
  local root t f arg out pf
  selftests=(
    "tools/bloomery/manifest.py --self-test"
    "tools/bloomery/records.py --self-test"
    "tools/bloomery/rows.py --self-test"
    "tools/bloomery/route_trace.py --self-test"
    "tools/bloomery/selftest.py --self-test"
    "tools/board/board.py --self-test"
    "tools/boxq-check.py --self-test"
    "tools/boxq.py --self-test"
    "tools/check-comment-only.py --self-test"
    "tools/check-defaults.py --self-test"
    "tools/check-ptx-names.py --self-test"
    "tools/check-records.py --self-test"
    "tools/flow/ds41_prefill.py --self-test"
    "tools/flow/pplb.py --self-test"
    "tools/flow/q38_step.py --self-test"
    "tools/flow/q38width.py --self-test"
    "tools/flow/routes.py --self-test"
    "tools/hf-arch-check.py --self-test"
    "tools/mac-disk.py --self-test"
    "tools/ref/asm-census.py --self-test"
    "tools/ref/check-int-twins.py --self-test"
    "tools/ref/clef/e2e.py --self-test"
    "tools/ref/clef_ref.py --self-test"
    "tools/ref/clefvis/check_hidden.py --self-test"
    "tools/ref/clefvis/check_preproc.py --self-test"
    "tools/ref/dma-dram-share.py --self-test"
    "tools/ref/draft-accept.py --self-test"
    "tools/ref/draft-vocab.py --self-test"
    "tools/ref/ds41copy.py --self-test"
    "tools/ref/ds41pp.py self-test"
    "tools/ref/flat-imatrix.py --self-test"
    "tools/ref/gguf-ranges.py --self-test"
    "tools/ref/gguf-respell.py --self-test"
    "tools/ref/hidden-diff.py --self-test"
    "tools/ref/lev_ref.py --self-test"
    "tools/ref/loop-floor.py --self-test"
    "tools/ref/nsys-bridge.py --self-test"
    "tools/ref/ptx-canon.py --self-test"
    "tools/ref/qdot-rate-floors.py --self-test"
    "tools/ref/route-trace-chat.py --self-test"
    "tools/ref/router-coverage.py --self-test"
    "tools/ref/router-residency.py --self-test"
    "tools/ref/set-diff.py --self-test"
    "tools/ref/window-union.py --self-test"
    "tools/review/diagnostics.py --self-test"
    "tools/review/labels.py --self-test"
    "tools/review/ledger.py --self-test"
    "tools/review/post.py --self-test"
    "tools/review/run.py --self-test"
    "tools/review/signals.py --self-test"
    "tools/review/usage.py --self-test"
    "tools/type-census.py --self-test"
    "tools/verdict-diff.py --self-test"
  )
  root="$(cd "$(dirname "$0")/.." && pwd)"
  listed=$(printf '%s\n' "${selftests[@]}" | cut -d' ' -f1 | sort)
  found=$(cd "$root" && grep -rlE -e '--self-test|"self-test"' --include='*.py' tools | grep -vx 'tools/recipes.py' | sort)
  if [ "$listed" != "$found" ]; then
    echo "check-recipes: the Python tools with a self-test and the list this check runs differ (< listed, > found):" >&2
    diff <(echo "$listed") <(echo "$found") >&2 || true
    return 1
  fi
  pf=$B/pytools.fails
  : > "$pf"
  for t in "${selftests[@]}"; do
    ( read -r f arg <<< "$t"
      if ! out=$(cd "$root" && python3 "$f" "$arg" 2>&1); then
        # A Python module this host lacks is one named line, never the tool's traceback: the host
        # nightly's install puts python3-numpy and python3-pil in place for these (tools/nightly/install.sh).
        if m=$(grep -m1 "ModuleNotFoundError: No module named" <<< "$out"); then
          mod=$(printf '%s' "$m" | sed -n "s/.*No module named '\([^']*\)'.*/\1/p")
          case ${mod%%.*} in numpy) pkg=python3-numpy ;; PIL) pkg=python3-pil ;; *) pkg= ;; esac
          printf 'check-recipes: %s %s needs the Python module %s this host lacks%s\n' \
            "$f" "$arg" "$mod" "${pkg:+ — the host nightly installs $pkg}" >> "$pf"
        else
          printf '%s\n%s\n' "$out" "check-recipes: $f $arg failed" >> "$pf"
        fi
      fi
    ) &
  done
  wait
  if [ -s "$pf" ]; then
    cat "$pf" >&2
    return 1
  fi
  echo "check-recipes: ${#selftests[@]} tool self-tests ok"
}
# The box queue's exit codes 80, 81 and 83-85 name one meaning each: no other tool, recipe or script produces one of them, and
# they overlap none of the codes tools/ and the justfile already name (tools/boxq-check.py codes-check).
blk_boxqcodes() {
  if ! bc=$(python3 "$(dirname "$0")/boxq-check.py" codes-check 2>&1); then
    echo "$bc" >&2
    echo "check-recipes: a tool or recipe produces one of the box queue's exit codes 80, 81, 83-85" >&2
    return 1
  fi
  echo "${bc##*$'\n'}"
}
# The fixture variants' table (tools/fixture-variants.tsv) against the tree, and the rules ref-paths.sh applies to a variant, on a
# temporary fixture root: six columns a row, unique tags of lower-case letters and digits, a family that has a fixture directory in
# ref-paths.sh's table, recipes that are recipes of the justfile; then the paths (a variant's root, its set prefix, the directory a
# writer is given) and each refusal by name and exit code: a real tier, a tag the table does not name, one of another family, one
# that is no tag, a family with no fixture file, a variant file that is not there.
blk_fixturevariants() {
  local jf=$JF tsv rp d n=0 out vars
  tsv=$(dirname "$0")/fixture-variants.tsv rp=$(dirname "$0")/ref/ref-paths.sh
  d=$(mktemp -d)
  # shellcheck disable=SC2064
  trap "rm -rf '$d'" RETURN
  local tags="" line tag family recipes r cols
  while IFS= read -r line; do
    case $line in '' | '#'*) continue ;; esac
    n=$((n + 1))
    cols=$(awk -F'\t' '{ print NF }' <<< "$line")
    [ "$cols" = 6 ] || { echo "check-recipes: $tsv row $n has $cols columns, want 6 (tag family scope map recipes why)" >&2; return 1; }
    IFS=$'\t' read -r tag family _ _ recipes _ <<< "$line"
    case $tag in '' | *[!a-z0-9]*) echo "check-recipes: $tsv: '$tag' is not lower-case letters and digits" >&2; return 1 ;; esac
    case " $tags " in *" $tag "*) echo "check-recipes: $tsv names the tag $tag twice" >&2; return 1 ;; esac
    tags="$tags $tag"
    grep -E "__fixture_dir=[a-z0-9]+ ;;" "$rp" | grep -vE "=(self|none) ;;" | grep -qE "(^|[ (|])$family([ )|]|\$)" ||
      { echo "check-recipes: $tsv: family $family has no fixture directory row in $rp" >&2; return 1; }
    for r in $recipes; do
      grep -qE "^$r:" "$jf" || { echo "check-recipes: $tsv: $r is no recipe of the justfile" >&2; return 1; }
    done
  done < "$tsv"
  [ "$n" -gt 0 ] || { echo "check-recipes: $tsv holds no variant" >&2; return 1; }
  grep -qE '^fixture-requant FAMILY TAG:' "$jf" || { echo "check-recipes: the justfile has no fixture-requant recipe" >&2; return 1; }
  # The tier's paths on a temporary root: $d/fixtures/<dir>/ the fixture, $d/fixture-variants/<tag>/<dir>/ its variant.
  tag=$(awk -F'\t' '$0 !~ /^#/ && NF { print $1; exit }' "$tsv") family=$(awk -F'\t' '$0 !~ /^#/ && NF { print $2; exit }' "$tsv")
  mkdir -p "$d/fixtures/glm5next" "$d/fixture-variants/$tag/glm5next"
  touch "$d/fixtures/glm5next/f-00001-of-00001.gguf" "$d/fixture-variants/$tag/glm5next/v-00001-of-00001.gguf"
  rps() { # rps NAME=value... : ref-paths.sh under those variables; stdout, then the exit code on the last line
    env -i PATH="$PATH" BLOOMERY_FIXTURE_ROOT="$d/fixtures" "$@" bash -c '. "$1" && printf "%s|%s|%s|%s" "$MODEL" "$FIXTURE_FILE" "$FIXTURE_DIR" "${SET_PREFIX-}"' _ "$rp" 2>&1
    echo "|rc=$?"
  }
  want() { # want WHAT EXPECTED_SUBSTRING NAME=value... : the output holds the substring
    local what=$1 sub=$2
    shift 2
    out=$(rps "$@")
    case $out in *"$sub"*) ;; *) echo "check-recipes: ref-paths.sh, $what: wanted '$sub' in: $out" >&2; return 1 ;; esac
  }
  want "base fixture" "|$d/fixtures/glm5next/f-00001-of-00001.gguf|$d/fixtures/glm5next|" BLOOMERY_MODEL="$family" BLOOMERY_TIER=fixture || return 1
  want "variant fixture" "|$d/fixture-variants/$tag/glm5next/v-00001-of-00001.gguf|$d/fixture-variants/$tag/glm5next|" BLOOMERY_MODEL="$family" BLOOMERY_TIER=fixture BLOOMERY_FIXTURE_VARIANT="$tag" || return 1
  want "a variant's set prefix" "|rc=0" BLOOMERY_MODEL="$family" BLOOMERY_TIER=fixture BLOOMERY_FIXTURE_VARIANT="$tag" || return 1
  out=$(env -i PATH="$PATH" BLOOMERY_FIXTURE_ROOT="$d/fixtures" BLOOMERY_MODEL="$family" BLOOMERY_TIER=fixture BLOOMERY_FIXTURE_VARIANT="$tag" bash -c '. "$1" && fixture_dump_tier t && printf %s "$SET_PREFIX"' _ "$rp" 2>&1)
  [ "$out" = "fx_${tag}_" ] || { echo "check-recipes: ref-paths.sh: a variant's set prefix is '$out', want fx_${tag}_" >&2; return 1; }
  want "the directory a writer is given" "|$d/fixture-variants/$tag/glm5next|" FIXTURE_NEW=1 BLOOMERY_MODEL="$family" BLOOMERY_TIER=fixture BLOOMERY_FIXTURE_VARIANT="$tag" || return 1
  want "the real tier" "the tier is real" BLOOMERY_MODEL="$family" BLOOMERY_FIXTURE_VARIANT="$tag" || return 1
  want "the real tier exit" "rc=64" BLOOMERY_MODEL="$family" BLOOMERY_FIXTURE_VARIANT="$tag" || return 1
  want "a tag the table does not name" "does not name (it holds:" BLOOMERY_MODEL="$family" BLOOMERY_TIER=fixture BLOOMERY_FIXTURE_VARIANT=nosuchtag || return 1
  want "a tag the table does not name, exit" "rc=64" BLOOMERY_MODEL="$family" BLOOMERY_TIER=fixture BLOOMERY_FIXTURE_VARIANT=nosuchtag || return 1
  want "a variant of another family" "is a variant of $family" BLOOMERY_MODEL=qwen4exp BLOOMERY_TIER=fixture BLOOMERY_FIXTURE_VARIANT="$tag" || return 1
  want "a string that is no tag" "lower-case letters and digits" BLOOMERY_MODEL="$family" BLOOMERY_TIER=fixture BLOOMERY_FIXTURE_VARIANT='../x' || return 1
  want "a family with no fixture file" "has no fixture file to be a variant of" BLOOMERY_MODEL=qwen3moe BLOOMERY_TIER=fixture BLOOMERY_FIXTURE_VARIANT="$tag" || return 1
  rm -f "$d/fixture-variants/$tag/glm5next/"*.gguf
  want "a variant file that is not there" "fixture-requant $family $tag" BLOOMERY_MODEL="$family" BLOOMERY_TIER=fixture BLOOMERY_FIXTURE_VARIANT="$tag" || return 1
  want "a variant file that is not there, exit" "rc=66" BLOOMERY_MODEL="$family" BLOOMERY_TIER=fixture BLOOMERY_FIXTURE_VARIANT="$tag" || return 1
  # tools/box.sh reads the variant like the tier, before it syncs anything: its refusals need no box.
  for c in "BLOOMERY_FIXTURE_VARIANT=$tag|the tier is real" "BLOOMERY_TIER=fixture BLOOMERY_FIXTURE_VARIANT=a/b|lower-case letters and digits" "BLOOMERY_TIER=fixture BLOOMERY_FIXTURE_VARIANT=ab BLOOMERY_BOX_ENV=BLOOMERY_FIXTURE_VARIANT=cd|name one"; do
    read -r -a vars <<< "${c%%|*}"
    out=$(env -i PATH="$PATH" HOME="$HOME" "${vars[@]}" BLOOMERY_BOX_READONLY=1 bash "$(dirname "$0")/box.sh" true 2>&1 || true)
    case $out in *"${c#*|}"*) ;; *) echo "check-recipes: box.sh with '${c%%|*}': wanted '${c#*|}' in: $out" >&2; return 1 ;; esac
  done
  # deepseek41, fixture tier, on this same temporary root. Sourcing resolves the fixture and does not exit when
  # v41/draft holds zero files or several: MODEL stays the fixture file, DSPARK_MODEL is empty (never the profile's
  # real draft), and the reason is kept. dspark_model is what refuses, exit 66, naming the draft directory.
  mkdir -p "$d/fixtures/v41"
  local shard=$d/fixtures/v41/t-00001-of-00001.gguf ds_rc ds_out err
  touch "$shard"
  ds() { # ds SCRIPT NAME=value... : `. ref-paths.sh && SCRIPT` for this fixture; ds_out, ds_rc, $d/ds.err
    local script=$1
    shift
    ds_rc=0
    ds_out=$(env -i PATH="$PATH" BLOOMERY_FIXTURE_ROOT="$d/fixtures" BLOOMERY_MODEL=deepseek41 BLOOMERY_TIER=fixture \
      name=DSPARK_MODEL "$@" bash -c "$script" _ "$rp" 2>"$d/ds.err") || ds_rc=$?
  }
  ds '. "$1" && printf "model=%s\nds=[%s]\n" "$MODEL" "${!name}"'
  err=$(cat "$d/ds.err")
  [ "$ds_rc" = 0 ] || { echo "check-recipes: ref-paths.sh, no draft: source exited $ds_rc, want 0: $err" >&2; return 1; }
  [ "$ds_out" = "model=$shard"$'\n'"ds=[]" ] || { echo "check-recipes: ref-paths.sh, no draft: wanted model=$shard and an empty draft, got: $ds_out" >&2; return 1; }
  case $err in *"(exit 66)"*) echo "check-recipes: ref-paths.sh, no draft: source printed a refusal: $err" >&2; return 1 ;; esac
  ds '. "$1" && dspark_model'
  err=$(cat "$d/ds.err")
  [ "$ds_rc" = 66 ] || { echo "check-recipes: dspark_model, no draft: exited $ds_rc, want 66: $err" >&2; return 1; }
  [ -z "$ds_out" ] || { echo "check-recipes: dspark_model, no draft: printed '$ds_out', want nothing" >&2; return 1; }
  case $err in *"$d/fixtures/v41/draft"*) ;; *) echo "check-recipes: dspark_model, no draft: wanted the draft directory in: $err" >&2; return 1 ;; esac
  case $err in *"fixture generate"*) ;; *) echo "check-recipes: dspark_model, no draft: wanted 'fixture generate' in: $err" >&2; return 1 ;; esac
  case $err in *"(exit 66)"*) ;; *) echo "check-recipes: dspark_model, no draft: wanted '(exit 66)' in: $err" >&2; return 1 ;; esac
  mv "$shard" "$shard.bak"
  ds '. "$1" && printf x' FIXTURE_NEW=1
  err=$(cat "$d/ds.err")
  [ "$ds_rc" = 0 ] || { echo "check-recipes: ref-paths.sh, FIXTURE_NEW with no shard: source exited $ds_rc, want 0: $err" >&2; return 1; }
  ds '. "$1" && printf "dir=%s\nds=[%s]" "$FIXTURE_DIR" "${!name}"' FIXTURE_NEW=1
  [ "$ds_out" = "dir=$d/fixtures/v41"$'\n'"ds=[]" ] || { echo "check-recipes: ref-paths.sh, FIXTURE_NEW: wanted an empty draft under $d/fixtures/v41, got: $ds_out" >&2; return 1; }
  ds '. "$1" && printf x'
  err=$(cat "$d/ds.err")
  [ "$ds_rc" = 66 ] || { echo "check-recipes: ref-paths.sh, no shard: source exited $ds_rc, want 66: $err" >&2; return 1; }
  case $err in *"no fixture file"*) ;; *) echo "check-recipes: ref-paths.sh, no shard: wanted 'no fixture file' in: $err" >&2; return 1 ;; esac
  mv "$shard.bak" "$shard"
  local one=$d/fixtures/v41/draft/one.gguf two=$d/fixtures/v41/draft/two.gguf
  mkdir -p "$d/fixtures/v41/draft"
  touch "$one"
  ds '. "$1" && dspark_model'
  [ "$ds_rc" = 0 ] && [ "$ds_out" = "$one" ] || { echo "check-recipes: dspark_model, one draft: rc=$ds_rc out='$ds_out', want $one" >&2; return 1; }
  touch "$two"
  ds '. "$1" && dspark_model'
  err=$(cat "$d/ds.err")
  [ "$ds_rc" = 66 ] || { echo "check-recipes: dspark_model, two drafts: exited $ds_rc, want 66: $err" >&2; return 1; }
  [ -z "$ds_out" ] || { echo "check-recipes: dspark_model, two drafts: printed '$ds_out', want nothing" >&2; return 1; }
  case $err in *"$one"*) ;; *) echo "check-recipes: dspark_model, two drafts: wanted $one in: $err" >&2; return 1 ;; esac
  case $err in *"$two"*) ;; *) echo "check-recipes: dspark_model, two drafts: wanted $two in: $err" >&2; return 1 ;; esac
  ds '. "$1" && dspark_model' BLOOMERY_DSPARK_MODEL=/x.gguf
  [ "$ds_rc" = 0 ] && [ "$ds_out" = /x.gguf ] || { echo "check-recipes: dspark_model, caller draft: rc=$ds_rc out='$ds_out', want /x.gguf" >&2; return 1; }
  local prof line real f_rc f_out
  prof=$(dirname "$0")/ref/models/deepseek41.sh
  line=$(sed -n '215p' "$prof")
  real=$(printf '%s\n' "$line" | sed -n 's/^DSPARK_MODEL=${BLOOMERY_DSPARK_MODEL:-\(.*\)}$/\1/p')
  [ -n "$real" ] || { echo "check-recipes: $prof:215 is not the DSPARK_MODEL default: $line" >&2; return 1; }
  f_rc=0
  f_out=$(env -i PATH="$PATH" BLOOMERY_MODEL=deepseek41 bash -c '. "$1" && dspark_model' _ "$rp" 2>"$d/f.err") || f_rc=$?
  [ "$f_rc" = 0 ] && [ "$f_out" = "$real" ] || { echo "check-recipes: dspark_model, real tier: rc=$f_rc out='$f_out', want '$real' ($(cat "$d/f.err"))" >&2; return 1; }
  echo "check-recipes: deepseek41 fixture draft: source stays up, dspark_model refuses"
  echo "check-recipes: fixture variants ok ($n row(s): $tags )"
}
# A fixture-tier command exports an empty draft path when it reads DSPARK_MODEL out of ref-paths.sh
# other than through dspark_model: zero drafts leave the variable empty and the export reaches a file
# open. dspark_model refuses that with exit 66. A justfile line or a tools shell script that prints the
# variable, or expands it after sourcing ref-paths.sh without calling dspark_model, fails here by file:line.
blk_dsparkread() {
  local out
  if ! out=$(python3 - "$JF" "$(dirname "$0")" << 'PY'
import os
import re
import sys

# The bad printf is built from parts so this file does not contain that byte string (the tree grep
# for it must stay empty, including on this checker).
PRINTF = re.compile(re.escape("printf %s \"" + "$" + "DSPARK_MODEL\""))
EXPAND = re.compile(r"(?<![A-Za-z0-9_])\$\{?DSPARK_MODEL\b")
SOURCE = re.compile(r"(?:\.|source)\s+\S*ref-paths\.sh")
CALL = re.compile(r"(?<![A-Za-z0-9_])dspark_model(?![A-Za-z0-9_])")


def strip_comment(line):
    out, q, i = [], None, 0
    while i < len(line):
        c = line[i]
        if q == "'":
            out.append(c)
            if c == "'":
                q = None
            i += 1
            continue
        if q == '"':
            out.append(c)
            if c == "\\" and i + 1 < len(line):
                out.append(line[i + 1])
                i += 2
                continue
            if c == '"':
                q = None
            i += 1
            continue
        if c in "'\"":
            q = c
            out.append(c)
            i += 1
            continue
        if c == "#":
            break
        out.append(c)
        i += 1
    return "".join(out)


def code_lines(text):
    for n, raw in enumerate(text.splitlines(), 1):
        code = strip_comment(raw)
        if code.strip() == "":
            continue
        yield n, code


def justfile_hits(text, path):
    hits = []
    for n, code in code_lines(text):
        if PRINTF.search(code):
            hits.append(f"{path}:{n}: reads the draft with printf instead of dspark_model")
        elif EXPAND.search(code) and not CALL.search(code):
            hits.append(f"{path}:{n}: reads the draft out of ref-paths.sh other than through dspark_model")
    return hits


def script_hits(text, path):
    lines = list(code_lines(text))
    sources = any(SOURCE.search(code) for _, code in lines)
    calls = any(CALL.search(code) for _, code in lines)
    hits = []
    for n, code in lines:
        if PRINTF.search(code):
            hits.append(f"{path}:{n}: reads the draft with printf instead of dspark_model")
            continue
        if not EXPAND.search(code):
            continue
        if (SOURCE.search(code) and not CALL.search(code)) or (sources and not calls):
            hits.append(f"{path}:{n}: reads the draft out of ref-paths.sh other than through dspark_model")
    return hits


def expect(name, hits, want):
    if len(hits) != want:
        return [f"check-recipes: draft-read fixture '{name}': {len(hits)} hits, want {want}: {hits}"]
    return []


def fixtures():
    bad_printf = "printf %s \"" + "$" + "DSPARK_MODEL\""
    expand = "\"$" + "DSPARK_MODEL\""
    brace = "\"${" + "DSPARK_MODEL:-}\""
    bad = []
    bad += expect("a justfile line that prints the variable", justfile_hits("recipe:\n    " + bad_printf + "\n", "fixture-justfile"), 1)
    if not bad and ":2:" not in justfile_hits("recipe:\n    " + bad_printf + "\n", "fixture-justfile")[0]:
        bad.append("check-recipes: draft-read fixture 'a justfile line that prints the variable': the hit has no file:line")
    bad += expect("a justfile line that calls dspark_model", justfile_hits("recipe:\n    __s=$(. tools/ref/ref-paths.sh && dspark_model)\n", "fixture-justfile"), 0)
    bad += expect("a script that sources ref-paths and expands the variable", script_hits("source tools/ref/ref-paths.sh\necho " + expand + "\n", "fixture.sh"), 1)
    bad += expect("a script that goes through dspark_model", script_hits("source tools/ref/ref-paths.sh\nx=$(dspark_model) || exit $?\necho " + expand + "\n", "fixture-ok.sh"), 0)
    bad += expect("printf beside a dspark_model call", script_hits("x=$(dspark_model)\n" + bad_printf + "\n", "fixture-both.sh"), 1)
    bad += expect("the function's own file", script_hits("dspark_model() {\n  printf '%s' " + expand + "\n}\n", "ref-paths.sh"), 0)
    # timing-card.sh expands the variable and does not source ref-paths.sh; this rule does not own that file.
    bad += expect("a script that does not source ref-paths", script_hits("echo " + brace + "\n", "timing-card.sh"), 0)
    return bad


def scan(justfile, tools):
    hits = []
    with open(justfile, encoding="utf-8") as fh:
        hits += justfile_hits(fh.read(), justfile)
    nsh = 0
    for root, dirs, files in os.walk(tools):
        dirs[:] = [d for d in dirs if d != ".git"]
        for name in files:
            if not name.endswith(".sh"):
                continue
            nsh += 1
            path = os.path.join(root, name)
            with open(path, encoding="utf-8") as fh:
                hits += script_hits(fh.read(), path)
    return hits, nsh


errs = fixtures()
if errs:
    print("\n".join(errs))
    sys.exit(1)
hits, nsh = scan(sys.argv[1], sys.argv[2])
if hits:
    print("check-recipes: a fixture-tier command exports an empty draft path:")
    print("\n".join(hits))
    sys.exit(1)
print(f"check-recipes: draft reads go through dspark_model ({nsh} shell files, the printf fixture is red)")
PY
  ); then
    printf '%s\n' "$out" >&2
    echo "check-recipes: a fixture-tier command exports an empty draft path" >&2
    return 1
  fi
  printf '%s\n' "$out"
}
# refset-publish.sh's assignment of the fx_ sets to a family (never by name), its named error on a set it cannot place, its
# left-out block (variant, DSpark) and its --upload refusal, on a fake tree, no box.
blk_refsetpublish() {
  if ! rpt=$(bash "$(dirname "$0")/refset-publish.sh" --self-test 2>&1); then
    echo "$rpt" >&2
    echo "check-recipes: the refset-publish self-test failed" >&2
    return 1
  fi
  echo "${rpt##*$'\n'}"
}
# Every #[test] in the workspace is run by some gate-* or lab-* recipe's cargo test call on the box: its
# target, the features its path's cfgs need, its name filter and its #[ignore] (tools/recipes.py
# orphan-tests). A test no gate runs is neither a test nor a gate: without gate-ds41-bind, gpu-gates' bind
# tests sit behind a feature no recipe enables. An input the scan cannot read fails here by file and line.
blk_orphan() {
  if ! ot=$(python3 "$(dirname "$0")/recipes.py" orphan-tests 2>&1); then
    echo "$ot" >&2
    echo "check-recipes: a test no gate-* or lab-* recipe runs, or a source the scan cannot read (tools/recipes.py orphan-tests)" >&2
    return 1
  fi
  echo "${ot##*$'\n'}"
}

# The device arch has one owner: the justfile's `arch` variable (BLOOMERY_ARCH, sm_86 unset; tools/arch.sh reads the same name for
# the tools that run on the box). A recipe line that spells an arch builds for it whatever BLOOMERY_ARCH says, so a contributor's
# `BLOOMERY_ARCH=sm_121 just gate-…` would build sm_86 and run it through the driver's JIT without a word. Held: exactly one
# `arch := env_var_or_default('BLOOMERY_ARCH', 'sm_NN')` line, no `sm_<digit>` anywhere else in the justfile, and every `--arch`
# followed by `{{arch}}`. The fixtures below are the FAIL-first: each spelling the rule forbids is red on its own copy.
blk_archowner() {
  python3 - "$JF" << 'PY'
import re
import sys

DEF = re.compile(r"^arch := env_var_or_default\('BLOOMERY_ARCH', 'sm_[0-9]+[af]?'\)$")
LITERAL = re.compile(r"sm_[0-9]")
OWNED = re.compile(r" \{\{arch\}\}(?![A-Za-z0-9_])")


def problems(text):
    out, defs = [], 0
    for n, line in enumerate(text.split("\n"), 1):
        if DEF.match(line):
            defs += 1
            continue
        if LITERAL.search(line):
            out.append(f"{n}: an arch literal outside the definition of `arch`: {line.strip()[:110]}")
        rest = line
        while (m := re.search(r"--arch", rest)) is not None:
            rest = rest[m.end():]
            if not OWNED.match(rest):
                out.append(f"{n}: `--arch` is not followed by {{{{arch}}}}: {line.strip()[:110]}")
    if defs != 1:
        out.append(f"the justfile defines `arch` {defs} times, want exactly one `arch := env_var_or_default('BLOOMERY_ARCH', 'sm_NN')` line")
    return out


good = "arch := env_var_or_default('BLOOMERY_ARCH', 'sm_86')\nx:\n    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p y'\n"
fixtures = [
    ("the owner and a recipe that takes it", good, 0),
    ("a literal arch after --arch", good.replace("--arch {{arch}}", "--arch sm_86"), 2),
    ("a literal arch that is not another arch's", good.replace("--arch {{arch}}", "--arch sm_121a"), 2),
    ("a literal arch in a comment", good + "# built for sm_86\n", 1),
    ("an arch literal in a recipe line", good + "y:\n    echo sm_90\n", 1),
    ("--arch spelt with an equals sign", good.replace("--arch {{arch}}", "--arch=sm_86"), 2),
    ("--arch with the variable spelt another way", good.replace("{{arch}}", "{{ arch }}"), 1),
    ("--arch with another variable", good.replace("--arch {{arch}}", "--arch {{archive}}"), 1),
    ("--arch with no value", good.replace("--arch {{arch}} ", "--arch"), 1),
    ("a second --arch on one line", good.replace("-- -p y", "--arch sm_90 -- -p y"), 2),
    ("no definition of the variable", good.split("\n", 1)[1], 1),
    ("two definitions", good + "arch := env_var_or_default('BLOOMERY_ARCH', 'sm_86')\n", 1),
]
bad = 0
for name, text, want in fixtures:
    got = problems(text)
    if len(got) != want:
        bad += 1
        print(f"check-recipes: arch owner fixture '{name}': {len(got)} problems, want {want}: {got}", file=sys.stderr)
if bad:
    sys.exit(1)
with open(sys.argv[1], encoding="utf-8") as fh:
    text = fh.read()
found = problems(text)
if found:
    print("check-recipes: the device arch has one owner, the justfile's `arch` variable (BLOOMERY_ARCH): a recipe line that spells an arch builds for it whatever BLOOMERY_ARCH says:", file=sys.stderr)
    for f in found:
        print(f"  justfile:{f}", file=sys.stderr)
    sys.exit(1)
uses = sum(1 for line in text.split("\n") if line.startswith(" ") and "--arch {{arch}}" in line)
print(f"check-recipes: arch owner ok ({uses} recipe lines build with --arch {{{{arch}}}}, {len(fixtures)} fixtures)")
PY
}

# The cpu-guard live pair reads a real /proc. It runs here, before the blocks, so it does not share the
# runner with them (on CI's two cores a foreign busy loop inside the 36-wide block can miss its window).
# No /proc is a printed skip. The product cpu threshold stays 50 %; only the pair sets CPU_BUSY_PCT=1.
# The wall is whole seconds, the same unit as the block times below.
proc_leg() {
  local t0=$SECONDS out nlive
  if [ ! -r /proc/self/stat ]; then
    echo "proc-leg: skipped, no /proc/self/stat"
    echo "proc-leg: wall $((SECONDS - t0)) s"
    return 0
  fi
  out=$(CPU_GUARD_PROC_LEG=1 bash "$(dirname "$0")/ref/card-tests/cpu-guard.sh" 2>&1) || {
    printf '%s\n' "$out" >&2
    echo "check-recipes: proc-leg: cpu-guard failed" >&2
    echo "proc-leg: wall $((SECONDS - t0)) s" >&2
    return 1
  }
  printf '%s\n' "$out"
  nlive=$(grep -c '^ok live:' <<< "$out" || true)
  if ! grep -Fxq 'cpu-guard: 11 tests, 0 failed' <<< "$out" \
    || [ "$nlive" != 2 ] \
    || grep -Eq 'skip live' <<< "$out"; then
    echo "check-recipes: proc-leg: want 'cpu-guard: 11 tests, 0 failed', both 'ok live:' lines, and no 'skip live'" >&2
    echo "proc-leg: wall $((SECONDS - t0)) s" >&2
    return 1
  fi
  echo "proc-leg: wall $((SECONDS - t0)) s"
}

# The holder judge already ran inside blk_gpugate. Linux: its success line, and not the /proc skip.
# A host with no /proc/self/stat: the skip line gpu-gate.sh prints (the judge name, then that echo).
proc_leg_gpu() {
  local log=$B/gpugate.full
  if [ ! -f "$log" ]; then
    echo "check-recipes: proc-leg: no gpu-gate log" >&2
    return 1
  fi
  if [ -r /proc/self/stat ]; then
    if grep -Fq 'ok   … its wait line names the first run as the holder' "$log" \
      && ! grep -Fq 'no /proc here' "$log"; then
      echo "proc-leg: gpu-gate holder named"
      return 0
    fi
    echo "check-recipes: proc-leg: the gpu-gate log must name the holder and must not skip /proc" >&2
    return 1
  fi
  if grep -Fq 'skip   … its wait line names the first run as the holder: no /proc here (lease_holders reads it)' "$log"; then
    echo "proc-leg: gpu-gate skip, no /proc"
    return 0
  fi
  echo "check-recipes: proc-leg: the gpu-gate log must print the /proc skip" >&2
  return 1
}

proc_leg
unset CPU_GUARD_PROC_LEG

BLOCKS=(recipescheck selftest smoke cardorder cardtests lease boxtracks loadgroups lcppfit coldblocks leverarms slotsarm lcppwarm q38srv
  maccheck gatebatch gpugate stackwatch ptxscan ptxspill scanargs narrowscan ldsscan mutantrun macstatic macremote boxsh seedtarget carry releasebuild pytools boxqcodes fixturevariants dsparkread
  refsetpublish orphan archowner wt)
for b in "${BLOCKS[@]}"; do
  ( set +e; t0=$SECONDS; blk_$b > "$B/$b.out" 2>&1; brc=$?; echo $((SECONDS - t0)) > "$B/$b.t"; echo $brc > "$B/$b.rc" ) & # set +e: a red block writes its own rc
done
wait
rc=0
for b in "${BLOCKS[@]}"; do
  brc=$(cat "$B/$b.rc" 2> /dev/null) || brc=255 # a block killed before its wrapper wrote the rc
  if [ "$brc" != 0 ]; then
    cat "$B/$b.out" >&2
    [ "$rc" != 0 ] || rc=$brc
  else
    cat "$B/$b.out"
  fi
done
if ! proc_leg_gpu; then
  [ "$rc" != 0 ] || rc=1
fi
# The five slowest blocks, whole seconds each (the wall is the slowest block's own time under the load of
# its siblings); a block killed before its wrapper wrote the time counts 0.
slowest=$(for b in "${BLOCKS[@]}"; do echo "$(cat "$B/$b.t" 2> /dev/null || echo 0) $b"; done |
  sort -rn | awk 'NR <= 5 {printf "%s%s %s s", (NR > 1 ? ", " : ""), $2, $1}')
echo "check-recipes: slowest: $slowest"
[ "$rc" = 0 ] || exit "$rc"
echo "check-recipes: ok"
