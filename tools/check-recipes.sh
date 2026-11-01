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
# blk_prelude: the justfile's line rules, grep only.
blk_prelude() {
  local bad raw lock bare unpinned
  bad=$(grep -nE '(cargo (oxide )?test|tools/gate\.sh).*\|\|' "$JF" || true)
  if [ -n "$bad" ]; then
    echo "check-recipes: a test-running recipe line carries '||' — it swallows the gate's exit code:" >&2
    echo "$bad" >&2
    return 1
  fi
  raw=$(grep -nE "box\.sh '.*cargo (oxide )?test" "$JF" || true)
  if [ -n "$raw" ]; then
    echo "check-recipes: bare 'cargo test' in a box recipe — route it through tools/gate.sh (bound + exit code):" >&2
    echo "$raw" >&2
    return 1
  fi
  # GPU 게이트 락의 소유자는 tools/gpu-gate.sh 하나다. 레시피가 락을 직접 잡으면 그 줄의 바이너리는 상한 없이 돈다
  # — 매달린 GPU 게이트 하나가 락을 쥐면 그 락을 기다리는 트랙이 전부 선다.
  lock=$(grep -nE '^[^#]*bloomery-gate\.lock' "$JF" || true)
  if [ -n "$lock" ]; then
    echo "check-recipes: a recipe takes the GPU gate lock itself — run the binary through tools/gpu-gate.sh (lock + bound + exit code):" >&2
    echo "$lock" >&2
    return 1
  fi
  # A host binary's bound and exit code belong to tools/host-gate.sh. A recipe that runs a
  # target/release binary behind its own `timeout` spells the bound and the timeout's exit code again,
  # and prints nothing when the bound is hit.
  bare=$(grep -nE '^[^#]*timeout[^#]*target/release/' "$JF" || true)
  if [ -n "$bare" ]; then
    echo "check-recipes: a recipe runs a target/release binary behind its own timeout — run it through tools/host-gate.sh (bound + exit code):" >&2
    echo "$bare" >&2
    return 1
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
    return 1
  fi
}
# Every recipe's cargo targets, features and runner binaries against the workspace (cargo metadata, no
# build): a `--bin`, `--test` or `-p` that names nothing, a feature the package lacks, a gpu-gate.sh or
# host-gate.sh name the recipe does not build, a named script that is not in the tree, a gate-* recipe
# with no cargo target, a build of a shipped bin (tools/release/build.sh's BINS) with a feature list
# but that file's. The parser is tools/recipes.py, the one tools/affected-gates.sh uses; its own
# tests (--self-test) run here too, so the parser the lead's batch lists come from is tested wherever
# this check runs.
# The check of the recipes against the workspace is the block blk_recipescheck. The blocks below are independent —
# each runs on fixtures in its own temp directory, as its own process (`--block NAME`) — so tools/static-sched.py runs
# them side by side, at most a cpu count at once, longest first, and they report in this file's order (`--report`):
# each block's own output (its last line where a serial form printed only that), its full output and its named error
# on failure, and the first red block's own rc ends the check. tools/mac-static.sh runs the same jobs in its own
# schedule, each keyed by the files it reads (blk_keyspec), so a block whose input key is green is skipped and named.
B=$(mktemp -d)
trap 'rm -rf "$B"' EXIT

# blk_NAME: one block. Its stdout is what the serial form printed; a failure prints its named error to
# stderr and returns nonzero. `$(dirname "$0")` still names this script's directory inside a function.
blk_recipescheck() {
  python3 "$(dirname "$0")/recipes.py" check
}
blk_gateclass() {
  python3 "$(dirname "$0")/gate_class.py" check
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
# tool_selftest SCRIPT [ERE]: bash tools/SCRIPT --self-test; its last line, or its full output and a
# named error. ERE, when named, is printed before that last line: the last line of the output that
# matches. A script that printed none leaves the pipeline empty: grep's miss is not a failure.
tool_selftest() {
  local script=$1 pat=${2:-} out
  if ! out=$(bash "$(dirname "$0")/$script" --self-test "${@:3}" 2>&1); then
    printf '%s\n' "$out" >&2
    echo "check-recipes: the ${script%.sh} self-test failed" >&2
    return 1
  fi
  if [ -n "$pat" ]; then
    printf '%s\n' "$out" | grep -E -- "$pat" | tail -n 1 || true
  fi
  printf '%s\n' "${out##*$'\n'}"
}
# lease.sh's own self-test: the timing-card check and witness on a stub nvidia-smi, and the trace,
# bounded arms and perf cancellation on temp leases and held stubs; no card or machine lease.
blk_lease() { tool_selftest ref/lease.sh; }
# box-tracks.sh deletes remote track directories: its selection (only the names given, never a new stale
# the reader did not see, a bare --remove refused) is tested on fixed input, no ssh.
blk_boxtracks() { tool_selftest box-tracks.sh; }
# blk_scancheck CHECKER: tool_selftest of tools/check-CHECKER.sh. The path is composed: a literal name would put the
# checker in the scripts recipes.py reads check-recipes as running, and through check-levers' fixture strings a script
# that reads a reference tree, so check-recipes would never skip.
blk_scancheck() { tool_selftest "check-$1.sh"; }
# check-unsafe.sh counts the crates' unsafe lines against their pins: a crate directory or source file it cannot list
# or read ends the check with 70. That refusal is tested here on missing paths and on a crate in a temporary directory
# (a build.rs that is a link, and a broken one), no tree walk.
blk_unsafecount() { blk_scancheck unsafe; }
# check-arch.sh scans the tree for the architecture split's five rules: a scan whose grep or find cannot read ends the
# check with no verdict. The architecture listing and each scan are tested here on a missing path, and rule 2's literal
# scan on fixed lines.
blk_archscan() { blk_scancheck arch; }
# check-loads.sh scans crates/*/src for a generated module load outside the shared slot: a source file it cannot list
# or read ends the check with 70. The family and load scans are tested here on fixed input and a missing file.
blk_loadscan() { blk_scancheck loads; }
# check-comments.sh lists crates/*/src and crates/*/tests and scans the files git sees for a retired word: a directory
# or file it cannot list or read ends the check with no verdict. The listing, the comment reader and the scan are
# tested here on fixed input and on input that cannot be read, no tree walk.
blk_commentscan() { blk_scancheck comments; }
# check-levers.sh walks crates/*/src, crates/*/tests, tools/ and .cargo/ for reads and names: a directory the walk
# cannot list or a file it cannot read ends the check with exit 2 and names it. That refusal is tested here on a
# temporary tree, no tree walk.
blk_leverscan() { blk_scancheck levers; }
# load-groups.sh decides which arms of a depth runner share one load, and their order: its grouping and
# rotation are tested on fixed keys, and under bash 4.1 or later its idle bound on a stalled process.
blk_loadgroups() { tool_selftest ref/load-groups.sh; }
# lcpp-fit.sh builds the depth runners' llama.cpp fit arms and reads what the fit chose: its flag
# rewrite, its --help probe (stub binaries) and its column (fixture output) are tested here, no box.
blk_lcppfit() { tool_selftest ref/lcpp-fit.sh; }
# cold-blocks.sh is the depth runners' cold tag and engine-block planner: its bound, its flag rewrite, its
# order refusal and its block plan on fixed arms are tested here, no box.
blk_coldblocks() { tool_selftest ref/cold-blocks.sh; }
# lever-arms.sh is the depth runners' lever-arm list check and their pre-lease `--levers` probe (lever_probe): its
# words and its args after `--`, no box.
blk_leverarms() { tool_selftest ref/lever-arms.sh; }
# slots-arm.sh is the depth runners' aggregate arm (<arm>@BLOOMERY_GEN_SLOTS=N): its N, its plain twin, the
# environment's refusal, its time pass and residency clauses and the plain tables' labels, no box.
blk_slotsarm() { tool_selftest ref/slots-arm.sh; }
# bin-label.sh is the depth runners' bin: row label (the tree's basename) and the refusal of two
# binaries that would share it: the stem and the refusal on fixed paths, no box.
blk_binlabel() { tool_selftest ref/bin-label.sh; }
# lcpp-warm.sh is the depth runners' llama-server arms: its flag translation (against V4.1's LCPP_CLI_FLAGS),
# its refusals, the context, the --help probe and a stub server's start, requests, checks and stop (python3
# and curl, 127.0.0.1) are tested here, no box.
blk_lcppwarm() { tool_selftest ref/lcpp-warm.sh; }
# q38-3090-server.sh is the lone-3090 Qwen3.8 server run (the discovery search's M4): its refusals (a set lever, a card that
# is not the one 3090, a relative or existing out dir), the chats' records and the stop by the server's own pid, against a stub
# gate runner and server (python3, curl, 127.0.0.1), are tested here, no box.
blk_q38srv() { tool_selftest ref/q38-3090-server.sh; }
# mac-check.sh runs the check and lint recipes' box commands on the Mac: its derivation from those
# recipes, its refusals, the ratchet and the prerequisite checks (a fake HOME) are tested here, no cargo.
blk_maccheck() { tool_selftest mac-check.sh; }
# gate-batch.sh's placement rules (the v41-load lane, a batch or a solo recipe's arm as an item, a cold
# build's times row) on a fixture justfile and fixture times rows, no box. The pattern is the
# slowest-cases line: tool_selftest prints its last match, then the count, which stays the block's last line.
blk_gatebatch() { tool_selftest gate-batch.sh '^[a-z-]+ self-test: slowest cases: ' --part "$1"; }
# gpu-gate.sh's card choice, its lease refusals and the V4.1 load lock, against lock files of its own and
# a stub nvidia-smi (flock and timeout stand-ins where the host has none, as on the Mac), no card.
blk_gpugate() {
  # The full transcript is the proc-leg's input (proc_leg_gpu); the self-test's own last line comes first.
  local ggt rc=0
  ggt=$(bash "$(dirname "$0")/gpu-gate.sh" --self-test 2>&1) || rc=$?
  printf '%s\n' "$ggt" > "$B/gpugate.full"
  if [ "$rc" != 0 ]; then
    printf '%s\n' "$ggt" >&2
    echo "check-recipes: the gpu-gate self-test failed" >&2
    return 1
  fi
  echo "${ggt##*$'\n'}"
  proc_leg_gpu
}
# stack-watch.sh (gpu-gate.sh's BLOOMERY_GATE_STACKS): output passed through, a quiet stub dumped and ended by
# its comm among the spawned command's descendants only, a quiet command with no such process left alone, a
# TERM during a dump or the last wait ending the dump and the command before the watch returns, no dump after a
# TERM, a TERM right after the command has started ending it.
blk_stackwatch() { tool_selftest ref/stack-watch.sh; }
# ptx-scan.sh's global load widths, legacy columns, entry boundaries and scan readers on synthetic PTX, no box.
blk_ptxscan() { tool_selftest ptx-scan.sh; }
# ptx-spill-check.sh's table reading (a process-substitution table read for every binary, a binary with no
# pinned row) and its verdict lines, through a stub ptx-scan.sh on fixed scans, no build.
blk_ptxspill() { tool_selftest ptx-spill-check.sh; }
# scan-args.sh is the three scan recipes' refusal of a cargo feature given where the scan's own words go
# (ptx-scan, sass-scan, lds-scan): its cases on a fixed feature list and the real crates/gpu-gates table.
blk_scanargs() { tool_selftest scan-args.sh; }
# narrow-scan.sh is `just narrow`'s box orchestration: the runner it ships to each side (one build per feature list, the
# scanner embedded verbatim), the cut of the box's output into per-bin scan logs, the one-run-at-a-time lock on the
# persistent remote dirs, its usage refusals, and the drift check against the ptx-scan recipe's build lines. No box.
blk_narrowscan() { tool_selftest narrow-scan.sh; }
# canon-check.sh is `just canon-check`'s proof that a kernel move is bit-identical: its per-entry verdicts, counts, exit codes and
# refusals over fixture scan logs and PTX modules through the real ptx-canon.py, and its call of narrow-scan.sh, on a stub, no box.
blk_canoncheck() { tool_selftest canon-check.sh; }
# ptx-compare.sh is `just ptx-compare`'s add and move proof: its two box calls (the base from a git archive in a remote dir of its own, the
# head from this tree), the base side's cache, the table's exit codes (0, 3 changed, 4 removed, 5 both, 2 refused) through the real
# recipes.py reader, its usage refusals and the build lines' drift against the ptx-scan recipe, on a stub box.sh in a temp git repo, no box.
blk_ptxcompare() { tool_selftest ptx-compare.sh; }
# lds-scan.sh's rows (its PTX and SASS counts over a fixture binary, stub extractor, ptxas and cuobjdump),
# its filter, its failed-scan banner and its usage refusal, no box.
blk_ldsscan() { tool_selftest lds-scan.sh; }
# mutant-run.sh's kill, survive, not-built and no-Compiling verdicts, its restore checks (a broken copy, an
# edit during the run, a TERM) and its refusals, in a temp git repo with a fake gate, no box.
blk_mutantrun() { tool_selftest mutant-run.sh; }
# mac-static.sh runs the round loop's jobs (fmt-check, lint, the scoped combos, the check scripts and this file's blocks)
# under tools/static-sched.py: its job list, its scheduling, its rc/wall capture, its skip lines and its red-step naming are
# tested against stub jobs, no cargo, no check script.
blk_macstatic() { tool_selftest mac-static.sh; }
# mac-remote.sh sends a tree to the second Mac and runs mac-static there: its argument, config-file, track-name,
# dirty-tree, lock and ledger rules, and (on macOS) the whole flow against a stub ssh that runs the "remote" in a
# fake home, are tested here, no remote and no cargo.
blk_macremote() { tool_selftest mac-remote.sh; }
# box.sh's hold on this side (~/.cache/bloomery/box-hold/<owner>, the header): its refusal before any rsync or ssh, the owner's and a
# read's pass, the `--hold up|down` rules and a looping caller stopping at its next call, on a scratch copy under a fake HOME with
# ssh and rsync stubs, no box.
blk_boxsh() { tool_selftest box.sh; }
# oxide-host.sh writes the generated .cargo/cuda-oxide.toml for a triple and refuses an llc that is unset, missing, of another
# major or without a version banner, each by name: its cases run on scratch trees and stub llc scripts, no build and no card.
blk_oxidehost() { tool_selftest oxide-host.sh; }
# oxide-backend.sh builds the patched backend and refuses, by name, one without this tree's patch, with another rev or missing: its cases
# run on a scratch tree with stub cargo and rustc, no build and no card.
blk_oxidebackend() { tool_selftest oxide-backend.sh; }
# seed-target.sh clones a warm tree's target/ into a new worktree and restores the seed's mtime on the files that are equal:
# its refusals (a DST with a target, a SRC without one, two volumes, a host that cannot clone) and the mtime pass (only
# equal files) are tested on two scratch trees, no cargo.
blk_seedtarget() { tool_selftest seed-target.sh; }
# wt.sh adds a worktree and seeds it from the nearest warm tree, and retires one through mac-disk: the refusals of `new` (an
# existing path, an existing branch, an add that fails or creates nothing, each with the repo unchanged and the next step
# not run, and a mutant of each guard that does run it), the seed choice, the read-only seed, and the in-flight rule of
# `retire`, on scratch git repos with stub seed-target and mac-disk, no cargo, no box.
blk_wt() { tool_selftest wt.sh; }
# mac-gate.sh runs the Metal gate on a Mac: the refusals (another host, a box card name, a macOS below the floor, no device), the
# bound and the lock, on a stub gate binary, on any host.
blk_macgate() { tool_selftest mac-gate.sh; }
# check-bounded.sh: the scan reads the tree, its self-test reads fixtures (path composed, see blk_scancheck).
blk_boundedscan() { local c=bounded; bash "$(dirname "$0")/check-$c.sh" > /dev/null && blk_scancheck "$c"; }
# gate-bound.sh's own self-test: the owner's cancellation table on stub timeouts (no card, no lock).
blk_gatebound() { tool_selftest gate-bound.sh; }
# The reference-set manifest writers, dump-dequant.sh and the tokenizer's oracle.sh: a failed md5 or size read ends the writer
# before its line, on stand-in files with md5sum and wc stand-ins (tools/ref/read-stubs.sh), no box.
blk_dumpdequant() { tool_selftest ref/dump-dequant.sh; }
blk_tokoracle() { tool_selftest ../crates/tokenizer/tools/oracle.sh; }
# ik-pin-tree.sh compares a pinned ik tree's cmake cache with the shared tree's before it builds: a cache it cannot read or that holds no
# entries is exit 2 and no build, a value with an `=` inside is one value, and a grep that fails is no verdict. Tested on fixture caches
# and a stub grep, no box.
blk_ikpintree() { tool_selftest ref/ik-pin-tree.sh; }
# build-qdot-ref.sh builds only the dumpers it is named: a NAME that is not one dumper, two names in one argument included, is 64 and builds
# nothing. Tested on a stub g++, no ik, no box.
blk_qdotrefargs() { tool_selftest ref/build-qdot-ref.sh; }
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
blk_releasebuild() { tool_selftest release/build.sh; }
# install.sh's archive choice (x86-64, GB10 aarch64 and Apple Silicon, and each refusal) on stub uname and nvidia-smi: the
# installer runs on users' machines only, so its choice is held here.
blk_releaseinstall() { tool_selftest release/install.sh; }
# publish.sh's first step: tools/rotation.py check refuses the publish on rc 1 (and on rc 2 or 70, named) before the tag
# check, on a stub rotation.py in a temporary repository; and its sitting step (3a): a portable verdict of the tarball's own
# binary is silent, any other is a warning that names it, and the status stays 0; the script runs on the lead's Mac only at a
# release.
blk_releasepublish() { tool_selftest release/publish.sh; }
# sitting.sh is the release sitting's runner: the cells file's faults, round 1 family-major and round 2 reversed with its arms
# rotated, the red stop and --keep-going, the hold on every exit path, the verdict's fields, the environment a cell sets and the
# --dry-run wall, on a stub depth-serve.sh and stub checks, then a mutant of each guard that its case must turn red; no box, no
# card. The release-sitting recipe must call it: a recipe that stops calling it runs no sitting, so the call is held here.
blk_releasesitting() {
  local out
  if ! out=$(bash "$(dirname "$0")/release/sitting.sh" --self-test 2>&1); then
    printf '%s\n' "$out" >&2
    echo "check-recipes: the release sitting self-test failed" >&2
    return 1
  fi
  if ! awk '/^release-sitting /{f=1; next} f && /^[^ ]/{exit} f' "$JF" | grep -q 'tools/release/sitting\.sh'; then
    echo "check-recipes: the release-sitting recipe must run tools/release/sitting.sh" >&2
    return 1
  fi
  printf '%s\n' "${out##*$'\n'}"
}
# Every Python tool's own tests, on the Mac and the Linux host nightly (seconds in all): a self-test that no check runs rots. A tool
# that grows one is listed here, and the comparison (unit pytools.list) fails on one that is not. Each tool is a unit of its own
# (pytools.K, the K-th line, run from the repository root), so the scheduler bounds them with every other unit.
pytools_list() {
  selftests=(
    "tools/bloomery/catalog.py --self-test"
    "tools/bloomery/manifest.py --self-test"
    "tools/bloomery/records.py --self-test"
    "tools/bloomery/route_trace.py --self-test"
    "tools/bloomery/rows.py --self-test"
    "tools/bloomery/selftest.py --self-test"
    "tools/boxq-check.py --self-test"
    "tools/boxq.py --self-test"
    "tools/check-catalog-docs.py --self-test"
    "tools/check-comment-only.py --self-test"
    "tools/check-defaults.py --self-test"
    "tools/check-ptx-names.py --self-test"
    "tools/check-records.py --self-test"
    "tools/flow/ds41_prefill.py --self-test"
    "tools/flow/nodes_scale.py --self-test"
    "tools/flow/pplb.py --self-test"
    "tools/flow/q38_step.py --self-test"
    "tools/flow/q38width.py --self-test"
    "tools/flow/resplay.py --self-test"
    "tools/flow/routes.py --self-test"
    "tools/gate_class.py --self-test"
    "tools/gate-paths-merge.py --self-test"
    "tools/hf-arch-check.py --self-test"
    "tools/mac-disk.py --self-test"
    "tools/mlx-fixture.py --self-test"
    "tools/mlx-logits.py --self-test"
    "tools/model_tax.py --self-test"
    "tools/owners_check.py --self-test"
    "tools/pr-gate-lines.py --self-test"
    "tools/recipes-key-test.py --self-test"
    "tools/ref/asm-census.py --self-test"
    "tools/ref/cache-options-test.py --self-test"
    "tools/ref/check-int-twins.py --self-test"
    "tools/ref/clef/e2e.py --self-test"
    "tools/ref/clef/sjcmp.py --self-test"
    "tools/ref/clef_ref.py --self-test"
    "tools/ref/clefvis/check_hidden.py --self-test"
    "tools/ref/clefvis/check_preproc.py --self-test"
    "tools/ref/dense-logits.py --self-test"
    "tools/ref/dense-serve.py --self-test"
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
    "tools/ref/q3pp.py --self-test"
    "tools/ref/qdot-rate-floors.py --self-test"
    "tools/ref/route-trace-chat.py --self-test"
    "tools/ref/router-coverage.py --self-test"
    "tools/ref/router-residency.py --self-test"
    "tools/ref/set-diff.py --self-test"
    "tools/ref/source-inspection-test.py --self-test"
    "tools/ref/tier-cost.py --self-test"
    "tools/ref/window-union.py --self-test"
    "tools/release/pass_checks.py --self-test"
    "tools/review/carry.py --self-test"
    "tools/review/diagnostics.py --self-test"
    "tools/review/labels.py --self-test"
    "tools/review/ledger.py --self-test"
    "tools/review/local.py --self-test"
    "tools/review/post.py --self-test"
    "tools/review/run.py --self-test"
    "tools/review/signals.py --self-test"
    "tools/review/usage.py --self-test"
    "tools/rotation.py --self-test"
    "tools/static-sched.py --self-test"
    "tools/type-census.py --self-test"
    "tools/verdict-diff.py --self-test"
  )
}
# blk_pytools list|K: the list against the tools that have a self-test, or the K-th tool's self-test (a Python module this
# host lacks is one named line, never the tool's traceback: the host nightly's install puts python3-numpy and python3-pil in
# place for these, tools/nightly/install.sh).
blk_pytools() {
  local root f arg out m mod pkg listed found
  root="$(cd "$(dirname "$0")/.." && pwd)"
  pytools_list
  if [ "$1" = list ]; then
    listed=$(printf '%s\n' "${selftests[@]}" | cut -d' ' -f1 | sort)
    found=$(cd "$root" && grep -rlE -e '--self-test|"self-test"' --include='*.py' tools | grep -vx 'tools/recipes.py' | sort)
    if [ "$listed" != "$found" ]; then
      echo "check-recipes: the Python tools with a self-test and the list this check runs differ (< listed, > found):" >&2
      diff <(echo "$listed") <(echo "$found") >&2 || true
      return 1
    fi
    echo "check-recipes: ${#selftests[@]} tool self-tests, one unit each"
    return 0
  fi
  read -r f arg <<< "${selftests[$(($1 - 1))]}"
  if out=$(cd "$root" && python3 "$f" "$arg" 2>&1); then
    return 0
  fi
  if m=$(grep -m1 "ModuleNotFoundError: No module named" <<< "$out"); then
    mod=$(printf '%s' "$m" | sed -n "s/.*No module named '\([^']*\)'.*/\1/p")
    case ${mod%%.*} in numpy) pkg=python3-numpy ;; PIL) pkg=python3-pil ;; *) pkg= ;; esac
    printf 'check-recipes: %s %s needs the Python module %s this host lacks%s\n' \
      "$f" "$arg" "$mod" "${pkg:+ — the host nightly installs $pkg}" >&2
  else
    printf '%s\n%s\n' "$out" "check-recipes: $f $arg failed" >&2
  fi
  return 1
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
# fixture-dirs.tsv's table, a non-empty map and recipes column (the reader's own refusal, crates/model/src/fixture/variant.rs),
# recipes that are recipes of the justfile; then the paths (a variant's root, its set prefix, the directory a
# writer is given) and each refusal by name and exit code: a real tier, a tag the table does not name, one of another family, one
# that is no tag, a family with no fixture file, a variant file that is not there.
blk_fixturevariants() {
  local jf=$JF tsv rp dirs d out vars summary
  tsv=$(dirname "$0")/fixture-variants.tsv rp=$(dirname "$0")/ref/ref-paths.sh dirs=$(dirname "$0")/fixture-dirs.tsv
  d=$(mktemp -d)
  # shellcheck disable=SC2064
  trap "rm -rf '$d'" RETURN
  # rows TSV: the table's row shape. The reader (crates/model/src/fixture/variant.rs) refuses a row
  # whose map or recipes column is empty at run time ("the map and the recipes are not empty"); the
  # checker refuses it here first, by name. bash's read folds an empty tab-separated column into the
  # next one, so the columns come out by awk.
  rows() {
    local file=$1 n=0 tags="" line tag family map recipes r cols
    while IFS= read -r line; do
      case $line in '' | '#'*) continue ;; esac
      n=$((n + 1))
      cols=$(awk -F'\t' '{ print NF }' <<< "$line")
      [ "$cols" = 6 ] || { echo "check-recipes: $file row $n has $cols columns, want 6 (tag family scope map recipes why)" >&2; return 1; }
      tag=$(awk -F'\t' '{ print $1 }' <<< "$line") family=$(awk -F'\t' '{ print $2 }' <<< "$line")
      map=$(awk -F'\t' '{ print $4 }' <<< "$line") recipes=$(awk -F'\t' '{ print $5 }' <<< "$line")
      case $tag in '' | *[!a-z0-9]*) echo "check-recipes: $file: '$tag' is not lower-case letters and digits" >&2; return 1 ;; esac
      case " $tags " in *" $tag "*) echo "check-recipes: $file names the tag $tag twice" >&2; return 1 ;; esac
      tags="$tags $tag"
      [ -n "$map" ] || { echo "check-recipes: $file row $n: the map column is empty (the reader refuses it: \"the map and the recipes are not empty\")" >&2; return 1; }
      [ -n "$recipes" ] || { echo "check-recipes: $file row $n: the recipes column is empty (the reader refuses it: \"the map and the recipes are not empty\")" >&2; return 1; }
      awk -F'\t' -v f="$family" '$0 !~ /^#/ && $1 == f && $2 != "self" { found = 1 } END { exit !found }' "$dirs" ||
        { echo "check-recipes: $file: family $family has no fixture directory row in $dirs" >&2; return 1; }
      for r in $recipes; do
        # A recipe with parameters (`gate-gpu-ds41-step *ARGS=…:`) is a recipe too.
        grep -qE "^$r([ :]|\$)" "$jf" || { echo "check-recipes: $file: $r is no recipe of the justfile" >&2; return 1; }
      done
    done < "$file"
    [ "$n" -gt 0 ] || { echo "check-recipes: $file holds no variant" >&2; return 1; }
    printf '%s row(s):%s\n' "$n" "$tags"
  }
  summary=$(rows "$tsv") || return 1
  # The empty-column refusal, proven on a one-row table whose other columns are well-formed.
  for bad in map recipes; do
    if [ "$bad" = map ]; then
      printf 'badrec\tglm5next\tall\t\tgate-x\tself-test\n' > "$d/bad.tsv"
    else
      printf 'badrec\tglm5next\tall\tffn_gate=q4_K\t\tself-test\n' > "$d/bad.tsv"
    fi
    out=$(rows "$d/bad.tsv" 2>&1) && { echo "check-recipes: fixture-variants.tsv: an empty $bad column read as fine: $out" >&2; return 1; }
    case $out in
      *"row 1: the $bad column is empty"*) ;;
      *) echo "check-recipes: fixture-variants.tsv: wanted 'row 1: the $bad column is empty' in: $out" >&2; return 1 ;;
    esac
  done
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
  want "a family with no fixture file" "has no fixture file to be a variant of" BLOOMERY_MODEL=deepseek2 BLOOMERY_TIER=fixture BLOOMERY_FIXTURE_VARIANT="$tag" || return 1
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
  # A deepseek41 variant: its map never retypes the draft, so a variant directory with no draft answers the fixture's one, one with
  # an identical draft of its own answers that, and one whose draft differs from the fixture's, or a variant over a fixture with no
  # draft, is refused by name (dspark_model, exit 66).
  local vtag vdir
  vtag=$(awk -F'\t' '$0 !~ /^#/ && $2 == "deepseek41" { print $1; exit }' "$tsv")
  [ -n "$vtag" ] || { echo "check-recipes: $tsv names no deepseek41 variant: the draft rules of a variant have no case to run on" >&2; return 1; }
  vdir=$d/fixture-variants/$vtag/v41
  rm -f "$two"
  mkdir -p "$vdir"
  touch "$vdir/t-00001-of-00001.gguf"
  ds '. "$1" && dspark_model' BLOOMERY_FIXTURE_VARIANT="$vtag"
  [ "$ds_rc" = 0 ] && [ "$ds_out" = "$one" ] || { echo "check-recipes: dspark_model, a variant with no draft: rc=$ds_rc out='$ds_out', want the fixture's $one" >&2; return 1; }
  mkdir -p "$vdir/draft"
  cp "$one" "$vdir/draft/one.gguf"
  ds '. "$1" && dspark_model' BLOOMERY_FIXTURE_VARIANT="$vtag"
  [ "$ds_rc" = 0 ] && [ "$ds_out" = "$vdir/draft/one.gguf" ] || { echo "check-recipes: dspark_model, a variant with the fixture's draft: rc=$ds_rc out='$ds_out', want $vdir/draft/one.gguf" >&2; return 1; }
  printf x > "$vdir/draft/one.gguf"
  ds '. "$1" && dspark_model' BLOOMERY_FIXTURE_VARIANT="$vtag"
  err=$(cat "$d/ds.err")
  [ "$ds_rc" = 66 ] && [ -z "$ds_out" ] || { echo "check-recipes: dspark_model, a variant with another draft: rc=$ds_rc out='$ds_out', want 66 and nothing: $err" >&2; return 1; }
  case $err in *"the variant $vtag holds the draft $vdir/draft/one.gguf, which differs from the fixture's $one"*) ;; *) echo "check-recipes: dspark_model, a variant with another draft: wanted the variant and both drafts in: $err" >&2; return 1 ;; esac
  rm -rf "$vdir/draft" "$one"
  ds '. "$1" && dspark_model' BLOOMERY_FIXTURE_VARIANT="$vtag"
  err=$(cat "$d/ds.err")
  [ "$ds_rc" = 66 ] && [ -z "$ds_out" ] || { echo "check-recipes: dspark_model, a variant over a fixture with no draft: rc=$ds_rc out='$ds_out', want 66 and nothing: $err" >&2; return 1; }
  case $err in *"$vdir/draft"*"the fixture's own draft directory holds"*) ;; *) echo "check-recipes: dspark_model, a variant over a fixture with no draft: wanted both draft directories in: $err" >&2; return 1 ;; esac
  echo "check-recipes: deepseek41 variant draft: the fixture's, an identical own, a differing own refused, none refused"
  local prof line real f_rc f_out
  prof=$(dirname "$0")/ref/models/deepseek41.sh
  line=$(sed -n '215p' "$prof")
  real=$(printf '%s\n' "$line" | sed -n 's/^DSPARK_MODEL=${BLOOMERY_DSPARK_MODEL:-\(.*\)}$/\1/p')
  [ -n "$real" ] || { echo "check-recipes: $prof:215 is not the DSPARK_MODEL default: $line" >&2; return 1; }
  f_rc=0
  f_out=$(env -i PATH="$PATH" BLOOMERY_MODEL=deepseek41 bash -c '. "$1" && dspark_model' _ "$rp" 2>"$d/f.err") || f_rc=$?
  [ "$f_rc" = 0 ] && [ "$f_out" = "$real" ] || { echo "check-recipes: dspark_model, real tier: rc=$f_rc out='$f_out', want '$real' ($(cat "$d/f.err"))" >&2; return 1; }
  echo "check-recipes: deepseek41 fixture draft: source stays up, dspark_model refuses"
  echo "check-recipes: fixture variants ok ($summary)"
}

# The fixture directory table (tools/fixture-dirs.tsv) as ref-paths.sh reads it, on a temporary fixture root: a row with a directory
# resolves the first shard of that directory, a `self` row runs on the profile's file (no fixture file, exit 0), a profile the table
# does not name has no fixture (66, naming the family) and a table that is not there is a named 66, never a family with no fixture.
blk_fixturedirs() {
  local dirs rp d n=0 arch dir out t
  dirs=$(dirname "$0")/fixture-dirs.tsv rp=$(dirname "$0")/ref/ref-paths.sh
  d=$(mktemp -d)
  # shellcheck disable=SC2064
  trap "rm -rf '$d'" RETURN
  rpd() { # rpd PROFILE [ref-paths.sh] : the fixture tier's FIXTURE_FILE and FIXTURE_DIR, then the exit code
    env -i PATH="$PATH" BLOOMERY_FIXTURE_ROOT="$d/fixtures" BLOOMERY_MODEL="$1" BLOOMERY_TIER=fixture bash -c '. "$1" && printf "%s|%s" "$FIXTURE_FILE" "$FIXTURE_DIR"' _ "${2:-$rp}" 2>&1
    echo "|rc=$?"
  }
  while IFS=$'\t' read -r arch dir; do
    case $arch in '' | '#'*) continue ;; esac
    n=$((n + 1))
    if [ "$dir" = self ]; then
      out=$(rpd "$arch")
      [ "$out" = "||rc=0" ] || { echo "check-recipes: $dirs: $arch is a self row, ref-paths.sh gave: $out" >&2; return 1; }
    else
      mkdir -p "$d/fixtures/$dir"
      touch "$d/fixtures/$dir/f-00001-of-00001.gguf"
      out=$(rpd "$arch")
      [ "$out" = "$d/fixtures/$dir/f-00001-of-00001.gguf|$d/fixtures/$dir|rc=0" ] ||
        { echo "check-recipes: $dirs: $arch is the directory $dir, ref-paths.sh gave: $out" >&2; return 1; }
    fi
  done < "$dirs"
  [ "$n" -gt 0 ] || { echo "check-recipes: $dirs holds no row" >&2; return 1; }
  # A copy of the reader beside a stub profile the table does not name, and then with no table.
  t=$d/tree
  mkdir -p "$t/tools/ref/models"
  cp "$rp" "$t/tools/ref/ref-paths.sh"
  cp "$dirs" "$t/tools/fixture-dirs.tsv"
  printf 'MODEL_NAME=nofixfam\nMODEL=/models/real/nofixfam.gguf\n' > "$t/tools/ref/models/nofixfam.sh"
  out=$(rpd nofixfam "$t/tools/ref/ref-paths.sh")
  case $out in
    *"the family 'nofixfam' has no fixture yet (no row in "*"fixture-dirs.tsv)"*"|rc=66") ;;
    *) echo "check-recipes: ref-paths.sh, a profile the table does not name: wanted a named 66 in: $out" >&2; return 1 ;;
  esac
  rm -f "$t/tools/fixture-dirs.tsv"
  out=$(rpd nofixfam "$t/tools/ref/ref-paths.sh")
  case $out in
    *"fixture-dirs.tsv, the table of the fixture directories, is not there"*"|rc=66") ;;
    *) echo "check-recipes: ref-paths.sh, no table file: wanted a named 66 in: $out" >&2; return 1 ;;
  esac
  echo "fixture dirs: $n rows read by ref-paths.sh"
}
# A fixture-tier command exports an empty draft path when it reads DSPARK_MODEL out of ref-paths.sh
# other than through dspark_model: zero drafts leave the variable empty and the export reaches a file
# open. dspark_model refuses that with exit 66. A justfile expansion, a printf of the variable, or any
# expansion in a shell script that sources ref-paths.sh fails here by file:line. A dspark_model call
# does not exempt another expansion in the same file. A quoted source path still counts as sourcing.
# A directory under tools/ that the walk cannot list ends the block with no verdict (2), never as one with no script.
blk_dsparkread() {
  local out rc=0
  out=$(python3 - "$JF" "$(dirname "$0")" << 'PY'
import os
import re
import sys
import tempfile

# The bad printf is built from parts so this file does not contain that byte string (the tree grep
# for it must stay empty, including on this checker).
PRINTF = re.compile(re.escape("printf %s \"" + "$" + "DSPARK_MODEL\""))
EXPAND = re.compile(r"(?<![A-Za-z0-9_])\$\{?DSPARK_MODEL\b")
SOURCE = re.compile(r"(?:\.|source)\s+[^\n]*ref-paths\.sh")


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
    return [f"{path}:{n}: reads DSPARK_MODEL instead of dspark_model"
            for n, code in code_lines(text) if EXPAND.search(code)]


def script_hits(text, path):
    lines = list(code_lines(text))
    sources = any(SOURCE.search(code) for _, code in lines)
    return [f"{path}:{n}: reads DSPARK_MODEL instead of dspark_model"
            for n, code in lines
            if PRINTF.search(code) or (sources and EXPAND.search(code))]


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
    bad += expect("a script that goes through dspark_model", script_hits("source tools/ref/ref-paths.sh\nx=$(dspark_model) || exit $?\necho " + '"$x"' + "\n", "fixture-ok.sh"), 0)
    bad += expect("a getter call then an unchecked export", script_hits("source tools/ref/ref-paths.sh; x=$(dspark_model); export BLOOMERY_DSPARK_MODEL=" + expand + "\n", "fixture-getter-export.sh"), 1)
    bad += expect(
        "a quoted source path then an unchecked export",
        script_hits(
            "source \"$(dirname \"$0\")/ref-paths.sh\"\n"
            "export BLOOMERY_DSPARK_MODEL=" + expand + "\n",
            "fixture-quoted-source.sh",
        ),
        1,
    )
    bad += expect("printf beside a dspark_model call", script_hits("x=$(dspark_model)\n" + bad_printf + "\n", "fixture-both.sh"), 1)
    bad += expect("the function's own file", script_hits("dspark_model() {\n  printf '%s' " + expand + "\n}\n", "ref-paths.sh"), 0)
    # timing-card.sh expands the variable and does not source ref-paths.sh; this rule does not own that file.
    bad += expect("a script that does not source ref-paths", script_hits("echo " + brace + "\n", "timing-card.sh"), 0)
    # a mode does not bind root, so the unlistable directory is a case only for another user
    if os.geteuid() != 0:
        d = tempfile.mkdtemp()
        os.mkdir(os.path.join(d, "hidden"), 0)
        try:
            scan(sys.argv[1], d)
            bad.append("check-recipes: draft-read fixture 'a directory the walk cannot list': scanned as one with no "
                       "script")
        except PermissionError:
            pass
        finally:
            os.rmdir(os.path.join(d, "hidden"))
            os.rmdir(d)
    return bad


def unlisted(e):
    raise e


def scan(justfile, tools):
    hits = []
    with open(justfile, encoding="utf-8") as fh:
        hits += justfile_hits(fh.read(), justfile)
    nsh = 0
    for root, dirs, files in os.walk(tools, onerror=unlisted):
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
try:
    hits, nsh = scan(sys.argv[1], sys.argv[2])
except OSError as e:
    print(f"check-recipes: draft reads: cannot read {e.filename} ({e.strerror}), no verdict")
    sys.exit(2)
if hits:
    print("check-recipes: a fixture-tier command exports an empty draft path:")
    print("\n".join(hits))
    sys.exit(1)
print(f"check-recipes: draft reads go through dspark_model ({nsh} shell files, the printf fixture is red)")
PY
  ) || rc=$?
  if [ "$rc" != 0 ]; then
    printf '%s\n' "$out" >&2
    [ "$rc" = 2 ] || echo "check-recipes: a fixture-tier command exports an empty draft path" >&2
    return "$rc"
  fi
  printf '%s\n' "$out"
}
# refset-publish.sh's assignment of the fx_ sets to a family (never by name), its named error on a set it cannot place, its
# left-out block (variant, DSpark) and its --upload refusal, on a fake tree, no box.
blk_refsetpublish() { tool_selftest refset-publish.sh; }
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

# The justfile reads and exports the shared architecture selection. Every recipe uses that value.
# PIN(2026-10-31): a machine whose cards are all cc 12.0 selects sm_120a and all cc 12.1 sm_121a through arch.sh, cards of
# different capabilities sm_86; recipes keep no independent target default.
blk_archowner() {
  python3 - "$JF" "$B" << 'PY'
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

DEF = 'arch := `bash tools/arch.sh`'
EXPORT = 'export BLOOMERY_ARCH := arch'
LITERAL = re.compile(r"sm_[0-9]")
OWNED = re.compile(r" \{\{arch\}\}(?![A-Za-z0-9_])")


def problems(text):
    out, defs, exports = [], 0, 0
    for n, line in enumerate(text.split("\n"), 1):
        if line == DEF:
            defs += 1
            continue
        if line == EXPORT:
            exports += 1
            continue
        if LITERAL.search(line):
            out.append(f"{n}: an arch literal outside the definition of `arch`: {line.strip()[:110]}")
        rest = line
        while (m := re.search(r"--arch", rest)) is not None:
            rest = rest[m.end():]
            if not OWNED.match(rest):
                out.append(f"{n}: `--arch` is not followed by {{{{arch}}}}: {line.strip()[:110]}")
    if defs != 1:
        out.append(f"the justfile defines `arch` {defs} times, want exactly one `{DEF}` line")
    if exports != 1:
        out.append(f"the justfile exports the selected arch {exports} times, want exactly one `{EXPORT}` line")
    return out


good = DEF + "\n" + EXPORT + "\nx:\n    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p y'\n"
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
    ("two definitions", good + DEF + "\n", 1),
    ("no export of the selected arch", good.replace(EXPORT + "\n", ""), 1),
    ("two exports", good + EXPORT + "\n", 1),
    ("independent target default", good.replace(DEF, "arch := env_var_or_default('BLOOMERY_ARCH', 'sm_86')"), 2),
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
fixture = Path(sys.argv[2]) / 'arch'
fixture.mkdir()
probe = fixture / 'nvidia-smi'
probe.write_text('#!/bin/sh\n[ "$*" = "--query-gpu=compute_cap --format=csv,noheader" ] || exit 2\n'
                 '[ "$ARCH_TEST_CAPABILITY" != error ] || exit 1\nprintf "%s\\n" "$ARCH_TEST_CAPABILITY"\n')
probe.chmod(0o755)
script = Path(sys.argv[1]).parent / 'tools/arch.sh'
env = dict(os.environ, PATH=str(fixture) + os.pathsep + os.environ['PATH'])
env.pop('BLOOMERY_ARCH', None)
bash = shutil.which('bash')
cases = [
    ('12.1', None, 'sm_121a', 0),
    ('12.1\n12.1', None, 'sm_121a', 0),
    ('12.1\n12.0', None, 'sm_86', 0),
    (None, None, 'sm_86', 0),
    ('12.0', None, 'sm_120a', 0),
    ('12.0\n12.0', None, 'sm_120a', 0),
    ('12.0\n8.6', None, 'sm_86', 0),
    ('8.6\n12.0', None, 'sm_86', 0),
    ('8.6', None, 'sm_86', 0),
    ('12.1', 'sm_121', 'sm_121', 0),
    ('error', 'sm_86', 'sm_86', 0),
    ('error', None, '', 64),
    ('12.1', '', '', 64),
    ('12.1', 'sm_121x', '', 64),
]
for capability, override, want, rc in cases:
    case = dict(env, ARCH_TEST_CAPABILITY=capability)
    if capability is None:
        case['ARCH_TEST_CAPABILITY'] = ''
        case['PATH'] = str(fixture / 'no-driver')
    if override is not None:
        case['BLOOMERY_ARCH'] = override
    for command in ([bash, str(script)],
                    [bash, '-c', '. "$1"; bloomery_arch fixture || exit $?; "$2" -c \'printf "%s\\n" "$BLOOMERY_ARCH"\'',
                     'bash', str(script), bash]):
        got = subprocess.run(command, env=case, text=True, capture_output=True, timeout=10)
        if got.returncode != rc or got.stdout.strip() != want:
            sys.exit(f'check-recipes: arch selection {capability=} {override=}: {got.returncode=} {got.stdout=} {got.stderr=}')
print(f"check-recipes: arch owner ok ({uses} recipe lines build with --arch {{{{arch}}}}, {len(fixtures)} ownership fixtures, {2 * len(cases)} selection cases)")
PY
}

# units.sh's tree_commit is the one commit label of contrib-check.sh, nightly/run.sh, gate-batch.sh, ptx-compare.sh and
# narrow-scan.sh: a `git status` that fails (a corrupt index) is a refusal naming the tree and git's exit code, never a commit
# published without `+local`. Its labels on temp repositories, and the first two scripts stopping on a corrupt index
# (contrib-check the real script in this tree, run.sh a clone under a temp NIGHTLY_HOME), no box and no cargo; the other three
# test their use in their own self-tests (blk_gatebatch, blk_ptxcompare, blk_narrowscan).
blk_treecommit() { tool_selftest nightly/selftest.sh; }

# A `git archive | tar` pipe fails now and then (rc 141, tar 0): a tar that stops at the end-of-archive marker (bsdtar, the Mac's)
# exits while git is still writing the zero padding of its last block, and git dies of SIGPIPE. units.sh's archive_tree and
# recipes.py's archive() are the one reader of a commit's tree, through a file; a script or tool that pipes again is red. The two
# fixture lines are the FAIL-first (the pipe is found, the file form is not); tools/nightly/selftest.sh pipes on purpose, to
# prove its shims lose the pipe.
blk_archivepipe() {
  source "$(dirname "$0")/grep-answer.sh" || return $?
  local d hits pattern='git[^|#]*archive[^|#]*\|[[:space:]]*tar|Popen\(\["git", "archive"'
  d=$(mktemp -d "$B/archive-pipe.XXXXXX") || return 1
  printf '%s\n' 'git -C "$r" archive --format=tar "$rev" | tar -x -C "$d"' > "$d/pipe.sh"
  printf '%s\n' 'git -C "$r" archive --format=tar -o "$f" "$rev" && tar -x -f "$f" -C "$d"' > "$d/file.sh"
  hits=$(grep_answer check-recipes -rnE "$pattern" --include='*.sh' --include='*.py' "$d") || return $?
  if [ "$hits" != "$d/pipe.sh:1:git -C \"\$r\" archive --format=tar \"\$rev\" | tar -x -C \"\$d\"" ]; then
    echo "check-recipes: the archive-pipe scan does not tell a pipe from a file on its fixtures" >&2
    return 1
  fi
  hits=$(grep_answer check-recipes -rnE "$pattern" --include='*.sh' --include='*.py' "$(dirname "$0")" | grep_answer check-recipes -vE '/(check-recipes|nightly/selftest)\.sh:') || return $?
  if [ -n "$hits" ]; then
    printf '%s\n' "$hits" >&2
    echo "check-recipes: a git archive piped into tar (the pipe loses to a tar that stops at the end marker): use archive_tree (tools/nightly/units.sh) or archive() (tools/recipes.py)" >&2
    return 1
  fi
  echo "check-recipes: archive pipe ok (no git archive piped into tar in tools/)"
}

# A shell function has one owner in its file: a second definition at the same indent with the same body is dead weight
# that reads as a second owner. The same name with another body is a stub or an override and stays. The fixtures are the
# FAIL-first: the pair the rule forbids is red on its own copy, the shapes it must leave are green.
blk_fnowner() {
  python3 - "$(cd "$(dirname "$0")" && pwd)" << 'PY'
import pathlib
import re
import sys

HEAD = re.compile(r"^([ \t]*)([A-Za-z_][A-Za-z0-9_:.-]*)[ \t]*\(\)[ \t]*\{(.*)$")
HEREDOC = re.compile(r"(?<!<)<<(?!<)-?[ \t]*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?")


def problems(text, where):
    lines = text.split("\n")
    seen, out, end = {}, [], None
    for i, line in enumerate(lines):
        # A here-document is another file's text (a stub script, a python body), not this file's functions.
        if end is not None:
            if line.strip() == end:
                end = None
            continue
        h = HEREDOC.search(line)
        if h is not None:
            end = h.group(1)
        m = HEAD.match(line)
        if m is None:
            continue
        indent, name, rest = m.groups()
        if rest.strip().endswith("}") and rest.strip() != "":
            body = rest.strip()[:-1].strip()
        elif rest.strip() == "":
            j = i + 1
            while j < len(lines) and lines[j] != indent + "}":
                j += 1
            body = "\n".join(lines[i + 1 : j])
        else:
            continue
        key = (name, indent, body)
        if key in seen:
            out.append(f"{where}:{i + 1}: {name} is defined again with the body of line {seen[key]}")
        else:
            seen[key] = i + 1
    return out


multi = "f() {\n  echo a\n}\n"
fixtures = [
    ("one definition", multi, 0),
    ("the same body twice at one indent", multi + multi, 1),
    ("the same body three times", multi * 3, 2),
    ("one line, the same body twice", "f() { echo a; }\nf() { echo a; }\n", 1),
    ("the same name with another body", multi + "f() {\n  echo b\n}\n", 0),
    ("the same body at another indent", multi + "  f() {\n    echo a\n  }\n", 0),
    ("two names with one body", multi + multi.replace("f()", "g()"), 0),
    ("the same one-line body in a here-document", "cat << 'EOF'\nf() { echo a; }\nEOF\nf() { echo a; }\n", 0),
    ("the same body twice inside one here-document", "cat << 'EOF'\n" + multi + multi + "EOF\n", 0),
]
bad = 0
for name, text, want in fixtures:
    got = problems(text, "fixture")
    if len(got) != want:
        bad += 1
        print(f"check-recipes: shell function owner fixture '{name}': {len(got)} problems, want {want}: {got}", file=sys.stderr)
if bad:
    sys.exit(1)
root = pathlib.Path(sys.argv[1])
found, scripts = [], 0
for path in sorted(root.rglob("*.sh")):
    scripts += 1
    found += problems(path.read_text(encoding="utf-8", errors="replace"), f"tools/{path.relative_to(root)}")
if found:
    print("check-recipes: a shell function is defined twice with the same body in one file (delete the second):", file=sys.stderr)
    for f in found:
        print(f"  {f}", file=sys.stderr)
    sys.exit(1)
print(f"check-recipes: shell function owner ok ({scripts} scripts, {len(fixtures)} fixtures)")
PY
}

# The cpu-guard live pair reads a real /proc. It runs alone (the scheduler's `alone` job), so it does not share the
# runner with the blocks (on CI's two cores a foreign busy loop beside it can miss its window).
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

# blk_gpugate's holder judge, read from its transcript at the block's end. Linux: its success line, and not the /proc skip.
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

BLOCKS=(prelude recipescheck gateclass selftest smoke cardorder cardtests lease boxtracks unsafecount archscan loadscan commentscan leverscan loadgroups lcppfit coldblocks leverarms slotsarm binlabel lcppwarm q38srv
  maccheck gatebatch gpugate stackwatch ptxscan ptxspill scanargs narrowscan canoncheck ptxcompare ldsscan mutantrun macstatic macremote boxsh oxidehost oxidebackend seedtarget carry releasebuild releaseinstall releasepublish releasesitting pytools boxqcodes fixturevariants fixturedirs dsparkread
  refsetpublish orphan archowner treecommit archivepipe fnowner wt macgate boundedscan gatebound dumpdequant tokoracle ikpintree qdotrefargs)
# blk_keyspec NAME: what a block's input key reads (tools/recipes.py static-key). Every block's is the whole tree: a
# narrower key needs proof that the block reads nothing outside it, and a green run on a copy whose other files are
# missing or empty is no proof (a scan of the justfile or crates/ passes on empty input). procleg is never skipped.
blk_keyspec() {
  case $1 in
    procleg) echo - ;;
    *) echo tree ;;
  esac
}
# units: the blocks as the scheduler runs them, one a line: gate-batch's self-test runs as its parts (`--part K`, the
# count its SELFTEST_PARTS line holds) and pytools as its list check and one unit a tool, so the scheduler sees each one's
# own wall and slot.
units() {
  local b k parts
  for b in "${BLOCKS[@]}"; do
    if [ "$b" = pytools ]; then
      echo pytools.list
      pytools_list
      for ((k = 1; k <= ${#selftests[@]}; k++)); do echo "pytools.$k"; done
    elif [ "$b" = gatebatch ]; then
      parts=$(sed -n 's/^SELFTEST_PARTS=\([1-9][0-9]*\)$/\1/p' "$(dirname "$0")/gate-batch.sh")
      [ -n "$parts" ] || { echo "check-recipes: tools/gate-batch.sh has no SELFTEST_PARTS line" >&2; return 70; }
      for ((k = 1; k <= parts; k++)); do echo "gatebatch.$k"; done
    else
      echo "$b"
    fi
  done
}
# jobs: every unit as a line of tools/static-sched.py's JOBS, run from the repository root. procleg, the cpu-guard
# live pair, runs alone.
jobs() {
  local u list
  list=$(units) || return
  printf 'check-recipes.procleg\talone\t-\tbash tools/check-recipes.sh --block procleg\n'
  for u in $list; do
    printf 'check-recipes.%s\t-\t%s\tbash tools/check-recipes.sh --block %s\n' "$u" "$(blk_keyspec "${u%%.*}")" "$u"
  done
}
# report DIR: the blocks' verdict from a tools/static-sched.py run's DIR, in this file's order, procleg first: a green
# block's output on stdout, a red one's on stderr, a block skipped by its green key as one line naming the green row;
# then the five slowest blocks that ran, and the first red block's rc (255: a block that wrote no rc).
report() {
  local d=$1 b brc rc=0 ran=0 skipped=0 slowest list
  list=$(units) || return
  for b in procleg $list; do
    if [ -f "$d/check-recipes.$b.skip" ]; then
      echo "check-recipes: $b skipped, its input key is $(cat "$d/check-recipes.$b.skip")"
      skipped=$((skipped + 1))
      continue
    fi
    ran=$((ran + 1))
    brc=$(cat "$d/check-recipes.$b.rc" 2> /dev/null) || brc=255
    if [ "$brc" != 0 ]; then
      cat "$d/check-recipes.$b.log" >&2 2> /dev/null || true
      [ "$brc" != 255 ] || echo "check-recipes: $b wrote no rc (ended before its scheduler reaped it)" >&2
      [ "$rc" != 0 ] || rc=$brc
    else
      cat "$d/check-recipes.$b.log"
    fi
  done
  slowest=$(for b in $list; do
    [ -f "$d/check-recipes.$b.skip" ] || echo "$(cat "$d/check-recipes.$b.wall" 2> /dev/null || echo 0) $b"
  done | sort -rn | awk 'NR <= 5 {printf "%s%s %s s", (NR > 1 ? ", " : ""), $2, $1}')
  echo "check-recipes: slowest: $slowest"
  echo "check-recipes: $ran blocks ran, $skipped skipped by a green input key"
  [ "$rc" = 0 ] || return "$rc"
  echo "check-recipes: ok"
}

case ${1:-} in
  --jobs)
    [ $# = 1 ] || { echo "check-recipes.sh: --jobs takes nothing" >&2; exit 64; }
    jobs
    exit 0
    ;;
  --block)
    [ $# = 2 ] && [[ " $(units | tr '\n' ' ')procleg " == *" $2 "* ]] || { echo "check-recipes.sh: --block takes one of: procleg $(units | tr '\n' ' ')" >&2; exit 64; }
    set +e
    case $2 in
      procleg) proc_leg ;;
      *.*) "blk_${2%%.*}" "${2#*.}" ;;
      *) "blk_$2" ;;
    esac
    exit $?
    ;;
  --report)
    [ $# = 2 ] && [ -d "$2" ] || { echo "check-recipes.sh: --report takes a tools/static-sched.py run's DIR" >&2; exit 64; }
    report "$2"
    exit $?
    ;;
  '') ;;
  *) echo "usage: tools/check-recipes.sh [--jobs | --block UNIT | --report DIR]" >&2; exit 64 ;;
esac
# Every block at once is ~60 processes, some of them parallel themselves: one scheduler, tools/static-sched.py, runs
# them at most a cpu count at once, the longest recorded first (tools/mac-static.sh gives it this file's jobs beside its
# own steps, keyed).
cd "$(dirname "$0")/.."
jobs > "$B/jobs.tsv"
src=0
python3 tools/static-sched.py run --out "$B/s" --times "$HOME/.cache/bloomery/static-times.tsv" "$B/jobs.tsv" || src=$?
[ "$src" = 0 ] || { echo "check-recipes: tools/static-sched.py ended rc $src" >&2; exit "$src"; }
report "$B/s"
