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
# with no cargo target. The parser is tools/recipes.py, the one tools/affected-gates.sh uses; its own
# tests (--self-test) run here too, so the parser the lead's batch lists come from is tested wherever
# this check runs.
python3 "$(dirname "$0")/recipes.py" check
# The blocks below are independent — each runs in its own subshell on fixtures in its own temp
# directory — so they run as parallel background jobs and report in this file's order after a barrier:
# each block's own output (its last line where the serial form printed only that), its full output and
# its named error on failure, and the first red block's own rc ends the check. The serial form's
# ~172 s wall was these blocks one after another (measured 2026-10-07); the barrier holds every
# assertion the serial form held.
B=$(mktemp -d)
trap 'rm -rf "$B"' EXIT

# blk_NAME: one block. Its stdout is what the serial form printed; a failure prints its named error to
# stderr and returns nonzero. `$(dirname "$0")` still names this script's directory inside a function.
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
    """The recipe's text with the justfile's variables in place; a parameter stays {{}}."""
    out = []
    for frags in recipes[name]["body"]:
        line = ""
        for f in frags:
            if isinstance(f, str):
                line += f
            elif len(f) == 1 and f[0][0] == "variable" and f[0][1] in assigns:
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
  if ! ggt=$(bash "$(dirname "$0")/gpu-gate.sh" --self-test 2>&1); then
    echo "$ggt" >&2
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
    "tools/boxq.py --self-test"
    "tools/check-comment-only.py --self-test"
    "tools/check-defaults.py --self-test"
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
    "tools/ref/gguf-ranges.py --self-test"
    "tools/ref/hidden-diff.py --self-test"
    "tools/ref/lev_ref.py --self-test"
    "tools/ref/nsys-bridge.py --self-test"
    "tools/ref/ptx-canon.py --self-test"
    "tools/ref/qdot-rate-floors.py --self-test"
    "tools/ref/route-trace-chat.py --self-test"
    "tools/ref/router-coverage.py --self-test"
    "tools/ref/router-residency.py --self-test"
    "tools/ref/set-diff.py --self-test"
    "tools/ref/window-union.py --self-test"
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
# they overlap none of the codes tools/ and the justfile already name (tools/boxq.py codes-check).
blk_boxqcodes() {
  if ! bc=$(python3 "$(dirname "$0")/boxq.py" codes-check 2>&1); then
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
  echo "check-recipes: fixture variants ok ($n row(s): $tags )"
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

BLOCKS=(selftest smoke cardorder cardtests lease boxtracks loadgroups lcppfit coldblocks slotsarm lcppwarm q38srv
  maccheck gatebatch gpugate stackwatch ptxspill scanargs narrowscan ldsscan mutantrun macstatic carry pytools boxqcodes fixturevariants
  orphan)
for b in "${BLOCKS[@]}"; do
  ( set +e; blk_$b > "$B/$b.out" 2>&1; echo $? > "$B/$b.rc" ) & # set +e: a red block writes its own rc
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
[ "$rc" = 0 ] || exit "$rc"
echo "check-recipes: ok"
