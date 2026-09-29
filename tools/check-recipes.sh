#!/usr/bin/env bash
# justfile 레시피 점검 — 맥에서 돈다(grep뿐, 빌드 없음).
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
# Every recipe's cargo targets, features and runner binaries against the workspace (cargo metadata, no
# build): a `--bin`, `--test` or `-p` that names nothing, a feature the package lacks, a gpu-gate.sh or
# host-gate.sh name the recipe does not build, a named script that is not in the tree, a gate-* recipe
# with no cargo target. The parser is tools/recipes.py, the one tools/affected-gates.sh uses; its own
# tests (--self-test) run here too, so the parser the lead's batch lists come from is tested wherever
# this check runs.
python3 "$(dirname "$0")/recipes.py" check
python3 "$(dirname "$0")/recipes.py" --self-test
"$(dirname "$0")/gate-batch.sh" --smoke --dry-run > /dev/null
# A timed recipe checks its card before it builds: tools/ref/card-precheck.sh (the `precheck` variable)
# comes before the first cargo build of its box command, so a missing or refused card costs no build.
# Timed is gate-batch.sh's class T (--classes: the classifier the batch runs, not a copy of its
# regexes). A recipe whose build is a just dependency cannot run the precheck first from its own box
# command; those are named on one line and do not fail here.
classes=$(mktemp)
trap 'rm -f "$classes"' EXIT
"$(dirname "$0")/gate-batch.sh" --classes > "$classes"
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
# The card and lease stub tests (tools/ref/card-tests/run.sh): card.py, lease_take's refusals,
# lease-hold.sh and gpu-ab.py's card check, against a copy of that code and a lock file of their own.
if ! cards=$("$(dirname "$0")/ref/card-tests/run.sh" 2>&1); then
  echo "$cards" >&2
  echo "check-recipes: the card tests failed" >&2
  exit 1
fi
echo "${cards##*$'\n'}"
# lease.sh's own self-test: lease_gpu_idle (the take's timing-card check) and the witness tag, on a
# stub nvidia-smi in a temp dir, no card, no lock, no /proc.
if ! lse=$(bash "$(dirname "$0")/ref/lease.sh" --self-test 2>&1); then
  echo "$lse" >&2
  echo "check-recipes: the lease self-test failed" >&2
  exit 1
fi
echo "${lse##*$'\n'}"
# box-tracks.sh deletes remote track directories: its selection (only the names given, never a new stale
# the reader did not see, a bare --remove refused) is tested on fixed input, no ssh.
if ! bt=$(bash "$(dirname "$0")/box-tracks.sh" --self-test 2>&1); then
  echo "$bt" >&2
  echo "check-recipes: the box-tracks self-test failed" >&2
  exit 1
fi
echo "${bt##*$'\n'}"
# load-groups.sh decides which arms of a depth runner share one load, and their order: its grouping and
# rotation are tested on fixed keys, no process started.
if ! lg=$(bash "$(dirname "$0")/ref/load-groups.sh" --self-test 2>&1); then
  echo "$lg" >&2
  echo "check-recipes: the load-groups self-test failed" >&2
  exit 1
fi
echo "${lg##*$'\n'}"
# lcpp-fit.sh builds the depth runners' llama.cpp fit arms and reads what the fit chose: its flag
# rewrite, its --help probe (stub binaries) and its column (fixture output) are tested here, no box.
if ! lf=$(bash "$(dirname "$0")/ref/lcpp-fit.sh" --self-test 2>&1); then
  echo "$lf" >&2
  echo "check-recipes: the lcpp-fit self-test failed" >&2
  exit 1
fi
echo "${lf##*$'\n'}"
# cold-blocks.sh is the depth runners' cold tag and engine-block planner: its bound, its flag rewrite, its
# order refusal and its block plan on fixed arms are tested here, no box.
if ! cb=$(bash "$(dirname "$0")/ref/cold-blocks.sh" --self-test 2>&1); then
  echo "$cb" >&2
  echo "check-recipes: the cold-blocks self-test failed" >&2
  exit 1
fi
echo "${cb##*$'\n'}"
# lcpp-warm.sh is the depth runners' llama-server arms: its flag translation (against V4.1's LCPP_CLI_FLAGS),
# its refusals, the context, the --help probe and a stub server's start, requests, checks and stop (python3
# and curl, 127.0.0.1) are tested here, no box.
if ! lw=$(bash "$(dirname "$0")/ref/lcpp-warm.sh" --self-test 2>&1); then
  echo "$lw" >&2
  echo "check-recipes: the lcpp-warm self-test failed" >&2
  exit 1
fi
echo "${lw##*$'\n'}"
# mac-check.sh runs the check and lint recipes' box commands on the Mac: its derivation from those
# recipes, its refusals, the ratchet and the prerequisite checks (a fake HOME) are tested here, no cargo.
if ! mc=$(bash "$(dirname "$0")/mac-check.sh" --self-test 2>&1); then
  echo "$mc" >&2
  echo "check-recipes: the mac-check self-test failed" >&2
  exit 1
fi
echo "${mc##*$'\n'}"
# gate-batch.sh's placement rules (the v41-load lane, a batch or a solo recipe's arm as an item, a cold
# build's times row) on a fixture justfile and fixture times rows, no box.
if ! gbt=$(bash "$(dirname "$0")/gate-batch.sh" --self-test 2>&1); then
  echo "$gbt" >&2
  echo "check-recipes: the gate-batch self-test failed" >&2
  exit 1
fi
echo "${gbt##*$'\n'}"
# gpu-gate.sh's card choice, its lease refusals and the V4.1 load lock, against lock files of its own and
# a stub nvidia-smi (flock and timeout stand-ins where the host has none, as on the Mac), no card.
if ! ggt=$(bash "$(dirname "$0")/gpu-gate.sh" --self-test 2>&1); then
  echo "$ggt" >&2
  echo "check-recipes: the gpu-gate self-test failed" >&2
  exit 1
fi
echo "${ggt##*$'\n'}"
# stack-watch.sh (gpu-gate.sh's BLOOMERY_GATE_STACKS): output passed through, a quiet stub dumped and ended by
# its comm among the spawned command's descendants only, a quiet command with no such process left alone.
if ! swt=$(bash "$(dirname "$0")/ref/stack-watch.sh" --self-test 2>&1); then
  echo "$swt" >&2
  echo "check-recipes: the stack-watch self-test failed" >&2
  exit 1
fi
echo "${swt##*$'\n'}"
# ptx-spill-check.sh's table reading (a process-substitution table read for every binary, a binary with no
# pinned row) and its verdict lines, through a stub ptx-scan.sh on fixed scans, no build.
if ! psc=$(bash "$(dirname "$0")/ptx-spill-check.sh" --self-test 2>&1); then
  echo "$psc" >&2
  echo "check-recipes: the ptx-spill-check self-test failed" >&2
  exit 1
fi
echo "${psc##*$'\n'}"
# scan-args.sh is the three scan recipes' refusal of a cargo feature given where the scan's own words go
# (ptx-scan, sass-scan, lds-scan): its cases on a fixed feature list and the real crates/gpu-gates table.
if ! sa=$(bash "$(dirname "$0")/scan-args.sh" --self-test 2>&1); then
  echo "$sa" >&2
  echo "check-recipes: the scan-args self-test failed" >&2
  exit 1
fi
echo "${sa##*$'\n'}"
# lds-scan.sh's rows (its PTX and SASS counts over a fixture binary, stub extractor, ptxas and cuobjdump),
# its filter, its failed-scan banner and its usage refusal, no box.
if ! ls=$(bash "$(dirname "$0")/lds-scan.sh" --self-test 2>&1); then
  echo "$ls" >&2
  echo "check-recipes: the lds-scan self-test failed" >&2
  exit 1
fi
echo "${ls##*$'\n'}"
# mutant-run.sh's kill, survive, not-built and no-Compiling verdicts, its restore checks (a broken copy, an
# edit during the run, a TERM) and its refusals, in a temp git repo with a fake gate, no box.
if ! mrt=$(bash "$(dirname "$0")/mutant-run.sh" --self-test 2>&1); then
  echo "$mrt" >&2
  echo "check-recipes: the mutant-run self-test failed" >&2
  exit 1
fi
echo "${mrt##*$'\n'}"
# Every Python tool's own tests, on the Mac (seconds in all): a self-test that no check runs rots. A tool
# that grows one is listed here, and the comparison below fails on one that is not.
selftests=(
  "tools/bloomery/manifest.py --self-test"
  "tools/bloomery/records.py --self-test"
  "tools/bloomery/route_trace.py --self-test"
  "tools/check-comment-only.py --self-test"
  "tools/flow/ds41_prefill.py --self-test"
  "tools/flow/pplb.py --self-test"
  "tools/flow/routes.py --self-test"
  "tools/mac-disk.py --self-test"
  "tools/ref/check-int-twins.py --self-test"
  "tools/ref/dma-dram-share.py --self-test"
  "tools/ref/draft-accept.py --self-test"
  "tools/ref/draft-vocab.py --self-test"
  "tools/ref/ds41copy.py --self-test"
  "tools/ref/ds41pp.py self-test"
  "tools/ref/gguf-ranges.py --self-test"
  "tools/ref/ptx-canon.py --self-test"
  "tools/ref/route-trace-chat.py --self-test"
  "tools/ref/router-coverage.py --self-test"
  "tools/ref/router-hotlist.py --self-test"
  "tools/ref/router-residency.py --self-test"
  "tools/ref/window-union.py --self-test"
  "tools/verdict-diff.py --self-test"
)
root="$(cd "$(dirname "$0")/.." && pwd)"
listed=$(printf '%s\n' "${selftests[@]}" | cut -d' ' -f1 | sort)
found=$(cd "$root" && grep -rlE -e '--self-test|"self-test"' --include='*.py' tools | grep -vx 'tools/recipes.py' | sort)
if [ "$listed" != "$found" ]; then
  echo "check-recipes: the Python tools with a self-test and the list this check runs differ (< listed, > found):" >&2
  diff <(echo "$listed") <(echo "$found") >&2 || true
  exit 1
fi
for t in "${selftests[@]}"; do
  read -r f arg <<< "$t"
  if ! out=$(cd "$root" && python3 "$f" "$arg" 2>&1); then
    echo "$out" >&2
    echo "check-recipes: $f $arg failed" >&2
    exit 1
  fi
done
echo "check-recipes: ${#selftests[@]} tool self-tests ok"
# Every #[test] in the workspace is run by some gate-* or lab-* recipe's cargo test call on the box: its
# target, the features its path's cfgs need, its name filter and its #[ignore] (tools/recipes.py
# orphan-tests). A test no gate runs is neither a test nor a gate: without gate-ds41-bind, gpu-gates' bind
# tests sit behind a feature no recipe enables. An input the scan cannot read fails here by file and line.
if ! ot=$(python3 "$(dirname "$0")/recipes.py" orphan-tests 2>&1); then
  echo "$ot" >&2
  echo "check-recipes: a test no gate-* or lab-* recipe runs, or a source the scan cannot read (tools/recipes.py orphan-tests)" >&2
  exit 1
fi
echo "${ot##*$'\n'}"
echo "check-recipes: ok"
