#!/usr/bin/env bash
# justfile 레시피 점검 — 맥에서 돈다(grep뿐, 빌드 없음).
# 막는 실패: 게이트 줄 뒤에 붙은 `||`가 종료 코드를 삼키는 것. ef9e579가 `cargo test … || echo "TIMED OUT"`로
# 열세 게이트 전부를 "빨강이어도 0"으로 만들었다(2026-09-20, tools/gate.sh 머리말 참조).
# 게이트의 종료 코드는 tools/gate.sh가 소유한다 — 시험을 돌리는 레시피 줄에 `||`가 있으면 빨강.
set -euo pipefail
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
# Every Python tool's own tests, on the Mac (seconds in all): a self-test that no check runs rots. A tool
# that grows one is listed here, and the comparison below fails on one that is not.
selftests=(
  "tools/bloomery/manifest.py --self-test"
  "tools/flow/ds41_prefill.py --self-test"
  "tools/ref/check-int-twins.py --self-test"
  "tools/ref/draft-accept.py --self-test"
  "tools/ref/gguf-ranges.py --self-test"
  "tools/ref/ptx-canon.py --self-test"
  "tools/ref/ds41pp.py self-test"
  "tools/ref/router-coverage.py --self-test"
  "tools/ref/router-hotlist.py --self-test"
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
echo "check-recipes: ok"
