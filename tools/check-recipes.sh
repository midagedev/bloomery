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
echo "check-recipes: ok"
