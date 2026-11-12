#!/usr/bin/env bash
# 게이트 러너 — 박스에서, box.sh가 들어간 원격 디렉터리에서 돈다. 인자는 `cargo test` 뒤에 그대로 붙는다.
#   ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model --test mt -- --ignored --nocapture'
# --oxide는 디바이스 크레이트용으로 `cargo oxide test --arch $BLOOMERY_ARCH --` 뒤에 인자를 붙인다(arch는
# tools/arch.sh가 읽는다, 설정이 없으면 보이는 카드에서 고른다. plain cargo는 디바이스 크레이트를 빌드하지 못한다 — AGENTS.md 「Never」).
#   ./tools/box.sh 'bash tools/gate.sh --gpu --oxide -p bloomery-gpu --release --lib'
#
# 막는 실패 둘:
#  1. 매달린 게이트(2026-09-20 q_nope2 무한루프가 gate-mt를 한 시간 넘게 붙잡음) — 900초 상한,
#     넘으면 빨간 게이트로 끝난다.
#  2. 삼켜진 종료 코드 — 상한을 처음 넣은 ef9e579의 레시피는 `timeout … cargo test … || echo "TIMED OUT"`
#     꼴이었고, echo가 성공하므로 시험 실패든 타임아웃이든 레시피가 0으로 끝났다(같은 날 저녁 리뷰에서
#     발견; 평범한 시험 실패에도 "TIMED OUT"이 찍혔다). 종료 코드의 소유자는 이 스크립트 하나다:
#     cargo의 코드를 그대로 돌려주고, 타임아웃 문구는 타임아웃일 때만 찍는다.
# Two bounds, parsed by tools/gate-bound.sh: a value it refuses ends this runner with 64 before cargo runs.
# BLOOMERY_GATE_BOUND (900) is the test's: its run (a GPU test's is gpu-gate.sh's, after its lock waits).
# BLOOMERY_GATE_BUILD_BOUND (2700) is the build's: a build that waits on the build-directory lock and then compiles a
# cold dependency cone is slow, not hung, and must not be killed with its compiler work thrown away. A call without
# --gpu builds first by its own cargo call (`cargo test … --no-run`: cargo's arguments, before the first `--`) under
# the build bound, then runs under the run bound; the second call finds everything built (a --doc call has no build call:
# cargo refuses --no-run with --doc). A build that runs out ends 124 and says it was the build phase; the run never starts.
# --gpu before --oxide selects one native --lib or --bin test target. Its bounded Cargo call compiles
# outside the card lock and invokes a capture-only target runner, never the test. Cargo owns the
# executable, arguments, package cwd and runtime environment; a private JSON capture retains them.
# Replay enters gpu-gate.sh in that context, so the common runner applies card selection last and
# alone owns the runtime bound (lock waits are outside it). No second Cargo call runs under the lock.
# The Cargo call passes no --target: it is the host build every other oxide recipe makes, so `bloomery-gpu` and the
# rest of the dependency cone are one set of compiled units for all of them (a --target build keys every unit apart,
# and the whole cone compiles again). The host triple's runner comes from --config, which outranks every other config
# source. Doctests, an explicit --target and a cross triple from CARGO_BUILD_TARGET, `--config build.target=` or a
# `[build] target` are refused with 64: a cross triple takes its own runner, or none, and would run the test before
# the card lock. Other calls keep their existing cargo-test path.
# The self-test's leading --test-locks DIR forwards gpu-gate.sh's existing isolated-lock fixture.
#
# A third failure it closes: a filter that matches nothing. libtest passes a run of 0 tests, so a
# call whose test-name filter names no test that exists (or none that runs under its ignore flags) is
# green and checks nothing. A call that carries a filter — an argument after its `--` that is not a
# flag, nor the value of an option in LIBTEST_VALUE below — and that cargo (or the GPU test) ends with 0 fails with 78
# when the `passed` counts of its `test result:` lines sum to 0 over the whole call (one target
# printing "running 0 tests" is normal: `--lib --test x -- f` names a test of one target). A call with
# no filter, or with `--list`, is not judged.
#
# Exit codes: cargo's or the GPU test's own (0 green, 101 a test failed or a build failed, …); 124 or 137 a bound ran
# out (the build's, named, or the run's); 64 invalid bound or GPU test selection; 78 a filter matched no test that passed.
# GPU calls also preserve gpu-gate.sh's refusals (2 missing executable, 69 lock-open failure, 75 contention).
#
# BLOOMERY_BUILD_ONLY=1 (tools/gate-batch.sh's build lane puts it in an item's box env): the call runs its build phase as
# above, under the build bound, and ends there with the build's own code; the test phase, the filter check and the GPU replay
# never start, so the call takes no card and no lock. A --doc call has no build phase (its doctests compile inside the run):
# it says so and ends 0, and the batch builds such a recipe inline (its rule, tools/gate-batch.sh). 0 or unset runs the test as
# always; any other value ends 64. tools/gpu-gate.sh reads the same variable and exits 0 before its locks.
set -uo pipefail
# Cargo's internal target runner: record its execution context, without running the executable.
if [ "${1:-}" = --capture-test ]; then
  [ $# -ge 3 ] && [ -d "$2" ] || exit 64
  shift
  exec python3 - "$@" <<'PY'
import json
import os
import sys
import tempfile

directory, executable, *args = sys.argv[1:]
fd, path = tempfile.mkstemp(prefix="run-", suffix=".json", dir=directory)
with os.fdopen(fd, "w", encoding="utf-8") as stream:
    json.dump({"argv": [os.path.abspath(executable), *args], "cwd": os.getcwd(), "env": dict(os.environ)}, stream)
PY
fi
# shellcheck source=tools/gate-bound.sh
source "${BASH_SOURCE[0]%/*}/gate-bound.sh"
gate_bound gate.sh || exit $?
gate_build_bound gate.sh || exit $?
case ${BLOOMERY_BUILD_ONLY:-0} in
  0 | 1) ;;
  *) echo "gate.sh: BLOOMERY_BUILD_ONLY is 1 (build, run nothing) or 0/unset, got '$BLOOMERY_BUILD_ONLY'" >&2; exit 64 ;;
esac
BUILD_ONLY=${BLOOMERY_BUILD_ONLY:-0}
# build_bounded <command…>: the build phase under BUILD_BOUND, in this shell (run_bounded's contract); returns its status.
# The caller ends with it, so a build that ran out or failed never starts the run.
build_bounded() {
  run_bounded timeout --kill-after=10 "$BUILD_BOUND" "$@"
  local rc=$?
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    echo "GATE TIMED OUT in the build phase after the ${BUILD_BOUND}s build bound (exit $rc, BLOOMERY_GATE_BUILD_BOUND) — no test ran; a gate that hangs is a red gate, not a silent one" >&2
  elif [ "$rc" -ne 0 ]; then
    echo "GATE RED (exit $rc) in the build phase — no test ran" >&2
  fi
  return "$rc"
}
GPU_ARGS=()
if [ "${1:-}" = --test-locks ]; then
  [ $# -ge 4 ] && [ -d "$2" ] && [ "$3" = --gpu ] || exit 64
  GPU_ARGS=(--test-locks "$2")
  shift 2
fi
GPU=0
if [ "${1:-}" = --gpu ]; then
  GPU=1
  shift
fi
RUN=(cargo test)
if [ "${1:-}" = --oxide ]; then
  shift
  # shellcheck source=tools/arch.sh
  source "${BASH_SOURCE[0]%/*}/arch.sh"
  bloomery_arch gate.sh || exit $?
  RUN=(cargo oxide test --arch "$ARCH" --)
fi
# The libtest options that take a value. The one list: tools/recipes.py reads this line (by its
# `LIBTEST_VALUE=(` prefix, one line, one word per option) to tell a recipe's filters from values.
LIBTEST_VALUE=(--skip --test-threads --format --color --logfile -Z --shuffle-seed)
takes_value() {
  local o
  for o in "${LIBTEST_VALUE[@]}"; do
    [ "$1" = "$o" ] && return 0
  done
  return 1
}
FILTERS=()
LISTING=0
after=0
value=0
for a in "$@"; do
  if [ "$after" = 0 ]; then
    [ "$a" = -- ] && after=1
    continue
  fi
  if [ "$value" = 1 ]; then
    value=0
    continue
  fi
  if takes_value "$a"; then
    value=1
    continue
  fi
  case $a in
    --list) LISTING=1 ;;
    -*) ;;
    *) FILTERS+=("$a") ;;
  esac
done
ARGS=("$@")
CAPTURE=''
if [ "$GPU" = 1 ]; then
  BUILD_ARGS=()
  TEST_TARGET=0
  ARGS=()
  after=0
  for a in "$@"; do
    if [ "$after" = 1 ]; then
      ARGS+=("$a")
    elif [ "$a" = -- ]; then
      after=1
    else
      case "$a" in
        --lib | --bin | --bin=*) TEST_TARGET=1 ;;
        --target | --target=* | --doc)
          echo "gate.sh --gpu: only native tests are supported; '$a' is not allowed" >&2
          exit 64 ;;
      esac
      BUILD_ARGS+=("$a")
    fi
  done
  [ "$TEST_TARGET" = 1 ] || { echo "gate.sh --gpu: select a test with --lib or --bin" >&2; exit 64; }
  SELF=$(cd "${BASH_SOURCE[0]%/*}" && pwd -P)/gate.sh
  GPU_RUNNER=${SELF%/*}/gpu-gate.sh
  # Cargo merges runner arrays; a scalar override replaces a configured runner or refuses a
  # type conflict instead of concatenating commands. Its path has no whitespace; the wrapper quotes arguments.
  CAPTURE=$(mktemp -d /tmp/bloomery-gate.XXXXXX) || exit 70
  trap '[ -z "$CAPTURE" ] || rm -rf "$CAPTURE"' EXIT
  printf '#!/usr/bin/env bash\nexec bash %q --capture-test %q "$@"\n' "$SELF" "$CAPTURE" > "$CAPTURE/runner" || exit 70
  chmod 700 "$CAPTURE/runner" || exit 70
  HOST_INFO=$("${RUSTC:-rustc}" -vV) || exit $?
  HOST=''
  while read -r key value; do
    [ "$key" != host: ] || HOST=$value
  done <<< "$HOST_INFO"
  [ -n "$HOST" ] || { echo "gate.sh --gpu: rustc did not report its host target" >&2; exit 64; }
  RUNNER_CONFIG=$(python3 - "$HOST" "$CAPTURE/runner" <<'PY'
import json
import sys

host, runner = sys.argv[1:]
print("target." + json.dumps(host) + ".runner=" + json.dumps(runner))
PY
  ) || exit $?
  # Without --target cargo builds for the triple the call, the environment or a config names. One other than the host
  # runs the test through its own runner, or none, before the card lock: refused by name, from the source cargo
  # reads first (--config, CARGO_BUILD_TARGET, then the config files from the working directory up to $CARGO_HOME).
  python3 - "$HOST" ${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"} <<'PY' || exit $?
import os
import sys
import tomllib
from pathlib import Path

host, *args = sys.argv[1:]


def configured():
    for i, arg in enumerate(args):
        if arg.startswith("--config="):
            setting = arg[len("--config="):]
        elif arg == "--config" and i + 1 < len(args):
            setting = args[i + 1]
        else:
            continue
        key, _, value = setting.partition("=")
        if key.strip() == "build.target":
            try:
                value = tomllib.loads("v = " + value)["v"]
            except tomllib.TOMLDecodeError:
                value = value.strip()
            return "--config build.target", value
    if os.environ.get("CARGO_BUILD_TARGET"):
        return "CARGO_BUILD_TARGET", os.environ["CARGO_BUILD_TARGET"]
    here = Path.cwd()
    for directory in [d / ".cargo" for d in (here, *here.parents)] + [Path(os.environ.get("CARGO_HOME") or Path.home() / ".cargo")]:
        for name in ("config.toml", "config"):
            path = directory / name
            if not path.is_file():
                continue
            try:
                with path.open("rb") as stream:
                    build = tomllib.load(stream).get("build", {})
            except (OSError, tomllib.TOMLDecodeError) as err:
                print(f"gate.sh --gpu: cannot read {path}: {err}", file=sys.stderr)
                sys.exit(64)
            if "target" in build:
                return f"[build] target of {path}", build["target"]
            break
    return None


found = configured()
if found:
    where, value = found
    for triple in value if isinstance(value, list) else [value]:
        if triple != host:
            print(f"gate.sh --gpu: {where} names the target {triple}, not the host {host}: its runner, or none, would run the "
                  "test before the card lock; unset it", file=sys.stderr)
            sys.exit(64)
PY
  # The CLI runner config for the host triple outranks a runner from the environment or a config file: cargo's call
  # only records the test, and the replay below takes the card lock.
  build_bounded "${RUN[@]}" ${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"} --config "$RUNNER_CONFIG" -- ${ARGS[@]+"${ARGS[@]}"} || exit $?
  [ -z "$BIN_SIGNAL" ] || exit 143
  [ "$BUILD_ONLY" = 0 ] || exit 0 # the capture runner only recorded the test; the replay takes the card lock
  RUN=(python3 -c '
import json
import os
from pathlib import Path
import sys

directory, runner, *runner_args = sys.argv[1:]
captures = list(Path(directory).glob("run-*.json"))
if len(captures) != 1:
    print("gate.sh --gpu: select exactly one test target; got " + str(len(captures)), file=sys.stderr)
    sys.exit(64)
with captures[0].open(encoding="utf-8") as stream:
    context = json.load(stream)
os.chdir(context["cwd"])
os.execvpe("bash", ["bash", runner, *runner_args, "--executable", *context["argv"]], context["env"])
' "$CAPTURE" "$GPU_RUNNER" ${GPU_ARGS[@]+"${GPU_ARGS[@]}"})
  ARGS=()
fi
# The build phase of a call without --gpu (the --gpu build is the capture call above): its own cargo call under its own bound,
# cargo's arguments up to the first `--` and --no-run; the libtest arguments after it are the run's. It runs after a filter's
# FIFO and tee are up (a setup that fails ends 70 before any cargo call), its output straight to this runner's own. A --doc call
# has none: cargo refuses --no-run with --doc, and a doctest call compiles the library inside its run, under the run bound.
build_phase() {
  [ "$GPU" = 0 ] || return 0
  local a build=()
  for a in "$@"; do
    [ "$a" != -- ] || break
    if [ "$a" = --doc ]; then
      [ "$BUILD_ONLY" = 0 ] || echo "gate.sh: BLOOMERY_BUILD_ONLY: a --doc call has no build phase (its doctests compile inside the run): nothing built" >&2
      return 0
    fi
    build+=("$a")
  done
  build_bounded "${RUN[@]}" ${build[@]+"${build[@]}"} --no-run || exit $?
  [ -z "$BIN_SIGNAL" ] || exit 143
}
run_tests() {
  if [ "$GPU" = 1 ]; then
    # gpu-gate.sh owns the runtime bound, after its lock waits.
    run_bounded "${RUN[@]}" ${ARGS[@]+"${ARGS[@]}"}
  else
    run_bounded timeout --kill-after=10 "$BOUND" "${RUN[@]}" ${ARGS[@]+"${ARGS[@]}"}
  fi
}
if [ "${#FILTERS[@]}" = 0 ] || [ "$LISTING" = 1 ]; then
  build_phase "$@"
  [ "$BUILD_ONLY" = 0 ] || exit 0
  run_tests
  rc=$?
else
  LOG=$(mktemp)
  TEE_PID='' TEE_EXIT=''
  # The reader may still be Bash opening the FIFO, before tee has its default signal handlers.
  trap '[ -z "$TEE_PID" ] || {
    trap "" TERM INT HUP
    kill -KILL "$TEE_PID" 2> /dev/null || :
    wait "$TEE_PID" 2> /dev/null || :
  }; rm -f "$LOG" "$LOG.out"; [ -z "$CAPTURE" ] || rm -rf "$CAPTURE"' EXIT
  # cargo's output reaches tee through a FIFO, not a pipe: in a pipeline run_bounded would run in a subshell that a TERM to this
  # runner does not reach. tee starts before run_bounded sets its trap, so until then a TERM, INT or HUP ends this runner and tee
  # with it (one that comes before tee's pid is known, as soon as it is): a tee left waiting to open the FIFO would hold this
  # runner's output open. run_bounded puts this trap back when it returns.
  for s in TERM:143 INT:130 HUP:129; do
    # shellcheck disable=SC2064 # the code now, TEE_PID when the trap runs
    trap "TEE_EXIT=${s#*:}; [ -z \"\$TEE_PID\" ] || exit ${s#*:}" "${s%:*}"
  done
  mkfifo "$LOG.out" || exit 70
  tee "$LOG" < "$LOG.out" &
  TEE_PID=$!
  [ -z "$TEE_EXIT" ] || exit "$TEE_EXIT"
  build_phase "$@"
  [ "$BUILD_ONLY" = 0 ] || exit 0 # the EXIT trap above ends tee and removes the log
  run_tests > "$LOG.out"
  rc=$?
  wait "$TEE_PID" || { tee_rc=$?; [ "$rc" -ne 0 ] || rc=$tee_rc; }
  TEE_PID=''
  if [ "$rc" -eq 0 ]; then
    passed=$(awk '/^test result: / { for (i = 2; i <= NF; i++) if ($i == "passed;") n += $(i - 1) } END { print n + 0 }' "$LOG")
    if [ "$passed" -eq 0 ]; then
      # tools/recipes.py's self-test reads the filters back from this line: keep "the filter … matched no test".
      echo "GATE RED (exit 78): the filter ${FILTERS[*]} matched no test — 0 passed over the whole call" >&2
      exit 78
    fi
  fi
fi
# 124 = timeout이 TERM으로 끝냄, 137 = TERM을 무시해 --kill-after의 KILL로 끝냄.
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
  echo "GATE TIMED OUT after the ${BOUND}s bound (exit $rc) — a gate that hangs is a red gate, not a silent one" >&2
elif [ "$rc" -ne 0 ]; then
  echo "GATE RED (exit $rc)" >&2
fi
exit "$rc"
