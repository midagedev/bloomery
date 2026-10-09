#!/usr/bin/env bash
# shellcheck shell=bash
# The default paths of the reference model, the data directory and the ik tree, for the build
# scripts (through ref-build-common.sh) and the runners alike. Sourced, never executed; it only
# defines variables, and exports none: a script whose children read one exports it itself.
#
# One place so that a harness built against one ik tree is not timed against another's
# llama-bench, and no runner reads a model or a data directory the harnesses and gates do not.
# Kept apart from ref-build-common.sh because that file is build-only (the caller sets -e first,
# it defines compiler flags) and the runners never compile.
#
# MODEL and BLOOMERY_DATA come from the variables the C++ harnesses (ref_paths.h) and the Rust
# gates (bloomery_gpu_gates::{ref_model_path, data_dir}) read, so one override reaches all three:
# BLOOMERY_REF_MODEL moves the model, BLOOMERY_DATA the data directory. ref_paths.h and data_dir
# carry the same defaults; ref_model_path has none, and tools/box.sh exports this file's MODEL as
# BLOOMERY_REF_MODEL into every box command. An empty variable counts as unset, as in
# ref_paths.h. BLOOMERY_REF_MODEL itself is left as the caller set it; MODEL is the name the
# scripts use.
#
# IK and IKBIN each keep their override: IK moves the whole tree (and IKBIN with it), IKBIN
# alone points at a different llama-bench.
#
# What is a property of the model — the model file, the ik tree and flags, the prompt set, the
# reference context, the oracle directories — lives in models/<architecture>.sh and BLOOMERY_MODEL
# picks one. The name is the GGUF general.architecture value. What is a property of the machine
# — BLOOMERY_DATA, and IKBIN's place inside a tree — stays here, shared by every model.
#
# Inside a tools/box.sh command the profile is already picked: box.sh resolves BLOOMERY_REF_MODEL
# under one profile and exports that profile's name as BLOOMERY_REF_MODEL_PROFILE. It is the default
# here, and a script that picks another profile is refused, because the export would not follow it:
# the second profile's set name, tokens and lease would be applied to the first profile's file.
#
# The tier. BLOOMERY_TIER is `real` (unset is the same: every name here is as the profile defines it) or `fixture`: the
# model file is the family's fixture — the small file `fixture generate` writes (crates/model/src/bin/fixture.rs), whose header
# names the card budget that forces the plan to offload (`fixture_budget`, below; tools/box.sh exports it as BLOOMERY_CARD_BUDGET).
# What the fixture tier does is the family's, one row of the table below:
#   a fixture directory    the first shard `*-00001-of-*.gguf` of the one directory under $BLOOMERY_FIXTURE_ROOT
#                          (default /models/fixtures), one a family; MODEL becomes it unless BLOOMERY_REF_MODEL names a file
#                          already (a caller's own wins), and FIXTURE_FILE names it either way. Not one file there: 66.
#   self                   the family has no fixture file: FIXTURE_FILE stays empty and a command runs on the profile's file, and
#                          one that opens no model runs as in the real tier; the gate fence (Tier::check_file, crates/gpu-gates/src/tier.rs)
#                          refuses that file as not a fixture, so tools/gate-batch.sh defers, as real-file, a recipe whose targets reach it.
#   none                   the family has no fixture yet — and so does a profile the table does not name: 66, naming the
#                          family. Never a run on the real file.
# Exit codes: 64 a BLOOMERY_TIER that is neither, 65 a fixture whose header is not a whole fixture (`fixture_budget`), 66 no
# fixture for the family.
#
# SC2034: the sourcing script reads these, which shellcheck does not see in this file alone.
# shellcheck disable=SC2034
: "${BLOOMERY_MODEL:=${BLOOMERY_REF_MODEL_PROFILE:-deepseek2}}"
if [ -n "${BLOOMERY_REF_MODEL_PROFILE:-}" ] && [ "$BLOOMERY_MODEL" != "$BLOOMERY_REF_MODEL_PROFILE" ]; then
  echo "ref-paths.sh: this command picks the profile '$BLOOMERY_MODEL', but tools/box.sh resolved" >&2
  echo "  BLOOMERY_REF_MODEL under '$BLOOMERY_REF_MODEL_PROFILE'; pick it on the Mac side instead:" >&2
  echo "  BLOOMERY_MODEL=$BLOOMERY_MODEL tools/box.sh '...'" >&2
  exit 64
fi
__ref_paths_profile="${BASH_SOURCE[0]%/*}/models/$BLOOMERY_MODEL.sh"
if [ ! -f "$__ref_paths_profile" ]; then
  echo "ref-paths.sh: no model profile for BLOOMERY_MODEL='$BLOOMERY_MODEL'" >&2
  echo "  expected $__ref_paths_profile" >&2
  exit 64
fi
# shellcheck source=tools/ref/models/deepseek2.sh
source "$__ref_paths_profile"
unset __ref_paths_profile
IKBIN=${IKBIN:-$IK/build/bin/llama-bench}
: "${BLOOMERY_DATA:=/root/bloomery-data}"

REF_PATHS_DIR=${BASH_SOURCE[0]%/*}
FIXTURE_FILE=
case "${BLOOMERY_TIER:-real}" in
  real) ;;
  fixture)
    case "$BLOOMERY_MODEL" in
      qwen4exp) __fixture_dir=qwen38 ;;
      deepseek41) __fixture_dir=v41 ;;
      glm5next) __fixture_dir=glm5next ;;
      deepseek2 | qwen3moe | qwen35moe | qwen35) __fixture_dir=self ;;
      *) __fixture_dir=none ;;
    esac
    case "$__fixture_dir" in
      self) ;;
      none)
        echo "ref-paths.sh: BLOOMERY_TIER=fixture, but the family '$BLOOMERY_MODEL' has no fixture yet (the table in $REF_PATHS_DIR/ref-paths.sh): not running it on its real file (exit 66)" >&2
        exit 66
        ;;
      *)
        __fixture_root=${BLOOMERY_FIXTURE_ROOT:-/models/fixtures}
        __fixture_files=("$__fixture_root/$__fixture_dir"/*-00001-of-*.gguf)
        if [ ! -e "${__fixture_files[0]}" ]; then
          echo "ref-paths.sh: BLOOMERY_TIER=fixture: the family '$BLOOMERY_MODEL' has no fixture file, no $__fixture_root/$__fixture_dir/*-00001-of-*.gguf (\`fixture generate\` writes it); not running it on its real file (exit 66)" >&2
          exit 66
        fi
        if [ "${#__fixture_files[@]}" != 1 ]; then
          echo "ref-paths.sh: BLOOMERY_TIER=fixture: $__fixture_root/$__fixture_dir holds ${#__fixture_files[@]} first shards (${__fixture_files[*]}), not one (exit 66)" >&2
          exit 66
        fi
        FIXTURE_FILE=${__fixture_files[0]}
        [ -n "${BLOOMERY_REF_MODEL:-}" ] || MODEL=$FIXTURE_FILE
        # A family whose fixture has a DSpark draft keeps it in a directory of its own beside the target's shards (the glob above sees
        # the target's first shard only): the fixture tier's draft is that file, never the real draft the profile names. A caller's own
        # BLOOMERY_DSPARK_MODEL wins, as it does in the profile (the Rust side refuses a draft that is not a whole fixture).
        if [ "$BLOOMERY_MODEL" = deepseek41 ] && [ -z "${BLOOMERY_DSPARK_MODEL:-}" ]; then
          __fixture_drafts=("$__fixture_root/$__fixture_dir"/draft/*.gguf)
          if [ ! -e "${__fixture_drafts[0]}" ] || [ "${#__fixture_drafts[@]}" != 1 ]; then
            echo "ref-paths.sh: BLOOMERY_TIER=fixture: $__fixture_root/$__fixture_dir/draft holds ${#__fixture_drafts[@]} files (${__fixture_drafts[*]}), want the one DSpark fixture draft (\`fixture generate\` writes it); not running a draft on the real file (exit 66)" >&2
            exit 66
          fi
          DSPARK_MODEL=${__fixture_drafts[0]}
        fi
        ;;
    esac
    unset __fixture_dir __fixture_root __fixture_files __fixture_drafts
    ;;
  *)
    echo "ref-paths.sh: BLOOMERY_TIER is real (unset: the same) or fixture, got '${BLOOMERY_TIER}'" >&2
    exit 64
    ;;
esac

# fixture_budget: what a command under the fixture tier needs of the fixture's header, once per command (tools/box.sh calls it):
# the card budget the generator recorded (bloomery.fixture.card_budget, bytes) on stdout, nothing for a family whose real file
# stands (FIXTURE_FILE empty). The header is read through tools/ref/gguf-ranges.py's reader; a file that is not a whole fixture
# — no bloomery.fixture.version key, a bloomery.fixture.subset key (it holds some of the planned tensors only), no
# positive integer card budget, or no readable header — is a named 65 (or the reader's own code), never a plan without a budget.
fixture_budget() {
  [ -n "$FIXTURE_FILE" ] || return 0
  python3 -I - "$REF_PATHS_DIR/gguf-ranges.py" "$FIXTURE_FILE" << 'PY'
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("gguf_ranges", sys.argv[1])
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
path = sys.argv[2]


def refuse(code, why):
    print(f"ref-paths.sh: BLOOMERY_TIER=fixture: {why} (exit {code})", file=sys.stderr)
    sys.exit(code)


try:
    meta = reader.header(path)[0]
except reader.Refusal as e:
    refuse(e.code, str(e))
if "bloomery.fixture.version" not in meta:
    refuse(65, f"{path} has no bloomery.fixture.version key: it is not a fixture file")
if "bloomery.fixture.subset" in meta:
    refuse(65, f"{path} carries bloomery.fixture.subset: it holds only some of the planned tensors, and the engine must not run it")
budget = meta.get("bloomery.fixture.card_budget")
if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
    refuse(65, f"{path} records no positive bloomery.fixture.card_budget (read {budget!r})")
print(budget)
PY
}

# fixture_dump_tier <script>: what a dump script (dump.sh, dump-mtp.sh) takes of the tier, before it names a set. Sets
# SET_PREFIX: `fx_` under the fixture tier, where a set is the fixture file's and carries its real twin's name behind that
# prefix (the fixture families of crates/refset name them), and nothing in the real tier. In the fixture tier it refuses, by
# name: a family whose real file stands (no fixture file, 66), and a model that is not the fixture first shard (64) — a caller's
# own BLOOMERY_REF_MODEL wins over the fixture in ref-paths.sh above, and an `fx_` set of another file would be a wrong answer
# under a right name. The model string is the one refset's `fixture::first_shard` returns, so the dumper's `# model` line is
# the string the fixture family checks.
fixture_dump_tier() {
  SET_PREFIX=
  [ "${BLOOMERY_TIER:-real}" = fixture ] || return 0
  if [ -z "$FIXTURE_FILE" ]; then
    echo "$1: BLOOMERY_TIER=fixture, but the $MODEL_NAME profile's real file stands in the fixture tier (it has no fixture file of its own), so there is no fixture set to dump; run it in the real tier (exit 66)" >&2
    exit 66
  fi
  if [ "$MODEL" != "$FIXTURE_FILE" ]; then
    echo "$1: BLOOMERY_TIER=fixture, but the model is $MODEL, not the fixture $FIXTURE_FILE (a BLOOMERY_REF_MODEL of the caller's?): an fx_ set is the fixture file's, never another file's (exit 64)" >&2
    exit 64
  fi
  SET_PREFIX=fx_
}

# fixture_line <script>: the `# fixture` header line of the fixture file, on stdout; nothing in the real tier. It is what
# `refset-check --fixture-line` prints (crates/refset/src/bin/refset_check.rs), the one owner of the line, and the dumpers write
# it verbatim beside `# model`. The binary is built the way `just refset-check` builds it; cargo is the staleness rule, so a
# binary older than its sources is rebuilt here, never run.
fixture_line() {
  [ "${BLOOMERY_TIER:-real}" = fixture ] || return 0
  local root line
  root=$(cd "$REF_PATHS_DIR/../.." && pwd)
  (cd "$root" && cargo build --release -p bloomery-refset --bin refset-check >&2) || {
    echo "$1: cargo build of refset-check failed (exit 65)" >&2
    exit 65
  }
  line=$("$root/target/release/refset-check" --fixture-line "$FIXTURE_FILE") || {
    echo "$1: refset-check --fixture-line refused $FIXTURE_FILE (exit 65)" >&2
    exit 65
  }
  case $line in
    "# fixture"$'\t'?*) ;;
    *) echo "$1: refset-check --fixture-line printed '$line', not a '# fixture<TAB>key=value ...' line (exit 65)" >&2; exit 65 ;;
  esac
  case $line in
    *$'\n'*) echo "$1: refset-check --fixture-line printed more than one line (exit 65)" >&2; exit 65 ;;
  esac
  printf '%s\n' "$line"
}
