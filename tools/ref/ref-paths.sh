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
# model is the family's fixture — the small file `fixture generate` writes (crates/model-hf/src/bin/fixture.rs), or the small
# checkpoint folder `fixture cut` writes (crates/model-hf/src/cut.rs), whose header names the card budget that forces the plan
# to offload (`fixture_budget`, below; tools/box.sh exports it as BLOOMERY_CARD_BUDGET).
# What the fixture tier does is the family's, one row of tools/fixture-dirs.tsv:
#   a fixture directory    the one directory under $BLOOMERY_FIXTURE_ROOT (default /models/fixtures) a family has. It holds GGUF
#                          shards, the first `*-00001-of-*.gguf`, or, with a model.safetensors.index.json beside them, a checkpoint
#                          folder, the first shard `*-00001-of-*.safetensors`; both kinds, or not one first shard there: 66.
#                          FIXTURE_FILE names the first shard either way (`fixture_line` and `fixture_budget` read it), and
#                          FIXTURE_MODEL what the model is: the first shard of a GGUF fixture, the folder of a cut. MODEL becomes
#                          FIXTURE_MODEL unless BLOOMERY_REF_MODEL names a model already (a caller's own wins).
#   self                   the family has no fixture file: FIXTURE_FILE stays empty and a command runs on the profile's file, and
#                          one that opens no model runs as in the real tier; the gate fence (Tier::check_file, crates/gpu-gates/src/tier.rs)
#                          refuses that file as not a fixture, so tools/gate-batch.sh defers, as real-file, a recipe whose targets reach it.
#   none                   the family has no fixture yet — and so does a profile the table does not name: 66, naming the
#                          family. Never a run on the real file.
# Exit codes: 64 a BLOOMERY_TIER that is neither, 65 a fixture whose header is not a whole fixture (`fixture_budget`), 66 no
# fixture for the family.
#
# A fixture variant. BLOOMERY_FIXTURE_VARIANT=<tag> moves the fixture root to fixture-variants/<tag> beside it (the box manifest keys
# every file under the root into every fixture-tier item, so variants never sit under it): the family's fixture requantized
# to the type map of the tag's row in tools/fixture-variants.tsv (`just fixture-requant`), in a directory of the same name. The tag is
# lower-case letters and digits and a row of the table whose family is this profile; the real tier has no variant, and a tag set there
# is a named 64, as is one the table does not name. refset's fixture::root is the Rust twin of this line, and a variant's sets carry
# the tag in their names (SET_PREFIX, below). A script that writes the variant file sets FIXTURE_NEW=1 before sourcing: the file is
# then not looked for, and FIXTURE_DIR is the directory it goes in (set whenever the family has a fixture directory).
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
FIXTURE_DIR=
FIXTURE_MODEL=
FIXTURE_VARIANT=${BLOOMERY_FIXTURE_VARIANT:-}
case $FIXTURE_VARIANT in
  *[!a-z0-9]*)
    echo "ref-paths.sh: BLOOMERY_FIXTURE_VARIANT is a tag of tools/fixture-variants.tsv, lower-case letters and digits, got '$FIXTURE_VARIANT' (exit 64)" >&2
    exit 64
    ;;
esac
case "${BLOOMERY_TIER:-real}" in
  real)
    if [ -n "$FIXTURE_VARIANT" ]; then
      echo "ref-paths.sh: BLOOMERY_FIXTURE_VARIANT=$FIXTURE_VARIANT, but the tier is real: a variant is a requantized fixture file, and nothing runs on a real file under a variant's name (exit 64)" >&2
      exit 64
    fi
    ;;
  fixture)
    # The family's row of tools/fixture-dirs.tsv: a directory, `self`, or no row at all (`none`).
    if [ ! -f "$REF_PATHS_DIR/../fixture-dirs.tsv" ]; then
      echo "ref-paths.sh: BLOOMERY_TIER=fixture, but $REF_PATHS_DIR/../fixture-dirs.tsv, the table of the fixture directories, is not there (exit 66)" >&2
      exit 66
    fi
    __fixture_dir=$(awk -F'\t' -v a="$BLOOMERY_MODEL" '$0 !~ /^#/ && $1 == a { print $2; exit }' "$REF_PATHS_DIR/../fixture-dirs.tsv")
    : "${__fixture_dir:=none}"
    case "$__fixture_dir" in
      self)
        if [ -n "$FIXTURE_VARIANT" ]; then
          echo "ref-paths.sh: BLOOMERY_FIXTURE_VARIANT=$FIXTURE_VARIANT, but the family '$BLOOMERY_MODEL' has no fixture file to be a variant of (exit 64)" >&2
          exit 64
        fi
        ;;
      none)
        echo "ref-paths.sh: BLOOMERY_TIER=fixture, but the family '$BLOOMERY_MODEL' has no fixture yet (no row in $REF_PATHS_DIR/../fixture-dirs.tsv): not running it on its real file (exit 66)" >&2
        exit 66
        ;;
      *)
        __fixture_root=${BLOOMERY_FIXTURE_ROOT:-/models/fixtures}
        if [ -n "$FIXTURE_VARIANT" ]; then
          __variant_family=$(awk -F'\t' -v t="$FIXTURE_VARIANT" '$0 !~ /^#/ && $1 == t { print $2 }' "$REF_PATHS_DIR/../fixture-variants.tsv")
          if [ -z "$__variant_family" ]; then
            echo "ref-paths.sh: BLOOMERY_FIXTURE_VARIANT=$FIXTURE_VARIANT, which tools/fixture-variants.tsv does not name (it holds: $(awk -F'\t' '$0 !~ /^#/ && NF { printf "%s ", $1 }' "$REF_PATHS_DIR/../fixture-variants.tsv")) (exit 64)" >&2
            exit 64
          fi
          if [ "$__variant_family" != "$BLOOMERY_MODEL" ]; then
            echo "ref-paths.sh: BLOOMERY_FIXTURE_VARIANT=$FIXTURE_VARIANT is a variant of $__variant_family, and this command picked the '$BLOOMERY_MODEL' profile (exit 64)" >&2
            exit 64
          fi
          __fixture_root=${__fixture_root%/*}/fixture-variants/$FIXTURE_VARIANT
        fi
        FIXTURE_DIR=$__fixture_root/$__fixture_dir
        if [ -z "${FIXTURE_NEW:-}" ]; then
          # A directory that holds a checkpoint folder's index is a folder fixture (`fixture cut`), its first shard the
          # safetensors one and the model the folder; any other holds GGUF shards (`fixture generate`), the model the first
          # shard. refset's fixture::first_shard_in is this rule's twin, down to a directory that holds both kinds.
          __fixture_ext=gguf
          __fixture_verb=generate
          __fixture_files=("$__fixture_root/$__fixture_dir"/*-00001-of-*.gguf)
          if [ -f "$FIXTURE_DIR/model.safetensors.index.json" ]; then
            if [ -e "${__fixture_files[0]}" ]; then
              echo "ref-paths.sh: BLOOMERY_TIER=fixture: $FIXTURE_DIR holds both a checkpoint folder's index and GGUF first shards (${__fixture_files[*]}), which are two fixtures in one directory (exit 66)" >&2
              exit 66
            fi
            __fixture_ext=safetensors
            __fixture_verb="cut"
            __fixture_files=("$FIXTURE_DIR"/*-00001-of-*.safetensors)
          fi
          if [ ! -e "${__fixture_files[0]}" ]; then
            echo "ref-paths.sh: BLOOMERY_TIER=fixture: the family '$BLOOMERY_MODEL' has no fixture file, no $FIXTURE_DIR/*-00001-of-*.$__fixture_ext (\`fixture $__fixture_verb\` writes it${FIXTURE_VARIANT:+, \`just fixture-requant $BLOOMERY_MODEL $FIXTURE_VARIANT\` writes the variant file}); not running it on its real file (exit 66)" >&2
            exit 66
          fi
          if [ "${#__fixture_files[@]}" != 1 ]; then
            echo "ref-paths.sh: BLOOMERY_TIER=fixture: $FIXTURE_DIR holds ${#__fixture_files[@]} first shards (${__fixture_files[*]}), not one (exit 66)" >&2
            exit 66
          fi
          FIXTURE_FILE=${__fixture_files[0]}
          if [ "$__fixture_ext" = safetensors ]; then FIXTURE_MODEL=$FIXTURE_DIR; else FIXTURE_MODEL=$FIXTURE_FILE; fi
          [ -n "${BLOOMERY_REF_MODEL:-}" ] || MODEL=$FIXTURE_MODEL
        fi
        # A family whose fixture has a DSpark draft keeps it in a directory of its own beside the target's shards (the glob above sees
        # the target's first shard only): the fixture tier's draft is that file, never the real draft the profile names. A caller's own
        # BLOOMERY_DSPARK_MODEL wins, as it does in the profile (the Rust side refuses a draft that is not a whole fixture). Zero files
        # or several do not exit here — MODEL, FIXTURE_FILE and FIXTURE_DIR stay as resolved above, DSPARK_MODEL is empty (the profile's
        # real path is not left in it), and the reason is kept for dspark_model, which is what refuses. Exactly one file: DSPARK_MODEL is
        # it. FIXTURE_NEW=1 skips the target-file lookup above and still comes through here, so a writer does not keep the real draft path.
        if [ "$BLOOMERY_MODEL" = deepseek41 ] && [ -z "${BLOOMERY_DSPARK_MODEL:-}" ]; then
          DSPARK_MODEL=
          DSPARK_MODEL_WHY=
          __fixture_drafts=("$__fixture_root/$__fixture_dir"/draft/*.gguf)
          # A variant's map retypes the target's tensors and never the draft, so a variant directory with no draft of its own answers
          # the fixture's one; a draft of its own that differs from the fixture's is refused (`cmp`), by name.
          __fixture_base=()
          if [ -n "$FIXTURE_VARIANT" ]; then
            __fixture_base=("${BLOOMERY_FIXTURE_ROOT:-/models/fixtures}/$__fixture_dir"/draft/*.gguf)
            if [ ! -e "${__fixture_drafts[0]}" ] && [ -e "${__fixture_base[0]}" ] && [ "${#__fixture_base[@]}" = 1 ]; then
              __fixture_drafts=("${__fixture_base[@]}")
            elif [ -e "${__fixture_drafts[0]}" ] && [ "${#__fixture_drafts[@]}" = 1 ] && [ -e "${__fixture_base[0]}" ] &&
              [ "${#__fixture_base[@]}" = 1 ] && ! cmp -s "${__fixture_drafts[0]}" "${__fixture_base[0]}"; then
              DSPARK_MODEL_WHY="ref-paths.sh: BLOOMERY_TIER=fixture: the variant $FIXTURE_VARIANT holds the draft ${__fixture_drafts[0]}, which differs from the fixture's ${__fixture_base[0]}; a variant's map never retypes the draft (exit 66)"
            fi
          fi
          if [ -n "$DSPARK_MODEL_WHY" ]; then
            :
          elif [ -e "${__fixture_drafts[0]}" ] && [ "${#__fixture_drafts[@]}" = 1 ]; then
            DSPARK_MODEL=${__fixture_drafts[0]}
          else
            DSPARK_MODEL_WHY="ref-paths.sh: BLOOMERY_TIER=fixture: $__fixture_root/$__fixture_dir/draft holds ${#__fixture_drafts[@]} files (${__fixture_drafts[*]}), want the one DSpark fixture draft (\`fixture generate\` writes it); not running a draft on the real file (exit 66)"
            if [ -n "$FIXTURE_VARIANT" ]; then
              DSPARK_MODEL_WHY="${DSPARK_MODEL_WHY%(exit 66)}and the fixture's own draft directory holds ${#__fixture_base[@]} files (${__fixture_base[*]}) (exit 66)"
            fi
          fi
        fi
        ;;
    esac
    unset __fixture_dir __fixture_root __fixture_files __fixture_drafts __fixture_base __fixture_ext __fixture_verb __variant_family
    ;;
  *)
    echo "ref-paths.sh: BLOOMERY_TIER is real (unset: the same) or fixture, got '${BLOOMERY_TIER}'" >&2
    exit 64
    ;;
esac

# dspark_model: the draft a command may export, on stdout, no trailing newline (the bytes the old
# `printf %s` of DSPARK_MODEL produced). Non-empty — the real tier's profile path, a caller's
# BLOOMERY_DSPARK_MODEL, or the one fixture draft — prints it and returns 0. Empty: print the reason
# kept above (or a named line when this profile has none) to stderr, ending with "(exit 66)", and
# exit 66. `$(. tools/ref/ref-paths.sh && dspark_model)` then ends the command's `&&` chain at 66,
# which tools/gate-batch.sh counts as deferred. It never prints an empty path.
# shellcheck source=tools/ref/dspark-path.sh
source "${BASH_SOURCE[0]%/*}/dspark-path.sh"
dspark_model() {
  dspark_path "${DSPARK_MODEL:-}" "${DSPARK_MODEL_WHY:-}"
}

# fixture_budget: what a command under the fixture tier needs of the fixture's header, once per command (tools/box.sh calls it):
# the card budget the generator recorded (bloomery.fixture.card_budget, bytes) on stdout, nothing for a family whose real file
# stands (FIXTURE_FILE empty). A GGUF header is read through tools/ref/gguf-ranges.py's reader, a cut folder's first shard through
# its safetensors `__metadata__` (every bloomery.fixture.* value a string, the budget decimal digits); a file that is not a whole
# fixture — no bloomery.fixture.version key, a bloomery.fixture.subset key (it holds some of the planned tensors only), no
# positive integer card budget, or no readable header — is a named 65 (or the reader's own code), never a plan without a budget.
fixture_budget() {
  [ -n "$FIXTURE_FILE" ] || return 0
  python3 -I - "$REF_PATHS_DIR/gguf-ranges.py" "$FIXTURE_FILE" << 'PY'
import importlib.util
import json
import sys

spec = importlib.util.spec_from_file_location("gguf_ranges", sys.argv[1])
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
path = sys.argv[2]
MAX_HEADER = 64 << 20  # refset::fixture's bound: a longer length word is a corrupt file


def refuse(code, why):
    print(f"ref-paths.sh: BLOOMERY_TIER=fixture: {why} (exit {code})", file=sys.stderr)
    sys.exit(code)


def safetensors_meta(path):
    """The `__metadata__` of a safetensors shard, the fixture keys of a folder fixture's first shard."""
    try:
        with open(path, "rb") as fh:
            word = fh.read(8)
            if len(word) != 8:
                refuse(65, f"{path} is not a readable safetensors header: no length word")
            n = int.from_bytes(word, "little")
            if n > MAX_HEADER:
                refuse(65, f"{path} is not a readable safetensors header: it claims {n} bytes, over {MAX_HEADER}")
            text = fh.read(n)
    except OSError as e:
        refuse(65, f"{path} is not a readable safetensors header: {e}")
    if len(text) != n:
        refuse(65, f"{path} is not a readable safetensors header: {len(text)} of {n} bytes")
    try:
        header = json.loads(text)
    except ValueError as e:
        refuse(65, f"{path} is not a readable safetensors header: {e}")
    if not isinstance(header, dict) or not isinstance(header.get("__metadata__"), dict):
        refuse(65, f"{path} is not a readable safetensors header: no __metadata__ object")
    for k, v in header["__metadata__"].items():
        if k.startswith("bloomery.fixture.") and not isinstance(v, str):
            refuse(65, f"{path}: {k}: {v!r} is not a string, which a safetensors metadata value is")
    return header["__metadata__"]


folder = path.endswith(".safetensors")
if folder:
    meta = safetensors_meta(path)
else:
    try:
        meta = reader.header(path)[0]
    except reader.Refusal as e:
        refuse(e.code, str(e))
if "bloomery.fixture.version" not in meta:
    refuse(65, f"{path} has no bloomery.fixture.version key: it is not a fixture file")
if "bloomery.fixture.subset" in meta:
    refuse(65, f"{path} carries bloomery.fixture.subset: it holds only some of the planned tensors, and the engine must not run it")
budget = meta.get("bloomery.fixture.card_budget")
if folder and isinstance(budget, str) and budget.isascii() and budget.isdecimal():
    budget = int(budget)  # a safetensors metadata value is text
if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
    refuse(65, f"{path} records no positive bloomery.fixture.card_budget (read {budget!r})")
print(budget)
PY
}

# fixture_dump_tier <script>: what a dump script (dump.sh, dump-mtp.sh) takes of the tier, before it names a set. Sets
# SET_PREFIX: `fx_` under the fixture tier, where a set is the fixture file's and carries its real twin's name behind that
# prefix (the fixture families of crates/refset name them), `fx_<tag>_` for a variant's (refset's fixture::variant_set_name is
# its twin), and nothing in the real tier. In the fixture tier it refuses, by
# name: a family whose real file stands (no fixture file, 66), and a model that is not the fixture (FIXTURE_MODEL, 64) — a caller's
# own BLOOMERY_REF_MODEL wins over the fixture in ref-paths.sh above, and an `fx_` set of another file would be a wrong answer
# under a right name. A GGUF fixture's model string is the one refset's `fixture::first_shard` returns, so the dumper's `# model`
# line is the string the fixture family checks; a cut folder's dumper writes that line from the folder's first shard itself.
fixture_dump_tier() {
  SET_PREFIX=
  [ "${BLOOMERY_TIER:-real}" = fixture ] || return 0
  if [ -z "$FIXTURE_FILE" ]; then
    echo "$1: BLOOMERY_TIER=fixture, but the $MODEL_NAME profile's real file stands in the fixture tier (it has no fixture file of its own), so there is no fixture set to dump; run it in the real tier (exit 66)" >&2
    exit 66
  fi
  if [ "$MODEL" != "$FIXTURE_MODEL" ]; then
    echo "$1: BLOOMERY_TIER=fixture, but the model is $MODEL, not the fixture $FIXTURE_MODEL (a BLOOMERY_REF_MODEL of the caller's?): an fx_ set is the fixture file's, never another file's (exit 64)" >&2
    exit 64
  fi
  SET_PREFIX=fx_${FIXTURE_VARIANT:+${FIXTURE_VARIANT}_}
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

# A dump runner's EXIT trap: the set staged at $STAGE is removed unless its manifest carries the
# `# complete` trailer, so a run that dies leaves no half-written staging set.
drop_incomplete_stage() {
  [ -d "$STAGE" ] || return 0
  grep -qs '^# complete' "$STAGE/MANIFEST.tsv" && return 0
  rm -rf "$STAGE"
  echo "[staging] removed the incomplete $STAGE" >&2
}

# file_md5 <file>: the file's md5 (md5sum, the box's), or md5sum's failure as the status with nothing printed. A manifest
# writer takes it into a variable before the line that prints it: inside a printf's arguments a failed read would print an
# empty field, and set -e would let the writer go on to its `# complete` trailer.
file_md5() {
  local sum
  sum=$(md5sum < "$1") || return
  printf '%s\n' "${sum%% *}"
}
