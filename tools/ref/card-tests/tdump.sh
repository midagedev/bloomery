#!/usr/bin/env bash
# The t dump: the two-sided 95 % t quantile each of the four readers of the t table uses, df 1..40,
# one row per df:
#   df  gpu-ab  card  depth-ds41  depth-qwen3moe
# gpu-ab is tools/gpu-ab.py's t975, card is tools/ref/card.py's t_table(), and the last two are what
# each depth runner's ratio_table prints: its awk runs on 41 rounds of records whose per-round ratios
# have a sample SD of sqrt(c) over c rounds, so the interval it prints for c rounds is t at
# df = c - 1 itself (4 decimals). The function runs as the runner holds it (cut out of the script)
# with T975 made the way the runner makes it.
#
#   tools/ref/card-tests/tdump.sh [TREE]     (TREE defaults to this tree)
#
# Mac and box, builds nothing. Exit 1 when a reader cannot be read (a missing function, no row for a
# df), never a short table.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TREE=$(cd "${1:-$HERE/../../..}" && pwd) || exit 1
MAXDF=40
ROUNDS=$((MAXDF + 1))
tmp=${TMPDIR:-/tmp}
tmp=$(mktemp -d "${tmp%/}/tdump.XXXXXX")
trap 'rm -rf "$tmp"' EXIT

python3 - "$TREE" "$MAXDF" > "$tmp/py" << 'PY' || exit 1
import importlib.util, os, sys

tree, maxdf = sys.argv[1], int(sys.argv[2])


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gpu_ab = load('gpu_ab', os.path.join(tree, 'tools', 'gpu-ab.py')).t975
card = load('card', os.path.join(tree, 'tools', 'ref', 'card.py')).t_table()
for df in range(1, maxdf + 1):
    print(df, f'{gpu_ab(df):.4f}', f'{card(df):.4f}')
PY

# Records for rounds c = 2..ROUNDS: key c holds c rounds of ours = 100 + a (r - (c + 1) / 2) with
# a = sqrt(12 / (c + 1)) against a reference of 1, a sample SD of sqrt(c).
python3 - "$ROUNDS" > "$tmp/records" << 'PY' || exit 1
import math, sys

rounds = int(sys.argv[1])
for c in range(2, rounds + 1):
    a = math.sqrt(12 / (c + 1))
    for r in range(1, c + 1):
        print(f'ours|{c}|{r}|{100 + a * (r - (c + 1) / 2):.15f}|')
        print(f'ik|{c}|{r}|1|')
PY
keys=$(seq -s ' ' 2 "$ROUNDS")

for runner in depth-ds41 depth-qwen3moe; do
  src=$TREE/tools/ref/$runner.sh
  fn=$(sed -n '/^ratio_table() {$/,/^}$/p' "$src")
  if [ -z "$fn" ]; then
    echo "tdump: no ratio_table() in $src" >&2
    exit 1
  fi
  T975=
  if [ -f "$TREE/tools/ref/tdist.py" ]; then
    T975=$(python3 "$TREE/tools/ref/tdist.py" "$ROUNDS") || exit 1
  fi
  # shellcheck disable=SC2030,SC2031 # ROUNDS and T975 are what the cut-out function reads
  (
    export T975 ROUNDS
    eval "$fn"
    ratio_table '' "$keys" ik 0 < "$tmp/records"
  ) | sed -n 's/^ *\([0-9][0-9]*\) *ours\/ik *mean [0-9.]* ± \([0-9.]*\) (n=\([0-9]*\)).*/\3 \2/p' |
    awk '{ print $1 - 1, $2 }' > "$tmp/$runner"
done

echo "df gpu-ab card depth-ds41 depth-qwen3moe"
rc=0
for ((df = 1; df <= MAXDF; df++)); do
  py=$(awk -v d="$df" '$1 == d { print $2, $3 }' "$tmp/py")
  a=$(awk -v d="$df" '$1 == d { print $2 }' "$tmp/depth-ds41")
  b=$(awk -v d="$df" '$1 == d { print $2 }' "$tmp/depth-qwen3moe")
  if [ -z "$py" ] || [ -z "$a" ] || [ -z "$b" ]; then
    echo "tdump: df $df: a reader printed no value (gpu-ab/card '${py}', depth-ds41 '${a}', depth-qwen3moe '${b}')" >&2
    rc=1
  fi
  echo "$df ${py:-?} ${a:-?} ${b:-?}"
done
exit "$rc"
