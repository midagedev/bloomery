# shellcheck shell=bash
# A bin: arm's row label in the depth runners (depth-qwen3moe.sh, depth-ds41.sh). The label is
# `bin:<basename of its tree>`: the tree is the path above `target/`, and the binary's own directory
# when the path has no `target/` segment. The runners' headers keep that basename — the label does not
# carry the path — and say every label is its own engine in the per-arm means. Two arms that would share
# the label and run different binaries are therefore refused by name before a run (exit 64). The same
# binary at two depths keeps one label; the depth is the mean's other key.
#
# The runner defines arm_refuse (<arm> <why>, exit 64) and, after every arm is parsed, calls
# bin_labels_ok. Sourced before the arms are parsed. `bash tools/ref/bin-label.sh --self-test` runs the
# cases on fixed paths, no box (check-recipes runs it).

# bin_tree <binary>: the tree, printed. The path above `target/`, or the binary's directory when the
# path has no `target/` segment.
bin_tree() {
  local tree=${1%/target/*}
  [ "$tree" != "$1" ] || tree=${1%/*}
  printf '%s\n' "$tree"
}
# bin_label_stem <binary>: `bin:<basename of its tree>`, printed. The runner appends its own `@prose`,
# `@<corpus>` and `@NAME=VALUE` tail.
bin_label_stem() {
  local tree
  tree=$(bin_tree "$1")
  printf 'bin:%s\n' "${tree##*/}"
}
# bin_labels_ok: a label two arms share while their binaries differ, refused by name (the runner's
# arm_refuse, exit 64) before a run. Reads ARMS, A_LABEL and A_BIN, one entry an arm. Arms of one
# binary keep the label.
bin_labels_ok() {
  local i j n=${#A_LABEL[@]}
  i=0
  while [ "$i" -lt "$n" ]; do
    j=$((i + 1))
    while [ "$j" -lt "$n" ]; do
      if [ -n "${A_BIN[$i]:-}" ] && [ "${A_LABEL[$i]}" = "${A_LABEL[$j]}" ] && [ "${A_BIN[$i]}" != "${A_BIN[$j]:-}" ]; then
        arm_refuse "${ARMS[$j]}" "shares the label '${A_LABEL[$i]}' with arm '${ARMS[$i]}', and the two run different binaries: a label is one engine in the per-arm means (bin:<basename of its tree>), so the rows would be averaged as one"
      fi
      j=$((j + 1))
    done
    i=$((i + 1))
  done
}

# `bash tools/ref/bin-label.sh --self-test`: the stem and the refusal on fixed paths (bash 3.2 on the
# Mac); one line per check, `ok <name>` or `FAIL <name>: got … want …`, the verdict last.
if [ "${BASH_SOURCE[0]}" = "$0" ] && [ "${1:-}" = --self-test ]; then
  fails=0 checks=0
  check() {
    checks=$((checks + 1))
    if [ "$2" = "$3" ]; then
      echo "ok $1"
    else
      echo "FAIL $1: got [$2] want [$3]"
      fails=$((fails + 1))
    fi
  }
  arm_refuse() {
    echo "runner.sh: arm '$1': $2" >&2
    exit 64
  }
  check stem-target "$(bin_label_stem /root/r/t/target/release/generate_qwen3moe)" bin:t
  check stem-nodir "$(bin_label_stem /root/r/b1ab/b1)" bin:b1ab
  check tree-target "$(bin_tree /root/r/t/target/release/generate_qwen3moe)" /root/r/t
  check tree-nodir "$(bin_tree /root/r/b1ab/b1)" /root/r/b1ab
  # The same binary at two depths is one engine.
  A_LABEL=(bin:t bin:t)
  A_BIN=(/root/r/t/target/release/generate_qwen3moe /root/r/t/target/release/generate_qwen3moe)
  ARMS=(bin:/root/r/t/target/release/generate_qwen3moe:6 bin:/root/r/t/target/release/generate_qwen3moe:512)
  out=$(bin_labels_ok 2>&1) && r=0 || r=$?
  check same-bin "$r|$out" '0|'
  # Two binaries in one directory share the directory's basename. That label is refused.
  A_LABEL=(bin:b1ab bin:b1ab)
  A_BIN=(/root/r/b1ab/b1 /root/r/b1ab/base)
  ARMS=(bin:/root/r/b1ab/b1:6 bin:/root/r/b1ab/base:6)
  out=$(bin_labels_ok 2>&1) && r=0 || r=$?
  check shared "$r" 64
  check shared-why "$out" "runner.sh: arm 'bin:/root/r/b1ab/base:6': shares the label 'bin:b1ab' with arm 'bin:/root/r/b1ab/b1:6', and the two run different binaries: a label is one engine in the per-arm means (bin:<basename of its tree>), so the rows would be averaged as one"
  # A suffix makes another label, so the means do not merge them.
  A_LABEL=(bin:b1ab bin:b1ab@prose)
  out=$(bin_labels_ok 2>&1) && r=0 || r=$?
  check suffix "$r|$out" '0|'
  # depth-ds41's witness prefix and block key call this stem. A second `bin:${…##*/}` is another owner.
  ds41=${BASH_SOURCE[0]%/*}/depth-ds41.sh
  check ds41-stem-calls "$(grep -c 'bin_label_stem' "$ds41")" 3
  check ds41-one-format "$(grep -c 'bin:${' "$ds41" || true)" 0
  echo "self-test: $([ "$fails" = 0 ] && echo ok || echo FAIL) ($checks checks, $fails failures)"
  [ "$fails" = 0 ]
  exit
fi
