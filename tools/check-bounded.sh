#!/usr/bin/env bash
# The static guard of the bounded-child owner (Mac, no build). A shell runner under tools/ starts a bounded child
# through the owner in tools/gate-bound.sh and nowhere else: run_bounded (the gate runners), bounded / lease_bounded
# (a foreground command), bounded_bg (a background one). The owner makes the child's process group, sets its
# cancellation before the child starts, reaps it and returns a cleanup failure as the status; a runner that writes its
# own `timeout`, trap and pid bookkeeping again leaves the lease or a gate lock held by a child no stop reaches.
# What this reads, and how little: it recognises a few spellings, so it guards the boundary and proves no behaviour
# (tools/gate-bound.sh --self-test runs the cancellation table that does).
#
#   bash tools/check-bounded.sh            scan every tools/**/*.sh against tools/bounded-allow.txt
#   bash tools/check-bounded.sh --hits     every direct `timeout` line, listed or not (what an entry's count counts)
#   bash tools/check-bounded.sh --self-test
#
# Rules, one line each, the line's own text only:
#   timeout   `timeout` in command position (the line's start, after && || ; | ( { ! ` $( or then/do/else/if/while/until,
#             after variable assignments) that does not follow `run_bounded`. The owner's own files are listed with
#             their exact counts in tools/bounded-allow.txt, as are the lines that must spell it: a printed command
#             line, a test fixture's stub, a command for another host. Quoted text is not command position.
#   status    a `lease_stop` call whose status goes nowhere (no `||`, `&&`, `if`, `!` or `return` on its line).
#   watch     `run_bounded`, `bounded` or `lease_bounded` around tools/ref/stack-watch.sh: the watch is itself the
#             supervisor of the command it runs, so a second one beside it races on the same grace period.
# tools/bounded-allow.txt: `<path> <count> <why>`; the count is exact, so a file with more hits fails and so does an
# entry above its file's count. Exit: 0 clean; 1 a rule broken or an entry stale (each named); 2 setup.
set -euo pipefail

# scan_file <path>: `<rule> <line> <text>` per hit.
scan_file() {
  awk '
    function strip(s) {
      gsub(/"([^"\\]|\\.)*"/, "Q", s)
      gsub(/\047[^\047]*\047/, "Q", s)
      return s
    }
    {
      raw = $0
      sub(/^[ \t]+/, "", raw)
      if (raw ~ /^#/ || raw == "") next
      s = strip(raw)
      sub(/[ \t]#.*$/, "", s)
      pre = "(^|[;&|({`!][ \t]*|\\$\\([ \t]*|(^|[ \t])(then|do|else|elif|if|while|until)[ \t]+)"
      asg = "(env[ \t]+)?(([A-Za-z_][A-Za-z0-9_]*=[^ \t]*|Q)[ \t]+)*"
      if (s ~ (pre asg "timeout([ \t]|$)")) print "timeout", NR, raw
      if (s ~ /(^|[^A-Za-z_])lease_stop[ \t]/ && s !~ /\|\||&&|(^|[ \t])(if|!|return)[ \t]/ && s !~ /^lease_stop\(\)/ && s !~ /(^|[ \t])(source|command|declare|local|echo|printf)[ \t]/) print "status", NR, raw
      if (s ~ /(run_bounded|(^|[^A-Za-z_])bounded|lease_bounded)[ \t].*stack-watch\.sh/) print "watch", NR, raw
    }
  ' "$1"
}

# scan_tree <root> <allow file>: every violation, one line each, to stdout; 1 when any.
scan_tree() {
  local root=$1 allow=$2 f rel n bad=0 want line rule at text have
  local -a files
  local tmp
  tmp=$(mktemp "${TMPDIR:-/tmp}/check-bounded.XXXXXX") || return 2
  while IFS= read -r f; do files+=("$f"); done < <(cd "$root" && find tools -name '*.sh' | LC_ALL=C sort)
  : > "$tmp"
  for rel in "${files[@]}"; do
    # The checker's own fixtures are the lines it must refuse; its self-test reads them.
    [ "$rel" != tools/check-bounded.sh ] || continue
    n=0
    while IFS= read -r line; do
      [ -n "$line" ] || continue
      rule=${line%% *}
      at=${line#* } at=${at%% *}
      text=${line#* }
      text=${text#* }
      case $rule in
        timeout) n=$((n + 1)) ;;
        *) echo "check-bounded: $rel:$at: [$rule] $text" >> "$tmp"; bad=1 ;;
      esac
    done < <(scan_file "$root/$rel")
    [ "$n" = 0 ] && continue
    want=$(awk -v f="$rel" '$1 == f { print $2 }' "$allow")
    [ -n "$want" ] || want=0
    if [ "$n" != "$want" ]; then
      echo "check-bounded: $rel: $n direct \`timeout\` line(s), tools/bounded-allow.txt allows $want: run the child through run_bounded, bounded, lease_bounded or bounded_bg (tools/gate-bound.sh), or state the line's reason there" >> "$tmp"
      bad=1
    fi
  done
  while read -r f want _; do
    case $f in '' | '#'*) continue ;; esac
    [ -f "$root/$f" ] || { echo "check-bounded: tools/bounded-allow.txt: $f is not in the tree" >> "$tmp"; bad=1; continue; }
    have=$(scan_file "$root/$f" | awk '$1 == "timeout"' | wc -l | tr -d ' ')
    [ "$have" -gt 0 ] || { echo "check-bounded: tools/bounded-allow.txt: $f has no direct \`timeout\` line left (entry says $want)" >> "$tmp"; bad=1; }
  done < "$allow"
  cat "$tmp"
  rm -f "$tmp"
  return "$bad"
}

self_test() {
  local t bad=0 out rc
  t=$(mktemp -d "${TMPDIR:-/tmp}/check-bounded-test.XXXXXX") || return 70
  # shellcheck disable=SC2064 # the path is fixed now
  trap "rm -rf '$t'" EXIT
  case_() { # name, want rc, want pattern, root
    rc=0
    out=$(scan_tree "$4" "$4/tools/bounded-allow.txt" 2>&1) || rc=$?
    if [ "$rc" = "$2" ] && { [ -z "$3" ] && [ -z "$out" ] || grep -Eq -- "$3" <<< "$out"; }; then echo "ok     $1"; else
      echo "FAIL   $1 (rc $rc, want $2, pattern '$3'; output:)"; sed 's/^/         /' <<< "$out"; bad=$((bad + 1))
    fi
  }
  mk() { # name: a tree of one runner, its text on stdin
    mkdir -p "$t/$1/tools"
    : > "$t/$1/tools/bounded-allow.txt"
    cat > "$t/$1/tools/r.sh"
  }
  mk clean << 'EOF'
#!/usr/bin/env bash
source tools/gate-bound.sh
run_bounded timeout --kill-after=10 "$BOUND" "$BIN"
bounded 60 "$BIN" --help
bounded_bg 60 "$BIN" serve
echo "timeout --kill-after=10 60 $BIN"   # a printed command line
# timeout 60 $BIN  (a comment)
command -v timeout > /dev/null || exit 1
timeout() { shift 2; "$@"; }
lease_stop "$pid" || rc=$?
if ! lease_stop "$pid"; then rc=1; fi
EOF
  case_ 'owner forms, quoted text, comments, a function named timeout, a handled stop: clean' 0 '' "$t/clean"
  mk bare << 'EOF'
timeout --kill-after=10 60 "$BIN"
EOF
  case_ 'a bare timeout line is refused with the runner named' 1 'r\.sh: 1 direct `timeout` line\(s\), tools/bounded-allow.txt allows 0' "$t/bare"
  mk spellings << 'EOF'
x=$(timeout 5 "$BIN")
"$BIN" && timeout 5 true
CUDA_VISIBLE_DEVICES=1 timeout 5 "$BIN" &
if timeout 5 "$BIN"; then :; fi
CMD=(timeout --kill-after=10 5 "$BIN")
EOF
  case_ 'a substitution, an && chain, an env prefix, an if, an array element: five direct timeouts' 1 'r\.sh: 5 direct `timeout` line\(s\)' "$t/spellings"
  mk allowed << 'EOF'
CMD=(timeout --kill-after=10 5 "$BIN")
EOF
  printf '%s\n' 'tools/r.sh 1 the printed command line' > "$t/allowed/tools/bounded-allow.txt"
  case_ 'an allowlisted line at its exact count: clean' 0 '' "$t/allowed"
  printf '%s\n' 'tools/r.sh 2 too many' > "$t/allowed/tools/bounded-allow.txt"
  case_ 'an entry above the file count is refused' 1 'allows 2' "$t/allowed"
  mk gone << 'EOF'
run_bounded timeout 5 true
EOF
  printf '%s\n' 'tools/r.sh 1 stale' > "$t/gone/tools/bounded-allow.txt"
  case_ 'an entry for a file with no direct timeout left is refused' 1 'has no direct `timeout` line left' "$t/gone"
  mk status << 'EOF'
lease_stop "$pid"
lease_stop "$pid" || :
EOF
  case_ 'a lease_stop whose status goes nowhere is refused' 1 'r\.sh:1: \[status\]' "$t/status"
  mk watch << 'EOF'
run_bounded timeout 60 bash tools/ref/stack-watch.sh 30 gate -- "$BIN"
bash tools/ref/stack-watch.sh 30 gate -- run_bounded timeout 60 "$BIN"
EOF
  case_ 'a bound around the stack watch is refused' 1 'r\.sh:1: \[watch\]' "$t/watch"
  echo "check-bounded self-test: $([ "$bad" = 0 ] && echo ok || echo FAIL) ($bad failures)"
  [ "$bad" = 0 ]
}

main() {
  local root allow
  root=$(cd "$(dirname "$0")/.." && pwd)
  allow=$root/tools/bounded-allow.txt
  [ -f "$allow" ] || { echo "check-bounded: $allow is missing" >&2; return 2; }
  scan_tree "$root" "$allow" >&2 || return 1
  echo "check-bounded: ok"
}

hits() {
  local root rel
  root=$(cd "$(dirname "$0")/.." && pwd)
  while IFS= read -r rel; do
    [ "$rel" != tools/check-bounded.sh ] || continue
    scan_file "$root/$rel" | awk -v f="$rel" '$1 == "timeout" { $1 = ""; n = $2; $2 = ""; sub(/^ +/, ""); printf "%s:%s: %s\n", f, n, $0 }'
  done < <(cd "$root" && find tools -name '*.sh' | LC_ALL=C sort)
}

case ${1:-} in
  --self-test) self_test ;;
  --hits) hits ;;
  '') main ;;
  *) echo "usage: check-bounded.sh [--self-test]" >&2; exit 2 ;;
esac
