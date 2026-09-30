# shellcheck shell=bash
# A lever arm's NAME=VALUE list, checked the same way in the depth runners that take one
# (depth-qwen3moe.sh, depth-glm5next.sh): `<D>@NAME=VALUE[,NAME=VALUE...]` sets those variables for that
# arm's process only. Sourced by the runner before it parses its arms.
#
# The runner sets, before it sources this file:
#   LEVER_ARM_RUNNER   the runner's name, which opens every message here
#   LEVER_REGISTRY     crates/levers/src/registry.rs, the rows a name is checked against
#   LEVER_ARM_PASS     (optional) one NAME=VALUE item the runner consumes itself and the binary never sees
#                      (depth-qwen3moe.sh's load driver's solo marker): skipped by the registry check
# and defines arm_refuse <arm> <why>, which prints the refusal in the runner's shape and exits 64.

# lever_rows_py <registry.rs>: every row of the lever registry as `<name> <site>` — Parsed or Direct (a
# lever), Retired, or Env (a runner's, a harness's or a path's own variable: the path() and runner()
# rows); a row it cannot read with a name and a site, or a name given twice, stops it by name.
lever_rows_py() {
  python3 - "$1" << 'PY'
import re, sys

path = sys.argv[1]
src = open(path, encoding='utf-8').read()


def die(why):
    sys.exit(f'{path}: {why}')


consts = dict(re.findall(r'\bconst\s+([A-Z][A-Z0-9_]*)\s*:\s*&(?:\'static\s+)?str\s*=\s*"([^"]*)"', src))
at = src.find('static REGISTRY')
if at < 0:
    die('no `static REGISTRY`')
body = src[at:]
specs = len(re.findall(r'\bLeverSpec\s*\{', body))
rows = []
for m in re.finditer(r'\bLeverSpec\s*\{.*?\bname:\s*(?:"([^"]*)"|([A-Z][A-Z0-9_]*))\s*,.*?\bsite:\s*Site::([A-Za-z]+)',
                     body, re.S):
    name = m.group(1) or consts.get(m.group(2))
    if not name:
        die(f'the row name {m.group(2)} is no const of the file')
    rows.append((name, m.group(3)))
if len(rows) != specs:
    die(f'{specs} LeverSpec rows, {len(rows)} read with a name and a site')
rows += [(m.group(1), 'Env') for m in re.finditer(r'\b(?:path|runner)\(\s*"([^"]*)"', body)]
names = [n for n, _ in rows]
twice = sorted({n for n in names if names.count(n) > 1})
if twice:
    die('names given twice: ' + ', '.join(twice))
for name, site in rows:
    print(name, site)
PY
}
# lever_rows_load: LEVER_ROWS from LEVER_REGISTRY, once, when the first lever arm is parsed; a missing or
# unreadable registry exits 2 by name.
LEVER_ROWS=
lever_rows_load() {
  [ -z "$LEVER_ROWS" ] || return 0
  [ -r "$LEVER_REGISTRY" ] || { echo "$LEVER_ARM_RUNNER: a lever arm is checked against the lever registry, and there is no lever registry at $LEVER_REGISTRY" >&2; exit 2; }
  LEVER_ROWS=$(lever_rows_py "$LEVER_REGISTRY") || { echo "$LEVER_ARM_RUNNER: the lever registry $LEVER_REGISTRY was not read (above)" >&2; exit 2; }
  [ -n "$LEVER_ROWS" ] || { echo "$LEVER_ARM_RUNNER: the lever registry $LEVER_REGISTRY has no rows" >&2; exit 2; }
}
# arm_envs_ok <arm> <NAME=VALUE list>: a lever arm's list, each refusal by name (the header's
# <D>@NAME=VALUE), exit 64.
arm_envs_ok() {
  local a=$1 list=$2 e name val site seen=,
  local -a kv
  [ -n "$list" ] || arm_refuse "$a" "an empty NAME=VALUE list after '@'"
  case $list in *[[:space:]]*) arm_refuse "$a" "white space in '$list' (a value holds none)" ;; esac
  case ,$list, in *,,*) arm_refuse "$a" "an empty item in '$list' (NAME=VALUE items are separated by one ',')" ;; esac
  IFS=, read -r -a kv <<< "$list"
  for e in "${kv[@]}"; do
    case $e in *=*) ;; *) arm_refuse "$a" "'$e' is no NAME=VALUE (a ',' separates two variables, so a value holds none)" ;; esac
    name=${e%%=*} val=${e#*=}
    [ -n "$name" ] || arm_refuse "$a" "'$e' has an empty name"
    [ -n "$val" ] || arm_refuse "$a" "$name has an empty value"
    [[ $name =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || arm_refuse "$a" "'$name' is no variable name"
    case $val in
      *@*) arm_refuse "$a" "the value of $name holds '@', which opens an arm's variables" ;;
      *'|'*) arm_refuse "$a" "the value of $name holds '|', the separator of this runner's records" ;;
    esac
    case $seen in *,"$name",*) arm_refuse "$a" "$name is given twice" ;; esac
    seen+="$name,"
    # The runner's own item (LEVER_ARM_PASS), never the binary's.
    [ -z "${LEVER_ARM_PASS:-}" ] || [ "$e" != "$LEVER_ARM_PASS" ] || continue
    if printenv "$name" > /dev/null; then
      arm_refuse "$a" "$name is set in the runner's own environment ($name=$(printenv "$name")), which every arm inherits, so the plain arms' labels would not show it: give it per arm only"
    fi
    lever_rows_load
    site=$(awk -v n="$name" '$1 == n { print $2 }' <<< "$LEVER_ROWS")
    case $site in
      Parsed | Direct) ;;
      Retired) arm_refuse "$a" "$name is a retired name ($LEVER_REGISTRY): the binary refuses it" ;;
      Env) arm_refuse "$a" "$name is no lever: its row in $LEVER_REGISTRY is a runner's, a harness's or a path's own variable, not a setting of the binary's" ;;
      '') arm_refuse "$a" "$name is no row of the lever registry ($LEVER_REGISTRY): a lever arm sets a lever" ;;
      *) arm_refuse "$a" "$name's row in $LEVER_REGISTRY is a Site::$site, which this runner does not know" ;;
    esac
  done
}
