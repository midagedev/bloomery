#!/usr/bin/env bash
# bloomery — what a Mac needs before its static tier runs, in one place: every host tool and version the round loop
# (`just mac-static`, `just check-recipes`, `tools/round-check.sh`) calls, each missing one named with its install line.
#
#   tools/mac-setup.sh --check       one line per piece, `ok` or `missing` with how to get it; rc 69 when any is missing
#   tools/mac-setup.sh --self-test   every check against stub tools on a scratch PATH; no install, no cargo (check-recipes runs it)
#
# The pieces and their owners:
#   python3 ≥ 3.11     tools/check-ptx-names.py reads tomllib (3.11); every tools/*.py runs under it
#   Python modules     the python3-* packages tools/nightly/install.sh puts on the Linux host nightly for the same self-tests
#                      (check-recipes' pytools units), read from that file: one list for both hosts. python3-numpy is the
#                      module numpy, python3-pil is PIL (pip and Homebrew name it pillow); a package this script cannot name
#                      is a named error (70), so a new one is mapped here when it is added there
#   just               it must parse the tree's justfile (`just --summary`): an older just refuses the recipe attributes
#   rsync ≥ 3          tools/box.sh and tools/mac-remote.sh sync with it; macOS's /usr/bin/rsync is openrsync (protocol 29,
#                      "2.6.9 compatible"), so a Homebrew rsync must come first on PATH
#   git                every tool reads the tree through it
#   the Rust side      tools/mac-check.sh prereqs: the pinned toolchain with clippy, rustfmt and the x86_64-linux std, the
#                      env file $HOME/opt/bloomery-mac-env.sh, cuda.h, the Linux sysroot and libclang (mac-check.sh is their
#                      one owner and prints each install line)
#
# A Python install line follows the interpreter: Homebrew's python3 is externally managed (PEP 668: pip refuses), so the
# line is `brew install …`; any other python3 gets `python3 -m pip install …`. Nothing is installed by this script.
# Exit: 0 every piece in place, 69 one or more missing, 64 an argument it does not take or a host that is not macOS, 70 a
# python3-* package of tools/nightly/install.sh with no module name here.
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
PY_MIN=3.11
RSYNC_MIN=3

say() { printf '%s\n' "$*"; }

# modules_of INSTALL_SH: `<deb> <module> <pip name>` for each python3-* package the install script names, or 70.
modules_of() {
  local f=$1 pkgs p
  pkgs=$(sed -n 's/^for pkg in \(.*\); do$/\1/p' "$f" | tr ' ' '\n' | grep '^python3-' || true)
  if [ -z "$pkgs" ]; then
    say "mac-setup: $f names no python3-* package on its \`for pkg in …; do\` line: the module list has no owner" >&2
    return 70
  fi
  for p in $pkgs; do
    case $p in
      python3-numpy) echo "$p numpy numpy" ;;
      python3-pil) echo "$p PIL pillow" ;;
      *) say "mac-setup: $f installs $p, which this script cannot name as a module: map it in modules_of" >&2; return 70 ;;
    esac
  done
}

# py_install PIP_NAMES…: the install line for this python3.
py_install() {
  local managed
  managed=$(python3 -c 'import os, sysconfig; print(os.path.exists(os.path.join(sysconfig.get_path("stdlib"), "EXTERNALLY-MANAGED")))' 2> /dev/null)
  if [ "$managed" = True ] && [ -n "$(command -v brew)" ]; then
    echo "brew install $*"
  else
    echo "python3 -m pip install $*"
  fi
}

check() {
  local miss=0 v got mods line
  [ "$(uname -s)" = Darwin ] || { say "mac-setup: this checks a Mac's static tier; $(uname -s) is not macOS" >&2; return 64; }
  # python3
  if ! v=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2> /dev/null); then
    say "mac-setup: missing python3: brew install python"
    miss=1
  elif [ "$(printf '%s\n%s\n' "$PY_MIN" "$v" | sort -t. -k1,1n -k2,2n | head -1)" != "$PY_MIN" ]; then
    say "mac-setup: missing python3 ≥ $PY_MIN ($(command -v python3) is $v): brew install python"
    miss=1
  else
    say "mac-setup: ok python3 $v ($(command -v python3))"
    mods=$(modules_of "$HERE/tools/nightly/install.sh") || return $?
    local want=()
    while read -r _ mod pip; do
      if python3 -c "import $mod" > /dev/null 2>&1; then
        say "mac-setup: ok python module $mod"
      else
        say "mac-setup: missing python module $mod (the check-recipes pytools self-tests import it)"
        want+=("$pip")
        miss=1
      fi
    done <<< "$mods"
    [ ${#want[@]} = 0 ] || say "mac-setup:   install: $(py_install "${want[@]}")"
  fi
  # just
  if [ -z "$(command -v just)" ]; then
    say "mac-setup: missing just: brew install just"
    miss=1
  elif ! got=$(just --justfile "$HERE/justfile" --working-directory "$HERE" --summary 2>&1 > /dev/null); then
    say "mac-setup: missing a just that parses the justfile ($(just --version 2> /dev/null), it says: ${got%%$'\n'*}): brew upgrade just"
    miss=1
  else
    say "mac-setup: ok $(just --version)"
  fi
  # rsync
  line=$(rsync --version 2> /dev/null | head -1)
  v=$(printf '%s\n' "$line" | sed -n 's/^rsync  *version v\{0,1\}\([0-9][0-9]*\)\..*/\1/p')
  if [ -z "$(command -v rsync)" ]; then
    say "mac-setup: missing rsync: brew install rsync"
    miss=1
  elif [ -z "$v" ] || [ "$v" -lt "$RSYNC_MIN" ]; then
    say "mac-setup: missing rsync ≥ $RSYNC_MIN first on PATH ($(command -v rsync) says '${line:-nothing}'): brew install rsync, and put $(brew --prefix 2> /dev/null || echo /opt/homebrew)/bin before /usr/bin on PATH"
    miss=1
  else
    say "mac-setup: ok rsync $(printf '%s\n' "$line" | awk '{ print $3 }') ($(command -v rsync))"
  fi
  # git
  if [ -z "$(command -v git)" ]; then
    say "mac-setup: missing git: xcode-select --install"
    miss=1
  else
    say "mac-setup: ok $(git --version)"
  fi
  # the Rust side: mac-check.sh owns it
  if got=$(bash "$HERE/tools/mac-check.sh" prereqs 2>&1); then
    say "mac-setup: ok ${got##*$'\n'}"
  else
    printf '%s\n' "$got" | sed 's/^mac-check.sh: missing/mac-setup: missing/'
    miss=1
  fi
  if [ "$miss" != 0 ]; then
    say "mac-setup: a piece is missing (above, each with its install line)"
    return 69
  fi
  say "mac-setup: ok, every piece the Mac static tier needs is in place"
}

self_test() {
  local d fails=0 n=0 rc out realpy
  realpy=$(command -v python3)
  d=$(mktemp -d "${TMPDIR:-/tmp}/mac-setup-test.XXXXXX") || return 70
  trap 'rm -rf "$d"' RETURN
  mkdir -p "$d/tree/tools/nightly" "$d/bin"
  cp "$HERE/tools/mac-setup.sh" "$d/tree/tools/"
  printf 'for pkg in libclang-common-18-dev python3-numpy python3-pil; do\n' > "$d/tree/tools/nightly/install.sh"
  printf 'x:\n\techo\n' > "$d/tree/justfile"
  # The stubs: each reads its behaviour from the environment, so one PATH serves every case.
  cat > "$d/tree/tools/mac-check.sh" << 'SH'
#!/usr/bin/env bash
[ "$1" = prereqs ] || exit 64
if [ -n "${ST_TOOLCHAIN_MISSING:-}" ]; then echo "mac-check.sh: missing the toolchain rust-toolchain.toml pins: rustup toolchain install x" >&2; exit 69; fi
echo "mac-check: prereqs ok"
SH
  cat > "$d/bin/python3" << SH
#!/usr/bin/env bash
case "\$*" in
  *sys.version_info*) echo "\${ST_PY:-3.14}"; exit 0 ;;
  *EXTERNALLY-MANAGED*) echo "\${ST_MANAGED:-True}"; exit 0 ;;
  "-c import numpy") [ -z "\${ST_NO_NUMPY:-}" ] || exit 1; exit 0 ;;
  "-c import PIL") [ -z "\${ST_NO_PIL:-}" ] || exit 1; exit 0 ;;
esac
exec "$realpy" "\$@"
SH
  cat > "$d/bin/rsync" << 'SH'
#!/usr/bin/env bash
case ${ST_RSYNC:-3} in
  3) echo "rsync  version 3.4.1  protocol version 32" ;;
  open) echo "openrsync: protocol version 29"; echo "rsync version 2.6.9 compatible" ;;
  old) echo "rsync  version 2.6.9  protocol version 29" ;;
esac
SH
  cat > "$d/bin/just" << 'SH'
#!/usr/bin/env bash
case "$*" in
  --version) echo "just 1.59.0" ;;
  *--summary*) [ -z "${ST_JUST_OLD:-}" ] || { echo "error: Unknown attribute \`group\`" >&2; exit 1; }; echo x ;;
esac
SH
  printf '#!/usr/bin/env bash\necho "git version 2.50.0"\n' > "$d/bin/git"
  printf '#!/usr/bin/env bash\necho /opt/homebrew\n' > "$d/bin/brew"
  printf '#!/usr/bin/env bash\necho Darwin\n' > "$d/bin/uname"
  chmod +x "$d"/bin/* "$d/tree/tools/mac-check.sh"
  # case NAME RC PATTERN [ENV=V…]: run --check on the stub PATH (the stubs and the base tools, nothing from Homebrew).
  t() {
    local name=$1 want=$2 pat=$3
    shift 3
    n=$((n + 1))
    rc=0
    out=$(env -i PATH="$d/bin:/usr/bin:/bin" HOME="$d" "$@" bash "$d/tree/tools/mac-setup.sh" --check 2>&1) || rc=$?
    if [ "$rc" != "$want" ] || ! grep -qE -- "$pat" <<< "$out"; then
      echo "mac-setup self-test FAIL: $name (rc $rc, want $want; pattern '$pat')" >&2
      printf '%s\n' "$out" | sed 's/^/  | /' >&2
      fails=$((fails + 1))
    fi
  }
  t 'every piece in place' 0 'mac-setup: ok, every piece'
  t 'numpy missing: named, with the brew line' 69 'install: brew install numpy$' ST_NO_NUMPY=1
  t 'numpy and PIL missing, unmanaged python: one pip line' 69 'install: python3 -m pip install numpy pillow$' ST_NO_NUMPY=1 ST_NO_PIL=1 ST_MANAGED=False
  t 'python below 3.11' 69 'missing python3 ≥ 3.11 .* is 3.9' ST_PY=3.9
  t 'python 3.10 is below 3.11 by number, not by text' 69 'is 3.10' ST_PY=3.10
  t 'openrsync first on PATH' 69 "missing rsync ≥ 3 .*openrsync: protocol version 29" ST_RSYNC=open
  t 'rsync 2.6.9' 69 'missing rsync ≥ 3' ST_RSYNC=old
  t 'a just that cannot parse the justfile' 69 'missing a just that parses the justfile .*Unknown attribute' ST_JUST_OLD=1
  t 'the toolchain, from mac-check.sh' 69 'mac-setup: missing the toolchain' ST_TOOLCHAIN_MISSING=1
  printf 'for pkg in python3-numpy python3-yaml; do\n' > "$d/tree/tools/nightly/install.sh"
  t 'a python3-* package with no module name is 70' 70 'installs python3-yaml, which this script cannot name'
  printf 'apt-get install libfoo\n' > "$d/tree/tools/nightly/install.sh"
  t 'no python3-* line is 70' 70 'names no python3-\* package'
  printf 'for pkg in python3-numpy python3-pil; do\n' > "$d/tree/tools/nightly/install.sh"
  printf '#!/usr/bin/env bash\necho Linux\n' > "$d/bin/uname"
  t 'not macOS is 64' 64 'Linux is not macOS'
  if [ "$fails" != 0 ]; then
    echo "mac-setup self-test: $fails of $n cases failed" >&2
    return 1
  fi
  echo "mac-setup self-test: $n cases ok"
}

case ${1:-} in
  --check) [ $# = 1 ] || { say "mac-setup.sh: --check takes nothing" >&2; exit 64; }; check; exit $? ;;
  --self-test) [ $# = 1 ] || { say "mac-setup.sh: --self-test takes nothing" >&2; exit 64; }; self_test; exit $? ;;
  *) say "usage: tools/mac-setup.sh --check | --self-test" >&2; exit 64 ;;
esac
