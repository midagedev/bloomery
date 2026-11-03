#!/usr/bin/env bash
# Publishes a built release: `just release-build <version>` makes the tarball on the
# box; this script is everything after it, so the upload cannot repeat v0.2.0's
# mistake (gh named the assets from temp filenames and every download URL 404'd).
#
#   tools/release/publish.sh <version> [--dry-run] [--no-tap] [--no-sm120a]
#   tools/release/publish.sh <version> --tap-only   step 5 alone, on the release's own .sha256 assets (after a GB10 upload)
#   tools/release/publish.sh --self-test     (the rotation's refusal on stub rotation.py exits, and step 3a's warnings; check-recipes runs it)
#   tools/release/publish.sh --sitting-check <tarball> <name> <dir>...   step 3a alone, on the verdicts in each <dir>
#   tools/release/publish.sh --tree-check <version> [<rev>]   steps 2 and 2b alone, on the working tree or on <rev>'s tree
#                                      (git show, nothing is checked out): `just release-build`, tools/release/build.sh and
#                                      tools/release/pick.sh call it before any archive is built, so the README commit
#                                      precedes the builds and the tag never moves after them
#   tools/release/publish.sh --bump-next <version> [<next>]   edits the working tree to the next version (step 6 names it)
#
# What it owns, in order:
#   0. the rotation: `python3 tools/rotation.py check` must be green (every F family has a green ledger row inside the
#      rotation's window: the release batch leaves those gates to the rotation); rc 1 stops here, and so does rc 2 or 70, each named;
#   1. the tag: HEAD must carry tag v<version> (the release is the tag; the binaries'
#      --version names this commit);
#   2. the versions: the serve AND gpu-gates crates must both say <version> (--version
#      reads gpu-gates, /props reads serve — v0.2.0 shipped with 0.1.0 in --version), and
#   2b. the README names this release: its status heading, the tarball it unpacks and the release link; both are
#      tree_check below, the one owner of the two steps, which the builds call first (--tree-check);
#   3. the artifacts: on the box, where readelf and the ELF binary run — the release's
#      own .sha256, build.sh's checks on the unpacked binary (for the archive's arch), and its --version;
#      then the tarball and .sha256 come here for the upload and the tap's digest; the
#      box holds two x86-64 archives, sm_86 (`just release-build <v>`) and sm_120a
#      (`BLOOMERY_ARCH=sm_120a just release-build <v>`, the RTX 50 series'), and a release
#      without the second one is published with --no-sm120a;
#      the GB10 archive (linux-aarch64-cuda-sm121a) the same way on the GB10 that built
#      it, named by BLOOMERY_GB10_BOX (unset: the release carries no GB10 archive, and
#      step 6 says so);
#   3a. the sitting's verdict (tools/release/sitting.sh): a `stage=portable` verdict, result=green, whose binary_sha256
#      equals the tarball's <name>/bin/bloomery-serve — the release's headline rows come from the binary it ships. The
#      box's target/release-dist/<name>.sitting is fetched, and every *.sitting in PUBLISH_SITTING_DIR (when set) is
#      read too. None that qualifies is a WARNING line naming what is missing (a stage=tree verdict, another binary's,
#      a red one); it never stops the publish, and the line is the release log's record of it;
#   3b. the Apple Silicon artifacts, which tools/release/build-mac.sh wrote into this tree's
#      target/mac-release/dist/ from this commit: their .sha256, build-mac.sh's binary checks
#      on the unpacked binaries, a Developer ID signature on each, and the commit the
#      archive's README names;
#   4. the upload: `gh release upload --clobber` with exactly the names the build scripts
#      wrote (bloomery-<v>-linux-x86_64-cuda-sm86.tar.gz, bloomery-<v>-linux-x86_64-cuda-sm120a.tar.gz,
#      bloomery-<v>-macos-arm64-metal.tar.gz, bloomery-<v>-linux-aarch64-cuda-sm121a.tar.gz with a GB10
#      archive, and their .sha256) —
#      it refuses to upload a file under any other name — then curls every URL for a 200;
#   5. the Homebrew tap: bumps Formula/bloomery.rb's version and each platform's url and
#      sha256 in ~/repo/homebrew-tap (the GB10 one where the formula has its url; the sm_120a
#      archive is the installer's pick, never the formula's), commits and pushes. A formula with a GB10 url and a
#      release without a GB10 archive leaves the tap as it is, named: the GB10 archive is built on a Spark after the
#      tag, and `--tap-only` bumps the tap once it is on the release (a bump without it would install an older GB10
#      build under the new version);
#   6. prints what is left to the lead: the release notes (gh release edit --notes-file),
#      the ghcr image — the v* tag push already triggered .github/workflows/container.yml — and the next-version edit
#      (`--bump-next <version>`: the serve and gpu-gates crates and their two Cargo.lock entries move to the next patch
#      version on main, so the tarballs built there before the next ask are that release's).
#
# --dry-run does 1–3 and prints 4–6 without touching the release or the tap.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"

# rotation_gate: tools/rotation.py check's verdict as this script's: 0 green, 1 and the named reason otherwise. The check's own lines
# (the families it names) go to stderr first.
rotation_gate() {
    local rc=0
    python3 tools/rotation.py check >&2 || rc=$?
    case "$rc" in
        0) ;;
        1) echo "publish.sh: the rotation is red: the lines above name each family with no green ledger row in the rotation's window — run \`just rotation-due\` and the batch it lists on the lead's ledger, then publish" >&2; return 1 ;;
        *) echo "publish.sh: tools/rotation.py check could not run (rc $rc; 2 a bad input, 70 a tool it runs failed): the lines above name it" >&2; return 1 ;;
    esac
}

# sitting_check <tarball> <name> <dir>...: step 3a. Prints `sitting verdict ok` for a `*.sitting` file in one of the <dir>s that
# is stage=portable, result=green and has the sha256 of the tarball's <name>/bin/bloomery-serve as its binary_sha256; else a
# WARNING that names what is missing and each verdict that was read and why it does not count. Always returns 0.
sitting_check() {
    local tb=$1 name=$2 tmp msha='' d f n stage res bsha why found='' notes=()
    shift 2
    tmp=$(mktemp) || return 0
    if tar -xzOf "$tb" "$name/bin/bloomery-serve" > "$tmp" 2> /dev/null && [ -s "$tmp" ]; then
        msha=$(shasum -a 256 "$tmp" | awk '{print $1}')
    fi
    rm -f "$tmp"
    for d in "$@"; do
        for f in "$d"/*.sitting; do
            [ -f "$f" ] || continue
            stage=$(sed -n 's/^stage=//p' "$f" | head -n1)
            res=$(sed -n 's/^result=//p' "$f" | head -n1)
            bsha=$(sed -n 's/^binary_sha256=//p' "$f" | head -n1)
            why=''
            if [ "$stage" != portable ]; then
                why="stage=${stage:-none} (a tree verdict is an iteration's, never the release's)"
            elif [ -z "$msha" ] || [ "$bsha" != "$msha" ]; then
                why="its binary sha256 ${bsha:-none} is not the tarball's member ${msha:-(unreadable)}"
            elif [ "$res" != green ]; then
                why="result=${res:-none}"
            fi
            if [ -z "$why" ]; then
                found=$f
                break 2
            fi
            notes+=("$f: $why")
        done
    done
    if [ -n "$found" ]; then
        echo "publish.sh: sitting verdict ok: $found (stage=portable, green, binary sha256 $msha is the tarball's member)"
        return 0
    fi
    echo "publish.sh: WARNING: no stage=portable sitting verdict for this tarball's binary (sha256 ${msha:-unreadable}): the headline rows are not from the binary the release ships; run tools/release/sitting.sh --binary portable --tarball <this tarball> on the box (just release-sitting)" >&2
    for n in ${notes[@]+"${notes[@]}"}; do
        echo "publish.sh: WARNING:   read $n" >&2
    done
    return 0
}

# release_assets <version> <gb10 0|1> <sm120a 0|1>: the archives one release uploads, one name a line, no extension. The x86-64
# sm_86 archive is first: step 4 uploads its .sha256 alone and last (the image workflow starts on it), so the files before it
# are the other archives'.
release_assets() {
    echo "bloomery-$1-linux-x86_64-cuda-sm86"
    [ "$3" = 0 ] || echo "bloomery-$1-linux-x86_64-cuda-sm120a"
    [ "$2" = 0 ] || echo "bloomery-$1-linux-aarch64-cuda-sm121a"
    echo "bloomery-$1-macos-arm64-metal"
}

# tree_check <version> [<rev>]: steps 2 and 2b. The serve and gpu-gates crates say <version>, and README.md names the release:
# its status heading, the sm_86 tarball it unpacks and the release link. Read from the working tree, or from <rev>'s tree when one
# is given (git show; nothing is checked out). Returns 0 with nothing printed, or 1 with the first refusal on stderr, naming the
# file and what to do. The check is the same wherever it runs: before the builds (tools/release/build.sh, tools/release/pick.sh)
# and again here at the publish.
tree_file() { # tree_file <path> [<rev>]
    if [ -n "${2:-}" ]; then git show "$2:$1"; else cat -- "$1"; fi
}
tree_check() {
    local version=$1 rev=${2:-} c text v want at=''
    [ -z "$rev" ] || at=" at $rev"
    for c in serve gpu-gates; do
        text=$(tree_file "crates/$c/Cargo.toml" "$rev" 2> /dev/null) || { echo "publish.sh: cannot read crates/$c/Cargo.toml$at" >&2; return 1; }
        v=$(awk -F'"' '/^version = "/ {print $2; exit}' <<< "$text")
        [ "$v" = "$version" ] || { echo "publish.sh: crates/$c says version ${v:-none}$at, not $version — bump it and commit before the builds" >&2; return 1; }
    done
    text=$(tree_file README.md "$rev" 2> /dev/null) || { echo "publish.sh: cannot read README.md$at" >&2; return 1; }
    for want in "## Status ($version)" "bloomery-$version-linux-x86_64-cuda-sm86.tar.gz" "releases/tag/v$version)"; do
        grep -qF -- "$want" <<< "$text" || { echo "publish.sh: README.md$at does not name '$want' — update its status heading and install lines and commit before the builds" >&2; return 1; }
    done
}

# bump_next <version> [<next>]: the working tree moves to <next> (default: <version>'s next patch) on the serve and gpu-gates
# crates and the two Cargo.lock entries, the shape of the commit "version: main moves to <next> after the <version> release". Nothing is written unless
# all four places say <version> now; the files are edited and the commit is the lead's (the command is printed). Returns 0, or 1
# and a named refusal, or 64 for a malformed version.
bump_next() {
    local cur=$1 next=${2:-} semver='^([0-9]+)\.([0-9]+)\.([0-9]+)$'
    [[ $cur =~ $semver ]] || { echo "publish.sh: --bump-next takes MAJOR.MINOR.PATCH, got '$cur'" >&2; return 64; }
    [ -n "$next" ] || next="${BASH_REMATCH[1]}.${BASH_REMATCH[2]}.$((BASH_REMATCH[3] + 1))"
    [[ $next =~ $semver ]] || { echo "publish.sh: --bump-next's next version is MAJOR.MINOR.PATCH, got '$next'" >&2; return 64; }
    [ "$next" != "$cur" ] || { echo "publish.sh: --bump-next: the next version is the current one, $cur" >&2; return 64; }
    python3 -I - "$cur" "$next" << 'PY' || return 1
import re
import sys

cur, nxt = sys.argv[1:3]
crates = {"crates/serve/Cargo.toml": "bloomery-serve", "crates/gpu-gates/Cargo.toml": "bloomery-gpu-gates"}
edits = {}
try:
    lock = open("Cargo.lock", encoding="utf-8").read()
    for path, name in crates.items():
        text = open(path, encoding="utf-8").read()
        m = re.search(r'^name = "' + re.escape(name) + r'"\nversion = "([^"]*)"$', text, re.M)
        if not m or m.group(1) != cur:
            print(f"publish.sh: --bump-next: {path} says version {m.group(1) if m else 'none'}, not {cur}: nothing was written", file=sys.stderr)
            sys.exit(1)
        edits[path] = text[:m.start(1)] + nxt + text[m.end(1):]
        m = re.search(r'^name = "' + re.escape(name) + r'"\nversion = "([^"]*)"$', lock, re.M)
        if not m or m.group(1) != cur:
            print(f"publish.sh: --bump-next: Cargo.lock's {name} entry says version {m.group(1) if m else 'none'}, not {cur}: nothing was written", file=sys.stderr)
            sys.exit(1)
        lock = lock[:m.start(1)] + nxt + lock[m.end(1):]
except OSError as e:
    print(f"publish.sh: --bump-next: {e}: nothing was written", file=sys.stderr)
    sys.exit(1)
edits["Cargo.lock"] = lock
for path, text in edits.items():
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
PY
    echo "publish.sh: the working tree moved from $cur to $next (crates/serve, crates/gpu-gates, Cargo.lock); the commit is the lead's:"
    echo "  git commit -m 'version: main moves to $next after the $cur release' -- Cargo.lock crates/serve/Cargo.toml crates/gpu-gates/Cargo.toml"
}

# formula_bump <formula> <version> <sha> <mac sha> <gb10 sha or ''>: the tap formula's version, urls and sha256s in place;
# exit 3 and no write when the formula names a GB10 archive and the release carries none.
formula_bump() {
    python3 - "$@" <<'PY'
import re, sys
f, v, sha, mac_sha, gb10_sha = sys.argv[1:6]
s = open(f).read()
s, n = re.subn(r'version "[^"]*"', f'version "{v}"', s, count=1)
assert n == 1, "no version line"
# Each platform's url and the sha256 on the line after it. The GB10 url is the formula's choice: bumped where it has
# one. A release without a GB10 archive leaves the formula alone (exit 3), since its GB10 url would keep naming an
# older release under the new version.
def bump(s, kind, digest):
    return re.subn(r'(url "[^"]*/releases/download/)v[^/]*/bloomery-[^"/]*-' + re.escape(kind) +
                   r'\.tar\.gz"(\s+)sha256 "[0-9a-f]{64}"',
                   lambda m: f'{m.group(1)}v{v}/bloomery-{v}-{kind}.tar.gz"{m.group(2)}sha256 "{digest}"', s)
for kind, digest in (("linux-x86_64-cuda-sm86", sha), ("macos-arm64-metal", mac_sha)):
    s, n = bump(s, kind, digest)
    assert n == 1, (kind, n)
gb10 = "linux-aarch64-cuda-sm121a"
if gb10_sha:
    s, n = bump(s, gb10, gb10_sha)
    assert n <= 1, (gb10, n)
    if n == 0:
        print(f"publish.sh: the tap's formula has no {gb10} url; Homebrew on a GB10 installs nothing until it has one", file=sys.stderr)
elif gb10 in s:
    print(f"publish.sh: the tap waits for the GB10 archive: its formula names {gb10}, which this release does not carry yet; "
          f"once it is on the release, run tools/release/publish.sh {v} --tap-only", file=sys.stderr)
    sys.exit(3)
open(f, 'w').write(s)
PY
}

# --self-test: a copy of this script in a temporary repository with no tag, beside a stub tools/rotation.py that exits with each
# code. rc 1, 2 and 70 stop at the rotation with a message that names it; rc 0 passes it and stops at the tag check after it, so the
# rotation is the first step and a green one lets the rest run.
if [ "${1:-}" = --self-test ]; then
    [ $# = 1 ] || { echo "usage: tools/release/publish.sh --self-test" >&2; exit 64; }
    t=$(mktemp -d); trap 'rm -rf "$t"' EXIT
    mkdir -p "$t/tools/release"
    cp "$0" "$t/tools/release/publish.sh"
    git -C "$t" init -q
    git -C "$t" -c user.name=selftest -c user.email=selftest@invalid commit -q --allow-empty -m selftest
    cases=0 fails=0
    # expect <stub rotation.py exit code> <message the run must print>: the run must fail, naming it, and print nothing of a later step
    expect() {
        cases=$((cases + 1))
        printf 'import sys\nsys.exit(%s)\n' "$1" > "$t/tools/rotation.py"
        rc=0
        got=$(bash "$t/tools/release/publish.sh" 0.0.0 2>&1) || rc=$?
        case "$rc:$got" in
            1:*"$2"*) ;;
            *) echo "publish.sh self-test: rotation exit $1: rc $rc, want rc 1 with '$2'; got: $got" >&2; fails=$((fails + 1)) ;;
        esac
        case "$got" in
            *"does not carry"*) [ "$1" = 0 ] || { echo "publish.sh self-test: rotation exit $1: the tag step ran after a refusal: $got" >&2; fails=$((fails + 1)); } ;;
        esac
    }
    expect 1 "the rotation is red"
    expect 2 "could not run (rc 2"
    expect 70 "could not run (rc 70"
    expect 0 "does not carry v0.0.0"
    # The asset list (step 4): the sm_86 archive first and the macOS one last, the sm_120a one beside it unless --no-sm120a, the
    # GB10 one when a GB10 archive was verified.
    assets_expect() { # assets_expect <case> <gb10> <sm120a> <want, space-separated>
        cases=$((cases + 1))
        got=$(release_assets 0.0.0 "$2" "$3" | tr '\n' ' ')
        got=${got% }
        [ "$got" = "$4" ] || { echo "publish.sh self-test: assets $1: '$got', want '$4'" >&2; fails=$((fails + 1)); }
    }
    assets_expect "all four" 1 1 "bloomery-0.0.0-linux-x86_64-cuda-sm86 bloomery-0.0.0-linux-x86_64-cuda-sm120a bloomery-0.0.0-linux-aarch64-cuda-sm121a bloomery-0.0.0-macos-arm64-metal"
    assets_expect "no GB10 archive" 0 1 "bloomery-0.0.0-linux-x86_64-cuda-sm86 bloomery-0.0.0-linux-x86_64-cuda-sm120a bloomery-0.0.0-macos-arm64-metal"
    assets_expect "--no-sm120a" 1 0 "bloomery-0.0.0-linux-x86_64-cuda-sm86 bloomery-0.0.0-linux-aarch64-cuda-sm121a bloomery-0.0.0-macos-arm64-metal"
    assets_expect "x86-64 and macOS only" 0 0 "bloomery-0.0.0-linux-x86_64-cuda-sm86 bloomery-0.0.0-macos-arm64-metal"
    # An argument that is no option is refused by name (64) before any step runs; --no-sm120a is an option.
    cases=$((cases + 1))
    rc=0
    got=$(bash "$t/tools/release/publish.sh" 0.0.0 --no-such-option 2>&1) || rc=$?
    case "$rc:$got" in
        64:*"unknown argument --no-such-option"*) ;;
        *) echo "publish.sh self-test: an unknown option: rc $rc, want 64 naming it; got: $got" >&2; fails=$((fails + 1)) ;;
    esac
    cases=$((cases + 1))
    printf 'import sys\nsys.exit(0)\n' > "$t/tools/rotation.py"
    rc=0
    got=$(bash "$t/tools/release/publish.sh" 0.0.0 --no-sm120a 2>&1) || rc=$?
    case "$rc:$got" in
        1:*"does not carry v0.0.0"*) ;;
        *) echo "publish.sh self-test: --no-sm120a: rc $rc, want rc 1 at the tag step; got: $got" >&2; fails=$((fails + 1)) ;;
    esac
    # The tap formula (step 5): a formula with a GB10 url and a release with no GB10 archive is left unchanged with rc 3,
    # never bumped with an older GB10 url under the new version; with a GB10 sum all three urls move.
    z=0000000000000000000000000000000000000000000000000000000000000000
    for k in linux-x86_64-cuda-sm86 linux-aarch64-cuda-sm121a macos-arm64-metal; do
        printf '  url "https://github.com/midagedev/bloomery/releases/download/v0.0.1/bloomery-0.0.1-%s.tar.gz"\n  sha256 "%s"\n' "$k" "$z"
    done > "$t/bloomery.rb.in"
    { echo '  version "0.0.1"'; cat "$t/bloomery.rb.in"; } > "$t/bloomery.rb"; cp "$t/bloomery.rb" "$t/bloomery.rb.orig"
    a=$(printf 'a%.0s' {1..64}); m=$(printf 'b%.0s' {1..64}); g=$(printf 'c%.0s' {1..64})
    cases=$((cases + 1)); rc=0
    got=$(formula_bump "$t/bloomery.rb" 0.0.2 "$a" "$m" "" 2>&1) || rc=$?
    if [ "$rc" != 3 ] || ! cmp -s "$t/bloomery.rb" "$t/bloomery.rb.orig" || [[ "$got" != *"--tap-only"* ]]; then
        echo "publish.sh self-test: tap without a GB10 archive: rc $rc, want rc 3, the formula unchanged and --tap-only named; got: $got" >&2
        fails=$((fails + 1))
    fi
    cases=$((cases + 1)); rc=0
    got=$(formula_bump "$t/bloomery.rb" 0.0.2 "$a" "$m" "$g" 2>&1) || rc=$?
    if [ "$rc" != 0 ] || [ "$(grep -c 'download/v0.0.2/bloomery-0.0.2-' "$t/bloomery.rb")" != 3 ] \
        || ! grep -q "version \"0.0.2\"" "$t/bloomery.rb" || [ "$(grep -c "sha256 \"[$a$m$g]\{64\}\"" "$t/bloomery.rb")" != 3 ]; then
        echo "publish.sh self-test: tap with a GB10 archive: rc $rc, want rc 0 and three urls and sums at 0.0.2; got: $got" >&2
        fails=$((fails + 1))
    fi
    # The sitting step (3a): a verdict the tarball's member matches passes silently, anything else is a WARNING that names it, and
    # the status is 0 in every case — publishing is never refused for it.
    tn=bloomery-0.0.0-linux-x86_64-cuda-sm86
    mkdir -p "$t/sit/$tn/bin" "$t/v-none"
    printf 'the shipped binary\n' > "$t/sit/$tn/bin/bloomery-serve"
    tar -czf "$t/sit/$tn.tar.gz" -C "$t/sit" "$tn"
    msha=$(shasum -a 256 "$t/sit/$tn/bin/bloomery-serve" | awk '{print $1}')
    # verdict <dir> <stage> <result> <sha>: the header lines of tools/release/sitting.sh's verdict that the step reads
    verdict() {
        mkdir -p "$1"
        printf 'sitting-verdict 1\nstage=%s\nresult=%s\nbinary_sha256=%s\n\n== round 1 cell 1 rc=0 result=green\n' "$2" "$3" "$4" > "$1/$tn.sitting"
    }
    # sit_expect <name> <dir> <text the run must print> <text it must not>: rc 0 and the text
    sit_expect() {
        cases=$((cases + 1))
        rc=0
        got=$(bash "$t/tools/release/publish.sh" --sitting-check "$t/sit/$tn.tar.gz" "$tn" "$2" 2>&1) || rc=$?
        case "$rc:$got" in
            0:*"$3"*) ;;
            *) echo "publish.sh self-test: sitting $1: rc $rc, want rc 0 with '$3'; got: $got" >&2; fails=$((fails + 1)) ;;
        esac
        if [ -n "$4" ]; then
            case "$got" in
                *"$4"*) echo "publish.sh self-test: sitting $1: the output holds '$4': $got" >&2; fails=$((fails + 1)) ;;
            esac
        fi
    }
    verdict "$t/v-ok" portable green "$msha"
    sit_expect "verdict present" "$t/v-ok" "sitting verdict ok" "WARNING"
    sit_expect "verdict absent" "$t/v-none" "WARNING: no stage=portable sitting verdict" "sitting verdict ok"
    verdict "$t/v-tree" tree green "$msha"
    sit_expect "stage=tree verdict" "$t/v-tree" "stage=tree" "sitting verdict ok"
    sit_expect "stage=tree verdict warns" "$t/v-tree" "WARNING: no stage=portable sitting verdict" ""
    verdict "$t/v-sha" portable green 0000
    sit_expect "another binary's verdict" "$t/v-sha" "is not the tarball's member" "sitting verdict ok"
    verdict "$t/v-red" portable red "$msha"
    sit_expect "red verdict" "$t/v-red" "result=red" "sitting verdict ok"
    # the publish flow calls the step on the tarball it uploads
    cases=$((cases + 1))
    grep -q '^sitting_check "\$TMP/\$NAME.tar.gz" "\$NAME"' "$t/tools/release/publish.sh" || {
        echo "publish.sh self-test: the publish flow does not call sitting_check on the tarball it uploads" >&2
        fails=$((fails + 1))
    }
    # tree_check (steps 2 and 2b, the builds' first look) and bump_next, on a tree of their own: a copy of this script in a git
    # repository with the serve and gpu-gates manifests, a README and a Cargo.lock. tc_cases runs every case against one copy of the
    # script and counts the red ones in tc_fails; the real script must have none, and each mutant of it must have some.
    tc="$t/tc"
    mk_tree() { # mk_tree <crates' version> <README's version>
        printf '[package]\nname = "bloomery-serve"\nversion = "%s"\nedition = "2024"\n' "$1" > "$tc/crates/serve/Cargo.toml"
        printf '[package]\nname = "bloomery-gpu-gates"\nversion = "%s"\nedition = "2024"\n' "$1" > "$tc/crates/gpu-gates/Cargo.toml"
        printf '# bloomery\n\n## Status (%s)\n\ntar xz bloomery-%s-linux-x86_64-cuda-sm86.tar.gz\n[release](https://example.invalid/releases/tag/v%s)\n' "$2" "$2" "$2" > "$tc/README.md"
        printf 'version = 4\n\n[[package]]\nname = "bloomery-gpu-gates"\nversion = "%s"\ndependencies = [\n "x",\n]\n\n[[package]]\nname = "bloomery-serve"\nversion = "%s"\n\n[[package]]\nname = "x"\nversion = "%s"\n' "$1" "$1" "$1" > "$tc/Cargo.lock"
    }
    tc_want() { # tc_want <case> <want rc> <want text> <text it must not hold or ''> <publish.sh args…>
        local name=$1 want_rc=$2 want=$3 unwanted=$4 out rc=0
        shift 4
        out=$(cd "$tc" && bash tools/release/publish.sh "$@" 2>&1) || rc=$?
        if [ "$rc" != "$want_rc" ] || [[ "$out" != *"$want"* ]] || { [ -n "$unwanted" ] && [[ "$out" == *"$unwanted"* ]]; }; then
            tc_fails=$((tc_fails + 1))
            [ "$tc_loud" = 0 ] || echo "publish.sh self-test: $name: rc $rc (want $want_rc), want '$want'${unwanted:+ without '$unwanted'}; got: $out" >&2
        fi
        [ "$tc_loud" = 0 ] || cases=$((cases + 1))
    }
    tc_cases() { # tc_cases <script copy> <loud 0|1>
        local script=$1 r1 r2 n
        tc_loud=$2 tc_fails=0
        rm -rf "$tc"
        mkdir -p "$tc/tools/release" "$tc/crates/serve" "$tc/crates/gpu-gates"
        cp "$script" "$tc/tools/release/publish.sh"
        printf 'import sys\nsys.exit(0)\n' > "$tc/tools/rotation.py"
        git -C "$tc" init -q
        mk_tree 0.0.1 0.0.1
        git -C "$tc" add -A
        git -C "$tc" -c user.name=selftest -c user.email=selftest@invalid commit -q -m one
        r1=$(git -C "$tc" rev-parse --short=12 HEAD)
        tc_want "tree check: a consistent tree" 0 "tree check ok" "" --tree-check 0.0.1
        tc_want "tree check: a crate behind the release" 1 "crates/serve says version 0.0.1, not 0.0.2 — bump it and commit before the builds" "" --tree-check 0.0.2
        mk_tree 0.0.2 0.0.1
        git -C "$tc" add -A
        git -C "$tc" -c user.name=selftest -c user.email=selftest@invalid commit -q -m two
        git -C "$tc" tag v0.0.2
        r2=$(git -C "$tc" rev-parse --short=12 HEAD)
        tc_want "tree check: a README that names the last release's status" 1 "README.md does not name '## Status (0.0.2)' — update its status heading and install lines and commit before the builds" "tree check ok" --tree-check 0.0.2
        tc_want "tree check: a rev's tree, not the working tree's" 0 "tree check ok" "" --tree-check 0.0.1 "$r1"
        tc_want "tree check: the rev named in the refusal" 1 "crates/serve says version 0.0.2 at $r2, not 0.0.1" "" --tree-check 0.0.1 "$r2"
        tc_want "tree check: a rev that is no commit" 1 "cannot read crates/serve/Cargo.toml at no-such-rev" "" --tree-check 0.0.1 no-such-rev
        printf '# bloomery\n\n## Status (0.0.2)\n\n[release](https://example.invalid/releases/tag/v0.0.2)\n' > "$tc/README.md"
        tc_want "tree check: a README that does not name the tarball" 1 "does not name 'bloomery-0.0.2-linux-x86_64-cuda-sm86.tar.gz'" "" --tree-check 0.0.2
        git -C "$tc" checkout -q -- README.md
        tc_want "tree check: no version at all" 64 "usage: tools/release/publish.sh --tree-check" "" --tree-check
        # The flow refuses at the README before it reaches the box: the archives are not built or fetched for a tree that cannot publish.
        tc_want "flow: the README check comes before the archives" 1 "README.md does not name '## Status (0.0.2)'" "box.sh" 0.0.2 --dry-run
        # bump_next: the three files move together, and a tree that disagrees is not touched.
        mk_tree 0.0.1 0.0.1
        cp "$tc/Cargo.lock" "$tc/lock.before"
        tc_want "bump: the next patch version" 0 "git commit -m 'version: main moves to 0.0.2 after the 0.0.1 release' -- Cargo.lock crates/serve/Cargo.toml crates/gpu-gates/Cargo.toml" "" --bump-next 0.0.1
        n=$(diff "$tc/lock.before" "$tc/Cargo.lock" | grep -c '^[<>]' || true)
        n=$n/$(grep -c 'version = "0.0.2"' "$tc/crates/serve/Cargo.toml" "$tc/crates/gpu-gates/Cargo.toml" | tr '\n' ' ')
        if [ "$n" != "4/$tc/crates/serve/Cargo.toml:1 $tc/crates/gpu-gates/Cargo.toml:1 " ]; then
            tc_fails=$((tc_fails + 1))
            [ "$tc_loud" = 0 ] || echo "publish.sh self-test: bump: the lock changed lines and the manifests' new versions are '$n', want 4 and one each" >&2
        fi
        [ "$tc_loud" = 0 ] || cases=$((cases + 1))
        mk_tree 0.0.1 0.0.1
        cp -R "$tc/crates" "$tc/crates.before"
        sed '/^name = "bloomery-gpu-gates"$/{n;s/0\.0\.1/0.0.0/;}' "$tc/Cargo.lock" > "$tc/Cargo.lock.new" && mv "$tc/Cargo.lock.new" "$tc/Cargo.lock"
        tc_want "bump: a Cargo.lock that disagrees writes nothing" 1 "Cargo.lock's bloomery-gpu-gates entry says version 0.0.0, not 0.0.1: nothing was written" "" --bump-next 0.0.1
        if ! diff -r "$tc/crates" "$tc/crates.before" > /dev/null; then
            tc_fails=$((tc_fails + 1))
            [ "$tc_loud" = 0 ] || echo "publish.sh self-test: bump: a refused bump changed a manifest" >&2
        fi
        [ "$tc_loud" = 0 ] || cases=$((cases + 1))
        mk_tree 0.0.1 0.0.1
        tc_want "bump: a tree that says another version" 1 "crates/serve/Cargo.toml says version 0.0.1, not 0.0.7: nothing was written" "" --bump-next 0.0.7
        tc_want "bump: a version that is no MAJOR.MINOR.PATCH" 64 "takes MAJOR.MINOR.PATCH, got '0.1'" "" --bump-next 0.1
        tc_want "bump: an explicit next version" 0 "moved from 0.0.1 to 0.1.0" "" --bump-next 0.0.1 0.1.0
        tc_want "bump: the same version again" 64 "the next version is the current one" "" --bump-next 0.1.0 0.1.0
    }
    # mutant_publish <name> <old text> <new text>: this script with the first <old text> replaced; its tc_cases must go red. Each
    # <old text> is the code's own: the first occurrence in the file, or one with a newline round it, which this self-test spells \n.
    mutant_publish() {
        cp "$0" "$t/mutant.sh"
        python3 -I - "$t/mutant.sh" "$2" "$3" << 'PY' || { echo "publish.sh self-test: mutant $1: the text to replace is not in the code" >&2; fails=$((fails + 1)); return 0; }
import sys

path, old, new = sys.argv[1:4]
s = open(path).read()
i = s.find(old)
if i < 0:
    sys.exit(1)
open(path, "w").write(s[:i] + new + s[i + len(old):])
PY
        cases=$((cases + 1))
        tc_cases "$t/mutant.sh" 0
        if [ "$tc_fails" = 0 ]; then
            echo "publish.sh self-test: mutant $1 left every tree and bump case green: the cases do not hold it" >&2
            fails=$((fails + 1))
        fi
    }
    tc_cases "$0" 1
    if [ "$tc_fails" != 0 ]; then fails=$((fails + tc_fails)); fi
    mutant_publish "no README look" 'grep -qF -- "$want" <<< "$text" ||' ': ||'
    mutant_publish "no version look" '[ "$v" = "$version" ] ||' '[ -n "$v" ] ||'
    mutant_publish "flow without the tree check" $'\ntree_check "$VERSION" || exit 1\n' $'\n:\n'
    mutant_publish "lock left behind" 'lock = lock[:m.start(1)] + nxt + lock[m.end(1):]' 'pass'
    if [ "$fails" = 0 ]; then
        echo "publish.sh: self-test ok ($cases cases)"
        exit 0
    fi
    echo "publish.sh: self-test: $fails failure(s) in $cases cases" >&2
    exit 1
fi

if [ "${1:-}" = --sitting-check ]; then
    [ $# -ge 4 ] || { echo "usage: tools/release/publish.sh --sitting-check <tarball> <name> <dir>..." >&2; exit 64; }
    shift
    sitting_check "$@"
    exit 0
fi

if [ "${1:-}" = --tree-check ]; then
    [ $# -ge 2 ] && [ $# -le 3 ] || { echo "usage: tools/release/publish.sh --tree-check <version> [<rev>]" >&2; exit 64; }
    tree_check "$2" "${3:-}" || exit 1
    echo "publish.sh: tree check ok: crates/serve and crates/gpu-gates say $2, README.md names its status, tarball and link${3:+ (at $3)}"
    exit 0
fi

if [ "${1:-}" = --bump-next ]; then
    [ $# -ge 2 ] && [ $# -le 3 ] || { echo "usage: tools/release/publish.sh --bump-next <version> [<next>]" >&2; exit 64; }
    bump_next "$2" "${3:-}"
    exit $?
fi

VERSION=${1:?usage: tools/release/publish.sh <version> [--dry-run] [--no-tap] [--no-sm120a] | <version> --tap-only | --self-test | --sitting-check <tarball> <name> <dir>... | --tree-check <version> [<rev>] | --bump-next <version> [<next>]}
DRY=0; TAP=1; BW=1; TAP_ONLY=0
for a in "${@:2}"; do
    case "$a" in
        --dry-run) DRY=1 ;;
        --no-tap) TAP=0 ;;
        --no-sm120a) BW=0 ;;
        --tap-only) TAP_ONLY=1 ;;
        *) echo "publish.sh: unknown argument $a" >&2; exit 64 ;;
    esac
done
REPO=midagedev/bloomery
BOX=${BLOOMERY_BOX:-ws}
REMOTE=${BLOOMERY_REMOTE:-"~/repo/$(basename "$ROOT")"}
GB10_BOX=${BLOOMERY_GB10_BOX:-}
NAME="bloomery-$VERSION-linux-x86_64-cuda-sm86"
BW_NAME="bloomery-$VERSION-linux-x86_64-cuda-sm120a"
GB10_NAME="bloomery-$VERSION-linux-aarch64-cuda-sm121a"
MAC_NAME="bloomery-$VERSION-macos-arm64-metal"
TAG="v$VERSION"

fail() { echo "publish.sh: $*" >&2; exit 1; }

# tap_bump: step 5 on SHA, MAC_SHA and GB10_SHA (GB10_SHA empty: the release carries no GB10 archive yet).
tap_bump() {
    TAPDIR="$HOME/repo/homebrew-tap"
    if [ "$TAP" = 1 ]; then
        if [ "$DRY" = 1 ]; then
            echo "publish.sh: dry run: would bump $TAPDIR/Formula/bloomery.rb to $VERSION / $SHA (linux) / ${GB10_SHA:-none} (gb10) / $MAC_SHA (macos) and push"
        else
            [ -d "$TAPDIR/Formula" ] || fail "no tap checkout at $TAPDIR (clone midagedev/homebrew-tap)"
            git -C "$TAPDIR" fetch -q origin && git -C "$TAPDIR" reset -q --hard origin/HEAD
            [ -z "$(git -C "$TAPDIR" status --porcelain)" ] || fail "the tap checkout is dirty"
            F="$TAPDIR/Formula/bloomery.rb"
            rc=0
            formula_bump "$F" "$VERSION" "$SHA" "$MAC_SHA" "$GB10_SHA" || rc=$?
            [ "$rc" = 3 ] && return 0
            [ "$rc" = 0 ] || fail "the formula bump failed (rc $rc)"
            git -C "$TAPDIR" add Formula/bloomery.rb
            git -C "$TAPDIR" commit -q -m "bloomery v$VERSION (linux sha256 $SHA, macos sha256 $MAC_SHA${GB10_SHA:+, gb10 sha256 $GB10_SHA})"
            git -C "$TAPDIR" push -q origin HEAD
            echo "publish.sh: tap bumped to $VERSION"
        fi
    fi
}

# --tap-only: step 5 alone, on the release's own .sha256 files, each held to GitHub's digest of its archive (an
# upload from another machine, the GB10's, is read the way the tap will fetch it).
if [ "$TAP_ONLY" = 1 ]; then
    TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
    DIGESTS=$(gh api "repos/$REPO/releases/tags/$TAG" --jq '.assets[] | "\(.name) \(.digest)"') || fail "no release $TAG"
    asset_sha() { # asset_sha <name>: the archive's sha256 from its .sha256 asset, equal to GitHub's digest; empty when absent
        local want got
        want=$(awk -v n="$1.tar.gz" '$1 == n {sub(/^sha256:/, "", $2); print $2}' <<<"$DIGESTS")
        [ -n "$want" ] || return 0
        gh release download "$TAG" --repo "$REPO" -D "$TMP" -p "$1.tar.gz.sha256" --clobber >&2 || fail "no $1.tar.gz.sha256 on $TAG"
        got=$(grep -o '[0-9a-f]\{64\}' "$TMP/$1.tar.gz.sha256" | head -1)
        [ "$got" = "$want" ] || fail "$1.tar.gz.sha256 names $got, GitHub's digest of the archive is $want"
        echo "$got"
    }
    SHA=$(asset_sha "$NAME"); MAC_SHA=$(asset_sha "$MAC_NAME"); GB10_SHA=$(asset_sha "$GB10_NAME")
    [ -n "$SHA" ] || fail "$TAG carries no $NAME.tar.gz"
    [ -n "$MAC_SHA" ] || fail "$TAG carries no $MAC_NAME.tar.gz"
    TAP=1
    tap_bump
    exit 0
fi

# 0. the rotation
rotation_gate || exit 1

# 1. the tag
COMMIT=$(git rev-parse --short=12 HEAD)
[ -z "$(git status --porcelain --untracked-files=no)" ] || fail "the tree has uncommitted changes"
git tag -l --points-at HEAD | grep -qx "$TAG" || fail "HEAD does not carry $TAG (tag the release first)"

# 2 and 2b. the versions (--version reads gpu-gates, /props reads serve) and the README's status heading, tarball and link; the
# builds ran the same check first, so a refusal here means the tree moved after them
tree_check "$VERSION" || exit 1

# 3. the Linux artifacts, each verified on the machine that built it (readelf/objdump and the ELF binary are Linux's)
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
# linux_archive <box> <name> <arch>: <name>'s .sha256, build.sh --check (for the archive of <arch>) and --version on <box>;
# then the tarball and .sha256 fetched into $TMP and hashed here. Prints the sha256.
linux_archive() {
    local box=$1 name=$2 arch=$3 sha got f
    local verify=(ssh "$box")
    # The box's command goes behind its guard; the GB10's runs over plain ssh, as the account that built it (box.sh's
    # guard reads /root's lease, which that account need not reach).
    [ "$box" != "$BOX" ] || verify=(bash tools/box.sh)
    BLOOMERY_BOX=$box "${verify[@]}" "set -e; cd $REMOTE/target/release-dist; \
sha256sum -c $name.tar.gz.sha256 > /dev/null; \
rm -rf /tmp/pub-$VERSION; mkdir /tmp/pub-$VERSION; \
tar -xzf $name.tar.gz -C /tmp/pub-$VERSION --strip-components=1; \
bash $REMOTE/tools/release/build.sh --check /tmp/pub-$VERSION/bin $arch; \
V=\$(/tmp/pub-$VERSION/bin/bloomery-serve --version); \
[ \"\$V\" = \"bloomery-serve $VERSION (commit $COMMIT)\" ] || { echo \"the packed binary says: \$V\" >&2; exit 1; }" >&2 \
        || fail "$name failed verification on $box (just release-build $VERSION there first)"
    for f in "$name.tar.gz" "$name.tar.gz.sha256"; do
        scp -q "$box:$REMOTE/target/release-dist/$f" "$TMP/$f" || fail "cannot fetch $f from $box"
    done
    sha=$(grep -o '[0-9a-f]\{64\}' "$TMP/$name.tar.gz.sha256" | head -1)
    [ -n "$sha" ] || fail "$name's .sha256 holds no digest"
    got=$(shasum -a 256 "$TMP/$name.tar.gz" | awk '{print $1}')
    [ "$got" = "$sha" ] || fail "the fetched $name.tar.gz hashes $got, its .sha256 says $sha"
    echo "publish.sh: $name.tar.gz $(stat -f %z "$TMP/$name.tar.gz") bytes, sha256 $sha, checks and --version ok on $box" >&2
    echo "$sha"
}
SHA=$(linux_archive "$BOX" "$NAME" sm_86) || exit 1
BW_SHA=
if [ "$BW" = 1 ]; then
    BW_SHA=$(linux_archive "$BOX" "$BW_NAME" sm_120a) || exit 1
fi
GB10_SHA=
if [ -n "$GB10_BOX" ]; then
    GB10_SHA=$(linux_archive "$GB10_BOX" "$GB10_NAME" sm_121a) || exit 1
fi

# 3a. the sitting's verdict beside the tarball on the box: its absence is the check's warning, not an error here
scp -q "$BOX:$REMOTE/target/release-dist/$NAME.sitting" "$TMP/$NAME.sitting" 2> /dev/null || true
sitting_check "$TMP/$NAME.tar.gz" "$NAME" "$TMP" ${PUBLISH_SITTING_DIR:+"$PUBLISH_SITTING_DIR"}

# 3b. the Apple Silicon artifacts, built here by tools/release/build-mac.sh
for f in "$MAC_NAME.tar.gz" "$MAC_NAME.tar.gz.sha256"; do
    [ -f "target/mac-release/dist/$f" ] || fail "no target/mac-release/dist/$f (MAC_SIGN_IDENTITY=… MAC_NOTARY_PROFILE=… tools/release/build-mac.sh $VERSION first)"
    cp "target/mac-release/dist/$f" "$TMP/$f"
done
MAC_SHA=$(grep -o '[0-9a-f]\{64\}' "$TMP/$MAC_NAME.tar.gz.sha256" | head -1)
[ "$(shasum -a 256 "$TMP/$MAC_NAME.tar.gz" | awk '{print $1}')" = "$MAC_SHA" ] || fail "$MAC_NAME.tar.gz does not hash to its .sha256"
mkdir "$TMP/mac" && tar -xzf "$TMP/$MAC_NAME.tar.gz" -C "$TMP/mac" --strip-components=1
bash tools/release/build-mac.sh --check "$TMP/mac/bin" || fail "the Apple Silicon binaries failed build-mac.sh --check"
for b in "$TMP"/mac/bin/*; do
    sig=$(codesign -dvv "$b" 2>&1)
    grep -q '^Authority=Developer ID Application' <<< "$sig" || fail "$(basename "$b") is not signed by a Developer ID (build-mac.sh without MAC_SIGN_IDENTITY)"
done
grep -qF "$(git rev-parse HEAD)" "$TMP/mac/README.md" || fail "$MAC_NAME's README names another commit than HEAD"
echo "publish.sh: $MAC_NAME.tar.gz $(stat -f %z "$TMP/$MAC_NAME.tar.gz") bytes, sha256 $MAC_SHA, checks and Developer ID ok"

# 4. the upload. gh uploads one call's files in parallel, and the image workflow starts when the x86-64 .sha256 is up,
# so that file goes up alone after every other one.
GB10=0; [ -z "$GB10_SHA" ] || GB10=1
ASSETS=()
while IFS= read -r a; do ASSETS+=("$a"); done < <(release_assets "$VERSION" "$GB10" "$BW")
FILES=()
for a in "${ASSETS[@]}"; do FILES+=("$a.tar.gz" "$a.tar.gz.sha256"); done
if [ "$DRY" = 1 ]; then
    echo "publish.sh: dry run: would upload ${ASSETS[*]} (.tar.gz and .sha256) to $TAG and curl every URL"
else
    (cd "$TMP" && gh release upload "$TAG" --repo "$REPO" --clobber "$NAME.tar.gz" "${FILES[@]:2}" \
        && gh release upload "$TAG" --repo "$REPO" --clobber "$NAME.tar.gz.sha256") \
        || fail "gh release upload failed (does release $TAG exist? gh release create $TAG first)"
    for u in "${FILES[@]}"; do
        code=$(curl -fsSI -o /dev/null -w '%{http_code}' "https://github.com/$REPO/releases/download/$TAG/$u" || echo err)
        [ "$code" = 302 ] || [ "$code" = 200 ] || fail "the asset URL answers $code: $u"
    done
    echo "publish.sh: uploaded and every URL resolves"
fi

# 5. the tap
tap_bump

# 6. the lead's remainder
echo "publish.sh: left to the lead:"
[ -n "$GB10_SHA" ] || echo "  - no GB10 archive in this release: BLOOMERY_GB10_BOX names the GB10 that ran 'just release-build $VERSION'"
[ -n "$BW_SHA" ] || echo "  - no sm_120a archive in this release (--no-sm120a): the installer takes the sm_86 archive on an RTX 50 series card"
[ -z "$BW_SHA" ] || echo "  - the sm_120a archive is install.sh's pick on a machine whose every GPU is compute capability 12.0 (sha256 $BW_SHA); Homebrew and the ghcr image stay on the sm_86 archive"
echo "  - the next version on main (the tarballs built there before the next ask are that release's): tools/release/publish.sh --bump-next $VERSION, then the commit it prints"
echo "  - release notes: gh release edit $TAG --repo $REPO --notes-file <file>"
echo "  - the ghcr image builds itself from the tag push (.github/workflows/container.yml); watch:"
echo "      gh run watch \$(gh run list --repo $REPO --workflow 'Docker image' --limit 1 --json databaseId -q '.[0].databaseId')"
