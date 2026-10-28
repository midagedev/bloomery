#!/usr/bin/env bash
# Publishes a built release: `just release-build <version>` makes the tarball on the
# box; this script is everything after it, so the upload cannot repeat v0.2.0's
# mistake (gh named the assets from temp filenames and every download URL 404'd).
#
#   tools/release/publish.sh <version> [--dry-run] [--no-tap] [--no-sm120a]
#   tools/release/publish.sh --self-test     (the rotation's refusal on stub rotation.py exits, and step 3a's warnings; check-recipes runs it)
#   tools/release/publish.sh --sitting-check <tarball> <name> <dir>...   step 3a alone, on the verdicts in each <dir>
#
# What it owns, in order:
#   0. the rotation: `python3 tools/rotation.py check` must be green (every F family has a green ledger row inside the
#      rotation's window: the release batch leaves those gates to the rotation); rc 1 stops here, and so does rc 2 or 70, each named;
#   1. the tag: HEAD must carry tag v<version> (the release is the tag; the binaries'
#      --version names this commit);
#   2. the versions: the serve AND gpu-gates crates must both say <version> (--version
#      reads gpu-gates, /props reads serve — v0.2.0 shipped with 0.1.0 in --version);
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
#      archive is the installer's pick, never the formula's), commits and pushes;
#   6. prints what is left to the lead: the release notes (gh release edit --notes-file)
#      and the ghcr image — the v* tag push already triggered .github/workflows/container.yml.
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

VERSION=${1:?usage: tools/release/publish.sh <version> [--dry-run] [--no-tap] [--no-sm120a] | --self-test | --sitting-check <tarball> <name> <dir>...}
DRY=0; TAP=1; BW=1
for a in "${@:2}"; do
    case "$a" in
        --dry-run) DRY=1 ;;
        --no-tap) TAP=0 ;;
        --no-sm120a) BW=0 ;;
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

# 0. the rotation
rotation_gate || exit 1

# 1. the tag
COMMIT=$(git rev-parse --short=12 HEAD)
[ -z "$(git status --porcelain --untracked-files=no)" ] || fail "the tree has uncommitted changes"
git tag -l --points-at HEAD | grep -qx "$TAG" || fail "HEAD does not carry $TAG (tag the release first)"

# 2. the versions: --version reads gpu-gates, /props reads serve
for c in serve gpu-gates; do
    v=$(sed -n 's/^version = "\(.*\)"/\1/p' "crates/$c/Cargo.toml" | head -1)
    [ "$v" = "$VERSION" ] || fail "crates/$c says version $v, not $VERSION — bump it, commit, re-tag, release-build again"
done

# 2b. the README names this release: its status heading, the tarball it unpacks and the release link
for want in "## Status ($VERSION)" "$NAME.tar.gz" "releases/tag/$TAG)"; do
    grep -qF -- "$want" README.md || fail "README.md does not name '$want' — update its status heading and install lines, commit, re-tag"
done

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
TAPDIR="$HOME/repo/homebrew-tap"
if [ "$TAP" = 1 ]; then
    if [ "$DRY" = 1 ]; then
        echo "publish.sh: dry run: would bump $TAPDIR/Formula/bloomery.rb to $VERSION / $SHA (linux) / ${GB10_SHA:-none} (gb10) / $MAC_SHA (macos) and push"
    else
        [ -d "$TAPDIR/Formula" ] || fail "no tap checkout at $TAPDIR (clone midagedev/homebrew-tap)"
        git -C "$TAPDIR" fetch -q origin && git -C "$TAPDIR" reset -q --hard origin/HEAD
        [ -z "$(git -C "$TAPDIR" status --porcelain)" ] || fail "the tap checkout is dirty"
        F="$TAPDIR/Formula/bloomery.rb"
        python3 - "$F" "$VERSION" "$SHA" "$MAC_SHA" "$GB10_SHA" <<'PY'
import re, sys
f, v, sha, mac_sha, gb10_sha = sys.argv[1:6]
s = open(f).read()
s, n = re.subn(r'version "[^"]*"', f'version "{v}"', s, count=1)
assert n == 1, "no version line"
# Each platform's url and the sha256 on the line after it. The GB10 url is the formula's choice: bumped where it has
# one, and a release without a GB10 archive is refused there, since its url would keep naming an older release.
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
    sys.exit(f"publish.sh: the tap's formula names a {gb10} archive and this release has none (BLOOMERY_GB10_BOX)")
open(f, 'w').write(s)
PY
        git -C "$TAPDIR" add Formula/bloomery.rb
        git -C "$TAPDIR" commit -q -m "bloomery v$VERSION (linux sha256 $SHA, macos sha256 $MAC_SHA)"
        git -C "$TAPDIR" push -q origin HEAD
        echo "publish.sh: tap bumped to $VERSION"
    fi
fi

# 6. the lead's remainder
echo "publish.sh: left to the lead:"
[ -n "$GB10_SHA" ] || echo "  - no GB10 archive in this release: BLOOMERY_GB10_BOX names the GB10 that ran 'just release-build $VERSION'"
[ -n "$BW_SHA" ] || echo "  - no sm_120a archive in this release (--no-sm120a): the installer takes the sm_86 archive on an RTX 50 series card"
[ -z "$BW_SHA" ] || echo "  - the sm_120a archive is install.sh's pick on a machine whose every GPU is compute capability 12.0 (sha256 $BW_SHA); Homebrew and the ghcr image stay on the sm_86 archive"
echo "  - release notes: gh release edit $TAG --repo $REPO --notes-file <file>"
echo "  - the ghcr image builds itself from the tag push (.github/workflows/container.yml); watch:"
echo "      gh run watch \$(gh run list --repo $REPO --workflow 'Docker image' --limit 1 --json databaseId -q '.[0].databaseId')"
