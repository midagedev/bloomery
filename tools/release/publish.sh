#!/usr/bin/env bash
# Publishes a built release: `just release-build <version>` makes the tarball on the
# box; this script is everything after it, so the upload cannot repeat v0.2.0's
# mistake (gh named the assets from temp filenames and every download URL 404'd).
#
#   tools/release/publish.sh <version> [--dry-run] [--no-tap]
#
# What it owns, in order:
#   1. the tag: HEAD must carry tag v<version> (the release is the tag; the binaries'
#      --version names this commit);
#   2. the versions: the serve AND gpu-gates crates must both say <version> (--version
#      reads gpu-gates, /props reads serve — v0.2.0 shipped with 0.1.0 in --version);
#   3. the artifacts: on the box, where readelf and the ELF binary run — the release's
#      own .sha256, build.sh's four checks on the unpacked binary, and its --version;
#      then the tarball and .sha256 come here for the upload and the tap's digest;
#   4. the upload: `gh release upload --clobber` with exactly the names build.sh
#      wrote (bloomery-<v>-linux-x86_64-cuda-sm86.tar.gz and .sha256) — it refuses
#      to upload a file under any other name — then curls both URLs for a 200;
#   5. the Homebrew tap: bumps Formula/bloomery.rb's version, url and sha256 in
#      ~/repo/homebrew-tap, commits and pushes;
#   6. prints what is left to the lead: the release notes (gh release edit --notes-file)
#      and the ghcr image — the v* tag push already triggered .github/workflows/container.yml.
#
# --dry-run does 1–3 and prints 4–6 without touching the release or the tap.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
VERSION=${1:?usage: tools/release/publish.sh <version> [--dry-run] [--no-tap]}
DRY=0; TAP=1
for a in "${@:2}"; do
    case "$a" in
        --dry-run) DRY=1 ;;
        --no-tap) TAP=0 ;;
        *) echo "publish.sh: unknown argument $a" >&2; exit 64 ;;
    esac
done
REPO=midagedev/bloomery
BOX=${BLOOMERY_BOX:-ws}
REMOTE=${BLOOMERY_REMOTE:-"~/repo/$(basename "$ROOT")"}
NAME="bloomery-$VERSION-linux-x86_64-cuda-sm86"
TAG="v$VERSION"

fail() { echo "publish.sh: $*" >&2; exit 1; }

# 1. the tag
COMMIT=$(git rev-parse --short=12 HEAD)
[ -z "$(git status --porcelain --untracked-files=no)" ] || fail "the tree has uncommitted changes"
git tag -l --points-at HEAD | grep -qx "$TAG" || fail "HEAD does not carry $TAG (tag the release first)"

# 2. the versions: --version reads gpu-gates, /props reads serve
for c in serve gpu-gates; do
    v=$(sed -n 's/^version = "\(.*\)"/\1/p' "crates/$c/Cargo.toml" | head -1)
    [ "$v" = "$VERSION" ] || fail "crates/$c says version $v, not $VERSION — bump it, commit, re-tag, release-build again"
done

# 3. the artifacts, verified on the box (readelf/objdump and the ELF binary are Linux's)
bash tools/box.sh "set -e; cd $REMOTE/target/release-dist; \
sha256sum -c $NAME.tar.gz.sha256 > /dev/null; \
rm -rf /tmp/pub-$VERSION; mkdir /tmp/pub-$VERSION; \
tar -xzf $NAME.tar.gz -C /tmp/pub-$VERSION --strip-components=1; \
bash $REMOTE/tools/release/build.sh --check /tmp/pub-$VERSION/bin; \
V=\$(/tmp/pub-$VERSION/bin/bloomery-serve --version); \
[ \"\$V\" = \"bloomery-serve $VERSION (commit $COMMIT)\" ] || { echo \"the packed binary says: \$V\" >&2; exit 1; }" \
    || fail "the artifacts failed verification on the box (just release-build $VERSION first)"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
for f in "$NAME.tar.gz" "$NAME.tar.gz.sha256"; do
    scp -q "$BOX:$REMOTE/target/release-dist/$f" "$TMP/$f" || fail "cannot fetch $f from the box"
done
SHA=$(grep -o '[0-9a-f]\{64\}' "$TMP/$NAME.tar.gz.sha256" | head -1)
[ -n "$SHA" ] || fail "the release's .sha256 holds no digest"
GOT=$(shasum -a 256 "$TMP/$NAME.tar.gz" | awk '{print $1}')
[ "$GOT" = "$SHA" ] || fail "the fetched tarball hashes $GOT, the release's .sha256 says $SHA"
echo "publish.sh: $NAME.tar.gz $(stat -f %z "$TMP/$NAME.tar.gz") bytes, sha256 $SHA, checks and --version ok on the box"

# 4. the upload
if [ "$DRY" = 1 ]; then
    echo "publish.sh: dry run: would upload $NAME.tar.gz and .sha256 to $TAG and curl both URLs"
else
    gh release upload "$TAG" --repo "$REPO" --clobber "$TMP/$NAME.tar.gz" "$TMP/$NAME.tar.gz.sha256" \
        || fail "gh release upload failed (does release $TAG exist? gh release create $TAG first)"
    for u in "$NAME.tar.gz" "$NAME.tar.gz.sha256"; do
        code=$(curl -fsSI -o /dev/null -w '%{http_code}' "https://github.com/$REPO/releases/download/$TAG/$u" || echo err)
        [ "$code" = 302 ] || [ "$code" = 200 ] || fail "the asset URL answers $code: $u"
    done
    echo "publish.sh: uploaded and both URLs resolve"
fi

# 5. the tap
TAPDIR="$HOME/repo/homebrew-tap"
if [ "$TAP" = 1 ]; then
    if [ "$DRY" = 1 ]; then
        echo "publish.sh: dry run: would bump $TAPDIR/Formula/bloomery.rb to $VERSION / $SHA and push"
    else
        [ -d "$TAPDIR/Formula" ] || fail "no tap checkout at $TAPDIR (clone midagedev/homebrew-tap)"
        git -C "$TAPDIR" fetch -q origin && git -C "$TAPDIR" reset -q --hard origin/HEAD
        [ -z "$(git -C "$TAPDIR" status --porcelain)" ] || fail "the tap checkout is dirty"
        F="$TAPDIR/Formula/bloomery.rb"
        python3 - "$F" "$VERSION" "$SHA" <<'PY'
import re, sys
f, v, sha = sys.argv[1:4]
s = open(f).read()
s2, n1 = re.subn(r'version "[^"]*"', f'version "{v}"', s, count=1)
s2, n2 = re.subn(r'sha256 "[0-9a-f]{64}"', f'sha256 "{sha}"', s2, count=1)
s2, n3 = re.subn(r'releases/download/v[^/]*/bloomery-[^"/]*\.tar\.gz',
                 f'releases/download/v{v}/bloomery-{v}-linux-x86_64-cuda-sm86.tar.gz', s2, count=1)
assert n1 == n2 == n3 == 1, (n1, n2, n3)
open(f, 'w').write(s2)
PY
        git -C "$TAPDIR" add Formula/bloomery.rb
        git -C "$TAPDIR" commit -q -m "bloomery v$VERSION (sha256 $SHA)"
        git -C "$TAPDIR" push -q origin HEAD
        echo "publish.sh: tap bumped to $VERSION"
    fi
fi

# 6. the lead's remainder
echo "publish.sh: left to the lead:"
echo "  - release notes: gh release edit $TAG --repo $REPO --notes-file <file>"
echo "  - the ghcr image builds itself from the tag push (.github/workflows/container.yml); watch:"
echo "      gh run watch \$(gh run list --repo $REPO --workflow 'Docker image' --limit 1 --json databaseId -q '.[0].databaseId')"
