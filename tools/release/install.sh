#!/bin/sh
# bloomery's installer, made to be run as
#
#   curl -fsSL https://raw.githubusercontent.com/midagedev/bloomery/main/tools/release/install.sh | sh
#
# or from a checkout (tools/release/install.sh [--version V]). No prompts, no
# sudo: the release lands in ~/.local/opt/bloomery-<version> and a symlink
# bloomery-serve goes into ~/.local/bin (created; a PATH note prints when the
# directory is not on PATH). Everything the download needs is checked against
# the release's own sha256 before anything is extracted over.
set -eu

REPO=midagedev/bloomery
VERSION=${VERSION:-}
PREFIX=${PREFIX:-"$HOME/.local"}
while [ $# -gt 0 ]; do
    case "$1" in
        --version) VERSION=$2; shift 2 ;;
        --prefix) PREFIX=$2; shift 2 ;;
        *) echo "install.sh: unknown argument $1 (takes --version V, --prefix DIR)" >&2; exit 64 ;;
    esac
done

fail() { echo "install.sh: $*" >&2; exit 1; }

[ "$(uname -s)" = "Linux" ] || fail "this release is Linux x86-64 (Windows: run it inside WSL2; macOS: no build)."
[ "$(uname -m)" = "x86_64" ] || fail "this release is x86-64; uname -m says $(uname -m)."

# The release's host code is built for x86-64-v3: on a CPU without AVX2, FMA or BMI2 it dies of
# SIGILL ("Illegal instruction") at the first such instruction, so the install stops here instead.
FLAGS=$(grep -m1 '^flags' /proc/cpuinfo 2> /dev/null || true)
if [ -n "$FLAGS" ]; then
    MISSING=
    for f in avx2 fma bmi2; do
        case " $FLAGS " in
            *" $f "*) ;;
            *) MISSING="$MISSING $f" ;;
        esac
    done
    [ -z "$MISSING" ] || fail "this CPU lacks${MISSING}, which the release's x86-64-v3 code needs (a virtual machine: pass the host's CPU type through)."
else
    echo "install.sh: could not read the CPU flags (/proc/cpuinfo); the release needs AVX2, FMA and BMI2 (x86-64-v3)." >&2
fi

GLIBC=$(ldd --version 2>/dev/null | head -1 | grep -o '[0-9]*\.[0-9]*$' || true)
if [ -n "$GLIBC" ]; then
    OLDER=$(printf '%s\n2.34\n' "$GLIBC" | sort -V | head -1)
    [ "$OLDER" = "2.34" ] || fail "glibc $GLIBC is older than the release's floor, 2.34."
else
    echo "install.sh: could not read the glibc version (ldd); continuing." >&2
fi

command -v curl > /dev/null || fail "curl is needed for the download."
command -v tar > /dev/null || fail "tar is needed for the archive."
if ! command -v nvidia-smi > /dev/null; then
    echo "install.sh: note: nvidia-smi is not on PATH — this build needs an NVIDIA GPU of compute" >&2
    echo "capability 8.6 or newer and its driver." >&2
fi

# The version to install: the given one, else the release the API names latest.
if [ -z "$VERSION" ]; then
    VERSION=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
        | grep -m1 '"tag_name"' | grep -o 'v[0-9.]*') \
        || fail "could not read the latest release from GitHub."
fi
ASSET="bloomery-${VERSION#v}-linux-x86_64-cuda-sm86.tar.gz"
URL="https://github.com/$REPO/releases/download/$VERSION/$ASSET"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
echo "downloading $URL"
curl -fSL "$URL" -o "$TMP/$ASSET"
curl -fsSL "$URL.sha256" -o "$TMP/$ASSET.sha256" || fail "no sha256 beside the release."

# The digest alone from the release's own file, rewritten with the name we
# downloaded under, so sha256sum -c / shasum -c take it on every distro.
SUM=$(grep -o "[0-9a-f]\{64\}" "$TMP/$ASSET.sha256" | head -1)
[ -n "$SUM" ] || fail "the release's .sha256 file holds no digest."
echo "$SUM  $ASSET" > "$TMP/$ASSET.sum"
if command -v sha256sum > /dev/null; then
    (cd "$TMP" && sha256sum -c "$ASSET.sum" > /dev/null) || fail "the sha256 does not match the release's."
elif command -v shasum > /dev/null; then
    (cd "$TMP" && shasum -a 256 -c "$ASSET.sum" > /dev/null) || fail "the sha256 does not match the release's."
else
    fail "neither sha256sum nor shasum is on PATH; the download cannot be checked."
fi

OPT="$PREFIX/opt/bloomery-${VERSION#v}"
mkdir -p "$OPT" "$PREFIX/bin"
rm -rf "$OPT.tmp"
mkdir "$OPT.tmp"
tar -xzf "$TMP/$ASSET" -C "$OPT.tmp" --strip-components=1
rm -rf "$OPT.old"
[ -d "$OPT" ] && mv "$OPT" "$OPT.old"
mv "$OPT.tmp" "$OPT"
rm -rf "$OPT.old"
ln -sf "$OPT/bin/bloomery-serve" "$PREFIX/bin/bloomery-serve"

case ":$PATH:" in
    *":$PREFIX/bin:"*) ;;
    *) echo "note: $PREFIX/bin is not on your PATH — add it (e.g. 'export PATH=\"$PREFIX/bin:\$PATH\"')." >&2 ;;
esac

echo "installed: $PREFIX/bin/bloomery-serve ($VERSION)"
echo "try:       bloomery-serve --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M --port 8080"
