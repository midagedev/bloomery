#!/bin/sh
# bloomery's installer, made to be run as
#
#   curl -fsSL https://raw.githubusercontent.com/midagedev/bloomery/main/tools/release/install.sh | sh
#
# or from a checkout (tools/release/install.sh [--version V]). No prompts, no
# sudo: the release lands in ~/.local/opt/bloomery-<version> and symlinks go into
# ~/.local/bin (created; a PATH note prints when the directory is not on PATH):
# bloomery-serve on Linux x86-64 (the sm_120a archive when every GPU is compute
# capability 12.0, the RTX 50 series; the sm_86 archive otherwise) and on a GB10
# (DGX Spark, Linux aarch64);
# bloomery-serve-metal, generate_metal and bloomery-prepare on an Apple Silicon
# Mac. Everything the download needs is checked against the release's own
# sha256 before anything is extracted over. --self-test checks the archive
# choice on stub uname and nvidia-smi output (tools/check-recipes.sh runs it).
set -eu

REPO=midagedev/bloomery
VERSION=${VERSION:-}
PREFIX=${PREFIX:-"$HOME/.local"}
SELF_TEST=
while [ $# -gt 0 ]; do
    case "$1" in
        --version) VERSION=$2; shift 2 ;;
        --prefix) PREFIX=$2; shift 2 ;;
        --self-test) SELF_TEST=1; shift ;;
        *) echo "install.sh: unknown argument $1 (takes --version V, --prefix DIR)" >&2; exit 64 ;;
    esac
done

fail() { echo "install.sh: $*" >&2; exit 1; }

# platform: the archive this machine takes, or the reason it takes none. The x86-64 sm_120a
# archive's GPU code is sm_120a PTX, which the driver loads on compute capability 12.0
# alone, so it is the pick only when nvidia-smi lists at least one GPU and every GPU it
# lists is 12.0; any other answer (no nvidia-smi, a failed query, no GPU, a 12.0 card beside
# another capability) takes the sm_86 archive, whose PTX the driver compiles for any card of
# 8.6 or newer. The aarch64 archive's GPU code is sm_121a PTX (GB10's native FP4), which the
# driver loads on compute capability 12.1 alone, so nvidia-smi must list at least one GPU and
# every GPU it lists must be a GB10.
platform() {
    case "$(uname -s)-$(uname -m)" in
        Linux-x86_64)
            if command -v nvidia-smi > /dev/null && caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null); then
                set -- $caps
                if [ "$#" -gt 0 ]; then
                    blackwell=1
                    for c in "$@"; do
                        [ "$c" = 12.0 ] || blackwell=
                    done
                    [ -z "$blackwell" ] || { echo linux-x86_64-cuda-sm120a; return; }
                fi
            fi
            echo linux-x86_64-cuda-sm86 ;;
        Linux-aarch64)
            command -v nvidia-smi > /dev/null \
                || fail "nvidia-smi is required to identify a GB10 (DGX Spark); install its driver first."
            caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader) \
                || fail "nvidia-smi could not read the GPU's compute capability; the aarch64 release needs a GB10 (DGX Spark) and its driver."
            set -- $caps
            [ "$#" -gt 0 ] || fail "nvidia-smi reported no GPUs; the aarch64 release needs a GB10 (DGX Spark)."
            for c in "$@"; do
                [ "$c" = 12.1 ] || fail "the Linux aarch64 release is for the GB10 (DGX Spark, compute capability 12.1); nvidia-smi reports $c. Build from source for this GPU (docs/BUILD.md)."
            done
            echo linux-aarch64-cuda-sm121a ;;
        Darwin-arm64) echo macos-arm64-metal ;;
        Darwin-*) fail "the macOS release is for Apple Silicon; uname -m says $(uname -m)." ;;
        Linux-*) fail "the Linux release is for x86-64 and for aarch64 with a GB10 (DGX Spark); uname -m says $(uname -m)." ;;
        *) fail "the release is for Linux x86-64, a GB10 (DGX Spark) and Apple Silicon Macs (Windows: run it inside WSL2); this is $(uname -s)." ;;
    esac
}

# archive <platform> <tag>: the platform whose archive the release <tag> carries. A release that
# predates the sm_120a archive answers 404 for its .sha256, and the machine then takes the sm_86
# archive; any other answer, a dead network included, leaves the platform to the download, which
# fails by name.
archive() {
    if [ "$1" = linux-x86_64-cuda-sm120a ]; then
        code=$(curl -sI -o /dev/null -w '%{http_code}' "https://github.com/$REPO/releases/download/$2/bloomery-${2#v}-$1.tar.gz.sha256" || true)
        if [ "$code" = 404 ]; then
            echo "install.sh: note: release $2 carries no $1 archive; installing linux-x86_64-cuda-sm86, whose GPU code the driver compiles for this card." >&2
            set -- linux-x86_64-cuda-sm86 "$2"
        fi
    fi
    echo "$1"
}

# The choice on stub uname and nvidia-smi: the stubs' directory is the whole PATH,
# so the machine's own nvidia-smi never answers.
self_test() {
    d=$(mktemp -d)
    trap 'rm -rf "$d"' EXIT
    cases=0 fails=0
    # expect <uname -s> <uname -m> <compute caps, "none" for no nvidia-smi, "fail" for one that fails> <want>:
    # platform prints want and exits 0, or its refusal starts with want ("install.sh: ...") and exits 1.
    expect() {
        cases=$((cases + 1))
        rm -rf "$d/bin" && mkdir "$d/bin"
        printf '#!/bin/sh\nif [ "$1" = -s ]; then echo %s; else echo %s; fi\n' "$1" "$2" > "$d/bin/uname"
        case "$3" in
            none) ;;
            fail) printf '#!/bin/sh\necho "No devices were found"\nexit 6\n' > "$d/bin/nvidia-smi" ;;
            *) printf '#!/bin/sh\nfor c in %s; do echo "$c"; done\n' "$3" > "$d/bin/nvidia-smi" ;;
        esac
        chmod +x "$d"/bin/*
        rc=0
        got=$(PATH="$d/bin"; platform 2>&1) || rc=$?
        case "$4" in install.sh:*) want_rc=1 ;; *) want_rc=0 ;; esac
        case "$want_rc:$rc:$got" in
            "0:0:$4"|"1:1:$4"*) ;;
            *) echo "install.sh self-test: $1-$2 ($3): rc $rc '$got', want rc $want_rc '$4'" >&2; fails=$((fails + 1)) ;;
        esac
    }
    expect Linux x86_64 8.6 linux-x86_64-cuda-sm86
    expect Linux x86_64 none linux-x86_64-cuda-sm86
    expect Linux x86_64 12.0 linux-x86_64-cuda-sm120a
    expect Linux x86_64 "12.0 12.0" linux-x86_64-cuda-sm120a
    expect Linux x86_64 "12.0 8.6" linux-x86_64-cuda-sm86
    expect Linux x86_64 "8.6 12.0" linux-x86_64-cuda-sm86
    expect Linux x86_64 12.1 linux-x86_64-cuda-sm86
    expect Linux x86_64 9.0 linux-x86_64-cuda-sm86
    expect Linux x86_64 "" linux-x86_64-cuda-sm86
    expect Linux x86_64 fail linux-x86_64-cuda-sm86
    expect Darwin arm64 none macos-arm64-metal
    expect Linux aarch64 12.1 linux-aarch64-cuda-sm121a
    expect Linux aarch64 none "install.sh: nvidia-smi is required"
    expect Linux aarch64 "" "install.sh: nvidia-smi reported no GPUs"
    expect Linux aarch64 9.0 "install.sh: the Linux aarch64 release is for the GB10"
    expect Linux aarch64 "12.1 9.0" "install.sh: the Linux aarch64 release is for the GB10"
    expect Linux aarch64 fail "install.sh: nvidia-smi could not read"
    expect Linux riscv64 none "install.sh: the Linux release is for x86-64 and for aarch64"
    expect Darwin x86_64 none "install.sh: the macOS release is for Apple Silicon"
    expect FreeBSD amd64 none "install.sh: the release is for Linux x86-64"
    # expect_archive <platform> <tag> <the code curl prints for the .sha256 URL> <want> <the note stderr must hold, or ->:
    # archive prints want.
    expect_archive() {
        cases=$((cases + 1))
        rm -rf "$d/bin" && mkdir "$d/bin"
        printf '#!/bin/sh\necho %s\n' "$3" > "$d/bin/curl"
        chmod +x "$d/bin/curl"
        got=$(PATH="$d/bin"; archive "$1" "$2" 2> "$d/err")
        if [ "$got" != "$4" ] || { [ "$5" != - ] && ! grep -qF "$5" "$d/err"; } || { [ "$5" = - ] && [ -s "$d/err" ]; }; then
            echo "install.sh self-test: archive $1 $2 (curl says $3): '$got' with note '$(cat "$d/err")', want '$4' and '$5'" >&2
            fails=$((fails + 1))
        fi
    }
    expect_archive linux-x86_64-cuda-sm120a v0.3.11 302 linux-x86_64-cuda-sm120a -
    expect_archive linux-x86_64-cuda-sm120a v0.3.11 200 linux-x86_64-cuda-sm120a -
    expect_archive linux-x86_64-cuda-sm120a v0.3.10 404 linux-x86_64-cuda-sm86 "install.sh: note: release v0.3.10 carries no linux-x86_64-cuda-sm120a archive"
    expect_archive linux-x86_64-cuda-sm120a v0.3.11 000 linux-x86_64-cuda-sm120a -
    expect_archive linux-x86_64-cuda-sm86 v0.3.10 404 linux-x86_64-cuda-sm86 -
    expect_archive linux-aarch64-cuda-sm121a v0.3.10 404 linux-aarch64-cuda-sm121a -
    if [ "$fails" = 0 ]; then
        echo "install.sh: self-test ok ($cases cases)"
    else
        echo "install.sh: self-test $fails of $cases cases failed" >&2
        return 1
    fi
}
if [ -n "$SELF_TEST" ]; then
    self_test
    exit $?
fi

PLATFORM=$(platform) || exit 1

command -v curl > /dev/null || fail "curl is needed for the download."
command -v tar > /dev/null || fail "tar is needed for the archive."

if [ "$PLATFORM" = macos-arm64-metal ]; then
# The Metal backend's floor; the binaries refuse an older macOS by name too.
OS=$(sw_vers -productVersion)
OLDER=$(printf '%s\n26.2\n' "$OS" | sort -t. -k1,1n -k2,2n -k3,3n | head -1)
[ "$OLDER" = "26.2" ] || fail "macOS $OS is older than the Metal backend's floor, 26.2."
else

case "$PLATFORM" in
linux-x86_64-*)
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
;;
esac

# tools/release/build.sh's floor for each archive.
FLOOR=2.34
case "$PLATFORM" in
linux-x86_64-*) ;;
*) FLOOR=2.39 ;;
esac
GLIBC=$(ldd --version 2>/dev/null | head -1 | grep -o '[0-9]*\.[0-9]*$' || true)
if [ -n "$GLIBC" ]; then
    OLDER=$(printf '%s\n%s\n' "$GLIBC" "$FLOOR" | sort -V | head -1)
    [ "$OLDER" = "$FLOOR" ] || fail "glibc $GLIBC is older than the release's floor, $FLOOR."
else
    echo "install.sh: could not read the glibc version (ldd); continuing." >&2
fi

if ! command -v nvidia-smi > /dev/null; then
    echo "install.sh: note: nvidia-smi is not on PATH — this build needs an NVIDIA GPU of compute" >&2
    echo "capability 8.6 or newer and its driver." >&2
fi
fi

# The version to install: the given one, else the release the API names latest.
if [ -z "$VERSION" ]; then
    VERSION=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
        | grep -m1 '"tag_name"' | grep -o 'v[0-9.]*') \
        || fail "could not read the latest release from GitHub."
fi
PLATFORM=$(archive "$PLATFORM" "$VERSION")
ASSET="bloomery-${VERSION#v}-$PLATFORM.tar.gz"
URL="https://github.com/$REPO/releases/download/$VERSION/$ASSET"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
echo "downloading $URL"
curl -fSL "$URL" -o "$TMP/$ASSET" || fail "could not download $ASSET; release $VERSION may carry no $PLATFORM archive."
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
if [ "$PLATFORM" = macos-arm64-metal ]; then
    ln -sf "$OPT/bin/bloomery-serve-metal" "$PREFIX/bin/bloomery-serve-metal"
    ln -sf "$OPT/bin/generate_metal" "$PREFIX/bin/generate_metal"
    ln -sf "$OPT/bin/prepare" "$PREFIX/bin/bloomery-prepare"
else
    ln -sf "$OPT/bin/bloomery-serve" "$PREFIX/bin/bloomery-serve"
fi

case ":$PATH:" in
    *":$PREFIX/bin:"*) ;;
    *) echo "note: $PREFIX/bin is not on your PATH — add it (e.g. 'export PATH=\"$PREFIX/bin:\$PATH\"')." >&2 ;;
esac

if [ "$PLATFORM" = macos-arm64-metal ]; then
    echo "installed: $PREFIX/bin/bloomery-serve-metal, generate_metal, bloomery-prepare ($VERSION)"
    echo "try:       bloomery-serve-metal -m PATH/Qwen3.6-35B-A3B-Q4_K_M.gguf --port 8080"
elif [ "$PLATFORM" = linux-aarch64-cuda-sm121a ]; then
    echo "installed: $PREFIX/bin/bloomery-serve ($VERSION)"
    echo "try:       bloomery-serve -m PATH/Qwen3.8-Flash-Next-FP4-Local --port 8080"
else
    echo "installed: $PREFIX/bin/bloomery-serve ($VERSION)"
    echo "try:       bloomery-serve --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M --port 8080"
fi
