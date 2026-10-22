#!/usr/bin/env bash
# Fetches the CUDA 13.3 headers — the three .deb files pinned (name, sha256) in cuda-debs.tsv beside this
# script, from NVIDIA's ubuntu2404/x86_64 apt repository — checks each sha256 against the pin and unpacks
# only the headers into one directory:
#
#   tools/nightly/fetch-cuda-headers.sh <dir>
#
# <dir> is what CUDA_TOOLKIT_PATH names (tools/nightly/install.sh's ~/opt/cuda-13.3): the headers under
# <dir>/targets/x86_64-linux/include, with <dir>/include a symlink to it (the build reads
# <dir>/include/cuda.h). Idempotent: when the three header files are already there, nothing is fetched —
# only the symlink is made sure of; otherwise the headers are fetched and copied into that include
# directory, and nothing else under <dir> is removed or replaced. The unpack goes to a staging directory
# first, so a failed download or sha check leaves <dir> untouched. A <dir>/include that is a directory of
# its own (not the symlink) is refused by name before anything is fetched.
#
# One owner: tools/nightly/install.sh and the CI lint job (.github/workflows/pr.yml, which caches <dir>
# keyed on cuda-debs.tsv's hash) both run this script — the pins and the unpack steps have no second copy.
#
# Works on Ubuntu 24.04 (dpkg-deb) and on macOS for a local proof (its tar is bsdtar, which reads the
# .deb's data member through libarchive; GNU tar cannot read a .deb). Needs curl, tar and sha256sum
# (shasum -a 256 on macOS). Exit: 0 the headers are in place (fetched or already there); 64 usage; 69 a
# prerequisite is missing or a fetch/unpack failed (named, with how to make it); 70 the pin list is
# malformed (named).
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
LIST=$HERE/cuda-debs.tsv
REPO=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64
INNER=usr/local/cuda-13.3/targets/x86_64-linux/include

say() { printf 'cuda-headers: %s\n' "$*" >&2; }

[ $# = 1 ] || { say "usage: tools/nightly/fetch-cuda-headers.sh <dir> (the directory CUDA_TOOLKIT_PATH names)"; exit 64; }
DEST=$1
case $DEST in
  '' | . | .. | / | /usr | /bin | /sbin | /etc | /lib | /lib64 | /opt | /var | /home | /Users | /tmp)
    say "refusing to make '$DEST' the CUDA header directory"
    exit 64
    ;;
esac

command -v curl > /dev/null 2>&1 || { say "no curl on PATH: install it (apt install curl)"; exit 69; }
command -v tar > /dev/null 2>&1 || { say "no tar on PATH: install it (apt install tar)"; exit 69; }
if command -v sha256sum > /dev/null 2>&1; then
  sha256_of() { sha256sum "$1" | cut -d' ' -f1; }
elif command -v shasum > /dev/null 2>&1; then
  sha256_of() { shasum -a 256 "$1" | cut -d' ' -f1; }
else
  say "no sha256sum and no shasum on PATH: install one (Ubuntu: the coreutils that are always there; macOS: /usr/bin/shasum)"
  exit 69
fi

# the three headers the pins provide — the completeness check (install.sh's own was these same three files)
complete() {
  [ -f "$1/targets/x86_64-linux/include/cuda.h" ] && [ -f "$1/targets/x86_64-linux/include/curand.h" ] &&
    [ -f "$1/targets/x86_64-linux/include/crt/host_config.h" ]
}

link_include() { ln -sfn targets/x86_64-linux/include "$1/include"; }
if [ -e "$DEST/include" ] && [ ! -L "$DEST/include" ]; then
  say "$DEST/include is a directory, not the symlink to targets/x86_64-linux/include this script keeps — name another directory"
  exit 64
fi

if complete "$DEST"; then
  link_include "$DEST"
  say "the three headers are already under $DEST — nothing fetched"
  exit 0
fi

# the pin list: <deb name> <sha256>, '#' lines are its rule; a row that is not that is a named 70
debs=()
shas=()
while IFS=$' \t' read -r deb sha rest; do
  case $deb in '' | '#'*) continue ;; esac
  if [ -n "${rest:-}" ] || ! [[ $deb =~ ^[A-Za-z0-9._+-]+\.deb$ ]] || ! [[ $sha =~ ^[0-9a-f]{64}$ ]]; then
    say "$LIST must hold '<deb file name> <sha256>' lines; it holds: $deb ${sha:-}"
    exit 70
  fi
  debs+=("$deb")
  shas+=("$sha")
done < "$LIST"
[ "${#debs[@]}" -gt 0 ] || { say "$LIST pins no .deb"; exit 70; }

T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
stage=$T/cuda-13.3
mkdir -p "$stage/targets/x86_64-linux/include"

# unpack_deb DEB DIR: the deb's inner include tree into DIR. On Ubuntu dpkg-deb prints the data member as
# a tar; on macOS (no dpkg-deb) bsdtar reads the .deb itself and the data member is extracted to stdout.
unpack_deb() {
  local deb=$1 dir=$2 member x
  x=$(mktemp -d)
  if command -v dpkg-deb > /dev/null 2>&1; then
    dpkg-deb --fsys-tarfile "$deb" | tar -xf - -C "$x"
  else
    member=$(tar -tf "$deb" | sed -n 's/^\(data\.tar[^/]*\)$/\1/p' | head -1)
    if [ -z "$member" ]; then
      say "this tar lists no data.tar member of $deb — GNU tar cannot read a .deb; macOS's bsdtar can"
      return 69
    fi
    tar -xOf "$deb" "$member" | tar -xf - -C "$x"
  fi
  if [ ! -d "$x/$INNER" ]; then
    say "$deb holds no ./$INNER — the package moved its tree; repin $LIST"
    return 69
  fi
  cp -a "$x/$INNER/." "$dir/"
  rm -rf "$x"
}

i=0
for deb in "${debs[@]}"; do
  curl -fsSL -m 600 -o "$T/$deb" "$REPO/$deb" || { say "fetching $REPO/$deb failed (curl rc $?)"; exit 69; }
  got=$(sha256_of "$T/$deb")
  if [ "$got" != "${shas[$i]}" ]; then
    say "sha256 of $deb differs from the pin in $LIST: got $got, want ${shas[$i]} — not unpacked"
    exit 69
  fi
  unpack_deb "$T/$deb" "$stage/targets/x86_64-linux/include"
  say "fetched and unpacked $deb (sha256 ok)"
  i=$((i + 1))
done

# the sentinels prove the unpack before anything is copied into <dir>; only its include directory is
# written, so the rest of an existing toolkit under it stays as it was
complete "$stage" || { say "the unpack left no cuda.h, curand.h, crt/host_config.h under $stage"; exit 69; }
mkdir -p "$DEST/targets/x86_64-linux/include"
cp -a "$stage/targets/x86_64-linux/include/." "$DEST/targets/x86_64-linux/include/"
link_include "$DEST"
say "CUDA 13.3 headers unpacked: $(grep -m1 'define CUDA_VERSION' "$DEST/targets/x86_64-linux/include/cuda.h" || echo 'cuda.h without a CUDA_VERSION line')"
