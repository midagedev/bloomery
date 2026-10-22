#!/usr/bin/env bash
# Installs the nightly host run on its machine (an Ubuntu 24.04 x86_64 VPS). Run as root by an operator, from a checkout that
# was reviewed — never by the service, which runs a public repository's code as bnightly; this script puts root-run files in place.
#   ssh root@vps 'NIGHTLY_SRC=/path/to/checkout bash -s' < tools/nightly/install.sh
# Idempotent: every step checks what is there. NIGHTLY_SRC is the checkout (tools/nightly/ and rust-toolchain.toml); it
# defaults to this script's own checkout when the script is run as a file.
#
# What it puts on the machine, and nothing else:
#   user       bnightly (system user, home /srv/bloomery-nightly, no login shell, no sudo)
#   dirs       /srv/bloomery-nightly (clone, target/, rustup and cargo homes, opt/, bin/), /var/lib/bloomery-nightly (runs)
#   apt        libclang-common-18-dev (clang's builtin headers, which bindgen needs; libclang 18 itself is already installed),
#              python3-numpy and python3-pil (the Python tool self-tests of tools/check-recipes.sh)
#   rustup     in bnightly's home, the channel rust-toolchain.toml pins, minimal profile (rustc, cargo, rust-std)
#   CUDA       the 13.3 headers only, from NVIDIA's apt repository's .deb files pinned in tools/nightly/cuda-debs.tsv,
#              fetched, sha256-checked and unpacked by tools/nightly/fetch-cuda-headers.sh into opt/cuda-13.3/ (no
#              repository added, no driver, no toolkit, no libcuda: cuda-bindings loads libcuda at run time and the units
#              open no card). cuda-cudart-dev (cuda.h), libcurand-dev (curand.h), cuda-crt (crt/host_config.h).
#   just       the prebuilt release binary (the version the Mac's recipes.py self-test is run with; its JSON dump differs between versions)
#   /usr/local/lib/bloomery-nightly/{run.sh,units.sh,mail.sh}   root-owned copies of this directory's scripts
#   /etc/systemd/system/bloomery-nightly.{service,timer}, the timer enabled
set -euo pipefail

SRC=${NIGHTLY_SRC:-}
if [ -z "$SRC" ] && [ -f "$0" ]; then SRC=$(cd "$(dirname "$0")/../.." && pwd); fi
[ -f "$SRC/tools/nightly/run.sh" ] && [ -f "$SRC/rust-toolchain.toml" ] || { echo "install: NIGHTLY_SRC must be a checkout with tools/nightly/ and rust-toolchain.toml" >&2; exit 64; }
[ "$(id -u)" = 0 ] || { echo "install: run as root" >&2; exit 64; }

H=/srv/bloomery-nightly
STATE=/var/lib/bloomery-nightly
LIB=/usr/local/lib/bloomery-nightly
JUST_VERSION=1.58.0
JUST_SHA=4a5cc2f53e6f0f8c59092a6cc38291eb729d46a7dd95d3ae582008881b84931d
# From /, with the channel named: run inside the checkout, rustup reads rust-toolchain.toml and installs its whole component
# list (rust-src, rustc-dev, rust-analyzer, clippy, rustfmt, llvm-tools: about a gigabyte no unit needs).
cd /
CHANNEL=$(sed -n 's/^channel *= *"\(.*\)"/\1/p' "$SRC/rust-toolchain.toml" | head -1)
[ -n "$CHANNEL" ] || { echo "install: no channel in rust-toolchain.toml" >&2; exit 1; }
as_b() { runuser -u bnightly -- env HOME="$H" PATH="$H/.cargo/bin:$H/bin:/usr/local/bin:/usr/bin:/bin" RUSTUP_TOOLCHAIN="$CHANNEL" "$@"; }
say() { printf 'install: %s\n' "$*"; }

# ---- user and dirs ----
if ! id bnightly > /dev/null 2>&1; then
  useradd --system --user-group --create-home --home-dir "$H" --shell /usr/sbin/nologin --comment "bloomery nightly host tests" bnightly
  say "user bnightly created"
fi
install -d -o bnightly -g bnightly -m 0750 "$H"
install -d -o bnightly -g bnightly -m 0755 "$STATE"

# ---- apt: clang's builtin headers for bindgen ----
for pkg in libclang-common-18-dev python3-numpy python3-pil; do
  if ! dpkg -s "$pkg" > /dev/null 2>&1; then
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$pkg"
  fi
done
ls /usr/lib/llvm-18/lib/libclang*.so* > /dev/null 2>&1 || { echo "install: no libclang 18 under /usr/lib/llvm-18/lib (apt install libclang1-18)" >&2; exit 1; }

# ---- rustup and the pinned channel ----
if [ ! -x "$H/.cargo/bin/rustup" ]; then
  as_b bash -c 'curl -sSf -m 120 https://sh.rustup.rs | sh -s -- -y --default-toolchain none --profile minimal --no-modify-path' > /dev/null
  say "rustup installed"
fi
if ! as_b rustup toolchain list | grep -q "^$CHANNEL"; then
  as_b rustup toolchain install "$CHANNEL" --profile minimal --no-self-update
  say "toolchain $CHANNEL installed"
fi

# ---- CUDA headers: the pins and the unpack are tools/nightly/cuda-debs.tsv and fetch-cuda-headers.sh,
# the one owner CI's lint job shares (it skips everything when the three headers are already there) ----
as_b bash "$SRC/tools/nightly/fetch-cuda-headers.sh" "$H/opt/cuda-13.3"
say "CUDA 13.3 headers in place: $(grep -m1 'define CUDA_VERSION' "$H/opt/cuda-13.3/include/cuda.h")"

# ---- just ----
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
chown bnightly: "$T"
if [ "$(as_b "$H/bin/just" --version 2> /dev/null || true)" != "just $JUST_VERSION" ]; then
  as_b mkdir -p "$H/bin"
  as_b curl -sSfL -m 300 -o "$T/just.tgz" "https://github.com/casey/just/releases/download/$JUST_VERSION/just-$JUST_VERSION-x86_64-unknown-linux-musl.tar.gz"
  [ "$(sha256sum "$T/just.tgz" | cut -d' ' -f1)" = "$JUST_SHA" ] || { echo "install: sha256 of just differs from the pin" >&2; exit 1; }
  as_b tar -xzf "$T/just.tgz" -C "$H/bin" just
  say "just $JUST_VERSION installed"
fi

# ---- the root-owned scripts and the units ----
install -d -o root -g root -m 0755 "$LIB"
install -o root -g root -m 0755 "$SRC/tools/nightly/run.sh" "$LIB/run.sh"
install -o root -g root -m 0755 "$SRC/tools/nightly/units.sh" "$LIB/units.sh"
install -o root -g root -m 0755 "$SRC/tools/nightly/mail.sh" "$LIB/mail.sh"
install -o root -g root -m 0644 "$SRC/tools/gate-bound.sh" "$LIB/gate-bound.sh"
install -o root -g root -m 0644 "$SRC/tools/nightly/bloomery-nightly.service" /etc/systemd/system/bloomery-nightly.service
install -o root -g root -m 0644 "$SRC/tools/nightly/bloomery-nightly.timer" /etc/systemd/system/bloomery-nightly.timer
systemd-analyze verify /etc/systemd/system/bloomery-nightly.service /etc/systemd/system/bloomery-nightly.timer
systemctl daemon-reload
systemctl enable --now bloomery-nightly.timer
say "done"
systemctl list-timers bloomery-nightly.timer --no-pager
