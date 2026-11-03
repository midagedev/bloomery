#!/usr/bin/env bash
# Builds the prebuilt Linux release on the box and packs it: `just release-build <version>` runs it through
# tools/box.sh with the Mac's commit. It builds into its own target dir (target/portable, below), so the measure runners'
# target/release is never touched.
#
# The release is the building machine's platform (uname -m) and CUDA arch (BLOOMERY_ARCH, tools/arch.sh), one archive each:
#   - x86_64, sm_86: bloomery-<v>-linux-x86_64-cuda-sm86, sm_86 PTX the driver compiles for any card of cc 8.6 or newer,
#     host code for x86-64-v3; built on the box with BLOOMERY_ARCH=sm_86, the selection on a machine whose cards are not
#     all cc 12.0;
#   - x86_64, sm_120a: bloomery-<v>-linux-x86_64-cuda-sm120a, sm_120a PTX (the Blackwell consumer parts' native FP4 and
#     MXFP8 kernels, which load on cc 12.0 alone), the same host code; built on the box with BLOOMERY_ARCH=sm_120a (the
#     selection on a machine whose every card is cc 12.0), into its own target dir (target/portable-sm120a) so neither
#     archive's build rebuilds the other's. A build machine with no cc 12.0 card cannot compile the PTX to cubins, so it
#     stays PTX and the driver compiles it on the card's first start;
#   - aarch64: bloomery-<v>-linux-aarch64-cuda-sm121a, sm_121a PTX (GB10's native FP4, which loads on cc 12.1 alone),
#     host code for GB10's cores (rustc's `gb10`); built on a GB10 itself, natively (tools/box.sh's local mode there,
#     BLOOMERY_BOX_LOCAL=1: box.sh's remote path is the x86_64 box's, with locks under /root and its cards picked by name; the
#     toolchain is docs/BUILD.md's, CUDA 13.3's headers included, which a Spark's own CUDA 13.0 packages lack: bindgen ends
#     E0425 on cuGraphNodeGetParams) or cross on an x86_64 machine (BLOOMERY_RELEASE_TARGET=aarch64: the rust aarch64 target, the cross gcc
#     for the host code and aarch64 binutils, the cargo --target, the triple's bin dir and its own target dir; every missing
#     tool, the rust target and, when qemu-aarch64 is on PATH, a QEMU_LD_PREFIX that is no aarch64 sysroot are refused by
#     name, rc 69, before anything is built, and any other value is rc 64).
#     Either way the packed binary carries that PTX compiled as cubins, and the PTX stays beside it as `.oxart.ptx`,
#     which bloomery-gpu loads for a bundle whose cubin the driver refuses, so a first start compiles nothing. On a
#     GB10, oxart_jit --cubin links each `.oxart` bundle on the building machine's card; cross, oxart_ptxas hands each
#     bundle's PTX to the toolkit's ptxas -arch=<the bundle's target> on the build host, no card and no gate lock, the
#     ptxas version printed. A cross binary's `--version` runs under qemu-aarch64 (user mode, QEMU_LD_PREFIX its
#     sysroot) before and after the cubins are packed; without qemu-aarch64 on PATH the check is named "not run" in the
#     output. What no x86_64 machine proves: the driver's load of the cubins on a GB10 and a request.
#
# The failures it blocks, each checked on the built binaries before anything is packed:
#   - a binary for another machine than the platform's (its ELF header);
#   - a binary that needs a glibc newer than GLIBC_FLOOR through a strong (non-WEAK) symbol version, so it would not
#     start on the oldest distribution the release names;
#   - a binary that links a shared library outside the allowed set (libcuda must be loaded at run time, never linked);
#   - on x86_64, host code built for this box's CPU (znver3) instead of x86-64-v3: an instruction past x86-64-v3 outside a
#     function that runs behind a runtime CPU check;
#   - a binary with no embedded PTX of the platform's arch, or PTX for another target;
#   - a packed binary whose `--version` fails to run, or names another commit or version than the archive's (a cross
#     binary runs under qemu-aarch64, or is named "not run" where that is absent).
# A BLOOMERY_ARCH that is no arch of the machine's archives (tools/arch.sh) ends a build with 64 before anything is built;
# `--check` reads the binaries and never that variable: the archive's arch is its argument, its machine is
# BLOOMERY_RELEASE_TARGET's or the machine's own, and an x86_64 binary checked with none is held to sm_86.
# Each archive builds in a directory of its own, so two archives build at the same time from one tree and neither rebuilds the
# other: cargo's artifacts in target/<its dir> (below) and the device codegen's files in target/<its dir>/oxide-out. The
# second is CUDA_OXIDE_PTX_DIR: cargo-oxide writes a crate's PTX, .ll and .oxide-artifacts/ into the compiler's working
# directory, the tree's root, unless that variable names another, and removes the previous ones first, so two builds of
# different archs from one root would delete and overwrite each other's.
# Before anything is built the tree is held to what the publish will hold it to: the serve and gpu-gates crates say <version>
# and README.md names the release's status, tarball and link (tools/release/publish.sh --tree-check, the one owner of the
# check), so a tree that cannot publish is refused here, not after every archive is built.
# --stage w builds at the follower's watermark, the newest all-green commit, ahead of the ask: the tree check holds the crates'
# versions alone (--tree-check <version> --crates-only), because that commit's README names the last release until the docs
# commit that the tag names. The archive is the same program, and its --version names the watermark's commit; tools/release/
# publish.sh accepts it for a tag commit that follows with only docs paths changed (its docs_delta). Without --stage the
# README is held too.
# Usage: tools/release/build.sh [--stage w] <version> <commit>
#        tools/release/build.sh [--stage w] --dry-run <version> <commit>   (the tree check and the directories this build would use; builds nothing)
#        tools/release/build.sh --check <dir> [<arch>]   (the binary checks alone, on <dir>/<bin>, for the archive of
#                                                         <arch>, default the archive machine's sm_86 or sm_121a; builds and packs nothing)
#        tools/release/build.sh --self-test     (the host-code scan on fixture listings, the platform table, the cross refusals and
#                                                the --version checks on stub binaries and a stub qemu; check-recipes runs it)
# BLOOMERY_RELEASE_TARGET=aarch64 builds the aarch64 archive cross on an x86_64 machine; a build and a --check both read it
# (`just release-build <v>` carries it to the box; QEMU_LD_PREFIX, the machine's aarch64 sysroot, is the box's own).
# The tarball and its .sha256 keep the names this script writes ($NAME.tar.gz…): install.sh resolves the release by
# them and the Homebrew tap's formula pins them, so `gh release upload` must be given the files as named, not copies
# under other names. The version the tarball carries is the serve crate's, so a release bumps that crate first.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
CHECK_ONLY= SELF_TEST= CHECK_ARCH= DRY= BUILD_STAGE=
if [ "${1:-}" = --check ]; then
  CHECK_ONLY=${2:?usage: tools/release/build.sh --check <dir> [<arch>]}
  CHECK_ARCH=${3:-}
  [ $# -le 3 ] || { echo "usage: tools/release/build.sh --check <dir> [<arch>]" >&2; exit 64; }
elif [ "${1:-}" = --self-test ]; then
  [ $# = 1 ] || { echo "usage: tools/release/build.sh --self-test" >&2; exit 64; }
  SELF_TEST=1
else
  BUILD_USAGE="usage: tools/release/build.sh [--stage w] [--dry-run] <version> <commit>"
  while [ $# -gt 0 ]; do
    case "$1" in
      --dry-run) DRY=1; shift ;;
      --stage)
        [ $# -ge 2 ] || { echo "build.sh: --stage needs a stage (w); $BUILD_USAGE" >&2; exit 64; }
        [ "$2" = w ] || { echo "build.sh: --stage takes w, got '$2'; $BUILD_USAGE" >&2; exit 64; }
        BUILD_STAGE=$2; shift 2 ;;
      -*) echo "build.sh: unknown argument $1; $BUILD_USAGE" >&2; exit 64 ;;
      *) break ;;
    esac
  done
  [ $# = 2 ] || { echo "$BUILD_USAGE" >&2; exit 64; }
  VERSION=$1 COMMIT=$2
fi

BINS=(bloomery-serve)
# PIN(2026-10-12): permit qdot's Q4_K AVX-512 VNNI bodies; every caller reaches them only when qdot::q4k_avx512 has
# seen AVX-512 F, BW and VNNI beside the AVX2 kernels' AVX2, FMA and F16C.
DISPATCHED='^(sha2::sha256::x86::digest_blocks|sha1::compress::x86::digest_blocks|crc32fast::specialized::pclmulqdq::calculate_avx(512|2)|qdot::dot_(q4k_q82x4_tile|lanes)_avx512::<[0-9]+>)$'

# scan_past_v3: the listing of `objdump -d -C` on stdin, bytes shown; prints `<count> <function>, first at <address>:
# <instruction>` for each function outside DISPATCHED that holds an instruction past x86-64-v3:
#   - an EVEX one: its opcode byte (after any 0x67 or segment prefix) is 0x62, which in 64-bit mode is EVEX alone, and
#     x86-64-v3 has no EVEX instruction; this is AVX-512 at any vector width, masked or not;
#   - one that names an opmask register (%k0 to %k7), which AVX-512 brings, as its VEX-encoded kmov and k-logic do;
#   - one from an extension x86-64-v3 lacks, by its mnemonic after the prefix words objdump prints before it ({vex},
#     addr32, rep, a segment and the like): SHA, SSE4a, clzero, monitorx, mwaitx, rdpid, AVX-VNNI with its INT8 and
#     INT16 forms, AVX-IFMA, GFNI, and VAES and VPCLMULQDQ on ymm.
# PIN(2026-10-14): VAES and VPCLMULQDQ count on ymm only. Their xmm forms need only AES-NI or PCLMULQDQ with AVX; like
# the legacy aesenc and pclmulqdq the scan leaves alone, dependencies select them at run time (crc32fast's SSE arm).
# A listing with an instruction line and no bytes, or with no instruction, is refused by name (exit 1): either
# would pass every function.
scan_past_v3() {
  awk -v allow="$DISPATCHED" '
    /^[0-9a-f]+ <.*>:$/ { fn = $0; sub(/^[0-9a-f]+ </, "", fn); sub(/>:$/, "", fn); next }
    /^ *[0-9a-f]+:\t/ {
      n = split($0, f, "\t")
      if ((f[2] " ") !~ /^([0-9a-f][0-9a-f] +)+$/) { unread++; next }
      if (n < 3 || f[3] == "") next # the bytes past the seventh of the instruction above
      insns++
      split(f[2], b, " ")
      i = 1
      while (b[i] ~ /^(26|2e|36|3e|64|65|67)$/) i++
      t = $0 # the instruction with its operands, whether a tab or spaces part them
      sub(/^[^\t]*\t[^\t]*\t/, "", t)
      gsub(/\t/, " ", t)
      m = t
      while (sub(/^(\{[a-z0-9]+\}|addr(16|32)|data(16|32)|rex[.A-Z0-9]*|lock|rep[a-z]*|notrack|bnd|xacquire|xrelease|[c-gs]s) +/, "", m)) ;
      sub(/ .*/, "", m)
      if (b[i] == "62" || t ~ /%k[0-7]/ ||
          m ~ /^(sha1[a-z0-9]*|sha256[a-z0-9]*|extrq|insertq|clzero|monitorx|mwaitx|rdpid|vpdp[a-z]+|vpmadd52[a-z]+|v?gf2p8[a-z]+)$/ ||
          (m ~ /^(vaes[a-z]*|vpclmul[a-z]*)$/ && t ~ /%ymm/))
        if (fn !~ allow && !bad[fn]++) {
          a = f[1]; sub(/^ */, "", a); sub(/:$/, "", a)
          first[fn] = a ": " t
        }
    }
    END {
      if (unread) { print "release: instruction lines with no bytes in the objdump listing: " unread "; the scan reads the bytes (objdump -d without --no-show-raw-insn)" > "/dev/stderr"; exit 1 }
      if (!insns) { print "release: the objdump listing holds no instruction; the scan reads GNU objdump -d lines (address, tab, bytes, tab, instruction)" > "/dev/stderr"; exit 1 }
      for (k in bad) print bad[k] " " k ", first at " first[k]
    }'
}

# set_platform <uname -m> <arch>: sets the archive's PLATFORM (its name), RELEASE_ARCH (its PTX target), HOST_CPU,
# ELF_MACHINE, LOADER, SCAN_V3, GLIBC_FLOOR, CUBIN and TARGET_DIR (its build dir under target/) for the machine and the
# arch the archive is for; an empty arch is the machine's own (sm_86 on x86_64, sm_121a on aarch64). Another machine or
# arch is refused by name, rc 64. The x86-64 binary's host code is held to x86-64-v3 by the scan below, and its floor is the
# oldest distribution the release names. The aarch64 one's target is GB10's own cores (rustc's `gb10`): its sm_121a PTX
# runs on no other machine, and on a GB10 the `.cargo` table's `native` names the same CPU, so there is no CPU past the
# target to scan for; its floor is DGX OS 7's glibc (Ubuntu 24.04), which every GB10 runs.
set_platform() {
  local machine=$1 arch=${2:-}
  case "$machine" in
    x86_64)
      HOST_CPU=x86-64-v3 ELF_MACHINE='Advanced Micro Devices X86-64'
      LOADER='ld-linux-x86-64\.so\.2' SCAN_V3=1 GLIBC_FLOOR=2.34 CUBIN=
      case "${arch:-sm_86}" in
        sm_86) PLATFORM=linux-x86_64-cuda-sm86 TARGET_DIR=portable ;;
        sm_120a) PLATFORM=linux-x86_64-cuda-sm120a TARGET_DIR=portable-sm120a ;;
        *)
          echo "release: no x86_64 archive for $arch; the x86_64 archives are sm_86 and sm_120a" >&2
          return 64 ;;
      esac
      RELEASE_ARCH=${arch:-sm_86} ;;
    aarch64)
      case "${arch:-sm_121a}" in
        sm_121a) ;;
        *)
          echo "release: no aarch64 archive for $arch; the aarch64 archive is sm_121a (BLOOMERY_ARCH unset on a GB10)" >&2
          return 64 ;;
      esac
      PLATFORM=linux-aarch64-cuda-sm121a RELEASE_ARCH=sm_121a HOST_CPU=gb10 ELF_MACHINE=AArch64 TARGET_DIR=portable
      LOADER='ld-linux-aarch64\.so\.1' SCAN_V3= GLIBC_FLOOR=2.39 CUBIN=1 ;;
    *)
      echo "release: no Linux release for $machine; the platforms are x86_64 and aarch64" >&2
      return 64 ;;
  esac
}

# set_target <uname -m> <BLOOMERY_RELEASE_TARGET>: MACHINE is the archive's machine and CROSS tells the cross build —
# the building machine's own, or aarch64 when an x86_64 machine builds the GB10 archive. Any other value, and the cross
# value on any other machine, is refused by name.
set_target() {
  MACHINE=$1 CROSS=
  case ${2:-} in
    '') ;;
    aarch64)
      if [ "$MACHINE" != x86_64 ]; then
        echo "release: BLOOMERY_RELEASE_TARGET=aarch64 builds the aarch64 archive on an x86_64 machine; this machine is $MACHINE (unset it to build this machine's own)" >&2
        return 64
      fi
      MACHINE=aarch64 CROSS=1 ;;
    *)
      echo "release: BLOOMERY_RELEASE_TARGET is 'aarch64' or unset, got '$2'" >&2
      return 64 ;;
  esac
}

# The cross build's own rows: its target dir (never the x86_64 archives', never a GB10's), the cargo triple, the cross
# linker, and the target binutils — the x86_64 objcopy cannot read an aarch64 binary, and the checks read the ELF with the
# target's readelf, objdump and strings. The cubin packer is a host tool (oxart_ptxas, no device code), so nothing that
# runs is built for the wrong machine; the row's CUBIN stays 1: the cross archive packs cubins like the GB10's own.
cross_platform() {
  TARGET_DIR=portable-aarch64 TRIPLE=aarch64-unknown-linux-gnu
  LINKER=aarch64-linux-gnu-gcc OBJCOPY=aarch64-linux-gnu-objcopy
  READELF=aarch64-linux-gnu-readelf OBJDUMP=aarch64-linux-gnu-objdump STRINGS=aarch64-linux-gnu-strings
}

# build_dirs <root>: this archive's two build directories, both under target/<TARGET_DIR> (set_platform, cross_platform): cargo's
# CARGO_TARGET_DIR and OXIDE_OUT, where cargo-oxide puts the device codegen's PTX, .ll and .oxide-artifacts/ (CUDA_OXIDE_PTX_DIR).
# Another archive's directory is never inside this one's.
build_dirs() {
  CARGO_TARGET_DIR="$1/target/$TARGET_DIR"
  OXIDE_OUT="$CARGO_TARGET_DIR/oxide-out"
}

# plan_lines: what this build does, for --dry-run (NAME, RELEASE_ARCH, MACHINE, DIST, CARGO_TARGET_DIR and OXIDE_OUT are set).
plan_lines() {
  echo "release: dry run: $NAME (arch $RELEASE_ARCH, $MACHINE) builds in $CARGO_TARGET_DIR"
  echo "release: dry run: its device files (PTX, .ll, .oxide-artifacts/) go to $OXIDE_OUT, not the tree's root"
  [ -z "$BUILD_STAGE" ] || echo "release: dry run: stage $BUILD_STAGE: the README was not checked; the archive's --version will name $COMMIT"
  echo "release: dry run: it packs $DIST/$NAME.tar.gz; nothing was built"
}

# need_tool <tool> <what to do>: refuses (69) a tool that is not on PATH, naming the command that provides it.
need_tool() {
  command -v "$1" > /dev/null || {
    echo "release: $1 is not on PATH: $2" >&2
    return 69
  }
}

# The cross binutils the checks read the binary with, refused by name (rc 69): a `--check` needs these alone.
cross_binutils() {
  local t
  for t in readelf objdump strings; do
    need_tool "aarch64-linux-gnu-$t" "apt install binutils-aarch64-linux-gnu (the checks read the aarch64 binary with the target's $t)" || return $?
  done
}

# The cross toolchain, refused by name before anything is built (rc 69): each tool names the package that provides it,
# the rust target the command that installs it, ptxas the toolkit it comes with. qemu-aarch64 is optional (the packed
# binary's --version is then named "not run", see cross_version); when it is on PATH, QEMU_LD_PREFIX must name the
# aarch64 sysroot it runs the binary in (the machine's own path; there is no default for it).
cross_tools() {
  need_tool aarch64-linux-gnu-gcc "apt install gcc-aarch64-linux-gnu (the cross linker)" || return $?
  need_tool aarch64-linux-gnu-objcopy "apt install binutils-aarch64-linux-gnu (the x86_64 objcopy cannot read an aarch64 binary)" || return $?
  cross_binutils || return $?
  need_tool ptxas "the CUDA toolkit's assembler compiles the archive's cubins (no card is needed)" || return $?
  local sysroot
  sysroot=$(rustc --print sysroot 2> /dev/null) || sysroot=
  if [ -z "$sysroot" ] || [ ! -d "$sysroot/lib/rustlib/aarch64-unknown-linux-gnu" ]; then
    echo "release: the aarch64-unknown-linux-gnu target is not installed for ${sysroot:-this rustc}: rustup target add aarch64-unknown-linux-gnu" >&2
    return 69
  fi
  command -v qemu-aarch64 > /dev/null || return 0
  if [ -z "${QEMU_LD_PREFIX:-}" ]; then
    echo "release: QEMU_LD_PREFIX is not set: name the aarch64 sysroot qemu-aarch64 runs the binary in (apt install libc6-arm64-cross; Ubuntu puts it at /usr/aarch64-linux-gnu)" >&2
    return 69
  fi
  if [ ! -e "$QEMU_LD_PREFIX/lib/ld-linux-aarch64.so.1" ]; then
    echo "release: QEMU_LD_PREFIX=$QEMU_LD_PREFIX holds no lib/ld-linux-aarch64.so.1: it is not an aarch64 sysroot (apt install libc6-arm64-cross)" >&2
    return 69
  fi
}

# check_version <binary> <version> <commit> [runner…]: the binary's `--version` (run by the runner when one is given: a
# cross binary needs qemu-aarch64) must name the commit and the version the tarball's name says — the serve crate's, so a
# release bumps that crate first. Prints the line; a failing run, a missing commit or a missing version ends it with 1.
check_version() {
  local f=$1 version=$2 commit=$3 v
  shift 3
  v=$("$@" "$f" --version) || {
    echo "release: ${f##*/} --version failed${1:+ under $1}" >&2
    return 1
  }
  case "$v" in
    *"$commit"*) ;;
    *) echo "release: ${f##*/} --version does not name the commit $commit: $v" >&2; return 1 ;;
  esac
  case "$v" in
    *" $version ("*) echo "release: ${f##*/} --version: $v" ;;
    *) echo "release: ${f##*/} --version does not name the version $version (bump crates/serve and crates/gpu-gates first): $v" >&2; return 1 ;;
  esac
}

# cross_version <binary> <version> <commit>: a cross binary's --version, run by qemu-aarch64 (the sysroot is
# QEMU_LD_PREFIX, which cross_tools checked). With no qemu-aarch64 on PATH the check does not run, and says so.
cross_version() {
  if command -v qemu-aarch64 > /dev/null; then
    check_version "$@" qemu-aarch64
  else
    echo "release: ${1##*/} --version not run: qemu-aarch64 is not on PATH (apt install qemu-user); a run on a GB10 checks it"
  fi
}

# The scan on listings in GNU objdump's form: address, bytes and instruction, tab-separated, and the bytes past the
# seventh on a line of their own. Each instruction's bytes are its assembler's encoding. A case holds one refused
# function at most, since awk lists them in no set order.
self_test() {
  local fails=0 cases=0 listing out rc in=qdot::f
  func() { printf '\n0000000000401000 <%s>:\n' "$1"; }
  insn() { printf '  401000:\t%-21s\t%s\n' "$1 " "$2"; }
  insn_rest() { printf '  401007:\t%s \n' "$1"; }
  expect() { # expect <case> <want rc> <want output>: scan_past_v3 on $listing
    cases=$((cases + 1))
    out=$(scan_past_v3 <<< "$listing" 2>&1) && rc=0 || rc=$?
    if [ "$rc" != "$2" ] || [ "$out" != "$3" ]; then
      echo "release self-test: $1: rc $rc (want $2), output:" >&2
      sed 's/^/  /' <<< "$out" >&2
      echo "  want: $3" >&2
      fails=$((fails + 1))
    fi
  }

  listing=$(func "$in"; insn '62 f2 75 28 7f c2' 'vpermt2ps %ymm2,%ymm1,%ymm0')
  expect evex-ymm 0 "1 $in, first at 401000: vpermt2ps %ymm2,%ymm1,%ymm0"
  listing=$(func "$in"; insn '62 f1 74 29 58 c2' 'vaddps %ymm2,%ymm1,%ymm0{%k1}')
  expect evex-ymm-masked 0 "1 $in, first at 401000: vaddps %ymm2,%ymm1,%ymm0{%k1}"
  listing=$(func crc32fast::specialized::pclmulqdq::calculate_avx512
    insn '62 f2 75 28 7f c2' 'vpermt2ps %ymm2,%ymm1,%ymm0'
    insn '62 f1 74 29 58 c2' 'vaddps %ymm2,%ymm1,%ymm0{%k1}'
    insn 'c5 f8 92 c8' 'kmovw %eax,%k1')
  expect dispatched 0 ""
  listing=$(func "$in"
    insn '62 f1 74 48 58 c2' 'vaddps %zmm2,%zmm1,%zmm0'
    insn '62 b1 75 08 fe c1' 'vpaddd %xmm17,%xmm1,%xmm0'
    insn '62 f1 74 a9 58 c2' 'vaddps %ymm2,%ymm1,%ymm0{%k1}{z}'
    insn '62 f1 75 28 db c2' 'vpandd %ymm2,%ymm1,%ymm0'
    insn '67 62 f2 75 28 7f 00' 'vpermt2ps (%eax),%ymm1,%ymm0'
    insn '64 62 f1 fe 28 6f 80' 'vmovdqu64 %fs:0x10(%rax),%ymm0'
    insn_rest '10 00 00 00')
  expect evex-forms 0 "6 $in, first at 401000: vaddps %zmm2,%zmm1,%zmm0"
  listing=$(func "$in"; insn 'c5 f8 92 c8' 'kmovw %eax,%k1')
  expect opmask 0 "1 $in, first at 401000: kmovw %eax,%k1"
  listing=$(func "$in"
    insn 'c5 f8 92 d0' $'kmovw\t%eax,%k2'
    insn 'c4 e2 71 dc c2' $'vaesenc\t%ymm2,%ymm1,%ymm0')
  expect operand-tab 0 "2 $in, first at 401000: kmovw %eax,%k2"
  listing=$(func "$in"
    insn 'c4 e2 75 50 c2' '{vex} vpdpbusd %ymm2,%ymm1,%ymm0'
    insn 'c4 e2 75 52 c2' '{vex} vpdpwssd %ymm2,%ymm1,%ymm0'
    insn 'c4 e2 77 50 c2' 'vpdpbssd %ymm2,%ymm1,%ymm0')
  expect vnni 0 "3 $in, first at 401000: {vex} vpdpbusd %ymm2,%ymm1,%ymm0"
  listing=$(func "$in"
    insn '0f 38 cb d1' 'sha256rnds2 %xmm0,%xmm1,%xmm2'
    insn 'c4 e2 75 dc c2' 'vaesenc %ymm2,%ymm1,%ymm0'
    insn 'c4 43 25 44 e2 11' 'vpclmulhqhqdq %ymm10,%ymm11,%ymm12'
    insn 'c4 e2 f5 b4 c2' '{vex} vpmadd52luq %ymm2,%ymm1,%ymm0'
    insn 'c4 e3 f5 ce c2 00' 'vgf2p8affineqb $0x0,%ymm2,%ymm1,%ymm0'
    insn '66 0f 38 cf c1' 'gf2p8mulb %xmm1,%xmm0')
  expect listed 0 "6 $in, first at 401000: sha256rnds2 %xmm0,%xmm1,%xmm2"
  listing=$(func "$in"
    insn '67 c4 e2 75 dc c2' 'addr32 vaesenc %ymm2,%ymm1,%ymm0'
    insn '67 2e c4 e2 75 dc c2' 'addr32 cs vaesenc %ymm2,%ymm1,%ymm0')
  expect prefix-words 0 "2 $in, first at 401000: addr32 vaesenc %ymm2,%ymm1,%ymm0"
  listing=$(func "$in"
    insn 'c5 f4 58 c2' 'vaddps %ymm2,%ymm1,%ymm0'
    insn 'c4 e3 69 44 d9 00' 'vpclmullqlqdq %xmm1,%xmm2,%xmm3'
    insn 'c4 e2 71 dc c2' 'vaesenc %xmm2,%xmm1,%xmm0'
    insn '48 b8 62 62 62 62 62' 'movabs $0x6262626262626262,%rax'
    insn_rest '62 62 62'
    insn 'c5 fe 6f 80 62 62 62' 'vmovdqu 0x62626262(%rax),%ymm0'
    insn_rest '62')
  expect v3 0 ""
  listing=$(func "$in"; printf '  401000:\t%s\n' 'vaddps %ymm2,%ymm1,%ymm0')
  expect no-bytes 1 "release: instruction lines with no bytes in the objdump listing: 1; the scan reads the bytes (objdump -d without --no-show-raw-insn)"
  listing=$(func "$in")
  expect no-instruction 1 "release: the objdump listing holds no instruction; the scan reads GNU objdump -d lines (address, tab, bytes, tab, instruction)"

  # The platform table: the archive each machine and arch names, and the refusal of any other.
  expect_platform() { # expect_platform <case> <uname -m> <arch> <want rc> <want output>
    cases=$((cases + 1))
    out=$( (set_platform "$2" "$3" && echo "$PLATFORM $RELEASE_ARCH $HOST_CPU $TARGET_DIR v3=$SCAN_V3 glibc=$GLIBC_FLOOR cubin=$CUBIN") 2>&1) && rc=0 || rc=$?
    if [ "$rc" != "$4" ] || [ "$out" != "$5" ]; then
      echo "release self-test: platform $1: rc $rc (want $4), output: $out" >&2
      echo "  want: $5" >&2
      fails=$((fails + 1))
    fi
  }
  local x86='x86-64-v3' x86_tail='v3=1 glibc=2.34 cubin='
  expect_platform x86-default x86_64 '' 0 "linux-x86_64-cuda-sm86 sm_86 $x86 portable $x86_tail"
  expect_platform x86-sm86 x86_64 sm_86 0 "linux-x86_64-cuda-sm86 sm_86 $x86 portable $x86_tail"
  expect_platform x86-sm120a x86_64 sm_120a 0 "linux-x86_64-cuda-sm120a sm_120a $x86 portable-sm120a $x86_tail"
  expect_platform x86-sm121a x86_64 sm_121a 64 "release: no x86_64 archive for sm_121a; the x86_64 archives are sm_86 and sm_120a"
  expect_platform x86-sm120f x86_64 sm_120f 64 "release: no x86_64 archive for sm_120f; the x86_64 archives are sm_86 and sm_120a"
  expect_platform arm-default aarch64 '' 0 "linux-aarch64-cuda-sm121a sm_121a gb10 portable v3= glibc=2.39 cubin=1"
  expect_platform arm-sm121a aarch64 sm_121a 0 "linux-aarch64-cuda-sm121a sm_121a gb10 portable v3= glibc=2.39 cubin=1"
  expect_platform arm-sm86 aarch64 sm_86 64 "release: no aarch64 archive for sm_86; the aarch64 archive is sm_121a (BLOOMERY_ARCH unset on a GB10)"
  expect_platform arm-sm120a aarch64 sm_120a 64 "release: no aarch64 archive for sm_120a; the aarch64 archive is sm_121a (BLOOMERY_ARCH unset on a GB10)"
  expect_platform riscv riscv64 '' 64 "release: no Linux release for riscv64; the platforms are x86_64 and aarch64"

  # set_target: the archive's machine — the building machine's own, or aarch64 when an x86_64 machine builds the GB10
  # archive cross; any other value, and the cross value on another machine, refused by name.
  expect_target() { # expect_target <case> <uname -m> <variable> <want rc> <want output>
    cases=$((cases + 1))
    out=$( (set_target "$2" "$3" && echo "$MACHINE cross=$CROSS") 2>&1) && rc=0 || rc=$?
    if [ "$rc" != "$4" ] || [ "$out" != "$5" ]; then
      echo "release self-test: target $1: rc $rc (want $4), output: $out" >&2
      echo "  want: $5" >&2
      fails=$((fails + 1))
    fi
  }
  expect_target native-x86 x86_64 '' 0 "x86_64 cross="
  expect_target native-arm aarch64 '' 0 "aarch64 cross="
  expect_target cross x86_64 aarch64 0 "aarch64 cross=1"
  expect_target other-value x86_64 riscv64 64 "release: BLOOMERY_RELEASE_TARGET is 'aarch64' or unset, got 'riscv64'"
  expect_target cross-on-arm aarch64 aarch64 64 "release: BLOOMERY_RELEASE_TARGET=aarch64 builds the aarch64 archive on an x86_64 machine; this machine is aarch64 (unset it to build this machine's own)"

  # The cross platform line: the aarch64 row with the cross build's own target dir, triple, linker and binutils.
  expect_cross() { # expect_cross <case> <uname -m> <variable> <want rc> <want output>
    cases=$((cases + 1))
    out=$( (set_target "$2" "$3" && set_platform "$MACHINE" '' && cross_platform &&
      echo "$PLATFORM $RELEASE_ARCH $HOST_CPU $TARGET_DIR v3=$SCAN_V3 glibc=$GLIBC_FLOOR cubin=$CUBIN triple=$TRIPLE linker=$LINKER objcopy=$OBJCOPY readelf=$READELF objdump=$OBJDUMP strings=$STRINGS") 2>&1) && rc=0 || rc=$?
    if [ "$rc" != "$4" ] || [ "$out" != "$5" ]; then
      echo "release self-test: cross $1: rc $rc (want $4), output: $out" >&2
      echo "  want: $5" >&2
      fails=$((fails + 1))
    fi
  }
  expect_cross cross-line x86_64 aarch64 0 "linux-aarch64-cuda-sm121a sm_121a gb10 portable-aarch64 v3= glibc=2.39 cubin=1 triple=aarch64-unknown-linux-gnu linker=aarch64-linux-gnu-gcc objcopy=aarch64-linux-gnu-objcopy readelf=aarch64-linux-gnu-readelf objdump=aarch64-linux-gnu-objdump strings=aarch64-linux-gnu-strings"

  # cross_tools on a stub PATH: each missing piece refused by name (69), the rust target named with its command, and
  # qemu-aarch64 optional: absent it passes, present it needs QEMU_LD_PREFIX to name an aarch64 sysroot. The stubs are
  # empty executables; the stub rustc names a sysroot the test fills when the rust target is due.
  local t qp=
  t=$(mktemp -d "${TMPDIR:-/tmp}/release-tools.XXXXXX") || return 70
  stub() { printf '#!/bin/sh\n' > "$t/$1" && chmod +x "$t/$1"; }
  expect_tools() { # expect_tools <case> <want rc> <want output>: cross_tools with QEMU_LD_PREFIX=$qp
    cases=$((cases + 1))
    out=$(QEMU_LD_PREFIX=$qp PATH="$t" cross_tools 2>&1) && rc=0 || rc=$?
    if [ "$rc" != "$2" ] || [ "$out" != "$3" ]; then
      echo "release self-test: tools $1: rc $rc (want $2), output: $out" >&2
      echo "  want: $3" >&2
      fails=$((fails + 1))
    fi
  }
  local binutils='apt install binutils-aarch64-linux-gnu'
  expect_tools none 69 "release: aarch64-linux-gnu-gcc is not on PATH: apt install gcc-aarch64-linux-gnu (the cross linker)"
  stub aarch64-linux-gnu-gcc
  expect_tools no-objcopy 69 "release: aarch64-linux-gnu-objcopy is not on PATH: $binutils (the x86_64 objcopy cannot read an aarch64 binary)"
  stub aarch64-linux-gnu-objcopy
  expect_tools no-readelf 69 "release: aarch64-linux-gnu-readelf is not on PATH: $binutils (the checks read the aarch64 binary with the target's readelf)"
  stub aarch64-linux-gnu-readelf
  expect_tools no-objdump 69 "release: aarch64-linux-gnu-objdump is not on PATH: $binutils (the checks read the aarch64 binary with the target's objdump)"
  stub aarch64-linux-gnu-objdump
  expect_tools no-strings 69 "release: aarch64-linux-gnu-strings is not on PATH: $binutils (the checks read the aarch64 binary with the target's strings)"
  stub aarch64-linux-gnu-strings
  expect_tools no-ptxas 69 "release: ptxas is not on PATH: the CUDA toolkit's assembler compiles the archive's cubins (no card is needed)"
  stub ptxas
  printf '#!/bin/sh\necho %s\n' "$t/sysroot" > "$t/rustc"
  chmod +x "$t/rustc"
  expect_tools no-rust-target 69 "release: the aarch64-unknown-linux-gnu target is not installed for $t/sysroot: rustup target add aarch64-unknown-linux-gnu"
  mkdir -p "$t/sysroot/lib/rustlib/aarch64-unknown-linux-gnu"
  expect_tools present-no-qemu 0 ""
  stub qemu-aarch64
  expect_tools qemu-no-sysroot-name 69 "release: QEMU_LD_PREFIX is not set: name the aarch64 sysroot qemu-aarch64 runs the binary in (apt install libc6-arm64-cross; Ubuntu puts it at /usr/aarch64-linux-gnu)"
  qp=$t/sysroot-aarch64
  expect_tools qemu-not-a-sysroot 69 "release: QEMU_LD_PREFIX=$qp holds no lib/ld-linux-aarch64.so.1: it is not an aarch64 sysroot (apt install libc6-arm64-cross)"
  mkdir -p "$qp/lib" && : > "$qp/lib/ld-linux-aarch64.so.1"
  expect_tools present-with-qemu 0 ""
  # A --check reads the binary with the target's binutils alone.
  cases=$((cases + 1))
  out=$(PATH="$t" cross_binutils 2>&1) && rc=0 || rc=$?
  if [ "$rc" != 0 ] || [ -n "$out" ]; then
    echo "release self-test: binutils present: rc $rc (want 0), output: $out" >&2
    fails=$((fails + 1))
  fi
  rm "$t/aarch64-linux-gnu-strings"
  cases=$((cases + 1))
  out=$(PATH="$t" cross_binutils 2>&1) && rc=0 || rc=$?
  if [ "$rc" != 69 ] || [ "$out" != "release: aarch64-linux-gnu-strings is not on PATH: $binutils (the checks read the aarch64 binary with the target's strings)" ]; then
    echo "release self-test: binutils no-strings: rc $rc (want 69), output: $out" >&2
    fails=$((fails + 1))
  fi

  # The build directories: each archive's own, the device codegen's inside it, and no archive's inside another's, so builds of
  # two archs at the same time from one tree share nothing. The plan line is what --dry-run prints.
  expect_dirs() { # expect_dirs <case> <uname -m> <variable> <arch> <want>
    cases=$((cases + 1))
    out=$( (set_target "$2" "$3" && set_platform "$MACHINE" "$4" && { [ -z "$CROSS" ] || cross_platform; } && build_dirs /r &&
      NAME=n DIST=/r/target/release-dist && plan_lines) 2>&1) && rc=0 || rc=$?
    if [ "$rc" != 0 ] || [ "$out" != "$5" ]; then
      echo "release self-test: dirs $1: rc $rc (want 0), output: $out" >&2
      echo "  want: $5" >&2
      fails=$((fails + 1))
    fi
  }
  expect_dirs sm86 x86_64 '' sm_86 "release: dry run: n (arch sm_86, x86_64) builds in /r/target/portable
release: dry run: its device files (PTX, .ll, .oxide-artifacts/) go to /r/target/portable/oxide-out, not the tree's root
release: dry run: it packs /r/target/release-dist/n.tar.gz; nothing was built"
  expect_dirs sm120a x86_64 '' sm_120a "release: dry run: n (arch sm_120a, x86_64) builds in /r/target/portable-sm120a
release: dry run: its device files (PTX, .ll, .oxide-artifacts/) go to /r/target/portable-sm120a/oxide-out, not the tree's root
release: dry run: it packs /r/target/release-dist/n.tar.gz; nothing was built"
  expect_dirs cross x86_64 aarch64 '' "release: dry run: n (arch sm_121a, aarch64) builds in /r/target/portable-aarch64
release: dry run: its device files (PTX, .ll, .oxide-artifacts/) go to /r/target/portable-aarch64/oxide-out, not the tree's root
release: dry run: it packs /r/target/release-dist/n.tar.gz; nothing was built"
  local a_dir a_out b_dir b_out pair d o
  read -r a_dir a_out <<< "$(set_platform x86_64 sm_86 && build_dirs /r && echo "$CARGO_TARGET_DIR $OXIDE_OUT")"
  read -r b_dir b_out <<< "$(set_platform x86_64 sm_120a && build_dirs /r && echo "$CARGO_TARGET_DIR $OXIDE_OUT")"
  cases=$((cases + 1))
  if [ -z "$a_dir" ] || [ "$a_dir" = "$b_dir" ] || [ "$a_out" = "$b_out" ]; then
    echo "release self-test: the sm_86 and sm_120a archives share a build directory: $a_dir $a_out / $b_dir $b_out" >&2
    fails=$((fails + 1))
  fi
  for pair in "$a_dir $b_out" "$b_dir $a_out"; do
    read -r d o <<< "$pair"
    cases=$((cases + 1))
    case "$o/" in "$d/"*)
      echo "release self-test: $o lies inside $d, another archive's build directory" >&2
      fails=$((fails + 1)) ;; esac
  done

  # The tree check comes before anything else is decided or built: a tree whose README names another release is refused by
  # name, and on a machine that is no Linux release machine too (the platform, which this Mac refuses, is not reached).
  local tc="$t/tree"
  mkdir -p "$tc/tools/release" "$tc/crates/serve" "$tc/crates/gpu-gates"
  cp "$0" "$tc/tools/release/build.sh"
  cp "$(dirname "$0")/publish.sh" "$tc/tools/release/publish.sh"
  printf '[package]\nname = "bloomery-serve"\nversion = "0.0.2"\n' > "$tc/crates/serve/Cargo.toml"
  printf '[package]\nname = "bloomery-gpu-gates"\nversion = "0.0.2"\n' > "$tc/crates/gpu-gates/Cargo.toml"
  printf '## Status (0.0.1)\nbloomery-0.0.1-linux-x86_64-cuda-sm86.tar.gz\nreleases/tag/v0.0.1)\n' > "$tc/README.md"
  expect_tree_check() { # expect_tree_check <case> <build.sh copy> <want rc> <want text>
    cases=$((cases + 1))
    out=$(cd "$tc" && bash "$2" --dry-run 0.0.2 abc123456789 2>&1) && rc=0 || rc=$?
    if [ "$rc" != "$3" ] || [[ "$out" != *"$4"* ]] || [[ "$out" == *"dry run: "* ]]; then
      echo "release self-test: tree check $1: rc $rc (want $3, with '$4' and no plan line), output: $out" >&2
      fails=$((fails + 1))
    fi
  }
  expect_tree_check stale-readme "$tc/tools/release/build.sh" 1 "README.md does not name '## Status (0.0.2)'"
  python3 -I - "$0" "$tc/tools/release/build-nocheck.sh" << 'PY' || { echo "release self-test: the tree check line is not in the code" >&2; fails=$((fails + 1)); }
import sys

src, dst = sys.argv[1:3]
s = open(src).read()
old = '\nif [ -z "$CHECK_ONLY" ]; then\n  bash "$ROOT/tools/release/publish.sh" --tree-check "$VERSION" ${BUILD_STAGE:+--crates-only} || exit 1\nfi\n'
i = s.find(old)
if i < 0:
    sys.exit(1)
open(dst, "w").write(s[:i] + "\n" + s[i + len(old):])
PY
  cases=$((cases + 1))
  out=$(cd "$tc" && bash tools/release/build-nocheck.sh --dry-run 0.0.2 abc123456789 2>&1) && rc=0 || rc=$?
  if [[ "$out" == *"does not name '## Status (0.0.2)'"* ]]; then
    echo "release self-test: the build without the tree check still refused the stale README: $out" >&2
    fails=$((fails + 1))
  fi

  # --stage w: the crates' versions are held and the README is not (a watermark's README names the last release). The tree above has
  # crates at 0.0.2 and a README at 0.0.1. A build on this machine goes on past the tree check to the platform (this Mac refuses it,
  # a Linux box reaches the plan), so these cases read the tree check's line and the refusals, never the rc past it.
  expect_args() { # expect_args <case> <build.sh copy> <want rc or ''> <want text> <text it must not hold or ''> <args…>
    local name=$1 copy=$2 want_rc=$3 want=$4 unwanted=$5
    shift 5
    cases=$((cases + 1))
    out=$(cd "$tc" && bash "$copy" "$@" 2>&1) && rc=0 || rc=$?
    if { [ -n "$want_rc" ] && [ "$rc" != "$want_rc" ]; } || [[ "$out" != *"$want"* ]] || { [ -n "$unwanted" ] && [[ "$out" == *"$unwanted"* ]]; }; then
      echo "release self-test: stage $name: rc $rc (want ${want_rc:-any}), want '$want'${unwanted:+ without '$unwanted'}; output: $out" >&2
      fails=$((fails + 1))
    fi
  }
  stage_cases() { # stage_cases <build.sh copy>: the cases of the stage, all against that copy
    expect_args "w passes a stale README" "$1" '' "tree check ok (crates only)" "does not name" --dry-run --stage w 0.0.2 abc123456789
    expect_args "w, either order" "$1" '' "tree check ok (crates only)" "does not name" --stage w --dry-run 0.0.2 abc123456789
    expect_args "w refuses a crate behind the release" "$1" 1 "crates/serve says version 0.0.2, not 0.0.3" "tree check ok" --dry-run --stage w 0.0.3 abc123456789
    expect_args "w builds with the stage and no dry run, the same tree check" "$1" 1 "crates/serve says version 0.0.2, not 0.0.3" "tree check ok" --stage w 0.0.3 abc123456789
    expect_args "no stage refuses the stale README" "$1" 1 "README.md does not name '## Status (0.0.2)'" "tree check ok" --dry-run 0.0.2 abc123456789
    expect_args "no stage, no dry run, refuses the stale README" "$1" 1 "README.md does not name '## Status (0.0.2)'" "tree check ok" 0.0.2 abc123456789
    expect_args "a stage that is not w" "$1" 64 "--stage takes w, got 'x'" "tree check ok" --stage x 0.0.2 abc123456789
    expect_args "a stage with no name" "$1" 64 "--stage needs a stage (w)" "" --stage
    expect_args "an option it does not take" "$1" 64 "unknown argument --no-such-option" "" --no-such-option 0.0.2 abc123456789
    expect_args "a commit missing" "$1" 64 "usage: tools/release/build.sh [--stage w] [--dry-run] <version> <commit>" "tree check ok" --stage w 0.0.2
  }
  stage_cases "$tc/tools/release/build.sh"
  # mutate_build <name> <old text> <new text>: the copy of this script in the temporary tree with the first <old text> replaced; its
  # stage cases must go red
  mutate_build() {
    local before=$fails
    python3 -I - "$0" "$tc/tools/release/build-mutant.sh" "$2" "$3" << 'PY' || { echo "release self-test: mutant $1: the text to replace is not in the code" >&2; fails=$((fails + 1)); return 0; }
import sys

src, dst, old, new = sys.argv[1:5]
s = open(src).read()
i = s.find(old)
if i < 0:
    sys.exit(1)
open(dst, "w").write(s[:i] + new + s[i + len(old):])
PY
    fails=0
    stage_cases "$tc/tools/release/build-mutant.sh" 2> /dev/null
    if [ "$fails" = 0 ]; then
      fails=$before
      echo "release self-test: mutant $1 left every stage case green: the cases do not hold it" >&2
      fails=$((fails + 1))
    else
      fails=$before
    fi
  }
  mutate_build "the stage is ignored" $' ${BUILD_STAGE:+--crates-only} || exit 1\n' $' || exit 1\n'
  mutate_build "the stage skips the tree check" $'\nif [ -z "$CHECK_ONLY" ]; then\n  bash "$ROOT/tools/release/publish.sh"' $'\nif [ -z "$CHECK_ONLY" ] && [ -z "$BUILD_STAGE" ]; then\n  bash "$ROOT/tools/release/publish.sh"'
  mutate_build "any stage name is taken" '[ "$2" = w ] ||' 'true ||'

  # check_version on stub binaries: the line it prints, the runner that runs a cross binary, and each refusal by name. The
  # stub runner says so on stderr and execs its argument, so a case that must run under one shows it.
  printf '#!/bin/sh\necho "bloomery-serve 0.3.12 (commit abc123456789)"\n' > "$t/serve"
  printf '#!/bin/sh\necho "bloomery-serve 0.3.12" >&2\nexit 7\n' > "$t/serve-dies"
  printf '#!/bin/sh\necho "ran under the runner" >&2\nexec "$@"\n' > "$t/runner"
  chmod +x "$t/serve" "$t/serve-dies" "$t/runner"
  expect_version() { # expect_version <case> <binary> <version> <commit> <want rc> <want output> [runner…]
    cases=$((cases + 1))
    local name=$1 binary=$2 version=$3 commit=$4 want_rc=$5 want=$6
    shift 6
    out=$(check_version "$t/$binary" "$version" "$commit" "$@" 2>&1) && rc=0 || rc=$?
    if [ "$rc" != "$want_rc" ] || [ "$out" != "$want" ]; then
      echo "release self-test: version $name: rc $rc (want $want_rc), output: $out" >&2
      echo "  want: $want" >&2
      fails=$((fails + 1))
    fi
  }
  local line='bloomery-serve 0.3.12 (commit abc123456789)'
  expect_version native serve 0.3.12 abc123456789 0 "release: serve --version: $line"
  expect_version under-runner serve 0.3.12 abc123456789 0 "ran under the runner
release: serve --version: $line" "$t/runner"
  expect_version other-commit serve 0.3.12 deadbeefdead 1 "release: serve --version does not name the commit deadbeefdead: $line"
  expect_version other-version serve 0.3.13 abc123456789 1 "release: serve --version does not name the version 0.3.13 (bump crates/serve and crates/gpu-gates first): $line"
  expect_version dies serve-dies 0.3.12 abc123456789 1 "bloomery-serve 0.3.12
release: serve-dies --version failed"
  expect_version dies-under-runner serve-dies 0.3.12 abc123456789 1 "ran under the runner
bloomery-serve 0.3.12
release: serve-dies --version failed under $t/runner" "$t/runner"

  # cross_version: a cross binary runs under qemu-aarch64 when it is on PATH, and is named "not run" when it is not.
  expect_cross_version() { # expect_cross_version <case> <version> <commit> <want rc> <want output>
    cases=$((cases + 1))
    out=$(PATH="$t" cross_version "$t/serve" "$2" "$3" 2>&1) && rc=0 || rc=$?
    if [ "$rc" != "$4" ] || [ "$out" != "$5" ]; then
      echo "release self-test: cross version $1: rc $rc (want $4), output: $out" >&2
      echo "  want: $5" >&2
      fails=$((fails + 1))
    fi
  }
  cp "$t/runner" "$t/qemu-aarch64"
  expect_cross_version under-qemu 0.3.12 abc123456789 0 "ran under the runner
release: serve --version: $line"
  expect_cross_version under-qemu-other-commit 0.3.12 deadbeefdead 1 "ran under the runner
release: serve --version does not name the commit deadbeefdead: $line"
  rm "$t/qemu-aarch64"
  expect_cross_version no-qemu 0.3.12 abc123456789 0 "release: serve --version not run: qemu-aarch64 is not on PATH (apt install qemu-user); a run on a GB10 checks it"
  rm -rf "$t"

  if [ "$fails" = 0 ]; then
    echo "release: self-test ok ($cases cases)"
  else
    echo "release: self-test $fails of $cases cases failed" >&2
    return 1
  fi
}
if [ -n "$SELF_TEST" ]; then
  self_test
  exit $?
fi

# The tree is held to what the publish will hold it to, first: nothing below runs for a tree that cannot publish.
if [ -z "$CHECK_ONLY" ]; then
  bash "$ROOT/tools/release/publish.sh" --tree-check "$VERSION" ${BUILD_STAGE:+--crates-only} || exit 1
fi

# The archive's machine: the building machine's own, or the cross one BLOOMERY_RELEASE_TARGET names. The archive's
# arch: a build's is BLOOMERY_ARCH, as tools/arch.sh resolves it, except that a cross build takes its row's own (the
# building machine's selection names none of the aarch64 archive's arches, and tools/box.sh exports it into the box
# command anyway); a check's is its argument (empty: the archive's own).
set_target "$(uname -m)" "${BLOOMERY_RELEASE_TARGET:-}" || exit $?
if [ -n "$CHECK_ONLY" ]; then
  ARCH=$CHECK_ARCH
elif [ -z "$CROSS" ]; then
  # shellcheck source=tools/arch.sh
  source "$ROOT/tools/arch.sh"
  bloomery_arch release/build.sh || exit $?
else
  ARCH=
fi
set_platform "$MACHINE" "$ARCH" || exit $?
OBJCOPY=objcopy READELF=readelf OBJDUMP=objdump STRINGS=strings
if [ -n "$CROSS" ]; then
  ARCH=$RELEASE_ARCH
  cross_platform
  if [ -n "$CHECK_ONLY" ]; then cross_binutils || exit $?; else cross_tools || exit $?; fi
fi
ALLOWED_NEEDED="^(libc\\.so\\.6|libm\\.so\\.6|libgcc_s\\.so\\.1|$LOADER)\$"

if [ -n "$CHECK_ONLY" ]; then
  BIN_DIR=$CHECK_ONLY
else
  NAME="bloomery-$VERSION-$PLATFORM"
  DIST="$ROOT/target/release-dist"
  STAGE="$DIST/$NAME"
  build_dirs "$ROOT"
  if [ -n "$DRY" ]; then
    plan_lines
    exit 0
  fi
  mkdir -p "$OXIDE_OUT"
  export CARGO_TARGET_DIR CUDA_OXIDE_PTX_DIR="$OXIDE_OUT"
  target_args=()
  if [ -n "$CROSS" ]; then
    # The cubin packer is a host tool: no device code, so the host's plain cargo builds it into the cross dir's host
    # half, before RUSTFLAGS names the target machine's CPU. The cross build's own binaries land under the triple.
    env -u RUSTFLAGS cargo build -p bloomery-gpu-gates --release --bin oxart_ptxas
    BIN_DIR="$CARGO_TARGET_DIR/$TRIPLE/release"
    export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER="$LINKER"
    target_args=(--target "$TRIPLE")
  else
    BIN_DIR="$CARGO_TARGET_DIR/release"
  fi
  # cargo-oxide appends RUSTFLAGS after the project's extra-rustflags, and rustc keeps the last -C target-cpu.
  export RUSTFLAGS="-C target-cpu=$HOST_CPU"
  bin_args=()
  for b in "${BINS[@]}"; do bin_args+=(--bin "$b"); done
  # oxart_jit links the cubins on the building machine's card, so only the GB10's own build wants it; cross, ptxas on
  # the host makes them and no oxart_jit is built.
  if [ -n "$CUBIN" ] && [ -z "$CROSS" ]; then bin_args+=(--bin oxart_jit); fi
  # The commit goes to cargo alone: the serve build script reads it, and a running binary never does.
  BLOOMERY_GIT_COMMIT="$COMMIT" cargo oxide build --arch "$ARCH" -- -p bloomery-gpu-gates --features glm5next,clef,vision,mimo2,solar2,minimaxm3,flint,mistrallarge4,gemma-embedding2,hf,checkpoint --release "${bin_args[@]}" ${target_args[@]+"${target_args[@]}"}
fi

fail=0
for b in "${BINS[@]}"; do
  f="$BIN_DIR/$b"
  [ -x "$f" ] || { echo "release: $f was not built" >&2; exit 1; }

  machine=$("$READELF" -h "$f" | sed -n 's/^ *Machine: *//p')
  if [ "$machine" != "$ELF_MACHINE" ]; then
    echo "release: $b is a binary for '$machine', not this platform's '$ELF_MACHINE' ($PLATFORM)" >&2
    fail=1
  fi

  strong=$("$READELF" -V "$f" | awk '/Name: GLIBC_/ && /Flags: none/ {sub("GLIBC_", "", $3); print $3}' | sort -V | tail -1)
  if [ "$(printf '%s\n%s\n' "$strong" "$GLIBC_FLOOR" | sort -V | tail -1)" != "$GLIBC_FLOOR" ]; then
    echo "release: $b needs GLIBC_$strong through a strong symbol version, past the floor $GLIBC_FLOOR" >&2
    fail=1
  fi

  bad_needed=$("$READELF" -d "$f" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p' | grep -Ev "$ALLOWED_NEEDED" || true)
  if [ -n "$bad_needed" ]; then
    echo "release: $b links a library outside the allowed set: $bad_needed" >&2
    fail=1
  fi

  # Instructions past x86-64-v3 may sit only in functions that run behind a runtime CPU check (sha2's and sha1's SHA-NI arms,
  # selected by cpufeatures; crc32fast's AVX-512 and AVX2+VPCLMULQDQ arms under png, selected by is_x86_feature_detected!;
  # qdot's Q4_K AVX-512 VNNI bodies, selected by qdot::q4k_avx512); anywhere else they mean the host code was built for this box's CPU.
  past_v3=
  if [ -n "$SCAN_V3" ] && ! past_v3=$("$OBJDUMP" -d -C "$f" | scan_past_v3); then
    echo "release: $b: the host-code scan failed on its objdump listing" >&2
    fail=1
  elif [ -n "$past_v3" ]; then
    echo "release: $b carries instructions past x86-64-v3 outside a runtime-dispatched function:" >&2
    echo "$past_v3" >&2
    fail=1
  fi

  # A cubin archive's binary reads its code from `.oxart` and keeps the PTX it came from as `.oxart.ptx`.
  if [ -n "$CHECK_ONLY" ] && [ -n "$CUBIN" ] &&
    [ "$("$READELF" -S -W "$f" | grep -cE ' \.oxart(\.ptx)? ')" != 2 ]; then
    echo "release: $b does not carry both .oxart (the cubins) and .oxart.ptx (their PTX)" >&2
    fail=1
  fi

  targets=$("$STRINGS" -n 8 "$f" | sed -n 's/^\.target[[:space:]]\+\([a-z0-9_]*\).*/\1/p' | sort -u | tr '\n' ' ')
  if [ "$targets" != "$RELEASE_ARCH " ]; then
    echo "release: $b embeds PTX targets [$targets], not $RELEASE_ARCH alone" >&2
    fail=1
  fi

  echo "release: $b glibc_strong_max=$strong needed=$("$READELF" -d "$f" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p' | tr '\n' ' ')ptx=[$targets] bytes=$(stat -c %s "$f")"
done
[ "$fail" = 0 ] || exit 1
if [ -n "$CHECK_ONLY" ]; then
  echo "release: checks passed on $BIN_DIR"
  exit 0
fi

# The packed binary runs: --version names the commit and the version. A cross binary runs under qemu-aarch64 (user mode,
# QEMU_LD_PREFIX its sysroot), the only run of the aarch64 code on the build host; without qemu it is named "not run".
for b in "${BINS[@]}"; do
  if [ -n "$CROSS" ]; then
    cross_version "$BIN_DIR/$b" "$VERSION" "$COMMIT" || exit $?
  else
    check_version "$BIN_DIR/$b" "$VERSION" "$COMMIT" || exit $?
  fi
done

rm -rf "$STAGE"
mkdir -p "$STAGE/bin"
for b in "${BINS[@]}"; do cp "$BIN_DIR/$b" "$STAGE/bin/"; done
if [ -n "$CUBIN" ]; then
  for b in "${BINS[@]}"; do
    f="$STAGE/bin/$b"
    "$OBJCOPY" --dump-section .oxart="$f.ptx.oxart" "$f"
    if [ -n "$CROSS" ]; then
      # ptxas compiles the bundles on the build host: no card, no gate lock.
      "$CARGO_TARGET_DIR/release/oxart_ptxas" "$f.ptx.oxart" "$f.cubin.oxart"
    else
      bash tools/gpu-gate.sh --executable "$CARGO_TARGET_DIR/release/oxart_jit" --cubin "$f.ptx.oxart" "$f.cubin.oxart"
    fi
    "$OBJCOPY" --rename-section .oxart=.oxart.ptx --add-section .oxart="$f.cubin.oxart" "$f"
    rm "$f.ptx.oxart" "$f.cubin.oxart"
  done
  bash "$0" --check "$STAGE/bin"
  # The objcopy rewrote the cross binary: it must still run.
  if [ -n "$CROSS" ]; then
    for b in "${BINS[@]}"; do cross_version "$STAGE/bin/$b" "$VERSION" "$COMMIT" || exit $?; done
  fi
fi

cp LICENSE THIRD_PARTY_NOTICES.md docs/serving.md "$STAGE/"
cp tools/release/README.release.md "$STAGE/README.md"
printf '%s\n%s\n' "$VERSION" "$COMMIT" > "$STAGE/VERSION"
(cd "$DIST" && tar -czf "$NAME.tar.gz" "$NAME" && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256")
echo "release: $DIST/$NAME.tar.gz $(stat -c %s "$DIST/$NAME.tar.gz") bytes"
cat "$DIST/$NAME.tar.gz.sha256"
