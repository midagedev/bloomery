#!/usr/bin/env bash
# Builds the prebuilt Linux release on the box and packs it: `just release-build <version>` runs it through
# tools/box.sh with the Mac's commit. It builds into its own target dir (target/portable, below), so the measure runners'
# target/release is never touched.
#
# The release is the building machine's platform (uname -m) and CUDA arch (BLOOMERY_ARCH, tools/arch.sh), one archive each:
#   - x86_64, sm_86: bloomery-<v>-linux-x86_64-cuda-sm86, sm_86 PTX the driver compiles for any card of cc 8.6 or newer,
#     host code for x86-64-v3; built on the box, BLOOMERY_ARCH unset;
#   - x86_64, sm_120a: bloomery-<v>-linux-x86_64-cuda-sm120a, sm_120a PTX (the Blackwell consumer parts' native FP4 and
#     MXFP8 kernels, which load on cc 12.0 alone), the same host code; built on the box with BLOOMERY_ARCH=sm_120a, into
#     its own target dir (target/portable-sm120a) so neither archive's build rebuilds the other's. A build machine with no
#     cc 12.0 card cannot compile the PTX to cubins, so it stays PTX and the driver compiles it on the card's first start;
#   - aarch64: bloomery-<v>-linux-aarch64-cuda-sm121a, sm_121a PTX (GB10's native FP4, which loads on cc 12.1 alone),
#     host code for GB10's cores; built on a GB10 (BLOOMERY_BOX names one, or BLOOMERY_BOX_LOCAL=1 on the Spark itself).
#     The packed binary carries the driver's JIT of that PTX as cubins: oxart_jit --cubin links each `.oxart` bundle on
#     the building GB10's card, the result becomes the `.oxart` section the loader reads, and the PTX stays beside it as
#     `.oxart.ptx`, which bloomery-gpu loads for a bundle whose cubin the driver refuses. A first start then compiles
#     nothing.
#
# The failures it blocks, each checked on the built binaries before anything is packed:
#   - a binary for another machine than the platform's (its ELF header);
#   - a binary that needs a glibc newer than GLIBC_FLOOR through a strong (non-WEAK) symbol version, so it would not
#     start on the oldest distribution the release names;
#   - a binary that links a shared library outside the allowed set (libcuda must be loaded at run time, never linked);
#   - on x86_64, host code built for this box's CPU (znver3) instead of x86-64-v3: an instruction past x86-64-v3 outside a
#     function that runs behind a runtime CPU check;
#   - a binary with no embedded PTX of the platform's arch, or PTX for another target.
# A BLOOMERY_ARCH that is no arch of the machine's archives (tools/arch.sh) ends a build with 64 before anything is built;
# `--check` reads the binaries and never the variable: the archive's arch is its argument, and an x86_64 binary checked
# with none is held to sm_86.
# Usage: tools/release/build.sh <version> <commit>
#        tools/release/build.sh --check <dir> [<arch>]   (the binary checks alone, on <dir>/<bin>, for the archive of
#                                                         <arch>, default the machine's sm_86 or sm_121a; builds and packs nothing)
#        tools/release/build.sh --self-test     (the host-code scan on fixture listings and the platform table; check-recipes runs it)
# The tarball and its .sha256 keep the names this script writes ($NAME.tar.gz…): install.sh resolves the release by
# them and the Homebrew tap's formula pins them, so `gh release upload` must be given the files as named, not copies
# under other names. The version the tarball carries is the serve crate's, so a release bumps that crate first.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
CHECK_ONLY= SELF_TEST= CHECK_ARCH=
if [ "${1:-}" = --check ]; then
  CHECK_ONLY=${2:?usage: tools/release/build.sh --check <dir> [<arch>]}
  CHECK_ARCH=${3:-}
  [ $# -le 3 ] || { echo "usage: tools/release/build.sh --check <dir> [<arch>]" >&2; exit 64; }
elif [ "${1:-}" = --self-test ]; then
  [ $# = 1 ] || { echo "usage: tools/release/build.sh --self-test" >&2; exit 64; }
  SELF_TEST=1
else
  VERSION=${1:?usage: tools/release/build.sh <version> <commit>}
  COMMIT=${2:?usage: tools/release/build.sh <version> <commit>}
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
          echo "release: no x86_64 archive for $arch; the x86_64 archives are sm_86 (BLOOMERY_ARCH unset) and sm_120a" >&2
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
  expect_platform x86-sm121a x86_64 sm_121a 64 "release: no x86_64 archive for sm_121a; the x86_64 archives are sm_86 (BLOOMERY_ARCH unset) and sm_120a"
  expect_platform x86-sm120f x86_64 sm_120f 64 "release: no x86_64 archive for sm_120f; the x86_64 archives are sm_86 (BLOOMERY_ARCH unset) and sm_120a"
  expect_platform arm-default aarch64 '' 0 "linux-aarch64-cuda-sm121a sm_121a gb10 portable v3= glibc=2.39 cubin=1"
  expect_platform arm-sm121a aarch64 sm_121a 0 "linux-aarch64-cuda-sm121a sm_121a gb10 portable v3= glibc=2.39 cubin=1"
  expect_platform arm-sm86 aarch64 sm_86 64 "release: no aarch64 archive for sm_86; the aarch64 archive is sm_121a (BLOOMERY_ARCH unset on a GB10)"
  expect_platform arm-sm120a aarch64 sm_120a 64 "release: no aarch64 archive for sm_120a; the aarch64 archive is sm_121a (BLOOMERY_ARCH unset on a GB10)"
  expect_platform riscv riscv64 '' 64 "release: no Linux release for riscv64; the platforms are x86_64 and aarch64"

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

# The archive's arch: a build's is BLOOMERY_ARCH, as tools/arch.sh resolves it; a check's is its argument (empty: the
# machine's own).
if [ -n "$CHECK_ONLY" ]; then
  ARCH=$CHECK_ARCH
else
  # shellcheck source=tools/arch.sh
  source "$ROOT/tools/arch.sh"
  bloomery_arch release/build.sh || exit $?
fi
set_platform "$(uname -m)" "$ARCH" || exit $?
ALLOWED_NEEDED="^(libc\\.so\\.6|libm\\.so\\.6|libgcc_s\\.so\\.1|$LOADER)\$"

if [ -n "$CHECK_ONLY" ]; then
  BIN_DIR=$CHECK_ONLY
else
  NAME="bloomery-$VERSION-$PLATFORM"
  DIST="$ROOT/target/release-dist"
  STAGE="$DIST/$NAME"
  export CARGO_TARGET_DIR="$ROOT/target/$TARGET_DIR"
  BIN_DIR="$CARGO_TARGET_DIR/release"
  # cargo-oxide appends RUSTFLAGS after the project's extra-rustflags, and rustc keeps the last -C target-cpu.
  export RUSTFLAGS="-C target-cpu=$HOST_CPU"
  bin_args=()
  for b in "${BINS[@]}"; do bin_args+=(--bin "$b"); done
  [ -z "$CUBIN" ] || bin_args+=(--bin oxart_jit)
  # The commit goes to cargo alone: the serve build script reads it, and a running binary never does.
  BLOOMERY_GIT_COMMIT="$COMMIT" cargo oxide build --arch "$ARCH" -- -p bloomery-gpu-gates --features glm5next,clef,vision,mimo2,solar2,minimaxm3,flint,gemma-embedding2,hf,checkpoint --release "${bin_args[@]}"
fi

fail=0
for b in "${BINS[@]}"; do
  f="$BIN_DIR/$b"
  [ -x "$f" ] || { echo "release: $f was not built" >&2; exit 1; }

  machine=$(readelf -h "$f" | sed -n 's/^ *Machine: *//p')
  if [ "$machine" != "$ELF_MACHINE" ]; then
    echo "release: $b is a binary for '$machine', not this platform's '$ELF_MACHINE' ($PLATFORM)" >&2
    fail=1
  fi

  strong=$(readelf -V "$f" | awk '/Name: GLIBC_/ && /Flags: none/ {sub("GLIBC_", "", $3); print $3}' | sort -V | tail -1)
  if [ "$(printf '%s\n%s\n' "$strong" "$GLIBC_FLOOR" | sort -V | tail -1)" != "$GLIBC_FLOOR" ]; then
    echo "release: $b needs GLIBC_$strong through a strong symbol version, past the floor $GLIBC_FLOOR" >&2
    fail=1
  fi

  bad_needed=$(readelf -d "$f" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p' | grep -Ev "$ALLOWED_NEEDED" || true)
  if [ -n "$bad_needed" ]; then
    echo "release: $b links a library outside the allowed set: $bad_needed" >&2
    fail=1
  fi

  # Instructions past x86-64-v3 may sit only in functions that run behind a runtime CPU check (sha2's and sha1's SHA-NI arms,
  # selected by cpufeatures; crc32fast's AVX-512 and AVX2+VPCLMULQDQ arms under png, selected by is_x86_feature_detected!;
  # qdot's Q4_K AVX-512 VNNI bodies, selected by qdot::q4k_avx512); anywhere else they mean the host code was built for this box's CPU.
  past_v3=
  if [ -n "$SCAN_V3" ] && ! past_v3=$(objdump -d -C "$f" | scan_past_v3); then
    echo "release: $b: the host-code scan failed on its objdump listing" >&2
    fail=1
  elif [ -n "$past_v3" ]; then
    echo "release: $b carries instructions past x86-64-v3 outside a runtime-dispatched function:" >&2
    echo "$past_v3" >&2
    fail=1
  fi

  # A cubin archive's binary reads its code from `.oxart` and keeps the PTX it came from as `.oxart.ptx`.
  if [ -n "$CHECK_ONLY" ] && [ -n "$CUBIN" ] &&
    [ "$(readelf -S -W "$f" | grep -cE ' \.oxart(\.ptx)? ')" != 2 ]; then
    echo "release: $b does not carry both .oxart (the cubins) and .oxart.ptx (their PTX)" >&2
    fail=1
  fi

  targets=$(strings -n 8 "$f" | sed -n 's/^\.target[[:space:]]\+\([a-z0-9_]*\).*/\1/p' | sort -u | tr '\n' ' ')
  if [ "$targets" != "$RELEASE_ARCH " ]; then
    echo "release: $b embeds PTX targets [$targets], not $RELEASE_ARCH alone" >&2
    fail=1
  fi

  echo "release: $b glibc_strong_max=$strong needed=$(readelf -d "$f" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p' | tr '\n' ' ')ptx=[$targets] bytes=$(stat -c %s "$f")"
done
[ "$fail" = 0 ] || exit 1
if [ -n "$CHECK_ONLY" ]; then
  echo "release: checks passed on $BIN_DIR"
  exit 0
fi

for b in "${BINS[@]}"; do
  v=$("$CARGO_TARGET_DIR/release/$b" --version)
  case "$v" in
    *"$COMMIT"*) ;;
    *) echo "release: $b --version does not name the commit $COMMIT: $v" >&2; exit 1 ;;
  esac
  # The tarball's name says VERSION, so the binary must say it too (the serve crate's version is bumped first).
  case "$v" in
    *" $VERSION ("*) echo "release: $b --version: $v" ;;
    *) echo "release: $b --version does not name the version $VERSION (bump crates/serve and crates/gpu-gates first): $v" >&2; exit 1 ;;
  esac
done

rm -rf "$STAGE"
mkdir -p "$STAGE/bin"
for b in "${BINS[@]}"; do cp "$CARGO_TARGET_DIR/release/$b" "$STAGE/bin/"; done
if [ -n "$CUBIN" ]; then
  for b in "${BINS[@]}"; do
    f="$STAGE/bin/$b"
    objcopy --dump-section .oxart="$f.ptx.oxart" "$f"
    bash tools/gpu-gate.sh --executable "$CARGO_TARGET_DIR/release/oxart_jit" --cubin "$f.ptx.oxart" "$f.cubin.oxart"
    objcopy --rename-section .oxart=.oxart.ptx --add-section .oxart="$f.cubin.oxart" "$f"
    rm "$f.ptx.oxart" "$f.cubin.oxart"
  done
  bash "$0" --check "$STAGE/bin"
fi
cp LICENSE THIRD_PARTY_NOTICES.md "$STAGE/"
cp tools/release/README.release.md "$STAGE/README.md"
printf '%s\n%s\n' "$VERSION" "$COMMIT" > "$STAGE/VERSION"
(cd "$DIST" && tar -czf "$NAME.tar.gz" "$NAME" && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256")
echo "release: $DIST/$NAME.tar.gz $(stat -c %s "$DIST/$NAME.tar.gz") bytes"
cat "$DIST/$NAME.tar.gz.sha256"
