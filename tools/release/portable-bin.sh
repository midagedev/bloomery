# portable_bin <tarball>: prints the tree-relative path of the bloomery-serve a release tarball packs. The archives whose
# binaries tools/release/build.sh packs with cubins (sm_120a, and the aarch64 GB10 one: tools/oxart-pack.sh rewrites the
# binary in the archive's stage dir, so the build's own target/portable* binary is not the member) name the stage dir's
# binary, target/release-dist/<archive name>/bin/bloomery-serve; every other archive, an empty name included, names the
# binary of build.sh's set_platform TARGET_DIR, target/portable. Sourced by tools/release/sitting.sh and
# tools/ref/depth-serve.sh, which hold a @portable binary to the tarball's member by sha256.
portable_bin() {
  local name=${1##*/}
  case $name in
    *-linux-x86_64-cuda-sm120a.tar.gz | *-linux-aarch64-cuda-sm121a.tar.gz)
      printf 'target/release-dist/%s/bin/bloomery-serve\n' "${name%.tar.gz}" ;;
    *) printf '%s\n' target/portable/release/bloomery-serve ;;
  esac
}
