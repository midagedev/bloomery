# portable_bin <tarball>: prints the tree-relative path of the bloomery-serve a release tarball packs, the binary of
# tools/release/build.sh's set_platform TARGET_DIR for the tarball's platform (target/portable-sm120a for the sm_120a archive,
# target/portable for every other, an empty name included). Sourced by tools/release/sitting.sh and tools/ref/depth-serve.sh,
# which hold a @portable binary to the tarball's member by sha256.
portable_bin() {
  case ${1##*/} in
    *-linux-x86_64-cuda-sm120a.tar.gz) printf '%s\n' target/portable-sm120a/release/bloomery-serve ;;
    *) printf '%s\n' target/portable/release/bloomery-serve ;;
  esac
}
