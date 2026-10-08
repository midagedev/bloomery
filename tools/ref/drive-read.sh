# shellcheck shell=bash
# The drive a model file is read from, and what the drive was asked to read (nvtier-read.sh,
# depth-qwen3moe.sh): the device is read from the file's mount, never guessed, and its counters are the
# kernel's own (/sys/block/<dev>/stat, field 3: sectors read, 512 B each — the number /proc/diskstats
# prints as its sixth column). Sourced; it defines functions and runs nothing.

# drive_dev <file>: the whole-disk device of the mount <file>'s directory sits on, on stdout; rc 2 with the
# reason on stderr when the mount or its counters cannot be read (a drive whose counters cannot be read is
# no instrument).
drive_dev() {
  local dir src dev
  dir=$(dirname "$1")
  src=$(findmnt -n -o SOURCE --target "$dir") || { echo "drive-read.sh: findmnt cannot name the mount of $dir" >&2; return 2; }
  dev=$(lsblk -n -o PKNAME "$src" 2> /dev/null | head -n1 || true)
  [ -n "$dev" ] || dev=$(basename "$src")
  [ -r "/sys/block/$dev/stat" ] || { echo "drive-read.sh: no counters at /sys/block/$dev/stat (mount source $src, device $dev)" >&2; return 2; }
  echo "$dev"
}

# drive_sectors <dev>: the sectors read from the device so far.
drive_sectors() { awk '{print $3}' "/sys/block/$1/stat"; }

# drive_col <sectors before> <sectors after> <dev>: a row's drive column, the bytes the device was asked
# to read between the two readings — every reader of the device, this arm's and a foreign one alike.
drive_col() {
  awk -v b="$1" -v a="$2" -v d="$3" 'BEGIN { printf " | drive_read_bytes %.0f (%s)", (a - b) * 512, d }'
}
