# shellcheck shell=bash
# shellcheck disable=SC2034 # the sourcing runner reads every EVICT_* name
# The page-cache eviction of a model file and the count of what stays resident (depth-qwen3moe.sh's capped arms,
# load-ab.sh's cold starts). Sourced; it defines functions and runs nothing.
#
# evict_set <file> drops every shard of <file>'s split set from the page cache and reads each shard's resident pages
# before and after, by mincore over a mapping that faults nothing. It advises those files alone: no other file, no
# drop_caches, no system setting, and it never writes one: each shard is opened read-only. A page another process holds
# mapped is not evicted, which is why the count after is the check and not the advice. Linux only (posix_fadvise,
# mincore): it runs on the box, never on the Mac.

# model_shards <file>: every shard of <file>'s split set, one a line (`<stem>-0000k-of-0000n.gguf`, k = 1..n,
# whichever shard <file> names), or <file> alone when its name is no shard's.
model_shards() {
  local f=$1 n k
  if [[ ${f##*/} =~ -([0-9]{5})-of-([0-9]{5})\.gguf$ ]]; then
    n=${BASH_REMATCH[2]} f=${f%-"${BASH_REMATCH[1]}"-of-"${BASH_REMATCH[2]}".gguf}
    for ((k = 1; k <= 10#$n; k++)); do printf '%s-%05d-of-%s.gguf\n' "$f" "$k" "$n"; done
  else
    echo "$f"
  fi
}
# The eviction over the shards in argv: each one's resident pages (mincore over a mapping of the file that faults
# nothing), posix_fadvise(DONTNEED) over the whole file, its resident pages again; prints
# `<shards> <pages> <resident before> <resident after>`.
EVICT_PY='
import ctypes, mmap, os, sys
libc = ctypes.CDLL(None, use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
page = os.sysconf("SC_PAGE_SIZE")
low = bytes(b & 1 for b in range(256))
def resident(fd, size):
    if size == 0:
        return 0
    addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
    if addr is None or addr == ctypes.c_void_p(-1).value:
        raise OSError(ctypes.get_errno(), "mmap")
    try:
        n = (size + page - 1) // page
        vec = ctypes.create_string_buffer(n)
        if libc.mincore(addr, size, vec) != 0:
            raise OSError(ctypes.get_errno(), "mincore")
        return n - vec.raw.translate(low).count(0)
    finally:
        libc.munmap(addr, size)
pages = before = after = 0
for path in sys.argv[1:]:
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        pages += (size + page - 1) // page
        before += resident(fd, size)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        after += resident(fd, size)
    finally:
        os.close(fd)
print(len(sys.argv) - 1, pages, before, after)
'
# evict_set <file>: EVICT_PY over every shard of <file>'s set. Returns 0 with EVICT_SHARDS, EVICT_PAGES, EVICT_BEFORE and
# EVICT_AFTER set (the pages resident before and after the advice); returns 1 when the advice did not run, with
# EVICT_SHARDS and EVICT_ERR (the python's last line) set. Whether a page left resident is a failure is the caller's.
EVICT_SHARDS='' EVICT_PAGES='' EVICT_BEFORE='' EVICT_AFTER='' EVICT_ERR=''
evict_set() {
  local f=$1 s out
  local -a shards=()
  while IFS= read -r s; do shards+=("$s"); done < <(model_shards "$f")
  EVICT_SHARDS=${#shards[@]} EVICT_PAGES='' EVICT_BEFORE='' EVICT_AFTER='' EVICT_ERR=''
  if ! out=$(python3 -c "$EVICT_PY" "${shards[@]}" 2>&1); then
    EVICT_ERR=$(tail -n 1 <<< "$out")
    return 1
  fi
  read -r EVICT_SHARDS EVICT_PAGES EVICT_BEFORE EVICT_AFTER <<< "$out"
}
