#!/usr/bin/env bash
# Where a serving process's threads wait, sampled: every 20 ms for SECONDS, each thread's
# /proc/<pid>/task/<tid>/wchan (and stat state), counted; beside it a bpftrace histogram of
# posix_fadvise64's duration and its total time over the same window. Diagnostic only, no
# lease of its own: run it inside a probe's window. Usage: k31-wchan-sample.sh PID SECONDS OUTDIR
set -u
HERE=$(cd "${BASH_SOURCE[0]%/*}" && pwd)
# shellcheck source=tools/gate-bound.sh
source "$HERE/../gate-bound.sh"
PID=$1; SEC=$2; OUT=$3
mkdir -p "$OUT"
if command -v bpftrace >/dev/null; then
  BG_OUT="$OUT/fadvise-bpf.txt" bounded_bg $((SEC + 5)) bpftrace -e "
tracepoint:syscalls:sys_enter_fadvise64 /pid == $PID/ { @s[tid] = nsecs; }
tracepoint:syscalls:sys_exit_fadvise64 /@s[tid]/ { \$d = nsecs - @s[tid]; @us = hist(\$d / 1000); @total_ns = sum(\$d); @calls = count(); delete(@s[tid]); }
kprobe:try_to_free_pages /pid == $PID/ { @r[tid] = nsecs; }
kretprobe:try_to_free_pages /@r[tid]/ { \$d = nsecs - @r[tid]; @reclaim_us = hist(\$d / 1000); @reclaim_total_ns = sum(\$d); @reclaim_calls = count(); delete(@r[tid]); }
tracepoint:sched:sched_switch /pid == $PID && args->prev_state != 0/ { @off[kstack(12)] = count(); }
interval:s:$SEC { exit(); }
END { clear(@s); clear(@r); }"
  BPF=1
else
  echo "no bpftrace" > "$OUT/fadvise-bpf.txt"; BPF=
fi
end=$((SECONDS + SEC)); n=0
: > "$OUT/wchan-samples.txt"
while [ $SECONDS -lt $end ]; do
  for t in /proc/$PID/task/*; do
    st=$(cut -d' ' -f3 "$t/stat" 2>/dev/null); w=$(cat "$t/wchan" 2>/dev/null)
    echo "$st ${w:-running}"
  done >> "$OUT/wchan-samples.txt"
  n=$((n + 1)); sleep 0.02
done
[ -n "$BPF" ] && bounded_bg_wait
echo "samples=$n threads=$(ls /proc/$PID/task 2>/dev/null | wc -l)"
echo "--- state+wchan counts (top 25)"; sort "$OUT/wchan-samples.txt" | uniq -c | sort -rn | head -25
echo "--- cgroup"; cat /proc/$PID/cgroup; CG=$(cut -d: -f3 /proc/$PID/cgroup | head -1); for f in memory.max memory.high memory.current; do echo "$f=$(cat /sys/fs/cgroup$CG/$f 2>/dev/null)"; done
echo "--- fadvise64 and off-cpu kernel stacks (bpftrace)"; cat "$OUT/fadvise-bpf.txt"
