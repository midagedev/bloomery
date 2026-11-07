#!/usr/bin/env bash
# Where the engine's threads spend a decode step: during the DECODE phase of one k31-nvme-probe.sh request, 30 s of
# `perf trace -s` (per-thread syscall time: futex = waiting on the pool or a fill, ioctl/poll on the nvidia fds = waiting on
# a card) and two `ps -L` snapshots (per-thread CPU share), then a hardware-cycles perf of the engine thread alone.
# Usage: K31_PROBE_SERVE=... K31_PROBE_ENV=... K31_PROBE_N=... k31-engine-sample.sh OUTDIR  (inside a lease-hold)
set -u
O=$1; export K31_PROBE_OUT=$O
bash tools/ref/k31-nvme-probe.sh & P=$!
until grep -q "generating" "$O/serve.log" 2>/dev/null; do sleep 2; kill -0 $P 2>/dev/null || exit 1; done
sleep 10
SP=$(pgrep -x bloomery-serve | head -1)
echo "serve-pid=$SP threads=$(ls /proc/$SP/task | wc -l)"
ps -L -o tid,comm,pcpu,stat --pid "$SP" | sort -k3 -rn | head -12 > "$O/ps1.txt"
perf trace -s -p "$SP" -o "$O/trace.txt" -- sleep 30 >/dev/null 2>&1 || echo "perf trace rc=$?"
ps -L -o tid,comm,pcpu,stat --pid "$SP" | sort -k3 -rn | head -12 > "$O/ps2.txt"
ET=$(ps -L -o tid,comm --pid "$SP" | awk '$2=="engine"{print $1; exit}')
echo "engine-tid=$ET"
if [ -n "$ET" ]; then
  perf record -e cycles -F 1999 -g -t "$ET" -o "$O/engine.perf" -- sleep 15 >/dev/null 2>&1
  perf report -i "$O/engine.perf" --no-children --stdio -g none --sort sym --percent-limit 1 2>/dev/null | grep -v "^#" | grep -v "^\s*$" | head -30 > "$O/engine-sym.txt"
  timeout --kill-after=5 40 bpftrace -e "
tracepoint:sched:sched_switch /args->prev_pid == $ET && args->prev_state != 0/ { @t = nsecs; @k[kstack(8)] = count(); }
tracepoint:sched:sched_switch /args->next_pid == $ET && @t/ { \$d = nsecs - @t; @off_us = hist(\$d / 1000); @off_total_ns = sum(\$d); @switches = count(); @t = 0; }
interval:s:30 { exit(); }" > "$O/engine-off.txt" 2>&1
fi
wait $P; echo "probe-rc=$?"
grep -h "tg_tok_s" "$O/summary.txt"
echo "--- ps (before / after trace)"; cat "$O/ps1.txt"; echo; cat "$O/ps2.txt"
echo "--- perf trace summary (top of each thread's table)"; grep -v "^\s*$" "$O/trace.txt" | head -150
echo "--- engine thread cycles"; cat "$O/engine-sym.txt"
echo "--- engine thread off-cpu"; grep -v "^\s*$" "$O/engine-off.txt" | head -60
