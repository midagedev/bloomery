#!/usr/bin/env bash
# k31-probe-sampled.sh plus a block-layer trace (request sizes and service times, both drives) over the same 30 s window.
# Usage: K31_PROBE_SERVE=... K31_PROBE_ENV=... K31_PROBE_N=... k31-probe-blk.sh OUTDIR
set -u
HERE=$(cd "${BASH_SOURCE[0]%/*}" && pwd)
# shellcheck source=tools/gate-bound.sh
source "$HERE/../gate-bound.sh"
O=$1; export K31_PROBE_OUT=$O
bash tools/ref/k31-nvme-probe.sh & P=$!
until grep -q request_start "$O/summary.txt" 2>/dev/null; do sleep 2; kill -0 $P 2>/dev/null || exit 1; done
sleep 25
SP=$(pgrep -x bloomery-serve | head -1)
echo "serve-pid=$SP threads=$(ls /proc/$SP/task | wc -l)"
bounded 40 bpftrace -e '
tracepoint:block:block_rq_issue { @t[args->dev, args->sector] = nsecs; @issue = count(); @bytes = sum(args->bytes); @kb = hist(args->bytes / 1024); }
tracepoint:block:block_rq_complete /@t[args->dev, args->sector]/ { $d = nsecs - @t[args->dev, args->sector]; @svc_us = hist($d / 1000); @svc_total_ns = sum($d); delete(@t[args->dev, args->sector]); }
interval:s:30 { exit(); }
END { clear(@t); }' > "$O/blk.txt" 2>&1 &
B=$!
bash tools/ref/k31-wchan-sample.sh "$SP" 30 "$O/sample" > "$O/sample.txt" 2>&1
wait $B
wait $P; echo "probe-rc=$?"
grep -h "pgmajfault_delta\|proc_io\|^decode\|tg_tok_s" "$O/summary.txt"
echo "--- blk"; grep -v "^\s*$" "$O/blk.txt" | head -40
echo "--- sample"; sed -n "/--- state/,/--- cgroup/p" "$O/sample.txt" | head -14
