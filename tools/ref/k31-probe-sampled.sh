#!/usr/bin/env bash
# One k31-nvme-probe.sh run with tools/ref/k31-wchan-sample.sh over 30 s of its step, 25 s after the
# request starts. Usage: K31_PROBE_SERVE=... k31-probe-sampled.sh OUTDIR  (inside a lease-hold)
set -u
O=$1; export K31_PROBE_OUT=$O
bash tools/ref/k31-nvme-probe.sh & P=$!
until grep -q request_start "$O/summary.txt" 2>/dev/null; do sleep 2; kill -0 $P 2>/dev/null || exit 1; done
sleep 25
SP=$(pgrep -x bloomery-serve | head -1)
echo "serve-pid=$SP threads=$(ls /proc/$SP/task | wc -l)"
bash tools/ref/k31-wchan-sample.sh "$SP" 30 "$O/sample" > "$O/sample.txt" 2>&1
wait $P; echo "probe-rc=$?"
grep -h "pgmajfault_delta\|proc_io\|^decode\|tg_tok_s" "$O/summary.txt"
cat "$O/sample.txt"
