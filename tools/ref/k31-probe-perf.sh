#!/usr/bin/env bash
# One k31-nvme-probe.sh run with a hardware-cycles perf profile (NMI-driven, so IRQ-off spinning is
# seen) of the server over 15 s of its step. Usage: K31_PROBE_SERVE=... k31-probe-perf.sh OUTDIR
set -u
O=$1; export K31_PROBE_OUT=$O
bash tools/ref/k31-nvme-probe.sh & P=$!
until grep -q request_start "$O/summary.txt" 2>/dev/null; do sleep 2; kill -0 $P 2>/dev/null || exit 1; done
sleep 25
SP=$(pgrep -x bloomery-serve | head -1)
perf record -e cycles -F 999 -g -p "$SP" -o "$O/perf.data" -- sleep 15 >/dev/null 2>&1 || echo "perf record rc=$?"
perf report -i "$O/perf.data" --no-children --stdio -g none --sort sym --percent-limit 0.6 2>/dev/null | grep -v "^#" | grep -v "^\s*$" | head -40 > "$O/perf-sym.txt"
perf report -i "$O/perf.data" --no-children --stdio -g none --sort dso 2>/dev/null | grep -v "^#" | grep -v "^\s*$" | head -4 > "$O/perf-dso.txt"
perf report -i "$O/perf.data" --no-children --stdio -G --sort sym --percent-limit 2 2>/dev/null | grep -v "^#" | grep -v "^\s*$" | head -120 > "$O/perf-callers.txt"
wait $P; echo "probe-rc=$?"
grep -h "pgmajfault_delta\|proc_io\|^decode\|tg_tok_s" "$O/summary.txt"
echo "--- dso"; cat "$O/perf-dso.txt"; echo "--- sym"; cat "$O/perf-sym.txt"; echo "--- callers"; cat "$O/perf-callers.txt"
