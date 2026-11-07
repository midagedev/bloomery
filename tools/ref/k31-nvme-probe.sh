#!/usr/bin/env bash
# One K3.1 request at --place bp with iostat beside it: which regime the NVMe is in while the host leg pages experts
# through the file mapping (docs/cards/k31-nvme-regime.card). Runs on the box under tools/ref/lease-hold.sh --card
# (both cards idle). Writes everything under $OUT (default /root/k31-nvme-probe): serve.log, iostat.log, io-before/after,
# response.json, summary.txt. Exits 0 when the response carries timings, else by name.
#
#   K31_PROBE_TARBALL  the release tarball whose bloomery-serve runs (default the k31rel tree's sm86 release-dist member)
#   K31_PROBE_SERVE    a built bloomery-serve to run instead of the tarball's (a tree's target/release binary)
#   K31_PROBE_MODEL    the first shard (default /models/Kimi-K3.1-Pro-IQ1_S/Kimi-K3.1-Pro-IQ1_S-00001-of-00031.gguf)
#   K31_PROBE_D / K31_PROBE_N   prompt ids and n_predict (default 512 / 128)
#   K31_PROBE_ENV      extra NAME=value words for the server (e.g. BLOOMERY_NVTIER_BYTES=32G), split on spaces
set -u
OUT=${K31_PROBE_OUT:-/root/k31-nvme-probe}
TARBALL=${K31_PROBE_TARBALL:-$HOME/repo/bloomery-k31rel/target/release-dist/bloomery-0.3.13-linux-x86_64-cuda-sm86.tar.gz}
MODEL=${K31_PROBE_MODEL:-/models/Kimi-K3.1-Pro-IQ1_S/Kimi-K3.1-Pro-IQ1_S-00001-of-00031.gguf}
D=${K31_PROBE_D:-512}; N=${K31_PROBE_N:-128}; PORT=${K31_PROBE_PORT:-18182}
IDS=${BLOOMERY_DATA:-/root/bloomery-data}/kimi-k3/corpus-prose.ids
mkdir -p "$OUT"; rm -f "$OUT"/*
die() { echo "k31-nvme-probe: $*" >&2; exit "${2:-1}"; }
[ -n "${K31_PROBE_SERVE:-}" ] || [ -f "$TARBALL" ] || die "no tarball at $TARBALL" 66
[ -f "$MODEL" ] || die "no model at $MODEL" 66
[ -f "$IDS" ] || die "no ids at $IDS" 66
command -v iostat >/dev/null || die "no iostat" 69
if [ -n "${K31_PROBE_SERVE:-}" ]; then
  SERVE=$K31_PROBE_SERVE; [ -x "$SERVE" ] || die "no binary at $SERVE" 66
else
  rm -rf "$OUT/bin"; mkdir -p "$OUT/bin"
  tar -xzf "$TARBALL" -C "$OUT/bin" || die "tar failed"
  SERVE=$(find "$OUT/bin" -name bloomery-serve -type f | head -1); [ -n "$SERVE" ] || die "no bloomery-serve in $TARBALL"
fi
sha256sum "$SERVE" | tee "$OUT/summary.txt"
{ for d in nvme0n1 nvme1n1; do echo "$d read_ahead_kb=$(cat /sys/block/$d/queue/read_ahead_kb) nr_requests=$(cat /sys/block/$d/queue/nr_requests) max_sectors_kb=$(cat /sys/block/$d/queue/max_sectors_kb) scheduler=$(cat /sys/block/$d/queue/scheduler)"; done
  echo "MemAvailable_kB=$(awk '/MemAvailable/{print $2}' /proc/meminfo) Cached_kB=$(awk '/^Cached/{print $2}' /proc/meminfo)"; } | tee -a "$OUT/summary.txt"

# the previous run's server may still be unmapping: its port must be free before this listener
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ":$PORT " || break; sleep 1; done
ss -ltn 2>/dev/null | grep -q ":$PORT " && die "port $PORT still in use after 60 s" 75
env -u BLOOMERY_REF_MODEL BLOOMERY_HOST_ROOM=252G ${K31_PROBE_ENV:-} timeout --kill-after=10 3600 "$SERVE" -m "$MODEL" --host 127.0.0.1 --port "$PORT" --place bp --ctx-size 2048 \
  > "$OUT/serve.log" 2>&1 &
SPID=$!; echo "$SPID" > "$OUT/serve.pid"
t0=$(date +%s)
while :; do
  kill -0 "$SPID" 2>/dev/null || die "bloomery-serve exited before /health (tail: $(tail -n 3 "$OUT/serve.log"))"
  curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q 200 && break
  [ $(( $(date +%s) - t0 )) -lt 1200 ] || { kill "$SPID"; die "no /health within 1200 s"; }
  sleep 2
done
echo "load_s=$(( $(date +%s) - t0 ))" | tee -a "$OUT/summary.txt"
grep -m1 '^plan ' "$OUT/serve.log" | tee -a "$OUT/summary.txt"
# the serve process that holds the mapping may be a child; take the pid whose exe is bloomery-serve with the mapping
PID=$(pgrep -P "$SPID" -x bloomery-serve | head -1); PID=${PID:-$SPID}
cat "/proc/$PID/io" > "$OUT/io-before"; grep pgmajfault /proc/vmstat > "$OUT/vm-before"
python3 - "$IDS" "$D" "$N" > "$OUT/request.json" <<'PY'
import json, sys
ids = [int(l) for l in open(sys.argv[1]).read().split()[: int(sys.argv[2])]]
assert len(ids) == int(sys.argv[2]), len(ids)
print(json.dumps({"prompt": ids, "n_predict": int(sys.argv[3]), "temperature": 0, "top_k": 1, "ignore_eos": True, "cache_prompt": False}))
PY
iostat -x -t 1 nvme0n1 nvme1n1 > "$OUT/iostat.log" 2>&1 &
IPID=$!; echo "$IPID" > "$OUT/iostat.pid"
sleep 2
req0=$(date +%s.%N); echo "request_start=$(date -Iseconds)" | tee -a "$OUT/summary.txt"
curl -s -m 3000 -H 'Content-Type: application/json' --data-binary @"$OUT/request.json" "http://127.0.0.1:$PORT/completion" > "$OUT/response.json"
rc=$?; req1=$(date +%s.%N)
sleep 2; kill "$IPID" 2>/dev/null
cat "/proc/$PID/io" > "$OUT/io-after"; grep pgmajfault /proc/vmstat > "$OUT/vm-after"
kill "$SPID" 2>/dev/null
# the server unmaps 580 GB on exit: wait for the pid, then for the port, so the next run's listener is not refused
for i in $(seq 1 120); do kill -0 "$SPID" 2>/dev/null || break; sleep 1; [ "$i" = 60 ] && kill -9 "$SPID" 2>/dev/null; done
wait "$SPID" 2>/dev/null
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ":$PORT " || break; sleep 1; done
echo "teardown_s=$i" | tee -a "$OUT/summary.txt"
[ "$rc" = 0 ] || die "curl rc $rc"
python3 - "$OUT" "$req0" "$req1" <<'PY' | tee -a "$OUT/summary.txt"
import json, sys, re, datetime
out, r0, r1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
resp = json.load(open(f"{out}/response.json"))
t = resp.get("timings") or sys.exit("no timings in the response: " + json.dumps(resp)[:300])
print("timings", json.dumps(t))
pp_s = t["prompt_ms"] / 1000; tg_s = t["predicted_ms"] / 1000
def io(f): return {l.split(":")[0]: int(l.split(":")[1]) for l in open(f)}
a, b = io(f"{out}/io-before"), io(f"{out}/io-after")
print(f"proc_io read_bytes_delta_GB={(b['read_bytes']-a['read_bytes'])/1e9:.2f} rchar_delta_GB={(b['rchar']-a['rchar'])/1e9:.2f}")
va = int(open(f"{out}/vm-before").read().split()[1]); vb = int(open(f"{out}/vm-after").read().split()[1])
print(f"pgmajfault_delta={vb-va}")
# iostat -t prints a timestamp line, then the device table; bucket the device rows by phase
rows = []; ts = None; hdr = None
for line in open(f"{out}/iostat.log"):
    s = line.strip()
    if not s: continue
    if re.match(r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2} (AM|PM)$", s):
        ts = datetime.datetime.strptime(s, "%m/%d/%Y %I:%M:%S %p").timestamp(); continue
    if re.match(r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}$", s):
        ts = datetime.datetime.strptime(s, "%m/%d/%Y %H:%M:%S").timestamp(); continue
    if s.startswith("Device"): hdr = s.split(); continue
    if hdr and s.startswith("nvme") and ts is not None:
        rows.append((ts, dict(zip(hdr, s.split()))))
def phase(dev, lo, hi):
    sel = [r for (ts, r) in rows if r["Device"] == dev and lo <= ts < hi]
    if not sel: return f"{dev}: no rows"
    f = lambda k: sum(float(r[k]) for r in sel) / len(sel)
    return (f"{dev} n={len(sel)} aqu-sz={f('aqu-sz'):.1f} rareq-sz={f('rareq-sz'):.0f}KiB r/s={f('r/s'):.0f} "
            f"rMB/s={f('rkB/s')/1024:.0f} r_await={f('r_await'):.2f}ms util={f('%util'):.0f}%")
for dev in ("nvme0n1", "nvme1n1"):
    print("prompt ", phase(dev, r0, r0 + pp_s))
    print("decode ", phase(dev, r0 + pp_s, r1))
print(f"pp_tok_s={t.get('prompt_per_second')} tg_tok_s={t.get('predicted_per_second')} prompt_n={t.get('prompt_n')} predicted_n={t.get('predicted_n')} wall_s={r1-r0:.0f}")
PY
