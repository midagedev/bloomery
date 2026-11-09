#!/usr/bin/env bash
# The two-drive mirror's A/B on a paged serving plan: the same bloomery-serve, the same request, with the set's
# `<prefix>.shard-dirs` sidecar absent (arm `one`: the shards read where they are) and present (arm `two`: the NVMe
# tier's reader picks each expert's parts' copy by id), ROUNDS rounds, the arm order rotated every round, a fresh
# server per row, one cold request each. Runs on the box under tools/ref/lease-hold.sh --card (the card carries the
# prediction; this script takes no lease of its own). The sidecar is written and removed by this script, so the set's
# own sidecar, if any, is moved aside first and restored at the end. Writes under OUT: per row `<round>-<arm>/`
# (serve.log, iostat.log, response.json, summary.txt) and `rows.tsv` (round, arm, tg, pp, load_s, majflt, nvme0 and
# nvme1 mean rMB/s over the request).
#   MAB_SERVE    the bloomery-serve binary (required)
#   MAB_MODEL    the set's first shard (required); MAB_MIRROR the directory holding the byte-identical copy (required)
#   MAB_IDS      the prompt ids file (required); MAB_D / MAB_N prompt ids and n_predict (512 / 128)
#   MAB_ARGS     the server's flags beyond -m/--host/--port (e.g. "--place bp --ctx-size 4096")
#   MAB_ENV      NAME=value words for the server (e.g. "BLOOMERY_HOST_ROOM=128G BLOOMERY_R8=off"), split on spaces
#   MAB_ROUNDS   rounds (2); MAB_PORT (18183); MAB_OUT (/root/mirror-ab)
set -u
HERE=$(cd "${BASH_SOURCE[0]%/*}" && pwd)
# shellcheck source=tools/gate-bound.sh
source "$HERE/../gate-bound.sh"
die() { echo "mirror-ab: $1" >&2; exit "${2:-1}"; }
SERVE=${MAB_SERVE:?the bloomery-serve binary}; MODEL=${MAB_MODEL:?the first shard}; MIRROR=${MAB_MIRROR:?the mirror dir}
IDS=${MAB_IDS:?the ids file}; D=${MAB_D:-512}; N=${MAB_N:-128}; ROUNDS=${MAB_ROUNDS:-2}; PORT=${MAB_PORT:-18183}
OUT=${MAB_OUT:-/root/mirror-ab}; ARGS=${MAB_ARGS:-}; ENVW=${MAB_ENV:-}
[ -x "$SERVE" ] || die "no binary at $SERVE" 66
[ -f "$MODEL" ] || die "no model at $MODEL" 66
[ -d "$MIRROR" ] || die "no mirror dir at $MIRROR" 66
[ -f "$IDS" ] || die "no ids at $IDS" 66
command -v iostat > /dev/null || die "no iostat" 69
base=$(basename "$MODEL"); prefix=${base%-00001-of-*.gguf}
[ "$prefix" != "$base" ] || die "$MODEL is not a first shard" 64
SIDECAR=$(dirname "$MODEL")/$prefix.shard-dirs
for s in "$(dirname "$MODEL")"/"$prefix"-*-of-*.gguf; do
  c=$MIRROR/$(basename "$s"); [ -f "$c" ] || die "no copy of $(basename "$s") in $MIRROR" 66
  [ "$(stat -c %s "$s")" = "$(stat -c %s "$c")" ] || die "$(basename "$s"): the copy's length differs" 65
done
mkdir -p "$OUT"
restore() { rm -f "$SIDECAR"; [ -f "$SIDECAR.mab-saved" ] && mv "$SIDECAR.mab-saved" "$SIDECAR"; }
[ -f "$SIDECAR" ] && mv "$SIDECAR" "$SIDECAR.mab-saved"
trap restore EXIT
{ sha256sum "$SERVE"; echo "model=$MODEL mirror=$MIRROR D=$D N=$N rounds=$ROUNDS args=[$ARGS] env=[$ENVW]"
  for d in nvme0n1 nvme1n1; do echo "$d read_ahead_kb=$(cat /sys/block/$d/queue/read_ahead_kb) scheduler=$(cat /sys/block/$d/queue/scheduler)"; done; } | tee "$OUT/summary.txt"
python3 - "$IDS" "$D" "$N" > "$OUT/request.json" <<'PY'
import json, sys
ids = [int(l) for l in open(sys.argv[1]).read().split()[: int(sys.argv[2])]]
assert len(ids) == int(sys.argv[2]), len(ids)
print(json.dumps({"prompt": ids, "n_predict": int(sys.argv[3]), "temperature": 0, "top_k": 1, "ignore_eos": True, "cache_prompt": False}))
PY
printf 'round\tarm\ttg_tok_s\tpp_tok_s\tload_s\tmajflt\tnvme0_rMB_s\tnvme1_rMB_s\n' > "$OUT/rows.tsv"

row() {
  local round=$1 arm=$2 o=$OUT/$1-$2 spid pid t0 i rc
  rm -rf "$o"; mkdir -p "$o"
  if [ "$arm" = two ]; then printf '# mirror-ab: the copy of this set, one directory a line\n%s\n' "$MIRROR" > "$SIDECAR"; else rm -f "$SIDECAR"; fi
  for i in $(seq 1 60); do ss -ltn 2> /dev/null | grep -q ":$PORT " || break; sleep 1; done
  ss -ltn 2> /dev/null | grep -q ":$PORT " && die "port $PORT still in use after 60 s" 75
  # shellcheck disable=SC2086 # ENVW and ARGS are word lists by contract
  BG_OUT=$o/serve.log bounded_bg 3600 env -u BLOOMERY_REF_MODEL $ENVW "$SERVE" -m "$MODEL" --host 127.0.0.1 --port "$PORT" $ARGS
  spid=$(bounded_bg_get PID); t0=$(date +%s)
  while :; do
    kill -0 "$spid" 2> /dev/null || die "bloomery-serve exited before /health (tail: $(tail -n 3 "$o/serve.log"))"
    curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2> /dev/null | grep -q 200 && break
    [ $(( $(date +%s) - t0 )) -lt 1500 ] || { bounded_bg_stop; die "no /health within 1500 s"; }
    sleep 2
  done
  echo "load_s=$(( $(date +%s) - t0 ))" | tee -a "$o/summary.txt"
  grep -m1 '^plan ' "$o/serve.log" | tee -a "$o/summary.txt"
  grep -m1 '^nvtier ' "$o/serve.log" | cut -c1-300 | tee -a "$o/summary.txt"
  pid=$(pgrep -x bloomery-serve | head -1); pid=${pid:-$spid}
  grep pgmajfault /proc/vmstat > "$o/vm-before"
  iostat -x -t 1 nvme0n1 nvme1n1 > "$o/iostat.log" 2>&1 &
  local ipid=$!
  sleep 2
  echo "request_start=$(date -Iseconds)" | tee -a "$o/summary.txt"
  curl -s -m 3000 -H 'Content-Type: application/json' --data-binary @"$OUT/request.json" "http://127.0.0.1:$PORT/completion" > "$o/response.json"
  rc=$?
  sleep 2; kill "$ipid" 2> /dev/null
  grep pgmajfault /proc/vmstat > "$o/vm-after"
  bounded_bg_stop
  for i in $(seq 1 60); do ss -ltn 2> /dev/null | grep -q ":$PORT " || break; sleep 1; done
  [ "$rc" = 0 ] || die "curl rc $rc"
  python3 - "$o" "$round" "$arm" >> "$OUT/rows.tsv" <<'PY'
import json, sys
o, rnd, arm = sys.argv[1:4]
r = json.load(open(f"{o}/response.json")); t = r.get("timings") or sys.exit("no timings: " + json.dumps(r)[:200])
load = [l for l in open(f"{o}/summary.txt") if l.startswith("load_s=")][-1].split("=")[1].strip()
vm = lambda f: int(open(f).read().split()[1])
maj = vm(f"{o}/vm-after") - vm(f"{o}/vm-before")
# iostat -x rows: Device r/s rkB/s ... ; mean rkB/s per device over the request window
acc = {}
for line in open(f"{o}/iostat.log"):
    p = line.split()
    if len(p) > 2 and p[0] in ("nvme0n1", "nvme1n1") and p[1] != "r/s":
        try: acc.setdefault(p[0], []).append(float(p[2]))
        except ValueError: pass
mb = lambda d: (sum(acc.get(d, [0])) / max(1, len(acc.get(d, [0])))) / 1024
print(f"{rnd}\t{arm}\t{t['predicted_per_second']:.2f}\t{t['prompt_per_second']:.2f}\t{load}\t{maj}\t{mb('nvme0n1'):.0f}\t{mb('nvme1n1'):.0f}")
PY
  tail -n 1 "$OUT/rows.tsv"
}

for r in $(seq 1 "$ROUNDS"); do
  if [ $((r % 2)) = 1 ]; then order="one two"; else order="two one"; fi
  for a in $order; do row "$r" "$a"; done
done
echo "=== rows"; cat "$OUT/rows.tsv"
python3 - "$OUT/rows.tsv" <<'PY' | tee -a "$OUT/summary.txt"
import sys, statistics as st
rows = [l.split("\t") for l in open(sys.argv[1]).read().strip().split("\n")[1:]]
by = {}
for r in rows: by.setdefault(r[1], []).append((float(r[2]), float(r[3])))
for k, v in by.items():
    print(f"mean {k}: tg {st.mean(x[0] for x in v):.2f} pp {st.mean(x[1] for x in v):.2f} over {len(v)} row(s)")
if "one" in by and "two" in by:
    a, b = st.mean(x[0] for x in by["one"]), st.mean(x[0] for x in by["two"])
    print(f"effect tg (two - one) / one = {100 * (b - a) / a:+.1f} %")
    a, b = st.mean(x[1] for x in by["one"]), st.mean(x[1] for x in by["two"])
    print(f"effect pp (two - one) / one = {100 * (b - a) / a:+.1f} %")
PY
