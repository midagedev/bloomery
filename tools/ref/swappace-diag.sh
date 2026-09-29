# swappace: the residency's diagnostic arms — generate_ds41 --place bp, two prose:64 arms under
# BLOOMERY_RESIDENCY=mid-p40-s1, once per BLOOMERY_SWAP_FILL in FILLS (default "1 2 4"; the lever is
# parsed at main and read by the load, so each value is a run of its own: generate_ds41's --arm takes no
# environment), through gpu-gate.sh (both gate locks, its BLOOMERY_GATE_BOUND), each run with a
# watchdog: once the first `arm` record is out, 90 s with no new output dumps generate_ds41's threads
# (gdb, else /proc task stacks) and ends it. The watched pid is found only among the descendants of the
# pid this script spawned, by comm. A witness line before and after each run: loadavg, IO and CPU
# pressure, and every compute process on the cards. At the end, tools/ref/swappace-table.py's table of
# every run that printed records.
#
# Wall: one build (~2 min), then per run one bp load and two arms (~3 min), each bounded by BOUND
# seconds (default 480) and its watch by BOUND + 60: at most 2 + 3 x 9 = 29 min.
#
# Exit status: 0 when every run's gpu-gate.sh returned 0 and no watchdog ended one; 1 otherwise (a build
# failure included).
set -u
OUT=target/swappace-diag
FILLS=${FILLS:-1 2 4}
BOUND=${BOUND:-480}
rm -rf "$OUT"; mkdir -p "$OUT"
export BLOOMERY_GATE_V41_LOAD=1 BLOOMERY_RESIDENCY=mid-p40-s1 BLOOMERY_STEP_STATS=1 BLOOMERY_GATE_BOUND=$BOUND
echo "build $(date +%T)"
cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu,deepseek41 --release --bin generate_ds41 > "$OUT/build.log" 2>&1 || { tail -30 "$OUT/build.log"; exit 1; }
grep 'Compiling bloomery' "$OUT/build.log"
witness() {
  echo "witness $1 $(date +%T) loadavg $(cut -d' ' -f1-3 /proc/loadavg) io $(head -1 /proc/pressure/io) cpu $(head -1 /proc/pressure/cpu)"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed "s/^/witness $1 app /"
}
find_gen() {
  local p=$1 c
  if [ "$(cat /proc/$p/comm 2>/dev/null)" = generate_ds41 ]; then echo "$p"; return; fi
  for c in $(cat /proc/$p/task/*/children 2>/dev/null); do find_gen "$c"; done
}
# One run at BLOOMERY_SWAP_FILL=$1 into $OUT/fill$1.*: its gpu-gate.sh rc, or 1 when the watchdog ended it
# or its watch bound passed.
run_fill() {
  local fill=$1 log="$OUT/fill$1.log" top gen='' last=-1 still=0 t0=$SECONDS masks='' killed=0 rc
  witness "pre fill=$fill"
  echo "run fill=$fill $(date +%T)"
  BLOOMERY_SWAP_FILL=$fill setsid bash tools/gpu-gate.sh generate_ds41 --place bp --arm prose:64 --arm prose:64 -n 96 --time > "$log" 2>&1 < /dev/null &
  top=$!
  echo "$top" > "$OUT/fill$fill.top.pid"
  echo "spawned $top comm $(cat /proc/$top/comm)"
  while kill -0 "$top" 2>/dev/null; do
    sleep 5
    if [ -z "$gen" ]; then
      gen=$(find_gen "$top" | head -n 1)
      [ -n "$gen" ] && { echo "$gen" > "$OUT/fill$fill.gen.pid"; echo "watching $gen comm $(cat /proc/$gen/comm) at $((SECONDS - t0)) s"; }
    fi
    if [ -n "$gen" ] && [ -z "$masks" ] && grep -q '^arm ' "$log"; then
      masks=1
      for t in /proc/"$gen"/task/*; do
        c=$(cat "$t/comm" 2>/dev/null)
        case $c in swap-staging|engram-prefetch|generate_ds41) echo "mask $c tid ${t##*/} $(grep Cpus_allowed_list "$t/status")";; esac
      done > "$OUT/fill$fill.masks.txt"
      echo "== masks at the first arm (fill=$fill)"; cat "$OUT/fill$fill.masks.txt"
    fi
    sz=$(stat -c %s "$log")
    if [ "$sz" = "$last" ]; then still=$((still + 5)); else still=0; last=$sz; fi
    if [ -n "$gen" ] && grep -q '^arm ' "$log" && [ "$still" -ge 90 ]; then
      echo "no output for $still s at $((SECONDS - t0)) s: dumping $gen"
      if command -v gdb > /dev/null; then
        timeout 120 gdb -p "$gen" -batch -ex 'thread apply all bt' > "$OUT/fill$fill.stacks.txt" 2>&1
      else
        for t in /proc/"$gen"/task/*; do echo "== $t $(cat "$t/comm") wchan=$(cat "$t/wchan")"; cat "$t/stack" 2>/dev/null; done > "$OUT/fill$fill.stacks.txt"
      fi
      kill -TERM "$gen"; sleep 10; kill -0 "$gen" 2>/dev/null && kill -KILL "$gen"
      killed=1
      break
    fi
    if [ $((SECONDS - t0)) -ge $((BOUND + 60)) ]; then echo "watch bound $((BOUND + 60)) s"; killed=1; break; fi
  done
  wait "$top"; rc=$?
  witness "post fill=$fill"; echo "gpu-gate rc=$rc fill=$fill $(date +%T)"
  echo "== fill$fill.log"; cat "$log"
  [ -f "$OUT/fill$fill.stacks.txt" ] && { echo "== stacks (fill=$fill)"; cat "$OUT/fill$fill.stacks.txt"; }
  [ "$killed" = 1 ] && [ "$rc" = 0 ] && rc=1
  return "$rc"
}
status=0 logs=()
for f in $FILLS; do
  run_fill "$f" || { echo "run fill=$f failed"; status=1; }
  logs+=("fill$f=$OUT/fill$f.log")
done
echo "== table"
python3 tools/ref/swappace-table.py "${logs[@]}" || status=1
echo "swappace-diag status=$status"
exit "$status"
